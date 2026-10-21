"""Phase 4.2: transcribe videos with Parakeet on this machine's GPU, politely.

The model runs in re2/asr_worker.py under its own Python env ([asr] python, with nemo_toolkit[asr]); this
module picks the videos, extracts their audio, and stores what the worker hears. The GPUs are shared with
the llama.cpp router, so a run starts only on a card with [asr] need_gb free and no model loading there, and
stops after the current video once the card drops below floor_gb free or the router starts loading a model.
Re-running resumes. Videos AmberNet is sure are in a language the model does not transcribe are
recorded and skipped (judge_language).
"""
from __future__ import annotations

import json
import logging
import os
import select
import socket
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from re2 import db, runner

log = logging.getLogger(__name__)

STAGE = "asr"
WINDOW_S = 60.0
WINDOW_CHARS = 600
FFMPEG_TIMEOUT = 1800
MIN_PARTIAL_BYTES = 44 + 32000   # a WAV header and 1 s of 16 kHz 16-bit mono
WORKER_TIMEOUT = 600       # seconds for one answer; a 0.8 h video takes well under a minute, and the
                           # first load (which may download the model) about 25 s
NO_SPEECH_ATTEMPTS = 99  # recorded as a "failure" so the video is never selected again
# The 25 languages Parakeet TDT 0.6B v3 transcribes (model card).
LANGUAGES = frozenset("bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk".split())
# A video is skipped as another language only when AmberNet is this sure of a language not in LANGUAGES.
# Measured on 105 archive videos (2026-10-21): it skipped none of 40 English ones (their wrong guesses
# never passed 0.42) and caught Hebrew, most Chinese and clear Japanese at 0.92-1.0. Below that, Parakeet
# transcribes; on foreign speech it mostly returns nothing.
LANGUAGE_MIN_P = 0.9
# AmberNet calls British English Welsh, often at 0.96+ (checked by transcribing them; Whisper made the same
# mistake on the same videos), so Welsh is never a reason to skip, nor recorded as the language.
NOT_TRUSTED = frozenset({"cy"})
AMBERNET_CODES = {"iw": "he"}   # to the ISO 639-1 codes Whisper recorded
WORKER = Path(__file__).with_name("asr_worker.py")
REDO_KEY = "asr_redo"      # kv: JSON list of media ids to transcribe again (mark_redo)


class Busy(RuntimeError):
    """The GPU has no room for the model now; nothing was started."""


def model_tag(cfg) -> str:
    return cfg.asr_model.rsplit("/", 1)[-1]


def gpu_free_gb() -> dict[int, float]:
    """Free memory per card, from nvidia-smi (the router's llama.cpp children do not show up in torch)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30, check=True).stdout
    return {int(i): int(m) / 1024 for i, m in (line.split(",") for line in out.strip().splitlines())}


def _local_routers(cfg) -> list[str]:
    """The llama.cpp servers in the config that run on this machine, whose model loads need its VRAM."""
    me = socket.gethostname().split(".")[0]
    out = []
    for url in {cfg.base_url, cfg.chat_base_url, cfg.wiki_base_url}:
        u = urlparse(url)
        host = u.hostname or ""
        if host in ("127.0.0.1", "localhost") or host.split(".")[0] == me:
            out.append(f"{u.scheme}://{u.netloc}")
    return sorted(set(out))


def router_loading(cfg) -> list[str]:
    """Models a local router is loading right now. A router that does not answer counts as not loading."""
    loading = []
    for base in _local_routers(cfg):
        try:
            data = httpx.get(f"{base}/models", timeout=5).json().get("data", [])
        except (httpx.HTTPError, ValueError):
            continue
        loading += [m["id"] for m in data if (m.get("status") or {}).get("value") == "loading"]
    return loading


def pick_card(cfg) -> int | None:
    """The card to load the model on (None for the CPU). Raises Busy when none has room right now."""
    if cfg.asr_device == "cpu":
        return None
    loading = router_loading(cfg)
    if loading:
        raise Busy(f"the router is loading {', '.join(loading)}; try again once it has loaded")
    free = gpu_free_gb()
    if cfg.asr_device.startswith("cuda:"):
        free = {n: f for n, f in free.items() if n == int(cfg.asr_device.split(":", 1)[1])}
    if not free:
        raise Busy(f"no GPU {cfg.asr_device} on this machine")
    card = max(free, key=free.get)
    if free[card] < cfg.asr_need_gb:
        raise Busy(f"GPU {card} has {free[card]:.1f} GB free, and speech to text needs {cfg.asr_need_gb:.1f} GB "
                   f"([asr] need_gb); the router's model is using the rest")
    return card


def why_stop(cfg, card: int | None) -> str | None:
    """A reason to stop before the next video, so the router can load or grow; None to carry on."""
    if card is None:
        return None
    loading = router_loading(cfg)
    if loading:
        return f"the router started loading {', '.join(loading)}"
    free = gpu_free_gb().get(card, 0.0)
    if free < cfg.asr_floor_gb:
        return f"GPU {card} is down to {free:.1f} GB free ([asr] floor_gb {cfg.asr_floor_gb:.1f})"
    return None


class Worker:
    """re2/asr_worker.py in the [asr] python env, answering one JSON line per audio file."""

    def __init__(self, cfg, card: int | None):
        python = Path(cfg.asr_python).expanduser()
        if not python.exists():
            raise FileNotFoundError(f"[asr] python {python} does not exist; see scripts/setup_asr_env.sh")
        tmp = Path(cfg.data_dir) / "tmp"          # NeMo unpacks the model here; /tmp may be a small disk
        tmp.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "TMPDIR": str(tmp), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "PYTHONUNBUFFERED": "1"}
        if card is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(card)
        log_dir = Path(cfg.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "asr-worker.log"
        self.err = open(self.log_path, "ab")
        # Its own session: Ctrl-C reaches re2, which finishes the current video and then closes the worker.
        self.proc = subprocess.Popen(
            [str(python), str(WORKER), "--model", cfg.asr_model, "--device", "cpu" if card is None else "cuda",
             "--chunk-s", str(cfg.asr_chunk_s)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, env=env,
            start_new_session=True)
        ready = self._read()
        log.info("speech model %s ready on %s in %.1f s", cfg.asr_model,
                 "CPU" if card is None else f"GPU {card}", ready.get("load_s", 0))
        self.load_s = ready.get("load_s")

    def _read(self) -> dict:
        if not select.select([self.proc.stdout], [], [], WORKER_TIMEOUT)[0]:
            self.proc.kill()
            raise RuntimeError(f"speech worker gave no answer in {WORKER_TIMEOUT} s; see {self.log_path}")
        line = self.proc.stdout.readline()
        if not line:
            self.proc.wait(timeout=30)
            tail = self.log_path.read_text(errors="replace").strip().splitlines()[-1:] or [""]
            raise RuntimeError(f"speech worker exited ({self.proc.returncode}): {tail[0][:300]}")
        return json.loads(line)

    def transcribe(self, wav: Path) -> tuple[str | None, float, list[Segment]]:
        """(language AmberNet hears, its probability, segments)."""
        self.proc.stdin.write(json.dumps({"wav": str(wav)}) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["language"], reply["language_p"], [Segment(a, b, t) for a, b, t in reply["segments"]]

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        self.err.close()


@dataclass
class Segment:
    start: float
    end: float
    text: str


def extract_audio(path: Path, wav: Path) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       capture_output=True, check=True, timeout=FFMPEG_TIMEOUT)
    except subprocess.CalledProcessError as exc:
        # A damaged audio track stops ffmpeg at the first bad packet, after it has written what came before
        # (e.g. 55 s of an 80 s clip); that much is worth transcribing.
        if wav.exists() and wav.stat().st_size > MIN_PARTIAL_BYTES:
            log.warning("%s: audio damaged, transcribing the %.0f s before the damage", path.name,
                        (wav.stat().st_size - 44) / 32000)
            return
        raise RuntimeError(f"ffmpeg audio extraction failed: {exc.stderr.decode(errors='replace')[:300]}") from exc


def judge_language(language: str | None, p: float) -> tuple[str | None, bool]:
    """(the language to record, or None when unsure; whether to skip the video as another language)."""
    language = AMBERNET_CODES.get(language, language)
    if not language or language in NOT_TRUSTED or p < LANGUAGE_MIN_P:
        return None, False
    return language, language not in LANGUAGES


def transcribe(worker, cfg, media_id: int, path: Path) -> tuple[str | None, float, list[Segment]]:
    wav = Path(cfg.data_dir) / "audio" / f"{media_id}.wav"
    try:
        extract_audio(path, wav)
        return worker.transcribe(wav)
    finally:
        wav.unlink(missing_ok=True)


def windows(segments: list[Segment]) -> list[Segment]:
    """Consecutive segments grouped into windows of at most WINDOW_S seconds or WINDOW_CHARS characters."""
    out: list[Segment] = []
    cur: list[Segment] = []
    for seg in segments:
        if cur and (seg.end - cur[0].start > WINDOW_S or sum(len(s.text) + 1 for s in cur) + len(seg.text) > WINDOW_CHARS):
            out.append(Segment(cur[0].start, cur[-1].end, " ".join(s.text for s in cur)))
            cur = []
        cur.append(seg)
    if cur:
        out.append(Segment(cur[0].start, cur[-1].end, " ".join(s.text for s in cur)))
    return out


def _set_language(conn: sqlite3.Connection, media_id: int, language: str | None, replace: bool) -> None:
    """Record the language; a redo replaces what the older engine recorded, even with "unknown"."""
    if language is None and not replace:
        return
    keep = "excluded.language" if replace else "COALESCE(media_meta.language, excluded.language)"
    conn.execute(f"""INSERT INTO media_meta(media_id, language) VALUES (?, ?)
                     ON CONFLICT(media_id) DO UPDATE SET language={keep}""", (media_id, language))


def redo_ids(conn: sqlite3.Connection) -> set[int]:
    return set(json.loads(db.kv_get(conn, REDO_KEY) or "[]"))


def mark_redo(conn: sqlite3.Connection, media_ids) -> int:
    """Queue videos to be transcribed again on the next run, even though they have a transcript or a
    no-speech verdict; their old transcript stays searchable until the new one replaces it."""
    ids = {int(i) for i in media_ids}
    with conn:
        db.kv_set(conn, REDO_KEY, json.dumps(sorted(redo_ids(conn) | ids)))
        conn.executemany("DELETE FROM failure WHERE stage=? AND media_id=?", [(STAGE, i) for i in ids])
    return len(ids)


def redo_candidates(conn: sqlite3.Connection, what: str) -> list[int]:
    """Videos to redo: "no_speech" (an earlier no-speech verdict) or a language code an earlier engine
    recorded (e.g. "cy": Whisper heard British English as Welsh and stored a Welsh translation)."""
    if what == "no_speech":
        sql, args = "SELECT media_id FROM failure WHERE stage=? AND error='no_speech'", (STAGE,)
    else:
        sql, args = ("""SELECT mm.media_id FROM media_meta mm JOIN media m ON m.id=mm.media_id
                        WHERE m.kind='video' AND mm.language=?""", (what,))
    return [r[0] for r in conn.execute(sql, args)]


def store(conn: sqlite3.Connection, cfg, media_id: int, segments: list[Segment] | None, language: str | None) -> int:
    """Windows stored, 0 for no speech, -1 for a language the model does not transcribe."""
    redo = redo_ids(conn)
    with conn:
        conn.execute("DELETE FROM transcript_segment WHERE media_id=?", (media_id,))
        conn.execute("DELETE FROM doc WHERE media_id=? AND kind='transcript'", (media_id,))
        _set_language(conn, media_id, language, replace=media_id in redo)
        if media_id in redo:
            db.kv_set(conn, REDO_KEY, json.dumps(sorted(redo - {media_id})))
        if segments is None:
            db.upsert_failure(conn, STAGE, media_id, None, f"language:{language}", attempts=NO_SPEECH_ATTEMPTS)
            return -1
        if not segments:
            db.upsert_failure(conn, STAGE, media_id, None, "no_speech", attempts=NO_SPEECH_ATTEMPTS)
            return 0
        conn.executemany("INSERT INTO transcript_segment(media_id, start_s, end_s, text) VALUES (?, ?, ?, ?)",
                         [(media_id, s.start, s.end, s.text) for s in segments])
        wins = windows(segments)
        conn.executemany(
            "INSERT INTO doc(media_id, kind, seq, start_s, end_s, text, model) VALUES (?, 'transcript', ?, ?, ?, ?, ?)",
            [(media_id, n, w.start, w.end, w.text, model_tag(cfg)) for n, w in enumerate(wins)])
        db.clear_failure(conn, STAGE, media_id)
        return len(wins)


@dataclass
class AsrStats:
    selected: int = 0
    done: int = 0
    no_speech: int = 0
    other_language: int = 0
    failed: int = 0
    audio_s: float = 0.0
    interrupted: bool = False
    stopped: str | None = None   # why it stopped early to make room on the GPU

    def summary(self) -> str:
        return (f"selected={self.selected} done={self.done} no_speech={self.no_speech} "
                f"other_language={self.other_language} failed={self.failed} audio_hours={self.audio_s / 3600:.2f}"
                + (" INTERRUPTED" if self.interrupted else "") + (f" STOPPED: {self.stopped}" if self.stopped else ""))


def select_jobs(conn: sqlite3.Connection, limit: int | None) -> list[sqlite3.Row]:
    """Videos with audio and no transcript yet, and the ones queued for a redo (mark_redo)."""
    skip = db.pending_failures(conn, STAGE)
    redo = redo_ids(conn)
    rows = conn.execute(
        """SELECT m.id, m.duration,
                  (SELECT p.rel_path FROM media_path p WHERE p.media_id=m.id AND p.readable=1
                   ORDER BY p.id LIMIT 1) AS rel_path
           FROM media m
           WHERE m.kind='video' AND m.status='READY' AND m.has_audio=1
           ORDER BY m.id""").fetchall()
    has = {r[0] for r in conn.execute("SELECT DISTINCT media_id FROM doc WHERE kind='transcript'")}
    jobs = [r for r in rows if r["rel_path"] and (r["id"] in redo or (r["id"] not in has and r["id"] not in skip))]
    return jobs[:limit] if limit is not None else jobs


def run(conn: sqlite3.Connection, cfg, limit: int | None = None, worker=None) -> AsrStats:
    """Transcribe pending videos. Raises Busy, before loading anything, when the GPU has no room."""
    jobs = select_jobs(conn, limit)
    stats = AsrStats(selected=len(jobs))
    if not jobs:
        return stats
    card = None
    own = worker is None
    if own:
        card = pick_card(cfg)
        worker = Worker(cfg, card)

    def polite():
        for row in jobs:
            reason = why_stop(cfg, card)
            if reason:
                stats.stopped = reason
                log.warning("stopping to make room on the GPU: %s; re-run to resume", reason)
                return
            yield row

    def work(row):
        return transcribe(worker, cfg, row["id"], Path(cfg.corpus_root) / row["rel_path"])

    def on_result(row, result):
        language, p, segments = result
        language, skip = judge_language(language, p)
        n = store(conn, cfg, row["id"], None if skip else segments, language)
        if n > 0:
            stats.done += 1
        elif n == 0:
            stats.no_speech += 1
        else:
            stats.other_language += 1
        stats.audio_s += row["duration"] or 0

    def on_error(row, exc):
        stats.failed += 1
        log.warning("asr %s (%s) failed: %s", row["id"], row["rel_path"], exc)
        with conn:
            db.upsert_failure(conn, STAGE, row["id"], None, f"{type(exc).__name__}: {exc}"[:2000])

    try:
        stats.interrupted = not runner.run_jobs(polite(), work, on_result, on_error, 1, total=len(jobs), desc="asr")
    finally:
        if own:
            worker.close()
    if stats.stopped:
        runner.interrupted = True   # recorded as interrupted: resumable, nothing wrong
    return stats
