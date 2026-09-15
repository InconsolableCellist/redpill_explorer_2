"""Phase 4.2: transcribe videos with faster-whisper on the local GPU."""
from __future__ import annotations

import ctypes
import importlib.util
import logging
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from re2 import db, runner

log = logging.getLogger(__name__)

STAGE = "asr"
MODEL_TAG = "faster-whisper-large-v3"
WINDOW_S = 60.0
WINDOW_CHARS = 600
FFMPEG_TIMEOUT = 1800
NO_SPEECH_ATTEMPTS = 99  # recorded as a "failure" so the video is never selected again


def preload_cuda_libraries() -> None:
    """ctranslate2 cannot find the pip-installed CUDA libraries unless they are loaded first."""
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue
        for so in sorted(Path(spec.submodule_search_locations[0], "lib").glob("*.so*")):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def load_model(cfg):
    preload_cuda_libraries()
    from faster_whisper import WhisperModel

    log.info("loading %s on %s (%s)", cfg.asr_model, cfg.asr_device, cfg.asr_compute_type)
    return WhisperModel(cfg.asr_model, device=cfg.asr_device, compute_type=cfg.asr_compute_type)


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
        raise RuntimeError(f"ffmpeg audio extraction failed: {exc.stderr.decode(errors='replace')[:300]}") from exc


def transcribe(model, cfg, media_id: int, path: Path) -> tuple[list[Segment], str | None]:
    wav = Path(cfg.data_dir) / "audio" / f"{media_id}.wav"
    try:
        extract_audio(path, wav)
        segments, info = model.transcribe(str(wav), beam_size=1, vad_filter=True)
        out = [Segment(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in segments if s.text.strip()]
        return out, getattr(info, "language", None)
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


def store(conn: sqlite3.Connection, media_id: int, segments: list[Segment], language: str | None) -> int:
    with conn:
        if not segments:
            db.upsert_failure(conn, STAGE, media_id, None, "no_speech", attempts=NO_SPEECH_ATTEMPTS)
            return 0
        conn.execute("DELETE FROM transcript_segment WHERE media_id=?", (media_id,))
        conn.execute("DELETE FROM doc WHERE media_id=? AND kind='transcript'", (media_id,))
        conn.executemany("INSERT INTO transcript_segment(media_id, start_s, end_s, text) VALUES (?, ?, ?, ?)",
                         [(media_id, s.start, s.end, s.text) for s in segments])
        wins = windows(segments)
        conn.executemany(
            "INSERT INTO doc(media_id, kind, seq, start_s, end_s, text, model) VALUES (?, 'transcript', ?, ?, ?, ?, ?)",
            [(media_id, n, w.start, w.end, w.text, MODEL_TAG) for n, w in enumerate(wins)])
        if language:
            conn.execute(
                """INSERT INTO media_meta(media_id, language) VALUES (?, ?)
                   ON CONFLICT(media_id) DO UPDATE SET language=COALESCE(media_meta.language, excluded.language)""",
                (media_id, language))
        db.clear_failure(conn, STAGE, media_id)
        return len(wins)


@dataclass
class AsrStats:
    selected: int = 0
    done: int = 0
    no_speech: int = 0
    failed: int = 0
    audio_s: float = 0.0
    interrupted: bool = False

    def summary(self) -> str:
        return (f"selected={self.selected} done={self.done} no_speech={self.no_speech} failed={self.failed} "
                f"audio_hours={self.audio_s / 3600:.2f}" + (" INTERRUPTED" if self.interrupted else ""))


def select_jobs(conn: sqlite3.Connection, limit: int | None) -> list[sqlite3.Row]:
    skip = db.pending_failures(conn, STAGE)
    rows = conn.execute(
        """SELECT m.id, m.duration,
                  (SELECT p.rel_path FROM media_path p WHERE p.media_id=m.id AND p.readable=1
                   ORDER BY p.id LIMIT 1) AS rel_path
           FROM media m
           WHERE m.kind='video' AND m.status='READY' AND m.has_audio=1
             AND NOT EXISTS (SELECT 1 FROM doc d WHERE d.media_id=m.id AND d.kind='transcript')
           ORDER BY m.id""").fetchall()
    jobs = [r for r in rows if r["id"] not in skip and r["rel_path"]]
    return jobs[:limit] if limit is not None else jobs


def run(conn: sqlite3.Connection, cfg, limit: int | None = None, model=None) -> AsrStats:
    jobs = select_jobs(conn, limit)
    stats = AsrStats(selected=len(jobs))
    if not jobs:
        return stats
    model = model or load_model(cfg)

    def work(row):
        return transcribe(model, cfg, row["id"], Path(cfg.corpus_root) / row["rel_path"])

    def on_result(row, result):
        segments, language = result
        if store(conn, row["id"], segments, language):
            stats.done += 1
        else:
            stats.no_speech += 1
        stats.audio_s += row["duration"] or 0

    def on_error(row, exc):
        stats.failed += 1
        log.warning("asr %s (%s) failed: %s", row["id"], row["rel_path"], exc)
        with conn:
            db.upsert_failure(conn, STAGE, row["id"], None, f"{type(exc).__name__}: {exc}"[:2000])

    stats.interrupted = not runner.run_jobs(jobs, work, on_result, on_error, 1, total=len(jobs), desc="asr")
    return stats
