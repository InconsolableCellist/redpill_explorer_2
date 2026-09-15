"""Phase 4.1: ffprobe metadata, keyframe JPEGs and thumbnails for videos (local CPU)."""
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image

from re2 import db, runner, thumbs
from re2.config import REPO_ROOT

log = logging.getLogger(__name__)

STAGE = "keyframe"
FRAME_INTERVAL_S = 15.0
MAX_FRAMES = 8
PHASH_MIN_DISTANCE = 7      # frames within 6 bits of a kept frame are dropped
FFPROBE_TIMEOUT = 60
FFMPEG_TIMEOUT = 120
WORKERS = 4


class VideoError(Exception):
    pass


def probe(path: Path) -> dict:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, check=True, timeout=FFPROBE_TIMEOUT)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", b"") or b""
        raise VideoError(f"ffprobe failed: {stderr.decode(errors='replace')[:300] or exc}") from exc
    data = json.loads(out.stdout or b"{}")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    duration = None
    for candidate in [(data.get("format") or {}).get("duration")] + [s.get("duration") for s in streams]:
        try:
            if candidate is not None and float(candidate) > 0:
                duration = float(candidate)
                break
        except ValueError:
            continue
    width, height = (video.get("width"), video.get("height")) if video else (None, None)
    rotation = 0
    for side in (video or {}).get("side_data_list") or []:
        if "rotation" in side:
            rotation = abs(int(side["rotation"])) % 180
    if rotation == 90 and width and height:
        width, height = height, width
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "has_video": video is not None,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def frame_times(duration: float) -> list[float]:
    first = min(1.0, duration / 2)
    times = [first]
    t = FRAME_INTERVAL_S
    while t < duration - 0.5 and len(times) < MAX_FRAMES:
        times.append(t)
        t += FRAME_INTERVAL_S
    return times


def extract_frame(path: Path, t: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
             "-vf", "scale='min(1024,iw)':-2", "-q:v", "3", str(out)],
            capture_output=True, check=True, timeout=FFMPEG_TIMEOUT)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.debug("frame at %.1fs failed for %s: %s", t, path, exc)
        return False
    return out.is_file() and out.stat().st_size > 0


def keyframe_dir(cfg, media_id: int) -> Path:
    return Path(cfg.data_dir) / "keyframes" / f"{media_id % 1000:03d}"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass
class ProbeResult:
    media_id: int
    info: dict
    frames: list[tuple[int, float, Path, str]] = field(default_factory=list)  # seq, t_s, path, phash
    thumb: str | None = None


def process(cfg, media_id: int, sha256: str | None, path: Path, needs_thumb: bool) -> ProbeResult:
    info = probe(path)
    result = ProbeResult(media_id, info)
    if not info["has_video"] or not info["duration"]:
        return result
    out_dir = keyframe_dir(cfg, media_id)
    kept: list[imagehash.ImageHash] = []
    for t in frame_times(info["duration"]):
        tmp = out_dir / f"{media_id}_tmp.jpg"
        if not extract_frame(path, t, tmp):
            continue
        with Image.open(tmp) as im:
            h = imagehash.phash(im)
        if any(h - k < PHASH_MIN_DISTANCE for k in kept):
            tmp.unlink(missing_ok=True)
            continue
        seq = len(kept)
        final = out_dir / f"{media_id}_{seq:02d}.jpg"
        tmp.replace(final)
        kept.append(h)
        result.frames.append((seq, round(t, 3), final, str(h)))
    if needs_thumb and result.frames and sha256:
        rel = thumbs.thumb_rel_path(cfg, sha256)
        thumbs.make_thumbnail(result.frames[0][2], REPO_ROOT / rel)
        result.thumb = str(rel)
    return result


def store(conn: sqlite3.Connection, r: ProbeResult) -> None:
    info = r.info
    with conn:
        # duration 0 marks "probed, nothing to extract" so the item is not selected again.
        conn.execute(
            """UPDATE media SET duration=?, width=COALESCE(?, width), height=COALESCE(?, height), has_audio=?,
                 thumb_path=COALESCE(thumb_path, ?), updated_at=datetime('now') WHERE id=?""",
            (info["duration"] or 0.0, info["width"], info["height"], int(info["has_audio"]), r.thumb, r.media_id))
        conn.execute("DELETE FROM keyframe WHERE media_id=?", (r.media_id,))
        conn.executemany(
            "INSERT INTO keyframe(media_id, seq, t_s, path, phash) VALUES (?, ?, ?, ?, ?)",
            [(r.media_id, seq, t, _rel(p), ph) for seq, t, p, ph in r.frames])
        db.clear_failure(conn, STAGE, r.media_id)


@dataclass
class VideoStats:
    selected: int = 0
    done: int = 0
    frames: int = 0
    failed: int = 0
    interrupted: bool = False

    def summary(self) -> str:
        return (f"selected={self.selected} done={self.done} frames={self.frames} failed={self.failed}"
                + (" INTERRUPTED" if self.interrupted else ""))


def run(conn: sqlite3.Connection, cfg, limit: int | None = None) -> VideoStats:
    skip = db.pending_failures(conn, STAGE)
    rows = conn.execute(
        """SELECT m.id, m.sha256, m.thumb_path,
                  (SELECT p.rel_path FROM media_path p WHERE p.media_id=m.id AND p.readable=1
                   ORDER BY p.id LIMIT 1) AS rel_path
           FROM media m WHERE m.kind='video' AND m.status='READY' AND m.duration IS NULL
           ORDER BY m.id""").fetchall()
    jobs = [r for r in rows if r["id"] not in skip and r["rel_path"]]
    if limit is not None:
        jobs = jobs[:limit]
    stats = VideoStats(selected=len(jobs))

    def work(row):
        return process(cfg, row["id"], row["sha256"], Path(cfg.corpus_root) / row["rel_path"],
                       needs_thumb=row["thumb_path"] is None)

    def on_result(row, result: ProbeResult):
        store(conn, result)
        stats.done += 1
        stats.frames += len(result.frames)

    def on_error(row, exc):
        stats.failed += 1
        log.warning("video %s (%s) failed: %s", row["id"], row["rel_path"], exc)
        with conn:
            db.upsert_failure(conn, STAGE, row["id"], None, f"{type(exc).__name__}: {exc}"[:2000])

    stats.interrupted = not runner.run_jobs(jobs, work, on_result, on_error, WORKERS,
                                            total=len(jobs), desc="video-probe")
    return stats
