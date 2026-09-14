"""Phase 2: 400-px JPEG thumbnails for every readable image."""
from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from re2 import db
from re2.config import REPO_ROOT

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

THUMB_WIDTH = 400
THUMB_QUALITY = 85
WORKERS = 8
STAGE = "thumb"


def thumb_rel_path(cfg, sha256: str) -> Path:
    """Sharded thumbnail location, relative to the repo root when possible."""
    h = sha256.lower()
    path = Path(cfg.thumbs_dir) / h[0] / h[1] / h[2] / f"{h[3:]}.jpg"
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def make_thumbnail(src: Path, dest: Path, width: int = THUMB_WIDTH) -> None:
    with Image.open(src) as im:
        im.seek(0)  # first frame of animated GIF/WebP
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
        os.close(fd)
        try:
            im.save(tmp, "JPEG", quality=THUMB_QUALITY)
            os.replace(tmp, dest)
        except BaseException:
            os.unlink(tmp)
            raise


@dataclass
class ThumbStats:
    selected: int = 0
    made: int = 0
    failed: int = 0

    def summary(self) -> str:
        return f"selected={self.selected} made={self.made} failed={self.failed}"


def run(conn: sqlite3.Connection, cfg, limit: int | None = None) -> ThumbStats:
    stats = ThumbStats()
    skip = db.pending_failures(conn, STAGE)
    rows = conn.execute(
        """SELECT m.id, m.sha256,
                  (SELECT p.rel_path FROM media_path p WHERE p.media_id=m.id AND p.readable=1
                   ORDER BY p.id LIMIT 1) AS rel_path
           FROM media m
           WHERE m.kind='image' AND m.status='READY' AND m.thumb_path IS NULL AND m.sha256 IS NOT NULL
           ORDER BY m.id"""
    ).fetchall()
    jobs = [r for r in rows if r["id"] not in skip and r["rel_path"]]
    if limit is not None:
        jobs = jobs[:limit]
    stats.selected = len(jobs)

    def work(row):
        rel = thumb_rel_path(cfg, row["sha256"])
        make_thumbnail(Path(cfg.corpus_root) / row["rel_path"], REPO_ROOT / rel)
        return rel

    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(work, r): r for r in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                rel = fut.result()
            except Exception as exc:  # noqa: BLE001 - Pillow raises many types
                stats.failed += 1
                log.warning("thumbnail failed for media %s (%s): %s", row["id"], row["rel_path"], exc)
                db.upsert_failure(conn, STAGE, row["id"], None, f"{type(exc).__name__}: {exc}")
                continue
            conn.execute("UPDATE media SET thumb_path=?, updated_at=datetime('now') WHERE id=?",
                         (str(rel), row["id"]))
            db.clear_failure(conn, STAGE, row["id"])
            stats.made += 1
            if n % 200 == 0:
                conn.commit()
                log.info("thumbnails %d/%d", n, len(jobs))
    conn.commit()
    return stats
