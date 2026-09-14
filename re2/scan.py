"""Phase 1: read-only manifest of the corpus.

Nothing in this module writes under cfg.corpus_root. Files are only opened
with mode 'rb'.
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import magic
from PIL import Image

from re2 import db

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # we only read headers; corpus images are trusted-ish

TRASH_DIR = ".Trash-1000"
CHUNK = 1 << 20
HASH_WORKERS = 8
COMMIT_EVERY = 200

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".wmv", ".mov", ".avi", ".m4v", ".flv", ".unknown_video"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".gif", ".bmp"}
EXCLUDED_EXTS = {".part", ".tmp", ".zip", ".py", ".yml", ".tsv", ".txt", ".json"}
SENSITIVE_NAME = re.compile(
    r"(^|[^a-z])(cookies?|credentials?|tokens?|\.netrc|\.env)([^a-z]|$)", re.IGNORECASE
)
MEDIA_KINDS = ("image", "video", "pdf")


def now_ts() -> str:
    """UTC timestamp with microseconds; sorts lexically."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def is_excluded(name: str) -> bool:
    return bool(SENSITIVE_NAME.search(name)) or Path(name).suffix.lower() in EXCLUDED_EXTS


def guess_kind_from_name(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext == ".pdf":
        return "pdf"
    return "other"


def classify(mime: str | None, name: str) -> str:
    ext = Path(name).suffix.lower()
    if not mime:
        return guess_kind_from_name(name)
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/") or (mime == "application/octet-stream" and ext in VIDEO_EXTS):
        return "video"
    if mime == "application/pdf":
        return "pdf"
    return "other"


def walk(root: Path):
    """Yield DirEntry objects for files and symlinks, never following symlinks."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as exc:
            log.warning("cannot list %s: %s", current, exc)
            continue
        for entry in entries:
            if entry.is_symlink():
                yield entry
            elif entry.is_dir(follow_symlinks=False):
                if entry.name != TRASH_DIR:
                    stack.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield entry


_local = threading.local()


def _magic() -> magic.Magic:
    # libmagic handles are not thread-safe; keep one per thread.
    m = getattr(_local, "magic", None)
    if m is None:
        m = _local.magic = magic.Magic(mime=True)
    return m


@dataclass
class Probe:
    sha256: str
    mime: str | None
    width: int | None = None
    height: int | None = None


def probe_file(path: Path) -> Probe:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    try:
        mime = _magic().from_file(str(path))
    except Exception:  # noqa: BLE001 - python-magic raises assorted errors
        mime = mimetypes.guess_type(path.name)[0]
    probe = Probe(h.hexdigest(), mime)
    if classify(mime, path.name) == "image":
        try:
            with Image.open(path) as im:
                probe.width, probe.height = im.size
        except Exception:  # noqa: BLE001
            pass
    return probe


@dataclass
class ScanStats:
    seen: int = 0
    hashed: int = 0
    unchanged: int = 0
    broken_symlinks: int = 0
    excluded: int = 0
    errors: int = 0
    deferred: int = 0
    started: float = field(default_factory=time.monotonic)

    def summary(self) -> str:
        return (
            f"seen={self.seen} new_hashed={self.hashed} unchanged={self.unchanged} "
            f"broken_symlinks={self.broken_symlinks} excluded={self.excluded} "
            f"errors={self.errors} deferred_by_limit={self.deferred} "
            f"elapsed={time.monotonic() - self.started:.1f}s"
        )


def _upsert_media(conn: sqlite3.Connection, uid: str, sha256: str | None, kind: str,
                  mime: str | None, size: int | None, status: str,
                  width: int | None = None, height: int | None = None) -> int:
    row = conn.execute("SELECT id FROM media WHERE uid=?", (uid,)).fetchone()
    if row:
        conn.execute(
            """UPDATE media SET sha256=COALESCE(?, sha256), mime=COALESCE(?, mime),
                 size=COALESCE(?, size), width=COALESCE(?, width), height=COALESCE(?, height),
                 kind=CASE WHEN kind='other' OR ? != 'other' THEN ? ELSE kind END,
                 updated_at=datetime('now')
               WHERE id=?""",
            (sha256, mime, size, width, height, kind, kind, row[0]),
        )
        return row[0]
    cur = conn.execute(
        """INSERT INTO media(uid, sha256, kind, mime, size, width, height, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (uid, sha256, kind, mime, size, width, height, status),
    )
    return cur.lastrowid


def _upsert_path(conn: sqlite3.Connection, media_id: int, rel_path: str, is_symlink: bool,
                 target: str | None, readable: bool, size: int | None, mtime_ns: int | None,
                 ts: str) -> None:
    conn.execute(
        """INSERT INTO media_path(media_id, rel_path, is_symlink, symlink_target, readable,
                                  size, mtime_ns, first_seen, last_seen)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(rel_path) DO UPDATE SET
             media_id=excluded.media_id, is_symlink=excluded.is_symlink,
             symlink_target=excluded.symlink_target, readable=excluded.readable,
             size=excluded.size, mtime_ns=excluded.mtime_ns, last_seen=excluded.last_seen""",
        (media_id, rel_path, int(is_symlink), target, int(readable), size, mtime_ns, ts, ts),
    )


def refresh_statuses(conn: sqlite3.Connection) -> int:
    """Derive READY / THUMBNAIL_ONLY / SOURCE_OFFLINE from paths and thumbnails."""
    cur = conn.execute(
        """UPDATE media SET status=new.status, updated_at=datetime('now')
           FROM (SELECT m.id,
                        CASE WHEN EXISTS (SELECT 1 FROM media_path p
                                          WHERE p.media_id=m.id AND p.readable=1) THEN 'READY'
                             WHEN m.thumb_path IS NOT NULL THEN 'THUMBNAIL_ONLY'
                             ELSE 'SOURCE_OFFLINE' END AS status
                 FROM media m
                 WHERE m.kind IN ('image', 'video', 'pdf')
                   AND m.status NOT IN ('EXCLUDED', 'UNSUPPORTED')) AS new
           WHERE media.id=new.id AND media.status != new.status"""
    )
    return cur.rowcount


def _delete_orphan_missing(conn: sqlite3.Connection) -> int:
    """Drop 'missing:' placeholders whose symlink has since been replaced by a real file."""
    cur = conn.execute(
        """DELETE FROM media WHERE uid LIKE 'missing:%'
             AND NOT EXISTS (SELECT 1 FROM media_path p WHERE p.media_id=media.id)
             AND NOT EXISTS (SELECT 1 FROM doc d WHERE d.media_id=media.id)
             AND NOT EXISTS (SELECT 1 FROM source s WHERE s.media_id=media.id)
             AND NOT EXISTS (SELECT 1 FROM media_tag t WHERE t.media_id=media.id)
             AND legacy_caption IS NULL"""
    )
    return cur.rowcount


def scan(conn: sqlite3.Connection, cfg, limit: int | None = None, rehash: bool = False) -> ScanStats:
    root = Path(cfg.corpus_root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus root not found: {root}")
    stats = ScanStats()
    scan_start = now_ts()
    conn.execute("DELETE FROM failure WHERE stage='scan'")

    known = {
        r["rel_path"]: (r["size"], r["mtime_ns"])
        for r in conn.execute("SELECT rel_path, size, mtime_ns FROM media_path")
    }

    to_hash: list[tuple[Path, str, bool, str | None, os.stat_result]] = []
    pending = 0

    def record_error(rel_path: str, exc: BaseException) -> None:
        stats.errors += 1
        log.warning("scan error %s: %s", rel_path, exc)
        conn.execute(
            "INSERT INTO failure(stage, media_id, doc_id, attempts, error) VALUES ('scan', NULL, NULL, 1, ?)",
            (f"{rel_path}: {exc}",),
        )

    for entry in walk(root):
        stats.seen += 1
        path = Path(entry.path)
        rel_path = os.path.relpath(entry.path, root)
        try:
            rel_path.encode("utf-8")
        except UnicodeEncodeError as exc:
            record_error(repr(rel_path), exc)
            continue
        ts = now_ts()
        try:
            is_link = entry.is_symlink()
            target = os.readlink(path) if is_link else None
            if is_link and not path.exists():
                stats.broken_symlinks += 1
                uid = "missing:" + hashlib.sha1(target.encode("utf-8", "surrogateescape")).hexdigest()
                media_id = _upsert_media(conn, uid, None, guess_kind_from_name(target), None, None,
                                         "SOURCE_OFFLINE")
                _upsert_path(conn, media_id, rel_path, True, target, False, None, None, ts)
                pending += 1
                continue
            st = path.stat()
            if is_link and not os.path.isfile(path):
                continue  # symlink to a directory: never followed

            if is_excluded(path.name):
                stats.excluded += 1
                if known.get(rel_path) != (st.st_size, st.st_mtime_ns):
                    # Never read excluded files; key them by path instead of content.
                    uid = "excluded:" + hashlib.sha1(rel_path.encode()).hexdigest()
                    media_id = _upsert_media(conn, uid, None, "other", None, st.st_size, "EXCLUDED")
                    conn.execute("UPDATE media SET status='EXCLUDED', kind='other' WHERE id=?", (media_id,))
                    _upsert_path(conn, media_id, rel_path, is_link, target, True, st.st_size,
                                 st.st_mtime_ns, ts)
                else:
                    conn.execute("UPDATE media_path SET last_seen=?, readable=1 WHERE rel_path=?", (ts, rel_path))
                pending += 1
                continue

            if not rehash and known.get(rel_path) == (st.st_size, st.st_mtime_ns):
                stats.unchanged += 1
                conn.execute("UPDATE media_path SET last_seen=?, readable=1 WHERE rel_path=?", (ts, rel_path))
                pending += 1
            elif limit is not None and len(to_hash) >= limit:
                stats.deferred += 1
                if rel_path in known:  # changed but deferred: keep the old row alive
                    conn.execute("UPDATE media_path SET last_seen=? WHERE rel_path=?", (ts, rel_path))
            else:
                to_hash.append((path, rel_path, is_link, target, st))
        except OSError as exc:
            record_error(rel_path, exc)
        if pending >= COMMIT_EVERY:
            conn.commit()
            pending = 0
    conn.commit()
    log.info("walk done: %d entries, %d to hash", stats.seen, len(to_hash))

    with ThreadPoolExecutor(HASH_WORKERS) as pool:
        futures = [(item, pool.submit(probe_file, item[0])) for item in to_hash]
        for n, ((path, rel_path, is_link, target, st), fut) in enumerate(futures, 1):
            try:
                probe = fut.result()
            except OSError as exc:
                record_error(rel_path, exc)
                continue
            kind = classify(probe.mime, path.name)
            status = "READY" if kind in MEDIA_KINDS else "UNSUPPORTED"
            media_id = _upsert_media(conn, probe.sha256, probe.sha256, kind, probe.mime, st.st_size,
                                     status, probe.width, probe.height)
            _upsert_path(conn, media_id, rel_path, is_link, target, True, st.st_size, st.st_mtime_ns,
                         now_ts())
            stats.hashed += 1
            if n % COMMIT_EVERY == 0:
                conn.commit()
                log.info("hashed %d/%d", n, len(to_hash))
    conn.commit()

    conn.execute(
        "UPDATE media_path SET readable=0 WHERE last_seen < ? AND readable=1", (scan_start,)
    )
    _delete_orphan_missing(conn)
    refresh_statuses(conn)
    db.kv_set(conn, "last_scan_at", scan_start)
    conn.commit()
    return stats
