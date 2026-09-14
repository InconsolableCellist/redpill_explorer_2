"""Phase 1: fold the pre-RE2 JSON artifacts into the database.

Inputs are read-only: legacy/*.json (copies made in Task 0.1),
<corpus>/output.json and <corpus>/archive_metadata.json.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from re2 import scan, tags
from re2.config import REPO_ROOT
from re2.thumbs import thumb_rel_path

log = logging.getLogger(__name__)

HANOI_PREFIX = "/mnt/hanoi_data/storage/redpills/"
LEGACY_MODEL = "cogvlm2-legacy"


def _load_json(path: Path):
    with open(path, "rb") as f:
        return json.load(f)


@dataclass
class LegacyStats:
    captions: int = 0
    created_offline: int = 0
    tagged: int = 0
    output_rekeyed: int = 0
    output_repointed: int = 0
    output_created: int = 0
    thumbs: int = 0
    docs: int = 0

    def summary(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.__dict__.items())


def _media_by_uid(conn: sqlite3.Connection, uid: str) -> sqlite3.Row | None:
    return conn.execute("SELECT id, legacy_caption FROM media WHERE uid=?", (uid,)).fetchone()


def _create_offline(conn: sqlite3.Connection, sha256: str, filename: str) -> int:
    kind = scan.guess_kind_from_name(filename or "")
    if kind == "other":
        kind = "image"
    return conn.execute(
        "INSERT INTO media(uid, sha256, kind, status) VALUES (?, ?, ?, 'SOURCE_OFFLINE')",
        (sha256, sha256, kind),
    ).lastrowid


def reconcile(conn: sqlite3.Connection, cfg) -> LegacyStats:
    stats = LegacyStats()
    legacy_dir = Path(cfg.legacy_dir)
    captioned: dict = _load_json(legacy_dir / "images_captioned.json")
    tagged: dict = _load_json(legacy_dir / "images_captioned_tagged.json")

    # 1. Captions, spiciness and tags keyed by sha256.
    for sha in sorted(set(captioned) | set(tagged)):
        rec_t = tagged.get(sha) or {}
        rec_c = captioned.get(sha) or {}
        description = rec_t.get("description") or rec_c.get("description")
        filename = rec_t.get("filename") or rec_c.get("filename") or ""
        row = _media_by_uid(conn, sha)
        if row is None:
            media_id = _create_offline(conn, sha, filename)
            stats.created_offline += 1
        else:
            media_id = row["id"]
        conn.execute(
            "UPDATE media SET legacy_caption=?, legacy_spicy=?, updated_at=datetime('now') WHERE id=?",
            (description, rec_t.get("spicy"), media_id),
        )
        stats.captions += 1
        if rec_t.get("tags"):
            tags.set_media_tags(conn, media_id, rec_t["tags"], "legacy")
            stats.tagged += 1
    conn.commit()

    # 2. output.json: old hanoi paths with known sha256 and older captions.
    output_path = Path(cfg.legacy_output_json)
    output: dict = _load_json(output_path) if output_path.exists() else {}
    for sha, rec in output.items():
        path = rec.get("path") or ""
        rel_path = path[len(HANOI_PREFIX):] if path.startswith(HANOI_PREFIX) else None
        existing = _media_by_uid(conn, sha)
        mp = None
        if rel_path:
            mp = conn.execute(
                "SELECT p.id AS path_id, m.id AS media_id, m.sha256 FROM media_path p "
                "JOIN media m ON m.id=p.media_id WHERE p.rel_path=?",
                (rel_path,),
            ).fetchone()
        if mp is not None and mp["sha256"] is None:
            if existing is None:
                conn.execute(
                    "UPDATE media SET sha256=?, uid=?, updated_at=datetime('now') WHERE id=?",
                    (sha, sha, mp["media_id"]),
                )
                stats.output_rekeyed += 1
            else:
                conn.execute("UPDATE media_path SET media_id=? WHERE id=?", (existing["id"], mp["path_id"]))
                stats.output_repointed += 1
            existing = _media_by_uid(conn, sha)
        if existing is None:
            if not rec.get("description"):
                continue
            _create_offline(conn, sha, rec.get("filename") or "")
            stats.output_created += 1
            existing = _media_by_uid(conn, sha)
        if existing["legacy_caption"] is None and rec.get("description"):
            conn.execute("UPDATE media SET legacy_caption=? WHERE id=?", (rec["description"], existing["id"]))
    scan._delete_orphan_missing(conn)
    conn.commit()

    # 3. Thumbnails already on disk.
    rows = conn.execute("SELECT id, sha256, thumb_path FROM media WHERE sha256 IS NOT NULL").fetchall()
    updates = []
    for r in rows:
        rel = thumb_rel_path(cfg, r["sha256"])
        found = str(rel) if (REPO_ROOT / rel).is_file() else None
        if found:
            stats.thumbs += 1
        if found != r["thumb_path"] and (found or r["thumb_path"]):
            updates.append((found, r["id"]))
    conn.executemany("UPDATE media SET thumb_path=? WHERE id=?", updates)
    scan.refresh_statuses(conn)
    conn.commit()

    # 4. One searchable doc per legacy caption.
    cur = conn.execute(
        f"""INSERT INTO doc(media_id, kind, seq, text, model)
            SELECT id, 'legacy_caption', 0, legacy_caption, '{LEGACY_MODEL}' FROM media
            WHERE legacy_caption IS NOT NULL AND trim(legacy_caption) != ''
            ON CONFLICT(media_id, kind, seq) DO UPDATE SET text=excluded.text
            WHERE doc.text != excluded.text"""
    )
    stats.docs = cur.rowcount
    conn.commit()
    return stats


@dataclass
class SourceStats:
    entries: int = 0
    files: int = 0
    linked: int = 0
    by_basename: int = 0
    not_found: int = 0

    def summary(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.__dict__.items())


def reconcile_sources(conn: sqlite3.Connection, cfg) -> SourceStats:
    """Attach archive_metadata.json URLs and dates to media via folder/filename."""
    stats = SourceStats()
    entries = _load_json(Path(cfg.archive_metadata))
    path_to_media = {
        r["rel_path"]: r["media_id"] for r in conn.execute("SELECT rel_path, media_id FROM media_path")
    }
    by_basename: dict[str, set[int]] = {}
    for rel_path, media_id in path_to_media.items():
        by_basename.setdefault(os.path.basename(rel_path), set()).add(media_id)

    rows = []
    for e in entries:
        stats.entries += 1
        folder = e.get("folder") or "."
        files = [("screenshot", e.get("screenshot"))]
        files += [("photo", n) for n in e.get("photos") or []]
        files += [("video", n) for n in e.get("videos") or []]
        for role, name in files:
            if not name:
                continue
            stats.files += 1
            rel_path = os.path.normpath(os.path.join(folder, name))
            media_id = path_to_media.get(rel_path)
            if media_id is None:
                candidates = by_basename.get(os.path.basename(name), set())
                if len(candidates) == 1:
                    media_id = next(iter(candidates))
                    stats.by_basename += 1
            if media_id is None:
                stats.not_found += 1
                log.debug("source file not found: %s", rel_path)
                continue
            stats.linked += 1
            rows.append((media_id, e.get("url"), e.get("date"), e.get("folder"),
                         int(bool(e.get("restricted"))), role))

    with conn:
        conn.execute("DELETE FROM source")
        conn.executemany(
            "INSERT INTO source(media_id, url, date, folder, restricted, role) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    if stats.not_found:
        log.info("%d archive_metadata files not found in the manifest", stats.not_found)
    return stats
