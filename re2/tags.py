"""Tag normalization (PMI and legacy JSON export arrive in Task 2.5)."""
from __future__ import annotations

import re
import sqlite3
import tomllib
from functools import lru_cache

from re2.config import REPO_ROOT

SYNONYMS_FILE = REPO_ROOT / "tag_synonyms.toml"
_WS = re.compile(r"\s+")


def _basic(name: str) -> str:
    name = name.strip().lower().replace("_", " ").replace("-", " ").lstrip("#")
    return _WS.sub(" ", name).strip()


@lru_cache(maxsize=1)
def synonyms() -> dict[str, str]:
    if not SYNONYMS_FILE.exists():
        return {}
    with open(SYNONYMS_FILE, "rb") as f:
        raw = tomllib.load(f)
    return {_basic(k): _basic(v) for k, v in raw.items()}


def normalize(name: str) -> str:
    """Canonical tag name, or '' when the tag is too short to keep."""
    name = _basic(name)
    name = synonyms().get(name, name)
    return name if len(name) >= 2 else ""


def tag_id(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM tag WHERE name=?", (name,)).fetchone()
    if row:
        return row[0]
    return conn.execute("INSERT INTO tag(name) VALUES (?)", (name,)).lastrowid


def set_media_tags(conn: sqlite3.Connection, media_id: int, weights: dict[str, float], origin: str) -> int:
    """Replace the media's tags for one origin. Names are normalized; collisions keep the max weight."""
    merged: dict[str, float] = {}
    for raw, weight in weights.items():
        name = normalize(raw)
        if name:
            merged[name] = max(merged.get(name, 0.0), float(weight))
    conn.execute("DELETE FROM media_tag WHERE media_id=? AND origin=?", (media_id, origin))
    conn.executemany(
        "INSERT INTO media_tag(media_id, tag_id, weight, origin) VALUES (?, ?, ?, ?)",
        [(media_id, tag_id(conn, name), w, origin) for name, w in merged.items()],
    )
    return len(merged)


def renormalize(conn: sqlite3.Connection, dry_run: bool = False) -> tuple[int, int] | list[tuple[str, str, int]]:
    """Re-apply normalize() and the synonym map to stored tags, merging duplicates.

    Returns (tags renamed or merged, media_tag rows moved); with dry_run, the planned
    (old name, new name, items) changes instead, and nothing is written.
    """
    synonyms.cache_clear()
    if dry_run:
        plan = []
        for tag_id_, name in conn.execute("SELECT id, name FROM tag").fetchall():
            canonical = normalize(name)
            if canonical != name:
                n = conn.execute("SELECT COUNT(DISTINCT media_id) FROM media_tag WHERE tag_id=?", (tag_id_,)).fetchone()[0]
                plan.append((name, canonical, n))
        return sorted(plan, key=lambda p: -p[2])
    changed = moved = 0
    with conn:
        for tag_id_, name in conn.execute("SELECT id, name FROM tag").fetchall():
            canonical = normalize(name)
            if canonical == name:
                continue
            changed += 1
            if not canonical:
                conn.execute("DELETE FROM media_tag WHERE tag_id=?", (tag_id_,))
                conn.execute("DELETE FROM tag WHERE id=?", (tag_id_,))
                continue
            target = conn.execute("SELECT id FROM tag WHERE name=?", (canonical,)).fetchone()
            if target is None:
                conn.execute("UPDATE tag SET name=? WHERE id=?", (canonical, tag_id_))
                continue
            # Keep the higher weight when a media already has the canonical tag from the same origin.
            for media_id, weight, origin in conn.execute(
                    "SELECT media_id, weight, origin FROM media_tag WHERE tag_id=?", (tag_id_,)).fetchall():
                conn.execute(
                    """INSERT INTO media_tag(media_id, tag_id, weight, origin) VALUES (?, ?, ?, ?)
                       ON CONFLICT(media_id, tag_id, origin) DO UPDATE SET weight=max(weight, excluded.weight)""",
                    (media_id, target[0], weight, origin))
                moved += 1
            conn.execute("DELETE FROM media_tag WHERE tag_id=?", (tag_id_,))
            conn.execute("DELETE FROM tag WHERE id=?", (tag_id_,))
        # Relatedness rows of merged-away tags point at nothing now.
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='tag_relation'").fetchone():
            conn.execute("DELETE FROM tag_relation WHERE a NOT IN (SELECT id FROM tag) OR b NOT IN (SELECT id FROM tag)")
            conn.execute("DELETE FROM tag_relation_done WHERE tag_id NOT IN (SELECT id FROM tag)")
    return changed, moved
