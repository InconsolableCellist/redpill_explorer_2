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
