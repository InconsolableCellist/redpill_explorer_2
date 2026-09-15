"""SQLite connection, schema and small shared helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path

MAX_ATTEMPTS = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
  id INTEGER PRIMARY KEY,
  uid TEXT NOT NULL UNIQUE,          -- sha256 hex when content is readable or known; else 'missing:' + sha1(symlink target)
  sha256 TEXT,                       -- NULL when unknown
  kind TEXT NOT NULL,                -- image | video | pdf | other
  mime TEXT,
  size INTEGER,
  width INTEGER, height INTEGER,     -- images and videos
  duration REAL,                     -- videos, seconds
  has_audio INTEGER,                 -- videos
  status TEXT NOT NULL,              -- READY | THUMBNAIL_ONLY | SOURCE_OFFLINE | UNSUPPORTED | EXCLUDED
  thumb_path TEXT,                   -- relative to repo root, NULL if none
  legacy_caption TEXT,
  legacy_spicy REAL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_media_sha256 ON media(sha256);
CREATE TABLE IF NOT EXISTS media_path (
  id INTEGER PRIMARY KEY,
  media_id INTEGER NOT NULL REFERENCES media(id),
  rel_path TEXT NOT NULL UNIQUE,     -- relative to corpus_root
  is_symlink INTEGER NOT NULL DEFAULT 0,
  symlink_target TEXT,
  readable INTEGER NOT NULL,         -- 1 if the file could be opened at last scan
  size INTEGER, mtime_ns INTEGER,
  first_seen TEXT NOT NULL, last_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_media_path_media ON media_path(media_id);

CREATE TABLE IF NOT EXISTS source (                 -- from archive_metadata.json
  id INTEGER PRIMARY KEY,
  media_id INTEGER NOT NULL REFERENCES media(id),
  url TEXT, date TEXT, folder TEXT, restricted INTEGER, role TEXT  -- role: screenshot | photo | video
);
CREATE INDEX IF NOT EXISTS ix_source_media ON source(media_id);

CREATE TABLE IF NOT EXISTS doc (                    -- the unit of retrieval and citation
  id INTEGER PRIMARY KEY,
  media_id INTEGER NOT NULL REFERENCES media(id),
  kind TEXT NOT NULL,                -- caption | ocr | legacy_caption | transcript | keyframe | pdf_page
  seq INTEGER NOT NULL DEFAULT 0,    -- ordering within media (window index, keyframe index, page)
  start_s REAL, end_s REAL,          -- transcript windows and keyframes
  page INTEGER,                      -- pdf pages
  text TEXT NOT NULL,
  json TEXT,                         -- full structured extraction for caption/keyframe docs
  model TEXT,                        -- model id that produced it
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(media_id, kind, seq)
);
CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
  text, content='doc', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS doc_ai AFTER INSERT ON doc BEGIN
  INSERT INTO doc_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS doc_ad AFTER DELETE ON doc BEGIN
  INSERT INTO doc_fts(doc_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS doc_au AFTER UPDATE ON doc BEGIN
  INSERT INTO doc_fts(doc_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO doc_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS tag (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE      -- normalized (tags.normalize)
);
CREATE TABLE IF NOT EXISTS media_tag (
  media_id INTEGER NOT NULL REFERENCES media(id),
  tag_id INTEGER NOT NULL REFERENCES tag(id),
  weight REAL NOT NULL DEFAULT 1.0,
  origin TEXT NOT NULL,              -- vlm | legacy
  PRIMARY KEY (media_id, tag_id, origin)
);
CREATE TABLE IF NOT EXISTS entity (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,  -- person | organization | place | event | platform
  UNIQUE(name, type)
);
CREATE TABLE IF NOT EXISTS media_entity (
  media_id INTEGER NOT NULL REFERENCES media(id),
  entity_id INTEGER NOT NULL REFERENCES entity(id),
  PRIMARY KEY (media_id, entity_id)
);
CREATE TABLE IF NOT EXISTS media_meta (            -- scalar fields from the VLM extraction
  media_id INTEGER PRIMARY KEY REFERENCES media(id),
  sensitivity REAL, media_type TEXT, language TEXT, low_res INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS transcript_segment (
  id INTEGER PRIMARY KEY,
  media_id INTEGER NOT NULL REFERENCES media(id),
  start_s REAL NOT NULL, end_s REAL NOT NULL, text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ts_media ON transcript_segment(media_id);
CREATE TABLE IF NOT EXISTS keyframe (
  id INTEGER PRIMARY KEY,
  media_id INTEGER NOT NULL REFERENCES media(id),
  seq INTEGER NOT NULL, t_s REAL NOT NULL, path TEXT NOT NULL, phash TEXT,
  UNIQUE(media_id, seq)
);
CREATE TABLE IF NOT EXISTS failure (
  id INTEGER PRIMARY KEY,
  stage TEXT NOT NULL,               -- scan | thumb | vlm_image | asr | keyframe | vlm_keyframe | pdf | embed
  media_id INTEGER, doc_id INTEGER,
  attempts INTEGER NOT NULL DEFAULT 1,
  error TEXT, updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(stage, media_id, doc_id)
);
CREATE TABLE IF NOT EXISTS chat_log (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  question TEXT NOT NULL, rewrite_json TEXT, retrieved_doc_ids TEXT,
  answer TEXT, citations_json TEXT, model TEXT, latency_ms INTEGER
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);  -- e.g. last_scan_at
CREATE TABLE IF NOT EXISTS pdf_ocr_queue (          -- scanned PDF pages waiting for the vision model
  media_id INTEGER NOT NULL REFERENCES media(id),
  page INTEGER NOT NULL,                            -- 1-based
  PRIMARY KEY (media_id, page)
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# Columns added after a table was first created: (table, column, type).
ADDED_COLUMNS = [
    ("media_meta", "item_date", "TEXT"),      # dates.py: YYYY-MM-DD or YYYY-MM
    ("media_meta", "date_source", "TEXT"),    # source | filename | content | mtime | none
]


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    for table, column, sql_type in ADDED_COLUMNS:
        if column not in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
    conn.commit()


def open_db(cfg) -> sqlite3.Connection:
    conn = connect(cfg.db_path)
    init_schema(conn)
    return conn


def kv_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def kv_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO kv(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


# UNIQUE(stage, media_id, doc_id) does not dedupe rows where one column is NULL,
# so failures are looked up with IS comparisons rather than ON CONFLICT.

def upsert_failure(conn: sqlite3.Connection, stage: str, media_id: int | None,
                   doc_id: int | None, error: str, attempts: int | None = None) -> None:
    row = conn.execute(
        "SELECT id FROM failure WHERE stage=? AND media_id IS ? AND doc_id IS ?",
        (stage, media_id, doc_id),
    ).fetchone()
    if row:
        if attempts is None:
            conn.execute(
                "UPDATE failure SET attempts=attempts+1, error=?, updated_at=datetime('now') WHERE id=?",
                (error, row[0]),
            )
        else:
            conn.execute(
                "UPDATE failure SET attempts=?, error=?, updated_at=datetime('now') WHERE id=?",
                (attempts, error, row[0]),
            )
    else:
        conn.execute(
            "INSERT INTO failure(stage, media_id, doc_id, attempts, error) VALUES (?, ?, ?, ?, ?)",
            (stage, media_id, doc_id, attempts or 1, error),
        )


def clear_failure(conn: sqlite3.Connection, stage: str, media_id: int | None,
                  doc_id: int | None = None) -> None:
    conn.execute(
        "DELETE FROM failure WHERE stage=? AND media_id IS ? AND doc_id IS ?",
        (stage, media_id, doc_id),
    )


def pending_failures(conn: sqlite3.Connection, stage: str, column: str = "media_id") -> set[int]:
    """Ids (media_id or doc_id) that have failed MAX_ATTEMPTS times and should be skipped."""
    if column not in ("media_id", "doc_id"):
        raise ValueError(column)
    rows = conn.execute(
        f"SELECT {column} FROM failure WHERE stage=? AND attempts>=? AND {column} IS NOT NULL",
        (stage, MAX_ATTEMPTS),
    )
    return {r[0] for r in rows}
