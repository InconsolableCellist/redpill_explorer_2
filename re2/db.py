"""SQLite connection, schema and small shared helpers."""
from __future__ import annotations

import re
import sqlite3
import time
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
CREATE TABLE IF NOT EXISTS work_claim (             -- one batch run per item when several run at once
  stage TEXT NOT NULL, media_id INTEGER NOT NULL,
  owner TEXT NOT NULL,                              -- host:pid:server of the run holding it
  expires_at REAL NOT NULL,                         -- unix time; a crashed run's claims lapse
  PRIMARY KEY (stage, media_id)
);
CREATE TABLE IF NOT EXISTS chat_session (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  title TEXT
);
CREATE TABLE IF NOT EXISTS post (                    -- one archived post: a screenshot plus its photos and videos
  id INTEGER PRIMARY KEY,
  url TEXT NOT NULL UNIQUE,
  date TEXT, folder TEXT, restricted INTEGER,
  primary_media_id INTEGER NOT NULL REFERENCES media(id)   -- the screenshot when there is one
);
CREATE TABLE IF NOT EXISTS post_member (
  post_id INTEGER NOT NULL REFERENCES post(id),
  media_id INTEGER NOT NULL REFERENCES media(id),
  role TEXT NOT NULL,                               -- screenshot | photo | video
  seq INTEGER NOT NULL,                             -- 0 = screenshot, then photos, then videos
  PRIMARY KEY (post_id, media_id)
);
CREATE INDEX IF NOT EXISTS ix_post_member_media ON post_member(media_id);
CREATE TABLE IF NOT EXISTS entity_alias (            -- entities.py: a merged-away entity and the one it became
  alias_id INTEGER PRIMARY KEY REFERENCES entity(id),
  canonical_id INTEGER NOT NULL REFERENCES entity(id)
);
CREATE TABLE IF NOT EXISTS cluster (                 -- topics and themes (cluster.py); the latest run is kv.cluster_run
  id INTEGER PRIMARY KEY,
  level TEXT NOT NULL,                              -- theme | topic
  parent_id INTEGER REFERENCES cluster(id),         -- a topic's theme
  run_id TEXT NOT NULL,
  label TEXT,
  label_source TEXT                                 -- keywords | llm
);
CREATE INDEX IF NOT EXISTS ix_cluster_run ON cluster(run_id, level);
CREATE TABLE IF NOT EXISTS cluster_member (
  cluster_id INTEGER NOT NULL REFERENCES cluster(id),
  media_id INTEGER NOT NULL REFERENCES media(id),
  PRIMARY KEY (cluster_id, media_id)
);
CREATE INDEX IF NOT EXISTS ix_cluster_member_media ON cluster_member(media_id);
CREATE TABLE IF NOT EXISTS pdf_ocr_queue (          -- scanned PDF pages waiting for the vision model
  media_id INTEGER NOT NULL REFERENCES media(id),
  page INTEGER NOT NULL,                            -- 1-based
  PRIMARY KEY (media_id, page)
);
CREATE TABLE IF NOT EXISTS selection (              -- items picked on the map, for sel:<id> scopes (search.py)
  id TEXT PRIMARY KEY,                              -- a hash of the id list, so the same set reuses its row
  media_ids TEXT NOT NULL,                          -- comma-separated item ids (post primaries)
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS item_digest (             -- digest.py: a short title and summary per item
  media_id INTEGER PRIMARY KEY REFERENCES media(id),
  title TEXT NOT NULL,                              -- <= 12 words
  summary TEXT NOT NULL,                            -- 1 to 3 sentences
  item_date TEXT,
  date_source TEXT NOT NULL,                        -- as media_meta.date_source, or 'none'
  digest_json TEXT,                                 -- {"event_hint": ...}
  source TEXT NOT NULL,                             -- caption (copied from a current caption) | model
  input_hash TEXT NOT NULL,                         -- what it was made from; a change makes it due again
  model TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
-- The wiki (wiki.py). cluster_id and entity_id carry no foreign keys: recluster deletes old runs.
CREATE TABLE IF NOT EXISTS wiki_page (
  id INTEGER PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,                        -- event/2024-11-trudeau-..., topic/..., theme/..., entity/..., overview
  kind TEXT NOT NULL,                               -- event | topic | theme | entity | overview
  title TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',                 -- one or two sentences, shown in lists
  body_md TEXT NOT NULL DEFAULT '',                 -- Markdown with [m:ID] citations and [p:slug] links; '' until written
  date_start TEXT, date_end TEXT,
  cluster_id INTEGER,
  entity_id INTEGER,
  parent_slug TEXT,
  named INTEGER NOT NULL DEFAULT 0,                 -- 1 once the model has titled it (topics and themes start as keywords)
  member_hash TEXT NOT NULL DEFAULT '',             -- members and what was read of them, when written
  stale INTEGER NOT NULL DEFAULT 1,
  model TEXT,
  members_read INTEGER,                             -- coverage when written: items the model read, directly or
  members_total INTEGER,                            --   through child pages that read them, of all members then
  members_cited INTEGER,
  build_json TEXT,                                  -- batches, reduce passes, child pages read, seconds
  events_split_at TEXT,                             -- topics: when their items were grouped into events
  built_at TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_wiki_page_parent ON wiki_page(parent_slug);
CREATE TABLE IF NOT EXISTS wiki_member (             -- items a page is about (post primaries, like clusters)
  page_id INTEGER NOT NULL REFERENCES wiki_page(id),
  media_id INTEGER NOT NULL REFERENCES media(id),
  origin TEXT NOT NULL DEFAULT 'cluster',           -- cluster | split | attached | entity
  read_hash TEXT,                                   -- the item's record as the page's writer read it; NULL = not read
  added_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (page_id, media_id)
);
CREATE INDEX IF NOT EXISTS ix_wiki_member_media ON wiki_member(media_id);
CREATE TABLE IF NOT EXISTS wiki_citation (           -- items a page's text cites
  page_id INTEGER NOT NULL REFERENCES wiki_page(id),
  media_id INTEGER NOT NULL REFERENCES media(id),
  n INTEGER NOT NULL,
  PRIMARY KEY (page_id, media_id)
);
CREATE TABLE IF NOT EXISTS wiki_link (
  from_page INTEGER NOT NULL REFERENCES wiki_page(id),
  to_page INTEGER NOT NULL REFERENCES wiki_page(id),
  PRIMARY KEY (from_page, to_page)
);
CREATE TABLE IF NOT EXISTS wiki_note (               -- map-step notes, so an interrupted build resumes without rereading
  hash TEXT PRIMARY KEY,                            -- of the batch's prompt
  page_id INTEGER NOT NULL,
  notes TEXT NOT NULL,
  model TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_wiki_note_page ON wiki_note(page_id);
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
  title, summary, body_md, content='wiki_page', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS wiki_page_ai AFTER INSERT ON wiki_page BEGIN
  INSERT INTO wiki_fts(rowid, title, summary, body_md) VALUES (new.id, new.title, new.summary, new.body_md);
END;
CREATE TRIGGER IF NOT EXISTS wiki_page_ad AFTER DELETE ON wiki_page BEGIN
  INSERT INTO wiki_fts(wiki_fts, rowid, title, summary, body_md) VALUES ('delete', old.id, old.title, old.summary, old.body_md);
END;
CREATE TRIGGER IF NOT EXISTS wiki_page_au AFTER UPDATE OF title, summary, body_md ON wiki_page BEGIN
  INSERT INTO wiki_fts(wiki_fts, rowid, title, summary, body_md) VALUES ('delete', old.id, old.title, old.summary, old.body_md);
  INSERT INTO wiki_fts(rowid, title, summary, body_md) VALUES (new.id, new.title, new.summary, new.body_md);
END;
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
    ("media_meta", "date_source", "TEXT"),    # owner | source | filename | content | mtime | none
    ("chat_log", "session_id", "TEXT"),       # ask.py
    ("chat_log", "mode", "TEXT"),             # informed | archive
    ("chat_log", "trace_json", "TEXT"),       # the agent's steps: action, args, result size, timing
    ("chat_log", "context_json", "TEXT"),     # what the user attached: items, related, depth
    ("chat_session", "kind", "TEXT"),         # NULL or 'ask' | 'argue'
    ("chat_session", "seed_json", "TEXT"),
    ("failure", "kind", "TEXT"),              # item | transient (the service was down, not the item's fault)    # argue: the Ask question and answer a debate started from
]


def classify_failures(conn: sqlite3.Connection) -> int:
    """Label rows written before failures were classified, so old outages stop retiring items."""
    rows = conn.execute("SELECT id, error FROM failure WHERE kind IS NULL").fetchall()
    conn.executemany("UPDATE failure SET kind=? WHERE id=?", [(failure_kind(e), i) for i, e in rows])
    return len(rows)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    for table, column, sql_type in ADDED_COLUMNS:
        if column not in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
    classify_failures(conn)
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

# A failure that says nothing about the item: the service was down, timed out, hung up, was out of
# context or returned a server error. Retrying these forever is right; counting them against the item
# is not, and it is how a few hours of a dead vision server used to retire images for good.
TRANSIENT = re.compile(
    r"ConnectError|ConnectTimeout|ReadTimeout|WriteTimeout|PoolTimeout|ReadError|WriteError|"
    r"RemoteProtocolError|ProxyError|Connection refused|Server disconnected|Temporary failure in name resolution|"
    r"Name or service not known|Server error '5\d\d|HTTP 5\d\d|Context size has been exceeded|"
    r"server_error|model unavailable|Errno 111|Errno 104|Errno 110",
    re.IGNORECASE)


def failure_kind(error: str) -> str:
    return "transient" if TRANSIENT.search(error or "") else "item"


def upsert_failure(conn: sqlite3.Connection, stage: str, media_id: int | None,
                   doc_id: int | None, error: str, attempts: int | None = None) -> None:
    row = conn.execute(
        "SELECT id FROM failure WHERE stage=? AND media_id IS ? AND doc_id IS ?",
        (stage, media_id, doc_id),
    ).fetchone()
    kind = failure_kind(error)
    if row:
        if attempts is None:
            conn.execute(
                "UPDATE failure SET attempts=attempts+1, error=?, kind=?, updated_at=datetime('now') WHERE id=?",
                (error, kind, row[0]),
            )
        else:
            conn.execute(
                "UPDATE failure SET attempts=?, error=?, kind=?, updated_at=datetime('now') WHERE id=?",
                (attempts, error, kind, row[0]),
            )
    else:
        conn.execute(
            "INSERT INTO failure(stage, media_id, doc_id, attempts, error, kind) VALUES (?, ?, ?, ?, ?, ?)",
            (stage, media_id, doc_id, attempts or 1, error, kind),
        )


def clear_failure(conn: sqlite3.Connection, stage: str, media_id: int | None,
                  doc_id: int | None = None) -> None:
    conn.execute(
        "DELETE FROM failure WHERE stage=? AND media_id IS ? AND doc_id IS ?",
        (stage, media_id, doc_id),
    )


def pending_failures(conn: sqlite3.Connection, stage: str, column: str = "media_id") -> set[int]:
    """Ids (media_id or doc_id) that have failed MAX_ATTEMPTS times for their own reasons and should be
    skipped. Failures from a service being down or overloaded are retried however often they happen."""
    if column not in ("media_id", "doc_id"):
        raise ValueError(column)
    rows = conn.execute(
        f"""SELECT {column} FROM failure WHERE stage=? AND attempts>=? AND {column} IS NOT NULL
            AND coalesce(kind, 'item') != 'transient'""",
        (stage, MAX_ATTEMPTS),
    )
    return {r[0] for r in rows}


def claim(conn: sqlite3.Connection, stage: str, media_id: int, owner: str, lease_s: float) -> bool:
    """Take (or renew) an item for this run; False when another run holds an unexpired claim."""
    now = time.time()
    with conn:
        cur = conn.execute(
            """INSERT INTO work_claim(stage, media_id, owner, expires_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(stage, media_id) DO UPDATE SET owner=excluded.owner, expires_at=excluded.expires_at
               WHERE work_claim.owner=excluded.owner OR work_claim.expires_at < ?""",
            (stage, media_id, owner, now + lease_s, now))
    return cur.rowcount == 1


def release(conn: sqlite3.Connection, stage: str, media_id: int, owner: str) -> None:
    with conn:
        conn.execute("DELETE FROM work_claim WHERE stage=? AND media_id=? AND owner=?", (stage, media_id, owner))
