# RE2 Coding Plan

Written 2026-09-14 for execution by a local coding model. Companion documents:
`RE2_INVESTIGATION_REPORT.md` (the audit) and `RE2_REPORT_REVIEW.md` (the
corrections this plan is based on). When the two disagree, this plan wins.

Work through the tasks in order. Each task says what to build, which files it
touches, how to verify it, and when to commit. Do not skip verification. Do not
start a task until the previous task's verification passes.

---

## 0. Ground rules (read before every session)

### 0.1 Hard constraints

1. **Never write under `~/Downloads/redpills`.** It is an NFS mount from the NAS.
   Open files there read-only. Never `os.remove`, `os.rename`, `shutil.move`,
   `open(..., 'w')`, or `os.symlink` on any path under it. All derived data goes
   under the repo's `data/` directory.
2. **No GPU work runs on this machine except ASR.** This machine has a GTX
   1080 Ti. Vision captioning, chat and embeddings are HTTP calls to the host
   `cube`. ASR (faster-whisper) runs locally on the 1080 Ti.
3. **Python 3.12 via `uv`.** The system Python is 3.14 and many ML wheels do
   not support it. Every Python command in this plan is run as
   `uv run <command>` from the repo root.
4. **One process owns the SQLite file at a time for writes.** Ingestion
   commands and the web server may run concurrently because the DB uses WAL
   mode, but never run two ingestion commands of the same kind at once.
5. **Long batch jobs are started by the human, not by the coding agent.** The
   coding agent runs every command with `--limit 5` (or similar) to prove it
   works, commits, and then tells the human the exact command to run for the
   full batch. Reason: `cube` loads only one model at a time, and the coding
   agent's own model runs on `cube`; a full captioning batch would evict it.

### 0.2 Hosts and endpoints

| Purpose | Value |
|---|---|
| Corpus root (read-only) | `/home/offipso/Downloads/redpills` |
| Trash to skip | any path containing `/.Trash-1000/` |
| LLM router (chat + vision) | `https://cube.tail99c274.ts.net:5000/v1` (OpenAI-compatible; plain `http://cube:5000` does NOT work) |
| Router behaviour | llama.cpp router mode, `--models-max 1`: the `model` field in each request selects the model and the router swaps automatically. Only one model is loaded at a time. |
| Vision model id | `gemma-4-31b-heretic-ara-batch-parallel6-q8-8k-cube` (8192 shared context, 6 slots). Use client concurrency 3 and `max_tokens` 1200. |
| Chat model id | `qwen38-27b-heretic-uncensored-q8-cube` |
| Embedding server | `http://cube:5010/v1/embeddings`, started separately on cube (Task 3.1). Model `nomic-embed-text-v1.5` (768 dims). Document prefix `search_document: `, query prefix `search_query: `. |
| ASR | faster-whisper `large-v3`, `device="cuda"`, `compute_type="int8_float32"` on the local 1080 Ti. Measured 19x realtime. |
| Web server | FastAPI on `127.0.0.1:4001` |

### 0.3 Existing artifacts that must be reconciled, never deleted

| File | Contents |
|---|---|
| `data/images_captioned_tagged.json` | 5,144 records keyed by sha256: `filename`, `description`, `tags` {name: weight}, `spicy` |
| `images_captioned.json` (also symlinked as `data/images_captioned.json`) | 5,503 records keyed by sha256: `filename`, `description` (359 never tagged) |
| `~/Downloads/redpills/output.json` | 3,610 records keyed by sha256: `filename`, `path` (old `/mnt/hanoi_data/storage/redpills/...` paths), `description` |
| `hash_path_cache.json` | 7,021 stale sha256 -> absolute path mappings from 2025-06-02 |
| `public/thumbnails/<h0>/<h1>/<h2>/<rest-of-hash>.jpg` | 7,392 400-px-wide JPEG thumbnails, sharded by the first three hex chars of the sha256 |
| `~/Downloads/redpills/archive_metadata.json` | list of 1,327 entries: `date`, `url`, `screenshot`, `photos[]`, `videos[]`, `restricted`, `folder` |
| `data/tags_with_sizes.json`, `data/tag_pairs_with_weights.json`, `data/adjusted_data.json` | derived files consumed by the existing D3/Three.js pages; regenerated in Task 2.6 |

### 0.4 Known facts about the corpus (2026-09-14)

- ~14,000 regular files: ~10,200 images (jpg, png, jpeg, webp, jfif, gif),
  3,718 videos (mp4, webm, mkv, wmv), 28 PDFs, a few dozen text/json/tsv/zip.
- 3,126 broken symlinks whose targets start with
  `/mnt/hanoi_data/storage/redpills/`. That volume is offline. 1,065 are videos.
  For 1,261 of them the sha256 is known from `output.json`, and 1,254 of those
  have a legacy caption and a thumbnail. None of the content exists elsewhere.
  If the human brings the volume back online later, the manifest must pick
  the files up automatically on the next scan (Task 1.4 handles this).
- Filenames may contain spaces, quotes, unicode, and `@`. Always pass paths
  as `pathlib.Path` objects, never build shell strings.

### 0.5 Repository layout to create

```
re2/                     Python package (all new backend code)
  __init__.py
  config.py              loads config.toml
  db.py                  SQLite connection, schema, migrations
  scan.py                Phase 1: manifest
  legacy.py              Phase 1: reconcile old JSON artifacts
  thumbs.py              Phase 2: thumbnails
  vlm.py                 Phase 2/4/5: vision model client + prompt + schema
  extract_images.py      Phase 2: run VLM over images
  tags.py                Phase 2: tag normalization, PMI, legacy JSON export
  embed.py               Phase 3: embedding client + index build/query
  search.py              Phase 3: lexical + dense + fusion
  api.py                 Phase 3/6: FastAPI app
  video.py               Phase 4: ffprobe, keyframes
  asr.py                 Phase 4: faster-whisper
  pdf.py                 Phase 5
  chat.py                Phase 6
  cli.py                 `re2` command entry point (argparse subcommands)
tests/                   pytest
public/                  static UI (existing pages kept, new search page added)
data/                    all derived data (gitignored): re2.sqlite, index/, keyframes/, audio/
config.toml              hosts, model ids, paths (committed; no secrets)
pyproject.toml
```

### 0.6 Conventions

- Every CLI subcommand accepts `--limit N` (process at most N new items) and
  `--dry-run` where it makes sense, prints a one-line summary at the end, and
  exits 0 on success.
- Every ingestion step is idempotent: re-running it processes only items that
  have no result row yet, and records failures in the `failure` table instead
  of crashing. A failed item is retried on the next run up to 3 attempts.
- Use `logging` (INFO to stderr), never bare `print` in library code.
- Use `httpx` for HTTP with explicit timeouts (vision: 600 s, chat: 300 s,
  embeddings: 60 s).
- Commit after each task with the message given in the task. Never commit
  `data/`, `public/thumbnails/`, `*.sqlite*`, `*.npy`, `node_modules/`.

---

## Phase 0: Repo hygiene and safety

### Task 0.1: Python project

**Do:**
1. Create `pyproject.toml`:
   ```toml
   [project]
   name = "re2"
   version = "0.1.0"
   requires-python = ">=3.12,<3.13"
   dependencies = [
     "fastapi>=0.115", "uvicorn[standard]>=0.30", "jinja2>=3.1",
     "httpx>=0.27", "pillow>=10", "numpy>=1.26", "tqdm>=4.66",
     "python-magic>=0.4.27", "imagehash>=4.3", "pymupdf>=1.24",
     "faster-whisper==1.2.1", "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
   ]
   [project.scripts]
   re2 = "re2.cli:main"
   [dependency-groups]
   dev = ["pytest>=8"]
   [tool.uv]
   python-preference = "managed"
   ```
2. Run `uv python install 3.12` then `uv sync`.
3. Replace `.gitignore` (currently a 696 KB generated file) with:
   ```
   node_modules/
   data/
   public/thumbnails/
   *.sqlite
   *.sqlite-wal
   *.sqlite-shm
   *.npy
   .venv/
   __pycache__/
   .idea/
   hash_path_cache.json
   images_captioned.json
   backup/
   ```
   Note: `data/images_captioned_tagged.json` and the other legacy JSON files
   under `data/` are inputs to Phase 1. They are already untracked; leave
   them on disk. Copy them once to `legacy/` (tracked? No: also gitignored;
   add `legacy/` to `.gitignore`). Concretely: `mkdir legacy && cp
   data/images_captioned_tagged.json images_captioned.json hash_path_cache.json
   legacy/` so Phase 1 reads from `legacy/` and `data/` can be regenerated.
4. Create `config.toml`:
   ```toml
   [paths]
   corpus_root = "/home/offipso/Downloads/redpills"
   data_dir = "data"
   legacy_dir = "legacy"
   thumbs_dir = "public/thumbnails"
   archive_metadata = "/home/offipso/Downloads/redpills/archive_metadata.json"
   legacy_output_json = "/home/offipso/Downloads/redpills/output.json"

   [llm]
   base_url = "https://cube.tail99c274.ts.net:5000/v1"
   vision_model = "gemma-4-31b-heretic-ara-batch-parallel6-q8-8k-cube"
   chat_model = "qwen38-27b-heretic-uncensored-q8-cube"
   vision_concurrency = 3
   vision_max_tokens = 1200
   vision_max_side = 1024

   [embed]
   url = "http://cube:5010/v1/embeddings"
   model = "nomic-embed-text-v1.5"
   dims = 768
   max_chars = 2000

   [asr]
   model = "large-v3"
   device = "cuda"
   compute_type = "int8_float32"

   [server]
   host = "127.0.0.1"
   port = 4001
   ```
5. `re2/config.py`: `load() -> Config` dataclass built from `config.toml`
   (use `tomllib`). Resolve relative paths against the repo root.
6. `re2/cli.py`: argparse with a `status` subcommand that prints the config
   and whether the corpus root and DB exist.

**Verify:** `uv run re2 status` prints the config. `git status` no longer
lists thousands of `node_modules` files as tracked-ignorable noise.

**Commit:** `Phase 0.1: uv project, config, gitignore`

### Task 0.2: Database module

**Do:** `re2/db.py` with `connect(path) -> sqlite3.Connection` that sets
`PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA synchronous=NORMAL`
and `row_factory = sqlite3.Row`, and `init_schema(conn)` that executes the
schema below (idempotent: `CREATE TABLE IF NOT EXISTS`). Store the DB at
`data/re2.sqlite`.

```sql
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
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE      -- normalized (Task 2.5)
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
```

Also add helper functions: `upsert_failure(conn, stage, media_id, doc_id, error)`
(increments attempts), `clear_failure(...)`, `pending_failures(stage) -> set of ids with attempts >= 3` (to skip).

**Verify:** `uv run python -c "from re2 import db; c=db.connect('data/re2.sqlite'); db.init_schema(c); print([r[0] for r in c.execute(\"select name from sqlite_master where type='table'\")])"` lists all tables. Write `tests/test_db.py` that inits an in-memory DB, inserts a doc, and checks `SELECT rowid FROM doc_fts WHERE doc_fts MATCH 'hello'` finds it.

**Commit:** `Phase 0.2: sqlite schema`

### Task 0.3: Fix the two security holes in the existing Node server (temporary)

The Node server is replaced in Phase 3, but it may keep being used until
then. Minimal edits to `server.js`:
1. Change `app.listen(port, ...)` to `app.listen(port, '127.0.0.1', ...)`.
2. In the `/images` middleware, after computing `fullPath`, compute
   `const real = await fs.realpath(fullPath)` inside the try and `return next()`
   unless `real.startsWith(IMAGE_PATHS[index] + '/')`.
3. In `/node-info/:hash`, wrap `nodeData.description` and each tag name in an
   `escapeHtml` helper (`&`, `<`, `>`, `"`, `'`).

**Verify:** `node server.js` starts; `curl -s -o /dev/null -w '%{http_code}' 'http://127.0.0.1:4001/images/../../.ssh/id_rsa'` prints 404.

**Commit:** `Phase 0.3: bind localhost, confine /images, escape node-info`

---

## Phase 1: Manifest

### Task 1.1: Scanner

**Do:** `re2/scan.py` with `scan(conn, cfg, limit=None, rehash=False)`:

1. Walk `corpus_root` with `os.scandir` recursively, `follow_symlinks=False`.
   Skip any directory named `.Trash-1000`.
2. For each entry that is a file or a symlink (skip directories):
   - `rel_path = path.relative_to(corpus_root)`.
   - If `entry.is_symlink()`: `target = os.readlink(path)`. If
     `Path(path).exists()` is False, this is a broken symlink: record it in
     `media_path` with `is_symlink=1, symlink_target=target, readable=0` and
     link it to a `media` row with `uid = 'missing:' + sha1(target)`,
     `status='SOURCE_OFFLINE'`, `kind` guessed from the target's extension
     (Task 1.1 step 4), `sha256=NULL`. Continue.
   - Otherwise `stat` it (following symlinks). If the (rel_path, size,
     mtime_ns) triple matches an existing `media_path` row and `rehash` is
     False, update `last_seen` and continue (no re-hash).
   - Compute sha256 by streaming in 1 MiB chunks.
   - Detect MIME with `magic.from_file(str(path), mime=True)` (python-magic).
     Fall back to extension if magic raises.
3. Classify `kind`: `image` if mime starts with `image/`; `video` if mime
   starts with `video/` or is `application/octet-stream` with a video-like
   extension; `pdf` if `application/pdf`; else `other`.
4. Exclusion rule (`status='EXCLUDED'`, `kind='other'`): any file whose name
   matches `cookies`, `.netrc`, `credentials`, `token`, `.env` (case
   insensitive), and any `.part`, `.tmp`, `.zip`, `.py`, `.yml`, `.tsv`,
   `.txt`, `.json` file. Excluded files are recorded so they are never
   rescanned, but their paths must never be returned by search.
5. Upsert `media` by `uid = sha256`. If the media row is new, set
   `status='READY'` for image/video/pdf, `'UNSUPPORTED'` for other. Insert or
   update the `media_path` row (`readable=1`). A hash seen under several paths
   gets several `media_path` rows and one `media` row.
6. For images: read `width, height` with Pillow (`Image.open(...).size`;
   catch errors and leave NULL). Videos get dimensions in Task 4.1.
7. After the walk, any `media_path` row whose `last_seen` is older than this
   scan's start time is marked `readable=0`. A `media` row whose paths are all
   unreadable and whose `thumb_path` is set becomes `THUMBNAIL_ONLY`; without a
   thumbnail it becomes `SOURCE_OFFLINE`. A `media` row that regains a readable
   path returns to `READY`. Store `kv.last_scan_at`.
8. `--limit N` stops after N newly hashed files (used for smoke tests).

CLI: `re2 scan [--limit N] [--rehash]`. Prints: files seen, new hashed,
skipped unchanged, broken symlinks, excluded, elapsed.

**Verify:** `uv run re2 scan --limit 200` completes. Then the full
`uv run re2 scan` (human runs it; it hashes ~57 GB over NFS, expect 15 to 60
minutes). Afterwards these queries in `sqlite3 data/re2.sqlite` must hold:
- `SELECT status, kind, COUNT(*) FROM media GROUP BY 1,2;` shows about 10,200
  READY images, 3,718 READY videos, 28 READY pdfs, and about 3,126
  SOURCE_OFFLINE rows.
- `SELECT COUNT(*) FROM media_path WHERE is_symlink=1 AND readable=0;` = 3126.
- `SELECT COUNT(*) FROM media_path WHERE rel_path LIKE '%.Trash-1000%';` = 0.
- Running `re2 scan` a second time finishes in under two minutes and reports
  zero new hashes.

Write `tests/test_scan.py` using a temp directory with two identical files,
one broken symlink, one `.Trash-1000` file and one `cookies.txt`; assert
one media row for the duplicates with two paths, one SOURCE_OFFLINE row, no
trash row, one EXCLUDED row.

**Commit:** `Phase 1.1: read-only manifest scanner`

### Task 1.2: Reconcile legacy captions, thumbnails and hash cache

**Do:** `re2/legacy.py` with `reconcile(conn, cfg)`:

1. Load `legacy/images_captioned.json` and `legacy/images_captioned_tagged.json`.
   For each sha256 key: if a `media` row with that `uid` exists, set
   `legacy_caption` (description) and `legacy_spicy` (from the tagged file if
   present). If no row exists, create one with `uid=sha256, sha256=sha256,
   kind='image', status='SOURCE_OFFLINE'` and the legacy fields. Also insert
   `media_tag` rows with `origin='legacy'` after normalizing tag names with
   `tags.normalize` (Task 2.5; implement `normalize` first as a tiny function:
   lowercase, strip, collapse whitespace, replace `_` and `-` with space).
2. Load `~/Downloads/redpills/output.json`. For each record: strip the prefix
   `/mnt/hanoi_data/storage/redpills/` from `path` to get `rel_path`. If a
   `media_path` row with that `rel_path` exists and its media row has
   `sha256 IS NULL` (a broken symlink recorded as `missing:`), **re-key** that
   media row: set `sha256`, and `uid = sha256`. If a media row with
   `uid=sha256` already exists (from step 1), instead re-point the
   `media_path` row at that existing media row and delete the orphaned
   `missing:` row. Set `legacy_caption` if still NULL.
3. Thumbnails: for every media row with a sha256, check
   `public/thumbnails/{h[0]}/{h[1]}/{h[2]}/{h[3:]}.jpg`; if it exists set
   `thumb_path`. Then re-apply the status rule from Task 1.1 step 7.
4. Insert one `doc` row per media with a legacy caption:
   `kind='legacy_caption', seq=0, text=legacy_caption, model='cogvlm2-legacy'`.
5. `hash_path_cache.json` is not needed once steps 1 to 3 are done; ignore it.

CLI: `re2 reconcile-legacy`.

**Verify:**
- `SELECT COUNT(*) FROM media WHERE legacy_caption IS NOT NULL;` is about 5,503.
- `SELECT status, COUNT(*) FROM media WHERE legacy_caption IS NOT NULL GROUP BY 1;`
  shows about 3,225 READY and about 1,919 THUMBNAIL_ONLY (all 1,919 orphaned
  captions have thumbnails), plus the never-tagged ones.
- `SELECT COUNT(*) FROM media WHERE uid LIKE 'missing:%';` is about
  3,126 - 1,261 = 1,865.
- `SELECT COUNT(*) FROM doc WHERE kind='legacy_caption';` is about 5,503.

**Commit:** `Phase 1.2: reconcile legacy captions, thumbnails, output.json`

### Task 1.3: Reconcile archive_metadata.json

**Do:** In `re2/legacy.py`, `reconcile_sources(conn, cfg)`: for each entry,
the files are `folder/screenshot`, `folder/photo` for each photo, and
`folder/video` for each video (verify the join rule on the first five
entries by checking the paths exist; if `folder` is empty the files are at
the corpus root). Look up each by `media_path.rel_path`; insert a `source`
row (`url`, `date`, `folder`, `restricted`, `role`). Count and log entries
whose files were not found.

CLI: `re2 reconcile-sources`.

**Verify:** `SELECT COUNT(DISTINCT media_id) FROM source;` is about 2,300.

**Commit:** `Phase 1.3: source URLs from archive_metadata.json`

### Task 1.4: Status report

**Do:** `re2 status` extended to print a table: media by status and kind;
paths readable/unreadable; docs by kind; tags; entities; sources; failures
by stage; embedding index size (0 for now); last scan time. This is the
command the human runs to see progress for the rest of the project.

**Verify:** Output matches the queries above.

**Commit:** `Phase 1.4: status report`

---

## Phase 2: Image extraction with the vision model

### Task 2.1: Thumbnails for every READY image

**Do:** `re2/thumbs.py`: for each `media` row with `kind='image'`,
`status='READY'`, `thumb_path IS NULL`: open the first readable path with
Pillow, `convert('RGB')`, resize to width 400 (keep aspect; do not upscale),
save JPEG quality 85 at the sharded path from Task 1.2 step 3, set
`thumb_path`. Animated GIFs: use the first frame. Record failures in
`failure(stage='thumb')`. Use a `ThreadPoolExecutor(8)`.

CLI: `re2 thumbs [--limit N]`.

**Verify:** after the full run, `SELECT COUNT(*) FROM media WHERE kind='image' AND status='READY' AND thumb_path IS NULL;` = 0 (minus failures).

**Commit:** `Phase 2.1: thumbnails for all readable images`

### Task 2.2: Vision model client

**Do:** `re2/vlm.py`:

1. `IMAGE_SCHEMA` (JSON schema, all fields required, `additionalProperties: false`):
   ```json
   {
     "type": "object",
     "properties": {
       "media_type": {"type": "string", "enum": ["screenshot_social_post", "screenshot_article", "screenshot_chat", "screenshot_other", "meme", "photo", "video_still", "chart_or_infographic", "document", "cartoon_or_illustration", "other"]},
       "description": {"type": "string"},
       "ocr_text": {"type": "string"},
       "subjects": {"type": "array", "items": {"type": "string"}},
       "actions": {"type": "string"},
       "meaning": {"type": "string"},
       "claims": {"type": "array", "items": {"type": "string"}},
       "intent": {"type": "string"},
       "humor": {"type": "string"},
       "entities": {
         "type": "object",
         "properties": {
           "people": {"type": "array", "items": {"type": "string"}},
           "organizations": {"type": "array", "items": {"type": "string"}},
           "places": {"type": "array", "items": {"type": "string"}},
           "events": {"type": "array", "items": {"type": "string"}},
           "platforms": {"type": "array", "items": {"type": "string"}}
         },
         "required": ["people", "organizations", "places", "events", "platforms"],
         "additionalProperties": false
       },
       "tags": {"type": "array", "items": {"type": "string"}},
       "date_hints": {"type": "array", "items": {"type": "string"}},
       "language": {"type": "string"},
       "sensitivity": {"type": "number"}
     },
     "required": ["media_type", "description", "ocr_text", "subjects", "actions", "meaning", "claims", "intent", "humor", "entities", "tags", "date_hints", "language", "sensitivity"],
     "additionalProperties": false
   }
   ```
2. `IMAGE_SYSTEM_PROMPT` (use verbatim):
   ```
   You are an archivist cataloguing a private research collection of political and social-media screenshots, memes, photographs, charts and documents so that they can be found again later by full-text and semantic search.

   Your only job is accuracy and completeness. Describe exactly what is present. The collection deliberately includes offensive, hateful, racist, sexual, violent, graphic, conspiratorial and extremist material; record all of it plainly and quote text verbatim. Never soften, paraphrase euphemistically, omit, or add warnings, disclaimers or moral commentary. Do not speculate beyond what is visible, but do state what the image is clearly trying to communicate.

   Fill every field:
   - media_type: the closest category.
   - description: a thorough visual description: layout, people (apparent age, sex, race or ethnicity when evident, clothing, expressions), objects, setting, on-screen UI elements, logos, watermarks, image quality.
   - ocr_text: ALL visible text verbatim in reading order, including usernames, handles, display names, timestamps, dates, view or like counts, captions, headlines, chyrons, watermarks and URLs. Preserve line breaks with \n. Empty string if there is no text.
   - subjects: the people, groups, objects or topics the image is about.
   - actions: what is happening or being done, including what the poster is doing (asking, mocking, accusing, reporting).
   - meaning: what the image communicates overall, including implied claims and subtext.
   - claims: each distinct factual or political claim made or implied, one per item, stated neutrally as a claim.
   - intent: why someone would create or share this (persuade, mock, document evidence, warn, recruit, provoke, inform, entertain).
   - humor: if there is a joke, meme format, sarcasm, irony or punchline, explain it and name the meme format if known; otherwise "none".
   - entities: proper nouns only, as full canonical names (e.g. "Kamala Harris", "Federal Bureau of Investigation", "X (Twitter)", "January 6 Capitol riot").
   - tags: 10 to 20 lowercase search tags: topics, people, organizations, places, events, platforms, content type, tone. Use spaces between words, no hashtags.
   - date_hints: any dates, times, or datable references visible or implied.
   - language: primary language of the text, ISO 639-1 code, or "none".
   - sensitivity: 0.0 for benign, 0.3 for political but civil, 0.6 for slurs, hate or violence, 0.8 for graphic injury or explicit sexual content, 1.0 for the most extreme material. This is a retrieval filter, not a judgement.
   ```
3. `KEYFRAME_SCHEMA`: same as `IMAGE_SCHEMA` minus `claims`, `intent`,
   `humor`, `date_hints`, `media_type` (fewer output tokens; used in Phase 4).
   `KEYFRAME_SYSTEM_PROMPT`: the same text with the removed fields deleted and
   the first sentence changed to say the image is a single frame from a video.
4. `prepare_image(path_or_bytes, max_side) -> bytes`: open with Pillow,
   convert RGB, downscale so the longer side is at most `max_side` (default
   1024 from config; never upscale), encode JPEG quality 90, return bytes.
5. `extract(client: httpx.Client, cfg, image_bytes, schema, system_prompt, user_text) -> dict`:
   POST `{base_url}/chat/completions` with
   ```json
   {"model": cfg.vision_model,
    "messages": [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": [{"type": "text", "text": user_text},
                                              {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "catalog", "schema": schema}},
    "temperature": 0.2, "max_tokens": cfg.vision_max_tokens}
   ```
   Parse `choices[0].message.content` with `json.loads`. On JSON error or
   HTTP error raise `VLMError(str)`. Retry HTTP 5xx and timeouts twice with a
   10 s sleep. Return the dict plus `usage` and elapsed seconds.

**Verify:** `uv run python -m re2.vlm public/thumbnails/4/a/3/3e5f768379478173ba32f38bd1f634a8339f8b8d297142f6905f333f807ef.jpg`
(add a `__main__` that runs `extract` on one file and pretty-prints) returns a
dict with all keys and non-empty `ocr_text`. This test was already run
successfully on 2026-09-14 against the `qwen38-flash-next` model with a
near-identical prompt, so failures are configuration problems, not model
problems.

**Commit:** `Phase 2.2: vision model client, prompt and schema`

### Task 2.3: Extract every image

**Do:** `re2/extract_images.py`, `run(conn, cfg, limit=None, include_thumbnail_only=True)`:

1. Select media rows with `kind='image'` and status in (`READY`,
   `THUMBNAIL_ONLY`) that have no `doc` row with `kind='caption'` and are not
   in `pending_failures('vlm_image')`. Order: READY first, then
   THUMBNAIL_ONLY. For READY use the first readable path; for THUMBNAIL_ONLY
   use `thumb_path` and set `low_res=1` in `media_meta`. Also set `low_res=1`
   when the source image's longer side is under 500 px.
2. Process with `ThreadPoolExecutor(cfg.vision_concurrency)`; each worker
   has its own `httpx.Client`. Per item: `prepare_image`, `extract` with
   user text `Catalogue this image.` (append `The image is a low-resolution thumbnail; transcribe what is legible.` when `low_res`).
3. On success, in one transaction:
   - `doc(kind='caption', seq=0, text=<caption text>, json=<full dict>, model=cfg.vision_model)`
     where caption text is built as:
     `f"{d['description']}\n\nSubjects: {', '.join(subjects)}\nActions: {actions}\nMeaning: {meaning}\nClaims: {'; '.join(claims)}\nIntent: {intent}\nHumor: {humor}"`.
   - `doc(kind='ocr', seq=0, text=ocr_text)` only if `ocr_text.strip()` is non-empty.
   - `media_meta` upsert: `sensitivity`, `media_type`, `language`, `low_res`.
   - `media_tag` rows with `origin='vlm'`, `weight=1.0`, names through `tags.normalize` (Task 2.5).
   - `entity` / `media_entity` rows for each of the five entity lists, type
     singular (`person`, `organization`, `place`, `event`, `platform`), name
     stripped; skip empty strings.
   - `clear_failure('vlm_image', media_id)`.
   On `VLMError`, `upsert_failure`.
4. Progress bar with tqdm; every 50 items log the running rate
   (images per minute) and the average `completion_tokens`.

CLI: `re2 extract-images [--limit N] [--no-thumbnail-only]`.

**Verify:** `uv run re2 extract-images --limit 5` succeeds and
`SELECT m.id, substr(d.text,1,80) FROM doc d JOIN media m ON m.id=d.media_id WHERE d.kind='caption' LIMIT 5;`
shows sensible captions. Print the measured rate. Then hand the full run to
the human: `nohup uv run re2 extract-images > data/extract_images.log 2>&1 &`.
Expected volume: about 10,200 READY images plus about 1,900 thumbnail-only
images. Expected rate: to be measured; if slower than 4 images per minute
the human may add a larger-context entry to `models-router.ini` on cube
(see note at the end of this file).

**Commit:** `Phase 2.3: image extraction pipeline`

### Task 2.4: Legacy captions for items with no image at all

Nothing to build. Media rows that are SOURCE_OFFLINE with no thumbnail keep
only their `legacy_caption` doc, which is already searchable through
`doc_fts`. Confirm with:
`SELECT COUNT(*) FROM media m WHERE status='SOURCE_OFFLINE' AND EXISTS (SELECT 1 FROM doc WHERE media_id=m.id);`
(about 1,254 expected before the hanoi volume returns, fewer after).

### Task 2.5: Tags, entities, PMI co-occurrence

**Do:** `re2/tags.py`:

1. `normalize(name) -> str`: lowercase; strip; replace `_` and `-` with
   space; collapse whitespace; strip leading `#`; return `''` for anything
   shorter than 2 chars (callers skip empty). Apply a small synonym map
   loaded from `tag_synonyms.toml` in the repo root (start it with:
   `"covid 19" = "covid-19"`? No: keep it hyphen-free: `"coronavirus" = "covid 19"`,
   `"antisemitism" = "anti semitism"`, `"judiasm" = "judaism"`,
   `"voter fraud" = "election fraud"`? No, leave that; start with the three
   obvious ones and let the human extend it).
2. `pmi(conn, min_items=5, min_pair=3) -> dict[str, dict[str, float]]`:
   over `media_tag` rows with `origin='vlm'` (fall back to `legacy` for
   media without vlm tags), compute item counts per tag, pair counts, and
   `PMI = log2( (pair/N) / ((a/N)*(b/N)) )`. Keep pairs with `pair >= min_pair`
   and `PMI > 0`. Normalize weights to 0..1 by dividing by the max PMI. Return
   the symmetric nested dict.
3. `export_legacy_json(conn, cfg)`: regenerate, from the DB, the three files
   the existing pages read, in their existing formats:
   - `data/images_captioned_tagged.json`: `{sha256: {filename, description, tags: {name: weight}, spicy}}` for every media with a caption or legacy caption (`description` = caption doc text or legacy caption; `spicy` = `media_meta.sensitivity` or `legacy_spicy`; `filename` = basename of first path).
   - `data/tags_with_sizes.json`: `{tag: {itemIds: [sha256...], normalizedSize}}` for tags with at least 5 items, `normalizedSize` linearly scaled 0..1 like `preprocess_data.js` did.
   - `data/tag_pairs_with_weights.json`: the PMI dict.
   - `data/adjusted_data.json`: same as the tagged file plus `relativePath`.
   Back up any existing file to `data/backup_<timestamp>/` first.

CLI: `re2 export-legacy-json`.

**Verify:** run it; start the old Node server; open `http://127.0.0.1:4001/`
and `/matrix`; both render with the new tags. `tests/test_tags.py` covers
`normalize` (case collisions like `USA Today` / `usa today` merge) and `pmi` on
a tiny synthetic table.

**Commit:** `Phase 2.5: tag normalization, PMI, legacy JSON export`

---

## Phase 3: Search API and search-first UI

### Task 3.1: Embedding server on cube (one-time setup; the human runs it)

Provide the human with `scripts/start_embed_server_on_cube.sh` containing:
```bash
#!/usr/bin/env bash
# Run ON cube. Serves nomic-embed-text-v1.5 on port 5010. Uses ~1 GB of one GPU.
cd ~/ML/llama.cpp
nohup ./build/bin/llama-server --embedding \
  -m /home/offipso/ML/models/nomic-embed/nomic-embed-text-v1.5.Q8_0.gguf \
  --host 0.0.0.0 --port 5010 -ngl 99 --device CUDA5 -c 8192 -b 8192 -ub 8192 \
  --pooling mean > ~/logs/embed-server.log 2>&1 &
```
**Verify (from this machine):**
`curl -s http://cube:5010/v1/embeddings -H 'Content-Type: application/json' -d '{"input":"search_query: test"}' | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data'][0]['embedding']))"`
prints `768`. If `--device` is rejected by that llama.cpp build, drop the
flag and set `CUDA_VISIBLE_DEVICES=5` before the command instead.

**Commit:** `Phase 3.1: embedding server script`

### Task 3.2: Embedding index

**Do:** `re2/embed.py`:
1. `embed_texts(texts: list[str], prefix) -> np.ndarray` (float32, L2
   normalized, shape N×768): POST `{"input": [prefix + t[:max_chars] ...]}` in
   batches of 32; `httpx` timeout 60 s; retry twice.
2. Index files: `data/index/docs.npy` (float32 N×768) and
   `data/index/doc_ids.npy` (int64 N). Load lazily into module-level globals;
   `reload()` re-reads them.
3. `build(conn, cfg, limit=None)`: select `doc.id, doc.text` for docs whose id
   is not in `doc_ids.npy` and whose kind is not `legacy_caption` when the
   same media also has a `caption` doc (avoid double-indexing). Embed with
   prefix `search_document: `. Append to the two `.npy` files (write to temp
   files and `os.replace`).
4. `query(text, k=50, kinds=None) -> list[(doc_id, score)]`: embed with
   `search_query: `, `scores = docs @ q`, `argpartition` top k, return ids and
   cosine scores.

CLI: `re2 embed [--limit N] [--rebuild]`.

**Verify:** `uv run re2 embed --limit 100`; then in Python
`embed.query("Ukraine long range missiles")` returns doc ids whose text
mentions Ukraine (check with a SELECT). Full run is fast enough for the agent
to run itself (tens of thousands of short texts; expect under 30 minutes).

**Commit:** `Phase 3.2: dense embedding index`

### Task 3.3: Search module

**Do:** `re2/search.py`:

1. `lexical(conn, q, k=50, kinds=None) -> list[(doc_id, bm25)]`: build an
   FTS5 query from the user string: split on whitespace, drop tokens shorter
   than 2 chars, quote each token (`"token"`), join with a space (implicit
   AND); if that returns fewer than 5 results, retry with `OR`. Use
   `SELECT rowid, bm25(doc_fts) AS s FROM doc_fts WHERE doc_fts MATCH ? ORDER BY s LIMIT ?`
   (lower bm25 is better; negate for scoring). Support phrase queries: if the
   user string contains double quotes, pass those spans through as phrases.
2. `dense(q, k=50, kinds=None)`: `embed.query`.
3. `fuse(lists, k=60) -> list[(doc_id, score)]`: reciprocal rank fusion,
   `score = Σ 1/(k + rank)`.
4. `search(conn, q, filters, limit=40, offset=0) -> SearchResult`: run lexical
   and dense, fuse, then join to `doc` and `media`, apply filters (`kind`
   in image/video/pdf, `status`, `folder` prefix of rel_path, `sensitivity`
   max, `entity` name, `tag` name, `date` range from `source.date`), **exclude
   EXCLUDED media**, group by `media_id` keeping the best doc per media as the
   `hit` (with `kind`, `start_s`, `end_s`, `page`, and a 300-char snippet
   around the first matched token), and return media ordered by best score.
   Also return facet counts for `kind`, `status`, top 20 `folder`, top 20
   `entity`, top 20 `tag` over the top 500 fused results.

Also `media_url(media) -> str | None`: `/media/{id}/file` if the media has a
readable path else None; `thumb_url(media)`.

**Verify:** `tests/test_search.py`: in-memory DB with three docs; lexical
finds exact token; fuse ranks a doc present in both lists first.

**Commit:** `Phase 3.3: hybrid search`

### Task 3.4: FastAPI server replacing server.js

**Do:** `re2/api.py` with `app = FastAPI()`, run by `re2 serve` (uvicorn,
host/port from config). Routes:

| Route | Behaviour |
|---|---|
| `GET /` | serves `public/search.html` |
| `GET /graph`, `/matrix`, `/spicy`, `/concept-map` | serve the existing `public/index.html`, `matrix.html`, `spicy.html`, `concept-map.html` |
| `GET /static/...` and the existing relative asset paths (`/script.js`, `/styles.css`, `/spicyCore.js`, ...) | `StaticFiles(directory="public")` mounted at `/` after the API routes; also mount `node_modules/three` at `/node_modules/three` |
| `GET /thumbnails/...` | StaticFiles over `public/thumbnails` |
| `GET /api/search?q=&kind=&status=&folder=&max_sensitivity=&entity=&tag=&date_from=&date_to=&limit=&offset=` | `search.search`; JSON `{results: [{media_id, kind, status, thumb_url, media_url, title, folder, hit: {doc_id, kind, snippet, start_s, end_s, page}, score, sensitivity, source_url}], facets: {...}, total}` |
| `GET /api/media/{id}` | full media record: paths, status, source rows, meta, tags, entities, docs (caption json, ocr, transcript windows, keyframes, pdf pages), similar items (Task 3.6) |
| `GET /media/{id}/file` | `FileResponse` of the first readable path. **Confinement:** `real = os.path.realpath(path)`; require `real.startswith(os.path.realpath(corpus_root) + os.sep)`; else 404. Support HTTP Range requests (FastAPI `FileResponse` does; verify with a video seek). |
| `GET /media/{id}/keyframe/{seq}` | FileResponse of `data/keyframes/...` |
| `GET /api/status` | the `re2 status` counts as JSON |
| `GET /data`, `/tags`, `/tag_pairs_with_weights` | the legacy JSON files (unchanged behaviour) |
| `GET /hash-to-path/{sha256}` | `{"url": "/media/{id}/file"}` or 404 (relative URL, no hostname) |
| `GET /images/{path:path}` | legacy route used by old pages: resolve under corpus root with the same realpath confinement |
| `GET /node-info/{sha256}` | Jinja2 template `templates/node_info.html` with autoescape on: thumbnail, description (caption doc), sensitivity, tags, paths, status |

Bind to `127.0.0.1` only. Delete `server.js` and `hashMapper.js`; keep
`package.json` because `node_modules/three` is still served.

**Verify:** `uv run re2 serve` then:
- `curl -s 'http://127.0.0.1:4001/api/search?q=ukraine+missiles' | python3 -m json.tool | head -40` returns results with snippets.
- `curl -s -o /dev/null -w '%{http_code}\n' 'http://127.0.0.1:4001/images/../../.ssh/id_rsa'` → 404.
- `curl -s -o /dev/null -w '%{http_code}\n' -r 0-100 http://127.0.0.1:4001/media/<some video id>/file` → 206.
- `/graph`, `/matrix`, `/spicy` render in a browser exactly as before.

**Commit:** `Phase 3.4: FastAPI server replaces server.js`

### Task 3.5: Search-first UI

**Do:** `public/search.html`, `public/search.js`, `public/search.css`. Plain
HTML, CSS and vanilla JavaScript; no build step; no framework. Layout:

- Top bar: search input (debounced 300 ms, also on Enter), kind toggle
  (all / images / videos / pdfs), max-sensitivity slider (default 1.0),
  status toggle (include offline items: on by default), link row to
  `/graph`, `/matrix`, `/spicy`.
- Left column: facets from the response (folders, entities, tags, status).
  Clicking one adds it as a filter chip; chips can be removed.
- Centre: results grid of cards: thumbnail (or a placeholder with the kind
  icon), one-line title (first 80 chars of description, or filename),
  snippet with the matched terms wrapped in `<mark>` (escape everything else
  with a `escapeHtml` function; never assign untrusted strings to
  `innerHTML` without escaping), kind badge, status badge (`offline` in red,
  `thumbnail only` in orange), sensitivity number, folder. Click opens the
  detail pane. "Load more" button uses `offset`.
- Right pane (detail): large preview: `<img>` for images; `<video controls>`
  for videos, seeking to `hit.start_s` when present; PDF opened via
  `<a target=_blank>`. Below: source URL link, all paths, status, entities
  as chips (click = search), tags as chips, the full caption fields
  (description, ocr text in a `<pre>`, meaning, claims, intent, humor), the
  transcript (Phase 4) as a list of timestamped lines that seek the video on
  click, keyframes strip (Phase 4), "Similar" grid (Task 3.6). Every URL
  displayed must be the confined `/media/{id}/file` URL, never a filesystem
  path shown as a link (filesystem paths are shown as text only).
- Chat panel placeholder (filled in Phase 6).
- Keep the visual style consistent with `public/styles.css` (dark theme).
- State (query, filters) is written to the URL hash so a search is
  shareable and the back button works.

**Verify:** manual: search "Trudeau Ukraine weapons", see results, open a
video result, confirm playback and seek. Search for a term with a `<script>`
in it and confirm it is rendered as text.

**Commit:** `Phase 3.5: search-first UI`

### Task 3.6: "More like this"

**Do:** In `search.py`, `similar(media_id, k=12)`: take the media's caption
doc embedding row from the index (look up its `doc_id` in `doc_ids.npy`),
compute cosine against all rows, exclude the same media, group by media, top
k. Expose in `/api/media/{id}` as `similar`. UI renders it in the detail pane.

**Commit:** `Phase 3.6: similar items`

---

## Phase 4: Video

### Task 4.1: Probe and keyframes

**Do:** `re2/video.py`:
1. `probe(path) -> dict`: run `ffprobe -v error -print_format json -show_format -show_streams` via `subprocess.run([...], capture_output=True, check=True)` (argument list, never a shell string). Extract `duration`, `width`, `height`, `has_audio` (any stream with `codec_type == 'audio'`). Store on `media`.
2. `keyframes(media, path, out_dir) -> list[(seq, t_s, file)]`: run
   `ffmpeg -v error -y -i <path> -vf "fps=1/15,scale='min(1024,iw)':-2" -frames:v 8 -q:v 3 <out_dir>/<media_id>_%02d.jpg`
   under `data/keyframes/<id[:2]>/`. Also always extract a frame at 1.0 s if
   the video is shorter than 15 s (`-ss 1 -frames:v 1`). Compute
   `imagehash.phash` for each; drop a frame whose hamming distance to any
   kept frame is ≤ 6. Insert `keyframe` rows (`t_s = (n-1)*15` for the fps
   filter output, 1.0 for the short-video frame).
3. Process media with `kind='video'`, `status='READY'` that have no
   `keyframe` rows and `duration IS NULL`, with `ThreadPoolExecutor(4)`.

CLI: `re2 video-probe [--limit N]`.

**Verify:** `--limit 5`, then check `data/keyframes/` has JPEGs and
`SELECT media_id, COUNT(*) FROM keyframe GROUP BY 1 LIMIT 5;`. Full run is
local CPU/NFS only; the agent may run it in the background (expect an hour
or two for 3,718 videos).

**Commit:** `Phase 4.1: ffprobe and keyframes`

### Task 4.2: ASR with faster-whisper on the local GPU

**Do:** `re2/asr.py`:
1. At import time, before importing `faster_whisper`, preload the CUDA
   libraries shipped by the nvidia pip packages (ctranslate2 needs them and
   cannot find them via LD_LIBRARY_PATH set after process start):
   ```python
   import ctypes, importlib.util, pathlib
   for pkg in ("nvidia.cublas", "nvidia.cudnn"):
       spec = importlib.util.find_spec(pkg)
       for so in sorted(pathlib.Path(spec.submodule_search_locations[0], "lib").glob("*.so*")):
           try: ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
           except OSError: pass
   from faster_whisper import WhisperModel
   ```
2. `transcribe(model, media, path) -> list[segments]`: extract audio with
   `ffmpeg -v error -y -i <path> -vn -ac 1 -ar 16000 <data/audio/<id>.wav>`,
   then `model.transcribe(wav, beam_size=1, vad_filter=True)`; iterate the
   generator; delete the wav afterwards. Insert `transcript_segment` rows.
3. Build transcript windows: group consecutive segments into windows of at
   most 60 seconds or 600 characters; insert `doc(kind='transcript', seq=n,
   start_s, end_s, text=window text, model='faster-whisper-large-v3')`. If
   the video has no audio stream or the transcript is empty, insert a
   `kv`-style marker: a `doc` row with `kind='transcript'`, `seq=0`,
   `text=''`? No: empty text would pollute search. Instead record success
   in `failure` with `error='no_speech'` and `attempts=99` so it is skipped.
4. Load the model once per process. Process videos with `has_audio=1` that
   have no transcript doc and are not skipped. Single-threaded (one GPU).

CLI: `re2 asr [--limit N]`.

**Verify:** `uv run re2 asr --limit 3`; check
`SELECT media_id, start_s, end_s, substr(text,1,60) FROM transcript_segment LIMIT 5;`.
Measured on 2026-09-14: 19x realtime on the 1080 Ti with `int8_float32`, so
the full 107.6 hours takes about 6 hours. The human runs
`nohup uv run re2 asr > data/asr.log 2>&1 &`.

**Commit:** `Phase 4.2: faster-whisper transcription`

### Task 4.3: Keyframe extraction with the vision model

**Do:** In `extract_images.py`, add `run_keyframes(conn, cfg, limit)`: for
every `keyframe` row without a `doc(kind='keyframe', seq=keyframe.seq)`,
call `vlm.extract` with `KEYFRAME_SCHEMA`, user text
`Catalogue this video frame taken at {t_s:.0f} seconds.`, `max_tokens` 700.
Insert `doc(kind='keyframe', seq, start_s=t_s, end_s=t_s, text=<description + "\nText: " + ocr_text>, json)`,
plus entities and tags (`origin='vlm'`). Failures to `failure(stage='vlm_keyframe', doc_id=keyframe.id)`.

CLI: `re2 extract-keyframes [--limit N]`.

**Verify:** `--limit 5`. Full run is a human-started background job (expect
roughly 15,000 to 20,000 frames; several days at 4 to 6 frames per minute; it
can be interrupted and resumed at any time).

**Commit:** `Phase 4.3: keyframe extraction`

### Task 4.4: Video in search and UI

**Do:** transcript and keyframe docs are already in `doc_fts`; run
`re2 embed` to add them to the dense index. In the UI detail pane, render
transcript windows with timestamps that seek the `<video>` element, and the
keyframe strip with `t_s` labels that seek on click. In result cards, when
the hit is a transcript window, show `mm:ss–mm:ss` in the snippet header,
and the detail pane opens the video at `hit.start_s`.

**Verify:** search a phrase you heard in a video; result opens at the right
second.

**Commit:** `Phase 4.4: video segments in search and UI`

---

## Phase 5: PDFs

### Task 5.1

**Do:** `re2/pdf.py`: for each `kind='pdf'` READY media without `pdf_page`
docs: open with `pymupdf`; for each page, `text = page.get_text()`. If
`len(text.strip()) >= 50`, insert `doc(kind='pdf_page', seq=page_no, page=page_no, text)`.
Otherwise render the page at 150 dpi (`page.get_pixmap(dpi=150)`) to JPEG
bytes and run `vlm.extract` with `IMAGE_SCHEMA`; store the `ocr_text` as the
doc text and the full json. Also create a thumbnail from page 1 into the
sharded thumbnail path and set `thumb_path`. Cap at 300 pages per PDF.

CLI: `re2 pdfs [--limit N]`. The UI opens `/media/{id}/file#page=N` for PDF hits.

**Verify:** 28 PDFs; `SELECT media_id, COUNT(*) FROM doc WHERE kind='pdf_page' GROUP BY 1;`. Run `re2 embed`.

**Commit:** `Phase 5: PDF pages`

---

## Phase 6: Chat with enforced citations

### Task 6.1: Chat pipeline

**Do:** `re2/chat.py`, `ask(conn, cfg, question, history=[]) -> Answer`:

1. **Rewrite.** Call the chat model with `response_format` json_schema
   `{"queries": [str], "entities": [str], "kind": "image|video|pdf|any", "max_sensitivity": number}`
   and system prompt:
   ```
   You turn a vague question about a personal archive of political screenshots, memes and video clips into search queries. Output 3 to 6 short keyword queries covering synonyms, related people, organizations, places and events. Output entity names you are confident are relevant. Do not answer the question.
   ```
   `temperature 0.3`, `max_tokens 300`. On failure fall back to
   `queries=[question]`.
2. **Retrieve.** For each query run `search.search` (lexical + dense fused)
   with the filters; also run one dense query with the original question.
   Fuse all result lists with RRF, keep the top 12 media, and for each media
   the best hit doc plus, for videos, up to two adjacent transcript windows.
3. **Context.** Number the evidence `[C1]`..`[Cn]`. For each: media id, kind,
   status (`available` / `thumbnail only` / `offline`), folder, filename,
   source URL and date if known, and for transcript hits the `mm:ss` range,
   for pdf pages the page number; then the doc text truncated to 1,200 chars.
4. **Answer.** System prompt:
   ```
   You are the archivist for a private research archive. Answer the user's question using ONLY the evidence items provided. Every factual sentence must end with one or more citation markers like [C3]. Quote or closely paraphrase the evidence; do not add outside knowledge. If the evidence does not answer the question, say what was found and what is missing. Mention when an item is offline or thumbnail-only. Do not moralize or add warnings about the content.
   ```
   User message: the question, then `Evidence:` and the context block.
   `temperature 0.3`, `max_tokens 900`.
5. **Validate.** Regex `\[C(\d+)\]`; any marker not in `1..n` is removed
   from the text. Build `citations = [{label, media_id, doc_id, kind, status,
   title, media_url, thumb_url, start_s, end_s, page, source_url}]` for the
   markers that remain. Sentences with no marker are kept but the response
   includes `uncited_sentences` count so the UI can style them.
6. **Log.** Insert a `chat_log` row.

`POST /api/ask` with `{question, history, filters}` → `{answer, citations, queries, latency_ms}`.

**Verify:** `curl -s -X POST http://127.0.0.1:4001/api/ask -H 'Content-Type: application/json' -d '{"question":"What was the thing about Trudeau and Ukraine using long-range weapons?"}'`
returns an answer citing the Disclose.tv clip in the `ukraine` folder with a
timestamp. `tests/test_chat.py` covers the citation validator (unknown
markers removed, known ones mapped).

**Commit:** `Phase 6.1: grounded chat with validated citations`

### Task 6.2: Chat UI

**Do:** In `search.html`, a chat panel (toggle button): message list, input.
Answers render with `[Cn]` markers replaced by clickable chips that open the
cited item in the detail pane (seeking video when `start_s` is set). Show the
generated queries in small text under the answer. All text escaped.

**Commit:** `Phase 6.2: chat UI`

---

## Phase 7: Optional, after everything above works

Only start these when the human asks. Listed so the design leaves room:

1. Image-to-image similarity with a SigLIP/CLIP model served on cube;
   second index `data/index/image.npy` keyed by media id.
2. A cross-encoder reranker over the top 50 fused hits.
3. Topic clustering of caption embeddings (k-means, LLM-named clusters) to
   replace `concept-map.html`.
4. QLoRA of the chat model from `chat_log` rows the human has rated; the log
   table already stores everything needed (question, rewrite, retrieved ids,
   answer, citations).
5. Instanced rendering for `spicy.html` if it becomes slow.

---

## Notes for the human (not for the coding agent)

- **Model swapping on cube.** The router loads one model at a time. Image
  extraction (gemma vision), chat (qwen38-27b) and your own opencode session
  all compete. Run the long extraction batches when you are not using cube
  for something else. The embedding server on port 5010 is a separate process
  and is unaffected.
- **Faster captioning.** If the measured rate in Task 2.3 is too slow, append
  this entry to `~/ML/llama.cpp/models-router.ini` on cube, restart
  llama-server, and set `vision_model` and `vision_concurrency = 4` in
  `config.toml`:
  ```ini
  [gemma-4-31b-heretic-ara-vision-parallel4-q8-16k-cube]
  model = /home/offipso/ML/models/gemma/gemma-4-31b-it-heretic-ara-Q8_0.gguf
  mmproj = /home/offipso/ML/models/gemma/mmproj-F16.gguf
  chat-template-file = /home/offipso/ML/llama.cpp/models/templates/google-gemma-4-31B-it.jinja
  cache-type-k = q4_0
  cache-type-v = q4_0
  kv-unified = true
  ctx-size = 16384
  parallel = 4
  threads = 32
  ```
- **Hanoi volume.** If you bring it back online and copy the missing files
  into storeroom4 (a write you make yourself; the pipeline never writes
  there), the next `re2 scan` hashes them, matches 1,261 of them to their
  existing legacy captions by sha256, and flips them to READY. They will then
  be picked up by `re2 thumbs` and `re2 extract-images` automatically.
- **Order of long jobs:** `scan` → `reconcile-legacy` → `reconcile-sources`
  → `thumbs` → `extract-images` (days) → `embed` → serve and use search →
  `video-probe` → `asr` (about 6 h, local) → `extract-keyframes` (days) →
  `embed` → `pdfs` → `embed` → chat.
