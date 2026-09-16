# RE2 Generic Plan: any collection, run from one command

Written 2026-09-16. Builds on `RE2_CODING_PLAN.md` (phases 1–8 are done). When
the two disagree about something this plan covers, this plan wins.

## What we are building

Today the explorer is tuned to one archive: prompts say "political and
social-media material", the corpus is one folder in `config.toml`, services are
edited in files, and long jobs are shell scripts the owner runs by hand.

After this plan, anyone with:

- an OpenAI-compatible **vision** model (llama.cpp, LM Studio, vLLM, OpenAI…),
- an OpenAI-compatible **chat** model (can be the same one),
- a **speech-to-text** service: local faster-whisper or an OpenAI-compatible
  `/audio/transcriptions` API,

runs one command, `re2 up`, and does everything else in the browser:

1. adds dataset folders ("roots") and describes the collection (a "profile"),
2. points the app at their model servers and sees whether each one works,
3. starts, stops, schedules and watches every processing stage on a
   **Pipeline** page, with sizes, rates, ETAs, failures and energy use,
4. searches, reads, maps, asks and argues exactly as today.

The owner's archive becomes the first profile and keeps working throughout.
Every phase ends with the app in a working state.

### Out of scope

- More than one collection per install. A second collection is a second
  install with its own `data/` directory.
- Docker images. Installation is `uv tool install` from the repo.
- Multi-user accounts. There is one password for the whole install.
- Moving LLM inference onto this machine. Models stay on HTTP services.

---

## 0. Ground rules (read before every session)

1. **Never write under a root.** Roots are the user's dataset folders and may
   be network mounts. Open files there with mode `rb` only. Never create,
   rename, move, delete or symlink anything under a root. Everything derived
   goes under `data/`.
2. **Python 3.12 via `uv`.** Every command is `uv run …` from the repo root.
3. **The live database is in use.** The owner runs captioning and ASR against
   `data/re2.sqlite` for days at a time. Schema changes are additive
   (`CREATE TABLE IF NOT EXISTS`, `db.ADDED_COLUMNS`) except the one table
   rebuild in Task 1.2, which runs only when no stage is running (it checks).
4. **Test against a copy for anything that writes.** For manual checks of
   migrations, copy the database first:
   `cp data/re2.sqlite /tmp/re2-copy.sqlite` and point a scratch
   `config.local.toml` (`[paths] data_dir`) at a scratch directory holding it.
5. **Never kill processes by pattern.** No `pkill -f`. Kill only PIDs you
   started. The owner's server runs on port 4001; use 4020–4029 for your own.
6. **Verify every task.** Each task lists tests and checks. `uv run pytest -q`
   must pass before every commit. For dashboard changes, also run
   `tools/browser_check.mjs` (Task 0.1) and read the screenshots it writes.
7. **Commit after each task** with the message given, ending with the
   session trailer line the harness asks for.
8. **Keep the house style.** Single-file vanilla HTML/CSS/JS dashboard, the
   riso two-ink palette (`--ink`, `--blue`, `--hot` for the cursor, `--focus`,
   `--stock-*`), vim keys, everything reachable by keyboard and by touch,
   sparse comments that say why. Match the code around you.

### Order of work

Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7. Do not start a phase before the one
before it is committed and verified.

---

## Phase 0: Foundations

### Task 0.1: A browser check tool in the repo

**Do:** add `tools/browser_check.mjs`, a copy of the CDP driver used during
development, so every later task can check the dashboard the same way.

- Usage: `node tools/browser_check.mjs <url> <out-dir> <viewport> [steps…]`
  where `<viewport>` is `desk` (1500×1000, mouse), `fold` (412×900, touch,
  mobile) or `open` (884×1000, touch, mobile).
- Steps: `js:<expression>` (prints the JSON result), `wait:<ms>`,
  `tap:<x>,<y>`, `key:<key>` (dispatches a keydown on `document`),
  `shot:<name>` (writes `<out-dir>/<name>.png`).
- It launches `chromium --headless=new` on port 9390 with a profile directory
  under `<out-dir>/profile`, sets device metrics and touch emulation for the
  viewport, navigates, waits 3.5 s, runs the steps, always writes
  `<out-dir>/final.png`, prints `errors: [...]` with any uncaught page
  exceptions, and exits 1 if there were any.

**Verify:** `node tools/browser_check.mjs http://127.0.0.1:4001/#/search
/tmp/bc desk 'js:S.mode'` prints `"search"` and `errors: []`.

**Commit:** `Generic 0.1: browser check tool`

### Task 0.2: A settings table

**Do:** everything the user configures in the browser lives in the database.

1. `re2/db.py` SCHEMA:
   ```sql
   CREATE TABLE IF NOT EXISTS setting (
     key TEXT PRIMARY KEY,               -- e.g. 'service.chat', 'profile', 'server.password_hash'
     value TEXT NOT NULL,                -- JSON
     updated_at TEXT NOT NULL DEFAULT (datetime('now'))
   );
   ```
2. `re2/db.py`: `setting_get(conn, key, default=None)` (JSON-decoded) and
   `setting_set(conn, key, value)` (JSON-encoded, upsert, commits).
3. `re2/config.py`: keep `config.toml` and `config.local.toml` as the
   **bootstrap** (they say where `data/` is and which host and port to
   listen on). Add `apply_db_settings(cfg, conn)`: for every service in
   `SERVICE_FIELDS`, if `setting` has `service.<name>`, copy its keys onto
   `cfg`. Precedence is database, then `config.local.toml`, then `config.toml`.
4. `config.save_services()` writes `service.<name>` rows instead of
   `config.local.toml`. On first run (no `service.*` rows yet), import the
   current effective values into the table once, so nothing changes for the
   owner.
5. Call `apply_db_settings` in `api.create_app` and at the start of every
   CLI command that uses services (`cli.main`, after `config.load()`).

**Verify:** `tests/test_settings.py`:
- with an empty `setting` table, `apply_db_settings` leaves `cfg` unchanged;
- after `setting_set(conn, "service.chat", {"model": "m2"})`, `cfg.chat_model == "m2"`;
- `POST /api/settings` (existing test in `tests/test_ask.py`) now writes a row
  and leaves `config.local.toml` untouched (update that test).

**Commit:** `Generic 0.2: settings live in the database`

---

## Phase 1: Roots (dataset folders)

### Task 1.1: The root table

**Do:** `re2/db.py` SCHEMA:

```sql
CREATE TABLE IF NOT EXISTS root (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,            -- shown in the UI, e.g. 'redpills'
  path TEXT NOT NULL UNIQUE,            -- absolute
  include_glob TEXT NOT NULL DEFAULT '*',   -- fnmatch patterns, ';'-separated
  exclude_glob TEXT NOT NULL DEFAULT '',
  importers TEXT NOT NULL DEFAULT '',   -- ';'-separated importer names (Phase 3)
  enabled INTEGER NOT NULL DEFAULT 1,
  added_at TEXT NOT NULL DEFAULT (datetime('now')),
  last_scan_at TEXT
);
```

Add `("media_path", "root_id", "INTEGER")` to `ADDED_COLUMNS`.

On `init_schema`, if `root` is empty and `cfg.corpus_root` exists, insert
`(1, <basename of corpus_root>, <corpus_root>)` and run
`UPDATE media_path SET root_id=1 WHERE root_id IS NULL`. `init_schema` needs
`cfg` for this: add an optional `cfg` parameter and pass it from `open_db`.

**Verify:** on a copy of the live database, `re2 status` shows the same
counts as before and `SELECT id, path FROM root` shows one row with
`~/Downloads/redpills`; `SELECT COUNT(*) FROM media_path WHERE root_id IS NULL`
is 0.

**Commit:** `Generic 1.1: root table, the corpus becomes root 1`

### Task 1.2: Paths are unique per root

`media_path.rel_path` is `UNIQUE` on its own, so two roots cannot both hold
`a.png`. SQLite cannot change a constraint in place.

**Do:** `re2/db.py`, `migrate_media_path_unique(conn)`:

1. If `kv.schema_media_path == '2'`, return.
2. If any `work_claim` row has `expires_at > now`, raise
   `MigrationBlocked("stages are running")`.
3. In one transaction:
   ```sql
   CREATE TABLE media_path_new (
     id INTEGER PRIMARY KEY,
     media_id INTEGER NOT NULL REFERENCES media(id),
     rel_path TEXT NOT NULL,
     is_symlink INTEGER NOT NULL DEFAULT 0,
     symlink_target TEXT,
     readable INTEGER NOT NULL,
     size INTEGER, mtime_ns INTEGER,
     first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
     root_id INTEGER NOT NULL REFERENCES root(id),   -- last: ALTER TABLE put it there in old databases
     UNIQUE(root_id, rel_path)
   );
   INSERT INTO media_path_new (id, media_id, rel_path, is_symlink, symlink_target, readable,
                               size, mtime_ns, first_seen, last_seen, root_id)
     SELECT id, media_id, rel_path, is_symlink, symlink_target, readable,
            size, mtime_ns, first_seen, last_seen, root_id FROM media_path;
   DROP TABLE media_path;
   ALTER TABLE media_path_new RENAME TO media_path;
   -- recreate every index media_path had (copy them from SCHEMA)
   ```
   then `kv_set(conn, 'schema_media_path', '2')`.
4. Call it from `open_db` after `init_schema`. `re2 serve` prints the
   `MigrationBlocked` message and continues without the migration; `re2 up`
   (Phase 6) refuses to start the supervisor until it has run.

Update SCHEMA's `media_path` definition to the new shape so fresh databases
never need the rebuild, and set `schema_media_path = '2'` for them.

**Verify:** `tests/test_db.py`: build the old shape, insert two paths, run the
migration, check both rows survive, check `(1,'a.png')` and `(2,'a.png')` can
both be inserted, check a second run is a no-op. Then run it on a copy of the
live database and compare `SELECT COUNT(*) FROM media_path` before and after.

**Commit:** `Generic 1.2: media paths are unique per root`

### Task 1.3: One function resolves a file

**Do:** add `re2/paths.py`:

```python
def root_dir(conn, root_id: int) -> Path            # cached per process; KeyError if unknown
def absolute(conn, root_id: int, rel_path: str) -> Path
def confined(conn, root_id: int, rel_path: str) -> Path | None   # api.confined_path, per root
```

Replace every `Path(cfg.corpus_root) / rel_path` with `paths.absolute(...)`.
The call sites today are `extract_images.py` (job_from_row), `thumbs.py`,
`video.py`, `asr.py`, `pdf.py` (two places) and `api.confined_path`. Each
query that selects `rel_path` must also select `root_id`. Rows from disabled
roots are skipped by every stage's job selection
(`JOIN root r ON r.id = p.root_id AND r.enabled = 1`).

`scan.scan(conn, cfg)` becomes `scan.scan(conn, cfg, root_id=None)`: with no
id it scans every enabled root in turn; it applies `include_glob` and
`exclude_glob` to the path relative to the root, and sets `root.last_scan_at`.

**Verify:** `grep -rn "corpus_root" re2/` shows only `config.py` (bootstrap)
and the root seeding in `db.py`. `tests/test_scan.py` gains a two-root case:
the same file name in both roots gives two `media_path` rows and, when the
bytes are equal, one `media` row. The full suite passes.

**Commit:** `Generic 1.3: files resolve through their root`

### Task 1.4: Roots API

**Do:** in `re2/api.py`:

| Method and path | Body | Result |
|---|---|---|
| `GET /api/roots` | | `{roots: [{id, name, path, include_glob, exclude_glob, importers, enabled, last_scan_at, files, bytes, by_kind: {image, video, pdf, other}, offline}]}` |
| `POST /api/roots` | `{name, path, include_glob?, exclude_glob?, importers?}` | the new root; 400 if the path is not an existing readable directory, is inside `data/`, or overlaps an existing root (one is a prefix of the other) |
| `PATCH /api/roots/{id}` | any of the fields above except `path`, plus `enabled` | the root |
| `DELETE /api/roots/{id}` | | `{removed_paths}`: deletes its `media_path` rows only; media left with no paths get status `SOURCE_OFFLINE` via `scan.refresh_statuses` |

All writes require the header `X-RE2-Write: 1` (as recaption does). Counts come
from one grouped query over `media_path JOIN media`; `bytes` is the sum of
`media_path.size`.

**Verify:** `tests/test_roots_api.py` covers each row of the table, the three
400 cases, the missing header (403) and that `DELETE` never touches the
filesystem (patch `os.remove`/`shutil.rmtree` to raise).

**Commit:** `Generic 1.4: roots API`

---

## Phase 2: Profiles (what the collection is)

### Task 2.1: The profile

**Do:** a profile is one JSON value in `setting` under key `profile`:

```json
{
  "name": "Redpill archive",
  "about": "A private research collection of political and social-media screenshots, memes, photographs, charts and documents.",
  "purpose": "Kept as evidence and examples of political claims, for comparison with what happened later.",
  "item_noun": "item",
  "tag_guidance": "10 to 20 lowercase tags: topics, events, people, places, and the item's stance or tone …",
  "sensitivity_scale": "0.0 benign, 0.3 political but civil, 0.6 slurs, hate or violence, 0.8 graphic injury or explicit sexual content, 1.0 the most extreme",
  "infer_saved_as": true,
  "infer_later_context": true,
  "tone": "Plain and neutral: describe and attribute; do not moralise, add warnings, or refuse because material is offensive."
}
```

and folder briefs move to a table:

```sql
CREATE TABLE IF NOT EXISTS folder_brief (
  root_id INTEGER NOT NULL REFERENCES root(id),
  folder TEXT NOT NULL,                 -- relative to the root; '.' for the top level
  brief TEXT NOT NULL,
  PRIMARY KEY (root_id, folder)
);
```

Add `profiles/redpill.json` (the owner's values, copied from today's prompt
text) and `profiles/general.json` (a neutral "personal media library" profile
with `infer_saved_as` and `infer_later_context` false). On first run with no
`profile` setting, load `redpill.json` if root 1 exists (the owner's install),
otherwise `general.json`. Import `folder_briefs.toml` into `folder_brief`
for root 1 once, then leave the file in place (it is no longer read).

`re2/profile.py`: `load(conn) -> Profile` (a frozen dataclass),
`save(conn, dict)` (validates: `name` and `about` non-empty, strings under
4,000 characters, booleans are booleans), `digest(profile) -> str` (the first
12 hex characters of sha256 over the canonical JSON), `brief(conn, root_id,
folder) -> str | None` (the deepest matching folder, as `vlm.folder_briefs`
does today).

**Verify:** `tests/test_profile.py`: validation errors, `digest` stable under
key order, `brief` picks the deepest folder, the one-time imports.

**Commit:** `Generic 2.1: collection profile and folder briefs in the database`

### Task 2.2: Prompts are built from the profile

**Do:** add `re2/prompts.py`, the only place prompt text lives. One function
per prompt, each taking a `Profile`:

| Function | Replaces |
|---|---|
| `image_system(p)` | `vlm.IMAGE_SYSTEM_PROMPT` |
| `page_system(p)` | `vlm.PAGE_SYSTEM_PROMPT` |
| `digest_system(p)` | `digest.SYSTEM_PROMPT` |
| `wiki_system(p)`, `wiki_notes_system(p)`, `wiki_split_system(p)` | the three in `wiki.py` |
| `ask_system(p)`, `ask_answer_system(p)`, `map_system(p)` | `ask.SYSTEM_PROMPT`, `ANSWER_SYSTEM`, `MAP_SYSTEM` |
| `steelman_system(p)`, `skeptic_system(p)`, `argue_answer_system(p)` | the three in `argue.py` |
| `relations_system(p, tags)` | `relations.SYSTEM_PROMPT` |

Rules for the builders:
- Start from today's text. Replace "a private research collection of political
  and social-media …" with `p.about`, the collector's purpose with `p.purpose`,
  the tag rule with `p.tag_guidance`, the sensitivity rule with
  `p.sensitivity_scale`, and the moralising rule with `p.tone`.
- When `infer_saved_as` is false, the `saved_as` field instruction becomes
  `"unclear"` always and the schema keeps the field (so stored JSON keeps one
  shape). Same for `later_context` with `"none known"`.
- With `profiles/redpill.json` loaded, every builder must return **exactly**
  today's text (byte for byte). This is what keeps the owner's captions
  current.

Each caption JSON gains `"profile": profile.digest(p)`. Callers load the
profile once per run (`profile.load(conn)`) and pass it down.

**Verify:** `tests/test_prompts.py` loads `profiles/redpill.json` and asserts
each builder equals the old constant (import the constants from a copy kept in
`tests/fixtures/old_prompts.py`, written before the constants are deleted).
Also: with `general.json`, no builder output contains "political".

**Commit:** `Generic 2.2: prompts come from the profile`

### Task 2.3: Redo after a profile change

**Do:** `extract_images.select_jobs` gains `profile_redo: bool = False`. When
true, captions whose `json_extract(json,'$.profile')` differs from the current
digest are selected after new images and older prompt versions, and never
captions with `user_context` (the owner's recaptions stay).
`re2 extract-images --profile-redo` passes it. `GET /api/profile` returns
`{profile, digest, stale_captions}` where `stale_captions` counts those rows.

**Verify:** `tests/test_extract_images.py`: a caption with an old digest is
selected only with the flag, and one with `user_context` never is.

**Commit:** `Generic 2.3: redo captions written under an older profile`

### Task 2.4: Profile API and editor

**Do:**
- `GET /api/profile` (above), `PUT /api/profile` (body: the profile; 400 on
  validation errors; needs `X-RE2-Write`), `GET /api/profile/presets` (names
  and contents of `profiles/*.json`), `GET/PUT /api/roots/{id}/briefs`
  (`{briefs: [{folder, brief}]}`; PUT replaces all of that root's briefs).
- Dashboard, inside Settings (`#/settings`), a **Collection** section above the
  services: the profile fields (textareas for the long ones, checkboxes for
  the two booleans), a preset picker that fills the form, Save, and after a
  save that changes the digest, a line "N captions were written with the
  previous profile · Redo them" whose button starts the extract stage with
  `profile_redo` (Phase 4; until then it shows the command
  `uv run re2 extract-images --profile-redo`). Below it, per root, a folder
  brief list: the root's folders (from `media_path`, top two levels) each with
  a one-line brief input.

**Verify:** `tools/browser_check.mjs` at `desk` and `fold`: the form loads the
current profile, a preset fills it, Save with an empty name shows the error.
API tests for the three endpoints.

**Commit:** `Generic 2.4: profile and folder-brief editor`

---

## Phase 3: Importers

### Task 3.1: The importer interface

**Do:** add the package `re2/importers/`:

```python
# re2/importers/__init__.py
@dataclass
class ImportStats:
    seen: int = 0
    matched: int = 0
    written: int = 0
    def summary(self) -> str: ...

class Importer(Protocol):
    name: str            # stored in root.importers
    label: str           # shown in the UI
    help: str            # one paragraph for the UI
    def run(self, conn, cfg, root_id: int) -> ImportStats: ...

REGISTRY: dict[str, Importer]   # filled by the modules below
def for_root(conn, root_id) -> list[Importer]
```

Importers run after a root's scan (Task 1.3 calls `for_root` and runs each in
order) and never write under the root.

**Commit:** `Generic 3.1: importer interface`

### Task 3.2: Move the owner's importers

**Do:** move code, do not change behaviour:

- `re2/importers/x_archive.py`, name `x_archive_metadata`: today's
  `legacy.reconcile_sources` and `build_posts`. It reads
  `<root>/archive_metadata.json` (no longer from `cfg.archive_metadata`).
- `re2/importers/legacy_captions.py`, name `legacy_captions`: today's
  `legacy.reconcile`. It reads the legacy JSON paths from the root's
  importer options: add `importer_options TEXT NOT NULL DEFAULT '{}'` to
  `root` (JSON, keyed by importer name), and seed root 1 with
  `{"legacy_captions": {"output_json": <cfg.legacy_output_json>,
  "legacy_dir": <cfg.legacy_dir>, "thumbs_dir": <cfg.thumbs_dir>}}`.

Root 1 gets `importers = 'legacy_captions;x_archive_metadata'`. Keep the
`reconcile-legacy` and `reconcile-sources` CLI commands as thin wrappers.

**Verify:** `tests/test_legacy.py` passes unchanged except for imports; on a
database copy, running both importers changes no rows
(`SELECT COUNT(*)` of `source`, `post`, `post_member`, `media` before and after).

**Commit:** `Generic 3.2: the owner's importers behind the interface`

### Task 3.3: A generic sidecar importer

**Do:** `re2/importers/sidecar.py`, name `sidecar_json`: for every file `X`
with a sibling `X.json` (or `X.<ext>.json`) whose object has any of `url`,
`date` (ISO 8601), `title`, `description`, `tags` (list of strings), write a
`source` row (`url`, `date`), a `legacy_caption` doc from `title` and
`description`, and `media_tag` rows with weight 1.0 through
`tags.normalize`. Sidecar files themselves are excluded from `media`
(`scan.is_excluded` gains `*.json` next to media files of the same stem when
the root uses this importer). This covers yt-dlp `--write-info-json`,
gallery-dl metadata and hand-written notes.

**Verify:** `tests/test_importers.py` with a temp root holding `a.jpg` and
`a.jpg.json`, and `v.mp4` and `v.info.json` (yt-dlp names: `upload_date`
`YYYYMMDD` maps to `date`, `webpage_url` to `url`).

**Commit:** `Generic 3.3: sidecar JSON importer`

---

## Phase 4: Services that say whether they work

### Task 4.1: Speech-to-text by API

**Do:**
1. `SERVICE_FIELDS["asr"]` gains `provider` (`local` or `api`), `base_url`,
   `api_model`, `api_key`.
2. `re2/asr.py`: `transcribe(model, cfg, media_id, path)` dispatches on
   `cfg.asr_provider`. The `api` path extracts 16 kHz mono FLAC with ffmpeg
   (as today, but FLAC), splits audio longer than 20 minutes into 20-minute
   chunks, and for each chunk POSTs multipart to
   `{base_url}/audio/transcriptions` with `model`, `file`,
   `response_format=verbose_json`, `timestamp_granularities[]=segment`, and
   `Authorization: Bearer <api_key>` when a key is set. Segment times are
   offset by the chunk start. It returns the same `list[Segment]` as the local
   path, so `windows` and `store` are unchanged.
3. `run()` loads the local model only for `provider == "local"`.

**Verify:** `tests/test_asr.py` with `httpx.MockTransport`: a 45-minute input
(mock ffmpeg with a fake duration) makes three requests, segment times are
offset correctly, a 401 becomes a failure row, not a crash.

**Commit:** `Generic 4.1: speech-to-text through an OpenAI-compatible API`

### Task 4.2: API keys

**Do:** every service gains an optional `api_key`, sent as
`Authorization: Bearer …` by `ask._stream_chat`, `writer.complete`,
`vlm.extract`, `embed.embed_texts`. `GET /api/settings` returns keys masked
(`"sk-…abcd"`, the last four characters); `POST /api/settings` treats a value
equal to the masked form as "unchanged". Keys are stored in `setting` like the
other fields; document that the database file therefore holds secrets.

**Verify:** `tests/test_settings.py`: masking, the unchanged rule, and a
MockTransport check that each client sends the header.

**Commit:** `Generic 4.2: API keys for every service`

### Task 4.3: Health probes

**Do:** add `re2/services.py`:

```python
@dataclass
class Health:
    service: str
    ok: bool
    checked_at: float
    latency_ms: int | None
    detail: str                 # one line a person can act on
    model_listed: bool | None   # the configured model is in /models
    loaded: bool | None         # llama.cpp router: status.value == 'loaded'
    context: int | None         # per request (vlm.server_limits logic)
    slots: int | None
    vision: bool | None         # vision only: accepted a 1×1 PNG
    json_schema: bool | None    # chat and vision: honoured response_format
    fingerprint: str | None     # embed only (Task 4.4)

def probe(cfg, service: str, deep: bool = False) -> Health
```

- The shallow probe lists `/models` (5 s timeout) and reads llama.cpp's
  `status` and args when present, as `vlm.server_limits` does.
- `deep=True` also sends one tiny request: vision sends a 1×1 PNG with a
  schema `{"ok": bool}`; chat sends the same schema without the image; embed
  embeds one fixed sentence; asr (api) sends 1 s of silence; asr (local)
  checks that `faster_whisper` imports and reports the CUDA device, or "CPU".
- Deep probes never load a model on an llama.cpp router that has a different
  model loaded (`loaded is False`); they report "would load <model>; run it
  from the Pipeline page instead" and set `ok = None`.
- `GET /api/services/health` returns shallow results for all services, cached
  30 s. `POST /api/services/{name}/probe` runs a deep probe.
- Also report host tools in the same response: `ffmpeg` and `ffprobe`
  (`shutil.which` plus `-version`), free disk space under `data/`, and the
  local embedding server's `/health`.

**Verify:** `tests/test_services.py` with MockTransport for each outcome:
unreachable, model missing, model not loaded (no deep request sent), vision
refused, JSON schema ignored (the reply is not JSON), all good.

**Commit:** `Generic 4.3: service health probes`

### Task 4.4: The embedding fingerprint

**Do:** the index is only valid with the embedding model and build that wrote
it.

1. `embed.fingerprint(client, cfg) -> list[float]`: the vector for the fixed
   sentence `"search_document: The quick brown fox jumps over the lazy dog."`,
   rounded to 4 decimals.
2. `build_docs` stores it in `kv.embed_fingerprint` (JSON) when the index is
   empty or rebuilt, and before adding documents compares the current
   fingerprint with the stored one. Cosine below 0.999 raises
   `EmbedError("the embedding service no longer matches the index (cosine
   0.92); rebuild it with --rebuild")`.
3. `services.probe(cfg, "embed", deep=True)` reports the cosine in `detail`.
4. Saving a changed `service.embed` in Settings runs that probe and shows the
   warning with the document count and an estimate (documents ÷ the measured
   rate).

**Verify:** `tests/test_embed.py`: matching fingerprints pass, a rotated one
raises, `--rebuild` resets it.

**Commit:** `Generic 4.4: the index remembers which embeddings made it`

### Task 4.5: Services UI

**Do:** in Settings, each service card shows a health line (green `--focus`
dot, orange `--hot`, hollow when unknown) with `detail`, latency, context and
slots, a **Test** button (deep probe) and, for ASR, the provider switch that
shows either the local fields or the API fields. A **Host** card lists
ffmpeg, disk space and the local embedding server.

**Verify:** `tools/browser_check.mjs` at `desk` and `fold` with the owner's
services: chat shows loaded or not, the embed card shows the cosine after
Test.

**Commit:** `Generic 4.5: service health in Settings`

---

## Phase 5: The job manager

### Task 5.1: Stage registry

**Do:** add `re2/stages.py`. A stage is:

```python
@dataclass(frozen=True)
class Stage:
    name: str                       # 'scan', 'thumbs', 'extract_images', …
    label: str                      # 'Read images'
    after: tuple[str, ...]          # stages whose output it needs
    services: tuple[str, ...]       # 'vision', 'chat', 'wiki', 'embed', 'asr'
    command: tuple[str, ...]        # argv after 're2', e.g. ('extract-images',)
    pending_sql: str                # one query returning the count still to do
    done_sql: str                   # one query returning the count done
    unit: str                       # 'images', 'videos', 'pages', 'documents', 'pages'
```

Register, in pipeline order: `scan`, `thumbs`, `video_probe`, `pdfs`,
`extract_images`, `pdf_ocr`, `asr`, `dates`, `embed`, `recluster`, `digests`,
`wiki_build`. Write each `pending_sql` from the stage's own `select_jobs`
query (count instead of select), so the numbers match what the stage would
do. `scan`, `dates` and `recluster` have no pending count; they report
`last_run` instead.

`re2 pipeline` prints a table of stage, done, pending, and the last run.

**Verify:** `tests/test_stages.py`: every `pending_sql` runs on an empty
schema; on a fixture database each count equals `len(select_jobs(...))` of the
matching module.

**Commit:** `Generic 5.1: stage registry with pending counts`

### Task 5.2: Workers report progress

**Do:**
1. `re2/db.py` SCHEMA:
   ```sql
   CREATE TABLE IF NOT EXISTS job_run (
     id INTEGER PRIMARY KEY,
     stage TEXT NOT NULL,
     pid INTEGER, host TEXT,
     state TEXT NOT NULL,           -- starting | running | stopping | finished | interrupted | failed | lost
     args_json TEXT NOT NULL DEFAULT '{}',
     selected INTEGER, done INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
     rate_per_min REAL,
     last_error TEXT,
     started_at TEXT NOT NULL DEFAULT (datetime('now')),
     heartbeat_at TEXT,
     finished_at TEXT
   );
   CREATE INDEX IF NOT EXISTS ix_job_run_stage ON job_run(stage, id);
   ```
2. `re2/runner.py`: `run_jobs` gains `progress: Callable[[int, int], None] |
   None = None`, called on the calling thread after each result or error with
   `(done, failed)` and at most once every 2 seconds.
3. `re2/jobreport.py`: `Report(conn, stage, args)` inserts a `job_run` row in
   state `running` (pid, hostname), and has `progress(done, failed)` (updates
   counts, rate over the last 5 minutes, `heartbeat_at`), `error(text)`, and
   `finish(state)`. Every stage's `run()` accepts `report: Report | None` and
   passes `report.progress` to `run_jobs`.
4. `cli.main` creates a `Report` for every stage command and finishes it with
   `finished`, `interrupted` (KeyboardInterrupt or the stats say so) or
   `failed` (exception, with the message).

**Verify:** `tests/test_runner.py`: `progress` is throttled and gets final
counts. Run `uv run re2 thumbs --limit 5` and check the `job_run` row.

**Commit:** `Generic 5.2: every stage run is recorded with live progress`

### Task 5.3: The supervisor

**Do:** add `re2/supervisor.py`, running as a thread inside `re2 serve`.

- **Start:** `start(stage, args)` refuses when:
  - a `job_run` for that stage is `starting`, `running` or `stopping`;
  - another running stage uses the same service `base_url` with a different
    model (an llama.cpp router would swap models under it);
  - `after` stages have never finished (message: "run scan first").

  Otherwise it spawns `[sys.executable, "-m", "re2.cli", *stage.command,
  *args]` with `start_new_session=True`, stdout and stderr appended to
  `data/logs/<stage>.log`, and records the pid.
- **Stop:** `stop(stage)` sends `SIGINT` (the runner finishes in-flight work)
  and sets `stopping`; `kill(stage)` sends a second `SIGINT`, then `SIGTERM`
  after 10 s. On Windows use `CTRL_BREAK_EVENT`; the supervisor runs there
  only through `re2 up` (Phase 6).
- **Watch:** every 5 s, runs whose pid is gone and whose state is still
  `running` or `stopping` become `lost`, as do runs with no heartbeat for 120 s
  whose pid is alive (log it; do not kill).
- **Restart:** workers are separate processes, so restarting the server does
  not stop them; on start the supervisor re-adopts `running` rows whose pid is
  alive and whose command line matches.
- **Retry failures:** `retry_failures(stage)` deletes that stage's `failure`
  rows so the next run tries those items again.

API (all writes need `X-RE2-Write`):

| Method and path | Result |
|---|---|
| `GET /api/pipeline` | stages with label, services, state, done, pending, rate, ETA (pending ÷ rate), last run, failures count, log tail (last 20 lines) |
| `POST /api/pipeline/{stage}/start` | body `{limit?, concurrency?, server?, model?, profile_redo?}` |
| `POST /api/pipeline/{stage}/stop` | |
| `POST /api/pipeline/{stage}/kill` | |
| `GET /api/pipeline/{stage}/failures` | `{failures: [{media_id, title, attempts, error, updated_at}]}` |
| `POST /api/pipeline/{stage}/retry` | |
| `POST /api/pipeline/run-all` | starts the first stage in order that has pending work and whose `after` stages are idle; when it finishes, the supervisor starts the next (a `run_all` flag in `setting`) |

**Verify:** `tests/test_supervisor.py` with a fake stage whose command is a
small Python script (sleeps, writes progress, handles SIGINT): start, the
refusals, stop finishing gracefully, kill, lost detection, re-adoption.

**Commit:** `Generic 5.3: supervisor starts, stops and watches stages`

### Task 5.4: Schedules and energy

**Do:**
1. `setting` key `schedule.<stage>`:
   `{"enabled": bool, "days": "mon,tue,…", "start": "10:00", "end": "16:00",
   "concurrency": int | null}`. The supervisor starts the stage when the
   window opens and it has pending work, and stops it gracefully when the
   window closes.
2. `setting` key `hosts`: `[{"name": "cube", "base_url": "…", "watts": 1400}]`.
   `GET /api/pipeline` adds, for a running stage whose service matches a host,
   `per_kwh = rate_per_min * 60 / (watts / 1000)` and `kwh_left = pending /
   per_kwh`.

**Verify:** `tests/test_supervisor.py`: a window that opens and closes with a
patched clock starts and stops the fake stage; the energy numbers for 6/min at
980 W are 367 per kWh.

**Commit:** `Generic 5.4: schedules and energy per item`

### Task 5.5: The Pipeline page

**Do:** a new dashboard mode `pipeline` (`g p`, a **Pipeline** button in the
top bar and bottom bar). Layout, top to bottom:

1. **Roots**: one row each: name, path, files, size, images/videos/PDFs,
   offline count, last scan, enabled switch, Scan now, Remove (with a confirm
   that says nothing on disk is deleted). An **Add folder** form (name, path,
   include/exclude, importers as checkboxes with their help text).
2. **Stages**: one row per stage in pipeline order: label, state dot, a bar
   of done ÷ (done + pending), "1,860 of 12,028 images", rate, ETA, service
   and its health dot, Start/Stop (Stop turns into Kill after 10 s), a
   concurrency input, a schedule summary that opens the schedule editor, and
   the failures count that opens the failures list (item title, error,
   Retry all). **Run all** above the table. Rows refresh every 3 s while the
   page is open (one `GET /api/pipeline`).
3. **Storage**: database size, thumbnails, keyframes, index files, logs, free
   space.
4. **Energy**: the hosts list with watts, and per running stage items per kWh
   and kWh left.

Keyboard: `j k` move between stage rows, `s` starts or stops the stage under
the cursor, `f` opens its failures, `L` opens its log tail. Touch: every
action is a button.

**Verify:** `tools/browser_check.mjs` at `desk`, `open` and `fold`: the page
lists the owner's stages with the same counts as `re2 pipeline`; starting the
`thumbs` stage with `limit 5` shows running, then finished; the failures list
opens. Read the screenshots in both themes.

**Commit:** `Generic 5.5: Pipeline page`

### Task 5.6: Retire the scripts

**Do:** every `scripts/*.sh` stays and keeps working (the owner uses them), but
their header says the Pipeline page is the usual way now. The help panel and
README describe the Pipeline page first.

**Commit:** `Generic 5.6: scripts point to the Pipeline page`

---

## Phase 6: One command

### Task 6.1: `re2 up`

**Do:** `re2 up [--host H] [--port P] [--no-auth]`:

1. Load bootstrap config, open the database, run migrations (fail with a clear
   message if Task 1.2's migration is blocked).
2. Start the local embedding server if the embed URL points at this machine
   (`embed.ensure_local_server`); if llama-server is not built, print the one
   command that builds it (`re2 setup-embed`, below) and continue with
   keyword-only search.
3. Start the web server with the supervisor thread.
4. Print the URL, and open it in a browser when stdout is a terminal.

`re2 setup-embed`: the Python version of `scripts/setup_embed_server.sh`
(clone llama.cpp at the pinned commit into `data/tools/llama.cpp`, CMake CPU
build of `llama-server`, download `nomic-embed-text-v1.5.Q8_0.gguf` from
Hugging Face `nomic-ai/nomic-embed-text-v1.5-GGUF` into `data/models/`, check
its sha256). It needs `git`, `cmake` and a C++ compiler; it says so if any is
missing.

Installation for a new user, documented in `README.md`:

```
git clone <repo> && cd redpill_explorer_2
uv tool install .
re2 setup-embed
re2 up
```

**Verify:** in a fresh clone and an empty `data/`, `re2 up` starts, the
browser shows the setup wizard (Task 6.2), and `Ctrl-C` stops the server
while a running stage keeps going (it is a separate process).

**Commit:** `Generic 6.1: re2 up`

### Task 6.2: First-run wizard

**Do:** when `root` is empty, every page redirects to `#/setup`, four steps
with Back and Next:

1. **Folders**: the Add folder form from Task 5.5; at least one root.
2. **Models**: the service cards from Task 4.5 with Test; Next is allowed
   when vision and chat have passed a deep probe (embed and ASR may be left
   for later; the stages that need them stay disabled with a note).
3. **Collection**: preset picker and the profile form from Task 2.4.
4. **Start**: a summary (files found by a quick count, which stages will
   run), and **Scan and start** (runs `scan`, then Run all).

**Verify:** `tools/browser_check.mjs` at `desk` and `fold` on an empty
database: walk all four steps with a MockTransport-free local fake server
(`tests/fake_openai.py`, a tiny FastAPI app that answers `/models`,
`/chat/completions` and `/embeddings`, started on port 4029 for the check).

**Commit:** `Generic 6.2: first-run wizard`

---

## Phase 7: Safe to leave running

### Task 7.1: A password

**Do:**
1. `setting` key `server.password_hash`: `scrypt` from `hashlib`
   (`n=2**14, r=8, p=1`, 16-byte salt), stored as `scrypt$<salt hex>$<hash hex>`.
2. A signing secret in `data/secret.key` (32 random bytes, created on first
   run, file mode 0600).
3. `POST /api/login {password}` sets the cookie `re2` = `<expiry>.<hmac>`
   (HMAC-SHA256 with the secret, 30-day expiry, `HttpOnly`, `SameSite=Strict`).
   `POST /api/logout` clears it.
4. A middleware allows `GET /login` and `POST /api/login` always; everything
   else needs a valid cookie when a password is set.
5. Every non-GET request needs `X-RE2-Write: 1` (the recaption, settings and
   roots endpoints already do; extend it to all).
6. `re2 up` refuses to listen on anything but `127.0.0.1` when no password is
   set, unless `--no-auth` is given, and says why.
7. Settings gains **Password** (set, change, remove).

**Verify:** `tests/test_auth.py`: with no password everything works; with one,
`/api/stream` is 401 without the cookie and 200 with it; a tampered or expired
cookie is 401; a POST without the header is 403; `re2 up --host 0.0.0.0`
without a password exits with the message.

**Commit:** `Generic 7.1: password and signed session`

### Task 7.2: Media only from roots

**Do:** `/media/{id}/file` and `/media/{id}/thumb` already confine files to the
corpus; after Phase 1 they use `paths.confined` for the media's first readable
path in an **enabled** root. Add a test that a path escaping the root through
`..` or a symlink is refused, and that a disabled root's files are 404.

**Commit:** `Generic 7.2: media served only from enabled roots`

---

## What the owner does after each phase

| After | The owner runs |
|---|---|
| Phase 1 | nothing new; `re2 status` shows the same counts |
| Phase 2 | checks that Settings → Collection shows the Redpill profile; nothing is redone |
| Phase 4 | opens Settings, presses Test on each service |
| Phase 5 | uses the Pipeline page instead of the scripts; enters watts for cube and whiteglove |
| Phase 6 | `re2 up` instead of `re2 serve` |
| Phase 7 | sets a password before listening on the LAN |
