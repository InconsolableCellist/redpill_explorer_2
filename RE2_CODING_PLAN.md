# RE2 Coding Plan

Rewritten 2026-09-14 (second revision). Companion documents:
`RE2_INVESTIGATION_REPORT.md` (the audit) and `RE2_REPORT_REVIEW.md` (the
corrections the first revision was based on). When they disagree with this
plan, this plan wins.

## What we are building

A **Redpill Dashboard**: a keyboard-driven knowledge dashboard over the
archive in `~/Downloads/redpills`. It has four layers, each built on the one
below:

1. **Evidence.** Every file hashed and catalogued; every image read by a
   vision model (description, verbatim OCR, entities, claims); every video
   transcribed with timestamps and its keyframes read; every PDF page read.
2. **Search.** Full-text (SQLite FTS5) plus dense embeddings, fused, with
   results that stream into the dashboard as you type and correlate across
   entities, events, topics, dates and folders.
3. **Compiled knowledge (the wiki).** An offline pass in which the local LLM
   reads the whole corpus once and writes a cited wiki about it: event pages,
   topic and theme pages, and entity pages, all linking down to the items
   that support them. Browsable in the dashboard and exportable as a folder of
   Markdown files (Obsidian-compatible).
4. **Ask.** A chat agent that answers questions about world events using its
   own knowledge coloured and corrected by the archive. It works through tools
   (search, open wiki page, open item, timeline), cites archive items, and has
   its citations checked by the server.

A QLoRA is out of scope. Fine-tuning does not give reliable, citable recall of
specific items, and every new download would need retraining. The chat log
table keeps what a future behaviour adapter would need.

Work through the phases in the order given in section "Order of work". Each
task says what to build, which files it touches, how to verify it, and when
to commit. Do not skip verification.

---

## 0. Ground rules (read before every session)

### 0.1 Hard constraints

1. **Never write under `~/Downloads/redpills`.** It is an NFS mount from the
   NAS. Open files there with mode `rb` only. Never `os.remove`, `os.rename`,
   `shutil.move`, `open(..., 'w')`, or `os.symlink` on any path under it. All
   derived data goes under the repo's `data/` directory.
2. **GPU work.** Vision, chat, digests, wiki writing and embeddings are HTTP
   calls to the host `cube`. ASR (faster-whisper) runs locally on the GTX
   1080 Ti. Nothing else uses a local GPU.
3. **Python 3.12 via `uv`.** Every Python command is `uv run <command>` from
   the repo root. The system Python is 3.14; do not use it.
4. **One writer per stage.** Stages may run concurrently (the DB is in WAL
   mode), but never two copies of the same stage.
5. **Long batch jobs are started by the human.** Every long job:
   - is a `re2` subcommand built on `re2/runner.py` (`run_jobs`), so Ctrl-C
     once saves in-flight items and exits, Ctrl-C twice exits immediately,
     and re-running resumes;
   - has a wrapper in `scripts/` with a header comment explaining what it
     loads on cube and how to stop and resume it;
   - is verified by the coding agent with `--limit 5` (or similar) only.
   Reason: cube's router loads one model at a time and long jobs evict the
   human's interactive model.
6. **Sensitive content is the point of the archive.** Prompts instruct models
   to describe and quote plainly without softening, warnings or moralising.
   Never add content filters. The dashboard blurs thumbnails above a
   sensitivity threshold the user controls; that is a viewing convenience,
   not a filter.

### 0.2 Hosts and endpoints

| Purpose | Value |
|---|---|
| Corpus root (read-only) | `/home/offipso/Downloads/redpills` (symlink to `/home/offipso/storage/Downloads/redpills`; always compare `realpath`s) |
| Trash to skip | any directory named `.Trash-1000` |
| LLM router | `https://cube.tail99c274.ts.net:5000/v1` (OpenAI-compatible, HTTPS only; `http://cube:5000` does not work). llama.cpp router mode, `--models-max 1`: the request's `model` field selects and swaps the model. |
| Vision model | `gemma-4-31b-heretic-ara-batch-parallel6-q8-8k-cube`: 8192 tokens of context **shared by all concurrent requests**, 6 slots. Client concurrency 3. Every request must send `"chat_template_kwargs": {"enable_thinking": false}` and `"reasoning_budget": 0`, or Gemma 4 spends the whole output budget thinking. |
| Bulk text model (digests) | `qwen-3.6-35b-a3b-q8-256k-cube` (MoE, fast), 3 slots |
| Writing / chat model | `qwen38-27b-heretic-uncensored-q8-cube`: 262k context per slot, 3 slots |
| Embedding server | `http://cube:5010/v1/embeddings`, a separate `llama-server --embedding` process on cube (Task 3.1), unaffected by router swaps. Model `nomic-embed-text-v1.5` (768 dims). Prefixes: `search_document: ` and `search_query: `. |
| ASR | faster-whisper `large-v3`, `device="cuda"`, `compute_type="int8_float32"`, local 1080 Ti, measured 19x realtime |
| Web server | FastAPI (uvicorn) on `127.0.0.1:4001` |

### 0.3 Conventions

- Every CLI subcommand accepts `--limit N` where it makes sense, prints a
  one-line summary at the end, exits 0 on success, and logs to
  `data/logs/<command>.log` (already wired in `re2/cli.py`).
- Every ingestion step is idempotent: it selects only items without a result
  row, records failures with `db.upsert_failure`, and skips items with 3
  failed attempts (`db.pending_failures`).
- `logging` in library code, never `print`.
- `httpx` with explicit timeouts: vision 600 s, chat/digest 300 s,
  embeddings 60 s.
- Paths are `pathlib.Path`; subprocesses get argument lists, never shell
  strings. Filenames contain spaces, quotes, unicode, `@` and full-width
  punctuation.
- Model-generated JSON is always requested with `response_format` JSON schema
  and parsed with `json.loads`; strip a leading `<|channel>thought…<channel|>`
  block (see `vlm.THOUGHT_PREFIX`).
- Tests use in-memory SQLite, temp directories and `httpx.MockTransport`;
  they never call cube.
- Commit after each task with the message given. Never commit `data/`,
  `legacy/`, `public/thumbnails/`, `*.sqlite*`, `*.npy`, `node_modules/`.
  End commit messages with the attribution line the session requires.

---

## Status on 2026-09-14

| Task | State | Commit |
|---|---|---|
| 0.1 uv project, config, gitignore | done | 5a4a5b6 |
| 0.2 SQLite schema (`re2/db.py`) | done | 822f5f2 |
| 0.3 `server.js` bound to localhost, `/images` confined, node-info escaped | done | 35c7d8e |
| 1.1 read-only scanner (`re2/scan.py`) | done, full scan run | e1cf80b |
| 1.2 legacy captions, tags, thumbnails, output.json (`re2/legacy.py`) | done, run | ba3dbba |
| 1.3 archive_metadata sources | done, run | ba3dbba |
| 1.4 `re2 status` | done | ba3dbba |
| 2.1 thumbnails (`re2/thumbs.py`) | done, run: every image has one | ba3dbba |
| 2.2 vision client (`re2/vlm.py`) | done, verified live | 4c622bd, 71f2291 |
| 2.3 image extraction (`re2/extract_images.py`, `scripts/extract_images.sh`) | done, verified live on 5 images; **full run is the human's** (in progress) | 4c622bd, 71f2291, be66a7b, d05de1e |
| 3.1 embedding server (`scripts/start_embed_server_on_cube.sh`) | done, running on cube:5010 | 4a7d284 |
| 3.2 embedding index (`re2/embed.py`, `scripts/embed.sh`) | done, run (re-run after extraction sessions) | 640eaff |
| 3.3 search core (`re2/search.py`) | done | dd721e1 |
| 3.4–3.6 FastAPI server, streaming search, similar items (`re2/api.py`); `server.js` deleted | done | 8667590 |
| 4.1 video probe and keyframes (`re2/video.py`, `scripts/video_probe.sh`) | done, full run: 4,641 videos, 16,113 keyframes | 4b64190 |
| 4.2 ASR (`re2/asr.py`, `scripts/asr.sh`) | done, verified on 3 videos; **full run is the human's** | 4b64190 |
| 5 PDFs (`re2/pdf.py`, `scripts/pdf_ocr.sh`) | text layer run (1,213 pages); 809 scanned pages queued for `pdf_ocr.sh` (not yet run live) | 1debd50 |
| 6.2 item dating (`re2/dates.py`) | done, run: every item dated | f496155 |
| 8.1–8.2 dashboard shell, streaming search, reader; timeline and graph of 8.4 | done, checked in headless Chromium | 07b340c, bdbb1b7 |
| Posts: a tweet's screenshot, photos and videos as one item (`post`, `post_member`) | done, 1,302 posts | f1ff73b, f9edde5 |
| Sorting (relevance, newest, oldest, recently read); `re2 review` | done | f1ff73b, 8044539 |
| Vision prompt: subject-first title and summary, schema caps, DRY on retries | done, tested on mlboy; takes effect on the next extraction restart | 6ad66cc, 3308301 |
| 6.4 deterministic entity merge (`re2 merge-entities`) | done, run; LLM verification still to do | fe078dd |
| 6.5 topics, themes, 2D map (`re2 recluster`) from mean doc vectors; 8.4 map mode | done, run: 181 topics, 17 themes; labels are provisional keywords | 41ad957 |
| 7 Ask agent (`re2/ask.py`, `/api/ask`) and 8.5 Ask mode | done, tested live against mlboy | e9f5d40, 241324b, a0d97ea |
| Which-key popup, view keys on buttons | done | 6c318fc |

Deviations from the task text, decided during implementation:

- Gemma 4 thinks by default; every vision request sends `enable_thinking: false` and `reasoning_budget: 0`.
- Images taller than 2.5x their width are sent as up to 6 overlapping 768-px slices. Images that run out of
  output tokens are retried in a one-at-a-time phase with 5,000 tokens, and text-dense portrait images are sliced
  on that retry. Unparseable outputs are kept in `data/logs/vlm_raw/`.
- Long jobs use `re2/runner.py` (Ctrl-C saves in-flight work) and log to `data/logs/<command>.log`.
- Item dates live on `media_meta.item_date` / `date_source` (added by `db.ADDED_COLUMNS`), not only on
  `item_digest`. Filenames are decoded more widely than Task 6.2 lists: 4chan millisecond and microsecond names,
  `screenshot_<topic>_<unix seconds>`, X tweet ids in video names, and pbs.twimg media keys (Twitter snowflakes).
- Scanned PDF pages use a slimmer `PAGE_SCHEMA` and concurrency 2; the queue is table `pdf_ocr_queue`.
- Veiling in the dashboard uses only the vision model's sensitivity; the 2024 tagger's `legacy_spicy` is shown
  but not trusted (it rates benign documents 0.9).
- The graph can show tags as well as entities (`T`) until vision extraction has produced enough entities.
- Clustering uses item vectors = mean of the item's doc vectors (no digests yet); `re2 recluster` should be re-run
  after extraction, ASR and `embed`, and again after digests exist.
- llama.cpp rejects JSON-schema `maxLength` of 2,000 or more ("failed to parse grammar"): OCR stays uncapped.
- `config.local.toml` (git-ignored) overrides `config.toml`; on this machine it points Ask at mlboy and binds the
  server to 0.0.0.0 for LAN access. The server has no authentication.
- Ask steers the agent by narrowing the action enum (no searches after 4, no opens after 5, answer on the last
  step) and writes the answer in a fresh conversation from research notes; otherwise Qwen answered in JSON.

Measured facts that replace the old estimates:

- Corpus: 79 GB, 17,137 files: 12,028 images, 4,641 videos, 51 PDFs,
  10 unsupported, 24 excluded (cookies, zips, `.part`, txt/json/tsv/yml/py).
  Zero broken symlinks: the hanoi files were migrated into the mount on
  2026-09-14. Full scan 12.6 min; rescan 5 s.
- 7,716 legacy captions (5,503 from `images_captioned.json`, the rest from
  `output.json`), all on readable files; 5,144 items carry legacy tags.
- 2,233 media have a source row (URL + date) from `archive_metadata.json`;
  9,989 images have no source date.
- Image extraction: ~880 prompt tokens and ~900 completion tokens per image;
  about 3 images per minute at concurrency 3, so ~67 hours for all images.
  Images taller than 2.5x their width (131) are sent as up to 6 overlapping
  768-px slices and processed last, one at a time, with a 3,000-token budget.
- Excluded files are keyed by path (`uid = 'excluded:' + sha1(rel_path)`) and
  never opened.

Files in the repo root that belong to the pre-RE2 system and are not used by
anything in this plan: `assign_weights.py`, `auto_caption_on_filesystem_changes.py`,
`caption_images_with_cogvlm2.py`, `caption_images_with_intern-vl3.py`,
`generate_tag_pairs.py`, `tag_caption_output.py`, `make_thumbnails.py`,
`preprocess_data.js`, `hashMapper.js`, `cogvlm2server/`, the two
`*_prompt_template.txt` files, `available_tags*.txt`. Task 9.3 moves them.

---

## Order of work

The image extraction run occupies cube's router for about three days. Work
that does not need the router goes first:

1. **Phase 3** (embedding server, index, search, API). The embedding server
   is a separate process and runs alongside extraction. Build the index from
   whatever captions exist, then refresh it as extraction progresses.
2. **Task 6.2** (item dating: no model needed, and the timeline and `date:`
   scopes depend on it), then **Phase 8, tasks 8.1, 8.2 and the timeline
   of 8.4** (dashboard shell, streaming search, reader, timeline).
3. **Phase 4.1–4.2** (ffprobe, keyframe files, ASR). All local; the human
   runs ASR while cube extracts images.
4. **Phase 5** (PDF text layer; scanned pages queue for the vision model).
5. After image extraction: **Phase 4.3** (keyframe vision run, human-started),
   then **Phase 6** (digests, dating, entity merge, clustering, wiki).
6. **Phase 7** (ask agent), **Phase 8, part 2** (wiki mode, ask mode,
   network and map views).
7. **Phase 9** (retire the old server and pages).

---

## Phase 2 remainder

### Task 2.4: Tags are demoted to a facet

The old tag graph, matrix and "spicy" 3D view were built on LLM-guessed tag
weights. This plan replaces tags as the primary navigation with entities,
dates and the wiki hierarchy (topic, theme, event). Tags stay as a secondary
facet in the dashboard, normalized by `tags.normalize`.

Nothing to build now. Do **not** implement PMI or regenerate the legacy JSON
files. The old pages keep reading their existing frozen JSON under `/legacy`
(Task 3.5) until Phase 9 retires them.

---

## Phase 3: Embeddings, search, API

### Task 3.1: Embedding server on cube

**Do:** `scripts/start_embed_server_on_cube.sh`, run by the human **on cube**:

```bash
#!/usr/bin/env bash
# Run ON cube. Serves nomic-embed-text-v1.5 on port 5010 on GPU 5 (~1 GB).
# Independent of the llama.cpp router on port 5000, so model swaps do not affect it.
# Stop: pkill -f 'llama-server --embedding'
set -euo pipefail
mkdir -p ~/logs
cd ~/ML/llama.cpp
nohup ./build/bin/llama-server --embedding \
  -m /home/offipso/ML/models/nomic-embed/nomic-embed-text-v1.5.Q8_0.gguf \
  --host 0.0.0.0 --port 5010 -ngl 99 --device CUDA5 \
  -c 8192 -b 8192 -ub 8192 --parallel 4 --pooling mean \
  > ~/logs/embed-server.log 2>&1 &
echo "started; log: ~/logs/embed-server.log"
```

The coding agent may start it itself with
`ssh -o BatchMode=yes cube 'bash -s' < scripts/start_embed_server_on_cube.sh`
(it does not touch the router).

**Verify (from this machine):**
`curl -s http://cube:5010/v1/embeddings -H 'Content-Type: application/json' -d '{"input":["search_query: test"]}' | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data'][0]['embedding']))"`
prints `768`. If `--device CUDA5` is rejected, drop it and prefix the command
with `CUDA_VISIBLE_DEVICES=5`.

**Commit:** `Phase 3.1: embedding server script`

### Task 3.2: Embedding index

**Do:** `re2/embed.py`.

1. `embed_texts(client, cfg, texts: list[str], prefix: str) -> np.ndarray`:
   float32, L2-normalized, shape N×768. POST `{"input": [prefix + t[:cfg.embed_max_chars] ...]}`
   in batches of 32 to `cfg.embed_url`; retry HTTP 5xx and timeouts twice with
   a 5 s sleep; raise `EmbedError` otherwise.
2. An index is a named pair of files under `data/index/`:
   `<name>.npy` (float32 N×768) and `<name>_ids.npy` (int64 N). Class
   `Index(name)` with `load()` (lazy, cached, reloads when the file mtime
   changes), `append(ids, vecs)` (write both to temp files in the same
   directory, then `os.replace`), `remove(ids)`, `search(qvec, k, allowed_ids=None) -> list[(id, score)]`
   (`scores = vecs @ q`; when `allowed_ids` is given, mask to them first;
   `np.argpartition` for the top k).
3. Index `docs` (ids are `doc.id`). `build_docs(conn, cfg, limit=None, rebuild=False)`:
   select docs not yet in the index, excluding a `legacy_caption` doc when
   the same media has a `caption` doc, and excluding docs of `EXCLUDED`
   media. When a media gains a `caption` doc, remove its `legacy_caption` doc
   id from the index in the same run. Embed with `search_document: `, using
   `runner.run_jobs` over batches of 32 (concurrency 4). Append once per 50
   batches and at the end, so Ctrl-C loses at most one chunk.
4. `query_vector(client, cfg, text) -> np.ndarray` with `search_query: `.
   Keep an LRU cache of the last 256 query vectors (typing re-sends prefixes).

CLI: `re2 embed [--limit N] [--rebuild]`. Script: `scripts/embed.sh` (header
notes: uses only the port 5010 server; safe to run during extraction; re-run
after every extraction session to index new captions).

**Verify:** `uv run re2 embed --limit 200`. Then
`uv run python -c "from re2 import config, embed; ..."` querying
"voter fraud ballots" returns doc ids whose text mentions ballots or
elections (check two with a SELECT). The full run over ~8k legacy captions
takes minutes; the agent may run it.

`tests/test_embed.py`: MockTransport returning fixed vectors; `append`,
`remove`, `search` with and without `allowed_ids`.

**Commit:** `Phase 3.2: dense embedding index`

### Task 3.3: Search core with scopes

**Do:** `re2/search.py`.

1. **Scopes.** A scope is a string `type:value`. Types and meaning:
   - `text:<words>` items with a doc matching the words (FTS)
   - `entity:<entity_id>`, `tag:<tag_id>`, `folder:<rel dir prefix>`
   - `kind:image|video|pdf`, `status:READY|THUMBNAIL_ONLY|SOURCE_OFFLINE`
   - `date:<from>..<to>` with `YYYY`, `YYYY-MM` or `YYYY-MM-DD` bounds, either
     side optional; uses `item_date` (Task 6.2; until then `source.date`)
   - `page:<wiki slug>` members of a wiki page (Phase 6)
   - `media:<id>` one item and its wiki neighbours (Phase 6)

   `scope_media_ids(conn, scopes) -> set[int] | None` intersects them (None
   means no restriction). EXCLUDED media are never returned by any function
   in this module.
2. `fts_query(user_text) -> str`: keep double-quoted spans as phrases; other
   tokens of 2+ characters are quoted; the last token gets a trailing `*`
   (prefix match, so results stream while the word is half typed). Join with
   spaces (AND).
3. `lexical(conn, q, k=200, allowed=None) -> list[(doc_id, score)]` using
   `bm25(doc_fts)`; if fewer than 5 hits, retry joined with `OR`.
4. `dense(client, cfg, q, k=200, allowed_doc_ids=None)` via `embed`.
5. `fuse(lists, k=60)`: reciprocal rank fusion.
6. `group_by_media(conn, fused, limit, offset) -> list[Hit]`: best doc per
   media; `Hit` has `media_id, doc_id, doc_kind, score, snippet, start_s,
   end_s, page`. The snippet is built with FTS5 `snippet(doc_fts, 0, '\x02', '\x03', '…', 24)`
   for lexical hits (the UI turns `\x02…\x03` into `<mark>` after escaping)
   and the first 240 characters for dense-only hits.
7. `lenses(conn, media_ids: list[int]) -> dict`: over the top 500 media:
   top 20 entities (with type and count), top 20 tags, top 20 folders, counts
   by kind and status, counts by month, and (Phase 6) top 10 events, topics
   and themes.
8. `correlations(conn, media_ids) -> list`: entities over-represented in the
   result set: `lift = (count_in_results / n_results) / (count_overall / n_items)`,
   require `count_in_results >= 3`, sort by `lift * log(count_in_results)`,
   top 15. These are the "cross-correlated" suggestions that are not already
   literal matches.
9. `item_card(conn, media_id) -> dict`: title (digest title when Phase 6 has
   run, else the first sentence of the caption or legacy caption, max 90
   characters, else filename), kind, status, `thumb_url`, `media_url`, date,
   folder, sensitivity, source URL.

**Verify:** `tests/test_search.py`: in-memory DB with five media and docs;
prefix matching (`"missi"` finds "missiles"); phrase queries; scope
intersection (entity plus date); fusion ranks a doc found by both lists
first; EXCLUDED media never appear; correlations surface a planted entity.

**Commit:** `Phase 3.3: hybrid search with scopes, lenses and correlations`

### Task 3.4: Streaming search endpoint

**Do:** `re2/api.py`, `app = FastAPI()`, started by `re2 serve` (uvicorn,
host and port from config; add `[server] allow_hosts` later only if the
human asks for remote access).

`GET /api/stream?q=<text>&s=<scope>&s=<scope>...` returns
`application/x-ndjson`, one JSON object per line, flushed as each is ready:

1. `{"t":"scope","scopes":[{"s":"entity:12","label":"Justin Trudeau"}],"n_in_scope":123}`
2. `{"t":"lexical","hits":[card+hit ...]}` (top 40; target under 40 ms)
3. `{"t":"lenses","data":{...}}` computed over the lexical hits
4. `{"t":"pages","hits":[...]}` wiki pages matching (empty until Phase 6)
5. `{"t":"fused","hits":[...]}` after the dense query (top 40), replacing
   the lexical list
6. `{"t":"lenses","data":{...}}` recomputed over fused hits
7. `{"t":"correlations","data":[...]}`
8. `{"t":"done","ms":{"lexical":12,"dense":48,"total":95}}`

With an empty `q`, stream the scope's items by date (newest first) instead
of steps 2 to 5. If the embedding server is down, skip step 5 and send
`{"t":"warn","msg":"dense search unavailable"}`; the endpoint must never
fail because cube is unreachable. Use `StreamingResponse` over a generator
that runs blocking DB work in a thread (`starlette.concurrency.iterate_in_threadpool`).
One SQLite connection per request, opened read-only
(`file:...?mode=ro`, `uri=True`).

`GET /api/items?s=...&offset=&limit=` pages through a scope (for "load more").

**Verify:** `uv run re2 serve`, then
`curl -sN 'http://127.0.0.1:4001/api/stream?q=ballot' | head -c 2000` shows
the lines in order; `tests/test_api.py` uses `fastapi.testclient.TestClient`
over a temp DB and checks line order and the warn path (embedding URL
pointed at a closed port).

**Commit:** `Phase 3.4: streaming search endpoint`

### Task 3.5: Item, media and legacy routes

**Do:** add to `re2/api.py`:

| Route | Behaviour |
|---|---|
| `GET /api/item/{id}` | card, all paths (text only), status, sources, media_meta, tags, entities, docs grouped by kind (caption JSON parsed, OCR, legacy caption, transcript windows, keyframes, PDF pages), wiki pages containing it (Phase 6), `similar` (Task 3.6) |
| `GET /media/{id}/file` | `FileResponse` of the first readable path. Confinement: `os.path.realpath(path)` must start with `os.path.realpath(corpus_root) + os.sep`, else 404. Range requests must work (206 for video seeks). |
| `GET /media/{id}/thumb` | the thumbnail file, 404 if none |
| `GET /media/{id}/keyframe/{seq}` | the keyframe JPEG under `data/keyframes/` |
| `GET /api/status` | `cli.status_counts` as JSON |
| `GET /legacy/`, `/legacy/matrix`, `/legacy/spicy`, `/legacy/concept-map` | the old `public/*.html` pages, unchanged. Mount `StaticFiles(directory="public")` at `/legacy/static` and `node_modules/three` at `/node_modules/three`; add the old JSON routes `/data`, `/tags`, `/tag_pairs_with_weights` serving the existing files in `data/`; `/hash-to-path/{sha256}` returns `{"url": "/media/{id}/file"}`; `/node-info/{sha256}` renders `templates/node_info.html` (Jinja2 autoescape) |
| `GET /` | `dashboard/index.html` (Phase 8); until it exists, a plain page linking to `/legacy/` |

The old pages reference assets by relative path (`script.js`, `styles.css`,
`/data`, ...). Serve `public/` at `/legacy/` so relative asset paths resolve;
fix any absolute `http://localhost:4001` or `http://bestiary:8000` URLs in
`public/*.js` to relative ones (the tag search that posts to bestiary is
removed: it points at a host that no longer exists).

Delete `server.js` and `hashMapper.js` in this task. Keep `package.json` for
`node_modules/three`.

**Verify:**
- `curl -s -o /dev/null -w '%{http_code}\n' --path-as-is 'http://127.0.0.1:4001/media/1/../../../etc/passwd'` → 404
- `curl -s -o /dev/null -w '%{http_code}\n' -r 0-100 http://127.0.0.1:4001/media/<video id>/file` → 206
- `/legacy/` and `/legacy/matrix` render as they did under Node.
- `tests/test_api.py`: confinement with a symlink pointing outside the corpus root → 404.

**Commit:** `Phase 3.5: FastAPI replaces server.js; legacy pages under /legacy`

### Task 3.6: Similar items

**Do:** `search.similar(conn, media_id, k=12)`: average the index vectors of
the media's docs (caption, OCR, transcript windows), search the `docs` index,
group by media, drop the media itself. Included in `/api/item/{id}`.

**Verify:** a test with planted vectors; manually, similar items for a
screenshot of a news story are about the same story.

**Commit:** `Phase 3.6: similar items`

---

## Phase 4: Video

### Task 4.1: Probe and keyframes (local)

**Do:** `re2/video.py`.

1. `probe(path) -> dict` with `ffprobe -v error -print_format json -show_format -show_streams`
   (argument list). Store `duration`, `width`, `height`, `has_audio` on `media`.
2. `keyframes(media_id, path, out_dir) -> list[(seq, t_s, file)]`: frames at
   `t = min(1.0, duration/2)` and then every 15 s, at most 8, each written with
   `ffmpeg -v error -y -ss <t> -i <path> -frames:v 1 -vf "scale='min(1024,iw)':-2" -q:v 3 <file>`
   to `data/keyframes/<id % 1000:03d>/<id>_<seq:02d>.jpg`. Compute
   `imagehash.phash`; drop a frame within Hamming distance 6 of a kept frame.
   Insert `keyframe` rows. Also set `media.thumb_path` from the first kept
   frame through `thumbs.make_thumbnail` at the sharded thumbnail path.
3. Run over `kind='video'` READY media with `duration IS NULL`, through
   `runner.run_jobs` with concurrency 4.

CLI `re2 video-probe [--limit N]`; script `scripts/video_probe.sh` (local
CPU and NFS only).

**Verify:** `--limit 5`; JPEGs exist; `SELECT media_id, COUNT(*) FROM keyframe GROUP BY 1 LIMIT 5;`.
The agent may run the full job in the background.

**Commit:** `Phase 4.1: ffprobe and keyframes`

### Task 4.2: ASR (local 1080 Ti)

**Do:** `re2/asr.py`.

1. Before importing `faster_whisper`, preload the CUDA libraries from the
   `nvidia.cublas` and `nvidia.cudnn` pip packages with
   `ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)` for every `lib/*.so*`
   (ctranslate2 cannot find them otherwise).
2. `transcribe(model, media_id, path)`: `ffmpeg -v error -y -i <path> -vn -ac 1 -ar 16000 data/audio/<id>.wav`;
   `model.transcribe(wav, beam_size=1, vad_filter=True)`; consume the
   generator; delete the wav in a `finally`. Store `language` in `media_meta`.
3. Insert `transcript_segment` rows, then transcript windows as `doc` rows
   (`kind='transcript'`, `seq=n`, `start_s`, `end_s`, text) grouping
   consecutive segments up to 60 s or 600 characters, all in one transaction.
4. No audio or empty transcript: `db.upsert_failure(conn, 'asr', id, None, 'no_speech', attempts=99)`.
5. One model per process, single worker. Use `runner.run_jobs` with
   concurrency 1 so Ctrl-C behaves like every other job.

CLI `re2 asr [--limit N]`; script `scripts/asr.sh` (header: uses the local
GPU only; ~6 to 8 hours for ~140 hours of audio; safe to run while cube is
extracting images).

**Verify:** `--limit 3`; segments and windows present; a window's text
matches the audio at its `start_s` (spot check one).

**Commit:** `Phase 4.2: faster-whisper transcription`

### Task 4.3: Keyframes through the vision model (cube)

**Do:** in `re2/extract_images.py`, `run_keyframes(conn, cfg, limit)`: for
every `keyframe` without a `doc(kind='keyframe', seq=keyframe.seq)`, call
`vlm.extract` with `KEYFRAME_SCHEMA`, user text
`Catalogue this video frame taken at {t_s:.0f} seconds.`, `max_tokens=900`.
Insert `doc(kind='keyframe', seq, start_s=t_s, end_s=t_s, text=description + "\nText: " + ocr_text, json)`,
tags (`origin='vlm'`) and entities. Failures: stage `vlm_keyframe`,
`doc_id=keyframe.id`.

CLI `re2 extract-keyframes [--limit N]`; script `scripts/extract_keyframes.sh`
(header: loads the gemma vision model; roughly 20k frames; several days at
3 per minute).

**Verify:** `--limit 5`.

**Commit:** `Phase 4.3: keyframe extraction`

---

## Phase 5: PDFs

### Task 5.1

**Do:** `re2/pdf.py`: for each READY PDF without `pdf_page` docs, open with
PyMuPDF (`pymupdf`). For each page up to 300: `text = page.get_text()`; if
`len(text.strip()) >= 50` insert `doc(kind='pdf_page', seq=page_no, page=page_no, text)`.
Otherwise render at 150 dpi to JPEG and add the page to a queue table
`pdf_ocr_queue(media_id, page)` (create in `db.SCHEMA`). Make a thumbnail
from page 1.

`re2 pdf-ocr [--limit N]` runs the queue through `vlm.extract` with
`IMAGE_SCHEMA` and stores `ocr_text` as the page doc text plus the JSON.

CLI `re2 pdfs`, `re2 pdf-ocr`. The UI opens `/media/{id}/file#page=N`.

**Verify:** `SELECT media_id, COUNT(*) FROM doc WHERE kind='pdf_page' GROUP BY 1;` covers the text-layer PDFs; `re2 pdf-ocr --limit 2`.

**Commit:** `Phase 5: PDF pages`

---

## Phase 6: Compiled knowledge (the wiki)

Runs after image extraction and ASR, and preferably after keyframes. It uses
the router, so its long steps are human-started scripts.

The hierarchy is **theme → topic → event → item**, plus **entity** pages
across it and one **overview** page for the whole archive.

### Task 6.1: Schema additions

**Do:** append to `db.SCHEMA` (idempotent):

```sql
CREATE TABLE IF NOT EXISTS item_digest (
  media_id INTEGER PRIMARY KEY REFERENCES media(id),
  title TEXT NOT NULL,                 -- <= 12 words
  summary TEXT NOT NULL,               -- 1 to 3 sentences
  item_date TEXT,                      -- YYYY-MM-DD, YYYY-MM or YYYY
  date_source TEXT NOT NULL,           -- source | content | filename | mtime | none
  model TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS entity_alias (
  alias_id INTEGER PRIMARY KEY REFERENCES entity(id),   -- merged-away entity
  canonical_id INTEGER NOT NULL REFERENCES entity(id)
);
CREATE TABLE IF NOT EXISTS cluster (
  id INTEGER PRIMARY KEY,
  level TEXT NOT NULL,                 -- theme | topic
  parent_id INTEGER REFERENCES cluster(id),
  run_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cluster_member (
  cluster_id INTEGER NOT NULL REFERENCES cluster(id),
  media_id INTEGER NOT NULL REFERENCES media(id),
  PRIMARY KEY (cluster_id, media_id)
);
CREATE TABLE IF NOT EXISTS wiki_page (
  id INTEGER PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,           -- e.g. event/2024-11-trudeau-ukraine-long-range-missiles
  kind TEXT NOT NULL,                  -- event | topic | theme | entity | overview
  title TEXT NOT NULL,
  summary TEXT NOT NULL,               -- one or two sentences, shown in lists
  body_md TEXT NOT NULL,               -- Markdown with [m:ID] citations and [[slug]] links
  date_start TEXT, date_end TEXT,
  cluster_id INTEGER REFERENCES cluster(id),
  entity_id INTEGER REFERENCES entity(id),
  parent_slug TEXT,
  member_hash TEXT NOT NULL,           -- sha1 of sorted member ids + child summaries
  stale INTEGER NOT NULL DEFAULT 1,
  model TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS wiki_member (             -- items a page is about
  page_id INTEGER NOT NULL REFERENCES wiki_page(id),
  media_id INTEGER NOT NULL REFERENCES media(id),
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
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
  title, summary, body_md, content='wiki_page', content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);
-- plus the same three insert/delete/update triggers as doc_fts
```

`search.scope_media_ids` gains `page:` and `media:`; `lenses` gains events,
topics and themes; `/api/stream` step 4 searches `wiki_fts` plus the `pages`
dense index.

**Commit:** `Phase 6.1: wiki schema`

### Task 6.2: Item dates (no model)

**Do:** `re2/dates.py`, `item_date(conn, media_id) -> (date, source)`, first
match wins:

1. `source`: earliest `source.date` for the media (truncate to `YYYY-MM-DD`).
2. `filename`, from any of the media's paths:
   - 13-digit Unix milliseconds `^(1[2-7]\d{11})\.` (4chan-style; 935 files)
   - `screenshot_YYYYMMDD_HHMMSS`, `Screenshot_YYYYMMDD-...`,
     `Screenshot from YYYY-MM-DD`, `screenshot-YYYY_MM_DD`
   - leading `YYYY-MM-DD` (e.g. `2020-06-11_15-32.png`)
   - 10-digit Unix seconds in a name, only if the date is between 2008 and today
3. `content`: the caption JSON `date_hints` or OCR text, parsed with
   `dateutil.parser.parse(fuzzy=False)` against these regexes only:
   `\b(20\d\d)-(\d\d)-(\d\d)\b`, `\b(Jan|Feb|...|Dec)[a-z]* \d{1,2},? (20\d\d)\b`,
   `\b\d{1,2} (Jan|...|Dec)[a-z]* (20\d\d)\b`. Take the latest plausible date
   (posts show the post date after quoted older dates). Reject dates after
   the file's mtime or before 2005.
4. `mtime`: the earliest `media_path.mtime_ns` as a date. This is weak;
   the UI shows mtime-sourced dates in `--ink-3` with a `~` prefix.
5. `none`.

Add dependency `python-dateutil`. Written into `item_digest` by Task 6.3; a
standalone `re2 dates` command fills a temporary `item_date` column on
`media_meta` (add `item_date TEXT, date_source TEXT` to `media_meta`) so the
timeline works before digests exist.

**Verify:** `tests/test_dates.py` with one case per rule, including the
"latest plausible date" rule and mtime rejection. Then
`SELECT date_source, COUNT(*) FROM media_meta GROUP BY 1;`.

**Commit:** `Phase 6.2: item dating`

### Task 6.3: Item digests (bulk text model)

**Do:** `re2/digest.py`. For every media with a caption, keyframe,
transcript, PDF page or legacy caption doc and no `item_digest` row, send
the digest model (`[wiki] digest_model` in config:
`qwen-3.6-35b-a3b-q8-256k-cube`, concurrency 3, `max_tokens` 400,
temperature 0.2, `chat_template_kwargs: {"enable_thinking": false}`) this
input, truncated to 6,000 characters:

```
Kind: <image|video|pdf>   Date: <item_date or unknown>   Folder: <folder>
Source URL: <url or none>
Description: <caption description or legacy caption>
Text in image: <ocr_text>
Claims: <claims joined by "; ">
Transcript (first 2,500 chars): <...>
Keyframes: <keyframe descriptions joined by " | ">
Entities: <entity names>
```

Schema: `{"title": str, "summary": str, "event_hint": str}`. System prompt:

```
You index items in a private archive of political and social-media material. For the item described, write:
- title: at most 12 words, specific and factual, naming the key people, places and events (e.g. "Trudeau says Ukraine may use Canadian long-range missiles in Russia"). No clickbait, no editorialising.
- summary: one to three sentences stating what the item shows and claims. Attribute claims to their source ("the post claims ..."). Do not soften offensive content or add warnings.
- event_hint: a short name for the specific real-world event or story this item is about, or "none" if it is not about a specific event.
```

Store `title`, `summary`, date (from `dates.item_date`), `model`. Keep
`event_hint` in a `digest_json` column (add it to the schema). Then embed
`title + "\n" + summary + "\nEntities: " + names` into a new index
`digests` (ids are media ids).

CLI `re2 digest [--limit N]`; script `scripts/digest.sh` (header: loads
qwen-3.6-35b-a3b; ~17k items; measure the rate on `--limit 20`).

**Verify:** `--limit 20`; titles are specific; `tests/test_digest.py` covers
input assembly and truncation with a mocked model.

**Commit:** `Phase 6.3: item digests`

### Task 6.4: Entity merging

**Do:** `re2/entities.py`, `merge(conn, cfg)`:

1. Key each entity name: casefold, strip punctuation except `&`, remove
   leading titles (`president`, `vice president`, `sen`, `senator`, `rep`,
   `dr`, `mr`, `mrs`, `ms`, `prime minister`, `pm`, `gov`, `governor`),
   collapse spaces. Same type and same key → merge automatically (the
   entity with the most items becomes canonical).
2. Candidate pairs (same type): one key's tokens are a subset of the
   other's (e.g. `trump` / `donald trump`), or the name embeddings
   (`search_document: <name>`) have cosine ≥ 0.90. Only consider entities
   with ≥ 2 items.
3. Verify candidates with the digest model in batches of 40:
   schema `{"pairs": [{"a": int, "b": int, "same": bool}]}`, prompt: "For
   each numbered pair of names of the same type, answer whether they refer
   to the same real-world entity. When unsure, answer false." Each line
   shows both names, the type, and the three most common co-occurring
   entities of each (context disambiguates "Biden" the son from the father).
4. Write `entity_alias`, repoint `media_entity` rows to canonical ids
   (`INSERT OR IGNORE` then delete old), give every canonical entity a
   `slug` (add `slug TEXT UNIQUE` and `item_count INTEGER` to `entity`).

CLI `re2 merge-entities [--dry-run]` (dry run prints the merges).

**Verify:** dry run on the real DB shows sensible merges and no obviously
wrong ones in a sample of 50; `tests/test_entities.py` for keys and repointing.

**Commit:** `Phase 6.4: entity merging`

### Task 6.5: Topics, themes and the 2D map

**Do:** `re2/cluster.py`, `recluster(conn, cfg)`. Add dependencies
`umap-learn` and `scikit-learn`.

1. Load the `digests` index (all dated and undated items).
2. `umap.UMAP(n_components=15, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42)`,
   then `sklearn.cluster.HDBSCAN(min_cluster_size=8, min_samples=4)` → topic
   labels. Noise items (label -1) join the topic with the nearest centroid
   (cosine on the original 768-d vectors) if cosine ≥ 0.60, else stay
   unclustered.
3. Themes: `sklearn.cluster.AgglomerativeClustering(metric="cosine", linkage="average", n_clusters=None, distance_threshold=t)`
   on topic centroids, choosing `t` by bisection so there are 12 to 25 themes.
4. 2D map: a second UMAP (`n_components=2, n_neighbors=30, min_dist=0.1`) over
   all items; save `data/index/map2d.npy` (float32 N×2) and
   `data/index/map2d_ids.npy`.
5. Write `cluster` and `cluster_member` rows under a new `run_id`
   (timestamp); keep previous runs for slug matching, delete runs older than
   the last two.

CLI `re2 recluster`. Local CPU only (UMAP on 17k points takes a few minutes).

**Verify:** `SELECT level, COUNT(*) FROM cluster WHERE run_id=(SELECT MAX(run_id) FROM cluster) GROUP BY 1;`
shows roughly 100 to 400 topics and 12 to 25 themes. Print ten topics with
five member titles each for a human sanity check.

**Commit:** `Phase 6.5: topics, themes, map`

### Task 6.6: Events

**Do:** in `re2/wiki.py`, `split_events(conn, cfg, topic_id)`: send the
writing model (`qwen38-27b-heretic-uncensored-q8-cube`, temperature 0.3,
thinking off) the topic's items sorted by date, one line each:
`<n> | <date or ?> | <title> | <entities, max 5> | hint: <event_hint>`,
at most 400 lines per call (split larger topics by date into consecutive
chunks of 400). Schema:

```json
{"events": [{"title": "string", "summary": "string", "members": [1, 2, 3]}],
 "not_events": [4, 5]}
```

Prompt: group the lines into specific real-world events or stories (same
incident, statement, controversy or news cycle, normally within a few weeks);
every line number appears exactly once in `events[].members` or in
`not_events`; an event needs at least 2 members; titles follow the digest
title style.

Validate: drop unknown numbers; assign any missing number to `not_events`;
if a number appears twice keep the first. Create or update
`wiki_page(kind='event')` rows with members, `date_start`/`date_end` from
member dates (ignore `mtime` dates unless all are mtime), `parent_slug` =
the topic page slug, `stale=1`.

Slugs: `event/<YYYY-MM>-<slugified title, max 60 chars>` (`undated` when no
dates), `topic/<slugified label>`, `theme/<...>`, `entity/<slug>`,
`overview`. On a recluster, a new cluster or event whose member set has
Jaccard ≥ 0.5 with an existing page's members keeps that page's slug and
row.

**Verify:** run on three topics with `re2 wiki-events --topic <id>`; every
member is accounted for; event titles read as specific stories.

**Commit:** `Phase 6.6: event splitting`

### Task 6.7: Writing pages

**Do:** `re2/wiki.py`, `build(conn, cfg, kinds, limit)` writes every page
with `stale=1`, in the order event → topic → theme → entity → overview (a
page's parents become stale when it changes). Same model and settings as
6.6, `max_tokens` 2,500, and **no JSON schema**: the output is Markdown.

Inputs per kind (keep each prompt under 60k tokens; truncate item text
first):
- **event**: each member as `[m:ID] <date> <title>` + summary + OCR (first
  800 chars) + transcript excerpt (first 1,500 chars) + source URL.
- **topic**: its event pages' titles, dates and summaries (`[[slug]]`), plus
  titles of members not in any event.
- **theme**: its topic pages' titles and summaries.
- **entity** (entities with ≥ 8 items after merging): the entity's events
  (titles, dates, summaries) and up to 60 of its items' titles and summaries,
  with the co-occurring entities list.
- **overview**: theme titles and summaries, counts by year and kind.

System prompt (shared):

```
You are writing a page of a private wiki that documents what a personal archive of political and social-media material contains. Readers use it to recall what they saved and to find the original items.

Rules:
- Write in plain, neutral encyclopedic English. Describe what the archived items show and claim; attribute claims to their sources ("a post by @user claims ..."). Say when items contradict each other.
- Every sentence that states something from the archive ends with one or more citations in the form [m:1234], using only the item ids provided. Link to other pages only with [[slug]] using only the slugs provided.
- You may add brief, clearly general background (who a person is, when an event happened) without citations, but never invent details about the archived items.
- Do not moralise, add warnings, or soften offensive material; quote it where it matters.
- Structure: a one-paragraph lead, then sections with ## headings as the material warrants (for events: What happened, Claims and framing, Timeline, Related). End with no conclusion or summary section.
- Length: events 250-700 words, topics and entities 300-900, themes 400-1000, overview 600-1200.
```

After generation:
1. Remove `[m:ID]` markers whose id was not in the input and `[[slug]]`
   links to unknown slugs (keep the link text as plain text).
2. First paragraph (markers stripped, max 300 chars) becomes `summary`.
3. Rewrite `wiki_citation` and `wiki_link`; set `member_hash`, `stale=0`.
4. Embed `title + summary` into index `pages` (ids are page ids).

CLI `re2 wiki-build [--kind event] [--limit N]`; script `scripts/wiki_build.sh`
(header: loads qwen38-27b; thousands of calls; resumable).

**Verify:** `--kind event --limit 3`; read the three pages; every sentence
about the archive carries a valid citation.
`tests/test_wiki.py`: citation and link validation, summary extraction,
stale propagation to parents.

**Commit:** `Phase 6.7: wiki page writing`

### Task 6.8: Incremental updates and export

**Do:**
1. `re2 wiki-update`: for items with a digest and no `wiki_member` row:
   nearest event by cosine to the event's mean digest vector, accepted if
   ≥ 0.75 and the item date is within the event's range ± 14 days (or
   undated); else nearest topic centroid if ≥ 0.65 (the item waits there);
   mark receiving pages stale. A topic with ≥ 10 waiting items reruns
   `split_events` for its undated-to-event items. Then run `wiki-build`.
2. `scripts/update_all.sh`: the full refresh chain, each step resumable:
   `scan → reconcile-sources → thumbs → extract-images → video-probe → asr →
   extract-keyframes → pdfs → embed → dates → digest → merge-entities →
   wiki-update → embed`. Header notes which steps load which cube model.
   `re2 recluster` is separate and occasional (monthly, or after large
   imports).
3. `re2 wiki-export [--out data/wiki]`: one Markdown file per page at
   `<out>/<slug>.md` with YAML front matter (`title`, `kind`, `date_start`,
   `date_end`, `updated`), `[[slug]]` links rewritten to
   `[[<slug>|<title>]]` (Obsidian style), and `[m:ID]` rewritten to numbered
   footnotes listing title, date, source URL and a dashboard link
   `http://127.0.0.1:4001/#/item/<ID>`. Write to a temp directory and swap it
   in.

**Verify:** add five fake new digests near an existing event in a test DB;
`wiki-update` attaches them and marks the event and its parents stale.
Open `data/wiki/overview.md` in a Markdown viewer.

**Commit:** `Phase 6.8: incremental wiki updates and Markdown export`

---

## Phase 7: Ask (chat agent)

### Task 7.1: Agent loop

**Do:** `re2/ask.py`. The agent is a JSON-schema action loop, not native tool
calling. That avoids depending on the router's chat-template and `--jinja`
settings, and grammar-constrained JSON never produces a malformed call.

Each step: POST to the writing model with `response_format` schema

```json
{"type": "object", "properties": {
  "thought": {"type": "string"},
  "action": {"type": "string", "enum": ["search", "find_pages", "open_page", "open_item", "timeline", "answer"]},
  "args": {"type": "object"}},
 "required": ["thought", "action", "args"], "additionalProperties": false}
```

Tools (Python functions over the DB and indexes; results are compact text
with stable reference ids):

| action | args | returns |
|---|---|---|
| `search` | `query`, optional `scopes` (list of scope strings), `limit` ≤ 20 | lines `[m:ID] date · kind · title — snippet (event: slug)` |
| `find_pages` | `query`, optional `kind` | lines `[[slug]] kind · dates · title — summary` |
| `open_page` | `slug` | page Markdown, truncated to 6,000 tokens |
| `open_item` | `id` | digest, caption fields, OCR, transcript windows with timestamps (truncated to 4,000 tokens), source URL |
| `timeline` | optional `scopes`, `from`, `to` | counts per month and event titles per month |
| `answer` | `{}` | ends the loop |

Rules: at most 10 tool steps; stop earlier when the context estimate
(characters / 3.5) exceeds 90k tokens; tool results are appended as user
messages `Tool result (<action>):\n...`. The user's dashboard scope stack is
passed as default `scopes` for `search`.

System prompt:

```
You are the research assistant for a private archive of political and social-media material (screenshots, memes, videos, documents) and a wiki compiled from it. You answer questions about world events and about what the archive contains.

Work step by step with the tools. Start broad (find_pages, search), then open the most relevant pages and items. Prefer wiki pages for overviews and items for specifics. Use timeline for "when" questions. Stop when you can answer.

You have general knowledge; use it to explain context. The archive is the user's own evidence: surface what it contains, including views and claims that differ from mainstream accounts, attributed to their sources. Never invent archive items or details.

Do not moralise, add warnings, or refuse because material is offensive.
```

Then a final **streamed** call without a schema: the conversation, the tool
results, and the instruction:

```
Write the answer now. Cite archive items as [m:ID] and wiki pages as [[slug]], using only ids and slugs that appeared in tool results. Put each citation directly after the sentence it supports. Sentences from general knowledge carry no citation.
Mode: <archive-only: answer only from the tool results | informed: combine general knowledge with the archive, and say clearly where the archive adds to, supports or contradicts the general account>.
```

Validation while streaming: buffer until a marker is complete, drop any
`[m:ID]` or `[[slug]]` not returned by a tool in this conversation, and
number the rest in order of first appearance. Sentences without a citation
are sent with `"general": true` so the UI styles them.

`chat_log` gets a new column `trace_json` (actions, args, result sizes,
timings) and `mode`. Conversations: add table
`chat_session(id TEXT PRIMARY KEY, created_at, title)` and `session_id` on
`chat_log`; history is the last 6 turns.

### Task 7.2: Ask endpoint

`POST /api/ask` with `{session_id?, question, scopes, mode}` returns
`text/event-stream`:
- `event: step`, data `{"action":"search","args":{...},"summary":"14 items"}`
- `event: token`, data `{"text":"...","general":false}`
- `event: cite`, data `{"n":1,"ref":"m:1234","card":{...}}` or `{"n":2,"ref":"p:event/...","page":{title,summary}}`
- `event: done`, data `{"session_id":"...","ms":...}`
- `event: error`, data `{"msg":"..."}` (for example when cube is busy loading another model)

**Verify:** `tests/test_ask.py` with a scripted MockTransport model:
unknown citations are dropped, known ones numbered, step cap enforced.
Live: `curl -sN -X POST ... -d '{"question":"What did Trudeau say about Ukraine and long-range weapons?","mode":"informed"}'`
cites the Disclose.tv clip in the `ukraine` folder with a timestamp.

**Commit:** `Phase 7: ask agent with validated citations`

---

## Phase 8: The Redpill Dashboard

### Design ethos (from `~/git/DEN`)

- **One file**: `dashboard/index.html` holds markup, styles and script. No
  framework, no build step, no CDN, no external fonts. Works offline.
- **Palette**: copy DEN's tokens verbatim ("two-ink riso palette on cool
  stock"): `--stock`, `--stock-well`, `--stock-card`, `--stock-bar`,
  `--rule`, `--rule-soft`, `--dot`, `--ink`, `--ink-2`, `--ink-3`, `--blue`,
  `--blue-w`, `--hot`, `--hot-w`, `--on-ink`, `--shadow`, `--shadow-lift`,
  the dark-mode overrides, and the `data-theme` light/dark overrides. Blue is
  the working ink; hot orange is scarce and means attention (current
  selection in visualisations, offline status, the cursor's focus rail).
- **Type**: `--font-display` (Iowan Old Style / Palatino serif) for titles
  and wiki headings; `--font-mono` for labels, counts, keys, dates and the
  omnibox; `--font-ui` for body text. `.label` style: mono 10.5px, uppercase,
  letter-spacing .13em, `--ink-3`.
- **Texture**: the dotted board background (`radial-gradient(var(--dot) 1px, transparent 1.3px)`, 15px),
  square corners, 1px rules, the brand mark (blue bar with a hot shadow).
- **Keyboard first**, fully usable with the mouse and on a phone.
- **Comments explain why**, in DEN's voice, sparingly.

### Task 8.1: Shell, routing and keys

**Do:** `dashboard/index.html`, served at `/` by `re2/api.py`.

Layout, desktop (≥ 1100px):
- Sticky top bar: brand mark + "Redpill" in the display face + a footnote
  (`17,137 items · updated <date>` from `/api/status`); the **scope stack**
  as chips; the **omnibox** (mono, fills the remaining width); mode buttons
  `SEARCH WIKI TIMELINE GRAPH MAP ASK` (`.btn`, pressed state blue); theme
  toggle.
- Three columns under it:
  - **Lenses** (280px, `--stock-well`): sections streamed from
    `/api/stream`: Correlated, Entities, Events, Topics, Themes, Folders,
    Dates (month sparkline), Kind/Status. Each row: name, mono count,
    a thin bar proportional to count.
  - **Results** (flex): wiki page hits first (display-face titles with
    summary), then item cards: thumbnail (blurred when sensitivity >
    threshold), title, snippet with `<mark>`, mono meta line
    (`IMAGE · 2024-11-18 · race/ · 0.6`), status badge (`OFFLINE` in hot,
    `THUMB ONLY` in hot outline), `mm:ss–mm:ss` for transcript hits, page
    number for PDF hits.
  - **Reader** (min 440px, `--stock-card`): the open item or page.
- Tablet (700–1099px): lenses become a horizontally scrolling strip of
  chips above results; the reader is a drawer from the right (`--shadow-lift`).
- Phone (< 700px): one column; the reader is a full-screen sheet with a close
  button; mode buttons move to a bottom bar that respects
  `env(safe-area-inset-bottom)`; the omnibox stays sticky at the top.

Routing in `location.hash`, so back and forward work and every view can be
linked: `#/search?q=...&s=entity:12&s=date:2024..2024-06`, `#/item/123`,
`#/wiki/event/2024-11-...`, `#/timeline?s=...`, `#/graph?s=...`,
`#/map?s=...`, `#/ask/<session_id>`.

Keys (normal mode, when no text field has focus; the omnibox gets them after
`Esc`). Counts work on motions (`3j`). A key panel toggled by `?` lists all
of them, grouped the way DEN's panel is:

| Keys | Action |
|---|---|
| `j` `k` / arrows | cursor down / up in the focused column |
| `h` `l` | focus column left / right (lenses ↔ results ↔ reader) |
| `gg` `G` | first / last in column |
| `Ctrl-d` `Ctrl-u` | half page down / up |
| `Enter` `o` | open the item or page under the cursor in the reader |
| `>` | **narrow**: push the thing under the cursor (entity, event, topic, theme, folder, month, tag) as a scope |
| `<` | pop the last scope |
| `/` | focus the omnibox (type to stream results) |
| `Enter` in the omnibox | push the typed text as a `text:` scope and clear the box (search within results) |
| `Backspace` in an empty omnibox | pop the last scope |
| `*` | narrow to the entity under the cursor, or the most prominent entity of the item under the cursor |
| `n` `N` | next / previous match highlight in the reader |
| `[` `]` | previous / next result while the reader is open |
| `Space` | play / pause the reader's video |
| `←` `→` in reader | seek 5 s; `t` jumps to the hit's timestamp |
| `x` | reveal / blur the item under the cursor; `X` sets the blur threshold |
| `yy` | copy a citation for the item or page (`title — date — URL — dashboard link`) |
| `gs` `gw` `gt` `gn` `gm` `ga` | modes: search, wiki, timeline, graph (network), map, ask |
| `a` | ask about the current scope (opens ask mode with the scopes attached) |
| `?` | key panel |
| `Esc` | close the key panel, then clear the count, then leave the omnibox, then close the reader, then clear the query. Never pops scopes. |

The cursor is shown as DEN does: a hot left rail on the focused row or card
and `--shadow-lift`.

**Verify:** manual pass over every key; phone layout at 390px wide in a
browser's device mode; no horizontal page scroll at any width.

**Commit:** `Phase 8.1: dashboard shell, routing and keys`

### Task 8.2: Streaming search and reader

**Do:**
- Omnibox input: debounce 90 ms; each request aborts the previous one
  (`AbortController`); read `/api/stream` with `fetch` +
  `response.body.getReader()`, split on newlines, render each section as it
  arrives. Sections from an older query are dimmed (`opacity:.55`) until
  replaced, never cleared first (no flicker). When `fused` replaces
  `lexical`, keep the cursor on the same media id if it is still present.
- Empty omnibox and no scopes: the home view shows the overview page summary
  (Phase 6, otherwise corpus counts), recent additions, top entities and the
  corpus timeline.
- Reader for an item: large preview (`<img>` from `/media/{id}/file` or the
  thumbnail when offline; `<video controls preload="metadata">` seeking to
  `hit.start_s`; PDFs as a link to `/media/{id}/file#page=N`), title, source
  URL, dates with their source, paths as text, entities and tags as chips
  (click or `>` narrows), caption fields (description, OCR in `<pre>`,
  meaning, claims list, intent, humor), transcript windows as timestamped
  lines that seek, keyframe strip with `t_s` labels that seek, wiki pages
  containing the item, similar items.
- Reader for a wiki page (Phase 8.3).
- **All untrusted strings go through `escapeHtml`** before any `innerHTML`;
  snippets are escaped first and only then have `\x02`/`\x03` replaced with
  `<mark>`/`</mark>`. Filesystem paths are shown as text, never as links.
- "Load more" at the end of results (and `G` at the last card) fetches
  `/api/items` with the next offset.

**Verify:** type "trud", "trudea", "trudeau ukr" and watch results stream and
refine; `>` on the Ukraine entity, then type "missile"; `<` restores; open a
video transcript hit and confirm it plays from the timestamp; search for
`<script>alert(1)</script>` and see it rendered as text.

**Commit:** `Phase 8.2: streaming search and reader`

### Task 8.3: Wiki mode

**Do:** server-side Markdown rendering in `re2/api.py` with
`markdown-it-py` (dependency; `html=False`, so raw HTML in model output is
escaped). Before rendering, replace `[m:ID]` with
`<a class="cite" data-media="ID" href="#/item/ID">n</a>` numbered per page
and `[[slug]]` with `<a class="wikilink" href="#/wiki/slug">Title</a>`.
`GET /api/page/{slug:path}` returns `{title, kind, dates, html, breadcrumbs
(theme › topic › event), children, members (cards), backlinks, cited items
(cards)}`.

Dashboard: wiki mode shows the overview page; theme and topic pages list
children as cards; page text in the display face at a readable measure
(~68ch). Citation chips open the cited item in the reader (in wiki mode the
page moves to the results column and the item opens beside it). `j`/`k`
move between links and citations in the page; `Enter` follows; `>` narrows
search to the page's members.

**Commit:** `Phase 8.3: wiki mode`

### Task 8.4: Visualisations

All canvas-based, in the same file, redrawn on theme change, driven by the
current scope stack, with keyboard cursors.

1. **Timeline** (`GET /api/timeline?s=...` → months with counts by kind, and
   events with date ranges): stacked month bars in `--ink-3` / `--blue`,
   event ranges as thin blue spans above; the cursor month in hot. `h`/`l`
   move by month (`H`/`L` by year), `v` starts a range selection, `>` pushes
   a `date:` scope, `Enter` lists that month's items in results.
2. **Graph** (`GET /api/graph?s=...&limit=200` → nodes = top entities in
   scope with counts; edges = co-occurrence in the same items with
   `n_ab >= 3`, weight = normalized PMI
   `log(p_ab / (p_a p_b)) / -log(p_ab)`): force layout computed in the page
   (simple velocity-Verlet, 300 iterations, then frozen; drag to adjust).
   Node size by count, entity type as a small mono glyph (`P` `O` `L` `E`
   `S`), edges in `--rule`, the cursor node and its edges in hot.
   `j`/`k` cycle nodes by degree, `Enter` opens the entity page, `>` narrows.
   This replaces the old tag graph and matrix.
3. **Map** (`GET /api/map` → `map2d` points `[x, y, topic_id, media_id]`,
   cached; `GET /api/map/labels` → topic label positions): every item as a
   2px dot in `--ink-3`; items in the current scope in blue; the cursor item
   in hot; topic labels in the display face at cluster centroids, fading
   with zoom. Wheel or `+`/`-` zoom, drag or `hjkl` pan, `Enter` on a label
   narrows to the topic, hovering a dot shows its title. This replaces the
   old "spicy" 3D view.

**Verify:** each view reacts to scope changes; performance: map redraw under
16 ms with 17k points on this machine.

**Commit:** `Phase 8.4: timeline, graph and map`

### Task 8.5: Ask mode

**Do:** ask mode replaces the results column with the conversation and keeps
the reader. The input sits at the bottom (`Enter` sends, `Shift-Enter` new
line). Each answer shows the agent's steps as mono lines as they stream
(`SEARCH trudeau ukraine long-range → 14 items`), then the answer text as it
streams. Citations render as numbered chips that open the item or page in
the reader (videos seek to `start_s`). Sentences marked `general` render in
`--ink-2` with a small `GK` label on hover. A toggle (and `m` key) switches
`archive-only` / `informed` (default informed). The current scope stack is
shown above the input and sent with the question. Sessions are listed in
the lenses column.

**Verify:** ask a vague question ("the thing about Ukraine and the
long-range weapons"); steps stream; answer cites items; clicking a citation
opens the clip at the right second.

**Commit:** `Phase 8.5: ask mode`

---

## Phase 9: Retire the old system

### Task 9.1

Once the dashboard's graph and map work: remove the `/legacy` routes, delete
`public/` except `public/thumbnails/`, delete `node_modules/`,
`package.json` and `package-lock.json`, and move the pre-RE2 Python scripts
and templates listed in "Status" into `old/` with a README line saying they
are superseded. Rewrite `README.md` for RE2: what it is, the layer diagram,
setup (`uv sync`, config, embed server), the refresh chain script, and the
dashboard keys (link to the in-app `?` panel). Keep the demo video link.

Ask the human before this task; they may want to keep the old views.

**Commit:** `Phase 9: retire the pre-RE2 server and pages`

---

## Notes for the human (not for the coding agent)

- **Model swapping on cube.** Vision extraction (gemma), digests
  (qwen-3.6-35b-a3b), wiki writing and ask (qwen38-27b) and your opencode
  model all share the router. The embedding server on port 5010 is separate.
- **Faster vision extraction.** The 8k context of the batch gemma entry is
  shared by all requests, which caps concurrency at 3. An entry with more
  context would allow 8 concurrent requests and roughly 2–3x throughput:
  ```ini
  [gemma-4-31b-heretic-ara-batch-parallel8-q8-32k-cube]
  model = /home/offipso/ML/models/gemma/gemma-4-31b-it-heretic-ara-Q8_0.gguf
  mmproj = /home/offipso/ML/models/gemma/mmproj-F16.gguf
  chat-template-file = /home/offipso/ML/llama.cpp/models/templates/google-gemma-4-31B-it.jinja
  cache-type-k = q4_0
  cache-type-v = q4_0
  kv-unified = true
  ctx-size = 32768
  parallel = 8
  threads = 32
  ```
  Then set `vision_model` to that name and `vision_concurrency = 7` in
  `config.toml`. Extraction resumes where it stopped.
- **Remote access.** The server binds to 127.0.0.1. To use the dashboard
  from a phone over Tailscale, bind to the Tailscale address only, never
  0.0.0.0; the archive and the media route have no authentication.
- **Order of long jobs:** `extract_images.sh` (days, cube) in parallel with
  `asr.sh` (hours, local); `embed.sh` after each extraction session;
  `extract_keyframes.sh` (days, cube); `digest.sh`; `wiki_build.sh`. After
  that, `update_all.sh` keeps everything current.
