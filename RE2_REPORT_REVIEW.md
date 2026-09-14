# Review of RE2_INVESTIGATION_REPORT.md

Reviewed: 2026-09-14. Reviewer verified the report's claims against the repo, the
`~/Downloads/redpills` mount, the NAS exports, and the GPU host `cube`.

## Verdict

The report's central conclusions are correct and I concur with them:

- RAG is the right primary architecture. A QLoRA cannot reliably recall exact
  files, hashes, URLs or timestamps and should be a late, optional polish step.
- The manifest / file-status layer (READY, THUMBNAIL_ONLY, SOURCE_OFFLINE, ...)
  is the right first investment.
- Full-text search, OCR, and video transcripts are the highest-value retrieval
  additions, in that order of effort-to-payoff.
- Observed tag co-occurrence (PMI) should replace the LLM-judged weight matrix.
- The security items (HTML escaping, path traversal, localhost binding) are real.

The numbers in the report are accurate. I independently reproduced: 5,144 tagged
records, 5,987 unique tags, 41 case-collision groups, 7,021 stale hash-cache
entries dated 2025-06-02, 7,392 thumbnails, 3,126 broken symlinks, 3,718 videos,
28 PDFs, 1,327 archive_metadata entries, and 3,225 tagged records that still
resolve to a readable original.

Where the report falls short is not analysis but scoping: it lists every
possible improvement without choosing, and it is written as if the GPUs and the
code live on the same machine. Both matter a lot for a plan that a smaller
model will execute unattended.

## Corrections (things the report got wrong or missed)

1. **The GPUs are on a different host, and that host already serves an
   OpenAI-compatible API.** This machine has one GTX 1080 Ti and a CPU-only
   torch. The six RTX 3090s are on `cube` (reachable by SSH; 440 GB RAM, 128
   cores, 306 GB free NVMe). `cube` runs llama-swap on port 5000 with, among
   others: `qwen38-27b-heretic-uncensored-q8-cube` (the chat model), several
   **vision-capable** models (`gemma-4-31b-*`, `qwen-3.6-27b`, `qwen-3.6-35b-a3b`,
   `qwen38-flash-next`), and a `nomic-embed-text-v1.5` GGUF on disk. The report's
   proposal to keep writing bespoke torch/FAISS servers (CogVLM2, InternVL3)
   should be dropped: every model call in the new pipeline can be a plain HTTP
   request to llama-swap. This is the single biggest simplification available
   and it makes the work far more tractable for a smaller coding model.
   I could not find the RTX 5090 on `cube`; the plan should not depend on it.

2. **The broken symlinks are not recoverable from here.** I tested rewriting
   the `/mnt/hanoi_data/storage/redpills/` prefix onto the current mount and
   onto every other storeroom export: zero of 3,126 resolve. The NAS
   (`192.168.1.20`) no longer exports `/hanoi/data` at all, though it is still
   in `/etc/fstab`. The report's Phase 1 ("restore/reconcile missing paths")
   therefore collapses to: mark them SOURCE_OFFLINE, keep their thumbnails, and
   let the owner decide separately whether the old hanoi disk still exists.
   The losses are concentrated: race 1,451, covid-19 872, voter fraud 292,
   groomers 245, ukraine 137.

3. **OCR should come from the VLM, not tesseract.** For memes and social-media
   screenshots, a modern VLM asked to "transcribe all visible text verbatim"
   outperforms tesseract and costs nothing extra if it is part of the same
   structured-JSON captioning call (description, ocr_text, entities, tags,
   sensitivity). llama.cpp supports `response_format`/JSON schema, so the
   output is parseable without regex. This collapses the report's image steps
   2 through 5 into one call per image.

4. **The ASR estimate is pessimistic.** faster-whisper `large-v3` on a 3090 runs
   roughly 20 to 40 times realtime, so 107.6 hours of audio is a few GPU-hours,
   not 13 to 36. Whisper is not installed on `cube` (only locally, where it
   cannot use the GPU), so the plan must create a Python environment on `cube`.

5. **Vector store choice is over-engineered.** For under 100k vectors of 768
   dimensions, brute-force cosine similarity over a numpy matrix answers a
   query in milliseconds. No FAISS, LanceDB, Qdrant or HNSW is needed. SQLite
   (metadata + FTS5) plus `.npy` embedding matrices is the whole store, and it
   can be upgraded to `sqlite-vec` later without changing the schema.

6. **QLoRA dataset sizes are inflated.** A behaviour adapter (citation
   discipline, query reformulation, refusal on missing evidence) needs on the
   order of 1k to 3k high-quality examples, not 10k to 100k. More importantly
   the right training data is real query and retrieval logs from the working
   RAG system, so building the log table into the chat endpoint from day one
   is the actual QLoRA preparation step. `~/ML/unsloth-axolotl` already exists
   on `cube`.

7. **Missed defects in the repo:**
   - `images_captioned.json` has 5,503 records but only 5,144 were tagged; 359
     captions were never tagged.
   - `server.js` references a `templates/` directory that does not exist, so
     `/templates/concept-map` always returns 500.
   - `make_thumbnails.py` calls port 3000 and reads a `path` key that
     `/hash-to-path` no longer returns (it returns `url`); it is broken as-is.
   - `.gitignore` is a 696 KB file that enumerates `node_modules` file by file.
     It should be replaced with a normal one.
   - `/hash-to-path` hardcodes `http://localhost:4001` into returned URLs.
   - The UI's tag search posts to `http://bestiary:8000/search`, a host that
     is not reachable from here.
   - Local Python is 3.14, which many ML wheels do not yet support. All new
     Python work should pin 3.12 via `uv`.

8. **Recaptioning should be the default, not the exception.** The old CogVLM2
   captions lack OCR, entities and structured fields, and they came from a
   weaker 2024 model. One uniform pass over all ~10.2k readable images with
   the current VLM produces a consistent corpus; the old captions are kept as
   a `legacy_caption` field and are what the 1,919 thumbnail-only items keep.
   Those 1,919 can also be recaptioned from their 400 px thumbnails, flagged
   as low-resolution.

## Scope reductions for the executable plan

The report lists these; the plan should defer them until the search and chat
core works: face detection and clustering, speaker diarization, hierarchical
topic clustering with LLM naming, geographic map view, timeline view,
instanced-mesh rewrite of the 3D view, the multi-axis sensitivity taxonomy,
image-to-image (CLIP/SigLIP) embeddings, a reranker model. None of them are
wrong; they just do not move the "vague question, cited answer, click through
to source" goal.

## Recommended stack (one choice each, no alternatives)

| Concern | Choice | Why |
|---|---|---|
| Language for all new backend code | Python 3.12 via `uv` | One language for ingestion, search and chat; the `cube` env is Python too |
| Web server | FastAPI, replacing `server.js`; old JSON routes kept for the existing visualizations | Avoids a Node/Python split; `server.js` is only 316 lines |
| Metadata + lexical index | SQLite + FTS5 (stdlib) | Zero dependencies, one file, resumable |
| Dense vectors | `nomic-embed-text-v1.5` served by llama-swap; `.npy` matrix + numpy cosine | Model is already on `cube`; no ANN library |
| Captions, OCR, entities, tags | One structured-JSON call per image to a llama-swap vision model | Already served; drops tesseract and the custom servers |
| ASR | faster-whisper on `cube`, segment timestamps kept | Fastest practical option on 3090s |
| Keyframes | ffmpeg scene detection, capped per video | Already installed on both hosts |
| PDFs | PyMuPDF text layer, VLM on pages with no text | 28 files, small |
| Chat | Deterministic pipeline: rewrite query, hybrid search with RRF fusion, answer with citation IDs validated server-side | Simpler than agentic tool calling; citations cannot be hallucinated |
| UI | Search-first page: query box, facets, result grid, detail pane with source status, chat pane; existing graph/matrix/3D pages kept as secondary | Matches the report's Section 6 recommendation |

## Revised phase order

0. Repo hygiene and safety: `uv` project, `.gitignore`, config file for hosts
   and paths, escape HTML, confine `/images` with realpath, bind 127.0.0.1.
1. Manifest: scan `~/Downloads/redpills` read-only, classify by magic bytes,
   stream-hash once (cached by size and mtime), record every path per hash,
   record symlink targets and status, reconcile old captions, thumbnails,
   `archive_metadata.json` and `output.json`. Write status per item. Report.
2. Image pass: thumbnails for every READY image; structured VLM extraction for
   every readable image; legacy captions retained; FTS5 populated; tag
   normalization and PMI co-occurrence regenerated; old visualization JSON
   regenerated from the DB so existing pages keep working.
3. Search API and search-first UI over images (lexical + dense + facets).
   This is the first point at which the system is visibly better than today.
4. Video: ffprobe, ASR with timestamps, keyframes with VLM extraction, segment
   rows with start and end, player page that seeks to a segment.
5. PDFs.
6. Chat with enforced citations and a query log.
7. Optional: image embeddings, reranker, topic clustering, QLoRA from logs.

## Open decisions for the owner

1. Replace `server.js` with a Python FastAPI server (recommended), or keep
   Node for serving and add Python only for ingestion and search?
2. Recaption all ~10.2k readable images with the new VLM (recommended), or
   only the ~7k uncaptioned ones?
3. Which llama-swap vision model should the plan target by default? The
   `gemma-4-31b-heretic-ara-batch-parallel6-q8-8k-cube` entry looks purpose-built
   for batch captioning; `qwen-3.6-35b-a3b-q8-256k-cube` would be faster.
4. GPU sharing: llama-swap currently holds ~18 GB on each 3090. ASR and any
   future CLIP service need a few GB alongside, or a llama-swap TTL so the LLM
   unloads between batches. The plan will assume "run ASR while no LLM is
   loaded" unless told otherwise.
5. Is the old `hanoi` disk gone for good? If it might still exist, keep the
   SOURCE_OFFLINE items' expected relative paths so a future remount can
   recover them automatically.
