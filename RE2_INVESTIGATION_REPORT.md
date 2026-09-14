# RE2 / RedPill Corpus Investigation Report

Generated: 2026-09-14

Scope:

- `/home/offipso/storage/git/redpill_explorer_2`
- `/home/offipso/Downloads/redpills`
- current derived JSON/thumbnail artifacts
- source-file metadata and readable-media inventory

This was a read-only audit. No files under `~/Downloads/redpills` were modified.

---

## Executive summary

RE2 is currently a **local image-caption/tagging/visualization prototype**, not yet a full multimodal archive or semantic search system.

Its core pipeline is:

1. Scan images.
2. Use a vision model to generate descriptions.
3. Use an LLM to generate tags and a subjective “spicy” score.
4. Build JSON files from descriptions/tags.
5. Serve JSON and thumbnails to several D3/Three.js visualizations.

The strongest existing ideas are:

- content-hash identity instead of filename identity;
- rich natural-language image descriptions;
- local-first organization;
- tag browsing and relationship visualizations.

The biggest gaps are:

- most accessible media are not actually indexed;
- video audio/transcripts are completely unindexed;
- the main UI search is simple substring filtering, not semantic retrieval;
- current vector-search infrastructure is weak and not present in the working repository;
- many source files are now inaccessible because of broken symlinks from an apparent migration away from `/mnt/hanoi_data`;
- derived indexes are stale and inconsistent;
- there is no source-grounded chat interface.

The desired chatbot behavior — vague topic recall plus retrieval of original material — is feasible, but **not primarily through QLoRA**. The correct architecture is:

> **Retrieval-Augmented Generation, with optional LoRA/QLoRA for citation behavior, taxonomy, query reformulation, and tool use.**

A 27B QLoRA is feasible on the proposed GPU hardware, but finetuning alone is the wrong tool for exact archival recall and source linking.

---

# 1. What RE2 currently does

## Current architecture

The main server is `server.js`.

It currently:

- serves the static UI from `public/`;
- serves files under `/home/offipso/Downloads/redpills` through a custom `/images` route;
- exposes three JSON endpoints:
  - `/data`
  - `/tags`
  - `/tag_pairs_with_weights`
- maps SHA-256 hashes to filesystem paths through `hashMapper.js`;
- renders item details at `/node-info/:hash`.

The main graph UI loads all data into the browser:

```text
/tags
/data
/tag_pairs_with_weights
```

It then performs tag filtering and description substring filtering client-side.

## Current ingestion pipeline

The historical pipeline is:

### Vision captioning

Scripts such as:

- `caption_images_with_cogvlm2.py`
- `caption_images_with_intern-vl3.py`

process files with these extensions:

```text
.jpg
.jpeg
.png
```

They:

1. calculate the file’s SHA-256 hash;
2. send the image to a remote captioning endpoint;
3. store the resulting description in `images_captioned.json`.

The caption prompt asks for text, people, races, NSFW content, memes, jokes, political content, and controversial context.

### Tag generation

`tag_caption_output.py`:

- reads descriptions;
- sends each description to an LLM;
- asks for approximately 15 tags with 0.0–1.0 relevance scores;
- asks for a “spicy” score;
- writes `data/images_captioned_tagged.json`.

The tag vocabulary originally came from `available_tags.txt`, but the prompt allowed new tags, and the current data contains many tags outside that file.

### Tag relationships

The system has two related artifacts:

- `data/tag_pairs.json`
- `data/tag_pairs_with_weights.json`

The latter was populated by asking an LLM to assign semantic similarity weights between tag pairs.

### Visualization layer

Existing views:

- `/` — D3 force-directed tag graph.
- `/spicy` — Three.js 3D view with items arranged by spiciness level.
- `/matrix` — tag-pair relationship matrix.
- `/concept-map` — currently broken/experimental.
- `/node-info/:hash` — item detail page.

---

# 2. Current data quality and integrity

## Current caption database

`data/images_captioned_tagged.json` contains:

| Metric | Value |
|---|---:|
| Caption records | 5,144 |
| Unique filenames in caption DB | 5,142 |
| Missing descriptions | 0 |
| Missing spicy score | 1 |
| Tag annotations | 67,298 |
| Unique tags in current caption DB | 5,987 |
| Average tags per item | 13.1 |
| Total caption words | approximately 817,000 |

Approximate token estimate for the existing caption corpus:

```text
~1.0–1.2 million tokens
```

This does not include tags, source URLs, transcripts, PDFs, or future synthetic QA data.

## Tag-vocabulary mismatch

There is a major inconsistency:

| Artifact | Current item count | Current tag count |
|---|---:|---:|
| `images_captioned_tagged.json` | 5,144 | 5,987 |
| `adjusted_data.json` | 4,760 | 5,783 |
| `tags_with_sizes.json` | 4,757 item IDs | 528 |
| `tag_pairs_with_weights.json` | N/A | 631 |

Consequences:

- the graph and tag UI do not reflect the full current tag vocabulary;
- many newer tags are invisible to visualizations;
- the matrix uses only the older 631-tag vocabulary;
- the relationship data was generated by LLM judgment, not observed co-occurrence.

The current tag set also contains case collisions, for example tags such as:

```text
USA Today
usa today
```

and similarly for Fox News, 4chan, GIF, and others. These should be normalized unless case is intentionally meaningful.

Tag relevance scores also have limited dynamic range:

- minimum: approximately 0.1
- maximum: 1.0
- average: approximately 0.88
- median: approximately 0.90

So tag scores currently provide weak signal.

## Current vector database situation

`cogvlm2server/server.py` contains a CogVLM2-based service that can:

- generate captions;
- compute embeddings;
- write a FAISS index;
- store embeddings in SQLite;
- search by image, image path, image hash, or text.

However, in the current working tree no usable local vector index was found. Specifically, these files are absent:

```text
faiss_index.bin
index_hash_keys.pkl
image_data.db
```

The existing embedding method is also not ideal:

- image embeddings come from the hidden state of the prompt/image generation path;
- text-query embeddings come from text-only hidden states;
- generated captions are not explicitly embedded;
- the model is being repurposed as an embedding model without contrastive training.

That is not a modern retrieval embedding design.

## Stale path cache

`hash_path_cache.json`:

- contains 7,021 mappings;
- last update timestamp corresponds to approximately:

```text
2025-06-02
```

The current readable image corpus is much larger than what the cache reflects.

The mapper also has several weaknesses:

- it only recognizes:

```text
.jpg
.jpeg
.png
.gif
```

- it loads each whole file into memory before hashing;
- duplicate content maps to only one path;
- it does not maintain a full “one hash to many paths” inventory;
- broken symlinks and inaccessible targets are not explicitly modeled.

## Thumbnail coverage

There are 7,392 thumbnails in `public/thumbnails`.

Current correspondence:

| Relationship | Count |
|---|---:|
| Current caption records with thumbnails | 4,760 |
| Current caption records missing thumbnails | 384 |
| Thumbnails not matching current caption DB | 2,632 |

However, thumbnails are still useful: every one of the 1,919 caption records that currently lacks a readable original has a thumbnail.

That means previews exist for many otherwise offline items.

---

# 3. What percentage of the RedPills dataset is ingested?

The answer depends on what denominator is used.

## Accessible regular files under `~/Downloads/redpills`

Excluding `.Trash-1000`:

| Category | Accessible regular files | Size |
|---|---:|---:|
| Images | 10,223 | approximately 2.54 GiB |
| Videos | 3,718 | approximately 53.83 GiB |
| PDFs | 28 | approximately 0.21 GiB |
| Other/mixed | approximately 26 | small |
| Total | 13,995 | approximately 56.7 GiB / 60.87 GB |

There are also:

```text
3,126 broken symlinks
```

Their targets mostly point to:

```text
/mnt/hanoi_data/storage/redpills/...
```

That path does not currently resolve.

This strongly suggests a storage migration artifact. The current share appears to be on:

```text
192.168.1.20:/storeroom4/data/redpills
```

while many symlinks still refer to the old `hanoi_data` path.

## Image inventory

Accessible regular image-like files:

| Extension | Count |
|---|---:|
| `.jpg` | 4,830 |
| `.png` | 4,715 |
| `.jpeg` | 541 |
| `.webp` | 93 |
| `.jfif` | 23 |
| `.gif` | 21 |
| Total | 10,223 |

There are another 2,022 dangling image-like symlinks.

So historically/structurally there appear to be:

```text
12,245 image-like entries
```

but only 10,223 currently have readable image files.

## Current functional image coverage

By direct content-hash matching against currently readable image files:

| Measurement | Value |
|---|---:|
| Accessible image files | 10,223 |
| Accessible image files currently content-matched to captions | 3,234 |
| Functional accessible-image coverage | 31.6% |
| Unique readable image contents | approximately 10,064 |
| Unique readable image contents with caption records | approximately 3,225 |
| Functional unique-image coverage | approximately 32.0% |

In other words, only about one-third of the currently readable image corpus is actually source-resolvable through the existing caption DB.

There are roughly:

```text
6,989 accessible image files not currently indexed
```

by caption/content matching.

## Metadata-record coverage

The caption DB has 5,144 image records.

Against the historical image-like entry count:

```text
5,144 / 12,245 = 42.0%
```

So if broken symlinks are counted as known historical entries, the metadata covers about **42% of image entries**.

But of those 5,144 records:

- approximately 3,225 correspond to readable current image content;
- approximately 1,919 currently have no readable original image;
- 1,919 of those still have thumbnails;
- some captioned filenames also appear as broken symlinks.

So:

| Definition | Approximate ingestion |
|---|---:|
| Metadata records over historical image entries | 42% |
| Source-resolvable metadata over currently readable images | 32% |
| Source-resolvable records over all currently accessible regular files | 23% |

The most honest functional statement is:

> **RE2 currently contains searchable metadata for about 42% of known image-like entries, but only about 32% of currently readable image content can be matched to readable originals. Across the entire readable file corpus, only about 23% is source-resolvably ingested.**

## Video/audio coverage

Current ingestion:

```text
0%
```

Accessible video files:

```text
3,718
```

Approximate duration, measured with `ffprobe`:

| Metric | Value |
|---|---:|
| Probed files | 3,717 of 3,718 |
| Total duration | approximately 107.6 hours |
| Average duration | approximately 104 seconds |
| Files with audio streams | 3,313 |

There are no meaningful standalone audio files in the accessible corpus.

The audio content is almost entirely embedded inside video files.

None of the following has been ingested:

- speech transcripts;
- segment timestamps;
- video frame descriptions;
- on-screen OCR;
- keyframes;
- shot boundaries;
- video similarity embeddings;
- speaker diarization.

For a corpus where many items are video clips, this is probably the single largest retrieval gap.

## PDF and document coverage

Current ingestion:

```text
0%
```

Accessible PDFs:

```text
28
```

PDFs in the corpus can include:

- text layers;
- scanned pages;
- tables;
- screenshots;
- articles;
- reports;
- embedded images.

None are currently part of the retrieval pipeline.

## Other files

There are also:

- JSON manifests;
- TXT link lists;
- TSV browser-history-like files;
- HTML fragments;
- archives;
- scripts;
- partially downloaded files;
- credential-looking files.

Some are potentially useful for provenance, but some should be excluded from indexing.

Files that appear to contain credentials, such as cookie files, should be excluded from indexing and should not have their paths surfaced in UI search results.

---

# 4. Existing useful metadata sources

There is useful metadata already sitting in the RedPills directory.

## `archive_metadata.json`

This file contains approximately:

```text
1,325 entries
```

Each entry can include:

- date;
- source URL;
- screenshot filename;
- photo filenames;
- video filenames;
- folder;
- restricted status;
- retry status.

This is highly valuable because it provides provenance that the current RE2 UI does not use.

Approximate metadata referenced-file status:

| Status | Count |
|---|---:|
| Regular accessible files referenced | approximately 2,332 |
| Missing referenced files | 12 |

That is useful.

## Other manifests

There are also files such as:

```text
failed_urls_2026.txt
failed_screenshots_403_2026-09.json
failed_screenshots_403_2026-09_results.json
list_of_links_blm_violence.txt
```

These could inform re-ingestion, source reconciliation, or link recovery.

## Old `output.json`

There is another caption artifact:

```text
~/Downloads/redpills/output.json
```

It contains:

- approximately 3,610 records;
- hash keys;
- old absolute paths under `/mnt/hanoi_data/storage/redpills/...`;
- captions different from the current tag DB.

Those paths are not currently resolvable.

This file is useful as historical provenance, but it should not be blindly merged as current filesystem truth.

---

# 5. Current visualization and retrieval assessment

## What works

RE2 currently has some genuinely good ideas:

- hash-based deduplication;
- detailed captions;
- tag annotations;
- sensitivity scoring;
- thumbnail sharding;
- relationship graph;
- matrix relationship view;
- local-first operation;
- source path resolution through hash lookup.

The tag graph is conceptually useful.

## Major retrieval problems

### 1. Main search is substring search

The UI filters descriptions with JavaScript `.includes()`.

It does not natively provide:

- stemming;
- fuzzy matching;
- synonyms;
- entity expansion;
- misspellings;
- ranking;
- phrase weighting;
- BM25;
- semantic similarity;
- multi-field search;
- folder/path search;
- source URL search;
- date search;
- transcript search.

For example, a user searching for “Ukraine drone” will only get exact literal substring behavior in the current client-side filtering model.

The data currently has:

| Signal | Approximate count |
|---|---:|
| Caption descriptions containing “Ukraine” | 154 |
| Current `Ukraine` tag records | 157 |

That is useful, but still far short of robust retrieval.

### 2. The vector DB is not integrated into a reliable search API

The CogVLM service has a `/search` endpoint, but:

- the main UI calls a hardcoded remote host;
- there is no local RE2 search abstraction;
- there is no local FAISS artifact in the repo;
- embeddings are not from a dedicated retrieval model;
- text and image embeddings are not necessarily well aligned.

### 3. Source links are not robust

Current source resolution is:

```text
hash -> path
```

But a hash can correspond to multiple paths, and a path can be a broken symlink.

There is no explicit state such as:

```text
ONLINE
OFFLINE_SYMLINK
DUPLICATE
RESTRICTED
FAILED
DELETED
```

The UI therefore does not clearly distinguish:

- searchable metadata;
- thumbnail-only item;
- fully accessible original;
- duplicate original;
- broken migration artifact.

### 4. File extension classification is unreliable

Some files have incorrect or misleading extensions, including odd suffixes such as:

```text
.mp4.part-Frag41.part
.mp4.QjkRCs
.mkv"
unknown_video
```

Classification should be based on magic bytes and media probing, not filename extension alone.

## Visualization problems

### Main graph

The main D3 graph loads all tags and relationships into the browser.

Current problems:

- no server-side pagination or graph slicing;
- arbitrary thresholding of semantic weights;
- dense edges;
- no cluster layout;
- no hierarchy;
- no significance testing;
- no topic summarization;
- no drilldown into representative items;
- poor scalability beyond current corpus size.

The tag-weight matrix contains 173,883 directed weight records, even though many are symmetric duplicates.

At the current `.75` display threshold, there are still approximately:

```text
31,588 directed edge records
```

That is far too dense for a simple force-directed visualization.

### 3D spicy view

The 3D view creates many individual Three.js meshes and uses raycasting over scene children on hover.

Problems:

- individual sphere mesh per item;
- individual material per item;
- random circular placement within slices;
- no spatial meaning;
- expensive hover raycasting;
- no instanced rendering;
- no thumbnail billboards;
- no efficient spatial index.

It may work for 5,000 items but will become unpleasant for tens of thousands of images plus video segments.

### Matrix view

The matrix view creates large numbers of text sprites for labels.

For 631 tags, that means roughly:

```text
1,262 label sprites
```

plus a dense matrix.

It also uses LLM-judged semantic relationships rather than observed co-occurrence. That is interesting as a concept, but it should not replace statistical co-occurrence.

### Concept map

`public/concept-map.html` is currently not functional as a serious feature.

It references undefined objects and is not integrated with the backend.

### Security/robustness concerns

Some UI code injects caption/tag/filename data into HTML without robust escaping.

That creates a stored-XSS-style risk if captions or filenames contain markup.

The custom `/images` media route should also be hardened against path traversal and confined through realpath checks.

The server currently binds port 4001 and exposes media paths without authentication. If the machine is reachable on a network, that should probably be changed to localhost or protected.

---

# 6. What RE2 could do better

## Retrieval should become hybrid

The modern baseline should be:

```text
lexical search
+ semantic search
+ image similarity
+ OCR/text search
+ structured filters
+ reranking
```

Specifically:

### Lexical

Use BM25 or SQLite FTS5 over:

- captions;
- transcripts;
- OCR;
- filenames;
- folder names;
- source URLs;
- extracted entities;
- normalized tags.

### Dense semantic

Use a dedicated embedding model for:

- captions;
- transcript chunks;
- OCR chunks;
- PDF chunks.

Do not use the last hidden layer of a generative VLM as the primary retrieval model.

### Multimodal

Use a contrastive multimodal embedding model for:

- images;
- video keyframes;
- text-to-image search;
- image-to-image search.

### OCR

This corpus is dominated by screenshots and social-media images. OCR is probably one of the highest-value retrieval signals.

OCR should cover:

- image text;
- PDF text;
- video frames;
- on-screen captions;
- usernames;
- handles;
- dates;
- quoted statements.

### Entity and topic normalization

Current tags are too raw and fragmented.

A better layer would distinguish:

- people;
- organizations;
- locations;
- geopolitical entities;
- events;
- media sources;
- platforms;
- dates;
- numeric claims;
- sensitive-content categories.

Example:

```text
Ukraine
Ukrainian
Kyiv
Russia
Russian Federation
Zelenskyy
NATO
drone
ballistic missile
Western aid
```

These should be related but not flattened into one giant tag pile.

## Visualization improvements

### Replace the raw tag matrix with statistical relationships

The current LLM-generated semantic matrix is not the most useful first visualization.

Better:

1. Build a co-occurrence matrix from tags/entities.
2. Use PMI or lift to identify statistically interesting pairs.
3. Cluster tags using embedding similarity.
4. Show top relationships above a significance threshold.
5. Let each cell drill into actual items.

The LLM-generated semantic similarity can remain a secondary layer.

### Use hierarchical topic clustering

Instead of forcing a user to explore thousands of raw tags, generate clusters:

- embed tags/entities;
- reduce dimensions;
- cluster;
- ask an LLM to name each topic;
- summarize topic contents;
- show representative items.

This would be much better than the current broken concept map.

### Timeline view

The archive has historical events.

A timeline should use:

- file modification times;
- filenames;
- OCR dates;
- archive-metadata dates;
- source URL dates;
- transcript mentions of dates;
- VLM-extracted dates.

### Map/geographic view

Useful where locations can be inferred:

- captions;
- OCR;
- transcripts;
- source URLs;
- explicit folder labels.

### Video transcript view

For each video:

- searchable transcript;
- timestamped transcript;
- keyframe strip;
- transcript-to-frame alignment;
- clip player;
- exact quoted lines with links.

A source link for video should not merely be:

```text
open the whole MP4
```

It should be:

```text
open MP4 at 00:43–00:51
```

or whatever segment supports the result.

### Search UI

The primary UI should be:

- search bar;
- faceted filters;
- results grid;
- preview pane;
- source path;
- media type;
- date;
- duplicate indicator;
- offline indicator;
- “more like this”;
- “show context”;
- transcript excerpt;
- OCR excerpt;
- caption excerpt.

The graph/matrix/3D views should be secondary discovery tools.

---

# 7. Recommended target architecture

## Core storage

Use three complementary stores.

### 1. Relational metadata store

A good simple option is SQLite, possibly with DuckDB for analytical queries.

Core tables:

```text
media_item
media_path
document_chunk
embedding
tag
entity
source_url
ingest_job
source_status
```

A media item should not be identified only by filename or path.

A stable identity model should be:

```text
media_id
  one content hash
    many filesystem paths
    many derivative assets
    many source URLs
    many chunks
    many embeddings
```

### 2. Vector store

Examples:

- LanceDB
- Qdrant
- FAISS with an external metadata service
- pgvector if you want a Postgres-centric setup

For local-first simplicity, LanceDB or FAISS+SQLite are reasonable.

### 3. Full-text index

Options:

- SQLite FTS5
- Tantivy
- Typesense
- Meilisearch

For this corpus, SQLite FTS5 may be sufficient initially.

## Ingestion pipeline

The pipeline should be incremental, resumable, and read-only with respect to RedPills.

### Step 1: scan and classify

For every non-trash file:

- determine whether it is:
  - regular file;
  - symlink;
  - broken symlink;
  - directory;
- detect MIME type using magic bytes;
- record size and modification time;
- stream SHA-256;
- store original path;
- store symlink target;
- determine accessibility state.

Do not copy original media unless you deliberately create a curated mirror.

### Step 2: image processing

For each image:

1. create thumbnail;
2. run OCR;
3. run vision captioning;
4. extract entities;
5. generate structured tags;
6. compute image embedding;
7. compute caption-text embedding;
8. optionally run face detection/clustering.

Use deduplication by content hash. Do not reprocess identical files.

### Step 3: video processing

For each video:

1. run `ffprobe`;
2. extract duration, streams, codec metadata;
3. extract audio when present;
4. transcribe audio with timestamps;
5. perform speaker diarization if needed;
6. detect scenes/keyframes;
7. extract representative frames;
8. run OCR and VLM captioning on keyframes;
9. embed transcript segments;
10. embed keyframes;
11. generate clip-level metadata.

Retrieval should happen at segment level:

```text
video_id + start_time + end_time
```

not only file level.

### Step 4: PDF processing

For each PDF:

1. extract embedded text;
2. detect scanned pages;
3. run OCR when necessary;
4. chunk by page/paragraph;
5. embed chunks;
6. preserve page numbers.

Retrieval should cite:

```text
file + page number + text span
```

### Step 5: source metadata reconciliation

Use:

- `archive_metadata.json`;
- URL lists;
- filename patterns;
- dates;
- source handles;
- folder labels.

This can turn many screenshots into source-aware items.

## Suggested media statuses

Each item should have explicit status:

```text
READY
THUMBNAIL_ONLY
SOURCE_OFFLINE
DUPLICATE
PROCESSING
FAILED
RESTRICTED
QUARANTINED
```

For the current corpus:

- many old caption records should be `THUMBNAIL_ONLY`;
- dangling symlinks should be `SOURCE_OFFLINE`;
- accessible hashed files should be `READY`.

---

# 8. Should raw images be ingested into the vector DB?

Yes — but not as raw pixels, and not as CogVLM hidden-state embeddings.

The right design is:

```text
Original files stay on disk.

Store:
- image embeddings
- text embeddings
- OCR embeddings
- transcript embeddings
- metadata
```

Use separate vector spaces:

| Vector type | Purpose |
|---|---|
| Image embedding | visually similar images, image-to-image search |
| Caption-text embedding | semantic search over generated descriptions |
| OCR-text embedding | exact/near-exact textual screenshot content |
| Transcript embedding | spoken video content |
| PDF-chunk embedding | documents |

A multimodal embedding model can provide aligned image/text vectors for cross-modal search.

This is very different from storing the generative model’s internal prompt/image hidden state and hoping it behaves like a retrieval embedding.

---

# 9. Estimated work required to ingest the rest

## Remaining accessible images

Approximately:

```text
6,989 accessible image files not currently indexed
```

or about:

```text
6,839 unique unindexed image contents
```

This does not include inaccessible broken-symlink originals.

Rough VLM captioning time:

| Hardware assumption | Approximate time |
|---|---:|
| One 24 GB GPU, 15–30 s/image | roughly 29–58 GPU-hours |
| 6 GPUs used efficiently | roughly 5–10 hours, ignoring overhead |

Actual time depends heavily on:

- VLM model size;
- batch strategy;
- image resolution;
- prompt length;
- whether OCR is included;
- whether captions are regenerated for already-captioned images;
- deduplication strategy.

## Videos

The accessible corpus contains:

```text
107.6 hours of accessible video
```

Whisper/ASR is usually the first bottleneck.

Rough estimate:

| Hardware assumption | Approximate ASR time |
|---|---:|
| Single modern GPU at 3–8× realtime | approximately 13–36 GPU-hours |
| Multiple GPUs | often several hours to roughly a day |

Then add:

- keyframe extraction;
- shot detection;
- OCR;
- VLM captioning of keyframes;
- embedding generation.

A realistic end-to-end video-ingestion effort may take:

```text
roughly one to several GPU-days
```

depending on model choices.

## PDFs

Only 28 files, but they may require OCR.

This should be comparatively small work.

## Broken symlinks

The 3,126 broken symlinks are not normal files.

To recover them, one of the following is required:

1. restore the original mount/path;
2. locate the original files elsewhere;
3. rebuild the symlinks in a separate writable mirror;
4. accept them as `SOURCE_OFFLINE` records;
5. reconstruct only from existing thumbnails and metadata.

Given the read-only requirement for the RedPills directory, the safest approach is:

> Do not modify `~/Downloads/redpills`.  
> Build a shadow manifest elsewhere that records these as offline and preserves their existing metadata/thumbnails.

If the underlying `hanoi_data` volume still exists somewhere, this could recover a large fraction of the missing originals. If not, the thumbnails are the only visual representation currently available locally.

---

# 10. Is QLoRA/finetuning feasible?

## Short answer

Yes, a QLoRA on a 27B model is technically feasible with the proposed hardware.

But it is probably not the right primary solution for the stated goal.

The goal is:

> “User asks vaguely about something; assistant retrieves, discusses, and links to original archive materials.”

That is fundamentally a **retrieval problem**, not purely a finetuning problem.

## Why finetuning is not ideal for exact archival recall

A 27B model can learn:

- how to answer in the desired style;
- how to format citations;
- how to use tools;
- domain vocabulary;
- the tag taxonomy;
- how to ask clarifying questions;
- how to avoid answering unsupported questions.

It is a poor mechanism for:

- exact file paths;
- exact hashes;
- exact source URLs;
- exact video timestamps;
- reliable recall of thousands of individual items;
- deterministic linking to sources.

Finetuning can make a model memorize fragments, but it tends to produce:

- plausible hallucinations;
- mixed-up items;
- broken citations;
- false confidence;
- drift during generation.

For archival retrieval, that is dangerous.

## Better architecture

Use:

```text
General-purpose 27B LLM
        +
Hybrid search/retrieval
        +
Optional QLoRA adapter
        +
Enforced citation format
```

The LLM should answer using retrieved evidence passed into context, not from internal parametric memory.

The retrieval system is the memory.

## What a QLoRA should be used for

A small QLoRA adapter can teach qwen38-27b to behave as a grounded archive assistant.

Train it to:

1. reformulate a vague question into retrieval queries;
2. request entity expansion;
3. ask for source metadata;
4. select relevant retrieved chunks;
5. answer only from those chunks;
6. cite media IDs;
7. cite video timestamps;
8. state when evidence is insufficient;
9. distinguish metadata-only from source-available material;
10. use the tag/topic taxonomy consistently.

That is much more valuable than trying to make it memorize the corpus.

## Dataset size for training

Current textual caption corpus:

```text
~0.8–1.2 million tokens
```

This is small for a 27B model.

After full video/PDF ingestion, the system may eventually have:

- image captions;
- OCR;
- video transcripts;
- PDF text;
- source URLs;
- tags;
- entities.

But even then, exact source recall should still be handled by search.

For LoRA behavior training, likely useful synthetic dataset sizes:

| Dataset type | Approximate useful starting size |
|---|---:|
| Tool-use conversations | 10k–50k |
| Grounded question/answer pairs | 20k–100k |
| Negative/unanswerable examples | 5k–20k |
| Citation-heavy multi-turn conversations | 5k–20k |
| Retrieval-ranking preference pairs | 10k–50k |

These should include source IDs and retrieved contexts.

## Hardware feasibility

The proposed hardware is:

```text
6× RTX 3090 = 6 × 24 GB = 144 GB VRAM
1× RTX 5090
```

A 5090 is commonly understood to have substantially more memory and bandwidth than the 3090. If it has 32 GB, total available VRAM is approximately:

```text
176 GB
```

That is more than enough for 27B QLoRA.

### Realistic expectations

| Task | Feasibility |
|---|---|
| 27B QLoRA inference adapter export | Yes |
| Single-GPU 27B QLoRA training | Likely feasible with small context and paged optimizers, but slow |
| Multi-GPU QLoRA training | Very feasible |
| 4k–8k context training | Comfortable |
| 16k context training | Feasible with good framework and packing/offload |
| 32k context training | Possible but more constrained |
| 128k+ full-context QLoRA | Not practical on consumer VRAM without aggressive chunking/offloading |
| Full 27B fine-tuning across 7 GPUs | Technically possible with ZeRO/offload, but unnecessary and much riskier |

A good setup would be:

```text
6× 3090 = training workers
1× 5090 = evaluation/inference/serving or long-context evaluation
```

If driver/CUDA/library compatibility becomes an issue between Ampere and Blackwell GPUs, keep training on the 3090s and use the 5090 for inference/evaluation.

## Important quantization caveat

The preferred model is:

```text
qwen38-27b-heretic-uncensored-q8
```

Q8 is generally an inference quantization.

For standard QLoRA:

- train against the base bf16/fp16 model, or a training framework supported by the equivalent model;
- train LoRA adapters;
- export adapters;
- optionally merge adapters and re-quantize to Q8 for inference.

Do not assume that an arbitrary Q8 GGUF file is directly trainable in all frameworks.

## Alternative LoRAs may be more useful

For this project, these may give better returns than finetuning the 27B chat model:

### 1. Retrieval embedding model LoRA

Train a small bi-encoder or cross-encoder on:

- user query;
- clicked source;
- relevant chunks;
- hard negatives.

This can directly improve search relevance.

### 2. Query-expansion adapter

Train a smaller model to:

- rewrite vague questions;
- expand entities;
- generate lexical and semantic queries;
- select filters.

### 3. Citation/instruction adapter

Train qwen38-27b to:

- answer only from citations;
- produce media IDs;
- produce timestamps;
- refuse unsupported claims.

### 4. Multimodal adapter

Only if the model should directly reason over retrieved images.

This would be more complex and is still not a substitute for retrieval.

---

# 11. Recommended chatbot design

The desired experience is feasible with a proper pipeline.

Example user request:

> “I’m trying to remember a topic concerning the actions of Ukraine, I think it was about such and such.”

Desired flow:

1. Parse vague query.
2. Expand entities:
   - Ukraine;
   - Ukrainian;
   - Russia;
   - Kyiv;
   - NATO;
   - missiles;
   - drones;
   - aid;
   - war;
   - etc.
3. Run hybrid retrieval:
   - folder search;
   - tag search;
   - caption BM25;
   - transcript BM25;
   - OCR search;
   - dense caption search;
   - image embedding similarity;
   - source URL search.
4. Rerank top results.
5. Group results into:
   - likely topics;
   - images;
   - videos;
   - PDFs;
   - duplicate copies;
   - offline metadata-only items.
6. Ask LLM to summarize using only retrieved evidence.
7. Every claim gets a citation:
   - media ID;
   - filename/folder;
   - original path if online;
   - thumbnail;
   - source URL if known;
   - transcript timestamp if video;
   - page number if PDF.
8. User clicks a citation and RE2 opens the material or explains that only a thumbnail/metadata copy is available.

That is a better “intimate familiarity” architecture than memorizing the archive inside a LLM.

---

# 12. Priority roadmap

## Phase 0 — Stabilize the audit

Priority: highest.

Actions:

1. Do not write to RedPills.
2. Build a read-only manifest database outside the archive.
3. Classify every file as:
   - regular;
   - symlink;
   - broken symlink;
   - trash;
   - duplicate;
   - unsupported.
4. Detect real MIME types.
5. Stream-hash all regular files.
6. Map hash → multiple paths.
7. Reconcile against:
   - current caption DB;
   - thumbnails;
   - old `output.json`;
   - `archive_metadata.json`.
8. Mark:
   - accessible originals;
   - thumbnail-only records;
   - offline records.
9. Rebuild thumbnails missing for currently accessible items.
10. Add basic auth/localhost-only protection.
11. Escape HTML in UI output.
12. Harden media serving against path traversal.

This alone would greatly improve reliability.

## Phase 1 — Restore/reconcile missing paths

Priority: high.

Actions:

- determine whether the old `hanoi_data` volume exists elsewhere;
- if the NAS/server has the originals, investigate whether the dangling symlinks are migration artifacts;
- build a safe dry-run repair plan outside the source directory if appropriate;
- do not repair the source archive until there is a verified backup or at least a complete manifest;
- if the data is gone, mark items as offline and preserve thumbnails.

## Phase 2 — Index remaining accessible images

Actions:

- process approximately 7k unindexed accessible images;
- generate captions, OCR, embeddings, and tags;
- regenerate the tag index;
- regenerate relationship data from observed co-occurrence;
- rebuild UI facets.

Expected result:

```text
approximately 100% accessible image coverage
```

excluding files intentionally skipped.

## Phase 3 — Ingest videos

Actions:

1. transcribe 107.6 hours of video;
2. preserve timestamps;
3. extract keyframes;
4. run OCR and VLM captioning on keyframes;
5. embed transcript segments;
6. add segment-level player links.

Expected result:

```text
video content becomes searchable and citable
```

This will likely be the biggest improvement for the chatbot use case.

## Phase 4 — Replace search backend

Replace the all-in-browser JSON model with a proper API.

Suggested endpoint concepts:

```text
/api/search
/api/media/:id
/api/documents/:id
/api/graph/tags
/api/graph/entities
/api/ask
/api/ingest/status
```

Search should return:

- documents;
- snippets;
- highlights;
- media ID;
- source status;
- thumbnail URL;
- original media URL if available;
- score;
- provenance.

## Phase 5 — Add chat with enforced citations

Use qwen38-27b as the response model, but require structured tool/retrieval interaction.

Suggested answer contract:

```text
answer:
  - claim with citation IDs

citations:
  - citation ID
  - media ID
  - title/description
  - path
  - URL if available
  - timestamp if video
  - page if PDF
```

The model should not be allowed to emit a source link that does not come from retrieved evidence.

## Phase 6 — Optional QLoRA

Train only after the RAG baseline is working.

Start with a small experiment:

- rank 8 or 16 LoRA;
- short context;
- tool/citation data;
- strict eval.

Do not attempt “memorize the entire archive” as the first training goal.

Measure:

- citation precision;
- citation recall;
- unsupported-answer rate;
- refusal accuracy;
- exact source-match rate;
- duplicate-awareness;
- timestamp accuracy;
- query-to-source success.

---

# 13. Practical modernization recommendations

## Search

Add:

- SQLite FTS5;
- synonym/entity dictionaries;
- Unicode normalization;
- case-insensitive search;
- fuzzy fallback;
- phrase search;
- quoted string search;
- negative terms;
- date filters;
- media-type filters;
- folder filters;
- online/offline filters;
- duplicate indicators.

## Embeddings

Use:

- a dedicated text embedding model;
- a dedicated multimodal embedding model;
- normalized vectors;
- HNSW or comparable ANN indexing;
- optional reranker.

## Tags

Move from raw prompt-generated tags to layered taxonomies:

1. controlled tags;
2. emergent tags;
3. entity links;
4. topic clusters;
5. content-risk categories.

Stop using a single “spiciness” number as the main sensitive-content axis.

Better dimensions include:

- violence;
- graphic injury;
- hate/harassment;
- sexual content;
- minors-related sensitivity;
- medical claims;
- election/legal claims;
- graphic death;
- privacy risk;
- confidence level.

These are for user filtering and retrieval, not necessarily censorship.

## Visualization

Replace or upgrade:

- force graph → hierarchical topic graph;
- raw matrix → clustered co-occurrence matrix;
- 3D spicy sphere view → instanced thumbnail billboards with multidimensional filters;
- broken concept map → server-generated topic summary graph.

## Chat

Make citations first-class:

- visible evidence snippets;
- confidence;
- offline indicator;
- “this item has metadata/thumbnail but no accessible original”;
- clickable original file/player page;
- source URL.

## Ingestion

Make it:

- resumable;
- idempotent;
- hash-first;
- config-driven;
- non-destructive;
- capable of processing all media types;
- separated from the web UI.

---

# 14. Direct answers to the three questions

## 1. What is RE2 doing, and what could it do better?

RE2 currently does hash-based image captioning, LLM tagging, subjective sensitivity scoring, JSON visualization, and simple client-side caption/tag filtering.

It could be substantially improved by becoming:

- a proper media manifest system;
- a hybrid lexical/semantic search engine;
- a multimodal archive with OCR, transcripts, and source provenance;
- a server-backed API rather than a giant browser-loaded JSON app;
- a citation-grounded chat interface;
- a visualization layer built from statistically meaningful and embedding-derived relationships.

The highest-value retrieval additions are:

1. full-text search;
2. proper dedicated embeddings;
3. OCR;
4. video transcript indexing;
5. source provenance from archive metadata;
6. online/offline status handling.

## 2. What percentage of the dataset is ingested?

Depending on definition:

| Coverage definition | Approximate percentage |
|---|---:|
| Caption records over historical image-like entries including broken symlinks | 42% |
| Source-resolvable caption records over currently readable images | 32% |
| Source-resolvable items over all currently accessible regular files | 23% |
| Video/audio ingestion | 0% |
| PDF ingestion | 0% |

Approximately:

```text
6,989 accessible image files remain unindexed
3,718 accessible videos remain unindexed
28 accessible PDFs remain unindexed
3,126 broken symlinks represent currently inaccessible historical entries
```

So the realistic answer is:

> **About 42% of known image entries have metadata records, but only about 32% of currently readable images are source-resolvably indexed, and roughly 23% of the entire accessible file corpus is usable as-is. Video and PDF ingestion are currently at zero.**

## 3. Is QLoRA/finetuning feasible for chat with links?

Technically:

```text
Yes — 27B QLoRA is feasible on 6×3090 + 1×5090.
```

Architecturally:

```text
Use RAG as the primary system.
Use QLoRA only to improve tool use, citation discipline, taxonomy, and answer behavior.
Do not expect finetuning alone to provide reliable exact-source recall.
```

The best target is:

> **qwen38-27b-heretic-uncensored + hybrid retrieval + structured citations + optional behavior LoRA.**

With the proposed hardware, a QLoRA can definitely improve how the model behaves as an archive assistant. But it should be the final polish layer, not the storage or retrieval mechanism.

---

# 15. Recommended final stance

The project has a useful skeleton but is currently more of a visualization demo than a durable archive assistant.

The right order of investment is:

1. **Manifest and file-status integrity.**
2. **Finish accessible image ingestion.**
3. **Add full-text search.**
4. **Transcribe and index video.**
5. **Add dedicated embeddings and reranking.**
6. **Build grounded chat with citations.**
7. **Then consider QLoRA.**

If built in that order, the “tell me what the Ukraine stuff was about and show me the sources” experience becomes realistic without needing to force a 27B model to memorize a 57 GB archive.
