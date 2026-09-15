#!/usr/bin/env bash
# Give every archive item a short title, summary and event hint, for the wiki and for result cards.
#
# Items captioned with the current image prompt are copied from their captions (no model, instant).
# The rest (images with only a 2024 caption, videos with transcripts, PDFs) go to the [wiki] text
# model in config (without a [wiki] section it is the [chat] model, e.g. qwen38-27b on mlboy via
# config.local.toml), about 10 items per request. On cube's router this loads that model and unloads
# whatever is there (e.g. the vision model a captioning run needs).
#
# Safe to stop and re-run: each batch is saved as it finishes, and a re-run only sends items with no
# digest or whose captions, transcripts or text changed since (new captions are copied for free).
#   Ctrl-C once   finish the requests in flight, then exit
#   Ctrl-C twice  exit immediately (those items are redone next run)
# Re-run after captioning or ASR sessions, then run wiki_build.sh.
# Extra arguments pass through, e.g. --limit 20 to measure the rate, or --concurrency 2 when the
# server has free slots. A log is appended to data/logs/digests.log.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 digests "$@"
