#!/usr/bin/env bash
# Transcribe scanned PDF pages (those without a text layer) with the gemma vision model on cube.
#
# Run `uv run re2 pdfs` first (local, seconds) to index text layers and queue scanned pages.
# Loads the vision model on cube, which unloads whatever model is there. About 800 pages.
# Do not run at the same time as scripts/extract_images.sh: both share the model's 8k context.
# Safe to Ctrl-C and re-run: finished pages leave the queue.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 pdf-ocr "$@"
