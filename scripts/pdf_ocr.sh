#!/usr/bin/env bash
# Transcribe scanned PDF pages (those without a text layer) with the vision model.
#
# Run `uv run re2 pdfs` first (local, seconds) to index text layers and queue scanned pages.
# Loads the vision model on its server, which may unload whatever model is there. About 800 pages.
# Do not run at the same time as scripts/extract_images.sh: both share the model's 8k context.
# Safe to Ctrl-C and re-run: finished pages leave the queue.
#
# In a terminal it first lists the presets (named servers in config.local.toml) with a live check of
# each and asks which to use: Enter keeps the configured server, n walks through adding a preset.
# --preset NAME uses one without asking; --no-pick, or no terminal, uses the configured server.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 pdf-ocr "$@"
