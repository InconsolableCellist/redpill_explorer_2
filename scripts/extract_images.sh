#!/usr/bin/env bash
# Caption, OCR, tag and extract entities for every image with the vision model.
#
# Safe to stop at any time with Ctrl-C and re-run: each image is saved as soon as
# it finishes, and a re-run skips images that already have a caption from the current prompt
# (PROMPT_VERSION in re2/extract_images.py). Uncaptioned images go first, then images whose
# caption came from an older prompt are redone.
#   Ctrl-C once   finish the few requests already in flight, then exit
#   Ctrl-C twice  exit immediately (in-flight images are redone next run)
#
# On a llama.cpp router (cube, mlboy) this loads the vision model, which may unload whatever is there.
# Extra arguments pass through, e.g. --limit 50, --no-thumbnail-only, or --concurrency 7.
# By default it asks the router how many slots and how much context the model has and runs as
# many requests at once as fit; the log's first lines say what it chose. --check prints that and exits.
# A log is appended to data/logs/extract-images.log.
#
# In a terminal it first lists the presets (named servers in config.local.toml) with a live check of
# each and asks which to use: Enter keeps the configured server, n walks through adding a preset.
# --preset NAME uses one without asking; --no-pick, or no terminal, uses the configured server.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 extract-images "$@"
