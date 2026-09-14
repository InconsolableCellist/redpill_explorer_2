#!/usr/bin/env bash
# Caption, OCR, tag and extract entities for every image with the vision model on cube.
#
# Safe to stop at any time with Ctrl-C and re-run: each image is saved as soon as
# it finishes, and a re-run skips images that already have a caption.
#   Ctrl-C once   finish the few requests already in flight, then exit
#   Ctrl-C twice  exit immediately (in-flight images are redone next run)
#
# Loads the gemma vision model on cube, which unloads whatever model is there.
# Extra arguments pass through, e.g. --limit 50 or --no-thumbnail-only.
# A log is appended to data/logs/extract-images.log.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 extract-images "$@"
