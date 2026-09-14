#!/usr/bin/env bash
# Add new captions, OCR, transcripts and legacy captions to the dense search index.
#
# Uses only the embedding server on cube port 5010 (scripts/start_embed_server_on_cube.sh),
# not the model router, so it is safe to run while extraction is running.
# Re-run after each extraction session to index what is new. Ctrl-C and re-run to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 embed "$@"
