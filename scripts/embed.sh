#!/usr/bin/env bash
# Add new captions, OCR, transcripts and legacy captions to the dense search index.
#
# Uses the local embedding server on port 5010 (scripts/start_embed_server.sh; started here if it is not
# running), not any LLM, so it is safe to run while extraction or ASR is running.
# Re-run after each extraction session to index what is new. Ctrl-C and re-run to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 embed "$@"
