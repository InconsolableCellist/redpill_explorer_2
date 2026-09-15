#!/usr/bin/env bash
# Probe every video with ffprobe and pull up to 8 keyframes (every 15 s) plus a thumbnail.
#
# Local CPU and NFS reads only; nothing runs on cube. Writes JPEGs under data/keyframes/.
# Safe to Ctrl-C and re-run: probed videos are skipped. Run this before scripts/asr.sh,
# which needs to know which videos have an audio track.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 video-probe "$@"
