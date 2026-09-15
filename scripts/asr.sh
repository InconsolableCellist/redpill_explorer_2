#!/usr/bin/env bash
# Transcribe every video that has an audio track with faster-whisper large-v3 on the local GPU.
#
# Uses the local GTX 1080 Ti only, so it is safe to run while cube extracts images.
# Needs scripts/video_probe.sh to have run first (it records which videos have audio).
# Roughly 19x realtime: a few hours for the whole archive.
# Ctrl-C once finishes the current video and exits; twice exits immediately. Re-run to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 asr "$@"
