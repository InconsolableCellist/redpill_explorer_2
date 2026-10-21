#!/usr/bin/env bash
# Transcribe every video that has an audio track with Parakeet TDT 0.6B v3 on this machine's GPU.
#
# The GPUs are shared with the llama.cpp router: it starts only on a card with [asr] need_gb free, and stops
# after the current video if the card runs low or the router starts loading a model. ~2.7 GB of VRAM.
# Needs scripts/video_probe.sh to have run first (it records which videos have audio), and the NeMo env
# from scripts/setup_asr_env.sh. Roughly 100x realtime: well under an hour for the whole archive.
# Ctrl-C once finishes the current video and exits; twice exits immediately. Re-run to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 asr "$@"
