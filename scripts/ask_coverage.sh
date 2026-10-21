#!/usr/bin/env bash
# Ask a few broad questions and print, for each answer, how many items matched (candidates), how many the
# answer's model actually read, and how many it cited. Run it now and again as captioning fills in, to see
# whether Ask keeps up with what the archive holds.
#
# Uses the [chat] model (Settings, or config.local.toml); each question is one Ask turn, logged as a conversation.
#   scripts/ask_coverage.sh                       default questions, normal (sampled) reading
#   scripts/ask_coverage.sh --exhaustive          read every candidate in batches (minutes per question)
#   scripts/ask_coverage.sh "What about X?" ...   your own questions
# Ctrl-C stops after the current step.
#
# In a terminal it first lists the presets (named servers in config.local.toml) with a live check of
# each and asks which to use: Enter keeps the configured server, n walks through adding a preset.
# --preset NAME uses one without asking; --no-pick, or no terminal, uses the configured server.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 ask-coverage "$@"
