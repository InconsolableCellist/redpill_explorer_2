#!/usr/bin/env bash
# Score how related the ~400 most used tags are, with the Ask/chat model ([chat] in config,
# overridden in config.local.toml). Also recomputes which tags co-occur in the archive (no model).
#
# One request per tag, each naming up to 25 related tags: about 400 requests. Slow servers are fine;
# Ctrl-C and re-run to resume. Re-run after large extraction runs to score new tags.
# Extra arguments pass through, e.g. --concurrency 4, --limit 10, or --rescore after a full
# re-extraction has changed which tags are most used.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 tag-relations "$@"
