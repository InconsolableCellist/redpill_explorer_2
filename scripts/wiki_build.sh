#!/usr/bin/env bash
# Compile the wiki: overview → themes → topics → events, plus entity pages, each written from ALL
# of its items (read in batches that become cited notes when there are too many for one request).
#
# Steps, each resumable:
#   1. wiki-update (no model): pages follow the latest `re2 recluster`, new items join their nearest
#      topic and event, and pages whose items changed (new captions, digests) are marked stale.
#   2. Topics not yet grouped into events are grouped (one request per topic).
#   3. Stale pages are written kind by kind: events, topics (from their event pages plus notes on
#      items no event holds), themes (from their topic pages), entities, the overview.
# Uses the [wiki] text model in config (without a [wiki] section, the [chat] model). Loading it on
# cube's router unloads whatever is there. Thousands of requests for the whole archive; run
# digests.sh first so items that only have a 2024 caption read well.
#
# Safe to stop and re-run: event groupings, notes from each batch and finished pages are saved as they
# complete, and a re-run continues with what is still stale.
#   Ctrl-C once   finish the requests in flight, then exit
#   Ctrl-C twice  exit immediately
# Extra arguments pass through: --topic 348 (one topic, its events and page), --kind event,
# --limit 2 (two topics end to end, two pages of each other kind). Then `uv run re2 wiki-export`
# writes Markdown to data/wiki. A log is appended to data/logs/wiki-build.log.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run re2 wiki-build "$@"
