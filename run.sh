#!/usr/bin/env bash
# Start the dashboard and API server (and the local embedding server, if it is not already up).
#
# Reads config.local.toml, then config.toml: today that means http://0.0.0.0:4001, reachable from
# the LAN with no password. Ctrl-C stops it; any processing job started from scripts/ keeps running.
#   ./run.sh                  foreground, logs to the terminal
#   ./run.sh --host 127.0.0.1 this machine only
#   ./run.sh --port 4002      a second instance
#   ./run.sh --background     detached, appending to data/logs/serve.log
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = "--background" ]; then
  shift
  mkdir -p data/logs
  setsid nohup uv run re2 serve "$@" >> data/logs/serve.log 2>&1 < /dev/null &
  for _ in $(seq 1 40); do
    sleep 0.5
    if curl -s -o /dev/null --max-time 1 "http://127.0.0.1:${RE2_PORT:-4001}/api/settings"; then
      echo "serving on port ${RE2_PORT:-4001}; log: data/logs/serve.log"
      exit 0
    fi
  done
  echo "did not come up; see data/logs/serve.log" >&2
  exit 1
fi
exec uv run re2 serve "$@"
