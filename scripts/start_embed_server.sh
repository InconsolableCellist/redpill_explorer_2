#!/usr/bin/env bash
# Serve nomic-embed-text-v1.5 on this machine (CPU), port 5010, for search, Ask and embed.sh.
# Same model file (Q8_0 GGUF, sha256 3e243421…) and llama.cpp commit as cube's server, so its vectors
# match the index (cosine 0.9999); no rebuild. `re2 serve` and `re2 embed` start it when it is not running.
# One-time setup: scripts/setup_embed_server.sh.  Stop: pkill -f 'llama-server --embedding'
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/storage/git/llama.cpp-re2/build/bin/llama-server}"
MODEL="${EMBED_MODEL_PATH:-data/models/nomic-embed-text-v1.5.Q8_0.gguf}"
if curl -s -o /dev/null --max-time 2 http://127.0.0.1:5010/health; then
  echo "embedding server already running"
  exit 0
fi
for f in "$LLAMA_SERVER" "$MODEL"; do
  [ -e "$f" ] || { echo "missing $f; run scripts/setup_embed_server.sh" >&2; exit 1; }
done
mkdir -p data/logs
setsid nohup "$LLAMA_SERVER" --embedding -m "$MODEL" --host 127.0.0.1 --port 5010 \
  -c 8192 -b 8192 -ub 8192 --parallel 4 --pooling mean --threads 8 \
  >> data/logs/embed-server.log 2>&1 < /dev/null &
for _ in $(seq 1 60); do
  curl -s -o /dev/null --max-time 1 http://127.0.0.1:5010/health && { echo "embedding server started on :5010"; exit 0; }
  sleep 0.5
done
echo "embedding server did not come up; see data/logs/embed-server.log" >&2
exit 1
