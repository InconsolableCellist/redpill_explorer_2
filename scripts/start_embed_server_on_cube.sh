#!/usr/bin/env bash
# Run ON cube (or: ssh -o BatchMode=yes cube 'bash -s' < scripts/start_embed_server_on_cube.sh).
# Serves nomic-embed-text-v1.5 on port 5010 on GPU 5 (~1 GB).
# Independent of the llama.cpp router on port 5000, so model swaps do not affect it.
# Stop: pkill -f 'llama-server --embedding'
set -euo pipefail
if pgrep -f 'llama-server --embedding' >/dev/null; then
  echo "embedding server already running"
  exit 0
fi
mkdir -p ~/logs
cd ~/ML/llama.cpp
nohup ./build/bin/llama-server --embedding \
  -m /home/offipso/ML/models/nomic-embed/nomic-embed-text-v1.5.Q8_0.gguf \
  --host 0.0.0.0 --port 5010 -ngl 99 --device CUDA5 \
  -c 8192 -b 8192 -ub 8192 --parallel 4 --pooling mean \
  > ~/logs/embed-server.log 2>&1 < /dev/null &
echo "started; log: ~/logs/embed-server.log"
