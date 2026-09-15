#!/usr/bin/env bash
# One-time: build llama.cpp's server for CPU and fetch the embedding model, so search runs without cube.
# Uses the llama.cpp commit cube's embedding server was built from and cube's Q8_0 model file: vectors
# from a different build or quantization (e.g. LM Studio's Q4_K_M) drift enough to scramble rankings.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${LLAMA_SRC:-$HOME/storage/git/llama.cpp-re2}"
COMMIT=df03399b8
MODEL=data/models/nomic-embed-text-v1.5.Q8_0.gguf
SHA=3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7
if [ ! -x "$SRC/build/bin/llama-server" ]; then
  [ -d "$SRC" ] || git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$SRC"
  git -C "$SRC" checkout "$COMMIT"
  cmake -S "$SRC" -B "$SRC/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "$SRC/build" --target llama-server -j "$(nproc)"
fi
if [ ! -f "$MODEL" ]; then
  mkdir -p data/models
  scp cube:/home/offipso/ML/models/nomic-embed/nomic-embed-text-v1.5.Q8_0.gguf "$MODEL"
fi
echo "$SHA  $MODEL" | sha256sum -c -
scripts/start_embed_server.sh
