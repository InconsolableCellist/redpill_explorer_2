#!/usr/bin/env bash
# One-time: the Python env that runs Parakeet for speech to text (re2/asr_worker.py), apart from RE2's own
# venv because NeMo and torch are large (~7 GB). Then set [asr] python to its interpreter if it is not
# the default (~/ML/micromamba/envs/parakeet/bin/python).
#
# torch comes from the cu126 wheels: they still include sm_70 (V100) kernels and need only a CUDA 12.6
# driver (mlboy has 565.77 / 12.7). The pin stops NeMo from swapping in a build that needs a newer driver.
# The models (~2.5 GB) download on the first run, into the Hugging Face cache (~/.cache/huggingface).
set -euo pipefail
cd "$(dirname "$0")/.."
ENV="${ASR_ENV:-parakeet}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/ML/micromamba}"
PY="$MAMBA_ROOT_PREFIX/envs/$ENV/bin/python"
[ -x "$PY" ] || micromamba create -y -n "$ENV" -c conda-forge python=3.12
uv pip install --python "$PY" torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$PY" 'nemo_toolkit[asr]==3.0.0' \
  --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match \
  -c <(printf 'torch==2.14.1+cu126\ntorchaudio==2.11.0+cu126\n')
"$PY" -c "import torch, nemo; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'nemo', nemo.__version__)"
echo "test it: uv run re2 asr --limit 1, or Settings → Speech to text → Test"
