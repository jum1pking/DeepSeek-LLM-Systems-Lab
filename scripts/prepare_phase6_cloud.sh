#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="/workspace/phase6-venv"
HF_HOME="/workspace/huggingface-cache"

mkdir -p \
    "$HF_HOME" \
    /workspace/phase6-logs

echo "============================================================"
echo "Phase 6 cloud preparation"
echo "This script does NOT require a GPU."
echo "============================================================"

if [[ ! -d "$VENV" ]]; then
    echo "Creating persistent virtual environment..."
    python -m venv \
        --system-site-packages \
        "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -U pip

echo "Installing persistent Phase 6 Python dependencies..."

DS_BUILD_OPS=0 \
python -m pip install -U \
    "transformers>=4.48" \
    "peft>=0.14" \
    "accelerate>=1.2" \
    "deepspeed==0.19.6" \
    "ninja"

export HF_HOME="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

echo
echo "Caching DeepSeek 7B model to persistent storage..."

python - <<'PY'
from huggingface_hub import snapshot_download

model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

path = snapshot_download(
    repo_id=model,
)

print("Model cached at:", path)
PY

echo
echo "Checking imports..."

python - <<'PY'
import accelerate
import deepspeed
import peft
import torch
import transformers

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("Accelerate:", accelerate.__version__)
print("DeepSpeed:", deepspeed.__version__)
PY

echo
echo "============================================================"
echo "Phase 6 preparation completed."
echo "Persistent venv: $VENV"
echo "Persistent model cache: $HF_HOME"
echo "============================================================"
