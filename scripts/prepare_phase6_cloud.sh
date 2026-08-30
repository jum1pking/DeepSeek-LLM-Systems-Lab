#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCAL_VENV="/root/phase6-venv"
VENV_ARCHIVE="/workspace/phase6-venv.tar"

export HF_HOME="/workspace/huggingface-cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1

mkdir -p \
    "$HF_HOME" \
    /workspace/phase6-logs

echo "============================================================"
echo "Phase 6 cloud preparation"
echo "GPU is NOT required for this step."
echo
echo "Fast local venv:   $LOCAL_VENV"
echo "Persistent backup: $VENV_ARCHIVE"
echo "Model cache:       $HF_HOME"
echo "============================================================"

restore_venv() {
    echo "Restoring persistent venv archive to local container disk..."
    rm -rf "$LOCAL_VENV"
    tar -xf "$VENV_ARCHIVE" -C /root
}

create_venv() {
    echo "Creating venv on fast local container disk..."
    rm -rf "$LOCAL_VENV"

    python -m venv \
        --copies \
        --system-site-packages \
        "$LOCAL_VENV"
}

if [[ -x "$LOCAL_VENV/bin/python" ]]; then
    echo "Local venv already exists."
elif [[ -f "$VENV_ARCHIVE" ]]; then
    restore_venv
else
    create_venv
fi

# shellcheck disable=SC1091
source "$LOCAL_VENV/bin/activate"

echo
echo "Checking Phase 6 dependencies..."

if python - <<'PY'
import accelerate
import deepspeed
import peft
import transformers

print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
print("deepspeed:", deepspeed.__version__)
PY
then
    echo "Required packages are already available."
else
    echo
    echo "Installing Phase 6 dependencies into local venv..."

    python -m pip install -U pip

    DS_BUILD_OPS=0 \
    python -m pip install -U \
        "transformers>=4.48" \
        "peft>=0.14" \
        "accelerate>=1.2" \
        "deepspeed==0.19.6" \
        "ninja"
fi

echo
echo "Verifying environment..."

python - <<'PY'
import accelerate
import deepspeed
import peft
import sys
import torch
import transformers

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
print("deepspeed:", deepspeed.__version__)
PY

echo
echo "Saving venv as one sequential archive on Network Volume..."

rm -f "${VENV_ARCHIVE}.tmp"

tar -cf "${VENV_ARCHIVE}.tmp" \
    -C /root \
    "$(basename "$LOCAL_VENV")"

mv "${VENV_ARCHIVE}.tmp" \
   "$VENV_ARCHIVE"

echo
echo "Venv archive:"
ls -lh "$VENV_ARCHIVE"

echo
echo "Caching DeepSeek 7B model on persistent storage..."

python - <<'PY'
from huggingface_hub import snapshot_download

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

path = snapshot_download(
    repo_id=MODEL_NAME,
)

print("Model cached at:", path)
PY

echo
echo "============================================================"
echo "Phase 6 preparation completed."
echo
echo "Local fast venv:"
echo "  $LOCAL_VENV"
echo
echo "Persistent venv archive:"
echo "  $VENV_ARCHIVE"
echo
echo "Persistent model cache:"
echo "  $HF_HOME"
echo "============================================================"
