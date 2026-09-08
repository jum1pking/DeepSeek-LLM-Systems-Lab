#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# prepare_phase6_cloud.sh creates or restores the venv at this path.
VENV="${PHASE6_VENV:-/root/phase6-venv}"
PYTHON="$VENV/bin/python"

export HF_HOME="/workspace/huggingface-cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

RESULT_DIR="$ROOT_DIR/results/training"
LOG_ROOT="/workspace/phase6-logs"
LOG_DIR="$LOG_ROOT/$RUN_ID"
PROFILE_DIR="$RESULT_DIR/phase6_profiles"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$PROFILE_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Phase 6 venv Python not found: $PYTHON" >&2
    echo "Run scripts/prepare_phase6_cloud.sh first." >&2
    exit 1
fi

GPU_COUNT="$("$PYTHON" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [[ "$GPU_COUNT" -lt 2 ]]; then
    echo "ERROR: Phase 6 full run requires 2 GPUs; found $GPU_COUNT." >&2
    exit 1
fi

echo "============================================================"
echo "Phase 6 full 2×A100 cloud run"
echo "============================================================"

# Preserve previous results before starting a new full run.
RUN_ID="${PHASE6_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARCHIVE_DIR="$RESULT_DIR/archive/$RUN_ID"

PREVIOUS_ARTIFACTS=(
    "$RESULT_DIR/phase6_nccl_allreduce.csv"
    "$RESULT_DIR/phase6_ddp_scaling.csv"
    "$RESULT_DIR/phase6_strategy_comparison.csv"
    "$RESULT_DIR/phase6_final_summary.csv"
    "$RESULT_DIR/phase6_environment.txt"
    "$RESULT_DIR/phase6_topology.txt"
    "$PROFILE_DIR"
)

ARCHIVED=false

for artifact in "${PREVIOUS_ARTIFACTS[@]}"; do
    if [[ -e "$artifact" ]]; then
        mkdir -p "$ARCHIVE_DIR"
        mv "$artifact" "$ARCHIVE_DIR/"
        ARCHIVED=true
    fi
done

if [[ "$ARCHIVED" == true ]]; then
    echo "Previous Phase 6 results archived to: $ARCHIVE_DIR"
else
    echo "No previous Phase 6 results found."
fi

mkdir -p "$PROFILE_DIR"

echo "=== Environment ===" | tee "$RESULT_DIR/phase6_environment.txt"

"$PYTHON" - <<'PY' | tee -a "$RESULT_DIR/phase6_environment.txt"
import sys
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
print("NCCL available:", torch.distributed.is_nccl_available())
print("NCCL version:", torch.cuda.nccl.version())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(
        f"GPU {i}: {p.name} | "
        f"VRAM={p.total_memory / 1024**3:.2f} GB | "
        f"CC={p.major}.{p.minor}"
    )
PY

nvidia-smi topo -m | tee "$RESULT_DIR/phase6_topology.txt"

echo
echo "=== 6.1 NCCL AllReduce ==="

"$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_nccl_allreduce.py \
    2>&1 | tee "$LOG_DIR/01_nccl_allreduce.log"

echo
echo "=== 6.2a 1-GPU DDP baseline ==="

CUDA_VISIBLE_DEVICES=0 \
"$PYTHON" training/phase6_ddp_lora.py \
    2>&1 | tee "$LOG_DIR/02_ddp_1gpu.log"

echo
echo "=== 6.2b 2-GPU DDP ==="

CUDA_VISIBLE_DEVICES=0,1 \
"$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_ddp_lora.py \
    2>&1 | tee "$LOG_DIR/03_ddp_2gpu.log"

echo
echo "=== 6.3 2-GPU FSDP2 ==="

CUDA_VISIBLE_DEVICES=0,1 \
"$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_fsdp2_lora.py \
    2>&1 | tee "$LOG_DIR/04_fsdp2_2gpu.log"

echo
echo "=== 6.4 2-GPU DeepSpeed ZeRO-3 ==="

CUDA_VISIBLE_DEVICES=0,1 \
"$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_zero3_lora.py \
    2>&1 | tee "$LOG_DIR/05_zero3_2gpu.log"

echo
echo "=== 6.5a DDP communication profile ==="

CUDA_VISIBLE_DEVICES=0,1 \
"$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_profile.py \
    --strategy ddp \
    2>&1 | tee "$LOG_DIR/06_profile_ddp.log"

echo
echo "=== 6.5b FSDP2 communication profile ==="

CUDA_VISIBLE_DEVICES=0,1 \
"$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_profile.py \
    --strategy fsdp2 \
    2>&1 | tee "$LOG_DIR/07_profile_fsdp2.log"

echo
echo "Phase 6 experimental stages completed."
echo
echo "============================================================"
echo "Phase 6 full run completed."
echo "Run ID:  $RUN_ID"
echo "Results: $RESULT_DIR"
echo "Logs:    $LOG_DIR"
echo "============================================================"
