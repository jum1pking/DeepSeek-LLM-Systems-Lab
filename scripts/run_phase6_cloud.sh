#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-/workspace/huggingface-cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

mkdir -p     "$HF_HOME"     results/training     /workspace/phase6-logs

echo "============================================================"
echo "Phase 6.1 / 6.2 cloud benchmark"
echo "============================================================"
echo "Repository: $ROOT_DIR"
echo "HF_HOME:    $HF_HOME"
echo

echo "=== Environment ==="
python - <<'PY'
import sys
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(
        f"GPU {i}: {p.name} | "
        f"VRAM={p.total_memory / 1024**3:.2f} GB | "
        f"CC={p.major}.{p.minor}"
    )

print("NCCL available:", torch.distributed.is_nccl_available())
print("NCCL version:", torch.cuda.nccl.version())
PY

nvidia-smi topo -m     | tee results/training/phase6_topology.txt

echo
echo "=== Check Python dependencies ==="

if ! python - <<'PY'
import transformers
import peft
import accelerate

print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
PY
then
    echo "Installing minimal Phase 6 dependencies..."
    python -m pip install -U         "transformers>=4.48"         "peft>=0.14"         "accelerate>=1.2"
fi

echo
echo "=== Cache 7B model once before launching multiple ranks ==="

python - <<'PY'
from huggingface_hub import snapshot_download

model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

path = snapshot_download(
    repo_id=model,
)

print("Model cached at:", path)
PY

echo
echo "=== Phase 6.1: 2-GPU NCCL AllReduce ==="

torchrun     --standalone     --nproc-per-node=2     training/phase6_nccl_allreduce.py     2>&1 | tee /workspace/phase6-logs/nccl_allreduce.log

echo
echo "=== Phase 6.2a: 1-GPU 7B LoRA baseline ==="

rm -f results/training/phase6_ddp_scaling.csv

CUDA_VISIBLE_DEVICES=0 torchrun     --standalone     --nproc-per-node=1     training/phase6_ddp_lora.py     2>&1 | tee /workspace/phase6-logs/ddp_1gpu.log

echo
echo "=== Phase 6.2b: 2-GPU 7B LoRA DDP ==="

CUDA_VISIBLE_DEVICES=0,1 torchrun     --standalone     --nproc-per-node=2     training/phase6_ddp_lora.py     2>&1 | tee /workspace/phase6-logs/ddp_2gpu.log

echo
echo "=== Scaling summary ==="

python - <<'PY'
import csv
from pathlib import Path

path = Path("results/training/phase6_ddp_scaling.csv")

with path.open(
    newline="",
    encoding="utf-8",
) as f:
    rows = list(csv.DictReader(f))

by_world_size = {
    int(row["world_size"]): row
    for row in rows
}

one = by_world_size[1]
two = by_world_size[2]

t1 = float(one["tokens_per_second"])
t2 = float(two["tokens_per_second"])

speedup = t2 / t1
efficiency = speedup / 2 * 100

print(f"1 GPU throughput:   {t1:.2f} tok/s")
print(f"2 GPU throughput:   {t2:.2f} tok/s")
print(f"Speedup:            {speedup:.3f}x")
print(f"Scaling efficiency: {efficiency:.2f}%")
PY

echo
echo "============================================================"
echo "Phase 6.1 / 6.2 completed."
echo "============================================================"
echo
echo "Important result files:"
echo "  results/training/phase6_topology.txt"
echo "  results/training/phase6_nccl_allreduce.csv"
echo "  results/training/phase6_ddp_scaling.csv"
echo
echo "Logs:"
echo "  /workspace/phase6-logs/"
