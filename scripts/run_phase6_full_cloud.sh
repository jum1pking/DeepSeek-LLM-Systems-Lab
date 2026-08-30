#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCAL_VENV="/root/phase6-venv"
VENV_ARCHIVE="/workspace/phase6-venv.tar"

export HF_HOME="/workspace/huggingface-cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

RESULT_DIR="$ROOT_DIR/results/training"
LOG_DIR="/workspace/phase6-logs"
PROFILE_DIR="$RESULT_DIR/phase6_profiles"

mkdir -p \
    "$RESULT_DIR" \
    "$LOG_DIR" \
    "$PROFILE_DIR"

restore_venv() {
    if [[ ! -f "$VENV_ARCHIVE" ]]; then
        echo "ERROR: persistent Phase 6 venv archive not found:" >&2
        echo "  $VENV_ARCHIVE" >&2
        echo >&2
        echo "Run this first, preferably with CPU Only:" >&2
        echo "  bash scripts/prepare_phase6_cloud.sh" >&2
        exit 1
    fi

    echo "Restoring Phase 6 venv from Network Volume..."
    rm -rf "$LOCAL_VENV"

    tar -xf "$VENV_ARCHIVE" \
        -C /root
}

if [[ ! -x "$LOCAL_VENV/bin/python" ]]; then
    restore_venv
fi

# shellcheck disable=SC1091
source "$LOCAL_VENV/bin/activate"

echo "=== Verify persistent model cache ==="

python - <<'PY'
from huggingface_hub import snapshot_download

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

path = snapshot_download(
    repo_id=MODEL_NAME,
    local_files_only=True,
)

print("Using cached model:", path)
PY

GPU_COUNT="$(
python - <<'PY'
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

rm -f \
    "$RESULT_DIR/phase6_nccl_allreduce.csv" \
    "$RESULT_DIR/phase6_ddp_scaling.csv" \
    "$RESULT_DIR/phase6_strategy_comparison.csv" \
    "$RESULT_DIR/phase6_final_summary.csv"

rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

echo "=== Environment ===" \
    | tee "$RESULT_DIR/phase6_environment.txt"

python - <<'PY' \
    | tee -a "$RESULT_DIR/phase6_environment.txt"
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

echo \
    | tee -a "$RESULT_DIR/phase6_environment.txt"

nvidia-smi topo -m \
    | tee "$RESULT_DIR/phase6_topology.txt"

echo
echo "=== 6.1 NCCL AllReduce ==="

torchrun \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_nccl_allreduce.py \
    2>&1 | tee "$LOG_DIR/01_nccl_allreduce.log"

echo
echo "=== 6.2a 1-GPU DDP baseline ==="

CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --standalone \
    --nproc-per-node=1 \
    training/phase6_ddp_lora.py \
    2>&1 | tee "$LOG_DIR/02_ddp_1gpu.log"

echo
echo "=== 6.2b 2-GPU DDP ==="

CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_ddp_lora.py \
    2>&1 | tee "$LOG_DIR/03_ddp_2gpu.log"

echo
echo "=== 6.3 2-GPU FSDP2 ==="

CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_fsdp2_lora.py \
    2>&1 | tee "$LOG_DIR/04_fsdp2_2gpu.log"

echo
echo "=== 6.4 2-GPU DeepSpeed ZeRO-3 ==="

CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_zero3_lora.py \
    2>&1 | tee "$LOG_DIR/05_zero3_2gpu.log"

echo
echo "=== 6.5a DDP communication profile ==="

CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_profile.py \
    --strategy ddp \
    2>&1 | tee "$LOG_DIR/06_profile_ddp.log"

echo
echo "=== 6.5b FSDP2 communication profile ==="

CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --standalone \
    --nproc-per-node=2 \
    training/phase6_profile.py \
    --strategy fsdp2 \
    2>&1 | tee "$LOG_DIR/07_profile_fsdp2.log"

echo
echo "=== Phase 6 summary ==="

python - <<'PY'
import csv
from pathlib import Path

result_dir = Path("results/training")

ddp_path = result_dir / "phase6_ddp_scaling.csv"
strategy_path = result_dir / "phase6_strategy_comparison.csv"
summary_path = result_dir / "phase6_final_summary.csv"

with ddp_path.open(
    newline="",
    encoding="utf-8",
) as f:
    ddp_rows = list(csv.DictReader(f))

with strategy_path.open(
    newline="",
    encoding="utf-8",
) as f:
    strategy_rows = list(csv.DictReader(f))

ddp_by_world_size = {
    int(row["world_size"]): row
    for row in ddp_rows
}

ddp1 = ddp_by_world_size[1]
ddp2 = ddp_by_world_size[2]

t1 = float(ddp1["tokens_per_second"])
t2 = float(ddp2["tokens_per_second"])

speedup = t2 / t1
efficiency = speedup / 2 * 100

print()
print("=== DDP Scaling ===")
print(f"1 GPU throughput:   {t1:.2f} tok/s")
print(f"2 GPU throughput:   {t2:.2f} tok/s")
print(f"Speedup:            {speedup:.3f}x")
print(f"Scaling efficiency: {efficiency:.2f}%")
print()

ddp2_vram = float(ddp2["peak_vram_gb"])

rows = [
    {
        "strategy": "ddp_lora",
        "world_size": 2,
        "tokens_per_second": t2,
        "average_step_time_ms": float(
            ddp2["average_step_time_ms"]
        ),
        "peak_vram_gb": ddp2_vram,
        "throughput_vs_ddp": 1.0,
        "memory_saving_vs_ddp_pct": 0.0,
    }
]

for source in strategy_rows:
    throughput = float(
        source["tokens_per_second"]
    )
    vram = float(
        source["peak_vram_gb"]
    )

    rows.append(
        {
            "strategy": source["strategy"],
            "world_size": int(source["world_size"]),
            "tokens_per_second": throughput,
            "average_step_time_ms": float(
                source["average_step_time_ms"]
            ),
            "peak_vram_gb": vram,
            "throughput_vs_ddp": (
                throughput / t2
            ),
            "memory_saving_vs_ddp_pct": (
                (ddp2_vram - vram)
                / ddp2_vram
                * 100
            ),
        }
    )

fieldnames = list(rows[0].keys())

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(rows)

print("=== 2-GPU Strategy Comparison ===")

for row in rows:
    print(
        f"{row['strategy']:24s} | "
        f"{row['tokens_per_second']:9.2f} tok/s | "
        f"{row['average_step_time_ms']:9.2f} ms | "
        f"{row['peak_vram_gb']:7.2f} GB | "
        f"throughput/DDP={row['throughput_vs_ddp']:.3f} | "
        f"memory saved={row['memory_saving_vs_ddp_pct']:.2f}%"
    )

print()
print("Saved:", summary_path)
PY

echo
echo "=== Copy logs into repository for cloud-results branch ==="

mkdir -p "$RESULT_DIR/phase6_logs"

cp -f "$LOG_DIR"/*.log \
    "$RESULT_DIR/phase6_logs/"

echo
echo "============================================================"
echo "Phase 6 full cloud run completed."
echo "============================================================"
echo
echo "Results:"
echo "  $RESULT_DIR/phase6_environment.txt"
echo "  $RESULT_DIR/phase6_topology.txt"
echo "  $RESULT_DIR/phase6_nccl_allreduce.csv"
echo "  $RESULT_DIR/phase6_ddp_scaling.csv"
echo "  $RESULT_DIR/phase6_strategy_comparison.csv"
echo "  $RESULT_DIR/phase6_final_summary.csv"
echo "  $RESULT_DIR/phase6_profiles/"
echo "  $RESULT_DIR/phase6_logs/"
echo
echo "Next:"
echo "  git status --short"
echo "  git add results/training/"
echo "  git commit -m \"results: add Phase 6 cloud benchmark outputs\""
echo "  git push"
