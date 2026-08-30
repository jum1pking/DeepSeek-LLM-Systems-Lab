#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-LLM-Systems-Lab
# Phase 5 GPU-kernel clean reproduction.
#
# Runs:
#   5.1 PyTorch RMSNorm baseline
#   5.2 Triton RMSNorm
#   5.3 Triton fused Add + RMSNorm
#   5.4 Native CUDA fused Add + RMSNorm
#   5.5 Same-process CUDA vs Triton A/B
#   5.5 Nsight Compute profiles for 512 / 2048
#
# Existing result files are archived rather than deleted.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="$ROOT_DIR/.venv"
RESULT_DIR="$ROOT_DIR/results/kernels"
PROFILER_DIR="$RESULT_DIR/profiler"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE_DIR="$RESULT_DIR/archive/$TIMESTAMP"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: .venv Python not found: $VENV/bin/python" >&2
    exit 1
fi

for required in \
    /usr/bin/gcc-14 \
    /usr/bin/g++-14 \
    /usr/local/cuda/bin/nvcc
do
    if [[ ! -x "$required" ]]; then
        echo "ERROR: required tool not found: $required" >&2
        exit 1
    fi
done

if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: Nsight Compute CLI (ncu) not found." >&2
    exit 1
fi

mkdir -p "$RESULT_DIR" "$PROFILER_DIR" "$ARCHIVE_DIR/profiler"

# Archive prior generated evidence.
for file in \
    rmsnorm_pytorch_baseline.csv \
    rmsnorm_triton.csv \
    fused_add_rmsnorm.csv \
    fused_add_rmsnorm_cuda.csv \
    fused_add_rmsnorm_backend_crossover.csv
do
    if [[ -f "$RESULT_DIR/$file" ]]; then
        mv "$RESULT_DIR/$file" "$ARCHIVE_DIR/"
    fi
done

for report in \
    triton_512.ncu-rep \
    triton_2048.ncu-rep \
    cuda_512.ncu-rep \
    cuda_2048.ncu-rep
do
    if [[ -f "$PROFILER_DIR/$report" ]]; then
        mv "$PROFILER_DIR/$report" "$ARCHIVE_DIR/profiler/"
    fi
done

# shellcheck disable=SC1091
source "$VENV/bin/activate"

export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export TORCH_CUDA_ARCH_LIST=12.0

echo "=== Phase 5 GPU Kernel Reproduction ==="
echo "Repository: $ROOT_DIR"
echo "Archive:    $ARCHIVE_DIR"
echo

echo "=== 5.1 PyTorch RMSNorm ==="
python -m kernels.pytorch.benchmark_rmsnorm

echo
echo "=== 5.2 Triton RMSNorm ==="
python -m kernels.triton.benchmark_rmsnorm_triton

echo
echo "=== 5.3 Triton Fused Add + RMSNorm ==="
python -m kernels.triton.benchmark_fused_add_rmsnorm

echo
echo "=== 5.4 Native CUDA Fused Add + RMSNorm ==="
python -m kernels.cuda.benchmark_fused_add_rmsnorm_cuda

echo
echo "=== 5.5 Same-process CUDA vs Triton A/B ==="
python scripts/benchmark_phase5_backend_crossover.py

echo
echo "=== 5.5 Nsight Compute: Triton 512 ==="
ncu \
    --set detailed \
    --nvtx \
    --nvtx-include "phase5_profile/" \
    --force-overwrite \
    -o "$PROFILER_DIR/triton_512" \
    python -m kernels.profile_fused_add_rmsnorm \
    --backend triton \
    --tokens 512

echo
echo "=== 5.5 Nsight Compute: Triton 2048 ==="
ncu \
    --set detailed \
    --nvtx \
    --nvtx-include "phase5_profile/" \
    --force-overwrite \
    -o "$PROFILER_DIR/triton_2048" \
    python -m kernels.profile_fused_add_rmsnorm \
    --backend triton \
    --tokens 2048

echo
echo "=== 5.5 Nsight Compute: CUDA 512 ==="
ncu \
    --set detailed \
    --nvtx \
    --nvtx-include "phase5_profile/" \
    --force-overwrite \
    -o "$PROFILER_DIR/cuda_512" \
    python -m kernels.profile_fused_add_rmsnorm \
    --backend cuda \
    --tokens 512

echo
echo "=== 5.5 Nsight Compute: CUDA 2048 ==="
ncu \
    --set detailed \
    --nvtx \
    --nvtx-include "phase5_profile/" \
    --force-overwrite \
    -o "$PROFILER_DIR/cuda_2048" \
    python -m kernels.profile_fused_add_rmsnorm \
    --backend cuda \
    --tokens 2048

deactivate

echo
echo "=== Phase 5 reproduction completed ==="
echo "Results: $RESULT_DIR"
echo "Previous run archived at: $ARCHIVE_DIR"
