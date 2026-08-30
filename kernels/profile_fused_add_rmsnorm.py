import argparse
import sys
from pathlib import Path

import torch


HIDDEN_SIZE = 1536
RMS_NORM_EPS = 1e-6
DTYPE = torch.bfloat16

WARMUP_RUNS = 20
PROFILE_RUNS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from kernels.triton.benchmark_fused_add_rmsnorm import (
    fused_add_rmsnorm_triton,
)

from kernels.cuda.benchmark_fused_add_rmsnorm_cuda import (
    load_cuda_extension,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5.5 targeted Nsight Compute profiling "
            "for fused Add + RMSNorm."
        )
    )

    parser.add_argument(
        "--backend",
        choices=["triton", "cuda"],
        required=True,
    )

    parser.add_argument(
        "--tokens",
        type=int,
        choices=[512, 2048],
        required=True,
    )

    return parser.parse_args()


def build_inputs(tokens):
    torch.manual_seed(2026 + tokens)

    x = torch.randn(
        tokens,
        HIDDEN_SIZE,
        device="cuda",
        dtype=DTYPE,
    )

    residual = torch.randn(
        tokens,
        HIDDEN_SIZE,
        device="cuda",
        dtype=DTYPE,
    )

    weight = torch.randn(
        HIDDEN_SIZE,
        device="cuda",
        dtype=DTYPE,
    )

    return x, residual, weight


def build_runner(backend):
    if backend == "triton":
        def run(x, residual, weight):
            return fused_add_rmsnorm_triton(
                x=x,
                residual=residual,
                weight=weight,
            )

        return run

    extension, _ = load_cuda_extension()

    def run(x, residual, weight):
        return extension.fused_add_rmsnorm(
            x,
            residual,
            weight,
            RMS_NORM_EPS,
        )

    return run


def warmup(run, x, residual, weight):
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            run(x, residual, weight)

    torch.cuda.synchronize()


def profile_region(run, x, residual, weight):
    """
    Keep the Nsight target intentionally small.

    We profile only 10 launches after warmup so NCU can collect detailed
    hardware metrics without turning the full benchmark into a very long run.
    """
    torch.cuda.nvtx.range_push(
        "phase5_profile"
    )

    with torch.inference_mode():
        for _ in range(PROFILE_RUNS):
            run(x, residual, weight)

    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        "=== Phase 5.5 Targeted Kernel Profile ==="
    )
    print(
        f"Backend:       {args.backend}"
    )
    print(
        f"Shape:         "
        f"[{args.tokens}, {HIDDEN_SIZE}]"
    )
    print(
        f"Warmup runs:   {WARMUP_RUNS}"
    )
    print(
        f"Profile runs:  {PROFILE_RUNS}"
    )

    run = build_runner(
        backend=args.backend
    )

    x, residual, weight = build_inputs(
        tokens=args.tokens
    )

    warmup(
        run=run,
        x=x,
        residual=residual,
        weight=weight,
    )

    profile_region(
        run=run,
        x=x,
        residual=residual,
        weight=weight,
    )

    print("Profile region completed.")


if __name__ == "__main__":
    main()
