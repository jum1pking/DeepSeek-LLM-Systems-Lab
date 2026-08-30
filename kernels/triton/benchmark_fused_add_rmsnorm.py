import csv
from pathlib import Path

import torch
import triton
import triton.language as tl


HIDDEN_SIZE = 1536
RMS_NORM_EPS = 1e-6
TOKEN_ROWS = [1, 16, 128, 512, 2048]

DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"

WARMUP_RUNS = 50
BENCHMARK_RUNS = 200

# File location:
#   kernels/triton/benchmark_fused_add_rmsnorm.py
# parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "kernels"
RESULT_FILE = RESULT_DIR / "fused_add_rmsnorm.csv"


@triton.jit
def fused_add_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One Triton program handles one token row.

    The important optimization is fusion:
      residual add
          +
      RMSNorm reduction / scaling
          ↓
      one GPU kernel

    No intermediate (x + residual) tensor is written to global memory.
    """
    row = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(
        x_ptr + row * n_cols + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    residual = tl.load(
        residual_ptr + row * n_cols + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(
        weight_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    summed = x + residual

    mean_square = (
        tl.sum(summed * summed, axis=0)
        / n_cols
    )

    inv_rms = tl.rsqrt(
        mean_square + eps
    )

    output = (
        summed
        * inv_rms
        * weight
    )

    tl.store(
        output_ptr + row * n_cols + offsets,
        output,
        mask=mask,
    )


def fused_add_rmsnorm_pytorch(
    x,
    residual,
    weight,
    eps,
):
    """
    Unfused PyTorch eager baseline.

    We intentionally use FP32 arithmetic after loading BF16 inputs so the
    mathematical semantics match the Triton kernel closely. PyTorch still
    materializes the residual-add result as an intermediate tensor.
    """
    summed = (
        x.float()
        + residual.float()
    )

    variance = (
        summed.pow(2)
        .mean(
            dim=-1,
            keepdim=True,
        )
    )

    normalized = (
        summed
        * torch.rsqrt(
            variance + eps
        )
    )

    return (
        normalized
        * weight.float()
    ).to(x.dtype)


def fused_add_rmsnorm_fp32_reference(
    x,
    residual,
    weight,
    eps,
):
    summed = (
        x.float()
        + residual.float()
    )

    variance = (
        summed.pow(2)
        .mean(
            dim=-1,
            keepdim=True,
        )
    )

    return (
        summed
        * torch.rsqrt(
            variance + eps
        )
        * weight.float()
    )


def fused_add_rmsnorm_triton(
    x,
    residual,
    weight,
):
    output = torch.empty_like(x)

    block_size = (
        triton.next_power_of_2(
            x.shape[-1]
        )
    )

    grid = (x.shape[0],)

    fused_add_rmsnorm_kernel[grid](
        x,
        residual,
        weight,
        output,
        n_cols=x.shape[-1],
        eps=RMS_NORM_EPS,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

    return output


def build_inputs(tokens):
    torch.manual_seed(
        2026 + tokens
    )

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

    return (
        x,
        residual,
        weight,
    )


def check_correctness(
    x,
    residual,
    weight,
):
    with torch.inference_mode():
        pytorch_output = (
            fused_add_rmsnorm_pytorch(
                x=x,
                residual=residual,
                weight=weight,
                eps=RMS_NORM_EPS,
            )
        )

        triton_output = (
            fused_add_rmsnorm_triton(
                x=x,
                residual=residual,
                weight=weight,
            )
        )

        fp32_reference = (
            fused_add_rmsnorm_fp32_reference(
                x=x,
                residual=residual,
                weight=weight,
                eps=RMS_NORM_EPS,
            )
        )

    torch.cuda.synchronize()

    max_abs_diff_vs_pytorch = (
        triton_output.float()
        - pytorch_output.float()
    ).abs().max().item()

    max_abs_error_vs_fp32 = (
        triton_output.float()
        - fp32_reference
    ).abs().max().item()

    return (
        max_abs_diff_vs_pytorch,
        max_abs_error_vs_fp32,
    )


def warmup_pytorch(
    x,
    residual,
    weight,
):
    with torch.inference_mode():
        for _ in range(
            WARMUP_RUNS
        ):
            fused_add_rmsnorm_pytorch(
                x=x,
                residual=residual,
                weight=weight,
                eps=RMS_NORM_EPS,
            )

    torch.cuda.synchronize()


def warmup_triton(
    x,
    residual,
    weight,
):
    with torch.inference_mode():
        for _ in range(
            WARMUP_RUNS
        ):
            fused_add_rmsnorm_triton(
                x=x,
                residual=residual,
                weight=weight,
            )

    torch.cuda.synchronize()


def benchmark_pytorch_latency_us(
    x,
    residual,
    weight,
):
    start_event = torch.cuda.Event(
        enable_timing=True
    )
    end_event = torch.cuda.Event(
        enable_timing=True
    )

    torch.cuda.synchronize()
    start_event.record()

    with torch.inference_mode():
        for _ in range(
            BENCHMARK_RUNS
        ):
            fused_add_rmsnorm_pytorch(
                x=x,
                residual=residual,
                weight=weight,
                eps=RMS_NORM_EPS,
            )

    end_event.record()
    torch.cuda.synchronize()

    total_ms = (
        start_event.elapsed_time(
            end_event
        )
    )

    return (
        total_ms
        * 1000.0
        / BENCHMARK_RUNS
    )


def benchmark_triton_latency_us(
    x,
    residual,
    weight,
):
    start_event = torch.cuda.Event(
        enable_timing=True
    )
    end_event = torch.cuda.Event(
        enable_timing=True
    )

    torch.cuda.synchronize()
    start_event.record()

    with torch.inference_mode():
        for _ in range(
            BENCHMARK_RUNS
        ):
            fused_add_rmsnorm_triton(
                x=x,
                residual=residual,
                weight=weight,
            )

    end_event.record()
    torch.cuda.synchronize()

    total_ms = (
        start_event.elapsed_time(
            end_event
        )
    )

    return (
        total_ms
        * 1000.0
        / BENCHMARK_RUNS
    )


def estimate_useful_bytes(tokens):
    """
    Common useful-traffic accounting for both implementations:

      read x
      read residual
      read weight
      write output

    This deliberately ignores PyTorch's intermediate tensor traffic, so the
    metric is useful for comparing implementations under one definition,
    not for claiming exact DRAM traffic.
    """
    element_size = torch.tensor(
        [],
        dtype=DTYPE,
    ).element_size()

    x_bytes = (
        tokens
        * HIDDEN_SIZE
        * element_size
    )

    residual_bytes = x_bytes

    weight_bytes = (
        HIDDEN_SIZE
        * element_size
    )

    output_bytes = x_bytes

    return (
        x_bytes
        + residual_bytes
        + weight_bytes
        + output_bytes
    )


def calculate_effective_bandwidth_gbps(
    tokens,
    latency_us,
):
    useful_bytes = (
        estimate_useful_bytes(
            tokens
        )
    )

    latency_s = (
        latency_us
        * 1e-6
    )

    return (
        useful_bytes
        / latency_s
        / 1e9
    )


def build_result_row(
    backend,
    implementation,
    tokens,
    latency_us,
    max_abs_diff_vs_pytorch,
    max_abs_error_vs_fp32,
):
    return {
        "backend": backend,
        "implementation": implementation,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": DTYPE_NAME,
        "tokens": tokens,
        "hidden_size": HIDDEN_SIZE,
        "shape": (
            f"[{tokens}, "
            f"{HIDDEN_SIZE}]"
        ),
        "eps": RMS_NORM_EPS,
        "block_size": (
            triton.next_power_of_2(
                HIDDEN_SIZE
            )
            if backend == "triton"
            else ""
        ),
        "num_warps": (
            8
            if backend == "triton"
            else ""
        ),
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": (
            BENCHMARK_RUNS
        ),
        "latency_us": round(
            latency_us,
            3,
        ),
        "effective_bandwidth_gbps": round(
            calculate_effective_bandwidth_gbps(
                tokens=tokens,
                latency_us=latency_us,
            ),
            3,
        ),
        "max_abs_diff_vs_pytorch": (
            max_abs_diff_vs_pytorch
        ),
        "max_abs_error_vs_fp32": (
            max_abs_error_vs_fp32
        ),
    }


def save_results(rows):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "backend",
        "implementation",
        "gpu",
        "dtype",
        "tokens",
        "hidden_size",
        "shape",
        "eps",
        "block_size",
        "num_warps",
        "warmup_runs",
        "benchmark_runs",
        "latency_us",
        "effective_bandwidth_gbps",
        "max_abs_diff_vs_pytorch",
        "max_abs_error_vs_fp32",
    ]

    with RESULT_FILE.open(
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


def run_shape(tokens):
    (
        x,
        residual,
        weight,
    ) = build_inputs(tokens)

    (
        max_abs_diff_vs_pytorch,
        max_abs_error_vs_fp32,
    ) = check_correctness(
        x=x,
        residual=residual,
        weight=weight,
    )

    warmup_pytorch(
        x=x,
        residual=residual,
        weight=weight,
    )

    pytorch_latency_us = (
        benchmark_pytorch_latency_us(
            x=x,
            residual=residual,
            weight=weight,
        )
    )

    warmup_triton(
        x=x,
        residual=residual,
        weight=weight,
    )

    triton_latency_us = (
        benchmark_triton_latency_us(
            x=x,
            residual=residual,
            weight=weight,
        )
    )

    speedup = (
        pytorch_latency_us
        / triton_latency_us
    )

    pytorch_row = build_result_row(
        backend="pytorch",
        implementation=(
            "unfused_add_rmsnorm"
        ),
        tokens=tokens,
        latency_us=(
            pytorch_latency_us
        ),
        max_abs_diff_vs_pytorch=0.0,
        max_abs_error_vs_fp32=(
            max_abs_error_vs_fp32
        ),
    )

    triton_row = build_result_row(
        backend="triton",
        implementation=(
            "fused_add_rmsnorm"
        ),
        tokens=tokens,
        latency_us=(
            triton_latency_us
        ),
        max_abs_diff_vs_pytorch=(
            max_abs_diff_vs_pytorch
        ),
        max_abs_error_vs_fp32=(
            max_abs_error_vs_fp32
        ),
    )

    del x
    del residual
    del weight

    return (
        pytorch_row,
        triton_row,
        speedup,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        "=== Phase 5.3 "
        "Fused Add + RMSNorm ==="
    )
    print(
        f"GPU:            "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"Dtype:          "
        f"{DTYPE_NAME}"
    )
    print(
        f"Hidden size:    "
        f"{HIDDEN_SIZE}"
    )
    print(
        f"RMSNorm eps:    "
        f"{RMS_NORM_EPS}"
    )
    print(
        f"Token rows:     "
        f"{TOKEN_ROWS}"
    )
    print(
        f"Triton version: "
        f"{triton.__version__}"
    )
    print(
        f"Warmup runs:    "
        f"{WARMUP_RUNS}"
    )
    print(
        f"Benchmark runs: "
        f"{BENCHMARK_RUNS}"
    )
    print()

    rows = []

    for tokens in TOKEN_ROWS:
        (
            pytorch_row,
            triton_row,
            speedup,
        ) = run_shape(tokens)

        rows.extend(
            [
                pytorch_row,
                triton_row,
            ]
        )

        print(
            f"Shape "
            f"{pytorch_row['shape']:>13} | "
            f"PyTorch="
            f"{pytorch_row['latency_us']:8.3f} us | "
            f"Triton="
            f"{triton_row['latency_us']:8.3f} us | "
            f"Speedup="
            f"{speedup:6.2f}x | "
            f"Diff="
            f"{triton_row['max_abs_diff_vs_pytorch']:.6e}"
        )

    save_results(rows)

    print()
    print(
        f"Results saved to: "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()
