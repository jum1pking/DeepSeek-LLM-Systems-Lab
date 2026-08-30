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

# This file lives at kernels/triton/, so parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "kernels"
RESULT_FILE = RESULT_DIR / "rmsnorm_triton.csv"


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One Triton program handles one token row.

    RMSNorm is a good fusion target because the reduction, normalization,
    and weight multiplication can stay inside one GPU kernel instead of
    materializing multiple intermediate tensors.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(
        x_ptr + row * n_cols + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(
        weight_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    mean_square = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)

    y = x * inv_rms * weight

    tl.store(
        output_ptr + row * n_cols + offsets,
        y,
        mask=mask,
    )


def rmsnorm_pytorch_reference(x, weight, eps):
    variance = x.float().pow(2).mean(
        dim=-1,
        keepdim=True,
    )
    normalized = x.float() * torch.rsqrt(
        variance + eps
    )
    return (normalized * weight.float()).to(x.dtype)


def rmsnorm_float32_reference(x, weight, eps):
    x_fp32 = x.float()
    weight_fp32 = weight.float()

    variance = x_fp32.pow(2).mean(
        dim=-1,
        keepdim=True,
    )

    return (
        x_fp32
        * torch.rsqrt(variance + eps)
        * weight_fp32
    )


def rmsnorm_triton(x, weight):
    output = torch.empty_like(x)

    block_size = triton.next_power_of_2(
        x.shape[-1]
    )

    grid = (x.shape[0],)

    rmsnorm_kernel[grid](
        x,
        weight,
        output,
        n_cols=x.shape[-1],
        eps=RMS_NORM_EPS,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

    return output


def build_inputs(tokens):
    torch.manual_seed(42 + tokens)

    x = torch.randn(
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

    return x, weight


def check_correctness(x, weight):
    with torch.inference_mode():
        triton_output = rmsnorm_triton(
            x=x,
            weight=weight,
        )

        pytorch_output = rmsnorm_pytorch_reference(
            x=x,
            weight=weight,
            eps=RMS_NORM_EPS,
        )

        fp32_reference = rmsnorm_float32_reference(
            x=x,
            weight=weight,
            eps=RMS_NORM_EPS,
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


def warmup(x, weight):
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            rmsnorm_triton(
                x=x,
                weight=weight,
            )

    torch.cuda.synchronize()


def benchmark_latency_us(x, weight):
    start_event = torch.cuda.Event(
        enable_timing=True
    )
    end_event = torch.cuda.Event(
        enable_timing=True
    )

    torch.cuda.synchronize()
    start_event.record()

    with torch.inference_mode():
        for _ in range(BENCHMARK_RUNS):
            rmsnorm_triton(
                x=x,
                weight=weight,
            )

    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(
        end_event
    )

    return (
        total_ms
        * 1000.0
        / BENCHMARK_RUNS
    )


def estimate_useful_bytes(tokens):
    """
    Keep exactly the same useful-bandwidth definition as the PyTorch
    baseline: read x + read weight + write y.

    This is intentionally not a claim about exact DRAM traffic. Its purpose
    is to compare implementations under one consistent accounting rule.
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

    weight_bytes = (
        HIDDEN_SIZE
        * element_size
    )

    output_bytes = (
        tokens
        * HIDDEN_SIZE
        * element_size
    )

    return (
        x_bytes
        + weight_bytes
        + output_bytes
    )


def calculate_effective_bandwidth_gbps(
    tokens,
    latency_us,
):
    useful_bytes = estimate_useful_bytes(
        tokens
    )

    latency_s = latency_us * 1e-6

    return (
        useful_bytes
        / latency_s
        / 1e9
    )


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
    x, weight = build_inputs(tokens)

    (
        max_abs_diff_vs_pytorch,
        max_abs_error_vs_fp32,
    ) = check_correctness(
        x=x,
        weight=weight,
    )

    warmup(
        x=x,
        weight=weight,
    )

    latency_us = benchmark_latency_us(
        x=x,
        weight=weight,
    )

    bandwidth_gbps = (
        calculate_effective_bandwidth_gbps(
            tokens=tokens,
            latency_us=latency_us,
        )
    )

    block_size = triton.next_power_of_2(
        HIDDEN_SIZE
    )

    result = {
        "backend": "triton",
        "implementation": "fused_rmsnorm",
        "gpu": torch.cuda.get_device_name(0),
        "dtype": DTYPE_NAME,
        "tokens": tokens,
        "hidden_size": HIDDEN_SIZE,
        "shape": f"[{tokens}, {HIDDEN_SIZE}]",
        "eps": RMS_NORM_EPS,
        "block_size": block_size,
        "num_warps": 8,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "latency_us": round(
            latency_us,
            3,
        ),
        "effective_bandwidth_gbps": round(
            bandwidth_gbps,
            3,
        ),
        "max_abs_diff_vs_pytorch": (
            max_abs_diff_vs_pytorch
        ),
        "max_abs_error_vs_fp32": (
            max_abs_error_vs_fp32
        ),
    }

    del x
    del weight

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this benchmark."
        )

    print("=== Phase 5.2 Triton RMSNorm ===")
    print(
        f"GPU:            "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(f"Dtype:          {DTYPE_NAME}")
    print(f"Hidden size:    {HIDDEN_SIZE}")
    print(f"RMSNorm eps:    {RMS_NORM_EPS}")
    print(f"Token rows:     {TOKEN_ROWS}")
    print(
        f"Triton version: {triton.__version__}"
    )
    print(
        f"Block size:     "
        f"{triton.next_power_of_2(HIDDEN_SIZE)}"
    )
    print("num_warps:      8")
    print(
        f"Warmup runs:    {WARMUP_RUNS}"
    )
    print(
        f"Benchmark runs: {BENCHMARK_RUNS}"
    )
    print()

    rows = []

    for tokens in TOKEN_ROWS:
        result = run_shape(tokens)
        rows.append(result)

        print(
            f"Shape {result['shape']:>13} | "
            f"Latency={result['latency_us']:8.3f} us | "
            f"Useful BW="
            f"{result['effective_bandwidth_gbps']:8.3f} GB/s | "
            f"Diff vs PyTorch="
            f"{result['max_abs_diff_vs_pytorch']:.6e} | "
            f"Error vs FP32="
            f"{result['max_abs_error_vs_fp32']:.6e}"
        )

    save_results(rows)

    print()
    print(
        f"Results saved to: "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()
