import csv
from pathlib import Path

import torch


HIDDEN_SIZE = 1536
RMS_NORM_EPS = 1e-6
TOKEN_ROWS = [1, 16, 128, 512, 2048]

DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"

WARMUP_RUNS = 50
BENCHMARK_RUNS = 200

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "kernels"
RESULT_FILE = RESULT_DIR / "rmsnorm_pytorch_baseline.csv"


def rmsnorm_pytorch(x, weight, eps):
    """PyTorch eager RMSNorm reference used as the Phase 5 baseline."""
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(x.dtype)


def rmsnorm_float32_reference(x, weight, eps):
    """Higher-precision correctness reference."""
    x_fp32 = x.float()
    weight_fp32 = weight.float()

    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return x_fp32 * torch.rsqrt(variance + eps) * weight_fp32


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
        output = rmsnorm_pytorch(
            x=x,
            weight=weight,
            eps=RMS_NORM_EPS,
        )

        reference = rmsnorm_float32_reference(
            x=x,
            weight=weight,
            eps=RMS_NORM_EPS,
        )

    max_abs_error = (
        output.float() - reference
    ).abs().max().item()

    return max_abs_error


def warmup(x, weight):
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            rmsnorm_pytorch(
                x=x,
                weight=weight,
                eps=RMS_NORM_EPS,
            )

    torch.cuda.synchronize()


def benchmark_latency_us(x, weight):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    with torch.inference_mode():
        for _ in range(BENCHMARK_RUNS):
            rmsnorm_pytorch(
                x=x,
                weight=weight,
                eps=RMS_NORM_EPS,
            )

    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)

    return total_ms * 1000.0 / BENCHMARK_RUNS


def estimate_useful_bytes(tokens):
    """
    Approximate useful memory traffic only:
      - read x once
      - read weight once
      - write y once

    PyTorch eager RMSNorm creates intermediate tensors and may move more bytes
    than this estimate. Therefore this is a useful-bandwidth metric for
    cross-implementation comparison, not a claim about exact DRAM traffic.
    """
    element_size = torch.tensor(
        [],
        dtype=DTYPE,
    ).element_size()

    x_bytes = tokens * HIDDEN_SIZE * element_size
    weight_bytes = HIDDEN_SIZE * element_size
    output_bytes = tokens * HIDDEN_SIZE * element_size

    return x_bytes + weight_bytes + output_bytes


def calculate_effective_bandwidth_gbps(tokens, latency_us):
    useful_bytes = estimate_useful_bytes(tokens)
    latency_s = latency_us * 1e-6

    return useful_bytes / latency_s / 1e9


def save_results(rows):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "backend",
        "implementation",
        "gpu",
        "dtype",
        "tokens",
        "hidden_size",
        "shape",
        "eps",
        "warmup_runs",
        "benchmark_runs",
        "latency_us",
        "effective_bandwidth_gbps",
        "max_abs_error",
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

    max_abs_error = check_correctness(
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

    bandwidth_gbps = calculate_effective_bandwidth_gbps(
        tokens=tokens,
        latency_us=latency_us,
    )

    result = {
        "backend": "pytorch",
        "implementation": "eager_rmsnorm",
        "gpu": torch.cuda.get_device_name(0),
        "dtype": DTYPE_NAME,
        "tokens": tokens,
        "hidden_size": HIDDEN_SIZE,
        "shape": f"[{tokens}, {HIDDEN_SIZE}]",
        "eps": RMS_NORM_EPS,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "latency_us": round(latency_us, 3),
        "effective_bandwidth_gbps": round(
            bandwidth_gbps,
            3,
        ),
        "max_abs_error": max_abs_error,
    }

    del x
    del weight

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")

    print("=== Phase 5.1 PyTorch RMSNorm Baseline ===")
    print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print(f"Dtype:         {DTYPE_NAME}")
    print(f"Hidden size:   {HIDDEN_SIZE}")
    print(f"RMSNorm eps:   {RMS_NORM_EPS}")
    print(f"Token rows:    {TOKEN_ROWS}")
    print(f"Warmup runs:   {WARMUP_RUNS}")
    print(f"Benchmark runs:{BENCHMARK_RUNS}")
    print()
    print(
        "Effective bandwidth is an approximate useful-bandwidth metric "
        "(read x + read weight + write y), not exact DRAM traffic."
    )

    rows = []

    for tokens in TOKEN_ROWS:
        result = run_shape(tokens)
        rows.append(result)

        print(
            f"Shape {result['shape']:>13} | "
            f"Latency={result['latency_us']:8.3f} us | "
            f"Useful BW={result['effective_bandwidth_gbps']:8.3f} GB/s | "
            f"Max abs error={result['max_abs_error']:.6e}"
        )

    save_results(rows)

    print()
    print(f"Results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
