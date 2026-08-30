import csv
import os
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


HIDDEN_SIZE = 1536
RMS_NORM_EPS = 1e-6
TOKEN_ROWS = [1, 16, 128, 512, 2048]

DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"

WARMUP_RUNS = 50
BENCHMARK_RUNS = 200

# File location:
#   kernels/cuda/benchmark_fused_add_rmsnorm_cuda.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = Path(__file__).with_name(
    "fused_add_rmsnorm_cuda.cu"
)

RESULT_DIR = PROJECT_ROOT / "results" / "kernels"
RESULT_FILE = (
    RESULT_DIR
    / "fused_add_rmsnorm_cuda.csv"
)

EXTENSION_NAME = (
    "deepseek_lab_fused_add_rmsnorm_cuda"
)


def load_cuda_extension():
    """
    JIT-compile the native CUDA source once.

    Compilation happens before any benchmark timing. PyTorch caches the built
    extension, so later runs normally reuse the compiled binary unless the
    source or build configuration changes.
    """
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST",
        "12.0",
    )

    gcc14 = Path("/usr/bin/gcc-14")
    gxx14 = Path("/usr/bin/g++-14")

    if not gcc14.exists() or not gxx14.exists():
        raise RuntimeError(
            "CUDA 12.8 requires GCC/G++ <= 14, but GCC 14 was not found. "
            "Expected /usr/bin/gcc-14 and /usr/bin/g++-14."
        )

    os.environ["CC"] = str(gcc14)
    os.environ["CXX"] = str(gxx14)

    start = time.perf_counter()

    extension = load(
        name=EXTENSION_NAME,
        sources=[str(CUDA_SOURCE)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
        ],
        verbose=True,
    )

    torch.cuda.synchronize()

    load_time_s = (
        time.perf_counter()
        - start
    )

    return extension, load_time_s


def fused_add_rmsnorm_pytorch(
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
    extension,
    x,
    residual,
    weight,
):
    with torch.inference_mode():
        cuda_output = (
            extension.fused_add_rmsnorm(
                x,
                residual,
                weight,
                RMS_NORM_EPS,
            )
        )

        pytorch_output = (
            fused_add_rmsnorm_pytorch(
                x=x,
                residual=residual,
                weight=weight,
                eps=RMS_NORM_EPS,
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
        cuda_output.float()
        - pytorch_output.float()
    ).abs().max().item()

    max_abs_error_vs_fp32 = (
        cuda_output.float()
        - fp32_reference
    ).abs().max().item()

    return (
        max_abs_diff_vs_pytorch,
        max_abs_error_vs_fp32,
    )


def warmup_cuda(
    extension,
    x,
    residual,
    weight,
):
    with torch.inference_mode():
        for _ in range(
            WARMUP_RUNS
        ):
            extension.fused_add_rmsnorm(
                x,
                residual,
                weight,
                RMS_NORM_EPS,
            )

    torch.cuda.synchronize()


def benchmark_cuda_latency_us(
    extension,
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
            extension.fused_add_rmsnorm(
                x,
                residual,
                weight,
                RMS_NORM_EPS,
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
    Same useful-bandwidth definition as Phase 5.3:

      read x
      read residual
      read weight
      write output

    It is intentionally an accounting metric rather than exact DRAM traffic.
    """
    element_size = torch.tensor(
        [],
        dtype=DTYPE,
    ).element_size()

    tensor_bytes = (
        tokens
        * HIDDEN_SIZE
        * element_size
    )

    weight_bytes = (
        HIDDEN_SIZE
        * element_size
    )

    return (
        tensor_bytes
        + tensor_bytes
        + weight_bytes
        + tensor_bytes
    )


def calculate_effective_bandwidth_gbps(
    tokens,
    latency_us,
):
    latency_s = (
        latency_us
        * 1e-6
    )

    return (
        estimate_useful_bytes(tokens)
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
        "threads_per_block",
        "items_per_thread",
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
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def run_shape(
    extension,
    tokens,
):
    (
        x,
        residual,
        weight,
    ) = build_inputs(tokens)

    (
        max_abs_diff_vs_pytorch,
        max_abs_error_vs_fp32,
    ) = check_correctness(
        extension=extension,
        x=x,
        residual=residual,
        weight=weight,
    )

    warmup_cuda(
        extension=extension,
        x=x,
        residual=residual,
        weight=weight,
    )

    latency_us = (
        benchmark_cuda_latency_us(
            extension=extension,
            x=x,
            residual=residual,
            weight=weight,
        )
    )

    row = {
        "backend": "cuda",
        "implementation": (
            "fused_add_rmsnorm"
        ),
        "gpu": torch.cuda.get_device_name(0),
        "dtype": DTYPE_NAME,
        "tokens": tokens,
        "hidden_size": HIDDEN_SIZE,
        "shape": (
            f"[{tokens}, "
            f"{HIDDEN_SIZE}]"
        ),
        "eps": RMS_NORM_EPS,
        "threads_per_block": 256,
        "items_per_thread": 6,
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

    del x
    del residual
    del weight

    return row


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        "=== Phase 5.4 Native CUDA "
        "Fused Add + RMSNorm ==="
    )
    print(
        f"GPU:             "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"PyTorch:         "
        f"{torch.__version__}"
    )
    print(
        f"PyTorch CUDA:    "
        f"{torch.version.cuda}"
    )
    print(
        f"Dtype:           "
        f"{DTYPE_NAME}"
    )
    print(
        f"Hidden size:     "
        f"{HIDDEN_SIZE}"
    )
    print(
        "CUDA layout:     "
        "1 block/token, "
        "256 threads/block, "
        "6 values/thread"
    )
    print(
        f"Token rows:      "
        f"{TOKEN_ROWS}"
    )
    print()

    print(
        "Loading / JIT-compiling CUDA extension..."
    )

    extension, load_time_s = (
        load_cuda_extension()
    )

    print(
        f"Extension ready: "
        f"{load_time_s:.3f} s"
    )
    print(
        "(Compilation time is not included "
        "in kernel benchmark latency.)"
    )
    print()

    rows = []

    for tokens in TOKEN_ROWS:
        row = run_shape(
            extension=extension,
            tokens=tokens,
        )

        rows.append(row)

        print(
            f"Shape {row['shape']:>13} | "
            f"CUDA="
            f"{row['latency_us']:8.3f} us | "
            f"Useful BW="
            f"{row['effective_bandwidth_gbps']:8.3f} GB/s | "
            f"Diff vs PyTorch="
            f"{row['max_abs_diff_vs_pytorch']:.6e} | "
            f"Error vs FP32="
            f"{row['max_abs_error_vs_fp32']:.6e}"
        )

    save_results(rows)

    print()
    print(
        f"Results saved to: "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()
