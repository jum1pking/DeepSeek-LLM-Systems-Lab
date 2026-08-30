import csv
import statistics
import sys
from pathlib import Path

import torch


HIDDEN_SIZE = 1536
RMS_NORM_EPS = 1e-6
TOKEN_ROWS = [512, 2048]

DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"

WARMUP_RUNS = 50
ROUNDS = 20
RUNS_PER_ROUND = 100

# This script is intended to live in:
#   scripts/benchmark_phase5_backend_crossover.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Import from the repository root only. The script lives outside kernels/,
# which avoids shadowing the third-party "triton" package with kernels/triton.
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.triton.benchmark_fused_add_rmsnorm import (  # noqa: E402
    fused_add_rmsnorm_triton,
)
from kernels.cuda.benchmark_fused_add_rmsnorm_cuda import (  # noqa: E402
    load_cuda_extension,
)


RESULT_DIR = PROJECT_ROOT / "results" / "kernels"
RESULT_FILE = (
    RESULT_DIR
    / "fused_add_rmsnorm_backend_crossover.csv"
)


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


def build_runners(extension):
    def run_triton(x, residual, weight):
        return fused_add_rmsnorm_triton(
            x=x,
            residual=residual,
            weight=weight,
        )

    def run_cuda(x, residual, weight):
        return extension.fused_add_rmsnorm(
            x,
            residual,
            weight,
            RMS_NORM_EPS,
        )

    return {
        "triton": run_triton,
        "cuda": run_cuda,
    }


def check_correctness(
    runners,
    x,
    residual,
    weight,
):
    with torch.inference_mode():
        triton_output = runners["triton"](
            x,
            residual,
            weight,
        )

        cuda_output = runners["cuda"](
            x,
            residual,
            weight,
        )

    torch.cuda.synchronize()

    return (
        triton_output.float()
        - cuda_output.float()
    ).abs().max().item()


def warmup(
    runners,
    x,
    residual,
    weight,
):
    # Warm both implementations before measurement so neither JIT compilation
    # nor first-use GPU state contaminates the A/B timing.
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            runners["triton"](
                x,
                residual,
                weight,
            )

            runners["cuda"](
                x,
                residual,
                weight,
            )

    torch.cuda.synchronize()


def measure_latency_us(
    run,
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
        for _ in range(RUNS_PER_ROUND):
            run(
                x,
                residual,
                weight,
            )

    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(
        end_event
    )

    return (
        total_ms
        * 1000.0
        / RUNS_PER_ROUND
    )


def benchmark_shape(
    runners,
    tokens,
):
    x, residual, weight = build_inputs(
        tokens
    )

    max_abs_diff = check_correctness(
        runners=runners,
        x=x,
        residual=residual,
        weight=weight,
    )

    warmup(
        runners=runners,
        x=x,
        residual=residual,
        weight=weight,
    )

    measurements = {
        "cuda": [],
        "triton": [],
    }

    rows = []

    for round_index in range(ROUNDS):
        # Alternate order every round to remove a systematic "first backend"
        # or boost-state advantage.
        if round_index % 2 == 0:
            order = ("cuda", "triton")
        else:
            order = ("triton", "cuda")

        round_values = {}

        for backend in order:
            latency_us = measure_latency_us(
                run=runners[backend],
                x=x,
                residual=residual,
                weight=weight,
            )

            measurements[backend].append(
                latency_us
            )

            round_values[backend] = (
                latency_us
            )

        rows.append(
            {
                "tokens": tokens,
                "hidden_size": HIDDEN_SIZE,
                "round": round_index + 1,
                "order": "->".join(order),
                "cuda_latency_us": round(
                    round_values["cuda"],
                    4,
                ),
                "triton_latency_us": round(
                    round_values["triton"],
                    4,
                ),
                "cuda_over_triton": round(
                    round_values["cuda"]
                    / round_values["triton"],
                    5,
                ),
                "max_abs_diff_cuda_vs_triton": (
                    max_abs_diff
                ),
            }
        )

    del x
    del residual
    del weight

    return measurements, rows, max_abs_diff


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def print_summary(
    tokens,
    measurements,
    max_abs_diff,
):
    cuda = summarize(
        measurements["cuda"]
    )

    triton = summarize(
        measurements["triton"]
    )

    median_ratio = (
        cuda["median"]
        / triton["median"]
    )

    if median_ratio < 1.0:
        winner = "CUDA"
        speedup = 1.0 / median_ratio
    else:
        winner = "Triton"
        speedup = median_ratio

    print()
    print(
        f"=== Shape [{tokens}, "
        f"{HIDDEN_SIZE}] ==="
    )

    print(
        "CUDA   median="
        f"{cuda['median']:.3f} us | "
        f"mean={cuda['mean']:.3f} us | "
        f"stdev={cuda['stdev']:.3f} us | "
        f"range=[{cuda['min']:.3f}, "
        f"{cuda['max']:.3f}]"
    )

    print(
        "Triton median="
        f"{triton['median']:.3f} us | "
        f"mean={triton['mean']:.3f} us | "
        f"stdev={triton['stdev']:.3f} us | "
        f"range=[{triton['min']:.3f}, "
        f"{triton['max']:.3f}]"
    )

    print(
        f"Winner: {winner} "
        f"({speedup:.3f}x by median)"
    )

    print(
        "Max abs diff CUDA vs Triton: "
        f"{max_abs_diff:.6e}"
    )


def save_results(rows):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "tokens",
        "hidden_size",
        "round",
        "order",
        "cuda_latency_us",
        "triton_latency_us",
        "cuda_over_triton",
        "max_abs_diff_cuda_vs_triton",
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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        "=== Phase 5.5 Same-Process "
        "CUDA vs Triton A/B ==="
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
        f"Shapes:          "
        f"{TOKEN_ROWS}"
    )

    print(
        f"Warmup runs:     "
        f"{WARMUP_RUNS}"
    )

    print(
        f"A/B rounds:      "
        f"{ROUNDS}"
    )

    print(
        f"Runs per round:  "
        f"{RUNS_PER_ROUND}"
    )

    print()
    print(
        "Loading native CUDA extension..."
    )

    extension, load_time_s = (
        load_cuda_extension()
    )

    print(
        f"Extension ready: "
        f"{load_time_s:.3f} s"
    )

    runners = build_runners(
        extension=extension
    )

    all_rows = []

    for tokens in TOKEN_ROWS:
        (
            measurements,
            rows,
            max_abs_diff,
        ) = benchmark_shape(
            runners=runners,
            tokens=tokens,
        )

        all_rows.extend(rows)

        print_summary(
            tokens=tokens,
            measurements=measurements,
            max_abs_diff=max_abs_diff,
        )

    save_results(all_rows)

    print()
    print(
        f"Results saved to: "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()
