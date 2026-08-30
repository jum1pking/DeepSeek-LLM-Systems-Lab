import csv
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist


TENSOR_SIZES_MB = [1, 4, 16, 64, 256]
WARMUP_STEPS = 5
BENCHMARK_STEPS = 20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "training"
    / "phase6_nccl_allreduce.csv"
)


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return local_rank


def check_correctness(local_rank):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    value = torch.tensor(
        [float(rank + 1)],
        device=local_rank,
    )

    dist.all_reduce(
        value,
        op=dist.ReduceOp.SUM,
    )

    expected = (
        world_size
        * (world_size + 1)
        / 2
    )

    if abs(value.item() - expected) > 1e-5:
        raise RuntimeError(
            f"Rank {rank}: AllReduce correctness failed: "
            f"got {value.item()}, expected {expected}"
        )


def benchmark_size(local_rank, tensor_size_mb):
    num_elements = (
        tensor_size_mb
        * 1024
        * 1024
        // torch.tensor([], dtype=torch.float32).element_size()
    )

    tensor = torch.ones(
        num_elements,
        dtype=torch.float32,
        device=local_rank,
    )

    for _ in range(WARMUP_STEPS):
        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(BENCHMARK_STEPS):
        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

    torch.cuda.synchronize()
    dist.barrier()

    average_s = (
        time.perf_counter() - start
    ) / BENCHMARK_STEPS

    tensor_bytes = tensor.numel() * tensor.element_size()

    algorithm_bw_gbps = tensor_bytes / average_s / 1e9
    world_size = dist.get_world_size()

    bus_bw_gbps = (
        algorithm_bw_gbps
        * 2
        * (world_size - 1)
        / world_size
    )

    return {
        "tensor_size_mb": tensor_size_mb,
        "average_latency_ms": average_s * 1000,
        "algorithm_bandwidth_gbps": algorithm_bw_gbps,
        "bus_bandwidth_gbps": bus_bw_gbps,
    }


def save_rows(rows):
    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "timestamp",
        "gpu",
        "world_size",
        "tensor_size_mb",
        "warmup_steps",
        "benchmark_steps",
        "average_latency_ms",
        "algorithm_bandwidth_gbps",
        "bus_bandwidth_gbps",
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


def main():
    local_rank = setup_distributed()

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size < 2:
            raise RuntimeError(
                "Phase 6 NCCL benchmark requires at least 2 GPUs."
            )

        check_correctness(local_rank)

        if rank == 0:
            print("=== Phase 6 NCCL AllReduce ===")
            print(f"GPU:        {torch.cuda.get_device_name(local_rank)}")
            print(f"World size: {world_size}")
            print("Correctness: PASS")
            print()

        rows = []

        for tensor_size_mb in TENSOR_SIZES_MB:
            result = benchmark_size(
                local_rank=local_rank,
                tensor_size_mb=tensor_size_mb,
            )

            if rank == 0:
                row = {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "gpu": torch.cuda.get_device_name(local_rank),
                    "world_size": world_size,
                    "warmup_steps": WARMUP_STEPS,
                    "benchmark_steps": BENCHMARK_STEPS,
                    **result,
                }
                rows.append(row)

                print(
                    f"{tensor_size_mb:4d} MB | "
                    f"latency={result['average_latency_ms']:8.3f} ms | "
                    f"algBW={result['algorithm_bandwidth_gbps']:8.2f} GB/s | "
                    f"busBW={result['bus_bandwidth_gbps']:8.2f} GB/s"
                )

        if rank == 0:
            save_rows(rows)
            print()
            print(f"Saved: {RESULT_FILE}")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
