import  os
import torch
import torch.distributed as dist
import common
import time
from datetime import datetime

TENSOR_SIZE_MB = 256
WARMUP_STEPS = 5
BENCHMARK_STEPS = 20

def setup_distributed():


    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return local_rank

def main():
    local_rank = setup_distributed()

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(
            f"Rank {rank} | "
            f"Local rank {local_rank} | "
            f"World size {world_size} | "
            f"GPU {torch.cuda.get_device_name(local_rank)}"
        )

        num_elements = (
            TENSOR_SIZE_MB *1024 *1024
            // torch.tensor([], dtype=torch.float32).element_size()
        )

        tensor = torch.ones(
            num_elements,
            dtype=torch.float32,
            device=local_rank,
        )

        for _ in range(WARMUP_STEPS):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()

        # Synchronize all ranks before and after timing so communication latency
        # is measured over the same interval on every GPU.
        dist.barrier()
        torch.cuda.synchronize()

        start_time = time.perf_counter()

        for _ in range(BENCHMARK_STEPS):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()
        dist.barrier()

        total_time = time.perf_counter() - start_time
        average_latency = total_time / BENCHMARK_STEPS

        tensor_bytes = tensor.numel() * tensor.element_size()

        algorithm_bandwidth_gbps = (
            tensor_bytes / average_latency / 1e9
        )

        # Convert algorithm bandwidth to effective bus bandwidth using the
        # standard Ring AllReduce communication-volume normalization.
        if world_size > 1:
            bus_bandwidth_gbps = (
                algorithm_bandwidth_gbps
                *2
                * (world_size -1)
                / world_size
            )
        else:
            bus_bandwidth_gbps = 0.0

        if rank == 0:
            print("\n=== NCCL AllReduce Benchmark ===")
            print(f"World size:          {world_size}")
            print(f"Tensor size:         {TENSOR_SIZE_MB} MB")
            print(f"Warmup steps:        {WARMUP_STEPS}")
            print(f"Benchmark steps:     {BENCHMARK_STEPS}")
            print(f"Average latency:     {average_latency * 1000:.3f} ms")
            print(f"Algorithm bandwidth: {algorithm_bandwidth_gbps:.2f} GB/s")

            if world_size > 1:
                print(f"Bus bandwidth:       {bus_bandwidth_gbps:.2f} GB/s")
            else:
                print("Bus bandwidth:       N/A (single GPU)")

            metrics = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "gpu": torch.cuda.get_device_name(local_rank),
                "world_size": world_size,
                "tensor_size_mb": TENSOR_SIZE_MB,
                "warmup_steps": WARMUP_STEPS,
                "benchmark_steps": BENCHMARK_STEPS,
                "average_latency_ms": average_latency * 1000,
                "algorithm_bandwidth_gbps": algorithm_bandwidth_gbps,
                "bus_bandwidth_gbps": (
                    bus_bandwidth_gbps if world_size > 1 else None
                ),
            }

            common.save_metrics(
                metrics,
                common.NCCL_RESULTS_CSV,
            )





    finally:
        common.cleanup_distributed()

if __name__ == "__main__":
    main()