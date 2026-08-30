import csv
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

DTYPE = torch.bfloat16
PER_DEVICE_BATCH_SIZE = 2
SEQUENCE_LENGTH = 512

WARMUP_STEPS = 3
BENCHMARK_STEPS = 10

LEARNING_RATE = 1e-4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "training"
    / "phase6_strategy_comparison.csv"
)


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    return local_rank


def build_model(local_rank):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.to(local_rank)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    base_model = model.get_base_model()

    # Shard transformer layers individually first, then the root PEFT model.
    # This keeps only the currently active layer unsharded during execution.
    for layer in base_model.model.layers:
        fully_shard(layer)

    fully_shard(model)

    return model


def build_optimizer(model):
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    return torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )


def build_batch(local_rank, vocab_size):
    generator = torch.Generator(
        device=f"cuda:{local_rank}"
    )

    generator.manual_seed(
        20260830 + dist.get_rank()
    )

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            PER_DEVICE_BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        generator=generator,
        device=local_rank,
        dtype=torch.long,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def train_step(model, optimizer, batch):
    optimizer.zero_grad(
        set_to_none=True
    )

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    return loss.detach()


def synchronize():
    torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()

    torch.cuda.synchronize()


def distributed_max(value, local_rank):
    tensor = torch.tensor(
        [value],
        dtype=torch.float64,
        device=local_rank,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.MAX,
    )

    return tensor.item()


def benchmark(
    model,
    optimizer,
    batch,
    local_rank,
):
    model.train()

    for _ in range(WARMUP_STEPS):
        train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
        )

    synchronize()

    torch.cuda.reset_peak_memory_stats(
        local_rank
    )

    start = time.perf_counter()
    final_loss = None

    for _ in range(BENCHMARK_STEPS):
        final_loss = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
        )

    synchronize()

    total_time_s = time.perf_counter() - start
    world_size = dist.get_world_size()

    global_tokens = (
        BENCHMARK_STEPS
        * PER_DEVICE_BATCH_SIZE
        * SEQUENCE_LENGTH
        * world_size
    )

    local_peak_vram_gb = (
        torch.cuda.max_memory_allocated(local_rank)
        / 1024**3
    )

    max_peak_vram_gb = distributed_max(
        local_peak_vram_gb,
        local_rank,
    )

    return {
        "total_time_s": total_time_s,
        "average_step_time_ms": (
            total_time_s
            / BENCHMARK_STEPS
            * 1000
        ),
        "global_tokens": global_tokens,
        "tokens_per_second": (
            global_tokens
            / total_time_s
        ),
        "peak_vram_gb": max_peak_vram_gb,
        "final_loss": (
            final_loss.item()
            if final_loss is not None
            else None
        ),
    }


def save_result(row):
    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = RESULT_FILE.exists()

    with RESULT_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    local_rank = setup_distributed()

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 2:
            raise RuntimeError(
                "Phase 6 FSDP2 benchmark expects exactly 2 GPUs."
            )

        if rank == 0:
            print("=== Phase 6.3 FSDP2 LoRA ===")
            print(f"Model:             {MODEL_NAME}")
            print(f"GPU:               {torch.cuda.get_device_name(local_rank)}")
            print(f"World size:        {world_size}")
            print(f"Per-device batch:  {PER_DEVICE_BATCH_SIZE}")
            print(f"Global batch:      {PER_DEVICE_BATCH_SIZE * world_size}")
            print(f"Sequence length:   {SEQUENCE_LENGTH}")
            print(f"Warmup steps:      {WARMUP_STEPS}")
            print(f"Benchmark steps:   {BENCHMARK_STEPS}")
            print()

        model = build_model(
            local_rank=local_rank
        )

        optimizer = build_optimizer(
            model=model
        )

        batch = build_batch(
            local_rank=local_rank,
            vocab_size=model.config.vocab_size,
        )

        result = benchmark(
            model=model,
            optimizer=optimizer,
            batch=batch,
            local_rank=local_rank,
        )

        if rank == 0:
            row = {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "strategy": "fsdp2_lora",
                "model": MODEL_NAME,
                "gpu": torch.cuda.get_device_name(local_rank),
                "precision": "bfloat16",
                "world_size": world_size,
                "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
                "global_batch_size": (
                    PER_DEVICE_BATCH_SIZE
                    * world_size
                ),
                "sequence_length": SEQUENCE_LENGTH,
                "warmup_steps": WARMUP_STEPS,
                "benchmark_steps": BENCHMARK_STEPS,
                **result,
            }

            save_result(row)

            print("=== Result ===")
            print(
                f"Average step: "
                f"{result['average_step_time_ms']:.3f} ms"
            )
            print(
                f"Throughput:   "
                f"{result['tokens_per_second']:.2f} tok/s"
            )
            print(
                f"Peak VRAM:    "
                f"{result['peak_vram_gb']:.2f} GB"
            )
            print(
                f"Final loss:   "
                f"{result['final_loss']:.4f}"
            )
            print(
                f"Saved:        {RESULT_FILE}"
            )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
