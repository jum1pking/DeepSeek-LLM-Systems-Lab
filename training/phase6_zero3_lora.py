import csv
import os
import time
from datetime import datetime
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

DTYPE = torch.bfloat16
PER_DEVICE_BATCH_SIZE = 2
SEQUENCE_LENGTH = 512

WARMUP_STEPS = 3
BENCHMARK_STEPS = 10

LEARNING_RATE = 1e-4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "phase6_zero3.json"
)
RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "training"
    / "phase6_strategy_comparison.csv"
)


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    deepspeed.init_distributed(
        dist_backend="nccl"
    )

    return local_rank


def build_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

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

    return get_peft_model(
        model,
        lora_config,
    )


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


def build_batch(device, vocab_size):
    rank = dist.get_rank()

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        20260830 + rank
    )

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            PER_DEVICE_BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        generator=generator,
        device=device,
        dtype=torch.long,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def train_step(engine, batch):
    outputs = engine(**batch)
    loss = outputs.loss

    engine.backward(loss)
    engine.step()

    return loss.detach()


def synchronize():
    torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()

    torch.cuda.synchronize()


def distributed_max(value, device):
    tensor = torch.tensor(
        [value],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.MAX,
    )

    return tensor.item()


def benchmark(
    engine,
    batch,
):
    engine.train()

    for _ in range(WARMUP_STEPS):
        train_step(
            engine=engine,
            batch=batch,
        )

    synchronize()

    torch.cuda.reset_peak_memory_stats(
        engine.device
    )

    start = time.perf_counter()
    final_loss = None

    for _ in range(BENCHMARK_STEPS):
        final_loss = train_step(
            engine=engine,
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
        torch.cuda.max_memory_allocated(
            engine.device
        )
        / 1024**3
    )

    max_peak_vram_gb = distributed_max(
        local_peak_vram_gb,
        engine.device,
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
                "Phase 6 ZeRO-3 benchmark expects exactly 2 GPUs."
            )

        if rank == 0:
            print("=== Phase 6.4 DeepSpeed ZeRO-3 LoRA ===")
            print(f"Model:             {MODEL_NAME}")
            print(f"GPU:               {torch.cuda.get_device_name(local_rank)}")
            print(f"World size:        {world_size}")
            print(f"Per-device batch:  {PER_DEVICE_BATCH_SIZE}")
            print(f"Global batch:      {PER_DEVICE_BATCH_SIZE * world_size}")
            print(f"Sequence length:   {SEQUENCE_LENGTH}")
            print(f"Warmup steps:      {WARMUP_STEPS}")
            print(f"Benchmark steps:   {BENCHMARK_STEPS}")
            print(f"Config:            {CONFIG_FILE}")
            print()

        model = build_model()
        vocab_size = model.config.vocab_size

        optimizer = build_optimizer(
            model=model
        )

        engine, optimizer, _, _ = (
            deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=str(CONFIG_FILE),
            )
        )

        batch = build_batch(
            device=engine.device,
            vocab_size=vocab_size,
        )

        result = benchmark(
            engine=engine,
            batch=batch,
        )

        if rank == 0:
            row = {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "strategy": "deepspeed_zero3_lora",
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
