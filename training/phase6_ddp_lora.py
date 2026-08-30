import csv
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
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
    / "phase6_ddp_scaling.csv"
)


def log(message):
    print(message, flush=True)


def setup_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase 6 DDP benchmark.")

    # 2-GPU cloud run: torch.distributed.run provides these variables.
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        log(
            "Distributed setup: torchrun environment detected "
            f"(LOCAL_RANK={local_rank}, "
            f"RANK={os.environ.get('RANK')}, "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')})."
        )
    else:
        # 1-GPU baseline: run with plain Python.
        # Still initialize a real NCCL process group so 1-GPU and 2-GPU
        # execute the same DDP code path.
        local_rank = 0
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        log(
            "Distributed setup: direct-python 1-GPU mode; "
            "creating a 1-process NCCL process group."
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    log("Distributed setup: calling dist.init_process_group(...)")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=120),
        device_id=device,
    )
    log(
        "Distributed setup: process group ready "
        f"(rank={dist.get_rank()}, world_size={dist.get_world_size()})."
    )

    return local_rank, device


def build_model(local_rank, device):
    log(f"Model load: starting {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    log("Model load: checkpoint loaded on CPU.")

    model.config.use_cache = False
    model.to(device)
    log(
        "Model load: moved to GPU; "
        f"allocated={torch.cuda.memory_allocated(device) / 1024**3:.2f} GB."
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    log("LoRA: adapters injected.")

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    log("DDP: wrapper constructed.")

    return model


def build_optimizer(model):
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )
    log(
        "Optimizer: AdamW ready; "
        f"trainable parameters={trainable_count:,}."
    )

    return optimizer


def build_batch(device, vocab_size):
    generator = torch.Generator(device=device)
    generator.manual_seed(20260830 + dist.get_rank())

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(PER_DEVICE_BATCH_SIZE, SEQUENCE_LENGTH),
        generator=generator,
        device=device,
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    log(
        "Batch: ready "
        f"(batch={PER_DEVICE_BATCH_SIZE}, "
        f"seq={SEQUENCE_LENGTH}, "
        f"tokens/device={PER_DEVICE_BATCH_SIZE * SEQUENCE_LENGTH})."
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    return loss.detach()


def synchronize(device):
    torch.cuda.synchronize(device)

    if dist.is_initialized():
        dist.barrier()

    torch.cuda.synchronize(device)


def benchmark(model, optimizer, batch, device):
    model.train()
    rank = dist.get_rank()

    if rank == 0:
        log(f"Warmup: starting {WARMUP_STEPS} step(s).")

    for step in range(WARMUP_STEPS):
        loss = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
        )
        if rank == 0:
            log(
                f"Warmup: step {step + 1}/{WARMUP_STEPS}, "
                f"loss={loss.item():.4f}"
            )

    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        log(f"Benchmark: starting {BENCHMARK_STEPS} step(s).")

    start = time.perf_counter()
    final_loss = None

    for step in range(BENCHMARK_STEPS):
        final_loss = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
        )
        if rank == 0:
            log(
                f"Benchmark: step {step + 1}/{BENCHMARK_STEPS}, "
                f"loss={final_loss.item():.4f}"
            )

    synchronize(device)

    total_time_s = time.perf_counter() - start
    world_size = dist.get_world_size()

    global_tokens = (
        BENCHMARK_STEPS
        * PER_DEVICE_BATCH_SIZE
        * SEQUENCE_LENGTH
        * world_size
    )

    return {
        "total_time_s": total_time_s,
        "average_step_time_ms": (
            total_time_s / BENCHMARK_STEPS * 1000
        ),
        "global_tokens": global_tokens,
        "tokens_per_second": global_tokens / total_time_s,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3
        ),
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
    try:
        local_rank, device = setup_distributed()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            log("=== Phase 6 DDP LoRA Scaling ===")
            log(f"Model:             {MODEL_NAME}")
            log(f"GPU:               {torch.cuda.get_device_name(device)}")
            log(f"World size:        {world_size}")
            log(f"Per-device batch:  {PER_DEVICE_BATCH_SIZE}")
            log(
                f"Global batch:      "
                f"{PER_DEVICE_BATCH_SIZE * world_size}"
            )
            log(f"Sequence length:   {SEQUENCE_LENGTH}")
            log(f"Warmup steps:      {WARMUP_STEPS}")
            log(f"Benchmark steps:   {BENCHMARK_STEPS}")
            log("")

        model = build_model(
            local_rank=local_rank,
            device=device,
        )

        optimizer = build_optimizer(model)

        batch = build_batch(
            device=device,
            vocab_size=model.module.config.vocab_size,
        )

        result = benchmark(
            model=model,
            optimizer=optimizer,
            batch=batch,
            device=device,
        )

        if rank == 0:
            row = {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "strategy": "ddp_lora_weak_scaling",
                "model": MODEL_NAME,
                "gpu": torch.cuda.get_device_name(device),
                "precision": "bfloat16",
                "world_size": world_size,
                "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
                "global_batch_size": (
                    PER_DEVICE_BATCH_SIZE * world_size
                ),
                "sequence_length": SEQUENCE_LENGTH,
                "warmup_steps": WARMUP_STEPS,
                "benchmark_steps": BENCHMARK_STEPS,
                **result,
            }

            save_result(row)

            log("=== Result ===")
            log(
                f"Average step: "
                f"{result['average_step_time_ms']:.3f} ms"
            )
            log(
                f"Throughput:   "
                f"{result['tokens_per_second']:.2f} tok/s"
            )
            log(
                f"Peak VRAM:    "
                f"{result['peak_vram_gb']:.2f} GB"
            )
            log(
                f"Final loss:   "
                f"{result['final_loss']:.4f}"
            )
            log(f"Saved:        {RESULT_FILE}")

    finally:
        if dist.is_initialized():
            if dist.get_rank() == 0:
                log("Distributed cleanup: destroying process group.")
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
