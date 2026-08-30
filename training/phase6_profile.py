import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

DTYPE = torch.bfloat16
PER_DEVICE_BATCH_SIZE = 2
SEQUENCE_LENGTH = 512

WARMUP_STEPS = 2
PROFILE_STEPS = 3
LEARNING_RATE = 1e-4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = (
    PROJECT_ROOT
    / "results"
    / "training"
    / "phase6_profiles"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        required=True,
        choices=[
            "ddp",
            "fsdp2",
        ],
    )

    return parser.parse_args()


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    return local_rank


def build_base_model(local_rank):
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

    return get_peft_model(
        model,
        lora_config,
    )


def wrap_model(
    model,
    strategy,
    local_rank,
):
    if strategy == "ddp":
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    base_model = model.get_base_model()

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


def get_vocab_size(model, strategy):
    if strategy == "ddp":
        return model.module.config.vocab_size

    return model.config.vocab_size


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
    dist.barrier()
    torch.cuda.synchronize()


def main():
    args = parse_args()
    local_rank = setup_distributed()

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 2:
            raise RuntimeError(
                "Phase 6 profiling expects exactly 2 GPUs."
            )

        if rank == 0:
            print("=== Phase 6.5 Communication Profile ===")
            print(f"Strategy: {args.strategy}")
            print(f"World size: {world_size}")
            print()

        model = build_base_model(
            local_rank=local_rank
        )

        model = wrap_model(
            model=model,
            strategy=args.strategy,
            local_rank=local_rank,
        )

        optimizer = build_optimizer(
            model=model
        )

        batch = build_batch(
            local_rank=local_rank,
            vocab_size=get_vocab_size(
                model,
                args.strategy,
            ),
        )

        model.train()

        for _ in range(WARMUP_STEPS):
            train_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
            )

        synchronize()

        TRACE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        trace_path = (
            TRACE_DIR
            / f"{args.strategy}_rank{rank}.json"
        )

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(PROFILE_STEPS):
                train_step(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                )
                prof.step()

        synchronize()

        prof.export_chrome_trace(
            str(trace_path)
        )

        print(
            f"Rank {rank}: trace saved to "
            f"{trace_path}"
        )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
