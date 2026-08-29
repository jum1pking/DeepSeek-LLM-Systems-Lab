import argparse
import time

import deepspeed
import torch
from datetime import datetime
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    DTYPE,
    LEARNING_RATE,
    MODEL_NAME,
    MAX_LENGTH,
    MAX_STEPS,
    PROJECT_ROOT,
    cleanup_distributed,
    load_training_dataset,
    save_metrics,
    BATCH_SIZE,
    DEEPSPEED_RESULTS_CSV,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSpeed ZeRO stages."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=[
            "zero0",
            "zero1",
            "zero2",
            "zero3",
            "zero2_cpu_offload",
            "zero3_cpu_offload",
        ],
        help="DeepSpeed ZeRO optimization stage.",
    )

    return parser.parse_args()

def get_deepspeed_config(config_name):
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "deepspeed"
        / f"{config_name}.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"DeepSpeed config not found: {config_path}"
        )

    return config_path

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Keep the LoRA configuration identical across ZeRO stages so the
    # benchmark isolates the effect of ZeRO partitioning.

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, tokenizer

def get_trainable_parameters(model):
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def build_optimizer(trainable_parameters):
    return torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

def prepare_batch(tokenizer, example, device):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    return  {
        key: value.to(device)
        for key, value in batch.items()
    }


def run_training(model_engine, tokenizer, dataset):
    model_engine.train()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.reset_peak_memory_stats()

    dist.barrier()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    total_tokens = 0
    final_loss = 0.0

    for step in range(MAX_STEPS):
        # Give each rank a different sample so multi-GPU ZeRO runs do not
        # repeatedly train on the same example at each global step.
        sample_index = (
            step * world_size + rank
        ) % len(dataset)

        batch = prepare_batch(
            tokenizer,
            dataset[sample_index],
            model_engine.device,
        )

        input_ids = batch["input_ids"]

        output = model_engine(
            **batch,
            labels=input_ids,
        )

        loss = output.loss

        model_engine.backward(loss)
        model_engine.step()

        final_loss = loss.item()
        total_tokens += input_ids.numel()

    torch.cuda.synchronize()
    dist.barrier()

    total_time = time.perf_counter() - start_time

    token_tensor = torch.tensor(
        total_tokens,
        dtype=torch.long,
        device=model_engine.device,
    )

    dist.all_reduce(
        token_tensor,
        op=dist.ReduceOp.SUM,
    )

    global_tokens = token_tensor.item()

    average_step_time = total_time / MAX_STEPS
    tokens_per_second = global_tokens / total_time

    peak_vram_gb =(
        torch.cuda.max_memory_allocated() / 1024**3
    )

    return {
        "total_time_s": total_time,
        "average_step_time_s": average_step_time,
        "peak_vram_gb": peak_vram_gb,
        "train_loss": final_loss,
        "tokens_per_second": tokens_per_second,
    }


def main():
    args = parse_args()

    # Keep the same model, LoRA setup, batch size, sequence length, and step count
    # across ZeRO stages so partitioning strategy is the primary changed variable.
    config_path = get_deepspeed_config(args.config)

    try:
        model, tokenizer = load_model_and_tokenizer()

        # Let DeepSpeed create its CPU-aware optimizer for ZeRO-Offload configs;
        # non-offload baselines keep the client-provided AdamW optimizer.
        trainable_parameters = get_trainable_parameters(model)

        trainable_param_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        uses_cpu_offload = "cpu_offload" in args.config

        if uses_cpu_offload:
            optimizer = None
        else:
            optimizer = build_optimizer(model)

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=(
                trainable_parameters
                if uses_cpu_offload
                else None
            ),
            config=str(config_path),
        )

        zero_stage = model_engine.zero_optimization_stage()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if model_engine.global_rank == 0:
            print("\n=== DeepSpeed Configuration ===")
            print(f"Config name: {args.config}")
            print(f"ZeRO stage:  {model_engine.zero_optimization_stage()}")
            print(f"Config path: {config_path}")
            print(f"Engine:      {type(model_engine)}")
            print(f"Device:      {model_engine.device}")
            print(f"Trainable:   {trainable_param_count:,}")

        dataset = load_training_dataset()

        results = run_training(
            model_engine=model_engine,
            tokenizer=tokenizer,
            dataset=dataset,
        )

        if rank == 0:
            metrics = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "strategy": args.config,
                "model": MODEL_NAME,
                "gpu": torch.cuda.get_device_name(model_engine.device),
                "precision": str(DTYPE),
                "zero_stage": zero_stage,
                "world_size": world_size,
                "batch_size": BATCH_SIZE,
                "global_batch_size": BATCH_SIZE * world_size,
                "max_length": MAX_LENGTH,
                "max_steps": MAX_STEPS,
                "learning_rate": LEARNING_RATE,
                **results,
            }

            save_metrics(
                metrics,
                DEEPSPEED_RESULTS_CSV,
            )

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()