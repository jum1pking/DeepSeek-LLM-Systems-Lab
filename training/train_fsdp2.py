import os

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

import common

from common import (
    MODEL_NAME,
    DTYPE,
)

def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    return local_rank

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

    return model, tokenizer


def apply_fsdp2(model):
    base_model = model.get_base_model()

    # Shard transformer layers individually before sharding the root model so
    # only the active layer needs to be unsharded during forward/backward.
    for layer in base_model.model.layers:
        fully_shard(layer)

    fully_shard(model)

    return model

def inspect_model_structure(model):
    base_model = model.get_base_model()

    print("\n=== FSDP2 Model Structure ===")
    print(f"Base model type: {type(base_model)}")
    print(f"Inner model type: {type(base_model.model)}")
    print(f"Number of layers: {len(base_model.model.layers)}")
    print(f"First layer type: {type(base_model.model.layers[0])}")

def main():
    local_rank = setup_distributed()

    try:
        rank = dist.get_rank()

        model, tokenizer = load_model_and_tokenizer()

        model = apply_fsdp2(model)

        dataset = common.load_training_dataset()

        trainer = common.build_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
        )

        results = common.run_training(trainer)

        world_size = dist.get_world_size()

        if rank == 0:
            metrics = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "strategy": "fsdp2",
                "model": common.MODEL_NAME,
                "gpu": torch.cuda.get_device_name(local_rank),
                "precision": str(common.DTYPE),
                "batch_size": common.BATCH_SIZE,
                "global_batch_size": common.BATCH_SIZE * world_size,
                "world_size": world_size,
                "max_length": common.MAX_LENGTH,
                "max_steps": common.MAX_STEPS,
                "learning_rate": common.LEARNING_RATE,
                **results,
            }

            common.save_metrics(
                metrics,
                common.RESULTS_CSV,
            )

    finally:
            common.cleanup_distributed()


if __name__ == "__main__":
    main()