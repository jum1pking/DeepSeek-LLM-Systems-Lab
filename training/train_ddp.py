import os

import torch
import torch.distributed as dist
from datetime import datetime
from peft import LoraConfig, TaskType,get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

from common import (
    MODEL_NAME,
    DTYPE,
    BATCH_SIZE,
    MAX_LENGTH,
    MAX_STEPS,
    LEARNING_RATE,
    RESULTS_CSV,
)


# Each DDP process is bound to one GPU through LOCAL_RANK to avoid
# multiple processes competing for the same CUDA device.
def setup_distributed():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def load_model_and_tokenizer(local_rank):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE
    )

    model.to(local_rank)

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

def main():
    local_rank = setup_distributed()

    try:
        rank = dist.get_rank()

        # Global batch size grows with DDP world size while keeping the
        # per-device batch size fixed for scaling experiments.
        world_size = dist.get_world_size()

        dataset = common.load_training_dataset()
        model, tokenizer = load_model_and_tokenizer(local_rank)

        trainer = common.build_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
        )

        results = common.run_training(trainer)

        common.cleanup_distributed()

        # Only rank 0 writes shared experiment results to avoid duplicate CSV rows.
        if rank == 0:
            metrics = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "strategy": "ddp",
                "model": MODEL_NAME,
                "gpu": torch.cuda.get_device_name(local_rank),
                "precision": str(DTYPE),
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "max_steps": MAX_STEPS,
                "learning_rate": LEARNING_RATE,
                "world_size": world_size,
                "global_batch_size": BATCH_SIZE * world_size,
                **results,
            }

            common.save_metrics(metrics, RESULTS_CSV)

    finally:
        common.cleanup_distributed()

if __name__ == "__main__":
    main()