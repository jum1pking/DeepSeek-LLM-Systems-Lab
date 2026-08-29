import argparse
from datetime import datetime

import common
import no_gc_common
import torch
import train_single
from torch.nn.attention import SDPBackend, sdpa_kernel

from common import (
    MODEL_NAME,
    DTYPE,
    BATCH_SIZE,
    MAX_LENGTH,
    MAX_STEPS,
    LEARNING_RATE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SDPA backends on the no-GC single-GPU LoRA workload."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["flash", "math"],
        help="SDPA backend to force during training.",
    )
    return parser.parse_args()


def get_backend(backend_name):
    if backend_name == "flash":
        return SDPBackend.FLASH_ATTENTION

    return SDPBackend.MATH


def main():
    args = parse_args()

    dataset = common.load_training_dataset()
    model, tokenizer = train_single.load_model_and_tokenizer()

    # Keep KV cache disabled so the SDPA backend is the only intended
    # experimental variable relative to the no-GC control.
    model.config.use_cache = False

    common.print_experiment_config(dataset, model)
    print(f"Gradient checkpointing: False")
    print(f"SDPA backend:           {args.backend}")

    trainer = no_gc_common.build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    backend = get_backend(args.backend)

    # Force exactly one SDPA backend for the full clean training run.
    with sdpa_kernel(backend):
        results = common.run_training(trainer)

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy": f"lora_no_gc_sdpa_{args.backend}",
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "precision": str(DTYPE),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        **results,
    }

    common.save_metrics(metrics, common.RESULTS_CSV)


if __name__ == "__main__":
    main()
