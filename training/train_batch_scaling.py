import argparse
from datetime import datetime

import common
import torch
import train_single
from trl import SFTConfig, SFTTrainer

from common import (
    MODEL_NAME,
    DTYPE,
    MAX_LENGTH,
    MAX_STEPS,
    LEARNING_RATE,
    OUTPUT_DIR,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-size scaling experiment for the no-GC single-GPU LoRA workload."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        choices=[2, 4, 8],
        help="Per-device training batch size to test.",
    )
    return parser.parse_args()

def print_experiment_config(dataset, model, batch_size):
    model.print_trainable_parameters()

    print("\n=== Experiment Configuration ===")
    print(f"Model:          {MODEL_NAME}")
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"Precision:      {DTYPE}")
    print(f"Batch size:     {batch_size}")
    print(f"Max length:     {MAX_LENGTH}")
    print(f"Max steps:      {MAX_STEPS}")
    print(f"Learning rate:  {LEARNING_RATE}")
    print(f"Dataset size:   {len(dataset)}")
    print("Gradient checkpointing: False")

def build_trainer(model, tokenizer, dataset, batch_size):
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=MAX_LENGTH,
        gradient_checkpointing=False,
    )

    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def main():
    args = parse_args()

    dataset = common.load_training_dataset()
    model, tokenizer = train_single.load_model_and_tokenizer()

    # Keep KV cache disabled so batch size is the only intended
    # experimental variable relative to the no-GC control.
    model.config.use_cache = False

    print_experiment_config(
        dataset=dataset,
        model=model,
        batch_size=args.batch_size,
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
    )

    results = common.run_training(trainer)

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy": f"lora_no_gc_batch{args.batch_size}",
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "precision": str(DTYPE),
        "batch_size": args.batch_size,
        "max_length": MAX_LENGTH,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        **results,
    }

    common.save_metrics(metrics, common.RESULTS_CSV)


if __name__ == "__main__":
    main()
