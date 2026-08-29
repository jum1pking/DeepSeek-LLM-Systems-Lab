from pathlib import Path


import time
import torch
import csv
import torch.distributed as dist
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use one shared metrics file so all training strategies follow the same schema.
RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "training"
    / "training_baselines.csv"
)

# Store NCCL communication benchmarks separately from model-training metrics.
NCCL_RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "training"
    / "nccl_benchmark.csv"
)

# Store DeepSpeed ZeRO benchmark results separately for stage-by-stage comparison.
DEEPSPEED_RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "training"
    / "deepspeed_baselines.csv"
)


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = PROJECT_ROOT / "datasets" / "tiny_sft.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "results" / "training" / "single_gpu_baseline"

DTYPE = torch.bfloat16
BATCH_SIZE = 1
MAX_LENGTH = 128
MAX_STEPS = 20
LEARNING_RATE = 1e-4


def load_training_dataset():
    return load_dataset(
        "json",
        data_files=str(DATA_PATH),
        split="train",
)

def print_experiment_config(dataset, model):
    model.print_trainable_parameters()

    print("\n=== Experiment Configuration ===")
    print(f"Model:          {MODEL_NAME}")
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"Precision:      {DTYPE}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Max length:     {MAX_LENGTH}")
    print(f"Max steps:      {MAX_STEPS}")
    print(f"Learning rate:  {LEARNING_RATE}")
    print(f"Dataset size:   {len(dataset)}")

def build_trainer(model, tokenizer, dataset):
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=MAX_LENGTH,
        gradient_checkpointing=True
    )

    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

def run_training(trainer):
    torch.cuda.reset_peak_memory_stats()
    synchronize_workers()

    start_time = time.perf_counter()

    train_result = trainer.train()

    synchronize_workers()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    average_step_time = total_time / MAX_STEPS
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    # Use the trainer's cumulative processed-token count to derive end-to-end
    # training throughput for comparison with later DDP/FSDP2 experiments.
    num_tokens = 0

    for log in reversed(trainer.state.log_history):
        if "num_tokens" in log:
            num_tokens = log["num_tokens"]
            break

    tokens_per_second = num_tokens / total_time if total_time > 0 else 0.0

    is_main_process = (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
    )

    if is_main_process:
        print("\n=== Training Results ===")
        print(f"Total training time: {total_time:.3f} s")
        print(f"Average step time:   {average_step_time:.3f} s")
        print(f"Peak VRAM:           {peak_vram_gb:.2f} GB")
        print(f"Final train loss:    {train_result.training_loss:.4f}")
        print(f"Tokens/s:            {tokens_per_second:.2f}")

    return {
        "total_time_s": total_time,
        "average_step_time_s": average_step_time,
        "peak_vram_gb": peak_vram_gb,
        "train_loss": train_result.training_loss,
        "tokens_per_second": tokens_per_second,
    }

def save_metrics(metrics, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)

# Synchronize all DDP ranks so distributed timing reflects the slowest worker
# rather than measuring each GPU from a different starting point.
def synchronize_workers():
    """Synchronize CUDA work and distributed workers before benchmarking."""
    torch.cuda.synchronize()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
