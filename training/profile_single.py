# DEPRECATED:
# This PyTorch Profiler path is kept for reference, but it is no longer used
# for GPU profiling on the current RTX 5060 Laptop environment.
#
# PyTorch 2.11.0+cu128 fails to initialize CUPTI CUDA activities with
# CUPTI_ERROR_INVALID_DEVICE, so the generated traces contain CPU/operator
# information but no reliable CUDA kernel timing.
#
# The issue has been reproduced upstream on the same RTX 5060 Laptop GPU
# with CUDA 12.8. GPU timeline profiling is therefore continued with
# NVIDIA Nsight Systems instead of changing the validated training environment.

from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from transformers import TrainerCallback

import common
from train_single import load_model_and_tokenizer


PROFILER_DIR = (
    common.PROJECT_ROOT
    / "results"
    / "profiler"
    / "single_lora"
)

WAIT_STEPS = 2
WARMUP_STEPS = 2
ACTIVE_STEPS = 4


class ProfilerStepCallback(TrainerCallback):
    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()
def trace_handler(prof):
    print("\n=== Top CUDA Operators ===")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    )

    print("\n=== Top CPU Operators ===")
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=20,
        )
    )

    tensorboard_trace_handler(str(PROFILER_DIR))(prof)

def main():
    dataset = common.load_training_dataset()
    model, tokenizer = load_model_and_tokenizer()

    common.print_experiment_config(dataset, model)

    trainer = common.build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    PROFILER_DIR.mkdir(parents=True, exist_ok=True)

    # Keep profiling separate from the clean baseline so instrumentation
    # overhead does not contaminate baseline performance measurements.
    profiler_schedule = schedule(
        wait=WAIT_STEPS,
        warmup=WARMUP_STEPS,
        active=ACTIVE_STEPS,
        repeat=1,
    )

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=profiler_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        trainer.add_callback(
            ProfilerStepCallback(prof)
        )

        trainer.train()

    print("\n=== Profiling Complete ===")
    print(f"Trace directory: {PROFILER_DIR}")


if __name__ == "__main__":
    main()