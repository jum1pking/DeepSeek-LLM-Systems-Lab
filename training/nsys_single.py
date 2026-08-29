# LIMITATION:
# Nsight Systems is kept for CUDA API tracing on the current environment,
# but it is not used for GPU kernel-level analysis.
#
# On the RTX 5060 Laptop + Windows 10/WSL2 setup, Nsight Systems 2026.4.1
# successfully captures CUDA API activity (e.g. kernel launches, stream
# synchronization, and memory copies), but both `cuda` and `cuda-sw`
# tracing produce no CUDA kernel timeline data.
#
# Kernel-level profiling is therefore continued with NVIDIA Nsight Compute
# (`ncu`), while this script is retained for CUDA API and NVTX tracing.
import torch

import common
from train_single import load_model_and_tokenizer
from transformers import TrainerCallback

PROFILE_STEP = 8
NVTX_RANGE_NAME = "lora_training"

class StepNvtxCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == PROFILE_STEP:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"profile_step_{PROFILE_STEP}")
            torch.cuda.cudart().cudaProfilerStart()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == PROFILE_STEP + 1:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.nvtx.range_pop()

def main():
    dataset = common.load_training_dataset()
    model, tokenizer = load_model_and_tokenizer()

    common.print_experiment_config(dataset, model)

    trainer = common.build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    trainer.add_callback(StepNvtxCallback())

    # Keep setup work outside the capture range so the Nsight Systems trace
    # represents training rather than model/data initialization.
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(NVTX_RANGE_NAME)

    try:
        trainer.train()

        # Ensure asynchronous CUDA work from the final training step finishes
        # before the NVTX capture range is closed.
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()

    print("\n=== Nsight Systems Training Run Complete ===")


if __name__ == "__main__":
    main()