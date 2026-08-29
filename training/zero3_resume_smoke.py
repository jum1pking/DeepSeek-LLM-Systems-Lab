from pathlib import Path

from datetime import datetime
import deepspeed
import torch
import torch.distributed as dist
import common
import deepspeed_common

from deepspeed_common import (
    ZERO3_CHECKPOINT_DIR,
    ZERO3_CONFIG_PATH,
    load_zero3_model_and_tokenizer,
    prepare_smoke_batch,
)

from common import (
    MODEL_NAME,
)

CHECKPOINT_TAG = "step_1"


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    try:
        model, tokenizer, hf_ds_config = load_zero3_model_and_tokenizer()

        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=trainable_parameters,
            config=str(ZERO3_CONFIG_PATH),
        )

        load_path, client_state = model_engine.load_checkpoint(
            str(ZERO3_CHECKPOINT_DIR),
            tag=CHECKPOINT_TAG,
        )

        rank = dist.get_rank()

        if load_path is None:
            raise RuntimeError(
                f"Failed to load checkpoint: {CHECKPOINT_TAG}"
            )

        loaded_global_step = model_engine.global_steps

        if rank == 0:
            print("\n=== ZeRO-3 Resume Smoke Test ===")
            print(f"Loaded from: {load_path}")
            print(f"ZeRO stage:  {model_engine.zero_optimization_stage()}")
            print(f"Global step: {loaded_global_step}")

        batch = prepare_smoke_batch(
            tokenizer,
            model_engine.device,
        )

        input_ids = batch["input_ids"]

        torch.cuda.reset_peak_memory_stats()

        outputs = model_engine(
            **batch,
            labels=input_ids,
        )

        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()



        if rank == 0:
            peak_vram_gb = (
                    torch.cuda.max_memory_allocated()
                    / 1024 ** 3
            )

            final_global_step = model_engine.global_steps

            metrics = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "test": "zero3_checkpoint_resume",
                "model": MODEL_NAME,
                "gpu": torch.cuda.get_device_name(model_engine.device),
                "zero_stage": model_engine.zero_optimization_stage(),
                "checkpoint_tag": CHECKPOINT_TAG,
                "loaded_global_step": loaded_global_step,
                "final_global_step": final_global_step,
                "loss": loss.item(),
                "peak_vram_gb": peak_vram_gb,
                "status": "PASS",
            }

            common.save_metrics(
                metrics,
                deepspeed_common.DEEPSPEED_VALIDATION_CSV,
            )

            print("\n=== Resumed Training Step ===")
            print(f"Loss:        {loss.item():.4f}")
            print(f"Global step: {final_global_step}")
            print(f"Peak VRAM:   {peak_vram_gb:.2f} GB")
            print("Checkpoint:  PASS")
            print("Backward:    PASS")
            print("Step:        PASS")

    finally:
        common.cleanup_distributed()


if __name__ == "__main__":
    main()