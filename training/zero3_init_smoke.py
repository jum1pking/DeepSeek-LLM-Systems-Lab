import deepspeed
import torch
import common
import deepspeed_common
import torch.distributed as dist
from datetime import datetime


from common import (
    MODEL_NAME,
    cleanup_distributed,
)

from deepspeed_common import(
    ZERO3_CONFIG_PATH,
    ZERO3_CHECKPOINT_DIR,
    load_zero3_model_and_tokenizer,
    prepare_smoke_batch,
)

def main():
    try:
        model, tokenizer, hf_ds_config = load_zero3_model_and_tokenizer()

        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        trainable_param_count = sum(
            getattr(parameter, "ds_numel", parameter.numel())
            for parameter in trainable_parameters
        )

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=trainable_parameters,
            config=str(ZERO3_CONFIG_PATH),
        )

        rank = dist.get_rank()

        if rank == 0:
            print("\n=== ZeRO-3 Init Smoke Test ===")
            print(f"Engine:      {type(model_engine)}")
            print(f"Device:      {model_engine.device}")
            print(
                f"ZeRO stage:  "
                f"{model_engine.zero_optimization_stage()}"
            )
            print(f"Trainable:   {trainable_param_count:,}")

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

        checkpoint_tag = "step_1"

        model_engine.save_checkpoint(
            str(ZERO3_CHECKPOINT_DIR),
            tag=checkpoint_tag,
        )

        if rank == 0:
            peak_vram_gb = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            metrics = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "test": "zero3_checkpoint_save",
                "model": MODEL_NAME,
                "gpu": torch.cuda.get_device_name(model_engine.device),
                "zero_stage": model_engine.zero_optimization_stage(),
                "checkpoint_tag": checkpoint_tag,
                "loaded_global_step": "",
                "final_global_step": model_engine.global_steps,
                "loss": loss.item(),
                "peak_vram_gb": peak_vram_gb,
                "status": "PASS",
            }

            common.save_metrics(
                metrics,
                deepspeed_common.DEEPSPEED_VALIDATION_CSV,
            )

            print("\n=== Training Step ===")
            print(f"Loss:       {loss.item():.4f}")
            print(f"Peak VRAM:  {peak_vram_gb:.2f} GB")
            print("Backward:   PASS")
            print("Step:       PASS")
            print(
                f"Checkpoint:  "
                f"{ZERO3_CHECKPOINT_DIR / checkpoint_tag}"
            )

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()