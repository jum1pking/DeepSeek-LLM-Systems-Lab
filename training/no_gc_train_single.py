from datetime import datetime

import common
import no_gc_common
import torch
import train_single

from common import (
    MODEL_NAME,
    DTYPE,
    BATCH_SIZE,
    MAX_LENGTH,
    MAX_STEPS,
    LEARNING_RATE,
)



def main():

    dataset = common.load_training_dataset()
    model, tokenizer = train_single.load_model_and_tokenizer()

    # Keep KV cache disabled so gradient checkpointing is
    # the only intended experimental variable.
    model.config.use_cache = False

    common.print_experiment_config(dataset, model)

    trainer = no_gc_common.build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    results = common.run_training(trainer)

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy": "lora_no_gc",
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
