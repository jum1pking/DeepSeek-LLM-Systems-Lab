from trl import SFTConfig, SFTTrainer

from common import (
    OUTPUT_DIR,
    BATCH_SIZE,
    MAX_LENGTH,
    MAX_STEPS,
    LEARNING_RATE,
)


def build_trainer(
    model,
    tokenizer,
    dataset,
):
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
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