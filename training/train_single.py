import common
import torch
from datetime import datetime
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    MODEL_NAME,
    DTYPE,
    BATCH_SIZE,
    MAX_LENGTH,
    MAX_STEPS,
    LEARNING_RATE,
)
def load_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(common.MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        common.MODEL_NAME,
        dtype=common.DTYPE,
        device_map="cuda",
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    return model, tokenizer

def main():
    dataset = common.load_training_dataset()
    model,tokenizer = load_model_and_tokenizer()

    common.print_experiment_config(dataset, model)

    trainer = common.build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    results = common.run_training(trainer)

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy": "lora",
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