import common
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from common import (
    MODEL_NAME,
    DTYPE,
    BATCH_SIZE,
    MAX_LENGTH,
    MAX_STEPS,
    LEARNING_RATE,
)


def load_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Use 4-bit NF4 quantization to reduce base-model memory while keeping
    # BF16 compute for LoRA training, enabling a fair LoRA vs QLoRA comparison.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
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



    model = prepare_model_for_kbit_training(model)

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
        "strategy": "qlora",
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "precision": str(DTYPE),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        **results,
    }

    common.save_metrics(metrics,common.RESULTS_CSV)

if __name__ == "__main__":
    main()