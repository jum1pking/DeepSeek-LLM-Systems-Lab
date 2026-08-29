import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

import common

from common import (
    MODEL_NAME,
    DTYPE,
)
from training.common import cleanup_distributed

# Start with ZeRO disabled to isolate DeepSpeed engine overhead before
# comparing ZeRO-1/2/3 under the same training configuration.

DS_CONFIG = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "bf16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 0,
    },
}

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Match the memory-saving behavior used by the existing LoRA baseline.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model,tokenizer

def build_optimizer(model):
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    return torch.optim.AdamW(
        trainable_parameters,
        lr=1e-4
    )

def main():
    # DeepSpeed initializes the distributed process group internally, so ensure
    # it is destroyed even when the training step raises an exception.
    try:
        model, tokenizer = load_model_and_tokenizer()
        model.print_trainable_parameters()
        optimizer = build_optimizer(model)

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=DS_CONFIG,
        )

        messages = [
            {
                "role": "user",
                "content": "Explain in one sentence what distributed training is.",
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        input_ids = inputs["input_ids"].to(model_engine.device)
        attention_mask = inputs["attention_mask"].to(model_engine.device)

        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()

        print("\n=== DeepSpeed Training Step ===")
        print(f"Loss:            {loss.item():.4f}")
        print(f"Peak VRAM:       {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print("Backward:        PASS")
        print("Optimizer step:  PASS")
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()