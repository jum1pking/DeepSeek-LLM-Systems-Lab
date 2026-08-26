import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT =Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "datasets" / "tiny_sft.jsonl"

dataset = load_dataset(
    "json",
    data_files = str(DATA_PATH),
    split="train",
)

print(dataset)
print(dataset[0])
print("Number of samples:", len(dataset))

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type= TaskType.CAUSAL_LM,
)

print(lora_config)

MODEL_NAME ="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer= AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir = "results/lora_smoke_test",
    per_device_train_batch_size=1,
    learning_rate=1e-4,
    max_steps=3,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    max_length=128,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class= tokenizer,
)

trainer.train()

ADAPTER_PATH= "results/lora_smoke_test/adapter"

model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

print(f"LoRA adapter saved to: {ADAPTER_PATH}")