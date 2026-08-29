import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER_PATH = "results/lora_smoke_test/adapter"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype = torch.bfloat16,
    device_map = "cuda",
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

messages = [
    {
        "role" : "user",
        "content" : "What is distributed training?"
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to("cuda")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_token=True,
)

print(response)