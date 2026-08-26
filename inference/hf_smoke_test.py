import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype = torch.bfloat16,
    device_map="cuda"
)

print("Model loaded successfully")

print("Tokenizer loaded successfully.")

messages =[
    {
        "role": "user",
        "content": "Explain in one sentence what distributed training is."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to("cuda")

# print the struct of prompt

# print(prompt)
# print(inputs["input_ids"].shape)

token_ids = inputs["input_ids"][0]

# print input by token

# print("Token IDs:")
# print(token_ids.tolist())
#
# print("\nTokens:")
# print(tokenizer.convert_ids_to_tokens(token_ids))

print("Generating...")

torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

print("\n=== Model Response ===")
print(response)

print("\n=== GPU Memory ===")
print(f"Peak VRAM: {peak_vram_gb:2f} GB")