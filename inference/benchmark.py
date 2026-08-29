import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import csv
from pathlib import Path
from datetime import datetime

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

PROMPT = "Explain in one sentence what distributed training is."

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda"
)

messages = [
    {
        "role": "user",
        "content": PROMPT,
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
).to("cuda")

# Reset VRAM statistics before the measured generation.
torch.cuda.reset_peak_memory_stats()

# CUDA operations are asynchronous, so synchronize before timing.
torch.cuda.synchronize()
WARMUP_RUNS = 1
BENCHMARK_RUNS = 5
print("Warming up...")

for _ in range(WARMUP_RUNS):
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

torch.cuda.synchronize()

latencies = []
throughputs = []
peak_vrams = []

print("Benchmarking...")
for i in range(BENCHMARK_RUNS):
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    input_tokens = inputs["input_ids"].shape[1]
    output_tokens = outputs.shape[1] - input_tokens

    latency =end_time - start_time
    throughput = output_tokens / latency
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    latencies.append(latency)
    throughputs.append(throughput)
    peak_vrams.append(peak_vram)

    print(
        f"Run {i + 1}:"
        f"{latency:.3f} s | "
        f"{throughput:.2f} tokens/s | "
        f"{peak_vram:.2f} GB"
    )

average_latency = sum(latencies) / len(latencies)
average_throughput = sum(throughputs) / len(throughputs)
max_peak_vram = max(peak_vrams)

print("=== Hugging Face Benchmark ===")
print(f"Input tokens:       {input_tokens}")
print(f"Output tokens:      {output_tokens}")
print(f"Generation latency: {average_latency:.3f} s")
print(f"Throughput:         {average_throughput: .2f} tokens/s")
print(f"Peak VRAM:          {max_peak_vram: .2f} GB")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "inference"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = RESULT_DIR / "hf_baseline.csv"

result ={
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "model": MODEL_NAME,
    "gpu": torch.cuda.get_device_name(0),
    "dtype": "bfloat16",
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "average_latency_s": round(average_latency, 3),
    "average_throughput_tok_s": round(average_throughput, 2),
    "peak_vram_gb": round(max_peak_vram, 2),
}

file_exists = RESULT_FILE.exists()
with RESULT_FILE.open("a", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=result.keys(),
    )

    if not file_exists:
        writer.writeheader()

    writer.writerow(result)

print(f"Results saved to: {RESULT_FILE}")