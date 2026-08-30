import csv
import os
from pathlib import Path

import torch


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PROMPT = "Explain in one sentence what distributed training is."

DEVICE = "cuda"
DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"

HF_WARMUP_RUNS = 1
VLLM_WARMUP_RUNS = 5
BENCHMARK_RUNS = 5
WARMUP_NEW_TOKENS = 32
BENCHMARK_NEW_TOKENS = 128

VLLM_GPU_MEMORY_UTILIZATION = 0.70
VLLM_MAX_MODEL_LEN = 4096

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "inference"


def build_messages():
    return [
        {
            "role": "user",
            "content": PROMPT,
        }
    ]


def average(values):
    return sum(values) / len(values)


def save_result(filename, result):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    result_file = RESULT_DIR / filename
    file_exists = result_file.exists()

    with result_file.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=result.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    return result_file


def configure_vllm_environment():
    # WSL2 does not provide the UVA path required by vLLM's V2 model runner.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    # Avoid the FlashInfer sampler JIT path that fails on this WSL2/Blackwell setup.
    # FlashAttention remains enabled; only the sampler backend is changed.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
