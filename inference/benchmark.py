import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common


def load_model_and_inputs():
    tokenizer = AutoTokenizer.from_pretrained(common.MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        common.MODEL_NAME,
        dtype=common.DTYPE,
        device_map=common.DEVICE,
    )

    prompt = tokenizer.apply_chat_template(
        common.build_messages(),
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(common.DEVICE)

    return model, tokenizer, inputs


def run_once(model, inputs, max_new_tokens):
    # Reset VRAM statistics before the measured generation.
    torch.cuda.reset_peak_memory_stats()

    # CUDA operations are asynchronous, so synchronize before timing.
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    input_tokens = inputs["input_ids"].shape[1]
    output_tokens = outputs.shape[1] - input_tokens

    latency = end_time - start_time
    throughput = output_tokens / latency
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": latency,
        "throughput_tok_s": throughput,
        "peak_vram_gb": peak_vram_gb,
    }


def warmup(model, inputs):
    print("Warming up...")

    for _ in range(common.HF_WARMUP_RUNS):
        run_once(
            model=model,
            inputs=inputs,
            max_new_tokens=common.WARMUP_NEW_TOKENS,
        )


def main():
    model, tokenizer, inputs = load_model_and_inputs()

    warmup(model, inputs)

    runs = []

    print("Benchmarking...")

    for i in range(common.BENCHMARK_RUNS):
        result = run_once(
            model=model,
            inputs=inputs,
            max_new_tokens=common.BENCHMARK_NEW_TOKENS,
        )
        runs.append(result)

        print(
            f"Run {i + 1}: "
            f"{result['latency_s']:.3f} s | "
            f"{result['throughput_tok_s']:.2f} tokens/s | "
            f"{result['peak_vram_gb']:.2f} GB"
        )

    average_latency = common.average(
        [run["latency_s"] for run in runs]
    )
    average_throughput = common.average(
        [run["throughput_tok_s"] for run in runs]
    )
    max_peak_vram = max(run["peak_vram_gb"] for run in runs)

    input_tokens = runs[-1]["input_tokens"]
    output_tokens = runs[-1]["output_tokens"]

    print("=== Hugging Face Benchmark ===")
    print(f"Input tokens:       {input_tokens}")
    print(f"Output tokens:      {output_tokens}")
    print(f"Generation latency: {average_latency:.3f} s")
    print(f"Throughput:         {average_throughput:.2f} tokens/s")
    print(f"Peak VRAM:          {max_peak_vram:.2f} GB")

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": common.MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": common.DTYPE_NAME,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "average_latency_s": round(average_latency, 3),
        "average_throughput_tok_s": round(average_throughput, 2),
        "peak_vram_gb": round(max_peak_vram, 2),
    }

    result_file = common.save_result(
        "hf_baseline.csv",
        result,
    )
    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    main()
