import gc
import time

from datetime import datetime

import bitsandbytes as bnb
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import common
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization",
)

QUANTIZATION_MODES = ["bf16", "int8", "nf4"]#
WARMUP_RUNS_PER_MODE = 2
RESULT_FILE = "hf_quantization_comparison.csv"


def build_quantization_config(mode):
    if mode == "bf16":
        return None

    if mode == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    if mode == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=common.DTYPE,
            bnb_4bit_use_double_quant=False,
        )

    raise ValueError(f"Unsupported quantization mode: {mode}")


def load_model_and_inputs(mode):
    tokenizer = AutoTokenizer.from_pretrained(
        common.MODEL_NAME,
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

    quantization_config = build_quantization_config(mode)

    load_start = time.perf_counter()

    model_dtype = (
        torch.float16
        if mode == "int8"
        else common.DTYPE
    )

    model = AutoModelForCausalLM.from_pretrained(
        common.MODEL_NAME,
        dtype=model_dtype,
        device_map=common.DEVICE,
        quantization_config=quantization_config,
    )

    torch.cuda.synchronize()
    load_end = time.perf_counter()

    resident_vram_gb = (
        torch.cuda.memory_allocated() / 1024**3
    )

    return (
        model,
        tokenizer,
        inputs,
        load_end - load_start,
        resident_vram_gb,
    )


def run_once(model, inputs):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )
        past_key_values = outputs.past_key_values

        torch.cuda.synchronize()
        first_token_time = time.perf_counter()

        generated_tokens = [next_token]

        for _ in range(common.BENCHMARK_NEW_TOKENS - 1):
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )

            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )
            past_key_values = outputs.past_key_values
            generated_tokens.append(next_token)

    torch.cuda.synchronize()
    total_end = time.perf_counter()

    output_tokens = len(generated_tokens)
    decode_tokens = max(output_tokens - 1, 0)

    ttft_s = first_token_time - total_start
    decode_time_s = total_end - first_token_time
    end_to_end_latency_s = total_end - total_start

    average_tpot_s = (
        decode_time_s / decode_tokens
        if decode_tokens > 0
        else 0.0
    )

    decode_throughput_tok_s = (
        decode_tokens / decode_time_s
        if decode_time_s > 0 and decode_tokens > 0
        else 0.0
    )

    end_to_end_throughput_tok_s = (
        output_tokens / end_to_end_latency_s
        if end_to_end_latency_s > 0
        else 0.0
    )

    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    return {
        "input_tokens": input_ids.shape[1],
        "output_tokens": output_tokens,
        "ttft_ms": ttft_s * 1000,
        "average_tpot_ms": average_tpot_s * 1000,
        "decode_throughput_tok_s": decode_throughput_tok_s,
        "end_to_end_latency_s": end_to_end_latency_s,
        "end_to_end_throughput_tok_s": (
            end_to_end_throughput_tok_s
        ),
        "peak_vram_gb": peak_vram_gb,
    }


def warmup(model, inputs, mode):
    print(f"Warming up {mode}...")

    for _ in range(WARMUP_RUNS_PER_MODE):
        run_once(
            model=model,
            inputs=inputs,
        )


def benchmark_mode(model, inputs, mode):
    runs = []

    print(f"\nBenchmarking {mode}...")

    for i in range(common.BENCHMARK_RUNS):
        result = run_once(
            model=model,
            inputs=inputs,
        )
        runs.append(result)

        print(
            f"Run {i + 1}: "
            f"TTFT={result['ttft_ms']:.2f} ms | "
            f"TPOT={result['average_tpot_ms']:.2f} ms | "
            f"Decode={result['decode_throughput_tok_s']:.2f} tok/s | "
            f"E2E={result['end_to_end_latency_s']:.3f} s | "
            f"Peak VRAM={result['peak_vram_gb']:.2f} GB"
        )

    return runs


def summarize_mode(
    mode,
    runs,
    load_time_s,
    resident_vram_gb,
):
    weight_bits = {
        "bf16": 16,
        "int8": 8,
        "nf4": 4,
    }[mode]

    quantization_backend = {
        "bf16": "none",
        "int8": "bitsandbytes_llm_int8",
        "nf4": "bitsandbytes_nf4",
    }[mode]

    return {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "model": common.MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "bitsandbytes_version": bnb.__version__,
        "mode": mode,
        "weight_bits": weight_bits,
        "quantization_backend": quantization_backend,
        "compute_dtype": (
            "float16"
            if mode == "int8"
            else common.DTYPE_NAME
        ),
        "timing_scope": "manual_decode_loop",
        "kv_cache": True,
        "input_tokens": runs[0]["input_tokens"],
        "output_tokens": runs[0]["output_tokens"],
        "load_time_s": round(load_time_s, 3),
        "resident_vram_gb": round(
            resident_vram_gb,
            3,
        ),
        "average_ttft_ms": round(
            common.average(
                [run["ttft_ms"] for run in runs]
            ),
            3,
        ),
        "average_tpot_ms": round(
            common.average(
                [run["average_tpot_ms"] for run in runs]
            ),
            3,
        ),
        "average_decode_throughput_tok_s": round(
            common.average(
                [
                    run["decode_throughput_tok_s"]
                    for run in runs
                ]
            ),
            2,
        ),
        "average_end_to_end_latency_s": round(
            common.average(
                [
                    run["end_to_end_latency_s"]
                    for run in runs
                ]
            ),
            3,
        ),
        "average_end_to_end_throughput_tok_s": round(
            common.average(
                [
                    run["end_to_end_throughput_tok_s"]
                    for run in runs
                ]
            ),
            2,
        ),
        "peak_vram_gb": round(
            max(
                run["peak_vram_gb"]
                for run in runs
            ),
            3,
        ),
    }


def print_summary(summary):
    print(
        f"\n=== {summary['mode']} Average ==="
    )
    print(
        f"Resident VRAM:      "
        f"{summary['resident_vram_gb']:.3f} GB"
    )
    print(
        f"Peak VRAM:          "
        f"{summary['peak_vram_gb']:.3f} GB"
    )
    print(
        f"TTFT:               "
        f"{summary['average_ttft_ms']:.2f} ms"
    )
    print(
        f"TPOT:               "
        f"{summary['average_tpot_ms']:.2f} ms/token"
    )
    print(
        "Decode throughput:  "
        f"{summary['average_decode_throughput_tok_s']:.2f} tokens/s"
    )
    print(
        "End-to-end latency: "
        f"{summary['average_end_to_end_latency_s']:.3f} s"
    )


def unload_model(model, tokenizer, inputs):
    del model
    del tokenizer
    del inputs

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def print_comparison(summaries):
    baseline = summaries["bf16"]

    print("\n=== Quantization Comparison vs BF16 ===")

    for mode in QUANTIZATION_MODES:
        summary = summaries[mode]

        memory_ratio = (
            summary["peak_vram_gb"]
            / baseline["peak_vram_gb"]
        )

        throughput_ratio = (
            summary["average_decode_throughput_tok_s"]
            / baseline["average_decode_throughput_tok_s"]
        )

        print(
            f"{mode:>4}: "
            f"Peak VRAM={summary['peak_vram_gb']:.3f} GB "
            f"({memory_ratio:.2f}x BF16) | "
            f"Decode={summary['average_decode_throughput_tok_s']:.2f} tok/s "
            f"({throughput_ratio:.2f}x BF16)"
        )


def main():
    print("=== Phase 4.6 Weight Quantization Benchmark ===")
    print(f"Model:              {common.MODEL_NAME}")
    print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"BitsAndBytes:       {bnb.__version__}")
    print(f"Modes:              {QUANTIZATION_MODES}")
    print(
        f"Output tokens:      "
        f"{common.BENCHMARK_NEW_TOKENS}"
    )
    print("KV cache:           enabled")
    print(
        "Timing scope:       manual decode loop "
        "(tokenization and initial H2D excluded)"
    )
    print(
        "Note: 4-bit mode uses NF4 weight quantization, "
        "not literal integer INT4."
    )

    summaries = {}

    for mode in QUANTIZATION_MODES:
        print(
            f"\n=== Loading {mode} model ==="
        )

        (
            model,
            tokenizer,
            inputs,
            load_time_s,
            resident_vram_gb,
        ) = load_model_and_inputs(mode)

        print(
            f"Load time:          {load_time_s:.3f} s"
        )
        print(
            f"Resident VRAM:      "
            f"{resident_vram_gb:.3f} GB"
        )

        warmup(
            model=model,
            inputs=inputs,
            mode=mode,
        )

        runs = benchmark_mode(
            model=model,
            inputs=inputs,
            mode=mode,
        )

        summary = summarize_mode(
            mode=mode,
            runs=runs,
            load_time_s=load_time_s,
            resident_vram_gb=resident_vram_gb,
        )

        summaries[mode] = summary
        print_summary(summary)

        common.save_result(
            RESULT_FILE,
            summary,
        )

        unload_model(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
        )

    print_comparison(summaries)

    result_file = common.RESULT_DIR / RESULT_FILE
    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    main()
