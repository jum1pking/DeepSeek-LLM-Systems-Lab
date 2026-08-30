import time
from datetime import datetime

import torch

import benchmark
import common


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
        if decode_time_s > 0
        else 0.0
    )
    end_to_end_throughput_tok_s = (
        output_tokens / end_to_end_latency_s
        if end_to_end_latency_s > 0
        else 0.0
    )
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "input_tokens": input_ids.shape[1],
        "output_tokens": output_tokens,
        "ttft_ms": ttft_s * 1000,
        "average_tpot_ms": average_tpot_s * 1000,
        "decode_throughput_tok_s": decode_throughput_tok_s,
        "end_to_end_latency_s": end_to_end_latency_s,
        "end_to_end_throughput_tok_s": end_to_end_throughput_tok_s,
        "peak_vram_gb": peak_vram_gb,
    }


def main():
    model, tokenizer, inputs = benchmark.load_model_and_inputs()

    print("=== Phase 4.1 Hugging Face Latency Benchmark ===")
    print(f"Model:              {common.MODEL_NAME}")
    print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"Precision:          {common.DTYPE}")
    print(f"Input tokens:       {inputs['input_ids'].shape[1]}")
    print(f"Max output tokens:  {common.BENCHMARK_NEW_TOKENS}")
    print("KV cache:           enabled")
    print(
        "Timing scope:       manual decode loop "
        "(tokenization and initial H2D excluded)"
    )

    print("\nWarming up...")
    for _ in range(common.HF_WARMUP_RUNS):
        run_once(model, inputs)

    runs = []

    print("Benchmarking...")
    for i in range(common.BENCHMARK_RUNS):
        result = run_once(model, inputs)
        runs.append(result)

        print(
            f"Run {i + 1}: "
            f"TTFT={result['ttft_ms']:.2f} ms | "
            f"TPOT={result['average_tpot_ms']:.2f} ms | "
            f"Decode={result['decode_throughput_tok_s']:.2f} tok/s | "
            f"E2E={result['end_to_end_latency_s']:.3f} s | "
            f"Peak VRAM={result['peak_vram_gb']:.2f} GB"
        )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": common.MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": common.DTYPE_NAME,
        "timing_scope": "manual_decode_loop",
        "kv_cache": True,
        "input_tokens": runs[0]["input_tokens"],
        "output_tokens": runs[0]["output_tokens"],
        "average_ttft_ms": round(
            common.average([run["ttft_ms"] for run in runs]),
            3,
        ),
        "average_tpot_ms": round(
            common.average([run["average_tpot_ms"] for run in runs]),
            3,
        ),
        "average_decode_throughput_tok_s": round(
            common.average(
                [run["decode_throughput_tok_s"] for run in runs]
            ),
            2,
        ),
        "average_end_to_end_latency_s": round(
            common.average(
                [run["end_to_end_latency_s"] for run in runs]
            ),
            3,
        ),
        "average_end_to_end_throughput_tok_s": round(
            common.average(
                [run["end_to_end_throughput_tok_s"] for run in runs]
            ),
            2,
        ),
        "peak_vram_gb": round(
            max(run["peak_vram_gb"] for run in runs),
            2,
        ),
    }

    print("\n=== Average Results ===")
    print(f"TTFT:               {summary['average_ttft_ms']:.2f} ms")
    print(f"TPOT:               {summary['average_tpot_ms']:.2f} ms/token")
    print(
        "Decode throughput:  "
        f"{summary['average_decode_throughput_tok_s']:.2f} tokens/s"
    )
    print(
        "End-to-end latency: "
        f"{summary['average_end_to_end_latency_s']:.3f} s"
    )
    print(
        "End-to-end throughput: "
        f"{summary['average_end_to_end_throughput_tok_s']:.2f} tokens/s"
    )
    print(f"Peak VRAM:          {summary['peak_vram_gb']:.2f} GB")

    result_file = common.save_result(
        "hf_latency_baseline.csv",
        summary,
    )
    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    main()
