import time
from datetime import datetime

import torch

import benchmark
import common


RESULT_FILE = "hf_kv_cache_comparison.csv"


def run_once(model, inputs, use_cache):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        past_key_values = (
            outputs.past_key_values
            if use_cache
            else None
        )

        # For the no-cache path, each decode step must receive the entire
        # growing sequence because no past key/value state is retained.
        full_input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

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

            if use_cache:
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            if use_cache:
                past_key_values = outputs.past_key_values
            else:
                full_input_ids = torch.cat(
                    [full_input_ids, next_token],
                    dim=1,
                )

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
        "end_to_end_throughput_tok_s": end_to_end_throughput_tok_s,
        "peak_vram_gb": peak_vram_gb,
    }


def warmup(model, inputs, use_cache):
    mode = "ON" if use_cache else "OFF"
    print(f"Warming up KV cache {mode}...")

    for _ in range(common.HF_WARMUP_RUNS):
        run_once(
            model=model,
            inputs=inputs,
            use_cache=use_cache,
        )


def benchmark_mode(model, inputs, use_cache):
    mode = "ON" if use_cache else "OFF"
    runs = []

    print(f"\nBenchmarking KV cache {mode}...")

    for i in range(common.BENCHMARK_RUNS):
        result = run_once(
            model=model,
            inputs=inputs,
            use_cache=use_cache,
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

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": common.MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": common.DTYPE_NAME,
        "timing_scope": "manual_decode_loop",
        "kv_cache": use_cache,
        "input_tokens": runs[0]["input_tokens"],
        "output_tokens": runs[0]["output_tokens"],
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
            max(run["peak_vram_gb"] for run in runs),
            2,
        ),
    }

    return summary


def print_summary(summary):
    mode = "ON" if summary["kv_cache"] else "OFF"

    print(f"\n=== KV Cache {mode} Average ===")
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
    print(
        "End-to-end throughput: "
        f"{summary['average_end_to_end_throughput_tok_s']:.2f} tokens/s"
    )
    print(
        f"Peak VRAM:          "
        f"{summary['peak_vram_gb']:.2f} GB"
    )


def print_comparison(cache_on, cache_off):
    latency_speedup = (
        cache_off["average_end_to_end_latency_s"]
        / cache_on["average_end_to_end_latency_s"]
    )

    decode_speedup = (
        cache_on["average_decode_throughput_tok_s"]
        / cache_off["average_decode_throughput_tok_s"]
    )

    print("\n=== KV Cache ON vs OFF ===")
    print(
        "E2E latency speedup: "
        f"{latency_speedup:.2f}x"
    )
    print(
        "Decode throughput speedup: "
        f"{decode_speedup:.2f}x"
    )
    print(
        "Peak VRAM delta (ON - OFF): "
        f"{cache_on['peak_vram_gb'] - cache_off['peak_vram_gb']:+.2f} GB"
    )


def main():
    model, tokenizer, inputs = benchmark.load_model_and_inputs()

    print("=== Phase 4.4 KV Cache Comparison ===")
    print(f"Model:              {common.MODEL_NAME}")
    print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"Precision:          {common.DTYPE}")
    print(
        f"Input tokens:       "
        f"{inputs['input_ids'].shape[1]}"
    )
    print(
        f"Output tokens:      "
        f"{common.BENCHMARK_NEW_TOKENS}"
    )
    print(
        "Controlled variable: use_cache only"
    )
    print(
        "Timing scope:       manual decode loop "
        "(tokenization and initial H2D excluded)"
    )

    # Warm each execution path before collecting measured runs.
    warmup(
        model=model,
        inputs=inputs,
        use_cache=True,
    )
    warmup(
        model=model,
        inputs=inputs,
        use_cache=False,
    )

    cache_on = benchmark_mode(
        model=model,
        inputs=inputs,
        use_cache=True,
    )
    cache_off = benchmark_mode(
        model=model,
        inputs=inputs,
        use_cache=False,
    )

    print_summary(cache_on)
    print_summary(cache_off)
    print_comparison(
        cache_on=cache_on,
        cache_off=cache_off,
    )

    result_file = common.save_result(
        RESULT_FILE,
        cache_on,
    )
    common.save_result(
        RESULT_FILE,
        cache_off,
    )

    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    main()
