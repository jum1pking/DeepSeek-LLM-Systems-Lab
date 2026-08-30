import time
from datetime import datetime

import common

common.configure_vllm_environment()

import torch
from vllm import LLM, SamplingParams


def build_engine():
    return LLM(
        model=common.MODEL_NAME,
        dtype=common.DTYPE_NAME,
        gpu_memory_utilization=common.VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len=common.VLLM_MAX_MODEL_LEN,
        enable_prefix_caching=False,
        disable_log_stats=False,
    )


def build_request():
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=common.BENCHMARK_NEW_TOKENS,
    )

    return common.build_messages(), sampling_params


def run_once(llm, messages, sampling_params):
    wall_start = time.perf_counter()

    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    wall_end = time.perf_counter()

    request_output = outputs[0]
    completion = request_output.outputs[0]
    metrics = request_output.metrics

    if metrics is None:
        raise RuntimeError(
            "vLLM request metrics are unavailable. "
            "Keep disable_log_stats=False in build_engine()."
        )

    output_tokens = len(completion.token_ids)
    input_tokens = len(request_output.prompt_token_ids or [])

    scheduled_ts = metrics.scheduled_ts
    first_token_ts = metrics.first_token_ts
    last_token_ts = metrics.last_token_ts
    queued_ts = metrics.queued_ts

    if scheduled_ts <= 0 or first_token_ts <= 0 or last_token_ts <= 0:
        raise RuntimeError(
            "vLLM returned incomplete request timestamps; "
            "cannot compute TTFT/TPOT reliably."
        )

    ttft_s = first_token_ts - scheduled_ts
    decode_time_s = last_token_ts - first_token_ts
    inference_time_s = last_token_ts - scheduled_ts
    queue_time_s = (
        scheduled_ts - queued_ts
        if queued_ts > 0
        else 0.0
    )

    decode_tokens = max(output_tokens - 1, 0)

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
        output_tokens / inference_time_s
        if inference_time_s > 0
        else 0.0
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": ttft_s * 1000,
        "average_tpot_ms": average_tpot_s * 1000,
        "decode_throughput_tok_s": decode_throughput_tok_s,
        "end_to_end_latency_s": inference_time_s,
        "end_to_end_throughput_tok_s": end_to_end_throughput_tok_s,
        "queue_time_ms": queue_time_s * 1000,
        "wall_latency_s": wall_end - wall_start,
        "num_cached_tokens": request_output.num_cached_tokens or 0,
    }


def main():
    print("=== Phase 4.3 vLLM Latency Benchmark ===")
    print(f"Model:              {common.MODEL_NAME}")
    print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"Precision:          {common.DTYPE}")
    print(f"Max output tokens:  {common.BENCHMARK_NEW_TOKENS}")
    print("Prefix caching:     disabled")
    print("Timing scope:       vLLM scheduled request -> last token")
    print("Building engine (startup is excluded from benchmark timing)...")

    llm = build_engine()
    messages, sampling_params = build_request()

    print("\nWarming up...")
    for _ in range(common.VLLM_WARMUP_RUNS):
        run_once(
            llm=llm,
            messages=messages,
            sampling_params=sampling_params,
        )

    runs = []

    print("Benchmarking...")
    for i in range(common.BENCHMARK_RUNS):
        result = run_once(
            llm=llm,
            messages=messages,
            sampling_params=sampling_params,
        )
        runs.append(result)

        print(
            f"Run {i + 1}: "
            f"TTFT={result['ttft_ms']:.2f} ms | "
            f"TPOT={result['average_tpot_ms']:.2f} ms | "
            f"Decode={result['decode_throughput_tok_s']:.2f} tok/s | "
            f"E2E={result['end_to_end_latency_s']:.3f} s | "
            f"Queue={result['queue_time_ms']:.2f} ms | "
            f"Wall={result['wall_latency_s']:.3f} s"
        )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": common.MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": common.DTYPE_NAME,
        "backend": "vllm",
        "runner": "v1",
        "flashinfer_sampler": False,
        "prefix_caching": False,
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
        "average_queue_time_ms": round(
            common.average([run["queue_time_ms"] for run in runs]),
            3,
        ),
        "average_wall_latency_s": round(
            common.average([run["wall_latency_s"] for run in runs]),
            3,
        ),
        "num_cached_tokens": max(
            run["num_cached_tokens"]
            for run in runs
        ),
    }

    print("\n=== Average Results ===")
    print(f"Input tokens:        {summary['input_tokens']}")
    print(f"Output tokens:       {summary['output_tokens']}")
    print(f"TTFT:                {summary['average_ttft_ms']:.2f} ms")
    print(f"TPOT:                {summary['average_tpot_ms']:.2f} ms/token")
    print(
        "Decode throughput:   "
        f"{summary['average_decode_throughput_tok_s']:.2f} tokens/s"
    )
    print(
        "End-to-end latency:  "
        f"{summary['average_end_to_end_latency_s']:.3f} s"
    )
    print(
        "End-to-end throughput: "
        f"{summary['average_end_to_end_throughput_tok_s']:.2f} tokens/s"
    )
    print(f"Queue time:          {summary['average_queue_time_ms']:.2f} ms")
    print(f"Wall latency:        {summary['average_wall_latency_s']:.3f} s")
    print(f"Cached prompt tokens:{summary['num_cached_tokens']}")

    result_file = common.save_result(
        "vllm_latency_baseline.csv",
        summary,
    )
    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    main()
