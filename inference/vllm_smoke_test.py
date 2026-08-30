import common

common.configure_vllm_environment()

from vllm import LLM, SamplingParams


def build_engine():
    return LLM(
        model=common.MODEL_NAME,
        dtype=common.DTYPE_NAME,
        gpu_memory_utilization=common.VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len=common.VLLM_MAX_MODEL_LEN,
    )


def run_once(llm):
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=common.BENCHMARK_NEW_TOKENS,
    )

    outputs = llm.chat(
        messages=common.build_messages(),
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    return outputs[0].outputs[0].text


def main():
    print("=== Phase 4.2 vLLM Smoke Test ===")
    print(f"Model: {common.MODEL_NAME}")
    print("Loading vLLM engine...")

    llm = build_engine()

    print("Engine loaded successfully.")
    print("Generating...")

    response = run_once(llm)

    print("\n=== Model Response ===")
    print(response)


if __name__ == "__main__":
    main()
