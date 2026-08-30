import torch

import benchmark
import common


def run_once(model, inputs):
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=common.BENCHMARK_NEW_TOKENS,
            do_sample=False,
        )

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    return outputs, peak_vram_gb


def main():
    print("Loading model...")
    model, tokenizer, inputs = benchmark.load_model_and_inputs()
    print("Model loaded successfully")
    print("Tokenizer loaded successfully.")

    print("Generating...")

    outputs, peak_vram_gb = run_once(
        model=model,
        inputs=inputs,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\n=== Model Response ===")
    print(response)

    print("\n=== GPU Memory ===")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")


if __name__ == "__main__":
    main()
