#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-LLM-Systems-Lab
# Re-run all mandatory Phase 4 inference experiments from a clean result set.
#
# The script intentionally:
#   1. archives existing inference result files instead of deleting them;
#   2. uses the main .venv for Hugging Face / bitsandbytes experiments;
#   3. uses .venv-vllm for vLLM experiments;
#   4. records GPU telemetry during the batch-scaling benchmark;
#   5. starts/stops a real vLLM server for the online continuous-batching test.
#
# Phase 4.7 TensorRT-LLM is optional and is NOT included.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAIN_VENV="$ROOT_DIR/.venv"
VLLM_VENV="$ROOT_DIR/.venv-vllm"
RESULT_DIR="$ROOT_DIR/results/inference"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE_DIR="$RESULT_DIR/archive/$TIMESTAMP"

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

SERVER_PID=""
TELEMETRY_PID=""

cleanup() {
    if [[ -n "${TELEMETRY_PID}" ]] && kill -0 "$TELEMETRY_PID" 2>/dev/null; then
        kill "$TELEMETRY_PID" 2>/dev/null || true
        wait "$TELEMETRY_PID" 2>/dev/null || true
    fi

    if [[ -n "${SERVER_PID}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file not found: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "ERROR: required directory not found: $1" >&2
        exit 1
    fi
}

run_step() {
    local title="$1"
    shift
    echo
    echo "================================================================"
    echo "$title"
    echo "================================================================"
    "$@"
}

echo "=== DeepSeek-LLM-Systems-Lab Phase 4 Reproduction ==="
echo "Repository: $ROOT_DIR"
echo "Model:      $MODEL"
echo "Run ID:     $TIMESTAMP"

require_dir "$MAIN_VENV"
require_dir "$VLLM_VENV"

require_file "$ROOT_DIR/inference/hf_smoke_test.py"
require_file "$ROOT_DIR/inference/benchmark.py"
require_file "$ROOT_DIR/inference/benchmark_latency.py"
require_file "$ROOT_DIR/inference/benchmark_kv_cache.py"
require_file "$ROOT_DIR/inference/benchmark_quantization.py"
require_file "$ROOT_DIR/inference/vllm_smoke_test.py"
require_file "$ROOT_DIR/inference/vllm_benchmark.py"
require_file "$ROOT_DIR/inference/benchmark_continuous_batching.py"

mkdir -p "$RESULT_DIR"
mkdir -p "$ARCHIVE_DIR"

# Preserve previous raw evidence. Only top-level files are moved; prior archives
# remain untouched.
find "$RESULT_DIR" -maxdepth 1 -type f -exec mv -t "$ARCHIVE_DIR" {} + 2>/dev/null || true

echo "Previous inference result files archived to:"
echo "  $ARCHIVE_DIR"

# ---------------------------------------------------------------------------
# Phase 4.1 / 4.4 / 4.6: Hugging Face + bitsandbytes (.venv)
# ---------------------------------------------------------------------------

# shellcheck disable=SC1091
source "$MAIN_VENV/bin/activate"

run_step "Phase 4.1a - Hugging Face smoke test" \
    python inference/hf_smoke_test.py

run_step "Phase 4.1b - Hugging Face model.generate baseline" \
    python inference/benchmark.py

run_step "Phase 4.1c - Hugging Face TTFT / TPOT manual-decode baseline" \
    python inference/benchmark_latency.py

run_step "Phase 4.4 - KV Cache ON/OFF" \
    python inference/benchmark_kv_cache.py

run_step "Phase 4.6 - BF16 / INT8 / NF4 quantization" \
    python inference/benchmark_quantization.py

deactivate

# ---------------------------------------------------------------------------
# Phase 4.2 / 4.3 / 4.5a / 4.5b: vLLM (.venv-vllm)
# ---------------------------------------------------------------------------

# shellcheck disable=SC1091
source "$VLLM_VENV/bin/activate"

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0

run_step "Phase 4.2 - vLLM smoke test" \
    python inference/vllm_smoke_test.py

run_step "Phase 4.3 - vLLM TTFT / TPOT / throughput baseline" \
    python inference/vllm_benchmark.py

echo
echo "================================================================"
echo "Phase 4.5a - Concurrent batching scaling + GPU telemetry"
echo "================================================================"

nvidia-smi \
    --query-gpu=timestamp,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv \
    -lms 500 \
    > "$RESULT_DIR/gpu_telemetry.csv" &
TELEMETRY_PID=$!

python inference/benchmark_continuous_batching.py

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
TELEMETRY_PID=""

echo
echo "================================================================"
echo "Phase 4.5b - Online continuous batching"
echo "================================================================"

SERVER_LOG="$RESULT_DIR/vllm_server_phase4.log"

vllm serve "$MODEL" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 4096 \
    --no-enable-prefix-caching \
    --generation-config vllm \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "vLLM server PID: $SERVER_PID"
echo "Server log:      $SERVER_LOG"
echo "Waiting for http://127.0.0.1:8000/health ..."

SERVER_READY=0
for _ in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: vLLM server exited before becoming ready." >&2
        echo "Last server log lines:" >&2
        tail -n 80 "$SERVER_LOG" >&2 || true
        exit 1
    fi

    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        SERVER_READY=1
        break
    fi

    sleep 1
done

if [[ "$SERVER_READY" -ne 1 ]]; then
    echo "ERROR: vLLM server was not ready after 120 seconds." >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
fi

echo "vLLM server ready."

vllm bench serve \
    --backend openai \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts 128 \
    --random-input-len 64 \
    --random-output-len 128 \
    --random-range-ratio '{"input":0.5,"output":0.75}' \
    --request-rate 32 \
    --burstiness 1.0 \
    --max-concurrency 64 \
    --num-warmups 8 \
    --temperature 0 \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename vllm_continuous_batching_online.json \
    --plot-timeline

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

deactivate

echo
echo "================================================================"
echo "Phase 4 reproduction completed"
echo "================================================================"
echo "Results:"
echo "  $RESULT_DIR"
echo
echo "Previous result archive:"
echo "  $ARCHIVE_DIR"
echo
echo "Reminder:"
echo "  --plot-timeline may warn if vllm[bench] plotting dependencies are absent."
echo "  The benchmark JSON is still valid when only timeline plotting fails."
