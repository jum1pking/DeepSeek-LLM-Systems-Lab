#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-LLM-Systems-Lab — Phase 3 experiment runner
#
# Scope:
#   - Current-code reproducible DeepSpeed / ZeRO runs (Phase 3.6)
#   - ZeRO-3 checkpoint smoke tests
#   - Profiling commands validated during Phase 3.7
#   - Profiling-driven GC on/off benchmark comparison
#   - SDPA backend comparison (Phase 3.8)
#   - Optional batch-size scaling extension
#
# Historical provenance:
#   The original ZeRO-0/1/2/3 benchmark results were produced when
#   training/train_deepspeed.py still accepted --zero-stage 0/1/2/3.
#   The current code has since been refactored to --config <name>, so this
#   runner uses the current CLI instead of replaying the obsolete interface.
#
# Intentionally excluded from normal entries:
#   - Environment setup / PATH / WSL / NVIDIA permission troubleshooting
#   - The failed gemm|gemv|cutlass Nsight Compute name-filter experiment
#   - The superseded flash.* diagnostic that only matched flash_fwd_kernel
#   - A one-command "run everything" mode; each experiment is explicit

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILER_DIR="results/profiler"
NCU_DIR="$PROFILER_DIR/ncu_single_lora"
NSYS_DIR="$PROFILER_DIR/nsys_single_lora"

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/run_phase3_experiments.sh <command> [args]

Phase 3.6 — DeepSpeed / ZeRO
  zero0                       Run ZeRO-0 20-step benchmark
  zero1                       Run ZeRO-1 20-step benchmark
  zero2                       Run ZeRO-2 20-step benchmark
  zero3                       Run ZeRO-3 20-step benchmark
  zero2-cpu-offload           Run ZeRO-2 CPU optimizer offload benchmark
  zero3-cpu-offload           Run ZeRO-3 CPU offload experiment
  zero3-init-smoke            Run ZeRO-3 init + save checkpoint smoke test
  zero3-resume-smoke          Run ZeRO-3 checkpoint resume smoke test
  zero3-checkpoint-files      List saved ZeRO-3 checkpoint files
  zero3-validation            Print DeepSpeed validation CSV

Phase 3.7 — Profiling and profiling-driven optimization
  gc-control                  Re-run the original GC-on single-GPU LoRA control
  no-gc                       Run the GC-off single-GPU LoRA experiment
  nsys-api                    Capture CUDA API + NVTX with Nsight Systems
  nsys-api-stats              Print CUDA API summary from trace_2026.nsys-rep
  ncu-baseline                Profile first 20 kernels in lora_training
  ncu-flash                   Profile 10 flash_fwd_kernel launches (legacy sample)
  ncu-step8-kernel2           Profile 10 Kernel2 launches in profile_step_8
  ncu-step8-flash             Profile 10 flash_fwd_kernel launches in profile_step_8
  ncu-step8-kernel2-duration  Census all Kernel2 durations in profile_step_8
  ncu-step8-flash-duration    Census all flash_fwd_kernel durations in profile_step_8
  ncu-step8-full              Census the complete profiled training step
  ncu-export <name>           Export <name>.ncu-rep to <name>_details.txt
  ncu-export-csv <name>       Export raw <name>.ncu-rep data to <name>.csv

Phase 3.8 — SDPA backend comparison
  sdpa-flash                  Run no-GC training with forced Flash SDPA
  sdpa-math                   Run no-GC training with forced Math SDPA

Optional extension — Batch-size scaling
  batch2                      Run no-GC batch-size 2 scaling test
  batch4                      Run no-GC batch-size 4 scaling test
  batch8                      Run no-GC batch-size 8 scaling test

Notes:
  - Activate the project environment before running this script.
  - zero3-cpu-offload is retained as an experimental entry; it did not produce
    a clean formal 20-step baseline because of host memory / transfer pressure.
  - Profiling runtimes are instrumentation-heavy and must not be compared with
    clean benchmark runtimes.
  - ncu-step8-full expects training/nsys_single.py to contain the validated
    cudaProfilerStart()/cudaProfilerStop() step-range control.
  - NCU reports are not overwritten by default. Set OVERWRITE=1 to permit -f.
USAGE
}

run_deepspeed() {
    local config_name="$1"
    "$PYTHON_BIN" -m torch.distributed.run \
        --nproc_per_node=1 \
        training/train_deepspeed.py \
        --config "$config_name"
}

NCU_OVERWRITE_ARGS=()

prepare_ncu_output() {
    local output_base="$1"
    NCU_OVERWRITE_ARGS=()

    if [[ -f "${output_base}.ncu-rep" ]]; then
        if [[ "${OVERWRITE:-0}" == "1" ]]; then
            NCU_OVERWRITE_ARGS=(-f)
        else
            echo "ERROR: report already exists: ${output_base}.ncu-rep" >&2
            echo "Set OVERWRITE=1 to replace it." >&2
            exit 4
        fi
    fi
}

command="${1:-}"

case "$command" in
    zero0)
        run_deepspeed zero0
        ;;

    zero1)
        run_deepspeed zero1
        ;;

    zero2)
        run_deepspeed zero2
        ;;

    zero3)
        run_deepspeed zero3
        ;;

    zero2-cpu-offload)
        run_deepspeed zero2_cpu_offload
        ;;

    zero3-cpu-offload)
        echo "WARNING: the original local ZeRO-3 CPU offload run showed heavy host-memory / transfer pressure."
        echo "This entry is experimental and is not a clean formal baseline."
        run_deepspeed zero3_cpu_offload
        ;;

    zero3-init-smoke)
        "$PYTHON_BIN" -m torch.distributed.run \
            --nproc_per_node=1 \
            training/zero3_init_smoke.py
        ;;

    zero3-resume-smoke)
        "$PYTHON_BIN" -m torch.distributed.run \
            --nproc_per_node=1 \
            training/zero3_resume_smoke.py
        ;;

    zero3-checkpoint-files)
        find results/training/deepspeed_zero3_checkpoint -maxdepth 2 -type f
        ;;

    zero3-validation)
        cat results/training/deepspeed_validation.csv
        ;;

    gc-control)
        "$PYTHON_BIN" training/train_single.py
        ;;

    no-gc)
        "$PYTHON_BIN" training/no_gc_train_single.py
        ;;

    sdpa-flash)
        "$PYTHON_BIN" training/train_sdpa_backend.py --backend flash
        ;;

    sdpa-math)
        "$PYTHON_BIN" training/train_sdpa_backend.py --backend math
        ;;

    batch2)
        "$PYTHON_BIN" training/train_batch_scaling.py --batch-size 2
        ;;

    batch4)
        "$PYTHON_BIN" training/train_batch_scaling.py --batch-size 4
        ;;

    batch8)
        "$PYTHON_BIN" training/train_batch_scaling.py --batch-size 8
        ;;

    nsys-api)
        mkdir -p "$NSYS_DIR"
        nsys profile \
            --trace=cuda,nvtx \
            --capture-range=nvtx \
            --nvtx-capture=lora_training \
            --capture-range-end=stop \
            --sample=none \
            --cpuctxsw=none \
            -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
            -o "$NSYS_DIR/trace_2026" \
            "$PYTHON_BIN" training/nsys_single.py
        ;;

    nsys-api-stats)
        nsys stats \
            --report cuda_api_sum \
            "$NSYS_DIR/trace_2026.nsys-rep"
        ;;

    ncu-baseline)
        mkdir -p "$NCU_DIR"
        output_base="$NCU_DIR/baseline"
        prepare_ncu_output "$output_base"
        ncu \
            "${NCU_OVERWRITE_ARGS[@]}" \
            --nvtx \
            --nvtx-include "lora_training/" \
            --launch-count 20 \
            --set basic \
            -o "$output_base" \
            "$PYTHON_BIN" training/nsys_single.py
        ;;

    ncu-flash)
        mkdir -p "$NCU_DIR"
        output_base="$NCU_DIR/flash_fwd_sample"
        prepare_ncu_output "$output_base"
        ncu \
            "${NCU_OVERWRITE_ARGS[@]}" \
            --nvtx \
            --nvtx-include "lora_training/" \
            --kernel-name-base function \
            --kernel-name flash_fwd_kernel \
            --launch-count 10 \
            --set basic \
            -o "$output_base" \
            "$PYTHON_BIN" training/nsys_single.py
        ;;

    ncu-step8-kernel2)
        mkdir -p "$NCU_DIR"
        output_base="$NCU_DIR/step8_kernel2"
        prepare_ncu_output "$output_base"
        ncu \
            "${NCU_OVERWRITE_ARGS[@]}" \
            --nvtx \
            --nvtx-include "profile_step_8/" \
            --kernel-name-base function \
            --kernel-name Kernel2 \
            --launch-count 10 \
            --set basic \
            -o "$output_base" \
            "$PYTHON_BIN" training/nsys_single.py
        ;;

    ncu-step8-flash)
        mkdir -p "$NCU_DIR"
        output_base="$NCU_DIR/step8_flash"
        prepare_ncu_output "$output_base"
        ncu \
            "${NCU_OVERWRITE_ARGS[@]}" \
            --nvtx \
            --nvtx-include "profile_step_8/" \
            --kernel-name-base function \
            --kernel-name flash_fwd_kernel \
            --launch-count 10 \
            --set basic \
            -o "$output_base" \
            "$PYTHON_BIN" training/nsys_single.py
        ;;

    ncu-step8-kernel2-duration)
        mkdir -p "$NCU_DIR"
        output_base="$NCU_DIR/step8_kernel2_duration"
        prepare_ncu_output "$output_base"
        ncu \
            "${NCU_OVERWRITE_ARGS[@]}" \
            --nvtx \
            --nvtx-include "profile_step_8/" \
            --kernel-name-base function \
            --kernel-name Kernel2 \
            --metrics gpu__time_duration.sum \
            -o "$output_base" \
            "$PYTHON_BIN" training/nsys_single.py
        ;;

    ncu-step8-flash-duration)
        mkdir -p "$NCU_DIR"
        output_base="$NCU_DIR/step8_flash_duration"
        prepare_ncu_output "$output_base"
        ncu \
            "${NCU_OVERWRITE_ARGS[@]}" \
            --nvtx \
            --nvtx-include "profile_step_8/" \
            --kernel-name-base function \
            --kernel-name flash_fwd_kernel \
            --metrics gpu__time_duration.sum \
            -o "$output_base" \
            "$PYTHON_BIN" training/nsys_single.py
        ;;

    ncu-step8-full)
        mkdir -p "$NCU_DIR"
        output_base="$NCU_DIR/step8_all_kernels_full"
        prepare_ncu_output "$output_base"
        ncu \
            "${NCU_OVERWRITE_ARGS[@]}" \
            --profile-from-start off \
            --kernel-name-base function \
            --metrics gpu__time_duration.sum \
            -o "$output_base" \
            "$PYTHON_BIN" training/nsys_single.py
        ;;

    ncu-export)
        report_name="${2:-}"
        if [[ -z "$report_name" ]]; then
            echo "ERROR: missing report name." >&2
            echo "Example: $0 ncu-export step8_kernel2" >&2
            exit 2
        fi

        report_file="$NCU_DIR/${report_name}.ncu-rep"
        output_file="$NCU_DIR/${report_name}_details.txt"

        if [[ ! -f "$report_file" ]]; then
            echo "ERROR: report does not exist: $report_file" >&2
            exit 3
        fi

        ncu \
            --import "$report_file" \
            --page details \
            > "$output_file"

        echo "Exported: $output_file"
        ;;

    ncu-export-csv)
        report_name="${2:-}"
        if [[ -z "$report_name" ]]; then
            echo "ERROR: missing report name." >&2
            echo "Example: $0 ncu-export-csv step8_all_kernels_full" >&2
            exit 2
        fi

        report_file="$NCU_DIR/${report_name}.ncu-rep"
        output_file="$NCU_DIR/${report_name}.csv"

        if [[ ! -f "$report_file" ]]; then
            echo "ERROR: report does not exist: $report_file" >&2
            exit 3
        fi

        ncu \
            --import "$report_file" \
            --page raw \
            --csv \
            > "$output_file"

        echo "Exported: $output_file"
        ;;

    -h|--help|help|"")
        usage
        ;;

    *)
        echo "ERROR: unknown command: $command" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac
