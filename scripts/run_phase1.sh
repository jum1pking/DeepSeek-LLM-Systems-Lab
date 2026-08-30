#!/usr/bin/env bash

# DeepSeek-LLM-Systems-Lab — Phase 1 Historical Command Collection
# 由 Phase 1 命令收集摘要转换而来。
# 注意：以下命令只表示旧聊天中确认出现过的 Historical exact commands。
# “未确认执行”的命令仍按原文保留；不要把本文件直接等同于当前仓库的最终可复现脚本。

# === Phase 1: experiment / smoke commands ===

python training/inference_smoke_test.py

python inference/hf_smoke_test.py

python training/lora_smoke_test.py


# === Phase 1: dependency / project preparation commands ===

uv pip install transformers accelerate safetensors

uv pip install peft datasets trl

mkdir -p training


# === Phase 1: development/edit command ===
# 这是旧聊天中出现过的编辑命令，不属于实验 benchmark。

nano training/lora_smoke_test.py
