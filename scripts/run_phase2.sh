#!/usr/bin/env bash

# DeepSeek-LLM-Systems-Lab — Phase 2 Historical Command Collection
# 由 Phase 2 命令收集摘要转换而来。
# 注意：以下命令只表示旧聊天中确认出现过的 Historical exact commands。
# “未确认执行”的命令仍按原文保留；不要把本文件直接等同于当前仓库的最终可复现脚本。

# === Phase 2: project Docker commands ===

docker build -t deepseek-systems-lab .

docker run --gpus all deepseek-systems-lab

docker compose up


# === Phase 2: Docker / GPU environment validation ===

docker --version

docker run hello-world

docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
