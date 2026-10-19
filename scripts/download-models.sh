#!/usr/bin/env bash
# Download the small models used by the NPU tests into models/ (not committed).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DL="hf download"
# 1. Qwen2.5-0.5B-Instruct safetensors: used by hawkpoint-npu-llm / xdna1-npu-lab IRON paths
$DL Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct --include "*.json" "*.safetensors" "merges.txt" "vocab.json"
# 2. Qwen3-0.6B safetensors: used by npu-lab (NPU generates every token)
$DL Qwen/Qwen3-0.6B --local-dir models/qwen3s --include "*.json" "*.safetensors" "merges.txt" "vocab.json" "*.jinja"
# 3. GGUFs for llama.cpp + ggml-xdna1 (prefill offload); 1.5B is big enough for the NPU gate to accept shapes
$DL bartowski/Qwen2.5-1.5B-Instruct-GGUF Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --local-dir models/gguf
$DL bartowski/Qwen2.5-0.5B-Instruct-GGUF Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --local-dir models/gguf
echo "downloads done"; du -sh models/*
