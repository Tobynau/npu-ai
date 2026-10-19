#!/usr/bin/env bash
# LLM test 1: llama.cpp prefill offload to the NPU via ggml-xdna1. Run inside the container.
# Usage: docker/run.sh scripts/03-llm-llamacpp-xdna1.sh [model.gguf]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
M="${1:-models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
B=build/llama.cpp/bin; P=build/ggml-xdna1/lib/libggml-xdna1.so
T="${THREADS:-8}"
echo "== model: $M  threads: $T"
echo; echo "== raw GEMM kernel test on the NPU (test-gemm 1024x1024x64, 20 iters)"
build/ggml-xdna1/bin/test-gemm - 1024 1024 64 20 4 2>&1 | tail -5
echo; echo "== llama-bench CPU only (pp512, tg32)"
$B/llama-bench -m "$M" -p 512 -n 32 -r 2 -t "$T" 2>&1 | grep -E "^\||model" | tail -4
echo; echo "== llama-bench with XDNA1 plugin (pp512, tg32)"
GGML_BACKEND_PATH="$PWD/$P" $B/llama-bench -m "$M" -p 512 -n 32 -r 2 -t "$T" 2>&1 | grep -E "^\||model|xdna" | tail -6
echo; echo "== generation with the plugin active (look for '[xdna1] ready' on stderr)"
GEN=$B/llama-completion; [ -x "$GEN" ] || GEN="$B/llama-cli -st"
GGML_BACKEND_PATH="$PWD/$P" XDNA1_PROFILE=1 timeout 600 $GEN -m "$M" --device XDNA1 -ngl 0 -t "$T" -c 2048 -n 48 --temp 0 \
  -p "Explain in two sentences what an NPU is and why it is power efficient." 2>&1 < /dev/null \
  | grep -vE "^llama_|^print_info|^load|^common_|^main:|^system_info|^sampler|^generate|^graph_" | tail -25
