#!/usr/bin/env bash
# Radeon 760M iGPU comparison: build llama.cpp with the Vulkan backend (host, no root: glslc + SPIRV-Headers
# extracted under tools/) and run llama-bench on the same GGUFs. Run on the HOST.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
G="$ROOT/tools/glslc-root"; S="$ROOT/tools/spirv-root"
export PATH="$G/usr/bin:$PATH" LD_LIBRARY_PATH="$G/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" CPATH="$S/usr/include"
if [ ! -x build/llama.cpp-vulkan/bin/llama-bench ]; then
  cmake -S third_party/llama.cpp -B build/llama.cpp-vulkan -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_VULKAN=ON \
        -DVulkan_GLSLC_EXECUTABLE="$G/usr/bin/glslc" -DCMAKE_PREFIX_PATH="$S/usr" 2>&1 | grep -E "Error|error" | head -5
  cmake --build build/llama.cpp-vulkan -j"$(nproc)" --target llama-bench 2>&1 | grep -E "error|Error|Linking.*llama-bench" | head -5
fi
for M in models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf models/gguf/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf; do
  echo; echo "### $(basename "$M") on the iGPU (Vulkan, all layers offloaded)"
  build/llama.cpp-vulkan/bin/llama-bench -m "$M" -ngl 99 -p 128,512 -n 32 -r 3 2>&1 | grep -E "^\| (qwen|ggml_vulkan)|Vulkan0" | cut -c1-200
done
