#!/usr/bin/env bash
# LLM benchmark matrix on the same GGUFs: CPU (6/12 threads), XDNA1 NPU plugin, and the Radeon 760M iGPU (Vulkan).
# Run on the HOST: scripts/08-llm-bench-matrix.sh   (NPU rows go through docker/run.sh; the rest runs natively)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
B=build/llama.cpp/bin; P="$ROOT/build/ggml-xdna1/lib/libggml-xdna1.so"
MODELS=(models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf models/gguf/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf)
ARGS=(-p 128,512 -n 32 -r 3)
GLSLC_ROOT="$ROOT/tools/glslc-root"
if [ ! -x build/llama.cpp-vulkan/bin/llama-bench ] && [ -x "$GLSLC_ROOT/usr/bin/glslc" ]; then
  echo "== building llama.cpp with the Vulkan backend for the iGPU (host)"
  export PATH="$GLSLC_ROOT/usr/bin:$PATH" LD_LIBRARY_PATH="$GLSLC_ROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  cmake -S third_party/llama.cpp -B build/llama.cpp-vulkan -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_VULKAN=ON \
        -DVulkan_GLSLC_EXECUTABLE="$GLSLC_ROOT/usr/bin/glslc" >/dev/null 2>&1 && \
  cmake --build build/llama.cpp-vulkan -j"$(nproc)" --target llama-bench 2>&1 | tail -2
fi
row() { grep -E "^\| (qwen|llama|gemma)" | sed "s/^/| $1 /"; }
for M in "${MODELS[@]}"; do
  echo; echo "### $(basename "$M")"
  echo "| config | model | size | params | backend | threads/ngl | test | t/s |"
  $B/llama-bench -m "$M" -t 6 "${ARGS[@]}" 2>/dev/null | row "CPU-6t"
  $B/llama-bench -m "$M" -t 12 "${ARGS[@]}" 2>/dev/null | row "CPU-12t"
  docker/run.sh env GGML_BACKEND_PATH="$P" "$ROOT/$B/llama-bench" -m "$ROOT/$M" -t 6 "${ARGS[@]}" 2>/dev/null | row "NPU+CPU-6t"
  if [ -x build/llama.cpp-vulkan/bin/llama-bench ]; then
    build/llama.cpp-vulkan/bin/llama-bench -m "$M" -ngl 99 "${ARGS[@]}" 2>/dev/null | row "iGPU-Vulkan"
  fi
done
echo; echo "matrix done"

echo; echo "### sanity generation with the NPU plugin (Qwen2.5-1.5B, greedy, 48 tokens)"
GEN=$B/llama-completion; [ -x "$GEN" ] || GEN="$B/llama-cli -st"
docker/run.sh env GGML_BACKEND_PATH="$P" XDNA1_PROFILE=1 timeout 600 $ROOT/$GEN -m "$ROOT/${MODELS[0]}" --device XDNA1 -ngl 0 -t 6 -c 2048 -n 48 --temp 0 \
  -p "Explain in two sentences what an NPU is and why it is power efficient." 2>&1 < /dev/null \
  | grep -E "xdna1|^[A-Z]|tokens per second|eval time|prompt eval" | tail -20
