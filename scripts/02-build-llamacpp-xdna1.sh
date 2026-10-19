#!/usr/bin/env bash
# Build llama.cpp (CPU) and the FireBurn/ggml-xdna1 backend plugin. Works on the host or in the container.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
J=$(nproc); GEN=""; command -v ninja >/dev/null && GEN="-G Ninja"
echo "== llama.cpp @ $(git -C third_party/llama.cpp log -1 --format='%h %cd' --date=short)"
cmake -S third_party/llama.cpp -B build/llama.cpp $GEN -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON >/dev/null
cmake --build build/llama.cpp -j"$J" --target llama-bench llama-cli llama-server test-backend-ops llama-completion 2>&1 | tail -2 || \
cmake --build build/llama.cpp -j"$J" --target llama-bench llama-cli llama-server test-backend-ops 2>&1 | tail -2
echo "== ggml-xdna1 plugin"
cmake -S reference/ggml-xdna1 -B build/ggml-xdna1 $GEN -DCMAKE_BUILD_TYPE=Release \
      -DGGML_SOURCE_DIR="$PWD/third_party/llama.cpp/ggml" >/dev/null
cmake --build build/ggml-xdna1 -j"$J" 2>&1 | tail -2
ls -la build/ggml-xdna1/lib/*.so build/ggml-xdna1/bin/ 2>/dev/null
echo "build done"
