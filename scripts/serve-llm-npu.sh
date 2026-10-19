#!/usr/bin/env bash
# OpenAI-compatible LLM server (llama-server) with prompt-processing GEMMs on the XDNA1 NPU.
# Usage: scripts/serve-llm-npu.sh [model.gguf] [port]      -> http://127.0.0.1:8080/v1/chat/completions
# Runs inside the container (needs memlock); the host sees the port because run.sh uses --network host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="${1:-$ROOT/models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"; PORT="${2:-8080}"
exec "$ROOT/docker/run.sh" env GGML_BACKEND_PATH="$ROOT/build/ggml-xdna1/lib/libggml-xdna1.so" \
  "$ROOT/build/llama.cpp/bin/llama-server" -m "$M" --device XDNA1 -ngl 0 -t 6 -c 4096 \
  --host 127.0.0.1 --port "$PORT" --jinja
