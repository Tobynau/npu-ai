#!/usr/bin/env bash
# Interactive ComfyUI (CPU + XDNA1 NPU nodes) at http://127.0.0.1:8188 — Ctrl+C stops it.
# Models: this repo's models/ plus ~/ComfyUI/models (read-only). Output: outputs/comfy/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/docker/run.sh" env PYTHONPATH="$ROOT" python3 "$ROOT/third_party/ComfyUI/main.py" --cpu \
  --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  --extra-model-paths-config "$ROOT/comfyui/extra_model_paths.yaml" \
  --output-directory "$ROOT/outputs/comfy" --base-directory "$ROOT/third_party/ComfyUI" "$@"
