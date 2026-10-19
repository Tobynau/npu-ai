#!/usr/bin/env bash
# Run a command inside the npu-ai container with the NPU (and iGPU) passed through.
# Usage: docker/run.sh [command...]      (no command = interactive shell)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${NPU_IMAGE:-npu-ai:latest}"
RENDER_GID="$(stat -c %g /dev/accel/accel0)"
TTY=(); [ -t 0 ] && TTY=(-it)
# Memory cap (NPU_MEM, default 14g, no swap) and a machine-wide lock: only ONE npu-ai container at a time,
# so two sessions can never load models concurrently and OOM the desktop (that happened on 2026-10-18).
MEM="${NPU_MEM:-14g}"
exec flock /tmp/npu-ai-container.lock docker run --rm "${TTY[@]}" \
  --memory "${MEM}" --memory-swap "${MEM}" --oom-score-adj 500 \
  --device /dev/accel/accel0 \
  $( [ -e /dev/dri ] && echo --device /dev/dri ) \
  $( [ -e /dev/kfd ] && echo --device /dev/kfd ) \
  --group-add "${RENDER_GID}" \
  --ulimit memlock=-1:-1 \
  --shm-size 8g \
  -v "${ROOT}:${ROOT}" \
  -v "${ROOT}:/work" \
  -v "${HOME}/.cache/huggingface:/hf" \
  $( [ -d "${HOME}/ComfyUI/models" ] && echo -v "${HOME}/ComfyUI/models:/comfy-models:ro" ) \
  -w "${ROOT}" \
  --network host \
  "${IMAGE}" "$@"
