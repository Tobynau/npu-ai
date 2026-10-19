#!/usr/bin/env bash
# ComfyUI image-generation test on the Radeon 760M iGPU (ROCm): the same workflow as 06 (scripts/07-comfy-api-run.py),
# run on the host (no container) with the ~/ComfyUI install and its venv's ROCm torch, started the way
# ~/ComfyUI/run-rocm.sh starts it (SDPA attention, --gpu-only, no pinned memory, gfx1103 as 11.0.0).
#   scripts/12-comfy-igpu-test.sh [steps...]        (default: 4 8)
# Run order per step count: first prompt (seed 7, includes model load and kernel warm-up), then two warm prompts.
# COMFY_DIR overrides the ComfyUI install (default ~/ComfyUI).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
# same machine-wide lock as docker/run.sh: never run next to an NPU container (memory, and clean timings)
if [ -z "${NPU_AI_LOCKED:-}" ]; then NPU_AI_LOCKED=1 exec flock /tmp/npu-ai-container.lock "$0" "$@"; fi
STEPS_LIST="${*:-4 8}"; PORT=8189
COMFY="${COMFY_DIR:-$HOME/ComfyUI}"
SP="$COMFY/venv/lib/python3.12/site-packages"
avail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
[ "$avail_gb" -lt 10 ] && { echo "only ${avail_gb} GB available, not starting (need 10)"; exit 1; }

# ~/ComfyUI/venv was created with Ubuntu's Python 3.12, which the 26.04 upgrade replaced with 3.14, so its
# python no longer sees the venv's packages. A shim venv on uv's CPython 3.12 adds that site-packages
# directory (with its .pth files) and leaves ~/ComfyUI untouched.
SHIM="$ROOT/build/igpu-scratch/venv312"
if [ ! -x "$SHIM/bin/python" ]; then
  PY312=$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*/bin/python3.12 2>/dev/null | head -1)
  [ -n "$PY312" ] || { echo "no Python 3.12 found (uv python install 3.12)"; exit 1; }
  "$PY312" -m venv --without-pip "$SHIM"
  echo "import site; site.addsitedir('$SP')" > "$SHIM/lib/python3.12/site-packages/comfy-venv.pth"
fi
"$SHIM/bin/python" -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)" || exit 1

mkdir -p logs outputs/comfy build/igpu-scratch/comfy-out
RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
e0=$(cat $RAPL); sleep 10; e1=$(cat $RAPL)
echo "idle package power before the server: $(( (e1 - e0) / 10000 )) mW; load average $(cut -d" " -f1-3 /proc/loadavg)"

export PYTHONUNBUFFERED=1 HSA_OVERRIDE_GFX_VERSION=11.0.0
(cd "$COMFY" && exec "$SHIM/bin/python" main.py --use-pytorch-cross-attention --gpu-only --disable-pinned-memory \
  --listen 127.0.0.1 --port $PORT --disable-auto-launch \
  --output-directory "$ROOT/build/igpu-scratch/comfy-out") > logs/comfy-server-igpu.log 2>&1 &
SRV=$!
echo "ComfyUI (iGPU) pid $SRV; waiting for the API ..."
for i in $(seq 1 120); do curl -s "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && break; sleep 2; done
curl -s "http://127.0.0.1:$PORT/system_stats" | head -c 600; echo

run() { python3 scripts/07-comfy-api-run.py --server "http://127.0.0.1:$PORT" --mode igpu "$@"; }
for STEPS in $STEPS_LIST; do
  echo; echo "== $STEPS steps, first prompt (seed 7)"; run --steps "$STEPS" --seed 7
  echo "== $STEPS steps, warm (seed 8)"; run --steps "$STEPS" --seed 8
  echo "== $STEPS steps, warm (seed 9)"; run --steps "$STEPS" --seed 9
done

echo; echo "== image difference, CPU-only vs iGPU (same seed 7)"
for STEPS in $STEPS_LIST; do
  CPU_IMG=$(ls -t outputs/comfy/xdna1_cpu_${STEPS}st_*.png 2>/dev/null | head -1)
  G_IMG=$(ls -t outputs/comfy/xdna1_igpu_${STEPS}st_*.png 2>/dev/null | sed -n 3p)   # seed 7 = third newest
  [ -n "$CPU_IMG" ] && [ -n "$G_IMG" ] && { echo "$CPU_IMG vs $G_IMG"; "$SHIM/bin/python" scripts/compare-images.py "$CPU_IMG" "$G_IMG"; }
done
echo; echo "== sampler steps and prompt times (server log, runs in order)"
grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+(s/it|it/s)\]|Prompt executed in [0-9.]+ seconds" logs/comfy-server-igpu.log \
  | grep -E "^(([0-9]+)/\2 |Prompt)" | tr "\n" " "; echo
echo "load average now: $(cut -d" " -f1-3 /proc/loadavg)"
grep -E "Traceback|Error|error" logs/comfy-server-igpu.log | head -20
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; echo "server stopped"
