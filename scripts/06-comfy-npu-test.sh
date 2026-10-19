#!/usr/bin/env bash
# ComfyUI image-generation test: CPU-only vs CPU+NPU (Linear layers + attention matmuls on XDNA1). Run inside the container:
#   docker/run.sh scripts/06-comfy-npu-test.sh [steps] [extra-attention-mode]
# The optional second argument (npu|cpu) adds a 4th run with the attention gate forced to that mode.
# RUNS selects the runs (default "cpu npu npu2"; e.g. RUNS=cpu for a CPU-only baseline at the current machine load).
# COMFY_EXTRA_ARGS adds ComfyUI server flags, e.g. --use-pytorch-cross-attention (SDPA instead of the default
# sub-quadratic CPU attention) for both the CPU and the NPU runs. docker/run.sh does not forward host variables,
# so set them inside the container:  docker/run.sh env COMFY_EXTRA_ARGS=--use-pytorch-cross-attention scripts/06-comfy-npu-test.sh 4
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
STEPS="${1:-8}"; EXTRA_ATTN="${2:-}"; PORT=8188; RUNS="${RUNS:-cpu npu npu2}"
has() { case " $RUNS " in *" $1 "*) return 0;; esac; return 1; }
mkdir -p logs outputs/comfy
export PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT"
python3 third_party/ComfyUI/main.py --cpu --listen 127.0.0.1 --port $PORT --disable-auto-launch ${COMFY_EXTRA_ARGS:-} \
  --extra-model-paths-config comfyui/extra_model_paths.yaml --output-directory "$ROOT/outputs/comfy" \
  --base-directory "$ROOT/third_party/ComfyUI" > logs/comfy-server.log 2>&1 &
SRV=$!
echo "ComfyUI pid $SRV; waiting for the API ..."
for i in $(seq 1 120); do curl -s "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && break; sleep 2; done
curl -s "http://127.0.0.1:$PORT/system_stats" | head -c 300; echo
curl -s "http://127.0.0.1:$PORT/object_info/XDNA1NpuLinearOffload" | head -c 200; echo
echo "load average now: $(cut -d" " -f1-3 /proc/loadavg)"
has cpu && { echo; echo "== run 1: CPU only"; python3 scripts/07-comfy-api-run.py --mode cpu --steps "$STEPS"; }
NPU_IMG=""
has npu && { echo; echo "== run 2: CPU + NPU (Linear layers + attention gate)"; python3 scripts/07-comfy-api-run.py --mode npu --steps "$STEPS" | tee logs/06-run2.out
  NPU_IMG=$(grep -o "outputs/comfy/[^']*png" logs/06-run2.out | head -1); }
# a different seed so ComfyUI does not serve run 2 from its node cache
has npu2 && { echo; echo "== run 3: CPU + NPU again (kernels compiled, weights resident, attention gate decided)"; python3 scripts/07-comfy-api-run.py --mode npu --steps "$STEPS" --seed 8; }
if [ -n "$EXTRA_ATTN" ]; then
  echo; echo "== run 4: CPU + NPU with attention forced to $EXTRA_ATTN"; python3 scripts/07-comfy-api-run.py --mode npu --steps "$STEPS" --attention "$EXTRA_ATTN"
fi
echo; echo "== image difference, CPU-only vs NPU (same seed)"
CPU_IMG=$(ls -t outputs/comfy/xdna1_cpu_${STEPS}st_*.png 2>/dev/null | head -1)
[ -n "$CPU_IMG" ] && [ -n "$NPU_IMG" ] && { echo "$CPU_IMG vs $NPU_IMG"; python3 scripts/compare-images.py "$CPU_IMG" "$NPU_IMG"; }
if [ -n "$EXTRA_ATTN" ]; then X_IMG=$(ls -t outputs/comfy/xdna1_npu_${STEPS}st_attn-${EXTRA_ATTN}_*.png 2>/dev/null | head -1); [ -n "$X_IMG" ] && { echo "$CPU_IMG vs $X_IMG"; python3 scripts/compare-images.py "$CPU_IMG" "$X_IMG"; }; fi
echo; echo "== sampler steps (s/it per UNet call, runs in order)"; grep -oE "[0-9]+/${STEPS} \[[0-9:]+<[0-9:]+, +[0-9.]+s/it\]" logs/comfy-server.log | tr "\n" " "; echo
echo "load average now: $(cut -d" " -f1-3 /proc/loadavg)"
echo; echo "== server log: NPU lines"; grep -E "xdna1-npu|Prompt executed|error|Error|Traceback" logs/comfy-server.log | tail -60 | cut -c1-400
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; echo "server stopped"
