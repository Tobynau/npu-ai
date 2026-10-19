#!/usr/bin/env bash
# Re-take every benchmark serially on an IDLE machine (waits for compiles/containers to end and load < 3).
# Run on the host: scripts/rerun-baselines.sh   (logs overwrite logs/*.log)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
busy_procs() { for p in rustc cc1plus cargo gmake ninja make clang clang++ cc1 ld lld; do pgrep -x "$p" >/dev/null && printf "%s " "$p"; done; }
idle() { [ -z "$(busy_procs)" ] && [ -z "$(docker ps -q)" ] && awk '{exit !($1 < 3.0)}' /proc/loadavg; }
wait_idle() {
  local n=0
  until idle; do sleep 10; n=$((n+1)); [ $((n % 18)) -eq 0 ] && echo "  still busy at $(date +%T): load $(cut -d' ' -f1 /proc/loadavg), procs: $(busy_procs), containers: $(docker ps -q | wc -l)"; done
  echo "idle at $(date +%T), load $(cut -d' ' -f1 /proc/loadavg)"
}
run() { echo; echo "################ $1 ################"; wait_idle; local t0=$SECONDS; eval "$2" > "logs/$3.log" 2>&1; echo "exit=$? in $((SECONDS - t0))s"; }
echo "waiting for an idle machine (no compilers, no containers, load < 3) ..."
run "08 LLM matrix: CPU 6/12 threads + NPU plugin (llama-bench)" "scripts/08-llm-bench-matrix.sh" 08-bench-matrix
grep -E "^\| (CPU|NPU)" logs/08-bench-matrix.log | cut -c1-200
run "09 iGPU Vulkan (llama-bench)" "scripts/09-igpu-vulkan-bench.sh" 09-igpu-vulkan
grep -E "^###|^\|" logs/09-igpu-vulkan.log | cut -c1-200
run "05 GEMM module vs torch (SD1.5 shapes)" "docker/run.sh python3 scripts/05-gemm-test.py" 05-gemm-test
tail -3 logs/05-gemm-test.log | cut -c1-200
run "06 ComfyUI CPU vs NPU (8 steps 512x512)" "docker/run.sh scripts/06-comfy-npu-test.sh 8" 06-comfy-npu-test
grep -E "^\[(cpu|npu)\] done|UNet call" logs/06-comfy-npu-test.log logs/comfy-server.log | cut -c1-160 | tail -20
echo; echo "all baselines re-taken at $(date +%T)"
