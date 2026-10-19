#!/usr/bin/env bash
# LLM test 3: c8dhjp4tyv-bit/hawkpoint-npu-llm (reference/hawkpoint-npu-llm): whole decoder layers run as
# single NPU dispatches (SmolLM2-135M engine, Qwen2.5-0.5B engine), plus its OpenAI-compatible server.
# The project pins Python 3.12 / mlir-aie 57d7494e99c (~v1.4.0) / Peano 21.0.0.2026072001, so it runs in
# its own image: build once with docker/build-hawkpoint.sh, then
#   NPU_IMAGE=npu-ai-hawkpoint:latest docker/run.sh scripts/10-llm-hawkpoint.sh [all|prepare|smollm|qwen|server|samples]
# Models go to models/hawkpoint/ (sources + converted); JSON reports and server logs to logs/10-*.
# Always go through docker/run.sh (14 GB memory cap, one npu-ai container at a time); steps run one model at a time
# and the server step kills its server before returning.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HP="$ROOT/reference/hawkpoint-npu-llm"
MODELS="$ROOT/models/hawkpoint"
SMOL="$MODELS/SmolLM2-135M-Instruct-xdna1-w8a16"
QWEN="$MODELS/Qwen2.5-0.5B-Instruct-xdna1-w8a16"
LOGS="$ROOT/logs"
STEP="${1:-all}"
# the project's attention kernel has a fixed 64-token context (prompt + generation); longer prompts are trimmed
SAMPLE_PROMPT="What is an NPU and why is it power efficient?"
SAMPLE_TOKENS=32
export PYTHONUNBUFFERED=1
unset XILINX_XRT   # the distro XRT is found via the linker; a set XILINX_XRT redirects the loader to $XILINX_XRT/lib/x86_64-linux-gnu
# six physical cores; SmolLM's LM head and the CPU reference run through NumPy/OpenBLAS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"
# keep IRON's compiled-kernel cache (NPU_CACHE_HOME) on the host so the next container start does not recompile
export NPU_CACHE_HOME="${NPU_CACHE_HOME:-$ROOT/build/npu-cache}"; mkdir -p "$NPU_CACHE_HOME" "$MODELS/sources" "$LOGS"
cd "$HP"

banner() { echo; echo "================ $* ($(date +%H:%M:%S))"; }
run() { echo "+ $*"; "$@"; local rc=$?; [ $rc -ne 0 ] && echo "!! exit $rc: $*"; return $rc; }
# Memory discipline (this 30 GB laptop OOM-killed the desktop on 2026-10-18 when two sessions loaded models at once):
# docker/run.sh caps the container at 14 GB and serialises npu-ai containers with a machine-wide flock; on top of
# that, refuse to start a model-loading step with less than 10 GB available host-wide (NPU_MIN_FREE_GB to change).
need_mem() {
  local avail; avail=$(free -g | awk '/^Mem:/ {print $7}')
  if [ "${avail:-0}" -lt "${NPU_MIN_FREE_GB:-10}" ]; then
    echo "!! only ${avail} GB available (need ${NPU_MIN_FREE_GB:-10}); not starting '$1' - stop other model loads first"; return 1
  fi
  echo "memory check ok: ${avail} GB available before '$1'"
}

prepare() {
  need_mem prepare || return 1
  banner "environment"
  python -c "import sys, aie, pyxrt; print('python', sys.version.split()[0]); print('aie', aie.__file__); print('pyxrt', pyxrt.__file__); print('device', pyxrt.device(0).get_info(pyxrt.xrt_info_device.name))"
  python -m pip list 2>/dev/null | grep -Ei "^(mlir-aie|llvm-aie|numpy|tokenizers|safetensors|ml-dtypes) " || uv pip list 2>/dev/null | grep -Ei "^(mlir-aie|llvm-aie|numpy|tokenizers|safetensors|ml-dtypes) "
  run python scripts/doctor.py || true
  XILINX_XRT=/opt/xilinx/xrt run python scripts/verify_hardware_versions.py --report "$LOGS/10-hardware-versions.json" || true
  python -c "import json; d=json.load(open('$LOGS/10-hardware-versions.json')); print('compatible:', d['compatible'], '| probe:', d['functional_probe']['detail'], '| version drift:', list(d['version_differences']))"
  banner "prepare models (SmolLM2-135M-Instruct download + convert, Qwen2.5-0.5B-Instruct from models/Qwen2.5-0.5B-Instruct)"
  # the local checkout is at the pinned revision 7ae5576; snapshot_download sees its .cache metadata and skips the download
  [ -e "$MODELS/sources/Qwen2.5-0.5B-Instruct" ] || ln -sfn "$ROOT/models/Qwen2.5-0.5B-Instruct" "$MODELS/sources/Qwen2.5-0.5B-Instruct"
  run python scripts/prepare_model.py --model smollm2-135m-xdna1 --model qwen2.5-0.5b-xdna1 --models-dir "$MODELS"
  du -sh "$SMOL" "$QWEN" 2>/dev/null
}

smollm() {
  need_mem smollm || return 1
  banner "SmolLM2-135M: runtime + chat acceptance (the project's hardware gate)"
  export HAWKPOINT_MODEL_DIR="$SMOL"
  run python npu_llm/tests/test_model_runtime.py
  run python npu_llm/tests/validate_chat_npu.py
  banner "SmolLM2-135M: native decode benchmark (engine, 16 tokens, 5 runs = README protocol; then 32 tokens)"
  run python scripts/benchmark_native.py "$SMOL" --tokens 16 --runs 5 --report "$LOGS/10-smollm-native-16.json" | grep -E '"(median_decode_tokens_per_second|compile_seconds|decode_tokens_per_second|ttft_seconds|peak_ram_mib|target_met)"'
  run python scripts/benchmark_native.py "$SMOL" --tokens 32 --runs 3 --report "$LOGS/10-smollm-native-32.json" | grep -E '"(median_decode_tokens_per_second|decode_tokens_per_second|ttft_seconds)"'
  banner "SmolLM2-135M: sample completion"
  run python npu_llm/chat.py --model "$SMOL" --max-new-tokens $SAMPLE_TOKENS --prompt "$SAMPLE_PROMPT"
  banner "SmolLM2-135M: engine vs CPU BF16 reference agreement (slow: CPU reference for 6 prompts)"
  run python npu_llm/tests/validate_engine_npu.py "$SMOL" --report "$LOGS/10-smollm-engine-agreement.json" | tail -15
}

qwen() {
  need_mem qwen || return 1
  banner "Qwen2.5-0.5B: component tests + 32-token CPU/NPU agreement gate"
  export HAWKPOINT_QWEN_MODEL_DIR="$QWEN"
  run python npu_llm/tests/test_qwen_components_npu.py | tail -20
  run python npu_llm/tests/validate_qwen_tokens.py "$QWEN" --tokens 32 --report "$LOGS/10-qwen-token-agreement.json" | tail -20
  banner "Qwen2.5-0.5B: native decode benchmark (engine, 16 tokens x 5 runs; 32 tokens x 3)"
  run python scripts/benchmark_native.py "$QWEN" --tokens 16 --runs 5 --min-tps 20 --report "$LOGS/10-qwen-native-16.json" | grep -E '"(median_decode_tokens_per_second|compile_seconds|decode_tokens_per_second|ttft_seconds|peak_ram_mib|target_met)"'
  run python scripts/benchmark_native.py "$QWEN" --tokens 32 --runs 3 --min-tps 20 --report "$LOGS/10-qwen-native-32.json" | grep -E '"(median_decode_tokens_per_second|decode_tokens_per_second|ttft_seconds)"'
  banner "Qwen2.5-0.5B: sample completion"
  run python npu_llm/chat.py --model "$QWEN" --max-new-tokens $SAMPLE_TOKENS --prompt "$SAMPLE_PROMPT"
  banner "Qwen2.5-0.5B: fused prefill benchmark vs NumPy BF16 (32 positions)"
  run python npu_llm/tests/benchmark_qwen_fused.py "$QWEN" --positions 32 --report "$LOGS/10-qwen-fused-benchmark.json" | tail -12
}

server() {
  need_mem server || return 1
  banner "OpenAI-compatible server: both models, streaming benchmark through HTTP"
  export HAWKPOINT_API_KEY="${HAWKPOINT_API_KEY:-npu-ai-local-test-key}"
  python npu_llm/api_server.py --models-dir "$MODELS" --api-key "$HAWKPOINT_API_KEY" --rate-limit-per-minute 0 > "$LOGS/10-api-server.log" 2>&1 &
  local pid=$!
  trap 'kill $pid 2>/dev/null' RETURN
  for _ in $(seq 1 240); do curl -fsS http://127.0.0.1:8000/ready >/dev/null 2>&1 && break; sleep 1; done
  curl -fsS http://127.0.0.1:8000/ready || { echo "server not ready; see logs/10-api-server.log"; tail -30 "$LOGS/10-api-server.log"; return 1; }
  echo; curl -sS http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $HAWKPOINT_API_KEY"; echo
  for m in smollm2-135m-xdna1 qwen2.5-0.5b-xdna1; do
    banner "API benchmark: $m (10 streaming requests, 32 tokens)"
    run python scripts/benchmark_api.py --url http://127.0.0.1:8000 --model "$m" --requests 10 --concurrency 1 --tokens 32 --format markdown -o "$LOGS/10-api-bench-$m.md"
    cat "$LOGS/10-api-bench-$m.md"
    banner "API sample: $m"
    curl -sS http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $HAWKPOINT_API_KEY" \
      -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"$SAMPLE_PROMPT\"}],\"max_tokens\":$SAMPLE_TOKENS}" \
      | python -c 'import json,sys; r=json.load(sys.stdin); print(json.dumps(r["choices"][0]["message"]["content"])); print({k:v for k,v in r.get("x_hawkpoint_stats",{}).items() if k in ("decode_tokens_per_second","ttft_seconds","generated_tokens","prompt_tokens")}, r.get("usage"))'
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; trap - RETURN
}

samples() {
  need_mem samples || return 1
  banner "sample completions (chat.py, local NPU, $SAMPLE_TOKENS tokens)"
  run python npu_llm/chat.py --model "$SMOL" --max-new-tokens $SAMPLE_TOKENS --prompt "$SAMPLE_PROMPT"
  run python npu_llm/chat.py --model "$QWEN" --max-new-tokens $SAMPLE_TOKENS --prompt "$SAMPLE_PROMPT"
}

case "$STEP" in
  all) prepare && smollm; qwen; server ;;
  samples) samples ;;
  prepare) prepare ;;
  smollm) smollm ;;
  qwen) qwen ;;
  server) server ;;
  *) echo "usage: $0 [all|prepare|smollm|qwen|server|samples]"; exit 2 ;;
esac
