#!/usr/bin/env bash
# LLM test 4: the two remaining XDNA1 runtimes of c8dhjp4tyv-bit/hawkpoint-npu-llm (reference/hawkpoint-npu-llm):
#   ollama  - the project's Ollama v0.33.3 patch + external GGML backend for amdxdna/XRT (ollama-xdna/): clone the
#             pinned tag, apply the patch, build Ollama's native CPU runtime and libggml-xdna.so, run the standalone
#             8-expert NPU correctness test, then serve the pinned qwen2.5:0.5b from the build tree and benchmark
#             CPU-only vs NPU placement. No system install (the project's install-stage.sh/polkit step is skipped).
#   colibri - the Colibri direct C/XDNA backend (colibri-xdna/): pinned JustVugg/colibri checkout + patch, the
#             C++/XRT backend linked into the GLM engine, and the standalone NPU ABI test (--test-npu).
# Runs in the npu-ai-hawkpoint image (docker/Dockerfile.hawkpoint step 5 adds Go 1.26, jq, ccache):
#   NPU_IMAGE=npu-ai-hawkpoint:latest docker/run.sh scripts/11-llm-hawkpoint-ollama-colibri.sh [all|ollama|ollama-build|ollama-bench|colibri]
# Build trees: build/ollama-xdna/ (source, native, stage, Go/ccache caches) and build/colibri-build/; Ollama's model
# store is models/ollama/; logs and JSON reports go to logs/11-*.
# Same memory discipline as scripts/10: always through docker/run.sh (14 GB cap, one npu-ai container at a time);
# model-loading steps refuse to start with < 10 GB available host-wide, run one model at a time, and the bench step
# kills its ollama server (and runner) before returning.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HP="$ROOT/reference/hawkpoint-npu-llm"
OB="$ROOT/build/ollama-xdna"          # --build-root of ollama-xdna/scripts/build-and-install.sh
CB="$ROOT/build/colibri-build"        # --build-root of colibri-xdna/scripts/build.sh
OMODELS="$ROOT/models/ollama"
LOGS="$ROOT/logs"
STEP="${1:-all}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:0.5b}"            # release-pins.json: ollama.model
PINNED_MANIFEST=a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67   # release-pins.json: ollama.model_manifest_sha256
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11435}"     # not 11434: never collide with a host Ollama
BENCH_REQUESTS="${BENCH_REQUESTS:-6}"
BENCH_TOKENS="${BENCH_TOKENS:-32}"
# the two placements whose tile cache thrashes run at ~0.15 tok/s here, so they get short runs by default
BENCH_REQUESTS_SLOW="${BENCH_REQUESTS_SLOW:-3}"
BENCH_TOKENS_SLOW="${BENCH_TOKENS_SLOW:-16}"
BENCH_PLACEMENTS="${BENCH_PLACEMENTS:-cpu npu npu-cache npu-stream npu-native}"
# Qwen2.5-0.5B through the backend's fixed 8x2048x2048 W8 program: every projection is padded into 32 MiB tiles,
# 9 tiles per layer (q,k,v,o,gate,up + 3 for down) = 216 tiles = 6.75 GiB per token; the default 512 MiB cache
# (16 tiles) thrashes, so 'npu-cache' gives it room for all of them (pinned host memory inside the 14 GB container cap)
NPU_CACHE_MB="${NPU_CACHE_MB:-7168}"
JOBS="${OLLAMA_XDNA_JOBS:-8}"
export PYTHONUNBUFFERED=1
unset XILINX_XRT   # the distro XRT is found via the linker; a set XILINX_XRT redirects the loader to $XILINX_XRT/lib/x86_64-linux-gnu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
mkdir -p "$OB" "$CB" "$OMODELS" "$LOGS"

banner() { echo; echo "================ $* ($(date +%H:%M:%S))"; }
run() { echo "+ $*"; "$@"; local rc=$?; [ $rc -ne 0 ] && echo "!! exit $rc: $*"; return $rc; }
need_mem() {
  local avail; avail=$(free -g | awk '/^Mem:/ {print $7}')
  if [ "${avail:-0}" -lt "${NPU_MIN_FREE_GB:-10}" ]; then
    echo "!! only ${avail} GB available (need ${NPU_MIN_FREE_GB:-10}); not starting '$1' - stop other model loads first"; return 1
  fi
  echo "memory check ok: ${avail} GB available before '$1'"
}

# ---------------------------------------------------------------- Ollama: build ----------------------------------------
ollama_build() {
  banner "Ollama XDNA1: toolchain"
  go version; cmake --version | head -1; ninja --version; jq --version; ccache --version | head -1
  # keep Go's module/build caches and ccache on the host so a second container start does not re-download/-compile
  export GOPATH="$OB/go" GOMODCACHE="$OB/go/pkg/mod" GOCACHE="$OB/go/cache" GOTOOLCHAIN=local
  export CCACHE_DIR="$OB/ccache" CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
  export OLLAMA_XDNA_JOBS="$JOBS"
  banner "Ollama XDNA1: verify-system.sh (the project's host checks, inside the container)"
  run "$HP/ollama-xdna/scripts/verify-system.sh" || return 1
  banner "Ollama XDNA1: build-and-install.sh --backend cpu --rebuild-runtime --no-install (clone v0.33.3, patch, go test, cmake native CPU runtime, libggml-xdna.so, NPU test, --list-devices)"
  local resume=(); [ -d "$OB/source/.git" ] && resume=(--resume)
  local t0=$SECONDS
  (cd "$HP" && run ollama-xdna/scripts/build-and-install.sh --backend cpu --rebuild-runtime --no-install --build-root "$OB" --jobs "$JOBS" "${resume[@]}") || { echo "!! Ollama build failed after $((SECONDS - t0)) s"; return 1; }
  echo "build time $((SECONDS - t0)) s"
  banner "Ollama XDNA1: stage layout"
  ls -la "$OB/stage/bin" "$OB/stage/lib/ollama" "$OB/stage/lib/ollama/xdna" | head -40
  "$OB/stage/bin/ollama" --version 2>/dev/null || true
  banner "Ollama XDNA1: standalone NPU test again (8 experts x 2048x2048 W8/BF16 GEMV, timed)"
  ( time "$OB/test-experts" "$OB/stage/lib/ollama/xdna/experts.xclbin" "$OB/stage/lib/ollama/xdna/insts.bin" ) 2>&1
  banner "Ollama XDNA1: llama-server --list-devices with the XDNA backend"
  LD_LIBRARY_PATH="$OB/stage/lib/ollama" GGML_BACKEND_PATH="$OB/stage/lib/ollama/xdna/libggml-xdna.so" \
    GGML_XDNA_XCLBIN="$OB/stage/lib/ollama/xdna/experts.xclbin" GGML_XDNA_INSTS="$OB/stage/lib/ollama/xdna/insts.bin" \
    "$OB/stage/lib/ollama/llama-server" --list-devices 2>&1 | tail -8
}

# ---------------------------------------------------------------- Ollama: serve + bench --------------------------------
SERVER_PID=
start_server() {   # start_server LABEL [VAR=value ...]  -> ollama serve from the stage, log logs/11-ollama-serve-LABEL.log
  local label="$1"; shift
  stop_server
  echo "+ ollama serve ($label) $*"
  env "$@" OLLAMA_MODELS="$OMODELS" OLLAMA_KEEP_ALIVE=15m OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
    "$OB/stage/bin/ollama" serve > "$LOGS/11-ollama-serve-$label.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do curl -fsS "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1 && break; sleep 1; done
  curl -fsS "http://$OLLAMA_HOST/api/version" >/dev/null || { echo "!! ollama serve did not come up; see logs/11-ollama-serve-$label.log"; tail -20 "$LOGS/11-ollama-serve-$label.log"; return 1; }
  echo "  ollama $(curl -fsS "http://$OLLAMA_HOST/api/version") on $OLLAMA_HOST (pid $SERVER_PID)"
}
stop_server() {
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=; fi
  pkill -f "$OB/stage/lib/ollama/llama-server" 2>/dev/null || true
}
trap stop_server EXIT

NPU_ENV=(GGML_BACKEND_PATH="$OB/stage/lib/ollama/xdna/libggml-xdna.so"
         GGML_XDNA_XCLBIN="$OB/stage/lib/ollama/xdna/experts.xclbin"
         GGML_XDNA_INSTS="$OB/stage/lib/ollama/xdna/insts.bin"
         OLLAMA_XDNA_GPU_LAYERS=0)   # the patch's default: -ngl 0 --fit off (XDNA1 allows one context, no fit probe)

ollama_native_compile() {   # the project's opt-in native Q4_K/Q6_K AIE programs, compiled with this image's mlir-aie/Peano
  banner "Ollama XDNA1: compile the native Q4_K/Q6_K expert programs (ollama-xdna/backend/compile_quantized.py)"
  export NPU_CACHE_HOME="${NPU_CACHE_HOME:-$ROOT/build/npu-cache}"
  local q; for q in q4_k q6_k; do
    local out="$OB/native-$q"
    if [ -f "$out/experts.xclbin" ] && [ -f "$out/insts.bin" ]; then echo "$out: already compiled"; continue; fi
    mkdir -p "$out"
    (cd "$HP" && time python ollama-xdna/backend/compile_quantized.py --quant "$q" --output "$out") 2>&1 | tail -6
    ls -la "$out/experts.xclbin" "$out/insts.bin" || return 1
  done
}

placement() {   # placement LABEL REQUESTS TOKENS [VAR=value ...]: restart the server with that environment, benchmark, show ps
  local label="$1" requests="$2" tokens="$3"; shift 3
  banner "Ollama placement '$label' ($requests x $tokens tokens): $*"
  start_server "$label" "$@" || return 1
  run python "$ROOT/scripts/11-ollama-bench.py" --url "http://$OLLAMA_HOST" --model "$OLLAMA_MODEL" \
      --requests "$requests" --tokens "$tokens" --threads "$OMP_NUM_THREADS" --label "$label" \
      --report "$LOGS/11-ollama-bench-$label.json"
  echo "+ ollama ps"; "$OB/stage/bin/ollama" ps
  echo "+ /api/ps"; curl -fsS "http://$OLLAMA_HOST/api/ps" | jq -c '.models[] | {name, size, size_vram, npu, npu_percent, details: .details.quantization_level}'
  echo "+ runner command line / XDNA lines in logs/11-ollama-serve-$label.log:"
  grep -E "llama-server|XDNA|xdna" "$LOGS/11-ollama-serve-$label.log" | grep -v "^time.*level=DEBUG" | sed 's/^\(.\{0,300\}\).*/\1/' | head -12
  grep -E "offload count|native" "$LOGS/11-ollama-serve-$label.log" | tail -2
  stop_server
}

ollama_bench() {
  need_mem ollama-bench || return 1
  [ -x "$OB/stage/bin/ollama" ] || { echo "!! no stage at $OB/stage; run '$0 ollama-build' first"; return 1; }
  banner "Ollama XDNA1: pull $OLLAMA_MODEL into models/ollama (manifest pinned in release-pins.json)"
  start_server pull || return 1
  run "$OB/stage/bin/ollama" pull "$OLLAMA_MODEL" || return 1
  local manifest="$OMODELS/manifests/registry.ollama.ai/library/${OLLAMA_MODEL%%:*}/${OLLAMA_MODEL##*:}"
  local sum; sum=$(sha256sum "$manifest" | cut -d' ' -f1)
  if [ "$sum" = "$PINNED_MANIFEST" ]; then echo "manifest sha256 matches release-pins.json ($sum)"; else echo "!! manifest sha256 $sum differs from the pinned $PINNED_MANIFEST (a newer upload of the same tag?)"; fi
  "$OB/stage/bin/ollama" show "$OLLAMA_MODEL" | head -12
  stop_server
  for pl in $BENCH_PLACEMENTS; do
    case "$pl" in
      # CPU only: Ollama's stock CPU runtime, no XDNA env -> the backend is not initialised (the guide's "100% CPU")
      cpu)        placement cpu "$BENCH_REQUESTS" "$BENCH_TOKENS" ;;
      # NPU, the guide's default: the backend takes every single-token dense MUL_MAT of a qwen* model (Q4_K/Q6_K rows
      # dequantised + requantised into W8 tiles kept in a 512 MiB persistent cache); prefill stays on the CPU by design
      npu)        placement npu "$BENCH_REQUESTS_SLOW" "$BENCH_TOKENS_SLOW" "${NPU_ENV[@]}" ;;
      # NPU with a cache large enough for all 216 tiles: weights are packed once, every later token only streams them
      npu-cache)  placement npu-cache "$BENCH_REQUESTS" "$BENCH_TOKENS" "${NPU_ENV[@]}" GGML_XDNA_WEIGHT_CACHE_MB="$NPU_CACHE_MB" ;;
      # NPU with the persistent cache disabled (the guide's A/B: GGML_XDNA_WEIGHT_CACHE_MB=0 = the old streaming path)
      npu-stream) placement npu-stream "$BENCH_REQUESTS_SLOW" "$BENCH_TOKENS_SLOW" "${NPU_ENV[@]}" GGML_XDNA_WEIGHT_CACHE_MB=0 ;;
      # NPU with the project's native Q4_K program (GGML blocks stay quantized in the weight BO, the AIE kernel dequantises;
      # opt-in, "performance not claimed" upstream) for the Q4_K tensors, the W8 cache above for the Q6_K ones
      npu-native) ollama_native_compile || continue
                  placement npu-native "$BENCH_REQUESTS_SLOW" "$BENCH_TOKENS_SLOW" "${NPU_ENV[@]}" GGML_XDNA_WEIGHT_CACHE_MB="$NPU_CACHE_MB" \
                    GGML_XDNA_NATIVE_QUANT_FORMAT=q4_k GGML_XDNA_NATIVE_QUANT_XCLBIN="$OB/native-q4_k/experts.xclbin" GGML_XDNA_NATIVE_QUANT_INSTS="$OB/native-q4_k/insts.bin" ;;
      *) echo "!! unknown placement '$pl' (cpu|npu|npu-cache|npu-stream|npu-native)" ;;
    esac
  done
  banner "Ollama XDNA1: summary"
  for l in $BENCH_PLACEMENTS; do
    [ -f "$LOGS/11-ollama-bench-$l.json" ] && jq -r '"\(.label): decode median \(.median_decode_tps|.*100|round/100) tok/s (\(.min_decode_tps|.*100|round/100)-\(.max_decode_tps|.*100|round/100)), TTFT median \(.median_ttft_s|.*1000|round/1000) s, prefill median \(.median_prefill_tps|.*10|round/10) tok/s, load \(.load_s|.*10|round/10) s, npu=\(.ps.models[0].npu // false)"' "$LOGS/11-ollama-bench-$l.json"
  done
}

# ---------------------------------------------------------------- Colibri ----------------------------------------------
colibri() {
  banner "Colibri direct C/XDNA: pinned checkout a8f2ca62 + patch + backend_xdna.cpp into the GLM engine, standalone NPU ABI test"
  local src=(); if [ -d "$CB/colibri/.git" ]; then
    git -C "$CB/colibri" checkout -q -- . && git -C "$CB/colibri" clean -qfd   # build.sh wants a clean tree and re-applies the patch
    src=(--source "$CB/colibri")
  fi
  local t0=$SECONDS
  (cd "$HP" && run colibri-xdna/scripts/build.sh --build-root "$CB" --jobs "$JOBS" --test-npu "${src[@]}") || { echo "!! Colibri build/test failed after $((SECONDS - t0)) s"; return 1; }
  echo "build+test time $((SECONDS - t0)) s"
  ls -la "$CB/colibri/c/colibri" "$CB/test_backend_xdna"
  banner "Colibri: standalone NPU test again, timed"
  ( time "$CB/test_backend_xdna" "$HP/ollama-xdna/backend/artifacts/experts-8x2048x2048/experts.xclbin" "$HP/ollama-xdna/backend/artifacts/experts-8x2048x2048/insts.bin" ) 2>&1
  banner "Colibri: engine with the XDNA backend"
  "$CB/colibri/c/colibri" --help 2>&1 | head -15 || true
  if [ -n "${COLI_MODEL:-}" ] && [ -d "$COLI_MODEL" ]; then
    need_mem colibri-chat || return 1
    banner "Colibri chat on $COLI_MODEL with COLI_CUDA=1 (= the XDNA backend)"
    export COLI_XDNA_XCLBIN="$HP/ollama-xdna/backend/artifacts/experts-8x2048x2048/experts.xclbin"
    export COLI_XDNA_INSTS="$HP/ollama-xdna/backend/artifacts/experts-8x2048x2048/insts.bin"
    export COLI_XDNA_MEMORY_MB="${COLI_XDNA_MEMORY_MB:-512}" COLI_CUDA=1 CUDA_EXPERT_GB="${CUDA_EXPERT_GB:-0.5}"
    # the engine binary refuses to run without the launcher ("./coli chat --model <dir>"); coli (Python) picks the
    # engine from the model's config.json, which for a GLM-5.2 container is the XDNA-linked c/colibri built above
    printf 'What is an NPU?\n' | run "$CB/colibri/c/coli" chat --model "$COLI_MODEL" 2>&1 | tail -30
  else
    echo "No Colibri-format model: the patched engine is Colibri's GLM-5.2 one (c/colibri.c, ~372 GB int4 container);"
    echo "the smaller families (OLMoE 7 GB, Qwen3.6 20 GB) build as separate engines that the XDNA patch does not link."
    echo "Set COLI_MODEL=/path/to/colibri-model to run the chat step."
  fi
}

case "$STEP" in
  all)          ollama_build && ollama_bench; colibri ;;
  ollama)       ollama_build && ollama_bench ;;
  ollama-build) ollama_build ;;
  ollama-bench) ollama_bench ;;
  colibri)      colibri ;;
  *) echo "usage: $0 [all|ollama|ollama-build|ollama-bench|colibri]"; exit 2 ;;
esac
