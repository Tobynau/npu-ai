# Results: XDNA1 NPU on this laptop (Ryzen 5 7640U, Ubuntu 26.04)

All runs: `docker/run.sh` container (XRT 2.21.75, mlir-aie 1.4.4, Peano 22.0 nightly 2026-09-21, Python 3.14),
in-tree `amdxdna` driver from kernel 7.2.6, NPU firmware 1.5.5.391. Logs are in `logs/`.
Sections 2c and 2d use their own image (`docker/Dockerfile.hawkpoint`: Python 3.12, mlir-aie at the commit that project pins; Go 1.26 for its Ollama build).

> **Caveat on all numbers below:** they were measured while another workload (a model training on the
> iGPU, plus a `cargo test` run) was active, so absolute values are pessimistic and noisy. Treat them
> as indicative. `scripts/rerun-baselines.sh` re-takes everything serially once the machine is idle.
> The under-load logs are kept in `logs/under-load/`.

## 1. Basic functionality (`scripts/01-npu-smoke.sh`)

| Check | Result |
|---|---|
| `xrt-smi examine` | `RyzenAI-npu1` at `0000:c2:00.1`, firmware 1.5.5.391 |
| `pyxrt.device(0)` | opens, name `RyzenAI-npu1` |
| IRON `vector_scalar_add` (compile with Peano + run + verify) | PASS, 1.5 s end to end |
| IRON `whole_array` matmul 512x512x512 i16 on 4 columns | PASS, NPU time 0.57 ms median, **448 GFLOPS** |

Memlock matters: on the host `ulimit -l` is 8192 KB and every NPU allocation fails silently; the
container runs with `--ulimit memlock=-1:-1` and reports `unlimited`.

## 2. LLM inference

### 2a. Qwen3-0.6B with every token generated on the NPU (`scripts/04-llm-npu-lab.sh`, Pedroj-64/npu-lab)

int8 4-column matvec kernels for decode, bf16 whole-array matmuls for prefill; attention, norms, RoPE on the CPU.

| Measurement | Result |
|---|---|
| Kernel builds (first run only) | 5 matvec xclbins in 12 s, 3 prefill xclbins in 11 s |
| Prefill, 65-token prompt | 0.6 s (~110 tok/s) |
| Decode | **8.95 tok/s** (111 ms/token; 212 NPU dispatches/token) |
| Output | coherent English ("An NPU is a type of processor that ... more energy-efficient ...") |

For reference the same project reports 18-20 tok/s on a Hawk Point 8645HS; Phoenix is the slower
sibling and this laptop was also running other work. Decode is bandwidth-bound: each token streams
the whole int8 model through the NPU's DMA.

### 2b. llama.cpp with prompt-processing GEMMs on the NPU (`scripts/03-llm-llamacpp-xdna1.sh`, FireBurn/ggml-xdna1)

The plugin talks to `/dev/accel/accel0` directly (no XRT) and offloads INT8 GEMMs with >= 64 rows,
i.e. prefill. Decode is deliberately left on the CPU (the NPU cannot beat DDR5 bandwidth on a
single row). Qwen2.5-1.5B-Instruct Q4_K_M, 8 threads, `llama-bench`:

| Backend | pp512 (tok/s) | tg32 (tok/s) |
|---|---|---|
| CPU | 116.9 +- 13.0 | 14.6 +- 2.4 |
| CPU + XDNA1 plugin | **207.8 +- 35.9** | 17.9 +- 0.5 |

Raw kernel: `test-gemm 1024x1024x64` on 4 columns passed bit-exact at 0.84 ms/dispatch.
A fuller matrix (6/12 threads, 0.5B model, iGPU Vulkan) is in section 4.

### 2c. hawkpoint-npu-llm: whole decoder layers as single NPU dispatches (`scripts/10-llm-hawkpoint.sh`, c8dhjp4tyv-bit/hawkpoint-npu-llm)

The project runs every decoder layer of SmolLM2-135M (30 layers) or Qwen2.5-0.5B (24 layers) as **one NPU
dispatch per token** (RMSNorm, projections, RoPE, attention with NPU-resident KV cache, SwiGLU, residuals all
as AIE2 kernels; weights streamed over six DMA channels), with a fixed 64-token context. SmolLM's 49k-row LM
head runs on the CPU, Qwen's on the NPU. It was validated by its author on a Hawk Point XDNA1 and lists
Phoenix as "expected compatible, unverified". Here it is verified: **both models pass the project's own
hardware gates on this Phoenix NPU**, at roughly 3/4 (SmolLM2) and 2/3 (Qwen2.5) of the Hawk Point speed.

Environment (`docker/Dockerfile.hawkpoint` -> `npu-ai-hawkpoint:latest`, built on `npu-ai:latest`):

| The project pins | What runs here | Note |
|---|---|---|
| CPython 3.12, `requirements.lock` with hashes | 3.12.14 via `uv python install`, `uv pip install --require-hashes` | numpy 2.5.4, tokenizers 0.23.3, safetensors 0.8.0, ml-dtypes 0.6.0, hf-hub 1.33.0 |
| mlir-aie git `57d7494e99c` (2026-07-28) | per-commit CI wheel `mlir_aie-1.3.5.dev115+gc69e4c3` (3 commits earlier) | the `v1.4.0` release one day later already has the IRON `Runtime(seq_fn)` rework (#3387) and breaks every design (`Runtime()` missing `seq_fn`); no wheel exists for `57d7494` itself; the 3 missing commits are a packet-flow routing pass, Windows test fixes, a cmake bump |
| Peano `llvm-aie 21.0.0.2026072001` | the same nightly wheel (`+ce8c0f8f`) | still on the llvm-aie nightly release page |
| pyxrt (XRT Python bindings) | compiled for 3.12 from `pyxrt.cpp` of the XRT `2.21.75` tag against the distro headers (`docker/pyxrt/`) | Ubuntu's `python3-xrt` only ships the 3.14 module; one `g++` line, needs `libboost-dev` |
| XRT 2.26.0, kernel 7.2.0-rc5, firmware 1.5.5.391 (Hawk Point, Fedora) | XRT 2.21.75, kernel 7.2.6, firmware 1.5.5.391 | `verify_hardware_versions.py`: **compatible** (version drift is non-fatal in its compatibility mode; its C++ probe passes all 7 XRT capabilities: device open, hw context, BO alloc/sync, xclbin load, kernel submit, ioctls; checked-in AIE kernel max error 7.5e-6). Needs `XILINX_XRT=/opt/xilinx/xrt`, a symlink shim in the image (`logs/10-hardware-versions.json`) |

Models: `scripts/prepare_model.py --models-dir models/hawkpoint` (SmolLM2-135M-Instruct downloaded at the pinned
revision; `models/Qwen2.5-0.5B-Instruct` was already at the pinned revision `7ae5576` and is reused through a
symlink). Converted packages: 443 MB (SmolLM2, W8 LM head + BF16 decoder) and 1.7 GB (Qwen2.5).

**SmolLM2-135M-Instruct** (`SmolLM2-135M-Instruct-xdna1-w8a16`, decoder engine, 30/30 layers on the NPU):

| Measurement | Phoenix 7640U (here) | Hawk Point (project README) |
|---|---:|---:|
| `validate_chat_npu.py` hardware gate | PASS: first-token argmax 198, same 32-token sky answer as the project's reference | argmax 198 |
| Native 16-token decode, 5-run median (`benchmark_native.py`) | **56.8 tok/s** (54.6-59.4) | 74.6 tok/s (73.2-75.8) |
| Native 32-token decode, 3-run median | 56.2 tok/s (51.4-59.4) | - |
| TTFT, 35-token prompt | 0.42-0.46 s | 0.28-0.29 s |
| TTFT, 31-token prompt / prefill rate | 0.38-0.39 s / 67 tok/s | - |
| Warm-up per process (kernels from `build/npu-cache`, weight upload) | 7 s (8.0 s first ever compile) | 5.1 s |
| Peak host RAM | 353 MiB | 354 MiB |
| Engine vs CPU-BF16 reference, 6 prompts x 63 positions (`validate_engine_npu.py`) | PASS: 338/377 top-1 (chunked graphs 335/377), the 7 hard misses are near-ties shared by both NPU paths | gate |

**Qwen2.5-0.5B-Instruct** (`Qwen2.5-0.5B-Instruct-xdna1-w8a16`, Qwen engine, 24/24 layers + LM head on the NPU):

| Measurement | Phoenix 7640U (here) | Hawk Point (project README) |
|---|---:|---:|
| `test_qwen_components_npu.py` (RMSNorm/projection/RoPE/attention kernels) | PASS (25 s incl. kernel compile) | gate |
| 32-token greedy sequence vs CPU BF16 vs checked-in reference (`validate_qwen_tokens.py`) | **32/32 identical** (NPU 58 ms/token vs CPU BF16 253 ms/token) | 32/32 |
| Fused 32-position prefill vs NumPy BF16 (`benchmark_qwen_fused.py`) | 31/32 exact argmax, 1 accepted BF16 near-tie; 53.7 ms vs 223 ms per position = 4.2x | gate |
| Native 16-token decode, 5-run median | **19.3 tok/s** (18.0-19.4) | 28.8 tok/s |
| Native 32-token decode, 3-run median | 19.1 tok/s (18.7-19.2) | - |
| TTFT, 40-token prompt | 1.52-1.71 s | 1.01-1.03 s |
| TTFT, 32-token prompt / prefill rate | 1.23-1.26 s / 25 tok/s | - |
| Warm-up per process | 7.5-9.2 s (1 GB of BF16 weights uploaded) | - |
| Peak host RAM | 1266 MiB | 1.28 GiB |

Sample completions (`chat.py --max-new-tokens 32`, greedy; the 64-token context forces short prompts, 48 new
tokens would leave 16 for the prompt and the server trims it):

- SmolLM2-135M, "What is an NPU and why is it power efficient?" -> *"An NPN (Negative Inverted CurrentPNU) is a
  type of power conversion circuit that uses a negative resistance to convert a high-voltage input to a"*
  (57.2 tok/s; a 135M model, the acceptance prompt about the blue sky gives the textbook answer).
- Qwen2.5-0.5B, same prompt -> *"An NPU (Neural Network Processing Unit) is a specialized hardware accelerator
  designed to accelerate the training and inference of deep neural networks. It is particularly useful for"*
  (19.1 tok/s).
- Qwen2.5-0.5B, the project's gate prompt ("Explain why the sky appears blue ... five sentences.") -> *"The sky
  appears blue during the day due to the scattering of sunlight by tiny water droplets in the Earth's
  atmosphere. These droplets, which are tiny ice"* (identical tokens on CPU and NPU).

OpenAI-compatible server (`npu_llm/api_server.py`, one NPU worker process, prefix cache on;
`scripts/benchmark_api.py`, 10 streaming chat requests of 32 tokens each, concurrency 1):

| Model | Requests | Decode tok/s p50 / mean | TTFT p50 / p95 | Latency p50 |
|---|---|---:|---:|---:|
| `smollm2-135m-xdna1` | 10/10 | 55.6 / 55.9 | 0.023 s / 0.26 s | 0.61 s |
| `qwen2.5-0.5b-xdna1` | 10/10 | 20.0 / 20.0 | 0.054 s / 13.4 s | 1.66 s |

The p50 TTFTs are prefix-cache hits (the benchmark repeats one prompt); the p95s are the real first request,
and Qwen's 13.4 s includes the model switch from SmolLM2 (the server keeps one model resident: engine load
plus 1 GB weight upload). A repeat of the same run while a Docker image was being exported on the CPU gave
51.4 / 19.8 tok/s p50 (SmolLM2's CPU LM head is the part that notices host load); that repeat is the one in
`logs/10-api-bench-*.md` and `logs/10-api-server.log`. API samples (`/v1/chat/completions`, 32 tokens, the
32-token prompt now includes the default system prompt): SmolLM2 *"An NPN (Negative Inverted Current Potential
Unit) is a type of power conversion device that converts alternating current (AC) to direct current (DC) using"*
(43.9 tok/s, TTFT 0.30 s); Qwen2.5 *"An NPU (Neural Network Processing Unit) is a specialized hardware platform
designed to accelerate the training and inference of deep learning models. It is particularly useful in"*
(19.2 tok/s, TTFT 0.85 s).

Reading: per token Qwen streams about 1 GB of BF16 weights through the NPU in 52 ms, i.e. ~19 GB/s of
effective DMA bandwidth from one dispatch per token, against ~5 GB/s for npu-lab's 212 dispatches per token
(section 2a, 9 tok/s on the 0.6B int8 Qwen3). That is the whole difference: same silicon, same firmware, same
bandwidth-bound workload, but dispatch count and weight streaming decide the decode rate. The Phoenix/Hawk Point
gap (0.76x on SmolLM2, 0.67x on Qwen2.5) matches the lower NPU clock of the 7640U; both numbers were taken on a
laptop shared with other work (a ComfyUI session was running on the CPU at the time).

Reproduce: `docker/build-hawkpoint.sh` once, then
`NPU_IMAGE=npu-ai-hawkpoint:latest docker/run.sh scripts/10-llm-hawkpoint.sh` (`all`, or `prepare|smollm|qwen|server|samples`).
Logs `logs/10-hawkpoint-*.log`, JSON gate/benchmark reports `logs/10-*.json`, kernel cache `build/npu-cache/`.
Memory: the Qwen conversion and benchmarks peak at about 1.3-2 GB, but this 30 GB laptop was OOM-killed once when
these runs overlapped with another session's ComfyUI model loads; `docker/run.sh` now caps a container at 14 GB and
runs one npu-ai container at a time, and the script refuses to start a step with less than 10 GB available.

### 2d. hawkpoint-npu-llm's other two runtimes: the Ollama XDNA1 patch and the Colibri C/XDNA backend (`scripts/11-llm-hawkpoint-ollama-colibri.sh`)

Section 2c ran the project's native MLIR-AIE engine. The project ships two more ways onto the XDNA1 NPU; both were
built and tested here inside the same `npu-ai-hawkpoint:latest` image, nothing installed on the host and no sudo
(`docker/Dockerfile.hawkpoint` step 5 adds Go 1.26.9 from go.dev, `jq` and `ccache`, +0.3 GB; cmake 4.2, ninja, g++ 15
and the distro XRT dev headers were already there, and the project's `verify-system.sh` passes all 14 host checks
inside the container through the `/opt/xilinx/xrt` symlink shim of 2c, including `lspci` seeing the NPU bound to
`amdxdna`). The Ollama guide's system-install step (`install-stage.sh`, polkit, systemd) is skipped: `ollama serve`
runs from the build tree instead.

| Path | What it is | Built here | On the NPU here |
|---|---|---|---|
| `ollama-xdna/` | patches upstream Ollama `v0.33.3` (commit `b79067b0`, `release-pins.json`) and adds an external GGML backend (`ggml-xdna.cpp`, XRT) that Ollama's bundled `llama-server` loads next to its CPU backend (the stage symlinks it as `libggml-openvino.so` so ggml's backend scan picks it up); one prebuilt AIE program (`experts.xclbin`: 8 experts x 2048x2048, int8 weights x BF16 activations, 2 cores) serves MoE expert projections and, for dense `qwen*` models, every single-token matrix-vector product | yes, **289 s** first time: clone + patch, `go test` of the 5 patched packages green, Ollama's complete native CPU runtime via `--rebuild-runtime` (llama.cpp `b10760`, 14 CPU variants, ccache on), `libggml-xdna.so`, the Go binary `0.33.3-xdna`; 2.1 GB in `build/ollama-xdna/` (stage = `stage/bin/ollama` + `stage/lib/ollama/`) | standalone hardware test **PASS** (8 experts x 2048x2048 W8/BF16 GEMV, 968 outputs checked, max error 7.5e-6, 0.24 s wall); `llama-server --list-devices` lists `XDNA0: AMD Ryzen AI XDNA1 NPU (0 MiB, 0 MiB free)`; `qwen2.5:0.5b` (the pinned manifest `a8b0c515...`) generates with its decode matmuls on the NPU, table below |
| `colibri-xdna/` | a native C++/XRT implementation of Colibri's resident-tensor backend ABI (`backend_cuda.h`: the GPU-tier hooks the pure-C engine calls "CUDA"), dropped into JustVugg/colibri at the pinned commit `a8f2ca62` with a 2.8 KB Makefile patch (`XDNA=1`): lazy dense decode GEMV, routed gate/up/SiLU/down for up to 8 experts, f32 / int8 / int4 / grouped-int4 ingestion requantised once to row-wise W8 | yes, **33 s**: blobless clone, patch, `make -C c colibri XDNA=1` (the GLM-5.2 engine, `-DCOLI_CUDA -DCOLI_XDNA`, 820 KB), 40 MB in `build/colibri-build/` | standalone ABI test **PASS** (`[XDNA] direct C backend initialized`; dense f32 matmul, grouped-int4 matmul, a 2-expert routed pipeline; max error 0.070 inside its 0.08/0.15 tolerances; 0.22 s wall) |

Both use the same checked-in xclbin (`ollama-xdna/backend/artifacts/experts-8x2048x2048/`, compiled by the project on
its Hawk Point toolchain); it loads and runs unchanged on this Phoenix `RyzenAI-npu1`, so the W8 program did not have
to be rebuilt. The project's opt-in native Q4_K/Q6_K programs (`compile_quantized.py`, GGML blocks stay quantized in
the weight buffer and the AIE kernel dequantises) are not shipped as binaries; both compiled here with the image's
mlir-aie `1.3.5.dev115` + Peano in **3 s each** (`build/ollama-xdna/native-q4_k/`, `native-q6_k/`).

**Colibri boundary.** The only engine the XDNA patch links is Colibri's GLM-5.2 one (`c/colibri.c`); it refuses to
start without a model and the GLM-5.2 container is ~372 GB (the project itself does not claim end-to-end validation
for the same reason). Colibri's small families are separate engines whose Makefile targets never take the XDNA
object (OLMoE, 7 GB, has no backend hook at all; `qwen36`, 20 GB, links the tier only with `CUDA=1`), and the generated
tiny fixtures (`make_qwen36_tiny.py` etc.) feed those engines, not `colibri`. So the chat step of `scripts/11` is gated
on `COLI_MODEL` and did not run; what is verified here is the complete Colibri ABI -> XRT -> AIE path the project's own
release gate checks.

**Ollama: CPU-only vs NPU placement, `qwen2.5:0.5b`** (Q4_K_M, 494M parameters, `ollama serve` from the stage with
`OLLAMA_MODELS=models/ollama`, 6 threads, context 4096; `scripts/11-ollama-bench.py`: streaming `/api/generate`,
greedy, one prompt per request; TTFT is wall time to the first streamed token, decode and prefill rates are Ollama's own
counters; the thrashing placements got 3 x 16-token requests, the others 6 x 32):

| Placement (`scripts/11` label) | Controls (the guide's) | Decode tok/s, median (range) | TTFT: first request / later median | Prefill tok/s | `ollama ps`, `/api/ps` |
|---|---|---:|---:|---:|---|
| `cpu` | no XDNA env: Ollama's stock CPU runtime (`zen4` variant) | **118.6** (112-125) | 0.01 s / 0.044 s | 1081 | `100% CPU`; no `npu` field |
| `npu` | `GGML_BACKEND_PATH`, `GGML_XDNA_XCLBIN`, `GGML_XDNA_INSTS` set, `OLLAMA_XDNA_GPU_LAYERS=0` -> the patched runner gets `-ngl 0 --fit off`; default 512 MiB persistent tile cache | **0.15** (0.15-0.15) | 7.0 s / 0.057 s | 739 | `100% CPU + NPU`; `npu: true`, `npu_percent: null` |
| `npu-cache` | + `GGML_XDNA_WEIGHT_CACHE_MB=7168` (room for every tile; 5.6 GiB of pinned host memory) | **0.52** (0.52-0.54) | 2.0 s / 0.059 s | 706 | same |
| `npu-stream` | + `GGML_XDNA_WEIGHT_CACHE_MB=0` (the guide's A/B: no persistent cache, re-pack every tile) | **0.32** (0.32-0.32) | 3.3 s / 0.087 s | 474 | same |
| `npu-native` | + `GGML_XDNA_NATIVE_QUANT_FORMAT=q4_k` with the Q4_K program compiled here (Q4_K tensors go to the AIE as GGML blocks, 20 MiB tiles, dequantised on the cores; the Q6_K tensors still take the 7 GiB W8 cache) | **0.54** (0.54-0.54) | 2.0 s / 0.069 s | 603 | same; runner log `XDNA native q4_k quantized kernel enabled`, 180 tile misses then all hits |

What the counters say (`logs/11-ollama-serve-*.log`, the backend's `XDNA dense Qwen offload count` lines): about 135
single-token `MUL_MAT`s per generated token go to the NPU (24 layers; `ne[1] <= 32768` keeps the 151936-row LM head on
the CPU, prefill batches stay on the CPU by design). The backend has exactly one program shape, so each of them is
padded into a 32 MiB tile of 8 x 2048 x 2048 int8 (plus a 64 KB scale slot): 180 distinct tiles for this model,
5.6 GiB of DMA per token for 0.5 GB of real weights (896-wide projections fill 1/18 of a tile's rows and 7/16 of its
columns). With the 512 MiB default (16 tiles, LRU) the cache thrashes (8 hits, 9456 misses after 56 tokens) and every
token re-dequantises, re-quantises and re-allocates XRT buffers; with no cache at all the backend reuses one buffer and
is 2x faster than the thrashing default; with everything resident (32012 hits / 180 misses) the 0.52 tok/s = 1.9 s per
token is pure tile streaming, ~3 GB/s effective, i.e. ~14 ms per 32 MiB dispatch. The first request of every NPU
placement also pays one NPU token for the prompt's last position (TTFT 2-7 s), later ones prefill on the CPU in ~60 ms;
the warm-up request after a load took 49 s (`npu`) / 19 s (`npu-cache`, packing 180 tiles) / 24 s (`npu-stream`).

Output: the three NPU placements agree token for token with each other and with the native engine's answer in 2c
("...is a type of hardware accelerator designed to accelerate the training and inference of deep learning models..."),
and differ from Ollama's CPU answer ("...is a type of computing device...") because the backend requantises the
dequantised Q4_K/Q6_K rows to row-wise int8 before the GEMV; both are sensible answers, neither is a bug.

Reading: on a dense 0.5B model the Ollama NPU path is a working placement demo, not a speed-up, and the project says
so for dense Qwen (its validated target is `qwen3-coder:30b`, a MoE whose 2048-wide experts fit the one program shape
exactly). Ollama's own CPU runtime does 119 tok/s here; the project's native engine of 2c, which streams each decoder
layer's real BF16 weights (1 GB/token) in one dispatch, does 19 tok/s; the Ollama backend streams 5.6 GiB of mostly
zeros in 135 dispatches and lands at 0.5 tok/s; the project's native Q4_K program, its announced next step ("estimated 3-8x"), changes nothing here (0.54 tok/s) because the cost is the padded tile dispatch itself, not the host-side dequantise/requantise it removes. For a reference point the project publishes no Ollama numbers in-tree
(the cells of its `BENCHMARKS.md` placement matrix are left empty by design and filled per release). A dense-shaped
program (or the native engine's layer-fused approach) is what the Ollama path would need on XDNA1.

Dead ends and gotchas on the way: Ubuntu's `golang-go` is not the Go 1.26 the patch needs, hence the go.dev tarball
in the image (checksum pinned); `docker/run.sh` forwards no host variables, so overrides go in as
`docker/run.sh env BENCH_PLACEMENTS=... scripts/11-...`; the bench's first run was stopped after the thrashing default
placement showed 0.15 tok/s (6 x 32 tokens would have taken 20 min per placement; `logs/11-ollama-bench-aborted-first-run.log`),
hence the 3 x 16 runs for `npu` and `npu-stream`; Colibri's engine binary refuses `colibri chat` and wants the
`c/coli` Python launcher with `--model`, which the project's README does not mention.

Reproduce: `docker/build-hawkpoint.sh` (rebuild adds step 5 to the existing image, the earlier layers are cached), then
`NPU_IMAGE=npu-ai-hawkpoint:latest docker/run.sh scripts/11-llm-hawkpoint-ollama-colibri.sh` (`all`, or
`ollama-build|ollama-bench|colibri`; `env BENCH_PLACEMENTS="cpu npu-cache"` to pick placements, `NPU_CACHE_MB`,
`BENCH_REQUESTS[_SLOW]`, `BENCH_TOKENS[_SLOW]`). Logs `logs/11-ollama-build.log`, `11-ollama-bench*.log`,
`11-ollama-serve-<placement>.log`, `11-colibri.log`, `11-build-image.log`; JSON per placement `logs/11-ollama-bench-<placement>.json`.
Memory: the biggest step is `npu-cache` (5.6 GiB of XRT host buffers + the model inside the 14 GB container cap); the
script refuses to start the bench with less than 10 GB available and stops its server and runner before returning.

## 3. Our GEMM module (`npu/xdna1_gemm.py`, `scripts/05-gemm-test.py`)

One compiled program (mlir-aie `whole_array`, 32x64x32 tiles, bf16 in / f32 out, 16 cores) with
dispatch-time M/K/N; weights resident; padding + chunking for arbitrary Linear shapes. Program
compile: 11.4 s once per process. Shapes below are the Linear layers of an SD1.5 UNet at
512x512 with CFG (batch 2), checked against torch's own bf16 matmul:

| M (tokens) | K | N | rel. max err vs torch | cos | NPU ms | wall ms | GFLOPS (NPU time) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 320 | 320 | 3.61e-07 | 1.00000 | 3.75 | 10.17 | 447 |
| 8192 | 320 | 2560 | 4.31e-07 | 1.00000 | 14.04 | 75.85 | 956 |
| 8192 | 1280 | 320 | 8.31e-07 | 1.00000 | 9.28 | 21.30 | 723 |
| 154 | 768 | 320 | 5.78e-07 | 1.00000 | 0.81 | 1.17 | 94 |
| 2048 | 640 | 640 | 6.50e-07 | 1.00000 | 1.90 | 3.33 | 884 |
| 2048 | 640 | 5120 | 6.21e-07 | 1.00000 | 17.62 | 31.92 | 762 |
| 2048 | 2560 | 640 | 1.66e-06 | 1.00000 | 6.67 | 11.17 | 1006 |
| 154 | 768 | 640 | 5.56e-07 | 1.00000 | 0.73 | 0.99 | 209 |
| 512 | 1280 | 1280 | 7.90e-07 | 1.00000 | 2.63 | 3.72 | 638 |
| 512 | 1280 | 10240 | 9.21e-07 | 1.00000 | 22.14 | 28.89 | 606 |
| 512 | 5120 | 1280 | 2.00e-06 | 1.00000 | 9.97 | 12.18 | 673 |
| 154 | 768 | 1280 | 6.09e-07 | 1.00000 | 0.84 | 1.05 | 359 |
| 128 | 1280 | 1280 | 7.35e-07 | 1.00000 | 0.91 | 1.14 | 460 |
| 128 | 1280 | 10240 | 8.24e-07 | 1.00000 | 6.04 | 7.37 | 555 |
| 128 | 5120 | 1280 | 2.14e-06 | 1.00000 | 2.94 | 3.38 | 572 |
| 100 | 100 | 100 | 2.31e-07 | 1.00000 | 0.56 | 0.68 | 4 |

one SD1.5 UNet pass worth of attention/FF linears (these shapes once each): 71 GFLOP, NPU 101 ms, wall 214 ms -> 333 GFLOPS end-to-end

NpuGemm: 96 calls, 144 dispatches, NPU 0.607s, wall 1.268s, 705 GFLOPS (NPU time), program compile 11.4s

The ~1 ms per-dispatch floor dominates small shapes; large-M shapes reach 500-700 GFLOPS of
NPU time. Roughly half of wall time is host work (bf16 conversion, result copy), which is the
first thing to optimise (keep activations in bf16, pipeline uploads).

## 3b. ComfyUI with Linear layers on the NPU (`scripts/06-comfy-npu-test.sh`)

ComfyUI 0.41 (`--cpu`, fp32 UNet), Realistic Vision V6 (SD1.5), 512x512, Euler, 8 steps, CFG 7
(so each step is one UNet call on a batch of 2). The custom node took over **160 Linear layers**
of the UNet (all attention q/k/v/out projections and feed-forward GEMMs, 1.7 s to upload as bf16).

| Run | Total prompt time | Per UNet call | Notes |
|---|---:|---:|---|
| CPU only | 132.4 s | 13.6 s | bit-for-bit reproducible between runs |
| CPU + NPU, first prompt | 144.9 s | 12.7 s steady | includes one-time 12.5 s program compile + 13.6 s CPU shape-discovery pass |
| CPU + NPU, next prompt | **114.8 s** | **12.3-12.8 s** | warm: 12% faster end to end |

Per UNet call the NPU executes 160 GEMMs = 466 GFLOP in 0.70-0.83 s of NPU time (600-670 GFLOPS);
the GEMM module's wall time is ~1.6 s per call, so ~0.85 s per call is host work (bf16 conversion
of activations, copying f32 results back). Implied split of a 13.6 s CPU step: ~2.5 s in these
Linear layers, ~11 s in convolutions, norms, attention softmax and the rest, which stay on the CPU.
That is why the gain is modest on SD1.5: the NPU removes the Linear share almost entirely, but the
UNet is convolution-heavy and the host round trip costs a third of what was saved.

Image check (same seed 7): same composition and content; mean absolute pixel difference 13/255,
44% of pixels within 8/255. That is the expected effect of bf16 GEMMs compounding over 8 diffusion
steps, not an error (the GEMM itself matches torch's bf16 result to 1e-7, section 3). Images are in
`outputs/comfy/` (`xdna1_cpu_8st_*.png` vs `xdna1_npu_8st_00001_.png`).

Section 3c removes the host round trip and adds the attention matmuls, 3d the fused attention kernel; `docs/03-next-steps.md` has the rest.

## 3c. Host overhead removed, attention matmuls on the NPU (`npu/xdna1_gemm.py`, `comfyui_xdna1_npu`)

Follow-up to 3 and 3b. Two changes, measured the same way (`scripts/05-gemm-test.py`,
`scripts/06-comfy-npu-test.sh 8 npu`), on a machine that other sessions kept at a load average of
12-20 (the 3b numbers were taken at 10-18, so absolute times are only comparable within one table).

**GEMM module.** The host no longer touches the data per call: torch converts the activations to
bf16 straight into the pinned XRT buffer, the f32 result is handed back as a zero-copy view of a
pooled pinned buffer (the buffer returns to the pool when the model drops the tensor, through a
finalizer on the array torch holds), and `linear()` builds the n-D shape with `as_strided` so even
N = 320 (padded to 384) costs no copy. Dispatch goes straight to the XRT kernel with a per-shape
instruction stream and buffer object that are generated once; runs are submitted without waiting,
and M-chunks of one GEMM go to sub-buffers of one A/C pair so chunk i+1 is converted while chunk i
computes. One thing to know when writing pinned memory from the host: XRT's `sync` on amdxdna is a
cache flush in both directions, so every buffer the NPU is about to write must be flushed first
(the zero fill of a fresh IRON buffer, a bias add or an in-place op of the model leaves dirty cache
lines, and the FROM_DEVICE flush after the dispatch would write them back over the result).

| M (tokens) | K | N | NPU ms before | wall ms before | NPU ms after | wall ms after | host ms before -> after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 320 | 320 | 3.75 | 10.17 | 2.09 | 4.82 | 6.4 -> 2.7 |
| 8192 | 320 | 2560 | 14.04 | 75.85 | 10.42 | 23.96 | 61.8 -> 13.5 |
| 8192 | 1280 | 320 | 9.28 | 21.30 | 5.33 | 6.73 | 12.0 -> 1.4 |
| 154 | 768 | 320 | 0.81 | 1.17 | 0.43 | 0.70 | 0.4 -> 0.3 |
| 2048 | 640 | 640 | 1.90 | 3.33 | 1.40 | 2.69 | 1.4 -> 1.3 |
| 2048 | 640 | 5120 | 17.62 | 31.92 | 27.08 | 31.72 | 14.3 -> 4.6 |
| 2048 | 2560 | 640 | 6.67 | 11.17 | 5.18 | 6.65 | 4.5 -> 1.5 |
| 154 | 768 | 640 | 0.73 | 0.99 | 0.37 | 0.58 | 0.3 -> 0.2 |
| 512 | 1280 | 1280 | 2.63 | 3.72 | 1.84 | 2.69 | 1.1 -> 0.8 |
| 512 | 1280 | 10240 | 22.14 | 28.89 | 24.87 | 25.18 | 6.8 -> 0.3 |
| 512 | 5120 | 1280 | 9.97 | 12.18 | 8.80 | 11.22 | 2.2 -> 2.4 |
| 154 | 768 | 1280 | 0.84 | 1.05 | 0.76 | 1.15 | 0.2 -> 0.4 |
| 128 | 1280 | 1280 | 0.91 | 1.14 | 1.02 | 1.55 | 0.2 -> 0.5 |
| 128 | 1280 | 10240 | 6.04 | 7.37 | 8.73 | 8.89 | 1.3 -> 0.2 |
| 128 | 5120 | 1280 | 2.94 | 3.38 | 2.77 | 3.22 | 0.4 -> 0.5 |
| 100 | 100 | 100 | 0.56 | 0.68 | 0.73 | 0.88 | 0.1 -> 0.2 |

All shapes still match torch (rel. err 1e-7 to 2e-6, same as section 3). The 16 shapes once each:
NPU 101 -> 102 ms, wall 214 -> 133 ms, i.e. the host share went from 113 ms to 31 ms (333 -> 538
GFLOPS end to end). "NPU ms" is now the span from the first submit to the last completion of a
call, so for the N-chunked shapes (N = 5120, 10240) it includes the assembly copies between chunks;
the dispatch floor is ~0.5 ms and dominates everything under 0.3 GFLOP.

**Attention.** The node installs ComfyUI's `optimized_attention_override`. For every (batch, head)
pair the NPU computes Q K^T (bf16, f32 scores), the CPU does the softmax in f32 and writes the
probabilities as bf16 into the next input buffer, and the NPU computes P V; heads are pipelined so
the NPU runs the next head's Q K^T while the CPU does the current softmax. A gate decides per shape:
ComfyUI's CPU kernel is timed during the shape-discovery pass, the NPU path once on the first
enabled call (its result is checked against the CPU kernel, max rel. err 3e-3 to 1e-2 for bf16
inputs and probabilities), and the faster side is kept. Measured on the UNet (SD1.5, 512x512,
batch 2, CPU attention = ComfyUI's default sub-quadratic kernel, 6 torch threads):

| attention shape (b, tq, tk, heads, d) | blocks / call | CPU ms | NPU ms | gate |
|---|---:|---:|---:|---|
| (2, 4096, 4096, 8, 40) self-attn 64x64 | 5 | 696 | 567 | NPU |
| (2, 1024, 1024, 8, 80) self-attn 32x32 | 5 | 56 | 37 | NPU |
| (2, 256, 256, 8, 160) self-attn 16x16 | 5 | 4 | 11 | CPU |
| (2, 64, 64, 8, 160) self-attn 8x8 | 1 | - | - | CPU (under 256 tokens) |
| (2, 4096, 77, 8, 40) cross-attn 64x64 | 5 | 9 | 39 | CPU |
| (2, 1024, 77, 8, 80) cross-attn 32x32 | 5 | 3 | 15 | CPU |
| (2, 256, 77, 8, 160) cross-attn 16x16 | 5 | 2 | 8 | CPU |

The NPU wins only where the CPU kernel materialises large score matrices (the 64x64 level: 16 heads
x 4096x4096 f32 = 1 GB per block). Standalone, torch's SDPA kernel does the 64x64 block in 270-460
ms and the 32x32 block in 27-40 ms, so against `--use-pytorch-cross-attention` the NPU path loses
every shape. Why it is slow: 32 dispatches per block at a ~0.5 ms floor, the f32 score matrix
travelling NPU -> CPU -> NPU (64 MB per head at 64x64, 40 ms of softmax + bf16 conversion per head
on this CPU), and d = 40 padding to K = 64 / N = 128 (37-69 % of the MACs are zeros). A fused
kernel that keeps all heads in one dispatch and does the softmax on the cores is what it would take.

**ComfyUI, 8 steps (`scripts/06-comfy-npu-test.sh 8 npu`).**
Same setup as 3b (ComfyUI 0.41 `--cpu`, fp32 UNet, Realistic Vision V6, 512x512, Euler, 8 steps, CFG 7, one
UNet call per step on a batch of 2). Two A/B sequences, each through one server in the order shown. During the
first the load average fell from 17 to 10 (the CPU-only steps themselves went from 10.6 s to 7.9 s within the
run), so the whole sequence was repeated at a load of 6-9. Per-call times are the sampler's s/it (CPU) or the
node's own per-call wall time (NPU); means over the 8 calls, range in brackets.

Second sequence (load 6-9, `logs/06-comfy-cpu-rerun-8st.log`, `logs/comfy-server-cpu-rerun.log`):

| Run | Prompt total | Per UNet call | GEMMs: NPU busy / host | attention: NPU / CPU | rest (convs, norms, ...) |
|---|---:|---:|---:|---:|---:|
| CPU only | 62.0 s | **6.7 s** (6.2-7.0) | - | ~3.1 s (sub-quadratic kernel) | - |
| NPU, `attention=auto`, first prompt | 66.1 s | 5.6 s (5.1-6.2) | 0.54 s / 0.12 s | 1.9 s (10 blocks) / 0.1 s (22) | 2.9 s |
| NPU, `attention=auto`, next prompt (warm) | **50.1 s** | **5.2 s** (5.0-5.6) | 0.54 s (870 GFLOPS) / 0.12 s | 1.84 s (10) / 0.10 s (22) | 2.6 s |

Per UNet call that is 6.7 -> 5.2 s (21 % faster, 19 % on the whole prompt), against 13.6 -> 12.7 s (7 %) in 3b
under a heavier load. The gate decided the same way in both sequences (64x64 self-attention: CPU 576 vs NPU
447 ms; 32x32: 49 vs 23 ms; everything else CPU).

First sequence (load 17 -> 10, `logs/06-comfy-npu-test.log`, `logs/comfy-server-8st-abab.log`), with the forced
run:

| Run | Prompt total | Per UNet call | GEMMs: NPU busy / host | attention: NPU / CPU | rest (convs, norms, ...) |
|---|---:|---:|---:|---:|---:|
| CPU only (load ~17) | 77.7 s | 9.0 s (10.6 -> 7.9) | - | ~3.7 s (sub-quadratic kernel) | - |
| NPU, `attention=auto`, first prompt | 83.1 s | 7.4 s (5.7-10.5) | 0.64 s / 0.20 s | 2.55 s (10 blocks) / 0.15 s (22) | 3.9 s |
| NPU, `attention=auto`, next prompt (warm) | 58.9 s | 6.5 s (5.2-8.0) | 0.63 s / 0.16 s | 2.29 s (10) / 0.14 s (22) | 3.3 s |
| NPU, `attention=npu` (forced, load ~10) | 57.1 s | 5.3 s (4.8-6.3) | 0.56 s / 0.11 s | 2.08 s (30) / 0.00 s (2) | 2.5 s |

The first NPU prompt still pays 5 s of program compile and a 9.4 s CPU shape-discovery pass (now also the
attention timing pass), and run 4 pays a second 6.5 s discovery pass because it is a new node instance; the
per-call numbers exclude both. GEMM host time per call is 0.11-0.35 s against 0.85 s in 3b (0.11-0.13 s in the quieter second
sequence), with the NPU busy 0.53-0.8 s (570-890 GFLOPS) for the same 466 GFLOP. The "rest" column shrinks run by run with the load (it is the
same CPU work every time), so differences of ~1 s between runs are environment, not NPU; the forced-attention row
is faster than the auto row mostly for that reason (its attention total is 2.08 s for 30 blocks vs 2.43 s for 10
blocks + 22 CPU calls, the rest is load). The per-shape gate timings in the attention table above are the fair
comparison: on this CPU kernel the NPU wins the two large self-attention levels and loses the small ones.

**With torch's SDPA on the CPU** (`docker/run.sh env COMFY_EXTRA_ARGS=--use-pytorch-cross-attention
scripts/06-comfy-npu-test.sh 4`, 4 steps, load 7-12, `logs/06-comfy-sdpa-4st.log`, `logs/comfy-server-sdpa-4st.log`):
CPU-only 6.2 s per UNet call (5.9-6.6) against 6.7 s with ComfyUI's default kernel at a similar load; the whole
attention of a call costs 1.0-1.25 s instead of ~3.1 s. The gate then keeps every attention shape on the CPU
(64x64 block: SDPA 149 ms vs NPU 568 ms; 32x32: 16 vs 31 ms), so the NPU contributes the Linear layers only:
5.1-6.0 s per call on the first prompt and 4.7-5.1 s warm (two calls hit by a load spike at 8.0 s excluded), i.e.
the GEMM offload is worth ~1.2 s per call on top of SDPA. Image difference CPU vs NPU after 4 steps: 2.5/255,
91.6 % of pixels within 8/255. For a user of this laptop the order of wins is therefore: start ComfyUI with
`--use-pytorch-cross-attention`, then add the NPU node.

**Where a UNet call goes now** (NPU runs): GEMM 0.65-0.8 s (NPU busy 0.55-0.65 s + 0.1-0.2 s host), attention 1.9-2.4 s (10 or 30
blocks on the NPU, the rest on the CPU), everything else 2.6-3.3 s (convolutions, GroupNorm/LayerNorm, GEGLU,
residual adds, time embedding). The old estimate of "~11 s of convolutions" in 3b was wrong: it hid ~3.5-6 s
(load-dependent) of ComfyUI's sub-quadratic CPU attention, now visible because the node times every attention
call. The NPU attention path itself runs at ~130 GFLOPS end to end (242 GFLOP in 1.8-2.1 s): 32 dispatches per block,
softmax and bf16 conversion on the CPU, 37-69 % padding at d = 40. What is left for the NPU to take over in SD1.5 is
mostly convolutions; what would speed up the attention further is a fused kernel (see `03-next-steps.md`).

**Image check** (same seed 7). CPU-only vs NPU with `attention=auto`: mean absolute pixel difference 13.8/255, median 7, 41.5 % of pixels within
8/255, identical in both sequences (3b, Linear layers only: 13/255, 44 %). With `attention=npu`: 15.1/255, 39.7 %. Same composition and content
in all three; the differences are bf16 GEMMs (and bf16 attention probabilities) compounding over 8 steps. The
4-step shakedown run (`logs/06-comfy-npu-test-4st.log`) shows how the difference grows with steps: 1.6/255 and
97 % within 8/255 after 4 steps. Images: `outputs/comfy/xdna1_cpu_8st_00003_.png`,
`xdna1_npu_8st_00003_.png`, `xdna1_npu_8st_attn-npu_00001_.png`; logs: `logs/06-comfy-npu-test.log`,
`logs/comfy-server-8st-abab.log`.

## 3d. Fused attention: one dispatch per block, softmax on the cores (`npu/xdna1_attention.py`, `npu/kernels/xdna1_attn.cc`, `scripts/11-attn-fused-test.py`)

Follow-up to 3c, where the NPU attention lost to torch's SDPA on the CPU (568 vs 149 ms for the SD1.5
64x64 self-attention block) because it needed 32 dispatches per block, a 64 MB f32 score matrix round
trip per head and a 40 ms CPU softmax per head. The fused kernel runs the whole block, all 16
(batch, head) pairs, as **one dispatch** with the softmax on the compute cores; the score matrix never
leaves a core's 2 KB scratch. `NpuGemm.attention()` uses it whenever a program exists for the head
dimension and the shape fits its tiling, and keeps the 3c per-head path as the fallback; the ComfyUI
gate is unchanged except that it now passes `batch * heads` to `attention_fits()`.

**Design.** Same topology as mlir-aie's `whole_array` matmul (4 columns x 4 rows of cores), with
attention's data flow in place of the GEMM's:

* The (batch, head) pairs are dealt to the columns in contiguous ranges; a column works through its
  pairs one after the other and through each pair's queries in rounds of `4 * bq` rows: the four
  cores of a column each own `bq` query rows of the round and share the pair's K/V stream.
* Per column the shim streams the round's Q rows (split into the four cores through the memtile), the
  pair's K and V in blocks of `bk` keys (interleaved `K_j, V_j` in host memory, re-tiled into 8x8
  tiles by the memtile DMA, broadcast to the four cores, and repeated once per round by the shim DMA's
  queue repeat) and drains the four cores' outputs (joined and un-tiled in the memtile; version 10
  below has the shim gather the heads out of the model's own (b, t, h*d) tensors and scatter the f32
  result back into one, so the host converts nothing but K and V). Every matrix
  lives in the cores as row-major 8x8 tiles, the layout `aie::mmul<8, 8, 8>` reads and writes, so
  no kernel ever reorders data; the DMAs do it for free (version 8 below moves K, V, S and O to 8x4
  blocks, the shape the native macs want, and transposes K once per key block).
* A core runs `xdna1_attn.cc` per round: `O = 0, m = -inf, l = 0`; per key block and row block of
  8 queries `S = Q K^T` (bf16 in, f32 out, K's tiles transposed in registers), the row maxima,
  `P = exp(S - m)` as bf16 (the runtime library's table-based `getExpBf16`; version 5 below replaces it
  with an exponent-field exp2 and the host pre-scales Q by log2 e), the row sums, `O += P V` in f32;
  at the end of the round `out = O / l` as bf16 (version 10: f32, in the output object itself). Q
  arrives pre-scaled by the attention scale (version 10: scaled on the core from the model's f32
  tensor), so the softmax has no multiply. The running max only moves when a row's new maximum beats it by more
  than 5.5 (natural log units), so the rescale of `O` and `l` (an emulated f32 multiply on AIE2) is
  rare; between moves the weights are at most e^5.5 = 245, which the f32 sums and bf16 weights absorb.
  Per-row values are used in the shape of one 8x8 tile (lane `r*8+c` holds row `r`), so they combine
  with the S and O tiles without lane shuffles; the row sums fall out of an mmul of `P` with a tile of
  ones in exactly that form, and the row maxima come from a bf16 transpose of the elementwise max.
  The kernel works one row block ahead: while row block i is exponentiated (vector unit, table
  lookups on the load unit) the scores of row block i+1 are already computed (MAC unit) into a
  second S buffer, and the last row block of a key block computes row block 0 of the *next* key
  block, whose K the core has already acquired (the K/V fifo is four objects deep for that
  look-ahead); the exp itself streams 16 lanes per loop iteration, the form the compiler pipelines.
* `tq`, `tk` and `batch * heads` are dispatch-time parameters (the trip counts reach the cores as
  runtime parameters behind a barrier, as in `whole_array`): one compiled program per head dimension
  serves every token count. `d` (multiple of 8), `bq`, `bk` are compile-time; the defaults are
  (64, 64) for d = 40 and (32, 32) for d = 80 (what fits the 64 KB core memory: 45 / 43 KB of
  buffers with the K/V fifo four deep, a 5-7 KB stack and 5 KB of exp tables; the core's 16 KB of
  program memory is the other limit, the fused step exists once and is called from both entry
  points). A shape must have `tq % (4 bq) == 0`, `tk % bk == 0`, `tk >= 2 bk` and
  `batch * heads % 4 == 0`; `NpuGemm` compiles the programs for d = 40, 80, 160 at construction (4 s
  each from the JIT cache, which now lives in `build/npu-cache-iron/` so it survives the container;
  `NPU_CACHE_HOME`).

**Correctness and speed, standalone** (`docker/run.sh python3 scripts/11-attn-fused-test.py`, random
inputs, error = max |NPU - SDPA| / max |SDPA| with SDPA in f32 on the same inputs; "floor" is the
same number for SDPA run on bf16-rounded inputs, i.e. what the bf16 conversion alone costs; load 1-3,
6 torch threads):

| shape (b, tq, tk, heads, d) | max rel err | bf16 floor | NPU ms | host ms | CPU SDPA ms | 3c per-head NPU ms |
|---|---:|---:|---:|---:|---:|---:|
| (2, 4096, 4096, 8, 40) SD1.5 64x64 | 1.0e-2 | 9.3e-3 | **67** | 5-6 | 144-160 | 568 |
| (2, 1024, 1024, 8, 80) SD1.5 32x32 | 8.3e-3 | 5.3e-3 | **7.9** | 2-3 | 13-18 | 31 |
| (2, 256, 256, 8, 160) SD1.5 16x16 | 6.6e-3 | 6.0e-3 | 4.7 (per-head path) | - | 2.5 | 11 |
| (1, 256, 256, 4, 40) | 7.0e-3 | 5.2e-3 | 0.3 | 0.3 | 0.3 | - |

"NPU ms" is the dispatch (submit to completion) of one block, 640-680 GFLOPS of the 43 / 5.4 GFLOP a
block holds; "host ms" is the bf16 conversion of Q/K/V into the pinned buffers plus the f32 output
copy, so through `NpuGemm.attention()` the block costs 74 / 10 ms wall against 144 / 13 ms for SDPA
in the same process (and 149 / 16 ms measured inside ComfyUI in 3c under load, 113-124 / 12-13 ms
inside ComfyUI at today's load of 2). The
errors sit at the bf16 floor: the kernel adds nothing measurable at 64x64 and 3e-3 at 32x32 (the bf16
`S - m` fed to the exp and the table exp itself). The 16x16 level's program compiles (16x16 tiles,
8 KB stack) but at 1.3 GFLOP per block the CPU's 2.5 ms is not worth chasing; the gate keeps it there.

**Where the time goes** (64x64 block; the kernel's `SKIP_*` switches remove one phase at a time,
`logs/11-attn-phase-timing-v2.log`, `11-attn-phase-timing-v3.log`). Version 2 of the kernel (phases
one after the other) took 72 ms: exp 25 ms, Q K^T ~16, P V ~13, row statistics 7, and 11.6 ms with all
compute skipped. That floor is the K/V broadcast stream: a core's input stream is 32 bits wide, so a
10 KB key block takes 2560 cycles to arrive, which at 4096 key blocks per core per block is exactly
the 11 ms measured; it is hidden under the compute. Version 1 (the AIE2 cubic `exp2_bf16` on f32
input, per-row scalar `reduce_max` / `reduce_add` over 8-lane extracts) took 98 ms (exp 36,
statistics 25); the table exp on bf16 input and the tile-shaped state with the ones-mmul row sums
brought it to 72; the one-row-block look-ahead of version 3 to 67 (32x32: 8.9 to 7.9), less than
the 16 ms of Q K^T because the look-ahead's deeper K/V fifo cost the memory of the transposed-K
buffer (the K tiles are transposed in registers again) and because the exp does not overlap: with
the exp skipped the block takes 41 ms whichever way the exp loop is written (one 64-lane tile per
iteration next to the macs, or 16 lanes per iteration in a loop of its own), i.e. the table exp
costs ~25 cycles per 16 lanes on this core and nothing else runs meanwhile. Things that did not
help: (32, 64) and (64, 32) tiles at d = 40 (80 and 82 ms), a deeper K/V fifo by itself,
`aie::mmul<4, 8, 8>` with 2x2 register blocking in place of the single-row 8x8x8 expansion (73.3 vs
73.4 ms, so the mac loops are not latency bound; both phases run at about the same fraction of peak
as the tuned GEMM of section 3 does on this NPU), splitting a block into two dispatches so the host
conversions overlap the NPU (75 vs 74 ms wall: the shorter dispatches lose what the overlap gains),
and fully unrolling the exp loop or a 20-tile mac loop (Peano's clang frontend crashes, exit 134).
What is left is the exp (~25 of the 67 ms); `03-next-steps.md` has the options.

**Dead ends on the way.** mlir-aie 1.4.4 ships `aie_kernels/linalg/mha.cc`, a flash-attention toolkit
(`partial_softmax`, `matmul_PV`, `rescale_O`), and its newer git tree adds `MHA_CAUSAL=0` and
`flash_attn_prefill.h` (Gemma-4 geometries, head_dim 256/512). The 1.4.4 kernels are causal-only
and, more to the point, their Q K^T product writes 4x8 tiles while the softmax reads row-major rows
and the P V product reads 8x8 tiles: IRON's own MHA runs them on three cores with the memtile
re-laying the data out between them. A kernel with one layout throughout was shorter than adapting
that. The 1.4.4 `aie::mmul<8, 8, 8>` bf16 path, the 8x8 bf16 `aie::transpose`, `aie::inv` and the
`getExpBf16` table (linked by compiling through `aie_kernels/common/lut_kernel.cc`, as the softmax
factory does) all work on npu1. One IRON detail: a joined output fifo's `to_stream` describes one
producer's piece, not the joined object.

**ComfyUI, 4 steps, torch SDPA on the CPU** (`docker/run.sh env COMFY_EXTRA_ARGS=--use-pytorch-cross-attention
scripts/06-comfy-npu-test.sh 4`, SD1.5 Realistic Vision V6 fp32, 512x512, batch 2 (CFG), 6 threads).
The machine was nearly idle this time (load 2-3 against 7-20 for every number in 3b/3c), so the CPU
side is faster than before and only the rows of one table compare.

First sequence (`logs/06-comfy-sdpa-fused-4st.log`, gate measuring its first NPU call cold):

| Run | Prompt total | Per UNet call | GEMMs: NPU busy / host | attention NPU / CPU | gate |
|---|---:|---:|---:|---:|---|
| CPU only (SDPA) | 22.0 s | **3.77 s** (3.74-3.91) | - | ~0.75 s (SDPA) | - |
| NPU, `attention=auto`, first prompt | 42.1 s | 3.33-4.08 s | 0.57 s / 0.12 s | 0.42 s (5 blocks) / 0.12 s (27) | 64x64: 113 vs 102 ms -> NPU; 32x32: 12 vs 22 ms -> CPU |
| NPU, next prompt (warm) | 20.0 s | **3.45 s** (3.37-3.71) | 0.57 s / 0.12 s | 0.41 s (5) / 0.11 s (27) | same |

Image difference CPU vs NPU (seed 7): mean 2.06/255, 94.7 % of pixels within 8/255 (3c, 4 steps,
Linear layers only: 2.5/255, 91.6 %). The gate's cold first call of the 32x32 shape measured 22 ms
(instruction stream, 10 MB of pinned buffers, one-off torch kernels) against the 12 ms SDPA took at
this load, so it kept that level on the CPU. `attention_fits()` now runs the whole fused path twice
on zeros when the node's shape-discovery pass first asks about a shape (standalone: the first real
call then takes 9.1 / 72.8 ms instead of 23.6 / 93.2), so the timed call is a steady-state one:

Final sequence, kernel version 3 with the warm-up (`logs/06-comfy-sdpa-fused-4st-final.log`,
load 1.6-3.4):

| Run | Prompt total | Per UNet call | GEMMs: NPU busy / host | attention NPU / CPU | gate |
|---|---:|---:|---:|---:|---|
| CPU only (SDPA) | 22.0 s | **3.68 s** (3.67-3.73) | - | ~0.7 s (SDPA) | - |
| NPU, `attention=auto`, first prompt | 26.0 s | 3.30-3.45 s | 0.58 s / 0.12 s | 0.37 s (5 blocks) / 0.11 s (27) | 64x64: 104 vs 73 ms -> NPU; 32x32: 12 vs 12 ms -> CPU |
| NPU, next prompt (warm) | **20.0 s** | **3.36 s** (3.27-3.52) | 0.58 s / 0.12 s | 0.37 s (5) / 0.10 s (27) | same |

Image difference CPU vs NPU: 2.62/255 mean, 90.5 % within 8/255 (version 2 of the kernel in the first
sequence: 2.06/255, 94.7 %; the two keep slightly different rounding paths for the running max).
The 64x64 level is on the NPU with a 1.4x margin in situ (73-75 ms per block against 104-124 ms for
SDPA at this load; the ComfyUI figure for SDPA in 3c was 149 ms under load). The 32x32 level is a tie:
in situ the fused call costs 12-13 ms, not the 9-10 ms of the standalone test, because its inputs
arrive cold from the NPU GEMM outputs and the f32 result is a fresh allocation each call, while SDPA
takes 12-13 ms at this load (16 ms in 3c); the gate keeps a tie on the CPU. Under the loads of 3b/3c
the same kernel wins that level too, and either way the per-head path of 3c (31 ms) is history.

Per UNet call the fused attention replaces 5 x 104-113 ms of SDPA with 5 x 74 ms (the 64x64 blocks);
the GEMM offload is unchanged. With SDPA on the CPU an SD1.5 UNet call is ~3.4-3.7 s here, of which
attention was ~0.7 s, so the NPU's attention win is worth ~0.15-0.2 s per call (4-5 %, 3.68 -> 3.36 s
together with the Linear layers' ~0.3 s at this load); the rest of the call (convolutions, norms,
GEGLU) is untouched. Against ComfyUI's default sub-quadratic CPU attention (3c: ~3 s of attention per
call) the same kernel would be worth ~2.5 s per call.

**Faster kernel (versions 5-8): an exp without tables, a layout without shuffles.** The second pass
over the kernel (`logs/11-attn-v5-exp2.log`, `11-attn-v6-pvoverlap.log`, `11-attn-v8-layout.log`,
`11-attn-v8-v9-ab.log`, `11-attn-v8-phase.log`, `11-attn-v8-final-gemm.log`) took the 64x64 block
from 67 to 43 ms and the 32x32 block from 8.0 to 6.3 ms, with the results bit-identical between the
layout versions and the errors unchanged or slightly better (9.9e-3 and 7.9e-3 against floors of
9.3e-3 and 5.3e-3). Three things were tried, two of them paid:

1. *An exponent-field exp2 on the int16 datapath* (replacing `getExpBf16`, 25 of the 67 ms). The host
   pre-scales Q by `scale * log2(e)` so the weights are `2^(S - m)` (the running-max threshold is 8.0 in
   log2 units, weights stay below 256). For bf16 input `y = k + f`, the bf16 bit pattern of `2^y` is
   `128 (k + 127) + round(128 (2^f - 1))`, and `bfloat16_to_int(y, 7)` (which floors whatever the
   rounding mode) already delivers `128 k + floor(128 f)`, the biased exponent plus the linear term
   of the mantissa, in the low 16 bits of each lane (two 16-lane conversions, one `shuffle` to pack
   the low halves). What remains is `r(f) = 2^f - 1 - f` (at most 0.09), a quadratic in the 7
   fraction bits of that same number: `t = a f + b` and `r + 127 = t f + c` as two 32-lane int16
   macs with an `srs` each (Q15 / Q22 fixed point, the +127 bias folded into `c`), then one int16 add
   and a reinterpretation as bf16. Ten vector ops per 32 lanes against ~25 cycles per 16 lanes for
   the two-table lookup, no table in memory (the 4 KB of exp tables are gone), and the fit (done
   bit-exactly in numpy, centred for the floor of the linear term) is within 1 bf16 ulp of `2^y` on
   every bf16 input (median 0.3 ulp; the table's `exp(int) * exp(frac)` was ~1.5). `-DLUT_EXP`
   restores the table version for comparison. Result: 67.7 -> 44.5 ms and 8.0 -> 6.6 ms; with the
   exp skipped the blocks take 40.6 / 6.3 ms, so the exp now costs ~4 ms / 0.3 ms instead of 26 / 1.7.
   The exp loop (one 8x8 tile of scores per iteration, two 32-lane halves) compiles to 33 bundles per
   tile.
2. *P V of row block i-1 in the same loop as the exp of row block i* (a second P buffer, the previous
   key block's V kept acquired one call longer, the loop over pairs of output tiles with the exp
   tiles spread over its iterations): **a dead end**. 7.1 / 49.5 ms against 6.6 / 44.5 for the exp-only
   kernel; the same restructured kernel with the overlap switched off was 6.8 / 51.0, with the exp
   skipped 6.4 / 42.9, so nothing overlapped and the restructuring itself cost. The disassembly
   (`llvm-objdump` of the cached object, `build/attn-dis/`) shows why: the fused loop body was 127
   bundles for 32 bf16 macs plus one exp tile with 46 stack references, and the step function with
   its 14 pointer arguments (the core has 8 pointer registers) spilled 240-370 times. AIE code is
   statically scheduled, so the listing is the cycle count; the compiler spilled instead of
   co-issuing. The variant is not in the tree.
3. *8x4 layouts and a transposed-K buffer* (version 8, from the same disassembly): `aie::mmul<8,8,8>`
   for bf16 on AIE2 is four native 4x8x4 macs plus two shuffles that unzip the B tile's columns, and
   the P V loop spent 95 bundles per 32 macs on them while Q K^T re-transposed every K tile in
   registers for every row block. Now K and V arrive from the memtile as 8x4 tiles (the B-operand
   shape of `aie::mmul<8,8,4>`, two native macs and no shuffle), S, O and the output are 8x4 f32
   blocks (the output un-tiled by the DMA as before), P stays an 8x8 tile (the exp zips two S
   blocks into one on the way out), and K^T is built once per key block into a 5 KB `KT` buffer (one
   `interleave_zip` and two 4x8 transposes per 8x8 region, in the look-ahead's call) instead of once
   per row block. Q K^T runs four accumulator chains, P V two or four; the per-row statistics use
   the 8x4 block form (lane `4r + c` holds row r). Result: 44.5 -> 42.7 ms and 6.6 -> 6.3 ms, less
   than the shuffle count promised: the P V loop went from 95 to 67 bundles per 32 macs, but the
   step is dominated by fixed costs now (see below). A version 9 that merged the two per-key-block
   entry points so the step inlines once (no call glue, ~100 bundles with 39 stack references per
   row block in version 8), ran five P V chains and unrolled the short loops measured 6.5 / 42.5:
   no better, not kept.

Where the time goes now (version 8, `logs/11-attn-v8-phase.log`, 10 iterations, load 1-3):

| phase removed | 32x32 block (2, 1024, 1024, 8, 80) | 64x64 block (2, 4096, 4096, 8, 40) |
|---|---:|---:|
| nothing (full kernel) | **6.3 ms** | **42.7 ms** |
| exp (`-DSKIP_EXP`) | 5.8 (exp = 0.5) | 36.9 (exp = 5.8) |
| P V (`-DSKIP_PV`) | 4.7 (P V = 1.6) | 31.0 (P V = 11.7) |
| all compute (`-DSKIP_ALL`, the K/V stream) | 3.1 | 11.2 |

A row-block step (8 queries x one key block) is ~1.5 us at 32x32 and ~1.3 us at 64x64 for 320
native macs, 4-8 exp tiles and the row statistics; the matmul loops are at roughly half the MAC
issue rate and the rest is the statistics chain (37 explicit nops in the row-sum/running-max
block), the call glue, and per-key-block work (K transpose, lock handshakes) that the shorter key
blocks of the 32x32 tiling amortise over half as many steps. The data-movement floor is untouched
(K and V still share one 32-bit stream per core).

Through `NpuGemm.attention()` with the warm-up (`--gemm --warmup`, `logs/11-attn-v8-final-gemm.log`):
46.5 ms wall for the 64x64 block and 8.5 ms for the 32x32 block (46.2 / 8.6 ms with
the padded-stride input views ComfyUI hands over, `--strided`); the f32 result is now a view of a
pooled pinned buffer instead of a fresh 5-10 MB allocation per call (in situ that allocation cost
~1 ms of page faults).

**ComfyUI, 4 steps, torch SDPA on the CPU, final kernel** (`logs/06-comfy-sdpa-v8-4st.log`, load 2.9-5.8):

| Run | Prompt total | Per UNet call | attention NPU / CPU | gate |
|---|---:|---:|---:|---|
| CPU only (SDPA) | 21.7 s | **3.9 s** | ~0.75 s (SDPA) | - |
| NPU, `attention=auto`, first prompt | 23.5 s | 3.00-3.34 s | 0.30 s (10 blocks) / 0.04 s (22) | 64x64: 104 vs 49 ms -> NPU; 32x32: 11 vs 10 ms -> NPU |
| NPU, next prompt (warm) | **17.5 s** | **3.0 s** | 0.30 s (10) / 0.04 s (22) | same |

Both self-attention levels run on the NPU with SDPA enabled (the version-3 kernel lost the 32x32
level on a tie). With the exp-only kernel (version 5, `logs/06-comfy-sdpa-exp2-4st.log`) the gate
had measured 64x64 at 116 vs 50 ms and 32x32 at 11 vs 11 ms (a hair under), UNet calls at 3.2 s
against 3.9 s CPU-only at that load. Image difference CPU vs NPU: 3.83/255 mean, 84.2 % of pixels within 8/255 (more of the attention is
bf16 now that both levels are on the NPU; the version-3 run with one level had 2.62/255, 90.5 %).
The 32x32 level is where the margin is thin: in situ the fused call costs 10 ms for a 6.3 ms
dispatch, the rest being the host's conversion of Q, K, V into the pinned bf16 buffers from the
NPU GEMMs' strided f32 views and the output's un-permute; `03-next-steps.md` has the options.

**Version 10: the host does no layout work any more (DMA head gather/scatter, f32 in and out).**
The 10 ms in situ for a 6.3 ms dispatch were three permuting f32 -> bf16 copies (Q scaled, K, V
from the GEMMs' padded (b, t, h*d) views into head-major pinned buffers) plus the bf16 -> f32
un-permute of the result. Now:

* *Output.* The core accumulates in its output fifo object itself (f32 8x4 blocks): `attn_round_init`
  zeroes it, `attn_round_finish` divides by the row sums in place, and the separate O buffer and the
  bf16 out object are gone (Q grows to f32, see below, so the core's memory is unchanged: K/V fifo
  4 x 5 KB, Q 10, out 10, KT 5, S 2-4, P 0.5-1, stack 5-7 KB). The memtile un-tiles the 8x4 blocks
  as before, in f32, and the shim's drain scatters each (batch, head) pair's `(tq x d)` block to its
  column in a pooled pinned `(b, tq, h*d)` f32 buffer, returned as a zero-copy torch view like the
  GEMM outputs. No bf16 rounding of the result either: the 64x64 block's error dropped from 9.9e-3
  to 8.8e-3, under the bf16-input floor of 9.3e-3.
* *Q.* The model's own f32 tensor goes to the NPU: the shim gathers the pair's `(tq x d)` block at
  the tensor's row stride (any stride, any element offset: the NPU GEMM's padded output views
  qualify, so inside ComfyUI Q is used in place, no copy, and its cache flush is skipped when
  torch's version counter says nobody wrote the view since it was handed out), the memtile tiles it
  8x8 in f32, and `attn_round_init` multiplies by `scale * log2(e)` (an f32 value, passed as a
  runtime parameter) and rounds to bf16 once, compacting the object in place. One rounding, exact
  scale: the errors are bit-identical to the host-scaled version. The f32 Q stream is 2x the bytes,
  irrelevant next to the repeated K/V stream.
* *K and V.* One contiguous `copy_` each (f32 -> bf16) into a pinned `(b, tk/bk, 2, bk, h*d)` buffer,
  K and V alternating in blocks of `bk` rows, so one shim pattern `[rounds (queue repeat), 2 tk/bk,
  bk, d]` at row stride `h*d` feeds the single K/V stream. The host still pays these two copies
  (0.3 ms each at 32x32, 0.6-1.2 at 64x64); a GEMM that wrote bf16 in this layout would remove them.
* The shim DMA takes three dimensions plus the queue repeat, with 10-bit wraps on the inner two, so
  rows are split into the round's `4 bq` (Q, output) or the block's `bk` (K/V); `d` words and
  `4 bq` rows fit, the outer dimension is unbounded. `h`, the output's `ho` (head-major tensors are
  the same layout with one head per row block), the first pair `p0`, Q's stride and offset and the
  scale are dispatch-time parameters next to `tq`, `tk`, `bh`; the block is issued as up to four
  dispatches over contiguous pair ranges (`NpuGemm.attn_parts`) so the conversion of part i+1 and the
  cache maintenance of part i overlap the NPU.
* torch gotcha found on the way: `torch.mul(x_f32, c, out=t_bf16)` allocates an f32 temporary and
  is 2-3x slower than `t_bf16.copy_(x_f32)` (0.54 vs 0.34 ms for 5 MB, 2.4 vs 0.8 ms from a padded
  view); that one line was the largest host cost of version 8.

Standalone (`logs/11-attn-v10-*.log`, load 1-3; the test's `--pinned` hands q/k/v over as padded
views of the runner's own pinned buffers, which is what ComfyUI's NPU Linear layers produce):

| | 32x32 block (2, 1024, 1024, 8, 80) | 64x64 block (2, 4096, 4096, 8, 40) |
|---|---:|---:|
| raw dispatch (`11-attn-v10-raw.log`) | 6.3 ms, err 8.0e-3 (floor 5.3e-3) | 43.1 ms, err 8.8e-3 (floor 9.3e-3) |
| through `NpuGemm.attention`, warm (`--gemm --warmup`) | 7.8 ms | 46.0 ms |
| same, ComfyUI's padded views (`--strided`) | 7.9 ms | 46.8 ms |
| same, pinned padded views, Q in place (`--pinned`) | 7.9 ms (7.0-7.1 at the quietest, see below) | 45.8 ms |
| version 8 for comparison (`11-attn-v8-final-gemm.log`, `-strided`) | 8.5 / 8.6 ms | 46.5 / 46.2 ms |

Where the remaining host time goes (`build/attn-scratch/attn_profile.py`, one dispatch per block,
per call; the K/V copies and the output invalidate are what the parts pipeline overlaps):

| step | 32x32 | 64x64 |
|---|---:|---:|
| K copy -> bf16 / V copy -> bf16 | 0.29-0.38 / 0.29-0.33 ms | 0.65-1.24 / 0.56-0.60 ms |
| Q copy f32 (only when Q is not a pinned view) | 0.55 ms | 0.78-1.39 ms |
| flushes Q / K-V / output, output invalidate | 0.09-0.15 each | 0.13-0.16 each, 0.17-0.37 |
| submit (XRT run) | 0.09-0.12 ms | 0.10-0.13 ms |
| NPU | 6.1-6.5 ms | 42.8-42.9 ms |
| `NpuGemm.attention` wall, Q in place, best of 5 rounds | **7.0 ms** (1 part 7.5, 2 parts 7.1, 4 parts 7.0) | **44.3-44.8 ms** (1 part 45.6) |
| same, Q copied | 7.3 ms | 44.7-45.2 ms |

The machine was in use during these runs (load 1-4); the run-to-run spread of a 10-iteration mean is
about 0.5 ms at 32x32, which is why the table shows ranges.

**In situ, and the cost nobody had measured: the NPU program switch.** Inside ComfyUI the gate still
measured the 32x32 level at 10 ms and the 64x64 level at 47 ms (`logs/06-comfy-sdpa-v10-4st.log`,
machine at load 2.5-5), although the same calls are 7.0-7.9 / 44-46 ms standalone. The per-shape
breakdown the runner now keeps (`NpuGemm.summary()`, per block, in situ, over the 42 blocks of the two
NPU runs) says where it goes:

| in situ, per block | 32x32 block | 64x64 block |
|---|---:|---:|
| convert K/V + flushes (host, before each part's dispatch) | 1.07 ms | 1.56 ms |
| submit (XRT) | 0.14 ms | 0.18 ms |
| waiting for the NPU | **8.66 ms** | 46.06 ms |
| output invalidate | 0.10 ms | 0.19 ms |
| wall (`attention()` entry to return) | 10.3 ms | 48.5 ms |
| Q used in place (no copy) | 41 of 42 calls | 41 of 42 |

The host side is what the standalone numbers promised; the NPU side is 2-3 ms longer than standalone
for both levels. Cold caches are not it (`attn_profile.py` evicts q/k/v before every call: 6.9-7.2 ms
either way) and DRAM contention is not it (`logs/11-attn-v10-layout-ab.log`: four memory-bandwidth hog
processes cost the dispatch 0.3 ms, the same for version 8's contiguous per-pair buffers as for the
head gathers). What differs in the UNet is that every attention dispatch sits between Linear GEMMs,
which run a *different* NPU program: both designs use all four columns, so the firmware reconfigures
the array on every change of hardware context. Measured directly (`build/attn-scratch/attn_ctxswitch.py`,
`logs/11-attn-v10-ctxswitch.log`): the 32x32 attention call costs 6.8-7.0 ms when attention calls
follow each other and 9.2-9.5 ms when a GEMM ran before it; a 256x640x640 GEMM costs 0.47 ms after
another GEMM and 2.5-2.6 ms after an attention call; the two attention programs (d = 40 and d = 80)
switching between each other pay the same (a small d = 40 block 0.4 -> 2.8 ms, the 32x32 block 6.9
-> 9.3). **About 2.4 ms per switch, on each side**, whichever two programs are involved, so an
attention layer in the UNet pays ~5 ms of reconfiguration (10 NPU attention layers per UNet call:
~50 ms of 3.2 s), and the gate's NPU figure for the 32x32 level cannot go below ~9.5 ms (6.3 dispatch
+ 2.5 switch + ~0.7 host) with attention and GEMMs in separate programs. This is also why the
version-3 and version-8 kernels measured 12-13 and 10 ms in situ against 9-10 and 8.5 standalone.

**ComfyUI, 4 steps, torch SDPA on the CPU, version 10** (`logs/06-comfy-sdpa-v10-4st.log`, load 2.5-5.2,
other work running on the machine throughout):

| Run | Prompt total | Per UNet call | gate |
|---|---:|---:|---|
| CPU only (SDPA) | 24.0 s | - (3.7-4.8 s/it incl. the sampler) | - |
| NPU, `attention=auto`, first prompt | 26.3 s | 4.0, 3.4, 3.4, 3.2 s | 64x64: 112 vs 47 ms -> NPU; 32x32: 13 vs 10 ms -> NPU |
| NPU, next prompt (warm) | **18.6 s** | 3.5, 3.3, 3.0, 3.0 s | same |

Image difference CPU vs NPU: **2.88/255 mean, 88.9 % of pixels within 8/255** (version 8: 3.83 / 84.2 %;
the result is no longer rounded to bf16 on the way out). Host work per 32x32 call is down from
~3.7 ms (version 8, in situ) to ~1.5 ms; the gate's figure moves only from 10 to 10 because 2.4 ms of
it is the program switch above, which only a GEMM-plus-attention program removes (`03-next-steps.md`).


## 4. Benchmark matrix (`scripts/08-llm-bench-matrix.sh`)

`llama-bench`, 3 repetitions, idle-ish machine. 6 threads = the six physical cores; 12 threads
(SMT) is clearly worse on this CPU for llama.cpp, so 6 is the fair baseline. The NPU rows use
ggml-xdna1 (INT8 prefill GEMMs on 4 hardware contexts) with 6 CPU threads.

| Model (Q4_K_M) | Config | pp128 tok/s | pp512 tok/s | tg32 tok/s |
|---|---|---:|---:|---:|
| Qwen2.5-1.5B | CPU 6 threads | 123.9 | 148.7 | 31.3 |
| Qwen2.5-1.5B | CPU 12 threads | 55.5 | 108.0 | 4.3 |
| Qwen2.5-1.5B | **NPU + CPU 6 threads** | **209.5** (1.69x) | **250.1** (1.68x) | 25.6 |
| Qwen2.5-0.5B | CPU 6 threads | 266.3 | 288.2 | 59.1 |
| Qwen2.5-0.5B | CPU 12 threads | 212.9 | 178.5 | 10.9 |
| Qwen2.5-0.5B | NPU + CPU 6 threads | 294.5 (1.11x) | 311.3 (1.08x) | 60.5 |

Reading: the NPU helps prompt processing most where the GEMMs are large (1.5B: +68%); on the
0.5B model the gate declines most shapes as too small to beat the CPU. Decode is unchanged by
design (the plugin never offloads single-row matmuls); the lower tg32 on the 1.5B NPU row is
within thermal/noise variation of back-to-back runs.

Sanity generation (llama-completion, Qwen2.5-1.5B, greedy, plugin active):
"An NPU is a neural network processor designed to accelerate machine learning tasks, and it is
power efficient because it is optimized for specific hardware architectures ..." at 24.8 tok/s decode.

For scale, the Radeon 760M iGPU would be the fastest local option for LLMs on this laptop
(community numbers for the same model class: ~34 tok/s decode on a 3B); the point of the NPU is
not to beat it but to run at a few watts and leave CPU/GPU free.

## 5. The real alternative: the Radeon 760M iGPU (`scripts/12-comfy-igpu-test.sh`)

Everything above compares the NPU with the CPU. The other option on this laptop is the iGPU, so the same
workflow (Realistic Vision V6, 512x512, Euler, CFG 7, seed 7, `scripts/07-comfy-api-run.py`) was run on it
with the existing `~/ComfyUI` install (0.37, torch 2.7.1+rocm6.2.4, flags as in `~/ComfyUI/run-rocm.sh`:
SDPA, `--gpu-only`, `--disable-pinned-memory`, `HSA_OVERRIDE_GFX_VERSION=11.0.0`). The CPU and NPU rows were
re-taken right after, on the same idle machine (load 0.6-5), with torch SDPA on the CPU and the version-10
fused attention. Energy is the RAPL package counter (`intel-rapl:0`), which on this APU rises from 11 W idle
to 33 W while only the iGPU works, so it covers the iGPU as well as the cores; the container cannot read it,
so the CPU/NPU energy comes from a host-side sampler (`build/igpu-scratch/rapl-npu.txt`) integrated over
each prompt (`logs/12-*.log`).

| 4 steps, 512x512 | Prompt time (warm) | Per UNet call | Package energy per image | Avg package power |
|---|---:|---:|---:|---:|
| CPU only (fp32, SDPA) | 22.4 s | 3.8 s | 654 J | 29 W |
| CPU + NPU (Linear + fused attention on the NPU) | 19.5 s | 3.3 s | 554 J | 28 W |
| **iGPU (ROCm, fp16)** | **4.0-4.5 s** | **~0.25 s** | **~150 J** | 33 W |

At 8 steps the iGPU takes 6.9-7.5 s per image (~240 J). Image difference against the CPU at seed 7:
iGPU 2.14/255 mean, 93.9 % of pixels within 8/255 (the NPU path: 2.88, 88.9 %).

Reading: for image generation the iGPU is ~4.5x faster than the CPU+NPU path and uses ~3.5x less energy
per image; the NPU's watts advantage does not show up here, because the CPU still does the convolutions
and keeps the package near its power limit for 20 s. The NPU only makes sense for SD next to the iGPU
(splitting work between them), or for always-on low-power jobs where the iGPU is not wanted.

Note: `~/ComfyUI/venv` was made with Ubuntu's Python 3.12, which the 26.04 upgrade replaced with 3.14, so
`run-rocm.sh` currently starts a python that cannot see the venv's packages. The test script works around it
with a shim venv on uv's CPython 3.12 (`build/igpu-scratch/venv312`) and leaves `~/ComfyUI` alone.
