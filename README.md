# npu-ai — AMD XDNA1 NPU on Linux: LLM inference and ComfyUI image generation

Target machine: Ryzen 5 7640U (Phoenix, first-gen XDNA NPU `RyzenAI-npu1`), Ubuntu 26.04, kernel 7.2.

The short version of the research (details in [docs/01-research.md](docs/01-research.md)):
the turnkey NPU LLM stacks on Linux (FastFlowLM, Lemonade, Ryzen AI Software) only support the
second-gen XDNA2 NPU. The first-gen NPU is driven on Linux with AMD's open toolchain
(XRT + mlir-aie/IRON + Peano) and a handful of community projects. This repo wires those up,
tests them, and adds a ComfyUI node that runs a diffusion model's Linear layers on the NPU.

## Layout

| Path | What |
|---|---|
| `docker/` | `Dockerfile` (Ubuntu 26.04 + distro XRT 2.21.75 + mlir-aie 1.4.4 + Peano + torch CPU + ComfyUI deps), `build.sh`, `run.sh`; `Dockerfile.hawkpoint` + `build-hawkpoint.sh` (second image: Python 3.12, the mlir-aie/Peano commits hawkpoint-npu-llm pins, pyxrt compiled for 3.12, Go 1.26 + jq + ccache for its Ollama/Colibri builds) |
| `setup/` | native alternative: `host-setup.sh` (apt + memlock, needs sudo once), `make-venv.sh`, `env.sh`; `amdxdna-usage/` = DKMS-patched `amdxdna` driver that reports NPU busy % / memory so GNOME Resources shows real NPU usage |
| `npu/xdna1_gemm.py` | reusable shape-agnostic bf16 GEMM on all 16 NPU cores (padding, chunking, resident weights); `attention()` = the fused kernel below with the per-head path as fallback |
| `npu/xdna1_attention.py`, `npu/kernels/xdna1_attn.cc` | fused scaled-dot-product attention: one NPU dispatch per attention block (all batch x head pairs), flash-attention loop with the softmax on the compute cores |
| `comfyui/custom_nodes/comfyui_xdna1_npu/` | ComfyUI nodes: *XDNA1 NPU: Offload Linear layers*, *Status*, *Release* |
| `scripts/` | numbered tests: `01` smoke, `02` build llama.cpp + ggml-xdna1, `03` llama.cpp NPU prefill, `04` npu-lab Qwen3 on NPU, `05` GEMM verify/bench, `06`/`07` ComfyUI CPU-vs-NPU, `11` fused attention vs torch SDPA, `10` hawkpoint-npu-llm (SmolLM2/Qwen2.5 decoder layers as single NPU dispatches + OpenAI server), `11-llm-hawkpoint-ollama-colibri.sh` hawkpoint-npu-llm's Ollama XDNA1 patch (build, NPU test, CPU-vs-NPU placement bench on qwen2.5:0.5b) and Colibri C/XDNA backend (build + NPU ABI test) |
| `reference/` | upstream/community repos cloned for study (not ours): mlir-aie, Triton-XDNA, xdna1-npu-lab, npu-lab, hawkpoint-npu-llm, ggml-xdna1, anima-xdna-comfyui, ... |
| `third_party/` | llama.cpp (pinned to the commit ggml-xdna1 targets), ComfyUI |
| `models/`, `build/`, `logs/`, `outputs/` | downloaded models, build trees, test logs, generated images (not committed) |
| `docs/` | research notes and results |

## Quick start (Docker, no root needed)

```bash
docker/build.sh                               # once (~5 GB image)
scripts/download-models.sh                    # Qwen3-0.6B, Qwen2.5-0.5B, two GGUFs
docker/run.sh scripts/01-npu-smoke.sh         # xrt-smi, pyxrt, IRON vector add, 4-column matmul
scripts/02-build-llamacpp-xdna1.sh            # host or container; builds llama.cpp + the NPU plugin
docker/run.sh scripts/03-llm-llamacpp-xdna1.sh   # llama.cpp with prefill GEMMs on the NPU
docker/run.sh scripts/04-llm-npu-lab.sh          # Qwen3-0.6B with every token generated on the NPU
docker/run.sh python3 scripts/05-gemm-test.py    # GEMM module vs torch + SD1.5 shapes benchmark
docker/run.sh python3 scripts/11-attn-fused-test.py   # fused NPU attention vs torch SDPA (correctness + per-block time)
docker/run.sh scripts/06-comfy-npu-test.sh 8     # ComfyUI: same image CPU-only vs CPU+NPU
docker/run.sh                                    # interactive shell with the NPU
docker/build-hawkpoint.sh                        # once: second image for hawkpoint-npu-llm (Python 3.12 + its pinned mlir-aie)
NPU_IMAGE=npu-ai-hawkpoint:latest docker/run.sh scripts/10-llm-hawkpoint.sh   # SmolLM2-135M 57 tok/s, Qwen2.5-0.5B 19 tok/s on the NPU
NPU_IMAGE=npu-ai-hawkpoint:latest docker/run.sh scripts/11-llm-hawkpoint-ollama-colibri.sh   # its Ollama v0.33.3 XDNA patch + Colibri C/XDNA backend: build, NPU tests, Ollama CPU-vs-NPU bench
```

`docker/run.sh` passes `/dev/accel/accel0` (+ `/dev/dri`, `/dev/kfd`), raises `memlock` to unlimited
(the one thing the host cannot do without root), mounts this folder at the same path, your
Hugging Face cache at `/hf`, and `~/ComfyUI/models` read-only at `/comfy-models`.

## Why Docker

The in-tree `amdxdna` driver and firmware are already on the host, and Ubuntu 26.04 ships XRT
in its archive, but XRT pins a 64 MB device heap and the default `RLIMIT_MEMLOCK` hard limit is
8 MB; raising it needs `/etc/security/limits.d` (root). `setup/host-setup.sh` does that plus
`apt install` of the XRT packages; after a re-login everything here also runs natively with
`setup/make-venv.sh` + `source setup/env.sh`.

## Results (details in [docs/02-results.md](docs/02-results.md))

| What | Result on this laptop |
|---|---|
| NPU visible + IRON vector add + 4-column matmul | works; 448 GFLOPS i16 matmul |
| Qwen3-0.6B, every token generated on the NPU (npu-lab) | 9 tok/s decode, ~110 tok/s prefill |
| hawkpoint-npu-llm, whole decoder as one NPU dispatch per token + OpenAI-compatible server | SmolLM2-135M **57 tok/s**, Qwen2.5-0.5B **19 tok/s** decode; passes the project's CPU-agreement gates (Hawk Point reference: 75 / 29 tok/s) |
| hawkpoint-npu-llm's Ollama v0.33.3 XDNA1 patch (external GGML backend) and Colibri C/XDNA backend, built in the same image | both build; the 8-expert NPU test and Colibri's NPU ABI test pass; `qwen2.5:0.5b` serves with `100% CPU + NPU` but its dense decode through the backend's padded 8x2048x2048 tiles runs at 0.15-0.5 tok/s vs **119 tok/s** on Ollama's CPU runtime (and 19 tok/s for the native engine above): a placement demo, not a speed-up, on a dense 0.5B model; Colibri's chat needs the 372 GB GLM-5.2 model |
| llama.cpp + ggml-xdna1, Qwen2.5-1.5B Q4_K_M, prompt processing | 149 -> 250 tok/s (1.7x vs 6-thread CPU); decode unchanged by design |
| Our bf16 GEMM module (`npu/xdna1_gemm.py`) | matches torch to 1e-7; 0.5-1.3 TFLOPS NPU time on big shapes; zero-copy host path (host share of an SD1.5 pass of Linear shapes 113 -> 31 ms) |
| ComfyUI SD1.5 512x512, 160 Linear layers + attention matmuls on the NPU | per UNet call 6.7 s CPU-only -> 5.2 s (21% faster, 8 steps: 62 -> 50 s); same image content |
| **Same SD1.5 image on the Radeon 760M iGPU** (ROCm, `scripts/12-comfy-igpu-test.sh`) | 4 steps: **4.0-4.5 s / ~150 J** per image vs 19.5 s / 554 J for CPU+NPU and 22.4 s / 654 J CPU-only (same idle machine): the iGPU is ~4.5x faster and ~3.5x more energy-efficient for image generation |
| Fused attention kernel (`npu/xdna1_attention.py`) | SD1.5 64x64 self-attention block 43 ms on the NPU vs 105-160 ms for torch SDPA on the CPU (3c's per-head path: 568 ms; the first fused version: 67); 32x32 block 6.3 vs 12-18 ms; errors at the bf16 floor (1e-2); exp is a table-free int16 exponent-field exp2, operands in the native 8x4 mac shape; the NPU's DMAs gather the heads from the model's own tensors and scatter the f32 result back (Q in place from the GEMM output, zero-copy output): 7.0-7.9 ms wall per 32x32 block through `NpuGemm` (was 8.5); the ComfyUI gate picks the NPU for both levels with SDPA enabled (UNet call 3.9 -> 3.0 s at 4 steps), where each attention call also pays a ~2.5 ms NPU program switch to and from the GEMM program (measured, 3d) |

Honest summary: on first-gen XDNA the NPU is a modest helper, not a GPU replacement. For image generation the iGPU wins outright (section 5 of the results). The NPU wins on
prompt processing and dense GEMMs at a few watts, loses on bandwidth-bound decode, and for SD1.5
the convolutions and ComfyUI's own CPU attention kernel dominate (`--use-pytorch-cross-attention` is the
biggest single CPU win). `docs/03-next-steps.md` lists what would move the needle.
