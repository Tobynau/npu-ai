# NPU research notes (AMD XDNA1 / Phoenix on Linux)

Date: 2026-10-18. Machine: AMD Ryzen 5 7640U (Phoenix), Radeon 760M iGPU, 30 GB RAM,
Ubuntu 26.04.1, kernel 7.2.6, Python 3.14.4, ROCm 7.2 installed, Docker available.

## What the NPU is

- PCI `1022:1502` "AMD IPU Device" = first-generation XDNA (AIE2), known to XRT as
  `RyzenAI-npu1`. 4 columns x 4 compute tiles (16 AIE2 cores) + 4 memory tiles.
  Marketed as ~10 TOPS INT8. Second-gen XDNA2 (Ryzen AI 300 "Strix", `npu4`) is ~50 TOPS.
- Kernel driver `amdxdna` is in-tree since Linux 6.14 and is loaded here (kernel 7.2.6).
  Device node: `/dev/accel/accel0` (group `render`; user is a member).
  Firmware: `/lib/firmware/amdnpu/1502_00/`, loaded version `1.5.5.391`.
- User space needs XRT (Xilinx Runtime) + the `xdna` shim (`libxrt_driver_xdna.so`).
  **Ubuntu 26.04 ships this natively** in universe: `libxrt2 libxrt-npu2 libxrt-utils
  libxrt-utils-npu libxrt-dev` (XRT 2.21.75). No PPA or source build needed.
  AMD's PPA `ppa:amd-team/xrt` also has `xrt 2.25.0` for resolute.
- `RLIMIT_MEMLOCK` must be raised: XRT/driver pins a 64 MB device heap; the default
  8 MB hard limit makes every NPU allocation fail silently ("mmap DEV_HEAP: Resource
  temporarily unavailable"). Fix: `/etc/security/limits.d/99-amdxdna.conf` with
  `* soft memlock unlimited` / `* hard memlock unlimited`, then re-login.
  Without root, Docker with `--ulimit memlock=-1:-1 --device /dev/accel/accel0` works.

## Ecosystem status for XDNA1 on Linux (the key finding)

The turnkey stacks target XDNA2 only on Linux:

| Stack | XDNA1 Linux? | Notes |
|---|---|---|
| AMD Ryzen AI Software 1.7.x (ONNX Runtime VitisAI EP) | No | Linux release targets Strix/Krackan |
| FastFlowLM (FLM) | No | Linux support (2026-03) is XDNA2 only |
| Lemonade Server NPU path | No | uses FLM; XDNA1 falls back to GPU/CPU |
| Ollama / LM Studio | No | no NPU backend at all |
| llama.cpp upstream | No | no XDNA backend; feature request stale |

What does work on XDNA1 Linux is the open low-level toolchain plus community projects:

| Project | What | Result on XDNA1 |
|---|---|---|
| Xilinx/mlir-aie (IRON) | Official open compiler + Python API for AIE; pip wheels cp311-cp314; Peano (llvm-aie) compiler wheel | foundation for everything below |
| amd/Triton-XDNA | Triton kernels -> NPU; matmul i8/i16/bf16, elementwise, softmax, layernorm; npu1 runs on Linux only | kernel-level; LLM prefill examples (npu2 focused) |
| Pedroj-64/npu-lab | Full local chat app, Qwen3 0.6B/1.7B with NPU generating every token; int16 matmul ~1.4 TOPS; MiniLM embeddings on NPU | most complete XDNA1 LLM app |
| c8dhjp4tyv-bit/hawkpoint-npu-llm | SmolLM2-135M 74 tok/s, Qwen2.5-0.5B 28.8 tok/s, OpenAI-compatible API; also an Ollama patch | fastest measured XDNA1 decode |
| FireBurn/ggml-xdna1 | llama.cpp backend plugin, talks to DRM directly (no XRT), INT8 GEMM | prefill 1.25-1.41x vs CPU; decode unchanged by design |
| hnws/xdna1-npu-lab | bf16 GEMM (up to 1.58 TFLOPS), Qwen2.5-0.5B demo 2.5 tok/s, YOLO-NAS int8 via Vitis AI | good GEMM kernel reference |
| uppiy47g/anima-xdna-comfyui | ComfyUI custom node: Linear + attention GEMMs on NPU via Triton-XDNA; Windows + npu2 validated; ~36-55 s/image 512x512 4 steps on a 50 TOPS npu2 | the only ComfyUI-on-XDNA integration; porting target |
| mcolsen/amd-npu-stable-diffusion-linux | SD on Linux NPU via Ryzen AI SDK precompiled ONNX | XDNA2 only |

Physics to keep in mind: XDNA1 reaches ~13.5 GB/s through the shim DMA vs ~65 GB/s
DDR5 for the CPU, so single-token LLM decode is bandwidth-bound and the NPU cannot
beat the CPU/iGPU on it. It wins on compute-dense work (prefill, big GEMMs, vision)
at a few watts. On this laptop the Radeon 760M via Vulkan does ~130 tok/s on a 0.5B
model and ~34 tok/s on a 3B model; the NPU's realistic ceiling is tens of tok/s on
sub-1B models.

## Chosen approach

1. Environment: Docker image from `ubuntu:26.04` with the distro XRT packages,
   Python 3.14 venv, mlir-aie 1.4.4 + Peano, Triton-XDNA. Reason: no root on the host
   right now (memlock + apt need sudo). A `setup/host-setup.sh` is provided for the
   native path once sudo is available.
2. Basic test: `xrt-smi examine` sees `RyzenAI-npu1`; run an IRON vector-add and a
   bf16 GEMM from Python and verify against NumPy.
3. LLM: run hawkpoint-npu-llm (SmolLM2/Qwen2.5-0.5B) and/or npu-lab (Qwen3) and
   benchmark against llama.cpp on CPU and the 760M.
4. ComfyUI: build a custom node that routes Linear-layer GEMMs of a loaded diffusion
   model to the NPU GEMM kernel (port of the anima-xdna approach to Linux/npu1),
   first on a small model, then measure.
