# npu-ai: the first-gen AMD NPU on Linux, measured honestly

What can the XDNA1 NPU in a Ryzen 7040 laptop (Phoenix, `RyzenAI-npu1`) actually do on Linux?
AMD's turnkey stacks (Ryzen AI Software, FastFlowLM, Lemonade) only support the second-gen XDNA2 NPU
on Linux, and Ollama, LM Studio and upstream llama.cpp have no NPU backend at all. This repo wires up
AMD's open toolchain (XRT + mlir-aie/IRON + Peano) and the community projects that do work on XDNA1,
adds a few pieces of its own, and benchmarks all of it against the CPU **and the iGPU**.

Tested on: Ryzen 5 7640U (Radeon 760M iGPU, 4x4 AIE2 NPU), 32 GB, Ubuntu 26.04, kernel 7.2, in-tree
`amdxdna` driver, NPU firmware 1.5.5.

## TL;DR

- **LLMs:** the NPU runs small models at a few watts (SmolLM2-135M at 57 tok/s, Qwen2.5-0.5B at
  19 tok/s, Qwen3-0.6B at 9 tok/s) and speeds up llama.cpp prompt processing 1.7x on a 1.5B model.
  Token generation is memory-bandwidth-bound and does not get faster.
- **Image generation (SD1.5, ComfyUI):** the custom node here moves the Linear layers and attention
  onto the NPU and gets ~13-22% over the CPU. The **iGPU is ~4.5x faster still and uses ~3.5x less
  energy per image**. For images, use the iGPU.
- **New bits:** a ComfyUI node for XDNA1 on Linux, a fused attention kernel (one NPU dispatch per
  attention block, softmax on the AIE cores, 2-3x faster than torch SDPA on the CPU), a reusable bf16
  GEMM module, and a driver patch that makes GNOME Resources show real NPU usage.

## Results

| Workload | CPU | NPU | iGPU (760M) |
|---|---:|---:|---:|
| SD1.5 512x512, 4 steps, per image | 22.4 s / 654 J | 19.5 s / 554 J (CPU+NPU) | **4.0-4.5 s / ~150 J** |
| Qwen2.5-1.5B Q4_K_M prompt processing (llama.cpp, pp512) | 149 tok/s | **250 tok/s** (ggml-xdna1) | not measured |
| SmolLM2-135M decode | - | 57 tok/s (hawkpoint-npu-llm) | - |
| Qwen2.5-0.5B decode | 119 tok/s (Ollama) | 19 tok/s (hawkpoint-npu-llm) | - |
| SD1.5 64x64 self-attention block | 105-160 ms (torch SDPA) | **43 ms** (fused kernel) | - |

Energy is the SoC package counter (RAPL), which covers the cores, iGPU and NPU together. Full tables,
methods and the dead ends are in [docs/02-results.md](docs/02-results.md).

## What's in here

| Path | What |
|---|---|
| `npu/xdna1_gemm.py` | shape-agnostic bf16 GEMM on all 16 cores: padding, chunking, resident weights, zero-copy pinned host buffers |
| `npu/xdna1_attention.py`, `npu/kernels/xdna1_attn.cc` | fused flash-style attention for all batch x head pairs in one dispatch |
| `comfyui/custom_nodes/comfyui_xdna1_npu/` | ComfyUI nodes: *Offload Linear layers*, *Status*, *Release*; a timing gate picks NPU or CPU per shape |
| `setup/amdxdna-usage/` | DKMS-patched `amdxdna` driver exposing `npu_busy_percent`, busy time and memory use in sysfs |
| `docker/` | image with Ubuntu's XRT, mlir-aie 1.4.4 and Peano; second image for hawkpoint-npu-llm |
| `scripts/` | numbered, reproducible tests (smoke test, LLM runtimes, GEMM, ComfyUI CPU/NPU/iGPU A/B) |
| `docs/` | research notes, results, next steps |

## Quick start

```bash
docker/build.sh                                  # once, ~5 GB image
docker/run.sh scripts/01-npu-smoke.sh            # NPU visible, IRON vector add, 4-column matmul
docker/run.sh python3 scripts/05-gemm-test.py    # GEMM module vs torch
docker/run.sh python3 scripts/11-attn-fused-test.py   # fused attention vs torch SDPA
docker/run.sh scripts/06-comfy-npu-test.sh 4     # ComfyUI: same image CPU-only vs CPU+NPU
```

Docker is only there because the NPU needs `memlock` raised above the 8 MB default. To run natively,
`setup/host-setup.sh` (sudo once, then reboot) installs Ubuntu's XRT packages and raises the limit.

## Built on

[mlir-aie / IRON](https://github.com/Xilinx/mlir-aie),
[hawkpoint-npu-llm](https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm),
[ggml-xdna1](https://github.com/FireBurn/ggml-xdna1),
[npu-lab](https://github.com/Pedroj-64/npu-lab),
[xdna1-npu-lab](https://github.com/hnws/xdna1-npu-lab),
[anima-xdna-comfyui](https://github.com/uppiy47g/anima-xdna-comfyui),
[ComfyUI](https://github.com/comfyanonymous/ComfyUI). Their code is not included here: clone ComfyUI and llama.cpp into `third_party/` and the others
into `reference/` before running the scripts that use them.

## License

MIT, except `setup/amdxdna-usage/dkms/` (Linux kernel code, GPL-2.0) and `docker/pyxrt/` (XRT, Apache-2.0).
