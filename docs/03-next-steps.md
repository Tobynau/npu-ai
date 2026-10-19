# Where to take this next

Ordered by expected payoff for this laptop (XDNA1, 4 columns, ~13-20 GB/s shim bandwidth).

## Make the NPU path native (5 minutes, needs sudo once)
`setup/host-setup.sh` installs Ubuntu's XRT packages and raises the memlock limit; after a
re-login `setup/make-venv.sh` + `source setup/env.sh` run everything without Docker, and the
ComfyUI node can be dropped into the existing `~/ComfyUI/custom_nodes/` (its venv currently has
no torch installed, so that install needs fixing first).

## GEMM module (`npu/xdna1_gemm.py`)
- Done (section 3c of the results): activations converted straight into pinned memory, zero-copy
  f32 outputs from a pooled buffer set, cached per-shape instruction streams, non-blocking runs
  with M-chunks pipelined through sub-buffers. Host share of the SD1.5 Linear shapes went from
  113 ms to 31 ms per UNet pass worth of shapes; what is left is the bf16 conversion itself and the
  ~0.5 ms dispatch floor.
- The dispatch floor is now the dominant cost of small shapes (154-row cross-attention k/v, the
  128-token 8x8 level): batch several small GEMMs of one layer group into one dispatch, or keep
  shapes under ~0.3 GFLOP on the CPU (`min_tokens` in the node does a crude version of this).
- Try the 64x64x32 tile (`m=64`) and `n_cols` sweeps; xdna1-npu-lab reports up to 1.58 TFLOPS
  with runtime loop counts, we see ~1.0-1.3 TFLOPS on the best shapes.
- Weight packing: 16-bit weights cost 2x the DMA of int8; an int8 weight / bf16 activation kernel
  (ggml-xdna1 style, with per-row scales) would double the effective bandwidth.
- Done (section 3d of the results): the fused attention kernel (`npu/xdna1_attention.py` +
  `npu/kernels/xdna1_attn.cc`) runs a whole block (all batch x head pairs) as one dispatch with the
  online softmax on the cores: 43 ms for the SD1.5 64x64 self-attention block against 105-160 ms for
  torch SDPA on the CPU, 6.3 ms against 12-18 ms at 32x32 (section 3d, versions 5-8: an int16
  exponent-field exp2 and 8x4 operand layouts). The per-head path (3c) is now only the fallback for
  shapes the fused tiling does not take.

## Fused attention kernel (`npu/xdna1_attention.py`, `npu/kernels/xdna1_attn.cc`)
State after the third pass (section 3d, versions 5-10): 64x64 block 43 ms (was 67), 32x32 block 6.3 ms
(was 8.0); exp 0.5 / 5.8 ms of that, P V 1-1.6 / 9-12, the K/V stream floor 3.1 / 11.2 (hidden), the
rest Q K^T, row statistics and per-step fixed costs. Version 10 removed the host's layout work: the
shim DMAs gather and scatter the heads, Q goes in as the model's own f32 tensor (in place when it is
an NPU GEMM output) and is scaled and rounded on the core, the result is accumulated in the output
object and returned as a zero-copy f32 view. The 32x32 call is 7.0-7.9 ms wall through `NpuGemm`
(was 8.5-8.6) for a 6.3 ms dispatch. Both SD1.5 self-attention levels run on the NPU in ComfyUI with
SDPA enabled, the 32x32 level still by a thin margin (gate: 10 vs 11-12 ms), and the reason is now
measured: the **NPU program switch**. GEMMs and attention are separate IRON programs that both use all
four columns, so the firmware reconfigures the array whenever a dispatch of the other program follows,
~2.5 ms each way (3d, `logs/11-attn-v10-ctxswitch.log`). Every attention layer in the UNet paid two of
them, and the gate's NPU figure included one (fixed in 3e). In order of expected payoff:
- Done (section 3e): one NPU program for the GEMMs and the attention (`npu/xdna1_program.py`), one per head
  dimension because two attention kernels do not fit a core's 16 KB of program memory; GEMMs run on whichever
  program ran last. A GEMM next to an attention block no longer reconfigures the array (32x32 attention after
  a GEMM 9.3 -> 7.5 ms, GEMM after attention 2.5 -> 0.6 ms); a UNet call has 2 switches left (the 64x64 /
  32x32 level changes) instead of ~20. What it left behind:
  - Q now reaches the cores through the row broadcast, four cores' worth per core: +0.2 ms (32x32) / +0.5 ms
    (64x64) per block. Converting round r+1's strips during round r (a second Qb, +5 KB) would hide it; d = 80
    has the room, d = 40 (~60 of 64 KB used) does not without shrinking something else.
  - Checked (3e, "GEMM K step"): the GEMM's C tile going through the registers every 32 of K costs nothing
    measurable. K steps of 64 (`gemm_kx=2`, two Y and two X objects per `gemm_step`, Y depth 4) measured the same as
    32 within noise, and the earlier 2048x2560x640 regression (5.2 -> 7.9 ms) was a noisy run: the same binary does
    4.4 ms (whole_array 5.2) and the SD1.5 set 64-70 ms (whole_array 102). The option is in the code but off
    (`GEMM_K64`). Long-K GEMMs are bound by the shim streams (~17 GB/s at 2048x2560x640), so the lever is bytes
    per flop: taller C tiles (B is re-sent once per 4 * bq rows of A; bq = 64 at d = 80 would halve that, but
    needs a 20 KB C tile the attention-sized buffers do not have) or int8 weights (below).
  - Program memory is at 14.5 (d = 80, after the `gemm_step` rewrite) and 15.2 (d = 40) of 16 KB, so kernel
    growth needs room first, d = 40 especially; `attn_round_finish` (1.8 KB, mostly the unrolled `bcast4_rows`)
    is the obvious place.
  - The last 2 switches per UNet call go away only with a runtime head dimension in the attention kernel
    (one program for 40 and 80), which costs the compile-time unrolling the kernel's speed depends on.
- What is left of the host side (section 3d's profile): the K and V copies into the interleaved
  bf16 buffer (0.3 ms each at 32x32, 0.6-1.2 at 64x64; the parts pipeline hides most of them), four
  cache-maintenance ioctls of ~0.1 ms each and the XRT submits (~0.1 ms each). A GEMM variant that
  writes bf16 in the `(b, tk/bk, 2, bk, h*d)` layout (the `to_k` / `to_v` Linear layers with their
  outputs interleaved) would remove the copies and halve the flushes; the ioctls only go away with a
  coherent (non-flushing) buffer type, which XRT on amdxdna does not expose.
- Per-step fixed costs. A row-block step is ~1500 cycles at 32x32 for 320 native macs; the matmul
  loops run at about half the MAC issue rate (`build/attn-dis/` has the listings: the P V loop is 67
  bundles per 32 macs, Q K^T similar), the row statistics chain has 37 explicit nops, and the step
  function's 11 pointer arguments spill around the call. Inlining the step (version 9) and five P V
  chains did not help; software-pipelining the statistics against the next matmul loop by hand, or
  two row blocks per step to halve the bookkeeping, are the untried moves. The disassembly, not the
  source, is the thing to optimise against: AIE code is statically scheduled, so bundle counts are
  cycle counts.
- Halve the data-movement floor: split K and V onto the core's two input DMA channels (two streams per
  core instead of one; the shim has only two MM2S channels per column, so Q would have to share one
  of them or be broadcast once per round), and feed Q as one broadcast object per round. Only matters
  once compute drops under ~12 ms / 3.5 ms per block; the 32x32 level is within 2x of it.
- The core's limits are what shape this kernel: 64 KB of data memory with buffers that cannot cross a
  bank, and 16 KB of program memory (the compiled object is 12.6-13.2 KB; a second copy of the step
  overflows it). Memory now: K/V fifo 4 x 5 KB, Q 10 (f32 in, bf16 after `attn_round_init`), out 10
  (the f32 accumulator), KT 5, S 2-4, P 0.5-1, stack 5-7.
- Dead ends, measured: moving the P V product of row block i-1 into the exp loop of row block i (the
  compiler spilled instead of co-issuing, +10 %); the two-table exp (replaced); full unrolls of long
  loops (Peano's frontend crashes at d = 160); `aie::mmul<4,8,8>` 2x2 blocking on 8x8 tiles (no gain,
  same shuffles).
- Cross-attention (tk = 77) needs tk padded to a multiple of 16 with a key mask (-inf scores for the
  pad keys) in the kernel; today those shapes fall back to the per-head path and the gate keeps them
  on the CPU, which is right at 11 ms vs 2 ms per block anyway.
- The 16x16 level (d = 160): the program compiles with a 16x16 tile and an 8 KB core stack, but the
  block is 1.3 GFLOP and the CPU does it in 2 ms; the NPU dispatch floor plus host conversions make it
  a tie at best. (The d = 160 program is not built by default: Peano's frontend crashes on it.)
- (Done in version 10: f32 output accumulated in the output object, no bf16 rounding of the result,
  heads gathered and scattered by the DMAs, Q in place from the GEMM output.)

## ComfyUI node
- Done: attention runs on the NPU behind a measuring gate (`attention` = auto|npu|cpu); with the
  fused kernel (3d) the gate picks the NPU for the 64x64 and 32x32 self-attention levels even
  against torch's SDPA on the CPU, and keeps the small levels and the cross-attention on the CPU.
- Biggest CPU win found on the way: ComfyUI's default sub-quadratic CPU attention costs ~6 s of a
  13.6 s UNet call here, torch's SDPA (`--use-pytorch-cross-attention`) ~1.8 s for the same work.
  Start the server with that flag (the A/B script takes `COMFY_EXTRA_ARGS=--use-pytorch-cross-attention`).
- Run the CPU side on the Radeon 760M instead (ROCm 7.2 torch with `HSA_OVERRIDE_GFX_VERSION=11.0.0`
  for gfx1103) and keep the NPU for GEMMs; Triton-XDNA's HSA runtime can share buffers between
  iGPU and NPU on Linux (zero copy), which removes the host round trip entirely.
- Test with a DiT-style model (almost everything is Linear) such as the small Anima/PixArt family
  once a text encoder that fits the CPU is chosen; with the host overhead gone the Linear share of
  such a model would run at the NPU's 600-1000 GFLOPS.
- Convolutions, norms and elementwise work are the remaining 2.5-3.5 s of a 5-6.5 s SD1.5 UNet call
  (3c); an im2col + GEMM path on the NPU would be the next experiment (3x3 convs at 320-1280
  channels become K = 2880-11520 GEMMs).

## LLMs
- `hawkpoint-npu-llm` (reference/) runs whole decoder layers as single NPU dispatches and reports
  74 tok/s on SmolLM2-135M and 29 tok/s on Qwen2.5-0.5B on Hawk Point; it pins Python 3.12 and a
  specific mlir-aie commit, so it needs its own venv (or a second Docker stage).
- `npu-lab` also has a web UI, document search with MiniLM embeddings on the NPU, and an
  OpenAI-compatible server (`python src/serve.py` / the Rust launcher); `scripts/04` only ran its
  CLI engine.
- `ggml-xdna1`: try `XDNA1_MIN_BATCH`/`XDNA1_FORCE` and larger models (Gemma-2-2B, Qwen3-4B) where
  prefill GEMMs are bigger; the author's data shows 1.4x on Gemma-2-2B.
- Upstream llama.cpp has no XDNA backend; `ggml-xdna1` is the closest thing to one.

## Beyond LLMs/images (where XDNA1 shines: always-on, few watts)
- Whisper encoder on the NPU (`drakosha/whisper-xdna`), YOLO-NAS int8 detection for Frigate at
  18.8 ms/detection (`hnws/xdna1-npu-lab`, needs AMD's Ryzen AI 1.7 Linux tarball), sentence
  embeddings for local semantic search (`npu-lab`).
