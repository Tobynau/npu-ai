# ComfyUI on the XDNA1 NPU (custom node)

`custom_nodes/comfyui_xdna1_npu` adds three nodes under **NPU/XDNA1**:

| Node | Use |
|---|---|
| **XDNA1 NPU: Offload Linear layers** (MODEL -> MODEL) | put it between the checkpoint loader (or LoRA loader) and the sampler |
| **XDNA1 NPU: Status** (IMAGE -> STRING) | prints GEMM counts, NPU busy time and GFLOPS to the console/UI |
| **XDNA1 NPU: Release** (MODEL -> MODEL) | hands all Linear layers back to the CPU |

How it works: on the first sampler call the node takes over every `torch.nn.Linear` of the
diffusion model with at least `min_features` inputs/outputs (attention q/k/v/out projections,
feed-forward GEMMs; time embeddings are skipped by name), uploads their weights once as bf16 into
NPU buffers, runs one CPU pass to discover the activation shapes (and to time every attention shape
on the CPU), then routes every Linear call with at least `min_tokens` rows to the NPU (bf16 inputs,
f32 accumulate). Activations are converted straight into pinned NPU memory and the f32 results are
handed back as zero-copy views of pooled pinned buffers (a buffer returns to the pool when the model
drops the tensor), so the host no longer copies anything per layer.

The node also installs ComfyUI's `optimized_attention_override`, so the attention QK^T and PV
matmuls can run on the NPU too (per batch x head, softmax on the CPU, pipelined so the NPU
computes the next head's QK^T while the CPU does the current softmax). The `attention` input
chooses the gate: `auto` times each shape once on the CPU and once on the NPU (the NPU result is
checked against the CPU kernel) and keeps the faster side from then on, `npu` forces the NPU where
the shape fits, `cpu` never uses it; shapes with fewer than `attn_min_tokens` query tokens, a mask
or grouped-query layout stay on the CPU and are logged once with the reason. On SD1.5 the gate
keeps most attention on the CPU: with 16 heads x 2 dispatches per block and the softmax on the CPU
the NPU path is slower than torch (see `docs/02-results.md`, section 3c). Convolutions, norms, the
VAE and the text encoder stay on the CPU. Weights are captured after ComfyUI applied its patches, so
LoRAs loaded before the node are included. Any NPU failure raises instead of silently falling back,
and the per-UNet-call log line reports NPU busy time, host time and the attention split.

Run ComfyUI with the node (interactive UI at http://127.0.0.1:8188):

```bash
scripts/comfy-ui.sh
```

Headless A/B (CPU-only image, then the same seed with the NPU; images land in `outputs/comfy/`;
an optional second argument `npu` adds a run with the attention forced to the NPU, and
`COMFY_EXTRA_ARGS=--use-pytorch-cross-attention` switches both runs to torch's SDPA attention on
the CPU, which is several times faster than ComfyUI's default sub-quadratic attention here):

```bash
docker/run.sh scripts/06-comfy-npu-test.sh 8 npu
```

```bash
docker/run.sh env COMFY_EXTRA_ARGS=--use-pytorch-cross-attention scripts/06-comfy-npu-test.sh 4
```

Limits: bf16 weights/activations (f32 accumulate), one NPU hardware context, attention per head
(no batched attention kernel yet, so the ~0.5 ms dispatch floor and the CPU softmax cap the attention
path), no convolutions. Expect a demonstration of hybrid NPU execution rather than a speed-up over a GPU.
