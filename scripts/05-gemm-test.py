#!/usr/bin/env python3
"""Verify npu/xdna1_gemm.py against torch and time the Linear shapes of an SD1.5 UNet at 512x512.
Run inside the container:  docker/run.sh python3 scripts/05-gemm-test.py
"""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
from npu.xdna1_gemm import NpuGemm

torch.manual_seed(0)
# (M tokens, K in_features, N out_features): SD1.5 UNet transformer blocks at 64x64 / 32x32 / 16x16 / 8x8 latent,
# batch 2 (CFG). 768 = CLIP context width for cross-attention k/v.
shapes = [
    (8192, 320, 320), (8192, 320, 2560), (8192, 1280, 320), (154, 768, 320),
    (2048, 640, 640), (2048, 640, 5120), (2048, 2560, 640), (154, 768, 640),
    (512, 1280, 1280), (512, 1280, 10240), (512, 5120, 1280), (154, 768, 1280),
    (128, 1280, 1280), (128, 1280, 10240), (128, 5120, 1280),
    (100, 100, 100),  # deliberately unaligned
]
g = NpuGemm()
print("preparing (compiling) instruction streams for", len(shapes), "shapes ...", flush=True)
t0 = time.perf_counter(); g.prepare(shapes); print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)

print(f"{'M':>6} {'K':>6} {'N':>6} | {'max|err|/max|ref|':>18} {'cos':>8} | {'npu ms':>8} {'wall ms':>8} {'GFLOPS':>8}")
tot_flops = tot_npu = tot_wall = 0.0
for (M, K, N) in shapes:
    x = torch.randn(M, K) * 0.5
    W = torch.randn(K, N) / np.sqrt(K)
    w = g.upload_weight(W)
    ref = (x.to(torch.bfloat16).float() @ W.to(torch.bfloat16).float()).numpy()
    y = g.matmul(x, w)  # warm-up (first call includes buffer allocation)
    s0 = dict(g.stats); t0 = time.perf_counter(); iters = 5
    for _ in range(iters):
        y = g.matmul(x, w)
    wall = (time.perf_counter() - t0) / iters
    npu = (g.stats["npu_seconds"] - s0["npu_seconds"]) / iters
    err = np.abs(y - ref).max() / np.abs(ref).max()
    cos = float(np.dot(y.ravel(), ref.ravel()) / (np.linalg.norm(y) * np.linalg.norm(ref) + 1e-12))
    fl = 2.0 * M * K * N
    tot_flops += fl; tot_npu += npu; tot_wall += wall
    print(f"{M:>6} {K:>6} {N:>6} | {err:>18.2e} {cos:>8.5f} | {npu*1e3:>8.2f} {wall*1e3:>8.2f} {fl/npu/1e9:>8.0f}", flush=True)
    assert err < 2e-2 and cos > 0.999, f"MISMATCH at {(M,K,N)}: err={err} cos={cos}"
print(f"\nall shapes match torch (bf16 inputs, f32 accumulate).")
print(f"one SD1.5 UNet pass worth of attention/FF linears (these shapes once each): "
      f"{tot_flops/1e9:.0f} GFLOP, NPU {tot_npu*1e3:.0f} ms, wall {tot_wall*1e3:.0f} ms -> {tot_flops/tot_wall/1e9:.0f} GFLOPS end-to-end")
print(g.summary())
