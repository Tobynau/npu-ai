#!/usr/bin/env python3
"""Quick correctness check of npu/xdna1_gemm.py (a few shapes, incl. unaligned)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
from npu.xdna1_gemm import NpuGemm
g = NpuGemm()
for (M, K, N) in [(512, 512, 512), (256, 320, 320), (100, 100, 100), (1024, 768, 640), (128, 1280, 10240)]:
    x = torch.randn(M, K) * 0.5; W = torch.randn(K, N) / np.sqrt(K)
    w = g.upload_weight(W)
    ref = (x.to(torch.bfloat16).float() @ W.to(torch.bfloat16).float()).numpy()
    y = g.matmul(x, w); t0 = time.perf_counter(); y = g.matmul(x, w); dt = time.perf_counter() - t0
    err = np.abs(y - ref).max() / np.abs(ref).max()
    print(f"{M}x{K}x{N}: rel max err {err:.2e}  {dt*1e3:.2f} ms  {2*M*K*N/dt/1e9:.0f} GFLOPS wall", flush=True)
    assert err < 2e-2, "MISMATCH"
print("OK", g.summary())
