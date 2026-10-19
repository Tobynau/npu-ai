# SD1.5 self-attention blocks on the Radeon 760M iGPU with torch SDPA (fp16), for comparison with the NPU's
# fused attention kernel (scripts/11-attn-fused-test.py). Shapes are b x heads x tq x tk x d, as there.
# Run on the HOST with the ROCm torch of ~/ComfyUI through the shim venv that scripts/12 creates:
#   HSA_OVERRIDE_GFX_VERSION=11.0.0 build/igpu-scratch/venv312/bin/python scripts/15-attn-igpu.py
import time, torch
import torch.nn.functional as F

SHAPES = [(2, 8, 4096, 4096, 40), (2, 8, 1024, 1024, 80)]
print("device:", torch.cuda.get_device_name(0))
for b, h, tq, tk, d in SHAPES:
    q = torch.randn(b, h, tq, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, tk, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, tk, d, device="cuda", dtype=torch.float16)
    ref = F.scaled_dot_product_attention(q.float().cpu(), k.float().cpu(), v.float().cpu())
    err = (F.scaled_dot_product_attention(q, k, v).float().cpu() - ref).abs().max().item()
    for _ in range(3):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        t = time.perf_counter(); F.scaled_dot_product_attention(q, k, v); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    print(f"{b}x{h}x{tq}x{tk}x{d}: median {ts[10] * 1e3:.1f} ms  best {ts[0] * 1e3:.1f} ms  max abs err vs f32 {err:.1e}")
