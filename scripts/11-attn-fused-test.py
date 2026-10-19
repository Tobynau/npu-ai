#!/usr/bin/env python3
"""Fused NPU attention (npu/xdna1_attention.py) vs torch SDPA: correctness and per-block time.

    docker/run.sh python3 scripts/11-attn-fused-test.py                 # default shapes (SD1.5 512x512 self-attention)
    docker/run.sh python3 scripts/11-attn-fused-test.py --shapes 1x4x256x256x40 2x8x4096x4096x40 --iters 5
    docker/run.sh python3 scripts/11-attn-fused-test.py --bq 32 --bk 64  # tile overrides for tuning
    docker/run.sh python3 scripts/11-attn-fused-test.py --gemm            # go through NpuGemm.attention() (the ComfyUI path)
    docker/run.sh python3 scripts/11-attn-fused-test.py --gemm --warmup --pinned   # ... with the gate's warm-up and q/k/v as
                                                                           # padded views of pinned buffers (NPU GEMM outputs)

Shapes are b x heads x tq x tk x d. For every shape the NPU result is compared with SDPA in f32 on the
same inputs (max |err| / max |ref|, the gate's metric, plus the same number for SDPA run on bf16-rounded
inputs, which is the floor set by the bf16 conversion alone) and timed over --iters dispatches; the CPU
SDPA time is measured in the same process with the same thread count.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from npu.xdna1_attention import BF16, LOG2E, ROWS, NpuAttention  # noqa: E402

import aie.iron as iron  # noqa: E402


def parse_shape(s: str):
    b, h, tq, tk, d = (int(x) for x in s.lower().split("x"))
    return b, h, tq, tk, d


def torch_view(t, dtype, shape):
    """torch view of an IRON tensor's pinned host memory (writes are flushed explicitly below)."""
    arr = t.data.reshape(-1)
    tt = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16) if dtype == torch.bfloat16 else torch.from_numpy(arr)
    return tt.view(*shape)


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float((a.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=["1x4x256x256x40", "1x4x512x256x80", "2x8x4096x4096x40", "2x8x1024x1024x80"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--bq", type=int, default=None)
    ap.add_argument("--bk", type=int, default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--gemm", action="store_true", help="run through NpuGemm.attention() instead of the raw program")
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU SDPA timing")
    ap.add_argument("--kv-depth", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=8.0, help="running-max move threshold (log2 units; nat with -DLUT_EXP)")
    ap.add_argument("--flags", default="", help="extra -D flags for the kernel (experiments), e.g. '-DSKIP_EXP'")
    ap.add_argument("--warmup", action="store_true", help="--gemm: call attention_fits(.., bh) first, as the ComfyUI node does")
    ap.add_argument("--strided", action="store_true",
                    help="hand q/k/v over as views with a padded row stride (as ComfyUI does with the NPU GEMM outputs)")
    ap.add_argument("--pinned", action="store_true",
                    help="--gemm: hand q/k/v over as padded-stride views of the runner's own pinned buffers, exactly what "
                         "ComfyUI's NPU Linear layers produce (Q then goes to the NPU in place, without a copy)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    shapes = [parse_shape(s) for s in args.shapes]

    gemm = None
    progs: dict[int, NpuAttention] = {}
    if args.gemm:
        from npu.xdna1_gemm import NpuGemm
        gemm = NpuGemm(attn_head_dims=tuple(sorted({d for *_, d in shapes})))
    else:
        for d in sorted({d for *_, d in shapes}):
            progs[d] = NpuAttention(d, bq=args.bq, bk=args.bk, kv_depth=args.kv_depth, rescale_threshold=args.threshold,
                                    extra_flags=args.flags)

    print(f"load avg {os.getloadavg()[0]:.1f}, torch threads {torch.get_num_threads()}")
    for b, h, tq, tk, d in shapes:
        torch.manual_seed(0)
        bh = b * h
        q = torch.randn(b, tq, h * d) * 1.5
        k = torch.randn(b, tk, h * d) * 1.5
        v = torch.randn(b, tk, h * d)
        if args.strided or args.pinned:  # rows padded to a multiple of 128 features, like the zero-copy GEMM outputs
            pad = (h * d + 127) // 128 * 128
            q, k, v = (torch.cat([t, torch.zeros(b, t.shape[1], pad - h * d)], dim=2)[:, :, :h * d] for t in (q, k, v))
        pinned = []
        if args.pinned and gemm is not None:   # the same views, but of pinned buffers of the runner's pool
            for i, t in enumerate((q, k, v)):
                buf = gemm.pool.lease(("TEST", i, b, t.shape[1], pad), b * t.shape[1], pad, torch.float32)
                view = buf.t2d.view(b, t.shape[1], pad)[:, :, :h * d]
                view.copy_(t)
                pinned.append(buf)
                if i == 0:
                    q = view
                elif i == 1:
                    k = view
                else:
                    v = view
        scale = d ** -0.5
        q_scale = 1.0 if "-DLUT_EXP" in args.flags else LOG2E   # the kernel's weights are 2^(S - m) (e^(S - m) with the LUT exp)
        qh, kh, vh = (t.view(b, -1, h, d).transpose(1, 2) for t in (q, k, v))
        ref = F.scaled_dot_product_attention(qh, kh, vh).transpose(1, 2).reshape(b, tq, h * d)
        qb, kb, vb = (t.to(torch.bfloat16).float() for t in (qh, kh, vh))
        ref_bf16 = F.scaled_dot_product_attention(qb, kb, vb).transpose(1, 2).reshape(b, tq, h * d)
        floor = rel_err(ref_bf16, ref)
        label = f"({b}, {tq}, {tk}, {h}, {d})"

        if gemm is not None:
            if args.warmup:
                gemm.attention_fits(tq, tk, d, bh)
            t0 = time.perf_counter()
            out = gemm.attention(q, k, v, h)
            first_ms = (time.perf_counter() - t0) * 1e3
            err = rel_err(out, ref)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                out = gemm.attention(q, k, v, h)
            npu_ms = (time.perf_counter() - t0) / args.iters * 1e3
            host_ms = float("nan")
            print(f"{label}: first call {first_ms:.1f} ms ({'after' if args.warmup else 'no'} warm-up)")
            for buf in pinned:
                buf.release()
        else:
            att = progs[d]
            if not att.fits(tq, tk, bh, d):
                print(f"{label}: does not fit the fused program (bq={att.bq}, bk={att.bk}); skipped")
                continue
            bk = att.bk
            # Natural layouts; the NPU's DMAs gather the heads: Q (b, tq, h*d) f32 (scaled and rounded to bf16 by
            # the cores), K/V blocks of bk rows alternating in (b, tk/bk, 2, bk, h*d) bf16, the f32 result
            # (b, tq, h*d) read in place.
            Qt = iron.zeros((bh * tq * d,), dtype=np.float32, device="npu")
            KVt = iron.zeros((bh * tk * d * 2,), dtype=BF16, device="npu")
            Ot = iron.zeros((bh * tq * d,), dtype=np.float32, device="npu")
            qv = torch_view(Qt, torch.float32, (b, tq, h * d))
            kvv = torch_view(KVt, torch.bfloat16, (b, tk // bk, 2, bk, h * d))
            ov = torch_view(Ot, torch.float32, (b, tq, h * d))

            def host_in():
                qv.copy_(q)
                kvv[:, :, 0].copy_(k.view(b, tk // bk, bk, h * d))
                kvv[:, :, 1].copy_(v.view(b, tk // bk, bk, h * d))
                Qt.storage.sync_to_device(0, Qt.nbytes)
                KVt.storage.sync_to_device(0, KVt.nbytes)
                Ot.storage.sync_to_device(0, Ot.nbytes)  # drop dirty lines before the NPU writes

            def host_out():
                Ot.storage.sync_from_device(0, Ot.nbytes)
                return ov   # zero-copy

            t0 = time.perf_counter()
            host_in()
            t_in = time.perf_counter() - t0
            def dispatch():
                att.wait(att.submit(Qt.buffer_object(), KVt.buffer_object(), Ot.buffer_object(), tq, tk, bh, h,
                                    q_scale=scale * q_scale))

            dispatch()
            t0 = time.perf_counter()
            out = host_out()
            t_out = time.perf_counter() - t0
            err = rel_err(out, ref)
            if not np.isfinite(err) or err > 0.1:
                bad = (out.float() - ref).abs().amax(dim=(1,)) if out.ndim == 3 else None
                print(f"{label}: MISMATCH max rel err {err:.3g}; per-batch max abs err {bad}")
                print("  first rows out:", out[0, :2, :6].numpy())
                print("  first rows ref:", ref[0, :2, :6].numpy())
            t0 = time.perf_counter()
            for _ in range(args.iters):
                dispatch()
            npu_ms = (time.perf_counter() - t0) / args.iters * 1e3
            host_ms = (t_in + t_out) * 1e3
            out = out.clone()   # the pinned buffers go away below
            del qv, kvv, ov, Qt, KVt, Ot

        flops = 4.0 * bh * tq * tk * d
        cpu_ms = float("nan")
        if not args.no_cpu:
            F.scaled_dot_product_attention(qh, kh, vh)
            t0 = time.perf_counter()
            n = max(1, min(args.iters, 3))
            for _ in range(n):
                F.scaled_dot_product_attention(qh, kh, vh).transpose(1, 2).reshape(b, tq, h * d)
            cpu_ms = (time.perf_counter() - t0) / n * 1e3
        print(f"{label}: max rel err {err:.2e} (bf16-input floor {floor:.2e}); NPU {npu_ms:.1f} ms/block "
              f"({flops / (npu_ms * 1e6):.0f} GFLOPS) + host {host_ms:.1f} ms; CPU SDPA {cpu_ms:.1f} ms; "
              f"load {os.getloadavg()[0]:.1f}", flush=True)


if __name__ == "__main__":
    main()
