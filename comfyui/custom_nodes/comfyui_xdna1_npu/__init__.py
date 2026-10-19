"""ComfyUI custom nodes: run the diffusion model's Linear layers and attention matmuls on the AMD XDNA1 NPU (Linux).

Hybrid execution: the NPU does the attention q/k/v/out projections and feed-forward GEMMs of the
UNet/DiT as bf16 GEMMs (f32 accumulate), and (optionally) the attention QK^T and PV matmuls per head,
with the softmax on the CPU. Convolutions, norms, VAE and text encoders stay on the CPU. Nothing
falls back silently: a failed NPU dispatch raises, and every attention shape the gate keeps on the
CPU is logged with its reason.

Nodes
  XDNA1 NPU: Offload Linear layers  (MODEL -> MODEL)   also installs the attention override
  XDNA1 NPU: Status                 (-> STRING)
  XDNA1 NPU: Release                (MODEL -> MODEL)
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()
ATTENTION_MODES = ["auto", "npu", "cpu"]


def _log(msg: str) -> None:
    print(f"[xdna1-npu] {msg}", flush=True)


class AttentionGate:
    """Per-shape decision whether an attention call runs on the NPU or on ComfyUI's CPU kernel.

    ``auto``: every shape is timed once on the CPU (during the shape-discovery pass) and once on the
    NPU (first NPU-enabled call, where the NPU result is also checked against the CPU one); the faster
    side is used from then on. ``npu``: always the NPU when the shape fits. ``cpu``: never the NPU.
    Shapes with a mask, GQA, fewer than ``min_tokens`` query rows or beyond the runner's single-chunk
    limits stay on the CPU in every mode; each such shape is logged once with its reason.
    """

    def __init__(self):
        self.mode = "auto"
        self.min_tokens = 256
        self.cpu_ms: dict[tuple, float] = {}
        self.npu_ms: dict[tuple, float] = {}
        self.decision: dict[tuple, str] = {}
        self.reason: dict[tuple, str] = {}
        self.max_err: dict[tuple, float] = {}
        self.counts = {"npu": 0, "cpu": 0}
        self.npu_seconds = 0.0
        self.cpu_seconds = 0.0
        self.flops_npu = 0.0

    def decide_static(self, key, mask, enable_gqa, fits) -> str | None:
        """Return 'cpu' with a logged reason when the NPU can never take this call, else None."""
        reason = None
        if self.mode == "cpu":
            reason = "attention mode = cpu"
        elif mask is not None:
            reason = "masked attention"
        elif enable_gqa:
            reason = "grouped-query attention"
        elif not fits:
            reason = "exceeds the runner's single-chunk limits"
        elif key[1] < self.min_tokens:
            reason = f"fewer than {self.min_tokens} query tokens"
        if reason is None:
            return None
        if key not in self.reason:
            self.reason[key] = reason
            _log(f"attention {key} stays on the CPU: {reason}")
        return "cpu"

    def table(self) -> str:
        rows = []
        for key in sorted(set(self.cpu_ms) | set(self.npu_ms) | set(self.reason)):
            d = self.decision.get(key, self.reason.get(key, "pending"))
            c = self.cpu_ms.get(key)
            n = self.npu_ms.get(key)
            e = self.max_err.get(key)
            rows.append(f"{key}: cpu {c:.0f} ms" if c is not None else f"{key}: cpu -")
            if n is not None:
                rows[-1] += f", npu {n:.0f} ms"
            if e is not None:
                rows[-1] += f", max rel err {e:.1e}"
            rows[-1] += f" -> {d}"
        return "; ".join(rows)

    def summary(self) -> str:
        return (f"attention: {self.counts['npu']} calls on NPU ({self.npu_seconds:.2f}s), "
                f"{self.counts['cpu']} on CPU ({self.cpu_seconds:.2f}s); gate: {self.table()}")


class XDNA1Runtime:
    """Process-wide NPU GEMM runtime plus the set of Linear modules it has taken over."""

    def __init__(self):
        from npu.xdna1_gemm import NpuGemm  # imports mlir-aie / pyxrt; fails loudly without the NPU stack
        self.gemm = NpuGemm()
        self.weights: dict[int, tuple] = {}      # id(module) -> (NpuWeight, bias torch.Tensor | None, name)
        self.installed: dict[int, tuple] = {}    # id(module) -> (module, original forward)
        self.enabled = False
        self.record_only = False
        self.recorded: set[tuple[int, int, int]] = set()
        self.min_tokens = 64
        self.calls = 0
        self.last_summary = ""
        self.attn = AttentionGate()

    # -- module takeover ------------------------------------------------------------------
    def take_over(self, diffusion_model: torch.nn.Module, min_features: int, skip: tuple[str, ...]) -> int:
        n_new = 0
        for name, mod in diffusion_model.named_modules():
            if not isinstance(mod, torch.nn.Linear) or id(mod) in self.installed:
                continue
            if any(s and s in name for s in skip):
                continue
            if mod.in_features < min_features or mod.out_features < min_features:
                continue
            if mod.weight.device.type != "cpu":
                continue
            W = mod.weight.detach().to(torch.float32).t().contiguous()  # (K=in, N=out)
            bias = mod.bias.detach().to(torch.float32).clone() if mod.bias is not None else None
            self.weights[id(mod)] = (self.gemm.upload_weight(W), bias, name)
            self.installed[id(mod)] = (mod, mod.forward)
            mod.forward = types.MethodType(_npu_linear_forward, mod)
            n_new += 1
        return n_new

    def release(self) -> None:
        for mod, orig in self.installed.values():
            mod.forward = orig
        self.installed.clear()
        self.weights.clear()
        self.enabled = False


def _npu_linear_forward(self: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    rt = _RUNTIME
    entry = rt.weights.get(id(self)) if rt is not None else None
    tokens = x.numel() // x.shape[-1] if x.numel() else 0
    if entry is None or tokens < rt.min_tokens or x.device.type != "cpu":
        return torch.nn.functional.linear(x, self.weight, self.bias)
    w, bias, _name = entry
    if rt.record_only or not rt.enabled:
        if rt.record_only:
            rt.recorded.add((tokens, w.K, w.N))
        return torch.nn.functional.linear(x, self.weight, self.bias)
    y = rt.gemm.linear(x, w, bias)  # f32 torch view of pinned NPU memory, no copy
    return y if y.dtype == x.dtype else y.to(dtype=x.dtype)


def _npu_attention_override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
                            skip_output_reshape=False, **kwargs):
    """transformer_options["optimized_attention_override"]: route QK^T / PV to the NPU when the gate says so.

    ``func`` is ComfyUI's configured attention kernel (the CPU fallback); ``kwargs`` still carries
    ``_inside_attn_wrapper`` so calling ``func`` does not re-enter this override.
    """
    rt = _RUNTIME

    def cpu():
        return func(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape, **kwargs)

    if rt is None or not (rt.enabled or rt.record_only) or q.device.type != "cpu":
        return cpu()
    gate = rt.attn
    if skip_reshape:
        b, h, tq, d = q.shape
        tk = k.shape[2]
    else:
        b, tq, inner = q.shape
        h, d = heads, inner // heads
        tk = k.shape[1]
    key = (b, tq, tk, h, d)
    fits = rt.gemm.attention_fits(tq, tk, d, b * h)  # also warms the fused path up for this shape (first call only)
    if gate.decide_static(key, mask, kwargs.get("enable_gqa", False), fits) == "cpu":
        t0 = time.perf_counter()
        out = cpu()
        gate.cpu_seconds += time.perf_counter() - t0
        gate.counts["cpu"] += 1
        return out

    scale = kwargs.get("scale", None)

    def npu():
        t0 = time.perf_counter()
        y = rt.gemm.attention(q, k, v, heads, scale=scale, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape)
        dt = time.perf_counter() - t0
        gate.npu_ms[key] = min(gate.npu_ms.get(key, 1e30), dt * 1e3)
        gate.npu_seconds += dt
        gate.counts["npu"] += 1
        gate.flops_npu += 4.0 * b * h * tq * tk * d
        return y.to(dtype=q.dtype) if y.dtype != q.dtype else y

    if rt.record_only:  # shape-discovery pass: time the CPU kernel
        t0 = time.perf_counter()
        out = cpu()
        gate.cpu_ms[key] = min(gate.cpu_ms.get(key, 1e30), (time.perf_counter() - t0) * 1e3)
        gate.cpu_seconds += time.perf_counter() - t0
        gate.counts["cpu"] += 1
        return out

    decision = gate.decision.get(key)
    if decision is None:
        # First NPU run of this shape in this mode: time it and, once per shape, check it against the CPU kernel
        # (bf16 inputs/probabilities vs the CPU's f32 path, so ~1e-2 relative is expected; more means a real bug).
        # The first call of a head dimension usually also loads its NPU program (~2.5 ms, once per UNet level, not
        # per call): time a second call then, so the gate compares steady states (dropping the first result
        # first, so its pooled output buffer is reused rather than a new one allocated inside the timing).
        switches = rt.gemm.stats["program_switches"]
        y = npu()
        if rt.gemm.stats["program_switches"] != switches:
            y = None
            y = npu()
        if key not in gate.max_err or key not in gate.cpu_ms:
            t0 = time.perf_counter()
            ref = cpu()
            gate.cpu_ms[key] = min(gate.cpu_ms.get(key, 1e30), (time.perf_counter() - t0) * 1e3)
            err = float((y.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-12))
            gate.max_err[key] = err
            if not np.isfinite(err) or err > 0.1:
                raise RuntimeError(f"NPU attention result for {key} does not match the CPU kernel (max rel err {err:.3g})")
        if gate.mode == "npu":
            decision = "npu"
        else:
            decision = "npu" if gate.npu_ms[key] < gate.cpu_ms[key] else "cpu"
        gate.decision[key] = decision
        _log(f"attention gate {key}: CPU {gate.cpu_ms[key]:.0f} ms vs NPU {gate.npu_ms[key]:.0f} ms, "
             f"max rel err {gate.max_err[key]:.1e} -> {decision.upper()}" + (" (forced)" if gate.mode == "npu" else ""))
        return y
    if decision == "npu":
        return npu()
    t0 = time.perf_counter()
    out = cpu()
    gate.cpu_seconds += time.perf_counter() - t0
    gate.counts["cpu"] += 1
    return out


def _get_runtime() -> XDNA1Runtime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            t0 = time.perf_counter()
            _RUNTIME = XDNA1Runtime()
            _log(f"NPU runtime ready in {time.perf_counter() - t0:.1f}s")
        return _RUNTIME


class XDNA1NpuLinearOffload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "min_tokens": ("INT", {"default": 64, "min": 1, "max": 65536,
                                       "tooltip": "Linear calls with fewer rows than this stay on the CPU"}),
                "min_features": ("INT", {"default": 320, "min": 1, "max": 65536,
                                         "tooltip": "Linear layers narrower than this stay on the CPU"}),
                "skip_names": ("STRING", {"default": "time_embed,label_emb,emb_layers",
                                          "tooltip": "comma-separated substrings of module names to keep on the CPU"}),
            },
            "optional": {
                "attention": (ATTENTION_MODES, {"default": "auto",
                                                "tooltip": "QK^T / PV matmuls: auto = time CPU vs NPU per shape once and keep the faster, "
                                                           "npu = always NPU where the shape fits, cpu = never"}),
                "attn_min_tokens": ("INT", {"default": 256, "min": 1, "max": 65536,
                                            "tooltip": "attention calls with fewer query tokens than this stay on the CPU"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "NPU/XDNA1"
    DESCRIPTION = "Runs the model's Linear layers (attention projections, feed-forward) and attention matmuls on the AMD XDNA1 NPU."

    def apply(self, model, enabled, min_tokens, min_features, skip_names, attention="auto", attn_min_tokens=256):
        if not enabled:
            return (model,)
        rt = _get_runtime()
        rt.min_tokens = int(min_tokens)
        mode = attention if attention in ATTENTION_MODES else "auto"
        if mode != rt.attn.mode or int(attn_min_tokens) != rt.attn.min_tokens:
            rt.attn.decision.clear()   # decisions depend on the mode; the CPU/NPU timings stay valid
            rt.attn.reason.clear()
            _log(f"attention mode {rt.attn.mode} -> {mode} (min tokens {attn_min_tokens}); gate decisions reset")
        rt.attn.mode = mode
        rt.attn.min_tokens = int(attn_min_tokens)
        skip = tuple(s.strip() for s in skip_names.split(",") if s.strip())
        state = {"prepared": False}

        def snapshot():
            g, a = rt.gemm.stats, rt.attn
            return dict(g, attn_npu=a.counts["npu"], attn_cpu=a.counts["cpu"], attn_npu_s=a.npu_seconds,
                        attn_cpu_s=a.cpu_seconds, attn_flops=a.flops_npu)

        def wrapper(apply_model, args):
            x, t, c = args["input"], args["timestep"], args["c"]
            dm = model.model.diffusion_model
            with _RUNTIME_LOCK:
                if not state["prepared"]:
                    # Weights are taken AFTER ModelPatcher applied its patches (LoRA etc.), i.e. now.
                    t0 = time.perf_counter()
                    n = rt.take_over(dm, int(min_features), skip)
                    _log(f"took over {n} new Linear layers ({len(rt.weights)} total) in {time.perf_counter() - t0:.1f}s")
                    # One CPU pass records every (tokens, K, N) so all instruction streams compile before the first
                    # NPU run, and times every attention shape on the CPU for the gate.
                    rt.record_only, rt.enabled = True, False
                    t0 = time.perf_counter()
                    out = apply_model(x, t, **c)
                    rt.record_only = False
                    shapes = sorted(rt.recorded)
                    _log(f"CPU shape-discovery pass {time.perf_counter() - t0:.1f}s; compiling {len(shapes)} GEMM shapes ...")
                    t0 = time.perf_counter()
                    rt.gemm.prepare(shapes)
                    _log(f"compiled in {time.perf_counter() - t0:.1f}s: {shapes}")
                    if rt.attn.cpu_ms:
                        _log("CPU attention times: " + ", ".join(f"{k}: {v:.0f} ms" for k, v in sorted(rt.attn.cpu_ms.items())))
                    state["prepared"] = True
                rt.enabled = True
                s0 = snapshot()
                t0 = time.perf_counter()
                try:
                    out = apply_model(x, t, **c)
                finally:
                    rt.enabled = False
                wall = time.perf_counter() - t0
                s1 = snapshot()
                npu_s = s1["npu_seconds"] - s0["npu_seconds"]
                gemm_wall = s1["total_seconds"] - s0["total_seconds"]
                calls = s1["calls"] - s0["calls"]
                fl = s1["flops"] - s0["flops"]
                an, ac = s1["attn_npu"] - s0["attn_npu"], s1["attn_cpu"] - s0["attn_cpu"]
                an_s, ac_s = s1["attn_npu_s"] - s0["attn_npu_s"], s1["attn_cpu_s"] - s0["attn_cpu_s"]
                afl = s1["attn_flops"] - s0["attn_flops"]
                rt.calls += 1
                rt.last_summary = (
                    f"UNet call {rt.calls}: {wall:.2f}s wall | {calls} Linear GEMMs on NPU: {fl / 1e9:.0f} GFLOP, "
                    f"NPU busy {npu_s:.2f}s ({fl / max(npu_s, 1e-9) / 1e9:.0f} GFLOPS), GEMM wall {gemm_wall:.2f}s "
                    f"(host {gemm_wall - npu_s:.2f}s) | attention: {an} calls on NPU ({an_s:.2f}s, {afl / 1e9:.0f} GFLOP), "
                    f"{ac} on CPU ({ac_s:.2f}s)")
                _log(rt.last_summary)
            return out

        m = model.clone()
        m.set_model_unet_function_wrapper(wrapper)
        to = m.model_options.setdefault("transformer_options", {})
        to["optimized_attention_override"] = _npu_attention_override
        return (m,)


class XDNA1NpuStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trigger": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "status"
    CATEGORY = "NPU/XDNA1"
    OUTPUT_NODE = True

    def status(self, trigger):
        rt = _RUNTIME
        if rt is None:
            s = "XDNA1 NPU runtime not started"
        else:
            s = (rt.gemm.summary() + " | layers on NPU: %d | %s | last: %s"
                 % (len(rt.weights), rt.attn.summary(), rt.last_summary))
        _log(s)
        return {"ui": {"text": [s]}, "result": (s,)}


class XDNA1NpuRelease:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "release"
    CATEGORY = "NPU/XDNA1"

    def release(self, model):
        if _RUNTIME is not None:
            _RUNTIME.release()
            _log("released all Linear layers back to the CPU")
        return (model,)


NODE_CLASS_MAPPINGS = {
    "XDNA1NpuLinearOffload": XDNA1NpuLinearOffload,
    "XDNA1NpuStatus": XDNA1NpuStatus,
    "XDNA1NpuRelease": XDNA1NpuRelease,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "XDNA1NpuLinearOffload": "XDNA1 NPU: Offload Linear layers",
    "XDNA1NpuStatus": "XDNA1 NPU: Status",
    "XDNA1NpuRelease": "XDNA1 NPU: Release",
}
