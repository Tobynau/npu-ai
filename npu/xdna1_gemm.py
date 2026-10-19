"""bf16 GEMM (and attention matmuls) on the AMD XDNA1 (Phoenix, ``RyzenAI-npu1``) NPU for arbitrary shapes.

C[M, N] (f32) = A[M, K] (bf16) @ B[K, N] (bf16) on all 16 compute cores (4 columns x 4 rows).

Built on mlir-aie's ``whole_array`` matmul design (programming_examples/basic/matrix_multiplication/
whole_array, Apache-2.0 WITH LLVM-exception), whose M, K and N are *dispatch-time* parameters: one
compiled NPU program serves every shape, and each call only gets a small instruction stream generated
on the host. Weights are uploaded once and stay resident in their own NPU buffer objects; shapes that
do not meet the alignment (M % 128, K % 64, N % 128 for the 32x64x32 tile) are zero-padded, and large
N / K are split into chunks.

Host path (the part that used to cost as much as the NPU itself):

* Activations are converted to bf16 by torch straight into the pinned XRT buffer (one pass, no
  intermediate numpy copy); the pinned memory is cacheable, so this runs at memcpy speed.
* Results are returned as zero-copy torch/numpy views of the pinned f32 output buffer. Output buffers
  come from a pool keyed by padded shape; a buffer goes back to the pool when the returned tensor is
  garbage-collected (``weakref.finalize`` on the ndarray torch holds), so nothing is ever copied and
  nothing is reused while the model still reads it.
* Dispatch goes straight to the XRT kernel: the per-shape instruction stream and its buffer object
  are generated once and cached, runs are submitted without waiting (``pyxrt`` run handles) and
  waited on later, so M-chunked GEMMs upload chunk i+1 while chunk i computes, and the attention path
  keeps the NPU busy with the next head's QK^T while the CPU does the softmax of the current one.
* Cache coherence is managed explicitly (sync the written A/B bytes to the device before a run, sync
  the C bytes from the device after it) instead of through the IRON tensor's coherence tracking.

Attention: ``attention()`` runs the whole block (all batch x head pairs) as a few fused NPU dispatches
with the softmax on the compute cores (``npu/xdna1_attention.py``) whenever a compiled program exists
for the head dimension and the shape meets its tiling (``NpuAttention.fits``). The NPU's DMAs gather
the heads out of the model's own layouts: Q goes in as it is (in place when it is one of this runner's
pinned GEMM outputs), K and V are converted to bf16 with one contiguous copy each, the f32 result comes
back as a zero-copy pooled view like the GEMM outputs. Otherwise it falls back to
the per-(batch, head) path (Q K^T and P V on the NPU, softmax on the CPU), which is slower than torch's
SDPA and only there for shapes the fused kernel does not take.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# IRON's JIT cache defaults to ~/.npu/cache, which is lost with the container; keep it in the repo
# (must be set before aie.utils.compile is imported).
os.environ.setdefault("NPU_CACHE_HOME", os.path.join(_ROOT, "build", "npu-cache-iron"))

import aie.iron as iron
import aie.utils as _aie_utils
from aie.iron import str_to_dtype
from aie.iron.device import from_name

_WA_DIR = os.path.join(_ROOT, "reference", "mlir-aie", "programming_examples", "basic", "matrix_multiplication", "whole_array")
if _WA_DIR not in sys.path:
    sys.path.insert(0, _WA_DIR)
from whole_array import whole_array  # noqa: E402

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from npu.xdna1_attention import LOG2E, NpuAttention  # noqa: E402

import pyxrt  # noqa: E402  (system python3-xrt, symlinked into the venv)
from aie.utils.hostruntime.xrtruntime.tensor import XrtTransport  # noqa: E402

BF16 = str_to_dtype("bf16")
ROWS = 4  # compute rows of the array; columns are a parameter
_COMPLETED = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED


def _pad_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def to_bf16(x) -> np.ndarray:
    """float32/float16/bf16 numpy or torch array -> contiguous numpy bf16 (ml_dtypes) array."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").contiguous().to(torch.bfloat16).view(torch.int16).numpy().view(BF16)
    x = np.ascontiguousarray(x)
    return x if x.dtype == BF16 else x.astype(np.float32).astype(BF16)


def _as_torch_2d(x) -> torch.Tensor:
    """numpy/torch (M, K) -> CPU torch tensor (any float dtype, strides kept)."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    x = np.asarray(x)
    if x.dtype == BF16:
        return torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


@dataclass
class NpuWeight:
    """A (K, N) weight resident on the NPU, split along K and N into chunks."""
    K: int
    N: int
    Kp: int
    chunks: list  # [(tensor, k0, k1, Kc, n0, n1, Nc)]

    @property
    def nbytes(self) -> int:
        return sum(c[0].numpy_view().nbytes for c in self.chunks)


class _Buf:
    """One pinned, NPU-visible buffer (an IRON XRTTensor) with cached torch views and sub-buffer handles."""
    __slots__ = ("key", "tensor", "nbytes", "t2d", "in_use", "pool", "base")

    def __init__(self, key, tensor, rows: int, cols: int, torch_dtype, pool):
        self.key, self.tensor, self.pool = key, tensor, pool
        self.nbytes = tensor.nbytes
        data = tensor.data.reshape(-1)  # raw host view: writes here are NOT tracked by IRON (we sync ourselves)
        self.base = data.ctypes.data    # host address, to recognise our own views handed back as inputs
        t = torch.from_numpy(data.view(np.int16)).view(torch.bfloat16) if torch_dtype == torch.bfloat16 else torch.from_numpy(data)
        self.t2d = t.view(rows, cols)
        self.in_use = False

    def bo(self, offset: int = 0, nbytes: int | None = None):
        nbytes = self.nbytes if nbytes is None else nbytes
        return self.tensor.storage.binding_handle(offset, nbytes)

    def to_device(self, offset: int = 0, nbytes: int | None = None) -> None:
        """Flush the host's writes to this range (XRT's sync is a cache flush on both directions).

        This must also precede every dispatch that *writes* a buffer the CPU has written to since (the zero
        fill at allocation, a bias add, an in-place op of the model on a returned view): otherwise the
        CPU's dirty lines are written back over the NPU's result by the FROM_DEVICE flush that follows.
        """
        self.tensor.storage.sync_to_device(offset, self.nbytes if nbytes is None else nbytes)

    def from_device(self, offset: int = 0, nbytes: int | None = None) -> None:
        self.tensor.storage.sync_from_device(offset, self.nbytes if nbytes is None else nbytes)

    def release(self) -> None:
        self.pool.release(self)


class _BufPool:
    """Pool of zero-initialised pinned buffers keyed by (kind, logical shape).

    Buffers are keyed by the *logical* shape they serve so that the zero padding written at allocation
    is never overwritten: a (M, K) activation buffer only ever receives rows < M and columns < K, so
    its padding columns stay zero and never multiply a NaN into the result.
    """

    def __init__(self, stats: dict):
        self._free: dict[tuple, list[_Buf]] = {}
        self._all: list[_Buf] = []
        self._lock = threading.Lock()
        self.stats = stats

    def find(self, ptr: int) -> tuple[_Buf, int] | None:
        """The pooled buffer holding host address ``ptr`` and the byte offset into it, or None."""
        for b in self._all:
            off = ptr - b.base
            if 0 <= off < b.nbytes:
                return b, off
        return None

    def lease(self, key: tuple, rows: int, cols: int, torch_dtype) -> _Buf:
        with self._lock:
            lst = self._free.get(key)
            if lst:
                b = lst.pop()
                b.in_use = True
                return b
        np_dtype = BF16 if torch_dtype == torch.bfloat16 else np.float32
        t0 = time.perf_counter()
        tensor = iron.zeros((rows * cols,), dtype=np_dtype, device="npu")
        tensor.to("npu")  # the zero fill left dirty cache lines; flush them now (see _Buf.to_device)
        b = _Buf(key, tensor, rows, cols, torch_dtype, self)
        with self._lock:
            self._all.append(b)
        self.stats["buffers"] += 1
        self.stats["buffer_bytes"] += b.nbytes
        self.stats["alloc_seconds"] += time.perf_counter() - t0
        b.in_use = True
        return b

    def release(self, b: _Buf) -> None:
        with self._lock:
            if b.in_use:
                b.in_use = False
                self._free.setdefault(b.key, []).append(b)


class NpuGemm:
    """Shape-agnostic bf16 GEMM runner. Default tile (m, k, n) = (32, 64, 32) on 4 columns -> align M%128, K%64, N%128."""

    def __init__(self, m: int = 32, k: int = 64, n: int = 32, n_cols: int = 4,
                 max_n_chunk: int = 4096, max_m_chunk: int = 4096, max_k_chunk: int = 8192, verbose: bool = True,
                 attn_head_dims: tuple[int, ...] = (40, 80), attn_lazy_compile: bool = False):
        """``attn_head_dims``: head dimensions the fused attention program is compiled for up front (SD1.5's
        64x64 and 32x32 levels: 40 / 80; the 16x16 level's d = 160 build crashes Peano and the CPU wins there
        anyway); other multiples of 8 compile on first use when ``attn_lazy_compile`` (the compile then lands
        inside that call, which a timing gate would count against the NPU)."""
        iron.set_current_device(from_name("npu", n_cols=n_cols))
        self.verbose = verbose
        self.attn_lazy_compile = attn_lazy_compile
        self.m, self.k, self.n, self.n_cols = m, k, n, n_cols
        self.align_m, self.align_k, self.align_n = m * ROWS, k, n * n_cols
        self.max_n_chunk = max(self.align_n, max_n_chunk - max_n_chunk % self.align_n)
        self.max_m_chunk = max(self.align_m, max_m_chunk - max_m_chunk % self.align_m)
        self.max_k_chunk = max(self.align_k, max_k_chunk - max_k_chunk % self.align_k)
        self.stats = {"calls": 0, "dispatches": 0, "npu_seconds": 0.0, "total_seconds": 0.0, "flops": 0.0,
                      "compile_seconds": 0.0, "buffers": 0, "buffer_bytes": 0, "alloc_seconds": 0.0,
                      "attn_calls": 0, "attn_pairs": 0, "attn_seconds": 0.0, "attn_flops": 0.0,
                      "attn_fused_calls": 0, "attn_fused_npu_seconds": 0.0, "attn_fused_seconds": 0.0,
                      "attn_fused_q_inplace": 0, "attn_t_convert": 0.0, "attn_t_submit": 0.0, "attn_t_wait": 0.0,
                      "attn_t_sync_out": 0.0, "attn_by_shape": {},   # (b, tq, tk, h, d) -> [blocks, convert, submit, wait, sync, in place, wall]
                      "attn_fused_flops": 0.0, "attn_compile_seconds": 0.0}
        self.pool = _BufPool(self.stats)
        self._insts: dict[tuple[int, int, int], tuple[object, object, int]] = {}
        self._lock = threading.RLock()
        # Compile the program once. The dispatch bridge bounds-checks every call against the host buffer
        # sizes the program was compiled with, so compile against the largest chunk buffers this runner
        # will ever dispatch (the real per-call buffers may be smaller; padding keeps accesses inside them).
        t0 = time.perf_counter()
        A_max, B_max, C_max = (self.max_m_chunk * self.max_k_chunk, self.max_k_chunk * self.max_n_chunk,
                               self.max_m_chunk * self.max_n_chunk)
        self.design = whole_array.specialize(
            A=np.ndarray[(A_max,), np.dtype[BF16]], B=np.ndarray[(B_max,), np.dtype[BF16]],
            C=np.ndarray[(C_max,), np.dtype[np.float32]],
            m=m, k=k, n=n, n_aie_cols=n_cols, b_col_maj=0, c_col_maj=0,
            emulate_bf16_mmul_with_bfp16=False, use_chess=False, scalar=False)
        a0 = iron.zeros((A_max,), dtype=BF16, device="npu")
        b0 = iron.zeros((B_max,), dtype=BF16, device="npu")
        c0 = iron.zeros((C_max,), dtype=np.float32, device="npu")
        self.design(a0, b0, c0, M=self.align_m, K=self.align_k, N=self.align_n)  # compile + warm-up run
        del a0, b0, c0
        kernels = list(getattr(self.design, "_kernel_cache", {}).values())
        if not kernels:
            raise RuntimeError("whole_array design did not cache its compiled kernel; cannot dispatch directly")
        self.kernel = kernels[0]
        self.rt = _aie_utils.DefaultNPURuntime
        self._handle = self.rt.load(self.kernel)  # cached XRT hw context + kernel for our xclbin
        self.stats["compile_seconds"] = time.perf_counter() - t0
        if verbose:
            print(f"[xdna1_gemm] program ready in {self.stats['compile_seconds']:.1f}s "
                  f"(tile {m}x{k}x{n}, {n_cols} columns, align M%{self.align_m} K%{self.align_k} N%{self.align_n})",
                  flush=True)
        self.attn_progs: dict[int, NpuAttention] = {}
        self._attn_warm: set[tuple[int, int, int, int]] = set()
        self.attn_parts = 2         # dispatches per fused attention block (the host work of part i+1 overlaps part i)
        self.attn_first_pairs = 0   # pairs in an extra small leading dispatch (0: none); see _attention_fused
        for d in attn_head_dims:
            self._attn_prog(d, compile_ok=True)

    def _attn_prog(self, d: int, compile_ok: bool | None = None) -> NpuAttention | None:
        """The fused attention program for head dim ``d`` (compiled on demand when allowed), else None."""
        prog = self.attn_progs.get(d)
        if prog is not None or d % 8 or d > 256:
            return prog
        if compile_ok is None:
            compile_ok = self.attn_lazy_compile
        if not compile_ok:
            return None
        t0 = time.perf_counter()
        try:
            prog = NpuAttention(d, n_cols=self.n_cols, verbose=self.verbose)
        except Exception as e:  # a tiling that does not fit the core, an unsupported d: fall back, once
            if self.verbose:
                print(f"[xdna1_gemm] fused attention program for d={d} unavailable: {e}", flush=True)
            prog = None
        self.attn_progs[d] = prog
        self.stats["attn_compile_seconds"] += time.perf_counter() - t0
        return prog

    # -- shapes -------------------------------------------------------------------------------
    def _m_chunks(self, M: int):
        r0 = 0
        while r0 < M:
            r1 = min(M, r0 + self.max_m_chunk)
            yield r0, r1, _pad_up(r1 - r0, self.align_m)
            r0 = r1

    def _n_chunks(self, N: int):
        n0 = 0
        while n0 < N:
            n1 = min(N, n0 + self.max_n_chunk)
            yield n0, n1, _pad_up(n1 - n0, self.align_n)
            n0 = n1

    def _k_chunks(self, K: int):
        k0 = 0
        while k0 < K:
            k1 = min(K, k0 + self.max_k_chunk)
            yield k0, k1, _pad_up(k1 - k0, self.align_k)
            k0 = k1

    def dispatch_shapes(self, M: int, K: int, N: int) -> list[tuple[int, int, int]]:
        return sorted({(Mc, Kc, Nc) for _, _, Mc in self._m_chunks(M) for _, _, Kc in self._k_chunks(K)
                       for _, _, Nc in self._n_chunks(N)})

    def prepare(self, shapes) -> None:
        """Generate the instruction streams and pre-allocate buffers for these logical (M, K, N) shapes."""
        for M, K, N in shapes:
            for Mc, Kc, Nc in self.dispatch_shapes(M, K, N):
                self._insts_for(Mc, Kc, Nc)
            Mp = _pad_up(M, self.align_m)
            self.pool.lease(("A", Mp, K), Mp, _pad_up(K, self.align_k), torch.bfloat16).release()
            for n0, n1, Nc in self._n_chunks(N):
                self.pool.lease(("C", Mp, Nc), Mp, Nc, torch.float32).release()

    # -- weights ------------------------------------------------------------------------------
    def upload_weight(self, w_kn) -> NpuWeight:
        """w_kn: (K, N) array (numpy or torch, any float dtype) -> resident bf16 NPU weight (chunked over K and N)."""
        wb = to_bf16(w_kn)
        K, N = wb.shape
        chunks = []
        for k0, k1, Kc in self._k_chunks(K):
            for n0, n1, Nc in self._n_chunks(N):
                blk = np.zeros((Kc, Nc), dtype=BF16)
                blk[: k1 - k0, : n1 - n0] = wb[k0:k1, n0:n1]
                chunks.append((iron.tensor(blk.reshape(-1), dtype=BF16, device="npu"), k0, k1, Kc, n0, n1, Nc))
        return NpuWeight(K=K, N=N, Kp=_pad_up(K, self.align_k), chunks=chunks)

    # -- dispatch -----------------------------------------------------------------------------
    def _insts_for(self, Mc: int, Kc: int, Nc: int):
        key = (Mc, Kc, Nc)
        ent = self._insts.get(key)
        if ent is None:
            words = self.kernel._generate_dispatch_insts({"M": Mc, "K": Kc, "N": Nc})
            store = XrtTransport(self.rt._device, words.nbytes, pyxrt.bo.cacheable, self._handle.kernel.group_id(1))
            store.host_bytes[:] = words.view(np.uint8).reshape(-1)
            store.to_device(0, words.nbytes)
            ent = (store, store.handle(0, words.nbytes), words.nbytes)
            self._insts[key] = ent
        return ent

    def _submit(self, a_bo, b_bo, c_bo, Mc: int, Kc: int, Nc: int):
        """Start one GEMM dispatch (non-blocking). Returns the pyxrt run handle to pass to _wait."""
        if not getattr(self._handle, "_is_valid", True):
            self._handle = self.rt.load(self.kernel)
        _, insts_bo, nbytes = self._insts_for(Mc, Kc, Nc)
        self.stats["dispatches"] += 1
        return self._handle.kernel(3, insts_bo, nbytes, a_bo, b_bo, c_bo)

    @staticmethod
    def _wait(run) -> None:
        r = run.wait()
        if r != _COMPLETED:
            raise RuntimeError(f"NPU GEMM dispatch failed: kernel returned {r}")

    def _run(self, a_t, b_t, c_t, M: int, K: int, N: int) -> None:
        """Blocking single dispatch on whole IRON tensors (A/B synced to the device first; read C with .numpy())."""
        t0 = time.perf_counter()
        a_t.to("npu")
        b_t.to("npu")
        c_t.to("npu")  # flush the host's dirty lines before the NPU writes C
        self._wait(self._submit(a_t.buffer_object(), b_t.buffer_object(), c_t.buffer_object(), M, K, N))
        self.stats["npu_seconds"] += time.perf_counter() - t0

    # -- zero-copy outputs --------------------------------------------------------------------
    def _output_view(self, c: _Buf, M: int, N: int) -> torch.Tensor:
        """(M, N) f32 torch view of a leased C buffer; the buffer returns to the pool when the view's storage dies."""
        arr = c.tensor.data.reshape(c.t2d.shape)[:]  # a fresh ndarray object torch will hold on to
        f = weakref.finalize(arr, c.release)
        f.atexit = False
        return torch.from_numpy(arr)[:M, :N]

    # -- GEMM ---------------------------------------------------------------------------------
    def matmul_t(self, x, w: NpuWeight, bias=None) -> torch.Tensor:
        """x: (M, K) numpy/torch float -> (M, N) float32 torch tensor = x @ W (+ bias), computed on the NPU.

        The result is a view of pinned NPU-visible memory (no copy); it stays valid for as long as the
        tensor (or any view of it) is alive.
        """
        t_start = time.perf_counter()
        x2 = _as_torch_2d(x)
        M, K = x2.shape
        assert K == w.K, f"K mismatch: x has {K}, weight has {w.K}"
        N = w.N
        Mp, Kp = _pad_up(M, self.align_m), _pad_up(K, self.align_k)
        single = K <= self.max_k_chunk and N <= self.max_n_chunk
        with self._lock:
            a = self.pool.lease(("A", Mp, K), Mp, Kp, torch.bfloat16)
            t_npu0 = None
            if single:
                Np = _pad_up(N, self.align_n)
                b_bo = w.chunks[0][0].buffer_object()
                c = self.pool.lease(("C", Mp, Np), Mp, Np, torch.float32)
                c.to_device(0, Mp * Np * 4)  # drop the CPU's dirty lines (bias add / model writes) before the NPU writes
                runs = []
                for r0, r1, Mc in self._m_chunks(Mp):   # convert chunk i+1 while chunk i computes
                    rows = min(r1, M) - r0
                    if rows > 0:
                        a.t2d[r0:r0 + rows, :K].copy_(x2[r0:r0 + rows])
                    a.to_device(r0 * Kp * 2, Mc * Kp * 2)
                    if t_npu0 is None:
                        t_npu0 = time.perf_counter()
                    runs.append(self._submit(a.bo(r0 * Kp * 2, Mc * Kp * 2), b_bo, c.bo(r0 * Np * 4, Mc * Np * 4), Mc, Kp, Np))
                for run in runs:
                    self._wait(run)
                self.stats["npu_seconds"] += time.perf_counter() - t_npu0
                c.from_device(0, Mp * Np * 4)
                a.release()
                out = self._output_view(c, M, N)
            else:
                # General path (K and/or N chunked): dispatch every (M, K, N) chunk and assemble with copies.
                out = torch.empty((M, N), dtype=torch.float32)
                full_k = K <= self.max_k_chunk
                if full_k:
                    a.t2d[:M, :K].copy_(x2)
                    a.to_device(0, Mp * Kp * 2)
                t_npu0 = time.perf_counter()
                for tensor, k0, k1, Kc, n0, n1, Nc in w.chunks:
                    if full_k:
                        ak = a
                    else:  # each K window needs its own contiguous (Mp, Kc) A slice
                        ak = self.pool.lease(("Ak", Mp, k1 - k0), Mp, Kc, torch.bfloat16)
                        ak.t2d[:M, : k1 - k0].copy_(x2[:, k0:k1])
                        ak.to_device(0, Mp * Kc * 2)
                    c = self.pool.lease(("C", Mp, Nc), Mp, Nc, torch.float32)
                    c.to_device(0, Mp * Nc * 4)
                    hs = [self._submit(ak.bo(r0 * Kc * 2, Mc * Kc * 2), tensor.buffer_object(),
                                       c.bo(r0 * Nc * 4, Mc * Nc * 4), Mc, Kc, Nc) for r0, _r1, Mc in self._m_chunks(Mp)]
                    for run in hs:
                        self._wait(run)
                    c.from_device(0, Mp * Nc * 4)
                    part = c.t2d[:M, : n1 - n0]
                    if k0 == 0:
                        out[:, n0:n1].copy_(part)
                    else:
                        out[:, n0:n1].add_(part)
                    c.release()
                    if ak is not a:
                        ak.release()
                self.stats["npu_seconds"] += time.perf_counter() - t_npu0
                a.release()
            if bias is not None:
                out.add_(bias if isinstance(bias, torch.Tensor) else torch.as_tensor(np.asarray(bias, dtype=np.float32)))
            self.stats["calls"] += 1
            self.stats["flops"] += 2.0 * M * K * N
            self.stats["total_seconds"] += time.perf_counter() - t_start
        return out

    def matmul(self, x, w: NpuWeight) -> np.ndarray:
        """x: (M, K) numpy/torch float -> (M, N) float32 numpy array (zero-copy view, see matmul_t)."""
        return self.matmul_t(x, w).numpy()

    def linear(self, x, w: NpuWeight, bias=None):
        """x: (..., K) numpy/torch -> (..., N) float32 = x @ W (+ bias). torch in -> torch out, numpy in -> numpy out.

        The result is a (strided) view of the pinned output buffer: when N is not a multiple of the N alignment
        the rows are Np apart, so the n-D shape is built with as_strided instead of a copying reshape.
        """
        is_torch = isinstance(x, torch.Tensor)
        xt = x.detach().to("cpu") if is_torch else _as_torch_2d(np.asarray(x).reshape(-1, np.asarray(x).shape[-1]))
        shp = tuple(x.shape)
        y2 = self.matmul_t(xt.reshape(-1, shp[-1]), w, bias)
        lead = shp[:-1]
        if len(lead) == 1:
            y = y2
        else:
            row_stride = y2.stride(0)
            strides, acc = [], row_stride
            for n in reversed(lead):
                strides.append(acc)
                acc *= n
            y = torch.as_strided(y2, (*lead, w.N), (*reversed(strides), 1), y2.storage_offset())
        return y if is_torch else y.numpy()

    # -- attention ----------------------------------------------------------------------------
    def attention_fits(self, tq: int, tk: int, d: int, bh: int | None = None) -> bool:
        """Whether ``attention()`` can run this shape on the NPU (fused kernel, or the per-head path).

        With ``bh`` (= batch * heads) given, the first query for a fused shape also warms that shape up: the
        instruction stream, the pinned buffers and one dummy dispatch, so a caller that times the first real
        call (the ComfyUI node's gate) measures the steady state and not ~10 ms of one-off setup.
        """
        prog = self._attn_prog(d)
        if prog is not None and prog.fits(tq, tk, bh if bh is not None else prog.n_cols, d):
            if bh is not None:
                self.attention_warmup(tq, tk, d, bh)
            return True
        return (_pad_up(tk, self.align_n) <= self.max_n_chunk and _pad_up(tk, self.align_k) <= self.max_k_chunk
                and _pad_up(d, self.align_n) <= self.max_n_chunk and _pad_up(d, self.align_k) <= self.max_k_chunk)

    def attention_warmup(self, tq: int, tk: int, d: int, bh: int) -> None:
        """Run the whole fused path once for a shape on zeros: instruction streams, pooled buffers, the torch
        conversion kernels and the dispatch, so the next real call runs at steady state."""
        key = (tq, tk, d, bh)
        if key in self._attn_warm:
            return
        prog = self._attn_prog(d)
        if prog is None or not prog.fits(tq, tk, bh, d):
            return
        self._attn_warm.add(key)
        t0 = time.perf_counter()
        b = 2 if bh % 2 == 0 else 1   # the real call's batch split is 2 batches of h heads when it can be
        h = bh // b
        z = torch.zeros((b, tq, h * d), dtype=torch.float32)
        zk = torch.zeros((b, tk, h * d), dtype=torch.float32)
        # Once with Q a padded view of one of our pinned buffers, as the NPU Linear layers hand it over (Q then
        # goes in place, with the padded row stride in its instruction stream), once with a plain tensor.
        pad = _pad_up(h * d, self.align_n)
        qp = self.pool.lease(("AQW", b, tq, pad), b * tq, pad, torch.float32)
        try:
            for q in (qp.t2d.view(b, tq, pad)[:, :, :h * d], z):
                self.attention(q, zk, zk, h)
        finally:
            qp.release()
        self.stats["attn_fused_calls"] -= 2
        if self.verbose:
            print(f"[xdna1_gemm] fused attention warm-up for tq={tq} tk={tk} d={d} b*h={bh}: "
                  f"{(time.perf_counter() - t0) * 1e3:.0f} ms", flush=True)

    def attention_fused_fits(self, tq: int, tk: int, d: int, bh: int) -> bool:
        prog = self._attn_prog(d)
        return prog is not None and prog.fits(tq, tk, bh, d)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, scale: float | None = None,
                  skip_reshape: bool = False, skip_output_reshape: bool = False) -> torch.Tensor:
        """softmax(Q K^T * scale) V on the NPU for all (batch, head) pairs of the block (bf16 in, f32 accumulate).

        q/k/v: (b, t, heads*d) float tensors, or (b, heads, t, d) when skip_reshape. Returns (b, tq, heads*d)
        f32 (or (b, heads, tq, d) when skip_output_reshape).

        Fused path (a few dispatches per block, softmax on the cores): used when a program exists for ``d`` and
        the shape fits its tiling, see ``NpuAttention.fits``. Otherwise the per-head path: for every
        (batch, head) pair Q K^T on the NPU, softmax on the CPU in f32 written as bf16 into the next
        matmul's input buffer, P V on the NPU; heads are pipelined (the NPU computes Q K^T of head i+1 and
        P V of head i-1 while the CPU does the softmax of head i).
        """
        t_start = time.perf_counter()
        q, k, v = (t.detach().to("cpu") for t in (q, k, v))
        if skip_reshape:
            b, h, tq, d = q.shape
            tk = k.shape[2]
            Q = lambda bi, hi: q[bi, hi]            # noqa: E731
            Kf = lambda bi, hi: k[bi, hi]           # noqa: E731
            Vf = lambda bi, hi: v[bi, hi]           # noqa: E731
        else:
            b, tq, inner = q.shape
            h = heads
            d = inner // heads
            tk = k.shape[1]
            Q = lambda bi, hi: q[bi, :, hi * d:(hi + 1) * d]    # noqa: E731
            Kf = lambda bi, hi: k[bi, :, hi * d:(hi + 1) * d]   # noqa: E731
            Vf = lambda bi, hi: v[bi, :, hi * d:(hi + 1) * d]   # noqa: E731
        scale = d ** -0.5 if scale is None else float(scale)
        prog = self._attn_prog(d)
        if prog is not None and prog.fits(tq, tk, b * h, d):
            return self._attention_fused(prog, q, k, v, b, h, tq, tk, d, scale, skip_reshape, skip_output_reshape, t_start)
        if not self.attention_fits(tq, tk, d):
            raise ValueError(f"attention shape tq={tq} tk={tk} d={d} exceeds the single-chunk limits of this runner")
        Mp, Kq, Nk = _pad_up(tq, self.align_m), _pad_up(d, self.align_k), _pad_up(tk, self.align_n)
        Kp2, Nd = _pad_up(tk, self.align_k), _pad_up(d, self.align_n)
        pairs = [(bi, hi) for bi in range(b) for hi in range(h)]
        P = len(pairs)
        out = torch.empty((b, h, tq, d), dtype=torch.float32) if skip_output_reshape else torch.empty((b, tq, h * d), dtype=torch.float32)

        def out_slot(bi, hi):
            return out[bi, hi] if skip_output_reshape else out[bi, :, hi * d:(hi + 1) * d]

        with self._lock:
            pool = self.pool
            A1 = [pool.lease(("A", Mp, d), Mp, Kq, torch.bfloat16) for _ in range(2)]          # Q (scaled)
            B1 = [pool.lease(("Bt", d, tk), Kq, Nk, torch.bfloat16) for _ in range(2)]        # K^T
            C1 = [pool.lease(("C", Mp, Nk), Mp, Nk, torch.float32) for _ in range(2)]         # scores
            A2 = [pool.lease(("A", Mp, tk), Mp, Kp2, torch.bfloat16) for _ in range(2)]       # P (softmax, bf16)
            B2 = [pool.lease(("Bv", tk, d), Kp2, Nd, torch.bfloat16) for _ in range(2)]       # V
            C2 = [pool.lease(("C", Mp, Nd), Mp, Nd, torch.float32) for _ in range(2)]         # P V
            for buf in C1 + C2:  # the CPU only reads these below, so one flush per call is enough
                buf.to_device()
            qk_runs: list = [None] * P
            av_runs: list = [None] * P

            def submit_qk(i):
                bi, hi = pairs[i]
                s = i % 2
                A1[s].t2d[:tq, :d].copy_(Q(bi, hi) * scale)
                A1[s].to_device(0, Mp * Kq * 2)
                B1[s].t2d[:d, :tk].copy_(Kf(bi, hi).transpose(0, 1))
                B1[s].to_device(0, Kq * Nk * 2)
                qk_runs[i] = self._submit(A1[s].bo(), B1[s].bo(), C1[s].bo(), Mp, Kq, Nk)

            def submit_av(i):
                bi, hi = pairs[i]
                s = i % 2
                self._wait(qk_runs[i])
                C1[s].from_device(0, Mp * Nk * 4)
                A2[s].t2d[:tq, :tk].copy_(torch.softmax(C1[s].t2d[:tq, :tk], dim=-1))
                A2[s].to_device(0, Mp * Kp2 * 2)
                B2[s].t2d[:tk, :d].copy_(Vf(bi, hi))
                B2[s].to_device(0, Kp2 * Nd * 2)
                av_runs[i] = self._submit(A2[s].bo(), B2[s].bo(), C2[s].bo(), Mp, Kp2, Nd)

            def finish_av(i):
                bi, hi = pairs[i]
                s = i % 2
                self._wait(av_runs[i])
                C2[s].from_device(0, Mp * Nd * 4)
                out_slot(bi, hi).copy_(C2[s].t2d[:tq, :d])

            try:
                submit_qk(0)
                for i in range(P):
                    if i + 1 < P:
                        submit_qk(i + 1)
                    submit_av(i)
                    if i >= 1:
                        finish_av(i - 1)
                finish_av(P - 1)
            finally:
                for lst in (A1, B1, C1, A2, B2, C2):
                    for buf in lst:
                        buf.release()
            self.stats["attn_calls"] += 1
            self.stats["attn_pairs"] += P
            self.stats["attn_flops"] += 4.0 * P * tq * tk * d
            self.stats["attn_seconds"] += time.perf_counter() - t_start
        return out

    def _attention_fused(self, prog: NpuAttention, q, k, v, b, h, tq, tk, d, scale, skip_reshape, skip_output_reshape,
                         t_start) -> torch.Tensor:
        """One or two NPU dispatches for the block, with the heads gathered and scattered by the NPU's DMAs.

        Q goes to the NPU as f32 in its own layout: when it is a view of one of this runner's pinned buffers (the
        NPU GEMM outputs are, padded row stride and all) it is used in place, otherwise it is copied once into a
        pinned f32 buffer; the cores scale it by scale * log2(e) (the kernel's weights are 2^(S - m)) and round
        it to bf16. The host converts K and V with one contiguous copy each into a pinned bf16 buffer with the
        K and V blocks of bk rows alternating, (b, tk/bk, 2, bk, h*d). The shim DMAs pick each (batch, head)
        pair's (rows x d) block out of those at the row stride and scatter the f32 result the same way into a
        pooled pinned (b, tq, h*d) buffer, returned as a zero-copy torch view (back to the pool when the view
        dies; inside ComfyUI a fresh 5-10 MB tensor per call cost ~1 ms of page faults). Head-major inputs
        (skip_reshape) or output (skip_output_reshape) are the same layout with one head per row block of b*h
        "batches", which the dispatch-time h / ho parameters express.

        With two parts (halves of the pair range: whole batches, or two head ranges of a single batch) the
        host's conversion of part 2 and the cache maintenance of part 1's result overlap the NPU's work.
        """
        bh, bk = b * h, prog.bk
        nkb = tk // bk
        # Inputs as (nb, t, hi * d) with hi heads per row (1 for head-major tensors), the output as (nbo, tq, ho * d).
        if skip_reshape:
            hi, nb = 1, bh
            qn, kn, vn = q.reshape(bh, tq, d), k.reshape(bh, tk, d), v.reshape(bh, tk, d)
        else:
            hi, nb = h, b
            qn, kn, vn = q, k, v
        ho = 1 if skip_output_reshape else h
        # Dispatches over contiguous pair ranges, (first pair, pairs): a leading part of attn_first_pairs so the
        # NPU starts after a fraction of the conversion, then attn_parts equal parts of the rest, as far as the
        # block divides into multiples of the column count.
        nc = prog.n_cols
        first = self.attn_first_pairs if self.attn_first_pairs and bh % nc == 0 and bh > 2 * self.attn_first_pairs else 0
        rest = bh - first
        parts = [(0, first)] if first else []
        n_parts = next((k for k in range(self.attn_parts, 0, -1) if rest % k == 0 and (rest // k) % nc == 0), 1)
        n = rest // n_parts
        parts += [(first + i * n, n) for i in range(n_parts)]
        kv_batch, o_batch = 2 * tk * hi * d * 2, tq * ho * d * 4   # bytes per batch row block

        def span(p0, n, hh, nbytes):
            """Byte range of the batch row blocks that pairs [p0, p0 + n) touch in a buffer with hh heads per row."""
            b0, b1 = p0 // hh, (p0 + n - 1) // hh + 1
            return b0 * nbytes, (b1 - b0) * nbytes

        with self._lock:
            pool = self.pool
            if skip_output_reshape:
                outb = pool.lease(("AOUT4", b, h, tq, d), bh * tq, d, torch.float32)
                out = self._output_view(outb, bh * tq, d).view(b, h, tq, d)
            else:
                outb = pool.lease(("AOUT", b, tq, h * d), b * tq, h * d, torch.float32)
                out = self._output_view(outb, b * tq, h * d).view(b, tq, h * d)
            # Q in place when it already lives in one of our pinned buffers as a (nb, tq, >= hi*d) f32 view with
            # rows at a uniform stride (the GEMM outputs), else copied once into a pinned f32 buffer.
            qb, qoff, ldq, q_owned = None, 0, hi * d, False
            if (qn.dtype == torch.float32 and qn.stride(2) == 1 and qn.stride(1) >= hi * d
                    and (nb == 1 or qn.stride(0) == tq * qn.stride(1))):
                found = pool.find(qn.data_ptr())
                if found is not None:
                    buf, off = found
                    last = off + ((nb - 1) * tq * qn.stride(1) + (tq - 1) * qn.stride(1) + hi * d) * 4
                    if off % 4 == 0 and last <= buf.nbytes:
                        qb, qoff, ldq = buf, off // 4, qn.stride(1)
            if qb is None:
                qb, q_owned = pool.lease(("AQF", bh, tq, d), bh * tq, d, torch.float32), True
                qv = qb.t2d.view(nb, tq, hi * d)
            q_batch = tq * ldq * 4
            kvb = pool.lease(("AKV", bh, tk, d), bh * tk * 2, d, torch.bfloat16)
            kvv = kvb.t2d.view(nb, nkb, 2, bk, hi * d)

            def convert_in(part):
                p0, n = part
                # The whole batches of the range in one copy, a partial first / last batch as a head (column) range.
                pieces, whole = [], []
                for bi in range(p0 // hi, (p0 + n - 1) // hi + 1):
                    lo, up = max(p0, bi * hi) - bi * hi, min(p0 + n, (bi + 1) * hi) - bi * hi
                    if lo == 0 and up == hi:
                        whole.append(bi)
                    else:
                        pieces.append((bi, slice(lo * d, up * d)))
                if whole:
                    pieces.append((slice(whole[0], whole[-1] + 1), slice(None)))
                for bs, cs in pieces:
                    if q_owned:
                        qv[bs, :, cs].copy_(qn[bs, :, cs])
                    kk, vv = kn[bs, :, cs], vn[bs, :, cs]
                    kvv[bs, :, 0, :, cs].copy_(kk.reshape(*kk.shape[:-2], nkb, bk, kk.shape[-1]))
                    kvv[bs, :, 1, :, cs].copy_(vv.reshape(*vv.shape[:-2], nkb, bk, vv.shape[-1]))
                # Flush the host's writes. A pinned view of ours that nobody wrote in place since it was handed
                # out (torch's version counter is still 0) holds no dirty lines: the NPU wrote it, the CPU only
                # read. Inference tensors (ComfyUI runs under inference_mode) keep no counter: always flushed.
                if q_owned or qn.is_inference() or qn._version != 0:
                    q0, qn_ = span(p0, n, hi, q_batch)
                    qb.to_device(qoff * 4 + q0, min(qn_, qb.nbytes - qoff * 4 - q0))
                kvb.to_device(*span(p0, n, hi, kv_batch))
                outb.to_device(*span(p0, n, ho, o_batch))  # drop the CPU's dirty lines before the NPU writes

            def submit(part):
                p0, n = part
                return prog.submit(qb.bo(), kvb.bo(), outb.bo(), tq, tk, n, hi, ho, p0, ldq=ldq, qoff=qoff,
                                   q_scale=scale * LOG2E)

            st = self.stats
            sh = st["attn_by_shape"].setdefault((b, tq, tk, h, d), [0, 0.0, 0.0, 0.0, 0.0, 0, 0.0])
            try:
                runs = []
                t0 = None
                for part in parts:
                    t1 = time.perf_counter()
                    convert_in(part)
                    t2 = time.perf_counter()
                    if t0 is None:
                        t0 = t2
                    runs.append(submit(part))
                    t3 = time.perf_counter()
                    sh[1] += t2 - t1
                    sh[2] += t3 - t2
                for part, run in zip(parts, runs):
                    t1 = time.perf_counter()
                    prog.wait(run)
                    t2 = time.perf_counter()
                    t_npu = t2 - t0
                    outb.from_device(*span(part[0], part[1], ho, o_batch))
                    sh[3] += t2 - t1
                    sh[4] += time.perf_counter() - t2
                st["attn_fused_npu_seconds"] += t_npu
            finally:
                if q_owned:
                    qb.release()
                kvb.release()
            st["attn_fused_calls"] += 1
            sh[0] += 1
            sh[5] += not q_owned
            sh[6] += time.perf_counter() - t_start
            self.stats["attn_fused_flops"] += 4.0 * bh * tq * tk * d
            self.stats["attn_fused_seconds"] += time.perf_counter() - t_start
        return out

    # -- reporting ----------------------------------------------------------------------------
    def summary(self) -> str:
        s = self.stats
        gf = s["flops"] / s["npu_seconds"] / 1e9 if s["npu_seconds"] else 0.0
        msg = (f"NpuGemm: {s['calls']} calls, {s['dispatches']} dispatches, NPU {s['npu_seconds']:.3f}s, "
               f"wall {s['total_seconds']:.3f}s, {gf:.0f} GFLOPS (NPU time), program compile {s['compile_seconds']:.1f}s, "
               f"{s['buffers']} pinned buffers ({s['buffer_bytes'] / 2**20:.0f} MiB, {s['alloc_seconds']:.2f}s to allocate)")
        if s["attn_fused_calls"]:
            msg += (f"; fused attention: {s['attn_fused_calls']} blocks, NPU {s['attn_fused_npu_seconds']:.3f}s, "
                    f"wall {s['attn_fused_seconds']:.3f}s, {s['attn_fused_flops'] / max(s['attn_fused_seconds'], 1e-9) / 1e9:.0f} GFLOPS "
                    f"end-to-end ({s['attn_fused_flops'] / max(s['attn_fused_npu_seconds'], 1e-9) / 1e9:.0f} NPU)")
            for key, (n, conv, sub, wait, sync, inpl, wall) in sorted(s["attn_by_shape"].items()):
                msg += (f"; {key}: {n} blocks, per block wall {wall / n * 1e3:.1f} ms = convert+flush {conv / n * 1e3:.2f} "
                        f"+ submit {sub / n * 1e3:.2f} + wait {wait / n * 1e3:.2f} + invalidate {sync / n * 1e3:.2f} + rest, "
                        f"Q in place {inpl}/{n}")
        if s["attn_calls"]:
            msg += (f"; per-head attention: {s['attn_calls']} calls / {s['attn_pairs']} heads, wall {s['attn_seconds']:.3f}s, "
                    f"{s['attn_flops'] / max(s['attn_seconds'], 1e-9) / 1e9:.0f} GFLOPS end-to-end")
        return msg
