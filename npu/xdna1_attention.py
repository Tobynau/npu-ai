"""Fused scaled-dot-product attention on the AMD XDNA1 NPU (one dispatch per attention block).

``out[b, :, h*d:(h+1)*d] = softmax(Q_bh K_bh^T * scale) V_bh`` for every (batch, head) pair of a block in one
NPU dispatch, with the softmax on the compute cores (flash attention with an online softmax): the
score matrix never exists outside a core's 2 KB scratch.

Dataflow (same topology as mlir-aie's whole_array matmul: 4 columns x 4 rows of cores):

* The (batch, head) pairs are dealt to the columns in contiguous ranges; a column works through its
  pairs one after the other, and through each pair's queries in rounds of ``4 * bq`` rows: the four
  cores of the column each own ``bq`` query rows of the round and share the pair's K/V stream.
* Per column the shim streams the round's Q rows (split into the four cores through the memtile),
  the pair's K and V in blocks of ``bk`` keys (re-tiled into 8x4 tiles by the memtile and broadcast
  to the four cores; the shim DMA repeats the whole stream once per round) and drains the four
  cores' outputs (joined and un-tiled in the memtile).
* The host buffers keep the model's natural layouts and the shim DMAs gather and scatter the heads:
  Q is the model's own ``(b, tq, h*d)`` f32 tensor (any row stride ``ldq``, any element offset
  ``qoff`` inside its buffer: the NPU GEMM's padded output views qualify, so Q costs the host
  nothing), K/V one ``(b, tk/bk, 2, bk, h*d)`` bf16 buffer (``K_j`` and ``V_j`` blocks of ``bk``
  rows alternating, so one stream carries both) and the output ``(b, tq, h*d)`` f32. A pair's Q,
  K/V and output are each a ``(rows x d)`` block at row stride ``h*d`` (``ldq`` for Q) of those
  buffers, a two-level DMA pattern (three with the row blocks, plus the repeat for K/V), so the
  host converts K and V with one contiguous copy each and reads the result in place.
* A core runs ``npu/kernels/xdna1_attn.cc``: per round it scales its f32 Q tiles by ``scale *
  LOG2E`` and rounds them to bf16 in place (one rounding) and zeroes its output object (the f32
  accumulator), then per key block computes ``S = Q K^T`` (bf16 in, f32 out), updates the running
  row max / row sum, writes the bf16 weights ``exp2(S - m)`` (an exponent-field exp2 on the int16
  datapath) and accumulates ``P V`` in f32; at the end of the round it divides by the row sums in
  place and releases the object to the DMA.

``tq``, ``tk``, the pair count ``bh``, the heads per batch of the inputs ``h`` and of the output
``ho`` (``h = 1`` reads a head-major ``(b*h, t, d)`` tensor, likewise ``ho``), the first pair
``p0``, Q's row stride ``ldq`` and element offset ``qoff`` and the Q scale (f32 bits) are
dispatch-time parameters (one compiled program per head dimension serves every token count);
``d`` (head dim, multiple of 8), ``bq`` and ``bk`` are compile-time.

Requirements for a shape to run here: ``d % 8 == 0``, ``tq % (4 * bq) == 0``, ``tk % bk == 0``,
``b * h % 4 == 0`` (``fits()`` says so); ``NpuGemm.attention`` falls back to its per-head path
otherwise.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# IRON's JIT cache defaults to ~/.npu/cache, which is lost with the container; keep it in the repo.
os.environ.setdefault("NPU_CACHE_HOME", os.path.join(_ROOT, "build", "npu-cache-iron"))

import aie.iron as iron  # noqa: E402
import aie.utils as _aie_utils  # noqa: E402
from aie.helpers.taplib import TensorAccessPattern  # noqa: E402
from aie.iron import (  # noqa: E402
    Buffer,
    CompileTime,
    DispatchTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,
    require,
    str_to_dtype,
)
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import from_name  # noqa: E402
from aie.utils import config as _aie_config  # noqa: E402

import pyxrt  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XrtTransport  # noqa: E402

BF16 = str_to_dtype("bf16")
ROWS = 4
LOG2E = 1.4426950408889634
_KERNEL_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "xdna1_attn.cc")
_COMPLETED = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED


def _f32_bits(x: float) -> int:
    """The bit pattern of float32(x) as a Python int in the int32 range (how the kernel receives a float)."""
    return int(np.array([x], dtype=np.float32).view(np.int32)[0])


@iron.jit
def sd_attention(
    Q: In,
    KV: In,
    O: Out,
    *,
    tq: DispatchTime[np.int32],
    tk: DispatchTime[np.int32],
    bh: DispatchTime[np.int32],
    h: DispatchTime[np.int32],
    ho: DispatchTime[np.int32],
    p0: DispatchTime[np.int32],
    ldq: DispatchTime[np.int32],
    qoff: DispatchTime[np.int32],
    qscale: DispatchTime[np.int32],
    hd: CompileTime[int],
    bq: CompileTime[int],
    bk: CompileTime[int],
    n_cols: CompileTime[int] = 4,
    kv_depth: CompileTime[int] = 4,
    q_depth: CompileTime[int] = 1,
    l3_depth: CompileTime[int] = 2,
    rescale_threshold: CompileTime[float] = 8.0,
    extra_flags: CompileTime[str] = "",
    stack_size: CompileTime[int] = 0x1500,
):
    n_rows = ROWS
    dev = iron.get_current_device()
    assert dev is not None
    if n_cols > dev.cols:
        raise ValueError(f"n_cols={n_cols} but the device has {dev.cols} columns")
    assert hd % 8 == 0 and bq % 16 == 0 and bk % 16 == 0

    q_ty = np.ndarray[(bq * hd,), np.dtype[np.float32]]        # f32 in, converted to bf16 in place by the core
    kv_ty = np.ndarray[(bk * hd,), np.dtype[BF16]]
    o_ty = np.ndarray[(bq * hd,), np.dtype[np.float32]]        # the core's accumulator is its output object
    q_l2_ty = np.ndarray[(n_rows * bq * hd,), np.dtype[np.float32]]
    o_l2_ty = np.ndarray[(n_rows * bq * hd,), np.dtype[np.float32]]
    acc_ty = o_ty
    s_ty = np.ndarray[(2 * 8 * bk,), np.dtype[np.float32]]   # current + look-ahead row block of scores
    p_ty = np.ndarray[(8 * bk,), np.dtype[BF16]]
    m_ty = np.ndarray[(bq,), np.dtype[BF16]]                 # running row max (bf16), 8 per row block
    l_ty = np.ndarray[(bq,), np.dtype[np.float32]]           # running row sums
    mx_ty = np.ndarray[(32,), np.dtype[BF16]]                # block max of the scores waiting in S
    rtp_ty = np.ndarray[(3,), np.dtype[np.int32]]

    flags = [f"-DBQ={bq}", f"-DBK={bk}", f"-DHD={hd}", f"-DRESCALE_THRESHOLD={float(rescale_threshold)}f"]
    flags += extra_flags.split()
    kernels_dir = _aie_config.aie_kernels_dir()
    src = _KERNEL_SRC
    if "-DLUT_EXP" in flags:
        # The table-based bf16 exp of the runtime library: compiling the kernel through lut_kernel.cc links
        # the tables in (this is what the softmax factory does). The IRON cache then hashes lut_kernel.cc, not
        # our source, so the object name carries our source's hash.
        src = os.path.join(str(kernels_dir), "common", "lut_kernel.cc")
        flags.append(f'-DAIE_LUT_KERNEL_SOURCE="{_KERNEL_SRC}"')
    with open(_KERNEL_SRC, "rb") as f:
        src_hash = hashlib.sha256(f.read() + " ".join(flags).encode()).hexdigest()[:12]
    obj_name = f"xdna1_attn_q{bq}_k{bk}_d{hd}_{src_hash}.o"
    main_k = ExternalFunction(
        "attn_block_main", source_file=src, object_file_name=obj_name,
        arg_types=[q_ty, kv_ty, kv_ty, acc_ty, m_ty, l_ty, s_ty, p_ty, mx_ty, kv_ty, np.int32],
        include_dirs=[str(kernels_dir), os.path.join(str(_aie_config.aie_runtime_lib_dir()), "AIE2"),
                      os.path.dirname(_KERNEL_SRC)],
        compile_flags=flags)
    last_k = main_k.object_file.bind("attn_block_last", [q_ty, kv_ty, kv_ty, acc_ty, m_ty, l_ty, s_ty, p_ty, mx_ty, kv_ty])
    init_k = main_k.object_file.bind("attn_round_init", [q_ty, acc_ty, m_ty, l_ty, np.int32])
    finish_k = main_k.object_file.bind("attn_round_finish", [acc_ty, l_ty])

    # Row-major (rows x hd) -> tiles on the way into a core (8x8 for Q, the A operand of the kernel's
    # matrix products; 8x4 for K and V, the B operand); the inverse of the 8x4 tiling on the way out.
    tile_q = TensorAccessPattern.full((bq, hd)).tile((8, 8))
    tile_kv = TensorAccessPattern.full((bk, hd)).tile((8, 4))
    untile_o = TensorAccessPattern.full((bq, hd)).tile((8, 4)).inverse()  # per joined piece (one core)

    q_l3l2, q_l2l1, kv_l3l2, kv_l2l1, o_l2l3, o_l1l2 = [], [], [], [], [], []
    for c in range(n_cols):
        qf = ObjectFifo(q_l2_ty, name=f"Q_L3L2_{c}", depth=l3_depth)
        q_l3l2.append(qf)
        q_l2l1.append(qf.cons().split(
            [w * bq * hd for w in range(n_rows)], obj_types=[q_ty] * n_rows,
            names=[f"Q_L2L1_{c}_{w}" for w in range(n_rows)], to_stream=[tile_q] * n_rows,
            depths=[q_depth] * n_rows))
        kvf = ObjectFifo(kv_ty, name=f"KV_L3L2_{c}", depth=2 * l3_depth)
        kv_l3l2.append(kvf)
        kv_l2l1.append(kvf.cons().forward(obj_type=kv_ty, name=f"KV_L2L1_{c}", to_stream=tile_kv, depth=kv_depth))
        of = ObjectFifo(o_l2_ty, name=f"O_L2L3_{c}", depth=l3_depth, to_stream=untile_o)
        o_l2l3.append(of)
        o_l1l2.append(of.prod().join(
            [w * bq * hd for w in range(n_rows)], obj_types=[o_ty] * n_rows,
            names=[f"O_L1L2_{c}_{w}" for w in range(n_rows)], depths=[1] * n_rows))

    rtps = [[Buffer(rtp_ty, name=f"rtp_{w}_{c}", use_write_rtp=True) for c in range(n_cols)] for w in range(n_rows)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(n_cols)] for _ in range(n_rows)]

    def core_fn(q_f, kv_f, o_f, rtp, barrier, S, P, m, l, mx, KT, init, main, last, finish):
        barrier.wait_for_value(1)
        n_kv = rtp[0]       # key blocks per round, at least 2
        n_rounds = rtp[1]
        q_scale = rtp[2]    # f32 bits of scale * log2(e)
        barrier.release_with_value(1)
        for _ in range_(n_rounds):
            q = q_f.acquire(1)
            Oacc = o_f.acquire(1)   # the round accumulates in the output object itself
            init(q, Oacc, m, l, q_scale)
            # Key block j runs with block j+1's K already acquired (the look-ahead): acquire the pair
            # (K_j, V_j), then a third object K_j+1 before the last row block, release the pair.
            kv = kv_f.acquire(2)
            main(q, kv[0], kv[1], Oacc, m, l, S, P, mx, KT, 1)
            kv = kv_f.acquire(3)
            last(q, kv[2], kv[1], Oacc, m, l, S, P, mx, KT)
            kv_f.release(2)
            for _ in range_(n_kv - 2):
                kv = kv_f.acquire(2)
                main(q, kv[0], kv[1], Oacc, m, l, S, P, mx, KT, 0)
                kv = kv_f.acquire(3)
                last(q, kv[2], kv[1], Oacc, m, l, S, P, mx, KT)
                kv_f.release(2)
            kv = kv_f.acquire(2)
            main(q, kv[0], kv[1], Oacc, m, l, S, P, mx, KT, 0)
            last(q, kv[0], kv[1], Oacc, m, l, S, P, mx, KT)   # no next block: the look-ahead is discarded
            kv_f.release(2)
            finish(Oacc, l)
            o_f.release(1)
            q_f.release(1)

    workers = Worker.grid(
        n_rows, n_cols,
        lambda w, c: Worker(
            core_fn,
            [
                q_l2l1[c][w].cons(), kv_l2l1[c].cons(), o_l1l2[c][w].prod(), rtps[w][c], barriers[w][c],
                Buffer(s_ty, name=f"S_{w}_{c}"), Buffer(p_ty, name=f"P_{w}_{c}"),
                Buffer(m_ty, name=f"m_{w}_{c}"), Buffer(l_ty, name=f"l_{w}_{c}"), Buffer(mx_ty, name=f"mx_{w}_{c}"),
                Buffer(kv_ty, name=f"KT_{w}_{c}"),
                init_k, main_k, last_k, finish_k,
            ],
            stack_size=stack_size,
        ),
    )
    flat_workers = [wk for row in workers for wk in row]

    q_prods = [f.prod() for f in q_l3l2]
    kv_prods = [f.prod() for f in kv_l3l2]
    o_conses = [f.cons() for f in o_l2l3]
    def sequence(Q, KV, O, tq, tk, bh, h, ho, p0, ldq, qoff, qscale, q_hs, kv_hs, o_hs):
        require(tq > 0, "tq must be positive")
        require(tk > 0, "tk must be positive")
        require(bh > 0, "the pair count must be positive")
        require(h > 0, "heads must be positive")
        require(ho > 0, "output heads must be positive")
        require(p0 >= 0, "the first pair must be >= 0")
        require(ldq >= h * hd, "Q's row stride must hold all heads")
        require(qoff >= 0, "Q's offset must be >= 0")
        require(tq % (bq * n_rows) == 0, "tq must be a multiple of 4 * bq")
        require(tk % bk == 0, "tk must be a multiple of bk")
        require(tk >= 2 * bk, "tk must hold at least two key blocks")
        require(bh % n_cols == 0, "the pair count must be a multiple of the number of columns")
        ppc = bh // n_cols                      # (batch, head) pairs per column
        rounds = tq // (bq * n_rows)            # query rounds per pair
        n_kv = tk // bk
        n_pairs = p0 + bh                       # pairs the buffers hold at least
        for w in range(n_rows):
            for c in range(n_cols):
                rtps[w][c][0] = n_kv
                rtps[w][c][1] = ppc * rounds
                rtps[w][c][2] = qscale
                barriers[w][c].set(1)
        # Natural layouts: pair p = (batch p // h, head p % h) owns the (tq x hd) column block of head p % h in
        # Q (b, tq, h*hd) and in the output (b, tq, ho*hd), and the (2 tk x hd) block of K/V (b, tk/bk, 2, bk,
        # h*hd) (K_j, V_j alternating in blocks of bk rows). Rows are split into the round's 4*bq (or the
        # block's bk) so no DMA dimension exceeds the shim's 10-bit wrap; the K/V walk is repeated once per
        # round by the shim's queue.
        nb = (n_pairs + h - 1) // h
        kv_nat = TensorAccessPattern.full((nb, 2 * tk, h, hd))
        o_nat = TensorAccessPattern.full(((n_pairs + ho - 1) // ho, tq, ho, hd))
        for pi, tg in TaskGroup.pipelined(ppc, depth=2):
            for c in range(n_cols):
                p = p0 + c * ppc + pi
                bi, hi = p // h, p % h
                bo, hio = p // ho, p % ho
                o_hs[c].drain(O, tap=o_nat[bo, :, hio, :].split(0, bq * n_rows), wait=True, group=tg)
                q_tap = TensorAccessPattern((Q.shape[0],), qoff + bi * tq * ldq + hi * hd,
                                            (rounds, bq * n_rows, hd), (bq * n_rows * ldq, ldq, 1))
                q_hs[c].fill(Q, tap=q_tap, group=tg)
                kv_hs[c].fill(KV, tap=kv_nat[bi, :, hi, :].split(0, bk).repeat(rounds), group=tg)

    rt = Runtime(sequence, [Q, KV, O, tq, tk, bh, h, ho, p0, ldq, qoff, qscale, q_prods, kv_prods, o_conses])
    return Program(dev, rt, workers=flat_workers).resolve_program()


# Per head dimension: (bq, bk) that fit the 64 KB core memory with double-buffered K/V.
DEFAULT_TILES = {40: (64, 64), 48: (64, 64), 56: (64, 64), 64: (32, 64), 72: (32, 32), 80: (32, 32), 96: (32, 32), 128: (16, 32), 160: (16, 16)}


class NpuAttention:
    """One compiled fused-attention program for a fixed head dimension, dispatched directly through XRT."""

    def __init__(self, hd: int, bq: int | None = None, bk: int | None = None, n_cols: int = 4,
                 max_elems: int = 1 << 24, verbose: bool = True, **compile_kwargs):
        if hd % 8:
            raise ValueError(f"head dim {hd} is not a multiple of 8")
        if bq is None or bk is None:
            bq, bk = DEFAULT_TILES.get(hd, (16, 32))
        self.hd, self.bq, self.bk, self.n_cols = hd, bq, bk, n_cols
        self.max_elems = max_elems
        self.stats = {"dispatches": 0, "npu_seconds": 0.0, "compile_seconds": 0.0}
        self._insts: dict[tuple[int, ...], tuple[object, object, int]] = {}
        if iron.get_current_device() is None:
            iron.set_current_device(from_name("npu", n_cols=n_cols))
        t0 = time.perf_counter()
        # The fused step spills more the wider the head: aiecc measures the need and refuses a smaller stack.
        compile_kwargs.setdefault("stack_size", 0x2400 if hd >= 128 else 0x1C00 if hd >= 64 else 0x1500)
        self.design = sd_attention.specialize(
            Q=np.ndarray[(max_elems,), np.dtype[np.float32]], KV=np.ndarray[(2 * max_elems,), np.dtype[BF16]],
            O=np.ndarray[(max_elems,), np.dtype[np.float32]], hd=hd, bq=bq, bk=bk, n_cols=n_cols, **compile_kwargs)
        q0 = iron.zeros((max_elems,), dtype=np.float32, device="npu")
        kv0 = iron.zeros((2 * max_elems,), dtype=BF16, device="npu")
        o0 = iron.zeros((max_elems,), dtype=np.float32, device="npu")
        # compile (or load from the cache) + warm-up run
        self.design(q0, kv0, o0, tq=bq * ROWS, tk=2 * bk, bh=n_cols, h=1, ho=1, p0=0, ldq=hd, qoff=0,
                    qscale=_f32_bits(LOG2E))
        del q0, kv0, o0
        kernels = list(getattr(self.design, "_kernel_cache", {}).values())
        if not kernels:
            raise RuntimeError("sd_attention design did not cache its compiled kernel; cannot dispatch directly")
        self.kernel = kernels[0]
        self.rt = _aie_utils.DefaultNPURuntime
        self._handle = self.rt.load(self.kernel)
        self.stats["compile_seconds"] = time.perf_counter() - t0
        if verbose:
            print(f"[xdna1_attention] d={hd} program ready in {self.stats['compile_seconds']:.1f}s "
                  f"(bq={bq}, bk={bk}, {n_cols} columns)", flush=True)

    def fits(self, tq: int, tk: int, bh: int, d: int) -> bool:
        return (d == self.hd and tq % (self.bq * ROWS) == 0 and tk % self.bk == 0 and tk >= 2 * self.bk
                and bh % self.n_cols == 0
                and bh * tq * d <= self.max_elems and bh * tk * d <= self.max_elems)

    def _insts_for(self, tq: int, tk: int, bh: int, h: int, ho: int, p0: int, ldq: int, qoff: int, qscale: int):
        key = (tq, tk, bh, h, ho, p0, ldq, qoff, qscale)
        ent = self._insts.get(key)
        if ent is None:
            words = self.kernel._generate_dispatch_insts({"tq": tq, "tk": tk, "bh": bh, "h": h, "ho": ho, "p0": p0,
                                                          "ldq": ldq, "qoff": qoff, "qscale": qscale})
            store = XrtTransport(self.rt._device, words.nbytes, pyxrt.bo.cacheable, self._handle.kernel.group_id(1))
            store.host_bytes[:] = words.view(np.uint8).reshape(-1)
            store.to_device(0, words.nbytes)
            ent = (store, store.handle(0, words.nbytes), words.nbytes)
            self._insts[key] = ent
        return ent

    def submit(self, q_bo, kv_bo, o_bo, tq: int, tk: int, bh: int, h: int = 1, ho: int | None = None, p0: int = 0,
               ldq: int | None = None, qoff: int = 0, q_scale: float = LOG2E):
        """Start one attention dispatch (non-blocking); returns the pyxrt run handle for wait().

        ``bh`` pairs starting at pair ``p0`` of buffers laid out as Q ``(b, tq, h*d)`` f32 with row stride
        ``ldq`` elements (default ``h*d``) starting ``qoff`` elements into its buffer, K/V ``(b, tk/bk, 2, bk,
        h*d)`` bf16 (K and V blocks of bk rows alternating) and the output ``(b, tq, ho*d)`` f32 (``ho`` defaults
        to ``h``); ``h = 1`` is the head-major ``(b*h, t, d)`` form. ``q_scale`` multiplies Q on the core: pass
        the attention scale times ``LOG2E`` (the softmax weights are ``2^(S - m)``).
        """
        if not getattr(self._handle, "_is_valid", True):
            self._handle = self.rt.load(self.kernel)
        _, insts_bo, nbytes = self._insts_for(tq, tk, bh, h, h if ho is None else ho, p0,
                                              h * self.hd if ldq is None else ldq, qoff, _f32_bits(q_scale))
        self.stats["dispatches"] += 1
        return self._handle.kernel(3, insts_bo, nbytes, q_bo, kv_bo, o_bo)

    @staticmethod
    def wait(run) -> None:
        r = run.wait()
        if r != _COMPLETED:
            raise RuntimeError(f"NPU attention dispatch failed: kernel returned {r}")
