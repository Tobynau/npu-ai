"""One NPU program for both the GEMMs and the fused attention on the AMD XDNA1 NPU.

The GEMM (mlir-aie's whole_array) and the fused attention (``npu/xdna1_attention.py``) used to be two
programs. Both use all four columns, so the firmware reconfigured the array whenever a dispatch of the other
program followed, ~2.5 ms each way (docs/02-results.md, 3d), and in a UNet every attention layer sits between
Linear layers. This module is a single IRON design whose cores run either computation, chosen per dispatch by
a runtime parameter, so a GEMM after an attention block (or the reverse) runs on the array as configured.

Program memory decides the granularity: the attention kernel for one head dimension is 9.5-10 KB of a core's
16 KB, so one program holds the GEMM and the attention for *one* head dimension. ``NpuGemm`` compiles one per
head dimension it serves (SD1.5: 40 and 80) and sends its GEMMs to whichever program ran last, so the array is
only reconfigured when the head dimension changes (twice per UNet call instead of twice per attention layer).

Dataflow (4 columns x 4 rows of cores, three object streams per core, laid out like whole_array):

* Y, one per row (shim -> memtile -> the row's four cores, broadcast), objects of (bq x 32) bf16 retiled into
  8x8 tiles by the memtile. GEMM: the A tiles of the row's bq output rows, 32 columns of K each. Attention: the
  round's Q, see below.
* X, one per column (shim -> memtile -> the column's four cores, broadcast), objects of (bk x hd) bf16 retiled
  into 8x4 tiles. GEMM: the B tiles of the column's output columns, bk rows of K each. Attention: the column's
  (batch, head) pair's K and V blocks, as in the standalone attention program.
* Z, one per column (the column's four cores -> memtile join -> shim), objects of (bq x hd) f32 in 8x4 blocks,
  un-tiled by the memtile. GEMM: C tiles. Attention: the round's output rows.

GEMM mode computes C (M x N, f32) = A (M x K, bf16) @ B (K x N, bf16): core (row r, column c) computes the
(bq x hd) C tiles at row block r of each 4*bq-row block and column tiles c, c+4, ... of width hd, so a GEMM
dispatch needs M % (4 bq), K % (bk * gemm_kx) and N % (4 hd) == 0. Per C tile it accumulates K / 32 A tiles
against K / bk B tiles in the attention's Q and K^T buffers and copies the result into the output object. One
``gemm_step`` call takes 32 of K (one Y object), or with ``gemm_kx = 2`` (d = 80) 64 of K (two Y and two X
objects), and passes the C tile through the core's registers once.

Attention mode is the flash attention of ``npu/xdna1_attention.py`` (kernel ``npu/kernels/xdna1_attn.cc``)
with two changes forced by the shared streams. The four columns work on four *consecutive* pairs, which in the
model's (b, t, h*d) layout are four adjacent heads of one batch, so the Q rows a row of cores needs in a round
are one (bq x 4*hd) slab of Q; the row's Y stream carries that slab in strips of 16 columns (each core keeps the
quarter of every strip that belongs to its column's head, converting it to bf16 as before). And Q is therefore
read into the core through a broadcast that carries four cores' worth of it, ~5 % of the dispatch at 32x32,
~1.5 % at 64x64. Dispatch parameters as before (``tq``, ``tk``, the pair count ``bh``, heads per batch of
the Q, K/V and output layouts ``hq``, ``hkv``, ``ho``, first pair ``p0``, Q's row stride ``ldq``, the Q
scale), except that Q's offset in its buffer is now a sub-buffer handle (XRT allows 16 kernel arguments: three
buffers and 13 parameters), and in addition ``hq % 4 == 0`` and ``p0 % 4 == 0`` (the runner copies Q into such
a layout when the model's is not).
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
from aie.iron.controlflow import else_, if_, range_  # noqa: E402
from aie.iron.device import from_name  # noqa: E402
from aie.utils import config as _aie_config  # noqa: E402

import pyxrt  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XrtTransport  # noqa: E402

BF16 = str_to_dtype("bf16")
ROWS = 4
LOG2E = 1.4426950408889634
YW = 32  # columns (bf16) of K in one A tile / Y object; 16 f32 columns of Q
_KERNEL_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "xdna1_attn.cc")
_COMPLETED = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
GEMM, ATTENTION = 0, 1


def _f32_bits(x: float) -> int:
    """The bit pattern of float32(x) as a Python int in the int32 range (how the kernel receives a float)."""
    return int(np.array([x], dtype=np.float32).view(np.int32)[0])


@iron.jit
def npu_program(
    X1: In,
    X2: In,
    Y: Out,
    *,
    mode: DispatchTime[np.int32],
    M: DispatchTime[np.int32],
    K: DispatchTime[np.int32],
    N: DispatchTime[np.int32],
    tq: DispatchTime[np.int32],
    tk: DispatchTime[np.int32],
    bh: DispatchTime[np.int32],
    hq: DispatchTime[np.int32],
    hkv: DispatchTime[np.int32],
    ho: DispatchTime[np.int32],
    p0: DispatchTime[np.int32],
    ldq: DispatchTime[np.int32],
    qscale: DispatchTime[np.int32],
    hd: CompileTime[int],
    bq: CompileTime[int],
    bk: CompileTime[int],
    n_cols: CompileTime[int] = 4,
    kv_depth: CompileTime[int] = 4,
    y_depth: CompileTime[int] = 2,
    gemm_kx: CompileTime[int] = 1,
    l3_depth: CompileTime[int] = 2,
    rescale_threshold: CompileTime[float] = 8.0,
    extra_flags: CompileTime[str] = "",
    stack_size: CompileTime[int] = 0x1500,
):
    """X1: A (bf16, M x K row-major) or Q (f32, addressed in bf16 units); X2: B (bf16, K x N) or K/V (bf16,
    (b, tk/bk, 2, bk, hkv*hd)); Y: C (f32, M x N) or the attention output (f32, (b, tq, ho*hd))."""
    n_rows = ROWS
    dev = iron.get_current_device()
    assert dev is not None
    if n_cols != 4 or n_cols > dev.cols:
        raise ValueError(f"the unified program needs 4 columns (got n_cols={n_cols}, device has {dev.cols})")
    assert hd % 8 == 0 and bq % 32 == 0 and bk % YW == 0
    y_per_x = bk // YW
    # X objects per gemm_step call (1, or 2 with bk = 32: one call takes 64 of K, two Y and two X objects, so the
    # C tile goes through the core's registers half as often; needs K % 64 and room for Y depth 4).
    if gemm_kx not in (1, 2) or (gemm_kx == 2 and y_per_x != 1):
        raise ValueError(f"gemm_kx={gemm_kx} needs 1, or 2 with bk == {YW} (got bk={bk})")
    n_strips = hd * n_cols // 16  # Q strips per round (16 f32 columns each)

    y_ty = np.ndarray[(bq * YW,), np.dtype[BF16]]
    x_ty = np.ndarray[(bk * hd,), np.dtype[BF16]]
    z_ty = np.ndarray[(bq * hd,), np.dtype[np.float32]]
    z_l2_ty = np.ndarray[(n_rows * bq * hd,), np.dtype[np.float32]]
    qb_ty = np.ndarray[(bq * hd,), np.dtype[BF16]]       # Q in bf16; in GEMM mode half of the C tile (as f32)
    kt_ty = x_ty                                          # K^T; in GEMM mode the other half of the C tile
    assert bk >= bq, "K^T must hold half a C tile (as Qb does)"
    s_ty = np.ndarray[(2 * 8 * bk,), np.dtype[np.float32]]
    p_ty = np.ndarray[(8 * bk,), np.dtype[BF16]]
    m_ty = np.ndarray[(bq,), np.dtype[BF16]]
    l_ty = np.ndarray[(bq,), np.dtype[np.float32]]
    mx_ty = np.ndarray[(32,), np.dtype[BF16]]
    rtp_ty = np.ndarray[(4,), np.dtype[np.int32]]

    flags = [f"-DBQ={bq}", f"-DBK={bk}", f"-DHD={hd}", f"-DRESCALE_THRESHOLD={float(rescale_threshold)}f", "-DUNIFIED",
             f"-DGEMM_KX={gemm_kx}"]
    flags += extra_flags.split()
    kernels_dir = _aie_config.aie_kernels_dir()
    with open(_KERNEL_SRC, "rb") as f:
        src_hash = hashlib.sha256(f.read() + " ".join(flags).encode()).hexdigest()[:12]
    gemm_k = ExternalFunction(
        "gemm_step", source_file=_KERNEL_SRC, object_file_name=f"xdna1_unified_q{bq}_k{bk}_d{hd}_{src_hash}.o",
        arg_types=[y_ty, x_ty, qb_ty, kt_ty, np.int32] if gemm_kx == 1 else [y_ty, y_ty, x_ty, x_ty, qb_ty, kt_ty],
        include_dirs=[str(kernels_dir), os.path.join(str(_aie_config.aie_runtime_lib_dir()), "AIE2"),
                      os.path.dirname(_KERNEL_SRC)],
        compile_flags=flags)
    obj = gemm_k.object_file
    gzero_k = obj.bind("gemm_zero", [qb_ty, kt_ty])
    gstore_k = obj.bind("gemm_store", [qb_ty, kt_ty, z_ty])
    qstrip_k = obj.bind("attn_q_strip", [y_ty, qb_ty, np.int32, np.int32, np.int32])
    init_k = obj.bind("attn_round_init_o", [z_ty, m_ty, l_ty])
    main_k = obj.bind("attn_block_main", [qb_ty, x_ty, x_ty, z_ty, m_ty, l_ty, s_ty, p_ty, mx_ty, kt_ty, np.int32])
    last_k = obj.bind("attn_block_last", [qb_ty, x_ty, x_ty, z_ty, m_ty, l_ty, s_ty, p_ty, mx_ty, kt_ty])
    finish_k = obj.bind("attn_round_finish", [z_ty, l_ty])

    # Memtile retiling: Y into 8x8 bf16 tiles (the A operand), X into 8x4 tiles (the B operand), Z back from 8x4.
    tile_y = TensorAccessPattern.full((bq, YW)).tile((8, 8))
    tile_x = TensorAccessPattern.full((bk, hd)).tile((8, 4))
    untile_z = TensorAccessPattern.full((bq, hd)).tile((8, 4)).inverse()

    y_l3l2, y_l2l1, x_l3l2, x_l2l1, z_l2l3, z_l1l2 = [], [], [], [], [], []
    for i in range(n_rows):
        yf = ObjectFifo(y_ty, name=f"Y_L3L2_{i}", depth=l3_depth)
        y_l3l2.append(yf)
        y_l2l1.append(yf.cons().forward(obj_type=y_ty, name=f"Y_L2L1_{i}", to_stream=tile_y, depth=y_depth))
    for c in range(n_cols):
        xf = ObjectFifo(x_ty, name=f"X_L3L2_{c}", depth=2 * l3_depth)
        x_l3l2.append(xf)
        x_l2l1.append(xf.cons().forward(obj_type=x_ty, name=f"X_L2L1_{c}", to_stream=tile_x, depth=kv_depth))
        zf = ObjectFifo(z_l2_ty, name=f"Z_L2L3_{c}", depth=l3_depth, to_stream=untile_z)
        z_l2l3.append(zf)
        z_l1l2.append(zf.prod().join(
            [w * bq * hd for w in range(n_rows)], obj_types=[z_ty] * n_rows,
            names=[f"Z_L1L2_{c}_{w}" for w in range(n_rows)], depths=[1] * n_rows))

    rtps = [[Buffer(rtp_ty, name=f"rtp_{w}_{c}", use_write_rtp=True) for c in range(n_cols)] for w in range(n_rows)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(n_cols)] for _ in range(n_rows)]

    def core_fn(y_f, x_f, z_f, rtp, barrier, Qb, KT, S, P, m, l, mx,
                gemm, gzero, gstore, qstrip, init, main, last, finish, col):
        barrier.wait_for_value(1)
        mode_ = rtp[0]
        n1 = rtp[1]    # GEMM: C tiles of this core;  attention: rounds (pairs per column x rounds per pair)
        n2 = rtp[2]    # GEMM: gemm_step calls (K / (bk * gemm_kx)) per C tile;  attention: key blocks per round, >= 2
        n3 = rtp[3]    # attention: f32 bits of scale * log2(e)
        barrier.release_with_value(1)
        with if_(mode_ == GEMM) as branch:
            for _ in range_(n1):
                gzero(Qb, KT)
                if gemm_kx == 2:
                    for _ in range_(n2):   # n2 = K / 64 here
                        b = x_f.acquire(2)
                        a = y_f.acquire(2)
                        gemm(a[0], a[1], b[0], b[1], Qb, KT)
                        y_f.release(2)
                        x_f.release(2)
                else:
                    for _ in range_(n2):
                        b = x_f.acquire(1)
                        for half in range(y_per_x):
                            a = y_f.acquire(1)
                            gemm(a, b, Qb, KT, half)
                            y_f.release(1)
                        x_f.release(1)
                c_out = z_f.acquire(1)
                gstore(Qb, KT, c_out)
                z_f.release(1)
        with else_(branch):
            for _ in range_(n1):
                for s in range_(n_strips):
                    a = y_f.acquire(1)
                    qstrip(a, Qb, s, col, n3)
                    y_f.release(1)
                Oacc = z_f.acquire(1)   # the round accumulates in the output object itself
                init(Oacc, m, l)
                # Key block j runs with block j+1's K already acquired (the look-ahead): acquire the pair
                # (K_j, V_j), then a third object K_j+1 before the last row block, release the pair.
                kv = x_f.acquire(2)
                main(Qb, kv[0], kv[1], Oacc, m, l, S, P, mx, KT, 1)
                kv = x_f.acquire(3)
                last(Qb, kv[2], kv[1], Oacc, m, l, S, P, mx, KT)
                x_f.release(2)
                for _ in range_(n2 - 2):
                    kv = x_f.acquire(2)
                    main(Qb, kv[0], kv[1], Oacc, m, l, S, P, mx, KT, 0)
                    kv = x_f.acquire(3)
                    last(Qb, kv[2], kv[1], Oacc, m, l, S, P, mx, KT)
                    x_f.release(2)
                kv = x_f.acquire(2)
                main(Qb, kv[0], kv[1], Oacc, m, l, S, P, mx, KT, 0)
                last(Qb, kv[0], kv[1], Oacc, m, l, S, P, mx, KT)   # no next block: the look-ahead is discarded
                x_f.release(2)
                finish(Oacc, l)
                z_f.release(1)

    workers = Worker.grid(
        n_rows, n_cols,
        lambda w, c: Worker(
            core_fn,
            [
                y_l2l1[w].cons(), x_l2l1[c].cons(), z_l1l2[c][w].prod(), rtps[w][c], barriers[w][c],
                Buffer(qb_ty, name=f"Qb_{w}_{c}"), Buffer(kt_ty, name=f"KT_{w}_{c}"),
                Buffer(s_ty, name=f"S_{w}_{c}"), Buffer(p_ty, name=f"P_{w}_{c}"),
                Buffer(m_ty, name=f"m_{w}_{c}"), Buffer(l_ty, name=f"l_{w}_{c}"), Buffer(mx_ty, name=f"mx_{w}_{c}"),
                gemm_k, gzero_k, gstore_k, qstrip_k, init_k, main_k, last_k, finish_k, c,
            ],
            stack_size=stack_size,
        ),
    )
    flat_workers = [wk for row in workers for wk in row]

    y_prods = [f.prod() for f in y_l3l2]
    x_prods = [f.prod() for f in x_l3l2]
    z_conses = [f.cons() for f in z_l2l3]

    def sequence(X1, X2, Y, mode, M, K, N, tq, tk, bh, hq, hkv, ho, p0, ldq, qscale, y_hs, x_hs, z_hs):
        # Both modes' parameters are checked on every dispatch; the host passes neutral values for the other mode.
        require(mode >= 0, "mode must be 0 (GEMM) or 1 (attention)")
        require(mode <= 1, "mode must be 0 (GEMM) or 1 (attention)")
        require(M > 0, "M must be positive")
        require(K > 0, "K must be positive")
        require(N > 0, "N must be positive")
        require(M % (bq * n_rows) == 0, "M must be a multiple of 4 * bq")
        require(K % (bk * gemm_kx) == 0, "K must be a multiple of bk * gemm_kx")
        require(N % (hd * n_cols) == 0, "N must be a multiple of 4 * hd")
        require(tq > 0, "tq must be positive")
        require(tk > 0, "tk must be positive")
        require(bh > 0, "the pair count must be positive")
        require(hq > 0, "heads must be positive")
        require(hkv > 0, "heads must be positive")
        require(ho > 0, "output heads must be positive")
        require(p0 >= 0, "the first pair must be >= 0")
        require(ldq >= hq * hd, "Q's row stride must hold all heads")
        require(tq % (bq * n_rows) == 0, "tq must be a multiple of 4 * bq")
        require(tk % bk == 0, "tk must be a multiple of bk")
        require(tk >= 2 * bk, "tk must hold at least two key blocks")
        require(bh % n_cols == 0, "the pair count must be a multiple of the number of columns")
        require(hq % n_cols == 0, "Q's heads per row must be a multiple of 4 (four adjacent heads per round)")
        require(p0 % n_cols == 0, "the first pair must be a multiple of 4")
        gemm = 1 - mode
        n_nt = N // (hd * n_cols)                   # C column tiles per core
        n_rb = M // (bq * n_rows) * gemm            # 4*bq-row blocks (0 in attention mode)
        ppc = bh // n_cols * mode                   # pairs per column (0 in GEMM mode)
        rounds = tq // (bq * n_rows)
        n_kv = tk // bk
        for w in range(n_rows):
            for c in range(n_cols):
                rtps[w][c][0] = mode
                rtps[w][c][1] = n_rb * n_nt + ppc * rounds
                rtps[w][c][2] = K // (bk * gemm_kx) * gemm + n_kv * mode
                rtps[w][c][3] = qscale * mode
                barriers[w][c].set(1)

        # GEMM: per block of 4*bq rows, row i's A rows (once per C column tile of a core: the repeat), column c's
        # B tiles (its C column tiles c, c+4, ..., each over all of K) and the joined C tiles of column c.
        for rb, tg in TaskGroup.pipelined(n_rb, depth=4):
            for i in range(n_rows):
                z_hs[i].drain(Y, tap=TensorAccessPattern(
                    (Y.shape[0],), rb * (bq * n_rows) * N + i * hd, (n_nt, bq * n_rows, hd), (hd * n_cols, N, 1)),
                    wait=True, group=tg)
                y_hs[i].fill(X1, tap=TensorAccessPattern(
                    (X1.shape[0],), (rb * (bq * n_rows) + i * bq) * K, (n_nt, K // YW, bq, YW), (0, YW, K, 1)),
                    group=tg)
                x_hs[i].fill(X2, tap=TensorAccessPattern(
                    (X2.shape[0],), i * hd, (n_nt, K // bk, bk, hd), (hd * n_cols, bk * N, N, 1)), group=tg)

        # Attention: per step the four columns take pairs pb .. pb+3 (pb = p0 + 4 * step), four adjacent heads of
        # one batch of Q. Row w's Y stream carries, per round, its bq query rows of that (bq x 4*hd) Q slab in
        # strips of 16 f32 columns (Q is f32, X1 is typed bf16: offsets and strides in bf16 units). Column c drains
        # pair pb+c's output rows and streams its K/V blocks once per round, as in the standalone program.
        n_pairs = p0 + bh
        kv_nat = TensorAccessPattern.full(((n_pairs + hkv - 1) // hkv, 2 * tk, hkv, hd))
        o_nat = TensorAccessPattern.full(((n_pairs + ho - 1) // ho, tq, ho, hd))
        for pi, tg in TaskGroup.pipelined(ppc, depth=2):
            pb = p0 + pi * n_cols
            qb_, qh_ = pb // hq, pb % hq
            for w in range(n_rows):
                y_hs[w].fill(X1, tap=TensorAccessPattern(
                    (X1.shape[0],), 2 * (qb_ * tq * ldq + qh_ * hd + w * bq * ldq),
                    (rounds, n_strips, bq, YW), (2 * bq * n_rows * ldq, YW, 2 * ldq, 1)), group=tg)
            for c in range(n_cols):
                p = pb + c
                z_hs[c].drain(Y, tap=o_nat[p // ho, :, p % ho, :].split(0, bq * n_rows), wait=True, group=tg)
                x_hs[c].fill(X2, tap=kv_nat[p // hkv, :, p % hkv, :].split(0, bk).repeat(rounds), group=tg)

    rt = Runtime(sequence, [X1, X2, Y, mode, M, K, N, tq, tk, bh, hq, hkv, ho, p0, ldq, qscale,
                            y_prods, x_prods, z_conses])
    return Program(dev, rt, workers=flat_workers).resolve_program()


# Per head dimension: attention tiles (bq, bk). bq is also the GEMM's C tile height, bk its K step, hd its C tile
# width. Within 64 KB of core memory: K/V or B 4 x 5 KB, the output 10 KB, Y 2 x 2-4 KB, Q and K^T 5 KB each.
DEFAULT_TILES = {40: (64, 64), 48: (64, 64), 56: (64, 64), 64: (32, 64), 72: (32, 32), 80: (32, 32), 96: (32, 32)}
# Head dims whose program runs the GEMM in K steps of 64 (gemm_kx = 2, Y depth 4: +4 KB of data memory, +0.2 KB of
# program memory; d = 40 at ~60 of 64 KB has no room). Off: measured no faster than K steps of 32 (within noise, the
# long-K GEMMs are bound by the shim streams, not by the C tile's trips through the registers; docs/02-results.md 3e).
GEMM_K64: set[int] = set()


class NpuProgram:
    """One compiled unified program (GEMM + fused attention for head dimension ``hd``), dispatched through XRT.

    ``max_in`` / ``max_out`` (elements) bound the buffers any dispatch may address: X1 and X2 as bf16 (Q, f32,
    counts double), Y as f32. The dispatch bridge refuses taps outside them.
    """

    def __init__(self, hd: int, bq: int | None = None, bk: int | None = None, n_cols: int = 4,
                 max_in: int = 1 << 25, max_out: int = 1 << 24, verbose: bool = True, **compile_kwargs):
        if hd % 8:
            raise ValueError(f"head dim {hd} is not a multiple of 8")
        if bq is None or bk is None:
            bq, bk = DEFAULT_TILES.get(hd, (32, 32))
        if hd in GEMM_K64 and bk == YW and "gemm_kx" not in compile_kwargs:
            compile_kwargs.update(gemm_kx=2, y_depth=4)
        self.gemm_kx = compile_kwargs.get("gemm_kx", 1)
        self.hd, self.bq, self.bk, self.n_cols = hd, bq, bk, n_cols
        self.align_m, self.align_k, self.align_n = ROWS * bq, max(bk, YW) * self.gemm_kx, n_cols * hd
        self.max_in, self.max_out = max_in, max_out
        self.stats = {"dispatches": 0, "gemm_dispatches": 0, "attn_dispatches": 0, "compile_seconds": 0.0}
        self._insts: dict[tuple[int, ...], tuple[object, object, int]] = {}
        if iron.get_current_device() is None:
            iron.set_current_device(from_name("npu", n_cols=n_cols))
        t0 = time.perf_counter()
        # The attention step spills more the wider the head: aiecc measures the need and refuses a smaller stack.
        compile_kwargs.setdefault("stack_size", 0x2400 if hd >= 128 else 0x1C00 if hd >= 64 else 0x1500)
        self.design = npu_program.specialize(
            X1=np.ndarray[(max_in,), np.dtype[BF16]], X2=np.ndarray[(max_in,), np.dtype[BF16]],
            Y=np.ndarray[(max_out,), np.dtype[np.float32]], hd=hd, bq=bq, bk=bk, n_cols=n_cols, **compile_kwargs)
        x1 = iron.zeros((max_in,), dtype=BF16, device="npu")
        x2 = iron.zeros((max_in,), dtype=BF16, device="npu")
        y = iron.zeros((max_out,), dtype=np.float32, device="npu")
        # compile (or load from the cache) + warm-up run: one GEMM block
        self.design(x1, x2, y, **self._params(GEMM, M=self.align_m, K=self.align_k, N=self.align_n))
        del x1, x2, y
        kernels = list(getattr(self.design, "_kernel_cache", {}).values())
        if not kernels:
            raise RuntimeError("npu_program did not cache its compiled kernel; cannot dispatch directly")
        self.kernel = kernels[0]
        self.rt = _aie_utils.DefaultNPURuntime
        self._handle = self.rt.load(self.kernel)
        self.stats["compile_seconds"] = time.perf_counter() - t0
        if verbose:
            print(f"[xdna1_program] d={hd} program (GEMM + attention) ready in {self.stats['compile_seconds']:.1f}s "
                  f"(attention bq={bq} bk={bk}; GEMM tile {bq}x{bk * self.gemm_kx}x{hd}, align M%{self.align_m} K%{self.align_k} "
                  f"N%{self.align_n})", flush=True)

    def _params(self, mode: int, **kw) -> dict:
        """All dispatch parameters, the other mode's set to values that pass its checks (and issue nothing)."""
        bq, bk, hd = self.bq, self.bk, self.hd
        p = dict(mode=mode, M=ROWS * bq, K=self.align_k, N=self.n_cols * hd, tq=ROWS * bq, tk=2 * bk, bh=self.n_cols,
                 hq=self.n_cols, hkv=1, ho=1, p0=0, ldq=self.n_cols * hd, qscale=0)
        p.update(kw)
        return p

    # -- GEMM -----------------------------------------------------------------------------------
    def gemm_fits(self, M: int, K: int, N: int) -> bool:
        return (M % self.align_m == 0 and K % self.align_k == 0 and N % self.align_n == 0
                and M * K <= self.max_in and K * N <= self.max_in and M * N <= self.max_out)

    # -- attention ------------------------------------------------------------------------------
    def fits(self, tq: int, tk: int, bh: int, d: int) -> bool:
        """Whether an attention block of this shape runs here (the Q layout condition is the caller's)."""
        return (d == self.hd and tq % (self.bq * ROWS) == 0 and tk % self.bk == 0 and tk >= 2 * self.bk
                and bh % self.n_cols == 0 and tq // (self.bq * ROWS) <= 64
                and 2 * bh * tq * d <= self.max_in and bh * tk * d * 2 <= self.max_in and bh * tq * d <= self.max_out)

    # -- dispatch -------------------------------------------------------------------------------
    def _insts_for(self, params: dict):
        key = tuple(params.values())
        ent = self._insts.get(key)
        if ent is None:
            words = self.kernel._generate_dispatch_insts(params)
            store = XrtTransport(self.rt._device, words.nbytes, pyxrt.bo.cacheable, self._handle.kernel.group_id(1))
            store.host_bytes[:] = words.view(np.uint8).reshape(-1)
            store.to_device(0, words.nbytes)
            ent = (store, store.handle(0, words.nbytes), words.nbytes)
            self._insts[key] = ent
        return ent

    def _submit(self, params: dict, bo1, bo2, bo3):
        if not getattr(self._handle, "_is_valid", True):
            self._handle = self.rt.load(self.kernel)
        _, insts_bo, nbytes = self._insts_for(params)
        self.stats["dispatches"] += 1
        return self._handle.kernel(3, insts_bo, nbytes, bo1, bo2, bo3)

    def gemm_params(self, M: int, K: int, N: int) -> dict:
        return self._params(GEMM, M=M, K=K, N=N)

    def submit_gemm(self, a_bo, b_bo, c_bo, M: int, K: int, N: int):
        """Start C (M x N f32) = A (M x K bf16) @ B (K x N bf16), all packed row-major; returns the run handle."""
        self.stats["gemm_dispatches"] += 1
        return self._submit(self.gemm_params(M, K, N), a_bo, b_bo, c_bo)

    def attn_params(self, tq: int, tk: int, bh: int, hq: int, hkv: int, ho: int, p0: int, ldq: int,
                    q_scale: float) -> dict:
        return self._params(ATTENTION, tq=tq, tk=tk, bh=bh, hq=hq, hkv=hkv, ho=ho, p0=p0, ldq=ldq,
                            qscale=_f32_bits(q_scale))

    def submit_attn(self, q_bo, kv_bo, o_bo, tq: int, tk: int, bh: int, hq: int, hkv: int, ho: int, p0: int = 0,
                    ldq: int | None = None, q_scale: float = LOG2E):
        """Start one attention dispatch (non-blocking); returns the pyxrt run handle for wait().

        ``bh`` pairs starting at pair ``p0`` of: Q ``(b, tq, hq*d)`` f32 with row stride ``ldq`` elements (default
        ``hq*d``) at the start of ``q_bo`` (a sub-buffer handle places it), ``hq % 4 == 0``; K/V ``(b, tk/bk, 2, bk, hkv*d)``
        bf16 (K and V blocks of bk rows alternating); the output ``(b, tq, ho*d)`` f32. ``h = 1`` is the head-major
        ``(b*h, t, d)`` form (for K/V and the output). ``q_scale`` multiplies Q on the core: pass the attention
        scale times ``LOG2E`` (the softmax weights are ``2^(S - m)``).
        """
        self.stats["attn_dispatches"] += 1
        params = self.attn_params(tq, tk, bh, hq, hkv, ho, p0, hq * self.hd if ldq is None else ldq, q_scale)
        return self._submit(params, q_bo, kv_bo, o_bo)

    @staticmethod
    def wait(run) -> None:
        r = run.wait()
        if r != _COMPLETED:
            raise RuntimeError(f"NPU dispatch failed: kernel returned {r}")
