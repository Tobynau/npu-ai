//===- xdna1_attn.cc --------------------------------------------*- C++ -*-===//
//
// Fused attention for one XDNA1 (AIE2) compute core: flash attention with an online softmax over a
// query block of BQ rows, consumed one key block of BK keys at a time.
//
//   attn_round_init    O = 0, m = -inf, l = 0                               (once per query block)
//   attn_block_main    row blocks 0 .. TQ-2 of one key block                 (once per key block)
//   attn_block_last    row block TQ-1, plus the look-ahead into the next key block
//   attn_round_finish  O = O / l, in place                                   (once per query block)
//
// O is the core's output object itself: the round accumulates straight into the f32 buffer the DMA
// then drains (no separate accumulator, no bf16 copy; the host gets f32). Q arrives as f32 too (the
// model's own tensor, no host conversion) and attn_round_init scales it by the attention scale times
// log2(e) and rounds it to bf16 once, compacting the object in place.
//
// Per row block i the work is S = Q_i K^T (MAC unit), the row maxima, P = 2^(S - m) (vector unit),
// the row sums and O_i += P V (MAC unit). The step for row block i also computes S of row block i+1
// (the last row block computes row block 0 of the NEXT key block, whose K the core has already
// acquired and transposes once into KT), with two S buffers in ping-pong; attn_block_main(first=1)
// computes row block 0 of the first key block of a round itself. (Moving the P V product of row block i-1 into the loop that
// exponentiates row block i, so that MAC and vector work could co-issue, was tried and lost 10 %:
// the compiler spilled instead of overlapping; see docs/02-results.md.)
//
// Q is pre-scaled by the attention scale times log2(e) (in attn_round_init), so the softmax weights are
// 2^(S - m) with no multiply, and the exp is an exponent-field exp2 on the 32-lane int16 datapath (exp2_32 below):
// floor(128 y) from the bf16 -> int conversion already holds the biased exponent and the linear term
// of the mantissa, a two-mac fixed-point quadratic adds the rest. (-DLUT_EXP restores the runtime
// library's table lookup getExpBf16, exp(int) * exp(frac), for comparison; the host then has to feed
// natural-log units.) The running max is only moved when a row's new maximum exceeds it by
// RESCALE_THRESHOLD (log2 units), so the rescale of O and l (an emulated f32 multiply on AIE2) is
// rare; between moves the weights are at most 2^RESCALE_THRESHOLD, which the f32 sums and bf16
// weights absorb.
//
// Layouts. Operand A of a bf16 matrix product (aie::mmul<8, 8, 4>, the shape AIE2 runs as two native
// 4x8x4 macs with no shuffles) is a row-major 8x8 tile; operand B is a row-major 8x4 block; the f32
// result is an 8x4 block. The memtile DMAs produce every input layout from the host's row-major
// matrices, so no kernel reorders data except for K's transpose (once per key block, below):
//   Q    (BQ x HD) bf16, 8x8 tiles: tile (i, kd) at (i * TD + kd) * 64, lane r*8+k = Q[8i+r][8kd+k]
//        (delivered as the same tiles in f32, scaled and compacted to bf16 by attn_round_init)
//   K, V (BK x HD) bf16, 8x4 tiles: tile (j, c) at (j * ND4 + c) * 32, lane r*4+n = X[8j+r][4c+n]
//   KT   (HD x BK) bf16 K transposed, 8x4 blocks: block (kd, q) at (kd * NK4 + q) * 32,
//        lane k*4+n = K[4q+n][8kd+k]; built from K once per key block (K's two tiles (j, 2kd),
//        (j, 2kd+1) are the 8x8 tile (keys 8j.., dims 8kd..) split by columns: zipping them back gives
//        keys 0-3 and 4-7 row-major, whose 4x8 transposes are the two B blocks)
//   S    2 x (8 x BK) f32 scores, 8x4 blocks: block q at q * 32, lane r*4+n = S[r][4q+n]
//   P    (8 x BK) bf16 weights, 8x8 tiles: tile j at j * 64 (the exp zips two S blocks into one)
//   O    (BQ x HD) f32 accumulator = the output, 8x4 blocks: block (i, c) at (i * ND4 + c) * 32,
//        lane r*4+n = O[8i+r][4c+n]
//   m    (TQ x 8) bf16 running max per row (a bf16 value: the max of bf16-rounded scores)
//   l    (TQ x 8) f32 running row sums
//   mx   (32) bf16: the elementwise max over the blocks of the row block waiting in S (carried
//        between calls)
// Per-row values are used in "block form", the shape of one 8x4 block (lane r*4+n holds row r), so
// they combine with S/O blocks without lane shuffling; the row sums come out of an mmul of P with a
// block of ones in exactly that form.
//
// Borrowed from mlir-aie (Apache-2.0 WITH LLVM-exception): zero_vectorized (aie_kernels/common/zero.h)
// and, under -DLUT_EXP, getExpBf16 (aie_runtime_lib/AIE2/lut_based_ops.h, compiled through
// aie_kernels/common/lut_kernel.cc so its tables are linked).
//
//===----------------------------------------------------------------------===//

#include "common/zero.h"

#include <aie_api/aie.hpp>
#ifdef LUT_EXP
#include <lut_based_ops.h>
#endif
#include <stdint.h>

#ifndef BQ
#define BQ 64
#endif
#ifndef BK
#define BK 64
#endif
#ifndef HD
#define HD 40
#endif
#ifndef RESCALE_THRESHOLD
#define RESCALE_THRESHOLD 8.0f // log2 units: weights stay below 2^8 = 256
#endif
// Experiment switches (timing only, results are wrong): SKIP_EXP, SKIP_PV, SKIP_ALL.

using bf16 = bfloat16;
using MM84 = aie::mmul<8, 8, 4, bf16, bf16, accauto>; // A 8x8 tile, B 8x4 block, C 8x4 f32 block
using Acc32f = aie::accum<accfloat, 32>;
using Acc16 = aie::accum<accfloat, 16>;
using V64bf = aie::vector<bf16, 64>;
using V8bf = aie::vector<bf16, 8>;
using V8f = aie::vector<float, 8>;
using V32bf = aie::vector<bf16, 32>;
using V32f = aie::vector<float, 32>;
using V32i = aie::vector<int16, 32>;

constexpr int TQ = BQ / 8; // row blocks of the query block
constexpr int TK = BK / 8; // column blocks of the key block
constexpr int TD = HD / 8; // Q tiles along the head dimension
constexpr int ND4 = HD / 4; // 8x4 blocks along the head dimension (K, V, O)
constexpr int NK4 = BK / 4; // 8x4 blocks along the keys (KT, S)
constexpr int SROW = NK4 * 32; // floats in one S half
static_assert(BQ % 16 == 0, "BQ must be a multiple of 16 (an even number of row blocks)");
static_assert(BK % 16 == 0, "BK must be a multiple of 16 (scores are computed four 8x4 blocks at a time)");
static_assert(HD % 8 == 0, "HD must be a multiple of 8 (one mmul tile)");

static const bf16 kLowest = bf16(-0x1.FEp127f);

#ifdef LUT_EXP
// e^x of 32 bf16 lanes, as bf16 (runtime library table lookup).
static inline __attribute__((always_inline)) V32bf exp2_32(const V32bf &x) {
  Acc16 e0(getExpBf16(x.template extract<16>(0)));
  Acc16 e1(getExpBf16(x.template extract<16>(1)));
  return aie::concat(e0.template to_vector<bf16>(), e1.template to_vector<bf16>());
}
#else
// 2^y of 32 bf16 lanes, as bf16, for y <= RESCALE_THRESHOLD (anything below -126 gives 2^-126).
// With y = k + f (k = floor(y), 0 <= f < 1) the bf16 bit pattern of 2^y is
//   128 (k + 127) + round(128 (2^f - 1)) = [128 k + floor(128 f)] + round(128 (r(f) + 127)),
// where r(f) = 2^f - 1 - f is the part of the mantissa the linear term misses (|r| < 0.09). The first
// bracket is the low 16 bits of floor(128 y), which bfloat16_to_int (it floors whatever crRnd says)
// delivers for 16 lanes at a time, and r is a quadratic in the 7 fraction bits of that same number:
// t = a f + b (Q15) and r + 127 = t f + c (Q22, the +127 bias folded into c), two int16 macs with an
// srs each; the fit is centred for the floor of the linear term. Over every bf16 input the error is
// at most 1 bf16 ulp (median 0.3), against ~1.5 for the two-table exp(int) * exp(frac) lookup, in 10
// vector ops per 32 lanes (max, 2 conversions, shuffle, band, 2 macs, 2 srs, add) with no table.
#define EXP2_A 11315      // round(0.3453 * 2^15)
#define EXP2_B -1437522   // round(-0.3427 * 2^22)
#define EXP2_C 532692367  // round((127 + 0.0063) * 2^22)
static inline __attribute__((always_inline)) V32bf exp2_32(V32bf y) {
  y = aie::max(y, aie::broadcast<bf16, 32>(bf16(-126.0f)));
  const V32i lin = V32i(::shuffle((v32int16)bfloat16_to_int(y.template extract<16>(0), 7),
                                  (v32int16)bfloat16_to_int(y.template extract<16>(1), 7), T16_32x2_lo));
  const V32i f7 = aie::bit_and(lin, aie::broadcast<int16, 32>(0x7F));
  aie::accum<acc32, 32> t(aie::broadcast<int32, 32>(EXP2_B));
  t = aie::mac(t, f7, aie::broadcast<int16, 32>(EXP2_A));
  aie::accum<acc32, 32> e(aie::broadcast<int32, 32>(EXP2_C));
  e = aie::mac(e, t.template to_vector<int16>(7), f7);
  return aie::add(lin, e.template to_vector<int16>(15)).template cast_to<bf16>();
}
#endif

// bf16 block form (lane r*4+n = v[r]) of 8 per-row values: four copies as the rows of a 4x8 matrix,
// transposed.
static inline __attribute__((always_inline)) V32bf bcast4(const V8bf &v) {
  return aie::transpose(aie::concat(v, v, v, v), 4, 8);
}

// f32 block form of 8 per-row bf16 values.
static inline __attribute__((always_inline)) V32f bcast4f(const V8bf &v) {
  Acc32f a;
  a.from_vector(bcast4(v));
  return a.template to_vector<float>();
}

// lane r*4+n = p[r] for 8 floats.
static inline __attribute__((always_inline)) V32f bcast4_rows(const float *p) {
  V32f v;
#pragma clang loop unroll(full)
  for (int r = 0; r < 8; r++)
    v.insert(r, aie::broadcast<float, 4>(p[r]));
  return v;
}

// Per-row max (lane r) of a block-form bf16 vector: transpose so a row's lanes become a column, then
// fold the four columns.
static inline __attribute__((always_inline)) V8bf row_max4(const V32bf &mx) {
  const V32bf t = aie::transpose(mx, 8, 4);
  const aie::vector<bf16, 16> h = aie::max(t.template extract<16>(0), t.template extract<16>(1));
  return aie::max(h.template extract<8>(0), h.template extract<8>(1));
}

// Lane r of a block-form f32 vector whose four lanes per row are equal.
static inline __attribute__((always_inline)) V8f rows4(const V32f &v) {
  return aie::filter_even(aie::filter_even(v, 1), 1);
}

// KT = K^T in 8x4 blocks, once per key block (see the layouts above).
static inline __attribute__((always_inline)) void transpose_k(const bf16 *__restrict K, bf16 *__restrict KT) {
  for (int kd = 0; kd < TD; kd++) {
#pragma clang loop unroll(full)
    for (int j = 0; j < TK; j++) {
      const V64bf v = aie::load_v<64>(K + (j * ND4 + 2 * kd) * 32);
      const auto [lo, hi] = aie::interleave_zip(v.template extract<32>(0), v.template extract<32>(1), 4);
      aie::store_v(KT + (kd * NK4 + 2 * j) * 32, aie::transpose(lo, 4, 8));
      aie::store_v(KT + (kd * NK4 + 2 * j + 1) * 32, aie::transpose(hi, 4, 8));
    }
  }
}

// S = Q_i K^T for row block Qi: the NK4 score blocks four at a time (four independent accumulator
// chains over the head dimension), f32 into S; the block max mx takes their bf16 rounding.
static inline __attribute__((always_inline)) void
qk_rowblock(const bf16 *__restrict Qi, const bf16 *__restrict KT, float *__restrict S, V32bf &mx) {
  for (int q = 0; q < NK4; q += 4) {
    const bf16 *__restrict kt = KT + q * 32;
    V64bf a = aie::load_v<64>(Qi);
    MM84 c0, c1, c2, c3;
    c0.mul(a, aie::load_v<32>(kt));
    c1.mul(a, aie::load_v<32>(kt + 32));
    c2.mul(a, aie::load_v<32>(kt + 64));
    c3.mul(a, aie::load_v<32>(kt + 96));
    // Unrolled by 4 rather than fully: at TD = 20 (d = 160) the full unroll crashes Peano's frontend.
#pragma clang loop unroll_count(4)
    for (int kd = 1; kd < TD; kd++) {
      a = aie::load_v<64>(Qi + kd * 64);
      const bf16 *__restrict k = kt + kd * NK4 * 32;
      c0.mac(a, aie::load_v<32>(k));
      c1.mac(a, aie::load_v<32>(k + 32));
      c2.mac(a, aie::load_v<32>(k + 64));
      c3.mac(a, aie::load_v<32>(k + 96));
    }
    aie::store_v(S + q * 32, c0.template to_vector<float>());
    aie::store_v(S + (q + 1) * 32, c1.template to_vector<float>());
    aie::store_v(S + (q + 2) * 32, c2.template to_vector<float>());
    aie::store_v(S + (q + 3) * 32, c3.template to_vector<float>());
    mx = aie::max(mx, aie::max(aie::max(c0.template to_vector<bf16>(), c1.template to_vector<bf16>()),
                               aie::max(c2.template to_vector<bf16>(), c3.template to_vector<bf16>())));
  }
}

// O_i += P V for row block i: the ND4 output blocks in groups of G (independent accumulator chains),
// each summed over the TK key tiles of P.
static inline __attribute__((always_inline)) void
pv_rowblock(const bf16 *__restrict P, const bf16 *__restrict V, float *__restrict Oi) {
  constexpr int G = (ND4 % 4 == 0) ? 4 : 2;
  for (int c = 0; c < ND4; c += G) {
    const bf16 *__restrict v = V + c * 32;
    MM84 y0(aie::load_v<32>(Oi + c * 32));
    MM84 y1(aie::load_v<32>(Oi + (c + 1) * 32));
    if constexpr (G == 4) {
      MM84 y2(aie::load_v<32>(Oi + (c + 2) * 32));
      MM84 y3(aie::load_v<32>(Oi + (c + 3) * 32));
#pragma clang loop unroll(full)
      for (int j = 0; j < TK; j++) {
        const V64bf pj = aie::load_v<64>(P + j * 64);
        const bf16 *__restrict vj = v + j * ND4 * 32;
        y0.mac(pj, aie::load_v<32>(vj));
        y1.mac(pj, aie::load_v<32>(vj + 32));
        y2.mac(pj, aie::load_v<32>(vj + 64));
        y3.mac(pj, aie::load_v<32>(vj + 96));
      }
      aie::store_v(Oi + (c + 2) * 32, y2.template to_vector<float>());
      aie::store_v(Oi + (c + 3) * 32, y3.template to_vector<float>());
    } else {
#pragma clang loop unroll(full)
      for (int j = 0; j < TK; j++) {
        const V64bf pj = aie::load_v<64>(P + j * 64);
        const bf16 *__restrict vj = v + j * ND4 * 32;
        y0.mac(pj, aie::load_v<32>(vj));
        y1.mac(pj, aie::load_v<32>(vj + 32));
      }
    }
    aie::store_v(Oi + c * 32, y0.template to_vector<float>());
    aie::store_v(Oi + (c + 1) * 32, y1.template to_vector<float>());
  }
}

// alpha = 2^(m_old - m_new) in f32 block form; out of line, it runs only when a row's max moves.
static __attribute__((noinline)) V32f rescale_factor(const V8bf &m_old, const V8bf &m_new) {
  Acc32f d;
  d.from_vector(bcast4(m_old));
  d = aie::sub(d, bcast4f(m_new));
  Acc32f e;
  e.from_vector(exp2_32(d.template to_vector<bf16>()));
  return e.template to_vector<float>();
}

// Row block i: its softmax step from the scores in S_cur (block max mx_cur), fused with the scores of
// the look-ahead row block (Qn against KT) into S_next; then O_i += P V. One copy in program memory
// (both entry points call it), hence noinline.
static __attribute__((noinline)) void
row_block_step(const float *__restrict S_cur, const V32bf &mx_cur, const bf16 *__restrict Qn,
               const bf16 *__restrict KT, float *__restrict S_next, V32bf &mx_next,
               const bf16 *__restrict V, float *__restrict Oi, bf16 *__restrict mi, float *__restrict li,
               bf16 *__restrict P) {
  // Running max per row: moved only where this block's row max beats it by more than the threshold.
  const V8bf rm = row_max4(mx_cur);
  const V8bf m_old = aie::load_v<8>(mi);
  aie::accum<accfloat, 8> lim(m_old);
  lim = aie::add(lim, aie::broadcast<float, 8>(RESCALE_THRESHOLD));
  const aie::mask<8> grow = aie::gt(rm, lim.template to_vector<bf16>());
  const bool moved = grow.to_uint32() != 0;
  V32f alpha;
  V8bf m_new = m_old;
  if (moved) {
    m_new = aie::select(m_old, rm, grow);
    alpha = rescale_factor(m_old, m_new);
    aie::store_v(mi, m_new);
  }
  const V32f mb = bcast4f(m_new);
  const V32bf ones = aie::broadcast<bf16, 32>(1.0f);

  // The next row block's scores (MAC unit).
  mx_next = aie::broadcast<bf16, 32>(kLowest);
  qk_rowblock(Qn, KT, S_next, mx_next);

  // P = 2^(S - m): per iteration the two S blocks of 4 keys that make one 8x8 P tile, zipped together
  // on the way out. A short body the compiler software-pipelines.
  const float *__restrict sp = S_cur;
  bf16 *__restrict pp = P;
#pragma clang loop min_iteration_count(4)
  for (int j = 0; j < TK; j++) {
    Acc32f xa, xb;
    xa.from_vector(aie::load_v<32>(sp));
    xb.from_vector(aie::load_v<32>(sp + 32));
    sp += 64;
    xa = aie::sub(xa, mb);
    xb = aie::sub(xb, mb);
#ifdef SKIP_EXP
    const V32bf ea = xa.template to_vector<bf16>(), eb = xb.template to_vector<bf16>();
#else
    const V32bf ea = exp2_32(xa.template to_vector<bf16>()), eb = exp2_32(xb.template to_vector<bf16>());
#endif
    const auto [lo, hi] = aie::interleave_zip(ea, eb, 4);
    aie::store_v(pp, lo);
    aie::store_v(pp + 32, hi);
    pp += 64;
  }
  // Row sums in block form: P times a block of ones.
  MM84 ls(aie::zeros<float, 32>());
  for (int j = 0; j < TK; j++)
    ls.mac(aie::load_v<64>(P + j * 64), ones);
  const V8f sums = rows4(ls.template to_vector<float>());

  // l = alpha * l + sums; O = alpha * O where the max moved.
  aie::accum<accfloat, 8> lacc(sums);
  if (moved) {
    lacc = aie::mac(lacc, rows4(alpha), aie::load_v<8>(li));
    for (int c = 0; c < ND4; c++) {
      Acc32f o = aie::mul(aie::load_v<32>(Oi + c * 32), alpha);
      aie::store_v(Oi + c * 32, o.template to_vector<float>());
    }
  } else {
    lacc = aie::add(lacc, aie::load_v<8>(li));
  }
  aie::store_v(li, lacc.template to_vector<float>());
#ifndef SKIP_PV
  pv_rowblock(P, V, Oi);
#endif
}

extern "C" {

// q_scale_bits: the f32 bit pattern of the attention scale times log2(e). Q is read as f32 and written
// back as bf16 in the same buffer: chunk k reads elements [32k, 32k+32) and writes bf16 elements
// [32k, 32k+32), i.e. bytes [64k, 64k+64), which only ever cover f32 input already consumed.
void attn_round_init(float *Q, float *__restrict O, bf16 *__restrict m, float *__restrict l,
                     int32_t q_scale_bits) {
  event0();
  aie::rounding_mode saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  float q_scale;
  __builtin_memcpy(&q_scale, &q_scale_bits, sizeof q_scale);
  bf16 *Qb = reinterpret_cast<bf16 *>(Q);
  for (int i = 0; i < BQ * HD; i += 32) {
    const V32f x = aie::load_v<32>(Q + i);
    Acc32f a = aie::mul(x, q_scale);
    aie::store_v(Qb + i, a.template to_vector<bf16>());
  }
  aie::set_rounding(saved_rounding);
  zero_vectorized<float, BQ, HD, false>(O);
  const aie::vector<bf16, 16> neg = aie::broadcast<bf16, 16>(kLowest);
  const aie::vector<float, 16> zero = aie::zeros<float, 16>();
  for (int i = 0; i < TQ * 8; i += 16) {
    aie::store_v(m + i, neg);
    aie::store_v(l + i, zero);
  }
  event1();
}

// Row blocks 0 .. TQ-2 of one key block. With first != 0 (the round's first key block) K is transposed
// into KT and the scores of row block 0 are computed here; otherwise the previous call's look-ahead
// left KT, the scores in S[0] and their block max in mx.
void attn_block_main(const bf16 *__restrict Q, const bf16 *__restrict K, const bf16 *__restrict V,
                     float *__restrict O, bf16 *__restrict m, float *__restrict l, float *__restrict S,
                     bf16 *__restrict P, bf16 *__restrict mx, bf16 *__restrict KT, int32_t first) {
  event0();
#ifdef SKIP_ALL
  event1();
  return;
#endif
  aie::rounding_mode saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  V32bf mx_cur;
  if (first) {
    transpose_k(K, KT);
    mx_cur = aie::broadcast<bf16, 32>(kLowest);
    qk_rowblock(Q, KT, S, mx_cur);
  } else {
    mx_cur = aie::load_v<32>(mx);
  }
  float *S_cur = S;
  float *S_next = S + SROW;
  for (int i = 0; i < TQ - 1; i++) {
    V32bf mx_next;
    row_block_step(S_cur, mx_cur, Q + (i + 1) * TD * 64, KT, S_next, mx_next, V, O + i * ND4 * 32,
                   m + i * 8, l + i * 8, P);
    float *t = S_cur;
    S_cur = S_next;
    S_next = t;
    mx_cur = mx_next;
  }
  aie::store_v(mx, mx_cur);
  aie::set_rounding(saved_rounding);
  event1();
}

// Row block TQ-1 of one key block; the look-ahead transposes Knext into KT (the current block's scores
// are all in S by now) and computes row block 0 of the next key block from it. Without a next block
// Knext is any valid block; the scores it produces are never read.
void attn_block_last(const bf16 *__restrict Q, const bf16 *__restrict Knext, const bf16 *__restrict V,
                     float *__restrict O, bf16 *__restrict m, float *__restrict l, float *__restrict S,
                     bf16 *__restrict P, bf16 *__restrict mx, bf16 *__restrict KT) {
  event0();
#ifdef SKIP_ALL
  event1();
  return;
#endif
  aie::rounding_mode saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  constexpr int i = TQ - 1;
  transpose_k(Knext, KT);
  // After TQ-1 swaps in attn_block_main the current scores sit in S[1] (TQ is even), so the look-ahead
  // lands in S[0], where the next call expects them.
  V32bf mx_next;
  row_block_step(S + SROW, aie::load_v<32>(mx), Q, KT, S, mx_next, V, O + i * ND4 * 32, m + i * 8,
                 l + i * 8, P);
  aie::store_v(mx, mx_next);
  aie::set_rounding(saved_rounding);
  event1();
}

void attn_round_finish(float *__restrict O, const float *__restrict l) {
  event0();
  alignas(32) float inv[8];
  for (int i = 0; i < TQ; i++) {
    aie::store_v(inv, aie::inv(aie::load_v<8>(l + i * 8)));
    const V32f ib = bcast4_rows(inv);
    float *__restrict Oi = O + i * ND4 * 32;
    for (int c = 0; c < ND4; c++) {
      Acc32f o = aie::mul(aie::load_v<32>(Oi + c * 32), ib);
      aie::store_v(Oi + c * 32, o.template to_vector<float>());
    }
  }
  event1();
}

} // extern "C"

#ifdef UNIFIED
//===----------------------------------------------------------------------===//
// The unified program (npu/xdna1_program.py): the same cores also run the GEMMs, so the NPU never switches
// between a GEMM program and an attention program. Both modes share the core's three object streams:
//
//   Y  (BQ x 32) bf16, 8x8 tiles: tile (i, j) at (i * 4 + j) * 64   GEMM: an A tile (32 columns of K)
//                                                                  attention: a strip of the round's Q (below)
//   X  (BK x HD) bf16, 8x4 tiles (the K/V layout above)            GEMM: a B tile (BK rows of K)
//   Z  (BQ x HD) f32, 8x4 blocks (the O layout above)              GEMM: a C tile
//
// GEMM: C (BQ x HD) = A (BQ x K) @ B (K x HD), accumulated over BK / 32 Y objects per X object. The C tile is
// kept in the two 5 KB buffers the attention uses for Q and K^T (row blocks [0, TQ/2) and [TQ/2, TQ)), which GEMM
// mode does not otherwise touch, and copied into the output object once complete, so the core works on the next
// tile while the DMA drains the last one (there is no room for a second output object). The C tile passes
// through registers once per gemm_step call: with -DGEMM_KX=2 (BK = 32) a call takes two Y and two X objects
// (64 of K), otherwise one Y object (32 of K).
//
// Q: a row of cores receives the Q of all four columns' pairs (four consecutive heads) as one (BQ x 4*HD) f32
// slab, sent in strips of 16 columns. A strip travels as a Y object: the memtile's 8x8 bf16 retiling moves it
// in 8-row x 4-float pieces, piece (i, j) at (i * 4 + j) * 32 floats. Pieces j = 2p and 2p + 1 are the left and
// right halves of one 8x8 Q tile of one head; each core converts those of its own column's head into Qb.
//===----------------------------------------------------------------------===//

static_assert(BK % 32 == 0, "the unified program needs BK to be a multiple of the 32-column A strip");
static_assert(TQ % 4 == 0, "the GEMM processes row blocks in pairs within each half of the C tile");
constexpr int GA = 4;               // 8x8 A tiles along K in one Y object
constexpr int CHALF = BQ * HD / 2;  // floats in one half of the C tile

#ifndef GEMM_KX
#define GEMM_KX 1 // X objects (B tiles) per gemm_step call: 1, or 2 (BK = 32: K step 64, two Y and two X objects)
#endif
static_assert(GEMM_KX == 1 || (GEMM_KX == 2 && BK == 32), "GEMM_KX = 2 pairs one Y object with each X object");

// One 32-column K segment of the row pair: acc (row blocks i, i+1 x column blocks c, c+1) += A0 / A1 (8 x 32) @
// B (32 x 8, column blocks c and c+1 of a 32-row B segment).
static inline __attribute__((always_inline)) void
gemm_seg(MM84 &y00, MM84 &y01, MM84 &y10, MM84 &y11, const bf16 *__restrict A0, const bf16 *__restrict A1,
         const bf16 *__restrict B, int c) {
#pragma clang loop unroll(full)
  for (int j = 0; j < GA; j++) {
    const V64bf a0 = aie::load_v<64>(A0 + j * 64);
    const V64bf a1 = aie::load_v<64>(A1 + j * 64);
    const bf16 *__restrict b = B + (j * ND4 + c) * 32;
    const V32bf b0 = aie::load_v<32>(b);
    const V32bf b1 = aie::load_v<32>(b + 32);
    y00.mac(a0, b0);
    y01.mac(a0, b1);
    y10.mac(a1, b0);
    y11.mac(a1, b1);
  }
}

// C0 / C1 (8 x HD, row blocks i and i+1) += A0 / A1 (8 x 32 per segment) @ B (32 x HD per segment), over GEMM_KX
// segments (A0[s], A1[s] against B[s]): two row blocks by two column blocks per step, four independent accumulator
// chains; each A tile serves two B blocks and vice versa. The C blocks go through registers once per call, i.e.
// once per 32 * GEMM_KX of K.
static inline __attribute__((always_inline)) void
gemm_rowpair(const bf16 *const A0[GEMM_KX], const bf16 *const A1[GEMM_KX], const bf16 *const B[GEMM_KX],
             float *__restrict C0, float *__restrict C1) {
  for (int c = 0; c < ND4; c += 2) {
    MM84 y00(aie::load_v<32>(C0 + c * 32));
    MM84 y01(aie::load_v<32>(C0 + (c + 1) * 32));
    MM84 y10(aie::load_v<32>(C1 + c * 32));
    MM84 y11(aie::load_v<32>(C1 + (c + 1) * 32));
#pragma clang loop unroll(full)
    for (int s = 0; s < GEMM_KX; s++)
      gemm_seg(y00, y01, y10, y11, A0[s], A1[s], B[s], c);
    aie::store_v(C0 + c * 32, y00.template to_vector<float>());
    aie::store_v(C0 + (c + 1) * 32, y01.template to_vector<float>());
    aie::store_v(C1 + c * 32, y10.template to_vector<float>());
    aie::store_v(C1 + (c + 1) * 32, y11.template to_vector<float>());
  }
}

extern "C" {

#if GEMM_KX == 2
// One K step of 64: C += A @ B over two Y objects (Aa, Ab: 32 columns of K each) and the two X objects that hold
// the matching 32 rows of B (Ba, Bb); C split over Clo / Chi (bf16 buffers used as f32).
void gemm_step(const bf16 *__restrict Aa, const bf16 *__restrict Ab, const bf16 *__restrict Ba,
               const bf16 *__restrict Bb, bf16 *__restrict Clo, bf16 *__restrict Chi) {
  event0();
  float *lo = reinterpret_cast<float *>(Clo);
  float *hi = reinterpret_cast<float *>(Chi);
  const bf16 *const B[2] = {Ba, Bb};
  for (int i = 0; i < TQ; i += 2) {
    float *Ci = i < TQ / 2 ? lo + i * ND4 * 32 : hi + (i - TQ / 2) * ND4 * 32;
    const bf16 *const A0[2] = {Aa + i * GA * 64, Ab + i * GA * 64};
    const bf16 *const A1[2] = {Aa + (i + 1) * GA * 64, Ab + (i + 1) * GA * 64};
    gemm_rowpair(A0, A1, B, Ci, Ci + ND4 * 32);
  }
  event1();
}
#else
// One K step of 32: C += A @ B[32 * half .. 32 * half + 32), C split over Clo / Chi (bf16 buffers used as f32).
void gemm_step(const bf16 *__restrict A, const bf16 *__restrict B, bf16 *__restrict Clo, bf16 *__restrict Chi,
               int32_t half) {
  event0();
  const bf16 *const Bh[1] = {B + half * GA * ND4 * 32};
  float *lo = reinterpret_cast<float *>(Clo);
  float *hi = reinterpret_cast<float *>(Chi);
  for (int i = 0; i < TQ; i += 2) {
    float *Ci = i < TQ / 2 ? lo + i * ND4 * 32 : hi + (i - TQ / 2) * ND4 * 32;
    const bf16 *const A0[1] = {A + i * GA * 64};
    const bf16 *const A1[1] = {A + (i + 1) * GA * 64};
    gemm_rowpair(A0, A1, Bh, Ci, Ci + ND4 * 32);
  }
  event1();
}
#endif

void gemm_zero(bf16 *__restrict Clo, bf16 *__restrict Chi) {
  event0();
  zero_vectorized<float, BQ / 2, HD, false>(reinterpret_cast<float *>(Clo));
  zero_vectorized<float, BQ / 2, HD, false>(reinterpret_cast<float *>(Chi));
  event1();
}

void gemm_store(const bf16 *__restrict Clo, const bf16 *__restrict Chi, float *__restrict C) {
  event0();
  const float *lo = reinterpret_cast<const float *>(Clo);
  const float *hi = reinterpret_cast<const float *>(Chi);
  for (int i = 0; i < CHALF; i += 32)
    aie::store_v(C + i, aie::load_v<32>(lo + i));
  for (int i = 0; i < CHALF; i += 32)
    aie::store_v(C + CHALF + i, aie::load_v<32>(hi + i));
  event1();
}

// attn_round_init without Q (which now arrives through attn_q_strip): O = 0, m = -inf, l = 0.
void attn_round_init_o(float *__restrict O, bf16 *__restrict m, float *__restrict l) {
  event0();
  zero_vectorized<float, BQ, HD, false>(O);
  const aie::vector<bf16, 16> neg = aie::broadcast<bf16, 16>(kLowest);
  const aie::vector<float, 16> zero = aie::zeros<float, 16>();
  for (int i = 0; i < TQ * 8; i += 16) {
    aie::store_v(m + i, neg);
    aie::store_v(l + i, zero);
  }
  event1();
}

// Strip s (slab columns [16s, 16s + 16)) of the round's Q: the 8x8 tiles in it that belong to head `col` (this
// core's column), scaled by the attention scale times log2(e) and rounded to bf16 into Qb.
void attn_q_strip(const bf16 *__restrict Y, bf16 *__restrict Qb, int32_t s, int32_t col, int32_t q_scale_bits) {
  event0();
  float q_scale;
  __builtin_memcpy(&q_scale, &q_scale_bits, sizeof q_scale);
  const float *__restrict Yf = reinterpret_cast<const float *>(Y);
  aie::rounding_mode saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  const int head0 = col * HD; // this core's head in the slab: columns [head0, head0 + HD)
  for (int p = 0; p < 2; p++) {
    const int c0 = 16 * s + 8 * p - head0; // column of the tile's first element in this core's head
    if (c0 < 0 || c0 >= HD)
      continue;
    bf16 *__restrict q = Qb + (c0 >> 3) * 64;
    for (int i = 0; i < TQ; i++) {
      const V32f a = aie::load_v<32>(Yf + (i * 4 + 2 * p) * 32);
      const V32f b = aie::load_v<32>(Yf + (i * 4 + 2 * p + 1) * 32);
      const auto [r03, r47] = aie::interleave_zip(a, b, 4); // rows 0-3 and 4-7 of the 8x8 tile
      Acc32f x0 = aie::mul(r03, q_scale);
      Acc32f x1 = aie::mul(r47, q_scale);
      aie::store_v(q + i * TD * 64, x0.template to_vector<bf16>());
      aie::store_v(q + i * TD * 64 + 32, x1.template to_vector<bf16>());
    }
  }
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
#endif // UNIFIED
