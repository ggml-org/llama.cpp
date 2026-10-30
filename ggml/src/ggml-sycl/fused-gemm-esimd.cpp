#include "fused-gemm-esimd.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// One thread owns MR rows of one expert's weight and one BN-wide tile of the schedule, over the
// whole K. Per k step (32 values) it loads the rows' stored bits, dequantizes them in registers to
// the f16 A operand of dpas (RC = 8 rows x 16 k, row major) and multiplies it with the packed B,
// which is already the dpas VNNI operand: [K/2][Npad][2] of f16 or bf16, 16 columns a row.
//
// The joint_matrix kernel dequantizes into SLM and reads it back in the same sub-group, which is
// staging, not sharing; dpas takes registers, so that round trip and its barriers are gone here.
// Its K split over GGML_SYCL_GG_KSPLIT sub-groups is reproduced as sequential K ranges summed in
// the same ascending order as its tile_c reduction, and every format below rounds A exactly as its
// fg_soa stage does, so the output is the same bits.
//
// A weight format is a gg_esimd_a<block> with: raw, the stored bits of the MR rows for one k step;
// load(), which fetches them from the reorder (SoA) slice; and decode(), the f16 A of one row group.
// It reads the same streams at the same offsets as the fg_soa stage of its format.
static constexpr int GGML_SYCL_GG_KSPLIT = 4; // FG_KSPLIT of the joint_matrix kernel
static constexpr int GGML_SYCL_GG_BN     = GGML_SYCL_FG_BN;
static constexpr int GG_MR               = 32; // rows of one thread: one 2D load covers them
static constexpr int GG_RC               = 8;  // rows of one dpas
static constexpr int GG_QK               = 32; // values of one k step

namespace gg_esimd {
using namespace sycl::ext::intel::esimd;

// f(std::integral_constant<int, I>) for I in [0, N): register selects at a runtime offset become
// indirect accesses, so loops whose index picks registers are spelled out at compile time
template <int I, int N> struct static_for {
    template <typename Fn> static ESIMD_INLINE void run(Fn & f) {
        if constexpr (I < N) {
            f(std::integral_constant<int, I>{});
            static_for<I + 1, N>::run(f);
        }
    }
};

// Where one thread's rows sit in an expert's reorder slice. nblocks counts the stored blocks of the
// slice, bpr those of one row.
struct slice {
    const uint8_t * x;
    int             M;
    int             bpr;
    int             nblocks;
    int             m0;

    // byte offsets of block ib of each of the MR rows in a stream of `size` bytes per block
    ESIMD_INLINE simd<uint32_t, GG_MR> offsets(const int ib, const int size) const {
        simd<uint32_t, GG_MR> o(0, (uint32_t) (bpr * size));
        return o + (uint32_t) ((m0 * bpr + ib) * size);
    }

    // a byte of each row at offsets o of stream s, read as part of its aligned dword
    ESIMD_INLINE simd<uint32_t, GG_MR> bytes(const uint8_t * s, simd<uint32_t, GG_MR> o) const {
        simd<uint32_t, GG_MR> w = gather<uint32_t, GG_MR>((const uint32_t *) s, o & ~3u);
        return (w >> ((o & 3u) * 8)) & 0xffu;
    }

    // the MR x W dwords at dword column x of a stream whose rows are `row_bytes` long
    template <int W> ESIMD_INLINE simd<uint32_t, GG_MR * W> rect(const uint8_t * s, const unsigned row_bytes, const int x) const {
        return load_2d<uint32_t, W, GG_MR>((const uint32_t *) s, row_bytes - 1, M - 1, row_bytes - 1, x, m0);
    }
};

// A decode returns its row group's A in the order the two dpas of a k step read it, [k half][row][16
// k]: from the code bytes that order needs only reads with a stride, while [row][32 k] needed
// strided writes, which split into SIMD4.
using a_halves = simd<sycl::half, GG_RC * GG_QK>;

// A 2D surface: rows `pitch` bytes apart, 16-byte aligned and at least 64 bytes
constexpr bool surface_ok(const int pitch) {
    return pitch % 16 == 0 && pitch >= 64;
}

// byte i of each lane's dword, i a runtime index: a shift by a scalar, not a register pick
ESIMD_INLINE simd<uint32_t, GG_MR> byte_of(simd<uint32_t, GG_MR> w, const int i) {
    return (w >> (8 * i)) & 0xffu;
}

// one of four per-lane dwords by a runtime index, as uniform branches rather than a register pick
ESIMD_INLINE simd<uint32_t, GG_MR> dword_of(simd<uint32_t, GG_MR> * w, const int i) {
    simd<uint32_t, GG_MR> r = w[0];
    if (i == 1) {
        r = w[1];
    } else if (i == 2) {
        r = w[2];
    } else if (i == 3) {
        r = w[3];
    }
    return r;
}

// a row group's 32 k-ordered code bytes a row as f32, in [k half][row][16 k]
ESIMD_INLINE simd<float, GG_RC * GG_QK> byte_codes(simd<uint32_t, GG_RC * 8> q) {
    auto                       bm = q.template bit_cast_view<uint8_t, GG_RC, GG_QK>();
    simd<float, GG_RC * GG_QK> v;
    v.template select<GG_RC * 16, 1>(0)          = bm.template select<GG_RC, 1, 16, 1>(0, 0);
    v.template select<GG_RC * 16, 1>(GG_RC * 16) = bm.template select<GG_RC, 1, 16, 1>(0, 16);
    return v;
}

// f16 of d * code - m in f32, each row's d and m over its 16 values of a k half: the expression of
// the K-quant stages with a min
ESIMD_INLINE simd<sycl::half, GG_RC * 16> scale_min_half(simd<float, GG_RC * 16> v, simd<float, GG_RC> d,
                                                         simd<float, GG_RC> m) {
    v = v * d.template replicate_vs_w_hs<GG_RC, 1, 16, 0>(0) - m.template replicate_vs_w_hs<GG_RC, 1, 16, 0>(0);
    return v;
}

// the two k halves of a row group, the code bytes times one (d, m) per row and half
ESIMD_INLINE a_halves halves_dm(simd<uint32_t, GG_RC * 8> q, simd<float, GG_RC> d0, simd<float, GG_RC> m0,
                                simd<float, GG_RC> d1, simd<float, GG_RC> m1, const float bias = 0.0f) {
    simd<float, GG_RC * GG_QK> c = byte_codes(q);
    if (bias != 0.0f) {
        c -= bias;
    }
    a_halves v;
    v.template select<GG_RC * 16, 1>(0)          = scale_min_half(c.template select<GG_RC * 16, 1>(0), d0, m0);
    v.template select<GG_RC * 16, 1>(GG_RC * 16) = scale_min_half(c.template select<GG_RC * 16, 1>(GG_RC * 16), d1, m1);
    return v;
}
} // namespace gg_esimd

template <typename block_q_t> struct gg_esimd_a {
    static constexpr bool supported = false;
};

// The K-quants: a 256-value superblock, k step j its sub-block j. Their value bytes for a k step
// are one rectangle of a stream, in k order after a shift and a mask per dword; scales, mins and d
// are the block's and are loaded once per block.

// [qs 128][scales 12][dm]: value l of sub-block j is nibble j & 1 of byte 32 (j / 2) + l, times
// dall * scale_j, minus dmin * min_j (6-bit scale and min, get_scale_min_k4)
template <> struct gg_esimd_a<block_q4_K> {
    static constexpr bool supported = true;
    static constexpr int  qk        = QK_K;
    static constexpr bool whole     = false;
    static constexpr int  slm_bytes = 0;
    static ESIMD_INLINE void init_slm() {}
    struct raw {
        gg_esimd::simd<uint32_t, GG_MR * 8> q;
        gg_esimd::simd<float, GG_MR>        d, m;
    };
    struct sb_t {
        gg_esimd::simd<uint32_t, GG_MR> sc[3]; // the 12 scale bytes
        gg_esimd::simd<float, GG_MR>    dall, dmin;
    };
    static bool ok(const int K) { return K % qk == 0 && gg_esimd::surface_ok(K / 2); }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib, const int qbytes = 128) {
        using namespace gg_esimd;
        const size_t          nb = s.nblocks;
        simd<uint32_t, GG_MR> so = s.offsets(ib, K_SCALE_SIZE);
        const uint32_t *      sp = (const uint32_t *) (s.x + nb * qbytes);
#pragma unroll
        for (int w = 0; w < 3; ++w) {
            sb.sc[w] = gather<uint32_t, GG_MR>(sp, so + (uint32_t) (4 * w));
        }
        simd<uint32_t, GG_MR> dm = gather<uint32_t, GG_MR>((const uint32_t *) (s.x + nb * (qbytes + K_SCALE_SIZE)), s.offsets(ib, 4));
        auto                  dh = dm.template bit_cast_view<sycl::half>();
        sb.dall                  = simd<sycl::half, GG_MR>(dh.template select<GG_MR, 2>(0));
        sb.dmin                  = simd<sycl::half, GG_MR>(dh.template select<GG_MR, 2>(1));
    }
    // get_scale_min_k4 for sub-block j over the lanes
    static ESIMD_INLINE void scale_min(sb_t & sb, const int j, gg_esimd::simd<float, GG_MR> & d,
                                       gg_esimd::simd<float, GG_MR> & m) {
        using namespace gg_esimd;
        simd<uint32_t, GG_MR> sc, mb;
        if (j < 4) {
            sc = byte_of(sb.sc[0], j) & 63;
            mb = byte_of(sb.sc[1], j) & 63;
        } else {
            sc = (byte_of(sb.sc[2], j - 4) & 0xf) | ((byte_of(sb.sc[0], j - 4) >> 6) << 4);
            mb = (byte_of(sb.sc[2], j - 4) >> 4) | ((byte_of(sb.sc[1], j - 4) >> 6) << 4);
        }
        d = sb.dall * simd<float, GG_MR>(sc);
        m = sb.dmin * simd<float, GG_MR>(mb);
    }
    static ESIMD_INLINE void load(raw & r, sb_t & sb, const gg_esimd::slice & s, const int kb, const int j) {
        using namespace gg_esimd;
        r.q = (s.rect<8>(s.x, (unsigned) s.bpr * (QK_K / 2), (kb / 8) * 32 + 8 * (j / 2)) >> (4 * (j & 1))) & 0x0F0F0F0Fu;
        scale_min(sb, j, r.d, r.m);
    }
    template <int g> static ESIMD_INLINE gg_esimd::a_halves decode(raw & r) {
        using namespace gg_esimd;
        simd<float, GG_RC> d = r.d.template select<GG_RC, 1>(g * GG_RC), m = r.m.template select<GG_RC, 1>(g * GG_RC);
        return halves_dm(r.q.template select<GG_RC * 8, 1>(g * GG_RC * 8), d, m, d, m);
    }
};

// [qs 128][qh 32][scales 12][dm]: q4_K plus bit j of qh byte l as 16
template <> struct gg_esimd_a<block_q5_K> {
    using q4                        = gg_esimd_a<block_q4_K>;
    static constexpr bool supported = true;
    static constexpr int  qk        = QK_K;
    static constexpr bool whole     = false;
    static constexpr int  slm_bytes = 0;
    static ESIMD_INLINE void init_slm() {}
    using raw  = q4::raw;
    using sb_t = q4::sb_t;
    static bool ok(const int K) { return K % qk == 0 && gg_esimd::surface_ok(K / 2) && gg_esimd::surface_ok(K / 8); }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib) {
        q4::load_sb(sb, s, ib, QK_K / 2 + QK_K / 8);
    }
    static ESIMD_INLINE void load(raw & r, sb_t & sb, const gg_esimd::slice & s, const int kb, const int j) {
        using namespace gg_esimd;
        const int             ib = kb / 8;
        simd<uint32_t, GG_MR * 8> qh = s.rect<8>(s.x + (size_t) s.nblocks * (QK_K / 2), (unsigned) s.bpr * (QK_K / 8), ib * 8);
        r.q = ((s.rect<8>(s.x, (unsigned) s.bpr * (QK_K / 2), ib * 32 + 8 * (j / 2)) >> (4 * (j & 1))) & 0x0F0F0F0Fu) |
              (((qh >> j) & 0x01010101u) << 4);
        q4::scale_min(sb, j, r.d, r.m);
    }
    template <int g> static ESIMD_INLINE gg_esimd::a_halves decode(raw & r) { return q4::template decode<g>(r); }
};

// [ql 128][qh 64][scales 16][d]: sub-block j is quarter r = j % 4 of half h = j / 4; value l is
// nibble r >> 1 of ql byte 64h + 32 (r & 1) + l with bits 2r, 2r+1 of qh byte 32h + l above it,
// minus 32, times d * the int8 scale 2j + l / 16
template <> struct gg_esimd_a<block_q6_K> {
    static constexpr bool supported = true;
    static constexpr int  qk        = QK_K;
    static constexpr bool whole     = false;
    static constexpr int  slm_bytes = 0;
    static ESIMD_INLINE void init_slm() {}
    struct raw {
        gg_esimd::simd<uint32_t, GG_MR * 8> q;
        gg_esimd::simd<float, GG_MR>        d0, d1; // per k half
    };
    struct sb_t {
        gg_esimd::simd<uint32_t, GG_MR> sc[4]; // the 16 int8 scales
        gg_esimd::simd<float, GG_MR>    d;
    };
    static bool ok(const int K) { return K % qk == 0 && gg_esimd::surface_ok(K / 2) && gg_esimd::surface_ok(K / 4); }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib) {
        using namespace gg_esimd;
        const size_t          nb = s.nblocks;
        simd<uint32_t, GG_MR> so = s.offsets(ib, QK_K / 16);
        const uint32_t *      sp = (const uint32_t *) (s.x + nb * (QK_K / 2 + QK_K / 4));
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            sb.sc[w] = gather<uint32_t, GG_MR>(sp, so + (uint32_t) (4 * w));
        }
        simd<uint16_t, GG_MR> dh = gather<uint16_t, GG_MR>((const uint16_t *) (s.x + nb * (QK_K / 2 + QK_K / 4 + QK_K / 16)),
                                                           s.offsets(ib, 2));
        sb.d = simd<sycl::half, GG_MR>(dh.template bit_cast_view<sycl::half>());
    }
    static ESIMD_INLINE void load(raw & r, sb_t & sb, const gg_esimd::slice & s, const int kb, const int j) {
        using namespace gg_esimd;
        const int ib = kb / 8, h = j / 4, q = j % 4;
        simd<uint32_t, GG_MR * 8> ql = s.rect<8>(s.x, (unsigned) s.bpr * (QK_K / 2), ib * 32 + 16 * h + 8 * (q & 1));
        simd<uint32_t, GG_MR * 8> qh = s.rect<8>(s.x + (size_t) s.nblocks * (QK_K / 2), (unsigned) s.bpr * (QK_K / 4), ib * 16 + 8 * h);
        r.q = ((ql >> (4 * (q >> 1))) & 0x0F0F0F0Fu) | (((qh >> (2 * q)) & 0x03030303u) << 4);
        // scales 2j and 2j + 1: bytes 2 (j % 2) and 2 (j % 2) + 1 of dword j / 2, sign extended
        simd<uint32_t, GG_MR> w  = dword_of(sb.sc, j / 2);
        simd<int32_t, GG_MR>  s0 = simd<int32_t, GG_MR>(w << (24 - 16 * (j % 2))) >> 24;
        simd<int32_t, GG_MR>  s1 = simd<int32_t, GG_MR>(w << (16 - 16 * (j % 2))) >> 24;
        r.d0 = sb.d * simd<float, GG_MR>(s0);
        r.d1 = sb.d * simd<float, GG_MR>(s1);
    }
    template <int g> static ESIMD_INLINE gg_esimd::a_halves decode(raw & r) {
        using namespace gg_esimd;
        simd<float, GG_RC> z = 0.0f;
        return halves_dm(r.q.template select<GG_RC * 8, 1>(g * GG_RC * 8), r.d0.template select<GG_RC, 1>(g * GG_RC), z,
                         r.d1.template select<GG_RC, 1>(g * GG_RC), z, 32.0f);
    }
};

// T is the dpas operand type, sycl::half or bfloat16: the type B is packed in. A is decoded in half
// either way and converted for bf16 (see the double rounding note at the conversion).
template <typename block_q_t, int KS, typename T> struct grouped_gemm_esimd_kernel {
    using F = gg_esimd_a<block_q_t>;

    const char *              src0_dd;
    size_t                    expert_stride;
    const ggml_sycl_gg_tile * tiles;
    const T *                 packed;
    float *                   dst;
    int                       M;
    int                       Npad;
    int                       K;
    ggml_sycl_gg_rows         dst_rows;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        namespace xmx = sycl::ext::intel::esimd::xmx;
        constexpr int MR = GG_MR;
        constexpr int RC = GG_RC;
        constexpr int RG = MR / RC; // dpas row groups of this thread
        constexpr int BN = GGML_SYCL_GG_BN;
        constexpr int NT = BN / 16;

        if constexpr (F::slm_bytes > 0) {
            // the format's lookup table; a work-group is one thread, so nothing else writes it
            slm_init<F::slm_bytes>();
            F::init_slm();
        }

        const int mblocks = M / MR;
        const int gid     = it.get_global_id(0);
        const int t       = gid / mblocks;
        const int m0      = (gid % mblocks) * MR;

        const ggml_sycl_gg_tile tile = tiles[t];
        if (tile.n1 <= tile.n0) {
            return;
        }
        const int b0 = t * BN; // the tile's packed-B columns

        // the reorder (SoA) slice of this expert
        const int             bpr = K / F::qk;
        const gg_esimd::slice sl  = { (const uint8_t *) src0_dd + (size_t) tile.expert * expert_stride, M, bpr, M * bpr, m0 };
        const uint32_t *      bp  = (const uint32_t *) packed;
        const unsigned        bw  = (unsigned) Npad * 4; // bytes of one packed B row
        const int             nstep = K / GG_QK;

        simd<float, RC * 16> acc[RG][NT];
        simd<float, RC * 16> total[KS > 1 ? RG : 1][KS > 1 ? NT : 1];
#pragma unroll
        for (int g = 0; g < RG; ++g) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                acc[g][nt] = 0.0f;
                if constexpr (KS > 1) {
                    total[g][nt] = 0.0f;
                }
            }
        }

        // The tile's route map, gathered once up front; the K loop hides its latency, and the epilogue
        // below reads it at compile-time indices.
        const int          ncols = tile.n1 - tile.n0;
        simd_mask<BN>      cm    = simd<int32_t, BN>(0, 1) < ncols;
        simd<int32_t, BN>  i2 = 0, r1 = 0; // r1: the dst row slot, i1 % ne1
        if (dst_rows.map) {
            const int32_t *    mp = (const int32_t *) (dst_rows.map + tile.n0);
            simd<uint32_t, BN> mo(0, sizeof(mmid_row_mapping));
            simd<int32_t, BN>  i1 = gather<int32_t, BN>(mp, mo, cm);
            i2                    = gather<int32_t, BN>(mp + 1, mo, cm);
            // once per thread in 32 bits: a 64-bit modulo per column is emulated, and at 20 columns a
            // tile it cost more than the whole GEMM (7.2 vs 3.1 ms)
            r1 = i1 % simd<int32_t, BN>((int32_t) dst_rows.ne1);
        }

        // One k step at a time: the format's stored bits for the MR rows, and the 16 k pairs x 16
        // columns of packed B per dpas column tile, one 2D block message each. A second stage in
        // flight spilled 7 KB of the 256 registers and ran 1.6x slower; the other threads of the
        // XVE hide the load latency instead.
        // A stored block's metadata (its sub-block scales) is loaded once per block. With KS > 1 the
        // K split of the joint_matrix kernel is reproduced for bit-identical output (a second set of
        // accumulators); KS == 1 sums the whole K in one, which is a different f32 rounding order and
        // frees the 4 KB of registers the second set holds.
        constexpr int KPB       = F::qk / GG_QK; // k steps per stored block
        int           split     = 0;
        int           split_end = KS > 1 ? nstep / KS : nstep;
        typename F::sb_t sb;
        for (int kb = 0; kb < nstep; ++kb) {
            const int j = kb % KPB;
            if (j == 0) {
                F::load_sb(sb, sl, kb / KPB);
            }
            typename F::raw r;
            F::load(r, sb, sl, kb, j);
            simd<uint32_t, 16 * 16> b[NT];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                b[nt] = load_2d<uint32_t, 16, 16>(bp, bw - 1, K / 2 - 1, bw - 1, b0 + nt * 16, kb * 16);
            }
            simd<sycl::half, MR * GG_QK> all;
            if constexpr (F::whole) {
                all = F::decode_all(r);
            }
            // one row group of the thread, g a compile-time constant: with a runtime g the larger
            // decodes were left rolled
            auto row_group = [&](auto gc) SYCL_ESIMD_FUNCTION {
                constexpr int g = decltype(gc)::value;
                gg_esimd::a_halves av;
                if constexpr (F::whole) {
                    // [row][32 k] of the whole tile to this row group's [k half][row][16 k]
                    auto am = all.template bit_cast_view<sycl::half, MR, GG_QK>();
                    av.template select<RC * 16, 1>(0)       = am.template select<RC, 1, 16, 1>(g * RC, 0);
                    av.template select<RC * 16, 1>(RC * 16) = am.template select<RC, 1, 16, 1>(g * RC, 16);
                } else {
                    av = F::template decode<g>(r);
                }
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    // A of this k16 half: rows x k [16h, 16h+16), row major
                    simd<sycl::half, RC * 16> ah = av.template select<RC * 16, 1>(h * RC * 16);
                    simd<T, RC * 16>          at;
                    if constexpr (std::is_same_v<T, sycl::half>) {
                        at = ah;
                    } else {
                        // rounded twice (f32 to half in the decode, half to bf16 here), so A's rounding
                        // error is about twice the joint_matrix bf16 stage's, which rounds f32 once
                        at = simd<float, RC * 16>(ah);
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        simd<uint32_t, 8 * 16> bh32 = b[nt].template select<8 * 16, 1>(h * 8 * 16);
                        simd<T, 16 * 16>       bh   = bh32.template bit_cast_view<T>();
                        acc[g][nt] = xmx::dpas<8, RC, float, float>(acc[g][nt], bh, at);
                    }
                }
            };
            gg_esimd::static_for<0, RG>::run(row_group);
            if constexpr (KS > 1) {
                // the K split's range ends, in k steps; a range's partial joins the total when it closes
                if (kb + 1 == split_end) {
#pragma unroll
                    for (int g = 0; g < RG; ++g) {
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt) {
                            total[g][nt] += acc[g][nt];
                            acc[g][nt] = 0.0f;
                        }
                    }
                    ++split;
                    split_end = ((split + 1) * nstep) / KS;
                }
            }
        }
        (void) split;
        (void) split_end;

        // column c of the tile: MR consecutive floats of one dst row. Unrolled, so c indexes the
        // accumulators at compile time; a runtime index sends them all to private memory.
#pragma unroll
        for (int c = 0; c < BN; ++c) {
            if (c >= ncols) {
                continue;
            }
            simd<float, MR> col;
#pragma unroll
            for (int g = 0; g < RG; ++g) {
                if constexpr (KS > 1) {
                    col.template select<RC, 1>(g * RC) = total[g][c / 16].template select<RC, 16>(c % 16);
                } else {
                    col.template select<RC, 1>(g * RC) = acc[g][c / 16].template select<RC, 16>(c % 16);
                }
            }
            const int n = tile.n0 + c;
            if (dst_rows.map) {
                float * out = (float *) (dst_rows.base + (size_t) r1[c] * dst_rows.nb1 + i2[c] * dst_rows.nb2) + m0;
                block_store<float, MR>(out, col, element_aligned_tag{});
            } else {
                block_store<float, MR>(dst + (size_t) n * M + m0, col, element_aligned_tag{});
            }
        }
    }

    auto get(syclexp::properties_tag) const { return syclexp::properties{ sycl::ext::intel::experimental::grf_size<256> }; }
};

// 0 off, 1 on (default), 2 on with the joint_matrix kernel's K-split order reproduced (bit-identical output)
static int ggml_sycl_grouped_gemm_esimd_mode() {
    static const int mode = ggml_sycl_get_env("GGML_SYCL_MMID_ESIMD", 1);
    return mode;
}

bool ggml_sycl_grouped_gemm_esimd_enabled() {
    return ggml_sycl_grouped_gemm_esimd_mode() != 0;
}

template <typename block_q_t, typename T>
static bool grouped_gemm_esimd_launch(const char * src0_dd, size_t expert_stride, const ggml_sycl_gg_tile * tiles,
                                      const T * packed, float * dst, int M, int Npad, int K,
                                      const ggml_sycl_gg_rows & dst_rows, int64_t n_tiles, dpct::queue_ptr stream) {
    using F = gg_esimd_a<block_q_t>;
    // whole row groups and stored blocks, every K-split range non-empty (also in mode 1, so both
    // modes take the same shapes); 2D block loads want
    // 64-byte aligned surfaces and pitches
    if (M % GG_MR != 0 || !F::ok(K) || K < GG_QK * GGML_SYCL_GG_KSPLIT || ((uintptr_t) src0_dd % 64) != 0 ||
        expert_stride % 64 != 0 || ((uintptr_t) packed % 64) != 0 || Npad % 16 != 0) {
        return false;
    }
    const size_t threads = (size_t) n_tiles * (M / GG_MR);
    if (ggml_sycl_grouped_gemm_esimd_mode() == 2) {
        stream->parallel_for(sycl::nd_range<1>(threads, 1),
                             grouped_gemm_esimd_kernel<block_q_t, GGML_SYCL_GG_KSPLIT, T>{
                                 src0_dd, expert_stride, tiles, packed, dst, M, Npad, K, dst_rows });
    } else {
        stream->parallel_for(sycl::nd_range<1>(threads, 1),
                             grouped_gemm_esimd_kernel<block_q_t, 1, T>{ src0_dd, expert_stride, tiles, packed, dst, M,
                                                                      Npad, K, dst_rows });
    }
    return true;
}

template <typename block_q_t, typename T>
bool ggml_sycl_grouped_gemm_esimd(const char * src0_dd, size_t expert_stride, const ggml_sycl_gg_tile * tiles,
                                  const T * packed, float * dst, int M, int Npad, int K,
                                  const ggml_sycl_gg_rows & dst_rows, int64_t n_tiles, dpct::queue_ptr stream) {
    static_assert(gg_esimd_a<block_q_t>::supported == ggml_sycl_grouped_gemm_esimd_has<block_q_t>,
                  "the header's format list and the formats implemented here must agree");
    return grouped_gemm_esimd_launch<block_q_t, T>(src0_dd, expert_stride, tiles, packed, dst, M, Npad, K, dst_rows,
                                                   n_tiles, stream);
}

#define GG_ESIMD_INSTANCE_T(block_q_t, T)                                                                     \
    template bool ggml_sycl_grouped_gemm_esimd<block_q_t, T>(const char *, size_t, const ggml_sycl_gg_tile *,   \
                                                             const T *, float *, int, int, int,                 \
                                                             const ggml_sycl_gg_rows &, int64_t, dpct::queue_ptr);
#define GG_ESIMD_INSTANCE(block_q_t)           \
    GG_ESIMD_INSTANCE_T(block_q_t, sycl::half) \
    GG_ESIMD_INSTANCE_T(block_q_t, sycl::ext::oneapi::bfloat16)
GG_ESIMD_INSTANCE(block_q4_K)
GG_ESIMD_INSTANCE(block_q5_K)
GG_ESIMD_INSTANCE(block_q6_K)
