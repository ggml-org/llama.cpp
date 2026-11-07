#include "fused-gemm-esimd.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// One thread owns MR rows of one expert's weight and one BN-wide tile of the schedule, over the
// whole K. Per k step (32 values) it loads the rows' stored bits, dequantizes them in registers to
// the f16 A operand of dpas (RC = 8 rows x 16 k, row major) and multiplies it with the packed B,
// which is already the dpas VNNI operand: [K/2][Npad][2] halves, 16 columns a row.
//
// The joint_matrix kernel dequantizes into SLM and reads it back in the same sub-group, which is
// staging, not sharing; dpas takes registers, so that round trip and its barriers are gone here.
// Its K split over GGML_SYCL_GG_KSPLIT sub-groups is reproduced as sequential K ranges summed in
// the same ascending order as its tile_c reduction, and every format below rounds A exactly as its
// joint_matrix A stage does, so the output is the same bits.
//
// A weight format is a gg_esimd_a<block> with: raw, the stored bits of the MR rows for one k step;
// load(), which fetches them from the expert's slice in either layout (gg_esimd::slice); and
// decode(), the f16 A of one row group.
// It reads the same fields as the joint_matrix A stage of its format.
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

// One field of a format's block. In the reorder (SoA) layout it is a stream of its own, `size` bytes
// a block, after `pre` bytes a block of the streams before it; in the canonical (AoS) layout it is
// `aos` bytes into the block.
struct field {
    int pre;
    int size;
    int aos;
};

// Where one thread's rows sit in an expert's slice, in either layout. nblocks counts the stored blocks
// of the slice, bpr those of one row, bs is the size of one block.
struct slice {
    const uint8_t * x;
    int             M;
    int             bpr;
    int             nblocks;
    int             m0;
    bool            aos;
    int             bs;

    // per row, the byte offset from x of byte `inner` of field f of block ib
    ESIMD_INLINE simd<uint32_t, GG_MR> at(const field & f, const int ib, const int inner) const {
        const int             stride = aos ? bs : f.size;
        simd<uint32_t, GG_MR> o(0, (uint32_t) (bpr * stride));
        return o + (uint32_t) ((aos ? f.aos : nblocks * f.pre) + (m0 * bpr + ib) * stride + inner);
    }

    // a field's bytes at offset `inner` start on a dword in every block (the AoS block and offset
    // are multiples of 4); the SoA streams always do
    ESIMD_INLINE bool dword_aligned(const field & f, const int inner) const {
        return !aos || (bs % 4 == 0 && (f.aos + inner) % 4 == 0);
    }

    ESIMD_INLINE simd<uint16_t, GG_MR> u16(const field & f, const int ib, const int inner) const {
        return gather<uint16_t, GG_MR>((const uint16_t *) x, at(f, ib, inner));
    }

    ESIMD_INLINE simd<uint32_t, GG_MR> u32(const field & f, const int ib, const int inner) const {
        if (dword_aligned(f, inner)) {
            return gather<uint32_t, GG_MR>((const uint32_t *) x, at(f, ib, inner));
        }
        return rows<1>(f, ib, inner);
    }

    // W dwords of each of the MR rows from byte `inner` (a multiple of 4) of field f of block ib, as
    // [row][W]. SoA: one 2D block message over the stream. AoS: the rows' dwords are gathered, as
    // [W][row], at offsets rounded down to a dword (every field starts on an even byte), shifted by
    // 16 bits where a row's block starts on 2 mod 4, and transposed with strided reads.
    template <int W> ESIMD_INLINE simd<uint32_t, GG_MR * W> rows(const field & f, const int ib, const int inner) const {
        if (!aos) {
            const unsigned pitch = (unsigned) (bpr * f.size);
            return load_2d<uint32_t, W, GG_MR>((const uint32_t *) (x + (size_t) nblocks * f.pre), pitch - 1, M - 1,
                                               pitch - 1, (ib * f.size + inner) / 4, m0);
        }
        const simd<uint32_t, GG_MR> o = at(f, ib, inner);
        simd<uint32_t, GG_MR * W>   g;
        if (dword_aligned(f, inner)) {
            g = gather<uint32_t, GG_MR * W, W>((const uint32_t *) x, o);
        } else {
            // per row: sh is 16 where the run starts on 2 mod 4, else 0; hm keeps the next dword's low
            // half only where it is needed
            const simd<uint32_t, GG_MR> oa = o & ~3u;
            const simd<uint32_t, GG_MR> sh = (o & 2u) << 3;
            const simd<uint32_t, GG_MR> hm = 0u - (sh >> 4);
            simd<uint32_t, GG_MR * W>   lo = gather<uint32_t, GG_MR * W, W>((const uint32_t *) x, oa);
            // the dword after the run holds its last 2 bytes where a row needs it; a row that does not
            // reads its first dword again rather than past the field
            const simd<uint32_t, GG_MR> tail = gather<uint32_t, GG_MR>((const uint32_t *) x, oa + ((4u * W) & hm));
            auto shift = [&](auto vc) SYCL_ESIMD_FUNCTION {
                constexpr int         v = decltype(vc)::value;
                simd<uint32_t, GG_MR> a = lo.template select<GG_MR, 1>(v * GG_MR);
                simd<uint32_t, GG_MR> b;
                if constexpr (v + 1 < W) {
                    b = lo.template select<GG_MR, 1>((v + 1) * GG_MR);
                } else {
                    b = tail;
                }
                g.template select<GG_MR, 1>(v * GG_MR) = (a >> sh) | ((b << 16) & hm);
            };
            static_for<0, W>::run(shift);
        }
        if constexpr (W == 1) {
            return g;
        } else {
            simd<uint32_t, GG_MR * W> t;
            auto row = [&](auto rc) SYCL_ESIMD_FUNCTION {
                constexpr int r = decltype(rc)::value;
                t.template select<W, 1>(r * W) = g.template select<W, GG_MR>(r);
            };
            static_for<0, GG_MR>::run(row);
            return t;
        }
    }
};

// A decode returns its row group's A in the order the two dpas of a k step read it, [k half][row][16
// k]: both iq formats store k and k + 16 in the same byte or grid dword half, so that order needs
// only reads with a stride, while [row][32 k] needed strided writes, which split into SIMD4.
using a_halves = simd<sycl::half, GG_RC * GG_QK>;

// f16 of the f32 product code * scale, each row's scale over its 16 values of a k half: the
// expression the joint_matrix stages evaluate before E::make rounds it
ESIMD_INLINE simd<sycl::half, GG_RC * 16> scale_half(simd<float, GG_RC * 16> v, simd<float, GG_RC> d) {
    v *= d.template replicate_vs_w_hs<GG_RC, 1, 16, 0>(0);
    return v;
}

// The iq4 codebook by byte, in SLM: entry b holds the values of its low and its high nibble as the
// two 16-bit halves of one dword, as T (half, or int16 for an f32 product). A read-only table every
// lane indexes, which is what SLM is for: register-indirect iselect() over the same lanes cost
// 7.3 of 12.7 ms, a 16-entry SLM table per nibble about 0.4.
template <typename T> ESIMD_INLINE void iq4_pair_table_to_slm() {
    static_assert(sizeof(T) == 2, "two values a dword");
    simd<uint32_t, 16> lo;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        lo[i] = sycl::bit_cast<uint16_t>((T) kvalues_iq4nl[i]);
    }
#pragma unroll
    for (int hi = 0; hi < 16; ++hi) {
        slm_block_store<uint32_t, 16>(hi * 64, lo | ((uint32_t) sycl::bit_cast<uint16_t>((T) kvalues_iq4nl[hi]) << 16));
    }
}

// 16 packed bytes a row; byte j holds k = j in its low and k = 16 + j in its high nibble (the order
// of fg_decode_iq4_xs and of the iq4_nl stage). One gather per byte; the low halves are k half 0,
// the high halves k half 1, each [row][16].
template <typename T> ESIMD_INLINE simd<T, GG_RC * GG_QK> iq4_codes(simd<uint32_t, GG_RC * 4> q) {
    simd<uint32_t, GG_RC * 16> off = q.template bit_cast_view<uint8_t>();
    simd<uint32_t, GG_RC * 16> pr  = slm_gather<uint32_t, GG_RC * 16>(off * 4);
    auto                       pv  = pr.template bit_cast_view<T>();
    simd<T, GG_RC * GG_QK>     v;
    v.template select<GG_RC * 16, 1>(0)          = pv.template select<GG_RC * 16, 2>(0);
    v.template select<GG_RC * 16, 1>(GG_RC * 16) = pv.template select<GG_RC * 16, 2>(1);
    return v;
}
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

// a row group's 32 values from grid dwords (byte b of entry e is k = 4e + b) and a sign dword per
// row (bit k negates k), times d: the expression of the iq3 stages
ESIMD_INLINE a_halves grid_signed(simd<uint32_t, GG_RC * 8> ge, simd<uint32_t, GG_RC> signs, simd<float, GG_RC> d) {
    auto                       gm = ge.template bit_cast_view<uint32_t, GG_RC, 8>();
    simd<uint32_t, GG_RC * 16> sg = signs.template replicate_vs_w_hs<GG_RC, 1, 16, 0>(0);
    simd<uint32_t, GG_RC * 16> k(0, 1);
    k &= 15;
    a_halves v;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        simd<uint32_t, GG_RC * 4> eh = gm.template select<GG_RC, 1, 4, 1>(0, 4 * h);
        simd<float, GG_RC * 16>   f  = eh.template bit_cast_view<uint8_t>();
        f.merge(-f, ((sg >> (k + 16 * h)) & 1) != 0);
        v.template select<GG_RC * 16, 1>(h * GG_RC * 16) = scale_half(f, d);
    }
    return v;
}
} // namespace gg_esimd

template <typename block_q_t> struct gg_esimd_a {
    static constexpr bool supported = false;
};

// [qs 32][d], one block a k step
template <> struct gg_esimd_a<block_q8_0> {
    static constexpr bool supported = true;
    static constexpr int  qk        = QK8_0;
    static constexpr bool whole     = true;  // decode_all() for the whole tile
    static constexpr int  slm_bytes = 0;
    static ESIMD_INLINE void init_slm() {}
    struct raw {
        gg_esimd::simd<uint32_t, GG_MR * 8> q;
        gg_esimd::simd<uint16_t, GG_MR>     d;
    };
    static constexpr gg_esimd::field QS = { 0, QK8_0, 2 }, D = { QK8_0, 2, 0 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || K >= 64); }
    struct sb_t {};
    static ESIMD_INLINE void load_sb(sb_t &, const gg_esimd::slice &, const int) {}
    static ESIMD_INLINE void load(raw & r, sb_t &, const gg_esimd::slice & s, const int kb, const int) {
        r.q = s.rows<8>(QS, kb, 0);
        r.d = s.u16(D, kb, 0);
    }
    // all MR x 32 quants at once: int8 -> half, times the row's scale. q * d needs at most 19
    // significant bits, so the f32 product the joint_matrix stage rounds to half is exact and one
    // correctly rounded half multiply gives the same bits.
    static ESIMD_INLINE gg_esimd::simd<sycl::half, GG_MR * GG_QK> decode_all(raw & r) {
        using namespace gg_esimd;
        simd<sycl::half, GG_MR * GG_QK> av = r.q.template bit_cast_view<int8_t>();
        simd<sycl::half, GG_MR>         dh = r.d.template bit_cast_view<sycl::half>();
        av *= dh.template replicate_vs_w_hs<GG_MR, 1, GG_QK, 0>(0);
        return av;
    }
};

// [qs 16][d], one block a k step
template <> struct gg_esimd_a<block_iq4_nl> {
    static constexpr bool supported = true;
    static constexpr int  qk        = QK4_NL;
    static constexpr bool whole     = false;
    static constexpr int  slm_bytes = 256 * sizeof(uint32_t);
    static ESIMD_INLINE void init_slm() { gg_esimd::iq4_pair_table_to_slm<sycl::half>(); }
    struct raw {
        gg_esimd::simd<uint32_t, GG_MR * 4>  q;
        gg_esimd::simd<sycl::half, GG_MR>    d;
    };
    static constexpr gg_esimd::field QS = { 0, QK4_NL / 2, 2 }, D = { QK4_NL / 2, 2, 0 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || K / 2 >= 64); }
    struct sb_t {};
    static ESIMD_INLINE void load_sb(sb_t &, const gg_esimd::slice &, const int) {}
    static ESIMD_INLINE void load(raw & r, sb_t &, const gg_esimd::slice & s, const int kb, const int) {
        using namespace gg_esimd;
        r.q                     = s.rows<4>(QS, kb, 0);
        simd<uint16_t, GG_MR> d = s.u16(D, kb, 0);
        r.d                     = d.template bit_cast_view<sycl::half>();
    }
    // the codebook values are integers below 128, exact in half, and d * value has at most 18
    // significant bits: one correctly rounded half multiply is the stage's f32 product rounded
    template <int g> static ESIMD_INLINE gg_esimd::a_halves decode(raw & r) {
        using namespace gg_esimd;
        a_halves                v  = iq4_codes<sycl::half>(r.q.template select<GG_RC * 4, 1>(g * GG_RC * 4));
        simd<sycl::half, GG_RC> dg = r.d.template select<GG_RC, 1>(g * GG_RC);
        simd<sycl::half, GG_RC * 16> dr = dg.template replicate_vs_w_hs<GG_RC, 1, 16, 0>(0);
        v.template select<GG_RC * 16, 1>(0) *= dr;
        v.template select<GG_RC * 16, 1>(GG_RC * 16) *= dr;
        return v;
    }
};

// [qs 128][d 2][scales_h 2][scales_l 4] per 256 values; k step ib8 of a superblock is sub-block ib8
template <> struct gg_esimd_a<block_iq4_xs> {
    static constexpr bool supported = true;
    static constexpr int  qk        = QK_K;
    static constexpr bool whole     = false;
    // int16 table and f32 products: d * (scale - 32) * value can need 24 significant bits, more
    // than a half multiply keeps
    static constexpr int  slm_bytes = 256 * sizeof(uint32_t);
    static ESIMD_INLINE void init_slm() { gg_esimd::iq4_pair_table_to_slm<int16_t>(); }
    struct raw {
        gg_esimd::simd<uint32_t, GG_MR * 4> q;
        gg_esimd::simd<float, GG_MR>        d;
    };
    struct sb_t {
        gg_esimd::simd<float, GG_MR>    d;
        gg_esimd::simd<uint32_t, GG_MR> sh; // scales_h
        gg_esimd::simd<uint32_t, GG_MR> sl; // scales_l, 4 bytes
    };
    static constexpr gg_esimd::field QS = { 0, QK_K / 2, 8 }, D = { QK_K / 2, 2, 0 }, SH = { QK_K / 2 + 2, 2, 2 },
                                     SL = { QK_K / 2 + 4, 4, 4 };
    static bool ok(const int K, const bool) { return K % qk == 0; }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib) {
        using namespace gg_esimd;
        simd<uint16_t, GG_MR> dh = s.u16(D, ib, 0);
        sb.d  = simd<sycl::half, GG_MR>(dh.template bit_cast_view<sycl::half>());
        sb.sh = s.u16(SH, ib, 0);
        sb.sl = s.u32(SL, ib, 0);
    }
    static ESIMD_INLINE void load(raw & r, sb_t & sb, const gg_esimd::slice & s, const int kb, const int j) {
        using namespace gg_esimd;
        r.q = s.rows<4>(QS, kb / 8, 16 * j);
        // d * (6-bit scale - 32), in f32 as the stage computes it
        simd<int32_t, GG_MR> sc = simd<int32_t, GG_MR>(((sb.sl >> (4 * j)) & 0xf) | (((sb.sh >> (2 * j)) & 3) << 4)) - 32;
        r.d = sb.d * simd<float, GG_MR>(sc);
    }
    template <int g> static ESIMD_INLINE gg_esimd::a_halves decode(raw & r) {
        using namespace gg_esimd;
        simd<int16_t, GG_RC * GG_QK> c  = iq4_codes<int16_t>(r.q.template select<GG_RC * 4, 1>(g * GG_RC * 4));
        simd<float, GG_RC>           dg = r.d.template select<GG_RC, 1>(g * GG_RC);
        a_halves                     v;
        v.template select<GG_RC * 16, 1>(0)          = scale_half(c.template select<GG_RC * 16, 1>(0), dg);
        v.template select<GG_RC * 16, 1>(GG_RC * 16) = scale_half(c.template select<GG_RC * 16, 1>(GG_RC * 16), dg);
        return v;
    }
};

// [qs 64][qh 8][signs 32][d 2, scales 4] per 256 values. Value k of sub-block ib8 is byte k % 4 of
// grid entry k / 4, the entry index being qs[k / 4] with bit 8 from qh bit k / 4, negated where bit
// k of the sub-block's sign dword is set; times d * (1 + 2 * 4-bit scale)
template <> struct gg_esimd_a<block_iq3_s> {
    static constexpr bool supported = true;
    static constexpr int  qk        = QK_K;
    static constexpr bool whole     = false;
    // the 512-entry grid in SLM, read by every lane: the global gather of it cost 2.4x the kernel
    static constexpr int  slm_bytes = 512 * sizeof(uint32_t);
    static ESIMD_INLINE void init_slm() {
        using namespace gg_esimd;
#pragma unroll
        for (int i = 0; i < 512; i += 64) {
            slm_block_store<uint32_t, 64>(i * 4, block_load<uint32_t, 64>(iq3s_grid + i));
        }
    }
    struct raw {
        gg_esimd::simd<uint32_t, GG_MR * 2> q;     // [row][8 qs bytes]
        gg_esimd::simd<uint32_t, GG_MR>     qh;    // the sub-block's qh byte
        gg_esimd::simd<uint32_t, GG_MR>     signs; // the sub-block's 32 sign bits
        gg_esimd::simd<float, GG_MR>        d;
    };
    // d and the scale bytes are a block's, kept over its k steps; the sign dword and the qh byte of
    // a k step are a 2D load each at that step's column, so no register is picked by a runtime index
    struct sb_t {
        gg_esimd::simd<float, GG_MR>    d;
        gg_esimd::simd<uint32_t, GG_MR> sc; // the 4 scale bytes
    };
    // the SoA stream {d, scales} is d and scales of the block in AoS, which are not adjacent there: MS
    // reads the scales at their SoA offset 2 in the stream, which is AoS offset 2 past 104
    static constexpr int             META = QK_K / 4 + QK_K / 32 + QK_K / 8;
    static constexpr gg_esimd::field QS = { 0, QK_K / 4, 2 }, QH = { QK_K / 4, QK_K / 32, 2 + QK_K / 4 },
                                     SG = { QK_K / 4 + QK_K / 32, QK_K / 8, 2 + QK_K / 4 + QK_K / 32 },
                                     MD = { META, 2 + IQ3S_N_SCALE, 0 }, MS = { META, 2 + IQ3S_N_SCALE, META };
    // the SoA qh and signs streams are 2D surfaces: their row pitches must be 16-byte aligned
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || ((K / QK_K * 8) % 16 == 0 && K / QK_K * 8 >= 64));
    }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib) {
        using namespace gg_esimd;
        simd<uint16_t, GG_MR> dh = s.u16(MD, ib, 0);
        simd<uint32_t, GG_MR> s0 = s.u16(MS, ib, 2);
        simd<uint32_t, GG_MR> s1 = s.u16(MS, ib, 4);
        sb.d  = simd<sycl::half, GG_MR>(dh.template bit_cast_view<sycl::half>());
        sb.sc = s0 | (s1 << 16);
    }
    static ESIMD_INLINE void load(raw & r, sb_t & sb, const gg_esimd::slice & s, const int kb, const int j) {
        using namespace gg_esimd;
        const int ib = kb / 8;
        r.q     = s.rows<2>(QS, ib, 8 * j);
        r.qh    = (s.rows<1>(QH, ib, 4 * (j / 4)) >> (8 * (j % 4))) & 0xffu;
        r.signs = s.rows<1>(SG, ib, 4 * j);
        r.d     = sb.d * simd<float, GG_MR>(simd<int32_t, GG_MR>(1 + 2 * ((sb.sc >> (4 * j)) & 0xf)));
    }
    template <int g> static ESIMD_INLINE gg_esimd::a_halves decode(raw & r) {
        using namespace gg_esimd;
        // the 8 grid entries of each of the row group's rows
        simd<uint16_t, GG_RC * 8> qb = r.q.template select<GG_RC * 2, 1>(g * GG_RC * 2).template bit_cast_view<uint8_t>();
        simd<uint32_t, GG_RC>     qr = r.qh.template select<GG_RC, 1>(g * GG_RC);
        simd<uint32_t, GG_RC * 8> qh = qr.template replicate_vs_w_hs<GG_RC, 1, 8, 0>(0);
        simd<uint32_t, GG_RC * 8> j(0, 1);
        j &= 7;
        simd<uint32_t, GG_RC * 8> idx = simd<uint32_t, GG_RC * 8>(qb) | (((qh >> j) & 1) << 8);
        simd<uint32_t, GG_RC * 8> ge  = slm_gather<uint32_t, GG_RC * 8>(idx * 4);
        return grid_signed(ge, r.signs.template select<GG_RC, 1>(g * GG_RC), r.d.template select<GG_RC, 1>(g * GG_RC));
    }
};

// The K-quants and iq3_xxs: a 256-value superblock, k step j its sub-block j. Their value bytes for
// a k step are one rectangle of a stream, in k order after a shift and a mask per dword; scales,
// mins and d are the block's and are loaded once per block.

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
    static constexpr gg_esimd::field QS = { 0, QK_K / 2, 16 }, SC = { QK_K / 2, K_SCALE_SIZE, 4 },
                                     DM = { QK_K / 2 + K_SCALE_SIZE, 4, 0 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || gg_esimd::surface_ok(K / 2)); }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib, const gg_esimd::field & sc = SC,
                                     const gg_esimd::field & dmf = DM) {
        using namespace gg_esimd;
#pragma unroll
        for (int w = 0; w < 3; ++w) {
            sb.sc[w] = s.u32(sc, ib, 4 * w);
        }
        simd<uint32_t, GG_MR> dm = s.u32(dmf, ib, 0);
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
        r.q = (s.rows<8>(QS, kb / 8, 32 * (j / 2)) >> (4 * (j & 1))) & 0x0F0F0F0Fu;
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
    static constexpr gg_esimd::field QS = { 0, QK_K / 2, 48 }, QH = { QK_K / 2, QK_K / 8, 16 },
                                     SC = { QK_K / 2 + QK_K / 8, K_SCALE_SIZE, 4 },
                                     DM = { QK_K / 2 + QK_K / 8 + K_SCALE_SIZE, 4, 0 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || (gg_esimd::surface_ok(K / 2) && gg_esimd::surface_ok(K / 8)));
    }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib) { q4::load_sb(sb, s, ib, SC, DM); }
    static ESIMD_INLINE void load(raw & r, sb_t & sb, const gg_esimd::slice & s, const int kb, const int j) {
        using namespace gg_esimd;
        const int                 ib = kb / 8;
        simd<uint32_t, GG_MR * 8> qh = s.rows<8>(QH, ib, 0);
        r.q = ((s.rows<8>(QS, ib, 32 * (j / 2)) >> (4 * (j & 1))) & 0x0F0F0F0Fu) |
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
    static constexpr gg_esimd::field QL = { 0, QK_K / 2, 0 }, QH = { QK_K / 2, QK_K / 4, QK_K / 2 },
                                     SC = { QK_K / 2 + QK_K / 4, QK_K / 16, QK_K / 2 + QK_K / 4 },
                                     D = { QK_K / 2 + QK_K / 4 + QK_K / 16, 2, QK_K / 2 + QK_K / 4 + QK_K / 16 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || (gg_esimd::surface_ok(K / 2) && gg_esimd::surface_ok(K / 4)));
    }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib) {
        using namespace gg_esimd;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            sb.sc[w] = s.u32(SC, ib, 4 * w);
        }
        simd<uint16_t, GG_MR> dh = s.u16(D, ib, 0);
        sb.d = simd<sycl::half, GG_MR>(dh.template bit_cast_view<sycl::half>());
    }
    static ESIMD_INLINE void load(raw & r, sb_t & sb, const gg_esimd::slice & s, const int kb, const int j) {
        using namespace gg_esimd;
        const int ib = kb / 8, h = j / 4, q = j % 4;
        simd<uint32_t, GG_MR * 8> ql = s.rows<8>(QL, ib, 64 * h + 32 * (q & 1));
        simd<uint32_t, GG_MR * 8> qh = s.rows<8>(QH, ib, 32 * h);
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

// [qs 64][scales 16][dm]: value l of sub-block j is bits 2 (j % 4) of byte 32 (j / 4) + l, times
// d * (scale & 15), minus dmin * (scale >> 4), scale byte 2j + l / 16
template <> struct gg_esimd_a<block_q2_K> {
    static constexpr bool supported = true;
    static constexpr int  qk        = QK_K;
    static constexpr bool whole     = false;
    static constexpr int  slm_bytes = 0;
    static ESIMD_INLINE void init_slm() {}
    struct raw {
        gg_esimd::simd<uint32_t, GG_MR * 8> q;
        gg_esimd::simd<float, GG_MR>        d0, m0, d1, m1;
    };
    struct sb_t {
        gg_esimd::simd<uint32_t, GG_MR> sc[4];
        gg_esimd::simd<float, GG_MR>    d, dmin;
    };
    static constexpr gg_esimd::field QS = { 0, QK_K / 4, QK_K / 16 }, SC = { QK_K / 4, QK_K / 16, 0 },
                                     DM = { QK_K / 4 + QK_K / 16, 4, QK_K / 4 + QK_K / 16 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || gg_esimd::surface_ok(K / 4)); }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib) {
        using namespace gg_esimd;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            sb.sc[w] = s.u32(SC, ib, 4 * w);
        }
        simd<uint32_t, GG_MR> dm = s.u32(DM, ib, 0);
        auto                  dh = dm.template bit_cast_view<sycl::half>();
        sb.d                     = simd<sycl::half, GG_MR>(dh.template select<GG_MR, 2>(0));
        sb.dmin                  = simd<sycl::half, GG_MR>(dh.template select<GG_MR, 2>(1));
    }
    static ESIMD_INLINE void load(raw & r, sb_t & sb, const gg_esimd::slice & s, const int kb, const int j) {
        using namespace gg_esimd;
        r.q = (s.rows<8>(QS, kb / 8, 32 * (j / 4)) >> (2 * (j % 4))) & 0x03030303u;
        simd<uint32_t, GG_MR> w  = dword_of(sb.sc, j / 2);
        simd<uint32_t, GG_MR> s0 = byte_of(w, 2 * (j % 2)), s1 = byte_of(w, 2 * (j % 2) + 1);
        r.d0 = sb.d * simd<float, GG_MR>(s0 & 0xf);
        r.m0 = sb.dmin * simd<float, GG_MR>(s0 >> 4);
        r.d1 = sb.d * simd<float, GG_MR>(s1 & 0xf);
        r.m1 = sb.dmin * simd<float, GG_MR>(s1 >> 4);
    }
    template <int g> static ESIMD_INLINE gg_esimd::a_halves decode(raw & r) {
        using namespace gg_esimd;
        return halves_dm(r.q.template select<GG_RC * 8, 1>(g * GG_RC * 8), r.d0.template select<GG_RC, 1>(g * GG_RC),
                         r.m0.template select<GG_RC, 1>(g * GG_RC), r.d1.template select<GG_RC, 1>(g * GG_RC),
                         r.m1.template select<GG_RC, 1>(g * GG_RC));
    }
};

// [qs 64][hmask 32][scales 12][d]: q2_K's 2 bits, minus 4 where bit j of hmask byte l is clear,
// times d * (6-bit scale - 32), scale s = 2j + l / 16 split over bytes s % 8 and 8 + s % 4
template <> struct gg_esimd_a<block_q3_K> {
    static constexpr bool supported = true;
    static constexpr int  qk        = QK_K;
    static constexpr bool whole     = false;
    static constexpr int  slm_bytes = 0;
    static ESIMD_INLINE void init_slm() {}
    struct raw {
        gg_esimd::simd<uint32_t, GG_MR * 8> q;
        gg_esimd::simd<float, GG_MR>        d0, d1;
    };
    struct sb_t {
        gg_esimd::simd<uint32_t, GG_MR> sc[4]; // 12 scale bytes, sc[3] unused
        gg_esimd::simd<float, GG_MR>    d;
    };
    static constexpr gg_esimd::field QS = { 0, QK_K / 4, QK_K / 8 }, HM = { QK_K / 4, QK_K / 8, 0 },
                                     SC = { QK_K / 4 + QK_K / 8, 12, QK_K / 4 + QK_K / 8 },
                                     D = { QK_K / 4 + QK_K / 8 + 12, 2, QK_K / 4 + QK_K / 8 + 12 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || (gg_esimd::surface_ok(K / 4) && gg_esimd::surface_ok(K / 8)));
    }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib) {
        using namespace gg_esimd;
#pragma unroll
        for (int w = 0; w < 3; ++w) {
            sb.sc[w] = s.u32(SC, ib, 4 * w);
        }
        sb.sc[3] = 0;
        simd<uint16_t, GG_MR> dh = s.u16(D, ib, 0);
        sb.d = simd<sycl::half, GG_MR>(dh.template bit_cast_view<sycl::half>());
    }
    static ESIMD_INLINE gg_esimd::simd<float, GG_MR> scale(sb_t & sb, const int sidx) {
        using namespace gg_esimd;
        simd<uint32_t, GG_MR> lo = (byte_of(dword_of(sb.sc, (sidx % 8) / 4), sidx % 4) >> (4 * (sidx / 8))) & 0xf;
        simd<uint32_t, GG_MR> hi = (byte_of(sb.sc[2], sidx % 4) >> (2 * (sidx / 4))) & 3;
        return sb.d * simd<float, GG_MR>(simd<int32_t, GG_MR>(lo | (hi << 4)) - 32);
    }
    static ESIMD_INLINE void load(raw & r, sb_t & sb, const gg_esimd::slice & s, const int kb, const int j) {
        using namespace gg_esimd;
        const int                 ib = kb / 8;
        simd<uint32_t, GG_MR * 8> hm = s.rows<8>(HM, ib, 0);
        r.q = ((s.rows<8>(QS, ib, 32 * (j / 4)) >> (2 * (j % 4))) & 0x03030303u) |
              (((hm >> j) & 0x01010101u) << 2);
        r.d0 = scale(sb, 2 * j);
        r.d1 = scale(sb, 2 * j + 1);
    }
    template <int g> static ESIMD_INLINE gg_esimd::a_halves decode(raw & r) {
        using namespace gg_esimd;
        simd<float, GG_RC> z = 0.0f;
        return halves_dm(r.q.template select<GG_RC * 8, 1>(g * GG_RC * 8), r.d0.template select<GG_RC, 1>(g * GG_RC), z,
                         r.d1.template select<GG_RC, 1>(g * GG_RC), z, 4.0f);
    }
};

// [qs 96][d]: per sub-block j, 8 grid indices at 8j and an aux dword at 64 + 4j whose 4 x 7 bits
// are sign bytes of ksigns_iq2xs and whose top 4 bits scale d * (0.5 + s) * 0.5. ksigns_iq2xs[i] is
// i with the parity of its 7 bits as bit 7, so the sign bytes are computed, not looked up.
template <> struct gg_esimd_a<block_iq3_xxs> {
    static constexpr bool supported = true;
    static constexpr int  qk        = QK_K;
    static constexpr bool whole     = false;
    // the 256-entry grid, a read-only table in SLM
    static constexpr int  slm_bytes = 256 * sizeof(uint32_t);
    static ESIMD_INLINE void init_slm() {
        using namespace gg_esimd;
#pragma unroll
        for (int i = 0; i < 256; i += 64) {
            slm_block_store<uint32_t, 64>(i * 4, block_load<uint32_t, 64>(iq3xxs_grid + i));
        }
    }
    struct raw {
        gg_esimd::simd<uint32_t, GG_MR * 2> q;   // [row][8 grid indices]
        gg_esimd::simd<uint32_t, GG_MR>     aux;
        gg_esimd::simd<float, GG_MR>        d;
    };
    struct sb_t {
        gg_esimd::simd<float, GG_MR> d;
    };
    static constexpr gg_esimd::field QS = { 0, 3 * QK_K / 8, 2 }, D = { 3 * QK_K / 8, 2, 0 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || gg_esimd::surface_ok(K / QK_K * 96)); }
    static ESIMD_INLINE void load_sb(sb_t & sb, const gg_esimd::slice & s, const int ib) {
        using namespace gg_esimd;
        simd<uint16_t, GG_MR> dh = s.u16(D, ib, 0);
        sb.d = simd<sycl::half, GG_MR>(dh.template bit_cast_view<sycl::half>());
    }
    static ESIMD_INLINE void load(raw & r, sb_t & sb, const gg_esimd::slice & s, const int kb, const int j) {
        using namespace gg_esimd;
        const int ib = kb / 8;
        r.q   = s.rows<2>(QS, ib, 8 * j);
        r.aux = s.rows<1>(QS, ib, 64 + 4 * j);
        r.d   = sb.d * (0.5f + simd<float, GG_MR>(r.aux >> 28)) * 0.5f;
    }
    template <int g> static ESIMD_INLINE gg_esimd::a_halves decode(raw & r) {
        using namespace gg_esimd;
        simd<uint32_t, GG_RC * 8> qb = r.q.template select<GG_RC * 2, 1>(g * GG_RC * 2).template bit_cast_view<uint8_t>();
        simd<uint32_t, GG_RC * 8> ge = slm_gather<uint32_t, GG_RC * 8>(qb * 4);
        simd<uint32_t, GG_RC>     ax = r.aux.template select<GG_RC, 1>(g * GG_RC);
        simd<uint32_t, GG_RC>     sg = 0;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            simd<uint32_t, GG_RC> si = (ax >> (7 * il)) & 127;
            sg |= (si | ((cbit(si) & 1) << 7)) << (8 * il);
        }
        return grid_signed(ge, sg, r.d.template select<GG_RC, 1>(g * GG_RC));
    }
};

// T is the dpas operand type, sycl::half or bfloat16: the type B is packed in. A is decoded in half
// either way and converted for bf16 (see the double rounding note at the conversion). AOS: the weight
// is in the canonical block layout rather than the reorder (SoA) one.
template <typename block_q_t, int KS, typename T, bool AOS> struct grouped_gemm_esimd_kernel {
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

        // the slice of this expert
        const int             bpr = K / F::qk;
        const gg_esimd::slice sl  = { (const uint8_t *) src0_dd + (size_t) tile.expert * expert_stride, M, bpr, M * bpr, m0,
                                      AOS, (int) sizeof(block_q_t) };
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

// the weights' surfaces: 2D block loads (SoA) want 64-byte aligned bases, the AoS dword gathers 4
static bool gg_esimd_weights_ok(const char * w, const size_t expert_stride, const bool reordered) {
    const size_t a = reordered ? 64 : 4;
    return (uintptr_t) w % a == 0 && expert_stride % a == 0;
}

// 0 off, 1 on (default), 2 on with the joint_matrix kernel's K-split order reproduced (bit-identical output)
static int ggml_sycl_grouped_gemm_esimd_mode() {
    static const int mode = ggml_sycl_get_env("GGML_SYCL_MMID_ESIMD", 1);
    return mode;
}

bool ggml_sycl_grouped_gemm_esimd_enabled() {
    return ggml_sycl_grouped_gemm_esimd_mode() != 0;
}

template <typename block_q_t, typename T, bool AOS>
static void grouped_gemm_esimd_launch(const char * src0_dd, size_t expert_stride, const ggml_sycl_gg_tile * tiles,
                                      const T * packed, float * dst, int M, int Npad, int K,
                                      const ggml_sycl_gg_rows & dst_rows, int64_t n_tiles, dpct::queue_ptr stream) {
    const sycl::nd_range<1> range((size_t) n_tiles * (M / GG_MR), 1);
    if (ggml_sycl_grouped_gemm_esimd_mode() == 2) {
        stream->parallel_for(range, grouped_gemm_esimd_kernel<block_q_t, GGML_SYCL_GG_KSPLIT, T, AOS>{
                                        src0_dd, expert_stride, tiles, packed, dst, M, Npad, K, dst_rows });
    } else {
        stream->parallel_for(range, grouped_gemm_esimd_kernel<block_q_t, 1, T, AOS>{
                                        src0_dd, expert_stride, tiles, packed, dst, M, Npad, K, dst_rows });
    }
}

template <typename block_q_t, typename T>
bool ggml_sycl_grouped_gemm_esimd(const char * src0_dd, size_t expert_stride, const ggml_sycl_gg_tile * tiles,
                                  const T * packed, float * dst, int M, int Npad, int K,
                                  const ggml_sycl_gg_rows & dst_rows, bool reordered, int64_t n_tiles,
                                  dpct::queue_ptr stream) {
    static_assert(gg_esimd_a<block_q_t>::supported == ggml_sycl_grouped_gemm_esimd_has<block_q_t>,
                  "the header's format list and the formats implemented here must agree");
    using F = gg_esimd_a<block_q_t>;
    // whole row groups and stored blocks, every K-split range non-empty (also in mode 1, so both
    // modes take the same shapes); B is read with 2D block loads, which want 64-byte aligned
    // surfaces and pitches
    if (M % GG_MR != 0 || !F::ok(K, !reordered) || K < GG_QK * GGML_SYCL_GG_KSPLIT ||
        !gg_esimd_weights_ok(src0_dd, expert_stride, reordered) || ((uintptr_t) packed % 64) != 0 || Npad % 16 != 0) {
        return false;
    }
    if (reordered) {
        grouped_gemm_esimd_launch<block_q_t, T, false>(src0_dd, expert_stride, tiles, packed, dst, M, Npad, K, dst_rows,
                                                       n_tiles, stream);
    } else {
        grouped_gemm_esimd_launch<block_q_t, T, true>(src0_dd, expert_stride, tiles, packed, dst, M, Npad, K, dst_rows,
                                                      n_tiles, stream);
    }
    return true;
}

#define GG_ESIMD_INSTANCE_T(block_q_t, T)                                                                         \
    template bool ggml_sycl_grouped_gemm_esimd<block_q_t, T>(const char *, size_t, const ggml_sycl_gg_tile *,       \
                                                             const T *, float *, int, int, int,                     \
                                                             const ggml_sycl_gg_rows &, bool, int64_t, dpct::queue_ptr);
#define GG_ESIMD_INSTANCE(block_q_t)              \
    GG_ESIMD_INSTANCE_T(block_q_t, sycl::half) \
    GG_ESIMD_INSTANCE_T(block_q_t, sycl::ext::oneapi::bfloat16)
GG_ESIMD_INSTANCE(block_q8_0)
GG_ESIMD_INSTANCE(block_iq4_nl)
GG_ESIMD_INSTANCE(block_iq4_xs)
GG_ESIMD_INSTANCE(block_iq3_s)
GG_ESIMD_INSTANCE(block_iq3_xxs)
GG_ESIMD_INSTANCE(block_q3_K)
GG_ESIMD_INSTANCE(block_q2_K)
GG_ESIMD_INSTANCE(block_q6_K)
GG_ESIMD_INSTANCE(block_q5_K)
GG_ESIMD_INSTANCE(block_q4_K)
