#include "fused-gemm-esimd.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "xmx-compat.hpp"

// One thread owns MR rows of one expert's weight and one BN-wide tile of the schedule, over the
// whole K. Per k step (32 values) it loads the rows' stored bits, dequantizes them in registers to
// the f16 A operand of dpas (RC = 8 rows x 16 k, row major) and multiplies it with the packed B,
// which is already the dpas VNNI operand: [K/2][Npad][2] halves, NW columns a dpas.
//
// The joint_matrix kernel dequantizes into SLM and reads it back in the same sub-group, which is
// staging, not sharing; dpas takes registers, so that round trip and its barriers are gone here.
// Its K split over GGML_SYCL_GROUPED_GEMM_KSPLIT sub-groups is reproduced as sequential K ranges summed in
// the same ascending order as its tile_c reduction, and every format below rounds A exactly as its
// joint_matrix A stage does, so the output is the same bits.
//
// A weight format is a grouped_gemm_esimd_a<block, MR> with: raw, the stored bits of the MR rows for one k
// step; load(), which fetches them from the expert's slice in either layout (grouped_gemm_esimd::slice); and
// decode(), the f16 A of one row group.
// It reads the same fields as the joint_matrix A stage of its format.
//
// The kernel is built for one grouped_gemm_arch: the dpas width NW (the columns of B one dpas takes, the
// sub-group size of the joint_matrix combination it stands in for), the rows of one thread, and
// whether 2D block loads exist. Xe2 and Xe-HPC: 16 wide, 32 rows, 2D loads. Xe-HPG (Arc A, ARL-H):
// 8 wide, no 2D block messages, and 256 registers of 32 bytes, half of Xe2's file, so 16 rows: its
// 32-row accumulators alone would take half the registers.
static constexpr int GGML_SYCL_GROUPED_GEMM_KSPLIT = FG_KSPLIT;
static constexpr int GGML_SYCL_GROUPED_GEMM_BN     = GGML_SYCL_FG_BN;
static constexpr int GROUPED_GEMM_MR               = 32;  // rows of one thread on Xe2: one 2D load covers them
static constexpr int GROUPED_GEMM_RC               = 8;   // rows of one dpas
static constexpr int GROUPED_GEMM_QK               = 32;  // values of one k step

// HW: the architecture (ggml_sycl_xmx::xe2, xehpg), MR: the rows of one thread
template <typename HW_, int MR_> struct grouped_gemm_arch {
    using HW                  = HW_;
    static constexpr int  NW  = HW::dpas_width;
    static constexpr int  MR  = MR_;
    static constexpr bool L2D = HW::block_2d;
};

using grouped_gemm_xe2   = grouped_gemm_arch<ggml_sycl_xmx::xe2, GROUPED_GEMM_MR>;
using grouped_gemm_xehpg = grouped_gemm_arch<ggml_sycl_xmx::xehpg, 16>;

// An AOT build compiles each kernel only for the targets of its dpas width: GGML_SYCL_XMX_AOT_SG for
// spir64_gen, FG_AOT_SG (fused-gemm.hpp) for intel_gpu_* targets, unless XMX is emulated (xmx-compat.hpp).
// A build without XMX (GGML_SYCL_SUPPORT_XMX off) compiles none of them
template <typename A> static constexpr bool grouped_gemm_built() {
#if !defined(GGML_SYCL_SUPPORT_XMX)
    return false;
#elif defined(GGML_SYCL_XMX_AOT_SG)
    return A::NW == GGML_SYCL_XMX_AOT_SG || ggml_sycl_xmx::emulated;
#elif defined(FG_AOT_SG)
    return A::NW == FG_AOT_SG || ggml_sycl_xmx::emulated;
#else
    return true;
#endif
}

namespace grouped_gemm_esimd {
using namespace sycl::ext::intel::esimd;
namespace xmx = ggml_sycl_xmx::esimd;

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
template <int MR, typename HW> struct slice {
    const uint8_t * x;
    int             M;
    int             bpr;
    int             nblocks;
    int             m0;
    bool            aos;
    int             bs;

    // per row, the byte offset from x of byte `inner` of field f of block ib
    ESIMD_INLINE simd<uint32_t, MR> at(const field & f, const int ib, const int inner) const {
        const int             stride = aos ? bs : f.size;
        simd<uint32_t, MR> o(0, (uint32_t) (bpr * stride));
        return o + (uint32_t) ((aos ? f.aos : nblocks * f.pre) + (m0 * bpr + ib) * stride + inner);
    }

    // a field's bytes at offset `inner` start on a dword in every block (the AoS block and offset
    // are multiples of 4); the SoA streams always do
    ESIMD_INLINE bool dword_aligned(const field & f, const int inner) const {
        return !aos || (bs % 4 == 0 && (f.aos + inner) % 4 == 0);
    }

    ESIMD_INLINE simd<uint32_t, MR> u8(const field & f, const int ib, const int inner) const {
        return gather<uint8_t, MR>(x, at(f, ib, inner));
    }

    ESIMD_INLINE simd<uint16_t, MR> u16(const field & f, const int ib, const int inner) const {
        return gather<uint16_t, MR>((const uint16_t *) x, at(f, ib, inner));
    }

    ESIMD_INLINE simd<uint32_t, MR> u32(const field & f, const int ib, const int inner) const {
        if (dword_aligned(f, inner)) {
            return gather<uint32_t, MR>((const uint32_t *) x, at(f, ib, inner));
        }
        return rows<1>(f, ib, inner);
    }

    // W dwords of each row from its byte offset o, as [W][row]: one gather per run of at most 8, the
    // most dwords an LSC gather takes per address
    template <int W> ESIMD_INLINE simd<uint32_t, MR * W> gather_runs(const simd<uint32_t, MR> o) const {
        if constexpr (W <= 8) {
            return gather<uint32_t, MR * W, W>((const uint32_t *) x, o);
        } else {
            static_assert(W % 8 == 0, "runs of 8 dwords");
            simd<uint32_t, MR * W> g;
            auto run = [&](auto cc) SYCL_ESIMD_FUNCTION {
                constexpr int c = decltype(cc)::value;
                g.template select<MR * 8, 1>(c * MR * 8) = gather<uint32_t, MR * 8, 8>((const uint32_t *) x, o + 32u * c);
            };
            static_for<0, W / 8>::run(run);
            return g;
        }
    }

    // W dwords of each of the MR rows from byte `inner` (a multiple of 4) of field f of block ib, as
    // [row][W]. SoA with 2D block loads (HW::block_2d): one message over the stream. Otherwise the rows' dwords
    // are gathered, as [W][row], at offsets rounded down to a dword (every field starts on an even
    // byte; the SoA streams are dword aligned), shifted by 16 bits where an AoS row's block starts on
    // 2 mod 4, and transposed with strided reads.
    template <int W> ESIMD_INLINE simd<uint32_t, MR * W> rows(const field & f, const int ib, const int inner) const {
        if constexpr (HW::block_2d) {
            if (!aos) {
                const unsigned pitch = (unsigned) (bpr * f.size);
                return xmx::load_2d<HW, uint32_t, W, MR>((const uint32_t *) (x + (size_t) nblocks * f.pre), pitch - 1,
                                                         M - 1, pitch - 1, (ib * f.size + inner) / 4, m0);
            }
        }
        const simd<uint32_t, MR> o = at(f, ib, inner);
        simd<uint32_t, MR * W>   g;
        if (dword_aligned(f, inner)) {
            g = gather_runs<W>(o);
        } else {
            // per row: sh is 16 where the run starts on 2 mod 4, else 0; hm keeps the next dword's low
            // half only where it is needed
            const simd<uint32_t, MR> oa = o & ~3u;
            const simd<uint32_t, MR> sh = (o & 2u) << 3;
            const simd<uint32_t, MR> hm = 0u - (sh >> 4);
            simd<uint32_t, MR * W>   lo = gather_runs<W>(oa);
            // the dword after the run holds its last 2 bytes where a row needs it; a row that does not
            // reads its first dword again rather than past the field
            const simd<uint32_t, MR> tail = gather<uint32_t, MR>((const uint32_t *) x, oa + ((4u * W) & hm));
            auto shift = [&](auto vc) SYCL_ESIMD_FUNCTION {
                constexpr int         v = decltype(vc)::value;
                simd<uint32_t, MR> a = lo.template select<MR, 1>(v * MR);
                simd<uint32_t, MR> b;
                if constexpr (v + 1 < W) {
                    b = lo.template select<MR, 1>((v + 1) * MR);
                } else {
                    b = tail;
                }
                g.template select<MR, 1>(v * MR) = (a >> sh) | ((b << 16) & hm);
            };
            static_for<0, W>::run(shift);
        }
        if constexpr (W == 1) {
            return g;
        } else {
            simd<uint32_t, MR * W> t;
            auto row = [&](auto rc) SYCL_ESIMD_FUNCTION {
                constexpr int r = decltype(rc)::value;
                t.template select<W, 1>(r * W) = g.template select<W, MR>(r);
            };
            static_for<0, MR>::run(row);
            return t;
        }
    }
};

// A decode returns its row group's A in the order the two dpas of a k step read it, [k half][row][16
// k]: both iq formats store k and k + 16 in the same byte or grid dword half, so that order needs
// only reads with a stride, while [row][32 k] needed strided writes, which split into SIMD4.
using a_halves = simd<sycl::half, GROUPED_GEMM_RC * GROUPED_GEMM_QK>;

// f16 of the f32 product code * scale, each row's scale over its 16 values of a k half: the
// expression the joint_matrix stages evaluate before E::make rounds it
ESIMD_INLINE simd<sycl::half, GROUPED_GEMM_RC * 16> scale_half(simd<float, GROUPED_GEMM_RC * 16> v,
                                                               simd<float, GROUPED_GEMM_RC>      d) {
    v *= d.template replicate_vs_w_hs<GROUPED_GEMM_RC, 1, 16, 0>(0);
    return v;
}

// A 16-entry nibble codebook (kvalues_iq4nl, kvalues_mxfp4) by byte, in SLM: entry b holds the values of
// its low and its high nibble as the two 16-bit halves of one dword, as T (half, or int16 for an f32
// product). A read-only table every lane indexes, which is what SLM is for: register-indirect iselect()
// over the same lanes cost 7.3 of 12.7 ms, a 16-entry SLM table per nibble about 0.4.
template <typename T> ESIMD_INLINE void pair_table_to_slm(const int8_t * kv) {
    static_assert(sizeof(T) == 2, "two values a dword");
    simd<uint32_t, 16> lo;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        lo[i] = sycl::bit_cast<uint16_t>((T) kv[i]);
    }
#pragma unroll
    for (int hi = 0; hi < 16; ++hi) {
        slm_block_store<uint32_t, 16>(hi * 64, lo | ((uint32_t) sycl::bit_cast<uint16_t>((T) kv[hi]) << 16));
    }
}

// 16 packed bytes a row; byte j holds k = j in its low and k = 16 + j in its high nibble (the order
// of fg_decode_iq4_xs and of the iq4_nl stage). One gather per byte; the low halves are k half 0,
// the high halves k half 1, each [row][16].
template <typename T>
ESIMD_INLINE simd<T, GROUPED_GEMM_RC * GROUPED_GEMM_QK> pair_codes(simd<uint32_t, GROUPED_GEMM_RC * 4> q) {
    simd<uint32_t, GROUPED_GEMM_RC * 16>       off = q.template bit_cast_view<uint8_t>();
    simd<uint32_t, GROUPED_GEMM_RC * 16>       pr  = slm_gather<uint32_t, GROUPED_GEMM_RC * 16>(off * 4);
    auto                       pv  = pr.template bit_cast_view<T>();
    simd<T, GROUPED_GEMM_RC * GROUPED_GEMM_QK> v;
    v.template select<GROUPED_GEMM_RC * 16, 1>(0)                    = pv.template select<GROUPED_GEMM_RC * 16, 2>(0);
    v.template select<GROUPED_GEMM_RC * 16, 1>(GROUPED_GEMM_RC * 16) = pv.template select<GROUPED_GEMM_RC * 16, 2>(1);
    return v;
}
// A 2D surface: rows `pitch` bytes apart, 16-byte aligned and at least 64 bytes
constexpr bool surface_ok(const int pitch) {
    return pitch % 16 == 0 && pitch >= 64;
}

// byte i of each lane's dword, i a runtime index: a shift by a scalar, not a register pick
template <int MR> ESIMD_INLINE simd<uint32_t, MR> byte_of(simd<uint32_t, MR> w, const int i) {
    return (w >> (8 * i)) & 0xffu;
}

// one of four per-lane dwords by a runtime index, as uniform branches rather than a register pick
template <int MR> ESIMD_INLINE simd<uint32_t, MR> dword_of(simd<uint32_t, MR> * w, const int i) {
    simd<uint32_t, MR> r = w[0];
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
ESIMD_INLINE simd<float, GROUPED_GEMM_RC * GROUPED_GEMM_QK> byte_codes(simd<uint32_t, GROUPED_GEMM_RC * 8> q) {
    auto bm = q.template bit_cast_view<uint8_t, GROUPED_GEMM_RC, GROUPED_GEMM_QK>();
    simd<float, GROUPED_GEMM_RC * GROUPED_GEMM_QK> v;
    v.template select<GROUPED_GEMM_RC * 16, 1>(0) = bm.template select<GROUPED_GEMM_RC, 1, 16, 1>(0, 0);
    v.template select<GROUPED_GEMM_RC * 16, 1>(GROUPED_GEMM_RC * 16) =
        bm.template select<GROUPED_GEMM_RC, 1, 16, 1>(0, 16);
    return v;
}

// f16 of d * code - m in f32, each row's d and m over its 16 values of a k half: the expression of
// the K-quant stages with a min
ESIMD_INLINE simd<sycl::half, GROUPED_GEMM_RC * 16> scale_min_half(simd<float, GROUPED_GEMM_RC * 16> v,
                                                                   simd<float, GROUPED_GEMM_RC>      d,
                                                                   simd<float, GROUPED_GEMM_RC>      m) {
    v = v * d.template replicate_vs_w_hs<GROUPED_GEMM_RC, 1, 16, 0>(0) -
        m.template replicate_vs_w_hs<GROUPED_GEMM_RC, 1, 16, 0>(0);
    return v;
}

// the two k halves of a row group from codes in [k half][row][16 k], times one (d, m) per row and half
ESIMD_INLINE a_halves codes_dm(simd<float, GROUPED_GEMM_RC * GROUPED_GEMM_QK> c,
                               simd<float, GROUPED_GEMM_RC>                   d0,
                               simd<float, GROUPED_GEMM_RC>                   m0,
                               simd<float, GROUPED_GEMM_RC>                   d1,
                               simd<float, GROUPED_GEMM_RC>                   m1) {
    a_halves v;
    v.template select<GROUPED_GEMM_RC * 16, 1>(0) =
        scale_min_half(c.template select<GROUPED_GEMM_RC * 16, 1>(0), d0, m0);
    v.template select<GROUPED_GEMM_RC * 16, 1>(GROUPED_GEMM_RC * 16) =
        scale_min_half(c.template select<GROUPED_GEMM_RC * 16, 1>(GROUPED_GEMM_RC * 16), d1, m1);
    return v;
}

// the two k halves of a row group, the code bytes times one (d, m) per row and half
ESIMD_INLINE a_halves halves_dm(simd<uint32_t, GROUPED_GEMM_RC * 8> q,
                                simd<float, GROUPED_GEMM_RC>        d0,
                                simd<float, GROUPED_GEMM_RC>        m0,
                                simd<float, GROUPED_GEMM_RC>        d1,
                                simd<float, GROUPED_GEMM_RC>        m1,
                                const float                         bias = 0.0f) {
    simd<float, GROUPED_GEMM_RC * GROUPED_GEMM_QK> c = byte_codes(q);
    if (bias != 0.0f) {
        c -= bias;
    }
    return codes_dm(c, d0, m0, d1, m1);
}

// 16 bytes a row, byte j holding k = j in its low and k = 16 + j in its high nibble: the codes in
// [k half][row][16 k]
ESIMD_INLINE simd<uint32_t, GROUPED_GEMM_RC * GROUPED_GEMM_QK> nibble_codes(simd<uint32_t, GROUPED_GEMM_RC * 4> q) {
    simd<uint32_t, GROUPED_GEMM_RC * 16>              b = q.template bit_cast_view<uint8_t>();
    simd<uint32_t, GROUPED_GEMM_RC * GROUPED_GEMM_QK> c;
    c.template select<GROUPED_GEMM_RC * 16, 1>(0)                    = b & 0xfu;
    c.template select<GROUPED_GEMM_RC * 16, 1>(GROUPED_GEMM_RC * 16) = b >> 4;
    return c;
}

// bit k of each row's dword as one code bit per value, shifted to bit `pos`, in [k half][row][16 k]
ESIMD_INLINE simd<uint32_t, GROUPED_GEMM_RC * GROUPED_GEMM_QK> bit_codes(simd<uint32_t, GROUPED_GEMM_RC> w, const int pos) {
    simd<uint32_t, GROUPED_GEMM_RC * 16> wr = w.template replicate_vs_w_hs<GROUPED_GEMM_RC, 1, 16, 0>(0);
    simd<uint32_t, GROUPED_GEMM_RC * 16> k(0, 1);
    k &= 15;
    simd<uint32_t, GROUPED_GEMM_RC * GROUPED_GEMM_QK> c;
    c.template select<GROUPED_GEMM_RC * 16, 1>(0)                    = ((wr >> k) & 1u) << pos;
    c.template select<GROUPED_GEMM_RC * 16, 1>(GROUPED_GEMM_RC * 16) = ((wr >> (k + 16)) & 1u) << pos;
    return c;
}

// GGML_E8M0_TO_FP32_HALF of each byte
template <int N> ESIMD_INLINE simd<float, N> e8m0_half(simd<uint32_t, N> e) {
    simd<uint32_t, N> bits = e << 23;
    bits.merge(0x00400000u, e == 0);
    simd<float, N> f = bits.template bit_cast_view<float>();
    return f * 0.5f;
}

// ggml_sycl_ue4m3_to_fp32 of each byte (4 exponent bits with bias 7, 3 mantissa bits, 0x7f is 0; halved)
template <int N> ESIMD_INLINE simd<float, N> ue4m3(simd<uint32_t, N> x) {
    const simd<uint32_t, N> e  = (x >> 3) & 15u;
    const simd<uint32_t, N> m  = x & 7u;
    simd<uint32_t, N>       nb = ((e + 119u) << 23) | (m << 20);
    simd<float, N>          r  = nb.template bit_cast_view<float>();
    r.merge(simd<float, N>(m) * (1.0f / 1024.0f), e == 0);
    r.merge(0.0f, x == 0x7fu);
    return r;
}

// iq1_s / iq1_m: a row group's 4 grid dwords a row (iq1s_grid_gpu: 8 nibbles, entry il holding k = 8 il + b in
// the low and k = 8 il + 4 + b in the high nibble of byte b) plus each entry's delta, in [k half][row][16 k]
ESIMD_INLINE simd<float, GROUPED_GEMM_RC * GROUPED_GEMM_QK> iq1_codes(simd<uint32_t, GROUPED_GEMM_RC * 4> g32,
                                                                       simd<float, GROUPED_GEMM_RC * 4>    delta) {
    constexpr int                        RC = GROUPED_GEMM_RC;
    simd<uint32_t, RC * 16>              b  = g32.template bit_cast_view<uint8_t>();
    simd<float, RC * 16>                 lo = b & 0xfu;
    simd<float, RC * 16>                 hi = b >> 4;
    simd<float, RC * 16>                 dl = delta.template replicate_vs_w_hs<RC * 4, 1, 4, 0>(0);
    lo += dl;
    hi += dl;
    auto                                 lm = lo.template bit_cast_view<float, RC, 16>();
    auto                                 hm = hi.template bit_cast_view<float, RC, 16>();
    simd<float, RC * GROUPED_GEMM_QK>    c;
    auto                                 cm = c.template bit_cast_view<float, 2 * RC, 16>();
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        cm.template select<RC, 1, 4, 1>((il / 2) * RC, 8 * (il % 2))     = lm.template select<RC, 1, 4, 1>(0, 4 * il);
        cm.template select<RC, 1, 4, 1>((il / 2) * RC, 8 * (il % 2) + 4) = hm.template select<RC, 1, 4, 1>(0, 4 * il);
    }
    return c;
}

// one dword a sub-block and row of the sign bytes ksigns_iq2xs[7-bit field il] for il in [0, 4) of each row's
// fields: ksigns_iq2xs[i] is i with the parity of its 7 bits as bit 7
ESIMD_INLINE simd<uint32_t, GROUPED_GEMM_RC> ksigns(simd<uint32_t, GROUPED_GEMM_RC * 4> f7) {
    simd<uint32_t, GROUPED_GEMM_RC> sg = 0;
    auto                            fm = f7.template bit_cast_view<uint32_t, GROUPED_GEMM_RC, 4>();
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        simd<uint32_t, GROUPED_GEMM_RC> si = fm.template select<GROUPED_GEMM_RC, 1, 1, 1>(0, il);
        sg |= (si | ((cbit(si) & 1) << 7)) << (8 * il);
    }
    return sg;
}

// the grid dword offsets in SLM of 8-byte entries idx, two a row and entry: [row][2 x entry]
template <int E> ESIMD_INLINE simd<uint32_t, GROUPED_GEMM_RC * 2 * E> grid64_offsets(simd<uint32_t, GROUPED_GEMM_RC * E> idx) {
    return idx.template replicate_vs_w_hs<GROUPED_GEMM_RC * E, 1, 2, 0>(0) * 8u +
           simd<uint32_t, 2>(0, 4).template replicate<GROUPED_GEMM_RC * E>();
}

// a uint64 grid table into SLM, `n` entries
ESIMD_INLINE void grid64_to_slm(const uint64_t * grid, const int n) {
    for (int i = 0; i < 2 * n; i += 64) {
        slm_block_store<uint32_t, 64>(i * 4, block_load<uint32_t, 64>((const uint32_t *) grid + i));
    }
}

// a row group's 32 values from grid dwords (byte b of entry e is k = 4e + b) and a sign dword per
// row (bit k negates k), times d0 for k < 16 and d1 above: the expression of the iq2 and iq3 stages
ESIMD_INLINE a_halves grid_signed(simd<uint32_t, GROUPED_GEMM_RC * 8> ge,
                                  simd<uint32_t, GROUPED_GEMM_RC>     signs,
                                  simd<float, GROUPED_GEMM_RC>        d0,
                                  simd<float, GROUPED_GEMM_RC>        d1) {
    auto                                 gm = ge.template bit_cast_view<uint32_t, GROUPED_GEMM_RC, 8>();
    simd<uint32_t, GROUPED_GEMM_RC * 16> sg = signs.template replicate_vs_w_hs<GROUPED_GEMM_RC, 1, 16, 0>(0);
    simd<uint32_t, GROUPED_GEMM_RC * 16> k(0, 1);
    k &= 15;
    a_halves v;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        simd<uint32_t, GROUPED_GEMM_RC * 4> eh = gm.template select<GROUPED_GEMM_RC, 1, 4, 1>(0, 4 * h);
        simd<float, GROUPED_GEMM_RC * 16>   f  = eh.template bit_cast_view<uint8_t>();
        f.merge(-f, ((sg >> (k + 16 * h)) & 1) != 0);
        v.template select<GROUPED_GEMM_RC * 16, 1>(h * GROUPED_GEMM_RC * 16) = scale_half(f, h == 0 ? d0 : d1);
    }
    return v;
}

ESIMD_INLINE a_halves grid_signed(simd<uint32_t, GROUPED_GEMM_RC * 8> ge,
                                  simd<uint32_t, GROUPED_GEMM_RC>     signs,
                                  simd<float, GROUPED_GEMM_RC>        d) {
    return grid_signed(ge, signs, d, d);
}

// What a weight format leaves out: no SLM table, no per-block metadata (sb_t, load_sb), A decoded a row
// group at a time, and a canonical and a reorder layout that differ (plain: the same bytes, one field
// over the whole block, so either is a 2D surface). bytes: the size of the format's block.
template <typename block_q_t> struct format {
    static constexpr bool supported = true;
    static constexpr bool whole     = false;
    static constexpr bool plain     = false;
    static constexpr int  bytes     = sizeof(block_q_t);
    static constexpr int  slm_bytes = 0;
    static ESIMD_INLINE void init_slm() {}
    struct sb_t {};
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t &, const Sl &, const int) {}
};
}  // namespace grouped_gemm_esimd

template <typename block_q_t, int MR = GROUPED_GEMM_MR> struct grouped_gemm_esimd_a {
    static constexpr bool supported = false;
};

// [qs 32][d], one block a k step
template <int MR> struct grouped_gemm_esimd_a<block_q8_0, MR> : grouped_gemm_esimd::format<block_q8_0> {
    static constexpr int  qk        = QK8_0;
    static constexpr bool whole     = true;  // decode_all() for the whole tile
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 8> q;
        grouped_gemm_esimd::simd<uint16_t, MR>     d;
    };

    static constexpr grouped_gemm_esimd::field QS = { 0, QK8_0, 2 }, D = { QK8_0, 2, 0 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || K >= 64); }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t &, const Sl & s, const int kb, const int) {
        r.q = s.template rows<8>(QS, kb, 0);
        r.d = s.u16(D, kb, 0);
    }
    // all MR x 32 quants at once: int8 -> half, times the row's scale. q * d needs at most 19
    // significant bits, so the f32 product the joint_matrix stage rounds to half is exact and one
    // correctly rounded half multiply gives the same bits.
    static ESIMD_INLINE grouped_gemm_esimd::simd<sycl::half, MR * GROUPED_GEMM_QK> decode_all(raw & r) {
        using namespace grouped_gemm_esimd;
        simd<sycl::half, MR * GROUPED_GEMM_QK> av = r.q.template bit_cast_view<int8_t>();
        simd<sycl::half, MR>         dh = r.d.template bit_cast_view<sycl::half>();
        av *= dh.template replicate_vs_w_hs<MR, 1, GROUPED_GEMM_QK, 0>(0);
        return av;
    }
};

// [qs 16][d], one block a k step
template <int MR> struct grouped_gemm_esimd_a<block_iq4_nl, MR> : grouped_gemm_esimd::format<block_iq4_nl> {
    static constexpr int  qk        = QK4_NL;
    static constexpr int  slm_bytes = 256 * sizeof(uint32_t);

    static ESIMD_INLINE void init_slm() { grouped_gemm_esimd::pair_table_to_slm<sycl::half>(kvalues_iq4nl); }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 4> q;
        grouped_gemm_esimd::simd<sycl::half, MR>   d;
    };

    static constexpr grouped_gemm_esimd::field QS = { 0, QK4_NL / 2, 2 }, D = { QK4_NL / 2, 2, 0 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || K / 2 >= 64); }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t &, const Sl & s, const int kb, const int) {
        using namespace grouped_gemm_esimd;
        r.q                     = s.template rows<4>(QS, kb, 0);
        simd<uint16_t, MR> d = s.u16(D, kb, 0);
        r.d                     = d.template bit_cast_view<sycl::half>();
    }
    // the codebook values are integers below 128, exact in half, and d * value has at most 18
    // significant bits: one correctly rounded half multiply is the stage's f32 product rounded
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        a_halves v = pair_codes<sycl::half>(r.q.template select<GROUPED_GEMM_RC * 4, 1>(g * GROUPED_GEMM_RC * 4));
        simd<sycl::half, GROUPED_GEMM_RC>      dg = r.d.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC);
        simd<sycl::half, GROUPED_GEMM_RC * 16> dr = dg.template replicate_vs_w_hs<GROUPED_GEMM_RC, 1, 16, 0>(0);
        v.template select<GROUPED_GEMM_RC * 16, 1>(0) *= dr;
        v.template select<GROUPED_GEMM_RC * 16, 1>(GROUPED_GEMM_RC * 16) *= dr;
        return v;
    }
};

// [qs 128][d 2][scales_h 2][scales_l 4] per 256 values; k step ib8 of a superblock is sub-block ib8
template <int MR> struct grouped_gemm_esimd_a<block_iq4_xs, MR> : grouped_gemm_esimd::format<block_iq4_xs> {
    static constexpr int  qk        = QK_K;
    // int16 table and f32 products: d * (scale - 32) * value can need 24 significant bits, more
    // than a half multiply keeps
    static constexpr int  slm_bytes = 256 * sizeof(uint32_t);

    static ESIMD_INLINE void init_slm() { grouped_gemm_esimd::pair_table_to_slm<int16_t>(kvalues_iq4nl); }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 4> q;
        grouped_gemm_esimd::simd<float, MR>        d;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<float, MR>    d;
        grouped_gemm_esimd::simd<uint32_t, MR> sh;  // scales_h
        grouped_gemm_esimd::simd<uint32_t, MR> sl;  // scales_l, 4 bytes
    };

    static constexpr grouped_gemm_esimd::field      QS = { 0, QK_K / 2, 8 }, D = { QK_K / 2, 2, 0 },
                                                    SH = { QK_K / 2 + 2, 2, 2 }, SL = { QK_K / 2 + 4, 4, 4 };
    static bool ok(const int K, const bool) { return K % qk == 0; }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
        simd<uint16_t, MR> dh = s.u16(D, ib, 0);
        sb.d  = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
        sb.sh = s.u16(SH, ib, 0);
        sb.sl = s.u32(SL, ib, 0);
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        r.q = s.template rows<4>(QS, kb / 8, 16 * j);
        // d * (6-bit scale - 32), in f32 as the stage computes it
        simd<int32_t, MR> sc = simd<int32_t, MR>(((sb.sl >> (4 * j)) & 0xf) | (((sb.sh >> (2 * j)) & 3) << 4)) - 32;
        r.d = sb.d * simd<float, MR>(sc);
    }

    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        simd<int16_t, GROUPED_GEMM_RC * GROUPED_GEMM_QK> c =
            pair_codes<int16_t>(r.q.template select<GROUPED_GEMM_RC * 4, 1>(g * GROUPED_GEMM_RC * 4));
        simd<float, GROUPED_GEMM_RC> dg = r.d.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC);
        a_halves                     v;
        v.template select<GROUPED_GEMM_RC * 16, 1>(0) = scale_half(c.template select<GROUPED_GEMM_RC * 16, 1>(0), dg);
        v.template select<GROUPED_GEMM_RC * 16, 1>(GROUPED_GEMM_RC * 16) =
            scale_half(c.template select<GROUPED_GEMM_RC * 16, 1>(GROUPED_GEMM_RC * 16), dg);
        return v;
    }
};

// [qs 64][qh 8][signs 32][d 2, scales 4] per 256 values. Value k of sub-block ib8 is byte k % 4 of
// grid entry k / 4, the entry index being qs[k / 4] with bit 8 from qh bit k / 4, negated where bit
// k of the sub-block's sign dword is set; times d * (1 + 2 * 4-bit scale)
template <int MR> struct grouped_gemm_esimd_a<block_iq3_s, MR> : grouped_gemm_esimd::format<block_iq3_s> {
    static constexpr int  qk        = QK_K;
    // the 512-entry grid in SLM, read by every lane: the global gather of it cost 2.4x the kernel
    static constexpr int  slm_bytes = 512 * sizeof(uint32_t);
    static ESIMD_INLINE void init_slm() {
        using namespace grouped_gemm_esimd;
#pragma unroll
        for (int i = 0; i < 512; i += 64) {
            slm_block_store<uint32_t, 64>(i * 4, block_load<uint32_t, 64>(iq3s_grid + i));
        }
    }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 2> q;      // [row][8 qs bytes]
        grouped_gemm_esimd::simd<uint32_t, MR>     qh;     // the sub-block's qh byte
        grouped_gemm_esimd::simd<uint32_t, MR>     signs;  // the sub-block's 32 sign bits
        grouped_gemm_esimd::simd<float, MR>        d;
    };
    // d and the scale bytes are a block's, kept over its k steps; the sign dword and the qh byte of
    // a k step are a 2D load each at that step's column, so no register is picked by a runtime index
    struct sb_t {
        grouped_gemm_esimd::simd<float, MR>    d;
        grouped_gemm_esimd::simd<uint32_t, MR> sc;  // the 4 scale bytes
    };
    // the SoA stream {d, scales} is d and scales of the block in AoS, which are not adjacent there: MS
    // reads the scales at their SoA offset 2 in the stream, which is AoS offset 2 past 104
    static constexpr int             META = QK_K / 4 + QK_K / 32 + QK_K / 8;
    static constexpr grouped_gemm_esimd::field QS = { 0, QK_K / 4, 2 }, QH = { QK_K / 4, QK_K / 32, 2 + QK_K / 4 },
                                               SG = { QK_K / 4 + QK_K / 32, QK_K / 8, 2 + QK_K / 4 + QK_K / 32 },
                                               MD = { META, 2 + IQ3S_N_SCALE, 0 },
                                               MS = { META, 2 + IQ3S_N_SCALE, META };
    // the SoA qh and signs streams are 2D surfaces: their row pitches must be 16-byte aligned
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || ((K / QK_K * 8) % 16 == 0 && K / QK_K * 8 >= 64));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
        simd<uint16_t, MR> dh = s.u16(MD, ib, 0);
        simd<uint32_t, MR> s0 = s.u16(MS, ib, 2);
        simd<uint32_t, MR> s1 = s.u16(MS, ib, 4);
        sb.d  = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
        sb.sc = s0 | (s1 << 16);
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        const int ib = kb / 8;
        r.q     = s.template rows<2>(QS, ib, 8 * j);
        r.qh    = (s.template rows<1>(QH, ib, 4 * (j / 4)) >> (8 * (j % 4))) & 0xffu;
        r.signs = s.template rows<1>(SG, ib, 4 * j);
        r.d     = sb.d * simd<float, MR>(simd<int32_t, MR>(1 + 2 * ((sb.sc >> (4 * j)) & 0xf)));
    }

    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        // the 8 grid entries of each of the row group's rows
        simd<uint16_t, GROUPED_GEMM_RC * 8> qb =
            r.q.template select<GROUPED_GEMM_RC * 2, 1>(g * GROUPED_GEMM_RC * 2).template bit_cast_view<uint8_t>();
        simd<uint32_t, GROUPED_GEMM_RC>     qr = r.qh.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC);
        simd<uint32_t, GROUPED_GEMM_RC * 8> qh = qr.template replicate_vs_w_hs<GROUPED_GEMM_RC, 1, 8, 0>(0);
        simd<uint32_t, GROUPED_GEMM_RC * 8> j(0, 1);
        j &= 7;
        simd<uint32_t, GROUPED_GEMM_RC * 8> idx = simd<uint32_t, GROUPED_GEMM_RC * 8>(qb) | (((qh >> j) & 1) << 8);
        simd<uint32_t, GROUPED_GEMM_RC * 8> ge  = slm_gather<uint32_t, GROUPED_GEMM_RC * 8>(idx * 4);
        return grid_signed(ge, r.signs.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC),
                           r.d.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC));
    }
};

// The K-quants and iq3_xxs: a 256-value superblock, k step j its sub-block j. Their value bytes for
// a k step are one rectangle of a stream, in k order after a shift and a mask per dword; scales,
// mins and d are the block's and are loaded once per block.

// [qs 128][scales 12][dm]: value l of sub-block j is nibble j & 1 of byte 32 (j / 2) + l, times
// dall * scale_j, minus dmin * min_j (6-bit scale and min, get_scale_min_k4)
template <int MR> struct grouped_gemm_esimd_a<block_q4_K, MR> : grouped_gemm_esimd::format<block_q4_K> {
    static constexpr int  qk        = QK_K;
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 8> q;
        grouped_gemm_esimd::simd<float, MR>        d, m;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<uint32_t, MR> sc[3];  // the 12 scale bytes
        grouped_gemm_esimd::simd<float, MR>    dall, dmin;
    };

    static constexpr grouped_gemm_esimd::field QS = { 0, QK_K / 2, 16 }, SC = { QK_K / 2, K_SCALE_SIZE, 4 },
                                               DM = { QK_K / 2 + K_SCALE_SIZE, 4, 0 };

    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / 2));
    }

    template <typename Sl>
    static ESIMD_INLINE void load_sb(sb_t &                            sb,
                                     const Sl &                        s,
                                     const int                         ib,
                                     const grouped_gemm_esimd::field & sc  = SC,
                                     const grouped_gemm_esimd::field & dmf = DM) {
        using namespace grouped_gemm_esimd;
#pragma unroll
        for (int w = 0; w < 3; ++w) {
            sb.sc[w] = s.u32(sc, ib, 4 * w);
        }
        simd<uint32_t, MR> dm = s.u32(dmf, ib, 0);
        auto                  dh = dm.template bit_cast_view<sycl::half>();
        sb.dall                  = simd<sycl::half, MR>(dh.template select<MR, 2>(0));
        sb.dmin                  = simd<sycl::half, MR>(dh.template select<MR, 2>(1));
    }
    // get_scale_min_k4 for sub-block j over the lanes
    static ESIMD_INLINE void scale_min(sb_t &                                sb,
                                       const int                             j,
                                       grouped_gemm_esimd::simd<float, MR> & d,
                                       grouped_gemm_esimd::simd<float, MR> & m) {
        using namespace grouped_gemm_esimd;
        simd<uint32_t, MR> sc, mb;
        if (j < 4) {
            sc = byte_of(sb.sc[0], j) & 63;
            mb = byte_of(sb.sc[1], j) & 63;
        } else {
            sc = (byte_of(sb.sc[2], j - 4) & 0xf) | ((byte_of(sb.sc[0], j - 4) >> 6) << 4);
            mb = (byte_of(sb.sc[2], j - 4) >> 4) | ((byte_of(sb.sc[1], j - 4) >> 6) << 4);
        }
        d = sb.dall * simd<float, MR>(sc);
        m = sb.dmin * simd<float, MR>(mb);
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        r.q = (s.template rows<8>(QS, kb / 8, 32 * (j / 2)) >> (4 * (j & 1))) & 0x0F0F0F0Fu;
        scale_min(sb, j, r.d, r.m);
    }

    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        simd<float, GROUPED_GEMM_RC> d = r.d.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC),
                                     m = r.m.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC);
        return halves_dm(r.q.template select<GROUPED_GEMM_RC * 8, 1>(g * GROUPED_GEMM_RC * 8), d, m, d, m);
    }
};

// [qs 128][qh 32][scales 12][dm]: q4_K plus bit j of qh byte l as 16
template <int MR> struct grouped_gemm_esimd_a<block_q5_K, MR> : grouped_gemm_esimd::format<block_q5_K> {
    using q4                           = grouped_gemm_esimd_a<block_q4_K, MR>;
    static constexpr int  qk        = QK_K;
    using raw  = typename q4::raw;
    using sb_t = typename q4::sb_t;
    static constexpr grouped_gemm_esimd::field QS = { 0, QK_K / 2, 48 }, QH = { QK_K / 2, QK_K / 8, 16 },
                                               SC = { QK_K / 2 + QK_K / 8, K_SCALE_SIZE, 4 },
                                               DM = { QK_K / 2 + QK_K / 8 + K_SCALE_SIZE, 4, 0 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || (grouped_gemm_esimd::surface_ok(K / 2) && grouped_gemm_esimd::surface_ok(K / 8)));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) { q4::load_sb(sb, s, ib, SC, DM); }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        const int                 ib = kb / 8;
        simd<uint32_t, MR * 8> qh = s.template rows<8>(QH, ib, 0);
        r.q = ((s.template rows<8>(QS, ib, 32 * (j / 2)) >> (4 * (j & 1))) & 0x0F0F0F0Fu) |
              (((qh >> j) & 0x01010101u) << 4);
        q4::scale_min(sb, j, r.d, r.m);
    }

    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        return q4::template decode<g>(r);
    }
};

// [ql 128][qh 64][scales 16][d]: sub-block j is quarter r = j % 4 of half h = j / 4; value l is
// nibble r >> 1 of ql byte 64h + 32 (r & 1) + l with bits 2r, 2r+1 of qh byte 32h + l above it,
// minus 32, times d * the int8 scale 2j + l / 16
template <int MR> struct grouped_gemm_esimd_a<block_q6_K, MR> : grouped_gemm_esimd::format<block_q6_K> {
    static constexpr int  qk        = QK_K;
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 8> q;
        grouped_gemm_esimd::simd<float, MR>        d0, d1;  // per k half
    };
    struct sb_t {
        grouped_gemm_esimd::simd<uint32_t, MR> sc[4];  // the 16 int8 scales
        grouped_gemm_esimd::simd<float, MR>    d;
    };

    static constexpr grouped_gemm_esimd::field QL = { 0, QK_K / 2, 0 }, QH = { QK_K / 2, QK_K / 4, QK_K / 2 },
                                               SC = { QK_K / 2 + QK_K / 4, QK_K / 16, QK_K / 2 + QK_K / 4 },
                                               D  = { QK_K / 2 + QK_K / 4 + QK_K / 16, 2,
                                                      QK_K / 2 + QK_K / 4 + QK_K / 16 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || (grouped_gemm_esimd::surface_ok(K / 2) && grouped_gemm_esimd::surface_ok(K / 4)));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            sb.sc[w] = s.u32(SC, ib, 4 * w);
        }
        simd<uint16_t, MR> dh = s.u16(D, ib, 0);
        sb.d = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        const int ib = kb / 8, h = j / 4, q = j % 4;
        simd<uint32_t, MR * 8> ql = s.template rows<8>(QL, ib, 64 * h + 32 * (q & 1));
        simd<uint32_t, MR * 8> qh = s.template rows<8>(QH, ib, 32 * h);
        r.q = ((ql >> (4 * (q >> 1))) & 0x0F0F0F0Fu) | (((qh >> (2 * q)) & 0x03030303u) << 4);
        // scales 2j and 2j + 1: bytes 2 (j % 2) and 2 (j % 2) + 1 of dword j / 2, sign extended
        simd<uint32_t, MR> w  = dword_of(sb.sc, j / 2);
        simd<int32_t, MR>  s0 = simd<int32_t, MR>(w << (24 - 16 * (j % 2))) >> 24;
        simd<int32_t, MR>  s1 = simd<int32_t, MR>(w << (16 - 16 * (j % 2))) >> 24;
        r.d0 = sb.d * simd<float, MR>(s0);
        r.d1 = sb.d * simd<float, MR>(s1);
    }

    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        simd<float, GROUPED_GEMM_RC> z = 0.0f;
        return halves_dm(r.q.template select<GROUPED_GEMM_RC * 8, 1>(g * GROUPED_GEMM_RC * 8),
                         r.d0.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC), z,
                         r.d1.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC), z, 32.0f);
    }
};

// [qs 64][scales 16][dm]: value l of sub-block j is bits 2 (j % 4) of byte 32 (j / 4) + l, times
// d * (scale & 15), minus dmin * (scale >> 4), scale byte 2j + l / 16
template <int MR> struct grouped_gemm_esimd_a<block_q2_K, MR> : grouped_gemm_esimd::format<block_q2_K> {
    static constexpr int  qk        = QK_K;
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 8> q;
        grouped_gemm_esimd::simd<float, MR>        d0, m0, d1, m1;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<uint32_t, MR> sc[4];
        grouped_gemm_esimd::simd<float, MR>    d, dmin;
    };

    static constexpr grouped_gemm_esimd::field QS = { 0, QK_K / 4, QK_K / 16 }, SC = { QK_K / 4, QK_K / 16, 0 },
                                               DM = { QK_K / 4 + QK_K / 16, 4, QK_K / 4 + QK_K / 16 };

    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / 4));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            sb.sc[w] = s.u32(SC, ib, 4 * w);
        }
        simd<uint32_t, MR> dm = s.u32(DM, ib, 0);
        auto                  dh = dm.template bit_cast_view<sycl::half>();
        sb.d                     = simd<sycl::half, MR>(dh.template select<MR, 2>(0));
        sb.dmin                  = simd<sycl::half, MR>(dh.template select<MR, 2>(1));
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        r.q = (s.template rows<8>(QS, kb / 8, 32 * (j / 4)) >> (2 * (j % 4))) & 0x03030303u;
        simd<uint32_t, MR> w  = dword_of(sb.sc, j / 2);
        simd<uint32_t, MR> s0 = byte_of(w, 2 * (j % 2)), s1 = byte_of(w, 2 * (j % 2) + 1);
        r.d0 = sb.d * simd<float, MR>(s0 & 0xf);
        r.m0 = sb.dmin * simd<float, MR>(s0 >> 4);
        r.d1 = sb.d * simd<float, MR>(s1 & 0xf);
        r.m1 = sb.dmin * simd<float, MR>(s1 >> 4);
    }

    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        return halves_dm(r.q.template select<GROUPED_GEMM_RC * 8, 1>(g * GROUPED_GEMM_RC * 8),
                         r.d0.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC),
                         r.m0.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC),
                         r.d1.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC),
                         r.m1.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC));
    }
};

// [qs 64][hmask 32][scales 12][d]: q2_K's 2 bits, minus 4 where bit j of hmask byte l is clear,
// times d * (6-bit scale - 32), scale s = 2j + l / 16 split over bytes s % 8 and 8 + s % 4
template <int MR> struct grouped_gemm_esimd_a<block_q3_K, MR> : grouped_gemm_esimd::format<block_q3_K> {
    static constexpr int  qk        = QK_K;
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 8> q;
        grouped_gemm_esimd::simd<float, MR>        d0, d1;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<uint32_t, MR> sc[4];  // 12 scale bytes, sc[3] unused
        grouped_gemm_esimd::simd<float, MR>    d;
    };

    static constexpr grouped_gemm_esimd::field QS = { 0, QK_K / 4, QK_K / 8 }, HM = { QK_K / 4, QK_K / 8, 0 },
                                               SC = { QK_K / 4 + QK_K / 8, 12, QK_K / 4 + QK_K / 8 },
                                               D  = { QK_K / 4 + QK_K / 8 + 12, 2, QK_K / 4 + QK_K / 8 + 12 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || (grouped_gemm_esimd::surface_ok(K / 4) && grouped_gemm_esimd::surface_ok(K / 8)));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
#pragma unroll
        for (int w = 0; w < 3; ++w) {
            sb.sc[w] = s.u32(SC, ib, 4 * w);
        }
        sb.sc[3] = 0;
        simd<uint16_t, MR> dh = s.u16(D, ib, 0);
        sb.d = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
    }

    static ESIMD_INLINE grouped_gemm_esimd::simd<float, MR> scale(sb_t & sb, const int sidx) {
        using namespace grouped_gemm_esimd;
        simd<uint32_t, MR> lo = (byte_of(dword_of(sb.sc, (sidx % 8) / 4), sidx % 4) >> (4 * (sidx / 8))) & 0xf;
        simd<uint32_t, MR> hi = (byte_of(sb.sc[2], sidx % 4) >> (2 * (sidx / 4))) & 3;
        return sb.d * simd<float, MR>(simd<int32_t, MR>(lo | (hi << 4)) - 32);
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        const int                 ib = kb / 8;
        simd<uint32_t, MR * 8> hm = s.template rows<8>(HM, ib, 0);
        r.q = ((s.template rows<8>(QS, ib, 32 * (j / 4)) >> (2 * (j % 4))) & 0x03030303u) |
              (((hm >> j) & 0x01010101u) << 2);
        r.d0 = scale(sb, 2 * j);
        r.d1 = scale(sb, 2 * j + 1);
    }

    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        simd<float, GROUPED_GEMM_RC> z = 0.0f;
        return halves_dm(r.q.template select<GROUPED_GEMM_RC * 8, 1>(g * GROUPED_GEMM_RC * 8),
                         r.d0.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC), z,
                         r.d1.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC), z, 4.0f);
    }
};

// [qs 96][d]: per sub-block j, 8 grid indices at 8j and an aux dword at 64 + 4j whose 4 x 7 bits
// are sign bytes of ksigns_iq2xs and whose top 4 bits scale d * (0.5 + s) * 0.5. ksigns_iq2xs[i] is
// i with the parity of its 7 bits as bit 7, so the sign bytes are computed, not looked up.
template <int MR> struct grouped_gemm_esimd_a<block_iq3_xxs, MR> : grouped_gemm_esimd::format<block_iq3_xxs> {
    static constexpr int  qk        = QK_K;
    // the 256-entry grid, a read-only table in SLM
    static constexpr int  slm_bytes = 256 * sizeof(uint32_t);
    static ESIMD_INLINE void init_slm() {
        using namespace grouped_gemm_esimd;
#pragma unroll
        for (int i = 0; i < 256; i += 64) {
            slm_block_store<uint32_t, 64>(i * 4, block_load<uint32_t, 64>(iq3xxs_grid + i));
        }
    }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 2> q;  // [row][8 grid indices]
        grouped_gemm_esimd::simd<uint32_t, MR>     aux;
        grouped_gemm_esimd::simd<float, MR>        d;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<float, MR> d;
    };

    static constexpr grouped_gemm_esimd::field QS = { 0, 3 * QK_K / 8, 2 }, D = { 3 * QK_K / 8, 2, 0 };

    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / QK_K * 96));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
        simd<uint16_t, MR> dh = s.u16(D, ib, 0);
        sb.d = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        const int ib = kb / 8;
        r.q   = s.template rows<2>(QS, ib, 8 * j);
        r.aux = s.template rows<1>(QS, ib, 64 + 4 * j);
        r.d   = sb.d * (0.5f + simd<float, MR>(r.aux >> 28)) * 0.5f;
    }

    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        simd<uint32_t, GROUPED_GEMM_RC * 8> qb =
            r.q.template select<GROUPED_GEMM_RC * 2, 1>(g * GROUPED_GEMM_RC * 2).template bit_cast_view<uint8_t>();
        simd<uint32_t, GROUPED_GEMM_RC * 8> ge = slm_gather<uint32_t, GROUPED_GEMM_RC * 8>(qb * 4);
        simd<uint32_t, GROUPED_GEMM_RC>     ax = r.aux.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC);
        simd<uint32_t, GROUPED_GEMM_RC>     sg = 0;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            simd<uint32_t, GROUPED_GEMM_RC> si = (ax >> (7 * il)) & 127;
            sg |= (si | ((cbit(si) & 1) << 7)) << (8 * il);
        }
        return grid_signed(ge, sg, r.d.template select<GROUPED_GEMM_RC, 1>(g * GROUPED_GEMM_RC));
    }
};

// q4_0, q4_1, q5_0, q5_1: [qs 16][qh 4 (q5)][d or dm], one block a k step. Byte j of qs holds k = j (low nibble)
// and k = 16 + j (high nibble), bit k of qh the fifth bit of k; the value is (code - 2^(bits - 1)) * d, or with a
// min code * d + m. QS_AOS and QH_AOS (-1: none) are the fields' offsets in the canonical block.
template <typename block_q_t, int MR, int QS_AOS, int QH_AOS, bool MIN>
struct grouped_gemm_esimd_nibbles : grouped_gemm_esimd::format<block_q_t> {
    static constexpr int  qk   = 32;
    static constexpr bool QH   = QH_AOS >= 0;
    static constexpr int  BIAS = QH ? 16 : 8;
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 4> q;
        grouped_gemm_esimd::simd<uint32_t, MR>     qh;
        grouped_gemm_esimd::simd<float, MR>        d;
        grouped_gemm_esimd::simd<float, MR>        m;
    };
    static constexpr grouped_gemm_esimd::field QS = { 0, 16, QS_AOS }, QHF = { 16, 4, QH ? QH_AOS : 0 },
                                               D = { QH ? 20 : 16, MIN ? 4 : 2, 0 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / 2)); }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, typename grouped_gemm_esimd_nibbles::sb_t &, const Sl & s,
                                                         const int kb, const int) {
        using namespace grouped_gemm_esimd;
        r.q = s.template rows<4>(QS, kb, 0);
        if constexpr (QH) {
            r.qh = s.u32(QHF, kb, 0);
        }
        if constexpr (MIN) {
            simd<uint32_t, MR> dm = s.u32(D, kb, 0);
            simd<uint16_t, MR> dl = dm, mh = dm >> 16;
            r.d = simd<sycl::half, MR>(dl.template bit_cast_view<sycl::half>());
            r.m = simd<sycl::half, MR>(mh.template bit_cast_view<sycl::half>());
        } else {
            simd<uint16_t, MR> dh = s.u16(D, kb, 0);
            r.d = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
        }
    }
    // in f32 as the stages compute it, (code - bias) * d or code * d + m, then rounded to half
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        constexpr int                RC = GROUPED_GEMM_RC;
        simd<uint32_t, RC * GROUPED_GEMM_QK> c = nibble_codes(r.q.template select<RC * 4, 1>(g * RC * 4));
        if constexpr (QH) {
            c |= bit_codes(r.qh.template select<RC, 1>(g * RC), 4);
        }
        simd<float, RC * GROUPED_GEMM_QK> f = c;
        simd<float, RC>                   d = r.d.template select<RC, 1>(g * RC);
        simd<float, RC>                   m = 0.0f;
        if constexpr (MIN) {
            m = -r.m.template select<RC, 1>(g * RC);
        } else {
            f -= (float) BIAS;
        }
        return codes_dm(f, d, m, d, m);
    }
};

static_assert(offsetof(block_q4_0, qs) == 2 && offsetof(block_q4_1, qs) == 4, "q4 AoS offsets");
static_assert(offsetof(block_q5_0, qh) == 2 && offsetof(block_q5_0, qs) == 6, "q5_0 AoS offsets");
static_assert(offsetof(block_q5_1, qh) == 4 && offsetof(block_q5_1, qs) == 8, "q5_1 AoS offsets");
template <int MR> struct grouped_gemm_esimd_a<block_q4_0, MR> : grouped_gemm_esimd_nibbles<block_q4_0, MR, 2, -1, false> {};
template <int MR> struct grouped_gemm_esimd_a<block_q4_1, MR> : grouped_gemm_esimd_nibbles<block_q4_1, MR, 4, -1, true> {};
template <int MR> struct grouped_gemm_esimd_a<block_q5_0, MR> : grouped_gemm_esimd_nibbles<block_q5_0, MR, 6, 2, false> {};
template <int MR> struct grouped_gemm_esimd_a<block_q5_1, MR> : grouped_gemm_esimd_nibbles<block_q5_1, MR, 8, 4, true> {};

// [qs 16][d] per 128 values, four k steps: bit k of the step's dword picks +d or -d
template <int MR> struct grouped_gemm_esimd_a<block_q1_0, MR> : grouped_gemm_esimd::format<block_q1_0> {
    static constexpr int qk = QK1_0;
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR> q;
        grouped_gemm_esimd::simd<float, MR>    d;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<float, MR> d;
    };
    static constexpr grouped_gemm_esimd::field QS = { 0, QK1_0 / 8, 2 }, D = { QK1_0 / 8, 2, 0 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / 8)); }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
        simd<uint16_t, MR> dh = s.u16(D, ib, 0);
        sb.d = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        r.q = s.u32(QS, kb / (QK1_0 / GROUPED_GEMM_QK), 4 * j);
        r.d = sb.d;
    }
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        constexpr int                     RC = GROUPED_GEMM_RC;
        simd<float, RC * GROUPED_GEMM_QK> f  = bit_codes(r.q.template select<RC, 1>(g * RC), 1);
        f -= 1.0f;
        simd<float, RC> d = r.d.template select<RC, 1>(g * RC);
        return codes_dm(f, d, 0.0f, d, 0.0f);
    }
};

// [qs 16][d] per 64 values, two k steps of two dwords of 16 2-bit codes; value (code - 1) * d
template <int MR> struct grouped_gemm_esimd_a<block_q2_0, MR> : grouped_gemm_esimd::format<block_q2_0> {
    static constexpr int qk = QK2_0;
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 2> q;
        grouped_gemm_esimd::simd<float, MR>        d;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<float, MR> d;
    };
    static constexpr grouped_gemm_esimd::field QS = { 0, QK2_0 / 4, 2 }, D = { QK2_0 / 4, 2, 0 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / 4)); }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
        simd<uint16_t, MR> dh = s.u16(D, ib, 0);
        sb.d = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        r.q = s.template rows<2>(QS, kb / 2, 8 * j);
        r.d = sb.d;
    }
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        constexpr int                      RC = GROUPED_GEMM_RC;
        const simd<uint32_t, RC * 16>      sh = simd<uint32_t, 16>(0, 2).template replicate<RC>();
        simd<float, RC * GROUPED_GEMM_QK>  f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            simd<uint32_t, RC>      w  = r.q.template select<RC, 2>(g * RC * 2 + h);
            simd<uint32_t, RC * 16> wr = w.template replicate_vs_w_hs<RC, 1, 16, 0>(0);
            f.template select<RC * 16, 1>(h * RC * 16) = (wr >> sh) & 3u;
        }
        f -= 1.0f;
        simd<float, RC> d = r.d.template select<RC, 1>(g * RC);
        return codes_dm(f, d, 0.0f, d, 0.0f);
    }
};

// [qs 16][e], one block a k step, in the MoE reorder layout only (the 17-byte canonical block puts its
// fields on odd bytes): byte j holds k = j and k = 16 + j as kvalues_mxfp4 indices, times the E8M0 scale
// halved, in f32 as the stage computes it
template <int MR> struct grouped_gemm_esimd_a<block_mxfp4, MR> : grouped_gemm_esimd::format<block_mxfp4> {
    static constexpr int qk        = QK_MXFP4;
    static constexpr int slm_bytes = 256 * sizeof(uint32_t);
    static ESIMD_INLINE void init_slm() { grouped_gemm_esimd::pair_table_to_slm<int16_t>(kvalues_mxfp4); }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 4> q;
        grouped_gemm_esimd::simd<float, MR>        d;
    };
    static constexpr grouped_gemm_esimd::field QS = { 0, QK_MXFP4 / 2, 1 }, E = { QK_MXFP4 / 2, 1, 0 };
    static bool ok(const int K, const bool aos) { return !aos && K % qk == 0 && grouped_gemm_esimd::surface_ok(K / 2); }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t &, const Sl & s, const int kb, const int) {
        r.q = s.template rows<4>(QS, kb, 0);
        r.d = grouped_gemm_esimd::e8m0_half(s.u8(E, kb, 0));
    }
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        constexpr int                     RC = GROUPED_GEMM_RC;
        simd<float, RC * GROUPED_GEMM_QK> f  = pair_codes<int16_t>(r.q.template select<RC * 4, 1>(g * RC * 4));
        simd<float, RC>                   d  = r.d.template select<RC, 1>(g * RC);
        return codes_dm(f, d, 0.0f, d, 0.0f);
    }
};

// [qs 32][d 4] per 64 values, two k steps of two 16-value sub-blocks with a UE4M3 scale byte each; byte j of a
// sub-block holds its value j (low nibble) and j + 8 (high nibble) as kvalues_mxfp4 indices
template <int MR> struct grouped_gemm_esimd_a<block_nvfp4, MR> : grouped_gemm_esimd::format<block_nvfp4> {
    static constexpr int qk        = QK_NVFP4;
    static constexpr int slm_bytes = 256 * sizeof(uint32_t);
    static ESIMD_INLINE void init_slm() { grouped_gemm_esimd::pair_table_to_slm<int16_t>(kvalues_mxfp4); }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 4> q;
        grouped_gemm_esimd::simd<float, MR>        d0;
        grouped_gemm_esimd::simd<float, MR>        d1;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<uint32_t, MR> d;
    };
    static_assert(offsetof(block_nvfp4, qs) == 4, "nvfp4 AoS offsets");
    static constexpr grouped_gemm_esimd::field QS = { 0, QK_NVFP4 / 2, 4 }, D = { QK_NVFP4 / 2, 4, 0 };
    static bool ok(const int K, const bool aos) { return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / 2)); }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) { sb.d = s.u32(D, ib, 0); }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        r.q  = s.template rows<4>(QS, kb / 2, 16 * j);
        r.d0 = grouped_gemm_esimd::ue4m3((sb.d >> (16 * j)) & 0xffu);
        r.d1 = grouped_gemm_esimd::ue4m3((sb.d >> (16 * j + 8)) & 0xffu);
    }
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        constexpr int           RC  = GROUPED_GEMM_RC;
        simd<uint32_t, RC * 4>  q   = r.q.template select<RC * 4, 1>(g * RC * 4);
        simd<uint32_t, RC * 16> off = q.template bit_cast_view<uint8_t>();
        simd<uint32_t, RC * 16> pr  = slm_gather<uint32_t, RC * 16>(off * 4);
        auto                    pm  = pr.template bit_cast_view<int16_t, RC, 32>(); // [row][byte][low, high]
        simd<float, RC * GROUPED_GEMM_QK> f;
        auto                              fm = f.template bit_cast_view<float, 2 * RC, 16>();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            fm.template select<RC, 1, 8, 1>(h * RC, 0) = pm.template select<RC, 1, 8, 2>(0, 16 * h);
            fm.template select<RC, 1, 8, 1>(h * RC, 8) = pm.template select<RC, 1, 8, 2>(0, 16 * h + 1);
        }
        return codes_dm(f, r.d0.template select<RC, 1>(g * RC), 0.0f, r.d1.template select<RC, 1>(g * RC), 0.0f);
    }
};

// f16 / bf16 weights: 32 values a block of the format, the same bytes in both layouts, nothing to decode; the
// kernel takes them as the dpas operand (see F::plain there)
template <typename E, int MR> struct grouped_gemm_esimd_plain : grouped_gemm_esimd::format<E> {
    using elem                       = E;
    static constexpr int  qk         = GROUPED_GEMM_QK;
    static constexpr int  bytes      = GROUPED_GEMM_QK * sizeof(E);
    static constexpr bool whole      = true;
    static constexpr bool plain      = true;
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 16> q;
    };
    static constexpr grouped_gemm_esimd::field QS = { 0, bytes, 0 };
    static bool ok(const int K, const bool) { return K % qk == 0 && grouped_gemm_esimd::surface_ok(K * (int) sizeof(E)); }
    template <typename Sl>
    static ESIMD_INLINE void load(raw & r, typename grouped_gemm_esimd_plain::sb_t &, const Sl & s, const int kb, const int) {
        r.q = s.template rows<16>(QS, kb, 0);
    }
};
template <int MR> struct grouped_gemm_esimd_a<sycl::half, MR> : grouped_gemm_esimd_plain<sycl::half, MR> {};
template <int MR>
struct grouped_gemm_esimd_a<sycl::ext::oneapi::bfloat16, MR> : grouped_gemm_esimd_plain<sycl::ext::oneapi::bfloat16, MR> {};

// [d][qs 64] per 256 values: per sub-block j, 4 grid indices at 8 j and an aux dword at 8 j + 4 whose 4 x 7 bits
// are ksigns_iq2xs sign bytes and whose top 4 bits scale d * (0.5 + s) * 0.25; iq2xxs_grid entries are 8 bytes
template <int MR> struct grouped_gemm_esimd_a<block_iq2_xxs, MR> : grouped_gemm_esimd::format<block_iq2_xxs> {
    static constexpr int qk        = QK_K;
    static constexpr int slm_bytes = 256 * 8;
    static ESIMD_INLINE void init_slm() { grouped_gemm_esimd::grid64_to_slm(iq2xxs_grid, 256); }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR> q;
        grouped_gemm_esimd::simd<uint32_t, MR> aux;
        grouped_gemm_esimd::simd<float, MR>    d;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<float, MR> d;
    };
    static constexpr grouped_gemm_esimd::field QS = { 0, QK_K / 4, 2 }, D = { QK_K / 4, 2, 0 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / QK_K * (QK_K / 4)));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
        simd<uint16_t, MR> dh = s.u16(D, ib, 0);
        sb.d = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        r.q   = s.template rows<1>(QS, kb / 8, 8 * j);
        r.aux = s.template rows<1>(QS, kb / 8, 8 * j + 4);
        r.d   = sb.d * (0.5f + simd<float, MR>(r.aux >> 28)) * 0.25f;
    }
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        constexpr int          RC  = GROUPED_GEMM_RC;
        simd<uint32_t, RC>     q   = r.q.template select<RC, 1>(g * RC);
        simd<uint32_t, RC * 4> idx = q.template bit_cast_view<uint8_t>();
        simd<uint32_t, RC * 8> ge  = slm_gather<uint32_t, RC * 8>(grid64_offsets<4>(idx));
        simd<uint32_t, RC>     ax  = r.aux.template select<RC, 1>(g * RC);
        simd<uint32_t, RC * 4> f7  = (ax.template replicate_vs_w_hs<RC, 1, 4, 0>(0) >>
                                      simd<uint32_t, 4>(simd<int32_t, 4>(0, 7)).template replicate<RC>()) & 127u;
        return grid_signed(ge, ksigns(f7), r.d.template select<RC, 1>(g * RC));
    }
};

// [d][qs 64][scales 8] per 256 values: per sub-block j, 4 uint16 at 8 j, each a 9-bit iq2xs_grid index and 7 bits
// of a ksigns_iq2xs sign byte; values 0..15 times d * (0.5 + low scale nibble) * 0.25, 16..31 the high nibble
template <int MR> struct grouped_gemm_esimd_a<block_iq2_xs, MR> : grouped_gemm_esimd::format<block_iq2_xs> {
    static constexpr int qk        = QK_K;
    static constexpr int slm_bytes = 512 * 8;
    static ESIMD_INLINE void init_slm() { grouped_gemm_esimd::grid64_to_slm(iq2xs_grid, 512); }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR * 2> q;
        grouped_gemm_esimd::simd<float, MR>        d0;
        grouped_gemm_esimd::simd<float, MR>        d1;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<float, MR>    d;
        grouped_gemm_esimd::simd<uint32_t, MR> sc0;
        grouped_gemm_esimd::simd<uint32_t, MR> sc1;
    };
    static constexpr grouped_gemm_esimd::field QS = { 0, QK_K / 4, 2 }, SC = { QK_K / 4, QK_K / 32, 2 + QK_K / 4 },
                                               D = { QK_K / 4 + QK_K / 32, 2, 0 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / QK_K * (QK_K / 4)));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
        simd<uint16_t, MR> dh = s.u16(D, ib, 0);
        sb.d   = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
        sb.sc0 = s.u32(SC, ib, 0);
        sb.sc1 = s.u32(SC, ib, 4);
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        r.q                         = s.template rows<2>(QS, kb / 8, 8 * j);
        const simd<uint32_t, MR> sc = ((j < 4 ? sb.sc0 : sb.sc1) >> (8 * (j % 4))) & 0xffu;
        r.d0                        = sb.d * (0.5f + simd<float, MR>(sc & 0xfu)) * 0.25f;
        r.d1                        = sb.d * (0.5f + simd<float, MR>(sc >> 4)) * 0.25f;
    }
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        constexpr int          RC  = GROUPED_GEMM_RC;
        simd<uint32_t, RC * 2> q   = r.q.template select<RC * 2, 1>(g * RC * 2);
        simd<uint32_t, RC * 4> q16 = q.template bit_cast_view<uint16_t>();
        simd<uint32_t, RC * 8> ge  = slm_gather<uint32_t, RC * 8>(grid64_offsets<4>(q16 & 511u));
        return grid_signed(ge, ksigns(q16 >> 9), r.d0.template select<RC, 1>(g * RC), r.d1.template select<RC, 1>(g * RC));
    }
};

// [d][qs 64][qh 8][scales 8] per 256 values: per sub-block j, grid entry l (8 values) is iq2s_grid[qs[4 j + l] | 2
// bits of qh[j]], value k negated where bit k of the sign dword at qs[32 + 4 j] is set; values 0..15 times
// d * (0.5 + low scale nibble) * 0.25 and 16..31 the high nibble
template <int MR> struct grouped_gemm_esimd_a<block_iq2_s, MR> : grouped_gemm_esimd::format<block_iq2_s> {
    static constexpr int qk        = QK_K;
    static constexpr int slm_bytes = 1024 * 8;
    static ESIMD_INLINE void init_slm() { grouped_gemm_esimd::grid64_to_slm(iq2s_grid, 1024); }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR> q;
        grouped_gemm_esimd::simd<uint32_t, MR> qh;
        grouped_gemm_esimd::simd<uint32_t, MR> signs;
        grouped_gemm_esimd::simd<float, MR>    d0;
        grouped_gemm_esimd::simd<float, MR>    d1;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<float, MR>    d;
        grouped_gemm_esimd::simd<uint32_t, MR> sc0;
        grouped_gemm_esimd::simd<uint32_t, MR> sc1;
        grouped_gemm_esimd::simd<uint32_t, MR> qh0;
        grouped_gemm_esimd::simd<uint32_t, MR> qh1;
    };
    static constexpr grouped_gemm_esimd::field QS = { 0, QK_K / 4, 2 }, QH = { QK_K / 4, QK_K / 32, 2 + QK_K / 4 },
                                               SC = { QK_K / 4 + QK_K / 32, QK_K / 32, 2 + QK_K / 4 + QK_K / 32 },
                                               D  = { QK_K / 4 + QK_K / 16, 2, 0 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / QK_K * (QK_K / 4)));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
        simd<uint16_t, MR> dh = s.u16(D, ib, 0);
        sb.d   = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
        sb.sc0 = s.u32(SC, ib, 0);
        sb.sc1 = s.u32(SC, ib, 4);
        sb.qh0 = s.u32(QH, ib, 0);
        sb.qh1 = s.u32(QH, ib, 4);
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        r.q                         = s.template rows<1>(QS, kb / 8, 4 * j);
        r.signs                     = s.template rows<1>(QS, kb / 8, QK_K / 8 + 4 * j);
        r.qh                        = ((j < 4 ? sb.qh0 : sb.qh1) >> (8 * (j % 4))) & 0xffu;
        const simd<uint32_t, MR> sc = ((j < 4 ? sb.sc0 : sb.sc1) >> (8 * (j % 4))) & 0xffu;
        r.d0                        = sb.d * (0.5f + simd<float, MR>(sc & 0xfu)) * 0.25f;
        r.d1                        = sb.d * (0.5f + simd<float, MR>(sc >> 4)) * 0.25f;
    }
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        constexpr int          RC  = GROUPED_GEMM_RC;
        simd<uint32_t, RC>     qr  = r.q.template select<RC, 1>(g * RC);
        simd<uint32_t, RC * 4> qb  = qr.template bit_cast_view<uint8_t>();
        simd<uint32_t, RC>     hr  = r.qh.template select<RC, 1>(g * RC);
        simd<uint32_t, RC * 4> h4  = hr.template replicate_vs_w_hs<RC, 1, 4, 0>(0);
        simd<uint32_t, RC * 4> sh  = simd<uint32_t, 4>(simd<int32_t, 4>(8, -2)).template replicate<RC>();
        simd<uint32_t, RC * 4> idx = qb | ((h4 << sh) & 0x300u);
        simd<uint32_t, RC * 8> ge  = slm_gather<uint32_t, RC * 8>(grid64_offsets<4>(idx));
        return grid_signed(ge, r.signs.template select<RC, 1>(g * RC), r.d0.template select<RC, 1>(g * RC),
                           r.d1.template select<RC, 1>(g * RC));
    }
};

// [d][qs 32][qh 16] per 256 values: per sub-block j, 4 grid bytes at 4 j and qh[j] holding 4 x 3 index bits, a
// 3-bit scale d * (2 s + 1) and the delta's sign; value d * (q + delta), q a nibble of iq1s_grid_gpu
template <int MR> struct grouped_gemm_esimd_a<block_iq1_s, MR> : grouped_gemm_esimd::format<block_iq1_s> {
    static constexpr int qk        = QK_K;
    static constexpr int slm_bytes = NGRID_IQ1S * sizeof(uint32_t);
    static ESIMD_INLINE void init_slm() {
        using namespace grouped_gemm_esimd;
        for (int i = 0; i < NGRID_IQ1S; i += 64) {
            slm_block_store<uint32_t, 64>(i * 4, block_load<uint32_t, 64>(iq1s_grid_gpu + i));
        }
    }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR> q;
        grouped_gemm_esimd::simd<uint32_t, MR> qh;
        grouped_gemm_esimd::simd<float, MR>    d;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<float, MR> d;
    };
    static constexpr grouped_gemm_esimd::field QS = { 0, QK_K / 8, 2 }, QH = { QK_K / 8, QK_K / 16, 2 + QK_K / 8 },
                                               D  = { QK_K / 8 + QK_K / 16, 2, 0 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / QK_K * (QK_K / 8)));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
        simd<uint16_t, MR> dh = s.u16(D, ib, 0);
        sb.d = simd<sycl::half, MR>(dh.template bit_cast_view<sycl::half>());
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        r.q  = s.template rows<1>(QS, kb / 8, 4 * j);
        r.qh = s.u16(QH, kb / 8, 2 * j);
        r.d  = sb.d * simd<float, MR>(2 * ((r.qh >> 12) & 7u) + 1);
    }
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        constexpr int          RC  = GROUPED_GEMM_RC;
        simd<uint32_t, RC>     q   = r.q.template select<RC, 1>(g * RC);
        simd<uint32_t, RC>     qh  = r.qh.template select<RC, 1>(g * RC);
        simd<uint32_t, RC * 4> sh  = simd<uint32_t, 4>(0, 3).template replicate<RC>();
        simd<uint32_t, RC * 4> idx = simd<uint32_t, RC * 4>(q.template bit_cast_view<uint8_t>()) |
                                     (((qh.template replicate_vs_w_hs<RC, 1, 4, 0>(0) >> sh) & 7u) << 8);
        simd<uint32_t, RC * 4> g32 = slm_gather<uint32_t, RC * 4>(idx * 4);
        simd<float, RC>        dl  = -1.0f + IQ1S_DELTA;
        dl.merge(-1.0f - IQ1S_DELTA, (qh & 0x8000u) != 0);
        simd<float, RC> d = r.d.template select<RC, 1>(g * RC);
        return codes_dm(iq1_codes(g32, dl.template replicate_vs_w_hs<RC, 1, 4, 0>(0)), d, 0.0f, d, 0.0f);
    }
};

// [qs 32][qh 16][scales 8] per 256 values, no d field: the superblock's f16 scale is the top nibbles of the 4
// scale words; 16 values share a 3-bit scale (2 s + 1), 8 values a grid entry (qs byte, 3 bits of their qh
// nibble) and a delta (the nibble's bit 3); value d * (q + delta), q a nibble of iq1s_grid_gpu
template <int MR> struct grouped_gemm_esimd_a<block_iq1_m, MR> : grouped_gemm_esimd::format<block_iq1_m> {
    static constexpr int qk        = QK_K;
    static constexpr int slm_bytes = NGRID_IQ1S * sizeof(uint32_t);
    static ESIMD_INLINE void init_slm() { grouped_gemm_esimd_a<block_iq1_s, MR>::init_slm(); }
    struct raw {
        grouped_gemm_esimd::simd<uint32_t, MR> q;
        grouped_gemm_esimd::simd<uint32_t, MR> qh;
        grouped_gemm_esimd::simd<float, MR>    d0;
        grouped_gemm_esimd::simd<float, MR>    d1;
    };
    struct sb_t {
        grouped_gemm_esimd::simd<float, MR>    d;
        grouped_gemm_esimd::simd<uint32_t, MR> sc0;
        grouped_gemm_esimd::simd<uint32_t, MR> sc1;
    };
    static constexpr grouped_gemm_esimd::field QS = { 0, QK_K / 8, 0 }, QH = { QK_K / 8, QK_K / 16, QK_K / 8 },
                                               SC = { QK_K / 8 + QK_K / 16, QK_K / 32, QK_K / 8 + QK_K / 16 };
    static bool ok(const int K, const bool aos) {
        return K % qk == 0 && (aos || grouped_gemm_esimd::surface_ok(K / QK_K * (QK_K / 8)));
    }
    template <typename Sl> static ESIMD_INLINE void load_sb(sb_t & sb, const Sl & s, const int ib) {
        using namespace grouped_gemm_esimd;
        sb.sc0 = s.u32(SC, ib, 0);
        sb.sc1 = s.u32(SC, ib, 4);
        simd<uint32_t, MR> u = ((sb.sc0 & 0xffffu) >> 12) | ((sb.sc0 >> 24) & 0xf0u) | (((sb.sc1 & 0xffffu) >> 4) & 0xf00u) |
                               ((sb.sc1 >> 16) & 0xf000u);
        simd<uint16_t, MR> uh = u;
        sb.d                  = simd<sycl::half, MR>(uh.template bit_cast_view<sycl::half>());
    }
    template <typename Sl> static ESIMD_INLINE void load(raw & r, sb_t & sb, const Sl & s, const int kb, const int j) {
        using namespace grouped_gemm_esimd;
        r.q  = s.template rows<1>(QS, kb / 8, 4 * j);
        r.qh = s.u16(QH, kb / 8, 2 * j);
        // the scale word of 16-value groups 2 j and 2 j + 1, 3 bits each at 3 * (group % 4)
        const simd<uint32_t, MR> w = ((j < 4 ? sb.sc0 : sb.sc1) >> (16 * ((j / 2) % 2))) & 0xffffu;
        r.d0 = sb.d * simd<float, MR>(2 * ((w >> (3 * ((2 * j) % 4))) & 7u) + 1);
        r.d1 = sb.d * simd<float, MR>(2 * ((w >> (3 * ((2 * j + 1) % 4))) & 7u) + 1);
    }
    template <int g> static ESIMD_INLINE grouped_gemm_esimd::a_halves decode(raw & r) {
        using namespace grouped_gemm_esimd;
        constexpr int          RC  = GROUPED_GEMM_RC;
        simd<uint32_t, RC>     q   = r.q.template select<RC, 1>(g * RC);
        simd<uint32_t, RC>     qh  = r.qh.template select<RC, 1>(g * RC);
        // entry il takes nibble il of the two qh bytes
        simd<uint32_t, RC * 4> nib = (qh.template replicate_vs_w_hs<RC, 1, 4, 0>(0) >>
                                      simd<uint32_t, 4>(0, 4).template replicate<RC>()) & 0xfu;
        simd<uint32_t, RC * 4> idx = simd<uint32_t, RC * 4>(q.template bit_cast_view<uint8_t>()) | ((nib & 7u) << 8);
        simd<uint32_t, RC * 4> g32 = slm_gather<uint32_t, RC * 4>(idx * 4);
        simd<float, RC * 4>    dl  = -1.0f + IQ1M_DELTA;
        dl.merge(-1.0f - IQ1M_DELTA, (nib & 8u) != 0);
        return codes_dm(iq1_codes(g32, dl), r.d0.template select<RC, 1>(g * RC), 0.0f, r.d1.template select<RC, 1>(g * RC),
                        0.0f);
    }
};

// T is the dpas operand type, sycl::half or bfloat16: the type B is packed in. A is decoded in half
// either way and converted for bf16 (see the double rounding note at the conversion). AOS: the weight
// is in the canonical block layout rather than the reorder (SoA) one. A: the grouped_gemm_arch built for.
template <typename block_q_t, int KS, typename T, bool AOS, typename A> struct grouped_gemm_esimd_kernel {
    using F = grouped_gemm_esimd_a<block_q_t, A::MR>;

    const char *              src0_dd;
    size_t                    expert_stride;
    const ggml_sycl_grouped_gemm_tile * tiles;
    const T *                 packed;
    float *                   dst;
    int                       M;
    int                       Npad;
    int                       K;
    ggml_sycl_grouped_gemm_rows         dst_rows;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        if constexpr (grouped_gemm_built<A>()) {
            run(it);
        }
    }

    ESIMD_INLINE void run(sycl::nd_item<1> it) const {
        using namespace sycl::ext::intel::esimd;
        namespace xmx = ggml_sycl_xmx::esimd;
        constexpr int MR = A::MR;
        constexpr int NW = A::NW;
        constexpr int RC = GROUPED_GEMM_RC;
        constexpr int RG = MR / RC; // dpas row groups of this thread
        constexpr int BN = GGML_SYCL_GROUPED_GEMM_BN;
        constexpr int NT = BN / NW;

        if constexpr (F::slm_bytes > 0) {
            // the format's lookup table; a work-group is one thread, so nothing else writes it
            slm_init<F::slm_bytes>();
            F::init_slm();
        }

        const int mblocks = M / MR;
        const int gid     = it.get_global_id(0);
        const int t       = gid / mblocks;
        const int m0      = (gid % mblocks) * MR;

        const ggml_sycl_grouped_gemm_tile tile = tiles[t];
        if (tile.n1 <= tile.n0) {
            return;
        }
        const int b0 = t * BN; // the tile's packed-B columns

        // the slice of this expert
        const int             bpr = K / F::qk;
        const grouped_gemm_esimd::slice<MR, typename A::HW> sl  = {
            (const uint8_t *) src0_dd + (size_t) tile.expert * expert_stride,
            M,
            bpr,
            M * bpr,
            m0,
            AOS,
            F::bytes
        };
        const uint32_t *      bp  = (const uint32_t *) packed;
        const unsigned        bw  = (unsigned) Npad * 4; // bytes of one packed B row
        const int             nstep = K / GROUPED_GEMM_QK;

        simd<float, RC * NW> acc[RG][NT];
        simd<float, RC * NW> total[KS > 1 ? RG : 1][KS > 1 ? NT : 1];
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

        // One k step at a time: the format's stored bits for the MR rows, and the 16 k pairs x NW
        // columns of packed B per dpas column tile, one 2D block message each (without 2D loads, one
        // block load per packed row, cut into the tiles in registers). A second stage in
        // flight spilled 7 KB of the 256 registers and ran 1.6x slower; the other threads of the
        // XVE hide the load latency instead.
        // A stored block's metadata (its sub-block scales) is loaded once per block. With KS > 1 the
        // K split of the joint_matrix kernel is reproduced for bit-identical output (a second set of
        // accumulators); KS == 1 sums the whole K in one, which is a different f32 rounding order and
        // frees the 4 KB of registers the second set holds.
        constexpr int    KPB       = F::qk / GROUPED_GEMM_QK;  // k steps per stored block
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
            simd<uint32_t, 16 * NW> b[NT]; // [16 k pairs][NW columns] each
            if constexpr (A::L2D) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    b[nt] = xmx::load_2d<typename A::HW, uint32_t, NW, 16>(bp, bw - 1, K / 2 - 1, bw - 1, b0 + nt * NW,
                                                                           kb * 16);
                }
            } else {
                simd<uint32_t, 16 * BN> bm;
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    bm.template select<BN, 1>(r * BN) =
                        block_load<uint32_t, BN>(bp + (size_t) (kb * 16 + r) * Npad + b0, overaligned_tag<16>{});
                }
                auto bv = bm.template bit_cast_view<uint32_t, 16, BN>();
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    b[nt] = bv.template select<16, 1, NW, 1>(0, nt * NW);
                }
            }
            simd<sycl::half, MR * GROUPED_GEMM_QK> all;
            simd<T, MR * GROUPED_GEMM_QK>          plain_a; // a plain format's stored values as the operand
            if constexpr (F::plain) {
                if constexpr (std::is_same_v<typename F::elem, T>) {
                    plain_a = r.q.template bit_cast_view<T>();
                } else {
                    plain_a = simd<float, MR * GROUPED_GEMM_QK>(r.q.template bit_cast_view<typename F::elem>());
                }
            } else if constexpr (F::whole) {
                all = F::decode_all(r);
            }
            // one row group of the thread, g a compile-time constant: with a runtime g the larger
            // decodes were left rolled
            auto row_group = [&](auto gc) SYCL_ESIMD_FUNCTION {
                constexpr int g = decltype(gc)::value;
                grouped_gemm_esimd::a_halves av;
                if constexpr (F::plain) {
                } else if constexpr (F::whole) {
                    // [row][32 k] of the whole tile to this row group's [k half][row][16 k]
                    auto am = all.template bit_cast_view<sycl::half, MR, GROUPED_GEMM_QK>();
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
                    if constexpr (F::plain) {
                        at = plain_a.template bit_cast_view<T, MR, GROUPED_GEMM_QK>().template select<RC, 1, 16, 1>(
                            g * RC, 16 * h);
                    } else if constexpr (std::is_same_v<T, sycl::half>) {
                        at = ah;
                    } else {
                        // rounded twice (f32 to half in the decode, half to bf16 here), so A's rounding
                        // error is about twice the joint_matrix bf16 stage's, which rounds f32 once
                        at = simd<float, RC * 16>(ah);
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        simd<uint32_t, 8 * NW> bh32 = b[nt].template select<8 * NW, 1>(h * 8 * NW);
                        simd<T, 16 * NW>       bh   = bh32.template bit_cast_view<T>();
                        acc[g][nt] = xmx::dpas<8, RC, float, float>(acc[g][nt], bh, at);
                    }
                }
            };
            grouped_gemm_esimd::static_for<0, RG>::run(row_group);
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
                    col.template select<RC, 1>(g * RC) = total[g][c / NW].template select<RC, NW>(c % NW);
                } else {
                    col.template select<RC, 1>(g * RC) = acc[g][c / NW].template select<RC, NW>(c % NW);
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

// the weights' surfaces: 2D block loads (SoA with L2D) want 64-byte aligned bases, the dword gathers 4
static bool grouped_gemm_esimd_weights_ok(const char * w, const size_t expert_stride, const bool l2d) {
    const size_t a = l2d ? 64 : 4;
    return (uintptr_t) w % a == 0 && expert_stride % a == 0;
}

// 0 off, 1 on (default), 2 on with the joint_matrix kernel's K-split order reproduced (bit-identical output)
static int ggml_sycl_grouped_gemm_esimd_mode() {
    return g_ggml_sycl_mmid_esimd;
}

bool ggml_sycl_grouped_gemm_esimd_enabled() {
#if defined(GGML_SYCL_SUPPORT_XMX)
    return g_ggml_sycl_enable_esimd && ggml_sycl_grouped_gemm_esimd_mode() != 0;
#else
    return false;
#endif
}

template <typename block_q_t, typename T, bool AOS, typename A>
static void grouped_gemm_esimd_launch(const char *                        src0_dd,
                                      size_t                              expert_stride,
                                      const ggml_sycl_grouped_gemm_tile * tiles,
                                      const T *                           packed,
                                      float *                             dst,
                                      int                                 M,
                                      int                                 Npad,
                                      int                                 K,
                                      const ggml_sycl_grouped_gemm_rows & dst_rows,
                                      int64_t                             n_tiles,
                                      dpct::queue_ptr                     stream) {
    const sycl::nd_range<1> range((size_t) n_tiles * (M / A::MR), 1);
    if (ggml_sycl_grouped_gemm_esimd_mode() == 2) {
        stream->parallel_for(range, grouped_gemm_esimd_kernel<block_q_t, GGML_SYCL_GROUPED_GEMM_KSPLIT, T, AOS, A>{
                                        src0_dd, expert_stride, tiles, packed, dst, M, Npad, K, dst_rows });
    } else {
        stream->parallel_for(range, grouped_gemm_esimd_kernel<block_q_t, 1, T, AOS, A>{
                                        src0_dd, expert_stride, tiles, packed, dst, M, Npad, K, dst_rows });
    }
}

template <int NW> using grouped_gemm_arch_of = std::conditional_t<NW == 16, grouped_gemm_xe2, grouped_gemm_xehpg>;

// the shapes the kernel takes: whole row groups and stored blocks, every K-split range non-empty
// (also in mode 1, so both modes take the same shapes)
template <typename block_q_t, typename A> static bool grouped_gemm_esimd_shape_ok(int M, int K, bool reordered) {
    using F = grouped_gemm_esimd_a<block_q_t, A::MR>;
    return M % A::MR == 0 && F::ok(K, !reordered && !F::plain) && K >= GROUPED_GEMM_QK * GGML_SYCL_GROUPED_GEMM_KSPLIT;
}

template <typename block_q_t, typename T, typename A>
static bool grouped_gemm_esimd_run(const char *                        src0_dd,
                                   size_t                              expert_stride,
                                   const ggml_sycl_grouped_gemm_tile * tiles,
                                   const T *                           packed,
                                   float *                             dst,
                                   int                                 M,
                                   int                                 Npad,
                                   int                                 K,
                                   const ggml_sycl_grouped_gemm_rows & dst_rows,
                                   bool                                reordered,
                                   int64_t                             n_tiles,
                                   dpct::queue_ptr                     stream) {
    // B is read with 2D block loads, which want 64-byte aligned surfaces and pitches, or with 16-byte
    // aligned block loads of whole tiles
    // a plain format's bytes are the same in both layouts: it is read as the reorder one, a 2D surface
    using F = grouped_gemm_esimd_a<block_q_t, A::MR>;
    reordered = reordered || F::plain;
    if (!grouped_gemm_esimd_shape_ok<block_q_t, A>(M, K, reordered) ||
        !grouped_gemm_esimd_weights_ok(src0_dd, expert_stride, reordered && A::L2D) || ((uintptr_t) packed % 64) != 0 ||
        Npad % 16 != 0) {
        return false;
    }
    if (F::plain || reordered) {
        grouped_gemm_esimd_launch<block_q_t, T, false, A>(src0_dd, expert_stride, tiles, packed, dst, M, Npad, K,
                                                          dst_rows, n_tiles, stream);
    } else if constexpr (!F::plain) {
        grouped_gemm_esimd_launch<block_q_t, T, true, A>(src0_dd, expert_stride, tiles, packed, dst, M, Npad, K,
                                                         dst_rows, n_tiles, stream);
    }
    return true;
}

template <typename block_q_t, typename T, int NW>
bool ggml_sycl_grouped_gemm_esimd(const char *                        src0_dd,
                                  size_t                              expert_stride,
                                  const ggml_sycl_grouped_gemm_tile * tiles,
                                  const T *                           packed,
                                  float *                             dst,
                                  int                                 M,
                                  int                                 Npad,
                                  int                                 K,
                                  const ggml_sycl_grouped_gemm_rows & dst_rows,
                                  bool                                reordered,
                                  int64_t                             n_tiles,
                                  dpct::queue_ptr                     stream) {
    static_assert(grouped_gemm_esimd_a<block_q_t>::supported == ggml_sycl_grouped_gemm_esimd_has<block_q_t>,
                  "the header's format list and the formats implemented here must agree");
    static_assert(NW == 8 || NW == 16, "the dpas widths of Xe-HPG and of Xe2 / Xe-HPC");
    return grouped_gemm_esimd_run<block_q_t, T, grouped_gemm_arch_of<NW>>(
        src0_dd, expert_stride, tiles, packed, dst, M, Npad, K, dst_rows, reordered, n_tiles, stream);
}

#define GROUPED_GEMM_ESIMD_INSTANCE_T(block_q_t, T, NW)                                               \
    template bool ggml_sycl_grouped_gemm_esimd<block_q_t, T, NW>(                                     \
        const char *, size_t, const ggml_sycl_grouped_gemm_tile *, const T *, float *, int, int, int, \
        const ggml_sycl_grouped_gemm_rows &, bool, int64_t, dpct::queue_ptr);
#define GROUPED_GEMM_ESIMD_INSTANCE(block_q_t)                                       \
    GROUPED_GEMM_ESIMD_INSTANCE_T(block_q_t, sycl::half, 16)                                    \
    GROUPED_GEMM_ESIMD_INSTANCE_T(block_q_t, sycl::ext::oneapi::bfloat16, 16)                   \
    GROUPED_GEMM_ESIMD_INSTANCE_T(block_q_t, sycl::half, 8)                                     \
    GROUPED_GEMM_ESIMD_INSTANCE_T(block_q_t, sycl::ext::oneapi::bfloat16, 8)
GROUPED_GEMM_ESIMD_INSTANCE(block_q8_0)
GROUPED_GEMM_ESIMD_INSTANCE(block_iq4_nl)
GROUPED_GEMM_ESIMD_INSTANCE(block_iq4_xs)
GROUPED_GEMM_ESIMD_INSTANCE(block_iq3_s)
GROUPED_GEMM_ESIMD_INSTANCE(block_iq3_xxs)
GROUPED_GEMM_ESIMD_INSTANCE(block_q3_K)
GROUPED_GEMM_ESIMD_INSTANCE(block_q2_K)
GROUPED_GEMM_ESIMD_INSTANCE(block_q6_K)
GROUPED_GEMM_ESIMD_INSTANCE(block_q5_K)
GROUPED_GEMM_ESIMD_INSTANCE(block_q4_K)
GROUPED_GEMM_ESIMD_INSTANCE(block_q4_0)
GROUPED_GEMM_ESIMD_INSTANCE(block_q4_1)
GROUPED_GEMM_ESIMD_INSTANCE(block_q5_0)
GROUPED_GEMM_ESIMD_INSTANCE(block_q5_1)
GROUPED_GEMM_ESIMD_INSTANCE(block_q1_0)
GROUPED_GEMM_ESIMD_INSTANCE(block_q2_0)
GROUPED_GEMM_ESIMD_INSTANCE(block_mxfp4)
GROUPED_GEMM_ESIMD_INSTANCE(block_nvfp4)
GROUPED_GEMM_ESIMD_INSTANCE(block_iq2_xxs)
GROUPED_GEMM_ESIMD_INSTANCE(block_iq2_xs)
GROUPED_GEMM_ESIMD_INSTANCE(block_iq2_s)
GROUPED_GEMM_ESIMD_INSTANCE(block_iq1_s)
GROUPED_GEMM_ESIMD_INSTANCE(block_iq1_m)
GROUPED_GEMM_ESIMD_INSTANCE(sycl::half)
GROUPED_GEMM_ESIMD_INSTANCE(sycl::ext::oneapi::bfloat16)
