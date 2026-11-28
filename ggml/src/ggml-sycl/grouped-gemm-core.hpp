#ifndef GGML_SYCL_GROUPED_GEMM_CORE_HPP
#define GGML_SYCL_GROUPED_GEMM_CORE_HPP

// The k loop shared by the ESIMD dequant GEMMs (grouped MoE, dense): a thread owns MR weight rows x NT * NW
// columns of packed B, dequantizes the rows' weights of each k step in registers and multiplies them with
// dpas. The kernels differ in how they map threads to tiles and how they store the result.

#include "fused-gemm.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include "xmx-compat.hpp"

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
    int off = 0; // the field's offset in a block of its stream, when it shares one (iq3_s d and scales)
};

// field F of format T's layout table (ggml_sycl_layout in fused-gemm.hpp)
template <typename T, int F> constexpr field field_of() {
    using L = ggml_sycl_layout<T>;
    return { ggml_sycl_layout_pre<L>(F), ggml_sycl_layout_stride<L>(F), L::seg[F].aos, ggml_sycl_layout_off<L>(F) };
}

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
        return o + (uint32_t) ((aos ? f.aos : nblocks * f.pre + f.off) + (m0 * bpr + ib) * stride + inner);
    }

    // a field's bytes at offset `inner` start on a dword in every block (the AoS block and offset
    // are multiples of 4); the SoA streams always do
    ESIMD_INLINE bool dword_aligned(const field & f, const int inner) const {
        return aos ? bs % 4 == 0 && (f.aos + inner) % 4 == 0 : f.size % 4 == 0 && (f.off + inner) % 4 == 0;
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
                                                         M - 1, pitch - 1, (ib * f.size + f.off + inner) / 4, m0);
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

} // namespace grouped_gemm_esimd

// Accumulators and k loop of one thread. F: the weight format (grouped_gemm_esimd_a), T: the dpas operand
// type (half, or bf16 rounded from the f16 decode), A: the grouped_gemm_arch, NT: dpas column tiles of NW
// columns. KS > 1 reproduces the joint_matrix kernel's K split, KS ranges summed in ascending order, at the
// cost of a second set of accumulators. A_FIRST: decode all row groups first, then stream the B tiles with
// one live (for wide NT); else load all B tiles first and decode one row group at a time.
template <typename F, typename T, typename A, int NT, int KS, bool A_FIRST> struct grouped_gemm_core {
    static constexpr int MR  = A::MR;
    static constexpr int NW  = A::NW;
    static constexpr int RC  = GROUPED_GEMM_RC;
    static constexpr int RG  = MR / RC;  // dpas row groups of the thread
    static constexpr int QK  = GROUPED_GEMM_QK;
    static constexpr int KPB = F::qk / QK;  // k steps a stored block

    static_assert(!A_FIRST || A::L2D, "A_FIRST reads B with 2D block loads");
    static_assert(!A_FIRST || KS == 1, "A_FIRST sums K in one range");

    template <typename V, int N> using simd = sycl::ext::intel::esimd::simd<V, N>;

    simd<float, RC * NW> acc[RG][NT];
    simd<float, RC * NW> total[KS > 1 ? RG : 1][KS > 1 ? NT : 1];
    int                  split     = 0;
    int                  split_end = 0;
    int                  nstep     = 0;
    typename F::sb_t     sb;  // the metadata of the current stored block

    // nstep: the k steps the thread runs
    ESIMD_INLINE void init(const int nstep_) {
        nstep = nstep_;
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
        split     = 0;
        split_end = KS > 1 ? nstep / KS : nstep;
    }

    // B tile nt of k step kb: [16 k pairs][NW columns]. bw: bytes of one packed B row
    static ESIMD_INLINE simd<uint32_t, 16 * NW> load_b(const uint32_t * bp, const unsigned bw, const int K,
                                                       const int b0, const int kb, const int nt) {
        return ggml_sycl_xmx::esimd::load_2d<typename A::HW, uint32_t, NW, 16>(bp, bw - 1, K / 2 - 1, bw - 1,
                                                                             b0 + nt * NW, kb * 16);
    }

    // The A operand of row group g, k half h (rows x k [16h, 16h+16), row major), from the decoded or
    // plain values of the k step
    template <int g>
    static ESIMD_INLINE simd<T, RC * 16> operand(const int h, grouped_gemm_esimd::a_halves & av, simd<T, MR * QK> & plain_a) {
        simd<T, RC * 16> at;
        if constexpr (F::plain) {
            at = plain_a.template bit_cast_view<T, MR, QK>().template select<RC, 1, 16, 1>(g * RC, 16 * h);
        } else if constexpr (std::is_same_v<T, sycl::half>) {
            at = av.template select<RC * 16, 1>(h * RC * 16);
        } else {
            // rounded twice (f32 to half in the decode, half to bf16 here), so A's rounding error is
            // about twice the joint_matrix bf16 stage's, which rounds f32 once
            simd<sycl::half, RC * 16> ah = av.template select<RC * 16, 1>(h * RC * 16);
            at                           = simd<float, RC * 16>(ah);
        }
        return at;
    }

    // The decoded A of row group g, [k half][row][16 k]; all holds a whole-tile decode (F::whole)
    template <int g>
    static ESIMD_INLINE grouped_gemm_esimd::a_halves decoded(typename F::raw & r,
                                                            simd<sycl::half, MR * QK> & all) {
        grouped_gemm_esimd::a_halves av;
        if constexpr (F::plain) {
        } else if constexpr (F::whole) {
            // [row][32 k] of the whole tile to this row group's [k half][row][16 k]
            auto am = all.template bit_cast_view<sycl::half, MR, QK>();
            av.template select<RC * 16, 1>(0)       = am.template select<RC, 1, 16, 1>(g * RC, 0);
            av.template select<RC * 16, 1>(RC * 16) = am.template select<RC, 1, 16, 1>(g * RC, 16);
        } else {
            av = F::template decode<g>(r);
        }
        return av;
    }

    // k step kb against the packed-B columns from b0 (Npad columns a packed row)
    template <typename Sl>
    ESIMD_INLINE void step(const Sl & sl, const uint32_t * bp, const int Npad, const int K, const int b0,
                           const int kb) {
        using namespace sycl::ext::intel::esimd;
        namespace xmx     = ggml_sycl_xmx::esimd;
        const unsigned bw = (unsigned) Npad * 4;
        const int      j  = kb % KPB;
        if (j == 0) {
            F::load_sb(sb, sl, kb / KPB);
        }
        typename F::raw r;
        F::load(r, sb, sl, kb, j);

        if constexpr (A_FIRST) {
            // the first B tile loads while the weights decode
            simd<uint32_t, 16 * NW>      b = load_b(bp, bw, K, b0, kb, 0);
            simd<sycl::half, MR * QK>    all;
            simd<T, MR * QK>             plain_a;
            if constexpr (F::plain) {
                plain_a = plain_values(r);
            } else if constexpr (F::whole) {
                all = F::decode_all(r);
            }
            grouped_gemm_esimd::a_halves av[RG];
            auto                         dec = [&](auto gc) SYCL_ESIMD_FUNCTION {
                constexpr int g = decltype(gc)::value;
                av[g]           = decoded<g>(r, all);
            };
            grouped_gemm_esimd::static_for<0, RG>::run(dec);
            auto col_tile = [&](auto ntc) SYCL_ESIMD_FUNCTION {
                constexpr int           nt = decltype(ntc)::value;
                simd<uint32_t, 16 * NW> bn;
                if constexpr (nt + 1 < NT) {
                    bn = load_b(bp, bw, K, b0, kb, nt + 1);
                }
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    simd<uint32_t, 8 * NW> bh32 = b.template select<8 * NW, 1>(h * 8 * NW);
                    simd<T, 16 * NW>       bh   = bh32.template bit_cast_view<T>();
                    auto                   rg   = [&](auto gc) SYCL_ESIMD_FUNCTION {
                        constexpr int g = decltype(gc)::value;
                        acc[g][nt] = xmx::dpas<8, RC, float, float>(acc[g][nt], bh, operand<g>(h, av[g], plain_a));
                    };
                    grouped_gemm_esimd::static_for<0, RG>::run(rg);
                }
                if constexpr (nt + 1 < NT) {
                    b = bn;
                }
            };
            grouped_gemm_esimd::static_for<0, NT>::run(col_tile);
        } else {
            simd<uint32_t, 16 * NW> b[NT];  // [16 k pairs][NW columns] each
            if constexpr (A::L2D) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    b[nt] = load_b(bp, bw, K, b0, kb, nt);
                }
            } else {
                // no 2D block loads: one block load per packed row, cut into the tiles in registers
                constexpr int           BN = NT * NW;
                simd<uint32_t, 16 * BN> bm;
#pragma unroll
                for (int r16 = 0; r16 < 16; ++r16) {
                    bm.template select<BN, 1>(r16 * BN) =
                        block_load<uint32_t, BN>(bp + (size_t) (kb * 16 + r16) * Npad + b0, overaligned_tag<16>{});
                }
                auto bv = bm.template bit_cast_view<uint32_t, 16, BN>();
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    b[nt] = bv.template select<16, 1, NW, 1>(0, nt * NW);
                }
            }
            simd<sycl::half, MR * QK> all;
            simd<T, MR * QK>          plain_a;  // a plain format's stored values as the operand
            if constexpr (F::plain) {
                plain_a = plain_values(r);
            } else if constexpr (F::whole) {
                all = F::decode_all(r);
            }
            // one row group of the thread, g a compile-time constant: with a runtime g the larger
            // decodes were left rolled
            auto row_group = [&](auto gc) SYCL_ESIMD_FUNCTION {
                constexpr int                g  = decltype(gc)::value;
                grouped_gemm_esimd::a_halves av = decoded<g>(r, all);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    simd<T, RC * 16> at = operand<g>(h, av, plain_a);
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        simd<uint32_t, 8 * NW> bh32 = b[nt].template select<8 * NW, 1>(h * 8 * NW);
                        simd<T, 16 * NW>       bh   = bh32.template bit_cast_view<T>();
                        acc[g][nt]                  = xmx::dpas<8, RC, float, float>(acc[g][nt], bh, at);
                    }
                }
            };
            grouped_gemm_esimd::static_for<0, RG>::run(row_group);
        }

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

    // column c of the tile, the MR rows in order. Call it with c a compile-time constant (an unrolled
    // loop): a runtime index sends the accumulators to private memory
    ESIMD_INLINE simd<float, MR> column(const int c) {
        simd<float, MR> col;
#pragma unroll
        for (int g = 0; g < RG; ++g) {
            if constexpr (KS > 1) {
                col.template select<RC, 1>(g * RC) = total[g][c / NW].template select<RC, NW>(c % NW);
            } else {
                col.template select<RC, 1>(g * RC) = acc[g][c / NW].template select<RC, NW>(c % NW);
            }
        }
        return col;
    }

  private:
    static ESIMD_INLINE simd<T, MR * QK> plain_values(typename F::raw & r) {
        if constexpr (std::is_same_v<typename F::elem, T>) {
            return r.q.template bit_cast_view<T>();
        } else {
            return simd<float, MR * QK>(r.q.template bit_cast_view<typename F::elem>());
        }
    }
};

#endif // GGML_SYCL_GROUPED_GEMM_CORE_HPP
