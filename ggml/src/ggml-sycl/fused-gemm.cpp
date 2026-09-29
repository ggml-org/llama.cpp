#include "fused-gemm.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace mx = sycl::ext::oneapi::experimental::matrix;

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
static constexpr int FG_BK     = QK4_NL;
static constexpr int FG_KSPLIT = 4;

// Element traits of A and B. The A stage and the B pack compute in f32 and convert once, when they
// write the element, so a wider type costs no extra pass.
template <typename TA> struct fg_elem;

template <> struct fg_elem<sycl::half> {
    using store = sycl::half;
    using pair  = sycl::half2;
    static store cvt(float x) { return (store) x; }
    static pair make(float x, float y) { return pair((store) x, (store) y); }
};

// tf32 is kept in f32 storage; round to nearest even so the result is not worse than f16. Plain bit
// ops, not round_to_tf32: that needs a SPIR-V extension the DG2 AOT target rejects.
static inline float fg_round_tf32(float x) {
    const uint32_t u = sycl::bit_cast<uint32_t>(x);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        return x; // inf or nan
    }
    return sycl::bit_cast<float>((u + 0xfffu + ((u >> 13) & 1)) & ~0x1fffu);
}

template <> struct fg_elem<mx::precision::tf32> {
    using store = float;
    using pair  = sycl::float2;
    static store cvt(float x) { return fg_round_tf32(x); }
    static pair make(float x, float y) { return pair(fg_round_tf32(x), fg_round_tf32(y)); }
};

// One joint_matrix combination (A/B element type, TM x TN x TK, sub-group size) and the tiling
// built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32) columns of B.
template <typename TA, int TM_, int TN_, int TK_, int SG_> struct fg_shape {
    using ta = TA;
    using E  = fg_elem<TA>;
    using ts = typename E::store;
    static constexpr int TM = TM_;
    static constexpr int TN = TN_;
    static constexpr int TK = TK_;
    static constexpr int SG = SG_;
    static constexpr int VNNI    = 4 / sizeof(ts);   // K rows of B packed in one 32-bit word
    static constexpr int SG_ROWS = TM > 16 ? TM : 16;
    static constexpr int RPL     = SG_ROWS / SG;     // A rows one lane decodes per k step
    static constexpr int MT      = SG_ROWS / TM;
    static constexpr int BN      = TN > 32 ? TN : 32;
    static constexpr int NT      = BN / TN;
    static constexpr int WG_SIZE = FG_KSPLIT * SG;
    static constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::row_major : mx::layout::ext_intel_packed;
    static constexpr bool is_tf32 = std::is_same_v<TA, mx::precision::tf32>;
    static_assert(SG_ROWS % SG == 0 && SG_ROWS % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per shape. f16 shapes come from the appendix of
// sycl_ext_oneapi_matrix; tf32 is the promoted fallback. fp16 data never needs it on known hardware.
using fg_f16_8x16x16  = fg_shape<sycl::half, 8, 16, 16, 16>;   // Xe2, Xe3, Xe-HPC
using fg_f16_16x16x16 = fg_shape<sycl::half, 16, 16, 16, 16>;  // Xe2, Xe3, Xe-HPC
using fg_f16_32x64x16 = fg_shape<sycl::half, 32, 64, 16, 16>;  // Xe2, Xe3, Xe-HPC
using fg_f16_32x64x32 = fg_shape<sycl::half, 32, 64, 32, 16>;  // Xe2, Xe3, Xe-HPC
using fg_f16_8x8x16   = fg_shape<sycl::half, 8, 8, 16, 8>;     // Xe-HPG (DG2, Arc A), ARL-H
using fg_tf32_8x16x8  = fg_shape<mx::precision::tf32, 8, 16, 8, 16>; // Xe2, Xe3, Xe-HPC

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the shapes of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
template <int SG> static constexpr bool fg_listed() {
#if defined(GGML_SYCL_XMX_AOT_SG)
    return SG == GGML_SYCL_XMX_AOT_SG;
#else
    return true;
#endif
}

template <typename S, typename F> static void fg_call_shape(F && f) {
    if constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

template <typename F> static void fg_visit_shape(int idx, F && f) {
    switch (idx) {
        case 0: fg_call_shape<fg_f16_8x16x16>(f);  break;
        case 1: fg_call_shape<fg_f16_16x16x16>(f); break;
        case 2: fg_call_shape<fg_f16_32x64x16>(f); break;
        case 3: fg_call_shape<fg_f16_32x64x32>(f); break;
        case 4: fg_call_shape<fg_f16_8x8x16>(f);   break;
        case 5: fg_call_shape<fg_tf32_8x16x8>(f);  break;
        default: GGML_ABORT("bad XMX shape %d", idx);
    }
}
static constexpr int FG_N_SHAPES = 6;

// Preference order: f16 before the promoted tf32, then the largest M x K. M and K are the weight
// dims and always full; N is the routed-token dim, narrow per expert and padded to BN. 32x64 tiles
// come last: a 64-wide N is mostly padding here and the 32x64 f32 accumulator alone needs 128
// registers per lane, so they spill (13x slower on B60). Devices that list them list 8x16 too.
static constexpr int fg_shape_order[FG_N_SHAPES] = { 1, 0, 4, 5, 3, 2 };

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. JIT compiles all of them; only a shape the device
// reports is ever launched.
#if defined(__SYCL_DEVICE_ONLY__)
#    if __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
        __SYCL_TARGET_INTEL_GPU_ARL_H__
#        define FG_AOT_SG 8
#    elif __SYCL_TARGET_INTEL_GPU_PVC__ || __SYCL_TARGET_INTEL_GPU_PVC_VG__ || __SYCL_TARGET_INTEL_GPU_BMG_G21__ || \
        __SYCL_TARGET_INTEL_GPU_BMG_G31__ || __SYCL_TARGET_INTEL_GPU_LNL_M__ || __SYCL_TARGET_INTEL_GPU_PTL_H__ ||   \
        __SYCL_TARGET_INTEL_GPU_PTL_U__ || __SYCL_TARGET_INTEL_GPU_WCL__ || __SYCL_TARGET_INTEL_GPU_NVL_S__ ||       \
        __SYCL_TARGET_INTEL_GPU_NVL_U__ || __SYCL_TARGET_INTEL_GPU_NVL_P__
#        define FG_AOT_SG 16
#    elif __SYCL_TARGET_INTEL_GPU_TGLLP__ || __SYCL_TARGET_INTEL_GPU_RKL__ || __SYCL_TARGET_INTEL_GPU_ADL_S__ || \
        __SYCL_TARGET_INTEL_GPU_ADL_P__ || __SYCL_TARGET_INTEL_GPU_ADL_N__ || __SYCL_TARGET_INTEL_GPU_DG1__ ||   \
        __SYCL_TARGET_INTEL_GPU_MTL_U__ || __SYCL_TARGET_INTEL_GPU_MTL_H__
#        define FG_AOT_SG 0 // no XMX
#    endif
#endif

template <int SG> static constexpr bool fg_built() {
#if defined(FG_AOT_SG)
    return SG == FG_AOT_SG;
#else
    return true;
#endif
}

static size_t grouped_gemm_packed_capacity(size_t size) {
    size_t capacity = 1;
    while (capacity < size) {
        capacity *= 2;
    }
    return capacity;
}

// the device lists S with an f32 accumulator
template <typename S> static bool fg_device_has_shape(const std::vector<mx::combination> & combinations) {
    const mx::matrix_type ab = S::is_tf32 ? mx::matrix_type::tf32 : mx::matrix_type::fp16;
    for (const auto & c : combinations) {
        if (c.atype == ab && c.btype == ab && c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            return true;
        }
    }
    return false;
}

static const char * fg_shape_name(int idx) {
    static const char * names[FG_N_SHAPES] = {
        "f16 8x16x16 sg16", "f16 16x16x16 sg16", "f16 32x64x16 sg16", "f16 32x64x32 sg16", "f16 8x8x16 sg8",
        "tf32 8x16x8 sg16",
    };
    return names[idx];
}

// First shape in preference order that the device reports, that this build compiled for it, and
// that GGML_SYCL_XMX_GATHER_SHAPES allows. -1 if none.
static int fg_pick_shape(const sycl::device & dev) {
    std::vector<mx::combination> combinations;
    std::vector<size_t>          sg_sizes;
    try {
        combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        sg_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
    } catch (const sycl::exception &) {
        return -1;
    }
    int available = 0;
    for (int idx = 0; idx < FG_N_SHAPES; ++idx) {
        fg_visit_shape(idx, [&](auto s) {
            using S = decltype(s);
            const bool sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
            if (sg_ok && fg_device_has_shape<S>(combinations)) {
                available |= 1 << idx;
            }
        });
    }
    int picked = -1;
    for (int idx : fg_shape_order) {
        if (available & g_ggml_sycl_xmx_gather_shapes & (1 << idx)) {
            picked = idx;
            break;
        }
    }
    GGML_LOG_INFO("%s: %s: available 0x%x, allowed 0x%x, using %s\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), available, g_ggml_sycl_xmx_gather_shapes & 0xffff,
                  picked >= 0 ? fg_shape_name(picked) : "none (XMX dequant-GEMM off)");
    return picked;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
template <typename E, typename T_src>
static void fused_gemm_pack_b(const T_src * y, typename E::store * packed, int N, int Npad, int K,
                              dpct::queue_ptr stream) {
    constexpr int V   = 4 / sizeof(typename E::store);
    const int     kqs = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        const int idx = id[0];
        const int n   = idx / kqs;
        const int kq  = idx - n * kqs;
        typename E::store vals[V] = {};
        if (n < N) {
            const T_src * src = y + (size_t) n * K + V * kq;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                vals[v] = E::cvt((float) src[v]);
            }
        }
        typename E::store * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq4_nl * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq4_nl blk = xrow[kb];
    const float        d   = (float) blk.d;
#pragma unroll
    for (int j = 0; j < QK4_NL / 2; j += 2) {
        const uint8_t q0 = blk.qs[j];
        const uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq3_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    const block_iq3_s * blk = xrow + kb / (QK_K / 32);
    const int           ib8 = kb % (QK_K / 32);
    const uint8_t *     qs  = blk->qs + 8 * ib8;
    const int           qh  = blk->qh[ib8];
    const float         d   = (float) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        const uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        const int      signs = blk->signs[4 * ib8 + il];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float g1a = (float) ((grid1 >> (16 * j + 0)) & 0xff);
            const float g1b = (float) ((grid1 >> (16 * j + 8)) & 0xff);
            const float g2a = (float) ((grid2 >> (16 * j + 0)) & 0xff);
            const float g2b = (float) ((grid2 >> (16 * j + 8)) & 0xff);
            const int   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
template <typename block_q_t> struct fg_block_traits;
template <> struct fg_block_traits<block_iq4_nl> { static constexpr int qk = QK4_NL; };
template <> struct fg_block_traits<block_iq3_s>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq3_xxs> { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq4_xs>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_xxs> { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_xs>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_s>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq1_s>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq1_m>   { static constexpr int qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
#define FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    const T * blk = xrow + kb / (QK_K / 32);                                     \
    const int ib  = kb % (QK_K / 32)

template <typename E>
static __dpct_inline__ void fg_pack_quarter(const float * __restrict__ t, typename E::pair * a, int il) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq4_xs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq4_xs);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    const float d = (float) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    const uint8_t * q4 = blk->qs + 16 * ib;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq3_xxs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq3_xxs);
    const uint8_t *  q3    = blk->qs + 8 * ib;
    const uint16_t * gas   = (const uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    const uint32_t   aux32 = gas[0] | (gas[1] << 16);
    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        float t[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_xxs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xxs);
    const uint16_t * q2    = blk->qs + 4 * ib;
    const uint8_t *  aux8  = (const uint8_t *) q2;
    const uint32_t   aux32 = q2[2] | (q2[3] << 16);
    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid  = (const uint8_t *) (iq2xxs_grid + aux8[il]);
        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_xs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xs);
    const uint16_t * q2 = blk->qs + 4 * ib;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid  = (const uint8_t *) (iq2xs_grid + (q2[il] & 511));
        const float     d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_s);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid =
            (const uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        const float   d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq1_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_s);
    const float delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d     = (float) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        const int8_t * q = (const int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq1_m * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_m);
    const uint16_t * sc = (const uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const int   ib16  = 2 * ib + il / 2;
        const float d     = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        const float delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        const int8_t * q = (const int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
template <typename S, typename block_q_t>
static void fused_dequant_gemm_tile(
    const block_q_t * __restrict__ x,
    const typename S::ts * __restrict__ packed_b,
    float * __restrict__ dst,
    const int M, const int Npad, const int K, const int ldd,
    const int b0, const int n0, const int n1,
    sycl::local_accessor<typename S::ts, 1> tile_a,
    sycl::local_accessor<float, 1> tile_c,
    const sycl::nd_item<2> & item) {
    if constexpr (fg_built<S::SG>()) {
        using E  = typename S::E;
        using TA = typename S::ta;
        const auto sg     = item.get_sub_group();
        const int  sg_id  = sg.get_group_id()[0];
        const int  lane   = sg.get_local_id()[0];
        const int  m0     = item.get_group(1) * S::SG_ROWS;
        const int  nstep  = K / FG_BK;
        const int  a_base = sg_id * S::SG_ROWS * FG_BK;
        const int  c_base = sg_id * S::SG_ROWS * S::BN;

        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, S::TM, S::TN> acc[S::MT][S::NT];
#pragma unroll
        for (int mt = 0; mt < S::MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
        const block_q_t *     xrow[S::RPL];
        bool                  row_ok[S::RPL];
        typename E::pair *    a[S::RPL];
#pragma unroll
        for (int r = 0; r < S::RPL; ++r) {
            const int row = m0 + r * S::SG + lane;
            row_ok[r] = row < M;
            xrow[r]   = x + (size_t) (row_ok[r] ? row : 0) * (K / fg_block_traits<block_q_t>::qk);
            a[r]      = (typename E::pair *) &tile_a[a_base + (r * S::SG + lane) * FG_BK];
        }

        const auto b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        const int b_stride = Npad * S::VNNI;

        const int kb_begin = (sg_id * nstep) / FG_KSPLIT;
        const int kb_end   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        for (int kb = kb_begin; kb < kb_end; ++kb) {
#pragma unroll
            for (int r = 0; r < S::RPL; ++r) {
                if (row_ok[r]) {
                    fg_stage_a<E>(xrow[r], kb, a[r]);
                } else {
#pragma unroll
                    for (int j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = E::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::group_barrier(sg);

#pragma unroll
            for (int kt = 0; kt < FG_BK / S::TK; ++kt) {
                const int kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_matrix<sycl::sub_group, TA, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
#pragma unroll
                for (int nt = 0; nt < S::NT; ++nt) {
                    mx::joint_matrix_load(sg, sub_b[nt], b_ptr + (size_t) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#pragma unroll
                for (int mt = 0; mt < S::MT; ++mt) {
                    mx::joint_matrix<sycl::sub_group, TA, mx::use::a, S::TM, S::TK, mx::layout::row_major> sub_a;
                    mx::joint_matrix_load(sg, sub_a,
                        tile_a.template get_multi_ptr<sycl::access::decorated::no>() + a_base + (mt * S::TM) * FG_BK + kt * S::TK,
                        FG_BK);
#pragma unroll
                    for (int nt = 0; nt < S::NT; ++nt) {
                        mx::joint_matrix_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
            // the next step overwrites tile_a
            sycl::group_barrier(sg);
        }

#pragma unroll
        for (int mt = 0; mt < S::MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_store(sg, acc[mt][nt],
                    tile_c.template get_multi_ptr<sycl::access::decorated::no>() + c_base + (mt * S::TM) * S::BN + nt * S::TN,
                    S::BN, mx::layout::row_major);
            }
        }
        sycl::group_barrier(item.get_group());

        // sum the K splits; consecutive lanes write consecutive rows of one dst column
        for (int idx = item.get_local_linear_id(); idx < S::SG_ROWS * S::BN; idx += S::WG_SIZE) {
            const int r = idx % S::SG_ROWS;
            const int c = idx / S::SG_ROWS;
            const int m = m0 + r;
            const int n = n0 + c;
            if (m < M && n < n1) {
                float sum = 0.0f;
#pragma unroll
                for (int s = 0; s < FG_KSPLIT; ++s) {
                    sum += tile_c[s * S::SG_ROWS * S::BN + r * S::BN + c];
                }
                dst[(size_t) n * ldd + m] = sum;
            }
        }
    }
}

template <typename S, typename block_q_t>
static void fused_dequant_gemm_launch(const void * src0, const typename S::ts * packed, float * dst, const int M,
                                      const int N, const int Npad, const int K, const int ldd,
                                      const int64_t groups_n, const int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::ts, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                const int n0 = item.get_group(0) * S::BN;
                fused_dequant_gemm_tile<S, block_q_t>((const block_q_t *) src0, packed, dst, M, Npad, K, ldd,
                                                      n0, n0, N, tile_a, tile_c, item);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
template <typename S, typename block_q_t>
static void grouped_dequant_gemm_launch(const char * src0_dd, const size_t expert_stride,
                                        const ggml_sycl_gg_tile * tiles_ptr, const typename S::ts * packed, float * dst,
                                        const int M, const int Npad, const int K, const int64_t n_tiles,
                                        const int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::ts, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                const int               t    = item.get_group(0);
                const ggml_sycl_gg_tile tile = tiles_ptr[t];
                const block_q_t *       x    = (const block_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, block_q_t>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0, tile.n1,
                                                      tile_a, tile_c, item);
            });
    });
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
template <typename S>
static void grouped_gemm_pack_b(const float * y, typename S::ts * packed, const ggml_sycl_gg_tile * tiles, int Npad,
                                int K, dpct::queue_ptr stream) {
    using E        = typename S::E;
    constexpr int V = S::VNNI;
    const int kqs   = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        const size_t idx = id[0];
        const int    kq  = idx / Npad;
        const int    n   = idx - (size_t) kq * Npad;
        const ggml_sycl_gg_tile tile = tiles[n / S::BN];
        const int    row = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        typename S::ts vals[V] = {};
        if (row < tile.n1) {
            const float * src = y + (size_t) row * K + V * kq;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        typename S::ts * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

template <typename T> struct fg_tag { using type = T; };

// calls f(fg_tag<block_q_t>{}) for the weight format; false if it has no A stage
template <typename F> static bool fg_visit_type(ggml_type type, F && f) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:  f(fg_tag<block_iq4_nl>{});  return true;
        case GGML_TYPE_IQ3_S:   f(fg_tag<block_iq3_s>{});   return true;
        case GGML_TYPE_IQ4_XS:  f(fg_tag<block_iq4_xs>{});  return true;
        case GGML_TYPE_IQ3_XXS: f(fg_tag<block_iq3_xxs>{}); return true;
        case GGML_TYPE_IQ2_XXS: f(fg_tag<block_iq2_xxs>{}); return true;
        case GGML_TYPE_IQ2_XS:  f(fg_tag<block_iq2_xs>{});  return true;
        case GGML_TYPE_IQ2_S:   f(fg_tag<block_iq2_s>{});   return true;
        case GGML_TYPE_IQ1_S:   f(fg_tag<block_iq1_s>{});   return true;
        case GGML_TYPE_IQ1_M:   f(fg_tag<block_iq1_m>{});   return true;
        default:                return false;
    }
}

// Shape index for the device, or -1. Cached per device, not once: on a mixed box the first
// caller's verdict is not the others'.
static int fg_device_shape(dpct::queue_ptr stream) {
    static std::mutex                            mtx;
    static std::unordered_map<sycl::device, int> known;
    const sycl::device                           dev = stream->get_device();
    std::lock_guard<std::mutex>                  lock(mtx);
    const auto                                   it = known.find(dev);
    if (it != known.end()) {
        return it->second;
    }
    const int idx = fg_pick_shape(dev);
    known.emplace(dev, idx);
    return idx;
}

bool ggml_sycl_fused_dequant_gemm_f16_device_ok(dpct::queue_ptr stream) {
    return fg_device_shape(stream) >= 0;
}

template <typename S>
static bool fg_fused_run(ggml_type src0_type, const void * src0, const sycl::half * src1_f16, float * dst, int64_t M,
                         int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    const int64_t groups_n = (N + S::BN - 1) / S::BN;
    const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    const int     Npad     = (int) (groups_n * S::BN);

    ggml_sycl_pool_alloc<typename S::ts> packed_b(pool, (size_t) K * Npad);
    fused_gemm_pack_b<typename S::E>(src1_f16, packed_b.get(), (int) N, Npad, (int) K, stream);

    const typename S::ts * packed = packed_b.get();
    return fg_visit_type(src0_type, [&](auto tag) {
        using block_q_t = typename decltype(tag)::type;
        fused_dequant_gemm_launch<S, block_q_t>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                groups_n, groups_m, stream);
    });
}

bool ggml_sycl_fused_dequant_gemm_f16(ggml_type src0_type, const void * src0, const sycl::half * src1_f16, float * dst,
                                      int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                                      dpct::queue_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM
    if (!ggml_sycl_xmx_gather_type_enabled(src0_type)) {
        return false;
    }
    if (!ggml_sycl_fused_dequant_gemm_f16_shape_ok(src0_type, M, N, K, ldd)) {
        return false;
    }
    const int shape = fg_device_shape(stream);
    if (shape < 0) {
        return false;
    }
    bool ok = false;
    fg_visit_shape(shape, [&](auto s) {
        ok = fg_fused_run<decltype(s)>(src0_type, src0, src1_f16, dst, M, N, K, ldd, pool, stream);
    });
    return ok;
}

template <typename S>
static bool fg_grouped_run(ggml_type src0_type, const void * src0_base, size_t expert_stride, const float * src1,
                           float * dst, const int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                           std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    for (int64_t e = 0; e < n_as; ++e) {
        const int64_t end = expert_row_offsets[e + 1];
        for (int64_t n0 = expert_row_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    const int64_t n_tiles  = tiles.size();
    const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    const int     Npad     = (int) (n_tiles * S::BN);

    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, n_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(tiles_dev.get(), tiles.data(), n_tiles * sizeof(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<typename S::ts> packed_b(pool, grouped_gemm_packed_capacity((size_t) K * Npad));
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (int) K, stream);

    const typename S::ts *    packed    = packed_b.get();
    const ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
    const char *              src0_dd   = (const char *) src0_base;
    return fg_visit_type(src0_type, [&](auto tag) {
        using block_q_t = typename decltype(tag)::type;
        grouped_dequant_gemm_launch<S, block_q_t>(src0_dd, expert_stride, tiles_ptr, packed, dst, (int) M, Npad,
                                                  (int) K, n_tiles, groups_m, stream);
    });
}

bool ggml_sycl_grouped_dequant_gemm_f16(ggml_type src0_type, const void * src0_base, size_t expert_stride,
                                        const float * src1, float * dst, const int64_t * expert_row_offsets,
                                        int64_t n_as, int64_t M, int64_t K, int64_t total_rows,
                                        std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                                        dpct::queue_ptr stream) {
    int64_t n_active = 0;
    for (int64_t e = 0; e < n_as; ++e) {
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
    }
    if (!ggml_sycl_xmx_gather_type_enabled(src0_type)) {
        return false;
    }
    if (!ggml_sycl_grouped_dequant_gemm_f16_shape_ok(src0_type, M, K, total_rows, n_active)) {
        return false;
    }
    const int shape = fg_device_shape(stream);
    if (shape < 0) {
        return false;
    }
    bool ok = false;
    fg_visit_shape(shape, [&](auto s) {
        ok = fg_grouped_run<decltype(s)>(src0_type, src0_base, expert_stride, src1, dst, expert_row_offsets, n_as, M,
                                         K, tiles, pool, stream);
    });
    return ok;
}
