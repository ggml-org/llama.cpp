#include "fused-gemm.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <string>
#include <tuple>
#include <mutex>
#include <set>
#include <unordered_map>

namespace mx = sycl::ext::oneapi::experimental::matrix;

// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
static constexpr int FG_BK     = QK4_NL;
static constexpr int FG_KSPLIT = 4;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
template <typename T> struct fg_elem;

template <> struct fg_elem<sycl::half> {
    using store = sycl::half;
    using pair  = sycl::half2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::fp16;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    static constexpr ggml_type       src   = GGML_TYPE_F16;
    static constexpr int             mant  = 10;
    static constexpr int             slow  = 0;
    static store cvt(float x) { return (store) x; }
    static pair make(float x, float y) { return pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
static inline uint32_t fg_round_bits(float x, int drop) {
    const uint32_t u = sycl::bit_cast<uint32_t>(x);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        return (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    return u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

struct alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

template <> struct fg_elem<sycl::ext::oneapi::bfloat16> {
    using store = sycl::ext::oneapi::bfloat16;
    using pair  = fg_bf16x2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::bf16;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    static constexpr ggml_type       src   = GGML_TYPE_BF16;
    static constexpr int             mant  = 7;
    static constexpr int             slow  = 0;
    static store cvt(float x) { return store(x); }
    static pair make(float x, float y) { return { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
template <> struct fg_elem<mx::precision::tf32> {
    using store = float;
    using pair  = sycl::float2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::tf32;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    static constexpr ggml_type       src   = GGML_TYPE_COUNT;
    static constexpr int             mant  = 10;
    static constexpr int             slow  = 1;
    static store cvt(float x) { return sycl::bit_cast<float>(fg_round_bits(x, 13) & ~0x1fffu); }
    static pair make(float x, float y) { return pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
template <typename TA, typename TB, int TM_, int TN_, int TK_, int SG_> struct fg_combo {
    using ta  = TA;
    using tb  = TB;
    using EA  = fg_elem<TA>;
    using EB  = fg_elem<TB>;
    using tsa = typename EA::store;
    using tsb = typename EB::store;
    static constexpr int TM = TM_;
    static constexpr int TN = TN_;
    static constexpr int TK = TK_;
    static constexpr int SG = SG_;
    static constexpr int VNNI    = 4 / sizeof(tsb);  // K rows of B packed in one 32-bit word
    static constexpr int SG_ROWS = TM > 16 ? TM : 16;
    static constexpr int RPL     = SG_ROWS / SG;     // A rows one lane decodes per k step
    static constexpr int MT      = SG_ROWS / TM;
    static constexpr int BN      = TN > 32 ? TN : 32;
    static constexpr int NT      = BN / TN;
    static constexpr int WG_SIZE = FG_KSPLIT * SG;
    static constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::row_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    static constexpr bool efficient = TN <= 32;
    static_assert(SG_ROWS % SG == 0 && SG_ROWS % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

using fg_half = sycl::half;
using fg_bf16 = sycl::ext::oneapi::bfloat16;
using fg_tf32 = mx::precision::tf32;

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
template <typename F> static void fg_visit_combo(int idx, F && f);
static constexpr int FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
template <int SG> static constexpr bool fg_listed() {
#if defined(GGML_SYCL_XMX_AOT_SG)
    return SG == GGML_SYCL_XMX_AOT_SG;
#else
    return true;
#endif
}

template <typename S, typename F> static void fg_call_combo(F && f) {
    if constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

template <typename F> static void fg_visit_combo(int idx, F && f) {
    switch (idx) {
        case 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
        case 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
        case 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   break; // Xe2, Xe3, Xe-HPC
        case 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
        case 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
        default: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
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

// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
static constexpr int64_t grouped_gemm_max_tiles(int64_t total_rows, int64_t n_as, int64_t BN) {
    return total_rows <= n_as ? total_rows : n_as + (total_rows - n_as) / BN;
}
// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);

// the device lists S with an f32 accumulator and output
template <typename S> static bool fg_device_has_combo(const std::vector<mx::combination> & combinations) {
    for (const auto & c : combinations) {
        if (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::matrix_type::fp32 &&
            c.dtype == mx::matrix_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            return true;
        }
    }
    return false;
}

template <typename T> static const char * fg_type_name() {
    return std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

static std::string fg_combo_name(int idx) {
    std::string name;
    fg_visit_combo(idx, [&](auto s) {
        using S = decltype(s);
        name = std::string(fg_type_name<typename S::ta>()) + "x" + fg_type_name<typename S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    return name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
static int fg_device_combos(const sycl::device & dev) {
    static std::mutex                            mtx;
    static std::unordered_map<sycl::device, int> known;
    std::lock_guard<std::mutex>                  lock(mtx);
    const auto                                   it = known.find(dev);
    if (it != known.end()) {
        return it->second;
    }
    int available = 0;
    try {
        const auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        const auto sg_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
        for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                const bool sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
                if (sg_ok && fg_device_has_combo<S>(combinations)) {
                    available |= 1 << idx;
                }
            });
        }
    } catch (const sycl::exception &) {
        available = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), available, g_ggml_sycl_xmx_gather_shapes);
    known.emplace(dev, available);
    return available;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
template <typename S> static int64_t fg_rank(ggml_type src1_type) {
    const int64_t spills  = !S::efficient;
    const int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    const int64_t convert = S::EB::src != src1_type;
    const int64_t lossy   = S::EB::mant < 10;
    const int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    const int64_t mk      = 1024 - S::TM * S::TK;
    return ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
static bool fg_mode_meets(int mode, int32_t src1_prec) {
    if (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        return true;
    }
    if (src1_prec >= GGML_PREC_BF16) {
        return mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    switch (g_ggml_sycl_dynamic_required_precision) {
        case GGML_SYCL_DYNAMIC_PRECISION_TF32: return mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        case GGML_SYCL_DYNAMIC_PRECISION_BF16: return mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        default:                               return false;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first).
static int fg_pick_combo(dpct::queue_ptr stream, ggml_type src1_type, int32_t src1_prec) {
    const sycl::device dev     = stream->get_device();
    const int          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    const bool         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    const int          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    int     best      = -1;
    int64_t best_rank = 0;
    for (int i = 0; i < 3 && best < 0; ++i) {
        const int mode = modes[i];
        if (!fg_mode_meets(mode, src1_prec)) {
            continue;
        }
        for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
            if (!(allowed & (1 << idx))) {
                continue;
            }
            fg_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                if (S::EA::mode != mode || S::EB::mode != mode) {
                    return;
                }
                const int64_t rank = fg_rank<S>(src1_type);
                if (best < 0 || rank < best_rank) {
                    best      = idx;
                    best_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    static std::mutex                                      mtx;
    static std::set<std::tuple<size_t, int, int32_t, int>> seen;
    std::lock_guard<std::mutex>                            lock(mtx);
    if (seen.emplace(std::hash<sycl::device>{}(dev), (int) src1_type, src1_prec, best).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      best >= 0 ? fg_combo_name(best).c_str() : "none (library GEMM)");
    }
    return best;
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
template <> struct fg_block_traits<block_q8_0>    { static constexpr int qk = QK8_0; };
template <> struct fg_block_traits<block_q4_K>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q5_K>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q6_K>    { static constexpr int qk = QK_K; };

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


// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
// live, as in dequantize.hpp.
template <typename E>
static __dpct_inline__ void fg_decode_q8_0(const int8_t * __restrict__ qs, const float d, typename E::pair * a) {
#pragma unroll
    for (int j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// same unpack as get_scale_min_k4() in dequantize.hpp
static __dpct_inline__ void fg_scale_min_k4(const int j, const uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// q4_K sub-block ib (0..7) uses scale/min pair ib and the low (even ib) or high (odd ib) nibbles of
// qs[32 * (ib / 2) ...], as in dequantize_row_q4_K
template <typename E>
static __dpct_inline__ void fg_decode_q4_K(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
                                           const sycl::half2 dm, const int ib, typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    const float     d     = (float) dm[0] * sc;
    const float     m     = (float) dm[1] * mb;
    const uint8_t * q     = qs + 32 * (ib / 2);
    const int       shift = 4 * (ib % 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        a[j] = E::make(d * ((q[2 * j] >> shift) & 0xF) - m, d * ((q[2 * j + 1] >> shift) & 0xF) - m);
    }
}

// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
template <typename E>
static __dpct_inline__ void fg_decode_q5_K(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ qh,
                                           const uint8_t * __restrict__ scales, const sycl::half2 dm, const int ib,
                                           typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    const float     d     = (float) dm[0] * sc;
    const float     m     = (float) dm[1] * mb;
    const uint8_t * q     = qs + 32 * (ib / 2);
    const int       shift = 4 * (ib % 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int l = 2 * j;
        a[j] = E::make(d * (((q[l] >> shift) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
                       d * (((q[l + 1] >> shift) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
    }
}

// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
template <typename E>
static __dpct_inline__ void fg_decode_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                                           const int8_t * __restrict__ scales, const float d, const int ib,
                                           typename E::pair * a) {
    const int       h  = ib / 4;
    const int       r  = ib % 4;
    const uint8_t * q  = ql + 64 * h + 32 * (r & 1);
    const uint8_t * hb = qh + 32 * h;
    const int8_t *  sc = scales + 8 * h + 2 * r;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int   l  = 2 * j;
        const float dl = d * sc[j / 8];
        const int   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
        const int   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q8_0 * __restrict__ xrow, const int kb, typename E::pair * a) {
    fg_decode_q8_0<E>(xrow[kb].qs, (float) xrow[kb].d, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q4_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q4_K);
    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q5_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q5_K);
    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q6_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q6_K);
    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (float) blk->d, ib, a);
}

// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
template <typename block_q_t> struct fg_soa {
    static constexpr bool supported = false;
};

template <> struct fg_soa<block_q8_0> {
    static constexpr bool supported = true;
    // [qs][d]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int,
                                      typename E::pair * a) {
        const float d = (float) ((const sycl::half *) (x + nblocks * QK8_0))[ib];
        fg_decode_q8_0<E>((const int8_t *) x + ib * QK8_0, d, a);
    }
};

template <> struct fg_soa<block_q4_K> {
    static constexpr bool supported = true;
    // [qs][scales][dm]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t *   scales = x + nblocks * (QK_K / 2);
        const sycl::half2 dm     = ((const sycl::half2 *) (scales + nblocks * K_SCALE_SIZE))[ib];
        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_SIZE, dm, kb % (QK_K / 32), a);
    }
};

template <> struct fg_soa<block_q5_K> {
    static constexpr bool supported = true;
    // [qs][qh][scales][dm]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t *   qh     = x + nblocks * (QK_K / 2);
        const uint8_t *   scales = qh + nblocks * (QK_K / 8);
        const sycl::half2 dm     = ((const sycl::half2 *) (scales + nblocks * K_SCALE_SIZE))[ib];
        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_SIZE, dm,
                          kb % (QK_K / 32), a);
    }
};

template <> struct fg_soa<block_q6_K> {
    static constexpr bool supported = true;
    // [ql][qh][scales][d]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t * qh     = x + nblocks * (QK_K / 2);
        const uint8_t * scales = qh + nblocks * (QK_K / 4);
        const float     d      = (float) ((const sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (const int8_t *) scales + ib * (QK_K / 16), d,
                          kb % (QK_K / 32), a);
    }
};


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
template <typename S, typename block_q_t, bool reordered>
static void fused_dequant_gemm_tile(
    const block_q_t * __restrict__ x,
    const typename S::tsb * __restrict__ packed_b,
    float * __restrict__ dst,
    const int M, const int Npad, const int K, const int ldd,
    const int b0, const int n0, const int n1,
    sycl::local_accessor<typename S::tsa, 1> tile_a,
    sycl::local_accessor<float, 1> tile_c,
    const sycl::nd_item<2> & item) {
    if constexpr (fg_built<S::SG>()) {
        using EA = typename S::EA;
        using TA = typename S::ta;
        using TB = typename S::tb;
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
        constexpr int         KPB     = fg_block_traits<block_q_t>::qk / FG_BK; // k steps per block
        const size_t          bpr     = K / fg_block_traits<block_q_t>::qk;
        const size_t          nblocks = (size_t) M * bpr;
        size_t                row_blk[S::RPL];
        bool                  row_ok[S::RPL];
        typename EA::pair *   a[S::RPL];
#pragma unroll
        for (int r = 0; r < S::RPL; ++r) {
            const int row = m0 + r * S::SG + lane;
            row_ok[r]  = row < M;
            row_blk[r] = (size_t) (row_ok[r] ? row : 0) * bpr;
            a[r]       = (typename EA::pair *) &tile_a[a_base + (r * S::SG + lane) * FG_BK];
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
                    if constexpr (reordered) {
                        fg_soa<block_q_t>::template stage<EA>((const uint8_t *) x, nblocks, row_blk[r] + kb / KPB, kb,
                                                              a[r]);
                    } else {
                        fg_stage_a<EA>(x + row_blk[r], kb, a[r]);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = EA::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::group_barrier(sg);

#pragma unroll
            for (int kt = 0; kt < FG_BK / S::TK; ++kt) {
                const int kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_matrix<sycl::sub_group, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
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
static void fused_dequant_gemm_launch(const void * src0, const typename S::tsb * packed, float * dst, const int M,
                                      const int N, const int Npad, const int K, const int ldd,
                                      const int64_t groups_n, const int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                const int n0 = item.get_group(0) * S::BN;
                fused_dequant_gemm_tile<S, block_q_t, false>((const block_q_t *) src0, packed, dst, M, Npad, K, ldd,
                                                      n0, n0, N, tile_a, tile_c, item);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
template <typename S, typename block_q_t, bool reordered>
static void grouped_dequant_gemm_launch(const char * src0_dd, const size_t expert_stride,
                                        const ggml_sycl_gg_tile * tiles_ptr, const typename S::tsb * packed, float * dst,
                                        const int M, const int Npad, const int K, const int64_t n_tiles,
                                        const int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                const int               t    = item.get_group(0);
                const ggml_sycl_gg_tile tile = tiles_ptr[t];
                const block_q_t *       x    = (const block_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, block_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
                                                                 tile.n1, tile_a, tile_c, item);
            });
    });
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
template <typename S>
static void grouped_gemm_pack_b(const float * y, typename S::tsb * packed, const ggml_sycl_gg_tile * tiles, int Npad,
                                int K, dpct::queue_ptr stream) {
    using E        = typename S::EB;
    constexpr int V = S::VNNI;
    const int kqs   = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        const size_t idx = id[0];
        const int    kq  = idx / Npad;
        const int    n   = idx - (size_t) kq * Npad;
        const ggml_sycl_gg_tile tile = tiles[n / S::BN];
        const int    row = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        typename S::tsb vals[V] = {};
        if (row < tile.n1) {
            const float * src = y + (size_t) row * K + V * kq;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        typename S::tsb * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
template <typename T> static constexpr bool fg_plain_ok() {
    return !std::is_same_v<T, block_q8_0> && !std::is_same_v<T, block_q4_K> && !std::is_same_v<T, block_q5_K> &&
           !std::is_same_v<T, block_q6_K>;
}

template <typename T, bool R> struct fg_tag {
    using type                     = T;
    static constexpr bool reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
template <typename T, typename F> static bool fg_visit_layout(bool reordered, F && f) {
    if (!reordered) {
        f(fg_tag<T, false>{});
        return true;
    }
    if constexpr (fg_soa<T>::supported) {
        f(fg_tag<T, true>{});
        return true;
    }
    return false;
}

template <typename F> static bool fg_visit_type(ggml_type type, bool reordered, F && f) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:  return fg_visit_layout<block_iq4_nl>(reordered, f);
        case GGML_TYPE_IQ3_S:   return fg_visit_layout<block_iq3_s>(reordered, f);
        case GGML_TYPE_IQ4_XS:  return fg_visit_layout<block_iq4_xs>(reordered, f);
        case GGML_TYPE_IQ3_XXS: return fg_visit_layout<block_iq3_xxs>(reordered, f);
        case GGML_TYPE_IQ2_XXS: return fg_visit_layout<block_iq2_xxs>(reordered, f);
        case GGML_TYPE_IQ2_XS:  return fg_visit_layout<block_iq2_xs>(reordered, f);
        case GGML_TYPE_IQ2_S:   return fg_visit_layout<block_iq2_s>(reordered, f);
        case GGML_TYPE_IQ1_S:   return fg_visit_layout<block_iq1_s>(reordered, f);
        case GGML_TYPE_IQ1_M:   return fg_visit_layout<block_iq1_m>(reordered, f);
        case GGML_TYPE_Q8_0:    return fg_visit_layout<block_q8_0>(reordered, f);
        case GGML_TYPE_Q4_K:    return fg_visit_layout<block_q4_K>(reordered, f);
        case GGML_TYPE_Q5_K:    return fg_visit_layout<block_q5_K>(reordered, f);
        case GGML_TYPE_Q6_K:    return fg_visit_layout<block_q6_K>(reordered, f);
        default:                return false;
    }
}

template <typename S>
static bool fg_fused_run(ggml_type src0_type, const void * src0, const void * src1, ggml_type src1_type,
                         float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                         dpct::queue_ptr stream) {
    const int64_t groups_n = (N + S::BN - 1) / S::BN;
    const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    const int     Npad     = (int) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * Npad);
    if (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<typename S::EB>((const sycl::half *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    } else if (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<typename S::EB>((const fg_bf16 *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    } else {
        fused_gemm_pack_b<typename S::EB>((const float *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    }

    const typename S::tsb * packed = packed_b.get();
    return fg_visit_type(src0_type, false, [&](auto tag) {
        using block_q_t = typename decltype(tag)::type;
        if constexpr (fg_plain_ok<block_q_t>()) {
            fused_dequant_gemm_launch<S, block_q_t>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                    groups_n, groups_m, stream);
        }
    });
}

bool ggml_sycl_fused_dequant_gemm(ggml_type src0_type, const void * src0, const void * src1, ggml_type src1_type,
                                  int32_t src1_prec, float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM
    bool plain_ok = false;
    fg_visit_type(src0_type, false, [&](auto tag) { plain_ok = fg_plain_ok<typename decltype(tag)::type>(); });
    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
        return false;
    }
    if (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        return false;
    }
    if (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        return false;
    }
    const int combo = fg_pick_combo(stream, src1_type, src1_prec);
    if (combo < 0) {
        return false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_fused_run<decltype(s)>(src0_type, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
    });
    return launched;
}

template <typename S>
static bool fg_grouped_run(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
                           const float * src1, float * dst, const int64_t * expert_row_offsets, int64_t n_as, int64_t M,
                           int64_t K, std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                           dpct::queue_ptr stream) {
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

    const int64_t max_tiles = grouped_gemm_max_tiles(expert_row_offsets[n_as], n_as, S::BN);
    GGML_ASSERT(n_tiles <= max_tiles);
    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, max_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(tiles_dev.get(), tiles.data(), n_tiles * sizeof(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * max_tiles * S::BN);
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (int) K, stream);

    const typename S::tsb *   packed    = packed_b.get();
    const ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
    const char *              src0_dd   = (const char *) src0_base;
    return fg_visit_type(src0_type, reordered, [&](auto tag) {
        using T = decltype(tag);
        grouped_dequant_gemm_launch<S, typename T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
                                                                       (int) M, Npad, (int) K, n_tiles, groups_m,
                                                                       stream);
    });
}

bool ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
                                    const float * src1, int32_t src1_prec, float * dst,
                                    const int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t total_rows, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    int64_t n_active = 0;
    for (int64_t e = 0; e < n_as; ++e) {
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
    }
    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_visit_type(src0_type, reordered, [](auto) {})) {
        return false;
    }
    if (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, total_rows, n_active)) {
        return false;
    }
    const int combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    if (combo < 0) {
        return false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_grouped_run<decltype(s)>(src0_type, reordered, src0_base, expert_stride, src1, dst,
                                               expert_row_offsets, n_as, M, K, tiles, pool, stream);
    });
    return launched;
}
