#pragma once

#include "common.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

using xmx_matrix_type = sycl::ext::oneapi::experimental::matrix::matrix_type;
using xmx_layout      = sycl::ext::oneapi::experimental::matrix::layout;
using xmx_combination = sycl::ext::oneapi::experimental::matrix::combination;
using xmx_tf32        = sycl::ext::oneapi::experimental::matrix::precision::tf32;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
template <typename T> struct xmx_elem;

template <> struct xmx_elem<sycl::half> {
    using store = sycl::half;
    using pair  = sycl::half2;
    static constexpr xmx_matrix_type mtype = xmx_matrix_type::fp16;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    static constexpr ggml_type       src   = GGML_TYPE_F16;
    static constexpr int             mant  = 10;
    static constexpr int             slow  = 0;
    static store cvt(float x) { return (store) x; }
    static pair make(float x, float y) { return pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
static inline uint32_t xmx_round_bits(float x, int drop) {
    const uint32_t u = sycl::bit_cast<uint32_t>(x);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        return (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    return u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

struct alignas(4) xmx_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

template <> struct xmx_elem<sycl::ext::oneapi::bfloat16> {
    using store = sycl::ext::oneapi::bfloat16;
    using pair  = xmx_bf16x2;
    static constexpr xmx_matrix_type mtype = xmx_matrix_type::bf16;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    static constexpr ggml_type       src   = GGML_TYPE_BF16;
    static constexpr int             mant  = 7;
    static constexpr int             slow  = 0;
    static store cvt(float x) { return store(x); }
    static pair make(float x, float y) { return { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
template <> struct xmx_elem<xmx_tf32> {
    using store = float;
    using pair  = sycl::float2;
    static constexpr xmx_matrix_type mtype = xmx_matrix_type::tf32;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    static constexpr ggml_type       src   = GGML_TYPE_COUNT;
    static constexpr int             mant  = 10;
    static constexpr int             slow  = 1;
    static store cvt(float x) { return sycl::bit_cast<float>(xmx_round_bits(x, 13) & ~0x1fffu); }
    static pair make(float x, float y) { return pair(cvt(x), cvt(y)); }
};

template <typename TA, typename TB, int TM_, int TN_, int TK_, int SG_> struct xmx_combo {
    using ta  = TA;
    using tb  = TB;
    using EA  = xmx_elem<TA>;
    using EB  = xmx_elem<TB>;
    using tsa = typename EA::store;
    using tsb = typename EB::store;
    static constexpr int TM = TM_;
    static constexpr int TN = TN_;
    static constexpr int TK = TK_;
    static constexpr int SG = SG_;
    static constexpr int VNNI    = 4 / sizeof(tsb);  // K rows of B packed in one 32-bit word
    static constexpr xmx_layout b_layout = VNNI == 1 ? xmx_layout::row_major : xmx_layout::ext_intel_packed;
    static constexpr bool efficient = TN <= 32;
};

using xmx_half = sycl::half;
using xmx_bf16 = sycl::ext::oneapi::bfloat16;

// One bit of the XMX shape mask per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
template <typename F> inline void xmx_visit_combo(int idx, F && f);
static constexpr int XMX_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
template <int SG> constexpr bool xmx_listed() {
#if defined(GGML_SYCL_XMX_AOT_SG)
    return SG == GGML_SYCL_XMX_AOT_SG;
#else
    return true;
#endif
}

template <typename S, typename F> inline void xmx_call_combo(F && f) {
    if constexpr (xmx_listed<S::SG>()) {
        f(S{});
    }
}

template <typename F> inline void xmx_visit_combo(int idx, F && f) {
    switch (idx) {
        case 0: xmx_call_combo<xmx_combo<xmx_half, xmx_half, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
        case 1: xmx_call_combo<xmx_combo<xmx_half, xmx_half, 16, 16, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 2: xmx_call_combo<xmx_combo<xmx_half, xmx_half, 32, 64, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 3: xmx_call_combo<xmx_combo<xmx_half, xmx_half, 32, 64, 32, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 4: xmx_call_combo<xmx_combo<xmx_half, xmx_half, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
        case 5: xmx_call_combo<xmx_combo<xmx_tf32, xmx_tf32, 8, 16, 8, 16>>(f);   break; // Xe2, Xe3, Xe-HPC
        case 6: xmx_call_combo<xmx_combo<xmx_bf16, xmx_bf16, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
        case 7: xmx_call_combo<xmx_combo<xmx_bf16, xmx_bf16, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
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
#        define XMX_AOT_SG 8
#    elif __SYCL_TARGET_INTEL_GPU_PVC__ || __SYCL_TARGET_INTEL_GPU_PVC_VG__ || __SYCL_TARGET_INTEL_GPU_BMG_G21__ || \
        __SYCL_TARGET_INTEL_GPU_BMG_G31__ || __SYCL_TARGET_INTEL_GPU_LNL_M__ || __SYCL_TARGET_INTEL_GPU_PTL_H__ ||   \
        __SYCL_TARGET_INTEL_GPU_PTL_U__ || __SYCL_TARGET_INTEL_GPU_WCL__ || __SYCL_TARGET_INTEL_GPU_NVL_S__ ||       \
        __SYCL_TARGET_INTEL_GPU_NVL_U__ || __SYCL_TARGET_INTEL_GPU_NVL_P__
#        define XMX_AOT_SG 16
#    elif __SYCL_TARGET_INTEL_GPU_TGLLP__ || __SYCL_TARGET_INTEL_GPU_RKL__ || __SYCL_TARGET_INTEL_GPU_ADL_S__ || \
        __SYCL_TARGET_INTEL_GPU_ADL_P__ || __SYCL_TARGET_INTEL_GPU_ADL_N__ || __SYCL_TARGET_INTEL_GPU_DG1__ ||   \
        __SYCL_TARGET_INTEL_GPU_MTL_U__ || __SYCL_TARGET_INTEL_GPU_MTL_H__
#        define XMX_AOT_SG 0 // no XMX
#    endif
#endif

template <int SG> constexpr bool xmx_built() {
#if defined(XMX_AOT_SG)
    return SG == XMX_AOT_SG;
#else
    return true;
#endif
}

// the device lists S with an f32 accumulator and output
template <typename S> inline bool xmx_device_has_combo(const std::vector<xmx_combination> & combinations) {
    for (const auto & c : combinations) {
        if (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == xmx_matrix_type::fp32 &&
            c.dtype == xmx_matrix_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            return true;
        }
    }
    return false;
}

template <typename T> inline const char * xmx_type_name() {
    return std::is_same_v<T, xmx_half> ? "f16" : std::is_same_v<T, xmx_bf16> ? "bf16" : "tf32";
}

inline std::string xmx_combo_name(int idx) {
    std::string name;
    xmx_visit_combo(idx, [&](auto s) {
        using S = decltype(s);
        name = std::string(xmx_type_name<typename S::ta>()) + "x" + xmx_type_name<typename S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    return name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
inline int xmx_device_combos(const sycl::device & dev) {
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
        for (int idx = 0; idx < XMX_N_COMBOS; ++idx) {
            xmx_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                const bool sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
                if (sg_ok && xmx_device_has_combo<S>(combinations)) {
                    available |= 1 << idx;
                }
            });
        }
    } catch (const sycl::exception &) {
        available = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX combinations available 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), available);
    known.emplace(dev, available);
    return available;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
template <typename S> inline int64_t xmx_rank(ggml_type src1_type) {
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
inline bool xmx_mode_meets(int mode, int32_t src1_prec) {
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
// for an F32 request (most mantissa first). Combinations in the prefer mask rank before the others of the same type.
inline int xmx_pick_combo(dpct::queue_ptr stream, ggml_type src1_type, int32_t src1_prec, int shapes, int prefer = 0) {
    const sycl::device dev     = stream->get_device();
    const int          allowed = xmx_device_combos(dev) & shapes;
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
        if (!xmx_mode_meets(mode, src1_prec)) {
            continue;
        }
        for (int idx = 0; idx < XMX_N_COMBOS; ++idx) {
            if (!(allowed & (1 << idx))) {
                continue;
            }
            xmx_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                if (S::EA::mode != mode || S::EB::mode != mode) {
                    return;
                }
                const int64_t rank = xmx_rank<S>(src1_type) - ((prefer >> idx) & 1 ? (int64_t) 1 << 40 : 0);
                if (best < 0 || rank < best_rank) {
                    best      = idx;
                    best_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    static std::mutex                                         mtx;
    static std::set<std::tuple<size_t, int, int32_t, int, int>> seen;
    std::lock_guard<std::mutex>                               lock(mtx);
    if (seen.emplace(std::hash<sycl::device>{}(dev), (int) src1_type, src1_prec, shapes, best).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      best >= 0 ? xmx_combo_name(best).c_str() : "none (fallback)");
    }
    return best;
}

