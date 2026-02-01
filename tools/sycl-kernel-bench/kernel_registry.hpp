#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

#include "ggml.h"

namespace sycl_bench {

enum class KernelKind {
    MMVQ,
    MMQ,
    ONEDNN_FP16_GEMM,
    ONEDNN_INT8_GEMM,
    ONEDNN_WOQ_GEMM,
    UNIFIED_MATMUL,
    MEMORY_BANDWIDTH,
    ROOFLINE_COMPUTE,
    DPAS_EXPLORATION,
};

struct KernelInfo {
    const char *     name;
    ggml_layout_mode layout;
    KernelKind       kind;
};

inline std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline const std::vector<KernelInfo> & kernel_list() {
    static const std::vector<KernelInfo> kernels = {
        { "mmvq_aos",       GGML_LAYOUT_AOS,       KernelKind::MMVQ },
        { "mmvq_aos_baseline", GGML_LAYOUT_AOS,    KernelKind::MMVQ },
        { "mmvq_soa",       GGML_LAYOUT_SOA,       KernelKind::MMVQ },
        { "mmvq_soa_baseline", GGML_LAYOUT_SOA,    KernelKind::MMVQ },
        { "mmvq_coalesced", GGML_LAYOUT_COALESCED, KernelKind::MMVQ },
        { "mmvq_slm_cached", GGML_LAYOUT_AOS,      KernelKind::MMVQ },
        { "mmvq_prefetch", GGML_LAYOUT_AOS,        KernelKind::MMVQ },
        { "mmvq_wide_load", GGML_LAYOUT_AOS,       KernelKind::MMVQ },
        { "mmvq_esimd_block_load", GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_esimd_slm", GGML_LAYOUT_AOS,        KernelKind::MMVQ },
        { "mmvq_xmx_tile_8x8",     GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_xmx_tile_16x16",   GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_xmx_aos_direct",   GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_xmx_soa_direct",   GGML_LAYOUT_SOA, KernelKind::MMVQ },
        { "mmvq_xmx_double_buffer", GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_esimd_dpas_1x16x32", GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_esimd_dpas_8x16x32", GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_esimd_dpas_chained", GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_xmx_tile_64x64",     GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_xmx_register_accum", GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_xmx_multi_wg",       GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_xmx_persistent",     GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_esimd_large_tile",   GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_esimd_persistent",   GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_esimd_lsc_prefetch", GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_hybrid_adaptive",    GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_xmx_fused",          GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_coalesced_xmx_aligned", GGML_LAYOUT_COALESCED, KernelKind::MMVQ },
        { "mmvq_esimd_hybrid",       GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_esimd_cooperative",  GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_q4_0_specialized",   GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_q6_k_specialized",   GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmvq_mxfp4_native",       GGML_LAYOUT_AOS, KernelKind::MMVQ },
        { "mmq_aos",        GGML_LAYOUT_AOS,       KernelKind::MMQ },
        { "mmq_soa",        GGML_LAYOUT_SOA,       KernelKind::MMQ },
        { "mmq_coalesced",  GGML_LAYOUT_COALESCED, KernelKind::MMQ },
        { "mmq",            GGML_LAYOUT_AOS,       KernelKind::MMQ },
        { "dpas_baseline",           GGML_LAYOUT_AOS, KernelKind::DPAS_EXPLORATION },
        { "dpas_sweep",              GGML_LAYOUT_AOS, KernelKind::DPAS_EXPLORATION },
        { "dpas_memory_patterns",    GGML_LAYOUT_AOS, KernelKind::DPAS_EXPLORATION },
        { "mmvq",           GGML_LAYOUT_AOS,       KernelKind::MMVQ },
        { "onednn_fp16_gemm", GGML_LAYOUT_AOS, KernelKind::ONEDNN_FP16_GEMM },
        { "onednn_int8_gemm", GGML_LAYOUT_AOS, KernelKind::ONEDNN_INT8_GEMM },
        { "onednn_woq_gemm", GGML_LAYOUT_AOS, KernelKind::ONEDNN_WOQ_GEMM },
        { "unified_matmul", GGML_LAYOUT_AOS, KernelKind::UNIFIED_MATMUL },
        { "memory_bandwidth", GGML_LAYOUT_AOS, KernelKind::MEMORY_BANDWIDTH },
        { "roofline_compute", GGML_LAYOUT_AOS, KernelKind::ROOFLINE_COMPUTE },
    };
    return kernels;
}

inline const KernelInfo * find_kernel(std::string_view name) {
    const std::string needle = to_lower(std::string(name));
    for (const auto & kernel : kernel_list()) {
        if (needle == kernel.name) {
            return &kernel;
        }
    }
    return nullptr;
}

inline const char * layout_name(ggml_layout_mode layout) {
    switch (layout) {
        case GGML_LAYOUT_AOS:       return "AOS";
        case GGML_LAYOUT_SOA:       return "SOA";
        case GGML_LAYOUT_COALESCED: return "COALESCED";
        case GGML_LAYOUT_XMX_TILED: return "XMX_TILED";
        case GGML_LAYOUT_XMX_GEMM_TILED: return "XMX_GEMM_TILED";
        default:                    return "UNKNOWN";
    }
}

inline bool kernel_supports_layout(ggml_type type, ggml_layout_mode layout) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q6_K:
            return layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED;
        case GGML_TYPE_Q4_K:
            return layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q5_K:
            return layout == GGML_LAYOUT_AOS;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
            return layout == GGML_LAYOUT_AOS;
        default:
            return false;
    }
}

}  // namespace sycl_bench
