#pragma once

#include <string>

#include "../tier1/mmvq_tier1.hpp"
#include "../tier2/mmvq_tier2.hpp"
#include "../tier3/mmvq_tier3.hpp"

namespace sycl_bench {
namespace mmvq_tier4 {

static inline bool validate_stream(const ggml_sycl::mmvq_bench_args & args,
                                   std::string & error) {
    if (args.stream == nullptr) {
        error = "SYCL stream is null.";
        return false;
    }
    return true;
}

inline bool launch_hybrid_adaptive(const ggml_sycl::mmvq_bench_args & args,
                                   std::vector<sycl::event> * events,
                                   std::string & error) {
    constexpr int MMVQ_THRESHOLD = 4;
    constexpr int XMX_THRESHOLD = 32;

    if (!validate_stream(args, error)) {
        return false;
    }

    if (args.layout == GGML_LAYOUT_COALESCED) {
        return run_mmvq_coalesced(args, events, error);
    }
    if (args.layout == GGML_LAYOUT_SOA) {
        if (args.batch <= MMVQ_THRESHOLD) {
            return run_mmvq_soa_baseline(args, events, error);
        }
        if (args.batch <= XMX_THRESHOLD) {
            return run_mmvq_xmx_soa_direct(args, events, error);
        }
        return run_mmvq_soa_baseline(args, events, error);
    }

    if (args.batch <= MMVQ_THRESHOLD) {
        return run_mmvq_wide_load(args, events, error);
    }
    if (args.batch <= XMX_THRESHOLD) {
        return run_mmvq_xmx_tile_16x16(args, events, error);
    }
    return run_mmvq_xmx_tile_64x64(args, events, error);
}

inline bool launch_xmx_fused(const ggml_sycl::mmvq_bench_args & args,
                             std::vector<sycl::event> * events,
                             std::string & error) {
    if (!validate_stream(args, error)) {
        return false;
    }
    if (args.layout != GGML_LAYOUT_AOS) {
        error = "MMVQ-XMX-Fused requires AOS layout.";
        return false;
    }
    return run_mmvq_xmx_tile_16x16(args, events, error);
}

inline bool launch_coalesced_xmx_aligned(const ggml_sycl::mmvq_bench_args & args,
                                         std::vector<sycl::event> * events,
                                         std::string & error) {
    if (!validate_stream(args, error)) {
        return false;
    }
    if (args.layout != GGML_LAYOUT_COALESCED) {
        error = "Coalesced-XMX-Aligned requires COALESCED layout.";
        return false;
    }
    return run_mmvq_coalesced(args, events, error);
}

inline bool launch_esimd_hybrid(const ggml_sycl::mmvq_bench_args & args,
                                std::vector<sycl::event> * events,
                                std::string & error) {
    constexpr int MMVQ_THRESHOLD = 4;

    if (!validate_stream(args, error)) {
        return false;
    }
    if (args.layout != GGML_LAYOUT_AOS) {
        error = "ESIMD-Hybrid requires AOS layout.";
        return false;
    }
    if (args.batch <= MMVQ_THRESHOLD) {
        return run_mmvq_esimd_dpas_1x16x32(args, events, error);
    }
    return run_mmvq_esimd_large_tile(args, events, error);
}

inline bool launch_esimd_cooperative(const ggml_sycl::mmvq_bench_args & args,
                                     std::vector<sycl::event> * events,
                                     std::string & error) {
    if (!validate_stream(args, error)) {
        return false;
    }
    if (args.layout != GGML_LAYOUT_AOS) {
        error = "ESIMD-Cooperative requires AOS layout.";
        return false;
    }
    return run_mmvq_esimd_persistent(args, events, error);
}

inline bool launch_q4_0_specialized(const ggml_sycl::mmvq_bench_args & args,
                                    std::vector<sycl::event> * events,
                                    std::string & error) {
    if (!validate_stream(args, error)) {
        return false;
    }
    if (args.layout != GGML_LAYOUT_AOS) {
        error = "Q4_0 specialized kernel requires AOS layout.";
        return false;
    }
    if (args.weight_type != GGML_TYPE_Q4_0) {
        error = "Q4_0 specialized kernel only supports Q4_0.";
        return false;
    }
    return run_mmvq_xmx_tile_64x64(args, events, error);
}

inline bool launch_q6_k_specialized(const ggml_sycl::mmvq_bench_args & args,
                                    std::vector<sycl::event> * events,
                                    std::string & error) {
    if (!validate_stream(args, error)) {
        return false;
    }
    if (args.layout != GGML_LAYOUT_AOS) {
        error = "Q6_K specialized kernel requires AOS layout.";
        return false;
    }
    if (args.weight_type != GGML_TYPE_Q6_K) {
        error = "Q6_K specialized kernel only supports Q6_K.";
        return false;
    }
    return run_mmvq_xmx_tile_64x64(args, events, error);
}

inline bool launch_mxfp4_native(const ggml_sycl::mmvq_bench_args & args,
                                std::vector<sycl::event> * events,
                                std::string & error) {
    if (!validate_stream(args, error)) {
        return false;
    }
    if (args.layout != GGML_LAYOUT_AOS) {
        error = "MXFP4 native kernel requires AOS layout.";
        return false;
    }
    if (args.weight_type != GGML_TYPE_MXFP4) {
        error = "MXFP4 native kernel only supports MXFP4.";
        return false;
    }
    return run_mmvq_xmx_tile_64x64(args, events, error);
}

}  // namespace mmvq_tier4
}  // namespace sycl_bench
