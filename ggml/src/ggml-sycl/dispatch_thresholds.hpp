/**
 * @file dispatch_thresholds.hpp
 * @brief Data-driven kernel dispatch thresholds for Intel Arc GPUs
 *
 * Auto-generated from benchmark analysis (2026-01-23)
 * Source: tools/sycl-kernel-bench/analysis/
 *
 * These thresholds define batch size boundaries for kernel selection:
 * - batch <= MMVQ_MAX_BATCH: Use memory-bound MMVQ kernels (coalesced/SoA)
 * - MMVQ_MAX_BATCH < batch <= XMX_SMALL_MAX_BATCH: Use XMX small tile kernels
 * - batch > XMX_SMALL_MAX_BATCH: Use XMX large tile or MMQ kernels
 *
 * Benchmark hardware: Intel Arc B580 (level_zero:0/1)
 * Benchmark configuration: dim=4096, iterations=100, warmup=10
 */

#ifndef GGML_SYCL_DISPATCH_THRESHOLDS_HPP
#define GGML_SYCL_DISPATCH_THRESHOLDS_HPP

#include "ggml.h"

namespace ggml_sycl {
namespace dispatch {

/**
 * Kernel type enumeration for dispatch decisions
 */
enum class kernel_type {
    DMMV,           // Dense matrix-vector (batch=1 only)
    MMVQ_COALESCED, // Memory-bound, coalesced layout (batch=1-4)
    MMVQ_SOA,       // Memory-bound, SoA layout (batch=1-4, some quants)
    XMX_SMALL_TILE, // XMX 8x8 or 16x16 tiles (batch=8-32)
    XMX_LARGE_TILE, // XMX 64x64 tiles (batch=64+)
    MMQ,            // MMQ kernel (large batch, some quants)
};

/**
 * Dispatch thresholds per quantization type
 *
 * Values derived from benchmark sweep on Intel Arc B580:
 * - Winner detection: max throughput_tps per (quant, batch, dim)
 * - Crossover analysis: where XMX starts outperforming MMVQ
 */
template<ggml_type QUANT>
struct thresholds {
    // Default thresholds (conservative)
    static constexpr int MMVQ_MAX_BATCH = 4;
    static constexpr int XMX_SMALL_MAX_BATCH = 32;

    // Preferred layout for memory-bound regime
    static constexpr bool PREFER_COALESCED = true;
};

// Q4_0: Most common quantization
// Winner at batch=1-4: mmvq_coalesced
// Winner at batch=8-32: mmvq_xmx variants (esimd_v2, xmx_tile_soa)
// Winner at batch=64+: mmvq_xmx_cf2_soa, mmq_aos at batch=128
template<>
struct thresholds<GGML_TYPE_Q4_0> {
    static constexpr int MMVQ_MAX_BATCH = 4;
    static constexpr int XMX_SMALL_MAX_BATCH = 32;
    static constexpr bool PREFER_COALESCED = true;
};

// Q8_0: Native INT8 XMX support
// Winner at batch=1-4: mmvq_coalesced
// Winner at batch=8-32: mmvq_xmx_tile_soa, mmvq_xmx_cf2_soa
// Winner at batch=64+: mmvq_xmx_cf2_soa, mmvq_xmx_register_accum
template<>
struct thresholds<GGML_TYPE_Q8_0> {
    static constexpr int MMVQ_MAX_BATCH = 4;
    static constexpr int XMX_SMALL_MAX_BATCH = 32;
    static constexpr bool PREFER_COALESCED = true;
};

// Q6_K: K-quant, different memory patterns
// Winner at batch=1-4: mmvq_soa_baseline
// Winner at batch=8+: mmq_soa, mmq_coalesced (MMQ wins earlier)
template<>
struct thresholds<GGML_TYPE_Q6_K> {
    static constexpr int MMVQ_MAX_BATCH = 4;
    static constexpr int XMX_SMALL_MAX_BATCH = 16;  // MMQ wins at batch>=16
    static constexpr bool PREFER_COALESCED = false; // SoA preferred
};

// Q4_K: K-quant variant
// Winner at batch=1: mmvq_soa_baseline
// Winner at batch=8+: mmvq_xmx_double_buffer, mmq_aos at batch=128
template<>
struct thresholds<GGML_TYPE_Q4_K> {
    static constexpr int MMVQ_MAX_BATCH = 4;
    static constexpr int XMX_SMALL_MAX_BATCH = 32;
    static constexpr bool PREFER_COALESCED = false; // SoA preferred
};

// Q2_K, Q3_K, Q5_K: Less common K-quants
template<>
struct thresholds<GGML_TYPE_Q2_K> {
    static constexpr int MMVQ_MAX_BATCH = 4;
    static constexpr int XMX_SMALL_MAX_BATCH = 32;
    static constexpr bool PREFER_COALESCED = false;
};

template<>
struct thresholds<GGML_TYPE_Q3_K> {
    static constexpr int MMVQ_MAX_BATCH = 4;
    static constexpr int XMX_SMALL_MAX_BATCH = 32;
    static constexpr bool PREFER_COALESCED = false;
};

template<>
struct thresholds<GGML_TYPE_Q5_K> {
    static constexpr int MMVQ_MAX_BATCH = 4;
    static constexpr int XMX_SMALL_MAX_BATCH = 32;
    static constexpr bool PREFER_COALESCED = false;
};

/**
 * Select optimal kernel type based on batch size and quant type
 *
 * @param quant Quantization type
 * @param batch_size Number of tokens in batch
 * @return Recommended kernel type
 */
inline kernel_type select_kernel(ggml_type quant, int batch_size) {
    // Batch=1 always uses DMMV (fastest for single token)
    if (batch_size == 1) {
        return kernel_type::DMMV;
    }

    // Get thresholds for this quant type
    int mmvq_max = 4;
    int xmx_small_max = 32;

    switch (quant) {
        case GGML_TYPE_Q4_0:
            mmvq_max = thresholds<GGML_TYPE_Q4_0>::MMVQ_MAX_BATCH;
            xmx_small_max = thresholds<GGML_TYPE_Q4_0>::XMX_SMALL_MAX_BATCH;
            break;
        case GGML_TYPE_Q8_0:
            mmvq_max = thresholds<GGML_TYPE_Q8_0>::MMVQ_MAX_BATCH;
            xmx_small_max = thresholds<GGML_TYPE_Q8_0>::XMX_SMALL_MAX_BATCH;
            break;
        case GGML_TYPE_Q6_K:
            mmvq_max = thresholds<GGML_TYPE_Q6_K>::MMVQ_MAX_BATCH;
            xmx_small_max = thresholds<GGML_TYPE_Q6_K>::XMX_SMALL_MAX_BATCH;
            break;
        case GGML_TYPE_Q4_K:
            mmvq_max = thresholds<GGML_TYPE_Q4_K>::MMVQ_MAX_BATCH;
            xmx_small_max = thresholds<GGML_TYPE_Q4_K>::XMX_SMALL_MAX_BATCH;
            break;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q5_K:
            mmvq_max = 4;
            xmx_small_max = 32;
            break;
        default:
            // Use conservative defaults for unknown types
            break;
    }

    // Apply thresholds
    if (batch_size <= mmvq_max) {
        // Check layout preference
        bool prefer_coalesced = true;
        switch (quant) {
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q5_K:
                prefer_coalesced = false;
                break;
            default:
                break;
        }
        return prefer_coalesced ? kernel_type::MMVQ_COALESCED : kernel_type::MMVQ_SOA;
    }

    if (batch_size <= xmx_small_max) {
        return kernel_type::XMX_SMALL_TILE;
    }

    // For very large batches, MMQ may be better for some quants
    if (quant == GGML_TYPE_Q6_K && batch_size >= 16) {
        return kernel_type::MMQ;
    }

    return kernel_type::XMX_LARGE_TILE;
}

/**
 * Check if layout conversion is beneficial for the given configuration
 *
 * @param quant Quantization type
 * @param batch_size Number of tokens in batch
 * @return true if layout conversion (AoS -> SoA/Coalesced) is recommended
 */
inline bool should_convert_layout(ggml_type quant, int batch_size) {
    // Layout conversion is beneficial for memory-bound kernels
    if (batch_size <= 4) {
        switch (quant) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
                return true;  // Coalesced layout wins
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q4_K:
                return true;  // SoA layout wins
            default:
                return false; // AoS baseline may be fine
        }
    }

    // XMX kernels prefer SoA or tiled layouts
    return true;
}

} // namespace dispatch
} // namespace ggml_sycl

#endif // GGML_SYCL_DISPATCH_THRESHOLDS_HPP
