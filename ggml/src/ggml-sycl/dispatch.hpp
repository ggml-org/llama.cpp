//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Unified Kernel Dispatch for SYCL Matmul
//
// This header provides a simplified, data-driven dispatch function that replaces
// the complex mul_mat dispatch logic (~200 lines) with a concise ~20 line implementation.
//
// Key features:
// - Environment variable gate (GGML_SYCL_UNIFIED_KERNEL)
// - Debug tracing (GGML_SYCL_DEBUG >= 1)
// - Integration with TuningEngine for cached/heuristic params
// - Fallback to legacy kernel for unsupported types
//

#pragma once

#include "unified-kernel.hpp"
#include "tuning-engine.hpp"
#include "tuning-engine-impl.hpp"
#include "cold-start.hpp"
#include "op-context.hpp"

#include <cstdlib>
#include <cstring>

namespace ggml_sycl {

// =============================================================================
// Quant Type Support Check
// =============================================================================

/**
 * Check if unified kernel should be used for this quantization type.
 *
 * Unified kernel supports quantized integer types. FP16/BF16 use oneDNN.
 *
 * @param type GGML quantization type
 * @return true if unified kernel supports this type
 */
inline bool should_use_unified(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q4_K:
            return true;
        default:
            return false;  // FP16, BF16, F32, etc. use oneDNN
    }
}

// =============================================================================
// Environment Variable Gates
// =============================================================================

/**
 * Check if unified kernel is enabled via environment variable.
 *
 * Set GGML_SYCL_UNIFIED_KERNEL=0 to disable (default: enabled).
 *
 * @return true if unified kernel is enabled
 */
inline bool is_unified_kernel_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("GGML_SYCL_UNIFIED_KERNEL");
        enabled = (!env || std::strcmp(env, "0") != 0) ? 1 : 0;  // Default: enabled
    }
    return enabled != 0;
}

/**
 * Get debug level from environment variable.
 *
 * @return Debug level (0 = off, 1+ = enabled)
 */
inline int get_debug_level() {
    static int level = -1;
    if (level < 0) {
        const char* env = std::getenv("GGML_SYCL_DEBUG");
        level = (env != nullptr) ? std::atoi(env) : 0;
    }
    return level;
}

// =============================================================================
// Batch Bucket Helper
// =============================================================================

/**
 * Map batch size (M dimension) to a batch bucket for tuning key lookup.
 *
 * @param M Output rows (batch size)
 * @return BatchBucket enum value
 */
inline ggml_sycl_tuning::BatchBucket get_batch_bucket(int64_t M) {
    return ggml_sycl_tuning::bucket_for_batch(static_cast<int>(M));
}

// =============================================================================
// Extended TunedParams with Confidence
// =============================================================================

/**
 * Extended tuned params that includes confidence from the tuning engine.
 * Used for dispatch tracing and decision making.
 */
struct TunedParamsWithConfidence {
    ggml_sycl_tuning::TunedParams params;
    double confidence = 0.0;
};

// =============================================================================
// Debug Tracing
// =============================================================================

/**
 * Trace dispatch decision to stderr.
 *
 * Output format:
 * [mul_mat] M=<M> N=<N> K=<K> type=<type> -> tile=<tm>x<tn> xmx=<yes/no> conf=<pct>% src=<source>
 *
 * @param M Output rows
 * @param N Output columns
 * @param K Reduction dimension
 * @param weight_type Quantization type
 * @param params Tuned parameters
 * @param confidence Confidence score (0.0 to 1.0)
 * @param source Source of params ("CACHED" or "HEURISTIC")
 */
inline void trace_dispatch(int64_t M, int64_t N, int64_t K,
                           ggml_type weight_type,
                           const ggml_sycl_tuning::TunedParams& params,
                           double confidence,
                           const char* source) {
    if (get_debug_level() >= 1) {
        fprintf(stderr, "[mul_mat] M=%lld N=%lld K=%lld type=%d -> tile=%dx%d xmx=%s conf=%d%% src=%s\n",
                static_cast<long long>(M),
                static_cast<long long>(N),
                static_cast<long long>(K),
                static_cast<int>(weight_type),
                static_cast<int>(params.tile_m),
                static_cast<int>(params.tile_n),
                params.use_dpas ? "yes" : "no",
                static_cast<int>(confidence * 100),
                source);
    }
}

// =============================================================================
// Kernel Args Builder
// =============================================================================

/**
 * Build UnifiedKernelArgs from dimensions and tuned parameters.
 *
 * @param M Output rows
 * @param N Output columns
 * @param K Reduction dimension
 * @param weight_type Quantization type
 * @param params Tuned parameters
 * @param weights Quantized weight data (device pointer)
 * @param activations Activation data (device pointer, F32)
 * @param output Output data (device pointer, F32)
 * @return Populated UnifiedKernelArgs struct
 */
inline ggml_sycl_unified::UnifiedKernelArgs build_kernel_args(
    int64_t M, int64_t N, int64_t K,
    ggml_type weight_type,
    const ggml_sycl_tuning::TunedParams& params,
    const void* weights,
    const float* activations,
    float* output)
{
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M = M;
    args.N = N;
    args.K = K;
    args.tile_m = params.tile_m;
    args.tile_n = params.tile_n;
    args.tile_k = params.tile_k;
    args.use_xmx = params.use_dpas;
    args.layout_mode = params.layout_mode;
    args.layout = static_cast<ggml_sycl_unified::LayoutMode>(params.layout_mode);
    args.quant_type = static_cast<int>(weight_type);
    args.prefetch_depth = params.prefetch_depth;
    args.weights = weights;
    args.activations = activations;
    args.output = output;
    return args;
}

// =============================================================================
// Simplified Operation Context
// =============================================================================

/**
 * Simplified operation context for dispatch.
 *
 * This is a lightweight struct that doesn't require GGML tensors,
 * making it suitable for unit testing and standalone dispatch.
 */
struct OperationContext {
    int64_t   M;                // Output rows (batch * tokens)
    int64_t   N;                // Output columns (hidden dim)
    int64_t   K;                // Reduction dimension
    ggml_type weight_type;      // Quantization type
    ggml_type activation_type;  // Activation type (typically F32)
    uint32_t  device_id;        // GPU identifier

    /**
     * Build an operation context from raw dimensions.
     *
     * @param M Output rows
     * @param N Output columns
     * @param K Reduction dimension
     * @param weight_type Quantization type
     * @param activation_type Activation type
     * @param device_id GPU device ID
     * @return Populated OperationContext
     */
    static OperationContext build(int64_t M, int64_t N, int64_t K,
                                   ggml_type weight_type,
                                   ggml_type activation_type,
                                   uint32_t device_id) {
        return OperationContext{
            .M = M,
            .N = N,
            .K = K,
            .weight_type = weight_type,
            .activation_type = activation_type,
            .device_id = device_id
        };
    }
};

// =============================================================================
// Main Unified Dispatch Function (~20 lines of core logic)
// =============================================================================

/**
 * Unified matmul dispatch function.
 *
 * This is the primary entry point for dispatching quantized matrix multiplications
 * through the unified kernel architecture. It:
 *
 * 1. Builds operation context from dimensions
 * 2. Gets tuned parameters (cached or heuristic)
 * 3. Traces the dispatch decision (if debug enabled)
 * 4. Builds kernel args and launches unified matmul
 * 5. Records observation for future tuning
 *
 * Core logic is ~20 lines, replacing 200+ line legacy dispatch.
 *
 * @tparam TuningEngine Type of tuning engine (for testing with mocks)
 * @param queue SYCL queue for kernel submission
 * @param tuning_engine Reference to tuning engine for params/observation
 * @param src0_data Quantized weight data (device pointer)
 * @param src1_data Activation data (device pointer, F32)
 * @param dst_data Output data (device pointer, F32)
 * @param M Output rows
 * @param N Output columns
 * @param K Reduction dimension
 * @param weight_type Quantization type
 * @param device_id GPU device identifier (default: 0)
 */
template<typename TuningEngineT>
inline void ggml_sycl_mul_mat_unified(
    sycl::queue& queue,
    TuningEngineT& tuning_engine,
    const void* src0_data,      // Weights (quantized)
    const float* src1_data,     // Activations
    float* dst_data,            // Output
    int64_t M, int64_t N, int64_t K,
    ggml_type weight_type,
    uint32_t device_id = 0)
{
    // 1. Build tuning key for cache lookup
    ggml_sycl_tuning::TuningKey key{
        static_cast<int32_t>(weight_type),
        get_batch_bucket(M),
        static_cast<int32_t>(K),
        static_cast<int32_t>(N)
    };

    // 2. Get tuned parameters (handles cold-start internally)
    ggml_sycl_tuning::TunedParams params = tuning_engine.get_params(key);
    double confidence = tuning_engine.get_confidence(key);

    // 3. Trace if debugging enabled
    const char* source = (confidence > 0.5) ? "CACHED" : "HEURISTIC";
    trace_dispatch(M, N, K, weight_type, params, confidence, source);

    // 4. Build kernel args and launch
    auto args = build_kernel_args(M, N, K, weight_type, params, src0_data, src1_data, dst_data);
    ggml_sycl_unified::launch_unified_matmul(queue, args);

    // 5. Record observation for tuning (non-blocking)
    tuning_engine.record_observation_async(key, params);

    // Suppress unused parameter warning for device_id (used for future multi-GPU)
    (void)device_id;
}

/**
 * Convenience wrapper for unified dispatch with default tuning engine.
 *
 * Creates a static TuningEngine instance and forwards to the template version.
 * Useful for integration points that don't manage their own tuning engine.
 *
 * @param queue SYCL queue
 * @param src0_data Quantized weight data
 * @param src1_data Activation data
 * @param dst_data Output data
 * @param M Output rows
 * @param N Output columns
 * @param K Reduction dimension
 * @param weight_type Quantization type
 */
inline void ggml_sycl_mul_mat_unified_default(
    sycl::queue& queue,
    const void* src0_data,
    const float* src1_data,
    float* dst_data,
    int64_t M, int64_t N, int64_t K,
    ggml_type weight_type)
{
    // Thread-safe static initialization of default tuning engine
    static ggml_sycl_tuning::TuningEngine default_engine;

    ggml_sycl_mul_mat_unified(queue, default_engine,
                               src0_data, src1_data, dst_data,
                               M, N, K, weight_type, 0);
}

}  // namespace ggml_sycl
