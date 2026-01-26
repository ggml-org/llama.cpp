/**
 * @file op-context.hpp
 * @brief Operation context builder for kernel dispatch and auto-tuning
 *
 * Extracts M, N, K dimensions from GGML tensor shapes following the
 * standard matrix multiplication convention:
 *   dst[M, N] = src1[M, K] @ src0[K, N]
 *
 * GGML Tensor Convention:
 *   ne[0] = innermost dimension (contiguous in memory)
 *   src0 (weights): ne[0]=K, ne[1]=N  (K cols, N rows)
 *   src1 (activations): ne[0]=K, ne[1]=M  (K cols, M rows/batch)
 *   dst: ne[0]=N, ne[1]=M  (N cols, M rows)
 *
 * For batched operations, M may span multiple dimensions (ne[1]*ne[2]*ne[3]).
 */

#ifndef GGML_SYCL_OP_CONTEXT_HPP
#define GGML_SYCL_OP_CONTEXT_HPP

#include "ggml.h"

namespace ggml_sycl {
namespace dispatch {

/**
 * Operation context for kernel dispatch decisions
 *
 * Contains all information needed to select the optimal kernel
 * and memory layout for a matrix multiplication operation.
 */
struct OperationContext {
    int64_t   M;                // Output rows (batch * tokens)
    int64_t   N;                // Output columns (hidden dim / output features)
    int64_t   K;                // Inner dimension (input features / reduction dim)
    ggml_type weight_type;      // src0 type (Q4_0, Q8_0, Q6_K, F16, etc.)
    ggml_type activation_type;  // src1 type (F32, F16)
    int       batch_size;       // Number of tokens (convenience, derived from M)
    uint32_t  device_id;        // GPU identifier for device-specific tuning
    bool      is_contiguous;    // Can use optimized contiguous memory paths
};

/**
 * Build operation context from tensor operands
 *
 * Extracts matrix dimensions following GGML MUL_MAT conventions:
 *   - K = src0->ne[0] = src1->ne[0] (inner/reduction dimension)
 *   - N = src0->ne[1] (output columns, weight rows)
 *   - M = src1->ne[1] * src1->ne[2] * src1->ne[3] (batched output rows)
 *
 * @param src0 Weight tensor [K, N, ...]
 * @param src1 Activation tensor [K, M, ...]
 * @param dst  Output tensor [N, M, ...] (used for contiguity check)
 * @param device_id GPU device identifier
 * @return OperationContext with extracted dimensions and metadata
 *
 * Performance: <1us per call (simple pointer arithmetic, no allocations)
 */
inline OperationContext build_op_context(const ggml_tensor * src0,
                                          const ggml_tensor * src1,
                                          const ggml_tensor * dst,
                                          uint32_t device_id) {
    // Extract dimensions from tensor shapes
    // GGML MUL_MAT: dst = src0 @ src1 (column-major convention)
    //   src0: [K, N, 1, 1] - weights
    //   src1: [K, M, ne2, ne3] - activations (may be batched)
    //   dst:  [N, M, ne2, ne3] - output

    const int64_t K = src0->ne[0];  // Inner dimension (shared between src0 and src1)
    const int64_t N = src0->ne[1];  // Output columns (weight matrix rows)

    // M is the product of all batch dimensions in src1
    // For standard matmul: M = src1->ne[1]
    // For batched matmul: M = src1->ne[1] * src1->ne[2] * src1->ne[3]
    const int64_t M = src1->ne[1] * src1->ne[2] * src1->ne[3];

    // Check contiguity for optimized paths
    // A tensor is contiguous if its elements can be accessed sequentially
    // without gaps (stride matches element size)
    const bool src0_contiguous = ggml_is_contiguous(src0);
    const bool src1_contiguous = ggml_is_contiguous(src1);
    const bool dst_contiguous  = ggml_is_contiguous(dst);
    const bool is_contiguous   = src0_contiguous && src1_contiguous && dst_contiguous;

    return OperationContext{
        .M               = M,
        .N               = N,
        .K               = K,
        .weight_type     = src0->type,
        .activation_type = src1->type,
        .batch_size      = static_cast<int>(M),  // Cast is safe for practical batch sizes
        .device_id       = device_id,
        .is_contiguous   = is_contiguous,
    };
}

/**
 * Check if operation context represents a memory-bound workload
 *
 * Memory-bound operations benefit from optimized memory layouts
 * (SoA, coalesced) rather than compute-heavy XMX kernels.
 *
 * @param ctx Operation context
 * @return true if memory-bound (small batch), false if compute-bound
 */
inline bool is_memory_bound(const OperationContext & ctx) {
    // Empirically determined threshold from benchmark analysis
    // batch <= 4: Memory bandwidth is the bottleneck
    // batch > 4: Compute becomes more significant
    return ctx.batch_size <= 4;
}

/**
 * Check if operation context is suitable for XMX acceleration
 *
 * XMX (Xe Matrix eXtensions) is beneficial for larger batch sizes
 * where the matrix multiply is compute-bound.
 *
 * @param ctx Operation context
 * @return true if XMX is likely beneficial
 */
inline bool is_xmx_candidate(const OperationContext & ctx) {
    // XMX is beneficial when:
    // 1. Batch size is large enough to amortize setup overhead
    // 2. K dimension is divisible by XMX tile K (typically 32)
    // 3. Weight type is supported (Q4_0, Q8_0 with dequant, or native INT8)
    const bool batch_large_enough = ctx.batch_size >= 8;
    const bool k_aligned          = (ctx.K % 32) == 0;

    return batch_large_enough && k_aligned;
}

/**
 * Get the compute intensity ratio (FLOPs / bytes)
 *
 * Higher values indicate compute-bound operations that benefit
 * from XMX acceleration. Lower values are memory-bound.
 *
 * @param ctx Operation context
 * @return Approximate compute intensity (FLOPs per byte transferred)
 */
inline float compute_intensity(const OperationContext & ctx) {
    // FLOPs for matmul: 2 * M * N * K (multiply + add)
    const float flops = 2.0f * static_cast<float>(ctx.M) *
                        static_cast<float>(ctx.N) *
                        static_cast<float>(ctx.K);

    // Bytes transferred (approximate, ignoring caching):
    // - src0 (weights): N * K * type_size
    // - src1 (activations): M * K * 4 (assuming F32)
    // - dst: M * N * 4 (assuming F32 output)
    const float weight_bytes = static_cast<float>(ctx.N * ctx.K) *
                               static_cast<float>(ggml_type_size(ctx.weight_type)) /
                               static_cast<float>(ggml_blck_size(ctx.weight_type));
    const float activation_bytes = static_cast<float>(ctx.M * ctx.K) *
                                   static_cast<float>(ggml_type_size(ctx.activation_type)) /
                                   static_cast<float>(ggml_blck_size(ctx.activation_type));
    const float output_bytes = static_cast<float>(ctx.M * ctx.N) * sizeof(float);

    const float total_bytes = weight_bytes + activation_bytes + output_bytes;

    return (total_bytes > 0.0f) ? (flops / total_bytes) : 0.0f;
}

}  // namespace dispatch
}  // namespace ggml_sycl

#endif  // GGML_SYCL_OP_CONTEXT_HPP
