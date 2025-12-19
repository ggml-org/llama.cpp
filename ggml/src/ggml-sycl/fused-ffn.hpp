//
// Fused FFN (gate_proj + up_proj + SwiGLU) for Intel Arc GPUs
// Eliminates intermediate gate/up tensors by fusing matmuls with activation
//
// Pattern: gate = x @ W_gate, up = x @ W_up, out = silu(gate) * up
// Fused:   out[i] = silu(dot(x, W_gate[:, i])) * dot(x, W_up[:, i])
//

#ifndef GGML_SYCL_FUSED_FFN_HPP
#define GGML_SYCL_FUSED_FFN_HPP

#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"
#include <sycl/sycl.hpp>

// Define to output raw gate_sum/up_sum for debugging (instead of SwiGLU)
// #define GGML_SYCL_FFN_DEBUG_RAW

// Define to enable detailed multi-row kernel debug output
// #define GGML_SYCL_FFN_MULTIROW_DEBUG

namespace ggml_sycl_ffn {

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
template<typename T>
__dpct_inline__ T silu(T x) {
    return x / (T(1.0f) + sycl::exp(-x));
}

// =============================================================================
// Configuration for multi-row kernel
// =============================================================================

// Multi-row kernel processes multiple output rows per work-group
// This amortizes input load cost across many rows
constexpr int FFN_ROWS_PER_WG = 64;      // Output rows per work-group
constexpr int FFN_THREADS_PER_WG = 256;  // 16 warps × 16 threads
constexpr int FFN_WARPS_PER_WG = FFN_THREADS_PER_WG / WARP_SIZE;  // 16

// =============================================================================
// Multi-row Fused Gate+Up+SwiGLU kernel with SLM caching
// Each work-group computes FFN_ROWS_PER_WG output elements, caching input in SLM
// =============================================================================

// Multi-row fused kernel with SLM input caching
// Key optimization: Input x is loaded once to SLM, reused for all output rows
// This reduces global memory traffic by ~64x for input reads
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void fused_ffn_multirow_kernel(
    const void * __restrict__ vx_gate,     // W_gate [nrows_out, ncols_in] Q4_0
    const void * __restrict__ vx_up,       // W_up [nrows_out, ncols_in] Q4_0
    const void * __restrict__ vy,          // x [batch_size, ncols_in] Q8_1
    float * __restrict__ dst,              // output [batch_size, nrows_out]
    const int ncols_in,                    // input dimension (4096)
    const int nrows_out,                   // output dimension (14336)
    const int batch_size,                  // number of tokens
    const sycl::nd_item<3> & item_ct1,
    block_q8_1 * slm_x                     // SLM for input caching
) {
    const int wg_id = item_ct1.get_group(2);
    const int tid = item_ct1.get_local_linear_id();
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int blocks_per_row = ncols_in / qk;  // e.g., 4096/32 = 128
    const int blocks_per_input = ncols_in / QK8_1;  // Q8_1 blocks for input
    const int row_base = wg_id * FFN_ROWS_PER_WG;

    const block_q_t * x_gate = (const block_q_t *) vx_gate;
    const block_q_t * x_up = (const block_q_t *) vx_up;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    // Process each token in the batch
    for (int token = 0; token < batch_size; token++) {
        // Phase 1: Cooperatively load input x to SLM
        // 256 threads load blocks_per_input blocks
        const block_q8_1 * y_token = y + token * blocks_per_input;
        for (int b = tid; b < blocks_per_input; b += FFN_THREADS_PER_WG) {
            slm_x[b] = y_token[b];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

#ifdef GGML_SYCL_FFN_MULTIROW_DEBUG
        // Debug: Print first few SLM values from thread 0
        if (wg_id == 0 && tid == 0 && token == 0) {
            // Print first block's scale and first 4 qs values
            sycl::ext::oneapi::experimental::printf(
                "[MULTIROW DEBUG] SLM block 0: ds.x=%f ds.y=%f qs[0-3]=%d,%d,%d,%d\n",
                (float)slm_x[0].ds.x(), (float)slm_x[0].ds.y(),
                (int)slm_x[0].qs[0], (int)slm_x[0].qs[1],
                (int)slm_x[0].qs[2], (int)slm_x[0].qs[3]);
        }
#endif

        // Phase 2: Each warp processes FFN_ROWS_PER_WG / FFN_WARPS_PER_WG rows
        constexpr int rows_per_warp = FFN_ROWS_PER_WG / FFN_WARPS_PER_WG;  // 4

        for (int r = 0; r < rows_per_warp; r++) {
            const int out_row = row_base + warp_id * rows_per_warp + r;
            if (out_row >= nrows_out) continue;

            float gate_sum = 0.0f;
            float up_sum = 0.0f;

            // Compute dot products using SLM-cached input
            // Using same vec_dot pattern as original kernel
            constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

            for (int i = lane_id / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
                // Weight block index: row * blocks_per_row + block_in_row
                const int ibx = out_row * blocks_per_row + i;

                // Input block index (from SLM), aligned with weight blocks
                const int iby = i * (qk / QK8_1);  // For Q4_0: qk=32, QK8_1=32, so ratio=1

                for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
                    const int iqs = elem + vdr * (lane_id % (qi / vdr));

                    // Use SLM-cached input instead of global memory
                    gate_sum += vec_dot_q_sycl(&x_gate[ibx], &slm_x[iby], iqs);
                    up_sum += vec_dot_q_sycl(&x_up[ibx], &slm_x[iby], iqs);
                }
            }

            // Warp-level reduction for both sums
            #pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
                gate_sum += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), gate_sum, mask);
                up_sum += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), up_sum, mask);
            }

            // Apply SwiGLU: silu(gate) * up and write result
            if (lane_id == 0) {
                float result = silu(gate_sum) * up_sum;
#ifdef GGML_SYCL_FFN_MULTIROW_DEBUG
                // Debug: Print first few output values
                if (token == 0 && out_row < 4) {
                    sycl::ext::oneapi::experimental::printf(
                        "[MULTIROW DEBUG] out_row=%d gate_sum=%f up_sum=%f silu=%f result=%f\n",
                        out_row, gate_sum, up_sum, silu(gate_sum), result);
                }
#endif
                dst[token * nrows_out + out_row] = result;
            }
        }

        // Barrier before loading next token's input
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

// =============================================================================
// Host launcher for multi-row fused FFN kernel
// =============================================================================

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void fused_ffn_multirow_sycl(
    const void * vx_gate,      // W_gate weights [nrows_out, ncols_in]
    const void * vx_up,        // W_up weights [nrows_out, ncols_in]
    const void * vy,           // input (Q8_1) [batch_size, ncols_in]
    float * dst,               // output [batch_size, nrows_out]
    const int ncols_in,        // input dimension (4096)
    const int nrows_out,       // output dimension (14336)
    const int batch_size,      // number of tokens
    dpct::queue_ptr stream
) {
    GGML_ASSERT(ncols_in % qk == 0);

    const int blocks_per_input = ncols_in / QK8_1;

    // Launch ceil(nrows_out / FFN_ROWS_PER_WG) work-groups
    const int num_wgs = (nrows_out + FFN_ROWS_PER_WG - 1) / FFN_ROWS_PER_WG;
    const sycl::range<3> block_dims(1, 1, FFN_THREADS_PER_WG);
    const sycl::range<3> grid_dims(1, 1, num_wgs);

    stream->submit([&](sycl::handler & cgh) {
        // Allocate SLM for input caching
        // Q8_1 block: 32 int8 + half2 = 36 bytes, 128 blocks = 4608 bytes
        sycl::local_accessor<block_q8_1, 1> slm_x(sycl::range<1>(blocks_per_input), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_ffn_multirow_kernel<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                    vx_gate, vx_up, vy, dst, ncols_in, nrows_out, batch_size,
                    item_ct1, slm_x.get_pointer());
            });
    });
}

// =============================================================================
// Original single-row Fused Gate+Up+SwiGLU kernel for Q4_0 weights
// Each work-group computes one output element (kept as fallback)
// =============================================================================

// Fused kernel for gate_proj + up_proj + SwiGLU
// Each work-group computes one output element (output row)
// Weights layout: [nrows_out, ncols_in] where:
//   - nrows_out = intermediate_size (14336 for Mistral)
//   - ncols_in = hidden_size (4096 for Mistral)
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void fused_ffn_gate_up_swiglu_kernel(
    const void * __restrict__ vx_gate,     // W_gate [nrows_out, ncols_in] Q4_0
    const void * __restrict__ vx_up,       // W_up [nrows_out, ncols_in] Q4_0
    const void * __restrict__ vy,          // x [batch_size, ncols_in] Q8_1
    float * __restrict__ dst,              // output [batch_size, nrows_out]
    const int ncols_in,                    // input dimension (4096)
    const int nrows_out,                   // output dimension (14336)
    const int batch_size,                  // number of tokens
    const sycl::nd_item<3> & item_ct1
) {
    // Each work-group handles one output row (one element of intermediate dim)
    const int out_row = item_ct1.get_group(2);

    if (out_row >= nrows_out) {
        return;
    }

    const int blocks_per_weight_row = ncols_in / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    const block_q_t * x_gate = (const block_q_t *) vx_gate;
    const block_q_t * x_up = (const block_q_t *) vx_up;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    // Process each token in the batch
    for (int token = 0; token < batch_size; token++) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;

        // Compute dot products for gate and up projections
        // Iterating over weight blocks in this row
        for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_weight_row; i += blocks_per_warp) {
            // Weight block index: row * blocks_per_row + block_in_row
            const int ibx = out_row * blocks_per_weight_row + i;

            // Input block index for this token, aligned with weight blocks
            const int iby = token * (ncols_in / QK8_1) + i * (qk / QK8_1);

            for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
                const int iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));

                // Compute both gate and up dot products simultaneously
                gate_sum += vec_dot_q_sycl(&x_gate[ibx], &y[iby], iqs);
                up_sum += vec_dot_q_sycl(&x_up[ibx], &y[iby], iqs);
            }
        }

        // Warp-level reduction for both sums
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            gate_sum += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), gate_sum, mask);
            up_sum += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), up_sum, mask);
        }

        // Apply SwiGLU: silu(gate) * up and write result
        // DEBUG: Output gate_sum at row 0, up_sum at row 1 (both for row 0's weights)
        if (item_ct1.get_local_id(2) == 0) {
#ifdef GGML_SYCL_FFN_DEBUG_RAW
            if (out_row == 0) {
                // Row 0 WG outputs both gate and up for row 0 at slots 0 and 1
                dst[token * nrows_out + 0] = gate_sum;
                dst[token * nrows_out + 1] = up_sum;
            } else if (out_row >= 2) {
                dst[token * nrows_out + out_row] = silu(gate_sum) * up_sum;
            }
            // Row 1 WG does nothing (slot 1 used by row 0 for up_sum)
#else
            dst[token * nrows_out + out_row] = silu(gate_sum) * up_sum;
#endif
        }
    }
}

// =============================================================================
// Host launcher for fused FFN kernel
// =============================================================================

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void fused_ffn_gate_up_swiglu_sycl(
    const void * vx_gate,      // W_gate weights [nrows_out, ncols_in]
    const void * vx_up,        // W_up weights [nrows_out, ncols_in]
    const void * vy,           // input (Q8_1) [batch_size, ncols_in]
    float * dst,               // output [batch_size, nrows_out]
    const int ncols_in,        // input dimension (4096)
    const int nrows_out,       // output dimension (14336)
    const int batch_size,      // number of tokens
    dpct::queue_ptr stream
) {
    GGML_ASSERT(ncols_in % qk == 0);

    // Launch one work-group per output element (nrows_out work-groups)
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> grid_dims(1, 1, nrows_out);

    stream->parallel_for(
        sycl::nd_range<3>(grid_dims * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            fused_ffn_gate_up_swiglu_kernel<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                vx_gate, vx_up, vy, dst, ncols_in, nrows_out, batch_size, item_ct1);
        });
}

// =============================================================================
// Dispatcher for different quantization types
// =============================================================================

// Try to use fused FFN kernel if conditions are met
// Returns true if fusion was applied, false to fall back to separate kernels
static bool try_fused_ffn_gate_up_swiglu(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * gate_weight,   // W_gate [nrows_out, ncols_in]
    const ggml_tensor * up_weight,     // W_up [nrows_out, ncols_in]
    const void * input_q8,             // x quantized to Q8_1 [batch_size, ncols_in]
    ggml_tensor * dst,                 // output [batch_size, nrows_out]
    const int batch_size,              // number of tokens
    dpct::queue_ptr stream
) {
    // Only support Q4_0 weights for now
    if (gate_weight->type != GGML_TYPE_Q4_0 || up_weight->type != GGML_TYPE_Q4_0) {
        return false;
    }

    // Verify shapes match between gate and up
    if (gate_weight->ne[0] != up_weight->ne[0] ||
        gate_weight->ne[1] != up_weight->ne[1]) {
        return false;
    }

    // Only support small batch sizes (token generation)
    // For large batches, separate kernels may be more efficient
    if (batch_size > 8) {
        return false;
    }

    // Weight dimensions:
    // ne[0] = ncols_in (input features, 4096 for Mistral)
    // ne[1] = nrows_out (output features, 14336 for Mistral)
    const int ncols_in = gate_weight->ne[0];
    const int nrows_out = gate_weight->ne[1];

    // Get device pointers
    const void * gate_data = gate_weight->data;
    const void * up_data = up_weight->data;
    float * dst_data = (float *) dst->data;

    // Use multi-row kernel by default (better GPU utilization via SLM caching)
    // Falls back to single-row kernel if GGML_SYCL_FFN_SINGLE_ROW=1
    static bool use_single_row = getenv("GGML_SYCL_FFN_SINGLE_ROW") != nullptr;

    if (use_single_row) {
        // Original single-row kernel (16 threads/WG, no SLM)
        fused_ffn_gate_up_swiglu_sycl<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
            gate_data, up_data, input_q8, dst_data, ncols_in, nrows_out, batch_size, stream);
    } else {
        // Multi-row kernel (256 threads/WG, SLM caching, 64 rows/WG)
        fused_ffn_multirow_sycl<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
            gate_data, up_data, input_q8, dst_data, ncols_in, nrows_out, batch_size, stream);
    }

    return true;
}

} // namespace ggml_sycl_ffn

#endif // GGML_SYCL_FUSED_FFN_HPP
