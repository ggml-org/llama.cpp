//
// Fused RMSNorm + MUL (gamma) + GEMM for Intel Arc GPUs
// Eliminates intermediate normalized tensor by fusing normalization into quantization
//
// Pattern: x → RMSNorm(x) → x * gamma → quantize → GEMM
// Fused:   x → (compute scales) → quantize_with_norm(x, scales, gamma) → GEMM
//

#ifndef GGML_SYCL_FUSED_NORM_GEMM_HPP
#define GGML_SYCL_FUSED_NORM_GEMM_HPP

#include "common.hpp"
#include "quantize.hpp"
#include <sycl/sycl.hpp>

// =============================================================================
// Phase 1: Compute RMS scales for all rows
// Output: scales[row] = rsqrt(mean(x[row]^2) + eps)
// =============================================================================

// Kernel to compute RMS scales using SLM for reduction
// One work-group per row, uses SLM for partial sum accumulation
static void compute_rms_scales_kernel(
    const float * __restrict__ x,      // [nrows, ncols] input
    float * __restrict__ scales,       // [nrows] output scales
    const int ncols,
    const float eps,
    const sycl::nd_item<3> & item,
    float * slm_partial_sums,          // [nwarps] SLM for reduction
    int block_size
) {
    const int row = item.get_group(2);
    const int tid = item.get_local_id(2);
    const int nwarps = block_size / WARP_SIZE;

    const float * row_ptr = x + row * ncols;

    // Each thread computes partial sum of squares
    float partial_sum = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float val = row_ptr[col];
        partial_sum += val * val;
    }

    // Warp-level reduction
    partial_sum = warp_reduce_sum(partial_sum, item);

    // Cross-warp reduction using SLM
    if (block_size > WARP_SIZE) {
        const auto sub_group = item.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();

        if (wi_in_sg == 0) {
            slm_partial_sums[sg_id] = partial_sum;
        }
        item.barrier(sycl::access::fence_space::local_space);

        partial_sum = 0.f;
        const size_t nreduce = (nwarps + WARP_SIZE - 1) / WARP_SIZE;
        for (size_t i = 0; i < nreduce; i++) {
            if (wi_in_sg + i * WARP_SIZE < (size_t)nwarps) {
                partial_sum += slm_partial_sums[wi_in_sg + i * WARP_SIZE];
            }
        }
        partial_sum = warp_reduce_sum(partial_sum, item);
    }

    // Compute and store final scale
    if (tid == 0) {
        float mean_sq = partial_sum / ncols;
        scales[row] = sycl::rsqrt(mean_sq + eps);
    }
}

// Host function to launch scale computation
static void compute_rms_scales_sycl(
    const float * x,
    float * scales,
    const int nrows,
    const int ncols,
    const float eps,
    queue_ptr stream,
    int device
) {
    const sycl::range<3> global_dims(1, 1, nrows);

    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    compute_rms_scales_kernel(x, scales, ncols, eps, item_ct1, nullptr, WARP_SIZE);
                });
        });
    } else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> slm_acc(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    compute_rms_scales_kernel(x, scales, ncols, eps, item_ct1,
                                              get_pointer(slm_acc), work_group_size);
                });
        });
    }
}

// =============================================================================
// Phase 2: Fused quantization with RMSNorm
// Applies: quantize(x * scale * gamma) where scale is pre-computed RMS scale
// =============================================================================

template <int ElementsPerWI>
struct quantize_with_rmsnorm_q8_1 {
    __dpct_inline__ void operator()(
        const float * __restrict__ x,          // [rows, cols] input (not normalized)
        const float * __restrict__ gamma,      // [cols] RMSNorm weight
        const float * __restrict__ rms_scales, // [rows] pre-computed RMS scales
        void * q8_tensor,
        const int kx,
        const int kx_padded,
        const sycl::nd_item<1> & it
    ) const {
        auto subgroup_id = it.get_group(0);
        auto wi_id = it.get_local_id(0);

        const int num_blocks_per_row = kx / QK8_1;
        auto row = subgroup_id / num_blocks_per_row;
        auto col_block = subgroup_id % num_blocks_per_row;
        const int pitch = kx_padded / QK8_1;

        // Get the pre-computed RMS scale for this row
        float rms_scale = rms_scales[row];

        // Load input values and gamma
        auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
        auto gamma_offset = col_block * QK8_1 + ElementsPerWI * wi_id;

        sycl::vec<float, ElementsPerWI> wi_f32_vals;
        wi_f32_vals = *reinterpret_cast<const sycl::vec<float, ElementsPerWI> *>(x + float_ptr_offset);

        // Apply RMSNorm inline: val = x * rms_scale * gamma
        float amax = 0.0f;
        float sum = 0.0f;
        sycl::vec<int8_t, ElementsPerWI> quantized_values;

        #pragma unroll
        for (int i = 0; i < ElementsPerWI; i++) {
            // Fused normalization: x * scale * gamma
            float val = wi_f32_vals[i] * rms_scale * gamma[gamma_offset + i];
            sum += val;
            amax = sycl::fmax(amax, sycl::fabs(val));
            wi_f32_vals[i] = val;  // Store normalized value for quantization
            quantized_values[i] = 0;
        }

        // Warp-level reduction for quantization scale
        sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
        amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
        float d = amax == 0 ? 1 : amax / 127;

        // Quantize normalized values
        #pragma unroll
        for (int i = 0; i < ElementsPerWI; i++) {
            quantized_values[i] = sycl::round(wi_f32_vals[i] / d);
        }
        d = amax == 0 ? 0 : d;

        // Write quantized output (same layout as regular quantize_q8_1)
        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto block_id = col_block + row * pitch;

        int8_t * qs = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_values;
        if (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

// Host function to launch fused quantization with RMSNorm
template <template <int> typename quantize_f>
void quantize_with_rmsnorm_q8_1_sycl(
    const float * x,
    const float * gamma,
    const float * rms_scales,
    void * vy,
    const int kx,
    const int ky,
    const int kx_padded,
    dpct::queue_ptr stream
) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range = std::size_t(WARP_SIZE);
    auto num_quant_blocks = ky * (kx / QK8_1);
    auto global_range = num_quant_blocks * local_range;

    stream->parallel_for(
        sycl::nd_range<1>({ global_range }, { local_range }),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            quantize_f<QK8_1 / WARP_SIZE>()(x, gamma, rms_scales, vy, kx, kx_padded, it);
        });
}

// =============================================================================
// Full fused operation: RMSNorm + MUL (gamma) + Quantize
// Eliminates intermediate normalized tensor
// =============================================================================

// Fused RMSNorm quantization that combines scale computation and quantization
// Uses SLM to cache RMS scales computed in first phase
static void fused_rmsnorm_quantize_q8_1_sycl(
    const float * x,           // [nrows, ncols] input
    const float * gamma,       // [ncols] RMSNorm weight
    void * q8_output,          // [nrows, ncols/QK8_1] Q8_1 output
    float * scales_buffer,     // [nrows] temporary buffer for scales
    const int nrows,
    const int ncols,
    const int ncols_padded,
    const float eps,
    dpct::queue_ptr stream,
    int device
) {
    // Phase 1: Compute RMS scales
    compute_rms_scales_sycl(x, scales_buffer, nrows, ncols, eps, stream, device);

    // Phase 2: Fused quantization with normalization
    quantize_with_rmsnorm_q8_1_sycl<quantize_with_rmsnorm_q8_1>(
        x, gamma, scales_buffer, q8_output, ncols, nrows, ncols_padded, stream);
}

#endif // GGML_SYCL_FUSED_NORM_GEMM_HPP
