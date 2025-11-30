//
// XMX-accelerated quantized matrix multiplication for Intel Arc GPUs
// Uses Intel Xe Matrix eXtensions (XMX) via SYCL joint_matrix API
//

#ifndef GGML_SYCL_MMQ_XMX_HPP
#define GGML_SYCL_MMQ_XMX_HPP

#include "common.hpp"

// Test function to verify XMX Int8 works correctly
void ggml_sycl_xmx_test_func(ggml_backend_sycl_context & ctx);

// Check if XMX is available for GEMM acceleration
bool ggml_sycl_xmx_available();

// Get XMX tile dimensions (M, N, K) for Int8
// For Intel Arc B50: M=8, N=16, K=32
void ggml_sycl_xmx_get_tile_dims(int* m, int* n, int* k);

// Check if a quantization type is supported by XMX kernels
bool ggml_sycl_xmx_supports_type(ggml_type type);

// XMX-accelerated quantized matrix multiplication operation
// Matches the signature of ggml_sycl_op_mul_mat_q for drop-in replacement
void ggml_sycl_op_mul_mat_q_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

#endif // GGML_SYCL_MMQ_XMX_HPP
