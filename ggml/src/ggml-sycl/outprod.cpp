#include "outprod.hpp"
#include "gemm.hpp"

void ggml_sycl_op_out_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    // Get SYCL queue
    dpct::queue_ptr stream = ctx.stream();

    // Dimension checks
    GGML_ASSERT(ne01 == ne11);  // Inner dimensions must match
    GGML_ASSERT(ne0 == ne00);   // Output rows match src0 rows
    GGML_ASSERT(ne1 == ne10);   // Output cols match src1 cols

    const int device = ctx.device;

    // Get data pointers
    const float * src0_d = (const float *) ggml_sycl_get_data_ptr(src0, device);
    const float * src1_d = (const float *) ggml_sycl_get_data_ptr(src1, device);
    float *       dst_d  = (float *) ggml_sycl_get_data_ptr(dst, device);

    // oneDNN can mis-handle this degenerate case; use a simple kernel instead.
    if (ne1 == 1 && ne01 == 1) {
        const int64_t n = ne0;
        const int block_size = 256;
        const int64_t num_blocks = (n + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                const int64_t i = item.get_global_id(2);
                if (i < n) {
                    dst_d[i] = src0_d[i] * src1_d[0];
                }
            });
        return;
    }

    // Handle transposition of src1
    const bool src1_T = ggml_is_transposed(src1);
    const int64_t ldb = (src1_T ? nb10 : nb11) / sizeof(float);

    try {
#if GGML_SYCL_DNNL
        // Use oneDNN for outer product (GEMM: C = A * B^T)
        // For outer product: A is (ne00, ne01), B is (ne10, ne11)
        // Result C is (ne00, ne10) = (ne0, ne1)

        // oneDNN row_gemm expects: C = A * B where A is (M, K) and B is (K, N)
        // For outer product with B transpose:
        //   C (ne0 x ne1) = A (ne0 x ne01) * B^T (ne01 x ne1)
        // This requires: M = ne0, N = ne1, K = ne01

        // Set up strides for the matrices
        // src0: (ne00, ne01) row-major, stride = ne00 (or from nb01/sizeof(float))
        // src1: if transposed (ne11, ne10), else (ne10, ne11)
        // dst: (ne0, ne1) row-major, stride = ne0

        int64_t str_a0 = 1;                         // column stride for A
        int64_t str_a1 = nb01 / sizeof(float);      // row stride for A
        int64_t str_a2 = nb02 / sizeof(float);      // batch stride for A

        int64_t str_b0 = 1;
        int64_t str_b1 = ldb;
        int64_t str_b2 = (src1_T ? nb12 : nb12) / sizeof(float);

        // Use DnnlGemmWrapper::gemm for the outer product
        // C = A * B^T when src1 is not transposed, C = A * B when src1 is transposed
        DnnlGemmWrapper::gemm(ctx, ne0, ne1, ne01, src0_d,
                              DnnlGemmWrapper::to_dt<float>(), str_a0, str_a1, str_a2,
                              src1_d, DnnlGemmWrapper::to_dt<float>(), str_b0, str_b1, str_b2,
                              dst_d, DnnlGemmWrapper::to_dt<float>(), stream, 1, 1);
#elif GGML_SYCL_HAS_ONEAPI_MATH
        // Fallback to oneAPI Math (MKL/oneMath) for GEMM
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const oneapi::math::transpose src1_op = src1_T ? oneapi::math::transpose::nontrans : oneapi::math::transpose::trans;

        oneapi::math::blas::column_major::gemm(get_onemath_backend(*stream), oneapi::math::transpose::nontrans, src1_op,
                                               ne0, ne1, ne01, alpha, src0_d, ne00, src1_d, ldb, beta, dst_d, ne0);
#else
        static_assert(false, "Either GGML_SYCL_DNNL or GGML_SYCL_HAS_ONEAPI_MATH must be defined for out_prod operation");
#endif
    }
    catch (sycl::exception const& exc) {
        std::cerr << exc.what() << std::endl;
        GGML_ASSERT(false);
    }
}
