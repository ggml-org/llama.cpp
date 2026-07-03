#include "conv3d.hpp"

static inline int64_t ggml_sycl_conv3d_calc_patch_total(const ggml_tensor * dst, int32_t n) {
    return (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

static inline int64_t ggml_sycl_conv3d_calc_knl_n_total(const ggml_tensor * src0, int32_t c) {
    return (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

static inline void ggml_sycl_conv3d_write_output(
        const ggml_tensor * dst,
        const float * src, float * dst_data,
        int64_t patch_total, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::queue_ptr stream) {
    const int64_t dst_nb0 = dst->nb[0];
    const int64_t dst_nb1 = dst->nb[1];
    const int64_t dst_nb2 = dst->nb[2];
    const int64_t dst_nb3 = dst->nb[3];
    const int64_t total = patch_total * oc;
    const int64_t block_size = 256;
    const int64_t num_work_items = ((total + block_size - 1) / block_size) * block_size;

    stream->parallel_for(sycl::range<1>(num_work_items), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        if (i >= total) {
            return;
        }

        const int64_t patch_idx = i / oc;
        const int64_t out_ch = i % oc;
        const int64_t p_in_batch = patch_idx % (dst_w * dst_h * dst_d);
        const int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        const int64_t dst_z = p_in_batch / (dst_w * dst_h);
        const int64_t dst_y = (p_in_batch % (dst_w * dst_h)) / dst_w;
        const int64_t dst_x = p_in_batch % dst_w;
        const int64_t ocn_idx = batch_idx * oc + out_ch;

        const int64_t dst_offset = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        const int64_t src_index = patch_idx + out_ch * patch_total;
        const float value = src[src_index];
        *(float *)((char *)dst_data + dst_offset) = value;
    });
}

void ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int32_t s0 = opts[0];
    const int32_t s1 = opts[1];
    const int32_t s2 = opts[2];
    const int32_t p0 = opts[3];
    const int32_t p1 = opts[4];
    const int32_t p2 = opts[5];
    const int32_t d0 = opts[6];
    const int32_t d1 = opts[7];
    const int32_t d2 = opts[8];
    const int32_t c  = opts[9];
    const int32_t n  = opts[10];
    const int32_t oc = opts[11];

    const int64_t knl_w = src0->ne[0];
    const int64_t knl_h = src0->ne[1];
    const int64_t knl_d = src0->ne[2];

    const int64_t patch_total = ggml_sycl_conv3d_calc_patch_total(dst, n);
    const int64_t knl_n_total = ggml_sycl_conv3d_calc_knl_n_total(src0, c);

    const size_t kernel_type_size = ggml_element_size(src0);

    ggml_sycl_pool_alloc<float> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_total * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_total;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = sizeof(float);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.data = gemm_output.get();
    dst_mat.buffer = dst->buffer;
    dst_mat.extra = dst->extra;

    dpct::queue_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<float> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<float> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((size_t) knl_n_total * patch_total);
    B_packed_alloc.alloc((size_t) knl_n_total * oc);

    float * A_packed = A_packed_alloc.get();
    float * B_packed = B_packed_alloc.get();

    const int m = (int) patch_total;
    const int n_gemm = (int) oc;
    const int k = (int) knl_n_total;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    const char * src1_base = (const char *) src1->data;
    const char * src0_base = (const char *) src0->data;
    const int64_t src1_nb0 = src1->nb[0];
    const int64_t src1_nb1 = src1->nb[1];
    const int64_t src1_nb2 = src1->nb[2];
    const int64_t src1_nb3 = src1->nb[3];
    const int64_t src1_w = src1->ne[0];
    const int64_t src1_h = src1->ne[1];
    const int64_t src1_d = src1->ne[2];

    const bool src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    const int64_t src0_packed_nb0 = kernel_type_size;
    const int64_t src0_packed_nb1 = kernel_type_size * knl_n_total;

    const int64_t KW = knl_w;
    const int64_t KH = knl_h;
    const int64_t KD = knl_d;
    const int64_t PW = dst->ne[0];
    const int64_t PH = dst->ne[1];
    const int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    const int64_t A_total = (int64_t)k * m;
    const int64_t A_block_size = 256;
    const int64_t A_num_work = ((A_total + A_block_size - 1) / A_block_size) * A_block_size;

    stream->parallel_for(sycl::range<1>(A_num_work), [=](sycl::id<1> id) {
        const int64_t t = id[0];
        if (t >= A_total) return;

        const int64_t row = t % k;
        const int64_t col = t / k;

        // Inline im2col for this element
        const int64_t k_index = row;
        const int64_t patch_idx = col;

        const int64_t ic = k_index / (KD * KH * KW);
        const int64_t rem = k_index - ic * (KD * KH * KW);
        const int64_t kz = rem / (KH * KW);
        const int64_t rem2 = rem - kz * (KH * KW);
        const int64_t ky = rem2 / KW;
        const int64_t kx = rem2 % KW;

        const int64_t p_in_batch = patch_idx % (PW * PH * PD);
        const int64_t batch_idx = patch_idx / (PW * PH * PD);
        const int64_t dst_z = p_in_batch / (PW * PH);
        const int64_t dst_y = (p_in_batch % (PW * PH)) / PW;
        const int64_t dst_x = p_in_batch % PW;

        const int64_t sx = dst_x * s0 + kx * d0 - p0;
        const int64_t sy = dst_y * s1 + ky * d1 - p1;
        const int64_t sz = dst_z * s2 + kz * d2 - p2;

        float val = 0.0f;
        if (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            const int64_t channel_idx = batch_idx * c + ic;
            const char * ptr = src1_base + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(const float *) ptr;
        }
        A_packed[row + col * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    const int64_t B_total = (int64_t)k * n_gemm;
    const int64_t B_block_size = 256;
    const int64_t B_num_work = ((B_total + B_block_size - 1) / B_block_size) * B_block_size;

    stream->parallel_for(sycl::range<1>(B_num_work), [=](sycl::id<1> id) {
        const int64_t t = id[0];
        if (t >= B_total) return;

        const int64_t row = t % k;
        const int64_t col = t / k;
        const char * src_ptr = src0_base + row * src0_packed_nb0 + col * src0_packed_nb1;
        float v;
        if (src0_is_f32) {
            v = *(const float *) src_ptr;
        } else {
            v = sycl::vec<sycl::half, 1>(*(const sycl::half *) src_ptr).convert<float, sycl::rounding_mode::automatic>()[0];
        }
        B_packed[row + col * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const int lda = k;
    const int ldb = k;
    const int ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_value(&alpha, *stream),
        (const float *) A_packed, lda,
        (const float *) B_packed, ldb,
        dpct::get_value(&beta, *stream),
        (float *) dst_mat.data, ldc)));

    const float * gemm_data = (const float *) dst_mat.data;
    float * dst_data = (float *) dst->data;

    ggml_sycl_conv3d_write_output(dst, gemm_data, dst_data, patch_total, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
