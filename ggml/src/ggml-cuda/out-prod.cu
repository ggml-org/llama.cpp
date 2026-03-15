#include "out-prod.cuh"
#include "convert.cuh"

#include <cstdint>
#include <cstring>
#include <vector>

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    // If src0 is quantized, dequantize to a temp F32 buffer on GPU
    ggml_cuda_pool_alloc<float> src0_f32_alloc;
    const float * src0_d;
    int64_t       lda;

    if (src0->type != GGML_TYPE_F32) {
        const int64_t n_elements = ggml_nelements(src0);
        src0_f32_alloc.alloc(ctx.pool(), n_elements);

        to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(src0->type);
        GGML_ASSERT(to_fp32 != nullptr);
        to_fp32(src0->data, src0_f32_alloc.ptr, n_elements, stream);

        src0_d = src0_f32_alloc.ptr;
        lda    = ne00; // dequantized data is contiguous: stride = ne00
    } else {
        src0_d = (const float *) src0->data;
        lda    = nb01 / sizeof(float);
    }

    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const int64_t ldc = nb1  / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op =  src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t           ldb            = (src1_T ?        nb10 :        nb11) /  sizeof(float);
    GGML_ASSERT(                             (src1_T ?        nb11 :        nb10) == sizeof(float));

    // data strides in dimensions 2/3 (for dequantized src0, use element-based strides)
    const size_t s02 = (src0->type != GGML_TYPE_F32) ? (ne00 * ne01)        : (nb02 / sizeof(float));
    const size_t s03 = (src0->type != GGML_TYPE_F32) ? (ne00 * ne01 * ne02) : (nb03 / sizeof(float));
    const size_t s12 = nb12 / sizeof(float);
    const size_t s13 = nb13 / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    // dps == dst per src0, used for group query attention
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    // TODO batched matrix multiplication
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            CUBLAS_CHECK(
                cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                        ne0, ne1, ne01,
                        &alpha, src0_d + (i3/dps3)*s03 + (i2/dps2)*s02, lda,
                                src1_d +  i3      *s13 +  i2      *s12, ldb,
                        &beta,  dst_d  +  i3      *s3  +  i2      *s2,  ldc));
        }
    }
}

// ggml_cuda_out_prod_id
//
// Scattered outer-product for the MUL_MAT_ID backward pass (gradient w.r.t. expert weights).
//
//   src0 = a   [cols, n_expert_used, n_tokens]  F32  — token activations
//   src1 = b   [rows, n_expert_used, n_tokens]  F32  — upstream gradient
//   src2 = ids [n_expert_used, n_tokens]        I32  — expert dispatch indices
//   dst        [cols, rows, n_expert, 1]         F32  — gradient w.r.t. expert weight matrices
//
//   dst[:, :, e] += sum_{(i,t): ids[i,t]==e} a[:, i, t] ⊗ b[:, i, t]
//
// Algorithm:
//   For each expert e: gather the token columns where ids[i,t]==e into contiguous
//   GPU buffers, then use cublasSgemm (beta=1) to accumulate the outer product.
//   ids may be CPU-resident (common in backward graphs where they are leaf tensors).
void ggml_cuda_out_prod_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // a   [cols, n_exp_used, n_tokens]
    const ggml_tensor * src1 = dst->src[1]; // b   [rows, n_exp_used, n_tokens]
    const ggml_tensor * ids  = dst->src[2]; // ids [n_exp_used, n_tokens]  i32

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t cols       = src0->ne[0];
    const int64_t n_exp_used = src0->ne[1];
    const int64_t n_tokens   = src0->ne[2];
    const int64_t rows       = src1->ne[0];
    const int64_t n_expert   = dst->ne[2];

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Zero destination tensor before accumulating
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    // Read ids to host — ids may be CPU-resident (backward graph leaf) or GPU-resident
    const size_t ids_nbytes = ggml_nbytes(ids);
    std::vector<char> ids_host(ids_nbytes);
    if (ids->buffer && !ggml_backend_buffer_is_host(ids->buffer)) {
        // GPU-resident: copy to host and synchronize so we can inspect the values
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ids_nbytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        memcpy(ids_host.data(), ids->data, ids_nbytes);
    }

    // Build per-expert token list: expert_tokens[e] = list of flat indices (iexp*n_tokens+itok)
    // whose dispatch id equals e.
    std::vector<std::vector<int64_t>> expert_tokens(n_expert);
    for (int64_t itok = 0; itok < n_tokens; ++itok) {
        for (int64_t iexp = 0; iexp < n_exp_used; ++iexp) {
            const int32_t eid = *(const int32_t *)(ids_host.data()
                + itok * ids->nb[1] + iexp * ids->nb[0]);
            GGML_ASSERT(eid >= 0 && eid < (int32_t)n_expert);
            expert_tokens[eid].push_back(iexp * n_tokens + itok);
        }
    }

    // Strides (in elements, not bytes)
    const int64_t a_stride_exp = src0->nb[1] / sizeof(float); // cols
    const int64_t a_stride_tok = src0->nb[2] / sizeof(float); // cols * n_exp_used
    const int64_t b_stride_exp = src1->nb[1] / sizeof(float); // rows
    const int64_t b_stride_tok = src1->nb[2] / sizeof(float); // rows * n_exp_used
    const int64_t dst_stride_e = dst->nb[2]  / sizeof(float); // cols * rows

    const float alpha_one = 1.0f;
    const float beta_acc  = 1.0f; // accumulate — dst is already zeroed above

    const float * a_base = (const float *) src0->data;
    const float * b_base = (const float *) src1->data;
    float       * d_base = (float       *)  dst->data;

    for (int64_t e = 0; e < n_expert; ++e) {
        const auto & toks = expert_tokens[e];
        if (toks.empty()) {
            continue;
        }

        const int64_t ntoks_e = (int64_t) toks.size();

        // Allocate contiguous gather buffers on GPU: a_e [cols, ntoks_e], b_e [rows, ntoks_e]
        ggml_cuda_pool_alloc<float> a_gathered(ctx.pool(), cols * ntoks_e);
        ggml_cuda_pool_alloc<float> b_gathered(ctx.pool(), rows * ntoks_e);

        // Gather token vectors from GPU src0/src1 into contiguous buffers
        for (int64_t ti = 0; ti < ntoks_e; ++ti) {
            const int64_t flat = toks[ti];
            const int64_t iexp = flat / n_tokens;
            const int64_t itok = flat % n_tokens;
            CUDA_CHECK(cudaMemcpyAsync(
                a_gathered.ptr + ti * cols,
                a_base + iexp * a_stride_exp + itok * a_stride_tok,
                cols * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                b_gathered.ptr + ti * rows,
                b_base + iexp * b_stride_exp + itok * b_stride_tok,
                rows * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }

        // dst[:, :, e] += a_gathered @ b_gathered^T
        // cuBLAS column-major: A=[cols, ntoks_e] lda=cols, B=[rows, ntoks_e] ldb=rows
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)cols, (int)rows, (int)ntoks_e,
            &alpha_one, a_gathered.ptr,         (int)cols,
                        b_gathered.ptr,         (int)rows,
            &beta_acc,  d_base + e*dst_stride_e, (int)cols));
    }
}
