//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_GEMM_HPP
#define GGML_SYCL_GEMM_HPP

#include "ggml-sycl.h"

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

class DnnlGemmWrapper {
public:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    template<typename T>
    static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) return dt::f32;
        else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
        else static_assert(0);
    }

    static void gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        const void * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        void * c, dt ct, const queue_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        dnnl::memory::dims a_dims = {batches_a, m, k };
        dnnl::memory::dims a_strides = {stra2, stra1, stra0};
        const auto a_in_md = dnnl::memory::desc(a_dims, at, a_strides);

        dnnl::memory::dims b_dims = {batches_b, k, n };
        dnnl::memory::dims b_strides = {strb2, strb0, strb1};
        const auto b_in_md = dnnl::memory::desc(b_dims, bt, b_strides);

        dnnl::memory::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::memory::dims c_strides = {m*n, 1,  m };
        const auto c_md    = dnnl::memory::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#ifdef GGML_SYCL_F16
        primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
#endif

        auto a_mem = dnnl::memory(a_in_md, eng, const_cast<void*>(a));
        auto b_mem = dnnl::memory(b_in_md, eng, const_cast<void*>(b));
        auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md, primitive_attr);
        auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

        auto scratchpad_md = matmul_pd.scratchpad_desc();
        auto scratchpad_mem = ctx.get_scratchpad_mem(scratchpad_md, eng, q);

        auto matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_args.insert({ DNNL_ARG_DST, c_mem });
        matmul_args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_args);
    }

    static void row_gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, const void * b, dt bt, void * c, dt ct, const queue_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }

    // Strided batch GEMM - C[i] = alpha * A[i] * B[i] + beta * C[i]
    // Matches dpct::gemm_batch interface for strided buffers
    static void gemm_batch_strided(
        ggml_backend_sycl_context & ctx,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        float alpha,
        const void * a, dt at, int lda, int64_t stride_a,
        const void * b, dt bt, int ldb, int64_t stride_b,
        float beta,
        void * c, dt ct, int ldc, int64_t stride_c,
        int batch_size,
        const queue_ptr & q)
    {
        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        // Set up dimensions based on transpose flags
        // oneDNN matmul: C = A * B where A is (batch, M, K), B is (batch, K, N), C is (batch, M, N)
        int a_rows = trans_a ? k : m;
        int a_cols = trans_a ? m : k;
        int b_rows = trans_b ? n : k;
        int b_cols = trans_b ? k : n;

        dnnl::memory::dims a_dims = {batch_size, a_rows, a_cols};
        dnnl::memory::dims b_dims = {batch_size, b_rows, b_cols};
        dnnl::memory::dims c_dims = {batch_size, m, n};

        // Strides: oneDNN expects {batch_stride, row_stride, col_stride}
        // For column-major (like MKL): row_stride = 1, col_stride = lda
        // For row-major: row_stride = lda, col_stride = 1
        // MKL uses column-major, so we need to transpose the operation
        dnnl::memory::dims a_strides = {stride_a, 1, lda};
        dnnl::memory::dims b_strides = {stride_b, 1, ldb};
        dnnl::memory::dims c_strides = {stride_c, 1, ldc};

        const auto a_md = dnnl::memory::desc(a_dims, at, a_strides);
        const auto b_md = dnnl::memory::desc(b_dims, bt, b_strides);
        const auto c_md = dnnl::memory::desc(c_dims, ct, c_strides);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        // Handle alpha and beta via post-ops if not 1.0/0.0
        if (alpha != 1.0f || beta != 0.0f) {
            dnnl::post_ops po;
            if (beta != 0.0f) {
                // C = alpha * (A * B) + beta * C
                // oneDNN does: dst = src * alpha + dst * beta with sum post-op
                po.append_sum(beta);
            }
            if (alpha != 1.0f) {
                po.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
            }
            attr.set_post_ops(po);
        }

#ifdef GGML_SYCL_F16
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
#endif

        auto a_mem = dnnl::memory(a_md, eng, const_cast<void*>(a));
        auto b_mem = dnnl::memory(b_md, eng, const_cast<void*>(b));

        auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
        auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

        auto scratchpad_md = matmul_pd.scratchpad_desc();
        auto scratchpad_mem = ctx.get_scratchpad_mem(scratchpad_md, eng, q);

        auto matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, a_mem});
        args.insert({DNNL_ARG_WEIGHTS, b_mem});
        args.insert({DNNL_ARG_DST, c_mem});
        args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

        matmul_prim.execute(stream, args);
    }

    // Pointer array batch GEMM - C[i] = alpha * A[i] * B[i] + beta * C[i]
    // For arrays of matrix pointers (non-contiguous batches)
    // Falls back to iterating over individual GEMM operations
    static void gemm_batch_array(
        ggml_backend_sycl_context & ctx,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        float alpha,
        const void ** a, dt at, int lda,
        const void ** b, dt bt, int ldb,
        float beta,
        void ** c, dt ct, int ldc,
        int batch_size,
        const queue_ptr & q)
    {
        // For pointer arrays, we iterate and call individual GEMM operations
        // This is less efficient than strided batch but handles non-contiguous data
        for (int i = 0; i < batch_size; ++i) {
            gemm_batch_strided(ctx, trans_a, trans_b, m, n, k,
                               alpha, a[i], at, lda, 0,
                               b[i], bt, ldb, 0,
                               beta, c[i], ct, ldc, 0,
                               1, q);
        }
    }

    // Simplified row-major batch GEMM (no transpose, alpha=1, beta=0)
    static void row_gemm_batch(
        ggml_backend_sycl_context & ctx,
        int m, int n, int k,
        const void * a, dt at, int64_t stride_a,
        const void * b, dt bt, int64_t stride_b,
        void * c, dt ct, [[maybe_unused]] int64_t stride_c,
        int batch_size,
        const queue_ptr & q)
    {
        // Use the existing gemm function which handles batching natively
        gemm(ctx, m, n, k,
             a, at, 1, k, stride_a,
             b, bt, 1, k, stride_b,
             c, ct, q, batch_size, batch_size);
    }
};

#endif

#endif // GGML_SYCL_GEMM_HPP
