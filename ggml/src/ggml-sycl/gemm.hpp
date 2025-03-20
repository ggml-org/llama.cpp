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

#include <fstream>
#include <iostream>

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

    static inline void row_gemm(sycl::queue& q, bool a_trans,
        bool b_trans, int m, int n, int k,
        const void* a, dt at, const void* b, dt bt, void* c, dt ct)
    {
        // Get the device associated with the queue
        sycl::device dev = q.get_device();
        // Get the context associated with the queue
        sycl::context ctx = q.get_context();
        const dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        const dnnl::stream stream = dnnl::sycl_interop::make_stream(eng, q);
        dnnl::memory::dims a_dims = { m, k };
        dnnl::memory::dims b_dims = { k, n };
        dnnl::memory::dims c_dims = { m, n };
        const auto a_in_md = dnnl::memory::desc(a_dims, at, a_trans ? tag::ba : tag::ab);
        const auto b_in_md = dnnl::memory::desc(b_dims, bt, b_trans ? tag::ba : tag::ab);
        const auto c_md = dnnl::memory::desc(c_dims, ct, tag::ab);
        auto a_mem = dnnl::memory(a_in_md, eng, const_cast<void*>(a));
        auto b_mem = dnnl::memory(b_in_md, eng, const_cast<void*>(b));
        auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md);
        auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

        // Create the primitive.
        auto matmul_prim = dnnl::matmul(matmul_pd);
        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });
        matmul_args.insert({ DNNL_ARG_DST, c_mem });

        matmul_prim.execute(stream, matmul_args);
    }


    static inline void row_gemm(const dnnl::stream& stream, bool a_trans,
        bool b_trans, int m, int n, int k,
        const void* a, dt at, const void* b, dt bt, void* c, dt ct)
    {
        auto const eng = stream.get_engine();
        dnnl::memory::dims a_dims = { m, k };
        dnnl::memory::dims b_dims = { k, n };
        dnnl::memory::dims c_dims = { m, n };
        const auto a_in_md = dnnl::memory::desc(a_dims, at, a_trans ? tag::ba : tag::ab);
        const auto b_in_md = dnnl::memory::desc(b_dims, bt, b_trans ? tag::ba : tag::ab);
        const auto c_md = dnnl::memory::desc(c_dims, ct, tag::ab);
        auto a_mem = dnnl::memory(a_in_md, eng, const_cast<void*>(a));
        auto b_mem = dnnl::memory(b_in_md, eng, const_cast<void*>(b));
        auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md);
        auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

        // Create the primitive.
        auto matmul_prim = dnnl::matmul(matmul_pd);
        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });
        matmul_args.insert({ DNNL_ARG_DST, c_mem });

        matmul_prim.execute(stream, matmul_args);
    }
};

#endif

#endif // GGML_SYCL_GEMM_HPP
