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

#include <string>
#include <unordered_map>

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
#ifdef GGML_SYCL_HAS_BF16
        else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) return dt::bf16;
#endif
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

        const auto scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        void * scratchpad_ptr = scratchpad_md.get_size() > 0 ? scratchpad.alloc(scratchpad_md.get_size()) : nullptr;
        auto scratchpad_mem = dnnl::memory(scratchpad_md, eng, scratchpad_ptr);

        auto matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_args.insert({ DNNL_ARG_DST, c_mem });
        matmul_args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_args);
    }

    // dst[n][m] = act[n][k] * w[m][k]^T, w = reordered Q4_0 read as u4 (zero point 8, f16 scale per 32 values, scales is [k/32][m])
    // Q4_0 nibble order in a block is 0,16,1,17,...,15,31: the caller permutes act the same way
    static void gemm_q4_0_u4(ggml_backend_sycl_context & ctx, int m, int n, int k, const sycl::half * act,
                             const void * w, const sycl::half * scales, float * dst, const queue_ptr & q) {
        auto stream = ctx.stream_dnnl(q);
        auto eng    = ctx.engine_dnnl(q);
        const int device = ggml_sycl_get_device();

        const auto act_md = dnnl::memory::desc({ n, k }, dt::f16, tag::ab);
        const auto w_md   = dnnl::memory::desc({ k, m }, dt::u4, tag::ba);
        const auto dst_md = dnnl::memory::desc({ n, m }, dt::f32, tag::ab);
        const auto sc_md  = dnnl::memory::desc({ k / QK4_0, m }, dt::f16, tag::ab);
        const auto zp_md  = dnnl::memory::desc({ 1 }, dt::u8, tag::a);

        // the zero point never changes: one byte per device, set once
        static uint8_t * zp8[GGML_SYCL_MAX_DEVICES] = {};
        if (!zp8[device]) {
            zp8[device] = sycl::malloc_device<uint8_t>(1, *q);
            q->memset(zp8[device], 8, 1).wait();
        }

        // build once per (engine, shape); a new llama context has a new engine, so the entry keeps its engine alive
        struct u4_gemm { dnnl::engine eng; dnnl::matmul::primitive_desc pd; dnnl::matmul prim; };
        static std::unordered_map<std::string, u4_gemm> cache;
        char key[96];
        snprintf(key, sizeof(key), "%p:%d:%d:%d", (void *) eng.get(), m, n, k);
        auto it = cache.find(key);
        if (it == cache.end()) {
            dnnl::primitive_attr attr;
            attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), { QK4_0, 1 }, dt::f16);
            attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, dt::u8);
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            auto pd = dnnl::matmul::primitive_desc(eng, act_md, w_md, dst_md, attr);
            it = cache.emplace(key, u4_gemm{ eng, pd, dnnl::matmul(pd) }).first;
        }
        const auto & pd   = it->second.pd;
        const auto & prim = it->second.prim;

        const auto scratchpad_md = pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        void * scratchpad_ptr = scratchpad_md.get_size() > 0 ? scratchpad.alloc(scratchpad_md.get_size()) : nullptr;

        std::unordered_map<int, dnnl::memory> args;
        args.insert({ DNNL_ARG_SRC, dnnl::memory(act_md, eng, const_cast<sycl::half *>(act)) });
        args.insert({ DNNL_ARG_WEIGHTS, dnnl::memory(w_md, eng, const_cast<void *>(w)) });
        args.insert({ DNNL_ARG_DST, dnnl::memory(pd.dst_desc(), eng, dst) });
        args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(sc_md, eng, const_cast<sycl::half *>(scales)) });
        args.insert({ DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, dnnl::memory(zp_md, eng, zp8[device]) });
        args.insert({ DNNL_ARG_SCRATCHPAD, dnnl::memory(scratchpad_md, eng, scratchpad_ptr) });

        prim.execute(stream, args);
    }

    static void row_gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, const void * b, dt bt, void * c, dt ct, const queue_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#endif

#endif // GGML_SYCL_GEMM_HPP
