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
#include "common.hpp"

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include <cstdio>
#include <array>
#include <unordered_map>
#include <mutex>

extern int g_ggml_sycl_debug;

// =============================================================================
// oneDNN Primitive Cache
// =============================================================================
// Caches oneDNN matmul primitives to avoid JIT compilation during SYCL graph
// recording. Primitive creation involves JIT which is incompatible with graph
// recording, but execute() on a pre-created primitive is graph-compatible.
//
// Usage:
// 1. During warmup (first inference), primitives are created and cached
// 2. During graph recording, cached primitives are reused (no JIT)
// 3. Cache key includes all parameters that affect primitive creation
// =============================================================================

struct DnnlPrimitiveKey {
    int64_t m, n, k;
    int64_t batches_a, batches_b;
    dnnl::memory::data_type at, bt, ct;
    // Strides for A
    int64_t stra0, stra1, stra2;
    // Strides for B
    int64_t strb0, strb1, strb2;
    // Strides for C
    int64_t strc0, strc1;
    // For batch_strided: transpose flags and alpha/beta
    bool trans_a, trans_b;
    float alpha, beta;
    int64_t stride_a, stride_b, stride_c;
    int lda, ldb, ldc;
    int batch_size;
    // Variant: 0 = gemm, 1 = gemm_batch_strided
    int variant;
    int64_t woq_group_size;
    int woq_scales_mask;
    int woq_zp_mask;

    bool operator==(const DnnlPrimitiveKey& other) const {
        return m == other.m && n == other.n && k == other.k &&
               batches_a == other.batches_a && batches_b == other.batches_b &&
               at == other.at && bt == other.bt && ct == other.ct &&
               stra0 == other.stra0 && stra1 == other.stra1 && stra2 == other.stra2 &&
               strb0 == other.strb0 && strb1 == other.strb1 && strb2 == other.strb2 &&
               strc0 == other.strc0 && strc1 == other.strc1 &&
               trans_a == other.trans_a && trans_b == other.trans_b &&
               alpha == other.alpha && beta == other.beta &&
               stride_a == other.stride_a && stride_b == other.stride_b && stride_c == other.stride_c &&
               lda == other.lda && ldb == other.ldb && ldc == other.ldc &&
               batch_size == other.batch_size && variant == other.variant &&
               woq_group_size == other.woq_group_size &&
               woq_scales_mask == other.woq_scales_mask &&
               woq_zp_mask == other.woq_zp_mask;
    }
};

struct DnnlPrimitiveKeyHash {
    size_t operator()(const DnnlPrimitiveKey& k) const {
        // Simple hash combining all fields
        size_t h = std::hash<int64_t>{}(k.m);
        h ^= std::hash<int64_t>{}(k.n) << 1;
        h ^= std::hash<int64_t>{}(k.k) << 2;
        h ^= std::hash<int64_t>{}(k.batches_a) << 3;
        h ^= std::hash<int64_t>{}(k.batches_b) << 4;
        h ^= std::hash<int>{}(static_cast<int>(k.at)) << 5;
        h ^= std::hash<int>{}(static_cast<int>(k.bt)) << 6;
        h ^= std::hash<int>{}(static_cast<int>(k.ct)) << 7;
        h ^= std::hash<int64_t>{}(k.stra0 + k.stra1 + k.stra2) << 8;
        h ^= std::hash<int64_t>{}(k.strb0 + k.strb1 + k.strb2) << 9;
        h ^= std::hash<int64_t>{}(k.strc0 + k.strc1) << 10;
        h ^= std::hash<int>{}(k.variant) << 11;
        h ^= std::hash<int>{}(k.batch_size) << 12;
        h ^= std::hash<int>{}(k.ldc) << 13;
        h ^= std::hash<int64_t>{}(k.woq_group_size) << 14;
        h ^= std::hash<int>{}(k.woq_scales_mask) << 15;
        h ^= std::hash<int>{}(k.woq_zp_mask) << 16;
        return h;
    }
};

struct DnnlCachedPrimitive {
    dnnl::matmul primitive;
    dnnl::engine engine;  // Engine the primitive was created with
    dnnl::memory::desc a_md;
    dnnl::memory::desc b_md;
    dnnl::memory::desc c_md;
    dnnl::memory::desc scratchpad_md;
    size_t scratchpad_size;
};

class DnnlPrimitiveCache {
public:
    // Get or create a cached primitive for the given key
    // Returns nullptr if creation fails
    // Note: Primitives are bound to a specific engine. If the engine changes
    // (e.g., new context between llama-bench runs), we recreate the primitive.
    const DnnlCachedPrimitive* get_or_create(
        const DnnlPrimitiveKey& key,
        const dnnl::engine& eng,
        const dnnl::memory::desc& a_md,
        const dnnl::memory::desc& b_md,
        const dnnl::memory::desc& c_md,
        const dnnl::primitive_attr& attr)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Check if cached primitive's engine matches current engine
            // oneDNN primitives are bound to a specific engine and cannot
            // be executed on a stream from a different engine
            if (it->second.engine == eng) {
                return &it->second;
            }
            // Engine mismatch - need to recreate primitive for new engine
            cache_.erase(it);
        }

        // Create new primitive
        try {
            DnnlCachedPrimitive cached;
            cached.engine = eng;  // Store engine for future comparisons
            auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
            cached.a_md = matmul_pd.src_desc();
            cached.b_md = matmul_pd.weights_desc();
            cached.c_md = matmul_pd.dst_desc();
            cached.scratchpad_md = matmul_pd.scratchpad_desc();
            cached.scratchpad_size = cached.scratchpad_md.get_size();
            cached.primitive = dnnl::matmul(matmul_pd);

            auto result = cache_.emplace(key, std::move(cached));
            return &result.first->second;
        } catch (const dnnl::error& e) {
            // Failed to create primitive
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN] matmul primitive creation failed: %s\n", e.what());
            }
            return nullptr;
        }
    }

    // Check if a primitive exists for the given key
    bool has(const DnnlPrimitiveKey& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.find(key) != cache_.end();
    }

    // Get cached primitive (returns nullptr if not found)
    const DnnlCachedPrimitive* get(const DnnlPrimitiveKey& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        return it != cache_.end() ? &it->second : nullptr;
    }

    // Clear the cache
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }

    // Get cache size
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

    // Get maximum scratchpad size across all cached primitives
    // Used to pre-allocate scratchpad pool before graph recording
    size_t get_max_scratchpad_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t max_size = 0;
        for (const auto& [key, cached] : cache_) {
            max_size = std::max(max_size, cached.scratchpad_size);
        }
        return max_size;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<DnnlPrimitiveKey, DnnlCachedPrimitive, DnnlPrimitiveKeyHash> cache_;
};

// Global primitive cache (shared across contexts)
inline DnnlPrimitiveCache& get_dnnl_primitive_cache() {
    static DnnlPrimitiveCache cache;
    return cache;
}

class DnnlGemmWrapper {
public:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    // Serialize oneDNN execution to avoid cross-thread primitive/memory races.
    // CPU uses a dedicated mutex. Each GPU device gets its own mutex so parallel
    // multi-GPU dispatch (e.g. B580 + B50) does not serialize against each other.
    static std::mutex & exec_mutex_cpu() {
        static std::mutex mutex;
        return mutex;
    }
    static std::mutex & exec_mutex(const queue_ptr & q) {
        if (q == ggml_sycl_get_cpu_queue()) {
            return exec_mutex_cpu();
        }
        // Per-GPU device mutex (allows parallel B580+B50 GEMM without serialization)
        static std::array<std::mutex, GGML_SYCL_MAX_DEVICES> gpu_mutexes;
        int dev_id = ggml_sycl_get_device_id_from_queue(*q);
        return gpu_mutexes[std::min(dev_id, static_cast<int>(gpu_mutexes.size()) - 1)];
    }

    template<typename T>
    static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) return dt::f32;
        else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
        else static_assert(0);
    }

    static void gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        const void * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        void * c, dt ct, const queue_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b, int ldc = -1) {
        std::lock_guard<std::mutex> lock(exec_mutex(q));

        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        if (ldc <= 0) {
            ldc = m;
        }

        // Build cache key
        DnnlPrimitiveKey key{};
        key.m = m;
        key.n = n;
        key.k = k;
        key.batches_a = batches_a;
        key.batches_b = batches_b;
        key.at = at;
        key.bt = bt;
        key.ct = ct;
        key.stra0 = stra0;
        key.stra1 = stra1;
        key.stra2 = stra2;
        key.strb0 = strb0;
        key.strb1 = strb1;
        key.strb2 = strb2;
        key.ldc = ldc;
        key.variant = 0;  // gemm variant

        // Build memory descriptors
        dnnl::memory::dims a_dims = {batches_a, m, k };
        dnnl::memory::dims a_strides = {stra2, stra1, stra0};
        const auto a_in_md = dnnl::memory::desc(a_dims, at, a_strides);

        dnnl::memory::dims b_dims = {batches_b, k, n };
        dnnl::memory::dims b_strides = {strb2, strb0, strb1};
        const auto b_in_md = dnnl::memory::desc(b_dims, bt, b_strides);

        dnnl::memory::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::memory::dims c_strides = {static_cast<dnnl_dim_t>(ldc) * n, 1, ldc};
        const auto c_md = dnnl::memory::desc(c_dims, ct, c_strides);

        dnnl::primitive_attr primitive_attr;
        primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#ifdef GGML_SYCL_F16
        primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
#endif

        // Get or create cached primitive
        auto& cache = get_dnnl_primitive_cache();
        const DnnlCachedPrimitive* cached = cache.get_or_create(key, eng, a_in_md, b_in_md, c_md, primitive_attr);

        if (!cached) {
            // Fallback: create primitive directly if caching fails
            auto a_mem = dnnl::memory(a_in_md, eng, const_cast<void*>(a));
            auto b_mem = dnnl::memory(b_in_md, eng, const_cast<void*>(b));
            auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md, primitive_attr);
            auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);
            auto scratchpad_md = matmul_pd.scratchpad_desc();
            const size_t scratchpad_size = scratchpad_md.get_size();
            auto matmul_prim = dnnl::matmul(matmul_pd);

            std::unordered_map<int, dnnl::memory> matmul_args;
            matmul_args.insert({ DNNL_ARG_SRC, a_mem });
            matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });
            matmul_args.insert({ DNNL_ARG_DST, c_mem });
            if (scratchpad_size > 0) {
                auto scratchpad_mem = ctx.get_scratchpad_mem(scratchpad_md, eng, q);
                matmul_args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });
            }
            matmul_prim.execute(stream, matmul_args);
            return;
        }

        // Use cached primitive - only memory binding and execute (graph-compatible)
        auto a_mem = dnnl::memory(cached->a_md, eng, const_cast<void*>(a));
        auto b_mem = dnnl::memory(cached->b_md, eng, const_cast<void*>(b));
        auto c_mem = dnnl::memory(cached->c_md, eng, c);
        const size_t scratchpad_size = cached->scratchpad_md.get_size();

        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });
        matmul_args.insert({ DNNL_ARG_DST, c_mem });
        if (scratchpad_size > 0) {
            auto scratchpad_mem = ctx.get_scratchpad_mem(cached->scratchpad_md, eng, q);
            matmul_args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });
        }

        cached->primitive.execute(stream, matmul_args);
    }

    static void row_gemm(ggml_backend_sycl_context & ctx, int m, int n, int k,
        const void * a, dt at, const void * b, dt bt, void * c, dt ct, const queue_ptr & q, int ldc = -1) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1, ldc);
    }

    // WoQ GEMM for Q4_0 weights (s4) with grouped scales/zero-points.
    // A: [m, k] row-major, B: [k, n] row-major (s4), C: [m, n] row-major.
    static bool woq_gemm_q4_0(
        ggml_backend_sycl_context & ctx,
        int m, int n, int k,
        const void * a, dt at,
        const void * b_s4,
        int64_t group_size,
        const float * scales,
        const int8_t * zero_points,
        void * c, dt ct,
        const queue_ptr & q,
        int64_t c_stride0,
        int64_t c_stride1) {
        return woq_gemm_q4_0_impl(ctx, m, n, k, a, at, b_s4, /* b_bytes = */ 0, /* b_is_packed = */ false,
                                 group_size, scales, zero_points, c, ct, q, c_stride0, c_stride1);
    }

    // WoQ GEMM with pre-packed oneDNN weights (b_packed uses cached->b_md layout).
    static bool woq_gemm_q4_0_packed(
        ggml_backend_sycl_context & ctx,
        int m, int n, int k,
        const void * a, dt at,
        const void * b_packed,
        size_t b_packed_bytes,
        int64_t group_size,
        const float * scales,
        const int8_t * zero_points,
        void * c, dt ct,
        const queue_ptr & q,
        int64_t c_stride0,
        int64_t c_stride1) {
        return woq_gemm_q4_0_impl(ctx, m, n, k, a, at, b_packed, b_packed_bytes, /* b_is_packed = */ true,
                                 group_size, scales, zero_points, c, ct, q, c_stride0, c_stride1);
    }

private:
    static bool woq_gemm_q4_0_impl(
        ggml_backend_sycl_context & ctx,
        int m, int n, int k,
        const void * a, dt at,
        const void * b_data,
        size_t b_bytes,
        bool b_is_packed,
        int64_t group_size,
        const float * scales,
        const int8_t * zero_points,
        void * c, dt ct,
        const queue_ptr & q,
        int64_t c_stride0,
        int64_t c_stride1) {

        if (!a || !b_data || !scales || !zero_points || !c) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] null pointer(s) provided\n");
            }
            return false;
        }
        if (m <= 0 || n <= 0 || k <= 0 || group_size <= 0) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] invalid dims m=%d n=%d k=%d group=%lld\n",
                             m, n, k, static_cast<long long>(group_size));
            }
            return false;
        }
        if (c_stride0 <= 0 || c_stride1 <= 0) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] invalid C strides %lld,%lld\n",
                             static_cast<long long>(c_stride0), static_cast<long long>(c_stride1));
            }
            return false;
        }
        if ((k % group_size) != 0) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] K not divisible by group size (k=%d group=%lld)\n",
                             k, static_cast<long long>(group_size));
            }
            return false;
        }

        const int64_t groups = k / group_size;

        std::lock_guard<std::mutex> lock(exec_mutex(q));
        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        const dnnl::memory::desc a_md({m, k}, at, {k, 1});
        const dnnl::memory::desc b_user_md({k, n}, dt::s4, {n, 1});
        const dnnl::memory::desc c_md({m, n}, ct, {c_stride0, c_stride1});
        const dnnl::memory::desc b_any_md({k, n}, dt::s4, tag::any);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        const int mask = (1 << 0) | (1 << 1);
        dnnl_dims_t group_dims = { group_size, 1 };
        if (dnnl_primitive_attr_set_scales(attr.get(), DNNL_ARG_WEIGHTS, mask, 2, group_dims,
                                           dnnl::memory::convert_to_c(dt::f32)) != dnnl_success) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] set_scales failed\n");
            }
            return false;
        }
        if (dnnl_primitive_attr_set_zero_points(attr.get(), DNNL_ARG_WEIGHTS, mask, 2, group_dims,
                                                dnnl::memory::convert_to_c(dt::s8)) != dnnl_success) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] set_zero_points failed\n");
            }
            return false;
        }
#ifdef GGML_SYCL_F16
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /* apply_to_int = */ true);
#endif

        DnnlPrimitiveKey key{};
        key.m = m;
        key.n = n;
        key.k = k;
        key.batches_a = 1;
        key.batches_b = 1;
        key.at = at;
        key.bt = dt::s4;
        key.ct = ct;
        key.stra0 = 1;
        key.stra1 = k;
        key.stra2 = static_cast<int64_t>(m) * k;
        key.strb0 = 1;
        key.strb1 = n;
        key.strb2 = static_cast<int64_t>(k) * n;
        key.ldc = n;
        key.variant = 2;
        key.strc0 = c_stride0;
        key.strc1 = c_stride1;
        key.woq_group_size = group_size;
        key.woq_scales_mask = mask;
        key.woq_zp_mask = mask;

        auto & cache = get_dnnl_primitive_cache();
        const DnnlCachedPrimitive * cached = cache.get_or_create(key, eng, a_md, b_any_md, c_md, attr);
        if (!cached) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] primitive cache miss+create failed\n");
            }
            return false;
        }

        auto a_mem = dnnl::memory(cached->a_md, eng, const_cast<void *>(a));
        auto c_mem = dnnl::memory(cached->c_md, eng, c);

        dnnl::memory b_mem = {};
        void * b_packed_dev = nullptr;
        if (b_is_packed) {
            const size_t packed_bytes = cached->b_md.get_size();
            if (b_bytes > 0 && b_bytes < packed_bytes) {
                if (g_ggml_sycl_debug) {
                    std::fprintf(stderr, "[ONEDNN][WOQ] packed weights too small (%zu < %zu)\n",
                                 b_bytes, packed_bytes);
                }
                return false;
            }
            b_mem = dnnl::memory(cached->b_md, eng, const_cast<void *>(b_data));
        } else {
            dnnl::memory b_user_mem(b_user_md, eng, const_cast<void *>(b_data));
            b_mem = b_user_mem;
            if (cached->b_md != b_user_mem.get_desc()) {
                const size_t packed_bytes = cached->b_md.get_size();
                b_packed_dev = sycl::aligned_alloc_device(64, packed_bytes, *q);
                if (!b_packed_dev) {
                    if (g_ggml_sycl_debug) {
                        std::fprintf(stderr, "[ONEDNN][WOQ] packed weights alloc failed (%zu bytes)\n",
                                     packed_bytes);
                    }
                    return false;
                }
                b_mem = dnnl::memory(cached->b_md, eng, b_packed_dev);
                dnnl::reorder(b_user_mem, b_mem).execute(stream, b_user_mem, b_mem);
                stream.wait();
            }
        }

        dnnl::memory scales_mem({{n, groups}, dt::f32, {1, n}}, eng, const_cast<float *>(scales));
        dnnl::memory zp_mem({{n, groups}, dt::s8, {1, n}}, eng, const_cast<int8_t *>(zero_points));

        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, a_mem});
        args.insert({DNNL_ARG_WEIGHTS, b_mem});
        args.insert({DNNL_ARG_DST, c_mem});
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales_mem});
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem});
        if (cached->scratchpad_size > 0) {
            auto scratchpad_mem = ctx.get_scratchpad_mem(cached->scratchpad_md, eng, q);
            args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});
        }

        cached->primitive.execute(stream, args);

        if (b_packed_dev) {
            sycl::free(b_packed_dev, *q);
        }
        return true;
    }

public:

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
        std::lock_guard<std::mutex> lock(exec_mutex(q));
        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        // Build cache key for batch_strided variant
        DnnlPrimitiveKey key{};
        key.m = m;
        key.n = n;
        key.k = k;
        key.at = at;
        key.bt = bt;
        key.ct = ct;
        key.trans_a = trans_a;
        key.trans_b = trans_b;
        key.alpha = alpha;
        key.beta = beta;
        key.stride_a = stride_a;
        key.stride_b = stride_b;
        key.stride_c = stride_c;
        key.lda = lda;
        key.ldb = ldb;
        key.ldc = ldc;
        key.batch_size = batch_size;
        key.variant = 1;  // gemm_batch_strided variant

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

        // Get or create cached primitive
        auto& cache = get_dnnl_primitive_cache();
        const DnnlCachedPrimitive* cached = cache.get_or_create(key, eng, a_md, b_md, c_md, attr);

        if (!cached) {
            // Fallback: create primitive directly if caching fails
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
            return;
        }

        // Use cached primitive - only memory binding and execute (graph-compatible)
        auto a_mem = dnnl::memory(cached->a_md, eng, const_cast<void*>(a));
        auto b_mem = dnnl::memory(cached->b_md, eng, const_cast<void*>(b));
        auto c_mem = dnnl::memory(cached->c_md, eng, c);
        auto scratchpad_mem = ctx.get_scratchpad_mem(cached->scratchpad_md, eng, q);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, a_mem});
        args.insert({DNNL_ARG_WEIGHTS, b_mem});
        args.insert({DNNL_ARG_DST, c_mem});
        args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

        cached->primitive.execute(stream, args);
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
