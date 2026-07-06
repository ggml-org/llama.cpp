//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Hybrid flash attention: dequantize K/V to F16 → oneDNN SDPA (XMX fused kernel).
// Combines the dequant pipeline from PR #25025 with the hardware SDPA from PR #25222.
// Covers quantized, BF16, and F32 KV caches that oneDNN's native F16-only gate rejects.

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>

#include "fattn-hybrid.hpp"
#include "fattn.hpp"         // for fallback: ggml_sycl_flash_attn_ext_mkl
#include "fattn-tile.hpp"
#include "convert.hpp"
#include "fattn-buffers.hpp"

#ifdef GGML_SYCL_DNNL
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#endif

// ---------------------------------------------------------------------------
// Gate: eligible shapes for dequant + oneDNN SDPA
// ---------------------------------------------------------------------------
bool ggml_sycl_flash_attn_ext_hybrid_supported(const ggml_tensor * dst) {
#if !GGML_SYCL_DNNL
    GGML_UNUSED(dst);
    return false;
#else
    // Respect the global kill switch
    if (!g_ggml_sycl_fa_onednn) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    // HYBRID handles non-F16 types (native F16 goes to ONEDNN when merged).
    if (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        return false;
    }

    // MKL prefill gate conditions
    if (Q->ne[1] < 32 || K->ne[1] < 1024) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (!(Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        return false;
    }

    // F16 stride alignment (from MKL gate)
    for (const ggml_tensor * t : {K, V}) {
        if (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
            return false;
        }
    }

    // SDPA conditions
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    if (sinks) {
        return false;
    }
    const int64_t d = K->ne[0];
    if (V->ne[0] != d || Q->ne[3] != 1) {
        return false;
    }
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }

    return true;
#endif
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------
#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

using namespace dnnl;
using namespace dnnl::graph;

// Build + compile the contiguous-input GQA SDPA graph.
// MatMul → Divide → Add(mask) → SoftMax → MatMul, f16 out.
// Partitions=1, sdp_primitive_kernel_t (the systolic XMX kernel).
// Originally from fattn-onednn.cpp (PR #25222 by @hmscider).
sdpa_partition build_sdpa(const engine & eng, int H, int Hkv, int q, int seq, int d) {
    using ltype = logical_tensor::layout_type;
    using dt    = logical_tensor::data_type;
    using ldims = logical_tensor::dims;
    const dt    fi = dt::f32, t = dt::f16;
    const int   rep = H / Hkv;
    const ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    int64_t        id = 0;
    sdpa_partition E;

    auto query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    auto key    = logical_tensor(id++, t,  kv_sz, ltype::strided);
    auto score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto bmm1   = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key}); bmm1.add_outputs({score});

    auto scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    auto scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto sdiv   = op(id++, op::kind::Divide, "scale_div");
    sdiv.add_inputs({score, scale}); sdiv.add_outputs({scaled});

    auto mask   = logical_tensor(id++, t,  msk,  ltype::strided);
    auto masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto madd   = op(id++, op::kind::Add, "mask_add");
    madd.add_inputs({scaled, mask}); madd.add_outputs({masked});

    auto probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    auto smax   = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({masked}); smax.add_outputs({probs});

    auto value  = logical_tensor(id++, t,  kv_sz, ltype::strided);
    // f16 output required to hit sdp_primitive_kernel_t.
    auto output = logical_tensor(id++, t,  o_sz, ltype::strided);
    auto bmm2   = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});

    dnnl::graph::graph g(eng.get_kind());
    g.add_op(bmm1); g.add_op(sdiv); g.add_op(madd); g.add_op(smax); g.add_op(bmm2);
    g.finalize();

    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) {
        return E;
    }
    E.ins      = parts[0].get_input_ports();
    E.out      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.out}, eng);
    E.out      = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q     = query.get_id(); E.id_k = key.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_mask = mask.get_id();
    E.ok       = true;
    return E;
}

void ggml_sycl_flash_attn_ext_hybrid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    ggml_tensor * KQV = dst;

    const int64_t d   = K->ne[0];
    const int64_t seq = K->ne[1];
    const int64_t Hkv = K->ne[2];
    const int64_t H   = Q->ne[2];
    const int64_t q   = Q->ne[1];
    const int64_t mb  = Q->ne[3];

    float kq_scale = 1.0f;
    memcpy(&kq_scale, (const float *) KQV->op_params + 0, sizeof(float));

    dpct::queue_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.stream_dnnl(stream);

    // --- Step 1: Dequant K to dense F16 ---
    ggml_sycl_fattn_alloc K_f16(ctx.fattn_buffers().K);
    {
        const char * K_data = (const char *)K->data;
        const bool k_interleaved =
            ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;

        K_f16.alloc(ggml_nelements(K));

        if (K->type == GGML_TYPE_F16) {
            if (!k_interleaved) {
                stream->memcpy(K_f16.ptr, K_data, ggml_nelements(K) * sizeof(sycl::half));
            } else {
                const int64_t ne0 = K->ne[0], ne1 = K->ne[1];
                const int64_t ne23 = (int64_t)K->ne[2] * K->ne[3];
                const int64_t src_nb1 = (int64_t)K->nb[1];
                const int64_t src_nb2 = (int64_t)K->nb[2];
                const sycl::half * src = (const sycl::half *)K_data;
                sycl::half * dst = K_f16.ptr;
                stream->parallel_for(
                    sycl::range<3>((size_t)ne23, (size_t)ne1, (size_t)ne0),
                    [=](sycl::item<3> it) {
                        int64_t hb = it.get_id(0), r = it.get_id(1), c = it.get_id(2);
                        const sycl::half * src_row = (const sycl::half *)(
                            (const char *)src + hb * src_nb2 + r * src_nb1);
                        dst[(hb * ne1 + r) * ne0 + c] = src_row[c];
                    });
            }
        } else if (ggml_is_contiguously_allocated(K) && !k_interleaved) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K_data, K_f16.ptr, ggml_nelements(K), stream);
        } else {
            const size_t bs = ggml_blck_size(K->type);
            const size_t ts = ggml_type_size(K->type);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            int64_t s01, s02, s03;
            if (k_interleaved) {
                const int64_t blk_per_row = (int64_t)K->ne[0] / bs;
                s01 = (int64_t)Hkv * blk_per_row;
                s02 = blk_per_row;
                s03 = (int64_t)K->ne[1] * s01;
            } else {
                s01 = (int64_t)K->nb[1] / ts;
                s02 = (int64_t)K->nb[2] / ts;
                s03 = (int64_t)K->nb[3] / ts;
            }
            to_fp16(K_data, K_f16.ptr,
                    K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                    s01, s02, s03, stream);
        }
    }

    // --- Step 2: Dequant V to dense F16 ---
    // K and V share buffers in quantized KV cache (V is a view of K).
    // Detect it: same data pointer means reuse K_f16.
    ggml_sycl_fattn_alloc V_f16(ctx.fattn_buffers().V);
    bool V_is_K_view = (K->type != GGML_TYPE_F32 && V->type != GGML_TYPE_F32 &&
                         K->data == V->data);
    if (V_is_K_view) {
        // V shares K's buffer — reuse the already-normalized dense K data
        V_f16.ptr = K_f16.ptr;  // don't allocate, just alias
    } else {
        const char * V_data = (const char *)V->data;
        const bool v_interleaved =
            ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

        V_f16.alloc(ggml_nelements(V));

        if (V->type == GGML_TYPE_F16) {
            if (!v_interleaved) {
                stream->memcpy(V_f16.ptr, V_data, ggml_nelements(V) * sizeof(sycl::half));
            } else {
                const int64_t ne0 = V->ne[0], ne1 = V->ne[1];
                const int64_t ne23 = (int64_t)V->ne[2] * V->ne[3];
                const int64_t src_nb1 = (int64_t)V->nb[1];
                const int64_t src_nb2 = (int64_t)V->nb[2];
                const sycl::half * src = (const sycl::half *)V_data;
                sycl::half * dst = V_f16.ptr;
                stream->parallel_for(
                    sycl::range<3>((size_t)ne23, (size_t)ne1, (size_t)ne0),
                    [=](sycl::item<3> it) {
                        int64_t hb = it.get_id(0), r = it.get_id(1), c = it.get_id(2);
                        const sycl::half * src_row = (const sycl::half *)(
                            (const char *)src + hb * src_nb2 + r * src_nb1);
                        dst[(hb * ne1 + r) * ne0 + c] = src_row[c];
                    });
            }
        } else if (ggml_is_contiguously_allocated(V) && !v_interleaved) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
            to_fp16(V_data, V_f16.ptr, ggml_nelements(V), stream);
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
            int64_t s01, s02, s03;
            if (v_interleaved) {
                const int64_t blk_per_row = (int64_t)V->ne[0] / bs;
                s01 = (int64_t)V->ne[2] * blk_per_row;
                s02 = blk_per_row;
                s03 = (int64_t)V->ne[1] * s01;
            } else {
                s01 = (int64_t)V->nb[1] / ts;
                s02 = (int64_t)V->nb[2] / ts;
                s03 = (int64_t)V->nb[3] / ts;
            }
            to_fp16(V_data, V_f16.ptr,
                    V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                    s01, s02, s03, stream);
        }
    }

    // --- Step 3: Copy Q to dense F16 ---
    ggml_sycl_pool_alloc<sycl::half> Qf(ctx.pool(), (size_t) H * q * d);
    {
        const char * Q_data = (const char *)Q->data;
        const int64_t n = H * q * d;
        sycl::half * Qf_ptr = Qf.get();
        size_t Q_nb1 = Q->nb[1];
        size_t Q_nb2 = Q->nb[2];
        size_t Q_nb3 = Q->nb[3];
        stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
            const int64_t gid = ix[0];
            int64_t       i   = gid;
            const int64_t i0 = i % d; i /= d;
            const int64_t i1 = i % q; i /= q;
            const int64_t i2 = i % H;
            const int64_t i3 = i / H;
            const float * p = (const float *) (Q_data + i1 * Q_nb1 + i2 * Q_nb2 + i3 * Q_nb3) + i0;
            Qf_ptr[gid] = (sycl::half) (*p);
        });
    }

    // --- Step 4: Build/run oneDNN SDPA ---
    const sycl::half scale_h = (sycl::half) (1.0f / kq_scale);
    ggml_sycl_pool_alloc<sycl::half> scbuf(ctx.pool(), 1);
    stream->memcpy(scbuf.get(), &scale_h, sizeof(sycl::half));

    ggml_sycl_pool_alloc<sycl::half> outf(ctx.pool(), (size_t) H * q * d);

    // Compile once per (device, shape), reuse across layers.
    static std::unordered_map<std::string, sdpa_partition> cache;
    char keyb[96];
    snprintf(keyb, sizeof(keyb), "%d:%lld:%lld:%lld:%lld:%lld", ggml_sycl_get_device(),
             (long long) H, (long long) Hkv, (long long) q, (long long) seq, (long long) d);
    auto it = cache.find(keyb);
    if (it == cache.end()) {
        it = cache.emplace(keyb, build_sdpa(eng, (int) H, (int) Hkv, (int) q, (int) seq, (int) d)).first;
    }
    sdpa_partition & E = it->second;
    if (!E.ok) {
        // Partition failed — fall back to MKL
        ggml_sycl_flash_attn_ext_mkl(ctx, dst);
        return;
    }

    auto id2ptr = [&](size_t r) -> void * {
        if (r == E.id_q)     return Qf.get();
        if (r == E.id_k)     return K_f16.ptr;
        if (r == E.id_v)     return V_f16.ptr;
        if (r == E.id_scale) return scbuf.get();
        if (r == E.id_mask)  return (void *) mask->data;
        return nullptr;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    for (auto & lt : E.ins) {
        ti.emplace_back(lt, eng, id2ptr(lt.get_id()));
    }
    tensor to(E.out, eng, outf.get());
    E.cp.execute(strm, ti, {to});

    // --- Step 5: Permute SDPA output → ggml dst ---
    const int64_t n = mb * H * q * d;
    float * dst_data = (float *) dst->data;
    const sycl::half * out_data = outf.get();
    const size_t dst_nb0 = sizeof(float);
    const size_t dst_nb1 = dst->nb[1];
    const size_t dst_nb2 = dst->nb[2];
    const size_t dst_nb3 = dst->nb[3];
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        const int64_t gid = ix[0];
        int64_t       i   = gid;
        const int64_t e = i % d; i /= d;
        const int64_t t = i % q; i /= q;
        const int64_t h = i % H; const int64_t b = i / H;
        int64_t off = (e * dst_nb0 + h * dst_nb1 + t * dst_nb2 + b * dst_nb3) / sizeof(float);
        dst_data[off] = (float) out_data[gid];
    });

    // Wait for all GPU work to finish before pool allocs go out of scope.
    // Without this, the pool frees and reuses memory while the GPU is still
    // reading it, corrupting the output.
    stream->wait();
}
catch (const std::exception & e) {
    GGML_LOG_WARN("%s: hybrid SDPA failed (%s); falling back to MKL\n", __func__, e.what());
    ggml_sycl_flash_attn_ext_mkl(ctx, dst);
}

#else  // !GGML_SYCL_DNNL

void ggml_sycl_flash_attn_ext_hybrid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    // oneDNN not compiled — fall back to MKL
    ggml_sycl_flash_attn_ext_mkl(ctx, dst);
}

#endif // GGML_SYCL_DNNL
