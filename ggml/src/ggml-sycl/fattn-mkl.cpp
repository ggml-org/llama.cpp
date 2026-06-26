//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Flash attention via oneMKL GEMM (XMX-accelerated).
// Uses column_major::gemm for Q*K^T and S*V matmuls
// with an online softmax SYCL kernel.
//
// All GQA query heads sharing a KV head are batched into single
// GEMM calls, amortizing MKL launch overhead across K and V reuse.
//

#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-buffers.hpp"
#include "convert.hpp"
#include "fattn.hpp"

#include <oneapi/mkl.hpp>
#include <cstdio>
#include <cfloat>
#include <chrono>

#define MKL_FA_CHUNK_SIZE_KV 8192

using oneapi::mkl::transpose;
using oneapi::mkl::blas::column_major::gemm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack all GQA Q heads for one KV head into fp16, applying q_scale.
// One ND-range over gqa * n_queries * DKQ elements — each work-item
// computes its GQA group from the global index.
static void mkl_fa_pack_q_fp16(
    dpct::queue_ptr stream,
    sycl::half * __restrict dst,
    const float * __restrict q_src,
    int n_queries, int n_query_rows, int DKQ,
    int gqa_ratio, int kvh_base_head,
    float q_scale, int64_t wg_size) {

    const int64_t q_per_head = (int64_t)n_queries * DKQ;
    const int64_t total      = (int64_t)n_query_rows * DKQ;
    const int64_t wg = ((total + wg_size - 1) / wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
            [=](sycl::nd_item<1> item) {
                int64_t e = item.get_global_id(0);
                if (e >= total) return;

                int     iqg = (int)(e / q_per_head);
                int64_t off = e - (int64_t)iqg * q_per_head;
                int     iqh = kvh_base_head + iqg;

                dst[e] = sycl::half(
                    q_src[(int64_t)iqh * q_per_head + off] * q_scale);
            });
    });
}

// Zero-initialize the online softmax state arrays.
// KQ_max → -inf, KQ_sum → 0, VKQ_accum → 0.
// Merged into one kernel to avoid per-array launch overhead.
static void mkl_fa_init_softmax_state(
    dpct::queue_ptr stream,
    float * kmax, float * ksum, float * vacc,
    int n_query_rows, int DV, int64_t wg_size) {

    const float    neg_inf   = -1e30f;
    const int64_t  n_maxsum  = n_query_rows;
    const int64_t  n_vacc    = (int64_t)n_query_rows * DV;
    const int64_t  total     = (n_vacc > n_maxsum) ? n_vacc : n_maxsum;
    const int64_t  wg = ((total + wg_size - 1) / wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
            [=](sycl::nd_item<1> item) {
                int64_t i = item.get_global_id(0);
                if (i < n_maxsum) {
                    kmax[i] = neg_inf;
                    ksum[i] = 0.0f;
                }
                if (i < n_vacc) {
                    vacc[i] = 0.0f;
                }
            });
    });
}

// Online softmax over one KV chunk for all GQA query rows.
// For each row: find local max → rescale previous VKQ_accum →
// compute exp(s - max) → write S_f16 → update running max/sum.
// Runs per-row independently (rows don't interact).
static void mkl_fa_online_softmax_chunk(
    dpct::queue_ptr stream,
    float * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    float * __restrict KQ_max,
    float * __restrict KQ_sum,
    float * __restrict VKQ_accum,
    int n_query_rows, int n_queries, int DV,
    int chunk_size, int chunk_start,
    int kvh_head, int gqa_ratio,
    const sycl::half * mask_data, int64_t mask_head_stride,
    int64_t mask_row_stride, int mask_n_heads,
    float logit_softcap, int64_t wg_size) {

    const int64_t wg = ((n_query_rows + wg_size - 1) / wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
            [=](sycl::nd_item<1> item) {
                int jc = item.get_global_id(0);
                if (jc >= n_query_rows) return;

                const int gqa_group = jc / n_queries;
                const int q_row     = jc % n_queries;

                const float * __restrict KQ_row = KQ_f32
                    + jc * (int64_t)chunk_size;
                float * __restrict vkq = VKQ_accum
                    + jc * (int64_t)DV;

                // Per-GQA-group mask resolution
                const sycl::half * mask_h = nullptr;
                int64_t m_stride = 0;
                if (mask_data) {
                    int m_head = (mask_n_heads > 1)
                        ? (kvh_head + gqa_group) : 0;
                    mask_h   = mask_data + (int64_t)m_head * mask_head_stride;
                    m_stride = mask_row_stride;
                }

                // Row-wise local maximum
                float local_max = -1e30f;
                for (int i = 0; i < chunk_size; i++) {
                    float s = KQ_row[i];
                    if (mask_h) {
                        s += (float)mask_h[q_row * m_stride
                            + (chunk_start + i)];
                    }
                    if (s > local_max) local_max = s;
                }

                // Rescale previous accumulator by exp(old_max - new_max)
                float old_max = KQ_max[jc];
                float new_max = (old_max > local_max) ? old_max : local_max;
                float rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::exp(old_max - new_max);

                for (int v = 0; v < DV; v++) {
                    vkq[v] *= rescale;
                }

                // Softmax and write S_f16
                float local_sum = 0.0f;
                sycl::half * __restrict S_row = S_f16
                    + jc * (int64_t)chunk_size;

                for (int i = 0; i < chunk_size; i++) {
                    float s = KQ_row[i];
                    if (mask_h) {
                        s += (float)mask_h[q_row * m_stride
                            + (chunk_start + i)];
                    }
                    if (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    float val = sycl::native::exp(s - new_max);
                    S_row[i] = sycl::half(val);
                    local_sum += val;
                }

                KQ_sum[jc] = KQ_sum[jc] * rescale + local_sum;
                KQ_max[jc] = new_max;
            });
    });
}

// Write one GQA group's normalized output to its destination head.
static void mkl_fa_normalize_head(
    dpct::queue_ptr stream,
    float * __restrict dst_batch,
    const float * __restrict VKQ_accum,
    const float * __restrict KQ_sum,
    int iqh, int n_queries, int DV,
    int64_t src_offset, int64_t dst_row_stride,
    int64_t wg_size) {

    const int64_t wg = ((n_queries + wg_size - 1) / wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
            [=](sycl::nd_item<1> item) {
                int jc = item.get_global_id(0);
                if (jc >= n_queries) return;

                int     ksum_idx = (int)(src_offset / DV) + jc;
                float   inv_sum  = 1.0f / KQ_sum[ksum_idx];
                const float * __restrict src = VKQ_accum
                    + src_offset + jc * (int64_t)DV;
                float * __restrict dst_row = dst_batch
                    + iqh * n_queries * (int64_t)dst_row_stride
                    + jc * dst_row_stride;

                for (int v = 0; v < DV; v++) {
                    dst_row[v] = src[v] * inv_sum;
                }
            });
    });
}

// ---------------------------------------------------------------------------
// MKL Flash Attention orchestrator
//
// Pipeline: dequantize K/V → for each KV head:
//   pack GQA Q heads → MKL GEMM KQ → online softmax →
//   MKL GEMM VKQ → accumulate → normalize → scatter to dst
// ---------------------------------------------------------------------------
void ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    // dst->src[4] = sinks (not yet supported)
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *)KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *)KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *)KQV->op_params + 2, sizeof(float));

    const bool use_logit_softcap = (logit_softcap != 0.0f);
    const float q_scale = use_logit_softcap ? (scale / logit_softcap) : scale;

    // --- Dimensions ---
    const int DKQ        = (int)K->ne[0];
    const int DV         = (int)V->ne[0];
    const int n_queries  = (int)Q->ne[1];
    const int n_q_heads  = (int)Q->ne[2];
    const int n_kv_heads = (int)K->ne[2];
    const int n_batch    = (int)Q->ne[3];
    const int n_kv       = (int)K->ne[1];
    const int gqa_ratio  = n_q_heads / n_kv_heads;
    const int n_query_rows = n_queries * gqa_ratio;

    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    GGML_ASSERT(max_bias == 0.0f);  // ALiBi not supported
    GGML_ASSERT(Q->ne[3] == K->ne[3] || K->ne[3] == 1);

    const int chunk_size = std::min(MKL_FA_CHUNK_SIZE_KV, n_kv);
    const int64_t wg_size = 256;

    // --- Debug output (gated by MKL_FA_DEBUG=1) ---
    static int mkl_call_count = 0;
    mkl_call_count++;
    static int mkl_debug = -1;
    if (mkl_debug < 0) {
        const char * e = getenv("MKL_FA_DEBUG");
        mkl_debug = (e && e[0] == '1') ? 1 : 0;
    }
    const bool do_print = (mkl_debug == 1);

    if (do_print) {
        fprintf(stderr, "[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d KQ_buf=%.1fMB\n",
                mkl_call_count, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_ratio, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_size,
                (double)((int64_t)n_query_rows * chunk_size * sizeof(float))
                    / (1024.0 * 1024.0));
        fflush(stderr);
    }

    // --- Stream and allocators ---
    dpct::queue_ptr stream = ctx.stream();
    stream->wait();  // drain pending work before MKL operations

    ggml_sycl_pool & pool = ctx.pool();
    ggml_sycl_fattn_kv_buffers & fbuf = ctx.fattn_buffers();

    // Timing helpers (local to this function)
#define MKL_TAKE_TIME(t0)  auto t0 = std::chrono::steady_clock::now()
#define MKL_ACCUM(acc, t0) acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count()

    int64_t gemm_kq_time_us  = 0;
    int64_t gemm_vkq_time_us = 0;
    int64_t softmax_time_us  = 0;
    int64_t dequant_time_us  = 0;

    MKL_TAKE_TIME(t_deq);

    // --- Dequantize K, V to fp16 ---
    ggml_sycl_fattn_alloc K_f16(fbuf.K);
    ggml_sycl_fattn_alloc V_f16(fbuf.V);

    const char * K_data = (const char *)K->data;
    const char * V_data = (const char *)V->data;

    // Strides in the fp16 dequant buffer (row-major within each head)
    size_t k_row_stride  = DKQ * sizeof(sycl::half);
    size_t k_head_stride = (size_t)n_kv * k_row_stride;
    size_t v_row_stride  = DV * sizeof(sycl::half);
    size_t v_head_stride = (size_t)n_kv * v_row_stride;

    const bool V_is_K_view = V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    if (K->type != GGML_TYPE_F16) {
        K_f16.alloc(ggml_nelements(K));
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K->data, K_f16.ptr, ggml_nelements(K), stream);
        } else {
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            const int64_t s01 = K->nb[1] / ggml_type_size(K->type);
            const int64_t s02 = K->nb[2] / ggml_type_size(K->type);
            const int64_t s03 = K->nb[3] / ggml_type_size(K->type);
            to_fp16(K->data, K_f16.ptr,
                    K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                    s01, s02, s03, stream);
        }
        K_data = (char *)K_f16.ptr;
    } else {
        k_row_stride  = K->nb[1];
        k_head_stride = K->nb[2];
    }

    if (V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data        = K_data;
            v_row_stride  = k_row_stride;
            v_head_stride = k_head_stride;
        } else {
            V_f16.alloc(ggml_nelements(V));
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V->data, V_f16.ptr, ggml_nelements(V), stream);
            } else {
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                const int64_t s01 = V->nb[1] / ggml_type_size(V->type);
                const int64_t s02 = V->nb[2] / ggml_type_size(V->type);
                const int64_t s03 = V->nb[3] / ggml_type_size(V->type);
                to_fp16(V->data, V_f16.ptr,
                        V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        s01, s02, s03, stream);
            }
            V_data = (char *)V_f16.ptr;
        }
    } else {
        if (V_is_K_view) {
            V_data        = K_data;
            v_row_stride  = k_row_stride;
            v_head_stride = k_head_stride;
        } else {
            v_row_stride  = V->nb[1];
            v_head_stride = V->nb[2];
        }
    }

    stream->wait();  // dequant complete
    MKL_ACCUM(dequant_time_us, t_deq);

    // --- Resolve mask pointers once (valid for all chunks) ---
    const sycl::half * mask_data = nullptr;
    int64_t mask_head_stride = 0;
    int64_t mask_row_stride  = 0;
    int     mask_n_heads     = 0;
    if (mask) {
        // Mask is resolved per-batch below; store strides here
        mask_head_stride = mask->nb[2] / sizeof(sycl::half);
        mask_row_stride  = mask->nb[1] / sizeof(sycl::half);
        mask_n_heads     = (int)mask->ne[2];
    }

    // --- Allocate intermediates (sized for all GQA query rows) ---
    ggml_sycl_pool_alloc<float>      KQ_f32(pool);
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);
    ggml_sycl_pool_alloc<float>      VKQ_chunk(pool);
    ggml_sycl_pool_alloc<float>      VKQ_accum(pool);
    ggml_sycl_pool_alloc<float>      KQ_max(pool);
    ggml_sycl_pool_alloc<float>      KQ_sum(pool);
    ggml_sycl_pool_alloc<sycl::half> Q_head_f16(pool);

    KQ_f32.alloc((size_t)n_query_rows * chunk_size);
    S_f16.alloc((size_t)n_query_rows * chunk_size);
    VKQ_chunk.alloc((size_t)n_query_rows * DV);
    VKQ_accum.alloc((size_t)n_query_rows * DV);
    KQ_max.alloc(n_query_rows);
    KQ_sum.alloc(n_query_rows);
    Q_head_f16.alloc((size_t)n_query_rows * DKQ);

    // Extract raw pointers — ggml_sycl_pool_alloc is not device-copyable
    sycl::half * Q_head_f16_ptr = Q_head_f16.ptr;
    float      * KQ_f32_ptr     = KQ_f32.ptr;
    sycl::half * S_f16_ptr      = S_f16.ptr;
    float      * VKQ_chunk_ptr  = VKQ_chunk.ptr;
    float      * VKQ_accum_ptr  = VKQ_accum.ptr;
    float      * KQ_max_ptr     = KQ_max.ptr;
    float      * KQ_sum_ptr     = KQ_sum.ptr;

    const int64_t dst_row_stride = KQV->nb[1] / sizeof(float);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    for (int ib = 0; ib < n_batch; ib++) {
        const float * Q_batch = (const float *)Q->data
            + ib * (Q->nb[3] / sizeof(float));
        float * dst_batch = (float *)KQV->data
            + ib * (KQV->nb[3] / sizeof(float));

        // Per-batch mask base
        const sycl::half * mask_batch = nullptr;
        if (mask) {
            int m_batch = (mask->ne[3] > 1) ? ib : 0;
            mask_batch = (const sycl::half *)mask->data
                + m_batch * (mask->nb[3] / sizeof(sycl::half));
        }

        for (int ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            int kvh_base_head = ikvh * gqa_ratio;

            // 1. Pack all GQA Q heads into fp16
            mkl_fa_pack_q_fp16(stream,
                Q_head_f16_ptr, Q_batch,
                n_queries, n_query_rows, DKQ,
                gqa_ratio, kvh_base_head,
                q_scale, wg_size);

            // 2. Initialize softmax state
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_sum_ptr, VKQ_accum_ptr,
                n_query_rows, DV, wg_size);

            // Sync before MKL GEMM (MKL may use an internal queue)
            stream->wait();

            // K/V base for this KV head
            const sycl::half * K_head = (const sycl::half *)K_data
                + ikvh * (k_head_stride / sizeof(sycl::half));
            const sycl::half * V_head = (const sycl::half *)V_data
                + ikvh * (v_head_stride / sizeof(sycl::half));

            // 3. Chunked KV loop
            for (int chunk_start = 0; chunk_start < n_kv; chunk_start += chunk_size) {
                int this_chunk = std::min(chunk_size, n_kv - chunk_start);
                const sycl::half * K_chunk = K_head
                    + (int64_t)chunk_start * DKQ;
                const sycl::half * V_chunk = V_head
                    + (int64_t)chunk_start * DV;

                // 3a. GEMM: KQ = Q_batched × K_chunk^T
                {
                    MKL_TAKE_TIME(t0);
                    sycl::event ev = gemm(*stream,
                        transpose::trans, transpose::nontrans,
                        this_chunk, n_query_rows, DKQ,
                        alpha,
                        K_chunk, DKQ,
                        Q_head_f16_ptr, DKQ,
                        beta,
                        KQ_f32_ptr, this_chunk);
                    stream->wait();
                    try { ev.wait_and_throw(); } catch (sycl::exception & e) {
                        fprintf(stderr, "[MKL-FA] GEMM KQ: %s\n", e.what());
                        GGML_ABORT("MKL GEMM KQ failed");
                    }
                    MKL_ACCUM(gemm_kq_time_us, t0);
                }

                // 3b. Online softmax over this chunk
                {
                    MKL_TAKE_TIME(t0);
                    mkl_fa_online_softmax_chunk(stream,
                        KQ_f32_ptr, S_f16_ptr,
                        KQ_max_ptr, KQ_sum_ptr, VKQ_accum_ptr,
                        n_query_rows, n_queries, DV,
                        this_chunk, chunk_start,
                        kvh_base_head, gqa_ratio,
                        mask_batch, mask_head_stride,
                        mask_row_stride, mask_n_heads,
                        logit_softcap, wg_size);
                    stream->wait();  // S_f16 must be ready for GEMM
                    MKL_ACCUM(softmax_time_us, t0);
                }

                // 3c. GEMM: VKQ_chunk = S × V_chunk
                {
                    MKL_TAKE_TIME(t0);
                    sycl::event ev = gemm(*stream,
                        transpose::nontrans, transpose::nontrans,
                        DV, n_query_rows, this_chunk,
                        alpha,
                        V_chunk, DV,
                        S_f16_ptr, this_chunk,
                        beta,
                        VKQ_chunk_ptr, DV);
                    stream->wait();
                    try { ev.wait_and_throw(); } catch (sycl::exception & e) {
                        fprintf(stderr, "[MKL-FA] GEMM VKQ: %s\n", e.what());
                        GGML_ABORT("MKL GEMM VKQ failed");
                    }
                    MKL_ACCUM(gemm_vkq_time_us, t0);
                }

                // 3d. VKQ_accum += VKQ_chunk
                {
                    const int64_t n_total = (int64_t)n_query_rows * DV;
                    const int64_t wg = ((n_total + wg_size - 1) / wg_size)
                        * wg_size;
                    stream->submit([&](sycl::handler & cgh) {
                        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
                            [=](sycl::nd_item<1> item) {
                                int64_t i = item.get_global_id(0);
                                if (i < n_total) {
                                    VKQ_accum_ptr[i] += VKQ_chunk_ptr[i];
                                }
                            });
                    });
                    // In-order queue: accumulate finishes before next
                    // chunk's GEMM or final normalize.
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            for (int iqg = 0; iqg < gqa_ratio; iqg++) {
                int     iqh        = kvh_base_head + iqg;
                int64_t src_offset = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_head(stream,
                    dst_batch, VKQ_accum_ptr, KQ_sum_ptr,
                    iqh, n_queries, DV,
                    src_offset, dst_row_stride, wg_size);
            }
            // In-order queue: normalize writes ordered before next KV head.
        }
    }

#undef MKL_TAKE_TIME
#undef MKL_ACCUM

    if (do_print) {
        fprintf(stderr, "[MKL-FA] #%d n_kv=%d n_q=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld\n",
                mkl_call_count, n_kv, n_queries,
                (long long)dequant_time_us,
                (long long)gemm_kq_time_us,
                (long long)softmax_time_us,
                (long long)gemm_vkq_time_us);
        fflush(stderr);
    }
}
