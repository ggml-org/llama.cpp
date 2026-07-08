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
#include <chrono>

#define MKL_FA_CHUNK_SIZE_KV 8192

// Number of query rows processed per tile. The score buffers (KQ_f32, S_f16)
// are sized q_tile_rows * chunk_size, so this bounds their footprint
// regardless of batch size (n_query_rows = n_queries * gqa_ratio). A typical
// single-ubatch prefill (e.g. ubatch 1024 * gqa 8 = 8192 rows) is exactly one
// tile, so it runs with no extra iterations. Larger batches tile and stay
// bounded. Override with GGML_SYCL_MKL_FA_Q_TILE.
#define MKL_FA_Q_TILE 8192

using oneapi::mkl::transpose;
using oneapi::mkl::blas::column_major::gemm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack all GQA Q heads for one KV head into fp16, applying q_scale.
// Launches one kernel per GQA group — each kernel copies exactly
// n_queries * DKQ elements using the per-group dst offset and
// per-head source stride.
static void mkl_fa_pack_q_fp16(
    dpct::queue_ptr stream,
    sycl::half * __restrict dst,
    const float * __restrict q_src,
    int n_queries, int n_query_rows, int DKQ,
    int gqa_ratio, int kvh_base_head,
    float q_scale, int64_t q_row_stride, int64_t q_head_stride,
    int64_t wg_size) {

    for (int iqg = 0; iqg < gqa_ratio; iqg++) {
        int     iqh       = kvh_base_head + iqg;
        sycl::half * dst_g = dst + (int64_t)iqg * n_queries * DKQ;

        const int64_t n_elem = (int64_t)n_queries * DKQ;
        const int64_t wg = ((n_elem + wg_size - 1) / wg_size) * wg_size;

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
                [=](sycl::nd_item<1> item) {
                    int64_t e = item.get_global_id(0);
                    if (e >= n_elem) return;

                    int64_t q = e / DKQ;
                    int64_t d = e - q * DKQ;

                    // Stride-aware source offset: handles permuted,
                    // sliced, or contiguous Q tensor layouts.
                    int64_t src_off = d
                        + q * q_row_stride
                        + (int64_t)iqh * q_head_stride;

                    dst_g[e] = sycl::half(
                        q_src[src_off] * q_scale);
                });
        });
    }
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

// Online softmax over one KV chunk for a tile of GQA query rows.
// The tile spans absolute rows [q0, q0 + q_rows). Score buffers
// (KQ_f32/S_f16) are indexed RELATIVE to the tile; the persistent state
// (VKQ_accum/KQ_max/KQ_sum) and mask are indexed by ABSOLUTE row.
// For each row: find local max → rescale previous VKQ_accum →
// compute exp(s - max) → write S_f16 → update running max/sum.
static void mkl_fa_online_softmax_chunk(
    dpct::queue_ptr stream,
    float * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    float * __restrict KQ_max,
    float * __restrict KQ_sum,
    float * __restrict VKQ_accum,
    int q0, int q_rows, int n_queries, int DV,
    int chunk_size, int chunk_start,
    int kvh_head, int gqa_ratio,
    const sycl::half * mask_data, int64_t mask_head_stride,
    int64_t mask_row_stride, int mask_n_heads,
    float logit_softcap, int64_t wg_size) {

    const int64_t wg = ((q_rows + wg_size - 1) / wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
            [=](sycl::nd_item<1> item) {
                int jc_rel = item.get_global_id(0);
                if (jc_rel >= q_rows) return;
                int jc_abs = q0 + jc_rel;

                const int gqa_group = jc_abs / n_queries;
                const int q_row     = jc_abs % n_queries;

                // Score buffers are tile-local (relative index).
                const float * __restrict KQ_row = KQ_f32
                    + jc_rel * (int64_t)chunk_size;
                // Persistent accumulator is full-sized (absolute index).
                float * __restrict vkq = VKQ_accum
                    + jc_abs * (int64_t)DV;

                const sycl::half * mask_h = nullptr;
                int64_t m_stride = 0;
                if (mask_data) {
                    int m_head = (mask_n_heads > 1)
                        ? (kvh_head + gqa_group) : 0;
                    mask_h   = mask_data + (int64_t)m_head * mask_head_stride;
                    m_stride = mask_row_stride;
                }

                // Row-wise local maximum (softcap before mask)
                float local_max = -1e30f;
                for (int i = 0; i < chunk_size; i++) {
                    float s = KQ_row[i];
                    if (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    if (mask_h) {
                        s += (float)mask_h[q_row * m_stride
                            + (chunk_start + i)];
                    }
                    if (s > local_max) local_max = s;
                }

                // Rescale previous accumulator by exp(old_max - new_max)
                float old_max = KQ_max[jc_abs];
                float new_max = (old_max > local_max) ? old_max : local_max;
                float rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::exp(old_max - new_max);

                for (int v = 0; v < DV; v++) {
                    vkq[v] *= rescale;
                }

                // Softmax and write S_f16 (tile-local index)
                float local_sum = 0.0f;
                sycl::half * __restrict S_row = S_f16
                    + jc_rel * (int64_t)chunk_size;

                for (int i = 0; i < chunk_size; i++) {
                    float s = KQ_row[i];
                    if (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    if (mask_h) {
                        s += (float)mask_h[q_row * m_stride
                            + (chunk_start + i)];
                    }
                    float val = sycl::native::exp(s - new_max);
                    S_row[i] = sycl::half(val);
                    local_sum += val;
                }

                KQ_sum[jc_abs] = KQ_sum[jc_abs] * rescale + local_sum;
                KQ_max[jc_abs] = new_max;
            });
    });
}

// Write one GQA group's normalized output to its destination head.
static void mkl_fa_normalize_head(
    dpct::queue_ptr stream,
    float * __restrict dst_batch,
    const float * __restrict VKQ_accum,
    const float * __restrict KQ_sum,
    int iqh, int n_queries, int DV, int n_q_heads,
    int64_t src_offset, int64_t wg_size) {

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
                // Interleaved dst layout (matching TILE):
                // rows alternate between heads, then increment query.
                // offset = (query * n_q_heads + head) * DV
                float * __restrict dst_row = dst_batch
                    + ((int64_t)jc * n_q_heads + iqh) * (int64_t)DV;

                for (int v = 0; v < DV; v++) {
                    dst_row[v] = src[v] * inv_sum;
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Per-chunk dequant
//
// Rather than dequantizing all of K/V up front (footprint scales with
// context), we dequant one KV-head chunk at a time into a dense
// [this_chunk x D] fp16 buffer (row-major, lda = D). The source address of
// element (head=ikvh, row=chunk_start+r, col=c) decomposes into independent
// linear terms head_off(ikvh) + row_off(chunk_start) + (r,c), so slicing a
// chunk is a clean pointer offset in every layout case. The true-Gemma-
// interleave vs padded-seq-view distinction is resolved once when the
// descriptor is built; slicing does not reintroduce it.
// ---------------------------------------------------------------------------
struct mkl_fa_kv_desc {
    const char * data = nullptr;
    ggml_type    type = GGML_TYPE_F16;
    int64_t      D    = 0;      // ne[0]
    int64_t      nb1  = 0;      // byte stride, seq dim
    int64_t      nb2  = 0;      // byte stride, head dim
    int          mode = 0;      // 0=f16 dense, 1=f16 interleaved, 2=quant contig, 3=quant nc
    int64_t      ts   = 0;      // type size (mode 3 base offset)
    int64_t      s01  = 0;      // nc row stride in blocks (mode 3)
    int64_t      s02  = 0;      // nc head stride in blocks (mode 3)
};

static mkl_fa_kv_desc mkl_fa_make_desc(const ggml_tensor * T, bool interleaved, int n_kv_heads) {
    mkl_fa_kv_desc d;
    d.data = (const char *)T->data;
    d.type = T->type;
    d.D    = T->ne[0];
    d.nb1  = (int64_t)T->nb[1];
    d.nb2  = (int64_t)T->nb[2];
    d.ts   = (int64_t)ggml_type_size(T->type);

    if (T->type == GGML_TYPE_F16) {
        d.mode = interleaved ? 1 : 0;
    } else if (ggml_is_contiguously_allocated(T) && !interleaved) {
        d.mode = 2;
    } else {
        d.mode = 3;
        const int64_t bs          = (int64_t)ggml_blck_size(T->type);
        const int64_t blk_per_row = T->ne[0] / bs;
        // True Gemma interleave packs heads within a row (nb[2] < ne[1]*nb[1])
        // → reconstruct physical strides. Padded seq-views (nb[2] > ne[1]*nb[1])
        // already have correct physical strides.
        const bool gemma = interleaved &&
            ((int64_t)T->nb[2] < (int64_t)T->ne[1] * (int64_t)T->nb[1]);
        if (gemma) {
            d.s01 = (int64_t)n_kv_heads * blk_per_row;
            d.s02 = blk_per_row;
        } else {
            d.s01 = d.nb1 / d.ts;
            d.s02 = d.nb2 / d.ts;
        }
    }
    return d;
}

// Dequant one KV-head chunk into a dense [this_chunk x D] fp16 buffer.
static void mkl_fa_dequant_chunk(
    dpct::queue_ptr stream, const mkl_fa_kv_desc & d, ggml_tensor * dst_ctx,
    sycl::half * out, int ikvh, int chunk_start, int this_chunk) {

    const int64_t D = d.D;
    switch (d.mode) {
        case 0: {  // F16 dense
            const char * base = d.data + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            stream->memcpy(out, base, (size_t)this_chunk * D * sizeof(sycl::half));
            break;
        }
        case 1: {  // F16 interleaved (Gemma or padded seq-view)
            const char * base = d.data + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            const int64_t row_halfs = d.nb1 / (int64_t)sizeof(sycl::half);
            const sycl::half * src = (const sycl::half *)base;
            stream->parallel_for(
                sycl::range<2>((size_t)this_chunk, (size_t)D),
                [=](sycl::item<2> it) {
                    int64_t r = it.get_id(0);
                    int64_t c = it.get_id(1);
                    out[r * D + c] = src[r * row_halfs + c];
                });
            break;
        }
        case 2: {  // quantized contiguous
            const char * base = d.data + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(d.type, dst_ctx);
            to_fp16(base, out, (int64_t)this_chunk * D, stream);
            break;
        }
        default: {  // 3: quantized non-contiguous
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(d.type);
            const int64_t base_blocks = (int64_t)ikvh * d.s02
                + (int64_t)chunk_start * d.s01;
            const char * base = d.data + base_blocks * d.ts;
            // ne02 = ne03 = 1 → s02/s03 inert; head+chunk offset carried by base.
            to_fp16(base, out, D, this_chunk, 1, 1, d.s01, d.s02, d.s02, stream);
            break;
        }
    }
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
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *)KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *)KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *)KQV->op_params + 2, sizeof(float));

    const float q_scale = scale;

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

    // Query rows are processed in tiles of q_tile_rows so the score buffers
    // (KQ_f32/S_f16 = q_tile_rows * chunk_size) stay bounded regardless of
    // batch size. n_query_rows <= Q_TILE is a single tile (no extra work).
    static int q_tile_env = -1;
    if (q_tile_env < 0) {
        q_tile_env = ggml_sycl_get_env("GGML_SYCL_MKL_FA_Q_TILE", MKL_FA_Q_TILE);
    }
    const int q_tile_rows = std::max(1, std::min(q_tile_env, n_query_rows));

    const int64_t wg_size = 256;

    // --- Debug output (gated by GGML_SYCL_MKL_FA_DEBUG=1) ---
    static int mkl_call_count = 0;
    mkl_call_count++;
    static int mkl_debug = -1;
    if (mkl_debug < 0) {
        mkl_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    }
    const bool do_print = (mkl_debug == 1);

    const int64_t q_row_stride  = Q->nb[1] / sizeof(float);
    const int64_t q_head_stride = Q->nb[2] / sizeof(float);

    const bool V_is_K_view = V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    // Early interleaved detection for debug output.
    // True interleaved detection happens after dequant (nb12_fp16 == nb11_fp16),
    // but we can pre-detect on the original tensor strides.
    const bool k_early_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]);
    const bool v_early_interleaved =
        !V_is_K_view && ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]);

    if (do_print) {
        GGML_LOG_INFO("[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d buf=%.1fMB%s%s\n",
                mkl_call_count, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_ratio, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_size,
                (double)((int64_t)n_query_rows * chunk_size * sizeof(float))
                    / (1024.0 * 1024.0),
                k_early_interleaved ? " K_ILV" : "",
                v_early_interleaved ? " V_ILV" : "");
        GGML_LOG_INFO("[MKL-FA] #%d Q-nb1=%lld Q-nb2=%lld "
                "q_rs=%lld q_hs=%lld dst_rs=%lld dst_hs=%lld\n",
                mkl_call_count,
                (long long)Q->nb[1], (long long)Q->nb[2],
                (long long)q_row_stride, (long long)q_head_stride,
                (long long)(KQV->nb[1] / sizeof(float)),
                (long long)(KQV->nb[2] / sizeof(float)));
    }

    // --- Stream and allocators ---
    dpct::queue_ptr stream = ctx.stream();

#define MKL_TAKE_TIME(t0)  auto t0 = std::chrono::steady_clock::now()
#define MKL_ACCUM(acc, t0) do { if (do_print) { \
    acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count(); \
} } while(0)

    int64_t gemm_kq_time_us  = 0;
    int64_t gemm_vkq_time_us = 0;
    int64_t softmax_time_us  = 0;
    int64_t dequant_time_us  = 0;

    MKL_TAKE_TIME(t_deq);

    // --- K/V dequant descriptors ---
    // Dequant is done per-chunk inside the KV loop (footprint independent of
    // context). Output is always dense row-major fp16 [this_chunk x D], lda=D.
    // Interleaved detection: ne[1]*nb[1] != nb[2] means heads are interleaved.
    const bool k_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
    const bool v_interleaved =
        ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

    const mkl_fa_kv_desc K_desc = mkl_fa_make_desc(K, k_interleaved, n_kv_heads);
    const mkl_fa_kv_desc V_desc = V_is_K_view
        ? K_desc : mkl_fa_make_desc(V, v_interleaved, n_kv_heads);

    MKL_ACCUM(dequant_time_us, t_deq);

    // --- Resolve mask pointers ---
    const sycl::half * mask_data = nullptr;
    int64_t mask_head_stride = 0;
    int64_t mask_row_stride  = 0;
    int     mask_n_heads     = 0;

    if (mask) {
        // Use actual fp16 device size (2 bytes), NOT sizeof(sycl::half)
        // which may be 4 on the host in oneAPI.
        mask_head_stride = mask->nb[2] / 2;
        mask_row_stride  = mask->nb[1] / 2;
        mask_n_heads     = (int)mask->ne[2];
    }

    // --- Allocate intermediates from pool ---
    ggml_sycl_pool & pool = ctx.pool();

    ggml_sycl_pool_alloc<float>      KQ_f32(pool);      // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);       // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<float>      VKQ_chunk(pool);   // [q_tile_rows x DV]
    ggml_sycl_pool_alloc<float>      VKQ_accum(pool);   // [n_query_rows x DV] (full)
    ggml_sycl_pool_alloc<float>      KQ_max(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<float>      KQ_sum(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<sycl::half> Q_head_f16(pool);  // [n_query_rows x DKQ] (full)
    ggml_sycl_pool_alloc<sycl::half> K_chunk_f16(pool); // [chunk x DKQ] (per-chunk dequant)
    ggml_sycl_pool_alloc<sycl::half> V_chunk_f16(pool); // [chunk x DV] (per-chunk dequant)

    KQ_f32.alloc((size_t)q_tile_rows * chunk_size);
    S_f16.alloc((size_t)q_tile_rows * chunk_size);
    VKQ_chunk.alloc((size_t)q_tile_rows * DV);
    VKQ_accum.alloc((size_t)n_query_rows * DV);
    KQ_max.alloc(n_query_rows);
    KQ_sum.alloc(n_query_rows);
    Q_head_f16.alloc((size_t)n_query_rows * DKQ);
    K_chunk_f16.alloc((size_t)chunk_size * DKQ);

    sycl::half * V_chunk_f16_ptr;
    if (V_is_K_view) {
        V_chunk_f16_ptr = K_chunk_f16.ptr;   // V aliases K (DV == DKQ)
    } else {
        V_chunk_f16.alloc((size_t)chunk_size * DV);
        V_chunk_f16_ptr = V_chunk_f16.ptr;
    }

    sycl::half * Q_head_f16_ptr  = Q_head_f16.ptr;
    float      * KQ_f32_ptr      = KQ_f32.ptr;
    sycl::half * S_f16_ptr       = S_f16.ptr;
    float      * VKQ_chunk_ptr   = VKQ_chunk.ptr;
    float      * VKQ_accum_ptr   = VKQ_accum.ptr;
    float      * KQ_max_ptr      = KQ_max.ptr;
    float      * KQ_sum_ptr      = KQ_sum.ptr;
    sycl::half * K_chunk_f16_ptr = K_chunk_f16.ptr;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    for (int ib = 0; ib < n_batch; ib++) {
        const float * Q_batch = (const float *)Q->data
            + ib * (Q->nb[3] / sizeof(float));
        float * dst_batch = (float *)KQV->data
            + ib * (KQV->nb[3] / sizeof(float));

        const sycl::half * mask_batch = nullptr;
        if (mask) {
            int m_batch = (mask->ne[3] > 1) ? ib : 0;
            mask_batch = (const sycl::half *)mask->data
                + m_batch * (mask->nb[3] / 2);  // 2 = actual fp16 device size
        }

        for (int ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            int kvh_base_head = ikvh * gqa_ratio;

            // 1. Pack all GQA Q heads into fp16 (full n_query_rows)
            mkl_fa_pack_q_fp16(stream,
                Q_head_f16_ptr, Q_batch,
                n_queries, n_query_rows, DKQ,
                gqa_ratio, kvh_base_head,
                q_scale, q_row_stride, q_head_stride, wg_size);

            // 2. Initialize softmax state (full n_query_rows)
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_sum_ptr, VKQ_accum_ptr,
                n_query_rows, DV, wg_size);

            // Sync before MKL GEMM (MKL may use an internal queue)
            stream->wait();

            // 3. KV chunk loop (OUTER): dequant each chunk once, then tile queries.
            for (int chunk_start = 0; chunk_start < n_kv; chunk_start += chunk_size) {
                int this_chunk = std::min(chunk_size, n_kv - chunk_start);

                // 3a. Dequant this KV chunk to dense fp16 (once per chunk)
                {
                    MKL_TAKE_TIME(t0);
                    mkl_fa_dequant_chunk(stream, K_desc, KQV,
                        K_chunk_f16_ptr, ikvh, chunk_start, this_chunk);
                    if (!V_is_K_view) {
                        mkl_fa_dequant_chunk(stream, V_desc, KQV,
                            V_chunk_f16_ptr, ikvh, chunk_start, this_chunk);
                    }
                    stream->wait();  // dequant must be ready before MKL GEMM
                    MKL_ACCUM(dequant_time_us, t0);
                }

                // 3b. Query tile loop (INNER) — bounds KQ_f32/S_f16 footprint.
                for (int q0 = 0; q0 < n_query_rows; q0 += q_tile_rows) {
                    int q_rows = std::min(q_tile_rows, n_query_rows - q0);

                    // GEMM: KQ = Q_tile × K_chunk^T
                    {
                        MKL_TAKE_TIME(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::trans, transpose::nontrans,
                            this_chunk, q_rows, DKQ,
                            alpha,
                            K_chunk_f16_ptr, DKQ,
                            Q_head_f16_ptr + (int64_t)q0 * DKQ, DKQ,
                            beta,
                            KQ_f32_ptr, this_chunk);
                        try { ev.wait_and_throw(); } catch (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM KQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM KQ failed");
                        }
                        MKL_ACCUM(gemm_kq_time_us, t0);
                    }
                    // Online softmax over this chunk for this query tile
                    {
                        MKL_TAKE_TIME(t0);
                        mkl_fa_online_softmax_chunk(stream,
                            KQ_f32_ptr, S_f16_ptr,
                            KQ_max_ptr, KQ_sum_ptr, VKQ_accum_ptr,
                            q0, q_rows, n_queries, DV,
                            this_chunk, chunk_start,
                            kvh_base_head, gqa_ratio,
                            mask_batch, mask_head_stride,
                            mask_row_stride, mask_n_heads,
                            logit_softcap, wg_size);
                        stream->wait();  // S_f16 must be ready for GEMM
                        MKL_ACCUM(softmax_time_us, t0);
                    }

                    // GEMM: VKQ_chunk = S × V_chunk
                    {
                        MKL_TAKE_TIME(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::nontrans, transpose::nontrans,
                            DV, q_rows, this_chunk,
                            alpha,
                            V_chunk_f16_ptr, DV,
                            S_f16_ptr, this_chunk,
                            beta,
                            VKQ_chunk_ptr, DV);
                        try { ev.wait_and_throw(); } catch (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM VKQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM VKQ failed");
                        }
                        MKL_ACCUM(gemm_vkq_time_us, t0);
                    }
                    // VKQ_accum[q0..] += VKQ_chunk
                    {
                        const int64_t n_total = (int64_t)q_rows * DV;
                        const int64_t wg = ((n_total + wg_size - 1) / wg_size)
                            * wg_size;
                        float * accum = VKQ_accum_ptr + (int64_t)q0 * DV;
                        stream->submit([&](sycl::handler & cgh) {
                            cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
                                [=](sycl::nd_item<1> item) {
                                    int64_t i = item.get_global_id(0);
                                    if (i < n_total) {
                                        accum[i] += VKQ_chunk_ptr[i];
                                    }
                                });
                        });
                    }
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            for (int iqg = 0; iqg < gqa_ratio; iqg++) {
                int     iqh        = kvh_base_head + iqg;
                int64_t src_offset = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_head(stream,
                    dst_batch, VKQ_accum_ptr, KQ_sum_ptr,
                    iqh, n_queries, DV, n_q_heads,
                    src_offset, wg_size);
            }
        }
    }

#undef MKL_TAKE_TIME
#undef MKL_ACCUM

    if (do_print) {
        const int64_t v_chunk_elems = V_is_K_view ? 0 : (int64_t)chunk_size * DV;
        double total_mb = (double)(
            (int64_t)q_tile_rows * chunk_size * sizeof(float)      // KQ_f32
          + (int64_t)q_tile_rows * chunk_size * sizeof(sycl::half) // S_f16
          + (int64_t)q_tile_rows * DV * sizeof(float)              // VKQ_chunk
          + (int64_t)n_query_rows * DV * sizeof(float)             // VKQ_accum
          + (int64_t)n_query_rows * sizeof(float)                  // KQ_max
          + (int64_t)n_query_rows * sizeof(float)                  // KQ_sum
          + (int64_t)n_query_rows * DKQ * sizeof(sycl::half)       // Q_head_f16
          + (int64_t)chunk_size * DKQ * sizeof(sycl::half)         // K_chunk_f16
          + v_chunk_elems * (int64_t)sizeof(sycl::half)            // V_chunk_f16
        ) / (1024.0 * 1024.0);
        GGML_LOG_INFO("[MKL-FA] #%d n_kv=%d n_q=%d q_tile=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld "
                "buf_mb=%.1f\n",
                mkl_call_count, n_kv, n_queries, q_tile_rows,
                (long long)dequant_time_us,
                (long long)gemm_kq_time_us,
                (long long)softmax_time_us,
                (long long)gemm_vkq_time_us,
                total_mb);
    }
}
