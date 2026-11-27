#include <HAP_farf.h>

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "dma-queue.h"
#include "hex-common.h"
#include "htp-trace.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "dsv4-hc-ops.h"

// ---------------------------------------------------------------------------
// DSV4_HC_PRE
// ---------------------------------------------------------------------------

// hvx_vec_fast_sigmoid_f32_guard_2it with fewer live constants and predicates
static inline HVX_Vector dsv4_hc_sigmoid_f32(HVX_Vector v, HVX_Vector max_exp, HVX_Vector min_exp) {
    v = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(v, min_exp), max_exp);
    v = Q6_Vqf32_vmpy_VsfVsf(v, hvx_vec_splat_f32(0.5f * 1.442695022f));

    HVX_Vector in_int = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_Vector r = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    for (int i = 0; i < 2; ++i) {
        const HVX_Vector rv = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r, v5));
        r = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vadd_VsfVsf(r, r), Q6_Vqf32_vmpy_VsfVsf(r, rv)));
    }

    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, r));
}

typedef void (*dsv4_hc_unpack_fn_t)(uint8_t * p);

static void dsv4_hc_unpack_noop(uint8_t * p) {
    (void) p;
}

static void dsv4_hc_unpack_f16_0(uint8_t * p) {
    *(HVX_Vector *) p = Q6_V_lo_W(hvx_vec_f16_to_f32(*(const HVX_Vector *) p));
}

static void dsv4_hc_unpack_f16_128(uint8_t * p) {
    *(HVX_Vector *) (p + 128) = Q6_V_lo_W(hvx_vec_f16_to_f32(*(const HVX_Vector *) (p + 128)));
}

static void dsv4_hc_unpack_f16_0_128(uint8_t * p) {
    *(HVX_Vector *) p         = Q6_V_lo_W(hvx_vec_f16_to_f32(*(const HVX_Vector *) p));
    *(HVX_Vector *) (p + 128) = Q6_V_lo_W(hvx_vec_f16_to_f32(*(const HVX_Vector *) (p + 128)));
}

#define DSV4_HC_LOAD2_f32(lo, hi, ptr)               \
    do {                                             \
        lo = *(const HVX_Vector *) (ptr);            \
        hi = *(const HVX_Vector *) ((ptr) + VLEN);   \
    } while (0)

#define DSV4_HC_LOAD2_f16(lo, hi, ptr)                                              \
    do {                                                                            \
        const HVX_VectorPair _p = hvx_vec_f16_to_f32(*(const HVX_Vector *) (ptr));  \
        lo = Q6_V_lo_W(_p);                                                         \
        hi = Q6_V_hi_W(_p);                                                         \
    } while (0)

#define DSV4_HC_LOAD2_f16s(lo, hi, ptr)                                                   \
    do {                                                                                  \
        const HVX_VectorPair _p = hvx_vec_f16_to_f32_shuff(*(const HVX_Vector *) (ptr));  \
        lo = Q6_V_lo_W(_p);                                                               \
        hi = Q6_V_hi_W(_p);                                                               \
    } while (0)

#define DSV4_HC_STORE2_f32(ptr, lo, hi)              \
    do {                                             \
        *(HVX_Vector *) (ptr)          = (lo);       \
        *(HVX_Vector *) ((ptr) + VLEN) = (hi);       \
    } while (0)

#define DSV4_HC_STORE2_f16(ptr, lo, hi)                          \
    do {                                                         \
        *(HVX_Vector *) (ptr) = hvx_vec_f32_to_f16((lo), (hi));  \
    } while (0)

#define DSV4_HC_STORE2_f16s(ptr, lo, hi)                               \
    do {                                                               \
        *(HVX_Vector *) (ptr) = hvx_vec_f32_to_f16_shuff((lo), (hi));  \
    } while (0)

#define DSV4_HC_STORE2_TAIL_f16s(ptr, lo, hi, has_hi)                  \
    do {                                                               \
        (void) (has_hi);                                               \
        *(HVX_Vector *) (ptr) = hvx_vec_f32_to_f16_shuff((lo), (hi));  \
    } while (0)

#define DSV4_HC_STORE2_TAIL_f32s(ptr, lo, hi, has_hi)                  \
    do {                                                               \
        const HVX_VectorPair _p = Q6_W_vshuff_VVR((hi), (lo), -4);     \
        *(HVX_Vector *) (ptr) = Q6_V_lo_W(_p);                         \
        if (has_hi) {                                                  \
            *(HVX_Vector *) ((ptr) + VLEN) = Q6_V_hi_W(_p);            \
        }                                                              \
    } while (0)

#define DSV4_HC_LOAD1_f32(v, ptr)                    \
    do {                                             \
        v = *(const HVX_Vector *) (ptr);             \
    } while (0)

#define DSV4_HC_LOAD1_f16(v, ptr)                                                \
    do {                                                                         \
        v = Q6_V_lo_W(hvx_vec_f16_to_f32(*(const HVX_Vector *) (ptr)));          \
    } while (0)

#define DSV4_HC_LOAD1_f16s(v, ptr) DSV4_HC_LOAD1_f16(v, ptr)

#define DSV4_HC_STORE1_f32(ptr, v)                   \
    do {                                             \
        *(HVX_Vector *) (ptr) = (v);                 \
    } while (0)

#define DSV4_HC_STORE1_f16(ptr, v)                               \
    do {                                                         \
        *(HVX_Vector *) (ptr) = hvx_vec_f32_to_f16((v), (v));    \
    } while (0)

#define DSV4_HC_STORE1_f16s(ptr, v) DSV4_HC_STORE1_f16(ptr, v)

struct dsv4_hc_pre_context;
struct dsv4_hc_block;

typedef void (*dsv4_hc_pre_plain_fn_t)(
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_block * block,
        const uint8_t * x_spad,
        uint8_t * d_spad,
        const float * wb);

typedef void (*dsv4_hc_pre_gated_fn_t)(
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_block * block,
        const uint8_t * x_spad,
        const uint8_t * g_spad,
        uint8_t * d_spad);

struct dsv4_hc_pre_context {
    struct htp_ops_context * octx;
    const struct htp_dsv4_hc_pre_kernel_params * kparams;
    uint32_t token_start;
    uint32_t n_chunks;
    uint32_t total_jobs;
    uint32_t jobs_per_hvx;
    float scale;
    uint8_t * vtcm;
    uint32_t dma_block_chunks;
    uint32_t dma_row_stride;
    uint32_t dma_buffer_size;
    uint32_t x_chunk_bytes;
    uint32_t w_chunk_bytes;
    uint32_t w_row_bytes;
    uint32_t dst_chunk_bytes;
    dsv4_hc_unpack_fn_t    unpack_weights;
    dsv4_hc_pre_plain_fn_t compute_plain;
    dsv4_hc_pre_gated_fn_t compute_gated;
};

struct dsv4_hc_block {
    uint32_t it;
    uint32_t chunk;
    uint32_t count;
};

static inline struct dsv4_hc_block dsv4_hc_pre_block_make(
        const struct dsv4_hc_pre_context * dctx,
        uint32_t job,
        uint32_t end) {
    const uint32_t local_token = fastdiv(job, &dctx->kparams->div_n_chunks);
    const uint32_t chunk       = job - local_token*dctx->n_chunks;
    const uint32_t count       = hex_smin(dctx->dma_block_chunks, hex_smin(dctx->n_chunks - chunk, end - job));
    return (struct dsv4_hc_block) {
        .it    = dctx->token_start + local_token,
        .chunk = chunk,
        .count = count,
    };
}

static inline void dsv4_hc_pre_push_plain(
        dma_queue * dma_q,
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_block * block,
        uint8_t * spad) {
    const struct htp_tensor * x       = dctx->octx->src[0];
    const struct htp_tensor * weights = dctx->octx->src[1];
    const uint32_t row_stride         = dctx->dma_row_stride;
    const uint32_t x_cb               = dctx->x_chunk_bytes;
    const uint32_t w_bytes            = dctx->w_row_bytes;
    const uint32_t x_size             = block->count * x_cb;

    dma_queue_push(dma_q, dma_make_data(spad, x->data + (block->it*x->nb[2] + block->chunk*x_cb)), row_stride, x->nb[1], x_size, 4);
    dma_queue_push(dma_q, dma_make_data(spad + 5*row_stride, weights->data + (block->it*weights->nb[1])), w_bytes, w_bytes, w_bytes, 1);
}

static inline void dsv4_hc_pre_push_gated(
        dma_queue * dma_q,
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_block * block,
        uint8_t * spad) {
    const struct htp_tensor * x       = dctx->octx->src[0];
    const struct htp_tensor * g       = dctx->octx->src[1];
    const uint32_t row_stride         = dctx->dma_row_stride;
    const uint32_t x_cb               = dctx->x_chunk_bytes;
    const uint32_t g_cb               = dctx->w_chunk_bytes;
    const uint32_t x_size             = block->count * x_cb;
    const uint32_t g_size             = block->count * g_cb;

    dma_queue_push(dma_q, dma_make_data(spad, x->data + (block->it*x->nb[2] + block->chunk*x_cb)), row_stride, x->nb[1], x_size, 4);
    dma_queue_push(dma_q, dma_make_data(spad + 4*row_stride, g->data + (block->it*g->nb[2] + block->chunk*g_cb)), row_stride, g->nb[1], g_size, 4);
}

#define DEFINE_DSV4_HC_PRE_PLAIN_BLOCK(SUFFIX, TX, TDST, X_CB, D_CB)                    \
static __attribute__((noinline)) void dsv4_hc_pre_plain_block_##SUFFIX(                 \
        const struct dsv4_hc_pre_context * dctx,                                        \
        const struct dsv4_hc_block * block,                                             \
        const uint8_t * x_spad,                                                         \
        uint8_t * d_spad,                                                               \
        const float * wb) {                                                             \
    const uint32_t row_stride = dctx->dma_row_stride;                                   \
    const float scale = dctx->scale;                                                    \
    const HVX_Vector w0 = hvx_vec_splat_f32(wb[0] * scale);                             \
    const HVX_Vector w1 = hvx_vec_splat_f32(wb[1] * scale);                             \
    const HVX_Vector w2 = hvx_vec_splat_f32(wb[2] * scale);                             \
    const HVX_Vector w3 = hvx_vec_splat_f32(wb[3] * scale);                             \
    const uint32_t count = block->count;                                                \
    uint32_t c = 0;                                                                     \
    for (; c + 2 <= count; c += 2) {                                                    \
        const uint8_t * xb = x_spad + c * (X_CB);                                       \
        HVX_Vector x0, x1;                                                              \
        DSV4_HC_LOAD2_##TX(x0, x1, xb);                                                 \
        HVX_Vector acc0 = hvx_vec_mul_f32_f32(x0, w0);                                  \
        HVX_Vector acc1 = hvx_vec_mul_f32_f32(x1, w0);                                  \
        DSV4_HC_LOAD2_##TX(x0, x1, xb + row_stride);                                    \
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(x0, w1));                  \
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(x1, w1));                  \
        DSV4_HC_LOAD2_##TX(x0, x1, xb + 2 * row_stride);                                \
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(x0, w2));                  \
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(x1, w2));                  \
        DSV4_HC_LOAD2_##TX(x0, x1, xb + 3 * row_stride);                                \
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(x0, w3));                  \
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(x1, w3));                  \
        DSV4_HC_STORE2_##TDST(d_spad + c * (D_CB), acc0, acc1);                         \
    }                                                                                   \
    if (c < count) {                                                                    \
        const uint8_t * xb = x_spad + c * (X_CB);                                       \
        HVX_Vector x0;                                                                  \
        DSV4_HC_LOAD1_##TX(x0, xb);                                                     \
        HVX_Vector acc0 = hvx_vec_mul_f32_f32(x0, w0);                                  \
        DSV4_HC_LOAD1_##TX(x0, xb + row_stride);                                        \
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(x0, w1));                  \
        DSV4_HC_LOAD1_##TX(x0, xb + 2 * row_stride);                                    \
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(x0, w2));                  \
        DSV4_HC_LOAD1_##TX(x0, xb + 3 * row_stride);                                    \
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(x0, w3));                  \
        DSV4_HC_STORE1_##TDST(d_spad + c * (D_CB), acc0);                               \
    }                                                                                   \
}

DEFINE_DSV4_HC_PRE_PLAIN_BLOCK(f32_f32, f32,  f32,  128, 128)
DEFINE_DSV4_HC_PRE_PLAIN_BLOCK(f32_f16, f32,  f16,  128, 64)
DEFINE_DSV4_HC_PRE_PLAIN_BLOCK(f16_f32, f16,  f32,  64,  128)
DEFINE_DSV4_HC_PRE_PLAIN_BLOCK(f16_f16, f16s, f16s, 64,  64)

#define DSV4_HC_SIGMOID2_f32(lo, hi, ptr, max_exp, min_exp)                                                  \
    do {                                                                                                     \
        lo = dsv4_hc_sigmoid_f32(*(const HVX_Vector *) (ptr), max_exp, min_exp);                             \
        hi = dsv4_hc_sigmoid_f32(*(const HVX_Vector *) ((ptr) + VLEN), max_exp, min_exp);                    \
    } while (0)

#define DSV4_HC_SIGMOID2_f16(lo, hi, ptr, max_exp, min_exp)                                                  \
    do {                                                                                                     \
        (void) (max_exp);                                                                                    \
        (void) (min_exp);                                                                                    \
        const HVX_VectorPair _p = hvx_vec_f16_to_f32(hvx_vec_fast_sigmoid_f16(*(const HVX_Vector *) (ptr))); \
        lo = Q6_V_lo_W(_p);                                                                                  \
        hi = Q6_V_hi_W(_p);                                                                                  \
    } while (0)

#define DSV4_HC_SIGMOID1_f32(v, ptr, max_exp, min_exp)                                                 \
    do {                                                                                               \
        v = dsv4_hc_sigmoid_f32(*(const HVX_Vector *) (ptr), max_exp, min_exp);                        \
    } while (0)

#define DSV4_HC_SIGMOID1_f16(v, ptr, max_exp, min_exp)                                                     \
    do {                                                                                                   \
        (void) (max_exp);                                                                                  \
        (void) (min_exp);                                                                                  \
        v = Q6_V_lo_W(hvx_vec_f16_to_f32(hvx_vec_fast_sigmoid_f16(*(const HVX_Vector *) (ptr))));          \
    } while (0)

#define DSV4_HC_PRE_GATED_STREAM2(TX, TG, xb, gb, off, acc0, acc1, max_exp, min_exp)    \
    do {                                                                                \
        const uint8_t * _xp = (xb) + (off);                                             \
        const uint8_t * _gp = (gb) + (off);                                             \
        asm volatile("" : "+v"(acc0), "+v"(acc1), "+r"(_xp), "+r"(_gp));                \
        HVX_Vector _x0, _x1, _g0, _g1;                                                  \
        DSV4_HC_LOAD2_##TX(_x0, _x1, _xp);                                              \
        DSV4_HC_SIGMOID2_##TG(_g0, _g1, _gp, max_exp, min_exp);                         \
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(_x0, _g0));                \
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(_x1, _g1));                \
    } while (0)

#define DEFINE_DSV4_HC_PRE_GATED_BLOCK(SUFFIX, TX, TG, TDST, X_CB, G_CB, D_CB)          \
static __attribute__((noinline)) void dsv4_hc_pre_gated_block_##SUFFIX(                 \
        const struct dsv4_hc_pre_context * dctx,                                        \
        const struct dsv4_hc_block * block,                                             \
        const uint8_t * x_spad,                                                         \
        const uint8_t * g_spad,                                                         \
        uint8_t * d_spad) {                                                             \
    const uint32_t row_stride = dctx->dma_row_stride;                                   \
    const HVX_Vector scale_v  = hvx_vec_splat_f32(dctx->scale);                         \
    const HVX_Vector max_exp  = hvx_vec_splat_f32(87.0f);                               \
    const HVX_Vector min_exp  = hvx_vec_splat_f32(-87.0f);                              \
    const uint32_t count      = block->count;                                           \
    uint32_t c = 0;                                                                     \
    for (; c + 2 <= count; c += 2) {                                                    \
        const uint8_t * xb = x_spad + c * (X_CB);                                       \
        const uint8_t * gb = g_spad + c * (G_CB);                                       \
        HVX_Vector x0, x1, g0, g1;                                                      \
        DSV4_HC_LOAD2_##TX(x0, x1, xb);                                                 \
        DSV4_HC_SIGMOID2_##TG(g0, g1, gb, max_exp, min_exp);                            \
        HVX_Vector acc0 = hvx_vec_mul_f32_f32(x0, g0);                                  \
        HVX_Vector acc1 = hvx_vec_mul_f32_f32(x1, g1);                                  \
        DSV4_HC_PRE_GATED_STREAM2(TX, TG, xb, gb, row_stride, acc0, acc1, max_exp,      \
                                  min_exp);                                             \
        DSV4_HC_PRE_GATED_STREAM2(TX, TG, xb, gb, 2 * row_stride, acc0, acc1, max_exp,  \
                                  min_exp);                                             \
        DSV4_HC_PRE_GATED_STREAM2(TX, TG, xb, gb, 3 * row_stride, acc0, acc1, max_exp,  \
                                  min_exp);                                             \
        acc0 = hvx_vec_mul_f32_f32(acc0, scale_v);                                      \
        acc1 = hvx_vec_mul_f32_f32(acc1, scale_v);                                      \
        DSV4_HC_STORE2_##TDST(d_spad + c * (D_CB), acc0, acc1);                         \
    }                                                                                   \
    if (c < count) {                                                                    \
        const uint8_t * xb = x_spad + c * (X_CB);                                       \
        const uint8_t * gb = g_spad + c * (G_CB);                                       \
        HVX_Vector x0, g0;                                                              \
        DSV4_HC_LOAD1_##TX(x0, xb);                                                     \
        DSV4_HC_SIGMOID1_##TG(g0, gb, max_exp, min_exp);                                \
        HVX_Vector acc0 = hvx_vec_mul_f32_f32(x0, g0);                                  \
        DSV4_HC_LOAD1_##TX(x0, xb + row_stride);                                        \
        DSV4_HC_SIGMOID1_##TG(g0, gb + row_stride, max_exp, min_exp);                   \
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(x0, g0));                  \
        DSV4_HC_LOAD1_##TX(x0, xb + 2 * row_stride);                                    \
        DSV4_HC_SIGMOID1_##TG(g0, gb + 2 * row_stride, max_exp, min_exp);               \
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(x0, g0));                  \
        DSV4_HC_LOAD1_##TX(x0, xb + 3 * row_stride);                                    \
        DSV4_HC_SIGMOID1_##TG(g0, gb + 3 * row_stride, max_exp, min_exp);               \
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(x0, g0));                  \
        acc0 = hvx_vec_mul_f32_f32(acc0, scale_v);                                      \
        DSV4_HC_STORE1_##TDST(d_spad + c * (D_CB), acc0);                               \
    }                                                                                   \
}

#define DSV4_HC_PRE_GATED_EVAL_F16(xb, gb, row_stride, scale_v, acc0, acc1)                              \
    do {                                                                                                 \
        const uint8_t * _xb = (xb);                                                                      \
        const uint8_t * _gb = (gb);                                                                      \
        const HVX_Vector _s0 = hvx_vec_fast_sigmoid_f16(*(const HVX_Vector *) _gb);                      \
        const HVX_Vector _s1 = hvx_vec_fast_sigmoid_f16(*(const HVX_Vector *) (_gb + (row_stride)));     \
        const HVX_Vector _s2 = hvx_vec_fast_sigmoid_f16(*(const HVX_Vector *) (_gb + 2 * (row_stride))); \
        const HVX_Vector _s3 = hvx_vec_fast_sigmoid_f16(*(const HVX_Vector *) (_gb + 3 * (row_stride))); \
        asm volatile("" : "+r"(_xb) : "v"(_s0), "v"(_s1), "v"(_s2), "v"(_s3));                           \
        HVX_VectorPair _acc = hvx_vec_mpy_f32_f16(*(const HVX_Vector *) _xb, _s0);                       \
        _acc = hvx_vec_mpyacc_f32_f16(_acc, *(const HVX_Vector *) (_xb + (row_stride)), _s1);            \
        _acc = hvx_vec_mpyacc_f32_f16(_acc, *(const HVX_Vector *) (_xb + 2 * (row_stride)), _s2);        \
        _acc = hvx_vec_mpyacc_f32_f16(_acc, *(const HVX_Vector *) (_xb + 3 * (row_stride)), _s3);        \
        acc0 = hvx_vec_mul_f32_f32(Q6_V_lo_W(_acc), scale_v);                                            \
        acc1 = hvx_vec_mul_f32_f32(Q6_V_hi_W(_acc), scale_v);                                            \
    } while (0)

#define DEFINE_DSV4_HC_PRE_GATED_F16_BLOCK(SUFFIX, TDST, D_CB)                          \
static __attribute__((noinline)) void dsv4_hc_pre_gated_block_##SUFFIX(                 \
        const struct dsv4_hc_pre_context * dctx,                                        \
        const struct dsv4_hc_block * block,                                             \
        const uint8_t * x_spad,                                                         \
        const uint8_t * g_spad,                                                         \
        uint8_t * d_spad) {                                                             \
    const uint32_t row_stride = dctx->dma_row_stride;                                   \
    const HVX_Vector scale_v  = hvx_vec_splat_f32(dctx->scale);                         \
    const uint32_t count      = block->count;                                           \
    for (uint32_t c = 0; c < count; c += 2) {                                           \
        HVX_Vector acc0, acc1;                                                          \
        DSV4_HC_PRE_GATED_EVAL_F16(x_spad + c * 64, g_spad + c * 64, row_stride,        \
                                   scale_v, acc0, acc1);                                \
        DSV4_HC_STORE2_TAIL_##TDST(d_spad + c * (D_CB), acc0, acc1, c + 1 < count);     \
    }                                                                                   \
}

DEFINE_DSV4_HC_PRE_GATED_BLOCK(f32_f32_f32, f32, f32, f32, 128, 128, 128)
DEFINE_DSV4_HC_PRE_GATED_BLOCK(f32_f32_f16, f32, f32, f16, 128, 128, 64)
DEFINE_DSV4_HC_PRE_GATED_BLOCK(f32_f16_f32, f32, f16, f32, 128, 64,  128)
DEFINE_DSV4_HC_PRE_GATED_BLOCK(f32_f16_f16, f32, f16, f16, 128, 64,  64)
DEFINE_DSV4_HC_PRE_GATED_BLOCK(f16_f32_f32, f16, f32, f32, 64,  128, 128)
DEFINE_DSV4_HC_PRE_GATED_BLOCK(f16_f32_f16, f16, f32, f16, 64,  128, 64)
DEFINE_DSV4_HC_PRE_GATED_F16_BLOCK(f16_f16_f32, f32s, 128)
DEFINE_DSV4_HC_PRE_GATED_F16_BLOCK(f16_f16_f16, f16s, 64)

static void dsv4_hc_pre_thread_plain(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_pre_context * dctx = (struct dsv4_hc_pre_context *) data;
    struct htp_ops_context *     octx = dctx->octx;

    const struct htp_tensor * dst    = octx->dst;
    const uint64_t            d_data = dst->data;

    const uint32_t row_stride  = dctx->dma_row_stride;
    const uint32_t buffer_size = dctx->dma_buffer_size;
    const uint32_t dst_cb      = dctx->dst_chunk_bytes;
    const uint32_t start       = dctx->jobs_per_hvx * ith;
    const uint32_t end         = hex_smin(start + dctx->jobs_per_hvx, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*buffer_size*ith,
        dctx->vtcm + 2*buffer_size*ith + buffer_size,
    };

    dma_queue * dma_q = th->dma_q;
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_block block = dsv4_hc_pre_block_make(dctx, prefetch_job, end);
        const uint32_t d_bytes = block.count * dst_cb;
        // empty dst writeback keeps pop order dst, x, weights in loop
        dma_queue_push(dma_q, dma_make_data(d_data, spad[i] + 4*row_stride), d_bytes, d_bytes, 0, 0);
        dsv4_hc_pre_push_plain(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace * tr = th->trace;
    const dsv4_hc_unpack_fn_t    unpack  = dctx->unpack_weights;
    const dsv4_hc_pre_plain_fn_t compute = dctx->compute_plain;

    uint32_t job  = start;
    uint32_t curr = 0;
    while (job < end) {
        const struct dsv4_hc_block block = dsv4_hc_pre_block_make(dctx, job, end);

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        uint8_t * w_spad = spad[curr] + 5*row_stride;
        uint8_t * d_spad = spad[curr] + 4*row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        unpack(w_spad);
        compute(dctx, &block, spad[curr], d_spad, (const float *) w_spad);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        const uint32_t d_bytes = block.count * dst_cb;
        const uint64_t d_dst   = d_data + (block.it*dst->nb[1] + block.chunk*dst_cb);
        dma_queue_push(dma_q, dma_make_data(d_dst, d_spad), d_bytes, d_bytes, d_bytes, 1);

        if (prefetch_job < end) {
            const struct dsv4_hc_block next = dsv4_hc_pre_block_make(dctx, prefetch_job, end);
            dsv4_hc_pre_push_plain(dma_q, dctx, &next, spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
    }

    dma_queue_flush(dma_q);

    (void) nth;
}

static void dsv4_hc_pre_thread_gated(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_pre_context * dctx = (struct dsv4_hc_pre_context *) data;
    struct htp_ops_context *     octx = dctx->octx;

    const struct htp_tensor * dst    = octx->dst;
    const uint64_t            d_data = dst->data;

    const uint32_t row_stride  = dctx->dma_row_stride;
    const uint32_t buffer_size = dctx->dma_buffer_size;
    const uint32_t dst_cb      = dctx->dst_chunk_bytes;
    const uint32_t start       = dctx->jobs_per_hvx * ith;
    const uint32_t end         = hex_smin(start + dctx->jobs_per_hvx, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*buffer_size*ith,
        dctx->vtcm + 2*buffer_size*ith + buffer_size,
    };

    dma_queue * dma_q = th->dma_q;
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_block block = dsv4_hc_pre_block_make(dctx, prefetch_job, end);
        const uint32_t d_bytes = block.count * dst_cb;
        // empty dst writeback keeps pop order dst, x, g in loop
        dma_queue_push(dma_q, dma_make_data(d_data, spad[i] + 8*row_stride), d_bytes, d_bytes, 0, 0);
        dsv4_hc_pre_push_gated(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace * tr = th->trace;
    const dsv4_hc_pre_gated_fn_t compute = dctx->compute_gated;

    uint32_t job  = start;
    uint32_t curr = 0;
    while (job < end) {
        const struct dsv4_hc_block block = dsv4_hc_pre_block_make(dctx, job, end);

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        uint8_t * d_spad = spad[curr] + 8*row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        compute(dctx, &block, spad[curr], spad[curr] + 4*row_stride, d_spad);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        const uint32_t d_bytes = block.count * dst_cb;
        const uint64_t d_dst   = d_data + (block.it*dst->nb[1] + block.chunk*dst_cb);
        dma_queue_push(dma_q, dma_make_data(d_dst, d_spad), d_bytes, d_bytes, d_bytes, 1);

        if (prefetch_job < end) {
            const struct dsv4_hc_block next = dsv4_hc_pre_block_make(dctx, prefetch_job, end);
            dsv4_hc_pre_push_gated(dma_q, dctx, &next, spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
    }

    dma_queue_flush(dma_q);

    (void) nth;
}

int op_dsv4_hc_pre(struct htp_ops_context * octx) {
    const struct htp_tensor * x       = octx->src[0];
    const struct htp_tensor * weights = octx->src[1];
    const struct htp_tensor * dst     = octx->dst;

    const struct htp_dsv4_hc_pre_kernel_params * kparams =
        (const struct htp_dsv4_hc_pre_kernel_params *) octx->kernel_params;

    const uint32_t n_hvx = kparams->n_hvx;
    if (!htp_ops_context_set_n_hvx_div(octx, n_hvx, kparams->div_n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t token_start     = 0;
    uint32_t n_tokens        = x->ne[2];
    uint32_t total_jobs      = kparams->total_jobs;
    uint32_t jobs_per_hvx    = kparams->jobs_per_hvx;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_tokens, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        token_start     = range.start;
        n_tokens        = range.count;
        total_jobs      = n_tokens * kparams->n_chunks;
        jobs_per_hvx    = fastdiv(total_jobs + n_hvx - 1, &kparams->div_n_hvx);
    }

    if (n_tokens == 0) {
        return HTP_STATUS_OK;
    }

    float scale;
    memcpy(&scale, octx->op_params, sizeof(scale));

    const uint32_t x_f16 = (x->type == HTP_TYPE_F16) ? 1u : 0u;
    const uint32_t w_f16 = (weights->type == HTP_TYPE_F16) ? 1u : 0u;
    const uint32_t d_f16 = (dst->type == HTP_TYPE_F16) ? 1u : 0u;

    static const dsv4_hc_pre_plain_fn_t plain_fns[4] = {
        dsv4_hc_pre_plain_block_f32_f32,
        dsv4_hc_pre_plain_block_f32_f16,
        dsv4_hc_pre_plain_block_f16_f32,
        dsv4_hc_pre_plain_block_f16_f16,
    };

    static const dsv4_hc_pre_gated_fn_t gated_fns[8] = {
        dsv4_hc_pre_gated_block_f32_f32_f32,
        dsv4_hc_pre_gated_block_f32_f32_f16,
        dsv4_hc_pre_gated_block_f32_f16_f32,
        dsv4_hc_pre_gated_block_f32_f16_f16,
        dsv4_hc_pre_gated_block_f16_f32_f32,
        dsv4_hc_pre_gated_block_f16_f32_f16,
        dsv4_hc_pre_gated_block_f16_f16_f32,
        dsv4_hc_pre_gated_block_f16_f16_f16,
    };

    struct dsv4_hc_pre_context dctx = {
        .octx             = octx,
        .kparams          = kparams,
        .token_start      = token_start,
        .n_chunks         = kparams->n_chunks,
        .total_jobs       = total_jobs,
        .jobs_per_hvx     = jobs_per_hvx,
        .scale            = scale,
        .vtcm             = (uint8_t *) octx->ctx->vtcm_base,
        .dma_block_chunks = kparams->dma_block_chunks,
        .dma_row_stride   = kparams->dma_row_stride,
        .dma_buffer_size  = kparams->dma_buffer_size,
        .x_chunk_bytes    = x_f16 ? 64u : 128u,
        .w_chunk_bytes    = w_f16 ? 64u : 128u,
        .w_row_bytes      = w_f16 ? 8u  : 16u,
        .dst_chunk_bytes  = d_f16 ? 64u : 128u,
        .unpack_weights   = w_f16 ? dsv4_hc_unpack_f16_0 : dsv4_hc_unpack_noop,
        .compute_plain    = plain_fns[(x_f16 << 1) | d_f16],
        .compute_gated    = gated_fns[(x_f16 << 2) | (w_f16 << 1) | d_f16],
    };

    work_queue_func_t worker = NULL;
    switch (kparams->kernel_type) {
        case HTP_DSV4_HC_PRE_KERNEL_PLAIN: worker = dsv4_hc_pre_thread_plain; break;
        case HTP_DSV4_HC_PRE_KERNEL_GATED: worker = dsv4_hc_pre_thread_gated; break;
        default: return HTP_STATUS_INVAL_PARAMS;
    }

    FARF(HIGH, "dsv4-hc-pre: (%ux%ux%u) -> (%ux%u) : ktype %u tokens %u:%u nhvx %u\n",
         x->ne[0], x->ne[1], x->ne[2], dst->ne[0], dst->ne[1], kparams->kernel_type,
         token_start, token_start + n_tokens, n_hvx);

    work_queue_run(octx->ctx->hvx_queue, worker, &dctx, n_hvx);
    return HTP_STATUS_OK;
}

// ---------------------------------------------------------------------------
// DSV4_HC_POST
// ---------------------------------------------------------------------------

typedef void (*dsv4_hc_post_compute_fn_t)(
        const uint8_t * xb,
        const uint8_t * rb,
        size_t r_stride,
        uint8_t * db,
        size_t d_stride,
        uint32_t count,
        const float * p,
        const float * c);

struct dsv4_hc_post_context {
    struct htp_ops_context * octx;
    const struct htp_dsv4_hc_post_kernel_params * kparams;
    uint32_t token_start;
    uint32_t total_jobs;
    uint32_t jobs_per_hvx;
    uint8_t * vtcm;
    uint32_t x_chunk_bytes;
    uint32_t r_chunk_bytes;
    uint32_t dst_chunk_bytes;
    uint32_t post_bytes;
    uint32_t comb_row_bytes;
    dsv4_hc_unpack_fn_t       unpack_params;
    dsv4_hc_post_compute_fn_t compute;
};

static inline struct dsv4_hc_block dsv4_hc_post_block_make(
        const struct dsv4_hc_post_context * dctx,
        uint32_t job,
        uint32_t end,
        uint32_t block_chunks) {
    const uint32_t local_token = fastdiv(job, &dctx->kparams->div_n_chunks);
    const uint32_t chunk       = job - local_token*dctx->kparams->n_chunks;
    const uint32_t count       = hex_smin(block_chunks, hex_smin(dctx->kparams->n_chunks - chunk, end - job));
    return (struct dsv4_hc_block) {
        .it    = dctx->token_start + local_token,
        .chunk = chunk,
        .count = count,
    };
}

#define DEFINE_DSV4_HC_POST_IDENTITY_BLOCK(SUFFIX, TX, TR, TDST, X_CB, R_CB, D_CB)      \
static __attribute__((noinline)) void dsv4_hc_post_identity_block_##SUFFIX(             \
        const uint8_t * xb,                                                             \
        const uint8_t * rb,                                                             \
        size_t r_stride,                                                                \
        uint8_t * db,                                                                   \
        size_t d_stride,                                                                \
        uint32_t count,                                                                 \
        const float * p,                                                                \
        const float * c_unused) {                                                       \
    (void) c_unused;                                                                    \
    const HVX_Vector pv0 = hvx_vec_splat_f32(p[0]);                                     \
    const HVX_Vector pv1 = hvx_vec_splat_f32(p[1]);                                     \
    const HVX_Vector pv2 = hvx_vec_splat_f32(p[2]);                                     \
    const HVX_Vector pv3 = hvx_vec_splat_f32(p[3]);                                     \
    uint32_t i = 0;                                                                     \
    for (; i + 2 <= count; i += 2) {                                                    \
        HVX_Vector x0, x1, r0, r1;                                                      \
        DSV4_HC_LOAD2_##TX(x0, x1, xb + i * (X_CB));                                    \
        const uint8_t * rr = rb + i * (R_CB);                                           \
        uint8_t *       dd = db + i * (D_CB);                                           \
        DSV4_HC_LOAD2_##TR(r0, r1, rr);                                                 \
        DSV4_HC_STORE2_##TDST(dd,                                                       \
            hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x0, pv0), r0),                      \
            hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x1, pv0), r1));                     \
        DSV4_HC_LOAD2_##TR(r0, r1, rr + r_stride);                                      \
        DSV4_HC_STORE2_##TDST(dd + d_stride,                                            \
            hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x0, pv1), r0),                      \
            hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x1, pv1), r1));                     \
        DSV4_HC_LOAD2_##TR(r0, r1, rr + 2 * r_stride);                                  \
        DSV4_HC_STORE2_##TDST(dd + 2 * d_stride,                                        \
            hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x0, pv2), r0),                      \
            hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x1, pv2), r1));                     \
        DSV4_HC_LOAD2_##TR(r0, r1, rr + 3 * r_stride);                                  \
        DSV4_HC_STORE2_##TDST(dd + 3 * d_stride,                                        \
            hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x0, pv3), r0),                      \
            hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x1, pv3), r1));                     \
    }                                                                                   \
    if (i < count) {                                                                    \
        HVX_Vector x0, r0;                                                              \
        DSV4_HC_LOAD1_##TX(x0, xb + i * (X_CB));                                        \
        const uint8_t * rr = rb + i * (R_CB);                                           \
        uint8_t *       dd = db + i * (D_CB);                                           \
        DSV4_HC_LOAD1_##TR(r0, rr);                                                     \
        DSV4_HC_STORE1_##TDST(dd,                                                       \
                              hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x0, pv0), r0));   \
        DSV4_HC_LOAD1_##TR(r0, rr + r_stride);                                          \
        DSV4_HC_STORE1_##TDST(dd + d_stride,                                            \
                              hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x0, pv1), r0));   \
        DSV4_HC_LOAD1_##TR(r0, rr + 2 * r_stride);                                      \
        DSV4_HC_STORE1_##TDST(dd + 2 * d_stride,                                        \
                              hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x0, pv2), r0));   \
        DSV4_HC_LOAD1_##TR(r0, rr + 3 * r_stride);                                      \
        DSV4_HC_STORE1_##TDST(dd + 3 * d_stride,                                        \
                              hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x0, pv3), r0));   \
    }                                                                                   \
}

DEFINE_DSV4_HC_POST_IDENTITY_BLOCK(f32_f32_f32, f32,  f32,  f32,  128, 128, 128)
DEFINE_DSV4_HC_POST_IDENTITY_BLOCK(f32_f32_f16, f32,  f32,  f16,  128, 128, 64)
DEFINE_DSV4_HC_POST_IDENTITY_BLOCK(f32_f16_f32, f32,  f16,  f32,  128, 64,  128)
DEFINE_DSV4_HC_POST_IDENTITY_BLOCK(f32_f16_f16, f32,  f16,  f16,  128, 64,  64)
DEFINE_DSV4_HC_POST_IDENTITY_BLOCK(f16_f32_f32, f16,  f32,  f32,  64,  128, 128)
DEFINE_DSV4_HC_POST_IDENTITY_BLOCK(f16_f32_f16, f16,  f32,  f16,  64,  128, 64)
DEFINE_DSV4_HC_POST_IDENTITY_BLOCK(f16_f16_f32, f16,  f16,  f32,  64,  64,  128)
DEFINE_DSV4_HC_POST_IDENTITY_BLOCK(f16_f16_f16, f16s, f16s, f16s, 64,  64,  64)

#define DEFINE_DSV4_HC_POST_COMB_BLOCK(SUFFIX, TX, TR, TDST, X_CB, R_CB, D_CB)          \
static __attribute__((noinline)) void dsv4_hc_post_comb_block_##SUFFIX(                 \
        const uint8_t * xb,                                                             \
        const uint8_t * rb,                                                             \
        size_t r_stride,                                                                \
        uint8_t * db,                                                                   \
        size_t d_stride,                                                                \
        uint32_t count,                                                                 \
        const float * p,                                                                \
        const float * c) {                                                              \
    for (uint32_t id = 0; id < 4; ++id) {                                               \
        const HVX_Vector pv = hvx_vec_splat_f32(p[id]);                                 \
        const HVX_Vector c0 = hvx_vec_splat_f32(c[id]);                                 \
        const HVX_Vector c1 = hvx_vec_splat_f32(c[4 + id]);                             \
        const HVX_Vector c2 = hvx_vec_splat_f32(c[8 + id]);                             \
        const HVX_Vector c3 = hvx_vec_splat_f32(c[12 + id]);                            \
        uint8_t * dd = db + id * d_stride;                                              \
        uint32_t i = 0;                                                                 \
        for (; i + 2 <= count; i += 2) {                                                \
            HVX_Vector v0, v1;                                                          \
            DSV4_HC_LOAD2_##TX(v0, v1, xb + i * (X_CB));                                \
            HVX_Vector acc0 = hvx_vec_mul_f32_f32(v0, pv);                              \
            HVX_Vector acc1 = hvx_vec_mul_f32_f32(v1, pv);                              \
            const uint8_t * r = rb + i * (R_CB);                                        \
            DSV4_HC_LOAD2_##TR(v0, v1, r);                                              \
            acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(v0, c0));              \
            acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(v1, c0));              \
            DSV4_HC_LOAD2_##TR(v0, v1, r + r_stride);                                   \
            acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(v0, c1));              \
            acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(v1, c1));              \
            DSV4_HC_LOAD2_##TR(v0, v1, r + 2 * r_stride);                               \
            acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(v0, c2));              \
            acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(v1, c2));              \
            DSV4_HC_LOAD2_##TR(v0, v1, r + 3 * r_stride);                               \
            acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(v0, c3));              \
            acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(v1, c3));              \
            DSV4_HC_STORE2_##TDST(dd + i * (D_CB), acc0, acc1);                         \
        }                                                                               \
        if (i < count) {                                                                \
            HVX_Vector v0;                                                              \
            DSV4_HC_LOAD1_##TX(v0, xb + i * (X_CB));                                    \
            HVX_Vector acc0 = hvx_vec_mul_f32_f32(v0, pv);                              \
            const uint8_t * r = rb + i * (R_CB);                                        \
            DSV4_HC_LOAD1_##TR(v0, r);                                                  \
            acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(v0, c0));              \
            DSV4_HC_LOAD1_##TR(v0, r + r_stride);                                       \
            acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(v0, c1));              \
            DSV4_HC_LOAD1_##TR(v0, r + 2 * r_stride);                                   \
            acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(v0, c2));              \
            DSV4_HC_LOAD1_##TR(v0, r + 3 * r_stride);                                   \
            acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(v0, c3));              \
            DSV4_HC_STORE1_##TDST(dd + i * (D_CB), acc0);                               \
        }                                                                               \
    }                                                                                   \
}

DEFINE_DSV4_HC_POST_COMB_BLOCK(f32_f32_f32, f32,  f32,  f32,  128, 128, 128)
DEFINE_DSV4_HC_POST_COMB_BLOCK(f32_f32_f16, f32,  f32,  f16,  128, 128, 64)
DEFINE_DSV4_HC_POST_COMB_BLOCK(f32_f16_f32, f32,  f16,  f32,  128, 64,  128)
DEFINE_DSV4_HC_POST_COMB_BLOCK(f32_f16_f16, f32,  f16,  f16,  128, 64,  64)
DEFINE_DSV4_HC_POST_COMB_BLOCK(f16_f32_f32, f16,  f32,  f32,  64,  128, 128)
DEFINE_DSV4_HC_POST_COMB_BLOCK(f16_f32_f16, f16,  f32,  f16,  64,  128, 64)
DEFINE_DSV4_HC_POST_COMB_BLOCK(f16_f16_f32, f16,  f16,  f32,  64,  64,  128)
DEFINE_DSV4_HC_POST_COMB_BLOCK(f16_f16_f16, f16s, f16s, f16s, 64,  64,  64)

static inline void dsv4_hc_post_push_identity(
        dma_queue * dma_q,
        const struct dsv4_hc_post_context * dctx,
        const struct dsv4_hc_block * block,
        uint8_t * spad) {
    const struct htp_tensor * x        = dctx->octx->src[0];
    const struct htp_tensor * residual = dctx->octx->src[1];
    const struct htp_tensor * post     = dctx->octx->src[2];
    const uint32_t row_stride          = dctx->kparams->dma_row_stride;
    const uint32_t x_cb                = dctx->x_chunk_bytes;
    const uint32_t r_cb                = dctx->r_chunk_bytes;
    const uint32_t p_bytes             = dctx->post_bytes;
    const uint32_t x_size              = block->count * x_cb;
    const uint32_t r_size              = block->count * r_cb;

    dma_queue_push(dma_q, dma_make_data(spad, residual->data + (block->it*residual->nb[2] + block->chunk*r_cb)),
                   row_stride, residual->nb[1], r_size, 4);
    dma_queue_push(dma_q, dma_make_data(spad + 4*row_stride, x->data + (block->it*x->nb[1] + block->chunk*x_cb)),
                   row_stride, x_size, x_size, 1);
    dma_queue_push(dma_q, dma_make_data(spad + 9*row_stride, post->data + (block->it*post->nb[1])),
                   p_bytes, p_bytes, p_bytes, 1);
}

static inline void dsv4_hc_post_push_comb(
        dma_queue * dma_q,
        const struct dsv4_hc_post_context * dctx,
        const struct dsv4_hc_block * block,
        uint8_t * spad) {
    const struct htp_tensor * comb = dctx->octx->src[3];
    const uint32_t row_stride      = dctx->kparams->dma_row_stride;
    const uint32_t c_bytes         = dctx->comb_row_bytes;

    dsv4_hc_post_push_identity(dma_q, dctx, block, spad);
    dma_queue_push(dma_q, dma_make_data(spad + 9*row_stride + 128, comb->data + (block->it*comb->nb[2])),
                   c_bytes, comb->nb[1], c_bytes, 4);
}

static void dsv4_hc_post_thread_identity(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_post_context * dctx = (struct dsv4_hc_post_context *) data;
    struct htp_ops_context *      octx = dctx->octx;

    const struct htp_tensor * dst    = octx->dst;
    const uint64_t            d_data = dst->data;

    const uint32_t block_chunks = dctx->kparams->dma_block_chunks;
    const uint32_t row_stride   = dctx->kparams->dma_row_stride;
    const uint32_t buffer_size  = dctx->kparams->dma_buffer_size;
    const uint32_t dst_cb       = dctx->dst_chunk_bytes;
    const uint32_t start        = dctx->jobs_per_hvx * ith;
    const uint32_t end          = hex_smin(start + dctx->jobs_per_hvx, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*buffer_size*ith,
        dctx->vtcm + 2*buffer_size*ith + buffer_size,
    };

    dma_queue * dma_q = th->dma_q;
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_block block = dsv4_hc_post_block_make(dctx, prefetch_job, end, block_chunks);
        // empty dst writeback keeps pop order dst, residual, x, post in loop
        dma_queue_push(dma_q, dma_make_data(d_data, spad[i] + 5*row_stride), dst->nb[1], row_stride, 0, 0);
        dsv4_hc_post_push_identity(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace *       tr      = th->trace;
    const dsv4_hc_unpack_fn_t       unpack  = dctx->unpack_params;
    const dsv4_hc_post_compute_fn_t compute = dctx->compute;

    uint32_t job  = start;
    uint32_t curr = 0;
    while (job < end) {
        const struct dsv4_hc_block block = dsv4_hc_post_block_make(dctx, job, end, block_chunks);

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        uint8_t * row9   = spad[curr] + 9*row_stride;
        uint8_t * d_spad = spad[curr] + 5*row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        unpack(row9);
        compute(spad[curr] + 4*row_stride, spad[curr], row_stride, d_spad, row_stride, block.count, (const float *) row9, NULL);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        const uint64_t d_dst = d_data + (block.it*dst->nb[2] + block.chunk*dst_cb);
        dma_queue_push(dma_q, dma_make_data(d_dst, d_spad), dst->nb[1], row_stride, block.count*dst_cb, 4);

        if (prefetch_job < end) {
            const struct dsv4_hc_block next = dsv4_hc_post_block_make(dctx, prefetch_job, end, block_chunks);
            dsv4_hc_post_push_identity(dma_q, dctx, &next, spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
    }

    dma_queue_flush(dma_q);

    (void) nth;
}

static void dsv4_hc_post_thread_comb(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_post_context * dctx = (struct dsv4_hc_post_context *) data;
    struct htp_ops_context *      octx = dctx->octx;

    const struct htp_tensor * dst    = octx->dst;
    const uint64_t            d_data = dst->data;

    const uint32_t block_chunks = dctx->kparams->dma_block_chunks;
    const uint32_t row_stride   = dctx->kparams->dma_row_stride;
    const uint32_t buffer_size  = dctx->kparams->dma_buffer_size;
    const uint32_t dst_cb       = dctx->dst_chunk_bytes;
    const uint32_t start        = dctx->jobs_per_hvx * ith;
    const uint32_t end          = hex_smin(start + dctx->jobs_per_hvx, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*buffer_size*ith,
        dctx->vtcm + 2*buffer_size*ith + buffer_size,
    };

    dma_queue * dma_q = th->dma_q;
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_block block = dsv4_hc_post_block_make(dctx, prefetch_job, end, block_chunks);
        // empty dst writeback keeps pop order dst, residual, x, post, comb in loop
        dma_queue_push(dma_q, dma_make_data(d_data, spad[i] + 5*row_stride), dst->nb[1], row_stride, 0, 0);
        dsv4_hc_post_push_comb(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace *       tr      = th->trace;
    const dsv4_hc_unpack_fn_t       unpack  = dctx->unpack_params;
    const dsv4_hc_post_compute_fn_t compute = dctx->compute;

    uint32_t job  = start;
    uint32_t curr = 0;
    while (job < end) {
        const struct dsv4_hc_block block = dsv4_hc_post_block_make(dctx, job, end, block_chunks);

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        uint8_t * row9   = spad[curr] + 9*row_stride;
        uint8_t * d_spad = spad[curr] + 5*row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        unpack(row9);
        compute(spad[curr] + 4*row_stride, spad[curr], row_stride, d_spad, row_stride, block.count,
                (const float *) row9, (const float *) (row9 + 128));
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        const uint64_t d_dst = d_data + (block.it*dst->nb[2] + block.chunk*dst_cb);
        dma_queue_push(dma_q, dma_make_data(d_dst, d_spad), dst->nb[1], row_stride, block.count*dst_cb, 4);

        if (prefetch_job < end) {
            const struct dsv4_hc_block next = dsv4_hc_post_block_make(dctx, prefetch_job, end, block_chunks);
            dsv4_hc_post_push_comb(dma_q, dctx, &next, spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
    }

    dma_queue_flush(dma_q);

    (void) nth;
}

int op_dsv4_hc_post(struct htp_ops_context * octx) {
    const struct htp_tensor * x        = octx->src[0];
    const struct htp_tensor * residual = octx->src[1];
    const struct htp_tensor * post     = octx->src[2];
    const struct htp_tensor * dst      = octx->dst;

    const struct htp_dsv4_hc_post_kernel_params * kparams =
        (const struct htp_dsv4_hc_post_kernel_params *) octx->kernel_params;
    const struct htp_tensor * comb = kparams->has_comb ? octx->src[3] : NULL;

    const uint32_t n_hvx = kparams->n_hvx;
    if (!htp_ops_context_set_n_hvx_div(octx, n_hvx, kparams->div_n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t token_start     = 0;
    uint32_t n_tokens        = x->ne[1];
    uint32_t total_jobs      = kparams->total_jobs;
    uint32_t jobs_per_hvx    = kparams->jobs_per_hvx;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_tokens, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        token_start     = range.start;
        n_tokens        = range.count;
        total_jobs      = n_tokens * kparams->n_chunks;
        jobs_per_hvx    = fastdiv(total_jobs + n_hvx - 1, &kparams->div_n_hvx);
    }

    if (n_tokens == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t x_f16 = (x->type == HTP_TYPE_F16) ? 1u : 0u;
    const uint32_t r_f16 = (residual->type == HTP_TYPE_F16) ? 1u : 0u;
    const uint32_t d_f16 = (dst->type == HTP_TYPE_F16) ? 1u : 0u;
    const uint32_t p_f16 = (post->type == HTP_TYPE_F16) ? 1u : 0u;
    const uint32_t c_f16 = (comb && comb->type == HTP_TYPE_F16) ? 1u : 0u;

    static const dsv4_hc_unpack_fn_t unpack_fns[4] = {
        dsv4_hc_unpack_noop,
        dsv4_hc_unpack_f16_0,
        dsv4_hc_unpack_f16_128,
        dsv4_hc_unpack_f16_0_128,
    };

    static const dsv4_hc_post_compute_fn_t identity_fns[8] = {
        dsv4_hc_post_identity_block_f32_f32_f32,
        dsv4_hc_post_identity_block_f32_f32_f16,
        dsv4_hc_post_identity_block_f32_f16_f32,
        dsv4_hc_post_identity_block_f32_f16_f16,
        dsv4_hc_post_identity_block_f16_f32_f32,
        dsv4_hc_post_identity_block_f16_f32_f16,
        dsv4_hc_post_identity_block_f16_f16_f32,
        dsv4_hc_post_identity_block_f16_f16_f16,
    };

    static const dsv4_hc_post_compute_fn_t comb_fns[8] = {
        dsv4_hc_post_comb_block_f32_f32_f32,
        dsv4_hc_post_comb_block_f32_f32_f16,
        dsv4_hc_post_comb_block_f32_f16_f32,
        dsv4_hc_post_comb_block_f32_f16_f16,
        dsv4_hc_post_comb_block_f16_f32_f32,
        dsv4_hc_post_comb_block_f16_f32_f16,
        dsv4_hc_post_comb_block_f16_f16_f32,
        dsv4_hc_post_comb_block_f16_f16_f16,
    };

    const uint32_t fn_idx = (x_f16 << 2) | (r_f16 << 1) | d_f16;

    struct dsv4_hc_post_context dctx = {
        .octx            = octx,
        .kparams         = kparams,
        .token_start     = token_start,
        .total_jobs      = total_jobs,
        .jobs_per_hvx    = jobs_per_hvx,
        .vtcm            = (uint8_t *) octx->ctx->vtcm_base,
        .x_chunk_bytes   = x_f16 ? 64u : 128u,
        .r_chunk_bytes   = r_f16 ? 64u : 128u,
        .dst_chunk_bytes = d_f16 ? 64u : 128u,
        .post_bytes      = p_f16 ? 8u  : 16u,
        .comb_row_bytes  = c_f16 ? 8u  : 16u,
        .unpack_params   = unpack_fns[p_f16 | (c_f16 << 1)],
        .compute         = (kparams->kernel_type == HTP_DSV4_HC_POST_KERNEL_COMB) ? comb_fns[fn_idx] : identity_fns[fn_idx],
    };

    work_queue_func_t worker = NULL;
    switch (kparams->kernel_type) {
        case HTP_DSV4_HC_POST_KERNEL_IDENTITY: worker = dsv4_hc_post_thread_identity; break;
        case HTP_DSV4_HC_POST_KERNEL_COMB:     worker = dsv4_hc_post_thread_comb;     break;
        default: return HTP_STATUS_INVAL_PARAMS;
    }

    FARF(HIGH, "dsv4-hc-post: (%ux%u) (%ux%ux%u) -> (%ux%ux%u) : ktype %u tokens %u:%u nhvx %u\n",
         x->ne[0], x->ne[1], residual->ne[0], residual->ne[1], residual->ne[2],
         dst->ne[0], dst->ne[1], dst->ne[2], kparams->kernel_type,
         token_start, token_start + n_tokens, n_hvx);

    work_queue_run(octx->ctx->hvx_queue, worker, &dctx, n_hvx);
    return HTP_STATUS_OK;
}
