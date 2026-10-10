#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt_memory.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "cpy-ops.h"
#include "dma-copy.h"
#include "htp-ctx.h"
#include "htp-fence.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-utils.h"

struct htp_copy_context {
    struct htp_ops_context *              octx;
    const struct htp_copy_kernel_params * kparams;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t src0_nrows_per_thread;

    uint32_t elem_start;
    uint32_t nelem;
    uint32_t elem_per_thread;

    uint32_t tile_start;
    uint32_t ntiles;
    uint32_t tiles_per_thread;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#define cpy_preamble                              \
    const struct htp_tensor *src0 = octx->src[0]; \
    const struct htp_tensor *dst  = octx->dst;    \
                                                  \
    const uint32_t ne00 = src0->ne[0];            \
    const uint32_t ne01 = src0->ne[1];            \
    const uint32_t ne02 = src0->ne[2];            \
    const uint32_t ne03 = src0->ne[3];            \
                                                  \
    const uint32_t nb00 = src0->nb[0];            \
    const uint32_t nb01 = src0->nb[1];            \
    const uint32_t nb02 = src0->nb[2];            \
    const uint32_t nb03 = src0->nb[3];            \
                                                  \
    const uint32_t  ne0 = dst->ne[0];             \
    const uint32_t  ne1 = dst->ne[1];             \
    const uint32_t  ne2 = dst->ne[2];             \
    const uint32_t  ne3 = dst->ne[3];             \
                                                  \
    const uint32_t  nb0 = dst->nb[0];             \
    const uint32_t  nb1 = dst->nb[1];             \
    const uint32_t  nb2 = dst->nb[2];             \
    const uint32_t  nb3 = dst->nb[3];

#define DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_SIZE)                                                \
static void cpy_thread_##NAME##_reshape(unsigned int nth, unsigned int ith, void * data) {            \
    struct htp_copy_context * ct = (struct htp_copy_context *) data;                                  \
    struct htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    const uint32_t th_nelem = ct->elem_per_thread;                                                    \
    const uint32_t th_start = ct->elem_start + ith * th_nelem;                                        \
    const uint32_t th_end   = MIN(th_start + th_nelem, ct->elem_start + ct->nelem);                   \
    if (th_start >= th_end) return;                                                                   \
                                                                                                      \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    const uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    const uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    const uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    const uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    const struct htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
    uint32_t e = th_start;                                                                            \
    uint32_t i13 = fastdiv(e, &rsh->div_ne2_ne1_ne0);                                                 \
    uint32_t rem = e - i13 * ne2_ne1_ne0;                                                             \
    uint32_t i12 = fastdiv(rem, &rsh->div_ne1_ne0);                                                   \
    uint32_t rem2 = rem - i12 * ne1_ne0;                                                              \
    uint32_t i11 = fastdiv(rem2, &rsh->div_ne0);                                                      \
    uint32_t i10 = rem2 - i11 * ne0;                                                                  \
                                                                                                      \
    uint32_t i03 = fastdiv(e, &rsh->div_ne02_ne01_ne00);                                              \
    uint32_t rem_s = e - i03 * ne02_ne01_ne00;                                                        \
    uint32_t i02 = fastdiv(rem_s, &rsh->div_ne01_ne00);                                               \
    uint32_t rem2_s = rem_s - i02 * ne01_ne00;                                                        \
    uint32_t i01 = fastdiv(rem2_s, &rsh->div_ne00);                                                   \
    uint32_t i00 = rem2_s - i01 * ne00;                                                               \
                                                                                                      \
    dma_addr_t dst_addr  = dst->data  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    const bool rows_contig = (nb00 == ELEM_SIZE) && (nb0 == ELEM_SIZE);                               \
                                                                                                      \
    while (e < th_end) {                                                                              \
        const uint32_t run = MIN(MIN(ne00 - i00, ne0 - i10), th_end - e);                             \
        if (rows_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, run * ELEM_SIZE);             \
        } else {                                                                                      \
            dma_cpy_push_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_SIZE, run);           \
        }                                                                                             \
        e += run;                                                                                     \
                                                                                                      \
        dst_addr += run * nb0;                                                                        \
        i10      += run;                                                                              \
        if (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            if (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                if (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->data + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += run * nb00;                                                                      \
        i00       += run;                                                                             \
        if (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            if (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                if (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->data + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_queue_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  float, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

// Advance a (r, i01, i02, i03) row cursor by n rows that do not cross a dim-1 run.
#define cpy_rows_advance(r, i01, i02, i03, n) \
    do {                                      \
        (r) += (n);                           \
        (i01) += (n);                         \
        if ((i01) == ne01) {                  \
            (i01) = 0;                        \
            if (++(i02) == ne02) {            \
                (i02) = 0;                    \
                (i03)++;                      \
            }                                 \
        }                                     \
    } while (0)

// Move several rows instead of single row
#define DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
static void cpy_thread_##NAME##_sameshape(unsigned int nth, unsigned int ith, void * data) { \
    struct htp_copy_context * ct = (struct htp_copy_context *) data;                         \
    struct htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    const uint32_t dr  = ct->src0_nrows_per_thread;                                          \
    const uint32_t ir0 = ct->row_start + dr * ith;                                           \
    const uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);                           \
    if (ir0 >= ir1) return;                                                                  \
                                                                                             \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                 \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    const struct htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    const uint32_t src0_buf_size = cvt->src0_buf_size;                                       \
    const uint32_t dst_buf_size  = cvt->dst_buf_size;                                        \
    const uint32_t src0_row_stride = cvt->src0_row_stride;                                   \
    const uint32_t dst_row_stride  = cvt->dst_row_stride;                                    \
    const uint32_t blk_rows        = cvt->blk_rows;                                          \
    uint8_t * vtcm_src0_base = ct->vtcm_src0 + ith * cvt->spad0_size_per_thread;             \
    uint8_t * vtcm_dst_base  = ct->vtcm_dst  + ith * cvt->spad1_size_per_thread;             \
    const uint32_t src0_row_size = ne00 * ct->kparams->src0_type_size;                       \
    const uint32_t dst_row_size  = ne00 * ct->kparams->dst_type_size;                        \
                                                                                             \
    const uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    uint32_t f_r = ir0, f_i01 = i01, f_i02 = i02, f_i03 = i03;                               \
    uint32_t c_r = ir0, c_i01 = i01, c_i02 = i02, c_i03 = i03;                               \
                                                                                             \
    for (uint32_t b = 0; b < 2 && f_r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_base + b * src0_buf_size;                             \
        uint8_t * dst_spad = vtcm_dst_base  + b * dst_buf_size;                              \
        const uint32_t nr = MIN(blk_rows, MIN(ne01 - f_i01, ir1 - f_r));                     \
        dma_addr_t f_src0_addr = src0->data + f_i01*nb01 + f_i02*nb02 + f_i03*nb03;          \
        dma_queue_push(dma_q, dma_make_data(dst->data, dst_spad),                            \
                       dst_row_size, dst_buf_size, dst_row_size, 0);                         \
        dma_queue_push(dma_q, dma_make_data(src_spad, f_src0_addr),                          \
                       src0_row_stride, nb01, src0_row_size, nr);                            \
        cpy_rows_advance(f_r, f_i01, f_i02, f_i03, nr);                                      \
    }                                                                                        \
                                                                                             \
    for (uint32_t blk = 0; c_r < ir1; ++blk) {                                               \
        const uint32_t nr = MIN(blk_rows, MIN(ne01 - c_i01, ir1 - c_r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;               \
                                                                                             \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        for (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_row_stride, src_spad + r * src0_row_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->data + c_i01*nb1 + c_i02*nb2 + c_i03*nb3;               \
        dma_queue_push(dma_q, dma_make_data(c_dst_addr, dst_spad),                           \
                       nb1, dst_row_stride, dst_row_size, nr);                               \
        cpy_rows_advance(c_r, c_i01, c_i02, c_i03, nr);                                      \
                                                                                             \
        if (f_r < ir1) {                                                                     \
            const uint32_t fnr = MIN(blk_rows, MIN(ne01 - f_i01, ir1 - f_r));                \
            dma_addr_t f_src0_addr = src0->data + f_i01*nb01 + f_i02*nb02 + f_i03*nb03;      \
            dma_queue_push(dma_q, dma_make_data(src_spad, f_src0_addr),                      \
                           src0_row_stride, nb01, src0_row_size, fnr);                       \
            cpy_rows_advance(f_r, f_i01, f_i02, f_i03, fnr);                                 \
        }                                                                                    \
    }                                                                                        \
    dma_queue_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

// Tile origin of the transpose walk: (a, b) element offsets plus the outer indices.
struct cpy_tile_pos {
    uint32_t a;
    uint32_t b;
    uint32_t i2;
    uint32_t i3;
};

static inline void cpy_tile_pos_init(struct cpy_tile_pos *                    p,
                                     const struct htp_copy_transpose_params * tp,
                                     uint32_t                                 tile) {
    const uint32_t ta = tile % tp->n_tiles_a;
    tile /= tp->n_tiles_a;
    const uint32_t tb = tile % tp->n_tiles_b;
    tile /= tp->n_tiles_b;
    p->a  = ta * tp->tile_a;
    p->b  = tb * tp->tile_b;
    p->i2 = tile % tp->ne2;
    p->i3 = tile / tp->ne2;
}

static inline void cpy_tile_pos_next(struct cpy_tile_pos * p, const struct htp_copy_transpose_params * tp) {
    p->a += tp->tile_a;
    if (p->a >= tp->ne_a) {
        p->a = 0;
        p->b += tp->tile_b;
        if (p->b >= tp->ne_b) {
            p->b = 0;
            if (++p->i2 == tp->ne2) {
                p->i2 = 0;
                p->i3++;
            }
        }
    }
}

#define DEFINE_CPY_TRANSPOSE(BITS, GATHER)                                                                       \
    static void cpy_thread_transpose_##BITS(unsigned int nth, unsigned int ith, void * data) {                   \
        struct htp_copy_context *                ct   = (struct htp_copy_context *) data;                        \
        struct htp_ops_context *                 octx = ct->octx;                                                \
        const struct htp_tensor *                src0 = octx->src[0];                                            \
        const struct htp_tensor *                dst  = octx->dst;                                               \
        const struct htp_copy_transpose_params * tp   = &ct->kparams->u.transpose;                               \
        const uint32_t                           es   = tp->elem_size;                                           \
                                                                                                                 \
        const uint32_t dt  = ct->tiles_per_thread;                                                               \
        const uint32_t it0 = ct->tile_start + dt * ith;                                                          \
        const uint32_t it1 = MIN(it0 + dt, ct->tile_start + ct->ntiles);                                         \
        if (it0 >= it1)                                                                                          \
            return;                                                                                              \
                                                                                                                 \
        dma_queue *               dma_q = octx->ctx->dma[ith];                                                   \
        struct htp_thread_trace * tr    = &octx->ctx->trace[ith];                                                \
                                                                                                                 \
        const uint32_t     a_pitch      = tp->a_pitch;                                                           \
        const uint32_t     b_pitch      = tp->b_pitch;                                                           \
        const uint32_t     period       = tp->period;                                                            \
        const uint32_t     period_elems = tp->period_elems;                                                      \
        const HVX_Vector * tab          = (const HVX_Vector *) ct->vtcm_src0;                                    \
        uint8_t *          spad         = ct->vtcm_src0 + tp->tab_size + ith * tp->spad_size_per_thread;         \
        uint8_t *          in_bufs      = spad;                                                                  \
        uint8_t *          out_bufs     = spad + 2 * tp->in_buf_size;                                            \
                                                                                                                 \
        struct cpy_tile_pos f, c;                                                                                \
        cpy_tile_pos_init(&f, tp, it0);                                                                          \
        c            = f;                                                                                        \
        uint32_t f_t = it0;                                                                                      \
                                                                                                                 \
        for (uint32_t k = 0; k < 2 && f_t < it1; ++k) {                                                          \
            uint8_t *      in_buf   = in_bufs + k * tp->in_buf_size;                                             \
            uint8_t *      out_buf  = out_bufs + k * tp->out_buf_size;                                           \
            const uint32_t na       = MIN(tp->tile_a, tp->ne_a - f.a);                                           \
            const uint32_t nb       = MIN(tp->tile_b, tp->ne_b - f.b);                                           \
            dma_addr_t     src_addr = src0->data + (dma_addr_t) f.a * tp->nb_a_src + (dma_addr_t) f.b * es +     \
                                  (dma_addr_t) f.i2 * tp->nb2_src + (dma_addr_t) f.i3 * tp->nb3_src;             \
            dma_queue_push(dma_q, dma_make_data(dst->data, out_buf), es, es, es, 0);                             \
            dma_queue_push(dma_q, dma_make_data(in_buf, src_addr), a_pitch, tp->nb_a_src, nb * es, na);          \
            cpy_tile_pos_next(&f, tp);                                                                           \
            ++f_t;                                                                                               \
        }                                                                                                        \
                                                                                                                 \
        for (uint32_t t = it0; t < it1; ++t) {                                                                   \
            const uint32_t na      = MIN(tp->tile_a, tp->ne_a - c.a);                                            \
            const uint32_t nb      = MIN(tp->tile_b, tp->ne_b - c.b);                                            \
            uint8_t *      out_buf = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).src;                           \
            uint8_t *      in_buf  = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;                           \
            const uint32_t nvec    = (na * es + VLEN - 1) / VLEN;                                                \
                                                                                                                 \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (t - it0));                             \
            for (uint32_t j = 0; j < nb; ++j) {                                                                  \
                uint8_t *       d   = out_buf + j * b_pitch;                                                     \
                const uint8_t * col = in_buf + j * es;                                                           \
                uint32_t        e0  = 0;                                                                         \
                uint32_t        k   = 0;                                                                         \
                for (uint32_t v = 0; v < nvec; ++v) {                                                            \
                    GATHER((HVX_Vector *) (d + v * VLEN), (size_t) (col + e0 * a_pitch),                         \
                           (na - e0 - 1) * a_pitch + es - 1, tab[k]);                                            \
                    if (++k == period) {                                                                         \
                        k = 0;                                                                                   \
                        e0 += period_elems;                                                                      \
                    }                                                                                            \
                }                                                                                                \
            }                                                                                                    \
            hvx_gather_sync(out_buf);                                                                            \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (t - it0));                              \
                                                                                                                 \
            dma_addr_t dst_addr = dst->data + (dma_addr_t) c.a * es + (dma_addr_t) c.b * tp->nb_b_dst +          \
                                  (dma_addr_t) c.i2 * tp->nb2_dst + (dma_addr_t) c.i3 * tp->nb3_dst;             \
            dma_queue_push(dma_q, dma_make_data(dst_addr, out_buf), tp->nb_b_dst, b_pitch, na * es, nb);         \
            cpy_tile_pos_next(&c, tp);                                                                           \
                                                                                                                 \
            if (f_t < it1) {                                                                                     \
                const uint32_t fna      = MIN(tp->tile_a, tp->ne_a - f.a);                                       \
                const uint32_t fnb      = MIN(tp->tile_b, tp->ne_b - f.b);                                       \
                dma_addr_t     src_addr = src0->data + (dma_addr_t) f.a * tp->nb_a_src + (dma_addr_t) f.b * es + \
                                      (dma_addr_t) f.i2 * tp->nb2_src + (dma_addr_t) f.i3 * tp->nb3_src;         \
                dma_queue_push(dma_q, dma_make_data(in_buf, src_addr), a_pitch, tp->nb_a_src, fnb * es, fna);    \
                cpy_tile_pos_next(&f, tp);                                                                       \
                ++f_t;                                                                                           \
            }                                                                                                    \
        }                                                                                                        \
        dma_queue_flush(dma_q);                                                                                  \
    }

DEFINE_CPY_TRANSPOSE(32, Q6_vgather_ARMVw)
DEFINE_CPY_TRANSPOSE(16, Q6_vgather_ARMVh)

static int cpy_scalar(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
        return HTP_STATUS_OK;
    }

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->data, src0->data, kparams->src0_type_size);
        dma_queue_flush(octx->ctx->dma[0]);
        return HTP_STATUS_OK;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_base;
    dma_addr_t d_vtcm = s_vtcm + VLEN;
    const uint32_t s_size = kparams->src0_type_size;
    const uint32_t d_size = kparams->dst_type_size;

    dma_queue_push(dma_q, dma_make_data(s_vtcm, src0->data), s_size, s_size, s_size, 1);
    dma_queue_pop(dma_q);

    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_base;
    uint8_t * d_ptr = s_ptr + VLEN;

    const HVX_Vector v_src = hvx_vmem(s_ptr);
    HVX_Vector v_dst;

    if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } else if (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } else if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_to_f16(v_src, v_src);
    } else if (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src));
    } else {
        return HTP_STATUS_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_queue_push(dma_q, dma_make_data(dst->data, d_vtcm), d_size, d_size, d_size, 1);
    dma_queue_flush(dma_q);
    return HTP_STATUS_OK;
}

static int cpy_1d_contig(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->total_elems;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        elem_start = range.start;
        nelem      = range.count;
    }

    if (nelem > 0) {
        dma_queue * q = octx->ctx->dma[0];
        const uint32_t type_size = kparams->src0_type_size;
        dma_addr_t dst_addr  = dst->data  + elem_start * type_size;
        dma_addr_t src0_addr = src0->data + elem_start * type_size;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_size);
        dma_queue_flush(q);
    }

    return HTP_STATUS_OK;
}

static int cpy_sameshape_sametype(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->total_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows > 0) {
        dma_queue * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_size, row_start, nrows);
        dma_queue_flush(q);
    }

    return HTP_STATUS_OK;
}

static int cpy_sameshape_convert(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->total_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const uint32_t n_threads = octx->n_threads;
    const struct htp_copy_convert_params * cvt = &kparams->u.convert;

    struct htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.row_start             = row_start;
    ct.nrows                 = nrows;
    ct.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    ct.vtcm_src0 = vtcm_base;
    ct.vtcm_dst  = vtcm_base + (size_t) n_threads * cvt->spad0_size_per_thread;

    work_queue_func_t copy_fun = NULL;
    if (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } else if (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } else {
        return HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    return HTP_STATUS_OK;
}

static int cpy_reshape(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->total_elems;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        elem_start = range.start;
        nelem      = range.count;
    }

    if (nelem == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = octx->n_threads;

    struct htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_start      = elem_start;
    ct.nelem           = nelem;
    ct.elem_per_thread = fastdiv(nelem + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t copy_fun = NULL;
    switch (src0->type) {
        case HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; break;
        case HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; break;
        case HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; break;
        default: return HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    return HTP_STATUS_OK;
}

static int cpy_transpose(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct htp_copy_transpose_params * tp          = &kparams->u.transpose;
    const uint32_t                           total_tiles = tp->n_tiles_a * tp->n_tiles_b * tp->ne2 * tp->ne3;

    uint32_t tile_start = 0;
    uint32_t ntiles     = total_tiles;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_tiles, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        tile_start = range.start;
        ntiles     = range.count;
    }

    if (ntiles == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const uint32_t n_threads = octx->n_threads;

    struct htp_copy_context ct;
    ct.octx             = octx;
    ct.kparams          = kparams;
    ct.tile_start       = tile_start;
    ct.ntiles           = ntiles;
    ct.tiles_per_thread = fastdiv(ntiles + n_threads - 1, &octx->n_threads_div);
    ct.vtcm_src0        = (uint8_t *) octx->ctx->vtcm_base;
    ct.vtcm_dst         = NULL;

    // one period of gather offsets, shared by all threads: lane q reads part q % w of element q / w
    const uint32_t lane = htp_copy_transpose_lane_size(tp->elem_size);
    const uint32_t w    = tp->elem_size / lane;
    const uint32_t nq   = tp->period * (VLEN / lane);
    uint32_t       i    = 0;
    uint32_t       p    = 0;

    work_queue_func_t copy_fun = NULL;
    if (lane == 4) {
        int32_t * tab = (int32_t *) octx->ctx->vtcm_base;
        for (uint32_t q = 0; q < nq; q++) {
            tab[q] = (int32_t) (i * tp->a_pitch + p * lane);
            if (++p == w) {
                p = 0;
                i++;
            }
        }
        copy_fun = cpy_thread_transpose_32;
    } else {
        int16_t * tab = (int16_t *) octx->ctx->vtcm_base;
        for (uint32_t q = 0; q < nq; q++) {
            tab[q] = (int16_t) (i * tp->a_pitch + p * lane);
            if (++p == w) {
                p = 0;
                i++;
            }
        }
        copy_fun = cpy_thread_transpose_16;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    return HTP_STATUS_OK;
}

int op_cpy(struct htp_ops_context * octx) {
    const struct htp_copy_kernel_params * kparams = (const struct htp_copy_kernel_params *) octx->kernel_params;
    int status = HTP_STATUS_OK;

    switch (kparams->kernel_type) {
        case HTP_COPY_KERNEL_SCALAR:
            status = cpy_scalar(octx, kparams);
            break;
        case HTP_COPY_KERNEL_1D_CONTIG:
            status = cpy_1d_contig(octx, kparams);
            break;
        case HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            status = cpy_sameshape_sametype(octx, kparams);
            break;
        case HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            status = cpy_sameshape_convert(octx, kparams);
            break;
        case HTP_COPY_KERNEL_RESHAPE:
            status = cpy_reshape(octx, kparams);
            break;
        case HTP_COPY_KERNEL_TRANSPOSE:
            status = cpy_transpose(octx, kparams);
            break;
        default:
            status = HTP_STATUS_NO_SUPPORT;
            break;
    }

    htp_ops_context_set_status(octx, status);

    if (octx->ctx->mdev.count > 1) {
        htp_mdev_group_barrier(octx);
    }

    if (octx->op == HTP_OP_CPY_FENCE) {
        if (octx->ctx->mdev.idx == 0) {
            const struct htp_tensor * sync = octx->src[1];
            const uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;
            htp_fence_write(sync_fence, seq, octx->status);

            FARF(HIGH, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        }
    }

    return octx->status;
}
