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
#include "gather-rows.h"

struct htp_copy_context {
    struct htp_ops_context *              octx;
    const struct htp_copy_kernel_params * kparams;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t src0_nrows_per_thread;

    uint32_t elem_start;
    uint32_t nelem;
    uint32_t elem_per_thread;

    struct htp_copy_convert_vtcm_layout layout;
    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;

    uint32_t tile_start;
    uint32_t ntiles;
    uint32_t tiles_per_thread;
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

struct cpy_row_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

static inline void cpy_cursor_advance(struct cpy_row_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
    c->r   += n;
    c->i01 += n;
    if (c->i01 == ne01) {
        c->i01 = 0;
        if (++c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

struct cpy_tile_cursor {
    uint32_t ibk;
    uint32_t ib0;
    uint32_t i2;
    uint32_t i3;
};

static inline void cpy_tile_cursor_advance(struct cpy_tile_cursor * c, const struct htp_copy_transpose_params * tp) {
    if (++c->ibk == tp->nkb) {
        c->ibk = 0;
        if (++c->ib0 == tp->n0b) {
            c->ib0 = 0;
            if (++c->i2 == tp->ne[2]) {
                c->i2 = 0;
                c->i3++;
            }
        }
    }
}

// Gather offsets for element e = l >> log2_m at (e % r, e / r)
static inline HVX_Vector cpy_transpose_offsets(const struct htp_copy_transpose_params * tp) {
    const uint32_t r_mask = (1u << tp->log2_r) - 1;
    if (tp->elem_size != 2) {
        const uint32_t m_mask = (1u << tp->log2_m) - 1;
        int32_t offsets[32] __attribute__((aligned(128)));
        for (uint32_t l = 0; l < 32; l++) {
            const uint32_t e = l >> tp->log2_m;
            offsets[l] = (e & r_mask) * tp->in_stride + (e >> tp->log2_r) * tp->elem_size + (l & m_mask) * 4;
        }
        return *(HVX_Vector *) offsets;
    }
    int16_t offsets[64] __attribute__((aligned(128)));
    for (uint32_t l = 0; l < 64; l++) {
        offsets[l] = (l & r_mask) * tp->in_stride + (l >> tp->log2_r) * 2;
    }
    return *(HVX_Vector *) offsets;
}

static inline void cpy_transpose_push_in(dma_queue * q, const struct htp_copy_transpose_params * tp, dma_addr_t src,
                                         const struct cpy_tile_cursor * c, uint8_t * buf) {
    const uint32_t a0  = c->ib0 * tp->blk0;
    const uint32_t ak  = c->ibk * tp->blkk;
    const uint32_t off = a0 * tp->src_nb[0] + ak * tp->elem_size + c->i2 * tp->src_nb[1] + c->i3 * tp->src_nb[2];
    dma_queue_push(q, dma_make_data(buf, src + off), tp->in_stride, tp->src_nb[0],
                   MIN(tp->blkk, tp->ne[1] - ak) * tp->elem_size, MIN(tp->blk0, tp->ne[0] - a0));
}

// Pipelined tile worker: DMA in -> vgather transpose -> DMA out
static void cpy_thread_transpose(unsigned int nth, unsigned int ith, void * data) {
    struct htp_copy_context * ct = (struct htp_copy_context *) data;
    struct htp_ops_context * octx = ct->octx;
    const struct htp_copy_transpose_params * tp = &ct->kparams->u.transpose;

    const uint32_t t0 = ct->tile_start + ith * ct->tiles_per_thread;
    const uint32_t t1 = MIN(t0 + ct->tiles_per_thread, ct->tile_start + ct->ntiles);
    if (t0 >= t1) {
        return;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const dma_addr_t src = octx->src[0]->data;
    const dma_addr_t dst = octx->dst->data;

    const uint32_t es         = tp->elem_size;
    const uint32_t ne0        = tp->ne[0];
    const uint32_t nek        = tp->ne[1];
    const uint32_t in_stride  = tp->in_stride;
    const uint32_t out_stride = tp->out_stride;

    const HVX_Vector offsets   = cpy_transpose_offsets(tp);
    const uint32_t packed_mu   = (ne0 - 1) * in_stride + 127;
    const uint32_t packed_step = 128 >> tp->log2_r;
    const uint32_t epv         = 1u << tp->log2_r;

    uint8_t * in_buf[2];
    uint8_t * out_buf[2];
    in_buf[0]  = ct->vtcm_src0 + ith * 2 * tp->in_buf_size;
    in_buf[1]  = in_buf[0] + tp->in_buf_size;
    out_buf[0] = ct->vtcm_dst + ith * 2 * tp->out_buf_size;
    out_buf[1] = out_buf[0] + tp->out_buf_size;

    struct cpy_tile_cursor cur;
    {
        const uint32_t q0 = fastdiv(t0, &tp->div_nkb);
        const uint32_t q1 = fastdiv(q0, &tp->div_n0b);
        cur.ibk = t0 - q0 * tp->nkb;
        cur.ib0 = q0 - q1 * tp->n0b;
        cur.i3  = fastdiv(q1, &tp->div_ne2);
        cur.i2  = q1 - cur.i3 * tp->ne[2];
    }

    struct cpy_tile_cursor nxt = cur;

    cpy_transpose_push_in(dma_q, tp, src, &cur, in_buf[0]);

    for (uint32_t t = t0, i = 0; t < t1; t++, i++) {
        while (dma_queue_depth(dma_q) > (i > 0 ? 1u : 0u)) {
            dma_queue_pop(dma_q);
        }

        uint8_t * in  = in_buf[i & 1];
        uint8_t * out = out_buf[i & 1];

        const uint32_t a0 = cur.ib0 * tp->blk0;
        const uint32_t ak = cur.ibk * tp->blkk;
        const uint32_t n0 = MIN(tp->blk0, ne0 - a0);
        const uint32_t nk = MIN(tp->blkk, nek - ak);

        if (t + 1 < t1) {
            cpy_tile_cursor_advance(&nxt, tp);
            cpy_transpose_push_in(dma_q, tp, src, &nxt, in_buf[(i + 1) & 1]);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) t);
        if (tp->packed) {
            const uint32_t nvec = (nk * ne0 * es + 127) >> 7;
            if (es != 2) {
                #pragma unroll(4)
                for (uint32_t v = 0; v < nvec; v++) {
                    Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (size_t) (in + v * packed_step), packed_mu, offsets);
                }
            } else {
                #pragma unroll(4)
                for (uint32_t v = 0; v < nvec; v++) {
                    Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (size_t) (in + v * packed_step), packed_mu, offsets);
                }
            }
        } else {
            for (uint32_t rb = 0; rb < n0; rb += epv) {
                const uint32_t  mu = (MIN(epv, n0 - rb) - 1) * in_stride + es - 1;
                uint8_t *       d  = out + rb * es;
                const uint8_t * s  = in + rb * in_stride;
                if (es != 2) {
                    #pragma unroll(4)
                    for (uint32_t ik = 0; ik < nk; ik++) {
                        Q6_vgather_ARMVw((HVX_Vector *) (d + ik * out_stride), (size_t) (s + ik * es), mu, offsets);
                    }
                } else {
                    #pragma unroll(4)
                    for (uint32_t ik = 0; ik < nk; ik++) {
                        Q6_vgather_ARMVh((HVX_Vector *) (d + ik * out_stride), (size_t) (s + ik * 2), mu, offsets);
                    }
                }
            }
        }
        hvx_gather_sync(out);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) t);

        const uint32_t doff = a0 * es + ak * tp->dst_nb[0] + cur.i2 * tp->dst_nb[1] + cur.i3 * tp->dst_nb[2];
        if (tp->packed) {
            const uint32_t n = nk * ne0 * es;
            dma_queue_push(dma_q, dma_make_data(dst + doff, out), n, n, n, 1);
        } else {
            dma_queue_push(dma_q, dma_make_data(dst + doff, out), tp->dst_nb[0], out_stride, n0 * es, nk);
        }

        cur = nxt;
    }

    dma_queue_flush(dma_q);
}

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
    const struct htp_copy_convert_vtcm_layout * L = &ct->layout;                             \
    const uint32_t src0_buf_size   = L->src0_buf_size;                                       \
    const uint32_t dst_buf_size    = L->dst_buf_size;                                        \
    const uint32_t src0_row_stride = L->src0_row_stride;                                     \
    const uint32_t dst_row_stride  = L->dst_row_stride;                                      \
    const uint32_t blk_rows        = L->blk_rows;                                            \
    uint8_t * vtcm_src0_base = ct->vtcm_src0 + ith * L->spad0_size_per_thread;               \
    uint8_t * vtcm_dst_base  = ct->vtcm_dst  + ith * L->spad1_size_per_thread;               \
    const uint32_t src0_row_size = ne00 * ct->kparams->src0_type_size;                       \
    const uint32_t dst_row_size  = ne00 * ct->kparams->dst_type_size;                        \
                                                                                             \
    const uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    struct cpy_row_cursor f = { ir0, i01, i02, i03 };                                        \
    struct cpy_row_cursor c = f;                                                             \
                                                                                             \
    for (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_base + b * src0_buf_size;                             \
        uint8_t * dst_spad = vtcm_dst_base  + b * dst_buf_size;                              \
        const uint32_t nr = MIN(blk_rows, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->data + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_queue_push(dma_q, dma_make_data(dst->data, dst_spad),                            \
                       dst_row_size, dst_buf_size, dst_row_size, 0);                         \
        dma_queue_push(dma_q, dma_make_data(src_spad, f_src0_addr),                          \
                       src0_row_stride, nb01, src0_row_size, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    for (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        const uint32_t nr = MIN(blk_rows, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;               \
                                                                                             \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        for (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_row_stride, src_spad + r * src0_row_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->data + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_queue_push(dma_q, dma_make_data(c_dst_addr, dst_spad),                           \
                       nb1, dst_row_stride, dst_row_size, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        if (f.r < ir1) {                                                                     \
            const uint32_t fnr = MIN(blk_rows, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->data + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_queue_push(dma_q, dma_make_data(src_spad, f_src0_addr),                      \
                           src0_row_stride, nb01, src0_row_size, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_queue_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

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
    uint8_t * s_ptr = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, 0);
    uint8_t * d_ptr = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, VLEN);
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) s_ptr;
    dma_addr_t d_vtcm = (dma_addr_t)(uintptr_t) d_ptr;
    const uint32_t s_size = kparams->src0_type_size;
    const uint32_t d_size = kparams->dst_type_size;

    dma_queue_push(dma_q, dma_make_data(s_vtcm, src0->data), s_size, s_size, s_size, 1);
    dma_queue_pop(dma_q);

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
    htp_copy_convert_vtcm_layout_build(&ct.layout, src0->ne[0], kparams->src0_type_size, kparams->dst_type_size, cvt->blk_rows, n_threads);

    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.row_start             = row_start;
    ct.nrows                 = nrows;
    ct.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    ct.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, ct.layout.off_src0);
    ct.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, ct.layout.off_dst);

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
    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct htp_copy_transpose_params * tp = &kparams->u.transpose;

    uint32_t tile_start = 0;
    uint32_t ntiles     = tp->total_tiles;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            tp->total_tiles, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
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

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    ct.vtcm_src0 = vtcm_base;
    ct.vtcm_dst  = vtcm_base + (size_t) n_threads * 2 * tp->in_buf_size;

    work_queue_run(octx->ctx->work_queue, cpy_thread_transpose, &ct, n_threads);
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
        case HTP_COPY_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, octx->src[0], NULL, &kparams->u.gather, kparams->n_threads);
            break;
        default:
            status = HTP_STATUS_NO_SUPPORT;
            break;
    }

    htp_ops_context_set_status(octx, status);

    if (octx->op == HTP_OP_CPY_FENCE) {
        if (octx->ctx->mdev.count > 1) {
            htp_mdev_group_barrier(octx);
        }

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
