#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-utils.h"

struct htp_copy_context {
    struct htp_ops_context * octx;

    uint32_t          src0_type_size;
    uint32_t          src0_block_size;

    uint32_t          dst_type_size;
    uint32_t          dst_block_size;

    uint32_t          src0_blocks_per_row;
    uint32_t          dst_blocks_per_row;

    uint32_t          src0_nrows_per_thread;

    void (*copy)(struct htp_copy_context * ct, struct htp_ops_context * octx, int nth, int ith);
};

#define cpy_preamble                       \
    struct htp_tensor *src0 = &octx->src0; \
    struct htp_tensor *dst  = &octx->dst;  \
                                           \
    const uint32_t ne00 = src0->ne[0];     \
    const uint32_t ne01 = src0->ne[1];     \
    const uint32_t ne02 = src0->ne[2];     \
    const uint32_t ne03 = src0->ne[3];     \
                                           \
    const uint32_t nb00 = src0->nb[0];     \
    const uint32_t nb01 = src0->nb[1];     \
    const uint32_t nb02 = src0->nb[2];     \
    const uint32_t nb03 = src0->nb[3];     \
                                           \
    const uint32_t  ne0 = dst->ne[0];      \
    const uint32_t  ne1 = dst->ne[1];      \
    const uint32_t  ne2 = dst->ne[2];      \
    const uint32_t  ne3 = dst->ne[3];      \
                                           \
    const uint32_t  nb0 = dst->nb[0];      \
    const uint32_t  nb1 = dst->nb[1];      \
    const uint32_t  nb2 = dst->nb[2];      \
    const uint32_t  nb3 = dst->nb[3];      \
                                           \
    const uint32_t   nr = ne01;

static void cpy_thread_sametype_sameshape(struct htp_copy_context * ct, struct htp_ops_context * octx, const int nth, const int ith) {
    cpy_preamble;

    // parallelize by src0 rows
    const uint32_t dr  = ct->src0_nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = (ir0 + dr) < nr ? (ir0 + dr) : nr;

    // copy by rows
    for (uint32_t i03 = 0; i03 < ne03; i03++) {
        for (uint32_t i02 = 0; i02 < ne02; i02++) {
            #pragma unroll(2)
            for (uint32_t i01 = ir0; i01 < ir1; i01++) {
                uint8_t* dst_ptr  = (uint8_t*) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;
                uint8_t* src0_ptr = (uint8_t*) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                hex_l2fetch(src0_ptr, ne00 * ct->src0_type_size, nb01, 2);
                hvx_copy_uu(dst_ptr, src0_ptr, ne00, ct->src0_type_size);
            }
        }
    }
}

static void cpy_thread_sametype_reshape(struct htp_copy_context * ct, struct htp_ops_context * octx, int nth, int ith) {
    cpy_preamble;

    // parallelize by src0 rows
    const uint32_t dr  = ct->src0_nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = (ir0 + dr) < nr ? (ir0 + dr) : nr;

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    const int64_t nk00 = ct->src0_blocks_per_row;
    const int64_t nk0  = ct->dst_blocks_per_row;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            while (k10 >= nk0) {
                k10 -= nk0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            for (int64_t i01 = ir0; i01 < ir1; i01++) {
                for (int64_t k00 = 0; k00 < nk00; k00++) {
                    const char * src0_ptr = ((char *) src0->data + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          char * dst_ptr  = ((char *)  dst->data + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);
                    memcpy(dst_ptr, src0_ptr, ct->dst_type_size);

                    if (++k10 == nk0) {
                        k10 = 0;
                        if (++i11 == ne1) {
                            i11 = 0;
                            if (++i12 == ne2) {
                                i12 = 0;
                                if (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            while (k10 >= nk0) {
                k10 -= nk0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

static void cpy_thread_f16_f32_sameshape(struct htp_copy_context * ct, struct htp_ops_context * octx, const int nth, const int ith) {
    cpy_preamble;

    // parallelize by src0 rows
    const uint32_t dr  = ct->src0_nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = (ir0 + dr) < nr ? (ir0 + dr) : nr;

    // copy by rows
    for (uint32_t i03 = 0; i03 < ne03; i03++) {
        for (uint32_t i02 = 0; i02 < ne02; i02++) {
            #pragma unroll(2)
            for (uint32_t i01 = ir0; i01 < ir1; i01++) {
                uint8_t* dst_ptr  = (uint8_t*) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;
                uint8_t* src0_ptr = (uint8_t*) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                hex_l2fetch(src0_ptr, ne00 * sizeof(float), nb01, 2);
                hvx_copy_f16_f32_uu(dst_ptr, src0_ptr, ne00);
            }
        }
    }
}

static void cpy_thread_f32_f16_sameshape(struct htp_copy_context * ct, struct htp_ops_context * octx, const int nth, const int ith) {
    cpy_preamble;

    // parallelize by src0 rows
    const uint32_t dr  = ct->src0_nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = (ir0 + dr) < nr ? (ir0 + dr) : nr;

    // copy by rows
    for (uint32_t i03 = 0; i03 < ne03; i03++) {
        for (uint32_t i02 = 0; i02 < ne02; i02++) {
            #pragma unroll(2)
            for (uint32_t i01 = ir0; i01 < ir1; i01++) {
                uint8_t* dst_ptr  = (uint8_t*) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;
                uint8_t* src0_ptr = (uint8_t*) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                hex_l2fetch(src0_ptr, ne00 * sizeof(__fp16), nb01, 2);
                hvx_copy_f32_f16_uu(dst_ptr, src0_ptr, ne00);
            }
        }
    }
}

static void cpy_work_func(unsigned int n, unsigned int i, void *data) {
    struct htp_copy_context *ct = (struct htp_copy_context *) data;
    ct->copy(ct, ct->octx, n, i);
}

int op_cpy(struct htp_ops_context * octx) {
    cpy_preamble;

    struct htp_copy_context ct;
    ct.octx = octx;

    switch (src0->type) {
    case HTP_TYPE_F32: ct.src0_type_size = 4; ct.src0_block_size = 1; ct.src0_blocks_per_row = ne00 / 1; break;
    case HTP_TYPE_F16: ct.src0_type_size = 2; ct.src0_block_size = 1; ct.src0_blocks_per_row = ne00 / 1; break;
    default:
        return HTP_STATUS_NO_SUPPORT;
    }

    switch (dst->type) {
    case HTP_TYPE_F32: ct.dst_type_size = 4; ct.dst_block_size = 1; ct.dst_blocks_per_row = ne0 / 1; break;
    case HTP_TYPE_F16: ct.dst_type_size = 2; ct.dst_block_size = 1; ct.dst_blocks_per_row = ne0 / 1; break;
    default:
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const bool sametype   = (src0->type == dst->type);
    const bool transposed = (nb00 > nb01) || (nb0 > nb1);
    const bool sameshape  = !transposed && (ne00 == ne0 && ne01 == ne1 && ne02 == ne2 && ne03 == ne3);

    const uint32_t n_jobs = MIN(nr, octx->n_threads);
    ct.src0_nrows_per_thread = (nr + n_jobs - 1) / n_jobs;

    if (sametype && sameshape) {
        ct.copy = cpy_thread_sametype_sameshape;
    } else if (sameshape) {
        /**/ if (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32)
            ct.copy = cpy_thread_f16_f32_sameshape;
        else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16)
            ct.copy = cpy_thread_f32_f16_sameshape;
        else
            return HTP_STATUS_NO_SUPPORT;
    } else if (sametype) {
        ct.copy = cpy_thread_sametype_reshape;
    } else {
        return HTP_STATUS_NO_SUPPORT;
    }

    worker_pool_run_func(octx->ctx->worker_pool, cpy_work_func, &ct, n_jobs);

    return HTP_STATUS_OK;
}
