// Depthwise 2D convolution (CONV_2D_DW) kernels for all supported filter sizes.

#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>
#include <stdint.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "conv2d-ops.h"
#include "dma-copy.h"
#include "dma-queue.h"
#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-trace.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hvx-base.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"

struct conv2d_dw_task_slice {
    dma_addr_t src_addr;
    dma_addr_t dst_addr;
    uint32_t   src_bytes;
    uint32_t   dst_bytes;
    uint32_t   ih_count;
    uint32_t   top_pad;
    uint32_t   bot_pad;
    uint32_t   oh_start;
    uint32_t   oh_len;
    int32_t    ih_first;
    uint32_t   c;
};

static inline void conv2d_dw_get_slice(
    struct conv2d_dw_task_slice * slice,
    uint32_t task,
    const struct htp_tensor * src1,
    const struct htp_tensor * dst,
    const struct htp_conv2d_dw_kernel_params * kp) {

    uint32_t chunk_idx;
    uint32_t n;
    uint32_t c = 0;

    if (kp->cwhn) {
        n = fastdiv(task, &kp->div_n_chunks);
        chunk_idx = task - n * kp->n_chunks;
    } else {
        const uint32_t plane = fastdiv(task, &kp->div_n_chunks);
        chunk_idx = task - plane * kp->n_chunks;
        n = fastdiv(plane, &kp->div_c);
        c = plane - n * kp->c;
    }

    const uint32_t oh_start = chunk_idx * kp->oh_chunk;
    const uint32_t oh_len   = MIN(kp->oh_chunk, kp->oh - oh_start);

    const int32_t  ih_min   = (int32_t)(oh_start * kp->s1) - kp->p1;
    const int32_t  ih_max   = (int32_t)((oh_start + oh_len - 1) * kp->s1 + (kp->kh - 1) * kp->d1) - kp->p1;
    const uint32_t ih_span  = (uint32_t)(ih_max - ih_min + 1);

    const int32_t  ih_first = MAX(0, ih_min);
    const int32_t  ih_last  = MIN((int32_t)kp->h - 1, ih_max);
    const uint32_t ih_count = (ih_first <= ih_last) ? (uint32_t)(ih_last - ih_first + 1) : 0;
    const uint32_t top_pad  = (ih_count > 0) ? (uint32_t)(ih_first - ih_min) : ih_span;
    const uint32_t bot_pad  = ih_span - top_pad - ih_count;

    slice->src_bytes = ih_count * kp->row_src_bytes;
    slice->dst_bytes = oh_len   * kp->row_dst_bytes;
    slice->ih_count  = ih_count;
    slice->top_pad   = top_pad;
    slice->bot_pad   = bot_pad;
    slice->oh_start  = oh_start;
    slice->oh_len    = oh_len;
    slice->ih_first  = ih_first;
    slice->c         = c;

    if (kp->cwhn) {
        slice->src_addr = src1->data + (size_t) n * src1->nb[3] + (size_t) ih_first * src1->nb[1];
        slice->dst_addr = dst->data  + (size_t) n * dst->nb[3]  + (size_t) oh_start * dst->nb[1];
    } else {
        slice->src_addr = src1->data + (size_t) n * src1->nb[3] + (size_t) c * src1->nb[2] + (size_t) ih_first * src1->nb[1];
        slice->dst_addr = dst->data  + (size_t) n * dst->nb[3]  + (size_t) c * dst->nb[2] + (size_t) oh_start * dst->nb[1];
    }
}

static inline HVX_Vector conv2d_dw_splat_w_f32(const void * ptr) {
    return hvx_vec_splat_f32(*(const float *) ptr);
}

static inline HVX_Vector conv2d_dw_splat_w_f16(const void * ptr) {
    const HVX_Vector vh = Q6_Vh_vsplat_R(*(const uint16_t *) ptr);
    const HVX_Vector v1 = Q6_Vh_vsplat_R(0x3c00);
    return Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(vh, v1)));
}

static inline HVX_Vector conv2d_dw_load_w_f32(const void * ptr) {
    return hvx_vmemu(ptr);
}

static inline HVX_Vector conv2d_dw_load_w_f16(const void * ptr) {
    return Q6_V_lo_W(hvx_vec_f16_to_f32(hvx_vmemu(ptr)));
}

static inline HVX_Vector conv2d_dw_qf32_mpy_add3(
    HVX_Vector acc,
    HVX_Vector x0, HVX_Vector w0,
    HVX_Vector x1, HVX_Vector w1,
    HVX_Vector x2, HVX_Vector w2) {
    const HVX_Vector p01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0), Q6_Vqf32_vmpy_VsfVsf(x1, w1));
    const HVX_Vector p2  = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x2, w2));
    return Q6_Vqf32_vadd_Vqf32Vqf32(p01, p2);
}

static inline void conv2d_dw_pad_vtcm_k3x3(
    float * in_vtcm,
    const struct conv2d_dw_task_slice * slice,
    uint32_t w_in,
    uint32_t extra_bot_rows) {
    if (slice->top_pad > 0) {
        hvx_splat_f32_a(in_vtcm, 0.0f, slice->top_pad * w_in);
    }
    const uint32_t tail_zero_rows = slice->bot_pad + extra_bot_rows;
    if (tail_zero_rows > 0) {
        hvx_splat_f32_a(in_vtcm + (size_t) (slice->top_pad + slice->ih_count) * w_in, 0.0f, tail_zero_rows * w_in);
    }
}

#define DEFINE_CONV2D_DW_WHCN_S1_K3X3(SUFFIX, SPLAT_W_FN)                                             \
__attribute__((noinline))                                                                             \
static void conv2d_dw_whcn_s1_k3x3_##SUFFIX(                                                          \
    const float * restrict in_vtcm,                                                                   \
    float * restrict       out_vtcm,                                                                  \
    const void *           w_vtcm,                                                                    \
    const struct conv2d_dw_task_slice * slice,                                                        \
    const struct htp_ops_context * octx,                                                              \
    const struct htp_conv2d_dw_kernel_params * kp) {                                                  \
                                                                                                      \
    const struct htp_tensor * src0 = octx->src[0];                                                    \
    const size_t w_nb0             = src0->nb[0];                                                     \
    const size_t w_nb1             = src0->nb[1];                                                     \
    const char * restrict w_chan   = (const char *) w_vtcm + (size_t) slice->c * src0->nb[3];         \
    const uint32_t oh_len          = slice->oh_len;                                                   \
    const uint32_t w_in            = kp->w;                                                           \
                                                                                                      \
    conv2d_dw_pad_vtcm_k3x3((float *) in_vtcm, slice, w_in, 0);                                       \
                                                                                                      \
    const HVX_Vector w00  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w01  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w02  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector w10  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w11  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w12  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector w20  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w21  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w22  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector zero = Q6_V_vzero();                                                             \
                                                                                                      \
    uint32_t ohi = 0;                                                                                 \
    for (; ohi + 4 <= oh_len; ohi += 4) {                                                             \
        const float * restrict r0 = in_vtcm + (size_t) ohi * w_in;                                    \
        const float * restrict r1 = r0 + w_in;                                                        \
        const float * restrict r2 = r1 + w_in;                                                        \
        const float * restrict r3 = r2 + w_in;                                                        \
        const float * restrict r4 = r3 + w_in;                                                        \
        const float * restrict r5 = r4 + w_in;                                                        \
                                                                                                      \
        float * restrict out0 = out_vtcm + (size_t) (ohi + 0) * w_in;                                 \
        float * restrict out1 = out0 + w_in;                                                          \
        float * restrict out2 = out1 + w_in;                                                          \
        float * restrict out3 = out2 + w_in;                                                          \
                                                                                                      \
        HVX_Vector p0 = zero, c0 = hvx_vmem(r0);                                                      \
        HVX_Vector p1 = zero, c1 = hvx_vmem(r1);                                                      \
        HVX_Vector p2 = zero, c2 = hvx_vmem(r2);                                                      \
        HVX_Vector p3 = zero, c3 = hvx_vmem(r3);                                                      \
        HVX_Vector p4 = zero, c4 = hvx_vmem(r4);                                                      \
        HVX_Vector p5 = zero, c5 = hvx_vmem(r5);                                                      \
                                                                                                      \
        for (uint32_t ox = 0; ox < w_in; ox += 32) {                                                  \
            const bool has_next = (ox + 32 < w_in);                                                   \
                                                                                                      \
            const HVX_Vector n0  = has_next ? hvx_vmem(r0 + ox + 32) : zero;                          \
            const HVX_Vector l0  = Q6_V_valign_VVR(c0, p0, 124);                                      \
            const HVX_Vector rg0 = Q6_V_valign_VVR(n0, c0, 4);                                        \
            HVX_Vector acc0      = conv2d_dw_qf32_mpy_add3(zero, l0, w00, c0, w01, rg0, w02);         \
            p0 = c0; c0 = n0;                                                                         \
                                                                                                      \
            const HVX_Vector n1  = has_next ? hvx_vmem(r1 + ox + 32) : zero;                          \
            const HVX_Vector l1  = Q6_V_valign_VVR(c1, p1, 124);                                      \
            const HVX_Vector rg1 = Q6_V_valign_VVR(n1, c1, 4);                                        \
            acc0                 = conv2d_dw_qf32_mpy_add3(acc0, l1, w10, c1, w11, rg1, w12);         \
            HVX_Vector acc1      = conv2d_dw_qf32_mpy_add3(zero, l1, w00, c1, w01, rg1, w02);         \
            p1 = c1; c1 = n1;                                                                         \
                                                                                                      \
            const HVX_Vector n2  = has_next ? hvx_vmem(r2 + ox + 32) : zero;                          \
            const HVX_Vector l2  = Q6_V_valign_VVR(c2, p2, 124);                                      \
            const HVX_Vector rg2 = Q6_V_valign_VVR(n2, c2, 4);                                        \
            acc0                 = conv2d_dw_qf32_mpy_add3(acc0, l2, w20, c2, w21, rg2, w22);         \
            hvx_vmem(out0 + ox)  = Q6_Vsf_equals_Vqf32(acc0);                                         \
            acc1                 = conv2d_dw_qf32_mpy_add3(acc1, l2, w10, c2, w11, rg2, w12);         \
            HVX_Vector acc2      = conv2d_dw_qf32_mpy_add3(zero, l2, w00, c2, w01, rg2, w02);         \
            p2 = c2; c2 = n2;                                                                         \
                                                                                                      \
            const HVX_Vector n3  = has_next ? hvx_vmem(r3 + ox + 32) : zero;                          \
            const HVX_Vector l3  = Q6_V_valign_VVR(c3, p3, 124);                                      \
            const HVX_Vector rg3 = Q6_V_valign_VVR(n3, c3, 4);                                        \
            acc1                 = conv2d_dw_qf32_mpy_add3(acc1, l3, w20, c3, w21, rg3, w22);         \
            hvx_vmem(out1 + ox)  = Q6_Vsf_equals_Vqf32(acc1);                                         \
            acc2                 = conv2d_dw_qf32_mpy_add3(acc2, l3, w10, c3, w11, rg3, w12);         \
            HVX_Vector acc3      = conv2d_dw_qf32_mpy_add3(zero, l3, w00, c3, w01, rg3, w02);         \
            p3 = c3; c3 = n3;                                                                         \
                                                                                                      \
            const HVX_Vector n4  = has_next ? hvx_vmem(r4 + ox + 32) : zero;                          \
            const HVX_Vector l4  = Q6_V_valign_VVR(c4, p4, 124);                                      \
            const HVX_Vector rg4 = Q6_V_valign_VVR(n4, c4, 4);                                        \
            acc2                 = conv2d_dw_qf32_mpy_add3(acc2, l4, w20, c4, w21, rg4, w22);         \
            hvx_vmem(out2 + ox)  = Q6_Vsf_equals_Vqf32(acc2);                                         \
            acc3                 = conv2d_dw_qf32_mpy_add3(acc3, l4, w10, c4, w11, rg4, w12);         \
            p4 = c4; c4 = n4;                                                                         \
                                                                                                      \
            const HVX_Vector n5  = has_next ? hvx_vmem(r5 + ox + 32) : zero;                          \
            const HVX_Vector l5  = Q6_V_valign_VVR(c5, p5, 124);                                      \
            const HVX_Vector rg5 = Q6_V_valign_VVR(n5, c5, 4);                                        \
            acc3                 = conv2d_dw_qf32_mpy_add3(acc3, l5, w20, c5, w21, rg5, w22);         \
            hvx_vmem(out3 + ox)  = Q6_Vsf_equals_Vqf32(acc3);                                         \
            p5 = c5; c5 = n5;                                                                         \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    for (; ohi < oh_len; ++ohi) {                                                                     \
        const float * restrict r0 = in_vtcm + (size_t) ohi * w_in;                                    \
        const float * restrict r1 = r0 + w_in;                                                        \
        const float * restrict r2 = r1 + w_in;                                                        \
        float * restrict out_row  = out_vtcm + (size_t) ohi * w_in;                                   \
                                                                                                      \
        HVX_Vector p0 = zero, c0 = hvx_vmem(r0);                                                      \
        HVX_Vector p1 = zero, c1 = hvx_vmem(r1);                                                      \
        HVX_Vector p2 = zero, c2 = hvx_vmem(r2);                                                      \
                                                                                                      \
        for (uint32_t ox = 0; ox < w_in; ox += 32) {                                                  \
            const bool has_next = (ox + 32 < w_in);                                                   \
                                                                                                      \
            const HVX_Vector n0  = has_next ? hvx_vmem(r0 + ox + 32) : zero;                          \
            const HVX_Vector l0  = Q6_V_valign_VVR(c0, p0, 124);                                      \
            const HVX_Vector rg0 = Q6_V_valign_VVR(n0, c0, 4);                                        \
            HVX_Vector acc       = conv2d_dw_qf32_mpy_add3(zero, l0, w00, c0, w01, rg0, w02);         \
            p0 = c0; c0 = n0;                                                                         \
                                                                                                      \
            const HVX_Vector n1  = has_next ? hvx_vmem(r1 + ox + 32) : zero;                          \
            const HVX_Vector l1  = Q6_V_valign_VVR(c1, p1, 124);                                      \
            const HVX_Vector rg1 = Q6_V_valign_VVR(n1, c1, 4);                                        \
            acc                  = conv2d_dw_qf32_mpy_add3(acc, l1, w10, c1, w11, rg1, w12);          \
            p1 = c1; c1 = n1;                                                                         \
                                                                                                      \
            const HVX_Vector n2  = has_next ? hvx_vmem(r2 + ox + 32) : zero;                          \
            const HVX_Vector l2  = Q6_V_valign_VVR(c2, p2, 124);                                      \
            const HVX_Vector rg2 = Q6_V_valign_VVR(n2, c2, 4);                                        \
            acc                  = conv2d_dw_qf32_mpy_add3(acc, l2, w20, c2, w21, rg2, w22);          \
            p2 = c2; c2 = n2;                                                                         \
                                                                                                      \
            hvx_vmem(out_row + ox) = Q6_Vsf_equals_Vqf32(acc);                                        \
        }                                                                                             \
    }                                                                                                 \
}

DEFINE_CONV2D_DW_WHCN_S1_K3X3(f32, conv2d_dw_splat_w_f32)
DEFINE_CONV2D_DW_WHCN_S1_K3X3(f16, conv2d_dw_splat_w_f16)

#define DEFINE_CONV2D_DW_WHCN_S2_K3X3_W32(SUFFIX, SPLAT_W_FN)                                         \
__attribute__((noinline))                                                                             \
static void conv2d_dw_whcn_s2_k3x3_w32_##SUFFIX(                                                      \
    const float * restrict in_vtcm,                                                                   \
    float * restrict       out_vtcm,                                                                  \
    const void *           w_vtcm,                                                                    \
    const struct conv2d_dw_task_slice * slice,                                                        \
    const struct htp_ops_context * octx,                                                              \
    const struct htp_conv2d_dw_kernel_params * kp) {                                                  \
                                                                                                      \
    (void) kp;                                                                                        \
    const struct htp_tensor * src0 = octx->src[0];                                                    \
    const size_t w_nb0             = src0->nb[0];                                                     \
    const size_t w_nb1             = src0->nb[1];                                                     \
    const char * restrict w_chan   = (const char *) w_vtcm + (size_t) slice->c * src0->nb[3];         \
    const uint32_t oh_len          = slice->oh_len;                                                   \
                                                                                                      \
    conv2d_dw_pad_vtcm_k3x3((float *) in_vtcm, slice, 32, (oh_len & 1) ? 2 : 0);                      \
                                                                                                      \
    const HVX_Vector w00  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w01  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w02  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector w10  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w11  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w12  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector w20  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w21  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w22  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector zero = Q6_V_vzero();                                                             \
                                                                                                      \
    const HVX_VectorPred q_not_lane16 = Q6_Q_or_QQn(Q6_Q_vsetq_R(64), Q6_Q_vsetq2_R(68));             \
    const float * restrict in_row     = in_vtcm;                                                      \
    HVX_Vector v_prev_b2              = hvx_vmem(in_row);                                             \
    in_row += 32;                                                                                     \
                                                                                                      \
    for (uint32_t ohi = 0; ohi < oh_len; ohi += 2) {                                                  \
        const HVX_Vector v_a0 = v_prev_b2;                                                            \
        const HVX_Vector v_a1 = hvx_vmem(in_row);                                                     \
        const HVX_Vector v_a2 = hvx_vmem(in_row + 32);                                                \
        const HVX_Vector v_b0 = v_a2;                                                                 \
        const HVX_Vector v_b1 = hvx_vmem(in_row + 64);                                                \
        const HVX_Vector v_b2 = hvx_vmem(in_row + 96);                                                \
        in_row               += 128;                                                                  \
        v_prev_b2             = v_b2;                                                                 \
                                                                                                      \
        const HVX_VectorPair vp0 = Q6_W_vdeal_VVR(v_b0, v_a0, -4);                                    \
        const HVX_Vector ev0     = Q6_V_lo_W(vp0);                                                    \
        const HVX_Vector od0     = Q6_V_hi_W(vp0);                                                    \
        const HVX_Vector lf0     = Q6_V_vand_QV(q_not_lane16, Q6_V_valign_VVR(od0, zero, 124));       \
        HVX_Vector acc           = conv2d_dw_qf32_mpy_add3(zero, lf0, w00, ev0, w01, od0, w02);       \
                                                                                                      \
        const HVX_VectorPair vp1 = Q6_W_vdeal_VVR(v_b1, v_a1, -4);                                    \
        const HVX_Vector ev1     = Q6_V_lo_W(vp1);                                                    \
        const HVX_Vector od1     = Q6_V_hi_W(vp1);                                                    \
        const HVX_Vector lf1     = Q6_V_vand_QV(q_not_lane16, Q6_V_valign_VVR(od1, zero, 124));       \
        acc                      = conv2d_dw_qf32_mpy_add3(acc, lf1, w10, ev1, w11, od1, w12);        \
                                                                                                      \
        const HVX_VectorPair vp2 = Q6_W_vdeal_VVR(v_b2, v_a2, -4);                                    \
        const HVX_Vector ev2     = Q6_V_lo_W(vp2);                                                    \
        const HVX_Vector od2     = Q6_V_hi_W(vp2);                                                    \
        const HVX_Vector lf2     = Q6_V_vand_QV(q_not_lane16, Q6_V_valign_VVR(od2, zero, 124));       \
        acc                      = conv2d_dw_qf32_mpy_add3(acc, lf2, w20, ev2, w21, od2, w22);        \
                                                                                                      \
        hvx_vmem(out_vtcm + (size_t) ohi * 16) = Q6_Vsf_equals_Vqf32(acc);                            \
    }                                                                                                 \
}

DEFINE_CONV2D_DW_WHCN_S2_K3X3_W32(f32, conv2d_dw_splat_w_f32)
DEFINE_CONV2D_DW_WHCN_S2_K3X3_W32(f16, conv2d_dw_splat_w_f16)

#define DEFINE_CONV2D_DW_WHCN_S2_K3X3_W64(SUFFIX, SPLAT_W_FN)                                         \
__attribute__((noinline))                                                                             \
static void conv2d_dw_whcn_s2_k3x3_w64_##SUFFIX(                                                      \
    const float * restrict in_vtcm,                                                                   \
    float * restrict       out_vtcm,                                                                  \
    const void *           w_vtcm,                                                                    \
    const struct conv2d_dw_task_slice * slice,                                                        \
    const struct htp_ops_context * octx,                                                              \
    const struct htp_conv2d_dw_kernel_params * kp) {                                                  \
                                                                                                      \
    (void) kp;                                                                                        \
    const struct htp_tensor * src0 = octx->src[0];                                                    \
    const size_t w_nb0             = src0->nb[0];                                                     \
    const size_t w_nb1             = src0->nb[1];                                                     \
    const char * restrict w_chan   = (const char *) w_vtcm + (size_t) slice->c * src0->nb[3];         \
    const uint32_t oh_len          = slice->oh_len;                                                   \
                                                                                                      \
    conv2d_dw_pad_vtcm_k3x3((float *) in_vtcm, slice, 64, 0);                                         \
                                                                                                      \
    const HVX_Vector w00  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w01  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w02  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector w10  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w11  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w12  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector w20  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w21  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w22  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector zero = Q6_V_vzero();                                                             \
                                                                                                      \
    const float * restrict in_row = in_vtcm;                                                          \
    const HVX_VectorPair vp0_init = Q6_W_vdeal_VVR(hvx_vmem(in_row + 32), hvx_vmem(in_row), -4);      \
    in_row += 64;                                                                                     \
    HVX_Vector ev0 = Q6_V_lo_W(vp0_init);                                                             \
    HVX_Vector od0 = Q6_V_hi_W(vp0_init);                                                             \
    HVX_Vector lf0 = Q6_V_valign_VVR(od0, zero, 124);                                                 \
                                                                                                      \
    for (uint32_t ohi = 0; ohi < oh_len; ++ohi) {                                                     \
        const float * restrict r1_ptr = in_row;                                                       \
        const float * restrict r2_ptr = in_row + 64;                                                  \
        in_row += 128;                                                                                \
                                                                                                      \
        HVX_Vector acc = conv2d_dw_qf32_mpy_add3(zero, lf0, w00, ev0, w01, od0, w02);                 \
                                                                                                      \
        const HVX_VectorPair vp1 = Q6_W_vdeal_VVR(hvx_vmem(r1_ptr + 32), hvx_vmem(r1_ptr), -4);       \
        const HVX_Vector ev1     = Q6_V_lo_W(vp1);                                                    \
        const HVX_Vector od1     = Q6_V_hi_W(vp1);                                                    \
        const HVX_Vector lf1     = Q6_V_valign_VVR(od1, zero, 124);                                   \
        acc                      = conv2d_dw_qf32_mpy_add3(acc, lf1, w10, ev1, w11, od1, w12);        \
                                                                                                      \
        const HVX_VectorPair vp2 = Q6_W_vdeal_VVR(hvx_vmem(r2_ptr + 32), hvx_vmem(r2_ptr), -4);       \
        const HVX_Vector ev2     = Q6_V_lo_W(vp2);                                                    \
        const HVX_Vector od2     = Q6_V_hi_W(vp2);                                                    \
        const HVX_Vector lf2     = Q6_V_valign_VVR(od2, zero, 124);                                   \
        acc                      = conv2d_dw_qf32_mpy_add3(acc, lf2, w20, ev2, w21, od2, w22);        \
                                                                                                      \
        hvx_vmem(out_vtcm + (size_t) ohi * 32) = Q6_Vsf_equals_Vqf32(acc);                            \
                                                                                                      \
        ev0 = ev2;                                                                                    \
        od0 = od2;                                                                                    \
        lf0 = lf2;                                                                                    \
    }                                                                                                 \
}

DEFINE_CONV2D_DW_WHCN_S2_K3X3_W64(f32, conv2d_dw_splat_w_f32)
DEFINE_CONV2D_DW_WHCN_S2_K3X3_W64(f16, conv2d_dw_splat_w_f16)

#define DEFINE_CONV2D_DW_WHCN_S2_K3X3(SUFFIX, SPLAT_W_FN)                                             \
__attribute__((noinline))                                                                             \
static void conv2d_dw_whcn_s2_k3x3_##SUFFIX(                                                          \
    const float * restrict in_vtcm,                                                                   \
    float * restrict       out_vtcm,                                                                  \
    const void *           w_vtcm,                                                                    \
    const struct conv2d_dw_task_slice * slice,                                                        \
    const struct htp_ops_context * octx,                                                              \
    const struct htp_conv2d_dw_kernel_params * kp) {                                                  \
                                                                                                      \
    const struct htp_tensor * src0 = octx->src[0];                                                    \
    const size_t w_nb0             = src0->nb[0];                                                     \
    const size_t w_nb1             = src0->nb[1];                                                     \
    const char * restrict w_chan   = (const char *) w_vtcm + (size_t) slice->c * src0->nb[3];         \
    const uint32_t oh_len          = slice->oh_len;                                                   \
    const uint32_t w_in            = kp->w;                                                           \
    const uint32_t ow              = kp->ow;                                                          \
                                                                                                      \
    conv2d_dw_pad_vtcm_k3x3((float *) in_vtcm, slice, w_in, 0);                                       \
                                                                                                      \
    const HVX_Vector w00  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w01  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w02  = SPLAT_W_FN(w_chan + 0 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector w10  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w11  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w12  = SPLAT_W_FN(w_chan + 1 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector w20  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 0 * w_nb0);                               \
    const HVX_Vector w21  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 1 * w_nb0);                               \
    const HVX_Vector w22  = SPLAT_W_FN(w_chan + 2 * w_nb1 + 2 * w_nb0);                               \
    const HVX_Vector zero = Q6_V_vzero();                                                             \
                                                                                                      \
    for (uint32_t ohi = 0; ohi < oh_len; ++ohi) {                                                     \
        const float * restrict r0 = in_vtcm + (size_t) (2 * ohi) * w_in;                              \
        const float * restrict r1 = r0 + w_in;                                                        \
        const float * restrict r2 = r1 + w_in;                                                        \
        float * restrict out_row  = out_vtcm + (size_t) ohi * ow;                                     \
                                                                                                      \
        HVX_Vector prev_od0 = zero;                                                                   \
        HVX_Vector prev_od1 = zero;                                                                   \
        HVX_Vector prev_od2 = zero;                                                                   \
                                                                                                      \
        for (uint32_t ox = 0; ox < ow; ox += 32) {                                                    \
            const uint32_t ix   = 2 * ox;                                                             \
            const bool     has1 = (ix + 32 < w_in);                                                   \
                                                                                                      \
            const HVX_VectorPair vp0 = Q6_W_vdeal_VVR(has1 ? hvx_vmem(r0 + ix + 32) : zero, hvx_vmem(r0 + ix), -4); \
            const HVX_Vector ev0     = Q6_V_lo_W(vp0);                                                \
            const HVX_Vector od0     = Q6_V_hi_W(vp0);                                                \
            const HVX_Vector lf0     = Q6_V_valign_VVR(od0, prev_od0, 124);                           \
            prev_od0                 = od0;                                                           \
            HVX_Vector acc           = conv2d_dw_qf32_mpy_add3(zero, lf0, w00, ev0, w01, od0, w02);   \
                                                                                                      \
            const HVX_VectorPair vp1 = Q6_W_vdeal_VVR(has1 ? hvx_vmem(r1 + ix + 32) : zero, hvx_vmem(r1 + ix), -4); \
            const HVX_Vector ev1     = Q6_V_lo_W(vp1);                                                \
            const HVX_Vector od1     = Q6_V_hi_W(vp1);                                                \
            const HVX_Vector lf1     = Q6_V_valign_VVR(od1, prev_od1, 124);                           \
            prev_od1                 = od1;                                                           \
            acc                      = conv2d_dw_qf32_mpy_add3(acc, lf1, w10, ev1, w11, od1, w12);    \
                                                                                                      \
            const HVX_VectorPair vp2 = Q6_W_vdeal_VVR(has1 ? hvx_vmem(r2 + ix + 32) : zero, hvx_vmem(r2 + ix), -4); \
            const HVX_Vector ev2     = Q6_V_lo_W(vp2);                                                \
            const HVX_Vector od2     = Q6_V_hi_W(vp2);                                                \
            const HVX_Vector lf2     = Q6_V_valign_VVR(od2, prev_od2, 124);                           \
            prev_od2                 = od2;                                                           \
            acc                      = conv2d_dw_qf32_mpy_add3(acc, lf2, w20, ev2, w21, od2, w22);    \
                                                                                                      \
            if (ox + 32 <= ow) {                                                                      \
                hvx_vmem(out_row + ox) = Q6_Vsf_equals_Vqf32(acc);                                    \
            } else {                                                                                  \
                const uint32_t rem = ow - ox;                                                         \
                const HVX_VectorPred q = Q6_Q_vsetq_R(rem * sizeof(float));                           \
                hvx_vec_store_u(out_row + ox, rem * sizeof(float), Q6_V_vmux_QVV(q, Q6_Vsf_equals_Vqf32(acc), zero)); \
            }                                                                                         \
        }                                                                                             \
    }                                                                                                 \
}

DEFINE_CONV2D_DW_WHCN_S2_K3X3(f32, conv2d_dw_splat_w_f32)
DEFINE_CONV2D_DW_WHCN_S2_K3X3(f16, conv2d_dw_splat_w_f16)

#define DEFINE_CONV2D_DW_WHCN_S1(SUFFIX, SPLAT_W_FN)                                                  \
__attribute__((noinline))                                                                             \
static void conv2d_dw_whcn_s1_##SUFFIX(                                                               \
    const float * in_vtcm,                                                                            \
    float *       out_vtcm,                                                                           \
    const void *  w_vtcm,                                                                             \
    const struct conv2d_dw_task_slice * slice,                                                        \
    const struct htp_ops_context * octx,                                                              \
    const struct htp_conv2d_dw_kernel_params * kp) {                                                  \
                                                                                                      \
    const struct htp_tensor * src0 = octx->src[0];                                                    \
    const char *   w_chan          = (const char *) w_vtcm + (size_t) slice->c * src0->nb[3];         \
    const int32_t  p0              = kp->p0;                                                          \
    const int32_t  p1              = kp->p1;                                                          \
    const int32_t  s1              = kp->s1;                                                          \
    const int32_t  d1              = kp->d1;                                                          \
    const int32_t  w               = (int32_t) kp->w;                                                 \
    const HVX_Vector zero          = Q6_V_vzero();                                                    \
                                                                                                      \
    for (uint32_t ohi = 0; ohi < slice->oh_len; ++ohi) {                                              \
        const uint32_t oh = slice->oh_start + ohi;                                                    \
        float * out_row   = out_vtcm + (size_t) ohi * kp->ow;                                         \
                                                                                                      \
        for (uint32_t ox = 0; ox < kp->ow; ox += 32) {                                                \
            const uint32_t rem    = kp->ow - ox;                                                      \
            const uint32_t nbytes = (rem < 32) ? (rem * sizeof(float)) : 128;                         \
            HVX_Vector acc0       = zero;                                                             \
            HVX_Vector acc1       = zero;                                                             \
            HVX_Vector acc2       = zero;                                                             \
            HVX_Vector acc3       = zero;                                                             \
                                                                                                      \
            for (uint32_t kh = 0; kh < kp->kh; ++kh) {                                                \
                const int32_t ih = (int32_t) (oh * s1 + kh * d1) - p1;                                \
                if (ih < 0 || ih >= (int32_t) kp->h) {                                                \
                    continue;                                                                         \
                }                                                                                     \
                const float * in_row = in_vtcm + (size_t) (ih - slice->ih_first) * kp->w;             \
                const char *  w_row  = w_chan + (size_t) kh * src0->nb[1];                            \
                                                                                                      \
                for (uint32_t kw = 0; kw < kp->kw; ++kw) {                                            \
                    const HVX_Vector w_v     = SPLAT_W_FN(w_row + (size_t) kw * src0->nb[0]);         \
                    const int32_t    base_iw = (int32_t) ox - p0 + (int32_t) kw;                      \
                    HVX_Vector       in_v;                                                            \
                                                                                                      \
                    if (base_iw >= 0) {                                                               \
                        in_v = hvx_vmemu(in_row + base_iw);                                           \
                    } else {                                                                          \
                        const int32_t pad  = -base_iw;                                                \
                        const HVX_Vector v = hvx_vmemu(in_row);                                       \
                        in_v = Q6_V_valign_VVR(v, zero, 128 - (uint32_t) pad * sizeof(float));        \
                    }                                                                                 \
                    if (base_iw + 31 >= w) {                                                          \
                        const int32_t valid    = w - base_iw;                                         \
                        const HVX_VectorPred q = Q6_Q_vsetq_R(MAX(0, valid) * sizeof(float));         \
                        in_v = Q6_V_vmux_QVV(q, in_v, zero);                                          \
                    }                                                                                 \
                                                                                                      \
                    const HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(in_v, w_v);                          \
                    switch (kw & 3) {                                                                 \
                        case 0: acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, prod); break;                   \
                        case 1: acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, prod); break;                   \
                        case 2: acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, prod); break;                   \
                        case 3: acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, prod); break;                   \
                    }                                                                                 \
                }                                                                                     \
            }                                                                                         \
                                                                                                      \
            const HVX_Vector acc01   = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, acc1);                          \
            const HVX_Vector acc23   = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, acc3);                          \
            HVX_Vector       acc_out = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(acc01, acc23));   \
            if (rem < 32) {                                                                           \
                const HVX_VectorPred q = Q6_Q_vsetq_R(rem * sizeof(float));                           \
                acc_out = Q6_V_vmux_QVV(q, acc_out, zero);                                            \
            }                                                                                         \
                                                                                                      \
            hvx_vec_store_u(out_row + ox, nbytes, acc_out);                                           \
        }                                                                                             \
    }                                                                                                 \
}

DEFINE_CONV2D_DW_WHCN_S1(f32, conv2d_dw_splat_w_f32)
DEFINE_CONV2D_DW_WHCN_S1(f16, conv2d_dw_splat_w_f16)

#define DEFINE_CONV2D_DW_WHCN_S2(SUFFIX, SPLAT_W_FN)                                                  \
__attribute__((noinline))                                                                             \
static void conv2d_dw_whcn_s2_##SUFFIX(                                                               \
    const float * in_vtcm,                                                                            \
    float *       out_vtcm,                                                                           \
    const void *  w_vtcm,                                                                             \
    const struct conv2d_dw_task_slice * slice,                                                        \
    const struct htp_ops_context * octx,                                                              \
    const struct htp_conv2d_dw_kernel_params * kp) {                                                  \
                                                                                                      \
    const struct htp_tensor * src0 = octx->src[0];                                                    \
    const char *   w_chan          = (const char *) w_vtcm + (size_t) slice->c * src0->nb[3];         \
    const int32_t  p0              = kp->p0;                                                          \
    const int32_t  p1              = kp->p1;                                                          \
    const int32_t  s1              = kp->s1;                                                          \
    const int32_t  d1              = kp->d1;                                                          \
    const int32_t  w               = (int32_t) kp->w;                                                 \
    const HVX_Vector zero          = Q6_V_vzero();                                                    \
                                                                                                      \
    for (uint32_t ohi = 0; ohi < slice->oh_len; ++ohi) {                                              \
        const uint32_t oh = slice->oh_start + ohi;                                                    \
        float * out_row   = out_vtcm + (size_t) ohi * kp->ow;                                         \
                                                                                                      \
        for (uint32_t ox = 0; ox < kp->ow; ox += 32) {                                                \
            const uint32_t rem    = kp->ow - ox;                                                      \
            const uint32_t nbytes = (rem < 32) ? (rem * sizeof(float)) : 128;                         \
            HVX_Vector acc0       = zero;                                                             \
            HVX_Vector acc1       = zero;                                                             \
            HVX_Vector acc2       = zero;                                                             \
            HVX_Vector acc3       = zero;                                                             \
                                                                                                      \
            for (uint32_t kh = 0; kh < kp->kh; ++kh) {                                                \
                const int32_t ih = (int32_t) (oh * s1 + kh * d1) - p1;                                \
                if (ih < 0 || ih >= (int32_t) kp->h) {                                                \
                    continue;                                                                         \
                }                                                                                     \
                const float * in_row = in_vtcm + (size_t) (ih - slice->ih_first) * kp->w;             \
                const char *  w_row  = w_chan + (size_t) kh * src0->nb[1];                            \
                                                                                                      \
                for (uint32_t kw = 0; kw < kp->kw; ++kw) {                                            \
                    const HVX_Vector w_v     = SPLAT_W_FN(w_row + (size_t) kw * src0->nb[0]);         \
                    const int32_t    base_iw = (int32_t) (2 * ox) - p0 + (int32_t) kw;                \
                    HVX_Vector       in_v;                                                            \
                                                                                                      \
                    if (base_iw >= 0) {                                                               \
                        const HVX_Vector v0 = hvx_vmemu(in_row + base_iw);                            \
                        const HVX_Vector v1 = hvx_vmemu(in_row + base_iw + 32);                       \
                        in_v = Q6_V_lo_W(Q6_W_vdeal_VVR(v1, v0, -4));                                 \
                    } else {                                                                          \
                        const int32_t pad       = -base_iw;                                           \
                        const HVX_Vector v0     = hvx_vmemu(in_row);                                  \
                        const HVX_Vector v1     = hvx_vmemu(in_row + 32);                             \
                        const HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, -4);                         \
                        if (pad & 1) {                                                                \
                            const uint32_t shift_lanes = (uint32_t) (pad + 1) / 2;                    \
                            const HVX_Vector v_odd     = Q6_V_hi_W(vp);                               \
                            in_v = Q6_V_valign_VVR(v_odd, zero, 128 - shift_lanes * sizeof(float));   \
                        } else {                                                                      \
                            const uint32_t shift_lanes = (uint32_t) pad / 2;                          \
                            const HVX_Vector v_even    = Q6_V_lo_W(vp);                               \
                            in_v = Q6_V_valign_VVR(v_even, zero, 128 - shift_lanes * sizeof(float));  \
                        }                                                                             \
                    }                                                                                 \
                    if (base_iw + 62 >= w) {                                                          \
                        const int32_t valid    = (w - base_iw + 1) / 2;                               \
                        const HVX_VectorPred q = Q6_Q_vsetq_R(MAX(0, valid) * sizeof(float));         \
                        in_v = Q6_V_vmux_QVV(q, in_v, zero);                                          \
                    }                                                                                 \
                                                                                                      \
                    const HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(in_v, w_v);                          \
                    switch (kw & 3) {                                                                 \
                        case 0: acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, prod); break;                   \
                        case 1: acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, prod); break;                   \
                        case 2: acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, prod); break;                   \
                        case 3: acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, prod); break;                   \
                    }                                                                                 \
                }                                                                                     \
            }                                                                                         \
                                                                                                      \
            const HVX_Vector acc01   = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, acc1);                          \
            const HVX_Vector acc23   = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, acc3);                          \
            HVX_Vector       acc_out = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(acc01, acc23));   \
            if (rem < 32) {                                                                           \
                const HVX_VectorPred q = Q6_Q_vsetq_R(rem * sizeof(float));                           \
                acc_out = Q6_V_vmux_QVV(q, acc_out, zero);                                            \
            }                                                                                         \
                                                                                                      \
            hvx_vec_store_u(out_row + ox, nbytes, acc_out);                                           \
        }                                                                                             \
    }                                                                                                 \
}

DEFINE_CONV2D_DW_WHCN_S2(f32, conv2d_dw_splat_w_f32)
DEFINE_CONV2D_DW_WHCN_S2(f16, conv2d_dw_splat_w_f16)

#define DEFINE_CONV2D_DW_CWHN(SUFFIX, LOAD_W_FN)                                                      \
__attribute__((noinline))                                                                             \
static void conv2d_dw_cwhn_##SUFFIX(                                                                  \
    const float * in_vtcm,                                                                            \
    float *       out_vtcm,                                                                           \
    const void *  w_vtcm,                                                                             \
    const struct conv2d_dw_task_slice * slice,                                                        \
    const struct htp_ops_context * octx,                                                              \
    const struct htp_conv2d_dw_kernel_params * kp) {                                                  \
                                                                                                      \
    const struct htp_tensor * src0 = octx->src[0];                                                    \
    const uint32_t in_row_stride   = kp->w * kp->c;                                                   \
    const uint32_t out_row_stride  = kp->ow * kp->c;                                                  \
    const HVX_Vector zero          = Q6_V_vzero();                                                    \
                                                                                                      \
    for (uint32_t cb = 0; cb < kp->c; cb += 32) {                                                     \
        const uint32_t rem    = MIN(32, kp->c - cb);                                                  \
        const char *   w_base = (const char *) w_vtcm + (size_t) cb * src0->nb[3];                    \
                                                                                                      \
        for (uint32_t ohi = 0; ohi < slice->oh_len; ++ohi) {                                          \
            const uint32_t oh = slice->oh_start + ohi;                                                \
            float * out_row   = out_vtcm + (size_t) ohi * out_row_stride + cb;                        \
                                                                                                      \
            for (uint32_t ow = 0; ow < kp->ow; ++ow) {                                                \
                HVX_Vector acc0 = zero;                                                               \
                HVX_Vector acc1 = zero;                                                               \
                HVX_Vector acc2 = zero;                                                               \
                HVX_Vector acc3 = zero;                                                               \
                                                                                                      \
                for (uint32_t kh = 0; kh < kp->kh; ++kh) {                                            \
                    const int32_t ih = (int32_t)(oh * kp->s1 + kh * kp->d1) - kp->p1;                 \
                    if (ih < 0 || ih >= (int32_t)kp->h) {                                             \
                        continue;                                                                     \
                    }                                                                                 \
                    const float * in_row = in_vtcm + (size_t)(ih - slice->ih_first) * in_row_stride + cb; \
                    const char *  w_row  = w_base + (size_t) kh * src0->nb[1];                        \
                                                                                                      \
                    for (uint32_t kw = 0; kw < kp->kw; ++kw) {                                        \
                        const int32_t iw = (int32_t)(ow * kp->s0 + kw * kp->d0) - kp->p0;             \
                        if (iw < 0 || iw >= (int32_t)kp->w) {                                         \
                            continue;                                                                 \
                        }                                                                             \
                                                                                                      \
                        const float *    in_ptr = in_row + (size_t) iw * kp->c;                       \
                        const char *     w_ptr  = w_row + (size_t) kw * src0->nb[0];                  \
                        const HVX_Vector v_in   = hvx_vmemu(in_ptr);                                  \
                        const HVX_Vector v_w    = LOAD_W_FN(w_ptr);                                   \
                        const HVX_Vector prod   = Q6_Vqf32_vmpy_VsfVsf(v_in, v_w);                    \
                        switch (kw & 3) {                                                             \
                            case 0: acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, prod); break;               \
                            case 1: acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, prod); break;               \
                            case 2: acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, prod); break;               \
                            case 3: acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, prod); break;               \
                        }                                                                             \
                    }                                                                                 \
                }                                                                                     \
                                                                                                      \
                const HVX_Vector acc01   = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, acc1);                      \
                const HVX_Vector acc23   = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, acc3);                      \
                const HVX_Vector acc_out = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(acc01, acc23)); \
                                                                                                      \
                float * dst_ptr = out_row + (size_t) ow * kp->c;                                      \
                if (rem == 32) {                                                                      \
                    hvx_vmemu(dst_ptr) = acc_out;                                                     \
                } else {                                                                              \
                    hvx_vec_store_u(dst_ptr, rem * sizeof(float), acc_out);                           \
                }                                                                                     \
            }                                                                                         \
        }                                                                                             \
    }                                                                                                 \
}

DEFINE_CONV2D_DW_CWHN(f32, conv2d_dw_load_w_f32)
DEFINE_CONV2D_DW_CWHN(f16, conv2d_dw_load_w_f16)

typedef void (*conv2d_dw_compute_fn_t)(
    const float * in_vtcm,
    float *       out_vtcm,
    const void *  w_vtcm,
    const struct conv2d_dw_task_slice * slice,
    const struct htp_ops_context * octx,
    const struct htp_conv2d_dw_kernel_params * kp);

static const conv2d_dw_compute_fn_t conv2d_dw_kernels[HTP_CONV2D_DW_KERNEL_COUNT][2] = {
    [HTP_CONV2D_DW_KERNEL_UNSUPPORTED]      = { NULL,                             NULL                             },
    [HTP_CONV2D_DW_KERNEL_CWHN]             = { conv2d_dw_cwhn_f32,               conv2d_dw_cwhn_f16               },
    [HTP_CONV2D_DW_KERNEL_WHCN_S1]          = { conv2d_dw_whcn_s1_f32,            conv2d_dw_whcn_s1_f16            },
    [HTP_CONV2D_DW_KERNEL_WHCN_S2]          = { conv2d_dw_whcn_s2_f32,            conv2d_dw_whcn_s2_f16            },
    [HTP_CONV2D_DW_KERNEL_WHCN_S1_K3X3]     = { conv2d_dw_whcn_s1_k3x3_f32,       conv2d_dw_whcn_s1_k3x3_f16       },
    [HTP_CONV2D_DW_KERNEL_WHCN_S2_K3X3_W32] = { conv2d_dw_whcn_s2_k3x3_w32_f32,   conv2d_dw_whcn_s2_k3x3_w32_f16   },
    [HTP_CONV2D_DW_KERNEL_WHCN_S2_K3X3_W64] = { conv2d_dw_whcn_s2_k3x3_w64_f32,   conv2d_dw_whcn_s2_k3x3_w64_f16   },
    [HTP_CONV2D_DW_KERNEL_WHCN_S2_K3X3]     = { conv2d_dw_whcn_s2_k3x3_f32,       conv2d_dw_whcn_s2_k3x3_f16       },
};

// ---- 5x5 s1 p2 d1 f16-weight WHCN kernel ----

// Kernel parameters derived on entry from htp_conv2d_dw_kernel_params (solved on the host).
struct htp_conv_2d_dw5_kernel_params {
    uint32_t n_threads;
    uint32_t planes;
    uint32_t src_row_stride;
    uint32_t src_slot_bytes;
    uint32_t weight_slot_bytes;
    uint32_t dst_slot_bytes;
    uint32_t off_src;
    uint32_t off_weight;
    uint32_t off_dst;
};

struct conv_2d_dw5_hvx_context {
    struct htp_ops_context *                     octx;
    const struct htp_conv_2d_dw5_kernel_params * kparams;
    uint32_t                                     plane_start;
    uint32_t                                     plane_count;
    uint32_t                                     planes_per_thread;
};

// Zero both source slots once per thread; DMA only writes the interior, so the 2-pixel halo stays zero.
static __attribute__((noinline)) void conv_2d_dw5_clear(uint8_t * restrict data, uint32_t bytes) {
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t offset = 0; offset < bytes; offset += VLEN) {
        *(HVX_Vector *) (data + offset) = zero;
    }
}

// Expand the 25 f16 taps in weight slot b into the f32 splat table that follows the two slots.
static __attribute__((noinline)) void conv_2d_dw5_prep_weights(uint8_t * restrict wbase, uint32_t b) {
    HVX_Vector *     wsplat = (HVX_Vector *) (wbase + 2 * VLEN);
    const HVX_Vector w      = Q6_V_lo_W(hvx_vec_f16_to_f32(*(const HVX_Vector *) (wbase + b * VLEN)));
    for (uint32_t t = 0; t < 25; ++t) {
        wsplat[t] = hvx_vec_repl4(Q6_V_vror_VR(w, t * sizeof(float)));
    }
}

// One output plane. src rows are 128B aligned with the 2-pixel left halo at the row start, so the five kx taps
// of a 32-wide output vector are valigns of two aligned loads.
static __attribute__((noinline)) void conv_2d_dw5_plane(const uint8_t * restrict    src,
                                                        const HVX_Vector * restrict wsplat,
                                                        float * restrict            dst,
                                                        uint32_t                    src_row_stride,
                                                        uint32_t                    ow,
                                                        uint32_t                    oh) {
    for (uint32_t oy = 0; oy < oh; ++oy) {
        for (uint32_t ox = 0; ox < ow; ox += 2 * VLEN_FP32) {
            HVX_Vector acc0 = Q6_V_vzero();
            HVX_Vector acc1 = Q6_V_vzero();

            for (uint32_t ky = 0; ky < 5; ++ky) {
                const HVX_Vector * row = (const HVX_Vector *) (src + (oy + ky) * src_row_stride + ox * sizeof(float));
                const HVX_Vector * w   = wsplat + ky * 5;
                const HVX_Vector   v0  = row[0];
                const HVX_Vector   v1  = row[1];
                const HVX_Vector   v2  = row[2];

                const HVX_Vector a01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v0, w[0]),
                                                                Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v1, v0, 4), w[1]));
                const HVX_Vector a23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v1, v0, 8), w[2]),
                                                                Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v1, v0, 12), w[3]));
                const HVX_Vector a4  = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v1, v0, 16), w[4]));
                acc0                 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a01, a23), a4);

                const HVX_Vector b01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, w[0]),
                                                                Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v2, v1, 4), w[1]));
                const HVX_Vector b23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v2, v1, 8), w[2]),
                                                                Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v2, v1, 12), w[3]));
                const HVX_Vector b4  = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(Q6_V_valign_VVR(v2, v1, 16), w[4]));
                acc1                 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(b01, b23), b4);
            }

            const uint32_t rem = ow - ox;
            float *        out = dst + oy * ow + ox;
            hvx_vec_store_u(out, MIN(rem, VLEN_FP32) * sizeof(float), Q6_Vsf_equals_Vqf32(acc0));
            if (rem > VLEN_FP32) {
                hvx_vec_store_u(out + VLEN_FP32, MIN(rem - VLEN_FP32, VLEN_FP32) * sizeof(float),
                                Q6_Vsf_equals_Vqf32(acc1));
            }
        }
    }
}

// Planes are processed in memory order (plane = n * C + c). Each plane is three DMAs: weights in, src in, dst out.
// Two buffers per stream: plane i+1 is in flight while plane i is computed.
static void conv_2d_dw5_hvx_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct conv_2d_dw5_hvx_context *       ctx    = (const struct conv_2d_dw5_hvx_context *) data;
    const struct htp_ops_context *               octx   = ctx->octx;
    const struct htp_conv_2d_dw5_kernel_params * p      = ctx->kparams;
    const struct htp_tensor *                    weight = octx->src[0];
    const struct htp_tensor *                    input  = octx->src[1];
    const struct htp_tensor *                    output = octx->dst;

    const uint32_t first = ctx->plane_start + ith * ctx->planes_per_thread;
    const uint32_t last  = MIN(first + ctx->planes_per_thread, ctx->plane_start + ctx->plane_count);
    if (first >= last) {
        return;
    }
    const uint32_t n = last - first;

    uint8_t *    vtcm        = octx->ctx->vtcm_base;
    uint8_t *    src_base    = vtcm + p->off_src + ith * 2 * p->src_slot_bytes;
    uint8_t *    weight_base = vtcm + p->off_weight + ith * p->weight_slot_bytes;
    uint8_t *    dst_base    = vtcm + p->off_dst + ith * 2 * p->dst_slot_bytes;
    uint8_t *    srcb[2]     = { src_base, src_base + p->src_slot_bytes };
    uint8_t *    wb[2]       = { weight_base, weight_base + VLEN };
    HVX_Vector * wsplat      = (HVX_Vector *) (weight_base + 2 * VLEN);
    uint8_t *    dstb[2]     = { dst_base, dst_base + p->dst_slot_bytes };

    dma_queue *               dma_q = th->dma_q;
    struct htp_thread_trace * tr    = th->trace;

    const uint32_t channels     = input->ne[2];
    const uint32_t ih           = input->ne[1];
    const uint32_t ow           = output->ne[0];
    const uint32_t oh           = output->ne[1];
    const uint32_t stride       = p->src_row_stride;
    const uint32_t in_row       = input->nb[1];
    const uint32_t in_plane     = input->nb[2];
    const uint32_t out_row      = output->nb[1];
    const uint32_t out_plane    = output->nb[2];
    const uint32_t w_plane      = weight->nb[3];
    const uint32_t weight_bytes = 25 * sizeof(__fp16);
    const uint32_t halo_offset  = 2 * stride + 2 * sizeof(float);

    conv_2d_dw5_clear(src_base, 2 * p->src_slot_bytes);

    // channel of the next plane to fetch
    uint32_t c_next = first % channels;

    // prologue: dummy dst completions keep the pop pattern uniform for the first two planes
    for (uint32_t i = 0; i < n && i < 2; ++i) {
        const uint32_t plane = first + i;
        dma_queue_push(dma_q, dma_make_data(output->data, dstb[i]), out_row, out_row, out_row, 0);
        dma_queue_push(dma_q, dma_make_data(wb[i], weight->data + (dma_addr_t) c_next * w_plane), VLEN, w_plane, weight_bytes, 1);
        dma_queue_push(dma_q, dma_make_data(srcb[i] + halo_offset, input->data + (dma_addr_t) plane * in_plane), stride, in_row,
                       in_row, ih);
        c_next = (c_next + 1 == channels) ? 0 : c_next + 1;
    }

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t plane = first + i;
        const uint32_t b     = i & 1;

        dma_queue_pop(dma_q);  // dst slot b free
        dma_queue_pop(dma_q);  // weights for plane
        dma_queue_pop(dma_q);  // src for plane

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        conv_2d_dw5_prep_weights(weight_base, b);
        conv_2d_dw5_plane(srcb[b], wsplat, (float *) dstb[b], stride, ow, oh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_queue_push(dma_q, dma_make_data(output->data + (dma_addr_t) plane * out_plane, dstb[b]), out_row, out_row, out_row, oh);

        if (i + 2 < n) {
            dma_queue_push(dma_q, dma_make_data(wb[b], weight->data + (dma_addr_t) c_next * w_plane), VLEN, w_plane, weight_bytes,
                           1);
            dma_queue_push(dma_q, dma_make_data(srcb[b] + halo_offset, input->data + (dma_addr_t) (plane + 2) * in_plane), stride,
                           in_row, in_row, ih);
            c_next = (c_next + 1 == channels) ? 0 : c_next + 1;
        }
    }

    // epilogue: drain the last outputs
    for (uint32_t i = 0; i < n && i < 2; ++i) {
        dma_queue_pop(dma_q);
    }

    FARF(HIGH, "conv2d-dw-hvx-dma-5x5 %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n", ith, nth, input->ne[0],
         input->ne[1], input->ne[2], input->ne[3], output->ne[0], output->ne[1], output->ne[2], output->ne[3], first,
         last);
}

static int conv_2d_dw5_run_hvx(struct htp_ops_context * octx, const struct htp_conv_2d_dw5_kernel_params * p) {
    const struct htp_tensor * output      = octx->dst;
    uint32_t                  plane_start = 0;
    uint32_t                  plane_count = p->planes;

    if (octx->ctx->mdev.count > 1) {
        // output planes are contiguous, split them on 128B line boundaries
        const uint32_t planes_per_chunk =
            HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(output->nb[2], HTP_TENSOR_MDEV_LINE_SIZE);
        const bool                         can_split = htp_tensor_mdev_data_aligned(output);
        const struct htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(plane_count, can_split ? planes_per_chunk : 0, octx->ctx->mdev.idx,
                                      octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        plane_start = range.start;
        plane_count = range.count;
    }
    if (plane_count == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t                 n_threads         = MIN(p->n_threads, plane_count);
    const uint32_t                 planes_per_thread = (plane_count + n_threads - 1) / n_threads;
    struct conv_2d_dw5_hvx_context ctx               = {
        .octx              = octx,
        .kparams           = p,
        .plane_start       = plane_start,
        .plane_count       = plane_count,
        .planes_per_thread = planes_per_thread,
    };

    work_queue_run(octx->ctx->hvx_queue, conv_2d_dw5_hvx_thread, &ctx, n_threads);
    return HTP_STATUS_OK;
}

static int op_conv_2d_dw5(struct htp_ops_context * octx) {
    // Tensor shapes, types and VTCM sizing were validated by the host (htp_kparams_solve_conv2d_dw).
    const struct htp_conv2d_dw_kernel_params * kp = (const struct htp_conv2d_dw_kernel_params *) octx->kernel_params;
    const struct htp_conv_2d_dw5_kernel_params p  = {
        .n_threads         = kp->n_hvx,
        .planes            = kp->total_tasks,
        .src_row_stride    = kp->row_src_bytes,
        .src_slot_bytes    = kp->pad_top_stride,
        .weight_slot_bytes = HTP_CONV2D_DW5_W_BYTES,
        .dst_slot_bytes    = kp->row_dst_bytes,
        .off_src           = 0,
        .off_weight        = 2 * kp->n_hvx * kp->pad_top_stride,
        .off_dst           = 2 * kp->n_hvx * kp->pad_top_stride + kp->n_hvx * HTP_CONV2D_DW5_W_BYTES,
    };

    return conv_2d_dw5_run_hvx(octx, &p);
}

struct conv2d_dw_dma_context {
    struct htp_ops_context *                   octx;
    const struct htp_conv2d_dw_kernel_params * kp;
    struct htp_conv2d_dw_vtcm_layout           layout;
    const void *                               w_vtcm;
    conv2d_dw_compute_fn_t                     compute_fn;
    uint32_t                                   task_start;
    uint32_t                                   task_count;
    uint32_t                                   tasks_per_hvx;
};

static void conv2d_dw_dma_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct conv2d_dw_dma_context * ctx  = (const struct conv2d_dw_dma_context *) data;
    const struct htp_ops_context * octx       = ctx->octx;
    const struct htp_conv2d_dw_kernel_params * kp = ctx->kp;
    const struct htp_conv2d_dw_vtcm_layout * L = &ctx->layout;
    const struct htp_tensor * src1            = octx->src[1];
    const struct htp_tensor * dst             = octx->dst;
    conv2d_dw_compute_fn_t compute_fn         = ctx->compute_fn;

    const uint32_t first = ctx->task_start + ith * ctx->tasks_per_hvx;
    const uint32_t last  = MIN(first + ctx->tasks_per_hvx, ctx->task_start + ctx->task_count);
    if (first >= last) {
        return;
    }

    (void) nth;

    struct htp_thread_trace * tr = th->trace;
    dma_queue * dma_q = th->dma_q;

    const uint32_t src_spad_half  = (uint32_t) L->src_spad_half_size;
    const uint32_t dst_spad_half  = (uint32_t) L->dst_spad_half_size;
    const uint32_t pad_top_stride = kp->pad_top_stride;

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    uint8_t * src_spad  = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, L->off_src) + ith * L->src_bytes_per_hvx;
    uint8_t * dst_spad  = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, L->off_dst) + ith * L->dst_bytes_per_hvx;

    float * srcb2[2] = { (float *) src_spad, (float *) (src_spad + src_spad_half) };
    float * dstb2[2] = { (float *) dst_spad, (float *) (dst_spad + dst_spad_half) };
    struct conv2d_dw_task_slice sl2[2];

    const uint32_t total = last - first;

    for (uint32_t i = 0; i < total && i < 2; ++i) {
        conv2d_dw_get_slice(&sl2[i], first + i, src1, dst, kp);

        dma_queue_push(dma_q,
                       dma_make_data(dst->data, dstb2[i]),
                       sl2[i].dst_bytes, dst_spad_half,
                       sl2[i].dst_bytes, 0);

        float * src_dst = srcb2[i] + (size_t) sl2[i].top_pad * pad_top_stride;
        dma_queue_push(dma_q,
                       dma_make_data(src_dst, sl2[i].src_addr),
                       src_spad_half, sl2[i].src_bytes,
                       sl2[i].src_bytes, sl2[i].src_bytes > 0 ? 1 : 0);
    }

    for (uint32_t i = 0; i < total; ++i) {
        const uint32_t task = first + i;
        const uint32_t buf  = i & 1u;
        float * srcb = srcb2[buf];
        float * dstb = dstb2[buf];

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) task);
        compute_fn(srcb, dstb, ctx->w_vtcm, &sl2[buf], octx, kp);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) task);

        dma_queue_push(dma_q,
                       dma_make_data(sl2[buf].dst_addr, dstb),
                       sl2[buf].dst_bytes, dst_spad_half,
                       sl2[buf].dst_bytes, sl2[buf].dst_bytes > 0 ? 1 : 0);

        if (i + 2 < total) {
            conv2d_dw_get_slice(&sl2[buf], task + 2, src1, dst, kp);

            float * src_dst = srcb + (size_t) sl2[buf].top_pad * pad_top_stride;
            dma_queue_push(dma_q,
                           dma_make_data(src_dst, sl2[buf].src_addr),
                           src_spad_half, sl2[buf].src_bytes,
                           sl2[buf].src_bytes, sl2[buf].src_bytes > 0 ? 1 : 0);
        }
    }

    dma_queue_flush(dma_q);

    FARF(HIGH, "conv2d_dw-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, kp->w, kp->h, kp->c, kp->n, kp->ow, kp->oh, kp->c, kp->n, first, last);
}

int op_conv_2d_dw(struct htp_ops_context * octx) {
    const struct htp_conv2d_dw_kernel_params * kp =
        (const struct htp_conv2d_dw_kernel_params *) octx->kernel_params;

    if (kp->total_tasks == 0) {
        return HTP_STATUS_OK;
    }

    if (kp->kernel_type <= HTP_CONV2D_DW_KERNEL_UNSUPPORTED ||
        kp->kernel_type >= HTP_CONV2D_DW_KERNEL_COUNT) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (kp->kernel_type == HTP_CONV2D_DW_KERNEL_WHCN_5X5_DMA) {
        return op_conv_2d_dw5(octx);
    }

    const uint32_t pad_rows      = (kp->pad_top_stride > 0) ? 4 : 0;
    const uint32_t max_ih_chunk  = MIN(kp->h, (kp->oh_chunk - 1) * (uint32_t) kp->s1 + (kp->kh - 1) * (uint32_t) kp->d1 + 1);
    const uint32_t max_src_bytes = (max_ih_chunk + pad_rows) * kp->row_src_bytes + 128;
    const uint32_t max_dst_bytes = kp->oh_chunk * kp->row_dst_bytes + 128;

    const struct htp_tensor * src0 = octx->src[0];
    const uint32_t elem_size       = kp->is_f16 ? sizeof(__fp16) : sizeof(float);
    const uint32_t w_size          = src0->size;

    struct htp_conv2d_dw_vtcm_layout layout;
    htp_conv2d_dw_vtcm_layout_build(&layout, max_src_bytes, max_dst_bytes, w_size, kp->n_hvx);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    void *    w_vtcm    = VTCM_LAYOUT_PTR(void, vtcm_base, layout.off_w);

    struct htp_tensor w_vtcm_tensor = *src0;
    w_vtcm_tensor.data = (dma_addr_t) w_vtcm;

    if (src0->nb[0] == elem_size) {
        dma_cpy_sametype_sameshape(octx->ctx->dma[0], &w_vtcm_tensor, src0, elem_size);
    } else {
        dma_sync_read(octx->ctx->dma[0], w_vtcm, src0->data, w_size);
    }
    dma_queue_flush(octx->ctx->dma[0]);

    uint32_t task_start = 0;
    uint32_t task_count = kp->total_tasks;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            task_count, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        task_start = range.start;
        task_count = range.count;
    }
    if (task_count == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t nth = MIN(kp->n_hvx, task_count);
    const uint32_t tasks_per_hvx = (task_count + nth - 1) / nth;

    struct conv2d_dw_dma_context ctx = {
        .octx             = octx,
        .kp               = kp,
        .layout           = layout,
        .w_vtcm           = w_vtcm,
        .compute_fn       = conv2d_dw_kernels[kp->kernel_type][kp->is_f16 ? 1 : 0],
        .task_start       = task_start,
        .task_count       = task_count,
        .tasks_per_hvx    = tasks_per_hvx,
    };

    work_queue_run(octx->ctx->hvx_queue, conv2d_dw_dma_thread, &ctx, nth);
    return HTP_STATUS_OK;
}
