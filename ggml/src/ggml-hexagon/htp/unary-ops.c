#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

struct htp_unary_context {
    struct htp_ops_context * octx;

    // Precomputed values
    const uint8_t *           data_src0;
    uint8_t *                 data_dst;

    size_t                    src0_row_size;
    size_t                    dst_row_size;

    size_t                    src0_row_size_aligned;
    size_t                    dst_row_size_aligned;

    size_t                    src0_spad_half_size;
    size_t                    dst_spad_half_size;

    uint32_t                  block;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_per_thread;
    uint32_t                  nc;
};

#define htp_unary_preamble            \
    const uint32_t ne00 = src->ne[0]; \
    const uint32_t ne01 = src->ne[1]; \
    const uint32_t ne02 = src->ne[2]; \
    const uint32_t ne03 = src->ne[3]; \
                                      \
    const uint32_t ne0 = dst->ne[0];  \
    const uint32_t ne1 = dst->ne[1];  \
    const uint32_t ne2 = dst->ne[2];  \
    const uint32_t ne3 = dst->ne[3];  \
                                      \
    const uint32_t nb00 = src->nb[0]; \
    const uint32_t nb01 = src->nb[1]; \
    const uint32_t nb02 = src->nb[2]; \
    const uint32_t nb03 = src->nb[3]; \
                                      \
    const uint32_t nb0 = dst->nb[0];  \
    const uint32_t nb1 = dst->nb[1];  \
    const uint32_t nb2 = dst->nb[2];  \
    const uint32_t nb3 = dst->nb[3];

static void hvx_fast_rms_norm_f32(const uint8_t * restrict src,
                                  uint8_t * restrict dst,
                                  uint8_t * restrict pad,
                                  const int num_elems,
                                  float     epsilon) {
    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    HVX_Vector sum_v     = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    int step_of_1 = num_elems >> 5;
    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v         = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v)); // replicated over all lanes

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }
}

static void scale_f32(const float * restrict src,
                      float * restrict dst,
                      uint8_t * restrict spad,
                      const uint32_t num_rows,
                      const uint32_t row_elems,
                      const size_t   row_size,
                      int32_t *      op_params) {
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_scale_offset_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, row_elems, scale, bias);
    }
}

static void rms_norm_f32(const float * restrict src,
                         float * restrict dst,
                         uint8_t * restrict spad,
                         const uint32_t num_rows,
                         const uint32_t row_elems,
                         const size_t   row_size,
                         int32_t *      op_params) {
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_fast_rms_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, spad, row_elems, epsilon);
    }
}

static void sqr_f32(const float * restrict src,
                    float * restrict dst,
                    uint8_t * restrict spad,
                    const uint32_t num_rows,
                    const uint32_t row_elems,
                    const size_t   row_size,
                    int32_t *      op_params) {

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, row_elems);
    }
}

static void sqrt_f32(const float * restrict src,
                     float * restrict dst,
                     uint8_t * restrict spad,
                     const uint32_t num_rows,
                     const uint32_t row_elems,
                     const size_t   row_size,
                     int32_t *      op_params) {

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, row_elems);
    }
}

static void unary_job_f32_per_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = &octx->src0;
    const struct htp_tensor * dst = &octx->dst;

    htp_unary_preamble;

    int                       htp_op = octx->op;
    int32_t *                 op_params = octx->op_params;
    uint32_t                  src0_nrows_per_thread = uctx->src0_nrows_per_thread;

    const size_t src0_row_size = uctx->src0_row_size;
    const size_t dst_row_size  = uctx->dst_row_size;

    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict data_src = uctx->data_src0;
    uint8_t * restrict       data_dst = uctx->data_dst;

    uint8_t * src0_spad_data = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_data  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    size_t src0_spad_half_size = uctx->src0_spad_half_size;
    size_t dst_spad_half_size  = uctx->dst_spad_half_size;

    const int BLOCK = uctx->block;
    if (BLOCK == 0) {
        FARF(ERROR, "unary-f32 : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n",
             octx->src0_spad.size_per_thread, src0_row_size_aligned);
        return;
    }

    dma_queue * dma_queue = octx->ctx->dma[ith];

    for (uint32_t ir = src0_start_row, spad_idx = 0; ir < src0_end_row && spad_idx < 2; ir += BLOCK, spad_idx++) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        // Dummy DMA transation for sequencing (interleaving dst,src,dst,...)
        dma_queue_push_vtcm_to_ddr(dma_queue,
            dma_make_ptr(data_dst, dst_spad_data + (spad_idx * dst_spad_half_size)),
            dst_row_size, dst_row_size_aligned, 0);

        dma_queue_push_ddr_to_vtcm(dma_queue,
            dma_make_ptr(src0_spad_data + (spad_idx * src0_spad_half_size), data_src + (ir * src0_row_size)),
            src0_row_size_aligned, src0_row_size, block_size);
    }

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir += BLOCK) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        float * dst_spad  = (float *) dma_queue_pop(dma_queue).src;
        float * src0_spad = (float *) dma_queue_pop(dma_queue).dst;

        // Process block in VTCM
        switch (htp_op) {
            case HTP_OP_RMS_NORM:
                rms_norm_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            case HTP_OP_SCALE:
                scale_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            case HTP_OP_SQR:
                sqr_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            case HTP_OP_SQRT:
                sqrt_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            default:
                break;
        }

        dma_queue_push_vtcm_to_ddr(dma_queue,
            dma_make_ptr(data_dst + (ir * dst_row_size), dst_spad),
            dst_row_size, dst_row_size_aligned, block_size);

        // prefetch N+2 loop iteration if any
        const uint32_t pref_block = (ir + BLOCK * 2);
        if (pref_block < src0_end_row) {
            const uint32_t pref_block_size = MIN(BLOCK, src0_end_row - pref_block);
            dma_queue_push_ddr_to_vtcm(dma_queue,
                dma_make_ptr(src0_spad, data_src + (pref_block * src0_row_size)),
                src0_row_size_aligned, src0_row_size, pref_block_size);
        }
    }

    dma_queue_flush(dma_queue);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "unary-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n", ith, nth, src->ne[0],
         src->ne[1], src->ne[2], src->ne[3], src0_start_row, src0_end_row, dst->ne[0], dst->ne[1], dst->ne[2],
         dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static int execute_op_unary_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = &octx->src0;
    struct htp_tensor *       dst  = &octx->dst;

    const char * op_type = NULL;

    switch (octx->op) {
        case HTP_OP_RMS_NORM:
            op_type = "rmsnorm-f32";
            break;
        case HTP_OP_SCALE:
            op_type = "scale-f32";
            break;
        case HTP_OP_SQR:
            op_type = "sqr-f32";
            break;
        case HTP_OP_SQRT:
            op_type = "sqrt-f32";
            break;

        default:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const int      n_threads  = octx->n_threads;
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    const size_t src0_row_size = src0->nb[1];
    const size_t dst_row_size  = dst->nb[1];

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    // VTCM scratchpads for all tensors
    // N rows per thread, padded to HVX vector size
    // Double buffering requires 2x size per buffer

    size_t spad_size_per_row   = 2 * (src0_row_size_aligned + dst_row_size_aligned);
    size_t vtcm_row_per_thread = (octx->ctx->vtcm_size)/ (n_threads * spad_size_per_row);

    // Make sure the reserved vtcm size is sufficient
    if (vtcm_row_per_thread == 0) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size,
             spad_size_per_row * n_threads);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.size_per_thread = src0_row_size_aligned * vtcm_row_per_thread * 2;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned * vtcm_row_per_thread * 2;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size;

    FARF(HIGH, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-spad-size %u src1-spad-size %u dst-spad-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size);

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs = MIN(n_threads, src0_nrows);

        struct htp_unary_context uctx = {
            .octx                  = octx,
            .src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs,
            .src0_nrows            = src0_nrows,

            .data_src0             = (const uint8_t *)src0->data,
            .data_dst              = (uint8_t *)dst->data,

            .src0_row_size         = src0_row_size,
            .dst_row_size          = dst_row_size,

            .src0_row_size_aligned = src0_row_size_aligned,
            .dst_row_size_aligned  = dst_row_size_aligned,

            .src0_spad_half_size   = octx->src0_spad.size_per_thread / 2,
            .dst_spad_half_size    = octx->dst_spad.size_per_thread / 2,

            .block                 = (octx->src0_spad.size_per_thread / 2) / src0_row_size_aligned,
            .nc                    = src0->ne[0],
        };

        worker_pool_run_func(octx->ctx->worker_pool, unary_job_f32_per_thread, &uctx, n_jobs);
    }

    return err;
}

int op_unary(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src0.type) {
        case HTP_TYPE_F32:
            err = execute_op_unary_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
