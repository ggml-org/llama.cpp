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

typedef void (*hvx_elemwise_f32_func)(uint8_t * data_dst, const uint8_t * src0, const uint8_t * src1, const uint32_t num_elems);

static hvx_elemwise_f32_func func_table_HVX[]     = { hvx_mul_f32, hvx_add_f32, hvx_sub_f32 };
static hvx_elemwise_f32_func func_table_HVX_opt[] = { hvx_mul_f32_aa, hvx_add_f32_aa, hvx_sub_f32_aa };

#define htp_binary_preamble            \
    const struct htp_tensor * src0 = &octx->src0; \
    const struct htp_tensor * src1 = &octx->src1; \
    const struct htp_tensor * src2 = &octx->src2; \
    struct htp_tensor *       dst  = &octx->dst;  \
                                       \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne10 = src1->ne[0]; \
    const uint32_t ne11 = src1->ne[1]; \
    const uint32_t ne12 = src1->ne[2]; \
    const uint32_t ne13 = src1->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb10 = src1->nb[0]; \
    const uint32_t nb11 = src1->nb[1]; \
    const uint32_t nb12 = src1->nb[2]; \
    const uint32_t nb13 = src1->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];   \
                                       \
    const uint32_t src0_nrows_per_thread = octx->src0_nrows_per_thread;

static void binary_job_f32_per_thread(struct htp_ops_context * octx,
                                      uint8_t *                spad_data,
                                      uint32_t                 nth,
                                      uint32_t                 ith,
                                      enum htp_op              op) {
    htp_binary_preamble;

    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;
    const size_t dst_row_size  = nb1;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows
    const uint32_t src1_nrows = ne11 * ne12 * ne13;  // src1 rows

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    int is_aligned = 1;
    int opt_path   = 0;
    if ((0 == hex_is_aligned((void *) src0->data, VLEN)) || (0 == hex_is_aligned((void *) src1->data, VLEN)) ||
        (0 == hex_is_aligned((void *) dst->data, VLEN))) {
        is_aligned = 0;
    }
    if ((1 == is_aligned) && !(nb01 & (VLEN - 1))) {
        opt_path = 1;
    }

    hvx_elemwise_f32_func func_HVX = (1 == opt_path) ? func_table_HVX_opt[op] : func_table_HVX[op];

    uint8_t * restrict spad_data_th = spad_data + (ith * src0_row_size);

    const uint8_t * restrict src0_ptr = (const uint8_t *) src0->data + (src0_start_row * src0_row_size);
    uint8_t * restrict dst_ptr        = (uint8_t *) dst->data + (src0_start_row * dst_row_size);

    const uint8_t * restrict data_src1 = (const uint8_t *) src1->data;

    const uint32_t ne02_ne01 = ne02 * ne01;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir++) {
        const uint32_t i03 = fastdiv(ir, &octx->src0_div21);
        const uint32_t i02 = fastdiv(ir - i03 * ne02_ne01, &octx->src0_div1);
        const uint32_t i01 = (ir - i03 * ne02_ne01 - i02 * ne01);

        const uint32_t i13 = fastmodulo(i03, ne13, &octx->src1_div3);
        const uint32_t i12 = fastmodulo(i02, ne12, &octx->src1_div2);
        const uint32_t i11 = fastmodulo(i01, ne11, &octx->src1_div1);

        const uint8_t * restrict src1_ptr = data_src1 + i13 * nb13 + i12 * nb12 + i11 * src1_row_size;

        if (ir + 1 < src0_end_row) {
            hex_l2fetch(src0_ptr + ne00, src0_row_size, src0_row_size, 1);
            if (src1_row_size == src0_row_size) {
                hex_l2fetch(src1_ptr, src1_row_size, src1_row_size, 1);
            }
        }

        const uint32_t nr0 = ne00 / ne10;
        if (nr0 > 1) {
            if ((1 == is_aligned) && (nr0 == ne00)) {
                hvx_splat_f32_a(spad_data_th, *(float *) src1_ptr, nr0);
            } else {
                for (uint32_t r = 0; r < nr0; r++) {
                    memcpy(spad_data_th + r * nb11, (const uint8_t *) src1_ptr, nb11);
                }
            }
            func_HVX((uint8_t *) dst_ptr, (const uint8_t *) src0_ptr, (const uint8_t *) spad_data_th, ne00);
        } else {
            func_HVX((uint8_t *) dst_ptr, (const uint8_t *) src0_ptr, (const uint8_t *) src1_ptr, ne00);
        }

        src0_ptr += src0_row_size;
        dst_ptr += dst_row_size;
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "binary-f32 %d/%d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth, opt_path,
         ne00, ne01, ne02, ne03, src0_start_row, src0_end_row, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void binary_add_id_job_f32_per_thread(struct htp_ops_context * octx,
                                             uint8_t *                spad_data,
                                             uint32_t                 nth,
                                             uint32_t                 ith,
                                             hvx_elemwise_f32_func    func_HVX) {
    htp_binary_preamble;

    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;
    const size_t dst_row_size  = nb1;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict data_src0 = (const uint8_t *) src0->data;
    const uint8_t * restrict data_src1 = (const uint8_t *) src1->data;
    uint8_t * restrict data_dst        = (uint8_t *) dst->data;

    const uint32_t ne02_ne01  = ne02 * ne01;
    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir++) {
        // src0 indices
        const uint32_t i03 = fastdiv(ir, &octx->src0_div21);
        const uint32_t i02 = fastdiv(ir - i03 * ne02_ne01, &octx->src0_div1);
        const uint32_t i01 = (ir - i03 * ne02_ne01 - i02 * ne01);

        // src1 indices
        const int i11 = *(int32_t *) ((char *) src2->data + i01 * src2->nb[0] + i02 * src2->nb[1]);
        assert(i11 >= 0 && i11 < ne11);

        float * restrict dst_ptr        = (float *) (data_dst + i03 * nb3 + i02 * nb2 + i01 * nb1);
        const float * restrict src0_ptr = (const float *) (data_src0 + i03 * nb03 + i02 * nb02 + i01 * nb01);
        const float * restrict src1_ptr = (const float *) (data_src1 + 0 + 0 + i11 * nb11);

        if (ir + 1 < src0_end_row) {
            hex_l2fetch(src0_ptr + ne00, src0_row_size, src0_row_size, 1);
            if (src1_row_size == src0_row_size) {
                hex_l2fetch(src1_ptr + ne10, src1_row_size, src1_row_size, 1);
            }
        }

        const uint32_t nr0 = ne00 / ne10;
        if (nr0 > 1) {
            for (uint32_t r = 0; r < nr0; r++) {
                memcpy(spad_data + r * nb10, (const uint8_t *) src1_ptr, nb10);
            }
            func_HVX((uint8_t *) dst_ptr, (const uint8_t *) src0_ptr, (const uint8_t *) spad_data, ne00);
        } else {
            func_HVX((uint8_t *) dst_ptr, (const uint8_t *) src0_ptr, (const uint8_t *) src1_ptr, ne00);
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "add-id-f32 %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u usec %u\n", ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],
         src1->ne[2], src1->ne[3], src2->ne[0], src2->ne[1], src2->ne[2], src2->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void binary_job_dispatcher_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = (struct htp_ops_context *) data;

    switch (octx->op) {
        case HTP_OP_MUL:
        case HTP_OP_ADD:
        case HTP_OP_SUB:
            binary_job_f32_per_thread(octx, octx->src1_spad.data, n, i, octx->op);
            break;

        case HTP_OP_ADD_ID:
            binary_add_id_job_f32_per_thread(octx, octx->src0_spad.data, n, i, hvx_add_f32);
            break;

        default:
            FARF(ERROR, "Unknown Binary Op %u", octx->op);
            break;
    }
}

static int execute_op_binary_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    worker_callback_t binary_op_func;
    const char *      op_type = NULL;

    switch (octx->op) {
        case HTP_OP_MUL:
            binary_op_func = binary_job_dispatcher_f32;
            op_type        = "mul-f32";
            break;

        case HTP_OP_ADD:
            binary_op_func = binary_job_dispatcher_f32;
            op_type        = "add-f32";
            break;

        case HTP_OP_SUB:
            binary_op_func = binary_job_dispatcher_f32;
            op_type        = "sub-f32";
            break;

        case HTP_OP_ADD_ID:
            binary_op_func = binary_job_dispatcher_f32;
            op_type        = "add-id-f32";
            break;

        default:
            FARF(ERROR, "Unsupported binary-Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const int      n_threads  = octx->n_threads;
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    const size_t src0_row_size = src0->nb[1];
    const size_t src1_row_size = src1->nb[1];
    const size_t dst_row_size  = dst->nb[1];

    // VTCM scratchpads for all tensors
    octx->dst_spad.size  = hex_round_up(dst_row_size, 128) * n_threads;
    octx->src0_spad.size = hex_round_up(src0_row_size, 128) * n_threads;
    octx->src1_spad.size = hex_round_up(src1_row_size, 128) * n_threads;

    size_t spad_size = octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size;

    FARF(HIGH,
         "%s: (%ux%ux%ux%u) * (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-spad-size %u src1-spad-size %u dst-spad-size %u\n",
         op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
         src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], octx->src0_spad.size, octx->src1_spad.size,
         octx->dst_spad.size);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "binary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type,
             octx->ctx->vtcm_size, spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs = MIN(n_threads, src0_nrows);

        octx->src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;

        octx->src0_div21 = init_fastdiv_values(src0->ne[2] * src0->ne[1]);
        octx->src0_div3  = init_fastdiv_values(src0->ne[3]);
        octx->src0_div2  = init_fastdiv_values(src0->ne[2]);
        octx->src0_div1  = init_fastdiv_values(src0->ne[1]);

        octx->src1_div21 = init_fastdiv_values(src1->ne[2] * src1->ne[1]);
        octx->src1_div3  = init_fastdiv_values(src1->ne[3]);
        octx->src1_div2  = init_fastdiv_values(src1->ne[2]);
        octx->src1_div1  = init_fastdiv_values(src1->ne[1]);

        worker_pool_run_func(octx->ctx->worker_pool, binary_op_func, octx, n_jobs);
    }

    return err;
}

int op_binary(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src0.type) {
        case HTP_TYPE_F32:
            err = execute_op_binary_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
