#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-common.h"
#include "dma-queue.h"
#include "dma-pipeline.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "hvx-quant.h"
#include "get-rows-ops.h"
#include "work-queue.h"

struct get_rows_context;

typedef void (*get_rows_flat_compute_fn_t)(float * dst_spad, const void * src_spad, uint32_t cur_elems);
typedef void (*get_rows_tiled_compute_fn_t)(void * dst_spad, const uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row);

struct get_rows_context {
    struct htp_ops_context * octx;
    const struct htp_get_rows_kernel_params * kparams;
    struct htp_get_rows_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_base;
    const void * src1_data;
    get_rows_flat_compute_fn_t  compute_flat;
    get_rows_tiled_compute_fn_t compute_tiled;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_hvx;
    uint32_t tile_size;
    uint32_t tile_stride;
    uint32_t src0_row_size;
    uint32_t dst_row_size;
    int idx_type;
};

#define get_rows_preamble                                              \
    const uint32_t __attribute__((unused)) ne00 = octx->src[0]->ne[0]; \
    const uint32_t __attribute__((unused)) ne01 = octx->src[0]->ne[1]; \
    const uint32_t __attribute__((unused)) ne02 = octx->src[0]->ne[2]; \
    const uint32_t __attribute__((unused)) ne03 = octx->src[0]->ne[3]; \
                                                                       \
    const uint32_t __attribute__((unused)) ne10 = octx->src[1]->ne[0]; \
    const uint32_t __attribute__((unused)) ne11 = octx->src[1]->ne[1]; \
    const uint32_t __attribute__((unused)) ne12 = octx->src[1]->ne[2]; \
    const uint32_t __attribute__((unused)) ne13 = octx->src[1]->ne[3]; \
                                                                       \
    const uint32_t __attribute__((unused)) ne0 = octx->dst->ne[0];     \
    const uint32_t __attribute__((unused)) ne1 = octx->dst->ne[1];     \
    const uint32_t __attribute__((unused)) ne2 = octx->dst->ne[2];     \
    const uint32_t __attribute__((unused)) ne3 = octx->dst->ne[3];     \
                                                                       \
    const uint32_t __attribute__((unused)) nb01 = octx->src[0]->nb[1]; \
    const uint32_t __attribute__((unused)) nb02 = octx->src[0]->nb[2]; \
    const uint32_t __attribute__((unused)) nb03 = octx->src[0]->nb[3]; \
                                                                       \
    const uint32_t __attribute__((unused)) nb10 = octx->src[1]->nb[0]; \
    const uint32_t __attribute__((unused)) nb11 = octx->src[1]->nb[1]; \
    const uint32_t __attribute__((unused)) nb12 = octx->src[1]->nb[2]; \
                                                                       \
    const uint32_t __attribute__((unused)) nb1 = octx->dst->nb[1];     \
    const uint32_t __attribute__((unused)) nb2 = octx->dst->nb[2];     \
    const uint32_t __attribute__((unused)) nb3 = octx->dst->nb[3];     \
                                                                       \
    const uint32_t __attribute__((unused)) nr = ne10 * ne11 * ne12;

static inline uint32_t get_rows_read_index(
    const void * src1,
    uint32_t idx,
    int idx_type
) {
    const uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? sizeof(int64_t) : sizeof(int32_t));
    return *(const uint32_t *) ((const uint8_t *) src1 + off);
}

static void get_rows_thread_sametype(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct get_rows_context * grctx = (struct get_rows_context *) data;
    struct htp_ops_context * octx = grctx->octx;
    const struct htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    const uint32_t dr  = grctx->tasks_per_hvx;
    const uint32_t ir0 = grctx->task_start + dr * ith;
    if (ir0 >= grctx->task_start + grctx->tasks) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);
    const uint32_t row_size_bytes = grctx->src0_row_size;
    dma_queue * dma_q = th->dma_q;

    for (uint32_t i = ir0; i < ir1; ++i) {
        const uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        const uint32_t rem = i - i12 * ne11 * ne10;
        const uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        const uint32_t i10 = rem - i11 * ne10;

        const uint32_t i01 = get_rows_read_index(grctx->src1_data, i, grctx->idx_type);
        assert(i01 < ne01);

        const uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        const uint32_t i02 = i11 - q02 * ne02;
        const uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        const uint32_t i03 = i12 - q03 * ne03;

        const dma_addr_t src0_data = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03;
        const dma_addr_t dst_data  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3;

        while (!dma_queue_push(dma_q, dma_make_data(dst_data, src0_data), nb1, nb01, row_size_bytes, 1)) {
            dma_queue_pop(dma_q);
        }
    }
    dma_queue_flush(dma_q);
}



static inline HVX_Vector get_rows_clamp_f16_finite(HVX_Vector v) {
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    const HVX_Vector inf     = hvx_vec_splat_f32(INFINITY);
    const HVX_VectorPred finite = Q6_Q_vcmp_gt_VsfVsf(inf, hvx_vec_abs_f32(v));
    const HVX_VectorPred high = Q6_Q_and_QQ(finite, Q6_Q_vcmp_gt_VsfVsf(v, max_f16));
    const HVX_VectorPred low  = Q6_Q_and_QQ(finite, Q6_Q_vcmp_gt_VsfVsf(min_f16, v));
    v = Q6_V_vmux_QVV(high, max_f16, v);
    return Q6_V_vmux_QVV(low, min_f16, v);
}

static __attribute__((noinline)) void get_rows_flat_f32_to_f16(
    float * restrict dst_ptr,
    const void * restrict src_ptr,
    uint32_t n
) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src_ptr;
    HVX_Vector       * restrict vdst = (HVX_Vector *) dst_ptr;

    const uint32_t n_vec = n / 64;
    const uint32_t n_rem = n % 64;

    for (uint32_t i = 0; i < n_vec; i++) {
        const HVX_Vector v0 = get_rows_clamp_f16_finite(vsrc[2 * i + 0]);
        const HVX_Vector v1 = get_rows_clamp_f16_finite(vsrc[2 * i + 1]);
        vdst[i] = hvx_vec_f32_to_f16(v0, v1);
    }

    if (n_rem > 0) {
        const HVX_Vector v0 = get_rows_clamp_f16_finite(vsrc[2 * n_vec + 0]);
        const HVX_Vector v1 = (n_rem > 32) ? get_rows_clamp_f16_finite(vsrc[2 * n_vec + 1]) : Q6_V_vzero();
        const HVX_Vector v  = hvx_vec_f32_to_f16(v0, v1);
        hvx_vec_store_a(&vdst[n_vec], n_rem * sizeof(__fp16), v);
    }
}

static inline void get_rows_pack_f32_to_f16_inplace(
    void * spad_ptr,
    uint32_t n
) {
    HVX_Vector * vec = (HVX_Vector *) spad_ptr;

    const uint32_t n_vec = n / 64;
    const uint32_t n_rem = n % 64;

    for (uint32_t i = 0; i < n_vec; i++) {
        const HVX_Vector v0 = get_rows_clamp_f16_finite(vec[2 * i + 0]);
        const HVX_Vector v1 = get_rows_clamp_f16_finite(vec[2 * i + 1]);
        vec[i] = hvx_vec_f32_to_f16(v0, v1);
    }

    if (n_rem > 0) {
        const HVX_Vector v0 = get_rows_clamp_f16_finite(vec[2 * n_vec + 0]);
        const HVX_Vector v1 = (n_rem > 32) ? get_rows_clamp_f16_finite(vec[2 * n_vec + 1]) : Q6_V_vzero();
        const HVX_Vector v  = hvx_vec_f32_to_f16(v0, v1);
        hvx_vec_store_a(&vec[n_vec], n_rem * sizeof(__fp16), v);
    }
}

#define DEFINE_GET_ROWS_FLAT_F16(type)                                              \
    static void get_rows_flat_##type##_f16(                                         \
        float * dst_spad, const void * src_spad, uint32_t n) {                      \
        hvx_dequantize_row_flat_##type##_f32(dst_spad, src_spad, n);                \
        get_rows_pack_f32_to_f16_inplace(dst_spad, n);                              \
    }

DEFINE_GET_ROWS_FLAT_F16(bf16)
DEFINE_GET_ROWS_FLAT_F16(q1_0)
DEFINE_GET_ROWS_FLAT_F16(q2_0)
DEFINE_GET_ROWS_FLAT_F16(q8_0)
DEFINE_GET_ROWS_FLAT_F16(q4_0)
DEFINE_GET_ROWS_FLAT_F16(q4_k)
DEFINE_GET_ROWS_FLAT_F16(q5_k)
DEFINE_GET_ROWS_FLAT_F16(q6_k)

#undef DEFINE_GET_ROWS_FLAT_F16

#define DEFINE_GET_ROWS_TILED_F16(type, tile_k)                                     \
    static void get_rows_tiled_##type##_f16(                                        \
        void * dst_spad, const uint8_t * src_spad, uint32_t n_k_tiles,              \
        uint32_t tile_stride, uint32_t row) {                                       \
        hvx_dequantize_row_tiled_##type##_f32(                                      \
            dst_spad, src_spad, n_k_tiles, tile_stride, row);                       \
        get_rows_pack_f32_to_f16_inplace(dst_spad, n_k_tiles * (tile_k));           \
    }

DEFINE_GET_ROWS_TILED_F16(q1_0, HTP_TENSOR_TILE_K_Q1_0)
DEFINE_GET_ROWS_TILED_F16(q2_0, HTP_TENSOR_TILE_K_Q2_0)
DEFINE_GET_ROWS_TILED_F16(q8_0, HTP_TENSOR_TILE_N_COLS)
DEFINE_GET_ROWS_TILED_F16(q4_0, HTP_TENSOR_TILE_N_COLS)
DEFINE_GET_ROWS_TILED_F16(q4_k, HTP_TENSOR_TILE_N_COLS)
DEFINE_GET_ROWS_TILED_F16(q5_k, HTP_TENSOR_TILE_N_COLS)
DEFINE_GET_ROWS_TILED_F16(q6_k, HTP_TENSOR_TILE_N_COLS)

#undef DEFINE_GET_ROWS_TILED_F16

struct get_rows_flat_task {
    dma_addr_t src0_data;
    dma_addr_t dst_data;
};

static inline struct get_rows_flat_task get_rows_flat_calc_task(
    const struct htp_ops_context * octx,
    const struct get_rows_context * grctx,
    uint32_t i,
    bool need_src0
) {
    const struct htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    const uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    const uint32_t rem = i - i12 * ne11 * ne10;
    const uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    const uint32_t i10 = rem - i11 * ne10;

    struct get_rows_flat_task task;
    task.src0_data = 0;
    task.dst_data  = octx->dst->data + i10*nb1 + i11*nb2 + i12*nb3;

    if (need_src0) {
        const uint32_t i01 = get_rows_read_index(grctx->src1_data, i, grctx->idx_type);
        assert(i01 < ne01);

        const uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        const uint32_t i02 = i11 - q02 * ne02;
        const uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        const uint32_t i03 = i12 - q03 * ne03;

        task.src0_data = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03;
    }

    return task;
}

struct get_rows_flat_pipe_state {
    const struct htp_ops_context * octx;
    const struct get_rows_context * grctx;
    get_rows_flat_compute_fn_t     compute;
    uint32_t                       ne00;
    uint32_t                       src0_bytes;
    uint32_t                       dst_bytes;
    size_t                         src0_spad_half;
    size_t                         dst_spad_half;
    uint32_t                       ir;
    uint32_t                       ir_end;
};

static bool get_rows_flat_next_task(void * ctx, struct dma_pipe_task * task) {
    struct get_rows_flat_pipe_state * s = (struct get_rows_flat_pipe_state *) ctx;
    if (s->ir >= s->ir_end) {
        return false;
    }
    task->row = s->ir++;
    task->off = 0;
    task->len = 0;
    task->aux = 0;
    return true;
}

static void get_rows_flat_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct get_rows_flat_pipe_state * s = (struct get_rows_flat_pipe_state *) ctx;
    struct get_rows_flat_task t = get_rows_flat_calc_task(s->octx, s->grctx, task->row, true);
    dma_queue_push(dma_q, dma_make_data(in_spad, t.src0_data),
                   s->src0_spad_half, s->src0_bytes, s->src0_bytes, 1);
}

static void get_rows_flat_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    (void) task;
    struct get_rows_flat_pipe_state * s = (struct get_rows_flat_pipe_state *) ctx;
    s->compute((float *) out_spad, in_spad, s->ne00);
}

static void get_rows_flat_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    struct get_rows_flat_pipe_state * s = (struct get_rows_flat_pipe_state *) ctx;
    struct get_rows_flat_task t = get_rows_flat_calc_task(s->octx, s->grctx, task->row, false);
    dma_queue_push(dma_q, dma_make_data(t.dst_data, out_spad),
                   s->dst_bytes, s->dst_spad_half, s->dst_bytes, 1);
}

static void get_rows_thread_flat(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct get_rows_context * grctx = (struct get_rows_context *) data;
    struct htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    const uint32_t dr  = grctx->tasks_per_hvx;
    const uint32_t ir0 = grctx->task_start + dr * ith;
    if (ir0 >= grctx->task_start + grctx->tasks) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);

    const struct htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_hvx;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_hvx;
    void *    in_buf[2]  = { vtcm_src0, vtcm_src0 + vtcm_layout->src0_spad_half_size };
    void *    out_buf[2] = { vtcm_dst,  vtcm_dst  + vtcm_layout->dst_spad_half_size };

    struct get_rows_flat_pipe_state s = {
        .octx           = octx,
        .grctx          = grctx,
        .compute        = grctx->compute_flat,
        .ne00           = ne00,
        .src0_bytes     = grctx->src0_row_size,
        .dst_bytes      = grctx->dst_row_size,
        .src0_spad_half = vtcm_layout->src0_spad_half_size,
        .dst_spad_half  = vtcm_layout->dst_spad_half_size,
        .ir             = ir0,
        .ir_end         = ir1,
    };

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf,
                    get_rows_flat_next_task, get_rows_flat_push_in, get_rows_flat_compute, get_rows_flat_push_out);
}



struct get_rows_tiled_task {
    dma_addr_t tile_src_base;
    dma_addr_t dst_data;
    uint32_t   row;
};

static inline struct get_rows_tiled_task get_rows_tiled_calc_task(
    const struct htp_ops_context * octx,
    const struct get_rows_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_size
) {
    const struct htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    const uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    const uint32_t rem = i - i12 * ne11 * ne10;
    const uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    const uint32_t i10 = rem - i11 * ne10;
    const uint32_t i01 = get_rows_read_index(grctx->src1_data, i, grctx->idx_type);
    assert(i01 < ne01);

    const uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    const uint32_t i02 = i11 - q02 * ne02;
    const uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    const uint32_t i03 = i12 - q03 * ne03;
    const uint32_t column_tile = i01 / HTP_TENSOR_TILE_N_ROWS;
    const uint32_t row = i01 % HTP_TENSOR_TILE_N_ROWS;
    const dma_addr_t matrix = octx->src[0]->data + i02*nb02 + i03*nb03;

    struct get_rows_tiled_task task;
    task.tile_src_base = matrix + (column_tile * n_k_tiles) * tile_size;
    task.dst_data      = octx->dst->data + i10*nb1 + i11*nb2 + i12*nb3;
    task.row           = row;
    return task;
}

struct get_rows_tiled_pipe_state {
    const struct htp_ops_context * octx;
    const struct get_rows_context * grctx;
    get_rows_tiled_compute_fn_t    compute_tiled;
    uint32_t                       n_k_tiles;
    uint32_t                       tile_size;
    uint32_t                       tile_stride;
    uint32_t                       dst_bytes;
    size_t                         dst_spad_half;
    uint32_t                       ir;
    uint32_t                       ir_end;
};

static bool get_rows_tiled_next_task(void * ctx, struct dma_pipe_task * task) {
    struct get_rows_tiled_pipe_state * s = (struct get_rows_tiled_pipe_state *) ctx;
    if (s->ir >= s->ir_end) {
        return false;
    }
    task->row = s->ir++;
    task->off = 0;
    task->len = 0;
    task->aux = 0;
    return true;
}

static void get_rows_tiled_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct get_rows_tiled_pipe_state * s = (struct get_rows_tiled_pipe_state *) ctx;
    struct get_rows_tiled_task t = get_rows_tiled_calc_task(s->octx, s->grctx, task->row, s->n_k_tiles, s->tile_size);
    dma_queue_push(dma_q, dma_make_data(in_spad, t.tile_src_base),
                   s->tile_stride, s->tile_size, s->tile_size, s->n_k_tiles);
}

static void get_rows_tiled_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    struct get_rows_tiled_pipe_state * s = (struct get_rows_tiled_pipe_state *) ctx;
    struct get_rows_tiled_task t = get_rows_tiled_calc_task(s->octx, s->grctx, task->row, s->n_k_tiles, s->tile_size);
    s->compute_tiled(out_spad, (const uint8_t *) in_spad, s->n_k_tiles, s->tile_stride, t.row);
}

static void get_rows_tiled_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    struct get_rows_tiled_pipe_state * s = (struct get_rows_tiled_pipe_state *) ctx;
    struct get_rows_tiled_task t = get_rows_tiled_calc_task(s->octx, s->grctx, task->row, s->n_k_tiles, s->tile_size);
    dma_queue_push(dma_q, dma_make_data(t.dst_data, out_spad),
                   s->dst_bytes, s->dst_spad_half, s->dst_bytes, 1);
}

static void get_rows_thread_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct get_rows_context * grctx = (struct get_rows_context *) data;
    struct htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    const uint32_t dr  = grctx->tasks_per_hvx;
    const uint32_t ir0 = grctx->task_start + dr * ith;
    if (ir0 >= grctx->task_start + grctx->tasks) {
        return;
    }

    const uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);
    const uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    const struct htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_base = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_hvx;
    uint8_t * dst_spad_base = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_hvx;
    void *    in_buf[2]     = { src_spad_base, src_spad_base + vtcm_layout->src0_spad_half_size };
    void *    out_buf[2]    = { dst_spad_base, dst_spad_base + vtcm_layout->dst_spad_half_size };

    struct get_rows_tiled_pipe_state s = {
        .octx          = octx,
        .grctx         = grctx,
        .compute_tiled = grctx->compute_tiled,
        .n_k_tiles     = n_k_tiles,
        .tile_size     = grctx->tile_size,
        .tile_stride   = grctx->tile_stride,
        .dst_bytes     = grctx->dst_row_size,
        .dst_spad_half = vtcm_layout->dst_spad_half_size,
        .ir            = ir0,
        .ir_end        = ir1,
    };

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf,
                    get_rows_tiled_next_task, get_rows_tiled_push_in, get_rows_tiled_compute, get_rows_tiled_push_out);
}

int op_get_rows(struct htp_ops_context * octx) {
    const struct htp_get_rows_kernel_params * kparams = (const struct htp_get_rows_kernel_params *) octx->kernel_params;

    const uint32_t ne00          = octx->src[0]->ne[0];
    const uint32_t src0_row_size = htp_tensor_get_row_size(octx->src[0]->type, ne00);
    const uint32_t dst_row_size  = htp_tensor_get_row_size(octx->dst->type, ne00);
    if (src0_row_size == 0 || dst_row_size == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (kparams->kernel_type == HTP_GET_ROWS_KERNEL_SAMETYPE) {
        if (octx->src[0]->type != octx->dst->type) {
            return HTP_STATUS_NO_SUPPORT;
        }
    } else if ((octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16) ||
               (octx->src[0]->type == HTP_TYPE_F32 && octx->dst->type == HTP_TYPE_F32)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t total_tasks    = kparams->total_tasks;

    uint32_t task_start = 0;
    uint32_t tasks      = total_tasks;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_tasks, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        task_start = range.start;
        tasks      = range.count;
    }

    if (tasks == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx = octx->n_hvx;

    struct get_rows_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;
    grctx.compute_flat = NULL;
    grctx.compute_tiled = NULL;
    grctx.task_start = task_start;
    grctx.tasks = tasks;
    grctx.tasks_per_hvx = fastdiv(tasks + n_hvx - 1, &octx->n_hvx_div);
    grctx.tile_size   = 0;
    grctx.tile_stride = 0;
    grctx.src0_row_size = src0_row_size;
    grctx.dst_row_size  = dst_row_size;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_rows_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, octx->dst->type, ne00, n_hvx);

    if (grctx.vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, grctx.vtcm_layout.total_bytes);
        return HTP_STATUS_INVAL_PARAMS;
    }

    work_queue_func_t q_func = NULL;
    switch (kparams->kernel_type) {
        case HTP_GET_ROWS_KERNEL_SAMETYPE:
            q_func = get_rows_thread_sametype;
            break;
        case HTP_GET_ROWS_KERNEL_TILED:
            if (octx->dst->type == HTP_TYPE_F16) {
                switch (octx->src[0]->type) {
                    case HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_row_tiled_f16_f16; break;
                    case HTP_TYPE_Q1_0: grctx.compute_tiled = get_rows_tiled_q1_0_f16;          break;
                    case HTP_TYPE_Q2_0: grctx.compute_tiled = get_rows_tiled_q2_0_f16;          break;
                    case HTP_TYPE_Q8_0: grctx.compute_tiled = get_rows_tiled_q8_0_f16;          break;
                    case HTP_TYPE_Q4_0: grctx.compute_tiled = get_rows_tiled_q4_0_f16;          break;
                    case HTP_TYPE_Q4_K: grctx.compute_tiled = get_rows_tiled_q4_k_f16;          break;
                    case HTP_TYPE_Q5_K: grctx.compute_tiled = get_rows_tiled_q5_k_f16;          break;
                    case HTP_TYPE_Q6_K: grctx.compute_tiled = get_rows_tiled_q6_k_f16;          break;
                    default:            return HTP_STATUS_NO_SUPPORT;
                }
            } else {
                switch (octx->src[0]->type) {
                    case HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_row_tiled_f16_f32;  break;
                    case HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q1_0_f32; break;
                    case HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q2_0_f32; break;
                    case HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q8_0_f32; break;
                    case HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q4_0_f32; break;
                    case HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q4_k_f32; break;
                    case HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q5_k_f32; break;
                    case HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q6_k_f32; break;
                    default:            return HTP_STATUS_NO_SUPPORT;
                }
            }
            grctx.tile_size   = htp_tensor_get_tile_size(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_size(octx->src[0]->type);
            q_func = get_rows_thread_tiled;
            break;
        case HTP_GET_ROWS_KERNEL_FLAT:
            if (octx->dst->type == HTP_TYPE_F16) {
                switch (octx->src[0]->type) {
                    case HTP_TYPE_F32:  grctx.compute_flat = get_rows_flat_f32_to_f16; break;
                    case HTP_TYPE_BF16: grctx.compute_flat = get_rows_flat_bf16_f16;   break;
                    case HTP_TYPE_Q1_0: grctx.compute_flat = get_rows_flat_q1_0_f16;   break;
                    case HTP_TYPE_Q2_0: grctx.compute_flat = get_rows_flat_q2_0_f16;   break;
                    case HTP_TYPE_Q8_0: grctx.compute_flat = get_rows_flat_q8_0_f16;   break;
                    case HTP_TYPE_Q4_0: grctx.compute_flat = get_rows_flat_q4_0_f16;   break;
                    case HTP_TYPE_Q4_K: grctx.compute_flat = get_rows_flat_q4_k_f16;   break;
                    case HTP_TYPE_Q5_K: grctx.compute_flat = get_rows_flat_q5_k_f16;   break;
                    case HTP_TYPE_Q6_K: grctx.compute_flat = get_rows_flat_q6_k_f16;   break;
                    default:            return HTP_STATUS_NO_SUPPORT;
                }
            } else {
                switch (octx->src[0]->type) {
                    case HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_row_flat_f16_f32;  break;
                    case HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_row_flat_bf16_f32; break;
                    case HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_row_flat_q1_0_f32; break;
                    case HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_row_flat_q2_0_f32; break;
                    case HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_row_flat_q8_0_f32; break;
                    case HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_row_flat_q4_0_f32; break;
                    case HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_row_flat_q4_k_f32; break;
                    case HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_row_flat_q5_k_f32; break;
                    case HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_row_flat_q6_k_f32; break;
                    default:            return HTP_STATUS_NO_SUPPORT;
                }
            }
            q_func = get_rows_thread_flat;
            break;
        default:
            return HTP_STATUS_NO_SUPPORT;
    }

    const size_t src1_bytes = hex_round_up(htp_tensor_nbytes(octx->src[1]), 128);
    void * src1_buf = (src1_bytes <= octx->ctx->ddr_spad_size) ? octx->ctx->ddr_spad_base : memalign(128, src1_bytes);
    if (!src1_buf) {
        return HTP_STATUS_INTERNAL_ERR;
    }
    if (htp_tensor_read(&octx->ctx->main_th, src1_buf, src1_bytes, octx->src[1]) == 0) {
        if (src1_buf != octx->ctx->ddr_spad_base) {
            free(src1_buf);
        }
        return HTP_STATUS_INTERNAL_ERR;
    }
    grctx.src1_data = src1_buf;

    FARF(HIGH, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-hvx %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_bytes_per_hvx * n_hvx,
         grctx.vtcm_layout.dst_bytes_per_hvx  * n_hvx,
         kparams->kernel_type, n_hvx);

    work_queue_run(octx->ctx->hvx_queue, q_func, &grctx, n_hvx);
    if (src1_buf != octx->ctx->ddr_spad_base) {
        free(src1_buf);
    }
    return HTP_STATUS_OK;
}
