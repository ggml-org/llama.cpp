#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_compute_res.h>

#include <math.h>
#include <string.h>
#include <stdatomic.h>

#include "dma-queue.h"
#include "hvx-utils.h"
#include "hvx-dump.h"
#include "hvx-arith.h"
#include "hvx-reduce.h"
#include "hvx-copy.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "matmul-ops.h"
#include "htp-vtcm.h"

typedef struct {
    dma_addr_t    dst_dma_addr;
    dma_addr_t    bias_addr;
    size_t        bias_bytes;
    dma_addr_t    act_dma_addr;
    dma_addr_t    weight;
    dma_queue *   weight_dma;
    int           m;
    int           k;
    int           n;
    int           act_stride;
    uint32_t      act_elem_size;   // 4=F32 src1, 2=F16 src1
    uint32_t      dst_elem_size;   // 4=F32 dst, 2=F16 dst
    uint32_t      bias_elem_size;  // 4=F32 bias, 2=F16 bias
    int           weight_stride;
    int           weight_type;
    int           dst_stride;
    int           dst_cols;
    uint32_t      bias_stride;
    int           tile_size;
    int           aligned_tile_size;
    int           ne02;
    int           ne03;
    int           ne12;
    int           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    int           r2;
    int           r3;
    struct fastdiv_values div_r2;
    struct fastdiv_values div_r3;
    int32_t       kernel_type;
    int           hmx_n_groups;
    struct fastdiv_values div_hmx_n_groups;
    struct fastdiv_values div_n_k_tiles;
} hmx_mm_batched_params_t;

static int htp_mm_init_context(
    struct htp_ops_context * octx,
    const struct htp_mm_kernel_params * kparams
) {
    if (!htp_ops_context_set_n_hvx(octx, (uint32_t) kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (kparams->n_hmx) {
        if (kparams->n_act_hvx <= 0 || kparams->n_act_hvx > (int32_t) octx->n_hvx) {
            return HTP_STATUS_INVAL_PARAMS;
        }
    }

    return HTP_STATUS_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
static const uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
    0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00, 0x00,
    0x00, 0x11, 0x10, 0x10, 0x10, 0x02, 0x00, 0x04, 0x00, 0x01, 0x02, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04,
    0x00, 0x00, 0x22, 0x20, 0x20, 0x20, 0x21, 0x22, 0x20, 0x24, 0x04, 0x00, 0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x11, 0x12, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08,
    0x01, 0x02, 0x00, 0x04, 0x44, 0x40, 0x40, 0x40, 0x41, 0x40, 0x40, 0x40, 0x42, 0x40, 0x44, 0x40, 0x41, 0x42, 0x48,
    0x48, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x12, 0x10, 0x10, 0x10, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00,
    0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x22, 0x20, 0x24, 0x20, 0x21, 0x22, 0x20, 0x20,
};

// IQ4_NL dequantization LUT: maps 4-bit index (0-15) to int8 kvalue
// kvalues: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
static const uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#include "iq2xxs-tables.h"

// VTCM copy of htp_iq2xxs_lut for the current op (set by hvx_mm_iq2xxs_lut_init)
static const uint8_t * iq2xxs_vtcm_lut;

static inline void iq2xxs_lut_copy(uint8_t * restrict lut) {
    for (uint32_t i = 0; i < HTP_IQ2XXS_LUT_BYTES / VLEN; i++) {
        hvx_vmem(lut + i * VLEN) = hvx_vmem(htp_iq2xxs_lut + i * VLEN);
    }
    iq2xxs_vtcm_lut = lut;
}

static inline void hvx_mm_iq2xxs_lut_init(uint8_t * restrict base, const struct htp_mm_hvx_vtcm_layout * restrict L) {
    if (L->lut_bytes == 0) {
        return;
    }
    iq2xxs_lut_copy(base + L->off_lut);
}

#define htp_matmul_tensors_preamble                                                 \
    const struct htp_tensor * restrict __attribute__((unused)) src0 = octx->src[0]; \
    const struct htp_tensor * restrict __attribute__((unused)) src1 = octx->src[1]; \
    const struct htp_tensor * restrict __attribute__((unused)) bias = octx->src[2]; \
    const struct htp_tensor * restrict __attribute__((unused))  dst = octx->dst;    \
    const struct htp_tensor * restrict __attribute__((unused))  act = src1;         \
                                                                                    \
    const uint32_t __attribute__((unused)) ne00 = src0->ne[0];                      \
    const uint32_t __attribute__((unused)) ne01 = src0->ne[1];                      \
    const uint32_t __attribute__((unused)) ne02 = src0->ne[2];                      \
    const uint32_t __attribute__((unused)) ne03 = src0->ne[3];                      \
                                                                                    \
    const uint32_t __attribute__((unused)) ne10 = src1->ne[0];                      \
    const uint32_t __attribute__((unused)) ne11 = src1->ne[1];                      \
    const uint32_t __attribute__((unused)) ne12 = src1->ne[2];                      \
    const uint32_t __attribute__((unused)) ne13 = src1->ne[3];                      \
                                                                                    \
    const uint32_t __attribute__((unused)) ne0 = dst->ne[0];                        \
    const uint32_t __attribute__((unused)) ne1 = dst->ne[1];                        \
    const uint32_t __attribute__((unused)) ne2 = dst->ne[2];                        \
    const uint32_t __attribute__((unused)) ne3 = dst->ne[3];                        \
                                                                                    \
    const uint32_t __attribute__((unused)) nb00 = src0->nb[0];                      \
    const uint32_t __attribute__((unused)) nb01 = src0->nb[1];                      \
    const uint32_t __attribute__((unused)) nb02 = src0->nb[2];                      \
    const uint32_t __attribute__((unused)) nb03 = src0->nb[3];                      \
                                                                                    \
    const uint32_t __attribute__((unused)) nb10 = src1->nb[0];                      \
    const uint32_t __attribute__((unused)) nb11 = src1->nb[1];                      \
    const uint32_t __attribute__((unused)) nb12 = src1->nb[2];                      \
    const uint32_t __attribute__((unused)) nb13 = src1->nb[3];                      \
                                                                                    \
    const uint32_t __attribute__((unused)) nb0 = dst->nb[0];                        \
    const uint32_t __attribute__((unused)) nb1 = dst->nb[1];                        \
    const uint32_t __attribute__((unused)) nb2 = dst->nb[2];                        \
    const uint32_t __attribute__((unused)) nb3 = dst->nb[3];

#define htp_matmul_preamble                                                          \
    struct htp_mm_context * mmctx                      = data;                       \
    struct htp_ops_context * octx                      = mmctx->octx;                \
    dma_queue * __attribute__((unused)) dma_q          = th->dma_q;                  \
    uint32_t __attribute__((unused)) src0_nrows_per_hvx = mmctx->src0_nrows_per_hvx; \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#include "hvx-mm-kernels-tiled.h"
#include "hvx-mm-kernels-float.h"
#include "hmx-mm-kernels-tiled-fp.h"
#include "hmx-mm-kernels-tiled-int.h"

struct htp_mm_context {
    const char * type;
    struct htp_ops_context * octx;
    const struct htp_tensor * act;
    const int32_t * ids_data;

    void (*vec_dot_1x1)(const uint32_t n, void * restrict s0,
         const void * restrict vx0,
         const void * restrict vy0, hvx_flat_store_1x1_fn_t store_1x1);

    void (*vec_dot_2x1)(const uint32_t n, void * restrict s0,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0, hvx_flat_store_2x1_fn_t store_2x1);

    void (*vec_dot_2x2)(const uint32_t n, void * restrict s0, void * restrict s1,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0, const void * restrict vy1, hvx_flat_store_2x1_fn_t store_2x1);

    void (*vec_dot_32x1)(const uint32_t n, void * restrict s,
         const void * restrict vx,
         const void * restrict vy, uint32_t valid_rows,
         const void * restrict sz, hvx_tiled_store_row_fn_t store_row);

    void (*vec_dot_32x2)(const uint32_t n, void * restrict s0, void * restrict s1,
         const void * restrict vx,
         const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows,
         const void * restrict sz0, const void * restrict sz1, hvx_tiled_store_row_fn_t store_row);

    hvx_tiled_store_row_fn_t store_row;
    hvx_flat_store_1x1_fn_t  flat_store_1x1;
    hvx_flat_store_2x1_fn_t  flat_store_2x1;

    // Precomputed values
    uint32_t src0_nrows_per_hvx;
    uint32_t src0_row_start;
    uint32_t src0_row_end;
    uint32_t src0_row_size_padded;
    uint32_t act_nrows;
    uint32_t cur_m_start;
    uint32_t cur_m_rows;

    struct fastdiv_values mm_div_ne12_ne1;
    struct fastdiv_values mm_div_ne1;
    struct fastdiv_values mm_div_r2;
    struct fastdiv_values mm_div_r3;
    struct fastdiv_values mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_first[HTP_MAX_NHVX];
    uint32_t          quant_ib_last[HTP_MAX_NHVX];
    uint32_t          quant_r[HTP_MAX_NHVX];
    uint32_t          quant_c[HTP_MAX_NHVX];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_rows_per_hvx;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    const uint32_t * matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows;
    uint32_t mapping_stride;

    // Dynamic VTCM pointers allocated sequentially
    uint8_t * vtcm_src0;
    uint8_t * vtcm_act;
    uint8_t * vtcm_bias;
    uint8_t * vtcm_dst;
    uint8_t * vtcm_act_raw;

    // Cached strides
    uint32_t vtcm_src0_stride;
    uint32_t vtcm_act_stride;
    uint32_t vtcm_act_raw_stride;

    // Cached per-HVX VTCM sizes
    uint32_t vtcm_src0_size_per_hvx;
    uint32_t vtcm_act_size_per_hvx;
    uint32_t vtcm_dst_size_per_hvx;
};

static void hvx_mm_2d_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    const uint32_t cur_m_start = mmctx->cur_m_start;

    const uint32_t src0_start_row = mmctx->src0_row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, mmctx->src0_row_end);

    struct htp_thread_trace * tr = th->trace;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    (void) n_prefetch;

    const size_t dst_row_size  = nb1;
    const size_t act_stride = mmctx->vtcm_act_stride;
    const uint32_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_hvx  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    const dma_addr_t src0_row = src0->data;

    const uint32_t tile_size = kparams->tile_size;
    const uint32_t aligned_tile_size = kparams->aligned_tile_size;

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    if (src0_start_row >= src0_end_row) {
        return;
    }

    dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) ct_start * tile_row_stride),
                   aligned_tile_size, tile_size, tile_size, n_k_tiles_w);

    for (uint32_t ct = ct_start; ct < ct_end; ct++) {
        const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

        int valid_rows = (int)ne0 - (int)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));
        const size_t ct_dst_off  = (size_t) ct * 32 * dst_elem_size;
        const size_t row_bytes   = (size_t) valid_rows * dst_elem_size;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        for (uint32_t m0 = 0; m0 < act_nrows; m0 += 32) {
            const uint32_t m_cnt = MIN(32u, act_nrows - m0);
            uint32_t r = 0;
            for (; r + 1 < m_cnt; r += 2) {
                const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (m0 + r + 0) * act_stride);
                const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (m0 + r + 1) * act_stride);
                void * dst_ptr0 = vtcm_dst_ptr + (size_t) (r + 0) * 128;
                void * dst_ptr1 = vtcm_dst_ptr + (size_t) (r + 1) * 128;
                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, NULL, NULL, mmctx->store_row);
            }
            for (; r < m_cnt; ++r) {
                const uint8_t * restrict act_col = (const uint8_t *) (act_data + (m0 + r) * act_stride);
                void * dst_ptr = vtcm_dst_ptr + (size_t) r * 128;
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL, mmctx->store_row);
            }
            const dma_addr_t dst_addr = dst->data + (dma_addr_t) (cur_m_start + m0) * dst_row_size + ct_dst_off;
            dma_queue_push(dma_q, dma_make_data(dst_addr, vtcm_dst_ptr), dst_row_size, 128, row_bytes, m_cnt);
            dma_queue_pop(dma_q);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        if (ct + 1 < ct_end) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) (ct + 1) * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }
    }
}

static void hvx_mv_2d_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const uint32_t src0_start_row = mmctx->src0_row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, mmctx->src0_row_end);

    struct htp_thread_trace * tr = th->trace;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    const uint32_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const uint32_t bias_elem_size = (bias && bias->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_hvx * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * act_data = mmctx->vtcm_act;

    float * tmp = (float *) vtcm_dst_ptr;

    const dma_addr_t src0_row = src0->data;

    const uint8_t * restrict act_col = (const uint8_t *) act_data;

    const uint32_t tile_size = kparams->tile_size;
    const uint32_t aligned_tile_size = kparams->aligned_tile_size;

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t push_ct = ct_start;
    if (src0_start_row < src0_end_row) {
        if (bias) {
            uint8_t * vtcm_bias_ptr = mmctx->vtcm_bias + (size_t) src0_start_row * bias_elem_size;
            const dma_addr_t bias_addr = bias->data + (size_t) src0_start_row * bias_elem_size;
            int slice_size = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;
            if (slice_size > 0) {
                const size_t slice_bytes = (size_t) slice_size * bias_elem_size;
                dma_queue_push(dma_q, dma_make_data(vtcm_bias_ptr, bias_addr),
                               slice_bytes, slice_bytes, slice_bytes, 1);
                dma_queue_pop(dma_q);
            }
        }
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }
    }

    if (src0_start_row >= src0_end_row) {
        return;
    }

    for (uint32_t ct = ct_start; ct < ct_end; ct++) {
        const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

        float * dst_ptr = &tmp[ct * 32 - src0_start_row];
        int valid_rows = (int)ne0 - (int)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL, hvx_tiled_store_row_f32_none);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        if (push_ct < ct_end) {
            dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            push_ct++;
        }
    }

    int copy_cnt = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;
    if (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
        const hvx_tiled_store_row_fn_t store_row = mmctx->store_row;
        const uint8_t * bias_in = bias ? (mmctx->vtcm_bias + (size_t) src0_start_row * bias_elem_size) : NULL;
        if (bias_in) {
            for (int c = 0; c < copy_cnt; c += 32) {
                const uint32_t valid_c = (uint32_t) MIN(32, copy_cnt - c);
                HVX_Vector v_sum = *(const HVX_UVector *) (tmp + c);
                store_row(vtcm_dst_ptr + (size_t) c * dst_elem_size, valid_c, v_sum, bias_in + (size_t) c * bias_elem_size);
            }
        } else if (dst_elem_size != sizeof(float)) {
            for (int c = 0; c < copy_cnt; c += 32) {
                const uint32_t valid_c = (uint32_t) MIN(32, copy_cnt - c);
                HVX_Vector v_sum = *(const HVX_UVector *) (tmp + c);
                store_row(vtcm_dst_ptr + (size_t) c * dst_elem_size, valid_c, v_sum, NULL);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
        dma_sync_write(dma_q, dst->data + (size_t) src0_start_row * dst_elem_size,
                       vtcm_dst_ptr, (size_t) copy_cnt * dst_elem_size);
    }
}

static void hvx_mm_nx_2d_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights = kparams->n_weights;

    const struct htp_tensor * restrict act = octx->src[n_weights]; /* x */
    const uint32_t ne10 = act->ne[0];
    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    const size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * restrict vtcm_dst_ptr    = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_hvx  * ith;
    uint8_t * restrict act_data        = mmctx->vtcm_act;

    struct htp_thread_trace * tr = th->trace;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    for (uint32_t widx = 0; widx < n_weights; widx++) {
        const struct htp_tensor * restrict src_w = octx->src[widx];
        const struct htp_tensor * restrict dst   = octx->dsts[widx];
        if (!src_w || !dst) continue;
        dma_queue * dma_q = th->dma_q;

        const uint32_t tile_size = kparams->tile_size;
        const uint32_t aligned_tile_size = kparams->aligned_tile_size;
        const uint32_t ne00 = src_w->ne[0];
        const uint32_t ne01 = src_w->ne[1];
        const size_t dst_row_size = dst->nb[1];
        const uint32_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
        const hvx_tiled_store_row_fn_t store_row = resolve_hvx_tiled_store_row_fn(dst_elem_size, false, sizeof(float));
        const dma_addr_t src_w_row = src_w->data;

        const uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        uint32_t src0_start_row = 0;
        uint32_t src0_end_row   = ne01;
        if (octx->ctx->mdev.count > 1) {
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, 32,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            src0_start_row = range.start;
            src0_end_row   = range.start + range.count;
        }

        const uint32_t nrows = src0_end_row - src0_start_row;
        uint32_t src0_nrows_per_hvx = fastdiv(nrows + nth - 1, &octx->n_hvx_div);
        src0_nrows_per_hvx = hex_round_up(src0_nrows_per_hvx, 32);

        const uint32_t start_row = src0_start_row + src0_nrows_per_hvx * ith;
        const uint32_t end_row   = MIN(start_row + src0_nrows_per_hvx, src0_end_row);
        if (start_row >= end_row) continue;

        uint32_t ct_start = start_row / 32;
        uint32_t ct_end   = (end_row + 31) / 32;

        if (act_nrows == 1) {
            uint32_t push_ct = ct_start;
            for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
                dma_queue_push(dma_q, dma_make_data(vtcm_weight_ptr + d * tile_row_transfer_size_aligned,
                                src_w_row + (dma_addr_t) push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }

            for (uint32_t ct = ct_start; ct < ct_end; ct++) {
                const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;
                int valid_rows = (int)dst->ne[0] - (int)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                void * dst_ptr = vtcm_dst_ptr + (size_t) (ct - ct_start) * 32 * dst_elem_size;
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_data, valid_rows, NULL, store_row);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                if (push_ct < ct_end) {
                    dma_queue_push(dma_q, dma_make_data(w_tile, src_w_row + (dma_addr_t) push_ct * tile_row_stride),
                                    aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                    push_ct++;
                }
            }
            const size_t copy_bytes = (size_t) (MIN(end_row, dst->ne[0]) - start_row) * dst_elem_size;
            dma_sync_write(dma_q, dst->data + (size_t) start_row * dst_elem_size, vtcm_dst_ptr, copy_bytes);
        } else {
            dma_queue_push(dma_q, dma_make_data(vtcm_weight_ptr, src_w_row + (dma_addr_t) ct_start * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);

            for (uint32_t ct = ct_start; ct < ct_end; ct++) {
                const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;
                int valid_rows = (int)dst->ne[0] - (int)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));
                const size_t ct_dst_off = (size_t) ct * 32 * dst_elem_size;
                const size_t row_bytes  = (size_t) valid_rows * dst_elem_size;

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                for (uint32_t m0 = 0; m0 < act_nrows; m0 += 32) {
                    const uint32_t m_cnt = MIN(32u, act_nrows - m0);
                    uint32_t r = 0;
                    for (; r + 1 < m_cnt; r += 2) {
                        const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (m0 + r + 0) * act_stride);
                        const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (m0 + r + 1) * act_stride);
                        void * dst_ptr0 = vtcm_dst_ptr + (size_t) (r + 0) * 128;
                        void * dst_ptr1 = vtcm_dst_ptr + (size_t) (r + 1) * 128;
                        mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, NULL, NULL, store_row);
                    }
                    for (; r < m_cnt; ++r) {
                        const uint8_t * restrict act_col = (const uint8_t *) (act_data + (m0 + r) * act_stride);
                        void * dst_ptr = vtcm_dst_ptr + (size_t) r * 128;
                        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL, store_row);
                    }
                    const dma_addr_t dst_addr = dst->data + (dma_addr_t) m0 * dst_row_size + ct_dst_off;
                    dma_queue_push(dma_q, dma_make_data(dst_addr, vtcm_dst_ptr), dst_row_size, 128, row_bytes, m_cnt);
                    dma_queue_pop(dma_q);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                if (ct + 1 < ct_end) {
                    dma_queue_push(dma_q, dma_make_data(vtcm_weight_ptr, src_w_row + (dma_addr_t) (ct + 1) * tile_row_stride),
                                   aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                }
            }
        }
    }
}

static void hvx_mm_transfer_act_dma(
    struct htp_ops_context * octx,
    const struct htp_mm_kernel_params * kparams,
    const struct htp_tensor * act,
    uint8_t * dst_base,
    size_t dst_row_size,
    uint32_t m_start,
    uint32_t m_rows
) {
    if (m_rows == 0) {
        return;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    const uint32_t ne0 = act->ne[0];
    const size_t elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const size_t row_bytes = ne0 * elem_size;
    const size_t act_nb1 = act->nb[1];
    const dma_addr_t act_base = act->data;

    const bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    if (is_contiguous) {
        const dma_addr_t src_addr = act_base + m_start * act_nb1;
        dma_queue_push(dma_q, dma_make_data(dst_base, src_addr),
                       dst_row_size, act_nb1, row_bytes, m_rows);
        dma_queue_pop(dma_q);
    } else {
        const uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        for (uint32_t ir = 0; ir < m_rows; ++ir) {
            const uint32_t ir1 = m_start + ir;
            const uint32_t i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
            const uint32_t rem = ir1 - i13 * ne12_ne1;
            const uint32_t i12 = fastdiv(rem, &kparams->div_ne1);
            const uint32_t i11 = rem - i12 * act->ne[1];
            const dma_addr_t row_src = act_base + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * row_dst = dst_base + ir * dst_row_size;
            dma_queue_push(dma_q, dma_make_data(row_dst, row_src),
                           dst_row_size, act_nb1, row_bytes, 1);
            dma_queue_pop(dma_q);
        }
    }

    if (dst_row_size > row_bytes) {
        struct htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
        const uint32_t pad_elems = (dst_row_size - row_bytes) / elem_size;
        if (elem_size == sizeof(float)) {
            for (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f32_u(dst_base + ir * dst_row_size + row_bytes, 0.0f, pad_elems);
            }
        } else {
            for (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f16_u(dst_base + ir * dst_row_size + row_bytes, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
    }
}

#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_expr)                                        \
static void name(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {               \
    (void) nth;                                                                                            \
    struct htp_mm_context * mmctx = data;                                                                  \
    const struct htp_tensor * src = mmctx->act;                                                            \
    const uint32_t ne0 = src->ne[0];                                                                       \
    const uint32_t nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                       \
    const uint32_t nrows_per_hvx = mmctx->n_quant_rows_per_hvx;                                            \
                                                                                                           \
    const uint32_t ir_first = nrows_per_hvx * ith;                                                         \
    if (ir_first >= nrows) {                                                                               \
        return;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    struct htp_thread_trace * tr = th->trace;                                                              \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * restrict dst = mmctx->vtcm_act;                                                              \
    const uint32_t ir_last = MIN(ir_first + nrows_per_hvx, nrows);                                         \
    const size_t raw_row_size = mmctx->vtcm_act_raw_stride;                                                \
    const size_t dst_row_size = (dst_row_size_expr);                                                       \
                                                                                                           \
    const uint8_t * restrict src_data = (const uint8_t *) mmctx->vtcm_act_raw + (raw_row_size * ir_first); \
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                             \
    kernel_fn(src_data, dst_data, NULL, ne0, ir_last - ir_first, raw_row_size, dst_row_size);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled,     "quantize-f32-q8_0_tiled",     quantize_f32_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled,     "quantize-f32-q8_1_tiled",     quantize_f32_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f16_q8_0_tiled,     "quantize-f16-q8_0_tiled",     quantize_f16_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f16_q8_1_tiled,     "quantize-f16-q8_1_tiled",     quantize_f16_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f16_q8_1_s16_tiled, "quantize-f16-q8_1_s16_tiled", quantize_f16_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_f16_tiled,      "quantize-f32-f16_tiled",      quantize_f32_f16_tiled_kernel,      htp_mm_f16_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f16_f16_tiled,      "quantize-f16-f16_tiled",      quantize_f16_f16_tiled_kernel,      htp_mm_f16_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f16_f32,            "quantize-f16-f32",            quantize_f16_f32_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,            "quantize-f32-f16",            quantize_f32_f16_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,            "quantize-f16-f16",            quantize_f16_f16_kernel,            mmctx->vtcm_act_stride)

static void quantize_f32_q8_0_tiled_block(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_block_kernel(
        (const float *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        NULL,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f32_q8_1_tiled_block(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_block_kernel(
        (const float *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        NULL,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f32_q8_1_s16_tiled_block(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_block_kernel(
        (const float *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        NULL,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f32_f16_tiled_block(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_block_kernel(
        (const float *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        NULL,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f16_f16_tiled_block(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_block_kernel(
        (const __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        NULL,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f16_q8_0_tiled_block(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f16_q8_0_tiled_block_kernel(
        (const __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        NULL,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f16_q8_1_tiled_block(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f16_q8_1_tiled_block_kernel(
        (const __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        NULL,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f16_q8_1_s16_tiled_block(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f16_q8_1_s16_tiled_block_kernel(
        (const __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        NULL,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
static inline work_queue_func_t htp_mm_act_quant_row_func(int weight_type, int act_type) {
    const bool act_f16 = (act_type == HTP_TYPE_F16);
    if (weight_type == HTP_TYPE_F16) {
        return act_f16 ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    if (weight_type == HTP_TYPE_Q2_K) {
        return act_f16 ? quantize_f16_q8_1_s16_tiled : quantize_f32_q8_1_s16_tiled;
    }
    if (htp_mm_weight_has_offset(weight_type)) {
        return act_f16 ? quantize_f16_q8_1_tiled : quantize_f32_q8_1_tiled;
    }
    return act_f16 ? quantize_f16_q8_0_tiled : quantize_f32_q8_0_tiled;
}

static inline work_queue_func_t htp_mm_act_quant_block_func(int weight_type, int act_type) {
    const bool act_f16 = (act_type == HTP_TYPE_F16);
    if (weight_type == HTP_TYPE_F16) {
        return act_f16 ? quantize_f16_f16_tiled_block : quantize_f32_f16_tiled_block;
    }
    if (weight_type == HTP_TYPE_Q2_K) {
        return act_f16 ? quantize_f16_q8_1_s16_tiled_block : quantize_f32_q8_1_s16_tiled_block;
    }
    if (htp_mm_weight_has_offset(weight_type)) {
        return act_f16 ? quantize_f16_q8_1_tiled_block : quantize_f32_q8_1_tiled_block;
    }
    return act_f16 ? quantize_f16_q8_0_tiled_block : quantize_f32_q8_0_tiled_block;
}

static void hvx_mm_4d_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    const uint32_t cur_m_start = mmctx->cur_m_start;

    const uint32_t src0_start_row = mmctx->src0_row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, mmctx->src0_row_end);

    struct htp_thread_trace * tr = th->trace;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    (void) n_prefetch;

    const size_t dst_row_size = nb1;
    const size_t act_stride = mmctx->vtcm_act_stride;
    const uint32_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    const uint32_t tile_size = kparams->tile_size;
    const uint32_t aligned_tile_size = kparams->aligned_tile_size;

    const uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    const uint32_t src0_slice_stride = ((ne01 + 31) / 32) * tile_row_stride;

    const uint32_t ct_start = src0_start_row / 32;
    const uint32_t ct_end   = (src0_end_row + 31) / 32;

    if (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        return;
    }

    const uint32_t total_batches = ne12 * ne13;
    const uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    while (b_grp_start < b_end) {
        const uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        const uint32_t b2 = b_grp_start - b3 * ne12;
        const uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        const uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        while (b_grp_end < b_end) {
            const uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            const uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            const uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            const uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            if (cur_i02 != i02 || cur_i03 != i03) {
                break;
            }
            b_grp_end++;
        }

        const uint32_t slice_idx = i03 * ne02 + i02;
        const dma_addr_t src0_slice = src0->data + (dma_addr_t) slice_idx * src0_slice_stride;

        uint8_t * restrict vtcm_dst_ptr = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_hvx * ith;

        dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_slice + (dma_addr_t) ct_start * tile_row_stride),
                       aligned_tile_size, tile_size, tile_size, n_k_tiles_w);

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

            int valid_rows = (int)ne0 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));
            const size_t ct_dst_off = (size_t) ct * 32 * dst_elem_size;
            const size_t row_bytes  = (size_t) valid_rows * dst_elem_size;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            for (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                const uint32_t b_m_start = b * ne11;
                const uint32_t m_first = MAX(cur_m_start, b_m_start);
                const uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                if (m_first >= m_last) continue;

                const uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                const uint32_t cur_b2 = b - cur_b3 * ne12;
                const dma_addr_t dst_batch_base = dst->data + (dma_addr_t) cur_b2 * nb2 + (dma_addr_t) cur_b3 * nb3;

                const uint32_t chunk_m_offset = m_first - cur_m_start;
                const uint32_t dst_m_offset   = m_first - b_m_start;
                const uint32_t batch_nrows    = m_last - m_first;

                for (uint32_t m0 = 0; m0 < batch_nrows; m0 += 32) {
                    const uint32_t m_cnt = MIN(32u, batch_nrows - m0);
                    uint32_t r = 0;
                    for (; r + 1 < m_cnt; r += 2) {
                        const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (chunk_m_offset + m0 + r + 0) * act_stride);
                        const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (chunk_m_offset + m0 + r + 1) * act_stride);
                        void * dst_ptr0 = vtcm_dst_ptr + (size_t) (r + 0) * 128;
                        void * dst_ptr1 = vtcm_dst_ptr + (size_t) (r + 1) * 128;
                        mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, NULL, NULL, mmctx->store_row);
                    }
                    for (; r < m_cnt; ++r) {
                        const uint8_t * restrict act_col = (const uint8_t *) (act_data + (chunk_m_offset + m0 + r) * act_stride);
                        void * dst_ptr = vtcm_dst_ptr + (size_t) r * 128;
                        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL, mmctx->store_row);
                    }
                    const dma_addr_t dst_addr = dst_batch_base + (dma_addr_t) (dst_m_offset + m0) * dst_row_size + ct_dst_off;
                    dma_queue_push(dma_q, dma_make_data(dst_addr, vtcm_dst_ptr), dst_row_size, 128, row_bytes, m_cnt);
                    dma_queue_pop(dma_q);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (ct + 1 < ct_end) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_slice + (dma_addr_t) (ct + 1) * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }
        }
        b_grp_start = b_grp_end;
    }
}

static void hvx_mm_2d_flat(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    (void) n_prefetch;

    const uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;
    const uint32_t cur_m_start = mmctx->cur_m_start;

    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_hvx, mmctx->src0_row_end);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    struct htp_thread_trace * tr = th->trace;

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const uint32_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t act_stride = mmctx->vtcm_act_stride;

    // Per-HVX VTCMs for all tensors
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_hvx  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;

    const dma_addr_t src0_row = src0->data;


    if (src0_start_row >= src0_end_row) {
        return;
    }

    if (src0_start_row < src0_end_row_x2) {
        dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) src0_start_row * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 2);
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
        const size_t ir0_dst_off = (size_t) ir0 * dst_elem_size;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        for (uint32_t m0 = 0; m0 < act_nrows; m0 += 32) {
            const uint32_t m_cnt = MIN(32u, act_nrows - m0);
            uint32_t r = 0;
            for (; r + 1 < m_cnt; r += 2) {
                const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (m0 + r + 0) * act_stride);
                const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (m0 + r + 1) * act_stride);
                void * dst_ptr0 = vtcm_dst_ptr + (size_t) (r + 0) * 16;
                void * dst_ptr1 = vtcm_dst_ptr + (size_t) (r + 1) * 16;
                mmctx->vec_dot_2x2(ne00, dst_ptr0, dst_ptr1, ss0, ss0 + src0_stride, act_col0, act_col1, mmctx->flat_store_2x1);
            }
            for (; r < m_cnt; ++r) {
                const uint8_t * restrict act_col = (const uint8_t *) (act_data + (m0 + r) * act_stride);
                void * dst_ptr = vtcm_dst_ptr + (size_t) r * 16;
                mmctx->vec_dot_2x1(ne00, dst_ptr, ss0, ss0 + src0_stride, act_col, mmctx->flat_store_2x1);
            }
            const dma_addr_t dst_addr = dst->data + (dma_addr_t) (cur_m_start + m0) * dst_row_size + ir0_dst_off;
            dma_queue_push(dma_q, dma_make_data(dst_addr, vtcm_dst_ptr), dst_row_size, 16, 2u * dst_elem_size, m_cnt);
            dma_queue_pop(dma_q);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        if (ir0 + 2 < src0_end_row_x2) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) (ir0 + 2) * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        const uint32_t ir0 = src0_end_row_x2;
        const size_t ir0_dst_off = (size_t) ir0 * dst_elem_size;
        dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        for (uint32_t m0 = 0; m0 < act_nrows; m0 += 32) {
            const uint32_t m_cnt = MIN(32u, act_nrows - m0);
            for (uint32_t r = 0; r < m_cnt; ++r) {
                const uint8_t * restrict act_col = (const uint8_t *) (act_data + (m0 + r) * act_stride);
                void * dst_ptr = vtcm_dst_ptr + (size_t) r * 16;
                mmctx->vec_dot_1x1(ne00, dst_ptr, ss0, act_col, mmctx->flat_store_1x1);
            }
            const dma_addr_t dst_addr = dst->data + (dma_addr_t) (cur_m_start + m0) * dst_row_size + ir0_dst_off;
            dma_queue_push(dma_q, dma_make_data(dst_addr, vtcm_dst_ptr), dst_row_size, 16, dst_elem_size, m_cnt);
            dma_queue_pop(dma_q);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
}

static void hvx_mv_2d_flat(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const uint32_t src0_start_row = mmctx->src0_row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, mmctx->src0_row_end);

    struct htp_thread_trace * tr = th->trace;

    const size_t src0_row_size = nb01;
    const uint32_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const uint32_t bias_elem_size = (bias && bias->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    const size_t src0_stride = mmctx->vtcm_src0_stride;

    // Per-HVX VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_hvx  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * act_data      = mmctx->vtcm_act;

    float * tmp = (float *) vtcm_dst_ptr;

    const dma_addr_t src0_row = src0->data;
    const uint8_t * restrict act_col = (const uint8_t *) act_data;

    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    if (src0_start_row < src0_end_row) {
        if (bias) {
            uint8_t * vtcm_bias_ptr = mmctx->vtcm_bias + (size_t) src0_start_row * bias_elem_size;
            const dma_addr_t bias_addr = bias->data + (size_t) src0_start_row * bias_elem_size;
            int slice_size = (int)src0_end_row - (int)src0_start_row;
            if (slice_size > 0) {
                const size_t slice_bytes = (size_t) slice_size * bias_elem_size;
                dma_queue_push(dma_q, dma_make_data(vtcm_bias_ptr, bias_addr),
                               slice_bytes, slice_bytes, slice_bytes, 1);
                dma_queue_pop(dma_q);
            }
        }
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint32_t is0 = (ir0 - src0_start_row);
            if (is0 >= n_prefetch) {
                break;
            }
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }


    if (src0_start_row >= src0_end_row) {
        return;
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, act_col, hvx_flat_store_2x1_f32);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        const uint32_t pr0 = (ir0 + n_prefetch);
        const uint32_t is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        const uint32_t ir0 = src0_end_row_x2;
        const uint32_t is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, act_col, hvx_flat_store_1x1_f32);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    int copy_cnt = src0_end_row - src0_start_row;
    if (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        const hvx_tiled_store_row_fn_t store_row = mmctx->store_row;
        const uint8_t * bias_in = bias ? (mmctx->vtcm_bias + (size_t) src0_start_row * bias_elem_size) : NULL;
        if (bias_in) {
            for (int c = 0; c < copy_cnt; c += 32) {
                const uint32_t valid_c = (uint32_t) MIN(32, copy_cnt - c);
                HVX_Vector v_sum = *(const HVX_UVector *) (tmp + c);
                store_row(vtcm_dst_ptr + (size_t) c * dst_elem_size, valid_c, v_sum, bias_in + (size_t) c * bias_elem_size);
            }
        } else if (dst_elem_size != sizeof(float)) {
            for (int c = 0; c < copy_cnt; c += 32) {
                const uint32_t valid_c = (uint32_t) MIN(32, copy_cnt - c);
                HVX_Vector v_sum = *(const HVX_UVector *) (tmp + c);
                store_row(vtcm_dst_ptr + (size_t) c * dst_elem_size, valid_c, v_sum, NULL);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        dma_sync_write(dma_q, dst->data + (size_t) src0_start_row * dst_elem_size,
                       vtcm_dst_ptr, (size_t) copy_cnt * dst_elem_size);
    }
}

static void hvx_mm_4d_flat(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    (void) n_prefetch;

    const uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    const uint32_t cur_m_start = mmctx->cur_m_start;

    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_hvx, mmctx->src0_row_end);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    struct htp_thread_trace * tr = th->trace;

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t act_stride  = mmctx->vtcm_act_stride;
    const uint32_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;


    if (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        return;
    }

    const uint32_t total_batches = ne12 * ne13;
    const uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    while (b_grp_start < b_end) {
        const uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        const uint32_t b2 = b_grp_start - b3 * ne12;
        const uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        const uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        while (b_grp_end < b_end) {
            const uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            const uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            const uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            const uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            if (cur_i02 != i02 || cur_i03 != i03) {
                break;
            }
            b_grp_end++;
        }

        const dma_addr_t src0_row = src0->data + ((dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03);
        uint8_t * restrict vtcm_dst_ptr = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_hvx * ith;

        if (src0_start_row < src0_end_row_x2) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) src0_start_row * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }

        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
            const size_t ir0_dst_off = (size_t) ir0 * dst_elem_size;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            for (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                const uint32_t b_m_start = b * ne11;
                const uint32_t m_first = MAX(cur_m_start, b_m_start);
                const uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                if (m_first >= m_last) continue;

                const uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                const uint32_t cur_b2 = b - cur_b3 * ne12;
                const dma_addr_t dst_batch_base = dst->data + (dma_addr_t) cur_b2 * nb2 + (dma_addr_t) cur_b3 * nb3;

                const uint32_t chunk_m_offset = m_first - cur_m_start;
                const uint32_t dst_m_offset   = m_first - b_m_start;
                const uint32_t batch_nrows    = m_last - m_first;

                for (uint32_t m0 = 0; m0 < batch_nrows; m0 += 32) {
                    const uint32_t m_cnt = MIN(32u, batch_nrows - m0);
                    uint32_t r = 0;
                    for (; r + 1 < m_cnt; r += 2) {
                        const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (chunk_m_offset + m0 + r + 0) * act_stride);
                        const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (chunk_m_offset + m0 + r + 1) * act_stride);
                        void * dst_ptr0 = vtcm_dst_ptr + (size_t) (r + 0) * 16;
                        void * dst_ptr1 = vtcm_dst_ptr + (size_t) (r + 1) * 16;
                        mmctx->vec_dot_2x2(ne00, dst_ptr0, dst_ptr1, ss0, ss0 + src0_stride, act_col0, act_col1, mmctx->flat_store_2x1);
                    }
                    for (; r < m_cnt; ++r) {
                        const uint8_t * restrict act_col = (const uint8_t *) (act_data + (chunk_m_offset + m0 + r) * act_stride);
                        void * dst_ptr = vtcm_dst_ptr + (size_t) r * 16;
                        mmctx->vec_dot_2x1(ne00, dst_ptr, ss0, ss0 + src0_stride, act_col, mmctx->flat_store_2x1);
                    }
                    const dma_addr_t dst_addr = dst_batch_base + (dma_addr_t) (dst_m_offset + m0) * dst_row_size + ir0_dst_off;
                    dma_queue_push(dma_q, dma_make_data(dst_addr, vtcm_dst_ptr), dst_row_size, 16, 2u * dst_elem_size, m_cnt);
                    dma_queue_pop(dma_q);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            if (ir0 + 2 < src0_end_row_x2) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) (ir0 + 2) * src0_row_size),
                               src0_stride, src0_row_size, src0_row_size, 2);
            }
        }

        if (src0_end_row != src0_end_row_x2) {
            const uint32_t ir0 = src0_end_row_x2;
            const size_t ir0_dst_off = (size_t) ir0 * dst_elem_size;
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 1);
            const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            for (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                const uint32_t b_m_start = b * ne11;
                const uint32_t m_first = MAX(cur_m_start, b_m_start);
                const uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                if (m_first >= m_last) continue;

                const uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                const uint32_t cur_b2 = b - cur_b3 * ne12;
                const dma_addr_t dst_batch_base = dst->data + (dma_addr_t) cur_b2 * nb2 + (dma_addr_t) cur_b3 * nb3;

                const uint32_t chunk_m_offset = m_first - cur_m_start;
                const uint32_t dst_m_offset   = m_first - b_m_start;
                const uint32_t batch_nrows    = m_last - m_first;

                for (uint32_t m0 = 0; m0 < batch_nrows; m0 += 32) {
                    const uint32_t m_cnt = MIN(32u, batch_nrows - m0);
                    for (uint32_t r = 0; r < m_cnt; ++r) {
                        const uint8_t * restrict act_col = (const uint8_t *) (act_data + (chunk_m_offset + m0 + r) * act_stride);
                        void * dst_ptr = vtcm_dst_ptr + (size_t) r * 16;
                        mmctx->vec_dot_1x1(ne00, dst_ptr, ss0, act_col, mmctx->flat_store_1x1);
                    }
                    const dma_addr_t dst_addr = dst_batch_base + (dma_addr_t) (dst_m_offset + m0) * dst_row_size + ir0_dst_off;
                    dma_queue_push(dma_q, dma_make_data(dst_addr, vtcm_dst_ptr), dst_row_size, 16, dst_elem_size, m_cnt);
                    dma_queue_pop(dma_q);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_start = b_grp_end;
    }
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * mmctx->mapping_stride + (i1)]

static void hvx_mm_id_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_hvx, mmctx->src0_row_end);


    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    (void) n_prefetch;

    const uint32_t n_as  = ne02;        // n_expert

    const uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;

    const uint32_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    const size_t act_stride = mmctx->vtcm_act_stride;

    // Per-HVX VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_hvx  * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) {
            continue;
        }

        const dma_addr_t src0_row = src0->data + (dma_addr_t) cur_a * nb02;

        const uint32_t tile_size = kparams->tile_size;
        const uint32_t aligned_tile_size = kparams->aligned_tile_size;
        const uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) ct_start * tile_row_stride),
                       aligned_tile_size, tile_size, tile_size, n_k_tiles_w);

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

            int valid_rows = (int)ne0 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));
            const size_t ct_dst_off = (size_t) ct * 32 * dst_elem_size;
            const size_t row_bytes  = (size_t) valid_rows * dst_elem_size;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            for (uint32_t cid0 = 0; cid0 < (uint32_t) cne1; cid0 += 32) {
                const uint32_t cnt = MIN(32u, (uint32_t) cne1 - cid0);
                for (uint32_t r = 0; r < cnt; ++r) {
                    struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid0 + r);
                    const int               rm1         = row_mapping.i1;
                    const int               rm2         = row_mapping.i2;
                    const uint32_t          ir1         = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);
                    const uint8_t * restrict act_col    = (const uint8_t *) (act_data + (ir1 + rm2 * ne11) * act_stride);
                    mmctx->vec_dot_32x1(ne10, vtcm_dst_ptr + (size_t) r * 128, w_tile, act_col, valid_rows, NULL, mmctx->store_row);
                }
                for (uint32_t r = 0; r < cnt; ++r) {
                    struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid0 + r);
                    const dma_addr_t dst_addr = dst->data + (dma_addr_t) row_mapping.i1 * nb1 + (dma_addr_t) row_mapping.i2 * nb2 + ct_dst_off;
                    dma_queue_push(dma_q, dma_make_data(dst_addr, vtcm_dst_ptr + (size_t) r * 128), row_bytes, row_bytes, row_bytes, 1);
                }
                dma_queue_flush(dma_q);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (ct + 1 < ct_end) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) (ct + 1) * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }
        }
    }
}

static void hvx_mv_id_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_hvx, mmctx->src0_row_end);


    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    assert(ne13 % ne03 == 0);

    const uint32_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    const uint32_t n_aids = ids->ne[0];   // num activated experts
    const uint32_t n_ids  = ne02;         // num experts
    (void) n_ids;

    // Per-HVX VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_hvx  * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    const uint32_t copy_cnt = MIN(src0_end_row, ne0) - src0_start_row;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        const int32_t eid = mmctx->ids_data[ie1];
        if (eid < 0) {
            continue;
        }
        assert(eid >= 0 && eid < (int32_t) n_ids);

        const dma_addr_t src0_row = src0->data + (dma_addr_t) eid * nb02;
        const uint8_t * restrict act_col = (const uint8_t *) act_data;
        const dma_addr_t dst_row = dst->data + (dma_addr_t) ie1 * nb1;

        const uint32_t tile_size = kparams->tile_size;
        const uint32_t aligned_tile_size = kparams->aligned_tile_size;
        const uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + (dma_addr_t) push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

            int valid_rows = (int)ne0 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, vtcm_dst_ptr + (size_t) (ct - ct_start) * 32 * dst_elem_size, w_tile, act_col, valid_rows, NULL, mmctx->store_row);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (push_ct < ct_end) {
                dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + (dma_addr_t) push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                push_ct++;
            }
        }
        dma_sync_write(dma_q, dst_row + (size_t) src0_start_row * dst_elem_size,
                       vtcm_dst_ptr, (size_t) copy_cnt * dst_elem_size);
    }
}

static void hvx_mv_id_nx_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = (struct htp_mm_context *) data;
    struct htp_ops_context * octx = mmctx->octx;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights      = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];


    struct htp_thread_trace * tr = th->trace;

    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    const uint32_t n_aids = ids->ne[0];
    const uint32_t n_ids  = src0->ne[2];
    (void) n_ids;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_hvx  * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        const int32_t eid = mmctx->ids_data[ie1];
        if (eid < 0) continue;
        assert(eid < (int32_t) n_ids);

        for (uint32_t p = 0; p < n_weights; ++p) {
            const struct htp_tensor * restrict src_w = octx->src[p];
            const struct htp_tensor * restrict dst   = octx->dsts[p];
            if (!src_w || !dst) continue;
            dma_queue * dma_q = th->dma_q;

            const uint32_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
            const hvx_tiled_store_row_fn_t store_row = resolve_hvx_tiled_store_row_fn(dst_elem_size, false, sizeof(float));
            const uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            if (octx->ctx->mdev.count > 1) {
                const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, 32, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
                start_row = range.start;
                end_row   = range.start + range.count;
            }

            const uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_hvx = fastdiv(nrows + nth - 1, &octx->n_hvx_div);
            src0_nrows_per_hvx = hex_round_up(src0_nrows_per_hvx, 32);

            const uint32_t src0_start_row = start_row + src0_nrows_per_hvx * ith;
            const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, end_row);
            if (src0_start_row >= src0_end_row) continue;

            const dma_addr_t src0_row = src_w->data + (dma_addr_t) eid * src_w->nb[2];
            const uint8_t * restrict act_col = (const uint8_t *) act_data;
            const dma_addr_t dst_row = dst->data + (dma_addr_t) ie1 * dst->nb[1];

            const uint32_t tile_size = kparams->tile_size;
            const uint32_t aligned_tile_size = kparams->aligned_tile_size;
            const uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            const uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

            const uint32_t ct_start = src0_start_row / 32;
            const uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t push_ct = ct_start;
            for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + (dma_addr_t) push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }

            for (uint32_t ct = ct_start; ct < ct_end; ct++) {
                const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

                int valid_rows = (int)dst->ne[0] - (int)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], vtcm_dst_ptr + (size_t) (ct - ct_start) * 32 * dst_elem_size, w_tile, act_col, valid_rows, NULL, store_row);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                if (push_ct < ct_end) {
                    dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + (dma_addr_t) push_ct * tile_row_stride),
                                    aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                    push_ct++;
                }
            }
            const size_t copy_bytes = (size_t) (MIN(src0_end_row, dst->ne[0]) - src0_start_row) * dst_elem_size;
            dma_sync_write(dma_q, dst_row + (size_t) src0_start_row * dst_elem_size, vtcm_dst_ptr, copy_bytes);
        }
    }
}

static void hvx_mm_id_nx_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = (struct htp_mm_context *) data;
    struct htp_ops_context * octx = mmctx->octx;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights      = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];


    struct htp_thread_trace * tr = th->trace;

    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    (void) n_prefetch;

    const uint32_t n_as = src0->ne[2];

    const uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows = mmctx->matrix_rows;

    const size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_hvx * ith;
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_hvx  * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        for (uint32_t p = 0; p < n_weights; ++p) {
            const struct htp_tensor * restrict src_w = octx->src[p];
            const struct htp_tensor * restrict dst   = octx->dsts[p];
            if (!src_w || !dst) continue;
            dma_queue * dma_q = th->dma_q;

            const uint32_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
            const hvx_tiled_store_row_fn_t store_row = resolve_hvx_tiled_store_row_fn(dst_elem_size, false, sizeof(float));
            const uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            if (octx->ctx->mdev.count > 1) {
                const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, 32, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
                start_row = range.start;
                end_row   = range.start + range.count;
            }

            const uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_hvx = fastdiv(nrows + nth - 1, &octx->n_hvx_div);
            src0_nrows_per_hvx = hex_round_up(src0_nrows_per_hvx, 32);

            const uint32_t src0_start_row = start_row + src0_nrows_per_hvx * ith;
            const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, end_row);
            if (src0_start_row >= src0_end_row) continue;

            const dma_addr_t src0_row = src_w->data + (dma_addr_t) cur_a * src_w->nb[2];

            const uint32_t tile_size = kparams->tile_size;
            const uint32_t aligned_tile_size = kparams->aligned_tile_size;
            const uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            const uint32_t tile_row_stride = n_k_tiles_w * tile_size;

            const uint32_t ct_start = src0_start_row / 32;
            const uint32_t ct_end   = (src0_end_row + 31) / 32;

            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) ct_start * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);

            for (uint32_t ct = ct_start; ct < ct_end; ct++) {
                const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

                int valid_rows = (int)dst->ne[0] - (int)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));
                const size_t ct_dst_off = (size_t) ct * 32 * dst_elem_size;
                const size_t row_bytes  = (size_t) valid_rows * dst_elem_size;

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                for (uint32_t cid0 = 0; cid0 < (uint32_t) cne1; cid0 += 32) {
                    const uint32_t cnt = MIN(32u, (uint32_t) cne1 - cid0);
                    for (uint32_t r = 0; r < cnt; ++r) {
                        struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid0 + r);
                        const int rm1 = row_mapping.i1;
                        const int rm2 = row_mapping.i2;
                        const uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                        const uint8_t * restrict act_col = (const uint8_t *) (act_data + (ir1 + rm2 * act->ne[1]) * act_stride);
                        mmctx->vec_dot_32x1(act->ne[0], vtcm_dst_ptr + (size_t) r * 128, w_tile, act_col, valid_rows, NULL, store_row);
                    }
                    for (uint32_t r = 0; r < cnt; ++r) {
                        struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid0 + r);
                        const dma_addr_t dst_addr = dst->data + (dma_addr_t) row_mapping.i1 * dst->nb[1] + (dma_addr_t) row_mapping.i2 * dst->nb[2] + ct_dst_off;
                        dma_queue_push(dma_q, dma_make_data(dst_addr, vtcm_dst_ptr + (size_t) r * 128), row_bytes, row_bytes, row_bytes, 1);
                    }
                    dma_queue_flush(dma_q);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                if (ct + 1 < ct_end) {
                    dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr, src0_row + (dma_addr_t) (ct + 1) * tile_row_stride),
                                   aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                }
            }
        }
    }
}

static int hvx_mm_init_tiled_vec_dot(struct htp_mm_context * mmctx, enum htp_data_type type) {
    switch (type) {
        case HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            return 0;
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            return 0;
        case HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            return 0;
        case HTP_TYPE_Q5_0:
            mmctx->type         = "q5_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_0_32x2;
            return 0;
        case HTP_TYPE_Q5_1:
        case HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            return 0;
        case HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            return 0;
        case HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            return 0;
        case HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            return 0;
        case HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            return 0;
        case HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            return 0;
        case HTP_TYPE_IQ2_XXS:
            mmctx->type         = "iq2xxs_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq2xxs_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq2xxs_32x2;
            return 0;
        case HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            return 0;
        case HTP_TYPE_NVFP4:
            mmctx->type         = "nvfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_nvfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_nvfp4_32x2;
            return 0;
        case HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            return 0;
        case HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            return 0;
        default:
            return -1;
    }
}

static int hvx_mm_matmul(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = ne01;
    const uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_row_start = 0;
    uint32_t src0_row_end   = src0_nrows;

    const size_t act_elem_size  = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const size_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const size_t bias_elem_size = (bias && bias->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, 32, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        src0_row_start = range.start;
        src0_row_end   = range.start + range.count;
    }

    if (src0_row_start >= src0_row_end) {
        return HTP_STATUS_OK;
    }

    const uint32_t nrows = src0_row_end - src0_row_start;
    mmctx->src0_row_start = src0_row_start;
    mmctx->src0_row_end   = src0_row_end;

    const bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_hvx
    mmctx->src0_nrows_per_hvx = fastdiv(nrows + octx->n_hvx - 1, &octx->n_hvx_div);
    if (is_tiled) {
        mmctx->src0_nrows_per_hvx = hex_round_up(mmctx->src0_nrows_per_hvx, 32);
    } else {
        mmctx->src0_nrows_per_hvx += (mmctx->src0_nrows_per_hvx & 1); // round up to even
    }

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = (size_t) ne0 * sizeof(float);
    const size_t act_row_size  = kparams->act_row_size;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    work_queue_func_t quant_task_func;
    work_queue_func_t matmul_task_func;
    uint32_t n_quant_tasks = 1;
    const bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    if (bias && (is_batched || act_nrows > 1)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (is_batched) {
        matmul_task_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } else if (act_nrows > 1) {
        matmul_task_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } else {
        matmul_task_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    mmctx->store_row      = resolve_hvx_tiled_store_row_fn(dst_elem_size, bias != NULL && !is_batched, bias_elem_size);
    mmctx->flat_store_1x1 = resolve_hvx_flat_store_1x1_fn(dst_elem_size);
    mmctx->flat_store_2x1 = resolve_hvx_flat_store_2x1_fn(dst_elem_size);

    bool need_quant = true;

    switch (kparams->kernel_type) {
        case HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F16) ? quantize_f16_f32 : NULL;
            need_quant             = (act->type == HTP_TYPE_F16);
            n_quant_tasks          = need_quant ? MIN(act_nrows, octx->n_hvx) : 1;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            break;

        case HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F16) ? quantize_f16_f32 : NULL;
            need_quant             = (act->type == HTP_TYPE_F16);
            n_quant_tasks          = need_quant ? MIN(act_nrows, octx->n_hvx) : 1;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            break;

        case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        case HTP_MM_KERNEL_HVX_QUANT_ROW:
        default: {
            if (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                return HTP_STATUS_NO_SUPPORT;
            }

            const uint32_t qk = HTP_MM_ACT_BLOCK_K;
            const uint32_t nb = (ne10 + qk - 1) / qk;
            const uint32_t total_nb = act_nrows * nb;

            if (act_nrows < octx->n_hvx && !is_batched) {
                n_quant_tasks = MIN(total_nb, octx->n_hvx);
                quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
                for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
                    uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_last[ith]  = ib_last;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } else {
                n_quant_tasks = MIN(act_nrows, octx->n_hvx);
                quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
            }
            break;
        }
    }

    const uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    const uint32_t m_layout_rows = m_chunk;

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_rows, octx->n_hvx,
                                 dst_row_size, src0_row_size, act_elem_size, bias ? bias->nb[1] : 0, kparams->n_prefetch, false, false);

    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_act_size_per_hvx = L.act_bytes;
    } else {
        mmctx->vtcm_act_size_per_hvx = fastdiv(L.act_bytes, &octx->n_hvx_div);
    }

    mmctx->vtcm_src0_size_per_hvx = fastdiv(L.src0_bytes, &octx->n_hvx_div);
    mmctx->vtcm_dst_size_per_hvx  = fastdiv(L.dst_bytes, &octx->n_hvx_div);

    const size_t vtcm_size = L.total_bytes;

    FARF(HIGH, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, act->data, dst->data);

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
    hvx_mm_iq2xxs_lut_init(base, &L);

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_act_stride  = act_row_size;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
        mmctx->vtcm_act_raw_stride = htp_mm_act_raw_row_size(ne10, act_elem_size);
    } else if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
               ((kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                 kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) &&
                act_elem_size == sizeof(__fp16))) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * act_elem_size, 128);
    } else {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    if (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        for (uint32_t m_start = 0; m_start < act_nrows; m_start += m_chunk) {
            const uint32_t cur_m_rows = MIN(act_nrows - m_start, m_chunk);
            mmctx->cur_m_start = m_start;
            mmctx->cur_m_rows  = cur_m_rows;

            if (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);

                const uint32_t qk = HTP_MM_ACT_BLOCK_K;
                const uint32_t nb = (ne10 + qk - 1) / qk;
                const uint32_t total_nb = cur_m_rows * nb;
                uint32_t quant_tasks;
                work_queue_func_t q_func;
                if (cur_m_rows < octx->n_hvx && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW)) {
                    quant_tasks = MIN(total_nb, octx->n_hvx);
                    q_func = htp_mm_act_quant_block_func(src0->type, act->type);
                    for (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_first = (total_nb * ith) / quant_tasks;
                        uint32_t ib_last  = (total_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_first[ith] = ib_first;
                        mmctx->quant_ib_last[ith]  = ib_last;
                        mmctx->quant_r[ith]        = ib_first / nb;
                        mmctx->quant_c[ith]        = ib_first % nb;
                    }
                } else {
                    quant_tasks = MIN(cur_m_rows, octx->n_hvx);
                    q_func = quant_task_func;
                    mmctx->n_quant_rows_per_hvx = (cur_m_rows + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_queue_run(octx->ctx->hvx_queue, q_func, mmctx, quant_tasks);
            } else {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_start, cur_m_rows);
            }

            work_queue_run(octx->ctx->hvx_queue, matmul_task_func, mmctx, octx->n_hvx);
        }
    } else {
        mmctx->cur_m_start = 0;
        mmctx->cur_m_rows  = act_nrows;

        if (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_rows_per_hvx = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_queue_run(octx->ctx->hvx_queue, quant_task_func, mmctx, n_quant_tasks);
        } else {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_queue_run(octx->ctx->hvx_queue, matmul_task_func, mmctx, octx->n_hvx);
    }

    return HTP_STATUS_OK;
}

static void dequantize_tiled_worker_loop(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) data;
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    for (unsigned int task_id = i; task_id < (unsigned int) state->n_tasks; task_id += n) {
        const uint32_t start = task_id * state->n_tiles_per_task;
        if (start >= state->n_tot_tiles) {
            break;
        }
        const uint32_t end = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        state->task_fn(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

static void transfer_output_chunk_worker_fn(const htp_thread_t * th, unsigned int n, unsigned int i, void *data) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = th->trace;

    int start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        const uint32_t chunk_idx = task_id * st->n_chunks_per_task;
        if (chunk_idx >= st->n_tot_chunks) {
            continue;
        }
        const uint32_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        void *       dst  = (uint8_t *) st->dst + (size_t) chunk_idx * st->dst_stride * st->dst_elem_size;
        const void * bias = st->bias ? ((const uint8_t *) st->bias + (size_t) chunk_idx * st->bias_stride * st->bias_elem_size) : NULL;
        st->col_chunk_fn(dst, bias, st->vtcm_src, chunk_idx, chunk_size,
                         st->n_cols, st->n_cols, st->dst_stride, st->bias_stride, st->dst_cols, st->bias_elem_size);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

typedef struct {
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    float                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_bytes_per_hvx;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
    uint32_t                        act_elem_size;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

typedef struct {
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    float                         * vtcm_f32_act;
    uint32_t                        n_rows;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_col_chunks;
    struct fastdiv_values           n_hvx_div;
    size_t                          vtcm_f32_act_bytes;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
    uint32_t                        act_elem_size;
} activation_transfer_col_chunk_state_t;

static void transfer_activation_chunk_to_fp16_dma_pipelined_col_chunk(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_first,
        uint32_t c_len,
        uint8_t *act_spad,
        uint32_t act_elem_size,
        struct htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    const uint32_t R = dma_step_rows;
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    const uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    const bool   act_is_f16     = (act_elem_size == sizeof(__fp16));
    const size_t row_bytes      = (size_t) c_len * act_elem_size;
    const size_t src_row_bytes  = (size_t) k_stride * act_elem_size;
    const size_t width_bytes    = (size_t) k_chunk_valid * act_elem_size;
    const size_t src_col_offset = (size_t) c_first * act_elem_size;

    // Push step 0
    if (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_data(act_spad, act_dma_addr + src_col_offset),
                       row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
    }
    // Push step 1
    if (n_steps > 1) {
        uint32_t next_r = R * 1;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            uint8_t *next_buf = act_spad + 1 * R * row_bytes;
            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * src_row_bytes + src_col_offset),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
    for (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        uint8_t *curr_buf = act_spad;

        if (r < n_rows) {
            curr_buf = (uint8_t *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        if (act_is_f16) {
            for (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                const __fp16 *pair_buf = (const __fp16 *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_f16_to_f16_col_chunk(
                    vtcm_dst, pair_buf, pair_buf + c_len, row_idx, k_block, c_first, c_len, k_chunk_valid, r0_valid, r1_valid
                );
            }
        } else {
            for (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                const float *pair_buf = (const float *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_fp32_to_fp16_col_chunk(
                    vtcm_dst, pair_buf, pair_buf + c_len, row_idx, k_block, c_first, c_len, k_chunk_valid, r0_valid, r1_valid
                );
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * src_row_bytes + src_col_offset),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
}

static void transfer_activation_chunk_col_chunk_worker_fn(const htp_thread_t * th, unsigned int n, unsigned int i, void *data) {
    activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) data;
    struct htp_thread_trace * tr = th->trace;

    uint32_t n_blocks = st->k_block / 32;
    uint32_t b_first;
    uint32_t b_last;
    if (st->act_elem_size == sizeof(__fp16)) {
        uint32_t n_pairs = n_blocks / 2;
        b_first = fastdiv(n_pairs * i, &st->n_hvx_div) * 2;
        b_last  = fastdiv(n_pairs * (i + 1), &st->n_hvx_div) * 2;
    } else {
        b_first = fastdiv(n_blocks * i, &st->n_hvx_div);
        b_last  = fastdiv(n_blocks * (i + 1), &st->n_hvx_div);
    }
    uint32_t c_first = b_first * 32;
    uint32_t c_last  = b_last * 32;
    uint32_t c_len   = c_last - c_first;

    if (c_len == 0) {
        return;
    }

    uint32_t k_chunk_valid = 0;
    if (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_last) - c_first;
    }

    __fp16 *dst = st->dst;

    size_t scratch_bytes_per_hvx = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_hvx_div), 128);
    uint8_t *act_spad = (uint8_t *)st->vtcm_f32_act + i * scratch_bytes_per_hvx;

    transfer_activation_chunk_to_fp16_dma_pipelined_col_chunk(
        th->dma_q, dst, st->act_dma_addr, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
        c_first, c_len, act_spad, st->act_elem_size, tr, st->dma_step_rows, st->dma_step_rows_shift
    );
}

static void transfer_activation_chunk_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *act_spad,
        uint32_t act_elem_size,
        struct htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    const uint32_t R = dma_step_rows;
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    const uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    const bool act_is_f16 = (act_elem_size == sizeof(__fp16));
    const size_t row_bytes = (size_t) k_block * act_elem_size;  // staging row (DMA dst stride)
    const size_t src_row_bytes = (size_t) k_stride * act_elem_size;  // DDR row (DMA src stride)
    const size_t width_bytes = (size_t) k_valid * act_elem_size;

    // Push step 0
    if (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_data(act_spad, act_dma_addr),
                       row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
    }
    // Push step 1 (if valid)
    if (n_steps > 1) {
        uint32_t next_r = R * 1;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            uint8_t *next_buf = act_spad + 1 * R * row_bytes;
            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
    for (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        uint8_t *curr_buf = act_spad;

        if (r < n_rows) {
            curr_buf = (uint8_t *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        if (act_is_f16) {
            for (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                const __fp16 *pair_buf = (const __fp16 *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_f16_to_f16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        } else {
            for (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                const float *pair_buf = (const float *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
}

static void transfer_activation_chunk_worker_fn(const htp_thread_t * th, unsigned int n, unsigned int i, void *data) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = th->trace;

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        const dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_size;

        uint8_t *act_spad = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_hvx;
        transfer_activation_chunk_to_fp16_dma_pipelined(
            th->dma_q, dst, act_dma_addr, chunk_size, st->k_block, st->k_stride, st->k_valid, act_spad, st->act_elem_size, tr, st->dma_step_rows, st->dma_step_rows_shift
        );
    }
}

static inline void push_gathered_act_rows_dma(
        dma_queue * dma_q,
        uint8_t * slot_buf,
        dma_addr_t act_dma_addr,
        uint32_t valid_rows,
        uint32_t start_row,
        uint32_t r,
        size_t row_bytes,
        size_t valid_bytes,
        const struct mmid_row_mapping * matrix_rows,
        uint32_t cur_a,
        uint32_t mapping_stride,
        uint32_t ne11,
        const struct fastdiv_values * ne11_div,
        size_t nb11,
        size_t nb12) {
    const size_t base_idx = (size_t) cur_a * mapping_stride + start_row + r;
    if (ne11 == 1) {
        for (uint32_t s = 0; s < valid_rows; ++s) {
            const struct mmid_row_mapping mapping = matrix_rows[base_idx + s];
            const dma_addr_t src_row = act_dma_addr + (size_t) mapping.i2 * nb12;
            dma_queue_push(dma_q, dma_make_data(slot_buf + (size_t) s * row_bytes, src_row),
                           valid_bytes, valid_bytes, valid_bytes, 1);
        }
    } else {
        for (uint32_t s = 0; s < valid_rows; ++s) {
            const struct mmid_row_mapping mapping = matrix_rows[base_idx + s];
            const uint32_t i11 = fastmodulo(mapping.i1, ne11, ne11_div);
            const dma_addr_t src_row = act_dma_addr + (size_t) i11 * nb11 + (size_t) mapping.i2 * nb12;
            dma_queue_push(dma_q, dma_make_data(slot_buf + (size_t) s * row_bytes, src_row),
                           valid_bytes, valid_bytes, valid_bytes, 1);
        }
    }
}

static inline void push_scattered_dst_rows_dma(
        dma_queue * dma_q,
        dma_addr_t dst_dma_addr,
        const uint8_t * vtcm_src,
        uint32_t start_row,
        uint32_t n_rows,
        size_t vtcm_row_bytes,
        size_t copy_bytes,
        const struct mmid_row_mapping * matrix_rows,
        uint32_t cur_a,
        uint32_t mapping_stride,
        size_t dst_nb1,
        size_t dst_nb2) {
    const size_t base_idx = (size_t) cur_a * mapping_stride + start_row;
    for (uint32_t r = 0; r < n_rows; ++r) {
        const struct mmid_row_mapping mapping = matrix_rows[base_idx + r];
        const dma_addr_t dst_row = dst_dma_addr + (size_t) mapping.i1 * dst_nb1 + (size_t) mapping.i2 * dst_nb2;
        dma_queue_push(dma_q, dma_make_data(dst_row, vtcm_src + (size_t) r * vtcm_row_bytes),
                       copy_bytes, copy_bytes, copy_bytes, 1);
    }
}

static inline void sync_scattered_dst_rows_dma(
        dma_queue * dma_q,
        dma_addr_t dst_dma_addr,
        const uint8_t * vtcm_src,
        uint32_t start_row,
        uint32_t n_rows,
        size_t vtcm_row_bytes,
        size_t copy_bytes,
        const struct mmid_row_mapping * matrix_rows,
        uint32_t cur_a,
        uint32_t mapping_stride,
        size_t dst_nb1,
        size_t dst_nb2) {
    for (uint32_t r = 0; r < n_rows; r += 128u) {
        const uint32_t batch = hex_smin(n_rows - r, 128u);
        push_scattered_dst_rows_dma(dma_q, dst_dma_addr, vtcm_src + (size_t) r * vtcm_row_bytes,
                                    start_row + r, batch, vtcm_row_bytes, copy_bytes,
                                    matrix_rows, cur_a, mapping_stride, dst_nb1, dst_nb2);
        for (uint32_t s = 0; s < batch; ++s) {
            dma_queue_pop(dma_q);
        }
    }
}

typedef struct {
    const struct mmid_row_mapping * matrix_rows;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    float                         * vtcm_f32_act;
    size_t                          vtcm_f32_act_bytes_per_hvx;
    size_t                          act_elem_size;
    uint32_t                        n_rows;
    uint32_t                        mc_pad;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    struct fastdiv_values           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        start_row;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

static void transfer_activation_chunk_gathered_worker_fn(const htp_thread_t * th, unsigned int n, unsigned int i, void *data) {
    (void) n;
    activation_transfer_gathered_task_state_t * st = (activation_transfer_gathered_task_state_t *) data;
    struct htp_thread_trace * tr = th->trace;
    dma_queue * dma_q = th->dma_q;

    const uint32_t vtcm_start_row = i * st->n_chunks_per_task;
    if (vtcm_start_row >= st->mc_pad) {
        return;
    }
    const uint32_t vtcm_end_row = hex_smin(vtcm_start_row + st->n_chunks_per_task, st->mc_pad);
    const bool     act_is_f16   = (st->act_elem_size == sizeof(__fp16));
    const size_t   row_bytes    = (size_t) st->k_block * st->act_elem_size;
    const size_t   valid_bytes  = (size_t) st->k_valid * st->act_elem_size;
    const size_t   slot_bytes   = 2u * row_bytes;
    const bool     double_buf   = (st->vtcm_f32_act_bytes_per_hvx >= 2u * slot_bytes);
    uint8_t *      act_spad     = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_hvx;

    if (double_buf) {
        if (vtcm_start_row < st->n_rows) {
            const uint32_t nrows = hex_smin(st->n_rows - vtcm_start_row, 2u);
            push_gathered_act_rows_dma(dma_q, act_spad, st->act_dma_addr, nrows, st->start_row, vtcm_start_row,
                                       row_bytes, valid_bytes, st->matrix_rows, st->cur_a, st->mapping_stride,
                                       st->ne11, &st->ne11_div, st->nb11, st->nb12);
        }
        const uint32_t r1 = vtcm_start_row + 2u;
        if (r1 < vtcm_end_row && r1 < st->n_rows) {
            const uint32_t nrows = hex_smin(st->n_rows - r1, 2u);
            push_gathered_act_rows_dma(dma_q, act_spad + slot_bytes, st->act_dma_addr, nrows, st->start_row, r1,
                                       row_bytes, valid_bytes, st->matrix_rows, st->cur_a, st->mapping_stride,
                                       st->ne11, &st->ne11_div, st->nb11, st->nb12);
        }
    }

    uint32_t step = 0;
    for (uint32_t r = vtcm_start_row; r < vtcm_end_row; r += 2u, ++step) {
        uint8_t *      curr_buf   = act_spad + (double_buf ? (step & 1u) * slot_bytes : 0u);
        const uint32_t valid_rows = (r < st->n_rows) ? hex_smin(st->n_rows - r, 2u) : 0u;

        if (valid_rows > 0) {
            if (!double_buf) {
                push_gathered_act_rows_dma(dma_q, curr_buf, st->act_dma_addr, valid_rows, st->start_row, r,
                                           row_bytes, valid_bytes, st->matrix_rows, st->cur_a, st->mapping_stride,
                                           st->ne11, &st->ne11_div, st->nb11, st->nb12);
            }
            for (uint32_t s = 0; s < valid_rows; ++s) {
                dma_queue_pop(dma_q);
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        if (act_is_f16) {
            const __fp16 * pair_buf = (const __fp16 *) curr_buf;
            transfer_activation_row_pair_f16_to_f16(
                st->dst, pair_buf, pair_buf + st->k_block, r, st->k_block, st->k_valid,
                valid_rows >= 1u, valid_rows >= 2u);
        } else {
            const float * pair_buf = (const float *) curr_buf;
            transfer_activation_row_pair_fp32_to_fp16(
                st->dst, pair_buf, pair_buf + st->k_block, r, st->k_block, st->k_valid,
                valid_rows >= 1u, valid_rows >= 2u);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        if (double_buf) {
            const uint32_t next_r = r + 4u;
            if (next_r < vtcm_end_row && next_r < st->n_rows) {
                const uint32_t nrows = hex_smin(st->n_rows - next_r, 2u);
                push_gathered_act_rows_dma(dma_q, curr_buf, st->act_dma_addr, nrows, st->start_row, next_r,
                                           row_bytes, valid_bytes, st->matrix_rows, st->cur_a, st->mapping_stride,
                                           st->ne11, &st->ne11_div, st->nb11, st->nb12);
            }
        }
    }
}

// --- HMX Dispatchers & Entry Points ---

static inline tiled_dequant_task_fn_t hmx_get_dequant_worker_fn(int weight_type, bool is_tiled) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   return dequantize_tiled_weight_to_fp16_task_q4_0;
        case HTP_TYPE_IQ4_NL: return dequantize_tiled_weight_to_fp16_task_iq4_nl;
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q4_K:   return dequantize_tiled_weight_to_fp16_task_q4_1;
        case HTP_TYPE_MXFP4:  return dequantize_tiled_weight_to_fp16_task_mxfp4;
        case HTP_TYPE_NVFP4:  return dequantize_tiled_weight_to_fp16_task_nvfp4;
        case HTP_TYPE_IQ2_XXS: return dequantize_tiled_weight_to_fp16_task_iq2_xxs;
        case HTP_TYPE_Q8_0:   return dequantize_tiled_weight_to_fp16_task_q8_0;
        case HTP_TYPE_Q5_0:   return dequantize_tiled_weight_to_fp16_task_q5_0;
        case HTP_TYPE_Q5_1:
        case HTP_TYPE_Q5_K:   return dequantize_tiled_weight_to_fp16_task_q5_k;
        case HTP_TYPE_Q6_K:   return dequantize_tiled_weight_to_fp16_task_q6_k;
        case HTP_TYPE_Q3_K:   return dequantize_tiled_weight_to_fp16_task_q3_k;
        case HTP_TYPE_Q2_K:   return dequantize_tiled_weight_to_fp16_task_q2_k;
        case HTP_TYPE_Q1_0:   return dequantize_tiled_weight_to_fp16_task_q1_0;
        case HTP_TYPE_Q2_0:   return dequantize_tiled_weight_to_fp16_task_q2_0;
        case HTP_TYPE_F16:    return is_tiled ? NULL : convert_f16_weight_to_fp16_tiles_task;
        case HTP_TYPE_BF16:   return convert_bf16_weight_to_fp16_tiles_task;
        case HTP_TYPE_F32:    return quantize_f32_weight_to_fp16_tiles_task;
        default:              return NULL;
    }
}

static void dequantize_tiled_weight_chunk_to_fp16_tiles(
        struct htp_ops_context *octx, __fp16 *vtcm_dst,
        const void *weight_src_ddr,
        int n_cols, int k_block,
        size_t row_stride, int weight_type,
        int n_k_tiles, struct fastdiv_values n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_worker_fn) {

    if (!dequant_worker_fn) {
        return;
    }

    assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0);

    const int      n_hvx          = (int) octx->n_hvx;
    const size_t   n_col_tiles    = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
    const uint32_t n_weight_tiles = k_block / htp_tensor_get_tile_k(weight_type);
    const size_t   n_tot_tiles    = n_col_tiles * n_weight_tiles;
    if (n_tot_tiles == 0) {
        return;
    }

    const size_t n_tiles_per_task = (n_hvx <= 1) ? n_tot_tiles
                                  : fastdiv(n_tot_tiles + (uint32_t) n_hvx - 1, &octx->n_hvx_div);

    tiled_dequantize_state_t state;
    state.n_tasks          = hex_smin((int) n_tot_tiles, n_hvx);
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_per_task = n_tiles_per_task;
    state.dst              = vtcm_dst;
    state.src              = (const uint8_t *)weight_src_ddr;
    state.task_fn          = dequant_worker_fn;
    state.n_cols           = n_cols;
    state.k_block          = k_block;
    state.row_stride       = row_stride;
    state.weight_type      = weight_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_weight_tiles   = n_weight_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    state.tile_size         = kparams->tile_size;
    state.aligned_tile_size = kparams->aligned_tile_size;

    work_queue_run(octx->ctx->hvx_queue, dequantize_tiled_worker_loop, &state, (int) state.n_tasks);
}

typedef struct {
    hmx_fp16_output_col_chunk_fn_t col_chunk_fn;
    void *                dst;
    const __fp16 *        vtcm_src;
    const void *          bias;
    uint32_t              n_rows;
    uint32_t              n_cols;
    uint32_t              dst_stride;
    uint32_t              bias_stride;
    uint32_t              dst_cols;
    size_t                dst_elem_size;
    size_t                bias_elem_size;
    struct fastdiv_values n_hvx_div;
} output_transfer_col_chunk_state_t;

static void transfer_output_chunk_col_chunk_worker_fn(const htp_thread_t * th, unsigned int n, unsigned int i, void *data) {
    (void) n;
    output_transfer_col_chunk_state_t *st = (output_transfer_col_chunk_state_t *) data;
    struct htp_thread_trace * tr = th->trace;

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
    uint32_t b_first;
    uint32_t b_last;
    if (st->dst_elem_size == sizeof(__fp16)) {
        uint32_t n_pairs = hmx_ceil_div(n_blocks, 2);
        b_first = fastdiv(n_pairs * i, &st->n_hvx_div) * 2;
        b_last  = MIN(n_blocks, fastdiv(n_pairs * (i + 1), &st->n_hvx_div) * 2);
    } else {
        b_first = fastdiv(n_blocks * i, &st->n_hvx_div);
        b_last  = fastdiv(n_blocks * (i + 1), &st->n_hvx_div);
    }
    if (b_first >= b_last) return;
    uint32_t c_first = b_first * 32;
    uint32_t c_last  = b_last * 32;
    uint32_t c_len   = c_last - c_first;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);

    const __fp16 * vtcm_src = st->vtcm_src + b_first * HTP_MM_HMX_TILE_N_ELMS;
    const void *   bias     = st->bias ? ((const uint8_t *) st->bias + (size_t) c_first * st->bias_elem_size) : NULL;
    void *         dst      = (uint8_t *) st->dst + (size_t) c_first * st->dst_elem_size;

    int chunk_dst_cols = (int)st->dst_cols - (int)c_first;
    if (chunk_dst_cols > 0) {
        st->col_chunk_fn(dst, bias, vtcm_src, 0, st->n_rows, c_len, st->n_cols,
                         st->dst_stride, st->bias_stride, (uint32_t)chunk_dst_cols, st->bias_elem_size);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

static void transfer_output_chunk_threaded(struct htp_ops_context *octx, void *dst, const void *bias, const __fp16 *vtcm_src,
                                           int n_rows, int n_cols, int dst_stride, uint32_t bias_stride, int dst_cols,
                                           size_t dst_elem_size, size_t bias_elem_size) {
    if (n_rows <= 0 || dst_cols <= 0) {
        return;
    }

    const bool dst_aligned_nt = ((((uintptr_t) dst | ((size_t) dst_stride * dst_elem_size)) & 127u) == 0);
    const hmx_fp16_output_col_chunk_fn_t col_chunk_fn =
        resolve_fp16_output_col_chunk_fn(dst_elem_size, bias, bias_elem_size, dst_aligned_nt);

    const int      n_hvx            = (int) octx->n_hvx;
    const uint32_t n_blocks         = hmx_ceil_div((uint32_t) n_cols, 32);
    const uint32_t n_units          = (dst_elem_size == sizeof(__fp16)) ? hmx_ceil_div(n_blocks, 2) : n_blocks;
    const bool     prefer_row_chunk = ((uint32_t) n_rows >= (uint32_t) n_hvx * 32u);
    if (n_hvx > 1 && !prefer_row_chunk && n_units >= (uint32_t) n_hvx) {
        const uint32_t n_row_pairs = ((uint32_t) n_rows + 1u) >> 1;
        const uint32_t row_max     = fastdiv(n_row_pairs + (uint32_t) n_hvx - 1u, &octx->n_hvx_div);
        const uint32_t col_max     = fastdiv(n_units + (uint32_t) n_hvx - 1u, &octx->n_hvx_div);
        if ((uint64_t) n_units * row_max > (uint64_t) n_row_pairs * col_max) {
            output_transfer_col_chunk_state_t col_state;
            col_state.col_chunk_fn   = col_chunk_fn;
            col_state.dst            = dst;
            col_state.bias           = bias;
            col_state.vtcm_src       = vtcm_src;
            col_state.n_rows         = (uint32_t) n_rows;
            col_state.n_cols         = (uint32_t) n_cols;
            col_state.dst_stride     = (uint32_t) dst_stride;
            col_state.bias_stride    = bias_stride;
            col_state.dst_cols       = (uint32_t) dst_cols;
            col_state.dst_elem_size  = dst_elem_size;
            col_state.bias_elem_size = bias_elem_size;
            col_state.n_hvx_div      = octx->n_hvx_div;

            work_queue_run(octx->ctx->hvx_queue, transfer_output_chunk_col_chunk_worker_fn, &col_state, n_hvx);
            return;
        }
    }

    const uint32_t chunk_align      = prefer_row_chunk ? 32u : 2u;
    uint32_t       n_chunks_per_task = (n_hvx <= 1) ? (uint32_t) n_rows
                                     : fastdiv((uint32_t) n_rows + (uint32_t) n_hvx - 1u, &octx->n_hvx_div);
    n_chunks_per_task               = hex_align_up(n_chunks_per_task, chunk_align);
    const int      actual_hvx       = hex_smin(n_hvx, ((int) n_rows + 1) >> 1);

    output_transfer_task_state_t state;
    state.col_chunk_fn      = col_chunk_fn;
    state.n_tasks           = (uint32_t) actual_hvx;
    state.n_tot_chunks      = (uint32_t) n_rows;
    state.n_chunks_per_task = n_chunks_per_task;
    state.dst               = dst;
    state.bias              = bias;
    state.vtcm_src          = vtcm_src;
    state.n_cols            = (uint32_t) n_cols;
    state.dst_stride        = (uint32_t) dst_stride;
    state.bias_stride       = bias_stride;
    state.dst_cols          = (uint32_t) dst_cols;
    state.dst_elem_size     = dst_elem_size;
    state.bias_elem_size    = bias_elem_size;

    work_queue_run(octx->ctx->hvx_queue, transfer_output_chunk_worker_fn, &state, actual_hvx);
}

struct activation_transfer_params {
    struct htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    int                           n_rows;
    int                           k_block;
    int                           k_stride;
    int                           n_hvx;
    const struct fastdiv_values * act_hvx_div;
    const struct fastdiv_values * k_div;
    int                           k_valid;
    float *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes;
    uint32_t                      act_elem_size;  // 4=F32 src1, 2=F16 src1
};

static void transfer_activation_chunk_threaded(const struct activation_transfer_params * params) {
    struct htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    const dma_addr_t              act_dma_addr       = params->act_dma_addr;
    int                           n_rows             = params->n_rows;
    int                           k_block            = params->k_block;
    int                           k_stride           = params->k_stride;
    int                           n_hvx              = params->n_hvx;
    const struct fastdiv_values * act_hvx_div        = params->act_hvx_div;
    const struct fastdiv_values * k_div              = params->k_div;
    int                           k_valid            = params->k_valid;
    float *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes = params->vtcm_f32_act_bytes;
    // element size of the activation rows (4 = F32, 2 = F16).
    const uint32_t                act_elem_size      = params->act_elem_size ? params->act_elem_size : (uint32_t) sizeof(float);

    if (n_rows <= 0) {
        return;
    }

    const size_t n_tasks = (n_rows + 31) >> 5;
    const bool col_chunk_ok = (act_elem_size == sizeof(float))
                            ? (k_block >= 32 * n_hvx)
                            : ((k_block & 63) == 0 && (k_block >> 6) >= n_hvx);
    if (n_hvx > 1 && col_chunk_ok && n_tasks < (size_t) n_hvx) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_step_rows = 2;
        uint32_t dma_step_rows_shift = 1;
        if (vtcm_f32_act_bytes > 0) {
            size_t scratch_bytes_per_hvx = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_hvx_div), 128);
            size_t scratch_elems_per_hvx = scratch_bytes_per_hvx / act_elem_size;
            size_t dma_step_rows_max = fastdiv(scratch_elems_per_hvx / 2, k_div);
            if (dma_step_rows_max >= 4) {
                dma_step_rows = 4;
                dma_step_rows_shift = 2;
            }
        }

        activation_transfer_col_chunk_state_t col_state;
        col_state.dst = dst;
        col_state.act_dma_addr = act_dma_addr;
        col_state.n_rows = n_rows;
        col_state.k_block = k_block;
        col_state.k_stride = k_stride;
        col_state.k_valid = k_valid;
        col_state.n_col_chunks = n_hvx;
        col_state.n_hvx_div = *act_hvx_div;
        col_state.vtcm_f32_act = vtcm_f32_act;
        col_state.vtcm_f32_act_bytes = vtcm_f32_act_bytes;
        col_state.dma_step_rows = dma_step_rows;
        col_state.dma_step_rows_shift = dma_step_rows_shift;
        col_state.act_elem_size = act_elem_size;

        work_queue_run(ctx->hvx_queue, transfer_activation_chunk_col_chunk_worker_fn, &col_state, n_hvx);
        return;
    }

    assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0 && k_stride % HTP_MM_HMX_TILE_N_COLS == 0);

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_hvx == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_hvx == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
    state.n_tot_chunks       = n_tot_chunks;
    state.n_chunks_per_task  = n_chunks_per_task;
    state.dst                = dst;
    state.act_dma_addr       = act_dma_addr;
    state.k_block            = k_block;
    state.k_stride           = k_stride;
    state.k_valid            = k_valid;
    state.vtcm_f32_act       = vtcm_f32_act;
    state.act_elem_size      = act_elem_size;

    state.vtcm_f32_act_bytes_per_hvx = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_hvx_div), 128);

    uint32_t dma_step_rows = 2;
    uint32_t dma_step_rows_shift = 1;
    if (state.vtcm_f32_act_bytes_per_hvx > 0) {
        size_t scratch_elems_per_hvx = state.vtcm_f32_act_bytes_per_hvx / act_elem_size;
        size_t dma_step_rows_max = fastdiv(scratch_elems_per_hvx / 2, k_div);
        if (dma_step_rows_max >= 4) {
            dma_step_rows = 4;
            dma_step_rows_shift = 2;
        }
    }
    state.dma_step_rows      = dma_step_rows;
    state.dma_step_rows_shift = dma_step_rows_shift;

    int active_hvx = hex_smin(n_hvx, (int)state.n_tasks);
    work_queue_run(ctx->hvx_queue, transfer_activation_chunk_worker_fn, &state, active_hvx);
}

// --- INT HMX W8A8 SM Threaded Helpers ---

static void rescale_int_weight_worker_loop(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    int_weight_rescale_state_t * state = (int_weight_rescale_state_t *) data;
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    for (unsigned int task_id = i; task_id < state->n_tasks; task_id += n) {
        uint32_t start = task_id * state->n_units_per_task;
        if (start >= state->n_tot_units) {
            break;
        }
        uint32_t end = hex_smin(start + state->n_units_per_task, state->n_tot_units);
        rescale_int_weight_to_int8_task(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

static void rescale_int_weight_chunk_to_int8_tiles_scales(
        struct htp_ops_context * octx,
        void * vtcm_weight_buf,
        size_t weight_bias_offset,
        float * dst_scales,
        const void * weight_src_raw,
        int n_cols,
        int k_block,
        int n_groups,
        const struct fastdiv_values * n_groups_div,
        int aligned_tile_size,
        int32_t weight_type) {

    const int n_hvx            = (int) octx->n_hvx;
    const uint32_t n_col_tiles = (uint32_t) hmx_ceil_div(n_cols, 32);
    const uint32_t n_k_tiles   = (uint32_t) (k_block >> 5);
    const uint32_t tail_tiles  = n_k_tiles - ((uint32_t) n_groups - 1u) * HTP_MM_HMX_W8A8_KGROUP_TILES;
    const uint32_t n_tot_units = n_col_tiles * (uint32_t) n_groups;
    if (n_tot_units == 0) {
        return;
    }

    const uint32_t n_units_per_task = (n_hvx <= 1) ? n_tot_units
                                    : fastdiv(n_tot_units + (uint32_t) n_hvx - 1, &octx->n_hvx_div);

    int_weight_rescale_state_t state;
    resolve_int_weight_rescale_fns(weight_type, &state.rescale_2k_fn, &state.rescale_tail_fn);
    state.dst_wt_base       = (uint8_t *) vtcm_weight_buf;
    state.dst_bias_base     = (uint8_t *) vtcm_weight_buf + weight_bias_offset;
    state.dst_scales_base   = dst_scales;
    state.src_raw           = (const uint8_t *) weight_src_raw;
    state.n_col_tiles       = n_col_tiles;
    state.n_k_tiles         = n_k_tiles;
    state.n_groups          = (uint32_t) n_groups;
    state.tail_tiles        = tail_tiles;
    state.aligned_tile_size = (uint32_t) aligned_tile_size;
    state.raw_shift         = (weight_type == HTP_TYPE_Q1_0) ? 2u
                            : ((weight_type == HTP_TYPE_Q2_0) ? 1u : 0u);
    state.n_tot_units       = n_tot_units;
    state.n_units_per_task  = n_units_per_task;
    state.n_tasks           = hex_smin(n_tot_units, (uint32_t) n_hvx);
    state.n_groups_div      = *n_groups_div;

    work_queue_run(octx->ctx->hvx_queue, rescale_int_weight_worker_loop, &state, (int) state.n_tasks);
}

static void rescale_int_weight_chunk_to_int8_tiles(
        struct htp_ops_context * octx,
        void * vtcm_weight_buf,
        size_t weight_bias_offset,
        size_t weight_scales_offset,
        const void * weight_src_raw,
        int n_cols,
        int k_block,
        int n_groups,
        const struct fastdiv_values * n_groups_div,
        int aligned_tile_size,
        int32_t weight_type) {
    float * dst_scales = (float *) ((uint8_t *) vtcm_weight_buf + weight_scales_offset);
    rescale_int_weight_chunk_to_int8_tiles_scales(
        octx, vtcm_weight_buf, weight_bias_offset, dst_scales,
        weight_src_raw, n_cols, k_block, n_groups, n_groups_div, aligned_tile_size, weight_type);
}

typedef void (*int_act_dma_pipelined_fn_t)(
        dma_queue * dma_q,
        uint8_t * restrict vtcm_act_dst,
        float * restrict vtcm_act_scales,
        dma_addr_t act_dma_addr,
        uint32_t row_start,
        uint32_t row_end,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        uint32_t n_groups,
        uint8_t * act_spad,
        size_t scratch_bytes_per_hvx,
        struct htp_thread_trace * tr);

typedef struct {
    int_act_dma_pipelined_fn_t dma_pipelined_fn;
    uint8_t *                  vtcm_act_dst;
    float *                    vtcm_act_scales;
    dma_addr_t                 act_dma_addr;
    float *                    vtcm_f32_act;
    uint32_t                   n_tasks;
    uint32_t                   n_rows;
    uint32_t                   mc_pad;
    uint32_t                   n_rows_per_task;
    uint32_t                   k_block;
    uint32_t                   k_stride;
    uint32_t                   k_valid;
    uint32_t                   n_groups;
    size_t                     vtcm_f32_act_bytes_per_hvx;
} int_activation_transfer_task_state_t;

#define DEFINE_TRANSFER_ACT_CHUNK_DMA_PIPELINED(SFX, ACT_T)                                                    \
static void transfer_activation_chunk_##SFX##_dma_pipelined(                                                   \
        dma_queue * dma_q,                                                                                     \
        uint8_t * restrict vtcm_act_dst,                                                                       \
        float * restrict vtcm_act_scales,                                                                      \
        dma_addr_t act_dma_addr,                                                                               \
        uint32_t row_start,                                                                                    \
        uint32_t row_end,                                                                                      \
        uint32_t n_rows,                                                                                       \
        uint32_t mc_pad,                                                                                       \
        uint32_t k_block,                                                                                      \
        uint32_t k_stride,                                                                                     \
        uint32_t k_valid,                                                                                      \
        uint32_t n_groups,                                                                                     \
        uint8_t * act_spad,                                                                                    \
        size_t scratch_bytes_per_hvx,                                                                          \
        struct htp_thread_trace * tr) {                                                                        \
    const uint32_t step_rows     = 4u;                                                                         \
    const size_t   row_bytes     = (size_t) k_block * sizeof(ACT_T);                                           \
    const size_t   src_row_bytes = (size_t) k_stride * sizeof(ACT_T);                                          \
    const size_t   width_bytes   = (size_t) k_valid * sizeof(ACT_T);                                           \
    const bool     double_buf    = (scratch_bytes_per_hvx >= 2 * step_rows * row_bytes);                       \
    const uint32_t n_k_tiles     = k_block >> 5;                                                               \
    if (double_buf) {                                                                                          \
        if (row_start < row_end && row_start < n_rows) {                                                       \
            uint32_t nrows_to_fetch = hex_smin(n_rows - row_start, step_rows);                                 \
            dma_queue_push(dma_q, dma_make_data(act_spad, act_dma_addr + (size_t) row_start * src_row_bytes),  \
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);                             \
        }                                                                                                      \
        if (row_start + step_rows < row_end && row_start + step_rows < n_rows) {                               \
            uint32_t next_r = row_start + step_rows;                                                           \
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, step_rows);                                    \
            uint8_t * next_buf = act_spad + step_rows * row_bytes;                                             \
            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * src_row_bytes),     \
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);                             \
        }                                                                                                      \
    }                                                                                                          \
    for (uint32_t r = row_start; r < row_end; r += step_rows) {                                                \
        uint8_t * curr_buf = act_spad;                                                                         \
        if (r < n_rows) {                                                                                      \
            if (double_buf) {                                                                                  \
                curr_buf = (uint8_t *) dma_queue_pop(dma_q).dst;                                               \
            } else {                                                                                           \
                uint32_t nrows_to_fetch = hex_smin(n_rows - r, step_rows);                                     \
                dma_queue_push(dma_q, dma_make_data(act_spad, act_dma_addr + (size_t) r * src_row_bytes),      \
                               row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);                         \
                curr_buf = (uint8_t *) dma_queue_pop(dma_q).dst;                                               \
            }                                                                                                  \
        }                                                                                                      \
        const uint32_t valid_rows = (r < n_rows) ? hex_smin(n_rows - r, step_rows) : 0u;                       \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, r);                                               \
        quantize_act_step_rows_##SFX(                                                                          \
            vtcm_act_dst, vtcm_act_scales, curr_buf, row_bytes,                                                \
            r, valid_rows, mc_pad, n_k_tiles, n_groups, k_valid);                                              \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, r);                                                \
        if (double_buf) {                                                                                      \
            uint32_t next_r = r + 2 * step_rows;                                                               \
            if (next_r < row_end && next_r < n_rows) {                                                         \
                uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, step_rows);                                \
                dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * src_row_bytes), \
                               row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);                         \
            }                                                                                                  \
        }                                                                                                      \
    }                                                                                                          \
}

DEFINE_TRANSFER_ACT_CHUNK_DMA_PIPELINED(f16_w8a8, __fp16)
DEFINE_TRANSFER_ACT_CHUNK_DMA_PIPELINED(f32_w8a8, float)

static inline int_act_dma_pipelined_fn_t resolve_int_act_dma_pipelined_fn(uint32_t act_elem_size) {
    return (act_elem_size == sizeof(__fp16)) ? transfer_activation_chunk_f16_w8a8_dma_pipelined
                                             : transfer_activation_chunk_f32_w8a8_dma_pipelined;
}

static void transfer_activation_chunk_int_worker_fn(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    int_activation_transfer_task_state_t * st = (int_activation_transfer_task_state_t *) data;
    struct htp_thread_trace * tr = th->trace;

    for (unsigned int task_id = i; task_id < st->n_tasks; task_id += n) {
        uint32_t row_start = task_id * st->n_rows_per_task;
        if (row_start >= st->mc_pad) {
            break;
        }
        uint32_t row_end   = hex_smin(row_start + st->n_rows_per_task, st->mc_pad);
        uint8_t * act_spad = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_hvx;
        st->dma_pipelined_fn(
            th->dma_q, st->vtcm_act_dst, st->vtcm_act_scales, st->act_dma_addr,
            row_start, row_end, st->n_rows, st->mc_pad,
            st->k_block, st->k_stride, st->k_valid, st->n_groups,
            act_spad, st->vtcm_f32_act_bytes_per_hvx, tr);
    }
}

static void transfer_activation_chunk_int_threaded(
        struct htp_context * ctx,
        uint8_t * vtcm_act_dst,
        float * vtcm_act_scales,
        dma_addr_t act_dma_addr,
        int n_rows,
        int k_block,
        int k_stride,
        int k_valid,
        int n_groups,
        int n_hvx,
        const struct fastdiv_values * act_hvx_div,
        float * vtcm_f32_act,
        size_t vtcm_f32_act_bytes,
        uint32_t act_elem_size) {

    if (n_rows <= 0) {
        return;
    }

    const uint32_t mc_pad         = hex_align_up((uint32_t) n_rows, 64u);
    const uint32_t resolved_esize = act_elem_size ? act_elem_size : (uint32_t) sizeof(float);
    uint32_t       rows_per_task  = (n_hvx <= 1) ? mc_pad : fastdiv(mc_pad + (uint32_t) n_hvx - 1, act_hvx_div);
    rows_per_task                 = hex_align_up(rows_per_task, 4u);

    int_activation_transfer_task_state_t state;
    state.dma_pipelined_fn           = resolve_int_act_dma_pipelined_fn(resolved_esize);
    state.vtcm_act_dst               = vtcm_act_dst;
    state.vtcm_act_scales            = vtcm_act_scales;
    state.act_dma_addr               = act_dma_addr;
    state.vtcm_f32_act               = vtcm_f32_act;
    state.n_rows                     = (uint32_t) n_rows;
    state.mc_pad                     = mc_pad;
    state.n_rows_per_task            = rows_per_task;
    state.n_tasks                    = (uint32_t) hex_smin(n_hvx, (int) (mc_pad >> 2));
    state.k_block                    = (uint32_t) k_block;
    state.k_stride                   = (uint32_t) k_stride;
    state.k_valid                    = (uint32_t) k_valid;
    state.n_groups                   = (uint32_t) n_groups;
    state.vtcm_f32_act_bytes_per_hvx = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_hvx_div), 128);

    work_queue_run(ctx->hvx_queue, transfer_activation_chunk_int_worker_fn, &state, (int) state.n_tasks);
}

typedef struct {
    int_output_col_chunk_fn_t col_chunk_fn;
    void *                    dst;
    const void *              bias;
    const uint8_t *           vtcm_src;
    const float *             vtcm_act_scales;
    const float *             vtcm_weight_scales;
    uint32_t                  n_rows;
    uint32_t                  mc_pad;
    uint32_t                  n_cols;
    uint32_t                  n_groups;
    uint32_t                  dst_stride;
    uint32_t                  bias_stride;
    uint32_t                  dst_cols;
    uint32_t                  n_tasks;
    uint32_t                  n_chunks_per_task;
    size_t                    dst_elem_size;
    size_t                    bias_elem_size;
    struct fastdiv_values     n_hvx_div;
} int_output_transfer_state_t;

static void transfer_output_chunk_int_col_chunk_worker_fn(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    int_output_transfer_state_t * st = (int_output_transfer_state_t *) data;
    struct htp_thread_trace * tr = th->trace;

    const uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
    uint32_t b_first;
    uint32_t b_last;
    if (st->dst_elem_size == sizeof(__fp16)) {
        const uint32_t n_pairs = hmx_ceil_div(n_blocks, 2);
        b_first = fastdiv(n_pairs * i, &st->n_hvx_div) * 2;
        b_last  = MIN(n_blocks, fastdiv(n_pairs * (i + 1), &st->n_hvx_div) * 2);
    } else {
        b_first = fastdiv(n_blocks * i, &st->n_hvx_div);
        b_last  = fastdiv(n_blocks * (i + 1), &st->n_hvx_div);
    }
    if (b_first >= b_last) {
        return;
    }
    const uint32_t c_first  = b_first * 32;
    const uint32_t c_last   = b_last * 32;
    const uint32_t c_len    = c_last - c_first;

    const int chunk_dst_cols = (int) st->dst_cols - (int) c_first;
    if (chunk_dst_cols <= 0) {
        return;
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
    const void * bias = st->bias ? ((const uint8_t *) st->bias + (size_t) c_first * st->bias_elem_size) : NULL;
    void *       dst  = (uint8_t *) st->dst + (size_t) c_first * st->dst_elem_size;

    st->col_chunk_fn(
        dst, bias, st->vtcm_src, st->vtcm_act_scales, st->vtcm_weight_scales,
        0, st->n_rows, st->mc_pad, c_first, c_len, st->n_cols, st->n_groups,
        st->dst_stride, st->bias_stride, (uint32_t) chunk_dst_cols, st->bias_elem_size);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

static void transfer_output_chunk_int_worker_fn(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    int_output_transfer_state_t * st = (int_output_transfer_state_t *) data;
    struct htp_thread_trace * tr = th->trace;

    const uint32_t n_quads = (st->n_rows + 3u) >> 2;
    const uint32_t q_first = (n_quads <= n) ? i : fastdiv(n_quads * i, &st->n_hvx_div);
    const uint32_t q_last  = (n_quads <= n) ? (i + 1u) : fastdiv(n_quads * (i + 1u), &st->n_hvx_div);
    const uint32_t row_start = q_first << 2;
    const uint32_t row_end   = hex_smin(st->n_rows, q_last << 2);
    if (row_start >= row_end) {
        return;
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, row_start);
    const uint32_t chunk_size = row_end - row_start;
    void *         dst        = (uint8_t *) st->dst + (size_t) row_start * st->dst_stride * st->dst_elem_size;
    const void *   bias       = st->bias ? ((const uint8_t *) st->bias + (size_t) row_start * st->bias_stride * st->bias_elem_size) : NULL;

    st->col_chunk_fn(
        dst, bias, st->vtcm_src, st->vtcm_act_scales, st->vtcm_weight_scales,
        row_start, chunk_size, st->mc_pad, 0, st->n_cols, st->n_cols, st->n_groups,
        st->dst_stride, st->bias_stride, st->dst_cols, st->bias_elem_size);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, row_start);
}

static void transfer_output_chunk_int_threaded(
        struct htp_ops_context * octx,
        void * dst,
        const void * bias,
        const void * vtcm_src,
        const float * vtcm_act_scales,
        const float * vtcm_weight_scales,
        int n_rows,
        int n_cols,
        int n_groups,
        int dst_stride,
        uint32_t bias_stride,
        int dst_cols,
        size_t dst_elem_size,
        size_t bias_elem_size) {

    if (n_rows <= 0 || dst_cols <= 0) {
        return;
    }

    const int      n_hvx          = (int) octx->n_hvx;
    const uint32_t mc_pad         = hex_align_up((uint32_t) n_rows, 64u);
    const bool     dst_aligned_nt = ((((uintptr_t) dst | ((size_t) dst_stride * dst_elem_size)) & 127u) == 0);

    int_output_transfer_state_t state;
    state.col_chunk_fn       = resolve_int_output_col_chunk_fn((uint32_t) n_groups, dst_elem_size, bias, bias_elem_size, dst_aligned_nt);
    state.dst                = dst;
    state.bias               = bias;
    state.vtcm_src           = (const uint8_t *) vtcm_src;
    state.vtcm_act_scales    = vtcm_act_scales;
    state.vtcm_weight_scales = vtcm_weight_scales;
    state.n_rows             = (uint32_t) n_rows;
    state.mc_pad             = mc_pad;
    state.n_cols             = (uint32_t) n_cols;
    state.n_groups           = (uint32_t) n_groups;
    state.dst_stride         = (uint32_t) dst_stride;
    state.bias_stride        = bias_stride;
    state.dst_cols           = (uint32_t) dst_cols;
    state.dst_elem_size      = dst_elem_size;
    state.bias_elem_size     = bias_elem_size;
    state.n_hvx_div          = octx->n_hvx_div;

    const uint32_t n_blocks         = hmx_ceil_div((uint32_t) n_cols, 32);
    const uint32_t n_units          = (dst_elem_size == sizeof(__fp16)) ? hmx_ceil_div(n_blocks, 2) : n_blocks;
    const bool     prefer_row_chunk = ((uint32_t) n_rows >= (uint32_t) n_hvx * 64u);
    if (n_hvx > 1 && !prefer_row_chunk && n_units >= (uint32_t) n_hvx) {
        work_queue_run(octx->ctx->hvx_queue, transfer_output_chunk_int_col_chunk_worker_fn, &state, n_hvx);
        return;
    }

    const int actual_hvx = hex_smin(n_hvx, (n_rows + 3) >> 2);
    work_queue_run(octx->ctx->hvx_queue, transfer_output_chunk_int_worker_fn, &state, actual_hvx);
}

static void transfer_output_chunk_int_glu_col_chunk_task(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    hmx_output_glu_state_t * st = (hmx_output_glu_state_t *) data;
    struct htp_thread_trace * tr = th->trace;

    const uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
    uint32_t b_first;
    uint32_t b_last;
    if (st->dst_elem_size == sizeof(__fp16)) {
        const uint32_t n_pairs = hmx_ceil_div(n_blocks, 2);
        b_first = fastdiv(n_pairs * i, &st->n_hvx_div) * 2;
        b_last  = MIN(n_blocks, fastdiv(n_pairs * (i + 1), &st->n_hvx_div) * 2);
    } else {
        b_first = fastdiv(n_blocks * i, &st->n_hvx_div);
        b_last  = fastdiv(n_blocks * (i + 1), &st->n_hvx_div);
    }
    if (b_first >= b_last) {
        return;
    }
    const uint32_t c_first        = b_first * 32;
    const uint32_t c_len          = (b_last - b_first) * 32;
    const int      chunk_dst_cols = (int) st->dst_cols - (int) c_first;
    if (chunk_dst_cols <= 0) {
        return;
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
    void * dst = (uint8_t *) st->dst + (size_t) c_first * st->dst_elem_size;
    const uint32_t eff_dst_cols = (uint32_t) hex_smin((size_t) c_len, (size_t) chunk_dst_cols);
    st->col_chunk_fn(st, dst, 0, st->n_rows, c_first, (c_len << 16) | eff_dst_cols);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

static void transfer_output_chunk_int_glu_row_task(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    hmx_output_glu_state_t * st = (hmx_output_glu_state_t *) data;
    struct htp_thread_trace * tr = th->trace;

    const uint32_t n_quads   = (st->n_rows + 3u) >> 2;
    const uint32_t q_first   = (n_quads <= n) ? i : fastdiv(n_quads * i, &st->n_hvx_div);
    const uint32_t q_last    = (n_quads <= n) ? (i + 1u) : fastdiv(n_quads * (i + 1u), &st->n_hvx_div);
    const uint32_t row_start = q_first << 2;
    const uint32_t row_end   = hex_smin(st->n_rows, q_last << 2);
    if (row_start >= row_end) {
        return;
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, row_start);
    const uint32_t eff_dst_cols = (uint32_t) hex_smin((size_t) st->n_cols, (size_t) st->dst_cols);
    const uint32_t c_packed     = (st->n_cols << 16) | eff_dst_cols;
    const uint32_t chunk_size   = row_end - row_start;
    void *         dst          = (uint8_t *) st->dst + (size_t) row_start * st->dst_stride * st->dst_elem_size;
    st->col_chunk_fn(st, dst, row_start, chunk_size, 0, c_packed);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, row_start);
}

static void transfer_output_chunk_int_glu_threaded(
        struct htp_ops_context * octx,
        hmx_output_glu_state_t * state,
        void * dst,
        const void * vtcm_gate,
        const void * vtcm_up,
        const float * vtcm_wt_scales_gate,
        const float * vtcm_wt_scales_up,
        int n_rows,
        int n_cols,
        int dst_cols) {

    if (n_rows <= 0 || dst_cols <= 0) {
        return;
    }

    const int n_hvx            = (int) octx->n_hvx;
    state->dst                 = dst;
    state->vtcm_gate           = (const uint8_t *) vtcm_gate;
    state->vtcm_up             = (const uint8_t *) vtcm_up;
    state->vtcm_wt_scales_gate = vtcm_wt_scales_gate;
    state->vtcm_wt_scales_up   = vtcm_wt_scales_up;
    state->n_rows              = (uint32_t) n_rows;
    state->mc_pad              = hex_align_up((uint32_t) n_rows, 64u);
    state->n_cols              = (uint32_t) n_cols;
    state->dst_cols            = (uint32_t) dst_cols;
    state->n_hvx_div           = octx->n_hvx_div;

    const uint32_t n_blocks         = hmx_ceil_div((uint32_t) n_cols, 32);
    const uint32_t n_units          = (state->dst_elem_size == sizeof(__fp16)) ? hmx_ceil_div(n_blocks, 2) : n_blocks;
    const bool     prefer_row_chunk = ((uint32_t) n_rows >= (uint32_t) n_hvx * 64u);
    if (n_hvx > 1 && !prefer_row_chunk && n_units >= (uint32_t) n_hvx) {
        work_queue_run(octx->ctx->hvx_queue, transfer_output_chunk_int_glu_col_chunk_task, state, n_hvx);
        return;
    }

    const int actual_hvx = hex_smin(n_hvx, (n_rows + 3) >> 2);
    work_queue_run(octx->ctx->hvx_queue, transfer_output_chunk_int_glu_row_task, state, actual_hvx);
}

typedef struct {
    const struct mmid_row_mapping * matrix_rows;
    uint8_t *                       vtcm_act_dst;
    float *                         vtcm_act_scales;
    dma_addr_t                      act_dma_addr;
    float *                         vtcm_f32_act;
    uint32_t                        act_elem_size;
    uint32_t                        n_tasks;
    uint32_t                        n_rows;
    uint32_t                        mc_pad;
    uint32_t                        k_block;
    uint32_t                        k_valid;
    uint32_t                        n_groups;
    size_t                          vtcm_f32_act_bytes_per_hvx;
    uint32_t                        start_row;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    struct fastdiv_values           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
} int_activation_transfer_gathered_state_t;

static void transfer_activation_chunk_int_gathered_worker_fn(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    int_activation_transfer_gathered_state_t * st = (int_activation_transfer_gathered_state_t *) data;
    struct htp_thread_trace * tr = th->trace;
    dma_queue * dma_q = th->dma_q;

    const uint32_t row_start  = i * 4u;
    const uint32_t row_stride = n * 4u;
    if (row_start >= st->mc_pad) {
        return;
    }

    const bool     act_is_f16  = (st->act_elem_size == sizeof(__fp16));
    const size_t   row_bytes   = hex_align_up((size_t) st->k_block * st->act_elem_size, 128);
    const size_t   valid_bytes = (size_t) st->k_valid * st->act_elem_size;
    const size_t   slot_bytes  = 4u * row_bytes;
    const bool     double_buf  = (st->vtcm_f32_act_bytes_per_hvx >= 2u * slot_bytes);
    const uint32_t n_k_tiles   = st->k_block >> 5;
    uint8_t *      act_spad    = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_hvx;

    if (double_buf) {
        if (row_start < st->n_rows) {
            const uint32_t nrows = hex_smin(st->n_rows - row_start, 4u);
            push_gathered_act_rows_dma(dma_q, act_spad, st->act_dma_addr, nrows, st->start_row, row_start,
                                       row_bytes, valid_bytes, st->matrix_rows, st->cur_a, st->mapping_stride,
                                       st->ne11, &st->ne11_div, st->nb11, st->nb12);
        }
        const uint32_t r1 = row_start + row_stride;
        if (r1 < st->mc_pad && r1 < st->n_rows) {
            const uint32_t nrows = hex_smin(st->n_rows - r1, 4u);
            push_gathered_act_rows_dma(dma_q, act_spad + slot_bytes, st->act_dma_addr, nrows, st->start_row, r1,
                                       row_bytes, valid_bytes, st->matrix_rows, st->cur_a, st->mapping_stride,
                                       st->ne11, &st->ne11_div, st->nb11, st->nb12);
        }
    }

    uint32_t step = 0;
    for (uint32_t r = row_start; r < st->mc_pad; r += row_stride, ++step) {
        uint8_t *      curr_buf   = act_spad + (double_buf ? (step & 1u) * slot_bytes : 0u);
        const uint32_t valid_rows = (r < st->n_rows) ? hex_smin(st->n_rows - r, 4u) : 0u;

        if (valid_rows > 0) {
            if (!double_buf) {
                push_gathered_act_rows_dma(dma_q, curr_buf, st->act_dma_addr, valid_rows, st->start_row, r,
                                           row_bytes, valid_bytes, st->matrix_rows, st->cur_a, st->mapping_stride,
                                           st->ne11, &st->ne11_div, st->nb11, st->nb12);
            }
            for (uint32_t s = 0; s < valid_rows; ++s) {
                dma_queue_pop(dma_q);
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, r);
        if (act_is_f16) {
            quantize_act_step_rows_f16_w8a8(
                st->vtcm_act_dst, st->vtcm_act_scales, curr_buf, row_bytes,
                r, valid_rows, st->mc_pad, n_k_tiles, st->n_groups, st->k_valid);
        } else {
            quantize_act_step_rows_f32_w8a8(
                st->vtcm_act_dst, st->vtcm_act_scales, curr_buf, row_bytes,
                r, valid_rows, st->mc_pad, n_k_tiles, st->n_groups, st->k_valid);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, r);

        if (double_buf) {
            const uint32_t next_r = r + 2u * row_stride;
            if (next_r < st->mc_pad && next_r < st->n_rows) {
                const uint32_t nrows = hex_smin(st->n_rows - next_r, 4u);
                push_gathered_act_rows_dma(dma_q, curr_buf, st->act_dma_addr, nrows, st->start_row, next_r,
                                           row_bytes, valid_bytes, st->matrix_rows, st->cur_a, st->mapping_stride,
                                           st->ne11, &st->ne11_div, st->nb11, st->nb12);
            }
        }
    }
}

static void transfer_activation_chunk_int_gathered_threaded(
        struct htp_context * ctx,
        uint8_t * vtcm_act_dst,
        float * vtcm_act_scales,
        dma_addr_t act_dma_addr,
        uint32_t act_elem_size,
        int start_row,
        int n_rows,
        int k_block,
        int k_valid,
        int n_groups,
        const struct mmid_row_mapping * matrix_rows,
        int cur_a,
        int mapping_stride,
        int ne11,
        size_t nb11,
        size_t nb12,
        int n_hvx,
        const struct fastdiv_values * act_hvx_div,
        float * vtcm_f32_act,
        size_t vtcm_f32_act_bytes) {
    if (n_rows <= 0) {
        return;
    }

    const uint32_t mc_pad         = hex_align_up((uint32_t) n_rows, 64u);
    const uint32_t resolved_esize = act_elem_size ? act_elem_size : (uint32_t) sizeof(float);

    int_activation_transfer_gathered_state_t state = {
        .matrix_rows                = matrix_rows,
        .vtcm_act_dst               = vtcm_act_dst,
        .vtcm_act_scales            = vtcm_act_scales,
        .act_dma_addr               = act_dma_addr,
        .vtcm_f32_act               = vtcm_f32_act,
        .act_elem_size              = resolved_esize,
        .n_tasks                    = (uint32_t) hex_smin(n_hvx, (int) (mc_pad >> 2)),
        .n_rows                     = (uint32_t) n_rows,
        .mc_pad                     = mc_pad,
        .k_block                    = (uint32_t) k_block,
        .k_valid                    = (uint32_t) k_valid,
        .n_groups                   = (uint32_t) n_groups,
        .vtcm_f32_act_bytes_per_hvx = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_hvx_div), 128),
        .start_row                  = (uint32_t) start_row,
        .cur_a                      = (uint32_t) cur_a,
        .mapping_stride             = (uint32_t) mapping_stride,
        .ne11                       = (uint32_t) ne11,
        .ne11_div                   = ne11 > 1 ? init_fastdiv_values(ne11) : (struct fastdiv_values){0, 0},
        .nb11                       = nb11,
        .nb12                       = nb12,
    };

    work_queue_run(ctx->hvx_queue, transfer_activation_chunk_int_gathered_worker_fn, &state, (int) state.n_tasks);
}

// --- Async HMX matmul task (for pipeline overlap) ---

typedef struct {
    __fp16 *        output;
    const __fp16 *  activation;
    const __fp16 *  weight;
    const __fp16 *  scales;
    const uint8_t * weight_bias;
    uint32_t        n_row_tiles;
    uint32_t        n_col_tiles;
    uint32_t        n_dot_tiles;
    uint32_t        n_groups;
} hmx_matmul_task_t;

static void hmx_matmul_fp16_worker_fn(const htp_thread_t * th, void * data) {
    (void) th;
    hmx_matmul_task_t * task = (hmx_matmul_task_t *) data;
    FARF(HIGH, "hmx-mm-fp16-task: n_row_tiles %u n_col_tiles %u n_dot_tiles %u",
         task->n_row_tiles, task->n_col_tiles, task->n_dot_tiles);
    core_dot_chunk_fp16(task->output, task->activation, task->weight, task->scales,
                        task->n_row_tiles, task->n_col_tiles, task->n_dot_tiles);
}

static void hmx_matmul_w8a8_worker_fn(const htp_thread_t * th, void * data) {
    (void) th;
    hmx_matmul_task_t * task = (hmx_matmul_task_t *) data;
    FARF(HIGH, "hmx-mm-w8a8-task: n_row_tiles %u n_col_tiles %u n_dot_tiles %u n_groups %u",
         task->n_row_tiles, task->n_col_tiles, task->n_dot_tiles, task->n_groups);
    core_dot_chunk_w8a8_sm((uint8_t *) task->output, (const uint8_t *) task->activation,
                           (const uint8_t *) task->weight, task->weight_bias,
                           task->n_row_tiles, task->n_col_tiles, task->n_dot_tiles, task->n_groups);
}

static inline hmx_queue_func hmx_get_matmul_worker_fn(int32_t kernel_type) {
    if (htp_mm_kernel_is_hmx_w8a8(kernel_type)) {
        return hmx_matmul_w8a8_worker_fn;
    }
    return hmx_matmul_fp16_worker_fn;
}

static inline void hmx_matmul_task_init(hmx_matmul_task_t * task,
                                       __fp16 *           output,
                                       const __fp16 *     activation,
                                       const __fp16 *     weight,
                                       const __fp16 *     scales,
                                       uint32_t           n_row_tiles,
                                       uint32_t           n_col_tiles,
                                       uint32_t           n_dot_tiles) {
    task->output      = output;
    task->activation  = activation;
    task->weight      = weight;
    task->scales      = scales;
    task->weight_bias = NULL;
    task->n_row_tiles = n_row_tiles;
    task->n_col_tiles = n_col_tiles;
    task->n_dot_tiles = n_dot_tiles;
    task->n_groups    = 1;
}

static inline void hmx_matmul_task_init_int(hmx_matmul_task_t * task,
                                           void *             output,
                                           const void *       activation,
                                           const void *       weight_buf,
                                           size_t             weight_bias_offset,
                                           uint32_t           n_row_tiles,
                                           uint32_t           n_col_tiles,
                                           uint32_t           n_dot_tiles,
                                           uint32_t           n_groups) {
    task->output      = (__fp16 *) output;
    task->activation  = (const __fp16 *) activation;
    task->weight      = (const __fp16 *) weight_buf;
    task->scales      = NULL;
    task->weight_bias = (const uint8_t *) weight_buf + weight_bias_offset;
    task->n_row_tiles = n_row_tiles;
    task->n_col_tiles = n_col_tiles;
    task->n_dot_tiles = n_dot_tiles;
    task->n_groups    = n_groups;
}

static int hmx_mm_2d_f32(struct htp_ops_context *octx,
                                  dma_queue *weight_dma,
                                  dma_addr_t dst_dma_addr,
                                  dma_addr_t bias_addr,
                                  size_t bias_bytes,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t weight,
                                  int m, int k, int n,
                                  int act_stride,
                                  uint32_t act_elem_size,  // 4=F32 src1, 2=F16 src1
                                  int weight_stride,
                                  int weight_type,
                                  int k_valid,
                                  int dst_stride,
                                  uint32_t bias_stride,
                                  int dst_cols,
                                  uint32_t dst_elem_size,
                                  uint32_t bias_elem_size,
                                  int m_chunk,
                                  int n_chunk,
                                  int pipeline,
                                  int act_hvx,
                                  const struct fastdiv_values * act_hvx_div,
                                  const struct fastdiv_values * k_div,
                                  int tile_size,
                                  int aligned_tile_size,
                                  int vtcm_size,
                                  int32_t kernel_type,
                                  int hmx_n_groups,
                                  const struct fastdiv_values * div_hmx_n_groups,
                                  const struct fastdiv_values * div_n_k_tiles) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const bool is_tiled   = (tile_size > 0);
    const bool is_int_hmx = htp_mm_kernel_is_hmx_int(kernel_type);
    const int  n_groups   = is_int_hmx ? hmx_n_groups : 1;

    if ((k & 31) != 0) {
        return -1;
    }
    if ((dst_dma_addr & (dst_elem_size - 1)) != 0) { return -1; }

    const struct fastdiv_values n_groups_div = *div_hmx_n_groups;
    const hmx_queue_func hmx_worker_fn = hmx_get_matmul_worker_fn(kernel_type);

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    const int n_k_tiles = k >> 5;
    const int n_weight_tiles = (weight_type == HTP_TYPE_Q4_0) ? n_k_tiles : (k / htp_tensor_get_tile_k(weight_type));
    const struct fastdiv_values n_k_tiles_div = *div_n_k_tiles;

    const size_t vtcm_budget  = ctx->vtcm_size;

    const uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    const uint32_t dma_src_stride  = is_tiled ? tile_size : weight_stride;
    const uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used      = vtcm_size;

    const bool has_bias    = (bias_bytes > 0 && bias_addr != 0);
    const bool has_2d_bias = (pipeline && !has_bias && bias_addr != 0);
    const size_t res_in_elem_size  = has_2d_bias ? bias_elem_size : 0;
    const size_t res_out_elem_size = dst_elem_size;

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build_ex(&L, is_int_hmx ? kernel_type : HTP_MM_KERNEL_HMX_2D, weight_type, k,
                                    m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_hvx, aligned_tile_size, bias_bytes,
                                    res_in_elem_size, res_out_elem_size);

    vtcm_used = L.total_bytes;
    if (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_used, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        return -1;
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    float   *vtcm_act_scales = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_act_scales, is_int_hmx);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    void    *vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    void    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_scratch[1], pipeline);
    void    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_dst[1], pipeline);
    void    *vtcm_res_in[2]  = {
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_res_in[0], has_2d_bias),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_res_in[1], has_2d_bias),
    };
    void    *vtcm_res_out[2] = {
        VTCM_LAYOUT_PTR(void, base, L.off_res_out[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_res_out[1], pipeline),
    };
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_scales, !is_int_hmx);

    if (!is_int_hmx) {
        hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16
    }
    if (L.lut_bytes > 0) {
        iq2xxs_lut_copy(base + L.off_lut);
    }

    void    *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_bias, has_bias);
    if (has_bias) {
        dma_queue_push(weight_dma, dma_make_data(vtcm_bias, bias_addr), hex_align_up(bias_bytes, 128), 0, bias_bytes, 1);
        dma_queue_pop(weight_dma);
    }

    FARF(HIGH, "hmx-mm-2d: ktype %d m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         (int) kernel_type, m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    int n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);
    const uint32_t m_tile_rows = is_int_hmx ? 64u : HTP_MM_HMX_TILE_N_ROWS;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    if (pipeline) {
        // --- Asynchronous Pipelined Loop (strict lockstep FIFO on weight_dma) ---
        hmx_matmul_task_t task_slots[2];
        void *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        void *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        void *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        const size_t   dst_row_bytes        = (size_t) dst_stride * dst_elem_size;
        const size_t   bias_row_bytes       = (size_t) bias_stride * bias_elem_size;
        const size_t   res_out_stride_bytes = hex_align_up(n_chunk_n_cols * dst_elem_size, 128);
        const int      res_out_stride_elems = (int) (res_out_stride_bytes / dst_elem_size);
        const size_t   res_in_stride_bytes  = has_2d_bias ? hex_align_up(n_chunk_n_cols * bias_elem_size, 128) : 0;
        const uint32_t res_in_stride_elems  = has_2d_bias ? (uint32_t) (res_in_stride_bytes / bias_elem_size) : 0;

        const bool preissue_w0 = is_int_hmx && L.can_preissue_weight && n > 0;

        for (size_t mr = 0; mr < (size_t) m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin((size_t) m - mr, m_chunk_n_rows);

            if (preissue_w0) {
                const size_t   n_cols_A0 = hex_smin((size_t) n, n_chunk_n_cols);
                const uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
                dma_proxy_push(ctx->dma_proxy, dma_make_data(vtcm_raw_bufs[0], weight),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);
            }

            if (is_int_hmx) {
                transfer_activation_chunk_int_threaded(
                    ctx, (uint8_t *) vtcm_f16_act, vtcm_act_scales,
                    act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                    (int) n_rows, k, act_stride, k_valid, n_groups,
                    act_hvx, act_hvx_div, vtcm_f32_act, L.act_f32_bytes, act_elem_size);
            } else {
                struct activation_transfer_params act_params = {
                    .ctx = ctx,
                    .dst = vtcm_f16_act,
                    .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                    .n_rows = (int) n_rows,
                    .k_block = k,
                    .k_stride = act_stride,
                    .n_hvx = act_hvx,
                    .act_hvx_div = act_hvx_div,
                    .k_div = k_div,
                    .k_valid = k_valid,
                    .vtcm_f32_act = vtcm_f32_act,
                    .vtcm_f32_act_bytes = L.act_f32_bytes,
                    .act_elem_size = act_elem_size,
                };
                transfer_activation_chunk_threaded(&act_params);
            }

            if (!preissue_w0) {
                const size_t   n_cols_A0 = hex_smin((size_t) n, n_chunk_n_cols);
                const uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
                dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);
            }
            if (1 < n_chunk_cnt) {
                const size_t   n_cols_A1 = hex_smin((size_t) n - n_chunk_n_cols, n_chunk_n_cols);
                const uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
            }
            if (has_2d_bias) {
                const int    vcols0 = hex_smin((int) hex_smin((size_t) n, n_chunk_n_cols), dst_cols);
                const size_t cbytes = (vcols0 > 0) ? (size_t) vcols0 * bias_elem_size : 0;
                dma_queue_push(weight_dma,
                               dma_make_data(vtcm_res_in[0], bias_addr + (mr * bias_stride) * bias_elem_size),
                               res_in_stride_bytes, bias_row_bytes, cbytes, cbytes ? n_rows : 0);
            }

            for (int i = 0; i < n_chunk_cnt; ++i) {
                const size_t nc     = (size_t) i * n_chunk_n_cols;
                const size_t n_cols = hex_smin((size_t) n - nc, n_chunk_n_cols);

                // 1. Pop w_i and dequantize / rescale into vtcm_weight_bufs[i % 4]
                void * curr_raw;
                if (i == 0 && preissue_w0) {
                    curr_raw = (void *) dma_proxy_pop(ctx->dma_proxy).dst;
                } else {
                    curr_raw = (void *) dma_queue_pop(weight_dma).dst;
                }
                if (is_int_hmx) {
                    rescale_int_weight_chunk_to_int8_tiles(
                        octx, vtcm_weight_bufs[i % 4], L.weight_bias_offset, L.weight_scales_offset,
                        curr_raw, (int) n_cols, k, n_groups, &n_groups_div, aligned_tile_size, weight_type);
                } else {
                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        octx, vtcm_weight_bufs[i % 4], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn);
                }

                if (i == 0) {
                    if (is_int_hmx) {
                        hmx_matmul_task_init_int(&task_slots[0], vtcm_output_bufs[0],
                                                vtcm_f16_act, vtcm_weight_bufs[0], L.weight_bias_offset,
                                                hmx_ceil_div(n_rows, m_tile_rows),
                                                hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS),
                                                (uint32_t) n_k_tiles, (uint32_t) n_groups);
                    } else {
                        hmx_matmul_task_init(&task_slots[0], (__fp16 *) vtcm_output_bufs[0],
                                            (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[0],
                                            vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                            hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), (uint32_t) n_k_tiles);
                    }
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task_slots[0]));

                    if (2 < n_chunk_cnt) {
                        const size_t   nc2       = 2u * n_chunk_n_cols;
                        const size_t   n_cols_p2 = hex_smin((size_t) n - nc2, n_chunk_n_cols);
                        const uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                        dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[2], weight + nc2 * weight_stride),
                                       dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                    }
                    if (has_2d_bias && 1 < n_chunk_cnt) {
                        const size_t nc1    = n_chunk_n_cols;
                        const int    vcols1 = hex_smin((int) hex_smin((size_t) n - nc1, n_chunk_n_cols), dst_cols - (int) nc1);
                        const size_t cbytes = (vcols1 > 0) ? (size_t) vcols1 * bias_elem_size : 0;
                        dma_queue_push(weight_dma,
                                       dma_make_data(vtcm_res_in[1], bias_addr + (mr * bias_stride + nc1) * bias_elem_size),
                                       res_in_stride_bytes, bias_row_bytes, cbytes, cbytes ? n_rows : 0);
                    }
                } else {
                    const int    prev     = (i - 1) & 1;
                    const size_t nc_prev  = (size_t) (i - 1) * n_chunk_n_cols;
                    const size_t n_cols_p = hex_smin((size_t) n - nc_prev, n_chunk_n_cols);

                    // 2. Wait C_{i-1} and submit C_i
                    hmx_queue_pop(ctx->hmx_queue);
                    if (is_int_hmx) {
                        hmx_matmul_task_init_int(&task_slots[i & 1], vtcm_output_bufs[i & 1],
                                                vtcm_f16_act, vtcm_weight_bufs[i % 4], L.weight_bias_offset,
                                                hmx_ceil_div(n_rows, m_tile_rows),
                                                hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS),
                                                (uint32_t) n_k_tiles, (uint32_t) n_groups);
                    } else {
                        hmx_matmul_task_init(&task_slots[i & 1], (__fp16 *) vtcm_output_bufs[i & 1],
                                            (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                            vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                            hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), (uint32_t) n_k_tiles);
                    }
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task_slots[i & 1]));

                    // 3. Pop r_{i-1} (if 2D residual); out_{i-2} uses vtcm_res_out[cur], popped after D_{i-1}
                    if (has_2d_bias) {
                        dma_queue_pop(weight_dma);
                    }

                    // 4. HVX epilogue D_{i-1} (VTCM -> vtcm_res_out[prev], parallel with C_i and out_{i-2} DMA)
                    const int chunk_dst_cols = dst_cols - (int) nc_prev;
                    if (chunk_dst_cols > 0) {
                        const void *   bias_chunk = has_2d_bias ? vtcm_res_in[prev]
                                                  : (has_bias ? ((const uint8_t *) vtcm_bias + (mr * bias_stride + nc_prev) * bias_elem_size) : NULL);
                        const uint32_t ep_bias_stride = has_2d_bias ? res_in_stride_elems : bias_stride;
                        if (is_int_hmx) {
                            const float * wt_scales_prev = (const float *) ((const uint8_t *) vtcm_weight_bufs[(i - 1) % 4] + L.weight_scales_offset);
                            transfer_output_chunk_int_threaded(
                                octx, vtcm_res_out[prev], bias_chunk, vtcm_output_bufs[prev],
                                vtcm_act_scales, wt_scales_prev, (int) n_rows, (int) n_cols_p,
                                n_groups, res_out_stride_elems, ep_bias_stride, chunk_dst_cols,
                                dst_elem_size, bias_elem_size);
                        } else {
                            transfer_output_chunk_threaded(octx, vtcm_res_out[prev], bias_chunk, vtcm_output_bufs[prev],
                                                           n_rows, n_cols_p, res_out_stride_elems, ep_bias_stride, chunk_dst_cols,
                                                           dst_elem_size, bias_elem_size);
                        }
                    }

                    // 5. Drain out_{i-2}, then push out_{i-1}, w_{i+2}, and r_{i+1}
                    if (i >= 2) {
                        dma_queue_pop(weight_dma);
                    }
                    {
                        const int    vcols_o = hex_smin((int) n_cols_p, chunk_dst_cols);
                        const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                        dma_queue_push(weight_dma,
                                       dma_make_data(dst_dma_addr + (mr * dst_stride + nc_prev) * dst_elem_size, vtcm_res_out[prev]),
                                       dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                    }
                    if (i + 2 < n_chunk_cnt) {
                        const size_t   nc_p2     = (size_t) (i + 2) * n_chunk_n_cols;
                        const size_t   n_cols_p2 = hex_smin((size_t) n - nc_p2, n_chunk_n_cols);
                        const uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                        dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                       dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                    }
                    if (has_2d_bias && i + 1 < n_chunk_cnt) {
                        const size_t nc_p1   = (size_t) (i + 1) * n_chunk_n_cols;
                        const int    vcols_r = hex_smin((int) hex_smin((size_t) n - nc_p1, n_chunk_n_cols), dst_cols - (int) nc_p1);
                        const size_t cbytes  = (vcols_r > 0) ? (size_t) vcols_r * bias_elem_size : 0;
                        dma_queue_push(weight_dma,
                                       dma_make_data(vtcm_res_in[prev], bias_addr + (mr * bias_stride + nc_p1) * bias_elem_size),
                                       res_in_stride_bytes, bias_row_bytes, cbytes, cbytes ? n_rows : 0);
                    }
                }
            }

            // Epilogue: wait C_{last}, pop r_{last} and out_{last-1}, run D_{last}, push+pop out_{last}
            {
                const int    last        = (n_chunk_cnt - 1) & 1;
                const size_t nc_last     = (size_t) (n_chunk_cnt - 1) * n_chunk_n_cols;
                const size_t n_cols_last = hex_smin((size_t) n - nc_last, n_chunk_n_cols);

                hmx_queue_pop(ctx->hmx_queue);
                if (has_2d_bias) {
                    dma_queue_pop(weight_dma);
                }
                if (n_chunk_cnt >= 2) {
                    dma_queue_pop(weight_dma);
                }

                const int chunk_dst_cols = dst_cols - (int) nc_last;
                if (chunk_dst_cols > 0) {
                    const void *   bias_chunk = has_2d_bias ? vtcm_res_in[last]
                                              : (has_bias ? ((const uint8_t *) vtcm_bias + (mr * bias_stride + nc_last) * bias_elem_size) : NULL);
                    const uint32_t ep_bias_stride = has_2d_bias ? res_in_stride_elems : bias_stride;
                    if (is_int_hmx) {
                        const float * wt_scales_last = (const float *) ((const uint8_t *) vtcm_weight_bufs[(n_chunk_cnt - 1) % 4] + L.weight_scales_offset);
                        transfer_output_chunk_int_threaded(
                            octx, vtcm_res_out[last], bias_chunk, vtcm_output_bufs[last],
                            vtcm_act_scales, wt_scales_last, (int) n_rows, (int) n_cols_last,
                            n_groups, res_out_stride_elems, ep_bias_stride, chunk_dst_cols,
                            dst_elem_size, bias_elem_size);
                    } else {
                        transfer_output_chunk_threaded(octx, vtcm_res_out[last], bias_chunk, vtcm_output_bufs[last],
                                                       n_rows, n_cols_last, res_out_stride_elems, ep_bias_stride, chunk_dst_cols,
                                                       dst_elem_size, bias_elem_size);
                    }
                }

                const int    vcols_o = hex_smin((int) n_cols_last, chunk_dst_cols);
                const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                dma_queue_push(weight_dma,
                               dma_make_data(dst_dma_addr + (mr * dst_stride + nc_last) * dst_elem_size, vtcm_res_out[last]),
                               dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                dma_queue_pop(weight_dma);
            }
        }
    } else {
        // --- Synchronous loop (m <= 32 or fallback) with lockstep DMA writeback ---
        hmx_matmul_task_t task;
        void *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        void *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        const size_t dst_row_bytes        = (size_t) dst_stride * dst_elem_size;
        const size_t res_out_stride_bytes = hex_align_up(n_chunk_n_cols * dst_elem_size, 128);
        const int    res_out_stride_elems = (int) (res_out_stride_bytes / dst_elem_size);

        const bool preissue_w0 = is_int_hmx && L.can_preissue_weight && n > 0;

        for (size_t mr = 0; mr < (size_t) m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin((size_t) m - mr, m_chunk_n_rows);

            if (preissue_w0) {
                const size_t   n_cols = hex_smin((size_t) n, n_chunk_n_cols);
                const uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                dma_proxy_push(ctx->dma_proxy, dma_make_data(vtcm_raw_bufs[0], weight),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height);
            }

            if (is_int_hmx) {
                transfer_activation_chunk_int_threaded(
                    ctx, (uint8_t *) vtcm_f16_act, vtcm_act_scales,
                    act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                    (int) n_rows, k, act_stride, k_valid, n_groups,
                    act_hvx, act_hvx_div, vtcm_f32_act, L.act_f32_bytes, act_elem_size);
            } else {
                struct activation_transfer_params act_params = {
                    .ctx = ctx,
                    .dst = vtcm_f16_act,
                    .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                    .n_rows = (int) n_rows,
                    .k_block = k,
                    .k_stride = act_stride,
                    .n_hvx = act_hvx,
                    .act_hvx_div = act_hvx_div,
                    .k_div = k_div,
                    .k_valid = k_valid,
                    .vtcm_f32_act = vtcm_f32_act,
                    .vtcm_f32_act_bytes = L.act_f32_bytes,
                    .act_elem_size = act_elem_size,
                };
                transfer_activation_chunk_threaded(&act_params);
            }

            if (!preissue_w0 && n > 0) {
                const size_t n_cols = hex_smin((size_t) n, n_chunk_n_cols);
                const uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
            }

            for (int i = 0; i < n_chunk_cnt; ++i) {
                const size_t nc = (size_t) i * n_chunk_n_cols;
                const size_t n_cols = hex_smin((size_t) n - nc, n_chunk_n_cols);
                const size_t n_row_tiles = hmx_ceil_div(n_rows, m_tile_rows);
                const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // A: Wait for weight DMA (and previous output writeback)
                void * curr_raw;
                if (i == 0 && preissue_w0) {
                    curr_raw = (void *) dma_proxy_pop(ctx->dma_proxy).dst;
                } else {
                    curr_raw = (void *) dma_queue_pop(weight_dma).dst;
                }
                if (i > 0) {
                    dma_queue_pop(weight_dma);
                }

                // B: Weight Dequantize / Rescale (Threaded)
                if (is_int_hmx) {
                    rescale_int_weight_chunk_to_int8_tiles(
                        octx, vtcm_weight_bufs[i % 2], L.weight_bias_offset, L.weight_scales_offset,
                        curr_raw, (int) n_cols, k, n_groups, &n_groups_div, aligned_tile_size, weight_type);
                } else {
                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        octx, vtcm_weight_bufs[i % 2], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn);
                }

                // Start weight DMA for the next chunk early
                const size_t nc_next = nc + n_chunk_n_cols;
                if (nc_next < (size_t) n) {
                    const size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
                    const uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                }

                // C: HMX Compute (Queue-based)
                if (is_int_hmx) {
                    hmx_matmul_task_init_int(&task, vtcm_output, vtcm_f16_act, vtcm_weight_bufs[i % 2], L.weight_bias_offset,
                                            n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles, (uint32_t) n_groups);
                } else {
                    hmx_matmul_task_init(&task, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles);
                }
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task));
                hmx_queue_pop(ctx->hmx_queue);

                // D: Output Store into vtcm_res_out[0] + 2D bypass DMA
                const void *bias_chunk = has_bias ? ((const uint8_t *) vtcm_bias + (mr * bias_stride + nc) * bias_elem_size) : NULL;
                int chunk_dst_cols = dst_cols - (int) nc;
                if (chunk_dst_cols > 0) {
                    if (is_int_hmx) {
                        const float * wt_scales = (const float *) ((const uint8_t *) vtcm_weight_bufs[i % 2] + L.weight_scales_offset);
                        transfer_output_chunk_int_threaded(
                            octx, vtcm_res_out[0], bias_chunk, vtcm_output,
                            vtcm_act_scales, wt_scales, (int) n_rows, (int) n_cols,
                            n_groups, res_out_stride_elems, bias_stride, chunk_dst_cols,
                            dst_elem_size, bias_elem_size);
                    } else {
                        transfer_output_chunk_threaded(octx, vtcm_res_out[0], bias_chunk, vtcm_output,
                                                       n_rows, n_cols, res_out_stride_elems, bias_stride, chunk_dst_cols,
                                                       dst_elem_size, bias_elem_size);
                    }
                }
                const int    vcols_o = hex_smin((int) n_cols, chunk_dst_cols);
                const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                dma_queue_push(weight_dma,
                               dma_make_data(dst_dma_addr + (mr * dst_stride + nc) * dst_elem_size, vtcm_res_out[0]),
                               dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
            }
            if (n_chunk_cnt > 0) {
                dma_queue_pop(weight_dma);
            }
        }
    }

    return 0;
}

static int hmx_mm_nx_glu_int_2d(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_tensor * restrict src_gate = octx->src[0];
    const struct htp_tensor * restrict src_up   = octx->src[1];
    const struct htp_tensor * restrict act      = octx->src[2];
    const struct htp_tensor * restrict dst      = octx->dst;

    const int        weight_type   = (int) src_gate->type;
    const int        k             = (int) act->ne[0];
    const int        k_valid       = (int) act->ne[0];
    const size_t     n             = (size_t) src_gate->ne[1];
    const int        m             = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
    const uint32_t   act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const size_t     dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const int        act_stride    = (int) (act->nb[1] / act_elem_size);
    const dma_addr_t act_dma_addr  = act->data;

    const int32_t kernel_type = kparams->kernel_type;
    const int     n_groups    = kparams->hmx_n_groups;
    const int     tile_k      = htp_tensor_get_tile_k(weight_type);

    if (tile_k <= 0 || (k % tile_k) != 0 || (k & 31) != 0 || !htp_mm_kernel_is_hmx_int(kernel_type)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (n == 0) {
        return HTP_STATUS_OK;
    }

    hmx_output_glu_state_t glu_state;
    if (!resolve_int_output_glu_fns((uint32_t) n_groups, kparams->glu_op, dst_elem_size,
                                    &glu_state.col_chunk_fn, &glu_state.store_quad_fn,
                                    &glu_state.store_tail_fn, &glu_state.store_strip_fn)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct fastdiv_values n_groups_div  = kparams->div_hmx_n_groups;
    const hmx_queue_func        hmx_worker_fn = hmx_get_matmul_worker_fn(kernel_type);

    const int n_k_tiles         = k >> 5;
    const int n_weight_tiles    = (weight_type == HTP_TYPE_Q4_0) ? n_k_tiles : (k / tile_k);
    const int tile_size         = kparams->tile_size;
    const int aligned_tile_size = kparams->aligned_tile_size;
    if (tile_size <= 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const size_t vtcm_budget                  = ctx->vtcm_size;
    const int    m_chunk_n_rows               = kparams->m_chunk;
    const int    n_chunk_n_cols               = kparams->n_chunk;
    const int    pipeline                     = kparams->pipeline;
    const int    act_hvx                      = kparams->n_act_hvx;
    const struct fastdiv_values * act_hvx_div = &kparams->div_n_act_hvx;

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build_glu(&L, kernel_type, weight_type, k,
                                     m_chunk_n_rows, n_chunk_n_cols, pipeline, act_hvx, aligned_tile_size,
                                     dst_elem_size);
    if (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-glu: VTCM overflow: used %zu budget %zu, m %d k %d n %zu mc %d nc %d",
             L.total_bytes, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    void * vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(void, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_weight[1], pipeline)
    };
    uint8_t * vtcm_act        = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    float   * vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    float   * vtcm_act_scales = VTCM_LAYOUT_PTR(float, base, L.off_act_scales);
    void    * vtcm_gate[2]    = {
        VTCM_LAYOUT_PTR(void, base, L.off_dst[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_dst[1], pipeline)
    };
    void    * vtcm_up[2]      = {
        VTCM_LAYOUT_PTR(void, base, L.off_dst_up[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_dst_up[1], pipeline)
    };
    void    * vtcm_scratch[2] = {
        VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_scratch[1], pipeline)
    };
    float   * vtcm_wt_scales_gate[2] = {
        VTCM_LAYOUT_PTR(float, base, L.off_wt_scales_gate[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_wt_scales_gate[1], pipeline)
    };
    float   * vtcm_wt_scales_up[2] = {
        VTCM_LAYOUT_PTR(float, base, L.off_wt_scales_up[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_wt_scales_up[1], pipeline)
    };
    void    * vtcm_res_out[2]   = {
        VTCM_LAYOUT_PTR(void, base, L.off_res_out[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_res_out[1], pipeline)
    };

    if (L.lut_bytes > 0) {
        iq2xxs_lut_copy(base + L.off_lut);
    }

    int m_start = 0;
    int m_rows  = m;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        m_start = (int) range.start;
        m_rows  = (int) range.count;
    }
    if (m_rows == 0) {
        return HTP_STATUS_OK;
    }

    FARF(HIGH, "hmx-mm-nx-glu: glu %u ktype %d m %d (%d..%d) k %d n %zu wtype %d mc %d nc %d vtcm %zu/%zu",
         (unsigned) kparams->glu_op, (int) kernel_type, m, m_start, m_start + m_rows, k, n, weight_type,
         m_chunk_n_rows, n_chunk_n_cols, L.total_bytes, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    const size_t     mr_end               = (size_t) (m_start + m_rows);
    const dma_addr_t w_gate               = src_gate->data;
    const dma_addr_t w_up                 = src_up->data;
    const size_t     w_gate_stride        = src_gate->nb[1];
    const size_t     w_up_stride          = src_up->nb[1];
    const dma_addr_t dst_dma_addr         = dst->data;
    const size_t     dst_stride           = dst->nb[1] / dst_elem_size;
    const size_t     dst_row_bytes        = dst->nb[1];
    const int        dst_cols             = (int) dst->ne[0];
    const size_t     res_out_stride_bytes = hex_align_up((size_t) n_chunk_n_cols * dst_elem_size, 128);
    const int        res_out_stride_elems = (int) (res_out_stride_bytes / dst_elem_size);
    const int        n_chunk_cnt          = hmx_ceil_div(n, n_chunk_n_cols);
    dma_queue *      weight_dma           = octx->ctx->dma[0];

    glu_state.vtcm_act_scales = vtcm_act_scales;
    glu_state.n_groups        = (uint32_t) n_groups;
    glu_state.dst_stride      = (uint32_t) res_out_stride_elems;
    glu_state.dst_elem_size   = dst_elem_size;

    if (pipeline) {
        // 4-buffer HMX output pipeline: vtcm_gate[2] + vtcm_up[2] (GLU(s-1) overlaps both HMX(gate_s) and HMX(up_s))
        hmx_matmul_task_t task_gate[2];
        hmx_matmul_task_t task_up[2];
        const bool preissue_gate0 = L.can_preissue_weight && n > 0;

        for (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            const size_t n_rows      = hex_smin(mr_end - mr, m_chunk_n_rows);
            const size_t n_row_tiles = hmx_ceil_div(n_rows, 64u);

            if (preissue_gate0) {
                const size_t   n_cols0 = hex_smin(n, (size_t) n_chunk_n_cols);
                const uint32_t height0 = hmx_ceil_div(n_cols0, 32) * n_weight_tiles;
                dma_proxy_push(ctx->dma_proxy, dma_make_data(vtcm_weight_raw[0], w_gate),
                               aligned_tile_size, tile_size, tile_size, height0);
            }

            transfer_activation_chunk_int_threaded(
                ctx, vtcm_act, vtcm_act_scales,
                act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                (int) n_rows, k, act_stride, k_valid, n_groups,
                act_hvx, act_hvx_div, vtcm_f32_act, L.act_f32_bytes, act_elem_size);

            // Prologue: push gate_0 (if not preissued) and up_0
            {
                const size_t   n_cols0 = hex_smin(n, (size_t) n_chunk_n_cols);
                const uint32_t height0 = hmx_ceil_div(n_cols0, 32) * n_weight_tiles;
                if (!preissue_gate0) {
                    dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], w_gate),
                                   aligned_tile_size, tile_size, tile_size, height0);
                }
                dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[1], w_up),
                               aligned_tile_size, tile_size, tile_size, height0);
            }

            for (int s = 0; s < n_chunk_cnt; ++s) {
                const int    cur         = s & 1;
                const int    prev        = (s - 1) & 1;
                const size_t nc          = (size_t) s * n_chunk_n_cols;
                const size_t n_cols      = hex_smin(n - nc, (size_t) n_chunk_n_cols);
                const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // 1. Pop gate_s, wait HMX(gate_{s-1}) so vtcm_scratch[0] is free, rescale and submit HMX(gate_s)
                if (s == 0 && preissue_gate0) {
                    dma_proxy_pop(ctx->dma_proxy);
                } else {
                    dma_queue_pop(weight_dma);
                }
                if (s > 0) {
                    hmx_queue_pop(ctx->hmx_queue);
                }
                rescale_int_weight_chunk_to_int8_tiles_scales(
                    octx, vtcm_scratch[0], L.weight_bias_offset, vtcm_wt_scales_gate[cur],
                    vtcm_weight_raw[0], (int) n_cols, k, n_groups, &n_groups_div, aligned_tile_size, weight_type);

                hmx_matmul_task_init_int(&task_gate[cur], vtcm_gate[cur],
                                        vtcm_act, vtcm_scratch[0], L.weight_bias_offset,
                                        n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles, (uint32_t) n_groups);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task_gate[cur]));

                if (s + 1 < n_chunk_cnt) {
                    const size_t   nc_n     = (size_t) (s + 1) * n_chunk_n_cols;
                    const size_t   n_cols_n = hex_smin(n - nc_n, (size_t) n_chunk_n_cols);
                    const uint32_t height_n = hmx_ceil_div(n_cols_n, 32) * n_weight_tiles;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], w_gate + nc_n * w_gate_stride),
                                   aligned_tile_size, tile_size, tile_size, height_n);
                }

                // 2. Pop up_s, wait HMX(up_{s-1}) so vtcm_scratch[1] and GLU(s-1) inputs are ready, rescale and submit HMX(up_s)
                dma_queue_pop(weight_dma);
                if (s > 0) {
                    hmx_queue_pop(ctx->hmx_queue);
                }
                rescale_int_weight_chunk_to_int8_tiles_scales(
                    octx, vtcm_scratch[1], L.weight_bias_offset, vtcm_wt_scales_up[cur],
                    vtcm_weight_raw[1], (int) n_cols, k, n_groups, &n_groups_div, aligned_tile_size, weight_type);

                hmx_matmul_task_init_int(&task_up[cur], vtcm_up[cur],
                                        vtcm_act, vtcm_scratch[1], L.weight_bias_offset,
                                        n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles, (uint32_t) n_groups);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task_up[cur]));

                if (s + 1 < n_chunk_cnt) {
                    const size_t   nc_n     = (size_t) (s + 1) * n_chunk_n_cols;
                    const size_t   n_cols_n = hex_smin(n - nc_n, (size_t) n_chunk_n_cols);
                    const uint32_t height_n = hmx_ceil_div(n_cols_n, 32) * n_weight_tiles;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[1], w_up + nc_n * w_up_stride),
                                   aligned_tile_size, tile_size, tile_size, height_n);
                }

                // 3. Run HVX GLU(s-1) in parallel with HMX(gate_s) + HMX(up_s) + out_{s-2} DMA, then drain out_{s-2} and push out_{s-1}
                if (s > 0) {
                    const size_t nc_prev        = (size_t) (s - 1) * n_chunk_n_cols;
                    const size_t n_cols_p       = hex_smin(n - nc_prev, (size_t) n_chunk_n_cols);
                    const int    chunk_dst_cols = dst_cols - (int) nc_prev;
                    if (chunk_dst_cols > 0) {
                        transfer_output_chunk_int_glu_threaded(
                            octx, &glu_state, vtcm_res_out[prev], vtcm_gate[prev], vtcm_up[prev],
                            vtcm_wt_scales_gate[prev], vtcm_wt_scales_up[prev],
                            (int) n_rows, (int) n_cols_p, chunk_dst_cols);
                    }
                    if (s >= 2) {
                        dma_queue_pop(weight_dma);
                    }
                    const int    vcols_o = hex_smin((int) n_cols_p, chunk_dst_cols);
                    const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                    dma_queue_push(weight_dma,
                                   dma_make_data(dst_dma_addr + (mr * dst_stride + nc_prev) * dst_elem_size, vtcm_res_out[prev]),
                                   dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                }
            }

            // Epilogue: wait HMX(gate_{last}) + HMX(up_{last}), drain out_{last-1}, run GLU(last), push+pop out_{last}
            {
                const int    last           = (n_chunk_cnt - 1) & 1;
                const size_t nc_last        = (size_t) (n_chunk_cnt - 1) * n_chunk_n_cols;
                const size_t n_cols_last    = hex_smin(n - nc_last, (size_t) n_chunk_n_cols);
                const int    chunk_dst_cols = dst_cols - (int) nc_last;

                hmx_queue_pop(ctx->hmx_queue);
                hmx_queue_pop(ctx->hmx_queue);
                if (n_chunk_cnt >= 2) {
                    dma_queue_pop(weight_dma);
                }

                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_int_glu_threaded(
                        octx, &glu_state, vtcm_res_out[last], vtcm_gate[last], vtcm_up[last],
                        vtcm_wt_scales_gate[last], vtcm_wt_scales_up[last],
                        (int) n_rows, (int) n_cols_last, chunk_dst_cols);
                }

                const int    vcols_o = hex_smin((int) n_cols_last, chunk_dst_cols);
                const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                dma_queue_push(weight_dma,
                               dma_make_data(dst_dma_addr + (mr * dst_stride + nc_last) * dst_elem_size, vtcm_res_out[last]),
                               dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                dma_queue_pop(weight_dma);
            }
        }
    } else {
        // Non-pipelined 2-buffer path (m <= 32)
        hmx_matmul_task_t task;

        for (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            const size_t n_rows      = hex_smin(mr_end - mr, m_chunk_n_rows);
            const size_t n_row_tiles = hmx_ceil_div(n_rows, 64u);

            transfer_activation_chunk_int_threaded(
                ctx, vtcm_act, vtcm_act_scales,
                act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                (int) n_rows, k, act_stride, k_valid, n_groups,
                act_hvx, act_hvx_div, vtcm_f32_act, L.act_f32_bytes, act_elem_size);

            for (int s = 0; s < n_chunk_cnt; ++s) {
                const size_t   nc          = (size_t) s * n_chunk_n_cols;
                const size_t   n_cols      = hex_smin(n - nc, (size_t) n_chunk_n_cols);
                const size_t   n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
                const uint32_t height      = hmx_ceil_div(n_cols, 32) * n_weight_tiles;

                dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], w_gate + nc * w_gate_stride),
                               aligned_tile_size, tile_size, tile_size, height);
                dma_queue_pop(weight_dma);
                rescale_int_weight_chunk_to_int8_tiles_scales(
                    octx, vtcm_scratch[0], L.weight_bias_offset, vtcm_wt_scales_gate[0],
                    vtcm_weight_raw[0], (int) n_cols, k, n_groups, &n_groups_div, aligned_tile_size, weight_type);
                hmx_matmul_task_init_int(&task, vtcm_gate[0], vtcm_act, vtcm_scratch[0], L.weight_bias_offset,
                                        n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles, (uint32_t) n_groups);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task));
                hmx_queue_pop(ctx->hmx_queue);

                dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], w_up + nc * w_up_stride),
                               aligned_tile_size, tile_size, tile_size, height);
                dma_queue_pop(weight_dma);
                rescale_int_weight_chunk_to_int8_tiles_scales(
                    octx, vtcm_scratch[0], L.weight_bias_offset, vtcm_wt_scales_up[0],
                    vtcm_weight_raw[0], (int) n_cols, k, n_groups, &n_groups_div, aligned_tile_size, weight_type);
                hmx_matmul_task_init_int(&task, vtcm_up[0], vtcm_act, vtcm_scratch[0], L.weight_bias_offset,
                                        n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles, (uint32_t) n_groups);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task));
                hmx_queue_pop(ctx->hmx_queue);

                const int chunk_dst_cols = dst_cols - (int) nc;
                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_int_glu_threaded(
                        octx, &glu_state, vtcm_res_out[0], vtcm_gate[0], vtcm_up[0],
                        vtcm_wt_scales_gate[0], vtcm_wt_scales_up[0],
                        (int) n_rows, (int) n_cols, chunk_dst_cols);
                }
                const int    vcols_o = hex_smin((int) n_cols, chunk_dst_cols);
                const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                dma_queue_push(weight_dma,
                               dma_make_data(dst_dma_addr + (mr * dst_stride + nc) * dst_elem_size, vtcm_res_out[0]),
                               dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                dma_queue_pop(weight_dma);
            }
        }
    }

    return HTP_STATUS_OK;
}

static int hmx_mm_nx_glu_fp16_2d(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_tensor * restrict src_gate = octx->src[0];
    const struct htp_tensor * restrict src_up   = octx->src[1];
    const struct htp_tensor * restrict act      = octx->src[2];
    const struct htp_tensor * restrict dst      = octx->dst;

    const int        weight_type   = (int) src_gate->type;
    const int        k             = (int) act->ne[0];
    const int        k_valid       = (int) act->ne[0];
    const size_t     n             = (size_t) src_gate->ne[1];
    const int        m             = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
    const uint32_t   act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const size_t     dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const int        act_stride    = (int) (act->nb[1] / act_elem_size);
    const dma_addr_t act_dma_addr  = act->data;

    const int32_t kernel_type = kparams->kernel_type;
    const int     tile_size   = kparams->tile_size;
    const int     aligned_tile_size = kparams->aligned_tile_size;
    const bool    is_tiled    = (tile_size > 0);
    const int     tile_k      = htp_tensor_get_tile_k(weight_type);

    if ((k & 31) != 0 || (is_tiled && (tile_k <= 0 || (k % tile_k) != 0)) || kernel_type != HTP_MM_KERNEL_HMX_2D) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (n == 0) {
        return HTP_STATUS_OK;
    }

    const size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    if (row_stride == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    hmx_output_glu_state_t glu_state;
    if (!resolve_fp16_output_glu_fns(kparams->glu_op, dst_elem_size,
                                     &glu_state.col_chunk_fn, &glu_state.store_quad_fn,
                                     &glu_state.store_tail_fn, &glu_state.store_strip_fn)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const hmx_queue_func        hmx_worker_fn     = hmx_get_matmul_worker_fn(kernel_type);
    const int                   n_k_tiles         = k >> 5;
    const int                   n_weight_tiles    = is_tiled ? ((weight_type == HTP_TYPE_Q4_0) ? n_k_tiles : (k / tile_k)) : 0;
    const struct fastdiv_values n_k_tiles_div     = kparams->div_n_k_tiles;
    tiled_dequant_task_fn_t     dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    const size_t vtcm_budget                  = ctx->vtcm_size;
    const int    m_chunk_n_rows               = kparams->m_chunk;
    const int    n_chunk_n_cols               = kparams->n_chunk;
    const int    pipeline                     = kparams->pipeline;
    const int    act_hvx                      = kparams->n_act_hvx;
    const struct fastdiv_values * act_hvx_div = &kparams->div_n_act_hvx;
    const struct fastdiv_values * k_div       = &kparams->div_ne00_padded;

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build_glu(&L, kernel_type, weight_type, k,
                                     m_chunk_n_rows, n_chunk_n_cols, pipeline, act_hvx, aligned_tile_size,
                                     dst_elem_size);
    if (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-glu-fp16: VTCM overflow: used %zu budget %zu, m %d k %d n %zu mc %d nc %d",
             L.total_bytes, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    void * vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(void, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_weight[1], pipeline)
    };
    __fp16 * vtcm_f16_act = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    float  * vtcm_f32_act = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    __fp16 * vtcm_scales  = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    __fp16 * vtcm_gate[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_dst[1], pipeline)
    };
    __fp16 * vtcm_up[2]   = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_dst_up[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_dst_up[1], pipeline)
    };
    void   * vtcm_scratch[2] = {
        VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_scratch[1], pipeline)
    };
    void   * vtcm_res_out[2] = {
        VTCM_LAYOUT_PTR(void, base, L.off_res_out[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_res_out[1], pipeline)
    };

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00)); // scale: 1.0, bias: 0.0 in FP16
    if (L.lut_bytes > 0) {
        iq2xxs_lut_copy(base + L.off_lut);
    }

    int m_start = 0;
    int m_rows  = m;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        m_start = (int) range.start;
        m_rows  = (int) range.count;
    }
    if (m_rows == 0) {
        return HTP_STATUS_OK;
    }

    FARF(HIGH, "hmx-mm-nx-glu-fp16: glu %u m %d (%d..%d) k %d n %zu wtype %d mc %d nc %d vtcm %zu/%zu",
         (unsigned) kparams->glu_op, m, m_start, m_start + m_rows, k, n, weight_type,
         m_chunk_n_rows, n_chunk_n_cols, L.total_bytes, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    const size_t     mr_end               = (size_t) (m_start + m_rows);
    const dma_addr_t w_gate               = src_gate->data;
    const dma_addr_t w_up                 = src_up->data;
    const size_t     w_gate_stride        = src_gate->nb[1];
    const size_t     w_up_stride          = src_up->nb[1];
    const dma_addr_t dst_dma_addr         = dst->data;
    const size_t     dst_stride           = dst->nb[1] / dst_elem_size;
    const size_t     dst_row_bytes        = dst->nb[1];
    const int        dst_cols             = (int) dst->ne[0];
    const size_t     res_out_stride_bytes = hex_align_up((size_t) n_chunk_n_cols * dst_elem_size, 128);
    const int        res_out_stride_elems = (int) (res_out_stride_bytes / dst_elem_size);
    const int        n_chunk_cnt          = hmx_ceil_div(n, n_chunk_n_cols);
    const uint32_t   dma_dst_stride       = is_tiled ? (uint32_t) aligned_tile_size : (uint32_t) row_stride;
    const uint32_t   dma_width_bytes      = is_tiled ? (uint32_t) tile_size : (uint32_t) row_stride;
    const uint32_t   dma_src_stride_g     = is_tiled ? (uint32_t) tile_size : (uint32_t) w_gate_stride;
    const uint32_t   dma_src_stride_u     = is_tiled ? (uint32_t) tile_size : (uint32_t) w_up_stride;
    dma_queue *      weight_dma           = octx->ctx->dma[0];

    glu_state.vtcm_act_scales   = NULL;
    glu_state.vtcm_wt_scales_up = NULL;
    glu_state.n_groups          = 1u;
    glu_state.dst_stride        = (uint32_t) res_out_stride_elems;
    glu_state.dst_elem_size     = dst_elem_size;

    if (pipeline) {
        // When dequant_worker_fn is NULL (pre-tiled F16), ping-pong raw/HMX across {weight_raw, scratch}.
        void * gate_raw_buf[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch[0] };
        void * up_raw_buf[2]   = { vtcm_weight_raw[1], dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch[1] };
        void * gate_hmx_buf[2] = { dequant_worker_fn ? vtcm_scratch[0] : vtcm_weight_raw[0], vtcm_scratch[0] };
        void * up_hmx_buf[2]   = { dequant_worker_fn ? vtcm_scratch[1] : vtcm_weight_raw[1], vtcm_scratch[1] };

        hmx_matmul_task_t task_gate[2];
        hmx_matmul_task_t task_up[2];

        for (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            const size_t n_rows      = hex_smin(mr_end - mr, m_chunk_n_rows);
            const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

            struct activation_transfer_params act_params = {
                .ctx                = ctx,
                .dst                = vtcm_f16_act,
                .act_dma_addr       = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                .n_rows             = (int) n_rows,
                .k_block            = k,
                .k_stride           = act_stride,
                .n_hvx              = act_hvx,
                .act_hvx_div        = act_hvx_div,
                .k_div              = k_div,
                .k_valid            = k_valid,
                .vtcm_f32_act       = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_size      = act_elem_size,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push gate_0 and up_0
            {
                const size_t   n_cols0 = hex_smin(n, (size_t) n_chunk_n_cols);
                const uint32_t height0 = is_tiled ? (hmx_ceil_div(n_cols0, 32) * n_weight_tiles) : (uint32_t) n_cols0;
                dma_queue_push(weight_dma, dma_make_data(gate_raw_buf[0], w_gate),
                               dma_dst_stride, dma_src_stride_g, dma_width_bytes, height0);
                dma_queue_push(weight_dma, dma_make_data(up_raw_buf[0], w_up),
                               dma_dst_stride, dma_src_stride_u, dma_width_bytes, height0);
            }

            for (int s = 0; s < n_chunk_cnt; ++s) {
                const int    cur         = s & 1;
                const int    next        = (s + 1) & 1;
                const int    prev        = (s - 1) & 1;
                const size_t nc          = (size_t) s * n_chunk_n_cols;
                const size_t n_cols      = hex_smin(n - nc, (size_t) n_chunk_n_cols);
                const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // 1. Pop gate_s, wait HMX(gate_{s-1}) so gate_hmx_buf[cur] is free, dequant and submit HMX(gate_s)
                dma_queue_pop(weight_dma);
                if (s > 0) {
                    hmx_queue_pop(ctx->hmx_queue);
                }
                if (dequant_worker_fn) {
                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        octx, gate_hmx_buf[cur], gate_raw_buf[cur],
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn);
                }

                hmx_matmul_task_init(&task_gate[cur], vtcm_gate[cur], vtcm_f16_act, (__fp16 *) gate_hmx_buf[cur],
                                    vtcm_scales, n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task_gate[cur]));

                if (s + 1 < n_chunk_cnt) {
                    const size_t   nc_n     = (size_t) (s + 1) * n_chunk_n_cols;
                    const size_t   n_cols_n = hex_smin(n - nc_n, (size_t) n_chunk_n_cols);
                    const uint32_t height_n = is_tiled ? (hmx_ceil_div(n_cols_n, 32) * n_weight_tiles) : (uint32_t) n_cols_n;
                    dma_queue_push(weight_dma, dma_make_data(gate_raw_buf[next], w_gate + nc_n * w_gate_stride),
                                   dma_dst_stride, dma_src_stride_g, dma_width_bytes, height_n);
                }

                // 2. Pop up_s, wait HMX(up_{s-1}) so up_hmx_buf[cur] and GLU(s-1) inputs are ready, dequant and submit HMX(up_s)
                dma_queue_pop(weight_dma);
                if (s > 0) {
                    hmx_queue_pop(ctx->hmx_queue);
                }
                if (dequant_worker_fn) {
                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        octx, up_hmx_buf[cur], up_raw_buf[cur],
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn);
                }

                hmx_matmul_task_init(&task_up[cur], vtcm_up[cur], vtcm_f16_act, (__fp16 *) up_hmx_buf[cur],
                                    vtcm_scales, n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task_up[cur]));

                if (s + 1 < n_chunk_cnt) {
                    const size_t   nc_n     = (size_t) (s + 1) * n_chunk_n_cols;
                    const size_t   n_cols_n = hex_smin(n - nc_n, (size_t) n_chunk_n_cols);
                    const uint32_t height_n = is_tiled ? (hmx_ceil_div(n_cols_n, 32) * n_weight_tiles) : (uint32_t) n_cols_n;
                    dma_queue_push(weight_dma, dma_make_data(up_raw_buf[next], w_up + nc_n * w_up_stride),
                                   dma_dst_stride, dma_src_stride_u, dma_width_bytes, height_n);
                }

                // 3. Run HVX GLU(s-1) in parallel with HMX(gate_s) + HMX(up_s) + out_{s-2} DMA, then drain out_{s-2} and push out_{s-1}
                if (s > 0) {
                    const size_t nc_prev        = (size_t) (s - 1) * n_chunk_n_cols;
                    const size_t n_cols_p       = hex_smin(n - nc_prev, (size_t) n_chunk_n_cols);
                    const int    chunk_dst_cols = dst_cols - (int) nc_prev;
                    if (chunk_dst_cols > 0) {
                        transfer_output_chunk_int_glu_threaded(
                            octx, &glu_state, vtcm_res_out[prev], vtcm_gate[prev], vtcm_up[prev],
                            NULL, NULL, (int) n_rows, (int) n_cols_p, chunk_dst_cols);
                    }
                    if (s >= 2) {
                        dma_queue_pop(weight_dma);
                    }
                    const int    vcols_o = hex_smin((int) n_cols_p, chunk_dst_cols);
                    const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                    dma_queue_push(weight_dma,
                                   dma_make_data(dst_dma_addr + (mr * dst_stride + nc_prev) * dst_elem_size, vtcm_res_out[prev]),
                                   dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                }
            }

            // Epilogue: wait HMX(gate_{last}) + HMX(up_{last}), drain out_{last-1}, run GLU(last), push+pop out_{last}
            {
                const int    last           = (n_chunk_cnt - 1) & 1;
                const size_t nc_last        = (size_t) (n_chunk_cnt - 1) * n_chunk_n_cols;
                const size_t n_cols_last    = hex_smin(n - nc_last, (size_t) n_chunk_n_cols);
                const int    chunk_dst_cols = dst_cols - (int) nc_last;

                hmx_queue_pop(ctx->hmx_queue);
                hmx_queue_pop(ctx->hmx_queue);
                if (n_chunk_cnt >= 2) {
                    dma_queue_pop(weight_dma);
                }

                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_int_glu_threaded(
                        octx, &glu_state, vtcm_res_out[last], vtcm_gate[last], vtcm_up[last],
                        NULL, NULL, (int) n_rows, (int) n_cols_last, chunk_dst_cols);
                }

                const int    vcols_o = hex_smin((int) n_cols_last, chunk_dst_cols);
                const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                dma_queue_push(weight_dma,
                               dma_make_data(dst_dma_addr + (mr * dst_stride + nc_last) * dst_elem_size, vtcm_res_out[last]),
                               dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                dma_queue_pop(weight_dma);
            }
        }
    } else {
        // Non-pipelined 2-buffer path (m <= 32)
        void * w_hmx = dequant_worker_fn ? vtcm_scratch[0] : vtcm_weight_raw[0];
        hmx_matmul_task_t task;

        for (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            const size_t n_rows      = hex_smin(mr_end - mr, m_chunk_n_rows);
            const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

            struct activation_transfer_params act_params = {
                .ctx                = ctx,
                .dst                = vtcm_f16_act,
                .act_dma_addr       = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                .n_rows             = (int) n_rows,
                .k_block            = k,
                .k_stride           = act_stride,
                .n_hvx              = act_hvx,
                .act_hvx_div        = act_hvx_div,
                .k_div              = k_div,
                .k_valid            = k_valid,
                .vtcm_f32_act       = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_size      = act_elem_size,
            };
            transfer_activation_chunk_threaded(&act_params);

            for (int s = 0; s < n_chunk_cnt; ++s) {
                const size_t   nc          = (size_t) s * n_chunk_n_cols;
                const size_t   n_cols      = hex_smin(n - nc, (size_t) n_chunk_n_cols);
                const size_t   n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
                const uint32_t height      = is_tiled ? (hmx_ceil_div(n_cols, 32) * n_weight_tiles) : (uint32_t) n_cols;

                dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], w_gate + nc * w_gate_stride),
                               dma_dst_stride, dma_src_stride_g, dma_width_bytes, height);
                dma_queue_pop(weight_dma);
                if (dequant_worker_fn) {
                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        octx, w_hmx, vtcm_weight_raw[0],
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn);
                }
                hmx_matmul_task_init(&task, vtcm_gate[0], vtcm_f16_act, (__fp16 *) w_hmx,
                                    vtcm_scales, n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task));
                hmx_queue_pop(ctx->hmx_queue);

                dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], w_up + nc * w_up_stride),
                               dma_dst_stride, dma_src_stride_u, dma_width_bytes, height);
                dma_queue_pop(weight_dma);
                if (dequant_worker_fn) {
                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        octx, w_hmx, vtcm_weight_raw[0],
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn);
                }
                hmx_matmul_task_init(&task, vtcm_up[0], vtcm_f16_act, (__fp16 *) w_hmx,
                                    vtcm_scales, n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task));
                hmx_queue_pop(ctx->hmx_queue);

                const int chunk_dst_cols = dst_cols - (int) nc;
                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_int_glu_threaded(
                        octx, &glu_state, vtcm_res_out[0], vtcm_gate[0], vtcm_up[0],
                        NULL, NULL, (int) n_rows, (int) n_cols, chunk_dst_cols);
                }
                const int    vcols_o = hex_smin((int) n_cols, chunk_dst_cols);
                const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                dma_queue_push(weight_dma,
                               dma_make_data(dst_dma_addr + (mr * dst_stride + nc) * dst_elem_size, vtcm_res_out[0]),
                               dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                dma_queue_pop(weight_dma);
            }
        }
    }

    return HTP_STATUS_OK;
}

static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const uint32_t n_weights = kparams->n_weights;
    if (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];

    const int weight_type = (int) src0->type;
    const int k           = (int) act->ne[0];
    const int k_valid     = (int) act->ne[0];
    const int m           = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
    const uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const int act_stride  = (int) (act->nb[1] / act_elem_size);
    const dma_addr_t act_dma_addr = act->data;

    const int32_t kernel_type = kparams->kernel_type;
    const bool is_int_hmx     = htp_mm_kernel_is_hmx_int(kernel_type);
    const int  n_groups       = is_int_hmx ? kparams->hmx_n_groups : 1;

    if ((k & 31) != 0) { return HTP_STATUS_NO_SUPPORT; }

    const struct fastdiv_values n_groups_div = kparams->div_hmx_n_groups;
    const hmx_queue_func hmx_worker_fn = hmx_get_matmul_worker_fn(kernel_type);

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    if (row_stride == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int n_k_tiles = k >> 5;
    const int n_weight_tiles = (weight_type == HTP_TYPE_Q4_0) ? n_k_tiles : (k / htp_tensor_get_tile_k(weight_type));
    const struct fastdiv_values n_k_tiles_div = kparams->div_n_k_tiles;

    const int tile_size       = kparams->tile_size;
    const int aligned_tile_size = kparams->aligned_tile_size;

    const bool is_tiled       = (tile_size > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);
    const size_t vtcm_budget  = ctx->vtcm_size;

    const int m_chunk_n_rows  = kparams->m_chunk;
    const int n_chunk_n_cols  = kparams->n_chunk;
    const int pipeline        = kparams->pipeline;
    const int act_hvx         = kparams->n_act_hvx;
    const struct fastdiv_values * act_hvx_div     = &kparams->div_n_act_hvx;
    const struct fastdiv_values * k_div           = &kparams->div_ne00_padded;

    const uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    const uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    size_t max_dst_elem_size = sizeof(__fp16);
    for (uint32_t p = 0; p < n_weights; ++p) {
        if (octx->dsts[p]) {
            const size_t es = (octx->dsts[p]->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
            if (es > max_dst_elem_size) max_dst_elem_size = es;
        }
    }

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build_ex(&L, kernel_type, weight_type, k,
                                    m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_hvx, aligned_tile_size, 0,
                                    0, max_dst_elem_size);

    if (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.total_bytes, vtcm_budget, m, k, m_chunk_n_rows, n_chunk_n_cols);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    float   *vtcm_act_scales = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_act_scales, is_int_hmx);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    void    *vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    void    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_scratch[1], pipeline);
    void    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_dst[1], pipeline);
    void    *vtcm_res_out[2] = {
        VTCM_LAYOUT_PTR(void, base, L.off_res_out[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_res_out[1], pipeline),
    };
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_scales, !is_int_hmx);

    if (!is_int_hmx) {
        hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16
    }
    if (L.lut_bytes > 0) {
        iq2xxs_lut_copy(base + L.off_lut);
    }

    int m_start = 0;
    int m_rows  = m;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        m_start = (int) range.start;
        m_rows  = (int) range.count;
    }

    if (m_rows == 0) {
        return HTP_STATUS_OK;
    }

    FARF(HIGH, "hmx-mm-nx-2d: ktype %d n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         (int) kernel_type, n_weights, m, m_start, m_start + m_rows, k, weight_type, m_chunk_n_rows, n_chunk_n_cols, L.total_bytes, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    const size_t mr_end = (size_t)(m_start + m_rows);
    const uint32_t m_tile_rows = is_int_hmx ? 64u : HTP_MM_HMX_TILE_N_ROWS;

    if (pipeline) {
        hmx_matmul_task_t task_slots[2];
        void *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        void *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        void *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        const bool preissue_p0 = is_int_hmx && L.can_preissue_weight && n_weights > 0
                                 && octx->src[0] && octx->dsts[0] && octx->src[0]->ne[1] > 0;

        for (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            if (preissue_p0) {
                const struct htp_tensor * src_w0    = octx->src[0];
                const size_t              n0        = src_w0->ne[1];
                const size_t              n_cols_A0 = hex_smin(n0, (size_t) n_chunk_n_cols);
                const uint32_t            height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
                const uint32_t            src_str0  = is_tiled ? tile_size : src_w0->nb[1];
                dma_proxy_push(ctx->dma_proxy, dma_make_data(vtcm_raw_bufs[0], src_w0->data),
                               dma_dst_stride, src_str0, dma_width_bytes, height_A0);
            }

            if (is_int_hmx) {
                transfer_activation_chunk_int_threaded(
                    ctx, (uint8_t *) vtcm_f16_act, vtcm_act_scales,
                    act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                    (int) n_rows, k, act_stride, k_valid, n_groups,
                    act_hvx, act_hvx_div, vtcm_f32_act, L.act_f32_bytes, act_elem_size);
            } else {
                struct activation_transfer_params act_params = {
                    .ctx = ctx,
                    .dst = vtcm_f16_act,
                    .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                    .n_rows = (int) n_rows,
                    .k_block = k,
                    .k_stride = act_stride,
                    .n_hvx = act_hvx,
                    .act_hvx_div = act_hvx_div,
                    .k_div = k_div,
                    .k_valid = k_valid,
                    .vtcm_f32_act = vtcm_f32_act,
                    .vtcm_f32_act_bytes = L.act_f32_bytes,
                    .act_elem_size = act_elem_size,
                };
                transfer_activation_chunk_threaded(&act_params);
            }

            for (uint32_t p = 0; p < n_weights; p++) {
                const struct htp_tensor * restrict src_w = octx->src[p];
                const struct htp_tensor * restrict dst   = octx->dsts[p];
                if (!src_w || !dst) continue;

                const dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma        = octx->ctx->dma[0];
                const dma_addr_t dst_dma_addr = dst->data;
                const size_t n                = src_w->ne[1];
                if (n == 0) continue;
                const size_t weight_stride    = src_w->nb[1];
                const size_t dst_elem_size    = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
                const size_t dst_stride       = dst->nb[1] / dst_elem_size;
                const size_t dst_row_bytes    = dst->nb[1];
                const size_t res_out_stride_bytes = hex_align_up((size_t) n_chunk_n_cols * dst_elem_size, 128);
                const int    res_out_stride_elems = (int) (res_out_stride_bytes / dst_elem_size);
                const int dst_cols            = (int) dst->ne[0];
                const int n_chunk_cnt         = hmx_ceil_div(n, n_chunk_n_cols);

                const uint32_t dma_src_stride = is_tiled ? tile_size : weight_stride;
                const bool     p_preissued    = (p == 0 && preissue_p0);

                if (!p_preissued) {
                    const size_t   n_cols_A0 = hex_smin(n, (size_t) n_chunk_n_cols);
                    const uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);
                }

                if (1 < n_chunk_cnt) {
                    const size_t   n_cols_A1 = hex_smin(n - (size_t) n_chunk_n_cols, (size_t) n_chunk_n_cols);
                    const uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[1], weight + (size_t) n_chunk_n_cols * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
                }

                for (int i = 0; i < n_chunk_cnt; ++i) {
                    const size_t nc     = (size_t) i * n_chunk_n_cols;
                    const size_t n_cols = hex_smin(n - nc, (size_t) n_chunk_n_cols);

                    void * curr_raw;
                    if (i == 0 && p_preissued) {
                        curr_raw = (void *) dma_proxy_pop(ctx->dma_proxy).dst;
                    } else {
                        curr_raw = (void *) dma_queue_pop(weight_dma).dst;
                    }

                    if (is_int_hmx) {
                        rescale_int_weight_chunk_to_int8_tiles(
                            octx, vtcm_weight_bufs[i % 4], L.weight_bias_offset, L.weight_scales_offset,
                            curr_raw, (int) n_cols, k, n_groups, &n_groups_div, aligned_tile_size, weight_type);
                    } else {
                        dequantize_tiled_weight_chunk_to_fp16_tiles(
                            octx, vtcm_weight_bufs[i % 4], curr_raw,
                            n_cols, k, row_stride, weight_type,
                            n_k_tiles, n_k_tiles_div, dequant_worker_fn);
                    }

                    if (i == 0) {
                        if (is_int_hmx) {
                            hmx_matmul_task_init_int(&task_slots[0], vtcm_output_bufs[0],
                                                    vtcm_f16_act, vtcm_weight_bufs[0], L.weight_bias_offset,
                                                    hmx_ceil_div(n_rows, m_tile_rows),
                                                    hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS),
                                                    (uint32_t) n_k_tiles, (uint32_t) n_groups);
                        } else {
                            hmx_matmul_task_init(&task_slots[0], (__fp16 *) vtcm_output_bufs[0],
                                                (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[0],
                                                vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                                hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), (uint32_t) n_k_tiles);
                        }
                        hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task_slots[0]));

                        if (2 < n_chunk_cnt) {
                            const size_t   nc2       = 2u * (size_t) n_chunk_n_cols;
                            const size_t   n_cols_p2 = hex_smin(n - nc2, (size_t) n_chunk_n_cols);
                            const uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                            dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[2], weight + nc2 * weight_stride),
                                           dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                        }
                    } else {
                        const int    prev     = (i - 1) & 1;
                        const size_t nc_prev  = (size_t) (i - 1) * n_chunk_n_cols;
                        const size_t n_cols_p = hex_smin(n - nc_prev, (size_t) n_chunk_n_cols);

                        hmx_queue_pop(ctx->hmx_queue);
                        if (is_int_hmx) {
                            hmx_matmul_task_init_int(&task_slots[i & 1], vtcm_output_bufs[i & 1],
                                                    vtcm_f16_act, vtcm_weight_bufs[i % 4], L.weight_bias_offset,
                                                    hmx_ceil_div(n_rows, m_tile_rows),
                                                    hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS),
                                                    (uint32_t) n_k_tiles, (uint32_t) n_groups);
                        } else {
                            hmx_matmul_task_init(&task_slots[i & 1], (__fp16 *) vtcm_output_bufs[i & 1],
                                                (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                                vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                                hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), (uint32_t) n_k_tiles);
                        }
                        hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task_slots[i & 1]));

                        const int chunk_dst_cols = dst_cols - (int) nc_prev;
                        if (chunk_dst_cols > 0) {
                            if (is_int_hmx) {
                                const float * wt_scales_prev = (const float *) ((const uint8_t *) vtcm_weight_bufs[(i - 1) % 4] + L.weight_scales_offset);
                                transfer_output_chunk_int_threaded(
                                    octx, vtcm_res_out[prev], NULL, vtcm_output_bufs[prev],
                                    vtcm_act_scales, wt_scales_prev, (int) n_rows, (int) n_cols_p,
                                    n_groups, res_out_stride_elems, 0, chunk_dst_cols,
                                    dst_elem_size, sizeof(float));
                            } else {
                                transfer_output_chunk_threaded(octx, vtcm_res_out[prev], NULL, vtcm_output_bufs[prev],
                                                               n_rows, n_cols_p, res_out_stride_elems, 0, chunk_dst_cols,
                                                               dst_elem_size, sizeof(float));
                            }
                        }

                        if (i >= 2) {
                            dma_queue_pop(weight_dma);
                        }

                        const int    vcols_o = hex_smin((int) n_cols_p, chunk_dst_cols);
                        const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                        dma_queue_push(weight_dma,
                                       dma_make_data(dst_dma_addr + (mr * dst_stride + nc_prev) * dst_elem_size, vtcm_res_out[prev]),
                                       dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);

                        if (i + 2 < n_chunk_cnt) {
                            const size_t   nc_p2     = (size_t) (i + 2) * n_chunk_n_cols;
                            const size_t   n_cols_p2 = hex_smin(n - nc_p2, (size_t) n_chunk_n_cols);
                            const uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                            dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                           dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                        }
                    }
                }

                {
                    const int    last        = (n_chunk_cnt - 1) & 1;
                    const size_t nc_last     = (size_t) (n_chunk_cnt - 1) * n_chunk_n_cols;
                    const size_t n_cols_last = hex_smin(n - nc_last, (size_t) n_chunk_n_cols);

                    hmx_queue_pop(ctx->hmx_queue);
                    if (n_chunk_cnt >= 2) {
                        dma_queue_pop(weight_dma);
                    }

                    const int chunk_dst_cols = dst_cols - (int) nc_last;
                    if (chunk_dst_cols > 0) {
                        if (is_int_hmx) {
                            const float * wt_scales_last = (const float *) ((const uint8_t *) vtcm_weight_bufs[(n_chunk_cnt - 1) % 4] + L.weight_scales_offset);
                            transfer_output_chunk_int_threaded(
                                octx, vtcm_res_out[last], NULL, vtcm_output_bufs[last],
                                vtcm_act_scales, wt_scales_last, (int) n_rows, (int) n_cols_last,
                                n_groups, res_out_stride_elems, 0, chunk_dst_cols,
                                dst_elem_size, sizeof(float));
                        } else {
                            transfer_output_chunk_threaded(octx, vtcm_res_out[last], NULL, vtcm_output_bufs[last],
                                                           n_rows, n_cols_last, res_out_stride_elems, 0, chunk_dst_cols,
                                                           dst_elem_size, sizeof(float));
                        }
                    }

                    const int    vcols_o = hex_smin((int) n_cols_last, chunk_dst_cols);
                    const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                    dma_queue_push(weight_dma,
                                   dma_make_data(dst_dma_addr + (mr * dst_stride + nc_last) * dst_elem_size, vtcm_res_out[last]),
                                   dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                    dma_queue_pop(weight_dma);
                }
            }
        }
    } else {
        hmx_matmul_task_t task;
        void *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        void *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        for (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            if (is_int_hmx) {
                transfer_activation_chunk_int_threaded(
                    ctx, (uint8_t *) vtcm_f16_act, vtcm_act_scales,
                    act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                    (int) n_rows, k, act_stride, k_valid, n_groups,
                    act_hvx, act_hvx_div, vtcm_f32_act, L.act_f32_bytes, act_elem_size);
            } else {
                struct activation_transfer_params act_params = {
                    .ctx = ctx,
                    .dst = vtcm_f16_act,
                    .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                    .n_rows = (int) n_rows,
                    .k_block = k,
                    .k_stride = act_stride,
                    .n_hvx = act_hvx,
                    .act_hvx_div = act_hvx_div,
                    .k_div = k_div,
                    .k_valid = k_valid,
                    .vtcm_f32_act = vtcm_f32_act,
                    .vtcm_f32_act_bytes = L.act_f32_bytes,
                    .act_elem_size = act_elem_size,
                };
                transfer_activation_chunk_threaded(&act_params);
            }

            for (uint32_t p = 0; p < n_weights; p++) {
                const struct htp_tensor * restrict src_w = octx->src[p];
                const struct htp_tensor * restrict dst   = octx->dsts[p];
                if (!src_w || !dst) continue;

                const dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma        = octx->ctx->dma[0];
                const dma_addr_t dst_dma_addr = dst->data;
                const size_t n                = src_w->ne[1];
                if (n == 0) continue;
                const size_t weight_stride    = src_w->nb[1];
                const size_t dst_elem_size    = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
                const size_t dst_stride       = dst->nb[1] / dst_elem_size;
                const size_t dst_row_bytes    = dst->nb[1];
                const size_t res_out_stride_bytes = hex_align_up((size_t) n_chunk_n_cols * dst_elem_size, 128);
                const int    res_out_stride_elems = (int) (res_out_stride_bytes / dst_elem_size);
                const int dst_cols            = (int) dst->ne[0];
                const int n_chunk_cnt         = hmx_ceil_div(n, n_chunk_n_cols);

                const uint32_t dma_src_stride = is_tiled ? tile_size : weight_stride;

                if (n > 0) {
                    const size_t n_cols = hex_smin(n, (size_t) n_chunk_n_cols);
                    const uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
                }

                for (int i = 0; i < n_chunk_cnt; ++i) {
                    const size_t nc = (size_t) i * n_chunk_n_cols;
                    const size_t n_cols = hex_smin(n - nc, (size_t) n_chunk_n_cols);
                    const size_t n_row_tiles = hmx_ceil_div(n_rows, m_tile_rows);
                    const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                    void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;
                    if (i > 0) {
                        dma_queue_pop(weight_dma);
                    }

                    if (is_int_hmx) {
                        rescale_int_weight_chunk_to_int8_tiles(
                            octx, vtcm_weight_bufs[i % 2], L.weight_bias_offset, L.weight_scales_offset,
                            curr_raw, (int) n_cols, k, n_groups, &n_groups_div, aligned_tile_size, weight_type);
                    } else {
                        dequantize_tiled_weight_chunk_to_fp16_tiles(
                            octx, vtcm_weight_bufs[i % 2], curr_raw,
                            n_cols, k, row_stride, weight_type,
                            n_k_tiles, n_k_tiles_div, dequant_worker_fn);
                    }

                    const size_t nc_next = nc + (size_t) n_chunk_n_cols;
                    if (nc_next < n) {
                        const size_t n_cols_next = hex_smin(n - nc_next, (size_t) n_chunk_n_cols);
                        const uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                        dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                    }

                    if (is_int_hmx) {
                        hmx_matmul_task_init_int(&task, vtcm_output, vtcm_f16_act, vtcm_weight_bufs[i % 2], L.weight_bias_offset,
                                                n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles, (uint32_t) n_groups);
                    } else {
                        hmx_matmul_task_init(&task, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, (uint32_t) n_k_tiles);
                    }
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_worker_fn, &task));
                    hmx_queue_pop(ctx->hmx_queue);

                    int chunk_dst_cols = dst_cols - (int) nc;
                    if (chunk_dst_cols > 0) {
                        if (is_int_hmx) {
                            const float * wt_scales = (const float *) ((const uint8_t *) vtcm_weight_bufs[i % 2] + L.weight_scales_offset);
                            transfer_output_chunk_int_threaded(
                                octx, vtcm_res_out[0], NULL, vtcm_output,
                                vtcm_act_scales, wt_scales, (int) n_rows, (int) n_cols,
                                n_groups, res_out_stride_elems, 0, chunk_dst_cols,
                                dst_elem_size, sizeof(float));
                        } else {
                            transfer_output_chunk_threaded(octx, vtcm_res_out[0], NULL, vtcm_output,
                                                           n_rows, n_cols, res_out_stride_elems, 0, chunk_dst_cols,
                                                           dst_elem_size, sizeof(float));
                        }
                    }
                    const int    vcols_o = hex_smin((int) n_cols, chunk_dst_cols);
                    const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * dst_elem_size : 0;
                    dma_queue_push(weight_dma,
                                   dma_make_data(dst_dma_addr + (mr * dst_stride + nc) * dst_elem_size, vtcm_res_out[0]),
                                   dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                }
                if (n_chunk_cnt > 0) {
                    dma_queue_pop(weight_dma);
                }
            }
        }
    }

    return HTP_STATUS_OK;
}

static inline dma_addr_t hmx_mm_weight_batch_data(const hmx_mm_batched_params_t *params,
                                                         int dst_b2, int dst_b3) {
    const size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    const size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    return params->weight + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

static inline dma_addr_t hmx_mm_act_batch_addr(const hmx_mm_batched_params_t *params,
                                                int dst_b2, int dst_b3) {
    return params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

static inline dma_addr_t hmx_mm_dst_batch_addr(const hmx_mm_batched_params_t *params,
                                               int dst_b2, int dst_b3) {
    return params->dst_dma_addr +
           (size_t) dst_b2 * params->dst_nb2 +
           (size_t) dst_b3 * params->dst_nb3;
}

static int hmx_mm_batched_simple(struct htp_ops_context *octx,
                                 const hmx_mm_batched_params_t *params,
                                 int m_chunk, int n_chunk, int pipeline, int act_hvx, int vtcm_size,
                                 const struct fastdiv_values * act_hvx_div, const struct fastdiv_values * k_div) {
    int ret = 0;
    for (int b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        for (int b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(octx, params->weight_dma, hmx_mm_dst_batch_addr(params, b2, b3),
                                cur_bias_addr, params->bias_bytes,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_weight_batch_data(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_size, params->weight_stride,
                                params->weight_type, params->k, params->dst_stride, params->bias_stride, params->dst_cols,
                                params->dst_elem_size, params->bias_elem_size,
                                m_chunk, n_chunk, pipeline, act_hvx,
                                act_hvx_div, k_div, params->tile_size, params->aligned_tile_size, vtcm_size,
                                params->kernel_type, params->hmx_n_groups,
                                &params->div_hmx_n_groups, &params->div_n_k_tiles);
        }
    }
    return ret;
}

static int hmx_mm_batched(struct htp_ops_context *octx, const hmx_mm_batched_params_t *params,
                          int m_chunk, int n_chunk, int pipeline, int act_hvx,
                          const struct fastdiv_values * act_hvx_div,
                          const struct fastdiv_values * k_div,
                          int vtcm_size) {
    struct htp_context * ctx = octx->ctx;
    const size_t row_stride = htp_tensor_get_row_size(params->weight_type, params->k);
    if (row_stride == 0) { return -1; }
    if (params->act_stride < params->k || (params->tile_size == 0 && (size_t) params->weight_stride < row_stride) || params->dst_stride < params->dst_cols) { return -1; }
    if (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { return -1; }
    if (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { return -1; }
    // N (the weight row count) does not have to be 32-aligned:
    if (params->k % 32 != 0) { return -1; }
    if ((params->dst_dma_addr & (params->dst_elem_size - 1)) != 0) { return -1; }

    const int group_size = params->r2;
    const size_t vtcm_budget  = ctx->vtcm_size;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool run_grouped = (params->weight_type == HTP_TYPE_F16 && params->tile_size == 0 && group_size > 1 &&
                        params->bias_nb2 == 0 && params->bias_nb3 == 0 && (size_t)vtcm_size <= vtcm_budget);
    if (!run_grouped) {
        return hmx_mm_batched_simple(octx, params, m_chunk, n_chunk, pipeline, act_hvx, vtcm_size, act_hvx_div, k_div);
    }

    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build_ex(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, false, act_hvx, 0, params->bias_bytes, 0, params->dst_elem_size);

    if (L.total_bytes > vtcm_budget) {
        FARF(HIGH, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        return hmx_mm_batched_simple(octx, params, m_chunk, n_chunk, pipeline, act_hvx, vtcm_size, act_hvx_div, k_div);
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    void    *vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    void    *vtcm_scratch1   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[1]);
    void    *vtcm_res_out    = VTCM_LAYOUT_PTR(void, base, L.off_res_out[0]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);

    const bool has_bias      = (params->bias_bytes > 0 && params->bias_addr != 0);
    void    *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_bias, has_bias);
    if (has_bias) {
        dma_queue_push(params->weight_dma, dma_make_data(vtcm_bias, params->bias_addr), hex_align_up(params->bias_bytes, 128), 0, params->bias_bytes, 1);
        dma_queue_pop(params->weight_dma);
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(HIGH, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, group_size, params->ne13,
            m_chunk_n_rows, n_chunk_n_cols,
            L.total_bytes, vtcm_budget);

    const size_t fp16_row_bytes   = (size_t) params->k * sizeof(__fp16);
    const size_t weight_row_bytes = (size_t) params->weight_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_task_t task;

    for (int b3 = 0; b3 < params->ne13; ++b3) {
        for (int b2_base = 0; b2_base < params->ne12; b2_base += group_size) {
            const dma_addr_t weight_group = hmx_mm_weight_batch_data(params, b2_base, b3);
            dma_queue * weight_dma = params->weight_dma;

            for (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_rows) {
                const size_t n_rows = hex_smin((size_t) params->m - mr, m_chunk_n_rows);
                const size_t n_row_tiles = hmx_ceil_div((int) n_rows, HTP_MM_HMX_TILE_N_ROWS);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                for (int g = 0; g < group_size; ++g) {
                    const dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_base + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_size;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                    struct activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_rows = (int) n_rows,
                        .k_block = params->k,
                        .k_stride = params->act_stride,
                        .n_hvx = act_hvx,
                        .act_hvx_div = act_hvx_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_bytes = L.act_f32_bytes,
                        .act_elem_size = params->act_elem_size,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                const size_t dst_row_bytes        = (size_t) params->dst_stride * params->dst_elem_size;
                const size_t res_out_stride_bytes = hex_align_up(n_chunk_n_cols * params->dst_elem_size, 128);
                const int    res_out_stride_elems = (int) (res_out_stride_bytes / params->dst_elem_size);

                // Prologue: push first weight chunk
                if (params->n > 0) {
                    const size_t n_cols_first = hex_smin((size_t) params->n, n_chunk_n_cols);
                    dma_queue_push(weight_dma, dma_make_data(vtcm_scratch0, weight_group),
                                   fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
                }

                for (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_cols) {
                    const size_t n_cols      = hex_smin((size_t) params->n - nc, n_chunk_n_cols);
                    const size_t n_col_tiles = hmx_ceil_div((int) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;
                    if (nc > 0) {
                        dma_queue_pop(weight_dma);
                    }

                    const size_t n_cols_tiled = hex_align_up(n_cols, HTP_MM_HMX_TILE_N_COLS);
                    hmx_interleave_rows_to_tiles(vtcm_weight, (const __fp16 *) curr_raw, (uint32_t) n_cols, params->k, params->k, 0, (uint32_t) n_cols_tiled);

                    // Reuse the interleaved weight for every q_head in this GQA group.
                    // Push next weight chunk right before the last head so earlier heads can
                    // sync-DMA their outputs on an empty queue, and the last head overlaps with next weight.
                    const size_t nc_next = nc + n_chunk_n_cols;
                    for (int g = 0; g < group_size; ++g) {
                        if (g == group_size - 1 && nc_next < (size_t) params->n) {
                            const size_t n_cols_next = hex_smin((size_t) params->n - nc_next, n_chunk_n_cols);
                            void * next_raw = (curr_raw == vtcm_scratch0) ? vtcm_scratch1 : vtcm_scratch0;
                            dma_queue_push(weight_dma, dma_make_data(next_raw, weight_group + nc_next * weight_row_bytes),
                                           fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
                        }

                        const __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                        hmx_matmul_task_init(&task, vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, params->k / 32);
                        hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_fp16_worker_fn, &task));
                        hmx_queue_pop(ctx->hmx_queue);

                        const void * bias_chunk = has_bias ? ((const uint8_t *) vtcm_bias + (mr * params->bias_stride + nc) * params->bias_elem_size) : NULL;
                        const int chunk_dst_cols = params->dst_cols - (int) nc;
                        if (chunk_dst_cols > 0) {
                            transfer_output_chunk_threaded(octx, vtcm_res_out, bias_chunk, vtcm_output, (int) n_rows, (int) n_cols,
                                                           res_out_stride_elems, params->bias_stride, chunk_dst_cols,
                                                           params->dst_elem_size, params->bias_elem_size);
                        }
                        const int    vcols_o = hex_smin((int) n_cols, chunk_dst_cols);
                        const size_t cbytes  = (vcols_o > 0) ? (size_t) vcols_o * params->dst_elem_size : 0;
                        const dma_addr_t out_addr = hmx_mm_dst_batch_addr(params, b2_base + g, b3) + (mr * params->dst_stride + nc) * params->dst_elem_size;
                        dma_queue_push(weight_dma, dma_make_data(out_addr, vtcm_res_out),
                                       dst_row_bytes, res_out_stride_bytes, cbytes, cbytes ? n_rows : 0);
                        if (g < group_size - 1) {
                            dma_queue_pop(weight_dma);
                        }
                    }
                }
                if (params->n > 0) {
                    dma_queue_pop(weight_dma);
                }
            }
        }
    }

    return 0;
}

static void transfer_activation_chunk_gathered_threaded(
            struct htp_ops_context *octx,
            __fp16 *dst,
            dma_addr_t act_dma_addr,
            size_t act_elem_size,
            int start_row,
            int n_rows,
            int k_block,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            int ne11,
            size_t nb11,
            size_t nb12,
            int k_valid,
            int n_hvx,
            const struct fastdiv_values * act_hvx_div,
            float * vtcm_f32_act,
            size_t vtcm_f32_act_bytes) {
    if (n_rows <= 0) {
        return;
    }
    const uint32_t mc_pad = hex_align_up((uint32_t) n_rows, HTP_MM_HMX_TILE_N_ROWS);
    uint32_t chunks_per_hvx = (n_hvx <= 1) ? mc_pad
                            : fastdiv(mc_pad + (uint32_t) n_hvx - 1u, act_hvx_div);
    chunks_per_hvx = hex_align_up(chunks_per_hvx, 2u);

    const int actual_hvx = (int) hmx_ceil_div(mc_pad, chunks_per_hvx);

    activation_transfer_gathered_task_state_t state = {
        .dst                        = dst,
        .act_dma_addr               = act_dma_addr,
        .vtcm_f32_act               = vtcm_f32_act,
        .vtcm_f32_act_bytes_per_hvx = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_hvx_div), 128),
        .act_elem_size              = act_elem_size,
        .n_rows                     = (uint32_t) n_rows,
        .mc_pad                     = mc_pad,
        .n_chunks_per_task          = chunks_per_hvx,
        .k_block                    = (uint32_t) k_block,
        .matrix_rows                = matrix_rows,
        .cur_a                      = (uint32_t) cur_a,
        .mapping_stride             = (uint32_t) mapping_stride,
        .ne11                       = (uint32_t) ne11,
        .ne11_div                   = ne11 > 1 ? init_fastdiv_values(ne11) : (struct fastdiv_values){0, 0},
        .nb11                       = nb11,
        .nb12                       = nb12,
        .start_row                  = (uint32_t) start_row,
        .k_valid                    = (uint32_t) k_valid,
    };

    work_queue_run(octx->ctx->hvx_queue, transfer_activation_chunk_gathered_worker_fn, &state, actual_hvx);
}

static __attribute__((noinline)) int hmx_mm_id_2d_int(
        struct htp_ops_context * octx,
        dma_queue * weight_dma,
        const dma_addr_t * dsts,
        dma_addr_t activation,
        size_t act_elem_size,
        size_t dst_elem_size,
        const dma_addr_t * weights,
        uint32_t n_weights,
        int m, int k, int n,
        int k_valid,
        int ne11,
        size_t act_nb1, size_t act_nb2,
        size_t dst_nb1, size_t dst_nb2,
        int weight_stride,
        int weight_type,
        int tile_size,
        int aligned_tile_size,
        const struct mmid_row_mapping * matrix_rows,
        int cur_a,
        int mapping_stride,
        int m_start,
        int m_end,
        int32_t kernel_type) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const int mr_limit = hex_smin(m, m_end);
    const int m_valid_core = mr_limit - m_start;
    if (m_valid_core <= 0 || n_weights == 0) {
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        return 0;
    }

    const int m_core_padded = hex_align_up(m_valid_core, 64);

    if (k % 32 != 0 || n % 32 != 0 || tile_size <= 0) { return -1; }
    if ((activation & (act_elem_size - 1)) != 0) { return -1; }
    for (uint32_t p = 0; p < n_weights; ++p) {
        if ((dsts[p] & (dst_elem_size - 1)) != 0) { return -1; }
    }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    const uint32_t n_groups       = ((uint32_t) k + HTP_MM_HMX_W8A8_KGROUP - 1u) / HTP_MM_HMX_W8A8_KGROUP;
    const uint32_t n_k_tiles      = (uint32_t) k / HTP_MM_HMX_TILE_N_COLS;
    const uint32_t n_weight_tiles = (uint32_t) k / htp_tensor_get_tile_k(weight_type);
    const struct fastdiv_values n_groups_div = init_fastdiv_values(n_groups);

    const size_t vtcm_budget = ctx->vtcm_size;
    size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0, vtcm_used = 0;
    int act_hvx = 0;
    if (!htp_mm_int_hmx_solve_2d_params(
            kernel_type, weight_type, (uint32_t) k, (uint32_t) m_core_padded,
            (uint32_t) n, (uint32_t) m_core_padded, (uint32_t) m_core_padded,
            (int) octx->n_hvx, /*pipeline=*/false, /*is_matmul_id=*/true,
            (uint32_t) aligned_tile_size, /*bias_size=*/0, dst_elem_size, vtcm_budget,
            &m_chunk_n_rows, &n_chunk_n_cols, &act_hvx, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d-int: VTCM too small: m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        return -1;
    }

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build_ex(
        &L, kernel_type, weight_type, (uint32_t) k,
        m_chunk_n_rows, n_chunk_n_cols, 1, /*pipeline=*/false,
        (uint32_t) act_hvx, (uint32_t) aligned_tile_size, 0, 0, dst_elem_size);
    if (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d-int: VTCM overflow: used %zu budget %zu", L.total_bytes, vtcm_budget);
        return -1;
    }

    uint8_t * const base      = (uint8_t *) ctx->vtcm_base;
    void    * vtcm_weight     = VTCM_LAYOUT_PTR(void, base, L.off_weight[0]);
    uint8_t * vtcm_act        = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    float   * vtcm_act_scales = VTCM_LAYOUT_PTR(float, base, L.off_act_scales);
    float   * vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    void    * vtcm_output     = VTCM_LAYOUT_PTR(void, base, L.off_dst[0]);
    void    * vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    uint8_t * vtcm_res_out    = VTCM_LAYOUT_PTR(uint8_t, base, L.off_res_out[0]);
    if (L.lut_bytes > 0) {
        iq2xxs_lut_copy(base + L.off_lut);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    const struct fastdiv_values act_hvx_div = init_fastdiv_values((uint32_t) act_hvx);
    const int n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);
    const size_t res_out_stride_bytes = hex_align_up(n_chunk_n_cols * dst_elem_size, 128);
    const int    res_out_stride_elems = (int) (res_out_stride_bytes / dst_elem_size);
    hmx_matmul_task_t task;

    const bool preissue_p0 = L.can_preissue_weight && n > 0 && n_weights > 0;

    for (size_t mr = (size_t) m_start; mr < (size_t) mr_limit; mr += m_chunk_n_rows) {
        const size_t n_rows         = hex_smin((size_t) mr_limit - mr, m_chunk_n_rows);
        const size_t n_row_tiles    = hmx_ceil_div(n_rows, 64u);
        const bool   overlap_w_next = (n_rows <= 200u);

        if (preissue_p0) {
            const size_t   n_cols0 = hex_smin((size_t) n, n_chunk_n_cols);
            const uint32_t height0 = hmx_ceil_div(n_cols0, 32) * n_weight_tiles;
            dma_proxy_push(ctx->dma_proxy, dma_make_data(vtcm_weight, weights[0]),
                           aligned_tile_size, tile_size, tile_size, height0);
        }

        transfer_activation_chunk_int_gathered_threaded(
            ctx, vtcm_act, vtcm_act_scales, activation, (uint32_t) act_elem_size,
            (int) mr, (int) n_rows, k, k_valid, (int) n_groups,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2,
            act_hvx, &act_hvx_div, vtcm_f32_act, L.act_f32_bytes);

        for (uint32_t p = 0; p < n_weights; ++p) {
            const dma_addr_t weight_p    = weights[p];
            const dma_addr_t dst_p       = dsts[p];
            const bool       p_preissued = (p == 0 && preissue_p0);

            if (n > 0 && !p_preissued) {
                const size_t n_cols = hex_smin((size_t) n, n_chunk_n_cols);
                const uint32_t height = hmx_ceil_div(n_cols, 32) * n_weight_tiles;
                dma_queue_push(weight_dma, dma_make_data(vtcm_weight, weight_p),
                               aligned_tile_size, tile_size, tile_size, height);
            }

            for (int i = 0; i < n_chunk_cnt; ++i) {
                const size_t nc          = (size_t) i * n_chunk_n_cols;
                const size_t n_cols      = hex_smin((size_t) n - nc, n_chunk_n_cols);
                const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                void * curr_raw;
                if (i == 0 && p_preissued) {
                    curr_raw = (void *) dma_proxy_pop(ctx->dma_proxy).dst;
                } else {
                    curr_raw = (void *) dma_queue_pop(weight_dma).dst;
                }
                if (overlap_w_next && i > 0) {
                    for (size_t s = 0; s < n_rows; ++s) {
                        dma_queue_pop(weight_dma);
                    }
                }

                rescale_int_weight_chunk_to_int8_tiles(
                    octx, vtcm_scratch0, L.weight_bias_offset, L.weight_scales_offset,
                    curr_raw, (int) n_cols, k, (int) n_groups, &n_groups_div,
                    aligned_tile_size, weight_type);

                const size_t nc_next = nc + n_chunk_n_cols;
                if (overlap_w_next && nc_next < (size_t) n) {
                    const size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
                    const uint32_t height_next = hmx_ceil_div(n_cols_next, 32) * n_weight_tiles;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_weight, weight_p + nc_next * weight_stride),
                                   aligned_tile_size, tile_size, tile_size, height_next);
                }

                hmx_matmul_task_init_int(&task, vtcm_output, vtcm_act, vtcm_scratch0, L.weight_bias_offset,
                                        n_row_tiles, n_col_tiles, n_k_tiles, n_groups);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_w8a8_worker_fn, &task));
                hmx_queue_pop(ctx->hmx_queue);

                const float * wt_scales = (const float *) ((const uint8_t *) vtcm_scratch0 + L.weight_scales_offset);
                transfer_output_chunk_int_threaded(
                    octx, vtcm_res_out, NULL, vtcm_output,
                    vtcm_act_scales, wt_scales, (int) n_rows, (int) n_cols,
                    (int) n_groups, res_out_stride_elems, 0, (int) n_cols,
                    dst_elem_size, sizeof(float));

                if (overlap_w_next) {
                    push_scattered_dst_rows_dma(
                        weight_dma, dst_p + nc * dst_elem_size, vtcm_res_out,
                        (uint32_t) mr, (uint32_t) n_rows, res_out_stride_bytes, n_cols * dst_elem_size,
                        matrix_rows, (uint32_t) cur_a, (uint32_t) mapping_stride, dst_nb1, dst_nb2);
                } else {
                    sync_scattered_dst_rows_dma(
                        weight_dma, dst_p + nc * dst_elem_size, vtcm_res_out,
                        (uint32_t) mr, (uint32_t) n_rows, res_out_stride_bytes, n_cols * dst_elem_size,
                        matrix_rows, (uint32_t) cur_a, (uint32_t) mapping_stride, dst_nb1, dst_nb2);
                    if (nc_next < (size_t) n) {
                        const size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
                        const uint32_t height_next = hmx_ceil_div(n_cols_next, 32) * n_weight_tiles;
                        dma_queue_push(weight_dma, dma_make_data(vtcm_weight, weight_p + nc_next * weight_stride),
                                       aligned_tile_size, tile_size, tile_size, height_next);
                    }
                }
            }
            if (overlap_w_next && n_chunk_cnt > 0) {
                for (size_t s = 0; s < n_rows; ++s) {
                    dma_queue_pop(weight_dma);
                }
            }
        }
    }

    return 0;
}

static int hmx_mm_id_2d_f32(struct htp_ops_context *octx,
                                         dma_queue *weight_dma,
                                         dma_addr_t dst,
                                         dma_addr_t activation,
                                         size_t act_elem_size,
                                         size_t dst_elem_size,
                                         dma_addr_t weight,
                                         int m, int k, int n,
                                         int k_valid,
                                         int ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         int weight_stride,
                                         int weight_type,
                                         int tile_size,
                                         int aligned_tile_size,
                                         const struct mmid_row_mapping *matrix_rows,
                                         int cur_a,
                                         int mapping_stride,
                                         int m_start,
                                         int m_end,
                                         int32_t kernel_type) {
    if (htp_mm_kernel_is_hmx_int(kernel_type)) {
        const dma_addr_t dsts[1]    = { dst };
        const dma_addr_t weights[1] = { weight };
        return hmx_mm_id_2d_int(
            octx, weight_dma, dsts, activation, act_elem_size, dst_elem_size,
            weights, 1, m, k, n, k_valid, ne11, act_nb1, act_nb2, dst_nb1, dst_nb2,
            weight_stride, weight_type, tile_size, aligned_tile_size,
            matrix_rows, cur_a, mapping_stride, m_start, m_end,
            kernel_type);
    }
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const int mr_limit = hex_smin(m, m_end);
    const int m_valid_core = mr_limit - m_start;
    if (m_valid_core <= 0) {
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        return 0;
    }

    const int m_core_padded = hex_align_up(m_valid_core, HTP_MM_HMX_TILE_N_ROWS);

    if (k % 32 != 0 || n % 32 != 0) { return -1; }
    if ((dst & (dst_elem_size - 1)) != 0 || (activation & (act_elem_size - 1)) != 0) { return -1; }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const size_t vtcm_budget  = ctx->vtcm_size;
    size_t vtcm_used = 0;

    const bool is_tiled = (tile_size > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    const uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    const uint32_t dma_src_stride  = is_tiled ? tile_size : weight_stride;
    const uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    int act_hvx = 0;
    if (!htp_mm_hmx_solve_2d_params(
            weight_type, (uint32_t) k, (uint32_t) m_core_padded,
            (uint32_t) n, (uint32_t) m_core_padded, (uint32_t) m_core_padded,
            (int) octx->n_hvx, /*pipeline=*/false, /*is_matmul_id=*/true,
            (uint32_t) aligned_tile_size, /*bias_size=*/0, dst_elem_size, vtcm_budget,
            &m_chunk_n_rows, &n_chunk_n_cols, &act_hvx, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        return -1;
    }

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build_ex(
        &L, HTP_MM_KERNEL_HMX_2D, weight_type, (uint32_t) k,
        m_chunk_n_rows, n_chunk_n_cols, 1, /*pipeline=*/false,
        (uint32_t) act_hvx, (uint32_t) aligned_tile_size, 0, 0, dst_elem_size);
    if (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", L.total_bytes, vtcm_budget);
        return -1;
    }

    uint8_t * const base   = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight   = VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]);
    __fp16  *vtcm_f16_act  = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    float   *vtcm_f32_act  = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    __fp16  *vtcm_output   = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    void    *vtcm_scratch0 = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    uint8_t *vtcm_res_out  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_res_out[0]);
    __fp16  *vtcm_scales   = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    uint8_t *vtcm_lut      = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_lut, L.lut_bytes > 0);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));
    if (L.lut_bytes > 0) {
        iq2xxs_lut_copy(vtcm_lut);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    const struct fastdiv_values act_hvx_div = init_fastdiv_values((uint32_t) act_hvx);
    hmx_matmul_task_t task;
    void *vtcm_raw_bufs[2] = { vtcm_weight, dequant_worker_fn ? vtcm_weight : vtcm_scratch0 };
    void *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight, vtcm_scratch0 };
    const int n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);
    const size_t res_out_stride_bytes = hex_align_up(n_chunk_n_cols * dst_elem_size, 128);
    const int    res_out_stride_elems = (int) (res_out_stride_bytes / dst_elem_size);

    for (size_t mr = (size_t) m_start; mr < (size_t) mr_limit; mr += m_chunk_n_rows) {
        const size_t n_rows         = hex_smin((size_t) mr_limit - mr, m_chunk_n_rows);
        const size_t n_row_tiles    = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
        const bool   overlap_w_next = (n_rows <= 200u);

        transfer_activation_chunk_gathered_threaded(
            octx, vtcm_f16_act, activation, act_elem_size, (int) mr, (int) n_rows, k,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2, k_valid,
            act_hvx, &act_hvx_div, vtcm_f32_act, L.act_f32_bytes);

        if (n > 0) {
            const size_t n_cols = hex_smin((size_t) n, n_chunk_n_cols);
            const uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
            dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height);
        }

        for (int i = 0; i < n_chunk_cnt; ++i) {
            const size_t nc = (size_t) i * n_chunk_n_cols;
            const size_t n_cols = hex_smin((size_t) n - nc, n_chunk_n_cols);
            const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

            void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;
            if (overlap_w_next && i > 0) {
                for (size_t s = 0; s < n_rows; ++s) {
                    dma_queue_pop(weight_dma);
                }
            }

            dequantize_tiled_weight_chunk_to_fp16_tiles(
                octx, vtcm_weight_bufs[i % 2], curr_raw,
                n_cols, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn
            );

            const size_t nc_next = nc + n_chunk_n_cols;
            if (overlap_w_next && nc_next < (size_t) n) {
                const size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
                const uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
            }

            hmx_matmul_task_init(&task, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_fp16_worker_fn, &task));
            hmx_queue_pop(ctx->hmx_queue);

            transfer_output_chunk_threaded(
                octx, vtcm_res_out, NULL, vtcm_output,
                n_rows, n_cols, res_out_stride_elems, 0, (int) n_cols,
                dst_elem_size, sizeof(float));

            if (overlap_w_next) {
                push_scattered_dst_rows_dma(
                    weight_dma, dst + nc * dst_elem_size, vtcm_res_out,
                    (uint32_t) mr, (uint32_t) n_rows, res_out_stride_bytes, n_cols * dst_elem_size,
                    matrix_rows, (uint32_t) cur_a, (uint32_t) mapping_stride, dst_nb1, dst_nb2);
            } else {
                sync_scattered_dst_rows_dma(
                    weight_dma, dst + nc * dst_elem_size, vtcm_res_out,
                    (uint32_t) mr, (uint32_t) n_rows, res_out_stride_bytes, n_cols * dst_elem_size,
                    matrix_rows, (uint32_t) cur_a, (uint32_t) mapping_stride, dst_nb1, dst_nb2);
                if (nc_next < (size_t) n) {
                    const size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
                    const uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                }
            }
        }
        if (overlap_w_next && n_chunk_cnt > 0) {
            for (size_t s = 0; s < n_rows; ++s) {
                dma_queue_pop(weight_dma);
            }
        }
    }

    return 0;
}

// --- Dispatchers and Public Entry Points ---

static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    int k = (int) src0->ne[0];
    int n = (int) src0->ne[1];
    const int m_total    = (int) act->ne[1];
    const uint32_t act_elem_size  = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const uint32_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const uint32_t bias_elem_size = (bias && bias->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const int act_stride = (int) (act->nb[1] / act_elem_size);

    int m_start = 0;
    int m_rows  = m_total;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_total, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        m_start = (int) range.start;
        m_rows  = (int) range.count;
    }

    if (m_rows == 0) {
        return HTP_STATUS_OK;
    }

    dma_addr_t bias_addr = 0;
    size_t bias_bytes = 0;
    uint32_t bias_stride = 0;
    size_t bias_nb2 = 0;
    size_t bias_nb3 = 0;
    if (bias) {
        bias_stride = (bias->ne[1] == 1) ? 0 : (uint32_t) (bias->nb[1] / bias_elem_size);
        bias_addr = bias->data + (size_t) m_start * bias_stride * bias_elem_size;
        const size_t slice_bytes = (bias->ne[1] == 1) ? bias->nb[1] : (size_t) m_rows * bias->nb[1];
        bias_bytes = MIN((size_t) kparams->vtcm_bias_size, slice_bytes);
        bias_nb2 = (bias->ne[2] == 1) ? 0 : bias->nb[2];
        bias_nb3 = (bias->ne[3] == 1) ? 0 : bias->nb[3];
    }

    const int dst_stride = (int) (dst->nb[1] / dst_elem_size);
    const dma_addr_t dst_addr = dst->data + (size_t) m_start * dst_stride * dst_elem_size;
    // byte offset, act_stride is in activation elements (F16 or F32)
    const dma_addr_t act_addr = act->data + (size_t) m_start * act_stride * act_elem_size;

    int ret = -1;
    if (htp_mm_kernel_is_hmx_batched(kparams->kernel_type)) {
        hmx_mm_batched_params_t batch_params = {
            .dst_dma_addr      = dst_addr,
            .bias_addr         = bias_addr,
            .bias_bytes        = bias_bytes,
            .act_dma_addr      = act_addr,
            .weight            = src0->data,
            .weight_dma        = octx->ctx->dma[0],
            .m                 = m_rows,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_size     = act_elem_size,
            .dst_elem_size     = dst_elem_size,
            .bias_elem_size    = bias_elem_size,
            .weight_stride     = (int) src0->nb[1],
            .weight_type       = (int) src0->type,
            .dst_stride        = dst_stride,
            .dst_cols          = (int) dst->ne[0],
            .bias_stride       = bias_stride,
            .tile_size         = kparams->tile_size,
            .aligned_tile_size = kparams->aligned_tile_size,
            .ne02              = ne02,
            .ne03              = ne03,
            .ne12              = ne12,
            .ne13              = ne13,
            .src0_nb2          = src0->nb[2],
            .src0_nb3          = src0->nb[3],
            .act_nb2           = act->nb[2],
            .act_nb3           = act->nb[3],
            .dst_nb2           = dst->nb[2],
            .dst_nb3           = dst->nb[3],
            .bias_nb2          = bias_nb2,
            .bias_nb3          = bias_nb3,
            .r2                = (ne02 > 0) ? (ne12 / ne02) : 1,
            .r3                = (ne03 > 0) ? (ne13 / ne03) : 1,
            .div_r2            = kparams->div_r2,
            .div_r3            = kparams->div_r3,
            .kernel_type       = kparams->kernel_type,
            .hmx_n_groups      = kparams->hmx_n_groups,
            .div_hmx_n_groups  = kparams->div_hmx_n_groups,
            .div_n_k_tiles     = kparams->div_n_k_tiles,
        };
        ret = hmx_mm_batched(octx, &batch_params,
                             kparams->m_chunk, kparams->n_chunk,
                             kparams->pipeline,
                             kparams->n_act_hvx,
                             &kparams->div_n_act_hvx,
                             &kparams->div_ne00_padded,
                             kparams->vtcm_size);
    } else {
        ret = hmx_mm_2d_f32(
            octx, octx->ctx->dma[0], dst_addr, bias_addr, bias_bytes,
            act_addr, src0->data,
            m_rows, k, n, act_stride, act_elem_size, (int) src0->nb[1], (int) src0->type, (int) act->ne[0],
            dst_stride, bias_stride, (int)dst->ne[0], dst_elem_size, bias_elem_size,
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline,
            kparams->n_act_hvx,
            &kparams->div_n_act_hvx,
            &kparams->div_ne00_padded,
            kparams->tile_size, kparams->aligned_tile_size, kparams->vtcm_size,
            kparams->kernel_type, kparams->hmx_n_groups,
            &kparams->div_hmx_n_groups, &kparams->div_n_k_tiles
        );
    }

    if (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        return HTP_STATUS_INTERNAL_ERR;
    }
    return HTP_STATUS_OK;
}

static int op_matmul_impl(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const int status = htp_mm_init_context(octx, kparams);
    if (status != HTP_STATUS_OK) {
        return status;
    }

    if (kparams->n_hmx) {
        return hmx_mm_op_matmul(octx, kparams);
    }

    return hvx_mm_matmul(octx);
}

int op_matmul(struct htp_ops_context * octx) {
    return op_matmul_impl(octx);
}

static int hmx_mm_op_matmul_id(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx
) {
    const uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    htp_matmul_tensors_preamble;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const int n_as  = ne02;

    const size_t act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const size_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    const bool mdev_split = (octx->ctx->mdev.count > 1);
    uint32_t n_active = 0;
    if (mdev_split) {
        for (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            if (matrix_row_counts[a] > 0) n_active++;
        }
    }
    const bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.count);

    uint32_t target_dev = 0;
    for (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        const int m_padded = hex_align_up(cne1, 32);
        int m_start = 0, m_end = m_padded;
        if (expert_split) {
            const bool my_expert = (target_dev == octx->ctx->mdev.idx);
            if (++target_dev == octx->ctx->mdev.count) {
                target_dev = 0;
            }
            if (!my_expert) continue;
        } else if (mdev_split) {
            const bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.count;
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            m_start = (int) range.start;
            m_end   = (int) (range.start + range.count);
        }
        if (m_start >= m_end || m_start >= cne1) continue;

        int ret = hmx_mm_id_2d_f32(octx, octx->ctx->dma[0], dst->data, act->data,
                                   act_elem_size, dst_elem_size,
                                   src0->data + (dma_addr_t) cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (int) src0->nb[1], (int) src0->type,
                                   kparams->tile_size, kparams->aligned_tile_size,
                                   matrix_rows, cur_a, mmctx->mapping_stride,
                                   m_start, m_end,
                                   kparams->kernel_type);
        if (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    return HTP_STATUS_OK;
}

static int hvx_mm_matmul_id(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    const uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    const uint32_t act_nrows            = mmctx->act_nrows;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const size_t src0_row_size = nb01;

    const uint32_t qk = HTP_MM_ACT_BLOCK_K;
    const uint32_t nb = (ne10 + qk - 1) / qk;
    const uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (act_nrows < octx->n_hvx) {
        n_quant_tasks = MIN(total_nb, octx->n_hvx);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(act_nrows, octx->n_hvx);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    const size_t act_row_size  = kparams->act_row_size;
    size_t act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    const size_t dst_row_size = (size_t) ne0 * sizeof(float);
    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_hvx,
                                 dst_row_size, src0_row_size, act_elem_size, 0, kparams->n_prefetch, true, false);

    const size_t vtcm_size = L.total_bytes;

    FARF(HIGH, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         act->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = NULL;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
    hvx_mm_iq2xxs_lut_init(base, &L);

    mmctx->vtcm_src0_stride    = src0_row_size_padded;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = htp_mm_act_raw_row_size(ne10, act_elem_size);

    mmctx->vtcm_src0_size_per_hvx = fastdiv(L.src0_bytes, &octx->n_hvx_div);
    mmctx->vtcm_act_size_per_hvx  = L.act_bytes;
    mmctx->vtcm_dst_size_per_hvx  = fastdiv(L.dst_bytes, &octx->n_hvx_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_hvx = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->hvx_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->hvx_queue, hvx_mmid_task_func, mmctx, octx->n_hvx);

    return HTP_STATUS_OK;
}

static int hmx_mm_op_matmul_id_nx(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx
) {
    const uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const int n_as = src0->ne[2];

    const size_t act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    const bool mdev_split = (octx->ctx->mdev.count > 1);
    uint32_t n_active = 0;
    if (mdev_split) {
        for (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            if (matrix_row_counts[a] > 0) n_active++;
        }
    }
    const bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.count);

    uint32_t target_dev = 0;
    for (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        const int m_padded = hex_align_up(cne1, 32);
        int m_start = 0, m_end = m_padded;
        if (expert_split) {
            const bool my_expert = (target_dev == octx->ctx->mdev.idx);
            if (++target_dev == octx->ctx->mdev.count) {
                target_dev = 0;
            }
            if (!my_expert) continue;
        } else if (mdev_split) {
            const bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.count;
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            m_start = (int) range.start;
            m_end   = (int) (range.start + range.count);
        }
        if (m_start >= m_end || m_start >= cne1) continue;

        if (htp_mm_kernel_is_hmx_int(kparams->kernel_type)) {
            const struct htp_tensor * restrict dst0 = octx->dsts[0];
            bool same_shape = true;
            dma_addr_t dst_addrs[HTP_OP_MAX_OUTPUTS];
            dma_addr_t weight_addrs[HTP_OP_MAX_OUTPUTS];
            for (uint32_t p = 0; p < n_weights && same_shape; ++p) {
                const struct htp_tensor * restrict sw = octx->src[p];
                const struct htp_tensor * restrict d  = octx->dsts[p];
                if (sw->ne[0] != src0->ne[0] || sw->ne[1] != src0->ne[1] ||
                    sw->nb[1] != src0->nb[1] || sw->type != src0->type ||
                    d->type != dst0->type || d->nb[1] != dst0->nb[1] || d->nb[2] != dst0->nb[2]) {
                    same_shape = false;
                    break;
                }
                dst_addrs[p]    = d->data;
                weight_addrs[p] = sw->data + (dma_addr_t) cur_a * sw->nb[2];
            }
            if (same_shape) {
                const size_t dst0_elem_size = (dst0->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
                int ret = hmx_mm_id_2d_int(
                    octx, octx->ctx->dma[0], dst_addrs, act->data,
                    act_elem_size, dst0_elem_size, weight_addrs, n_weights,
                    cne1, src0->ne[0], src0->ne[1], act->ne[0], act->ne[1],
                    act->nb[1], act->nb[2], dst0->nb[1], dst0->nb[2],
                    (int) src0->nb[1], (int) src0->type,
                    kparams->tile_size, kparams->aligned_tile_size,
                    matrix_rows, cur_a, mmctx->mapping_stride,
                    m_start, m_end, kparams->kernel_type);
                if (ret != 0) {
                    FARF(ERROR, "HMX matmul ID NX failed for expert %u, error %d\n", cur_a, ret);
                    return HTP_STATUS_NO_SUPPORT;
                }
                continue;
            }
        }

        for (uint32_t p = 0; p < n_weights; ++p) {
            const struct htp_tensor * restrict src_w = octx->src[p];
            const struct htp_tensor * restrict dst   = octx->dsts[p];
            if (!src_w || !dst) continue;
            const size_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

            int ret = hmx_mm_id_2d_f32(octx, octx->ctx->dma[0], dst->data, act->data,
                                       act_elem_size, dst_elem_size,
                                       src_w->data + (dma_addr_t) cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (int) src_w->nb[1], (int) src_w->type,
                                       kparams->tile_size, kparams->aligned_tile_size,
                                       matrix_rows, cur_a, mmctx->mapping_stride,
                                       m_start, m_end,
                                       kparams->kernel_type);
            if (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                return HTP_STATUS_NO_SUPPORT;
            }
        }
    }

    return HTP_STATUS_OK;
}

static int hvx_mm_matmul_id_nx(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    const uint32_t act_nrows            = mmctx->act_nrows;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const size_t src0_row_size = src0->nb[1];

    const uint32_t qk = HTP_MM_ACT_BLOCK_K;
    const uint32_t nb = (act->ne[0] + qk - 1) / qk;
    const uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (act_nrows < octx->n_hvx) {
        n_quant_tasks = MIN(total_nb, octx->n_hvx);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(act_nrows, octx->n_hvx);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    const size_t act_row_size  = kparams->act_row_size;
    size_t act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    size_t max_dst_row_size = 0;
    for (uint32_t p = 0; p < n_weights; ++p) {
        if (octx->dsts[p]) {
            const size_t sz = (size_t) octx->dsts[p]->ne[0] * sizeof(float);
            if (sz > max_dst_row_size) max_dst_row_size = sz;
        }
    }
    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_hvx,
                                 max_dst_row_size, src0_row_size, act_elem_size, 0, kparams->n_prefetch, true, false);

    const size_t vtcm_size = L.total_bytes;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
    hvx_mm_iq2xxs_lut_init(base, &L);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = htp_mm_act_raw_row_size(act->ne[0], act_elem_size);

    mmctx->vtcm_src0_size_per_hvx = fastdiv(L.src0_bytes, &octx->n_hvx_div);
    mmctx->vtcm_act_size_per_hvx  = L.act_bytes;
    mmctx->vtcm_dst_size_per_hvx  = fastdiv(L.dst_bytes, &octx->n_hvx_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    FARF(HIGH, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, nhvx %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.total_bytes, octx->ctx->vtcm_size, octx->n_hvx);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_hvx    = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_queue_run(octx->ctx->hvx_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->hvx_queue, hvx_mmid_task_func, mmctx, octx->n_hvx);

    return HTP_STATUS_OK;
}

static __attribute__((noinline)) uint32_t scan_expert_ids_n(
    const int32_t * ids_data,
    uint32_t ids_ne1,
    const uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    struct mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    uint32_t n_skipped = 0;

    for (uint32_t iid1 = 0; iid1 < ids_ne1; ++iid1) {
        const int32_t * row_ptr = ids_data + (size_t) iid1 * n_ids;
        for (uint32_t id = 0; id < n_ids; ++id) {
            const int32_t i02 = row_ptr[id];
            if (i02 < 0) {
                n_skipped++;
                continue;
            }
            assert(i02 >= 0 && i02 < (int32_t) n_as);

            if (matrix_rows) {
                matrix_rows[i02 * mapping_stride + counts[i02]] = (struct mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
    return n_skipped;
}

static __attribute__((noinline)) uint32_t scan_expert_ids(
    const int32_t * ids_data,
    uint32_t ids_ne1,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    struct mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    switch (n_ids) {
        case 8:  return scan_expert_ids_n(ids_data, ids_ne1, 8,     n_as, counts, matrix_rows, mapping_stride);
        case 4:  return scan_expert_ids_n(ids_data, ids_ne1, 4,     n_as, counts, matrix_rows, mapping_stride);
        case 2:  return scan_expert_ids_n(ids_data, ids_ne1, 2,     n_as, counts, matrix_rows, mapping_stride);
        default: return scan_expert_ids_n(ids_data, ids_ne1, n_ids, n_as, counts, matrix_rows, mapping_stride);
    }
}

static __attribute__((noinline)) uint32_t count_skipped_ids_single(const int32_t * ids_data, uint32_t n_ids) {
    uint32_t n_skipped = 0;
    for (uint32_t id = 0; id < n_ids; ++id) {
        if (ids_data[id] < 0) {
            n_skipped++;
        }
    }
    return n_skipped;
}

// Zero rows for skipped slots (ids < 0). Matmuls only write rows that an expert owns.
static void zero_skipped_dst_rows(
    struct htp_ops_context * octx,
    const int32_t * ids_data,
    uint32_t ids_ne1,
    uint32_t n_ids,
    const struct htp_tensor * dst
) {
    const uint32_t elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    uint32_t col_start = 0;
    uint32_t col_count = dst->ne[0];

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            dst->ne[0], 32, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        col_start = range.start;
        col_count = range.count;
    }

    if (col_count == 0) {
        return;
    }

    const size_t dst_nb1 = dst->nb[1];
    const size_t dst_nb2 = dst->nb[2];
    const size_t col_offset_bytes = (size_t) col_start * elem_size;
    const size_t row_bytes        = (size_t) col_count * elem_size;

    // Zero one row in VTCM and copy to skipped dst rows via bypass DMA.
    dma_queue * const dma_q       = octx->ctx->dma[0];
    uint8_t * const   vtcm_zero   = octx->ctx->vtcm_base;
    const size_t      chunk_bytes = MIN(row_bytes, octx->ctx->vtcm_size);
    hvx_splat_u8_a(vtcm_zero, 0, (uint32_t) hex_round_up(chunk_bytes, 128));

    uint32_t pending = 0;
    for (uint32_t iid1 = 0; iid1 < ids_ne1; ++iid1) {
        const int32_t * ids_row = ids_data + (size_t) iid1 * n_ids;
        const dma_addr_t dst_row_group = dst->data + (dma_addr_t) iid1 * dst_nb2 + col_offset_bytes;
        for (uint32_t id = 0; id < n_ids; ++id) {
            if (ids_row[id] < 0) {
                const dma_addr_t row_base = dst_row_group + (dma_addr_t) id * dst_nb1;
                for (size_t off = 0; off < row_bytes; off += chunk_bytes) {
                    const size_t n = MIN(chunk_bytes, row_bytes - off);
                    dma_queue_push(dma_q, dma_make_data(row_base + off, vtcm_zero), n, n, n, 1);
                    if (++pending >= 128) {
                        dma_queue_flush(dma_q);
                        pending = 0;
                    }
                }
            }
        }
    }
    if (pending > 0) {
        dma_queue_flush(dma_q);
    }
}

static int htp_mmid_prepare_mapping(
    struct htp_ops_context * octx,
    const struct htp_tensor * ids,
    uint8_t * ids_stack,
    size_t ids_stack_size,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t act_nrows,
    const int32_t ** out_ids_data,
    void ** out_ids_alloc,
    void ** out_map_alloc,
    uint32_t ** out_matrix_row_counts,
    struct mmid_row_mapping ** out_matrix_rows,
    uint32_t * out_mapping_stride,
    uint32_t * out_n_skipped
) {
    const size_t ids_bytes = hex_round_up(htp_tensor_nbytes(ids), 128);
    uint8_t * ids_buf      = NULL;
    void *    ids_alloc    = NULL;
    uint8_t * spad_base    = (uint8_t *) octx->ctx->ddr_spad_base;
    size_t    spad_avail   = octx->ctx->ddr_spad_size;

    if (ids_bytes <= ids_stack_size) {
        ids_buf = ids_stack;
    } else if (ids_bytes <= spad_avail) {
        ids_buf     = spad_base;
        spad_base  += ids_bytes;
        spad_avail -= ids_bytes;
    } else {
        ids_alloc = memalign(128, ids_bytes);
        if (!ids_alloc) {
            return HTP_STATUS_INTERNAL_ERR;
        }
        ids_buf = (uint8_t *) ids_alloc;
    }

    if (htp_tensor_read(&octx->ctx->main_th, ids_buf, ids_bytes, ids) == 0) {
        free(ids_alloc);
        return HTP_STATUS_INTERNAL_ERR;
    }

    const int32_t * ids_data = (const int32_t *) ids_buf;

    void *     map_alloc         = NULL;
    uint8_t  * mapping_buf       = spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = NULL;
    struct mmid_row_mapping * matrix_rows = NULL;
    uint32_t n_skipped = 0;

    if (act_nrows > 1) {
        const size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
        if (matrix_row_counts_size > spad_avail) {
            map_alloc = memalign(128, matrix_row_counts_size);
            if (!map_alloc) {
                free(ids_alloc);
                return HTP_STATUS_INTERNAL_ERR;
            }
            mapping_buf = (uint8_t *) map_alloc;
        }
        matrix_row_counts = (uint32_t *) mapping_buf;

        hvx_splat_u(matrix_row_counts, Q6_V_vzero(), n_as, sizeof(uint32_t));
        n_skipped = scan_expert_ids(ids_data, ids->ne[1], n_ids, n_as, matrix_row_counts, NULL, 0);

        uint32_t max_count = hvx_reduce_max_i32((const uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_count > 0 ? max_count : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * sizeof(struct mmid_row_mapping);
        const size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        if (total_map_size > spad_avail) {
            free(map_alloc);
            map_alloc = memalign(128, total_map_size);
            if (!map_alloc) {
                free(ids_alloc);
                return HTP_STATUS_INTERNAL_ERR;
            }
            mapping_buf = (uint8_t *) map_alloc;
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (struct mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        hvx_splat_u(matrix_row_counts, Q6_V_vzero(), n_as, sizeof(uint32_t));
        scan_expert_ids(ids_data, ids->ne[1], n_ids, n_as, matrix_row_counts, matrix_rows, mapping_stride);
    } else {
        n_skipped = count_skipped_ids_single(ids_data, n_ids);
    }

    *out_ids_data          = ids_data;
    *out_ids_alloc         = ids_alloc;
    *out_map_alloc         = map_alloc;
    *out_matrix_row_counts = matrix_row_counts;
    *out_matrix_rows       = matrix_rows;
    *out_mapping_stride    = mapping_stride;
    *out_n_skipped         = n_skipped;

    return HTP_STATUS_OK;
}

int op_matmul_id(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;

    const int status = htp_mm_init_context(octx, kparams);
    if (status != HTP_STATUS_OK) {
        return status;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    const struct htp_tensor * restrict ids = octx->src[2];

    const size_t src0_row_size = nb01;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t src0_nrows = ne01;  // per expert
    const uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    const int n_ids = ids->ne[0];  // n_expert_used
    const int n_as  = ne02;        // n_expert

    uint8_t __attribute__((aligned(128))) ids_stack[256];
    const int32_t * ids_data          = NULL;
    void *          ids_alloc         = NULL;
    void *          map_alloc         = NULL;
    uint32_t *      matrix_row_counts = NULL;
    struct mmid_row_mapping * matrix_rows = NULL;
    uint32_t        mapping_stride    = 1;
    uint32_t        n_skipped         = 0;

    int map_status = htp_mmid_prepare_mapping(octx, ids, ids_stack, sizeof(ids_stack), n_ids, n_as, act_nrows,
                                              &ids_data, &ids_alloc, &map_alloc,
                                              &matrix_row_counts, &matrix_rows,
                                              &mapping_stride, &n_skipped);
    if (map_status != HTP_STATUS_OK) {
        return map_status;
    }

    if (n_skipped > 0) {
        zero_skipped_dst_rows(octx, ids_data, ids->ne[1], n_ids, dst);
    }

    mmctx->ids_data             = ids_data;
    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_row_size_padded = src0_row_size_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_start          = 0;
    mmctx->cur_m_rows           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    int s;
    if (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } else {
        uint32_t src0_row_start = 0;
        uint32_t src0_row_end   = src0_nrows;
        const uint32_t dst_elem_size = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
        if (octx->ctx->mdev.count > 1) {
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, 32, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            src0_row_start = range.start;
            src0_row_end   = range.start + range.count;
        }

        if (src0_row_start >= src0_row_end) {
            free(map_alloc);
            free(ids_alloc);
            return HTP_STATUS_OK;
        }

        const uint32_t nrows = src0_row_end - src0_row_start;
        mmctx->src0_row_start = src0_row_start;
        mmctx->src0_row_end   = src0_row_end;

        mmctx->src0_nrows_per_hvx = fastdiv(nrows + octx->n_hvx - 1, &octx->n_hvx_div);
        mmctx->src0_nrows_per_hvx = hex_round_up(mmctx->src0_nrows_per_hvx, 32);

        mmctx->store_row = resolve_hvx_tiled_store_row_fn(dst_elem_size, false, sizeof(float));

        if (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } else {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    free(map_alloc);
    free(ids_alloc);

    return s;
}

int op_matmul_id_nx(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;

    const int status = htp_mm_init_context(octx, kparams);
    if (status != HTP_STATUS_OK) {
        return status;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    const uint32_t n_weights = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];

    mmctx->act = act;

    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    const int n_ids = ids->ne[0];
    const int n_as  = src0->ne[2];

    uint8_t __attribute__((aligned(128))) ids_stack[256];
    const int32_t * ids_data          = NULL;
    void *          ids_alloc         = NULL;
    void *          map_alloc         = NULL;
    uint32_t *      matrix_row_counts = NULL;
    struct mmid_row_mapping * matrix_rows = NULL;
    uint32_t        mapping_stride    = 1;
    uint32_t        n_skipped         = 0;

    int map_status = htp_mmid_prepare_mapping(octx, ids, ids_stack, sizeof(ids_stack), n_ids, n_as, act_nrows,
                                              &ids_data, &ids_alloc, &map_alloc,
                                              &matrix_row_counts, &matrix_rows,
                                              &mapping_stride, &n_skipped);
    if (map_status != HTP_STATUS_OK) {
        return map_status;
    }

    if (n_skipped > 0) {
        for (uint32_t p = 0; p < n_weights; ++p) {
            const struct htp_tensor * restrict d = octx->dsts[p];
            if (d) {
                zero_skipped_dst_rows(octx, ids_data, ids->ne[1], n_ids, d);
            }
        }
    }

    mmctx->ids_data             = ids_data;
    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_row_size_padded = src0_row_size_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_start          = 0;
    mmctx->cur_m_rows           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    int s;
    if (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } else {
        if (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } else {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    free(map_alloc);
    free(ids_alloc);

    return s;
}
static int op_matmul_nx_impl(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const int status = htp_mm_init_context(octx, kparams);
    if (status != HTP_STATUS_OK) {
        return status;
    }

    if (octx->op == HTP_OP_MUL_MAT_NX_GLU) {
        return htp_mm_kernel_is_hmx_int(kparams->kernel_type)
            ? hmx_mm_nx_glu_int_2d(octx, kparams)
            : hmx_mm_nx_glu_fp16_2d(octx, kparams);
    }

    if (kparams->n_hmx) {
        return hmx_mm_nx_2d_f32(octx, kparams);
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const uint32_t n_weights = kparams->n_weights;

    const struct htp_tensor * restrict src0 = octx->src[0]; // first weight
    const struct htp_tensor * restrict act  = octx->src[n_weights]; // activation x

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    const size_t src0_row_size = src0->nb[1];

    if (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t qk = HTP_MM_ACT_BLOCK_K;
    const uint32_t nb = (act->ne[0] + qk - 1) / qk;
    const uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (act_nrows < octx->n_hvx) {
        n_quant_tasks = MIN(total_nb, octx->n_hvx);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(act_nrows, octx->n_hvx);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }

    const size_t act_row_size  = kparams->act_row_size;
    const size_t act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    size_t max_dst_row_size = 0;
    for (uint32_t p = 0; p < n_weights; ++p) {
        if (octx->dsts[p]) {
            const size_t sz = (size_t) octx->dsts[p]->ne[0] * sizeof(float);
            if (sz > max_dst_row_size) max_dst_row_size = sz;
        }
    }
    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_hvx,
                                 max_dst_row_size, src0_row_size, act_elem_size, 0, kparams->n_prefetch, false, true);

    const size_t vtcm_size = L.total_bytes;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_act_raw, quant_task_func != NULL);
    hvx_mm_iq2xxs_lut_init(base, &L);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = htp_mm_act_raw_row_size(act->ne[0], act_elem_size);

    mmctx->vtcm_src0_size_per_hvx = fastdiv(L.src0_bytes, &octx->n_hvx_div);
    mmctx->vtcm_act_size_per_hvx  = L.act_bytes;
    mmctx->vtcm_dst_size_per_hvx  = fastdiv(L.dst_bytes, &octx->n_hvx_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    if (quant_task_func) {
        hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
        mmctx->n_quant_rows_per_hvx = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
        mmctx->n_quant_tasks = n_quant_tasks;
        work_queue_run(octx->ctx->hvx_queue, quant_task_func, mmctx, n_quant_tasks);
    } else {
        hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
    }

    work_queue_run(octx->ctx->hvx_queue, hvx_mm_nx_2d_tiled, mmctx, octx->n_hvx);

    return HTP_STATUS_OK;
}

int op_matmul_nx(struct htp_ops_context * octx) {
    return op_matmul_nx_impl(octx);
}
