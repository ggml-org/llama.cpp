#include "htp-ctx.h"
#include "htp-ops.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "hex-dma.h"
#include "vtcm-utils.h"
#include "hvx-utils.h"
#include "hex-fastdiv.h"
#include <string.h>

struct htp_concat_context {
    struct htp_ops_context * octx;
    uint32_t dim;
    uint32_t nrows_per_thread;
    struct fastdiv_values div_ne0;
    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne2;
};

static void concat_2d_f32_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];
    const uint32_t ne1      = dst->ne[1];

    const uint32_t start_i = ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < ne1) ? (start_i + cctx->nrows_per_thread) : ne1;
    if (start_i >= end_i) return;

    dma_queue * q = octx->ctx->dma[ith];

    uint8_t * spad0_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * spad1_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;

    const uint32_t block_i = 32;
    const uint32_t spad1_stride = block_i * sizeof(float);

    int32_t offsets[32] __attribute__((aligned(128)));
    for(int k=0; k<32; k++) {
        offsets[k] = k * spad1_stride;
    }
    HVX_Vector vv = *(HVX_Vector*)offsets;
    const uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 32);
    const uint32_t spad0_row_bytes = hex_round_up((src0_ne0 + src1_ne0_padded) * sizeof(float), VLEN);
    uint32_t mu = src1_ne0_padded * spad1_stride;

    for (uint32_t i = start_i; i < end_i; i += block_i) {
        uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;

        uint32_t src1_width_bytes = current_block_i * sizeof(float);
        uint8_t * src1_ptr = (uint8_t *)src1->data + i * src1->nb[1];
        dma_queue_push(q, dma_make_ptr(spad1_base, src1_ptr), spad1_stride, src1->nb[0], src1_width_bytes, src1_ne0);

        uint32_t src0_row_bytes = src0_ne0 * sizeof(float);
        uint8_t * src0_ptr = (uint8_t *)src0->data + i * src0->nb[1];
        dma_queue_push(q, dma_make_ptr(spad0_base, src0_ptr), spad0_row_bytes, src0->nb[1], src0_row_bytes, current_block_i);

        dma_queue_pop(q); // src1

        HVX_Vector * vtcm_tmp = (HVX_Vector *)(spad1_base + src1_ne0_padded * spad1_stride);

        for (uint32_t j = 0; j < src1_ne0_padded; j += 32) {
            #pragma unroll(4)
            for (uint32_t ii = 0; ii < current_block_i; ii++) {
                size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(float));
                Q6_vgather_ARMVw(&vtcm_tmp[ii], rt, mu, vv);
                uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(float);
                hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
            }
        }

        dma_queue_pop(q); // src0

        uint8_t * dst_ptr = (uint8_t *)dst->data + i * dst->nb[1];
        dma_queue_push(q, dma_make_ptr(dst_ptr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(float), current_block_i);

        dma_queue_pop(q);
    }
}

static void concat_2d_f16_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];
    const uint32_t ne1      = dst->ne[1];

    const uint32_t start_i = ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < ne1) ? (start_i + cctx->nrows_per_thread) : ne1;
    if (start_i >= end_i) return;

    dma_queue * q = octx->ctx->dma[ith];

    uint8_t * spad0_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * spad1_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;

    const uint32_t block_i = 64;
    const uint32_t spad1_stride = block_i * sizeof(__fp16);

    int16_t offsets[64] __attribute__((aligned(128)));
    for(int k=0; k<64; k++) {
        offsets[k] = k * spad1_stride;
    }
    HVX_Vector vv = *(HVX_Vector*)offsets;
    const uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 64);
    const uint32_t spad0_row_bytes = hex_round_up((src0_ne0 + src1_ne0_padded) * sizeof(__fp16), VLEN);
    uint32_t mu = src1_ne0_padded * spad1_stride;

    for (uint32_t i = start_i; i < end_i; i += block_i) {
        uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;

        uint32_t src1_width_bytes = current_block_i * sizeof(__fp16);
        uint8_t * src1_ptr = (uint8_t *)src1->data + i * src1->nb[1];
        dma_queue_push(q, dma_make_ptr(spad1_base, src1_ptr), spad1_stride, src1->nb[0], src1_width_bytes, src1_ne0);

        uint32_t src0_row_bytes = src0_ne0 * sizeof(__fp16);
        uint8_t * src0_ptr = (uint8_t *)src0->data + i * src0->nb[1];
        dma_queue_push(q, dma_make_ptr(spad0_base, src0_ptr), spad0_row_bytes, src0->nb[1], src0_row_bytes, current_block_i);

        dma_queue_pop(q); // src1

        HVX_Vector * vtcm_tmp = (HVX_Vector *)(spad1_base + src1_ne0_padded * spad1_stride);

        for (uint32_t j = 0; j < src1_ne0_padded; j += 64) {
            #pragma unroll(4)
            for (uint32_t ii = 0; ii < current_block_i; ii++) {
                size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(__fp16));
                Q6_vgather_ARMVh(&vtcm_tmp[ii], rt, mu, vv);
                uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(__fp16);
                hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
            }
        }

        dma_queue_pop(q); // src0

        uint8_t * dst_ptr = (uint8_t *)dst->data + i * dst->nb[1];
        dma_queue_push(q, dma_make_ptr(dst_ptr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(__fp16), current_block_i);

        dma_queue_pop(q);
    }
}

static void concat_generic(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const int dim = cctx->dim;
    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    const uint32_t ne[4] = {dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]};
    const uint32_t total_elements = ne[0] * ne[1] * ne[2] * ne[3];
    const uint32_t chunk_size = (total_elements + nth - 1) / nth;

    const uint32_t start_idx = MIN(ith * chunk_size, total_elements);
    const uint32_t end_idx   = MIN(start_idx + chunk_size, total_elements);

    // Naive scalar element-wise copy
    for (uint32_t idx = start_idx; idx < end_idx; idx++) {
        uint32_t idx_div_ne0 = fastdiv(idx, &cctx->div_ne0);
        uint32_t i0 = idx - idx_div_ne0 * ne[0];

        uint32_t idx_div_ne01 = fastdiv(idx_div_ne0, &cctx->div_ne1);
        uint32_t i1 = idx_div_ne0 - idx_div_ne01 * ne[1];

        uint32_t idx_div_ne012 = fastdiv(idx_div_ne01, &cctx->div_ne2);
        uint32_t i2 = idx_div_ne01 - idx_div_ne012 * ne[2];
        uint32_t i3 = idx_div_ne012;

        uint8_t * dst_ptr = (uint8_t *)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];

        uint32_t idx_dim = 0;
        if (dim == 0) idx_dim = i0;
        else if (dim == 1) idx_dim = i1;
        else if (dim == 2) idx_dim = i2;
        else if (dim == 3) idx_dim = i3;

        const struct htp_tensor * src = (idx_dim < src0->ne[dim]) ? src0 : src1;

        uint32_t s0 = i0;
        uint32_t s1 = i1;
        uint32_t s2 = i2;
        uint32_t s3 = i3;

        if (dim == 0 && src == src1) s0 -= src0->ne[0];
        if (dim == 1 && src == src1) s1 -= src0->ne[1];
        if (dim == 2 && src == src1) s2 -= src0->ne[2];
        if (dim == 3 && src == src1) s3 -= src0->ne[3];

        uint8_t * src_ptr = (uint8_t *)src->data + s3 * src->nb[3] + s2 * src->nb[2] + s1 * src->nb[1] + s0 * src->nb[0];

        if (type_size == 4) {
            *(float*)dst_ptr = *(float*)src_ptr;
        } else {
            *(__fp16*)dst_ptr = *(__fp16*)src_ptr;
        }
    }
}

int op_concat(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    int dim = octx->op_params[0];

    bool is_2d = dst->ne[2] == 1 && dst->ne[3] == 1;

    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;
    bool is_src1_transposed  = (src1->nb[0] > src1->nb[1]);
    bool is_src0_transposed  = (src0->nb[0] > src0->nb[1]);

    uint32_t n_threads = octx->n_threads;
    struct htp_concat_context cctx;
    cctx.octx = octx;
    cctx.dim = dim;
    cctx.div_ne0 = init_fastdiv_values(dst->ne[0]);
    cctx.div_ne1 = init_fastdiv_values(dst->ne[1]);
    cctx.div_ne2 = init_fastdiv_values(dst->ne[2]);

    void (*worker_func)(unsigned int, unsigned int, void *) = concat_generic;

    if (dim == 0 && is_2d && is_src1_transposed && !is_src0_transposed) {
        n_threads = MIN(dst->ne[1], n_threads);
        if (n_threads < 1) {
            n_threads = 1;
        }
        uint32_t block_i = (type_size == 4) ? 32 : 64;

        cctx.nrows_per_thread = hmx_ceil_div(dst->ne[1], n_threads);

        // Allocate VTCM
        uint32_t spad1_stride = block_i * type_size;

        uint32_t src1_ne0_padded = hex_round_up(src1->ne[0], block_i);
        uint32_t spad0_row_bytes = hex_round_up((src0->ne[0] + src1_ne0_padded) * type_size, VLEN);

        octx->src0_spad.size_per_thread = block_i * spad0_row_bytes;
        octx->src1_spad.size_per_thread = src1_ne0_padded * spad1_stride + block_i * VLEN;

        octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
        octx->src1_spad.size = n_threads * octx->src1_spad.size_per_thread;

        if (octx->src0_spad.size + octx->src1_spad.size > octx->ctx->vtcm_size) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }

        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
        octx->src0_spad.src  = NULL;
        octx->src1_spad.src  = NULL;

        if (type_size == 4) {
            worker_func = concat_2d_f32_transposed;
        } else {
            worker_func = concat_2d_f16_transposed;
        }
    }

    worker_pool_run_func(octx->ctx->worker_pool, worker_func, &cctx, n_threads);
    return HTP_STATUS_OK;
}
