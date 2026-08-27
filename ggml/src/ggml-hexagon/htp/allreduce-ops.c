#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <stdatomic.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "htp-tensor.h"
#include "hex-dma.h"
#include "hex-profile.h"
#include "allreduce-ops.h"

struct htp_allreduce_context {
    struct htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t n_dsts;
    uint32_t nelem;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t row_size_aligned;
    uint32_t rank_elem_start;
    uint32_t rank_nelem;
    uint32_t elems_per_thread;
    uint32_t block_elems;
    uint32_t vtcm_size_per_thread;
    bool     is_row_bcast;
    uint8_t * src_spad_base[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_base;
    uint8_t * res_spad_base;
};

#define DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD)                                             \
static void allreduce_thread_dma_1d_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                       \
    struct htp_allreduce_context * actx = (struct htp_allreduce_context *) data;                                      \
    struct htp_ops_context * octx = actx->octx;                                                                       \
                                                                                                                      \
    const uint32_t n_ranks     = actx->n_ranks;                                                                       \
    const uint32_t n_dsts      = actx->n_dsts;                                                                        \
    const uint32_t block_elems = actx->block_elems;                                                                   \
                                                                                                                      \
    const uint32_t dr  = actx->elems_per_thread;                                                                      \
    const uint32_t ir0 = actx->rank_elem_start + dr * ith;                                                            \
    const uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_start + actx->rank_nelem);                                     \
    if (ir0 >= ir1) return;                                                                                           \
                                                                                                                      \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                            \
    dma_queue * q = octx->ctx->dma[ith];                                                                              \
                                                                                                                      \
    uint8_t * src_spad_base[HTP_ALLREDUCE_MAX_RANKS];                                                                 \
    for (uint32_t s = 0; s < n_ranks; s++) {                                                                          \
        src_spad_base[s] = actx->src_spad_base[s] + (ith * actx->vtcm_size_per_thread);                               \
    }                                                                                                                 \
    uint8_t * dst_spad_base = actx->dst_spad_base + (ith * actx->vtcm_size_per_thread);                               \
    uint8_t * res_spad_base = HAS_ADD ? (actx->res_spad_base + (ith * actx->vtcm_size_per_thread)) : NULL;            \
                                                                                                                      \
    const size_t spad_half = actx->vtcm_size_per_thread / 2;                                                          \
    uint32_t ir_prefetch = ir0;                                                                                       \
    int spad_idx = 0;                                                                                                 \
                                                                                                                      \
    for (int k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                                \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir_prefetch);                                                     \
        size_t   cur_bytes = cur_elems * sizeof(TYPE);                                                                \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                      \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                       \
            uint8_t * d_ddr = (uint8_t *) octx->dsts[d]->data + ir_prefetch * sizeof(TYPE);                           \
            dma_queue_push(q, dma_make_ptr(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 0);                       \
        }                                                                                                             \
        for (uint32_t s = 0; s < n_ranks; s++) {                                                                      \
            uint8_t * s_spad = src_spad_base[s] + spad_idx * spad_half;                                               \
            const uint8_t * s_ddr = (const uint8_t *) octx->src[s]->data + ir_prefetch * sizeof(TYPE);                \
            dma_queue_push(q, dma_make_ptr(s_spad, s_ddr), cur_bytes, cur_bytes, cur_bytes, 1);                       \
        }                                                                                                             \
        if (HAS_ADD) {                                                                                                \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                                  \
            const uint8_t * r_ddr = (const uint8_t *) octx->src[2 * n_ranks]->data + ir_prefetch * sizeof(TYPE);      \
            dma_queue_push(q, dma_make_ptr(r_spad, r_ddr), cur_bytes, cur_bytes, cur_bytes, 1);                       \
        }                                                                                                             \
        ir_prefetch += cur_elems;                                                                                     \
        spad_idx ^= 1;                                                                                                \
    }                                                                                                                 \
                                                                                                                      \
    for (uint32_t ir = ir0; ir < ir1; ) {                                                                             \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir);                                                              \
        size_t   cur_bytes = cur_elems * sizeof(TYPE);                                                                \
        uint8_t * d_spad = NULL;                                                                                      \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                       \
            d_spad = (uint8_t *) dma_queue_pop(q).src;                                                                \
        }                                                                                                             \
        uint8_t * s_spad[HTP_ALLREDUCE_MAX_RANKS];                                                                    \
        for (uint32_t s = 0; s < n_ranks; s++) {                                                                      \
            s_spad[s] = (uint8_t *) dma_queue_pop(q).dst;                                                             \
        }                                                                                                             \
        uint8_t * r_spad = HAS_ADD ? (uint8_t *) dma_queue_pop(q).dst : NULL;                                         \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                             \
        HVX_ADD_FN(d_spad, s_spad[0], s_spad[1], cur_elems);                                                          \
        for (uint32_t s = 2; s < n_ranks; s++) {                                                                      \
            HVX_ADD_FN(d_spad, d_spad, s_spad[s], cur_elems);                                                         \
        }                                                                                                             \
        if (HAS_ADD) {                                                                                                \
            HVX_ADD_FN(d_spad, d_spad, r_spad, cur_elems);                                                            \
        }                                                                                                             \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                              \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                       \
            uint8_t * d_ddr = (uint8_t *) octx->dsts[d]->data + ir * sizeof(TYPE);                                    \
            dma_queue_push(q, dma_make_ptr(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 1);                       \
        }                                                                                                             \
        if (ir_prefetch < ir1) {                                                                                      \
            uint32_t next_elems = MIN(block_elems, ir1 - ir_prefetch);                                                \
            size_t   next_bytes = next_elems * sizeof(TYPE);                                                          \
            for (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
                const uint8_t * s_next = (const uint8_t *) octx->src[s]->data + ir_prefetch * sizeof(TYPE);           \
                dma_queue_push(q, dma_make_ptr(s_spad[s], s_next), next_bytes, next_bytes, next_bytes, 1);            \
            }                                                                                                         \
            if (HAS_ADD) {                                                                                            \
                const uint8_t * r_next = (const uint8_t *) octx->src[2 * n_ranks]->data + ir_prefetch * sizeof(TYPE); \
                dma_queue_push(q, dma_make_ptr(r_spad, r_next), next_bytes, next_bytes, next_bytes, 1);               \
            }                                                                                                         \
            ir_prefetch += next_elems;                                                                                \
        }                                                                                                             \
        ir += cur_elems;                                                                                              \
    }                                                                                                                 \
    dma_queue_flush(q);                                                                                               \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_add_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     float,  hvx_add_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f16, __fp16, hvx_add_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f32, float,  hvx_add_f32_aaa, 1)

#define DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD, IS_ROW_BCAST)                                                           \
static void allreduce_thread_dma_2d_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                                   \
    struct htp_allreduce_context * actx = (struct htp_allreduce_context *) data;                                                                  \
    struct htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    const uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    const uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    const uint32_t ne0              = actx->ne0;                                                                                                  \
    const uint32_t block_rows       = actx->block_elems;                                                                                          \
    const uint32_t row_size_aligned = actx->row_size_aligned;                                                                                     \
    const uint32_t row_bytes        = ne0 * sizeof(TYPE);                                                                                         \
                                                                                                                                                  \
    const uint32_t dr  = actx->elems_per_thread;                                                                                                  \
    const uint32_t r0  = actx->rank_elem_start + dr * ith;                                                                                        \
    const uint32_t r1  = MIN(r0 + dr, actx->rank_elem_start + actx->rank_nelem);                                                                  \
    if (r0 >= r1) return;                                                                                                                         \
                                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_queue * q = octx->ctx->dma[ith];                                                                                                          \
                                                                                                                                                  \
    uint8_t * src_spad_base[HTP_ALLREDUCE_MAX_RANKS];                                                                                             \
    for (uint32_t s = 0; s < n_ranks; s++) {                                                                                                      \
        src_spad_base[s] = actx->src_spad_base[s] + (ith * actx->vtcm_size_per_thread);                                                           \
    }                                                                                                                                             \
    uint8_t * dst_spad_base = actx->dst_spad_base + (ith * actx->vtcm_size_per_thread);                                                           \
    uint8_t * res_spad_base = HAS_ADD ? (IS_ROW_BCAST ? actx->res_spad_base : (actx->res_spad_base + (ith * actx->vtcm_size_per_thread))) : NULL; \
                                                                                                                                                  \
    const size_t spad_half = actx->vtcm_size_per_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    int spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    for (int k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_rows = MIN(block_rows, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                                                  \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            uint8_t * d_ddr  = (uint8_t *) octx->dsts[d]->data + r_prefetch * octx->dsts[d]->nb[1];                                               \
            dma_queue_push(q, dma_make_ptr(d_ddr, d_spad), octx->dsts[d]->nb[1], row_size_aligned, row_bytes, 0);                                 \
        }                                                                                                                                         \
        for (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = src_spad_base[s] + spad_idx * spad_half;                                                                           \
            const uint8_t * s_ddr = (const uint8_t *) octx->src[s]->data + r_prefetch * octx->src[s]->nb[1];                                      \
            dma_queue_push(q, dma_make_ptr(s_spad, s_ddr), row_size_aligned, octx->src[s]->nb[1], row_bytes, cur_rows);                           \
        }                                                                                                                                         \
        if (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                                                              \
            const uint8_t * r_ddr = (const uint8_t *) octx->src[2 * n_ranks]->data + r_prefetch * octx->src[2 * n_ranks]->nb[1];                  \
            dma_queue_push(q, dma_make_ptr(r_spad, r_ddr), row_size_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, cur_rows);                 \
        }                                                                                                                                         \
        r_prefetch += cur_rows;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    for (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_rows = MIN(block_rows, r1 - r);                                                                                              \
        uint8_t * d_spad = NULL;                                                                                                                  \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            d_spad = (uint8_t *) dma_queue_pop(q).src;                                                                                            \
        }                                                                                                                                         \
        uint8_t * s_spad[HTP_ALLREDUCE_MAX_RANKS];                                                                                                \
        for (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            s_spad[s] = (uint8_t *) dma_queue_pop(q).dst;                                                                                         \
        }                                                                                                                                         \
        uint8_t * r_spad = (HAS_ADD && !IS_ROW_BCAST) ? (uint8_t *) dma_queue_pop(q).dst : NULL;                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        for (uint32_t row = 0; row < cur_rows; row++) {                                                                                           \
            uint8_t * d_row = d_spad + row * row_size_aligned;                                                                                    \
            const uint8_t * s0_row = s_spad[0] + row * row_size_aligned;                                                                          \
            const uint8_t * s1_row = s_spad[1] + row * row_size_aligned;                                                                          \
            HVX_ADD_FN(d_row, s0_row, s1_row, ne0);                                                                                               \
            for (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                const uint8_t * ss_row = s_spad[s] + row * row_size_aligned;                                                                      \
                HVX_ADD_FN(d_row, d_row, ss_row, ne0);                                                                                            \
            }                                                                                                                                     \
            if (HAS_ADD) {                                                                                                                        \
                const uint8_t * res_row = IS_ROW_BCAST ? res_spad_base : (r_spad + row * row_size_aligned);                                       \
                HVX_ADD_FN(d_row, d_row, res_row, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            uint8_t * d_ddr = (uint8_t *) octx->dsts[d]->data + r * octx->dsts[d]->nb[1];                                                         \
            dma_queue_push(q, dma_make_ptr(d_ddr, d_spad), octx->dsts[d]->nb[1], row_size_aligned, row_bytes, cur_rows);                          \
        }                                                                                                                                         \
        if (r_prefetch < r1) {                                                                                                                    \
            uint32_t next_rows = MIN(block_rows, r1 - r_prefetch);                                                                                \
            for (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                const uint8_t * s_next = (const uint8_t *) octx->src[s]->data + r_prefetch * octx->src[s]->nb[1];                                 \
                dma_queue_push(q, dma_make_ptr(s_spad[s], s_next), row_size_aligned, octx->src[s]->nb[1], row_bytes, next_rows);                  \
            }                                                                                                                                     \
            if (HAS_ADD && !IS_ROW_BCAST) {                                                                                                       \
                const uint8_t * r_next = (const uint8_t *) octx->src[2 * n_ranks]->data + r_prefetch * octx->src[2 * n_ranks]->nb[1];             \
                dma_queue_push(q, dma_make_ptr(r_spad, r_next), row_size_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, next_rows);           \
            }                                                                                                                                     \
            r_prefetch += next_rows;                                                                                                              \
        }                                                                                                                                         \
        r += cur_rows;                                                                                                                            \
    }                                                                                                                                             \
    dma_queue_flush(q);                                                                                                                           \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_add_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           float,  hvx_add_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f16,       __fp16, hvx_add_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f32,       float,  hvx_add_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f16, __fp16, hvx_add_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f32, float,  hvx_add_f32_aaa, 1, 1)

int op_allreduce(struct htp_ops_context * octx) {
    const struct htp_allreduce_kernel_params * kparams = (const struct htp_allreduce_kernel_params *) octx->kernel_params;
    const struct htp_tensor * dst = octx->dst;

    const uint32_t rank    = (uint32_t) kparams->rank;
    const uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    if (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    const uint32_t fence_seq_entry = (uint32_t) octx->op_params[0];
    const uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    // 1. Entry Barrier: Synchronize all ranks before reading
    struct htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    const struct htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) my_sync->data;

    atomic_store(&my_fence[0], fence_seq_entry);
    asm volatile ("syncht" : : : "memory");
    Q6_dccleaninva_A((void *) my_fence);

    for (uint32_t j = 0; j < n_ranks; j++) {
        if (j == rank) continue;
        const struct htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) peer_sync->data;
        uint64_t spins = 0;
        while (1) {
            Q6_dccleaninva_A((void *) peer_fence);
            uint32_t val = atomic_load(&peer_fence[0]);
            if (val == fence_seq_entry || val == fence_seq_exit) {
                break;
            }
            if (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT: rank %u waiting on %u (fence %p seq %u)\n", rank, j, peer_fence, fence_seq_entry);
                return HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm volatile ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    // 2. Multi-threaded Reduction across assigned rank chunk
    if (nelem > 0) {
        const uint32_t n_threads            = (uint32_t) kparams->n_threads;
        const uint32_t block_elems          = (uint32_t) kparams->block_elems;
        const uint32_t elems_per_thread     = (uint32_t) kparams->elems_per_thread;
        const uint32_t vtcm_size_per_thread = (uint32_t) kparams->vtcm_size_per_thread;

        const bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);

        struct htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.row_size_aligned     = (uint32_t) kparams->row_size_aligned;
        actx.rank_elem_start      = (uint32_t) kparams->rank_elem_start;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_per_thread     = elems_per_thread;
        actx.block_elems          = block_elems;
        actx.vtcm_size_per_thread = vtcm_size_per_thread;
        actx.is_row_bcast         = (kparams->is_row_bcast != 0);

        work_queue_func_t reduce_fun = NULL;
        switch (kparams->kernel_type) {
            case HTP_ALLREDUCE_KERNEL_DMA_1D:
                if (has_add) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_add_f16 : allreduce_thread_dma_1d_add_f32;
                } else {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                break;
            case HTP_ALLREDUCE_KERNEL_DMA_2D:
                if (has_add) {
                    if (kparams->is_row_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_bcast_f16 : allreduce_thread_dma_2d_add_bcast_f32;
                    } else {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_f16 : allreduce_thread_dma_2d_add_f32;
                    }
                } else {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                break;
            default:
                return HTP_STATUS_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
        for (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_base[s] = vtcm_ptr;
            vtcm_ptr += n_threads * vtcm_size_per_thread;
        }
        actx.dst_spad_base = vtcm_ptr;
        vtcm_ptr += n_threads * vtcm_size_per_thread;
        if (has_add) {
            actx.res_spad_base = vtcm_ptr;
            vtcm_ptr += (actx.is_row_bcast ? 1 : n_threads) * vtcm_size_per_thread;
        }

        if (has_add && actx.is_row_bcast) {
            const uint8_t * r_ddr = (const uint8_t *) octx->src[2 * n_ranks]->data;
            const uint32_t row_bytes = actx.ne0 * (dst->type == HTP_TYPE_F16 ? sizeof(__fp16) : sizeof(float));
            dma_queue * q = octx->ctx->dma[0];
            dma_queue_push(q, dma_make_ptr(actx.res_spad_base, r_ddr), actx.row_size_aligned, 0, row_bytes, 1);
            dma_queue_pop(q);
        }

        work_queue_run(octx->ctx->work_queue, reduce_fun, &actx, n_threads);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    atomic_store(&my_fence[0], fence_seq_exit);
    asm volatile ("syncht" : : : "memory");
    Q6_dccleaninva_A((void *) my_fence);

    for (uint32_t j = 0; j < n_ranks; j++) {
        if (j == rank) continue;
        const struct htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) peer_sync->data;
        uint64_t spins = 0;
        while (1) {
            Q6_dccleaninva_A((void *) peer_fence);
            uint32_t val = atomic_load(&peer_fence[0]);
            if (val == fence_seq_exit) {
                break;
            }
            if (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT: rank %u waiting on %u (fence %p seq %u)\n", rank, j, peer_fence, fence_seq_exit);
                return HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm volatile ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    return HTP_STATUS_OK;
}
