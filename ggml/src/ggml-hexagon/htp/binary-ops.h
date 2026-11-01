#ifndef HTP_BINARY_OPS_H
#define HTP_BINARY_OPS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-ops.h"
#include "htp-vtcm.h"

enum htp_binary_kernel_type {
    HTP_BINARY_KERNEL_SAME_SHAPE = 0,
    HTP_BINARY_KERNEL_ROW_BCAST,
    HTP_BINARY_KERNEL_SCALAR,
    HTP_BINARY_KERNEL_ADD_ID,
    HTP_BINARY_KERNEL_COMPLEX,
    HTP_BINARY_KERNEL_REPEAT,
    HTP_BINARY_KERNEL_CHUNKED,
};

struct htp_binary_kernel_params {
    uint8_t  kernel_type;
    uint8_t  split_at_ne01;
    uint8_t  split_at_ne02;
    uint8_t  src1_strided;

    uint8_t  n_srcs;
    uint8_t  is_scalar;
    uint8_t  pad[2];

    uint32_t n_threads;
    uint32_t rows_per_buffer;

    uint32_t src0_row_size_aligned;
    uint32_t src1_row_size_aligned;
    uint32_t dst_row_size_aligned;

    uint32_t src1_size;
    uint32_t src2_size;
    uint32_t vtcm_size;

    uint32_t chunk_size;
    uint32_t chunk_bytes;

    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02;
    struct fastdiv_values div_ne012;

    struct fastdiv_values div_ne11;
    struct fastdiv_values div_ne12;
    struct fastdiv_values div_ne13;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_binary_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

struct ggml_tensor;
static inline bool htp_binary_strides_safe(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    size_t elem_size
) {
    if ((src0->ne[0] != 1 && src0->nb[0] != elem_size) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_size) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_size)) {
        return false;
    }
    if ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_size) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_size) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_size)) {
        return false;
    }
    return true;
}
#else
_Static_assert(sizeof(struct htp_binary_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#endif

struct htp_binary_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t src1_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t src1_spad_half_size;
    size_t dst_spad_half_size;

    size_t src1_size;
    size_t src2_size;
    uint32_t rows_per_buffer;
};

static inline void htp_binary_vtcm_layout_build(
    struct htp_binary_vtcm_layout * L,
    const struct htp_binary_kernel_params * kparams,
    size_t vtcm_size
) {
    memset(L, 0, sizeof(*L));

    const uint32_t n_threads = kparams->n_threads;
    if (n_threads == 0) {
        return;
    }

    const uint32_t n_srcs = kparams->n_srcs > 2 ? kparams->n_srcs : 2;

    if (kparams->kernel_type == HTP_BINARY_KERNEL_CHUNKED) {
        const size_t chunk_bytes = kparams->chunk_bytes;
        if (chunk_bytes == 0) {
            return;
        }

        L->src0_bytes_per_thread = 2 * chunk_bytes;
        L->src1_bytes_per_thread = kparams->is_scalar ? 0 : (2 * (n_srcs - 1) * chunk_bytes);
        L->dst_bytes_per_thread  = 2 * chunk_bytes;

        L->src0_spad_half_size = chunk_bytes;
        L->src1_spad_half_size = kparams->is_scalar ? 0 : chunk_bytes;
        L->dst_spad_half_size  = chunk_bytes;

        L->rows_per_buffer = 1;
        L->src1_size = 0;

        const size_t src0_total = n_threads * L->src0_bytes_per_thread;
        const size_t src1_total = kparams->is_scalar ? 128 : (n_threads * L->src1_bytes_per_thread);
        const size_t dst_total  = n_threads * L->dst_bytes_per_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_total);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_total);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_total);

        if (off > vtcm_size) {
            return;
        }

        L->total_bytes = off;
        return;
    }

    const size_t spad_row_total = (kparams->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_row_size_aligned + (n_srcs - 1) * kparams->src1_row_size_aligned + kparams->dst_row_size_aligned)
        : 2 * (kparams->src0_row_size_aligned + kparams->dst_row_size_aligned);

    const size_t static_vtcm_size = kparams->src1_size + kparams->src2_size;
    if (spad_row_total == 0 || vtcm_size < static_vtcm_size) {
        return;
    }

    const size_t rows_per_buffer = (kparams->rows_per_buffer > 0)
        ? kparams->rows_per_buffer
        : (vtcm_size - static_vtcm_size) / (n_threads * spad_row_total);
    if (rows_per_buffer == 0) {
        return;
    }

    L->rows_per_buffer = (uint32_t) rows_per_buffer;
    L->src1_size = kparams->src1_size;
    L->src2_size = kparams->src2_size;

    L->src0_bytes_per_thread = rows_per_buffer * 2 * kparams->src0_row_size_aligned;
    L->dst_bytes_per_thread  = rows_per_buffer * 2 * kparams->dst_row_size_aligned;
    L->src1_bytes_per_thread = (kparams->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE)
        ? rows_per_buffer * 2 * (n_srcs - 1) * kparams->src1_row_size_aligned
        : 0;

    L->src0_spad_half_size = L->src0_bytes_per_thread / 2;
    L->src1_spad_half_size = (kparams->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE)
        ? rows_per_buffer * kparams->src1_row_size_aligned
        : 0;
    L->dst_spad_half_size  = L->dst_bytes_per_thread / 2;

    const size_t src0_total = n_threads * L->src0_bytes_per_thread;
    const size_t src1_total = (kparams->src1_size > 0)
        ? kparams->src1_size
        : n_threads * L->src1_bytes_per_thread;
    const size_t src2_total = kparams->src2_size;
    const size_t dst_total  = n_threads * L->dst_bytes_per_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_total);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_total);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_total);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_total);

    L->total_bytes = off;
}

static inline bool htp_binary_solve_chunked(
    struct htp_binary_vtcm_layout * L,
    struct htp_binary_kernel_params * kparams,
    uint32_t total_elems,
    uint32_t elem_size,
    uint32_t max_threads,
    uint32_t n_srcs,
    bool is_scalar,
    size_t vtcm_size
) {
    const size_t n_bufs = is_scalar ? 2 : (size_t) n_srcs + 1;
    const size_t scalar_bytes = is_scalar ? 128 : 0;
    const size_t budget_per_buf = (max_threads > 0 && vtcm_size > scalar_bytes)
        ? hex_align_down((vtcm_size - scalar_bytes) / (2 * max_threads * n_bufs), 128)
        : 0;
    const uint32_t max_chunk_bytes = (budget_per_buf > 0 && budget_per_buf < 32768) ? (uint32_t) budget_per_buf : 32768u;
    const uint32_t max_chunk_elems = max_chunk_bytes / elem_size;
    if (max_chunk_elems == 0) {
        return false;
    }

    const uint32_t min_chunk_elems = 256 < max_chunk_elems ? 256 : max_chunk_elems;
    const uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * max_threads) - 1) / (2 * max_threads), 32);
    const uint32_t chunk_size  = (target_chunk_elems < min_chunk_elems) ? min_chunk_elems
                               : (target_chunk_elems > max_chunk_elems) ? max_chunk_elems
                               : target_chunk_elems;
    const uint32_t chunk_bytes  = hex_round_up(chunk_size * elem_size, 128);
    const uint32_t total_chunks = (total_elems + chunk_size - 1) / chunk_size;
    const uint32_t n_threads    = (total_chunks >= 2) ? (max_threads < total_chunks ? max_threads : total_chunks) : 1;

    kparams->kernel_type           = HTP_BINARY_KERNEL_CHUNKED;
    kparams->n_threads             = n_threads;
    kparams->n_srcs                = (uint8_t) n_srcs;
    kparams->rows_per_buffer       = 1;
    kparams->src1_row_size_aligned = is_scalar ? 0 : kparams->src0_row_size_aligned;
    kparams->src1_size             = 0;
    kparams->chunk_size            = chunk_size;
    kparams->chunk_bytes           = chunk_bytes;
    kparams->is_scalar             = is_scalar ? 1 : 0;

    htp_binary_vtcm_layout_build(L, kparams, vtcm_size);
    if (L->total_bytes == 0 || L->total_bytes > vtcm_size) {
        return false;
    }

    kparams->vtcm_size = (uint32_t) L->total_bytes;
    return true;
}

#endif
