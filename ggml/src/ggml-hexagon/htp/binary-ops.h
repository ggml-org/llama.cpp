#ifndef HTP_BINARY_OPS_H
#define HTP_BINARY_OPS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hex-common.h"
#include "htp-ops.h"
#include "htp-vtcm.h"

enum htp_binary_kernel_type {
    HTP_BINARY_KERNEL_SAME_SHAPE = 0,
    HTP_BINARY_KERNEL_ROW_BCAST,
    HTP_BINARY_KERNEL_SCALAR_DMA,
    HTP_BINARY_KERNEL_SCALAR,
    HTP_BINARY_KERNEL_ADD_ID,
    HTP_BINARY_KERNEL_COMPLEX,
    HTP_BINARY_KERNEL_REPEAT,
};

struct htp_binary_kernel_params {
    uint32_t kernel_type;
    uint32_t n_threads;
    uint32_t rows_per_buffer;

    uint32_t src0_row_size_aligned;
    uint32_t src1_row_size_aligned;
    uint32_t dst_row_size_aligned;

    uint32_t src1_size;
    uint32_t vtcm_size;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_binary_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_binary_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#endif

struct htp_binary_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t src1_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t src1_spad_half_size;
    size_t dst_spad_half_size;

    size_t src1_size;
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

    const size_t spad_row_total = (kparams->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_row_size_aligned + kparams->src1_row_size_aligned + kparams->dst_row_size_aligned)
        : 2 * (kparams->src0_row_size_aligned + kparams->dst_row_size_aligned);

    if (spad_row_total == 0 || vtcm_size < kparams->src1_size) {
        return;
    }

    const size_t rows_per_buffer = (vtcm_size - kparams->src1_size) / (n_threads * spad_row_total);
    if (rows_per_buffer == 0) {
        return;
    }

    L->rows_per_buffer = (uint32_t) rows_per_buffer;
    L->src1_size = kparams->src1_size;

    L->src0_bytes_per_thread = rows_per_buffer * 2 * kparams->src0_row_size_aligned;
    L->dst_bytes_per_thread  = rows_per_buffer * 2 * kparams->dst_row_size_aligned;
    L->src1_bytes_per_thread = (kparams->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE)
        ? rows_per_buffer * 2 * kparams->src1_row_size_aligned
        : 0;

    L->src0_spad_half_size = L->src0_bytes_per_thread / 2;
    L->src1_spad_half_size = L->src1_bytes_per_thread / 2;
    L->dst_spad_half_size  = L->dst_bytes_per_thread / 2;

    const size_t src0_total = n_threads * L->src0_bytes_per_thread;
    const size_t src1_total = (kparams->src1_size > 0)
        ? kparams->src1_size
        : n_threads * L->src1_bytes_per_thread;
    const size_t dst_total  = n_threads * L->dst_bytes_per_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_total);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_total);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_total);

    L->total_bytes = off;
}

#endif
