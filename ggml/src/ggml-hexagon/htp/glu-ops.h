#ifndef HTP_GLU_OPS_H
#define HTP_GLU_OPS_H

#include "hex-common.h"
#include "htp-ops.h"
#include "htp-vtcm.h"
#include <stdint.h>

struct htp_glu_kernel_params {
    uint32_t n_threads;
    uint32_t vtcm_row_per_thread;
    uint32_t vtcm_size;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_glu_kernel_params) <= 128, "htp_glu_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_glu_kernel_params) <= 128, "htp_glu_kernel_params is too large for kernel_params blob");
#endif

static inline bool htp_op_is_glu(uint32_t opcode) {
    switch (opcode) {
        case HTP_OP_GLU_SWIGLU:
        case HTP_OP_GLU_SWIGLU_OAI:
        case HTP_OP_GLU_SWIGLU_CLAMP:
        case HTP_OP_GLU_GEGLU:
        case HTP_OP_GLU_GEGLU_QUICK:
        case HTP_OP_GLU_GEGLU_ERF:
        case HTP_OP_GLU_REGLU:
        case HTP_OP_GLU_SIGMOID:
            return true;
        default:
            return false;
    }
}

struct htp_glu_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t row_size_aligned;
    size_t src0_bytes_per_thread;
    size_t src1_bytes_per_thread;
    size_t dst_bytes_per_thread;

    uint32_t vtcm_row_per_thread;
    uint32_t block;
};

static inline void htp_glu_vtcm_layout_build(
    struct htp_glu_vtcm_layout * L,
    size_t                       dst_row_size,
    uint32_t                     n_threads,
    uint32_t                     vtcm_row_per_thread
) {
    const size_t row_size_aligned = hex_round_up(dst_row_size, 128);

    L->row_size_aligned      = row_size_aligned;
    L->vtcm_row_per_thread   = vtcm_row_per_thread;
    L->block                 = vtcm_row_per_thread >> 1;

    L->src0_bytes_per_thread = row_size_aligned * vtcm_row_per_thread;
    L->src1_bytes_per_thread = row_size_aligned * vtcm_row_per_thread;
    L->dst_bytes_per_thread  = row_size_aligned * vtcm_row_per_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_src1, L->src1_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread * n_threads);

    L->total_bytes = off;
}

#endif // HTP_GLU_OPS_H
