#ifndef HTP_UNARY_OPS_H
#define HTP_UNARY_OPS_H

#include "hex-common.h"
#include "htp-ops.h"

// Op-specific struct for precomputed unary params
struct htp_unary_kernel_params {
    uint32_t  n_threads;
    uint32_t  col_tile;
    uint32_t  vtcm_row_per_thread;
    uint32_t  block;
    uint32_t  broadcast_weight;

    uint32_t  vtcm_src0_size_per_thread;
    uint32_t  vtcm_src1_size_per_thread;
    uint32_t  vtcm_dst_size_per_thread;

    uint32_t  vtcm_src0_size;
    uint32_t  vtcm_src1_size;
    uint32_t  vtcm_dst_size;

    uint32_t  src0_row_size_aligned;
    uint32_t  src1_row_size_aligned;
    uint32_t  dst_row_size_aligned;

    uint32_t  vtcm_size;

    // Fastdiv helpers
    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02;
    struct fastdiv_values div_ne012;
    struct fastdiv_values div_tpr;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#endif

static inline bool htp_op_is_unary(uint32_t opcode) {
    switch (opcode) {
        case HTP_OP_NORM:
        case HTP_OP_RMS_NORM:
        case HTP_OP_RMS_NORM_MUL:
        case HTP_OP_SCALE:
        case HTP_OP_SQR:
        case HTP_OP_SQRT:
        case HTP_OP_UNARY_NEG:
        case HTP_OP_UNARY_EXP:
        case HTP_OP_UNARY_SIGMOID:
        case HTP_OP_UNARY_SOFTPLUS:
        case HTP_OP_UNARY_TANH:
        case HTP_OP_L2_NORM:
        case HTP_OP_TRI:
            return true;
        default:
            return false;
    }
}

struct htp_unary_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_bytes;
    size_t src1_bytes;
    size_t dst_bytes;
};

static inline void htp_unary_vtcm_layout_build(
    struct htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool broadcast_weight,
    uint32_t n_threads,
    size_t vtcm_size,
    uint32_t * out_col_tile,
    uint32_t * out_vtcm_row_per_thread
) {
    const size_t src0_data_row_size = ne00 * sizeof(float);
    const size_t dst_data_row_size  = ne10 * sizeof(float);

    const size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    const size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    size_t src1_row_size_aligned = 0;
    if (op == HTP_OP_RMS_NORM_MUL) {
        const size_t src1_data_row_size = ne11 * sizeof(float);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
    }

    size_t vtcm_size_per_row = 0;
    size_t vtcm_row_per_thread = 0;

    if (op == HTP_OP_RMS_NORM_MUL) {
        if (broadcast_weight) {
            size_t available_vtcm = vtcm_size;
            size_t src1_vtcm_total = n_threads * src1_row_size_aligned;
            if (available_vtcm > src1_vtcm_total) {
                available_vtcm -= src1_vtcm_total;
            } else {
                available_vtcm = 0;
            }
            vtcm_size_per_row = 2 * (src0_row_size_aligned + dst_row_size_aligned);
            vtcm_row_per_thread = available_vtcm / (n_threads * vtcm_size_per_row);
        } else {
            vtcm_size_per_row = 2 * (src0_row_size_aligned + dst_row_size_aligned + src1_row_size_aligned);
            vtcm_row_per_thread = vtcm_size / (n_threads * vtcm_size_per_row);
        }
    } else {
        vtcm_size_per_row   = 2 * (src0_row_size_aligned + dst_row_size_aligned);
        vtcm_row_per_thread = vtcm_size / (n_threads * vtcm_size_per_row);
    }

    const bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM);
    uint32_t col_tile = 0;

    if (vtcm_row_per_thread == 0 && !is_reduction) {
        const size_t per_thread_budget = vtcm_size / n_threads;
        const size_t col_tile_bytes = hex_align_down(per_thread_budget / 4, 128);
        col_tile = (uint32_t) (col_tile_bytes / sizeof(float));

        L->src0_bytes = col_tile_bytes * 2;
        L->dst_bytes  = col_tile_bytes * 2;
        L->src1_bytes = 0;
    } else {
        L->src0_bytes = src0_row_size_aligned * vtcm_row_per_thread * 2;
        L->dst_bytes  = dst_row_size_aligned * vtcm_row_per_thread * 2;
        if (op == HTP_OP_RMS_NORM_MUL) {
            if (broadcast_weight) {
                L->src1_bytes = src1_row_size_aligned;
            } else {
                L->src1_bytes = src1_row_size_aligned * vtcm_row_per_thread * 2;
            }
        } else {
            L->src1_bytes = 0;
        }
    }

    L->off_src0 = 0;
    if (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_bytes * n_threads;
        L->off_dst  = L->off_src1 + L->src1_bytes * n_threads;
    } else {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_bytes * n_threads;
    }

    L->total_bytes = L->off_dst + L->dst_bytes * n_threads;

    *out_col_tile = col_tile;
    *out_vtcm_row_per_thread = vtcm_row_per_thread;
}

#endif /* HTP_UNARY_OPS_H */
