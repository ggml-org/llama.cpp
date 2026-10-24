#ifndef HTP_UNARY_OPS_H
#define HTP_UNARY_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-ops.h"
#include "htp-vtcm.h"

// Op-specific struct for precomputed unary params
struct htp_unary_kernel_params {
    uint32_t  n_threads;
    uint32_t  col_tile;
    uint32_t  block;
    uint32_t  broadcast_weight;

    uint32_t  src0_contig;
    uint32_t  src1_contig;

    uint32_t  vtcm_size;
    uint32_t  tiles_per_row;

    // Fastdiv helpers
    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02;
    struct fastdiv_values div_ne012;
    struct fastdiv_values div_tpr;

    // collapsed shape of elementwise ops
    uint32_t  ne[HTP_OP_MAX_DIMS];
    uint32_t  src0_nb[HTP_OP_MAX_DIMS];
    uint32_t  dst_nb[HTP_OP_MAX_DIMS];
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#endif

static inline bool htp_op_is_unary(uint32_t opcode) {
    switch (opcode) {
        case HTP_OP_CLAMP:
        case HTP_OP_LEAKY_RELU:
        case HTP_OP_NORM:
        case HTP_OP_RMS_NORM:
        case HTP_OP_RMS_NORM_MUL:
        case HTP_OP_SCALE:
        case HTP_OP_SQR:
        case HTP_OP_SQRT:
        case HTP_OP_UNARY_NEG:
        case HTP_OP_UNARY_EXP:
        case HTP_OP_UNARY_SIGMOID:
        case HTP_OP_UNARY_SILU:
        case HTP_OP_UNARY_GELU:
        case HTP_OP_UNARY_GELU_QUICK:
        case HTP_OP_UNARY_GELU_ERF:
        case HTP_OP_UNARY_SOFTPLUS:
        case HTP_OP_UNARY_TANH:
        case HTP_OP_UNARY_ABS:
        case HTP_OP_UNARY_LOG:
        case HTP_OP_UNARY_RELU:
        case HTP_OP_UNARY_STEP:
        case HTP_OP_UNARY_CEIL:
        case HTP_OP_UNARY_SGN:
        case HTP_OP_L2_NORM:
        case HTP_OP_TRI:
        case HTP_OP_HADAMARD:
            return true;
        default:
            return false;
    }
}

static inline bool htp_unary_is_elementwise(uint32_t opcode) {
    switch (opcode) {
        case HTP_OP_NORM:
        case HTP_OP_RMS_NORM:
        case HTP_OP_RMS_NORM_MUL:
        case HTP_OP_L2_NORM:
        case HTP_OP_HADAMARD:
        case HTP_OP_TRI:
            return false;
        default:
            return true;
    }
}

struct htp_unary_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_row_size_aligned;
    size_t src1_row_size_aligned;
    size_t dst_row_size_aligned;

    size_t src0_bytes;
    size_t src1_bytes;
    size_t dst_bytes;
};

static inline void htp_unary_vtcm_layout_build(
    struct htp_unary_vtcm_layout * L,
    uint32_t op,
    size_t src0_data_row_size,
    size_t dst_data_row_size,
    size_t src1_data_row_size,
    size_t elem_size,
    uint32_t col_tile,
    uint32_t block,
    bool broadcast_weight,
    uint32_t n_threads
) {
    L->src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    L->dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);
    L->src1_row_size_aligned = (op == HTP_OP_RMS_NORM_MUL) ? hex_round_up(src1_data_row_size, 128) : 0;

    if (col_tile > 0) {
        const size_t col_tile_bytes = col_tile * elem_size;
        L->src0_bytes = col_tile_bytes * 2;
        L->dst_bytes  = col_tile_bytes * 2;
        L->src1_bytes = 0;
    } else {
        L->src0_bytes = L->src0_row_size_aligned * block * 2;
        L->dst_bytes  = L->dst_row_size_aligned  * block * 2;
        if (op == HTP_OP_RMS_NORM_MUL) {
            L->src1_bytes = broadcast_weight ? L->src1_row_size_aligned : (L->src1_row_size_aligned * block * 2);
        } else {
            L->src1_bytes = 0;
        }
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes * n_threads);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_src1, L->src1_bytes * n_threads, L->src1_bytes > 0);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes * n_threads);
    L->total_bytes = off;
}

static inline void htp_unary_solve_layout(
    struct htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    uint32_t max_block,
    bool broadcast_weight,
    uint32_t n_threads,
    size_t vtcm_size,
    size_t elem_size,
    uint32_t * out_col_tile,
    uint32_t * out_vtcm_row_per_thread
) {
    const size_t dst_elem_size      = (op == HTP_OP_HADAMARD) ? sizeof(float) : elem_size;
    const size_t src0_data_row_size = ne00 * elem_size;
    const size_t dst_data_row_size  = ne10 * dst_elem_size;
    const size_t src1_data_row_size = (op == HTP_OP_RMS_NORM_MUL) ? (ne11 * sizeof(float)) : 0;

    const size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    const size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);
    const size_t src1_row_size_aligned = (op == HTP_OP_RMS_NORM_MUL) ? hex_round_up(src1_data_row_size, 128) : 0;

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

    if (max_block > 0 && vtcm_row_per_thread > max_block) {
        vtcm_row_per_thread = max_block;
    }

    const bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM ||
                               op == HTP_OP_HADAMARD);
    // The tiled fallback path below only has F32 task functions (unary_task_f32_tiled_*);
    // F16 has no tiled kernels, so it must stay on the row-block path like reduction ops.
    // NOTE: if F16 ends up with vtcm_row_per_thread == 0 here (row too large for the VTCM
    // budget), execute_op_unary() will see BLOCK == 0 and skip computation for that op
    // (logged via FARF(ERROR, ...)) since there is no F16 tiled fallback. This is a known
    // limitation; supporting it would require adding F16 tiled kernels.
    const bool is_f16 = (elem_size == sizeof(_Float16));
    uint32_t col_tile = 0;

    if (vtcm_row_per_thread == 0 && !is_reduction && !is_f16) {
        const size_t per_thread_budget = vtcm_size / n_threads;
        const size_t col_tile_bytes = hex_align_down(per_thread_budget / 4, 128);
        col_tile = (uint32_t) (col_tile_bytes / elem_size);
    }

    const uint32_t block = col_tile ? 0 : (uint32_t) vtcm_row_per_thread;
    htp_unary_vtcm_layout_build(L, op, src0_data_row_size, dst_data_row_size, src1_data_row_size,
                                elem_size, col_tile, block, broadcast_weight, n_threads);

    *out_col_tile = col_tile;
    *out_vtcm_row_per_thread = (uint32_t) vtcm_row_per_thread;
}

#endif /* HTP_UNARY_OPS_H */
