#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <HAP_farf.h>
#include <HAP_compute_res.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "hex-dma.h"
#include "hex-fastdiv.h"
#include "worker-pool.h"

#include "hvx-utils.h"
#include "hvx-dump.h"
#include "htp-ctx.h"
#include "htp-ops.h"

#include "hmx-ops.h"
#include "hmx-utils.h"
#include "hmx-queue.h"

#include "vtcm-utils.h"
#include "matmul-ops.h"


// MXFP4 dequantization LUT: maps 4-bit index to fp16 mantissa value
// kvalues: 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6
static const __fp16 mxfp4_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    0, 0, 0.5, 0, 1, 0, 1.5, 0, 2, 0, 3, 0, 4, 0, 6, 0, 0, 0, -0.5, 0, -1, 0, -1.5, 0, -2, 0, -3, 0, -4, 0, -6, 0,
};

static const __fp16 iq4_nl_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
    1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
};

// --- tiled format dequantizers ---

typedef struct {
    struct htp_context      * ctx;
    struct htp_thread_trace * traces;
    __fp16                  * dst;
    const uint8_t           * src;

    struct fastdiv_values     n_k_tiles_div;
    int                       n_k_tiles;
    int                       n_tot_tiles;
    int                       n_tiles_per_task;
    int                       tile_size;
    int                       aligned_tile_size;
    int                       n_tasks;
    int                       n_cols;
    int                       k_block;
    size_t                    row_stride;
    int                       weight_type;
} tiled_dequantize_state_t;

#define DEQUANTIZE_WORKER_LOOP_IMPL(SUFFIX)                                                     \
static void dequantize_tiled_worker_loop_##SUFFIX(unsigned int n, unsigned int i, void *data) { \
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;                         \
    struct htp_thread_trace * tr = state->traces ? &state->traces[i] : NULL;                    \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);                                  \
    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {      \
        int start = task_id * state->n_tiles_per_task;                                          \
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);              \
        dequantize_tiled_weight_to_fp16_task_##SUFFIX(state, start, end);                       \
    }                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);                                   \
}

// Dequantize a single tile from tiled weight data (already in VTCM) to tile-major FP16.
static void dequantize_tiled_weight_to_fp16_task_q4_0(
        const tiled_dequantize_state_t *state,
        int start_tile, int end_tile) {

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    for (int t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HMX_FP16_TILE_N_ELMS;

        HVX_Vector v_sc = hvx_vmem(tile_src + 512);
        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_Vector v_lo0 = Q6_V_vand_VV(vq0, mask_h4);
        HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_Vector v_lo1 = Q6_V_vand_VV(vq1, mask_h4);
        HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_Vector v_lo2 = Q6_V_vand_VV(vq2, mask_h4);
        HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_Vector v_lo3 = Q6_V_vand_VV(vq3, mask_h4);
        HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Offsetting (-8)
        v_lo0 = Q6_Vb_vsub_VbVb(v_lo0, i8);
        v_hi0 = Q6_Vb_vsub_VbVb(v_hi0, i8);
        v_lo1 = Q6_Vb_vsub_VbVb(v_lo1, i8);
        v_hi1 = Q6_Vb_vsub_VbVb(v_hi1, i8);
        v_lo2 = Q6_Vb_vsub_VbVb(v_lo2, i8);
        v_hi2 = Q6_Vb_vsub_VbVb(v_hi2, i8);
        v_lo3 = Q6_Vb_vsub_VbVb(v_lo3, i8);
        v_hi3 = Q6_Vb_vsub_VbVb(v_hi3, i8);

        // Shuffling
        HVX_VectorPair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_VectorPair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_VectorPair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_VectorPair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_VectorPair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_VectorPair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_VectorPair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_VectorPair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_VectorPair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_VectorPair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_VectorPair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_VectorPair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert and scale multiplication
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated));
        HVX_Vector v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated));
        HVX_Vector v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated));

        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated));
        HVX_Vector v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated));
        HVX_Vector v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated));

        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated));
        HVX_Vector v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated));
        HVX_Vector v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated));

        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated));
        HVX_Vector v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated));
        HVX_Vector v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated));

        // Parallel Stores
        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp0_2;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp0_3;

        hvx_vmem(dst_ptr +  4 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp1_2;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp1_3;

        hvx_vmem(dst_ptr +  8 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp2_2;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp2_3;

        hvx_vmem(dst_ptr + 12 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp3_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp3_2;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp3_3;
    }
}

static void dequantize_tiled_weight_to_fp16_task_q4_1(
        const tiled_dequantize_state_t *state,
        int start_tile, int end_tile) {

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    for (int t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HMX_FP16_TILE_N_ELMS;

        HVX_Vector vscale_offset = hvx_vmem(tile_src + 512);
        HVX_VectorPair dm_deal = Q6_W_vdeal_VVR(vscale_offset, vscale_offset, -2);
        HVX_Vector vd = Q6_V_lo_W(dm_deal);
        HVX_Vector vm = Q6_V_hi_W(dm_deal);

        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
        HVX_Vector v_offset_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));

        // Load all 4 groups in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_Vector v_lo0 = Q6_V_vand_VV(vq0, mask_h4);
        HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_Vector v_lo1 = Q6_V_vand_VV(vq1, mask_h4);
        HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_Vector v_lo2 = Q6_V_vand_VV(vq2, mask_h4);
        HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_Vector v_lo3 = Q6_V_vand_VV(vq3, mask_h4);
        HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_VectorPair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_VectorPair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_VectorPair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_VectorPair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Unpack to 16-bit
        HVX_VectorPair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_VectorPair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_VectorPair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_VectorPair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_VectorPair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_VectorPair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_VectorPair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_VectorPair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));

        // Convert, multiply, add offset
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_offset_duplicated));

        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_offset_duplicated));

        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_offset_duplicated));

        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_offset_duplicated));
        HVX_Vector v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_offset_duplicated));

        // Parallel Stores
        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp0_2;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp0_3;

        hvx_vmem(dst_ptr +  4 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp1_2;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp1_3;

        hvx_vmem(dst_ptr +  8 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp2_2;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp2_3;

        hvx_vmem(dst_ptr + 12 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp3_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp3_2;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp3_3;
    }
}

static void dequantize_tiled_weight_to_fp16_task_iq4_nl(
        const tiled_dequantize_state_t *state,
        int start_tile, int end_tile) {

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector vlut_cvt = hvx_vmem(iq4_nl_to_fp16_lut);

    for (int t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HMX_FP16_TILE_N_ELMS;

        HVX_Vector v_sc = hvx_vmem(tile_src + 512);
        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load all 4 groups in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_Vector v_lo0 = Q6_V_vand_VV(vq0, mask_h4);
        HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_Vector v_lo1 = Q6_V_vand_VV(vq1, mask_h4);
        HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_Vector v_lo2 = Q6_V_vand_VV(vq2, mask_h4);
        HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_Vector v_lo3 = Q6_V_vand_VV(vq3, mask_h4);
        HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_VectorPair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_VectorPair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_VectorPair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_VectorPair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_Vector v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_Vector v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_Vector v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_Vector v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_Vector v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_Vector v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_Vector v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_Vector v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_VectorPair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_VectorPair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_VectorPair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_VectorPair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_VectorPair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_VectorPair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_VectorPair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_VectorPair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_Vector v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_Vector v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_Vector v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_Vector v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_Vector v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_Vector v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_Vector v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_Vector v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

        // Parallel Stores
        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp0_2;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp0_3;

        hvx_vmem(dst_ptr +  4 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp1_2;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp1_3;

        hvx_vmem(dst_ptr +  8 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp2_2;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp2_3;

        hvx_vmem(dst_ptr + 12 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp3_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp3_2;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp3_3;
    }
}

DEQUANTIZE_WORKER_LOOP_IMPL(q4_0)

DEQUANTIZE_WORKER_LOOP_IMPL(q4_1)

DEQUANTIZE_WORKER_LOOP_IMPL(iq4_nl)

static void dequantize_tiled_weight_to_fp16_task_mxfp4(
        const tiled_dequantize_state_t *state,
        int start_tile, int end_tile) {

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector vlut_cvt = hvx_vmem(mxfp4_to_fp16_lut);

    for (int t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HMX_FP16_TILE_N_ELMS;

        HVX_Vector v = hvx_vmem(tile_src + 512);
        HVX_Vector vh = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v));
        vh = Q6_Vh_vsub_VhVh(vh, Q6_Vh_vsplat_R(112));
        vh = Q6_Vh_vmax_VhVh(vh, Q6_V_vzero());
        vh = Q6_Vh_vmin_VhVh(vh, Q6_Vh_vsplat_R(30));
        vh = Q6_Vh_vasl_VhR(vh, 10);

        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vh, vh, -2));

        // Load all 4 groups in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        // Nibble extraction
        HVX_Vector v_lo0 = Q6_V_vand_VV(vq0, mask_h4);
        HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
        HVX_Vector v_lo1 = Q6_V_vand_VV(vq1, mask_h4);
        HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
        HVX_Vector v_lo2 = Q6_V_vand_VV(vq2, mask_h4);
        HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
        HVX_Vector v_lo3 = Q6_V_vand_VV(vq3, mask_h4);
        HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);

        // Shuffling
        HVX_VectorPair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
        HVX_VectorPair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
        HVX_VectorPair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
        HVX_VectorPair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);

        // Shuffle for LUT lookup
        HVX_Vector v_q_lo0 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf0));
        HVX_Vector v_q_hi0 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf0));
        HVX_Vector v_q_lo1 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf1));
        HVX_Vector v_q_hi1 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf1));
        HVX_Vector v_q_lo2 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf2));
        HVX_Vector v_q_hi2 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf2));
        HVX_Vector v_q_lo3 = Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf3));
        HVX_Vector v_q_hi3 = Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf3));

        // LUT lookup
        HVX_VectorPair vp_lo0 = Q6_Wh_vlut16_VbVhR(v_q_lo0, vlut_cvt, 0);
        HVX_VectorPair vp_hi0 = Q6_Wh_vlut16_VbVhR(v_q_hi0, vlut_cvt, 0);
        HVX_VectorPair vp_lo1 = Q6_Wh_vlut16_VbVhR(v_q_lo1, vlut_cvt, 0);
        HVX_VectorPair vp_hi1 = Q6_Wh_vlut16_VbVhR(v_q_hi1, vlut_cvt, 0);
        HVX_VectorPair vp_lo2 = Q6_Wh_vlut16_VbVhR(v_q_lo2, vlut_cvt, 0);
        HVX_VectorPair vp_hi2 = Q6_Wh_vlut16_VbVhR(v_q_hi2, vlut_cvt, 0);
        HVX_VectorPair vp_lo3 = Q6_Wh_vlut16_VbVhR(v_q_lo3, vlut_cvt, 0);
        HVX_VectorPair vp_hi3 = Q6_Wh_vlut16_VbVhR(v_q_hi3, vlut_cvt, 0);

        // Convert and scale multiplication
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo0), v_scale_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo0), v_scale_duplicated));
        HVX_Vector v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi0), v_scale_duplicated));
        HVX_Vector v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi0), v_scale_duplicated));

        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo1), v_scale_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo1), v_scale_duplicated));
        HVX_Vector v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi1), v_scale_duplicated));
        HVX_Vector v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi1), v_scale_duplicated));

        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo2), v_scale_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo2), v_scale_duplicated));
        HVX_Vector v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi2), v_scale_duplicated));
        HVX_Vector v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi2), v_scale_duplicated));

        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo3), v_scale_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo3), v_scale_duplicated));
        HVX_Vector v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi3), v_scale_duplicated));
        HVX_Vector v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi3), v_scale_duplicated));

        // Parallel Stores
        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp0_2;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp0_3;

        hvx_vmem(dst_ptr +  4 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp1_2;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp1_3;

        hvx_vmem(dst_ptr +  8 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp2_2;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp2_3;

        hvx_vmem(dst_ptr + 12 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp3_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp3_2;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp3_3;
    }
}

DEQUANTIZE_WORKER_LOOP_IMPL(mxfp4)

static void dequantize_tiled_weight_to_fp16_task_q8_0(
        const tiled_dequantize_state_t *state,
        int start_tile, int end_tile) {

    for (int t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HMX_FP16_TILE_N_ELMS;

        HVX_Vector v_sc = hvx_vmem(tile_src + 1024);
        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        // Load groups 0-3 in parallel
        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);

        HVX_VectorPair vp_int16_0 = Q6_Wh_vunpack_Vb(vq0);
        HVX_VectorPair vp_int16_1 = Q6_Wh_vunpack_Vb(vq1);
        HVX_VectorPair vp_int16_2 = Q6_Wh_vunpack_Vb(vq2);
        HVX_VectorPair vp_int16_3 = Q6_Wh_vunpack_Vb(vq3);

        // Load groups 4-7 in parallel
        HVX_Vector vq4 = hvx_vmem(tile_src + 4 * 128);
        HVX_Vector vq5 = hvx_vmem(tile_src + 5 * 128);
        HVX_Vector vq6 = hvx_vmem(tile_src + 6 * 128);
        HVX_Vector vq7 = hvx_vmem(tile_src + 7 * 128);

        HVX_VectorPair vp_int16_4 = Q6_Wh_vunpack_Vb(vq4);
        HVX_VectorPair vp_int16_5 = Q6_Wh_vunpack_Vb(vq5);
        HVX_VectorPair vp_int16_6 = Q6_Wh_vunpack_Vb(vq6);
        HVX_VectorPair vp_int16_7 = Q6_Wh_vunpack_Vb(vq7);

        // Convert and scale multiply for groups 0-3
        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_0)), v_scale_duplicated));
        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_0)), v_scale_duplicated));
        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_1)), v_scale_duplicated));
        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_1)), v_scale_duplicated));
        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_2)), v_scale_duplicated));
        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_2)), v_scale_duplicated));
        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_3)), v_scale_duplicated));
        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_3)), v_scale_duplicated));

        // Store groups 0-3
        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
        hvx_vmem(dst_ptr +  2 * 64) = v_grp1_0;
        hvx_vmem(dst_ptr +  3 * 64) = v_grp1_1;
        hvx_vmem(dst_ptr +  4 * 64) = v_grp2_0;
        hvx_vmem(dst_ptr +  5 * 64) = v_grp2_1;
        hvx_vmem(dst_ptr +  6 * 64) = v_grp3_0;
        hvx_vmem(dst_ptr +  7 * 64) = v_grp3_1;

        // Convert and scale multiply for groups 4-7
        HVX_Vector v_grp4_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_4)), v_scale_duplicated));
        HVX_Vector v_grp4_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_4)), v_scale_duplicated));
        HVX_Vector v_grp5_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_5)), v_scale_duplicated));
        HVX_Vector v_grp5_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_5)), v_scale_duplicated));
        HVX_Vector v_grp6_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_6)), v_scale_duplicated));
        HVX_Vector v_grp6_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_6)), v_scale_duplicated));
        HVX_Vector v_grp7_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_7)), v_scale_duplicated));
        HVX_Vector v_grp7_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_7)), v_scale_duplicated));

        // Store groups 4-7
        hvx_vmem(dst_ptr +  8 * 64) = v_grp4_0;
        hvx_vmem(dst_ptr +  9 * 64) = v_grp4_1;
        hvx_vmem(dst_ptr + 10 * 64) = v_grp5_0;
        hvx_vmem(dst_ptr + 11 * 64) = v_grp5_1;
        hvx_vmem(dst_ptr + 12 * 64) = v_grp6_0;
        hvx_vmem(dst_ptr + 13 * 64) = v_grp6_1;
        hvx_vmem(dst_ptr + 14 * 64) = v_grp7_0;
        hvx_vmem(dst_ptr + 15 * 64) = v_grp7_1;
    }
}

DEQUANTIZE_WORKER_LOOP_IMPL(q8_0)

static void convert_f16_weight_to_fp16_tiles_task(
        const tiled_dequantize_state_t *state,
        int start_tile, int end_tile) {

    const int n_k_tiles = state->n_k_tiles;
    const struct fastdiv_values n_k_tiles_div = state->n_k_tiles_div;

    const HVX_Vector v_scat_base  = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step  = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

    unsigned ct = fastdiv((unsigned)start_tile, &n_k_tiles_div);
    unsigned kt = fastmodulo((unsigned)start_tile, n_k_tiles, &n_k_tiles_div);

    for (unsigned t = start_tile; t < (unsigned)end_tile; ) {
        if (kt >= (unsigned)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_base = state->dst + t * HMX_FP16_TILE_N_ELMS;
        {
            int byte_off = kt * 32 * sizeof(__fp16);

            HVX_Vector v_off = v_scat_base;
            for (int r = 0; r < HMX_FP16_TILE_N_ROWS; r += 2) {
                int row0 = ct * HMX_FP16_TILE_N_COLS + r;
                int row1 = row0 + 1;

                const uint8_t *r0 = state->src + row0 * state->row_stride;
                const uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_Vector v0 = hvx_vmemu((const __fp16 *)(r0 + byte_off));
                HVX_Vector v1 = (row1 < state->n_cols) ? hvx_vmemu((const __fp16 *)(r1 + byte_off)) : Q6_V_vzero();

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
            (void) *(volatile HVX_Vector *)(tile_base);
        }
        ++t; ++kt;
    }

    if (start_tile < end_tile) {
        (void) *(volatile HVX_Vector *)(state->dst + (end_tile - 1) * HMX_FP16_TILE_N_ELMS);
    }
}

static void convert_f16_worker_loop(unsigned int n, unsigned int i, void *data) {
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;
    struct htp_thread_trace * tr = state->traces ? &state->traces[i] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        convert_f16_weight_to_fp16_tiles_task(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

static void quantize_f32_weight_to_fp16_tiles_task(
        const tiled_dequantize_state_t *state,
        int start_tile, int end_tile) {

    const int n_k_tiles = state->n_k_tiles;
    const struct fastdiv_values n_k_tiles_div = state->n_k_tiles_div;

    const HVX_Vector v_scat_base  = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step  = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

    unsigned ct = fastdiv((unsigned)start_tile, &n_k_tiles_div);
    unsigned kt = fastmodulo((unsigned)start_tile, n_k_tiles, &n_k_tiles_div);

    for (unsigned t = start_tile; t < (unsigned)end_tile; ) {
        if (kt >= (unsigned)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_base = state->dst + t * HMX_FP16_TILE_N_ELMS;
        {
            int byte_off = kt * 32 * sizeof(float);

            HVX_Vector v_off = v_scat_base;
            for (int r = 0; r < HMX_FP16_TILE_N_ROWS; r += 2) {
                int row0 = ct * HMX_FP16_TILE_N_COLS + r;
                int row1 = row0 + 1;

                const uint8_t *r0 = state->src + row0 * state->row_stride;
                const uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_Vector v0_f32 = hvx_vmem((const float *)(r0 + byte_off));
                HVX_Vector v1_f32 = (row1 < state->n_cols) ? hvx_vmem((const float *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_Vector v_out = hvx_vec_f32_to_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v_out);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);

                HVX_Vector v_out_hi = Q6_V_vror_VR(v_out, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v_out_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
            (void) *(volatile HVX_Vector *)(tile_base);
        }
        ++t; ++kt;
    }

    if (start_tile < end_tile) {
        (void) *(volatile HVX_Vector *)(state->dst + (end_tile - 1) * HMX_FP16_TILE_N_ELMS);
    }
}

static void quantize_f32_worker_loop(unsigned int n, unsigned int i, void *data) {
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;

    struct htp_thread_trace * tr = state->traces ? &state->traces[i] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);

    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        quantize_f32_weight_to_fp16_tiles_task(state, start, end);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);
}

static void dequantize_tiled_weight_chunk_to_fp16_tiles(
        struct htp_context *ctx, __fp16 *vtcm_dst,
        const void *weight_src_ddr,
        int n_cols, int k_block,
        size_t row_stride, int weight_type,
        int n_k_tiles, struct fastdiv_values n_k_tiles_div,
        worker_callback_t dequant_worker_fn, int n_threads) {

    assert(n_cols  % HMX_FP16_TILE_N_COLS == 0);
    assert(k_block % HMX_FP16_TILE_N_COLS == 0);

    size_t n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
    size_t n_tot_tiles = n_col_tiles * n_k_tiles;

    size_t n_tiles_per_task = (n_threads == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_threads);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_per_task = n_tiles_per_task;
    state.dst              = vtcm_dst;
    state.src              = (const uint8_t *)weight_src_ddr;
    state.n_cols           = n_cols;
    state.k_block          = k_block;
    state.row_stride       = row_stride;
    state.weight_type      = weight_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    switch (weight_type) {
        case HTP_TYPE_Q4_0:   state.tile_size = 576;  state.aligned_tile_size = 640; break;
        case HTP_TYPE_Q4_1:   state.tile_size = 640;  state.aligned_tile_size = 640; break;
        case HTP_TYPE_IQ4_NL: state.tile_size = 576;  state.aligned_tile_size = 640; break;
        case HTP_TYPE_MXFP4:  state.tile_size = 544;  state.aligned_tile_size = 640; break;
        case HTP_TYPE_Q8_0:   state.tile_size = 1088; state.aligned_tile_size = 1152; break;
        default:              state.tile_size = 0;    state.aligned_tile_size = 0; break;
    }

    if (state.n_tasks == 1 || n_threads == 1) {
        dequant_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, dequant_worker_fn, &state, n_threads);
    }
}

// --- End tiled dequantizers ---

#pragma clang diagnostic ignored "-Wbackend-plugin" // spurios warning for hmx intrinsics

// requires external HMX lock
static void core_dot_chunk_fp16(__fp16 *restrict output, const __fp16 *restrict activation, const __fp16 *restrict weight, const __fp16 *restrict scales,
                                int n_row_tiles, int n_col_tiles, int n_dot_tiles) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    Q6_bias_mxmem2_A((void *)scales);
    for (int r = 0; r < n_row_tiles; ++r) {
        for (size_t c = 0; c < n_col_tiles; ++c) {
            Q6_mxclracc_hf();

            const __fp16 *row_tiles = activation + r * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
            const __fp16 *col_tiles = weight + c * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

            for (int k = 0, k_block; k < n_dot_tiles; k += k_block) {
                k_block = hex_smin(n_dot_tiles - k, 32);
                const uint32_t range = 2048u * (uint32_t)k_block - 1;
                Q6_activation_hf_mxmem_RR_deep((unsigned int)row_tiles, range);
                Q6_weight_hf_mxmem_RR((unsigned int)col_tiles, range);
                row_tiles += k_block * HMX_FP16_TILE_N_ELMS;
                col_tiles += k_block * HMX_FP16_TILE_N_ELMS;
            }

            __fp16 *out_tile = output + (r * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
            Q6_mxmem_AR_after_hf(out_tile, 0);
        }
    }
}

// C += AB
static void core_mma_chunk_fp16(__fp16 *restrict c, const __fp16 *restrict a, const __fp16 *restrict b,
                                const __fp16 *restrict col_scales, const __fp16 *restrict eye_tile,
                                int n_row_tiles, int n_col_tiles, int n_dot_tiles, bool zero_init) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    Q6_bias_mxmem2_A((void *)col_scales);

    const size_t dot_tile_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    for (size_t i = 0; i < n_row_tiles; ++i) {
        const __fp16 *row_base = a + i * dot_tile_stride;
        __fp16 *res_base = c + i * n_col_tiles * HMX_FP16_TILE_N_ELMS;
        for (size_t j = 0; j < n_col_tiles; ++j) {
            Q6_mxclracc_hf();

            const __fp16 *col_tiles = b + j * dot_tile_stride;
            const __fp16 *row_tiles = row_base;
            __fp16 *accum_tile = res_base + j * HMX_FP16_TILE_N_ELMS;
            if (!zero_init) {
                Q6_activation_hf_mxmem_RR((unsigned int)accum_tile, 2047);
                Q6_weight_hf_mxmem_RR((unsigned int)eye_tile, 2047);
            }

            for (int k = 0, k_block; k < n_dot_tiles; k += k_block) {
                k_block = hex_smin(n_dot_tiles - k, 32);
                const uint32_t range = 2048u * (uint32_t)k_block - 1;
                Q6_activation_hf_mxmem_RR_deep((unsigned int)row_tiles, range);
                Q6_weight_hf_mxmem_RR((unsigned int)col_tiles, range);
                row_tiles += k_block * HMX_FP16_TILE_N_ELMS;
                col_tiles += k_block * HMX_FP16_TILE_N_ELMS;
            }

            Q6_mxmem_AR_after_hf(accum_tile, 0);
        }
    }
}

// --- Async HMX matmul job (for pipeline overlap) ---

typedef struct {
    __fp16 *       output;
    const __fp16 * activation;
    const __fp16 * weight;
    const __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_job_t;

static void hmx_matmul_worker_fn(void * data) {
    hmx_matmul_job_t * job = (hmx_matmul_job_t *) data;
    FARF(HIGH, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
    core_dot_chunk_fp16(job->output, job->activation, job->weight, job->scales, job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
}

static inline void hmx_matmul_job_init(hmx_matmul_job_t * job,
                                       __fp16 *           output,
                                       const __fp16 *     activation,
                                       const __fp16 *     weight,
                                       const __fp16 *     scales,
                                       int                n_row_tiles,
                                       int                n_col_tiles,
                                       int                n_dot_tiles) {
    job->output      = output;
    job->activation  = activation;
    job->weight      = weight;
    job->scales      = scales;
    job->n_row_tiles = n_row_tiles;
    job->n_col_tiles = n_col_tiles;
    job->n_dot_tiles = n_dot_tiles;
}

// output : fp16 -> f32p

static void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict vtcm_src, int n_rows, int n_cols, int dst_stride, int dst_cols) {
    assert(n_cols % HMX_FP16_TILE_N_COLS == 0);
    const size_t tile_row_stride = (n_cols / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_N_ELMS;

    const HVX_Vector one = hvx_vec_splat_f16(1.0);

    for (size_t r = 0; r < n_rows; r += 2) {
        const size_t r0 = r / HMX_FP16_TILE_N_ROWS;
        const size_t r1 = (r % HMX_FP16_TILE_N_ROWS) / 2;  // index of the row pair within the tile
        const __fp16 *row_base = vtcm_src + r0 * tile_row_stride;
        float *output_row_base = dst + r * dst_stride;  // global memory row base for row r (and r+1)

        #pragma unroll(4)
        for (size_t c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
            const size_t c0 = c / HMX_FP16_TILE_N_COLS;
            const __fp16 *tile = row_base + c0 * HMX_FP16_TILE_N_ELMS;
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            volatile HVX_Vector *pv_out0 = (volatile HVX_Vector *) (output_row_base + c + 0);
            volatile HVX_Vector *pv_out1 = (volatile HVX_Vector *) (output_row_base + c + dst_stride);  // next row in global memory

            int valid_c = dst_cols - (int)c;
            if (valid_c >= 32) {
                *pv_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
                if (r + 1 < n_rows) {
                    *pv_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                }
            } else if (valid_c > 0) {
                hvx_vec_store_u(output_row_base + c, valid_c * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp)));
                if (r + 1 < n_rows) {
                    hvx_vec_store_u(output_row_base + c + dst_stride, valid_c * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp)));
                }
            }
        }
    }
}

typedef struct {
    const __fp16  *vtcm_src;
    float         *dst;
    int            n_tasks;
    int            n_tot_chunks;
    int            n_chunks_per_task;
    int            n_cols;
    int            dst_stride;  // DDR row stride
    int            dst_cols;    // Actual output columns
    struct htp_thread_trace * traces;
} output_transfer_task_state_t;

static void transfer_output_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = st->traces ? &st->traces[i] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, i);

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        float        *dst      = st->dst      + chunk_idx * st->dst_stride;
        const __fp16 *vtcm_src = st->vtcm_src + chunk_idx * st->n_cols;
        transfer_output_chunk_fp16_to_fp32(dst, vtcm_src, chunk_size, st->n_cols, st->dst_stride, st->dst_cols);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, i);
}

static void transfer_output_chunk_threaded(struct htp_context *ctx, float *dst, const __fp16 *vtcm_src,
                                              int n_rows, int n_cols, int dst_stride, int dst_cols, int n_threads) {
    assert(n_cols % HMX_FP16_TILE_N_COLS == 0);

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : HMX_FP16_TILE_N_ROWS;  // must be multiple of HMX_FP16_TILE_N_ROWS (32)

    output_transfer_task_state_t state;
    state.n_tasks           = (n_tot_chunks + n_chunks_per_task - 1) / n_chunks_per_task;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_per_task = n_chunks_per_task;
    state.dst               = dst;
    state.vtcm_src          = vtcm_src;
    state.n_cols            = n_cols;
    state.dst_stride        = dst_stride;
    state.dst_cols          = dst_cols;
    state.traces            = ctx->trace;

    if (state.n_tasks == 1 || n_threads == 1) {
        transfer_output_chunk_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_worker_fn, &state, n_threads);
    }
}

// activations : fp32 -> fp16

static void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src, int n_rows, int k_block, int k_stride, int k_valid) {
    const int n_rows_padded = hex_align_up(n_rows, HMX_FP16_TILE_N_ROWS);
    const int n_rows_tiled  = (n_rows / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;

    int r = 0;

    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
        int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

        const float *ptr_in0 = src + (r + 0) * k_stride;
        const float *ptr_in1 = src + (r + 1) * k_stride;
        for (int c = 0; c < k_block; c += 32) {
            HVX_Vector v0 = *(const HVX_Vector *)(ptr_in0 + c);
            HVX_Vector v1 = *(const HVX_Vector *)(ptr_in1 + c);

            if (c + 32 > k_valid) {
                int rem = k_valid - c;
                HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
                v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
                v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());
            }

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            // compute output position
            int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
            int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }

    for (; r < n_rows_padded; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
        int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

        const bool row0_valid = r       < n_rows;
        const bool row1_valid = (r + 1) < n_rows;

        const float *ptr_in0 = row0_valid ? (src + (r + 0) * k_stride) : NULL;
        const float *ptr_in1 = row1_valid ? (src + (r + 1) * k_stride) : NULL;
        for (int c = 0; c < k_block; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (row0_valid) v0 = *(const HVX_Vector *)(ptr_in0 + c);
            if (row1_valid) v1 = *(const HVX_Vector *)(ptr_in1 + c);

            if (c + 32 > k_valid) {
                int rem = k_valid - c;
                HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
                v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
                v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());
            }

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            // compute output position
            int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
            int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }
}

typedef struct {
    __fp16      *dst;
    const float *src;
    int          n_tasks;
    int          n_tot_chunks;
    int          n_chunks_per_task;
    int          k_block;
    int          k_stride;
    int          k_valid;
    struct htp_thread_trace * traces;
    struct htp_context * ctx;
    float              * vtcm_f32_act;
} activation_transfer_task_state_t;

static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        const float *restrict src,
        int n_rows,
        int k_block,
        int k_stride,
        int k_valid,
        float *thread_f32_act) {

    const int n_rows_padded = hex_align_up(n_rows, HMX_FP16_TILE_N_ROWS);
    const int n_steps       = n_rows_padded / 2;

    // pre-fetch step 0
    if (n_steps > 0 && n_rows > 0) {
        int nrows_to_fetch = (1 < n_rows) ? 2 : 1;
        dma_queue_push(dma_q, dma_make_ptr(thread_f32_act, src),
                       k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
    }

    for (int s = 0; s < n_steps; ++s) {
        int r = 2 * s;
        float *curr_buf = thread_f32_act + (s % 2) * 2 * k_block;

        if (r < n_rows) {
            dma_queue_pop(dma_q);
        }

        int next_s = s + 1;
        int next_r = 2 * next_s;
        if (next_r < n_rows) {
            int nrows_to_fetch = (next_r + 1 < n_rows) ? 2 : 1;
            const float *next_src = src + next_r * k_stride;
            float *next_buf = thread_f32_act + (next_s % 2) * 2 * k_block;
            dma_queue_push(dma_q, dma_make_ptr(next_buf, next_src),
                           k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
        }

        const bool row0_valid = (r < n_rows);
        const bool row1_valid = (r + 1) < n_rows;

        const float *ptr_in0 = curr_buf;
        const float *ptr_in1 = curr_buf + k_block;

        for (int c = 0; c < k_block; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (row0_valid) v0 = *(const HVX_Vector *)(ptr_in0 + c);
            if (row1_valid) v1 = *(const HVX_Vector *)(ptr_in1 + c);

            if (c + 32 > k_valid) {
                int rem = k_valid - c;
                HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
                v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
                v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());
            }

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            int r0       = r / HMX_FP16_TILE_N_ROWS;  // tile row index
            int r1       = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx
            int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
            int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }
}

static void transfer_activation_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = st->traces ? &st->traces[i] : NULL;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, i);

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        const float *src = st->src + chunk_idx * st->k_stride;

        if (st->vtcm_f32_act) {
            float *thread_f32_act = st->vtcm_f32_act + i * 4 * st->k_block;
            transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
                st->ctx->dma[i], dst, src, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_f32_act
            );
        } else {
            transfer_activation_chunk_fp32_to_fp16(dst, src, chunk_size, st->k_block, st->k_stride, st->k_valid);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i);
}

static void transfer_activation_chunk_threaded(
        struct htp_context *ctx,
        __fp16 *dst,
        const float *src,
        int n_rows,
        int k_block,
        int k_stride,
        int n_threads,
        int k_valid,
        float *vtcm_f32_act) {
    assert(k_block % HMX_FP16_TILE_N_COLS == 0 && k_stride % HMX_FP16_TILE_N_COLS == 0);
    assert(VLEN == 32 * sizeof(float));

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks           = (n_tot_chunks + n_chunks_per_task - 1) / n_chunks_per_task;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_per_task = n_chunks_per_task;
    state.dst               = dst;
    state.src               = src;
    state.k_block           = k_block;
    state.k_stride          = k_stride;
    state.k_valid           = k_valid;
    state.traces            = ctx->trace;
    state.ctx               = ctx;
    state.vtcm_f32_act      = vtcm_f32_act;

    if (state.n_tasks == 1 || n_threads == 1) {
        transfer_activation_chunk_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_worker_fn, &state, n_threads);
    }
}

int hmx_matmul_2d_f32(struct htp_context *ctx,
                                  float *restrict dst,
                                  const float *activation,
                                  const uint8_t *weight,
                                  int m, int k, int n,
                                  int act_stride,
                                  int weight_stride,
                                  int weight_type,
                                  int k_valid,
                                  int dst_stride,
                                  int dst_cols,
                                  int m_chunk,
                                  int n_chunk,
                                  int use_pipeline,
                                  int num_threads,
                                  int tile_size,
                                  int aligned_tile_size,
                                  int spad_size) {
    if (k % 32 != 0 || n % 32 != 0) { return -1; }

    if (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN) || !hex_is_aligned(weight, VLEN)) {
        return -1;
    }

    size_t row_stride = htp_get_tiled_row_stride(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    worker_callback_t dequant_worker_fn = NULL;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_0; break;
        case HTP_TYPE_IQ4_NL: dequant_worker_fn = dequantize_tiled_worker_loop_iq4_nl; break;
        case HTP_TYPE_Q4_1:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
        case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
        case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return -1;
    }

    const int n_k_tiles = k / HMX_FP16_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const bool is_quant     = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);
    const size_t vec_dot_size = k * sizeof(__fp16);
    const size_t vtcm_budget  = ctx->vtcm_size;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used      = spad_size;

    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;

    const int act_threads         = hmx_get_act_threads(num_threads, use_pipeline);
    const size_t act_f32_size     = hex_align_up(act_threads * 4 * k * sizeof(float), HMX_FP16_TILE_SIZE);

    const size_t weight_area_size = is_quant
        ? hex_align_up((n_chunk_n_cols / 32) * n_k_tiles * aligned_tile_size, HMX_FP16_TILE_SIZE)
        : hex_align_up(n_chunk_n_cols * row_stride, HMX_FP16_TILE_SIZE);
    const size_t act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HMX_FP16_TILE_SIZE);

    size_t scratch0_size, scratch1_size, scratch2_size;
    scratch0_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);  // dequant buf 0
    scratch1_size = use_pipeline ? scratch0_size : 0;                                 // dequant buf 1
    scratch2_size = use_pipeline ? output_area_size : 0;                              // output  buf 1

    uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = { NULL, NULL };
    if (weight_area_size) {
        if (use_pipeline) {
            vtcm_weight_raw[0] = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
            vtcm_weight_raw[1] = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
        } else {
            vtcm_weight_raw[0] = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
        }
    }
    __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_size);
    float   *vtcm_f32_act    = (float *) vtcm_seq_alloc(&vtcm_ptr, act_f32_size);
    __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
    void    *vtcm_scratch1   = scratch1_size ? vtcm_seq_alloc(&vtcm_ptr, scratch1_size) : NULL;
    void    *vtcm_scratch2   = scratch2_size ? vtcm_seq_alloc(&vtcm_ptr, scratch2_size) : NULL;
    __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    if (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_used, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        return -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(HIGH, "hmx-mm-2d-precomputed: standard : m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    int n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    if (use_pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_job_t job_slots[2];  // persistent double-buffered job descriptors

        for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            void *vtcm_weight_bufs[2] = { vtcm_scratch0, vtcm_scratch1 };
            void *vtcm_output_bufs[2] = { vtcm_output,   vtcm_scratch2 };

            transfer_activation_chunk_threaded(ctx, vtcm_activation, activation + mr * act_stride, n_rows, k, act_stride, act_threads, k_valid, vtcm_f32_act);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            const size_t n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
            if (is_quant) {
                dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[0], weight), aligned_tile_size, tile_size, tile_size, (n_cols_A0 / 32) * n_k_tiles);
            } else {
                dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[0], weight), row_stride, weight_stride, row_stride, n_cols_A0);
            }

            if (1 < n_chunk_cnt) {
                const size_t n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                if (is_quant) {
                    dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[1], weight + n_chunk_n_cols * weight_stride), aligned_tile_size, tile_size, tile_size, (n_cols_A1 / 32) * n_k_tiles);
                } else {
                    dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[1], weight + n_chunk_n_cols * weight_stride), row_stride, weight_stride, row_stride, n_cols_A1);
                }
            }

            // pop A0 -> dequantize A0 -> submit C0
            dma_queue_pop(ctx->dma[0]);
            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_weight_bufs[0], vtcm_weight_raw[0],
                n_cols_A0, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, num_threads);

            hmx_matmul_job_init(&job_slots[0], (__fp16 *) vtcm_output_bufs[0], (__fp16 *) vtcm_activation,
                                (__fp16 *) vtcm_weight_bufs[0], vtcm_scales,
                                hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS),
                                hmx_ceil_div(n_cols_A0, HMX_FP16_TILE_N_COLS), k / HMX_FP16_TILE_N_ROWS);
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[0]));

            // Main loop: pop/dequantize A_{i+1} -> push A_{i+2} -> submit C_{i+1} -> wait C_i and store D_i
            for (int i = 0; i < n_chunk_cnt; ++i) {
                const size_t nc    = i * n_chunk_n_cols;
                const size_t nc_p1 = nc + 1 * n_chunk_n_cols;
                const size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                const size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                const size_t n_cols_p1 = hex_smin(n - nc_p1, n_chunk_n_cols);
                const size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                // 1. pop A_{i+1} and dequantize it (if i+1 < n_chunk_cnt)
                if (i + 1 < n_chunk_cnt) {
                    dma_queue_pop(ctx->dma[0]);
                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[(i + 1) % 2], vtcm_weight_raw[(i + 1) % 2],
                        n_cols_p1, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, num_threads);
                }

                // 2. push A_{i+2} (if i+2 < n_chunk_cnt)
                if (i + 2 < n_chunk_cnt) {
                    if (is_quant) {
                        dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[(i + 2) % 2], weight + nc_p2 * weight_stride), aligned_tile_size, tile_size, tile_size, (n_cols_p2 / 32) * n_k_tiles);
                    } else {
                        dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[(i + 2) % 2], weight + nc_p2 * weight_stride), row_stride, weight_stride, row_stride, n_cols_p2);
                    }
                }

                // 3. submit C_{i+1} (if i+1 < n_chunk_cnt)
                if (i + 1 < n_chunk_cnt) {
                    hmx_matmul_job_init(&job_slots[(i + 1) % 2], (__fp16 *) vtcm_output_bufs[(i + 1) % 2],
                                        (__fp16 *) vtcm_activation, (__fp16 *) vtcm_weight_bufs[(i + 1) % 2],
                                        vtcm_scales, hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS),
                                        hmx_ceil_div(n_cols_p1, HMX_FP16_TILE_N_COLS), k / HMX_FP16_TILE_N_ROWS);
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[(i + 1) % 2]));
                }

                // 4. wait C_i and store D_i (multi-thread HVX, parallel with C_{i+1})
                hmx_queue_pop(ctx->hmx_queue);
                float *output_chunk = dst + (mr * dst_stride + nc);
                int chunk_dst_cols = dst_cols - (int)nc;
                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, vtcm_output_bufs[i % 2], n_rows, n_cols, dst_stride, chunk_dst_cols, num_threads);
                }
            }
        }
        hmx_queue_suspend(ctx->hmx_queue);
    } else {
        // --- Synchronous Un-pipelined loop (m <= 32 or fallback) ---
        HAP_compute_res_hmx_lock(ctx->vtcm_rctx);
        for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            transfer_activation_chunk_threaded(ctx, vtcm_activation, activation + mr * act_stride, n_rows, k, act_stride, act_threads, k_valid, vtcm_f32_act);

            for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
                const size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                const size_t n_row_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
                const size_t n_col_tiles = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS);

                // A: Weight DMA (Synchronous)
                if (is_quant) {
                    dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[0], weight + nc * weight_stride), aligned_tile_size, tile_size, tile_size, (n_cols / 32) * n_k_tiles);
                } else {
                    dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight_raw[0], weight + nc * weight_stride), row_stride, weight_stride, row_stride, n_cols);
                }
                dma_queue_pop(ctx->dma[0]);

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_scratch0, vtcm_weight_raw[0],
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, num_threads);

                // C: HMX Compute (Synchronous)
                core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_scratch0, vtcm_scales, n_row_tiles, n_col_tiles, k / HMX_FP16_TILE_N_ROWS);

                // D: Output Store
                float *output_chunk = dst + (mr * dst_stride + nc);
                int chunk_dst_cols = dst_cols - (int)nc;
                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, vtcm_output, n_rows, n_cols, dst_stride, chunk_dst_cols, num_threads);
                }
            }
        }
        HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
    }

    return 0;
}



//

static inline int hmx_matmul_batch_r2(const hmx_matmul_f16_f32_batched_params_t *params) {
    return params->ne02 > 0 ? params->ne12 / params->ne02 : 1;
}

static inline int hmx_matmul_batch_r3(const hmx_matmul_f16_f32_batched_params_t *params) {
    return params->ne03 > 0 ? params->ne13 / params->ne03 : 1;
}

static inline const __fp16 *hmx_matmul_weight_batch_ptr(const hmx_matmul_f16_f32_batched_params_t *params,
                                                        int dst_b2, int dst_b3) {
    const int r2 = hmx_matmul_batch_r2(params);
    const int r3 = hmx_matmul_batch_r3(params);
    return (const __fp16 *) ((const uint8_t *) params->weight +
                             (size_t) (dst_b2 / r2) * params->src0_nb2 +
                             (size_t) (dst_b3 / r3) * params->src0_nb3);
}

static inline const float *hmx_matmul_activation_batch_ptr(const hmx_matmul_f16_f32_batched_params_t *params,
                                                           int dst_b2, int dst_b3) {
    return (const float *) ((const uint8_t *) params->activation +
                            (size_t) dst_b2 * params->src1_nb2 +
                            (size_t) dst_b3 * params->src1_nb3);
}

static inline float *hmx_matmul_dst_batch_ptr(const hmx_matmul_f16_f32_batched_params_t *params,
                                              int dst_b2, int dst_b3) {
    return (float *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

static int hmx_matmul_f16_f32_batched_simple(struct htp_context *ctx,
                                                      const hmx_matmul_f16_f32_batched_params_t *params,
                                                      int m_chunk, int n_chunk, int use_pipeline, int num_threads, int spad_size) {
    int ret = 0;
    for (int b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        for (int b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            ret = hmx_matmul_2d_f32(ctx, hmx_matmul_dst_batch_ptr(params, b2, b3),
                                           hmx_matmul_activation_batch_ptr(params, b2, b3),
                                           (const uint8_t *)hmx_matmul_weight_batch_ptr(params, b2, b3),
                                           params->m, params->k, params->n,
                                           params->act_stride, params->weight_stride * (int)sizeof(__fp16),
                                           HTP_TYPE_F16, params->k, params->n, params->n,
                                           m_chunk, n_chunk, use_pipeline, num_threads,
                                           0, 0, spad_size);
        }
    }
    return ret;
}

int hmx_matmul_f16_f32_batched(struct htp_context *ctx, const hmx_matmul_f16_f32_batched_params_t *params,
                               int m_chunk, int n_chunk, int use_pipeline, int num_threads, int spad_size) {
    if (!ctx || !params || !params->dst || !params->activation || !params->weight) { return -1; }
    if (!params->m || !params->k || !params->n) { return -1; }
    if (params->act_stride < params->k || params->weight_stride < params->k || params->dst_stride < params->n) { return -1; }
    if (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { return -1; }
    if (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { return -1; }
    if (params->k % 32 != 0 || params->n % 32 != 0) { return -1; }

    if (!hex_is_aligned(params->dst, VLEN) ||
        !hex_is_aligned(params->activation, VLEN) ||
        !hex_is_aligned(params->weight, VLEN)) {
        return -1;
    }

    const int group_size = hmx_matmul_batch_r2(params);
    const size_t vtcm_budget  = ctx->vtcm_size;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid if group_size > 1 and it fits within VTCM budget.
    bool run_grouped = (group_size > 1 && (size_t)spad_size <= vtcm_budget);

    if (!run_grouped) {
        FARF(HIGH, "%s: using simple batched loop", __func__);
        return hmx_matmul_f16_f32_batched_simple(ctx, params, m_chunk, n_chunk, use_pipeline, num_threads, spad_size);
    }

    const size_t vec_dot_size = params->k * sizeof(__fp16);

    // When the activation has a large stride (e.g. permuted Q tensor with
    // act_stride >> k), HVX vector loads from strided DDR thrash L2 cache.
    // Allocate an F32 scratch buffer in VTCM and use 2D DMA to gather
    // strided rows into a contiguous block before the F32->F16 conversion.
    const bool use_dma_activation = (params->act_stride > params->k);
    const int act_threads = hmx_get_act_threads(num_threads, use_pipeline);
    const size_t f32_scratch_size = use_dma_activation
        ? hex_align_up(act_threads * 4 * (size_t) params->k * sizeof(float), HMX_FP16_TILE_SIZE) : 0;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used = spad_size;

    const size_t act_head_stride      = m_chunk_n_rows * (size_t) params->k;  // fp16 elements between heads
    const size_t weight_area_size     = hex_align_up(n_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t activation_area_size = hex_align_up(group_size * m_chunk_n_rows * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t output_area_size     = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HMX_FP16_TILE_SIZE);
    const size_t scratch_area_size    = hex_align_up(n_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);

    uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
    __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
    __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
    void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
    __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
    float   *vtcm_f32_act    = use_dma_activation ? (float *) vtcm_seq_alloc(&vtcm_ptr, f32_scratch_size) : NULL;

    if ((size_t) (vtcm_ptr - (uint8_t *) ctx->vtcm_base) > vtcm_budget) {
        FARF(HIGH, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        return hmx_matmul_f16_f32_batched_simple(ctx, params, m_chunk, n_chunk, use_pipeline, num_threads, spad_size);
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(HIGH, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, group_size, params->ne13,
            m_chunk_n_rows, n_chunk_n_cols,
            (size_t) (vtcm_ptr - (uint8_t *) ctx->vtcm_base), vtcm_budget);

    const size_t fp16_row_bytes   = (size_t) params->k * sizeof(__fp16);
    const size_t weight_row_bytes = (size_t) params->weight_stride * sizeof(__fp16);

    HAP_compute_res_hmx_lock(ctx->vtcm_rctx);

    for (int b3 = 0; b3 < params->ne13; ++b3) {
        for (int b2_base = 0; b2_base < params->ne12; b2_base += group_size) {
            const __fp16 *weight_group = hmx_matmul_weight_batch_ptr(params, b2_base, b3);

            for (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_rows) {
                const size_t n_rows = hex_smin((size_t) params->m - mr, m_chunk_n_rows);
                const size_t n_row_tiles = hmx_ceil_div((int) n_rows, HMX_FP16_TILE_N_ROWS);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                for (int g = 0; g < group_size; ++g) {
                    const float *activation_chunk = hmx_matmul_activation_batch_ptr(params, b2_base + g, b3) + mr * params->act_stride;
                    __fp16 *vtcm_act_g = vtcm_activation + (size_t) g * act_head_stride;
                    if (use_dma_activation) {
                        transfer_activation_chunk_threaded(ctx, vtcm_act_g,
                                                               activation_chunk, (int) n_rows,
                                                               params->k, params->act_stride, act_threads, params->k, vtcm_f32_act);
                    } else {
                        transfer_activation_chunk_threaded(ctx, vtcm_act_g,
                                                               activation_chunk, (int) n_rows,
                                                               params->k, params->act_stride, act_threads, params->k, NULL);
                    }
                }

                void *buf_curr = vtcm_scratch0;
                void *buf_next = vtcm_scratch1;

                {
                    const size_t n_cols_first = hex_smin((size_t) params->n, n_chunk_n_cols);
                    dma_queue_push(ctx->dma[0], dma_make_ptr(buf_curr, weight_group),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
                }

                for (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_cols) {
                    const size_t n_cols = hex_smin((size_t) params->n - nc, n_chunk_n_cols);
                    const size_t n_col_tiles = hmx_ceil_div((int) n_cols, HMX_FP16_TILE_N_COLS);

                    {
                        dma_queue_pop(ctx->dma[0]);

                        const size_t nc_next = nc + n_chunk_n_cols;
                        if (nc_next < (size_t) params->n) {
                            const size_t n_cols_next = hex_smin((size_t) params->n - nc_next, n_chunk_n_cols);
                            const __fp16 *next_weight_chunk = weight_group + nc_next * params->weight_stride;

                            dma_queue_push(ctx->dma[0], dma_make_ptr(buf_next, next_weight_chunk),
                                              fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
                        }

                        hmx_interleave_rows_to_tiles(vtcm_weight, (const __fp16 *) buf_curr, n_cols, params->k, params->k, 0, n_cols);
                        hex_swap_ptr(&buf_curr, &buf_next);
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    for (int g = 0; g < group_size; ++g) {
                        struct htp_thread_trace * tr = &ctx->trace[HTP_MAX_NTHREADS];
                        htp_trace_event_start(tr, HTP_TRACE_EVT_HMX_COMP, g);
                        {
                            const __fp16 * vtcm_act_g = vtcm_activation + (size_t) g * act_head_stride;
                            core_dot_chunk_fp16(vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles,
                                                params->k / 32);
                        }
                        htp_trace_event_stop(tr, HTP_TRACE_EVT_HMX_COMP, g);

                        {
                            float *output = hmx_matmul_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
                            int chunk_dst_cols = params->n - (int)nc;
                            if (chunk_dst_cols > 0) {
                                transfer_output_chunk_threaded(ctx, output, vtcm_output, (int) n_rows, (int) n_cols, params->dst_stride, chunk_dst_cols, ctx->n_threads);
                            }
                        }
                    }
                }
            }
        }
    }

    HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);

    return 0;
}



struct mmid_row_mapping {
    uint32_t i1;
    uint32_t i2;
};

typedef struct {
    __fp16                         *dst;
    const float                    *src;
    int                             n_tasks;
    int                             n_tot_chunks;
    int                             n_chunks_per_task;
    int                             k_block;
    const struct mmid_row_mapping  *matrix_rows;
    int                             cur_a;
    int                             mapping_stride;
    int                             ne11;
    struct fastdiv_values           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    int                             start_row;
    int                             cne1;
    int                             k_valid;
    struct htp_thread_trace        *traces;
} activation_transfer_gathered_task_state_t;

typedef struct {
    const __fp16                   *vtcm_src;
    float                          *dst;
    int                             n_tasks;
    int                             n_tot_chunks;
    int                             n_chunks_per_task;
    int                             n_cols;
    const struct mmid_row_mapping  *matrix_rows;
    int                             cur_a;
    int                             mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    int                             start_row;
    int                             cne1;
    struct htp_thread_trace        *traces;
} output_transfer_scattered_task_state_t;

static void transfer_activation_chunk_fp32_to_fp16_gathered(
            __fp16 *restrict vtcm_dst,
            const float *restrict src,
            int start_row,
            int n_rows,
            int k_block,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            int ne11,
            const struct fastdiv_values * ne11_div,
            size_t nb11,
            size_t nb12,
            int cne1,
            int k_valid) {
    const int n_rows_padded = hex_align_up(n_rows, HMX_FP16_TILE_N_ROWS);
    const int n_rows_tiled  = (n_rows / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;

    int r = 0;

    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
        int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

        int r_idx0 = start_row + r + 0;
        int r_idx1 = start_row + r + 1;

        struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];

        int i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
        int i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);

        const float *row0_ptr = (const float *) ((const uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        const float *row1_ptr = (const float *) ((const uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);

        for (int c = 0; c < k_block; c += 32) {
            HVX_Vector v0 = *(const HVX_Vector *)(row0_ptr + c);
            HVX_Vector v1 = *(const HVX_Vector *)(row1_ptr + c);

            if (c + 32 > k_valid) {
                int rem = k_valid - c;
                HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
                v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
                v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());
            }

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
            int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }

    for (; r < n_rows_padded; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
        int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

        const bool row0_valid = (start_row + r + 0) < cne1;
        const bool row1_valid = (start_row + r + 1) < cne1;

        const float *row0_ptr = NULL;
        const float *row1_ptr = NULL;

        if (row0_valid) {
            struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + (start_row + r + 0)];
            int i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
            row0_ptr = (const float *) ((const uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        }
        if (row1_valid) {
            struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + (start_row + r + 1)];
            int i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);
            row1_ptr = (const float *) ((const uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);
        }

        for (int c = 0; c < k_block; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (row0_valid) v0 = *(const HVX_Vector *)(row0_ptr + c);
            if (row1_valid) v1 = *(const HVX_Vector *)(row1_ptr + c);

            if (c + 32 > k_valid) {
                int rem = k_valid - c;
                HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
                v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
                v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());
            }

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
            int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }
}

static void transfer_activation_chunk_gathered_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_gathered_task_state_t *st = data;
    struct htp_thread_trace * tr = st->traces ? &st->traces[i] : NULL;
    int chunk_idx = i;
    int chunk_size = st->n_chunks_per_task;
    int start_row = st->start_row + chunk_idx * chunk_size;
    int n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        __fp16 *dst = st->dst + (size_t)(start_row - st->start_row) * st->k_block;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered(
            dst, st->src, start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

static void transfer_activation_chunk_gathered_threaded(
            struct htp_context *ctx,
            __fp16 *dst,
            const float *src,
            int start_row,
            int n_rows,
            int k_block,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            int ne11,
            size_t nb11,
            size_t nb12,
            int cne1,
            int n_threads,
            int k_valid) {
    if (n_rows <= 0) return;
    int chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, HMX_FP16_TILE_N_ROWS);

    int actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_threads,
        .n_tot_chunks      = n_rows,
        .n_chunks_per_task = chunks_per_thread,
        .k_block           = k_block,
        .matrix_rows       = matrix_rows,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = init_fastdiv_values(ne11),
        .nb11              = nb11,
        .nb12              = nb12,
        .start_row         = start_row,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    if (actual_threads <= 1) {
        transfer_activation_chunk_gathered_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_gathered_worker_fn, &state, actual_threads);
    }
}

static void transfer_output_chunk_fp16_to_fp32_scattered(
            float *restrict dst,
            const __fp16 *restrict vtcm_src,
            int start_row,
            int n_rows,
            int n_cols,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            int cne1) {
    assert(n_cols % HMX_FP16_TILE_N_COLS == 0);
    const size_t tile_row_stride = (n_cols / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_N_ELMS;

    const HVX_Vector one = hvx_vec_splat_f16(1.0);

    for (size_t r = 0; r < n_rows; r += 2) {
        const size_t r0 = r / HMX_FP16_TILE_N_ROWS;
        const size_t r1 = (r % HMX_FP16_TILE_N_ROWS) / 2;  // index of the row pair within the tile
        const __fp16 *row_base = vtcm_src + r0 * tile_row_stride;

        int r_idx0 = start_row + (int)r + 0;
        int r_idx1 = start_row + (int)r + 1;

        if (r_idx0 >= cne1) break;

        struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        float *output_row0 = (float *) ((uint8_t *) dst + mapping0.i1 * dst_nb1 + mapping0.i2 * dst_nb2);

        float *output_row1 = NULL;
        if (r_idx1 < cne1) {
            struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];
            output_row1 = (float *) ((uint8_t *) dst + mapping1.i1 * dst_nb1 + mapping1.i2 * dst_nb2);
        }

        #pragma unroll(4)
        for (size_t c = 0; c < (size_t)n_cols; c += HMX_FP16_TILE_N_COLS) {
            const size_t c0 = c / HMX_FP16_TILE_N_COLS;
            const __fp16 *tile = row_base + c0 * HMX_FP16_TILE_N_ELMS;
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            volatile HVX_Vector *pv_out0 = (volatile HVX_Vector *) (output_row0 + c);
            volatile HVX_Vector *pv_out1 = output_row1 ? (volatile HVX_Vector *) (output_row1 + c) : NULL;

            *pv_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            if (pv_out1) {
                *pv_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
            }
        }
    }
}

static void transfer_output_chunk_scattered_worker_fn(unsigned int n, unsigned int i, void *data) {
    output_transfer_scattered_task_state_t *st = data;
    struct htp_thread_trace * tr = st->traces ? &st->traces[i] : NULL;
    int chunk_idx = i;
    int chunk_size = st->n_chunks_per_task;
    int start_row = st->start_row + chunk_idx * chunk_size;
    int n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        const __fp16 *src = st->vtcm_src + (size_t)(start_row - st->start_row) * st->n_cols;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_to_fp32_scattered(
            st->dst, src, start_row, n_rows, st->n_cols,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

static void transfer_output_chunk_scattered_threaded(
            struct htp_context *ctx,
            float *dst,
            const __fp16 *vtcm_src,
            int start_row,
            int n_rows,
            int n_cols,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            int cne1,
            int n_threads) {
    if (n_rows <= 0) return;
    int chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, HMX_FP16_TILE_N_ROWS);

    int actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_threads,
        .n_tot_chunks      = n_rows,
        .n_chunks_per_task = chunks_per_thread,
        .n_cols            = n_cols,
        .matrix_rows       = matrix_rows,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .start_row         = start_row,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    if (actual_threads <= 1) {
        transfer_output_chunk_scattered_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_scattered_worker_fn, &state, actual_threads);
    }
}

int hmx_matmul_id_2d_f32(struct htp_context *ctx,
                                         float *restrict dst,
                                         const float *activation,
                                         const uint8_t *weight,
                                         int m, int k, int n,
                                         int k_valid,
                                         int ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         int weight_stride,
                                         int weight_type,
                                         const struct mmid_row_mapping *matrix_rows,
                                         int cur_a,
                                         int mapping_stride) {
    const int cne1 = m;
    const int m_padded = hex_align_up(m, 32);

    if (k % 32 != 0 || n % 32 != 0) { return -1; }

    if (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN) || !hex_is_aligned(weight, VLEN)) {
        return -1;
    }

    size_t row_stride = htp_get_tiled_row_stride(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    worker_callback_t dequant_worker_fn = NULL;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_0; break;
        case HTP_TYPE_IQ4_NL: dequant_worker_fn = dequantize_tiled_worker_loop_iq4_nl; break;
        case HTP_TYPE_Q4_1:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
        case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
        case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return -1;
    }

    const int n_k_tiles = k / HMX_FP16_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const int num_threads = ctx->n_threads;
    const bool is_quant   = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);

    const size_t vec_dot_size = k * sizeof(__fp16);
    const size_t vtcm_budget  = ctx->vtcm_size;
    size_t vtcm_used = 0;

    int tile_size = 0;
    int aligned_tile_size = 0;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   tile_size = 576;  aligned_tile_size = 640; break;
        case HTP_TYPE_Q4_1:   tile_size = 640;  aligned_tile_size = 640; break;
        case HTP_TYPE_IQ4_NL: tile_size = 576;  aligned_tile_size = 640; break;
        case HTP_TYPE_MXFP4:  tile_size = 544;  aligned_tile_size = 640; break;
        case HTP_TYPE_Q8_0:   tile_size = 1088; aligned_tile_size = 1152; break;
        default:              tile_size = 0;    aligned_tile_size = 0; break;
    }

    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;

    const size_t weight_row_stride = is_quant ? qweight_row_stride : row_stride;
    const size_t size_per_n = weight_row_stride + vec_dot_size;
    const size_t size_per_mn = sizeof(__fp16);

    size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    if (hmx_compute_chunks(vtcm_budget, /*overhead=*/256, size_per_n, /*per_m=*/vec_dot_size, size_per_mn,
                           m_padded, n,
                           /*m_block_cost=*/(size_t) n * 3,
                           /*n_block_cost=*/(size_t) m_padded * 2, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_padded, k, n, vtcm_budget);
        return -1;
    }

    const size_t weight_area_size = hex_align_up(n_chunk_n_cols * weight_row_stride, HMX_FP16_TILE_SIZE);
    const size_t act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HMX_FP16_TILE_SIZE);

    size_t scratch0_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);

    uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = weight_area_size ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size) : NULL;
    __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_size);
    __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
    __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    if (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_used, vtcm_budget);
        return -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    HAP_compute_res_hmx_lock(ctx->vtcm_rctx);

    for (size_t mr = 0; mr < (size_t) m_padded; mr += m_chunk_n_rows) {
        const size_t n_rows = hex_smin(m_padded - mr, m_chunk_n_rows);
        const size_t n_row_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_activation, activation, (int) mr, (int) n_rows, k,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, num_threads, k_valid);

        for (size_t nc = 0; nc < (size_t) n; nc += n_chunk_n_cols) {
            const size_t n_cols = hex_smin((size_t) n - nc, n_chunk_n_cols);
            const size_t n_col_tiles = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS);

            if (is_quant) {
                dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight, weight + nc * weight_stride), aligned_tile_size, tile_size, tile_size, (n_cols / 32) * n_k_tiles);
            } else {
                dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_weight, weight + nc * weight_stride), row_stride, weight_stride, row_stride, n_cols);
            }
            dma_queue_pop(ctx->dma[0]);

            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_scratch0, vtcm_weight,
                n_cols, k, row_stride, weight_type, 
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, num_threads
            );

            core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_scratch0, vtcm_scales, n_row_tiles, n_col_tiles, k / HMX_FP16_TILE_N_ROWS);

            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (int) mr, (int) n_rows, (int) n_cols,
                matrix_rows, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, num_threads);
        }
    }

    HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
    return 0;
}
