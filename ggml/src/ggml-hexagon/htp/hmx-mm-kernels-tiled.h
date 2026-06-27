#include "hmx-utils.h"
#include "hmx-queue.h"

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
    uint32_t                  n_k_tiles;
    uint32_t                  n_tot_tiles;
    uint32_t                  n_tiles_per_task;
    uint32_t                  tile_size;
    uint32_t                  aligned_tile_size;
    uint32_t                  n_tasks;
    uint32_t                  n_cols;
    uint32_t                  k_block;
    size_t                    row_stride;
    uint32_t                  weight_type;
} tiled_dequantize_state_t;

// Dequantize a single tile from tiled weight data (already in VTCM) to tile-major FP16.
static void dequantize_tiled_weight_to_fp16_task_q4_0(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

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
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

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
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector vlut_cvt = hvx_vmem(iq4_nl_to_fp16_lut);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

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

static void dequantize_tiled_weight_to_fp16_task_mxfp4(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector vlut_cvt = hvx_vmem(mxfp4_to_fp16_lut);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

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

static void dequantize_tiled_weight_to_fp16_task_q8_0(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

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

static void convert_f16_weight_to_fp16_tiles_task(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const uint32_t n_k_tiles = state->n_k_tiles;
    const struct fastdiv_values n_k_tiles_div = state->n_k_tiles_div;

    const HVX_Vector v_scat_base  = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step  = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

    unsigned ct = fastdiv((unsigned)start_tile, &n_k_tiles_div);
    unsigned kt = fastmodulo((unsigned)start_tile, n_k_tiles, &n_k_tiles_div);

    for (unsigned t = start_tile; t < (unsigned)end_tile; ) {
        if (kt >= (unsigned)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_base = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * sizeof(__fp16);

            HVX_Vector v_off = v_scat_base;
            for (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ROWS; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_COLS + r;
                uint32_t row1 = row0 + 1;

                const uint8_t *r0 = state->src + row0 * state->row_stride;
                const uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_Vector v0 = hvx_vmemu((const __fp16 *)(r0 + byte_off));
                HVX_Vector v1 = (row1 < state->n_cols) ? hvx_vmemu((const __fp16 *)(r1 + byte_off)) : Q6_V_vzero();

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
            (void) *(volatile HVX_Vector *)(tile_base);
        }
        ++t; ++kt;
    }

    if (start_tile < end_tile) {
        (void) *(volatile HVX_Vector *)(state->dst + (end_tile - 1) * HTP_MM_HMX_TILE_N_ELMS);
    }
}

static void quantize_f32_weight_to_fp16_tiles_task(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const uint32_t n_k_tiles = state->n_k_tiles;
    const struct fastdiv_values n_k_tiles_div = state->n_k_tiles_div;

    const HVX_Vector v_scat_base  = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step  = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

    unsigned ct = fastdiv((unsigned)start_tile, &n_k_tiles_div);
    unsigned kt = fastmodulo((unsigned)start_tile, n_k_tiles, &n_k_tiles_div);

    for (unsigned t = start_tile; t < (unsigned)end_tile; ) {
        if (kt >= (unsigned)n_k_tiles) { kt = 0; ct++; }

        __fp16 *tile_base = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
        {
            uint32_t byte_off = kt * 32 * sizeof(float);

            HVX_Vector v_off = v_scat_base;
            for (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ROWS; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_COLS + r;
                uint32_t row1 = row0 + 1;

                const uint8_t *r0 = state->src + row0 * state->row_stride;
                const uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_Vector v0_f32 = hvx_vmem((const float *)(r0 + byte_off));
                HVX_Vector v1_f32 = (row1 < state->n_cols) ? hvx_vmem((const float *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_Vector v_out = hvx_vec_f32_to_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v_out);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);

                HVX_Vector v_out_hi = Q6_V_vror_VR(v_out, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v_out_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
            (void) *(volatile HVX_Vector *)(tile_base);
        }
        ++t; ++kt;
    }

    if (start_tile < end_tile) {
        (void) *(volatile HVX_Vector *)(state->dst + (end_tile - 1) * HTP_MM_HMX_TILE_N_ELMS);
    }
}

// --- End tiled dequantizers ---

// requires external HMX lock
static void core_dot_chunk_fp16(__fp16 *restrict output, const __fp16 *restrict activation, const __fp16 *restrict weight, const __fp16 *restrict scales,
                                uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    Q6_bias_mxmem2_A((void *)scales);
    for (uint32_t r = 0; r < n_row_tiles; ++r) {
        for (size_t c = 0; c < n_col_tiles; ++c) {
            Q6_mxclracc_hf();

            const __fp16 *row_tiles = activation + r * n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
            const __fp16 *col_tiles = weight + c * n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

            for (uint32_t k = 0, k_block; k < n_dot_tiles; k += k_block) {
                k_block = hex_smin(n_dot_tiles - k, 32);
                const uint32_t range = 2048u * (uint32_t)k_block - 1;
                Q6_activation_hf_mxmem_RR_deep((unsigned int)row_tiles, range);
                Q6_weight_hf_mxmem_RR((unsigned int)col_tiles, range);
                row_tiles += k_block * HTP_MM_HMX_TILE_N_ELMS;
                col_tiles += k_block * HTP_MM_HMX_TILE_N_ELMS;
            }

            __fp16 *out_tile = output + (r * n_col_tiles + c) * HTP_MM_HMX_TILE_N_ELMS;
            Q6_mxmem_AR_after_hf(out_tile, 0);
        }
    }
}

// C += AB
static void core_mma_chunk_fp16(__fp16 *restrict c, const __fp16 *restrict a, const __fp16 *restrict b,
                                const __fp16 *restrict col_scales, const __fp16 *restrict eye_tile,
                                uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles, bool zero_init) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    Q6_bias_mxmem2_A((void *)col_scales);

    const size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    for (size_t i = 0; i < n_row_tiles; ++i) {
        const __fp16 *row_base = a + i * dot_tile_stride;
        __fp16 *res_base = c + i * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
        for (size_t j = 0; j < n_col_tiles; ++j) {
            Q6_mxclracc_hf();

            const __fp16 *col_tiles = b + j * dot_tile_stride;
            const __fp16 *row_tiles = row_base;
            __fp16 *accum_tile = res_base + j * HTP_MM_HMX_TILE_N_ELMS;
            if (!zero_init) {
                Q6_activation_hf_mxmem_RR((unsigned int)accum_tile, 2047);
                Q6_weight_hf_mxmem_RR((unsigned int)eye_tile, 2047);
            }

            for (uint32_t k = 0, k_block; k < n_dot_tiles; k += k_block) {
                k_block = hex_smin(n_dot_tiles - k, 32);
                const uint32_t range = 2048u * k_block - 1;
                Q6_activation_hf_mxmem_RR_deep((unsigned int)row_tiles, range);
                Q6_weight_hf_mxmem_RR((unsigned int)col_tiles, range);
                row_tiles += k_block * HTP_MM_HMX_TILE_N_ELMS;
                col_tiles += k_block * HTP_MM_HMX_TILE_N_ELMS;
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
                                       uint32_t           n_row_tiles,
                                       uint32_t           n_col_tiles,
                                       uint32_t           n_dot_tiles) {
    job->output      = output;
    job->activation  = activation;
    job->weight      = weight;
    job->scales      = scales;
    job->n_row_tiles = n_row_tiles;
    job->n_col_tiles = n_col_tiles;
    job->n_dot_tiles = n_dot_tiles;
}

// output : fp16 -> f32p

static void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict vtcm_src, uint32_t start_row, uint32_t n_rows, uint32_t n_cols, uint32_t dst_stride, uint32_t dst_cols) {
    assert(n_cols % HTP_MM_HMX_TILE_N_COLS == 0);
    const size_t tile_row_stride = (n_cols / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;

    const HVX_Vector one = hvx_vec_splat_f16(1.0);

    const size_t limit_c         = hex_smin(n_cols, dst_cols);
    const size_t limit_c_aligned = (limit_c & ~31);

    for (size_t r = 0; r < n_rows; r += 2) {
        const size_t r_idx0 = start_row + r + 0;
        const size_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ROWS;
        const size_t r1 = (r_idx0 % HTP_MM_HMX_TILE_N_ROWS) / 2;  // index of the row pair within the tile
        const __fp16 *row_base = vtcm_src + r0 * tile_row_stride;
        float *output_row_base = dst + r * dst_stride;  // global memory row base for row r (and r+1)

        #pragma unroll(4)
        for (size_t c = 0; c < limit_c_aligned; c += HTP_MM_HMX_TILE_N_COLS) {
            const size_t c0    = c / HTP_MM_HMX_TILE_N_COLS;
            const __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_Vector *pv_out0 = (HVX_Vector *) (output_row_base + c + 0);
            HVX_Vector *pv_out1 = (HVX_Vector *) (output_row_base + c + dst_stride);

            *pv_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            if (r + 1 < n_rows) {
                *pv_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
            }
        }

        if (limit_c_aligned < limit_c) {
            size_t c = limit_c_aligned;
            size_t valid_c = limit_c - c;
            const size_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            const __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            hvx_vec_store_u(output_row_base + c, valid_c * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp)));
            if (r + 1 < n_rows) {
                hvx_vec_store_u(output_row_base + c + dst_stride, valid_c * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp)));
            }
        }
    }
}

typedef struct {
    const __fp16  *vtcm_src;
    float         *dst;
    uint32_t       n_tasks;
    uint32_t       n_tot_chunks;
    uint32_t       n_chunks_per_task;
    uint32_t       n_cols;
    uint32_t       dst_stride;  // DDR row stride
    uint32_t       dst_cols;    // Actual output columns
    struct htp_thread_trace * traces;
} output_transfer_task_state_t;

// activations : fp32 -> fp16

static void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src, uint32_t n_rows, uint32_t k_block, uint32_t k_stride, uint32_t k_valid) {
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
    const uint32_t n_rows_tiled  = (n_rows / HTP_MM_HMX_TILE_N_ROWS) * HTP_MM_HMX_TILE_N_ROWS;

    uint32_t r = 0;

    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        const float *ptr_in0 = src + (r + 0) * k_stride;
        const float *ptr_in1 = src + (r + 1) * k_stride;

        uint32_t c = 0;
        for (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = *(const HVX_Vector *)(ptr_in0 + c);
            HVX_Vector v1 = *(const HVX_Vector *)(ptr_in1 + c);
            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        if (c < k_block) {
            HVX_Vector v0 = *(const HVX_Vector *)(ptr_in0 + c);
            HVX_Vector v1 = *(const HVX_Vector *)(ptr_in1 + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }

    for (; r < n_rows_padded; r += 2) {
        uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        const bool row0_valid = r       < n_rows;
        const bool row1_valid = (r + 1) < n_rows;

        const float *ptr_in0 = row0_valid ? (src + (r + 0) * k_stride) : NULL;
        const float *ptr_in1 = row1_valid ? (src + (r + 1) * k_stride) : NULL;

        uint32_t c = 0;
        for (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (row0_valid) v0 = *(const HVX_Vector *)(ptr_in0 + c);
            if (row1_valid) v1 = *(const HVX_Vector *)(ptr_in1 + c);

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        if (c < k_block) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (row0_valid) v0 = *(const HVX_Vector *)(ptr_in0 + c);
            if (row1_valid) v1 = *(const HVX_Vector *)(ptr_in1 + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }
}

typedef struct {
    __fp16      *dst;
    const float *src;
    uint32_t     n_tasks;
    uint32_t     n_tot_chunks;
    uint32_t     n_chunks_per_task;
    uint32_t     k_block;
    uint32_t     k_stride;
    uint32_t     k_valid;
    struct htp_thread_trace * traces;
    struct htp_context * ctx;
    float              * vtcm_f32_act;
} activation_transfer_task_state_t;

static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        const float *restrict src,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        float *thread_f32_act) {

    const uint32_t R = HTP_MM_DMA_ACT_ROWS_PER_STEP;
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    const uint32_t n_steps = n_rows_padded / R;

    // pre-fetch step 0
    if (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_ptr(thread_f32_act, src),
                       k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
    }

    for (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = R * s;
        float *curr_buf = thread_f32_act + (s % 2) * R * k_block;

        if (r < n_rows) {
            dma_queue_pop(dma_q);
        }

        uint32_t next_s = s + 1;
        uint32_t next_r = R * next_s;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            const float *next_src = src + next_r * k_stride;
            float *next_buf = thread_f32_act + (next_s % 2) * R * k_block;
            dma_queue_push(dma_q, dma_make_ptr(next_buf, next_src),
                           k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
        }

        #pragma unroll
        for (uint32_t i = 0; i < HTP_MM_DMA_ACT_ROWS_PER_STEP; i += 2) {
            uint32_t curr_r = r + i;
            const bool row0_valid = (curr_r < n_rows);
            const bool row1_valid = (curr_r + 1) < n_rows;

            const float *ptr_in0 = curr_buf + i * k_block;
            const float *ptr_in1 = curr_buf + (i + 1) * k_block;

            uint32_t c = 0;
            for (; c + 32 <= k_valid; c += 32) {
                HVX_Vector v0 = Q6_V_vzero();
                HVX_Vector v1 = Q6_V_vzero();
                if (row0_valid) v0 = *(const HVX_Vector *)(ptr_in0 + c);
                if (row1_valid) v1 = *(const HVX_Vector *)(ptr_in1 + c);

                HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

                uint32_t r0       = curr_r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
                uint32_t r1       = curr_r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx
                uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
                uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

                HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
                tile[r1 / 2]     = v_out;
            }
            if (c < k_block) {
                HVX_Vector v0 = Q6_V_vzero();
                HVX_Vector v1 = Q6_V_vzero();
                if (row0_valid) v0 = *(const HVX_Vector *)(ptr_in0 + c);
                if (row1_valid) v1 = *(const HVX_Vector *)(ptr_in1 + c);

                uint32_t rem = k_valid - c;
                HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
                v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
                v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

                HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

                uint32_t r0       = curr_r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
                uint32_t r1       = curr_r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx
                uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
                uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

                HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
                tile[r1 / 2]     = v_out;
            }
        }
    }
}

typedef struct {
    const struct mmid_row_mapping  *matrix_rows;
    __fp16                         *dst;
    const float                    *src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    struct fastdiv_values           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        start_row;
    uint32_t                        cne1;
    uint32_t                        k_valid;
    struct htp_thread_trace        *traces;
} activation_transfer_gathered_task_state_t;

typedef struct {
    const struct mmid_row_mapping  *matrix_rows;
    const __fp16                   *vtcm_src;
    float                          *dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        n_cols;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        start_row;
    uint32_t                        cne1;
    struct htp_thread_trace        *traces;
} output_transfer_scattered_task_state_t;

static void transfer_activation_chunk_fp32_to_fp16_gathered(
            __fp16 *restrict vtcm_dst,
            const float *restrict src,
            uint32_t start_row,
            uint32_t n_rows,
            uint32_t k_block,
            const struct mmid_row_mapping *matrix_rows,
            uint32_t cur_a,
            uint32_t mapping_stride,
            uint32_t ne11,
            const struct fastdiv_values * ne11_div,
            size_t nb11,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
    const uint32_t n_rows_tiled  = (n_rows / HTP_MM_HMX_TILE_N_ROWS) * HTP_MM_HMX_TILE_N_ROWS;

    uint32_t r = 0;

    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = r_idx0 % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];

        uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
        uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);

        const float *row0_ptr = (const float *) ((const uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        const float *row1_ptr = (const float *) ((const uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);

        uint32_t c = 0;
        for (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = *(const HVX_Vector *)(row0_ptr + c);
            HVX_Vector v1 = *(const HVX_Vector *)(row1_ptr + c);
            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        if (c < k_block) {
            HVX_Vector v0 = *(const HVX_Vector *)(row0_ptr + c);
            HVX_Vector v1 = *(const HVX_Vector *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }

    for (; r < n_rows_padded; r += 2) {
        uint32_t r_idx0 = start_row + r;
        uint32_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = r_idx0 % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        const bool row0_valid = (start_row + r + 0) < cne1;
        const bool row1_valid = (start_row + r + 1) < cne1;

        const float *row0_ptr = NULL;
        const float *row1_ptr = NULL;

        if (row0_valid) {
            struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + (start_row + r + 0)];
            uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
            row0_ptr = (const float *) ((const uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        }
        if (row1_valid) {
            struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + (start_row + r + 1)];
            uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);
            row1_ptr = (const float *) ((const uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        for (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (row0_valid) v0 = *(const HVX_Vector *)(row0_ptr + c);
            if (row1_valid) v1 = *(const HVX_Vector *)(row1_ptr + c);

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        if (c < k_block) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (row0_valid) v0 = *(const HVX_Vector *)(row0_ptr + c);
            if (row1_valid) v1 = *(const HVX_Vector *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }
}

static void transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            __fp16 *restrict vtcm_dst,
            const float *restrict src,
            uint32_t start_row,
            uint32_t n_rows,
            uint32_t k_block,
            const struct mmid_row_mapping *matrix_rows,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t nb12,
            uint32_t cne1,
            uint32_t k_valid) {
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
    const uint32_t n_rows_tiled  = (n_rows / HTP_MM_HMX_TILE_N_ROWS) * HTP_MM_HMX_TILE_N_ROWS;

    uint32_t r = 0;

    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = r_idx0 % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];

        const float *row0_ptr = (const float *) ((const uint8_t *) src + mapping0.i2 * nb12);
        const float *row1_ptr = (const float *) ((const uint8_t *) src + mapping1.i2 * nb12);

        uint32_t c = 0;
        for (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = *(const HVX_Vector *)(row0_ptr + c);
            HVX_Vector v1 = *(const HVX_Vector *)(row1_ptr + c);
            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        if (c < k_block) {
            HVX_Vector v0 = *(const HVX_Vector *)(row0_ptr + c);
            HVX_Vector v1 = *(const HVX_Vector *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }

    for (; r < n_rows_padded; r += 2) {
        uint32_t r_idx0 = start_row + r;
        uint32_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = r_idx0 % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        const bool row0_valid = (start_row + r + 0) < cne1;
        const bool row1_valid = (start_row + r + 1) < cne1;

        const float *row0_ptr = NULL;
        const float *row1_ptr = NULL;

        if (row0_valid) {
            struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + (start_row + r + 0)];
            row0_ptr = (const float *) ((const uint8_t *) src + mapping0.i2 * nb12);
        }
        if (row1_valid) {
            struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + (start_row + r + 1)];
            row1_ptr = (const float *) ((const uint8_t *) src + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        for (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (row0_valid) v0 = *(const HVX_Vector *)(row0_ptr + c);
            if (row1_valid) v1 = *(const HVX_Vector *)(row1_ptr + c);

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        if (c < k_block) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (row0_valid) v0 = *(const HVX_Vector *)(row0_ptr + c);
            if (row1_valid) v1 = *(const HVX_Vector *)(row1_ptr + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }
}

static void transfer_output_chunk_fp16_to_fp32_scattered(
            float *restrict dst,
            const __fp16 *restrict vtcm_src,
            uint32_t start_row,
            uint32_t n_rows,
            uint32_t n_cols,
            const struct mmid_row_mapping *matrix_rows,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            uint32_t cne1) {
    assert(n_cols % HTP_MM_HMX_TILE_N_COLS == 0);
    const size_t tile_row_stride = (n_cols / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;

    const HVX_Vector one = hvx_vec_splat_f16(1.0);

    for (size_t r = 0; r < n_rows; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        const size_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ROWS;
        const size_t r1 = (r_idx0 % HTP_MM_HMX_TILE_N_ROWS) / 2;  // index of the row pair within the tile
        const __fp16 *row_base = vtcm_src + r0 * tile_row_stride;

        if (r_idx0 >= cne1) break;

        struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        float *output_row0 = (float *) ((uint8_t *) dst + mapping0.i1 * dst_nb1 + mapping0.i2 * dst_nb2);

        float *output_row1 = NULL;
        if (r_idx1 < cne1) {
            struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];
            output_row1 = (float *) ((uint8_t *) dst + mapping1.i1 * dst_nb1 + mapping1.i2 * dst_nb2);
        }

        #pragma unroll(4)
        for (size_t c = 0; c < (size_t)n_cols; c += HTP_MM_HMX_TILE_N_COLS) {
            const size_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            const __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_Vector *pv_out0 = (HVX_Vector *) (output_row0 + c);
            HVX_Vector *pv_out1 = output_row1 ? (HVX_Vector *) (output_row1 + c) : NULL;

            *pv_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            if (pv_out1) {
                *pv_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
            }
        }
    }
}
