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

typedef struct tiled_dequantize_state tiled_dequantize_state_t;
typedef void (*tiled_dequant_task_fn_t)(const tiled_dequantize_state_t * state, uint32_t start_tile, uint32_t end_tile);

struct tiled_dequantize_state {
    __fp16                  * dst;
    const uint8_t           * src;
    tiled_dequant_task_fn_t   task_fn;

    struct fastdiv_values     n_k_tiles_div;
    uint32_t                  n_k_tiles;
    uint32_t                  n_weight_tiles;
    uint32_t                  n_tot_tiles;
    uint32_t                  n_tiles_per_task;
    uint32_t                  tile_size;
    uint32_t                  aligned_tile_size;
    uint32_t                  n_tasks;
    uint32_t                  n_cols;
    uint32_t                  k_block;
    size_t                    row_stride;
    uint32_t                  weight_type;
};

static void dequantize_tiled_weight_to_fp16_task_q1_0(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector v_neg_one = Q6_Vb_vsplat_R(-1);
    const HVX_Vector v_pos_one = Q6_Vb_vsplat_R(1);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + 4 * t * HTP_MM_HMX_TILE_N_ELMS;

        const HVX_Vector v_sc = hvx_vmem(tile_src + 512);
        const HVX_Vector v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        for (uint32_t q = 0; q < 4; q++) {
            const HVX_Vector v_plane = hvx_vmem(tile_src + q * 128);
            __fp16 * quarter_dst = dst_ptr + q * HTP_MM_HMX_TILE_N_ELMS;

            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const HVX_Vector     v_q  = unpack_q1_0_group(v_plane, g, v_neg_one, v_pos_one);
                const HVX_VectorPair vp16 = Q6_Wh_vunpack_Vb(v_q);
                const HVX_VectorPair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(quarter_dst + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(quarter_dst + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

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

static void dequantize_tiled_weight_to_fp16_task_q5_k(
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

        // Q5_K: OR in the 5th bit from the plane
        HVX_Vector v_plane = hvx_vmem(tile_src + 640);
        v_lo0 = hvx_q5k_or_hibit(v_lo0, v_plane, 0);
        v_hi0 = hvx_q5k_or_hibit(v_hi0, v_plane, 1);
        v_lo1 = hvx_q5k_or_hibit(v_lo1, v_plane, 2);
        v_hi1 = hvx_q5k_or_hibit(v_hi1, v_plane, 3);
        v_lo2 = hvx_q5k_or_hibit(v_lo2, v_plane, 4);
        v_hi2 = hvx_q5k_or_hibit(v_hi2, v_plane, 5);
        v_lo3 = hvx_q5k_or_hibit(v_lo3, v_plane, 6);
        v_hi3 = hvx_q5k_or_hibit(v_hi3, v_plane, 7);

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

static void dequantize_tiled_weight_to_fp16_task_q5_0(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc = hvx_vmem(tile_src + 640);
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

        // Q5_0: OR 0xF0 where the plane flags a clear 5th bit (q - 16)
        HVX_Vector v_plane = hvx_vmem(tile_src + 512);
        v_lo0 = hvx_q5_0_or_neg(v_lo0, v_plane, 0);
        v_hi0 = hvx_q5_0_or_neg(v_hi0, v_plane, 1);
        v_lo1 = hvx_q5_0_or_neg(v_lo1, v_plane, 2);
        v_hi1 = hvx_q5_0_or_neg(v_hi1, v_plane, 3);
        v_lo2 = hvx_q5_0_or_neg(v_lo2, v_plane, 4);
        v_hi2 = hvx_q5_0_or_neg(v_hi2, v_plane, 5);
        v_lo3 = hvx_q5_0_or_neg(v_lo3, v_plane, 6);
        v_hi3 = hvx_q5_0_or_neg(v_hi3, v_plane, 7);

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

// Q6_K stores 6-bit weights and one fp16 scale per 16 k, see HTP_TENSOR_TILE_SIZE_Q6_K.
// A k-group holds 4 k per row, the HMX tile holds 2, so each group is dealt into two tiles.
static void dequantize_tiled_weight_to_fp16_task_q6_k(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector i32     = Q6_Vb_vsplat_R(32);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const HVX_Vector * vptr = (const HVX_Vector *) (state->src + t * state->aligned_tile_size);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc      = vptr[6];
        HVX_Vector v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_Vector v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_Vector v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #pragma unroll
        for (int g = 0; g < 8; g += 4) {
            const HVX_Vector v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_Vector v_q0 = unpack_q6_k_group(vptr, g + 0, mask_0f, mask_03, i32);
            HVX_Vector v_q1 = unpack_q6_k_group(vptr, g + 1, mask_0f, mask_03, i32);
            HVX_Vector v_q2 = unpack_q6_k_group(vptr, g + 2, mask_0f, mask_03, i32);
            HVX_Vector v_q3 = unpack_q6_k_group(vptr, g + 3, mask_0f, mask_03, i32);

            HVX_VectorPair vp16_0 = Q6_Wh_vunpack_Vb(v_q0);
            HVX_VectorPair vp16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_VectorPair vp16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_VectorPair vp16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_VectorPair vp_k0  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_0), Q6_V_lo_W(vp16_0), -4);
            HVX_VectorPair vp_k1  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_1), Q6_V_lo_W(vp16_1), -4);
            HVX_VectorPair vp_k2  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_2), Q6_V_lo_W(vp16_2), -4);
            HVX_VectorPair vp_k3  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_3), Q6_V_lo_W(vp16_3), -4);

            HVX_Vector v_out00 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale));
            HVX_Vector v_out01 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale));
            HVX_Vector v_out10 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale));
            HVX_Vector v_out11 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale));
            HVX_Vector v_out20 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k2)), v_scale));
            HVX_Vector v_out21 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k2)), v_scale));
            HVX_Vector v_out30 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k3)), v_scale));
            HVX_Vector v_out31 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k3)), v_scale));

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = v_out00;
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = v_out01;
            hvx_vmem(dst_ptr + (2 * g + 2) * 64) = v_out10;
            hvx_vmem(dst_ptr + (2 * g + 3) * 64) = v_out11;
            hvx_vmem(dst_ptr + (2 * g + 4) * 64) = v_out20;
            hvx_vmem(dst_ptr + (2 * g + 5) * 64) = v_out21;
            hvx_vmem(dst_ptr + (2 * g + 6) * 64) = v_out30;
            hvx_vmem(dst_ptr + (2 * g + 7) * 64) = v_out31;
        }
    }
}

#define NVFP4_UE4M3_F16_MAN_SHIFT  7       // 10 - 3 mantissa bits
#define NVFP4_UE4M3_F16_REBIAS     0x2000  // (15 - 7) << 10
#define NVFP4_UE4M3_F16_SUB_SCALE  0x1800  // 2^-9 as fp16 bits
#define IQ2XXS_D_SCALE_F16         0x3000  // 0.125 as fp16 bits

// UE4M3 codes in the low byte of each halfword -> fp16
static inline HVX_Vector nvfp4_ue4m3_to_f16(HVX_Vector v_code) {
    HVX_Vector v_mask = Q6_Vh_vsplat_R(NVFP4_UE4M3_MASK);
    HVX_Vector v_x    = Q6_V_vand_VV(v_code, v_mask);

    HVX_Vector v_norm = Q6_Vh_vadd_VhVh(Q6_Vh_vasl_VhR(v_x, NVFP4_UE4M3_F16_MAN_SHIFT), Q6_Vh_vsplat_R(NVFP4_UE4M3_F16_REBIAS));
    HVX_Vector v_sub  = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(v_x), Q6_Vh_vsplat_R(NVFP4_UE4M3_F16_SUB_SCALE)));

    HVX_VectorPred q_sub = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(NVFP4_UE4M3_MIN_NORMAL), v_x);
    HVX_VectorPred q_nan = Q6_Q_vcmp_eq_VhVh(v_x, v_mask);

    return Q6_V_vmux_QVV(q_nan, Q6_V_vzero(), Q6_V_vmux_QVV(q_sub, v_sub, v_norm));
}

// NVFP4 tiles are MXFP4 quants with one UE4M3 scale per 16 k (sub-block 0: k 0..15, sub-block 1: k 16..31)
static void dequantize_tiled_weight_to_fp16_task_nvfp4(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector vlut_cvt = hvx_vmem(mxfp4_to_fp16_lut);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_d = nvfp4_ue4m3_to_f16(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmem(tile_src + NVFP4_SCALE_OFF))));
        HVX_VectorPair vp_d = Q6_W_vshuff_VVR(v_d, v_d, -2);
        HVX_Vector v_scale_k0  = Q6_V_lo_W(vp_d);
        HVX_Vector v_scale_k16 = Q6_V_hi_W(vp_d);

        #pragma unroll
        for (int g = 0; g < 4; g++) {
            const HVX_Vector v_scale = (g < 2) ? v_scale_k0 : v_scale_k16;

            HVX_Vector vq   = hvx_vmem(tile_src + g * 128);
            HVX_Vector v_lo = Q6_V_vand_VV(vq, mask_h4);
            HVX_Vector v_hi = Q6_Vub_vlsr_VubR(vq, 4);

            HVX_VectorPair vp_shuf = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
            HVX_VectorPair vp_lo = Q6_Wh_vlut16_VbVhR(Q6_Vb_vshuff_Vb(Q6_V_lo_W(vp_shuf)), vlut_cvt, 0);
            HVX_VectorPair vp_hi = Q6_Wh_vlut16_VbVhR(Q6_Vb_vshuff_Vb(Q6_V_hi_W(vp_shuf)), vlut_cvt, 0);

            hvx_vmem(dst_ptr + (4 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo), v_scale));
            hvx_vmem(dst_ptr + (4 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo), v_scale));
            hvx_vmem(dst_ptr + (4 * g + 2) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi), v_scale));
            hvx_vmem(dst_ptr + (4 * g + 3) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi), v_scale));
        }
    }
}

// IQ2_XXS: int8 grid weights from the VTCM LUT, scaled by d / 8 * (2 * scale + 1) per row
static void dequantize_tiled_weight_to_fp16_task_iq2_xxs(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    HVX_Vector v_d = Q6_V_vzero();
    const HVX_Vector v_eighth = Q6_Vh_vsplat_R(IQ2XXS_D_SCALE_F16);

    HVX_Vector v_sgn = Q6_V_vzero();
    if (start_tile < end_tile) {
        v_sgn = iq2xxs_gather_issue(state->src + start_tile * state->aligned_tile_size);
    }

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        // rows hold all k tiles and n_k_tiles is a multiple of 8, so t % 8 is the tile within its super-block
        if (t == start_tile || (t & (IQ2XXS_TILES_PER_SB - 1)) == 0) {
            v_d = iq2xxs_gather_d(tile - (t & (IQ2XXS_TILES_PER_SB - 1)) * state->aligned_tile_size);
        }

        HVX_Vector v_c[2], v_w[2 * IQ2XXS_GROUPS], v_ls;
        iq2xxs_gather_wait(tile, v_c);
        const HVX_Vector v_sgn_cur = v_sgn;
        if (t + 1 < end_tile) {
            v_sgn = iq2xxs_gather_issue(tile + state->aligned_tile_size);
        }
        iq2xxs_decode_expand(v_c, v_sgn_cur, v_w, &v_ls);

        HVX_Vector v_ls_h  = Q6_Vhf_equals_Vh(Q6_Vh_vpacke_VwVw(v_ls, v_ls));
        HVX_Vector v_sc    = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_d, v_ls_h));
        HVX_Vector v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        #pragma unroll
        for (int g = 0; g < 2 * IQ2XXS_GROUPS; g++) {
            HVX_VectorPair vp16 = Q6_Wh_vunpack_Vb(v_w[g]);
            HVX_VectorPair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            // d / 8 is subnormal in fp16 for real weights, so apply the 1/8 after the multiply by d * ls
            HVX_Vector v_lo = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
            HVX_Vector v_hi = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_lo, v_eighth));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hi, v_eighth));
        }
    }
}

// Q3_K stores 3-bit weights and one fp16 scale per 16 k, see HTP_TENSOR_TILE_SIZE_Q3_K.
static void dequantize_tiled_weight_to_fp16_task_q3_k(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const HVX_Vector * vptr = (const HVX_Vector *) (state->src + t * state->aligned_tile_size);
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc      = vptr[3];
        HVX_Vector v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_Vector v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_Vector v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #pragma unroll
        for (int g = 0; g < 8; g++) {
            const HVX_Vector v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;

            HVX_Vector     v_q   = unpack_q3_k_group(vptr, g, mask_03);
            HVX_VectorPair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_VectorPair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
        }
    }
}

// Q2_K stores 2-bit weights, the ggml scale bytes and d / -dmin per super-block, see HTP_TENSOR_TILE_SIZE_Q2_K.
static void dequantize_tiled_weight_to_fp16_task_q2_k(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    HVX_Vector v_d = Q6_V_vzero();
    HVX_Vector v_dm = Q6_V_vzero();

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile = state->src + t * state->aligned_tile_size;
        const HVX_Vector * vptr = (const HVX_Vector *) tile;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        // rows hold all k tiles and n_k_tiles is a multiple of 8, so t % 8 is the tile within its super-block
        if (t == start_tile || (t & 7) == 0) {
            q2_k_load_dmin(tile - (t & 7) * state->aligned_tile_size, &v_d, &v_dm);
        }

        HVX_Vector v_sc, v_m;
        q2_k_unpack_scales(tile, v_d, v_dm, &v_sc, &v_m);
        HVX_Vector v_sc_k16   = Q6_V_vror_VR(v_sc, 64);
        HVX_Vector v_m_k16    = Q6_V_vror_VR(v_m, 64);
        HVX_Vector v_scale_k0   = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_Vector v_scale_k16  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));
        HVX_Vector v_offset_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m, v_m, -2));
        HVX_Vector v_offset_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_m_k16, v_m_k16, -2));

        #pragma unroll
        for (int g = 0; g < 8; g++) {
            const HVX_Vector v_scale  = (g < 4) ? v_scale_k0  : v_scale_k16;
            const HVX_Vector v_offset = (g < 4) ? v_offset_k0 : v_offset_k16;

            HVX_Vector     v_q   = unpack_q3_k_low2(vptr, g, mask_03);
            HVX_VectorPair vp16  = Q6_Wh_vunpack_Vb(v_q);
            HVX_VectorPair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale), v_offset));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(
                Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale), v_offset));
        }
    }
}

// Q2_0 stores codes 0..3 (value code - 1) and one fp16 scale per row, see HTP_MM_WEIGHT_TILE_SIZE_Q2_0.
// One packed 64-k tile expands into two consecutive 32-k fp16 tiles.
static void dequantize_tiled_weight_to_fp16_task_q2_0(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector i1      = Q6_Vb_vsplat_R(1);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const HVX_Vector * vptr = (const HVX_Vector *) (state->src + t * state->aligned_tile_size);
        __fp16 * dst_base = state->dst + 2 * t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc    = vptr[4];
        HVX_Vector v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        for (int half = 0; half < 2; half++) {
            __fp16 * dst_ptr = dst_base + half * HTP_MM_HMX_TILE_N_ELMS;

            #pragma unroll
            for (int g = 0; g < 8; g++) {
                HVX_Vector     v_q  = Q6_Vb_vsub_VbVb(unpack_q3_k_low2(vptr + 2 * half, g, mask_03), i1);
                HVX_VectorPair vp16 = Q6_Wh_vunpack_Vb(v_q);
                HVX_VectorPair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

                hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
                hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
                    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
            }
        }
    }
}

static __attribute__((noinline))
void convert_f16_weight_to_fp16_tiles_task(
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

                HVX_Vector v0 = (row0 < state->n_cols) ? hvx_vmemu((const __fp16 *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_Vector v1 = (row1 < state->n_cols) ? hvx_vmemu((const __fp16 *)(r1 + byte_off)) : Q6_V_vzero();

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
        }
        ++t; ++kt;
    }
}

static __attribute__((noinline))
void quantize_f32_weight_to_fp16_tiles_task(
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

                HVX_Vector v0_f32 = (row0 < state->n_cols) ? hvx_vmem((const float *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_Vector v1_f32 = (row1 < state->n_cols) ? hvx_vmem((const float *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_Vector v_out = hvx_vec_f32_to_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v_out);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);

                HVX_Vector v_out_hi = Q6_V_vror_VR(v_out, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v_out_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
        }
        ++t; ++kt;
    }
}

static __attribute__((noinline))
void convert_bf16_weight_to_fp16_tiles_task(
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
            uint32_t byte_off = kt * 32 * sizeof(uint16_t);

            HVX_Vector v_off = v_scat_base;
            for (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ROWS; r += 2) {
                uint32_t row0 = ct * HTP_MM_HMX_TILE_N_COLS + r;
                uint32_t row1 = row0 + 1;

                const uint8_t *r0 = state->src + row0 * state->row_stride;
                const uint8_t *r1 = state->src + row1 * state->row_stride;

                HVX_Vector v0_raw = (row0 < state->n_cols) ? hvx_vmemu((const void *)(r0 + byte_off)) : Q6_V_vzero();
                HVX_Vector v1_raw = (row1 < state->n_cols) ? hvx_vmemu((const void *)(r1 + byte_off)) : Q6_V_vzero();

                HVX_Vector v0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v0_raw)), 16);
                HVX_Vector v1_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(v1_raw)), 16);

                HVX_Vector v_out = hvx_vec_f32_to_f16(v0_f32, v1_f32);

                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v_out);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);

                HVX_Vector v_out_hi = Q6_V_vror_VR(v_out, 64);
                Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v_out_hi);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
        }
        ++t; ++kt;
    }
}

// --- End tiled dequantizers ---

// dot-chunk functions require external HMX lock

static void core_dot_chunk_fp16_short(__fp16 *restrict output, const __fp16 *restrict activation,
                                const __fp16 *restrict weight, const __fp16 *restrict scales,
                                uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)scales));

    const size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const uint32_t range = 2048u * n_dot_tiles - 1;

    for (uint32_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 *row_base = activation + r * dot_stride;
        const __fp16 *col_base = weight;
        __fp16 *out_tile = output + r * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;

        for (size_t c = 0; c < n_col_tiles; ++c) {
            asm volatile(HMX_CLRACC_F16());
            asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_base), "r"(col_base));
            asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(out_tile), "r"(0) : "memory");
            col_base += dot_stride;
            out_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

static void core_dot_chunk_fp16(__fp16 *restrict output, const __fp16 *restrict activation,
                          const __fp16 *restrict weight, const __fp16 *restrict scales,
                          uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles) {
    if (n_dot_tiles <= 32) {
        core_dot_chunk_fp16_short(output, activation, weight, scales, n_row_tiles, n_col_tiles, n_dot_tiles);
        return;
    }
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)scales));

    const size_t dot_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    for (uint32_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 *row_base = activation + r * dot_stride;
        const __fp16 *col_base = weight;
        __fp16 *out_tile = output + r * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;

        for (size_t c = 0; c < n_col_tiles; ++c) {
            const __fp16 *row_tiles = row_base;
            const __fp16 *col_tiles = col_base;

            asm volatile(HMX_CLRACC_F16());

            const uint32_t n_loops = n_dot_tiles / 32;
            const uint32_t rem = n_dot_tiles % 32;

            for (uint32_t l = 0; l < n_loops; ++l) {
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(row_tiles), "r"(col_tiles));
                row_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                col_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            if (rem > 0) {
                const uint32_t range = 2048u * rem - 1;
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_tiles), "r"(col_tiles));
            }

            asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(out_tile), "r"(0) : "memory");

            col_base += dot_stride;
            out_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

static void core_mma_chunk_fp16_short(__fp16 *restrict c, const __fp16 *restrict a, const __fp16 *restrict b,
                                const __fp16 *restrict col_scales, const __fp16 *restrict eye_tile,
                                uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles, bool zero_init) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);
    __builtin_assume(n_dot_tiles <= 32);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)col_scales));

    const size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const uint32_t range = 2048u * n_dot_tiles - 1;

    for (size_t i = 0; i < n_row_tiles; ++i) {
        const __fp16 *row_base = a + i * dot_tile_stride;
        __fp16 *res_base = c + i * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
        const __fp16 *col_base = b;
        __fp16 *accum_tile = res_base;

        for (size_t j = 0; j < n_col_tiles; ++j) {
            asm volatile(HMX_CLRACC_F16());

            if (!zero_init) {
                asm volatile(HMX_LOAD_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_base), "r"(col_base));

            asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            col_base   += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

static void core_mma_chunk_fp16(__fp16 *restrict c, const __fp16 *restrict a, const __fp16 *restrict b,
                          const __fp16 *restrict col_scales, const __fp16 *restrict eye_tile,
                          uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles, bool zero_init) {
    if (n_dot_tiles <= 32) {
        core_mma_chunk_fp16_short(c, a, b, col_scales, eye_tile, n_row_tiles, n_col_tiles, n_dot_tiles, zero_init);
        return;
    }
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 32);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)col_scales));

    const size_t dot_tile_stride = n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    for (size_t i = 0; i < n_row_tiles; ++i) {
        const __fp16 *row_base = a + i * dot_tile_stride;
        __fp16 *res_base = c + i * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
        const __fp16 *col_base = b;
        __fp16 *accum_tile = res_base;

        for (size_t j = 0; j < n_col_tiles; ++j) {
            const __fp16 *col_tiles = col_base;
            const __fp16 *row_tiles = row_base;

            asm volatile(HMX_CLRACC_F16());

            if (!zero_init) {
                asm volatile(HMX_LOAD_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }

            const uint32_t n_loops = n_dot_tiles / 32;
            const uint32_t rem = n_dot_tiles % 32;

            for (uint32_t l = 0; l < n_loops; ++l) {
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(row_tiles), "r"(col_tiles));
                row_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                col_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }

            if (rem > 0) {
                const uint32_t range = 2048u * rem - 1;
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_tiles), "r"(col_tiles));
            }

            asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            col_base += dot_tile_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

// output : fp16 -> f32 / f16

static inline HVX_Vector hmx_load_src2_f32_32(const void * src2_row_base, size_t c, size_t src2_elem_size) {
    if (src2_elem_size == sizeof(__fp16)) {
        const __fp16 * p = (const __fp16 *) src2_row_base + c;
        return Q6_V_lo_W(hvx_vec_f16_to_f32(hvx_vmemu(p)));
    }
    const float * p = (const float *) src2_row_base + c;
    return hvx_vmemu(p);
}

static inline HVX_VectorPair hmx_load_src2_f32_64(const void * src2_row_base, size_t c, size_t src2_elem_size) {
    if (src2_elem_size == sizeof(__fp16)) {
        const __fp16 * p = (const __fp16 *) src2_row_base + c;
        return hvx_vec_f16_to_f32(hvx_vmemu(p));
    }
    const float * p = (const float *) src2_row_base + c;
    return Q6_W_vcombine_VV(hvx_vmemu(p + 32), hvx_vmemu(p));
}

static inline HVX_Vector hmx_pack_f32_to_f16_clamped(HVX_Vector v0, HVX_Vector v1) {
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    v0 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v0));
    v1 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v1));
    return hvx_vec_f32_to_f16(v0, v1);
}

static inline HVX_Vector hmx_add_src2_to_f16_vec64(HVX_Vector v_f16, const void * src2_row_base, size_t c, size_t src2_elem_size) {
    HVX_VectorPair vp_dst  = hvx_vec_f16_to_f32(v_f16);
    HVX_VectorPair vp_src2 = hmx_load_src2_f32_64(src2_row_base, c, src2_elem_size);
    HVX_Vector v0 = hvx_vec_add_f32_f32(Q6_V_lo_W(vp_dst), Q6_V_lo_W(vp_src2));
    HVX_Vector v1 = hvx_vec_add_f32_f32(Q6_V_hi_W(vp_dst), Q6_V_hi_W(vp_src2));
    return hmx_pack_f32_to_f16_clamped(v0, v1);
}

static inline HVX_Vector hmx_add_src2_to_f16_tail(HVX_Vector v_f16, const void * src2_row_base, size_t c, size_t valid_c, size_t src2_elem_size) {
    HVX_VectorPair vp_dst = hvx_vec_f16_to_f32(v_f16);
    HVX_Vector s0 = hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size);
    HVX_Vector v0 = hvx_vec_add_f32_f32(Q6_V_lo_W(vp_dst), s0);
    HVX_Vector v1 = Q6_V_vzero();
    if (valid_c > 32) {
        HVX_Vector s1 = (src2_elem_size == sizeof(__fp16))
            ? Q6_V_hi_W(hvx_vec_f16_to_f32(hvx_vmemu((const __fp16 *) src2_row_base + c)))
            : hvx_vmemu((const float *) src2_row_base + c + 32);
        v1 = hvx_vec_add_f32_f32(Q6_V_hi_W(vp_dst), s1);
    }
    return hmx_pack_f32_to_f16_clamped(v0, v1);
}

static void transfer_output_chunk_fp16_to_fp32_col_chunk(
    float *restrict dst,
    const void *restrict src2,
    const __fp16 *restrict vtcm_src,
    uint32_t start_row,
    uint32_t n_rows,
    uint32_t c_len,
    uint32_t total_n_cols,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_cols,
    size_t src2_elem_size
) {
    const size_t tile_row_stride = hmx_ceil_div(total_n_cols, HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;

    const HVX_Vector one = hvx_vec_splat_f16(1.0);

    const size_t limit_c         = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned = (limit_c & ~31);

    for (size_t r = 0; r < n_rows; r += 2) {
        const size_t r_idx0 = start_row + r + 0;
        const size_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ROWS;
        const size_t r1 = (r_idx0 % HTP_MM_HMX_TILE_N_ROWS) / 2;  // index of the row pair within the tile
        const __fp16 *row_base = vtcm_src + r0 * tile_row_stride;
        float *output_row_base = dst + r * dst_stride;  // global memory row base for row r (and r+1)
        const uint8_t *src2_row0 = src2 ? ((const uint8_t *) src2 + r * src2_stride * src2_elem_size) : NULL;
        const uint8_t *src2_row1 = src2 ? (src2_row0 + src2_stride * src2_elem_size) : NULL;

        #pragma unroll(4)
        for (size_t c = 0; c < limit_c_aligned; c += HTP_MM_HMX_TILE_N_COLS) {
            const size_t c0    = c / HTP_MM_HMX_TILE_N_COLS;
            const __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            if (src2_row0) {
                HVX_Vector v_src2_0 = hmx_load_src2_f32_32(src2_row0, c, src2_elem_size);
                v_out0 = hvx_vec_add_f32_f32(v_out0, v_src2_0);
            }
            hvx_vmemu(output_row_base + c + 0) = v_out0;

            if (r + 1 < n_rows) {
                HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                if (src2_row1) {
                    HVX_Vector v_src2_1 = hmx_load_src2_f32_32(src2_row1, c, src2_elem_size);
                    v_out1 = hvx_vec_add_f32_f32(v_out1, v_src2_1);
                }
                hvx_vmemu(output_row_base + c + dst_stride) = v_out1;
            }
        }

        if (limit_c_aligned < limit_c) {
            size_t c = limit_c_aligned;
            size_t valid_c = limit_c - c;
            const size_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            const __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            if (src2_row0) {
                HVX_Vector v_src2_0 = hmx_load_src2_f32_32(src2_row0, c, src2_elem_size);
                v_out0 = hvx_vec_add_f32_f32(v_out0, v_src2_0);
            }
            hvx_vec_store_u(output_row_base + c, valid_c * sizeof(float), v_out0);

            if (r + 1 < n_rows) {
                HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                if (src2_row1) {
                    HVX_Vector v_src2_1 = hmx_load_src2_f32_32(src2_row1, c, src2_elem_size);
                    v_out1 = hvx_vec_add_f32_f32(v_out1, v_src2_1);
                }
                hvx_vec_store_u(output_row_base + c + dst_stride, valid_c * sizeof(float), v_out1);
            }
        }
    }
}

static inline void transfer_output_chunk_fp16_to_fp32(
    float *restrict dst,
    const void *restrict src2,
    const __fp16 *restrict vtcm_src,
    uint32_t start_row,
    uint32_t n_rows,
    uint32_t n_cols,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_cols,
    size_t src2_elem_size
) {
    transfer_output_chunk_fp16_to_fp32_col_chunk(
        dst, src2, vtcm_src, start_row, n_rows, n_cols, n_cols, dst_stride, src2_stride, dst_cols, src2_elem_size
    );
}

static void transfer_output_chunk_fp16_to_fp16_col_chunk(
    __fp16 *restrict dst,
    const void *restrict src2,
    const __fp16 *restrict vtcm_src,
    uint32_t start_row,
    uint32_t n_rows,
    uint32_t c_len,
    uint32_t total_n_cols,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_cols,
    size_t src2_elem_size
) {
    const size_t tile_row_stride   = hmx_ceil_div(total_n_cols, HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
    const size_t limit_c           = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned64 = (limit_c & ~63u);
    const bool   dst_aligned_nt    = ((((uintptr_t) dst | ((size_t) dst_stride * sizeof(__fp16))) & 127u) == 0);

    for (size_t r = 0; r < n_rows; r += 2) {
        const size_t r_idx0 = start_row + r;
        const size_t r0 = r_idx0 / HTP_MM_HMX_TILE_N_ROWS;
        const size_t r1 = (r_idx0 % HTP_MM_HMX_TILE_N_ROWS) / 2;
        const __fp16 *row_base = vtcm_src + r0 * tile_row_stride;
        __fp16 *output_row_base = dst + r * dst_stride;
        const uint8_t *src2_row0 = src2 ? ((const uint8_t *) src2 + r * src2_stride * src2_elem_size) : NULL;
        const uint8_t *src2_row1 = src2 ? (src2_row0 + src2_stride * src2_elem_size) : NULL;

        for (size_t c = 0; c < limit_c_aligned64; c += 64) {
            const size_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            const __fp16 *tile0 = row_base + (c0 + 0) * HTP_MM_HMX_TILE_N_ELMS;
            const __fp16 *tile1 = row_base + (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v_t0 = ((const HVX_Vector *) tile0)[r1];
            HVX_Vector v_t1 = ((const HVX_Vector *) tile1)[r1];
            HVX_VectorPair vp = Q6_W_vdeal_VVR(v_t1, v_t0, -2);

            HVX_Vector v_out0 = Q6_V_lo_W(vp);
            if (src2_row0) {
                v_out0 = hmx_add_src2_to_f16_vec64(v_out0, src2_row0, c, src2_elem_size);
            }
            if (dst_aligned_nt) {
                hvx_vmem_store_nt(output_row_base + c, v_out0);
            } else {
                hvx_vmemu(output_row_base + c) = v_out0;
            }

            if (r + 1 < n_rows) {
                HVX_Vector v_out1 = Q6_V_hi_W(vp);
                if (src2_row1) {
                    v_out1 = hmx_add_src2_to_f16_vec64(v_out1, src2_row1, c, src2_elem_size);
                }
                if (dst_aligned_nt) {
                    hvx_vmem_store_nt(output_row_base + dst_stride + c, v_out1);
                } else {
                    hvx_vmemu(output_row_base + dst_stride + c) = v_out1;
                }
            }
        }

        if (limit_c_aligned64 < limit_c) {
            const size_t c       = limit_c_aligned64;
            const size_t valid_c = limit_c - c;
            const size_t c0      = c / HTP_MM_HMX_TILE_N_COLS;
            const __fp16 *tile0  = row_base + (c0 + 0) * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v_t0 = ((const HVX_Vector *) tile0)[r1];
            HVX_Vector v_t1 = (valid_c > 32)
                ? ((const HVX_Vector *) (row_base + (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS))[r1]
                : Q6_V_vzero();
            HVX_VectorPair vp = Q6_W_vdeal_VVR(v_t1, v_t0, -2);

            HVX_Vector v_out0 = Q6_V_lo_W(vp);
            if (src2_row0) {
                v_out0 = hmx_add_src2_to_f16_tail(v_out0, src2_row0, c, valid_c, src2_elem_size);
            }
            hvx_vec_store_u(output_row_base + c, valid_c * sizeof(__fp16), v_out0);

            if (r + 1 < n_rows) {
                HVX_Vector v_out1 = Q6_V_hi_W(vp);
                if (src2_row1) {
                    v_out1 = hmx_add_src2_to_f16_tail(v_out1, src2_row1, c, valid_c, src2_elem_size);
                }
                hvx_vec_store_u(output_row_base + dst_stride + c, valid_c * sizeof(__fp16), v_out1);
            }
        }
    }
}

static inline void transfer_output_chunk_fp16_to_fp16(
    __fp16 *restrict dst,
    const void *restrict src2,
    const __fp16 *restrict vtcm_src,
    uint32_t start_row,
    uint32_t n_rows,
    uint32_t n_cols,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_cols,
    size_t src2_elem_size
) {
    transfer_output_chunk_fp16_to_fp16_col_chunk(
        dst, src2, vtcm_src, start_row, n_rows, n_cols, n_cols, dst_stride, src2_stride, dst_cols, src2_elem_size
    );
}

typedef struct {
    const __fp16  *vtcm_src;
    void          *dst;
    const void    *src2;
    size_t         dst_elem_size;
    size_t         src2_elem_size;
    uint32_t       n_tasks;
    uint32_t       n_tot_chunks;
    uint32_t       n_chunks_per_task;
    uint32_t       n_cols;
    uint32_t       dst_stride;  // DDR row stride
    uint32_t       src2_stride; // DDR row stride for residual
    uint32_t       dst_cols;    // Actual output columns
} output_transfer_task_state_t;

// activations : fp32 -> fp16

static void transfer_activation_row_pair_fp32_to_fp16(
        __fp16 *restrict vtcm_dst,
        const float *restrict row0,
        const float *restrict row1,
        uint32_t r,
        uint32_t k_block,
        uint32_t k_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

    uint32_t c = 0;
    for (; c + 32 <= k_valid; c += 32) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        if (row0_valid) v0 = *(const HVX_Vector *)(row0 + c);
        if (row1_valid) v1 = *(const HVX_Vector *)(row1 + c);

        HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
        uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

        HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_out;
    }
    if (c < k_block) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        if (row0_valid) v0 = *(const HVX_Vector *)(row0 + c);
        if (row1_valid) v1 = *(const HVX_Vector *)(row1 + c);

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

// F16-input variant of transfer_activation_row_pair_fp32_to_fp16, for F16 activation
// (src1). Same shape as the F16 Q-prep in hmx-fa-kernels.h: one 128-byte load carries 64 f16
// = two tile columns, and Q6_W_vshuff_VVR interleaves the two rows straight into the HMX tile
// layout, so no F32 round-trip is needed. Rows are only 64-byte aligned when k_block is an odd
// multiple of the tile width, hence the unaligned load type.
static void transfer_activation_row_pair_f16_to_f16(__fp16 * restrict vtcm_dst,
                                                    const __fp16 * restrict row0,
                                                    const __fp16 * restrict row1,
                                                    uint32_t r,
                                                    uint32_t k_block,
                                                    uint32_t k_valid,
                                                    bool     row0_valid,
                                                    bool     row1_valid) {
    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

    const uint32_t n_tile_cols = k_block / HTP_MM_HMX_TILE_N_COLS;
    __fp16 * restrict tile_row = vtcm_dst + (size_t) r0 * n_tile_cols * HTP_MM_HMX_TILE_N_ELMS;

    const HVX_UVector * pv0 = row0_valid ? (const HVX_UVector *) row0 : NULL;
    const HVX_UVector * pv1 = row1_valid ? (const HVX_UVector *) row1 : NULL;

    uint32_t c = 0;
    for (; c + 64 <= k_valid; c += 64) {
        HVX_Vector     v0 = pv0 ? pv0[c / 64] : Q6_V_vzero();
        HVX_Vector     v1 = pv1 ? pv1[c / 64] : Q6_V_vzero();
        HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;

        HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

        tile0[r1 / 2] = Q6_V_lo_W(vp);
        tile1[r1 / 2] = Q6_V_hi_W(vp);
    }
    // Tail: fewer than 64 valid columns left, plus the k_valid..k_block padding that HMX will
    // still multiply, so it has to be written as zeros.
    for (; c < k_block; c += 64) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();

        if (c < k_valid) {
            uint32_t       rem  = k_valid - c;  // 1..63 valid f16 lanes
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(__fp16));
            if (pv0) {
                v0 = Q6_V_vmux_QVV(mask, pv0[c / 64], Q6_V_vzero());
            }
            if (pv1) {
                v1 = Q6_V_vmux_QVV(mask, pv1[c / 64], Q6_V_vzero());
            }
        }

        HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;

        HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        tile0[r1 / 2]      = Q6_V_lo_W(vp);

        if (c0 + 1 < n_tile_cols) {
            HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
            tile1[r1 / 2]      = Q6_V_hi_W(vp);
        }
    }
}

static void transfer_activation_row_pair_fp32_to_fp16_col_chunk(
        __fp16 *restrict vtcm_dst,
        const float *restrict row0, // offset by c_first
        const float *restrict row1, // offset by c_first
        uint32_t r,
        uint32_t k_block,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t k_chunk_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

    uint32_t c = 0;
    for (; c + 32 <= k_chunk_valid; c += 32) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        if (row0_valid) v0 = *(const HVX_Vector *)(row0 + c);
        if (row1_valid) v1 = *(const HVX_Vector *)(row1 + c);

        HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

        uint32_t c0       = (c_first + c) / HTP_MM_HMX_TILE_N_COLS;  // tile column index
        uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

        HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_out;
    }
    if (c < c_len) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        if (row0_valid) v0 = *(const HVX_Vector *)(row0 + c);
        if (row1_valid) v1 = *(const HVX_Vector *)(row1 + c);

        uint32_t rem = (k_chunk_valid > c) ? (k_chunk_valid - c) : 0;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
        v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

        HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

        uint32_t c0       = (c_first + c) / HTP_MM_HMX_TILE_N_COLS;  // tile column index
        uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

        HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_out;
    }
}

static void transfer_activation_row_pair_f16_to_f16_col_chunk(
        __fp16 *restrict vtcm_dst,
        const __fp16 *restrict row0,
        const __fp16 *restrict row1,
        uint32_t r,
        uint32_t k_block,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t k_chunk_valid,
        bool row0_valid,
        bool row1_valid) {

    const uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;
    const uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;
    const uint32_t n_tile_cols = k_block / HTP_MM_HMX_TILE_N_COLS;
    __fp16 * restrict tile_row = vtcm_dst + (size_t) r0 * n_tile_cols * HTP_MM_HMX_TILE_N_ELMS;

    const HVX_UVector * pv0 = row0_valid ? (const HVX_UVector *) row0 : NULL;
    const HVX_UVector * pv1 = row1_valid ? (const HVX_UVector *) row1 : NULL;

    uint32_t c = 0;
    for (; c + 64 <= k_chunk_valid; c += 64) {
        HVX_Vector     v0 = pv0 ? pv0[c / 64] : Q6_V_vzero();
        HVX_Vector     v1 = pv1 ? pv1[c / 64] : Q6_V_vzero();
        HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        const uint32_t c0 = (c_first + c) / HTP_MM_HMX_TILE_N_COLS;
        HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

        tile0[r1 / 2] = Q6_V_lo_W(vp);
        tile1[r1 / 2] = Q6_V_hi_W(vp);
    }
    for (; c < c_len; c += 64) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();

        if (c < k_chunk_valid) {
            const uint32_t rem  = k_chunk_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(__fp16));
            if (pv0) {
                v0 = Q6_V_vmux_QVV(mask, pv0[c / 64], Q6_V_vzero());
            }
            if (pv1) {
                v1 = Q6_V_vmux_QVV(mask, pv1[c / 64], Q6_V_vzero());
            }
        }

        HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
        const uint32_t c0 = (c_first + c) / HTP_MM_HMX_TILE_N_COLS;

        HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
        tile0[r1 / 2]      = Q6_V_lo_W(vp);

        if (c + 32 < c_len && c0 + 1 < n_tile_cols) {
            HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
            tile1[r1 / 2]      = Q6_V_hi_W(vp);
        }
    }
}

static inline HVX_Vector hvx_vec_load_u(const void * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) ((uintptr_t) src & ~(uintptr_t) 127);
    const uint32_t left_off  = (uint32_t) ((uintptr_t) src & 127);
    const uint32_t right_off = left_off + n;
    HVX_Vector v = (right_off > 128) ? Q6_V_valign_VVR(vsrc[1], vsrc[0], left_off) : Q6_V_vror_VR(vsrc[0], left_off);
    return Q6_V_vand_QV(Q6_Q_vsetq2_R(n), v);
}

static void transfer_activation_chunk_fp32_to_fp16_gathered(
            __fp16 *restrict vtcm_dst,
            const float *restrict src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
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
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];

        uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
        uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);

        const float *row0_ptr = (const float *) ((const uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        const float *row1_ptr = (const float *) ((const uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);

        uint32_t c = 0;
        for (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = hvx_vmemu(row0_ptr + c);
            HVX_Vector v1 = hvx_vmemu(row1_ptr + c);
            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        for (; c < k_block; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (c < k_valid) {
                uint32_t rem_b = (k_valid - c) * sizeof(float);
                v0 = hvx_vec_load_u(row0_ptr + c, rem_b);
                v1 = hvx_vec_load_u(row1_ptr + c, rem_b);
            }

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }

    for (; r < n_rows_padded; r += 2) {
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

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
            HVX_Vector v0 = row0_ptr ? hvx_vmemu(row0_ptr + c) : Q6_V_vzero();
            HVX_Vector v1 = row1_ptr ? hvx_vmemu(row1_ptr + c) : Q6_V_vzero();

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        for (; c < k_block; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (c < k_valid) {
                uint32_t rem_b = (k_valid - c) * sizeof(float);
                if (row0_ptr) v0 = hvx_vec_load_u(row0_ptr + c, rem_b);
                if (row1_ptr) v1 = hvx_vec_load_u(row1_ptr + c, rem_b);
            }

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
            uint32_t vtcm_start_row,
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
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

        struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];

        const float *row0_ptr = (const float *) ((const uint8_t *) src + mapping0.i2 * nb12);
        const float *row1_ptr = (const float *) ((const uint8_t *) src + mapping1.i2 * nb12);

        uint32_t c = 0;
        for (; c + 32 <= k_valid; c += 32) {
            HVX_Vector v0 = hvx_vmemu(row0_ptr + c);
            HVX_Vector v1 = hvx_vmemu(row1_ptr + c);
            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        for (; c < k_block; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (c < k_valid) {
                uint32_t rem_b = (k_valid - c) * sizeof(float);
                v0 = hvx_vec_load_u(row0_ptr + c, rem_b);
                v1 = hvx_vec_load_u(row1_ptr + c, rem_b);
            }

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }

    for (; r < n_rows_padded; r += 2) {
        uint32_t lr = vtcm_start_row + r;           // vtcm-local row
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

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
            HVX_Vector v0 = row0_ptr ? hvx_vmemu(row0_ptr + c) : Q6_V_vzero();
            HVX_Vector v1 = row1_ptr ? hvx_vmemu(row1_ptr + c) : Q6_V_vzero();

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
        for (; c < k_block; c += 32) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (c < k_valid) {
                uint32_t rem_b = (k_valid - c) * sizeof(float);
                if (row0_ptr) v0 = hvx_vec_load_u(row0_ptr + c, rem_b);
                if (row1_ptr) v1 = hvx_vec_load_u(row1_ptr + c, rem_b);
            }

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;
            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
            tile[r1 / 2]     = v_out;
        }
    }
}

static void transfer_activation_chunk_f16_to_f16_gathered(
            __fp16 *restrict vtcm_dst,
            const __fp16 *restrict src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
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
    const uint32_t n_tile_cols   = k_block / HTP_MM_HMX_TILE_N_COLS;

    uint32_t r = 0;

    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t lr = vtcm_start_row + r;
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;
        __fp16 * restrict tile_row = vtcm_dst + (size_t) r0 * n_tile_cols * HTP_MM_HMX_TILE_N_ELMS;

        struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];

        uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
        uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);

        const __fp16 *row0_ptr = (const __fp16 *) ((const uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        const __fp16 *row1_ptr = (const __fp16 *) ((const uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);

        uint32_t c = 0;
        for (; c + 64 <= k_valid; c += 64) {
            HVX_Vector     v0 = hvx_vmemu(row0_ptr + c);
            HVX_Vector     v1 = hvx_vmemu(row1_ptr + c);
            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
            HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

            tile0[r1 / 2] = Q6_V_lo_W(vp);
            tile1[r1 / 2] = Q6_V_hi_W(vp);
        }
        for (; c < k_block; c += 64) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (c < k_valid) {
                uint32_t rem_b = (k_valid - c) * sizeof(__fp16);
                v0 = hvx_vec_load_u(row0_ptr + c, rem_b);
                v1 = hvx_vec_load_u(row1_ptr + c, rem_b);
            }

            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
            tile0[r1 / 2] = Q6_V_lo_W(vp);

            if (c0 + 1 < n_tile_cols) {
                HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
                tile1[r1 / 2] = Q6_V_hi_W(vp);
            }
        }
    }

    for (; r < n_rows_padded; r += 2) {
        uint32_t lr = vtcm_start_row + r;
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;
        __fp16 * restrict tile_row = vtcm_dst + (size_t) r0 * n_tile_cols * HTP_MM_HMX_TILE_N_ELMS;

        const bool row0_valid = (start_row + r + 0) < cne1;
        const bool row1_valid = (start_row + r + 1) < cne1;

        const __fp16 *row0_ptr = NULL;
        const __fp16 *row1_ptr = NULL;

        if (row0_valid) {
            struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + (start_row + r + 0)];
            uint32_t i11_0 = fastmodulo(mapping0.i1, ne11, ne11_div);
            row0_ptr = (const __fp16 *) ((const uint8_t *) src + i11_0 * nb11 + mapping0.i2 * nb12);
        }
        if (row1_valid) {
            struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + (start_row + r + 1)];
            uint32_t i11_1 = fastmodulo(mapping1.i1, ne11, ne11_div);
            row1_ptr = (const __fp16 *) ((const uint8_t *) src + i11_1 * nb11 + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        for (; c + 64 <= k_valid; c += 64) {
            HVX_Vector     v0 = row0_ptr ? hvx_vmemu(row0_ptr + c) : Q6_V_vzero();
            HVX_Vector     v1 = row1_ptr ? hvx_vmemu(row1_ptr + c) : Q6_V_vzero();
            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
            HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

            tile0[r1 / 2] = Q6_V_lo_W(vp);
            tile1[r1 / 2] = Q6_V_hi_W(vp);
        }
        for (; c < k_block; c += 64) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (c < k_valid) {
                uint32_t rem_b = (k_valid - c) * sizeof(__fp16);
                if (row0_ptr) v0 = hvx_vec_load_u(row0_ptr + c, rem_b);
                if (row1_ptr) v1 = hvx_vec_load_u(row1_ptr + c, rem_b);
            }

            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
            tile0[r1 / 2] = Q6_V_lo_W(vp);

            if (c0 + 1 < n_tile_cols) {
                HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
                tile1[r1 / 2] = Q6_V_hi_W(vp);
            }
        }
    }
}

static void transfer_activation_chunk_f16_to_f16_gathered_flat(
            __fp16 *restrict vtcm_dst,
            const __fp16 *restrict src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
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
    const uint32_t n_tile_cols   = k_block / HTP_MM_HMX_TILE_N_COLS;

    uint32_t r = 0;

    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t lr = vtcm_start_row + r;
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;
        __fp16 * restrict tile_row = vtcm_dst + (size_t) r0 * n_tile_cols * HTP_MM_HMX_TILE_N_ELMS;

        struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];

        const __fp16 *row0_ptr = (const __fp16 *) ((const uint8_t *) src + mapping0.i2 * nb12);
        const __fp16 *row1_ptr = (const __fp16 *) ((const uint8_t *) src + mapping1.i2 * nb12);

        uint32_t c = 0;
        for (; c + 64 <= k_valid; c += 64) {
            HVX_Vector     v0 = hvx_vmemu(row0_ptr + c);
            HVX_Vector     v1 = hvx_vmemu(row1_ptr + c);
            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
            HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

            tile0[r1 / 2] = Q6_V_lo_W(vp);
            tile1[r1 / 2] = Q6_V_hi_W(vp);
        }
        for (; c < k_block; c += 64) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (c < k_valid) {
                uint32_t rem_b = (k_valid - c) * sizeof(__fp16);
                v0 = hvx_vec_load_u(row0_ptr + c, rem_b);
                v1 = hvx_vec_load_u(row1_ptr + c, rem_b);
            }

            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
            tile0[r1 / 2] = Q6_V_lo_W(vp);

            if (c0 + 1 < n_tile_cols) {
                HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
                tile1[r1 / 2] = Q6_V_hi_W(vp);
            }
        }
    }

    for (; r < n_rows_padded; r += 2) {
        uint32_t lr = vtcm_start_row + r;
        uint32_t r0 = lr / HTP_MM_HMX_TILE_N_ROWS;
        uint32_t r1 = lr % HTP_MM_HMX_TILE_N_ROWS;
        __fp16 * restrict tile_row = vtcm_dst + (size_t) r0 * n_tile_cols * HTP_MM_HMX_TILE_N_ELMS;

        const bool row0_valid = (start_row + r + 0) < cne1;
        const bool row1_valid = (start_row + r + 1) < cne1;

        const __fp16 *row0_ptr = NULL;
        const __fp16 *row1_ptr = NULL;

        if (row0_valid) {
            struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + (start_row + r + 0)];
            row0_ptr = (const __fp16 *) ((const uint8_t *) src + mapping0.i2 * nb12);
        }
        if (row1_valid) {
            struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + (start_row + r + 1)];
            row1_ptr = (const __fp16 *) ((const uint8_t *) src + mapping1.i2 * nb12);
        }

        uint32_t c = 0;
        for (; c + 64 <= k_valid; c += 64) {
            HVX_Vector     v0 = row0_ptr ? hvx_vmemu(row0_ptr + c) : Q6_V_vzero();
            HVX_Vector     v1 = row1_ptr ? hvx_vmemu(row1_ptr + c) : Q6_V_vzero();
            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
            HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);

            tile0[r1 / 2] = Q6_V_lo_W(vp);
            tile1[r1 / 2] = Q6_V_hi_W(vp);
        }
        for (; c < k_block; c += 64) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();
            if (c < k_valid) {
                uint32_t rem_b = (k_valid - c) * sizeof(__fp16);
                if (row0_ptr) v0 = hvx_vec_load_u(row0_ptr + c, rem_b);
                if (row1_ptr) v1 = hvx_vec_load_u(row1_ptr + c, rem_b);
            }

            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
            tile0[r1 / 2] = Q6_V_lo_W(vp);

            if (c0 + 1 < n_tile_cols) {
                HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
                tile1[r1 / 2] = Q6_V_hi_W(vp);
            }
        }
    }
}

static void transfer_output_chunk_fp16_to_fp32_scattered(
            float *restrict dst,
            const __fp16 *restrict vtcm_src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
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
        uint32_t     lr = vtcm_start_row + r;                 // vtcm-local row
        const size_t r0 = (lr / HTP_MM_HMX_TILE_N_ROWS);
        const size_t r1 = (lr % HTP_MM_HMX_TILE_N_ROWS) / 2;  // index of the row pair within the tile
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

            hvx_vmemu(output_row0 + c) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            if (output_row1) {
                hvx_vmemu(output_row1 + c) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
            }
        }
    }
}

static void transfer_output_chunk_fp16_to_fp16_scattered(
            __fp16 *restrict dst,
            const __fp16 *restrict vtcm_src,
            uint32_t start_row,
            uint32_t vtcm_start_row,
            uint32_t n_rows,
            uint32_t n_cols,
            const struct mmid_row_mapping *matrix_rows,
            uint32_t cur_a,
            uint32_t mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            uint32_t cne1) {
    assert(n_cols % HTP_MM_HMX_TILE_N_COLS == 0);
    const size_t tile_row_stride  = (n_cols / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
    const size_t n_cols_aligned64 = ((size_t) n_cols / 64) * 64;

    for (size_t r = 0; r < n_rows; r += 2) {
        uint32_t r_idx0 = start_row + r + 0;
        uint32_t r_idx1 = start_row + r + 1;
        uint32_t     lr = vtcm_start_row + r;
        const size_t r0 = (lr / HTP_MM_HMX_TILE_N_ROWS);
        const size_t r1 = (lr % HTP_MM_HMX_TILE_N_ROWS) / 2;
        const __fp16 *row_base = vtcm_src + r0 * tile_row_stride;

        if (r_idx0 >= cne1) break;

        struct mmid_row_mapping mapping0 = matrix_rows[cur_a * mapping_stride + r_idx0];
        __fp16 *output_row0 = (__fp16 *) ((uint8_t *) dst + mapping0.i1 * dst_nb1 + mapping0.i2 * dst_nb2);

        __fp16 *output_row1 = NULL;
        if (r_idx1 < cne1) {
            struct mmid_row_mapping mapping1 = matrix_rows[cur_a * mapping_stride + r_idx1];
            output_row1 = (__fp16 *) ((uint8_t *) dst + mapping1.i1 * dst_nb1 + mapping1.i2 * dst_nb2);
        }

        size_t c = 0;
        #pragma unroll(2)
        for (; c < n_cols_aligned64; c += 64) {
            const size_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            const __fp16 *tile0 = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            const __fp16 *tile1 = row_base + (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v_t0 = ((const HVX_Vector *) tile0)[r1];
            HVX_Vector v_t1 = ((const HVX_Vector *) tile1)[r1];
            HVX_VectorPair vp = Q6_W_vdeal_VVR(v_t1, v_t0, -2);

            hvx_vmemu(output_row0 + c) = Q6_V_lo_W(vp);
            if (output_row1) {
                hvx_vmemu(output_row1 + c) = Q6_V_hi_W(vp);
            }
        }

        if (c < (size_t) n_cols) {
            const size_t valid_c = (size_t) n_cols - c;
            const size_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
            const __fp16 *tile0 = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v_t0 = ((const HVX_Vector *) tile0)[r1];
            HVX_Vector v_t1 = Q6_V_vzero();
            HVX_VectorPair vp = Q6_W_vdeal_VVR(v_t1, v_t0, -2);

            hvx_vec_store_u(output_row0 + c, valid_c * sizeof(__fp16), Q6_V_lo_W(vp));
            if (output_row1) {
                hvx_vec_store_u(output_row1 + c, valid_c * sizeof(__fp16), Q6_V_hi_W(vp));
            }
        }
    }
}

// --- INT HMX (W8A8 SM and W8A16 :2x1) Track A kernels ---

typedef struct {
    uint8_t *             dst_wt_base;
    uint8_t *             dst_bias_base;
    float *               dst_scales_base;
    const uint8_t *       src_raw;
    uint32_t              n_col_tiles;
    uint32_t              n_k_tiles;
    uint32_t              n_groups;
    uint32_t              tiles_per_group;
    uint32_t              aligned_tile_size;
    int32_t               kernel_type;
    uint32_t              n_tot_units;
    uint32_t              n_units_per_task;
    uint32_t              n_tasks;
    struct fastdiv_values n_groups_div;
} int_weight_rescale_state_t;

static inline HVX_Vector hvx_vec_f32_to_qf32(HVX_Vector v) {
#if __HVX_ARCH__ >= 81
    return Q6_Vqf32_equals_Vsf(v);
#else
    return Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vzero());
#endif
}

// Convert two 32-float vectors (already scaled to [-32766, 32766]) directly to interleaved uint16 (even=vqf_0, odd=vqf_1) with +32768 offset.
static inline HVX_Vector hvx_vec_u16_zp32768_from_qf32_pair_shuff(HVX_Vector vqf_0, HVX_Vector vqf_1) {
    const HVX_Vector v_bias_u16 = Q6_V_vsplat_R(0x4B408000); // 12582912.0f + 32768.0f (1 ULP == 1.0f)
    HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(vqf_0, v_bias_u16));
    HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(vqf_1, v_bias_u16));
    return Q6_Vh_vshuffe_VhVh(y1, y0);
}

static inline void emit_int_hmx_scales_and_bias(
        HVX_Vector v_max_sf,
        float scale_const,
        HVX_Vector v_sum_w,
        HVX_Vector v_l1_w,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        bool is_w8a8) {

    // |sum_k (a - zp) * w_hat| < 128 * l1_w = (l1_w << 7) <= 32768 * 2^p
    HVX_Vector v_max_mag = Q6_Vw_vasl_VwR(v_l1_w, 7);
    HVX_Vector v_p       = Q6_Vw_vmax_VwVw(Q6_V_vzero(), Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(17), Q6_Vuw_vcl0_Vuw(v_max_mag)));
    HVX_Vector v_ib      = Q6_Vw_vsub_VwVw(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(32768), v_p), Q6_Vw_vasl_VwR(v_sum_w, 7));

    HVX_Vector v_dw_base       = hvx_vec_mul_f32_f32(v_max_sf, hvx_vec_splat_f32(scale_const));
    HVX_Vector v_pow2_sf       = Q6_Vsf_equals_Vw(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(1), v_p));
    *(HVX_Vector *) dst_scales = hvx_vec_mul_f32_f32(v_dw_base, v_pow2_sf);

    if (is_w8a8) {
        HVX_Vector v_exp_hi = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(16), v_p);
        HVX_Vector v_exp_lo = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(24), v_p);
        bias_out[0] = Q6_Vw_vasl_VwR(v_exp_hi, 10);
        bias_out[1] = v_ib;
        bias_out[2] = Q6_Vw_vasl_VwR(v_exp_lo, 10);
        bias_out[3] = v_ib;
    } else {
        HVX_Vector v_exp = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(16), v_p);
        bias_out[0] = Q6_Vw_vasl_VwR(v_exp, 10);
        bias_out[1] = v_ib;
    }
}

// Rescale one (ct, g) superblock of Q4_0 tiles into int8 weight.b tiles,
// compute sum_w and exact l1_w per output channel, and write the Bias Register bank(s) and D_w_eff.
static __attribute__((noinline)) void rescale_q4_0_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector mask_f0    = Q6_Vb_vsplat_R(0xF0);
    const HVX_Vector sign_80    = Q6_Vb_vsplat_R(0x80);
    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    const HVX_Vector v_ones_s8  = Q6_Vb_vsplat_R(1);
    const HVX_Vector v_one_f16  = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v12M_sf    = Q6_V_vsplat_R(0x4B400000); // 12582912.0f (1 ULP == 1.0f)

    // Pass 1: find per-OC max |d(oc, kt)| across the superblock
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector v_sc = Q6_V_vand_VV(hvx_vmem(tile_src + 512), v_abs_mask);
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, v_sc);
    }

    HVX_Vector v_max_sf    = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_max_sc), v_one_f16)));
    HVX_VectorPred q_pos   = Q6_Q_vcmp_gt_VsfVsf(v_max_sf, hvx_vec_splat_f32(1e-30f));
    HVX_Vector v_inv_raw   = hvx_vec_inverse_f32(v_max_sf);
    HVX_Vector v_inv127_sf = Q6_V_vmux_QVV(q_pos, hvx_vec_mul_f32_f32(v_inv_raw, hvx_vec_splat_f32(127.0f)), Q6_V_vzero());
    HVX_Vector v_inv127_qf = hvx_vec_f32_to_qf32(v_inv127_sf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    // Pass 2: rescale nibbles into int8 weight.b tiles and accumulate sum_w and exact l1_w
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);

        HVX_Vector v_sc       = hvx_vmem(tile_src + 512);
        HVX_Vector v_sc_qf    = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_sc), v_one_f16));
        HVX_Vector v_m_biased = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(v_sc_qf, v_inv127_qf), v12M_sf));
        HVX_Vector v_m_h      = Q6_Vh_vpacke_VwVw(v_m_biased, v_m_biased);
        HVX_Vector v_m_b      = Q6_Vb_vshuff_Vb(Q6_Vb_vpacke_VhVh(v_m_h, v_m_h));

        for (int i = 0; i < 4; ++i) {
            HVX_Vector vq   = hvx_vmem(tile_src + i * 128);
            HVX_Vector v_lo = Q6_V_vxor_VV(Q6_V_vand_VV(Q6_Vh_vasl_VhR(vq, 4), mask_f0), sign_80);
            HVX_Vector v_hi = Q6_V_vxor_VV(Q6_V_vand_VV(vq, mask_f0), sign_80);

            HVX_VectorPair vp_shuf    = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
            HVX_VectorPair vp_p01     = Q6_Wh_vmpy_VbVb(Q6_V_lo_W(vp_shuf), v_m_b);
            HVX_Vector     v_neg1_h   = Q6_Vh_vsplat_R(-1);
            HVX_VectorPair vp_tie_adj = Q6_W_vcombine_VV(v_neg1_h, v_neg1_h);
            HVX_VectorPair vp_p23     = Q6_Wh_vmpyacc_WhVbVb(vp_tie_adj, Q6_V_hi_W(vp_shuf), v_m_b);

#if __HVX_ARCH__ >= 79
            HVX_Vector v_b_01 = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(vp_p01), Q6_V_lo_W(vp_p01), 7);
            HVX_Vector v_b_23 = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(vp_p23), Q6_V_lo_W(vp_p23), 7);
#else
            HVX_Vector v_b_01 = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(Q6_V_hi_W(vp_p01), 1),
                                                      Q6_Vh_vasl_VhR(Q6_V_lo_W(vp_p01), 1));
            HVX_Vector v_b_23 = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(Q6_V_hi_W(vp_p23), 1),
                                                      Q6_Vh_vasl_VhR(Q6_V_lo_W(vp_p23), 1));
#endif
            HVX_Vector v_W0 = Q6_Vh_vshuff_Vh(v_b_01);
            HVX_Vector v_W1 = Q6_Vh_vshuff_Vh(v_b_23);

            tile_dst[2 * i + 0] = v_W0;
            tile_dst[2 * i + 1] = v_W1;

            HVX_Vector v_pair_sum = Q6_Vw_vrmpyacc_VwVbVb(Q6_Vw_vrmpy_VbVb(v_W0, v_ones_s8), v_W1, v_ones_s8);
            v_sum_w               = Q6_Vw_vadd_VwVw(v_sum_w, v_pair_sum);

            HVX_Vector v_pair_l1 = Q6_Vw_vrmpyacc_VwVubVb(Q6_Vw_vrmpy_VubVb(Q6_Vb_vabs_Vb(v_W0), v_ones_s8), Q6_Vb_vabs_Vb(v_W1), v_ones_s8);
            v_l1_w               = Q6_Vw_vadd_VwVw(v_l1_w, v_pair_l1);
        }
    }

    const float scale_const = is_w8a8 ? (8.0f / 127.0f) : (2048.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static void rescale_q4_0_weight_to_int8_task(
        const int_weight_rescale_state_t * state,
        uint32_t start_unit,
        uint32_t end_unit) {

    const bool is_w8a8 = htp_mm_kernel_is_hmx_w8a8(state->kernel_type);
    const size_t bias_stride = is_w8a8 ? (2 * HTP_MM_HMX_BIAS_BANK_SIZE) : HTP_MM_HMX_BIAS_BANK_SIZE;

    for (uint32_t u = start_unit; u < end_unit; ++u) {
        const uint32_t ct = fastdiv(u, &state->n_groups_div);
        const uint32_t g  = u - ct * state->n_groups;

        const size_t kt_start = (size_t) ct * state->n_k_tiles + (size_t) g * state->tiles_per_group;
        const uint8_t * src_base = state->src_raw + kt_start * state->aligned_tile_size;
        uint8_t * dst_wt         = state->dst_wt_base + kt_start * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
        float * dst_scales       = state->dst_scales_base + (size_t) u * 32;
        HVX_Vector * bias_out    = (HVX_Vector *) (state->dst_bias_base + (size_t) u * bias_stride);

        rescale_q4_0_weight_superblock(
            src_base, dst_wt, dst_scales, bias_out,
            state->tiles_per_group, state->aligned_tile_size, is_w8a8);
    }
}

static inline void quantize_f16_quad_step_to_u8_sm(
        uint8_t * restrict tile_row_base,
        uint32_t kt,
        uint32_t rg,
        HVX_Vector vh0,
        HVX_Vector vh1,
        HVX_Vector vh2,
        HVX_Vector vh3,
        HVX_Vector v_inv0_qf,
        HVX_Vector v_inv1_qf,
        HVX_Vector v_inv2_qf,
        HVX_Vector v_inv3_qf,
        HVX_Vector v_one_f16,
        HVX_Vector v_bias_u8,
        HVX_Vector v_min_u8,
        HVX_Vector v_max_u8) {

    HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh0), v_one_f16);
    HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh1), v_one_f16);
    HVX_VectorPair vp2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh2), v_one_f16);
    HVX_VectorPair vp3 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh3), v_one_f16);

    {
        HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp0), v_inv0_qf), v_bias_u8));
        HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp1), v_inv1_qf), v_bias_u8));
        HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp2), v_inv2_qf), v_bias_u8));
        HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp3), v_inv3_qf), v_bias_u8));
        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
        ((HVX_Vector *) (tile_row_base + (size_t) (kt + 0) * HTP_MM_HMX_TILE_SIZE))[rg] = Q6_Vb_vshuffe_VbVb(vh13, vh02);
    }
    {
        HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp0), v_inv0_qf), v_bias_u8));
        HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp1), v_inv1_qf), v_bias_u8));
        HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp2), v_inv2_qf), v_bias_u8));
        HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp3), v_inv3_qf), v_bias_u8));
        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
        ((HVX_Vector *) (tile_row_base + (size_t) (kt + 1) * HTP_MM_HMX_TILE_SIZE))[rg] = Q6_Vb_vshuffe_VbVb(vh13, vh02);
    }
}

static inline void quantize_f16_pair_step_to_u16_sm(
        uint16_t * restrict tile_row_base,
        uint32_t kt,
        uint32_t r1,
        HVX_Vector vh0,
        HVX_Vector vh1,
        HVX_Vector v_inv0_qf,
        HVX_Vector v_inv1_qf,
        HVX_Vector v_one_f16) {

    HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh0), v_one_f16);
    HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh1), v_one_f16);

    ((HVX_Vector *) (tile_row_base + (size_t) (kt + 0) * HTP_MM_HMX_TILE_N_ELMS))[r1] =
        hvx_vec_u16_zp32768_from_qf32_pair_shuff(
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp0), v_inv0_qf),
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp1), v_inv1_qf));
    ((HVX_Vector *) (tile_row_base + (size_t) (kt + 1) * HTP_MM_HMX_TILE_N_ELMS))[r1] =
        hvx_vec_u16_zp32768_from_qf32_pair_shuff(
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp0), v_inv0_qf),
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp1), v_inv1_qf));
}

// Quantize one superblock g (k_group columns, multiple of 32) of a 4-row FP32 group into W8A8 SM 64x32 tiles.
static __attribute__((noinline)) void quantize_act_superblock_quad_f32_to_u8_sm(
        uint8_t * restrict tile_row_base,
        float * restrict d_a_out,
        const float * restrict p0,
        const float * restrict p1,
        const float * restrict p2,
        const float * restrict p3,
        uint32_t rg,
        uint32_t k_start,
        uint32_t k_group,
        uint32_t k_valid,
        uint32_t valid_rows) {

    assert(hex_is_aligned(p0 + k_start, 128));
    assert(hex_is_aligned(p1 + k_start, 128));
    assert(hex_is_aligned(p2 + k_start, 128));
    assert(hex_is_aligned(p3 + k_start, 128));

    const uint32_t k_end      = k_start + k_group;
    const uint32_t c_end_full = (k_end <= k_valid) ? k_end
                              : ((k_valid > k_start) ? (k_start + ((k_valid - k_start) & ~31u)) : k_start);
    HVX_Vector vmax0 = Q6_V_vzero();
    HVX_Vector vmax1 = Q6_V_vzero();
    HVX_Vector vmax2 = Q6_V_vzero();
    HVX_Vector vmax3 = Q6_V_vzero();

    for (uint32_t c = k_start; c < c_end_full; c += 32) {
        HVX_Vector vx0 = hvx_vmem(p0 + c);
        HVX_Vector vx1 = hvx_vmem(p1 + c);
        HVX_Vector vx2 = hvx_vmem(p2 + c);
        HVX_Vector vx3 = hvx_vmem(p3 + c);
        vmax0 = Q6_Vsf_vmax_VsfVsf(vmax0, hvx_vec_abs_f32(vx0));
        vmax1 = Q6_Vsf_vmax_VsfVsf(vmax1, hvx_vec_abs_f32(vx1));
        vmax2 = Q6_Vsf_vmax_VsfVsf(vmax2, hvx_vec_abs_f32(vx2));
        vmax3 = Q6_Vsf_vmax_VsfVsf(vmax3, hvx_vec_abs_f32(vx3));
    }
    if (c_end_full < k_end && c_end_full < k_valid) {
        const uint32_t c    = c_end_full;
        const uint32_t rem  = k_valid - c;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
        HVX_Vector vx0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
        HVX_Vector vx1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        HVX_Vector vx2 = Q6_V_vmux_QVV(mask, hvx_vmem(p2 + c), Q6_V_vzero());
        HVX_Vector vx3 = Q6_V_vmux_QVV(mask, hvx_vmem(p3 + c), Q6_V_vzero());
        vmax0 = Q6_Vsf_vmax_VsfVsf(vmax0, hvx_vec_abs_f32(vx0));
        vmax1 = Q6_Vsf_vmax_VsfVsf(vmax1, hvx_vec_abs_f32(vx1));
        vmax2 = Q6_Vsf_vmax_VsfVsf(vmax2, hvx_vec_abs_f32(vx2));
        vmax3 = Q6_Vsf_vmax_VsfVsf(vmax3, hvx_vec_abs_f32(vx3));
    }

    vmax0 = (valid_rows > 0) ? hvx_vec_reduce_max_f32(vmax0) : Q6_V_vzero();
    vmax1 = (valid_rows > 1) ? hvx_vec_reduce_max_f32(vmax1) : Q6_V_vzero();
    vmax2 = (valid_rows > 2) ? hvx_vec_reduce_max_f32(vmax2) : Q6_V_vzero();
    vmax3 = (valid_rows > 3) ? hvx_vec_reduce_max_f32(vmax3) : Q6_V_vzero();

    const HVX_Vector v_inv_127 = hvx_vec_splat_f32(1.0f / 127.0f);
    const HVX_Vector v_127     = hvx_vec_splat_f32(127.0f);
    const HVX_Vector v_eps     = hvx_vec_splat_f32(1e-30f);

    HVX_Vector vd0 = hvx_vec_mul_f32_f32(vmax0, v_inv_127);
    HVX_Vector vd1 = hvx_vec_mul_f32_f32(vmax1, v_inv_127);
    HVX_Vector vd2 = hvx_vec_mul_f32_f32(vmax2, v_inv_127);
    HVX_Vector vd3 = hvx_vec_mul_f32_f32(vmax3, v_inv_127);
    HVX_Vector v_d02   = Q6_V_lo_W(Q6_W_vshuff_VVR(vd2, vd0, -4));
    HVX_Vector v_d13   = Q6_V_lo_W(Q6_W_vshuff_VVR(vd3, vd1, -4));
    HVX_Vector v_d0123 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_d13, v_d02, -4));
    hvx_vec_store_u(d_a_out, 4 * sizeof(float), v_d0123);

    HVX_Vector v_inv0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax0, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax0), v_127), Q6_V_vzero());
    HVX_Vector v_inv1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax1, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax1), v_127), Q6_V_vzero());
    HVX_Vector v_inv2 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax2, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax2), v_127), Q6_V_vzero());
    HVX_Vector v_inv3 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax3, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax3), v_127), Q6_V_vzero());

    const HVX_Vector v_bias_u8 = Q6_V_vsplat_R(0x4B400080); // 12582912.0f + 128.0f
    const HVX_Vector v_min_u8  = Q6_Vh_vsplat_R(1);
    const HVX_Vector v_max_u8  = Q6_Vh_vsplat_R(255);

    for (uint32_t c = k_start; c < c_end_full; c += 32) {
        HVX_Vector vx0 = hvx_vmem(p0 + c);
        HVX_Vector vx1 = hvx_vmem(p1 + c);
        HVX_Vector vx2 = hvx_vmem(p2 + c);
        HVX_Vector vx3 = hvx_vmem(p3 + c);

        HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0), v_bias_u8));
        HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1), v_bias_u8));
        HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx2, v_inv2), v_bias_u8));
        HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx3, v_inv3), v_bias_u8));

        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
        HVX_Vector v_u8 = Q6_Vb_vshuffe_VbVb(vh13, vh02);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_SIZE))[rg] = v_u8;
    }
    for (uint32_t c = c_end_full; c < k_end; c += 32) {
        HVX_Vector vx0 = Q6_V_vzero();
        HVX_Vector vx1 = Q6_V_vzero();
        HVX_Vector vx2 = Q6_V_vzero();
        HVX_Vector vx3 = Q6_V_vzero();
        if (c < k_valid) {
            uint32_t rem        = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
            vx0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
            vx1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
            vx2 = Q6_V_vmux_QVV(mask, hvx_vmem(p2 + c), Q6_V_vzero());
            vx3 = Q6_V_vmux_QVV(mask, hvx_vmem(p3 + c), Q6_V_vzero());
        }
        HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0), v_bias_u8));
        HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1), v_bias_u8));
        HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx2, v_inv2), v_bias_u8));
        HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx3, v_inv3), v_bias_u8));

        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
        HVX_Vector v_u8 = Q6_Vb_vshuffe_VbVb(vh13, vh02);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_SIZE))[rg] = v_u8;
    }
}

// Quantize one superblock g (k_group columns, multiple of 128) of a 4-row FP16 group into W8A8 SM 64x32 tiles.
static __attribute__((noinline)) void quantize_act_superblock_quad_f16_to_u8_sm(
        uint8_t * restrict tile_row_base,
        float * restrict d_a_out,
        const __fp16 * restrict p0,
        const __fp16 * restrict p1,
        const __fp16 * restrict p2,
        const __fp16 * restrict p3,
        uint32_t rg,
        uint32_t k_start,
        uint32_t k_group,
        uint32_t k_valid,
        uint32_t valid_rows) {

    assert(hex_is_aligned(p0 + k_start, 128));
    assert(hex_is_aligned(p1 + k_start, 128));
    assert(hex_is_aligned(p2 + k_start, 128));
    assert(hex_is_aligned(p3 + k_start, 128));

    const uint32_t k_end      = k_start + k_group;
    const uint32_t c_end_full = (k_end <= k_valid) ? k_end
                              : ((k_valid > k_start) ? (k_start + ((k_valid - k_start) & ~63u)) : k_start);
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_abs_f16 = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector vmax_hf0 = Q6_V_vzero();
    HVX_Vector vmax_hf1 = Q6_V_vzero();
    HVX_Vector vmax_hf2 = Q6_V_vzero();
    HVX_Vector vmax_hf3 = Q6_V_vzero();

    for (uint32_t c = k_start; c < c_end_full; c += 64) {
        HVX_Vector vh0 = hvx_vmem(p0 + c);
        HVX_Vector vh1 = hvx_vmem(p1 + c);
        HVX_Vector vh2 = hvx_vmem(p2 + c);
        HVX_Vector vh3 = hvx_vmem(p3 + c);
        vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vand_VV(vh0, v_abs_f16));
        vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vand_VV(vh1, v_abs_f16));
        vmax_hf2 = Q6_Vhf_vmax_VhfVhf(vmax_hf2, Q6_V_vand_VV(vh2, v_abs_f16));
        vmax_hf3 = Q6_Vhf_vmax_VhfVhf(vmax_hf3, Q6_V_vand_VV(vh3, v_abs_f16));
    }
    if (c_end_full < k_end && c_end_full < k_valid) {
        const uint32_t c     = c_end_full;
        const uint32_t rem_b = (k_valid - c) * sizeof(__fp16);
        HVX_VectorPred mask  = Q6_Q_vsetq2_R(rem_b);
        HVX_Vector vh0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
        HVX_Vector vh1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        HVX_Vector vh2 = Q6_V_vmux_QVV(mask, hvx_vmem(p2 + c), Q6_V_vzero());
        HVX_Vector vh3 = Q6_V_vmux_QVV(mask, hvx_vmem(p3 + c), Q6_V_vzero());
        vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vand_VV(vh0, v_abs_f16));
        vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vand_VV(vh1, v_abs_f16));
        vmax_hf2 = Q6_Vhf_vmax_VhfVhf(vmax_hf2, Q6_V_vand_VV(vh2, v_abs_f16));
        vmax_hf3 = Q6_Vhf_vmax_VhfVhf(vmax_hf3, Q6_V_vand_VV(vh3, v_abs_f16));
    }

    vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vror_VR(vmax_hf0, 64));
    vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vror_VR(vmax_hf1, 64));
    vmax_hf2 = Q6_Vhf_vmax_VhfVhf(vmax_hf2, Q6_V_vror_VR(vmax_hf2, 64));
    vmax_hf3 = Q6_Vhf_vmax_VhfVhf(vmax_hf3, Q6_V_vror_VR(vmax_hf3, 64));

    HVX_Vector vmax0 = (valid_rows > 0) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf0), v_one_f16)))) : Q6_V_vzero();
    HVX_Vector vmax1 = (valid_rows > 1) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf1), v_one_f16)))) : Q6_V_vzero();
    HVX_Vector vmax2 = (valid_rows > 2) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf2), v_one_f16)))) : Q6_V_vzero();
    HVX_Vector vmax3 = (valid_rows > 3) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf3), v_one_f16)))) : Q6_V_vzero();

    const HVX_Vector v_inv_127 = hvx_vec_splat_f32(1.0f / 127.0f);
    const HVX_Vector v_127     = hvx_vec_splat_f32(127.0f);
    const HVX_Vector v_eps     = hvx_vec_splat_f32(1e-30f);

    HVX_Vector vd0 = hvx_vec_mul_f32_f32(vmax0, v_inv_127);
    HVX_Vector vd1 = hvx_vec_mul_f32_f32(vmax1, v_inv_127);
    HVX_Vector vd2 = hvx_vec_mul_f32_f32(vmax2, v_inv_127);
    HVX_Vector vd3 = hvx_vec_mul_f32_f32(vmax3, v_inv_127);
    HVX_Vector v_d02   = Q6_V_lo_W(Q6_W_vshuff_VVR(vd2, vd0, -4));
    HVX_Vector v_d13   = Q6_V_lo_W(Q6_W_vshuff_VVR(vd3, vd1, -4));
    HVX_Vector v_d0123 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_d13, v_d02, -4));
    hvx_vec_store_u(d_a_out, 4 * sizeof(float), v_d0123);

    HVX_Vector v_inv0_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax0, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax0), v_127), Q6_V_vzero()));
    HVX_Vector v_inv1_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax1, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax1), v_127), Q6_V_vzero()));
    HVX_Vector v_inv2_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax2, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax2), v_127), Q6_V_vzero()));
    HVX_Vector v_inv3_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax3, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax3), v_127), Q6_V_vzero()));

    const HVX_Vector v_bias_u8 = Q6_V_vsplat_R(0x4B400080);
    const HVX_Vector v_min_u8  = Q6_Vh_vsplat_R(1);
    const HVX_Vector v_max_u8  = Q6_Vh_vsplat_R(255);

    for (uint32_t c = k_start; c < c_end_full; c += 64) {
        HVX_Vector vh0 = hvx_vmem(p0 + c);
        HVX_Vector vh1 = hvx_vmem(p1 + c);
        HVX_Vector vh2 = hvx_vmem(p2 + c);
        HVX_Vector vh3 = hvx_vmem(p3 + c);
        quantize_f16_quad_step_to_u8_sm(tile_row_base, c >> 5, rg, vh0, vh1, vh2, vh3, v_inv0_qf, v_inv1_qf, v_inv2_qf, v_inv3_qf, v_one_f16, v_bias_u8, v_min_u8, v_max_u8);
    }
    for (uint32_t c = c_end_full; c < k_end; c += 64) {
        HVX_Vector vh0 = Q6_V_vzero();
        HVX_Vector vh1 = Q6_V_vzero();
        HVX_Vector vh2 = Q6_V_vzero();
        HVX_Vector vh3 = Q6_V_vzero();
        if (c < k_valid) {
            uint32_t rem_b      = (k_valid - c) * sizeof(__fp16);
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem_b);
            vh0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
            vh1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
            vh2 = Q6_V_vmux_QVV(mask, hvx_vmem(p2 + c), Q6_V_vzero());
            vh3 = Q6_V_vmux_QVV(mask, hvx_vmem(p3 + c), Q6_V_vzero());
        }
        quantize_f16_quad_step_to_u8_sm(tile_row_base, c >> 5, rg, vh0, vh1, vh2, vh3, v_inv0_qf, v_inv1_qf, v_inv2_qf, v_inv3_qf, v_one_f16, v_bias_u8, v_min_u8, v_max_u8);
    }
}

// Quantize one superblock g (k_group columns, multiple of 32) of a 2-row FP32 pair into W8A16 SM 32x32 uint16 tiles.
static __attribute__((noinline)) void quantize_act_superblock_pair_f32_to_u16_sm(
        uint16_t * restrict tile_row_base,
        float * restrict d_a_out,
        const float * restrict p0,
        const float * restrict p1,
        uint32_t r1,
        uint32_t k_start,
        uint32_t k_group,
        uint32_t k_valid,
        uint32_t valid_rows) {

    assert(hex_is_aligned(p0 + k_start, 128));
    assert(hex_is_aligned(p1 + k_start, 128));

    const uint32_t k_end      = k_start + k_group;
    const uint32_t c_end_full = (k_end <= k_valid) ? k_end
                              : ((k_valid > k_start) ? (k_start + ((k_valid - k_start) & ~31u)) : k_start);
    HVX_Vector vmax0 = Q6_V_vzero();
    HVX_Vector vmax1 = Q6_V_vzero();

    for (uint32_t c = k_start; c < c_end_full; c += 32) {
        HVX_Vector vx0 = hvx_vmem(p0 + c);
        HVX_Vector vx1 = hvx_vmem(p1 + c);
        vmax0 = Q6_Vsf_vmax_VsfVsf(vmax0, hvx_vec_abs_f32(vx0));
        vmax1 = Q6_Vsf_vmax_VsfVsf(vmax1, hvx_vec_abs_f32(vx1));
    }
    if (c_end_full < k_end && c_end_full < k_valid) {
        const uint32_t c    = c_end_full;
        const uint32_t rem  = k_valid - c;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
        HVX_Vector vx0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
        HVX_Vector vx1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        vmax0 = Q6_Vsf_vmax_VsfVsf(vmax0, hvx_vec_abs_f32(vx0));
        vmax1 = Q6_Vsf_vmax_VsfVsf(vmax1, hvx_vec_abs_f32(vx1));
    }

    vmax0 = (valid_rows > 0) ? hvx_vec_reduce_max_f32(vmax0) : Q6_V_vzero();
    vmax1 = (valid_rows > 1) ? hvx_vec_reduce_max_f32(vmax1) : Q6_V_vzero();

    const HVX_Vector v_inv_32766 = hvx_vec_splat_f32(1.0f / 32766.0f);
    const HVX_Vector v_32766     = hvx_vec_splat_f32(32766.0f);
    const HVX_Vector v_eps       = hvx_vec_splat_f32(1e-30f);

    HVX_Vector vd0   = hvx_vec_mul_f32_f32(vmax0, v_inv_32766);
    HVX_Vector vd1   = hvx_vec_mul_f32_f32(vmax1, v_inv_32766);
    HVX_Vector v_d01 = Q6_V_lo_W(Q6_W_vshuff_VVR(vd1, vd0, -4));
    hvx_vec_store_u(d_a_out, 2 * sizeof(float), v_d01);

    HVX_Vector v_inv0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax0, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax0), v_32766), Q6_V_vzero());
    HVX_Vector v_inv1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax1, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax1), v_32766), Q6_V_vzero());

    for (uint32_t c = k_start; c < c_end_full; c += 32) {
        HVX_Vector vx0   = hvx_vmem(p0 + c);
        HVX_Vector vx1   = hvx_vmem(p1 + c);
        HVX_Vector vqf0  = Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0);
        HVX_Vector vqf1  = Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1);
        HVX_Vector v_u16 = hvx_vec_u16_zp32768_from_qf32_pair_shuff(vqf0, vqf1);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_N_ELMS))[r1] = v_u16;
    }
    for (uint32_t c = c_end_full; c < k_end; c += 32) {
        HVX_Vector vx0 = Q6_V_vzero();
        HVX_Vector vx1 = Q6_V_vzero();
        if (c < k_valid) {
            uint32_t rem        = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
            vx0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
            vx1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        }
        HVX_Vector vqf0  = Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0);
        HVX_Vector vqf1  = Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1);
        HVX_Vector v_u16 = hvx_vec_u16_zp32768_from_qf32_pair_shuff(vqf0, vqf1);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_N_ELMS))[r1] = v_u16;
    }
}

// Quantize one superblock g (k_group columns, multiple of 128) of a 2-row FP16 pair into W8A16 SM 32x32 uint16 tiles.
static __attribute__((noinline)) void quantize_act_superblock_pair_f16_to_u16_sm(
        uint16_t * restrict tile_row_base,
        float * restrict d_a_out,
        const __fp16 * restrict p0,
        const __fp16 * restrict p1,
        uint32_t r1,
        uint32_t k_start,
        uint32_t k_group,
        uint32_t k_valid,
        uint32_t valid_rows) {

    assert(hex_is_aligned(p0 + k_start, 128));
    assert(hex_is_aligned(p1 + k_start, 128));

    const uint32_t k_end          = k_start + k_group;
    const uint32_t c_end_full     = (k_end <= k_valid) ? k_end
                                  : ((k_valid > k_start) ? (k_start + ((k_valid - k_start) & ~63u)) : k_start);
    const HVX_Vector v_one_f16    = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_abs_f16    = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector vmax_hf0 = Q6_V_vzero();
    HVX_Vector vmax_hf1 = Q6_V_vzero();

    for (uint32_t c = k_start; c < c_end_full; c += 64) {
        HVX_Vector vh0 = hvx_vmem(p0 + c);
        HVX_Vector vh1 = hvx_vmem(p1 + c);
        vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vand_VV(vh0, v_abs_f16));
        vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vand_VV(vh1, v_abs_f16));
    }
    if (c_end_full < k_end && c_end_full < k_valid) {
        const uint32_t c     = c_end_full;
        const uint32_t rem_b = (k_valid - c) * sizeof(__fp16);
        HVX_VectorPred mask  = Q6_Q_vsetq2_R(rem_b);
        HVX_Vector vh0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
        HVX_Vector vh1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vand_VV(vh0, v_abs_f16));
        vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vand_VV(vh1, v_abs_f16));
    }

    vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vror_VR(vmax_hf0, 64));
    vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vror_VR(vmax_hf1, 64));

    HVX_Vector vmax0 = (valid_rows > 0) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf0), v_one_f16)))) : Q6_V_vzero();
    HVX_Vector vmax1 = (valid_rows > 1) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf1), v_one_f16)))) : Q6_V_vzero();

    const HVX_Vector v_inv_32766 = hvx_vec_splat_f32(1.0f / 32766.0f);
    const HVX_Vector v_32766     = hvx_vec_splat_f32(32766.0f);
    const HVX_Vector v_eps       = hvx_vec_splat_f32(1e-30f);

    HVX_Vector vd0   = hvx_vec_mul_f32_f32(vmax0, v_inv_32766);
    HVX_Vector vd1   = hvx_vec_mul_f32_f32(vmax1, v_inv_32766);
    HVX_Vector v_d01 = Q6_V_lo_W(Q6_W_vshuff_VVR(vd1, vd0, -4));
    hvx_vec_store_u(d_a_out, 2 * sizeof(float), v_d01);

    HVX_Vector v_inv0_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax0, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax0), v_32766), Q6_V_vzero()));
    HVX_Vector v_inv1_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax1, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax1), v_32766), Q6_V_vzero()));

    for (uint32_t c = k_start; c < c_end_full; c += 64) {
        HVX_Vector vh0 = hvx_vmem(p0 + c);
        HVX_Vector vh1 = hvx_vmem(p1 + c);
        quantize_f16_pair_step_to_u16_sm(tile_row_base, c >> 5, r1, vh0, vh1, v_inv0_qf, v_inv1_qf, v_one_f16);
    }
    for (uint32_t c = c_end_full; c < k_end; c += 64) {
        HVX_Vector vh0 = Q6_V_vzero();
        HVX_Vector vh1 = Q6_V_vzero();
        if (c < k_valid) {
            uint32_t rem_b      = (k_valid - c) * sizeof(__fp16);
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem_b);
            vh0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
            vh1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        }
        quantize_f16_pair_step_to_u16_sm(tile_row_base, c >> 5, r1, vh0, vh1, v_inv0_qf, v_inv1_qf, v_one_f16);
    }
}

// Core HMX W8A8 SM dot-product kernel over all superblocks g = 0..n_groups-1.
static void core_dot_chunk_w8a8_sm(
        uint8_t * restrict vtcm_out,
        const uint8_t * restrict vtcm_act,
        const uint8_t * restrict vtcm_weight_tiles,
        const uint8_t * restrict vtcm_weight_bias,
        uint32_t n_row_tiles_64,
        uint32_t n_col_tiles,
        uint32_t n_k_tiles,
        uint32_t n_groups,
        uint32_t tiles_per_group) {

    assert(hex_is_aligned(vtcm_out, HTP_MM_HMX_TILE_SIZE));
    const uint32_t cvt_ctrl_hi = hmx_cvt_ctrl(0, 0, 0, 1);
    const uint32_t cvt_ctrl_lo = hmx_cvt_ctrl(1, 0, 1, 0);
    const uint32_t out_cm_rng  = hmx_out_cm_range();

    for (uint32_t g = 0; g < n_groups; ++g) {
        const uint32_t kt_base = g * tiles_per_group;

        for (uint32_t j = 0; j < n_col_tiles; ++j) {
            const uint8_t * wt_base = vtcm_weight_tiles + ((size_t) j * n_k_tiles + kt_base) * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
            const uint8_t * bias_hi = vtcm_weight_bias  + (((size_t) j * n_groups + g) * 2 + 0) * HTP_MM_HMX_BIAS_BANK_SIZE;
            const uint8_t * bias_lo = vtcm_weight_bias  + (((size_t) j * n_groups + g) * 2 + 1) * HTP_MM_HMX_BIAS_BANK_SIZE;
            hmx_load_bias_bank(bias_hi, 0);
            hmx_load_bias_bank(bias_lo, 1);

            for (uint32_t i = 0; i < n_row_tiles_64; ++i) {
                const uint8_t * act_base = vtcm_act + ((size_t) i * n_k_tiles + kt_base) * HTP_MM_HMX_TILE_SIZE;

                const size_t out_idx = (((size_t) g * n_row_tiles_64 + i) * n_col_tiles + j) * 2;
                uint8_t * out_hi = vtcm_out + (out_idx + 0) * HTP_MM_HMX_TILE_SIZE;
                uint8_t * out_lo = vtcm_out + (out_idx + 1) * HTP_MM_HMX_TILE_SIZE;

                asm volatile(HMX_CLRACC_INT());

                uint32_t rem = tiles_per_group;
                const uint8_t * a_ptr = act_base;
                const uint8_t * w_ptr = wt_base;
                while (rem > 0) {
                    const uint32_t step  = (rem > 64) ? 64 : rem;
                    const uint32_t a_rng = hmx_act_sm_range(step);
                    const uint32_t w_rng = hmx_wt_b_range(step * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
                    asm volatile(HMX_LOAD_MPY_DEEP_U8_S8_SM("%0", "%1", "%2", "%3")
                                 : : "r"(a_ptr), "r"(a_rng), "r"(w_ptr), "r"(w_rng) : "memory");
                    a_ptr += step * HTP_MM_HMX_TILE_SIZE;
                    w_ptr += step * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
                    rem   -= step;
                }

                asm volatile(HMX_CVT_UB_ACC("%0") : : "r"(cvt_ctrl_hi) : "memory");
                asm volatile(HMX_STORE_CVT_CM("%0", "%1") : : "r"(out_hi), "r"(out_cm_rng) : "memory");

                asm volatile(HMX_CVT_UB_ACC("%0") : : "r"(cvt_ctrl_lo) : "memory");
                asm volatile(HMX_STORE_CVT_CM("%0", "%1") : : "r"(out_lo), "r"(out_cm_rng) : "memory");
            }
        }
    }
}

// Core HMX W8A16 :2x1 dot-product kernel over all superblocks g = 0..n_groups-1.
static void core_dot_chunk_w8a16_2x1(
        uint16_t * restrict vtcm_out,
        const uint16_t * restrict vtcm_act,
        const uint8_t * restrict vtcm_weight_tiles,
        const uint8_t * restrict vtcm_weight_bias,
        uint32_t n_row_tiles_32,
        uint32_t n_col_tiles,
        uint32_t n_k_tiles,
        uint32_t n_groups,
        uint32_t tiles_per_group) {

    const uint32_t cvt_ctrl_16 = hmx_cvt_ctrl(0, 0, 0, 0);
    const uint32_t out_sm_rng1 = hmx_act_sm_range(1);

    for (uint32_t g = 0; g < n_groups; ++g) {
        const uint32_t kt_base = g * tiles_per_group;

        for (uint32_t j = 0; j < n_col_tiles; ++j) {
            const uint8_t * wt_base  = vtcm_weight_tiles + ((size_t) j * n_k_tiles + kt_base) * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
            const uint8_t * bias_ptr = vtcm_weight_bias  + ((size_t) j * n_groups + g) * HTP_MM_HMX_BIAS_BANK_SIZE;
            hmx_load_bias_bank(bias_ptr, 0);

            for (uint32_t i = 0; i < n_row_tiles_32; ++i) {
                const uint8_t * act_base = (const uint8_t *) (vtcm_act + ((size_t) i * n_k_tiles + kt_base) * HTP_MM_HMX_TILE_N_ELMS);
                uint16_t      * out_tile = vtcm_out + (((size_t) g * n_row_tiles_32 + i) * n_col_tiles + j) * HTP_MM_HMX_TILE_N_ELMS;

                asm volatile(HMX_CLRACC_INT());

                uint32_t rem = tiles_per_group;
                const uint8_t * a_ptr = act_base;
                const uint8_t * w_ptr = wt_base;
                while (rem > 0) {
                    const uint32_t step  = (rem > 64) ? 64 : rem;
                    const uint32_t a_rng = hmx_act_sm_range(step);
                    const uint32_t w_rng = hmx_wt_b_range(step * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
                    asm volatile(HMX_LOAD_MPY_DEEP_U8_S8_SM("%0", "%1", "%2", "%3")
                                 : : "r"(a_ptr), "r"(a_rng), "r"(w_ptr), "r"(w_rng) : "memory");
                    a_ptr += step * HTP_MM_HMX_TILE_SIZE;
                    w_ptr += step * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
                    rem   -= step;
                }

                asm volatile(HMX_CVT_UH_ACC_2X1("%0") : : "r"(cvt_ctrl_16) : "memory");
                asm volatile(HMX_STORE_CVT_SM("%0", "%1") : : "r"(out_tile), "r"(out_sm_rng1) : "memory");
            }
        }
    }
}

static inline HVX_Vector hvx_qf32_mul_sf_scalar(HVX_Vector v_sf, uint32_t scalar_sf_bits) {
#if __HVX_ARCH__ >= 79
    return Q6_Vqf32_vmpy_VsfRsf(v_sf, (int32_t) scalar_sf_bits);
#else
    return Q6_Vqf32_vmpy_VsfVsf(v_sf, Q6_V_vsplat_R((int32_t) scalar_sf_bits));
#endif
}

static inline uint32_t (*htp_da_buf_ptr(uint32_t (*buf)[32]))[32] {
    asm("%0 = or(%0, #0)" : "+r"(buf));
    return buf;
}

// HVX Epilogue for W8A8 CM output tiles -> FP32 DDR output with superblock scale accumulation.
static void transfer_output_chunk_w8a8_sm_to_fp32_col_chunk(
        float * restrict dst,
        const void * restrict src2,
        const uint8_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_stride,
        uint32_t src2_stride,
        uint32_t dst_cols,
        size_t src2_elem_size) {

    const size_t total_col_tiles = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_64  = mc_pad / 64;
    const size_t group_stride    = n_row_tiles_64 * total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t row_tile_stride = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t c0_first        = c_first / 32;

    const HVX_Vector sign_80     = Q6_Vb_vsplat_R(0x80);
    const size_t limit_c         = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned = (limit_c & ~31u);
    const bool   dst_aligned_nt  = ((((uintptr_t) dst | ((size_t) dst_stride * sizeof(float))) & 127u) == 0);
    const size_t src2_row_bytes  = (size_t) src2_stride * src2_elem_size;

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = start_row + r_base;
            const size_t r_blk      = r_idx_base >> 5;
            const size_t r0         = r_idx_base >> 6;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r_blk << 5));

            for (size_t r = r_base; r < r_end; r += 4) {
                const size_t   r_idx0 = start_row + r;
                const size_t   rg     = (r_idx0 & 63u) >> 2;
                const size_t   lane   = r_idx0 & 31u;
                const uint32_t da0    = da_ptr[0][lane + 0];
                const uint32_t da1    = da_ptr[0][lane + 1];
                const uint32_t da2    = da_ptr[0][lane + 2];
                const uint32_t da3    = da_ptr[0][lane + 3];

                const uint8_t * tile_base = vtcm_src + r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;
                const float *   dw_base   = vtcm_weight_scales + c0_first * 32;
                float *         out_row0  = dst + r * dst_stride;
                const uint8_t * src2_row0 = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;
                const bool      full_quad = (r + 3 < n_rows);

                if (dst_aligned_nt && full_quad && !src2) {
                    for (size_t c = 0; c < limit_c_aligned; c += 32) {
                        const size_t    c0         = c >> 5;
                        const uint8_t * tp         = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;
                        const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);
                        const HVX_Vector v_lo_u8   = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);
                        const HVX_Vector v_dw_qf   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));

                        HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                        HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
                        HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
                        HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
                        HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));

                        HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw_qf));
                        HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw_qf));
                        HVX_Vector v_out2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da2), v_dw_qf));
                        HVX_Vector v_out3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da3), v_dw_qf));

                        hvx_vmem_store_nt(out_row0 + c + 0 * dst_stride, v_out0);
                        hvx_vmem_store_nt(out_row0 + c + 1 * dst_stride, v_out1);
                        hvx_vmem_store_nt(out_row0 + c + 2 * dst_stride, v_out2);
                        hvx_vmem_store_nt(out_row0 + c + 3 * dst_stride, v_out3);
                    }
                } else {
                    for (size_t c = 0; c < limit_c_aligned; c += 32) {
                        const size_t    c0         = c >> 5;
                        const uint8_t * tp         = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;
                        const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);
                        const HVX_Vector v_lo_u8   = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);
                        const HVX_Vector v_dw_qf   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));

                        HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                        HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
                        HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
                        HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
                        HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));

                        HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw_qf);
                        HVX_Vector v_out0    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row0 + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row0 + c + 0 * dst_stride, v_out0);
                        } else {
                            hvx_vmemu(out_row0 + c + 0 * dst_stride) = v_out0;
                        }

                        if (r + 1 < n_rows) {
                            HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw_qf);
                            HVX_Vector v_out1    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row0 + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                            if (dst_aligned_nt) {
                                hvx_vmem_store_nt(out_row0 + c + 1 * dst_stride, v_out1);
                            } else {
                                hvx_vmemu(out_row0 + c + 1 * dst_stride) = v_out1;
                            }
                        }
                        if (r + 2 < n_rows) {
                            HVX_Vector v_acc2_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da2), v_dw_qf);
                            HVX_Vector v_out2    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row0 + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
                            if (dst_aligned_nt) {
                                hvx_vmem_store_nt(out_row0 + c + 2 * dst_stride, v_out2);
                            } else {
                                hvx_vmemu(out_row0 + c + 2 * dst_stride) = v_out2;
                            }
                        }
                        if (r + 3 < n_rows) {
                            HVX_Vector v_acc3_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da3), v_dw_qf);
                            HVX_Vector v_out3    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row0 + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
                            if (dst_aligned_nt) {
                                hvx_vmem_store_nt(out_row0 + c + 3 * dst_stride, v_out3);
                            } else {
                                hvx_vmemu(out_row0 + c + 3 * dst_stride) = v_out3;
                            }
                        }
                    }
                }

                if (limit_c_aligned < limit_c) {
                    const size_t    c           = limit_c_aligned;
                    const size_t    c0          = c >> 5;
                    const size_t    valid_bytes = (limit_c - c) * sizeof(float);
                    const uint8_t * tp          = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;
                    const HVX_Vector v_hi_s8    = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);
                    const HVX_Vector v_lo_u8    = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);
                    const HVX_Vector v_dw_qf    = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));

                    HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                    HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
                    HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

                    HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
                    HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw_qf);
                    HVX_Vector v_out0    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row0 + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);
                    hvx_vec_store_u(out_row0 + c + 0 * dst_stride, valid_bytes, v_out0);

                    if (r + 1 < n_rows) {
                        HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
                        HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw_qf);
                        HVX_Vector v_out1    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row0 + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                        hvx_vec_store_u(out_row0 + c + 1 * dst_stride, valid_bytes, v_out1);
                    }
                    if (r + 2 < n_rows) {
                        HVX_Vector v_f32_2   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
                        HVX_Vector v_acc2_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da2), v_dw_qf);
                        HVX_Vector v_out2    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row0 + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
                        hvx_vec_store_u(out_row0 + c + 2 * dst_stride, valid_bytes, v_out2);
                    }
                    if (r + 3 < n_rows) {
                        HVX_Vector v_f32_3   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));
                        HVX_Vector v_acc3_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da3), v_dw_qf);
                        HVX_Vector v_out3    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row0 + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
                        hvx_vec_store_u(out_row0 + c + 3 * dst_stride, valid_bytes, v_out3);
                    }
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = start_row + r_base;
        const size_t r_blk      = r_idx_base >> 5;
        const size_t r0         = r_idx_base >> 6;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r_blk << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 4) {
            const size_t r_idx0 = start_row + r;
            const size_t rg     = (r_idx0 & 63u) >> 2;
            const size_t lane   = r_idx0 & 31u;

            float *         out_row_base  = dst + r * dst_stride;
            const uint8_t * src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;

            for (size_t c = 0; c < limit_c_aligned; c += 32) {
                const size_t c0      = c >> 5;
                const size_t c0_glob = c0_first + c0;

                HVX_Vector v_acc0_qf = Q6_V_vzero();
                HVX_Vector v_acc1_qf = Q6_V_vzero();
                HVX_Vector v_acc2_qf = Q6_V_vzero();
                HVX_Vector v_acc3_qf = Q6_V_vzero();

                for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
                    const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
                    if (n_groups > 16u) {
                        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                            *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r_blk << 5));
                        }
                    }
                    for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                        const uint32_t   g         = g_base + gi;
                        const uint8_t *  tile_pair = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * 2 * HTP_MM_HMX_TILE_SIZE;
                        const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(((const HVX_Vector *) tile_pair)[rg], sign_80);
                        const HVX_Vector v_lo_u8   = ((const HVX_Vector *) (tile_pair + HTP_MM_HMX_TILE_SIZE))[rg];

                        HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                        HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
                        HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
                        HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
                        HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));

                        const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
                        const uint32_t * da_row  = da_ptr[gi] + lane;

                        v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
                        v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
                        v_acc2_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc2_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da_row[2]), v_dw_qf));
                        v_acc3_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc3_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da_row[3]), v_dw_qf));
                    }
                }

                HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);
                if (dst_aligned_nt) {
                    hvx_vmem_store_nt(out_row_base + c + 0 * dst_stride, v_out0);
                } else {
                    hvx_vmemu(out_row_base + c + 0 * dst_stride) = v_out0;
                }

                if (r + 1 < n_rows) {
                    HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 1 * dst_stride, v_out1);
                    } else {
                        hvx_vmemu(out_row_base + c + 1 * dst_stride) = v_out1;
                    }
                }
                if (r + 2 < n_rows) {
                    HVX_Vector v_out2 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row_base + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 2 * dst_stride, v_out2);
                    } else {
                        hvx_vmemu(out_row_base + c + 2 * dst_stride) = v_out2;
                    }
                }
                if (r + 3 < n_rows) {
                    HVX_Vector v_out3 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row_base + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 3 * dst_stride, v_out3);
                    } else {
                        hvx_vmemu(out_row_base + c + 3 * dst_stride) = v_out3;
                    }
                }
            }

            if (limit_c_aligned < limit_c) {
                const size_t c           = limit_c_aligned;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_bytes = (limit_c - c) * sizeof(float);

                HVX_Vector v_acc0_qf = Q6_V_vzero();
                HVX_Vector v_acc1_qf = Q6_V_vzero();
                HVX_Vector v_acc2_qf = Q6_V_vzero();
                HVX_Vector v_acc3_qf = Q6_V_vzero();

                for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
                    const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
                    if (n_groups > 16u) {
                        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                            *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r_blk << 5));
                        }
                    }
                    for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                        const uint32_t   g         = g_base + gi;
                        const uint8_t *  tile_pair = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * 2 * HTP_MM_HMX_TILE_SIZE;
                        const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(((const HVX_Vector *) tile_pair)[rg], sign_80);
                        const HVX_Vector v_lo_u8   = ((const HVX_Vector *) (tile_pair + HTP_MM_HMX_TILE_SIZE))[rg];

                        HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                        HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
                        HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
                        HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
                        HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));

                        const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
                        const uint32_t * da_row  = da_ptr[gi] + lane;

                        v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
                        v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
                        v_acc2_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc2_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da_row[2]), v_dw_qf));
                        v_acc3_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc3_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da_row[3]), v_dw_qf));
                    }
                }

                HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);
                hvx_vec_store_u(out_row_base + c + 0 * dst_stride, valid_bytes, v_out0);

                if (r + 1 < n_rows) {
                    HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                    hvx_vec_store_u(out_row_base + c + 1 * dst_stride, valid_bytes, v_out1);
                }
                if (r + 2 < n_rows) {
                    HVX_Vector v_out2 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row_base + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
                    hvx_vec_store_u(out_row_base + c + 2 * dst_stride, valid_bytes, v_out2);
                }
                if (r + 3 < n_rows) {
                    HVX_Vector v_out3 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row_base + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
                    hvx_vec_store_u(out_row_base + c + 3 * dst_stride, valid_bytes, v_out3);
                }
            }
        }
        r_base = r_end;
    }
}

static inline void w8a8_sm_eval_tile_quad_1g(
        const uint8_t * restrict tile_base,
        const float * restrict dw_base,
        size_t c,
        uint32_t da0,
        uint32_t da1,
        uint32_t da2,
        uint32_t da3,
        HVX_Vector sign_80,
        const uint8_t * restrict src2_row0,
        size_t src2_row_bytes,
        size_t src2_elem_size,
        uint32_t n_rows,
        size_t r,
        HVX_Vector * v_out0,
        HVX_Vector * v_out1,
        HVX_Vector * v_out2,
        HVX_Vector * v_out3) {
    const size_t     c0       = c >> 5;
    const uint8_t *  tp       = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;
    const HVX_Vector v_hi_s8  = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);
    const HVX_Vector v_lo_u8  = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);
    const HVX_Vector v_dw_qf  = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));

    HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
    HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
    HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

    HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
    HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw_qf);
    *v_out0 = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row0 + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);

    if (r + 1 < n_rows) {
        HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
        HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw_qf);
        *v_out1 = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row0 + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
    }
    if (r + 2 < n_rows) {
        HVX_Vector v_f32_2   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
        HVX_Vector v_acc2_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da2), v_dw_qf);
        *v_out2 = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row0 + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
    }
    if (r + 3 < n_rows) {
        HVX_Vector v_f32_3   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));
        HVX_Vector v_acc3_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da3), v_dw_qf);
        *v_out3 = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row0 + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
    }
}

static inline void w8a8_sm_eval_tile_quad_ng(
        const uint8_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        size_t group_stride,
        size_t row_tile_stride,
        size_t r0,
        size_t r_blk,
        size_t rg,
        size_t lane,
        uint32_t mc_pad,
        size_t c0_glob,
        uint32_t n_groups,
        uint32_t (*da_ptr)[32],
        HVX_Vector sign_80,
        const uint8_t * restrict src2_row_base,
        size_t c,
        size_t src2_row_bytes,
        size_t src2_elem_size,
        uint32_t n_rows,
        size_t r,
        HVX_Vector * v_out0,
        HVX_Vector * v_out1,
        HVX_Vector * v_out2,
        HVX_Vector * v_out3) {
    HVX_Vector v_acc0_qf = Q6_V_vzero();
    HVX_Vector v_acc1_qf = Q6_V_vzero();
    HVX_Vector v_acc2_qf = Q6_V_vzero();
    HVX_Vector v_acc3_qf = Q6_V_vzero();

    for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
        const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
        if (n_groups > 16u) {
            for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r_blk << 5));
            }
        }
        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
            const uint32_t   g         = g_base + gi;
            const uint8_t *  tile_pair = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * 2 * HTP_MM_HMX_TILE_SIZE;
            const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(((const HVX_Vector *) tile_pair)[rg], sign_80);
            const HVX_Vector v_lo_u8   = ((const HVX_Vector *) (tile_pair + HTP_MM_HMX_TILE_SIZE))[rg];

            HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
            HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
            HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

            HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
            HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
            HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
            HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));

            const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
            const uint32_t * da_row  = da_ptr[gi] + lane;

            v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
            v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
            v_acc2_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc2_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da_row[2]), v_dw_qf));
            v_acc3_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc3_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da_row[3]), v_dw_qf));
        }
    }

    *v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);
    if (r + 1 < n_rows) {
        *v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
    }
    if (r + 2 < n_rows) {
        *v_out2 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row_base + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
    }
    if (r + 3 < n_rows) {
        *v_out3 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row_base + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
    }
}

// HVX Epilogue for W8A8 CM output tiles -> FP16 DDR output with superblock scale accumulation.
static __attribute__((noinline)) void transfer_output_chunk_w8a8_sm_to_fp16_col_chunk(
        __fp16 * restrict dst,
        const void * restrict src2,
        const uint8_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_stride,
        uint32_t src2_stride,
        uint32_t dst_cols,
        size_t src2_elem_size) {

    const size_t total_col_tiles   = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_64    = mc_pad / 64;
    const size_t group_stride      = n_row_tiles_64 * total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t row_tile_stride   = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t c0_first          = c_first / 32;

    const HVX_Vector sign_80       = Q6_Vb_vsplat_R(0x80);
    const size_t limit_c           = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned64 = (limit_c & ~63u);
    const bool   dst_aligned_nt    = ((((uintptr_t) dst | ((size_t) dst_stride * sizeof(__fp16))) & 127u) == 0);
    const size_t src2_row_bytes    = (size_t) src2_stride * src2_elem_size;

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = start_row + r_base;
            const size_t r_blk      = r_idx_base >> 5;
            const size_t r0         = r_idx_base >> 6;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r_blk << 5));

            for (size_t r = r_base; r < r_end; r += 4) {
                const size_t   r_idx0 = start_row + r;
                const size_t   rg     = (r_idx0 & 63u) >> 2;
                const size_t   lane   = r_idx0 & 31u;
                const uint32_t da0    = da_ptr[0][lane + 0];
                const uint32_t da1    = da_ptr[0][lane + 1];
                const uint32_t da2    = da_ptr[0][lane + 2];
                const uint32_t da3    = da_ptr[0][lane + 3];

                const uint8_t * tile_base = vtcm_src + r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;
                const float *   dw_base   = vtcm_weight_scales + c0_first * 32;
                __fp16 *        out_row0  = dst + r * dst_stride;
                const uint8_t * src2_row0 = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;

                for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                    HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c, da0, da1, da2, da3, sign_80,
                                              src2_row0, src2_row_bytes, src2_elem_size, n_rows, r,
                                              &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c + 32, da0, da1, da2, da3, sign_80,
                                              src2_row0, src2_row_bytes, src2_elem_size, n_rows, r,
                                              &v0_hi, &v1_hi, &v2_hi, &v3_hi);

                    HVX_Vector v_f16_0 = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row0 + c + 0 * dst_stride, v_f16_0);
                    } else {
                        hvx_vmemu(out_row0 + c + 0 * dst_stride) = v_f16_0;
                    }
                    if (r + 1 < n_rows) {
                        HVX_Vector v_f16_1 = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row0 + c + 1 * dst_stride, v_f16_1);
                        } else {
                            hvx_vmemu(out_row0 + c + 1 * dst_stride) = v_f16_1;
                        }
                    }
                    if (r + 2 < n_rows) {
                        HVX_Vector v_f16_2 = hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row0 + c + 2 * dst_stride, v_f16_2);
                        } else {
                            hvx_vmemu(out_row0 + c + 2 * dst_stride) = v_f16_2;
                        }
                    }
                    if (r + 3 < n_rows) {
                        HVX_Vector v_f16_3 = hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row0 + c + 3 * dst_stride, v_f16_3);
                        } else {
                            hvx_vmemu(out_row0 + c + 3 * dst_stride) = v_f16_3;
                        }
                    }
                }

                if (limit_c_aligned64 < limit_c) {
                    const size_t c           = limit_c_aligned64;
                    const size_t valid_c     = limit_c - c;
                    const size_t valid_bytes = valid_c * sizeof(__fp16);

                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                    HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c, da0, da1, da2, da3, sign_80,
                                              src2_row0, src2_row_bytes, src2_elem_size, n_rows, r,
                                              &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                    if (valid_c > 32) {
                        w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c + 32, da0, da1, da2, da3, sign_80,
                                                  src2_row0, src2_row_bytes, src2_elem_size, n_rows, r,
                                                  &v0_hi, &v1_hi, &v2_hi, &v3_hi);
                    }

                    hvx_vec_store_u(out_row0 + c + 0 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                    if (r + 1 < n_rows) {
                        hvx_vec_store_u(out_row0 + c + 1 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                    }
                    if (r + 2 < n_rows) {
                        hvx_vec_store_u(out_row0 + c + 2 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi));
                    }
                    if (r + 3 < n_rows) {
                        hvx_vec_store_u(out_row0 + c + 3 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi));
                    }
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = start_row + r_base;
        const size_t r_blk      = r_idx_base >> 5;
        const size_t r0         = r_idx_base >> 6;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r_blk << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 4) {
            const size_t r_idx0 = start_row + r;
            const size_t rg     = (r_idx0 & 63u) >> 2;
            const size_t lane   = r_idx0 & 31u;

            __fp16 *        out_row_base  = dst + r * dst_stride;
            const uint8_t * src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;

            for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                const size_t c0_glob = c0_first + (c >> 5);
                HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob, n_groups, da_ptr, sign_80,
                                          src2_row_base, c, src2_row_bytes, src2_elem_size, n_rows, r,
                                          &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob + 1, n_groups, da_ptr, sign_80,
                                          src2_row_base, c + 32, src2_row_bytes, src2_elem_size, n_rows, r,
                                          &v0_hi, &v1_hi, &v2_hi, &v3_hi);

                HVX_Vector v_f16_0 = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);
                if (dst_aligned_nt) {
                    hvx_vmem_store_nt(out_row_base + c + 0 * dst_stride, v_f16_0);
                } else {
                    hvx_vmemu(out_row_base + c + 0 * dst_stride) = v_f16_0;
                }
                if (r + 1 < n_rows) {
                    HVX_Vector v_f16_1 = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 1 * dst_stride, v_f16_1);
                    } else {
                        hvx_vmemu(out_row_base + c + 1 * dst_stride) = v_f16_1;
                    }
                }
                if (r + 2 < n_rows) {
                    HVX_Vector v_f16_2 = hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 2 * dst_stride, v_f16_2);
                    } else {
                        hvx_vmemu(out_row_base + c + 2 * dst_stride) = v_f16_2;
                    }
                }
                if (r + 3 < n_rows) {
                    HVX_Vector v_f16_3 = hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 3 * dst_stride, v_f16_3);
                    } else {
                        hvx_vmemu(out_row_base + c + 3 * dst_stride) = v_f16_3;
                    }
                }
            }

            if (limit_c_aligned64 < limit_c) {
                const size_t c           = limit_c_aligned64;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_c     = limit_c - c;
                const size_t valid_bytes = valid_c * sizeof(__fp16);

                HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob, n_groups, da_ptr, sign_80,
                                          src2_row_base, c, src2_row_bytes, src2_elem_size, n_rows, r,
                                          &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                if (valid_c > 32) {
                    w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                              group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                              mc_pad, c0_glob + 1, n_groups, da_ptr, sign_80,
                                              src2_row_base, c + 32, src2_row_bytes, src2_elem_size, n_rows, r,
                                              &v0_hi, &v1_hi, &v2_hi, &v3_hi);
                }

                hvx_vec_store_u(out_row_base + c + 0 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                if (r + 1 < n_rows) {
                    hvx_vec_store_u(out_row_base + c + 1 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                }
                if (r + 2 < n_rows) {
                    hvx_vec_store_u(out_row_base + c + 2 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi));
                }
                if (r + 3 < n_rows) {
                    hvx_vec_store_u(out_row_base + c + 3 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi));
                }
            }
        }
        r_base = r_end;
    }
}

// HVX Epilogue for W8A16 :2x1 SM output tiles -> FP32 DDR output with superblock scale accumulation.
static void transfer_output_chunk_w8a16_sm_to_fp32_col_chunk(
        float * restrict dst,
        const void * restrict src2,
        const uint16_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_stride,
        uint32_t src2_stride,
        uint32_t dst_cols,
        size_t src2_elem_size) {

    const size_t total_col_tiles = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_32  = mc_pad / 32;
    const size_t group_stride    = n_row_tiles_32 * total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t row_tile_stride = total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t c0_first        = c_first / 32;

    const HVX_Vector sign_8000   = Q6_Vh_vsplat_R(0x8000);
    const size_t limit_c         = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned = (limit_c & ~31u);
    const bool   dst_aligned_nt  = ((((uintptr_t) dst | ((size_t) dst_stride * sizeof(float))) & 127u) == 0);
    const size_t src2_row_bytes  = (size_t) src2_stride * src2_elem_size;

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = start_row + r_base;
            const size_t r0         = r_idx_base >> 5;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r0 << 5));

            for (size_t r = r_base; r < r_end; r += 2) {
                const size_t   lane = (start_row + r) & 31u;
                const size_t   r1   = lane >> 1;
                const uint32_t da0  = da_ptr[0][lane + 0];
                const uint32_t da1  = da_ptr[0][lane + 1];

                const uint16_t * tile_base     = vtcm_src + r0 * row_tile_stride + c0_first * HTP_MM_HMX_TILE_N_ELMS + r1 * 64;
                const float *    dw_base       = vtcm_weight_scales + c0_first * 32;
                float *          out_row_base  = dst + r * dst_stride;
                const uint8_t *  src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;
                const bool       has_row1      = (r + 1 < n_rows);

                if (dst_aligned_nt && has_row1 && !src2) {
                    #pragma unroll(2)
                    for (size_t c = 0; c < limit_c_aligned; c += 32) {
                        const size_t     c0     = c >> 5;
                        const HVX_Vector v_i16  = Q6_V_vxor_VV(*(const HVX_Vector *) (tile_base + c0 * HTP_MM_HMX_TILE_N_ELMS), sign_8000);
                        const HVX_Vector v_dw   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));
                        HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));

                        HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw));
                        HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw));

                        hvx_vmem_store_nt(out_row_base + c, v_out0);
                        hvx_vmem_store_nt(out_row_base + c + dst_stride, v_out1);
                    }
                } else {
                    for (size_t c = 0; c < limit_c_aligned; c += 32) {
                        const size_t     c0     = c >> 5;
                        const HVX_Vector v_i16  = Q6_V_vxor_VV(*(const HVX_Vector *) (tile_base + c0 * HTP_MM_HMX_TILE_N_ELMS), sign_8000);
                        const HVX_Vector v_dw   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));
                        HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                        HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                        HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw);
                        HVX_Vector v_out0    = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row_base + c, v_out0);
                        } else {
                            hvx_vmemu(out_row_base + c) = v_out0;
                        }

                        if (has_row1) {
                            HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));
                            HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw);
                            HVX_Vector v_out1    = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                            if (dst_aligned_nt) {
                                hvx_vmem_store_nt(out_row_base + c + dst_stride, v_out1);
                            } else {
                                hvx_vmemu(out_row_base + c + dst_stride) = v_out1;
                            }
                        }
                    }
                }

                if (limit_c_aligned < limit_c) {
                    const size_t     c           = limit_c_aligned;
                    const size_t     c0          = c >> 5;
                    const size_t     valid_bytes = (limit_c - c) * sizeof(float);
                    const HVX_Vector v_i16       = Q6_V_vxor_VV(*(const HVX_Vector *) (tile_base + c0 * HTP_MM_HMX_TILE_N_ELMS), sign_8000);
                    const HVX_Vector v_dw        = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));
                    HVX_VectorPair   vp_i32      = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                    HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                    HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw);
                    HVX_Vector v_out0    = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);
                    hvx_vec_store_u(out_row_base + c, valid_bytes, v_out0);

                    if (has_row1) {
                        HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));
                        HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw);
                        HVX_Vector v_out1    = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                        hvx_vec_store_u(out_row_base + c + dst_stride, valid_bytes, v_out1);
                    }
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = start_row + r_base;
        const size_t r0         = r_idx_base >> 5;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r0 << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 2) {
            const size_t lane = (start_row + r) & 31u;
            const size_t r1   = lane >> 1;

            float *         out_row_base  = dst + r * dst_stride;
            const uint8_t * src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;
            const bool      has_row1      = (r + 1 < n_rows);

            for (size_t c = 0; c < limit_c_aligned; c += 32) {
                const size_t c0      = c >> 5;
                const size_t c0_glob = c0_first + c0;

                HVX_Vector v_acc0_qf = Q6_V_vzero();
                HVX_Vector v_acc1_qf = Q6_V_vzero();

                for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
                    const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
                    if (n_groups > 16u) {
                        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                            *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r0 << 5));
                        }
                    }
                    for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                        const uint32_t   g      = g_base + gi;
                        const uint16_t * tile   = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * HTP_MM_HMX_TILE_N_ELMS;
                        const HVX_Vector v_i16  = Q6_V_vxor_VV(((const HVX_Vector *) tile)[r1], sign_8000);
                        HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));

                        const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
                        const uint32_t * da_row  = da_ptr[gi] + lane;

                        v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
                        v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
                    }
                }

                HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);
                if (dst_aligned_nt) {
                    hvx_vmem_store_nt(out_row_base + c, v_out0);
                } else {
                    hvx_vmemu(out_row_base + c) = v_out0;
                }

                if (has_row1) {
                    HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + dst_stride, v_out1);
                    } else {
                        hvx_vmemu(out_row_base + c + dst_stride) = v_out1;
                    }
                }
            }

            if (limit_c_aligned < limit_c) {
                const size_t c           = limit_c_aligned;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_bytes = (limit_c - c) * sizeof(float);

                HVX_Vector v_acc0_qf = Q6_V_vzero();
                HVX_Vector v_acc1_qf = Q6_V_vzero();

                for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
                    const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
                    if (n_groups > 16u) {
                        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                            *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r0 << 5));
                        }
                    }
                    for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                        const uint32_t   g      = g_base + gi;
                        const uint16_t * tile   = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * HTP_MM_HMX_TILE_N_ELMS;
                        const HVX_Vector v_i16  = Q6_V_vxor_VV(((const HVX_Vector *) tile)[r1], sign_8000);
                        HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));

                        const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
                        const uint32_t * da_row  = da_ptr[gi] + lane;

                        v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
                        v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
                    }
                }

                HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);
                hvx_vec_store_u(out_row_base + c, valid_bytes, v_out0);

                if (has_row1) {
                    HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                    hvx_vec_store_u(out_row_base + c + dst_stride, valid_bytes, v_out1);
                }
            }
        }
        r_base = r_end;
    }
}

static inline void w8a16_sm_eval_tile_pair_1g(
        const uint16_t * restrict tile_base,
        const float * restrict dw_base,
        size_t c,
        uint32_t da0,
        uint32_t da1,
        HVX_Vector sign_8000,
        const uint8_t * restrict src2_row_base,
        size_t src2_row_bytes,
        size_t src2_elem_size,
        bool has_row1,
        HVX_Vector * v_out0,
        HVX_Vector * v_out1) {
    const size_t     c0     = c >> 5;
    const HVX_Vector v_i16  = Q6_V_vxor_VV(*(const HVX_Vector *) (tile_base + c0 * HTP_MM_HMX_TILE_N_ELMS), sign_8000);
    const HVX_Vector v_dw   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));
    HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

    HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
    HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw);
    *v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);

    if (has_row1) {
        HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));
        HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw);
        *v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
    }
}

static inline void w8a16_sm_eval_tile_pair_ng(
        const uint16_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        size_t group_stride,
        size_t row_tile_stride,
        size_t r0,
        size_t r1,
        size_t lane,
        uint32_t mc_pad,
        size_t c0_glob,
        uint32_t n_groups,
        uint32_t (*da_ptr)[32],
        HVX_Vector sign_8000,
        const uint8_t * restrict src2_row_base,
        size_t c,
        size_t src2_row_bytes,
        size_t src2_elem_size,
        bool has_row1,
        HVX_Vector * v_out0,
        HVX_Vector * v_out1) {
    HVX_Vector v_acc0_qf = Q6_V_vzero();
    HVX_Vector v_acc1_qf = Q6_V_vzero();

    for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
        const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
        if (n_groups > 16u) {
            for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r0 << 5));
            }
        }
        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
            const uint32_t   g      = g_base + gi;
            const uint16_t * tile   = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * HTP_MM_HMX_TILE_N_ELMS;
            const HVX_Vector v_i16  = Q6_V_vxor_VV(((const HVX_Vector *) tile)[r1], sign_8000);
            HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

            HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
            HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));

            const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
            const uint32_t * da_row  = da_ptr[gi] + lane;

            v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
            v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
        }
    }

    *v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);
    if (has_row1) {
        *v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
    }
}

// HVX Epilogue for W8A16 :2x1 SM output tiles -> FP16 DDR output with superblock scale accumulation.
static __attribute__((noinline)) void transfer_output_chunk_w8a16_sm_to_fp16_col_chunk(
        __fp16 * restrict dst,
        const void * restrict src2,
        const uint16_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_stride,
        uint32_t src2_stride,
        uint32_t dst_cols,
        size_t src2_elem_size) {

    const size_t total_col_tiles   = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_32    = mc_pad / 32;
    const size_t group_stride      = n_row_tiles_32 * total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t row_tile_stride   = total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t c0_first          = c_first / 32;

    const HVX_Vector sign_8000     = Q6_Vh_vsplat_R(0x8000);
    const size_t limit_c           = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned64 = (limit_c & ~63u);
    const bool   dst_aligned_nt    = ((((uintptr_t) dst | ((size_t) dst_stride * sizeof(__fp16))) & 127u) == 0);
    const size_t src2_row_bytes    = (size_t) src2_stride * src2_elem_size;

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = start_row + r_base;
            const size_t r0         = r_idx_base >> 5;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r0 << 5));

            for (size_t r = r_base; r < r_end; r += 2) {
                const size_t   lane = (start_row + r) & 31u;
                const size_t   r1   = lane >> 1;
                const uint32_t da0  = da_ptr[0][lane + 0];
                const uint32_t da1  = da_ptr[0][lane + 1];

                const uint16_t * tile_base     = vtcm_src + r0 * row_tile_stride + c0_first * HTP_MM_HMX_TILE_N_ELMS + r1 * 64;
                const float *    dw_base       = vtcm_weight_scales + c0_first * 32;
                __fp16 *         out_row_base  = dst + r * dst_stride;
                const uint8_t *  src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;
                const bool       has_row1      = (r + 1 < n_rows);

                for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                    HVX_Vector v0_hi, v1_hi = Q6_V_vzero();

                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c, da0, da1, sign_8000,
                                               src2_row_base, src2_row_bytes, src2_elem_size, has_row1,
                                               &v0_lo, &v1_lo);
                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c + 32, da0, da1, sign_8000,
                                               src2_row_base, src2_row_bytes, src2_elem_size, has_row1,
                                               &v0_hi, &v1_hi);

                    HVX_Vector v_f16_0 = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c, v_f16_0);
                    } else {
                        hvx_vmemu(out_row_base + c) = v_f16_0;
                    }
                    if (has_row1) {
                        HVX_Vector v_f16_1 = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row_base + dst_stride + c, v_f16_1);
                        } else {
                            hvx_vmemu(out_row_base + dst_stride + c) = v_f16_1;
                        }
                    }
                }

                if (limit_c_aligned64 < limit_c) {
                    const size_t c           = limit_c_aligned64;
                    const size_t valid_c     = limit_c - c;
                    const size_t valid_bytes = valid_c * sizeof(__fp16);

                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                    HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero();

                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c, da0, da1, sign_8000,
                                               src2_row_base, src2_row_bytes, src2_elem_size, has_row1,
                                               &v0_lo, &v1_lo);
                    if (valid_c > 32) {
                        w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c + 32, da0, da1, sign_8000,
                                                   src2_row_base, src2_row_bytes, src2_elem_size, has_row1,
                                                   &v0_hi, &v1_hi);
                    }

                    hvx_vec_store_u(out_row_base + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                    if (has_row1) {
                        hvx_vec_store_u(out_row_base + dst_stride + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                    }
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = start_row + r_base;
        const size_t r0         = r_idx_base >> 5;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r0 << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 2) {
            const size_t lane = (start_row + r) & 31u;
            const size_t r1   = lane >> 1;

            __fp16 *        out_row_base  = dst + r * dst_stride;
            const uint8_t * src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;
            const bool      has_row1      = (r + 1 < n_rows);

            for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                const size_t c0_glob = c0_first + (c >> 5);
                HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                HVX_Vector v0_hi, v1_hi = Q6_V_vzero();

                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob, n_groups, da_ptr, sign_8000,
                                           src2_row_base, c, src2_row_bytes, src2_elem_size, has_row1,
                                           &v0_lo, &v1_lo);
                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob + 1, n_groups, da_ptr, sign_8000,
                                           src2_row_base, c + 32, src2_row_bytes, src2_elem_size, has_row1,
                                           &v0_hi, &v1_hi);

                HVX_Vector v_f16_0 = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);
                if (dst_aligned_nt) {
                    hvx_vmem_store_nt(out_row_base + c, v_f16_0);
                } else {
                    hvx_vmemu(out_row_base + c) = v_f16_0;
                }
                if (has_row1) {
                    HVX_Vector v_f16_1 = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + dst_stride + c, v_f16_1);
                    } else {
                        hvx_vmemu(out_row_base + dst_stride + c) = v_f16_1;
                    }
                }
            }

            if (limit_c_aligned64 < limit_c) {
                const size_t c           = limit_c_aligned64;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_c     = limit_c - c;
                const size_t valid_bytes = valid_c * sizeof(__fp16);

                HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero();

                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob, n_groups, da_ptr, sign_8000,
                                           src2_row_base, c, src2_row_bytes, src2_elem_size, has_row1,
                                           &v0_lo, &v1_lo);
                if (valid_c > 32) {
                    w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                               group_stride, row_tile_stride, r0, r1, lane,
                                               mc_pad, c0_glob + 1, n_groups, da_ptr, sign_8000,
                                               src2_row_base, c + 32, src2_row_bytes, src2_elem_size, has_row1,
                                               &v0_hi, &v1_hi);
                }

                hvx_vec_store_u(out_row_base + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                if (has_row1) {
                    hvx_vec_store_u(out_row_base + dst_stride + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                }
            }
        }
        r_base = r_end;
    }
}
