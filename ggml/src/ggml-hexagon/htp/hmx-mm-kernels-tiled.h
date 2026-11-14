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

// output : fp16 -> f32p

static void transfer_output_chunk_fp16_to_fp32_col_chunk(
    float *restrict dst,
    const float *restrict src2,
    const __fp16 *restrict vtcm_src,
    uint32_t start_row,
    uint32_t n_rows,
    uint32_t c_len,
    uint32_t total_n_cols,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_cols
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
        const float *src2_row_base = src2 ? (src2 + r * src2_stride) : NULL;

        #pragma unroll(4)
        for (size_t c = 0; c < limit_c_aligned; c += HTP_MM_HMX_TILE_N_COLS) {
            const size_t c0    = c / HTP_MM_HMX_TILE_N_COLS;
            const __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);

            HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
            if (src2_row_base) {
                HVX_Vector v_src2_0 = hvx_vmemu(src2_row_base + c + 0);
                v_out0 = hvx_vec_add_f32_f32(v_out0, v_src2_0);
            }
            hvx_vmemu(output_row_base + c + 0) = v_out0;

            if (r + 1 < n_rows) {
                HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                if (src2_row_base) {
                    HVX_Vector v_src2_1 = hvx_vmemu(src2_row_base + c + src2_stride);
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
            if (src2_row_base) {
                HVX_Vector v_src2_0 = hvx_vmemu(src2_row_base + c + 0);
                v_out0 = hvx_vec_add_f32_f32(v_out0, v_src2_0);
            }
            hvx_vec_store_u(output_row_base + c, valid_c * sizeof(float), v_out0);

            if (r + 1 < n_rows) {
                HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                if (src2_row_base) {
                    HVX_Vector v_src2_1 = hvx_vmemu(src2_row_base + c + src2_stride);
                    v_out1 = hvx_vec_add_f32_f32(v_out1, v_src2_1);
                }
                hvx_vec_store_u(output_row_base + c + dst_stride, valid_c * sizeof(float), v_out1);
            }
        }
    }
}

static inline void transfer_output_chunk_fp16_to_fp32(
    float *restrict dst,
    const float *restrict src2,
    const __fp16 *restrict vtcm_src,
    uint32_t start_row,
    uint32_t n_rows,
    uint32_t n_cols,
    uint32_t dst_stride,
    uint32_t src2_stride,
    uint32_t dst_cols
) {
    transfer_output_chunk_fp16_to_fp32_col_chunk(
        dst, src2, vtcm_src, start_row, n_rows, n_cols, n_cols, dst_stride, src2_stride, dst_cols
    );
}

typedef struct {
    const __fp16  *vtcm_src;
    float         *dst;
    const float   *src2;
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

// Convert two 32-float vectors (in [-32766, 32766]) to interleaved int16 (even=vsf_0, odd=vsf_1).
static inline HVX_Vector hvx_vec_i16_from_sf_rnd_sat_shuff(HVX_Vector vsf_0, HVX_Vector vsf_1) {
    const HVX_Vector v192K = Q6_V_vsplat_R(0x48400000); // 196608.0f
    HVX_Vector v0 = hvx_vec_add_f32_f32(vsf_0, v192K);
    HVX_Vector v1 = hvx_vec_add_f32_f32(vsf_1, v192K);
    v0 = Q6_Vw_vasl_VwR(v0, 10);
    v1 = Q6_Vw_vasl_VwR(v1, 10);
    return Q6_Vh_vround_VwVw_sat(v1, v0);
}

// Rescale one (ct, g) superblock of Q4_0 tiles into int8 weight.b tiles,
// compute sum_w and l1_w per output channel, and write the Bias Register bank(s) and D_w_eff.
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

    // Pass 1: find per-OC max |d(oc, kt)| across the superblock
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector v_sc = Q6_V_vand_VV(hvx_vmem(tile_src + 512), v_abs_mask);
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, v_sc);
    }

    HVX_Vector v_max_dup    = Q6_V_lo_W(Q6_W_vshuff_VVR(v_max_sc, v_max_sc, -2));
    HVX_Vector v_max_sf     = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_max_dup, v_one_f16)));
    HVX_VectorPred q_pos    = Q6_Q_vcmp_gt_VsfVsf(v_max_sf, hvx_vec_splat_f32(1e-30f));
    HVX_Vector v_inv_raw    = hvx_vec_inverse_f32(v_max_sf);
    HVX_Vector v_inv_max_sf = Q6_V_vmux_QVV(q_pos, hvx_vec_mul_f32_f32(v_inv_raw, hvx_vec_splat_f32(16256.0f)), Q6_V_vzero());

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    // Pass 2: rescale nibbles into int8 weight.b tiles and accumulate sum_w and l1_w
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);

        HVX_Vector v_sc     = hvx_vmem(tile_src + 512);
        HVX_Vector v_sc_dup = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_Vector v_sc_sf  = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_sc_dup, v_one_f16)));
        HVX_Vector v_m_sf   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_sc_sf, v_inv_max_sf));
        HVX_Vector v_m_q11  = Q6_Vh_vasl_VhR(hvx_vec_i16_from_sf_rnd_sat_shuff(v_m_sf, v_m_sf), 1);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            HVX_Vector vq   = hvx_vmem(tile_src + i * 128);
            HVX_Vector v_lo = Q6_V_vxor_VV(Q6_V_vand_VV(Q6_Vh_vasl_VhR(vq, 4), mask_f0), sign_80);
            HVX_Vector v_hi = Q6_V_vxor_VV(Q6_V_vand_VV(vq, mask_f0), sign_80);

            HVX_VectorPair vp_shuf  = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
            HVX_VectorPair vp_16_lo = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf));
            HVX_VectorPair vp_16_hi = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf));

            HVX_Vector v_s16_0 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_V_lo_W(vp_16_lo), v_m_q11);
            HVX_Vector v_s16_1 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_V_hi_W(vp_16_lo), v_m_q11);
            HVX_Vector v_b_01  = Q6_Vb_vpack_VhVh_sat(v_s16_1, v_s16_0);
            HVX_Vector v_W0    = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vror_VR(v_b_01, 64), v_b_01, -2));

            HVX_Vector v_s16_2 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_V_lo_W(vp_16_hi), v_m_q11);
            HVX_Vector v_s16_3 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_V_hi_W(vp_16_hi), v_m_q11);
            HVX_Vector v_b_23  = Q6_Vb_vpack_VhVh_sat(v_s16_3, v_s16_2);
            HVX_Vector v_W1    = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vror_VR(v_b_23, 64), v_b_23, -2));

            tile_dst[2 * i + 0] = v_W0;
            tile_dst[2 * i + 1] = v_W1;

            v_sum_w = Q6_Vw_vrmpyacc_VwVbVb(v_sum_w, v_W0, v_ones_s8);
            v_sum_w = Q6_Vw_vrmpyacc_VwVbVb(v_sum_w, v_W1, v_ones_s8);
            v_l1_w  = Q6_Vw_vrmpyacc_VwVbVb(v_l1_w, Q6_Vb_vabs_Vb(v_W0), v_ones_s8);
            v_l1_w  = Q6_Vw_vrmpyacc_VwVbVb(v_l1_w, Q6_Vb_vabs_Vb(v_W1), v_ones_s8);
        }
    }

    HVX_Vector v_max_mag = Q6_Vw_vasl_VwR(v_l1_w, 7);
    HVX_Vector v_p       = Q6_Vw_vmax_VwVw(Q6_V_vzero(), Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(17), Q6_Vuw_vcl0_Vuw(v_max_mag)));
    HVX_Vector v_ib      = Q6_Vw_vsub_VwVw(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(32768), v_p), Q6_Vw_vasl_VwR(v_sum_w, 7));

    const float scale_const = is_w8a8 ? (8.0f / 127.0f) : (2048.0f / 127.0f);
    HVX_Vector v_dw_base    = hvx_vec_mul_f32_f32(v_max_sf, hvx_vec_splat_f32(scale_const));
    HVX_Vector v_pow2_sf    = Q6_Vsf_equals_Vw(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(1), v_p));
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

    const uint32_t k_end = k_start + k_group;
    HVX_Vector vmax0 = Q6_V_vzero();
    HVX_Vector vmax1 = Q6_V_vzero();
    HVX_Vector vmax2 = Q6_V_vzero();
    HVX_Vector vmax3 = Q6_V_vzero();

    for (uint32_t c = k_start; c < k_end; c += 32) {
        HVX_Vector vx0 = *(const HVX_UVector *) (p0 + c);
        HVX_Vector vx1 = *(const HVX_UVector *) (p1 + c);
        HVX_Vector vx2 = *(const HVX_UVector *) (p2 + c);
        HVX_Vector vx3 = *(const HVX_UVector *) (p3 + c);
        if (c + 32 > k_valid) {
            uint32_t rem = (c < k_valid) ? (k_valid - c) : 0;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
            vx0 = Q6_V_vmux_QVV(mask, vx0, Q6_V_vzero());
            vx1 = Q6_V_vmux_QVV(mask, vx1, Q6_V_vzero());
            vx2 = Q6_V_vmux_QVV(mask, vx2, Q6_V_vzero());
            vx3 = Q6_V_vmux_QVV(mask, vx3, Q6_V_vzero());
        }
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

    const HVX_Vector v_min_127 = Q6_Vh_vsplat_R(-127);
    const HVX_Vector v_max_127 = Q6_Vh_vsplat_R(127);
    const HVX_Vector v_zp_128  = Q6_Vb_vsplat_R(0x80);

    for (uint32_t c = k_start; c < k_end; c += 32) {
        HVX_Vector vx0 = *(const HVX_UVector *) (p0 + c);
        HVX_Vector vx1 = *(const HVX_UVector *) (p1 + c);
        HVX_Vector vx2 = *(const HVX_UVector *) (p2 + c);
        HVX_Vector vx3 = *(const HVX_UVector *) (p3 + c);
        if (c + 32 > k_valid) {
            uint32_t rem = (c < k_valid) ? (k_valid - c) : 0;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
            vx0 = Q6_V_vmux_QVV(mask, vx0, Q6_V_vzero());
            vx1 = Q6_V_vmux_QVV(mask, vx1, Q6_V_vzero());
            vx2 = Q6_V_vmux_QVV(mask, vx2, Q6_V_vzero());
            vx3 = Q6_V_vmux_QVV(mask, vx3, Q6_V_vzero());
        }
        HVX_Vector vsf0 = hvx_vec_mul_f32_f32(vx0, v_inv0);
        HVX_Vector vsf1 = hvx_vec_mul_f32_f32(vx1, v_inv1);
        HVX_Vector vsf2 = hvx_vec_mul_f32_f32(vx2, v_inv2);
        HVX_Vector vsf3 = hvx_vec_mul_f32_f32(vx3, v_inv3);

        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(hvx_vec_i16_from_sf_rnd_sat_shuff(vsf0, vsf2), v_min_127), v_max_127);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(hvx_vec_i16_from_sf_rnd_sat_shuff(vsf1, vsf3), v_min_127), v_max_127);
        HVX_Vector v_u8 = Q6_V_vxor_VV(Q6_Vb_vshuffe_VbVb(vh13, vh02), v_zp_128);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_SIZE))[rg] = v_u8;
    }
}

// Quantize one superblock g (k_group columns, multiple of 32) of a 4-row FP16 group into W8A8 SM 64x32 tiles.
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

    const uint32_t k_end       = k_start + k_group;
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    HVX_Vector vmax0 = Q6_V_vzero();
    HVX_Vector vmax1 = Q6_V_vzero();
    HVX_Vector vmax2 = Q6_V_vzero();
    HVX_Vector vmax3 = Q6_V_vzero();

    for (uint32_t c = k_start; c < k_end; c += 32) {
        uint32_t valid_b = (c + 32 <= k_valid) ? 64u : ((c < k_valid) ? (k_valid - c) * 2u : 0u);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(valid_b);
        HVX_Vector vh0 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p0 + c), Q6_V_vzero());
        HVX_Vector vh1 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p1 + c), Q6_V_vzero());
        HVX_Vector vh2 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p2 + c), Q6_V_vzero());
        HVX_Vector vh3 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p3 + c), Q6_V_vzero());

        HVX_Vector vx0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh0, vh0, -2)), v_one_f16)));
        HVX_Vector vx1 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh1, vh1, -2)), v_one_f16)));
        HVX_Vector vx2 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh2, vh2, -2)), v_one_f16)));
        HVX_Vector vx3 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh3, vh3, -2)), v_one_f16)));

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

    const HVX_Vector v_min_127 = Q6_Vh_vsplat_R(-127);
    const HVX_Vector v_max_127 = Q6_Vh_vsplat_R(127);
    const HVX_Vector v_zp_128  = Q6_Vb_vsplat_R(0x80);

    for (uint32_t c = k_start; c < k_end; c += 32) {
        uint32_t valid_b = (c + 32 <= k_valid) ? 64u : ((c < k_valid) ? (k_valid - c) * 2u : 0u);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(valid_b);
        HVX_Vector vh0 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p0 + c), Q6_V_vzero());
        HVX_Vector vh1 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p1 + c), Q6_V_vzero());
        HVX_Vector vh2 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p2 + c), Q6_V_vzero());
        HVX_Vector vh3 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p3 + c), Q6_V_vzero());

        HVX_Vector vx0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh0, vh0, -2)), v_one_f16)));
        HVX_Vector vx1 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh1, vh1, -2)), v_one_f16)));
        HVX_Vector vx2 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh2, vh2, -2)), v_one_f16)));
        HVX_Vector vx3 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh3, vh3, -2)), v_one_f16)));

        HVX_Vector vsf0 = hvx_vec_mul_f32_f32(vx0, v_inv0);
        HVX_Vector vsf1 = hvx_vec_mul_f32_f32(vx1, v_inv1);
        HVX_Vector vsf2 = hvx_vec_mul_f32_f32(vx2, v_inv2);
        HVX_Vector vsf3 = hvx_vec_mul_f32_f32(vx3, v_inv3);

        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(hvx_vec_i16_from_sf_rnd_sat_shuff(vsf0, vsf2), v_min_127), v_max_127);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(hvx_vec_i16_from_sf_rnd_sat_shuff(vsf1, vsf3), v_min_127), v_max_127);
        HVX_Vector v_u8 = Q6_V_vxor_VV(Q6_Vb_vshuffe_VbVb(vh13, vh02), v_zp_128);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_SIZE))[rg] = v_u8;
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

    const uint32_t k_end = k_start + k_group;
    HVX_Vector vmax0 = Q6_V_vzero();
    HVX_Vector vmax1 = Q6_V_vzero();

    for (uint32_t c = k_start; c < k_end; c += 32) {
        HVX_Vector vx0 = *(const HVX_UVector *) (p0 + c);
        HVX_Vector vx1 = *(const HVX_UVector *) (p1 + c);
        if (c + 32 > k_valid) {
            uint32_t rem = (c < k_valid) ? (k_valid - c) : 0;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
            vx0 = Q6_V_vmux_QVV(mask, vx0, Q6_V_vzero());
            vx1 = Q6_V_vmux_QVV(mask, vx1, Q6_V_vzero());
        }
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

    const HVX_Vector v_zp_32768 = Q6_Vh_vsplat_R(0x8000);

    for (uint32_t c = k_start; c < k_end; c += 32) {
        HVX_Vector vx0 = *(const HVX_UVector *) (p0 + c);
        HVX_Vector vx1 = *(const HVX_UVector *) (p1 + c);
        if (c + 32 > k_valid) {
            uint32_t rem = (c < k_valid) ? (k_valid - c) : 0;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
            vx0 = Q6_V_vmux_QVV(mask, vx0, Q6_V_vzero());
            vx1 = Q6_V_vmux_QVV(mask, vx1, Q6_V_vzero());
        }
        HVX_Vector vsf0  = hvx_vec_mul_f32_f32(vx0, v_inv0);
        HVX_Vector vsf1  = hvx_vec_mul_f32_f32(vx1, v_inv1);
        HVX_Vector v_u16 = Q6_V_vxor_VV(hvx_vec_i16_from_sf_rnd_sat_shuff(vsf0, vsf1), v_zp_32768);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_N_ELMS))[r1] = v_u16;
    }
}

// Quantize one superblock g (k_group columns, multiple of 32) of a 2-row FP16 pair into W8A16 SM 32x32 uint16 tiles.
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

    const uint32_t k_end       = k_start + k_group;
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    HVX_Vector vmax0 = Q6_V_vzero();
    HVX_Vector vmax1 = Q6_V_vzero();

    for (uint32_t c = k_start; c < k_end; c += 32) {
        uint32_t valid_b = (c + 32 <= k_valid) ? 64u : ((c < k_valid) ? (k_valid - c) * 2u : 0u);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(valid_b);
        HVX_Vector vh0 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p0 + c), Q6_V_vzero());
        HVX_Vector vh1 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p1 + c), Q6_V_vzero());

        HVX_Vector vx0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh0, vh0, -2)), v_one_f16)));
        HVX_Vector vx1 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh1, vh1, -2)), v_one_f16)));

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

    const HVX_Vector v_zp_32768 = Q6_Vh_vsplat_R(0x8000);

    for (uint32_t c = k_start; c < k_end; c += 32) {
        uint32_t valid_b = (c + 32 <= k_valid) ? 64u : ((c < k_valid) ? (k_valid - c) * 2u : 0u);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(valid_b);
        HVX_Vector vh0 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p0 + c), Q6_V_vzero());
        HVX_Vector vh1 = Q6_V_vmux_QVV(mask, *(const HVX_UVector *) (p1 + c), Q6_V_vzero());

        HVX_Vector vx0   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh0, vh0, -2)), v_one_f16)));
        HVX_Vector vx1   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(Q6_W_vshuff_VVR(vh1, vh1, -2)), v_one_f16)));
        HVX_Vector vsf0  = hvx_vec_mul_f32_f32(vx0, v_inv0);
        HVX_Vector vsf1  = hvx_vec_mul_f32_f32(vx1, v_inv1);
        HVX_Vector v_u16 = Q6_V_vxor_VV(hvx_vec_i16_from_sf_rnd_sat_shuff(vsf0, vsf1), v_zp_32768);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_N_ELMS))[r1] = v_u16;
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

    const uint32_t cvt_ctrl_hi = hmx_cvt_ctrl(0, 0, 0, 1);
    const uint32_t cvt_ctrl_lo = hmx_cvt_ctrl(1, 0, 1, 0);
    const uint32_t out_sm_rng1 = hmx_act_sm_range(1);

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
                asm volatile(HMX_STORE_CVT_SM("%0", "%1") : : "r"(out_hi), "r"(out_sm_rng1) : "memory");

                asm volatile(HMX_CVT_UB_ACC("%0") : : "r"(cvt_ctrl_lo) : "memory");
                asm volatile(HMX_STORE_CVT_SM("%0", "%1") : : "r"(out_lo), "r"(out_sm_rng1) : "memory");
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

// HVX Epilogue for W8A8 SM output tiles -> FP32 DDR output with superblock scale accumulation.
static void transfer_output_chunk_w8a8_sm_to_fp32_col_chunk(
        float * restrict dst,
        const float * restrict src2,
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
        uint32_t dst_cols) {

    const size_t total_col_tiles = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_64  = mc_pad / 64;
    const size_t group_stride    = n_row_tiles_64 * total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t row_tile_stride = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t c0_first        = c_first / 32;

    const HVX_Vector sign_80 = Q6_Vb_vsplat_R(0x80);
    const size_t limit_c         = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned = (limit_c & ~31u);

    for (size_t r = 0; r < n_rows; r += 4) {
        const size_t r_idx0     = start_row + r;
        const size_t r0         = r_idx0 >> 6;
        const size_t rg         = (r_idx0 & 63) >> 2;
        const size_t r_vec_idx  = r_idx0 >> 5;
        const size_t r_byte_off = (r_idx0 & 31) * sizeof(float);

        const HVX_Vector v_da_g0  = Q6_V_vror_VR(((const HVX_Vector *) vtcm_act_scales)[r_vec_idx], r_byte_off);
        const HVX_Vector v_da0_g0 = hvx_vec_repl4(v_da_g0);
        const HVX_Vector v_da1_g0 = hvx_vec_repl4(Q6_V_vror_VR(v_da_g0, 4));
        const HVX_Vector v_da2_g0 = hvx_vec_repl4(Q6_V_vror_VR(v_da_g0, 8));
        const HVX_Vector v_da3_g0 = hvx_vec_repl4(Q6_V_vror_VR(v_da_g0, 12));

        float * out_row_base       = dst + r * dst_stride;
        const float * src2_row_base = src2 ? (src2 + r * src2_stride) : NULL;

        for (size_t c = 0; c < limit_c; c += 32) {
            const size_t c0      = c >> 5;
            const size_t c0_glob = c0_first + c0;

            HVX_Vector v_acc0 = Q6_V_vzero();
            HVX_Vector v_acc1 = Q6_V_vzero();
            HVX_Vector v_acc2 = Q6_V_vzero();
            HVX_Vector v_acc3 = Q6_V_vzero();

            for (uint32_t g = 0; g < n_groups; ++g) {
                const uint8_t * tile_pair = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * 2 * HTP_MM_HMX_TILE_SIZE;
                const HVX_Vector v_hi_s8  = Q6_V_vxor_VV(((const HVX_Vector *) tile_pair)[rg], sign_80);
                const HVX_Vector v_lo_u8  = ((const HVX_Vector *) (tile_pair + HTP_MM_HMX_TILE_SIZE))[rg];

                HVX_VectorPair vp_i16      = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                HVX_VectorPair vp_even_odd = Q6_W_vdeal_VVR(Q6_V_hi_W(vp_i16), Q6_V_lo_W(vp_i16), -2);
                HVX_VectorPair vp_i32_02   = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(Q6_V_lo_W(vp_even_odd)));
                HVX_VectorPair vp_i32_13   = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(Q6_V_hi_W(vp_even_odd)));

                HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_02));
                HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_02));
                HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_13));
                HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_13));

                const HVX_Vector v_dw = *(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32);
                HVX_Vector v_da0 = v_da0_g0;
                HVX_Vector v_da1 = v_da1_g0;
                HVX_Vector v_da2 = v_da2_g0;
                HVX_Vector v_da3 = v_da3_g0;
                if (g > 0) {
                    const HVX_Vector v_da_v = Q6_V_vror_VR(((const HVX_Vector *) (vtcm_act_scales + (size_t) g * mc_pad))[r_vec_idx], r_byte_off);
                    v_da0 = hvx_vec_repl4(v_da_v);
                    v_da1 = hvx_vec_repl4(Q6_V_vror_VR(v_da_v, 4));
                    v_da2 = hvx_vec_repl4(Q6_V_vror_VR(v_da_v, 8));
                    v_da3 = hvx_vec_repl4(Q6_V_vror_VR(v_da_v, 12));
                }

                HVX_Vector v_sc0 = hvx_vec_mul_f32_f32(v_dw, v_da0);
                HVX_Vector v_sc1 = hvx_vec_mul_f32_f32(v_dw, v_da1);
                HVX_Vector v_sc2 = hvx_vec_mul_f32_f32(v_dw, v_da2);
                HVX_Vector v_sc3 = hvx_vec_mul_f32_f32(v_dw, v_da3);

                v_acc0 = hvx_vec_add_f32_f32(v_acc0, hvx_vec_mul_f32_f32(v_f32_0, v_sc0));
                v_acc1 = hvx_vec_add_f32_f32(v_acc1, hvx_vec_mul_f32_f32(v_f32_1, v_sc1));
                v_acc2 = hvx_vec_add_f32_f32(v_acc2, hvx_vec_mul_f32_f32(v_f32_2, v_sc2));
                v_acc3 = hvx_vec_add_f32_f32(v_acc3, hvx_vec_mul_f32_f32(v_f32_3, v_sc3));
            }

            if (c < limit_c_aligned) {
                if (src2_row_base) {
                    v_acc0 = hvx_vec_add_f32_f32(v_acc0, hvx_vmemu(src2_row_base + c + 0 * src2_stride));
                }
                hvx_vmemu(out_row_base + c + 0 * dst_stride) = v_acc0;

                if (r + 1 < n_rows) {
                    if (src2_row_base) {
                        v_acc1 = hvx_vec_add_f32_f32(v_acc1, hvx_vmemu(src2_row_base + c + 1 * src2_stride));
                    }
                    hvx_vmemu(out_row_base + c + 1 * dst_stride) = v_acc1;
                }
                if (r + 2 < n_rows) {
                    if (src2_row_base) {
                        v_acc2 = hvx_vec_add_f32_f32(v_acc2, hvx_vmemu(src2_row_base + c + 2 * src2_stride));
                    }
                    hvx_vmemu(out_row_base + c + 2 * dst_stride) = v_acc2;
                }
                if (r + 3 < n_rows) {
                    if (src2_row_base) {
                        v_acc3 = hvx_vec_add_f32_f32(v_acc3, hvx_vmemu(src2_row_base + c + 3 * src2_stride));
                    }
                    hvx_vmemu(out_row_base + c + 3 * dst_stride) = v_acc3;
                }
            } else {
                const size_t valid_bytes = (limit_c - c) * sizeof(float);
                if (src2_row_base) {
                    v_acc0 = hvx_vec_add_f32_f32(v_acc0, hvx_vmemu(src2_row_base + c + 0 * src2_stride));
                }
                hvx_vec_store_u(out_row_base + c + 0 * dst_stride, valid_bytes, v_acc0);

                if (r + 1 < n_rows) {
                    if (src2_row_base) {
                        v_acc1 = hvx_vec_add_f32_f32(v_acc1, hvx_vmemu(src2_row_base + c + 1 * src2_stride));
                    }
                    hvx_vec_store_u(out_row_base + c + 1 * dst_stride, valid_bytes, v_acc1);
                }
                if (r + 2 < n_rows) {
                    if (src2_row_base) {
                        v_acc2 = hvx_vec_add_f32_f32(v_acc2, hvx_vmemu(src2_row_base + c + 2 * src2_stride));
                    }
                    hvx_vec_store_u(out_row_base + c + 2 * dst_stride, valid_bytes, v_acc2);
                }
                if (r + 3 < n_rows) {
                    if (src2_row_base) {
                        v_acc3 = hvx_vec_add_f32_f32(v_acc3, hvx_vmemu(src2_row_base + c + 3 * src2_stride));
                    }
                    hvx_vec_store_u(out_row_base + c + 3 * dst_stride, valid_bytes, v_acc3);
                }
            }
        }
    }
}

// HVX Epilogue for W8A16 :2x1 SM output tiles -> FP32 DDR output with superblock scale accumulation.
static void transfer_output_chunk_w8a16_sm_to_fp32_col_chunk(
        float * restrict dst,
        const float * restrict src2,
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
        uint32_t dst_cols) {

    const size_t total_col_tiles = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_32  = mc_pad / 32;
    const size_t group_stride    = n_row_tiles_32 * total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t row_tile_stride = total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t c0_first        = c_first / 32;

    const HVX_Vector sign_8000   = Q6_Vh_vsplat_R(0x8000);
    const size_t limit_c         = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned = (limit_c & ~31u);

    for (size_t r = 0; r < n_rows; r += 2) {
        const size_t r_idx0     = start_row + r;
        const size_t r0         = r_idx0 >> 5;
        const size_t r1         = (r_idx0 & 31) >> 1;
        const size_t r_byte_off = (r_idx0 & 31) * sizeof(float);

        const HVX_Vector v_da_g0  = Q6_V_vror_VR(((const HVX_Vector *) vtcm_act_scales)[r0], r_byte_off);
        const HVX_Vector v_da0_g0 = hvx_vec_repl4(v_da_g0);
        const HVX_Vector v_da1_g0 = hvx_vec_repl4(Q6_V_vror_VR(v_da_g0, 4));

        float * out_row_base        = dst + r * dst_stride;
        const float * src2_row_base = src2 ? (src2 + r * src2_stride) : NULL;

        for (size_t c = 0; c < limit_c; c += 32) {
            const size_t c0      = c >> 5;
            const size_t c0_glob = c0_first + c0;

            HVX_Vector v_acc0 = Q6_V_vzero();
            HVX_Vector v_acc1 = Q6_V_vzero();

            for (uint32_t g = 0; g < n_groups; ++g) {
                const uint16_t * tile   = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * HTP_MM_HMX_TILE_N_ELMS;
                const HVX_Vector v_i16  = Q6_V_vxor_VV(((const HVX_Vector *) tile)[r1], sign_8000);
                HVX_VectorPair vp_i32   = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));

                const HVX_Vector v_dw = *(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32);
                HVX_Vector v_da0 = v_da0_g0;
                HVX_Vector v_da1 = v_da1_g0;
                if (g > 0) {
                    const HVX_Vector v_da_v = Q6_V_vror_VR(((const HVX_Vector *) (vtcm_act_scales + (size_t) g * mc_pad))[r0], r_byte_off);
                    v_da0 = hvx_vec_repl4(v_da_v);
                    v_da1 = hvx_vec_repl4(Q6_V_vror_VR(v_da_v, 4));
                }

                HVX_Vector v_sc0 = hvx_vec_mul_f32_f32(v_dw, v_da0);
                HVX_Vector v_sc1 = hvx_vec_mul_f32_f32(v_dw, v_da1);
                v_acc0 = hvx_vec_add_f32_f32(v_acc0, hvx_vec_mul_f32_f32(v_f32_0, v_sc0));
                v_acc1 = hvx_vec_add_f32_f32(v_acc1, hvx_vec_mul_f32_f32(v_f32_1, v_sc1));
            }

            if (c < limit_c_aligned) {
                if (src2_row_base) {
                    v_acc0 = hvx_vec_add_f32_f32(v_acc0, hvx_vmemu(src2_row_base + c));
                }
                hvx_vmemu(out_row_base + c) = v_acc0;

                if (r + 1 < n_rows) {
                    if (src2_row_base) {
                        v_acc1 = hvx_vec_add_f32_f32(v_acc1, hvx_vmemu(src2_row_base + c + src2_stride));
                    }
                    hvx_vmemu(out_row_base + c + dst_stride) = v_acc1;
                }
            } else {
                const size_t valid_bytes = (limit_c - c) * sizeof(float);
                if (src2_row_base) {
                    v_acc0 = hvx_vec_add_f32_f32(v_acc0, hvx_vmemu(src2_row_base + c));
                }
                hvx_vec_store_u(out_row_base + c, valid_bytes, v_acc0);

                if (r + 1 < n_rows) {
                    if (src2_row_base) {
                        v_acc1 = hvx_vec_add_f32_f32(v_acc1, hvx_vmemu(src2_row_base + c + src2_stride));
                    }
                    hvx_vec_store_u(out_row_base + c + dst_stride, valid_bytes, v_acc1);
                }
            }
        }
    }
}
