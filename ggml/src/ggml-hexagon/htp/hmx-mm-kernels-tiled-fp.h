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

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q1_0);
    const HVX_Vector      v_neg_one   = Q6_Vb_vsplat_R(-1);
    const HVX_Vector      v_pos_one   = Q6_Vb_vsplat_R(1);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * (4 * HTP_MM_HMX_TILE_N_ELMS);

    for (uint32_t t = start_tile; t < end_tile; t++) {
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
        tile_src += tile_stride;
        dst_ptr  += 4 * HTP_MM_HMX_TILE_N_ELMS;
    }
}

static inline void dequantize_q4_0_fp16_tile_step(
        const uint8_t * restrict tile_src,
        __fp16 * restrict dst_ptr,
        HVX_Vector v_scale,
        HVX_Vector mask_h4,
        HVX_Vector i8) {

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        HVX_Vector     vq     = hvx_vmem(tile_src + i * 128);
        HVX_Vector     v_lo   = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vq, mask_h4), i8);
        HVX_Vector     v_hi   = Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(vq, 4), i8);
        HVX_VectorPair vp_lo  = Q6_Wh_vunpack_Vb(v_lo);
        HVX_VectorPair vp_k0  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp_lo), Q6_V_lo_W(vp_lo), -4);
        HVX_VectorPair vp_hi  = Q6_Wh_vunpack_Vb(v_hi);
        HVX_VectorPair vp_k1  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp_hi), Q6_V_lo_W(vp_hi), -4);

        hvx_vmem(dst_ptr + (4 * i + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale));
        hvx_vmem(dst_ptr + (4 * i + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale));
        hvx_vmem(dst_ptr + (4 * i + 2) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale));
        hvx_vmem(dst_ptr + (4 * i + 3) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale));
    }
}

// Dequantize a single tile from tiled weight data (already in VTCM) to tile-major FP16.
static void dequantize_tiled_weight_to_fp16_task_q4_0(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    if (start_tile >= end_tile) {
        return;
    }

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q4_0);
    const HVX_Vector      mask_h4     = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector      i8          = Q6_Vb_vsplat_R(8);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    HVX_Vector v_sc    = hvx_vmem(tile_src + 512);
    HVX_Vector v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t t = start_tile; t + 1 < end_tile; t++) {
        HVX_Vector v_sc_next    = hvx_vmem(tile_src + tile_stride + 512);
        HVX_Vector v_scale_next = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_next, v_sc_next, -2));
        dequantize_q4_0_fp16_tile_step(tile_src, dst_ptr, v_scale, mask_h4, i8);
        v_scale   = v_scale_next;
        tile_src += tile_stride;
        dst_ptr  += HTP_MM_HMX_TILE_N_ELMS;
    }
    dequantize_q4_0_fp16_tile_step(tile_src, dst_ptr, v_scale, mask_h4, i8);
}

static inline void dequantize_q4_1_fp16_tile_step(
        const uint8_t * restrict tile_src,
        __fp16 * restrict dst_ptr,
        HVX_Vector v_scale,
        HVX_Vector v_offset,
        HVX_Vector mask_h4) {

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        HVX_Vector     vq    = hvx_vmem(tile_src + i * 128);
        HVX_Vector     v_lo  = Q6_V_vand_VV(vq, mask_h4);
        HVX_Vector     v_hi  = Q6_Vub_vlsr_VubR(vq, 4);
        HVX_VectorPair vp_lo = Q6_Wh_vunpack_Vb(v_lo);
        HVX_VectorPair vp_k0 = Q6_W_vdeal_VVR(Q6_V_hi_W(vp_lo), Q6_V_lo_W(vp_lo), -4);
        HVX_VectorPair vp_hi = Q6_Wh_vunpack_Vb(v_hi);
        HVX_VectorPair vp_k1 = Q6_W_vdeal_VVR(Q6_V_hi_W(vp_hi), Q6_V_lo_W(vp_hi), -4);

        hvx_vmem(dst_ptr + (4 * i + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale), v_offset));
        hvx_vmem(dst_ptr + (4 * i + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale), v_offset));
        hvx_vmem(dst_ptr + (4 * i + 2) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale), v_offset));
        hvx_vmem(dst_ptr + (4 * i + 3) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale), v_offset));
    }
}

static void dequantize_tiled_weight_to_fp16_task_q4_1(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    if (start_tile >= end_tile) {
        return;
    }

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q4_1);
    const HVX_Vector      mask_h4     = Q6_Vb_vsplat_R(0x0F);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    HVX_Vector     v_dm     = hvx_vmem(tile_src + 512);
    HVX_VectorPair dm_deal  = Q6_W_vdeal_VVR(v_dm, v_dm, -2);
    HVX_Vector     v_scale  = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_lo_W(dm_deal), Q6_V_lo_W(dm_deal), -2));
    HVX_Vector     v_offset = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(dm_deal), Q6_V_hi_W(dm_deal), -2));

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t t = start_tile; t + 1 < end_tile; t++) {
        HVX_Vector     v_dm_next     = hvx_vmem(tile_src + tile_stride + 512);
        HVX_VectorPair dm_deal_next  = Q6_W_vdeal_VVR(v_dm_next, v_dm_next, -2);
        HVX_Vector     v_scale_next  = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_lo_W(dm_deal_next), Q6_V_lo_W(dm_deal_next), -2));
        HVX_Vector     v_offset_next = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(dm_deal_next), Q6_V_hi_W(dm_deal_next), -2));
        dequantize_q4_1_fp16_tile_step(tile_src, dst_ptr, v_scale, v_offset, mask_h4);
        v_scale   = v_scale_next;
        v_offset  = v_offset_next;
        tile_src += tile_stride;
        dst_ptr  += HTP_MM_HMX_TILE_N_ELMS;
    }
    dequantize_q4_1_fp16_tile_step(tile_src, dst_ptr, v_scale, v_offset, mask_h4);
}

static inline void dequantize_4bit_lut_fp16_tile_step(
        const uint8_t * restrict tile_src,
        __fp16 * restrict dst_ptr,
        HVX_Vector v_scale,
        HVX_Vector vlut_cvt,
        HVX_Vector mask_h4) {

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        HVX_Vector     vq    = hvx_vmem(tile_src + i * 128);
        HVX_Vector     v_lo  = Q6_V_vand_VV(vq, mask_h4);
        HVX_Vector     v_hi  = Q6_Vub_vlsr_VubR(vq, 4);
        HVX_VectorPair vp_d  = Q6_W_vdeal_VVR(v_hi, v_lo, -2);
        HVX_VectorPair vp_s  = Q6_W_vshuff_VVR(Q6_V_hi_W(vp_d), Q6_V_lo_W(vp_d), -1);
        HVX_VectorPair vp_lo = Q6_Wh_vlut16_VbVhR(Q6_V_lo_W(vp_s), vlut_cvt, 0);
        HVX_VectorPair vp_hi = Q6_Wh_vlut16_VbVhR(Q6_V_hi_W(vp_s), vlut_cvt, 0);

        hvx_vmem(dst_ptr + (4 * i + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo), v_scale));
        hvx_vmem(dst_ptr + (4 * i + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo), v_scale));
        hvx_vmem(dst_ptr + (4 * i + 2) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi), v_scale));
        hvx_vmem(dst_ptr + (4 * i + 3) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi), v_scale));
    }
}

static void dequantize_tiled_weight_to_fp16_task_iq4_nl(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    if (start_tile >= end_tile) {
        return;
    }

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_IQ4_NL);
    const HVX_Vector      mask_h4     = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector      vlut_cvt    = hvx_vmem(iq4_nl_to_fp16_lut);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    HVX_Vector v_sc    = hvx_vmem(tile_src + 512);
    HVX_Vector v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t t = start_tile; t + 1 < end_tile; t++) {
        HVX_Vector v_sc_next    = hvx_vmem(tile_src + tile_stride + 512);
        HVX_Vector v_scale_next = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_next, v_sc_next, -2));
        dequantize_4bit_lut_fp16_tile_step(tile_src, dst_ptr, v_scale, vlut_cvt, mask_h4);
        v_scale   = v_scale_next;
        tile_src += tile_stride;
        dst_ptr  += HTP_MM_HMX_TILE_N_ELMS;
    }
    dequantize_4bit_lut_fp16_tile_step(tile_src, dst_ptr, v_scale, vlut_cvt, mask_h4);
}

static inline HVX_Vector mxfp4_e8m0_to_f16_scale(HVX_Vector v_raw) {
    HVX_Vector vh = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v_raw));
    vh = Q6_Vh_vsub_VhVh(vh, Q6_Vh_vsplat_R(112));
    vh = Q6_Vh_vmax_VhVh(vh, Q6_V_vzero());
    vh = Q6_Vh_vmin_VhVh(vh, Q6_Vh_vsplat_R(30));
    vh = Q6_Vh_vasl_VhR(vh, 10);
    return Q6_V_lo_W(Q6_W_vshuff_VVR(vh, vh, -2));
}

static void dequantize_tiled_weight_to_fp16_task_mxfp4(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    if (start_tile >= end_tile) {
        return;
    }

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_MXFP4);
    const HVX_Vector      mask_h4     = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector      vlut_cvt    = hvx_vmem(mxfp4_to_fp16_lut);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    HVX_Vector v_scale = mxfp4_e8m0_to_f16_scale(hvx_vmem(tile_src + 512));

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t t = start_tile; t + 1 < end_tile; t++) {
        HVX_Vector v_scale_next = mxfp4_e8m0_to_f16_scale(hvx_vmem(tile_src + tile_stride + 512));
        dequantize_4bit_lut_fp16_tile_step(tile_src, dst_ptr, v_scale, vlut_cvt, mask_h4);
        v_scale   = v_scale_next;
        tile_src += tile_stride;
        dst_ptr  += HTP_MM_HMX_TILE_N_ELMS;
    }
    dequantize_4bit_lut_fp16_tile_step(tile_src, dst_ptr, v_scale, vlut_cvt, mask_h4);
}

static inline void dequantize_q8_0_fp16_tile_step(
        const uint8_t * restrict tile_src,
        __fp16 * restrict dst_ptr,
        HVX_Vector v_scale) {

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        HVX_VectorPair vp16 = Q6_Wh_vunpack_Vb(hvx_vmem(tile_src + i * 128));
        HVX_VectorPair vp_k = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);
        hvx_vmem(dst_ptr + (2 * i + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
        hvx_vmem(dst_ptr + (2 * i + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
    }
}

static void dequantize_tiled_weight_to_fp16_task_q8_0(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    if (start_tile >= end_tile) {
        return;
    }

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q8_0);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    HVX_Vector v_sc    = hvx_vmem(tile_src + 1024);
    HVX_Vector v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t t = start_tile; t + 1 < end_tile; t++) {
        HVX_Vector v_sc_next    = hvx_vmem(tile_src + tile_stride + 1024);
        HVX_Vector v_scale_next = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_next, v_sc_next, -2));
        dequantize_q8_0_fp16_tile_step(tile_src, dst_ptr, v_scale);
        v_scale   = v_scale_next;
        tile_src += tile_stride;
        dst_ptr  += HTP_MM_HMX_TILE_N_ELMS;
    }
    dequantize_q8_0_fp16_tile_step(tile_src, dst_ptr, v_scale);
}

static inline void dequantize_q5_k_fp16_tile_step(
        const uint8_t * restrict tile_src,
        __fp16 * restrict dst_ptr,
        HVX_Vector v_scale,
        HVX_Vector v_offset,
        HVX_Vector mask_h4) {

    HVX_Vector v_plane = hvx_vmem(tile_src + 640);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        HVX_Vector     vq    = hvx_vmem(tile_src + i * 128);
        HVX_Vector     v_lo  = hvx_q5k_or_hibit(Q6_V_vand_VV(vq, mask_h4), v_plane, 2 * i + 0);
        HVX_Vector     v_hi  = hvx_q5k_or_hibit(Q6_Vub_vlsr_VubR(vq, 4),   v_plane, 2 * i + 1);
        HVX_VectorPair vp_lo = Q6_Wh_vunpack_Vb(v_lo);
        HVX_VectorPair vp_k0 = Q6_W_vdeal_VVR(Q6_V_hi_W(vp_lo), Q6_V_lo_W(vp_lo), -4);
        HVX_VectorPair vp_hi = Q6_Wh_vunpack_Vb(v_hi);
        HVX_VectorPair vp_k1 = Q6_W_vdeal_VVR(Q6_V_hi_W(vp_hi), Q6_V_lo_W(vp_hi), -4);

        hvx_vmem(dst_ptr + (4 * i + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale), v_offset));
        hvx_vmem(dst_ptr + (4 * i + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale), v_offset));
        hvx_vmem(dst_ptr + (4 * i + 2) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale), v_offset));
        hvx_vmem(dst_ptr + (4 * i + 3) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale), v_offset));
    }
}

static void dequantize_tiled_weight_to_fp16_task_q5_k(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    if (start_tile >= end_tile) {
        return;
    }

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q5_K);
    const HVX_Vector      mask_h4     = Q6_Vb_vsplat_R(0x0F);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    HVX_Vector     v_dm     = hvx_vmem(tile_src + 512);
    HVX_VectorPair dm_deal  = Q6_W_vdeal_VVR(v_dm, v_dm, -2);
    HVX_Vector     v_scale  = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_lo_W(dm_deal), Q6_V_lo_W(dm_deal), -2));
    HVX_Vector     v_offset = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(dm_deal), Q6_V_hi_W(dm_deal), -2));

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t t = start_tile; t + 1 < end_tile; t++) {
        HVX_Vector     v_dm_next     = hvx_vmem(tile_src + tile_stride + 512);
        HVX_VectorPair dm_deal_next  = Q6_W_vdeal_VVR(v_dm_next, v_dm_next, -2);
        HVX_Vector     v_scale_next  = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_lo_W(dm_deal_next), Q6_V_lo_W(dm_deal_next), -2));
        HVX_Vector     v_offset_next = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(dm_deal_next), Q6_V_hi_W(dm_deal_next), -2));
        dequantize_q5_k_fp16_tile_step(tile_src, dst_ptr, v_scale, v_offset, mask_h4);
        v_scale   = v_scale_next;
        v_offset  = v_offset_next;
        tile_src += tile_stride;
        dst_ptr  += HTP_MM_HMX_TILE_N_ELMS;
    }
    dequantize_q5_k_fp16_tile_step(tile_src, dst_ptr, v_scale, v_offset, mask_h4);
}

static inline void dequantize_q5_0_fp16_tile_step(
        const uint8_t * restrict tile_src,
        __fp16 * restrict dst_ptr,
        HVX_Vector v_scale,
        HVX_Vector mask_h4) {

    HVX_Vector v_plane = hvx_vmem(tile_src + 512);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        HVX_Vector     vq    = hvx_vmem(tile_src + i * 128);
        HVX_Vector     v_lo  = hvx_q5_0_or_neg(Q6_V_vand_VV(vq, mask_h4), v_plane, 2 * i + 0);
        HVX_Vector     v_hi  = hvx_q5_0_or_neg(Q6_Vub_vlsr_VubR(vq, 4),   v_plane, 2 * i + 1);
        HVX_VectorPair vp_lo = Q6_Wh_vunpack_Vb(v_lo);
        HVX_VectorPair vp_k0 = Q6_W_vdeal_VVR(Q6_V_hi_W(vp_lo), Q6_V_lo_W(vp_lo), -4);
        HVX_VectorPair vp_hi = Q6_Wh_vunpack_Vb(v_hi);
        HVX_VectorPair vp_k1 = Q6_W_vdeal_VVR(Q6_V_hi_W(vp_hi), Q6_V_lo_W(vp_hi), -4);

        hvx_vmem(dst_ptr + (4 * i + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale));
        hvx_vmem(dst_ptr + (4 * i + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale));
        hvx_vmem(dst_ptr + (4 * i + 2) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale));
        hvx_vmem(dst_ptr + (4 * i + 3) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale));
    }
}

static void dequantize_tiled_weight_to_fp16_task_q5_0(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    if (start_tile >= end_tile) {
        return;
    }

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q5_0);
    const HVX_Vector      mask_h4     = Q6_Vb_vsplat_R(0x0F);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    HVX_Vector v_sc    = hvx_vmem(tile_src + 640);
    HVX_Vector v_scale = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t t = start_tile; t + 1 < end_tile; t++) {
        HVX_Vector v_sc_next    = hvx_vmem(tile_src + tile_stride + 640);
        HVX_Vector v_scale_next = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_next, v_sc_next, -2));
        dequantize_q5_0_fp16_tile_step(tile_src, dst_ptr, v_scale, mask_h4);
        v_scale   = v_scale_next;
        tile_src += tile_stride;
        dst_ptr  += HTP_MM_HMX_TILE_N_ELMS;
    }
    dequantize_q5_0_fp16_tile_step(tile_src, dst_ptr, v_scale, mask_h4);
}

// Q6_K stores 6-bit weights and one fp16 scale per 16 k, see HTP_TENSOR_TILE_SIZE_Q6_K.
// A k-group holds 4 k per row, the HMX tile holds 2, so each group is dealt into two tiles.
static void dequantize_tiled_weight_to_fp16_task_q6_k(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q6_K);
    const HVX_Vector      mask_0f     = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector      mask_03     = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector      i32         = Q6_Vb_vsplat_R(32);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const HVX_Vector * vptr = (const HVX_Vector *) tile_src;

        HVX_Vector v_sc      = vptr[6];
        HVX_Vector v_sc_k16  = Q6_V_vror_VR(v_sc, 64);
        HVX_Vector v_scale_k0  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));
        HVX_Vector v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));

        #pragma unroll
        for (int g = 0; g < 8; g++) {
            const HVX_Vector v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;
            HVX_Vector       v_q     = unpack_q6_k_group(vptr, g, mask_0f, mask_03, i32);
            HVX_VectorPair   vp16    = Q6_Wh_vunpack_Vb(v_q);
            HVX_VectorPair   vp_k    = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);

            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
        }
        tile_src += tile_stride;
        dst_ptr  += HTP_MM_HMX_TILE_N_ELMS;
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

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_NVFP4);
    const HVX_Vector      mask_h4     = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector      vlut_cvt    = hvx_vmem(mxfp4_to_fp16_lut);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    for (uint32_t t = start_tile; t < end_tile; t++) {
        HVX_Vector v_d = nvfp4_ue4m3_to_f16(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmem(tile_src + NVFP4_SCALE_OFF))));
        HVX_VectorPair vp_d = Q6_W_vshuff_VVR(v_d, v_d, -2);
        HVX_Vector v_scale_k0  = Q6_V_lo_W(vp_d);
        HVX_Vector v_scale_k16 = Q6_V_hi_W(vp_d);

        #pragma unroll
        for (int g = 0; g < 4; g++) {
            const HVX_Vector v_scale = (g < 2) ? v_scale_k0 : v_scale_k16;

            HVX_Vector     vq    = hvx_vmem(tile_src + g * 128);
            HVX_Vector     v_lo  = Q6_V_vand_VV(vq, mask_h4);
            HVX_Vector     v_hi  = Q6_Vub_vlsr_VubR(vq, 4);
            HVX_VectorPair vp_d  = Q6_W_vdeal_VVR(v_hi, v_lo, -2);
            HVX_VectorPair vp_s  = Q6_W_vshuff_VVR(Q6_V_hi_W(vp_d), Q6_V_lo_W(vp_d), -1);
            HVX_VectorPair vp_lo = Q6_Wh_vlut16_VbVhR(Q6_V_lo_W(vp_s), vlut_cvt, 0);
            HVX_VectorPair vp_hi = Q6_Wh_vlut16_VbVhR(Q6_V_hi_W(vp_s), vlut_cvt, 0);

            hvx_vmem(dst_ptr + (4 * g + 0) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_lo), v_scale));
            hvx_vmem(dst_ptr + (4 * g + 1) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_lo), v_scale));
            hvx_vmem(dst_ptr + (4 * g + 2) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_hi), v_scale));
            hvx_vmem(dst_ptr + (4 * g + 3) * 64) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_hi), v_scale));
        }
        tile_src += tile_stride;
        dst_ptr  += HTP_MM_HMX_TILE_N_ELMS;
    }
}

// IQ2_XXS: int8 grid weights from the VTCM LUT, scaled by d / 8 * (2 * scale + 1) per row
static void dequantize_tiled_weight_to_fp16_task_iq2_xxs(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_IQ2_XXS);
    HVX_Vector            v_d         = Q6_V_vzero();
    const HVX_Vector      v_eighth    = Q6_Vh_vsplat_R(IQ2XXS_D_SCALE_F16);
    const uint8_t * restrict tile     = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    HVX_Vector v_sgn = Q6_V_vzero();
    if (start_tile < end_tile) {
        v_sgn = iq2xxs_gather_issue(tile);
    }

    for (uint32_t t = start_tile; t < end_tile; t++) {
        // rows hold all k tiles and n_k_tiles is a multiple of 8, so t % 8 is the tile within its super-block
        if (t == start_tile || (t & (IQ2XXS_TILES_PER_SB - 1)) == 0) {
            v_d = iq2xxs_gather_d(tile - (size_t) (t & (IQ2XXS_TILES_PER_SB - 1)) * tile_stride);
        }

        HVX_Vector v_c[2], v_w[2 * IQ2XXS_GROUPS], v_ls;
        iq2xxs_gather_wait(tile, v_c);
        const HVX_Vector v_sgn_cur = v_sgn;
        if (t + 1 < end_tile) {
            v_sgn = iq2xxs_gather_issue(tile + tile_stride);
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
        tile    += tile_stride;
        dst_ptr += HTP_MM_HMX_TILE_N_ELMS;
    }
}

// Q3_K stores 3-bit weights and one fp16 scale per 16 k, see HTP_TENSOR_TILE_SIZE_Q3_K.
static void dequantize_tiled_weight_to_fp16_task_q3_k(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q3_K);
    const HVX_Vector      mask_03     = Q6_Vb_vsplat_R(0x03);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const HVX_Vector * vptr = (const HVX_Vector *) tile_src;

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
        tile_src += tile_stride;
        dst_ptr  += HTP_MM_HMX_TILE_N_ELMS;
    }
}

// Q2_K stores 2-bit weights, the ggml scale bytes and d / -dmin per super-block, see HTP_TENSOR_TILE_SIZE_Q2_K.
static void dequantize_tiled_weight_to_fp16_task_q2_k(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q2_K);
    const HVX_Vector      mask_03     = Q6_Vb_vsplat_R(0x03);
    HVX_Vector            v_d         = Q6_V_vzero();
    HVX_Vector            v_dm        = Q6_V_vzero();
    const uint8_t * restrict tile     = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_ptr  = state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS;

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const HVX_Vector * vptr = (const HVX_Vector *) tile;

        // rows hold all k tiles and n_k_tiles is a multiple of 8, so t % 8 is the tile within its super-block
        if (t == start_tile || (t & 7) == 0) {
            q2_k_load_dmin(tile - (size_t) (t & 7) * tile_stride, &v_d, &v_dm);
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
        tile    += tile_stride;
        dst_ptr += HTP_MM_HMX_TILE_N_ELMS;
    }
}

// Q2_0 stores codes 0..3 (value code - 1) and one fp16 scale per row, see HTP_MM_WEIGHT_TILE_SIZE_Q2_0.
// One packed 64-k tile expands into two consecutive 32-k fp16 tiles.
static void dequantize_tiled_weight_to_fp16_task_q2_0(
        const tiled_dequantize_state_t *state,
        uint32_t start_tile, uint32_t end_tile) {

    const size_t          tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q2_0);
    const HVX_Vector      mask_03     = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector      i1          = Q6_Vb_vsplat_R(1);
    const uint8_t * restrict tile_src = state->src + (size_t) start_tile * tile_stride;
    __fp16 * restrict        dst_base = state->dst + (size_t) start_tile * (2 * HTP_MM_HMX_TILE_N_ELMS);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const HVX_Vector * vptr = (const HVX_Vector *) tile_src;

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
        tile_src += tile_stride;
        dst_base += 2 * HTP_MM_HMX_TILE_N_ELMS;
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
    hvx_gather_sync(state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS);
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
    hvx_gather_sync(state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS);
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
    hvx_gather_sync(state->dst + (size_t) start_tile * HTP_MM_HMX_TILE_N_ELMS);
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

// output : fp16 -> f32 / f16

static inline HVX_Vector hmx_load_bias_f32_32_f16(const void * bias_row_base, size_t c) {
    const __fp16 * p = (const __fp16 *) bias_row_base + c;
    return Q6_V_lo_W(hvx_vec_f16_to_f32(hvx_vmemu(p)));
}

static inline HVX_Vector hmx_load_bias_f32_32_f32(const void * bias_row_base, size_t c) {
    const float * p = (const float *) bias_row_base + c;
    return hvx_vmemu(p);
}

static inline HVX_VectorPair hmx_load_bias_f32_64_f16(const void * bias_row_base, size_t c) {
    const __fp16 * p = (const __fp16 *) bias_row_base + c;
    return hvx_vec_f16_to_f32(hvx_vmemu(p));
}

static inline HVX_VectorPair hmx_load_bias_f32_64_f32(const void * bias_row_base, size_t c) {
    const float * p = (const float *) bias_row_base + c;
    return Q6_W_vcombine_VV(hvx_vmemu(p + 32), hvx_vmemu(p));
}

static inline HVX_Vector hmx_pack_f32_to_f16_clamped(HVX_Vector v0, HVX_Vector v1) {
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    v0 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v0));
    v1 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v1));
    return hvx_vec_f32_to_f16(v0, v1);
}

static inline HVX_Vector hmx_add_bias_to_f16_vec64_f16(HVX_Vector v_f16, const void * bias_row_base, size_t c) {
    HVX_VectorPair vp_dst  = hvx_vec_f16_to_f32(v_f16);
    HVX_VectorPair vp_bias = hmx_load_bias_f32_64_f16(bias_row_base, c);
    HVX_Vector v0 = hvx_vec_add_f32_f32(Q6_V_lo_W(vp_dst), Q6_V_lo_W(vp_bias));
    HVX_Vector v1 = hvx_vec_add_f32_f32(Q6_V_hi_W(vp_dst), Q6_V_hi_W(vp_bias));
    return hmx_pack_f32_to_f16_clamped(v0, v1);
}

static inline HVX_Vector hmx_add_bias_to_f16_vec64_f32(HVX_Vector v_f16, const void * bias_row_base, size_t c) {
    HVX_VectorPair vp_dst  = hvx_vec_f16_to_f32(v_f16);
    HVX_VectorPair vp_bias = hmx_load_bias_f32_64_f32(bias_row_base, c);
    HVX_Vector v0 = hvx_vec_add_f32_f32(Q6_V_lo_W(vp_dst), Q6_V_lo_W(vp_bias));
    HVX_Vector v1 = hvx_vec_add_f32_f32(Q6_V_hi_W(vp_dst), Q6_V_hi_W(vp_bias));
    return hmx_pack_f32_to_f16_clamped(v0, v1);
}

static inline HVX_Vector hmx_add_bias_to_f16_tail_f16(HVX_Vector v_f16, const void * bias_row_base, size_t c, size_t valid_c) {
    HVX_VectorPair vp_dst  = hvx_vec_f16_to_f32(v_f16);
    HVX_VectorPair vp_bias = hmx_load_bias_f32_64_f16(bias_row_base, c);
    HVX_Vector v0 = hvx_vec_add_f32_f32(Q6_V_lo_W(vp_dst), Q6_V_lo_W(vp_bias));
    HVX_Vector v1 = (valid_c > 32) ? hvx_vec_add_f32_f32(Q6_V_hi_W(vp_dst), Q6_V_hi_W(vp_bias)) : Q6_V_vzero();
    return hmx_pack_f32_to_f16_clamped(v0, v1);
}

static inline HVX_Vector hmx_add_bias_to_f16_tail_f32(HVX_Vector v_f16, const void * bias_row_base, size_t c, size_t valid_c) {
    HVX_VectorPair vp_dst = hvx_vec_f16_to_f32(v_f16);
    HVX_Vector v0 = hvx_vec_add_f32_f32(Q6_V_lo_W(vp_dst), hmx_load_bias_f32_32_f32(bias_row_base, c));
    HVX_Vector v1 = (valid_c > 32) ? hvx_vec_add_f32_f32(Q6_V_hi_W(vp_dst), hmx_load_bias_f32_32_f32(bias_row_base, c + 32)) : Q6_V_vzero();
    return hmx_pack_f32_to_f16_clamped(v0, v1);
}

#define HMX_FP16_ADD_BIAS_F32_NONE(v_out, row_ptr, c) (v_out)
#define HMX_FP16_ADD_BIAS_F32_F16(v_out, row_ptr, c)  hvx_vec_add_f32_f32((v_out), hmx_load_bias_f32_32_f16((row_ptr), (c)))
#define HMX_FP16_ADD_BIAS_F32_F32(v_out, row_ptr, c)  hvx_vec_add_f32_f32((v_out), hmx_load_bias_f32_32_f32((row_ptr), (c)))

#define HMX_FP16_ADD_BIAS_F16_64_NONE(v_out, row_ptr, c) (v_out)
#define HMX_FP16_ADD_BIAS_F16_64_F16(v_out, row_ptr, c)  hmx_add_bias_to_f16_vec64_f16((v_out), (row_ptr), (c))
#define HMX_FP16_ADD_BIAS_F16_64_F32(v_out, row_ptr, c)  hmx_add_bias_to_f16_vec64_f32((v_out), (row_ptr), (c))

#define HMX_FP16_ADD_BIAS_F16_TAIL_NONE(v_out, row_ptr, c, vc) (v_out)
#define HMX_FP16_ADD_BIAS_F16_TAIL_F16(v_out, row_ptr, c, vc)  hmx_add_bias_to_f16_tail_f16((v_out), (row_ptr), (c), (vc))
#define HMX_FP16_ADD_BIAS_F16_TAIL_F32(v_out, row_ptr, c, vc)  hmx_add_bias_to_f16_tail_f32((v_out), (row_ptr), (c), (vc))

#define HMX_FP16_BIAS_ROW0_none(bias, off) NULL
#define HMX_FP16_BIAS_ROW0_f16(bias, off)  ((const uint8_t *) (bias) + (off))
#define HMX_FP16_BIAS_ROW0_f32(bias, off)  ((const uint8_t *) (bias) + (off))

#define HMX_FP16_BIAS_ROW1_none(row0, rb)  NULL
#define HMX_FP16_BIAS_ROW1_f16(row0, rb)   ((row0) + (rb))
#define HMX_FP16_BIAS_ROW1_f32(row0, rb)   ((row0) + (rb))

#define HMX_FP16_BIAS_STEP_none(ptr, step) (void) 0
#define HMX_FP16_BIAS_STEP_f16(ptr, step)  ((ptr) += (step))
#define HMX_FP16_BIAS_STEP_f32(ptr, step)  ((ptr) += (step))

#define HMX_STORE_VEC_nt(PTR, VEC) hvx_vmem_store_nt((PTR), (VEC))
#define HMX_STORE_VEC_u(PTR, VEC)  (hvx_vmemu((PTR)) = (VEC))

#define HMX_FP16_TO_FP32_PAIR_STEP(STORE_MODE, ADD_BIAS)                                                 \
    do {                                                                                                 \
        _Pragma("unroll(4)")                                                                             \
        for (size_t c = 0; c < limit_c_aligned; c += HTP_MM_HMX_TILE_N_COLS) {                           \
            const size_t c0    = c / HTP_MM_HMX_TILE_N_COLS;                                             \
            const __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;                                 \
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];                                              \
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);                                            \
            HVX_Vector v_out0 = ADD_BIAS(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp)), bias_row0, c);              \
            HVX_Vector v_out1 = ADD_BIAS(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp)), bias_row1, c);              \
            HMX_STORE_VEC_##STORE_MODE(output_row_base + c, v_out0);                                     \
            HMX_STORE_VEC_##STORE_MODE(output_row_base + c + dst_stride, v_out1);                        \
        }                                                                                                \
        if (limit_c_aligned < limit_c) {                                                                 \
            size_t c = limit_c_aligned;                                                                  \
            size_t valid_c = limit_c - c;                                                                \
            const size_t c0 = c / HTP_MM_HMX_TILE_N_COLS;                                                \
            const __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;                                 \
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];                                              \
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);                                            \
            HVX_Vector v_out0 = ADD_BIAS(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp)), bias_row0, c);              \
            HVX_Vector v_out1 = ADD_BIAS(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp)), bias_row1, c);              \
            hvx_vec_store_u(output_row_base + c, valid_c * sizeof(float), v_out0);                       \
            hvx_vec_store_u(output_row_base + c + dst_stride, valid_c * sizeof(float), v_out1);          \
        }                                                                                                \
    } while (0)

#define HMX_FP16_TO_FP32_SINGLE_STEP(STORE_MODE, ADD_BIAS)                                               \
    do {                                                                                                 \
        _Pragma("unroll(4)")                                                                             \
        for (size_t c = 0; c < limit_c_aligned; c += HTP_MM_HMX_TILE_N_COLS) {                           \
            const size_t c0    = c / HTP_MM_HMX_TILE_N_COLS;                                             \
            const __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;                                 \
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];                                              \
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);                                            \
            HVX_Vector v_out0 = ADD_BIAS(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp)), bias_row0, c);              \
            HMX_STORE_VEC_##STORE_MODE(output_row_base + c, v_out0);                                     \
        }                                                                                                \
        if (limit_c_aligned < limit_c) {                                                                 \
            size_t c = limit_c_aligned;                                                                  \
            size_t valid_c = limit_c - c;                                                                \
            const size_t c0 = c / HTP_MM_HMX_TILE_N_COLS;                                                \
            const __fp16 *tile = row_base + c0 * HTP_MM_HMX_TILE_N_ELMS;                                 \
            HVX_Vector v = ((const HVX_Vector *) tile)[r1];                                              \
            HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(v, one);                                            \
            HVX_Vector v_out0 = ADD_BIAS(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp)), bias_row0, c);              \
            hvx_vec_store_u(output_row_base + c, valid_c * sizeof(float), v_out0);                       \
        }                                                                                                \
    } while (0)

#define DEFINE_FP16_TO_FP32_COL_CHUNK_MODE(SFX, ADD_BIAS, STORE_MODE)                                    \
static __attribute__((noinline)) void transfer_output_chunk_fp16_to_fp32_col_chunk_##SFX##_##STORE_MODE( \
    void * restrict dst_ptr,                                                                             \
    const void * restrict bias,                                                                          \
    const __fp16 * restrict vtcm_src,                                                                    \
    uint32_t start_row,                                                                                  \
    uint32_t n_rows,                                                                                     \
    uint32_t c_len,                                                                                      \
    uint32_t total_n_cols,                                                                               \
    uint32_t dst_stride,                                                                                 \
    uint32_t bias_stride,                                                                                \
    uint32_t dst_cols,                                                                                   \
    size_t bias_elem_size                                                                                \
) {                                                                                                      \
    float * restrict dst           = (float *) dst_ptr;                                                  \
    const size_t tile_row_stride   = hmx_ceil_div(total_n_cols,                                          \
                                                  HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;      \
    const HVX_Vector one           = hvx_vec_splat_f16(1.0);                                             \
    const size_t limit_c           = hex_smin(c_len, dst_cols);                                          \
    const size_t limit_c_aligned   = (limit_c & ~31u);                                                   \
    const size_t bias_row_bytes    = (size_t) bias_stride * bias_elem_size;                              \
    const size_t dst_pair_stride   = 2 * (size_t) dst_stride;                                            \
    const size_t __attribute__((unused)) bias_pair_bytes = 2 * bias_row_bytes;                           \
    float * output_row_base        = dst;                                                                \
    const uint8_t * __attribute__((unused)) bias_row0 = HMX_FP16_BIAS_ROW0_##SFX(bias, 0);               \
    const uint8_t * __attribute__((unused)) bias_row1 = HMX_FP16_BIAS_ROW1_##SFX(bias_row0,              \
                                                                                 bias_row_bytes);        \
    size_t r_idx0                  = start_row;                                                          \
    const __fp16 * row_base        = vtcm_src + (r_idx0 >> 5) * tile_row_stride;                         \
    size_t r1                      = (r_idx0 & 31u) >> 1;                                                \
    size_t r                       = 0;                                                                  \
    for (; r + 1 < n_rows; r += 2) {                                                                     \
        HMX_FP16_TO_FP32_PAIR_STEP(STORE_MODE, ADD_BIAS);                                                \
        output_row_base += dst_pair_stride;                                                              \
        HMX_FP16_BIAS_STEP_##SFX(bias_row0, bias_pair_bytes);                                            \
        HMX_FP16_BIAS_STEP_##SFX(bias_row1, bias_pair_bytes);                                            \
        if (++r1 == 16) {                                                                                \
            r1 = 0;                                                                                      \
            row_base += tile_row_stride;                                                                 \
        }                                                                                                \
    }                                                                                                    \
    if (r < n_rows) {                                                                                    \
        HMX_FP16_TO_FP32_SINGLE_STEP(STORE_MODE, ADD_BIAS);                                              \
    }                                                                                                    \
}

#define DEFINE_FP16_TO_FP32_COL_CHUNK(SFX, ADD_BIAS)         \
    DEFINE_FP16_TO_FP32_COL_CHUNK_MODE(SFX, ADD_BIAS, nt)    \
    DEFINE_FP16_TO_FP32_COL_CHUNK_MODE(SFX, ADD_BIAS, u)

DEFINE_FP16_TO_FP32_COL_CHUNK(none, HMX_FP16_ADD_BIAS_F32_NONE)
DEFINE_FP16_TO_FP32_COL_CHUNK(f16,  HMX_FP16_ADD_BIAS_F32_F16)
DEFINE_FP16_TO_FP32_COL_CHUNK(f32,  HMX_FP16_ADD_BIAS_F32_F32)

#define HMX_FP16_TO_FP16_PAIR_STEP(STORE_MODE, ADD_BIAS_64, ADD_BIAS_TAIL)                               \
    do {                                                                                                 \
        for (size_t c = 0; c < limit_c_aligned64; c += 64) {                                             \
            const size_t c0     = c / HTP_MM_HMX_TILE_N_COLS;                                            \
            const __fp16 *tile0 = row_base + (c0 + 0) * HTP_MM_HMX_TILE_N_ELMS;                          \
            const __fp16 *tile1 = row_base + (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS;                          \
            HVX_Vector v_t0     = ((const HVX_Vector *) tile0)[r1];                                      \
            HVX_Vector v_t1     = ((const HVX_Vector *) tile1)[r1];                                      \
            HVX_VectorPair vp   = Q6_W_vdeal_VVR(v_t1, v_t0, -2);                                        \
            HVX_Vector v_out0   = ADD_BIAS_64(Q6_V_lo_W(vp), bias_row0, c);                              \
            HVX_Vector v_out1   = ADD_BIAS_64(Q6_V_hi_W(vp), bias_row1, c);                              \
            HMX_STORE_VEC_##STORE_MODE(output_row_base + c, v_out0);                                     \
            HMX_STORE_VEC_##STORE_MODE(output_row_base + dst_stride + c, v_out1);                        \
        }                                                                                                \
        if (limit_c_aligned64 < limit_c) {                                                               \
            const size_t c       = limit_c_aligned64;                                                    \
            const size_t valid_c = limit_c - c;                                                          \
            const size_t c0      = c / HTP_MM_HMX_TILE_N_COLS;                                           \
            const __fp16 *tile0  = row_base + (c0 + 0) * HTP_MM_HMX_TILE_N_ELMS;                         \
            HVX_Vector v_t0 = ((const HVX_Vector *) tile0)[r1];                                          \
            HVX_Vector v_t1 = (valid_c > 32)                                                             \
                ? ((const HVX_Vector *) (row_base + (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS))[r1]              \
                : Q6_V_vzero();                                                                          \
            HVX_VectorPair vp = Q6_W_vdeal_VVR(v_t1, v_t0, -2);                                          \
            HVX_Vector v_out0 = ADD_BIAS_TAIL(Q6_V_lo_W(vp), bias_row0, c, valid_c);                     \
            HVX_Vector v_out1 = ADD_BIAS_TAIL(Q6_V_hi_W(vp), bias_row1, c, valid_c);                     \
            hvx_vec_store_u(output_row_base + c, valid_c * sizeof(__fp16), v_out0);                      \
            hvx_vec_store_u(output_row_base + dst_stride + c, valid_c * sizeof(__fp16), v_out1);         \
        }                                                                                                \
    } while (0)

#define HMX_FP16_TO_FP16_SINGLE_STEP(STORE_MODE, ADD_BIAS_64, ADD_BIAS_TAIL)                             \
    do {                                                                                                 \
        for (size_t c = 0; c < limit_c_aligned64; c += 64) {                                             \
            const size_t c0     = c / HTP_MM_HMX_TILE_N_COLS;                                            \
            const __fp16 *tile0 = row_base + (c0 + 0) * HTP_MM_HMX_TILE_N_ELMS;                          \
            const __fp16 *tile1 = row_base + (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS;                          \
            HVX_Vector v_t0     = ((const HVX_Vector *) tile0)[r1];                                      \
            HVX_Vector v_t1     = ((const HVX_Vector *) tile1)[r1];                                      \
            HVX_VectorPair vp   = Q6_W_vdeal_VVR(v_t1, v_t0, -2);                                        \
            HVX_Vector v_out0   = ADD_BIAS_64(Q6_V_lo_W(vp), bias_row0, c);                              \
            HMX_STORE_VEC_##STORE_MODE(output_row_base + c, v_out0);                                     \
        }                                                                                                \
        if (limit_c_aligned64 < limit_c) {                                                               \
            const size_t c       = limit_c_aligned64;                                                    \
            const size_t valid_c = limit_c - c;                                                          \
            const size_t c0      = c / HTP_MM_HMX_TILE_N_COLS;                                           \
            const __fp16 *tile0  = row_base + (c0 + 0) * HTP_MM_HMX_TILE_N_ELMS;                         \
            HVX_Vector v_t0 = ((const HVX_Vector *) tile0)[r1];                                          \
            HVX_Vector v_t1 = (valid_c > 32)                                                             \
                ? ((const HVX_Vector *) (row_base + (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS))[r1]              \
                : Q6_V_vzero();                                                                          \
            HVX_VectorPair vp = Q6_W_vdeal_VVR(v_t1, v_t0, -2);                                          \
            HVX_Vector v_out0 = ADD_BIAS_TAIL(Q6_V_lo_W(vp), bias_row0, c, valid_c);                     \
            hvx_vec_store_u(output_row_base + c, valid_c * sizeof(__fp16), v_out0);                      \
        }                                                                                                \
    } while (0)

#define DEFINE_FP16_TO_FP16_COL_CHUNK_MODE(SFX, ADD_BIAS_64, ADD_BIAS_TAIL, STORE_MODE)                  \
static __attribute__((noinline)) void transfer_output_chunk_fp16_to_fp16_col_chunk_##SFX##_##STORE_MODE( \
    void * restrict dst_ptr,                                                                             \
    const void * restrict bias,                                                                          \
    const __fp16 * restrict vtcm_src,                                                                    \
    uint32_t start_row,                                                                                  \
    uint32_t n_rows,                                                                                     \
    uint32_t c_len,                                                                                      \
    uint32_t total_n_cols,                                                                               \
    uint32_t dst_stride,                                                                                 \
    uint32_t bias_stride,                                                                                \
    uint32_t dst_cols,                                                                                   \
    size_t bias_elem_size                                                                                \
) {                                                                                                      \
    __fp16 * restrict dst          = (__fp16 *) dst_ptr;                                                 \
    const size_t tile_row_stride   = hmx_ceil_div(total_n_cols,                                          \
                                                  HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;      \
    const size_t limit_c           = hex_smin(c_len, dst_cols);                                          \
    const size_t limit_c_aligned64 = (limit_c & ~63u);                                                   \
    const size_t bias_row_bytes    = (size_t) bias_stride * bias_elem_size;                              \
    const size_t dst_pair_stride   = 2 * (size_t) dst_stride;                                            \
    const size_t __attribute__((unused)) bias_pair_bytes = 2 * bias_row_bytes;                           \
    __fp16 * output_row_base       = dst;                                                                \
    const uint8_t * __attribute__((unused)) bias_row0 = HMX_FP16_BIAS_ROW0_##SFX(bias, 0);               \
    const uint8_t * __attribute__((unused)) bias_row1 = HMX_FP16_BIAS_ROW1_##SFX(bias_row0,              \
                                                                                 bias_row_bytes);        \
    size_t r_idx0                  = start_row;                                                          \
    const __fp16 * row_base        = vtcm_src + (r_idx0 >> 5) * tile_row_stride;                         \
    size_t r1                      = (r_idx0 & 31u) >> 1;                                                \
    size_t r                       = 0;                                                                  \
    for (; r + 1 < n_rows; r += 2) {                                                                     \
        HMX_FP16_TO_FP16_PAIR_STEP(STORE_MODE, ADD_BIAS_64, ADD_BIAS_TAIL);                              \
        output_row_base += dst_pair_stride;                                                              \
        HMX_FP16_BIAS_STEP_##SFX(bias_row0, bias_pair_bytes);                                            \
        HMX_FP16_BIAS_STEP_##SFX(bias_row1, bias_pair_bytes);                                            \
        if (++r1 == 16) {                                                                                \
            r1 = 0;                                                                                      \
            row_base += tile_row_stride;                                                                 \
        }                                                                                                \
    }                                                                                                    \
    if (r < n_rows) {                                                                                    \
        HMX_FP16_TO_FP16_SINGLE_STEP(STORE_MODE, ADD_BIAS_64, ADD_BIAS_TAIL);                            \
    }                                                                                                    \
}

#define DEFINE_FP16_TO_FP16_COL_CHUNK(SFX, ADD_BIAS_64, ADD_BIAS_TAIL)         \
    DEFINE_FP16_TO_FP16_COL_CHUNK_MODE(SFX, ADD_BIAS_64, ADD_BIAS_TAIL, nt)    \
    DEFINE_FP16_TO_FP16_COL_CHUNK_MODE(SFX, ADD_BIAS_64, ADD_BIAS_TAIL, u)

DEFINE_FP16_TO_FP16_COL_CHUNK(none, HMX_FP16_ADD_BIAS_F16_64_NONE, HMX_FP16_ADD_BIAS_F16_TAIL_NONE)
DEFINE_FP16_TO_FP16_COL_CHUNK(f16,  HMX_FP16_ADD_BIAS_F16_64_F16,  HMX_FP16_ADD_BIAS_F16_TAIL_F16)
DEFINE_FP16_TO_FP16_COL_CHUNK(f32,  HMX_FP16_ADD_BIAS_F16_64_F32,  HMX_FP16_ADD_BIAS_F16_TAIL_F32)

typedef void (*hmx_fp16_output_col_chunk_fn_t)(
    void * restrict dst,
    const void * restrict bias,
    const __fp16 * restrict vtcm_src,
    uint32_t start_row,
    uint32_t n_rows,
    uint32_t c_len,
    uint32_t total_n_cols,
    uint32_t dst_stride,
    uint32_t bias_stride,
    uint32_t dst_cols,
    size_t bias_elem_size
);

static inline hmx_fp16_output_col_chunk_fn_t resolve_fp16_output_col_chunk_fn(
    size_t dst_elem_size,
    const void * bias,
    size_t bias_elem_size,
    bool dst_aligned_nt
) {
    const bool dst_f16  = (dst_elem_size == sizeof(__fp16));
    const bool bias_f16 = (bias_elem_size == sizeof(__fp16));
    if (dst_f16) {
        if (dst_aligned_nt) {
            return !bias ? transfer_output_chunk_fp16_to_fp16_col_chunk_none_nt
                         : (bias_f16 ? transfer_output_chunk_fp16_to_fp16_col_chunk_f16_nt
                                     : transfer_output_chunk_fp16_to_fp16_col_chunk_f32_nt);
        }
        return !bias ? transfer_output_chunk_fp16_to_fp16_col_chunk_none_u
                     : (bias_f16 ? transfer_output_chunk_fp16_to_fp16_col_chunk_f16_u
                                 : transfer_output_chunk_fp16_to_fp16_col_chunk_f32_u);
    }
    if (dst_aligned_nt) {
        return !bias ? transfer_output_chunk_fp16_to_fp32_col_chunk_none_nt
                     : (bias_f16 ? transfer_output_chunk_fp16_to_fp32_col_chunk_f16_nt
                                 : transfer_output_chunk_fp16_to_fp32_col_chunk_f32_nt);
    }
    return !bias ? transfer_output_chunk_fp16_to_fp32_col_chunk_none_u
                 : (bias_f16 ? transfer_output_chunk_fp16_to_fp32_col_chunk_f16_u
                             : transfer_output_chunk_fp16_to_fp32_col_chunk_f32_u);
}

typedef struct {
    hmx_fp16_output_col_chunk_fn_t col_chunk_fn;
    const __fp16  *vtcm_src;
    void          *dst;
    const void    *bias;
    size_t         dst_elem_size;
    size_t         bias_elem_size;
    uint32_t       n_tasks;
    uint32_t       n_tot_chunks;
    uint32_t       n_chunks_per_task;
    uint32_t       n_cols;
    uint32_t       dst_stride;  // DDR row stride
    uint32_t       bias_stride; // DDR row stride for residual
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
    for (; c < k_block; c += 32) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        if (c < k_valid) {
            if (row0_valid) v0 = *(const HVX_Vector *)(row0 + c);
            if (row1_valid) v1 = *(const HVX_Vector *)(row1 + c);

            uint32_t rem = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());
        }

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

// --- Shared HMX GLU Store Helpers & FP16-HMX GLU Epilogue ---

static inline HVX_Vector hmx_glu_geglu_arg_f16(HVX_Vector x) {
    // 0.5 * x * (1 + tanh(inner)) = x * sigmoid(2 * inner)
    const HVX_Vector v_c0 = hvx_vec_splat_f16(2.0f * 0.7978845608028654f * 0.044715f);
    const HVX_Vector v_c1 = hvx_vec_splat_f16(2.0f * 0.7978845608028654f);
    const HVX_Vector x2   = hvx_vec_mul_f16_f16(x, x);
    const HVX_Vector poly = hvx_vec_add_f16_f16(hvx_vec_mul_f16_f16(x2, v_c0), v_c1);
    return hvx_vec_mul_f16_f16(x, poly);
}

static inline HVX_Vector hmx_glu_geglu_arg_f32(HVX_Vector x) {
    const HVX_Vector v_c0 = hvx_vec_splat_f32(2.0f * 0.7978845608028654f * 0.044715f);
    const HVX_Vector v_c1 = hvx_vec_splat_f32(2.0f * 0.7978845608028654f);
    const HVX_Vector x2   = hvx_vec_mul_f32_f32(x, x);
    const HVX_Vector poly = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x2, v_c0), v_c1);
    return hvx_vec_mul_f32_f32(x, poly);
}

#define HMX_GLU_ARG_SWIGLU_F16(x)       (x)
#define HMX_GLU_ARG_SWIGLU_F32(x)       (x)
#define HMX_GLU_ARG_GEGLU_F16(x)        hmx_glu_geglu_arg_f16(x)
#define HMX_GLU_ARG_GEGLU_F32(x)        hmx_glu_geglu_arg_f32(x)
#define HMX_GLU_ARG_GEGLU_QUICK_F16(x)  hvx_vec_mul_f16_f16((x), hvx_vec_splat_f16(1.702f))
#define HMX_GLU_ARG_GEGLU_QUICK_F32(x)  hvx_vec_mul_f32_f32((x), hvx_vec_splat_f32(1.702f))

// Sigmoid-based GLU stores: out = g * sigmoid(ARG(g)) * u (vtcm_res_out is always 128B aligned).
#define DEFINE_HMX_SIG_GLU_STORE(NAME, ARG_F16, ARG_F32)                                                         \
static __attribute__((noinline)) void hmx_glu_##NAME##_store_tail_fp16(                                          \
        void * restrict dst_ptr,                                                                                 \
        size_t dst_stride,                                                                                       \
        const HVX_Vector * restrict g_hf,                                                                        \
        const HVX_Vector * restrict u_hf,                                                                        \
        uint32_t rows_and_bytes) {                                                                               \
    __fp16 * restrict dst         = (__fp16 *) dst_ptr;                                                          \
    const uint32_t    valid_rows  = rows_and_bytes >> 16;                                                        \
    const size_t      valid_bytes = rows_and_bytes & 0xffffu;                                                    \
    _Pragma("clang loop pipeline(disable) unroll(disable)")                                                      \
    for (uint32_t k = 0; k < valid_rows; ++k) {                                                                  \
        const HVX_Vector g   = g_hf[k];                                                                          \
        const HVX_Vector sig = hvx_vec_fast_sigmoid_f16(ARG_F16(g));                                             \
        const HVX_Vector out = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(g, sig), u_hf[k]);                        \
        hvx_vec_store_u(dst + k * dst_stride, valid_bytes, out);                                                 \
    }                                                                                                            \
}                                                                                                                \
static __attribute__((noinline)) void hmx_glu_##NAME##_store_quad_fp16(                                          \
        void * restrict dst_ptr,                                                                                 \
        size_t dst_stride,                                                                                       \
        const HVX_Vector * restrict g_hf,                                                                        \
        const HVX_Vector * restrict u_hf,                                                                        \
        uint32_t valid_rows) {                                                                                   \
    __fp16 * restrict dst = (__fp16 *) dst_ptr;                                                                  \
    if (valid_rows == 4u) {                                                                                      \
        const HVX_Vector     g0  = g_hf[0];                                                                      \
        const HVX_Vector     g1  = g_hf[1];                                                                      \
        const HVX_VectorPair s01 = hvx_vec_fast_sigmoid_f16_pair(ARG_F16(g0), ARG_F16(g1));                      \
        hvx_vmem(dst + 0 * dst_stride) = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(g0, Q6_V_lo_W(s01)), u_hf[0]);  \
        hvx_vmem(dst + 1 * dst_stride) = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(g1, Q6_V_hi_W(s01)), u_hf[1]);  \
        const HVX_Vector     g2  = g_hf[2];                                                                      \
        const HVX_Vector     g3  = g_hf[3];                                                                      \
        const HVX_VectorPair s23 = hvx_vec_fast_sigmoid_f16_pair(ARG_F16(g2), ARG_F16(g3));                      \
        hvx_vmem(dst + 2 * dst_stride) = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(g2, Q6_V_lo_W(s23)), u_hf[2]);  \
        hvx_vmem(dst + 3 * dst_stride) = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(g3, Q6_V_hi_W(s23)), u_hf[3]);  \
        return;                                                                                                  \
    }                                                                                                            \
    hmx_glu_##NAME##_store_tail_fp16(dst, dst_stride, g_hf, u_hf, (valid_rows << 16) | 128u);                    \
}                                                                                                                \
static __attribute__((noinline)) void hmx_glu_##NAME##_store_tail_fp32(                                          \
        void * restrict dst_ptr,                                                                                 \
        size_t dst_stride,                                                                                       \
        const HVX_Vector * restrict g_q,                                                                         \
        const HVX_Vector * restrict u_q,                                                                         \
        uint32_t rows_and_bytes) {                                                                               \
    float * restrict dst         = (float *) dst_ptr;                                                            \
    const uint32_t   valid_rows  = rows_and_bytes >> 16;                                                         \
    const size_t     valid_bytes = rows_and_bytes & 0xffffu;                                                     \
    const HVX_Vector v_one       = hvx_vec_splat_f32(1.0f);                                                      \
    const HVX_Vector v_max_exp   = hvx_vec_splat_f32(87.0f);                                                     \
    const HVX_Vector v_min_exp   = hvx_vec_splat_f32(-87.0f);                                                    \
    _Pragma("clang loop unroll(disable)")                                                                        \
    for (uint32_t k = 0; k < valid_rows; ++k) {                                                                  \
        const HVX_Vector g   = g_q[k];                                                                           \
        const HVX_Vector sig = hvx_vec_fast_sigmoid_f32_guard(ARG_F32(g), v_one, v_max_exp, v_min_exp);          \
        const HVX_Vector out = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(g, sig), u_q[k]);                         \
        hvx_vec_store_u(dst + k * dst_stride, valid_bytes, out);                                                 \
    }                                                                                                            \
}                                                                                                                \
static __attribute__((noinline)) void hmx_glu_##NAME##_store_quad_fp32(                                          \
        void * restrict dst_ptr,                                                                                 \
        size_t dst_stride,                                                                                       \
        const HVX_Vector * restrict g_q,                                                                         \
        const HVX_Vector * restrict u_q,                                                                         \
        uint32_t valid_rows) {                                                                                   \
    float * restrict dst       = (float *) dst_ptr;                                                              \
    const HVX_Vector v_one     = hvx_vec_splat_f32(1.0f);                                                        \
    const HVX_Vector v_max_exp = hvx_vec_splat_f32(87.0f);                                                       \
    const HVX_Vector v_min_exp = hvx_vec_splat_f32(-87.0f);                                                      \
    if (valid_rows == 4u) {                                                                                      \
        const HVX_Vector     g0  = g_q[0];                                                                       \
        const HVX_Vector     g1  = g_q[1];                                                                       \
        const HVX_VectorPair s01 = hvx_vec_fast_sigmoid_f32_guard_pair(ARG_F32(g0), ARG_F32(g1),                 \
                                                                       v_one, v_max_exp, v_min_exp);             \
        hvx_vmem(dst + 0 * dst_stride) = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(g0, Q6_V_lo_W(s01)), u_q[0]);   \
        hvx_vmem(dst + 1 * dst_stride) = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(g1, Q6_V_hi_W(s01)), u_q[1]);   \
        const HVX_Vector     g2  = g_q[2];                                                                       \
        const HVX_Vector     g3  = g_q[3];                                                                       \
        const HVX_VectorPair s23 = hvx_vec_fast_sigmoid_f32_guard_pair(ARG_F32(g2), ARG_F32(g3),                 \
                                                                       v_one, v_max_exp, v_min_exp);             \
        hvx_vmem(dst + 2 * dst_stride) = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(g2, Q6_V_lo_W(s23)), u_q[2]);   \
        hvx_vmem(dst + 3 * dst_stride) = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(g3, Q6_V_hi_W(s23)), u_q[3]);   \
        return;                                                                                                  \
    }                                                                                                            \
    hmx_glu_##NAME##_store_tail_fp32(dst, dst_stride, g_q, u_q, (valid_rows << 16) | 128u);                      \
}

DEFINE_HMX_SIG_GLU_STORE(swiglu,      HMX_GLU_ARG_SWIGLU_F16,      HMX_GLU_ARG_SWIGLU_F32)
DEFINE_HMX_SIG_GLU_STORE(geglu,       HMX_GLU_ARG_GEGLU_F16,       HMX_GLU_ARG_GEGLU_F32)
DEFINE_HMX_SIG_GLU_STORE(geglu_quick, HMX_GLU_ARG_GEGLU_QUICK_F16, HMX_GLU_ARG_GEGLU_QUICK_F32)

#define DEFINE_HMX_SIG_GLU_STRIP_FP16(NAME, ARG_F16)                                                 \
static __attribute__((noinline)) void hmx_glu_##NAME##_store_strip_fp16(                             \
        void * restrict dst_ptr,                                                                     \
        size_t dst_stride,                                                                           \
        const __fp16 * restrict u_buf,                                                               \
        size_t u_stride,                                                                             \
        size_t n_pairs) {                                                                            \
    __fp16 * restrict dst = (__fp16 *) dst_ptr;                                                      \
    _Pragma("clang loop unroll(disable)")                                                            \
    for (size_t p = 0; p < n_pairs; ++p) {                                                           \
        __fp16 *       d0        = dst + (p << 6);                                                   \
        const __fp16 * u0        = u_buf + (p << 6);                                                 \
        const HVX_Vector     g0  = hvx_vmem(d0 + 0 * dst_stride);                                    \
        const HVX_Vector     g1  = hvx_vmem(d0 + 1 * dst_stride);                                    \
        const HVX_VectorPair s01 = hvx_vec_fast_sigmoid_f16_pair(ARG_F16(g0), ARG_F16(g1));          \
        hvx_vmem(d0 + 0 * dst_stride) = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(g0, Q6_V_lo_W(s01)), \
                                                            hvx_vmem(u0 + 0 * u_stride));            \
        hvx_vmem(d0 + 1 * dst_stride) = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(g1, Q6_V_hi_W(s01)), \
                                                            hvx_vmem(u0 + 1 * u_stride));            \
        const HVX_Vector     g2  = hvx_vmem(d0 + 2 * dst_stride);                                    \
        const HVX_Vector     g3  = hvx_vmem(d0 + 3 * dst_stride);                                    \
        const HVX_VectorPair s23 = hvx_vec_fast_sigmoid_f16_pair(ARG_F16(g2), ARG_F16(g3));          \
        hvx_vmem(d0 + 2 * dst_stride) = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(g2, Q6_V_lo_W(s23)), \
                                                            hvx_vmem(u0 + 2 * u_stride));            \
        hvx_vmem(d0 + 3 * dst_stride) = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(g3, Q6_V_hi_W(s23)), \
                                                            hvx_vmem(u0 + 3 * u_stride));            \
    }                                                                                                \
}

DEFINE_HMX_SIG_GLU_STRIP_FP16(swiglu,      HMX_GLU_ARG_SWIGLU_F16)
DEFINE_HMX_SIG_GLU_STRIP_FP16(geglu,       HMX_GLU_ARG_GEGLU_F16)
DEFINE_HMX_SIG_GLU_STRIP_FP16(geglu_quick, HMX_GLU_ARG_GEGLU_QUICK_F16)

static __attribute__((noinline)) void hmx_glu_reglu_store_tail_fp16(
        void * restrict dst_ptr,
        size_t dst_stride,
        const HVX_Vector * restrict g_hf,
        const HVX_Vector * restrict u_hf,
        uint32_t rows_and_bytes) {
    __fp16 * restrict dst         = (__fp16 *) dst_ptr;
    const uint32_t    valid_rows  = rows_and_bytes >> 16;
    const size_t      valid_bytes = rows_and_bytes & 0xffffu;
    const HVX_Vector  v_zero      = Q6_V_vzero();
    for (uint32_t k = 0; k < valid_rows; ++k) {
        const HVX_Vector out = hvx_vec_mul_f16_f16(Q6_Vhf_vmax_VhfVhf(g_hf[k], v_zero), u_hf[k]);
        hvx_vec_store_u(dst + k * dst_stride, valid_bytes, out);
    }
}

static __attribute__((noinline)) void hmx_glu_reglu_store_quad_fp16(
        void * restrict dst_ptr,
        size_t dst_stride,
        const HVX_Vector * restrict g_hf,
        const HVX_Vector * restrict u_hf,
        uint32_t valid_rows) {
    __fp16 * restrict dst    = (__fp16 *) dst_ptr;
    const HVX_Vector  v_zero = Q6_V_vzero();
    for (uint32_t k = 0; k < valid_rows; ++k) {
        hvx_vmem(dst + k * dst_stride) = hvx_vec_mul_f16_f16(Q6_Vhf_vmax_VhfVhf(g_hf[k], v_zero), u_hf[k]);
    }
}

static __attribute__((noinline)) void hmx_glu_reglu_store_strip_fp16(
        void * restrict dst_ptr,
        size_t dst_stride,
        const __fp16 * restrict u_buf,
        size_t u_stride,
        size_t n_pairs) {
    __fp16 * restrict dst    = (__fp16 *) dst_ptr;
    const HVX_Vector  v_zero = Q6_V_vzero();
    _Pragma("clang loop unroll(disable)")
    for (size_t p = 0; p < n_pairs; ++p) {
        __fp16 *       d0 = dst + (p << 6);
        const __fp16 * u0 = u_buf + (p << 6);
        hvx_vmem(d0 + 0 * dst_stride) = hvx_vec_mul_f16_f16(Q6_Vhf_vmax_VhfVhf(hvx_vmem(d0 + 0 * dst_stride), v_zero), hvx_vmem(u0 + 0 * u_stride));
        hvx_vmem(d0 + 1 * dst_stride) = hvx_vec_mul_f16_f16(Q6_Vhf_vmax_VhfVhf(hvx_vmem(d0 + 1 * dst_stride), v_zero), hvx_vmem(u0 + 1 * u_stride));
        hvx_vmem(d0 + 2 * dst_stride) = hvx_vec_mul_f16_f16(Q6_Vhf_vmax_VhfVhf(hvx_vmem(d0 + 2 * dst_stride), v_zero), hvx_vmem(u0 + 2 * u_stride));
        hvx_vmem(d0 + 3 * dst_stride) = hvx_vec_mul_f16_f16(Q6_Vhf_vmax_VhfVhf(hvx_vmem(d0 + 3 * dst_stride), v_zero), hvx_vmem(u0 + 3 * u_stride));
    }
}

static __attribute__((noinline)) void hmx_glu_reglu_store_tail_fp32(
        void * restrict dst_ptr,
        size_t dst_stride,
        const HVX_Vector * restrict g_q,
        const HVX_Vector * restrict u_q,
        uint32_t rows_and_bytes) {
    float * restrict dst         = (float *) dst_ptr;
    const uint32_t   valid_rows  = rows_and_bytes >> 16;
    const size_t     valid_bytes = rows_and_bytes & 0xffffu;
    const HVX_Vector v_zero      = Q6_V_vzero();
    for (uint32_t k = 0; k < valid_rows; ++k) {
        const HVX_Vector out = hvx_vec_mul_f32_f32(Q6_Vsf_vmax_VsfVsf(g_q[k], v_zero), u_q[k]);
        hvx_vec_store_u(dst + k * dst_stride, valid_bytes, out);
    }
}

static __attribute__((noinline)) void hmx_glu_reglu_store_quad_fp32(
        void * restrict dst_ptr,
        size_t dst_stride,
        const HVX_Vector * restrict g_q,
        const HVX_Vector * restrict u_q,
        uint32_t valid_rows) {
    float * restrict dst    = (float *) dst_ptr;
    const HVX_Vector v_zero = Q6_V_vzero();
    for (uint32_t k = 0; k < valid_rows; ++k) {
        hvx_vmem(dst + k * dst_stride) = hvx_vec_mul_f32_f32(Q6_Vsf_vmax_VsfVsf(g_q[k], v_zero), u_q[k]);
    }
}

typedef void (*hmx_glu_store_quad_fn_t)(
        void * restrict dst_ptr,
        size_t dst_stride,
        const HVX_Vector * restrict g_vec,
        const HVX_Vector * restrict u_vec,
        uint32_t valid_rows);

typedef void (*hmx_glu_store_tail_fn_t)(
        void * restrict dst_ptr,
        size_t dst_stride,
        const HVX_Vector * restrict g_vec,
        const HVX_Vector * restrict u_vec,
        uint32_t rows_and_bytes);

typedef void (*hmx_glu_store_strip_fn_t)(
        void * restrict dst_ptr,
        size_t dst_stride,
        const __fp16 * restrict u_buf,
        size_t u_stride,
        size_t n_pairs);

typedef void (*hmx_output_glu_col_chunk_fn_t)(
        const void * st_void,
        void * restrict dst_ptr,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t c_first,
        uint32_t c_len_and_dst_cols);

typedef struct {
    void *                        dst;
    const uint8_t *               vtcm_gate;
    const uint8_t *               vtcm_up;
    const float *                 vtcm_act_scales;
    const float *                 vtcm_wt_scales_gate;
    const float *                 vtcm_wt_scales_up;
    uint32_t                      n_rows;
    uint32_t                      mc_pad;
    uint32_t                      n_cols;
    uint32_t                      n_groups;
    uint32_t                      dst_stride;
    uint32_t                      dst_cols;
    size_t                        dst_elem_size;
    uint32_t                      n_tasks;
    uint32_t                      n_chunks_per_task;
    struct fastdiv_values         n_hvx_div;
    hmx_output_glu_col_chunk_fn_t col_chunk_fn;
    hmx_glu_store_quad_fn_t       store_quad_fn;
    hmx_glu_store_tail_fn_t       store_tail_fn;
    hmx_glu_store_strip_fn_t      store_strip_fn;
} hmx_output_glu_state_t;

static inline bool resolve_hmx_glu_store_fns(
        uint8_t glu_op,
        size_t dst_elem_size,
        hmx_glu_store_quad_fn_t * store_quad_fn,
        hmx_glu_store_tail_fn_t * store_tail_fn,
        hmx_glu_store_strip_fn_t * store_strip_fn) {
    if (dst_elem_size == sizeof(__fp16)) {
        switch (glu_op) {
            case HTP_OP_GLU_SWIGLU:
                *store_quad_fn  = hmx_glu_swiglu_store_quad_fp16;
                *store_tail_fn  = hmx_glu_swiglu_store_tail_fp16;
                *store_strip_fn = hmx_glu_swiglu_store_strip_fp16;
                return true;
            case HTP_OP_GLU_GEGLU:
                *store_quad_fn  = hmx_glu_geglu_store_quad_fp16;
                *store_tail_fn  = hmx_glu_geglu_store_tail_fp16;
                *store_strip_fn = hmx_glu_geglu_store_strip_fp16;
                return true;
            case HTP_OP_GLU_GEGLU_QUICK:
                *store_quad_fn  = hmx_glu_geglu_quick_store_quad_fp16;
                *store_tail_fn  = hmx_glu_geglu_quick_store_tail_fp16;
                *store_strip_fn = hmx_glu_geglu_quick_store_strip_fp16;
                return true;
            case HTP_OP_GLU_REGLU:
                *store_quad_fn  = hmx_glu_reglu_store_quad_fp16;
                *store_tail_fn  = hmx_glu_reglu_store_tail_fp16;
                *store_strip_fn = hmx_glu_reglu_store_strip_fp16;
                return true;
            default:
                return false;
        }
    }
    *store_strip_fn = NULL;
    switch (glu_op) {
        case HTP_OP_GLU_SWIGLU:
            *store_quad_fn = hmx_glu_swiglu_store_quad_fp32;
            *store_tail_fn = hmx_glu_swiglu_store_tail_fp32;
            return true;
        case HTP_OP_GLU_GEGLU:
            *store_quad_fn = hmx_glu_geglu_store_quad_fp32;
            *store_tail_fn = hmx_glu_geglu_store_tail_fp32;
            return true;
        case HTP_OP_GLU_GEGLU_QUICK:
            *store_quad_fn = hmx_glu_geglu_quick_store_quad_fp32;
            *store_tail_fn = hmx_glu_geglu_quick_store_tail_fp32;
            return true;
        case HTP_OP_GLU_REGLU:
            *store_quad_fn = hmx_glu_reglu_store_quad_fp32;
            *store_tail_fn = hmx_glu_reglu_store_tail_fp32;
            return true;
        default:
            return false;
    }
}

static __attribute__((noinline)) void transfer_output_chunk_fp16_glu_fp16(
        const void * st_void,
        void * restrict dst_ptr,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t c_first,
        uint32_t c_len_and_dst_cols) {
    const hmx_output_glu_state_t * st = (const hmx_output_glu_state_t *) st_void;
    __fp16 * restrict dst             = (__fp16 *) dst_ptr;
    const __fp16 * vtcm_gate          = (const __fp16 *) st->vtcm_gate;
    const __fp16 * vtcm_up            = (const __fp16 *) st->vtcm_up;
    const size_t tile_row_stride      = hmx_ceil_div(st->n_cols, HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
    const size_t c0_first             = c_first >> 5;
    const size_t limit_c              = hex_smin(c_len_and_dst_cols >> 16, c_len_and_dst_cols & 0xffffu);
    const size_t limit_c_aligned      = limit_c & ~63u;
    const size_t dst_stride           = st->dst_stride;

    HVX_Vector g_hf[4] __attribute__((aligned(128)));
    HVX_Vector u_hf[4] __attribute__((aligned(128)));

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = start_row + r_base;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;
        const size_t tile_off   = (r_idx_base >> 5) * tile_row_stride + c0_first * HTP_MM_HMX_TILE_N_ELMS;
        const __fp16 * g_row    = vtcm_gate + tile_off;
        const __fp16 * u_row    = vtcm_up + tile_off;

        for (size_t r = r_base; r < r_end; r += 4) {
            const size_t   r_idx0     = start_row + r;
            const size_t   r1         = (r_idx0 & 31u) >> 1;
            const uint32_t valid_rows = (uint32_t) hex_smin((size_t) 4u, r_end - r);
            __fp16 *       out_row0   = dst + r * dst_stride;

            #pragma clang loop unroll(disable)
            for (size_t c = 0; c < limit_c_aligned; c += 64) {
                const size_t c0   = c >> 5;
                const size_t off0 = c0 * HTP_MM_HMX_TILE_N_ELMS;
                const size_t off1 = off0 + HTP_MM_HMX_TILE_N_ELMS;
                const HVX_VectorPair gp0 = Q6_W_vdeal_VVR(((const HVX_Vector *) (g_row + off1))[r1],
                                                          ((const HVX_Vector *) (g_row + off0))[r1], -2);
                const HVX_VectorPair up0 = Q6_W_vdeal_VVR(((const HVX_Vector *) (u_row + off1))[r1],
                                                          ((const HVX_Vector *) (u_row + off0))[r1], -2);
                g_hf[0] = Q6_V_lo_W(gp0);
                g_hf[1] = Q6_V_hi_W(gp0);
                u_hf[0] = Q6_V_lo_W(up0);
                u_hf[1] = Q6_V_hi_W(up0);
                if (valid_rows > 2u) {
                    const HVX_VectorPair gp1 = Q6_W_vdeal_VVR(((const HVX_Vector *) (g_row + off1))[r1 + 1],
                                                              ((const HVX_Vector *) (g_row + off0))[r1 + 1], -2);
                    const HVX_VectorPair up1 = Q6_W_vdeal_VVR(((const HVX_Vector *) (u_row + off1))[r1 + 1],
                                                              ((const HVX_Vector *) (u_row + off0))[r1 + 1], -2);
                    g_hf[2] = Q6_V_lo_W(gp1);
                    g_hf[3] = Q6_V_hi_W(gp1);
                    u_hf[2] = Q6_V_lo_W(up1);
                    u_hf[3] = Q6_V_hi_W(up1);
                }
                st->store_quad_fn(out_row0 + c, dst_stride, g_hf, u_hf, valid_rows);
            }
            if (limit_c_aligned < limit_c) {
                const size_t   c           = limit_c_aligned;
                const size_t   c0          = c >> 5;
                const size_t   off0        = c0 * HTP_MM_HMX_TILE_N_ELMS;
                const size_t   off1        = off0 + HTP_MM_HMX_TILE_N_ELMS;
                const size_t   valid_c     = limit_c - c;
                const uint32_t valid_bytes = (uint32_t) (valid_c * sizeof(__fp16));
                const HVX_Vector g_t1_0 = (valid_c > 32) ? ((const HVX_Vector *) (g_row + off1))[r1] : Q6_V_vzero();
                const HVX_Vector u_t1_0 = (valid_c > 32) ? ((const HVX_Vector *) (u_row + off1))[r1] : Q6_V_vzero();
                const HVX_VectorPair gp0 = Q6_W_vdeal_VVR(g_t1_0, ((const HVX_Vector *) (g_row + off0))[r1], -2);
                const HVX_VectorPair up0 = Q6_W_vdeal_VVR(u_t1_0, ((const HVX_Vector *) (u_row + off0))[r1], -2);
                g_hf[0] = Q6_V_lo_W(gp0);
                g_hf[1] = Q6_V_hi_W(gp0);
                u_hf[0] = Q6_V_lo_W(up0);
                u_hf[1] = Q6_V_hi_W(up0);
                if (valid_rows > 2u) {
                    const HVX_Vector g_t1_1 = (valid_c > 32) ? ((const HVX_Vector *) (g_row + off1))[r1 + 1] : Q6_V_vzero();
                    const HVX_Vector u_t1_1 = (valid_c > 32) ? ((const HVX_Vector *) (u_row + off1))[r1 + 1] : Q6_V_vzero();
                    const HVX_VectorPair gp1 = Q6_W_vdeal_VVR(g_t1_1, ((const HVX_Vector *) (g_row + off0))[r1 + 1], -2);
                    const HVX_VectorPair up1 = Q6_W_vdeal_VVR(u_t1_1, ((const HVX_Vector *) (u_row + off0))[r1 + 1], -2);
                    g_hf[2] = Q6_V_lo_W(gp1);
                    g_hf[3] = Q6_V_hi_W(gp1);
                    u_hf[2] = Q6_V_lo_W(up1);
                    u_hf[3] = Q6_V_hi_W(up1);
                }
                st->store_tail_fn(out_row0 + c, dst_stride, g_hf, u_hf, (valid_rows << 16) | valid_bytes);
            }
        }
        r_base = r_end;
    }
}

static __attribute__((noinline)) void transfer_output_chunk_fp16_glu_fp32(
        const void * st_void,
        void * restrict dst_ptr,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t c_first,
        uint32_t c_len_and_dst_cols) {
    const hmx_output_glu_state_t * st = (const hmx_output_glu_state_t *) st_void;
    float * restrict dst              = (float *) dst_ptr;
    const __fp16 * vtcm_gate          = (const __fp16 *) st->vtcm_gate;
    const __fp16 * vtcm_up            = (const __fp16 *) st->vtcm_up;
    const size_t tile_row_stride      = hmx_ceil_div(st->n_cols, HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
    const size_t c0_first             = c_first >> 5;
    const size_t limit_c              = hex_smin(c_len_and_dst_cols >> 16, c_len_and_dst_cols & 0xffffu);
    const size_t limit_c_aligned      = limit_c & ~31u;
    const size_t dst_stride           = st->dst_stride;
    const HVX_Vector one              = hvx_vec_splat_f16(1.0f);

    HVX_Vector g_q[4] __attribute__((aligned(128)));
    HVX_Vector u_q[4] __attribute__((aligned(128)));

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = start_row + r_base;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;
        const size_t tile_off   = (r_idx_base >> 5) * tile_row_stride + c0_first * HTP_MM_HMX_TILE_N_ELMS;
        const __fp16 * g_row    = vtcm_gate + tile_off;
        const __fp16 * u_row    = vtcm_up + tile_off;

        for (size_t r = r_base; r < r_end; r += 4) {
            const size_t   r_idx0     = start_row + r;
            const size_t   r1         = (r_idx0 & 31u) >> 1;
            const uint32_t valid_rows = (uint32_t) hex_smin((size_t) 4u, r_end - r);
            float *        out_row0   = dst + r * dst_stride;

            #pragma clang loop unroll(disable)
            for (size_t c = 0; c < limit_c_aligned; c += 32) {
                const size_t off0 = (c >> 5) * HTP_MM_HMX_TILE_N_ELMS;
                const HVX_VectorPair gp0 = Q6_Wqf32_vmpy_VhfVhf(((const HVX_Vector *) (g_row + off0))[r1], one);
                const HVX_VectorPair up0 = Q6_Wqf32_vmpy_VhfVhf(((const HVX_Vector *) (u_row + off0))[r1], one);
                g_q[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(gp0));
                g_q[1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(gp0));
                u_q[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(up0));
                u_q[1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(up0));
                if (valid_rows > 2u) {
                    const HVX_VectorPair gp1 = Q6_Wqf32_vmpy_VhfVhf(((const HVX_Vector *) (g_row + off0))[r1 + 1], one);
                    const HVX_VectorPair up1 = Q6_Wqf32_vmpy_VhfVhf(((const HVX_Vector *) (u_row + off0))[r1 + 1], one);
                    g_q[2] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(gp1));
                    g_q[3] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(gp1));
                    u_q[2] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(up1));
                    u_q[3] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(up1));
                }
                st->store_quad_fn(out_row0 + c, dst_stride, g_q, u_q, valid_rows);
            }
            if (limit_c_aligned < limit_c) {
                const size_t   c           = limit_c_aligned;
                const size_t   off0        = (c >> 5) * HTP_MM_HMX_TILE_N_ELMS;
                const uint32_t valid_bytes = (uint32_t) ((limit_c - c) * sizeof(float));
                const HVX_VectorPair gp0 = Q6_Wqf32_vmpy_VhfVhf(((const HVX_Vector *) (g_row + off0))[r1], one);
                const HVX_VectorPair up0 = Q6_Wqf32_vmpy_VhfVhf(((const HVX_Vector *) (u_row + off0))[r1], one);
                g_q[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(gp0));
                g_q[1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(gp0));
                u_q[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(up0));
                u_q[1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(up0));
                if (valid_rows > 2u) {
                    const HVX_VectorPair gp1 = Q6_Wqf32_vmpy_VhfVhf(((const HVX_Vector *) (g_row + off0))[r1 + 1], one);
                    const HVX_VectorPair up1 = Q6_Wqf32_vmpy_VhfVhf(((const HVX_Vector *) (u_row + off0))[r1 + 1], one);
                    g_q[2] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(gp1));
                    g_q[3] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(gp1));
                    u_q[2] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(up1));
                    u_q[3] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(up1));
                }
                st->store_tail_fn(out_row0 + c, dst_stride, g_q, u_q, (valid_rows << 16) | valid_bytes);
            }
        }
        r_base = r_end;
    }
}

static inline bool resolve_fp16_output_glu_fns(
        uint8_t glu_op,
        size_t dst_elem_size,
        hmx_output_glu_col_chunk_fn_t * col_chunk_fn,
        hmx_glu_store_quad_fn_t * store_quad_fn,
        hmx_glu_store_tail_fn_t * store_tail_fn,
        hmx_glu_store_strip_fn_t * store_strip_fn) {
    if (!resolve_hmx_glu_store_fns(glu_op, dst_elem_size, store_quad_fn, store_tail_fn, store_strip_fn)) {
        return false;
    }
    *col_chunk_fn = (dst_elem_size == sizeof(__fp16)) ? transfer_output_chunk_fp16_glu_fp16
                                                      : transfer_output_chunk_fp16_glu_fp32;
    return true;
}

