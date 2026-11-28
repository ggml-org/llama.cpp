// --- INT HMX W8A8 SM kernels ---

#include "hmx-int-utils.h"

typedef void (*int_rescale_superblock_fn_t)(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride);

typedef struct {
    int_rescale_superblock_fn_t rescale_2k_fn;
    int_rescale_superblock_fn_t rescale_tail_fn;
    uint8_t *                   dst_wt_base;
    uint8_t *                   dst_bias_base;
    float *                     dst_scales_base;
    const uint8_t *             src_raw;
    uint32_t                    n_col_tiles;
    uint32_t                    n_k_tiles;
    uint32_t                    n_groups;
    uint32_t                    tail_tiles;
    uint32_t                    aligned_tile_size;
    uint32_t                    raw_shift;
    uint32_t                    n_tot_units;
    uint32_t                    n_units_per_task;
    uint32_t                    n_tasks;
    struct fastdiv_values       n_groups_div;
} int_weight_rescale_state_t;

static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_x8_lut[128] = {
    0,    0, 8,    0, 16,   0, 24,   0, 32,   0, 48,   0, 64,   0, 96,   0,
    0,    0, 0xf8, 0, 0xf0, 0, 0xe8, 0, 0xe0, 0, 0xd0, 0, 0xc0, 0, 0xa0, 0,
};

static inline void int_hmx_rescale_4bit_lut_step(
        HVX_Vector vq,
        HVX_Vector v_lut,
        HVX_Vector v_m_b4,
        HVX_Vector * restrict tile_dst,
        int pair_idx,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    const HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_lo          = Q6_V_vand_VV(vq, mask_0f);
    HVX_Vector v_hi          = Q6_Vub_vlsr_VubR(vq, 4);
    HVX_Vector v_q0          = Q6_Vb_vlut32_VbVbI(v_lo, v_lut, 0);
    HVX_Vector v_q1          = Q6_Vb_vlut32_VbVbI(v_hi, v_lut, 0);
    int_hmx_rescale_pair_wb(v_q0, v_q1, v_m_b4, tile_dst, pair_idx, v_sum_w, v_l1_w);
}

#define DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(NAME)                                                      \
static __attribute__((noinline)) void rescale_##NAME##_weight_superblock_2k(                              \
        const uint8_t * restrict src_base,                                                                \
        uint8_t * restrict dst_wt,                                                                        \
        float * restrict dst_scales,                                                                      \
        HVX_Vector * restrict bias_out,                                                                   \
        uint32_t tiles_per_group,                                                                         \
        size_t dst_tile_stride) {                                                                         \
    (void) tiles_per_group;                                                                               \
    rescale_##NAME##_weight_superblock_impl(                                                              \
        src_base, dst_wt, dst_scales, bias_out, HTP_MM_HMX_W8A8_KGROUP_TILES, dst_tile_stride);           \
}                                                                                                         \
static __attribute__((noinline)) void rescale_##NAME##_weight_superblock_tail(                            \
        const uint8_t * restrict src_base,                                                                \
        uint8_t * restrict dst_wt,                                                                        \
        float * restrict dst_scales,                                                                      \
        HVX_Vector * restrict bias_out,                                                                   \
        uint32_t tiles_per_group,                                                                         \
        size_t dst_tile_stride) {                                                                         \
    rescale_##NAME##_weight_superblock_impl(                                                              \
        src_base, dst_wt, dst_scales, bias_out, tiles_per_group, dst_tile_stride);                        \
}

static inline HVX_Vector int_hmx_scan_max_f16_scales(
        const uint8_t * restrict src_base,
        size_t tile_stride,
        size_t scale_offset,
        uint32_t tiles_per_group) {

    assert(tiles_per_group >= 2 && (tiles_per_group & 1) == 0);

    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max0 = Q6_V_vzero();
    HVX_Vector v_max1 = Q6_V_vzero();
    const uint8_t * sc_ptr = src_base + scale_offset;
    for (uint32_t kt = 0; kt < tiles_per_group; kt += 2) {
        HVX_Vector v_sc0 = Q6_V_vand_VV(hvx_vmem(sc_ptr), v_abs_mask);
        HVX_Vector v_sc1 = Q6_V_vand_VV(hvx_vmem(sc_ptr + tile_stride), v_abs_mask);
        v_max0 = Q6_Vhf_vmax_VhfVhf(v_max0, v_sc0);
        v_max1 = Q6_Vhf_vmax_VhfVhf(v_max1, v_sc1);
        sc_ptr += 2 * tile_stride;
    }
    return Q6_Vhf_vmax_VhfVhf(v_max0, v_max1);
}

static inline void rescale_q4_0_tile_step(
        const uint8_t * restrict tile_src,
        HVX_Vector * restrict tile_dst,
        HVX_Vector v_m_b4,
        HVX_Vector mask_f0,
        HVX_Vector sign_88,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        HVX_Vector vq_s = Q6_V_vxor_VV(hvx_vmem(tile_src + i * 128), sign_88);
        HVX_Vector v_lo = Q6_V_vand_VV(Q6_Vh_vasl_VhR(vq_s, 4), mask_f0);
        HVX_Vector v_hi = Q6_V_vand_VV(vq_s, mask_f0);
        int_hmx_rescale_pair_wb(v_lo, v_hi, v_m_b4, tile_dst, i, v_sum_w, v_l1_w);
    }
}

// Rescale one (ct, g) superblock of Q4_0 tiles into int8 weight.b tiles,
// compute sum_w and exact l1_w per output channel, and write the Bias Register bank(s) and D_w_eff.
static __attribute__((always_inline)) inline void rescale_q4_0_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    assert(tiles_per_group >= 2 && (tiles_per_group & 1) == 0);

    const size_t     tile_stride     = htp_tensor_get_tile_padded_size(HTP_TYPE_Q4_0);
    const size_t     dst_stride_vecs = dst_tile_stride / sizeof(HVX_Vector);
    const HVX_Vector mask_f0         = Q6_Vb_vsplat_R(0xF0);
    const HVX_Vector sign_88         = Q6_Vb_vsplat_R(0x88);

    HVX_Vector v_max_sc = int_hmx_scan_max_f16_scales(src_base, tile_stride, 512, tiles_per_group);

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    const uint8_t * tile_src = src_base;
    HVX_Vector    * tile_dst = (HVX_Vector *) dst_wt;
    HVX_Vector      v_m_b4   = int_hmx_m_b4_from_f16(hvx_vmem(tile_src + 512), v_inv127_qf);

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt + 1 < tiles_per_group; ++kt) {
        HVX_Vector v_m_next = int_hmx_m_b4_from_f16(hvx_vmem(tile_src + tile_stride + 512), v_inv127_qf);
        rescale_q4_0_tile_step(tile_src, tile_dst, v_m_b4, mask_f0, sign_88, &v_sum_w, &v_l1_w);
        v_m_b4    = v_m_next;
        tile_src += tile_stride;
        tile_dst += dst_stride_vecs;
    }
    rescale_q4_0_tile_step(tile_src, tile_dst, v_m_b4, mask_f0, sign_88, &v_sum_w, &v_l1_w);

    emit_int_hmx_scales_and_bias(v_max_sf, 8.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(q4_0)

static inline void rescale_q8_0_tile_step(
        const uint8_t * restrict tile_src,
        HVX_Vector * restrict tile_dst,
        HVX_Vector v_m_b4,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        HVX_Vector vq0 = hvx_vmem(tile_src + (2 * i + 0) * 128);
        HVX_Vector vq1 = hvx_vmem(tile_src + (2 * i + 1) * 128);
        int_hmx_rescale_pair_wb(vq0, vq1, v_m_b4, tile_dst, i, v_sum_w, v_l1_w);
    }
}

static __attribute__((always_inline)) inline void rescale_q8_0_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    assert(tiles_per_group >= 2 && (tiles_per_group & 1) == 0);

    const size_t tile_stride     = htp_tensor_get_tile_padded_size(HTP_TYPE_Q8_0);
    const size_t dst_stride_vecs = dst_tile_stride / sizeof(HVX_Vector);
    HVX_Vector   v_max_sc        = int_hmx_scan_max_f16_scales(src_base, tile_stride, 1024, tiles_per_group);

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    const uint8_t * tile_src = src_base;
    HVX_Vector    * tile_dst = (HVX_Vector *) dst_wt;
    HVX_Vector      v_m_b4   = int_hmx_m_b4_from_f16(hvx_vmem(tile_src + 1024), v_inv127_qf);

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt + 1 < tiles_per_group; ++kt) {
        HVX_Vector v_m_next = int_hmx_m_b4_from_f16(hvx_vmem(tile_src + tile_stride + 1024), v_inv127_qf);
        rescale_q8_0_tile_step(tile_src, tile_dst, v_m_b4, &v_sum_w, &v_l1_w);
        v_m_b4    = v_m_next;
        tile_src += tile_stride;
        tile_dst += dst_stride_vecs;
    }
    rescale_q8_0_tile_step(tile_src, tile_dst, v_m_b4, &v_sum_w, &v_l1_w);

    emit_int_hmx_scales_and_bias(v_max_sf, 128.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(q8_0)

static inline void rescale_4bit_lut_tile_step(
        const uint8_t * restrict tile_src,
        HVX_Vector * restrict tile_dst,
        HVX_Vector v_lut,
        HVX_Vector v_m_b4,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int_hmx_rescale_4bit_lut_step(hvx_vmem(tile_src + i * 128), v_lut, v_m_b4, tile_dst, i, v_sum_w, v_l1_w);
    }
}

static inline HVX_Vector int_hmx_mxfp4_e8m0_to_f32(HVX_Vector v_raw) {
    HVX_Vector v_e_h     = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v_raw));
    HVX_Vector v_e_w     = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(v_e_h));
    HVX_VectorPred q_nan = Q6_Q_vcmp_eq_VwVw(v_e_w, Q6_V_vsplat_R(0xff));
    return Q6_V_vmux_QVV(q_nan, Q6_V_vzero(), Q6_Vw_vasl_VwR(v_e_w, 23));
}

static __attribute__((always_inline)) inline void rescale_mxfp4_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    assert(tiles_per_group >= 2 && (tiles_per_group & 1) == 0);

    const size_t     tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_MXFP4);
    const HVX_Vector v_lut       = *(const HVX_Vector *) kvalues_mxfp4_x8_lut;
    HVX_Vector       v_max0      = Q6_V_vzero();
    HVX_Vector       v_max1      = Q6_V_vzero();
    const uint8_t  * sc_ptr      = src_base + 512;
    for (uint32_t kt = 0; kt < tiles_per_group; kt += 2) {
        v_max0  = Q6_Vsf_vmax_VsfVsf(v_max0, int_hmx_mxfp4_e8m0_to_f32(hvx_vmem(sc_ptr)));
        v_max1  = Q6_Vsf_vmax_VsfVsf(v_max1, int_hmx_mxfp4_e8m0_to_f32(hvx_vmem(sc_ptr + tile_stride)));
        sc_ptr += 2 * tile_stride;
    }
    HVX_Vector v_max_sf = Q6_Vsf_vmax_VsfVsf(v_max0, v_max1);

    HVX_Vector v_inv127_qf = int_hmx_inv127_from_max_f32(v_max_sf);
    HVX_Vector v_sum_w     = Q6_V_vzero();
    HVX_Vector v_l1_w      = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * tile_stride;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * dst_tile_stride);
        HVX_Vector      v_sc_qf  = hvx_vec_f32_to_qf32(int_hmx_mxfp4_e8m0_to_f32(hvx_vmem(tile_src + 512)));
        HVX_Vector      v_m_b4   = int_hmx_m_b4_from_sc_qf(v_sc_qf, v_inv127_qf);
        rescale_4bit_lut_tile_step(tile_src, tile_dst, v_lut, v_m_b4, &v_sum_w, &v_l1_w);
    }

    emit_int_hmx_scales_and_bias(v_max_sf, 8.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(mxfp4)

static inline void rescale_q5_0_tile_step(
        const uint8_t * restrict tile_src,
        HVX_Vector * restrict tile_dst,
        HVX_Vector v_m_b4,
        HVX_Vector mask_78,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    HVX_Vector v_plane = hvx_vmem(tile_src + 512);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        HVX_Vector vq   = hvx_vmem(tile_src + i * 128);
        HVX_Vector v_lo = Q6_V_vand_VV(Q6_Vh_vasl_VhR(vq, 3), mask_78);
        HVX_Vector v_hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(vq, 1), mask_78);
        v_lo            = Q6_V_vandor_VQR(v_lo, Q6_Q_vand_VR(v_plane, 0x01010101u << (2 * i)),     0x80808080);
        v_hi            = Q6_V_vandor_VQR(v_hi, Q6_Q_vand_VR(v_plane, 0x01010101u << (2 * i + 1)), 0x80808080);
        int_hmx_rescale_pair_wb(v_lo, v_hi, v_m_b4, tile_dst, i, v_sum_w, v_l1_w);
    }
}

static __attribute__((always_inline)) inline void rescale_q5_0_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    assert(tiles_per_group >= 2 && (tiles_per_group & 1) == 0);

    const size_t     tile_stride     = htp_tensor_get_tile_padded_size(HTP_TYPE_Q5_0);
    const size_t     dst_stride_vecs = dst_tile_stride / sizeof(HVX_Vector);
    const HVX_Vector mask_78         = Q6_Vb_vsplat_R(0x78);
    HVX_Vector       v_max_sc        = int_hmx_scan_max_f16_scales(src_base, tile_stride, 640, tiles_per_group);

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    const uint8_t * tile_src = src_base;
    HVX_Vector    * tile_dst = (HVX_Vector *) dst_wt;
    HVX_Vector      v_m_b4   = int_hmx_m_b4_from_f16(hvx_vmem(tile_src + 640), v_inv127_qf);

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt + 1 < tiles_per_group; ++kt) {
        HVX_Vector v_m_next = int_hmx_m_b4_from_f16(hvx_vmem(tile_src + tile_stride + 640), v_inv127_qf);
        rescale_q5_0_tile_step(tile_src, tile_dst, v_m_b4, mask_78, &v_sum_w, &v_l1_w);
        v_m_b4    = v_m_next;
        tile_src += tile_stride;
        tile_dst += dst_stride_vecs;
    }
    rescale_q5_0_tile_step(tile_src, tile_dst, v_m_b4, mask_78, &v_sum_w, &v_l1_w);

    emit_int_hmx_scales_and_bias(v_max_sf, 16.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(q5_0)

static __attribute__((always_inline)) inline void rescale_iq4_nl_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    assert(tiles_per_group >= 2 && (tiles_per_group & 1) == 0);

    const size_t     tile_stride     = htp_tensor_get_tile_padded_size(HTP_TYPE_IQ4_NL);
    const size_t     dst_stride_vecs = dst_tile_stride / sizeof(HVX_Vector);
    const HVX_Vector v_lut           = *(const HVX_Vector *) kvalues_iq4nl_lut;
    HVX_Vector       v_max_sc        = int_hmx_scan_max_f16_scales(src_base, tile_stride, 512, tiles_per_group);

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    const uint8_t * tile_src = src_base;
    HVX_Vector    * tile_dst = (HVX_Vector *) dst_wt;

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        HVX_Vector v_m_b4 = int_hmx_m_b4_from_f16(hvx_vmem(tile_src + 512), v_inv127_qf);
        rescale_4bit_lut_tile_step(tile_src, tile_dst, v_lut, v_m_b4, &v_sum_w, &v_l1_w);
        tile_src += tile_stride;
        tile_dst += dst_stride_vecs;
    }

    emit_int_hmx_scales_and_bias(v_max_sf, 128.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(iq4_nl)

static inline void rescale_q1_0_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    const size_t     tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q1_0);
    const uint32_t   raw_tiles   = tiles_per_group >> 2;
    const HVX_Vector v_abs_mask  = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t rt = 0; rt < raw_tiles; ++rt) {
        const uint8_t * tile_src = src_base + (size_t) rt * tile_stride;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 512), v_abs_mask));
    }

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    for (uint32_t rt = 0; rt < raw_tiles; ++rt) {
        const uint8_t * tile_src = src_base + (size_t) rt * tile_stride;
        HVX_Vector      v_pos_m  = int_hmx_m_b4_from_f16(hvx_vmem(tile_src + 512), v_inv127_qf);
        HVX_Vector      v_neg_m  = Q6_Vb_vsub_VbVb(Q6_V_vzero(), v_pos_m);

        for (uint32_t q = 0; q < 4; ++q) {
            HVX_Vector   v_plane  = hvx_vmem(tile_src + q * 128);
            HVX_Vector * tile_dst = (HVX_Vector *) (dst_wt + (size_t) (4 * rt + q) * dst_tile_stride);
            for (int i = 0; i < 4; ++i) {
                HVX_Vector v_W0 = unpack_q1_0_group(v_plane, 2 * i + 0, v_neg_m, v_pos_m);
                HVX_Vector v_W1 = unpack_q1_0_group(v_plane, 2 * i + 1, v_neg_m, v_pos_m);
                int_hmx_store_and_accum_pair(tile_dst, i, v_W0, v_W1, &v_sum_w, &v_l1_w);
            }
        }
    }

    emit_int_hmx_scales_and_bias(v_max_sf, 1.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(q1_0)

static inline void rescale_q2_0_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    const size_t     tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q2_0);
    const uint32_t   raw_tiles   = tiles_per_group >> 1;
    const HVX_Vector mask_03     = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector i32         = Q6_Vb_vsplat_R(32);
    const HVX_Vector v_abs_mask  = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t rt = 0; rt < raw_tiles; ++rt) {
        const uint8_t * tile_src = src_base + (size_t) rt * tile_stride;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 512), v_abs_mask));
    }

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    for (uint32_t rt = 0; rt < raw_tiles; ++rt) {
        const HVX_Vector * vptr   = (const HVX_Vector *) (src_base + (size_t) rt * tile_stride);
        HVX_Vector         v_m_b4 = int_hmx_m_b4_from_f16(vptr[4], v_inv127_qf);

        for (uint32_t half = 0; half < 2; ++half) {
            HVX_Vector * tile_dst = (HVX_Vector *) (dst_wt + (size_t) (2 * rt + half) * dst_tile_stride);
            for (int i = 0; i < 4; ++i) {
                HVX_Vector v_q0 = Q6_Vb_vsub_VbVb(Q6_Vh_vasl_VhR(unpack_q3_k_low2(vptr + 2 * half, 2 * i + 0, mask_03), 5), i32);
                HVX_Vector v_q1 = Q6_Vb_vsub_VbVb(Q6_Vh_vasl_VhR(unpack_q3_k_low2(vptr + 2 * half, 2 * i + 1, mask_03), 5), i32);
                int_hmx_rescale_pair_wb(v_q0, v_q1, v_m_b4, tile_dst, i, &v_sum_w, &v_l1_w);
            }
        }
    }

    emit_int_hmx_scales_and_bias(v_max_sf, 4.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(q2_0)

static inline void rescale_nvfp4_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    const size_t     tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_NVFP4);
    const HVX_Vector v_lut       = *(const HVX_Vector *) kvalues_mxfp4_x8_lut;
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * tile_stride;
        HVX_Vector v_sc = nvfp4_ue4m3_to_f16(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmem(tile_src + NVFP4_SCALE_OFF))));
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, v_sc);
    }
    v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vror_VR(v_max_sc, 64));

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * tile_stride;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * dst_tile_stride);
        HVX_Vector      v_sc     = nvfp4_ue4m3_to_f16(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmem(tile_src + NVFP4_SCALE_OFF))));
        HVX_Vector      v_m_b0, v_m_b1;
        int_hmx_m_b4_pair_from_f16(v_sc, v_inv127_qf, &v_m_b0, &v_m_b1);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (int i = 0; i < 4; ++i) {
            HVX_Vector v_m_b4 = (i < 2) ? v_m_b0 : v_m_b1;
            int_hmx_rescale_4bit_lut_step(hvx_vmem(tile_src + i * 128), v_lut, v_m_b4, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    emit_int_hmx_scales_and_bias(v_max_sf, 8.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(nvfp4)

static inline HVX_Vector unpack_q3_k_group_shl5(const HVX_Vector * restrict vptr, int g, HVX_Vector mask_03) {
    HVX_VectorPred q_neg = Q6_Q_vand_VR(vptr[2], 0x01010101u << g);
    HVX_Vector     v_lo2 = Q6_Vh_vasl_VhR(unpack_q3_k_low2(vptr, g, mask_03), 5);
    return Q6_V_vandor_VQR(v_lo2, q_neg, 0x80808080);
}

static inline void rescale_q3_k_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    const size_t     tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q3_K);
    const HVX_Vector mask_03     = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector v_abs_mask  = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * tile_stride;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 384), v_abs_mask));
    }
    v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vror_VR(v_max_sc, 64));

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const HVX_Vector * vptr     = (const HVX_Vector *) (src_base + (size_t) kt * tile_stride);
        HVX_Vector       * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * dst_tile_stride);
        HVX_Vector         v_m_b0, v_m_b1;
        int_hmx_m_b4_pair_from_f16(vptr[3], v_inv127_qf, &v_m_b0, &v_m_b1);

        for (int i = 0; i < 4; ++i) {
            HVX_Vector v_m_b4 = (i < 2) ? v_m_b0 : v_m_b1;
            HVX_Vector v_q0   = unpack_q3_k_group_shl5(vptr, 2 * i + 0, mask_03);
            HVX_Vector v_q1   = unpack_q3_k_group_shl5(vptr, 2 * i + 1, mask_03);
            int_hmx_rescale_pair_wb(v_q0, v_q1, v_m_b4, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    emit_int_hmx_scales_and_bias(v_max_sf, 4.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(q3_k)

static inline void rescale_q6_k_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    const size_t     tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q6_K);
    const HVX_Vector mask_0f     = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector mask_03     = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector sign_80     = Q6_Vb_vsplat_R(0x80);
    const HVX_Vector v_abs_mask  = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * tile_stride;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 768), v_abs_mask));
    }
    v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vror_VR(v_max_sc, 64));

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const HVX_Vector * vptr     = (const HVX_Vector *) (src_base + (size_t) kt * tile_stride);
        HVX_Vector       * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * dst_tile_stride);
        HVX_Vector         v_m_b0, v_m_b1;
        int_hmx_m_b4_pair_from_f16(vptr[6], v_inv127_qf, &v_m_b0, &v_m_b1);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (int i = 0; i < 4; ++i) {
            HVX_Vector v_m_b4 = (i < 2) ? v_m_b0 : v_m_b1;
            HVX_Vector v_q0   = Q6_V_vxor_VV(Q6_Vh_vasl_VhR(unpack_q6_k_group(vptr, 2 * i + 0, mask_0f, mask_03, Q6_V_vzero()), 2), sign_80);
            HVX_Vector v_q1   = Q6_V_vxor_VV(Q6_Vh_vasl_VhR(unpack_q6_k_group(vptr, 2 * i + 1, mask_0f, mask_03, Q6_V_vzero()), 2), sign_80);
            int_hmx_rescale_pair_wb(v_q0, v_q1, v_m_b4, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    emit_int_hmx_scales_and_bias(v_max_sf, 32.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(q6_k)

static inline void int_hmx_pack_q6_pair(
        HVX_Vector v_hi0,
        HVX_Vector v_lo0,
        HVX_Vector v_hi1,
        HVX_Vector v_lo1,
        HVX_Vector * restrict v_W0_out,
        HVX_Vector * restrict v_W1_out) {
#if __HVX_ARCH__ >= 79
    *v_W0_out = Q6_Vb_vasr_VhVhR_rnd_sat(v_hi0, v_lo0, 6);
    *v_W1_out = Q6_Vb_vasr_VhVhR_rnd_sat(v_hi1, v_lo1, 6);
#else
    *v_W0_out = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(v_hi0, 2), Q6_Vh_vasl_VhR(v_lo0, 2));
    *v_W1_out = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(v_hi1, 2), Q6_Vh_vasl_VhR(v_lo1, 2));
#endif
}

static inline HVX_Vector int_hmx_asym_update_wmax_qf(
        HVX_Vector v_wmax_sf,
        HVX_Vector vd_qf,
        HVX_Vector vm_qf,
        HVX_Vector v_qmax_qf) {

    const HVX_Vector v_abs_mask_sf = Q6_V_vsplat_R(0x7fffffff);
    HVX_Vector vm_sf               = Q6_Vsf_equals_Vqf32(vm_qf);
    HVX_Vector v_end               = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(vd_qf, v_qmax_qf), vm_qf));
    HVX_Vector v_ext0              = Q6_V_vand_VV(vm_sf, v_abs_mask_sf);
    HVX_Vector v_ext1              = Q6_V_vand_VV(v_end, v_abs_mask_sf);
    return Q6_Vsf_vmax_VsfVsf(v_wmax_sf, Q6_Vsf_vmax_VsfVsf(v_ext0, v_ext1));
}

static inline void int_hmx_asym_alpha_beta_wb(
        HVX_Vector vd_qf,
        HVX_Vector vm_qf,
        HVX_Vector v_scale_q6_qf,
        HVX_Vector * restrict v_alpha_h_out,
        HVX_Vector * restrict v_beta_h_out) {

    HVX_Vector v_a_bias      = hvx_vec_i32_from_qf32_rnd(Q6_Vqf32_vmpy_Vqf32Vqf32(vd_qf, v_scale_q6_qf));
    HVX_Vector v_b_bias      = hvx_vec_i32_from_qf32_rnd(Q6_Vqf32_vmpy_Vqf32Vqf32(vm_qf, v_scale_q6_qf));
    *v_alpha_h_out           = Q6_Vh_vshuffe_VhVh(v_a_bias, v_a_bias);
    *v_beta_h_out            = Q6_Vh_vshuffe_VhVh(v_b_bias, v_b_bias);
}

static inline void int_hmx_rescale_asym_pair_wb(
        HVX_Vector v_q0,
        HVX_Vector v_q1,
        HVX_Vector v_alpha_h,
        HVX_Vector v_beta_h,
        HVX_Vector * restrict tile_dst,
        int pair_idx,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    const HVX_Vector v_one_s8 = Q6_Vb_vsplat_R(1);
    HVX_VectorPair vp0        = Q6_Wh_vmpy_VbVb(v_q0, v_one_s8);
    HVX_VectorPair vp1        = Q6_Wh_vmpy_VbVb(v_q1, v_one_s8);
    HVX_Vector v_p0_e         = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_lo_W(vp0), v_alpha_h);
    HVX_Vector v_p0_o         = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_hi_W(vp0), v_alpha_h);
    HVX_Vector v_p1_e         = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_lo_W(vp1), v_alpha_h);
    HVX_Vector v_p1_o         = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_hi_W(vp1), v_alpha_h);
    HVX_Vector v_W0, v_W1;
    int_hmx_pack_q6_pair(v_p0_o, v_p0_e, v_p1_o, v_p1_e, &v_W0, &v_W1);
    int_hmx_store_and_accum_pair(tile_dst, pair_idx, v_W0, v_W1, v_sum_w, v_l1_w);
}

static inline void int_hmx_iq2xxs_decode_pair(
        HVX_Vector v_c2,
        HVX_Vector v_sgn,
        int l,
        HVX_Vector v_nib_lut,
        HVX_Vector * restrict v_w0_out,
        HVX_Vector * restrict v_w1_out) {

    const HVX_Vector v_zero     = Q6_V_vzero();
    const HVX_Vector v_one      = Q6_V_vsplat_R(1);
    const HVX_Vector v_sgn_mask = Q6_V_vsplat_R(IQ2XXS_SIGN_MASK);

    HVX_Vector v_s7     = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(v_sgn, IQ2XXS_SIGN_BITS * l), v_sgn_mask);
    HVX_Vector v_t      = Q6_Vh_vpopcount_Vh(v_s7);
    HVX_Vector v_s      = Q6_V_vor_VV(v_s7, Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_t, v_one), IQ2XXS_PARITY_SHIFT));
    HVX_Vector v_s16    = Q6_V_vor_VV(v_s, Q6_Vw_vasl_VwR(v_s, 8));
    HVX_Vector v_rep    = Q6_V_vor_VV(v_s16, Q6_Vw_vasl_VwR(v_s16, 16));
    HVX_VectorPred q_lo = Q6_Q_vand_VR(v_rep, IQ2XXS_SIGN_PRED_LO);
    HVX_VectorPred q_hi = Q6_Q_vand_VR(v_rep, IQ2XXS_SIGN_PRED_HI);

    HVX_Vector v_d      = (l & 1) ? Q6_Vh_vshuffo_VhVh(v_c2, v_c2) : Q6_Vh_vshuffe_VhVh(v_c2, v_c2);
    HVX_Vector v_n      = Q6_V_vor_VV(Q6_V_vand_VV(v_d, Q6_V_vsplat_R(IQ2XXS_NIB_LO_MASK)),
                                      Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(v_d, 4), Q6_V_vsplat_R(IQ2XXS_NIB_HI_MASK)));
    HVX_VectorPair p_g  = Q6_Wh_vlut16_VbVhR(v_n, v_nib_lut, 0);
    HVX_Vector v_g0     = Q6_V_lo_W(p_g);
    HVX_Vector v_g1     = Q6_V_hi_W(p_g);

    *v_w0_out = Q6_V_vmux_QVV(q_lo, Q6_Vb_vsub_VbVb(v_zero, v_g0), v_g0);
    *v_w1_out = Q6_V_vmux_QVV(q_hi, Q6_Vb_vsub_VbVb(v_zero, v_g1), v_g1);
}

static inline HVX_Vector int_hmx_iq2xxs_gather_d(const uint8_t * restrict sb) {
    const size_t tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_IQ2_XXS);
    HVX_Vector v_dd = Q6_V_vzero();
    #pragma clang loop pipeline(disable) unroll(disable)
    for (int j = 0; j < IQ2XXS_TILES_PER_SB; j++) {
        HVX_Vector v = hvx_vmem(sb + (size_t) j * tile_stride + IQ2XXS_D_SLICE_OFF);
        v_dd = Q6_V_vmux_QVV(Q6_Q_vsetq_R(IQ2XXS_D_SLICE_BYTES * j), v_dd,
                             Q6_V_vror_VR(v, (VLEN - IQ2XXS_D_SLICE_BYTES * j) & (VLEN - 1)));
    }
    return v_dd;
}

static inline void rescale_iq2_xxs_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    const size_t     tile_stride   = htp_tensor_get_tile_padded_size(HTP_TYPE_IQ2_XXS);
    const HVX_Vector v_abs_mask_sf = Q6_V_vsplat_R(0x7fffffff);
    const HVX_Vector v_one_w       = Q6_V_vsplat_R(1);
    const HVX_Vector v_one_s8      = Q6_Vb_vsplat_R(1);
    const HVX_Vector v_nib_lut     = hvx_vmem(iq2xxs_vtcm_lut);

    HVX_Vector v_max_sf = Q6_V_vzero();
    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t sb_kt = 0; sb_kt < tiles_per_group; sb_kt += IQ2XXS_TILES_PER_SB) {
        const uint8_t * sb      = src_base + (size_t) sb_kt * tile_stride;
        HVX_Vector v_d_sf       = Q6_V_lo_W(hvx_vec_f16_to_f32(int_hmx_iq2xxs_gather_d(sb)));
        HVX_Vector v_d_abs_sf   = Q6_V_vand_VV(v_d_sf, v_abs_mask_sf);
        HVX_Vector v_max_sc_u32 = Q6_V_vzero();

        #pragma clang loop pipeline(disable) unroll(disable)
        for (uint32_t j = 0; j < IQ2XXS_TILES_PER_SB; ++j) {
            const uint8_t * tile_src = sb + (size_t) j * tile_stride;
            HVX_Vector v_sgn         = Q6_V_hi_W(Q6_W_vdeal_VVR(hvx_vmem(tile_src + VLEN), hvx_vmem(tile_src), -IQ2XXS_WORD_BYTES));
            v_max_sc_u32             = Q6_Vw_vmax_VwVw(v_max_sc_u32, Q6_Vuw_vlsr_VuwR(v_sgn, IQ2XXS_SCALE_SHIFT));
        }

        HVX_Vector v_max_ls = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(v_max_sc_u32, 1), v_one_w);
        v_max_sf            = Q6_Vsf_vmax_VsfVsf(v_max_sf, hvx_vec_mul_f32_f32(v_d_abs_sf, Q6_Vsf_equals_Vw(v_max_ls)));
    }

    HVX_Vector v_scale_q6_qf = int_hmx_scale_qf_from_max_f32(v_max_sf, 8128.0f / 43.0f);
    HVX_Vector v_sum_w       = Q6_V_vzero();
    HVX_Vector v_l1_w        = Q6_V_vzero();
    HVX_Vector v_sgn         = iq2xxs_gather_issue(src_base);

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t sb_kt = 0; sb_kt < tiles_per_group; sb_kt += IQ2XXS_TILES_PER_SB) {
        const uint8_t * sb    = src_base + (size_t) sb_kt * tile_stride;
        HVX_Vector v_d_qf     = hvx_vec_f32_to_qf32(Q6_V_lo_W(hvx_vec_f16_to_f32(int_hmx_iq2xxs_gather_d(sb))));
        HVX_Vector v_d_sc_qf  = Q6_Vqf32_vmpy_Vqf32Vqf32(v_d_qf, v_scale_q6_qf);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (uint32_t j = 0; j < IQ2XXS_TILES_PER_SB; ++j) {
            const uint32_t  kt       = sb_kt + j;
            const uint8_t * tile_src = sb + (size_t) j * tile_stride;
            HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * dst_tile_stride);

            HVX_Vector v_c[2];
            iq2xxs_gather_wait(tile_src, v_c);
            const HVX_Vector v_sgn_cur = v_sgn;
            if (kt + 1 < tiles_per_group) {
                v_sgn = iq2xxs_gather_issue(tile_src + tile_stride);
            }

            HVX_Vector v_ls      = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_Vuw_vlsr_VuwR(v_sgn_cur, IQ2XXS_SCALE_SHIFT), 1), v_one_w);
            HVX_Vector v_ls_qf   = hvx_vec_f32_to_qf32(Q6_Vsf_equals_Vw(v_ls));
            HVX_Vector v_m_bias  = hvx_vec_i32_from_qf32_rnd(Q6_Vqf32_vmpy_Vqf32Vqf32(v_d_sc_qf, v_ls_qf));
            HVX_Vector v_alpha_h = Q6_Vh_vshuffe_VhVh(v_m_bias, v_m_bias);

            #pragma clang loop pipeline(disable) unroll(disable)
            for (int i = 0; i < 4; ++i) {
                HVX_Vector v_w0, v_w1;
                int_hmx_iq2xxs_decode_pair((i < 2) ? v_c[0] : v_c[1], v_sgn_cur, i, v_nib_lut, &v_w0, &v_w1);
                HVX_VectorPair vp0 = Q6_Wh_vmpy_VbVb(v_w0, v_one_s8);
                HVX_VectorPair vp1 = Q6_Wh_vmpy_VbVb(v_w1, v_one_s8);
                HVX_Vector v_p0_e  = Q6_Vh_vmpyi_VhVh(Q6_V_lo_W(vp0), v_alpha_h);
                HVX_Vector v_p0_o  = Q6_Vh_vmpyi_VhVh(Q6_V_hi_W(vp0), v_alpha_h);
                HVX_Vector v_p1_e  = Q6_Vh_vmpyi_VhVh(Q6_V_lo_W(vp1), v_alpha_h);
                HVX_Vector v_p1_o  = Q6_Vh_vmpyi_VhVh(Q6_V_hi_W(vp1), v_alpha_h);
                HVX_Vector v_W0, v_W1;
                int_hmx_pack_q6_pair(v_p0_o, v_p0_e, v_p1_o, v_p1_e, &v_W0, &v_W1);
                int_hmx_store_and_accum_pair(tile_dst, i, v_W0, v_W1, &v_sum_w, &v_l1_w);
            }
        }
    }

    emit_int_hmx_scales_and_bias(v_max_sf, 0.125f * 43.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(iq2_xxs)

static inline void rescale_q4_1_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    const size_t     tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q4_1);
    const HVX_Vector mask_0f     = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector v_one_f16   = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_qmax_qf   = hvx_vec_f32_to_qf32(hvx_vec_splat_f32(15.0f));

    HVX_Vector v_wmax_sf = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * tile_stride;
        HVX_VectorPair vp_dm_qf  = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(tile_src + 512), v_one_f16);
        v_wmax_sf                = int_hmx_asym_update_wmax_qf(v_wmax_sf, Q6_V_lo_W(vp_dm_qf), Q6_V_hi_W(vp_dm_qf), v_qmax_qf);
    }

    HVX_Vector v_scale_q6_qf = int_hmx_scale_qf_from_max_f32(v_wmax_sf, 8128.0f);
    HVX_Vector v_sum_w       = Q6_V_vzero();
    HVX_Vector v_l1_w        = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * tile_stride;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * dst_tile_stride);
        HVX_VectorPair vp_dm_qf  = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(tile_src + 512), v_one_f16);
        HVX_Vector v_alpha_h, v_beta_h;
        int_hmx_asym_alpha_beta_wb(Q6_V_lo_W(vp_dm_qf), Q6_V_hi_W(vp_dm_qf), v_scale_q6_qf, &v_alpha_h, &v_beta_h);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (int i = 0; i < 4; ++i) {
            HVX_Vector vq   = hvx_vmem(tile_src + i * 128);
            HVX_Vector v_lo = Q6_V_vand_VV(vq, mask_0f);
            HVX_Vector v_hi = Q6_Vub_vlsr_VubR(vq, 4);
            int_hmx_rescale_asym_pair_wb(v_lo, v_hi, v_alpha_h, v_beta_h, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    emit_int_hmx_scales_and_bias(v_wmax_sf, 1.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(q4_1)

static inline void rescale_q5_k_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    const size_t     tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q5_K);
    const HVX_Vector mask_0f     = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector v_one_f16   = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_qmax_qf   = hvx_vec_f32_to_qf32(hvx_vec_splat_f32(31.0f));

    HVX_Vector v_wmax_sf = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * tile_stride;
        HVX_VectorPair vp_dm_qf  = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(tile_src + 512), v_one_f16);
        v_wmax_sf                = int_hmx_asym_update_wmax_qf(v_wmax_sf, Q6_V_lo_W(vp_dm_qf), Q6_V_hi_W(vp_dm_qf), v_qmax_qf);
    }

    HVX_Vector v_scale_q6_qf = int_hmx_scale_qf_from_max_f32(v_wmax_sf, 8128.0f);
    HVX_Vector v_sum_w       = Q6_V_vzero();
    HVX_Vector v_l1_w        = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * tile_stride;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * dst_tile_stride);
        HVX_Vector      v_plane  = hvx_vmem(tile_src + 640);
        HVX_VectorPair  vp_dm_qf = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(tile_src + 512), v_one_f16);
        HVX_Vector v_alpha_h, v_beta_h;
        int_hmx_asym_alpha_beta_wb(Q6_V_lo_W(vp_dm_qf), Q6_V_hi_W(vp_dm_qf), v_scale_q6_qf, &v_alpha_h, &v_beta_h);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (int i = 0; i < 4; ++i) {
            HVX_Vector vq   = hvx_vmem(tile_src + i * 128);
            HVX_Vector v_lo = hvx_q5k_or_hibit(Q6_V_vand_VV(vq, mask_0f), v_plane, 2 * i + 0);
            HVX_Vector v_hi = hvx_q5k_or_hibit(Q6_Vub_vlsr_VubR(vq, 4),  v_plane, 2 * i + 1);
            int_hmx_rescale_asym_pair_wb(v_lo, v_hi, v_alpha_h, v_beta_h, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    emit_int_hmx_scales_and_bias(v_wmax_sf, 1.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(q5_k)

static inline void int_hmx_q2_k_load_dmin(
        const uint8_t * restrict sb,
        HVX_Vector * restrict v_d,
        HVX_Vector * restrict v_m) {

    const size_t tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q2_K);
    HVX_Vector v_dd = Q6_V_vzero();
    #pragma clang loop pipeline(disable) unroll(disable)
    for (int j = 0; j < 8; j++) {
        HVX_Vector v = hvx_vmem(sb + (size_t) j * tile_stride + 256);
        v_dd = Q6_V_vmux_QVV(Q6_Q_vsetq_R(16 * j), v_dd, Q6_V_vror_VR(v, (64 - 16 * j) & 127));
    }
    HVX_VectorPred q_lo = Q6_Q_vsetq_R(64);
    HVX_Vector     v_r  = Q6_V_vror_VR(v_dd, 64);
    *v_d = Q6_V_vmux_QVV(q_lo, v_dd, v_r);
    *v_m = Q6_V_vmux_QVV(q_lo, v_r, v_dd);
}

static inline void rescale_q2_k_weight_superblock_impl(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        size_t dst_tile_stride) {

    const size_t     tile_stride = htp_tensor_get_tile_padded_size(HTP_TYPE_Q2_K);
    const HVX_Vector v_one_f16   = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_qmax_qf   = hvx_vec_f32_to_qf32(hvx_vec_splat_f32(3.0f));

    HVX_Vector v_wmax_sf = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t sb_kt = 0; sb_kt < tiles_per_group; sb_kt += 8) {
        const uint8_t * sb = src_base + (size_t) sb_kt * tile_stride;
        HVX_Vector v_d, v_dm;
        int_hmx_q2_k_load_dmin(sb, &v_d, &v_dm);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (uint32_t j = 0; j < 8; ++j) {
            const uint8_t * tile_src = sb + (size_t) j * tile_stride;
            HVX_Vector v_D, v_M;
            q2_k_unpack_scales(tile_src, v_d, v_dm, &v_D, &v_M);
            HVX_VectorPair vp_d_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_D), v_one_f16);
            HVX_VectorPair vp_m_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_M), v_one_f16);
            v_wmax_sf = int_hmx_asym_update_wmax_qf(v_wmax_sf, Q6_V_lo_W(vp_d_qf), Q6_V_lo_W(vp_m_qf), v_qmax_qf);
            v_wmax_sf = int_hmx_asym_update_wmax_qf(v_wmax_sf, Q6_V_hi_W(vp_d_qf), Q6_V_hi_W(vp_m_qf), v_qmax_qf);
        }
    }

    HVX_Vector v_scale_q6_qf = int_hmx_scale_qf_from_max_f32(v_wmax_sf, 8128.0f);
    HVX_Vector v_sum_w       = Q6_V_vzero();
    HVX_Vector v_l1_w        = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t sb_kt = 0; sb_kt < tiles_per_group; sb_kt += 8) {
        const uint8_t * sb = src_base + (size_t) sb_kt * tile_stride;
        HVX_Vector v_d, v_dm;
        int_hmx_q2_k_load_dmin(sb, &v_d, &v_dm);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t     kt       = sb_kt + j;
            const uint8_t    * tile_src = sb + (size_t) j * tile_stride;
            const HVX_Vector * vptr     = (const HVX_Vector *) tile_src;
            HVX_Vector       * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * dst_tile_stride);

            HVX_Vector v_D, v_M;
            q2_k_unpack_scales(tile_src, v_d, v_dm, &v_D, &v_M);

            #pragma clang loop pipeline(disable) unroll(disable)
            for (int sub = 0; sub < 2; ++sub) {
                HVX_Vector v_D_sub = (sub == 0) ? v_D : Q6_V_vror_VR(v_D, 64);
                HVX_Vector v_M_sub = (sub == 0) ? v_M : Q6_V_vror_VR(v_M, 64);
                HVX_Vector v_d_qf  = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_D_sub), v_one_f16));
                HVX_Vector v_m_qf  = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_M_sub), v_one_f16));
                HVX_Vector v_alpha_h, v_beta_h;
                int_hmx_asym_alpha_beta_wb(v_d_qf, v_m_qf, v_scale_q6_qf, &v_alpha_h, &v_beta_h);

                const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);
                for (int p = 0; p < 2; ++p) {
                    const int  i    = 2 * sub + p;
                    HVX_Vector v_q0 = unpack_q3_k_low2(vptr, 2 * i + 0, mask_03);
                    HVX_Vector v_q1 = unpack_q3_k_low2(vptr, 2 * i + 1, mask_03);
                    int_hmx_rescale_asym_pair_wb(v_q0, v_q1, v_alpha_h, v_beta_h, tile_dst, i, &v_sum_w, &v_l1_w);
                }
            }
        }
    }

    emit_int_hmx_scales_and_bias(v_wmax_sf, 1.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}
DEFINE_RESCALE_WEIGHT_SUPERBLOCK_ENTRY(q2_k)

static inline void resolve_int_weight_rescale_fns(
        int32_t weight_type,
        int_rescale_superblock_fn_t * restrict fn_2k,
        int_rescale_superblock_fn_t * restrict fn_tail) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:    *fn_2k = rescale_q4_0_weight_superblock_2k;    *fn_tail = rescale_q4_0_weight_superblock_tail;    break;
        case HTP_TYPE_Q8_0:    *fn_2k = rescale_q8_0_weight_superblock_2k;    *fn_tail = rescale_q8_0_weight_superblock_tail;    break;
        case HTP_TYPE_IQ4_NL:  *fn_2k = rescale_iq4_nl_weight_superblock_2k;  *fn_tail = rescale_iq4_nl_weight_superblock_tail;  break;
        case HTP_TYPE_MXFP4:   *fn_2k = rescale_mxfp4_weight_superblock_2k;   *fn_tail = rescale_mxfp4_weight_superblock_tail;   break;
        case HTP_TYPE_Q5_0:    *fn_2k = rescale_q5_0_weight_superblock_2k;    *fn_tail = rescale_q5_0_weight_superblock_tail;    break;
        case HTP_TYPE_Q1_0:    *fn_2k = rescale_q1_0_weight_superblock_2k;    *fn_tail = rescale_q1_0_weight_superblock_tail;    break;
        case HTP_TYPE_Q2_0:    *fn_2k = rescale_q2_0_weight_superblock_2k;    *fn_tail = rescale_q2_0_weight_superblock_tail;    break;
        case HTP_TYPE_NVFP4:   *fn_2k = rescale_nvfp4_weight_superblock_2k;   *fn_tail = rescale_nvfp4_weight_superblock_tail;   break;
        case HTP_TYPE_Q3_K:    *fn_2k = rescale_q3_k_weight_superblock_2k;    *fn_tail = rescale_q3_k_weight_superblock_tail;    break;
        case HTP_TYPE_Q6_K:    *fn_2k = rescale_q6_k_weight_superblock_2k;    *fn_tail = rescale_q6_k_weight_superblock_tail;    break;
        case HTP_TYPE_IQ2_XXS: *fn_2k = rescale_iq2_xxs_weight_superblock_2k; *fn_tail = rescale_iq2_xxs_weight_superblock_tail; break;
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q4_K:    *fn_2k = rescale_q4_1_weight_superblock_2k;    *fn_tail = rescale_q4_1_weight_superblock_tail;    break;
        case HTP_TYPE_Q5_1:
        case HTP_TYPE_Q5_K:    *fn_2k = rescale_q5_k_weight_superblock_2k;    *fn_tail = rescale_q5_k_weight_superblock_tail;    break;
        case HTP_TYPE_Q2_K:    *fn_2k = rescale_q2_k_weight_superblock_2k;    *fn_tail = rescale_q2_k_weight_superblock_tail;    break;
        default:               *fn_2k = NULL;                                 *fn_tail = NULL;                                   break;
    }
}

static void rescale_int_weight_to_int8_task(
        const int_weight_rescale_state_t * state,
        uint32_t start_unit,
        uint32_t end_unit) {

    const uint32_t                    n_k_tiles         = state->n_k_tiles;
    const uint32_t                    n_groups          = state->n_groups;
    const uint32_t                    tail_tiles        = state->tail_tiles;
    const uint32_t                    aligned_tile_size = state->aligned_tile_size;
    const uint32_t                    raw_shift         = state->raw_shift;
    const int_rescale_superblock_fn_t rescale_2k_fn     = state->rescale_2k_fn;
    const int_rescale_superblock_fn_t tail_fn           = (tail_tiles == HTP_MM_HMX_W8A8_KGROUP_TILES)
                                                        ? rescale_2k_fn
                                                        : state->rescale_tail_fn;

    uint32_t       ct           = fastdiv(start_unit, &state->n_groups_div);
    uint32_t       g            = start_unit - ct * n_groups;
    const uint32_t n_col_paired = state->n_col_tiles & ~1u;

    for (uint32_t u = start_unit; u < end_unit; ++u) {
        const bool                        is_full_group  = (g + 1u < n_groups);
        const size_t                      kt_group_start = (size_t) g * HTP_MM_HMX_W8A8_KGROUP_TILES;
        const size_t                      kt_start       = (size_t) ct * n_k_tiles + kt_group_start;
        const uint32_t                    tiles_in_group = is_full_group ? HTP_MM_HMX_W8A8_KGROUP_TILES : tail_tiles;
        const int_rescale_superblock_fn_t fn             = is_full_group ? rescale_2k_fn : tail_fn;
        const uint8_t *                   src_base       = state->src_raw + (kt_start >> raw_shift) * aligned_tile_size;
        uint8_t * dst_wt;
        size_t    dst_tile_stride;
        if (ct < n_col_paired) {
            // Interleave paired column tiles (2*jp, 2*jp+1) into 2048B K-tiles for weight.b:deep.
            dst_wt          = state->dst_wt_base
                            + ((size_t) (ct & ~1u) * n_k_tiles + kt_group_start * 2 + (ct & 1u)) * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
            dst_tile_stride = 2 * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
        } else {
            dst_wt          = state->dst_wt_base + kt_start * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
            dst_tile_stride = HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
        }
        float *                           dst_scales     = state->dst_scales_base + (size_t) u * 32;
        HVX_Vector *                      bias_out       = (HVX_Vector *) (state->dst_bias_base + (size_t) u * (2 * HTP_MM_HMX_BIAS_BANK_SIZE));

        fn(src_base, dst_wt, dst_scales, bias_out, tiles_in_group, dst_tile_stride);

        if (++g == n_groups) {
            g = 0;
            ++ct;
        }
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
        HVX_Vector v_min_u8,
        HVX_Vector v_max_u8) {

#if __HVX_ARCH__ >= 81
    (void) v_min_u8;
    (void) v_max_u8;

    HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(vh0, v_one_f16);
    HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(vh1, v_one_f16);
    HVX_VectorPair vp2 = Q6_Wqf32_vmpy_VhfVhf(vh2, v_one_f16);
    HVX_VectorPair vp3 = Q6_Wqf32_vmpy_VhfVhf(vh3, v_one_f16);

    HVX_Vector hf0 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
        Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp0), v_inv0_qf),
        Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp0), v_inv0_qf)));
    HVX_Vector hf1 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
        Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp1), v_inv1_qf),
        Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp1), v_inv1_qf)));
    HVX_Vector hf2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
        Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp2), v_inv2_qf),
        Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp2), v_inv2_qf)));
    HVX_Vector hf3 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
        Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp3), v_inv3_qf),
        Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp3), v_inv3_qf)));

    hmx_pack_hf_quad_to_u8_cm_pair(tile_row_base, kt, rg, hf0, hf1, hf2, hf3);
#else
    HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh0), v_one_f16);
    HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh1), v_one_f16);
    HVX_VectorPair vp2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh2), v_one_f16);
    HVX_VectorPair vp3 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh3), v_one_f16);

    {
        HVX_Vector y0 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp0), v_inv0_qf));
        HVX_Vector y1 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp1), v_inv1_qf));
        HVX_Vector y2 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp2), v_inv2_qf));
        HVX_Vector y3 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp3), v_inv3_qf));
        HVX_Vector vh01 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vpacke_VwVw(y1, y0), v_min_u8), v_max_u8);
        HVX_Vector vh23 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vpacke_VwVw(y3, y2), v_min_u8), v_max_u8);
        ((HVX_Vector *) (tile_row_base + (size_t) (kt + 0) * HTP_MM_HMX_TILE_SIZE))[rg] = Q6_Vb_vpacke_VhVh(vh23, vh01);
    }
    {
        HVX_Vector y0 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp0), v_inv0_qf));
        HVX_Vector y1 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp1), v_inv1_qf));
        HVX_Vector y2 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp2), v_inv2_qf));
        HVX_Vector y3 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp3), v_inv3_qf));
        HVX_Vector vh01 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vpacke_VwVw(y1, y0), v_min_u8), v_max_u8);
        HVX_Vector vh23 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vpacke_VwVw(y3, y2), v_min_u8), v_max_u8);
        ((HVX_Vector *) (tile_row_base + (size_t) (kt + 1) * HTP_MM_HMX_TILE_SIZE))[rg] = Q6_Vb_vpacke_VhVh(vh23, vh01);
    }
#endif
}

static __attribute__((noinline)) void quantize_act_superblock_quad_f32_to_u8_sm_2k(
        uint8_t * restrict tile_row_base,
        float * restrict d_a_out,
        const float * restrict p0,
        const float * restrict p1,
        const float * restrict p2,
        const float * restrict p3,
        uint32_t rg,
        uint32_t k_start,
        uint32_t valid_rows) {

    const float * p0_base       = p0 + k_start;
    const float * p1_base       = p1 + k_start;
    const float * p2_base       = p2 + k_start;
    const float * p3_base       = p3 + k_start;
    uint8_t     * tile_dst_base = tile_row_base + (size_t) (k_start >> 5) * HTP_MM_HMX_TILE_SIZE;

    assert(hex_is_aligned(p0_base, 128));
    assert(hex_is_aligned(p1_base, 128));
    assert(hex_is_aligned(p2_base, 128));
    assert(hex_is_aligned(p3_base, 128));

    HVX_Vector vmax0 = Q6_V_vzero();
    HVX_Vector vmax1 = Q6_V_vzero();
    HVX_Vector vmax2 = Q6_V_vzero();
    HVX_Vector vmax3 = Q6_V_vzero();

    for (uint32_t kt = 0; kt < HTP_MM_HMX_W8A8_KGROUP_TILES; ++kt) {
        const uint32_t off = kt * 32;
        HVX_Vector vx0 = hvx_vmem(p0_base + off);
        HVX_Vector vx1 = hvx_vmem(p1_base + off);
        HVX_Vector vx2 = hvx_vmem(p2_base + off);
        HVX_Vector vx3 = hvx_vmem(p3_base + off);
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

#if __HVX_ARCH__ >= 81
    for (uint32_t step = 0; step < HTP_MM_HMX_W8A8_KGROUP_TILES / 2; ++step) {
        const uint32_t off = step * 64;
        HVX_Vector hf0 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(p0_base + off + 32), v_inv0),
            Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(p0_base + off + 0),  v_inv0))));
        HVX_Vector hf1 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(p1_base + off + 32), v_inv1),
            Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(p1_base + off + 0),  v_inv1))));
        HVX_Vector hf2 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(p2_base + off + 32), v_inv2),
            Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(p2_base + off + 0),  v_inv2))));
        HVX_Vector hf3 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(p3_base + off + 32), v_inv3),
            Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(p3_base + off + 0),  v_inv3))));

        hmx_pack_hf_quad_to_u8_cm_pair(tile_dst_base, step * 2, rg, hf0, hf1, hf2, hf3);
    }
#else
    const HVX_Vector v_min_u8  = Q6_Vh_vsplat_R(1);
    const HVX_Vector v_max_u8  = Q6_Vh_vsplat_R(255);

    for (uint32_t kt = 0; kt < HTP_MM_HMX_W8A8_KGROUP_TILES; ++kt) {
        const uint32_t off = kt * 32;
        HVX_Vector vx0 = hvx_vmem(p0_base + off);
        HVX_Vector vx1 = hvx_vmem(p1_base + off);
        HVX_Vector vx2 = hvx_vmem(p2_base + off);
        HVX_Vector vx3 = hvx_vmem(p3_base + off);

        HVX_Vector y0 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0));
        HVX_Vector y1 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1));
        HVX_Vector y2 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx2, v_inv2));
        HVX_Vector y3 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx3, v_inv3));

        HVX_Vector vh01 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vpacke_VwVw(y1, y0), v_min_u8), v_max_u8);
        HVX_Vector vh23 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vpacke_VwVw(y3, y2), v_min_u8), v_max_u8);
        HVX_Vector v_u8 = Q6_Vb_vpacke_VhVh(vh23, vh01);

        ((HVX_Vector *) (tile_dst_base + (size_t) kt * HTP_MM_HMX_TILE_SIZE))[rg] = v_u8;
    }
#endif
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

    const HVX_Vector v_min_u8  = Q6_Vh_vsplat_R(1);
    const HVX_Vector v_max_u8  = Q6_Vh_vsplat_R(255);

    for (uint32_t c = k_start; c < c_end_full; c += 32) {
        HVX_Vector vx0 = hvx_vmem(p0 + c);
        HVX_Vector vx1 = hvx_vmem(p1 + c);
        HVX_Vector vx2 = hvx_vmem(p2 + c);
        HVX_Vector vx3 = hvx_vmem(p3 + c);

        HVX_Vector y0 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0));
        HVX_Vector y1 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1));
        HVX_Vector y2 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx2, v_inv2));
        HVX_Vector y3 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx3, v_inv3));

        HVX_Vector vh01 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vpacke_VwVw(y1, y0), v_min_u8), v_max_u8);
        HVX_Vector vh23 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vpacke_VwVw(y3, y2), v_min_u8), v_max_u8);
        HVX_Vector v_u8 = Q6_Vb_vpacke_VhVh(vh23, vh01);

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
        HVX_Vector y0 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0));
        HVX_Vector y1 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1));
        HVX_Vector y2 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx2, v_inv2));
        HVX_Vector y3 = hmx_qf32_to_i32_zp128(Q6_Vqf32_vmpy_VsfVsf(vx3, v_inv3));

        HVX_Vector vh01 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vpacke_VwVw(y1, y0), v_min_u8), v_max_u8);
        HVX_Vector vh23 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vpacke_VwVw(y3, y2), v_min_u8), v_max_u8);
        HVX_Vector v_u8 = Q6_Vb_vpacke_VhVh(vh23, vh01);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_SIZE))[rg] = v_u8;
    }
}

static __attribute__((noinline)) void quantize_act_superblock_quad_f16_to_u8_sm_2k(
        uint8_t * restrict tile_row_base,
        float * restrict d_a_out,
        const __fp16 * restrict p0,
        const __fp16 * restrict p1,
        const __fp16 * restrict p2,
        const __fp16 * restrict p3,
        uint32_t rg,
        uint32_t k_start,
        uint32_t valid_rows) {

    const __fp16 * p0_base       = p0 + k_start;
    const __fp16 * p1_base       = p1 + k_start;
    const __fp16 * p2_base       = p2 + k_start;
    const __fp16 * p3_base       = p3 + k_start;
    uint8_t      * tile_dst_base = tile_row_base + (size_t) (k_start >> 5) * HTP_MM_HMX_TILE_SIZE;

    assert(hex_is_aligned(p0_base, 128));
    assert(hex_is_aligned(p1_base, 128));
    assert(hex_is_aligned(p2_base, 128));
    assert(hex_is_aligned(p3_base, 128));

    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_abs_f16 = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector vmax_hf0 = Q6_V_vzero();
    HVX_Vector vmax_hf1 = Q6_V_vzero();
    HVX_Vector vmax_hf2 = Q6_V_vzero();
    HVX_Vector vmax_hf3 = Q6_V_vzero();

    for (uint32_t step = 0; step < HTP_MM_HMX_W8A8_KGROUP_TILES / 2; ++step) {
        const uint32_t off = step * 64;
        HVX_Vector vh0 = hvx_vmem(p0_base + off);
        HVX_Vector vh1 = hvx_vmem(p1_base + off);
        HVX_Vector vh2 = hvx_vmem(p2_base + off);
        HVX_Vector vh3 = hvx_vmem(p3_base + off);
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

    const HVX_Vector v_min_u8  = Q6_Vh_vsplat_R(1);
    const HVX_Vector v_max_u8  = Q6_Vh_vsplat_R(255);

    for (uint32_t step = 0; step < HTP_MM_HMX_W8A8_KGROUP_TILES / 2; ++step) {
        const uint32_t off = step * 64;
        HVX_Vector vh0 = hvx_vmem(p0_base + off);
        HVX_Vector vh1 = hvx_vmem(p1_base + off);
        HVX_Vector vh2 = hvx_vmem(p2_base + off);
        HVX_Vector vh3 = hvx_vmem(p3_base + off);
        quantize_f16_quad_step_to_u8_sm(tile_dst_base, step * 2, rg, vh0, vh1, vh2, vh3,
                                        v_inv0_qf, v_inv1_qf, v_inv2_qf, v_inv3_qf,
                                        v_one_f16, v_min_u8, v_max_u8);
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

    const HVX_Vector v_min_u8  = Q6_Vh_vsplat_R(1);
    const HVX_Vector v_max_u8  = Q6_Vh_vsplat_R(255);

    for (uint32_t c = k_start; c < c_end_full; c += 64) {
        HVX_Vector vh0 = hvx_vmem(p0 + c);
        HVX_Vector vh1 = hvx_vmem(p1 + c);
        HVX_Vector vh2 = hvx_vmem(p2 + c);
        HVX_Vector vh3 = hvx_vmem(p3 + c);
        quantize_f16_quad_step_to_u8_sm(tile_row_base, c >> 5, rg, vh0, vh1, vh2, vh3, v_inv0_qf, v_inv1_qf, v_inv2_qf, v_inv3_qf, v_one_f16, v_min_u8, v_max_u8);
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
        quantize_f16_quad_step_to_u8_sm(tile_row_base, c >> 5, rg, vh0, vh1, vh2, vh3, v_inv0_qf, v_inv1_qf, v_inv2_qf, v_inv3_qf, v_one_f16, v_min_u8, v_max_u8);
    }
}

#define DEFINE_QUANTIZE_ACT_STEP_ROWS_W8A8(SFX, ACT_T, QUANT_SB_2K_FN, QUANT_SB_TAIL_FN)                       \
static __attribute__((noinline)) void quantize_act_step_rows_##SFX(                                            \
        uint8_t * restrict vtcm_act_dst,                                                                       \
        float * restrict vtcm_act_scales,                                                                      \
        const uint8_t * restrict curr_buf,                                                                     \
        size_t row_bytes,                                                                                      \
        uint32_t r,                                                                                            \
        uint32_t valid_rows,                                                                                   \
        uint32_t mc_pad,                                                                                       \
        uint32_t n_k_tiles,                                                                                    \
        uint32_t n_groups,                                                                                     \
        uint32_t k_valid) {                                                                                    \
    const uint32_t r0 = r >> 6;                                                                                \
    const uint32_t rg = (r & 63u) >> 2;                                                                        \
    uint8_t * tile_row_base = vtcm_act_dst + (size_t) r0 * n_k_tiles * HTP_MM_HMX_TILE_SIZE;                   \
    if (valid_rows == 0) {                                                                                     \
        const HVX_Vector v_zero = Q6_V_vzero();                                                                \
        const HVX_Vector v_zp   = Q6_Vb_vsplat_R(0x80);                                                        \
        for (uint32_t g = 0; g < n_groups; ++g) {                                                              \
            hvx_vec_store_u(vtcm_act_scales + (size_t) g * mc_pad + r, 4 * sizeof(float), v_zero);             \
        }                                                                                                      \
        for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {                                                          \
            ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_SIZE))[rg] = v_zp;                  \
        }                                                                                                      \
    } else {                                                                                                   \
        const uint32_t k_block = n_k_tiles << 5;                                                               \
        const ACT_T * p0 = (const ACT_T *) curr_buf;                                                           \
        const ACT_T * p1 = (valid_rows > 1) ? (const ACT_T *) (curr_buf + 1 * row_bytes) : p0;                 \
        const ACT_T * p2 = (valid_rows > 2) ? (const ACT_T *) (curr_buf + 2 * row_bytes) : p0;                 \
        const ACT_T * p3 = (valid_rows > 3) ? (const ACT_T *) (curr_buf + 3 * row_bytes) : p0;                 \
        for (uint32_t g = 0; g < n_groups; ++g) {                                                              \
            const uint32_t k_start = g * HTP_MM_HMX_W8A8_KGROUP;                                               \
            const uint32_t k_group = (g + 1u < n_groups) ? HTP_MM_HMX_W8A8_KGROUP : (k_block - k_start);       \
            float * d_a_out = vtcm_act_scales + (size_t) g * mc_pad + r;                                       \
            if (k_group == HTP_MM_HMX_W8A8_KGROUP && k_start + HTP_MM_HMX_W8A8_KGROUP <= k_valid) {            \
                QUANT_SB_2K_FN(tile_row_base, d_a_out, p0, p1, p2, p3, rg, k_start, valid_rows);               \
            } else {                                                                                           \
                QUANT_SB_TAIL_FN(tile_row_base, d_a_out, p0, p1, p2, p3, rg, k_start, k_group, k_valid,        \
                                 valid_rows);                                                                  \
            }                                                                                                  \
        }                                                                                                      \
    }                                                                                                          \
}

DEFINE_QUANTIZE_ACT_STEP_ROWS_W8A8(f16_w8a8, __fp16, quantize_act_superblock_quad_f16_to_u8_sm_2k, quantize_act_superblock_quad_f16_to_u8_sm)
DEFINE_QUANTIZE_ACT_STEP_ROWS_W8A8(f32_w8a8, float,  quantize_act_superblock_quad_f32_to_u8_sm_2k, quantize_act_superblock_quad_f32_to_u8_sm)

// Core HMX W8A8 SM dot-product kernel over all superblocks g = 0..n_groups-1.
static void core_dot_chunk_w8a8_sm(
        uint8_t * restrict vtcm_out,
        const uint8_t * restrict vtcm_act,
        const uint8_t * restrict vtcm_weight_tiles,
        const uint8_t * restrict vtcm_weight_bias,
        uint32_t n_row_tiles_64,
        uint32_t n_col_tiles,
        uint32_t n_k_tiles,
        uint32_t n_groups) {

    assert(hex_is_aligned(vtcm_out, HTP_MM_HMX_TILE_SIZE));
    assert(hex_is_aligned(vtcm_act, HTP_MM_HMX_TILE_SIZE));
    assert(hex_is_aligned(vtcm_weight_tiles, HTP_MM_HMX_TILE_SIZE));

    const uint32_t out_cm_rng = hmx_out_cm_range();
    const uint32_t tail_tiles = n_k_tiles - (n_groups - 1u) * HTP_MM_HMX_W8A8_KGROUP_TILES;

    // Clearing convert (retain_acc = 0) clears the active accumulator and implicitly swaps primary/secondary.
#if __HVX_ARCH__ >= 79
    const uint32_t cvt_p_hi = hmx_cvt_ctrl(0, 0, 0, 1);
    const uint32_t cvt_p_lo = hmx_cvt_ctrl(1, 0, 1, 0);
    const uint32_t cvt_s_hi = hmx_cvt_ctrl(2, 0, 0, 1);
    const uint32_t cvt_s_lo = hmx_cvt_ctrl(3, 0, 1, 0);
#else
    const uint32_t cvt_p_hi = hmx_cvt_ctrl(0, 0, 0, 1);
    const uint32_t cvt_p_lo = hmx_cvt_ctrl(0, 0, 1, 0);
    const uint32_t cvt_s_hi = cvt_p_hi;
    const uint32_t cvt_s_lo = cvt_p_lo;
#endif

    for (uint32_t g = 0; g < n_groups; ++g) {
        const uint32_t kt_base        = g * HTP_MM_HMX_W8A8_KGROUP_TILES;
        const bool     is_tail        = (g + 1u == n_groups && tail_tiles < HTP_MM_HMX_W8A8_KGROUP_TILES);
        const uint32_t tiles_in_group = is_tail ? tail_tiles : HTP_MM_HMX_W8A8_KGROUP_TILES;
        const uint32_t a_rng          = hmx_act_sm_range(tiles_in_group);

        for (uint32_t j = 0; j + 1u < n_col_tiles; j += 2) {
            const uint8_t * wt_pair_base = vtcm_weight_tiles
                                         + ((size_t) (j >> 1) * n_k_tiles + kt_base) * (2 * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
            const uint8_t * bias_p_hi = vtcm_weight_bias + (((size_t) (j + 0) * n_groups + g) * 2 + 0) * HTP_MM_HMX_BIAS_BANK_SIZE;
            const uint8_t * bias_p_lo = vtcm_weight_bias + (((size_t) (j + 0) * n_groups + g) * 2 + 1) * HTP_MM_HMX_BIAS_BANK_SIZE;
            const uint8_t * bias_s_hi = vtcm_weight_bias + (((size_t) (j + 1) * n_groups + g) * 2 + 0) * HTP_MM_HMX_BIAS_BANK_SIZE;
            const uint8_t * bias_s_lo = vtcm_weight_bias + (((size_t) (j + 1) * n_groups + g) * 2 + 1) * HTP_MM_HMX_BIAS_BANK_SIZE;

#if __HVX_ARCH__ >= 79
            hmx_load_bias_bank(bias_p_hi, 0);
            hmx_load_bias_bank(bias_p_lo, 1);
            hmx_load_bias_bank(bias_s_hi, 2);
            hmx_load_bias_bank(bias_s_lo, 3);
#endif

            for (uint32_t i = 0; i < n_row_tiles_64; ++i) {
                const uint8_t * act_base = vtcm_act + ((size_t) i * n_k_tiles + kt_base) * HTP_MM_HMX_TILE_SIZE;

                const size_t out_idx  = (((size_t) g * n_row_tiles_64 + i) * n_col_tiles + j) * 2;
                uint8_t *    out_p_hi = vtcm_out + (out_idx + 0) * HTP_MM_HMX_TILE_SIZE;
                uint8_t *    out_p_lo = vtcm_out + (out_idx + 1) * HTP_MM_HMX_TILE_SIZE;
                uint8_t *    out_s_hi = vtcm_out + (out_idx + 2) * HTP_MM_HMX_TILE_SIZE;
                uint8_t *    out_s_lo = vtcm_out + (out_idx + 3) * HTP_MM_HMX_TILE_SIZE;

                asm volatile(HMX_CLRACC_INT() : : : "memory");
                // weight.b:deep processes 2048B/tile; cap each single-packet range to 32 tiles (64KB).
                for (uint32_t kt = 0; kt < tiles_in_group; kt += 32u) {
                    const uint32_t  n_kt  = hex_smin(tiles_in_group - kt, 32u);
                    const uint32_t  a_r   = hmx_act_sm_range(n_kt);
                    const uint32_t  w_r   = hmx_wt_b_range(n_kt * (2 * HTP_MM_HMX_WEIGHT_B_TILE_SIZE));
                    const uint8_t * a_ptr = act_base     + (size_t) kt * HTP_MM_HMX_TILE_SIZE;
                    const uint8_t * w_ptr = wt_pair_base + (size_t) kt * (2 * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
                    asm volatile(HMX_LOAD_MPY_DEEP_U8_S8_CM_64C("%0", "%1", "%2", "%3")
                                 : : "r"(a_ptr), "r"(a_r), "r"(w_ptr), "r"(w_r) : "memory");
                }

#if __HVX_ARCH__ >= 79
                asm volatile(
                    HMX_CVT_UB_ACC("%0")
                    HMX_STORE_CVT_CM("%4", "%8")
                    HMX_CVT_UB_ACC("%1")
                    HMX_STORE_CVT_CM("%5", "%8")
                    HMX_CVT_UB_ACC("%2")
                    HMX_STORE_CVT_CM("%6", "%8")
                    HMX_CVT_UB_ACC("%3")
                    HMX_STORE_CVT_CM("%7", "%8")
                    :
                    : "r"(cvt_p_hi), "r"(cvt_p_lo), "r"(cvt_s_hi), "r"(cvt_s_lo),
                      "r"(out_p_hi), "r"(out_p_lo), "r"(out_s_hi), "r"(out_s_lo),
                      "r"(out_cm_rng)
                    : "memory");
#else
                hmx_load_bias_bank(bias_p_hi, 0);
                asm volatile(HMX_CVT_UB_ACC("%0") HMX_STORE_CVT_CM("%1", "%2")
                             : : "r"(cvt_p_hi), "r"(out_p_hi), "r"(out_cm_rng) : "memory");
                hmx_load_bias_bank(bias_p_lo, 0);
                asm volatile(HMX_CVT_UB_ACC("%0") HMX_STORE_CVT_CM("%1", "%2")
                             : : "r"(cvt_p_lo), "r"(out_p_lo), "r"(out_cm_rng) : "memory");
                hmx_load_bias_bank(bias_s_hi, 0);
                asm volatile(HMX_CVT_UB_ACC("%0") HMX_STORE_CVT_CM("%1", "%2")
                             : : "r"(cvt_s_hi), "r"(out_s_hi), "r"(out_cm_rng) : "memory");
                hmx_load_bias_bank(bias_s_lo, 0);
                asm volatile(HMX_CVT_UB_ACC("%0") HMX_STORE_CVT_CM("%1", "%2")
                             : : "r"(cvt_s_lo), "r"(out_s_lo), "r"(out_cm_rng) : "memory");
#endif
            }
        }

        if (n_col_tiles & 1u) {
            const uint32_t  j       = n_col_tiles - 1u;
            const uint32_t  w_rng   = hmx_wt_b_range(tiles_in_group * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
            const uint8_t * wt_base = vtcm_weight_tiles + ((size_t) j * n_k_tiles + kt_base) * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
            const uint8_t * bias_hi = vtcm_weight_bias  + (((size_t) j * n_groups + g) * 2 + 0) * HTP_MM_HMX_BIAS_BANK_SIZE;
            const uint8_t * bias_lo = vtcm_weight_bias  + (((size_t) j * n_groups + g) * 2 + 1) * HTP_MM_HMX_BIAS_BANK_SIZE;

#if __HVX_ARCH__ >= 79
            hmx_load_bias_bank(bias_hi, 0);
            hmx_load_bias_bank(bias_lo, 1);
#endif

            for (uint32_t i = 0; i < n_row_tiles_64; ++i) {
                const uint8_t * act_base = vtcm_act + ((size_t) i * n_k_tiles + kt_base) * HTP_MM_HMX_TILE_SIZE;

                const size_t out_idx = (((size_t) g * n_row_tiles_64 + i) * n_col_tiles + j) * 2;
                uint8_t *    out_hi  = vtcm_out + (out_idx + 0) * HTP_MM_HMX_TILE_SIZE;
                uint8_t *    out_lo  = vtcm_out + (out_idx + 1) * HTP_MM_HMX_TILE_SIZE;

                asm volatile(HMX_CLRACC_INT() : : : "memory");
                asm volatile(HMX_LOAD_MPY_DEEP_U8_S8_CM("%0", "%1", "%2", "%3")
                             : : "r"(act_base), "r"(a_rng), "r"(wt_base), "r"(w_rng) : "memory");
#if __HVX_ARCH__ < 79
                hmx_load_bias_bank(bias_hi, 0);
#endif
                asm volatile(HMX_CVT_UB_ACC("%0") : : "r"(cvt_p_hi) : "memory");
                asm volatile(HMX_STORE_CVT_CM("%0", "%1") : : "r"(out_hi), "r"(out_cm_rng) : "memory");
#if __HVX_ARCH__ < 79
                hmx_load_bias_bank(bias_lo, 0);
#endif
                asm volatile(HMX_CVT_UB_ACC("%0") : : "r"(cvt_p_lo) : "memory");
                asm volatile(HMX_STORE_CVT_CM("%0", "%1") : : "r"(out_lo), "r"(out_cm_rng) : "memory");
            }
        }
    }
}

// Stage activation scales from VTCM to 128B-aligned L1D stack memory so scalar Rsf loads avoid VTCM stalls and SROA vextract spills.
static inline uint32_t (*stage_act_scales_to_l1d(
        uint32_t (* restrict dst)[32],
        const float * restrict vtcm_scales,
        size_t group_stride,
        uint32_t n_groups))[32] {
    asm("" : "+r"(dst));
    for (uint32_t g = 0; g < n_groups; ++g) {
        *(HVX_Vector *) dst[g] = hvx_vmem(vtcm_scales + g * group_stride);
    }
    return dst;
}

#define HMX_INT_FIN_ROW_NONE(acc_qf, row_idx, bias_base, bias_rb, c) \
    Q6_Vsf_equals_Vqf32(acc_qf)

#define HMX_INT_FIN_ROW_F16(acc_qf, row_idx, bias_base, bias_rb, c) \
    Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf((acc_qf), hmx_load_bias_f32_32_f16((bias_base) + (row_idx) * (bias_rb), (c))))

#define HMX_INT_FIN_ROW_F32(acc_qf, row_idx, bias_base, bias_rb, c) \
    Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf((acc_qf), hmx_load_bias_f32_32_f32((bias_base) + (row_idx) * (bias_rb), (c))))

#define HMX_INT_BIAS_ROW_none(bias, off) NULL
#define HMX_INT_BIAS_ROW_f16(bias, off)  ((const uint8_t *) (bias) + (off))
#define HMX_INT_BIAS_ROW_f32(bias, off)  ((const uint8_t *) (bias) + (off))

static inline HVX_Vector hmx_pack_f32_to_f16_bias_f16(
        HVX_Vector v_lo, HVX_Vector v_hi, const uint8_t * bias_row, size_t c) {
    const HVX_VectorPair vp_b = hmx_load_bias_f32_64_f16(bias_row, c);
    return hmx_pack_f32_to_f16_clamped(
            hvx_vec_add_f32_f32(v_lo, Q6_V_lo_W(vp_b)),
            hvx_vec_add_f32_f32(v_hi, Q6_V_hi_W(vp_b)));
}

static inline HVX_Vector hmx_pack_f32_to_f16_bias_f32(
        HVX_Vector v_lo, HVX_Vector v_hi, const uint8_t * bias_row, size_t c) {
    return hmx_pack_f32_to_f16_clamped(
            hvx_vec_add_f32_f32(v_lo, hmx_load_bias_f32_32_f32(bias_row, c)),
            hvx_vec_add_f32_f32(v_hi, hmx_load_bias_f32_32_f32(bias_row, c + 32)));
}

static inline HVX_Vector hmx_pack_f32_to_f16_tail_bias_f16(
        HVX_Vector v_lo, HVX_Vector v_hi, const uint8_t * bias_row, size_t c, size_t valid_c) {
    const HVX_VectorPair vp_b = hmx_load_bias_f32_64_f16(bias_row, c);
    const HVX_Vector     s0   = hvx_vec_add_f32_f32(v_lo, Q6_V_lo_W(vp_b));
    const HVX_Vector     s1   = (valid_c > 32) ? hvx_vec_add_f32_f32(v_hi, Q6_V_hi_W(vp_b)) : Q6_V_vzero();
    return hmx_pack_f32_to_f16_clamped(s0, s1);
}

static inline HVX_Vector hmx_pack_f32_to_f16_tail_bias_f32(
        HVX_Vector v_lo, HVX_Vector v_hi, const uint8_t * bias_row, size_t c, size_t valid_c) {
    const HVX_Vector s0 = hvx_vec_add_f32_f32(v_lo, hmx_load_bias_f32_32_f32(bias_row, c));
    const HVX_Vector s1 = (valid_c > 32) ? hvx_vec_add_f32_f32(v_hi, hmx_load_bias_f32_32_f32(bias_row, c + 32)) : Q6_V_vzero();
    return hmx_pack_f32_to_f16_clamped(s0, s1);
}

#define HMX_INT_PACK_FP16_ROW_none(v_lo, v_hi, bias_base, bias_rb, row_idx, c) \
    hmx_pack_f32_to_f16_clamped((v_lo), (v_hi))
#define HMX_INT_PACK_FP16_ROW_f16(v_lo, v_hi, bias_base, bias_rb, row_idx, c) \
    hmx_pack_f32_to_f16_bias_f16((v_lo), (v_hi), (bias_base) + (row_idx) * (bias_rb), (c))
#define HMX_INT_PACK_FP16_ROW_f32(v_lo, v_hi, bias_base, bias_rb, row_idx, c) \
    hmx_pack_f32_to_f16_bias_f32((v_lo), (v_hi), (bias_base) + (row_idx) * (bias_rb), (c))

#define HMX_INT_PACK_FP16_TAIL_none(v_lo, v_hi, bias_base, bias_rb, row_idx, c, valid_c) \
    hmx_pack_f32_to_f16_clamped((v_lo), (v_hi))
#define HMX_INT_PACK_FP16_TAIL_f16(v_lo, v_hi, bias_base, bias_rb, row_idx, c, valid_c) \
    hmx_pack_f32_to_f16_tail_bias_f16((v_lo), (v_hi), (bias_base) + (row_idx) * (bias_rb), (c), (valid_c))
#define HMX_INT_PACK_FP16_TAIL_f32(v_lo, v_hi, bias_base, bias_rb, row_idx, c, valid_c) \
    hmx_pack_f32_to_f16_tail_bias_f32((v_lo), (v_hi), (bias_base) + (row_idx) * (bias_rb), (c), (valid_c))

#define DEFINE_W8A8_EVAL_TILE_QUAD(SFX, FIN_ROW)                                                                                            \
static inline void w8a8_sm_eval_tile_quad_1g_##SFX(                                                                                         \
        const uint8_t * restrict tile_base,                                                                                                 \
        const float * restrict dw_base,                                                                                                     \
        size_t c,                                                                                                                           \
        const uint32_t * restrict da,                                                                                                       \
        const uint8_t * restrict bias_row0,                                                                                                 \
        size_t bias_row_bytes,                                                                                                              \
        uint32_t n_rows,                                                                                                                    \
        size_t r,                                                                                                                           \
        bool full_quad,                                                                                                                     \
        HVX_Vector * v_out0,                                                                                                                \
        HVX_Vector * v_out1,                                                                                                                \
        HVX_Vector * v_out2,                                                                                                                \
        HVX_Vector * v_out3) {                                                                                                              \
    (void) bias_row0;                                                                                                                       \
    (void) bias_row_bytes;                                                                                                                  \
    const HVX_Vector sign_80  = Q6_Vb_vsplat_R(0x80);                                                                                       \
    const size_t     c0       = c >> 5;                                                                                                     \
    const uint8_t *  tp       = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;                                                                  \
    const HVX_Vector v_hi_s8  = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);                                                            \
    const HVX_Vector v_lo_u8  = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);                                                          \
    const HVX_Vector v_dw_qf  = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));                                                   \
    HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);                                                                       \
    HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));                                                                         \
    HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));                                                                         \
    HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));                                                                          \
    HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da[0]), v_dw_qf);                                       \
    *v_out0 = FIN_ROW(v_acc0_qf, 0, bias_row0, bias_row_bytes, c);                                                                          \
    if (full_quad || r + 1 < n_rows) {                                                                                                      \
        HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));                                                                      \
        HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da[1]), v_dw_qf);                                   \
        *v_out1 = FIN_ROW(v_acc1_qf, 1, bias_row0, bias_row_bytes, c);                                                                      \
    }                                                                                                                                       \
    if (full_quad || r + 2 < n_rows) {                                                                                                      \
        HVX_Vector v_f32_2   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));                                                                      \
        HVX_Vector v_acc2_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da[2]), v_dw_qf);                                   \
        *v_out2 = FIN_ROW(v_acc2_qf, 2, bias_row0, bias_row_bytes, c);                                                                      \
    }                                                                                                                                       \
    if (full_quad || r + 3 < n_rows) {                                                                                                      \
        HVX_Vector v_f32_3   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));                                                                      \
        HVX_Vector v_acc3_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da[3]), v_dw_qf);                                   \
        *v_out3 = FIN_ROW(v_acc3_qf, 3, bias_row0, bias_row_bytes, c);                                                                      \
    }                                                                                                                                       \
}                                                                                                                                           \
static inline void w8a8_sm_eval_tile_quad_2g_##SFX(                                                                                         \
        const uint8_t * restrict tile_base0,                                                                                                \
        const uint8_t * restrict tile_base1,                                                                                                \
        const float * restrict dw_base,                                                                                                     \
        size_t c,                                                                                                                           \
        const uint32_t * restrict da0,                                                                                                      \
        const uint32_t * restrict da1,                                                                                                      \
        const uint8_t * restrict bias_row0,                                                                                                 \
        size_t bias_row_bytes,                                                                                                              \
        uint32_t n_rows,                                                                                                                    \
        size_t r,                                                                                                                           \
        bool full_quad,                                                                                                                     \
        HVX_Vector * v_out0,                                                                                                                \
        HVX_Vector * v_out1,                                                                                                                \
        HVX_Vector * v_out2,                                                                                                                \
        HVX_Vector * v_out3) {                                                                                                              \
    (void) bias_row0;                                                                                                                       \
    (void) bias_row_bytes;                                                                                                                  \
    const HVX_Vector sign_80   = Q6_Vb_vsplat_R(0x80);                                                                                      \
    const size_t     c0        = c >> 5;                                                                                                    \
    const size_t     tile_off  = c0 * 2 * HTP_MM_HMX_TILE_SIZE;                                                                             \
    const uint8_t *  tp0       = tile_base0 + tile_off;                                                                                     \
    const uint8_t *  tp1       = tile_base1 + tile_off;                                                                                     \
    const float *    dwp       = dw_base + c0 * 64;                                                                                         \
    const HVX_Vector v_hi0_s8  = Q6_V_vxor_VV(*(const HVX_Vector *) tp0, sign_80);                                                          \
    const HVX_Vector v_lo0_u8  = *(const HVX_Vector *) (tp0 + HTP_MM_HMX_TILE_SIZE);                                                        \
    const HVX_Vector v_dw0_qf  = hvx_vec_f32_to_qf32(*(const HVX_Vector *) dwp);                                                            \
    HVX_VectorPair   vp0_i16   = Q6_W_vshuff_VVR(v_hi0_s8, v_lo0_u8, -1);                                                                   \
    HVX_VectorPair   vp0_01    = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp0_i16));                                                                      \
    HVX_VectorPair   vp0_23    = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp0_i16));                                                                      \
    const HVX_Vector v_hi1_s8  = Q6_V_vxor_VV(*(const HVX_Vector *) tp1, sign_80);                                                          \
    const HVX_Vector v_lo1_u8  = *(const HVX_Vector *) (tp1 + HTP_MM_HMX_TILE_SIZE);                                                        \
    const HVX_Vector v_dw1_qf  = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dwp + 32));                                                     \
    HVX_VectorPair   vp1_i16   = Q6_W_vshuff_VVR(v_hi1_s8, v_lo1_u8, -1);                                                                   \
    HVX_VectorPair   vp1_01    = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp1_i16));                                                                      \
    HVX_VectorPair   vp1_23    = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp1_i16));                                                                      \
    HVX_Vector v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(                                                                                        \
            Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_lo_W(vp0_01)), da0[0]), v_dw0_qf),                        \
            Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_lo_W(vp1_01)), da1[0]), v_dw1_qf));                       \
    *v_out0 = FIN_ROW(v_acc0_qf, 0, bias_row0, bias_row_bytes, c);                                                                          \
    if (full_quad || r + 1 < n_rows) {                                                                                                      \
        HVX_Vector v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(                                                                                    \
                Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_hi_W(vp0_01)), da0[1]), v_dw0_qf),                    \
                Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_hi_W(vp1_01)), da1[1]), v_dw1_qf));                   \
        *v_out1 = FIN_ROW(v_acc1_qf, 1, bias_row0, bias_row_bytes, c);                                                                      \
    }                                                                                                                                       \
    if (full_quad || r + 2 < n_rows) {                                                                                                      \
        HVX_Vector v_acc2_qf = Q6_Vqf32_vadd_Vqf32Vqf32(                                                                                    \
                Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_lo_W(vp0_23)), da0[2]), v_dw0_qf),                    \
                Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_lo_W(vp1_23)), da1[2]), v_dw1_qf));                   \
        *v_out2 = FIN_ROW(v_acc2_qf, 2, bias_row0, bias_row_bytes, c);                                                                      \
    }                                                                                                                                       \
    if (full_quad || r + 3 < n_rows) {                                                                                                      \
        HVX_Vector v_acc3_qf = Q6_Vqf32_vadd_Vqf32Vqf32(                                                                                    \
                Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_hi_W(vp0_23)), da0[3]), v_dw0_qf),                    \
                Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_hi_W(vp1_23)), da1[3]), v_dw1_qf));                   \
        *v_out3 = FIN_ROW(v_acc3_qf, 3, bias_row0, bias_row_bytes, c);                                                                      \
    }                                                                                                                                       \
}                                                                                                                                           \
static inline void w8a8_sm_eval_tile_quad_ng_##SFX(                                                                                         \
        const uint8_t * restrict tile_base,                                                                                                 \
        const float * restrict dw_base,                                                                                                     \
        size_t group_stride,                                                                                                                \
        uint32_t n_groups,                                                                                                                  \
        const uint32_t * restrict da_quad_base,                                                                                             \
        HVX_Vector sign_80,                                                                                                                 \
        const uint8_t * restrict bias_row_base,                                                                                             \
        size_t c,                                                                                                                           \
        size_t bias_row_bytes,                                                                                                              \
        uint32_t n_rows,                                                                                                                    \
        size_t r,                                                                                                                           \
        bool full_quad,                                                                                                                     \
        HVX_Vector * v_out0,                                                                                                                \
        HVX_Vector * v_out1,                                                                                                                \
        HVX_Vector * v_out2,                                                                                                                \
        HVX_Vector * v_out3) {                                                                                                              \
    (void) bias_row_base;                                                                                                                   \
    (void) bias_row_bytes;                                                                                                                  \
    const size_t     c0 = c >> 5;                                                                                                           \
    const uint8_t *  tp = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;                                                                        \
    const float *    dw = dw_base + c0 * (size_t) n_groups * 32;                                                                            \
    const uint32_t * da = da_quad_base;                                                                                                     \
    HVX_Vector v_acc0_qf = Q6_V_vzero();                                                                                                    \
    HVX_Vector v_acc1_qf = Q6_V_vzero();                                                                                                    \
    HVX_Vector v_acc2_qf = Q6_V_vzero();                                                                                                    \
    HVX_Vector v_acc3_qf = Q6_V_vzero();                                                                                                    \
    for (uint32_t g = 0; g < n_groups; ++g) {                                                                                               \
        const HVX_Vector v_hi_s8  = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);                                                        \
        const HVX_Vector v_lo_u8  = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);                                                      \
        const HVX_Vector v_dw_qf  = hvx_vec_f32_to_qf32(*(const HVX_Vector *) dw);                                                          \
        HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);                                                                   \
        HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));                                                                     \
        HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));                                                                     \
        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));                                                                        \
        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));                                                                        \
        HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));                                                                        \
        HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));                                                                        \
        v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da[0]), v_dw_qf));         \
        v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da[1]), v_dw_qf));         \
        v_acc2_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc2_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da[2]), v_dw_qf));         \
        v_acc3_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc3_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da[3]), v_dw_qf));         \
        tp += group_stride;                                                                                                                 \
        dw += 32;                                                                                                                           \
        da += 32;                                                                                                                           \
    }                                                                                                                                       \
    *v_out0 = FIN_ROW(v_acc0_qf, 0, bias_row_base, bias_row_bytes, c);                                                                      \
    if (full_quad || r + 1 < n_rows) {                                                                                                      \
        *v_out1 = FIN_ROW(v_acc1_qf, 1, bias_row_base, bias_row_bytes, c);                                                                  \
    }                                                                                                                                       \
    if (full_quad || r + 2 < n_rows) {                                                                                                      \
        *v_out2 = FIN_ROW(v_acc2_qf, 2, bias_row_base, bias_row_bytes, c);                                                                  \
    }                                                                                                                                       \
    if (full_quad || r + 3 < n_rows) {                                                                                                      \
        *v_out3 = FIN_ROW(v_acc3_qf, 3, bias_row_base, bias_row_bytes, c);                                                                  \
    }                                                                                                                                       \
}                                                                                                                                           \
static inline void w8a8_sm_eval_tile_quad_chunked_##SFX(                                                                                    \
        const uint8_t * restrict tile_base,                                                                                                 \
        const float * restrict dw_base,                                                                                                     \
        const float * restrict act_scales_blk_base,                                                                                         \
        size_t group_stride,                                                                                                                \
        uint32_t mc_pad,                                                                                                                    \
        uint32_t n_groups,                                                                                                                  \
        size_t lane,                                                                                                                        \
        uint32_t (*da_ptr)[32],                                                                                                             \
        HVX_Vector sign_80,                                                                                                                 \
        const uint8_t * restrict bias_row_base,                                                                                             \
        size_t c,                                                                                                                           \
        size_t bias_row_bytes,                                                                                                              \
        uint32_t n_rows,                                                                                                                    \
        size_t r,                                                                                                                           \
        bool full_quad,                                                                                                                     \
        HVX_Vector * v_out0,                                                                                                                \
        HVX_Vector * v_out1,                                                                                                                \
        HVX_Vector * v_out2,                                                                                                                \
        HVX_Vector * v_out3) {                                                                                                              \
    (void) bias_row_base;                                                                                                                   \
    (void) bias_row_bytes;                                                                                                                  \
    const size_t    c0 = c >> 5;                                                                                                            \
    const uint8_t * tp = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;                                                                         \
    const float *   dw = dw_base + c0 * (size_t) n_groups * 32;                                                                             \
    HVX_Vector v_acc0_qf = Q6_V_vzero();                                                                                                    \
    HVX_Vector v_acc1_qf = Q6_V_vzero();                                                                                                    \
    HVX_Vector v_acc2_qf = Q6_V_vzero();                                                                                                    \
    HVX_Vector v_acc3_qf = Q6_V_vzero();                                                                                                    \
    for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {                                                                           \
        const uint32_t   g_cnt = hex_smin(n_groups - g_base, 16u);                                                                          \
        da_ptr                 = stage_act_scales_to_l1d(da_ptr, act_scales_blk_base + (size_t) g_base * mc_pad, mc_pad, g_cnt);            \
        const uint32_t * da    = &da_ptr[0][lane];                                                                                          \
        for (uint32_t gi = 0; gi < g_cnt; ++gi) {                                                                                           \
            const HVX_Vector v_hi_s8  = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);                                                    \
            const HVX_Vector v_lo_u8  = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);                                                  \
            const HVX_Vector v_dw_qf  = hvx_vec_f32_to_qf32(*(const HVX_Vector *) dw);                                                      \
            HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);                                                               \
            HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));                                                                 \
            HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));                                                                 \
            HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));                                                                    \
            HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));                                                                    \
            HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));                                                                    \
            HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));                                                                    \
            v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da[0]), v_dw_qf));     \
            v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da[1]), v_dw_qf));     \
            v_acc2_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc2_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da[2]), v_dw_qf));     \
            v_acc3_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc3_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da[3]), v_dw_qf));     \
            tp += group_stride;                                                                                                             \
            dw += 32;                                                                                                                       \
            da += 32;                                                                                                                       \
        }                                                                                                                                   \
    }                                                                                                                                       \
    *v_out0 = FIN_ROW(v_acc0_qf, 0, bias_row_base, bias_row_bytes, c);                                                                      \
    if (full_quad || r + 1 < n_rows) {                                                                                                      \
        *v_out1 = FIN_ROW(v_acc1_qf, 1, bias_row_base, bias_row_bytes, c);                                                                  \
    }                                                                                                                                       \
    if (full_quad || r + 2 < n_rows) {                                                                                                      \
        *v_out2 = FIN_ROW(v_acc2_qf, 2, bias_row_base, bias_row_bytes, c);                                                                  \
    }                                                                                                                                       \
    if (full_quad || r + 3 < n_rows) {                                                                                                      \
        *v_out3 = FIN_ROW(v_acc3_qf, 3, bias_row_base, bias_row_bytes, c);                                                                  \
    }                                                                                                                                       \
}

DEFINE_W8A8_EVAL_TILE_QUAD(none, HMX_INT_FIN_ROW_NONE)
DEFINE_W8A8_EVAL_TILE_QUAD(f16,  HMX_INT_FIN_ROW_F16)
DEFINE_W8A8_EVAL_TILE_QUAD(f32,  HMX_INT_FIN_ROW_F32)

#define HMX_INT_STORE_VEC_nt(ptr, val) hvx_vmem_store_nt((ptr), (val))
#define HMX_INT_STORE_VEC_u(ptr, val)  (hvx_vmemu((ptr)) = (val))

#define W8A8_COL_FULL_QUAD_1G_FP32(SFX, STORE_MODE)                                                      \
    for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                   \
        HVX_Vector v0, v1, v2, v3;                                                                       \
        w8a8_sm_eval_tile_quad_1g_##SFX(tile_base, dw_base, c, da,                                       \
                                        bias_row0, bias_row_bytes, n_rows, r, true, &v0, &v1, &v2, &v3); \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 0 * dst_stride, v0);                               \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 1 * dst_stride, v1);                               \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 2 * dst_stride, v2);                               \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 3 * dst_stride, v3);                               \
    }

#define W8A8_COL_STEP_1G_FP32(SFX, STORE_MODE)                                                                \
    if (r + 3 < n_rows) {                                                                                     \
        W8A8_COL_FULL_QUAD_1G_FP32(SFX, STORE_MODE)                                                           \
    } else {                                                                                                  \
        for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                    \
            HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                           \
            w8a8_sm_eval_tile_quad_1g_##SFX(tile_base, dw_base, c, da,                                        \
                                            bias_row0, bias_row_bytes, n_rows, r, false, &v0, &v1, &v2, &v3); \
            hvx_vmemu(out_row0 + c + 0 * dst_stride) = v0;                                                    \
            if (r + 1 < n_rows) hvx_vmemu(out_row0 + c + 1 * dst_stride) = v1;                                \
            if (r + 2 < n_rows) hvx_vmemu(out_row0 + c + 2 * dst_stride) = v2;                                \
            if (r + 3 < n_rows) hvx_vmemu(out_row0 + c + 3 * dst_stride) = v3;                                \
        }                                                                                                     \
    }                                                                                                         \
    if (limit_c_aligned < limit_c) {                                                                          \
        const size_t c           = limit_c_aligned;                                                           \
        const size_t valid_bytes = (limit_c - c) * sizeof(float);                                             \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                               \
        w8a8_sm_eval_tile_quad_1g_##SFX(tile_base, dw_base, c, da,                                            \
                                        bias_row0, bias_row_bytes, n_rows, r, false, &v0, &v1, &v2, &v3);     \
        hvx_vec_store_u(out_row0 + c + 0 * dst_stride, valid_bytes, v0);                                      \
        if (r + 1 < n_rows) hvx_vec_store_u(out_row0 + c + 1 * dst_stride, valid_bytes, v1);                  \
        if (r + 2 < n_rows) hvx_vec_store_u(out_row0 + c + 2 * dst_stride, valid_bytes, v2);                  \
        if (r + 3 < n_rows) hvx_vec_store_u(out_row0 + c + 3 * dst_stride, valid_bytes, v3);                  \
    }

#define W8A8_COL_FULL_QUAD_2G_FP32(SFX, STORE_MODE)                                                        \
    for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                     \
        HVX_Vector v0, v1, v2, v3;                                                                         \
        w8a8_sm_eval_tile_quad_2g_##SFX(tile_base0, tile_base1, dw_base, c, da0, da1,                      \
                                        bias_row0, bias_row_bytes, n_rows, r, true, &v0, &v1, &v2, &v3);   \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 0 * dst_stride, v0);                                 \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 1 * dst_stride, v1);                                 \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 2 * dst_stride, v2);                                 \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 3 * dst_stride, v3);                                 \
    }

#define W8A8_COL_STEP_2G_FP32(SFX, STORE_MODE)                                                                  \
    if (r + 3 < n_rows) {                                                                                       \
        W8A8_COL_FULL_QUAD_2G_FP32(SFX, STORE_MODE)                                                             \
    } else {                                                                                                    \
        for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                      \
            HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                             \
            w8a8_sm_eval_tile_quad_2g_##SFX(tile_base0, tile_base1, dw_base, c, da0, da1,                       \
                                            bias_row0, bias_row_bytes, n_rows, r, false, &v0, &v1, &v2, &v3);   \
            hvx_vmemu(out_row0 + c + 0 * dst_stride) = v0;                                                      \
            if (r + 1 < n_rows) hvx_vmemu(out_row0 + c + 1 * dst_stride) = v1;                                  \
            if (r + 2 < n_rows) hvx_vmemu(out_row0 + c + 2 * dst_stride) = v2;                                  \
            if (r + 3 < n_rows) hvx_vmemu(out_row0 + c + 3 * dst_stride) = v3;                                  \
        }                                                                                                       \
    }                                                                                                           \
    if (limit_c_aligned < limit_c) {                                                                            \
        const size_t c           = limit_c_aligned;                                                             \
        const size_t valid_bytes = (limit_c - c) * sizeof(float);                                               \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                                 \
        w8a8_sm_eval_tile_quad_2g_##SFX(tile_base0, tile_base1, dw_base, c, da0, da1,                           \
                                        bias_row0, bias_row_bytes, n_rows, r, false, &v0, &v1, &v2, &v3);       \
        hvx_vec_store_u(out_row0 + c + 0 * dst_stride, valid_bytes, v0);                                        \
        if (r + 1 < n_rows) hvx_vec_store_u(out_row0 + c + 1 * dst_stride, valid_bytes, v1);                    \
        if (r + 2 < n_rows) hvx_vec_store_u(out_row0 + c + 2 * dst_stride, valid_bytes, v2);                    \
        if (r + 3 < n_rows) hvx_vec_store_u(out_row0 + c + 3 * dst_stride, valid_bytes, v3);                    \
    }

#define W8A8_COL_FULL_QUAD_NG_FP32(SFX, STORE_MODE)                                                             \
    for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                          \
        HVX_Vector v0, v1, v2, v3;                                                                              \
        w8a8_sm_eval_tile_quad_ng_##SFX(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,      \
                                        bias_row_base, c, bias_row_bytes, n_rows, r, true, &v0, &v1, &v2, &v3); \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 0 * dst_stride, v0);                                  \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 1 * dst_stride, v1);                                  \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 2 * dst_stride, v2);                                  \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 3 * dst_stride, v3);                                  \
    }

#define W8A8_COL_STEP_NG_FP32(SFX, STORE_MODE)                                                                        \
    if (r + 3 < n_rows) {                                                                                             \
        W8A8_COL_FULL_QUAD_NG_FP32(SFX, STORE_MODE)                                                                   \
    } else {                                                                                                          \
        for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                            \
            HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                                   \
            w8a8_sm_eval_tile_quad_ng_##SFX(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,        \
                                            bias_row_base, c, bias_row_bytes, n_rows, r, false, &v0, &v1, &v2, &v3);  \
            hvx_vmemu(out_row_base + c + 0 * dst_stride) = v0;                                                        \
            if (r + 1 < n_rows) hvx_vmemu(out_row_base + c + 1 * dst_stride) = v1;                                    \
            if (r + 2 < n_rows) hvx_vmemu(out_row_base + c + 2 * dst_stride) = v2;                                    \
            if (r + 3 < n_rows) hvx_vmemu(out_row_base + c + 3 * dst_stride) = v3;                                    \
        }                                                                                                             \
    }                                                                                                                 \
    if (limit_c_aligned < limit_c) {                                                                                  \
        const size_t c           = limit_c_aligned;                                                                   \
        const size_t valid_bytes = (limit_c - c) * sizeof(float);                                                     \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                                       \
        w8a8_sm_eval_tile_quad_ng_##SFX(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,            \
                                        bias_row_base, c, bias_row_bytes, n_rows, r, false, &v0, &v1, &v2, &v3);      \
        hvx_vec_store_u(out_row_base + c + 0 * dst_stride, valid_bytes, v0);                                          \
        if (r + 1 < n_rows) hvx_vec_store_u(out_row_base + c + 1 * dst_stride, valid_bytes, v1);                      \
        if (r + 2 < n_rows) hvx_vec_store_u(out_row_base + c + 2 * dst_stride, valid_bytes, v2);                      \
        if (r + 3 < n_rows) hvx_vec_store_u(out_row_base + c + 3 * dst_stride, valid_bytes, v3);                      \
    }

#define W8A8_COL_FULL_QUAD_CHUNKED_FP32(SFX, STORE_MODE)                                                             \
    for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                               \
        HVX_Vector v0, v1, v2, v3;                                                                                   \
        w8a8_sm_eval_tile_quad_chunked_##SFX(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,          \
                                             n_groups, lane, da_ptr, sign_80,                                        \
                                             bias_row_base, c, bias_row_bytes, n_rows, r, true, &v0, &v1, &v2, &v3); \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 0 * dst_stride, v0);                                       \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 1 * dst_stride, v1);                                       \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 2 * dst_stride, v2);                                       \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 3 * dst_stride, v3);                                       \
    }

#define W8A8_COL_STEP_CHUNKED_FP32(SFX, STORE_MODE)                                                                       \
    if (r + 3 < n_rows) {                                                                                                 \
        W8A8_COL_FULL_QUAD_CHUNKED_FP32(SFX, STORE_MODE)                                                                  \
    } else {                                                                                                              \
        for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                                \
            HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                                       \
            w8a8_sm_eval_tile_quad_chunked_##SFX(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,           \
                                                 n_groups, lane, da_ptr, sign_80,                                         \
                                                 bias_row_base, c, bias_row_bytes, n_rows, r, false, &v0, &v1, &v2, &v3); \
            hvx_vmemu(out_row_base + c + 0 * dst_stride) = v0;                                                            \
            if (r + 1 < n_rows) hvx_vmemu(out_row_base + c + 1 * dst_stride) = v1;                                        \
            if (r + 2 < n_rows) hvx_vmemu(out_row_base + c + 2 * dst_stride) = v2;                                        \
            if (r + 3 < n_rows) hvx_vmemu(out_row_base + c + 3 * dst_stride) = v3;                                        \
        }                                                                                                                 \
    }                                                                                                                     \
    if (limit_c_aligned < limit_c) {                                                                                      \
        const size_t c           = limit_c_aligned;                                                                       \
        const size_t valid_bytes = (limit_c - c) * sizeof(float);                                                         \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                                           \
        w8a8_sm_eval_tile_quad_chunked_##SFX(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,               \
                                             n_groups, lane, da_ptr, sign_80,                                             \
                                             bias_row_base, c, bias_row_bytes, n_rows, r, false, &v0, &v1, &v2, &v3);     \
        hvx_vec_store_u(out_row_base + c + 0 * dst_stride, valid_bytes, v0);                                              \
        if (r + 1 < n_rows) hvx_vec_store_u(out_row_base + c + 1 * dst_stride, valid_bytes, v1);                          \
        if (r + 2 < n_rows) hvx_vec_store_u(out_row_base + c + 2 * dst_stride, valid_bytes, v2);                          \
        if (r + 3 < n_rows) hvx_vec_store_u(out_row_base + c + 3 * dst_stride, valid_bytes, v3);                          \
    }

#define W8A8_COL_FULL_QUAD_1G_FP16(SFX, STORE_MODE)                                                                 \
    for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                              \
        HVX_Vector v0_lo, v1_lo, v2_lo, v3_lo;                                                                      \
        HVX_Vector v0_hi, v1_hi, v2_hi, v3_hi;                                                                      \
        w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c, da, NULL, 0, n_rows, r, true,                         \
                                       &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                             \
        w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c + 32, da, NULL, 0, n_rows, r, true,                    \
                                       &v0_hi, &v1_hi, &v2_hi, &v3_hi);                                             \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 0 * dst_stride,                                               \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v0_lo, v0_hi, bias_row0, bias_row_bytes, 0, c)); \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 1 * dst_stride,                                               \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v1_lo, v1_hi, bias_row0, bias_row_bytes, 1, c)); \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 2 * dst_stride,                                               \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v2_lo, v2_hi, bias_row0, bias_row_bytes, 2, c)); \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 3 * dst_stride,                                               \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v3_lo, v3_hi, bias_row0, bias_row_bytes, 3, c)); \
    }

#define W8A8_COL_STEP_1G_FP16(SFX, STORE_MODE)                                                                     \
    (void) bias_row0;                                                                                              \
    (void) bias_row_bytes;                                                                                         \
    if (r + 3 < n_rows) {                                                                                          \
        W8A8_COL_FULL_QUAD_1G_FP16(SFX, STORE_MODE)                                                                \
    } else {                                                                                                       \
        for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                         \
            HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                    \
            HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();                    \
            w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c, da, NULL, 0, n_rows, r, false,                   \
                                           &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                        \
            w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c + 32, da, NULL, 0, n_rows, r, false,              \
                                           &v0_hi, &v1_hi, &v2_hi, &v3_hi);                                        \
            hvx_vmemu(out_row0 + c + 0 * dst_stride) =                                                             \
                HMX_INT_PACK_FP16_ROW_##SFX(v0_lo, v0_hi, bias_row0, bias_row_bytes, 0, c);                        \
            if (r + 1 < n_rows) {                                                                                  \
                hvx_vmemu(out_row0 + c + 1 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v1_lo, v1_hi, bias_row0, bias_row_bytes, 1, c);                    \
            }                                                                                                      \
            if (r + 2 < n_rows) {                                                                                  \
                hvx_vmemu(out_row0 + c + 2 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v2_lo, v2_hi, bias_row0, bias_row_bytes, 2, c);                    \
            }                                                                                                      \
            if (r + 3 < n_rows) {                                                                                  \
                hvx_vmemu(out_row0 + c + 3 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v3_lo, v3_hi, bias_row0, bias_row_bytes, 3, c);                    \
            }                                                                                                      \
        }                                                                                                          \
    }                                                                                                              \
    if (limit_c_aligned < limit_c) {                                                                               \
        const size_t c           = limit_c_aligned;                                                                \
        const size_t valid_c     = limit_c - c;                                                                    \
        const size_t valid_bytes = valid_c * sizeof(__fp16);                                                       \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                        \
        HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();         \
        w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c, da, NULL, 0, n_rows, r, false,                       \
                                       &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                            \
        if (valid_c > 32) {                                                                                        \
            w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c + 32, da, NULL, 0, n_rows, r, false,              \
                                           &v0_hi, &v1_hi, &v2_hi, &v3_hi);                                        \
        }                                                                                                          \
        hvx_vec_store_u(out_row0 + c + 0 * dst_stride, valid_bytes,                                                \
                        HMX_INT_PACK_FP16_TAIL_##SFX(v0_lo, v0_hi, bias_row0, bias_row_bytes, 0, c, valid_c));     \
        if (r + 1 < n_rows) {                                                                                      \
            hvx_vec_store_u(out_row0 + c + 1 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v1_lo, v1_hi, bias_row0, bias_row_bytes, 1, c, valid_c)); \
        }                                                                                                          \
        if (r + 2 < n_rows) {                                                                                      \
            hvx_vec_store_u(out_row0 + c + 2 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v2_lo, v2_hi, bias_row0, bias_row_bytes, 2, c, valid_c)); \
        }                                                                                                          \
        if (r + 3 < n_rows) {                                                                                      \
            hvx_vec_store_u(out_row0 + c + 3 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v3_lo, v3_hi, bias_row0, bias_row_bytes, 3, c, valid_c)); \
        }                                                                                                          \
    }

#define W8A8_COL_FULL_QUAD_2G_FP16(SFX, STORE_MODE)                                                                \
    for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                             \
        HVX_Vector v0_lo, v1_lo, v2_lo, v3_lo;                                                                     \
        HVX_Vector v0_hi, v1_hi, v2_hi, v3_hi;                                                                     \
        w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c, da0, da1, NULL, 0, n_rows, r, true,     \
                                       &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                            \
        w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c + 32, da0, da1, NULL, 0, n_rows, r, true,\
                                       &v0_hi, &v1_hi, &v2_hi, &v3_hi);                                            \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 0 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v0_lo, v0_hi, bias_row0, bias_row_bytes, 0, c));\
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 1 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v1_lo, v1_hi, bias_row0, bias_row_bytes, 1, c));\
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 2 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v2_lo, v2_hi, bias_row0, bias_row_bytes, 2, c));\
        HMX_INT_STORE_VEC_##STORE_MODE(out_row0 + c + 3 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v3_lo, v3_hi, bias_row0, bias_row_bytes, 3, c));\
    }

#define W8A8_COL_STEP_2G_FP16(SFX, STORE_MODE)                                                                     \
    (void) bias_row0;                                                                                              \
    (void) bias_row_bytes;                                                                                         \
    if (r + 3 < n_rows) {                                                                                          \
        W8A8_COL_FULL_QUAD_2G_FP16(SFX, STORE_MODE)                                                                \
    } else {                                                                                                       \
        for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                         \
            HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                    \
            HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();                    \
            w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c, da0, da1, NULL, 0, n_rows, r, false,\
                                           &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                        \
            w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c + 32, da0, da1, NULL, 0, n_rows, r,  \
                                           false, &v0_hi, &v1_hi, &v2_hi, &v3_hi);                                 \
            hvx_vmemu(out_row0 + c + 0 * dst_stride) =                                                             \
                HMX_INT_PACK_FP16_ROW_##SFX(v0_lo, v0_hi, bias_row0, bias_row_bytes, 0, c);                        \
            if (r + 1 < n_rows) {                                                                                  \
                hvx_vmemu(out_row0 + c + 1 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v1_lo, v1_hi, bias_row0, bias_row_bytes, 1, c);                    \
            }                                                                                                      \
            if (r + 2 < n_rows) {                                                                                  \
                hvx_vmemu(out_row0 + c + 2 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v2_lo, v2_hi, bias_row0, bias_row_bytes, 2, c);                    \
            }                                                                                                      \
            if (r + 3 < n_rows) {                                                                                  \
                hvx_vmemu(out_row0 + c + 3 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v3_lo, v3_hi, bias_row0, bias_row_bytes, 3, c);                    \
            }                                                                                                      \
        }                                                                                                          \
    }                                                                                                              \
    if (limit_c_aligned < limit_c) {                                                                               \
        const size_t c           = limit_c_aligned;                                                                \
        const size_t valid_c     = limit_c - c;                                                                    \
        const size_t valid_bytes = valid_c * sizeof(__fp16);                                                       \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                        \
        HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();         \
        w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c, da0, da1, NULL, 0, n_rows, r, false,    \
                                       &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                            \
        if (valid_c > 32) {                                                                                        \
            w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c + 32, da0, da1, NULL, 0, n_rows, r,  \
                                           false, &v0_hi, &v1_hi, &v2_hi, &v3_hi);                                 \
        }                                                                                                          \
        hvx_vec_store_u(out_row0 + c + 0 * dst_stride, valid_bytes,                                                \
                        HMX_INT_PACK_FP16_TAIL_##SFX(v0_lo, v0_hi, bias_row0, bias_row_bytes, 0, c, valid_c));     \
        if (r + 1 < n_rows) {                                                                                      \
            hvx_vec_store_u(out_row0 + c + 1 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v1_lo, v1_hi, bias_row0, bias_row_bytes, 1, c, valid_c)); \
        }                                                                                                          \
        if (r + 2 < n_rows) {                                                                                      \
            hvx_vec_store_u(out_row0 + c + 2 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v2_lo, v2_hi, bias_row0, bias_row_bytes, 2, c, valid_c)); \
        }                                                                                                          \
        if (r + 3 < n_rows) {                                                                                      \
            hvx_vec_store_u(out_row0 + c + 3 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v3_lo, v3_hi, bias_row0, bias_row_bytes, 3, c, valid_c)); \
        }                                                                                                          \
    }

#define W8A8_COL_FULL_QUAD_NG_FP16(SFX, STORE_MODE)                                                                    \
    for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                                 \
        HVX_Vector v0_lo, v1_lo, v2_lo, v3_lo;                                                                         \
        HVX_Vector v0_hi, v1_hi, v2_hi, v3_hi;                                                                         \
        w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,              \
                                       NULL, c, 0, n_rows, r, true, &v0_lo, &v1_lo, &v2_lo, &v3_lo);                   \
        w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,              \
                                       NULL, c + 32, 0, n_rows, r, true, &v0_hi, &v1_hi, &v2_hi, &v3_hi);              \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 0 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v0_lo, v0_hi, bias_row_base, bias_row_bytes, 0, c));\
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 1 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v1_lo, v1_hi, bias_row_base, bias_row_bytes, 1, c));\
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 2 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v2_lo, v2_hi, bias_row_base, bias_row_bytes, 2, c));\
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 3 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v3_lo, v3_hi, bias_row_base, bias_row_bytes, 3, c));\
    }

#define W8A8_COL_STEP_NG_FP16(SFX, STORE_MODE)                                                                         \
    (void) bias_row_base;                                                                                              \
    (void) bias_row_bytes;                                                                                             \
    if (r + 3 < n_rows) {                                                                                              \
        W8A8_COL_FULL_QUAD_NG_FP16(SFX, STORE_MODE)                                                                    \
    } else {                                                                                                           \
        for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                             \
            HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                        \
            HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();                        \
            w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,          \
                                           NULL, c, 0, n_rows, r, false, &v0_lo, &v1_lo, &v2_lo, &v3_lo);              \
            w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,          \
                                           NULL, c + 32, 0, n_rows, r, false, &v0_hi, &v1_hi, &v2_hi, &v3_hi);         \
            hvx_vmemu(out_row_base + c + 0 * dst_stride) =                                                             \
                HMX_INT_PACK_FP16_ROW_##SFX(v0_lo, v0_hi, bias_row_base, bias_row_bytes, 0, c);                        \
            if (r + 1 < n_rows) {                                                                                      \
                hvx_vmemu(out_row_base + c + 1 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v1_lo, v1_hi, bias_row_base, bias_row_bytes, 1, c);                    \
            }                                                                                                          \
            if (r + 2 < n_rows) {                                                                                      \
                hvx_vmemu(out_row_base + c + 2 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v2_lo, v2_hi, bias_row_base, bias_row_bytes, 2, c);                    \
            }                                                                                                          \
            if (r + 3 < n_rows) {                                                                                      \
                hvx_vmemu(out_row_base + c + 3 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v3_lo, v3_hi, bias_row_base, bias_row_bytes, 3, c);                    \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    if (limit_c_aligned < limit_c) {                                                                                   \
        const size_t c           = limit_c_aligned;                                                                    \
        const size_t valid_c     = limit_c - c;                                                                        \
        const size_t valid_bytes = valid_c * sizeof(__fp16);                                                           \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                            \
        HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();             \
        w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,              \
                                       NULL, c, 0, n_rows, r, false, &v0_lo, &v1_lo, &v2_lo, &v3_lo);                  \
        if (valid_c > 32) {                                                                                            \
            w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,          \
                                           NULL, c + 32, 0, n_rows, r, false, &v0_hi, &v1_hi, &v2_hi, &v3_hi);         \
        }                                                                                                              \
        hvx_vec_store_u(out_row_base + c + 0 * dst_stride, valid_bytes,                                                \
                        HMX_INT_PACK_FP16_TAIL_##SFX(v0_lo, v0_hi, bias_row_base, bias_row_bytes, 0, c, valid_c));     \
        if (r + 1 < n_rows) {                                                                                          \
            hvx_vec_store_u(out_row_base + c + 1 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v1_lo, v1_hi, bias_row_base, bias_row_bytes, 1, c, valid_c)); \
        }                                                                                                              \
        if (r + 2 < n_rows) {                                                                                          \
            hvx_vec_store_u(out_row_base + c + 2 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v2_lo, v2_hi, bias_row_base, bias_row_bytes, 2, c, valid_c)); \
        }                                                                                                              \
        if (r + 3 < n_rows) {                                                                                          \
            hvx_vec_store_u(out_row_base + c + 3 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v3_lo, v3_hi, bias_row_base, bias_row_bytes, 3, c, valid_c)); \
        }                                                                                                              \
    }

#define W8A8_COL_FULL_QUAD_CHUNKED_FP16(SFX, STORE_MODE)                                                               \
    for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                                 \
        HVX_Vector v0_lo, v1_lo, v2_lo, v3_lo;                                                                         \
        HVX_Vector v0_hi, v1_hi, v2_hi, v3_hi;                                                                         \
        w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,             \
                                            n_groups, lane, da_ptr, sign_80, NULL, c, 0, n_rows, r, true,              \
                                            &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                           \
        w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,             \
                                            n_groups, lane, da_ptr, sign_80, NULL, c + 32, 0, n_rows, r, true,         \
                                            &v0_hi, &v1_hi, &v2_hi, &v3_hi);                                           \
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 0 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v0_lo, v0_hi, bias_row_base, bias_row_bytes, 0, c));\
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 1 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v1_lo, v1_hi, bias_row_base, bias_row_bytes, 1, c));\
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 2 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v2_lo, v2_hi, bias_row_base, bias_row_bytes, 2, c));\
        HMX_INT_STORE_VEC_##STORE_MODE(out_row_base + c + 3 * dst_stride,                                              \
                                       HMX_INT_PACK_FP16_ROW_##SFX(v3_lo, v3_hi, bias_row_base, bias_row_bytes, 3, c));\
    }

#define W8A8_COL_STEP_CHUNKED_FP16(SFX, STORE_MODE)                                                                    \
    (void) bias_row_base;                                                                                              \
    (void) bias_row_bytes;                                                                                             \
    if (r + 3 < n_rows) {                                                                                              \
        W8A8_COL_FULL_QUAD_CHUNKED_FP16(SFX, STORE_MODE)                                                               \
    } else {                                                                                                           \
        for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                             \
            HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                        \
            HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();                        \
            w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,         \
                                                n_groups, lane, da_ptr, sign_80, NULL, c, 0, n_rows, r, false,         \
                                                &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                       \
            w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,         \
                                                n_groups, lane, da_ptr, sign_80, NULL, c + 32, 0, n_rows, r, false,    \
                                                &v0_hi, &v1_hi, &v2_hi, &v3_hi);                                       \
            hvx_vmemu(out_row_base + c + 0 * dst_stride) =                                                             \
                HMX_INT_PACK_FP16_ROW_##SFX(v0_lo, v0_hi, bias_row_base, bias_row_bytes, 0, c);                        \
            if (r + 1 < n_rows) {                                                                                      \
                hvx_vmemu(out_row_base + c + 1 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v1_lo, v1_hi, bias_row_base, bias_row_bytes, 1, c);                    \
            }                                                                                                          \
            if (r + 2 < n_rows) {                                                                                      \
                hvx_vmemu(out_row_base + c + 2 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v2_lo, v2_hi, bias_row_base, bias_row_bytes, 2, c);                    \
            }                                                                                                          \
            if (r + 3 < n_rows) {                                                                                      \
                hvx_vmemu(out_row_base + c + 3 * dst_stride) =                                                         \
                    HMX_INT_PACK_FP16_ROW_##SFX(v3_lo, v3_hi, bias_row_base, bias_row_bytes, 3, c);                    \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    if (limit_c_aligned < limit_c) {                                                                                   \
        const size_t c           = limit_c_aligned;                                                                    \
        const size_t valid_c     = limit_c - c;                                                                        \
        const size_t valid_bytes = valid_c * sizeof(__fp16);                                                           \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                            \
        HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();             \
        w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,             \
                                            n_groups, lane, da_ptr, sign_80, NULL, c, 0, n_rows, r, false,             \
                                            &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                           \
        if (valid_c > 32) {                                                                                            \
            w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,         \
                                                n_groups, lane, da_ptr, sign_80, NULL, c + 32, 0, n_rows, r, false,    \
                                                &v0_hi, &v1_hi, &v2_hi, &v3_hi);                                       \
        }                                                                                                              \
        hvx_vec_store_u(out_row_base + c + 0 * dst_stride, valid_bytes,                                                \
                        HMX_INT_PACK_FP16_TAIL_##SFX(v0_lo, v0_hi, bias_row_base, bias_row_bytes, 0, c, valid_c));     \
        if (r + 1 < n_rows) {                                                                                          \
            hvx_vec_store_u(out_row_base + c + 1 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v1_lo, v1_hi, bias_row_base, bias_row_bytes, 1, c, valid_c)); \
        }                                                                                                              \
        if (r + 2 < n_rows) {                                                                                          \
            hvx_vec_store_u(out_row_base + c + 2 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v2_lo, v2_hi, bias_row_base, bias_row_bytes, 2, c, valid_c)); \
        }                                                                                                              \
        if (r + 3 < n_rows) {                                                                                          \
            hvx_vec_store_u(out_row_base + c + 3 * dst_stride, valid_bytes,                                            \
                            HMX_INT_PACK_FP16_TAIL_##SFX(v3_lo, v3_hi, bias_row_base, bias_row_bytes, 3, c, valid_c)); \
        }                                                                                                              \
    }

#define DEFINE_W8A8_COL_CHUNK_MODE(DST_SFX, DST_T, ALIGN_MASK, STEP_1G, STEP_2G, STEP_NG, STEP_CHUNKED, SFX, STORE_MODE)       \
static __attribute__((noinline)) void transfer_output_chunk_w8a8_sm_to_##DST_SFX##_col_chunk_1g_##SFX##_##STORE_MODE(          \
        void * restrict dst_ptr,                                                                                               \
        const void * restrict bias,                                                                                            \
        const void * restrict vtcm_src_void,                                                                                   \
        const float * restrict vtcm_act_scales,                                                                                \
        const float * restrict vtcm_weight_scales,                                                                             \
        uint32_t start_row,                                                                                                    \
        uint32_t n_rows,                                                                                                       \
        uint32_t mc_pad,                                                                                                       \
        uint32_t c_first,                                                                                                      \
        uint32_t c_len,                                                                                                        \
        uint32_t total_n_cols,                                                                                                 \
        uint32_t n_groups,                                                                                                     \
        uint32_t dst_stride,                                                                                                   \
        uint32_t bias_stride,                                                                                                  \
        uint32_t dst_cols,                                                                                                     \
        size_t bias_elem_size) {                                                                                               \
    (void) mc_pad;                                                                                                             \
    (void) n_groups;                                                                                                           \
    DST_T * restrict dst              = (DST_T *) dst_ptr;                                                                     \
    const uint8_t * restrict vtcm_src = (const uint8_t *) vtcm_src_void;                                                       \
    const size_t total_col_tiles      = hmx_ceil_div(total_n_cols, 32);                                                        \
    const size_t row_tile_stride      = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                                            \
    const size_t c0_first             = c_first / 32;                                                                          \
    const size_t limit_c              = hex_smin(c_len, dst_cols);                                                             \
    const size_t limit_c_aligned      = (limit_c & (ALIGN_MASK));                                                              \
    const size_t bias_row_bytes       = (size_t) bias_stride * bias_elem_size;                                                 \
    uint32_t da_buf[1][32] __attribute__((aligned(128)));                                                                      \
    for (size_t r_base = 0; r_base < n_rows; ) {                                                                               \
        const size_t r_idx_base = start_row + r_base;                                                                          \
        const size_t r_blk      = r_idx_base >> 5;                                                                             \
        const size_t r0         = r_idx_base >> 6;                                                                             \
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));                     \
        const size_t r_end      = r_base + r_step;                                                                             \
        uint32_t (*da_ptr)[32]  = stage_act_scales_to_l1d(da_buf, vtcm_act_scales + (r_blk << 5), 0, 1);                       \
        for (size_t r = r_base; r < r_end; r += 4) {                                                                           \
            const size_t     r_idx0    = start_row + r;                                                                        \
            const size_t     rg        = (r_idx0 & 63u) >> 2;                                                                  \
            const size_t     lane      = r_idx0 & 31u;                                                                         \
            const uint32_t * da        = &da_ptr[0][lane];                                                                     \
            const uint8_t *  tile_base = vtcm_src + r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;     \
            const float *    dw_base   = vtcm_weight_scales + c0_first * 32;                                                   \
            DST_T *          out_row0  = dst + r * dst_stride;                                                                 \
            const uint8_t *  bias_row0 = HMX_INT_BIAS_ROW_##SFX(bias, r * bias_row_bytes);                                     \
            STEP_1G(SFX, STORE_MODE)                                                                                           \
        }                                                                                                                      \
        r_base = r_end;                                                                                                        \
    }                                                                                                                          \
}                                                                                                                              \
static __attribute__((noinline)) void transfer_output_chunk_w8a8_sm_to_##DST_SFX##_col_chunk_2g_##SFX##_##STORE_MODE(          \
        void * restrict dst_ptr,                                                                                               \
        const void * restrict bias,                                                                                            \
        const void * restrict vtcm_src_void,                                                                                   \
        const float * restrict vtcm_act_scales,                                                                                \
        const float * restrict vtcm_weight_scales,                                                                             \
        uint32_t start_row,                                                                                                    \
        uint32_t n_rows,                                                                                                       \
        uint32_t mc_pad,                                                                                                       \
        uint32_t c_first,                                                                                                      \
        uint32_t c_len,                                                                                                        \
        uint32_t total_n_cols,                                                                                                 \
        uint32_t n_groups,                                                                                                     \
        uint32_t dst_stride,                                                                                                   \
        uint32_t bias_stride,                                                                                                  \
        uint32_t dst_cols,                                                                                                     \
        size_t bias_elem_size) {                                                                                               \
    (void) n_groups;                                                                                                           \
    DST_T * restrict dst              = (DST_T *) dst_ptr;                                                                     \
    const uint8_t * restrict vtcm_src = (const uint8_t *) vtcm_src_void;                                                       \
    const size_t total_col_tiles      = hmx_ceil_div(total_n_cols, 32);                                                        \
    const size_t n_row_tiles_64       = mc_pad / 64;                                                                           \
    const size_t group_stride         = n_row_tiles_64 * total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                           \
    const size_t row_tile_stride      = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                                            \
    const size_t c0_first             = c_first / 32;                                                                          \
    const size_t limit_c              = hex_smin(c_len, dst_cols);                                                             \
    const size_t limit_c_aligned      = (limit_c & (ALIGN_MASK));                                                              \
    const size_t bias_row_bytes       = (size_t) bias_stride * bias_elem_size;                                                 \
    uint32_t da_buf[2][32] __attribute__((aligned(128)));                                                                      \
    for (size_t r_base = 0; r_base < n_rows; ) {                                                                               \
        const size_t r_idx_base = start_row + r_base;                                                                          \
        const size_t r_blk      = r_idx_base >> 5;                                                                             \
        const size_t r0         = r_idx_base >> 6;                                                                             \
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));                     \
        const size_t r_end      = r_base + r_step;                                                                             \
        uint32_t (*da_ptr)[32]  = stage_act_scales_to_l1d(da_buf, vtcm_act_scales + (r_blk << 5), mc_pad, 2);                  \
        for (size_t r = r_base; r < r_end; r += 4) {                                                                           \
            const size_t     r_idx0     = start_row + r;                                                                       \
            const size_t     rg         = (r_idx0 & 63u) >> 2;                                                                 \
            const size_t     lane       = r_idx0 & 31u;                                                                        \
            const uint32_t * da0        = &da_ptr[0][lane];                                                                    \
            const uint32_t * da1        = &da_ptr[1][lane];                                                                    \
            const uint8_t *  tile_base0 = vtcm_src + r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;    \
            const uint8_t *  tile_base1 = tile_base0 + group_stride;                                                           \
            const float *    dw_base    = vtcm_weight_scales + c0_first * 64;                                                  \
            DST_T *          out_row0   = dst + r * dst_stride;                                                                \
            const uint8_t *  bias_row0  = HMX_INT_BIAS_ROW_##SFX(bias, r * bias_row_bytes);                                    \
            STEP_2G(SFX, STORE_MODE)                                                                                           \
        }                                                                                                                      \
        r_base = r_end;                                                                                                        \
    }                                                                                                                          \
}                                                                                                                              \
static __attribute__((noinline)) void transfer_output_chunk_w8a8_sm_to_##DST_SFX##_col_chunk_ng_##SFX##_##STORE_MODE(          \
        void * restrict dst_ptr,                                                                                               \
        const void * restrict bias,                                                                                            \
        const void * restrict vtcm_src_void,                                                                                   \
        const float * restrict vtcm_act_scales,                                                                                \
        const float * restrict vtcm_weight_scales,                                                                             \
        uint32_t start_row,                                                                                                    \
        uint32_t n_rows,                                                                                                       \
        uint32_t mc_pad,                                                                                                       \
        uint32_t c_first,                                                                                                      \
        uint32_t c_len,                                                                                                        \
        uint32_t total_n_cols,                                                                                                 \
        uint32_t n_groups,                                                                                                     \
        uint32_t dst_stride,                                                                                                   \
        uint32_t bias_stride,                                                                                                  \
        uint32_t dst_cols,                                                                                                     \
        size_t bias_elem_size) {                                                                                               \
    DST_T * restrict dst              = (DST_T *) dst_ptr;                                                                     \
    const uint8_t * restrict vtcm_src = (const uint8_t *) vtcm_src_void;                                                       \
    const size_t total_col_tiles      = hmx_ceil_div(total_n_cols, 32);                                                        \
    const size_t n_row_tiles_64       = mc_pad / 64;                                                                           \
    const size_t group_stride         = n_row_tiles_64 * total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                           \
    const size_t row_tile_stride      = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                                            \
    const size_t c0_first             = c_first / 32;                                                                          \
    const HVX_Vector sign_80          = Q6_Vb_vsplat_R(0x80);                                                                  \
    const size_t limit_c              = hex_smin(c_len, dst_cols);                                                             \
    const size_t limit_c_aligned      = (limit_c & (ALIGN_MASK));                                                              \
    const size_t bias_row_bytes       = (size_t) bias_stride * bias_elem_size;                                                 \
    const float * dw_base             = vtcm_weight_scales + c0_first * (size_t) n_groups * 32;                                \
    uint32_t da_buf[16][32] __attribute__((aligned(128)));                                                                     \
    for (size_t r_base = 0; r_base < n_rows; ) {                                                                               \
        const size_t r_idx_base = start_row + r_base;                                                                          \
        const size_t r_blk      = r_idx_base >> 5;                                                                             \
        const size_t r0         = r_idx_base >> 6;                                                                             \
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));                     \
        const size_t r_end      = r_base + r_step;                                                                             \
        uint32_t (*da_ptr)[32]  = stage_act_scales_to_l1d(da_buf, vtcm_act_scales + (r_blk << 5), mc_pad, n_groups);           \
        for (size_t r = r_base; r < r_end; r += 4) {                                                                           \
            const size_t r_idx0 = start_row + r;                                                                               \
            const size_t rg     = (r_idx0 & 63u) >> 2;                                                                         \
            const size_t lane   = r_idx0 & 31u;                                                                                \
            const uint8_t *  tile_base     = vtcm_src + r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE +           \
                                             rg * 128;                                                                         \
            const uint32_t * da_quad_base  = &da_ptr[0][lane];                                                                 \
            DST_T *          out_row_base  = dst + r * dst_stride;                                                             \
            const uint8_t *  bias_row_base = HMX_INT_BIAS_ROW_##SFX(bias, r * bias_row_bytes);                                 \
            STEP_NG(SFX, STORE_MODE)                                                                                           \
        }                                                                                                                      \
        r_base = r_end;                                                                                                        \
    }                                                                                                                          \
}                                                                                                                              \
static __attribute__((noinline)) void transfer_output_chunk_w8a8_sm_to_##DST_SFX##_col_chunk_chunked_##SFX##_##STORE_MODE(     \
        void * restrict dst_ptr,                                                                                               \
        const void * restrict bias,                                                                                            \
        const void * restrict vtcm_src_void,                                                                                   \
        const float * restrict vtcm_act_scales,                                                                                \
        const float * restrict vtcm_weight_scales,                                                                             \
        uint32_t start_row,                                                                                                    \
        uint32_t n_rows,                                                                                                       \
        uint32_t mc_pad,                                                                                                       \
        uint32_t c_first,                                                                                                      \
        uint32_t c_len,                                                                                                        \
        uint32_t total_n_cols,                                                                                                 \
        uint32_t n_groups,                                                                                                     \
        uint32_t dst_stride,                                                                                                   \
        uint32_t bias_stride,                                                                                                  \
        uint32_t dst_cols,                                                                                                     \
        size_t bias_elem_size) {                                                                                               \
    DST_T * restrict dst              = (DST_T *) dst_ptr;                                                                     \
    const uint8_t * restrict vtcm_src = (const uint8_t *) vtcm_src_void;                                                       \
    const size_t total_col_tiles      = hmx_ceil_div(total_n_cols, 32);                                                        \
    const size_t n_row_tiles_64       = mc_pad / 64;                                                                           \
    const size_t group_stride         = n_row_tiles_64 * total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                           \
    const size_t row_tile_stride      = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                                            \
    const size_t c0_first             = c_first / 32;                                                                          \
    const HVX_Vector sign_80          = Q6_Vb_vsplat_R(0x80);                                                                  \
    const size_t limit_c              = hex_smin(c_len, dst_cols);                                                             \
    const size_t limit_c_aligned      = (limit_c & (ALIGN_MASK));                                                              \
    const size_t bias_row_bytes       = (size_t) bias_stride * bias_elem_size;                                                 \
    const float * dw_base             = vtcm_weight_scales + c0_first * (size_t) n_groups * 32;                                \
    uint32_t da_buf[16][32] __attribute__((aligned(128)));                                                                     \
    uint32_t (*da_ptr)[32] = da_buf;                                                                                           \
    for (size_t r_base = 0; r_base < n_rows; ) {                                                                               \
        const size_t r_idx_base           = start_row + r_base;                                                                \
        const size_t r_blk                = r_idx_base >> 5;                                                                   \
        const size_t r0                   = r_idx_base >> 6;                                                                   \
        const size_t r_step               = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));           \
        const size_t r_end                = r_base + r_step;                                                                   \
        const float * act_scales_blk_base = vtcm_act_scales + (r_blk << 5);                                                    \
        for (size_t r = r_base; r < r_end; r += 4) {                                                                           \
            const size_t r_idx0 = start_row + r;                                                                               \
            const size_t rg     = (r_idx0 & 63u) >> 2;                                                                         \
            const size_t lane   = r_idx0 & 31u;                                                                                \
            const uint8_t * tile_base     = vtcm_src + r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;  \
            DST_T *         out_row_base  = dst + r * dst_stride;                                                              \
            const uint8_t * bias_row_base = HMX_INT_BIAS_ROW_##SFX(bias, r * bias_row_bytes);                                  \
            STEP_CHUNKED(SFX, STORE_MODE)                                                                                      \
        }                                                                                                                      \
        r_base = r_end;                                                                                                        \
    }                                                                                                                          \
}

#define DEFINE_W8A8_COL_CHUNK(DST_SFX, DST_T, ALIGN_MASK, STEP_1G, STEP_2G, STEP_NG, STEP_CHUNKED, SFX)          \
    DEFINE_W8A8_COL_CHUNK_MODE(DST_SFX, DST_T, ALIGN_MASK, STEP_1G, STEP_2G, STEP_NG, STEP_CHUNKED, SFX, nt)     \
    DEFINE_W8A8_COL_CHUNK_MODE(DST_SFX, DST_T, ALIGN_MASK, STEP_1G, STEP_2G, STEP_NG, STEP_CHUNKED, SFX, u)

DEFINE_W8A8_COL_CHUNK(fp32, float,  ~31u, W8A8_COL_STEP_1G_FP32, W8A8_COL_STEP_2G_FP32, W8A8_COL_STEP_NG_FP32, W8A8_COL_STEP_CHUNKED_FP32, none)
DEFINE_W8A8_COL_CHUNK(fp32, float,  ~31u, W8A8_COL_STEP_1G_FP32, W8A8_COL_STEP_2G_FP32, W8A8_COL_STEP_NG_FP32, W8A8_COL_STEP_CHUNKED_FP32, f16)
DEFINE_W8A8_COL_CHUNK(fp32, float,  ~31u, W8A8_COL_STEP_1G_FP32, W8A8_COL_STEP_2G_FP32, W8A8_COL_STEP_NG_FP32, W8A8_COL_STEP_CHUNKED_FP32, f32)
DEFINE_W8A8_COL_CHUNK(fp16, __fp16, ~63u, W8A8_COL_STEP_1G_FP16, W8A8_COL_STEP_2G_FP16, W8A8_COL_STEP_NG_FP16, W8A8_COL_STEP_CHUNKED_FP16, none)
DEFINE_W8A8_COL_CHUNK(fp16, __fp16, ~63u, W8A8_COL_STEP_1G_FP16, W8A8_COL_STEP_2G_FP16, W8A8_COL_STEP_NG_FP16, W8A8_COL_STEP_CHUNKED_FP16, f16)
DEFINE_W8A8_COL_CHUNK(fp16, __fp16, ~63u, W8A8_COL_STEP_1G_FP16, W8A8_COL_STEP_2G_FP16, W8A8_COL_STEP_NG_FP16, W8A8_COL_STEP_CHUNKED_FP16, f32)

#define W8A8_SCAT_STEP_1G_FP32                                                             \
    for (size_t c = 0; c < limit_c_aligned; c += 32) {                                     \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();            \
        w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c, da,                          \
                                       NULL, 0, n_rows, r, false, &v0, &v1, &v2, &v3);     \
        hvx_vmemu(out_row0 + c) = v0;                                                      \
        if (out_row1) hvx_vmemu(out_row1 + c) = v1;                                        \
        if (out_row2) hvx_vmemu(out_row2 + c) = v2;                                        \
        if (out_row3) hvx_vmemu(out_row3 + c) = v3;                                        \
    }                                                                                      \
    if (limit_c_aligned < limit_c) {                                                       \
        const size_t c           = limit_c_aligned;                                        \
        const size_t valid_bytes = (limit_c - c) * sizeof(float);                          \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();            \
        w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c, da,                          \
                                       NULL, 0, n_rows, r, false, &v0, &v1, &v2, &v3);     \
        hvx_vec_store_u(out_row0 + c, valid_bytes, v0);                                    \
        if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, v1);                      \
        if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, v2);                      \
        if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, v3);                      \
    }

#define W8A8_SCAT_STEP_2G_FP32                                                                       \
    for (size_t c = 0; c < limit_c_aligned; c += 32) {                                               \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                      \
        w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c, da0, da1,                 \
                                       NULL, 0, n_rows, r, false, &v0, &v1, &v2, &v3);               \
        hvx_vmemu(out_row0 + c) = v0;                                                                \
        if (out_row1) hvx_vmemu(out_row1 + c) = v1;                                                  \
        if (out_row2) hvx_vmemu(out_row2 + c) = v2;                                                  \
        if (out_row3) hvx_vmemu(out_row3 + c) = v3;                                                  \
    }                                                                                                \
    if (limit_c_aligned < limit_c) {                                                                 \
        const size_t c           = limit_c_aligned;                                                  \
        const size_t valid_bytes = (limit_c - c) * sizeof(float);                                    \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                      \
        w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c, da0, da1,                 \
                                       NULL, 0, n_rows, r, false, &v0, &v1, &v2, &v3);               \
        hvx_vec_store_u(out_row0 + c, valid_bytes, v0);                                              \
        if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, v1);                                \
        if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, v2);                                \
        if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, v3);                                \
    }

#define W8A8_SCAT_STEP_NG_FP32                                                                          \
    for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                  \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                         \
        w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base,        \
                                       sign_80, NULL, c, 0, n_rows, r, false, &v0, &v1, &v2, &v3);      \
        hvx_vmemu(out_row0 + c) = v0;                                                                   \
        if (out_row1) hvx_vmemu(out_row1 + c) = v1;                                                     \
        if (out_row2) hvx_vmemu(out_row2 + c) = v2;                                                     \
        if (out_row3) hvx_vmemu(out_row3 + c) = v3;                                                     \
    }                                                                                                   \
    if (limit_c_aligned < limit_c) {                                                                    \
        const size_t c           = limit_c_aligned;                                                     \
        const size_t valid_bytes = (limit_c - c) * sizeof(float);                                       \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                         \
        w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base,        \
                                       sign_80, NULL, c, 0, n_rows, r, false, &v0, &v1, &v2, &v3);      \
        hvx_vec_store_u(out_row0 + c, valid_bytes, v0);                                                 \
        if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, v1);                                   \
        if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, v2);                                   \
        if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, v3);                                   \
    }

#define W8A8_SCAT_STEP_CHUNKED_FP32                                                                          \
    for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                       \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                              \
        w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,   \
                                            n_groups, lane, da_ptr, sign_80, NULL, c, 0, n_rows, r, false,   \
                                            &v0, &v1, &v2, &v3);                                             \
        hvx_vmemu(out_row0 + c) = v0;                                                                        \
        if (out_row1) hvx_vmemu(out_row1 + c) = v1;                                                          \
        if (out_row2) hvx_vmemu(out_row2 + c) = v2;                                                          \
        if (out_row3) hvx_vmemu(out_row3 + c) = v3;                                                          \
    }                                                                                                        \
    if (limit_c_aligned < limit_c) {                                                                         \
        const size_t c           = limit_c_aligned;                                                          \
        const size_t valid_bytes = (limit_c - c) * sizeof(float);                                            \
        HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();                              \
        w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,   \
                                            n_groups, lane, da_ptr, sign_80, NULL, c, 0, n_rows, r, false,   \
                                            &v0, &v1, &v2, &v3);                                             \
        hvx_vec_store_u(out_row0 + c, valid_bytes, v0);                                                      \
        if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, v1);                                        \
        if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, v2);                                        \
        if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, v3);                                        \
    }

#define W8A8_SCAT_STEP_1G_FP16                                                                               \
    for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                       \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                  \
        HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();                  \
        w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c, da,                                            \
                                       NULL, 0, n_rows, r, false, &v0_lo, &v1_lo, &v2_lo, &v3_lo);           \
        w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c + 32, da,                                       \
                                       NULL, 0, n_rows, r, false, &v0_hi, &v1_hi, &v2_hi, &v3_hi);           \
        hvx_vmemu(out_row0 + c) = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);                                 \
        if (out_row1) hvx_vmemu(out_row1 + c) = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);                   \
        if (out_row2) hvx_vmemu(out_row2 + c) = hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi);                   \
        if (out_row3) hvx_vmemu(out_row3 + c) = hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi);                   \
    }                                                                                                        \
    if (limit_c_aligned < limit_c) {                                                                         \
        const size_t c           = limit_c_aligned;                                                          \
        const size_t valid_c     = limit_c - c;                                                              \
        const size_t valid_bytes = valid_c * sizeof(__fp16);                                                 \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                  \
        HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();   \
        w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c, da,                                            \
                                       NULL, 0, n_rows, r, false, &v0_lo, &v1_lo, &v2_lo, &v3_lo);           \
        if (valid_c > 32) {                                                                                  \
            w8a8_sm_eval_tile_quad_1g_none(tile_base, dw_base, c + 32, da,                                   \
                                           NULL, 0, n_rows, r, false, &v0_hi, &v1_hi, &v2_hi, &v3_hi);       \
        }                                                                                                    \
        hvx_vec_store_u(out_row0 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));               \
        if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi)); \
        if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi)); \
        if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi)); \
    }

#define W8A8_SCAT_STEP_2G_FP16                                                                               \
    for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                       \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                  \
        HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();                  \
        w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c, da0, da1,                         \
                                       NULL, 0, n_rows, r, false, &v0_lo, &v1_lo, &v2_lo, &v3_lo);           \
        w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c + 32, da0, da1,                    \
                                       NULL, 0, n_rows, r, false, &v0_hi, &v1_hi, &v2_hi, &v3_hi);           \
        hvx_vmemu(out_row0 + c) = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);                                 \
        if (out_row1) hvx_vmemu(out_row1 + c) = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);                   \
        if (out_row2) hvx_vmemu(out_row2 + c) = hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi);                   \
        if (out_row3) hvx_vmemu(out_row3 + c) = hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi);                   \
    }                                                                                                        \
    if (limit_c_aligned < limit_c) {                                                                         \
        const size_t c           = limit_c_aligned;                                                          \
        const size_t valid_c     = limit_c - c;                                                              \
        const size_t valid_bytes = valid_c * sizeof(__fp16);                                                 \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                  \
        HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();   \
        w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c, da0, da1,                         \
                                       NULL, 0, n_rows, r, false, &v0_lo, &v1_lo, &v2_lo, &v3_lo);           \
        if (valid_c > 32) {                                                                                  \
            w8a8_sm_eval_tile_quad_2g_none(tile_base0, tile_base1, dw_base, c + 32, da0, da1,                \
                                           NULL, 0, n_rows, r, false, &v0_hi, &v1_hi, &v2_hi, &v3_hi);       \
        }                                                                                                    \
        hvx_vec_store_u(out_row0 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));               \
        if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi)); \
        if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi)); \
        if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi)); \
    }

#define W8A8_SCAT_STEP_NG_FP16                                                                                 \
    for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                         \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                    \
        HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();                    \
        w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,      \
                                       NULL, c, 0, n_rows, r, false, &v0_lo, &v1_lo, &v2_lo, &v3_lo);          \
        w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,      \
                                       NULL, c + 32, 0, n_rows, r, false, &v0_hi, &v1_hi, &v2_hi, &v3_hi);     \
        hvx_vmemu(out_row0 + c) = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);                                   \
        if (out_row1) hvx_vmemu(out_row1 + c) = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);                     \
        if (out_row2) hvx_vmemu(out_row2 + c) = hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi);                     \
        if (out_row3) hvx_vmemu(out_row3 + c) = hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi);                     \
    }                                                                                                          \
    if (limit_c_aligned < limit_c) {                                                                           \
        const size_t c           = limit_c_aligned;                                                            \
        const size_t valid_c     = limit_c - c;                                                                \
        const size_t valid_bytes = valid_c * sizeof(__fp16);                                                   \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                    \
        HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();     \
        w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,      \
                                       NULL, c, 0, n_rows, r, false, &v0_lo, &v1_lo, &v2_lo, &v3_lo);          \
        if (valid_c > 32) {                                                                                    \
            w8a8_sm_eval_tile_quad_ng_none(tile_base, dw_base, group_stride, n_groups, da_quad_base, sign_80,  \
                                           NULL, c + 32, 0, n_rows, r, false, &v0_hi, &v1_hi, &v2_hi, &v3_hi); \
        }                                                                                                      \
        hvx_vec_store_u(out_row0 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));                 \
        if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));   \
        if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi));   \
        if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi));   \
    }

#define W8A8_SCAT_STEP_CHUNKED_FP16                                                                               \
    for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                            \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                       \
        HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();                       \
        w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,        \
                                            n_groups, lane, da_ptr, sign_80, NULL, c, 0, n_rows, r, false,        \
                                            &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                      \
        w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,        \
                                            n_groups, lane, da_ptr, sign_80, NULL, c + 32, 0, n_rows, r, false,   \
                                            &v0_hi, &v1_hi, &v2_hi, &v3_hi);                                      \
        hvx_vmemu(out_row0 + c) = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);                                      \
        if (out_row1) hvx_vmemu(out_row1 + c) = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);                        \
        if (out_row2) hvx_vmemu(out_row2 + c) = hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi);                        \
        if (out_row3) hvx_vmemu(out_row3 + c) = hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi);                        \
    }                                                                                                             \
    if (limit_c_aligned < limit_c) {                                                                              \
        const size_t c           = limit_c_aligned;                                                               \
        const size_t valid_c     = limit_c - c;                                                                   \
        const size_t valid_bytes = valid_c * sizeof(__fp16);                                                      \
        HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();                       \
        HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();        \
        w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,        \
                                            n_groups, lane, da_ptr, sign_80, NULL, c, 0, n_rows, r, false,        \
                                            &v0_lo, &v1_lo, &v2_lo, &v3_lo);                                      \
        if (valid_c > 32) {                                                                                       \
            w8a8_sm_eval_tile_quad_chunked_none(tile_base, dw_base, act_scales_blk_base, group_stride, mc_pad,    \
                                                n_groups, lane, da_ptr, sign_80, NULL, c + 32, 0, n_rows, r,      \
                                                false, &v0_hi, &v1_hi, &v2_hi, &v3_hi);                           \
        }                                                                                                         \
        hvx_vec_store_u(out_row0 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));                    \
        if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));      \
        if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi));      \
        if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi));      \
    }

typedef void (*int_output_col_chunk_fn_t)(
        void * restrict dst,
        const void * restrict bias,
        const void * restrict vtcm_src,
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
        uint32_t bias_stride,
        uint32_t dst_cols,
        size_t bias_elem_size);

#define HMX_SELECT_INT_COL_FN(DST_SFX, GRP_SFX, bias, bias_f16, dst_aligned_nt)                                \
    ((dst_aligned_nt)                                                                                          \
        ? (!(bias) ? transfer_output_chunk_w8a8_sm_to_##DST_SFX##_col_chunk_##GRP_SFX##_none_nt                \
                   : ((bias_f16) ? transfer_output_chunk_w8a8_sm_to_##DST_SFX##_col_chunk_##GRP_SFX##_f16_nt   \
                                 : transfer_output_chunk_w8a8_sm_to_##DST_SFX##_col_chunk_##GRP_SFX##_f32_nt)) \
        : (!(bias) ? transfer_output_chunk_w8a8_sm_to_##DST_SFX##_col_chunk_##GRP_SFX##_none_u                 \
                   : ((bias_f16) ? transfer_output_chunk_w8a8_sm_to_##DST_SFX##_col_chunk_##GRP_SFX##_f16_u    \
                                 : transfer_output_chunk_w8a8_sm_to_##DST_SFX##_col_chunk_##GRP_SFX##_f32_u)))

static inline int_output_col_chunk_fn_t resolve_int_output_col_chunk_fn(
        uint32_t n_groups,
        size_t dst_elem_size,
        const void * bias,
        size_t bias_elem_size,
        bool dst_aligned_nt) {
    const bool dst_f16  = (dst_elem_size == sizeof(__fp16));
    const bool bias_f16 = (bias_elem_size == sizeof(__fp16));
    if (dst_f16) {
        if (n_groups <= 1u) {
            return HMX_SELECT_INT_COL_FN(fp16, 1g, bias, bias_f16, dst_aligned_nt);
        }
        if (n_groups == 2u) {
            return HMX_SELECT_INT_COL_FN(fp16, 2g, bias, bias_f16, dst_aligned_nt);
        }
        if (n_groups <= 16u) {
            return HMX_SELECT_INT_COL_FN(fp16, ng, bias, bias_f16, dst_aligned_nt);
        }
        return HMX_SELECT_INT_COL_FN(fp16, chunked, bias, bias_f16, dst_aligned_nt);
    }
    if (n_groups <= 1u) {
        return HMX_SELECT_INT_COL_FN(fp32, 1g, bias, bias_f16, dst_aligned_nt);
    }
    if (n_groups == 2u) {
        return HMX_SELECT_INT_COL_FN(fp32, 2g, bias, bias_f16, dst_aligned_nt);
    }
    if (n_groups <= 16u) {
        return HMX_SELECT_INT_COL_FN(fp32, ng, bias, bias_f16, dst_aligned_nt);
    }
    return HMX_SELECT_INT_COL_FN(fp32, chunked, bias, bias_f16, dst_aligned_nt);
}

// --- Fused W8A8 GLU Epilogue (paired gate/up -> f(gate) * up) ---

typedef struct {
    const float * act_scales_blk_base;
    uint32_t (*da_ptr)[32];
    size_t        group_stride;
    uint32_t      mc_pad;
    uint32_t      n_groups;
    uint32_t      lane;
} w8a8_sm_chunked_eval_args_t;

// Thin noinline wrappers around w8a8_sm_eval_tile_quad_*_none to bound live ranges in GLU loops.
static __attribute__((noinline)) void w8a8_sm_glu_eval_1g_quad(
        const uint8_t * restrict  tp,
        const float * restrict    dwp,
        const uint32_t * restrict da0,
        HVX_Vector * restrict     out) {
    w8a8_sm_eval_tile_quad_1g_none(tp, dwp, 0, da0, NULL, 0, 4, 0, true, &out[0], &out[1], &out[2], &out[3]);
}

static __attribute__((noinline)) void w8a8_sm_glu_eval_1g_pair_hf(
        const uint8_t * restrict  tp,
        const float * restrict    dwp,
        const uint32_t * restrict da0,
        HVX_Vector * restrict     out_hf) {
    HVX_Vector lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3;
    w8a8_sm_eval_tile_quad_1g_none(tp, dwp, 0,  da0, NULL, 0, 4, 0, true, &lo0, &lo1, &lo2, &lo3);
    w8a8_sm_eval_tile_quad_1g_none(tp, dwp, 32, da0, NULL, 0, 4, 0, true, &hi0, &hi1, &hi2, &hi3);
    out_hf[0] = hmx_pack_f32_to_f16_clamped(lo0, hi0);
    out_hf[1] = hmx_pack_f32_to_f16_clamped(lo1, hi1);
    out_hf[2] = hmx_pack_f32_to_f16_clamped(lo2, hi2);
    out_hf[3] = hmx_pack_f32_to_f16_clamped(lo3, hi3);
}

static __attribute__((noinline)) void w8a8_sm_glu_eval_1g_strip_hf(
        const uint8_t * restrict  tp,
        const float * restrict    dwp,
        const uint32_t * restrict da0,
        __fp16 * restrict         dst,
        size_t                    dst_stride,
        size_t                    n_pairs) {
    _Pragma("clang loop unroll(disable)")
    for (size_t p = 0; p < n_pairs; ++p) {
        const size_t c = p << 6;
        HVX_Vector lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3;
        w8a8_sm_eval_tile_quad_1g_none(tp, dwp, c,      da0, NULL, 0, 4, 0, true, &lo0, &lo1, &lo2, &lo3);
        w8a8_sm_eval_tile_quad_1g_none(tp, dwp, c + 32, da0, NULL, 0, 4, 0, true, &hi0, &hi1, &hi2, &hi3);
        hvx_vmem(dst + c + 0 * dst_stride) = hmx_pack_f32_to_f16_clamped(lo0, hi0);
        hvx_vmem(dst + c + 1 * dst_stride) = hmx_pack_f32_to_f16_clamped(lo1, hi1);
        hvx_vmem(dst + c + 2 * dst_stride) = hmx_pack_f32_to_f16_clamped(lo2, hi2);
        hvx_vmem(dst + c + 3 * dst_stride) = hmx_pack_f32_to_f16_clamped(lo3, hi3);
    }
}

static __attribute__((noinline)) void w8a8_sm_glu_eval_2g_quad(
        const uint8_t * restrict  tp0,
        const uint8_t * restrict  tp1,
        const float * restrict    dwp,
        const uint32_t * restrict da0,
        const uint32_t * restrict da1,
        HVX_Vector * restrict     out) {
    w8a8_sm_eval_tile_quad_2g_none(tp0, tp1, dwp, 0, da0, da1, NULL, 0, 4, 0, true, &out[0], &out[1], &out[2], &out[3]);
}

static __attribute__((noinline)) void w8a8_sm_glu_eval_2g_pair_hf(
        const uint8_t * restrict  tp0,
        const uint8_t * restrict  tp1,
        const float * restrict    dwp,
        const uint32_t * restrict da0,
        const uint32_t * restrict da1,
        HVX_Vector * restrict     out_hf) {
    HVX_Vector lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3;
    w8a8_sm_eval_tile_quad_2g_none(tp0, tp1, dwp, 0,  da0, da1, NULL, 0, 4, 0, true, &lo0, &lo1, &lo2, &lo3);
    w8a8_sm_eval_tile_quad_2g_none(tp0, tp1, dwp, 32, da0, da1, NULL, 0, 4, 0, true, &hi0, &hi1, &hi2, &hi3);
    out_hf[0] = hmx_pack_f32_to_f16_clamped(lo0, hi0);
    out_hf[1] = hmx_pack_f32_to_f16_clamped(lo1, hi1);
    out_hf[2] = hmx_pack_f32_to_f16_clamped(lo2, hi2);
    out_hf[3] = hmx_pack_f32_to_f16_clamped(lo3, hi3);
}

static __attribute__((noinline)) void w8a8_sm_glu_eval_2g_strip_hf(
        const uint8_t * restrict  tp0,
        const uint8_t * restrict  tp1,
        const float * restrict    dwp,
        const uint32_t * restrict da0,
        const uint32_t * restrict da1,
        __fp16 * restrict         dst,
        size_t                    dst_stride,
        size_t                    n_pairs) {
    _Pragma("clang loop unroll(disable)")
    for (size_t p = 0; p < n_pairs; ++p) {
        const size_t c = p << 6;
        HVX_Vector lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3;
        w8a8_sm_eval_tile_quad_2g_none(tp0, tp1, dwp, c,      da0, da1, NULL, 0, 4, 0, true, &lo0, &lo1, &lo2, &lo3);
        w8a8_sm_eval_tile_quad_2g_none(tp0, tp1, dwp, c + 32, da0, da1, NULL, 0, 4, 0, true, &hi0, &hi1, &hi2, &hi3);
        hvx_vmem(dst + c + 0 * dst_stride) = hmx_pack_f32_to_f16_clamped(lo0, hi0);
        hvx_vmem(dst + c + 1 * dst_stride) = hmx_pack_f32_to_f16_clamped(lo1, hi1);
        hvx_vmem(dst + c + 2 * dst_stride) = hmx_pack_f32_to_f16_clamped(lo2, hi2);
        hvx_vmem(dst + c + 3 * dst_stride) = hmx_pack_f32_to_f16_clamped(lo3, hi3);
    }
}

static __attribute__((noinline)) void w8a8_sm_glu_eval_ng_quad(
        const uint8_t * restrict  tp,
        const float * restrict    dw,
        size_t                    group_stride,
        uint32_t                  n_groups,
        const uint32_t * restrict da,
        HVX_Vector * restrict     out) {
    const HVX_Vector sign_80 = Q6_Vb_vsplat_R(0x80);
    w8a8_sm_eval_tile_quad_ng_none(tp, dw, group_stride, n_groups, da, sign_80, NULL, 0, 0, 4, 0, true,
                                   &out[0], &out[1], &out[2], &out[3]);
}

static __attribute__((noinline)) void w8a8_sm_glu_eval_chunked_quad(
        const uint8_t * restrict            tp,
        const float * restrict              dw,
        const w8a8_sm_chunked_eval_args_t * args,
        HVX_Vector * restrict               out) {
    const HVX_Vector sign_80 = Q6_Vb_vsplat_R(0x80);
    w8a8_sm_eval_tile_quad_chunked_none(tp, dw, args->act_scales_blk_base, args->group_stride, args->mc_pad,
                                        args->n_groups, args->lane, args->da_ptr, sign_80, NULL, 0, 0, 4, 0, true,
                                        &out[0], &out[1], &out[2], &out[3]);
}

static __attribute__((noinline)) void w8a8_sm_glu_pack_quad_hf(const HVX_Vector * restrict q, HVX_Vector * restrict out_hf) {
    out_hf[0] = hmx_pack_f32_to_f16_clamped(q[0], q[4]);
    out_hf[1] = hmx_pack_f32_to_f16_clamped(q[1], q[5]);
    out_hf[2] = hmx_pack_f32_to_f16_clamped(q[2], q[6]);
    out_hf[3] = hmx_pack_f32_to_f16_clamped(q[3], q[7]);
}

static __attribute__((noinline)) void w8a8_sm_glu_pack_and_store_tail_fp16(
        const hmx_output_glu_state_t * st,
        __fp16 * restrict dst,
        size_t dst_stride,
        const HVX_Vector * restrict g_q,
        const HVX_Vector * restrict u_q,
        uint32_t valid_rows,
        size_t valid_c) {
    HVX_Vector g_hf[4] __attribute__((aligned(128)));
    HVX_Vector u_hf[4] __attribute__((aligned(128)));
    if (valid_c > 32) {
        g_hf[0] = hmx_pack_f32_to_f16_clamped(g_q[0], g_q[4]);
        g_hf[1] = hmx_pack_f32_to_f16_clamped(g_q[1], g_q[5]);
        g_hf[2] = hmx_pack_f32_to_f16_clamped(g_q[2], g_q[6]);
        g_hf[3] = hmx_pack_f32_to_f16_clamped(g_q[3], g_q[7]);
        u_hf[0] = hmx_pack_f32_to_f16_clamped(u_q[0], u_q[4]);
        u_hf[1] = hmx_pack_f32_to_f16_clamped(u_q[1], u_q[5]);
        u_hf[2] = hmx_pack_f32_to_f16_clamped(u_q[2], u_q[6]);
        u_hf[3] = hmx_pack_f32_to_f16_clamped(u_q[3], u_q[7]);
    } else {
        const HVX_Vector z = Q6_V_vzero();
        g_hf[0] = hmx_pack_f32_to_f16_clamped(g_q[0], z);
        g_hf[1] = hmx_pack_f32_to_f16_clamped(g_q[1], z);
        g_hf[2] = hmx_pack_f32_to_f16_clamped(g_q[2], z);
        g_hf[3] = hmx_pack_f32_to_f16_clamped(g_q[3], z);
        u_hf[0] = hmx_pack_f32_to_f16_clamped(u_q[0], z);
        u_hf[1] = hmx_pack_f32_to_f16_clamped(u_q[1], z);
        u_hf[2] = hmx_pack_f32_to_f16_clamped(u_q[2], z);
        u_hf[3] = hmx_pack_f32_to_f16_clamped(u_q[3], z);
    }
    const uint32_t valid_bytes = (uint32_t) (valid_c * sizeof(__fp16));
    st->store_tail_fn(dst, dst_stride, g_hf, u_hf, (valid_rows << 16) | valid_bytes);
}

#define W8A8_GLU_EVAL_TILE_1G(tp_g, tp_u, dw_g, dw_u, out_g, out_u)              \
    w8a8_sm_glu_eval_1g_quad((tp_g), (dw_g), da0, (out_g));                      \
    w8a8_sm_glu_eval_1g_quad((tp_u), (dw_u), da0, (out_u));

#define W8A8_GLU_EVAL_TILE_2G(tp_g, tp_u, dw_g, dw_u, out_g, out_u)                         \
    w8a8_sm_glu_eval_2g_quad((tp_g), (tp_g) + group_stride, (dw_g), da0, da1, (out_g));     \
    w8a8_sm_glu_eval_2g_quad((tp_u), (tp_u) + group_stride, (dw_u), da0, da1, (out_u));

#define W8A8_GLU_EVAL_TILE_NG(tp_g, tp_u, dw_g, dw_u, out_g, out_u)                              \
    w8a8_sm_glu_eval_ng_quad((tp_g), (dw_g), group_stride, n_groups, da_quad_base, (out_g));     \
    w8a8_sm_glu_eval_ng_quad((tp_u), (dw_u), group_stride, n_groups, da_quad_base, (out_u));

#define W8A8_GLU_EVAL_TILE_CHK(tp_g, tp_u, dw_g, dw_u, out_g, out_u)             \
    w8a8_sm_glu_eval_chunked_quad((tp_g), (dw_g), &cargs, (out_g));              \
    w8a8_sm_glu_eval_chunked_quad((tp_u), (dw_u), &cargs, (out_u));

#define W8A8_GLU_EVAL_PAIR_1G(tp_g, tp_u, dw_g, dw_u, out_g, out_u)              \
    w8a8_sm_glu_eval_1g_pair_hf((tp_g), (dw_g), da0, (out_g));                   \
    w8a8_sm_glu_eval_1g_pair_hf((tp_u), (dw_u), da0, (out_u));

#define W8A8_GLU_EVAL_PAIR_2G(tp_g, tp_u, dw_g, dw_u, out_g, out_u)                            \
    w8a8_sm_glu_eval_2g_pair_hf((tp_g), (tp_g) + group_stride, (dw_g), da0, da1, (out_g));     \
    w8a8_sm_glu_eval_2g_pair_hf((tp_u), (tp_u) + group_stride, (dw_u), da0, da1, (out_u));

#define W8A8_GLU_EVAL_STRIP_1G(tp_g, tp_u, dw_g, dw_u, dst_g, dst_s, dst_u, u_s, n_p)                         \
    w8a8_sm_glu_eval_1g_strip_hf((tp_g), (dw_g), da0, (dst_g), (dst_s), (n_p));                               \
    w8a8_sm_glu_eval_1g_strip_hf((tp_u), (dw_u), da0, (dst_u), (u_s), (n_p));

#define W8A8_GLU_EVAL_STRIP_2G(tp_g, tp_u, dw_g, dw_u, dst_g, dst_s, dst_u, u_s, n_p)                         \
    w8a8_sm_glu_eval_2g_strip_hf((tp_g), (tp_g) + group_stride, (dw_g), da0, da1, (dst_g), (dst_s), (n_p));   \
    w8a8_sm_glu_eval_2g_strip_hf((tp_u), (tp_u) + group_stride, (dw_u), da0, da1, (dst_u), (u_s), (n_p));

#define W8A8_GLU_TAIL_FP16(EVAL_TILE, DW_STRIDE_ELEMS)                                                     \
    if (limit_c_aligned < limit_c) {                                                                       \
        HVX_Vector   g_q[8] __attribute__((aligned(128)));                                                 \
        HVX_Vector   u_q[8] __attribute__((aligned(128)));                                                 \
        const size_t c       = limit_c_aligned;                                                            \
        const size_t c0      = c >> 5;                                                                     \
        const size_t off0    = c0 * 2 * HTP_MM_HMX_TILE_SIZE;                                              \
        const size_t dw0     = c0 * (DW_STRIDE_ELEMS);                                                     \
        const size_t valid_c = limit_c - c;                                                                \
        EVAL_TILE(g_base + off0, u_base + off0, dw_g_base + dw0, dw_u_base + dw0, &g_q[0], &u_q[0])        \
        if (valid_c > 32) {                                                                                \
            const size_t off1 = off0 + 2 * HTP_MM_HMX_TILE_SIZE;                                           \
            const size_t dw1  = dw0 + (DW_STRIDE_ELEMS);                                                   \
            EVAL_TILE(g_base + off1, u_base + off1, dw_g_base + dw1, dw_u_base + dw1, &g_q[4], &u_q[4])    \
        }                                                                                                  \
        w8a8_sm_glu_pack_and_store_tail_fp16(st, out_row0 + c, dst_stride, g_q, u_q, valid_rows, valid_c); \
    }

#define W8A8_GLU_LOOP_FP16_PAIR(EVAL_STRIP, EVAL_PAIR, EVAL_TILE, DW_STRIDE_ELEMS)                               \
    {                                                                                                            \
        const uint32_t valid_rows = (uint32_t) hex_smin((size_t) 4u, (size_t) n_rows - r);                       \
        if (valid_rows == 4u) {                                                                                  \
            __fp16 u_buf[4 * 256] __attribute__((aligned(128)));                                                 \
            _Pragma("clang loop unroll(disable)")                                                                \
            for (size_t c = 0; c < limit_c_aligned; c += 256) {                                                  \
                const size_t n_pairs = hex_smin((limit_c_aligned - c) >> 6, (size_t) 4u);                        \
                const size_t c0      = c >> 5;                                                                   \
                const size_t off0    = c0 * 2 * HTP_MM_HMX_TILE_SIZE;                                            \
                const size_t dw0     = c0 * (DW_STRIDE_ELEMS);                                                   \
                EVAL_STRIP(g_base + off0, u_base + off0, dw_g_base + dw0, dw_u_base + dw0,                       \
                           out_row0 + c, dst_stride, u_buf, 256, n_pairs)                                        \
                st->store_strip_fn(out_row0 + c, dst_stride, u_buf, 256, n_pairs);                               \
            }                                                                                                    \
        } else {                                                                                                 \
            HVX_Vector g_hf[4] __attribute__((aligned(128)));                                                    \
            HVX_Vector u_hf[4] __attribute__((aligned(128)));                                                    \
            _Pragma("clang loop unroll(disable)")                                                                \
            for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                   \
                const size_t c0   = c >> 5;                                                                      \
                const size_t off0 = c0 * 2 * HTP_MM_HMX_TILE_SIZE;                                               \
                const size_t dw0  = c0 * (DW_STRIDE_ELEMS);                                                      \
                EVAL_PAIR(g_base + off0, u_base + off0, dw_g_base + dw0, dw_u_base + dw0, g_hf, u_hf)            \
                st->store_quad_fn(out_row0 + c, dst_stride, g_hf, u_hf, valid_rows);                             \
            }                                                                                                    \
        }                                                                                                        \
        W8A8_GLU_TAIL_FP16(EVAL_TILE, DW_STRIDE_ELEMS)                                                           \
    }

#define W8A8_GLU_LOOP_FP16_GEN(EVAL_TILE, DW_STRIDE_ELEMS)                                                 \
    {                                                                                                      \
        const uint32_t valid_rows = (uint32_t) hex_smin((size_t) 4u, (size_t) n_rows - r);                 \
        HVX_Vector     g_q[8]  __attribute__((aligned(128)));                                              \
        HVX_Vector     u_q[8]  __attribute__((aligned(128)));                                              \
        HVX_Vector     g_hf[4] __attribute__((aligned(128)));                                              \
        HVX_Vector     u_hf[4] __attribute__((aligned(128)));                                              \
        _Pragma("clang loop unroll(disable)")                                                              \
        for (size_t c = 0; c < limit_c_aligned; c += 64) {                                                 \
            const size_t c0   = c >> 5;                                                                    \
            const size_t off0 = c0 * 2 * HTP_MM_HMX_TILE_SIZE;                                             \
            const size_t off1 = off0 + 2 * HTP_MM_HMX_TILE_SIZE;                                           \
            const size_t dw0  = c0 * (DW_STRIDE_ELEMS);                                                    \
            const size_t dw1  = dw0 + (DW_STRIDE_ELEMS);                                                   \
            EVAL_TILE(g_base + off0, u_base + off0, dw_g_base + dw0, dw_u_base + dw0, &g_q[0], &u_q[0])    \
            EVAL_TILE(g_base + off1, u_base + off1, dw_g_base + dw1, dw_u_base + dw1, &g_q[4], &u_q[4])    \
            w8a8_sm_glu_pack_quad_hf(g_q, g_hf);                                                           \
            w8a8_sm_glu_pack_quad_hf(u_q, u_hf);                                                           \
            st->store_quad_fn(out_row0 + c, dst_stride, g_hf, u_hf, valid_rows);                           \
        }                                                                                                  \
        W8A8_GLU_TAIL_FP16(EVAL_TILE, DW_STRIDE_ELEMS)                                                     \
    }

#define W8A8_GLU_LOOP_FP32(EVAL_TILE, DW_STRIDE_ELEMS)                                                     \
    {                                                                                                      \
        const uint32_t valid_rows = (uint32_t) hex_smin((size_t) 4u, (size_t) n_rows - r);                 \
        HVX_Vector     g_q[4] __attribute__((aligned(128)));                                               \
        HVX_Vector     u_q[4] __attribute__((aligned(128)));                                               \
        _Pragma("clang loop unroll(disable)")                                                              \
        for (size_t c = 0; c < limit_c_aligned; c += 32) {                                                 \
            const size_t c0   = c >> 5;                                                                    \
            const size_t off0 = c0 * 2 * HTP_MM_HMX_TILE_SIZE;                                             \
            const size_t dw0  = c0 * (DW_STRIDE_ELEMS);                                                    \
            EVAL_TILE(g_base + off0, u_base + off0, dw_g_base + dw0, dw_u_base + dw0, &g_q[0], &u_q[0])    \
            st->store_quad_fn(out_row0 + c, dst_stride, g_q, u_q, valid_rows);                             \
        }                                                                                                  \
        if (limit_c_aligned < limit_c) {                                                                   \
            const size_t   c           = limit_c_aligned;                                                  \
            const size_t   c0          = c >> 5;                                                           \
            const size_t   off0        = c0 * 2 * HTP_MM_HMX_TILE_SIZE;                                    \
            const size_t   dw0         = c0 * (DW_STRIDE_ELEMS);                                           \
            const uint32_t valid_bytes = (uint32_t) ((limit_c - c) * sizeof(float));                       \
            EVAL_TILE(g_base + off0, u_base + off0, dw_g_base + dw0, dw_u_base + dw0, &g_q[0], &u_q[0])    \
            st->store_tail_fn(out_row0 + c, dst_stride, g_q, u_q, (valid_rows << 16) | valid_bytes);       \
        }                                                                                                  \
    }

#define W8A8_GLU_LOOP_FP16_1G  W8A8_GLU_LOOP_FP16_PAIR(W8A8_GLU_EVAL_STRIP_1G, W8A8_GLU_EVAL_PAIR_1G, W8A8_GLU_EVAL_TILE_1G, 32)
#define W8A8_GLU_LOOP_FP16_2G  W8A8_GLU_LOOP_FP16_PAIR(W8A8_GLU_EVAL_STRIP_2G, W8A8_GLU_EVAL_PAIR_2G, W8A8_GLU_EVAL_TILE_2G, 64)
#define W8A8_GLU_LOOP_FP16_NG  W8A8_GLU_LOOP_FP16_GEN(W8A8_GLU_EVAL_TILE_NG,  dw_tile_elems)
#define W8A8_GLU_LOOP_FP16_CHK W8A8_GLU_LOOP_FP16_GEN(W8A8_GLU_EVAL_TILE_CHK, dw_tile_elems)

#define W8A8_GLU_LOOP_FP32_1G  W8A8_GLU_LOOP_FP32(W8A8_GLU_EVAL_TILE_1G,  32)
#define W8A8_GLU_LOOP_FP32_2G  W8A8_GLU_LOOP_FP32(W8A8_GLU_EVAL_TILE_2G,  64)
#define W8A8_GLU_LOOP_FP32_NG  W8A8_GLU_LOOP_FP32(W8A8_GLU_EVAL_TILE_NG,  dw_tile_elems)
#define W8A8_GLU_LOOP_FP32_CHK W8A8_GLU_LOOP_FP32(W8A8_GLU_EVAL_TILE_CHK, dw_tile_elems)

#define DEFINE_W8A8_GLU_COL_CHUNKS(DST_SFX, DST_T, ALIGN_MASK, LOOP_1G, LOOP_2G, LOOP_NG, LOOP_CHK)                \
static __attribute__((noinline)) void transfer_output_chunk_w8a8_sm_glu_##DST_SFX##_1g(                            \
        const void * st_void, void * restrict dst_ptr,                                                             \
        uint32_t start_row, uint32_t n_rows, uint32_t c_first, uint32_t c_len_and_dst_cols) {                      \
    const hmx_output_glu_state_t * st = (const hmx_output_glu_state_t *) st_void;                                  \
    DST_T * restrict dst              = (DST_T *) dst_ptr;                                                         \
    const size_t total_col_tiles      = hmx_ceil_div(st->n_cols, 32);                                              \
    const size_t row_tile_stride      = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                                \
    const size_t c0_first             = c_first >> 5;                                                              \
    const size_t limit_c              = hex_smin(c_len_and_dst_cols >> 16, c_len_and_dst_cols & 0xffffu);          \
    const size_t limit_c_aligned      = limit_c & (ALIGN_MASK);                                                    \
    const size_t dst_stride           = st->dst_stride;                                                            \
    const float * dw_g_base           = st->vtcm_wt_scales_gate + c0_first * 32;                                   \
    const float * dw_u_base           = st->vtcm_wt_scales_up + c0_first * 32;                                     \
    uint32_t da_buf[1][32] __attribute__((aligned(128)));                                                          \
    for (size_t r_base = 0; r_base < n_rows; ) {                                                                   \
        const size_t r_idx_base = start_row + r_base;                                                              \
        const size_t r_blk      = r_idx_base >> 5;                                                                 \
        const size_t r0         = r_idx_base >> 6;                                                                 \
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));         \
        const size_t r_end      = r_base + r_step;                                                                 \
        uint32_t (*da_ptr)[32]  = stage_act_scales_to_l1d(da_buf, st->vtcm_act_scales + (r_blk << 5), 0, 1);       \
        for (size_t r = r_base; r < r_end; r += 4) {                                                               \
            const size_t     r_idx0   = start_row + r;                                                             \
            const size_t     rg       = (r_idx0 & 63u) >> 2;                                                       \
            const size_t     lane     = r_idx0 & 31u;                                                              \
            const uint32_t * da0      = &da_ptr[0][lane];                                                          \
            const size_t     tile_off = r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;     \
            const uint8_t *  g_base   = st->vtcm_gate + tile_off;                                                  \
            const uint8_t *  u_base   = st->vtcm_up + tile_off;                                                    \
            DST_T *          out_row0 = dst + r * dst_stride;                                                      \
            LOOP_1G                                                                                                \
        }                                                                                                          \
        r_base = r_end;                                                                                            \
    }                                                                                                              \
}                                                                                                                  \
static __attribute__((noinline)) void transfer_output_chunk_w8a8_sm_glu_##DST_SFX##_2g(                            \
        const void * st_void, void * restrict dst_ptr,                                                             \
        uint32_t start_row, uint32_t n_rows, uint32_t c_first, uint32_t c_len_and_dst_cols) {                      \
    const hmx_output_glu_state_t * st = (const hmx_output_glu_state_t *) st_void;                                  \
    DST_T * restrict dst              = (DST_T *) dst_ptr;                                                         \
    const size_t total_col_tiles      = hmx_ceil_div(st->n_cols, 32);                                              \
    const size_t n_row_tiles_64       = st->mc_pad >> 6;                                                           \
    const size_t row_tile_stride      = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                                \
    const size_t group_stride         = n_row_tiles_64 * row_tile_stride;                                          \
    const size_t c0_first             = c_first >> 5;                                                              \
    const size_t limit_c              = hex_smin(c_len_and_dst_cols >> 16, c_len_and_dst_cols & 0xffffu);          \
    const size_t limit_c_aligned      = limit_c & (ALIGN_MASK);                                                    \
    const size_t dst_stride           = st->dst_stride;                                                            \
    const float * dw_g_base           = st->vtcm_wt_scales_gate + c0_first * 64;                                   \
    const float * dw_u_base           = st->vtcm_wt_scales_up + c0_first * 64;                                     \
    uint32_t da_buf[2][32] __attribute__((aligned(128)));                                                          \
    for (size_t r_base = 0; r_base < n_rows; ) {                                                                   \
        const size_t r_idx_base = start_row + r_base;                                                              \
        const size_t r_blk      = r_idx_base >> 5;                                                                 \
        const size_t r0         = r_idx_base >> 6;                                                                 \
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));         \
        const size_t r_end      = r_base + r_step;                                                                 \
        const float * sc_base   = st->vtcm_act_scales + (r_blk << 5);                                              \
        uint32_t (*da_ptr)[32]  = stage_act_scales_to_l1d(da_buf, sc_base, st->mc_pad, 2);                         \
        for (size_t r = r_base; r < r_end; r += 4) {                                                               \
            const size_t     r_idx0   = start_row + r;                                                             \
            const size_t     rg       = (r_idx0 & 63u) >> 2;                                                       \
            const size_t     lane     = r_idx0 & 31u;                                                              \
            const uint32_t * da0      = &da_ptr[0][lane];                                                          \
            const uint32_t * da1      = &da_ptr[1][lane];                                                          \
            const size_t     tile_off = r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;     \
            const uint8_t *  g_base   = st->vtcm_gate + tile_off;                                                  \
            const uint8_t *  u_base   = st->vtcm_up + tile_off;                                                    \
            DST_T *          out_row0 = dst + r * dst_stride;                                                      \
            LOOP_2G                                                                                                \
        }                                                                                                          \
        r_base = r_end;                                                                                            \
    }                                                                                                              \
}                                                                                                                  \
static __attribute__((noinline)) void transfer_output_chunk_w8a8_sm_glu_##DST_SFX##_ng(                            \
        const void * st_void, void * restrict dst_ptr,                                                             \
        uint32_t start_row, uint32_t n_rows, uint32_t c_first, uint32_t c_len_and_dst_cols) {                      \
    const hmx_output_glu_state_t * st = (const hmx_output_glu_state_t *) st_void;                                  \
    DST_T * restrict dst              = (DST_T *) dst_ptr;                                                         \
    const uint32_t n_groups           = st->n_groups;                                                              \
    const size_t total_col_tiles      = hmx_ceil_div(st->n_cols, 32);                                              \
    const size_t n_row_tiles_64       = st->mc_pad >> 6;                                                           \
    const size_t row_tile_stride      = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                                \
    const size_t group_stride         = n_row_tiles_64 * row_tile_stride;                                          \
    const size_t dw_tile_elems        = (size_t) n_groups * 32;                                                    \
    const size_t c0_first             = c_first >> 5;                                                              \
    const size_t limit_c              = hex_smin(c_len_and_dst_cols >> 16, c_len_and_dst_cols & 0xffffu);          \
    const size_t limit_c_aligned      = limit_c & (ALIGN_MASK);                                                    \
    const size_t dst_stride           = st->dst_stride;                                                            \
    const float * dw_g_base           = st->vtcm_wt_scales_gate + c0_first * dw_tile_elems;                        \
    const float * dw_u_base           = st->vtcm_wt_scales_up + c0_first * dw_tile_elems;                          \
    uint32_t da_buf[16][32] __attribute__((aligned(128)));                                                         \
    for (size_t r_base = 0; r_base < n_rows; ) {                                                                   \
        const size_t r_idx_base = start_row + r_base;                                                              \
        const size_t r_blk      = r_idx_base >> 5;                                                                 \
        const size_t r0         = r_idx_base >> 6;                                                                 \
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));         \
        const size_t r_end      = r_base + r_step;                                                                 \
        const float * sc_base   = st->vtcm_act_scales + (r_blk << 5);                                              \
        uint32_t (*da_ptr)[32]  = stage_act_scales_to_l1d(da_buf, sc_base, st->mc_pad, n_groups);                  \
        for (size_t r = r_base; r < r_end; r += 4) {                                                               \
            const size_t     r_idx0       = start_row + r;                                                         \
            const size_t     rg           = (r_idx0 & 63u) >> 2;                                                   \
            const size_t     lane         = r_idx0 & 31u;                                                          \
            const uint32_t * da_quad_base = &da_ptr[0][lane];                                                      \
            const size_t     tile_off     = r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128; \
            const uint8_t *  g_base       = st->vtcm_gate + tile_off;                                              \
            const uint8_t *  u_base       = st->vtcm_up + tile_off;                                                \
            DST_T *          out_row0     = dst + r * dst_stride;                                                  \
            LOOP_NG                                                                                                \
        }                                                                                                          \
        r_base = r_end;                                                                                            \
    }                                                                                                              \
}                                                                                                                  \
static __attribute__((noinline)) void transfer_output_chunk_w8a8_sm_glu_##DST_SFX##_chunked(                       \
        const void * st_void, void * restrict dst_ptr,                                                             \
        uint32_t start_row, uint32_t n_rows, uint32_t c_first, uint32_t c_len_and_dst_cols) {                      \
    const hmx_output_glu_state_t * st = (const hmx_output_glu_state_t *) st_void;                                  \
    DST_T * restrict dst              = (DST_T *) dst_ptr;                                                         \
    const uint32_t n_groups           = st->n_groups;                                                              \
    const size_t total_col_tiles      = hmx_ceil_div(st->n_cols, 32);                                              \
    const size_t n_row_tiles_64       = st->mc_pad >> 6;                                                           \
    const size_t row_tile_stride      = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;                                \
    const size_t group_stride         = n_row_tiles_64 * row_tile_stride;                                          \
    const size_t dw_tile_elems        = (size_t) n_groups * 32;                                                    \
    const size_t c0_first             = c_first >> 5;                                                              \
    const size_t limit_c              = hex_smin(c_len_and_dst_cols >> 16, c_len_and_dst_cols & 0xffffu);          \
    const size_t limit_c_aligned      = limit_c & (ALIGN_MASK);                                                    \
    const size_t dst_stride           = st->dst_stride;                                                            \
    const float * dw_g_base           = st->vtcm_wt_scales_gate + c0_first * dw_tile_elems;                        \
    const float * dw_u_base           = st->vtcm_wt_scales_up + c0_first * dw_tile_elems;                          \
    uint32_t da_buf[16][32] __attribute__((aligned(128)));                                                         \
    for (size_t r_base = 0; r_base < n_rows; ) {                                                                   \
        const size_t r_idx_base = start_row + r_base;                                                              \
        const size_t r_blk      = r_idx_base >> 5;                                                                 \
        const size_t r0         = r_idx_base >> 6;                                                                 \
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));         \
        const size_t r_end      = r_base + r_step;                                                                 \
        w8a8_sm_chunked_eval_args_t cargs;                                                                         \
        cargs.act_scales_blk_base = st->vtcm_act_scales + (r_blk << 5);                                            \
        cargs.da_ptr              = da_buf;                                                                        \
        cargs.group_stride        = group_stride;                                                                  \
        cargs.mc_pad              = st->mc_pad;                                                                    \
        cargs.n_groups            = n_groups;                                                                      \
        for (size_t r = r_base; r < r_end; r += 4) {                                                               \
            const size_t    r_idx0   = start_row + r;                                                              \
            const size_t    rg       = (r_idx0 & 63u) >> 2;                                                        \
            cargs.lane               = (uint32_t) (r_idx0 & 31u);                                                  \
            const size_t    tile_off = r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;      \
            const uint8_t * g_base   = st->vtcm_gate + tile_off;                                                   \
            const uint8_t * u_base   = st->vtcm_up + tile_off;                                                     \
            DST_T *         out_row0 = dst + r * dst_stride;                                                       \
            LOOP_CHK                                                                                               \
        }                                                                                                          \
        r_base = r_end;                                                                                            \
    }                                                                                                              \
}

DEFINE_W8A8_GLU_COL_CHUNKS(fp16, __fp16, ~63u, W8A8_GLU_LOOP_FP16_1G, W8A8_GLU_LOOP_FP16_2G, W8A8_GLU_LOOP_FP16_NG, W8A8_GLU_LOOP_FP16_CHK)
DEFINE_W8A8_GLU_COL_CHUNKS(fp32, float,  ~31u, W8A8_GLU_LOOP_FP32_1G, W8A8_GLU_LOOP_FP32_2G, W8A8_GLU_LOOP_FP32_NG, W8A8_GLU_LOOP_FP32_CHK)

static inline bool resolve_int_output_glu_fns(
        uint32_t n_groups,
        uint8_t glu_op,
        size_t dst_elem_size,
        hmx_output_glu_col_chunk_fn_t * col_chunk_fn,
        hmx_glu_store_quad_fn_t * store_quad_fn,
        hmx_glu_store_tail_fn_t * store_tail_fn,
        hmx_glu_store_strip_fn_t * store_strip_fn) {
    if (!resolve_hmx_glu_store_fns(glu_op, dst_elem_size, store_quad_fn, store_tail_fn, store_strip_fn)) {
        return false;
    }
    if (dst_elem_size == sizeof(__fp16)) {
        if (n_groups <= 1u)       *col_chunk_fn = transfer_output_chunk_w8a8_sm_glu_fp16_1g;
        else if (n_groups == 2u)  *col_chunk_fn = transfer_output_chunk_w8a8_sm_glu_fp16_2g;
        else if (n_groups <= 16u) *col_chunk_fn = transfer_output_chunk_w8a8_sm_glu_fp16_ng;
        else                      *col_chunk_fn = transfer_output_chunk_w8a8_sm_glu_fp16_chunked;
    } else {
        if (n_groups <= 1u)       *col_chunk_fn = transfer_output_chunk_w8a8_sm_glu_fp32_1g;
        else if (n_groups == 2u)  *col_chunk_fn = transfer_output_chunk_w8a8_sm_glu_fp32_2g;
        else if (n_groups <= 16u) *col_chunk_fn = transfer_output_chunk_w8a8_sm_glu_fp32_ng;
        else                      *col_chunk_fn = transfer_output_chunk_w8a8_sm_glu_fp32_chunked;
    }
    return true;
}

