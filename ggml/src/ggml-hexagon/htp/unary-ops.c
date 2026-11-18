#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "hex-fastdiv.h"
#include "hvx-exp.h"
#include "hvx-sigmoid.h"
#include "hvx-utils.h"
#include "unary-ops.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "htp-trace.h"

struct htp_unary_context;

typedef void (*unary_compute_fn_t)(const void * restrict src,
                                   void * restrict dst,
                                   uint32_t num_rows,
                                   uint32_t ne0,
                                   const struct htp_unary_context * uctx);

typedef void (*unary_rms_norm_mul_compute_fn_t)(const void * restrict src,
                                                const void * restrict weight,
                                                void * restrict dst,
                                                uint32_t num_rows,
                                                const struct htp_unary_context * uctx);

typedef void (*unary_rms_norm_mul_add_compute_fn_t)(const void * restrict src,
                                                    const void * restrict weight,
                                                    const void * restrict add,
                                                    void * restrict dst,
                                                    uint32_t num_rows,
                                                    const struct htp_unary_context * uctx);

typedef void (*unary_tri_compute_fn_t)(const void * restrict src,
                                       void * restrict dst,
                                       uint32_t num_rows,
                                       uint32_t ir,
                                       const struct htp_unary_context * uctx);

typedef void (*unary_tiled_tri_compute_fn_t)(const void * restrict src,
                                             void * restrict dst,
                                             uint32_t tile_elems,
                                             uint32_t col_start,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

typedef HVX_Vector (*rms_row_scale_fn_t)(const uint8_t * restrict src,
                                         uint32_t num_elems,
                                         float epsilon,
                                         float scale);

typedef float (*rms_tile_sum_sq_fn_t)(const uint8_t * restrict src,
                                      uint32_t num_elems);

typedef void (*rms_apply_row_fn_t)(const uint8_t * restrict src,
                                   const uint8_t * restrict weight,
                                   const uint8_t * restrict add,
                                   uint8_t * restrict dst,
                                   uint32_t n,
                                   HVX_Vector scale_v);

struct htp_unary_context {
    struct htp_ops_context * octx;
    const struct htp_unary_kernel_params * kparams;

    void *                    compute;
    rms_row_scale_fn_t        rms_row_scale_fn;
    rms_tile_sum_sq_fn_t      rms_tile_sum_sq_fn;
    rms_apply_row_fn_t        rms_apply_row_fn;

    dma_addr_t                data_src0;
    dma_addr_t                data_src1;            // weight/scale tensor for RMS_NORM_MUL
    dma_addr_t                data_src2;            // residual tensor for RMS_NORM_MUL_ADD
    dma_addr_t                data_dst;

    size_t                    src0_data_row_size;   // actual data bytes per row
    size_t                    src1_data_row_size;
    size_t                    src2_data_row_size;
    size_t                    dst_data_row_size;    // actual data bytes per row

    size_t                    src0_row_size_aligned;
    size_t                    src1_row_size_aligned;
    size_t                    src2_row_size_aligned;
    size_t                    dst_row_size_aligned;

    size_t                    src0_vtcm_half_size;
    size_t                    src1_vtcm_half_size;
    size_t                    src2_vtcm_half_size;
    size_t                    dst_vtcm_half_size;

    uint32_t                  block;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_per_hvx;
    uint32_t                  row_start;
    uint32_t                  nc;
    uint32_t                  col_tile;             // tiled mode
    bool                      broadcast_weight;
    bool                      src0_f16;
    bool                      src1_f16;
    bool                      src2_f16;
    bool                      dst_f16;

    uint8_t *                 vtcm_src0;
    uint8_t *                 vtcm_src1;
    uint8_t *                 vtcm_src2;
    uint8_t *                 vtcm_dst;

    size_t                    vtcm_src0_size_per_hvx;
    size_t                    vtcm_src1_size_per_hvx;
    size_t                    vtcm_src2_size_per_hvx;
    size_t                    vtcm_dst_size_per_hvx;
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
static inline size_t unary_row_offset(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      const struct fastdiv_values * div_ne1,
                                      const struct fastdiv_values * div_ne2,
                                      const struct fastdiv_values * div_ne12,
                                      size_t nb1, size_t nb2, size_t nb3) {
    const uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    const uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    const uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    const uint32_t i3 = fastdiv(ir, div_ne12);
    return i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the dim-1 slice boundary
// when non-contiguous so the nb1 stride stays valid for all rows.
static inline uint32_t unary_block_size(uint32_t ir,
                                        uint32_t end_row,
                                        uint32_t block,
                                        bool contig,
                                        uint32_t ne1,
                                        const struct fastdiv_values * div_ne1) {
    uint32_t limit = MIN(block, end_row - ir);

    if (!contig) {
        const uint32_t slice_end = (fastdiv(ir, div_ne1) + 1) * ne1;
        limit = MIN(limit, slice_end - ir);
    }

    return limit;
}

#define htp_unary_preamble            \
    const uint32_t ne00 = src->ne[0]; \
    const uint32_t ne01 = src->ne[1]; \
    const uint32_t ne02 = src->ne[2]; \
    const uint32_t ne03 = src->ne[3]; \
                                      \
    const uint32_t ne0 = dst->ne[0];  \
    const uint32_t ne1 = dst->ne[1];  \
    const uint32_t ne2 = dst->ne[2];  \
    const uint32_t ne3 = dst->ne[3];  \
                                      \
    const uint32_t nb00 = src->nb[0]; \
    const uint32_t nb01 = src->nb[1]; \
    const uint32_t nb02 = src->nb[2]; \
    const uint32_t nb03 = src->nb[3]; \
                                      \
    const uint32_t nb0 = dst->nb[0];  \
    const uint32_t nb1 = dst->nb[1];  \
    const uint32_t nb2 = dst->nb[2];  \
    const uint32_t nb3 = dst->nb[3];

#define htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned; \
    const size_t dst_row_size_aligned = uctx->dst_row_size_aligned;

static inline HVX_Vector hvx_vec_clamp_f16_f32(HVX_Vector x) {
    const HVX_Vector f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector f16_min = hvx_vec_splat_f32(-65504.0f);
    return Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(x, f16_min), f16_max);
}

static void scale_f32(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const uint32_t ne0,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0, scale, bias);
    }
}

static void clamp_f32(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const uint32_t ne0,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float min = 0.f;
    float max = 0.f;
    memcpy(&min, &op_params[0], sizeof(float));
    memcpy(&max, &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f32_aa(dst_local, src_local, min, max, ne0);
    }
}

static inline HVX_Vector hvx_vec_leaky_relu_f32(HVX_Vector x, HVX_Vector slope, HVX_Vector zero) {
    HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(x, zero);
    HVX_Vector     neg  = hvx_vec_mul_f32_f32(x, slope);
    return Q6_V_vmux_QVV(pred, x, neg);
}

static inline void unary_leaky_relu_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, float slope, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const HVX_Vector vslope = hvx_vec_splat_f32(slope);
    const HVX_Vector vzero  = Q6_V_vzero();

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = hvx_vec_leaky_relu_f32(vsrc[i], vslope, vzero);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(float), hvx_vec_leaky_relu_f32(vsrc[nvec], vslope, vzero));
    }
}

static void leaky_relu_f32(const void * restrict src,
                           void * restrict dst,
                           const uint32_t num_rows,
                           const uint32_t ne0,
                           const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float negative_slope = 0.f;
    memcpy(&negative_slope, &op_params[0], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        unary_leaky_relu_f32_aa(dst_local, src_local, negative_slope, ne0);
    }
}

static inline HVX_Vector rms_norm_sum_sq_f32(const uint8_t * restrict src, uint32_t num_elems, HVX_Vector sum_v) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    const uint32_t nvec = num_elems / VLEN_FP32;
    const uint32_t nloe = num_elems % VLEN_FP32;

    #pragma unroll(4)
    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(v1, v1));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * sizeof(float));
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(v1, v1));
    }

    return sum_v;
}

static inline HVX_Vector rms_norm_sum_sq_f16(const uint8_t * restrict src, uint32_t num_elems, HVX_Vector sum_v) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    const uint32_t nvec = num_elems / VLEN_FP16;
    const uint32_t nloe = num_elems % VLEN_FP16;

#if __HVX_ARCH__ >= 79
    HVX_VectorPair sum_p = Q6_W_vzero();

    #pragma unroll(4)
    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = v_src[i];
        sum_p = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_p, v, v);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * sizeof(_Float16));
        HVX_Vector v = Q6_V_vand_QV(bmask, v_src[nvec]);
        sum_p = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_p, v, v);
    }

    sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sum_p), Q6_V_hi_W(sum_p)));
#else
    HVX_Vector sum0 = Q6_V_vzero();
    HVX_Vector sum1 = Q6_V_vzero();

    #pragma unroll(4)
    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = v_src[i];
        HVX_VectorPair sq = Q6_Wqf32_vmpy_VhfVhf(v, v);
        sum0 = Q6_Vqf32_vadd_Vqf32Vqf32(sum0, Q6_V_lo_W(sq));
        sum1 = Q6_Vqf32_vadd_Vqf32Vqf32(sum1, Q6_V_hi_W(sq));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * sizeof(_Float16));
        HVX_Vector v = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair sq = Q6_Wqf32_vmpy_VhfVhf(v, v);
        sum0 = Q6_Vqf32_vadd_Vqf32Vqf32(sum0, Q6_V_lo_W(sq));
        sum1 = Q6_Vqf32_vadd_Vqf32Vqf32(sum1, Q6_V_hi_W(sq));
    }

    sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vadd_Vqf32Vqf32(sum0, sum1));
#endif

    return sum_v;
}

static inline HVX_Vector rms_norm_scale_from_sum_f32(HVX_Vector sum_v, uint32_t num_elems, float epsilon, float scale) {
    HVX_Vector t_v            = hvx_vec_splat_f32((float) (int32_t) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, hvx_vec_splat_f32(epsilon));
    HVX_Vector scale_v        = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    if (scale != 0.0f) {
        scale_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(scale_v, hvx_vec_splat_f32(scale)));
    }

    return scale_v;
}

static __attribute__((noinline)) HVX_Vector rms_norm_row_scale_f32(const uint8_t * restrict src,
                                                                   uint32_t num_elems,
                                                                   float epsilon,
                                                                   float scale) {
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    sum_v = rms_norm_sum_sq_f32(src, num_elems, sum_v);
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));
    return rms_norm_scale_from_sum_f32(sum_v, num_elems, epsilon, scale);
}

static __attribute__((noinline)) HVX_Vector rms_norm_row_scale_f16(const uint8_t * restrict src,
                                                                   uint32_t num_elems,
                                                                   float epsilon,
                                                                   float scale) {
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    sum_v = rms_norm_sum_sq_f16(src, num_elems, sum_v);
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));
    return rms_norm_scale_from_sum_f32(sum_v, num_elems, epsilon, scale);
}

static __attribute__((noinline)) float rms_norm_tile_sum_sq_f32(const uint8_t * restrict src,
                                                                uint32_t num_elems) {
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    sum_v = rms_norm_sum_sq_f32(src, num_elems, sum_v);
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));
    return hvx_vec_get_f32(sum_v);
}

static __attribute__((noinline)) float rms_norm_tile_sum_sq_f16(const uint8_t * restrict src,
                                                                uint32_t num_elems) {
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    sum_v = rms_norm_sum_sq_f16(src, num_elems, sum_v);
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));
    return hvx_vec_get_f32(sum_v);
}

static inline HVX_VectorPair rms_load_pair_f32(const uint8_t * restrict ptr, uint32_t i, bool is_f16) {
    const HVX_Vector * restrict v_ptr = (const HVX_Vector *) ptr;
    if (is_f16) {
        return hvx_vec_f16_to_f32(v_ptr[i >> 6]);
    }
    return Q6_W_vcombine_VV(v_ptr[(i >> 5) + 1], v_ptr[i >> 5]);
}

static inline HVX_Vector rms_load_vec_f32(const uint8_t * restrict ptr, uint32_t i, bool is_f16) {
    const HVX_Vector * restrict v_ptr = (const HVX_Vector *) ptr;
    if (!is_f16) {
        return v_ptr[i >> 5];
    }
    HVX_VectorPair p = hvx_vec_f16_to_f32(v_ptr[i >> 6]);
    return (i & 32u) ? Q6_V_hi_W(p) : Q6_V_lo_W(p);
}

static inline void rms_store_pair_f32(uint8_t * restrict dst, uint32_t i, HVX_Vector r0, HVX_Vector r1, bool is_f16) {
    HVX_Vector * restrict v_dst = (HVX_Vector *) dst;
    if (is_f16) {
        r0 = hvx_vec_clamp_f16_f32(r0);
        r1 = hvx_vec_clamp_f16_f32(r1);
        v_dst[i >> 6] = hvx_vec_f32_to_f16(r0, r1);
    } else {
        v_dst[(i >> 5) + 0] = r0;
        v_dst[(i >> 5) + 1] = r1;
    }
}

static inline void rms_store_tail_f32(uint8_t * restrict dst, uint32_t i, uint32_t rem, HVX_Vector r0, HVX_Vector r1, bool is_f16) {
    HVX_Vector * restrict v_dst = (HVX_Vector *) dst;
    if (is_f16) {
        r0 = hvx_vec_clamp_f16_f32(r0);
        r1 = hvx_vec_clamp_f16_f32(r1);
        HVX_Vector packed = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a(&v_dst[i >> 6], rem * sizeof(_Float16), packed);
    } else if (rem <= 32) {
        hvx_vec_store_a(&v_dst[i >> 5], rem * sizeof(float), r0);
    } else {
        v_dst[(i >> 5) + 0] = r0;
        hvx_vec_store_a(&v_dst[(i >> 5) + 1], (rem - 32) * sizeof(float), r1);
    }
}

static inline HVX_Vector rms_clamp_pack_f16_shuff(HVX_Vector r0, HVX_Vector r1, HVX_Vector f16_min, HVX_Vector f16_max) {
    r0 = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(r0, f16_min), f16_max);
    r1 = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(r1, f16_min), f16_max);
    return hvx_vec_f32_to_f16_shuff(r0, r1);
}

static inline HVX_Vector rms_clamp_pack_f16(HVX_Vector r0, HVX_Vector r1, HVX_Vector f16_min, HVX_Vector f16_max) {
    return Q6_Vh_vdeal_Vh(rms_clamp_pack_f16_shuff(r0, r1, f16_min, f16_max));
}

static inline void rms_norm_apply_row_f16(const uint8_t * restrict src,
                                          uint8_t * restrict dst,
                                          uint32_t n,
                                          HVX_Vector scale_v) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    const HVX_Vector one_hf  = hvx_vec_splat_f16(1.0);
    const HVX_Vector f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector f16_min = hvx_vec_splat_f32(-65504.0f);
#if __HVX_ARCH__ < 79
    const HVX_Vector scale_qf = Q6_Vqf32_vadd_VsfVsf(scale_v, Q6_V_vzero());
#endif

    #pragma unroll(4)
    for (uint32_t k = 0; k < nvec; k++) {
#if __HVX_ARCH__ >= 79
        HVX_VectorPair sp = Q6_Wsf_vmpy_VhfVhf(v_src[k], one_hf);
        HVX_Vector     r0 = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sp), scale_v);
        HVX_Vector     r1 = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sp), scale_v);
#else
        HVX_VectorPair sp = Q6_Wqf32_vmpy_VhfVhf(v_src[k], one_hf);
        HVX_Vector     r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sp), scale_qf));
        HVX_Vector     r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sp), scale_qf));
#endif
        v_dst[k] = rms_clamp_pack_f16_shuff(r0, r1, f16_min, f16_max);
    }

    if (nloe > 0) {
#if __HVX_ARCH__ >= 79
        HVX_VectorPair sp = Q6_Wsf_vmpy_VhfVhf(v_src[nvec], one_hf);
        HVX_Vector     r0 = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sp), scale_v);
        HVX_Vector     r1 = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sp), scale_v);
#else
        HVX_VectorPair sp = Q6_Wqf32_vmpy_VhfVhf(v_src[nvec], one_hf);
        HVX_Vector     r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sp), scale_qf));
        HVX_Vector     r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sp), scale_qf));
#endif
        HVX_Vector packed = rms_clamp_pack_f16_shuff(r0, r1, f16_min, f16_max);
        hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(_Float16), packed);
    }
}

static inline void rms_norm_apply_row_mul_f16_f32(const uint8_t * restrict src,
                                                  const uint8_t * restrict weight,
                                                  uint8_t * restrict dst,
                                                  uint32_t n,
                                                  HVX_Vector scale_v) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    const HVX_Vector * restrict v_w   = (const HVX_Vector *) weight;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    const HVX_Vector one_hf  = hvx_vec_splat_f16(1.0);
    const HVX_Vector f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector f16_min = hvx_vec_splat_f32(-65504.0f);

    #pragma unroll(2)
    for (uint32_t k = 0; k < nvec; k++) {
#if __HVX_ARCH__ >= 79
        HVX_VectorPair sp  = Q6_Wsf_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_src[k]), one_hf);
        HVX_Vector     ws0 = Q6_Vsf_vmpy_VsfVsf(v_w[2 * k + 0], scale_v);
        HVX_Vector     ws1 = Q6_Vsf_vmpy_VsfVsf(v_w[2 * k + 1], scale_v);
        HVX_Vector     r0  = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sp), ws0);
        HVX_Vector     r1  = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sp), ws1);
#else
        HVX_VectorPair sp  = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_src[k]), one_hf);
        HVX_Vector     ws0 = Q6_Vqf32_vmpy_VsfVsf(v_w[2 * k + 0], scale_v);
        HVX_Vector     ws1 = Q6_Vqf32_vmpy_VsfVsf(v_w[2 * k + 1], scale_v);
        HVX_Vector     r0  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sp), ws0));
        HVX_Vector     r1  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sp), ws1));
#endif
        v_dst[k] = rms_clamp_pack_f16(r0, r1, f16_min, f16_max);
    }

    if (nloe > 0) {
#if __HVX_ARCH__ >= 79
        HVX_VectorPair sp  = Q6_Wsf_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_src[nvec]), one_hf);
        HVX_Vector     ws0 = Q6_Vsf_vmpy_VsfVsf(v_w[2 * nvec + 0], scale_v);
        HVX_Vector     r0  = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sp), ws0);
        HVX_Vector     r1  = Q6_V_vzero();
        if (nloe > 32) {
            HVX_Vector ws1 = Q6_Vsf_vmpy_VsfVsf(v_w[2 * nvec + 1], scale_v);
            r1 = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sp), ws1);
        }
#else
        HVX_VectorPair sp  = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_src[nvec]), one_hf);
        HVX_Vector     ws0 = Q6_Vqf32_vmpy_VsfVsf(v_w[2 * nvec + 0], scale_v);
        HVX_Vector     r0  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sp), ws0));
        HVX_Vector     r1  = Q6_V_vzero();
        if (nloe > 32) {
            HVX_Vector ws1 = Q6_Vqf32_vmpy_VsfVsf(v_w[2 * nvec + 1], scale_v);
            r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sp), ws1));
        }
#endif
        HVX_Vector packed = rms_clamp_pack_f16(r0, r1, f16_min, f16_max);
        hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(_Float16), packed);
    }
}

static inline void rms_norm_apply_row_mul_f16_f16(const uint8_t * restrict src,
                                                  const uint8_t * restrict weight,
                                                  uint8_t * restrict dst,
                                                  uint32_t n,
                                                  HVX_Vector scale_v) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    const HVX_Vector * restrict v_w   = (const HVX_Vector *) weight;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    const HVX_Vector f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector f16_min = hvx_vec_splat_f32(-65504.0f);
#if __HVX_ARCH__ < 79
    const HVX_Vector scale_qf = Q6_Vqf32_vadd_VsfVsf(scale_v, Q6_V_vzero());
#endif

    #pragma unroll(4)
    for (uint32_t k = 0; k < nvec; k++) {
#if __HVX_ARCH__ >= 79
        HVX_VectorPair sw = Q6_Wsf_vmpy_VhfVhf(v_src[k], v_w[k]);
        HVX_Vector     r0 = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sw), scale_v);
        HVX_Vector     r1 = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sw), scale_v);
#else
        HVX_VectorPair sw = Q6_Wqf32_vmpy_VhfVhf(v_src[k], v_w[k]);
        HVX_Vector     r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sw), scale_qf));
        HVX_Vector     r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sw), scale_qf));
#endif
        v_dst[k] = rms_clamp_pack_f16_shuff(r0, r1, f16_min, f16_max);
    }

    if (nloe > 0) {
#if __HVX_ARCH__ >= 79
        HVX_VectorPair sw = Q6_Wsf_vmpy_VhfVhf(v_src[nvec], v_w[nvec]);
        HVX_Vector     r0 = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sw), scale_v);
        HVX_Vector     r1 = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sw), scale_v);
#else
        HVX_VectorPair sw = Q6_Wqf32_vmpy_VhfVhf(v_src[nvec], v_w[nvec]);
        HVX_Vector     r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sw), scale_qf));
        HVX_Vector     r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sw), scale_qf));
#endif
        HVX_Vector packed = rms_clamp_pack_f16_shuff(r0, r1, f16_min, f16_max);
        hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(_Float16), packed);
    }
}

static inline void rms_norm_apply_row_mul_add_f16_f32_f16(const uint8_t * restrict src,
                                                          const uint8_t * restrict weight,
                                                          const uint8_t * restrict add,
                                                          uint8_t * restrict dst,
                                                          uint32_t n,
                                                          HVX_Vector scale_v) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    const HVX_Vector * restrict v_w   = (const HVX_Vector *) weight;
    const HVX_Vector * restrict v_add = (const HVX_Vector *) add;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    const HVX_Vector one_hf  = hvx_vec_splat_f16(1.0);
    const HVX_Vector f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector f16_min = hvx_vec_splat_f32(-65504.0f);

    #pragma unroll(2)
    for (uint32_t k = 0; k < nvec; k++) {
#if __HVX_ARCH__ >= 79
        HVX_VectorPair sp  = Q6_Wsf_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_src[k]), one_hf);
        HVX_VectorPair ap  = Q6_Wsf_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_add[k]), one_hf);
        HVX_Vector     ws0 = Q6_Vsf_vmpy_VsfVsf(v_w[2 * k + 0], scale_v);
        HVX_Vector     ws1 = Q6_Vsf_vmpy_VsfVsf(v_w[2 * k + 1], scale_v);
        HVX_Vector     r0  = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sp), ws0), Q6_V_lo_W(ap));
        HVX_Vector     r1  = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sp), ws1), Q6_V_hi_W(ap));
#else
        HVX_VectorPair sp  = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_src[k]), one_hf);
        HVX_VectorPair ap  = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_add[k]), one_hf);
        HVX_Vector     ws0 = Q6_Vqf32_vmpy_VsfVsf(v_w[2 * k + 0], scale_v);
        HVX_Vector     ws1 = Q6_Vqf32_vmpy_VsfVsf(v_w[2 * k + 1], scale_v);
        HVX_Vector     m0  = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sp), ws0);
        HVX_Vector     m1  = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sp), ws1);
        HVX_Vector     r0  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(m0, Q6_V_lo_W(ap)));
        HVX_Vector     r1  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(m1, Q6_V_hi_W(ap)));
#endif
        v_dst[k] = rms_clamp_pack_f16(r0, r1, f16_min, f16_max);
    }

    if (nloe > 0) {
#if __HVX_ARCH__ >= 79
        HVX_VectorPair sp  = Q6_Wsf_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_src[nvec]), one_hf);
        HVX_VectorPair ap  = Q6_Wsf_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_add[nvec]), one_hf);
        HVX_Vector     ws0 = Q6_Vsf_vmpy_VsfVsf(v_w[2 * nvec + 0], scale_v);
        HVX_Vector     r0  = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sp), ws0), Q6_V_lo_W(ap));
        HVX_Vector     r1  = Q6_V_vzero();
        if (nloe > 32) {
            HVX_Vector ws1 = Q6_Vsf_vmpy_VsfVsf(v_w[2 * nvec + 1], scale_v);
            r1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sp), ws1), Q6_V_hi_W(ap));
        }
#else
        HVX_VectorPair sp  = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_src[nvec]), one_hf);
        HVX_VectorPair ap  = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_add[nvec]), one_hf);
        HVX_Vector     ws0 = Q6_Vqf32_vmpy_VsfVsf(v_w[2 * nvec + 0], scale_v);
        HVX_Vector     m0  = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sp), ws0);
        HVX_Vector     r0  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(m0, Q6_V_lo_W(ap)));
        HVX_Vector     r1  = Q6_V_vzero();
        if (nloe > 32) {
            HVX_Vector ws1 = Q6_Vqf32_vmpy_VsfVsf(v_w[2 * nvec + 1], scale_v);
            HVX_Vector m1  = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sp), ws1);
            r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(m1, Q6_V_hi_W(ap)));
        }
#endif
        HVX_Vector packed = rms_clamp_pack_f16(r0, r1, f16_min, f16_max);
        hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(_Float16), packed);
    }
}

static inline void rms_norm_apply_row_mul_add_f16_f16_f16(const uint8_t * restrict src,
                                                          const uint8_t * restrict weight,
                                                          const uint8_t * restrict add,
                                                          uint8_t * restrict dst,
                                                          uint32_t n,
                                                          HVX_Vector scale_v) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    const HVX_Vector * restrict v_w   = (const HVX_Vector *) weight;
    const HVX_Vector * restrict v_add = (const HVX_Vector *) add;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    const HVX_Vector one_hf  = hvx_vec_splat_f16(1.0);
    const HVX_Vector f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector f16_min = hvx_vec_splat_f32(-65504.0f);
#if __HVX_ARCH__ < 79
    const HVX_Vector scale_qf = Q6_Vqf32_vadd_VsfVsf(scale_v, Q6_V_vzero());
#endif

    #pragma unroll(2)
    for (uint32_t k = 0; k < nvec; k++) {
#if __HVX_ARCH__ >= 79
        HVX_VectorPair sw = Q6_Wsf_vmpy_VhfVhf(v_src[k], v_w[k]);
        HVX_VectorPair ap = Q6_Wsf_vmpy_VhfVhf(v_add[k], one_hf);
        HVX_Vector     r0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sw), scale_v), Q6_V_lo_W(ap));
        HVX_Vector     r1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sw), scale_v), Q6_V_hi_W(ap));
#else
        HVX_VectorPair sw = Q6_Wqf32_vmpy_VhfVhf(v_src[k], v_w[k]);
        HVX_VectorPair ap = Q6_Wqf32_vmpy_VhfVhf(v_add[k], one_hf);
        HVX_Vector     m0 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sw), scale_qf);
        HVX_Vector     m1 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sw), scale_qf);
        HVX_Vector     r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(m0, Q6_V_lo_W(ap)));
        HVX_Vector     r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(m1, Q6_V_hi_W(ap)));
#endif
        v_dst[k] = rms_clamp_pack_f16_shuff(r0, r1, f16_min, f16_max);
    }

    if (nloe > 0) {
#if __HVX_ARCH__ >= 79
        HVX_VectorPair sw = Q6_Wsf_vmpy_VhfVhf(v_src[nvec], v_w[nvec]);
        HVX_VectorPair ap = Q6_Wsf_vmpy_VhfVhf(v_add[nvec], one_hf);
        HVX_Vector     r0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sw), scale_v), Q6_V_lo_W(ap));
        HVX_Vector     r1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sw), scale_v), Q6_V_hi_W(ap));
#else
        HVX_VectorPair sw = Q6_Wqf32_vmpy_VhfVhf(v_src[nvec], v_w[nvec]);
        HVX_VectorPair ap = Q6_Wqf32_vmpy_VhfVhf(v_add[nvec], one_hf);
        HVX_Vector     m0 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sw), scale_qf);
        HVX_Vector     m1 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sw), scale_qf);
        HVX_Vector     r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(m0, Q6_V_lo_W(ap)));
        HVX_Vector     r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(m1, Q6_V_hi_W(ap)));
#endif
        HVX_Vector packed = rms_clamp_pack_f16_shuff(r0, r1, f16_min, f16_max);
        hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(_Float16), packed);
    }
}

static __attribute__((noinline)) void rms_norm_apply_row_f16_f16(const uint8_t * restrict src,
                                                                 const uint8_t * restrict weight,
                                                                 const uint8_t * restrict add,
                                                                 uint8_t * restrict dst,
                                                                 uint32_t n,
                                                                 HVX_Vector scale_v) {
    (void) weight;
    (void) add;
    rms_norm_apply_row_f16(src, dst, n, scale_v);
}

static __attribute__((noinline)) void rms_norm_apply_row_mul_f16_f32_f16(const uint8_t * restrict src,
                                                                         const uint8_t * restrict weight,
                                                                         const uint8_t * restrict add,
                                                                         uint8_t * restrict dst,
                                                                         uint32_t n,
                                                                         HVX_Vector scale_v) {
    (void) add;
    rms_norm_apply_row_mul_f16_f32(src, weight, dst, n, scale_v);
}

static __attribute__((noinline)) void rms_norm_apply_row_mul_f16_f16_f16(const uint8_t * restrict src,
                                                                         const uint8_t * restrict weight,
                                                                         const uint8_t * restrict add,
                                                                         uint8_t * restrict dst,
                                                                         uint32_t n,
                                                                         HVX_Vector scale_v) {
    (void) add;
    rms_norm_apply_row_mul_f16_f16(src, weight, dst, n, scale_v);
}

static __attribute__((noinline)) void rms_norm_apply_row_mul_add_f16_f32_f16_f16(const uint8_t * restrict src,
                                                                                 const uint8_t * restrict weight,
                                                                                 const uint8_t * restrict add,
                                                                                 uint8_t * restrict dst,
                                                                                 uint32_t n,
                                                                                 HVX_Vector scale_v) {
    rms_norm_apply_row_mul_add_f16_f32_f16(src, weight, add, dst, n, scale_v);
}

static __attribute__((noinline)) void rms_norm_apply_row_mul_add_f16_f16_f16_f16(const uint8_t * restrict src,
                                                                                 const uint8_t * restrict weight,
                                                                                 const uint8_t * restrict add,
                                                                                 uint8_t * restrict dst,
                                                                                 uint32_t n,
                                                                                 HVX_Vector scale_v) {
    rms_norm_apply_row_mul_add_f16_f16_f16(src, weight, add, dst, n, scale_v);
}

#define DEFINE_RMS_NORM_APPLY_ROW(SFX, HAS_W, HAS_A, S0_F16, S1_F16, S2_F16, D_F16)                                    \
    static __attribute__((noinline)) void rms_norm_apply_row_##SFX(const uint8_t * restrict src,                       \
                                                                   const uint8_t * restrict weight,                    \
                                                                   const uint8_t * restrict add,                       \
                                                                   uint8_t * restrict dst,                             \
                                                                   uint32_t n,                                         \
                                                                   HVX_Vector scale_v) {                               \
        (void) weight;                                                                                                 \
        (void) add;                                                                                                    \
        uint32_t i = 0;                                                                                                \
        for (; i + 64 <= n; i += 64) {                                                                                 \
            HVX_VectorPair sp = rms_load_pair_f32(src, i, S0_F16);                                                     \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(sp), scale_v));                         \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(sp), scale_v));                         \
            if (HAS_W) {                                                                                               \
                HVX_VectorPair wp = rms_load_pair_f32(weight, i, S1_F16);                                              \
                r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0, Q6_V_lo_W(wp)));                                     \
                r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1, Q6_V_hi_W(wp)));                                     \
            }                                                                                                          \
            if (HAS_A) {                                                                                               \
                HVX_VectorPair ap = rms_load_pair_f32(add, i, S2_F16);                                                 \
                r0 = hvx_vec_add_f32_f32(r0, Q6_V_lo_W(ap));                                                           \
                r1 = hvx_vec_add_f32_f32(r1, Q6_V_hi_W(ap));                                                           \
            }                                                                                                          \
            rms_store_pair_f32(dst, i, r0, r1, D_F16);                                                                 \
        }                                                                                                              \
        if (i < n) {                                                                                                   \
            const uint32_t rem = n - i;                                                                                \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(rms_load_vec_f32(src, i, S0_F16), scale_v));      \
            HVX_Vector r1 = Q6_V_vzero();                                                                              \
            if (rem > 32) {                                                                                            \
                r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(rms_load_vec_f32(src, i + 32, S0_F16), scale_v));        \
            }                                                                                                          \
            if (HAS_W) {                                                                                               \
                r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0, rms_load_vec_f32(weight, i, S1_F16)));               \
                if (rem > 32) {                                                                                        \
                    r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1, rms_load_vec_f32(weight, i + 32, S1_F16)));      \
                }                                                                                                      \
            }                                                                                                          \
            if (HAS_A) {                                                                                               \
                r0 = hvx_vec_add_f32_f32(r0, rms_load_vec_f32(add, i, S2_F16));                                        \
                if (rem > 32) {                                                                                        \
                    r1 = hvx_vec_add_f32_f32(r1, rms_load_vec_f32(add, i + 32, S2_F16));                               \
                }                                                                                                      \
            }                                                                                                          \
            rms_store_tail_f32(dst, i, rem, r0, r1, D_F16);                                                            \
        }                                                                                                              \
    }

DEFINE_RMS_NORM_APPLY_ROW(f32_f32,                 false, false, false, false, false, false)
DEFINE_RMS_NORM_APPLY_ROW(f32_f16,                 false, false, false, false, false, true)
DEFINE_RMS_NORM_APPLY_ROW(f16_f32,                 false, false, true,  false, false, false)

DEFINE_RMS_NORM_APPLY_ROW(mul_f32_f32_f32,         true,  false, false, false, false, false)
DEFINE_RMS_NORM_APPLY_ROW(mul_f32_f16_f32,         true,  false, false, true,  false, false)
DEFINE_RMS_NORM_APPLY_ROW(mul_f32_f32_f16,         true,  false, false, false, false, true)
DEFINE_RMS_NORM_APPLY_ROW(mul_f32_f16_f16,         true,  false, false, true,  false, true)
DEFINE_RMS_NORM_APPLY_ROW(mul_f16_f32_f32,         true,  false, true,  false, false, false)
DEFINE_RMS_NORM_APPLY_ROW(mul_f16_f16_f32,         true,  false, true,  true,  false, false)

DEFINE_RMS_NORM_APPLY_ROW(mul_add_f32_f32_f32_f32, true,  true,  false, false, false, false)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f32_f32_f32_f16, true,  true,  false, false, false, true)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f32_f32_f16_f32, true,  true,  false, false, true,  false)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f32_f32_f16_f16, true,  true,  false, false, true,  true)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f32_f16_f32_f32, true,  true,  false, true,  false, false)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f32_f16_f32_f16, true,  true,  false, true,  false, true)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f32_f16_f16_f32, true,  true,  false, true,  true,  false)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f32_f16_f16_f16, true,  true,  false, true,  true,  true)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f16_f32_f32_f32, true,  true,  true,  false, false, false)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f16_f32_f32_f16, true,  true,  true,  false, false, true)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f16_f32_f16_f32, true,  true,  true,  false, true,  false)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f16_f16_f32_f32, true,  true,  true,  true,  false, false)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f16_f16_f32_f16, true,  true,  true,  true,  false, true)
DEFINE_RMS_NORM_APPLY_ROW(mul_add_f16_f16_f16_f32, true,  true,  true,  true,  true,  false)

#undef DEFINE_RMS_NORM_APPLY_ROW

static const rms_apply_row_fn_t rms_norm_apply_row_table[2][2] = {
    { rms_norm_apply_row_f32_f32, rms_norm_apply_row_f32_f16 },
    { rms_norm_apply_row_f16_f32, rms_norm_apply_row_f16_f16 },
};

static const rms_apply_row_fn_t rms_norm_mul_apply_row_table[2][2][2] = {
    {
        { rms_norm_apply_row_mul_f32_f32_f32, rms_norm_apply_row_mul_f32_f32_f16 },
        { rms_norm_apply_row_mul_f32_f16_f32, rms_norm_apply_row_mul_f32_f16_f16 },
    },
    {
        { rms_norm_apply_row_mul_f16_f32_f32, rms_norm_apply_row_mul_f16_f32_f16 },
        { rms_norm_apply_row_mul_f16_f16_f32, rms_norm_apply_row_mul_f16_f16_f16 },
    },
};

static const rms_apply_row_fn_t rms_norm_mul_add_apply_row_table[2][2][2][2] = {
    {
        {
            { rms_norm_apply_row_mul_add_f32_f32_f32_f32, rms_norm_apply_row_mul_add_f32_f32_f32_f16 },
            { rms_norm_apply_row_mul_add_f32_f32_f16_f32, rms_norm_apply_row_mul_add_f32_f32_f16_f16 },
        },
        {
            { rms_norm_apply_row_mul_add_f32_f16_f32_f32, rms_norm_apply_row_mul_add_f32_f16_f32_f16 },
            { rms_norm_apply_row_mul_add_f32_f16_f16_f32, rms_norm_apply_row_mul_add_f32_f16_f16_f16 },
        },
    },
    {
        {
            { rms_norm_apply_row_mul_add_f16_f32_f32_f32, rms_norm_apply_row_mul_add_f16_f32_f32_f16 },
            { rms_norm_apply_row_mul_add_f16_f32_f16_f32, rms_norm_apply_row_mul_add_f16_f32_f16_f16 },
        },
        {
            { rms_norm_apply_row_mul_add_f16_f16_f32_f32, rms_norm_apply_row_mul_add_f16_f16_f32_f16 },
            { rms_norm_apply_row_mul_add_f16_f16_f16_f32, rms_norm_apply_row_mul_add_f16_f16_f16_f16 },
        },
    },
};

static __attribute__((noinline)) void rms_norm_apply_tile(const uint8_t * restrict src,
                                                          const uint8_t * restrict weight,
                                                          const uint8_t * restrict add,
                                                          uint8_t * restrict dst,
                                                          uint32_t n,
                                                          uint32_t ne0,
                                                          float sum_sq,
                                                          float epsilon,
                                                          float scale,
                                                          const struct htp_unary_context * uctx) {
    const HVX_Vector scale_v = rms_norm_scale_from_sum_f32(hvx_vec_splat_f32(sum_sq), ne0, epsilon, scale);
    uctx->rms_apply_row_fn(src, weight, add, dst, n, scale_v);
}

static void rms_norm_rows(const void * restrict src,
                          void * restrict dst,
                          const uint32_t num_rows,
                          const uint32_t ne0,
                          const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    float scale   = 0.f;
    memcpy(&epsilon, &op_params[0], sizeof(float));
    memcpy(&scale,   &op_params[1], sizeof(float));

    const rms_row_scale_fn_t row_scale_fn = uctx->rms_row_scale_fn;
    const rms_apply_row_fn_t apply_row_fn = uctx->rms_apply_row_fn;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        HVX_Vector scale_v = row_scale_fn(src_local, ne0, epsilon, scale);
        apply_row_fn(src_local, NULL, NULL, dst_local, ne0, scale_v);
    }
}

static void rms_norm_mul_rows(const void * restrict src,
                              const void * restrict weight,
                              void * restrict dst,
                              const uint32_t num_rows,
                              const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const uint32_t ne0 = uctx->nc;
    float epsilon = 0.f;
    float scale   = 0.f;
    memcpy(&epsilon, &op_params[0], sizeof(float));
    memcpy(&scale,   &op_params[1], sizeof(float));

    const rms_row_scale_fn_t row_scale_fn = uctx->rms_row_scale_fn;
    const rms_apply_row_fn_t apply_row_fn = uctx->rms_apply_row_fn;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *) src + (ir * src0_row_size_aligned);
        const uint8_t * restrict w_local   = (const uint8_t *) weight + (uctx->broadcast_weight ? 0 : ir * uctx->src1_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        HVX_Vector scale_v = row_scale_fn(src_local, ne0, epsilon, scale);
        apply_row_fn(src_local, w_local, NULL, dst_local, ne0, scale_v);
    }
}

static void rms_norm_mul_add_rows(const void * restrict src,
                                  const void * restrict weight,
                                  const void * restrict add,
                                  void * restrict dst,
                                  const uint32_t num_rows,
                                  const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const uint32_t ne0 = uctx->nc;
    float epsilon = 0.f;
    float scale   = 0.f;
    memcpy(&epsilon, &op_params[0], sizeof(float));
    memcpy(&scale,   &op_params[1], sizeof(float));

    const uint8_t * restrict w_local      = (const uint8_t *) weight;
    const rms_row_scale_fn_t row_scale_fn = uctx->rms_row_scale_fn;
    const rms_apply_row_fn_t apply_row_fn = uctx->rms_apply_row_fn;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *) src + (ir * src0_row_size_aligned);
        const uint8_t * restrict add_local = (const uint8_t *) add + (ir * uctx->src2_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        HVX_Vector scale_v = row_scale_fn(src_local, ne0, epsilon, scale);
        apply_row_fn(src_local, w_local, add_local, dst_local, ne0, scale_v);
    }
}

static void norm_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

static void sqr_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void sqrt_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void scale_f16(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const uint32_t ne0,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0, scale, bias);
    }
}

static void clamp_f16(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const uint32_t ne0,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float min = 0.f;
    float max = 0.f;
    memcpy(&min, &op_params[0], sizeof(float));
    memcpy(&max, &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f16_aa(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

static inline void hvx_leaky_relu_scalar_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, float slope, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const HVX_Vector vslope = hvx_vec_splat_f32(slope);
    const HVX_Vector vzero  = Q6_V_vzero();

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[i]);
        HVX_Vector lo = hvx_vec_clamp_f16_f32(hvx_vec_leaky_relu_f32(Q6_V_lo_W(f32p), vslope, vzero));
        HVX_Vector hi = hvx_vec_clamp_f16_f32(hvx_vec_leaky_relu_f32(Q6_V_hi_W(f32p), vslope, vzero));
        vdst[i] = hvx_vec_f32_to_f16(lo, hi);
    }
    if (nloe) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[nvec]);
        HVX_Vector lo = hvx_vec_clamp_f16_f32(hvx_vec_leaky_relu_f32(Q6_V_lo_W(f32p), vslope, vzero));
        HVX_Vector hi = hvx_vec_clamp_f16_f32(hvx_vec_leaky_relu_f32(Q6_V_hi_W(f32p), vslope, vzero));
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), hvx_vec_f32_to_f16(lo, hi));
    }
}

static void leaky_relu_f16(const void * restrict src,
                           void * restrict dst,
                           const uint32_t num_rows,
                           const uint32_t ne0,
                           const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float negative_slope = 0.f;
    memcpy(&negative_slope, &op_params[0], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_leaky_relu_scalar_f16_aa(dst_local, src_local, negative_slope, ne0);
    }
}

static void norm_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f16((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

static void sqr_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void sqrt_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void abs_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void log_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void step_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void ceil_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_ceil_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void sgn_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sgn_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void l2_norm_f16(const void * restrict src,
                        void * restrict dst,
                        const uint32_t num_rows,
                        const uint32_t ne0,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_f = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f16((const uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

static void neg_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

static inline void unary_neg_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[i]);
        HVX_Vector lo = hvx_vec_neg_f32(Q6_V_lo_W(f32p));
        HVX_Vector hi = hvx_vec_neg_f32(Q6_V_hi_W(f32p));
        vdst[i] = hvx_vec_f32_to_f16(lo, hi);
    }
    if (nloe) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[nvec]);
        HVX_Vector lo = hvx_vec_neg_f32(Q6_V_lo_W(f32p));
        HVX_Vector hi = hvx_vec_neg_f32(Q6_V_hi_W(f32p));
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), hvx_vec_f32_to_f16(lo, hi));
    }
}

static void neg_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        unary_neg_f16_aa(dst_local, src_local, ne0);
    }
}

static __attribute__((noinline)) HVX_Vector unary_vec_exp_f32(HVX_Vector x) {
    return hvx_vec_exp_f32(x);
}

static inline void unary_exp_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = unary_vec_exp_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(float), unary_vec_exp_f32(vsrc[nvec]));
    }
}

static void exp_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        unary_exp_f32_aa(dst_local, src_local, ne0);
    }
}

static __attribute__((noinline)) HVX_Vector unary_vec_exp_f16(HVX_Vector v) {
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    HVX_VectorPair f32p = hvx_vec_f16_to_f32(v);
    HVX_Vector lo = hvx_vec_clamp_f16_f32(hvx_vec_exp_f32(Q6_Vsf_vmin_VsfVsf(Q6_V_lo_W(f32p), max_exp)));
    HVX_Vector hi = hvx_vec_clamp_f16_f32(hvx_vec_exp_f32(Q6_Vsf_vmin_VsfVsf(Q6_V_hi_W(f32p), max_exp)));
    return hvx_vec_f32_to_f16(lo, hi);
}

static inline void hvx_exp_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = unary_vec_exp_f16(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), unary_vec_exp_f16(vsrc[nvec]));
    }
}

static void exp_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_exp_f16_aa(dst_local, src_local, ne0);
    }
}

static __attribute__((noinline)) HVX_Vector unary_vec_sigmoid_f32(HVX_Vector x) {
    return hvx_vec_sigmoid_f32(x);
}

static inline void unary_sigmoid_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = unary_vec_sigmoid_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(float), unary_vec_sigmoid_f32(vsrc[nvec]));
    }
}

static void sigmoid_f32(const void * restrict src,
                        void * restrict dst,
                        const uint32_t num_rows,
                        const uint32_t ne0,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        unary_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

static inline void unary_sigmoid_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[i]);
        HVX_Vector lo = hvx_vec_sigmoid_f32(Q6_V_lo_W(f32p));
        HVX_Vector hi = hvx_vec_sigmoid_f32(Q6_V_hi_W(f32p));
        vdst[i] = hvx_vec_f32_to_f16(lo, hi);
    }
    if (nloe) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[nvec]);
        HVX_Vector lo = hvx_vec_sigmoid_f32(Q6_V_lo_W(f32p));
        HVX_Vector hi = hvx_vec_sigmoid_f32(Q6_V_hi_W(f32p));
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), hvx_vec_f32_to_f16(lo, hi));
    }
}

static void sigmoid_f16(const void * restrict src,
                        void * restrict dst,
                        const uint32_t num_rows,
                        const uint32_t ne0,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        unary_sigmoid_f16_aa(dst_local, src_local, ne0);
    }
}

static __attribute__((noinline)) HVX_Vector unary_vec_silu_f32(HVX_Vector x) {
    return hvx_vec_silu_f32(x);
}

static inline void unary_silu_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = unary_vec_silu_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(float), unary_vec_silu_f32(vsrc[nvec]));
    }
}

// silu(x) = x * sigmoid(x)
static void silu_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        unary_silu_f32_aa(dst_local, src_local, ne0);
    }
}

static inline void hvx_silu_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[i]);
        HVX_Vector lo = hvx_vec_clamp_f16_f32(hvx_vec_silu_f32(Q6_V_lo_W(f32p)));
        HVX_Vector hi = hvx_vec_clamp_f16_f32(hvx_vec_silu_f32(Q6_V_hi_W(f32p)));
        vdst[i] = hvx_vec_f32_to_f16(lo, hi);
    }
    if (nloe) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[nvec]);
        HVX_Vector lo = hvx_vec_clamp_f16_f32(hvx_vec_silu_f32(Q6_V_lo_W(f32p)));
        HVX_Vector hi = hvx_vec_clamp_f16_f32(hvx_vec_silu_f32(Q6_V_hi_W(f32p)));
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), hvx_vec_f32_to_f16(lo, hi));
    }
}

static void silu_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_silu_f16_aa(dst_local, src_local, ne0);
    }
}

// GELU uses the tanh approximation.
static inline HVX_Vector hvx_vec_gelu_f32_inline(HVX_Vector x) {
    const HVX_Vector half = hvx_vec_splat_f32(0.5f);
    const HVX_Vector one  = hvx_vec_splat_f32(1.0f);

    HVX_Vector inner = hvx_vec_mul_f32_f32(x, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
    inner = hvx_vec_add_f32_f32(inner, one);
    inner = hvx_vec_mul_f32_f32(inner, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));

    return hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
                               hvx_vec_add_f32_f32(one, hvx_vec_tanh_f32(inner)));
}

static __attribute__((noinline)) HVX_Vector hvx_vec_gelu_f32(HVX_Vector x) {
    return hvx_vec_gelu_f32_inline(x);
}

static inline void hvx_gelu_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), hvx_vec_gelu_f32(vsrc[i]));
    }
}

static __attribute__((noinline)) HVX_Vector unary_vec_gelu_f16(HVX_Vector v) {
    HVX_VectorPair f32p = hvx_vec_f16_to_f32(v);
    HVX_Vector lo = hvx_vec_clamp_f16_f32(hvx_vec_gelu_f32_inline(Q6_V_lo_W(f32p)));
    HVX_Vector hi = hvx_vec_clamp_f16_f32(hvx_vec_gelu_f32_inline(Q6_V_hi_W(f32p)));
    return hvx_vec_f32_to_f16(lo, hi);
}

static inline void hvx_gelu_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = unary_vec_gelu_f16(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), unary_vec_gelu_f16(vsrc[nvec]));
    }
}

// GELU_QUICK uses x * sigmoid(1.702 * x).
static inline HVX_Vector hvx_vec_gelu_quick_f32_inline(HVX_Vector x) {
    const HVX_Vector scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    const HVX_Vector sigmoid = hvx_vec_sigmoid_f32(scaled);

    return hvx_vec_mul_f32_f32(x, sigmoid);
}

static __attribute__((noinline)) HVX_Vector hvx_vec_gelu_quick_f32(HVX_Vector x) {
    return hvx_vec_gelu_quick_f32_inline(x);
}

static inline void hvx_gelu_quick_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_quick_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), hvx_vec_gelu_quick_f32(vsrc[i]));
    }
}

static __attribute__((noinline)) HVX_Vector unary_vec_gelu_quick_f16(HVX_Vector v) {
    HVX_VectorPair f32p = hvx_vec_f16_to_f32(v);
    HVX_Vector lo = hvx_vec_clamp_f16_f32(hvx_vec_gelu_quick_f32_inline(Q6_V_lo_W(f32p)));
    HVX_Vector hi = hvx_vec_clamp_f16_f32(hvx_vec_gelu_quick_f32_inline(Q6_V_hi_W(f32p)));
    return hvx_vec_f32_to_f16(lo, hi);
}

static inline void hvx_gelu_quick_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = unary_vec_gelu_quick_f16(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), unary_vec_gelu_quick_f16(vsrc[nvec]));
    }
}

static __attribute__((noinline)) HVX_Vector unary_vec_gelu_erf_f16(HVX_Vector v) {
    HVX_VectorPair f32p = hvx_vec_f16_to_f32(v);
    HVX_Vector lo = hvx_vec_clamp_f16_f32(hvx_vec_gelu_erf_f32(Q6_V_lo_W(f32p)));
    HVX_Vector hi = hvx_vec_clamp_f16_f32(hvx_vec_gelu_erf_f32(Q6_V_hi_W(f32p)));
    return hvx_vec_f32_to_f16(lo, hi);
}

static inline void hvx_gelu_erf_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = unary_vec_gelu_erf_f16(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), unary_vec_gelu_erf_f16(vsrc[nvec]));
    }
}

static void gelu_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

static void gelu_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_f16_aa(dst_local, src_local, ne0);
    }
}

static void gelu_quick_f32(const void * restrict src,
                           void * restrict dst,
                           const uint32_t num_rows,
                           const uint32_t ne0,
                           const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_quick_f32_aa(dst_local, src_local, ne0);
    }
}

static void gelu_quick_f16(const void * restrict src,
                           void * restrict dst,
                           const uint32_t num_rows,
                           const uint32_t ne0,
                           const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_quick_f16_aa(dst_local, src_local, ne0);
    }
}

static void gelu_erf_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const uint32_t ne0,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

static void gelu_erf_f16(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const uint32_t ne0,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_gelu_erf_f16_aa(dst_local, src_local, ne0);
    }
}

static void tri_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ir,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const uint32_t ne0 = uctx->nc;
    const int32_t ttype = op_params[0];
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    const uint32_t nvec  = ne0 / VLEN_FP32;
    const uint32_t nloe  = ne0 % VLEN_FP32;

    const uint32_t ne01 = uctx->octx->src[0]->ne[1];
    const struct fastdiv_values * div_ne01 = &uctx->kparams->div_ne01;

    for (uint32_t b = 0; b < num_rows; b++) {
        const uint32_t abs_row = ir + b;
        const uint32_t i01     = fastmodulo(abs_row, ne01, div_ne01);

        const HVX_Vector * restrict v_src = (const HVX_Vector *) ((const uint8_t *) src + b * src0_row_size_aligned);
        HVX_Vector * restrict v_dst       = (HVX_Vector *) ((uint8_t *) dst + b * dst_row_size_aligned);

        uint32_t boundary;
        int      keep_left;
        switch (ttype) {
            case 0: boundary = i01;     keep_left = 0; break;  // keep col >= row
            case 1: boundary = i01 + 1; keep_left = 0; break;  // keep col > row
            case 2: boundary = i01 + 1; keep_left = 1; break;  // keep col <= row
            case 3: boundary = i01;     keep_left = 1; break;  // keep col < row
            default: boundary = 0; keep_left = 0; break;
        }
        if (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        for (uint32_t i = 0; i < nvec; i++) {
            const uint32_t vec_start = i * VLEN_FP32;
            const uint32_t vec_end   = vec_start + VLEN_FP32;
            if (keep_left) {
                if (vec_end <= boundary) {
                    v_dst[i] = v_src[i];
                } else if (vec_start >= boundary) {
                    v_dst[i] = zero;
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * sizeof(float));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
                }
            } else {
                if (vec_end <= boundary) {
                    v_dst[i] = zero;
                } else if (vec_start >= boundary) {
                    v_dst[i] = v_src[i];
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * sizeof(float));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        if (nloe > 0) {
            const uint32_t abs_start = nvec * VLEN_FP32;
            const uint32_t abs_end   = abs_start + nloe;
            HVX_Vector     tail_val;
            if (keep_left) {
                if (abs_end <= boundary) {
                    tail_val = v_src[nvec];
                } else if (abs_start >= boundary) {
                    tail_val = zero;
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                    tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
                }
            } else {
                if (abs_end <= boundary) {
                    tail_val = zero;
                } else if (abs_start >= boundary) {
                    tail_val = v_src[nvec];
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                    tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(float), tail_val);
        }
    }
}

// softplus(x) = log(1 + exp(x))
// Match CPU reference: return x for x > 20 to avoid overflow.
static inline HVX_Vector hvx_vec_softplus_f32_inline(HVX_Vector x) {
    const HVX_Vector twenty = hvx_vec_splat_f32(20.0f);
    const HVX_Vector one    = hvx_vec_splat_f32(1.0f);

    HVX_VectorPred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
    HVX_Vector     expx      = hvx_vec_exp_f32(x);
    HVX_Vector     logv      = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
    return Q6_V_vmux_QVV(pred_gt20, x, logv);
}

static __attribute__((noinline)) HVX_Vector hvx_vec_softplus_f32(HVX_Vector x) {
    return hvx_vec_softplus_f32_inline(x);
}

static inline void hvx_softplus_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    for (uint32_t i = 0; i < nvec; i++) {
        v_dst[i] = hvx_vec_softplus_f32(v_src[i]);
    }

    if (nloe > 0) {
        hvx_vec_store_a((void *) &v_dst[nvec], nloe * sizeof(float), hvx_vec_softplus_f32(v_src[nvec]));
    }
}

static __attribute__((noinline)) HVX_Vector unary_vec_softplus_f16(HVX_Vector v) {
    HVX_VectorPair f32p = hvx_vec_f16_to_f32(v);
    HVX_Vector lo = hvx_vec_clamp_f16_f32(hvx_vec_softplus_f32_inline(Q6_V_lo_W(f32p)));
    HVX_Vector hi = hvx_vec_clamp_f16_f32(hvx_vec_softplus_f32_inline(Q6_V_hi_W(f32p)));
    return hvx_vec_f32_to_f16(lo, hi);
}

static inline void hvx_softplus_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = unary_vec_softplus_f16(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), unary_vec_softplus_f16(vsrc[nvec]));
    }
}

static void softplus_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const uint32_t ne0,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
    }
}

static void softplus_f16(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const uint32_t ne0,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_softplus_f16_aa(dst_local, src_local, ne0);
    }
}

static void l2_norm_f32(const void * restrict src,
                        void * restrict dst,
                        const uint32_t num_rows,
                        const uint32_t ne0,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_f = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f32((const uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

static inline HVX_VectorPair hvx_fwht_stage_pair(HVX_Vector v0, HVX_Vector v1, int32_t b) {
    HVX_VectorPair p = Q6_W_vshuff_VVR(v1, v0, b);
    HVX_Vector s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    HVX_Vector d = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    return Q6_W_vshuff_VVR(d, s, b);
}

static inline HVX_VectorPair hvx_fwht32_pair(HVX_Vector v0_sf, HVX_Vector v1_sf, HVX_Vector v_scale) {
    HVX_Vector v0 = Q6_Vqf32_vmpy_VsfVsf(v0_sf, v_scale);
    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(v1_sf, v_scale);
    HVX_VectorPair p;
    p = hvx_fwht_stage_pair(v0, v1, 4);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 8);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 16);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 32);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 64);
    return p;
}

static inline void hvx_fwht_store32(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
}

static inline void hvx_fwht_store64(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    HVX_Vector u = Q6_V_lo_W(p);
    HVX_Vector v = Q6_V_hi_W(p);
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(u, v));
    v_dst[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(u, v));
}

static inline void hvx_fwht_tail_f32(HVX_Vector * restrict v_dst, uint32_t nvec) {
    uint32_t len_vec = 2;
    for (; len_vec * 2 < nvec; len_vec <<= 2) {
        const uint32_t step = len_vec * 4;
        for (uint32_t i = 0; i < nvec; i += step) {
            for (uint32_t j = 0; j < len_vec; j++) {
                HVX_Vector a0 = v_dst[i + j];
                HVX_Vector a1 = v_dst[i + j + len_vec];
                HVX_Vector a2 = v_dst[i + j + 2 * len_vec];
                HVX_Vector a3 = v_dst[i + j + 3 * len_vec];

                HVX_Vector t0 = Q6_Vqf32_vadd_VsfVsf(a0, a1);
                HVX_Vector t1 = Q6_Vqf32_vsub_VsfVsf(a0, a1);
                HVX_Vector t2 = Q6_Vqf32_vadd_VsfVsf(a2, a3);
                HVX_Vector t3 = Q6_Vqf32_vsub_VsfVsf(a2, a3);

                v_dst[i + j]               = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t0, t2));
                v_dst[i + j + len_vec]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t1, t3));
                v_dst[i + j + 2 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t0, t2));
                v_dst[i + j + 3 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t1, t3));
            }
        }
    }

    if (len_vec < nvec) {
        for (uint32_t i = 0; i < nvec; i += 2 * len_vec) {
            for (uint32_t j = 0; j < len_vec; j++) {
                HVX_Vector u = v_dst[i + j];
                HVX_Vector v = v_dst[i + len_vec + j];
                v_dst[i + j]           = hvx_vec_add_f32_f32(u, v);
                v_dst[i + len_vec + j] = hvx_vec_sub_f32_f32(u, v);
            }
        }
    }
}

static inline void hvx_fwht_row_f32(const uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    const uint32_t nvec               = n / VLEN_FP32;

    if (nvec == 1) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[0], v_src[0], v_scale);
        hvx_fwht_store32(v_dst, p);
        return;
    }

    for (uint32_t i = 0; i < nvec; i += 2) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[i], v_src[i + 1], v_scale);
        hvx_fwht_store64(&v_dst[i], p);
    }

    hvx_fwht_tail_f32(v_dst, nvec);
}

static inline void hvx_fwht_row_f16(const uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    const uint32_t nvec               = n / VLEN_FP32;

    if (nvec == 1) {
        HVX_Vector v0 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src[0]));
        HVX_VectorPair p = hvx_fwht32_pair(v0, v0, v_scale);
        hvx_fwht_store32(v_dst, p);
        return;
    }

    const uint32_t nvec_f16 = n / VLEN_FP16;
    for (uint32_t i = 0; i < nvec_f16; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_VectorPair p    = hvx_fwht32_pair(Q6_V_lo_W(f32p), Q6_V_hi_W(f32p), v_scale);
        hvx_fwht_store64(&v_dst[2 * i], p);
    }

    hvx_fwht_tail_f32(v_dst, nvec);
}

static void hadamard_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const uint32_t ne0,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((float) ne0));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_row = (const uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fwht_row_f32(src_row, dst_row, ne0, v_scale);
    }
}

static void hadamard_f16(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const uint32_t ne0,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((float) ne0));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_row = (const uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fwht_row_f16(src_row, dst_row, ne0, v_scale);
    }
}

static __attribute__((noinline)) HVX_Vector unary_vec_tanh_f32(HVX_Vector x) {
    return hvx_vec_tanh_f32(x);
}

static inline void unary_tanh_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = unary_vec_tanh_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(float), unary_vec_tanh_f32(vsrc[nvec]));
    }
}

static void tanh_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        unary_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

static inline void unary_tanh_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[i]);
        HVX_Vector lo = hvx_vec_tanh_f32(Q6_V_lo_W(f32p));
        HVX_Vector hi = hvx_vec_tanh_f32(Q6_V_hi_W(f32p));
        vdst[i] = hvx_vec_f32_to_f16(lo, hi);
    }
    if (nloe) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[nvec]);
        HVX_Vector lo = hvx_vec_tanh_f32(Q6_V_lo_W(f32p));
        HVX_Vector hi = hvx_vec_tanh_f32(Q6_V_hi_W(f32p));
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), hvx_vec_f32_to_f16(lo, hi));
    }
}

static void tanh_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        unary_tanh_f16_aa(dst_local, src_local, ne0);
    }
}

static void abs_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

static void relu_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_max_scalar_f32_aa(dst_local, src_local, 0.0f, ne0);
    }
}

static inline void hvx_relu_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    const HVX_Vector zero = Q6_V_vzero();

    const uint32_t nvec = n / VLEN_FP16;
    const uint32_t nloe = n % VLEN_FP16;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[i]);
        HVX_Vector lo = Q6_Vsf_vmax_VsfVsf(Q6_V_lo_W(f32p), zero);
        HVX_Vector hi = Q6_Vsf_vmax_VsfVsf(Q6_V_hi_W(f32p), zero);
        vdst[i] = hvx_vec_f32_to_f16(lo, hi);
    }
    if (nloe) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(vsrc[nvec]);
        HVX_Vector lo = Q6_Vsf_vmax_VsfVsf(Q6_V_lo_W(f32p), zero);
        HVX_Vector hi = Q6_Vsf_vmax_VsfVsf(Q6_V_hi_W(f32p), zero);
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(_Float16), hvx_vec_f32_to_f16(lo, hi));
    }
}

static void relu_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_relu_f16_aa(dst_local, src_local, ne0);
    }
}

static void step_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f32_aa(dst_local, src_local, ne0);
    }
}

static void ceil_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const uint32_t ne0,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_ceil_f32_aa(dst_local, src_local, ne0);
    }
}

static void sgn_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sgn_f32_aa(dst_local, src_local, ne0);
    }
}

static void log_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ne0,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

static void tri_apply_tile_f32(const void * restrict src, void * restrict dst,
                               uint32_t tile_elems, uint32_t col_start, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    int      keep_left;
    switch (ttype) {
        case 0: boundary = i01;     keep_left = 0; break;
        case 1: boundary = i01 + 1; keep_left = 0; break;
        case 2: boundary = i01 + 1; keep_left = 1; break;
        case 3: boundary = i01;     keep_left = 1; break;
        default: boundary = 0; keep_left = 0; break;
    }
    if (boundary > ne0) boundary = ne0;

    const uint32_t nvec = tile_elems / VLEN_FP32;
    const uint32_t nloe = tile_elems % VLEN_FP32;

    for (uint32_t i = 0; i < nvec; i++) {
        const uint32_t abs_start = col_start + i * VLEN_FP32;
        const uint32_t abs_end   = abs_start + VLEN_FP32;
        if (keep_left) {
            if (abs_end <= boundary) {
                v_dst[i] = v_src[i];
            } else if (abs_start >= boundary) {
                v_dst[i] = zero;
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
            }
        } else {
            if (abs_end <= boundary) {
                v_dst[i] = zero;
            } else if (abs_start >= boundary) {
                v_dst[i] = v_src[i];
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
            }
        }
    }

    if (nloe > 0) {
        const uint32_t abs_start = col_start + nvec * VLEN_FP32;
        const uint32_t abs_end   = abs_start + nloe;
        HVX_Vector     tail_val;
        if (keep_left) {
            if (abs_end <= boundary) {
                tail_val = v_src[nvec];
            } else if (abs_start >= boundary) {
                tail_val = zero;
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
            }
        } else {
            if (abs_end <= boundary) {
                tail_val = zero;
            } else if (abs_start >= boundary) {
                tail_val = v_src[nvec];
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(float), tail_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
static void unary_thread_row_block(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = th->trace;

    htp_unary_preamble;

    const uint32_t src0_nrows_per_hvx = uctx->src0_nrows_per_hvx;
    const size_t src0_data_row_size = uctx->src0_data_row_size;
    const size_t dst_data_row_size  = uctx->dst_data_row_size;
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_hvx);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_hvx);

    const size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    const size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    const bool src0_contig = uctx->kparams->src0_contig;
    const uint32_t BLOCK   = uctx->block;

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;

    dma_queue * dma_q = th->dma_q;

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, ne01, div_ne01);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = src0_contig ? ((size_t) ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, ne01, div_ne01);

        void * dst_vtcm  = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src0_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, ne0, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        const size_t dst_off = (size_t) ir * nb1;
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              ne01, div_ne01);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  ne01, div_ne01);
                const size_t src0_pref_off = src0_contig ? ((size_t) pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
static void unary_thread_rms_norm_mul(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = th->trace;

    htp_unary_preamble;

    const uint32_t src0_nrows_per_hvx = uctx->src0_nrows_per_hvx;
    const size_t src0_data_row_size = uctx->src0_data_row_size;
    const size_t dst_data_row_size  = uctx->dst_data_row_size;
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src  = uctx->data_src0;
    const dma_addr_t data_src1 = uctx->data_src1;
    const dma_addr_t data_dst  = uctx->data_dst;

    const struct htp_tensor * src1 = octx->src[1];
    const uint32_t nb11 = src1->nb[1];
    const uint32_t nb12 = src1->nb[2];
    const uint32_t nb13 = src1->nb[3];

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_hvx);
    uint8_t * src1_vtcm_data = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_size_per_hvx)) : NULL;
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_hvx);

    const size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    const size_t src1_vtcm_half_size = uctx->src1_vtcm_half_size;
    const size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    const bool src0_contig  = uctx->kparams->src0_contig;
    const bool src1_contig  = uctx->kparams->src1_contig;
    const bool block_contig = src0_contig && src1_contig;
    const uint32_t BLOCK    = uctx->block;

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;

    dma_queue * dma_q = th->dma_q;

    if (uctx->broadcast_weight) {
        dma_queue_push(dma_q, dma_make_data(src1_vtcm_data, data_src1),
                       uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);
        dma_queue_flush(dma_q);
    }

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_contig, ne01, div_ne01);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = src0_contig ? ((size_t) ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        if (!uctx->broadcast_weight) {
            const size_t src1_off = src1_contig ? ((size_t) ir * nb11) :
                unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11, nb12, nb13);
            dma_queue_push(dma_q,
                dma_make_data(src1_vtcm_data + (vtcm_idx * src1_vtcm_half_size), data_src1 + src1_off),
                uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, block_size);
        }

        ir += block_size;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_contig, ne01, div_ne01);

        void * dst_vtcm  = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src0_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        void * src1_vtcm = NULL;
        if (!uctx->broadcast_weight) {
            src1_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        const void * w = uctx->broadcast_weight ? (const void *) src1_vtcm_data : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        const size_t dst_off = (size_t) ir * nb1;
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, block_contig,
                                                              ne01, div_ne01);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, block_contig,
                                                                  ne01, div_ne01);
                const size_t src0_pref_off = src0_contig ? ((size_t) pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);

                if (!uctx->broadcast_weight) {
                    const size_t src1_pref_off = src1_contig ? ((size_t) pref_ir * nb11) :
                        unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11, nb12, nb13);
                    dma_queue_push(dma_q,
                        dma_make_data(src1_vtcm, data_src1 + src1_pref_off),
                        uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, pref_block_size);
                }
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 2b. RMS_NORM_MUL_ADD row-block task with broadcast weight and residual add.
static void unary_thread_rms_norm_mul_add(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = th->trace;

    htp_unary_preamble;

    const struct htp_tensor * src2 = octx->src[2];
    const uint32_t nb21 = src2->nb[1];
    const uint32_t nb22 = src2->nb[2];
    const uint32_t nb23 = src2->nb[3];

    const uint32_t src0_nrows_per_hvx  = uctx->src0_nrows_per_hvx;
    const size_t src0_data_row_size    = uctx->src0_data_row_size;
    const size_t src2_data_row_size    = uctx->src2_data_row_size;
    const size_t dst_data_row_size     = uctx->dst_data_row_size;
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t src2_row_size_aligned = uctx->src2_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows     = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src  = uctx->data_src0;
    const dma_addr_t data_src1 = uctx->data_src1;
    const dma_addr_t data_src2 = uctx->data_src2;
    const dma_addr_t data_dst  = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_hvx);
    uint8_t * src1_vtcm_data = uctx->vtcm_src1 + (ith * uctx->vtcm_src1_size_per_hvx);
    uint8_t * src2_vtcm_data = uctx->vtcm_src2 + (ith * uctx->vtcm_src2_size_per_hvx);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst  + (ith * uctx->vtcm_dst_size_per_hvx);

    const size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    const size_t src2_vtcm_half_size = uctx->src2_vtcm_half_size;
    const size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    const bool src0_contig = uctx->kparams->src0_contig;
    const uint32_t BLOCK   = uctx->block;

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;

    dma_queue * dma_q = th->dma_q;

    dma_queue_push(dma_q, dma_make_data(src1_vtcm_data, data_src1),
                   uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);
    dma_queue_flush(dma_q);

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, ne01, div_ne01);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = src0_contig ? ((size_t) ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        const size_t src2_off = src0_contig ? ((size_t) ir * nb21) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb21, nb22, nb23);
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);
        dma_queue_push(dma_q,
            dma_make_data(src2_vtcm_data + (vtcm_idx * src2_vtcm_half_size), data_src2 + src2_off),
            src2_row_size_aligned, nb21, src2_data_row_size, block_size);

        ir += block_size;
    }

    unary_rms_norm_mul_add_compute_fn_t compute = (unary_rms_norm_mul_add_compute_fn_t) uctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, ne01, div_ne01);

        void * dst_vtcm  = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src0_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        void * src2_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, src1_vtcm_data, src2_vtcm, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        const size_t dst_off = (size_t) ir * nb1;
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              ne01, div_ne01);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  ne01, div_ne01);
                const size_t src0_pref_off = src0_contig ? ((size_t) pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                const size_t src2_pref_off = src0_contig ? ((size_t) pref_ir * nb21) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb21, nb22, nb23);
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
                dma_queue_push(dma_q,
                    dma_make_data(src2_vtcm, data_src2 + src2_pref_off),
                    src2_row_size_aligned, nb21, src2_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 2c. 2-pass column-tiled task for wide-row RMS_NORM / RMS_NORM_MUL / RMS_NORM_MUL_ADD.
static void unary_thread_tiled_rms_norm(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = th->trace;

    htp_unary_preamble;

    const uint32_t src0_nrows_per_hvx = uctx->src0_nrows_per_hvx;
    const uint32_t col_tile           = uctx->col_tile;

    const uint32_t src0_nrows     = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const bool has_weight = (octx->op == HTP_OP_RMS_NORM_MUL || octx->op == HTP_OP_RMS_NORM_MUL_ADD);
    const bool has_src2   = (octx->op == HTP_OP_RMS_NORM_MUL_ADD);

    const struct htp_tensor * src1 = has_weight ? octx->src[1] : NULL;
    const struct htp_tensor * src2 = has_src2   ? octx->src[2] : NULL;

    const uint32_t nb10 = has_weight ? src1->nb[0] : 0;
    const uint32_t nb11 = has_weight ? src1->nb[1] : 0;
    const uint32_t nb12 = has_weight ? src1->nb[2] : 0;
    const uint32_t nb13 = has_weight ? src1->nb[3] : 0;

    const uint32_t nb20 = has_src2 ? src2->nb[0] : 0;
    const uint32_t nb21 = has_src2 ? src2->nb[1] : 0;
    const uint32_t nb22 = has_src2 ? src2->nb[2] : 0;
    const uint32_t nb23 = has_src2 ? src2->nb[3] : 0;

    const dma_addr_t data_src  = uctx->data_src0;
    const dma_addr_t data_src1 = uctx->data_src1;
    const dma_addr_t data_src2 = uctx->data_src2;
    const dma_addr_t data_dst  = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_hvx);
    uint8_t * src1_vtcm_data = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_size_per_hvx)) : NULL;
    uint8_t * src2_vtcm_data = uctx->vtcm_src2 ? (uctx->vtcm_src2 + (ith * uctx->vtcm_src2_size_per_hvx)) : NULL;
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst  + (ith * uctx->vtcm_dst_size_per_hvx);

    const size_t src0_half = uctx->src0_vtcm_half_size;
    const size_t src1_half = uctx->src1_vtcm_half_size;
    const size_t src2_half = uctx->src2_vtcm_half_size;
    const size_t dst_half  = uctx->dst_vtcm_half_size;

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;

    const uint32_t tiles_per_row = uctx->kparams->tiles_per_row;
    const bool     src0_contig   = uctx->kparams->src0_contig;
    const bool     src1_contig   = uctx->kparams->src1_contig;

    float epsilon = 0.f;
    float scale   = 0.f;
    memcpy(&epsilon, &octx->op_params[0], sizeof(float));
    memcpy(&scale,   &octx->op_params[1], sizeof(float));

    dma_queue * dma_q = th->dma_q;
    const rms_tile_sum_sq_fn_t tile_sum_sq_fn = uctx->rms_tile_sum_sq_fn;

    for (uint32_t row = src0_start_row; row < src0_end_row; row++) {
        const size_t src0_row_off = src0_contig ? ((size_t) row * nb01) :
            unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        const size_t src1_row_off = (!has_weight || uctx->broadcast_weight) ? 0 :
            (src1_contig ? ((size_t) row * nb11) :
             unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11, nb12, nb13));
        const size_t src2_row_off = !has_src2 ? 0 :
            (src0_contig ? ((size_t) row * nb21) :
             unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb21, nb22, nb23));
        const size_t dst_row_off  = (size_t) row * nb1;

        for (uint32_t t = 0; t < tiles_per_row && t < 2; t++) {
            const uint32_t col = t * col_tile;
            const uint32_t tw  = MIN(col_tile, ne0 - col);
            const size_t   tb0 = (size_t) tw * nb00;
            dma_queue_push(dma_q,
                dma_make_data(src0_vtcm_data + (t * src0_half), data_src + src0_row_off + (size_t) col * nb00),
                tb0, tb0, tb0, 1);
        }

        float sum_sq = 0.0f;
        for (uint32_t t = 0; t < tiles_per_row; t++) {
            const uint32_t col = t * col_tile;
            const uint32_t tw  = MIN(col_tile, ne0 - col);
            const uint8_t * s0 = (const uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row);
            sum_sq += tile_sum_sq_fn(s0, tw);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row);

            const uint32_t pt = t + 2;
            if (pt < tiles_per_row) {
                const uint32_t pcol = pt * col_tile;
                const uint32_t ptw  = MIN(col_tile, ne0 - pcol);
                const size_t   ptb0 = (size_t) ptw * nb00;
                dma_queue_push(dma_q,
                    dma_make_data((void *) s0, data_src + src0_row_off + (size_t) pcol * nb00),
                    ptb0, ptb0, ptb0, 1);
            }
        }

        for (uint32_t t = 0; t < tiles_per_row && t < 2; t++) {
            const uint32_t col = t * col_tile;
            const uint32_t tw  = MIN(col_tile, ne0 - col);
            const size_t   tb0 = (size_t) tw * nb00;

            dma_queue_push(dma_q, dma_make_data(data_dst, dst_vtcm_data + (t * dst_half)), 0, 0, 0, 0);
            dma_queue_push(dma_q,
                dma_make_data(src0_vtcm_data + (t * src0_half), data_src + src0_row_off + (size_t) col * nb00),
                tb0, tb0, tb0, 1);
            if (has_weight) {
                const size_t tb1 = (size_t) tw * nb10;
                dma_queue_push(dma_q,
                    dma_make_data(src1_vtcm_data + (t * src1_half), data_src1 + src1_row_off + (size_t) col * nb10),
                    tb1, tb1, tb1, 1);
            }
            if (has_src2) {
                const size_t tb2 = (size_t) tw * nb20;
                dma_queue_push(dma_q,
                    dma_make_data(src2_vtcm_data + (t * src2_half), data_src2 + src2_row_off + (size_t) col * nb20),
                    tb2, tb2, tb2, 1);
            }
        }

        for (uint32_t t = 0; t < tiles_per_row; t++) {
            const uint32_t col = t * col_tile;
            const uint32_t tw  = MIN(col_tile, ne0 - col);

            uint8_t *       d_vtcm  = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).src;
            const uint8_t * s0_vtcm = (const uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;
            const uint8_t * s1_vtcm = has_weight ? (const uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst : NULL;
            const uint8_t * s2_vtcm = has_src2   ? (const uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst : NULL;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row);
            rms_norm_apply_tile(s0_vtcm, s1_vtcm, s2_vtcm, d_vtcm, tw, ne0, sum_sq, epsilon, scale, uctx);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row);

            const size_t tb_dst = (size_t) tw * nb0;
            dma_queue_push(dma_q,
                dma_make_data(data_dst + dst_row_off + (size_t) col * nb0, d_vtcm),
                tb_dst, tb_dst, tb_dst, 1);

            const uint32_t pt = t + 2;
            if (pt < tiles_per_row) {
                const uint32_t pcol = pt * col_tile;
                const uint32_t ptw  = MIN(col_tile, ne0 - pcol);
                const size_t   ptb0 = (size_t) ptw * nb00;
                dma_queue_push(dma_q,
                    dma_make_data((void *) s0_vtcm, data_src + src0_row_off + (size_t) pcol * nb00),
                    ptb0, ptb0, ptb0, 1);
                if (has_weight) {
                    const size_t ptb1 = (size_t) ptw * nb10;
                    dma_queue_push(dma_q,
                        dma_make_data((void *) s1_vtcm, data_src1 + src1_row_off + (size_t) pcol * nb10),
                        ptb1, ptb1, ptb1, 1);
                }
                if (has_src2) {
                    const size_t ptb2 = (size_t) ptw * nb20;
                    dma_queue_push(dma_q,
                        dma_make_data((void *) s2_vtcm, data_src2 + src2_row_off + (size_t) pcol * nb20),
                        ptb2, ptb2, ptb2, 1);
                }
            }
        }

        dma_queue_flush(dma_q);
    }
}

// 3. TRI row-block task with row index ir.
static void unary_thread_tri_f32(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = th->trace;

    htp_unary_preamble;

    const uint32_t src0_nrows_per_hvx = uctx->src0_nrows_per_hvx;
    const size_t src0_data_row_size = uctx->src0_data_row_size;
    const size_t dst_data_row_size  = uctx->dst_data_row_size;
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_hvx);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_hvx);

    const size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    const size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;
    const uint32_t BLOCK             = uctx->block;

    dma_queue * dma_q = th->dma_q;

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = (size_t) ir * nb01;
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        void * dst_vtcm  = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src0_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        const size_t dst_off = (size_t) ir * nb1;
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = MIN(BLOCK, src0_end_row - next_ir);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = MIN(BLOCK, src0_end_row - pref_ir);
                const size_t src0_pref_off = (size_t) pref_ir * nb01;
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Pointwise tiled unary task.
static void unary_thread_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = th->trace;

    htp_unary_preamble;

    const uint32_t src0_nrows_per_hvx    = uctx->src0_nrows_per_hvx;
    const uint32_t col_tile              = uctx->col_tile;

    const uint32_t src0_nrows     = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_hvx);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_hvx);

    const size_t src0_half = uctx->src0_vtcm_half_size;
    const size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = th->dma_q;

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;
    const struct fastdiv_values * div_tpr   = &uctx->kparams->div_tpr;

    const uint32_t tiles_per_row = uctx->kparams->tiles_per_row;
    const bool     src0_contig   = uctx->kparams->src0_contig;

    const uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    for (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        const uint32_t row  = src0_start_row + fastdiv(t, div_tpr);
        const uint32_t col  = fastmodulo(t, tiles_per_row, div_tpr) * col_tile;
        const uint32_t tw   = MIN(col_tile, ne0 - col);
        const size_t   tb   = (size_t) tw * nb00;
        const size_t   soff = (src0_contig ? ((size_t) row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) col * nb00;

        dma_queue_push(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_push(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    for (uint32_t t = 0; t < total_tiles; t++) {
        void * dst_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        const uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, 1, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        const size_t doff = (size_t) row * nb1 + (size_t) col * nb0;
        const size_t tb   = (size_t) tw * nb0;
        dma_queue_push(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        const uint32_t pt = t + 2;
        if (pt < total_tiles) {
            const uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            const size_t   ptb  = (size_t) ptw * nb00;
            const size_t   psoff = (src0_contig ? ((size_t) prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * nb00;
            dma_queue_push(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        if (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
        }

        ptile_in_row++;
        pcol += col_tile;
        if (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

// 5. TRI tiled task.
static void unary_thread_tiled_tri_f32(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = th->trace;

    htp_unary_preamble;

    const uint32_t src0_nrows_per_hvx    = uctx->src0_nrows_per_hvx;
    const int32_t * op_params            = octx->op_params;
    const uint32_t col_tile              = uctx->col_tile;

    const uint32_t src0_nrows     = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_hvx, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_hvx);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_hvx);

    const size_t src0_half = uctx->src0_vtcm_half_size;
    const size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = th->dma_q;

    const struct fastdiv_values * div_ne01 = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_tpr  = &uctx->kparams->div_tpr;

    const uint32_t tiles_per_row = uctx->kparams->tiles_per_row;
    const int32_t  tri_ttype     = op_params[0];

    const uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    for (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        const uint32_t row  = src0_start_row + fastdiv(t, div_tpr);
        const uint32_t col  = fastmodulo(t, tiles_per_row, div_tpr) * col_tile;
        const uint32_t tw   = MIN(col_tile, ne0 - col);
        const size_t   tb   = (size_t) tw * sizeof(float);
        const size_t   soff = (size_t) row * nb01 + (size_t) col * sizeof(float);

        dma_queue_push(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_push(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;
    uint32_t i01 = fastmodulo(row, ne01, div_ne01);

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    for (uint32_t t = 0; t < total_tiles; t++) {
        void * dst_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        const uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, col, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        const size_t doff = (size_t) row * nb1 + (size_t) col * sizeof(float);
        const size_t tb   = (size_t) tw * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        const uint32_t pt = t + 2;
        if (pt < total_tiles) {
            const uint32_t ptw   = MIN(col_tile, ne0 - pcol);
            const size_t   ptb   = (size_t) ptw * sizeof(float);
            const size_t   psoff = (size_t) prow * nb01 + (size_t) pcol * sizeof(float);
            dma_queue_push(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        if (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
            i01++;
            if (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_in_row++;
        pcol += col_tile;
        if (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

static int execute_op_unary(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const bool is_f16 = (src0->type == HTP_TYPE_F16);

    const char * op_type = NULL;

    switch (octx->op) {
        case HTP_OP_NORM:             op_type = is_f16 ? "norm-f16"            : "norm-f32";            break;
        case HTP_OP_RMS_NORM:         op_type = is_f16 ? "rmsnorm-f16"         : "rmsnorm-f32";         break;
        case HTP_OP_RMS_NORM_MUL:     op_type = is_f16 ? "rmsnorm-mul-f16"     : "rmsnorm-mul-f32";     break;
        case HTP_OP_RMS_NORM_MUL_ADD: op_type = is_f16 ? "rmsnorm-mul-add-f16" : "rmsnorm-mul-add-f32"; break;
        case HTP_OP_SCALE:            op_type = is_f16 ? "scale-f16"           : "scale-f32";           break;
        case HTP_OP_CLAMP:            op_type = is_f16 ? "clamp-f16"           : "clamp-f32";           break;
        case HTP_OP_LEAKY_RELU:       op_type = is_f16 ? "leaky-relu-f16"      : "leaky-relu-f32";      break;
        case HTP_OP_SQR:              op_type = is_f16 ? "sqr-f16"             : "sqr-f32";             break;
        case HTP_OP_SQRT:             op_type = is_f16 ? "sqrt-f16"            : "sqrt-f32";            break;
        case HTP_OP_UNARY_NEG:        op_type = is_f16 ? "neg-f16"             : "neg-f32";             break;
        case HTP_OP_UNARY_EXP:        op_type = is_f16 ? "exp-f16"             : "exp-f32";             break;
        case HTP_OP_UNARY_SIGMOID:    op_type = is_f16 ? "sigmoid-f16"         : "sigmoid-f32";         break;
        case HTP_OP_UNARY_SILU:       op_type = is_f16 ? "silu-f16"            : "silu-f32";            break;
        case HTP_OP_UNARY_GELU:       op_type = is_f16 ? "gelu-f16"            : "gelu-f32";            break;
        case HTP_OP_UNARY_GELU_QUICK: op_type = is_f16 ? "gelu-quick-f16"      : "gelu-quick-f32";      break;
        case HTP_OP_UNARY_GELU_ERF:   op_type = is_f16 ? "gelu-erf-f16"        : "gelu-erf-f32";        break;
        case HTP_OP_UNARY_SOFTPLUS:   op_type = is_f16 ? "softplus-f16"        : "softplus-f32";        break;
        case HTP_OP_UNARY_TANH:       op_type = is_f16 ? "tanh-f16"            : "tanh-f32";            break;
        case HTP_OP_UNARY_ABS:        op_type = is_f16 ? "abs-f16"             : "abs-f32";             break;
        case HTP_OP_UNARY_LOG:        op_type = is_f16 ? "log-f16"             : "log-f32";             break;
        case HTP_OP_UNARY_RELU:       op_type = is_f16 ? "relu-f16"            : "relu-f32";            break;
        case HTP_OP_UNARY_STEP:       op_type = is_f16 ? "step-f16"            : "step-f32";            break;
        case HTP_OP_UNARY_CEIL:       op_type = is_f16 ? "ceil-f16"            : "ceil-f32";            break;
        case HTP_OP_UNARY_SGN:        op_type = is_f16 ? "sgn-f16"             : "sgn-f32";             break;
        case HTP_OP_L2_NORM:          op_type = is_f16 ? "l2norm-f16"          : "l2norm-f32";          break;
        case HTP_OP_TRI:              op_type = "tri-f32";                                              break;
        case HTP_OP_HADAMARD:         op_type = is_f16 ? "hadamard-f16"        : "hadamard-f32";        break;
        default:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_unary_kernel_params * kparams = (const struct htp_unary_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const bool has_weight = (octx->op == HTP_OP_RMS_NORM_MUL || octx->op == HTP_OP_RMS_NORM_MUL_ADD);
    const bool has_src2   = (octx->op == HTP_OP_RMS_NORM_MUL_ADD);

    const struct htp_tensor * src1 = has_weight ? octx->src[1] : NULL;
    const struct htp_tensor * src2 = has_src2   ? octx->src[2] : NULL;

    const size_t src0_elem_size = is_f16 ? sizeof(_Float16) : sizeof(float);
    const size_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? sizeof(_Float16) : sizeof(float);
    const size_t src1_elem_size = (has_weight && src1->type == HTP_TYPE_F16) ? sizeof(_Float16) : sizeof(float);
    const size_t src2_elem_size = (has_src2   && src2->type == HTP_TYPE_F16) ? sizeof(_Float16) : sizeof(float);

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t src0_data_row_size = src0->ne[0] * src0_elem_size;
    const size_t dst_data_row_size  = dst->ne[0]  * dst_elem_size;
    const size_t src1_data_row_size = has_weight ? (src1->ne[0] * src1_elem_size) : 0;
    const size_t src2_data_row_size = has_src2   ? (src2->ne[0] * src2_elem_size) : 0;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) dst_elem_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx = octx->n_hvx;

    const uint32_t col_tile = kparams->col_tile;

    bool broadcast_weight = kparams->broadcast_weight;

    if ((!col_tile && kparams->block == 0) || octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size, (size_t) kparams->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = NULL;
    octx->src1_spad.src = NULL;
    octx->dst_spad.src  = NULL;

    struct htp_unary_vtcm_layout L;
    htp_unary_vtcm_layout_build(&L, octx->op, src0_data_row_size, dst_data_row_size, src1_data_row_size, src2_data_row_size,
                                src0_elem_size, dst_elem_size, src1_elem_size, src2_elem_size,
                                col_tile, kparams->block, broadcast_weight, kparams->n_hvx);

    FARF(HIGH, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         L.src0_bytes * n_hvx, L.src1_bytes * n_hvx, L.dst_bytes * n_hvx);

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    struct htp_unary_context uctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .src0_nrows_per_hvx    = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div),
        .src0_nrows            = nrows,
        .row_start             = row_start,

        .data_src0             = src0->data,
        .data_src1             = has_weight ? src1->data : 0,
        .data_src2             = has_src2 ? src2->data : 0,
        .data_dst              = dst->data,

        .src0_data_row_size    = src0_data_row_size,
        .src1_data_row_size    = src1_data_row_size,
        .src2_data_row_size    = src2_data_row_size,
        .dst_data_row_size     = dst_data_row_size,

        .src0_row_size_aligned = L.src0_row_size_aligned,
        .src1_row_size_aligned = L.src1_row_size_aligned,
        .src2_row_size_aligned = L.src2_row_size_aligned,
        .dst_row_size_aligned  = L.dst_row_size_aligned,

        .src0_vtcm_half_size   = L.src0_bytes >> 1,
        .src1_vtcm_half_size   = has_weight ? (L.src1_bytes >> ((broadcast_weight && !col_tile) ? 0 : 1)) : 0,
        .src2_vtcm_half_size   = L.src2_bytes >> 1,
        .dst_vtcm_half_size    = L.dst_bytes >> 1,

        .block                 = kparams->block,
        .nc                    = src0->ne[0],
        .col_tile              = col_tile,
        .broadcast_weight      = broadcast_weight,
        .src0_f16              = is_f16,
        .src1_f16              = has_weight && (src1->type == HTP_TYPE_F16),
        .src2_f16              = has_src2   && (src2->type == HTP_TYPE_F16),
        .dst_f16               = (dst->type == HTP_TYPE_F16),

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_src1, L.src1_bytes > 0),
        .vtcm_src2             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_src2, L.src2_bytes > 0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst),

        .vtcm_src0_size_per_hvx = L.src0_bytes,
        .vtcm_src1_size_per_hvx = L.src1_bytes,
        .vtcm_src2_size_per_hvx = L.src2_bytes,
        .vtcm_dst_size_per_hvx  = L.dst_bytes,
    };

    FARF(HIGH, "%s: %s mode (col_tile %u)\n", op_type, col_tile ? "tiled" : "row-block", col_tile);

    work_queue_func_t task_func = col_tile ? unary_thread_tiled : unary_thread_row_block;
    void * compute_func = NULL;

    switch (octx->op) {
        case HTP_OP_RMS_NORM:
            task_func               = col_tile ? unary_thread_tiled_rms_norm : unary_thread_row_block;
            compute_func            = (void *) rms_norm_rows;
            uctx.rms_row_scale_fn   = is_f16 ? rms_norm_row_scale_f16   : rms_norm_row_scale_f32;
            uctx.rms_tile_sum_sq_fn = is_f16 ? rms_norm_tile_sum_sq_f16 : rms_norm_tile_sum_sq_f32;
            uctx.rms_apply_row_fn   = rms_norm_apply_row_table[is_f16][uctx.dst_f16];
            break;
        case HTP_OP_RMS_NORM_MUL:
            task_func               = col_tile ? unary_thread_tiled_rms_norm : unary_thread_rms_norm_mul;
            compute_func            = (void *) rms_norm_mul_rows;
            uctx.rms_row_scale_fn   = is_f16 ? rms_norm_row_scale_f16   : rms_norm_row_scale_f32;
            uctx.rms_tile_sum_sq_fn = is_f16 ? rms_norm_tile_sum_sq_f16 : rms_norm_tile_sum_sq_f32;
            uctx.rms_apply_row_fn   = rms_norm_mul_apply_row_table[is_f16][uctx.src1_f16][uctx.dst_f16];
            break;
        case HTP_OP_RMS_NORM_MUL_ADD:
            task_func               = col_tile ? unary_thread_tiled_rms_norm : unary_thread_rms_norm_mul_add;
            compute_func            = (void *) rms_norm_mul_add_rows;
            uctx.rms_row_scale_fn   = is_f16 ? rms_norm_row_scale_f16   : rms_norm_row_scale_f32;
            uctx.rms_tile_sum_sq_fn = is_f16 ? rms_norm_tile_sum_sq_f16 : rms_norm_tile_sum_sq_f32;
            uctx.rms_apply_row_fn   = rms_norm_mul_add_apply_row_table[is_f16][uctx.src1_f16][uctx.src2_f16][uctx.dst_f16];
            break;
        default:
            break;
    }

    if (!compute_func && is_f16) {
        switch (octx->op) {
            case HTP_OP_NORM:             compute_func = (void *) norm_f16;        break;
            case HTP_OP_SCALE:            compute_func = (void *) scale_f16;       break;
            case HTP_OP_CLAMP:            compute_func = (void *) clamp_f16;       break;
            case HTP_OP_LEAKY_RELU:       compute_func = (void *) leaky_relu_f16;  break;
            case HTP_OP_SQR:              compute_func = (void *) sqr_f16;         break;
            case HTP_OP_SQRT:             compute_func = (void *) sqrt_f16;        break;
            case HTP_OP_UNARY_NEG:        compute_func = (void *) neg_f16;         break;
            case HTP_OP_UNARY_EXP:        compute_func = (void *) exp_f16;         break;
            case HTP_OP_UNARY_SIGMOID:    compute_func = (void *) sigmoid_f16;     break;
            case HTP_OP_UNARY_SILU:       compute_func = (void *) silu_f16;        break;
            case HTP_OP_UNARY_GELU:       compute_func = (void *) gelu_f16;        break;
            case HTP_OP_UNARY_GELU_QUICK: compute_func = (void *) gelu_quick_f16;  break;
            case HTP_OP_UNARY_GELU_ERF:   compute_func = (void *) gelu_erf_f16;    break;
            case HTP_OP_UNARY_SOFTPLUS:   compute_func = (void *) softplus_f16;    break;
            case HTP_OP_UNARY_TANH:       compute_func = (void *) tanh_f16;        break;
            case HTP_OP_UNARY_ABS:        compute_func = (void *) abs_f16;         break;
            case HTP_OP_UNARY_LOG:        compute_func = (void *) log_f16;         break;
            case HTP_OP_UNARY_RELU:       compute_func = (void *) relu_f16;        break;
            case HTP_OP_UNARY_STEP:       compute_func = (void *) step_f16;        break;
            case HTP_OP_UNARY_CEIL:       compute_func = (void *) ceil_f16;        break;
            case HTP_OP_UNARY_SGN:        compute_func = (void *) sgn_f16;         break;
            case HTP_OP_L2_NORM:          compute_func = (void *) l2_norm_f16;     break;
            case HTP_OP_HADAMARD:         compute_func = (void *) hadamard_f16;    break;
            default:                      break;
        }
    } else if (!compute_func) {
        switch (octx->op) {
            case HTP_OP_NORM:             compute_func = (void *) norm_f32;        break;
            case HTP_OP_SCALE:            compute_func = (void *) scale_f32;       break;
            case HTP_OP_CLAMP:            compute_func = (void *) clamp_f32;       break;
            case HTP_OP_LEAKY_RELU:       compute_func = (void *) leaky_relu_f32;  break;
            case HTP_OP_SQR:              compute_func = (void *) sqr_f32;         break;
            case HTP_OP_SQRT:             compute_func = (void *) sqrt_f32;        break;
            case HTP_OP_UNARY_NEG:        compute_func = (void *) neg_f32;         break;
            case HTP_OP_UNARY_EXP:        compute_func = (void *) exp_f32;         break;
            case HTP_OP_UNARY_SIGMOID:    compute_func = (void *) sigmoid_f32;     break;
            case HTP_OP_UNARY_SILU:       compute_func = (void *) silu_f32;        break;
            case HTP_OP_UNARY_GELU:       compute_func = (void *) gelu_f32;        break;
            case HTP_OP_UNARY_GELU_QUICK: compute_func = (void *) gelu_quick_f32;  break;
            case HTP_OP_UNARY_GELU_ERF:   compute_func = (void *) gelu_erf_f32;    break;
            case HTP_OP_UNARY_SOFTPLUS:   compute_func = (void *) softplus_f32;    break;
            case HTP_OP_UNARY_TANH:       compute_func = (void *) tanh_f32;        break;
            case HTP_OP_UNARY_ABS:        compute_func = (void *) abs_f32;         break;
            case HTP_OP_UNARY_LOG:        compute_func = (void *) log_f32;         break;
            case HTP_OP_UNARY_RELU:       compute_func = (void *) relu_f32;        break;
            case HTP_OP_UNARY_STEP:       compute_func = (void *) step_f32;        break;
            case HTP_OP_UNARY_CEIL:       compute_func = (void *) ceil_f32;        break;
            case HTP_OP_UNARY_SGN:        compute_func = (void *) sgn_f32;         break;
            case HTP_OP_L2_NORM:          compute_func = (void *) l2_norm_f32;     break;
            case HTP_OP_HADAMARD:         compute_func = (void *) hadamard_f32;    break;
            case HTP_OP_TRI:
                task_func    = col_tile ? unary_thread_tiled_tri_f32 : unary_thread_tri_f32;
                compute_func = col_tile ? (void *) tri_apply_tile_f32 : (void *) tri_f32;
                break;
            default:                      break;
        }
    }

    if (!task_func || !compute_func) {
        FARF(ERROR, "execute_op_unary: task function is NULL for op %d\n", octx->op);
        return HTP_STATUS_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_queue_run(octx->ctx->hvx_queue, task_func, &uctx, n_hvx);

    return err;
}

int op_unary(struct htp_ops_context * octx) {
    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
        case HTP_TYPE_F16:
            return execute_op_unary(octx);

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
