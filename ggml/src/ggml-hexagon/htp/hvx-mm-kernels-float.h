#ifndef HVX_MM_KERNELS_FLOAT_H
#define HVX_MM_KERNELS_FLOAT_H

#include "hvx-utils.h"
#include "htp-tensor.h"

// Float activation copy/quantization kernels (VTCM -> VTCM)

static inline void quantize_f32_f32_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (void) tmp_data;
    for (uint32_t i = 0; i < nrows; ++i) {
        hvx_copy_f32_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }
}

static inline void quantize_f32_f16_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (void) tmp_data;
    for (uint32_t i = 0; i < nrows; ++i) {
        hvx_copy_f16_f32_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }
}

static inline void quantize_f16_f16_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (void) tmp_data;
    for (uint32_t i = 0; i < nrows; ++i) {
        hvx_copy_f16_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }
}

static inline void quantize_f16_f32_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (void) tmp_data;
    const size_t dst_valid_bytes = ne0 * sizeof(float);
    for (uint32_t i = 0; i < nrows; ++i) {
        hvx_copy_f32_f16_au(dst_data, src_data, ne0);
        if (dst_stride > dst_valid_bytes) {
            hvx_splat_f32_u(dst_data + dst_valid_bytes, 0.0f, (dst_stride - dst_valid_bytes) / sizeof(float));
        }

        dst_data += dst_stride;
        src_data += src_stride;
    }
}

// Float dot product kernels (HVX)

typedef void (*hvx_flat_store_1x1_fn_t)(void * restrict s, HVX_Vector rsum);
typedef void (*hvx_flat_store_2x1_fn_t)(void * restrict s, HVX_Vector rsum0, HVX_Vector rsum1);

static __attribute__((noinline)) void hvx_flat_store_1x1_f32(void * restrict s, HVX_Vector rsum) {
    hvx_vec_store_u(s, 4, hvx_vec_reduce_sum_f32(rsum));
}

static __attribute__((noinline)) void hvx_flat_store_1x1_f16(void * restrict s, HVX_Vector rsum) {
    HVX_Vector v_sum = hvx_vec_reduce_sum_f32(rsum);
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    v_sum = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v_sum));
    hvx_vec_store_u(s, 2, hvx_vec_f32_to_f16(v_sum, Q6_V_vzero()));
}

static __attribute__((noinline)) void hvx_flat_store_2x1_f32(void * restrict s, HVX_Vector rsum0, HVX_Vector rsum1) {
    hvx_vec_store_u(s, 8, hvx_vec_reduce_sum_f32x2(rsum0, rsum1));
}

static __attribute__((noinline)) void hvx_flat_store_2x1_f16(void * restrict s, HVX_Vector rsum0, HVX_Vector rsum1) {
    HVX_Vector v_sum = hvx_vec_reduce_sum_f32x2(rsum0, rsum1);
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    v_sum = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v_sum));
    hvx_vec_store_u(s, 4, hvx_vec_f32_to_f16(v_sum, Q6_V_vzero()));
}

static inline hvx_flat_store_1x1_fn_t resolve_hvx_flat_store_1x1_fn(uint32_t dst_elem_size) {
    return (dst_elem_size == sizeof(__fp16)) ? hvx_flat_store_1x1_f16 : hvx_flat_store_1x1_f32;
}

static inline hvx_flat_store_2x1_fn_t resolve_hvx_flat_store_2x1_fn(uint32_t dst_elem_size) {
    return (dst_elem_size == sizeof(__fp16)) ? hvx_flat_store_2x1_f16 : hvx_flat_store_2x1_f32;
}

#if __HVX_ARCH__ < 79
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

static inline void flat_vec_dot_f32_f32_aa_1x1(const uint32_t n, void * restrict s, const void * restrict vx, const void * restrict vy, hvx_flat_store_1x1_fn_t store_1x1) {
    const HVX_Vector * restrict x = (const HVX_Vector *) vx;
    const HVX_Vector * restrict y = (const HVX_Vector *) vy;

    uint32_t nvec = n / VLEN_FP32; // num full fp32 hvx vectors
    uint32_t nloe = n % VLEN_FP32; // leftover elements

    HVX_Vector rsum = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        HVX_Vector prod = HVX_OP_MUL_F32(x[i], y[i]);
        rsum = HVX_OP_ADD_F32(rsum, prod);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector x_sf = Q6_V_vand_QV(bmask, x[i]);
        HVX_Vector y_sf = Q6_V_vand_QV(bmask, y[i]);
        HVX_Vector prod = HVX_OP_MUL_F32(x_sf, y_sf);
        rsum = HVX_OP_ADD_F32(rsum, prod);
    }

    store_1x1(s, rsum);
}

static inline void flat_vec_dot_f32_f32_aa_2x1(const uint32_t n, void * restrict s0,
                                const void * restrict vx0, const void * restrict vx1,
                                const void * restrict vy0, hvx_flat_store_2x1_fn_t store_2x1) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx0;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) vx1;
    const HVX_Vector * restrict y  = (const HVX_Vector *) vy0;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_Vector rsum0 = Q6_V_vzero();
    HVX_Vector rsum1 = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector y_sf = y[i];
        HVX_Vector prod0 = HVX_OP_MUL_F32(x0[i], y_sf);
        HVX_Vector prod1 = HVX_OP_MUL_F32(x1[i], y_sf);
        rsum0 = HVX_OP_ADD_F32(rsum0, prod0);
        rsum1 = HVX_OP_ADD_F32(rsum1, prod1);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector y_sf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_Vector x0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector x1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_Vector prod0 = HVX_OP_MUL_F32(x0_sf, y_sf);
        HVX_Vector prod1 = HVX_OP_MUL_F32(x1_sf, y_sf);
        rsum0 = HVX_OP_ADD_F32(rsum0, prod0);
        rsum1 = HVX_OP_ADD_F32(rsum1, prod1);
    }

    store_2x1(s0, rsum0, rsum1);
}

static inline void flat_vec_dot_f32_f32_aa_2x2(const uint32_t n, void * restrict s0, void * restrict s1,
                                const void * restrict vx0, const void * restrict vx1,
                                const void * restrict vy0, const void * restrict vy1, hvx_flat_store_2x1_fn_t store_2x1) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx0;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) vx1;
    const HVX_Vector * restrict y0 = (const HVX_Vector *) vy0;
    const HVX_Vector * restrict y1 = (const HVX_Vector *) vy1;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_Vector r0_c0_sum = Q6_V_vzero();
    HVX_Vector r0_c1_sum = Q6_V_vzero();
    HVX_Vector r1_c0_sum = Q6_V_vzero();
    HVX_Vector r1_c1_sum = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector r0_sf = x0[i];
        HVX_Vector r1_sf = x1[i];
        HVX_Vector c0_sf = y0[i];
        HVX_Vector c1_sf = y1[i];

        r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);

        HVX_Vector r0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector r1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_Vector c0_sf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_Vector c1_sf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    store_2x1(s0, r0_c0_sum, r1_c0_sum);
    store_2x1(s1, r0_c1_sum, r1_c1_sum);
}

static inline void flat_vec_dot_bf16_f32_aa_1x1(const uint32_t n, void * restrict s, const void * restrict vx, const void * restrict vy, hvx_flat_store_1x1_fn_t store_1x1) {
    const HVX_Vector * restrict x = (const HVX_Vector *) vx;
    const HVX_Vector * restrict y = (const HVX_Vector *) vy;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_Vector rsum0 = Q6_V_vzero();
    HVX_Vector rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector v_bf16 = x[i];
        HVX_VectorPair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_Vector w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_Vector w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y[2 * i + 0]));
        rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y[2 * i + 1]));
    }

    if (rem) {
        HVX_VectorPred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_Vector v_bf16 = Q6_V_vand_QV(bmask, x[i]);
        HVX_VectorPair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_Vector w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_Vector w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        if (rem <= 32) {
            HVX_VectorPred a_mask = Q6_Q_vsetq2_R(rem * 4);
            HVX_Vector y0 = Q6_V_vand_QV(a_mask, y[2 * i + 0]);
            rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));
        } else {
            HVX_Vector y0 = y[2 * i + 0];
            rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));

            HVX_VectorPred a_mask = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_Vector y1 = Q6_V_vand_QV(a_mask, y[2 * i + 1]);
            rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y1));
        }
    }

    HVX_Vector rsum = HVX_OP_ADD_F32(rsum0, rsum1);
    store_1x1(s, rsum);
}

static inline void flat_vec_dot_bf16_f32_aa_2x1(const uint32_t n, void * restrict s0,
                                const void * restrict vx0, const void * restrict vx1,
                                const void * restrict vy0, hvx_flat_store_2x1_fn_t store_2x1) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx0;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) vx1;
    const HVX_Vector * restrict y  = (const HVX_Vector *) vy0;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_Vector rsum0 = Q6_V_vzero();
    HVX_Vector rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_VectorPair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_VectorPair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_Vector w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_Vector w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_Vector w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_Vector w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_Vector y0 = y[2 * i + 0];
        HVX_Vector y1 = y[2 * i + 1];

        rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
        rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
        rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
    }

    if (rem) {
        HVX_VectorPred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_VectorPair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_VectorPair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_Vector w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_Vector w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_Vector w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_Vector w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        if (rem <= 32) {
            HVX_VectorPred a_mask = Q6_Q_vsetq2_R(rem * 4);
            HVX_Vector y0 = Q6_V_vand_QV(a_mask, y[2 * i + 0]);
            rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        } else {
            HVX_Vector y0 = y[2 * i + 0];
            rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));

            HVX_VectorPred a_mask = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_Vector y1 = Q6_V_vand_QV(a_mask, y[2 * i + 1]);
            rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
            rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
        }
    }

    store_2x1(s0, rsum0, rsum1);
}

static inline void flat_vec_dot_bf16_f32_aa_2x2(const uint32_t n, void * restrict s0, void * restrict s1,
                                const void * restrict vx0, const void * restrict vx1,
                                const void * restrict vy0, const void * restrict vy1, hvx_flat_store_2x1_fn_t store_2x1) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx0;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) vx1;
    const HVX_Vector * restrict y0 = (const HVX_Vector *) vy0;
    const HVX_Vector * restrict y1 = (const HVX_Vector *) vy1;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_Vector r0_c0_sum = Q6_V_vzero();
    HVX_Vector r0_c1_sum = Q6_V_vzero();
    HVX_Vector r1_c0_sum = Q6_V_vzero();
    HVX_Vector r1_c1_sum = Q6_V_vzero();

    uint32_t i = 0;
    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_VectorPair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_VectorPair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_Vector w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_Vector w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_Vector w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_Vector w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_Vector c0_0 = y0[2 * i + 0];
        HVX_Vector c0_1 = y0[2 * i + 1];
        HVX_Vector c1_0 = y1[2 * i + 0];
        HVX_Vector c1_1 = y1[2 * i + 1];

        r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(w00_f32, c0_0));
        r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(w00_f32, c1_0));
        r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(w10_f32, c0_0));
        r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(w10_f32, c1_0));

        r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(w01_f32, c0_1));
        r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(w01_f32, c1_1));
        r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(w11_f32, c0_1));
        r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(w11_f32, c1_1));
    }

    if (rem) {
        HVX_VectorPred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_VectorPair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_VectorPair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_Vector w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_Vector w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_Vector w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_Vector w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        if (rem <= 32) {
            HVX_VectorPred a_mask = Q6_Q_vsetq2_R(rem * 4);
            HVX_Vector c0_0 = Q6_V_vand_QV(a_mask, y0[2 * i + 0]);
            HVX_Vector c1_0 = Q6_V_vand_QV(a_mask, y1[2 * i + 0]);
            r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(w10_f32, c1_0));
        } else {
            HVX_Vector c0_0 = y0[2 * i + 0];
            HVX_Vector c1_0 = y1[2 * i + 0];
            r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(w10_f32, c1_0));

            HVX_VectorPred a_mask = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_Vector c0_1 = Q6_V_vand_QV(a_mask, y0[2 * i + 1]);
            HVX_Vector c1_1 = Q6_V_vand_QV(a_mask, y1[2 * i + 1]);
            r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(w01_f32, c0_1));
            r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(w01_f32, c1_1));
            r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(w11_f32, c0_1));
            r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(w11_f32, c1_1));
        }
    }

    store_2x1(s0, r0_c0_sum, r1_c0_sum);
    store_2x1(s1, r0_c1_sum, r1_c1_sum);
}

#undef HVX_OP_ADD_F32
#undef HVX_OP_MUL_F32

static inline void flat_vec_dot_f16_f16_aa_1x1(const uint32_t n, void * restrict s, const void * restrict vx, const void * restrict vy, hvx_flat_store_1x1_fn_t store_1x1) {
    const HVX_Vector * restrict x = (const HVX_Vector *) vx;
    const HVX_Vector * restrict y = (const HVX_Vector *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_VectorPair rsum_p = Q6_W_vzero();

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x[i], y[i]);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, y[i]);
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_Vector rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p)));
    store_1x1(s, rsum);
}

static inline void flat_vec_dot_f16_f16_aa_2x1(const uint32_t n, void * restrict s0,
                                const void * restrict vx0, const void * restrict vx1,
                                const void * restrict vy0, hvx_flat_store_2x1_fn_t store_2x1) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx0;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) vx1;
    const HVX_Vector * restrict y  = (const HVX_Vector *) vy0;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_VectorPair rsum0_p = Q6_W_vzero();
    HVX_VectorPair rsum1_p = Q6_W_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf = y[i];
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0[i], y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1[i], y_hf);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector y_hf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_Vector x0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector x1_hf = Q6_V_vand_QV(bmask, x1[i]);
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
    }

    HVX_Vector rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p)));
    HVX_Vector rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p)));
    store_2x1(s0, rsum0, rsum1);
}

static inline void flat_vec_dot_f16_f16_aa_2x2(const uint32_t n, void * restrict s0, void * restrict s1,
                                const void * restrict vx0, const void * restrict vx1,
                                const void * restrict vy0, const void * restrict vy1, hvx_flat_store_2x1_fn_t store_2x1) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx0;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) vx1;
    const HVX_Vector * restrict y0 = (const HVX_Vector *) vy0;
    const HVX_Vector * restrict y1 = (const HVX_Vector *) vy1;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    // Row sums (sf) - 4 accumulators for 2x2 tile
    HVX_VectorPair r0_c0_sum_p = Q6_W_vzero();
    HVX_VectorPair r0_c1_sum_p = Q6_W_vzero();
    HVX_VectorPair r1_c0_sum_p = Q6_W_vzero();
    HVX_VectorPair r1_c1_sum_p = Q6_W_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector r0_hf = x0[i];
        HVX_Vector r1_hf = x1[i];
        HVX_Vector c0_hf = y0[i];
        HVX_Vector c1_hf = y1[i];

        // Compute 4 dot products: r0xc0, r0xc1, r1xc0, r1xc1
        r0_c0_sum_p = hvx_vec_mpyacc_f32_f16(r0_c0_sum_p, r0_hf, c0_hf);
        r0_c1_sum_p = hvx_vec_mpyacc_f32_f16(r0_c1_sum_p, r0_hf, c1_hf);
        r1_c0_sum_p = hvx_vec_mpyacc_f32_f16(r1_c0_sum_p, r1_hf, c0_hf);
        r1_c1_sum_p = hvx_vec_mpyacc_f32_f16(r1_c1_sum_p, r1_hf, c1_hf);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);

        HVX_Vector r0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector r1_hf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_Vector c0_hf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_Vector c1_hf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_sum_p = hvx_vec_mpyacc_f32_f16(r0_c0_sum_p, r0_hf, c0_hf);
        r0_c1_sum_p = hvx_vec_mpyacc_f32_f16(r0_c1_sum_p, r0_hf, c1_hf);
        r1_c0_sum_p = hvx_vec_mpyacc_f32_f16(r1_c0_sum_p, r1_hf, c0_hf);
        r1_c1_sum_p = hvx_vec_mpyacc_f32_f16(r1_c1_sum_p, r1_hf, c1_hf);
    }

    HVX_Vector r0_c0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c0_sum_p), Q6_V_hi_W(r0_c0_sum_p)));
    HVX_Vector r0_c1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c1_sum_p), Q6_V_hi_W(r0_c1_sum_p)));
    HVX_Vector r1_c0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c0_sum_p), Q6_V_hi_W(r1_c0_sum_p)));
    HVX_Vector r1_c1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c1_sum_p), Q6_V_hi_W(r1_c1_sum_p)));

    store_2x1(s0, r0_c0_sum, r1_c0_sum);
    store_2x1(s1, r0_c1_sum, r1_c1_sum);
}

typedef void (*hvx_tensor_add_row_fn_t)(void * restrict dst_ptr, const void * restrict bias_ptr, uint32_t n);

static __attribute__((noinline)) void hvx_tensor_add_row_f32_f32(void * restrict dst_ptr, const void * restrict bias_ptr, uint32_t n) {
    float * restrict       d = (float *) dst_ptr;
    const float * restrict b = (const float *) bias_ptr;
    uint32_t c = 0;
    for (; c + 32 <= n; c += 32) {
        hvx_vmemu(d + c) = hvx_vec_add_f32_f32(hvx_vmemu(d + c), hvx_vmemu(b + c));
    }
    if (c < n) {
        const uint32_t rem = n - c;
        HVX_Vector v_sum = hvx_vec_add_f32_f32(hvx_vmemu(d + c), hvx_vmemu(b + c));
        hvx_vec_store_u(d + c, rem * sizeof(float), v_sum);
    }
}

static __attribute__((noinline)) void hvx_tensor_add_row_f32_f16(void * restrict dst_ptr, const void * restrict bias_ptr, uint32_t n) {
    float * restrict        d = (float *) dst_ptr;
    const __fp16 * restrict b = (const __fp16 *) bias_ptr;
    uint32_t c = 0;
    for (; c + 64 <= n; c += 64) {
        HVX_VectorPair vp_b = hvx_vec_f16_to_f32(hvx_vmemu(b + c));
        hvx_vmemu(d + c +  0) = hvx_vec_add_f32_f32(hvx_vmemu(d + c +  0), Q6_V_lo_W(vp_b));
        hvx_vmemu(d + c + 32) = hvx_vec_add_f32_f32(hvx_vmemu(d + c + 32), Q6_V_hi_W(vp_b));
    }
    if (c + 32 <= n) {
        HVX_Vector v_b = Q6_V_lo_W(hvx_vec_f16_to_f32(hvx_vmemu(b + c)));
        hvx_vmemu(d + c) = hvx_vec_add_f32_f32(hvx_vmemu(d + c), v_b);
        c += 32;
    }
    if (c < n) {
        const uint32_t rem = n - c;
        HVX_Vector v_b = Q6_V_lo_W(hvx_vec_f16_to_f32(hvx_vmemu(b + c)));
        HVX_Vector v_sum = hvx_vec_add_f32_f32(hvx_vmemu(d + c), v_b);
        hvx_vec_store_u(d + c, rem * sizeof(float), v_sum);
    }
}

static __attribute__((noinline)) void hvx_tensor_add_row_f16_f32(void * restrict dst_ptr, const void * restrict bias_ptr, uint32_t n) {
    __fp16 * restrict      d = (__fp16 *) dst_ptr;
    const float * restrict b = (const float *) bias_ptr;
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    uint32_t c = 0;
    for (; c + 64 <= n; c += 64) {
        HVX_VectorPair vp_d = hvx_vec_f16_to_f32(hvx_vmemu(d + c));
        HVX_Vector v0 = hvx_vec_add_f32_f32(Q6_V_lo_W(vp_d), hvx_vmemu(b + c +  0));
        HVX_Vector v1 = hvx_vec_add_f32_f32(Q6_V_hi_W(vp_d), hvx_vmemu(b + c + 32));
        v0 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v0));
        v1 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v1));
        hvx_vmemu(d + c) = hvx_vec_f32_to_f16(v0, v1);
    }
    if (c < n) {
        const uint32_t rem = n - c;
        HVX_VectorPair vp_d = hvx_vec_f16_to_f32(hvx_vmemu(d + c));
        HVX_Vector v0 = hvx_vec_add_f32_f32(Q6_V_lo_W(vp_d), hvx_vmemu(b + c));
        HVX_Vector v1 = (rem > 32) ? hvx_vec_add_f32_f32(Q6_V_hi_W(vp_d), hvx_vmemu(b + c + 32)) : Q6_V_vzero();
        v0 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v0));
        v1 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v1));
        hvx_vec_store_u(d + c, rem * sizeof(__fp16), hvx_vec_f32_to_f16(v0, v1));
    }
}

static __attribute__((noinline)) void hvx_tensor_add_row_f16_f16(void * restrict dst_ptr, const void * restrict bias_ptr, uint32_t n) {
    __fp16 * restrict        d = (__fp16 *) dst_ptr;
    const __fp16 * restrict  b = (const __fp16 *) bias_ptr;
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    uint32_t c = 0;
    for (; c + 64 <= n; c += 64) {
        HVX_VectorPair vp_d = hvx_vec_f16_to_f32(hvx_vmemu(d + c));
        HVX_VectorPair vp_b = hvx_vec_f16_to_f32(hvx_vmemu(b + c));
        HVX_Vector v0 = hvx_vec_add_f32_f32(Q6_V_lo_W(vp_d), Q6_V_lo_W(vp_b));
        HVX_Vector v1 = hvx_vec_add_f32_f32(Q6_V_hi_W(vp_d), Q6_V_hi_W(vp_b));
        v0 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v0));
        v1 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v1));
        hvx_vmemu(d + c) = hvx_vec_f32_to_f16(v0, v1);
    }
    if (c < n) {
        const uint32_t rem = n - c;
        HVX_VectorPair vp_d = hvx_vec_f16_to_f32(hvx_vmemu(d + c));
        HVX_VectorPair vp_b = hvx_vec_f16_to_f32(hvx_vmemu(b + c));
        HVX_Vector v0 = hvx_vec_add_f32_f32(Q6_V_lo_W(vp_d), Q6_V_lo_W(vp_b));
        HVX_Vector v1 = (rem > 32) ? hvx_vec_add_f32_f32(Q6_V_hi_W(vp_d), Q6_V_hi_W(vp_b)) : Q6_V_vzero();
        v0 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v0));
        v1 = Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v1));
        hvx_vec_store_u(d + c, rem * sizeof(__fp16), hvx_vec_f32_to_f16(v0, v1));
    }
}

static inline hvx_tensor_add_row_fn_t resolve_hvx_tensor_add_row_fn(uint32_t dst_elem_size, uint32_t bias_elem_size) {
    if (dst_elem_size == sizeof(__fp16)) {
        return (bias_elem_size == sizeof(__fp16)) ? hvx_tensor_add_row_f16_f16 : hvx_tensor_add_row_f16_f32;
    }
    return (bias_elem_size == sizeof(__fp16)) ? hvx_tensor_add_row_f32_f16 : hvx_tensor_add_row_f32_f32;
}

static inline void hvx_tensor_add_f32_grid(
    const struct htp_tensor * restrict dst,
    const struct htp_tensor * restrict bias,
    uint32_t start_row,
    uint32_t end_row,
    uint32_t start_col,
    uint32_t end_col,
    const struct fastdiv_values * div_ne11_12,
    const struct fastdiv_values * div_ne11
) {
    if (start_row >= end_row || start_col >= end_col) return;
    const uint32_t nb1 = dst->nb[1];
    const uint32_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const uint32_t bias_elem_size = (bias->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const hvx_tensor_add_row_fn_t add_row_fn = resolve_hvx_tensor_add_row_fn(dst_elem_size, bias_elem_size);
    const uint32_t n_cols = end_col - start_col;
    const size_t dst_col_off  = (size_t) start_col * dst_elem_size;
    const size_t bias_col_off = (size_t) start_col * bias_elem_size;

    const uint32_t ne11 = dst->ne[1];
    const uint32_t ne12 = dst->ne[2];
    const uint32_t ne11_12 = ne11 * ne12;

    const size_t bias_nb1 = (bias->ne[1] == 1) ? 0 : bias->nb[1];
    const size_t bias_nb2 = (bias->ne[2] == 1) ? 0 : bias->nb[2];
    const size_t bias_nb3 = (bias->ne[3] == 1) ? 0 : bias->nb[3];

    for (uint32_t r = start_row; r < end_row; r++) {
        uint32_t i13 = fastdiv(r, div_ne11_12);
        uint32_t i12 = fastdiv(r - i13 * ne11_12, div_ne11);
        uint32_t i11 = r - i13 * ne11_12 - i12 * ne11;

        uint8_t * dst_ptr = (uint8_t *) dst->data +
                            (size_t) i11 * nb1 + (size_t) i12 * dst->nb[2] + (size_t) i13 * dst->nb[3] + dst_col_off;
        const uint8_t * bias_ptr = (const uint8_t *) bias->data +
                                   (size_t) i11 * bias_nb1 + (size_t) i12 * bias_nb2 + (size_t) i13 * bias_nb3 + bias_col_off;

        add_row_fn(dst_ptr, bias_ptr, n_cols);
    }
}

#endif // HVX_MM_KERNELS_FLOAT_H
