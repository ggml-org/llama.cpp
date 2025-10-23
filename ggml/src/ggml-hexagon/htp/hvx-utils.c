#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif

#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_ps.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hvx-utils.h"

#define htp_binary_ops_preamble                                                                                \
    int step_of_4 = num_elems >> 7;                                                                            \
    int step_of_2 = (num_elems - step_of_4 * VLEN_FP32 * 4) >> 6;                                              \
    int step_of_1 = (num_elems - step_of_4 * VLEN_FP32 * 4 - step_of_2 * VLEN_FP32 * 2) >> 5;                  \
    int remaining = num_elems - step_of_4 * VLEN_FP32 * 4 - step_of_2 * VLEN_FP32 * 2 - step_of_1 * VLEN_FP32; \
                                                                                                               \
    const uint8_t * restrict src0_curr = src0;                                                                 \
    const uint8_t * restrict src1_curr = src1;                                                                 \
    uint8_t * restrict dst_curr        = dst;

void hvx_mul_f32(const uint8_t * restrict src0,
                 const uint8_t * restrict src1,
                 uint8_t * restrict dst,
                 const int num_elems) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if ((0 == htp_is_aligned((void *) src0, VLEN)) || (0 == htp_is_aligned((void *) src1, VLEN)) ||
        (0 == htp_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_mul_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_mul_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    if (0 == unaligned_loop) {
        HVX_Vector * restrict vec_in1 = (HVX_Vector *) src0;
        HVX_Vector * restrict vec_in2 = (HVX_Vector *) src1;
        HVX_Vector * restrict vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(*vec_in1++, *vec_in2++);
            *vec_out++   = Q6_Vsf_equals_Vqf32(v);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in1 = *(HVX_UVector *) (src0 + i * SIZEOF_FP32);
            HVX_Vector in2 = *(HVX_UVector *) (src1 + i * SIZEOF_FP32);

            HVX_Vector out = Q6_Vqf32_vmpy_VsfVsf(in1, in2);

            *(HVX_UVector *) (dst + i * SIZEOF_FP32) = Q6_Vsf_equals_Vqf32(out);
        }
    }

    if (left_over > 0) {
        const float * src0f = (const float *) src0 + num_elems_whole;
        const float * src1f = (const float *) src1 + num_elems_whole;
        float *       dstf  = (float *) dst + num_elems_whole;

        HVX_Vector in1 = *(HVX_UVector *) src0f;
        HVX_Vector in2 = *(HVX_UVector *) src1f;

        HVX_Vector out = Q6_Vqf32_vmpy_VsfVsf(in1, in2);
        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(out));
    }
}

void hvx_mul_f32_opt(const uint8_t * restrict src0,
                     const uint8_t * restrict src1,
                     uint8_t * restrict dst,
                     const int num_elems) {
    htp_binary_ops_preamble;

    for (int i = 0; i < step_of_4; i++) {
        HVX_Vector v1a = *(HVX_Vector *) src0_curr;

        HVX_Vector v1b = *(HVX_Vector *) src1_curr;

        HVX_Vector v2a = *(HVX_Vector *) (src0_curr + VLEN);

        HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(v1a, v1b);

        HVX_Vector v2b = *(HVX_Vector *) (src1_curr + VLEN);

        HVX_Vector v3a = *(HVX_Vector *) (src0_curr + 2 * VLEN);

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v2a, v2b);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v1);

        HVX_Vector v3b = *(HVX_Vector *) (src1_curr + 2 * VLEN);

        HVX_Vector v4a = *(HVX_Vector *) (src0_curr + 3 * VLEN);

        src0_curr += 4 * VLEN;

        HVX_Vector v3 = Q6_Vqf32_vmpy_VsfVsf(v3a, v3b);

        *(HVX_Vector *) (dst_curr + VLEN) = Q6_Vsf_equals_Vqf32(v2);

        HVX_Vector v4b = *(HVX_Vector *) (src1_curr + 3 * VLEN);

        *(HVX_Vector *) (dst_curr + 2 * VLEN) = Q6_Vsf_equals_Vqf32(v3);

        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v4a, v4b);

        src1_curr += 4 * VLEN;

        *(HVX_Vector *) (dst_curr + 3 * VLEN) = Q6_Vsf_equals_Vqf32(v4);

        dst_curr += 4 * VLEN;
    }

    for (int i = 0; i < step_of_2; i++) {
        HVX_Vector v1a = *(HVX_Vector *) src0_curr;

        HVX_Vector v1b = *(HVX_Vector *) src1_curr;

        HVX_Vector v2a = *(HVX_Vector *) (src0_curr + VLEN);

        HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(v1a, v1b);

        HVX_Vector v2b = *(HVX_Vector *) (src1_curr + VLEN);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v1);

        src0_curr += 2 * VLEN;

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v2a, v2b);

        src1_curr += 2 * VLEN;

        *(HVX_Vector *) (dst_curr + VLEN) = Q6_Vsf_equals_Vqf32(v2);

        dst_curr += 2 * VLEN;
    }

    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector va = *(HVX_Vector *) src0_curr;

        src0_curr += VLEN;

        HVX_Vector vb = *(HVX_Vector *) src1_curr;

        src1_curr += VLEN;

        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(va, vb);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v);

        dst_curr += VLEN;
    }

    if (remaining > 0) {
        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *) src0_curr, *(HVX_Vector *) src1_curr);
        hvx_vec_store_u((void *) dst_curr, remaining * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(v));
    }
}

void hvx_mul_mul_f32_opt(const uint8_t * restrict src0,
                         const uint8_t * restrict src1,
                         const uint8_t * restrict src2,
                         uint8_t * restrict dst,
                         const int num_elems) {
    const uint8_t * restrict src0_curr = src0;
    const uint8_t * restrict src1_curr = src1;
    const uint8_t * restrict src2_curr = src2;
    uint8_t * restrict dst_curr        = dst;

    int step_of_2 = num_elems >> 6;
    int step_of_1 = (num_elems - step_of_2 * VLEN_FP32 * 2) >> 5;
    int remaining = num_elems - step_of_2 * VLEN_FP32 * 2 - step_of_1 * VLEN_FP32;

    for (int i = 0; i < step_of_2; i++) {
        HVX_Vector v1a = *(HVX_Vector *) src0_curr;
        HVX_Vector v1b = *(HVX_Vector *) src1_curr;
        HVX_Vector v1c = *(HVX_Vector *) src2_curr;

        HVX_Vector v2a = *(HVX_Vector *) (src0_curr + VLEN);

        HVX_Vector v1_ = Q6_Vqf32_vmpy_VsfVsf(v1a, v1b);
        HVX_Vector v1  = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v1_), v1c);

        HVX_Vector v2b = *(HVX_Vector *) (src1_curr + VLEN);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v1);

        HVX_Vector v2c = *(HVX_Vector *) (src2_curr + VLEN);

        src0_curr += 2 * VLEN;

        HVX_Vector v2_ = Q6_Vqf32_vmpy_VsfVsf(v2a, v2b);
        HVX_Vector v2  = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2_), v2c);

        src1_curr += 2 * VLEN;
        src2_curr += 2 * VLEN;

        *(HVX_Vector *) (dst_curr + VLEN) = Q6_Vsf_equals_Vqf32(v2);

        dst_curr += 2 * VLEN;
    }
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector va = *(HVX_Vector *) src0_curr;
        src0_curr += VLEN;

        HVX_Vector vb = *(HVX_Vector *) src1_curr;
        src1_curr += VLEN;

        HVX_Vector vc = *(HVX_Vector *) src2_curr;
        src2_curr += VLEN;

        HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(va, vb);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v1), vc);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v2);
        dst_curr += VLEN;
    }
    if (remaining > 0) {
        HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *) src0_curr, *(HVX_Vector *) src1_curr);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v1), *(HVX_Vector *) src2_curr);
        hvx_vec_store_u((void *) dst_curr, remaining * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(v2));
    }
}

void hvx_add_f32(const uint8_t * restrict src0,
                 const uint8_t * restrict src1,
                 uint8_t * restrict dst,
                 const int num_elems) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if ((0 == htp_is_aligned((void *) src0, VLEN)) || (0 == htp_is_aligned((void *) src1, VLEN)) ||
        (0 == htp_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_add_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_add_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    if (0 == unaligned_loop) {
        HVX_Vector * restrict vec_in1 = (HVX_Vector *) src0;
        HVX_Vector * restrict vec_in2 = (HVX_Vector *) src1;
        HVX_Vector * restrict vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector v = Q6_Vqf32_vadd_VsfVsf(*vec_in1++, *vec_in2++);
            *vec_out++   = Q6_Vsf_equals_Vqf32(v);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in1 = *(HVX_UVector *) (src0 + i * SIZEOF_FP32);
            HVX_Vector in2 = *(HVX_UVector *) (src1 + i * SIZEOF_FP32);

            HVX_Vector out = Q6_Vqf32_vadd_VsfVsf(in1, in2);

            *(HVX_UVector *) (dst + i * SIZEOF_FP32) = Q6_Vsf_equals_Vqf32(out);
        }
    }

    if (left_over > 0) {
        const float * src0f = (const float *) src0 + num_elems_whole;
        const float * src1f = (const float *) src1 + num_elems_whole;
        float *       dstf  = (float *) dst + num_elems_whole;

        HVX_Vector in1 = *(HVX_UVector *) src0f;
        HVX_Vector in2 = *(HVX_UVector *) src1f;

        HVX_Vector out = Q6_Vqf32_vadd_VsfVsf(in1, in2);
        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(out));
    }
}

void hvx_add_f32_opt(const uint8_t * restrict src0,
                     const uint8_t * restrict src1,
                     uint8_t * restrict dst,
                     const int num_elems) {
    htp_binary_ops_preamble;

    for (int i = 0; i < step_of_4; i++) {
        HVX_Vector v1a = *(HVX_Vector *) src0_curr;

        HVX_Vector v1b = *(HVX_Vector *) src1_curr;

        HVX_Vector v2a = *(HVX_Vector *) (src0_curr + VLEN);

        HVX_Vector v1 = Q6_Vqf32_vadd_VsfVsf(v1a, v1b);

        HVX_Vector v2b = *(HVX_Vector *) (src1_curr + VLEN);

        HVX_Vector v3a = *(HVX_Vector *) (src0_curr + 2 * VLEN);

        HVX_Vector v2 = Q6_Vqf32_vadd_VsfVsf(v2a, v2b);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v1);

        HVX_Vector v3b = *(HVX_Vector *) (src1_curr + 2 * VLEN);

        HVX_Vector v4a = *(HVX_Vector *) (src0_curr + 3 * VLEN);

        src0_curr += 4 * VLEN;

        HVX_Vector v3 = Q6_Vqf32_vadd_VsfVsf(v3a, v3b);

        *(HVX_Vector *) (dst_curr + VLEN) = Q6_Vsf_equals_Vqf32(v2);

        HVX_Vector v4b = *(HVX_Vector *) (src1_curr + 3 * VLEN);

        *(HVX_Vector *) (dst_curr + 2 * VLEN) = Q6_Vsf_equals_Vqf32(v3);

        HVX_Vector v4 = Q6_Vqf32_vadd_VsfVsf(v4a, v4b);

        src1_curr += 4 * VLEN;

        *(HVX_Vector *) (dst_curr + 3 * VLEN) = Q6_Vsf_equals_Vqf32(v4);

        dst_curr += 4 * VLEN;
    }
    for (int i = 0; i < step_of_2; i++) {
        HVX_Vector v1a = *(HVX_Vector *) src0_curr;

        HVX_Vector v1b = *(HVX_Vector *) src1_curr;

        HVX_Vector v2a = *(HVX_Vector *) (src0_curr + VLEN);

        HVX_Vector v1 = Q6_Vqf32_vadd_VsfVsf(v1a, v1b);

        HVX_Vector v2b = *(HVX_Vector *) (src1_curr + VLEN);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v1);

        src0_curr += 2 * VLEN;

        HVX_Vector v2 = Q6_Vqf32_vadd_VsfVsf(v2a, v2b);

        src1_curr += 2 * VLEN;

        *(HVX_Vector *) (dst_curr + VLEN) = Q6_Vsf_equals_Vqf32(v2);

        dst_curr += 2 * VLEN;
    }
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector va = *(HVX_Vector *) src0_curr;

        src0_curr += VLEN;

        HVX_Vector vb = *(HVX_Vector *) src1_curr;

        src1_curr += VLEN;

        HVX_Vector v = Q6_Vqf32_vadd_VsfVsf(va, vb);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v);

        dst_curr += VLEN;
    }
    if (remaining > 0) {
        HVX_Vector v = Q6_Vqf32_vadd_VsfVsf(*(HVX_Vector *) src0_curr, *(HVX_Vector *) src1_curr);
        hvx_vec_store_u((void *) dst_curr, remaining * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(v));
    }
}

void hvx_add_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems) {
    size_t left_over       = num_elems & (VLEN_FP32 - 1);
    size_t num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if ((0 == htp_is_aligned((void *) src, VLEN)) || (0 == htp_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_add_scalar_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_add_scalar_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    HVX_Vector val_vec = hvx_vec_splat_fp32(val);

    if (0 == unaligned_loop) {
        HVX_Vector * restrict vec_in1 = (HVX_Vector *) src;
        HVX_Vector * restrict vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector v = Q6_Vqf32_vadd_VsfVsf(*vec_in1++, val_vec);
            *vec_out++   = Q6_Vsf_equals_Vqf32(v);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            HVX_Vector out = Q6_Vqf32_vadd_VsfVsf(in, val_vec);

            *(HVX_UVector *) (dst + i * SIZEOF_FP32) = Q6_Vsf_equals_Vqf32(out);
        }
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_Vector in = *(HVX_UVector *) srcf;

        HVX_Vector out = Q6_Vqf32_vadd_VsfVsf(in, val_vec);
        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(out));
    }
}

void hvx_mul_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems) {
    size_t left_over       = num_elems & (VLEN_FP32 - 1);
    size_t num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if ((0 == htp_is_aligned((void *) src, VLEN)) || (0 == htp_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_mul_scalar_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_mul_scalar_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    HVX_Vector val_vec = hvx_vec_splat_fp32(val);

    if (0 == unaligned_loop) {
        HVX_Vector * restrict vec_in1 = (HVX_Vector *) src;
        HVX_Vector * restrict vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(*vec_in1++, val_vec);
            *vec_out++   = Q6_Vsf_equals_Vqf32(v);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            HVX_Vector out = Q6_Vqf32_vmpy_VsfVsf(in, val_vec);

            *(HVX_UVector *) (dst + i * SIZEOF_FP32) = Q6_Vsf_equals_Vqf32(out);
        }
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_Vector in = *(HVX_UVector *) srcf;

        HVX_Vector out = Q6_Vqf32_vmpy_VsfVsf(in, val_vec);
        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(out));
    }
}

void hvx_sub_f32(const uint8_t * restrict src0,
                 const uint8_t * restrict src1,
                 uint8_t * restrict dst,
                 const int num_elems) {
    size_t left_over       = num_elems & (VLEN_FP32 - 1);
    size_t num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if ((0 == htp_is_aligned((void *) src0, VLEN)) || (0 == htp_is_aligned((void *) src1, VLEN)) ||
        (0 == htp_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_sub_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_sub_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    if (0 == unaligned_loop) {
        HVX_Vector * restrict vec_in1 = (HVX_Vector *) src0;
        HVX_Vector * restrict vec_in2 = (HVX_Vector *) src1;
        HVX_Vector * restrict vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector v = Q6_Vqf32_vsub_VsfVsf(*vec_in1++, *vec_in2++);
            *vec_out++   = Q6_Vsf_equals_Vqf32(v);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in1 = *(HVX_UVector *) (src0 + i * SIZEOF_FP32);
            HVX_Vector in2 = *(HVX_UVector *) (src1 + i * SIZEOF_FP32);

            HVX_Vector out = Q6_Vqf32_vsub_VsfVsf(in1, in2);

            *(HVX_UVector *) (dst + i * SIZEOF_FP32) = Q6_Vsf_equals_Vqf32(out);
        }
    }

    if (left_over > 0) {
        const float * src0f = (const float *) src0 + num_elems_whole;
        const float * src1f = (const float *) src1 + num_elems_whole;
        float *       dstf  = (float *) dst + num_elems_whole;

        HVX_Vector in1 = *(HVX_UVector *) src0f;
        HVX_Vector in2 = *(HVX_UVector *) src1f;

        HVX_Vector out = Q6_Vqf32_vsub_VsfVsf(in1, in2);
        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(out));
    }
}

void hvx_sub_f32_opt(const uint8_t * restrict src0,
                     const uint8_t * restrict src1,
                     uint8_t * restrict dst,
                     const int num_elems) {
    htp_binary_ops_preamble;

    for (int i = 0; i < step_of_4; i++) {
        HVX_Vector v1a = *(HVX_Vector *) src0_curr;

        HVX_Vector v1b = *(HVX_Vector *) src1_curr;

        HVX_Vector v2a = *(HVX_Vector *) (src0_curr + VLEN);

        HVX_Vector v1 = Q6_Vqf32_vsub_VsfVsf(v1a, v1b);

        HVX_Vector v2b = *(HVX_Vector *) (src1_curr + VLEN);

        HVX_Vector v3a = *(HVX_Vector *) (src0_curr + 2 * VLEN);

        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v2a, v2b);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v1);

        HVX_Vector v3b = *(HVX_Vector *) (src1_curr + 2 * VLEN);

        HVX_Vector v4a = *(HVX_Vector *) (src0_curr + 3 * VLEN);

        src0_curr += 4 * VLEN;

        HVX_Vector v3 = Q6_Vqf32_vsub_VsfVsf(v3a, v3b);

        *(HVX_Vector *) (dst_curr + VLEN) = Q6_Vsf_equals_Vqf32(v2);

        HVX_Vector v4b = *(HVX_Vector *) (src1_curr + 3 * VLEN);

        *(HVX_Vector *) (dst_curr + 2 * VLEN) = Q6_Vsf_equals_Vqf32(v3);

        HVX_Vector v4 = Q6_Vqf32_vsub_VsfVsf(v4a, v4b);

        src1_curr += 4 * VLEN;

        *(HVX_Vector *) (dst_curr + 3 * VLEN) = Q6_Vsf_equals_Vqf32(v4);

        dst_curr += 4 * VLEN;
    }
    for (int i = 0; i < step_of_2; i++) {
        HVX_Vector v1a = *(HVX_Vector *) src0_curr;

        HVX_Vector v1b = *(HVX_Vector *) src1_curr;

        HVX_Vector v2a = *(HVX_Vector *) (src0_curr + VLEN);

        HVX_Vector v1 = Q6_Vqf32_vsub_VsfVsf(v1a, v1b);

        HVX_Vector v2b = *(HVX_Vector *) (src1_curr + VLEN);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v1);

        src0_curr += 2 * VLEN;

        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v2a, v2b);

        src1_curr += 2 * VLEN;

        *(HVX_Vector *) (dst_curr + VLEN) = Q6_Vsf_equals_Vqf32(v2);

        dst_curr += 2 * VLEN;
    }
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector va = *(HVX_Vector *) src0_curr;

        src0_curr += VLEN;

        HVX_Vector vb = *(HVX_Vector *) src1_curr;

        src1_curr += VLEN;

        HVX_Vector v = Q6_Vqf32_vsub_VsfVsf(va, vb);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v);

        dst_curr += VLEN;
    }
    if (remaining > 0) {
        HVX_Vector v = Q6_Vqf32_vsub_VsfVsf(*(HVX_Vector *) src0_curr, *(HVX_Vector *) src1_curr);
        hvx_vec_store_u((void *) dst_curr, remaining * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(v));
    }
}

void hvx_sub_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems) {
    size_t left_over       = num_elems & (VLEN_FP32 - 1);
    size_t num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if ((0 == htp_is_aligned((void *) src, VLEN)) || (0 == htp_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_sub_scalar_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_sub_scalar_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    HVX_Vector val_vec = hvx_vec_splat_fp32(val);

    if (0 == unaligned_loop) {
        HVX_Vector * restrict vec_in1 = (HVX_Vector *) src;
        HVX_Vector * restrict vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector v = Q6_Vqf32_vsub_VsfVsf(*vec_in1++, val_vec);
            *vec_out++   = Q6_Vsf_equals_Vqf32(v);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            HVX_Vector out = Q6_Vqf32_vsub_VsfVsf(in, val_vec);

            *(HVX_UVector *) (dst + i * SIZEOF_FP32) = Q6_Vsf_equals_Vqf32(out);
        }
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_Vector in = *(HVX_UVector *) srcf;

        HVX_Vector out = Q6_Vqf32_vsub_VsfVsf(in, val_vec);
        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(out));
    }
}

float hvx_sum_of_squares_f32(const uint8_t * restrict src, const int num_elems) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    if (0 == htp_is_aligned((void *) src, VLEN)) {
        FARF(HIGH, "hvx_sum_of_squares_f32: unaligned address in hvx op, possibly slower execution\n");
    }

    assert((1 == htp_is_aligned((void *) src, VLEN)) || (0 == num_elems_whole));

    HVX_Vector * restrict vec_in1 = (HVX_Vector *) src;

    HVX_Vector sum_vec_acc = Q6_V_vsplat_R(0x00000000);
    HVX_Vector zero_vec    = Q6_V_vsplat_R(0x00000000);

    #pragma unroll(4)
    for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(*vec_in1, *vec_in1);
        sum_vec_acc  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_vec_acc, v);
        vec_in1++;
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;

        HVX_Vector vec_left = *(HVX_UVector *) srcf;

        HVX_Vector vec_left_sq = Q6_Vqf32_vmpy_VsfVsf(vec_left, vec_left);
        HVX_Vector vec_tmp     = Q6_V_valign_VVR(vec_left_sq, zero_vec, left_over * SIZEOF_FP32);

        sum_vec_acc = Q6_Vqf32_vadd_Vqf32Vqf32(sum_vec_acc, vec_tmp);
    }

    HVX_Vector v = hvx_vec_qf32_reduce_sum(sum_vec_acc);
    return hvx_vec_get_fp32(Q6_Vsf_equals_Vqf32(v));
}

float hvx_self_sum_f32(const uint8_t * restrict src, const int num_elems) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if (0 == htp_is_aligned((void *) src, VLEN)) {
        FARF(HIGH, "hvx_self_sum_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_self_sum_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    HVX_Vector sum_vec  = Q6_V_vsplat_R(0x00000000);
    HVX_Vector zero_vec = Q6_V_vsplat_R(0x00000000);

    if (0 == unaligned_loop) {
        HVX_Vector * vec_in = (HVX_Vector *) src;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            // sum_vec = Q6_Vqf32_vadd_Vqf32Vsf(sum_vec, *vec_in++);
            sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), *vec_in++);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), in);
        }
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;

        HVX_Vector vec_left = *(HVX_UVector *) srcf;
        HVX_Vector vec_tmp  = Q6_V_valign_VVR(vec_left, zero_vec, left_over * SIZEOF_FP32);
        // sum_vec = Q6_Vqf32_vadd_Vqf32Vsf(sum_vec, vec_tmp);
        sum_vec             = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), vec_tmp);
    }

    HVX_Vector v = hvx_vec_qf32_reduce_sum(sum_vec);
    return hvx_vec_get_fp32(Q6_Vsf_equals_Vqf32(v));
}

void hvx_scale_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems, const float scale) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if ((0 == htp_is_aligned((void *) src, VLEN)) || (0 == htp_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_scale_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_scale_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    HVX_Vector scale_vec = hvx_vec_splat_fp32(scale);

    if (0 == unaligned_loop) {
        HVX_Vector * vec_in1 = (HVX_Vector *) src;
        HVX_Vector * vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(*vec_in1++, scale_vec);
            *vec_out++   = Q6_Vsf_equals_Vqf32(v);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            HVX_Vector out = Q6_Vqf32_vmpy_VsfVsf(in, scale_vec);

            *(HVX_UVector *) (dst + i * SIZEOF_FP32) = Q6_Vsf_equals_Vqf32(out);
        }
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_Vector in = *(HVX_UVector *) srcf;

        HVX_Vector out = Q6_Vqf32_vmpy_VsfVsf(in, scale_vec);
        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(out));
    }
}

float hvx_self_max_f32(const uint8_t * restrict src, const int num_elems) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if (0 == htp_is_aligned((void *) src, VLEN)) {
        FARF(HIGH, "hvx_self_max_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }

    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_self_max_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    HVX_Vector vec_max   = hvx_vec_splat_fp32(((const float *) src)[0]);
    HVX_Vector vec_first = hvx_vec_splat_fp32(((const float *) src)[0]);

    if (0 == unaligned_loop) {
        HVX_Vector * restrict vec_in = (HVX_Vector *) src;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            vec_max = Q6_Vsf_vmax_VsfVsf(vec_max, *vec_in++);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            vec_max = Q6_Vsf_vmax_VsfVsf(vec_max, in);
        }
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;

        HVX_Vector in = *(HVX_UVector *) srcf;

        HVX_Vector temp = Q6_V_valign_VVR(in, vec_first, left_over * SIZEOF_FP32);
        vec_max         = Q6_Vsf_vmax_VsfVsf(vec_max, temp);
    }

    HVX_Vector v = hvx_vec_reduce_max_fp32(vec_max);
    return hvx_vec_get_fp32(v);
}

void hvx_min_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems) {
    size_t left_over       = num_elems & (VLEN_FP32 - 1);
    size_t num_elems_whole = num_elems - left_over;

    if ((0 == htp_is_aligned((void *) src, VLEN)) || (0 == htp_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_min_scalar_f32: unaligned address in hvx op, possibly slower execution\n");
    }

    assert((1 == htp_is_aligned((void *) src, VLEN)) || (0 == num_elems_whole));

    const float * src_f = (const float *) src;

    HVX_Vector vec_min = Q6_V_vsplat_R(val);

    HVX_Vector * restrict vec_in  = (HVX_Vector *) src;
    HVX_Vector * restrict vec_out = (HVX_Vector *) dst;

    #pragma unroll(4)
    for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
        vec_min    = Q6_Vsf_vmin_VsfVsf(vec_min, *vec_in++);
        *vec_out++ = Q6_Vsf_equals_Vqf32(vec_min);
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_Vector in = *(HVX_UVector *) srcf;

        vec_min = Q6_Vsf_vmin_VsfVsf(vec_min, in);

        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(vec_min));
    }
}

void hvx_clamp_scalar_f32(const uint8_t * restrict src,
                          const float limit_left,
                          const float limit_right,
                          uint8_t * restrict dst,
                          const int num_elems) {
    size_t left_over       = num_elems & (VLEN_FP32 - 1);
    size_t num_elems_whole = num_elems - left_over;

    if ((0 == htp_is_aligned((void *) src, VLEN)) || (0 == htp_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_clamp_scalar_f32: unaligned address in hvx op, possibly slower execution\n");
    }

    assert((1 == htp_is_aligned((void *) src, VLEN)) || (0 == num_elems_whole));

    HVX_Vector * restrict vec_in  = (HVX_Vector *) src;
    HVX_Vector * restrict vec_out = (HVX_Vector *) dst;

    HVX_Vector range_left  = hvx_vec_splat_fp32(limit_left);
    HVX_Vector range_right = hvx_vec_splat_fp32(limit_right);

    #pragma unroll(4)
    for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
        HVX_Vector in_vec = *vec_in++;
        HVX_Vector temp_v = in_vec;

        HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(in_vec, range_right);
        HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(range_left, in_vec);

        in_vec = Q6_V_vmux_QVV(pred_cap_right, range_right, temp_v);
        in_vec = Q6_V_vmux_QVV(pred_cap_left, range_left, temp_v);

        *vec_out++ = Q6_Vsf_equals_Vqf32(in_vec);
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_Vector in = *(HVX_UVector *) srcf;

        HVX_Vector temp_v = in;

        HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(in, range_right);
        HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(range_left, in);

        in = Q6_V_vmux_QVV(pred_cap_right, range_right, temp_v);
        in = Q6_V_vmux_QVV(pred_cap_left, range_left, temp_v);

        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, Q6_Vsf_equals_Vqf32(in));
    }
}
