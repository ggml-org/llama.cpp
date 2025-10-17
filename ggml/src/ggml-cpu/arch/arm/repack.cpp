#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#define GGML_CPU_CLANG_WORKAROUND
#include "../../repack.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 3] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 11] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 15] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#endif
}

void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 4; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][2 * j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 3] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[0][2 * j + 1], id[0]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][2 * j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[1][2 * j + 1], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][2 * j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[2][2 * j + 1], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][2 * j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[3][2 * j + 1], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
        }
    }

#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x8_generic(x, vy, k);
#endif
}

void ggml_gemv_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx;

    for (int c = 0; c < nc; c += ncols_interleaved) {
        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        float32x4_t acc = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16_t b0 = vld1q_s8((const int8_t *) b_ptr->qs);
            int8x16_t b1 = vld1q_s8((const int8_t *) b_ptr->qs + 16);
            int8x16_t b2 = vld1q_s8((const int8_t *) b_ptr->qs + 32);
            int8x16_t b3 = vld1q_s8((const int8_t *) b_ptr->qs + 48);
            float16x4_t bd = vld1_f16((const __fp16 *) b_ptr->d);

            int8x16_t a0 = vld1q_s8(a_ptr->qs);
            int8x16_t a1 = vld1q_s8(a_ptr->qs + qk/2);
            float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);

            int32x4_t ret = vdupq_n_s32(0);

            ret = vdotq_laneq_s32(ret, b0 << 4, a0, 0);
            ret = vdotq_laneq_s32(ret, b1 << 4, a0, 1);
            ret = vdotq_laneq_s32(ret, b2 << 4, a0, 2);
            ret = vdotq_laneq_s32(ret, b3 << 4, a0, 3);

            ret = vdotq_laneq_s32(ret, b0 & 0xf0U, a1, 0);
            ret = vdotq_laneq_s32(ret, b1 & 0xf0U, a1, 1);
            ret = vdotq_laneq_s32(ret, b2 & 0xf0U, a1, 2);
            ret = vdotq_laneq_s32(ret, b3 & 0xf0U, a1, 3);

            acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                            vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++;
            b_ptr++;
        }
        vst1q_f32(s, acc);
        s += ncols_interleaved;
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx;

    for (int c = 0; c < nc; c += ncols_interleaved) {
        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        float32x4_t acc = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16_t b0 = vld1q_s8((const int8_t *) b_ptr->qs);
            int8x16_t b1 = vld1q_s8((const int8_t *) b_ptr->qs + 16);
            int8x16_t b2 = vld1q_s8((const int8_t *) b_ptr->qs + 32);
            int8x16_t b3 = vld1q_s8((const int8_t *) b_ptr->qs + 48);
            float16x4_t bd = vld1_f16((const __fp16 *) b_ptr->d);

            int8x16_t a0 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs);
            int8x16_t a1 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs + 1);
            int8x16_t a2 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs + 2);
            int8x16_t a3 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs + 3);
            float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);

            int32x4_t ret0 = vdupq_n_s32(0);
            int32x4_t ret1 = vdupq_n_s32(0);

            ret0 = vdotq_s32(ret0, b0 << 4, a0);
            ret1 = vdotq_s32(ret1, b1 << 4, a0);
            ret0 = vdotq_s32(ret0, b2 << 4, a1);
            ret1 = vdotq_s32(ret1, b3 << 4, a1);

            ret0 = vdotq_s32(ret0, b0 & 0xf0U, a2);
            ret1 = vdotq_s32(ret1, b1 & 0xf0U, a2);
            ret0 = vdotq_s32(ret0, b2 & 0xf0U, a3);
            ret1 = vdotq_s32(ret1, b3 & 0xf0U, a3);

            int32x4_t ret = vpaddq_s32(ret0, ret1);

            acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                    vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++;
            b_ptr++;
        }
        vst1q_f32(s, acc);
        s += ncols_interleaved;
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q4_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
#if defined(__ARM_FEATURE_SVE)
    if (ggml_cpu_get_sve_cnt() == QK8_0) {
        const void * b_ptr = vx;
        const void * a_ptr = vy;
        float * res_ptr = s;

        __asm__ __volatile__(
            "ptrue p0.b\n"
            "add %x[b_ptr], %x[b_ptr], #0x10\n"
            "1:"  // Column loop
            "add x22, %x[a_ptr], #0x2\n"
            "mov z31.b, #0x0\n"
            "mov x21, %x[nb]\n"
            "2:"  // Block loop
            "ld1b { z30.b }, p0/Z, [%x[b_ptr]]\n"
            "ld1b { z29.b }, p0/Z, [%x[b_ptr], #1, MUL VL]\n"
            "mov z28.s, #0x0\n"
            "mov z27.s, #0x0\n"
            "ld1rd { z26.d }, p0/Z, [x22]\n"
            "ld1b { z25.b }, p0/Z, [%x[b_ptr], #2, MUL VL]\n"
            "sub x20, x22, #0x2\n"
            "sub x21, x21, #0x1\n"
            "ld1b { z24.b }, p0/Z, [%x[b_ptr], #3, MUL VL]\n"
            "ld1rd { z23.d }, p0/Z, [x22, #8]\n"
            "lsl z22.b, z30.b, #0x4\n"
            "lsl z16.b, z29.b, #0x4\n"
            "and z30.b, z30.b, #0xf0\n"
            "and z29.b, z29.b, #0xf0\n"
            "ld1rd { z21.d }, p0/Z, [x22, #16]\n"
            "ld1rd { z20.d }, p0/Z, [x22, #24]\n"
            "lsl z19.b, z25.b, #0x4\n"
            "and z25.b, z25.b, #0xf0\n"
            "ld1rh { z17.h }, p0/Z, [x20]\n"
            "ld1h { z18.s }, p0/Z, [%x[b_ptr], #-1, MUL VL]\n"
            "sdot z28.s, z22.b, z26.b\n"
            "sdot z27.s, z16.b, z26.b\n"
            "lsl z16.b, z24.b, #0x4\n"
            "add x22, x22, #0x22\n"
            "and z24.b, z24.b, #0xf0\n"
            "add %x[b_ptr], %x[b_ptr], #0x90\n"
            "fcvt z17.s, p0/m, z17.h\n"
            "fcvt z18.s, p0/m, z18.h\n"
            "sdot z28.s, z19.b, z23.b\n"
            "sdot z27.s, z16.b, z23.b\n"
            "fmul z18.s, z18.s, z17.s\n"
            "sdot z28.s, z30.b, z21.b\n"
            "sdot z27.s, z29.b, z21.b\n"
            "sdot z28.s, z25.b, z20.b\n"
            "sdot z27.s, z24.b, z20.b\n"
            "uzp1 z17.s, z28.s, z27.s\n"
            "uzp2 z16.s, z28.s, z27.s\n"
            "add z17.s, z17.s, z16.s\n"
            "asr z17.s, z17.s, #0x4\n"
            "scvtf z17.s, p0/m, z17.s\n"
            "fmla z31.s, p0/M, z17.s, z18.s\n"
            "cbnz x21, 2b\n"
            "sub %x[nc], %x[nc], #0x8\n"
            "st1w { z31.s }, p0, [%x[res_ptr]]\n"
            "add %x[res_ptr], %x[res_ptr], #0x20\n"
            "cbnz %x[nc], 1b\n"
            : [b_ptr] "+&r" (b_ptr), [res_ptr] "+&r" (res_ptr), [nc] "+&r" (nc)
            : [a_ptr] "r" (a_ptr), [nb] "r" (nb)
            : "memory", "p0", "x20", "x21", "x22", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
        );
        return;
    }
#endif // #if defined(__ARM_FEATURE_SVE)

#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t kvalues = vld1q_s8(kvalues_iq4nl);
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    float * res_ptr = s;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

        float32x4_t sumf = vdupq_n_f32(0);
        for (int l = 0; l < nb; l++) {
            uint8x16_t b_0 = vld1q_u8(b_ptr[l].qs + 0);
            uint8x16_t b_1 = vld1q_u8(b_ptr[l].qs + 16);
            uint8x16_t b_2 = vld1q_u8(b_ptr[l].qs + 32);
            uint8x16_t b_3 = vld1q_u8(b_ptr[l].qs + 48);

            int8x16_t b_0_hi = vqtbl1q_s8(kvalues, b_0 >> 4);
            int8x16_t b_0_lo = vqtbl1q_s8(kvalues, b_0 & 0x0F);
            int8x16_t b_1_hi = vqtbl1q_s8(kvalues, b_1 >> 4);
            int8x16_t b_1_lo = vqtbl1q_s8(kvalues, b_1 & 0x0F);
            int8x16_t b_2_hi = vqtbl1q_s8(kvalues, b_2 >> 4);
            int8x16_t b_2_lo = vqtbl1q_s8(kvalues, b_2 & 0x0F);
            int8x16_t b_3_hi = vqtbl1q_s8(kvalues, b_3 >> 4);
            int8x16_t b_3_lo = vqtbl1q_s8(kvalues, b_3 & 0x0F);

            int8x16_t a_0 = vld1q_s8(a_ptr[l].qs + 0);
            int8x16_t a_1 = vld1q_s8(a_ptr[l].qs + 16);

            int32x4_t sumi = vdupq_n_s32(0);
            sumi = vdotq_laneq_s32(sumi, b_0_lo, a_0, 0);
            sumi = vdotq_laneq_s32(sumi, b_0_hi, a_1, 0);
            sumi = vdotq_laneq_s32(sumi, b_1_lo, a_0, 1);
            sumi = vdotq_laneq_s32(sumi, b_1_hi, a_1, 1);
            sumi = vdotq_laneq_s32(sumi, b_2_lo, a_0, 2);
            sumi = vdotq_laneq_s32(sumi, b_2_hi, a_1, 2);
            sumi = vdotq_laneq_s32(sumi, b_3_lo, a_0, 3);
            sumi = vdotq_laneq_s32(sumi, b_3_hi, a_1, 3);

            float32x4_t a_d = vcvt_f32_f16(vld1_dup_f16((const float16_t *)&a_ptr[l].d));
            float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *)b_ptr[l].d));
            float32x4_t d = a_d * b_d;

            sumf = vmlaq_f32(sumf, d, vcvtq_f32_s32(sumi));
        }

        vst1q_f32(res_ptr + x * 4, sumf);
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    ggml_gemv_iq4_nl_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const void * b_ptr = vx;
    const void * a_ptr = vy;
    float * res_ptr = s;
    size_t res_stride = bs * sizeof(float);

    __asm__ __volatile__(
        "mov x10, %x[nr]\n"
        "mov x9, #0x88\n"
        "cmp x10, #0x10\n"
        "mul x9, %x[nb], x9\n"
        "blt 4f\n"
        "1:"  // Row loop
        "add x28, %x[b_ptr], #0x8\n"
        "mov x27, %x[nc]\n"
        "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:"  // Column loop
        "add x25, %x[a_ptr], #0x8\n"
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "mov x24, %x[nb]\n"
        "add x23, x25, x9\n"
        "movi v18.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "add x22, x23, x9\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "add x21, x22, x9\n"
        "movi v23.16b, #0x0\n"
        "movi v16.16b, #0x0\n"
        "movi v25.16b, #0x0\n"
        "movi v7.16b, #0x0\n"
        "movi v0.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v21.16b, #0x0\n"
        "movi v8.16b, #0x0\n"
        "movi v1.16b, #0x0\n"
        "3:"  // Block loop
        "ldr q3, [x28, #0x0]\n"
        "ldr q31, [x25, #0x0]\n"
        "movi v28.16b, #0x4\n"
        "movi v10.4s, #0x0\n"
        "ldr q22, [x28, #0x10]\n"
        "ldr q6, [x25, #0x10]\n"
        "movi v29.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "ldr q27, [x28, #0x20]\n"
        "ldr q30, [x28, #0x30]\n"
        "movi v20.4s, #0x0\n"
        "movi v24.16b, #0xf0\n"
        "ldr d2, [x25, #-0x8]\n"
        "ldr d26, [x23, #-0x8]\n"
        "sshl v12.16b, v3.16b, v28.16b\n"
        "sub x20, x28, #0x8\n"
        "ldr d17, [x20, #0x0]\n"
        "and v3.16b, v3.16b, v24.16b\n"
        "subs x24, x24, #0x1\n"
        "add x28, x28, #0x48\n"
        ".inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]\n"
        ".inst 0x4fbfe19d  // sdot v29.4s, v12.16b, v31.4b[1]\n"
        ".inst 0x4f9fe989  // sdot v9.4s, v12.16b, v31.4b[2]\n"
        ".inst 0x4fbfe994  // sdot v20.4s, v12.16b, v31.4b[3]\n"
        "sshl v31.16b, v22.16b, v28.16b\n"
        "and v22.16b, v22.16b, v24.16b\n"
        "fcvtl v17.4s, v17.4h\n"
        "fcvtl v2.4s, v2.4h\n"
        "fcvtl v26.4s, v26.4h\n"
        ".inst 0x4f86e3ea  // sdot v10.4s, v31.16b, v6.4b[0]\n"
        ".inst 0x4fa6e3fd  // sdot v29.4s, v31.16b, v6.4b[1]\n"
        ".inst 0x4f86ebe9  // sdot v9.4s, v31.16b, v6.4b[2]\n"
        ".inst 0x4fa6ebf4  // sdot v20.4s, v31.16b, v6.4b[3]\n"
        "sshl v6.16b, v27.16b, v28.16b\n"
        "sshl v28.16b, v30.16b, v28.16b\n"
        "and v27.16b, v27.16b, v24.16b\n"
        "and v30.16b, v30.16b, v24.16b\n"
        "ldr q24, [x25, #0x20]\n"
        ".inst 0x4f98e0ca  // sdot v10.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8c9  // sdot v9.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8d4  // sdot v20.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x30]\n"
        ".inst 0x4f98e38a  // sdot v10.4s, v28.16b, v24.4b[0]\n"
        ".inst 0x4fb8e39d  // sdot v29.4s, v28.16b, v24.4b[1]\n"
        ".inst 0x4f98eb89  // sdot v9.4s, v28.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb94  // sdot v20.4s, v28.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x40]\n"
        ".inst 0x4f98e06a  // sdot v10.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e869  // sdot v9.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e874  // sdot v20.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x50]\n"
        ".inst 0x4f98e2ca  // sdot v10.4s, v22.16b, v24.4b[0]\n"
        ".inst 0x4fb8e2dd  // sdot v29.4s, v22.16b, v24.4b[1]\n"
        ".inst 0x4f98eac9  // sdot v9.4s, v22.16b, v24.4b[2]\n"
        ".inst 0x4fb8ead4  // sdot v20.4s, v22.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x60]\n"
        ".inst 0x4f98e36a  // sdot v10.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb69  // sdot v9.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb74  // sdot v20.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x70]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x4f98e3ca  // sdot v10.4s, v30.16b, v24.4b[0]\n"
        ".inst 0x4fb8e3dd  // sdot v29.4s, v30.16b, v24.4b[1]\n"
        ".inst 0x4f98ebc9  // sdot v9.4s, v30.16b, v24.4b[2]\n"
        ".inst 0x4fb8ebd4  // sdot v20.4s, v30.16b, v24.4b[3]\n"
        "fmul v24.4s, v17.4s, v2.s[0]\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v15.4s, v10.4s, v24.4s\n"
        "ldr q24, [x23, #0x0]\n"
        "fmul v10.4s, v17.4s, v2.s[1]\n"
        "fmla v19.4s, v29.4s, v10.4s\n"
        "ldr q10, [x23, #0x10]\n"
        "fmul v29.4s, v17.4s, v2.s[2]\n"
        "fmul v2.4s, v17.4s, v2.s[3]\n"
        "fmla v18.4s, v9.4s, v29.4s\n"
        "movi v9.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        ".inst 0x4f98e189  // sdot v9.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e19d  // sdot v29.4s, v12.16b, v24.4b[1]\n"
        "fmla v14.4s, v20.4s, v2.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        ".inst 0x4f98e994  // sdot v20.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x20]\n"
        ".inst 0x4f8ae3e9  // sdot v9.4s, v31.16b, v10.4b[0]\n"
        ".inst 0x4faae3fd  // sdot v29.4s, v31.16b, v10.4b[1]\n"
        ".inst 0x4f8aebf4  // sdot v20.4s, v31.16b, v10.4b[2]\n"
        ".inst 0x4faaebe2  // sdot v2.4s, v31.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x30]\n"
        ".inst 0x4f98e0c9  // sdot v9.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8d4  // sdot v20.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x40]\n"
        ".inst 0x4f8ae389  // sdot v9.4s, v28.16b, v10.4b[0]\n"
        ".inst 0x4faae39d  // sdot v29.4s, v28.16b, v10.4b[1]\n"
        ".inst 0x4f8aeb94  // sdot v20.4s, v28.16b, v10.4b[2]\n"
        ".inst 0x4faaeb82  // sdot v2.4s, v28.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x50]\n"
        ".inst 0x4f98e069  // sdot v9.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e874  // sdot v20.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x60]\n"
        ".inst 0x4f8ae2c9  // sdot v9.4s, v22.16b, v10.4b[0]\n"
        ".inst 0x4faae2dd  // sdot v29.4s, v22.16b, v10.4b[1]\n"
        ".inst 0x4f8aead4  // sdot v20.4s, v22.16b, v10.4b[2]\n"
        ".inst 0x4faaeac2  // sdot v2.4s, v22.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x70]\n"
        "add x23, x23, #0x88\n"
        ".inst 0x4f98e369  // sdot v9.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb74  // sdot v20.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x0]\n"
        ".inst 0x4f8ae3c9  // sdot v9.4s, v30.16b, v10.4b[0]\n"
        ".inst 0x4faae3dd  // sdot v29.4s, v30.16b, v10.4b[1]\n"
        ".inst 0x4f8aebd4  // sdot v20.4s, v30.16b, v10.4b[2]\n"
        ".inst 0x4faaebc2  // sdot v2.4s, v30.16b, v10.4b[3]\n"
        "fmul v10.4s, v17.4s, v26.s[0]\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v11.4s, v9.4s, v10.4s\n"
        "ldr q9, [x22, #0x10]\n"
        "fmul v10.4s, v17.4s, v26.s[1]\n"
        "fmla v13.4s, v29.4s, v10.4s\n"
        "ldr d29, [x22, #-0x8]\n"
        "fmul v10.4s, v17.4s, v26.s[2]\n"
        "fmul v26.4s, v17.4s, v26.s[3]\n"
        "fcvtl v29.4s, v29.4h\n"
        "fmla v23.4s, v20.4s, v10.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "fmla v16.4s, v2.4s, v26.4s\n"
        "movi v26.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        ".inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
        ".inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x20]\n"
        ".inst 0x4f89e3f4  // sdot v20.4s, v31.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
        ".inst 0x4f89ebfa  // sdot v26.4s, v31.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebe2  // sdot v2.4s, v31.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x30]\n"
        ".inst 0x4f98e0d4  // sdot v20.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0ca  // sdot v10.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8da  // sdot v26.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x40]\n"
        ".inst 0x4f89e394  // sdot v20.4s, v28.16b, v9.4b[0]\n"
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
        ".inst 0x4f89eb9a  // sdot v26.4s, v28.16b, v9.4b[2]\n"
        ".inst 0x4fa9eb82  // sdot v2.4s, v28.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x50]\n"
        ".inst 0x4f98e074  // sdot v20.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e06a  // sdot v10.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e87a  // sdot v26.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x60]\n"
        ".inst 0x4f89e2d4  // sdot v20.4s, v22.16b, v9.4b[0]\n"
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
        ".inst 0x4f89eada  // sdot v26.4s, v22.16b, v9.4b[2]\n"
        ".inst 0x4fa9eac2  // sdot v2.4s, v22.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x70]\n"
        "add x22, x22, #0x88\n"
        ".inst 0x4f98e374  // sdot v20.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e36a  // sdot v10.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb7a  // sdot v26.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x21, #0x0]\n"
        ".inst 0x4f89e3d4  // sdot v20.4s, v30.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ca  // sdot v10.4s, v30.16b, v9.4b[1]\n"
        ".inst 0x4f89ebda  // sdot v26.4s, v30.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebc2  // sdot v2.4s, v30.16b, v9.4b[3]\n"
        "fmul v9.4s, v17.4s, v29.s[0]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v25.4s, v20.4s, v9.4s\n"
        "ldr q9, [x21, #0x10]\n"
        "fmul v20.4s, v17.4s, v29.s[1]\n"
        "fmla v7.4s, v10.4s, v20.4s\n"
        "ldr d20, [x21, #-0x8]\n"
        "fmul v10.4s, v17.4s, v29.s[2]\n"
        "fmul v29.4s, v17.4s, v29.s[3]\n"
        "fcvtl v20.4s, v20.4h\n"
        "fmla v0.4s, v26.4s, v10.4s\n"
        "movi v26.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "fmla v4.4s, v2.4s, v29.4s\n"
        "movi v2.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        ".inst 0x4f98e19a  // sdot v26.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
        ".inst 0x4f98e982  // sdot v2.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e99d  // sdot v29.4s, v12.16b, v24.4b[3]\n"
        "ldr q12, [x21, #0x20]\n"
        "fmul v24.4s, v17.4s, v20.s[0]\n"
        ".inst 0x4f89e3fa  // sdot v26.4s, v31.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
        ".inst 0x4f89ebe2  // sdot v2.4s, v31.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebfd  // sdot v29.4s, v31.16b, v9.4b[3]\n"
        "ldr q9, [x21, #0x30]\n"
        "fmul v31.4s, v17.4s, v20.s[1]\n"
        ".inst 0x4f8ce0da  // sdot v26.4s, v6.16b, v12.4b[0]\n"
        ".inst 0x4face0ca  // sdot v10.4s, v6.16b, v12.4b[1]\n"
        ".inst 0x4f8ce8c2  // sdot v2.4s, v6.16b, v12.4b[2]\n"
        ".inst 0x4face8dd  // sdot v29.4s, v6.16b, v12.4b[3]\n"
        "ldr q12, [x21, #0x40]\n"
        "fmul v6.4s, v17.4s, v20.s[2]\n"
        "fmul v20.4s, v17.4s, v20.s[3]\n"
        ".inst 0x4f89e39a  // sdot v26.4s, v28.16b, v9.4b[0]\n"
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
        ".inst 0x4f89eb82  // sdot v2.4s, v28.16b, v9.4b[2]\n"
        ".inst 0x4fa9eb9d  // sdot v29.4s, v28.16b, v9.4b[3]\n"
        "ldr q9, [x21, #0x50]\n"
        ".inst 0x4f8ce07a  // sdot v26.4s, v3.16b, v12.4b[0]\n"
        ".inst 0x4face06a  // sdot v10.4s, v3.16b, v12.4b[1]\n"
        ".inst 0x4f8ce862  // sdot v2.4s, v3.16b, v12.4b[2]\n"
        ".inst 0x4face87d  // sdot v29.4s, v3.16b, v12.4b[3]\n"
        "ldr q12, [x21, #0x60]\n"
        ".inst 0x4f89e2da  // sdot v26.4s, v22.16b, v9.4b[0]\n"
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
        ".inst 0x4f89eac2  // sdot v2.4s, v22.16b, v9.4b[2]\n"
        ".inst 0x4fa9eadd  // sdot v29.4s, v22.16b, v9.4b[3]\n"
        "ldr q17, [x21, #0x70]\n"
        "add x21, x21, #0x88\n"
        ".inst 0x4f8ce37a  // sdot v26.4s, v27.16b, v12.4b[0]\n"
        ".inst 0x4face36a  // sdot v10.4s, v27.16b, v12.4b[1]\n"
        ".inst 0x4f8ceb62  // sdot v2.4s, v27.16b, v12.4b[2]\n"
        ".inst 0x4faceb7d  // sdot v29.4s, v27.16b, v12.4b[3]\n"
        ".inst 0x4f91e3da  // sdot v26.4s, v30.16b, v17.4b[0]\n"
        ".inst 0x4fb1e3ca  // sdot v10.4s, v30.16b, v17.4b[1]\n"
        ".inst 0x4f91ebc2  // sdot v2.4s, v30.16b, v17.4b[2]\n"
        ".inst 0x4fb1ebdd  // sdot v29.4s, v30.16b, v17.4b[3]\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "fmla v5.4s, v26.4s, v24.4s\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "fmla v21.4s, v10.4s, v31.4s\n"
        "fmla v8.4s, v2.4s, v6.4s\n"
        "fmla v1.4s, v29.4s, v20.4s\n"
        "bgt 3b\n"
        "mov x20, %x[res_ptr]\n"
        "subs x27, x27, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q14, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q11, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q16, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q7, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q0, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q4, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q21, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q8, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q1, [x20, #0x0]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x10, x10, #0x10\n"
        "cmp x10, #0x10\n"
        "mov %x[res_ptr], x26\n"
        "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
        "bge 1b\n"
        "4:"  // Row loop skip
        "cbz x10, 9f\n"
        "5:"  // Row tail: Row loop
        "add x24, %x[b_ptr], #0x8\n"
        "mov x23, %x[nc]\n"
        "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:"  // Row tail: Column loop
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "add x25, %x[a_ptr], #0x8\n"
        "mov x21, %x[nb]\n"
        "movi v18.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "7:"  // Row tail: Block loop
        "ldr q7, [x24, #0x0]\n"
        "ldr q5, [x25, #0x0]\n"
        "movi v9.16b, #0x4\n"
        "movi v4.4s, #0x0\n"
        "ldr q3, [x24, #0x10]\n"
        "ldr q2, [x25, #0x10]\n"
        "movi v1.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "ldr q13, [x24, #0x20]\n"
        "ldr q31, [x25, #0x20]\n"
        "movi v30.4s, #0x0\n"
        "movi v29.16b, #0xf0\n"
        "ldr q28, [x24, #0x30]\n"
        "ldr q27, [x25, #0x30]\n"
        "sshl v20.16b, v7.16b, v9.16b\n"
        "sub x20, x24, #0x8\n"
        "ldr q26, [x25, #0x40]\n"
        "ldr q25, [x25, #0x50]\n"
        "sshl v17.16b, v3.16b, v9.16b\n"
        "and v7.16b, v7.16b, v29.16b\n"
        "ldr q24, [x25, #0x60]\n"
        "ldr q16, [x25, #0x70]\n"
        "sshl v22.16b, v13.16b, v9.16b\n"
        "and v3.16b, v3.16b, v29.16b\n"
        "ldr d21, [x20, #0x0]\n"
        "ldr d12, [x25, #-0x8]\n"
        ".inst 0x4f85e284  // sdot v4.4s, v20.16b, v5.4b[0]\n"
        ".inst 0x4fa5e281  // sdot v1.4s, v20.16b, v5.4b[1]\n"
        ".inst 0x4f85ea80  // sdot v0.4s, v20.16b, v5.4b[2]\n"
        ".inst 0x4fa5ea9e  // sdot v30.4s, v20.16b, v5.4b[3]\n"
        "sshl v9.16b, v28.16b, v9.16b\n"
        "subs x21, x21, #0x1\n"
        "and v13.16b, v13.16b, v29.16b\n"
        "and v28.16b, v28.16b, v29.16b\n"
        "add x25, x25, #0x88\n"
        "add x24, x24, #0x48\n"
        "fcvtl v21.4s, v21.4h\n"
        "fcvtl v12.4s, v12.4h\n"
        ".inst 0x4f82e224  // sdot v4.4s, v17.16b, v2.4b[0]\n"
        ".inst 0x4fa2e221  // sdot v1.4s, v17.16b, v2.4b[1]\n"
        ".inst 0x4f82ea20  // sdot v0.4s, v17.16b, v2.4b[2]\n"
        ".inst 0x4fa2ea3e  // sdot v30.4s, v17.16b, v2.4b[3]\n"
        "fmul v11.4s, v21.4s, v12.s[0]\n"
        "fmul v23.4s, v21.4s, v12.s[1]\n"
        "fmul v17.4s, v21.4s, v12.s[2]\n"
        ".inst 0x4f9fe2c4  // sdot v4.4s, v22.16b, v31.4b[0]\n"
        "fmul v6.4s, v21.4s, v12.s[3]\n"
        ".inst 0x4fbfe2c1  // sdot v1.4s, v22.16b, v31.4b[1]\n"
        ".inst 0x4f9feac0  // sdot v0.4s, v22.16b, v31.4b[2]\n"
        ".inst 0x4fbfeade  // sdot v30.4s, v22.16b, v31.4b[3]\n"
        ".inst 0x4f9be124  // sdot v4.4s, v9.16b, v27.4b[0]\n"
        ".inst 0x4fbbe121  // sdot v1.4s, v9.16b, v27.4b[1]\n"
        ".inst 0x4f9be920  // sdot v0.4s, v9.16b, v27.4b[2]\n"
        ".inst 0x4fbbe93e  // sdot v30.4s, v9.16b, v27.4b[3]\n"
        ".inst 0x4f9ae0e4  // sdot v4.4s, v7.16b, v26.4b[0]\n"
        ".inst 0x4fbae0e1  // sdot v1.4s, v7.16b, v26.4b[1]\n"
        ".inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]\n"
        ".inst 0x4fbae8fe  // sdot v30.4s, v7.16b, v26.4b[3]\n"
        ".inst 0x4f99e064  // sdot v4.4s, v3.16b, v25.4b[0]\n"
        ".inst 0x4fb9e061  // sdot v1.4s, v3.16b, v25.4b[1]\n"
        ".inst 0x4f99e860  // sdot v0.4s, v3.16b, v25.4b[2]\n"
        ".inst 0x4fb9e87e  // sdot v30.4s, v3.16b, v25.4b[3]\n"
        ".inst 0x4f98e1a4  // sdot v4.4s, v13.16b, v24.4b[0]\n"
        ".inst 0x4fb8e1a1  // sdot v1.4s, v13.16b, v24.4b[1]\n"
        ".inst 0x4f98e9a0  // sdot v0.4s, v13.16b, v24.4b[2]\n"
        ".inst 0x4fb8e9be  // sdot v30.4s, v13.16b, v24.4b[3]\n"
        ".inst 0x4f90e384  // sdot v4.4s, v28.16b, v16.4b[0]\n"
        ".inst 0x4fb0e381  // sdot v1.4s, v28.16b, v16.4b[1]\n"
        ".inst 0x4f90eb80  // sdot v0.4s, v28.16b, v16.4b[2]\n"
        ".inst 0x4fb0eb9e  // sdot v30.4s, v28.16b, v16.4b[3]\n"
        "scvtf v4.4s, v4.4s, #0x4\n"
        "scvtf v1.4s, v1.4s, #0x4\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "fmla v15.4s, v4.4s, v11.4s\n"
        "scvtf v30.4s, v30.4s, #0x4\n"
        "fmla v19.4s, v1.4s, v23.4s\n"
        "fmla v18.4s, v0.4s, v17.4s\n"
        "fmla v14.4s, v30.4s, v6.4s\n"
        "bgt 7b\n"
        "mov x20, %x[res_ptr]\n"
        "cmp x10, #0x1\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x2\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x3\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "str q14, [x20, #0x0]\n"
        "8:"  // Row tail: Accumulator store skip
        "subs x23, x23, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "bne 6b\n"
        "subs x10, x10, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x9\n"
        "mov %x[res_ptr], x22\n"
        "bgt 5b\n"
        "9:"  // Row tail: Row loop skip
        : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
        : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [nb] "r" (nb), [res_stride] "r" (res_stride), [nc] "r" (nc)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    ggml_gemm_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const void * b_ptr = vx;
    const void * a_ptr = vy;
    float * res_ptr = s;
    size_t res_stride = bs * sizeof(float);

    __asm__ __volatile__(
        "mov x10, %x[nr]\n"
        "mov x9, #0x88\n"
        "cmp x10, #0x10\n"
        "mul x9, %x[nb], x9\n"
        "blt 4f\n"
        "1:"  // Row loop
        "add x28, %x[b_ptr], #0x8\n"
        "mov x27, %x[nc]\n"
        "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:"  // Column loop
        "add x25, %x[a_ptr], #0x8\n"
        "movi v2.16b, #0x0\n"
        "movi v10.16b, #0x0\n"
        "mov x24, %x[nb]\n"
        "add x23, x25, x9\n"
        "movi v12.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "add x22, x23, x9\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "add x21, x22, x9\n"
        "movi v22.16b, #0x0\n"
        "movi v23.16b, #0x0\n"
        "movi v25.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v7.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "movi v6.16b, #0x0\n"
        "movi v30.16b, #0x0\n"
        "movi v24.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "3:"  // Block loop
        "ldr q21, [x28, #0x0]\n"
        "ldr q16, [x28, #0x10]\n"
        "movi v1.16b, #0x4\n"
        "movi v19.4s, #0x0\n"
        "ldr q27, [x25, #0x0]\n"
        "ldr q15, [x25, #0x10]\n"
        "movi v26.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        "ldr q29, [x28, #0x20]\n"
        "ldr q3, [x28, #0x30]\n"
        "movi v17.4s, #0x0\n"
        "movi v0.16b, #0xf0\n"
        "ldr d20, [x25, #-0x8]\n"
        "ldr d9, [x23, #-0x8]\n"
        "sshl v8.16b, v21.16b, v1.16b\n"
        "sshl v31.16b, v16.16b, v1.16b\n"
        "and v21.16b, v21.16b, v0.16b\n"
        "and v16.16b, v16.16b, v0.16b\n"
        "sub x20, x28, #0x8\n"
        "subs x24, x24, #0x1\n"
        "add x28, x28, #0x48\n"
        ".inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b\n"
        ".inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b\n"
        "ldr q27, [x25, #0x20]\n"
        ".inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b\n"
        ".inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b\n"
        "sshl v15.16b, v29.16b, v1.16b\n"
        "sshl v1.16b, v3.16b, v1.16b\n"
        "and v29.16b, v29.16b, v0.16b\n"
        "and v3.16b, v3.16b, v0.16b\n"
        "ldr q0, [x25, #0x30]\n"
        "fcvtl v20.4s, v20.4h\n"
        ".inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b\n"
        "fcvtl v9.4s, v9.4h\n"
        ".inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b\n"
        "ldr q27, [x25, #0x40]\n"
        ".inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b\n"
        ".inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b\n"
        "ldr q0, [x25, #0x50]\n"
        ".inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b\n"
        ".inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b\n"
        "ldr q27, [x25, #0x60]\n"
        ".inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b\n"
        ".inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b\n"
        "ldr q0, [x25, #0x70]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b\n"
        ".inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b\n"
        "ldr d27, [x20, #0x0]\n"
        ".inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b\n"
        ".inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b\n"
        "fcvtl v27.4s, v27.4h\n"
        "uzp1 v0.2d, v19.2d, v26.2d\n"
        "uzp2 v26.2d, v19.2d, v26.2d\n"
        "fmul v19.4s, v27.4s, v20.s[0]\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "fmla v2.4s, v0.4s, v19.4s\n"
        "ldr q19, [x23, #0x0]\n"
        "uzp1 v0.2d, v18.2d, v17.2d\n"
        "uzp2 v18.2d, v18.2d, v17.2d\n"
        "fmul v17.4s, v27.4s, v20.s[1]\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v10.4s, v26.4s, v17.4s\n"
        "ldr q17, [x23, #0x10]\n"
        "fmul v26.4s, v27.4s, v20.s[2]\n"
        "fmul v20.4s, v27.4s, v20.s[3]\n"
        "fmla v12.4s, v0.4s, v26.4s\n"
        "ldr d0, [x22, #-0x8]\n"
        "ldr d26, [x21, #-0x8]\n"
        "fcvtl v0.4s, v0.4h\n"
        "fmla v28.4s, v18.4s, v20.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
        ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
        "ldr q19, [x23, #0x20]\n"
        "fcvtl v26.4s, v26.4h\n"
        ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
        ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
        "ldr q19, [x23, #0x40]\n"
        ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
        ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
        "ldr q19, [x23, #0x60]\n"
        ".inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b\n"
        ".inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b\n"
        "uzp1 v19.2d, v20.2d, v18.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp2 v20.2d, v20.2d, v18.2d\n"
        "fmul v18.4s, v27.4s, v9.s[0]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v11.4s, v19.4s, v18.4s\n"
        "ldr q18, [x22, #0x0]\n"
        "fmul v19.4s, v27.4s, v9.s[1]\n"
        "fmla v13.4s, v20.4s, v19.4s\n"
        "movi v19.4s, #0x0\n"
        "movi v20.4s, #0x0\n"
        ".inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b\n"
        ".inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b\n"
        "ldr q17, [x23, #0x30]\n"
        ".inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b\n"
        ".inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b\n"
        "ldr q17, [x23, #0x50]\n"
        ".inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b\n"
        ".inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b\n"
        "ldr q17, [x23, #0x70]\n"
        "add x23, x23, #0x88\n"
        ".inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b\n"
        ".inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b\n"
        "uzp1 v17.2d, v19.2d, v20.2d\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "uzp2 v20.2d, v19.2d, v20.2d\n"
        "fmul v19.4s, v27.4s, v9.s[2]\n"
        "fmul v9.4s, v27.4s, v9.s[3]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v22.4s, v17.4s, v19.4s\n"
        "ldr q17, [x22, #0x10]\n"
        "movi v19.4s, #0x0\n"
        ".inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b\n"
        "fmla v23.4s, v20.4s, v9.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        ".inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b\n"
        "ldr q18, [x22, #0x20]\n"
        ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
        ".inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b\n"
        ".inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b\n"
        "ldr q18, [x22, #0x40]\n"
        ".inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b\n"
        ".inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b\n"
        "ldr q18, [x22, #0x60]\n"
        ".inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b\n"
        ".inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b\n"
        "ldr q17, [x22, #0x30]\n"
        ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
        ".inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b\n"
        "ldr q17, [x22, #0x50]\n"
        ".inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b\n"
        ".inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b\n"
        "ldr q17, [x22, #0x70]\n"
        "add x22, x22, #0x88\n"
        ".inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b\n"
        ".inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b\n"
        "uzp1 v17.2d, v19.2d, v20.2d\n"
        "uzp2 v20.2d, v19.2d, v20.2d\n"
        "fmul v19.4s, v27.4s, v0.s[0]\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v25.4s, v17.4s, v19.4s\n"
        "ldr q19, [x21, #0x0]\n"
        "fmul v17.4s, v27.4s, v0.s[1]\n"
        "fmla v5.4s, v20.4s, v17.4s\n"
        "ldr q17, [x21, #0x10]\n"
        "uzp1 v20.2d, v9.2d, v18.2d\n"
        "uzp2 v9.2d, v9.2d, v18.2d\n"
        "fmul v18.4s, v27.4s, v0.s[2]\n"
        "fmul v0.4s, v27.4s, v0.s[3]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "fmla v7.4s, v20.4s, v18.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
        ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
        "ldr q19, [x21, #0x20]\n"
        "fmla v4.4s, v9.4s, v0.4s\n"
        "movi v9.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
        "fmul v8.4s, v27.4s, v26.s[0]\n"
        ".inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b\n"
        "ldr q17, [x21, #0x30]\n"
        ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
        "fmul v31.4s, v27.4s, v26.s[1]\n"
        ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
        "ldr q19, [x21, #0x40]\n"
        ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
        "fmul v15.4s, v27.4s, v26.s[2]\n"
        "fmul v27.4s, v27.4s, v26.s[3]\n"
        ".inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b\n"
        "ldr q1, [x21, #0x50]\n"
        ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
        ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
        "ldr q26, [x21, #0x60]\n"
        ".inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b\n"
        ".inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b\n"
        "ldr q21, [x21, #0x70]\n"
        "add x21, x21, #0x88\n"
        ".inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b\n"
        ".inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b\n"
        ".inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b\n"
        ".inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b\n"
        "uzp1 v29.2d, v20.2d, v18.2d\n"
        "uzp2 v21.2d, v20.2d, v18.2d\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "uzp1 v18.2d, v9.2d, v0.2d\n"
        "uzp2 v16.2d, v9.2d, v0.2d\n"
        "scvtf v21.4s, v21.4s, #0x4\n"
        "fmla v6.4s, v29.4s, v8.4s\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v30.4s, v21.4s, v31.4s\n"
        "fmla v24.4s, v18.4s, v15.4s\n"
        "fmla v14.4s, v16.4s, v27.4s\n"
        "bgt 3b\n"
        "mov x20, %x[res_ptr]\n"
        "subs x27, x27, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q10, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q28, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q11, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q7, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q4, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q6, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q24, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q14, [x20, #0x0]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x10, x10, #0x10\n"
        "cmp x10, #0x10\n"
        "mov %x[res_ptr], x26\n"
        "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
        "bge 1b\n"
        "4:"  // Row loop skip
        "cbz x10, 9f\n"
        "5:"  // Row tail: Row loop
        "add x24, %x[b_ptr], #0x8\n"
        "mov x23, %x[nc]\n"
        "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:"  // Row tail: Column loop
        "movi v2.16b, #0x0\n"
        "movi v10.16b, #0x0\n"
        "add x25, %x[a_ptr], #0x8\n"
        "mov x21, %x[nb]\n"
        "movi v12.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "7:"  // Row tail: Block loop
        "ldr q6, [x24, #0x0]\n"
        "ldr q5, [x24, #0x10]\n"
        "movi v17.16b, #0x4\n"
        "movi v8.4s, #0x0\n"
        "ldr q4, [x25, #0x0]\n"
        "ldr q13, [x25, #0x10]\n"
        "movi v27.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "ldr q31, [x24, #0x20]\n"
        "ldr q14, [x24, #0x30]\n"
        "movi v29.4s, #0x0\n"
        "movi v22.16b, #0xf0\n"
        "ldr q11, [x25, #0x20]\n"
        "ldr q23, [x25, #0x30]\n"
        "sshl v21.16b, v6.16b, v17.16b\n"
        "sshl v16.16b, v5.16b, v17.16b\n"
        "ldr q20, [x25, #0x40]\n"
        "ldr q26, [x25, #0x50]\n"
        "and v6.16b, v6.16b, v22.16b\n"
        "and v5.16b, v5.16b, v22.16b\n"
        "ldr q25, [x25, #0x60]\n"
        "ldr q3, [x25, #0x70]\n"
        "sshl v19.16b, v31.16b, v17.16b\n"
        "sshl v18.16b, v14.16b, v17.16b\n"
        "ldr d17, [x25, #-0x8]\n"
        ".inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b\n"
        ".inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b\n"
        "and v31.16b, v31.16b, v22.16b\n"
        ".inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b\n"
        ".inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b\n"
        "and v14.16b, v14.16b, v22.16b\n"
        "sub x20, x24, #0x8\n"
        "ldr d16, [x20, #0x0]\n"
        "subs x21, x21, #0x1\n"
        "add x25, x25, #0x88\n"
        "fcvtl v17.4s, v17.4h\n"
        "add x24, x24, #0x48\n"
        ".inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b\n"
        ".inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b\n"
        ".inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b\n"
        ".inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b\n"
        "fcvtl v16.4s, v16.4h\n"
        ".inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b\n"
        ".inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b\n"
        "fmul v23.4s, v16.4s, v17.s[0]\n"
        "fmul v21.4s, v16.4s, v17.s[1]\n"
        "fmul v1.4s, v16.4s, v17.s[2]\n"
        "fmul v20.4s, v16.4s, v17.s[3]\n"
        ".inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b\n"
        ".inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b\n"
        ".inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b\n"
        ".inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b\n"
        ".inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b\n"
        ".inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b\n"
        "uzp1 v19.2d, v8.2d, v27.2d\n"
        "uzp2 v18.2d, v8.2d, v27.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp1 v17.2d, v0.2d, v29.2d\n"
        "uzp2 v16.2d, v0.2d, v29.2d\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v2.4s, v19.4s, v23.4s\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v10.4s, v18.4s, v21.4s\n"
        "fmla v12.4s, v17.4s, v1.4s\n"
        "fmla v28.4s, v16.4s, v20.4s\n"
        "bgt 7b\n"
        "mov x20, %x[res_ptr]\n"
        "cmp x10, #0x1\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x2\n"
        "str q10, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x3\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "str q28, [x20, #0x0]\n"
        "8:"  // Row tail: Accumulator store skip
        "subs x23, x23, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "bne 6b\n"
        "subs x10, x10, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x9\n"
        "mov %x[res_ptr], x22\n"
        "bgt 5b\n"
        "9:"  // Row tail: Row loop skip
        : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
        : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [nb] "r" (nb), [res_stride] "r" (res_stride), [nc] "r" (nc)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    ggml_gemm_q4_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (ggml_cpu_get_sve_cnt() == QK8_0) {
        const void * b_ptr = vx;
        const void * a_ptr = vy;
        float * res_ptr = s;
        size_t res_stride = bs * sizeof(float);

        __asm__ __volatile__(
            "mov x20, #0x4\n"
            "mov x13, %x[nr]\n"
            "mov z28.s, #-0x4\n"
            "mov x12, #0x88\n"
            "ptrue p1.b\n"
            "whilelt p0.s, XZR, x20\n"
            "cmp x13, #0x10\n"
            "mul x12, %x[nb], x12\n"
            "blt 4f\n"
            "1:"  // Row loop
            "add x11, %x[b_ptr], #0x10\n"
            "mov x10, %x[nc]\n"
            "add x9, %x[res_ptr], %x[res_stride], LSL #4\n"
            "2:"  // Column loop
            "add x28, %x[a_ptr], #0x8\n"
            "mov z24.b, #0x0\n"
            "mov z15.b, #0x0\n"
            "mov x27, %x[nb]\n"
            "add x26, x28, x12\n"
            "mov z12.b, #0x0\n"
            "mov z0.b, #0x0\n"
            "add x25, x26, x12\n"
            "mov z13.b, #0x0\n"
            "mov z1.b, #0x0\n"
            "add x24, x25, x12\n"
            "mov z20.b, #0x0\n"
            "mov z25.b, #0x0\n"
            "mov z11.b, #0x0\n"
            "mov z16.b, #0x0\n"
            "mov z19.b, #0x0\n"
            "mov z26.b, #0x0\n"
            "mov z8.b, #0x0\n"
            "mov z29.b, #0x0\n"
            "mov z27.b, #0x0\n"
            "mov z10.b, #0x0\n"
            "3:"  // Block loop
            "ld1b { z30.b }, p1/Z, [x11]\n"
            "ld1b { z21.b }, p1/Z, [x11, #1, MUL VL]\n"
            "mov z18.s, #0x0\n"
            "mov z7.s, #0x0\n"
            "ld1rqb { z3.b }, p1/Z, [x28]\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #16]\n"
            "mov z9.s, #0x0\n"
            "mov z22.s, #0x0\n"
            "ld1b { z4.b }, p1/Z, [x11, #2, MUL VL]\n"
            "ld1b { z17.b }, p1/Z, [x11, #3, MUL VL]\n"
            "sub x20, x11, #0x10\n"
            "sub x23, x28, #0x8\n"
            "lsl z31.b, z30.b, #0x4\n"
            "lsl z6.b, z21.b, #0x4\n"
            "ld1h { z23.s }, p1/Z, [x20]\n"
            "sub x22, x26, #0x8\n"
            "and z30.b, z30.b, #0xf0\n"
            "and z21.b, z21.b, #0xf0\n"
            "sub x21, x25, #0x8\n"
            "sub x20, x24, #0x8\n"
            "lsl z14.b, z4.b, #0x4\n"
            "lsl z2.b, z17.b, #0x4\n"
            "subs x27, x27, #0x1\n"
            "add x11, x11, #0x90\n"
            ".inst 0x451f9872  // smmla z18.s, z3.b, z31.b\n"
            ".inst 0x45069867  // smmla z7.s, z3.b, z6.b\n"
            "ld1rqb { z3.b }, p1/Z, [x28, #32]\n"
            "and z4.b, z4.b, #0xf0\n"
            ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
            ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #48]\n"
            "and z17.b, z17.b, #0xf0\n"
            "fcvt z23.s, p1/m, z23.h\n"
            ".inst 0x450e9872  // smmla z18.s, z3.b, z14.b\n"
            ".inst 0x45029867  // smmla z7.s, z3.b, z2.b\n"
            "ld1rqb { z3.b }, p1/Z, [x28, #64]\n"
            ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
            ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #80]\n"
            "fscale z23.s, p1/m, z23.s, z28.s\n"
            ".inst 0x451e9872  // smmla z18.s, z3.b, z30.b\n"
            ".inst 0x45159867  // smmla z7.s, z3.b, z21.b\n"
            "ld1rqb { z3.b }, p1/Z, [x28, #96]\n"
            ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
            ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #112]\n"
            "add x28, x28, #0x88\n"
            ".inst 0x45049872  // smmla z18.s, z3.b, z4.b\n"
            ".inst 0x45119867  // smmla z7.s, z3.b, z17.b\n"
            "ld1h { z3.s }, p0/Z, [x23]\n"
            ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
            ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
            "fcvt z3.s, p1/m, z3.h\n"
            "uzp1 z5.d, z18.d, z7.d\n"
            "uzp2 z18.d, z18.d, z7.d\n"
            "mov z3.q, z3.q[0]\n"
            "uzp1 z7.d, z9.d, z22.d\n"
            "uzp2 z22.d, z9.d, z22.d\n"
            "fmul z9.s, z23.s, z3.s[0]\n"
            "scvtf z5.s, p1/m, z5.s\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "scvtf z7.s, p1/m, z7.s\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z24.s, p1/M, z5.s, z9.s\n"
            "ld1rqb { z5.b }, p1/Z, [x26]\n"
            "fmul z9.s, z23.s, z3.s[1]\n"
            "fmla z15.s, p1/M, z18.s, z9.s\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #16]\n"
            "fmul z9.s, z23.s, z3.s[2]\n"
            "fmul z3.s, z23.s, z3.s[3]\n"
            "fmla z12.s, p1/M, z7.s, z9.s\n"
            "mov z9.s, #0x0\n"
            "ld1h { z7.s }, p0/Z, [x22]\n"
            ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
            "fmla z0.s, p1/M, z22.s, z3.s\n"
            "mov z22.s, #0x0\n"
            "ld1h { z3.s }, p0/Z, [x21]\n"
            ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
            "ld1rqb { z5.b }, p1/Z, [x26, #32]\n"
            "fcvt z7.s, p1/m, z7.h\n"
            "fcvt z3.s, p1/m, z3.h\n"
            ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
            ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
            "ld1rqb { z5.b }, p1/Z, [x26, #64]\n"
            "mov z7.q, z7.q[0]\n"
            "mov z3.q, z3.q[0]\n"
            ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
            ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
            "ld1rqb { z5.b }, p1/Z, [x26, #96]\n"
            ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
            ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
            "uzp1 z5.d, z9.d, z22.d\n"
            "scvtf z5.s, p1/m, z5.s\n"
            "uzp2 z22.d, z9.d, z22.d\n"
            "fmul z9.s, z23.s, z7.s[0]\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z13.s, p1/M, z5.s, z9.s\n"
            "ld1rqb { z9.b }, p1/Z, [x25]\n"
            "fmul z5.s, z23.s, z7.s[1]\n"
            "fmla z1.s, p1/M, z22.s, z5.s\n"
            "mov z5.s, #0x0\n"
            "mov z22.s, #0x0\n"
            ".inst 0x451f9a45  // smmla z5.s, z18.b, z31.b\n"
            ".inst 0x45069a56  // smmla z22.s, z18.b, z6.b\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #48]\n"
            ".inst 0x450e9a45  // smmla z5.s, z18.b, z14.b\n"
            ".inst 0x45029a56  // smmla z22.s, z18.b, z2.b\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #80]\n"
            ".inst 0x451e9a45  // smmla z5.s, z18.b, z30.b\n"
            ".inst 0x45159a56  // smmla z22.s, z18.b, z21.b\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #112]\n"
            "add x26, x26, #0x88\n"
            ".inst 0x45049a45  // smmla z5.s, z18.b, z4.b\n"
            ".inst 0x45119a56  // smmla z22.s, z18.b, z17.b\n"
            "uzp1 z18.d, z5.d, z22.d\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "uzp2 z22.d, z5.d, z22.d\n"
            "fmul z5.s, z23.s, z7.s[2]\n"
            "fmul z7.s, z23.s, z7.s[3]\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z20.s, p1/M, z18.s, z5.s\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #16]\n"
            "ld1h { z5.s }, p0/Z, [x20]\n"
            "fcvt z5.s, p1/m, z5.h\n"
            "fmla z25.s, p1/M, z22.s, z7.s\n"
            "mov z22.s, #0x0\n"
            "mov z7.s, #0x0\n"
            ".inst 0x451f9936  // smmla z22.s, z9.b, z31.b\n"
            ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
            "ld1rqb { z9.b }, p1/Z, [x25, #32]\n"
            "mov z5.q, z5.q[0]\n"
            ".inst 0x450e9936  // smmla z22.s, z9.b, z14.b\n"
            ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
            "ld1rqb { z9.b }, p1/Z, [x25, #64]\n"
            ".inst 0x451e9936  // smmla z22.s, z9.b, z30.b\n"
            ".inst 0x45159927  // smmla z7.s, z9.b, z21.b\n"
            "ld1rqb { z9.b }, p1/Z, [x25, #96]\n"
            ".inst 0x45049936  // smmla z22.s, z9.b, z4.b\n"
            ".inst 0x45119927  // smmla z7.s, z9.b, z17.b\n"
            "uzp1 z9.d, z22.d, z7.d\n"
            "scvtf z9.s, p1/m, z9.s\n"
            "uzp2 z22.d, z22.d, z7.d\n"
            "fmul z7.s, z23.s, z3.s[0]\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z11.s, p1/M, z9.s, z7.s\n"
            "ld1rqb { z9.b }, p1/Z, [x24]\n"
            "fmul z7.s, z23.s, z3.s[1]\n"
            "fmla z16.s, p1/M, z22.s, z7.s\n"
            "mov z22.s, #0x0\n"
            "mov z7.s, #0x0\n"
            ".inst 0x451f9a56  // smmla z22.s, z18.b, z31.b\n"
            ".inst 0x45069a47  // smmla z7.s, z18.b, z6.b\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #48]\n"
            ".inst 0x450e9a56  // smmla z22.s, z18.b, z14.b\n"
            ".inst 0x45029a47  // smmla z7.s, z18.b, z2.b\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #80]\n"
            ".inst 0x451e9a56  // smmla z22.s, z18.b, z30.b\n"
            ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #112]\n"
            "add x25, x25, #0x88\n"
            ".inst 0x45049a56  // smmla z22.s, z18.b, z4.b\n"
            ".inst 0x45119a47  // smmla z7.s, z18.b, z17.b\n"
            "uzp1 z18.d, z22.d, z7.d\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "uzp2 z7.d, z22.d, z7.d\n"
            "fmul z22.s, z23.s, z3.s[2]\n"
            "fmul z3.s, z23.s, z3.s[3]\n"
            "scvtf z7.s, p1/m, z7.s\n"
            "fmla z19.s, p1/M, z18.s, z22.s\n"
            "ld1rqb { z18.b }, p1/Z, [x24, #16]\n"
            "fmul z22.s, z23.s, z5.s[0]\n"
            "fmla z26.s, p1/M, z7.s, z3.s\n"
            "mov z3.s, #0x0\n"
            "mov z7.s, #0x0\n"
            ".inst 0x451f9923  // smmla z3.s, z9.b, z31.b\n"
            ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
            "ld1rqb { z9.b }, p1/Z, [x24, #32]\n"
            ".inst 0x450e9923  // smmla z3.s, z9.b, z14.b\n"
            ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
            "mov z9.s, #0x0\n"
            ".inst 0x451f9a49  // smmla z9.s, z18.b, z31.b\n"
            "mov z31.s, #0x0\n"
            ".inst 0x45069a5f  // smmla z31.s, z18.b, z6.b\n"
            "ld1rqb { z6.b }, p1/Z, [x24, #48]\n"
            "ld1rqb { z18.b }, p1/Z, [x24, #64]\n"
            ".inst 0x450e98c9  // smmla z9.s, z6.b, z14.b\n"
            "fmul z14.s, z23.s, z5.s[1]\n"
            ".inst 0x450298df  // smmla z31.s, z6.b, z2.b\n"
            "ld1rqb { z6.b }, p1/Z, [x24, #80]\n"
            "fmul z2.s, z23.s, z5.s[2]\n"
            "fmul z23.s, z23.s, z5.s[3]\n"
            ".inst 0x451e9a43  // smmla z3.s, z18.b, z30.b\n"
            ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
            "ld1rqb { z5.b }, p1/Z, [x24, #96]\n"
            ".inst 0x451e98c9  // smmla z9.s, z6.b, z30.b\n"
            ".inst 0x451598df  // smmla z31.s, z6.b, z21.b\n"
            "ld1rqb { z18.b }, p1/Z, [x24, #112]\n"
            "add x24, x24, #0x88\n"
            ".inst 0x450498a3  // smmla z3.s, z5.b, z4.b\n"
            ".inst 0x451198a7  // smmla z7.s, z5.b, z17.b\n"
            ".inst 0x45049a49  // smmla z9.s, z18.b, z4.b\n"
            ".inst 0x45119a5f  // smmla z31.s, z18.b, z17.b\n"
            "uzp1 z18.d, z3.d, z7.d\n"
            "uzp2 z5.d, z3.d, z7.d\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "uzp1 z6.d, z9.d, z31.d\n"
            "uzp2 z9.d, z9.d, z31.d\n"
            "scvtf z5.s, p1/m, z5.s\n"
            "fmla z8.s, p1/M, z18.s, z22.s\n"
            "scvtf z6.s, p1/m, z6.s\n"
            "scvtf z9.s, p1/m, z9.s\n"
            "fmla z29.s, p1/M, z5.s, z14.s\n"
            "fmla z27.s, p1/M, z6.s, z2.s\n"
            "fmla z10.s, p1/M, z9.s, z23.s\n"
            "bgt 3b\n"
            "mov x20, %x[res_ptr]\n"
            "subs x10, x10, #0x8\n"
            "add %x[res_ptr], %x[res_ptr], #0x20\n"
            "st1w { z24.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z15.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z12.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z0.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z13.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z1.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z20.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z25.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z11.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z16.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z19.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z26.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z8.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z29.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z27.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z10.s }, p1, [x20]\n"
            "bne 2b\n"
            "mov x20, #0x4\n"
            "sub x13, x13, #0x10\n"
            "cmp x13, #0x10\n"
            "mov %x[res_ptr], x9\n"
            "madd %x[a_ptr], x20, x12, %x[a_ptr]\n"
            "bge 1b\n"
            "4:"  // Row loop skip
            "cbz x13, 9f\n"
            "5:"  // Row tail: Row loop
            "add x25, %x[b_ptr], #0x10\n"
            "mov x24, %x[nc]\n"
            "add x23, %x[res_ptr], %x[res_stride], LSL #2\n"
            "6:"  // Row tail: Column loop
            "mov z24.b, #0x0\n"
            "mov z15.b, #0x0\n"
            "add x28, %x[a_ptr], #0x8\n"
            "mov x22, %x[nb]\n"
            "mov z12.b, #0x0\n"
            "mov z0.b, #0x0\n"
            "7:"  // Row tail: Block loop
            "ld1b { z3.b }, p1/Z, [x25]\n"
            "ld1b { z6.b }, p1/Z, [x25, #1, MUL VL]\n"
            "mov z2.s, #0x0\n"
            "mov z25.s, #0x0\n"
            "ld1rqb { z26.b }, p1/Z, [x28]\n"
            "ld1rqb { z21.b }, p1/Z, [x28, #16]\n"
            "mov z27.s, #0x0\n"
            "mov z19.s, #0x0\n"
            "ld1b { z29.b }, p1/Z, [x25, #2, MUL VL]\n"
            "ld1b { z16.b }, p1/Z, [x25, #3, MUL VL]\n"
            "sub x21, x25, #0x10\n"
            "sub x20, x28, #0x8\n"
            "lsl z20.b, z3.b, #0x4\n"
            "lsl z4.b, z6.b, #0x4\n"
            "ld1rqb { z10.b }, p1/Z, [x28, #32]\n"
            "ld1rqb { z23.b }, p1/Z, [x28, #48]\n"
            "and z3.b, z3.b, #0xf0\n"
            "and z6.b, z6.b, #0xf0\n"
            "ld1rqb { z11.b }, p1/Z, [x28, #64]\n"
            "ld1rqb { z7.b }, p1/Z, [x28, #80]\n"
            "lsl z8.b, z29.b, #0x4\n"
            "lsl z14.b, z16.b, #0x4\n"
            "ld1rqb { z18.b }, p1/Z, [x28, #96]\n"
            "ld1rqb { z30.b }, p1/Z, [x28, #112]\n"
            ".inst 0x45149b42  // smmla z2.s, z26.b, z20.b\n"
            ".inst 0x45049b59  // smmla z25.s, z26.b, z4.b\n"
            "and z29.b, z29.b, #0xf0\n"
            "ld1h { z17.s }, p1/Z, [x21]\n"
            ".inst 0x45149abb  // smmla z27.s, z21.b, z20.b\n"
            ".inst 0x45049ab3  // smmla z19.s, z21.b, z4.b\n"
            "and z16.b, z16.b, #0xf0\n"
            "ld1h { z4.s }, p0/Z, [x20]\n"
            "subs x22, x22, #0x1\n"
            "add x28, x28, #0x88\n"
            "fcvt z17.s, p1/m, z17.h\n"
            "add x25, x25, #0x90\n"
            ".inst 0x45089942  // smmla z2.s, z10.b, z8.b\n"
            ".inst 0x450e9959  // smmla z25.s, z10.b, z14.b\n"
            "fcvt z4.s, p1/m, z4.h\n"
            ".inst 0x45089afb  // smmla z27.s, z23.b, z8.b\n"
            ".inst 0x450e9af3  // smmla z19.s, z23.b, z14.b\n"
            "fscale z17.s, p1/m, z17.s, z28.s\n"
            "mov z4.q, z4.q[0]\n"
            ".inst 0x45039962  // smmla z2.s, z11.b, z3.b\n"
            ".inst 0x45069979  // smmla z25.s, z11.b, z6.b\n"
            "fmul z23.s, z17.s, z4.s[0]\n"
            "fmul z9.s, z17.s, z4.s[1]\n"
            "fmul z21.s, z17.s, z4.s[2]\n"
            "fmul z4.s, z17.s, z4.s[3]\n"
            ".inst 0x450398fb  // smmla z27.s, z7.b, z3.b\n"
            ".inst 0x450698f3  // smmla z19.s, z7.b, z6.b\n"
            ".inst 0x451d9a42  // smmla z2.s, z18.b, z29.b\n"
            ".inst 0x45109a59  // smmla z25.s, z18.b, z16.b\n"
            ".inst 0x451d9bdb  // smmla z27.s, z30.b, z29.b\n"
            ".inst 0x45109bd3  // smmla z19.s, z30.b, z16.b\n"
            "uzp1 z31.d, z2.d, z25.d\n"
            "uzp2 z13.d, z2.d, z25.d\n"
            "scvtf z31.s, p1/m, z31.s\n"
            "uzp1 z17.d, z27.d, z19.d\n"
            "uzp2 z18.d, z27.d, z19.d\n"
            "scvtf z13.s, p1/m, z13.s\n"
            "fmla z24.s, p1/M, z31.s, z23.s\n"
            "scvtf z17.s, p1/m, z17.s\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "fmla z15.s, p1/M, z13.s, z9.s\n"
            "fmla z12.s, p1/M, z17.s, z21.s\n"
            "fmla z0.s, p1/M, z18.s, z4.s\n"
            "bgt 7b\n"
            "mov x20, %x[res_ptr]\n"
            "cmp x13, #0x1\n"
            "st1w { z24.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "cmp x13, #0x2\n"
            "st1w { z15.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "cmp x13, #0x3\n"
            "st1w { z12.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "st1w { z0.s }, p1, [x20]\n"
            "8:"  // Row tail: Accumulator store skip
            "subs x24, x24, #0x8\n"
            "add %x[res_ptr], %x[res_ptr], #0x20\n"
            "bne 6b\n"
            "subs x13, x13, #0x4\n"
            "add %x[a_ptr], %x[a_ptr], x12\n"
            "mov %x[res_ptr], x23\n"
            "bgt 5b\n"
            "9:"  // Row tail: Row loop skip
            : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
            : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [nb] "r" (nb), [res_stride] "r" (res_stride), [nc] "r" (nc)
            : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
        );
        return;
    }
#endif // #if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)

#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t kvalues = vld1q_s8(kvalues_iq4nl);

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

            float32x4_t sumf[4];
            for (int m = 0; m < 4; m++) {
                sumf[m] = vdupq_n_f32(0);
            }

            for (int l = 0; l < nb; l++) {
                float32x4_t a_d = vcvt_f32_f16(vld1_f16((const float16_t *)a_ptr[l].d));
                float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *)b_ptr[l].d));

                int32x4_t sumi_0 = vdupq_n_s32(0);
                int32x4_t sumi_1 = vdupq_n_s32(0);
                int32x4_t sumi_2 = vdupq_n_s32(0);
                int32x4_t sumi_3 = vdupq_n_s32(0);

                for (int k = 0; k < 4; k++) {
                    int8x16_t a_0 = vld1q_s8(a_ptr[l].qs + 16 * k + 0);
                    int8x16_t a_1 = vld1q_s8(a_ptr[l].qs + 16 * k + 64);

                    uint8x16_t b = vld1q_u8(b_ptr[l].qs + 16 * k);
                    int8x16_t b_hi = vqtbl1q_s8(kvalues, b >> 4);
                    int8x16_t b_lo = vqtbl1q_s8(kvalues, b & 0xF);

                    sumi_0 = vdotq_laneq_s32(sumi_0, b_lo, a_0, 0);
                    sumi_1 = vdotq_laneq_s32(sumi_1, b_lo, a_0, 1);
                    sumi_2 = vdotq_laneq_s32(sumi_2, b_lo, a_0, 2);
                    sumi_3 = vdotq_laneq_s32(sumi_3, b_lo, a_0, 3);
                    sumi_0 = vdotq_laneq_s32(sumi_0, b_hi, a_1, 0);
                    sumi_1 = vdotq_laneq_s32(sumi_1, b_hi, a_1, 1);
                    sumi_2 = vdotq_laneq_s32(sumi_2, b_hi, a_1, 2);
                    sumi_3 = vdotq_laneq_s32(sumi_3, b_hi, a_1, 3);
                }

                sumf[0] = vmlaq_f32(sumf[0], vmulq_laneq_f32(b_d, a_d, 0), vcvtq_f32_s32(sumi_0));
                sumf[1] = vmlaq_f32(sumf[1], vmulq_laneq_f32(b_d, a_d, 1), vcvtq_f32_s32(sumi_1));
                sumf[2] = vmlaq_f32(sumf[2], vmulq_laneq_f32(b_d, a_d, 2), vcvtq_f32_s32(sumi_2));
                sumf[3] = vmlaq_f32(sumf[3], vmulq_laneq_f32(b_d, a_d, 3), vcvtq_f32_s32(sumi_3));
            }

            for (int m = 0; m < 4; m++) {
                vst1q_f32(s + (y * 4 + m) * bs + x * 4, sumf[m]);
            }
        }
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    ggml_gemm_iq4_nl_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

static inline void print_s8x16(const char *label, uint8x16_t v, bool print) {
    if (!print) return;
    int8_t tmp[16];
    vst1q_s8(tmp, v); // store vector to memory
    printf("%s:  ", label);
    for (int i = 0; i < 16; i++) {
        printf("%02x ", (uint8_t)tmp[i]);
        if ((i + 1) % 8 == 0) printf("- ");  // wrap lines
    }
    printf("\n");
}

static inline void print_u8x16(const char *label, uint8x16_t v, bool print) {
    if (!print) return;
    uint8_t tmp[16];
    vst1q_u8(tmp, v); // store vector to memory
    printf("%s:  ", label);
    for (int i = 0; i < 16; i++) {
        printf("%02x ", tmp[i]);
        if ((i + 1) % 8 == 0) printf("- ");  // wrap lines
    }
    printf("\n");
}

static inline void print_u16x8(const char *label, uint16x8_t v, bool print) {
    if (!print) return;
    uint16_t tmp[8];
    vst1q_u16(tmp, v);

    printf("%s:  ", label);
    for (int i = 0; i < 8; i++) {
        printf("%04x ", tmp[i]);
    }
    printf("\n");
}


static inline void print_s32x4(const char *label, int32x4_t v, bool print) {
    if (!print) return;
    int32_t tmp[4];
    vst1q_s32(tmp, v);
    printf("%s:  %d %d %d %d\n", label, tmp[0], tmp[1], tmp[2], tmp[3]);
}


static inline uint64_t ns_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

void ggml_gemm_q4_K_8x8_q8_K(int                        n, float * GGML_RESTRICT      s,
                             size_t                     bs, const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy, int                        nr,
                             int                        nc) {
    constexpr int qk = QK_K;
    const int nb = n / qk;

    constexpr int ncols_interleaved = 8;
    constexpr int blocklen = 8;
    constexpr int q8_k_blocklen = 4;

    constexpr uint32_t kmask1 = 0x3f3f3f3f;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    constexpr uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    const uint8x16_t m4b = vdupq_n_u8(0x0f);

    float sumf[blocklen][ncols_interleaved];
    float sum_minf[blocklen][ncols_interleaved];

    uint32_t utmp[32];

    int sumi1;
    int sumi2;
    int sumi;

    static bool print_block = true;
    if (print_block) {
        printf("\n[DEBUG q4_K repack] ----\n");
        const block_q4_Kx8 * q4_ptr = static_cast<const block_q4_Kx8*>(vx);
        for (int sample = 0; sample < 2; sample++) {
            printf("\nvy.qs (interleaved):\n");
            for (int i = 0; i < 512; i++) {       // print first 256 bytes of result
                printf("%02x ", ((const uint8_t*)q4_ptr[sample].qs)[i]);
                if ((i + 1) % 8 == 0) printf("- ");  // wrap lines
                if ((i + 1) % 64 == 0) printf("\n");  // wrap lines
            }
            printf("\n------------------------------------------------\n");
        }
        printf("\n[DEBUG q8_K block] ----\n");
        const block_q8_Kx4 *q8_ptr = static_cast<const block_q8_Kx4*>(vy);
        for (int sample = 0; sample < 2; sample++) {
            printf("\nq8_ptr->qs (interleaved) sample %d:\n", sample);
            const uint8_t *qs = reinterpret_cast<const uint8_t *>(q8_ptr->qs);
            for (int i = 0; i < 256; i++) {
                printf("%02x ", qs[256 * sample + i]);
                if ((i + 1) % 8 == 0)  printf("- ");
                if ((i + 1) % 64 == 0) printf("\n");
            }
            printf("\n------------------------------------------------\n");
        }
    }

    // INFO: TILING
    // TODO: Fix boundaries for the tiling once the inner mul_mat is done
    constexpr int y_step = blocklen / q8_k_blocklen; // 8 / 4 == 2
    for (int y = 0; y < nr / q8_k_blocklen; y += y_step) {
        const block_q8_Kx4 * GGML_RESTRICT q8_ptr = (const block_q8_Kx4 *) vy + (y * nb);

        // TODO: Fix boundaries for the tiling once the inner mul_mat is done
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * GGML_RESTRICT q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);

            // TODO: float32x4x8
            for (int m = 0; m < blocklen; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }

            // 8 accumulators: 2 row pairs × 4 col pairs
            int32x4_t acc[8];
            for (int i = 0; i < 8; ++i) {
                acc[i] = vdupq_n_s32(0);
            }
            // INFO: Computation of a single 8x4 block
            for (int b = 0; b < nb; b++) {

                // two rows in q4_Kx8 matches a single row in q8_Kx4
                int32x4_t sb_acc[4];
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // decode scales and mins
                    // TODO: Need lo and hi scales 2*12 = 24 bytes per sb, 4 sbs -> 4 * 24 = 96 bytes total
                    int8_t q4sb_scales[2][8];
                    int16x8_t q4sb_mins[2];  // int16 as its needed for bias correction later
                    {  //r0-7 0..31
                        uint32_t scales_mins[3];
                        memcpy(scales_mins, &q4_ptr[b].scales[sb * 24], 12);
                        const uint32_t mins_0_3 = scales_mins[1] & kmask1;
                        const uint32_t mins_4_7 = ((scales_mins[2] >> 4) & kmask2) | (((scales_mins[1] >> 6) & kmask3) << 4);
                        const uint32x2_t mins = {mins_0_3, mins_4_7};
                        q4sb_mins[0] = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins)));
                        uint32_t scales[2];
                        scales[0] = scales_mins[0] & kmask1; // scales 0~3
                        scales[1] = (scales_mins[2] & kmask2) | (((scales_mins[0] >> 6) & kmask3) << 4); // scales 4~7
                        memcpy(q4sb_scales[0], scales, 8);
                    }
                    {  //r 0-7 32..64
                        uint32_t scales_mins[3];
                        memcpy(scales_mins, &q4_ptr[b].scales[sb * 24 + 12], 12);
                        const uint32_t mins_0_3 = scales_mins[1] & kmask1;
                        const uint32_t mins_4_7 = ((scales_mins[2] >> 4) & kmask2) | (((scales_mins[1] >> 6) & kmask3) << 4);
                        const uint32x2_t mins = {mins_0_3, mins_4_7};
                        q4sb_mins[1] = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins)));
                        uint32_t scales[2];
                        scales[0] = scales_mins[0] & kmask1; // scales 0~3
                        scales[1] = (scales_mins[2] & kmask2) | (((scales_mins[0] >> 6) & kmask3) << 4); // scales 4~7
                        memcpy(q4sb_scales[1], scales, 8);
                    }

                    if (b == 0) print_block = true;
                    if(print_block) printf(" SB == %d (B = %d) -------  \n", sb, b);
                    if(print_block) printf(" blk = %d ", 2 * sb);
                    if(print_block) printf("q4_scales_lo: ");
                    for (int ii = 0; ii < 8; ii++) {
                        if(print_block) printf("%d ", q4sb_scales[0][ii]);
                    }
                    if(print_block) printf("\n");
                    if(print_block) printf(" blk = %d ", 2 * sb + 1);
                    if(print_block) printf("q4_scales_hi: ");
                    for (int ii = 0; ii < 8; ii++) {
                        if(print_block) printf("%d ", q4sb_scales[1][ii]);
                    }
                    if(print_block) printf("\n");
                    print_u16x8("q4sb_mins_lo", q4sb_mins[0], print_block);
                    print_u16x8("q4sb_mins_hi", q4sb_mins[1], print_block);
                    print_block = false;

                    // TODO: x4 reads
                    uint8x16_t q4_qs_01_0 = vld1q_u8(q4_ptr[b].qs + sb * QK_K);
                    uint8x16_t q4_qs_23_0 = vld1q_u8(q4_ptr[b].qs + 16 + sb * QK_K);
                    uint8x16_t q4_qs_45_0 = vld1q_u8(q4_ptr[b].qs + 32 + sb * QK_K);
                    uint8x16_t q4_qs_67_0 = vld1q_u8(q4_ptr[b].qs + 48 + sb * QK_K);

                    print_u8x16("q4_qs_01_0", q4_qs_01_0, print_block);
                    print_u8x16("q4_qs_23_0", q4_qs_23_0, print_block);
                    print_u8x16("q4_qs_45_0", q4_qs_45_0, print_block);
                    print_u8x16("q4_qs_67_0", q4_qs_67_0, print_block);

                    uint8x16_t q4_qs_01_1 = vld1q_u8(q4_ptr[b].qs + 64 + sb * QK_K);
                    uint8x16_t q4_qs_23_1 = vld1q_u8(q4_ptr[b].qs + 80 + sb * QK_K);
                    uint8x16_t q4_qs_45_1 = vld1q_u8(q4_ptr[b].qs + 96 + sb * QK_K);
                    uint8x16_t q4_qs_67_1 = vld1q_u8(q4_ptr[b].qs + 112 + sb * QK_K);
                    print_u8x16("q4_qs_01_1", q4_qs_01_1, print_block);
                    print_u8x16("q4_qs_23_1", q4_qs_23_1, print_block);
                    print_u8x16("q4_qs_45_1", q4_qs_45_1, print_block);
                    print_u8x16("q4_qs_67_1", q4_qs_67_1, print_block);

                    uint8x16_t q4_qs_01_2 = vld1q_u8(q4_ptr[b].qs + 128 + sb * QK_K);
                    uint8x16_t q4_qs_23_2 = vld1q_u8(q4_ptr[b].qs + 144 + sb * QK_K);
                    uint8x16_t q4_qs_45_2 = vld1q_u8(q4_ptr[b].qs + 160 + sb * QK_K);
                    uint8x16_t q4_qs_67_2 = vld1q_u8(q4_ptr[b].qs + 176 + sb * QK_K);
                    print_u8x16("q4_qs_01_2", q4_qs_01_2, print_block);
                    print_u8x16("q4_qs_23_2", q4_qs_23_2, print_block);
                    print_u8x16("q4_qs_45_2", q4_qs_45_2, print_block);
                    print_u8x16("q4_qs_67_2", q4_qs_67_2, print_block);

                    uint8x16_t q4_qs_01_3 = vld1q_u8(q4_ptr[b].qs + 192 +sb * QK_K);
                    uint8x16_t q4_qs_23_3 = vld1q_u8(q4_ptr[b].qs + 208 + sb * QK_K);
                    uint8x16_t q4_qs_45_3 = vld1q_u8(q4_ptr[b].qs + 224 + sb * QK_K);
                    uint8x16_t q4_qs_67_3 = vld1q_u8(q4_ptr[b].qs + 240 + sb * QK_K);
                    print_u8x16("q4_qs_01_3", q4_qs_01_3, print_block);
                    print_u8x16("q4_qs_23_3", q4_qs_23_3, print_block);
                    print_u8x16("q4_qs_45_3", q4_qs_45_3, print_block);
                    print_u8x16("q4_qs_67_3", q4_qs_67_3, print_block);

                    int8x16_t q8_qs_01_0 = vld1q_s8(q8_ptr[b].qs + sb * 256 +   0);  // Q8-01, 0..7
                    int8x16_t q8_qs_23_0 = vld1q_s8(q8_ptr[b].qs + sb * 256 +  16);  // Q8-23, 0..7
                    int8x16_t q8_qs_01_1 = vld1q_s8(q8_ptr[b].qs + sb * 256 +  32);  // Q8-01, 8..15
                    int8x16_t q8_qs_23_1 = vld1q_s8(q8_ptr[b].qs + sb * 256 +  48);  // Q8-23, 8..15

                    int8x16_t q8_qs_01_2 = vld1q_s8(q8_ptr[b].qs + sb * 256 +  64);  // Q8-01, 16..23
                    int8x16_t q8_qs_23_2 = vld1q_s8(q8_ptr[b].qs + sb * 256 +  80);  // Q8-23, 16..23
                    int8x16_t q8_qs_01_3 = vld1q_s8(q8_ptr[b].qs + sb * 256 +  96);  // Q8-01, 24..31
                    int8x16_t q8_qs_23_3 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 112);  // Q8-23, 24..31

                    int8x16_t q8_qs_01_4 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 128);  // Q8-01, 32..39
                    int8x16_t q8_qs_23_4 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 144);  // Q8-23, 32..39
                    int8x16_t q8_qs_01_5 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 160);  // Q8-01, 40..47
                    int8x16_t q8_qs_23_5 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 176);  // Q8-23, 40..47

                    int8x16_t q8_qs_01_6 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 192);  // Q8-01, 48..55
                    int8x16_t q8_qs_23_6 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 208);  // Q8-23, 48..55
                    int8x16_t q8_qs_01_7 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 224);  // Q8-01, 56..63
                    int8x16_t q8_qs_23_7 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 240);  // Q8-23, 56..63

                    // ---- q4_qs_01 ----
                    for (int i = 0; i < 4; ++i) {
                        sb_acc[i] = vdupq_n_s32(0);
                    }

                    int8x16_t q4_qs_01_0_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_01_0, m4b)); // Q4-01 0..7
                    int8x16_t q4_qs_01_1_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_01_1, m4b)); // Q4-01 8..15
                    int8x16_t q4_qs_01_2_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_01_2, m4b)); // Q4-01 16..23
                    int8x16_t q4_qs_01_3_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_01_3, m4b)); // Q4-01 24..31
                    int8x16_t q4_qs_01_0_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_01_0, 4)); // Q4-01 32..39
                    int8x16_t q4_qs_01_1_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_01_1, 4)); // Q4-01 40..47
                    int8x16_t q4_qs_01_2_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_01_2, 4)); // Q4-01 48..55
                    int8x16_t q4_qs_01_3_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_01_3, 4)); // Q4-01 56..63

                    // 01-01
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_01_0_lo, q8_qs_01_0); // 0011 blk0
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_01_1_lo, q8_qs_01_1);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_01_2_lo, q8_qs_01_2);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_01_3_lo, q8_qs_01_3);

                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_01_0_hi, q8_qs_01_4); // 0011 blk1
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_01_1_hi, q8_qs_01_5);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_01_2_hi, q8_qs_01_6);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_01_3_hi, q8_qs_01_7);

                    // 23-01
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_01_0_lo, q8_qs_23_0); // 0011 blk0
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_01_1_lo, q8_qs_23_1);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_01_2_lo, q8_qs_23_2);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_01_3_lo, q8_qs_23_3);

                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_01_0_hi, q8_qs_23_4); // 0011 blk1
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_01_1_hi, q8_qs_23_5);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_01_2_hi, q8_qs_23_6);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_01_3_hi, q8_qs_23_7);


                    int blk = 0;
                    int row_begin = 0;
                    for (int tid = 0; tid < 4; tid++) {
                        const int32x4_t block_scale = {
                            (int32_t) q4sb_scales[blk][0],
                            (int32_t) q4sb_scales[blk][0],
                            (int32_t) q4sb_scales[blk][1],
                            (int32_t) q4sb_scales[blk][1],
                        };
                        // if (print_block) printf("-- ACC = %d (0011 blk %d) ", tid, blk);
                        blk ^= 1;
                        // if (b == 0) print_block = true;
                        // print_s32x4("block_scale", block_scale, print_block);
                        // print_s32x4("sb_acc", sb_acc[tid], print_block);
                        acc[tid / 2 + row_begin] = vmlaq_s32(acc[tid / 2 + row_begin], sb_acc[tid], block_scale);
                        // print_s32x4("acc[tid]", acc[tid / 2], print_block);
                        print_block=false;
                    }

                    // ---- q4_qs_23 ----
                    for (int i = 0; i < 4; ++i) {
                        sb_acc[i] = vdupq_n_s32(0);
                    }
                    int8x16_t q4_qs_23_0_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_23_0, m4b)); // Q4-23 0..7
                    int8x16_t q4_qs_23_1_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_23_1, m4b)); // Q4-23 8..15
                    int8x16_t q4_qs_23_2_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_23_2, m4b)); // Q4-23 16..23
                    int8x16_t q4_qs_23_3_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_23_3, m4b)); // Q4-23 24..31
                    int8x16_t q4_qs_23_0_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_23_0, 4)); // Q4-23 32..39
                    int8x16_t q4_qs_23_1_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_23_1, 4)); // Q4-23 40..47
                    int8x16_t q4_qs_23_2_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_23_2, 4)); // Q4-23 48..55
                    int8x16_t q4_qs_23_3_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_23_3, 4)); // Q4-23 56..63

                    // 01-23
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_23_0_lo, q8_qs_01_0);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_23_1_lo, q8_qs_01_1);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_23_2_lo, q8_qs_01_2);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_23_3_lo, q8_qs_01_3);

                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_23_0_hi, q8_qs_01_4);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_23_1_hi, q8_qs_01_5);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_23_2_hi, q8_qs_01_6);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_23_3_hi, q8_qs_01_7);

                    // 23-23
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_23_0_lo, q8_qs_23_0);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_23_1_lo, q8_qs_23_1);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_23_2_lo, q8_qs_23_2);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_23_3_lo, q8_qs_23_3);

                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_23_0_hi, q8_qs_23_4);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_23_1_hi, q8_qs_23_5);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_23_2_hi, q8_qs_23_6);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_23_3_hi, q8_qs_23_7);

                    blk = 0;
                    row_begin += 2;
                    for (int tid = 0; tid < 4; tid++) {
                        const int32x4_t block_scale = {
                            (int32_t) q4sb_scales[blk][2],
                            (int32_t) q4sb_scales[blk][2],
                            (int32_t) q4sb_scales[blk][3],
                            (int32_t) q4sb_scales[blk][3],
                        };
                        if (b == 0) print_block = true;
                        if (print_block) printf("-- ACC = %d (begin %d blk %d) ", tid, row_begin, blk);
                        blk ^= 1;
                        print_s32x4("block_scale", block_scale, print_block);
                        print_s32x4("sb_acc", sb_acc[tid], print_block);
                        acc[tid / 2 + row_begin] = vmlaq_s32(acc[tid / 2 + row_begin], sb_acc[tid], block_scale);
                        print_s32x4("acc[tid]", acc[tid / 2 + row_begin], print_block);
                        print_block=false;
                    }


                    // ---- q4_qs_45 ----
                    for (int i = 0; i < 4; ++i) {
                        sb_acc[i] = vdupq_n_s32(0);
                    }
                    int8x16_t q4_qs_45_0_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_45_0, m4b)); // Q4-45 0..7
                    int8x16_t q4_qs_45_1_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_45_1, m4b)); // Q4-45 8..15
                    int8x16_t q4_qs_45_2_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_45_2, m4b)); // Q4-45 16..23
                    int8x16_t q4_qs_45_3_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_45_3, m4b)); // Q4-45 24..31
                    int8x16_t q4_qs_45_0_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_45_0, 4)); // Q4-45 32..39
                    int8x16_t q4_qs_45_1_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_45_1, 4)); // Q4-45 40..47
                    int8x16_t q4_qs_45_2_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_45_2, 4)); // Q4-45 48..55
                    int8x16_t q4_qs_45_3_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_45_3, 4)); // Q4-45 56..63
                    // 01-45
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_45_0_lo, q8_qs_01_0);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_45_1_lo, q8_qs_01_1);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_45_2_lo, q8_qs_01_2);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_45_3_lo, q8_qs_01_3);

                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_45_0_hi, q8_qs_01_4);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_45_1_hi, q8_qs_01_5);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_45_2_hi, q8_qs_01_6);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_45_3_hi, q8_qs_01_7);

                    // 23-45
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_45_0_lo, q8_qs_23_0);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_45_1_lo, q8_qs_23_1);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_45_2_lo, q8_qs_23_2);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_45_3_lo, q8_qs_23_3);

                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_45_0_hi, q8_qs_23_4);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_45_1_hi, q8_qs_23_5);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_45_2_hi, q8_qs_23_6);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_45_3_hi, q8_qs_23_7);

                    blk = 0;
                    row_begin += 2;
                    for (int tid = 0; tid < 4; tid++) {
                        const int32x4_t block_scale = {
                            (int32_t) q4sb_scales[blk][4],
                            (int32_t) q4sb_scales[blk][4],
                            (int32_t) q4sb_scales[blk][5],
                            (int32_t) q4sb_scales[blk][5],
                        };
                        if (b == 0) print_block = true;
                        if (print_block) printf("-- ACC = %d (begin %d blk %d) ", tid, row_begin, blk);
                        blk ^= 1;
                        print_s32x4("block_scale", block_scale, print_block);
                        print_s32x4("sb_acc", sb_acc[tid], print_block);
                        acc[tid / 2 + row_begin] = vmlaq_s32(acc[tid / 2 + row_begin], sb_acc[tid], block_scale);
                        print_s32x4("acc[tid]", acc[tid / 2 + row_begin], print_block);
                        print_block=false;
                    }

                    // ---- q4_qs_67 ----
                    for (int i = 0; i < 4; ++i) {
                        sb_acc[i] = vdupq_n_s32(0);
                    }
                    int8x16_t q4_qs_67_0_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_67_0, m4b)); // Q4-67 0..7
                    int8x16_t q4_qs_67_1_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_67_1, m4b)); // Q4-67 8..15
                    int8x16_t q4_qs_67_2_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_67_2, m4b)); // Q4-67 16..23
                    int8x16_t q4_qs_67_3_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs_67_3, m4b)); // Q4-67 24..31
                    int8x16_t q4_qs_67_0_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_67_0, 4)); // Q4-67 32..39
                    int8x16_t q4_qs_67_1_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_67_1, 4)); // Q4-67 40..47
                    int8x16_t q4_qs_67_2_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_67_2, 4)); // Q4-67 48..55
                    int8x16_t q4_qs_67_3_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_67_3, 4)); // Q4-67 56..63
                    // 01-67
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_67_0_lo, q8_qs_01_0);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_67_1_lo, q8_qs_01_1);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_67_2_lo, q8_qs_01_2);
                    sb_acc[0] = vmmlaq_s32(sb_acc[0], q4_qs_67_3_lo, q8_qs_01_3);

                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_67_0_hi, q8_qs_01_4);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_67_1_hi, q8_qs_01_5);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_67_2_hi, q8_qs_01_6);
                    sb_acc[1] = vmmlaq_s32(sb_acc[1], q4_qs_67_3_hi, q8_qs_01_7);

                    // 23-67
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_67_0_lo, q8_qs_23_0);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_67_1_lo, q8_qs_23_1);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_67_2_lo, q8_qs_23_2);
                    sb_acc[2] = vmmlaq_s32(sb_acc[2], q4_qs_67_3_lo, q8_qs_23_3);

                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_67_0_hi, q8_qs_23_4);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_67_1_hi, q8_qs_23_5);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_67_2_hi, q8_qs_23_6);
                    sb_acc[3] = vmmlaq_s32(sb_acc[3], q4_qs_67_3_hi, q8_qs_23_7);

                    blk = 0;
                    row_begin += 2;
                    for (int tid = 0; tid < 4; tid++) {
                        const int32x4_t block_scale = {
                            (int32_t) q4sb_scales[blk][6],
                            (int32_t) q4sb_scales[blk][6],
                            (int32_t) q4sb_scales[blk][7],
                            (int32_t) q4sb_scales[blk][7],
                        };
                        if (b == 0) print_block = true;
                        if (print_block) printf("-- ACC = %d (row %d blk %d) ", tid, row_begin, blk);
                        blk ^= 1;
                        print_s32x4("block_scale", block_scale, print_block);
                        print_s32x4("sb_acc", sb_acc[tid], print_block);
                        acc[tid / 2 + row_begin] = vmlaq_s32(acc[tid / 2 + row_begin], sb_acc[tid], block_scale);
                        print_s32x4("acc[tid]", acc[tid / 2 + 6], print_block);
                        print_block=false;
                    }

                    // Acc has 00, 01, 10, 11
                    // Will need Accs for:
                    // 00, 01, 10, 11, sb_acc[0]
                    // 20, 21, 30, 31, sb_acc[1]
                    //
                    // 02, 03, 12, 13 sb_acc[2]
                    // 22, 23, 32, 33 sb_acc[3]
                    //
                    // 04, 05, 14, 15 sb_acc[4]
                    // 24, 25, 34, 35 sb_acc[5]
                    //
                    // 06, 07, 16, 17 sb_acc[6]
                    // 26, 27, 36, 37 sb_acc[7]

                    // if (sb == 1) print_block = false;

                    }
            }
            GGML_ABORT(" A" );


            // for (int l = 0; l < nb; l++) {
            //     for (int sb = 0; sb < 8; sb++) {
            //         memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
            //         utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
            //         const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
            //         utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
            //         utmp[sb * 4 + 2] = uaux_0;
            //         utmp[sb * 4 + 0] &= kmask1;
            //     }
            //     for (int k = 0; k < (qk / (2 * blocklen)); k++) {
            //         uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
            //         uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
            //         for (int m = 0; m < 4; m++) {
            //             for (int j = 0; j < ncols_interleaved; j++) {
            //                 sumi1 = 0;
            //                 sumi2 = 0;
            //                 sumi = 0;
            //                 for (int i = 0; i < blocklen; ++i) {
            //                     const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
            //                     const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
            //                     sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
            //                     sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
            //                     sumi1 = sumi1 * scales_0[j];
            //                     sumi2 = sumi2 * scales_1[j];
            //                     sumi += sumi1 + sumi2;
            //                 }
            //                 sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
            //             }
            //         }
            //     }
            //     for (int sb = 0; sb < 8; sb++) {
            //         uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
            //         for(int m = 0; m < 4; m++) {
            //             const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
            //             for(int j = 0; j < ncols_interleaved; j++) {
            //                 sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
            //             }
            //         }
            //     }
            // }
            // for (int m = 0; m < 4; m++) {
            //     for (int j = 0; j < ncols_interleaved; j++) {
            //         s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
            //     }
            // }
        }
    }
}

