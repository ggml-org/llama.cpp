#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../quants.h"
#include "../../ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

#define UNUSED GGML_UNUSED

#if defined(__e2k__) && __iset__ >= 5

static inline float e2k_hsum_float_4(__m128 v) {
    __m128 t = _mm_add_ps(v, _mm_movehl_ps(v, v));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}

#endif

#if defined(__e2k__) && __iset__ >= 7

// Byte permutation across two vectors using e2k qppermb
// Equivalent to _mm_shuffle_epi8 but with second source
#define e2k_mm_shuffle2_epi8(a, b, c) \
    ((__m128i)__builtin_e2k_qppermb((__v2di)(b), (__v2di)(a), (__v2di)(c)))

static inline __v2di e2k_dot_q4_0_q8_0_quants_native(__v2di bx, __v2di by0, __v2di by1) {
    const __v2di lowMask = __builtin_e2k_qppackdl(0x0f0f0f0f0f0f0f0fLL, 0x0f0f0f0f0f0f0f0fLL);
    const __v2di bias     = __builtin_e2k_qppackdl(0x0808080808080808LL, 0x0808080808080808LL);

    __v2di bx0 = __builtin_e2k_qpand(bx, lowMask);
    __v2di bx1 = __builtin_e2k_qpsrlh(bx, 4);
    bx1 = __builtin_e2k_qpand(bx1, lowMask);

    bx0 = __builtin_e2k_qpsubb(bx0, bias);
    bx1 = __builtin_e2k_qpsubb(bx1, bias);

    __v2di dot = __builtin_e2k_qpidotsbwss(bx0, by0, __builtin_e2k_qppackdl(0, 0));
    dot = __builtin_e2k_qpidotsbwss(bx1, by1, dot);
    return dot;
}

#endif

#if defined(__e2k__) && __iset__ >= 5

static inline __m128i e2k_dot_q4_0_q8_0_half(__m128i bx, __m128i by) {
    __m128i c8 = _mm_set1_epi8(8);
    __m128i sy = _mm_maddubs_epi16(c8, by);
    __m128i dot = _mm_maddubs_epi16(bx, by);
    return _mm_sub_epi16(dot, sy);
}

void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                             const void * GGML_RESTRICT vx, size_t bx,
                             const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    __m128 acc = _mm_setzero_ps();

    int ib;

#if defined(__e2k__) && __iset__ >= 7
#pragma loop count(1000)
    for (ib = 0; ib < nb - 3; ib += 4, x += 4, y += 4) {
        __v2di bx0 = *((const __v2di *)x[0].qs);
        __v2di bx1 = *((const __v2di *)x[1].qs);
        __v2di bx2 = *((const __v2di *)x[2].qs);
        __v2di bx3 = *((const __v2di *)x[3].qs);

        __v2di by0l = ((const __v2di *)y[0].qs)[0];
        __v2di by0h = ((const __v2di *)y[0].qs)[1];
        __v2di by1l = ((const __v2di *)y[1].qs)[0];
        __v2di by1h = ((const __v2di *)y[1].qs)[1];
        __v2di by2l = ((const __v2di *)y[2].qs)[0];
        __v2di by2h = ((const __v2di *)y[2].qs)[1];
        __v2di by3l = ((const __v2di *)y[3].qs)[0];
        __v2di by3h = ((const __v2di *)y[3].qs)[1];

        __v2di d0 = e2k_dot_q4_0_q8_0_quants_native(bx0, by0l, by0h);
        __v2di d1 = e2k_dot_q4_0_q8_0_quants_native(bx1, by1l, by1h);
        __v2di d2 = e2k_dot_q4_0_q8_0_quants_native(bx2, by2l, by2h);
        __v2di d3 = e2k_dot_q4_0_q8_0_quants_native(bx3, by3l, by3h);

        __m128i s01 = _mm_hadd_epi32((__m128i)d0, (__m128i)d1);
        s01 = _mm_hadd_epi32(s01, _mm_setzero_si128());

        __m128i s23 = _mm_hadd_epi32((__m128i)d2, (__m128i)d3);
        s23 = _mm_hadd_epi32(s23, _mm_setzero_si128());

        __m128i sums = _mm_unpacklo_epi64(s01, s23);

        __m128 fsum = _mm_cvtepi32_ps(sums);

        float xd[4] = {
            GGML_CPU_FP16_TO_FP32(x[0].d),
            GGML_CPU_FP16_TO_FP32(x[1].d),
            GGML_CPU_FP16_TO_FP32(x[2].d),
            GGML_CPU_FP16_TO_FP32(x[3].d),
        };
        float yd[4] = {
            GGML_CPU_FP16_TO_FP32(y[0].d),
            GGML_CPU_FP16_TO_FP32(y[1].d),
            GGML_CPU_FP16_TO_FP32(y[2].d),
            GGML_CPU_FP16_TO_FP32(y[3].d),
        };

        __m128 xv = _mm_loadu_ps(xd);
        __m128 yv = _mm_loadu_ps(yd);

#if __iset__ >= 6
        acc = _mm_fmadd_ps(_mm_mul_ps(xv, yv), fsum, acc);
#else
        acc = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(xv, yv), fsum), acc);
#endif
    }
#else
    ib = 0;
#endif

    for (; ib < nb; ++ib) {
        __m128i qx = _mm_loadu_si128((const __m128i *)x[ib].qs);

        __m128i qxl = _mm_and_si128(qx, _mm_set1_epi8(0x0F));
        __m128i qxh = _mm_and_si128(_mm_srli_epi16(qx, 4), _mm_set1_epi8(0x0F));

        __m128i qyl = _mm_loadu_si128((const __m128i *)y[ib].qs);
        __m128i qyh = _mm_loadu_si128((const __m128i *)(y[ib].qs + 16));

        __m128i dotl = e2k_dot_q4_0_q8_0_half(qxl, qyl);
        __m128i doth = e2k_dot_q4_0_q8_0_half(qxh, qyh);

        __m128i sum = _mm_add_epi32(
            _mm_madd_epi16(dotl, _mm_set1_epi16(1)),
            _mm_madd_epi16(doth, _mm_set1_epi16(1)));

        __m128 fdot = _mm_cvtepi32_ps(sum);

        float d = GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
#if __iset__ >= 6
        __m128 fd = _mm_set1_ps(d);
        acc = _mm_fmadd_ps(fd, fdot, acc);
#else
        __m128 fd = _mm_set1_ps(d);
        acc = _mm_add_ps(_mm_mul_ps(fd, fdot), acc);
#endif
    }

    *s = e2k_hsum_float_4(acc);
}

#else

void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                             const void * GGML_RESTRICT vx, size_t bx,
                             const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
}
#endif
