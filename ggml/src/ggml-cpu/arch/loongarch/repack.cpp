#include <lasxintrin.h>

#include <cstdint>
#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cassert>
#include <cmath>
#include <cstdio>   // for GGML_ASSERT
#include <cstdlib>  // for qsort
#include <cstring>

#define GGML_CPU_CLANG_WORKAROUND
#include "../../repack.h"

#if defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

#if defined(__loongarch_asx)
static inline __m128 lasx_extract_128_lo_s(__m256 a) {
    __m128 r;
    __asm__("" : "=f"(r) : "0"(a));
    return r;
}

static inline __m128 lasx_extract_128_hi_s(__m256 a) {
    __m256i t = __lasx_xvpermi_d((__m256i) a, 0x0e);
    __m128  r;
    __asm__("" : "=f"(r) : "0"(t));
    return r;
}
#endif

void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__loongarch_asx)
    float  id[4];
    __m256 srcv[4][4];
    __m256 idvec[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            __m256 v0 = (__m256) __lasx_xvld(x + row_iter * k + i * 32, 0);
            __m256 v1 = (__m256) __lasx_xvld(x + row_iter * k + i * 32, 32);
            __m256 v2 = (__m256) __lasx_xvld(x + row_iter * k + i * 32, 64);
            __m256 v3 = (__m256) __lasx_xvld(x + row_iter * k + i * 32, 96);

            const __m256 sign_bit = (__m256) __lasx_xvreplgr2vr_w(0x80000000);
            __m256       max_abs  = (__m256) __lasx_xvandn_v((__m256i) sign_bit, (__m256i) v0);
            max_abs = __lasx_xvfmax_s(max_abs, (__m256) __lasx_xvandn_v((__m256i) sign_bit, (__m256i) v1));
            max_abs = __lasx_xvfmax_s(max_abs, (__m256) __lasx_xvandn_v((__m256i) sign_bit, (__m256i) v2));
            max_abs = __lasx_xvfmax_s(max_abs, (__m256) __lasx_xvandn_v((__m256i) sign_bit, (__m256i) v3));

            __m128 max4 = __lsx_vfmax_s(lasx_extract_128_hi_s(max_abs), lasx_extract_128_lo_s(max_abs));
            max4        = __lsx_vfmax_s(max4, (__m128) __lsx_vshuf4i_w((__m128i) max4, 0x4e));
            max4        = __lsx_vfmax_s(max4, (__m128) __lsx_vshuf4i_w((__m128i) max4, 0xb1));
            float max_scalar;
            __asm__("" : "=f"(max_scalar) : "0"(max4));

            // Divided by 127.f to mirror results in quantize_row_q8_0
            const float d = max_scalar / 127.f;
            id[row_iter]  = (max_scalar != 0.0f) ? 127.f / max_scalar : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);

            srcv[row_iter][0] = v0;
            srcv[row_iter][1] = v1;
            srcv[row_iter][2] = v2;
            srcv[row_iter][3] = v3;
            idvec[row_iter]   = (__m256) __lasx_xvldrepl_w(id + row_iter, 0);
        }

        for (int j = 0; j < 4; j++) {
            __m256 v0 = __lasx_xvfmul_s(srcv[0][j], idvec[0]);
            __m256 v1 = __lasx_xvfmul_s(srcv[1][j], idvec[1]);
            __m256 v2 = __lasx_xvfmul_s(srcv[2][j], idvec[2]);
            __m256 v3 = __lasx_xvfmul_s(srcv[3][j], idvec[3]);

            __m256i i0 = __lasx_xvftintrne_w_s(v0);
            __m256i i1 = __lasx_xvftintrne_w_s(v1);
            __m256i i2 = __lasx_xvftintrne_w_s(v2);
            __m256i i3 = __lasx_xvftintrne_w_s(v3);

            i0 = __lasx_xvssrani_h_w(i1, i0, 0);
            i2 = __lasx_xvssrani_h_w(i3, i2, 0);
            i0 = __lasx_xvssrani_b_h(i2, i0, 0);

            const int     perm_mask[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };
            const __m256i perm         = __lasx_xvld(perm_mask, 0);
            i0                         = __lasx_xvperm_w(i0, perm);

            __lasx_xvst(i0, y[i].qs + 32 * j, 0);
        }
    }

#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x8_generic(x, vy, k);
#endif
}

#if defined(__loongarch_asx)
static inline __m256i lasx_q4_0_dot(__m256i dot, __m256i b_lo, __m256i b_hi, __m256i a_lo, __m256i a_hi) {
    dot = __lasx_xvmaddwev_h_b(dot, b_lo, a_lo);
    dot = __lasx_xvmaddwod_h_b(dot, b_lo, a_lo);
    dot = __lasx_xvmaddwev_h_b(dot, b_hi, a_hi);
    return __lasx_xvmaddwod_h_b(dot, b_hi, a_hi);
}

static inline __m256i lasx_q4_0_sum(__m256i cols_0_3, __m256i cols_4_7) {
    // Eight products of signed 4-bit and 8-bit values fit in int16.
    cols_0_3 = __lasx_xvhaddw_w_h(cols_0_3, cols_0_3);
    cols_4_7 = __lasx_xvhaddw_w_h(cols_4_7, cols_4_7);
    return __lasx_xvpermi_d(
        __lasx_xvadd_w(__lasx_xvpickev_w(cols_4_7, cols_0_3), __lasx_xvpickod_w(cols_4_7, cols_0_3)), 0xd8);
}
#endif

void ggml_gemv_q4_0_8x8_q8_0(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
#if defined(__loongarch_asx)
    assert(n % QK4_0 == 0);
    assert(nc % 8 == 0);
    assert(nr == 1);
    UNUSED(bs);
    UNUSED(nr);

    const int          nb    = n / QK4_0;
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / 8; x++) {
        const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + x * nb;
        __m256               sumf  = (__m256) __lasx_xvldi(0);
        for (int l = 0; l < nb; l++) {
            __m256i sum0 = __lasx_xvldi(0);
            __m256i sum1 = __lasx_xvldi(0);
            for (int k = 0; k < 2; k++) {
                const __m256i a_lo  = __lasx_xvldrepl_d(a_ptr[l].qs + k * 8, 0);
                const __m256i a_hi  = __lasx_xvldrepl_d(a_ptr[l].qs + k * 8, 16);
                const __m256i b0    = __lasx_xvld(b_ptr[l].qs + k * 64, 0);
                const __m256i b1    = __lasx_xvld(b_ptr[l].qs + k * 64, 32);
                const __m256i b0_lo = __lasx_xvsrai_b(__lasx_xvslli_b(b0, 4), 4);
                const __m256i b0_hi = __lasx_xvsrai_b(b0, 4);
                const __m256i b1_lo = __lasx_xvsrai_b(__lasx_xvslli_b(b1, 4), 4);
                const __m256i b1_hi = __lasx_xvsrai_b(b1, 4);
                sum0                = lasx_q4_0_dot(sum0, b0_lo, b0_hi, a_lo, a_hi);
                sum1                = lasx_q4_0_dot(sum1, b1_lo, b1_hi, a_lo, a_hi);
            }
            const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(&a_ptr[l].d, 0));
            const __m256 b_d = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].d, 0), 0x10));
            const __m256 dot = __lasx_xvffint_s_w(lasx_q4_0_sum(sum0, sum1));
            sumf             = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf);
        }
        __lasx_xvst(sumf, s + x * 8, 0);
    }
#else
    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_q4_0_8x8_q8_0(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
#if defined(__loongarch_asx)
    assert(n % QK4_0 == 0);
    assert(nr % 4 == 0);
    assert(nc % 8 == 0);

    const int nb = n / QK4_0;
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + y * nb;
        for (int x = 0; x < nc / 8; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + x * nb;
            __m256               sumf[4];
            for (int m = 0; m < 4; m++) {
                sumf[m] = (__m256) __lasx_xvldi(0);
            }
            for (int l = 0; l < nb; l++) {
                __m256i sum0_0 = __lasx_xvldi(0);
                __m256i sum0_1 = __lasx_xvldi(0);
                __m256i sum0_2 = __lasx_xvldi(0);
                __m256i sum0_3 = __lasx_xvldi(0);
                __m256i sum1_0 = __lasx_xvldi(0);
                __m256i sum1_1 = __lasx_xvldi(0);
                __m256i sum1_2 = __lasx_xvldi(0);
                __m256i sum1_3 = __lasx_xvldi(0);
                for (int k = 0; k < 2; k++) {
                    const __m256i a_lo = __lasx_xvld(a_ptr[l].qs + k * 32, 0);
                    const __m256i a_hi = __lasx_xvld(a_ptr[l].qs + k * 32, 64);
                    const __m256i b    = __lasx_xvld(b_ptr[l].qs + k * 64, 0);
                    const __m256i b_lo = __lasx_xvsrai_b(__lasx_xvslli_b(b, 4), 4);
                    const __m256i b_hi = __lasx_xvsrai_b(b, 4);
                    sum0_0 = lasx_q4_0_dot(sum0_0, b_lo, b_hi, __lasx_xvreplve0_d(a_lo), __lasx_xvreplve0_d(a_hi));
                    sum0_1 =
                        lasx_q4_0_dot(sum0_1, b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0x55), __lasx_xvpermi_d(a_hi, 0x55));
                    sum0_2 =
                        lasx_q4_0_dot(sum0_2, b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xaa), __lasx_xvpermi_d(a_hi, 0xaa));
                    sum0_3 =
                        lasx_q4_0_dot(sum0_3, b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xff), __lasx_xvpermi_d(a_hi, 0xff));
                }
                for (int k = 0; k < 2; k++) {
                    const __m256i a_lo = __lasx_xvld(a_ptr[l].qs + k * 32, 0);
                    const __m256i a_hi = __lasx_xvld(a_ptr[l].qs + k * 32, 64);
                    const __m256i b    = __lasx_xvld(b_ptr[l].qs + k * 64, 32);
                    const __m256i b_lo = __lasx_xvsrai_b(__lasx_xvslli_b(b, 4), 4);
                    const __m256i b_hi = __lasx_xvsrai_b(b, 4);
                    sum1_0 = lasx_q4_0_dot(sum1_0, b_lo, b_hi, __lasx_xvreplve0_d(a_lo), __lasx_xvreplve0_d(a_hi));
                    sum1_1 =
                        lasx_q4_0_dot(sum1_1, b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0x55), __lasx_xvpermi_d(a_hi, 0x55));
                    sum1_2 =
                        lasx_q4_0_dot(sum1_2, b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xaa), __lasx_xvpermi_d(a_hi, 0xaa));
                    sum1_3 =
                        lasx_q4_0_dot(sum1_3, b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xff), __lasx_xvpermi_d(a_hi, 0xff));
                }
                const __m256 b_d = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].d, 0), 0x10));
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 0, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_q4_0_sum(sum0_0, sum1_0));
                    sumf[0]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[0]);
                }
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 1, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_q4_0_sum(sum0_1, sum1_1));
                    sumf[1]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[1]);
                }
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 2, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_q4_0_sum(sum0_2, sum1_2));
                    sumf[2]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[2]);
                }
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 3, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_q4_0_sum(sum0_3, sum1_3));
                    sumf[3]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[3]);
                }
            }
            for (int m = 0; m < 4; m++) {
                __lasx_xvst(sumf[m], s + (y * 4 + m) * bs + x * 8, 0);
            }
        }
    }
#else
    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __loongarch_asx
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
#else
ggml_gemv_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#ifdef __loongarch_asx
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
#else
    ggml_gemm_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
#endif
}
