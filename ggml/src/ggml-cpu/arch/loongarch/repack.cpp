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

static inline void lasx_decode_q4_K_scales(const uint8_t * scales, uint32_t * out) {
    const int     perm[8]      = { 0, 1, 2, 2, 3, 4, 5, 5 };
    const int     last_perm[8] = { 2, 3, 4, 4, 5, 6, 7, 7 };
    const int     shifts[8]    = { 0, 4, 0, 4, 0, 4, 0, 4 };
    const __m256i shift        = __lasx_xvld(shifts, 0);
    for (int i = 0; i < 4; i++) {
        const int     offset  = i < 3 ? i * 24 : 64;
        const __m256i indices = __lasx_xvld(i < 3 ? perm : last_perm, 0);
        const __m256i raw     = __lasx_xvperm_w(__lasx_xvld(scales + offset, 0), indices);
        const __m256i low     = __lasx_xvandi_b(raw, 0x3f);
        __m256i       high    = __lasx_xvslli_b(__lasx_xvsrli_b(raw, 6), 4);
        __m256i       nibble  = __lasx_xvsrl_w(__lasx_xvshuf4i_w(raw, 0xaa), shift);
        high                  = __lasx_xvor_v(high, __lasx_xvandi_b(nibble, 0x0f));
        __lasx_xvst(__lasx_xvilvl_w(high, low), out + i * 8, 0);
    }
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

#if defined(__loongarch_asx)
static inline __m256i lasx_iq4_nl_values() {
    return __lasx_xvilvl_d(__lasx_xvldrepl_d(kvalues_iq4nl, 8), __lasx_xvldrepl_d(kvalues_iq4nl, 0));
}

static inline __m256i lasx_iq4_nl_dot(__m256i b_lo, __m256i b_hi, __m256i a_lo, __m256i a_hi) {
    __m256i lo = __lasx_xvmulwev_h_b(b_lo, a_lo);
    __m256i hi = __lasx_xvmulwev_h_b(b_hi, a_hi);
    lo         = __lasx_xvmaddwod_h_b(lo, b_lo, a_lo);
    hi         = __lasx_xvmaddwod_h_b(hi, b_hi, a_hi);
    // Two products fit in int16; widen before accumulating more products.
    return __lasx_xvadd_w(__lasx_xvhaddw_w_h(lo, lo), __lasx_xvhaddw_w_h(hi, hi));
}

static inline __m256i lasx_iq4_nl_sum(__m256i cols_0_3, __m256i cols_4_7) {
    return __lasx_xvpermi_d(
        __lasx_xvadd_w(__lasx_xvpickev_w(cols_4_7, cols_0_3), __lasx_xvpickod_w(cols_4_7, cols_0_3)), 0xd8);
}
#endif

void ggml_gemv_iq4_nl_8x8_q8_0(int                        n,
                               float * GGML_RESTRICT      s,
                               size_t                     bs,
                               const void * GGML_RESTRICT vx,
                               const void * GGML_RESTRICT vy,
                               int                        nr,
                               int                        nc) {
#if defined(__loongarch_asx)
    assert(n % QK4_NL == 0);
    assert(nc % 8 == 0);
    assert(nr == 1);
    UNUSED(bs);
    UNUSED(nr);

    const int          nb     = n / QK4_NL;
    const __m256i      values = lasx_iq4_nl_values();
    const block_q8_0 * a_ptr  = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / 8; x++) {
        const block_iq4_nlx8 * b_ptr = (const block_iq4_nlx8 *) vx + x * nb;
        __m256                 sumf  = (__m256) __lasx_xvldi(0);
        for (int l = 0; l < nb; l++) {
            __m256i sum0 = __lasx_xvldi(0);
            __m256i sum1 = __lasx_xvldi(0);
            for (int k = 0; k < 2; k++) {
                const __m256i a_lo  = __lasx_xvldrepl_d(a_ptr[l].qs + k * 8, 0);
                const __m256i a_hi  = __lasx_xvldrepl_d(a_ptr[l].qs + k * 8, 16);
                const __m256i b0    = __lasx_xvld(b_ptr[l].qs + k * 64, 0);
                const __m256i b1    = __lasx_xvld(b_ptr[l].qs + k * 64, 32);
                const __m256i b0_lo = __lasx_xvshuf_b(values, values, __lasx_xvandi_b(b0, 0x0f));
                const __m256i b0_hi = __lasx_xvshuf_b(values, values, __lasx_xvsrli_b(b0, 4));
                const __m256i b1_lo = __lasx_xvshuf_b(values, values, __lasx_xvandi_b(b1, 0x0f));
                const __m256i b1_hi = __lasx_xvshuf_b(values, values, __lasx_xvsrli_b(b1, 4));
                sum0                = __lasx_xvadd_w(sum0, lasx_iq4_nl_dot(b0_lo, b0_hi, a_lo, a_hi));
                sum1                = __lasx_xvadd_w(sum1, lasx_iq4_nl_dot(b1_lo, b1_hi, a_lo, a_hi));
            }
            const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(&a_ptr[l].d, 0));
            const __m256 b_d = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].d, 0), 0x10));
            const __m256 dot = __lasx_xvffint_s_w(lasx_iq4_nl_sum(sum0, sum1));
            sumf             = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf);
        }
        __lasx_xvst(sumf, s + x * 8, 0);
    }
#else
    ggml_gemv_iq4_nl_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_iq4_nl_8x8_q8_0(int                        n,
                               float * GGML_RESTRICT      s,
                               size_t                     bs,
                               const void * GGML_RESTRICT vx,
                               const void * GGML_RESTRICT vy,
                               int                        nr,
                               int                        nc) {
#if defined(__loongarch_asx)
    assert(n % QK4_NL == 0);
    assert(nr % 4 == 0);
    assert(nc % 8 == 0);

    const int     nb     = n / QK4_NL;
    const __m256i values = lasx_iq4_nl_values();
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + y * nb;
        for (int x = 0; x < nc / 8; x++) {
            const block_iq4_nlx8 * b_ptr = (const block_iq4_nlx8 *) vx + x * nb;
            __m256                 sumf[4];
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
                    const __m256i b_lo = __lasx_xvshuf_b(values, values, __lasx_xvandi_b(b, 0x0f));
                    const __m256i b_hi = __lasx_xvshuf_b(values, values, __lasx_xvsrli_b(b, 4));
                    sum0_0             = __lasx_xvadd_w(
                        sum0_0, lasx_iq4_nl_dot(b_lo, b_hi, __lasx_xvreplve0_d(a_lo), __lasx_xvreplve0_d(a_hi)));
                    sum0_1 = __lasx_xvadd_w(sum0_1, lasx_iq4_nl_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0x55),
                                                                    __lasx_xvpermi_d(a_hi, 0x55)));
                    sum0_2 = __lasx_xvadd_w(sum0_2, lasx_iq4_nl_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xaa),
                                                                    __lasx_xvpermi_d(a_hi, 0xaa)));
                    sum0_3 = __lasx_xvadd_w(sum0_3, lasx_iq4_nl_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xff),
                                                                    __lasx_xvpermi_d(a_hi, 0xff)));
                }
                for (int k = 0; k < 2; k++) {
                    const __m256i a_lo = __lasx_xvld(a_ptr[l].qs + k * 32, 0);
                    const __m256i a_hi = __lasx_xvld(a_ptr[l].qs + k * 32, 64);
                    const __m256i b    = __lasx_xvld(b_ptr[l].qs + k * 64, 32);
                    const __m256i b_lo = __lasx_xvshuf_b(values, values, __lasx_xvandi_b(b, 0x0f));
                    const __m256i b_hi = __lasx_xvshuf_b(values, values, __lasx_xvsrli_b(b, 4));
                    sum1_0             = __lasx_xvadd_w(
                        sum1_0, lasx_iq4_nl_dot(b_lo, b_hi, __lasx_xvreplve0_d(a_lo), __lasx_xvreplve0_d(a_hi)));
                    sum1_1 = __lasx_xvadd_w(sum1_1, lasx_iq4_nl_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0x55),
                                                                    __lasx_xvpermi_d(a_hi, 0x55)));
                    sum1_2 = __lasx_xvadd_w(sum1_2, lasx_iq4_nl_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xaa),
                                                                    __lasx_xvpermi_d(a_hi, 0xaa)));
                    sum1_3 = __lasx_xvadd_w(sum1_3, lasx_iq4_nl_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xff),
                                                                    __lasx_xvpermi_d(a_hi, 0xff)));
                }
                const __m256 b_d = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].d, 0), 0x10));
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 0, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_iq4_nl_sum(sum0_0, sum1_0));
                    sumf[0]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[0]);
                }
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 1, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_iq4_nl_sum(sum0_1, sum1_1));
                    sumf[1]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[1]);
                }
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 2, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_iq4_nl_sum(sum0_2, sum1_2));
                    sumf[2]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[2]);
                }
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 3, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_iq4_nl_sum(sum0_3, sum1_3));
                    sumf[3]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[3]);
                }
            }
            for (int m = 0; m < 4; m++) {
                __lasx_xvst(sumf[m], s + (y * 4 + m) * bs + x * 8, 0);
            }
        }
    }
#else
    ggml_gemm_iq4_nl_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

#if defined(__loongarch_asx)
static inline __m256i lasx_q4_1_dot(__m256i b_lo, __m256i b_hi, __m256i a_lo, __m256i a_hi) {
    __m256i dot = __lasx_xvmulwev_h_b(b_lo, a_lo);
    dot         = __lasx_xvmaddwod_h_b(dot, b_lo, a_lo);
    dot         = __lasx_xvmaddwev_h_b(dot, b_hi, a_hi);
    return __lasx_xvmaddwod_h_b(dot, b_hi, a_hi);
}

static inline __m256i lasx_q4_1_sum(__m256i cols_0_3, __m256i cols_4_7) {
    // Eight products of a 4-bit value and a signed byte fit in int16.
    return lasx_iq4_nl_sum(__lasx_xvhaddw_w_h(cols_0_3, cols_0_3), __lasx_xvhaddw_w_h(cols_4_7, cols_4_7));
}
#endif

void ggml_gemv_q4_1_8x8_q8_1(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
#if defined(__loongarch_asx)
    assert(n % QK4_1 == 0);
    assert(nc % 8 == 0);
    assert(nr == 1);
    UNUSED(bs);
    UNUSED(nr);

    const int          nb    = n / QK4_1;
    const block_q8_1 * a_ptr = (const block_q8_1 *) vy;
    for (int x = 0; x < nc / 8; x++) {
        const block_q4_1x8 * b_ptr = (const block_q4_1x8 *) vx + x * nb;
        __m256               sumf  = (__m256) __lasx_xvldi(0);
        for (int l = 0; l < nb; l++) {
            __m256i sum0 = __lasx_xvldi(0);
            __m256i sum1 = __lasx_xvldi(0);
            for (int k = 0; k < 2; k++) {
                const __m256i a_lo  = __lasx_xvldrepl_d(a_ptr[l].qs + k * 8, 0);
                const __m256i a_hi  = __lasx_xvldrepl_d(a_ptr[l].qs + k * 8, 16);
                const __m256i b0    = __lasx_xvld(b_ptr[l].qs + k * 64, 0);
                const __m256i b1    = __lasx_xvld(b_ptr[l].qs + k * 64, 32);
                const __m256i b0_lo = __lasx_xvandi_b(b0, 0x0f);
                const __m256i b0_hi = __lasx_xvsrli_b(b0, 4);
                const __m256i b1_lo = __lasx_xvandi_b(b1, 0x0f);
                const __m256i b1_hi = __lasx_xvsrli_b(b1, 4);
                sum0                = __lasx_xvadd_h(sum0, lasx_q4_1_dot(b0_lo, b0_hi, a_lo, a_hi));
                sum1                = __lasx_xvadd_h(sum1, lasx_q4_1_dot(b1_lo, b1_hi, a_lo, a_hi));
            }
            const __m256 a_d =
                __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(&a_ptr[l].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d, 0));
            const __m256 b_d = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].d, 0), 0x10));
            const __m256 a_s =
                __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(&a_ptr[l].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.s, 0));
            const __m256 b_m = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].m, 0), 0x10));
            const __m256 dot = __lasx_xvffint_s_w(lasx_q4_1_sum(sum0, sum1));
            sumf             = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf);
            sumf             = __lasx_xvfmadd_s(a_s, b_m, sumf);
        }
        __lasx_xvst(sumf, s + x * 8, 0);
    }
#else
    ggml_gemv_q4_1_8x8_q8_1_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_q4_1_8x8_q8_1(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
#if defined(__loongarch_asx)
    assert(n % QK4_1 == 0);
    assert(nr % 4 == 0);
    assert(nc % 8 == 0);

    const int nb = n / QK4_1;
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_1x4 * a_ptr = (const block_q8_1x4 *) vy + y * nb;
        for (int x = 0; x < nc / 8; x++) {
            const block_q4_1x8 * b_ptr = (const block_q4_1x8 *) vx + x * nb;
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
                    const __m256i b_lo = __lasx_xvandi_b(b, 0x0f);
                    const __m256i b_hi = __lasx_xvsrli_b(b, 4);
                    sum0_0             = __lasx_xvadd_h(
                        sum0_0, lasx_q4_1_dot(b_lo, b_hi, __lasx_xvreplve0_d(a_lo), __lasx_xvreplve0_d(a_hi)));
                    sum0_1 = __lasx_xvadd_h(
                        sum0_1, lasx_q4_1_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0x55), __lasx_xvpermi_d(a_hi, 0x55)));
                    sum0_2 = __lasx_xvadd_h(
                        sum0_2, lasx_q4_1_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xaa), __lasx_xvpermi_d(a_hi, 0xaa)));
                    sum0_3 = __lasx_xvadd_h(
                        sum0_3, lasx_q4_1_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xff), __lasx_xvpermi_d(a_hi, 0xff)));
                }
                for (int k = 0; k < 2; k++) {
                    const __m256i a_lo = __lasx_xvld(a_ptr[l].qs + k * 32, 0);
                    const __m256i a_hi = __lasx_xvld(a_ptr[l].qs + k * 32, 64);
                    const __m256i b    = __lasx_xvld(b_ptr[l].qs + k * 64, 32);
                    const __m256i b_lo = __lasx_xvandi_b(b, 0x0f);
                    const __m256i b_hi = __lasx_xvsrli_b(b, 4);
                    sum1_0             = __lasx_xvadd_h(
                        sum1_0, lasx_q4_1_dot(b_lo, b_hi, __lasx_xvreplve0_d(a_lo), __lasx_xvreplve0_d(a_hi)));
                    sum1_1 = __lasx_xvadd_h(
                        sum1_1, lasx_q4_1_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0x55), __lasx_xvpermi_d(a_hi, 0x55)));
                    sum1_2 = __lasx_xvadd_h(
                        sum1_2, lasx_q4_1_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xaa), __lasx_xvpermi_d(a_hi, 0xaa)));
                    sum1_3 = __lasx_xvadd_h(
                        sum1_3, lasx_q4_1_dot(b_lo, b_hi, __lasx_xvpermi_d(a_lo, 0xff), __lasx_xvpermi_d(a_hi, 0xff)));
                }
                const __m256 b_m = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].m, 0), 0x10));
                const __m256 b_d = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].d, 0), 0x10));
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 0, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_q4_1_sum(sum0_0, sum1_0));
                    sumf[0]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[0]);
                    const __m256 a_s = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].s + 0, 0));
                    sumf[0]          = __lasx_xvfmadd_s(a_s, b_m, sumf[0]);
                }
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 1, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_q4_1_sum(sum0_1, sum1_1));
                    sumf[1]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[1]);
                    const __m256 a_s = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].s + 1, 0));
                    sumf[1]          = __lasx_xvfmadd_s(a_s, b_m, sumf[1]);
                }
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 2, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_q4_1_sum(sum0_2, sum1_2));
                    sumf[2]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[2]);
                    const __m256 a_s = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].s + 2, 0));
                    sumf[2]          = __lasx_xvfmadd_s(a_s, b_m, sumf[2]);
                }
                {
                    const __m256 a_d = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 3, 0));
                    const __m256 dot = __lasx_xvffint_s_w(lasx_q4_1_sum(sum0_3, sum1_3));
                    sumf[3]          = __lasx_xvfmadd_s(dot, __lasx_xvfmul_s(a_d, b_d), sumf[3]);
                    const __m256 a_s = __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].s + 3, 0));
                    sumf[3]          = __lasx_xvfmadd_s(a_s, b_m, sumf[3]);
                }
            }
            for (int m = 0; m < 4; m++) {
                __lasx_xvst(sumf[m], s + (y * 4 + m) * bs + x * 8, 0);
            }
        }
    }
#else
    ggml_gemm_q4_1_8x8_q8_1_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_quantize_mat_q8_K_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

#if defined(__loongarch_asx)
    float  iscale[4];
    __m256 iscale_vec[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            const float * src = x + row_iter * k + i * QK_K;
            __m256        v0  = (__m256) __lasx_xvld(src + 0, 0);
            __m256        v1  = (__m256) __lasx_xvld(src + 8, 0);
            __m256        v2  = (__m256) __lasx_xvld(src + 16, 0);
            __m256        v3  = (__m256) __lasx_xvld(src + 24, 0);

            __m256 max0 = v0;
            __m256 max1 = v1;
            __m256 max2 = v2;
            __m256 max3 = v3;
            __m256 min0 = v0;
            __m256 min1 = v1;
            __m256 min2 = v2;
            __m256 min3 = v3;

            for (int sb = 1; sb < 8; sb++) {
                v0   = (__m256) __lasx_xvld(src + sb * 32 + 0, 0);
                v1   = (__m256) __lasx_xvld(src + sb * 32 + 8, 0);
                v2   = (__m256) __lasx_xvld(src + sb * 32 + 16, 0);
                v3   = (__m256) __lasx_xvld(src + sb * 32 + 24, 0);
                max0 = __lasx_xvfmax_s(max0, v0);
                max1 = __lasx_xvfmax_s(max1, v1);
                max2 = __lasx_xvfmax_s(max2, v2);
                max3 = __lasx_xvfmax_s(max3, v3);
                min0 = __lasx_xvfmin_s(min0, v0);
                min1 = __lasx_xvfmin_s(min1, v1);
                min2 = __lasx_xvfmin_s(min2, v2);
                min3 = __lasx_xvfmin_s(min3, v3);
            }

            max0        = __lasx_xvfmax_s(__lasx_xvfmax_s(max0, max1), __lasx_xvfmax_s(max2, max3));
            __m128 max4 = __lsx_vfmax_s(lasx_extract_128_hi_s(max0), lasx_extract_128_lo_s(max0));
            max4        = __lsx_vfmax_s(max4, (__m128) __lsx_vshuf4i_w((__m128i) max4, 0x4e));
            max4        = __lsx_vfmax_s(max4, (__m128) __lsx_vshuf4i_w((__m128i) max4, 0xb1));
            float max_scalar;
            __asm__("" : "=f"(max_scalar) : "0"(max4));
            min0        = __lasx_xvfmin_s(__lasx_xvfmin_s(min0, min1), __lasx_xvfmin_s(min2, min3));
            __m128 min4 = __lsx_vfmin_s(lasx_extract_128_hi_s(min0), lasx_extract_128_lo_s(min0));
            min4        = __lsx_vfmin_s(min4, (__m128) __lsx_vshuf4i_w((__m128i) min4, 0x4e));
            min4        = __lsx_vfmin_s(min4, (__m128) __lsx_vshuf4i_w((__m128i) min4, 0xb1));
            float min_scalar;
            __asm__("" : "=f"(min_scalar) : "0"(min4));

            max_scalar           = max_scalar >= -min_scalar ? max_scalar : min_scalar;
            iscale[row_iter]     = max_scalar ? -127.f / max_scalar : 0.0f;
            y[i].d[row_iter]     = max_scalar ? 1.0f / iscale[row_iter] : 0.0f;
            iscale_vec[row_iter] = (__m256) __lasx_xvldrepl_w(iscale + row_iter, 0);
        }

        const int     perm_mask[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };
        const __m256i perm         = __lasx_xvld(perm_mask, 0);
        for (int sb = 0; sb < 4; sb++) {
            __m256i sums[4];
            for (int g = 0; g < 4; g++) {
                __m256i quants[2];
                for (int t = 0; t < 2; t++) {
                    const int     j   = sb * 8 + g * 2 + t;
                    const float * src = x + i * QK_K + j * 8;
                    __m256i       i0 =
                        __lasx_xvftintrne_w_s(__lasx_xvfmul_s((__m256) __lasx_xvld(src + 0 * k, 0), iscale_vec[0]));
                    __m256i i1 =
                        __lasx_xvftintrne_w_s(__lasx_xvfmul_s((__m256) __lasx_xvld(src + 1 * k, 0), iscale_vec[1]));
                    __m256i i2 =
                        __lasx_xvftintrne_w_s(__lasx_xvfmul_s((__m256) __lasx_xvld(src + 2 * k, 0), iscale_vec[2]));
                    __m256i i3 =
                        __lasx_xvftintrne_w_s(__lasx_xvfmul_s((__m256) __lasx_xvld(src + 3 * k, 0), iscale_vec[3]));
                    i0        = __lasx_xvssrani_h_w(i1, i0, 0);
                    i2        = __lasx_xvssrani_h_w(i3, i2, 0);
                    i0        = __lasx_xvssrani_b_h(i2, i0, 0);
                    quants[t] = __lasx_xvperm_w(i0, perm);
                    __lasx_xvst(quants[t], y[i].qs + 32 * j, 0);
                }
                __m256i sum =
                    __lasx_xvadd_h(__lasx_xvhaddw_h_b(quants[0], quants[0]), __lasx_xvhaddw_h_b(quants[1], quants[1]));
                sum     = __lasx_xvhaddw_w_h(sum, sum);
                sums[g] = __lasx_xvhaddw_d_w(sum, sum);
            }
            const __m256i sums01 = __lasx_xvssrani_w_d(sums[1], sums[0], 0);
            const __m256i sums23 = __lasx_xvssrani_w_d(sums[3], sums[2], 0);
            const __m256i packed = __lasx_xvssrani_h_w(sums23, sums01, 0);
            const __m256i even   = __lasx_xvpickev_h(packed, packed);
            const __m256i odd    = __lasx_xvpickod_h(packed, packed);
            __lasx_xvst(__lasx_xvilvl_d(odd, even), y[i].bsums + sb * 16, 0);
        }
    }
#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_K_4x8_generic(x, vy, k);
#endif
}

void ggml_gemv_q4_K_8x8_q8_K(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
#if defined(__loongarch_asx)
    const int qk                = QK_K;
    const int nb                = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen          = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    uint32_t utmp[32];

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + x * nb;

        __m256 vsumf = (__m256) __lasx_xvldi(0);
        __m256 vminf = (__m256) __lasx_xvldi(0);
        for (int l = 0; l < nb; l++) {
            lasx_decode_q4_K_scales(b_ptr[l].scales, utmp);
            __m256 a_scale  = (__m256) __lasx_xvldrepl_w(&a_ptr[l].d, 0);
            __m256 b_scale  = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].d, 0), 0x10));
            __m256 ab_scale = __lasx_xvfmul_s(a_scale, b_scale);
            for (int k = 0; k < 4; k++) {
                __m256i sum_lo_0 = __lasx_xvldi(0);
                __m256i sum_lo_1 = __lasx_xvldi(0);
                __m256i sum_hi_0 = __lasx_xvldi(0);
                __m256i sum_hi_1 = __lasx_xvldi(0);
                for (int j = 0; j < 4; j++) {
                    const uint8_t * b     = b_ptr[l].qs + k * 256 + j * 64;
                    const int8_t *  a     = a_ptr[l].qs + k * 64 + j * blocklen;
                    const __m256i   a_lo  = __lasx_xvldrepl_d(a, 0);
                    const __m256i   a_hi  = __lasx_xvldrepl_d(a, 32);
                    const __m256i   b0    = __lasx_xvld(b, 0);
                    const __m256i   b1    = __lasx_xvld(b, 32);
                    const __m256i   b0_lo = __lasx_xvandi_b(b0, 0x0f);
                    const __m256i   b1_lo = __lasx_xvandi_b(b1, 0x0f);
                    const __m256i   b0_hi = __lasx_xvsrli_b(b0, 4);
                    const __m256i   b1_hi = __lasx_xvsrli_b(b1, 4);

                    // Each halfword accumulates eight products, bounded by 15360.
                    sum_lo_0 = __lasx_xvmaddwev_h_b(sum_lo_0, b0_lo, a_lo);
                    sum_lo_0 = __lasx_xvmaddwod_h_b(sum_lo_0, b0_lo, a_lo);
                    sum_lo_1 = __lasx_xvmaddwev_h_b(sum_lo_1, b1_lo, a_lo);
                    sum_lo_1 = __lasx_xvmaddwod_h_b(sum_lo_1, b1_lo, a_lo);
                    sum_hi_0 = __lasx_xvmaddwev_h_b(sum_hi_0, b0_hi, a_hi);
                    sum_hi_0 = __lasx_xvmaddwod_h_b(sum_hi_0, b0_hi, a_hi);
                    sum_hi_1 = __lasx_xvmaddwev_h_b(sum_hi_1, b1_hi, a_hi);
                    sum_hi_1 = __lasx_xvmaddwod_h_b(sum_hi_1, b1_hi, a_hi);
                }
                sum_lo_0             = __lasx_xvhaddw_w_h(sum_lo_0, sum_lo_0);
                sum_lo_1             = __lasx_xvhaddw_w_h(sum_lo_1, sum_lo_1);
                sum_hi_0             = __lasx_xvhaddw_w_h(sum_hi_0, sum_hi_0);
                sum_hi_1             = __lasx_xvhaddw_w_h(sum_hi_1, sum_hi_1);
                const __m256i sum_lo = __lasx_xvpermi_d(
                    __lasx_xvadd_w(__lasx_xvpickev_w(sum_lo_1, sum_lo_0), __lasx_xvpickod_w(sum_lo_1, sum_lo_0)), 0xd8);
                const __m256i sum_hi = __lasx_xvpermi_d(
                    __lasx_xvadd_w(__lasx_xvpickev_w(sum_hi_1, sum_hi_0), __lasx_xvpickod_w(sum_hi_1, sum_hi_0)), 0xd8);
                const __m256i scales   = __lasx_xvld((uint8_t *) utmp + k * 32, 0);
                const __m256i scale_lo = __lasx_vext2xv_w_b(scales);
                const __m256i scale_hi = __lasx_vext2xv_w_b(__lasx_xvpermi_q(scales, scales, 0x03));
                const __m256i sum      = __lasx_xvmadd_w(__lasx_xvmul_w(sum_lo, scale_lo), sum_hi, scale_hi);
                vsumf                  = __lasx_xvfmadd_s(__lasx_xvffint_s_w(sum), ab_scale, vsumf);
            }
            {
                __m256i mins_0_1 = __lasx_xvld(utmp, 8);
                __m256i mins_0   = __lasx_vext2xv_w_b(mins_0_1);
                __m256i mins_1   = __lasx_vext2xv_w_b(__lasx_xvpermi_q(mins_0_1, mins_0_1, 0x03));
                __m256i mins_2_3 = __lasx_xvld(utmp, 40);
                __m256i mins_2   = __lasx_vext2xv_w_b(mins_2_3);
                __m256i mins_3   = __lasx_vext2xv_w_b(__lasx_xvpermi_q(mins_2_3, mins_2_3, 0x03));
                __m256i mins_4_5 = __lasx_xvld(utmp, 72);
                __m256i mins_4   = __lasx_vext2xv_w_b(mins_4_5);
                __m256i mins_5   = __lasx_vext2xv_w_b(__lasx_xvpermi_q(mins_4_5, mins_4_5, 0x03));
                __m256i mins_6_7 = __lasx_xvpermi_d(__lasx_xvld(utmp, 96), 0xf5);
                __m256i mins_6   = __lasx_vext2xv_w_b(mins_6_7);
                __m256i mins_7   = __lasx_vext2xv_w_b(__lasx_xvpermi_q(mins_6_7, mins_6_7, 0x03));

                __m256i vbsums = __lasx_xvld(a_ptr[l].bsums, 0);
                vbsums         = __lasx_xvhaddw_w_h(vbsums, vbsums);

                __m256i bias = __lasx_xvldi(0);

                const __m256i bsum_0 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 0));
                const __m256i bsum_1 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 1));
                const __m256i bsum_2 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 2));
                const __m256i bsum_3 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 3));
                const __m256i bsum_4 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 4));
                const __m256i bsum_5 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 5));
                const __m256i bsum_6 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 6));
                const __m256i bsum_7 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 7));

                bias = __lasx_xvmadd_w(bias, mins_0, bsum_0);
                bias = __lasx_xvmadd_w(bias, mins_1, bsum_1);
                bias = __lasx_xvmadd_w(bias, mins_2, bsum_2);
                bias = __lasx_xvmadd_w(bias, mins_3, bsum_3);
                bias = __lasx_xvmadd_w(bias, mins_4, bsum_4);
                bias = __lasx_xvmadd_w(bias, mins_5, bsum_5);
                bias = __lasx_xvmadd_w(bias, mins_6, bsum_6);
                bias = __lasx_xvmadd_w(bias, mins_7, bsum_7);

                __m256       dmin      = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].dmin, 0), 0x10));
                __m256       a_d       = (__m256) __lasx_xvldrepl_w(&a_ptr[l].d, 0);
                const __m256 min_scale = __lasx_xvfmul_s(a_d, dmin);
                vminf                  = __lasx_xvfmadd_s(__lasx_xvffint_s_w(bias), min_scale, vminf);
            }
        }
        __lasx_xvst(__lasx_xvfsub_s(vsumf, vminf), s + x * ncols_interleaved, 0);
    }
#else
    ggml_gemv_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_q4_K_8x8_q8_K(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
#if defined(__loongarch_asx)
    const int             qk                = QK_K;
    const int             nb                = n / qk;
    const int             ncols_interleaved = 8;
    static const uint32_t kmask1            = 0x3f3f3f3f;
    static const uint32_t kmask2            = 0x0f0f0f0f;
    static const uint32_t kmask3            = 0x03030303;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    __m256   sumvf[4];
    uint32_t utmp[32];

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + y * nb;
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + x * nb;
            for (int m = 0; m < 4; m++) {
                sumvf[m] = (__m256) __lasx_xvldi(0);
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2]      = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                __m256  b_d      = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].d, 0), 0x10));
                __m256i sumv_0_0 = __lasx_xvldi(0);
                __m256i sumv_0_1 = __lasx_xvldi(0);
                __m256i sumv_0_2 = __lasx_xvldi(0);
                __m256i sumv_0_3 = __lasx_xvldi(0);
                __m256i sumv_1_0 = __lasx_xvldi(0);
                __m256i sumv_1_1 = __lasx_xvldi(0);
                __m256i sumv_1_2 = __lasx_xvldi(0);
                __m256i sumv_1_3 = __lasx_xvldi(0);
                for (int sb = 0; sb < 4; sb++) {
                    const uint8_t * scales    = (uint8_t *) utmp + sb * 32;
                    __m256i         vscale    = __lasx_xvld(scales, 0);
                    __m256i         vscale_lo = __lasx_vext2xv_w_b(vscale);
                    __m256i         vscale_hi = __lasx_vext2xv_w_b(__lasx_xvpermi_q(vscale, vscale, 0x03));
                    __m256i         lo        = __lasx_xvilvl_w(vscale_lo, vscale_lo);
                    __m256i         hi        = __lasx_xvilvh_w(vscale_lo, vscale_lo);
                    const __m256i   vscales_0 = __lasx_xvpermi_q(lo, hi, 0x02);
                    const __m256i   vscales_1 = __lasx_xvpermi_q(lo, hi, 0x13);

                    lo = __lasx_xvilvl_w(vscale_hi, vscale_hi);
                    hi = __lasx_xvilvh_w(vscale_hi, vscale_hi);

                    const __m256i vscales_2 = __lasx_xvpermi_q(lo, hi, 0x02);
                    const __m256i vscales_3 = __lasx_xvpermi_q(lo, hi, 0x13);
                    {
                        __m256i sum_lo_0 = __lasx_xvldi(0);
                        __m256i sum_lo_1 = __lasx_xvldi(0);
                        __m256i sum_lo_2 = __lasx_xvldi(0);
                        __m256i sum_lo_3 = __lasx_xvldi(0);
                        __m256i sum_hi_0 = __lasx_xvldi(0);
                        __m256i sum_hi_1 = __lasx_xvldi(0);
                        __m256i sum_hi_2 = __lasx_xvldi(0);
                        __m256i sum_hi_3 = __lasx_xvldi(0);
#    pragma GCC unroll 2
                        for (int ik = 0; ik < 4; ik++) {
                            const int8_t * a_k   = a_ptr[l].qs + sb * 256 + ik * 32;
                            const __m256i  a_lo  = __lasx_xvld(a_k, 0);
                            const __m256i  a_hi  = __lasx_xvld(a_k, 128);
                            const __m256i  vx    = __lasx_xvld(b_ptr[l].qs + sb * 256 + ik * 64, 0);
                            const __m256i  vx_lo = __lasx_xvandi_b(vx, 0x0f);
                            const __m256i  vx_hi = __lasx_xvsrli_b(vx, 4);
                            {
                                const __m256i vy_lo = __lasx_xvreplve0_d(a_lo);
                                const __m256i vy_hi = __lasx_xvreplve0_d(a_hi);
                                sum_lo_0            = __lasx_xvmaddwev_h_b(sum_lo_0, vx_lo, vy_lo);
                                sum_lo_0            = __lasx_xvmaddwod_h_b(sum_lo_0, vx_lo, vy_lo);
                                sum_hi_0            = __lasx_xvmaddwev_h_b(sum_hi_0, vx_hi, vy_hi);
                                sum_hi_0            = __lasx_xvmaddwod_h_b(sum_hi_0, vx_hi, vy_hi);
                            }
                            {
                                const __m256i vy_lo = __lasx_xvpermi_d(a_lo, 0x55);
                                const __m256i vy_hi = __lasx_xvpermi_d(a_hi, 0x55);
                                sum_lo_1            = __lasx_xvmaddwev_h_b(sum_lo_1, vx_lo, vy_lo);
                                sum_lo_1            = __lasx_xvmaddwod_h_b(sum_lo_1, vx_lo, vy_lo);
                                sum_hi_1            = __lasx_xvmaddwev_h_b(sum_hi_1, vx_hi, vy_hi);
                                sum_hi_1            = __lasx_xvmaddwod_h_b(sum_hi_1, vx_hi, vy_hi);
                            }
                            {
                                const __m256i vy_lo = __lasx_xvpermi_d(a_lo, 0xaa);
                                const __m256i vy_hi = __lasx_xvpermi_d(a_hi, 0xaa);
                                sum_lo_2            = __lasx_xvmaddwev_h_b(sum_lo_2, vx_lo, vy_lo);
                                sum_lo_2            = __lasx_xvmaddwod_h_b(sum_lo_2, vx_lo, vy_lo);
                                sum_hi_2            = __lasx_xvmaddwev_h_b(sum_hi_2, vx_hi, vy_hi);
                                sum_hi_2            = __lasx_xvmaddwod_h_b(sum_hi_2, vx_hi, vy_hi);
                            }
                            {
                                const __m256i vy_lo = __lasx_xvpermi_d(a_lo, 0xff);
                                const __m256i vy_hi = __lasx_xvpermi_d(a_hi, 0xff);
                                sum_lo_3            = __lasx_xvmaddwev_h_b(sum_lo_3, vx_lo, vy_lo);
                                sum_lo_3            = __lasx_xvmaddwod_h_b(sum_lo_3, vx_lo, vy_lo);
                                sum_hi_3            = __lasx_xvmaddwev_h_b(sum_hi_3, vx_hi, vy_hi);
                                sum_hi_3            = __lasx_xvmaddwod_h_b(sum_hi_3, vx_hi, vy_hi);
                            }
                        }
                        sum_lo_0 = __lasx_xvhaddw_w_h(sum_lo_0, sum_lo_0);
                        sum_hi_0 = __lasx_xvhaddw_w_h(sum_hi_0, sum_hi_0);
                        sumv_0_0 = __lasx_xvmadd_w(sumv_0_0, sum_lo_0, vscales_0);
                        sumv_0_0 = __lasx_xvmadd_w(sumv_0_0, sum_hi_0, vscales_2);
                        sum_lo_1 = __lasx_xvhaddw_w_h(sum_lo_1, sum_lo_1);
                        sum_hi_1 = __lasx_xvhaddw_w_h(sum_hi_1, sum_hi_1);
                        sumv_0_1 = __lasx_xvmadd_w(sumv_0_1, sum_lo_1, vscales_0);
                        sumv_0_1 = __lasx_xvmadd_w(sumv_0_1, sum_hi_1, vscales_2);
                        sum_lo_2 = __lasx_xvhaddw_w_h(sum_lo_2, sum_lo_2);
                        sum_hi_2 = __lasx_xvhaddw_w_h(sum_hi_2, sum_hi_2);
                        sumv_0_2 = __lasx_xvmadd_w(sumv_0_2, sum_lo_2, vscales_0);
                        sumv_0_2 = __lasx_xvmadd_w(sumv_0_2, sum_hi_2, vscales_2);
                        sum_lo_3 = __lasx_xvhaddw_w_h(sum_lo_3, sum_lo_3);
                        sum_hi_3 = __lasx_xvhaddw_w_h(sum_hi_3, sum_hi_3);
                        sumv_0_3 = __lasx_xvmadd_w(sumv_0_3, sum_lo_3, vscales_0);
                        sumv_0_3 = __lasx_xvmadd_w(sumv_0_3, sum_hi_3, vscales_2);
                    }
                    {
                        __m256i sum_lo_0 = __lasx_xvldi(0);
                        __m256i sum_lo_1 = __lasx_xvldi(0);
                        __m256i sum_lo_2 = __lasx_xvldi(0);
                        __m256i sum_lo_3 = __lasx_xvldi(0);
                        __m256i sum_hi_0 = __lasx_xvldi(0);
                        __m256i sum_hi_1 = __lasx_xvldi(0);
                        __m256i sum_hi_2 = __lasx_xvldi(0);
                        __m256i sum_hi_3 = __lasx_xvldi(0);
#    pragma GCC unroll 2
                        for (int ik = 0; ik < 4; ik++) {
                            const int8_t * a_k   = a_ptr[l].qs + sb * 256 + ik * 32;
                            const __m256i  a_lo  = __lasx_xvld(a_k, 0);
                            const __m256i  a_hi  = __lasx_xvld(a_k, 128);
                            const __m256i  vx    = __lasx_xvld(b_ptr[l].qs + sb * 256 + ik * 64, 32);
                            const __m256i  vx_lo = __lasx_xvandi_b(vx, 0x0f);
                            const __m256i  vx_hi = __lasx_xvsrli_b(vx, 4);
                            {
                                const __m256i vy_lo = __lasx_xvreplve0_d(a_lo);
                                const __m256i vy_hi = __lasx_xvreplve0_d(a_hi);
                                sum_lo_0            = __lasx_xvmaddwev_h_b(sum_lo_0, vx_lo, vy_lo);
                                sum_lo_0            = __lasx_xvmaddwod_h_b(sum_lo_0, vx_lo, vy_lo);
                                sum_hi_0            = __lasx_xvmaddwev_h_b(sum_hi_0, vx_hi, vy_hi);
                                sum_hi_0            = __lasx_xvmaddwod_h_b(sum_hi_0, vx_hi, vy_hi);
                            }
                            {
                                const __m256i vy_lo = __lasx_xvpermi_d(a_lo, 0x55);
                                const __m256i vy_hi = __lasx_xvpermi_d(a_hi, 0x55);
                                sum_lo_1            = __lasx_xvmaddwev_h_b(sum_lo_1, vx_lo, vy_lo);
                                sum_lo_1            = __lasx_xvmaddwod_h_b(sum_lo_1, vx_lo, vy_lo);
                                sum_hi_1            = __lasx_xvmaddwev_h_b(sum_hi_1, vx_hi, vy_hi);
                                sum_hi_1            = __lasx_xvmaddwod_h_b(sum_hi_1, vx_hi, vy_hi);
                            }
                            {
                                const __m256i vy_lo = __lasx_xvpermi_d(a_lo, 0xaa);
                                const __m256i vy_hi = __lasx_xvpermi_d(a_hi, 0xaa);
                                sum_lo_2            = __lasx_xvmaddwev_h_b(sum_lo_2, vx_lo, vy_lo);
                                sum_lo_2            = __lasx_xvmaddwod_h_b(sum_lo_2, vx_lo, vy_lo);
                                sum_hi_2            = __lasx_xvmaddwev_h_b(sum_hi_2, vx_hi, vy_hi);
                                sum_hi_2            = __lasx_xvmaddwod_h_b(sum_hi_2, vx_hi, vy_hi);
                            }
                            {
                                const __m256i vy_lo = __lasx_xvpermi_d(a_lo, 0xff);
                                const __m256i vy_hi = __lasx_xvpermi_d(a_hi, 0xff);
                                sum_lo_3            = __lasx_xvmaddwev_h_b(sum_lo_3, vx_lo, vy_lo);
                                sum_lo_3            = __lasx_xvmaddwod_h_b(sum_lo_3, vx_lo, vy_lo);
                                sum_hi_3            = __lasx_xvmaddwev_h_b(sum_hi_3, vx_hi, vy_hi);
                                sum_hi_3            = __lasx_xvmaddwod_h_b(sum_hi_3, vx_hi, vy_hi);
                            }
                        }
                        sum_lo_0 = __lasx_xvhaddw_w_h(sum_lo_0, sum_lo_0);
                        sum_hi_0 = __lasx_xvhaddw_w_h(sum_hi_0, sum_hi_0);
                        sumv_1_0 = __lasx_xvmadd_w(sumv_1_0, sum_lo_0, vscales_1);
                        sumv_1_0 = __lasx_xvmadd_w(sumv_1_0, sum_hi_0, vscales_3);
                        sum_lo_1 = __lasx_xvhaddw_w_h(sum_lo_1, sum_lo_1);
                        sum_hi_1 = __lasx_xvhaddw_w_h(sum_hi_1, sum_hi_1);
                        sumv_1_1 = __lasx_xvmadd_w(sumv_1_1, sum_lo_1, vscales_1);
                        sumv_1_1 = __lasx_xvmadd_w(sumv_1_1, sum_hi_1, vscales_3);
                        sum_lo_2 = __lasx_xvhaddw_w_h(sum_lo_2, sum_lo_2);
                        sum_hi_2 = __lasx_xvhaddw_w_h(sum_hi_2, sum_hi_2);
                        sumv_1_2 = __lasx_xvmadd_w(sumv_1_2, sum_lo_2, vscales_1);
                        sumv_1_2 = __lasx_xvmadd_w(sumv_1_2, sum_hi_2, vscales_3);
                        sum_lo_3 = __lasx_xvhaddw_w_h(sum_lo_3, sum_lo_3);
                        sum_hi_3 = __lasx_xvhaddw_w_h(sum_hi_3, sum_hi_3);
                        sumv_1_3 = __lasx_xvmadd_w(sumv_1_3, sum_lo_3, vscales_1);
                        sumv_1_3 = __lasx_xvmadd_w(sumv_1_3, sum_hi_3, vscales_3);
                    }
                }
                {
                    __m256  a_d  = (__m256) __lasx_xvldrepl_w(a_ptr[l].d, 0);
                    __m256i even = __lasx_xvpickev_w(sumv_1_0, sumv_0_0);
                    __m256i odd  = __lasx_xvpickod_w(sumv_1_0, sumv_0_0);

                    sumv_0_0 = __lasx_xvpermi_d(__lasx_xvadd_w(even, odd), 0xd8);

                    const __m256 ab_d = __lasx_xvfmul_s(a_d, b_d);
                    sumvf[0]          = __lasx_xvfmadd_s(__lasx_xvffint_s_w(sumv_0_0), ab_d, sumvf[0]);
                }
                {
                    __m256  a_d  = (__m256) __lasx_xvldrepl_w(a_ptr[l].d + 1, 0);
                    __m256i even = __lasx_xvpickev_w(sumv_1_1, sumv_0_1);
                    __m256i odd  = __lasx_xvpickod_w(sumv_1_1, sumv_0_1);

                    sumv_0_1 = __lasx_xvpermi_d(__lasx_xvadd_w(even, odd), 0xd8);

                    const __m256 ab_d = __lasx_xvfmul_s(a_d, b_d);
                    sumvf[1]          = __lasx_xvfmadd_s(__lasx_xvffint_s_w(sumv_0_1), ab_d, sumvf[1]);
                }
                {
                    __m256  a_d  = (__m256) __lasx_xvldrepl_w(a_ptr[l].d + 2, 0);
                    __m256i even = __lasx_xvpickev_w(sumv_1_2, sumv_0_2);
                    __m256i odd  = __lasx_xvpickod_w(sumv_1_2, sumv_0_2);

                    sumv_0_2 = __lasx_xvpermi_d(__lasx_xvadd_w(even, odd), 0xd8);

                    const __m256 ab_d = __lasx_xvfmul_s(a_d, b_d);
                    sumvf[2]          = __lasx_xvfmadd_s(__lasx_xvffint_s_w(sumv_0_2), ab_d, sumvf[2]);
                }
                {
                    __m256  a_d  = (__m256) __lasx_xvldrepl_w(a_ptr[l].d + 3, 0);
                    __m256i even = __lasx_xvpickev_w(sumv_1_3, sumv_0_3);
                    __m256i odd  = __lasx_xvpickod_w(sumv_1_3, sumv_0_3);

                    sumv_0_3 = __lasx_xvpermi_d(__lasx_xvadd_w(even, odd), 0xd8);

                    const __m256 ab_d = __lasx_xvfmul_s(a_d, b_d);
                    sumvf[3]          = __lasx_xvfmadd_s(__lasx_xvffint_s_w(sumv_0_3), ab_d, sumvf[3]);
                }
                __m256i bias[4];
                for (int i = 0; i < 4; i++) {
                    bias[i] = __lasx_xvldi(0);
                }
                {
                    {
                        __m256i vbsums = __lasx_xvld(a_ptr[l].bsums, 0);
                        vbsums         = __lasx_xvhaddw_w_h(vbsums, vbsums);

                        const __m256i bsum_0 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 0));
                        const __m256i bsum_1 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 1));
                        const __m256i bsum_2 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 2));
                        const __m256i bsum_3 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 3));
                        const __m256i bsum_4 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 4));
                        const __m256i bsum_5 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 5));
                        const __m256i bsum_6 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 6));
                        const __m256i bsum_7 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 7));

                        __m256i mins_0_1 = __lasx_xvld(utmp, 8);
                        __m256i mins_0   = __lasx_vext2xv_w_b(mins_0_1);
                        __m256i mins_1   = __lasx_vext2xv_w_b(__lasx_xvpermi_q(mins_0_1, mins_0_1, 0x03));

                        bias[0] = __lasx_xvmadd_w(bias[0], mins_0, bsum_0);
                        bias[1] = __lasx_xvmadd_w(bias[1], mins_0, bsum_2);
                        bias[2] = __lasx_xvmadd_w(bias[2], mins_0, bsum_4);
                        bias[3] = __lasx_xvmadd_w(bias[3], mins_0, bsum_6);
                        bias[0] = __lasx_xvmadd_w(bias[0], mins_1, bsum_1);
                        bias[1] = __lasx_xvmadd_w(bias[1], mins_1, bsum_3);
                        bias[2] = __lasx_xvmadd_w(bias[2], mins_1, bsum_5);
                        bias[3] = __lasx_xvmadd_w(bias[3], mins_1, bsum_7);
                    }
                    {
                        __m256i vbsums = __lasx_xvld(a_ptr[l].bsums, 32);
                        vbsums         = __lasx_xvhaddw_w_h(vbsums, vbsums);

                        const __m256i bsum_0 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 0));
                        const __m256i bsum_1 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 1));
                        const __m256i bsum_2 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 2));
                        const __m256i bsum_3 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 3));
                        const __m256i bsum_4 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 4));
                        const __m256i bsum_5 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 5));
                        const __m256i bsum_6 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 6));
                        const __m256i bsum_7 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 7));

                        __m256i mins_0_1 = __lasx_xvld(utmp, 40);
                        __m256i mins_0   = __lasx_vext2xv_w_b(mins_0_1);
                        __m256i mins_1   = __lasx_vext2xv_w_b(__lasx_xvpermi_q(mins_0_1, mins_0_1, 0x03));

                        bias[0] = __lasx_xvmadd_w(bias[0], mins_0, bsum_0);
                        bias[1] = __lasx_xvmadd_w(bias[1], mins_0, bsum_2);
                        bias[2] = __lasx_xvmadd_w(bias[2], mins_0, bsum_4);
                        bias[3] = __lasx_xvmadd_w(bias[3], mins_0, bsum_6);
                        bias[0] = __lasx_xvmadd_w(bias[0], mins_1, bsum_1);
                        bias[1] = __lasx_xvmadd_w(bias[1], mins_1, bsum_3);
                        bias[2] = __lasx_xvmadd_w(bias[2], mins_1, bsum_5);
                        bias[3] = __lasx_xvmadd_w(bias[3], mins_1, bsum_7);
                    }
                    {
                        __m256i vbsums = __lasx_xvld(a_ptr[l].bsums, 64);
                        vbsums         = __lasx_xvhaddw_w_h(vbsums, vbsums);

                        const __m256i bsum_0 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 0));
                        const __m256i bsum_1 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 1));
                        const __m256i bsum_2 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 2));
                        const __m256i bsum_3 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 3));
                        const __m256i bsum_4 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 4));
                        const __m256i bsum_5 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 5));
                        const __m256i bsum_6 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 6));
                        const __m256i bsum_7 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 7));

                        __m256i mins_0_1 = __lasx_xvld(utmp, 72);
                        __m256i mins_0   = __lasx_vext2xv_w_b(mins_0_1);
                        __m256i mins_1   = __lasx_vext2xv_w_b(__lasx_xvpermi_q(mins_0_1, mins_0_1, 0x03));

                        bias[0] = __lasx_xvmadd_w(bias[0], mins_0, bsum_0);
                        bias[1] = __lasx_xvmadd_w(bias[1], mins_0, bsum_2);
                        bias[2] = __lasx_xvmadd_w(bias[2], mins_0, bsum_4);
                        bias[3] = __lasx_xvmadd_w(bias[3], mins_0, bsum_6);
                        bias[0] = __lasx_xvmadd_w(bias[0], mins_1, bsum_1);
                        bias[1] = __lasx_xvmadd_w(bias[1], mins_1, bsum_3);
                        bias[2] = __lasx_xvmadd_w(bias[2], mins_1, bsum_5);
                        bias[3] = __lasx_xvmadd_w(bias[3], mins_1, bsum_7);
                    }
                    {
                        __m256i vbsums = __lasx_xvld(a_ptr[l].bsums, 96);
                        vbsums         = __lasx_xvhaddw_w_h(vbsums, vbsums);

                        const __m256i bsum_0 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 0));
                        const __m256i bsum_1 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 1));
                        const __m256i bsum_2 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 2));
                        const __m256i bsum_3 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 3));
                        const __m256i bsum_4 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 4));
                        const __m256i bsum_5 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 5));
                        const __m256i bsum_6 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 6));
                        const __m256i bsum_7 = __lasx_xvreplgr2vr_w(__lasx_xvpickve2gr_w(vbsums, 7));

                        __m256i mins_0_1 = __lasx_xvpermi_d(__lasx_xvld(utmp, 96), 0xf5);
                        __m256i mins_0   = __lasx_vext2xv_w_b(mins_0_1);
                        __m256i mins_1   = __lasx_vext2xv_w_b(__lasx_xvpermi_q(mins_0_1, mins_0_1, 0x03));

                        bias[0] = __lasx_xvmadd_w(bias[0], mins_0, bsum_0);
                        bias[1] = __lasx_xvmadd_w(bias[1], mins_0, bsum_2);
                        bias[2] = __lasx_xvmadd_w(bias[2], mins_0, bsum_4);
                        bias[3] = __lasx_xvmadd_w(bias[3], mins_0, bsum_6);
                        bias[0] = __lasx_xvmadd_w(bias[0], mins_1, bsum_1);
                        bias[1] = __lasx_xvmadd_w(bias[1], mins_1, bsum_3);
                        bias[2] = __lasx_xvmadd_w(bias[2], mins_1, bsum_5);
                        bias[3] = __lasx_xvmadd_w(bias[3], mins_1, bsum_7);
                    }
                    __m256 b_dmin = (__m256) __lasx_xvld(b_ptr[l].dmin, 0);
                    b_dmin        = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(b_dmin, 0x10));
                    __m256 a_d    = (__m256) __lasx_xvldrepl_w(a_ptr[l].d + 0, 0);
                    sumvf[0] = __lasx_xvfnmsub_s(__lasx_xvfmul_s(b_dmin, __lasx_xvffint_s_w(bias[0])), a_d, sumvf[0]);
                    a_d      = (__m256) __lasx_xvldrepl_w(a_ptr[l].d + 1, 0);
                    sumvf[1] = __lasx_xvfnmsub_s(__lasx_xvfmul_s(b_dmin, __lasx_xvffint_s_w(bias[1])), a_d, sumvf[1]);
                    a_d      = (__m256) __lasx_xvldrepl_w(a_ptr[l].d + 2, 0);
                    sumvf[2] = __lasx_xvfnmsub_s(__lasx_xvfmul_s(b_dmin, __lasx_xvffint_s_w(bias[2])), a_d, sumvf[2]);
                    a_d      = (__m256) __lasx_xvldrepl_w(a_ptr[l].d + 3, 0);
                    sumvf[3] = __lasx_xvfnmsub_s(__lasx_xvfmul_s(b_dmin, __lasx_xvffint_s_w(bias[3])), a_d, sumvf[3]);
                }
            }
            for (int m = 0; m < 4; m++) {
                __lasx_xvst((__m256i) sumvf[m], &s[(y * 4 + m) * bs + x * ncols_interleaved], 0);
            }
        }
    }
#else
    ggml_gemm_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
#endif
}
