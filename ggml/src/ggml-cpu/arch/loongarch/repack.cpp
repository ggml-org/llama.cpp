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
            // Load elements into 4 LASX vectors
            __m256 v0 = (__m256) __lasx_xvld(x + row_iter * k + i * 32, 0);
            __m256 v1 = (__m256) __lasx_xvld(x + row_iter * k + i * 32, 32);
            __m256 v2 = (__m256) __lasx_xvld(x + row_iter * k + i * 32, 64);
            __m256 v3 = (__m256) __lasx_xvld(x + row_iter * k + i * 32, 96);

            // Compute max(abs(e)) for the block
            const __m256 signBit = (__m256) __lasx_xvreplgr2vr_w(0x80000000);
            __m256       maxAbs  = (__m256) __lasx_xvandn_v((__m256i) signBit, (__m256i) v0);
            maxAbs               = __lasx_xvfmax_s(maxAbs, (__m256) __lasx_xvandn_v((__m256i) signBit, (__m256i) v1));
            maxAbs               = __lasx_xvfmax_s(maxAbs, (__m256) __lasx_xvandn_v((__m256i) signBit, (__m256i) v2));
            maxAbs               = __lasx_xvfmax_s(maxAbs, (__m256) __lasx_xvandn_v((__m256i) signBit, (__m256i) v3));

            __m128 max4 = __lsx_vfmax_s(lasx_extract_128_hi_s(maxAbs), lasx_extract_128_lo_s(maxAbs));
            max4        = __lsx_vfmax_s(max4, (__m128) __lsx_vshuf4i_w((__m128i) max4, 0x4E));
            max4        = __lsx_vfmax_s(max4, (__m128) __lsx_vshuf4i_w((__m128i) max4, 0xB1));
            float maxScalar;
            __asm__("" : "=f"(maxScalar) : "0"(max4));

            // Divided by 127.f to mirror results in quantize_row_q8_0
            const float d = maxScalar / 127.f;
            id[row_iter]  = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;  //d ? 1.0f / d : 0.0f;

            // Store the scale for the individual block
            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);

            // Store the values in blocks of eight values - Aim is to use these later for block interleaving

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

            const int     permMask[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };
            const __m256i perm        = __lasx_xvld(permMask, 0);
            i0                        = __lasx_xvperm_w(i0, perm);

            __lasx_xvst(i0, (y[i].qs + 32 * j), 0);
        }
    }

#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x8_generic(x, vy, k);
#endif
}

void ggml_gemv_q4_0_8x8_q8_0(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
#if defined __loongarch_asx
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen          = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    __m256  sumv;
    __m256i sumi[4];

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {  // block row
        const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);
        sumv                       = (__m256) __lasx_xvldi(0);
        for (int l = 0; l < nb; l++) {
            __m256i vy       = __lasx_xvld(a_ptr[l].qs, 0);
            __m256i vy_0_7   = __lasx_vext2xv_h_b(__lasx_xvreplve0_d(vy));
            __m256i vy_8_15  = __lasx_vext2xv_h_b(__lasx_xvreplve_d(vy, 1));
            __m256i vy_16_23 = __lasx_vext2xv_h_b(__lasx_xvpermi_d(vy, 0xaa));
            __m256i vy_24_31 = __lasx_vext2xv_h_b(__lasx_xvpermi_d(vy, 0xff));

            __m256i vx       = __lasx_xvld(b_ptr[l].qs, 0);
            __m256i vxlo     = __lasx_xvsrai_b(__lasx_xvslli_b(vx, 4), 4);
            __m256i vxhi     = __lasx_xvsrai_b(vx, 4);
            __m256i vxlo_0_1 = __lasx_vext2xv_h_b(vxlo);
            __m256i vxlo_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vxlo, vxlo, 0x03));
            __m256i vxhi_0_1 = __lasx_vext2xv_h_b(vxhi);
            __m256i vxhi_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vxhi, vxhi, 0x03));
            sumi[0]          = __lasx_xvmul_h(vxlo_0_1, vy_0_7);
            sumi[0]          = __lasx_xvmadd_h(sumi[0], vxhi_0_1, vy_16_23);
            sumi[1]          = __lasx_xvmul_h(vxlo_2_3, vy_0_7);
            sumi[1]          = __lasx_xvmadd_h(sumi[1], vxhi_2_3, vy_16_23);

            vx       = __lasx_xvld(b_ptr[l].qs + 32, 0);
            vxlo     = __lasx_xvsrai_b(__lasx_xvslli_b(vx, 4), 4);
            vxhi     = __lasx_xvsrai_b(vx, 4);
            vxlo_0_1 = __lasx_vext2xv_h_b(vxlo);
            vxlo_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vxlo, vxlo, 0x03));
            vxhi_0_1 = __lasx_vext2xv_h_b(vxhi);
            vxhi_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vxhi, vxhi, 0x03));
            sumi[2]  = __lasx_xvmul_h(vxlo_0_1, vy_0_7);
            sumi[2]  = __lasx_xvmadd_h(sumi[2], vxhi_0_1, vy_16_23);
            sumi[3]  = __lasx_xvmul_h(vxlo_2_3, vy_0_7);
            sumi[3]  = __lasx_xvmadd_h(sumi[3], vxhi_2_3, vy_16_23);

            vx       = __lasx_xvld(b_ptr[l].qs + 64, 0);
            vxlo     = __lasx_xvsrai_b(__lasx_xvslli_b(vx, 4), 4);
            vxhi     = __lasx_xvsrai_b(vx, 4);
            vxlo_0_1 = __lasx_vext2xv_h_b(vxlo);
            vxlo_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vxlo, vxlo, 0x03));
            vxhi_0_1 = __lasx_vext2xv_h_b(vxhi);
            vxhi_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vxhi, vxhi, 0x03));
            sumi[0]  = __lasx_xvmadd_h(sumi[0], vxlo_0_1, vy_8_15);
            sumi[0]  = __lasx_xvmadd_h(sumi[0], vxhi_0_1, vy_24_31);
            sumi[1]  = __lasx_xvmadd_h(sumi[1], vxlo_2_3, vy_8_15);
            sumi[1]  = __lasx_xvmadd_h(sumi[1], vxhi_2_3, vy_24_31);

            vx       = __lasx_xvld(b_ptr[l].qs + 96, 0);
            vxlo     = __lasx_xvsrai_b(__lasx_xvslli_b(vx, 4), 4);
            vxhi     = __lasx_xvsrai_b(vx, 4);
            vxlo_0_1 = __lasx_vext2xv_h_b(vxlo);
            vxlo_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vxlo, vxlo, 0x03));
            vxhi_0_1 = __lasx_vext2xv_h_b(vxhi);
            vxhi_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vxhi, vxhi, 0x03));
            sumi[2]  = __lasx_xvmadd_h(sumi[2], vxlo_0_1, vy_8_15);
            sumi[2]  = __lasx_xvmadd_h(sumi[2], vxhi_0_1, vy_24_31);
            sumi[3]  = __lasx_xvmadd_h(sumi[3], vxlo_2_3, vy_8_15);
            sumi[3]  = __lasx_xvmadd_h(sumi[3], vxhi_2_3, vy_24_31);

            sumi[0] = __lasx_xvhaddw_w_h(sumi[0], sumi[0]);
            sumi[0] = __lasx_xvadd_w(sumi[0], __lasx_xvshuf4i_w(sumi[0], 0x4E));
            sumi[0] = __lasx_xvadd_w(sumi[0], __lasx_xvshuf4i_w(sumi[0], 0xB1));
            sumi[1] = __lasx_xvhaddw_w_h(sumi[1], sumi[1]);
            sumi[1] = __lasx_xvadd_w(sumi[1], __lasx_xvshuf4i_w(sumi[1], 0x4E));
            sumi[1] = __lasx_xvadd_w(sumi[1], __lasx_xvshuf4i_w(sumi[1], 0xB1));
            sumi[2] = __lasx_xvhaddw_w_h(sumi[2], sumi[2]);
            sumi[2] = __lasx_xvadd_w(sumi[2], __lasx_xvshuf4i_w(sumi[2], 0x4E));
            sumi[2] = __lasx_xvadd_w(sumi[2], __lasx_xvshuf4i_w(sumi[2], 0xB1));
            sumi[3] = __lasx_xvhaddw_w_h(sumi[3], sumi[3]);
            sumi[3] = __lasx_xvadd_w(sumi[3], __lasx_xvshuf4i_w(sumi[3], 0x4E));
            sumi[3] = __lasx_xvadd_w(sumi[3], __lasx_xvshuf4i_w(sumi[3], 0xB1));

            sumi[0]         = __lasx_xvpickev_d(sumi[1], sumi[0]);
            sumi[2]         = __lasx_xvpickev_d(sumi[3], sumi[2]);
            __m256 scalar_x = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld((b_ptr[l].d), 0), 0x50));
            __m256 tf       = __lasx_xvffint_s_w(__lasx_xvpickev_w(sumi[2], sumi[0]));

            const __m256  scalar = __lasx_xvfmul_s(scalar_x, __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(&(a_ptr[l].d), 0)));
            const int32_t sum_perm_data[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };
            const __m256i sum_perm         = __lasx_xvld(sum_perm_data, 0);

            tf   = (__m256) __lasx_xvperm_w((__m256i) tf, sum_perm);
            sumv = __lasx_xvfmadd_s(tf, scalar, sumv);
        }
        __lasx_xvst(sumv, s + x * ncols_interleaved, 0);
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
#if defined __loongarch_asx
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen          = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    __m256  sumf[4];
    __m256i t[4];

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            for (int i = 0; i < 4; i++) {
                sumf[i] = (__m256) __lasx_xvldi(0);
            }
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);
            for (int l = 0; l < nb; l++) {
                const __m256i vx00       = __lasx_xvld(b_ptr[l].qs, 0);
                const __m256i vx00_lo    = __lasx_xvsrai_b(__lasx_xvslli_b(vx00, 4), 4);
                const __m256i vx00_hi    = __lasx_xvsrai_b(vx00, 4);
                const __m256i vx0_lo_0_1 = __lasx_vext2xv_h_b(vx00_lo);
                const __m256i vx0_lo_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vx00_lo, vx00_lo, 0x03));
                const __m256i vx0_hi_0_1 = __lasx_vext2xv_h_b(vx00_hi);
                const __m256i vx0_hi_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vx00_hi, vx00_hi, 0x03));

                const __m256i vx01       = __lasx_xvld(b_ptr[l].qs, 32);
                const __m256i vx01_lo    = __lasx_xvsrai_b(__lasx_xvslli_b(vx01, 4), 4);
                const __m256i vx01_hi    = __lasx_xvsrai_b(vx01, 4);
                const __m256i vx0_lo_4_5 = __lasx_vext2xv_h_b(vx01_lo);
                const __m256i vx0_lo_6_7 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vx01_lo, vx01_lo, 0x03));
                const __m256i vx0_hi_4_5 = __lasx_vext2xv_h_b(vx01_hi);
                const __m256i vx0_hi_6_7 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vx01_hi, vx01_hi, 0x03));

                const __m256i vx10       = __lasx_xvld(b_ptr[l].qs, 64);
                const __m256i vx10_lo    = __lasx_xvsrai_b(__lasx_xvslli_b(vx10, 4), 4);
                const __m256i vx10_hi    = __lasx_xvsrai_b(vx10, 4);
                const __m256i vx1_lo_0_1 = __lasx_vext2xv_h_b(vx10_lo);
                const __m256i vx1_lo_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vx10_lo, vx10_lo, 0x03));
                const __m256i vx1_hi_0_1 = __lasx_vext2xv_h_b(vx10_hi);
                const __m256i vx1_hi_2_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vx10_hi, vx10_hi, 0x03));

                const __m256i vx11       = __lasx_xvld(b_ptr[l].qs, 96);
                const __m256i vx11_lo    = __lasx_xvsrai_b(__lasx_xvslli_b(vx11, 4), 4);
                const __m256i vx11_hi    = __lasx_xvsrai_b(vx11, 4);
                const __m256i vx1_lo_4_5 = __lasx_vext2xv_h_b(vx11_lo);
                const __m256i vx1_lo_6_7 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vx11_lo, vx11_lo, 0x03));
                const __m256i vx1_hi_4_5 = __lasx_vext2xv_h_b(vx11_hi);
                const __m256i vx1_hi_6_7 = __lasx_vext2xv_h_b(__lasx_xvpermi_q(vx11_hi, vx11_hi, 0x03));

                const __m256i vy0_lo = __lasx_xvld(a_ptr[l].qs, 0);
                const __m256i vy0_hi = __lasx_xvld(a_ptr[l].qs, 64);
                const __m256i vy1_lo = __lasx_xvld(a_ptr[l].qs, 32);
                const __m256i vy1_hi = __lasx_xvld(a_ptr[l].qs, 96);

                const int32_t scale_perm[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };
                __m256        scalar_x      = __lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lasx_xvld(b_ptr[l].d, 0), 0x50));
                scalar_x                    = (__m256) __lasx_xvperm_w((__m256i) scalar_x, __lasx_xvld(scale_perm, 0));

                {
                    const __m256i vy0_lo_0 = __lasx_vext2xv_h_b(__lasx_xvreplve0_d(vy0_lo));
                    const __m256i vy1_lo_0 = __lasx_vext2xv_h_b(__lasx_xvreplve0_d(vy1_lo));
                    const __m256i vy0_hi_0 = __lasx_vext2xv_h_b(__lasx_xvreplve0_d(vy0_hi));
                    const __m256i vy1_hi_0 = __lasx_vext2xv_h_b(__lasx_xvreplve0_d(vy1_hi));

                    t[0] = __lasx_xvmul_h(vy0_lo_0, vx0_lo_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy0_hi_0, vx0_hi_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy1_lo_0, vx1_lo_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy1_hi_0, vx1_hi_0_1);
                    t[1] = __lasx_xvmul_h(vy0_lo_0, vx0_lo_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy1_lo_0, vx1_lo_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy0_hi_0, vx0_hi_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy1_hi_0, vx1_hi_2_3);
                    t[2] = __lasx_xvmul_h(vy0_lo_0, vx0_lo_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy0_hi_0, vx0_hi_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy1_lo_0, vx1_lo_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy1_hi_0, vx1_hi_4_5);
                    t[3] = __lasx_xvmul_h(vy0_lo_0, vx0_lo_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy0_hi_0, vx0_hi_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy1_lo_0, vx1_lo_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy1_hi_0, vx1_hi_6_7);

                    t[0] = __lasx_xvhaddw_w_h(t[0], t[0]);
                    t[0] = __lasx_xvadd_w(t[0], __lasx_xvshuf4i_w(t[0], 0x4E));
                    t[0] = __lasx_xvadd_w(t[0], __lasx_xvshuf4i_w(t[0], 0xB1));
                    t[1] = __lasx_xvhaddw_w_h(t[1], t[1]);
                    t[1] = __lasx_xvadd_w(t[1], __lasx_xvshuf4i_w(t[1], 0x4E));
                    t[1] = __lasx_xvadd_w(t[1], __lasx_xvshuf4i_w(t[1], 0xB1));
                    t[2] = __lasx_xvhaddw_w_h(t[2], t[2]);
                    t[2] = __lasx_xvadd_w(t[2], __lasx_xvshuf4i_w(t[2], 0x4E));
                    t[2] = __lasx_xvadd_w(t[2], __lasx_xvshuf4i_w(t[2], 0xB1));
                    t[3] = __lasx_xvhaddw_w_h(t[3], t[3]);
                    t[3] = __lasx_xvadd_w(t[3], __lasx_xvshuf4i_w(t[3], 0x4E));
                    t[3] = __lasx_xvadd_w(t[3], __lasx_xvshuf4i_w(t[3], 0xB1));

                    t[0] = __lasx_xvpickev_d(t[1], t[0]);
                    t[2] = __lasx_xvpickev_d(t[3], t[2]);

                    const __m256 tf = __lasx_xvffint_s_w(__lasx_xvpickev_w(t[2], t[0]));
                    const __m256 scalar =
                        __lasx_xvfmul_s(scalar_x, __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d, 0)));
                    sumf[0] = __lasx_xvfmadd_s(tf, scalar, sumf[0]);
                }
                {
                    const __m256i vy0_lo_1 = __lasx_vext2xv_h_b(__lasx_xvreplve_d(vy0_lo, 1));
                    const __m256i vy1_lo_1 = __lasx_vext2xv_h_b(__lasx_xvreplve_d(vy1_lo, 1));
                    const __m256i vy0_hi_1 = __lasx_vext2xv_h_b(__lasx_xvreplve_d(vy0_hi, 1));
                    const __m256i vy1_hi_1 = __lasx_vext2xv_h_b(__lasx_xvreplve_d(vy1_hi, 1));

                    t[0] = __lasx_xvmul_h(vy0_lo_1, vx0_lo_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy0_hi_1, vx0_hi_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy1_lo_1, vx1_lo_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy1_hi_1, vx1_hi_0_1);
                    t[1] = __lasx_xvmul_h(vy0_lo_1, vx0_lo_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy0_hi_1, vx0_hi_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy1_lo_1, vx1_lo_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy1_hi_1, vx1_hi_2_3);
                    t[2] = __lasx_xvmul_h(vy0_lo_1, vx0_lo_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy0_hi_1, vx0_hi_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy1_lo_1, vx1_lo_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy1_hi_1, vx1_hi_4_5);
                    t[3] = __lasx_xvmul_h(vy0_lo_1, vx0_lo_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy0_hi_1, vx0_hi_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy1_lo_1, vx1_lo_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy1_hi_1, vx1_hi_6_7);

                    t[0] = __lasx_xvhaddw_w_h(t[0], t[0]);
                    t[0] = __lasx_xvadd_w(t[0], __lasx_xvshuf4i_w(t[0], 0x4E));
                    t[0] = __lasx_xvadd_w(t[0], __lasx_xvshuf4i_w(t[0], 0xB1));
                    t[1] = __lasx_xvhaddw_w_h(t[1], t[1]);
                    t[1] = __lasx_xvadd_w(t[1], __lasx_xvshuf4i_w(t[1], 0x4E));
                    t[1] = __lasx_xvadd_w(t[1], __lasx_xvshuf4i_w(t[1], 0xB1));
                    t[2] = __lasx_xvhaddw_w_h(t[2], t[2]);
                    t[2] = __lasx_xvadd_w(t[2], __lasx_xvshuf4i_w(t[2], 0x4E));
                    t[2] = __lasx_xvadd_w(t[2], __lasx_xvshuf4i_w(t[2], 0xB1));
                    t[3] = __lasx_xvhaddw_w_h(t[3], t[3]);
                    t[3] = __lasx_xvadd_w(t[3], __lasx_xvshuf4i_w(t[3], 0x4E));
                    t[3] = __lasx_xvadd_w(t[3], __lasx_xvshuf4i_w(t[3], 0xB1));

                    t[0] = __lasx_xvpickev_d(t[1], t[0]);
                    t[2] = __lasx_xvpickev_d(t[3], t[2]);

                    const __m256 tf = __lasx_xvffint_s_w(__lasx_xvpickev_w(t[2], t[0]));
                    const __m256 scalar =
                        __lasx_xvfmul_s(scalar_x, __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 1, 0)));
                    sumf[1] = __lasx_xvfmadd_s(tf, scalar, sumf[1]);
                }
                {
                    const __m256i vy0_lo_2 = __lasx_vext2xv_h_b(__lasx_xvpermi_d(vy0_lo, 0xaa));
                    const __m256i vy1_lo_2 = __lasx_vext2xv_h_b(__lasx_xvpermi_d(vy1_lo, 0xaa));
                    const __m256i vy0_hi_2 = __lasx_vext2xv_h_b(__lasx_xvpermi_d(vy0_hi, 0xaa));
                    const __m256i vy1_hi_2 = __lasx_vext2xv_h_b(__lasx_xvpermi_d(vy1_hi, 0xaa));

                    t[0] = __lasx_xvmul_h(vy0_lo_2, vx0_lo_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy0_hi_2, vx0_hi_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy1_lo_2, vx1_lo_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy1_hi_2, vx1_hi_0_1);
                    t[1] = __lasx_xvmul_h(vy0_lo_2, vx0_lo_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy0_hi_2, vx0_hi_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy1_lo_2, vx1_lo_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy1_hi_2, vx1_hi_2_3);
                    t[2] = __lasx_xvmul_h(vy0_lo_2, vx0_lo_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy0_hi_2, vx0_hi_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy1_lo_2, vx1_lo_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy1_hi_2, vx1_hi_4_5);
                    t[3] = __lasx_xvmul_h(vy0_lo_2, vx0_lo_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy0_hi_2, vx0_hi_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy1_lo_2, vx1_lo_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy1_hi_2, vx1_hi_6_7);
                    t[0] = __lasx_xvhaddw_w_h(t[0], t[0]);

                    t[0] = __lasx_xvadd_w(t[0], __lasx_xvshuf4i_w(t[0], 0x4E));
                    t[0] = __lasx_xvadd_w(t[0], __lasx_xvshuf4i_w(t[0], 0xB1));
                    t[1] = __lasx_xvhaddw_w_h(t[1], t[1]);
                    t[1] = __lasx_xvadd_w(t[1], __lasx_xvshuf4i_w(t[1], 0x4E));
                    t[1] = __lasx_xvadd_w(t[1], __lasx_xvshuf4i_w(t[1], 0xB1));
                    t[2] = __lasx_xvhaddw_w_h(t[2], t[2]);
                    t[2] = __lasx_xvadd_w(t[2], __lasx_xvshuf4i_w(t[2], 0x4E));
                    t[2] = __lasx_xvadd_w(t[2], __lasx_xvshuf4i_w(t[2], 0xB1));
                    t[3] = __lasx_xvhaddw_w_h(t[3], t[3]);
                    t[3] = __lasx_xvadd_w(t[3], __lasx_xvshuf4i_w(t[3], 0x4E));
                    t[3] = __lasx_xvadd_w(t[3], __lasx_xvshuf4i_w(t[3], 0xB1));

                    t[0] = __lasx_xvpickev_d(t[1], t[0]);
                    t[2] = __lasx_xvpickev_d(t[3], t[2]);

                    const __m256 tf = __lasx_xvffint_s_w(__lasx_xvpickev_w(t[2], t[0]));
                    const __m256 scalar =
                        __lasx_xvfmul_s(scalar_x, __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 2, 0)));
                    sumf[2] = __lasx_xvfmadd_s(tf, scalar, sumf[2]);
                }
                {
                    const __m256i vy0_lo_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_d(vy0_lo, 0xff));
                    const __m256i vy1_lo_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_d(vy1_lo, 0xff));
                    const __m256i vy0_hi_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_d(vy0_hi, 0xff));
                    const __m256i vy1_hi_3 = __lasx_vext2xv_h_b(__lasx_xvpermi_d(vy1_hi, 0xff));

                    t[0] = __lasx_xvmul_h(vy0_lo_3, vx0_lo_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy0_hi_3, vx0_hi_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy1_lo_3, vx1_lo_0_1);
                    t[0] = __lasx_xvmadd_h(t[0], vy1_hi_3, vx1_hi_0_1);
                    t[1] = __lasx_xvmul_h(vy0_lo_3, vx0_lo_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy0_hi_3, vx0_hi_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy1_lo_3, vx1_lo_2_3);
                    t[1] = __lasx_xvmadd_h(t[1], vy1_hi_3, vx1_hi_2_3);
                    t[2] = __lasx_xvmul_h(vy0_lo_3, vx0_lo_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy0_hi_3, vx0_hi_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy1_lo_3, vx1_lo_4_5);
                    t[2] = __lasx_xvmadd_h(t[2], vy1_hi_3, vx1_hi_4_5);
                    t[3] = __lasx_xvmul_h(vy0_lo_3, vx0_lo_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy0_hi_3, vx0_hi_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy1_lo_3, vx1_lo_6_7);
                    t[3] = __lasx_xvmadd_h(t[3], vy1_hi_3, vx1_hi_6_7);

                    t[0] = __lasx_xvhaddw_w_h(t[0], t[0]);
                    t[0] = __lasx_xvadd_w(t[0], __lasx_xvshuf4i_w(t[0], 0x4E));
                    t[0] = __lasx_xvadd_w(t[0], __lasx_xvshuf4i_w(t[0], 0xB1));
                    t[1] = __lasx_xvhaddw_w_h(t[1], t[1]);
                    t[1] = __lasx_xvadd_w(t[1], __lasx_xvshuf4i_w(t[1], 0x4E));
                    t[1] = __lasx_xvadd_w(t[1], __lasx_xvshuf4i_w(t[1], 0xB1));
                    t[2] = __lasx_xvhaddw_w_h(t[2], t[2]);
                    t[2] = __lasx_xvadd_w(t[2], __lasx_xvshuf4i_w(t[2], 0x4E));
                    t[2] = __lasx_xvadd_w(t[2], __lasx_xvshuf4i_w(t[2], 0xB1));
                    t[3] = __lasx_xvhaddw_w_h(t[3], t[3]);
                    t[3] = __lasx_xvadd_w(t[3], __lasx_xvshuf4i_w(t[3], 0x4E));
                    t[3] = __lasx_xvadd_w(t[3], __lasx_xvshuf4i_w(t[3], 0xB1));

                    t[0] = __lasx_xvpickev_d(t[1], t[0]);
                    t[2] = __lasx_xvpickev_d(t[3], t[2]);

                    const __m256 tf = __lasx_xvffint_s_w(__lasx_xvpickev_w(t[2], t[0]));
                    const __m256 scalar =
                        __lasx_xvfmul_s(scalar_x, __lasx_xvfcvtl_s_h(__lasx_xvldrepl_h(a_ptr[l].d + 3, 0)));
                    sumf[3] = __lasx_xvfmadd_s(tf, scalar, sumf[3]);
                }
            }
            const int32_t sum_perm_data[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };
            const __m256i sum_perm         = __lasx_xvld(sum_perm_data, 0);

            for (int m = 0; m < 4; m++) {
                __lasx_xvst(__lasx_xvperm_w((__m256i) sumf[m], sum_perm), s + (y * 4 + m) * bs + x * ncols_interleaved,
                            0);
            }
        }
    }
#else
    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#endif
}
