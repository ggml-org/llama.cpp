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

#if defined __riscv_v
    if (__riscv_vlenb() >= QK4_0) {
        const size_t vl = QK4_0;

        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);

            vfloat32m1_t sumf = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
            for (int l = 0; l < nb; l++) {
                const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[0];
                const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[8];
                const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[16];
                const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[24];
                __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment constraints
                const vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a0, vl / 4));
                const vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a1, vl / 4));
                const vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a2, vl / 4));
                const vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a3, vl / 4));

                const vint8m4_t rhs_raw_vec = __riscv_vle8_v_i8m4((const int8_t *)b_ptr[l].qs, vl * 4);
                const vint8m4_t rhs_vec_lo = __riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(rhs_raw_vec, 4, vl * 4), 4, vl * 4);
                const vint8m4_t rhs_vec_hi = __riscv_vsra_vx_i8m4(rhs_raw_vec, 4, vl * 4);
                const vint8m2_t rhs_vec_lo_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 0);
                const vint8m2_t rhs_vec_lo_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 1);
                const vint8m2_t rhs_vec_hi_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 0);
                const vint8m2_t rhs_vec_hi_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 1);

                const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                const vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                const vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                const vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_hi_m));
                const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                const vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                const vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                const vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                const vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                const vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                const vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                const vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                // vector version needs Zvfhmin extension
                const float a_scale = GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                const float b_scales[8] = {
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[4]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[5]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[6]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[7])
                };
                const vfloat32m1_t b_scales_vec = __riscv_vle32_v_f32m1(b_scales, vl / 4);
                const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scale, vl / 4);
                sumf = __riscv_vfmacc_vv_f32m1(sumf, tmp1, b_scales_vec, vl / 4);
            }
            __riscv_vse32_v_f32m1(s + x * ncols_interleaved, sumf, vl / 4);
        }
        return;
    }

#endif
    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
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

#if defined __riscv_v
    if (__riscv_vlenb() >= QK4_0) {
        const size_t vl = QK4_0;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);
                vfloat32m1_t sumf0 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                vfloat32m1_t sumf1 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                vfloat32m1_t sumf2 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                vfloat32m1_t sumf3 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                for (int l = 0; l < nb; l++) {
                    const vint8m4_t rhs_raw_vec = __riscv_vle8_v_i8m4((const int8_t *)b_ptr[l].qs, vl * 4);
                    const vint8m4_t rhs_vec_lo = __riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(rhs_raw_vec, 4, vl * 4), 4, vl * 4);
                    const vint8m4_t rhs_vec_hi = __riscv_vsra_vx_i8m4(rhs_raw_vec, 4, vl * 4);
                    const vint8m2_t rhs_vec_lo_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 0);
                    const vint8m2_t rhs_vec_lo_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 1);
                    const vint8m2_t rhs_vec_hi_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 0);
                    const vint8m2_t rhs_vec_hi_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 1);

                    // vector version needs Zvfhmin extension
                    const float a_scales[4] = {
                        GGML_CPU_FP16_TO_FP32(a_ptr[l].d[0]),
                        GGML_CPU_FP16_TO_FP32(a_ptr[l].d[1]),
                        GGML_CPU_FP16_TO_FP32(a_ptr[l].d[2]),
                        GGML_CPU_FP16_TO_FP32(a_ptr[l].d[3])
                    };
                    const float b_scales[8] = {
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[4]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[5]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[6]),
                        GGML_CPU_FP16_TO_FP32(b_ptr[l].d[7])
                    };
                    const vfloat32m1_t b_scales_vec = __riscv_vle32_v_f32m1(b_scales, vl / 4);

                    const int64_t A0 = *(const int64_t *)&a_ptr[l].qs[0];
                    const int64_t A4 = *(const int64_t *)&a_ptr[l].qs[32];
                    const int64_t A8 = *(const int64_t *)&a_ptr[l].qs[64];
                    const int64_t Ac = *(const int64_t *)&a_ptr[l].qs[96];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l0;
                    {
                        const vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A0, vl / 4));
                        const vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A4, vl / 4));
                        const vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A8, vl / 4));
                        const vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ac, vl / 4));
                        const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        const vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        const vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        const vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l0 = sumi_hi_m;
                    }

                    {
                        const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l0));
                        const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        const vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        const vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        const vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        const vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        const vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        const vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        const vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[0], vl / 4);
                        sumf0 = __riscv_vfmacc_vv_f32m1(sumf0, tmp1, b_scales_vec, vl / 4);
                    }

                    const int64_t A1 = *(const int64_t *)&a_ptr[l].qs[8];
                    const int64_t A5 = *(const int64_t *)&a_ptr[l].qs[40];
                    const int64_t A9 = *(const int64_t *)&a_ptr[l].qs[72];
                    const int64_t Ad = *(const int64_t *)&a_ptr[l].qs[104];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l1;
                    {
                        const vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A1, vl / 4));
                        const vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A5, vl / 4));
                        const vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A9, vl / 4));
                        const vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ad, vl / 4));
                        const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        const vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        const vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        const vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l1 = sumi_hi_m;
                    }

                    {
                        const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l1));
                        const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        const vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        const vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        const vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        const vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        const vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        const vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        const vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[1], vl / 4);
                        sumf1 = __riscv_vfmacc_vv_f32m1(sumf1, tmp1, b_scales_vec, vl / 4);
                    }

                    const int64_t A2 = *(const int64_t *)&a_ptr[l].qs[16];
                    const int64_t A6 = *(const int64_t *)&a_ptr[l].qs[48];
                    const int64_t Aa = *(const int64_t *)&a_ptr[l].qs[80];
                    const int64_t Ae = *(const int64_t *)&a_ptr[l].qs[112];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l2;
                    {
                        const vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A2, vl / 4));
                        const vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A6, vl / 4));
                        const vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Aa, vl / 4));
                        const vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ae, vl / 4));
                        const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        const vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        const vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        const vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l2 = sumi_hi_m;
                    }

                    {
                        const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l2));
                        const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        const vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        const vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        const vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        const vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        const vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        const vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        const vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[2], vl / 4);
                        sumf2 = __riscv_vfmacc_vv_f32m1(sumf2, tmp1, b_scales_vec, vl / 4);
                    }

                    const int64_t A3 = *(const int64_t *)&a_ptr[l].qs[24];
                    const int64_t A7 = *(const int64_t *)&a_ptr[l].qs[56];
                    const int64_t Ab = *(const int64_t *)&a_ptr[l].qs[88];
                    const int64_t Af = *(const int64_t *)&a_ptr[l].qs[120];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l3;
                    {
                        const vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A3, vl / 4));
                        const vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A7, vl / 4));
                        const vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ab, vl / 4));
                        const vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Af, vl / 4));
                        const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        const vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        const vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        const vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l3 = sumi_hi_m;
                    }

                    {
                        const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l3));
                        const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        const vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        const vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        const vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        const vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        const vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        const vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        const vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[3], vl / 4);
                        sumf3 = __riscv_vfmacc_vv_f32m1(sumf3, tmp1, b_scales_vec, vl / 4);
                    }
                }
                __riscv_vse32_v_f32m1(&s[(y * 4 + 0) * bs + x * ncols_interleaved], sumf0, vl / 4);
                __riscv_vse32_v_f32m1(&s[(y * 4 + 1) * bs + x * ncols_interleaved], sumf1, vl / 4);
                __riscv_vse32_v_f32m1(&s[(y * 4 + 2) * bs + x * ncols_interleaved], sumf2, vl / 4);
                __riscv_vse32_v_f32m1(&s[(y * 4 + 3) * bs + x * ncols_interleaved], sumf3, vl / 4);
            }
        }

        return;
    }

#endif
    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

template<int ncols_interleaved>
static inline void ggml_gemv_f16_1xM_f16(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    GGML_UNUSED(bs);

    const int nb = n / 1;

    assert (nr == 1);
    assert(n % 1 == 0);
    assert(nc % ncols_interleaved == 0);

    const _Float16 * a_ptr = (const _Float16 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_f16<ncols_interleaved, 1> * b_ptr = (const block_f16<ncols_interleaved, 1> *) vx + (x * nb);

        // Accumulators
        vfloat32m4_t sumf_0 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);

        for (int l = 0; l < nb; l++) {
            vfloat16m2_t b_0 = __riscv_vle16_v_f16m2((const _Float16 *)&b_ptr[l].d[0], ncols_interleaved);

            sumf_0 = __riscv_vfwmacc_vf_f32m4(sumf_0, *(const _Float16*)(&a_ptr[l]), b_0, ncols_interleaved);
        }

        __riscv_vse32_v_f32m4(&s[x * ncols_interleaved], sumf_0, ncols_interleaved);
    }

    return;
}

void ggml_gemv_f16_1x16_f16(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemv_f16_1xM_f16<16>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemv_f16_1x16_f16_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemv_f16_1x32_f16(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemv_f16_1xM_f16<32>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemv_f16_1x32_f16_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemv_f16_1x64_f16(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemv_f16_1xM_f16<64>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemv_f16_1x64_f16_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemv_f16_1x128_f16(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemv_f16_1xM_f16<128>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemv_f16_1x128_f16_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

template<int ncols_interleaved>
static inline void ggml_gemv_f32_1xM_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    GGML_UNUSED(bs);

    const int nb = n / 1;

    assert (nr == 1);
    assert(n % 1 == 0);
    assert(nc % ncols_interleaved == 0);

    const float * a_ptr = (const float *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_f32<ncols_interleaved, 1> * b_ptr = (const block_f32<ncols_interleaved, 1> *) vx + (x * nb);

        // Accumulators
        vfloat32m4_t sumf_0 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);

        for (int l = 0; l < nb; l++) {
            vfloat32m4_t b_0 = __riscv_vle32_v_f32m4((const float *)&b_ptr[l].d[0], ncols_interleaved);

            sumf_0 = __riscv_vfmacc_vf_f32m4(sumf_0, *(const float*)(&a_ptr[l]), b_0, ncols_interleaved);
        }

        __riscv_vse32_v_f32m4(&s[x * ncols_interleaved], sumf_0, ncols_interleaved);
    }

    return;
}

void ggml_gemv_f32_1x16_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemv_f32_1xM_f32<16>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemv_f32_1x16_f32_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemv_f32_1x32_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemv_f32_1xM_f32<32>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemv_f32_1x32_f32_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemv_f32_1x64_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemv_f32_1xM_f32<64>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemv_f32_1x64_f32_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemv_f32_1x128_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemv_f32_1xM_f32<128>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemv_f32_1x128_f32_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

template<int ncols_interleaved>
static inline void ggml_gemm_f16_7x1xM_f16(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / 1;

    assert (nr % 7 == 0);
    assert(n % 1 == 0);
    assert(nc % ncols_interleaved == 0);

    for (int y = 0; y < nr / 7; y++) {
        const block_f16_7x1 * a_ptr = (const block_f16_7x1*) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_f16<ncols_interleaved, 1> * b_ptr = (const block_f16<ncols_interleaved, 1> *) vx + (x * nb);

            // Accumulators
            vfloat32m4_t sumf_0 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_1 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_2 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_3 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_4 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_5 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_6 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);

            for (int l = 0; l < nb; l++) {
                vfloat16m2_t b_0 = __riscv_vle16_v_f16m2((const _Float16 *)&b_ptr[l].d[0], ncols_interleaved);

                sumf_0 = __riscv_vfwmacc_vf_f32m4(sumf_0, *(const _Float16*)&a_ptr[l].d[0], b_0, ncols_interleaved);
                sumf_1 = __riscv_vfwmacc_vf_f32m4(sumf_1, *(const _Float16*)&a_ptr[l].d[1], b_0, ncols_interleaved);
                sumf_2 = __riscv_vfwmacc_vf_f32m4(sumf_2, *(const _Float16*)&a_ptr[l].d[2], b_0, ncols_interleaved);
                sumf_3 = __riscv_vfwmacc_vf_f32m4(sumf_3, *(const _Float16*)&a_ptr[l].d[3], b_0, ncols_interleaved);
                sumf_4 = __riscv_vfwmacc_vf_f32m4(sumf_4, *(const _Float16*)&a_ptr[l].d[4], b_0, ncols_interleaved);
                sumf_5 = __riscv_vfwmacc_vf_f32m4(sumf_5, *(const _Float16*)&a_ptr[l].d[5], b_0, ncols_interleaved);
                sumf_6 = __riscv_vfwmacc_vf_f32m4(sumf_6, *(const _Float16*)&a_ptr[l].d[6], b_0, ncols_interleaved);
            }

            __riscv_vse32_v_f32m4(&s[(y * 7 + 0) * bs + x * ncols_interleaved], sumf_0, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 1) * bs + x * ncols_interleaved], sumf_1, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 2) * bs + x * ncols_interleaved], sumf_2, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 3) * bs + x * ncols_interleaved], sumf_3, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 4) * bs + x * ncols_interleaved], sumf_4, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 5) * bs + x * ncols_interleaved], sumf_5, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 6) * bs + x * ncols_interleaved], sumf_6, ncols_interleaved);
        }
    }
    return;
}

void ggml_gemm_f16_7x1x16_f16(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemm_f16_7x1xM_f16<16>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemm_f16_7x1x16_f16_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_f16_7x1x32_f16(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemm_f16_7x1xM_f16<32>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemm_f16_7x1x32_f16_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_f16_7x1x64_f16(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemm_f16_7x1xM_f16<64>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemm_f16_7x1x64_f16_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_f16_7x1x128_f16(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemm_f16_7x1xM_f16<128>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemm_f16_7x1x128_f16_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

template<int ncols_interleaved>
static inline void ggml_gemm_f32_7x1xM_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / 1;

    assert (nr % 7 == 0);
    assert(n % 1 == 0);
    assert(nc % ncols_interleaved == 0);

    for (int y = 0; y < nr / 7; y++) {
        const block_f32_7x1 * a_ptr = (const block_f32_7x1*) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_f32<ncols_interleaved, 1> * b_ptr = (const block_f32<ncols_interleaved, 1> *) vx + (x * nb);

            // Accumulators
            vfloat32m4_t sumf_0 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_1 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_2 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_3 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_4 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_5 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);
            vfloat32m4_t sumf_6 = __riscv_vfmv_v_f_f32m4(0.0f, ncols_interleaved);

            for (int l = 0; l < nb; l++) {
                vfloat32m4_t b_0 = __riscv_vle32_v_f32m4((const float*)&b_ptr[l].d[0], ncols_interleaved);

                sumf_0 = __riscv_vfmacc_vf_f32m4(sumf_0, *(const float*)&a_ptr[l].d[0], b_0, ncols_interleaved);
                sumf_1 = __riscv_vfmacc_vf_f32m4(sumf_1, *(const float*)&a_ptr[l].d[1], b_0, ncols_interleaved);
                sumf_2 = __riscv_vfmacc_vf_f32m4(sumf_2, *(const float*)&a_ptr[l].d[2], b_0, ncols_interleaved);
                sumf_3 = __riscv_vfmacc_vf_f32m4(sumf_3, *(const float*)&a_ptr[l].d[3], b_0, ncols_interleaved);
                sumf_4 = __riscv_vfmacc_vf_f32m4(sumf_4, *(const float*)&a_ptr[l].d[4], b_0, ncols_interleaved);
                sumf_5 = __riscv_vfmacc_vf_f32m4(sumf_5, *(const float*)&a_ptr[l].d[5], b_0, ncols_interleaved);
                sumf_6 = __riscv_vfmacc_vf_f32m4(sumf_6, *(const float*)&a_ptr[l].d[6], b_0, ncols_interleaved);
            }

            __riscv_vse32_v_f32m4(&s[(y * 7 + 0) * bs + x * ncols_interleaved], sumf_0, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 1) * bs + x * ncols_interleaved], sumf_1, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 2) * bs + x * ncols_interleaved], sumf_2, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 3) * bs + x * ncols_interleaved], sumf_3, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 4) * bs + x * ncols_interleaved], sumf_4, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 5) * bs + x * ncols_interleaved], sumf_5, ncols_interleaved);
            __riscv_vse32_v_f32m4(&s[(y * 7 + 6) * bs + x * ncols_interleaved], sumf_6, ncols_interleaved);
        }
    }
    return;
}

void ggml_gemm_f32_7x1x16_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemm_f32_7x1xM_f32<16>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemm_f32_7x1x16_f32_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_f32_7x1x32_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemm_f32_7x1xM_f32<32>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemm_f32_7x1x32_f32_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_f32_7x1x64_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemm_f32_7x1xM_f32<64>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemm_f32_7x1x64_f32_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_f32_7x1x128_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined __riscv_zvfh
    ggml_gemm_f32_7x1xM_f32<128>(n, s, bs, vx, vy, nr, nc);
#else
    ggml_gemm_f32_7x1x128_f32_generic(n, s, bs, vx, vy, nr, nc);
#endif
}
