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

void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

#if defined(__riscv_v_intrinsic)
    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;
    const size_t vl_calc = __riscv_vsetvl_e32m8(QK8_0);
    const size_t vl_save = __riscv_vsetvl_e64m2(4);
    vfloat32m1_t v_scalar_zero = __riscv_vfmv_s_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));

    for (int i = 0; i < nb; i++) {
        const float *x_block_base = x + i * QK8_0;
        vint8m2_t q_r0, q_r1, q_r2, q_r3;
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 0 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[0] = GGML_CPU_FP32_TO_FP16(d);

            float id = d ? 1.0f / d : 0.0f;
            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r0 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm volatile ("" ::: "memory");

        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 1 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[1] = GGML_CPU_FP32_TO_FP16(d);
            float id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r1 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm volatile ("" ::: "memory");
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 2 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[2] = GGML_CPU_FP32_TO_FP16(d);
            float id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r2 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        asm volatile ("" ::: "memory");
        {
            vfloat32m8_t v_src = __riscv_vle32_v_f32m8(x_block_base + 3 * k, vl_calc);
            vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_src, vl_calc);
            vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_abs, v_scalar_zero, vl_calc);
            float amax = __riscv_vfmv_f_s_f32m1_f32(v_max);

            float d = amax / 127.0f;
            y[i].d[3] = GGML_CPU_FP32_TO_FP16(d);
            float id = d ? 1.0f / d : 0.0f;

            vfloat32m8_t v_scaled = __riscv_vfmul_vf_f32m8(v_src, id, vl_calc);
            vint16m4_t v_i16 = __riscv_vfncvt_x_f_w_i16m4_rm(v_scaled, 4, vl_calc);
            q_r3 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl_calc);
        }
        vint64m2_t v_q64_r0 = __riscv_vreinterpret_v_i8m2_i64m2(q_r0);
        vint64m2_t v_q64_r1 = __riscv_vreinterpret_v_i8m2_i64m2(q_r1);
        vint64m2_t v_q64_r2 = __riscv_vreinterpret_v_i8m2_i64m2(q_r2);
        vint64m2_t v_q64_r3 = __riscv_vreinterpret_v_i8m2_i64m2(q_r3);
        vint64m2x4_t v_quant_tuple = __riscv_vcreate_v_i64m2x4(v_q64_r0, v_q64_r1, v_q64_r2, v_q64_r3);
        __riscv_vsseg4e64_v_i64m2x4((int64_t*)y[i].qs, v_quant_tuple, vl_save);
    }
#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#endif
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

void ggml_gemv_iq4_nl_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

#if defined __riscv_v_intrinsic
    const vint8m2_t values = __riscv_vle8_v_i8m2(kvalues_iq4nl, 16);
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

        vfloat32mf2_t sumf = __riscv_vfmv_v_f_f32mf2(0.0, 4);
        for (int l = 0; l < nb; l++) {
            // Load first 8 bytes of `a`.
            const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[0];
            const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[8];
            const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[16];
            const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[24];
            __asm__ __volatile__("" ::: "memory");

            // Load `b_ptr`.
            const vuint8m2_t b_0_packed = __riscv_vle8_v_u8m2((const uint8_t *)b_ptr[l].qs, QK4_NL * 2);
            const vint8m2_t b_0_lo = __riscv_vrgather_vv_i8m2(values, __riscv_vand_vx_u8m2(b_0_packed, 0xf, QK4_NL * 2), QK4_NL * 2);
            const vint8m2_t b_0_hi = __riscv_vrgather_vv_i8m2(values, __riscv_vsrl_vx_u8m2(b_0_packed, 4, QK4_NL * 2), QK4_NL * 2);

            // Create 4 segments from `b`.
            const vint8m1_t b_lo_0 = __riscv_vget_v_i8m2_i8m1(b_0_lo, 0);
            const vint8m1_t b_lo_1 = __riscv_vget_v_i8m2_i8m1(b_0_lo, 1);
            const vint8m1_t b_hi_0 = __riscv_vget_v_i8m2_i8m1(b_0_hi, 0);
            const vint8m1_t b_hi_1 = __riscv_vget_v_i8m2_i8m1(b_0_hi, 1);

            // Broadcast `a_ptr` across 4 registers (8 bytes / register).
            const vint8m1_t a_0 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a0, 4));
            const vint8m1_t a_1 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a1, 4));
            const vint8m1_t a_2 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a2, 4));
            const vint8m1_t a_3 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a3, 4));

            // Multiply and accumulate.
            const vint16m2_t sumi_lo_0 = __riscv_vwmul_vv_i16m2(b_lo_0, a_0, QK4_NL);
            const vint16m2_t sumi_lo_1 = __riscv_vwmul_vv_i16m2(b_lo_1, a_1, QK4_NL);
            const vint16m2_t sumi_hi_0 = __riscv_vwmul_vv_i16m2(b_hi_0, a_2, QK4_NL);
            const vint16m2_t sumi_hi_1 = __riscv_vwmul_vv_i16m2(b_hi_1, a_3, QK4_NL);
            const vint32m4_t sumi_lo = __riscv_vwadd_vv_i32m4(sumi_lo_0, sumi_lo_1, QK4_NL);
            const vint32m4_t sumi_hi = __riscv_vwadd_vv_i32m4(sumi_hi_0, sumi_hi_1, QK4_NL);
            const vint32m4_t sumi = __riscv_vadd_vv_i32m4(sumi_lo, sumi_hi, QK4_NL);

            // In-place reduction.
            const vuint64m4_t sumi_i32 = __riscv_vreinterpret_v_i64m4_u64m4(__riscv_vreinterpret_v_i32m4_i64m4(sumi));
            const vuint32m2_t sumi_h2_0 = __riscv_vnsrl_wx_u32m2(sumi_i32, 0, QK4_NL / 2);
            const vuint32m2_t sumi_h2_1 = __riscv_vnsrl_wx_u32m2(sumi_i32, 32, QK4_NL / 2);
            const vuint32m2_t sumi_h2 = __riscv_vadd_vv_u32m2(sumi_h2_0, sumi_h2_1, QK4_NL/ 2);
            const vuint64m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u32m2_u64m2(sumi_h2);
            const vuint32m1_t sumi_h4_0 = __riscv_vnsrl_wx_u32m1(sumi_h2_i32, 0, QK4_NL / 4);
            const vuint32m1_t sumi_h4_1 = __riscv_vnsrl_wx_u32m1(sumi_h2_i32, 32, QK4_NL / 4);
            const vuint32m1_t sumi_h4 = __riscv_vadd_vv_u32m1(sumi_h4_0, sumi_h4_1, QK4_NL / 4);
            const vuint64m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u32m1_u64m1(sumi_h4);
            const vint32mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vnsrl_wx_u32mf2(sumi_h4_i32, 0, QK4_NL / 8));
            const vint32mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vnsrl_wx_u32mf2(sumi_h4_i32, 32, QK4_NL / 8));
            const vint32mf2_t sumi_h8 = __riscv_vadd_vv_i32mf2(sumi_h8_0, sumi_h8_1, QK4_NL / 8);
            const vfloat32mf2_t facc = __riscv_vfcvt_f_x_v_f32mf2(sumi_h8, QK4_NL / 8);

            // Multiply with scales.
            const vfloat16mf4_t b_d = __riscv_vle16_v_f16mf4((const _Float16 *)b_ptr[l].d, 4);
            const vfloat32mf2_t d_0 = __riscv_vfwmul_vf_f32mf2(b_d, *(const _Float16*)&a_ptr[l].d, 4);
            sumf = __riscv_vfmacc_vv_f32mf2(sumf, facc, d_0, QK4_NL / 8);
        }
        __riscv_vse32_v_f32mf2(s + x * ncols_interleaved, sumf, QK4_NL / 8);
    }
    return;

#endif
    ggml_gemv_iq4_nl_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq4_nl_4x16_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

#if defined __riscv_v_intrinsic
    const vint8m1_t values = __riscv_vle8_v_i8m1(kvalues_iq4nl, 16);
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

        vfloat32m1_t sumf = __riscv_vfmv_v_f_f32m1(0.0f, 4);
        for (int l = 0; l + 1 < nb; l += 2) {
                vuint8m1_t b_0_packed = __riscv_vle8_v_u8m1(b_ptr[l].qs + 0, 16);
                vuint8m1_t b_1_packed = __riscv_vle8_v_u8m1(b_ptr[l].qs + 16, 16);
                vuint8m1_t b_2_packed = __riscv_vle8_v_u8m1(b_ptr[l].qs + 32, 16);
                vuint8m1_t b_3_packed = __riscv_vle8_v_u8m1(b_ptr[l].qs + 48, 16);
                vuint8m1_t b_4_packed = __riscv_vle8_v_u8m1(b_ptr[l + 1].qs + 0, 16);
                vuint8m1_t b_5_packed = __riscv_vle8_v_u8m1(b_ptr[l + 1].qs + 16, 16);
                vuint8m1_t b_6_packed = __riscv_vle8_v_u8m1(b_ptr[l + 1].qs + 32, 16);
                vuint8m1_t b_7_packed = __riscv_vle8_v_u8m1(b_ptr[l + 1].qs + 48, 16);

                vuint8m1_t b_0_lo = __riscv_vand_vx_u8m1(b_0_packed, 0xf, 16);
                vuint8m1_t b_0_hi = __riscv_vsrl_vx_u8m1(b_0_packed, 4, 16);
                vuint8m1_t b_1_lo = __riscv_vand_vx_u8m1(b_1_packed, 0xf, 16);
                vuint8m1_t b_1_hi = __riscv_vsrl_vx_u8m1(b_1_packed, 4, 16);
                vuint8m1_t b_2_lo = __riscv_vand_vx_u8m1(b_2_packed, 0xf, 16);
                vuint8m1_t b_2_hi = __riscv_vsrl_vx_u8m1(b_2_packed, 4, 16);
                vuint8m1_t b_3_lo = __riscv_vand_vx_u8m1(b_3_packed, 0xf, 16);
                vuint8m1_t b_3_hi = __riscv_vsrl_vx_u8m1(b_3_packed, 4, 16);
                vuint8m1_t b_4_lo = __riscv_vand_vx_u8m1(b_4_packed, 0xf, 16);
                vuint8m1_t b_4_hi = __riscv_vsrl_vx_u8m1(b_4_packed, 4, 16);
                vuint8m1_t b_5_lo = __riscv_vand_vx_u8m1(b_5_packed, 0xf, 16);
                vuint8m1_t b_5_hi = __riscv_vsrl_vx_u8m1(b_5_packed, 4, 16);
                vuint8m1_t b_6_lo = __riscv_vand_vx_u8m1(b_6_packed, 0xf, 16);
                vuint8m1_t b_6_hi = __riscv_vsrl_vx_u8m1(b_6_packed, 4, 16);
                vuint8m1_t b_7_lo = __riscv_vand_vx_u8m1(b_7_packed, 0xf, 16);
                vuint8m1_t b_7_hi = __riscv_vsrl_vx_u8m1(b_7_packed, 4, 16);

                vint8m1_t b_0 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_0_lo, b_0_hi, 16, 32), 32);
                vint8m1_t b_1 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_1_lo, b_1_hi, 16, 32), 32);
                vint8m1_t b_2 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_2_lo, b_2_hi, 16, 32), 32);
                vint8m1_t b_3 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_3_lo, b_3_hi, 16, 32), 32);
                vint8m1_t b_4 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_4_lo, b_4_hi, 16, 32), 32);
                vint8m1_t b_5 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_5_lo, b_5_hi, 16, 32), 32);
                vint8m1_t b_6 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_6_lo, b_6_hi, 16, 32), 32);
                vint8m1_t b_7 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_7_lo, b_7_hi, 16, 32), 32);

                vint8m1_t a_0 = __riscv_vle8_v_i8m1(a_ptr[l].qs, 32);
                vint8m1_t a_1 = __riscv_vle8_v_i8m1(a_ptr[l + 1].qs, 32);

                vint32m1_t sumi_0 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_0, b_0, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                vint32m1_t sumi_1 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_0, b_1, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                vint32m1_t sumi_2 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_0, b_2, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                vint32m1_t sumi_3 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_0, b_3, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                vint32m1_t sumi_4 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_1, b_4, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                vint32m1_t sumi_5 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_1, b_5, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                vint32m1_t sumi_6 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_1, b_6, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                vint32m1_t sumi_7 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_1, b_7, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);

                int sumi_temp[8];
                __riscv_vse32_v_i32m1(&sumi_temp[0], sumi_0, 1);
                __riscv_vse32_v_i32m1(&sumi_temp[1], sumi_1, 1);
                __riscv_vse32_v_i32m1(&sumi_temp[2], sumi_2, 1);
                __riscv_vse32_v_i32m1(&sumi_temp[3], sumi_3, 1);
                __riscv_vse32_v_i32m1(&sumi_temp[4], sumi_4, 1);
                __riscv_vse32_v_i32m1(&sumi_temp[5], sumi_5, 1);
                __riscv_vse32_v_i32m1(&sumi_temp[6], sumi_6, 1);
                __riscv_vse32_v_i32m1(&sumi_temp[7], sumi_7, 1);
                vint32m1_t sum_0 = __riscv_vle32_v_i32m1(&sumi_temp[0], 4);
                vint32m1_t sum_1 = __riscv_vle32_v_i32m1(&sumi_temp[4], 4);

                vfloat16mf2_t b_d_0 = __riscv_vle16_v_f16mf2((_Float16 *)b_ptr[l].d, 4);
                vfloat32m1_t d_0 = __riscv_vfwmul_vf_f32m1(b_d_0, *(const _Float16 *)&a_ptr[l].d, 4);
                vfloat16mf2_t b_d_1 = __riscv_vle16_v_f16mf2((_Float16 *)b_ptr[l + 1].d, 4);
                vfloat32m1_t d_1 = __riscv_vfwmul_vf_f32m1(b_d_1, *(const _Float16 *)&a_ptr[l + 1].d, 4);

                sumf = __riscv_vfmacc_vv_f32m1(sumf, d_0, __riscv_vfcvt_f_x_v_f32m1(sum_0, 4), 4);
                sumf = __riscv_vfmacc_vv_f32m1(sumf, d_1, __riscv_vfcvt_f_x_v_f32m1(sum_1, 4), 4);
        }
        __riscv_vse32_v_f32m1(s + x * ncols_interleaved, sumf, 4);
    }
    return;
#endif
    ggml_gemv_iq4_nl_4x16_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq4_nl_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

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

#if defined __riscv_v_intrinsic
    const vint8mf2_t values = __riscv_vle8_v_i8mf2(kvalues_iq4nl, 16);
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx16 * b_ptr = (const block_iq4_nlx16 *) vx + (x * nb);

        // 1x16 Accumulator1
        vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);

        for (int l = 0; l < nb; l++) {
            // 1x16 integer accumulator
            vint32m2_t sumi = __riscv_vmv_v_x_i32m2(0.0f, 16);

            // Accumulation loop.
            for (int i = 0; i < QK4_NL / 2; i++) {
                // Load `b_ptr`.
                const vuint8mf2_t b_0_packed = __riscv_vle8_v_u8mf2((const uint8_t *)&b_ptr[l].qs[i * 16], 16);
                const vint8mf2_t b_0_lo = __riscv_vrgather_vv_i8mf2(values, __riscv_vand_vx_u8mf2(b_0_packed, 0xf, 16), 16);
                const vint8mf2_t b_0_hi = __riscv_vrgather_vv_i8mf2(values, __riscv_vsrl_vx_u8mf2(b_0_packed, 4, 16), 16);
                // const vint16m1_t b_0_lo_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_lo, 16);
                // const vint16m1_t b_0_hi_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_hi, 16);

                const vint16m1_t sumi_lo = __riscv_vwmul_vx_i16m1(b_0_lo, a_ptr[l].qs[i], 16);
                const vint16m1_t sumi_hi = __riscv_vwmul_vx_i16m1(b_0_hi, a_ptr[l].qs[16 + i], 16);
                sumi = __riscv_vadd_vv_i32m2(sumi, __riscv_vwadd_vv_i32m2(sumi_lo, sumi_hi, 16), 16);
            }

            const vfloat16m1_t b_d = __riscv_vle16_v_f16m1((_Float16 *)b_ptr[l].d, 16);
            const vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(const _Float16 *)&a_ptr[l].d, 16);

            sumf = __riscv_vfmacc_vv_f32m2(sumf, __riscv_vfcvt_f_x_v_f32m2(sumi, 16), d_0, 16);
        }

        __riscv_vse32_v_f32m2(s + x * 16, sumf, 16);
    }
    return;
#endif
    ggml_gemv_iq4_nl_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq4_nl_4x16_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

#if defined __riscv_v_intrinsic
    const vint8m1_t values = __riscv_vle8_v_i8m1(kvalues_iq4nl, 16);

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

            // 4x4 Accumulators
            vfloat32m1_t sumf_0 = __riscv_vfmv_v_f_f32m1(0.0f, 4);
            vfloat32m1_t sumf_1 = __riscv_vfmv_v_f_f32m1(0.0f, 4);
            vfloat32m1_t sumf_2 = __riscv_vfmv_v_f_f32m1(0.0f, 4);
            vfloat32m1_t sumf_3 = __riscv_vfmv_v_f_f32m1(0.0f, 4);

            for (int l = 0; l < nb; l++) {
                int sumi_temp[16];
                uint8_t index[4] = {0, 8, 64, 72};
                vuint8mf8_t i_vec = __riscv_vle8_v_u8mf8(&index[0], 4);
                vuint8m1_t b_0_packed = __riscv_vle8_v_u8m1(b_ptr[l].qs + 0, 16);
                vuint8m1_t b_1_packed = __riscv_vle8_v_u8m1(b_ptr[l].qs + 16, 16);
                vuint8m1_t b_2_packed = __riscv_vle8_v_u8m1(b_ptr[l].qs + 32, 16);
                vuint8m1_t b_3_packed = __riscv_vle8_v_u8m1(b_ptr[l].qs + 48, 16);

                vuint8m1_t b_0_lo = __riscv_vand_vx_u8m1(b_0_packed, 0xf, 16);
                vuint8m1_t b_0_hi = __riscv_vsrl_vx_u8m1(b_0_packed, 4, 16);
                vuint8m1_t b_1_lo = __riscv_vand_vx_u8m1(b_1_packed, 0xf, 16);
                vuint8m1_t b_1_hi = __riscv_vsrl_vx_u8m1(b_1_packed, 4, 16);
                vuint8m1_t b_2_lo = __riscv_vand_vx_u8m1(b_2_packed, 0xf, 16);
                vuint8m1_t b_2_hi = __riscv_vsrl_vx_u8m1(b_2_packed, 4, 16);
                vuint8m1_t b_3_lo = __riscv_vand_vx_u8m1(b_3_packed, 0xf, 16);
                vuint8m1_t b_3_hi = __riscv_vsrl_vx_u8m1(b_3_packed, 4, 16);

                vint8m1_t b_0 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_0_lo, b_0_hi, 16, 32), 32);
                vint8m1_t b_1 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_1_lo, b_1_hi, 16, 32), 32);
                vint8m1_t b_2 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_2_lo, b_2_hi, 16, 32), 32);
                vint8m1_t b_3 = __riscv_vrgather_vv_i8m1(values, __riscv_vslideup_vx_u8m1(b_3_lo, b_3_hi, 16, 32), 32);

                #pragma unroll 4
                for (int i = 0; i < 4; i++) {
                    vint8m1_t a_i = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vloxei8_v_i64m1((int64_t*)(a_ptr[l].qs + i * 16), i_vec, 4));
                    vint32m1_t sumi_0 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_i, b_0, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                    vint32m1_t sumi_1 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_i, b_1, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                    vint32m1_t sumi_2 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_i, b_2, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                    vint32m1_t sumi_3 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(a_i, b_3, 32), __riscv_vmv_v_x_i32m1(0, 1), 32);
                    __riscv_vse32_v_i32m1(&sumi_temp[i * 4 + 0], sumi_0, 1);
                    __riscv_vse32_v_i32m1(&sumi_temp[i * 4 + 1], sumi_1, 1);
                    __riscv_vse32_v_i32m1(&sumi_temp[i * 4 + 2], sumi_2, 1);
                    __riscv_vse32_v_i32m1(&sumi_temp[i * 4 + 3], sumi_3, 1);
                }

                vint32m1_t sum_0 = __riscv_vle32_v_i32m1(&sumi_temp[0], 4);
                vint32m1_t sum_1 = __riscv_vle32_v_i32m1(&sumi_temp[4], 4);
                vint32m1_t sum_2 = __riscv_vle32_v_i32m1(&sumi_temp[8], 4);
                vint32m1_t sum_3 = __riscv_vle32_v_i32m1(&sumi_temp[12], 4);

                vfloat16mf2_t b_d = __riscv_vle16_v_f16mf2((_Float16 *)b_ptr[l].d, 4);
                vfloat32m1_t d_0 = __riscv_vfwmul_vf_f32m1(b_d, *(const _Float16 *)&a_ptr[l].d[0], 4);
                vfloat32m1_t d_1 = __riscv_vfwmul_vf_f32m1(b_d, *(const _Float16 *)&a_ptr[l].d[1], 4);
                vfloat32m1_t d_2 = __riscv_vfwmul_vf_f32m1(b_d, *(const _Float16 *)&a_ptr[l].d[2], 4);
                vfloat32m1_t d_3 = __riscv_vfwmul_vf_f32m1(b_d, *(const _Float16 *)&a_ptr[l].d[3], 4);

                sumf_0 = __riscv_vfmacc_vv_f32m1(sumf_0, d_0, __riscv_vfcvt_f_x_v_f32m1(sum_0, 4), 4);
                sumf_1 = __riscv_vfmacc_vv_f32m1(sumf_1, d_1, __riscv_vfcvt_f_x_v_f32m1(sum_1, 4), 4);
                sumf_2 = __riscv_vfmacc_vv_f32m1(sumf_2, d_2, __riscv_vfcvt_f_x_v_f32m1(sum_2, 4), 4);
                sumf_3 = __riscv_vfmacc_vv_f32m1(sumf_3, d_3, __riscv_vfcvt_f_x_v_f32m1(sum_3, 4), 4);
            }

            __riscv_vse32_v_f32m1(s + (y * 4 + 0) * bs + x * 4, sumf_0, 4);
            __riscv_vse32_v_f32m1(s + (y * 4 + 1) * bs + x * 4, sumf_1, 4);
            __riscv_vse32_v_f32m1(s + (y * 4 + 2) * bs + x * 4, sumf_2, 4);
            __riscv_vse32_v_f32m1(s + (y * 4 + 3) * bs + x * 4, sumf_3, 4);
        }
    }
    return;
#endif
    ggml_gemm_iq4_nl_4x16_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq4_nl_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

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

#if defined __riscv_v_intrinsic
    const vint8mf2_t values = __riscv_vle8_v_i8mf2(kvalues_iq4nl, 16);

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx16 * b_ptr = (const block_iq4_nlx16 *) vx + (x * nb);

            // 4x16 Accumulators
            vfloat32m2_t sumf_0 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_1 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_2 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
            vfloat32m2_t sumf_3 = __riscv_vfmv_v_f_f32m2(0.0f, 16);

            for (int l = 0; l < nb; l++) {
                // 4x16 integer accumulators
                vint32m2_t sumi_0 = __riscv_vmv_v_x_i32m2(0.0f, 16);
                vint32m2_t sumi_1 = __riscv_vmv_v_x_i32m2(0.0f, 16);
                vint32m2_t sumi_2 = __riscv_vmv_v_x_i32m2(0.0f, 16);
                vint32m2_t sumi_3 = __riscv_vmv_v_x_i32m2(0.0f, 16);

                // Accumulation loop.
                for (int i = 0; i < QK4_NL / 2; i++) {
                    // Load `b_ptr`.
                    const vuint8mf2_t b_0_packed = __riscv_vle8_v_u8mf2((const uint8_t *)&b_ptr[l].qs[i * 16], 16);
                    const vint8mf2_t b_0_lo = __riscv_vrgather_vv_i8mf2(values, __riscv_vand_vx_u8mf2(b_0_packed, 0xf, 16), 16);
                    const vint8mf2_t b_0_hi = __riscv_vrgather_vv_i8mf2(values, __riscv_vsrl_vx_u8mf2(b_0_packed, 4, 16), 16);
                    // const vint16m1_t b_0_lo_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_lo, 16);
                    // const vint16m1_t b_0_hi_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_hi, 16);

                    const vint16m1_t sumi_0_lo = __riscv_vwmul_vx_i16m1(b_0_lo, a_ptr[l].qs[i * 4], 16);
                    const vint16m1_t sumi_1_lo = __riscv_vwmul_vx_i16m1(b_0_lo, a_ptr[l].qs[i * 4 + 1], 16);
                    const vint16m1_t sumi_2_lo = __riscv_vwmul_vx_i16m1(b_0_lo, a_ptr[l].qs[i * 4 + 2], 16);
                    const vint16m1_t sumi_3_lo = __riscv_vwmul_vx_i16m1(b_0_lo, a_ptr[l].qs[i * 4 + 3], 16);

                    const vint16m1_t sumi_0_hi = __riscv_vwmul_vx_i16m1(b_0_hi, a_ptr[l].qs[64 + i * 4], 16);
                    const vint16m1_t sumi_1_hi = __riscv_vwmul_vx_i16m1(b_0_hi, a_ptr[l].qs[64 + i * 4 + 1], 16);
                    const vint16m1_t sumi_2_hi = __riscv_vwmul_vx_i16m1(b_0_hi, a_ptr[l].qs[64 + i * 4 + 2], 16);
                    const vint16m1_t sumi_3_hi = __riscv_vwmul_vx_i16m1(b_0_hi, a_ptr[l].qs[64 + i * 4 + 3], 16);

                    sumi_0 = __riscv_vadd_vv_i32m2(sumi_0, __riscv_vwadd_vv_i32m2(sumi_0_lo, sumi_0_hi, 16), 16);
                    sumi_1 = __riscv_vadd_vv_i32m2(sumi_1, __riscv_vwadd_vv_i32m2(sumi_1_lo, sumi_1_hi, 16), 16);
                    sumi_2 = __riscv_vadd_vv_i32m2(sumi_2, __riscv_vwadd_vv_i32m2(sumi_2_lo, sumi_2_hi, 16), 16);
                    sumi_3 = __riscv_vadd_vv_i32m2(sumi_3, __riscv_vwadd_vv_i32m2(sumi_3_lo, sumi_3_hi, 16), 16);
                }

                const vfloat16m1_t b_d = __riscv_vle16_v_f16m1((_Float16 *)b_ptr[l].d, 16);
                const vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(const _Float16 *)&a_ptr[l].d[0], 16);
                const vfloat32m2_t d_1 = __riscv_vfwmul_vf_f32m2(b_d, *(const _Float16 *)&a_ptr[l].d[1], 16);
                const vfloat32m2_t d_2 = __riscv_vfwmul_vf_f32m2(b_d, *(const _Float16 *)&a_ptr[l].d[2], 16);
                const vfloat32m2_t d_3 = __riscv_vfwmul_vf_f32m2(b_d, *(const _Float16 *)&a_ptr[l].d[3], 16);

                sumf_0 = __riscv_vfmacc_vv_f32m2(sumf_0, __riscv_vfcvt_f_x_v_f32m2(sumi_0, 16), d_0, 16);
                sumf_1 = __riscv_vfmacc_vv_f32m2(sumf_1, __riscv_vfcvt_f_x_v_f32m2(sumi_1, 16), d_1, 16);
                sumf_2 = __riscv_vfmacc_vv_f32m2(sumf_2, __riscv_vfcvt_f_x_v_f32m2(sumi_2, 16), d_2, 16);
                sumf_3 = __riscv_vfmacc_vv_f32m2(sumf_3, __riscv_vfcvt_f_x_v_f32m2(sumi_3, 16), d_3, 16);
            }

            __riscv_vse32_v_f32m2(s + (y * 4 + 0) * bs + x * 16, sumf_0, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 1) * bs + x * 16, sumf_1, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 2) * bs + x * 16, sumf_2, 16);
            __riscv_vse32_v_f32m2(s + (y * 4 + 3) * bs + x * 16, sumf_3, 16);
        }
    }
    return;
#endif
    ggml_gemm_iq4_nl_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

#if defined __riscv_v_intrinsic
    const vint8m4_t values = __riscv_vle8_v_i8m4(kvalues_iq4nl, 16);
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx8 * b_ptr = (const block_iq4_nlx8 *) vx + (x * nb);

        vfloat32m1_t sumf = __riscv_vfmv_v_f_f32m1(0.0, 8);
        for (int l = 0; l < nb; l++) {
            const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[0];
            const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[8];
            const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[16];
            const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[24];
            __asm__ __volatile__("" ::: "memory");

            // Broadcast `a_ptr` across 4 registers (8 bytes / register).
            const vint8m2_t a_0 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a0, 8));
            const vint8m2_t a_1 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a1, 8));
            const vint8m2_t a_2 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a2, 8));
            const vint8m2_t a_3 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a3, 8));

            // Load `b_ptr`.
            const vuint8m4_t b_0_packed = __riscv_vle8_v_u8m4((const uint8_t *)b_ptr[l].qs, QK4_NL * 4);
            const vint8m4_t b_0_lo = __riscv_vrgather_vv_i8m4(values, __riscv_vand_vx_u8m4(b_0_packed, 0xf, QK4_NL * 4), QK4_NL * 4);
            const vint8m4_t b_0_hi = __riscv_vrgather_vv_i8m4(values, __riscv_vsrl_vx_u8m4(b_0_packed, 4, QK4_NL * 4), QK4_NL * 4);

            // Create 4 segments from `b`.
            const vint8m2_t b_lo_0 = __riscv_vget_v_i8m4_i8m2(b_0_lo, 0);
            const vint8m2_t b_lo_1 = __riscv_vget_v_i8m4_i8m2(b_0_lo, 1);
            const vint8m2_t b_hi_0 = __riscv_vget_v_i8m4_i8m2(b_0_hi, 0);
            const vint8m2_t b_hi_1 = __riscv_vget_v_i8m4_i8m2(b_0_hi, 1);

            // Multiply and accumulate.
            const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(b_lo_0, a_0, QK4_NL * 2);
            const vint16m4_t sumi_lo_1 = __riscv_vwmul_vv_i16m4(b_lo_1, a_1, QK4_NL * 2);
            const vint16m4_t sumi_hi_0 = __riscv_vwmul_vv_i16m4(b_hi_0, a_2, QK4_NL * 2);
            const vint16m4_t sumi_hi_1 = __riscv_vwmul_vv_i16m4(b_hi_1, a_3, QK4_NL * 2);
            const vint32m8_t sumi_lo = __riscv_vwadd_vv_i32m8(sumi_lo_0, sumi_lo_1, QK4_NL * 2);
            const vint32m8_t sumi_hi = __riscv_vwadd_vv_i32m8(sumi_hi_0, sumi_hi_1, QK4_NL * 2);
            const vint32m8_t sumi = __riscv_vadd_vv_i32m8(sumi_lo, sumi_hi, QK4_NL * 2);

            // In-place reduction.
            const vuint64m8_t sumi_i32 = __riscv_vreinterpret_v_i64m8_u64m8(__riscv_vreinterpret_v_i32m8_i64m8(sumi));
            const vuint32m4_t sumi_h2_0 = __riscv_vnsrl_wx_u32m4(sumi_i32, 0, QK4_NL);
            const vuint32m4_t sumi_h2_1 = __riscv_vnsrl_wx_u32m4(sumi_i32, 32, QK4_NL);
            const vuint32m4_t sumi_h2 = __riscv_vadd_vv_u32m4(sumi_h2_0, sumi_h2_1, QK4_NL);
            const vuint64m4_t sumi_h2_i32 = __riscv_vreinterpret_v_u32m4_u64m4(sumi_h2);
            const vuint32m2_t sumi_h4_0 = __riscv_vnsrl_wx_u32m2(sumi_h2_i32, 0, QK4_NL / 2);
            const vuint32m2_t sumi_h4_1 = __riscv_vnsrl_wx_u32m2(sumi_h2_i32, 32, QK4_NL / 2);
            const vuint32m2_t sumi_h4 = __riscv_vadd_vv_u32m2(sumi_h4_0, sumi_h4_1, QK4_NL / 2);
            const vuint64m2_t sumi_h4_i32 = __riscv_vreinterpret_v_u32m2_u64m2(sumi_h4);
            const vint32m1_t sumi_h8_0 = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(sumi_h4_i32, 0, QK4_NL / 4));
            const vint32m1_t sumi_h8_1 = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(sumi_h4_i32, 32, QK4_NL / 4));
            const vint32m1_t sumi_h8 = __riscv_vadd_vv_i32m1(sumi_h8_0, sumi_h8_1, QK4_NL / 4);
            const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, QK4_NL / 4);

            // Multiply with scales.
            const vfloat16mf2_t b_d = __riscv_vle16_v_f16mf2((const _Float16 *)b_ptr[l].d, 8);
            const vfloat32m1_t d_0 = __riscv_vfwmul_vf_f32m1(b_d, *(const _Float16*)&a_ptr[l].d, 8);
            sumf = __riscv_vfmacc_vv_f32m1(sumf, facc, d_0, QK4_NL / 4);
        }
        __riscv_vse32_v_f32m1(s + x * ncols_interleaved, sumf, QK4_NL / 4);
    }
    return;

#endif
    ggml_gemv_iq4_nl_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
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

void ggml_gemm_iq4_nl_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

#if defined __riscv_v_intrinsic
    const vint8m2_t values = __riscv_vle8_v_i8m2(kvalues_iq4nl, 16);
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

            // 4x4 accumulators.
            vfloat32mf2_t sumf0 = __riscv_vfmv_v_f_f32mf2(0.0, 4);
            vfloat32mf2_t sumf1 = __riscv_vfmv_v_f_f32mf2(0.0, 4);
            vfloat32mf2_t sumf2 = __riscv_vfmv_v_f_f32mf2(0.0, 4);
            vfloat32mf2_t sumf3 = __riscv_vfmv_v_f_f32mf2(0.0, 4);

            for (int l = 0; l < nb; l++) {
                // Load `b_ptr`.
                const vuint8m2_t b_0_packed = __riscv_vle8_v_u8m2((const uint8_t *)b_ptr[l].qs, QK4_NL * 2);
                const vint8m2_t b_0_lo = __riscv_vrgather_vv_i8m2(values, __riscv_vand_vx_u8m2(b_0_packed, 0xf, QK4_NL * 2), QK4_NL * 2);
                const vint8m2_t b_0_hi = __riscv_vrgather_vv_i8m2(values, __riscv_vsrl_vx_u8m2(b_0_packed, 4, QK4_NL * 2), QK4_NL * 2);

                // Create 4 segments from `b`.
                const vint8m1_t b_lo_0 = __riscv_vget_v_i8m2_i8m1(b_0_lo, 0);
                const vint8m1_t b_lo_1 = __riscv_vget_v_i8m2_i8m1(b_0_lo, 1);
                const vint8m1_t b_hi_0 = __riscv_vget_v_i8m2_i8m1(b_0_hi, 0);
                const vint8m1_t b_hi_1 = __riscv_vget_v_i8m2_i8m1(b_0_hi, 1);

                // Load scales for `b`.
                const vfloat16mf4_t b_d = __riscv_vle16_v_f16mf4((const _Float16 *)b_ptr[l].d, 4);

                // Load first 8 bytes of `a`.
                {
                    const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[0];
                    const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[32];
                    const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[64];
                    const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[96];
                    __asm__ __volatile__("" ::: "memory");

                    // Broadcast `a_ptr` across 4 registers (8 bytes / register).
                    const vint8m1_t a_0 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a0, 4));
                    const vint8m1_t a_1 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a1, 4));
                    const vint8m1_t a_2 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a2, 4));
                    const vint8m1_t a_3 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a3, 4));

                    // Multiply and accumulate.
                    const vint16m2_t sumi_lo_0 = __riscv_vwmul_vv_i16m2(b_lo_0, a_0, QK4_NL);
                    const vint16m2_t sumi_lo_1 = __riscv_vwmul_vv_i16m2(b_lo_1, a_1, QK4_NL);
                    const vint16m2_t sumi_hi_0 = __riscv_vwmul_vv_i16m2(b_hi_0, a_2, QK4_NL);
                    const vint16m2_t sumi_hi_1 = __riscv_vwmul_vv_i16m2(b_hi_1, a_3, QK4_NL);
                    const vint32m4_t sumi_lo = __riscv_vwadd_vv_i32m4(sumi_lo_0, sumi_lo_1, QK4_NL);
                    const vint32m4_t sumi_hi = __riscv_vwadd_vv_i32m4(sumi_hi_0, sumi_hi_1, QK4_NL);
                    const vint32m4_t sumi = __riscv_vadd_vv_i32m4(sumi_lo, sumi_hi, QK4_NL);

                    // In-place reduction.
                    const vuint64m4_t sumi_i32 = __riscv_vreinterpret_v_i64m4_u64m4(__riscv_vreinterpret_v_i32m4_i64m4(sumi));
                    const vuint32m2_t sumi_h2_0 = __riscv_vnsrl_wx_u32m2(sumi_i32, 0, QK4_NL / 2);
                    const vuint32m2_t sumi_h2_1 = __riscv_vnsrl_wx_u32m2(sumi_i32, 32, QK4_NL / 2);
                    const vuint32m2_t sumi_h2 = __riscv_vadd_vv_u32m2(sumi_h2_0, sumi_h2_1, QK4_NL/ 2);
                    const vuint64m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u32m2_u64m2(sumi_h2);
                    const vuint32m1_t sumi_h4_0 = __riscv_vnsrl_wx_u32m1(sumi_h2_i32, 0, QK4_NL / 4);
                    const vuint32m1_t sumi_h4_1 = __riscv_vnsrl_wx_u32m1(sumi_h2_i32, 32, QK4_NL / 4);
                    const vuint32m1_t sumi_h4 = __riscv_vadd_vv_u32m1(sumi_h4_0, sumi_h4_1, QK4_NL / 4);
                    const vuint64m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u32m1_u64m1(sumi_h4);
                    const vint32mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vnsrl_wx_u32mf2(sumi_h4_i32, 0, QK4_NL / 8));
                    const vint32mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vnsrl_wx_u32mf2(sumi_h4_i32, 32, QK4_NL / 8));
                    const vint32mf2_t sumi_h8 = __riscv_vadd_vv_i32mf2(sumi_h8_0, sumi_h8_1, QK4_NL / 8);
                    const vfloat32mf2_t facc = __riscv_vfcvt_f_x_v_f32mf2(sumi_h8, QK4_NL / 8);

                    // Multiply with scales.
                    const vfloat32mf2_t d_0 = __riscv_vfwmul_vf_f32mf2(b_d, *(const _Float16*)&a_ptr[l].d[0], 4);
                    sumf0 = __riscv_vfmacc_vv_f32mf2(sumf0, facc, d_0, QK4_NL / 8);
                }

                // Load second 8 bytes of `a`.
                {
                    const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[8];
                    const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[40];
                    const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[72];
                    const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[104];
                    __asm__ __volatile__("" ::: "memory");

                    // Broadcast `a_ptr` across 4 registers (8 bytes / register).
                    const vint8m1_t a_0 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a0, 4));
                    const vint8m1_t a_1 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a1, 4));
                    const vint8m1_t a_2 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a2, 4));
                    const vint8m1_t a_3 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a3, 4));

                    // Multiply and accumulate.
                    const vint16m2_t sumi_lo_0 = __riscv_vwmul_vv_i16m2(b_lo_0, a_0, QK4_NL);
                    const vint16m2_t sumi_lo_1 = __riscv_vwmul_vv_i16m2(b_lo_1, a_1, QK4_NL);
                    const vint16m2_t sumi_hi_0 = __riscv_vwmul_vv_i16m2(b_hi_0, a_2, QK4_NL);
                    const vint16m2_t sumi_hi_1 = __riscv_vwmul_vv_i16m2(b_hi_1, a_3, QK4_NL);
                    const vint32m4_t sumi_lo = __riscv_vwadd_vv_i32m4(sumi_lo_0, sumi_lo_1, QK4_NL);
                    const vint32m4_t sumi_hi = __riscv_vwadd_vv_i32m4(sumi_hi_0, sumi_hi_1, QK4_NL);
                    const vint32m4_t sumi = __riscv_vadd_vv_i32m4(sumi_lo, sumi_hi, QK4_NL);

                    // In-place reduction.
                    const vuint64m4_t sumi_i32 = __riscv_vreinterpret_v_i64m4_u64m4(__riscv_vreinterpret_v_i32m4_i64m4(sumi));
                    const vuint32m2_t sumi_h2_0 = __riscv_vnsrl_wx_u32m2(sumi_i32, 0, QK4_NL / 2);
                    const vuint32m2_t sumi_h2_1 = __riscv_vnsrl_wx_u32m2(sumi_i32, 32, QK4_NL / 2);
                    const vuint32m2_t sumi_h2 = __riscv_vadd_vv_u32m2(sumi_h2_0, sumi_h2_1, QK4_NL/ 2);
                    const vuint64m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u32m2_u64m2(sumi_h2);
                    const vuint32m1_t sumi_h4_0 = __riscv_vnsrl_wx_u32m1(sumi_h2_i32, 0, QK4_NL / 4);
                    const vuint32m1_t sumi_h4_1 = __riscv_vnsrl_wx_u32m1(sumi_h2_i32, 32, QK4_NL / 4);
                    const vuint32m1_t sumi_h4 = __riscv_vadd_vv_u32m1(sumi_h4_0, sumi_h4_1, QK4_NL / 4);
                    const vuint64m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u32m1_u64m1(sumi_h4);
                    const vint32mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vnsrl_wx_u32mf2(sumi_h4_i32, 0, QK4_NL / 8));
                    const vint32mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vnsrl_wx_u32mf2(sumi_h4_i32, 32, QK4_NL / 8));
                    const vint32mf2_t sumi_h8 = __riscv_vadd_vv_i32mf2(sumi_h8_0, sumi_h8_1, QK4_NL / 8);
                    const vfloat32mf2_t facc = __riscv_vfcvt_f_x_v_f32mf2(sumi_h8, QK4_NL / 8);

                    // Multiply with scales.
                    const vfloat32mf2_t d_0 = __riscv_vfwmul_vf_f32mf2(b_d, *(const _Float16*)&a_ptr[l].d[1], 4);
                    sumf1 = __riscv_vfmacc_vv_f32mf2(sumf1, facc, d_0, QK4_NL / 8);
                }

                // Load third 8 bytes of `a`.
                {
                    const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[16];
                    const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[48];
                    const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[80];
                    const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[112];
                    __asm__ __volatile__("" ::: "memory");

                    // Broadcast `a_ptr` across 4 registers (8 bytes / register).
                    const vint8m1_t a_0 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a0, 4));
                    const vint8m1_t a_1 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a1, 4));
                    const vint8m1_t a_2 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a2, 4));
                    const vint8m1_t a_3 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a3, 4));

                    // Multiply and accumulate.
                    const vint16m2_t sumi_lo_0 = __riscv_vwmul_vv_i16m2(b_lo_0, a_0, QK4_NL);
                    const vint16m2_t sumi_lo_1 = __riscv_vwmul_vv_i16m2(b_lo_1, a_1, QK4_NL);
                    const vint16m2_t sumi_hi_0 = __riscv_vwmul_vv_i16m2(b_hi_0, a_2, QK4_NL);
                    const vint16m2_t sumi_hi_1 = __riscv_vwmul_vv_i16m2(b_hi_1, a_3, QK4_NL);
                    const vint32m4_t sumi_lo = __riscv_vwadd_vv_i32m4(sumi_lo_0, sumi_lo_1, QK4_NL);
                    const vint32m4_t sumi_hi = __riscv_vwadd_vv_i32m4(sumi_hi_0, sumi_hi_1, QK4_NL);
                    const vint32m4_t sumi = __riscv_vadd_vv_i32m4(sumi_lo, sumi_hi, QK4_NL);

                    // In-place reduction.
                    const vuint64m4_t sumi_i32 = __riscv_vreinterpret_v_i64m4_u64m4(__riscv_vreinterpret_v_i32m4_i64m4(sumi));
                    const vuint32m2_t sumi_h2_0 = __riscv_vnsrl_wx_u32m2(sumi_i32, 0, QK4_NL / 2);
                    const vuint32m2_t sumi_h2_1 = __riscv_vnsrl_wx_u32m2(sumi_i32, 32, QK4_NL / 2);
                    const vuint32m2_t sumi_h2 = __riscv_vadd_vv_u32m2(sumi_h2_0, sumi_h2_1, QK4_NL/ 2);
                    const vuint64m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u32m2_u64m2(sumi_h2);
                    const vuint32m1_t sumi_h4_0 = __riscv_vnsrl_wx_u32m1(sumi_h2_i32, 0, QK4_NL / 4);
                    const vuint32m1_t sumi_h4_1 = __riscv_vnsrl_wx_u32m1(sumi_h2_i32, 32, QK4_NL / 4);
                    const vuint32m1_t sumi_h4 = __riscv_vadd_vv_u32m1(sumi_h4_0, sumi_h4_1, QK4_NL / 4);
                    const vuint64m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u32m1_u64m1(sumi_h4);
                    const vint32mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vnsrl_wx_u32mf2(sumi_h4_i32, 0, QK4_NL / 8));
                    const vint32mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vnsrl_wx_u32mf2(sumi_h4_i32, 32, QK4_NL / 8));
                    const vint32mf2_t sumi_h8 = __riscv_vadd_vv_i32mf2(sumi_h8_0, sumi_h8_1, QK4_NL / 8);
                    const vfloat32mf2_t facc = __riscv_vfcvt_f_x_v_f32mf2(sumi_h8, QK4_NL / 8);

                    // Multiply with scales.
                    const vfloat32mf2_t d_0 = __riscv_vfwmul_vf_f32mf2(b_d, *(const _Float16*)&a_ptr[l].d[2], 4);
                    sumf2 = __riscv_vfmacc_vv_f32mf2(sumf2, facc, d_0, QK4_NL / 8);
                }

                // Load fourth 8 bytes of `a`.
                {
                    const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[24];
                    const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[56];
                    const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[88];
                    const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[120];
                    __asm__ __volatile__("" ::: "memory");

                    // Broadcast `a_ptr` across 4 registers (8 bytes / register).
                    const vint8m1_t a_0 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a0, 4));
                    const vint8m1_t a_1 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a1, 4));
                    const vint8m1_t a_2 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a2, 4));
                    const vint8m1_t a_3 = __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vmv_v_x_i64m1(a3, 4));

                    // Multiply and accumulate.
                    const vint16m2_t sumi_lo_0 = __riscv_vwmul_vv_i16m2(b_lo_0, a_0, QK4_NL);
                    const vint16m2_t sumi_lo_1 = __riscv_vwmul_vv_i16m2(b_lo_1, a_1, QK4_NL);
                    const vint16m2_t sumi_hi_0 = __riscv_vwmul_vv_i16m2(b_hi_0, a_2, QK4_NL);
                    const vint16m2_t sumi_hi_1 = __riscv_vwmul_vv_i16m2(b_hi_1, a_3, QK4_NL);
                    const vint32m4_t sumi_lo = __riscv_vwadd_vv_i32m4(sumi_lo_0, sumi_lo_1, QK4_NL);
                    const vint32m4_t sumi_hi = __riscv_vwadd_vv_i32m4(sumi_hi_0, sumi_hi_1, QK4_NL);
                    const vint32m4_t sumi = __riscv_vadd_vv_i32m4(sumi_lo, sumi_hi, QK4_NL);

                    // In-place reduction.
                    const vuint64m4_t sumi_i32 = __riscv_vreinterpret_v_i64m4_u64m4(__riscv_vreinterpret_v_i32m4_i64m4(sumi));
                    const vuint32m2_t sumi_h2_0 = __riscv_vnsrl_wx_u32m2(sumi_i32, 0, QK4_NL / 2);
                    const vuint32m2_t sumi_h2_1 = __riscv_vnsrl_wx_u32m2(sumi_i32, 32, QK4_NL / 2);
                    const vuint32m2_t sumi_h2 = __riscv_vadd_vv_u32m2(sumi_h2_0, sumi_h2_1, QK4_NL/ 2);
                    const vuint64m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u32m2_u64m2(sumi_h2);
                    const vuint32m1_t sumi_h4_0 = __riscv_vnsrl_wx_u32m1(sumi_h2_i32, 0, QK4_NL / 4);
                    const vuint32m1_t sumi_h4_1 = __riscv_vnsrl_wx_u32m1(sumi_h2_i32, 32, QK4_NL / 4);
                    const vuint32m1_t sumi_h4 = __riscv_vadd_vv_u32m1(sumi_h4_0, sumi_h4_1, QK4_NL / 4);
                    const vuint64m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u32m1_u64m1(sumi_h4);
                    const vint32mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vnsrl_wx_u32mf2(sumi_h4_i32, 0, QK4_NL / 8));
                    const vint32mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vnsrl_wx_u32mf2(sumi_h4_i32, 32, QK4_NL / 8));
                    const vint32mf2_t sumi_h8 = __riscv_vadd_vv_i32mf2(sumi_h8_0, sumi_h8_1, QK4_NL / 8);
                    const vfloat32mf2_t facc = __riscv_vfcvt_f_x_v_f32mf2(sumi_h8, QK4_NL / 8);

                    // Multiply with scales.
                    const vfloat32mf2_t d_0 = __riscv_vfwmul_vf_f32mf2(b_d, *(const _Float16*)&a_ptr[l].d[3], 4);
                    sumf3 = __riscv_vfmacc_vv_f32mf2(sumf3, facc, d_0, QK4_NL / 8);
                }
            }

            __riscv_vse32_v_f32mf2(&s[(y * 4 + 0) * bs + x * ncols_interleaved], sumf0, 8);
            __riscv_vse32_v_f32mf2(&s[(y * 4 + 1) * bs + x * ncols_interleaved], sumf1, 8);
            __riscv_vse32_v_f32mf2(&s[(y * 4 + 2) * bs + x * ncols_interleaved], sumf2, 8);
            __riscv_vse32_v_f32mf2(&s[(y * 4 + 3) * bs + x * ncols_interleaved], sumf3, 8);
        }
    }
    return;

#endif
    ggml_gemm_iq4_nl_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

#if defined __riscv_v_intrinsic
    const vint8m4_t values = __riscv_vle8_v_i8m4(kvalues_iq4nl, 16);
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx8 * b_ptr = (const block_iq4_nlx8 *) vx + (x * nb);

            // 4x8 accumulators.
            vfloat32m1_t sumf0 = __riscv_vfmv_v_f_f32m1(0.0, 8);
            vfloat32m1_t sumf1 = __riscv_vfmv_v_f_f32m1(0.0, 8);
            vfloat32m1_t sumf2 = __riscv_vfmv_v_f_f32m1(0.0, 8);
            vfloat32m1_t sumf3 = __riscv_vfmv_v_f_f32m1(0.0, 8);

            for (int l = 0; l < nb; l++) {
                // Load `b_ptr`.
                const vuint8m4_t b_0_packed = __riscv_vle8_v_u8m4((const uint8_t *)b_ptr[l].qs, QK4_NL * 4);
                const vint8m4_t b_0_lo = __riscv_vrgather_vv_i8m4(values, __riscv_vand_vx_u8m4(b_0_packed, 0xf, QK4_NL * 4), QK4_NL * 4);
                const vint8m4_t b_0_hi = __riscv_vrgather_vv_i8m4(values, __riscv_vsrl_vx_u8m4(b_0_packed, 4, QK4_NL * 4), QK4_NL * 4);

                // Create 4 segments from `b`.
                const vint8m2_t b_lo_0 = __riscv_vget_v_i8m4_i8m2(b_0_lo, 0);
                const vint8m2_t b_lo_1 = __riscv_vget_v_i8m4_i8m2(b_0_lo, 1);
                const vint8m2_t b_hi_0 = __riscv_vget_v_i8m4_i8m2(b_0_hi, 0);
                const vint8m2_t b_hi_1 = __riscv_vget_v_i8m4_i8m2(b_0_hi, 1);

                // Load scales for `b`.
                const vfloat16mf2_t b_d = __riscv_vle16_v_f16mf2((const _Float16 *)b_ptr[l].d, 8);

                {
                    // Load first 8 bytes of `a`.
                    const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[0];
                    const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[32];
                    const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[64];
                    const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[96];
                    __asm__ __volatile__("" ::: "memory");

                    // Broadcast `a_ptr` across 4 registers (8 bytes / register).
                    const vint8m2_t a_0 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a0, 8));
                    const vint8m2_t a_1 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a1, 8));
                    const vint8m2_t a_2 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a2, 8));
                    const vint8m2_t a_3 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a3, 8));

                    // Multiply and accumulate.
                    const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(b_lo_0, a_0, QK4_NL * 2);
                    const vint16m4_t sumi_lo_1 = __riscv_vwmul_vv_i16m4(b_lo_1, a_1, QK4_NL * 2);
                    const vint16m4_t sumi_hi_0 = __riscv_vwmul_vv_i16m4(b_hi_0, a_2, QK4_NL * 2);
                    const vint16m4_t sumi_hi_1 = __riscv_vwmul_vv_i16m4(b_hi_1, a_3, QK4_NL * 2);
                    const vint32m8_t sumi_lo = __riscv_vwadd_vv_i32m8(sumi_lo_0, sumi_lo_1, QK4_NL * 2);
                    const vint32m8_t sumi_hi = __riscv_vwadd_vv_i32m8(sumi_hi_0, sumi_hi_1, QK4_NL * 2);
                    const vint32m8_t sumi = __riscv_vadd_vv_i32m8(sumi_lo, sumi_hi, QK4_NL * 2);

                    // In-place reduction.
                    const vuint64m8_t sumi_i32 = __riscv_vreinterpret_v_i64m8_u64m8(__riscv_vreinterpret_v_i32m8_i64m8(sumi));
                    const vuint32m4_t sumi_h2_0 = __riscv_vnsrl_wx_u32m4(sumi_i32, 0, QK4_NL);
                    const vuint32m4_t sumi_h2_1 = __riscv_vnsrl_wx_u32m4(sumi_i32, 32, QK4_NL);
                    const vuint32m4_t sumi_h2 = __riscv_vadd_vv_u32m4(sumi_h2_0, sumi_h2_1, QK4_NL);
                    const vuint64m4_t sumi_h2_i32 = __riscv_vreinterpret_v_u32m4_u64m4(sumi_h2);
                    const vuint32m2_t sumi_h4_0 = __riscv_vnsrl_wx_u32m2(sumi_h2_i32, 0, QK4_NL / 2);
                    const vuint32m2_t sumi_h4_1 = __riscv_vnsrl_wx_u32m2(sumi_h2_i32, 32, QK4_NL / 2);
                    const vuint32m2_t sumi_h4 = __riscv_vadd_vv_u32m2(sumi_h4_0, sumi_h4_1, QK4_NL / 2);
                    const vuint64m2_t sumi_h4_i32 = __riscv_vreinterpret_v_u32m2_u64m2(sumi_h4);
                    const vint32m1_t sumi_h8_0 = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(sumi_h4_i32, 0, QK4_NL / 4));
                    const vint32m1_t sumi_h8_1 = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(sumi_h4_i32, 32, QK4_NL / 4));
                    const vint32m1_t sumi_h8 = __riscv_vadd_vv_i32m1(sumi_h8_0, sumi_h8_1, QK4_NL / 4);
                    const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, QK4_NL / 4);

                    // Multiply with scales.
                    const vfloat32m1_t d_0 = __riscv_vfwmul_vf_f32m1(b_d, *(const _Float16*)&a_ptr[l].d[0], 8);
                    sumf0 = __riscv_vfmacc_vv_f32m1(sumf0, facc, d_0, QK4_NL / 4);
                }

                // Load second 8 bytes of `a`.
                {
                    const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[8];
                    const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[40];
                    const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[72];
                    const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[104];
                    __asm__ __volatile__("" ::: "memory");

                    // Broadcast `a_ptr` across 4 registers (8 bytes / register).
                    const vint8m2_t a_0 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a0, 8));
                    const vint8m2_t a_1 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a1, 8));
                    const vint8m2_t a_2 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a2, 8));
                    const vint8m2_t a_3 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a3, 8));

                    // Multiply and accumulate.
                    const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(b_lo_0, a_0, QK4_NL * 2);
                    const vint16m4_t sumi_lo_1 = __riscv_vwmul_vv_i16m4(b_lo_1, a_1, QK4_NL * 2);
                    const vint16m4_t sumi_hi_0 = __riscv_vwmul_vv_i16m4(b_hi_0, a_2, QK4_NL * 2);
                    const vint16m4_t sumi_hi_1 = __riscv_vwmul_vv_i16m4(b_hi_1, a_3, QK4_NL * 2);
                    const vint32m8_t sumi_lo = __riscv_vwadd_vv_i32m8(sumi_lo_0, sumi_lo_1, QK4_NL * 2);
                    const vint32m8_t sumi_hi = __riscv_vwadd_vv_i32m8(sumi_hi_0, sumi_hi_1, QK4_NL * 2);
                    const vint32m8_t sumi = __riscv_vadd_vv_i32m8(sumi_lo, sumi_hi, QK4_NL * 2);

                    // In-place reduction.
                    const vuint64m8_t sumi_i32 = __riscv_vreinterpret_v_i64m8_u64m8(__riscv_vreinterpret_v_i32m8_i64m8(sumi));
                    const vuint32m4_t sumi_h2_0 = __riscv_vnsrl_wx_u32m4(sumi_i32, 0, QK4_NL);
                    const vuint32m4_t sumi_h2_1 = __riscv_vnsrl_wx_u32m4(sumi_i32, 32, QK4_NL);
                    const vuint32m4_t sumi_h2 = __riscv_vadd_vv_u32m4(sumi_h2_0, sumi_h2_1, QK4_NL);
                    const vuint64m4_t sumi_h2_i32 = __riscv_vreinterpret_v_u32m4_u64m4(sumi_h2);
                    const vuint32m2_t sumi_h4_0 = __riscv_vnsrl_wx_u32m2(sumi_h2_i32, 0, QK4_NL / 2);
                    const vuint32m2_t sumi_h4_1 = __riscv_vnsrl_wx_u32m2(sumi_h2_i32, 32, QK4_NL / 2);
                    const vuint32m2_t sumi_h4 = __riscv_vadd_vv_u32m2(sumi_h4_0, sumi_h4_1, QK4_NL / 2);
                    const vuint64m2_t sumi_h4_i32 = __riscv_vreinterpret_v_u32m2_u64m2(sumi_h4);
                    const vint32m1_t sumi_h8_0 = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(sumi_h4_i32, 0, QK4_NL / 4));
                    const vint32m1_t sumi_h8_1 = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(sumi_h4_i32, 32, QK4_NL / 4));
                    const vint32m1_t sumi_h8 = __riscv_vadd_vv_i32m1(sumi_h8_0, sumi_h8_1, QK4_NL / 4);
                    const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, QK4_NL / 4);

                    // Multiply with scales.
                    const vfloat32m1_t d_0 = __riscv_vfwmul_vf_f32m1(b_d, *(const _Float16*)&a_ptr[l].d[1], 8);
                    sumf1 = __riscv_vfmacc_vv_f32m1(sumf1, facc, d_0, QK4_NL / 4);
                }

                // Load third 8 bytes of `a`.
                {
                    const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[16];
                    const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[48];
                    const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[80];
                    const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[112];
                    __asm__ __volatile__("" ::: "memory");

                    // Broadcast `a_ptr` across 4 registers (8 bytes / register).
                    const vint8m2_t a_0 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a0, 8));
                    const vint8m2_t a_1 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a1, 8));
                    const vint8m2_t a_2 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a2, 8));
                    const vint8m2_t a_3 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a3, 8));

                    // Multiply and accumulate.
                    const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(b_lo_0, a_0, QK4_NL * 2);
                    const vint16m4_t sumi_lo_1 = __riscv_vwmul_vv_i16m4(b_lo_1, a_1, QK4_NL * 2);
                    const vint16m4_t sumi_hi_0 = __riscv_vwmul_vv_i16m4(b_hi_0, a_2, QK4_NL * 2);
                    const vint16m4_t sumi_hi_1 = __riscv_vwmul_vv_i16m4(b_hi_1, a_3, QK4_NL * 2);
                    const vint32m8_t sumi_lo = __riscv_vwadd_vv_i32m8(sumi_lo_0, sumi_lo_1, QK4_NL * 2);
                    const vint32m8_t sumi_hi = __riscv_vwadd_vv_i32m8(sumi_hi_0, sumi_hi_1, QK4_NL * 2);
                    const vint32m8_t sumi = __riscv_vadd_vv_i32m8(sumi_lo, sumi_hi, QK4_NL * 2);

                    // In-place reduction.
                    const vuint64m8_t sumi_i32 = __riscv_vreinterpret_v_i64m8_u64m8(__riscv_vreinterpret_v_i32m8_i64m8(sumi));
                    const vuint32m4_t sumi_h2_0 = __riscv_vnsrl_wx_u32m4(sumi_i32, 0, QK4_NL);
                    const vuint32m4_t sumi_h2_1 = __riscv_vnsrl_wx_u32m4(sumi_i32, 32, QK4_NL);
                    const vuint32m4_t sumi_h2 = __riscv_vadd_vv_u32m4(sumi_h2_0, sumi_h2_1, QK4_NL);
                    const vuint64m4_t sumi_h2_i32 = __riscv_vreinterpret_v_u32m4_u64m4(sumi_h2);
                    const vuint32m2_t sumi_h4_0 = __riscv_vnsrl_wx_u32m2(sumi_h2_i32, 0, QK4_NL / 2);
                    const vuint32m2_t sumi_h4_1 = __riscv_vnsrl_wx_u32m2(sumi_h2_i32, 32, QK4_NL / 2);
                    const vuint32m2_t sumi_h4 = __riscv_vadd_vv_u32m2(sumi_h4_0, sumi_h4_1, QK4_NL / 2);
                    const vuint64m2_t sumi_h4_i32 = __riscv_vreinterpret_v_u32m2_u64m2(sumi_h4);
                    const vint32m1_t sumi_h8_0 = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(sumi_h4_i32, 0, QK4_NL / 4));
                    const vint32m1_t sumi_h8_1 = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(sumi_h4_i32, 32, QK4_NL / 4));
                    const vint32m1_t sumi_h8 = __riscv_vadd_vv_i32m1(sumi_h8_0, sumi_h8_1, QK4_NL / 4);
                    const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, QK4_NL / 4);

                    // Multiply with scales.
                    const vfloat32m1_t d_0 = __riscv_vfwmul_vf_f32m1(b_d, *(const _Float16*)&a_ptr[l].d[2], 8);
                    sumf2 = __riscv_vfmacc_vv_f32m1(sumf2, facc, d_0, QK4_NL / 4);
                }

                {
                    // Load fourth 8 bytes of `a`.
                    const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[24];
                    const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[56];
                    const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[88];
                    const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[120];
                    __asm__ __volatile__("" ::: "memory");

                    // Broadcast `a_ptr` across 4 registers (8 bytes / register).
                    const vint8m2_t a_0 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a0, 8));
                    const vint8m2_t a_1 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a1, 8));
                    const vint8m2_t a_2 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a2, 8));
                    const vint8m2_t a_3 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a3, 8));

                    // Multiply and accumulate.
                    const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(b_lo_0, a_0, QK4_NL * 2);
                    const vint16m4_t sumi_lo_1 = __riscv_vwmul_vv_i16m4(b_lo_1, a_1, QK4_NL * 2);
                    const vint16m4_t sumi_hi_0 = __riscv_vwmul_vv_i16m4(b_hi_0, a_2, QK4_NL * 2);
                    const vint16m4_t sumi_hi_1 = __riscv_vwmul_vv_i16m4(b_hi_1, a_3, QK4_NL * 2);
                    const vint32m8_t sumi_lo = __riscv_vwadd_vv_i32m8(sumi_lo_0, sumi_lo_1, QK4_NL * 2);
                    const vint32m8_t sumi_hi = __riscv_vwadd_vv_i32m8(sumi_hi_0, sumi_hi_1, QK4_NL * 2);
                    const vint32m8_t sumi = __riscv_vadd_vv_i32m8(sumi_lo, sumi_hi, QK4_NL * 2);

                    // In-place reduction.
                    const vuint64m8_t sumi_i32 = __riscv_vreinterpret_v_i64m8_u64m8(__riscv_vreinterpret_v_i32m8_i64m8(sumi));
                    const vuint32m4_t sumi_h2_0 = __riscv_vnsrl_wx_u32m4(sumi_i32, 0, QK4_NL);
                    const vuint32m4_t sumi_h2_1 = __riscv_vnsrl_wx_u32m4(sumi_i32, 32, QK4_NL);
                    const vuint32m4_t sumi_h2 = __riscv_vadd_vv_u32m4(sumi_h2_0, sumi_h2_1, QK4_NL);
                    const vuint64m4_t sumi_h2_i32 = __riscv_vreinterpret_v_u32m4_u64m4(sumi_h2);
                    const vuint32m2_t sumi_h4_0 = __riscv_vnsrl_wx_u32m2(sumi_h2_i32, 0, QK4_NL / 2);
                    const vuint32m2_t sumi_h4_1 = __riscv_vnsrl_wx_u32m2(sumi_h2_i32, 32, QK4_NL / 2);
                    const vuint32m2_t sumi_h4 = __riscv_vadd_vv_u32m2(sumi_h4_0, sumi_h4_1, QK4_NL / 2);
                    const vuint64m2_t sumi_h4_i32 = __riscv_vreinterpret_v_u32m2_u64m2(sumi_h4);
                    const vint32m1_t sumi_h8_0 = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(sumi_h4_i32, 0, QK4_NL / 4));
                    const vint32m1_t sumi_h8_1 = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vnsrl_wx_u32m1(sumi_h4_i32, 32, QK4_NL / 4));
                    const vint32m1_t sumi_h8 = __riscv_vadd_vv_i32m1(sumi_h8_0, sumi_h8_1, QK4_NL / 4);
                    const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, QK4_NL / 4);

                    // Multiply with scales.
                    const vfloat32m1_t d_0 = __riscv_vfwmul_vf_f32m1(b_d, *(const _Float16*)&a_ptr[l].d[3], 8);
                    sumf3 = __riscv_vfmacc_vv_f32m1(sumf3, facc, d_0, QK4_NL / 4);
                }
            }

            __riscv_vse32_v_f32m1(&s[(y * 4 + 0) * bs + x * ncols_interleaved], sumf0, 8);
            __riscv_vse32_v_f32m1(&s[(y * 4 + 1) * bs + x * ncols_interleaved], sumf1, 8);
            __riscv_vse32_v_f32m1(&s[(y * 4 + 2) * bs + x * ncols_interleaved], sumf2, 8);
            __riscv_vse32_v_f32m1(&s[(y * 4 + 3) * bs + x * ncols_interleaved], sumf3, 8);
        }
    }
    return;

#endif
    ggml_gemm_iq4_nl_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
