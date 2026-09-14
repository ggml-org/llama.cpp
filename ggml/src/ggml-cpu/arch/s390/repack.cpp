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

#define GGML_CPU_CLANG_WORKAROUND
#include "../../repack.h"

#define UNUSED GGML_UNUSED

void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__VXE__) || defined(__VXE2__)
    float32x4_t v_src[4][8];
    float       id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t v_asrc[8];
        float32x4_t v_amax[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) v_src[row_iter][j] = vec_xl(0, x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) v_asrc[j] = vec_abs(v_src[row_iter][j]);

            for (int j = 0; j < 4; j++) v_amax[2 * j] = vec_max(v_asrc[2 * j], v_asrc[2 * j + 1]);
            for (int j = 0; j < 2; j++) v_amax[4 * j] = vec_max(v_amax[4 * j], v_amax[4 * j + 2]);
            for (int j = 0; j < 1; j++) v_amax[8 * j] = vec_max(v_amax[8 * j], v_amax[8 * j + 4]);

            const float amax = MAX(MAX(vec_extract(v_amax[0], 0), vec_extract(v_amax[0], 1)),
                                   MAX(vec_extract(v_amax[0], 2), vec_extract(v_amax[0], 3)));

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            /* Uses non-default rounding for vec_signed or vec_round */
            const int32x4_t v_qs0 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[0][j], id[0]), 4, 1));
            const int32x4_t v_qs1 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[1][j], id[1]), 4, 1));
            const int32x4_t v_qs2 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[2][j], id[2]), 4, 1));
            const int32x4_t v_qs3 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[3][j], id[3]), 4, 1));

            const int16x8_t v_qs01 = vec_packs(v_qs0, v_qs1);
            const int16x8_t v_qs23 = vec_packs(v_qs2, v_qs3);

            vec_xst(vec_packs(v_qs01, v_qs23), 0, y[i].qs + 16 * j);
        }
    }
#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#endif
}

#if defined(__VXE__) || defined(__VXE2__)
static inline int16x8_t vxe_dot_acc(const int8x16_t v_x, const int8x16_t v_y, const int16x8_t v_acc) {
    return vec_meadd(v_x, v_y, vec_moadd(v_x, v_y, v_acc));
}

static inline int8x16_t vxe_splat_granule(const int8_t * qs) {
    uint32_t g;
    memcpy(&g, qs, sizeof(g));
    return (int8x16_t)vec_splats(g);
}

static inline int32x4_t vxe_fold(const int16x8_t v_sumi) {
    const int16x8_t v_ones = vec_splats((int16_t)1);
    return vec_add(vec_mule(v_sumi, v_ones), vec_mulo(v_sumi, v_ones));
}
#endif

void ggml_gemv_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

#if defined(__VXE__) || defined(__VXE2__)
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    float * res_ptr = s;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        float32x4_t v_sumf = vec_splats(0.0f);

        for (int l = 0; l < nb; l++) {
            const int8_t * x_qs = b_ptr[l].qs;

            const int8x16_t v_x0 = vec_xl( 0, x_qs);
            const int8x16_t v_x1 = vec_xl(16, x_qs);
            const int8x16_t v_x2 = vec_xl(32, x_qs);
            const int8x16_t v_x3 = vec_xl(48, x_qs);

            const int8x16_t v_x0l = vec_sra(vec_sl(v_x0, 4), 4);
            const int8x16_t v_x1l = vec_sra(vec_sl(v_x1, 4), 4);
            const int8x16_t v_x2l = vec_sra(vec_sl(v_x2, 4), 4);
            const int8x16_t v_x3l = vec_sra(vec_sl(v_x3, 4), 4);

            const int8x16_t v_x0h = vec_sra(v_x0, 4);
            const int8x16_t v_x1h = vec_sra(v_x1, 4);
            const int8x16_t v_x2h = vec_sra(v_x2, 4);
            const int8x16_t v_x3h = vec_sra(v_x3, 4);

            const int8_t * y_lo = a_ptr[l].qs;
            const int8_t * y_hi = y_lo + qk / 2;

            int16x8_t v_sumi = vec_splats((int16_t)0);

            v_sumi = vxe_dot_acc(v_x0l, vxe_splat_granule(y_lo +  0), v_sumi);
            v_sumi = vxe_dot_acc(v_x1l, vxe_splat_granule(y_lo +  4), v_sumi);
            v_sumi = vxe_dot_acc(v_x2l, vxe_splat_granule(y_lo +  8), v_sumi);
            v_sumi = vxe_dot_acc(v_x3l, vxe_splat_granule(y_lo + 12), v_sumi);

            v_sumi = vxe_dot_acc(v_x0h, vxe_splat_granule(y_hi +  0), v_sumi);
            v_sumi = vxe_dot_acc(v_x1h, vxe_splat_granule(y_hi +  4), v_sumi);
            v_sumi = vxe_dot_acc(v_x2h, vxe_splat_granule(y_hi +  8), v_sumi);
            v_sumi = vxe_dot_acc(v_x3h, vxe_splat_granule(y_hi + 12), v_sumi);

            const float32x4_t v_yd = vec_splats(GGML_CPU_FP16_TO_FP32(a_ptr[l].d));
            const float32x4_t v_xd = __lzs_f16cx4_load(b_ptr[l].d);
            const float32x4_t v_d  = vec_mul(v_yd, v_xd);

            v_sumf = vec_madd(vec_float(vxe_fold(v_sumi)), v_d, v_sumf);
        }

        vec_xst(v_sumf, 0, res_ptr + x * ncols_interleaved);
    }
#else
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    ggml_gemv_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;

    assert(nr % 4 == 0);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

#if defined(__VXE__) || defined(__VXE2__)
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);

        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

            float32x4_t v_sumf[4];
            for (int m = 0; m < 4; m++) {
                v_sumf[m] = vec_splats(0.0f);
            }

            for (int l = 0; l < nb; l++) {
                int16x8_t v_sumi0 = vec_splats((int16_t)0);
                int16x8_t v_sumi1 = vec_splats((int16_t)0);
                int16x8_t v_sumi2 = vec_splats((int16_t)0);
                int16x8_t v_sumi3 = vec_splats((int16_t)0);

                for (int k = 0; k < 4; k++) {
                    const int8x16_t v_x  = vec_xl(0, b_ptr[l].qs + 16 * k);
                    const int8x16_t v_xl = vec_sra(vec_sl(v_x, 4), 4);
                    const int8x16_t v_xh = vec_sra(v_x, 4);

                    const int8_t * y_lo = a_ptr[l].qs + 16 * k;
                    const int8_t * y_hi = y_lo + qk / 2 * 4;

                    v_sumi0 = vxe_dot_acc(v_xl, vxe_splat_granule(y_lo +  0), v_sumi0);
                    v_sumi1 = vxe_dot_acc(v_xl, vxe_splat_granule(y_lo +  4), v_sumi1);
                    v_sumi2 = vxe_dot_acc(v_xl, vxe_splat_granule(y_lo +  8), v_sumi2);
                    v_sumi3 = vxe_dot_acc(v_xl, vxe_splat_granule(y_lo + 12), v_sumi3);

                    v_sumi0 = vxe_dot_acc(v_xh, vxe_splat_granule(y_hi +  0), v_sumi0);
                    v_sumi1 = vxe_dot_acc(v_xh, vxe_splat_granule(y_hi +  4), v_sumi1);
                    v_sumi2 = vxe_dot_acc(v_xh, vxe_splat_granule(y_hi +  8), v_sumi2);
                    v_sumi3 = vxe_dot_acc(v_xh, vxe_splat_granule(y_hi + 12), v_sumi3);
                }

                const float32x4_t v_yd = __lzs_f16cx4_load(a_ptr[l].d);
                const float32x4_t v_xd = __lzs_f16cx4_load(b_ptr[l].d);

                v_sumf[0] = vec_madd(vec_float(vxe_fold(v_sumi0)), vec_mul(v_xd, vec_splat(v_yd, 0)), v_sumf[0]);
                v_sumf[1] = vec_madd(vec_float(vxe_fold(v_sumi1)), vec_mul(v_xd, vec_splat(v_yd, 1)), v_sumf[1]);
                v_sumf[2] = vec_madd(vec_float(vxe_fold(v_sumi2)), vec_mul(v_xd, vec_splat(v_yd, 2)), v_sumf[2]);
                v_sumf[3] = vec_madd(vec_float(vxe_fold(v_sumi3)), vec_mul(v_xd, vec_splat(v_yd, 3)), v_sumf[3]);
            }

            for (int m = 0; m < 4; m++) {
                vec_xst(v_sumf[m], 0, s + (y * 4 + m) * bs + x * ncols_interleaved);
            }
        }
    }
#else
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    ggml_gemm_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#endif
}
