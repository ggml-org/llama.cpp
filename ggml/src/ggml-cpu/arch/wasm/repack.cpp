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

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

#include "../../repack.h"

#define UNUSED GGML_UNUSED

#if defined(__wasm_simd128__)

// Wasm SIMD128 optimized quantization for Q8_0 4x4 interleaved blocks
void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    for (int i = 0; i < nb; i++) {
        v128_t srcv[4][8];
        float id[4];

        // Process 4 rows
        for (int row = 0; row < 4; row++) {
            v128_t asrcv[8];
            v128_t amaxv[8];

            // Load 8 vectors of 4 floats each (32 floats total per row)
            for (int j = 0; j < 8; j++) {
                srcv[row][j] = wasm_v128_load(x + row * k + i * 32 + 4 * j);
            }

            // Compute absolute values
            for (int j = 0; j < 8; j++) {
                asrcv[j] = wasm_f32x4_abs(srcv[row][j]);
            }

            // Find maximum across all 8 vectors using pairwise reduction
            for (int j = 0; j < 4; j++) {
                amaxv[2 * j] = wasm_f32x4_max(asrcv[2 * j], asrcv[2 * j + 1]);
            }
            for (int j = 0; j < 2; j++) {
                amaxv[4 * j] = wasm_f32x4_max(amaxv[4 * j], amaxv[4 * j + 2]);
            }
            amaxv[0] = wasm_f32x4_max(amaxv[0], amaxv[4]);

            // Extract maximum from the final vector
            float amax = wasm_f32x4_extract_lane(amaxv[0], 0);
            amax = fmaxf(amax, wasm_f32x4_extract_lane(amaxv[0], 1));
            amax = fmaxf(amax, wasm_f32x4_extract_lane(amaxv[0], 2));
            amax = fmaxf(amax, wasm_f32x4_extract_lane(amaxv[0], 3));

            const float d = amax / ((1 << 7) - 1);
            id[row] = d ? 1.0f / d : 0.0f;

            y[i].d[row] = GGML_CPU_FP32_TO_FP16(d);
        }

        // Quantize and interleave with blocklen=4
        for (int j = 0; j < 8; j++) {
            for (int row = 0; row < 4; row++) {
                v128_t v = wasm_f32x4_mul(srcv[row][j], wasm_f32x4_splat(id[row]));
                v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

                // Store interleaved: row0[0-3], row1[0-3], row2[0-3], row3[0-3]
                y[i].qs[16 * j + row * 4 + 0] = wasm_i32x4_extract_lane(vi, 0);
                y[i].qs[16 * j + row * 4 + 1] = wasm_i32x4_extract_lane(vi, 1);
                y[i].qs[16 * j + row * 4 + 2] = wasm_i32x4_extract_lane(vi, 2);
                y[i].qs[16 * j + row * 4 + 3] = wasm_i32x4_extract_lane(vi, 3);
            }
        }
    }
}

// Wasm SIMD128 optimized quantization for Q8_0 4x8 interleaved blocks
void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    for (int i = 0; i < nb; i++) {
        v128_t srcv[4][8];
        float id[4];

        // Process 4 rows
        for (int row = 0; row < 4; row++) {
            v128_t asrcv[8];
            v128_t amaxv[8];

            // Load 8 vectors of 4 floats each (32 floats total per row)
            for (int j = 0; j < 8; j++) {
                srcv[row][j] = wasm_v128_load(x + row * k + i * 32 + 4 * j);
            }

            // Compute absolute values
            for (int j = 0; j < 8; j++) {
                asrcv[j] = wasm_f32x4_abs(srcv[row][j]);
            }

            // Find maximum across all 8 vectors
            for (int j = 0; j < 4; j++) {
                amaxv[2 * j] = wasm_f32x4_max(asrcv[2 * j], asrcv[2 * j + 1]);
            }
            for (int j = 0; j < 2; j++) {
                amaxv[4 * j] = wasm_f32x4_max(amaxv[4 * j], amaxv[4 * j + 2]);
            }
            amaxv[0] = wasm_f32x4_max(amaxv[0], amaxv[4]);

            float amax = wasm_f32x4_extract_lane(amaxv[0], 0);
            amax = fmaxf(amax, wasm_f32x4_extract_lane(amaxv[0], 1));
            amax = fmaxf(amax, wasm_f32x4_extract_lane(amaxv[0], 2));
            amax = fmaxf(amax, wasm_f32x4_extract_lane(amaxv[0], 3));

            const float d = amax / ((1 << 7) - 1);
            id[row] = d ? 1.0f / d : 0.0f;

            y[i].d[row] = GGML_CPU_FP32_TO_FP16(d);
        }

        // Quantize and interleave with blocklen=8
        for (int j = 0; j < 4; j++) {
            for (int row = 0; row < 4; row++) {
                // First 4 floats of block
                v128_t v0 = wasm_f32x4_mul(srcv[row][2 * j], wasm_f32x4_splat(id[row]));
                v128_t vi0 = wasm_i32x4_trunc_sat_f32x4(v0);

                // Second 4 floats of block
                v128_t v1 = wasm_f32x4_mul(srcv[row][2 * j + 1], wasm_f32x4_splat(id[row]));
                v128_t vi1 = wasm_i32x4_trunc_sat_f32x4(v1);

                // Store interleaved with blocklen=8
                y[i].qs[32 * j + row * 8 + 0] = wasm_i32x4_extract_lane(vi0, 0);
                y[i].qs[32 * j + row * 8 + 1] = wasm_i32x4_extract_lane(vi0, 1);
                y[i].qs[32 * j + row * 8 + 2] = wasm_i32x4_extract_lane(vi0, 2);
                y[i].qs[32 * j + row * 8 + 3] = wasm_i32x4_extract_lane(vi0, 3);
                y[i].qs[32 * j + row * 8 + 4] = wasm_i32x4_extract_lane(vi1, 0);
                y[i].qs[32 * j + row * 8 + 5] = wasm_i32x4_extract_lane(vi1, 1);
                y[i].qs[32 * j + row * 8 + 6] = wasm_i32x4_extract_lane(vi1, 2);
                y[i].qs[32 * j + row * 8 + 7] = wasm_i32x4_extract_lane(vi1, 3);
            }
        }
    }
}

// Wasm SIMD128 optimized GEMV for Q4_0 4x4 with Q8_0 activation
void ggml_gemv_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;

    for (int x_idx = 0; x_idx < nc / ncols_interleaved; x_idx++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x_idx * nb);

        v128_t acc = wasm_f32x4_splat(0.0f);

        for (int l = 0; l < nb; l++) {
            float a_d = GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
            v128_t b_d = wasm_f32x4_make(
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3])
            );

            v128_t sumi = wasm_i32x4_splat(0);

            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int i = 0; i < blocklen; i++) {
                    int base = k * ncols_interleaved * blocklen + i;

                    int8_t b0 = b_ptr[l].qs[base + 0 * blocklen];
                    int8_t b1 = b_ptr[l].qs[base + 1 * blocklen];
                    int8_t b2 = b_ptr[l].qs[base + 2 * blocklen];
                    int8_t b3 = b_ptr[l].qs[base + 3 * blocklen];

                    v128_t v0 = wasm_i32x4_make(
                        (int32_t)(int8_t)(b0 << 4),
                        (int32_t)(int8_t)(b1 << 4),
                        (int32_t)(int8_t)(b2 << 4),
                        (int32_t)(int8_t)(b3 << 4)
                    );
                    v128_t v1 = wasm_i32x4_make(
                        (int32_t)(int8_t)(b0 & 0xF0),
                        (int32_t)(int8_t)(b1 & 0xF0),
                        (int32_t)(int8_t)(b2 & 0xF0),
                        (int32_t)(int8_t)(b3 & 0xF0)
                    );

                    int32_t a_val_lo = a_ptr[l].qs[k * blocklen + i];
                    int32_t a_val_hi = a_ptr[l].qs[k * blocklen + i + qk / 2];

                    v128_t mul0 = wasm_i32x4_mul(v0, wasm_i32x4_splat(a_val_lo));
                    v128_t mul1 = wasm_i32x4_mul(v1, wasm_i32x4_splat(a_val_hi));
                    v128_t sum = wasm_i32x4_add(mul0, mul1);
                    sum = wasm_i32x4_shr(sum, 4);
                    sumi = wasm_i32x4_add(sumi, sum);
                }
            }

            v128_t sumf = wasm_f32x4_convert_i32x4(sumi);
            v128_t scale = wasm_f32x4_mul(b_d, wasm_f32x4_splat(a_d));
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(sumf, scale));
        }

        wasm_v128_store(s + x_idx * ncols_interleaved, acc);
    }
}

// Wasm SIMD128 optimized GEMM for Q4_0 4x4 with Q8_0 activation
void ggml_gemm_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);
    assert(nr % 4 == 0);

    UNUSED(bs);

    for (int row = 0; row < nr; row += 4) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) ((const char *)vy + row * nb * sizeof(block_q8_0));

        for (int x_idx = 0; x_idx < nc / ncols_interleaved; x_idx++) {
            const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x_idx * nb);

            v128_t acc[4] = {
                wasm_f32x4_splat(0.0f),
                wasm_f32x4_splat(0.0f),
                wasm_f32x4_splat(0.0f),
                wasm_f32x4_splat(0.0f)
            };

            for (int l = 0; l < nb; l++) {
                v128_t b_d = wasm_f32x4_make(
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3])
                );

                for (int r = 0; r < 4; r++) {
                    float a_d = GGML_CPU_FP16_TO_FP32(a_ptr[l].d[r]);
                    v128_t sumi = wasm_i32x4_splat(0);

                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int i = 0; i < blocklen; i++) {
                            int base = k * ncols_interleaved * blocklen + i;

                            int8_t b0 = b_ptr[l].qs[base + 0 * blocklen];
                            int8_t b1 = b_ptr[l].qs[base + 1 * blocklen];
                            int8_t b2 = b_ptr[l].qs[base + 2 * blocklen];
                            int8_t b3 = b_ptr[l].qs[base + 3 * blocklen];

                            v128_t v0 = wasm_i32x4_make(
                                (int32_t)(int8_t)(b0 << 4),
                                (int32_t)(int8_t)(b1 << 4),
                                (int32_t)(int8_t)(b2 << 4),
                                (int32_t)(int8_t)(b3 << 4)
                            );
                            v128_t v1 = wasm_i32x4_make(
                                (int32_t)(int8_t)(b0 & 0xF0),
                                (int32_t)(int8_t)(b1 & 0xF0),
                                (int32_t)(int8_t)(b2 & 0xF0),
                                (int32_t)(int8_t)(b3 & 0xF0)
                            );

                            int32_t a_val_lo = a_ptr[l].qs[k * 16 + r * blocklen + i];
                            int32_t a_val_hi = a_ptr[l].qs[k * 16 + r * blocklen + i + 64];

                            v128_t mul0 = wasm_i32x4_mul(v0, wasm_i32x4_splat(a_val_lo));
                            v128_t mul1 = wasm_i32x4_mul(v1, wasm_i32x4_splat(a_val_hi));
                            v128_t sum = wasm_i32x4_add(mul0, mul1);
                            sum = wasm_i32x4_shr(sum, 4);
                            sumi = wasm_i32x4_add(sumi, sum);
                        }
                    }

                    v128_t sumf = wasm_f32x4_convert_i32x4(sumi);
                    v128_t scale = wasm_f32x4_mul(b_d, wasm_f32x4_splat(a_d));
                    acc[r] = wasm_f32x4_add(acc[r], wasm_f32x4_mul(sumf, scale));
                }
            }

            for (int r = 0; r < 4; r++) {
                wasm_v128_store(s + (row + r) * nc + x_idx * ncols_interleaved, acc[r]);
            }
        }
    }
}

// For other functions, fall back to generic implementations
void ggml_gemv_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q4_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;

    for (int x_idx = 0; x_idx < nc / ncols_interleaved; x_idx++) {
        const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x_idx * nb);

        v128_t acc0 = wasm_f32x4_splat(0.0f);
        v128_t acc1 = wasm_f32x4_splat(0.0f);

        for (int l = 0; l < nb; l++) {
            float a_d = GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
            v128_t b_d0 = wasm_f32x4_make(
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3])
            );
            v128_t b_d1 = wasm_f32x4_make(
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[4]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[5]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[6]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[7])
            );

            v128_t sumi0 = wasm_i32x4_splat(0);
            v128_t sumi1 = wasm_i32x4_splat(0);

            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int i = 0; i < blocklen; i++) {
                    int base = k * ncols_interleaved * blocklen + i;

                    int8_t b0 = b_ptr[l].qs[base + 0 * blocklen];
                    int8_t b1 = b_ptr[l].qs[base + 1 * blocklen];
                    int8_t b2 = b_ptr[l].qs[base + 2 * blocklen];
                    int8_t b3 = b_ptr[l].qs[base + 3 * blocklen];
                    int8_t b4 = b_ptr[l].qs[base + 4 * blocklen];
                    int8_t b5 = b_ptr[l].qs[base + 5 * blocklen];
                    int8_t b6 = b_ptr[l].qs[base + 6 * blocklen];
                    int8_t b7 = b_ptr[l].qs[base + 7 * blocklen];

                    v128_t v0_0 = wasm_i32x4_make(
                        (int32_t)(int8_t)(b0 << 4),
                        (int32_t)(int8_t)(b1 << 4),
                        (int32_t)(int8_t)(b2 << 4),
                        (int32_t)(int8_t)(b3 << 4)
                    );
                    v128_t v1_0 = wasm_i32x4_make(
                        (int32_t)(int8_t)(b0 & 0xF0),
                        (int32_t)(int8_t)(b1 & 0xF0),
                        (int32_t)(int8_t)(b2 & 0xF0),
                        (int32_t)(int8_t)(b3 & 0xF0)
                    );

                    v128_t v0_1 = wasm_i32x4_make(
                        (int32_t)(int8_t)(b4 << 4),
                        (int32_t)(int8_t)(b5 << 4),
                        (int32_t)(int8_t)(b6 << 4),
                        (int32_t)(int8_t)(b7 << 4)
                    );
                    v128_t v1_1 = wasm_i32x4_make(
                        (int32_t)(int8_t)(b4 & 0xF0),
                        (int32_t)(int8_t)(b5 & 0xF0),
                        (int32_t)(int8_t)(b6 & 0xF0),
                        (int32_t)(int8_t)(b7 & 0xF0)
                    );

                    int32_t a_val_lo = a_ptr[l].qs[k * blocklen + i];
                    int32_t a_val_hi = a_ptr[l].qs[k * blocklen + i + qk / 2];

                    v128_t mul0_0 = wasm_i32x4_mul(v0_0, wasm_i32x4_splat(a_val_lo));
                    v128_t mul1_0 = wasm_i32x4_mul(v1_0, wasm_i32x4_splat(a_val_hi));
                    v128_t sum0 = wasm_i32x4_add(mul0_0, mul1_0);
                    sum0 = wasm_i32x4_shr(sum0, 4);
                    sumi0 = wasm_i32x4_add(sumi0, sum0);

                    v128_t mul0_1 = wasm_i32x4_mul(v0_1, wasm_i32x4_splat(a_val_lo));
                    v128_t mul1_1 = wasm_i32x4_mul(v1_1, wasm_i32x4_splat(a_val_hi));
                    v128_t sum1 = wasm_i32x4_add(mul0_1, mul1_1);
                    sum1 = wasm_i32x4_shr(sum1, 4);
                    sumi1 = wasm_i32x4_add(sumi1, sum1);
                }
            }

            v128_t sumf0 = wasm_f32x4_convert_i32x4(sumi0);
            v128_t sumf1 = wasm_f32x4_convert_i32x4(sumi1);
            v128_t scale0 = wasm_f32x4_mul(b_d0, wasm_f32x4_splat(a_d));
            v128_t scale1 = wasm_f32x4_mul(b_d1, wasm_f32x4_splat(a_d));
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(sumf0, scale0));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(sumf1, scale1));
        }

        wasm_v128_store(s + x_idx * ncols_interleaved, acc0);
        wasm_v128_store(s + x_idx * ncols_interleaved + 4, acc1);
    }
}

void ggml_gemm_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q4_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);
    assert(nr % 4 == 0);

    UNUSED(bs);

    for (int row = 0; row < nr; row += 4) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) ((const char *)vy + row * nb * sizeof(block_q8_0));

        for (int x_idx = 0; x_idx < nc / ncols_interleaved; x_idx++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x_idx * nb);

            v128_t acc0[4] = {
                wasm_f32x4_splat(0.0f),
                wasm_f32x4_splat(0.0f),
                wasm_f32x4_splat(0.0f),
                wasm_f32x4_splat(0.0f)
            };
            v128_t acc1[4] = {
                wasm_f32x4_splat(0.0f),
                wasm_f32x4_splat(0.0f),
                wasm_f32x4_splat(0.0f),
                wasm_f32x4_splat(0.0f)
            };

            for (int l = 0; l < nb; l++) {
                v128_t b_d0 = wasm_f32x4_make(
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3])
                );
                v128_t b_d1 = wasm_f32x4_make(
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[4]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[5]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[6]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[7])
                );

                for (int r = 0; r < 4; r++) {
                    float a_d = GGML_CPU_FP16_TO_FP32(a_ptr[l].d[r]);
                    v128_t sumi0 = wasm_i32x4_splat(0);
                    v128_t sumi1 = wasm_i32x4_splat(0);

                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int i = 0; i < blocklen; i++) {
                            int base = k * ncols_interleaved * blocklen + i;

                            int8_t b0 = b_ptr[l].qs[base + 0 * blocklen];
                            int8_t b1 = b_ptr[l].qs[base + 1 * blocklen];
                            int8_t b2 = b_ptr[l].qs[base + 2 * blocklen];
                            int8_t b3 = b_ptr[l].qs[base + 3 * blocklen];
                            int8_t b4 = b_ptr[l].qs[base + 4 * blocklen];
                            int8_t b5 = b_ptr[l].qs[base + 5 * blocklen];
                            int8_t b6 = b_ptr[l].qs[base + 6 * blocklen];
                            int8_t b7 = b_ptr[l].qs[base + 7 * blocklen];

                            v128_t v0_0 = wasm_i32x4_make(
                                (int32_t)(int8_t)(b0 << 4),
                                (int32_t)(int8_t)(b1 << 4),
                                (int32_t)(int8_t)(b2 << 4),
                                (int32_t)(int8_t)(b3 << 4)
                            );
                            v128_t v1_0 = wasm_i32x4_make(
                                (int32_t)(int8_t)(b0 & 0xF0),
                                (int32_t)(int8_t)(b1 & 0xF0),
                                (int32_t)(int8_t)(b2 & 0xF0),
                                (int32_t)(int8_t)(b3 & 0xF0)
                            );

                            v128_t v0_1 = wasm_i32x4_make(
                                (int32_t)(int8_t)(b4 << 4),
                                (int32_t)(int8_t)(b5 << 4),
                                (int32_t)(int8_t)(b6 << 4),
                                (int32_t)(int8_t)(b7 << 4)
                            );
                            v128_t v1_1 = wasm_i32x4_make(
                                (int32_t)(int8_t)(b4 & 0xF0),
                                (int32_t)(int8_t)(b5 & 0xF0),
                                (int32_t)(int8_t)(b6 & 0xF0),
                                (int32_t)(int8_t)(b7 & 0xF0)
                            );

                            int32_t a_val_lo = a_ptr[l].qs[k * 4 * blocklen + r * blocklen + i];
                            int32_t a_val_hi = a_ptr[l].qs[k * 4 * blocklen + r * blocklen + i + qk / 2 * 4];

                            v128_t mul0_0 = wasm_i32x4_mul(v0_0, wasm_i32x4_splat(a_val_lo));
                            v128_t mul1_0 = wasm_i32x4_mul(v1_0, wasm_i32x4_splat(a_val_hi));
                            v128_t sum0 = wasm_i32x4_add(mul0_0, mul1_0);
                            sum0 = wasm_i32x4_shr(sum0, 4);
                            sumi0 = wasm_i32x4_add(sumi0, sum0);

                            v128_t mul0_1 = wasm_i32x4_mul(v0_1, wasm_i32x4_splat(a_val_lo));
                            v128_t mul1_1 = wasm_i32x4_mul(v1_1, wasm_i32x4_splat(a_val_hi));
                            v128_t sum1 = wasm_i32x4_add(mul0_1, mul1_1);
                            sum1 = wasm_i32x4_shr(sum1, 4);
                            sumi1 = wasm_i32x4_add(sumi1, sum1);
                        }
                    }

                    v128_t sumf0 = wasm_f32x4_convert_i32x4(sumi0);
                    v128_t sumf1 = wasm_f32x4_convert_i32x4(sumi1);
                    v128_t scale0 = wasm_f32x4_mul(b_d0, wasm_f32x4_splat(a_d));
                    v128_t scale1 = wasm_f32x4_mul(b_d1, wasm_f32x4_splat(a_d));
                    acc0[r] = wasm_f32x4_add(acc0[r], wasm_f32x4_mul(sumf0, scale0));
                    acc1[r] = wasm_f32x4_add(acc1[r], wasm_f32x4_mul(sumf1, scale1));
                }
            }

            for (int r = 0; r < 4; r++) {
                wasm_v128_store(s + (row + r) * nc + x_idx * ncols_interleaved, acc0[r]);
                wasm_v128_store(s + (row + r) * nc + x_idx * ncols_interleaved + 4, acc1[r]);
            }
        }
    }
}

// Q4_K functions - fall back to generic
void ggml_quantize_mat_q8_K_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    ggml_quantize_mat_q8_K_4x4_generic(x, vy, k);
}

void ggml_quantize_mat_q8_K_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    ggml_quantize_mat_q8_K_4x8_generic(x, vy, k);
}

void ggml_gemv_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

// Q2_K functions - fall back to generic
void ggml_gemv_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q2_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q2_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

// IQ4_NL functions - fall back to generic
void ggml_gemv_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_iq4_nl_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_iq4_nl_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_iq4_nl_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_iq4_nl_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

// Q8_0 functions - fall back to generic
void ggml_gemv_q8_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q8_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q8_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q8_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q8_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q8_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q8_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q8_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

#endif // __wasm_simd128__

