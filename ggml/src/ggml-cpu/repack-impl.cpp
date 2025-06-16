#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "traits.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#include "repack.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Functions to create the interleaved data layout formats

// interleave 4 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x4
// in the interleaved block_q4_0x4, place deltas for 4 block_q4_0 blocks
// first, then interleave quants from 4 block_q4_0s in blocks of blck_size_interleave
//
// - in                  : an array of block_q4_0 pointers
// - blck_size_interleave : the block_q4_0 quants bytes are interleaved in blocks of
//                         blck_size_interleave bytes
// - xor_mask            : the mask to convert the nibbles in block_q4_0 quants bytes
//                         from bias offset form to pure sign form (this saves subtract
//                         operations durin unpacking)
//

void ggml_quantize_mat_q8_0_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_quantize_mat_q8_0_4x4)

void ggml_quantize_mat_q8_0_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_quantize_mat_q8_0_4x8)

void ggml_quantize_mat_q8_K_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_quantize_mat_q8_K_4x8)

void ggml_gemv_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}
GGML_CPU_NATIVE_IMPL(ggml_gemv_q4_0_4x4_q8_0)

void ggml_gemv_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}
GGML_CPU_NATIVE_IMPL(ggml_gemv_q4_0_4x8_q8_0)

void ggml_gemv_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    {
        float sumf[8];
        int sumi;

        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);

            for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        sumi = 0;
                        for (int i = 0; i < blocklen; ++i) {
                            const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                            const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                            sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                        }
                        sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d);
                    }
                }
            }
            for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_gemv_q4_0_8x8_q8_0)

void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

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
                    sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_gemv_q4_K_8x8_q8_K)

void ggml_gemv_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    {
        float sumf[4];
        int sumi;

        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

            for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        sumi = 0;
                        for (int i = 0; i < blocklen; ++i) {
                            const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                            const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                            sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                        }
                        sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d);
                    }
                }
            }
            for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_gemv_iq4_nl_4x4_q8_0)

void ggml_gemm_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_gemm_q4_0_4x4_q8_0)

void ggml_gemm_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    float sumf[4][4];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_gemm_q4_0_4x8_q8_0)

void ggml_gemm_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_gemm_q4_0_8x8_q8_0)

void ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

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
                            sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
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
}
GGML_CPU_NATIVE_IMPL(ggml_gemm_q4_K_8x8_q8_K)

void ggml_gemm_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}
GGML_CPU_NATIVE_IMPL(ggml_gemm_iq4_nl_4x4_q8_0)
