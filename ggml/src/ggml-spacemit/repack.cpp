#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "repack.h"

#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ime_kernels.h"
#include "rvv_kernels.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

// clang-format off
#if defined(__riscv)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh)
#error "riscv zfh extension not enabled"
#endif

#else
#error "riscv not enabled in this build"
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

template <int K> constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}

template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

template <int K, int N> struct block_with_zp {
    ggml_half d[N];                         // deltas for N qK_1 blocks
    uint8_t   zp[N];                        // zero points for N qK_1 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_1 blocks
};

// control size
static_assert(sizeof(block<4, 16>) == 16 * sizeof(ggml_half) + QK4_0 * 8, "wrong block<4,16> size/padding");
static_assert(sizeof(block_with_zp<4, 16>) == 16 * sizeof(ggml_half) + QK4_0 * 8 + 16 * sizeof(uint8_t),
              "wrong block_with_zp<4,16> size/padding");

static_assert(sizeof(block<8, 16>) == 16 * sizeof(ggml_half) + QK4_0 * 16, "wrong block<8,16> size/padding");

static_assert(sizeof(block<4, 32>) == 32 * sizeof(ggml_half) + QK4_0 * 16, "wrong block<4,32> size/padding");
static_assert(sizeof(block_with_zp<4, 32>) == 32 * sizeof(ggml_half) + QK4_0 * 16 + 32 * sizeof(uint8_t),
              "wrong block_with_zp<4,32> size/padding");

using block_q4_0x16 = block<4, 16>;
using block_q4_1x16 = block_with_zp<4, 16>;
using block_q8_0x16 = block<8, 16>;

using block_q4_0x32 = block<4, 32>;
using block_q4_1x32 = block_with_zp<4, 32>;
using block_q8_0x32 = block<8, 32>;

struct block_q4_0x32x256 {
    block_q4_0x32 blocks[8];  // [f16 * 32 | i4 * 32 * 32] * 8
};

struct block_q4_1x32x256 {
    block_q4_0x32 blocks[8];
    uint8_t       zps[32 * 8];
};

static block_q4_0x16 make_block_q4_0x16(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x16 out;
    GGML_ASSERT(QK4_0 / blck_size_interleave == 2);

    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    for (int i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        for (int j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            out.qs[i * QK4_0 / 4 + j] = (in[i].qs[j] & 0x0F) | ((in[i].qs[j + QK4_0 / 4] & 0x0F) << 4);
        }
    }

    for (int i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        for (int j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            out.qs[4 * QK4_0 + i * QK4_0 / 4 + j] = ((in[i].qs[j] & 0xF0) >> 4) | (in[i].qs[j + QK4_0 / 4] & 0xF0);
        }
    }

    return out;
}

static block_q4_1x16 make_block_q4_1x16(block_q4_1 * in, unsigned int blck_size_interleave) {
    block_q4_1x16 out;
    GGML_ASSERT(QK4_1 / blck_size_interleave == 2);

    for (int i = 0; i < 16; i++) {
        float d   = GGML_FP16_TO_FP32(in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        float m   = GGML_FP16_TO_FP32(in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        float mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        out.d[i]  = GGML_FP32_TO_FP16(d);
        out.zp[i] = static_cast<uint8_t>(mid);
    }

    for (int i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        for (int j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            out.qs[i * QK4_1 / 4 + j] = (in[i].qs[j] & 0x0F) | ((in[i].qs[j + QK4_1 / 4] & 0x0F) << 4);
        }
    }

    for (int i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        for (int j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            out.qs[4 * QK4_1 + i * QK4_1 / 4 + j] = ((in[i].qs[j] & 0xF0) >> 4) | (in[i].qs[j + QK4_1 / 4] & 0xF0);
        }
    }

    return out;
}

static int repack_q4_0_to_q4_0_16_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 16);

    constexpr int nrows_interleaved = 16;

    block_q4_0x16 *    dst = (block_q4_0x16 *) t->data;
    const block_q4_0 * src = (const block_q4_0 *) data;
    block_q4_0         dst_tmp[16];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_1_to_q4_1_16_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_block == 16);

    constexpr int nrows_interleaved = 16;

    block_q4_1x16 *    dst = (block_q4_1x16 *) t->data;
    const block_q4_1 * src = (const block_q4_1 *) data;
    block_q4_1         dst_tmp[16];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_1));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_1x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static inline void get_scale_min_k4(int                           j,
                                    const uint8_t * GGML_RESTRICT q,
                                    uint8_t * GGML_RESTRICT       d,
                                    uint8_t * GGML_RESTRICT       m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static int repack_q4_k_to_q4_1_16_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 16);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr int nrows_interleaved = 16;

    block_q4_1x16 *    dst = (block_q4_1x16 *) t->data;
    const block_q4_K * src = (const block_q4_K *) data;
    block_q4_1         dst_tmp[16];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK_K;

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    const float d = GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    const float min =
                        GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    const float d1 = d * sc;
                    const float m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_TO_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_TO_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    const uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    if (j % 2 == 0) {
                        for (int ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } else {
                        for (int ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_block_q4_1x16(dst_tmp, interleave_block);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_q4_0x32 make_block_q4_0x32(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x32 out;
    assert(QK4_0 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    for (int i = 0; i < 32; i++) {
        out.d[i] = in[i].d;
    }

    for (int i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        for (int j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] .........  [b14 b15]
            out.qs[i * QK4_0 / 2 + j] = (in[i].qs[j * 2] & 0x0F) | ((in[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    for (int i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        for (int j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b17] ......... [b30 b31]
            out.qs[i * QK4_0 / 2 + QK4_0 / 4 + j] = ((in[i].qs[j * 2] & 0xF0) >> 4) | (in[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    return out;
}

static block_q4_1x32 make_block_q4_1x32(block_q4_1 * in, unsigned int blck_size_interleave) {
    block_q4_1x32 out;
    GGML_ASSERT(QK4_1 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    for (int i = 0; i < 32; i++) {
        float d   = GGML_FP16_TO_FP32(in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        float m   = GGML_FP16_TO_FP32(in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        float mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        out.d[i]  = GGML_FP32_TO_FP16(d);
        out.zp[i] = static_cast<uint8_t>(mid);
    }

    for (int i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        for (int j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] ......... [b14 b15]
            out.qs[i * QK4_1 / 2 + j] = (in[i].qs[j * 2] & 0x0F) | ((in[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    for (int i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        for (int j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            out.qs[i * QK4_1 / 2 + QK4_1 / 4 + j] = ((in[i].qs[j * 2] & 0xF0) >> 4) | (in[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    return out;
}

static block_q8_0x32 make_block_q8_0x32(block_q8_0 * in, unsigned int blck_size_interleave) {
    block_q8_0x32 out;
    GGML_ASSERT(QK8_0 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    for (int i = 0; i < 32; i++) {
        out.d[i] = in[i].d;
    }

    for (int i = 0; i < 32; i++) {
        memcpy(out.qs + i * QK8_0, in[i].qs, QK8_0);
    }

    return out;
}

static int repack_q2_k_to_q2_k_32_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr int nrows_interleaved = 32;

    const block_q2_K * src = (const block_q2_K *) data;

    auto * dst = (spacemit_kernels::nrow_block_q2_k<32> *) t->data;

    int nrow    = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q2_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        return -1;
    }

    uint8_t qs_aux[256] = { 0 };
    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                const block_q2_K * src_block = &src[(b + i) * nblocks + x];

                // scale for [16, N]
                for (int j = 0; j < 16; j++) {
                    auto zp_aux = (dst->scales[j * nrows_interleaved + i]) & 0xF0;

                    dst->scales[j * nrows_interleaved + i] = (src_block->scales[j] & 0x0F) | zp_aux;
                }

                // zp for [N, 16]
                for (int j = 0; j < 16; j++) {
                    auto scale_aux = (dst->scales[16 * i + j]) & 0x0F;

                    dst->scales[16 * i + j] = (src_block->scales[j] & 0xF0) | scale_aux;
                }

                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_block->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_block->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 16; j++) {
                        uint8_t qs0  = qs_aux[j + k * 64];
                        uint8_t qs16 = qs_aux[j + 16 + k * 64];
                        uint8_t qs32 = qs_aux[j + 32 + k * 64];
                        uint8_t qs48 = qs_aux[j + 48 + k * 64];

                        dst->qs[(k * nrows_interleaved + i) * 16 + j] =
                            (qs0 & 0x03) | ((qs16 & 0x03) << 2) | ((qs32 & 0x03) << 4) | ((qs48 & 0x03) << 6);
                    }
                }

                dst->scales16[i] = src_block->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                dst->zeros16[i]  = src_block->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
            }
            dst++;
        }
    }

    return 0;
}

static int repack_q3_k_to_q3_k_32_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q3_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr int nrows_interleaved = 32;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * src = (const block_q3_K *) data;

    auto * dst = (spacemit_kernels::nrow_block_q3_k<32> *) t->data;

    int nrow    = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q3_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        return -1;
    }

    uint32_t b_scale_aux[4] = { 0 };
    uint8_t  qs_aux[256]    = { 0 };

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                const block_q3_K * src_block = &src[(b + i) * nblocks + x];

                uint32_t * auxs  = b_scale_aux;
                int8_t *   scale = (int8_t *) auxs;
                memcpy(auxs, src_block->scales, 12);

                uint32_t tmp = auxs[2];
                auxs[2]      = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                auxs[3]      = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                auxs[0]      = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                auxs[1]      = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

                for (int j = 0; j < 16; j++) {
                    dst->scales[j * nrows_interleaved + i] = scale[j] - 32;
                }

                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_block->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_block->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 16; j++) {
                        uint8_t qs0  = qs_aux[j + k * 64];
                        uint8_t qs16 = qs_aux[j + 16 + k * 64];
                        uint8_t qs32 = qs_aux[j + 32 + k * 64];
                        uint8_t qs48 = qs_aux[j + 48 + k * 64];

                        dst->qs[(k * nrows_interleaved + i) * 16 + j] =
                            (qs0 & 0x03) | ((qs16 & 0x03) << 2) | ((qs32 & 0x03) << 4) | ((qs48 & 0x03) << 6);
                    }
                }

                //memcpy(dst->hmask + i * 32, src_block->hmask, 32);

                // from nrows_interleaved * [32byte]
                // to 16 * [nrows_interleaved * uint16_t]
                uint16_t * dst_mask = ((uint16_t *) dst->hmask) + i;
                for (int j = 0; j < 16; j++, dst_mask += nrows_interleaved) {
                    uint8_t   b_shift    = j / 2;
                    uint8_t * b_mask_col = (uint8_t *) (src_block->hmask + (j % 2) * 16);
                    // b0 - b15
                    uint16_t  msk_out_0  = 0;

                    for (int k = 0; k < 8; k++) {
                        msk_out_0 |= (uint16_t) ((b_mask_col[k] >> b_shift) & 0x01) << k;
                    }
                    for (int k = 8; k < 16; k++) {
                        msk_out_0 |= (uint16_t) ((b_mask_col[k] >> b_shift) & 0x01) << k;
                    }

                    dst_mask[0] = msk_out_0;
                }

                dst->scales16[i] = src_block->d;
            }

            dst++;
        }
    }

    return 0;
}

static int repack_q4_0_to_q4_0_32_bl_ref(ggml_tensor *              t,
                                         int                        interleave_block,
                                         const void * GGML_RESTRICT data,
                                         size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr int nrows_interleaved = 32;

    block_q4_0x32 *    dst = (block_q4_0x32 *) t->data;
    const block_q4_0 * src = (const block_q4_0 *) data;
    block_q4_0         dst_tmp[32];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_0_to_q4_0_256_32_bl_ref(ggml_tensor *              t,
                                             int                        interleave_block,
                                             const void * GGML_RESTRICT data,
                                             size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr int nrows_interleaved = 32;

    block_q4_0x32x256 * dst = (block_q4_0x32x256 *) t->data;
    const block_q4_0 *  src = (const block_q4_0 *) data;
    block_q4_0          dst_tmp[32];
    int                 nrow    = ggml_nrows(t);
    int                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x += 8) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }
                dst->blocks[j] = make_block_q4_0x32(dst_tmp, interleave_block);
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_0_to_q4_1_256_32_bl_ref(ggml_tensor *              t,
                                             int                        interleave_block,
                                             const void * GGML_RESTRICT data,
                                             size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr int nrows_interleaved = 32;

    block_q4_1x32x256 * dst = (block_q4_1x32x256 *) t->data;
    const block_q4_1 *  src = (const block_q4_1 *) data;
    block_q4_1          dst_tmp[32];
    int                 nrow    = ggml_nrows(t);
    int                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_1));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x += 8) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }

                block_q4_0x32 * dst_block = &dst->blocks[j];
                uint8_t *       dst_zp    = dst->zps + j * nrows_interleaved;

                for (int i = 0; i < nrows_interleaved; i++) {
                    float d   = GGML_FP16_TO_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    float m   = GGML_FP16_TO_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
                    float mid = -std::nearbyintf(m / d);
                    mid       = std::min(15.0f, std::max(0.0f, mid));

                    dst_block->d[i] = GGML_FP32_TO_FP16(d);
                    dst_zp[i]       = static_cast<uint8_t>(mid);
                }

                for (int i = 0; i < nrows_interleaved; i++) {
                    for (int k = 0; k < QK4_1 / 4; k++) {
                        dst_block->qs[i * QK4_1 / 2 + k] =
                            (dst_tmp[i].qs[k * 2] & 0x0F) | ((dst_tmp[i].qs[k * 2 + 1] & 0x0F) << 4);
                    }
                }

                for (int i = 0; i < nrows_interleaved; i++) {
                    for (int k = 0; k < QK4_1 / 4; k++) {
                        dst_block->qs[i * QK4_1 / 2 + QK4_1 / 4 + k] =
                            ((dst_tmp[i].qs[k * 2] & 0xF0) >> 4) | (dst_tmp[i].qs[k * 2 + 1] & 0xF0);
                    }
                }
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

// Map Q4_K K32 scales/minima to the Q4_1 superblock layout, with min=-dmin*sub_min.
static int repack_q4_k_to_q4_1_256_32_bl_ref(ggml_tensor *              t,
                                             int                        interleave_block,
                                             const void * GGML_RESTRICT data,
                                             size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr int nrows_interleaved = 32;

    block_q4_1x32x256 * dst = (block_q4_1x32x256 *) t->data;
    const block_q4_K *  src = (const block_q4_K *) data;
    block_q4_1          dst_tmp[32];
    int                 nrow    = ggml_nrows(t);
    int                 nblocks = t->ne[0] / QK_K;

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    const float d = GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    const float min =
                        GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    const float d1 = d * sc;
                    const float m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_TO_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_TO_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    const uint8_t * q = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    if (j % 2 == 0) {
                        for (int ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } else {
                        for (int ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }

                block_q4_0x32 * dst_block = &dst->blocks[j];
                uint8_t *       dst_zp    = dst->zps + j * nrows_interleaved;

                for (int i = 0; i < nrows_interleaved; i++) {
                    float d   = GGML_FP16_TO_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    float m   = GGML_FP16_TO_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
                    float mid = -std::nearbyintf(m / d);
                    mid       = std::min(15.0f, std::max(0.0f, mid));

                    dst_block->d[i] = GGML_FP32_TO_FP16(d);
                    dst_zp[i]       = static_cast<uint8_t>(mid);
                }

                for (int i = 0; i < nrows_interleaved; i++) {
                    for (int k = 0; k < QK4_1 / 4; k++) {
                        dst_block->qs[i * QK4_1 / 2 + k] =
                            (dst_tmp[i].qs[k * 2] & 0x0F) | ((dst_tmp[i].qs[k * 2 + 1] & 0x0F) << 4);
                    }
                }

                for (int i = 0; i < nrows_interleaved; i++) {
                    for (int k = 0; k < QK4_1 / 4; k++) {
                        dst_block->qs[i * QK4_1 / 2 + QK4_1 / 4 + k] =
                            ((dst_tmp[i].qs[k * 2] & 0xF0) >> 4) | (dst_tmp[i].qs[k * 2 + 1] & 0xF0);
                    }
                }
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

// RVV optimized version of repack_q4_0_to_q4_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack.
static int repack_q4_0_to_q4_0_32_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 32);

    constexpr int nrows_interleaved = 32;
    constexpr int qs_bytes          = QK4_0 / 2;  // 16

    block_q4_0x32 *    dst     = (block_q4_0x32 *) t->data;
    const block_q4_0 * src     = (const block_q4_0 *) data;
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        return -1;
    }

    const ptrdiff_t row_stride = (ptrdiff_t) nblocks * sizeof(block_q4_0);

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            const block_q4_0 * col_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            // d is at offset 0 of each block_q4_0, stride between rows = row_stride
            {
                const uint8_t * d_base    = (const uint8_t *) &col_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          remaining = 32;
                size_t          offset    = 0;
                while (remaining > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(remaining);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((const uint16_t *) (d_base + offset * row_stride), row_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + offset), vd, vl);
                    offset += vl;
                    remaining -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            // For each row i:
            //   src qs[16]: [b0|b16] [b1|b17] ... [b15|b31]  (lo nibble = b_j, hi nibble = b_{j+16})
            //   dst qs low  8B: (qs[2j] & 0x0F) | ((qs[2j+1] & 0x0F) << 4)  for j=0..7
            //   dst qs high 8B: ((qs[2j] >> 4))  | (qs[2j+1] & 0xF0)         for j=0..7
            {
                const size_t vl8 = __riscv_vsetvl_e8m1(8);
                for (int i = 0; i < 32; i++) {
                    const uint8_t * sq = col_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_bytes;

                    // stride-2 load to separate even/odd bytes
                    vuint8m1_t v_even = __riscv_vlse8_v_u8m1(sq, 2, vl8);      // qs[0], qs[2], ..., qs[14]
                    vuint8m1_t v_odd  = __riscv_vlse8_v_u8m1(sq + 1, 2, vl8);  // qs[1], qs[3], ..., qs[15]

                    // low nibble part: (even & 0x0F) | ((odd & 0x0F) << 4)
                    vuint8m1_t v_even_lo = __riscv_vand_vx_u8m1(v_even, 0x0F, vl8);
                    vuint8m1_t v_odd_lo  = __riscv_vand_vx_u8m1(v_odd, 0x0F, vl8);
                    vuint8m1_t v_lo      = __riscv_vor_vv_u8m1(v_even_lo, __riscv_vsll_vx_u8m1(v_odd_lo, 4, vl8), vl8);

                    // high nibble part: (even >> 4) | (odd & 0xF0)
                    vuint8m1_t v_even_hi = __riscv_vsrl_vx_u8m1(v_even, 4, vl8);
                    vuint8m1_t v_odd_hi  = __riscv_vand_vx_u8m1(v_odd, 0xF0, vl8);
                    vuint8m1_t v_hi      = __riscv_vor_vv_u8m1(v_even_hi, v_odd_hi, vl8);

                    __riscv_vse8_v_u8m1(dq, v_lo, vl8);
                    __riscv_vse8_v_u8m1(dq + 8, v_hi, vl8);
                }
            }

            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_1_to_q4_1_32_bl_ref(ggml_tensor *              t,
                                         int                        interleave_block,
                                         const void * GGML_RESTRICT data,
                                         size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr int nrows_interleaved = 32;

    block_q4_1x32 *    dst = (block_q4_1x32 *) t->data;
    const block_q4_1 * src = (const block_q4_1 *) data;
    block_q4_1         dst_tmp[32];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_1));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_1x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

// RVV optimized version of repack_q4_1_to_q4_1_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack + zp computation.
static int repack_q4_1_to_q4_1_32_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_block == 32);

    constexpr int nrows_interleaved = 32;
    constexpr int qs_bytes          = QK4_1 / 2;  // 16

    block_q4_1x32 *    dst     = (block_q4_1x32 *) t->data;
    const block_q4_1 * src     = (const block_q4_1 *) data;
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_1));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        return -1;
    }

    const ptrdiff_t row_stride = (ptrdiff_t) nblocks * sizeof(block_q4_1);

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            const block_q4_1 * col_src = src + x;

            // --- 1) Gather d and m, compute zp = clamp(nearbyint(-m/d), 0, 15) ---
            // block_q4_1 layout: [d(f16), m(f16), qs[16]]
            // d is at byte offset 0, m is at byte offset 2 from each block start
            {
                const uint8_t * dm_base   = (const uint8_t *) &col_src->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                ggml_half *     d_dst     = dst->d;
                uint8_t *       zp_dst    = dst->zp;
                size_t          remaining = 32;
                size_t          offset    = 0;
                while (remaining > 0) {
                    size_t vl = __riscv_vsetvl_e16m1(remaining);

                    // stride load d (f16) from each row
                    vuint16m1_t vd_raw =
                        __riscv_vlse16_v_u16m1((const uint16_t *) (dm_base + offset * row_stride), row_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + offset), vd_raw, vl);

                    // stride load m (f16) from each row (offset +2 bytes from d)
                    vuint16m1_t vm_raw =
                        __riscv_vlse16_v_u16m1((const uint16_t *) (dm_base + 2 + offset * row_stride), row_stride, vl);

                    // convert to f32 for zp computation: zp = nearbyint(-m / d)
                    vfloat16m1_t vd_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vd_raw);
                    vfloat16m1_t vm_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vm_raw);

                    // -m / d in f16 directly (SpaceMIT X60 supports f16 arithmetic)
                    vfloat16m1_t v_neg_m = __riscv_vfneg_v_f16m1(vm_f16, vl);
                    vfloat16m1_t v_ratio = __riscv_vfdiv_vv_f16m1(v_neg_m, vd_f16, vl);

                    // Convert to f32 for nearbyint, then clamp
                    vfloat32m2_t v_ratio_f32 = __riscv_vfwcvt_f_f_v_f32m2(v_ratio, vl);

                    // Use integer rounding: convert f32 -> int (rounds to nearest)
                    vint32m2_t v_zp_i32 = __riscv_vfcvt_x_f_v_i32m2(v_ratio_f32, vl);

                    // clamp to [0, 15]
                    v_zp_i32 = __riscv_vmax_vx_i32m2(v_zp_i32, 0, vl);
                    v_zp_i32 = __riscv_vmin_vx_i32m2(v_zp_i32, 15, vl);

                    // narrow i32 -> u8
                    vint16m1_t  v_zp_i16 = __riscv_vncvt_x_x_w_i16m1(v_zp_i32, vl);
                    vint8mf2_t  v_zp_i8  = __riscv_vncvt_x_x_w_i8mf2(v_zp_i16, vl);
                    vuint8mf2_t v_zp_u8  = __riscv_vreinterpret_v_i8mf2_u8mf2(v_zp_i8);
                    __riscv_vse8_v_u8mf2(zp_dst + offset, v_zp_u8, vl);

                    offset += vl;
                    remaining -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            {
                const size_t vl8 = __riscv_vsetvl_e8m1(8);
                for (int i = 0; i < 32; i++) {
                    const uint8_t * sq = col_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_bytes;

                    // stride-2 load to separate even/odd bytes
                    vuint8m1_t v_even = __riscv_vlse8_v_u8m1(sq, 2, vl8);
                    vuint8m1_t v_odd  = __riscv_vlse8_v_u8m1(sq + 1, 2, vl8);

                    // low nibble part: (even & 0x0F) | ((odd & 0x0F) << 4)
                    vuint8m1_t v_even_lo = __riscv_vand_vx_u8m1(v_even, 0x0F, vl8);
                    vuint8m1_t v_odd_lo  = __riscv_vand_vx_u8m1(v_odd, 0x0F, vl8);
                    vuint8m1_t v_lo      = __riscv_vor_vv_u8m1(v_even_lo, __riscv_vsll_vx_u8m1(v_odd_lo, 4, vl8), vl8);

                    // high nibble part: (even >> 4) | (odd & 0xF0)
                    vuint8m1_t v_even_hi = __riscv_vsrl_vx_u8m1(v_even, 4, vl8);
                    vuint8m1_t v_odd_hi  = __riscv_vand_vx_u8m1(v_odd, 0xF0, vl8);
                    vuint8m1_t v_hi      = __riscv_vor_vv_u8m1(v_even_hi, v_odd_hi, vl8);

                    __riscv_vse8_v_u8m1(dq, v_lo, vl8);
                    __riscv_vse8_v_u8m1(dq + 8, v_hi, vl8);
                }
            }

            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_k_to_q4_1_32_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr int nrows_interleaved = 32;

    block_q4_1x32 *    dst = (block_q4_1x32 *) t->data;
    const block_q4_K * src = (const block_q4_K *) data;
    block_q4_1         dst_tmp[32];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK_K;

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    const float d = GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    const float min =
                        GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    const float d1 = d * sc;
                    const float m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_TO_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_TO_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    const uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    if (j % 2 == 0) {
                        for (int ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } else {
                        for (int ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_block_q4_1x32(dst_tmp, interleave_block);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q6_k_to_q8_0_32_bl_ref(ggml_tensor *              t,
                                         int                        interleave_block,
                                         const void * GGML_RESTRICT data,
                                         size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr int nrows_interleaved = 32;

    block_q8_0x32 *    dst = (block_q8_0x32 *) t->data;
    const block_q6_K * src = (const block_q6_K *) data;
    block_q8_0         dst_tmp[32];
    int8_t             aux8[QK4_1];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK_K;

    if (t->ne[0] % QK_K != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrow_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        for (int64_t x = 0; x < nblocks; x++) {
            for (int bi = 0; bi < 8; bi++) {
                int i = 0;
                for (; i < nrow_real; i++) {
                    const uint8_t * q4     = src[x + i * nblocks].ql;
                    const uint8_t * qh     = src[x + i * nblocks].qh;
                    const int8_t *  scales = src[x + i * nblocks].scales;
                    float           d      = GGML_FP16_TO_FP32(src[x + i * nblocks].d);

                    q4 += 64 * (bi / 4);
                    qh += 32 * (bi / 4);
                    int8_t * GGML_RESTRICT a = aux8;

                    int8_t bi_idx = bi % 4;

                    if (bi_idx == 0) {
                        for (int l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        }
                    } else if (bi_idx == 1) {
                        for (int l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        }
                    } else if (bi_idx == 2) {
                        for (int l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        }
                    } else if (bi_idx == 3) {
                        for (int l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        }
                    }
                    a = aux8;

                    float a_max_abs = 0.0f;
                    float scale_0   = scales[bi * 2 + 0] * d;
                    float scale_1   = scales[bi * 2 + 1] * d;
                    for (int l = 0; l < 16; ++l) {
                        a_max_abs = std::max(a_max_abs, std::abs(a[l] * scale_0));
                    }

                    for (int l = 16; l < 32; ++l) {
                        a_max_abs = std::max(a_max_abs, std::abs(a[l] * scale_1));
                    }

                    float reflect_scale   = a_max_abs / ((1 << 7) - 1);
                    float reflect_scale_0 = scale_0 / reflect_scale;
                    float reflect_scale_1 = scale_1 / reflect_scale;

                    for (int l = 0; l < 16; ++l) {
                        float a_temp = std::clamp(std::nearbyintf(a[l] * reflect_scale_0), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_temp);
                    }

                    for (int l = 16; l < 32; ++l) {
                        float a_temp = std::clamp(std::nearbyintf(a[l] * reflect_scale_1), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_temp);
                    }

                    dst_tmp[i].d = GGML_FP32_TO_FP16(reflect_scale);

                    memcpy(dst_tmp[i].qs, a, 32 * sizeof(int8_t));
                }

                for (; i < nrows_interleaved; i++) {
                    memset(&dst_tmp[i], 0, sizeof(block_q8_0));
                }

                *dst++ = make_block_q8_0x32(dst_tmp, interleave_block);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

// RVV optimized version of repack_q6_k_to_q8_0_32_bl
// Vectorizes the Q6_K dequant -> requant pipeline using RVV intrinsics.
// For each sub-block (bi), dequant 32 Q6_K values to int6 -> apply two sub-block scales ->
// find max abs -> compute reflect_scale -> requant to int8 -> gather d with stride load.
static int repack_q6_k_to_q8_0_32_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr int nrows_interleaved = 32;

    block_q8_0x32 *    dst     = (block_q8_0x32 *) t->data;
    const block_q6_K * src     = (const block_q6_K *) data;
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK_K;

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        return -1;
    }

    const ptrdiff_t row_stride = (ptrdiff_t) nblocks * sizeof(block_q6_K);

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int bi = 0; bi < 8; bi++) {
                // --- 1) Gather 32 d values with stride load ---
                // We need to compute reflect_scale per row first, so gather d later.
                // Process each row: dequant Q6_K sub-block -> requant to Q8_0
                for (int i = 0; i < nrows_interleaved; i++) {
                    const block_q6_K * src_blk = &src[x + i * nblocks];
                    const uint8_t *    q4      = src_blk->ql + 64 * (bi / 4);
                    const uint8_t *    qh      = src_blk->qh + 32 * (bi / 4);
                    const int8_t *     scales  = src_blk->scales;
                    float              d       = GGML_FP16_TO_FP32(src_blk->d);

                    int8_t bi_idx = bi % 4;

                    // --- Dequant 32 Q6_K values to int6 (range [-32, 31]) using RVV ---
                    // vl = 32 for e8m2 (VLEN=256) or loop for smaller VLEN
                    const size_t vl16 = __riscv_vsetvl_e8m1(16);

                    vint8m1_t va_lo, va_hi;  // 16 elements each

                    if (bi_idx == 0) {
                        // a[l] = (q4[l] & 0xF) | (((qh[l] >> 0) & 3) << 4) - 32
                        vuint8m1_t vq4_lo = __riscv_vle8_v_u8m1(q4, vl16);
                        vuint8m1_t vq4_hi = __riscv_vle8_v_u8m1(q4 + 16, vl16);
                        vuint8m1_t vqh_lo = __riscv_vle8_v_u8m1(qh, vl16);
                        vuint8m1_t vqh_hi = __riscv_vle8_v_u8m1(qh + 16, vl16);

                        vuint8m1_t vlo4_lo = __riscv_vand_vx_u8m1(vq4_lo, 0x0F, vl16);
                        vuint8m1_t vlo4_hi = __riscv_vand_vx_u8m1(vq4_hi, 0x0F, vl16);
                        vuint8m1_t vh_lo   = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(vqh_lo, 0x03, vl16), 4, vl16);
                        vuint8m1_t vh_hi   = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(vqh_hi, 0x03, vl16), 4, vl16);

                        vuint8m1_t vcomb_lo = __riscv_vor_vv_u8m1(vlo4_lo, vh_lo, vl16);
                        vuint8m1_t vcomb_hi = __riscv_vor_vv_u8m1(vlo4_hi, vh_hi, vl16);

                        va_lo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_lo), 32, vl16);
                        va_hi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_hi), 32, vl16);
                    } else if (bi_idx == 1) {
                        // a[l] = (q4[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4) - 32
                        vuint8m1_t vq4_lo = __riscv_vle8_v_u8m1(q4 + 32, vl16);
                        vuint8m1_t vq4_hi = __riscv_vle8_v_u8m1(q4 + 48, vl16);
                        vuint8m1_t vqh_lo = __riscv_vle8_v_u8m1(qh, vl16);
                        vuint8m1_t vqh_hi = __riscv_vle8_v_u8m1(qh + 16, vl16);

                        vuint8m1_t vlo4_lo = __riscv_vand_vx_u8m1(vq4_lo, 0x0F, vl16);
                        vuint8m1_t vlo4_hi = __riscv_vand_vx_u8m1(vq4_hi, 0x0F, vl16);
                        vuint8m1_t vh_lo   = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_lo, 2, vl16), 0x03, vl16), 4, vl16);
                        vuint8m1_t vh_hi = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_hi, 2, vl16), 0x03, vl16), 4, vl16);

                        vuint8m1_t vcomb_lo = __riscv_vor_vv_u8m1(vlo4_lo, vh_lo, vl16);
                        vuint8m1_t vcomb_hi = __riscv_vor_vv_u8m1(vlo4_hi, vh_hi, vl16);

                        va_lo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_lo), 32, vl16);
                        va_hi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_hi), 32, vl16);
                    } else if (bi_idx == 2) {
                        // a[l] = (q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32
                        vuint8m1_t vq4_lo = __riscv_vle8_v_u8m1(q4, vl16);
                        vuint8m1_t vq4_hi = __riscv_vle8_v_u8m1(q4 + 16, vl16);
                        vuint8m1_t vqh_lo = __riscv_vle8_v_u8m1(qh, vl16);
                        vuint8m1_t vqh_hi = __riscv_vle8_v_u8m1(qh + 16, vl16);

                        vuint8m1_t vhi4_lo = __riscv_vsrl_vx_u8m1(vq4_lo, 4, vl16);
                        vuint8m1_t vhi4_hi = __riscv_vsrl_vx_u8m1(vq4_hi, 4, vl16);
                        vuint8m1_t vh_lo   = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_lo, 4, vl16), 0x03, vl16), 4, vl16);
                        vuint8m1_t vh_hi = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_hi, 4, vl16), 0x03, vl16), 4, vl16);

                        vuint8m1_t vcomb_lo = __riscv_vor_vv_u8m1(vhi4_lo, vh_lo, vl16);
                        vuint8m1_t vcomb_hi = __riscv_vor_vv_u8m1(vhi4_hi, vh_hi, vl16);

                        va_lo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_lo), 32, vl16);
                        va_hi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_hi), 32, vl16);
                    } else {  // bi_idx == 3
                        // a[l] = (q4[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32
                        vuint8m1_t vq4_lo = __riscv_vle8_v_u8m1(q4 + 32, vl16);
                        vuint8m1_t vq4_hi = __riscv_vle8_v_u8m1(q4 + 48, vl16);
                        vuint8m1_t vqh_lo = __riscv_vle8_v_u8m1(qh, vl16);
                        vuint8m1_t vqh_hi = __riscv_vle8_v_u8m1(qh + 16, vl16);

                        vuint8m1_t vhi4_lo = __riscv_vsrl_vx_u8m1(vq4_lo, 4, vl16);
                        vuint8m1_t vhi4_hi = __riscv_vsrl_vx_u8m1(vq4_hi, 4, vl16);
                        vuint8m1_t vh_lo   = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_lo, 6, vl16), 0x03, vl16), 4, vl16);
                        vuint8m1_t vh_hi = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_hi, 6, vl16), 0x03, vl16), 4, vl16);

                        vuint8m1_t vcomb_lo = __riscv_vor_vv_u8m1(vhi4_lo, vh_lo, vl16);
                        vuint8m1_t vcomb_hi = __riscv_vor_vv_u8m1(vhi4_hi, vh_hi, vl16);

                        va_lo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_lo), 32, vl16);
                        va_hi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_hi), 32, vl16);
                    }

                    // --- Widen to i16 for scaled abs computation ---
                    float scale_0 = scales[bi * 2 + 0] * d;
                    float scale_1 = scales[bi * 2 + 1] * d;

                    // Widen i8 -> i16 -> f32 for abs*scale computation
                    vint16m2_t va_lo_w = __riscv_vsext_vf2_i16m2(va_lo, vl16);
                    vint16m2_t va_hi_w = __riscv_vsext_vf2_i16m2(va_hi, vl16);

                    // Compute |a[l] * scale_0| for lo half, |a[l] * scale_1| for hi half
                    vfloat32m4_t vf_lo = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf2_i32m4(va_lo_w, vl16), vl16);
                    vfloat32m4_t vf_hi = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf2_i32m4(va_hi_w, vl16), vl16);

                    vfloat32m4_t vabs_lo = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(vf_lo, scale_0, vl16), vl16);
                    vfloat32m4_t vabs_hi = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(vf_hi, scale_1, vl16), vl16);

                    // Find max abs across both halves
                    vfloat32m4_t vabs_max = __riscv_vfmax_vv_f32m4(vabs_lo, vabs_hi, vl16);

                    // Reduce to scalar max
                    vfloat32m1_t vzero     = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                    vfloat32m1_t vmax_red  = __riscv_vfredmax_vs_f32m4_f32m1(vabs_max, vzero, vl16);
                    float        a_max_abs = __riscv_vfmv_f_s_f32m1_f32(vmax_red);

                    float reflect_scale   = a_max_abs / 127.0f;
                    float reflect_scale_0 = scale_0 / reflect_scale;
                    float reflect_scale_1 = scale_1 / reflect_scale;

                    // --- Requant: a[l] = clamp(nearbyint(a[l] * reflect_scale_x), -128, 127) ---
                    vfloat32m4_t vscaled_lo = __riscv_vfmul_vf_f32m4(vf_lo, reflect_scale_0, vl16);
                    vfloat32m4_t vscaled_hi = __riscv_vfmul_vf_f32m4(vf_hi, reflect_scale_1, vl16);

                    // fcvt.x rounds to nearest (using current rounding mode)
                    vint32m4_t vi_lo = __riscv_vfcvt_x_f_v_i32m4(vscaled_lo, vl16);
                    vint32m4_t vi_hi = __riscv_vfcvt_x_f_v_i32m4(vscaled_hi, vl16);

                    // Clamp to [-128, 127]
                    vi_lo = __riscv_vmax_vx_i32m4(vi_lo, -128, vl16);
                    vi_lo = __riscv_vmin_vx_i32m4(vi_lo, 127, vl16);
                    vi_hi = __riscv_vmax_vx_i32m4(vi_hi, -128, vl16);
                    vi_hi = __riscv_vmin_vx_i32m4(vi_hi, 127, vl16);

                    // Narrow i32 -> i16 -> i8
                    vint16m2_t vi16_lo = __riscv_vncvt_x_x_w_i16m2(vi_lo, vl16);
                    vint16m2_t vi16_hi = __riscv_vncvt_x_x_w_i16m2(vi_hi, vl16);
                    vint8m1_t  vi8_lo  = __riscv_vncvt_x_x_w_i8m1(vi16_lo, vl16);
                    vint8m1_t  vi8_hi  = __riscv_vncvt_x_x_w_i8m1(vi16_hi, vl16);

                    // Store d and qs directly into dst block
                    dst->d[i]   = GGML_FP32_TO_FP16(reflect_scale);
                    int8_t * dq = (int8_t *) dst->qs + i * QK8_0;
                    __riscv_vse8_v_i8m1(dq, vi8_lo, vl16);
                    __riscv_vse8_v_i8m1(dq + 16, vi8_hi, vl16);
                }
                dst++;
            }
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q8_0_to_q8_0_32_bl_ref(ggml_tensor *              t,
                                         int                        interleave_block,
                                         const void * GGML_RESTRICT data,
                                         size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr int nrows_interleaved = 32;

    block_q8_0x32 *    dst = (block_q8_0x32 *) t->data;
    const block_q8_0 * src = (const block_q8_0 *) data;
    block_q8_0         dst_tmp[32];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q8_0));

    if (t->ne[0] % QK8_0 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrows_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        for (int64_t x = 0; x < nblocks; x++) {
            int i = 0;
            for (; i < nrows_real; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            for (; i < nrows_interleaved; i++) {
                memset(&dst_tmp[i], 0, sizeof(block_q8_0));
            }
            *dst++ = make_block_q8_0x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

// RVV optimized version of repack_q8_0_to_q8_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes scale gather + qs copy.
static int repack_q8_0_to_q8_0_32_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q5_0 || t->type == GGML_TYPE_F32);
    GGML_ASSERT(interleave_block == 32);

    constexpr int nrows_interleaved = 32;

    block_q8_0x32 *    dst     = (block_q8_0x32 *) t->data;
    const block_q8_0 * src     = (const block_q8_0 *) data;
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q8_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        return -1;
    }

    const ptrdiff_t row_stride = (ptrdiff_t) nblocks * sizeof(block_q8_0);

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            const block_q8_0 * col_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            {
                const uint8_t * d_base    = (const uint8_t *) &col_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          remaining = 32;
                size_t          offset    = 0;
                while (remaining > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(remaining);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((const uint16_t *) (d_base + offset * row_stride), row_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + offset), vd, vl);
                    offset += vl;
                    remaining -= vl;
                }
            }

            // --- 2) Copy qs for each of the 32 rows (32 bytes per row) ---
            {
                for (int i = 0; i < 32; i++) {
                    const int8_t * sq = col_src[i * nblocks].qs;
                    int8_t *       dq = (int8_t *) dst->qs + i * QK8_0;

                    size_t len = QK8_0;
                    size_t idx = 0;
                    while (len > 0) {
                        size_t    vl = __riscv_vsetvl_e8m2(len);
                        vint8m2_t vs = __riscv_vle8_v_i8m2(sq + idx, vl);
                        __riscv_vse8_v_i8m2(dq + idx, vs, vl);
                        idx += vl;
                        len -= vl;
                    }
                }
            }

            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static spacemit_kernels::nrow_block_q5_1<32> make_block_q5_1x32(spacemit_kernels::nrow_block_q5_1<1> * in,
                                                                unsigned int blck_size_interleave) {
    spacemit_kernels::nrow_block_q5_1<32> out;
    GGML_ASSERT(QK5_1 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    for (int i = 0; i < 32; i++) {
        out.scales16[i] = in[i].scales16[0];
        out.zp[i]       = in[i].zp[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    for (int i = 0; i < 32; i++) {
        // low half [0..15]
        for (int j = 0; j < QK5_1 / 4; j++) {
            out.qs[i * QK5_1 / 2 + j] = (in[i].qs[j * 2] & 0x0F) | ((in[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        for (int j = 0; j < QK5_1 / 4; j++) {
            out.qs[i * QK5_1 / 2 + QK5_1 / 4 + j] = ((in[i].qs[j * 2] & 0xF0) >> 4) | (in[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 4; j++) {
            out.qh[i * 4 + j] = in[i].qh[j];
        }
    }

    return out;
}

static spacemit_kernels::nrow_block_q5_0<32> make_block_q5_0x32(spacemit_kernels::nrow_block_q5_0<1> * in,
                                                                unsigned int blck_size_interleave) {
    spacemit_kernels::nrow_block_q5_0<32> out;
    GGML_ASSERT(QK5_0 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    for (int i = 0; i < 32; i++) {
        out.scales16[i] = in[i].scales16[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    for (int i = 0; i < 32; i++) {
        // low half [0..15]
        for (int j = 0; j < QK5_0 / 4; j++) {
            out.qs[i * QK5_0 / 2 + j] = (in[i].qs[j * 2] & 0x0F) | ((in[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        for (int j = 0; j < QK5_0 / 4; j++) {
            out.qs[i * QK5_0 / 2 + QK5_0 / 4 + j] = ((in[i].qs[j * 2] & 0xF0) >> 4) | (in[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 4; j++) {
            out.qh[i * 4 + j] = in[i].qh[j];
        }
    }

    return out;
}

static int repack_q5_0_to_q5_0_32_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr int nrows_interleaved = 32;

    spacemit_kernels::nrow_block_q5_0<32> * dst = (spacemit_kernels::nrow_block_q5_0<32> *) t->data;
    const block_q5_0 *                      src = (const block_q5_0 *) data;
    spacemit_kernels::nrow_block_q5_0<1>    dst_tmp[32];
    int                                     nrow    = ggml_nrows(t);
    int                                     nblocks = t->ne[0] / QK5_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q5_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_0 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                const block_q5_0 & s = src[x + i * nblocks];

                dst_tmp[i].scales16[0] = s.d;
                memcpy(dst_tmp[i].qs, s.qs, sizeof(dst_tmp[i].qs));
                memcpy(dst_tmp[i].qh, s.qh, sizeof(dst_tmp[i].qh));
            }
            *dst++ = make_block_q5_0x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_q5_1_to_q5_1_32_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr int nrows_interleaved = 32;

    spacemit_kernels::nrow_block_q5_1<32> * dst = (spacemit_kernels::nrow_block_q5_1<32> *) t->data;
    const block_q5_1 *                      src = (const block_q5_1 *) data;
    spacemit_kernels::nrow_block_q5_1<1>    dst_tmp[32];
    int                                     nrow    = ggml_nrows(t);
    int                                     nblocks = t->ne[0] / QK5_1;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q5_1));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_1 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                const block_q5_1 & s = src[x + i * nblocks];

                float d = GGML_FP16_TO_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                float m = GGML_FP16_TO_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);

                if (d == 0.0f) {
                    dst_tmp[i].scales16[0] = GGML_FP32_TO_FP16(std::fabs(m));
                    dst_tmp[i].zp[0]       = m < 0.0f ? 1 : 0;
                    memset(dst_tmp[i].qh, 0, sizeof(dst_tmp[i].qh));
                    memset(dst_tmp[i].qs, m > 0.0f ? 0x11 : 0x00, sizeof(dst_tmp[i].qs));
                    continue;
                }

                float mid = std::nearbyintf(-m / d);
                mid       = std::min(31.0f, std::max(0.0f, mid));

                dst_tmp[i].scales16[0] = GGML_FP32_TO_FP16(d);
                dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                // qs: copy low 4 bits directly (same nibble packing)
                memcpy(dst_tmp[i].qs, s.qs, QK5_1 / 2);

                // qh: copy 5th bit directly
                memcpy(dst_tmp[i].qh, s.qh, 4);
            }
            *dst++ = make_block_q5_1x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_q5_k_to_q5_1_32_bl(ggml_tensor *              t,
                                     int                        interleave_block,
                                     const void * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K / QK5_1 == 8);

    constexpr int nrows_interleaved = 32;

    spacemit_kernels::nrow_block_q5_1<32> * dst = (spacemit_kernels::nrow_block_q5_1<32> *) t->data;
    const block_q5_K *                      src = (const block_q5_K *) data;
    spacemit_kernels::nrow_block_q5_1<1>    dst_tmp[32];
    int                                     nrow    = ggml_nrows(t);
    int                                     nblocks = t->ne[0] / QK_K;

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    const float d = GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    const float min =
                        GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);

                    float d1 = d * sc;
                    float m1 = min * m;

                    float mid              = std::nearbyintf(m1 / d1);
                    mid                    = std::min(31.0f, std::max(0.0f, mid));
                    dst_tmp[i].scales16[0] = GGML_FP32_TO_FP16(d1);
                    dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    const uint8_t * q = src[x + i * nblocks].qs + (j / 2) * QK5_1;
                    if (j % 2 == 0) {
                        for (int ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } else {
                        for (int ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }

                    // Extract the 5th bit (qh) for this sub-block
                    // block_q5_K.qh[32]: for sub-block j, the 5th bit is at bit position j in qh[l]
                    // qs was reordered: dst_qs maps to src weights [0,16,1,17,...,15,31]
                    // So qh must follow the same reorder to stay aligned with qs
                    // dst qh[4] = 32 bits for 32 weights in the reordered layout:
                    //   byte 0: weights 0..7   (from src_qh[0..7])
                    //   byte 1: weights 8..15  (from src_qh[8..15])
                    //   byte 2: weights 16..23 (from src_qh[16..23])
                    //   byte 3: weights 24..31 (from src_qh[24..31])
                    const uint8_t * src_qh = src[x + i * nblocks].qh;
                    for (int bi = 0; bi < 4; bi++) {
                        uint8_t qh_byte = 0;
                        for (int k = 0; k < 8; k++) {
                            int src_idx = bi * 8 + k;
                            qh_byte |= ((src_qh[src_idx] >> j) & 1) << k;
                        }
                        dst_tmp[i].qh[bi] = qh_byte;
                    }
                }
                *dst++ = make_block_q5_1x32(dst_tmp, interleave_block);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

namespace ggml::cpu::riscv64_spacemit {

static void q8_0x32_lane_arrange(ggml_tensor * t) {
    block_q8_0x32 * blk = (block_q8_0x32 *) t->data;
    const int64_t   n   = (ggml_nrows(t) + 31) / 32 * (t->ne[0] / QK8_0);
    static_assert(sizeof(blk->qs) == 4 * 256, "q8_0x32 data is four 256B register pairs");
    for (int64_t i = 0; i < n; i++) {
        for (int p = 0; p < 4; p++) {
            uint8_t * pair = blk[i].qs + p * 256;
            uint8_t   tmp[256];
            for (int o = 0; o < 16; o++) {
                memcpy(tmp + o * 16, pair + ((o % 8) * 2 + o / 8) * 16, 16);
            }
            memcpy(pair, tmp, sizeof(tmp));
        }
    }
}

static int q8_0x32_finish(ggml_tensor * t, int ret) {
    if (ret == 0) {
        q8_0x32_lane_arrange(t);
    }
    return ret;
}

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> int repack(ggml_tensor *, const void *, size_t);

template <> int repack<block_q4_0, 32, 16>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_16_bl(t, 16, data, data_size);
}

template <> int repack<block_q4_1, 32, 16>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_1_to_q4_1_16_bl(t, 16, data, data_size);
}

template <> int repack<block_q4_K, 32, 16>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_k_to_q4_1_16_bl(t, 16, data, data_size);
}

template <> int repack<block_q2_K, 256, 32>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q2_k_to_q2_k_32_bl(t, 32, data, data_size);
}

template <> int repack<block_q3_K, 256, 32>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q3_k_to_q3_k_32_bl(t, 32, data, data_size);
}

template <> int repack<block_q4_0, 32, 32>(ggml_tensor * t, const void * data, size_t data_size) {
#if 0
    return repack_q4_0_to_q4_0_32_bl_ref(t, 32, data, data_size);
#else
    return repack_q4_0_to_q4_0_32_bl(t, 32, data, data_size);
#endif
}

template <> int repack<block_q4_0, 256, 32>(ggml_tensor * t, const void * data, size_t data_size) {
#if 1
    int ret = repack_q4_0_to_q4_0_256_32_bl_ref(t, 32, data, data_size);
#else
    //return repack_q4_0_to_q4_0_256_32_bl(t, 32, data, data_size);
#endif
    if (ret == 0) {
        // MoE Q4_0 kernels consume signed nibbles: qs ^ 0x88 folds the fixed zero point into the dot.
        block_q4_0x32x256 * dst = (block_q4_0x32x256 *) t->data;
        const int64_t       n   = ggml_nrows(t) / 32 * (t->ne[0] / 256);
        for (int64_t i = 0; i < n; i++) {
            for (auto & blk : dst[i].blocks) {
                for (auto & q : blk.qs) {
                    q ^= 0x88;
                }
            }
        }
    }
    return ret;
}

template <> int repack<block_q4_1, 32, 32>(ggml_tensor * t, const void * data, size_t data_size) {
#if 0
    return repack_q4_1_to_q4_1_32_bl_ref(t, 32, data, data_size);
#else
    return repack_q4_1_to_q4_1_32_bl(t, 32, data, data_size);
#endif
}

template <> int repack<block_q4_1, 256, 32>(ggml_tensor * t, const void * data, size_t data_size) {
#if 1
    return repack_q4_0_to_q4_1_256_32_bl_ref(t, 32, data, data_size);
#else
    return repack_q4_1_to_q4_1_256_32_bl(t, 32, data, data_size);
#endif
}

template <> int repack<block_q4_K, 256, 32>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_k_to_q4_1_256_32_bl_ref(t, 32, data, data_size);
}

template <> int repack<block_q4_K, 32, 32>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_k_to_q4_1_32_bl(t, 32, data, data_size);
}

template <> int repack<block_q6_K, 32, 32>(ggml_tensor * t, const void * data, size_t data_size) {
#if 1
    return q8_0x32_finish(t, repack_q6_k_to_q8_0_32_bl_ref(t, 32, data, data_size));
#else
    return repack_q6_k_to_q8_0_32_bl(t, 32, data, data_size);
#endif
}

// Q5_0 -> Q8_0 at load time, exact: Q5_0 value d * (q - 16) with q in [0, 31] is Q8_0 with the same d and
// qs = q - 16. The result is repacked like a Q8_0 tensor and runs on the Q8_0 IME kernels.
int repack_q5_0_to_q8_0_32(ggml_tensor * t, const void * data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_0);
    const int64_t nblocks = ggml_nrows(t) * (t->ne[0] / QK5_0);
    GGML_ASSERT(t->ne[0] % QK5_0 == 0);
    GGML_ASSERT(data_size == (size_t) nblocks * sizeof(block_q5_0));

    const block_q5_0 *      src = (const block_q5_0 *) data;
    std::vector<block_q8_0> q8(nblocks);
    for (int64_t i = 0; i < nblocks; i++) {
        uint32_t qh;
        memcpy(&qh, src[i].qh, sizeof(qh));
        q8[i].d = src[i].d;
        for (int j = 0; j < QK5_0 / 2; j++) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            q8[i].qs[j]             = (int8_t) (((src[i].qs[j] & 0x0F) | xh_0) - 16);
            q8[i].qs[j + QK5_0 / 2] = (int8_t) (((src[i].qs[j] >> 4) | xh_1) - 16);
        }
    }
    return q8_0x32_finish(t, repack_q8_0_to_q8_0_32_bl(t, 32, q8.data(), q8.size() * sizeof(block_q8_0)));
}

int repack_f32_to_q8_0_32(ggml_tensor * t, const void * data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    GGML_ASSERT(t->ne[0] % QK8_0 == 0);
    const int64_t nrows = ggml_nrows(t);
    GGML_ASSERT(data_size == (size_t) nrows * t->ne[0] * sizeof(float));
    const int64_t           nb = t->ne[0] / QK8_0;
    std::vector<block_q8_0> q8(nrows * nb);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_q8_0_ref((const float *) data + r * t->ne[0], q8.data() + r * nb, t->ne[0]);
    }
    return q8_0x32_finish(t, repack_q8_0_to_q8_0_32_bl(t, 32, q8.data(), q8.size() * sizeof(block_q8_0)));
}

int unpack_q8_0_32_to_f32(const ggml_tensor * t, void * data, size_t data_size) {
    const int64_t nrows = ggml_nrows(t);
    if (t->type != GGML_TYPE_F32 || t->ne[0] % QK8_0 != 0 || t->ne[1] % 32 != 0 ||
        data_size != static_cast<size_t>(nrows) * t->ne[0] * sizeof(float)) {
        return -1;
    }

    const int64_t nblocks = t->ne[0] / QK8_0;
    const auto * src = static_cast<const block_q8_0x32 *>(t->data);
    auto * dst = static_cast<uint8_t *>(data);
    for (int64_t n0 = 0; n0 < nrows; n0 += 32) {
        for (int64_t kb = 0; kb < nblocks; ++kb) {
            const auto & block = src[(n0 / 32) * nblocks + kb];
            const auto * qs = reinterpret_cast<const int8_t *>(block.qs);
            for (int row = 0; row < 32; ++row) {
                const float d = GGML_FP16_TO_FP32(block.d[row]);
                // Each group of eight rows stores its low K16 lanes before its high K16 lanes.
                const size_t lane = (row / 8) * 256 + (row % 8) * 16;
                uint8_t * out = dst + ((n0 + row) * t->ne[0] + kb * QK8_0) * sizeof(float);
                for (int k = 0; k < 16; ++k) {
                    const float lo = d * qs[lane + k];
                    const float hi = d * qs[lane + 128 + k];
                    memcpy(out + k * sizeof(float), &lo, sizeof(lo));
                    memcpy(out + (k + 16) * sizeof(float), &hi, sizeof(hi));
                }
            }
        }
    }
    return 0;
}

template <> int repack<block_q8_0, 32, 32>(ggml_tensor * t, const void * data, size_t data_size) {
#if 1
    return q8_0x32_finish(t, repack_q8_0_to_q8_0_32_bl_ref(t, 32, data, data_size));
#else
    return repack_q8_0_to_q8_0_32_bl(t, 32, data, data_size);
#endif
}

template <> int repack<block_q5_0, 32, 32>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q5_0_to_q5_0_32_bl(t, 32, data, data_size);
}

template <> int repack<block_q5_1, 32, 32>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q5_1_to_q5_1_32_bl(t, 32, data, data_size);
}

template <> int repack<block_q5_K, 32, 32>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q5_k_to_q5_1_32_bl(t, 32, data, data_size);
}

int repack_q4_0_to_q41(ggml_tensor * t, const void * data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);

    const int64_t ne00 = t->ne[0];  // K
    const int64_t ne01 = t->ne[1];  // N
    if (ne00 % 32 != 0 || ne01 % 32 != 0) {
        return -1;
    }

    const int64_t kblocks = ne00 / 32;
    const int64_t nrows = ggml_nrows(t);

    if (data_size != static_cast<size_t>(nrows) * kblocks * sizeof(block_q4_0)) {
        return -1;
    }

    auto                dst = reinterpret_cast<Q41Block<32, 32, 2> *>(t->data);
    const block_q4_0 * src = reinterpret_cast<const block_q4_0 *>(data);

    // Q41 blocks cover 32 rows x K32; preserve each row’s scale bits and set zp=8.
    for (int64_t n0 = 0; n0 < nrows; n0 += 32) {
        for (int64_t j = 0; j < kblocks; j++) {
            Q41Block<32, 32, 2> & blk = dst[(n0 / 32) * kblocks + j];
            memset(blk.zp, 8, sizeof(blk.zp));
            for (int i = 0; i < 32; i++) {
                const block_q4_0 & b = src[(n0 + i) * kblocks + j];
                memcpy(&blk.scale[i], &b.d, sizeof(ggml_half));
                // Convert split-half nibbles (p, p+16) to adjacent pairs (2p, 2p+1).
                uint8_t e[32];
                for (int p = 0; p < 16; p++) {
                    e[p]      = static_cast<uint8_t>(b.qs[p] & 0x0F);
                    e[p + 16] = static_cast<uint8_t>((b.qs[p] >> 4) & 0x0F);
                }
                uint8_t * q41row = reinterpret_cast<uint8_t *>(blk.data) + i * 16;
                for (int p = 0; p < 16; p++) {
                    q41row[p] = static_cast<uint8_t>((e[2 * p] & 0x0F) | ((e[2 * p + 1] & 0x0F) << 4));
                }
            }
        }
    }

    return 0;
}

// Approximate offsets with uint8 zp, including values above 15; zero scale uses zp=0.
static uint8_t q41_zero_point(float scale, float offset) {
    if (scale == 0.0f) {
        return 0;
    }
    const float zp = std::round(-offset / scale);
    if (!(zp > 0.0f)) {
        return 0;
    }
    return zp >= 255.0f ? 255 : static_cast<uint8_t>(zp);
}

int repack_q4_1_to_q41(ggml_tensor * t, const void * data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    const int64_t kblocks = t->ne[0] / QK4_1;
    const int64_t nrows = ggml_nrows(t);
    if (t->ne[0] % QK4_1 != 0 || t->ne[1] % 32 != 0 ||
        data_size != static_cast<size_t>(nrows) * kblocks * sizeof(block_q4_1)) {
        return -1;
    }

    auto * dst = reinterpret_cast<Q41Block<32, 32, 2> *>(t->data);
    const auto * src = reinterpret_cast<const block_q4_1 *>(data);
    for (int64_t n0 = 0; n0 < nrows; n0 += 32) {
        for (int64_t j = 0; j < kblocks; ++j) {
            auto & blk = dst[(n0 / 32) * kblocks + j];
            for (int i = 0; i < 32; ++i) {
                const auto & b = src[(n0 + i) * kblocks + j];
                memcpy(&blk.scale[i], &b.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d, sizeof(ggml_half));
                blk.zp[i] = q41_zero_point(
                    GGML_FP16_TO_FP32(b.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d),
                    GGML_FP16_TO_FP32(b.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m));
                uint8_t * row = reinterpret_cast<uint8_t *>(blk.data) + i * 16;
                for (int p = 0; p < 8; ++p) {
                    row[p] = (b.qs[2 * p] & 15) | ((b.qs[2 * p + 1] & 15) << 4);
                    row[p + 8] = (b.qs[2 * p] >> 4) | (b.qs[2 * p + 1] & 0xf0);
                }
            }
        }
    }
    return 0;
}

// Q4_K folds each K32 scale into FP16 and rounds its offset to a saturated uint8 zp.
int repack_q4_k_to_q41(ggml_tensor * t, const void * data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    const int64_t kblocks = t->ne[0] / 32;
    const int64_t src_blocks = t->ne[0] / QK_K;
    const int64_t nrows = ggml_nrows(t);
    if (t->ne[0] % QK_K != 0 || t->ne[1] % 32 != 0 ||
        data_size != static_cast<size_t>(nrows) * src_blocks * sizeof(block_q4_K)) {
        return -1;
    }

    auto * dst = reinterpret_cast<Q41Block<32, 32, 2> *>(t->data);
    const auto * src = reinterpret_cast<const block_q4_K *>(data);
    for (int64_t n0 = 0; n0 < nrows; n0 += 32) {
        for (int64_t j = 0; j < kblocks; ++j) {
            auto & blk = dst[(n0 / 32) * kblocks + j];
            const int sub = j % (QK_K / 32);
            for (int i = 0; i < 32; ++i) {
                const auto & b = src[(n0 + i) * src_blocks + j / (QK_K / 32)];
                uint8_t sc, mn;
                get_scale_min_k4(sub, b.scales, &sc, &mn);
                const ggml_half d = GGML_FP32_TO_FP16(
                    GGML_FP16_TO_FP32(b.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d) * sc);
                const float offset = -GGML_FP16_TO_FP32(b.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin) * mn;
                memcpy(&blk.scale[i], &d, sizeof(d));
                blk.zp[i] = q41_zero_point(GGML_FP16_TO_FP32(d), offset);
                const uint8_t * qs = b.qs + (sub / 2) * 32;
                const int shift = (sub % 2) * 4;
                uint8_t * row = reinterpret_cast<uint8_t *>(blk.data) + i * 16;
                for (int p = 0; p < 16; ++p) {
                    row[p] = ((qs[2 * p] >> shift) & 15) | (((qs[2 * p + 1] >> shift) & 15) << 4);
                }
            }
        }
    }
    return 0;
}

int unpack_q41_to_q4_0_common(const ggml_tensor * t, void * data, size_t data_size) {
    static_assert(sizeof(Q41Block<32, 32, 2>) == 608, "Q41Block layout changed");

    const int64_t ne00 = t->ne[0];  // K
    if (ne00 % 32 != 0 || t->ne[1] % 32 != 0) {
        return -1;
    }
    if (t->type == GGML_TYPE_Q4_K && ne00 % QK_K != 0) {
        return -1;
    }

    const int64_t kblocks = ne00 / 32;
    const int64_t nrows = ggml_nrows(t);

    const size_t block_size = t->type == GGML_TYPE_Q4_0 ? sizeof(block_q4_0)
                               : t->type == GGML_TYPE_Q4_1 ? sizeof(block_q4_1)
                                                            : sizeof(block_q4_K);
    const int64_t src_blocks = t->type == GGML_TYPE_Q4_K ? kblocks / 8 : kblocks;
    if (data_size != static_cast<size_t>(nrows) * src_blocks * block_size) {
        return -1;
    }

    const Q41Block<32, 32, 2> * src = reinterpret_cast<const Q41Block<32, 32, 2> *>(t->data);

    if (t->type == GGML_TYPE_Q4_K) {
        // Q4_K cannot represent the Q41 grid exactly; reconstruct and requantize it.
        block_q4_K * dst = reinterpret_cast<block_q4_K *>(data);
        float        vals[QK_K];
        for (int64_t n0 = 0; n0 < nrows; n0 += 32) {
            for (int64_t jblk = 0; jblk < src_blocks; jblk++) {
                for (int i = 0; i < 32; i++) {
                    for (int jj = 0; jj < QK_K / 32; jj++) {
                        const Q41Block<32, 32, 2> & blk = src[(n0 / 32) * kblocks + jblk * 8 + jj];
                        ggml_half                s_h;
                        memcpy(&s_h, &blk.scale[i], sizeof(s_h));
                        const float    d    = GGML_FP16_TO_FP32(s_h);
                        const uint8_t  zp   = blk.zp[i];
                        const uint8_t * row = reinterpret_cast<const uint8_t *>(blk.data) + i * 16;
                        for (int p = 0; p < 16; p++) {
                            vals[jj * 32 + 2 * p]     = ((row[p] & 0x0F) - zp) * d;
                            vals[jj * 32 + 2 * p + 1] = (((row[p] >> 4) & 0x0F) - zp) * d;
                        }
                    }
                    quantize_row_q4_K_ref(vals, &dst[(n0 + i) * src_blocks + jblk], QK_K);
                }
            }
        }
        return 0;
    }

    block_q4_0 * dst0 = t->type == GGML_TYPE_Q4_0 ? reinterpret_cast<block_q4_0 *>(data) : nullptr;
    block_q4_1 * dst1 = t->type == GGML_TYPE_Q4_1 ? reinterpret_cast<block_q4_1 *>(data) : nullptr;
    for (int64_t n0 = 0; n0 < nrows; n0 += 32) {
        for (int64_t j = 0; j < kblocks; j++) {
            const Q41Block<32, 32, 2> & blk = src[(n0 / 32) * kblocks + j];
            for (int i = 0; i < 32; i++) {
                // Convert adjacent nibbles back to split-half order.
                const uint8_t * row = reinterpret_cast<const uint8_t *>(blk.data) + i * 16;
                uint8_t         e[32];
                for (int p = 0; p < 16; p++) {
                    e[2 * p]     = static_cast<uint8_t>(row[p] & 0x0F);
                    e[2 * p + 1] = static_cast<uint8_t>((row[p] >> 4) & 0x0F);
                }
                uint8_t qs[16];
                for (int p = 0; p < 16; p++) {
                    qs[p] = static_cast<uint8_t>((e[p] & 0x0F) | ((e[p + 16] & 0x0F) << 4));
                }
                if (dst0 != nullptr) {
                    block_q4_0 & b = dst0[(n0 + i) * kblocks + j];
                    memcpy(&b.d, &blk.scale[i], sizeof(ggml_half));
                    memcpy(b.qs, qs, sizeof(qs));
                } else {
                    block_q4_1 & b = dst1[(n0 + i) * kblocks + j];
                    memcpy(&b.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d, &blk.scale[i], sizeof(ggml_half));
                    memcpy(b.qs, qs, sizeof(qs));
                    ggml_half s_h;
                    memcpy(&s_h, &blk.scale[i], sizeof(s_h));
                    b.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m =
                        GGML_FP32_TO_FP16(-blk.zp[i] * GGML_FP16_TO_FP32(s_h));
                }
            }
        }
    }
    return 0;
}

}  // namespace ggml::cpu::riscv64_spacemit
