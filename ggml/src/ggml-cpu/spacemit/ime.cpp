#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ime.h"

#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ime_kernels.h"
#include "traits.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <stdexcept>
#include <thread>

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

#if defined(RISCV64_SPACEMIT_IME1)
#else
#error "RISCV64_SPACEMIT_IME1 not defined"
#endif

#else

#error "riscv not enabled in this build"

#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#if defined(RISCV64_SPACEMIT_IME1)
#define QGEMM_STRIDEN_THREAD_ALIGN 16
#else
#define QGEMM_STRIDEN_THREAD_ALIGN 32
#endif

// clang-format on

struct qnbitgemm_spacemit_ime_args {
    const float *     a_ptr               = nullptr;
    size_t            lda                 = 0;
    const std::byte * packed_quant_b_data = nullptr;
    const float *     quant_b_scale       = nullptr;
    const void *      quant_b_zp          = nullptr;
    const float *     quant_b_blksum      = nullptr;
    const float *     bias                = nullptr;
    float *           c_ptr               = nullptr;
    size_t            ldc                 = 0;
};

constexpr size_t div_round_up(size_t up, size_t down) {
    return (up + down - 1) / down;
}

constexpr size_t q8_blk_size(size_t blk_len) {
    const size_t blk_size = sizeof(float) + blk_len * sizeof(int8_t);
    // Currently, the strictest alignment requirement of a block is for a float.
    // Ensure contiguous blocks are suitably aligned.
    assert(blk_size % alignof(float) == 0);
    return blk_size;
}

namespace ggml::cpu::riscv64_spacemit {

const int num_ai_cores = std::thread::hardware_concurrency() / 2;

}  // namespace ggml::cpu::riscv64_spacemit

static void sqnbitgemm_spacemit_ime_i8i4(const size_t                        blk_len,
                                         const size_t                        gemm_k,
                                         const qnbitgemm_spacemit_ime_args * gemm_args,
                                         void * const                        per_gemm_ws,
                                         const size_t                        m_start,
                                         const size_t                        m_count,
                                         const size_t                        n_start,
                                         const size_t                        n_count) {
    constexpr size_t scale_stride = sizeof(uint16_t);
    constexpr size_t blk_bitwidth = 4;

    const size_t k_blks = div_round_up(gemm_k, blk_len);

    const size_t      lda         = k_blks * q8_blk_size(blk_len);
    const size_t      ldc         = gemm_args->ldc;
    const size_t      ldb         = k_blks * (blk_len * blk_bitwidth / 8);
    const std::byte * quant_a_ptr = static_cast<const std::byte *>(per_gemm_ws) + m_start * lda;

    const size_t      zero_point_stride   = gemm_args->quant_b_zp != nullptr ? sizeof(uint8_t) : 0;
    const size_t      packed_b_stride     = ldb + k_blks * (scale_stride + zero_point_stride);
    const std::byte * packed_quant_b_data = gemm_args->packed_quant_b_data + n_start * packed_b_stride;

    float * c_ptr = gemm_args->c_ptr + m_start * ldc + n_start;

    size_t       count_n               = 0;
    const size_t compute_block_count_n = m_count == 1 ? n_count : 16;
    for (size_t n = 0; n < n_count; n += count_n) {
        count_n = std::min(n_count - n, compute_block_count_n);

        const std::byte * a_row    = quant_a_ptr;
        const std::byte * b_col    = packed_quant_b_data + n * packed_b_stride;
        const std::byte * b_col_zp = (zero_point_stride != 0) ? b_col : nullptr;
        float *           c_blk    = c_ptr + n;

        int32_t rows_remaining = m_count;

        while (rows_remaining > 0) {
            const auto rows_handled = sqnbitgemm_spacemit_ime::ime1::gemm_kernel_i8i4(
                blk_len, a_row, b_col, nullptr, b_col_zp, c_blk, rows_remaining, count_n, gemm_k, k_blks, ldc, nullptr,
                scale_stride);

            c_blk += rows_handled * ldc;
            a_row += rows_handled * lda;

            rows_remaining -= rows_handled;
        }
    }
}

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

using block_q4_0x16 = block<4, 16>;
using block_q4_1x16 = block_with_zp<4, 16>;
using block_q8_0x16 = block<8, 16>;

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

static int repack_q4_0_to_q4_0_16_bl(struct ggml_tensor *       t,
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

static int repack_q4_1_to_q4_1_16_bl(struct ggml_tensor *       t,
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

static int repack_q4_k_to_q4_1_16_bl(struct ggml_tensor *       t,
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

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(struct ggml_tensor *, const void *, size_t);

template <> int repack<block_q4_0, 8, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_16_bl(t, 16, data, data_size);
}

template <> int repack<block_q4_1, 8, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_1_to_q4_1_16_bl(t, 16, data, data_size);
}

template <> int repack<block_q4_K, 8, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_k_to_q4_1_16_bl(t, 16, data, data_size);
}

class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(struct ggml_tensor * t, const void * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> class tensor_traits : public tensor_traits_base {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                size = ggml_row_size(GGML_TYPE_Q8_0, ggml_nelements(op->src[1])) * 4;
                size = ((size + QK4_0 - 1) / QK4_0) * (QK4_0 * sizeof(float) + sizeof(float));
                return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->type == GGML_TYPE_Q4_0 ||  //
                    op->src[0]->type == GGML_TYPE_Q4_1 ||  //
                    op->src[0]->type == GGML_TYPE_Q4_K) {
                    forward_mul_mat_q4(params, op);
                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    void forward_mul_mat_q4(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = params->ith;
        int nth = params->nth;

        [[maybe_unused]] const enum ggml_type type = src0->type;

        void *        w_data  = (void *) src0->data;
        const float * feature = (const float *) src1->data;
        float *       output  = (float *) dst->data;

        const size_t                  batch_feature = ne12 * ne13;
        [[maybe_unused]] const size_t batch_weight  = ne02 * ne03;
        const size_t                  gemm_m        = ne11;
        const size_t                  gemm_k        = ne10;
        const size_t                  gemm_n        = ne01;

        GGML_ASSERT(batch_weight == 1);

        const size_t block_count_k           = div_round_up(gemm_k, QK4_0);
        const size_t per_gemm_workspace_size = gemm_m * block_count_k * q8_blk_size(QK4_0);
        const size_t per_gemm_workspace_stride =
            div_round_up(per_gemm_workspace_size, alignof(uint64_t)) * alignof(uint64_t);
        const size_t gemm_workspace_size = batch_feature * per_gemm_workspace_stride;
        const size_t desired_wsize       = gemm_workspace_size + alignof(uint64_t) - 1;

        if (ith == 0 && params->wsize < desired_wsize) {
            throw std::runtime_error("wsize less than desired_wsize");
        }

        std::vector<qnbitgemm_spacemit_ime_args> qnbitgemm_args(batch_feature);

        for (size_t i = 0; i < batch_feature; i++) {
            qnbitgemm_args[i].a_ptr               = feature + gemm_m * gemm_k * i;
            qnbitgemm_args[i].lda                 = gemm_k;
            qnbitgemm_args[i].packed_quant_b_data = (const std::byte *) w_data;
            qnbitgemm_args[i].quant_b_scale       = nullptr;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
                qnbitgemm_args[i].quant_b_zp = nullptr;
            } else {
                qnbitgemm_args[i].quant_b_zp = w_data;
            }

            qnbitgemm_args[i].bias  = nullptr;
            qnbitgemm_args[i].c_ptr = output + gemm_m * gemm_n * i;
            qnbitgemm_args[i].ldc   = gemm_n;
        }

        const uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);
        void *          ws = reinterpret_cast<void *>((ws_ptr + alignof(uint64_t) - 1) & (~(alignof(uint64_t) - 1)));
        const size_t    quant_a_stride = block_count_k * q8_blk_size(QK4_0);

        {
            constexpr size_t block_size_m           = 4;
            size_t           per_gemm_block_count_m = div_round_up(gemm_m, block_size_m);
            int32_t          task_count             = batch_feature * per_gemm_block_count_m;
            int32_t          task_per_thread        = (task_count + nth - 1) / nth;
            int32_t          start                  = ith * task_per_thread;
            int32_t          end                    = std::min((ith + 1) * task_per_thread, task_count);
            for (int32_t compute_idx = start; compute_idx < end; compute_idx++) {
                int32_t                             gemm_idx = compute_idx / per_gemm_block_count_m;
                int32_t                             block_idx_in_gemm = compute_idx % per_gemm_block_count_m;
                int32_t                             m_idx    = block_idx_in_gemm * block_size_m;
                const qnbitgemm_spacemit_ime_args & data     = qnbitgemm_args[gemm_idx];
                int32_t rows_tobe_handled = (gemm_m - m_idx) > block_size_m ? block_size_m : (gemm_m - m_idx);

                if (rows_tobe_handled == block_size_m) {
                    const float * a_row_ptr = data.a_ptr + m_idx * data.lda;
                    std::byte *   quant_a_row_ptr =
                        static_cast<std::byte *>(ws) + gemm_idx * per_gemm_workspace_stride + m_idx * quant_a_stride;
                    sqnbitgemm_spacemit_ime::ime1::quantize_a_4row_i8(QK4_0, a_row_ptr, gemm_k, quant_a_row_ptr);
                } else {
                    while (rows_tobe_handled) {
                        const float * a_row_ptr       = data.a_ptr + m_idx * data.lda;
                        std::byte *   quant_a_row_ptr = static_cast<std::byte *>(ws) +
                                                      gemm_idx * per_gemm_workspace_stride + m_idx * quant_a_stride;
                        sqnbitgemm_spacemit_ime::ime1::quantize_a_row_i8(QK4_0, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        if (ith >= ggml::cpu::riscv64_spacemit::num_ai_cores) {
            return;
        }
        nth = std::min(nth, int{ ggml::cpu::riscv64_spacemit::num_ai_cores });

        size_t           threads_per_gemm = nth / batch_feature;
        constexpr size_t gemm_m_stride    = 128;
        size_t           nc               = gemm_n;
        const size_t     gemm_m_blocked   = div_round_up(gemm_m, gemm_m_stride);
        const size_t     max_nc           = div_round_up(gemm_n * gemm_m_blocked, threads_per_gemm);
        if (max_nc < nc) {
            nc = std::min(nc, div_round_up(max_nc, QGEMM_STRIDEN_THREAD_ALIGN) * QGEMM_STRIDEN_THREAD_ALIGN);
        }
        const size_t gemm_n_stride  = nc;
        const size_t thread_count_m = div_round_up(gemm_m, gemm_m_stride);
        const size_t thread_count_n = div_round_up(gemm_n, gemm_n_stride);
        threads_per_gemm            = thread_count_m * thread_count_n;

        {
            int task_count      = batch_feature * threads_per_gemm;
            int task_per_thread = (task_count + nth - 1) / nth;
            int start           = ith * task_per_thread;
            int end             = std::min((ith + 1) * task_per_thread, task_count);
            for (int compute_idx = start; compute_idx < end; compute_idx++) {
                const auto   gemm_i = compute_idx / threads_per_gemm;
                const auto   blk_i  = compute_idx % threads_per_gemm;
                const auto * data   = &qnbitgemm_args[gemm_i];

                const auto tid_n = blk_i / thread_count_m;
                const auto tid_m = blk_i % thread_count_m;

                const size_t m_start = tid_m * gemm_m_stride;
                const size_t m_count = std::min(gemm_m - m_start, (size_t) gemm_m_stride);

                const size_t n_start = tid_n * gemm_n_stride;
                const size_t n_count = std::min(gemm_n - n_start, (size_t) gemm_n_stride);

                void * per_gemm_ws = reinterpret_cast<std::byte *>(ws) + gemm_i * per_gemm_workspace_stride;

                sqnbitgemm_spacemit_ime_i8i4(QK4_0, gemm_k, data, per_gemm_ws, m_start, m_count, n_start, n_count);
            }
        }
    }

    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

class tensor_traits_common : public tensor_traits_base {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        switch (op->op) {
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
                size = 0;
                return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_NORM:
                forward_norm_f32(params, op);
                return true;
            case GGML_OP_RMS_NORM:
                forward_rms_norm_f32(params, op);
                return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    void forward_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        ggml_tensor *       dst  = op;
        GGML_ASSERT(ggml_are_same_shape(src0, dst));
        GGML_ASSERT(src0->nb[0] == sizeof(float));

        const int ith = params->ith;
        const int nth = params->nth;

        GGML_TENSOR_UNARY_OP_LOCALS

        float epsilon;
        memcpy(&epsilon, dst->op_params, sizeof(float));

        GGML_ASSERT(epsilon > 0.0f);

        auto * input  = (float *) src0->data;
        auto * output = (float *) dst->data;

        const auto hidden_size     = ne00;
        const auto task_count      = ne01 * ne02 * ne03;
        const auto task_per_thread = (task_count + nth - 1) / nth;

        const auto task_begin = ith * task_per_thread;
        const auto task_end   = std::min((ith + 1) * task_per_thread, task_count);

        for (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
            auto   offset  = task_idx * hidden_size;
            auto * p_input = const_cast<float *>(input + offset);

            auto *       p_output      = output + offset;
            auto *       p_temp_output = p_output;
            auto *       p_gamma_data  = (const float *) nullptr;
            auto *       p_beta_data   = (const float *) nullptr;
            size_t       gvl           = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t sum           = __riscv_vfmv_v_f_f32m4(0.f, gvl);
            vfloat32m4_t sum_sq        = __riscv_vfmv_v_f_f32m4(0.f, gvl);
            int64_t      length        = hidden_size;
            while (length > 0) {
                gvl                   = __riscv_vsetvl_e32m4(length);
                // load data
                vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_input, gvl);

                sum    = __riscv_vfadd_vv_f32m4(sum, src_data, gvl);
                sum_sq = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);

                __riscv_vse32_v_f32m4(p_temp_output, src_data, gvl);

                p_input += gvl;
                p_temp_output += gvl;
                length -= gvl;
            }

            gvl = __riscv_vsetvlmax_e32m1();

            float        mean   = 0.f;
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
            vfloat32m1_t mean_v =
                __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum, 0), __riscv_vget_v_f32m4_f32m1(sum, 1), gvl);
            mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(sum, 2), gvl);
            mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(sum, 3), gvl);
            mean_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_v, zero_v, gvl);
            mean   = __riscv_vfmv_f_s_f32m1_f32(mean_v);
            mean /= hidden_size;

            vfloat32m1_t mean_square_v = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum_sq, 0),
                                                                __riscv_vget_v_f32m4_f32m1(sum_sq, 1), gvl);
            mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 2), gvl);
            mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 3), gvl);
            mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

            float mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
            mean_square /= hidden_size;
            mean_square = sqrt(mean_square - mean * mean + epsilon);

            mean_square   = 1.0f / mean_square;
            length        = hidden_size;
            p_temp_output = p_output;

            if (p_gamma_data == nullptr && p_beta_data == nullptr) {
                while (length > 0) {
                    gvl                   = __riscv_vsetvl_e32m4(length);
                    vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                    src_data              = __riscv_vfsub_vf_f32m4(src_data, mean, gvl);
                    src_data              = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                    __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                    p_temp_output += gvl;
                    p_output += gvl;
                    length -= gvl;
                }
            } else if (p_beta_data == nullptr) {
                while (length > 0) {
                    gvl                       = __riscv_vsetvl_e32m4(length);
                    vfloat32m4_t src_data     = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                    vfloat32m4_t gamma_data_v = __riscv_vle32_v_f32m4(p_gamma_data, gvl);
                    src_data                  = __riscv_vfsub_vf_f32m4(src_data, mean, gvl);
                    src_data                  = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                    src_data                  = __riscv_vfmul_vv_f32m4(src_data, gamma_data_v, gvl);
                    __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                    p_temp_output += gvl;
                    p_output += gvl;
                    p_gamma_data += gvl;
                    length -= gvl;
                }
            } else if (p_gamma_data != nullptr) {
                while (length > 0) {
                    gvl                       = __riscv_vsetvl_e32m4(length);
                    vfloat32m4_t src_data     = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                    vfloat32m4_t gamma_data_v = __riscv_vle32_v_f32m4(p_gamma_data, gvl);
                    src_data                  = __riscv_vfsub_vf_f32m4(src_data, mean, gvl);
                    src_data                  = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                    src_data                  = __riscv_vfmul_vv_f32m4(src_data, gamma_data_v, gvl);
                    vfloat32m4_t beta_data_v  = __riscv_vle32_v_f32m4(p_beta_data, gvl);
                    src_data                  = __riscv_vfadd_vv_f32m4(src_data, beta_data_v, gvl);
                    p_beta_data += gvl;
                    __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                    p_temp_output += gvl;
                    p_output += gvl;
                    p_gamma_data += gvl;
                    length -= gvl;
                }
            }
        }
    }

    void forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        ggml_tensor *       dst  = op;
        GGML_ASSERT(ggml_are_same_shape(src0, dst));
        GGML_ASSERT(src0->nb[0] == sizeof(float));

        const int ith = params->ith;
        const int nth = params->nth;

        GGML_TENSOR_UNARY_OP_LOCALS

        float epsilon;
        memcpy(&epsilon, dst->op_params, sizeof(float));

        GGML_ASSERT(epsilon > 0.0f);

        auto * input  = (float *) src0->data;
        auto * output = (float *) dst->data;

        const auto hidden_size     = ne00;
        const auto task_count      = ne01 * ne02 * ne03;
        const auto task_per_thread = (task_count + nth - 1) / nth;

        const auto task_begin = ith * task_per_thread;
        const auto task_end   = std::min((ith + 1) * task_per_thread, task_count);

        for (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
            auto   offset        = task_idx * hidden_size;
            auto * p_input       = const_cast<float *>(input + offset);
            auto * p_output      = output + offset;
            auto * p_temp_output = p_output;
            auto * p_gamma_data  = (const float *) nullptr;
            auto * p_beta_data   = (const float *) nullptr;

            size_t       gvl    = __riscv_vsetvlmax_e32m4();
            // vfloat32m4_t sum = __riscv_vfmv_v_f_f32m4(0.f, gvl);
            vfloat32m4_t sum_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
            int64_t      length = hidden_size;
            while (length > 0) {
                gvl                   = __riscv_vsetvl_e32m4(length);
                // load data
                vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_input, gvl);

                sum_sq = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);

                __riscv_vse32_v_f32m4(p_temp_output, src_data, gvl);

                p_input += gvl;
                p_temp_output += gvl;
                length -= gvl;
            }

            gvl = __riscv_vsetvlmax_e32m1();

            // float mean = 0.f;
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);

            vfloat32m1_t mean_square_v = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum_sq, 0),
                                                                __riscv_vget_v_f32m4_f32m1(sum_sq, 1), gvl);
            mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 2), gvl);
            mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 3), gvl);
            mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

            float mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
            mean_square /= hidden_size;

            mean_square = sqrt(mean_square + epsilon);

            mean_square   = 1.0f / mean_square;
            length        = hidden_size;
            p_temp_output = p_output;

            if (p_gamma_data == nullptr && p_beta_data == nullptr) {
                while (length > 0) {
                    gvl                   = __riscv_vsetvl_e32m4(length);
                    vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                    src_data              = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                    __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                    p_temp_output += gvl;
                    p_output += gvl;
                    length -= gvl;
                }
            } else if (p_beta_data == nullptr) {
                while (length > 0) {
                    gvl                       = __riscv_vsetvl_e32m4(length);
                    vfloat32m4_t src_data     = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                    vfloat32m4_t gamma_data_v = __riscv_vle32_v_f32m4(p_gamma_data, gvl);
                    src_data                  = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                    src_data                  = __riscv_vfmul_vv_f32m4(src_data, gamma_data_v, gvl);
                    __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                    p_temp_output += gvl;
                    p_output += gvl;
                    p_gamma_data += gvl;
                    length -= gvl;
                }
            } else if (p_gamma_data != nullptr) {
                while (length > 0) {
                    gvl                       = __riscv_vsetvl_e32m4(length);
                    vfloat32m4_t src_data     = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                    vfloat32m4_t gamma_data_v = __riscv_vle32_v_f32m4(p_gamma_data, gvl);
                    src_data                  = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                    src_data                  = __riscv_vfmul_vv_f32m4(src_data, gamma_data_v, gvl);
                    vfloat32m4_t beta_data_v  = __riscv_vle32_v_f32m4(p_beta_data, gvl);
                    src_data                  = __riscv_vfadd_vv_f32m4(src_data, beta_data_v, gvl);
                    p_beta_data += gvl;
                    __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                    p_temp_output += gvl;
                    p_output += gvl;
                    p_gamma_data += gvl;
                    length -= gvl;
                }
            }
        }
    }

    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        return 0;
    }
};

static const tensor_traits<block_q4_0, 8, 16> q4_0_16x8_q8_0;
static const tensor_traits<block_q4_1, 8, 16> q4_1_16x8_q8_0;
static const tensor_traits<block_q4_K, 8, 16> q4_k_16x8_q8_0;
static const tensor_traits_common             rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

static const ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(const struct ggml_tensor * cur) {
    if (cur->type == GGML_TYPE_Q4_0) {
        if (cur->ne[1] % 16 == 0) {
            return &ggml::cpu::riscv64_spacemit::q4_0_16x8_q8_0;
        }
    } else if (cur->type == GGML_TYPE_Q4_1) {
        if (cur->ne[1] % 16 == 0) {
            return &ggml::cpu::riscv64_spacemit::q4_1_16x8_q8_0;
        }
    } else if (cur->type == GGML_TYPE_Q4_K) {
        if (cur->ne[1] % 16 == 0) {
            return &ggml::cpu::riscv64_spacemit::q4_k_16x8_q8_0;
        }
    } else if (cur->type == GGML_TYPE_F32) {
        return &ggml::cpu::riscv64_spacemit::rvv_impl;
    }

    return nullptr;
}

static enum ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         struct ggml_tensor *  tensor) {
    tensor->extra =
        (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            struct ggml_tensor *  tensor,
                                                            const void *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    if (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_riscv64_spacemit_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_riscv64_spacemit_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_buffer_type_t buft,
                                                       const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    size_t       nbytes;
    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
        }
    } else {
        nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
        if (tensor->type == GGML_TYPE_Q4_K) {
            GGML_ASSERT(nbytes % sizeof(block_q4_K) == 0);
            nbytes = (nbytes / sizeof(block_q4_K)) * sizeof(block_q4_1) * 8;
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                nbytes += (tensor->ne[i] - 1) * (tensor->nb[i] / sizeof(block_q4_K)) * sizeof(block_q4_1) * 8;
            }
        } else {
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
            }
        }
    }

    GGML_UNUSED(buft);
    return nbytes;
}

namespace ggml::cpu::riscv64_spacemit {

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        return false;
                    }
                    if (op->src[1]->type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    return true;
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                    return (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
                return (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            default:
                // GGML_ABORT("fatal error");
                break;
        }

        return nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name         = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name,
         /* .alloc_buffer     = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer,
         /* .get_alignment    = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment,
         /* .get_max_size     = */ nullptr,
         /* .get_alloc_size   = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .is_host          = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        new ggml::cpu::riscv64_spacemit::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}
