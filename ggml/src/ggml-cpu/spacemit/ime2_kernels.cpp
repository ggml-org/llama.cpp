#include "ggml-impl.h"
#include "ggml.h"
#include "ime_kernels.h"
#include "rvv_kernels.h"
#include "string.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#else
#    include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#endif

#if defined(RISCV64_SPACEMIT_IME2)
#else
#    error "RISCV64_SPACEMIT_IME2 not defined"
#endif

#if defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace spacemit_kernels {
namespace ime2 {

template <size_t MB_ROWS, size_t NB_COLS>
void gemm_kernel_i8i2k_mrow_ref(size_t          blk_len,
                                const uint8_t * quant_a_ptr,
                                const uint8_t * quant_b_data,
                                float *         c_ptr,
                                size_t          count_m,
                                size_t          count_n,
                                size_t          k_blks,
                                size_t          ldc) {
    using blk_type                 = nrow_block_q2_k<NB_COLS>;
    constexpr float refactor_scale = 16.0f;
    constexpr float factor_scale   = 1.0f / refactor_scale;

    int64_t a_blk_stride        = q8k_blk_size(256);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t b_ncol_block_stride = sizeof(blk_type);

    float      output[MB_ROWS * NB_COLS]     = { 0 };
    _Float16   output_f16[MB_ROWS * NB_COLS] = { 0 };
    blk_type * quant_b_blk_data              = (blk_type *) (quant_b_data);

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);

        int8_t * a_data = (int8_t *) quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS * 16;

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < NB_COLS; ci++) {
                output[ci + mi * NB_COLS] = 0;
            }
        }

        for (size_t ki = 0; ki < k_blks; ki++, quant_b_blk_data++, a_data += a_nrow_block_stride) {
            uint8_t * b_data   = quant_b_blk_data->qs;
            uint8_t * scales   = quant_b_blk_data->scales;
            uint8_t * scales16 = (uint8_t *) (quant_b_blk_data->scales16);
            uint8_t * zeros16  = (uint8_t *) (quant_b_blk_data->zeros16);

            _Float16 * scales_fp16 = (_Float16 *) scales16;
            _Float16 * zeros_fp16  = (_Float16 *) zeros16;

            float *   a_scale_row = (float *) (a_data - sizeof(float) * MB_ROWS - sizeof(int16_t) * MB_ROWS * 16);
            int16_t * a_sum_row   = (int16_t *) (a_data - sizeof(int16_t) * MB_ROWS * 16);

            memset(output_f16, 0, sizeof(output_f16));

            uint8_t * scales_temp = scales;
            uint8_t * zps_temp    = scales;
            for (size_t kii = 0; kii < 16; kii++, scales_temp += NB_COLS, zps_temp++) {
                size_t b_shift = (kii % 4) * 2;

                uint8_t * b_data_col = b_data + (kii / 4) * NB_COLS * 16;

                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    int16_t a_sum = a_sum_row[mi * 16 + kii];
                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        _Float16 acc_0 = 0.0;

                        uint8_t b_zp    = zps_temp[ci * 16] >> 4;
                        uint8_t b_scale = scales_temp[ci] & 0x0F;
                        for (size_t bi = 0; bi < 16; bi++) {
                            int8_t  a0 = a_data[mi * 256 + bi + kii * 16];
                            uint8_t b0 = b_data_col[ci * 16 + bi];
                            acc_0 += static_cast<int16_t>(a0) * static_cast<int16_t>((b0 >> b_shift) & 0x03);
                        }

                        _Float16 scale_item =
                            static_cast<_Float16>(b_scale) * static_cast<_Float16>(factor_scale) * scales_fp16[ci];

                        output_f16[ci + mi * NB_COLS] += acc_0 * scale_item;
                        output[ci + mi * NB_COLS] += b_zp * a_sum * a_scale_row[mi] * zeros_fp16[ci];
                    }
                }
            }

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                auto a_scale = a_scale_row[mi] * refactor_scale;
                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    output[ci + mi * NB_COLS] += output_f16[ci + mi * NB_COLS] * a_scale;
                }
            }
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < nb_real; ci++) {
                c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}

template <size_t MB_ROWS, size_t NB_COLS>
void gemm_kernel_i8i3k_mrow_ref(size_t          blk_len,
                                const uint8_t * quant_a_ptr,
                                const uint8_t * quant_b_data,
                                float *         c_ptr,
                                size_t          count_m,
                                size_t          count_n,
                                size_t          k_blks,
                                size_t          ldc) {
    using blk_type                 = nrow_block_q2_k<NB_COLS>;
    constexpr float refactor_scale = 16.0f;
    constexpr float factor_scale   = 1.0f / refactor_scale;

    int64_t a_blk_stride        = q8k_blk_size(256);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t b_ncol_block_stride = sizeof(blk_type);

    float    output[MB_ROWS * NB_COLS]     = { 0 };
    _Float16 output_f16[MB_ROWS * NB_COLS] = { 0 };

    blk_type * quant_b_blk_data = (blk_type *) (quant_b_data);

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);

        int8_t * a_data = (int8_t *) quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS * 16;

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < NB_COLS; ci++) {
                output[ci + mi * NB_COLS] = 0;
            }
        }

        for (size_t ki = 0; ki < k_blks; ki++, quant_b_blk_data++, a_data += a_nrow_block_stride) {
            uint8_t * b_data   = quant_b_blk_data->qs;
            uint8_t * b_hmask  = quant_b_blk_data->hmask;
            int8_t *  scales   = quant_b_blk_data->scales;
            uint8_t * scales16 = (uint8_t *) (quant_b_blk_data->scales16);

            _Float16 * scales_fp16 = (_Float16 *) scales16;

            float *   a_scale_row = (float *) (a_data - sizeof(float) * MB_ROWS - sizeof(int16_t) * MB_ROWS * 16);
            int16_t * a_sum_row   = (int16_t *) (a_data - sizeof(int16_t) * MB_ROWS * 16);

            memset(output_f16, 0, sizeof(output_f16));

            int8_t *   scales_temp = scales;
            uint16_t * b_mask_col  = (uint16_t *) b_hmask;

            float acc_0_max = 0.0f;
            for (size_t kii = 0; kii < 16; kii++, scales_temp += NB_COLS, b_mask_col += NB_COLS) {
                size_t b_shift = (kii % 4) * 2;

                uint8_t * b_data_col = b_data + (kii / 4) * NB_COLS * 16;

                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        _Float16 acc_0        = 0;
                        // blk 2 * kii + 0
                        uint16_t b_shift_mask = 1;
                        for (size_t bi = 0; bi < 16; bi++, b_shift_mask <<= 1) {
                            int8_t a0 = a_data[mi * 256 + bi + kii * 16];
                            int8_t b0 = static_cast<int8_t>((b_data_col[ci * 16 + bi] >> b_shift) & 0x03);
                            b0 -= b_mask_col[ci] & b_shift_mask ? 0 : 4;
                            acc_0 += static_cast<int16_t>(a0) * static_cast<int16_t>(b0);
                        }

                        _Float16 scale_item = static_cast<_Float16>(scales_temp[ci]) * scales_fp16[ci] *
                                              static_cast<_Float16>(factor_scale);

                        output_f16[ci + mi * NB_COLS] += acc_0 * scale_item;
                    }
                }
            }

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                auto a_scale = a_scale_row[mi] * refactor_scale;
                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    output[ci + mi * NB_COLS] += output_f16[ci + mi * NB_COLS] * a_scale;
                }
            }
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < nb_real; ci++) {
                c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}

template <size_t MB_ROWS, size_t NB_COLS>
void gemm_kernel_i8i4_mrow_ref(size_t          blk_len,
                               const uint8_t * quant_a_ptr,
                               const uint8_t * quant_b_data,
                               const uint8_t * quant_b_zp,
                               float *         c_ptr,
                               size_t          count_m,
                               size_t          count_n,
                               size_t          k_blks,
                               size_t          ldc) {
    constexpr size_t kblks_per_blk = 16;
    GGML_ASSERT(k_blks % kblks_per_blk == 0);

    int64_t b_blk_stride        = (sizeof(_Float16) + (blk_len / 2) + (quant_b_zp ? sizeof(uint8_t) : 0));
    int64_t b_stride            = k_blks * b_blk_stride;
    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t b_ncol_block_stride = b_blk_stride * NB_COLS;

    float    output[MB_ROWS * NB_COLS]     = { 0 };
    _Float16 output_f16[MB_ROWS * NB_COLS] = { 0 };

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
        uint8_t * b_data  = (uint8_t *) quant_b_data + ni * b_stride + NB_COLS * sizeof(_Float16);
        if (quant_b_zp) {
            b_data += NB_COLS * sizeof(uint8_t);
        }

        int8_t * a_data = (int8_t *) quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS;

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < NB_COLS; ci++) {
                output[ci + mi * NB_COLS]     = 0.0f;
                output_f16[ci + mi * NB_COLS] = static_cast<_Float16>(0.0f);
            }
        }

        size_t kii = 0;
        for (size_t ki = 0; ki < k_blks; ki++, a_data += a_nrow_block_stride, b_data += b_ncol_block_stride) {
            _Float16 * b_scale_fp16 = (_Float16 *) (b_data - NB_COLS * sizeof(_Float16));
            uint8_t *  b_zp         = nullptr;
            if (quant_b_zp) {
                b_scale_fp16 = (_Float16 *) (b_data - NB_COLS * sizeof(_Float16) - NB_COLS * sizeof(uint8_t));
                b_zp         = (uint8_t *) (b_data - NB_COLS * sizeof(uint8_t));
            }

            float *   a_scale_row = (float *) (a_data - sizeof(float) * MB_ROWS - sizeof(int16_t) * MB_ROWS);
            int16_t * a_sum_row   = (int16_t *) (a_data - sizeof(int16_t) * MB_ROWS);

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                _Float16 a_scale = a_scale_row[mi];
                int16_t  a_sum   = a_sum_row[mi];

                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    _Float16 b_scale = b_scale_fp16[ci];
                    int32_t  acc     = 0;
                    if (b_zp) {
                        acc += a_sum * b_zp[ci];
                    } else {
                        acc += a_sum * 8;
                    }
                    for (size_t bi = 0; bi < blk_len / 2; bi++) {
                        int8_t  a0 = a_data[mi * blk_len + 2 * bi];
                        int8_t  a1 = a_data[mi * blk_len + 2 * bi + 1];
                        uint8_t b  = b_data[ci * blk_len / 2 + bi];
                        int8_t  b0 = static_cast<int8_t>(b & 0x0F);
                        int8_t  b1 = static_cast<int8_t>((b & 0xF0) >> 4);
                        acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0) +
                               static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
                    }
                    output_f16[ci + mi * NB_COLS] +=
                        static_cast<float>(acc) * static_cast<float>(a_scale) * static_cast<float>(b_scale);
                }
            }

            if (kii == kblks_per_blk - 1) {
                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        output[ci + mi * NB_COLS] += static_cast<float>(output_f16[ci + mi * NB_COLS]);
                        output_f16[ci + mi * NB_COLS] = 0.0f;
                    }
                }
                kii = 0;
            } else {
                kii++;
            }
        }

        if (kii == kblks_per_blk - 1) {
            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    output[ci + mi * NB_COLS] += static_cast<float>(output_f16[ci + mi * NB_COLS]);
                    output_f16[ci + mi * NB_COLS] = 0.0f;
                }
            }
            kii = 0;
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < nb_real; ci++) {
                c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}

template <size_t MB_ROWS, size_t NB_COLS>
void gemm_kernel_i8i4_hp_mrow_ref(size_t          blk_len,
                                  const uint8_t * quant_a_ptr,
                                  const uint8_t * quant_b_data,
                                  const uint8_t * quant_b_zp,
                                  float *         c_ptr,
                                  size_t          count_m,
                                  size_t          count_n,
                                  size_t          k_blks,
                                  size_t          ldc) {
    constexpr size_t k_subblks_per_superblk = 8;

    struct block_q4_0x32_layout {
        _Float16 d[NB_COLS];
        uint8_t  qs[16 * NB_COLS];
    };

    GGML_ASSERT(blk_len == 256);

    const size_t b_superblk_stride = sizeof(block_q4_0x32_layout) * k_subblks_per_superblk +
                                     (quant_b_zp ? NB_COLS * k_subblks_per_superblk * sizeof(uint8_t) : 0);
    const size_t b_tile_stride = k_blks * b_superblk_stride;

    const size_t a_nrow_block_stride = q8_hp_blk_size(blk_len, true, true) * MB_ROWS;
    const size_t a_subblk_stride     = q8_hp_blk_size(32, false, false) * MB_ROWS;

    float output[MB_ROWS * NB_COLS] = { 0 };
    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t          nb_real     = std::min<size_t>(NB_COLS, count_n - ni);
        const uint8_t * b_tile_base = quant_b_data + (ni / NB_COLS) * b_tile_stride;
        int8_t *        a_data      = (int8_t *) quant_a_ptr;

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < NB_COLS; ci++) {
                output[ci + mi * NB_COLS] = 0.0f;
            }
        }

        for (size_t ki = 0; ki < k_blks; ki++, a_data += a_nrow_block_stride) {
            _Float16 output_f16[MB_ROWS * NB_COLS] = { 0 };

            const uint8_t *              b_superblk_ptr = b_tile_base + ki * b_superblk_stride;
            const block_q4_0x32_layout * b_blocks = reinterpret_cast<const block_q4_0x32_layout *>(b_superblk_ptr);
            const uint8_t *              b_zps =
                quant_b_zp ? b_superblk_ptr + sizeof(block_q4_0x32_layout) * k_subblks_per_superblk : nullptr;

            _Float16 * a_sum_row       = (_Float16 *) (a_data + a_subblk_stride * k_subblks_per_superblk);
            _Float16 * a_scale_avg_row = (_Float16 *) (a_data + a_nrow_block_stride - sizeof(_Float16) * MB_ROWS);
            _Float16   scale_factor    = a_scale_avg_row[0];

            for (size_t ksi = 0; ksi < k_subblks_per_superblk; ++ksi) {
                const _Float16 * a_scale_row = reinterpret_cast<const _Float16 *>(a_data + a_subblk_stride * ksi);
                int8_t *         a_subblk    = a_data + a_subblk_stride * ksi + MB_ROWS * sizeof(_Float16);
                const _Float16   a_scale     = a_scale_row[0];
                const block_q4_0x32_layout & b_block = b_blocks[ksi];

                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        const uint8_t * b_qs    = b_block.qs + ci * 16;
                        _Float16        b_scale = b_block.d[ci] * a_scale;

                        int16_t acc = 0;
                        for (size_t bi = 0; bi < 16; bi++) {
                            uint8_t b  = b_qs[bi];
                            int8_t  b0 = static_cast<int8_t>(b & 0x0F);
                            int8_t  b1 = static_cast<int8_t>((b & 0xF0) >> 4);

                            acc += static_cast<int16_t>(a_subblk[mi * 32 + 2 * bi]) * static_cast<int16_t>(b0) +
                                   static_cast<int16_t>(a_subblk[mi * 32 + 2 * bi + 1]) * static_cast<int16_t>(b1);
                        }

                        const _Float16 scaled_acc = static_cast<_Float16>(acc) * b_scale;
                        output_f16[ci + mi * NB_COLS] += scaled_acc;
                    }
                }
            }

            for (size_t ksi = 0; ksi < k_subblks_per_superblk; ++ksi) {
                const _Float16 * a_scale_row = reinterpret_cast<const _Float16 *>(a_data + a_subblk_stride * ksi);
                const block_q4_0x32_layout & b_block  = b_blocks[ksi];
                const uint8_t *              b_zp_row = b_zps ? b_zps + ksi * NB_COLS : nullptr;
                const _Float16               a_scale  = a_scale_row[0];

                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    const _Float16 a_sum = a_sum_row[mi * k_subblks_per_superblk + ksi];
                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        _Float16 b_scale   = b_block.d[ci] * a_scale;
                        _Float16 a_sum_bzp = a_sum;
                        if (b_zp_row) {
                            a_sum_bzp = a_sum * static_cast<_Float16>(0.125f) * static_cast<_Float16>(b_zp_row[ci]);
                        }

                        const _Float16 scaled_acc = a_sum_bzp * b_scale;
                        output[ci + mi * NB_COLS] += scaled_acc * scale_factor;
                    }
                }
            }

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    auto val = static_cast<float>(output_f16[ci + mi * NB_COLS]) * static_cast<float>(scale_factor);
                    output[ci + mi * NB_COLS] += val;
                }
            }
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < nb_real; ci++) {
                c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}

template <size_t MB_ROWS, size_t NB_COLS>
void moe_gemm_kernel_i8i4_mrow_ref(size_t           blk_len,
                                   const uint8_t ** quant_a_ptr,
                                   const uint8_t *  quant_b_data,
                                   const uint8_t *  quant_b_zp,
                                   float **         c_ptr,
                                   size_t           count_m,
                                   size_t           count_n,
                                   size_t           k_blks,
                                   size_t           ldc) {
    int64_t b_blk_stride        = (sizeof(ggml_fp16_t) + (blk_len / 2) + (quant_b_zp ? sizeof(uint8_t) : 0));
    int64_t b_stride            = k_blks * b_blk_stride;
    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t b_ncol_block_stride = b_blk_stride * NB_COLS;

    float                         output[MB_ROWS * NB_COLS] = { 0 };
    std::array<int8_t *, MB_ROWS> a_data;
    std::array<float *, MB_ROWS>  c_data;

    for (size_t mi = 0; mi < MB_ROWS; mi++) {
        c_data[mi] = c_ptr[mi];
    }

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
        uint8_t * b_data  = (uint8_t *) quant_b_data + ni * b_stride + NB_COLS * sizeof(ggml_fp16_t);
        if (quant_b_zp) {
            b_data += NB_COLS * sizeof(uint8_t);
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            a_data[mi] = (int8_t *) quant_a_ptr[mi] + sizeof(float) + sizeof(int16_t);
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < NB_COLS; ci++) {
                output[ci + mi * NB_COLS] = 0;
            }
        }

        for (size_t ki = 0; ki < k_blks; ki++, b_data += b_ncol_block_stride) {
            ggml_fp16_t * b_scale_fp16 = (ggml_fp16_t *) (b_data - NB_COLS * sizeof(ggml_fp16_t));
            uint8_t *     b_zp         = nullptr;
            if (quant_b_zp) {
                b_scale_fp16 = (ggml_fp16_t *) (b_data - NB_COLS * sizeof(ggml_fp16_t) - NB_COLS * sizeof(uint8_t));
                b_zp         = (uint8_t *) (b_data - NB_COLS * sizeof(uint8_t));
            }

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                float *   a_scale_row = (float *) (a_data[mi] - sizeof(float) - sizeof(int16_t));
                int16_t * a_sum_row   = (int16_t *) (a_data[mi] - sizeof(int16_t));

                float   a_scale = *a_scale_row;
                int16_t a_sum   = *a_sum_row;

                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    float   b_scale = ggml_fp16_to_fp32(b_scale_fp16[ci]);
                    int32_t acc     = 0;
                    if (b_zp) {
                        acc += a_sum * b_zp[ci];
                    } else {
                        acc += a_sum * 8;
                    }
                    for (size_t bi = 0; bi < blk_len / 2; bi++) {
                        int8_t  a0 = (a_data[mi])[2 * bi];
                        int8_t  a1 = (a_data[mi])[2 * bi + 1];
                        uint8_t b  = b_data[ci * blk_len / 2 + bi];
                        int8_t  b0 = static_cast<int8_t>(b & 0x0F);
                        int8_t  b1 = static_cast<int8_t>((b & 0xF0) >> 4);
                        acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0) +
                               static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
                    }
                    output[ci + mi * NB_COLS] += static_cast<float>(acc) * a_scale * b_scale;
                }
            }

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                a_data[mi] += a_blk_stride;
            }
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < nb_real; ci++) {
                (c_data[mi])[ci] = output[mi * NB_COLS + ci];
            }
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            c_data[mi] += NB_COLS;
        }
    }
}

template <size_t MB_ROWS, size_t NB_COLS>
void moe_gemm_kernel_i8i5_mrow_ref(size_t           blk_len,
                                   const uint8_t ** quant_a_ptr,
                                   const uint8_t *  quant_b_data,
                                   const uint8_t *  quant_b_zp,
                                   float **         c_ptr,
                                   size_t           count_m,
                                   size_t           count_n,
                                   size_t           k_blks,
                                   size_t           ldc) {
    GGML_UNUSED(count_m);
    GGML_UNUSED(ldc);

    // blk_len is expected to be 32 for Q5 types.
    int64_t a_blk_stride = q8_blk_size(blk_len, true);

    float                         output[MB_ROWS * NB_COLS] = { 0 };
    std::array<int8_t *, MB_ROWS> a_data;
    std::array<float *, MB_ROWS>  c_data;

    for (size_t mi = 0; mi < MB_ROWS; ++mi) {
        c_data[mi] = c_ptr[mi];
    }

    if (quant_b_zp) {
        using blk_type = nrow_block_q5_1<NB_COLS>;

        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t     nb_real          = std::min<size_t>(NB_COLS, count_n - ni);
            blk_type * quant_b_blk_data = (blk_type *) quant_b_data + (ni / NB_COLS) * k_blks;

            for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                a_data[mi] = (int8_t *) quant_a_ptr[mi] + sizeof(float) + sizeof(int16_t);
            }

            for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                for (size_t ci = 0; ci < NB_COLS; ++ci) {
                    output[ci + mi * NB_COLS] = 0;
                }
            }

            for (size_t ki = 0; ki < k_blks; ++ki, ++quant_b_blk_data) {
                for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                    float *   a_scale_row = (float *) (a_data[mi] - sizeof(float) - sizeof(int16_t));
                    int16_t * a_sum_row   = (int16_t *) (a_data[mi] - sizeof(int16_t));
                    float     a_scale     = *a_scale_row;
                    int16_t   a_sum       = *a_sum_row;

                    for (size_t ci = 0; ci < NB_COLS; ++ci) {
                        float   b_scale  = ggml_fp16_to_fp32(quant_b_blk_data->scales16[ci]);
                        uint8_t b_zp_val = quant_b_blk_data->zp[ci];
                        int32_t acc      = a_sum * static_cast<int32_t>(b_zp_val);

                        for (size_t bi = 0; bi < blk_len / 2; ++bi) {
                            int8_t  a0       = a_data[mi][2 * bi];
                            int8_t  a1       = a_data[mi][2 * bi + 1];
                            uint8_t qs_byte  = quant_b_blk_data->qs[ci * (blk_len / 2) + bi];
                            int8_t  b0       = static_cast<int8_t>(qs_byte & 0x0F);
                            int8_t  b1       = static_cast<int8_t>((qs_byte >> 4) & 0x0F);
                            uint8_t qh_byte0 = quant_b_blk_data->qh[ci * 4 + (2 * bi) / 8];
                            uint8_t qh_byte1 = quant_b_blk_data->qh[ci * 4 + (2 * bi + 1) / 8];
                            uint8_t h0       = (qh_byte0 >> ((2 * bi) % 8)) & 1;
                            uint8_t h1       = (qh_byte1 >> ((2 * bi + 1) % 8)) & 1;

                            b0 |= (h0 << 4);
                            b1 |= (h1 << 4);

                            acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0) +
                                   static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
                        }

                        output[ci + mi * NB_COLS] += static_cast<float>(acc) * a_scale * b_scale;
                    }

                    a_data[mi] += a_blk_stride;
                }
            }

            for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                for (size_t ci = 0; ci < nb_real; ++ci) {
                    c_data[mi][ci] = output[mi * NB_COLS + ci];
                }
                c_data[mi] += NB_COLS;
            }
        }
    } else {
        using blk_type = nrow_block_q5_0<NB_COLS>;

        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t     nb_real          = std::min<size_t>(NB_COLS, count_n - ni);
            blk_type * quant_b_blk_data = (blk_type *) quant_b_data + (ni / NB_COLS) * k_blks;

            for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                a_data[mi] = (int8_t *) quant_a_ptr[mi] + sizeof(float) + sizeof(int16_t);
            }

            for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                for (size_t ci = 0; ci < NB_COLS; ++ci) {
                    output[ci + mi * NB_COLS] = 0;
                }
            }

            for (size_t ki = 0; ki < k_blks; ++ki, ++quant_b_blk_data) {
                for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                    float *   a_scale_row = (float *) (a_data[mi] - sizeof(float) - sizeof(int16_t));
                    int16_t * a_sum_row   = (int16_t *) (a_data[mi] - sizeof(int16_t));
                    float     a_scale     = *a_scale_row;
                    int16_t   a_sum       = *a_sum_row;

                    for (size_t ci = 0; ci < NB_COLS; ++ci) {
                        float   b_scale = ggml_fp16_to_fp32(quant_b_blk_data->scales16[ci]);
                        int32_t acc     = a_sum * 16;

                        for (size_t bi = 0; bi < blk_len / 2; ++bi) {
                            int8_t  a0       = a_data[mi][2 * bi];
                            int8_t  a1       = a_data[mi][2 * bi + 1];
                            uint8_t qs_byte  = quant_b_blk_data->qs[ci * (blk_len / 2) + bi];
                            int8_t  b0       = static_cast<int8_t>(qs_byte & 0x0F);
                            int8_t  b1       = static_cast<int8_t>((qs_byte >> 4) & 0x0F);
                            uint8_t qh_byte0 = quant_b_blk_data->qh[ci * 4 + (2 * bi) / 8];
                            uint8_t qh_byte1 = quant_b_blk_data->qh[ci * 4 + (2 * bi + 1) / 8];
                            uint8_t h0       = (qh_byte0 >> ((2 * bi) % 8)) & 1;
                            uint8_t h1       = (qh_byte1 >> ((2 * bi + 1) % 8)) & 1;

                            b0 |= (h0 << 4);
                            b1 |= (h1 << 4);

                            acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0) +
                                   static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
                        }

                        output[ci + mi * NB_COLS] += static_cast<float>(acc) * a_scale * b_scale;
                    }

                    a_data[mi] += a_blk_stride;
                }
            }

            for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                for (size_t ci = 0; ci < nb_real; ++ci) {
                    c_data[mi][ci] = output[mi * NB_COLS + ci];
                }
                c_data[mi] += NB_COLS;
            }
        }
    }
}

template <size_t MB_ROWS, size_t NB_COLS>
void gemm_kernel_i8i8_mrow_ref(size_t          blk_len,
                               const uint8_t * quant_a_ptr,
                               const uint8_t * quant_b_data,
                               const uint8_t * quant_b_zp,
                               float *         c_ptr,
                               size_t          count_m,
                               size_t          count_n,
                               size_t          k_blks,
                               size_t          ldc) {
    int64_t b_blk_stride        = (sizeof(ggml_fp16_t) + blk_len);
    int64_t b_stride            = k_blks * b_blk_stride;
    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t b_ncol_block_stride = b_blk_stride * NB_COLS;

    float output[MB_ROWS * NB_COLS] = { 0 };

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t   nb_real = std::min<size_t>(NB_COLS, count_n - ni);
        int8_t * b_data  = (int8_t *) quant_b_data + ni * b_stride + NB_COLS * sizeof(ggml_fp16_t);

        int8_t * a_data = (int8_t *) quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS;

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < NB_COLS; ci++) {
                output[ci + mi * NB_COLS] = 0;
            }
        }

        for (size_t ki = 0; ki < k_blks; ki++, a_data += a_nrow_block_stride, b_data += b_ncol_block_stride) {
            ggml_fp16_t * b_scale_fp16 = (ggml_fp16_t *) (b_data - NB_COLS * sizeof(ggml_fp16_t));

            float * a_scale_row = (float *) (a_data - sizeof(float) * MB_ROWS - sizeof(int16_t) * MB_ROWS);

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                float a_scale = a_scale_row[mi];
                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    float   b_scale = ggml_fp16_to_fp32(b_scale_fp16[ci]);
                    int32_t acc     = 0;
                    for (size_t bi = 0; bi < blk_len; bi++) {
                        int8_t a0 = a_data[mi * blk_len + bi];
                        int8_t b0 = b_data[ci * blk_len + bi];
                        acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0);
                    }
                    output[ci + mi * NB_COLS] += static_cast<float>(acc) * a_scale * b_scale;
                }
            }
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < nb_real; ci++) {
                c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}

template <size_t MB_ROWS, size_t NB_COLS>
void gemm_kernel_i8i5_mrow_ref(size_t          blk_len,
                               const uint8_t * quant_a_ptr,
                               const uint8_t * quant_b_data,
                               const uint8_t * quant_b_zp,
                               float *         c_ptr,
                               size_t          count_m,
                               size_t          count_n,
                               size_t          k_blks,
                               size_t          ldc) {
    // blk_len is expected to be 32 for Q5 types
    // quant_b_zp != nullptr => nrow_block_q5_1<NB_COLS> (has zp)
    // quant_b_zp == nullptr => nrow_block_q5_0<NB_COLS> (no zp)

    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;

    float output[MB_ROWS * NB_COLS] = { 0 };

    if (quant_b_zp) {
        // nrow_block_q5_1<NB_COLS>: scales16[NB_COLS] + zp[NB_COLS] + qh[4*NB_COLS] + qs[16*NB_COLS]
        using blk_type                 = nrow_block_q5_1<NB_COLS>;
        int64_t    b_ncol_block_stride = sizeof(blk_type);
        blk_type * quant_b_blk_data    = (blk_type *) quant_b_data;

        for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
            size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);

            int8_t * a_data = (int8_t *) quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS;

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    output[ci + mi * NB_COLS] = 0;
                }
            }

            for (size_t ki = 0; ki < k_blks; ki++, quant_b_blk_data++, a_data += a_nrow_block_stride) {
                float *   a_scale_row = (float *) (a_data - sizeof(float) * MB_ROWS - sizeof(int16_t) * MB_ROWS);
                int16_t * a_sum_row   = (int16_t *) (a_data - sizeof(int16_t) * MB_ROWS);

                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    float   a_scale = a_scale_row[mi];
                    int16_t a_sum   = a_sum_row[mi];

                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        float   b_scale  = ggml_fp16_to_fp32(quant_b_blk_data->scales16[ci]);
                        uint8_t b_zp_val = quant_b_blk_data->zp[ci];
                        int32_t acc      = a_sum * static_cast<int32_t>(b_zp_val);

                        for (size_t bi = 0; bi < blk_len / 2; bi++) {
                            int8_t  a0      = a_data[mi * blk_len + 2 * bi];
                            int8_t  a1      = a_data[mi * blk_len + 2 * bi + 1];
                            uint8_t qs_byte = quant_b_blk_data->qs[ci * (blk_len / 2) + bi];
                            int8_t  b0      = static_cast<int8_t>(qs_byte & 0x0F);
                            int8_t  b1      = static_cast<int8_t>((qs_byte >> 4) & 0x0F);

                            // Extract high bits from qh
                            // qh is packed as 4 bytes per column (32 bits for 32 elements)
                            uint8_t qh_byte0 = quant_b_blk_data->qh[ci * 4 + (2 * bi) / 8];
                            uint8_t qh_byte1 = quant_b_blk_data->qh[ci * 4 + (2 * bi + 1) / 8];
                            uint8_t h0       = (qh_byte0 >> ((2 * bi) % 8)) & 1;
                            uint8_t h1       = (qh_byte1 >> ((2 * bi + 1) % 8)) & 1;

                            b0 |= (h0 << 4);
                            b1 |= (h1 << 4);

                            acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0) +
                                   static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
                        }
                        output[ci + mi * NB_COLS] += static_cast<float>(acc) * a_scale * b_scale;
                    }
                }
            }

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                for (size_t ci = 0; ci < nb_real; ci++) {
                    c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
                }
            }
        }
    } else {
        // nrow_block_q5_0<NB_COLS>: scales16[NB_COLS] + qh[4*NB_COLS] + qs[16*NB_COLS]
        using blk_type                 = nrow_block_q5_0<NB_COLS>;
        int64_t    b_ncol_block_stride = sizeof(blk_type);
        blk_type * quant_b_blk_data    = (blk_type *) quant_b_data;

        for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
            size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);

            int8_t * a_data = (int8_t *) quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS;

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    output[ci + mi * NB_COLS] = 0;
                }
            }

            for (size_t ki = 0; ki < k_blks; ki++, quant_b_blk_data++, a_data += a_nrow_block_stride) {
                float *   a_scale_row = (float *) (a_data - sizeof(float) * MB_ROWS - sizeof(int16_t) * MB_ROWS);
                int16_t * a_sum_row   = (int16_t *) (a_data - sizeof(int16_t) * MB_ROWS);

                for (size_t mi = 0; mi < MB_ROWS; mi++) {
                    float   a_scale = a_scale_row[mi];
                    int16_t a_sum   = a_sum_row[mi];

                    for (size_t ci = 0; ci < NB_COLS; ci++) {
                        float   b_scale = ggml_fp16_to_fp32(quant_b_blk_data->scales16[ci]);
                        // Q5_0 has no zp, use default offset 16 (midpoint of 5-bit unsigned range)
                        int32_t acc     = a_sum * 16;

                        for (size_t bi = 0; bi < blk_len / 2; bi++) {
                            int8_t  a0      = a_data[mi * blk_len + 2 * bi];
                            int8_t  a1      = a_data[mi * blk_len + 2 * bi + 1];
                            uint8_t qs_byte = quant_b_blk_data->qs[ci * (blk_len / 2) + bi];
                            int8_t  b0      = static_cast<int8_t>(qs_byte & 0x0F);
                            int8_t  b1      = static_cast<int8_t>((qs_byte >> 4) & 0x0F);

                            // Extract high bits from qh
                            uint8_t qh_byte0 = quant_b_blk_data->qh[ci * 4 + (2 * bi) / 8];
                            uint8_t qh_byte1 = quant_b_blk_data->qh[ci * 4 + (2 * bi + 1) / 8];
                            uint8_t h0       = (qh_byte0 >> ((2 * bi) % 8)) & 1;
                            uint8_t h1       = (qh_byte1 >> ((2 * bi + 1) % 8)) & 1;

                            b0 |= (h0 << 4);
                            b1 |= (h1 << 4);

                            acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0) +
                                   static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
                        }
                        output[ci + mi * NB_COLS] += static_cast<float>(acc) * a_scale * b_scale;
                    }
                }
            }

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                for (size_t ci = 0; ci < nb_real; ci++) {
                    c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
                }
            }
        }
    }
}

template <size_t MB_ROWS, size_t NB_COLS>
void gemm_kernel_i8mxfp4_mrow_ref(size_t          blk_len,
                                  const uint8_t * quant_a_ptr,
                                  const uint8_t * quant_b_data,
                                  const uint8_t * quant_b_zp,
                                  float *         c_ptr,
                                  size_t          count_m,
                                  size_t          count_n,
                                  size_t          k_blks,
                                  size_t          ldc) {
    // blk_len is expected to be 32 (QK_MXFP4)
    // quant_b_zp is unused for MXFP4 (symmetric quantization)
    GGML_UNUSED(quant_b_zp);

    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;

    float output[MB_ROWS * NB_COLS] = { 0 };

    using blk_type              = nrow_block_mxfp4<NB_COLS>;
    blk_type * quant_b_blk_data = (blk_type *) quant_b_data;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);

        int8_t * a_data = (int8_t *) quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS;

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < NB_COLS; ci++) {
                output[ci + mi * NB_COLS] = 0;
            }
        }

        for (size_t ki = 0; ki < k_blks; ki++, quant_b_blk_data++, a_data += a_nrow_block_stride) {
            float *   a_scale_row = (float *) (a_data - sizeof(float) * MB_ROWS - sizeof(int16_t) * MB_ROWS);
            int16_t * a_sum_row   = (int16_t *) (a_data - sizeof(int16_t) * MB_ROWS);

            for (size_t mi = 0; mi < MB_ROWS; mi++) {
                float a_scale = a_scale_row[mi];

                for (size_t ci = 0; ci < NB_COLS; ci++) {
                    float b_scale = GGML_E8M0_TO_FP32_HALF(quant_b_blk_data->e[ci]);

                    // Read 32 sign bits for this column
                    uint32_t sign_bits;
                    memcpy(&sign_bits, &quant_b_blk_data->qh[ci * 4], 4);

                    int32_t acc = 0;
                    for (size_t bi = 0; bi < blk_len / 2; bi++) {
                        int8_t a0 = a_data[mi * blk_len + 2 * bi];
                        int8_t a1 = a_data[mi * blk_len + 2 * bi + 1];

                        // qs[ci*16 + bi] stores abs(vals[bi*2]) in low 4 bits
                        // and abs(vals[bi*2+1]) in high 4 bits
                        uint8_t qs_byte = quant_b_blk_data->qs[ci * 16 + bi];
                        int8_t  b_abs0  = static_cast<int8_t>(qs_byte & 0x0F);
                        int8_t  b_abs1  = static_cast<int8_t>((qs_byte >> 4) & 0x0F);

                        // Extract sign bits: bit (2*bi) for vals[2*bi], bit (2*bi+1) for vals[2*bi+1]
                        int8_t b0 = (sign_bits >> (2 * bi)) & 1 ? -b_abs0 : b_abs0;
                        int8_t b1 = (sign_bits >> (2 * bi + 1)) & 1 ? -b_abs1 : b_abs1;

                        acc += static_cast<int32_t>(a0) * static_cast<int32_t>(b0) +
                               static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
                    }
                    output[ci + mi * NB_COLS] += static_cast<float>(acc) * a_scale * b_scale;
                }
            }
        }

        for (size_t mi = 0; mi < MB_ROWS; mi++) {
            for (size_t ci = 0; ci < nb_real; ci++) {
                c_ptr[mi * ldc + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}

void gemm_kernel_i8i2k_m1(size_t          blk_len,
                          const uint8_t * quant_a_ptr,
                          const uint8_t * quant_b_data,
                          float *         c_ptr,
                          size_t          count_m,
                          size_t          count_n,
                          size_t          k_blks,
                          size_t          ldc) {
    constexpr size_t NB_COLS = 32;
    using blk_type           = nrow_block_q2_k<NB_COLS>;

    int64_t b_ncol_block_stride = sizeof(blk_type) * k_blks;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t * b_data = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_ncol_block_stride;
        int8_t *  a_data = (int8_t *) quant_a_ptr;
        float *   dst_c  = (float *) c_ptr + ni;

        asm volatile(
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v31, v31, v31           \n\t"
            "mv             s1, %[BK]               \n\t"

            ".align 4                               \n\t"
            "BLK_LOOP%=:                            \n\t"
            // load scale A
            "flw            fa0, (%[A])             \n\t"
            "addi           %[A], %[A], 4           \n\t"

            "li             t1, 4                   \n\t"
            "addi           t2, %[B], 512           \n\t"  // B data addr
            "addi           t3, %[A], 32            \n\t"  // A data addr
            "addi           s3, %[B], 0             \n\t"
            "vxor.vv        v30, v29, v29           \n\t"  // tmp result

            "INNER_K_LOOP%=:                        \n\t"
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vxor.vv        v2, v2, v2              \n\t"
            "vxor.vv        v3, v3, v3              \n\t"
            "vxor.vv        v4, v4, v4              \n\t"
            "vxor.vv        v5, v5, v5              \n\t"
            "vxor.vv        v6, v6, v6              \n\t"
            "vxor.vv        v28, v28, v28           \n\t"
            "vxor.vv        v29, v29, v29           \n\t"

            // load scale  B
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vle8.v         v0, (%[B])              \n\t"
            "addi           %[B], %[B], 128         \n\t"

            // A data, 1x64@i8
            "vsetivli       t0, 16, e8, mf4         \n\t"
            "vle8.v         v2, (t3)                \n\t"
            "addi           t3, t3, 16              \n\t"

            "vsetivli       t0, 16, e8, mf4         \n\t"
            "vle8.v         v4, (t3)                \n\t"
            "addi           t3, t3, 16              \n\t"

            "vsetivli       t0, 16, e8, mf4         \n\t"
            "vle8.v         v5, (t3)                \n\t"
            "addi           t3, t3, 16              \n\t"

            "vsetivli       t0, 16, e8, mf4         \n\t"
            "vle8.v         v6, (t3)                \n\t"
            "addi           t3, t3, 16              \n\t"

            "vsetvli        t0, x0, e64, mf2        \n\t"
            "vslideup.vi    v3, v4, 2               \n\t"
            "vslideup.vi    v28, v5, 4              \n\t"
            "vslideup.vi    v29, v6, 6              \n\t"

            // init the accumu to zero
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v20, v18, v18           \n\t"
            "vxor.vv        v22, v18, v18           \n\t"
            "vxor.vv        v24, v18, v18           \n\t"
            "vxor.vv        v26, v18, v18           \n\t"

            // B data, 32x64@i2
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vl4r.v         v4, (t2)                \n\t"
            "addi           t2, t2, 512             \n\t"
            "vand.vi        v8, v4, 0x3             \n\t"  // 0-15
            "vsrl.vi        v9, v4, 2               \n\t"
            "vsrl.vi        v10, v4, 4              \n\t"
            "vsrl.vi        v11, v4, 6              \n\t"  // 48-63
            "vand.vi        v9, v9, 0x3             \n\t"  // 16-31
            "vand.vi        v10, v10, 0x3           \n\t"  // 32-47

            "vand.vi        v12, v5, 0x3            \n\t"  // 0-15
            "vsrl.vi        v13, v5, 2              \n\t"
            "vsrl.vi        v14, v5, 4              \n\t"
            "vsrl.vi        v15, v5, 6              \n\t"  // 48-63
            "vand.vi        v13, v13, 0x3           \n\t"  // 16-31
            "vand.vi        v14, v14, 0x3           \n\t"  // 32-47

            "vand.vi        v16, v6, 0x3            \n\t"  // 0-15
            "vsrl.vi        v17, v6, 2              \n\t"
            "vsrl.vi        v18, v6, 4              \n\t"
            "vsrl.vi        v19, v6, 6              \n\t"  // 48-63
            "vand.vi        v17, v17, 0x3           \n\t"  // 16-31
            "vand.vi        v18, v18, 0x3           \n\t"  // 32-47

            "vand.vi        v4, v7, 0x3             \n\t"  // 0-15
            "vsrl.vi        v5, v7, 2               \n\t"
            "vsrl.vi        v6, v7, 4               \n\t"
            "vsrl.vi        v7, v7, 6               \n\t"  // 48-63
            "vand.vi        v5, v5, 0x3             \n\t"  // 16-31
            "vand.vi        v6, v6, 0x3             \n\t"  // 32-47

            // i2 * i8 vmadot
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vmadotsu       v20, v2, v8, i8         \n\t"
            "vmadotsu       v22, v2, v12, i8        \n\t"
            "vmadotsu       v24, v2, v16, i8        \n\t"
            "vmadotsu       v26, v2, v4, i8         \n\t"

            "vmadotsu       v20, v3, v9, i8         \n\t"
            "vmadotsu       v22, v3, v13, i8        \n\t"
            "vmadotsu       v24, v3, v17, i8        \n\t"
            "vmadotsu       v26, v3, v5, i8         \n\t"

            "vmadotsu       v20, v28, v10, i8       \n\t"
            "vmadotsu       v22, v28, v14, i8       \n\t"
            "vmadotsu       v24, v28, v18, i8       \n\t"
            "vmadotsu       v26, v28, v6, i8        \n\t"

            "vmadotsu       v20, v29, v11, i8       \n\t"
            "vmadotsu       v22, v29, v15, i8       \n\t"
            "vmadotsu       v24, v29, v19, i8       \n\t"
            "vmadotsu       v26, v29, v7, i8        \n\t"

            "vand.vi        v10, v0, 0xf            \n\t"  // scale
            "vwadd.vx       v12, v10, x0            \n\t"
            "vsetvli        t0, x0, e16, m2         \n\t"
            "vwadd.vx       v16, v12, x0            \n\t"

            "vsetvli        t0, x0, e32, m1         \n\t"
            "vpack.vv       v2, v20, v22, 2         \n\t"
            "vpack.vv       v4, v24, v26, 2         \n\t"
            "vpack.vv       v6, v2, v4, 3           \n\t"  // 0,1
            "vpack.vv       v8, v3, v5, 3           \n\t"  // 2,3

            // mul scale
            "vmacc.vv       v30, v6, v16            \n\t"
            "vmacc.vv       v30, v7, v17            \n\t"
            "vmacc.vv       v30, v8, v18            \n\t"
            "vmacc.vv       v30, v9, v19            \n\t"

            "addi           t1, t1, -1              \n\t"
            "bgtz           t1, INNER_K_LOOP%=      \n\t"

            // load zp B
            "vsetvli        t0, x0, e8, m4          \n\t"
            "vle8.v         v4, (s3)                \n\t"
            "vsrl.vi        v8, v4, 4               \n\t"  // zp

            // asum * zp
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v20, v20, v20           \n\t"
            "vxor.vv        v22, v22, v22           \n\t"
            "vxor.vv        v24, v24, v24           \n\t"
            "vxor.vv        v26, v26, v26           \n\t"

            "vsetvli        t0, x0, e16, mf4        \n\t"
            "vle16.v        v2, (%[A])              \n\t"
            "vsetvli        t0, x0, e8, mf4         \n\t"
            "vnsrl.wi       v12, v2, 0              \n\t"  // low 8
            "vnsra.wi       v13, v2, 8              \n\t"  // high 8

            "vsetvli        t0, x0, e32, m1         \n\t"
            "vmadotsu       v20, v13, v8, i8        \n\t"
            "vmadotsu       v22, v13, v9, i8        \n\t"
            "vmadotsu       v24, v13, v10, i8       \n\t"
            "vmadotsu       v26, v13, v11, i8       \n\t"

            "vsll.vi        v20, v20, 8             \n\t"
            "vsll.vi        v22, v22, 8             \n\t"
            "vsll.vi        v24, v24, 8             \n\t"
            "vsll.vi        v26, v26, 8             \n\t"

            "vmadotu        v20, v12, v8, i8        \n\t"
            "vmadotu        v22, v12, v9, i8        \n\t"
            "vmadotu        v24, v12, v10, i8       \n\t"
            "vmadotu        v26, v12, v11, i8       \n\t"

            "vpack.vv       v2, v20, v22, 2         \n\t"
            "vpack.vv       v4, v24, v26, 2         \n\t"
            "vpack.vv       v28, v2, v4, 3          \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "vle16.v        v0, (t2)                \n\t"  // scale16
            "addi           t2, t2, 64              \n\t"
            "vle16.v        v1, (t2)                \n\t"  // zero16
            "vfwcvt.f.f.v   v2, v0                  \n\t"
            "vfwcvt.f.f.v   v4, v1                  \n\t"
            "vsetvli        t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v    v30, v30                \n\t"
            "vfcvt.f.x.v    v28, v28                \n\t"
            "addi           %[B], t2, 64            \n\t"
            "mv             %[A], t3                \n\t"

            "vfmul.vv       v30, v30, v2            \n\t"  // mul scale16
            "vfmacc.vv      v30, v28, v4            \n\t"  // + mul zero16
            "vfmacc.vf      v31, fa0, v30           \n\t"
            "addi           s1, s1, -1              \n\t"
            "bgtz           s1, BLK_LOOP%=          \n\t"

            // save
            "vsetvli        t0, x0, e32, m1         \n\t"
            "vse32.v        v31, (%[DST])           \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [BK] "r"(k_blks)
            : "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
              "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "fa0", "t4", "t5", "t6", "s1", "s2", "s3");
    }
}

void gemm_kernel_i8i2k_m4(size_t          blk_len,
                          const uint8_t * quant_a_ptr,
                          const uint8_t * quant_b_data,
                          float *         c_ptr,
                          size_t          count_m,
                          size_t          count_n,
                          size_t          k_blks,
                          size_t          ldc) {
    constexpr size_t NB_COLS = 32;
    using blk_type           = nrow_block_q2_k<NB_COLS>;

    int64_t  b_ncol_block_stride = sizeof(blk_type) * k_blks;
    _Float16 scale               = 0.0625f;
    _Float16 scale_1             = 16.0f;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t * b_data = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_ncol_block_stride;
        int8_t *  a_data = (int8_t *) quant_a_ptr;
        float *   dst_c  = (float *) c_ptr + ni;

        asm volatile(
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v28, v31, v31           \n\t"  // init result
            "vxor.vv        v29, v31, v31           \n\t"
            "vxor.vv        v30, v31, v31           \n\t"
            "vxor.vv        v31, v31, v31           \n\t"
            "mv             s1, %[BK]               \n\t"

            ".align 4                               \n\t"
            "BLK_LOOP%=:                            \n\t"
            // load scale A
            "flw            fa0, (%[A])             \n\t"
            "flw            fa1, 4(%[A])            \n\t"
            "flw            fa2, 8(%[A])            \n\t"
            "flw            fa3, 12(%[A])           \n\t"
            "addi           %[A], %[A], 16          \n\t"

            "li             t1, 4                   \n\t"
            "addi           t2, %[B], 512           \n\t"  // B data addr
            "addi           t3, %[A], 128           \n\t"  // A data addr
            "addi           s4, t2, 1024            \n\t"  // scale16 addr
            "addi           s4, s4, 1024            \n\t"  // TODO
            "addi           s3, %[B], 0             \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "vle16.v        v1, (s4)                \n\t"  // load scale16
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vpack.vv       v22, v1, v1, 3          \n\t"

            "addi           s4, t3, 256             \n\t"  // addr 1
            "addi           s5, t3, 512             \n\t"  // addr 2
            "addi           s6, t3, 768             \n\t"  // addr 3

            // init the accu to 0
            "vxor.vv        v24, v24, v24           \n\t"
            "vxor.vv        v25, v25, v25           \n\t"
            "vxor.vv        v26, v26, v26           \n\t"
            "vxor.vv        v27, v27, v27           \n\t"

            "INNER_K_LOOP%=:                        \n\t"
            // load scale  B
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vle8.v         v1, (%[B])              \n\t"
            "addi           %[B], %[B], 128         \n\t"
            "vand.vi        v1, v1, 0xf             \n\t"

            "vfwcvt.f.x.v   v20, v1                 \n\t"  // f16 scale B
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vfmul.vv       v0, v20, v22            \n\t"  // mul scale16
            "vfmul.vv       v1, v21, v22            \n\t"  // mul scale16
            "vfmul.vf       v0, v0, %[SCALE]        \n\t"  // mul magic
            "vfmul.vf       v1, v1, %[SCALE]        \n\t"  // mul magic

            // A data, 4x64@i8
            "vsetvli        t0, x0, e8, mf2         \n\t"
            "vle8.v         v2, (t3)                \n\t"
            "addi           t3, t3, 64              \n\t"
            "vle8.v         v3, (s4)                \n\t"
            "addi           s4, s4, 64              \n\t"
            "vle8.v         v4, (s5)                \n\t"
            "addi           s5, s5, 64              \n\t"
            "vle8.v         v5, (s6)                \n\t"
            "addi           s6, s6, 64              \n\t"

            // 4x64 => 4x16x4
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vpack.vv       v6, v2, v3, 1           \n\t"
            "vpack.vv       v8, v4, v5, 1           \n\t"
            "vpack.vv       v2, v6, v8, 2           \n\t"  // 0, 2

            "vpack.vv       v20, v2, v2, 3          \n\t"  // 1
            "vor.vv         v23, v21, v21           \n\t"
            "vpack.vv       v20, v3, v3, 3          \n\t"  // 3

            // B data, 32x64@i2
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vl4r.v         v4, (t2)                \n\t"
            "addi           t2, t2, 512             \n\t"
            "vand.vi        v8, v4, 0x3             \n\t"  // 0-15
            "vsrl.vi        v9, v4, 2               \n\t"
            "vsrl.vi        v10, v4, 4              \n\t"
            "vsrl.vi        v11, v4, 6              \n\t"  // 48-63
            "vand.vi        v9, v9, 0x3             \n\t"  // 16-31
            "vand.vi        v10, v10, 0x3           \n\t"  // 32-47

            "vand.vi        v12, v5, 0x3            \n\t"  // 0-15
            "vsrl.vi        v13, v5, 2              \n\t"
            "vsrl.vi        v14, v5, 4              \n\t"
            "vsrl.vi        v15, v5, 6              \n\t"  // 48-63
            "vand.vi        v13, v13, 0x3           \n\t"  // 16-31
            "vand.vi        v14, v14, 0x3           \n\t"  // 32-47

            "vand.vi        v16, v6, 0x3            \n\t"  // 0-15
            "vsrl.vi        v17, v6, 2              \n\t"
            "vsrl.vi        v18, v6, 4              \n\t"
            "vsrl.vi        v19, v6, 6              \n\t"  // 48-63
            "vand.vi        v17, v17, 0x3           \n\t"  // 16-31
            "vand.vi        v18, v18, 0x3           \n\t"  // 32-47

            "vand.vi        v4, v7, 0x3             \n\t"  // 0-15
            "vsrl.vi        v5, v7, 2               \n\t"
            "vsrl.vi        v6, v7, 4               \n\t"
            "vsrl.vi        v7, v7, 6               \n\t"  // 48-63
            "vand.vi        v5, v5, 0x3             \n\t"  // 16-31
            "vand.vi        v6, v6, 0x3             \n\t"  // 32-47

            // i2 * i8 vmadot
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vmadotsu.hp    v24, v2, v8, v0, 0, i8  \n\t"
            "vmadotsu.hp    v25, v2, v12, v0, 1, i8 \n\t"
            "vmadotsu.hp    v26, v2, v16, v0, 2, i8 \n\t"
            "vmadotsu.hp    v27, v2, v4, v0, 3, i8  \n\t"

            "vmadotsu.hp    v24, v23, v9, v0, 4, i8 \n\t"
            "vmadotsu.hp    v25, v23, v13, v0, 5, i8\n\t"
            "vmadotsu.hp    v26, v23, v17, v0, 6, i8\n\t"
            "vmadotsu.hp    v27, v23, v5, v0, 7, i8 \n\t"

            "vmadotsu.hp    v24, v3, v10, v1, 0, i8 \n\t"
            "vmadotsu.hp    v25, v3, v14, v1, 1, i8 \n\t"
            "vmadotsu.hp    v26, v3, v18, v1, 2, i8 \n\t"
            "vmadotsu.hp    v27, v3, v6, v1, 3, i8  \n\t"

            "vmadotsu.hp    v24, v21, v11, v1, 4, i8\n\t"
            "vmadotsu.hp    v25, v21, v15, v1, 5, i8\n\t"
            "vmadotsu.hp    v26, v21, v19, v1, 6, i8\n\t"
            "vmadotsu.hp    v27, v21, v7, v1, 7, i8 \n\t"

            "addi           t1, t1, -1              \n\t"
            "bgtz           t1, INNER_K_LOOP%=      \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vpack.vv       v2, v24, v25, 1         \n\t"
            "vpack.vv       v4, v26, v27, 1         \n\t"
            "vpack.vv       v6, v2, v4, 2           \n\t"  // 0,1,2,3

            "vxor.vv        v18, v18, v18           \n\t"
            "vxor.vv        v20, v20, v20           \n\t"
            "vxor.vv        v22, v22, v22           \n\t"
            "vxor.vv        v24, v24, v24           \n\t"
            // load zp B, 16x8x4@int4
            "vsetvli        t0, x0, e8, m4          \n\t"
            "vle8.v         v0, (s3)                \n\t"
            "vsrl.vi        v0, v0, 4               \n\t"  // zp

            // 4x16@int16
            "vsetvli        t0, x0, e16, m1         \n\t"  // a sum
            "vle16.v         v12, (%[A])             \n\t"
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vnsrl.wi       v10, v12, 0             \n\t"  // low 8
            "vnsra.wi       v11, v12, 8             \n\t"  // high 8

            // asum * zp
            "vsetvli        t0, x0, e32, m1          \n\t"
            "vmadotsu       v18, v11, v0, i8        \n\t"
            "vmadotsu       v20, v11, v1, i8        \n\t"
            "vmadotsu       v22, v11, v2, i8        \n\t"
            "vmadotsu       v24, v11, v3, i8        \n\t"
            "vsll.vi        v18, v18, 8             \n\t"
            "vsll.vi        v20, v20, 8             \n\t"
            "vsll.vi        v22, v22, 8             \n\t"
            "vsll.vi        v24, v24, 8             \n\t"
            "vmadotu        v18, v10, v0, i8        \n\t"
            "vmadotu        v20, v10, v1, i8        \n\t"
            "vmadotu        v22, v10, v2, i8        \n\t"
            "vmadotu        v24, v10, v3, i8        \n\t"

            "vpack.vv       v10, v18, v20, 2        \n\t"
            "vpack.vv       v12, v22, v24, 2        \n\t"
            "vpack.vv       v14, v10, v12, 3        \n\t"
            "vpack.vv       v16, v11, v13, 3        \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "addi           t2, t2, 64              \n\t"
            "vle16.v        v20, (t2)               \n\t"  // zero16
            "vfwcvt.f.f.v   v22, v20                \n\t"

            // mul 1/magic
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vfwmul.vf      v0, v6, %[SCALE_1]      \n\t"
            "vfwmul.vf      v2, v7, %[SCALE_1]      \n\t"

            "vsetvli        t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v    v14, v14                \n\t"
            "vfcvt.f.x.v    v15, v15                \n\t"
            "vfcvt.f.x.v    v16, v16                \n\t"
            "vfcvt.f.x.v    v17, v17                \n\t"

            "addi           %[B], t2, 64            \n\t"
            "mv             %[A], s6                \n\t"

            "vfmacc.vv      v0, v14, v22            \n\t"  // + mul zero16
            "vfmacc.vv      v1, v15, v22            \n\t"
            "vfmacc.vv      v2, v16, v22            \n\t"
            "vfmacc.vv      v3, v17, v22            \n\t"

            "vfmacc.vf      v28, fa0, v0            \n\t"  // mul a scale
            "vfmacc.vf      v29, fa1, v1            \n\t"
            "vfmacc.vf      v30, fa2, v2            \n\t"
            "vfmacc.vf      v31, fa3, v3            \n\t"

            "addi           s1, s1, -1              \n\t"
            "bgtz           s1, BLK_LOOP%=          \n\t"

            // save
            "vsetvli        t0, x0, e32, m1         \n\t"
            "add            t1, %[LDC], %[DST]      \n\t"
            "vse32.v        v28, (%[DST])           \n\t"
            "vse32.v        v29, (t1)               \n\t"
            "add            t1, t1, %[LDC]          \n\t"
            "vse32.v        v30, (t1)               \n\t"
            "add            t1, t1, %[LDC]          \n\t"
            "vse32.v        v31, (t1)               \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [BK] "r"(k_blks), [LDC] "r"(ldc * 4), [SCALE] "f"(scale), [SCALE_1] "f"(scale_1)
            : "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
              "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "fa0", "t4", "t5", "t6", "s1", "s2", "s3", "s4", "s5", "s6");
    }
}

void gemm_kernel_i8i3k_m1(size_t          blk_len,
                          const uint8_t * quant_a_ptr,
                          const uint8_t * quant_b_data,
                          float *         c_ptr,
                          size_t          count_m,
                          size_t          count_n,
                          size_t          k_blks,
                          size_t          ldc) {
    constexpr size_t NB_COLS = 32;  //only support 32 in ASM
    using blk_type           = nrow_block_q3_k<NB_COLS>;

    const blk_type * b_base = reinterpret_cast<const blk_type *>(quant_b_data);

    int64_t a_blk_stride        = q8k_blk_size(256);
    int64_t a_nrow_block_stride = a_blk_stride;
    int64_t b_ncol_block_stride = sizeof(blk_type);

    // Constants used by q3_k scaling in HP branch:
    // - k_q3k_scale_step: per-nibble scale factor (1/16).
    // - k_a_scale_post_mul: A_scale needs an extra *16 at the end (pairs with 1/16 above).
    const _Float16 k_q3k_scale_step   = (_Float16) 0.0625f;  // 1 / 16
    const float    k_a_scale_post_mul = 16.0f;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t           nb_real          = std::min<size_t>(NB_COLS, count_n - ni);
        const blk_type * quant_b_blk_data = b_base + (ni / NB_COLS) * k_blks;
#if 0
        //------------------------------------------------------------------------------
        // A format
        // Ascale   fp32 * 1    32bit
        // Asum     int16 * 16  256bit
        // A M1K256 int8        2048bit
        //------------------------------------------------------------------------------
        // B format
        // B_scl    uint8*N32*16    4096bit
        // B_Hmask  N32K16*16 1bit  8192bit
        // B_Qs     N32K16*16 2bit  16384bit
        // B scl16  fp16 * N32      512bit;
        //------------------------------------------------------------------------------
        //bias always be nullptr
        __asm__ volatile(
            // t2 = k_blks (each is K256 superblock)
            "mv           t2, %[KBLKS]            \n\t"
            // t3 = 256/64 = 4 (K64 iterations per superblock)
            "li           t3, 4                   \n\t"
            "mv           s2, %[pA]               \n\t"  // s2 = pASCL
            "addi         s3, %[pA], 4+32         \n\t"  // s3 = pAData, (pA+AScl+ASum)

            // B block layout for nrow_block_q3_k<32>:
            // scales: 512B, hmask: 1024B, qs: 2048B, scales16: 64B
            "addi         s5, %[pB], 32*16        \n\t"  // s5 = pB_hmask
            "mv           s4, %[pB]               \n\t"  // s4 = pB_scales
            "addi         s6, s5, 1024            \n\t"  // s6 = pB_qs
            "mv           s7, %[pB]               \n\t"  // s7 = pB_base

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v31, v0, v0             \n\t"  // clear acc
            "vxor.vv      v30, v0, v0             \n\t"  // clear acc of K256

            // ordinary vmadot: vle*10 vecIns*78 vmadot*16
            ".align 4                             \n\t"
            "BLK_LPST%=:                          \n\t"
            "K64_LPST%=:                          \n\t"

            // K0-15
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vle8.v       v2, (s4)                \n\t"
            "addi         s4, s4, 128             \n\t"

            // load B qs chunk (128B per K16, 16 times => 2048B)
            "vle8.v       v4, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v5, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v6, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v7, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v1, (s3)                \n\t"
            "addi         s3, s3, 64              \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vand.vi      v12, v4, 0x3            \n\t"
            "vand.vi      v13, v5, 0x3            \n\t"
            "vand.vi      v14, v6, 0x3            \n\t"
            "vand.vi      v15, v7, 0x3            \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v1, v12, i8        \n\t"
            "vmadot       v18, v1, v13, i8        \n\t"
            "vmadot       v20, v1, v14, i8        \n\t"
            "vmadot       v22, v1, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"  // N0-N31 in v16

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v18, v2, x0             \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v19, v18, x0            \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v30, v16, v19           \n\t"

            //K16-31
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v2, v2, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v1, v1, 2             \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v8, v4, 4               \n\t"
            "vsll.vi      v9, v5, 4               \n\t"
            "vsll.vi      v10, v6, 4              \n\t"
            "vsll.vi      v11, v7, 4              \n\t"
            "vnot.v       v0, v0                  \n\t"

            "vsrl.vi      v12, v8, 6              \n\t"
            "vsrl.vi      v13, v9, 6              \n\t"
            "vsrl.vi      v14, v10, 6             \n\t"
            "vsrl.vi      v15, v11, 6             \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v1, v12, i8        \n\t"
            "vmadot       v18, v1, v13, i8        \n\t"
            "vmadot       v20, v1, v14, i8        \n\t"
            "vmadot       v22, v1, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"  // N0-N31 in v16

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v18, v2, x0              \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v19, v18, x0              \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v30, v16, v19            \n\t"

            //K32-47
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v2, v2, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v1, v1, 2             \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v8, v4, 2               \n\t"
            "vsll.vi      v9, v5, 2               \n\t"
            "vsll.vi      v10, v6, 2              \n\t"
            "vsll.vi      v11, v7, 2              \n\t"
            "vnot.v       v0, v0                  \n\t"

            "vsrl.vi      v12, v8, 6              \n\t"
            "vsrl.vi      v13, v9, 6              \n\t"
            "vsrl.vi      v14, v10, 6             \n\t"
            "vsrl.vi      v15, v11, 6             \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v1, v12, i8        \n\t"
            "vmadot       v18, v1, v13, i8        \n\t"
            "vmadot       v20, v1, v14, i8        \n\t"
            "vmadot       v22, v1, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v18, v2, x0              \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v19, v18, x0              \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v30, v16, v19            \n\t"

            // K48-63
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v2, v2, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v1, v1, 2             \n\t"

            "vsetvli      t0, x0, e8, m1          \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vsrl.vi      v12, v4, 6              \n\t"
            "vsrl.vi      v13, v5, 6              \n\t"
            "vsrl.vi      v14, v6, 6              \n\t"
            "vsrl.vi      v15, v7, 6              \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v1, v12, i8        \n\t"
            "vmadot       v18, v1, v13, i8        \n\t"
            "vmadot       v20, v1, v14, i8        \n\t"
            "vmadot       v22, v1, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v18, v2, x0             \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v19, v18, x0            \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v30, v16, v19           \n\t"

            "addi         t3, t3, -1              \n\t"
            "bgtz         t3, K64_LPST%=          \n\t"
            "K64_LPND%=:                          \n\t"

            // load A scale (fp32) and advance A to next superblock
            "flw          f0, (s2)                \n\t"
            "addi         s2, s2, 4+32+256        \n\t"
            "add          t4, s7, %[B_STR]        \n\t"  // t4 = next B blk base
            "addi         s3, s2, 4+32            \n\t"

            // load B scales16[32] (fp16) at end of qs region
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vle16.v      v2, (s6)                \n\t"

            // pointer modify
            "addi         s5, t4, 32*16           \n\t"
            "mv           s4, t4                  \n\t"
            "addi         s6, s5, 32*32           \n\t"
            "addi         s7, t4, 0               \n\t"

            // b_scale fp16 -> fp32
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vfwcvt.f.f.v v24, v2                 \n\t"

            // a_scale * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v v26, v30                 \n\t"
            "vfmul.vf     v1, v24, f0             \n\t"
            "vsetvli      t0, x0, e32, m1         \n\t"
            // static_cast<float>(qsum) * a_scale * b_scale;
            "vfmacc.vv    v31, v1, v26            \n\t"

            // next K-superblock
            "addi         t2, t2, -1              \n\t"
            "vxor.vv      v30, v0, v0             \n\t"  // clear acc of K256
            "li           t3, 4                   \n\t"
            "bgtz         t2, BLK_LPST%=          \n\t"

            "BLK_LPND%=:                          \n\t"
            "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
            "vse32.v      v31, (%[pC])            \n\t"
            "FUNC_END%=:                          \n\t"

            :
            : [KBLKS] "r"(k_blks), [NBLKS] "r"(nb_real), [pA] "r"(quant_a_ptr), [pB] "r"(quant_b_blk_data),
              [pC] "r"(c_ptr), [B_STR] "r"(b_ncol_block_stride)
            : "cc", "memory", "t0", "t2", "t3", "t4", "t5", "f0", "s2", "s3", "s4", "s5", "s6", "s7");
#else

        __asm__ volatile(
            // =========================
            // Kernel overview (M1 x N32)
            // =========================
            // Process one output row (M=1) and 32 columns (N=32) per call.
            //
            // Loop structure:
            //   - Outer loop: K superblocks of size K=256 (k_blks times)
            //   - Each K256 superblock is broken into 4 x K64
            //   - Each K64 is processed as 4 x K16 "sub-blocks" (via unpack+dot)
            //
            // Data layout (high level):
            //   A (q8k K=256, per superblock):
            //     [ fp32 a_scale ][ int16 a_sum[16] ][ int8 a_qs[256] ]
            //   B (nrow_block_q3_k<32>, per superblock):
            //     [ int8  scales[32*16] ][ hmask[1024] ][ qs[2048] ][ fp16 scales16[32] ]
            //
            // Registers/pointers:
            //   s2: pA (points at A superblock header; used to load fp32 a_scale)
            //   s3: pA_qs (points at A int8 data within the current superblock)
            //   s4: pB_scales (points at B int8 per-K16 scales)
            //   s5: pB_hmask (points at B sign mask area)
            //   s6: pB_qs (points at B 2-bit packed qs area)
            //   s8: pB_scales16 (points at B fp16 scales16[32] at the end of block)
            //   s7: pB_base (base pointer to current B block; used for block-to-block stride)

            // t2 = number of K256 superblocks
            "mv           t2, %[KBLKS]            \n\t"
            // t3 = number of K64 chunks per K256 superblock (256 / 64)
            "li           t3, 4                   \n\t"

            // A pointers
            "mv           s2, %[pA]               \n\t"  // s2 = pA_superblock (a_scale at +0)
            "addi         s3, %[pA], 4+32         \n\t"  // s3 = pA_qs (skip a_scale + a_sum[16])

            // B pointers for nrow_block_q3_k<32>
            "addi         s5, %[pB], 32*16        \n\t"  // s5 = pB_hmask  (skip scales[32*16])
            "mv           s4, %[pB]               \n\t"  // s4 = pB_scales
            "addi         s6, s5, 1024            \n\t"  // s6 = pB_qs     (skip hmask)
            // scales16 is at the end of the block: qs(2048) after hmask
            "addi         s8, s6, 1024            \n\t"
            "addi         s8, s8, 1024            \n\t"  // s8 = pB_scales16 (fp16 scales16[32])
            "mv           s7, %[pB]               \n\t"  // s7 = pB_base (for next-block address calc)

            // v31: final FP32 accumulator for N=32
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v31, v0, v0             \n\t"

            // ---- Preload B scales16[32] and build FP16 scale vector used by vmadot.hp ----
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vle16.v      v1, (s8)                \n\t"  // load fp16 scales16[32]
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v26, v1, v1, 3          \n\t"  // broadcast/pack to match lanes
            "vmv.v.v      v17, v26                \n\t"
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vfmul.vf     v30, v17, %[q3_step]    \n\t"  // v30 = scales16 * (1/16)

            // v24-v27: fp16 partial accumulators for a K64 chunk (vmadot.hp outputs)
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v24, v16, v16           \n\t"
            "vxor.vv      v25, v16, v16           \n\t"
            "vxor.vv      v26, v16, v16           \n\t"
            "vxor.vv      v27, v16, v16           \n\t"

            // HP vmadot: vle*10 vecIns*38 vmadot.hp*16
            ".align 4                             \n\t"
            "BLK_LPST%=:                          \n\t"  // loop over K256 superblocks
            "K64_LPST%=:                          \n\t"  // loop over 4 x K64 chunks

            // ------------------------------------------------------------
            // K0-15: load B scales + {hmask, qs} + A data; unpack and dot
            // ------------------------------------------------------------
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vle8.v       v2, (s4)                \n\t"  // B int8 scales for this K16
            "addi         s4, s4, 128             \n\t"

            "vle8.v       v4, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v5, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v6, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v7, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"

            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"  // B hmask for this K16
            "addi         s5, s5, 64              \n\t"

            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v3, (s3)                \n\t"  // A int8 data for this K16
            "addi         s3, s3, 64              \n\t"

            // Convert B int8 scales to FP16 and apply scales16*(1/16)
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vfwcvt.f.x.v v28, v2                 \n\t"  // int8 -> fp16
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vfmul.vv     v1, v28, v30            \n\t"  // v1: FP16 scale vector for vmadot.hp
            "vfmul.vv     v29, v29, v30           \n\t"

            // Unpack B 2-bit qs + hmask -> signed int8 in v12..v15
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vand.vi      v12, v4, 0x3            \n\t"
            "vand.vi      v13, v5, 0x3            \n\t"
            "vand.vi      v14, v6, 0x3            \n\t"
            "vand.vi      v15, v7, 0x3            \n\t"
            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            // (Next K16 unpack path uses a fresh hmask load)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // Prepare another group from packed qs (bit shifts) + apply sign from hmask
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v8, v4, 4               \n\t"
            "vsll.vi      v9, v5, 4               \n\t"
            "vsll.vi      v10, v6, 4              \n\t"
            "vsll.vi      v11, v7, 4              \n\t"
            "vsrl.vi      v16, v8, 6              \n\t"
            "vsrl.vi      v17, v9, 6              \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vsrl.vi      v18, v10, 6             \n\t"
            "vsrl.vi      v19, v11, 6             \n\t"
            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v16, v16, -4, v0.t      \n\t"

            // A shift for the second dot within this K64
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v2, v3, 2             \n\t"

            // Dot products with FP16 scaling (accumulate into v24..v27)
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmadot.hp    v24, v3, v12, v1, 0, i8 \n\t"
            "vmadot.hp    v25, v3, v13, v1, 1, i8 \n\t"
            "vmadot.hp    v26, v3, v14, v1, 2, i8 \n\t"
            "vmadot.hp    v27, v3, v15, v1, 3, i8 \n\t"
            "vmadot.hp    v24, v2, v16, v1, 4, i8 \n\t"
            "vmadot.hp    v25, v2, v17, v1, 5, i8 \n\t"
            "vmadot.hp    v26, v2, v18, v1, 6, i8 \n\t"
            "vmadot.hp    v27, v2, v19, v1, 7, i8 \n\t"

            // (K32-47 / K48-63 blocks continue unchanged...)
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vmv.v.v      v1, v29                 \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v3, v3, 4             \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v8, v4, 2               \n\t"
            "vsll.vi      v9, v5, 2               \n\t"
            "vsll.vi      v10, v6, 2              \n\t"
            "vsll.vi      v11, v7, 2              \n\t"

            "vsrl.vi      v20, v8, 6              \n\t"
            "vsrl.vi      v21, v9, 6              \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vsrl.vi      v22, v10, 6             \n\t"
            "vsrl.vi      v23, v11, 6             \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v20, v20, -4, v0.t      \n\t"

            // K48-63
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsrl.vi      v8, v4, 6               \n\t"
            "vsrl.vi      v9, v5, 6               \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vsrl.vi      v10, v6, 6              \n\t"
            "vsrl.vi      v11, v7, 6              \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v8, v8, -4, v0.t        \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, mf2        \n\t"
            "vslidedown.vi  v2, v3, 2             \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmadot.hp    v24, v3, v20, v1, 0, i8 \n\t"
            "vmadot.hp    v25, v3, v21, v1, 1, i8 \n\t"
            "vmadot.hp    v26, v3, v22, v1, 2, i8 \n\t"
            "vmadot.hp    v27, v3, v23, v1, 3, i8 \n\t"
            "vmadot.hp    v24, v2, v8, v1, 4, i8  \n\t"
            "vmadot.hp    v25, v2, v9, v1, 5, i8  \n\t"
            "vmadot.hp    v26, v2, v10, v1, 6, i8 \n\t"
            "vmadot.hp    v27, v2, v11, v1, 7, i8 \n\t"

            "addi         t3, t3, -1              \n\t"
            "bgtz         t3, K64_LPST%=          \n\t"
            "K64_LPND%=:                          \n\t"

            // ---- End of K64 chunk: reduce fp16 accumulators -> fp32 and scale by A ----
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v12, v24, v25, 1        \n\t"
            "vpack.vv     v14, v26, v27, 1        \n\t"
            "vpack.vv     v16, v12, v14, 2        \n\t"
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vfwcvt.f.f.v v26, v16                \n\t"  // fp16 -> fp32 vector (qsum * b_scales)

            // Load A scale and advance A pointer to next K256 superblock
            "flw          f0, (s2)                \n\t"
            "addi         s2, s2, 4+32+256        \n\t"
            "add          t4, s7, %[B_STR]        \n\t"  // next B block base
            "addi         s3, s2, 4+32            \n\t"  // reset A data pointer for next block

            // Advance B pointers to next K256 superblock
            "addi         s5, t4, 32*16           \n\t"
            "mv           s4, t4                  \n\t"
            "addi         s6, s5, 32*32           \n\t"
            "addi         s8, s6, 1024            \n\t"
            "addi         s8, s8, 1024            \n\t"
            "addi         s7, t4, 0               \n\t"
            "addi         t2, t2, -1              \n\t"

            // Final per-block scaling: a_scale * 16.0f
            "fmul.s       f0, f0, %[a_post_mul]   \n\t"
            // acc += (qsum * b_scales) * (a_scale*16)
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfmacc.vf    v31, f0, v26            \n\t"

            "beqz         t2, BLK_LPND%=          \n\t"

            // Preload next block's scales16 and rebuild v30 for vmadot.hp
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vle16.v      v1, (s8)                \n\t"
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v26, v1, v1, 3          \n\t"
            "vmv.v.v      v17, v26                \n\t"
            "vsetvli      t0, x0, e16, m1         \n\t"
            "vfmul.vf     v30, v17, %[q3_step]    \n\t"

            // Reset fp16 partial accumulators for next K64 loop(s)
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v24, v16, v16           \n\t"
            "vxor.vv      v25, v16, v16           \n\t"
            "vxor.vv      v26, v16, v16           \n\t"
            "vxor.vv      v27, v16, v16           \n\t"

            "li           t3, 4                   \n\t"
            "bgtz         t2, BLK_LPST%=          \n\t"

            "BLK_LPND%=:                          \n\t"
            "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
            "vse32.v      v31, (%[pC])            \n\t"

            :
            : [KBLKS] "r"(k_blks), [NBLKS] "r"(nb_real), [pA] "r"(quant_a_ptr), [pB] "r"(quant_b_blk_data),
              [pC] "r"(c_ptr), [B_STR] "r"(b_ncol_block_stride), [q3_step] "f"(k_q3k_scale_step),
              [a_post_mul] "f"(k_a_scale_post_mul)
            : "cc", "memory", "t0", "t2", "t3", "t4", "t5", "f0", "f1", "s2", "s3", "s4", "s5", "s6", "s7", "s8");
#endif
    }
}

void gemm_kernel_i8i3k_m4(size_t          blk_len,
                          const uint8_t * quant_a_ptr,
                          const uint8_t * quant_b_data,
                          float *         c_ptr,
                          size_t          count_m,
                          size_t          count_n,
                          size_t          k_blks,
                          size_t          ldc) {
    using blk_type           = nrow_block_q3_k<32>;
    constexpr size_t NB_COLS = 32;  //only support 32 in ASM

    const blk_type * b_base = reinterpret_cast<const blk_type *>(quant_b_data);

    int64_t a_blk_stride        = q8k_blk_size(256);
    int64_t a_nrow_block_stride = a_blk_stride * 4;
    int64_t b_ncol_block_stride = sizeof(blk_type);

    for (size_t ni = 0; ni < count_n; ni += NB_COLS, c_ptr += NB_COLS) {
        size_t           nb_real          = std::min<size_t>(NB_COLS, count_n - ni);
        const blk_type * quant_b_blk_data = b_base + (ni / NB_COLS) * k_blks;

        //------------------------------------------------------------------------------
        // A format
        // Ascale   fp32 * 1* 4row    128bit
        // Asum     int16 * 16 4row  1024bit
        // A M1K256 int8 4row        8192bit
        //------------------------------------------------------------------------------
        // B format
        // B_scl    uint8*N32*16    4096bit
        // B_Hmask  N32K16*16 1bit  8192bit
        // B_Qs     N32K16*16 2bit  16384bit
        // B scl16  fp16 * N32      512bit;
        //------------------------------------------------------------------------------
        //bias always be nullptr
        __asm__ volatile(
            // t2 = k_blks (each is K256 superblock)
            "mv           t2, %[KBLKS]            \n\t"
            // t3 = 256/64 = 4 (K64 iterations per superblock)
            "li           t3, 4                   \n\t"
            "mv           s2, %[pA]               \n\t"  // s2 = pASCL
            "addi         s3, %[pA], 16+128       \n\t"  // s3 = pAData, (pA+AScl+ASum)

            // B block layout for nrow_block_q3_k<32>:
            // scales: 512B, hmask: 1024B, qs: 2048B, scales16: 64B
            "addi         s5, %[pB], 32*16        \n\t"  // s5 = pB_hmask (skip scales)
            "mv           s4, %[pB]               \n\t"  // s4 = pB_scales
            "addi         s6, s5, 1024            \n\t"  // s6 = pB_qs (skip hmask)
            "mv           s7, %[pB]               \n\t"  // s7 = pB_base

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v24, v0, v0             \n\t"  // v24-v27: K256 temp accumulator
            "vxor.vv      v25, v0, v0             \n\t"
            "vxor.vv      v26, v0, v0             \n\t"
            "vxor.vv      v27, v0, v0             \n\t"
            "vxor.vv      v28, v0, v0             \n\t"  // v28-v31: final accumulator
            "vxor.vv      v29, v0, v0             \n\t"
            "vxor.vv      v30, v0, v0             \n\t"
            "vxor.vv      v31, v0, v0             \n\t"

            // ordinary vmadot: vle*13 vecIns*96 vmadot*16
            ".align 4                             \n\t"
            "BLK_LPST%=:                          \n\t"
            "K64_LPST%=:                          \n\t"

            // ========== K0-15: First K16 sub-block ==========
            // Load B INT8 scale factors (32 cols × 16 K16 blocks)
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vle8.v       v8, (s4)                \n\t"
            "addi         s4, s4, 128             \n\t"

            // Load B quantized data (32 cols × 16 elements × 2bit, stored in 4 groups)
            "vle8.v       v4, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v5, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v6, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"
            "vle8.v       v7, (s6)                \n\t"
            "addi         s6, s6, 128             \n\t"

            // Load B hmask (32 cols × 16bit sign mask)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // Load A data (4 rows × 16 elements × INT8)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v12, (s3)               \n\t"
            "addi         s3, s3, 256             \n\t"  // Jump to next row
            "vle8.v       v13, (s3)               \n\t"
            "addi         s3, s3, 256             \n\t"
            "vle8.v       v14, (s3)               \n\t"
            "addi         s3, s3, 256             \n\t"
            "vle8.v       v15, (s3)               \n\t"
            "addi         s3, s3, -768+64         \n\t"  // Back to first row, advance 16 elements

            // Pack A data: merge 4 rows into 2 vectors
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vpack.vv     v16, v12, v13, 1        \n\t"
            "vpack.vv     v18, v14, v15, 1        \n\t"
            "vpack.vv     v2, v16, v18, 2         \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vand.vi      v12, v4, 0x3            \n\t"
            "vand.vi      v13, v5, 0x3            \n\t"
            "vand.vi      v14, v6, 0x3            \n\t"
            "vand.vi      v15, v7, 0x3            \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v2, v12, i8        \n\t"  // 4 rows × cols 0-7
            "vmadot       v18, v2, v13, i8        \n\t"  // 4 rows × cols 8-15
            "vmadot       v20, v2, v14, i8        \n\t"  // 4 rows × cols 16-23
            "vmadot       v22, v2, v15, i8        \n\t"  // 4 rows × cols 24-31

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v12, v16, v18, 2        \n\t"  // Merge cols 0-15
            "vpack.vv     v14, v20, v22, 2        \n\t"  // Merge cols 16-31
            "vpack.vv     v16, v12, v14, 3        \n\t"  // Inter-row results (INT16)
            "vpack.vv     v18, v13, v15, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v21, v8, x0             \n\t"  // INT8 → INT16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v23, v21, x0            \n\t"  // INT16 → INT32

            // Accumulate to K256 accumulator: qsum * b_scale
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v24, v16, v23           \n\t"  // Row 0
            "vmacc.vv     v25, v17, v23           \n\t"  // Row 1
            "vmacc.vv     v26, v18, v23           \n\t"  // Row 2
            "vmacc.vv     v27, v19, v23           \n\t"

            // ========== K16-31, K32-47, K48-63: Similar processing ==========
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v8, v8, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v2, v2, 8             \n\t"

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v12, v4, 4              \n\t"
            "vsll.vi      v13, v5, 4              \n\t"
            "vsll.vi      v14, v6, 4              \n\t"
            "vsll.vi      v15, v7, 4              \n\t"
            "vnot.v       v0, v0                  \n\t"

            "vsrl.vi      v12, v12, 6             \n\t"
            "vsrl.vi      v13, v13, 6             \n\t"
            "vsrl.vi      v14, v14, 6             \n\t"
            "vsrl.vi      v15, v15, 6             \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v2, v12, i8        \n\t"
            "vmadot       v18, v2, v13, i8        \n\t"
            "vmadot       v20, v2, v14, i8        \n\t"
            "vmadot       v22, v2, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v12, v16, v18, 2        \n\t"
            "vpack.vv     v14, v20, v22, 2        \n\t"
            "vpack.vv     v16, v12, v14, 3        \n\t"  // N0-N31 in v16
            "vpack.vv     v18, v13, v15, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v21, v8, x0             \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v23, v21, x0            \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v24, v16, v23           \n\t"
            "vmacc.vv     v25, v17, v23           \n\t"
            "vmacc.vv     v26, v18, v23           \n\t"
            "vmacc.vv     v27, v19, v23           \n\t"

            //K32-47
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v8, v8, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)

            // unpack 2-bit qs + hmask -> signed values
            "vsetvli      t0, x0, e8, m1          \n\t"
            "vsll.vi      v12, v4, 2              \n\t"
            "vsll.vi      v13, v5, 2              \n\t"
            "vsll.vi      v14, v6, 2              \n\t"
            "vsll.vi      v15, v7, 2              \n\t"
            "vnot.v       v0, v0                  \n\t"

            "vsrl.vi      v12, v12, 6             \n\t"
            "vsrl.vi      v13, v13, 6             \n\t"
            "vsrl.vi      v14, v14, 6             \n\t"
            "vsrl.vi      v15, v15, 6             \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v3, v12, i8        \n\t"
            "vmadot       v18, v3, v13, i8        \n\t"
            "vmadot       v20, v3, v14, i8        \n\t"
            "vmadot       v22, v3, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v12, v16, v18, 2        \n\t"
            "vpack.vv     v14, v20, v22, 2        \n\t"
            "vpack.vv     v16, v12, v14, 3        \n\t"  // N0-N31 in v16
            "vpack.vv     v18, v13, v15, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v21, v8, x0             \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v23, v21, x0            \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v24, v16, v23           \n\t"
            "vmacc.vv     v25, v17, v23           \n\t"
            "vmacc.vv     v26, v18, v23           \n\t"
            "vmacc.vv     v27, v19, v23           \n\t"

            // K48-63
            // load B scales (32 bytes per K16, 16 times => 512B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v8, v8, 4             \n\t"

            // load B hmask chunk (64B per K16, 16 times => 1024B)
            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s5)                \n\t"
            "addi         s5, s5, 64              \n\t"

            // load A data (16 bytes per K16, 16 times => 256B)
            "vsetvli      t0, x0, e64, m1         \n\t"
            "vslidedown.vi  v3, v3, 8             \n\t"

            "vsetvli      t0, x0, e8, m1          \n\t"
            "vnot.v       v0, v0                  \n\t"
            "vsrl.vi      v12, v4, 6              \n\t"
            "vsrl.vi      v13, v5, 6              \n\t"
            "vsrl.vi      v14, v6, 6              \n\t"
            "vsrl.vi      v15, v7, 6              \n\t"

            "vsetvli      t0, x0, e8, m4          \n\t"
            "vadd.vi      v12, v12, -4, v0.t      \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v3, v12, i8        \n\t"
            "vmadot       v18, v3, v13, i8        \n\t"
            "vmadot       v20, v3, v14, i8        \n\t"
            "vmadot       v22, v3, v15, i8        \n\t"

            "vsetvli      t0, x0, e16, m1         \n\t"
            "vpack.vv     v12, v16, v18, 2        \n\t"
            "vpack.vv     v14, v20, v22, 2        \n\t"
            "vpack.vv     v16, v12, v14, 3        \n\t"  // N0-N31 in v16
            "vpack.vv     v18, v13, v15, 3        \n\t"

            // apply B int8 scales (-32 bias has been applyed)
            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vwadd.vx     v21, v8, x0             \n\t"  // int8 -> int16

            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vwadd.vx     v23, v21, x0            \n\t"  // int8 -> int16

            // static_cast<int32_t>(qsum) * b_scale;
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vmacc.vv     v24, v16, v23           \n\t"
            "vmacc.vv     v25, v17, v23           \n\t"
            "vmacc.vv     v26, v18, v23           \n\t"
            "vmacc.vv     v27, v19, v23           \n\t"

            "addi         t3, t3, -1              \n\t"
            "bgtz         t3, K64_LPST%=          \n\t"
            "K64_LPND%=:                          \n\t"

            // ========== K256 superblock complete, apply scale factors ==========
            // Load A's 4 row scale factors (FP32)
            "flw          f0, (s2)                \n\t"
            "flw          f1, 4(s2)               \n\t"
            "flw          f2, 8(s2)               \n\t"
            "flw          f3, 12(s2)              \n\t"
            "add          s2, s2, %[A_STR]        \n\t"  // Advance to next superblock
            "add          t4, s7, %[B_STR]        \n\t"  // t4 = next B block address
            "addi         s3, s2, (4+32)*4        \n\t"

            // Load B FP16 global scale factors (32 cols)
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vle16.v      v8, (s6)                \n\t"

            // Update B pointers to next block
            "addi         s5, t4, 32*16           \n\t"
            "mv           s4, t4                  \n\t"
            "addi         s6, s5, 32*32           \n\t"
            "addi         s7, t4, 0               \n\t"

            // ========== Type conversion and final scaling ==========
            // FP16 → FP32
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vfwcvt.f.f.v v9, v8                 \n\t"

            // INT32 → FP32
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v  v24, v24                \n\t"
            "vfcvt.f.x.v  v25, v25                \n\t"
            "vfcvt.f.x.v  v26, v26                \n\t"
            "vfcvt.f.x.v  v27, v27                \n\t"

            // Compute a_scale * b_scale (4 rows)
            "vfmul.vf     v12, v9, f0             \n\t"
            "vfmul.vf     v13, v9, f1             \n\t"
            "vfmul.vf     v14, v9, f2             \n\t"
            "vfmul.vf     v15, v9, f3             \n\t"

            // Final accumulation: result += qsum * a_scale * b_scale
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfmacc.vv    v28, v12, v24           \n\t"
            "vfmacc.vv    v29, v13, v25           \n\t"
            "vfmacc.vv    v30, v14, v26           \n\t"
            "vfmacc.vv    v31, v15, v27           \n\t"

            // Prepare for next K superblock
            "addi         t2, t2, -1              \n\t"
            "vxor.vv      v24, v0, v0             \n\t"  // Clear K256 accumulator
            "vxor.vv      v25, v0, v0             \n\t"
            "vxor.vv      v26, v0, v0             \n\t"
            "vxor.vv      v27, v0, v0             \n\t"
            "li           t3, 4                   \n\t"
            "bgtz         t2, BLK_LPST%=          \n\t"

            "BLK_LPND%=:                          \n\t"

            // ========== Store results (4 rows × 32 cols) ==========
            "mv           t5, %[pC]               \n\t"
            "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
            "vse32.v      v28, (%[pC])            \n\t"
            "add          t5, t5, %[LDC]          \n\t"
            "vse32.v      v29, (t5)               \n\t"
            "add          t5, t5, %[LDC]          \n\t"
            "vse32.v      v30, (t5)               \n\t"
            "add          t5, t5, %[LDC]          \n\t"
            "vse32.v      v31, (t5)               \n\t"
            "add          t5, t5, %[LDC]          \n\t"
            "FUNC_END%=:                          \n\t"

            :
            : [KBLKS] "r"(k_blks), [NBLKS] "r"(nb_real), [pA] "r"(quant_a_ptr), [pB] "r"(quant_b_blk_data),
              [pC] "r"(c_ptr), [B_STR] "r"(b_ncol_block_stride), [A_STR] "r"(a_nrow_block_stride), [LDC] "r"(ldc * 4)
            : "cc", "memory", "t0", "t2", "t3", "t4", "t5", "f0", "f1", "f2", "f3", "s2", "s3", "s4", "s5", "s6", "s7");
    }
}

void gemm_kernel_i8i4_m1(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    if (quant_b_zp == NULL) {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * blk_len / 2 +      // b data
                                      n * k_blks * sizeof(_Float16);  // scale
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
            // A M1K32 int8    256bit
            // Ascale fp32 * 1  32bit
            // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
            // || Element                          || Element                          || ...
            // B format
            // B N8K32 int4    1024bit
            //   4VRF, N32K32, 4096bit
            // Bscale fp16 * N32 512bit;
            // || scl*32..(fp16) | blk0 blk1 ... blk31 || scl*32..(fp16) | blk0 blk1 ... blk31 || ...
            // || Element                              || Element                              || ...
#if 0
            //bias always be nullptr
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBdata;
                "mv           s6, %[pC]               \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc

                // ordinary vmadot: vle*6 flw*1 vecIns*21 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+64        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v0, (s4)                \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64+128*4        \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu     v16, v10, v4, i4        \n\t"  // M0 N0 - N7 INT32(256bit)
                "vmadotsu     v18, v10, v5, i4        \n\t"  // M0 N8 - N15
                "vmadotsu     v20, v10, v6, i4        \n\t"  // M0 N16 - N23
                "vmadotsu     v22, v10, v7, i4        \n\t"  // M0 N24 - N31

                "vsll.vi      v16, v16, 4             \n\t"
                "vsll.vi      v18, v18, 4             \n\t"
                "vsll.vi      v20, v20, 4             \n\t"
                "vsll.vi      v22, v22, 4             \n\t"

                "vmadotu      v16, v8, v4, i4         \n\t"
                "vmadotu      v18, v8, v5, i4         \n\t"
                "vmadotu      v20, v8, v6, i4         \n\t"
                "vmadotu      v22, v8, v7, i4         \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v28, 8                  \n\t"
                "vpack.vv     v24, v16, v18, 2        \n\t"
                "vpack.vv     v26, v20, v22, 2        \n\t"
                "vpack.vv     v16, v24, v26, 3        \n\t"

                "vwmul.vx     v24, v28, t2            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v16, v16, v24           \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v24, v0                 \n\t"
                // mac result i32 -> fp32
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v16                \n\t"
                // a_scale * b_scale;
                "vfmul.vf     v1, v24, f0             \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v2, v1, v26             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#else
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // init the scale
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBdata;
                "mv           s6, %[pC]               \n\t"

                "vsll.vi      v1, v0, 4               \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc
                "vfcvt.f.x.v  v0, v0                  \n\t"
                "vfcvt.f.x.v  v1, v1                  \n\t"

                // vmadot hp: vle*7 flw*1 vecIns*14 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+64        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v30, (s4)               \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64+128*4        \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v28, 8                  \n\t"  // Bzp u8 -> u16
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmul.vx      v26, v28, t2            \n\t"  // asum*zp i16*i16
                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vfcvt.f.x.v  v16, v26                \n\t"  // zp i16 -> fp16
                "vadd.vi      v18, v16, 0             \n\t"
                "vadd.vi      v20, v16, 0             \n\t"
                "vadd.vi      v22, v16, 0             \n\t"

                "vmadotsu.hp  v16, v10, v4, v1, 0, i4 \n\t"  // high 4
                "vmadotsu.hp  v18, v10, v5, v1, 0, i4 \n\t"
                "vmadotsu.hp  v20, v10, v6, v1, 0, i4 \n\t"
                "vmadotsu.hp  v22, v10, v7, v1, 0, i4 \n\t"
                "vmadotu.hp   v16, v8, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp   v18, v8, v5, v0, 0, i4  \n\t"
                "vmadotu.hp   v20, v8, v6, v0, 0, i4  \n\t"
                "vmadotu.hp   v22, v8, v7, v0, 0, i4  \n\t"

                "vpack.vv     v24, v16, v18, 1        \n\t"
                "vpack.vv     v26, v20, v22, 1        \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"

                "vsetvli      t0, x0, e16, mf2        \n\t"
                // mac result * b_scale; f16*f16->f32
                "vfwmul.vv     v31, v30, v16          \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum * b_scale) * a_scale;
                "vfmacc.vf    v2, f0, v31             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");

#endif
        }
    } else {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * blk_len / 2 +      // b data
                                      n * k_blks * sizeof(uint8_t) +  // b zp
                                      n * k_blks * sizeof(_Float16);  // scale
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
            // A M1K32 int8    256bit
            // Ascale fp32 * 1  32bit
            // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
            // || Element                          || Element                          || ...
            // B format
            // B N8K32 int4    1024bit
            //   4VRF, N32K32, 4096bit
            // Bscale fp16 * N32 512bit;
            // Bzp uint8_t * N32 256bit;
            // || scl*32..(fp16) | zp*32(uint8) | blk0 blk1 ... blk31 || scl*32..(fp16)  ...
            // || Element                                             || Element         ...

            //bias always be nullptr
#if 0
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t"  // s5 = pBdata, (pB+BScl+Bzp)
                "mv           s6, %[pC]               \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc

                // ordinary vmadot: vle*6 flw*1 vecIns*21 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+96        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v0, (s4)                \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64              \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu     v16, v10, v4, i4        \n\t"  // M0 N0 - N7 INT32(256bit)
                "vmadotsu     v18, v10, v5, i4        \n\t"  // M0 N8 - N15
                "vmadotsu     v20, v10, v6, i4        \n\t"  // M0 N16 - N23
                "vmadotsu     v22, v10, v7, i4        \n\t"  // M0 N24 - N31

                "vsll.vi      v16, v16, 4             \n\t"
                "vsll.vi      v18, v18, 4             \n\t"
                "vsll.vi      v20, v20, 4             \n\t"
                "vsll.vi      v22, v22, 4             \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v1, (s4)                \n\t"  // Bzp
                "addi         s4, s4, 32+128*4        \n\t"

                "vmadotu      v16, v8, v4, i4         \n\t"
                "vmadotu      v18, v8, v5, i4         \n\t"
                "vmadotu      v20, v8, v6, i4         \n\t"
                "vmadotu      v22, v8, v7, i4         \n\t"

                "vwaddu.vx    v28, v1, x0             \n\t"  // uint8 -> uint16
                "vpack.vv     v24, v16, v18, 2        \n\t"
                "vpack.vv     v26, v20, v22, 2        \n\t"
                "vpack.vv     v16, v24, v26, 3        \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vwmul.vx     v24, v28, t2            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v16, v16, v24           \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v24, v0                 \n\t"
                // mac result i32 -> fp32
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v16                \n\t"
                // a_scale * b_scale;
                "vfmul.vf     v1, v24, f0             \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v2, v1, v26             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#else
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // init the scale
                "mv           s2, %[pA]               \n\t"  // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t"  // s5 = pBdata, (pB+BScl+Bzp)
                "mv           s6, %[pC]               \n\t"

                "vsll.vi      v1, v0, 4               \n\t"
                "vxor.vv      v2, v0, v0              \n\t"  // clear acc
                "vfcvt.f.x.v  v0, v0                  \n\t"
                "vfcvt.f.x.v  v1, v1                  \n\t"

                // vmadot hp: vle*6 flw*1 vecIns*14 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+96        \n\t"  // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v30, (s4)               \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64              \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v31, (s4)               \n\t"  // B zp 32Row*uint8 = 256bit
                "addi         s4, s4, 32+128*4        \n\t"

                "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t"  // A Scale fp32
                "lh           t2, 4(s2)               \n\t"  // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vnpack4.vv   v8, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t"  // hi4 of A

                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu.hp  v16, v10, v4, v1, 0, i4 \n\t"  // high 4
                "vmadotsu.hp  v18, v10, v5, v1, 0, i4 \n\t"
                "vmadotsu.hp  v20, v10, v6, v1, 0, i4 \n\t"
                "vmadotsu.hp  v22, v10, v7, v1, 0, i4 \n\t"
                "vmadotu.hp   v16, v8, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp   v18, v8, v5, v0, 0, i4  \n\t"
                "vmadotu.hp   v20, v8, v6, v0, 0, i4  \n\t"
                "vmadotu.hp   v22, v8, v7, v0, 0, i4  \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vwaddu.vx    v28, v31, x0            \n\t"  // Bzp u8 -> u16

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v24, v16, v18, 1        \n\t"
                "vpack.vv     v26, v20, v22, 1        \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"

                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vmul.vx      v26, v28, t2            \n\t"  // asum*zp i16*i16
                "vfwcvt.f.f.v v22, v30                \n\t"  // b_scale fp16 -> fp32
                "vfcvt.f.x.v  v18, v26                \n\t"  // zp i16 -> fp16
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vfwadd.vv    v20, v18, v16           \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // mac result * b_scale; f32*f32->f32
                "vfmul.vv     v31, v22, v20           \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum * b_scale) * a_scale;
                "vfmacc.vf    v2, f0, v31             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#endif
        }
    }
}

void gemm_kernel_i8i4_hp_m1(size_t          blk_len,
                            const uint8_t * quant_a_ptr,
                            const uint8_t * quant_b_data,
                            const uint8_t * quant_b_zp,
                            float *         c_ptr,
                            size_t          count_m,
                            size_t          count_n,
                            size_t          k_blks,
                            size_t          ldc) {
    constexpr size_t NB_COLS                = 32;
    constexpr size_t k_subblks_per_superblk = 8;

    struct block_q4_0x32_layout {
        _Float16 d[NB_COLS];
        uint8_t  qs[16 * NB_COLS];
    };

    GGML_ASSERT(blk_len == 256);

    const size_t b_superblk_stride = sizeof(block_q4_0x32_layout) * k_subblks_per_superblk +
                                     (quant_b_zp ? NB_COLS * k_subblks_per_superblk * sizeof(uint8_t) : 0);
    const size_t b_tile_stride = k_blks * b_superblk_stride;

    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_tile_stride;
            int8_t *  a_data = (int8_t *) quant_a_ptr;
            float *   dst_c  = c_ptr + ni;

            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v31, v31, v31           \n\t"  // init acc to zero
                "mv             t4, %[BK]               \n\t"
                "li             t0, 0x4c00              \n\t"  // 16 in fp16
                "fmv.h.x        fa0, t0                 \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                "li             t5, 8                   \n\t"
                "addi           t6, %[A], 288           \n\t"  // point to blk scale
                "flh            ft1, (t6)               \n\t"
                "addi           t6, %[A], 272           \n\t"  // point to asum

                // init the acc fp16
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v16, v18, v18           \n\t"
                "vxor.vv        v17, v18, v18           \n\t"
                "vxor.vv        v18, v18, v18           \n\t"
                "vxor.vv        v19, v18, v18           \n\t"

                "INNER_BLK_LOOP%=:                      \n\t"
                // load a sum and scale
                "flh            fa1, (t6)               \n\t"
                "addi           t6, t6, 2               \n\t"
                "flh            ft0, (%[A])             \n\t"
                "addi           %[A], %[A], 2           \n\t"
                // load A
                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v3, (%[A])              \n\t"  // 1x32@i8
                "addi           %[A], %[A], 32          \n\t"

                // load scale B and B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v8, (%[B])              \n\t"  // b_scale fp16
                "addi           %[B], %[B], 64          \n\t"
                "vl4r.v         v4, (%[B])              \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"
                "vfmul.vf       v8, v8, ft0             \n\t"  // scale b * scale a
                "vfmul.vf       v9, v8, fa0             \n\t"
                "vfmul.vf       v10, v8, fa1            \n\t"  // scale b * scale a * asm
                "vfwmacc.vf     v31, ft1, v10           \n\t"  // asum * scale a * scale b * blk scale

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vpack.vv       v0, v8, v9, 3           \n\t"
                "vsrl.vi        v28, v3, 4              \n\t"

                "vsetvli        t0, x0, e16, m1         \n\t"
                "vnpack4.vv     v2, v3, v3, 3           \n\t"  // lo4 of A
                "vnpack4.vv     v3, v28, v28, 3         \n\t"  // hi4 of A

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmadotsu.hp    v16, v3, v4, v0, 4, i4  \n\t"  // high 4
                "vmadotsu.hp    v17, v3, v5, v0, 5, i4  \n\t"
                "vmadotsu.hp    v18, v3, v6, v0, 6, i4  \n\t"
                "vmadotsu.hp    v19, v3, v7, v0, 7, i4  \n\t"
                "vmadotu.hp     v16, v2, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp     v17, v2, v5, v0, 1, i4  \n\t"
                "vmadotu.hp     v18, v2, v6, v0, 2, i4  \n\t"
                "vmadotu.hp     v19, v2, v7, v0, 3, i4  \n\t"

                "addi           t5, t5, -1              \n\t"
                "bgtz           t5, INNER_BLK_LOOP%=    \n\t"

                "vpack.vv       v8, v16, v17, 1         \n\t"
                "vpack.vv       v12, v18, v19, 1        \n\t"
                "vpack.vv       v20, v8, v12, 2         \n\t"

                "vsetvli        t0, x0, e16, mf2        \n\t"
                "addi           t4, t4, -1              \n\t"
                "vfwmacc.vf     v31, ft1, v20           \n\t"
                //"vsetvli        t0, x0, e32, m1         \n\t"
                //"vfmul.vf       v31, v31, ft1           \n\t"  // blk scale

                // update A ptr
                "addi           %[A], t6, 2             \n\t"

                "bgtz           t4, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vse32.v        v31, (%[DST])           \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                  "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "ft0", "ft1");
        }
    } else {
        // TODO: support quant_b_zp for i8i4 hp kernel
        GGML_ABORT("gemm_kernel_i8i4_hp_m1 with quant_b_zp is not supported yet");
    }
}

void gemm_kernel_i8i4_m4(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    int64_t b_data_stride =
        k_blks * (sizeof(ggml_fp16_t) + 16 * sizeof(int8_t) + (quant_b_zp != NULL ? sizeof(int8_t) : 0));
    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data = (int8_t *) quant_a_ptr;
            float *   dst_c  = c_ptr + ni;
#if 0
            asm volatile(
                "li             t1,  8              \n\t"
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vxor.vv        v28, v28, v28       \n\t"
                "vxor.vv        v29, v29, v29       \n\t"
                "vxor.vv        v30, v30, v30       \n\t"
                "vxor.vv        v31, v31, v31       \n\t"
                "mv             t4, %[BK]           \n\t"

                ".align 4                           \n\t"
                "BLK_LOOP%=:                        \n\t"
                // load scale A
                "flw            fa0, (%[A])         \n\t"
                "flw            fa1, 4(%[A])        \n\t"
                "flw            fa2, 8(%[A])        \n\t"
                "flw            fa3, 12(%[A])       \n\t"
                "addi           %[A], %[A], 16      \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2    \n\t"
                "vle16.v        v12, (%[B])         \n\t"
                "addi           %[B], %[B], 64      \n\t"
                "vfwcvt.f.f.v   v14, v12            \n\t"

                "vsetivli       t0, 4, e16, mf2     \n\t"
                "vle16.v        v8, (%[A])          \n\t"  // asum
                "addi           %[A], %[A], 8       \n\t"
                "vwmul.vx       v10, v8, t1         \n\t"  // 8*asum

                "vsetvli        t0, x0, e8, m1      \n\t"
                "vl1r.v         v0, (%[A])          \n\t"
                "addi           %[A], %[A], 128     \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])          \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512     \n\t"
                "vsrl.vi        v1, v0, 4           \n\t"
                "vnpack4.vv     v12, v0, v1, 3      \n\t"  // A low  u4
                "vupack.vv      v2, v12, v12, 2     \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vxor.vv        v16, v16, v16       \n\t"
                "vxor.vv        v18, v16, v16       \n\t"
                "vxor.vv        v20, v16, v16       \n\t"
                "vxor.vv        v22, v16, v16       \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vmadotsu       v16, v3, v4, i4     \n\t"   // high 4
                "vmadotsu       v18, v3, v5, i4     \n\t"
                "vmadotsu       v20, v3, v6, i4     \n\t"
                "vmadotsu       v22, v3, v7, i4     \n\t"
                "vsll.vi        v16, v16, 4         \n\t"
                "vsll.vi        v18, v18, 4         \n\t"
                "vsll.vi        v20, v20, 4         \n\t"
                "vsll.vi        v22, v22, 4         \n\t"
                "vmadotu        v16, v2, v4, i4     \n\t"   // low 4
                "vmadotu        v18, v2, v5, i4     \n\t"
                "vmadotu        v20, v2, v6, i4     \n\t"
                "vmadotu        v22, v2, v7, i4     \n\t"

                "vpack.vv       v0, v16, v18, 2     \n\t"
                "vpack.vv       v2, v20, v22, 2     \n\t"
                "vpack.vv       v16, v0, v2, 3      \n\t"
                "vpack.vv       v18, v1, v3, 3      \n\t"

                "vrgather.vi    v0, v10, 0          \n\t"
                "vrgather.vi    v1, v10, 1          \n\t"
                "vrgather.vi    v2, v10, 2          \n\t"
                "vrgather.vi    v3, v10, 3          \n\t"

                "vadd.vv        v16, v16, v0        \n\t"
                "vadd.vv        v17, v17, v1        \n\t"
                "vadd.vv        v18, v18, v2        \n\t"
                "vadd.vv        v19, v19, v3        \n\t"

                "vfcvt.f.x.v    v16, v16            \n\t"
                "vfcvt.f.x.v    v17, v17            \n\t"
                "vfcvt.f.x.v    v18, v18            \n\t"
                "vfcvt.f.x.v    v19, v19            \n\t"

                // mul scale
                "vfmul.vv       v16, v16, v14       \n\t"
                "vfmul.vv       v17, v17, v14       \n\t"
                "vfmul.vv       v18, v18, v14       \n\t"
                "vfmul.vv       v19, v19, v14       \n\t"

                "addi           t4, t4, -1          \n\t"
                "vfmacc.vf      v28, fa0, v16       \n\t"
                "vfmacc.vf      v29, fa1, v17       \n\t"
                "vfmacc.vf      v30, fa2, v18       \n\t"
                "vfmacc.vf      v31, fa3, v19       \n\t"

                "bgtz           t4, BLK_LOOP%=      \n\t"

                // save
                "vsetvli        t0, x0, e32, m1     \n\t"
                "add            t2, %[LDC], %[DST]  \n\t"
                "vse32.v        v28, (%[DST])       \n\t"
                "add            t3, %[LDC], t2      \n\t"
                "vse32.v        v29, (t2)           \n\t"
                "add            t2, %[LDC], t3      \n\t"
                "vse32.v        v30, (t3)           \n\t"
                "vse32.v        v31, (t2)           \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc*4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                  "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
#else
            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v29, v29, v29           \n\t"
                "vxor.vv        v30, v30, v30           \n\t"
                "vxor.vv        v31, v31, v31           \n\t"
                "vmv.v.i        v0, 1                   \n\t"  // init the scale
                "vsll.vi        v1, v0, 4               \n\t"
                "vfcvt.f.x.v    v0, v0                  \n\t"
                "vfcvt.f.x.v    v1, v1                  \n\t"
                "mv             t4, %[BK]               \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // load scale A
                "flw            fa0, (%[A])             \n\t"
                "flw            fa1, 4(%[A])            \n\t"
                "flw            fa2, 8(%[A])            \n\t"
                "flw            fa3, 12(%[A])           \n\t"
                "addi           %[A], %[A], 16          \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v12, (%[B])             \n\t"
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v14, v12, v12, 3        \n\t"

                "vsetivli       t0, 4, e16, mf2         \n\t"
                "vle16.v        v8, (%[A])              \n\t"  // asum
                "addi           %[A], %[A], 8           \n\t"
                "vsll.vi        v8, v8, 3               \n\t"  // asum * 8
                "vfcvt.f.x.v    v9, v8                  \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vrgather.vi    v10, v9, 0              \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vl1r.v         v16, (%[A])             \n\t"
                "addi           %[A], %[A], 128         \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])              \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"
                "vsrl.vi        v17, v16, 4             \n\t"
                "vnpack4.vv     v12, v16, v17, 3        \n\t"  // A low  u4
                "vupack.vv      v2, v12, v12, 2         \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v16, v10, v10,0         \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vpack.vv       v20, v16, v16,0         \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vpack.vv       v18, v20, v20, 0        \n\t"
                "vor.vv         v20, v18, v18           \n\t"
                "vor.vv         v21, v18, v18           \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmadotsu.hp    v18, v3, v4, v1, 0, i4  \n\t"  // high 4
                "vmadotsu.hp    v19, v3, v5, v1, 0, i4  \n\t"
                "vmadotsu.hp    v20, v3, v6, v1, 0, i4  \n\t"
                "vmadotsu.hp    v21, v3, v7, v1, 0, i4  \n\t"
                "vmadotu.hp     v18, v2, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp     v19, v2, v5, v0, 0, i4  \n\t"
                "vmadotu.hp     v20, v2, v6, v0, 0, i4  \n\t"
                "vmadotu.hp     v21, v2, v7, v0, 0, i4  \n\t"

                "vpack.vv       v8, v18, v19, 1         \n\t"
                "vpack.vv       v12, v20, v21, 1        \n\t"
                "vpack.vv       v20, v8, v12, 2         \n\t"

                "vfwmul.vv      v16, v20, v14           \n\t"
                "vfwmul.vv      v18, v21, v14           \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"

                "addi           t4, t4, -1              \n\t"
                "vfmacc.vf      v28, fa0, v16           \n\t"
                "vfmacc.vf      v29, fa1, v17           \n\t"
                "vfmacc.vf      v30, fa2, v18           \n\t"
                "vfmacc.vf      v31, fa3, v19           \n\t"

                "bgtz           t4, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "add            t2, %[LDC], %[DST]      \n\t"
                "vse32.v        v28, (%[DST])           \n\t"
                "add            t3, %[LDC], t2          \n\t"
                "vse32.v        v29, (t2)               \n\t"
                "add            t2, %[LDC], t3          \n\t"
                "vse32.v        v30, (t3)               \n\t"
                "vse32.v        v31, (t2)               \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
#endif
        }
    } else {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data = (int8_t *) quant_a_ptr;
            float *   dst_c  = c_ptr + ni;

            asm volatile(
                "li             t1,  8          \n\t"
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v28, v28, v28   \n\t"
                "vxor.vv        v29, v29, v29   \n\t"
                "vxor.vv        v30, v30, v30   \n\t"
                "vxor.vv        v31, v31, v31   \n\t"
                "mv             t4, %[BK]       \n\t"

                ".align 4                        \n\t"
                "BLK_LOOP%=:                     \n\t"
                // load scale A
                "flw            fa0, (%[A])     \n\t"
                "flw            fa1, 4(%[A])    \n\t"
                "flw            fa2, 8(%[A])    \n\t"
                "flw            fa3, 12(%[A])   \n\t"
                "addi           %[A], %[A], 16  \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2\n\t"
                "vle16.v        v12, (%[B])     \n\t"
                "addi           %[B], %[B], 64  \n\t"
                "vfwcvt.f.f.v   v14, v12        \n\t"

                // load zp
                "vsetvli        t0, x0, e8, mf4 \n\t"
                "vle8.v         v8, (%[B])      \n\t"
                "addi           %[B], %[B], 32  \n\t"
                "vwaddu.vx      v10, v8, x0     \n\t"

                // load a sum
                "lh             s1, (%[A])      \n\t"
                "lh             s2, 2(%[A])     \n\t"
                "lh             s3, 4(%[A])     \n\t"
                "lh             s4, 6(%[A])     \n\t"
                "addi           %[A], %[A], 8   \n\t"

                "vsetvli        t0, x0, e8, m1  \n\t"
                "vl1r.v         v0, (%[A])      \n\t"
                "addi           %[A], %[A], 128 \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])      \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512 \n\t"
                "vsrl.vi        v1, v0, 4       \n\t"
                "vnpack4.vv     v12, v0, v1, 3  \n\t"  // A low  u4
                "vupack.vv      v2, v12, v12, 2 \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v16, v16, v16   \n\t"
                "vxor.vv        v18, v16, v16   \n\t"
                "vxor.vv        v20, v16, v16   \n\t"
                "vxor.vv        v22, v16, v16   \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vmadotsu       v16, v3, v4, i4 \n\t"  // high 4
                "vmadotsu       v18, v3, v5, i4 \n\t"
                "vmadotsu       v20, v3, v6, i4 \n\t"
                "vmadotsu       v22, v3, v7, i4 \n\t"
                "vsll.vi        v16, v16, 4     \n\t"
                "vsll.vi        v18, v18, 4     \n\t"
                "vsll.vi        v20, v20, 4     \n\t"
                "vsll.vi        v22, v22, 4     \n\t"
                "vmadotu        v16, v2, v4, i4 \n\t"  // low 4
                "vmadotu        v18, v2, v5, i4 \n\t"
                "vmadotu        v20, v2, v6, i4 \n\t"
                "vmadotu        v22, v2, v7, i4 \n\t"

                "vpack.vv       v0, v16, v18, 2 \n\t"
                "vpack.vv       v2, v20, v22, 2 \n\t"
                "vpack.vv       v16, v0, v2, 3  \n\t"
                "vpack.vv       v18, v1, v3, 3  \n\t"

                "vsetvli        t0, x0, e16, m1 \n\t"
                "vwmul.vx       v0, v10, s1     \n\t"
                "vwmul.vx       v2, v10, s2     \n\t"
                "vwmul.vx       v4, v10, s3     \n\t"
                "vwmul.vx       v6, v10, s4     \n\t"

                "vsetvli        t0, x0, e32, m1 \n\t"
                "vadd.vv        v16, v16, v0    \n\t"
                "vadd.vv        v17, v17, v2    \n\t"
                "vadd.vv        v18, v18, v4    \n\t"
                "vadd.vv        v19, v19, v6    \n\t"

                "vfcvt.f.x.v    v16, v16        \n\t"
                "vfcvt.f.x.v    v17, v17        \n\t"
                "vfcvt.f.x.v    v18, v18        \n\t"
                "vfcvt.f.x.v    v19, v19        \n\t"

                // mul scale
                "vfmul.vv       v16, v16, v14   \n\t"
                "vfmul.vv       v17, v17, v14   \n\t"
                "vfmul.vv       v18, v18, v14   \n\t"
                "vfmul.vv       v19, v19, v14   \n\t"

                "addi           t4, t4, -1      \n\t"
                "vfmacc.vf      v28, fa0, v16   \n\t"
                "vfmacc.vf      v29, fa1, v17   \n\t"
                "vfmacc.vf      v30, fa2, v18   \n\t"
                "vfmacc.vf      v31, fa3, v19   \n\t"

                "bgtz           t4, BLK_LOOP%=  \n\t"

                // save
                "vsetvli        t0, x0, e32, m1 \n\t"
                "add            t2, %[LDC], %[DST]\n\t"
                "vse32.v        v28, (%[DST])   \n\t"
                "add            t3, %[LDC], t2  \n\t"
                "vse32.v        v29, (t2)       \n\t"
                "add            t2, %[LDC], t3  \n\t"
                "vse32.v        v30, (t3)       \n\t"
                "vse32.v        v31, (t2)       \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3", "s1", "s2", "s3", "s4");
        }
    }
}

void gemm_kernel_i8i4_hp_m4(size_t          blk_len,
                            const uint8_t * quant_a_ptr,
                            const uint8_t * quant_b_data,
                            const uint8_t * quant_b_zp,
                            float *         c_ptr,
                            size_t          count_m,
                            size_t          count_n,
                            size_t          k_blks,
                            size_t          ldc) {
    constexpr size_t NB_COLS                = 32;
    constexpr size_t K_SUBBLKS_PER_SUPERBLK = 8;
    constexpr size_t K_SUBBLK_LEN           = 32;

    struct block_q4_0x32_layout {
        _Float16 d[NB_COLS];
        uint8_t  qs[16 * NB_COLS];
    };

    GGML_ASSERT(blk_len == 256);
    GGML_ASSERT(count_m >= 4);

    // Contract:
    // - computes a 4-row x 32-col tile per inner invocation
    // - A is q8 HP packed in m4 layout, one logical K256 block at a time
    // - B is q4 HP packed in N32 tiles, optionally with a separate zp area
    // - tail-N is currently not handled here; the caller must provide full N32 tiles

    const size_t b_superblk_stride = sizeof(block_q4_0x32_layout) * K_SUBBLKS_PER_SUPERBLK +
                                     (quant_b_zp ? NB_COLS * K_SUBBLKS_PER_SUPERBLK * sizeof(uint8_t) : 0);
    const size_t b_tile_stride       = k_blks * b_superblk_stride;
    const size_t a_nrow_block_stride = q8_hp_blk_size(blk_len, true, true) * 4;
    const size_t a_subblk_stride     = q8_hp_blk_size(K_SUBBLK_LEN, false, false) * 4;

    if (quant_b_zp != nullptr) {
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            const size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            if (nb_real != NB_COLS) {
                break;
            }

            uint8_t * b_tile_base = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_tile_stride;
            uint8_t * a_block     = (uint8_t *) quant_a_ptr;
            float *   dst_c       = c_ptr + ni;

            // Data layout summary for the with-zp path.
            //
            // A: M4 x K256 q8 HP block
            //   - split into 8 x K32 subblocks
            //   - each K32 subblock is 136B:
            //       8B   = 4 x fp16 row scales
            //       128B = 4 x int8[32] row payloads
            //   - trailer after 8 subblocks is 72B:
            //       4 rows x fp16[8] a_sum values, indexed as [row][ksi]
            //       4 rows x fp16 scale_avg tail
            //
            // B: N32 x K256 q4 HP block with explicit zp area
            //   - each K32 subblock is 576B:
            //       64B  = fp16 scale[32]
            //       512B = packed q4 payload for 32 columns x 32 k-elements
            //   - zp is stored separately, not interleaved with the 576B payload block
            //   - one K256 superblock is laid out as:
            //       8 x (scale + qs) blocks = 4608B
            //       8 x zp[32]              =  256B
            //
            // C: 4 rows x 32 fp32 outputs
            //
            // ASM pointer convention:
            //   - t6: current A K32 subblock base
            //   - t2: current A a_sum base for this ksi
            //         row1/row2/row3 are at +16/+32/+48 bytes
            //   - s5: current B (scale + qs) K32 subblock base
            //   - s6: current B zp[32] base for this ksi
            //
            // Loop progression:
            //   - per ksi: A += 136, a_sum += 2, B_data += 576, B_zp += 32
            //   - per ki : skip the 72B A trailer and advance B to the next 4864B superblock

            const _Float16 hp_scale_16   = (_Float16) 16.0f;
            const _Float16 hp_scale_1    = (_Float16) 1.0f;
            const _Float16 hp_scale_0125 = (_Float16) 0.125f;

            // VPR grouping used below:
            // - v4-v7   : B q4 payload for N32 split as 4 x N8 groups
            // - v8/v10  : zp u8 / widened fp16
            // - v12     : B fp16 scale[32]
            // - v14-v15 : packed (Bscale * Ascale) for rows [0,1] / [2,3]
            // - v16-v19 : temporary per-row scaled B scales
            // - v28-v31 : final fp32 accumulators for rows 0..3

            asm volatile(
                "mv             t5, %[BK]                 \n\t"
                "mv             t6, %[A]                  \n\t"
                "mv             s5, %[B]                  \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v28, v28, v28             \n\t"
                "vxor.vv        v29, v29, v29             \n\t"
                "vxor.vv        v30, v30, v30             \n\t"
                "vxor.vv        v31, v31, v31             \n\t"
                "li             t4, 8                     \n\t"
                "li             t1, 4608                  \n\t"
                "addi           t2, t6, 1088              \n\t"  // 8 * 136B A K32 subblocks, a_sum trailer starts here
                "add            s6, s5, t1                \n\t"  // 8 * 576B B(scale+qs), zp area starts here

                ".align 4                                 \n\t"
                "_BLK_LPST%=:                             \n\t"
                "flh            fa1, 64(t2)               \n\t"  // a_scale_avg_row[0]
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v18, v30, v30             \n\t"
                "vxor.vv        v19, v31, v31             \n\t"
                "vxor.vv        v20, v30, v30             \n\t"
                "vxor.vv        v21, v31, v31             \n\t"
                "_KsubBLK_LPST%=:                         \n\t"
                // load first subblock scales for 4 rows
                "flh            fa0,   0(t6)              \n\t"  // ascale_fp16

                // load B fp16 scales[32]
                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vle16.v        v12, (s5)                 \n\t"

                // load Bzp[32] for the current ksi from the dedicated zp area
                "vsetvli        t0, x0, e8, mf4           \n\t"
                "vle8.v         v8, (s6)                  \n\t"

                "fmul.h         fa2, fa0, %[HP16]         \n\t"
                "vfwcvt.f.xu.v  v10, v8                   \n\t"  // uint8 -> fp16

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfmul.vf       v16, v12, fa0             \n\t"  // row0: Bscale * Ascale
                "vfmul.vf       v17, v12, fa2             \n\t"

                // load a_sum[row][ksi] from the trailer; t2 points to row0[ksi]
                "flh            ft1, 0(t2)                \n\t"
                "flh            ft2, 16(t2)               \n\t"
                "flh            ft3, 32(t2)               \n\t"
                "flh            ft4, 48(t2)               \n\t"

                "fmul.h         ft1, ft1, %[HP0125]       \n\t"
                "fmul.h         ft2, ft2, %[HP0125]       \n\t"
                "fmul.h         ft3, ft3, %[HP0125]       \n\t"
                "fmul.h         ft4, ft4, %[HP0125]       \n\t"

                // load A payload from current K32 subblock and B q4 payload from current 576B block
                "addi           t3, t6, 8                 \n\t"
                "vsetvli        t0, x0, e8, m1            \n\t"
                "vl1r.v         v0, (t3)                  \n\t"  //A
                "addi           t3, s5, 64                \n\t"
                "vl4r.v         v4, (t3)                  \n\t"  //B

                "vsetvli        t0, x0, e8, m1            \n\t"
                "vsrl.vi        v1, v0, 4                 \n\t"
                "vnpack4.vv     v12, v0, v1, 3            \n\t"
                "vpack.vv       v0, v17, v16, 3           \n\t"
                "vupack.vv      v2, v12, v12, 2           \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"  // mf2 -> mf2
                "vfmul.vv       v10, v10, v16             \n\t"  // zp * ascale * bscale; fp16*fp16

                "vsetvli        t0, x0, e16, mf2          \n\t"  // mf2 -> m1
                "vfmul.vf       v12, v10, ft1             \n\t"  // zp(1:n)* abscale * asum_m0; fp16*fp16
                "vfmul.vf       v13, v10, ft2             \n\t"  // zp(1:n)* abscale * asum_m1; fp16*fp16
                "vfmul.vf       v24, v10, ft3             \n\t"  // zp(1:n)* abscale * asum_m2; fp16*fp16
                "vfmul.vf       v25, v10, ft4             \n\t"  // zp(1:n)* abscale * asum_m3; fp16*fp16

                "vsetvli        t0, x0, e16, mf2           \n\t"
                "vfwmacc.vf     v28, fa1, v12             \n\t"  // row0/1 accum += dot * packed scale
                "vfwmacc.vf     v29, fa1, v13             \n\t"
                "vfwmacc.vf     v30, fa1, v24             \n\t"
                "vfwmacc.vf     v31, fa1, v25             \n\t"

                "vsetvli        t0, x0, e32, m1           \n\t"
                "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"  //lo4;n0n7
                "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"  //lo4;n8n15
                "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"  //lo4;n16n23
                "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"  //lo4;n24n31
                "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"  //hi4;n0n7
                "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"  //hi4;n8n15
                "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"  //hi4;n16n23
                "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"  //hi4;n24n31

                "addi           t4, t4, -1                \n\t"
                "addi           t6, t6, 8+128             \n\t"  // next A K32 subblock
                "addi           t2, t2, 2                 \n\t"  // next ksi entry in each a_sum row
                "addi           s5, s5, 64+512            \n\t"  // next B (scale + qs) K32 block
                "addi           s6, s6, 32                \n\t"  // next zp[32]
                "bgtz           t4, _KsubBLK_LPST%=       \n\t"

                "vsetvli        t0, x0, e16, m1           \n\t"
                "vpack.vv       v8, v18, v19, 1           \n\t"  // 128(16*8)->256(16*16)
                "vpack.vv       v12, v20, v21, 1          \n\t"
                "vpack.vv       v26, v8, v12, 2           \n\t"  // 256(16*16)->512(16*32)

                "vsetvli        t0, x0, e16, m1           \n\t"
                "vfwmacc.vf     v28, fa1, v26             \n\t"  // row0/1 accum += dot * packed scale
                "vfwmacc.vf     v30, fa1, v27             \n\t"

                "li             t4, 8                     \n\t"
                "addi           t5, t5, -1                \n\t"
                "addi           t6, t6, 72                \n\t"  // skip A trailer after 8 subblocks and scale_avg tail
                "mv             s5, s6                    \n\t"  // s6 already points to next B superblock base
                "addi           t2, t6, 1088              \n\t"  // 8 * 136B A K32 subblocks, a_sum trailer starts here
                "add            s6, s5, t1                \n\t"  // 8 * 576B B(scale+qs), zp area starts here
                "bgtz           t5, _BLK_LPST%=           \n\t"

                "_BLK_LPND%=:                             \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "add            t2, %[LDC], %[DST]        \n\t"
                "vse32.v        v28, (%[DST])             \n\t"
                "add            t3, %[LDC], t2            \n\t"
                "vse32.v        v29, (t2)                 \n\t"
                "add            t2, %[LDC], t3            \n\t"
                "vse32.v        v30, (t3)                 \n\t"
                "vse32.v        v31, (t2)                 \n\t"
                : [A] "+r"(a_block), [B] "+r"(b_tile_base)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16),
                  [HP1] "f"(hp_scale_1), [HP0125] "f"(hp_scale_0125)
                : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s5", "s6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                  "v8", "v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "ft1", "ft2", "ft3", "ft4",
                  "memory");
        }
        return;
    } else {
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            const size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            if (nb_real != NB_COLS) {
                break;
            }

            uint8_t * b_tile_base = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_tile_stride;
            uint8_t * a_block     = (uint8_t *) quant_a_ptr;
            float *   dst_c       = c_ptr + ni;

            // Data layout summary for the no-zp path.
            //
            // A layout is identical to the with-zp branch.
            //
            // B: N32 x K256 q4 HP block without explicit zp storage
            //   - each K32 subblock is still 576B:
            //       64B  = fp16 scale[32]
            //       512B = packed q4 payload
            //   - zp is implicit and treated as a constant value 8 in the kernel
            //   - one K256 superblock therefore contains only:
            //       8 x (scale + qs) blocks = 4608B
            //
            // C: 4 rows x 32 fp32 outputs
            //
            // ASM pointer convention:
            //   - t6: current A K32 subblock base
            //   - t2: current A a_sum base for this ksi
            //   - s5: current B (scale + qs) K32 subblock base
            //
            // Loop progression:
            //   - per ksi: A += 136, a_sum += 2, B_data += 576
            //   - per ki : skip the 72B A trailer and advance B to the next 4608B superblock

            const _Float16 hp_scale_16 = (_Float16) 16.0f;
            const _Float16 hp_scale_1  = (_Float16) 1.0f;

            // VPR grouping used below matches the with-zp path:
            // - v4-v7   : B q4 payload for N32 split as 4 x N8 groups
            // - v8/v10  : implicit zp lane / widened fp16
            // - v12     : B fp16 scale[32]
            // - v14-v15 : packed (Bscale * Ascale) for rows [0,1] / [2,3]
            // - v16-v19 : temporary per-row scaled B scales
            // - v28-v31 : final fp32 accumulators for rows 0..3

            asm volatile(
                "mv             t5, %[BK]                 \n\t"
                "mv             t6, %[A]                  \n\t"
                "mv             s5, %[B]                  \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v28, v28, v28             \n\t"
                "vxor.vv        v29, v29, v29             \n\t"
                "vxor.vv        v30, v30, v30             \n\t"
                "vxor.vv        v31, v31, v31             \n\t"
                "li             t4, 8                     \n\t"
                "addi           t2, t6, 1088              \n\t"  // 8 * 136B A K32 subblocks, a_sum trailer starts here

                ".align 4                                 \n\t"
                "_BLK_LPST%=:                             \n\t"
                "flh            fa1, 64(t2)               \n\t"  // a_scale_avg_row[0]
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v18, v30, v30             \n\t"
                "vxor.vv        v19, v31, v31             \n\t"
                "vxor.vv        v20, v30, v30             \n\t"
                "vxor.vv        v21, v31, v31             \n\t"
                "_KsubBLK_LPST%=:                         \n\t"
                // load first subblock scales for 4 rows
                "flh            fa0,   0(t6)              \n\t"  // ascale_fp16

                // load B fp16 scales[32]
                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vle16.v        v12, (s5)                 \n\t"

                "fmul.h         fa2, fa0, %[HP16]         \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfmul.vf       v16, v12, fa0             \n\t"  // row0: Bscale * Ascale
                "vfmul.vf       v17, v12, fa2             \n\t"

                // load a_sum[row][ksi] from the trailer; t2 points to row0[ksi]
                "flh            ft1, 0(t2)                \n\t"
                "flh            ft2, 16(t2)               \n\t"
                "flh            ft3, 32(t2)               \n\t"
                "flh            ft4, 48(t2)               \n\t"

                // load A payload from current K32 subblock and B q4 payload from current 576B block
                "addi           t3, t6, 8                 \n\t"
                "vsetvli        t0, x0, e8, m1            \n\t"
                "vl1r.v         v0, (t3)                  \n\t"  //A
                "addi           t3, s5, 64                \n\t"
                "vl4r.v         v4, (t3)                  \n\t"  //B

                "vsetvli        t0, x0, e8, m1            \n\t"
                "vsrl.vi        v1, v0, 4                 \n\t"
                "vnpack4.vv     v12, v0, v1, 3            \n\t"
                "vpack.vv       v0, v17, v16, 3           \n\t"
                "vupack.vv      v2, v12, v12, 2           \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"  // mf2 -> m1
                "vfmul.vf       v12, v16, ft1             \n\t"  // zp(1:n)* abscale * asum_m0; fp16*fp16
                "vfmul.vf       v13, v16, ft2             \n\t"  // zp(1:n)* abscale * asum_m1; fp16*fp16
                "vfmul.vf       v24, v16, ft3             \n\t"  // zp(1:n)* abscale * asum_m2; fp16*fp16
                "vfmul.vf       v25, v16, ft4             \n\t"  // zp(1:n)* abscale * asum_m3; fp16*fp16

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfwmacc.vf     v28, fa1, v12             \n\t"
                "vfwmacc.vf     v29, fa1, v13             \n\t"
                "vfwmacc.vf     v30, fa1, v24             \n\t"
                "vfwmacc.vf     v31, fa1, v25             \n\t"

                "vsetvli        t0, x0, e32, m1           \n\t"
                "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"  //lo4;n0n7
                "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"  //lo4;n8n15
                "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"  //lo4;n16n23
                "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"  //lo4;n24n31
                "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"  //hi4;n0n7
                "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"  //hi4;n8n15
                "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"  //hi4;n16n23
                "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"  //hi4;n24n31

                "addi           t4, t4, -1                \n\t"

                "addi           t6, t6, 8+128             \n\t"  // next A K32 subblock
                "addi           t2, t2, 2                 \n\t"  // next ksi entry in each a_sum row
                "addi           s5, s5, 64+512            \n\t"  // next B (scale + qs) K32 block
                "bgtz           t4, _KsubBLK_LPST%=       \n\t"

                "vsetvli        t0, x0, e16, m1           \n\t"  //N32in1register
                "vpack.vv       v8, v18, v19, 1           \n\t"  // 128(16*8)->256(16*16)
                "vpack.vv       v12, v20, v21, 1          \n\t"
                "vpack.vv       v26, v8, v12, 2           \n\t"  // 256(16*16)->512(16*32)

                "vsetvli        t0, x0, e16, m1           \n\t"
                "vfwmacc.vf     v28, fa1, v26             \n\t"  // row0/1 accum += dot * packed scale
                "vfwmacc.vf     v30, fa1, v27             \n\t"

                "li             t4, 8                     \n\t"
                "addi           t5, t5, -1                \n\t"
                "addi           t6, t6, 72                \n\t"  // skip A trailer after 8 subblocks and scale_avg tail
                // s5 already points to next B superblock base
                "addi           t2, t6, 1088              \n\t"  // 8 * 136B A K32 subblocks, a_sum trailer starts here
                "bgtz           t5, _BLK_LPST%=           \n\t"

                "_BLK_LPND%=:                             \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "add            t2, %[LDC], %[DST]        \n\t"
                "vse32.v        v28, (%[DST])             \n\t"
                "add            t3, %[LDC], t2            \n\t"
                "vse32.v        v29, (t2)                 \n\t"
                "add            t2, %[LDC], t3            \n\t"
                "vse32.v        v30, (t3)                 \n\t"
                "vse32.v        v31, (t2)                 \n\t"
                : [A] "+r"(a_block), [B] "+r"(b_tile_base)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16), [HP1] "f"(hp_scale_1)
                : "t0", "t2", "t3", "t4", "t5", "t6", "s5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24", "v25", "v26",
                  "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "ft1", "ft2", "ft3", "ft4", "memory");
        }
        return;
    }
}

void gemm_kernel_i8mxfp4_m1(size_t          blk_len,
                            const uint8_t * quant_a_ptr,
                            const uint8_t * quant_b_data,
                            const uint8_t * quant_b_zp,
                            float *         c_ptr,
                            size_t          count_m,
                            size_t          count_n,
                            size_t          k_blks,
                            size_t          ldc) {
    constexpr size_t NB_COLS = 32;
    constexpr size_t K_TILE  = 32;
    using blk_type           = nrow_block_mxfp4<NB_COLS>;

    GGML_ASSERT(blk_len == K_TILE);
    GGML_ASSERT(count_m == 1);
    GGML_UNUSED(quant_b_zp);

    const size_t a_blk_stride  = q8_blk_size(blk_len, true);
    const size_t b_blk_stride  = sizeof(blk_type);
    const size_t b_tile_stride = k_blks * b_blk_stride;

    if (quant_b_zp == NULL) {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            // MXFP4 no-zp: per column per k-block stride = scale_e8m0(1B) + qs(16B) + qh(4B) = 21B
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +     //
                                      n * k_blks * (blk_len / 8) +   // qh sign/high-bit mask: n×k_blks×4
                                      n * k_blks * blk_len / 2 +     // qs packed 4-bit magnitudes: n×k_blks×16
                                      n * k_blks * sizeof(uint8_t);  // scale: n×k_blks×1
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format (q8 block with per-block scale and stored sum field):
            //   || scl(fp32,4B) | asum(int16,2B) | data(int8,32B) || × k_blks
            //
            // Register map:
            //   t3 = k_blks loop counter   t4 = nblks (tail)
            //   f0 = A scale (fp32)
            //   s2 = pA (scale/asum)       s3 = pA data
            //   s4 = pB scales (u8×32)
            //   s5 = pB qh (sign/high-bit mask, 128B)
            //   s6 = pB qs (packed 4-bit magnitudes, 512B)
            //   s7 = pC
            //   v3  = fp32 accumulator (N32)
            //   v2  = B scales u8 (loaded as bytes; later widened)
            //   v0  = qh mask bytes (also used as v0.t mask after load)
            //   v1  = A int8 (K32)
            //   v8..v15 / v16..v23 = qs unpack/pack temporaries (build signed vmadot lanes)
            //   v24/v26/v28/v30    = int32 dot accumulators & packing temps

            __asm__ volatile(
                "mv           t3, %[BCK]              \n\t"  // t3 = k_blks
                "mv           t4, %[NBLKS]            \n\t"  // t4 = nblks (tail guard)

                // ---- pre-loop: init fp16 constants in e16 m1 context ----
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // v0 = int16(1)
                "vfcvt.f.x.v  v0, v0                  \n\t"  // v0  = 1.0_fp16
                "vxor.vv      v3, v16, v16            \n\t"

                // ---- pointer setup ----
                "mv           s2, %[pA]               \n\t"  // s2 = pA (scale, fp32)
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pA data (skip scale+asum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32           \n\t"  // s5 = pBh  (pB + 32B scale)
                "addi         s6, %[pB], 32+128       \n\t"  // s6 = pBs  (pB + 32 + 128 = pB+192)
                "mv           s7, %[pC]               \n\t"  // s7 = pC

                // =====================================================================
                // K-block loop: each iteration processes one N32×K32 block
                // Stride per k-block = 672B = 32(scl) + 512(Bs) + 128(Bh)
                // =====================================================================
                ".align 4                             \n\t"
                "BLK_LPST%=:                          \n\t"

                // ---- load qs (512B = 4 VRF) from s6, advance s6 by 672 ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v8, (s6)                \n\t"  // v8..v11 = qs N32K32 packed 4-bit magnitudes
                "addi         s6, s6, 128*4+128+32    \n\t"  // s6 += 672 (512+128+32)

                // ---- load B scale (32B = 32×u8) from s4, advance s4 by 672 ----
                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v2, (s4)                \n\t"  // v2 = scale_u8 × 32
                "addi         s4, s4, 32+128*4+128    \n\t"  // s4 += 672 (32+512+128)

                // ---- load qh (128B = 1 VRF) from s5, advance s5 by 672 ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v0, (s5)                \n\t"  // v0 = qh N32K32 sign/high-bit packed
                "addi         s5, s5, 128+32+128*4    \n\t"  // s5 += 672 (128+32+512)

                // ---- load A data (32B = K32 int8) from s3 ----
                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v1, (s3)                \n\t"  // v1 = A M1K32 int8
                "addi         s3, s3, 32+6            \n\t"  // s3 += 38 (data + scl + asum)

                // ---- load A scale (fp32) and asum (int16) from s2 ----
                "flw          f0, (s2)                \n\t"  // f0 = A scale (fp32)
                "addi         s2, s2, 6+32            \n\t"  // s2 += 38

                // ---- Decode packed MXFP4 payload into a vmadot-friendly signed-lane layout ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vand.vi      v12, v8, 0xF            \n\t"  //8bit(lo4) //[8*32]
                "vand.vi      v13, v9, 0xF            \n\t"
                "vand.vi      v14, v10, 0xF           \n\t"
                "vand.vi      v15, v11, 0xF           \n\t"
                "vsrl.vi      v8, v8, 4               \n\t"  //8bit(hi4)
                "vsrl.vi      v9, v9, 4               \n\t"
                "vsrl.vi      v10, v10, 4             \n\t"
                "vsrl.vi      v11, v11, 4             \n\t"

                // [4*32]*2
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v16, v12, v8, 0         \n\t"
                "vpack.vv     v18, v13, v9, 0         \n\t"
                "vpack.vv     v20, v14, v10, 0        \n\t"
                "vpack.vv     v22, v15, v11, 0        \n\t"

                "vsetvli      t0, x0, e8, m8          \n\t"
                "vrsub.vi     v16, v16, 0, v0.t       \n\t"

                // [4*32]*2 -> [8*16]
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vupack.vv    v8, v16, v17, 1         \n\t"
                "vupack.vv    v10, v18, v19, 1        \n\t"
                "vupack.vv    v12, v20, v21, 1        \n\t"
                "vupack.vv    v14, v22, v23, 1        \n\t"

                "vsetvli      t0, x0, e64, m1         \n\t"
                "vslidedown.vi  v16, v1, 2            \n\t"

                // init the accumu to 0
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v24, v16, v16           \n\t"
                "vxor.vv      v26, v16, v16           \n\t"
                "vxor.vv      v28, v16, v16           \n\t"
                "vxor.vv      v30, v16, v16           \n\t"

                // ---- int8 dot products over the decoded MXFP4 lane groups ----
                "vmadot       v24, v1, v8, i8         \n\t"  // N0..7
                "vmadot       v26, v1, v10, i8        \n\t"  // N8..15
                "vmadot       v28, v1, v12, i8        \n\t"  // N16..23
                "vmadot       v30, v1, v14, i8        \n\t"  // N24..31
                "vmadot       v24, v16, v9, i8        \n\t"  // N0..7
                "vmadot       v26, v16, v11, i8       \n\t"  // N8..15
                "vmadot       v28, v16, v13, i8       \n\t"  // N16..23
                "vmadot       v30, v16, v15, i8       \n\t"  // N24..31

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"  // v16 = N0..15
                "vpack.vv     v18, v28, v30, 2        \n\t"  // v18 = N16..31
                "vpack.vv     v24, v16, v18, 3        \n\t"  // v24 = N0..31

                "lui          t1, 0x00200             \n\t"
                "vmv.v.x      v30, t1                 \n\t"
                // b_scale e8m0 -> fp32
                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vwaddu.vx    v28, v2, x0             \n\t"
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vwadd.vx     v2, v28, x0             \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vmsle.vi     v0, v2, 1               \n\t"
                "vadd.vi      v28, v2, -1             \n\t"
                "vsll.vi      v28, v28, 23            \n\t"
                "vsll.vv      v28, v30, v2, v0.t      \n\t"

                // a_scale * b_scale;
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v24                \n\t"
                "vfmul.vf     v30, v28, f0            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v3, v30, v26            \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, BLK_LPST%=          \n\t"
                "BLK_LPND%=:                          \n\t"
                "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
                "vse32.v      v3, (%[pC])             \n\t"
                "FUNC_END%=:                          \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6", "s7", "v0", "v1",
                  "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v16", "v17", "v18", "v19",
                  "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
    }
}

void gemm_kernel_i8mxfp4_m4(size_t          blk_len,
                            const uint8_t * quant_a_ptr,
                            const uint8_t * quant_b_data,
                            const uint8_t * quant_b_zp,
                            float *         c_ptr,
                            size_t          count_m,
                            size_t          count_n,
                            size_t          k_blks,
                            size_t          ldc) {
    constexpr size_t NB_COLS = 32;
    constexpr size_t K_TILE  = 32;
    using blk_type           = nrow_block_mxfp4<NB_COLS>;

    GGML_ASSERT(blk_len == K_TILE);
    GGML_ASSERT(count_m == 4);
    GGML_UNUSED(quant_b_zp);

    const size_t a_blk_stride  = q8_blk_size(blk_len, true);
    const size_t b_blk_stride  = sizeof(blk_type);
    const size_t b_tile_stride = k_blks * b_blk_stride;

    if (quant_b_zp == NULL) {
        // MXFP4 block layout per K32/N32 tile:
        //   [scale_e8m0 x 32][qh sign/high-bit mask x 128B][qs packed 4-bit magnitudes x 512B]
        // There is no explicit zp stream; qh is combined with qs to reconstruct signed MXFP4 values.
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            uint8_t * b_data  = (uint8_t *) quant_b_data + (ni / NB_COLS) * b_tile_stride;
            uint8_t * a_data  = (uint8_t *) quant_a_ptr;
            float *   dst_c   = c_ptr + ni;
            size_t    cnt     = k_blks;

            asm volatile(
                // v4-v7 are the fp32 accumulators for rows 0..3 of the current N32 tile.
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v4, v4, v4              \n\t"
                "vxor.vv        v5, v5, v5              \n\t"
                "vxor.vv        v6, v6, v6              \n\t"
                "vxor.vv        v7, v7, v7              \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // Load the 4 A-row scales for this K32 block and build row data pointers.
                "flw            fa0, 0(%[A])            \n\t"
                "flw            fa1, 4(%[A])            \n\t"
                "flw            fa2, 8(%[A])            \n\t"
                "flw            fa3, 12(%[A])           \n\t"
                "addi           t3, %[A], 24            \n\t"
                "addi           t4, t3, 32              \n\t"
                "addi           t5, t3, 64              \n\t"
                "addi           t6, t3, 96              \n\t"
                "addi           %[A], %[A], 152         \n\t"

                // B-side pointers:
                //   t1 -> qh bitmask stream, t2 -> qs low-nibble stream.
                "addi           t1, %[B], 32            \n\t"
                "addi           t2, %[B], 160           \n\t"
                "vsetvli        t0, x0, e8, mf2         \n\t"
                "vle8.v         v2, (%[B])              \n\t"
                "addi           %[B], %[B], 672         \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v0, (t1)                \n\t"
                "vl4r.v         v8, (t2)                \n\t"

                // Decode the packed MXFP4 payload once for the whole tile and expand it
                // into a vmadot-friendly layout.
                "vand.vi        v12, v8, 0xF            \n\t"
                "vand.vi        v13, v9, 0xF            \n\t"
                "vand.vi        v14, v10, 0xF           \n\t"
                "vand.vi        v15, v11, 0xF           \n\t"
                "vsrl.vi        v8, v8, 4               \n\t"
                "vsrl.vi        v9, v9, 4               \n\t"
                "vsrl.vi        v10, v10, 4             \n\t"
                "vsrl.vi        v11, v11, 4             \n\t"

                "vpack.vv       v16, v12, v8, 0         \n\t"
                "vpack.vv       v18, v13, v9, 0         \n\t"
                "vpack.vv       v20, v14, v10, 0        \n\t"
                "vpack.vv       v22, v15, v11, 0        \n\t"

                "vsetvli        t0, x0, e8, m8          \n\t"
                "vrsub.vi       v16, v16, 0, v0.t       \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vupack.vv      v8, v16, v17, 1         \n\t"
                "vupack.vv      v10, v18, v19, 1        \n\t"
                "vupack.vv      v12, v20, v21, 1        \n\t"
                "vupack.vv      v14, v22, v23, 1        \n\t"

                "lui            t1, 0x00200             \n\t"
                "vmv.v.x        v30, t1                 \n\t"
                // b_scale e8m0 -> fp32
                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vwaddu.vx      v28, v2, x0             \n\t"
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vwadd.vx       v26, v28, x0            \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vmsle.vi       v0, v26, 1              \n\t"
                "vadd.vi        v24, v26, -1            \n\t"
                "vsll.vi        v18, v24, 23            \n\t"
                "vsll.vv        v18, v30, v26, v0.t     \n\t"

                // Row 0: dot(A0, decoded MXFP4 lane groups), accumulate in int32 and
                // then apply A/B scaling.
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v1, (t3)                \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vupack.vv      v16, v1, v2, 1          \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v24, v24, v24           \n\t"
                "vxor.vv        v26, v26, v26           \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v30, v30, v30           \n\t"
                "vmadot         v24, v16, v8, i8        \n\t"
                "vmadot         v26, v16, v10, i8       \n\t"
                "vmadot         v28, v16, v12, i8       \n\t"
                "vmadot         v30, v16, v14, i8       \n\t"
                "vmadot         v24, v17, v9, i8        \n\t"
                "vmadot         v26, v17, v11, i8       \n\t"
                "vmadot         v28, v17, v13, i8       \n\t"
                "vmadot         v30, v17, v15, i8       \n\t"
                "vpack.vv       v16, v24, v26, 2        \n\t"
                "vpack.vv       v20, v28, v30, 2        \n\t"
                "vpack.vv       v24, v16, v20, 3        \n\t"
                "vpack.vv       v26, v17, v21, 3        \n\t"
                "vfcvt.f.x.v    v24, v24                \n\t"
                "vfcvt.f.x.v    v25, v25                \n\t"
                "vfcvt.f.x.v    v26, v26                \n\t"
                "vfcvt.f.x.v    v27, v27                \n\t"
                "vfmul.vv       v24, v24, v18           \n\t"
                "vfmul.vv       v25, v25, v18           \n\t"
                "vfmul.vv       v26, v26, v18           \n\t"
                "vfmul.vv       v27, v27, v18           \n\t"
                "vfmacc.vf      v4, fa0, v24            \n\t"
                "vfmacc.vf      v5, fa1, v25            \n\t"
                "vfmacc.vf      v6, fa2, v26            \n\t"
                "vfmacc.vf      v7, fa3, v27            \n\t"

                "addi           %[BK], %[BK], -1        \n\t"
                "bgtz           %[BK], BLK_LOOP%=       \n\t"

                // Tail-aware store for the final N tile (`nb_real` may be < 32).
                "vsetvli        t0, %[NBLKS], e32, m1   \n\t"
                "add            t1, %[LDC], %[DST]      \n\t"
                "vse32.v        v4, (%[DST])            \n\t"
                "vse32.v        v5, (t1)                \n\t"
                "add            t2, t1, %[LDC]          \n\t"
                "vse32.v        v6, (t2)                \n\t"
                "add            t3, t2, %[LDC]          \n\t"
                "vse32.v        v7, (t3)                \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data), [BK] "+r"(cnt)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [NBLKS] "r"(nb_real)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s1", "s2", "s3", "s4", "v0", "v1", "v2",
                  "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                  "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                  "fa0", "fa1", "fa2", "fa3");
        }
    }
}

void gemm_kernel_i8i5_m1(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    // =========================================================================
    // i8i5: 8-bit activation × 5-bit weight (4-bit low + 1-bit high mask)
    //
    // B layout per N32K32 k-block (no-zp):
    //   [0  .. 63 ] : scale_fp16 × 32              (64B)
    //   [64 .. 191] : Bh i1-high-bit  × 32N × 32K  (128B = 1 VRF)
    //   [192.. 703] : Bs i4-low-nibble × 32N × 32K (512B = 4 VRF)
    //   Total: 704B per k-block stride
    //
    // B layout per N32K32 k-block (with-zp):
    //   [0  .. 63 ] : scale_fp16 × 32              (64B)
    //   [64 .. 95 ] : zp_uint8 × 32                (32B)
    //   [96 .. 223] : Bh i1-high-bit  × 32N × 32K  (128B = 1 VRF)
    //   [224.. 735] : Bs i4-low-nibble × 32N × 32K (512B = 4 VRF)
    //   Total: 736B per k-block stride
    //
    // Bh format per N8K32 sub-block (32B):
    //   K rows × N cols × 1bit packed as bytes (8 cols per byte, K groups of 4B)
    //   Byte k gives 8 mask bits for columns N7..N0 at k-th K-element.
    //
    // Computation:
    //   B5bit_signed = (Bs | (Bh << 4)) - zp
    //   dot(A, B5) = dot(A, Bs_u4) + 16*dot(A, Bh_u1) - zp*asum
    //   No-zp: implicit zp = 16 (unsigned [0..31] centered at 16)
    //   With-zp: explicit zp from data
    //
    // =========================================================================

    if (quant_b_zp == NULL) {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            // i8i5 no-zp: per column per k-block stride = fp16(2B) + i4(16B) + i1(4B) = 22B
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * (blk_len / 8) +    // Bh i1 mask: n×k_blks×4
                                      n * k_blks * blk_len / 2 +      // Bs i4 data: n×k_blks×16
                                      n * k_blks * sizeof(_Float16);  // scale: n×k_blks×2
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format (same as i8i4):
            //   || scl(fp32,4B) | asum(int16,2B) | data(int8,32B) || × k_blks
            //
            // Register map:
            //   t3 = k_blks loop counter   t4 = nblks (tail)
            //   t2 = A asum (int16) << 4   f0 = A scale (fp32)
            //   s2 = pA (scale/asum)       s3 = pA data
            //   s4 = pB scales (fp16×32)
            //   s5 = pB Bh (i1 mask, 128B)
            //   s6 = pB Bs (i4 packed, 512B)
            //   s7 = pC
            //   v3  = fp32 accumulator (N32)
            //   v2  = B scales fp16 (loaded as bytes; later widened)
            //   v0  = Bh mask bytes (also used as v0.t mask after load)
            //   v1  = A int8 (K32)
            //   v8..v15 / v16..v23 = Bs unpack/pack temporaries (build b5bit bytes)
            //   v24/v26/v28/v30    = int32 dot accumulators & packing temps

            __asm__ volatile(
                "mv           t3, %[BCK]              \n\t"  // t3 = k_blks
                "mv           t4, %[NBLKS]            \n\t"  // t4 = nblks (tail guard)

                // ---- pre-loop: init fp16 constants in e16 m1 context ----
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // v0 = int16(1)
                "vfcvt.f.x.v  v0, v0                  \n\t"  // v0  = 1.0_fp16
                "vxor.vv      v3, v16, v16            \n\t"

                // ---- pointer setup ----
                "mv           s2, %[pA]               \n\t"  // s2 = pA (scale, fp32)
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pA data (skip scale+asum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBh  (pB + 64B scale)
                "addi         s6, %[pB], 32*2+128     \n\t"  // s6 = pBs  (pB + 64 + 128 = pB+192)
                "mv           s7, %[pC]               \n\t"  // s7 = pC

                // =====================================================================
                // K-block loop: each iteration processes one N32×K32 block
                // Stride per k-block = 704B = 64(scl) + 512(Bs) + 128(Bh)
                // =====================================================================
                ".align 4                             \n\t"
                "BLK_LPST%=:                          \n\t"

                // ---- load Bs (512B = 4 VRF) from s6, advance s6 by 704 ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v8, (s6)                \n\t"  // v8..v11 = Bs N32K32 i4
                "addi         s6, s6, 128*4+128+64    \n\t"  // s6 += 704 (512+128+64)

                // ---- load B scale (64B = 32×fp16) from s4, advance s4 by 704 ----
                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v2, (s4)                \n\t"  // v2 = scale_fp16 × 32
                "addi         s4, s4, 64+128*4+128    \n\t"  // s4 += 704 (64+512+128)

                // ---- load Bh (128B = 1 VRF) from s5, advance s5 by 704 ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v0, (s5)                \n\t"  // v0 = Bh N32K32 1-bit packed
                "addi         s5, s5, 128+64+128*4    \n\t"  // s5 += 704 (128+64+512)

                // ---- load A data (32B = K32 int8) from s3 ----
                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v1, (s3)                \n\t"  // v1 = A M1K32 int8
                "addi         s3, s3, 32+6            \n\t"  // s3 += 38 (data + scl + asum)

                // ---- load A scale (fp32) and asum (int16) from s2 ----
                "flw          f0, (s2)                \n\t"  // f0 = A scale (fp32)
                "lh           t2, 4(s2)               \n\t"  // t2 = A asum (int16)
                "addi         s2, s2, 6+32            \n\t"  // s2 += 38

                //// ---- A nibble unpacking ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vand.vi      v12, v8, 0xF            \n\t"  //8bit(lo4) //[8*32]
                "vand.vi      v13, v9, 0xF            \n\t"
                "vand.vi      v14, v10, 0xF           \n\t"
                "vand.vi      v15, v11, 0xF           \n\t"
                "vsrl.vi      v8, v8, 4               \n\t"  //8bit(hi4)
                "vsrl.vi      v9, v9, 4               \n\t"
                "vsrl.vi      v10, v10, 4             \n\t"
                "vsrl.vi      v11, v11, 4             \n\t"

                "slli         t2, t2, 4               \n\t"  // a_sum * 16;
                // [4*32]*2
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v16, v12, v8, 0         \n\t"
                "vpack.vv     v18, v13, v9, 0         \n\t"
                "vpack.vv     v20, v14, v10, 0        \n\t"
                "vpack.vv     v22, v15, v11, 0        \n\t"

                "li           t1, 16                  \n\t"
                "vsetvli      t0, x0, e8, m8          \n\t"
                "vadd.vx      v16, v16, t1, v0.t      \n\t"

                // [4*32]*2 -> [8*16]
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vupack.vv    v8, v16, v17, 1         \n\t"
                "vupack.vv    v10, v18, v19, 1        \n\t"
                "vupack.vv    v12, v20, v21, 1        \n\t"
                "vupack.vv    v14, v22, v23, 1        \n\t"

                "vsetvli      t0, x0, e64, m1         \n\t"
                "vslidedown.vi  v16, v1, 2            \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v24, v16, v16   \n\t"
                "vxor.vv        v26, v16, v16   \n\t"
                "vxor.vv        v28, v16, v16   \n\t"
                "vxor.vv        v30, v16, v16   \n\t"

                // ---- i8 main dot products ----
                // vmadot: A × unsigned Bh × 16 → fp16 accumulate
                "vmadot       v24, v1, v8, i8         \n\t"  // N0..7
                "vmadot       v26, v1, v10, i8        \n\t"  // N8..15
                "vmadot       v28, v1, v12, i8        \n\t"  // N16..23
                "vmadot       v30, v1, v14, i8        \n\t"  // N24..31
                //// vmadot: A × unsigned Bh × 1 → fp16 accumulate
                "vmadot       v24, v16, v9, i8        \n\t"  // N0..7
                "vmadot       v26, v16, v11, i8       \n\t"  // N8..15
                "vmadot       v28, v16, v13, i8       \n\t"  // N16..23
                "vmadot       v30, v16, v15, i8       \n\t"  // N24..31

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"  // v16 = N0..15
                "vpack.vv     v18, v28, v30, 2        \n\t"  // v18 = N16..31
                "vpack.vv     v24, v16, v18, 3        \n\t"  // v24 = N0..31

                "vadd.vx      v24, v24, t2            \n\t"
                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v28, v2                 \n\t"

                // a_scale * b_scale;
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v24                \n\t"
                "vfmul.vf     v30, v28, f0            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v3, v30, v26            \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, BLK_LPST%=          \n\t"
                "BLK_LPND%=:                          \n\t"
                "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
                "vse32.v      v3, (%[pC])             \n\t"
                "FUNC_END%=:                          \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6", "s7", "v0", "v1",
                  "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v16", "v17", "v18", "v19",
                  "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
    } else {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
            // i8i5 with-zp: per column per k-block stride = fp16(2B)+zp(1B)+i4(16B)+i1(4B)=23B
            uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                      n * k_blks * blk_len / 2 +      // Bs i4: n×k_blks×16
                                      n * k_blks * (blk_len / 8) +    // Bh i1: n×k_blks×4
                                      n * k_blks * sizeof(uint8_t) +  // zp: n×k_blks×1
                                      n * k_blks * sizeof(_Float16);  // scale: n×k_blks×2
            float * CPtr = c_ptr + n;
            size_t  cnt  = k_blks;

            // A format (same as i8i4):
            //   || scl(fp32,4B) | asum(int16,2B) | data(int8,32B) || × k_blks
            //
            // Register map:
            //   t3 = k_blks loop counter   t4 = nblks (tail)
            //   t2 = A asum (int16) << 4   f0 = A scale (fp32)
            //   s2 = pA (scale/asum)       s3 = pA data
            //   s4 = pB scales (fp16×32); 每个 k-block 先 +64 指向 zp，再 +672 到下一个 block
            //   s5 = pB Bh (i1 mask, 128B) (offset +96)
            //   s6 = pB Bs (i4 packed, 512B) (offset +224)
            //   s7 = pC
            //   v3  = fp32 accumulator (N32)
            //   v2  = B scales fp16 (loaded as bytes; later widened)
            //   v0  = Bh mask bytes (also used as v0.t mask after load)
            //   v1  = A int8 (K32) / later reused to hold Bzp bytes
            //   v8..v15 / v16..v23 = Bs unpack/pack temporaries (build b5bit bytes)
            //   v24/v26/v28/v30    = int32 dot accumulators & packing temps

            __asm__ volatile(
                "mv           t3, %[BCK]              \n\t"  // t3 = k_blks
                "mv           t4, %[NBLKS]            \n\t"  // t4 = nblks (tail guard)

                // ---- pre-loop: init fp16 constants in e16 m1 context ----
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t"  // v0 = int16(1)
                "vfcvt.f.x.v  v0, v0                  \n\t"  // v0  = 1.0_fp16
                "vxor.vv      v3, v16, v16            \n\t"

                // ---- pointer setup ----
                "mv           s2, %[pA]               \n\t"  // s2 = pA (scale, fp32)
                "addi         s3, %[pA], 4+2          \n\t"  // s3 = pA data (skip scale+asum)
                "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t"  // s5 = pBh  (pB + 64B scale + 32B zp = pB+96)
                "addi         s6, %[pB], 32*3+128     \n\t"  // s6 = pBs  (pB + 96 + 128 = pB+224)
                "mv           s7, %[pC]               \n\t"  // s7 = pC

                // =====================================================================
                // K-block loop: each iteration processes one N32×K32 block
                // Stride per k-block = 736B = 64(scale) + 32(zp) + 128(Bh) + 512(Bs)
                // =====================================================================
                ".align 4                             \n\t"
                "BLK_LPST%=:                          \n\t"

                // ---- load Bs (512B = 4 VRF) from s6, advance s6 by 736 ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v8, (s6)                \n\t"  // v8..v11 = Bs N32K32 i4
                "addi         s6, s6, 128*4+128+96    \n\t"  // s6 += 736 (512+128+96)

                // ---- load B scale (64B = 32×fp16) from s4; then s4 points to zp[32] ----
                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v2, (s4)                \n\t"  // v2 = scale_fp16 × 32
                "addi         s4, s4, 64              \n\t"  // s4 += 64 (now points to zp)

                // ---- load Bh (128B = 1 VRF) from s5, advance s5 by 736 ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v0, (s5)                \n\t"  // v0 = Bh N32K32 1-bit packed
                "addi         s5, s5, 128+96+128*4    \n\t"  // s5 += 736 (128+96+512)

                // ---- load A data (32B = K32 int8) from s3 ----
                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v1, (s3)                \n\t"  // v1 = A M1K32 int8
                "addi         s3, s3, 32+6            \n\t"  // s3 += 38 (data + scl + asum)

                // ---- load A scale (fp32) and asum (int16) from s2 ----
                "flw          f0, (s2)                \n\t"  // f0 = A scale (fp32)
                "lh           t2, 4(s2)               \n\t"  // t2 = A asum (int16)
                "addi         s2, s2, 6+32            \n\t"  // s2 += 38

                //// ---- A nibble unpacking ----
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vand.vi      v12, v8, 0xF            \n\t"  //8bit(lo4) //[8*32]
                "vand.vi      v13, v9, 0xF            \n\t"
                "vand.vi      v14, v10, 0xF           \n\t"
                "vand.vi      v15, v11, 0xF           \n\t"
                "vsrl.vi      v8, v8, 4               \n\t"  //8bit(hi4)
                "vsrl.vi      v9, v9, 4               \n\t"
                "vsrl.vi      v10, v10, 4             \n\t"
                "vsrl.vi      v11, v11, 4             \n\t"

                // [4*32]*2
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v16, v12, v8, 0         \n\t"
                "vpack.vv     v18, v13, v9, 0         \n\t"
                "vpack.vv     v20, v14, v10, 0        \n\t"
                "vpack.vv     v22, v15, v11, 0        \n\t"

                "li           t1, 16                  \n\t"
                "vsetvli      t0, x0, e8, m8          \n\t"
                "vadd.vx      v16, v16, t1, v0.t      \n\t"

                // [4*32]*2 -> [8*16]
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vupack.vv    v8, v16, v17, 1         \n\t"
                "vupack.vv    v10, v18, v19, 1        \n\t"
                "vupack.vv    v12, v20, v21, 1        \n\t"
                "vupack.vv    v14, v22, v23, 1        \n\t"

                "vsetvli      t0, x0, e64, m1         \n\t"
                "vslidedown.vi  v16, v1, 2            \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v24, v16, v16           \n\t"
                "vxor.vv      v26, v16, v16           \n\t"
                "vxor.vv      v28, v16, v16           \n\t"
                "vxor.vv      v30, v16, v16           \n\t"

                // ---- i8 main dot products ----
                // vmadot: A × unsigned Bh × 16 → fp16 accumulate
                "vmadot       v24, v1, v8, i8         \n\t"  // N0..7
                "vmadot       v26, v1, v10, i8        \n\t"  // N8..15
                "vmadot       v28, v1, v12, i8        \n\t"  // N16..23
                "vmadot       v30, v1, v14, i8        \n\t"  // N24..31
                // vmadot: A × unsigned Bh × 1 → fp16 accumulate
                "vmadot       v24, v16, v9, i8        \n\t"  // N0..7
                "vmadot       v26, v16, v11, i8       \n\t"  // N8..15
                "vmadot       v28, v16, v13, i8       \n\t"  // N16..23
                "vmadot       v30, v16, v15, i8       \n\t"  // N24..31

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v1, (s4)                \n\t"  // Bzp
                "addi         s4, s4, 32+128*4+128    \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"  // v16 = N0..15
                "vpack.vv     v18, v28, v30, 2        \n\t"  // v18 = N16..31
                "vpack.vv     v24, v16, v18, 3        \n\t"  // v24 = N0..31

                "vwaddu.vx    v28, v1, x0             \n\t"  // uint8 -> uint16

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vwmul.vx     v30, v28, t2            \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v28, v2                 \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v24, v24, v30           \n\t"

                // a_scale * b_scale;
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfmul.vf     v30, v28, f0            \n\t"
                "vfcvt.f.x.v  v26, v24                \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v3, v30, v26            \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, BLK_LPST%=          \n\t"
                "BLK_LPND%=:                          \n\t"
                "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
                "vse32.v      v3, (%[pC])             \n\t"
                "FUNC_END%=:                          \n\t"
                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6", "s7", "v0", "v1",
                  "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v16", "v17", "v18", "v19",
                  "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
    }
}

void gemm_kernel_i8i5_m4(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    constexpr size_t NB_COLS = 32;

    GGML_UNUSED(count_m);
    GGML_UNUSED(blk_len);

    // This kernel computes a 4x32 output tile. For each K32 block we decode the
    // packed Q5 weights once and reuse the decoded vectors across the 4 A rows.
    constexpr size_t B_Q50_BLK_STRIDE = sizeof(nrow_block_q5_0<NB_COLS>);
    constexpr size_t B_Q51_BLK_STRIDE = sizeof(nrow_block_q5_1<NB_COLS>);

    if (quant_b_zp) {
        // Q5_1 block layout per K32/N32 tile:
        //   [scale_fp16 x 32][zp_u8 x 32][qh high-bit mask x 128B][qs low nibbles x 512B]
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            uint8_t * b_data  = (uint8_t *) quant_b_data + (ni / NB_COLS) * k_blks * B_Q51_BLK_STRIDE;
            uint8_t * a_data  = (uint8_t *) quant_a_ptr;
            float *   dst_c   = c_ptr + ni;
            size_t    cnt     = k_blks;

            asm volatile(
                // v4-v7 are the fp32 accumulators for rows 0..3 of the current N32 tile.
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v4, v4, v4              \n\t"
                "vxor.vv        v5, v5, v5              \n\t"
                "vxor.vv        v6, v6, v6              \n\t"
                "vxor.vv        v7, v7, v7              \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // Load the 4 A-row scales/sums for this K32 block and build row data pointers.
                "flw            fa0, 0(%[A])            \n\t"
                "flw            fa1, 4(%[A])            \n\t"
                "flw            fa2, 8(%[A])            \n\t"
                "flw            fa3, 12(%[A])           \n\t"
                "lh             s1, 16(%[A])            \n\t"
                "lh             s2, 18(%[A])            \n\t"
                "lh             s3, 20(%[A])            \n\t"
                "lh             s4, 22(%[A])            \n\t"
                "addi           t3, %[A], 24            \n\t"
                "addi           t4, t3, 32              \n\t"
                "addi           t5, t3, 64              \n\t"
                "addi           t6, t3, 96              \n\t"
                "addi           %[A], %[A], 152         \n\t"

                // B-side pointers:
                //   t1 -> zp stream, t2 -> qh bitmask stream, s5 -> qs low-nibble stream.
                "addi           t1, %[B], 64            \n\t"
                "addi           t2, %[B], 96            \n\t"
                "addi           s5, %[B], 224           \n\t"
                "vsetvli        t0, x0, e8, mf2         \n\t"
                "vle8.v         v2, (%[B])              \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v0, (t2)                \n\t"
                "vl4r.v         v8, (s5)                \n\t"
                "addi           %[B], %[B], 736         \n\t"

                // Decode Q5 payload once for the whole tile:
                //   1) split `qs` low/high nibbles,
                //   2) repack into bytes,
                //   3) use the `qh` mask to inject bit4 (+16) where needed,
                //   4) expand into the vmadot-friendly layout reused by all 4 rows.
                "vand.vi        v12, v8, 0xF            \n\t"
                "vand.vi        v13, v9, 0xF            \n\t"
                "vand.vi        v14, v10, 0xF           \n\t"
                "vand.vi        v15, v11, 0xF           \n\t"
                "vsrl.vi        v8, v8, 4               \n\t"
                "vsrl.vi        v9, v9, 4               \n\t"
                "vsrl.vi        v10, v10, 4             \n\t"
                "vsrl.vi        v11, v11, 4             \n\t"

                "vpack.vv       v16, v12, v8, 0         \n\t"
                "vpack.vv       v18, v13, v9, 0         \n\t"
                "li             t2, 16                  \n\t"
                "vpack.vv       v20, v14, v10, 0        \n\t"
                "vpack.vv       v22, v15, v11, 0        \n\t"

                "vsetvli        t0, x0, e8, m8          \n\t"
                "vadd.vx        v16, v16, t2, v0.t      \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vupack.vv      v8, v16, v17, 1         \n\t"
                "vupack.vv      v10, v18, v19, 1        \n\t"
                "vupack.vv      v12, v20, v21, 1        \n\t"
                "vupack.vv      v14, v22, v23, 1        \n\t"

                // Convert per-column fp16 scales once; the same scale vector is shared by all 4 rows.
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v   v18, v2                 \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v3, (t1)                \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"

                // Row 0: dot(A0, decoded_q5) + a_sum0 * zp, then scale by A/B scales.
                // The widen/mul correction sequence intentionally matches the proven m1 Q5_1 path.
                "vle8.v         v1, (t3)                \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vupack.vv      v16, v1, v2, 1          \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v24, v24, v24           \n\t"
                "vxor.vv        v26, v26, v26           \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v30, v30, v30           \n\t"
                "vmadot         v24, v16, v8, i8        \n\t"
                "vmadot         v26, v16, v10, i8       \n\t"
                "vmadot         v28, v16, v12, i8       \n\t"
                "vmadot         v30, v16, v14, i8       \n\t"
                "vmadot         v24, v17, v9, i8        \n\t"
                "vmadot         v26, v17, v11, i8       \n\t"
                "vmadot         v28, v17, v13, i8       \n\t"
                "vmadot         v30, v17, v15, i8       \n\t"
                "vpack.vv       v16, v24, v26, 2        \n\t"
                "vpack.vv       v20, v28, v30, 2        \n\t"
                "vpack.vv       v24, v16, v20, 3        \n\t"
                "vpack.vv       v26, v17, v21, 3        \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vwaddu.vx      v28, v3, x0             \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vwmul.vx       v12, v28, s1            \n\t"
                "vwmul.vx       v14, v28, s2            \n\t"
                "vwmul.vx       v20, v28, s3            \n\t"
                "vwmul.vx       v22, v28, s4            \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vadd.vv        v24, v24, v12           \n\t"
                "vadd.vv        v25, v25, v14           \n\t"
                "vadd.vv        v26, v26, v20           \n\t"
                "vadd.vv        v27, v27, v22           \n\t"
                "vfcvt.f.x.v    v12, v24                \n\t"
                "vfcvt.f.x.v    v14, v25                \n\t"
                "vfcvt.f.x.v    v20, v26                \n\t"
                "vfcvt.f.x.v    v22, v27                \n\t"
                "vfmul.vv       v12, v12, v18           \n\t"
                "vfmul.vv       v14, v14, v18           \n\t"
                "vfmul.vv       v20, v20, v18           \n\t"
                "vfmul.vv       v22, v22, v18           \n\t"
                "vfmacc.vf      v4, fa0, v12            \n\t"
                "vfmacc.vf      v5, fa1, v14            \n\t"
                "vfmacc.vf      v6, fa2, v20            \n\t"
                "vfmacc.vf      v7, fa3, v22            \n\t"

                "addi           %[BK], %[BK], -1        \n\t"
                "bgtz           %[BK], BLK_LOOP%=       \n\t"

                // Tail-aware store for the final N tile (`nb_real` may be < 32).
                "vsetvli        t0, %[NBLKS], e32, m1   \n\t"
                "add            t1, %[LDC], %[DST]      \n\t"
                "vse32.v        v4, (%[DST])            \n\t"
                "vse32.v        v5, (t1)                \n\t"
                "add            t2, t1, %[LDC]          \n\t"
                "vse32.v        v6, (t2)                \n\t"
                "add            t3, t2, %[LDC]          \n\t"
                "vse32.v        v7, (t3)                \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data), [BK] "+r"(cnt)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [NBLKS] "r"(nb_real)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s1", "s2", "s3", "s4", "s5", "v0", "v1",
                  "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
                  "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
                  "v31", "fa0", "fa1", "fa2", "fa3");
        }
    } else {
        // Q5_0 block layout per K32/N32 tile:
        //   [scale_fp16 x 32][qh high-bit mask x 128B][qs low nibbles x 512B]
        // There is no explicit zp stream; the implicit midpoint correction is +16.
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            uint8_t * b_data  = (uint8_t *) quant_b_data + (ni / NB_COLS) * k_blks * B_Q50_BLK_STRIDE;
            uint8_t * a_data  = (uint8_t *) quant_a_ptr;
            float *   dst_c   = c_ptr + ni;
            size_t    cnt     = k_blks;

            asm volatile(
                // v4-v7 are the fp32 accumulators for rows 0..3 of the current N32 tile.
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v4, v4, v4              \n\t"
                "vxor.vv        v5, v5, v5              \n\t"
                "vxor.vv        v6, v6, v6              \n\t"
                "vxor.vv        v7, v7, v7              \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // Load the 4 A-row scales/sums for this K32 block and build row data pointers.
                "flw            fa0, 0(%[A])            \n\t"
                "flw            fa1, 4(%[A])            \n\t"
                "flw            fa2, 8(%[A])            \n\t"
                "flw            fa3, 12(%[A])           \n\t"
                "lh             s1, 16(%[A])            \n\t"
                "lh             s2, 18(%[A])            \n\t"
                "lh             s3, 20(%[A])            \n\t"
                "lh             s4, 22(%[A])            \n\t"
                "addi           t3, %[A], 24            \n\t"
                "addi           t4, t3, 32              \n\t"
                "addi           t5, t3, 64              \n\t"
                "addi           t6, t3, 96              \n\t"
                "addi           %[A], %[A], 152         \n\t"

                // B-side pointers:
                //   t1 -> qh bitmask stream, t2 -> qs low-nibble stream.
                "addi           t1, %[B], 64            \n\t"
                "addi           t2, %[B], 192           \n\t"
                "vsetvli        t0, x0, e8, mf2         \n\t"
                "vle8.v         v2, (%[B])              \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v0, (t1)                \n\t"
                "vl4r.v         v8, (t2)                \n\t"
                "addi           %[B], %[B], 704         \n\t"

                // Decode Q5 payload once for the whole tile and expand it into the vmadot layout.
                "vand.vi        v12, v8, 0xF            \n\t"
                "vand.vi        v13, v9, 0xF            \n\t"
                "vand.vi        v14, v10, 0xF           \n\t"
                "vand.vi        v15, v11, 0xF           \n\t"
                "vsrl.vi        v8, v8, 4               \n\t"
                "vsrl.vi        v9, v9, 4               \n\t"
                "vsrl.vi        v10, v10, 4             \n\t"
                "vsrl.vi        v11, v11, 4             \n\t"

                "vpack.vv       v16, v12, v8, 0         \n\t"
                "vpack.vv       v18, v13, v9, 0         \n\t"
                "li             t2, 16                  \n\t"
                "vpack.vv       v20, v14, v10, 0        \n\t"
                "vpack.vv       v22, v15, v11, 0        \n\t"

                "vsetvli        t0, x0, e8, m8          \n\t"
                "vadd.vx        v16, v16, t2, v0.t      \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vupack.vv      v8, v16, v17, 1         \n\t"
                "vupack.vv      v10, v18, v19, 1        \n\t"
                "vupack.vv      v12, v20, v21, 1        \n\t"
                "vupack.vv      v14, v22, v23, 1        \n\t"

                // Convert per-column fp16 scales once; the same scale vector is shared by all 4 rows.
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v   v18, v2                 \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"

                // Row 0: dot(A0, decoded_q5) + a_sum0 * 16 (implicit Q5_0 midpoint correction).
                "vle8.v         v1, (t3)                \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vupack.vv      v16, v1, v2, 1          \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v24, v24, v24           \n\t"
                "vxor.vv        v26, v26, v26           \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v30, v30, v30           \n\t"
                "vmadot         v24, v16, v8, i8        \n\t"
                "vmadot         v26, v16, v10, i8       \n\t"
                "vmadot         v28, v16, v12, i8       \n\t"
                "vmadot         v30, v16, v14, i8       \n\t"
                "vmadot         v24, v17, v9, i8        \n\t"
                "vmadot         v26, v17, v11, i8       \n\t"
                "vmadot         v28, v17, v13, i8       \n\t"
                "vmadot         v30, v17, v15, i8       \n\t"
                "vpack.vv       v16, v24, v26, 2        \n\t"
                "slli           s1, s1, 4               \n\t"
                "vpack.vv       v20, v28, v30, 2        \n\t"
                "slli           s2, s2, 4               \n\t"
                "vpack.vv       v24, v16, v20, 3        \n\t"
                "slli           s3, s3, 4               \n\t"
                "vpack.vv       v26, v17, v21, 3        \n\t"
                "slli           s4, s4, 4               \n\t"
                "vadd.vx        v24, v24, s1            \n\t"
                "vadd.vx        v25, v25, s2            \n\t"
                "vadd.vx        v26, v26, s3            \n\t"
                "vadd.vx        v27, v27, s4            \n\t"
                "vfcvt.f.x.v    v24, v24                \n\t"
                "vfcvt.f.x.v    v25, v25                \n\t"
                "vfcvt.f.x.v    v26, v26                \n\t"
                "vfcvt.f.x.v    v27, v27                \n\t"
                "vfmul.vv       v24, v24, v18           \n\t"
                "vfmul.vv       v25, v25, v18           \n\t"
                "vfmul.vv       v26, v26, v18           \n\t"
                "vfmul.vv       v27, v27, v18           \n\t"
                "vfmacc.vf      v4, fa0, v24            \n\t"
                "vfmacc.vf      v5, fa1, v25            \n\t"
                "vfmacc.vf      v6, fa2, v26            \n\t"
                "vfmacc.vf      v7, fa3, v27            \n\t"

                "addi           %[BK], %[BK], -1        \n\t"
                "bgtz           %[BK], BLK_LOOP%=       \n\t"

                // Tail-aware store for the final N tile (`nb_real` may be < 32).
                "vsetvli        t0, %[NBLKS], e32, m1   \n\t"
                "add            t1, %[LDC], %[DST]      \n\t"
                "vse32.v        v4, (%[DST])            \n\t"
                "vse32.v        v5, (t1)                \n\t"
                "add            t2, t1, %[LDC]          \n\t"
                "vse32.v        v6, (t2)                \n\t"
                "add            t3, t2, %[LDC]          \n\t"
                "vse32.v        v7, (t3)                \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data), [BK] "+r"(cnt)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [NBLKS] "r"(nb_real)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s1", "s2", "s3", "s4", "v0", "v1", "v2",
                  "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                  "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                  "fa0", "fa1", "fa2", "fa3");
        }
    }
}

void gemm_kernel_i8i8_m1(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    for (size_t n = 0; n < count_n; n += 32) {
        size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
        uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                  n * k_blks * blk_len +          // b data
                                  n * k_blks * sizeof(_Float16);  // scale
        float * CPtr = c_ptr + n;
        size_t  cnt  = k_blks;

        // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
        // A M1K32 int8    256bit
        // Ascale fp32 * 1  32bit
        // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
        // || Element                          || Element                          || ...
        // B format
        // B N8K32 int4    2048bit
        //   4VRF, N32K32, 8192bit
        // Bscale fp16 * N32 512bit;
        // || scl*32..(fp16) | blk0 blk1 ... blk31 || scl*32..(fp16) | blk0 blk1 ... blk31 || ...
        // || Element                              || Element                              || ...

        //bias always be nullptr
        __asm__ volatile(

            // t3 = k/32
            "mv           t3, %[BCK]              \n\t"
            "mv           t4, %[NBLKS]            \n\t"
            "mv           s2, %[pA]               \n\t"  // s2 = pASCL
            "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
            "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
            "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBdata;
            "mv           s6, %[pC]               \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v2, v0, v0              \n\t"  // clear acc

            // ordinary vmadot: vle*6 flw*1 vecIns*64 vmadot*8
            ".align 4                             \n\t"
            "_K_LPST%=:                           \n\t"

            "vsetvli      t0, x0, e8, m1          \n\t"
            "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
            "addi         s5, s5, 128*4           \n\t"
            "vl4r.v       v8, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
            "addi         s5, s5, 128*4+64        \n\t"

            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s4)                \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
            "addi         s4, s4, 64+128*8        \n\t"

            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
            "addi         s3, s3, 32+6            \n\t"

            "flw          f0, (s2)                \n\t"  // A Scale fp32
            "addi         s2, s2, 6+32            \n\t"  // AScale + Asum(FP32+i16)

            "vsetvli      t0, zero, e32, m1       \n\t"
            "vupack.vv    v24, v4, v5, 1          \n\t"
            "vupack.vv    v26, v6, v7, 1          \n\t"
            "vupack.vv    v28, v8, v9, 1          \n\t"
            "vupack.vv    v30, v10, v11, 1        \n\t"

            "vslidedown.vi  v4, v3, 4             \n\t"

            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v3, v24, i8         \n\t"  // M0 N0 - N7 INT32(256bit)
            "vmadot       v18, v3, v26, i8         \n\t"  // M0 N8 - N15
            "vmadot       v20, v3, v28, i8         \n\t"  // M0 N16 - N23
            "vmadot       v22, v3, v30, i8         \n\t"  // M0 N24 - N31

            "vmadot       v16, v4, v25, i8         \n\t"
            "vmadot       v18, v4, v27, i8         \n\t"
            "vmadot       v20, v4, v29, i8         \n\t"
            "vmadot       v22, v4, v31, i8         \n\t"

            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"

            // b_scale fp16 -> fp32
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vfwcvt.f.f.v v24, v0                 \n\t"
            // mac result i32 -> fp32
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v  v26, v16                \n\t"
            // a_scale * b_scale;
            "vfmul.vf     v1, v24, f0             \n\t"
            // static_cast<float>(qsum) * a_scale * b_scale;
            "vfmacc.vv    v2, v1, v26             \n\t"

            "addi         t3, t3, -1              \n\t"
            "bgtz         t3, _K_LPST%=           \n\t"
            "_K_LPND%=:                           \n\t"

            //-----------------------------------------
            // STORE Equal 32N-------------------------
            "_ST32%=:                             \n\t"
            "vsetvli      t0, t4, e32, m1         \n\t"
            "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

            "_FUNC_END%=:                         \n\t"

            :
            : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
            : "cc", "t0", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
    }
}

void gemm_kernel_i8i8_m4(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    int64_t b_data_stride = k_blks * sizeof(ggml_fp16_t) + k_blks * blk_len;
    for (size_t ni = 0; ni < count_n; ni += 32) {
        uint8_t * b_data = (uint8_t *) quant_b_data + ni * b_data_stride;
        int8_t *  a_data = (int8_t *) quant_a_ptr;
        float *   dst_c  = c_ptr + ni;

        asm volatile(
            "vsetvli        t0, x0, e32, m1       \n\t"
            "vxor.vv        v28, v28, v28         \n\t"
            "vxor.vv        v29, v29, v29         \n\t"
            "vxor.vv        v30, v30, v30         \n\t"
            "vxor.vv        v31, v31, v31         \n\t"

            ".align 4                             \n\t"
            "BLK_LOOP%=:                          \n\t"
            // load scale A
            "flw            fa0, (%[A])           \n\t"
            "flw            fa1, 4(%[A])          \n\t"
            "flw            fa2, 8(%[A])          \n\t"
            "flw            fa3, 12(%[A])         \n\t"
            "addi           %[A], %[A], 16+8      \n\t"  // Ascl+Asum; FP32*4+i16*4

            // load scale B
            "vsetvli        t0, x0, e16, mf2      \n\t"
            "vle16.v        v12, (%[B])           \n\t"
            "addi           %[B], %[B], 64        \n\t"
            "vfwcvt.f.f.v   v14, v12              \n\t"

            "vsetvli        t0, x0, e8, m1        \n\t"
            "vl1r.v         v0, (%[A])            \n\t"
            "addi           %[A], %[A], 128       \n\t"  // 4*32@i8
            "vl4r.v         v4, (%[B])            \n\t"  // 32*32@i8
            "addi           %[B], %[B], 512       \n\t"
            "vl4r.v         v8, (%[B])            \n\t"  // 32*32@i8
            "addi           %[B], %[B], 512       \n\t"

            "vsetvli        t0, zero, e32, m1     \n\t"
            "vupack.vv      v2, v0, v0, 1         \n\t"

            "vupack.vv      v24, v4, v5, 1        \n\t"
            "vupack.vv      v26, v6, v7, 1        \n\t"
            "vupack.vv      v4, v8, v9, 1         \n\t"
            "vupack.vv      v6, v10, v11, 1       \n\t"

            // init the accumu to asum * zp
            "vsetvli        t0, x0, e32, m1       \n\t"
            "vxor.vv        v16, v16, v16         \n\t"
            "vxor.vv        v18, v16, v16         \n\t"
            "vxor.vv        v20, v16, v16         \n\t"
            "vxor.vv        v22, v16, v16         \n\t"

            // i4 * i4 vmadot
            "vsetvli        t0, x0, e32, m1       \n\t"
            "vmadot         v16, v2, v24, i8      \n\t"
            "vmadot         v18, v2, v26, i8      \n\t"
            "vmadot         v20, v2, v4, i8       \n\t"
            "vmadot         v22, v2, v6, i8       \n\t"
            "vmadot         v16, v3, v25, i8      \n\t"
            "vmadot         v18, v3, v27, i8      \n\t"
            "vmadot         v20, v3, v5, i8       \n\t"
            "vmadot         v22, v3, v7, i8       \n\t"

            "vpack.vv       v0, v16, v18, 2       \n\t"
            "vpack.vv       v2, v20, v22, 2       \n\t"
            "vpack.vv       v16, v0, v2, 3        \n\t"
            "vpack.vv       v18, v1, v3, 3        \n\t"

            "vfcvt.f.x.v    v16, v16              \n\t"
            "vfcvt.f.x.v    v17, v17              \n\t"
            "vfcvt.f.x.v    v18, v18              \n\t"
            "vfcvt.f.x.v    v19, v19              \n\t"

            // mul scale
            "vfmul.vv       v16, v16, v14         \n\t"
            "vfmul.vv       v17, v17, v14         \n\t"
            "vfmul.vv       v18, v18, v14         \n\t"
            "vfmul.vv       v19, v19, v14         \n\t"

            "addi           %[BK], %[BK], -1      \n\t"
            "vfmacc.vf      v28, fa0, v16         \n\t"
            "vfmacc.vf      v29, fa1, v17         \n\t"
            "vfmacc.vf      v30, fa2, v18         \n\t"
            "vfmacc.vf      v31, fa3, v19         \n\t"

            "bgtz           %[BK], BLK_LOOP%=     \n\t"

            // save
            "vsetvli        t0, x0, e32, m1       \n\t"
            "add            t2, %[LDC], %[DST]    \n\t"
            "vse32.v        v28, (%[DST])         \n\t"
            "add            t3, %[LDC], t2        \n\t"
            "vse32.v        v29, (t2)             \n\t"
            "add            t2, %[LDC], t3        \n\t"
            "vse32.v        v30, (t3)             \n\t"
            "vse32.v        v31, (t2)             \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
            : "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
              "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
    }
}

void moe_m2_gemm_kernel_i8i4_impl(size_t           blk_len,
                                  const uint8_t ** quant_a_ptr,
                                  const uint8_t *  quant_b_data,
                                  const uint8_t *  quant_b_zp,
                                  float **         c_ptr,
                                  size_t           count_m,
                                  size_t           count_n,
                                  size_t           k_blks,
                                  size_t           ldc) {
#if 0
    moe_gemm_kernel_i8i4_mrow_ref<2, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks,
                                     ldc);
#else
    int64_t b_data_stride =
        k_blks * (sizeof(ggml_fp16_t) + 16 * sizeof(int8_t) + (quant_b_zp != NULL ? sizeof(int8_t) : 0));
    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data  = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data0 = (int8_t *) quant_a_ptr[0];
            int8_t *  a_data1 = (int8_t *) quant_a_ptr[1];
            float *   dst_c0  = (float *) c_ptr[0] + ni;
            float *   dst_c1  = (float *) c_ptr[1] + ni;

            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v29, v29, v29           \n\t"
                "vmv.v.i        v0, 1                   \n\t"  // init the scale
                "vsll.vi        v1, v0, 4               \n\t"
                "vfcvt.f.x.v    v0, v0                  \n\t"
                "vfcvt.f.x.v    v1, v1                  \n\t"
                "mv             t3, %[BK]               \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // load scale A0
                "flw            fa0, (%[A0])            \n\t"  // A0 scale
                "lh             t1, 4(%[A0])            \n\t"  // A0 asum
                "addi           %[A0], %[A0], 6         \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v12, (%[B])             \n\t"
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v14, v12, v12, 3        \n\t"

                // load scale A1
                "flw            fa1, (%[A1])            \n\t"  // A1 scale
                "lh             t2, 4(%[A1])            \n\t"  // A1 asum
                "addi           %[A1], %[A1], 6         \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmv.v.x        v10, t1                 \n\t"
                "vmv.v.x        v11, t2                 \n\t"

                "vpack.vv       v18, v10, v11, 1        \n\t"
                "vsll.vi        v18, v18, 3             \n\t"  // mul 8
                "vfcvt.f.x.v    v18, v18                \n\t"

                "vsetvli        t0, x0, e8, mf4         \n\t"  // A0 data
                "vle8.v         v16, (%[A0])            \n\t"
                "addi           %[A0], %[A0], 32        \n\t"  // 1*32@i8
                "vle8.v         v20, (%[A1])            \n\t"
                "addi           %[A1], %[A1], 32        \n\t"  // 1*32@i8

                "vl4r.v         v4, (%[B])              \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"

                "vsrl.vi        v17, v16, 4             \n\t"
                "vsrl.vi        v21, v20, 4             \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vnpack4.vv     v2, v16, v20, 2         \n\t"  // low  u4
                "vnpack4.vv     v3, v17, v21, 2         \n\t"  // high s4

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vor.vv         v19, v18, v18           \n\t"
                "vor.vv         v20, v18, v18           \n\t"
                "vor.vv         v21, v18, v18           \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmadotsu.hp    v18, v3, v4, v1, 0, i4  \n\t"  // high 4
                "vmadotsu.hp    v19, v3, v5, v1, 0, i4  \n\t"
                "vmadotsu.hp    v20, v3, v6, v1, 0, i4  \n\t"
                "vmadotsu.hp    v21, v3, v7, v1, 0, i4  \n\t"
                "vmadotu.hp     v18, v2, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp     v19, v2, v5, v0, 0, i4  \n\t"
                "vmadotu.hp     v20, v2, v6, v0, 0, i4  \n\t"
                "vmadotu.hp     v21, v2, v7, v0, 0, i4  \n\t"

                "vpack.vv       v8, v18, v19, 1         \n\t"
                "vpack.vv       v12, v20, v21, 1        \n\t"
                "vpack.vv       v20, v8, v12, 2         \n\t"

                "vfwmul.vv      v16, v20, v14           \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"

                "addi           t3, t3, -1              \n\t"
                "vfmacc.vf      v28, fa0, v16           \n\t"
                "vfmacc.vf      v29, fa1, v17           \n\t"

                "bgtz           t3, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vse32.v        v28, (%[DST0])          \n\t"
                "vse32.v        v29, (%[DST1])          \n\t"
                : [A0] "+r"(a_data0), [A1] "+r"(a_data1), [B] "+r"(b_data)
                : [DST0] "r"(dst_c0), [DST1] "r"(dst_c1), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                  "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
        }
    } else {
#    if 0
        moe_gemm_kernel_i8i4_mrow_ref<2, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                         k_blks, ldc);
#    else
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t * b_data  = (uint8_t *) quant_b_data + ni * b_data_stride;
            int8_t *  a_data0 = (int8_t *) quant_a_ptr[0];
            int8_t *  a_data1 = (int8_t *) quant_a_ptr[1];
            float *   dst_c0  = (float *) c_ptr[0] + ni;
            float *   dst_c1  = (float *) c_ptr[1] + ni;

            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v29, v29, v29           \n\t"
                "vmv.v.i        v0, 1                   \n\t"  // init the scale
                "vsll.vi        v1, v0, 4               \n\t"
                "vfcvt.f.x.v    v0, v0                  \n\t"
                "vfcvt.f.x.v    v1, v1                  \n\t"
                "mv             t3, %[BK]               \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // load scale A0
                "flw            fa0, (%[A0])            \n\t"  // A0 scale
                "lh             t1, 4(%[A0])            \n\t"  // A0 asum
                "addi           %[A0], %[A0], 6         \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v12, (%[B])             \n\t"
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v14, v12, v12, 3        \n\t"

                // load scale A1
                "flw            fa1, (%[A1])            \n\t"  // A1 scale
                "lh             t2, 4(%[A1])            \n\t"  // A1 asum
                "addi           %[A1], %[A1], 6         \n\t"

                // load zp
                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v8, (%[B])              \n\t"
                "addi           %[B], %[B], 32          \n\t"
                "vwaddu.vx      v10, v8, x0             \n\t"

                "vsetvli        t0, x0, e8, mf4         \n\t"  // A0 data
                "vle8.v         v16, (%[A0])            \n\t"
                "addi           %[A0], %[A0], 32        \n\t"  // 1*32@i8
                "vle8.v         v20, (%[A1])            \n\t"
                "addi           %[A1], %[A1], 32        \n\t"  // 1*32@i8

                "vl4r.v         v4, (%[B])              \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"

                "vsrl.vi        v17, v16, 4             \n\t"
                "vsrl.vi        v21, v20, 4             \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vnpack4.vv     v2, v16, v20, 2         \n\t"  // low  u4
                "vnpack4.vv     v3, v17, v21, 2         \n\t"  // high s4

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v18, v18, v18           \n\t"
                "vxor.vv        v19, v19, v19           \n\t"
                "vxor.vv        v20, v20, v20           \n\t"
                "vxor.vv        v21, v21, v21           \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmadotsu.hp    v18, v3, v4, v1, 0, i4  \n\t"  // high 4
                "vmadotsu.hp    v19, v3, v5, v1, 0, i4  \n\t"
                "vmadotsu.hp    v20, v3, v6, v1, 0, i4  \n\t"
                "vmadotsu.hp    v21, v3, v7, v1, 0, i4  \n\t"
                "vmadotu.hp     v18, v2, v4, v0, 0, i4  \n\t"  // low 4
                "vmadotu.hp     v19, v2, v5, v0, 0, i4  \n\t"
                "vmadotu.hp     v20, v2, v6, v0, 0, i4  \n\t"
                "vmadotu.hp     v21, v2, v7, v0, 0, i4  \n\t"

                "vpack.vv       v8, v18, v19, 1         \n\t"
                "vpack.vv       v12, v20, v21, 1        \n\t"
                "vpack.vv       v20, v8, v12, 2         \n\t"
                // asum*zp
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vwmul.vx       v2, v10, t1             \n\t"
                "vwmul.vx       v4, v10, t2             \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"

                "vfcvt.f.x.v    v2, v2                  \n\t"
                "vfcvt.f.x.v    v4, v4                  \n\t"

                "vsetvli        t0, x0, e16, m1         \n\t"
                "vfwcvt.f.f.v   v16, v20                \n\t"

                "vfwcvt.f.f.v   v18, v14                \n\t"

                // +asum*zp
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vfadd.vv       v16, v16, v2            \n\t"
                "vfadd.vv       v17, v17, v4            \n\t"
                "vfmul.vv       v16, v16, v18           \n\t"
                "vfmul.vv       v17, v17, v18           \n\t"

                "addi           t3, t3, -1              \n\t"
                "vfmacc.vf      v28, fa0, v16           \n\t"
                "vfmacc.vf      v29, fa1, v17           \n\t"

                "bgtz           t3, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vse32.v        v28, (%[DST0])          \n\t"
                "vse32.v        v29, (%[DST1])          \n\t"
                : [A0] "+r"(a_data0), [A1] "+r"(a_data1), [B] "+r"(b_data)
                : [DST0] "r"(dst_c0), [DST1] "r"(dst_c1), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                  "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
        }
#    endif
    }
#endif
}

void moe_m2_gemm_kernel_i8i5_impl(size_t           blk_len,
                                  const uint8_t ** quant_a_ptr,
                                  const uint8_t *  quant_b_data,
                                  const uint8_t *  quant_b_zp,
                                  float **         c_ptr,
                                  size_t           count_m,
                                  size_t           count_n,
                                  size_t           k_blks,
                                  size_t           ldc) {
    constexpr size_t NB_COLS          = 32;
    constexpr size_t B_Q50_BLK_STRIDE = sizeof(nrow_block_q5_0<NB_COLS>);
    constexpr size_t B_Q51_BLK_STRIDE = sizeof(nrow_block_q5_1<NB_COLS>);

    GGML_UNUSED(blk_len);
    GGML_UNUSED(count_m);
    GGML_UNUSED(ldc);

    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            uint8_t * b_data  = (uint8_t *) quant_b_data + (ni / NB_COLS) * k_blks * B_Q50_BLK_STRIDE;
            int8_t *  a_data0 = (int8_t *) quant_a_ptr[0];
            int8_t *  a_data1 = (int8_t *) quant_a_ptr[1];
            float *   dst_c0  = (float *) c_ptr[0] + ni;
            float *   dst_c1  = (float *) c_ptr[1] + ni;

            asm volatile(
                "mv             t4, %[BK]               \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v2, v0, v0              \n\t"
                "vxor.vv        v3, v0, v0              \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // ---- load B scale/Bh/Bs and advance to the next q5_0 k-block ----
                "vsetvli        t0, x0, e8, mf2         \n\t"
                "vle8.v         v1, (%[B])              \n\t"  // v1 = scale_fp16 × 32
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v0, (%[B])              \n\t"  // v0 = Bh N32K32 1-bit packed
                "addi           %[B], %[B], 128         \n\t"
                "vl4r.v         v8, (%[B])              \n\t"  // v8..v11 = Bs N32K32 i4
                "addi           %[B], %[B], 512         \n\t"

                // ---- load A0/A1 header then payload, each block stride = 38B ----
                "flw            f0, (%[A0])             \n\t"  // f0 = A0 scale (fp32)
                "lh             t2, 4(%[A0])            \n\t"  // t2 = A0 asum (int16)
                "addi           %[A0], %[A0], 6         \n\t"
                "flw            f1, (%[A1])             \n\t"  // f1 = A1 scale (fp32)
                "lh             t3, 4(%[A1])            \n\t"  // t3 = A1 asum (int16)
                "addi           %[A1], %[A1], 6         \n\t"
                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v4, (%[A0])             \n\t"  // v4 = A0 M1K32 int8
                "addi           %[A0], %[A0], 32        \n\t"
                "vle8.v         v5, (%[A1])             \n\t"  // v5 = A1 M1K32 int8
                "addi           %[A1], %[A1], 32        \n\t"

                //// ---- A nibble unpacking ----
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vand.vi        v12, v8, 0xF            \n\t"  //8bit(lo4) //[8*32]
                "vand.vi        v13, v9, 0xF            \n\t"
                "vand.vi        v14, v10, 0xF           \n\t"
                "vand.vi        v15, v11, 0xF           \n\t"
                "vsrl.vi        v8, v8, 4               \n\t"  //8bit(hi4)
                "vsrl.vi        v9, v9, 4               \n\t"
                "vsrl.vi        v10, v10, 4             \n\t"
                "vsrl.vi        v11, v11, 4             \n\t"

                "slli           t2, t2, 4               \n\t"  // a_sum * 16;
                "slli           t3, t3, 4               \n\t"
                // [4*32]*2
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vpack.vv       v16, v12, v8, 0         \n\t"
                "vpack.vv       v18, v13, v9, 0         \n\t"
                "vpack.vv       v20, v14, v10, 0        \n\t"
                "vpack.vv       v22, v15, v11, 0        \n\t"

                "li             t1, 16                  \n\t"
                "vsetvli        t0, x0, e8, m8          \n\t"
                "vadd.vx        v16, v16, t1, v0.t      \n\t"

                // [4*32]*2 -> [8*16]
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vupack.vv      v8, v16, v17, 1         \n\t"
                "vupack.vv      v10, v18, v19, 1        \n\t"
                "vupack.vv      v12, v20, v21, 1        \n\t"
                "vupack.vv      v14, v22, v23, 1        \n\t"

                "vpack.vv       v6, v4, v5, 2           \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v24, v16, v16           \n\t"
                "vxor.vv        v26, v16, v16           \n\t"
                "vupack.vv      v4, v6, v7, 1           \n\t"
                "vxor.vv        v28, v16, v16           \n\t"
                "vxor.vv        v30, v16, v16           \n\t"

                // ---- i8 main dot products ----
                // vmadot: A × unsigned Bh × 16 → fp16 accumulate
                "vmadot         v24, v4, v8, i8         \n\t"  // N0..7
                "vmadot         v26, v4, v10, i8        \n\t"  // N8..15
                "vmadot         v28, v4, v12, i8        \n\t"  // N16..23
                "vmadot         v30, v4, v14, i8        \n\t"  // N24..31
                // vmadot: A × unsigned Bh × 1 → fp16 accumulate
                "vmadot         v24, v5, v9, i8         \n\t"  // N0..7
                "vmadot         v26, v5, v11, i8        \n\t"  // N8..15
                "vmadot         v28, v5, v13, i8        \n\t"  // N16..23
                "vmadot         v30, v5, v15, i8        \n\t"  // N24..31

                "vpack.vv       v16, v24, v26, 2        \n\t"  // v16 = N0..15
                "vpack.vv       v18, v28, v30, 2        \n\t"  // v18 = N16..31
                "vpack.vv       v24, v16, v18, 3        \n\t"  // v24 = N0..31

                "vadd.vx        v24, v24, t2            \n\t"
                "vadd.vx        v25, v25, t3            \n\t"
                // b_scale fp16 -> fp32
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v   v28, v1                 \n\t"

                // a_scale * b_scale;
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v    v26, v24                \n\t"
                "vfcvt.f.x.v    v27, v25                \n\t"
                "vfmul.vf       v30, v28, f0            \n\t"
                "vfmul.vf       v31, v28, f1            \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv      v2, v30, v26            \n\t"
                "vfmacc.vv      v3, v31, v27            \n\t"

                "addi           t4, t4, -1              \n\t"
                "bgtz           t4, BLK_LOOP%=          \n\t"

                "vsetvli        t0, %[NR], e32, m1      \n\t"
                "vse32.v        v2, (%[DST0])           \n\t"
                "vse32.v        v3, (%[DST1])           \n\t"
                : [A0] "+r"(a_data0), [A1] "+r"(a_data1), [B] "+r"(b_data)
                : [DST0] "r"(dst_c0), [DST1] "r"(dst_c1), [BK] "r"(k_blks), [NR] "r"(nb_real)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                  "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
                  "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "f0", "f1");
        }
    } else {
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t    nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            uint8_t * b_data  = (uint8_t *) quant_b_data + (ni / NB_COLS) * k_blks * B_Q51_BLK_STRIDE;
            int8_t *  a_data0 = (int8_t *) quant_a_ptr[0];
            int8_t *  a_data1 = (int8_t *) quant_a_ptr[1];
            float *   dst_c0  = (float *) c_ptr[0] + ni;
            float *   dst_c1  = (float *) c_ptr[1] + ni;

            asm volatile(
                "mv             t4, %[BK]               \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v2, v0, v0              \n\t"
                "vxor.vv        v3, v0, v0              \n\t"
                "addi           t5, %[B], 64            \n\t"  // t5 = zp   (32B)
                "addi           t6, %[B], 96            \n\t"  // t6 = qh   (128B)
                "addi           s1, %[B], 224           \n\t"  // s1 = qs   (512B)

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // ---- load B scale/zp/Bh/Bs and advance to the next q5_1 k-block ----
                "vsetvli        t0, x0, e8, mf2         \n\t"
                "vle8.v         v1, (%[B])              \n\t"  // v1 = scale_fp16 × 32
                "addi           %[B], %[B], 736         \n\t"
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vle8.v         v0, (t6)                \n\t"  // v0 = Bh N32K32 1-bit packed
                "addi           t6, t6, 736             \n\t"
                "vl4r.v         v8, (s1)                \n\t"  // v8..v11 = Bs N32K32 i4
                "addi           s1, s1, 736             \n\t"

                // ---- load A0/A1 header then payload, each block stride = 38B ----
                "flw            f0, (%[A0])             \n\t"  // f0 = A0 scale (fp32)
                "lh             t2, 4(%[A0])            \n\t"  // t2 = A0 asum (int16)
                "addi           %[A0], %[A0], 6         \n\t"
                "flw            f1, (%[A1])             \n\t"  // f1 = A1 scale (fp32)
                "lh             t3, 4(%[A1])            \n\t"  // t3 = A1 asum (int16)
                "addi           %[A1], %[A1], 6         \n\t"
                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v4, (%[A0])             \n\t"  // v4 = A0 M1K32 int8
                "addi           %[A0], %[A0], 32        \n\t"
                "vle8.v         v5, (%[A1])             \n\t"  // v5 = A1 M1K32 int8
                "addi           %[A1], %[A1], 32        \n\t"

                //// ---- A nibble unpacking ----
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vand.vi        v12, v8, 0xF            \n\t"  //8bit(lo4) //[8*32]
                "vand.vi        v13, v9, 0xF            \n\t"
                "vand.vi        v14, v10, 0xF           \n\t"
                "vand.vi        v15, v11, 0xF           \n\t"
                "vsrl.vi        v8, v8, 4               \n\t"  //8bit(hi4)
                "vsrl.vi        v9, v9, 4               \n\t"
                "vsrl.vi        v10, v10, 4             \n\t"
                "vsrl.vi        v11, v11, 4             \n\t"

                // q5_1 uses explicit zp, so keep a_sum unshifted here.
                // [4*32]*2
                "vpack.vv       v16, v12, v8, 0         \n\t"
                "vpack.vv       v18, v13, v9, 0         \n\t"
                "vpack.vv       v20, v14, v10, 0        \n\t"
                "vpack.vv       v22, v15, v11, 0        \n\t"

                "li             t1, 16                  \n\t"
                "vsetvli        t0, x0, e8, m8          \n\t"
                "vadd.vx        v16, v16, t1, v0.t      \n\t"

                // [4*32]*2 -> [8*16]
                "vsetvli        t0, x0, e8, m1          \n\t"
                "vupack.vv      v8, v16, v17, 1         \n\t"
                "vupack.vv      v10, v18, v19, 1        \n\t"
                "vupack.vv      v12, v20, v21, 1        \n\t"
                "vupack.vv      v14, v22, v23, 1        \n\t"

                "vpack.vv       v6, v4, v5, 2           \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v24, v16, v16           \n\t"
                "vxor.vv        v26, v16, v16           \n\t"
                "vupack.vv      v4, v6, v7, 1           \n\t"
                "vxor.vv        v28, v16, v16           \n\t"
                "vxor.vv        v30, v16, v16           \n\t"

                // ---- i8 main dot products ----
                // vmadot: A × unsigned Bh × 16 → fp16 accumulate
                "vmadot         v24, v4, v8, i8         \n\t"  // N0..7
                "vmadot         v26, v4, v10, i8        \n\t"  // N8..15
                "vmadot         v28, v4, v12, i8        \n\t"  // N16..23
                "vmadot         v30, v4, v14, i8        \n\t"  // N24..31
                // vmadot: A × unsigned Bh × 1 → fp16 accumulate
                "vmadot         v24, v5, v9, i8         \n\t"  // N0..7
                "vmadot         v26, v5, v11, i8        \n\t"  // N8..15
                "vmadot         v28, v5, v13, i8        \n\t"  // N16..23
                "vmadot         v30, v5, v15, i8        \n\t"  // N24..31

                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v4, (t5)                \n\t"  // v4 = Bzp N32 uint8
                "addi           t5, t5, 736             \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vpack.vv       v16, v24, v26, 2        \n\t"  // v16 = N0..15
                "vpack.vv       v18, v28, v30, 2        \n\t"  // v18 = N16..31
                "vpack.vv       v24, v16, v18, 3        \n\t"  // v24 = N0..31

                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vwaddu.vx      v28, v4, x0             \n\t"

                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vwmul.vx       v30, v28, t2            \n\t"
                "vwmul.vx       v31, v28, t3            \n\t"

                // b_scale fp16 -> fp32
                "vfwcvt.f.f.v   v28, v1                 \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"
                "vadd.vv        v24, v24, v30           \n\t"
                "vadd.vv        v25, v25, v31           \n\t"

                // a_scale * b_scale;
                "vfcvt.f.x.v    v26, v24                \n\t"
                "vfcvt.f.x.v    v27, v25                \n\t"
                "vfmul.vf       v30, v28, f0            \n\t"
                "vfmul.vf       v31, v28, f1            \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv      v2, v30, v26            \n\t"
                "vfmacc.vv      v3, v31, v27            \n\t"

                "addi           t4, t4, -1              \n\t"
                "bgtz           t4, BLK_LOOP%=          \n\t"

                "vsetvli        t0, %[NR], e32, m1      \n\t"
                "vse32.v        v2, (%[DST0])           \n\t"
                "vse32.v        v3, (%[DST1])           \n\t"
                : [A0] "+r"(a_data0), [A1] "+r"(a_data1), [B] "+r"(b_data)
                : [DST0] "r"(dst_c0), [DST1] "r"(dst_c1), [BK] "r"(k_blks), [NR] "r"(nb_real)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s1", "v0", "v1", "v2", "v3", "v4", "v5",
                  "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                  "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "f0", "f1");
        }
    }
}

size_t gemm_kernel_i8i2k(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    if (count_m >= 4) {
#if 0
        gemm_kernel_i8i2k_mrow_ref<4, 32>(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_m, count_n, k_blks, ldc);
#else
        gemm_kernel_i8i2k_m4(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 4;
    } else {
#if 0
        gemm_kernel_i8i2k_mrow_ref<1, 32>(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_m, count_n, k_blks,
                                      ldc);
#else
        gemm_kernel_i8i2k_m1(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 1;
    }
}

size_t gemm_kernel_i8i3k(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    if (count_m >= 4) {
#if 0
        gemm_kernel_i8i3k_mrow_ref<4, 32>(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_m, count_n, k_blks, ldc);
#else
        gemm_kernel_i8i3k_m4(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 4;
    } else {
#if 0
        gemm_kernel_i8i3k_mrow_ref<1, 32>(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_m, count_n, k_blks, ldc);
#else
        gemm_kernel_i8i3k_m1(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 1;
    }
}

size_t gemm_kernel_i8i4(size_t          blk_len,
                        const uint8_t * quant_a_ptr,
                        const uint8_t * quant_b_data,
                        const uint8_t * quant_b_zp,
                        float *         c_ptr,
                        size_t          count_m,
                        size_t          count_n,
                        size_t          k_blks,
                        size_t          ldc) {
    if (count_m >= 4) {
#if 0
        gemm_kernel_i8i4_mrow_ref<4, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                         k_blks, ldc);
#else
        gemm_kernel_i8i4_m4(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 4;
    } else {
#if 0
        gemm_kernel_i8i4_mrow_ref<1, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                         k_blks, ldc);
#else
        gemm_kernel_i8i4_m1(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 1;
    }
}

size_t gemm_kernel_i8i4_hp(size_t          blk_len,
                           const uint8_t * quant_a_ptr,
                           const uint8_t * quant_b_data,
                           const uint8_t * quant_b_zp,
                           float *         c_ptr,
                           size_t          count_m,
                           size_t          count_n,
                           size_t          k_blks,
                           size_t          ldc) {
    if (count_m >= 4) {
#if 0
        gemm_kernel_i8i4_hp_mrow_ref<4, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                            k_blks, ldc);
#else
        gemm_kernel_i8i4_hp_m4(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 4;
    } else {
#if 0
        gemm_kernel_i8i4_hp_mrow_ref<1, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                            k_blks, ldc);
#else
        gemm_kernel_i8i4_hp_m1(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 1;
    }
}

size_t moe_m2_gemm_kernel_i8i4(size_t           blk_len,
                               const uint8_t ** quant_a_ptr,
                               const uint8_t *  quant_b_data,
                               const uint8_t *  quant_b_zp,
                               float **         c_ptr,
                               size_t           count_m,
                               size_t           count_n,
                               size_t           k_blks,
                               size_t           ldc) {
    moe_m2_gemm_kernel_i8i4_impl(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
    return 2;
}

size_t gemm_kernel_i8i8(size_t          blk_len,
                        const uint8_t * quant_a_ptr,
                        const uint8_t * quant_b_data,
                        const uint8_t * quant_b_zp,
                        float *         c_ptr,
                        size_t          count_m,
                        size_t          count_n,
                        size_t          k_blks,
                        size_t          ldc) {
    if (count_m >= 4) {
#if 0
        gemm_kernel_i8i8_mrow_ref<4, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                         k_blks, ldc);
#else
        gemm_kernel_i8i8_m4(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 4;
    } else {
#if 0
        gemm_kernel_i8i8_mrow_ref<1, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                         k_blks, ldc);
#else
        gemm_kernel_i8i8_m1(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 1;
    }
}

size_t gemm_kernel_i8mxfp4(size_t          blk_len,
                           const uint8_t * quant_a_ptr,
                           const uint8_t * quant_b_data,
                           const uint8_t * quant_b_zp,
                           float *         c_ptr,
                           size_t          count_m,
                           size_t          count_n,
                           size_t          k_blks,
                           size_t          ldc) {
    if (count_m >= 4) {
#if 1
        gemm_kernel_i8mxfp4_mrow_ref<4, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                            k_blks, ldc);
#else
        gemm_kernel_i8mxfp4_m4(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 4;
    } else {
#if 1
        gemm_kernel_i8mxfp4_mrow_ref<1, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                            k_blks, ldc);
#else
        gemm_kernel_i8mxfp4_m1(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 1;
    }
}

size_t moe_m2_gemm_kernel_i8mxfp4(size_t           blk_len,
                                  const uint8_t ** quant_a_ptr,
                                  const uint8_t *  quant_b_data,
                                  const uint8_t *  quant_b_zp,
                                  float **         c_ptr,
                                  size_t           count_m,
                                  size_t           count_n,
                                  size_t           k_blks,
                                  size_t           ldc) {
    //moe_m2_gemm_kernel_i8mxfp4_impl(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
    return 2;
}

size_t gemm_kernel_i8i5(size_t          blk_len,
                        const uint8_t * quant_a_ptr,
                        const uint8_t * quant_b_data,
                        const uint8_t * quant_b_zp,
                        float *         c_ptr,
                        size_t          count_m,
                        size_t          count_n,
                        size_t          k_blks,
                        size_t          ldc) {
    if (count_m >= 4) {
#if 0
        gemm_kernel_i8i5_mrow_ref<4, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                         k_blks, ldc);
#else
        gemm_kernel_i8i5_m4(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 4;
    } else {
#if 0
        gemm_kernel_i8i5_mrow_ref<1, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                         k_blks, ldc);
#else
        gemm_kernel_i8i5_m1(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
        return 1;
    }
}

size_t moe_m2_gemm_kernel_i8i5(size_t           blk_len,
                               const uint8_t ** quant_a_ptr,
                               const uint8_t *  quant_b_data,
                               const uint8_t *  quant_b_zp,
                               float **         c_ptr,
                               size_t           count_m,
                               size_t           count_n,
                               size_t           k_blks,
                               size_t           ldc) {
#if 0
    moe_gemm_kernel_i8i5_mrow_ref<2, 32>(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n,
                                         k_blks, ldc);
#else
    moe_m2_gemm_kernel_i8i5_impl(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
#endif
    return 2;
}

}  // namespace ime2
}  // namespace spacemit_kernels
