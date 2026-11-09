#pragma once

#include <cassert>
#include <cstddef>
#include <functional>

namespace spacemit_kernels {

#define BLOCK_QNK_LEN 256

template <int N> struct nrow_block_q2_k {
    // [4bit scale + 4bit zp] * N * 16
    uint8_t  scales[N * BLOCK_QNK_LEN / 16];
    // [b0, b16, b32, b48] [b1, b17, b33, b49] ... [b15, b31, b47, b63]
    // [b64, b80, b96, b112] ...[b79, b95, b111, b127]
    // [b128, b144, b160, b176] ...[b143, b159, b175, b191]
    // [b192, b208, b224, b240] ...[b207, b223, b239, b255]
    uint8_t  qs[N * BLOCK_QNK_LEN / 4];
    uint16_t scales16[N];
    uint16_t zeros16[N];
};

template <int N> struct nrow_block_q3_k {
    // [8bit scale] * N * 16
    int8_t   scales[N * 16];
    // [b0, b1, b2, b3, b4, b5, b6, b7] ... [b248, b249, b250, b251, b252, b253, b254, b255]
    uint8_t  hmask[N * BLOCK_QNK_LEN / 8];
    // [b0, b16, b32, b48] [b1, b17, b33, b49] ... [b15, b31, b47, b63]
    // [b64, b80, b96, b112] ...[b79, b95, b111, b127]
    // [b128, b144, b160, b176] ...[b143, b159, b175, b191]
    // [b192, b208, b224, b240] ...[b207, b223, b239, b255]
    uint8_t  qs[N * BLOCK_QNK_LEN / 4];
    uint16_t scales16[N];
};

template <int N> struct __attribute__((packed)) nrow_block_q5_1 {
    uint16_t scales16[N];
    uint8_t  zp[N];
    // n0 [bh0, bh1, bh2, bh3, bh4, bh5, bh6, bh7] ....
    uint8_t  qh[4 * N];
    // n0 [b0, b1], [b2, b3] ....  [b30, b31]
    // n1 [b0, b1], [b2, b3] ....  [b30, b31]
    uint8_t  qs[16 * N];
};

static_assert(sizeof(nrow_block_q5_1<1>) == sizeof(uint8_t) + 22, "wrong nrow_block_q5_1 block size/padding");

template <int N> struct __attribute__((packed)) nrow_block_q5_0 {
    uint16_t scales16[N];
    // n0 [bh0, bh1, bh2, bh3, bh4, bh5, bh6, bh7] ....
    uint8_t  qh[4 * N];
    // n0 [b0, b1], [b2, b3] ....  [b30, b31]
    // n1 [b0, b1], [b2, b3] ....  [b30, b31]
    uint8_t  qs[16 * N];
};

static_assert(sizeof(nrow_block_q5_0<1>) == 22, "wrong nrow_block_q5_0 block size/padding");

enum class a_pack : int {
    r1,     // one row per call, natural row layout
    r4,     // 4-row group, the family's natural group layout
    r8,     // 8-row group (k256 hp aligned family only)
    r4_al,  // 4-row group, aligned (SoA) layout
};

using gemm_kernel_quantize_def = std::function<
    size_t(a_pack, size_t, const uint8_t *, const uint8_t *, const uint8_t *, float *, size_t, size_t, size_t, size_t)>;

using gemm_rows_def = std::function<
    size_t(a_pack, size_t, const uint8_t **, const uint8_t *, const uint8_t *, float **, size_t, size_t, size_t, size_t)>;

namespace ime1 {
size_t gemm_i8i4_k32(a_pack          ap,
                     size_t          blk_len,
                     const uint8_t * quant_a_ptr,
                     const uint8_t * quant_b_data,
                     const uint8_t * quant_b_zp,
                     float *         c_ptr,
                     size_t          count_m,
                     size_t          count_n,
                     size_t          k_blks,
                     size_t          ldc);

void quantize_a_i8_r1(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_i8_r4(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

}  // namespace ime1

namespace ime2 {
size_t gemm_i8i2k_k256(a_pack          ap,
                       size_t          blk_len,
                       const uint8_t * quant_a_ptr,
                       const uint8_t * quant_b_data,
                       const uint8_t * quant_b_zp,
                       float *         c_ptr,
                       size_t          count_m,
                       size_t          count_n,
                       size_t          k_blks,
                       size_t          ldc);

size_t gemm_i8i3k_k256(a_pack          ap,
                       size_t          blk_len,
                       const uint8_t * quant_a_ptr,
                       const uint8_t * quant_b_data,
                       const uint8_t * quant_b_zp,
                       float *         c_ptr,
                       size_t          count_m,
                       size_t          count_n,
                       size_t          k_blks,
                       size_t          ldc);

size_t gemm_i8i4_k32(a_pack          ap,
                     size_t          blk_len,
                     const uint8_t * quant_a_ptr,
                     const uint8_t * quant_b_data,
                     const uint8_t * quant_b_zp,
                     float *         c_ptr,
                     size_t          count_m,
                     size_t          count_n,
                     size_t          k_blks,
                     size_t          ldc);

// MoE Q4 k256: null zp selects folded Q4_0 nibbles; otherwise B has unsigned nibbles and a zp area.
size_t gemm_i8i4_k256(a_pack          ap,
                      size_t          blk_len,
                      const uint8_t * quant_a_ptr,
                      const uint8_t * quant_b_data,
                      const uint8_t * quant_b_zp,
                      float *         c_ptr,
                      size_t          count_m,
                      size_t          count_n,
                      size_t          k_blks,
                      size_t          ldc);

size_t gemm_i8i8_k32(a_pack          ap,
                     size_t          blk_len,
                     const uint8_t * quant_a_ptr,
                     const uint8_t * quant_b_data,
                     const uint8_t * quant_b_zp,
                     float *         c_ptr,
                     size_t          count_m,
                     size_t          count_n,
                     size_t          k_blks,
                     size_t          ldc);

size_t gemm_i8i5_k32(a_pack          ap,
                     size_t          blk_len,
                     const uint8_t * quant_a_ptr,
                     const uint8_t * quant_b_data,
                     const uint8_t * quant_b_zp,
                     float *         c_ptr,
                     size_t          count_m,
                     size_t          count_n,
                     size_t          k_blks,
                     size_t          ldc);

size_t gemm_i8i4_k32_rows(a_pack           ap,
                         size_t           blk_len,
                         const uint8_t ** quant_a_ptr,
                         const uint8_t *  quant_b_data,
                         const uint8_t *  quant_b_zp,
                         float **         c_ptr,
                         size_t           count_m,
                         size_t           count_n,
                         size_t           k_blks,
                         size_t           ldc);

size_t gemm_i8i5_k32_rows(a_pack           ap,
                         size_t           blk_len,
                         const uint8_t ** quant_a_ptr,
                         const uint8_t *  quant_b_data,
                         const uint8_t *  quant_b_zp,
                         float **         c_ptr,
                         size_t           count_m,
                         size_t           count_n,
                         size_t           k_blks,
                         size_t           ldc);

// F16 tile kernels accept actual extents up to M16xN32; ldc is the row stride in elements.

// Q41Block GEMV uses per-row uint8 zero points with INT32 correction and FP32 accumulation.
void gemm_kernel_i8i4_m1_q41(const uint8_t * quant_a_ptr,
                           const uint8_t * quant_b_data,
                           float *         c_ptr,
                           size_t          count_n,
                           size_t          k_blks);

// A/B layouts: [K/8][M16|N32][K8]; gemm_k is the K extent.
void gemm_kernel_f16f16f16_m16_vfwmadot(size_t           gemm_k,
                                        const _Float16 * b_f16_k8,
                                        _Float16 *       a_f16_k8,
                                        _Float16 *       c_ptr,
                                        size_t           count_m,
                                        size_t           count_n,
                                        size_t           ldc,
                                        bool             accumulate);

// Same packed A/B layouts; accumulate seeds the F32 result from C.
void gemm_kernel_f16f16f32_m16_vfwmadot(size_t           gemm_k,
                                        const _Float16 * b_f16_k8,
                                        _Float16 *       a_f16_k8,
                                        float *          c_ptr,
                                        size_t           count_m,
                                        size_t           count_n,
                                        size_t           ldc,
                                        bool             accumulate = false);
}  // namespace ime2
}  // namespace spacemit_kernels
