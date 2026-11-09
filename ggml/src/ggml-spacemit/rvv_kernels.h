#pragma once

#include "spacemit-context.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace spacemit_kernels {

template <typename T, typename U> constexpr auto div_round_up(T up, U down) {
    return (up + down - 1) / down;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_size(size_t blk_len, bool with_blk_sum = false) {
    const size_t blk_size = sizeof(float) + blk_len * sizeof(int8_t) + (with_blk_sum ? sizeof(int16_t) : 0);
    return blk_size;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_size(size_t blk_len, bool with_blk_sum = false, bool with_blk_scale = false) {
    const size_t subblk_count = div_round_up(blk_len, size_t(32));
    const size_t blk_size     = blk_len * sizeof(int8_t) + subblk_count * sizeof(_Float16) +
                            (with_blk_sum ? subblk_count * sizeof(_Float16) : 0) +
                            (with_blk_scale ? sizeof(_Float16) : 0);
    return blk_size;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_size(size_t blk_len) {
    const size_t blk_size = sizeof(float) + blk_len * sizeof(int8_t) + sizeof(int16_t) * blk_len / 16;
    return blk_size;
}

using quantize_a_row_def = std::function<void(size_t, const float *, size_t, uint8_t *)>;

namespace rvv {
void memcpy1d(void * dst, const void * src, int64_t size);

void memcpy2d(void * dst, int64_t dst_stride, const void * src, int64_t src_stride, int64_t tile_rows, int64_t size);

void forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ggml::spacemit::context & ctx,
                                                        ggml_tensor *             dst,
                                                        int                       ir0,
                                                        int                       ir1);

// split-KV decode: returns KV chunks per KV head when applicable, else 0
int  flash_attn_ext_f16_split_kv_chunks_vlen1024_vf16(const ggml_tensor * dst, int nth);
void forward_flash_attn_ext_f16_split_kv_partial_vlen1024_vf16(ggml::spacemit::context & ctx,
                                                               ggml_tensor *               dst,
                                                               int                         n_chunks,
                                                               float *                     partials,
                                                               void *                      tcm_buffer,
                                                               size_t                      tcm_buffer_size);
void forward_flash_attn_ext_f16_split_kv_reduce_vlen1024_vf16(ggml::spacemit::context & ctx,
                                                              ggml_tensor *               dst,
                                                              int                         n_chunks,
                                                              const float *               partials);

void forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ggml::spacemit::context & ctx,
                                                    ggml_tensor *             dst,
                                                    int                       ir0,
                                                    int                       ir1);

void forward_rms_norm_f32(ggml::spacemit::context & ctx, ggml_tensor * op);

void forward_norm_f32(ggml::spacemit::context & ctx, ggml_tensor * op);

void forward_cont_with_permute(ggml::spacemit::context & ctx, ggml_tensor * op);

void forward_cpy_with_permute(ggml::spacemit::context & ctx, ggml_tensor * op);

template <typename T> void forward_get_rows(ggml::spacemit::context & ctx, ggml_tensor * op);

template <typename T> void forward_concat(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_gated_delta_net(ggml::spacemit::context & ctx, ggml_tensor * op);
// dim-0 concat of a contiguous src0 and a transposed src1 (channels contiguous), vectorized over rows;
// writes columns >= col0 only
bool forward_concat_dim0_transposed_f32(ggml::spacemit::context & ctx, ggml_tensor * op, int64_t col0);
// ssm_conv, d_conv == 4, vectorized over channels
bool forward_ssm_conv_f32(ggml::spacemit::context & ctx, ggml_tensor * dst);
// ssm_conv reading its concat(conv_states, transpose(qkv)) inputs directly (fused conv)
bool forward_ssm_conv_concat_f32(ggml::spacemit::context & ctx, ggml_tensor * dst);
// RMS_NORM followed by MUL with a weight row, written to the MUL's dst
void forward_rms_norm_mul_f32(ggml::spacemit::context & ctx, const ggml_tensor * rms, ggml_tensor * mul);

template <ggml_op op_type, typename T> void forward_binary(ggml::spacemit::context & ctx, ggml_tensor * op);

void forward_scale_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_cpy_strided_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
// F32 -> F32 with contiguous rows (nb[0] == sizeof(float)) but arbitrary outer strides on both sides;
// used by the GDN rollback-snapshot copies into the row-strided cache_s views
void forward_cpy_f32_row_strided(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_cpy_f32_to_f16(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_cpy_f16_to_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_unary_tanh_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_unary_gelu_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_unary_silu_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_unary_exp_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_unary_sigmoid_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_unary_neg_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_unary_softplus_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_soft_max_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_mul_mat_f16_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
template <typename T> void forward_cont_general(ggml::spacemit::context & ctx, ggml_tensor * op);

void forward_glu_geglu_f32(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_glu_swiglu_f32(ggml::spacemit::context & ctx, ggml_tensor * op);

void forward_set_rows(ggml::spacemit::context & ctx, ggml_tensor * op);
void forward_rope(ggml::spacemit::context & ctx, ggml_tensor * op);

template <typename T> void forward_sum_rows(ggml::spacemit::context & ctx, ggml_tensor * op);

template <typename T> void forward_repeat_nrows(ggml::spacemit::context & ctx, ggml_tensor * op);

template <typename T> void forward_repeat_dim1(ggml::spacemit::context & ctx, ggml_tensor * op);

void quantize_a_row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_4row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_4row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
