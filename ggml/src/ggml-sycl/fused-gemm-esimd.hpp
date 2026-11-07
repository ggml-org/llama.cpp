#ifndef GGML_SYCL_FUSED_GEMM_ESIMD_HPP
#define GGML_SYCL_FUSED_GEMM_ESIMD_HPP

#include "fused-gemm.hpp"

// The grouped (MUL_MAT_ID) dequant GEMM, written with ESIMD so the dequantized weights go from
// registers straight into the dpas A operand: no SLM, no barrier. Same inputs and outputs as the
// joint_matrix grouped kernel of the 8x16x16 sub-group-16 combination of T (sycl::half or bfloat16):
// the schedule's tiles, B packed as [K/2][Npad][2] T, and dst through dst_rows. The weight is in the
// reorder (SoA) layout if `reordered`, else in the canonical block layout. Returns false when the
// shape is not covered.
template <typename block_q_t>
constexpr bool ggml_sycl_grouped_gemm_esimd_has = std::is_same_v<block_q_t, block_q8_0> ||
                                                  std::is_same_v<block_q_t, block_iq4_nl> ||
                                                  std::is_same_v<block_q_t, block_iq4_xs> ||
                                                  std::is_same_v<block_q_t, block_iq3_s> ||
                                                  std::is_same_v<block_q_t, block_iq3_xxs> ||
                                                  std::is_same_v<block_q_t, block_q2_K> ||
                                                  std::is_same_v<block_q_t, block_q3_K> ||
                                                  std::is_same_v<block_q_t, block_q4_K> ||
                                                  std::is_same_v<block_q_t, block_q5_K> ||
                                                  std::is_same_v<block_q_t, block_q6_K>;

template <typename block_q_t, typename T>
bool ggml_sycl_grouped_gemm_esimd(const char * src0_dd, size_t expert_stride, const ggml_sycl_gg_tile * tiles,
                                  const T * packed, float * dst, int M, int Npad, int K,
                                  const ggml_sycl_gg_rows & dst_rows, bool reordered, int64_t n_tiles,
                                  dpct::queue_ptr stream);

// GGML_SYCL_MMID_ESIMD: 1 (default) takes the ESIMD kernel above where it applies, 0 never does
bool ggml_sycl_grouped_gemm_esimd_enabled();

#endif // GGML_SYCL_FUSED_GEMM_ESIMD_HPP
