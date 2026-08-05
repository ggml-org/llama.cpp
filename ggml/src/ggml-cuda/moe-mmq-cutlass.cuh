#pragma once

#include "moe-mmq-repack.cuh"

bool ggml_cuda_moe_cutlass_compiled();

struct ggml_cuda_moe_cutlass_config {
    int  tile_n;
    bool swap_ab;
    bool pdl;
};

size_t ggml_cuda_moe_cutlass_activation_size(int64_t n_rows, int64_t n_cols);
size_t ggml_cuda_moe_cutlass_scale_size(int64_t n_rows, int n_experts, int64_t n_cols);

void ggml_cuda_moe_cutlass_quantize_broadcast(const float *   src,
                                              const int32_t * ids,
                                              const int32_t * ids_src1,
                                              const int32_t * expert_bounds,
                                              uint8_t *       dst,
                                              uint8_t *       scales,
                                              int64_t         n_cols,
                                              int64_t         n_cols_padded,
                                              int64_t         stride_token,
                                              int64_t         n_tokens,
                                              int             n_experts,
                                              int             n_expert_used,
                                              int64_t         ids_stride,
                                              cudaStream_t    stream);

bool ggml_cuda_moe_cutlass_quantize_broadcast_cta(const float *   src,
                                                  const int32_t * ids,
                                                  const int32_t * ids_src1,
                                                  const int32_t * expert_bounds,
                                                  uint8_t *       dst,
                                                  uint8_t *       scales,
                                                  int64_t         n_cols,
                                                  int64_t         n_cols_padded,
                                                  int64_t         stride_token,
                                                  int64_t         n_tokens,
                                                  int             n_experts,
                                                  int             n_expert_used,
                                                  int64_t         ids_stride,
                                                  cudaStream_t    stream);

void ggml_cuda_moe_cutlass_quantize_routes(const float *   src,
                                           const int32_t * ids,
                                           const int32_t * ids_src1,
                                           const int32_t * expert_bounds,
                                           uint8_t *       dst,
                                           uint8_t *       scales,
                                           int64_t         n_cols,
                                           int64_t         n_cols_padded,
                                           int64_t         n_tokens,
                                           int             n_experts,
                                           int             n_expert_used,
                                           int64_t         ids_stride,
                                           cudaStream_t    stream);

bool ggml_cuda_moe_cutlass_gemm(ggml_backend_cuda_context &       ctx,
                                const ggml_cuda_moe_weight_view & weight,
                                const uint8_t *                   activation,
                                const uint8_t *                   activation_scales,
                                const int32_t *                   expert_bounds,
                                void *                            dst,
                                int                               n_experts,
                                int64_t                           n_rows,
                                int64_t                           n,
                                int64_t                           k,
                                int                               sm_count,
                                ggml_cuda_moe_cutlass_config      config,
                                cudaStream_t                      stream,
                                bool                              require);

void ggml_cuda_moe_cutlass_scatter(const void *    src,
                                   const int32_t * ids_dst,
                                   float *         dst,
                                   int64_t         n_cols,
                                   int64_t         n_rows,
                                   cudaStream_t    stream);

void ggml_cuda_moe_cutlass_w13_epilogue(const void *    gate_up,
                                        const float *   bias,
                                        const int32_t * ids,
                                        const int32_t * ids_dst,
                                        const int32_t * expert_bounds,
                                        uint8_t *       dst,
                                        uint8_t *       scales,
                                        int64_t         n_ff,
                                        int64_t         n_ff_padded,
                                        int64_t         n_rows,
                                        int             n_experts,
                                        int             n_expert_used,
                                        int64_t         ids_stride,
                                        cudaStream_t    stream);

bool ggml_cuda_moe_cutlass_w13_epilogue_cta(const void *    gate_up,
                                            const float *   bias,
                                            const int32_t * ids,
                                            const int32_t * ids_dst,
                                            const int32_t * row_expert,
                                            const int32_t * expert_bounds,
                                            uint8_t *       dst,
                                            uint8_t *       scales,
                                            int64_t         n_ff,
                                            int64_t         n_ff_padded,
                                            int64_t         n_rows,
                                            int             n_experts,
                                            int             n_expert_used,
                                            int             rows_per_cta,
                                            int64_t         ids_stride,
                                            cudaStream_t    stream);

void ggml_cuda_moe_cutlass_w2_finalize(const void *    down,
                                       const float *   bias,
                                       const float *   weights,
                                       const int32_t * ids,
                                       const int32_t * ids_src1,
                                       float *         dst,
                                       int64_t         n_embd,
                                       int64_t         n_tokens,
                                       int             n_expert_used,
                                       int64_t         ids_stride,
                                       cudaStream_t    stream);
