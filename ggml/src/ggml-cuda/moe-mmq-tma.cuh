#pragma once

#include "common.cuh"

struct mmq_args;
struct ggml_cuda_moe_mmq_state;
struct ggml_cuda_moe_weight_view;

bool ggml_cuda_moe_mmq_tma_supported(const ggml_cuda_moe_weight_view & weight,
                                     int                               tile_rows,
                                     bool                              warp_specialized,
                                     size_t                            smpbo,
                                     int                               epilogue = 0);

bool ggml_cuda_moe_mmq_tma(ggml_backend_cuda_context &     ctx,
                           const mmq_args &                args,
                           const ggml_cuda_moe_mmq_state & state,
                           cudaStream_t                    stream);
