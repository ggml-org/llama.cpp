#pragma once

#include "moe-mmq.cuh"

void ggml_cuda_moe_mmq_w13_epilogue_staged(
    const ggml_cuda_moe_mmq_args & args, int64_t ids_stride, cudaStream_t stream);

void ggml_cuda_moe_mmq_w13_epilogue_fused(
    const ggml_cuda_moe_mmq_args & args, int64_t ids_stride, cudaStream_t stream);

void ggml_cuda_moe_mmq_w13_epilogue_quantize(const ggml_cuda_moe_mmq_args & args,
                                              const int32_t *                ids_dst,
                                              void *                         activation_q,
                                              int64_t                        activation_q_ne0,
                                              int64_t                        ids_stride,
                                              cudaStream_t                   stream);

void ggml_cuda_moe_mmq_w2_epilogue_staged(
    const ggml_cuda_moe_mmq_args & args, int64_t ids_stride, cudaStream_t stream);

void ggml_cuda_moe_mmq_w2_epilogue_fused(
    const ggml_cuda_moe_mmq_args & args, int64_t ids_stride, cudaStream_t stream);

void ggml_cuda_moe_mmq_reduce_weighted(const ggml_cuda_moe_mmq_args & args, cudaStream_t stream);
