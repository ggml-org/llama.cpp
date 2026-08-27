#include "common.cuh"
#include "ggml.h"

bool ggml_cuda_op_dsv4_hc_mixes(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        int cc,
        int warp_size);

bool ggml_cuda_use_qwen4exp_hc_combine();
bool ggml_cuda_use_qwen4exp_hc_reduce();
bool ggml_cuda_use_qwen4exp_hc_scale_silu();

bool ggml_cuda_op_qwen4exp_hc_combine(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * residual,
        const ggml_tensor * block_repeat,
        const ggml_tensor * inject,
        ggml_tensor * dst);

bool ggml_cuda_op_qwen4exp_hc_reduce(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * streams,
        ggml_tensor * dst);

bool ggml_cuda_op_qwen4exp_hc_scale_silu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        ggml_tensor * dst);

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
