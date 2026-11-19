#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_route(ggml_backend_cuda_context & ctx, ggml_tensor * pre, ggml_tensor * post, ggml_tensor * comb);
void ggml_cuda_op_dsv4_hc_route_pre(ggml_backend_cuda_context & ctx, ggml_tensor * pre, ggml_tensor * post, ggml_tensor * comb);
void ggml_cuda_op_dsv4_hc_route_pre_norm(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        const ggml_tensor * weights, ggml_tensor * dst, ggml_tensor * post, ggml_tensor * comb);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);

void ggml_cuda_op_dsv4_hc_post_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm);
