#pragma once

#include "common.cuh"

bool ggml_cuda_diffusion_sample_topk(
        ggml_backend_t backend,
        const ggml_tensor * logits,
        const ggml_cuda_diffusion_sample_params * params,
        ggml_cuda_diffusion_sample_result * result);
