#pragma once

#include "common.cuh"
#include "ggml-sampling.h"

bool ggml_cuda_sample(ggml_backend_cuda_context & ctx, const ggml_tensor * logits,
        const ggml_tensor * draft_probs, const ggml_sample_params * params, ggml_sample_result * result);
