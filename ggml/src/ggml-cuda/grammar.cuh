#pragma once
#include "common.cuh"
#include "../ggml-grammar.h"

bool ggml_cuda_grammar_compute(ggml_backend_cuda_context & ctx, const ggml_tensor * program, const ggml_tensor * vocab,
        const ggml_tensor * state, ggml_tensor * scratch, ggml_tensor * cache, ggml_tensor * mask, ggml_tensor * status);
