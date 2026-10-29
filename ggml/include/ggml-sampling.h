#pragma once

#include "ggml-backend.h"

// Experimental backend sampling extension, resolved through get_proc_address.
#define GGML_SAMPLE_MAX_DRAFT 32
#define GGML_SAMPLE_MAX_BIAS 256

struct ggml_sample_params {
    int32_t n_draft;
    int32_t rows[GGML_SAMPLE_MAX_DRAFT + 1];
    int32_t draft[GGML_SAMPLE_MAX_DRAFT];
    int32_t q_row;
    float temperature;
    float top_p;
    int32_t min_keep;
    bool temperature_first;
    int32_t n_bias;
    int32_t bias_ids[GGML_SAMPLE_MAX_BIAS];
    float bias_values[GGML_SAMPLE_MAX_BIAS];
    float uniforms[2 * GGML_SAMPLE_MAX_DRAFT + 1];
};

struct ggml_sample_result {
    int32_t n_tokens;
    int32_t tokens[GGML_SAMPLE_MAX_DRAFT + 1];
};

typedef bool (*ggml_backend_sample_t)(ggml_backend_t backend,
        const struct ggml_tensor * logits, const struct ggml_tensor * draft_probs,
        const struct ggml_sample_params * params, struct ggml_sample_result * result);
