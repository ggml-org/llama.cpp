#pragma once

#include <cstdint>

enum class ggml_cuda_mmvq_batch6_type {
    other,
    q4_k,
    q6_k,
};

struct ggml_cuda_mmvq_batch6_input {
    bool gfx1030_native;
    bool rdna2;
    ggml_cuda_mmvq_batch6_type type;
    bool model_hint;
    int64_t n_expert_used;
};

inline bool ggml_cuda_mmvq_mmid_batch6(const ggml_cuda_mmvq_batch6_input & input) {
    const bool supported_type = input.type == ggml_cuda_mmvq_batch6_type::q4_k ||
                                input.type == ggml_cuda_mmvq_batch6_type::q6_k;
    const bool bounded_generic = input.n_expert_used >= 1 && input.n_expert_used <= 4;
    return input.gfx1030_native && input.rdna2 && supported_type &&
        (input.model_hint || bounded_generic);
}