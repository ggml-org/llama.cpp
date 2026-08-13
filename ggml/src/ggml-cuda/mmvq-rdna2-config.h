#pragma once

#include <cstdint>

enum class ggml_cuda_mmvq_rdna2_type {
    other,
    q8_0,
};

struct ggml_cuda_mmvq_rdna2_q8_w8_input {
    ggml_cuda_mmvq_rdna2_type type;
    bool has_ids;
    bool standard_q8_1_layout;
    int64_t ncols_x;
    int64_t nrows_x;
    int64_t ncols_dst;
};

inline bool ggml_cuda_mmvq_use_rdna2_q8_w8(const ggml_cuda_mmvq_rdna2_q8_w8_input & input) {
    return input.type == ggml_cuda_mmvq_rdna2_type::q8_0 &&
           !input.has_ids && input.standard_q8_1_layout &&
           input.ncols_x == 6656 && input.nrows_x == 128 && input.ncols_dst == 1;
}
