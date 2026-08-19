#pragma once

#include <cstdint>

enum class ggml_cuda_mmvq_rdna2_type {
    other,
    q4_0,
    q4_k,
    q6_k,
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

struct ggml_cuda_mmvq_rdna2_w8_rows2_input {
    ggml_cuda_mmvq_rdna2_type type;
    bool enabled;
    bool has_ids;
    bool standard_q8_1_layout;
    int64_t ncols_x;
    int64_t nrows_x;
    int64_t ncols_dst;
};

inline bool ggml_cuda_mmvq_use_rdna2_w8_rows2(const ggml_cuda_mmvq_rdna2_w8_rows2_input & input) {
    if (!input.enabled || input.has_ids || !input.standard_q8_1_layout ||
            input.ncols_x <= 0 || input.ncols_x % 32 != 0 || input.ncols_dst != 8) {
        return false;
    }

    switch (input.type) {
        case ggml_cuda_mmvq_rdna2_type::q4_0:
            return (input.ncols_x == 5120 &&
                       (input.nrows_x == 12   || input.nrows_x == 256  || input.nrows_x == 1536 ||
                        input.nrows_x == 2560 || input.nrows_x == 3072 || input.nrows_x == 4352)) ||
                   (input.nrows_x == 5120 &&
                       (input.ncols_x == 1536 || input.ncols_x == 4352));
        case ggml_cuda_mmvq_rdna2_type::q4_k:
            return (input.ncols_x == 5120 &&
                       (input.nrows_x == 256  || input.nrows_x == 1024 || input.nrows_x == 1280 ||
                        input.nrows_x == 4096 || input.nrows_x == 17408)) ||
                   (input.nrows_x == 5120 &&
                       (input.ncols_x == 4096 || input.ncols_x == 17408 || input.ncols_x == 25600));
        case ggml_cuda_mmvq_rdna2_type::q6_k:
            return (input.ncols_x == 5120 &&
                       (input.nrows_x == 1024 || input.nrows_x == 248320)) ||
                   (input.ncols_x == 17408 && input.nrows_x == 5120);
        default:
            return false;
    }
}
