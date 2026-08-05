#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>

struct ggml_cuda_mmq_J_setting {
    enum class mode {
        absent,
        heuristic,
        forced,
        invalid,
    };

    mode state;
    int value;
};

inline ggml_cuda_mmq_J_setting ggml_cuda_mmq_parse_J_setting(const char * value) {
    if (value == nullptr) {
        return {ggml_cuda_mmq_J_setting::mode::absent, 0};
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "default") == 0) {
        return {ggml_cuda_mmq_J_setting::mode::heuristic, 0};
    }
    if (value[0] == '\0') {
        return {ggml_cuda_mmq_J_setting::mode::invalid, 0};
    }

    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 8 || parsed > 128 || parsed % 8 != 0) {
        return {ggml_cuda_mmq_J_setting::mode::invalid, 0};
    }
    return {ggml_cuda_mmq_J_setting::mode::forced, int(parsed)};
}

struct ggml_cuda_mmq_auto_J_input {
    bool hint_j16;
    bool rdna2;
    bool q4_k;
    bool routed_ids;
    bool routed_bounds;
    int64_t ncols_x;
    int64_t nrows_x;
    int64_t ncols_dst;
    int64_t nchannels_x;
    int64_t nchannels_y;
    int64_t nsamples_x;
    int64_t nsamples_y;
    int64_t ncols_max;
};

inline int ggml_cuda_mmq_auto_J(const ggml_cuda_mmq_auto_J_input & input) {
    if (!input.hint_j16 || !input.rdna2 || !input.routed_ids || !input.routed_bounds ||
            input.nchannels_x != 256 || input.nchannels_y != 256 ||
            input.nsamples_x != 1 || input.nsamples_y != 1 || input.ncols_max <= 0 ||
            input.ncols_dst <= 0 || input.ncols_dst % input.ncols_max != 0) {
        return 0;
    }

    const int64_t top_k = input.ncols_dst / input.ncols_max;
    const bool qwen35_122b_q4_k = input.q4_k && input.ncols_x == 3072 && input.nrows_x == 256 &&
        input.ncols_max == 256 && top_k == 8;
    const bool qwen36_35b_q4_k = input.q4_k && input.ncols_x == 2048 &&
        (input.nrows_x == 512 || input.nrows_x == 128) && input.ncols_max == 256 && top_k == 8;
    const bool deepseek_v4 = top_k == 6 &&
        ((input.ncols_x == 4096 && input.nrows_x == 512) ||
         (input.ncols_x == 2048 && input.nrows_x == 1024));
    return qwen35_122b_q4_k || qwen36_35b_q4_k || deepseek_v4 ? 16 : 0;
}
