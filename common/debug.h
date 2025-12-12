#pragma once
#include "llama.h"

float ggml_get_float_value(const uint8_t * data, enum ggml_type type, const size_t * nb, size_t i0, size_t i1, size_t i2, size_t i3);

#ifdef __cplusplus
#include <string>
#include <vector>

// common debug functions and structs
struct base_callback_data {
    std::vector<uint8_t> data;
};

std::string ggml_ne_string(const ggml_tensor * t);
template <bool abort> void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n);
template <bool abort> bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data);
#endif

