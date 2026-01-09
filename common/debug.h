#pragma once
#include "common.h"

float ggml_get_float_value(const uint8_t * data, enum ggml_type type, const size_t * nb, size_t i0, size_t i1, size_t i2, size_t i3);

#ifdef __cplusplus
#include <string>
#include <vector>
#include <regex>

// common debug functions and structs
std::string ggml_ne_string(const ggml_tensor * t);
template <bool abort> void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n);
template <bool abort> bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data);
struct base_callback_data {
    std::vector<uint8_t>    data;
    std::vector<std::regex> tensor_filters;

    base_callback_data() = default;

    base_callback_data(common_params & params, const std::vector<std::string> & filter_patterns) {
        for (const auto & pattern : filter_patterns) {
            try {
                std::string anchored_pattern = "^" + pattern;
                tensor_filters.emplace_back(anchored_pattern, std::regex::optimize);
            } catch (const std::regex_error & e) {
                throw std::runtime_error("Invalid regex pattern '" + pattern + "': " + e.what());
            }
        }
        params.cb_eval           = ggml_debug<false>;
        params.cb_eval_user_data = this;
    }
};
#endif
