#pragma once
#include "common.h"
#include <string>
#include <vector>
#include <regex>

// common debug functions and structs

// Print a tensor's detailed data
// data - the tensor's data in byte format
// type - the tensor's quantization type
// ne   - the tensor dimensions array
// nb   - the tensor strides array
// n    - the number of rows/columns to fully print
template <bool abort_on_nan> void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n);

// Intended to use as callback for ggml_backend_sched_eval_callback
// prints tensors that are processed in the computation graph
// by default prints all tensors, but can be configured by creating a `base_callback_data` instance with
// non-empty filter_patterns. See examples/debug.ccp for possible usage patterns
// The template parameter determins whether an error should be thrown whenever a NaN is encountered
// in a tensor (useful for stopping debug sessions on first erroneous tensor)
// The callback data will be passed as the third parameter (user_data)
template <bool abort_on_nan> bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);
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
        params.cb_eval           = common_debug_cb_eval<false>;
        params.cb_eval_user_data = this;
    }
};
