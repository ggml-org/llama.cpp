#pragma once

#include <memory>
#include <string>
#include <vector>

// common debug functions and structs

struct common_params;

// Intended to use as callback for ggml_backend_sched_eval_callback
// prints tensors that are processed in the computation graph
// by default prints all tensors, but can be configured by creating a `common_debug_cb_user_data` instance with
// non-empty filter_patterns. See examples/debug.cpp for possible usage patterns
// `common_debug_cb_user_data` contains `abort_on_nan` flag that determines whether an error should be thrown whenever a NaN is encountered
// in a tensor (useful for stopping debug sessions on first erroneous tensor)
// The callback data will be passed as the third parameter (user_data)
bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

struct common_debug_cb_user_data {
    struct impl;
    std::unique_ptr<impl> pimpl;

    common_debug_cb_user_data();
    ~common_debug_cb_user_data();

    common_debug_cb_user_data(const common_debug_cb_user_data &) = delete;
    common_debug_cb_user_data & operator=(const common_debug_cb_user_data &) = delete;

    common_debug_cb_user_data(common_params & params, const std::vector<std::string> & filter_patterns, bool abort_on_nan = false);
};
