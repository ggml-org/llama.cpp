#pragma once

#include "ggml.h"

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
enum common_params_fit_status common_fit_params(
                               const char   * path_model,
                struct llama_model_params   * mparams,
                struct llama_context_params * cparams,
                                      float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
    struct llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                                     size_t * margins,               // margins of memory to leave per device in bytes
                                   uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
                        enum ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// print estimated memory to stdout
void common_fit_print(
                               const char   * path_model,
                struct llama_model_params   * mparams,
                struct llama_context_params * cparams);

void common_memory_breakdown_print(const struct llama_context * ctx);
