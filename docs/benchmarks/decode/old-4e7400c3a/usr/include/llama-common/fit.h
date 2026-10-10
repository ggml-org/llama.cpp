#pragma once

#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct common_moe_cache_params;

const char * common_moe_cache_tensor_override_pattern();

struct common_moe_cache_fit_device_input {
    int physical_device = -1;
    int compute_capability = 0;
    int64_t free_bytes = 0;
    size_t used_bytes = 0;
};

struct common_moe_cache_fit_shape_input {
    enum ggml_type type = GGML_TYPE_COUNT;
    size_t expert_size = 0;
    size_t tensor_bytes = 0;
    size_t scratch_bytes = 0;
    size_t pool_bytes = 0;
    bool cacheable = false;
};

struct common_moe_cache_fit_device {
    int physical_device = -1;
    int compute_capability = 0;
    int64_t free_bytes = 0;
    int64_t used_bytes = 0;
    size_t cache_bytes = 0;
};

struct common_moe_cache_fit_result {
    bool feasible = false;
    std::string reason;
    std::vector<common_moe_cache_fit_device> devices;
    size_t expert_bytes = 0;
    size_t cache_bytes = 0;
    size_t minimum_device_bytes = 0;
};

common_moe_cache_fit_result common_moe_cache_plan_fit(
        const std::vector<common_moe_cache_fit_device_input> & devices,
        const std::vector<common_moe_cache_fit_shape_input> & shapes,
        size_t reserve_bytes,
        size_t budget_bytes,
        int min_devices,
        size_t minimum_slab_bytes = 0);

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
    const char * path_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
           common_moe_cache_params * moe_cache,
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);

// Load a parent model + context with no_alloc, then return the per-device memory breakdown of a child model + context.
common_device_memory_data_vec common_get_device_memory_data_with_parent(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
                         const char * path_parent,
           const llama_model_params * mparams_parent,
         const llama_context_params * cparams_parent,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
