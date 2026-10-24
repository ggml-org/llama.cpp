#pragma once

#include "ggml.h"
#include "llama.h"

#include <vector>

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

// one row of the memory breakdown of a context: a device, the host, or a buffer type not tied to a device
struct common_memory_breakdown_row {
    enum kind_t { DEVICE, HOST, BUFFER_TYPE };

    kind_t      kind = DEVICE;
    std::string name;        // device or buffer type name, "Host" for the host row
    std::string description; // device description, empty otherwise
    int64_t     total = 0;   // device memory in bytes, 0 unless kind == DEVICE
    int64_t     free  = 0;   // free device memory in bytes, 0 unless kind == DEVICE
    size_t      model   = 0; // bytes held by this context, by kind of buffer
    size_t      context = 0;
    size_t      compute = 0;
};

// memory that a model needs at the two ends of what --fit may pick, see tools/server/README.md
struct common_memory_reports {
    uint32_t n_ctx_train = 0;
    uint32_t n_ctx_max   = 0; // the largest context that fits the free device memory right now
    uint32_t n_ctx_min   = 0; // --fit-ctx (the smallest context --fit may pick), with the placement set in params
    std::vector<common_memory_breakdown_row> info_max; // breakdown at n_ctx_max
    std::vector<common_memory_breakdown_row> info_min; // breakdown at n_ctx_min
};

// same numbers as common_memory_breakdown_print, the devices of the model first, then the host
std::vector<common_memory_breakdown_row> common_memory_breakdown_rows(const llama_context * ctx);

// load the model and create a context with no_alloc, return their memory breakdown
//   - the parameters are used as they are, use common_fit_params first to fit them to free memory
//   - n_ctx_train and n_ctx are optional outputs
//   - throws std::runtime_error if the model or the context cannot be created
std::vector<common_memory_breakdown_row> common_fit_measure(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
                           uint32_t * n_ctx_train,
                           uint32_t * n_ctx,
                     ggml_log_level   log_level);

struct common_params;

// measure without allocating; params may be modified the same way common_init_result does with --fit
// throws std::runtime_error on failure
common_memory_reports common_measure_memory(common_params & params);

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
