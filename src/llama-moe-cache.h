#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// MUL_MAT_ID ops on these experts run on the device and only the cache misses are uploaded
class llama_moe_cache {
public:
    llama_moe_cache(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size);
    ~llama_moe_cache();

    ggml_backend_t backend() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // ggml_backend_sched callbacks, user_data is the llama_moe_cache
    static bool sched_resolve(void * user_data, const ggml_tensor * node, ggml_backend_t backend, ggml_tensor ** cached_weight, void ** cache_entry);
    static void sched_begin(void * user_data);
    static bool sched_prepare(void * user_data, void * cache_entry, const int32_t * ids, size_t n_ids, const int32_t ** remapped_ids);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
