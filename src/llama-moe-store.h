#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// device cache for MoE experts that live in host memory
// small batches read the experts from cache slots through remapped ids
// big batches use the expert copy of the scheduler, cached experts are copied on the device
class llama_moe_store {
public:
    // n_cache_layers: layer-equivalents to cache, < 0 = auto (size from budget_bytes)
    // budget_bytes:   device memory to spend, used only for auto
    llama_moe_store(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
            int32_t n_cache_layers, size_t budget_bytes);
    ~llama_moe_store();

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // run the small batches of the cached experts on the store backend
    void attach(ggml_backend_sched_t sched);

    // call before each graph compute
    void begin();

    // fill dst, the copy of the host experts src on backend, with the experts set in used
    // returns false if the store does not manage src
    bool copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<bool> & used);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
