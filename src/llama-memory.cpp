#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-cache-iswa.h"

// Resolve storage roles without assuming a hybrid memory layout or changing ownership.
llama_memory_kv_resources llama_memory_kv_resources_get(llama_memory_t memory) noexcept {
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        return {hybrid->get_mem_attn(), nullptr, hybrid->get_mem_recr()};
    }
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
        return {iswa->get_base(), iswa->get_swa(), nullptr};
    }
    if (auto * cache = dynamic_cast<llama_kv_cache *>(memory)) {
        return {cache, nullptr, nullptr};
    }
    return {};
}

// Resolve the full-cache microbatch alongside the matching storage adapter.
const llama_kv_cache_context * llama_memory_kv_context_get(const llama_memory_context_i * context) noexcept {
    if (const auto * hybrid = dynamic_cast<const llama_memory_hybrid_context *>(context)) {
        return hybrid->get_attn();
    }
    if (const auto * iswa = dynamic_cast<const llama_kv_cache_iswa_context *>(context)) {
        return iswa->get_base();
    }
    return dynamic_cast<const llama_kv_cache_context *>(context);
}

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}
