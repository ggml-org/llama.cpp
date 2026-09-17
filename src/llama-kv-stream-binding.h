#pragma once

#include "llama-kv-stream-policy.h"
#include "llama-memory-executor.h"

#include <functional>

// Resolve the explicit device-local CUDA type without linking llama to CUDA. No default-type fallback.
ggml_backend_buffer_type_t llama_kv_stream_device_buffer_type(ggml_backend_dev_t device);

struct llama_kv_stream_binding_view {
    uint64_t cache_id = 0, revision = 0;
    ggml_backend_buffer_t buffer = nullptr;
    void * base = nullptr;
    size_t capacity = 0;
    llama_kv_stream_policy_config config;
    llama_kv_stream_policy_state initial_policy;
};

// Build idle device resources from validated metadata; copy any fields needed after this call.
// The executable owns its host-cache dependencies, but must not free the borrowed device region.
// Construction/failure cleanup must not submit work; destruction retires all device references without throwing.
using llama_kv_stream_binding_factory = std::function<std::unique_ptr<llama_memory_executable>(const llama_kv_stream_binding_view &)>;

// Owner-thread-only coarse binding. Cache IDs are nonzero and session-unique, independent of arena generations.
// expected_type is a trusted allocation contract: production CUDA callers use the resolver above.
// Buffer type/backend and queued execution pins must obey the common executor lifetime contract.
class llama_kv_stream_binding {
public:
    llama_kv_stream_binding(uint64_t cache_id, ggml_backend_buffer_type_t expected_type,
            uint64_t previous_revision = 0);
    // No implicit drain: queued users must retain their pins until backend completion.
    ~llama_kv_stream_binding() = default;
    llama_kv_stream_binding(const llama_kv_stream_binding &) = delete;
    llama_kv_stream_binding & operator=(const llama_kv_stream_binding &) = delete;

    // Retain one lease, validate the requested pool, construct idle resources, then publish atomically.
    // Null/invalid grants or factory failure leave this binding unchanged. Non-allocation exceptions propagate.
    bool bind(ggml_backend_memory_lease_t lease, const llama_kv_stream_policy_config & config,
            const llama_kv_stream_binding_factory & factory);

    // Cache metadata is borrowed only while ready; copy it into any longer-lived device resources.
    const llama_kv_stream_binding_view * view() const noexcept;
    uint64_t cache_id() const noexcept;
    bool ready() const noexcept;

    // One pin can cover a queued sequence. No allocation/base lookup or arena transaction occurs here.
    llama_memory_execution acquire() const;
    void quiesce() noexcept;

    // Drain all users, destroy native resources, then release the region. Failures retain it and stay closed.
    llama_memory_executor_result detach(llama_memory_executor_backend & backend);

private:
    uint64_t identity;
    uint64_t revision = 0;
    ggml_backend_buffer_type_t expected_type;
    llama_kv_stream_binding_view current;
    std::vector<ggml_backend_memory_lease_t> dependencies;
    bool busy = false;
    // The executor retains the actual lease and retires native resources before releasing it.
    llama_memory_executor executor;
};
