#pragma once

#include "llama-memory-executor.h"
#include "llama-context-workspace.h"

// One serial scheduler lifetime; native caches and arena leases retire before scheduler destruction.
// The caller owns the scheduler/backends and must not submit or mutate their graph caches concurrently.
class llama_context_memory {
public:
    // Unknown native-cache lifetimes keep the existing milestone-3 allocation path.
    static bool supported(const std::vector<ggml_backend_t> & backends);

    // Allocate exactly the existing measured group maxima; leave non-view groups scheduler-owned.
    static std::unique_ptr<llama_context_memory> create(ggml_backend_sched_t sched,
            const std::vector<ggml_backend_t> & backends,
            const std::vector<ggml_backend_memory_workspace_group> & groups);
    ~llama_context_memory();
    llama_context_memory(const llama_context_memory &) = delete;
    llama_context_memory & operator=(const llama_context_memory &) = delete;

    // Reuse one conservative pin for this immutable scheduler lifetime; retirement drains before releasing it.
    ggml_status compute_async(ggml_cgraph * graph);
    // Observe completion without rebuilding the immutable lease-validation state on the next token.
    void synchronize();
    bool uses_arenas() const noexcept;
    // Borrowed handles; consumers retain them before capturing addresses from this workspace.
    const std::vector<ggml_backend_memory_lease_t> & workspace_leases() const noexcept;

private:
    struct implementation;
    explicit llama_context_memory(std::unique_ptr<implementation> impl);
    std::unique_ptr<implementation> impl;
};
