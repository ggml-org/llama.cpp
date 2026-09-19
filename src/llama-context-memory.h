#pragma once

#include "llama-memory-executor.h"
#include "llama-memory-phase.h"
#include "llama-context-workspace.h"

class llama_kv_stream_model;

struct llama_context_memory_diagnostics {
    llama_memory_text_phase phase = llama_memory_text_phase::unspecified;
    size_t parent_bytes = 0, workspace_bytes = 0, kv_pool_bytes = 0;
    size_t kv_writer_bytes = 0, kv_attention_bytes = 0, unused_bytes = 0;
    size_t reclaimed_workspace_bytes = 0;
    uint64_t arena_generation = 0, transition_count = 0, layout_revision = 0;
    uint64_t last_transition_us = 0;
    uint32_t resident_pages_per_layer = 0, ring_slots = 0, active_pages = 0;
    size_t last_copy_bytes = 0, last_copy_calls = 0;
    double last_copy_ms = 0, last_elapsed_ms = 0;
    bool streaming_active = false;
    bool executable_storage_external = true;
};

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
    static std::unique_ptr<llama_context_memory> create(ggml_backend_sched_t sched,
            const std::vector<ggml_backend_t> & backends,
            const llama_compute_workspace_plan & plan);
    static std::unique_ptr<llama_context_memory> create(ggml_backend_sched_t sched,
            const std::vector<ggml_backend_t> & backends,
            const llama_compute_workspace_plan & plan, llama_kv_stream_model * stream);
    ~llama_context_memory();
    llama_context_memory(const llama_context_memory &) = delete;
    llama_context_memory & operator=(const llama_context_memory &) = delete;

    // Reuse one conservative pin for this immutable scheduler lifetime; retirement drains before releasing it.
    ggml_status compute_async(ggml_cgraph * graph);
    // Observe completion without rebuilding the immutable lease-validation state on the next token.
    void synchronize();

    // Record text intent and activate the matching shared-parent layout only when the phase changes.
    llama_memory_text_phase_result signal_text_phase(const llama_memory_text_phase_signal & signal) noexcept;
    llama_memory_text_phase_snapshot text_phase() const noexcept;
    bool uses_arenas() const noexcept;
    // Borrowed handles; consumers retain them before capturing addresses from this workspace.
    const std::vector<ggml_backend_memory_lease_t> & workspace_leases() const noexcept;
    bool shares_kv_memory() const noexcept;
    ggml_backend_buffer_t shared_parent() const noexcept;
    size_t shared_parent_capacity() const noexcept;
    uint64_t shared_arena_generation() const noexcept;
    uint64_t phase_transition_count() const noexcept;
    bool diagnostics(llama_context_memory_diagnostics & output) const noexcept;

private:
    struct implementation;
    explicit llama_context_memory(std::unique_ptr<implementation> impl);
    std::unique_ptr<implementation> impl;
};
