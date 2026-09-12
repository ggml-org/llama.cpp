#pragma once

#include "llama-memory-transition.h"

#include <functional>

struct llama_memory_workspace_group {
    ggml_backend_memory_workspace_group workspace;
    llama_memory_resource resource;
};

// Hooks cover all executable graphs using this scheduler, including fallback groups.
// They must be idempotent; invalidate retires captures and marks tensor graph bindings for reconstruction.
struct llama_memory_workspace_hooks {
    std::function<bool()> quiesce;
    std::function<bool()> invalidate;
};

// Owner-thread-only consumer for selected view-capable scheduler groups.
// Scheduler, backends, buffer types, and hooks must outlive this object and its preparations.
// Omitted groups remain scheduler-owned. Callers must gate execution through the transition coordinator.
class llama_memory_workspace : public llama_memory_consumer {
public:
    llama_memory_workspace(ggml_backend_sched_t sched,
            std::vector<llama_memory_workspace_group> groups, llama_memory_workspace_hooks hooks);
    ~llama_memory_workspace() override;
    llama_memory_workspace(const llama_memory_workspace &) = delete;
    llama_memory_workspace & operator=(const llama_memory_workspace &) = delete;

    // Append one resource per buffer-type group and the same measured maximum to each named stage.
    // No backend storage is allocated; failure leaves the plan unchanged.
    bool register_resources(llama_memory_execution_plan & plan, const std::vector<llama_memory_stage_id> & stages) const;

    // Validate exact maximum-workspace grants and stage lease attachment without touching live bindings.
    bool prepare(const llama_memory_transition_target & target, const llama_memory_layout & layout,
            std::unique_ptr<llama_memory_preparation> & output) override;

    // Drain, invalidate, and detach only owned leases. Failure retains remaining ownership and keeps this consumer closed.
    bool close();
    bool ready() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> impl;
};
