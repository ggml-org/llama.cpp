#pragma once

#include "llama-memory-phase.h"
#include "llama-memory-transition.h"

#include <memory>
#include <array>
#include <vector>

struct llama_memory_phase_target {
    llama_memory_work_phase phase = llama_memory_work_phase::none;
    llama_memory_transition_target target;
    uint32_t max_tokens = UINT32_MAX;
};
// Byte measurements exclude persistent weights and KV. Extra staging must not be included in graph_bytes.
struct llama_memory_phase_workspace_measurement {
    std::array<size_t,4> target_verify_graph_bytes{};
    std::array<size_t,4> mtp_catchup_graph_bytes{};
    size_t target_prefill_graph_bytes = 0;
    size_t mtp_draft_graph_bytes = 0;
    size_t rollback_stage_bytes = 0;
    size_t mtp_stage_bytes = 0;
    size_t alignment = 1;
};

struct llama_memory_phase_workspace_budget {
    size_t target_prefill_bytes = 0;
    size_t target_verify_bytes = 0;
    size_t mtp_catchup_bytes = 0;
    size_t mtp_draft_bytes = 0;
    size_t shared_parent_bytes = 0;
    size_t reclaimed_during_mtp_catchup_bytes = 0;
    size_t reclaimed_during_mtp_draft_bytes = 0;
};

// Plan one backend domain. Serial phases share the peak grant, not their sum.
bool llama_memory_phase_workspace_budget_make(const llama_memory_phase_workspace_measurement & measurement,
        llama_memory_phase_workspace_budget & output) noexcept;

// One measured backend domain. Graph bytes exclude the separately live device/output bytes.
// Availability refers to this exact workload. Fallback fields describe the selected fallback, not the primary path.
struct llama_memory_workload_measurement {
    llama_memory_workload_requirement workload;
    size_t graph_bytes = 0;
    size_t live_device_bytes = 0;
    size_t host_feature_bytes = 0;
    size_t alignment = 1;
    bool execution_available = false;
    bool fallback_selected = false;
    bool fallback_measured = false;
};

// Include backend tensor/allocation padding, not just encoded payload bytes.
struct llama_memory_cache_allocation_measurement {
    llama_memory_phase_cache_identity cache;
    size_t bytes = 0;
    size_t alignment = 1;
    uint32_t capacity_rows = 0;
    llama_memory_cache_format format;
};

enum class llama_memory_workload_budget_status {
    success,
    invalid_cache,
    missing_cache,
    stale_cache,
    invalid_workload,
    missing_measurement,
    missing_kernel,
    unbudgeted_fallback,
    invalid_measurement,
    insufficient_budget,
    allocation_failed,
};

struct llama_memory_workload_budget_result {
    llama_memory_workload_budget_status status = llama_memory_workload_budget_status::success;
    size_t workload = 0;
    size_t additional_bytes = 0;
};

struct llama_memory_workload_budget {
    size_t persistent_cache_bytes = 0;
    size_t workspace_peak_bytes = 0;
    size_t host_feature_peak_bytes = 0;
    size_t parent_minimum_bytes = 0;
    std::vector<size_t> workload_bytes;
};

// Check the complete declared workload catalog before taking the serial peak.
// Bounded resident caches stay live. Full-history backing/aliases add no implicit device storage;
// quotes must include their live device grant, staging and retained outputs. No backend probing or leases here.
// persistent_extra_bytes is an already planned prefix; cache and workspace placements follow it with alignment.
llama_memory_workload_budget_result llama_memory_workload_budget_make(
        const std::vector<llama_memory_cache_requirement> & caches,
        const std::vector<llama_memory_cache_allocation_measurement> & cache_allocations,
        const std::vector<llama_memory_workload_requirement> & workloads,
        const std::vector<llama_memory_workload_measurement> & measurements,
        size_t persistent_extra_bytes, size_t capacity, llama_memory_workload_budget & output) noexcept;

enum class llama_memory_phase_owner_status {
    changed,
    unchanged,
    invalid_phase,
    invalid_signal,
    stale_identity,
    busy,
    transition_failed,
    session_invalid,
};

struct llama_memory_phase_owner_result {
    llama_memory_phase_owner_status status = llama_memory_phase_owner_status::invalid_signal;
    llama_memory_transition_result transition;
};

// The caller owns arenas and consumers for the lifetime of this serial owner.
class llama_memory_phase_owner {
public:
    static std::unique_ptr<llama_memory_phase_owner> create(
            std::vector<llama_memory_phase_target> targets,
            std::vector<llama_memory_transition_arena> arenas,
            std::vector<llama_memory_consumer *> consumers,
            llama_memory_phase_cache_identity target,
            llama_memory_phase_cache_identity mtp);
    ~llama_memory_phase_owner();
    llama_memory_phase_owner(const llama_memory_phase_owner &) = delete;
    llama_memory_phase_owner & operator=(const llama_memory_phase_owner &) = delete;

    llama_memory_phase_owner_result enter(llama_memory_work_phase phase, uint32_t tokens,
            llama_memory_phase_cache_identity target, llama_memory_phase_cache_identity mtp);
    uint64_t admit() noexcept;
    bool finish(uint64_t admission, llama_memory_phase_cache_identity target,
            llama_memory_phase_cache_identity mtp) noexcept;
    llama_memory_work_phase_snapshot snapshot() const noexcept;
    llama_memory_phase_cache_identity target_identity() const noexcept;
    llama_memory_phase_cache_identity mtp_identity() const noexcept;
private:
    struct implementation;
    explicit llama_memory_phase_owner(std::unique_ptr<implementation> impl);
    std::unique_ptr<implementation> impl;
};
