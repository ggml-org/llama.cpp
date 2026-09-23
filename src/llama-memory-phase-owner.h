#pragma once

#include "llama-memory-phase.h"
#include "llama-memory-transition.h"

#include <memory>
#include <array>
#include <vector>

struct llama_memory_phase_cache_identity {
    uint64_t id = 0;
    uint64_t generation = 0;
};

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
