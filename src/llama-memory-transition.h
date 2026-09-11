#pragma once

#include "llama-memory-layout.h"

#include <exception>
#include <memory>

// Owned snapshot for one proposed transition. Preparation does not execute the stage.
struct llama_memory_transition_target {
    llama_memory_execution_plan plan;
    llama_memory_stage_id stage = 0;
    std::vector<llama_memory_arena_budget> budgets;
    std::vector<llama_memory_fixed_region> fixed;
};

// Destroying a preparation discards only temporary state, never an active binding.
// Destructors must not throw; activation and ownership transfer are added separately.
struct llama_memory_preparation {
    virtual ~llama_memory_preparation() = default;
};

struct llama_memory_consumer {
    virtual ~llama_memory_consumer() = default;

    // Prepare without changing active state or submitting device work.
    // True plus a null output means no transition is needed for this consumer, including its external state.
    // False or an exception discards every returned preparation, including this consumer's partial output.
    // Target/layout references remain valid until the returned preparation is destroyed.
    virtual bool prepare(
            const llama_memory_transition_target & target,
            const llama_memory_layout & layout,
            std::unique_ptr<llama_memory_preparation> & output) = 0;
};

enum class llama_memory_transition_state {
    idle,
    executing,
    preparing,
    prepared,
    discarding,
};

enum class llama_memory_transition_status {
    prepared,
    no_change,
    busy,
    invalid_consumer,
    layout_error,
    consumer_failed,
    consumer_exception,
    cancelled,
    allocation_failed,
};

struct llama_memory_transition_result {
    llama_memory_transition_status status = llama_memory_transition_status::no_change;
    size_t consumer = std::numeric_limits<size_t>::max();
    llama_memory_layout_result layout;
    std::exception_ptr exception;
};

// Owner-thread-only gate. Callbacks may reenter admission, prepare, and cancel, but must not destroy this object.
// Consumers are borrowed and must outlive this object and all pending preparations.
// The caller registers the complete participant list; no-op decisions are not inferred from layout equality.
class llama_memory_transition {
public:
    explicit llama_memory_transition(std::vector<llama_memory_consumer *> consumers);
    ~llama_memory_transition();

    llama_memory_transition(const llama_memory_transition &) = delete;
    llama_memory_transition & operator=(const llama_memory_transition &) = delete;

    // Admit one host operation, returning a nonzero completion ID, or zero while admission is closed.
    uint64_t admit() noexcept;

    // Finish only the matching host operation. This does not imply that device work has completed.
    bool finish(uint64_t admission) noexcept;

    // Calculate an owned candidate and prepare consumers in registration order; do not activate or rebind.
    llama_memory_transition_result prepare(const llama_memory_transition_target & target);

    // Defer cancellation inside prepare; otherwise discard pending state immediately in reverse order.
    bool cancel() noexcept;

    // Inspect the gate and completed preparation only, never a partially constructed candidate.
    llama_memory_transition_state state() const noexcept;
    const llama_memory_transition_target * pending_target() const noexcept;
    const llama_memory_layout * pending_layout() const noexcept;

private:
    struct pending_state;
    void discard_pending() noexcept;

    std::vector<llama_memory_consumer *> consumers;
    std::unique_ptr<pending_state> pending;
    llama_memory_transition_state phase = llama_memory_transition_state::idle;
    uint64_t last_admission = 0;
    bool cancellation_requested = false;
};
