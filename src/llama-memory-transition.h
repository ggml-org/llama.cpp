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

struct llama_memory_transition_arena {
    llama_memory_domain_id domain = 0;
    llama_memory_allocation_class allocation_class = LLAMA_MEMORY_ALLOCATION_HOST;
    ggml_backend_memory_arena_t arena = nullptr;
};

// Borrowed during bind; a consumer must retain any lease it keeps after activation.
struct llama_memory_region_binding {
    size_t arena = 0;
    ggml_backend_memory_region region = {};
    ggml_backend_memory_lease_t lease = nullptr;
};

// Destruction releases temporary state only. Successfully activated ownership stays with the consumer.
// Activation callbacks must not submit new execution; defaults reject unsupported activation protocols.
struct llama_memory_preparation {
    virtual ~llama_memory_preparation() = default;

    // Close affected submission paths, including any consumer-internal executable changes.
    virtual bool quiesce(const std::vector<llama_memory_resource_id> &) { return false; }
    // Complete affected compute/copies before any resource is released.
    virtual bool drain() { return false; }
    // Retire affected native executables while their leased dependencies still exist.
    virtual bool invalidate() { return false; }
    // Drop changed bindings only; exact surviving persistent leases may remain.
    virtual bool release() { return false; }
    // Retain candidate leases without publishing them as active.
    virtual bool bind(const std::vector<llama_memory_region_binding> &) { return false; }
    // Publish already-bound state without submitting execution.
    virtual bool activate() { return false; }
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
    validating,
    quiescing,
    draining,
    invalidating,
    releasing,
    committing,
    binding,
    activating,
    failed,
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
    activated,
    not_prepared,
    invalid_arena,
    activation_failed,
};

struct llama_memory_transition_result {
    llama_memory_transition_status status = llama_memory_transition_status::no_change;
    size_t consumer = std::numeric_limits<size_t>::max();
    llama_memory_layout_result layout;
    std::exception_ptr exception;
    size_t arena = std::numeric_limits<size_t>::max();
    llama_memory_resource_id resource = 0;
    llama_memory_transition_state failed_at = llama_memory_transition_state::idle;
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

    // Activate against exclusive, caller-identified parent arenas; this does not allocate device storage.
    // Preflight rejection keeps the proposal prepared; failures after quiescing remain closed for recovery.
    llama_memory_transition_result activate(const std::vector<llama_memory_transition_arena> & arenas);

    // Defer cancellation inside callbacks. Once activation starts, cancellation leaves the session failed/closed.
    bool cancel() noexcept;

    // Inspect the gate and completed preparation only, never a partially constructed candidate.
    llama_memory_transition_state state() const noexcept;
    const llama_memory_transition_target * pending_target() const noexcept;
    const llama_memory_layout * pending_layout() const noexcept;
    const llama_memory_transition_target * active_target() const noexcept;
    const llama_memory_layout * active_layout() const noexcept;

private:
    struct pending_state;
    void discard_pending() noexcept;
    llama_memory_transition_result activation_error(llama_memory_transition_status status,
            size_t consumer, size_t arena = std::numeric_limits<size_t>::max(),
            llama_memory_resource_id resource = 0, std::exception_ptr exception = {});

    std::vector<llama_memory_consumer *> consumers;
    std::unique_ptr<pending_state> active;
    std::unique_ptr<pending_state> pending;
    llama_memory_transition_state phase = llama_memory_transition_state::idle;
    uint64_t last_admission = 0;
    bool cancellation_requested = false;
};
