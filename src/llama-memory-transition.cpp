#include "llama-memory-transition.h"

#include <new>
#include <utility>

static constexpr size_t no_consumer = std::numeric_limits<size_t>::max();

struct llama_memory_transition::pending_state {
    llama_memory_transition_target target;
    llama_memory_layout layout;
    std::vector<std::unique_ptr<llama_memory_preparation>> preparations;

    // Keep callback metadata independent of the caller's request lifetime.
    explicit pending_state(const llama_memory_transition_target & target) : target(target) {}

    // Discard dependents before their prerequisites, while all snapshot metadata remains alive.
    ~pending_state() {
        for (auto it = preparations.rbegin(); it != preparations.rend(); ++it) {
            it->reset();
        }
    }
};

// Retain the registration list, not ownership of the consumer objects.
llama_memory_transition::llama_memory_transition(std::vector<llama_memory_consumer *> consumers) :
    consumers(std::move(consumers)) {}

// A pending proposal owns only preparatory state; active bindings remain with their consumers.
llama_memory_transition::~llama_memory_transition() {
    discard_pending();
}

// Issue a unique host-operation ID without reopening a closed gate or wrapping the counter.
uint64_t llama_memory_transition::admit() noexcept {
    if (phase != llama_memory_transition_state::idle ||
            last_admission == std::numeric_limits<uint64_t>::max()) {
        return 0;
    }
    phase = llama_memory_transition_state::executing;
    return ++last_admission;
}

// Ignore duplicate/stale completions and completions received during a transition.
bool llama_memory_transition::finish(uint64_t admission) noexcept {
    if (phase != llama_memory_transition_state::executing || admission == 0 || admission != last_admission) {
        return false;
    }
    phase = llama_memory_transition_state::idle;
    return true;
}

// Keep the gate closed while arbitrary consumer cleanup callbacks run.
void llama_memory_transition::discard_pending() noexcept {
    phase = llama_memory_transition_state::discarding;
    pending.reset();
    cancellation_requested = false;
    phase = llama_memory_transition_state::idle;
}

// Prepare a candidate without publishing it as active or altering any live arena.
llama_memory_transition_result llama_memory_transition::prepare(const llama_memory_transition_target & target) {
    using status = llama_memory_transition_status;
    if (phase != llama_memory_transition_state::idle) {
        return {status::busy, no_consumer, {}, {}};
    }
    if (consumers.empty()) {
        return {status::invalid_consumer, no_consumer, {}, {}};
    }
    for (size_t i = 0; i < consumers.size(); ++i) {
        if (consumers[i] == nullptr) {
            return {status::invalid_consumer, i, {}, {}};
        }
        for (size_t j = 0; j < i; ++j) {
            if (consumers[j] == consumers[i]) {
                return {status::invalid_consumer, i, {}, {}};
            }
        }
    }

    phase = llama_memory_transition_state::preparing;
    cancellation_requested = false;
    size_t current_consumer = no_consumer;
    try {
        pending = std::make_unique<pending_state>(target);
        const auto & snapshot = pending->target;
        const auto layout = llama_memory_layout_elastic(
            snapshot.plan, snapshot.stage, snapshot.budgets, snapshot.fixed, pending->layout);
        if (layout.status != llama_memory_layout_status::success) {
            discard_pending();
            return {status::layout_error, no_consumer, layout, {}};
        }
        pending->preparations.resize(consumers.size());
        if (cancellation_requested) {
            discard_pending();
            return {status::cancelled, no_consumer, {}, {}};
        }

        bool changed = false;
        for (size_t i = 0; i < consumers.size(); ++i) {
            current_consumer = i;
            const bool ready = consumers[i]->prepare(snapshot, pending->layout, pending->preparations[i]);
            if (cancellation_requested) {
                discard_pending();
                return {status::cancelled, i, {}, {}};
            }
            if (!ready) {
                discard_pending();
                return {status::consumer_failed, i, {}, {}};
            }
            changed = changed || pending->preparations[i] != nullptr;
        }
        if (!changed) {
            discard_pending();
            return {status::no_change, no_consumer, {}, {}};
        }
        phase = llama_memory_transition_state::prepared;
        return {status::prepared, no_consumer, {}, {}};
    } catch (const std::bad_alloc &) {
        const auto exception = std::current_exception();
        discard_pending();
        return {status::allocation_failed, current_consumer, {}, exception};
    } catch (...) {
        const auto exception = std::current_exception();
        discard_pending();
        return {status::consumer_exception, current_consumer, {}, exception};
    }
}

// Never destroy metadata or a partial output underneath an executing prepare callback.
bool llama_memory_transition::cancel() noexcept {
    if (phase == llama_memory_transition_state::preparing) {
        cancellation_requested = true;
        return true;
    }
    if (phase == llama_memory_transition_state::prepared) {
        discard_pending();
        return true;
    }
    return false;
}

// Report logical admission state, not backend completion.
llama_memory_transition_state llama_memory_transition::state() const noexcept {
    return phase;
}

// Expose only fully prepared snapshots.
const llama_memory_transition_target * llama_memory_transition::pending_target() const noexcept {
    return phase == llama_memory_transition_state::prepared ? &pending->target : nullptr;
}

// Expose only fully prepared layouts; cancellation invalidates this borrowed pointer.
const llama_memory_layout * llama_memory_transition::pending_layout() const noexcept {
    return phase == llama_memory_transition_state::prepared ? &pending->layout : nullptr;
}
