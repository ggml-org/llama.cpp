#include "llama-memory-transition.h"

#include <algorithm>
#include <new>
#include <utility>

static constexpr size_t no_consumer = std::numeric_limits<size_t>::max();

using transition_arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using transition_lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

struct llama_memory_transition::pending_state {
    struct arena_record {
        transition_arena_ptr arena{nullptr, ggml_backend_memory_arena_free};
        std::vector<ggml_backend_memory_region> before;
        uint64_t generation = 0;
        bool changed = false;
        bool planning = false;
    };
    llama_memory_transition_target target;
    llama_memory_layout layout;
    std::vector<arena_record> arenas;
    std::vector<transition_lease_ptr> leases;
    std::vector<llama_memory_region_binding> bindings;
    std::vector<llama_memory_resource_id> changed_resources;
    std::vector<std::unique_ptr<llama_memory_preparation>> preparations;

    // Keep callback metadata independent of the caller's request lifetime.
    explicit pending_state(const llama_memory_transition_target & target) : target(target) {}

    // Discard dependents before their prerequisites, while all snapshot metadata remains alive.
    ~pending_state() { clear_preparations(); }

    // Keep snapshots, arenas, and staged leases alive throughout consumer cleanup.
    void clear_preparations() noexcept {
        for (auto it = preparations.rbegin(); it != preparations.rend(); ++it) {
            it->reset();
        }
        preparations.clear();
    }
};

// Retain the registration list, not ownership of the consumer objects.
llama_memory_transition::llama_memory_transition(std::vector<llama_memory_consumer *> consumers) :
    consumers(std::move(consumers)) {}

// A pending proposal owns only preparatory state; active bindings remain with their consumers.
llama_memory_transition::~llama_memory_transition() {
    discard_pending();
    phase = llama_memory_transition_state::discarding;
    active.reset();
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
    if (phase == llama_memory_transition_state::preparing ||
            phase == llama_memory_transition_state::validating ||
            (phase >= llama_memory_transition_state::quiescing && phase <= llama_memory_transition_state::activating)) {
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
    return phase == llama_memory_transition_state::prepared || phase == llama_memory_transition_state::failed ? &pending->target : nullptr;
}

// Expose only fully prepared layouts; cancellation invalidates this borrowed pointer.
const llama_memory_layout * llama_memory_transition::pending_layout() const noexcept {
    return phase == llama_memory_transition_state::prepared || phase == llama_memory_transition_state::failed ? &pending->layout : nullptr;
}

// Compare complete region identity; matching sizes alone do not preserve a view.
static bool same_transition_region(const ggml_backend_memory_region & a, const ggml_backend_memory_region & b) {
    return a.id == b.id && a.offset == b.offset && a.size == b.size && a.alignment == b.alignment && a.flags == b.flags;
}

// Find a region without borrowing a view whose lifetime may change during commit.
static const ggml_backend_memory_region * find_transition_region(
        const std::vector<ggml_backend_memory_region> & regions, uint64_t id) {
    for (const auto & region : regions) {
        if (region.id == id) return &region;
    }
    return nullptr;
}

// Preserve failure evidence and close every touched arena; recovery is a separate operation.
llama_memory_transition_result llama_memory_transition::activation_error(
        llama_memory_transition_status status, size_t consumer, size_t arena,
        llama_memory_resource_id resource, std::exception_ptr exception) {
    llama_memory_transition_result result{status, consumer, {}, std::move(exception)};
    result.arena = arena;
    result.resource = resource;
    result.failed_at = phase;
    phase = llama_memory_transition_state::failed;
    for (auto & record : pending->arenas) {
        if (record.arena) {
            if (record.planning) {
                ggml_backend_memory_arena_rollback(record.arena.get());
                record.planning = false;
            }
            ggml_backend_memory_arena_quiesce(record.arena.get());
        }
    }
    return result;
}

// Commit only after all affected work is drained and all changed bindings have been released.
llama_memory_transition_result llama_memory_transition::activate(
        const std::vector<llama_memory_transition_arena> & supplied) {
    using status = llama_memory_transition_status;
    using step = llama_memory_transition_state;
    if (phase != step::prepared) {
        return {phase == step::idle ? status::not_prepared : status::busy, no_consumer, {}, {}};
    }

    phase = step::validating;
    cancellation_requested = false;
    bool started = false;
    size_t current_consumer = no_consumer;
    size_t current_arena = no_consumer;
    uint64_t current_resource = 0;
    auto reject = [&](status code, size_t arena = no_consumer, uint64_t resource = 0, std::exception_ptr exception = {}) {
        phase = step::prepared;
        cancellation_requested = false;
        llama_memory_transition_result result{code, no_consumer, {}, std::move(exception)};
        result.arena = arena;
        result.resource = resource;
        result.failed_at = step::validating;
        return result;
    };

    try {
        if (supplied.size() != pending->layout.arenas.size() ||
                (active && supplied.size() != active->arenas.size())) {
            return reject(status::invalid_arena);
        }
        std::vector<pending_state::arena_record> records(supplied.size());
        std::vector<llama_memory_resource_id> changed;
        for (size_t i = 0; i < supplied.size(); ++i) {
            current_arena = i;
            const auto & input = supplied[i];
            const auto & planned = pending->layout.arenas[i];
            if (input.domain != planned.budget.domain || input.allocation_class != planned.budget.allocation_class) {
                return reject(status::invalid_arena, i);
            }
            if (active) {
                bool matched = false;
                for (size_t j = 0; j < active->arenas.size(); ++j) {
                    const auto & previous = active->layout.arenas[j].budget;
                    if (previous.domain == input.domain && previous.allocation_class == input.allocation_class) {
                        matched = active->arenas[j].arena.get() == input.arena;
                        break;
                    }
                }
                if (!matched) return reject(status::invalid_arena, i);
            }
            if (!input.arena) {
                if (planned.budget.capacity != 0 || !planned.regions.empty()) {
                    return reject(status::invalid_arena, i);
                }
                continue;
            }
            for (size_t j = 0; j < i; ++j) {
                if (supplied[j].arena == input.arena) return reject(status::invalid_arena, i);
            }
            if (ggml_backend_memory_arena_get_state(input.arena) != GGML_BACKEND_MEMORY_ARENA_STATE_OPEN ||
                    ggml_backend_memory_arena_capacity(input.arena) < planned.budget.capacity) {
                return reject(status::invalid_arena, i);
            }
            const size_t alignment = ggml_backend_buffer_get_alignment(ggml_backend_memory_arena_parent(input.arena));
            for (const auto & region : planned.regions) {
                // Do not silently change planned metadata or assume stronger native alignment than reported.
                if (region.alignment != alignment) return reject(status::invalid_arena, i, region.id);
            }
            auto & record = records[i];
            record.arena.reset(ggml_backend_memory_arena_retain(input.arena));
            record.generation = ggml_backend_memory_arena_generation(input.arena);
            record.before.resize(ggml_backend_memory_arena_region_count(input.arena));
            for (size_t j = 0; j < record.before.size(); ++j) {
                if (!ggml_backend_memory_arena_get_region_at(input.arena, j, &record.before[j])) {
                    return reject(status::invalid_arena, i);
                }
            }
            if (record.generation != ggml_backend_memory_arena_generation(input.arena) ||
                    ggml_backend_memory_arena_get_state(input.arena) != GGML_BACKEND_MEMORY_ARENA_STATE_OPEN) {
                return reject(status::invalid_arena, i);
            }
            record.changed = record.before.size() != planned.regions.size();
            if (!record.changed) {
                for (size_t j = 0; j < record.before.size(); ++j) {
                    if (!same_transition_region(record.before[j], planned.regions[j])) record.changed = true;
                }
            }
            if (record.changed) {
                auto add_changed = [&](const ggml_backend_memory_region & region,
                        const std::vector<ggml_backend_memory_region> & other) {
                    const auto * counterpart = find_transition_region(other, region.id);
                    if (!(region.flags & GGML_BACKEND_MEMORY_REGION_PERSISTENT) || !counterpart ||
                            !same_transition_region(region, *counterpart)) {
                        if (std::find(changed.begin(), changed.end(), region.id) == changed.end()) changed.push_back(region.id);
                    }
                };
                for (const auto & region : record.before) add_changed(region, planned.regions);
                for (const auto & region : planned.regions) add_changed(region, record.before);
            }
        }
        for (const auto & fixed : pending->target.fixed) {
            const auto * before = find_transition_region(records[fixed.arena].before, fixed.region.id);
            if (!before || !same_transition_region(*before, fixed.region)) {
                return reject(status::invalid_arena, fixed.arena, fixed.region.id);
            }
        }
        if (cancellation_requested) {
            discard_pending();
            return {status::cancelled, no_consumer, {}, {}};
        }
        pending->arenas = std::move(records);
        pending->changed_resources = std::move(changed);
        started = true;
        phase = step::quiescing;
        for (size_t i = 0; i < pending->arenas.size(); ++i) {
            auto & record = pending->arenas[i];
            if (record.arena && (record.generation != ggml_backend_memory_arena_generation(record.arena.get()) ||
                    !ggml_backend_memory_arena_quiesce(record.arena.get()))) {
                return activation_error(status::activation_failed, no_consumer, i);
            }
        }

        current_arena = no_consumer;
        for (auto operation : {step::quiescing, step::draining, step::invalidating, step::releasing}) {
            phase = operation;
            for (size_t i = 0; i < pending->preparations.size(); ++i) {
                auto & preparation = pending->preparations[i];
                if (!preparation) continue;
                current_consumer = i;
                bool ok = false;
                switch (operation) {
                    case step::quiescing:    ok = preparation->quiesce(pending->changed_resources); break;
                    case step::draining:     ok = preparation->drain(); break;
                    case step::invalidating: ok = preparation->invalidate(); break;
                    case step::releasing:    ok = preparation->release(); break;
                    default: GGML_ABORT("invalid transition phase");
                }
                if (cancellation_requested) return activation_error(status::cancelled, i);
                if (!ok) return activation_error(status::activation_failed, i);
            }
        }

        current_consumer = no_consumer;
        phase = step::committing;
        for (size_t i = 0; i < pending->arenas.size(); ++i) {
            current_arena = i;
            auto & record = pending->arenas[i];
            if (!record.arena) continue;
            if (record.generation != ggml_backend_memory_arena_generation(record.arena.get())) {
                return activation_error(status::activation_failed, no_consumer, i);
            }
            if (!record.changed) continue;
            if (!ggml_backend_memory_arena_begin(record.arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE)) {
                return activation_error(status::activation_failed, no_consumer, i);
            }
            record.planning = true;
            for (const auto & region : pending->layout.arenas[i].regions) {
                if (!ggml_backend_memory_arena_reserve_at(record.arena.get(), region.id, region.offset,
                        region.size, region.alignment, region.flags, nullptr)) {
                    return activation_error(status::activation_failed, no_consumer, i, region.id);
                }
            }
            if (!ggml_backend_memory_arena_commit(record.arena.get())) {
                return activation_error(status::activation_failed, no_consumer, i);
            }
            record.planning = false;
            record.generation = ggml_backend_memory_arena_generation(record.arena.get());
        }

        phase = step::binding;
        for (size_t i = 0; i < pending->arenas.size(); ++i) {
            current_arena = i;
            auto & record = pending->arenas[i];
            if (!record.arena) continue;
            if (!ggml_backend_memory_arena_resume(record.arena.get())) {
                return activation_error(status::activation_failed, no_consumer, i);
            }
            for (const auto & region : pending->layout.arenas[i].regions) {
                current_resource = region.id;
                transition_lease_ptr lease(ggml_backend_memory_arena_acquire(record.arena.get(), region.id),
                    ggml_backend_memory_lease_free);
                if (!lease) return activation_error(status::allocation_failed, no_consumer, i, region.id);
                pending->leases.push_back(std::move(lease));
                pending->bindings.push_back({i, region, pending->leases.back().get()});
            }
            ggml_backend_memory_arena_quiesce(record.arena.get());
        }

        current_arena = no_consumer;
        current_resource = 0;
        for (auto operation : {step::binding, step::activating}) {
            phase = operation;
            for (size_t i = 0; i < pending->preparations.size(); ++i) {
                auto & preparation = pending->preparations[i];
                if (!preparation) continue;
                current_consumer = i;
                const bool ok = operation == step::binding ? preparation->bind(pending->bindings) : preparation->activate();
                if (cancellation_requested) return activation_error(status::cancelled, i);
                if (!ok) return activation_error(status::activation_failed, i);
            }
        }
        current_consumer = no_consumer;
        phase = step::discarding;
        pending->clear_preparations();
        pending->bindings.clear();
        pending->leases.clear();
        phase = step::activating;
        for (size_t i = 0; i < pending->arenas.size(); ++i) {
            auto & record = pending->arenas[i];
            if (record.arena && (record.generation != ggml_backend_memory_arena_generation(record.arena.get()) ||
                    !ggml_backend_memory_arena_resume(record.arena.get()))) {
                return activation_error(status::activation_failed, no_consumer, i);
            }
        }
        active = std::move(pending);
        cancellation_requested = false;
        phase = step::idle;
        return {status::activated, no_consumer, {}, {}};
    } catch (const std::bad_alloc &) {
        const auto exception = std::current_exception();
        return started ? activation_error(status::allocation_failed, current_consumer, current_arena, current_resource, exception)
                       : reject(status::allocation_failed, current_arena, current_resource, exception);
    } catch (...) {
        const auto exception = std::current_exception();
        return started ? activation_error(status::consumer_exception, current_consumer, current_arena, current_resource, exception)
                       : reject(status::consumer_exception, current_arena, current_resource, exception);
    }
}

// Return the last published snapshot, not a claim that a failed transition rolled physical state back.
const llama_memory_transition_target * llama_memory_transition::active_target() const noexcept {
    return active ? &active->target : nullptr;
}

// Failed transitions leave this logical snapshot unchanged and keep execution admission closed.
const llama_memory_layout * llama_memory_transition::active_layout() const noexcept {
    return active ? &active->layout : nullptr;
}
