#include "llama-memory-executor.h"

#include <new>
#include <utility>

using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

struct llama_memory_captured_binding {
    lease_ptr lease;
    ggml_backend_memory_region region;
    ggml_backend_buffer_t buffer;
};

struct llama_memory_execution_state {
    std::vector<llama_memory_captured_binding> bindings;
    uint64_t revision = 0;
    // Declared last so native capture destruction precedes release of its leased dependencies.
    std::unique_ptr<llama_memory_executable> executable;
};

// Compare immutable binding identity, not arena or lease acquisition generations.
static bool same_binding(const llama_memory_captured_binding & binding,
        const ggml_backend_memory_region & region, ggml_backend_buffer_t buffer) {
    const auto & saved = binding.region;
    return binding.buffer == buffer && saved.id == region.id && saved.offset == region.offset &&
           saved.size == region.size && saved.alignment == region.alignment && saved.flags == region.flags;
}

// Read a valid, session-identified lease without relying on a native pointer representation.
static bool read_binding(ggml_backend_memory_lease_t lease,
        ggml_backend_memory_region & region, ggml_backend_buffer_t & buffer) {
    if (!ggml_backend_memory_lease_get_region(lease, &region) || region.id == 0) {
        return false;
    }
    buffer = ggml_backend_memory_lease_buffer(lease);
    return buffer != nullptr;
}

// Retain one capture snapshot for asynchronous execution.
llama_memory_execution::llama_memory_execution(std::shared_ptr<llama_memory_execution_state> state) :
    state(std::move(state)) {}

// An empty pin cannot authorize submission.
llama_memory_execution::operator bool() const noexcept {
    return state != nullptr;
}

// Keep the native capture accessible only through a live pin.
llama_memory_executable * llama_memory_execution::executable() const noexcept {
    return state ? state->executable.get() : nullptr;
}

// The caller must have observed backend completion before dropping its pin.
void llama_memory_execution::reset() noexcept {
    state.reset();
}

// Outstanding backend pins retain the capture and leases even after its cache owner disappears.
llama_memory_executor::~llama_memory_executor() {
    busy = true;
    accepting = false;
    captured.reset();
}

// Adopt only after validation and retention succeed, leaving caller ownership unchanged on rejection.
bool llama_memory_executor::capture(
        std::unique_ptr<llama_memory_executable> & executable,
        const std::vector<ggml_backend_memory_lease_t> & bindings,
        uint64_t runtime_revision) {
    if (busy || captured || !executable) {
        return false;
    }
    try {
        auto next = std::make_shared<llama_memory_execution_state>();
        next->revision = runtime_revision;
        next->bindings.reserve(bindings.size());
        for (auto lease : bindings) {
            ggml_backend_memory_region region;
            ggml_backend_buffer_t buffer;
            if (!read_binding(lease, region, buffer)) {
                return false;
            }
            bool alias = false;
            for (const auto & existing : next->bindings) {
                if (existing.region.id == region.id) {
                    if (!same_binding(existing, region, buffer)) {
                        return false;
                    }
                    alias = true;
                    break;
                }
            }
            if (!alias) {
                next->bindings.push_back({
                    lease_ptr(ggml_backend_memory_lease_retain(lease), ggml_backend_memory_lease_free),
                    region, buffer,
                });
            }
        }
        next->executable = std::move(executable);
        captured = std::move(next);
        accepting = true;
    } catch (const std::bad_alloc &) {
        return false;
    }
    return true;
}

// Compare dependency sets independent of input order and exact duplicate aliases.
bool llama_memory_executor::matches(
        const std::vector<ggml_backend_memory_lease_t> & bindings, uint64_t runtime_revision) const {
    if (!captured || captured->revision != runtime_revision) {
        return false;
    }
    for (auto lease : bindings) {
        ggml_backend_memory_region region;
        ggml_backend_buffer_t buffer;
        if (!read_binding(lease, region, buffer)) {
            return false;
        }
        bool found = false;
        for (const auto & saved : captured->bindings) {
            if (same_binding(saved, region, buffer)) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    for (const auto & saved : captured->bindings) {
        bool found = false;
        for (auto lease : bindings) {
            ggml_backend_memory_region region;
            GGML_ASSERT(ggml_backend_memory_lease_get_region(lease, &region));
            if (region.id == saved.region.id) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

// Retain the capture before handing its native resources to a backend submission.
llama_memory_execution llama_memory_executor::acquire(
        const std::vector<ggml_backend_memory_lease_t> & bindings, uint64_t runtime_revision) const {
    if (!ready() || !matches(bindings, runtime_revision)) {
        return {};
    }
    return llama_memory_execution(captured);
}

// Ignore unrelated arena changes while retaining dependencies that remain valid.
bool llama_memory_executor::affected_by(const std::vector<llama_memory_resource_id> & resources) const noexcept {
    if (captured) {
        for (const auto & binding : captured->bindings) {
            for (auto resource : resources) {
                if (binding.region.id == resource) {
                    return true;
                }
            }
        }
    }
    return false;
}

// Close launch admission before calling the backend and keep it closed after an incomplete drain.
llama_memory_executor_result llama_memory_executor::retire(llama_memory_executor_backend & backend) {
    using status = llama_memory_executor_status;
    if (busy) {
        return {status::busy, {}};
    }
    if (!captured) {
        return {status::unchanged, {}};
    }
    accepting = false;
    busy = true;
    try {
        if (!backend.drain()) {
            busy = false;
            return {status::drain_failed, {}};
        }
    } catch (...) {
        busy = false;
        return {status::drain_failed, std::current_exception()};
    }
    if (outstanding() != 0) {
        busy = false;
        return {status::pending, {}};
    }
    captured.reset();
    busy = false;
    return {status::retired, {}};
}

// Preserve a capture without draining when none of its dependencies is affected.
llama_memory_executor_result llama_memory_executor::retire_if_affected(
        llama_memory_executor_backend & backend, const std::vector<llama_memory_resource_id> & resources) {
    if (busy) {
        return {llama_memory_executor_status::busy, {}};
    }
    if (!affected_by(resources)) {
        return {};
    }
    return retire(backend);
}

// Launch admission is separate from snapshot equality.
bool llama_memory_executor::ready() const noexcept {
    return captured && accepting && !busy;
}

// Only the executor owner and move-only execution pins hold shared snapshot references.
size_t llama_memory_executor::outstanding() const noexcept {
    return captured ? static_cast<size_t>(captured.use_count() - 1) : 0;
}
