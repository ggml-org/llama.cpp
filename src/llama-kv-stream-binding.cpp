#include "llama-kv-stream-binding.h"

#include <limits>
#include <new>
#include <utility>

using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

struct binding_operation {
    bool & busy;
    // Restore callback admission even when native construction throws.
    ~binding_operation() { busy = false; }
};

// Resolve by registry-local ordinal, preserving virtual-device identity and rejecting fallback types.
ggml_backend_buffer_type_t llama_kv_stream_device_buffer_type(ggml_backend_dev_t device) {
    if (!device) return nullptr;
    auto * reg = ggml_backend_dev_backend_reg(device);
    if (!reg) return nullptr;
    using factory_t = ggml_backend_buffer_type_t (*)(int);
    auto factory = reinterpret_cast<factory_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_device_buffer_type"));
    if (!factory) return nullptr;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg) && i <= size_t(std::numeric_limits<int>::max()); ++i) {
        if (ggml_backend_reg_dev_get(reg, i) != device) continue;
        auto * type = factory(int(i));
        return type && ggml_backend_buft_get_device(type) == device && !ggml_backend_buft_is_host(type) ? type : nullptr;
    }
    return nullptr;
}

// Host-cache identity survives device detach and must be assigned by the cache owner.
llama_kv_stream_binding::llama_kv_stream_binding(uint64_t id, ggml_backend_buffer_type_t type) :
    identity(id), expected_type(type) {}

// Validate and retain before exposing an address to native construction; publish only after capture succeeds.
bool llama_kv_stream_binding::bind(ggml_backend_memory_lease_t lease,
        const llama_kv_stream_policy_config & config, const llama_kv_stream_binding_factory & factory) {
    if (busy || !dependencies.empty() || !identity || !expected_type || !factory || revision == UINT64_MAX) return false;
    busy = true;
    binding_operation operation{busy};
    try {
        ggml_backend_memory_region region;
        if (!ggml_backend_memory_lease_get_region(lease, &region) || !region.id) return false;
        auto * buffer = ggml_backend_memory_lease_buffer(lease);
        if (!buffer || ggml_backend_buffer_get_type(buffer) != expected_type ||
                ggml_backend_buffer_get_size(buffer) != region.size || config.pool_bytes > region.size) return false;

        llama_kv_stream_binding_view next;
        if (llama_kv_stream_policy_initialize(config, next.initial_policy).status != llama_kv_stream_policy_status::success) return false;
        next.base = ggml_backend_buffer_get_base(buffer);
        const auto address = reinterpret_cast<uintptr_t>(next.base);
        if (!address || address % config.shape.alignment || config.pool_bytes > UINTPTR_MAX - address) return false;
        next.cache_id = identity;
        next.revision = revision + 1;
        next.buffer = buffer;
        next.capacity = config.pool_bytes;
        next.config = config;

        // This temporary retention also protects callbacks and cleanup if construction or capture fails.
        lease_ptr retained(ggml_backend_memory_lease_retain(lease), ggml_backend_memory_lease_free);
        std::vector<ggml_backend_memory_lease_t> candidate{retained.get()};
        auto native = factory(next);
        if (!native || !executor.capture(native, candidate, next.revision)) return false;
        current = next;
        dependencies = std::move(candidate);
        revision = next.revision;
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

// Do not expose a submission view while construction, quiescing, or retirement is in progress.
const llama_kv_stream_binding_view * llama_kv_stream_binding::view() const noexcept {
    return ready() ? &current : nullptr;
}

// Device rebinding does not create or replace authoritative host contents.
uint64_t llama_kv_stream_binding::cache_id() const noexcept { return identity; }

// Local readiness is not the transition coordinator's global execution gate.
bool llama_kv_stream_binding::ready() const noexcept { return !busy && executor.ready(); }

// Reuse the retained one-element dependency list; the common pin protects all queued users.
llama_memory_execution llama_kv_stream_binding::acquire() const {
    return ready() ? executor.acquire(dependencies, revision) : llama_memory_execution{};
}

// Close local submission admission before the coordinator drains affected consumers.
void llama_kv_stream_binding::quiesce() noexcept {
    if (!busy) executor.quiesce();
}

// Failed draining leaves the native executable and lease retained for retry.
llama_memory_executor_result llama_kv_stream_binding::detach(llama_memory_executor_backend & backend) {
    if (busy) return {llama_memory_executor_status::busy, {}};
    busy = true;
    binding_operation operation{busy};
    const auto result = executor.retire(backend);
    if (result.status == llama_memory_executor_status::retired) {
        dependencies.clear();
        current = {};
    }
    return result;
}
