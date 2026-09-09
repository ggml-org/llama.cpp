#include "llama-context-workspace.h"

#include "ggml-cpp.h"
#include "llama-impl.h"

#include <new>
#include <utility>

enum class llama_compute_arena_prepare_result {
    success,
    unsupported,
    error,
};

// Prepare one group without changing the output binding on fallback or failure.
static llama_compute_arena_prepare_result prepare_compute_arena_binding(
        ggml_backend_sched_t sched,
        ggml_backend_t backend,
        const ggml_backend_memory_workspace_group & group,
        llama_compute_arena_binding & binding) {
    if (sched == nullptr || backend == nullptr || group.buft == nullptr || group.size == 0) {
        return llama_compute_arena_prepare_result::error;
    }

    ggml_backend_buffer_ptr parent(ggml_backend_buft_alloc_buffer(group.buft, group.size));
    if (!parent) {
        return llama_compute_arena_prepare_result::error;
    }
    if (!ggml_backend_buffer_supports_views(parent.get())) {
        LLAMA_LOG_DEBUG("%s: %s does not support arena views, using scheduler allocation\n",
                __func__, ggml_backend_buft_name(group.buft));
        return llama_compute_arena_prepare_result::unsupported;
    }

    llama_compute_arena_ptr arena(ggml_backend_memory_arena_new_from_buffer(parent.get()));
    constexpr uint64_t workspace_region_id = 1;
    if (!arena ||
            !ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE) ||
            !ggml_backend_memory_arena_reserve_at(
                arena.get(), workspace_region_id, 0, group.size, group.alignment,
                GGML_BACKEND_MEMORY_REGION_NONE, nullptr) ||
            !ggml_backend_memory_arena_commit(arena.get())) {
        return llama_compute_arena_prepare_result::error;
    }

    std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)> lease(
        ggml_backend_memory_arena_acquire(arena.get(), workspace_region_id),
        ggml_backend_memory_lease_free);
    if (!lease) {
        return llama_compute_arena_prepare_result::error;
    }
    ggml_backend_buffer_set_usage(
        ggml_backend_memory_lease_buffer(lease.get()), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    if (!ggml_backend_sched_attach_memory_lease(sched, backend, lease.get())) {
        return llama_compute_arena_prepare_result::error;
    }

    binding = {
        group.buft,
        group.size,
        group.first_slot,
        std::move(arena),
    };
    return llama_compute_arena_prepare_result::success;
}

bool llama_prepare_compute_arena_bindings(
        ggml_backend_sched_t sched,
        const std::vector<ggml_backend_t> & backends,
        const std::vector<ggml_backend_memory_workspace_group> & groups,
        std::vector<llama_compute_arena_binding> & bindings) {
    if (sched == nullptr || !bindings.empty()) {
        return false;
    }

    std::vector<llama_compute_arena_binding> next;
    try {
        next.reserve(groups.size());
    } catch (const std::bad_alloc &) {
        return false;
    }

    auto fail = [&]() {
        for (auto it = next.rbegin(); it != next.rend(); ++it) {
            GGML_ASSERT(it->first_slot < backends.size());
            GGML_ASSERT(ggml_backend_sched_detach_memory_lease(sched, backends[it->first_slot]));
        }
        next.clear();
        return false;
    };

    for (const auto & group : groups) {
        if (group.first_slot >= backends.size()) {
            return fail();
        }
        llama_compute_arena_binding binding;
        switch (prepare_compute_arena_binding(sched, backends[group.first_slot], group, binding)) {
            case llama_compute_arena_prepare_result::success:
                next.push_back(std::move(binding));
                break;
            case llama_compute_arena_prepare_result::unsupported:
                break;
            case llama_compute_arena_prepare_result::error:
                return fail();
        }
    }

    bindings = std::move(next);
    return true;
}
