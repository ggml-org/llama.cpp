#include "llama-context-memory.h"
#include "llama-memory-workspace.h"
#include "../ggml/src/ggml-cuda-graph.h"
#include "ggml-cpp.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

struct context_cache {
    ggml_backend_t backend;
    ggml_backend_cuda_graph_release_all_t release;
};

// Discover only verified native-cache lifetimes; unsupported backends keep the legacy path.
static bool context_cache_for(ggml_backend_t backend, context_cache & cache) {
    if (!backend) return false;
    auto * dev = ggml_backend_get_device(backend);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return false;
    cache = {backend, nullptr};
    if (std::strcmp(ggml_backend_reg_name(reg), "CPU") == 0) return true;
    cache.release = reinterpret_cast<ggml_backend_cuda_graph_release_all_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_release_all"));
    return cache.release != nullptr;
}

struct context_executable : llama_memory_executable {
    std::vector<context_cache> caches;
    explicit context_executable(std::vector<context_cache> caches) : caches(std::move(caches)) {}

    // All scheduler work has drained; discard native captures before the guard releases leased storage.
    ~context_executable() override {
        for (const auto & cache : caches) if (cache.release) cache.release(cache.backend);
    }
};

struct llama_context_memory::implementation : llama_memory_executor_backend {
    ggml_backend_sched_t sched = nullptr;
    std::vector<context_cache> caches;
    std::vector<llama_compute_arena_binding> arenas;
    std::vector<ggml_backend_memory_lease_t> bindings;
    std::vector<std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>> leases;
    llama_memory_executor executor;
    llama_memory_execution pending;
    std::unique_ptr<llama_memory_workspace> workspace;
    std::unique_ptr<llama_memory_transition> transition;

    // Constructor failures and normal teardown use the same ordering while the scheduler remains alive.
    ~implementation() {
        transition.reset();
        GGML_ASSERT(invalidate());
        if (workspace) GGML_ASSERT(workspace->close());
        workspace.reset();
    }

    // Completion is independent of host admission; one pin can cover many queued graph splits.
    bool drain() override {
        if (sched) ggml_backend_sched_synchronize(sched);
        pending.reset();
        return true;
    }

    // Retire the full backend cache domain, not a borrowed scheduler split descriptor.
    bool invalidate() {
        const auto result = executor.retire(*this);
        return result.status == llama_memory_executor_status::retired || result.status == llama_memory_executor_status::unchanged;
    }
};

// CPU has no retained native executable cache; CUDA must provide whole-cache invalidation.
bool llama_context_memory::supported(const std::vector<ggml_backend_t> & backends) {
    if (backends.empty()) return false;
    size_t cuda_backends = 0;
    for (auto * backend : backends) {
        context_cache cache{};
        if (!context_cache_for(backend, cache)) return false;
        if (cache.release && ++cuda_backends > 1) return false;
    }
    return true;
}

// Build a fixed-lifetime coordinator with exactly the milestone-3 group capacities.
std::unique_ptr<llama_context_memory> llama_context_memory::create(ggml_backend_sched_t sched,
        const std::vector<ggml_backend_t> & backends,
        const std::vector<ggml_backend_memory_workspace_group> & groups) {
    if (!sched || !supported(backends) ||
            backends.size() != static_cast<size_t>(ggml_backend_sched_get_n_backends(sched))) return {};
    for (size_t i = 0; i < backends.size(); ++i) {
        if (ggml_backend_sched_get_backend(sched, static_cast<int>(i)) != backends[i]) return {};
    }
    // Validate all group labels before the first physical allocation.
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto & group = groups[i];
        if (!group.buft || group.size == 0 || group.first_slot >= backends.size() ||
                group.alignment != ggml_backend_buft_get_alignment(group.buft) || group.alignment == 0 ||
                group.size % group.alignment != 0 ||
                ggml_backend_sched_get_buffer_type(sched, backends[group.first_slot]) != group.buft) return {};
        for (size_t j = 0; j < i; ++j) if (groups[j].buft == group.buft) return {};
        for (size_t j = 0; j < group.first_slot; ++j) {
            if (ggml_backend_sched_get_buffer_type(sched, backends[j]) == group.buft) return {};
        }
    }
    try {
        auto state = std::make_unique<implementation>();
        state->sched = sched;
        for (auto * backend : backends) {
            context_cache cache{};
            if (!context_cache_for(backend, cache)) return {};
            state->caches.push_back(cache);
        }
        std::vector<llama_memory_workspace_group> selected;
        llama_memory_transition_target target;
        target.stage = 1;
        target.plan.stages = {{1, {}, {}}, {2, {1}, {}}};
        std::vector<llama_memory_transition_arena> supplied;
        for (size_t i = 0; i < groups.size(); ++i) {
            const auto & group = groups[i];
            ggml_backend_buffer_ptr parent(ggml_backend_buft_alloc_buffer(group.buft, group.size));
            if (!parent) return {};
            if (!ggml_backend_buffer_supports_views(parent.get())) continue;
            // Match the actual factories used by the supported CPU/CUDA backend set.
            auto allocation = LLAMA_MEMORY_ALLOCATION_HOST;
            if (!ggml_backend_buft_is_host(group.buft)) {
                allocation = std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") ?
                    LLAMA_MEMORY_ALLOCATION_MANAGED : LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
            } else {
                for (const auto & cache : state->caches) {
                    if (cache.release && group.buft == ggml_backend_dev_host_buffer_type(ggml_backend_get_device(cache.backend))) {
                        allocation = LLAMA_MEMORY_ALLOCATION_HOST_PINNED;
                    }
                }
            }
            const uint64_t id = i + 1;
            llama_compute_arena_ptr arena(ggml_backend_memory_arena_new_from_buffer(parent.get()));
            if (!arena) return {};
            selected.push_back({group, {id, id, allocation, llama_memory_content::discardable}});
            target.plan.domains.push_back({id, allocation, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            target.budgets.push_back({id, allocation, group.size, group.alignment});
            supplied.push_back({id, allocation, arena.get()});
            state->arenas.push_back({group.buft, group.size, group.first_slot, std::move(arena)});
        }
        auto * owner = state.get();
        state->workspace = std::make_unique<llama_memory_workspace>(sched, selected, llama_memory_workspace_hooks{
            [owner] { owner->executor.quiesce(); return true; },
            [owner] { return owner->invalidate(); },
        });
        if (!state->workspace->register_resources(target.plan, {1, 2})) return {};
        state->transition = std::make_unique<llama_memory_transition>(
            std::vector<llama_memory_consumer *>{state->workspace.get()});
        const auto prepared = state->transition->prepare(target);
        if (prepared.status == llama_memory_transition_status::prepared) {
            if (state->transition->activate(supplied).status != llama_memory_transition_status::activated) return {};
        } else if (prepared.status != llama_memory_transition_status::no_change) {
            return {};
        }
        for (size_t i = 0; i < selected.size(); ++i) {
            state->leases.emplace_back(ggml_backend_memory_arena_acquire(state->arenas[i].arena.get(), selected[i].resource.id),
                ggml_backend_memory_lease_free);
            if (!state->leases.back()) return {};
            state->bindings.push_back(state->leases.back().get());
        }
        std::unique_ptr<llama_memory_executable> native = std::make_unique<context_executable>(state->caches);
        if (!state->executor.capture(native, state->bindings, 1)) return {};
        return std::unique_ptr<llama_context_memory>(new llama_context_memory(std::move(state)));
    } catch (const std::bad_alloc &) {
        return {};
    }
}

// Keep ownership in one object declared after the scheduler in llama_context.
llama_context_memory::llama_context_memory(std::unique_ptr<implementation> impl) : impl(std::move(impl)) {}
llama_context_memory::~llama_context_memory() = default;

// A bounded host gate plus one queue pin protects unchanged workspaces without replanning per token.
ggml_status llama_context_memory::compute_async(ggml_cgraph * graph) {
    if (!graph || !impl->executor.ready()) return GGML_STATUS_FAILED;
    const auto admission = impl->transition->admit();
    if (!admission) return GGML_STATUS_FAILED;
    if (!impl->pending) impl->pending = impl->executor.acquire(impl->bindings, 1);
    if (!impl->pending) {
        impl->transition->finish(admission);
        return GGML_STATUS_FAILED;
    }
    try {
        const auto result = ggml_backend_sched_graph_compute_async(impl->sched, graph);
        if (result != GGML_STATUS_SUCCESS) impl->executor.quiesce();
        GGML_ASSERT(impl->transition->finish(admission));
        return result;
    } catch (...) {
        impl->executor.quiesce();
        impl->transition->finish(admission);
        throw;
    }
}

// Keep the immutable lifetime pin between requests; executor retirement uses implementation::drain to release it.
void llama_context_memory::synchronize() { ggml_backend_sched_synchronize(impl->sched); }

// Fallback-only schedulers need no physical arena while retaining the same teardown protocol.
bool llama_context_memory::uses_arenas() const noexcept { return !impl->arenas.empty(); }

const std::vector<ggml_backend_memory_lease_t> & llama_context_memory::workspace_leases() const noexcept { return impl->bindings; }
