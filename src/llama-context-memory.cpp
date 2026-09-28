#include "llama-context-memory.h"
#include "llama-memory-workspace.h"
#include "llama-kv-stream-model.h"
#include "../ggml/src/ggml-cuda-graph.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    llama_memory_executor executor;
    llama_memory_execution pending;
    std::unique_ptr<llama_memory_workspace> workspace;
    std::unique_ptr<llama_memory_transition> transition;
    llama_memory_text_phase_tracker text_phase;
    llama_kv_stream_model * shared_stream = nullptr;
    llama_context_memory * serial_parent = nullptr;
    llama_context_memory * serial_child = nullptr;
    llama_context_memory * self = nullptr;
    llama_context_memory * serial_active = nullptr;
    bool serial_borrowed = false;
    llama_memory_resource_id kv_pool_resource = 0;
    ggml_backend_buffer_t kv_parent = nullptr;
    size_t kv_parent_capacity = 0;
    size_t kv_arena = std::numeric_limits<size_t>::max();
    llama_memory_resource_id kv_workspace_resource = 0;
    size_t kv_workspace_max = 0;
    llama_memory_transition_target phase_target;
    std::vector<llama_memory_transition_arena> supplied;
    llama_memory_stage_id active_stage = 0, prefill_stage = 0, decode_stage = 0;
    uint64_t transition_count = 0, executor_revision = 0;
    uint64_t last_transition_us = 0;

    // Borrowed scratch may alias discardable graph/attention bytes, never persistent KV.
    size_t pool_offset(llama_memory_stage_id stage) const {
        if (!shared_stream || !kv_pool_resource || !stage) return 0;
        llama_memory_layout layout;
        if (llama_memory_layout_elastic(phase_target.plan, stage,
                phase_target.budgets, phase_target.fixed, layout).status !=
                llama_memory_layout_status::success || kv_arena >= layout.arenas.size()) return 0;
        for (const auto & region : layout.arenas[kv_arena].regions) {
            if (region.id == kv_pool_resource) return region.offset;
        }
        return 0;
    }

    // Constructor failures and normal teardown use the same ordering while the scheduler remains alive.
    ~implementation() {
        // The owner may outlive its draft or be freed first; neither leaves a dangling handoff pointer.
        if (serial_child) {
            GGML_ASSERT(serial_child->impl->drain());
            serial_child->impl->serial_parent = nullptr;
            serial_child = nullptr;
            serial_active = nullptr;
        }
        if (serial_parent && serial_parent->impl->serial_child == self) {
            serial_parent->impl->serial_child = nullptr;
            serial_parent->impl->serial_active = nullptr;
        }
        transition.reset();
        GGML_ASSERT(invalidate());
        if (workspace) GGML_ASSERT(workspace->close());
        workspace.reset();
        // A draft context may still retain a populated MTP layer when the target
        // shuts down. Retire its ring guard before detaching the shared KV arena.
        if (shared_stream && shared_stream->has_mtp_layer() &&
                !shared_stream->release_mtp_layer()) {
            shared_stream->abort();
            GGML_ASSERT(shared_stream->release_mtp_layer());
        }
        if (shared_stream) GGML_ASSERT(shared_stream->detach_shared_memory());
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

    bool capture_execution() {
        if (!workspace || !workspace->ready() ||
                executor_revision == UINT64_MAX) return false;
        bindings = workspace->leases();
        // A stage label may change while the exact scheduler lease and native
        // capture remain valid. Do not replace an executable that was not retired.
        if (executor.ready() && executor.matches(bindings,executor_revision)) return true;

        std::unique_ptr<llama_memory_executable> native =
            std::make_unique<context_executable>(caches);
        const uint64_t next = executor_revision+1;
        if (!executor.capture(native,bindings,next)) return false;
        executor_revision = next;
        return true;
    }

    bool activate_stage(llama_memory_stage_id stage) {
        if (!transition || !stage) return false;
        if (stage == active_stage) return true;
        const int64_t started = ggml_time_us();
        auto target = phase_target;
        target.stage = stage;
        try {
            const auto prepared = transition->prepare(target);
            if (prepared.status == llama_memory_transition_status::no_change) {
                active_stage = stage;
                return true;
            }
            if (prepared.status != llama_memory_transition_status::prepared) return false;
            const auto activated = transition->activate(supplied);
            if (activated.status != llama_memory_transition_status::activated) {
                const auto recovered = transition->recover();
                if (recovered.status == llama_memory_transition_status::recovered) {
                    capture_execution();
                }
                return false;
            }
            if (!capture_execution()) return false;
            active_stage = stage;
            ++transition_count;
            last_transition_us = uint64_t(ggml_time_us()-started);
            return true;
        } catch (...) {
            return false;
        }
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

// Preserve the old fixed-maximum construction for callers without phase measurements.
std::unique_ptr<llama_context_memory> llama_context_memory::create(ggml_backend_sched_t sched,
        const std::vector<ggml_backend_t> & backends,
        const std::vector<ggml_backend_memory_workspace_group> & groups) {
    llama_compute_workspace_plan plan;
    plan.groups = groups;
    plan.phase_sizes.assign(2, std::vector<size_t>(groups.size()));
    for (auto & phase : plan.phase_sizes) {
        for (size_t group = 0; group < groups.size(); ++group) {
            phase[group] = groups[group].size;
        }
    }
    return create(sched, backends, plan);
}

// Record distinct phase requirements while activating the conservative maximum until phase switching is enabled.
std::unique_ptr<llama_context_memory> llama_context_memory::create(ggml_backend_sched_t sched,
        const std::vector<ggml_backend_t> & backends,
        const llama_compute_workspace_plan & plan) {
    return create(sched,backends,plan,nullptr);
}

std::unique_ptr<llama_context_memory> llama_context_memory::create(ggml_backend_sched_t sched,
        const std::vector<ggml_backend_t> & backends,
        const llama_compute_workspace_plan & plan, llama_kv_stream_model * stream,
        llama_context_memory * serial_parent) {
    const auto & groups = plan.groups;
    if (stream && serial_parent) return {};
    auto * borrowed_parent = serial_parent ? serial_parent->shared_parent() : nullptr;
    if (serial_parent && (!serial_parent->shares_kv_memory() || !borrowed_parent)) return {};
    if (serial_parent && (serial_parent->impl->serial_child || plan.phase_sizes.size() != 2)) return {};
    if (!sched || !supported(backends) || plan.phase_sizes.empty() ||
            backends.size() != static_cast<size_t>(ggml_backend_sched_get_n_backends(sched))) return {};
    for (size_t i = 0; i < backends.size(); ++i) {
        if (ggml_backend_sched_get_backend(sched, static_cast<int>(i)) != backends[i]) return {};
    }
    for (const auto & phase : plan.phase_sizes) if (phase.size() != groups.size()) return {};
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
        size_t maximum = 0;
        for (const auto & phase : plan.phase_sizes) {
            if (phase[i] > group.size || (phase[i] != 0 && phase[i] % group.alignment != 0)) return {};
            maximum = std::max(maximum,phase[i]);
        }
        if (maximum != group.size) return {};
    }

    llama_kv_stream_memory_requirements kv;
    if (stream && (plan.phase_sizes.size() != 2 || !stream->memory_requirements(kv) ||
            !stream->prepare_shared_memory())) return {};
    struct handoff_guard {
        llama_kv_stream_model * stream;
        bool complete = false;
        ~handoff_guard() {
            if (stream && !complete) stream->resume_private_memory();
        }
    } handoff{stream};

    try {
        auto state = std::make_unique<implementation>();
        state->sched = sched;
        for (auto * backend : backends) {
            context_cache cache{};
            if (!context_cache_for(backend,cache)) return {};
            state->caches.push_back(cache);
        }
        std::vector<llama_memory_stage_id> stages{1};
        llama_memory_transition_target target;
        target.plan.stages.push_back({1,{},{}});
        for (size_t phase = 0; phase < plan.phase_sizes.size(); ++phase) {
            if (phase > static_cast<size_t>(std::numeric_limits<llama_memory_stage_id>::max()-2)) return {};
            const llama_memory_stage_id id = phase+2;
            stages.push_back(id);
            target.plan.stages.push_back({id,{id-1},{}});
        }
        target.stage = stages.front();

        const auto align_up = [](size_t value, size_t alignment, size_t & result) {
            if (!alignment || (alignment & (alignment-1)) ||
                    value > SIZE_MAX-(alignment-1)) return false;
            result = (value+alignment-1)&~(alignment-1);
            return true;
        };
        const size_t kv_attention = stream ?
            std::max(kv.attention_prefill_bytes,kv.attention_decode_bytes) : 0;
        size_t kv_group = std::numeric_limits<size_t>::max();
        size_t kv_arena = std::numeric_limits<size_t>::max();
        llama_memory_domain_id kv_domain = 0;

        std::vector<llama_memory_workspace_group> selected;
        std::vector<llama_memory_transition_arena> supplied;
        bool borrowed_group_seen = false;
        for (size_t i = 0; i < groups.size(); ++i) {
            const auto & group = groups[i];
            const bool carries_kv = stream && group.buft == kv.buffer_type;
            const bool borrowed_group = borrowed_parent &&
                group.buft == ggml_backend_buffer_get_type(borrowed_parent);
            if (borrowed_group_seen && borrowed_group) return {};
            if (carries_kv && kv_group != std::numeric_limits<size_t>::max()) return {};
            size_t capacity = group.size;
            if (carries_kv) {
                if (kv.shared_device_memory_bytes) {
                    if (kv.shared_device_memory_bytes < group.size) return {};
                    capacity = kv.shared_device_memory_bytes;
                } else {
                    for (size_t bytes : {kv.pool_bytes,kv.writer_bytes,kv_attention}) {
                        if (!align_up(capacity,group.alignment,capacity) ||
                                bytes > SIZE_MAX-capacity) return {};
                        capacity += bytes;
                    }
                }
            }
            if (borrowed_group) {
                capacity = serial_parent->shared_parent_capacity();
                if (capacity < group.size) return {};
                borrowed_group_seen = true;
                if (plan.phase_sizes[0][i] > serial_parent->impl->pool_offset(
                            serial_parent->impl->prefill_stage) ||
                        plan.phase_sizes[1][i] > serial_parent->impl->pool_offset(
                            serial_parent->impl->decode_stage)) return {};
            }
            ggml_backend_buffer_ptr parent(borrowed_group ?
                ggml_backend_buffer_retain(borrowed_parent) :
                ggml_backend_buft_alloc_buffer(group.buft,capacity));
            if (!parent) return {};
            if (!ggml_backend_buffer_supports_views(parent.get())) {
                if (carries_kv) return {};
                continue;
            }
            auto allocation = LLAMA_MEMORY_ALLOCATION_HOST;
            if (carries_kv) {
                allocation = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
            } else if (!ggml_backend_buft_is_host(group.buft)) {
                allocation = std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") ?
                    LLAMA_MEMORY_ALLOCATION_MANAGED : LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
            } else {
                for (const auto & cache : state->caches) {
                    if (cache.release && group.buft ==
                            ggml_backend_dev_host_buffer_type(ggml_backend_get_device(cache.backend))) {
                        allocation = LLAMA_MEMORY_ALLOCATION_HOST_PINNED;
                    }
                }
            }
            const uint64_t id = i+1;
            llama_compute_arena_ptr arena(ggml_backend_memory_arena_new_from_buffer(parent.get()));
            if (!arena) return {};
            llama_memory_workspace_group configured{
                group,{id,id,allocation,llama_memory_content::discardable},{}};
            configured.stages.push_back({stages.front(),group.size});
            for (size_t phase = 0; phase < plan.phase_sizes.size(); ++phase) {
                configured.stages.push_back({stages[phase+1],plan.phase_sizes[phase][i]});
            }
            selected.push_back(std::move(configured));
            target.plan.domains.push_back({id,allocation,LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            target.budgets.push_back({id,allocation,capacity,group.alignment});
            supplied.push_back({id,allocation,arena.get()});
            state->arenas.push_back({group.buft,capacity,group.first_slot,std::move(arena)});
            if (carries_kv) {
                kv_group = i;
                kv_arena = state->arenas.size()-1;
                kv_domain = id;
                state->kv_parent = ggml_backend_memory_arena_parent(state->arenas.back().arena.get());
                state->kv_parent_capacity = capacity;
                state->kv_workspace_resource = id;
                state->kv_workspace_max = group.size;
            }
        }
        if (serial_parent && !borrowed_group_seen) return {};
        if (stream && kv_group == std::numeric_limits<size_t>::max()) return {};

        auto * owner = state.get();
        state->workspace = std::make_unique<llama_memory_workspace>(sched,selected,llama_memory_workspace_hooks{
            [owner] { owner->executor.quiesce(); return true; },
            [owner] { return owner->invalidate(); },
        });
        if (!state->workspace->register_resources(target.plan,stages)) return {};

        uint64_t pool_id = 0, writer_id = 0, attention_id = 0;
        if (stream) {
            if (groups.size() > size_t(UINT64_MAX-3)) return {};
            pool_id = uint64_t(groups.size()+1);
            writer_id = pool_id+1;
            state->kv_pool_resource = pool_id;
            attention_id = writer_id+1;
            target.plan.resources.push_back({
                pool_id,kv_domain,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,
                llama_memory_content::reconstructible});
            target.plan.resources.push_back({
                writer_id,kv_domain,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,
                llama_memory_content::discardable});
            target.plan.resources.push_back({
                attention_id,kv_domain,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,
                llama_memory_content::discardable});
            for (size_t i = 0; i < target.plan.stages.size(); ++i) {
                const size_t attention = i == 0 ? kv_attention :
                    (i == 1 ? kv.attention_prefill_bytes : kv.attention_decode_bytes);
                auto & requirements = target.plan.stages[i].requirements;
                const size_t pool_preferred = (kv.shared_device_memory_bytes || i == 2) ?
                    target.budgets[kv_arena].capacity : kv.pool_bytes;
                // The serial MTP prefill workspace may extend slightly beyond
                // target compute into discardable attention scratch. Keep the
                // reconstructible KV pool after both, outside that alias.
                requirements.push_back({
                    attention_id,attention,attention,
                    groups[kv_group].alignment,LLAMA_MEMORY_ACCESS_WRITE,
                    LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
                requirements.push_back({
                    writer_id,kv.writer_bytes,kv.writer_bytes,
                    groups[kv_group].alignment,LLAMA_MEMORY_ACCESS_WRITE,
                    LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
                requirements.push_back({
                    pool_id,kv.pool_bytes,pool_preferred,
                    groups[kv_group].alignment,LLAMA_MEMORY_ACCESS_READ_WRITE,
                    LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            }
            target.plan.inputs.push_back(pool_id);
            target.plan.outputs.push_back(pool_id);
            if (llama_memory_plan_validate(target.plan).status != llama_memory_plan_status::success) return {};
            for (const auto & stage : target.plan.stages) {
                llama_memory_layout candidate;
                if (llama_memory_layout_elastic(
                        target.plan,stage.id,target.budgets,target.fixed,candidate).status !=
                        llama_memory_layout_status::success) return {};
            }
        }

        if (stream) {
            llama_memory_layout initial;
            if (llama_memory_layout_elastic(
                    target.plan,target.stage,target.budgets,target.fixed,initial).status !=
                    llama_memory_layout_status::success ||
                    initial.arenas.size() != state->arenas.size()) return {};
            for (size_t i = 0; i < initial.arenas.size(); ++i) {
                auto * arena = state->arenas[i].arena.get();
                if (!ggml_backend_memory_arena_quiesce(arena) ||
                        !ggml_backend_memory_arena_begin(arena,GGML_BACKEND_MEMORY_PLAN_NONE)) return {};
                for (const auto & region : initial.arenas[i].regions) {
                    if (!ggml_backend_memory_arena_reserve_at(
                            arena,region.id,region.offset,region.size,
                            region.alignment,region.flags,nullptr)) return {};
                }
                if (!ggml_backend_memory_arena_commit(arena) ||
                        !ggml_backend_memory_arena_resume(arena)) return {};
            }
        }

        state->transition = std::make_unique<llama_memory_transition>(
            std::vector<llama_memory_consumer *>{state->workspace.get()});
        const auto prepared = state->transition->prepare(target);
        if (prepared.status == llama_memory_transition_status::prepared) {
            if (state->transition->activate(supplied).status !=
                    llama_memory_transition_status::activated) return {};
        } else if (prepared.status != llama_memory_transition_status::no_change) {
            return {};
        }
        state->bindings = state->workspace->leases();

        if (stream) {
            std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> pool(
                ggml_backend_memory_arena_acquire(state->arenas[kv_arena].arena.get(),pool_id),
                ggml_backend_memory_lease_free);
            std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> writer(
                ggml_backend_memory_arena_acquire(state->arenas[kv_arena].arena.get(),writer_id),
                ggml_backend_memory_lease_free);
            std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> attention(
                ggml_backend_memory_arena_acquire(state->arenas[kv_arena].arena.get(),attention_id),
                ggml_backend_memory_lease_free);
            if (!pool || !writer || !attention ||
                    !stream->attach_shared_memory({
                        state->kv_parent,pool.get(),writer.get(),attention.get(),
                        pool_id,writer_id,attention_id,stages[1],stages[2]})) return {};
            state->shared_stream = stream;
            auto * consumer = stream->memory_consumer();
            if (!consumer) return {};
            state->transition = std::make_unique<llama_memory_transition>(
                std::vector<llama_memory_consumer *>{state->workspace.get(),consumer});
            state->phase_target = target;
            state->supplied = supplied;
            state->active_stage = stages.front();
            state->prefill_stage = stages[1];
            state->decode_stage = stages[2];
            state->kv_arena = kv_arena;
        }
        if (serial_parent) {
            state->phase_target = target;
            state->supplied = supplied;
            state->active_stage = stages.front();
            state->prefill_stage = stages[1];
            state->decode_stage = stages[2];
            state->serial_borrowed = true;
        }
        if (!state->capture_execution()) return {};
        state->serial_parent = serial_parent;
        std::unique_ptr<llama_context_memory> result(
            new llama_context_memory(std::move(state)));
        result->impl->self = result.get();
        if (serial_parent) serial_parent->impl->serial_child = result.get();
        handoff.complete = true;
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

// Keep ownership in one object declared after the scheduler in llama_context.
llama_context_memory::llama_context_memory(std::unique_ptr<implementation> impl) : impl(std::move(impl)) {}
llama_context_memory::~llama_context_memory() = default;
// Serial target/draft schedulers may alias scratch, but never execute or publish into it concurrently.
bool llama_context_memory::prepare_serial_target() noexcept {
    if (!impl->serial_child) return true;
    auto * child = impl->serial_child->impl.get();
    if (!child || child->serial_parent != this || child->serial_borrowed == false) return false;
    if (impl->serial_active == impl->serial_child && !child->drain()) return false;
    // Target streaming regains every ring slot; only the MTP resident mirror survives this handoff.
    if (impl->shared_stream && impl->shared_stream->has_mtp_layer() &&
            !impl->shared_stream->release_mtp_layer()) return false;
    impl->serial_active = this;
    return true;
}

bool llama_context_memory::prepare_serial_draft(llama_memory_text_phase phase) noexcept {
    if (!impl->serial_borrowed || !impl->serial_parent ||
            (phase != llama_memory_text_phase::prefill && phase != llama_memory_text_phase::decode)) return false;
    auto * parent = impl->serial_parent->impl.get();
    if (!parent || parent->serial_child != this || !parent->shared_stream) return false;
    if (parent->serial_active != this && !parent->drain()) return false;
    if (phase == llama_memory_text_phase::prefill && parent->active_stage == parent->decode_stage) {
        if (parent->shared_stream->has_mtp_layer() && !parent->shared_stream->release_mtp_layer()) return false;
        if (!parent->activate_stage(parent->prefill_stage)) return false;
    }
    const auto stage = phase == llama_memory_text_phase::decode ? impl->decode_stage : impl->prefill_stage;
    if (!impl->activate_stage(stage)) return false;
    const size_t protected_begin = parent->pool_offset(parent->active_stage);
    bool found = false;
    for (auto * lease : impl->workspace->leases()) {
        auto * buffer = lease ? ggml_backend_memory_lease_buffer(lease) : nullptr;
        if (!buffer || ggml_backend_buffer_get_type(buffer) != ggml_backend_buffer_get_type(parent->kv_parent)) continue;
        ggml_backend_memory_region region{};
        if (found || !ggml_backend_memory_lease_get_region(lease, &region) ||
                !protected_begin || region.offset != 0 || region.size > protected_begin) return false;
        found = true;
    }
    if (!found) return false;
    parent->serial_active = this;
    return true;
}


// A bounded host gate plus one queue pin protects unchanged workspaces without replanning per token.
ggml_status llama_context_memory::compute_async(ggml_cgraph * graph) {
    if (!graph || !impl->executor.ready()) return GGML_STATUS_FAILED;
    if (impl->serial_borrowed && (!impl->serial_parent ||
            impl->serial_parent->impl->serial_active != this)) return GGML_STATUS_FAILED;
    if (impl->serial_child && impl->serial_active != this) return GGML_STATUS_FAILED;
    const auto admission = impl->transition->admit();
    if (!admission) return GGML_STATUS_FAILED;
    if (!impl->pending) {
        impl->pending = impl->executor.acquire(
            impl->bindings,impl->executor_revision);
    }
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

const std::vector<ggml_backend_memory_lease_t> & llama_context_memory::workspace_leases() const noexcept {
    return impl->workspace->leases();
}
bool llama_context_memory::borrows_serial_parent() const noexcept { return impl->serial_borrowed; }


bool llama_context_memory::shares_kv_memory() const noexcept { return impl->shared_stream != nullptr; }
ggml_backend_buffer_t llama_context_memory::shared_parent() const noexcept { return impl->kv_parent; }
size_t llama_context_memory::shared_parent_capacity() const noexcept { return impl->kv_parent_capacity; }
uint64_t llama_context_memory::shared_arena_generation() const noexcept {
    return impl->kv_arena < impl->arenas.size() ?
        ggml_backend_memory_arena_generation(impl->arenas[impl->kv_arena].arena.get()) : 0;
}
uint64_t llama_context_memory::phase_transition_count() const noexcept {
    return impl->transition_count;
}

bool llama_context_memory::diagnostics(
        llama_context_memory_diagnostics & output) const noexcept {
    if (!impl->shared_stream || !impl->kv_parent || !impl->kv_workspace_resource) return false;
    llama_kv_stream_runtime_diagnostics stream;
    if (!impl->shared_stream->runtime_diagnostics(stream)) return false;
    size_t workspace_bytes = 0;
    for (auto * lease : impl->workspace->leases()) {
        ggml_backend_memory_region region;
        if (lease && ggml_backend_memory_lease_get_region(lease,&region) &&
                region.id == impl->kv_workspace_resource) {
            if (workspace_bytes) return false;
            workspace_bytes = region.size;
        }
    }
    size_t used = workspace_bytes;
    for (size_t bytes : {stream.pool_bytes,stream.writer_bytes,stream.attention_bytes}) {
        if (bytes > SIZE_MAX-used) return false;
        used += bytes;
    }
    if (used > impl->kv_parent_capacity || workspace_bytes > impl->kv_workspace_max) return false;
    output = {
        impl->active_stage == impl->decode_stage ?
            llama_memory_text_phase::decode : llama_memory_text_phase::prefill,
        impl->kv_parent_capacity,workspace_bytes,stream.pool_bytes,
        stream.writer_bytes,stream.attention_bytes,impl->kv_parent_capacity-used,
        impl->kv_workspace_max-workspace_bytes,
        shared_arena_generation(),impl->transition_count,stream.layout_revision,
        impl->last_transition_us,
        stream.resident_pages_per_layer,stream.ring_slots,stream.active_pages,
        stream.last_copy_bytes,stream.last_copy_calls,stream.last_copy_ms,stream.last_elapsed_ms,stream.streaming_active,true,
    };
    return true;
}

llama_memory_text_phase_result llama_context_memory::signal_text_phase(
        const llama_memory_text_phase_signal & signal) noexcept {
    const auto previous = impl->text_phase;
    auto result = impl->text_phase.notify(signal);
    if ((result.status != llama_memory_text_phase_status::changed &&
            result.status != llama_memory_text_phase_status::unchanged) ||
            !impl->shared_stream) return result;
    const auto stage = signal.phase == llama_memory_text_phase::decode ?
        impl->decode_stage : impl->prefill_stage;
    if (signal.phase == llama_memory_text_phase::prefill &&
            impl->active_stage != impl->decode_stage) return result;
    // The draft may have moved the physical parent back to prefill while the
    // target tracker still said decode, so reconcile the stage even on unchanged.
    bool ready = true;
    if (stage == impl->decode_stage && impl->serial_child) {
        auto * child = impl->serial_child->impl.get();
        ready = child && child->serial_parent == this && child->drain() &&
            child->activate_stage(child->decode_stage);
    }
    if (ready && impl->activate_stage(stage)) return result;
    impl->text_phase = previous;
    result.status = llama_memory_text_phase_status::transition_failed;
    result.after = result.before;
    result.revision = previous.snapshot().revision;
    return result;
}

llama_memory_text_phase_snapshot llama_context_memory::text_phase() const noexcept {
    return impl->text_phase.snapshot();
}
