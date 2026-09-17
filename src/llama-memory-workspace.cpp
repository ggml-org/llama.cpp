#include "llama-memory-workspace.h"

#include <algorithm>
#include <new>
#include <utility>

using workspace_lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

// Compare all region fields; geometry alone is not a binding identity.
static bool same_workspace_region(const ggml_backend_memory_region & a, const ggml_backend_memory_region & b) {
    return a.id == b.id && a.offset == b.offset && a.size == b.size && a.alignment == b.alignment && a.flags == b.flags;
}

struct llama_memory_workspace::implementation {
    struct placement {
        size_t group;
        size_t arena;
        ggml_backend_memory_region region;
    };
    struct attachment {
        placement where;
        workspace_lease_ptr lease;
    };
    ggml_backend_sched_t sched;
    std::vector<llama_memory_workspace_group> groups;
    llama_memory_workspace_hooks hooks;
    std::vector<attachment> active;
    bool accepting = false;
    bool pending = false;
    bool closing = false;

    // Validate canonical selected groups without probing or allocating device storage.
    bool valid() const {
        if (!sched || !hooks.quiesce || !hooks.invalidate) return false;
        const size_t slots = static_cast<size_t>(ggml_backend_sched_get_n_backends(sched));
        for (size_t i = 0; i < groups.size(); ++i) {
            const auto & group = groups[i].workspace;
            const auto & resource = groups[i].resource;
            if (!group.buft || group.size == 0 || group.first_slot >= slots || group.alignment == 0 ||
                    group.alignment != ggml_backend_buft_get_alignment(group.buft) ||
                    group.size % group.alignment != 0 || resource.id == 0 || resource.domain == 0 ||
                    resource.content != llama_memory_content::discardable) return false;
            if (!groups[i].stages.empty()) {
                size_t maximum = 0;
                for (size_t stage = 0; stage < groups[i].stages.size(); ++stage) {
                    const auto & requirement = groups[i].stages[stage];
                    if (!requirement.id || requirement.size > group.size ||
                            (requirement.size != 0 && requirement.size % group.alignment != 0)) return false;
                    for (size_t previous = 0; previous < stage; ++previous) {
                        if (groups[i].stages[previous].id == requirement.id) return false;
                    }
                    maximum = std::max(maximum, requirement.size);
                }
                if (maximum != group.size) return false;
            }
            auto * selected = ggml_backend_sched_get_backend(sched, static_cast<int>(group.first_slot));
            if (ggml_backend_sched_get_buffer_type(sched, selected) != group.buft) return false;
            for (size_t slot = 0; slot < group.first_slot; ++slot) {
                auto * earlier = ggml_backend_sched_get_backend(sched, static_cast<int>(slot));
                if (ggml_backend_sched_get_buffer_type(sched, earlier) == group.buft) return false;
            }
            for (size_t j = 0; j < i; ++j) {
                if (groups[j].workspace.buft == group.buft || groups[j].workspace.first_slot == group.first_slot ||
                        groups[j].resource.id == resource.id) return false;
            }
        }
        return true;
    }

    bool stage_size(const llama_memory_workspace_group & group, llama_memory_stage_id stage, size_t & size) const {
        if (group.stages.empty()) {
            size = group.workspace.size;
            return true;
        }
        const auto found = std::find_if(group.stages.begin(), group.stages.end(),
            [&](const llama_memory_workspace_stage & candidate) { return candidate.id == stage; });
        if (found == group.stages.end()) return false;
        size = found->size;
        return true;
    }

    // Resolve the scheduler slot only after group validation has established bounds.
    ggml_backend_t backend(size_t group) const {
        return ggml_backend_sched_get_backend(sched, static_cast<int>(groups[group].workspace.first_slot));
    }

    // Detach only leases owned by this consumer; foreign/fallback workspaces remain untouched.
    bool detach() {
        while (!active.empty()) {
            if (!ggml_backend_sched_detach_memory_lease(sched, backend(active.back().where.group))) return false;
            active.pop_back();
        }
        return true;
    }

    struct proposal : llama_memory_preparation {
        implementation & owner;
        std::vector<placement> desired;
        std::vector<placement> before;
        bool was_ready;
        bool target_ready;
        bool primary;
        bool affected = false;

        // Keep metadata snapshots, not old leases that would prevent arena repartition.
        proposal(implementation & owner, std::vector<placement> desired, bool target_ready, bool primary) :
            owner(owner), desired(std::move(desired)), was_ready(owner.accepting), target_ready(target_ready), primary(primary) {
            for (const auto & binding : owner.active) {
                auto previous = binding.where;
                // Arena indices belong to a target layout, not to a resource's permanent identity.
                for (const auto & next : this->desired) {
                    if (next.group == previous.group) previous.arena = next.arena;
                }
                before.push_back(previous);
            }
        }

        // Reverse preparations share the original preparation's admission exclusion.
        ~proposal() override {
            if (primary) owner.pending = false;
        }

        // Any changed scheduler group requires invalidating all of this scheduler's executable bindings.
        bool quiesce(const std::vector<llama_memory_resource_id> & changed) override {
            affected = !primary || !owner.accepting || desired.size() != owner.active.size();
            for (size_t i = 0; i < desired.size() && !affected; ++i) {
                const auto & old = owner.active[i].where;
                const auto & next = desired[i];
                affected = old.group != next.group || old.arena != next.arena || !same_workspace_region(old.region, next.region);
            }
            for (const auto & group : owner.groups) {
                affected = affected || std::find(changed.begin(), changed.end(), group.resource.id) != changed.end();
            }
            if (!affected) return true;
            owner.accepting = false;
            return owner.hooks.quiesce();
        }

        // Complete scheduler compute/copies before invalidating graphs or detaching leases.
        bool drain() override {
            if (affected) ggml_backend_sched_synchronize(owner.sched);
            return true;
        }

        // The caller retires native captures and invalidates graph metadata before scheduler placement reset.
        bool invalidate() override {
            if (!affected) return true;
            if (!owner.hooks.invalidate()) return false;
            ggml_backend_sched_reset(owner.sched);
            return true;
        }

        // All consumers have drained and invalidated before the coordinator reaches this phase.
        bool release() override {
            return !affected || owner.detach();
        }

        // Validate all staged bindings before attachment; roll back partial new attachments on failure.
        bool bind(const std::vector<llama_memory_region_binding> & bindings) override {
            std::vector<attachment> next;
            next.reserve(desired.size());
            for (const auto & place : desired) {
                const llama_memory_region_binding * found = nullptr;
                for (const auto & binding : bindings) {
                    if (binding.region.id == place.region.id) {
                        if (found) return false;
                        found = &binding;
                    }
                }
                ggml_backend_memory_region actual{};
                if (!found || found->arena != place.arena || !same_workspace_region(found->region, place.region) ||
                        !ggml_backend_memory_lease_get_region(found->lease, &actual) ||
                        !same_workspace_region(actual, place.region)) return false;
                auto * buffer = ggml_backend_memory_lease_buffer(found->lease);
                if (ggml_backend_buffer_get_type(buffer) != owner.groups[place.group].workspace.buft) return false;
                next.push_back({place, workspace_lease_ptr(ggml_backend_memory_lease_retain(found->lease), ggml_backend_memory_lease_free)});
            }
            if (!affected) {
                if (next.size() != owner.active.size()) return false;
                for (size_t i = 0; i < next.size(); ++i) {
                    if (ggml_backend_memory_lease_buffer(next[i].lease.get()) !=
                            ggml_backend_memory_lease_buffer(owner.active[i].lease.get())) return false;
                }
                return true;
            }
            if (!owner.active.empty()) return false;
            owner.active.reserve(next.size());
            for (auto & binding : next) {
                ggml_backend_buffer_set_usage(ggml_backend_memory_lease_buffer(binding.lease.get()), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (!ggml_backend_sched_attach_memory_lease(owner.sched, owner.backend(binding.where.group), binding.lease.get())) {
                    owner.detach();
                    return false;
                }
                owner.active.push_back(std::move(binding));
            }
            return true;
        }

        // Publication does not reserve or execute a graph; the caller must rebuild invalidated bindings first.
        bool activate() override {
            owner.accepting = target_ready;
            return true;
        }

        // Scratch bytes are discardable; restore attachment metadata and require graph reconstruction, not byte rollback.
        bool prepare_recovery(std::unique_ptr<llama_memory_preparation> & output) override {
            if (affected) output = std::make_unique<proposal>(owner, before, was_ready, false);
            return true;
        }
    };
};

// Retain immutable group metadata and borrow the scheduler/lifecycle hooks.
llama_memory_workspace::llama_memory_workspace(ggml_backend_sched_t sched,
        std::vector<llama_memory_workspace_group> groups, llama_memory_workspace_hooks hooks) :
    impl(new implementation{sched, std::move(groups), std::move(hooks), {}}) {}

// Keep the scheduler and callback dependencies alive until all owned leases are detached.
llama_memory_workspace::~llama_memory_workspace() {
    GGML_ASSERT(close());
}

// Register exact measured phase requirements without modifying the plan on rejection.
bool llama_memory_workspace::register_resources(
        llama_memory_execution_plan & plan, const std::vector<llama_memory_stage_id> & stages) const {
    if (!impl->valid() || stages.empty()) return false;
    try {
        auto next = plan;
        for (const auto & group : impl->groups) {
            for (const auto & resource : next.resources) if (resource.id == group.resource.id) return false;
            next.resources.push_back(group.resource);
        }
        for (size_t i = 0; i < stages.size(); ++i) {
            for (size_t j = 0; j < i; ++j) if (stages[j] == stages[i]) return false;
            auto stage = std::find_if(next.stages.begin(), next.stages.end(),
                [&](const llama_memory_stage & candidate) { return candidate.id == stages[i]; });
            if (stage == next.stages.end()) return false;
            for (const auto & group : impl->groups) {
                size_t size = 0;
                if (!impl->stage_size(group, stages[i], size)) return false;
                stage->requirements.push_back({group.resource.id, size, size,
                    group.workspace.alignment, LLAMA_MEMORY_ACCESS_WRITE, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            }
        }
        for (const auto & group : impl->groups) {
            if (group.stages.empty()) continue;
            if (group.stages.size() != stages.size()) return false;
            for (const auto & configured : group.stages) {
                if (std::find(stages.begin(), stages.end(), configured.id) == stages.end()) return false;
            }
        }
        if (llama_memory_plan_validate(next).status != llama_memory_plan_status::success) return false;
        plan = std::move(next);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

// Reject unregistered, undersized, or mismatched placements before touching scheduler state.
bool llama_memory_workspace::prepare(const llama_memory_transition_target & target, const llama_memory_layout & layout,
        std::unique_ptr<llama_memory_preparation> & output) {
    if (!impl->valid() || impl->pending || impl->closing || output ||
            llama_memory_plan_validate(target.plan).status != llama_memory_plan_status::success) return false;
    const auto stage = std::find_if(target.plan.stages.begin(), target.plan.stages.end(),
        [&](const llama_memory_stage & candidate) { return candidate.id == target.stage; });
    if (stage == target.plan.stages.end()) return false;
    std::vector<implementation::placement> desired;
    for (size_t i = 0; i < impl->groups.size(); ++i) {
        const auto & group = impl->groups[i];
        const auto resource = std::find_if(target.plan.resources.begin(), target.plan.resources.end(),
            [&](const llama_memory_resource & candidate) { return candidate.id == group.resource.id; });
        const auto requirement = std::find_if(stage->requirements.begin(), stage->requirements.end(),
            [&](const llama_memory_requirement & candidate) { return candidate.resource == group.resource.id; });
        size_t size = 0;
        if (!impl->stage_size(group, target.stage, size)) return false;
        if (resource == target.plan.resources.end() || resource->domain != group.resource.domain ||
                resource->allocation_class != group.resource.allocation_class || resource->content != llama_memory_content::discardable ||
                requirement == stage->requirements.end() || requirement->size_min != size ||
                requirement->size_preferred != size || requirement->alignment != group.workspace.alignment ||
                requirement->access != LLAMA_MEMORY_ACCESS_WRITE ||
                !(requirement->capabilities & LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS)) return false;
        bool found = false;
        for (size_t a = 0; a < layout.arenas.size(); ++a) {
            const auto & arena = layout.arenas[a];
            for (const auto & region : arena.regions) {
                if (region.id != group.resource.id) continue;
                if (found || arena.budget.domain != group.resource.domain || arena.budget.allocation_class != group.resource.allocation_class ||
                        region.size != size || region.alignment != group.workspace.alignment) return false;
                found = true;
                desired.push_back({i, a, region});
            }
        }
        if (found != (size != 0)) return false;
    }
    auto next = std::make_unique<implementation::proposal>(*impl, std::move(desired), true, true);
    impl->pending = true;
    output = std::move(next);
    return true;
}

// Teardown follows the same ordering as transition activation and rejects reentrant/pending use.
bool llama_memory_workspace::close() {
    if (impl->pending || impl->closing) return false;
    if (impl->active.empty()) {
        impl->accepting = false;
        return true;
    }
    impl->closing = true;
    impl->accepting = false;
    bool ok = false;
    try {
        if (impl->hooks.quiesce()) {
            ggml_backend_sched_synchronize(impl->sched);
            if (impl->hooks.invalidate()) {
                ggml_backend_sched_reset(impl->sched);
                ok = impl->detach();
            }
        }
    } catch (...) {
        ok = false;
    }
    impl->closing = false;
    return ok;
}

// This is logical attachment readiness, not proof that a new tensor graph has been rebuilt or submitted.
bool llama_memory_workspace::ready() const noexcept {
    return impl->accepting;
}
