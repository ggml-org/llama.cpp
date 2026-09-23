#include "llama-memory-phase-owner.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

bool llama_memory_phase_workspace_budget_make(const llama_memory_phase_workspace_measurement & m,
        llama_memory_phase_workspace_budget & output) noexcept {
    if (!m.alignment || (m.alignment & (m.alignment-1)) ||
            !m.target_prefill_graph_bytes || !m.mtp_draft_graph_bytes) return false;
    const auto aligned_sum = [&](size_t graph, size_t stage, size_t & bytes) {
        if (!graph || stage > SIZE_MAX-graph) return false;
        const size_t total=graph+stage;
        if (total > SIZE_MAX-(m.alignment-1)) return false;
        bytes=(total+m.alignment-1)&~(m.alignment-1);
        return true;
    };
    llama_memory_phase_workspace_budget next;
    if (!aligned_sum(m.target_prefill_graph_bytes,0,next.target_prefill_bytes) ||
            !aligned_sum(m.mtp_draft_graph_bytes,0,next.mtp_draft_bytes)) return false;
    for (size_t width=0;width<4;++width) {
        size_t verify=0,catchup=0;
        if (!aligned_sum(m.target_verify_graph_bytes[width],m.rollback_stage_bytes,verify) ||
                !aligned_sum(m.mtp_catchup_graph_bytes[width],m.mtp_stage_bytes,catchup)) return false;
        next.target_verify_bytes=std::max(next.target_verify_bytes,verify);
        next.mtp_catchup_bytes=std::max(next.mtp_catchup_bytes,catchup);
    }
    next.shared_parent_bytes=std::max({
        next.target_prefill_bytes,next.target_verify_bytes,
        next.mtp_catchup_bytes,next.mtp_draft_bytes});
    next.reclaimed_during_mtp_catchup_bytes=next.shared_parent_bytes-next.mtp_catchup_bytes;
    next.reclaimed_during_mtp_draft_bytes=next.shared_parent_bytes-next.mtp_draft_bytes;
    output=next;
    return true;
}

static bool same_identity(llama_memory_phase_cache_identity a, llama_memory_phase_cache_identity b) {
    return a.id == b.id && a.generation == b.generation;
}

struct llama_memory_phase_owner::implementation {
    std::vector<llama_memory_phase_target> targets;
    std::vector<llama_memory_transition_arena> arenas;
    llama_memory_transition transition;
    llama_memory_work_phase_tracker tracker;
    llama_memory_phase_cache_identity target;
    llama_memory_phase_cache_identity mtp;

    implementation(std::vector<llama_memory_phase_target> targets,
            std::vector<llama_memory_transition_arena> arenas,
            std::vector<llama_memory_consumer *> consumers,
            llama_memory_phase_cache_identity target, llama_memory_phase_cache_identity mtp) :
        targets(std::move(targets)), arenas(std::move(arenas)),
        transition(std::move(consumers)), target(target), mtp(mtp) {}
};

llama_memory_phase_owner::llama_memory_phase_owner(std::unique_ptr<implementation> impl) : impl(std::move(impl)) {}
llama_memory_phase_owner::~llama_memory_phase_owner() = default;

std::unique_ptr<llama_memory_phase_owner> llama_memory_phase_owner::create(
        std::vector<llama_memory_phase_target> targets,
        std::vector<llama_memory_transition_arena> arenas,
        std::vector<llama_memory_consumer *> consumers,
        llama_memory_phase_cache_identity target, llama_memory_phase_cache_identity mtp) {
    if (!target.id || !target.generation || !mtp.id || !mtp.generation ||
            target.id == mtp.id || targets.empty() || arenas.empty() || consumers.empty()) return {};
    for (size_t i=0;i<targets.size();++i) {
        if (targets[i].phase == llama_memory_work_phase::none || !targets[i].target.stage || !targets[i].max_tokens) return {};
        for (size_t j=0;j<i;++j) if (targets[i].phase == targets[j].phase) return {};
    }
    for (size_t i=0;i<consumers.size();++i) {
        if (!consumers[i]) return {};
        for (size_t j=0;j<i;++j) if (consumers[i] == consumers[j]) return {};
    }
    try {
        auto state=std::make_unique<implementation>(
            std::move(targets),std::move(arenas),std::move(consumers),target,mtp);
        return std::unique_ptr<llama_memory_phase_owner>(new llama_memory_phase_owner(std::move(state)));
    } catch (const std::bad_alloc &) {
        return {};
    }
}

llama_memory_phase_owner_result llama_memory_phase_owner::enter(llama_memory_work_phase phase,
        uint32_t tokens, llama_memory_phase_cache_identity target, llama_memory_phase_cache_identity mtp) {
    using status=llama_memory_phase_owner_status;
    auto & s=*impl;
    if (!same_identity(target,s.target) || !same_identity(mtp,s.mtp)) return {status::stale_identity,{}};
    if (!tokens || phase == llama_memory_work_phase::none) return {status::invalid_signal,{}};
    if (s.transition.state() == llama_memory_transition_state::invalid) return {status::session_invalid,{}};
    if (s.transition.state() != llama_memory_transition_state::idle) return {status::busy,{}};
    const auto found=std::find_if(s.targets.begin(),s.targets.end(),
        [&](const auto & entry) { return entry.phase == phase; });
    if (found == s.targets.end()) return {status::invalid_phase,{}};
    if (tokens > found->max_tokens) return {status::invalid_signal,{}};
    const auto current=s.tracker.snapshot();
    if (current.notifications == std::numeric_limits<uint64_t>::max() ||
            (current.phase != phase && current.revision == std::numeric_limits<uint64_t>::max())) {
        return {status::invalid_signal,{}};
    }
    if (current.phase == phase) {
        s.tracker.notify({phase,tokens,true});
        return {status::unchanged,{}};
    }
    auto prepared=s.transition.prepare(found->target);
    if (prepared.status != llama_memory_transition_status::prepared &&
            prepared.status != llama_memory_transition_status::no_change) return {status::transition_failed,prepared};
    if (prepared.status == llama_memory_transition_status::prepared) {
        auto activated=s.transition.activate(s.arenas);
        if (activated.status != llama_memory_transition_status::activated) {
            if (s.transition.state() == llama_memory_transition_state::failed &&
                    s.transition.recover().status != llama_memory_transition_status::recovered) {
                return {status::session_invalid,activated};
            }
            return {status::transition_failed,activated};
        }
    }
    s.tracker.notify({phase,tokens,true});
    return {status::changed,{}};
}

uint64_t llama_memory_phase_owner::admit() noexcept {
    return impl->tracker.snapshot().phase == llama_memory_work_phase::none ? 0 : impl->transition.admit();
}

bool llama_memory_phase_owner::finish(uint64_t admission,
        llama_memory_phase_cache_identity target, llama_memory_phase_cache_identity mtp) noexcept {
    auto & s=*impl;
    if (!admission || target.id != s.target.id || mtp.id != s.mtp.id ||
            target.generation < s.target.generation || mtp.generation < s.mtp.generation ||
            !s.transition.finish(admission)) return false;
    s.target=target;
    s.mtp=mtp;
    return true;
}

llama_memory_work_phase_snapshot llama_memory_phase_owner::snapshot() const noexcept {
    return impl->tracker.snapshot();
}
llama_memory_phase_cache_identity llama_memory_phase_owner::target_identity() const noexcept {
    return impl->target;
}
llama_memory_phase_cache_identity llama_memory_phase_owner::mtp_identity() const noexcept {
    return impl->mtp;
}
