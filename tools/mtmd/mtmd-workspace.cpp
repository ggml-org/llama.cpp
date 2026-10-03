#include "mtmd-workspace.h"
#include "../../src/llama-memory-workspace.h"
#include "../../src/llama-memory-executor.h"
#include "../../src/llama-context-memory.h"
#include "../../ggml/src/ggml-cuda-graph.h"
#include "../../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>
#include <unordered_set>

namespace {
struct workspace_cache {
    ggml_backend_t backend;
    ggml_backend_cuda_graph_release_all_t release = nullptr;
};

struct workspace_executable : llama_memory_executable {
    std::vector<workspace_cache> caches;
    explicit workspace_executable(std::vector<workspace_cache> caches) : caches(std::move(caches)) {}
    ~workspace_executable() override {
        for (const auto & cache : caches) if (cache.release) cache.release(cache.backend);
    }
};

struct workspace_gate {
    bool & busy;
    explicit workspace_gate(bool & busy) : busy(busy) { busy = true; }
    ~workspace_gate() { busy = false; }
};

// Scheduler measurement rewrites split inputs and may optimize tensor metadata.
struct workspace_graph_restore {
    ggml_cgraph * graph;
    std::vector<ggml_tensor *> nodes,leafs;
    std::vector<std::pair<ggml_tensor *,ggml_tensor>> tensors;
    explicit workspace_graph_restore(ggml_cgraph * graph) : graph(graph),
        nodes(graph->nodes,graph->nodes+graph->n_nodes),leafs(graph->leafs,graph->leafs+graph->n_leafs) {
        std::unordered_set<ggml_tensor *> seen;
        auto pending=nodes;
        pending.insert(pending.end(),leafs.begin(),leafs.end());
        while (!pending.empty()) {
            auto * tensor=pending.back(); pending.pop_back();
            if (!tensor || !seen.insert(tensor).second) continue;
            tensors.emplace_back(tensor,*tensor);
            for (auto * src : tensor->src) pending.push_back(src);
            pending.push_back(tensor->view_src);
        }
    }
    ~workspace_graph_restore() {
        for (const auto & tensor : tensors) *tensor.first=tensor.second;
        std::copy(nodes.begin(),nodes.end(),graph->nodes);
        std::copy(leafs.begin(),leafs.end(),graph->leafs);
        graph->n_nodes=int(nodes.size());
        graph->n_leafs=int(leafs.size());
    }
};
}

struct mtmd_compute_workspace::implementation : llama_memory_executor_backend {
    ggml_backend_sched_t sched;
    std::vector<ggml_backend_t> backends;
    size_t graph_capacity;
    ggml_backend_sched_ptr measurement;
    std::vector<workspace_cache> caches;
    std::vector<ggml_backend_memory_workspace_group> measured,granted;
    std::unique_ptr<llama_memory_workspace> consumer;
    std::unique_ptr<llama_context_memory> borrowed;
    llama_memory_executor executor;
    llama_memory_execution pending;
    ggml_cgraph * graph = nullptr;
    uint64_t revision = 0;
    bool valid = false, busy = false;

    implementation(ggml_backend_sched_t sched,std::vector<ggml_backend_t> backends,size_t graph_capacity) :
        sched(sched),backends(std::move(backends)),graph_capacity(graph_capacity) {
        if (!sched || !graph_capacity || this->backends.empty() ||
                this->backends.size() != size_t(ggml_backend_sched_get_n_backends(sched))) return;
        for (size_t i=0;i<this->backends.size();++i) {
            auto * backend=this->backends[i];
            if (!backend || ggml_backend_sched_get_backend(sched,int(i)) != backend) return;
            auto * dev=ggml_backend_get_device(backend);
            auto * reg=dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (!reg) return;
            workspace_cache cache{backend};
            cache.release=reinterpret_cast<ggml_backend_cuda_graph_release_all_t>(
                ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_release_all"));
            if (!cache.release && std::strcmp(ggml_backend_reg_name(reg),"CPU")) return;
            caches.push_back(cache);
        }
        std::vector<ggml_backend_buffer_type_t> types;
        for (auto * backend : this->backends) types.push_back(ggml_backend_sched_get_buffer_type(sched,backend));
        measurement.reset(ggml_backend_sched_new(this->backends.data(),types.data(),int(types.size()),graph_capacity,false,true));
        valid=bool(measurement);
    }

    bool drain() override {
        if (borrowed) borrowed->synchronize();
        ggml_backend_sched_synchronize(sched);
        pending.reset();
        return true;
    }

    bool invalidate() {
        if (borrowed) { graph=nullptr; return borrowed->retire_graph(); }
        const auto result=executor.retire(*this);
        if (result.status != llama_memory_executor_status::retired &&
                result.status != llama_memory_executor_status::unchanged) return false;
        if (result.status == llama_memory_executor_status::unchanged) {
            drain();
            for (const auto & cache : caches) if (cache.release) cache.release(cache.backend);
        }
        graph=nullptr;
        return true;
    }

    bool close() {
        borrowed.reset();
        if (consumer && !consumer->close()) return false;
        consumer.reset();
        if (!invalidate()) return false;
        granted.clear();
        return true;
    }

    bool capture() {
        if (!consumer || !consumer->ready() || revision == UINT64_MAX) return false;
        std::unique_ptr<llama_memory_executable> executable=std::make_unique<workspace_executable>(caches);
        if (!executor.capture(executable,consumer->leases(),revision+1)) return false;
        ++revision;
        return true;
    }

    bool requirements(ggml_cgraph * candidate,std::vector<ggml_backend_memory_workspace_group> & groups) {
        if (!candidate || ggml_graph_n_nodes(candidate) <= 0 ||
                size_t(ggml_graph_n_nodes(candidate)) > graph_capacity ||
                size_t(candidate->n_leafs) > graph_capacity-size_t(ggml_graph_n_nodes(candidate))) return false;
        std::vector<size_t> sizes(backends.size());
        std::vector<ggml_backend_buffer_type_t> types;
        for (auto * backend : backends) types.push_back(ggml_backend_sched_get_buffer_type(sched,backend));
        workspace_graph_restore restore(candidate);
        ggml_backend_sched_reset(measurement.get());
        ggml_backend_sched_reserve_size(measurement.get(),candidate,sizes.data());
        groups.resize(backends.size());
        size_t count=groups.size();
        if (!ggml_backend_memory_plan_workspace_groups(types.data(),sizes.data(),1,types.size(),groups.data(),&count)) return false;
        groups.resize(count);
        return true;
    }
};

mtmd_compute_workspace::mtmd_compute_workspace(ggml_backend_sched_t sched,std::vector<ggml_backend_t> backends,size_t graph_capacity) :
    impl(new implementation(sched,std::move(backends),graph_capacity)) {}

mtmd_compute_workspace::~mtmd_compute_workspace() { GGML_ASSERT(release()); }

bool mtmd_compute_workspace::measure(ggml_cgraph * graph,std::vector<ggml_backend_memory_workspace_group> & output) {
    auto & s=*impl;
    if (!s.valid || s.busy || s.consumer || s.borrowed || !graph) return false;
    workspace_gate gate(s.busy);
    try {
        if (!s.invalidate()) return false;
        std::vector<ggml_backend_memory_workspace_group> next;
        if (!s.requirements(graph,next) || next.empty()) return false;
        auto result=next;
        s.measured=std::move(next);
        output=std::move(result);
        return true;
    } catch (...) { return false; }
}

bool mtmd_compute_workspace::attach(const std::vector<ggml_backend_memory_lease_t> & leases) {
    auto & s=*impl;
    if (!s.valid || s.busy || s.measured.empty() || leases.size() != s.measured.size()) return false;
    workspace_gate gate(s.busy);
    try {
        llama_memory_transition_target target;
        target.stage=1;
        target.plan.stages={{1,{},{}}};
        llama_memory_layout layout;
        std::vector<llama_memory_workspace_group> groups;
        std::vector<llama_memory_region_binding> bindings;
        std::vector<ggml_backend_memory_workspace_group> grants;
        for (size_t i=0;i<leases.size();++i) {
            ggml_backend_memory_region region{};
            auto * buffer=leases[i] ? ggml_backend_memory_lease_buffer(leases[i]) : nullptr;
            const auto & need=s.measured[i];
            if (!buffer || !ggml_backend_memory_lease_get_region(leases[i],&region) || !region.id ||
                    ggml_backend_buffer_get_type(buffer) != need.buft || region.size != ggml_backend_buffer_get_size(buffer) ||
                    region.size < need.size || region.alignment != need.alignment || region.size%need.alignment ||
                    region.offset > SIZE_MAX-region.size) return false;
            for (const auto & group : groups) if (group.resource.id == region.id) return false;
            auto grant=need; grant.size=region.size;
            const auto allocation=ggml_backend_buft_is_host(need.buft) ? LLAMA_MEMORY_ALLOCATION_HOST : LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
            groups.push_back({grant,{region.id,region.id,allocation,llama_memory_content::discardable},{}});
            grants.push_back(grant);
            target.plan.domains.push_back({region.id,allocation,LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            const llama_memory_arena_budget budget{region.id,allocation,region.offset+region.size,region.alignment};
            target.budgets.push_back(budget);
            layout.arenas.push_back({budget,{region},region.size,region.offset+region.size,region.offset});
            bindings.push_back({i,region,leases[i]});
        }
        auto candidate=std::make_unique<llama_memory_workspace>(s.sched,std::move(groups),llama_memory_workspace_hooks{
            [&s] { s.executor.quiesce(); return true; }, [&s] { return s.invalidate(); }});
        if (!candidate->register_resources(target.plan,{1})) return false;
        std::unique_ptr<llama_memory_preparation> proposal;
        if (!candidate->prepare(target,layout,proposal) || !proposal) return false;
        if (!s.close()) return false;
        if (!proposal->quiesce({}) || !proposal->drain() || !proposal->invalidate() || !proposal->release() ||
                !proposal->bind(bindings) || !proposal->activate()) return false;
        proposal.reset();
        s.consumer=std::move(candidate);
        s.granted=std::move(grants);
        if (s.capture()) return true;
        s.close();
        return false;
    } catch (...) { return false; }
}

bool mtmd_compute_workspace::alloc_graph(ggml_cgraph * graph) {
    auto & s=*impl;
    if (!s.valid || s.busy || !graph || (!s.borrowed && (!s.consumer || !s.consumer->ready()))) return false;
    workspace_gate gate(s.busy);
    try {
        if (!s.invalidate() || (s.borrowed && !s.borrowed->prepare_serial_consumer(llama_memory_text_phase::prefill))) return false;
        std::vector<ggml_backend_memory_workspace_group> actual;
        if (!s.requirements(graph,actual)) return false;
        for (const auto & need : actual) {
            const auto grant=std::find_if(s.granted.begin(),s.granted.end(),[&](const auto & group) { return group.buft == need.buft; });
            if (grant == s.granted.end() || need.size > grant->size) return false;
        }
        ggml_backend_sched_reset(s.sched);
        if (!ggml_backend_sched_alloc_graph(s.sched,graph) || (!s.borrowed && !s.capture())) return false;
        s.graph=graph;
        return true;
    } catch (...) { return false; }
}

ggml_status mtmd_compute_workspace::compute_async(ggml_cgraph * graph) {
    auto & s=*impl;
    if (!s.valid || s.busy || !graph || graph != s.graph) return GGML_STATUS_FAILED;
    workspace_gate gate(s.busy);
    if (s.borrowed) return s.borrowed->compute_async(graph);
    if (!s.consumer || !s.executor.ready()) return GGML_STATUS_FAILED;
    if (!s.pending) s.pending=s.executor.acquire(s.consumer->leases(),s.revision);
    if (!s.pending) return GGML_STATUS_FAILED;
    try {
        const auto result=ggml_backend_sched_graph_compute_async(s.sched,graph);
        if (result != GGML_STATUS_SUCCESS) s.executor.quiesce();
        return result;
    } catch (...) { s.executor.quiesce(); return GGML_STATUS_FAILED; }
}

bool mtmd_compute_workspace::drain() {
    if (!impl->valid || impl->busy) return false;
    workspace_gate gate(impl->busy);
    try { return impl->drain(); } catch (...) { return false; }
}

bool mtmd_compute_workspace::release() {
    if (impl->busy) return false;
    if (!impl->valid) return true;
    workspace_gate gate(impl->busy);
    try { return impl->close(); } catch (...) { return false; }
}

bool mtmd_compute_workspace::retire_graph() {
    if (!impl->valid || impl->busy) return false;
    workspace_gate gate(impl->busy);
    try { return impl->invalidate(); } catch (...) { return false; }
}

bool mtmd_compute_workspace::supported() const noexcept { return impl->valid; }
bool mtmd_compute_workspace::ready() const noexcept {
    return impl->borrowed ? impl->borrowed->serial_ready() : impl->consumer && impl->consumer->ready();
}

bool mtmd_compute_workspace::borrow(llama_context_memory & parent) {
    auto & s=*impl;
    if (!s.valid || s.busy || s.measured.empty() || s.consumer || s.borrowed) return false;
    workspace_gate gate(s.busy);
    try {
        auto child=llama_context_memory::borrow_workspace(s.sched,s.backends,s.measured,parent);
        if (!child || !child->prepare_serial_consumer(llama_memory_text_phase::prefill)) return false;
        std::vector<ggml_backend_memory_workspace_group> grants=s.measured;
        s.borrowed=std::move(child);
        s.granted=std::move(grants);
        return true;
    } catch (...) { return false; }
}
