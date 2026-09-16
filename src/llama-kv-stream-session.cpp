#include "llama-kv-stream-session.h"
#include "llama-kv-stream-capture.h"
#include "llama-impl.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-kv-stream-device.h"
#include <array>
#include <numeric>
#include <cstring>

using lease_ptr = std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;

struct resident_graph {
    ggml_context_ptr context;
    ggml_tensor q{}, mask{}, output{};
    float scale = 0;
    std::unique_ptr<llama_kv_stream_cuda_executor> executor;
};

static bool same_tensor_storage(const ggml_tensor & a,const ggml_tensor & b) {
    return a.type == b.type && a.data == b.data && a.buffer == b.buffer &&
        !std::memcmp(a.ne,b.ne,sizeof(a.ne)) && !std::memcmp(a.nb,b.nb,sizeof(a.nb));
}

struct llama_kv_stream_session::implementation : llama_memory_executor_backend {
    ggml_backend_t backend = nullptr;
    std::shared_ptr<llama_kv_stream_content> content;
    llama_kv_stream_session_config config;
    std::unique_ptr<llama_kv_stream_publications> publications;
    llama_kv_stream_publication_ticket publication;
    std::vector<std::unique_ptr<llama_kv_stream_publication_pair>> publication_pairs;
    std::array<lease_ptr,3> leases{{{nullptr,ggml_backend_memory_lease_free},{nullptr,ggml_backend_memory_lease_free},{nullptr,ggml_backend_memory_lease_free}}};
    std::unique_ptr<llama_kv_stream_binding> binding;
    llama_kv_stream_resident * resident = nullptr;
    llama_memory_execution pin;
    llama_kv_stream_policy_state state;
    std::vector<uint32_t> order;
    std::vector<lease_ptr> graph_leases;
    std::vector<ggml_backend_memory_lease_t> graph_bindings;
    std::vector<std::unique_ptr<resident_graph>> graphs;
    size_t committed = 0, target = 0, grant = 0, span = 1;
    uint32_t next = 0, queries = 0;
    uint64_t revision = 0, expected_generation = 0;
    bool running = false, produced = false, poisoned = false, busy = false;
    bool direct_mode = false, report_layout = false;

    bool retire_publication() {
        bool released=true;
        for (auto & pair : publication_pairs) if (pair && pair->device_ready()) released &= pair->release_device(backend);
        publication_pairs.clear();
        if (!publication.pending()) return true;
        if (!publication.committed() && !publication.failed()) publication.cancel();
        return publication.retire() && released;
    }

    // End copy use before releasing the execution pin that owns native metadata.
    bool drain() override {
        if (resident) resident->cancel_sequence();
        ggml_backend_synchronize(backend);
        const bool retired = retire_publication();
        pin.reset();
        return retired;
    }
    // Prepare metadata first; publish policy only after the new writer binding succeeds.
    bool install(const llama_kv_stream_policy_state & candidate) {
        if (revision == UINT64_MAX) return false;
        auto replacement = std::make_unique<llama_kv_stream_binding>(content->host()->cache_id(),
            ggml_backend_buffer_get_type(ggml_backend_memory_lease_buffer(leases[0].get())));
        llama_kv_stream_resident * native = nullptr;
        if (!replacement->bind(leases[0].get(),config.policy,[&](const auto & view) {
            auto result = llama_kv_stream_resident::create(view,content,backend,&candidate);
            native = result.get(); return result;
        })) return false;
        graphs.clear();
        if (binding && binding->detach(*this).status != llama_memory_executor_status::retired) { poisoned = true; return false; }
        binding = std::move(replacement); resident = native;
        if (!resident->configure_writes(config.max_batch_rows,leases[1].get()) ||
                !resident->configure_feedback(config.measure) ||
                !resident->configure_native_graph_attention(config.native_graph_attention) ||
                !resident->configure_resumed_decode(config.resume_decode)) { poisoned = true; return false; }
        state = candidate; ++revision; return true;
    }

    // Capture owned leaf aliases, never the caller's model-graph metadata or unleased mutable storage.
    bool replay(uint32_t layer,ggml_tensor * q,ggml_tensor * mask,ggml_tensor * output,float scale) {
        if (graphs.empty()) graphs.resize(config.policy.layers);
        auto & cached = graphs[layer];
        if (cached && (!same_tensor_storage(cached->q,*q) || !same_tensor_storage(cached->mask,*mask) ||
                !same_tensor_storage(cached->output,*output) || cached->scale != scale || !cached->executor->ready(target))) cached.reset();
        if (!cached) {
            auto next = std::make_unique<resident_graph>();
            next->context.reset(ggml_init({65536,nullptr,true}));
            if (!next->context) return false;
            const auto alias = [&](const ggml_tensor * source) {
                auto * result = ggml_new_tensor(next->context.get(),source->type,4,source->ne);
                std::memcpy(result->nb,source->nb,sizeof(result->nb));
                return ggml_backend_tensor_alloc(source->buffer,result,source->data) == GGML_STATUS_SUCCESS ? result : nullptr;
            };
            auto * query = alias(q); auto * causal = alias(mask);
            if (!query || !causal) return false;
            auto * attention = resident->attention(next->context.get(),layer,query,causal,target,scale);
            if (!attention || ggml_backend_tensor_alloc(output->buffer,attention,output->data) != GGML_STATUS_SUCCESS) return false;
            auto * graph = ggml_new_graph_custom(next->context.get(),64,false); ggml_build_forward_expand(graph,attention);
            auto owner = binding->acquire();
            auto dependencies = graph_bindings; dependencies.push_back(leases[0].get());
            next->executor = std::make_unique<llama_kv_stream_cuda_executor>(backend);
            if (!next->executor->bind(*resident,owner,graph,dependencies,target)) return false;
            next->q = *q; next->mask = *mask; next->output = *output; next->scale = scale;
            cached = std::move(next);
        }
        return cached->executor->compute_async(target) == GGML_STATUS_SUCCESS;
    }
};

struct session_operation {
    bool & busy;
    explicit session_operation(bool & busy) : busy(busy) { busy = true; }
    ~session_operation() { busy = false; }
};

llama_kv_stream_session::llama_kv_stream_session() = default;
llama_kv_stream_session::~llama_kv_stream_session() {
    if (impl) {
        impl->drain(); impl->graphs.clear();
        if (impl->resident) impl->resident->release_write_workspace();
    }
}

// Validate all grants before native construction. The caller, not this consumer, allocates device storage.
std::unique_ptr<llama_kv_stream_session> llama_kv_stream_session::create(ggml_backend_t backend,
        std::shared_ptr<llama_kv_stream_content> content, const llama_kv_stream_session_config & config,
        ggml_backend_memory_lease_t pool, ggml_backend_memory_lease_t writer, ggml_backend_memory_lease_t partial) {
    if (!backend || !content || !config.max_batch_rows || config.max_batch_rows > INT32_MAX || !config.query_heads ||
            config.query_heads > SIZE_MAX/config.max_batch_rows || config.policy.shape.head_dim_k != 256 ||
            config.policy.shape.head_dim_v != 256 || config.policy.shape.heads <= 0 ||
            config.query_heads%config.policy.shape.heads) return {};
    auto * type = llama_kv_stream_device_buffer_type(ggml_backend_get_device(backend));
    if (!type || ggml_backend_buffer_get_type(content->host()->buffer()) != llama_kv_stream_host_buffer_type(ggml_backend_get_device(backend))) return {};
    llama_kv_stream_policy_state state;
    if (llama_kv_stream_policy_initialize(config.policy,state).status != llama_kv_stream_policy_status::success) return {};
    ggml_kv_stream_block_layout work;
    if (ggml_kv_stream_block_layout_make(size_t(config.query_heads)*config.max_batch_rows,256,work).status != ggml_kv_stream_partial_status::success) return {};
    const std::array<ggml_backend_memory_lease_t,3> grants{pool,writer,partial};
    std::array<ggml_backend_memory_region,3> regions{};
    std::array<uintptr_t,3> addresses{};
    size_t total = 0;
    for (size_t i = 0; i < grants.size(); ++i) {
        auto * buffer = ggml_backend_memory_lease_buffer(grants[i]);
        if (!buffer || !ggml_backend_memory_lease_get_region(grants[i],&regions[i]) || !regions[i].id || !regions[i].size ||
                ggml_backend_buffer_get_type(buffer) != type || ggml_backend_buffer_get_size(buffer) != regions[i].size) return {};
        addresses[i] = uintptr_t(ggml_backend_buffer_get_base(buffer));
        if (!addresses[i] || addresses[i]%128 || addresses[i] > UINTPTR_MAX-regions[i].size || regions[i].size > SIZE_MAX-total) return {};
        total += regions[i].size;
        for (size_t j = 0; j < i; ++j) if (regions[i].id == regions[j].id ||
                (addresses[i] < addresses[j]+regions[j].size && addresses[j] < addresses[i]+regions[i].size)) return {};
    }
    const size_t partial_bytes = config.native_graph_attention ? content->host()->layout().bytes : work.bytes;
    if (regions[0].size < config.policy.pool_bytes || regions[2].size < partial_bytes) return {};
    try {
        std::unique_ptr<llama_kv_stream_session> result(new llama_kv_stream_session);
        result->impl = std::make_unique<implementation>(); auto & s = *result->impl;
        s.backend = backend; s.content = std::move(content); s.config = config; s.grant = total;
        for (size_t i = 0; i < grants.size(); ++i) s.leases[i].reset(ggml_backend_memory_lease_retain(grants[i]));
        s.order.resize(config.policy.layers); std::iota(s.order.begin(),s.order.end(),0);
        s.expected_generation = s.content->generation();
        if (!s.install(state)) return {};
        s.publications = llama_kv_stream_publications::create({0,s.content->host()->config().context_tokens,s.expected_generation,1});
        if (!s.publications) return {};
        return result;
    } catch (const std::bad_alloc &) { return {}; }
}

// A failed native transition closes admission; host bytes and leases remain owned for safe teardown.
bool llama_kv_stream_session::begin(size_t active, uint32_t queries, bool decode) {
    auto & s = *impl;
    if (s.busy || s.running || s.poisoned || !s.publications || s.publications->failed() || s.publication.pending() || !s.leases[2] ||
            !queries || queries > s.config.max_batch_rows || (decode && queries != 1) ||
            active < s.committed || active-s.committed != queries || active > s.content->host()->config().context_tokens) return false;
    session_operation guard(s.busy);
    if (s.content->generation() != s.expected_generation) { s.poisoned = true; return false; }
    llama_kv_stream_policy_decision decision;
    if (!s.resident->recommend_policy(s.state,active,queries,decision,decode,!decode)) return false;
    try {
        if (decision.layout_changed) {
            if (!s.install(decision.next)) return false;
        } else s.state = decision.next;
        if (decision.layout_changed) {
            const size_t page_tokens=size_t(s.state.budget.shape.page_tokens);
            const size_t active_pages=(active-1)/page_tokens+1;
            const size_t streamed=active_pages > s.state.resident_pages_per_layer ?
                (active_pages-s.state.resident_pages_per_layer)*s.state.budget.layers : 0;
            const double mib=double(streamed)*s.state.budget.page.storage.bytes/1048576.0;
            LLAMA_LOG_WARN("%s: KV layout revision %llu, resident pages/layer %u, ring slots %u, active pages %zu, padded H2D %.2f MiB/eval\n",
                __func__,(unsigned long long)s.revision,s.state.resident_pages_per_layer,s.state.ring_slots,active_pages,mib);
            s.report_layout = true;
        }
        s.span = std::max(size_t(1),s.resident->suggested_span_pages());
        s.pin = s.binding->acquire();
        llama_kv_stream_capture_stamp stamp;
        s.direct_mode = queries == 1 && s.config.native_graph_attention && !s.graph_bindings.empty() &&
            s.resident->capture_state(s.backend,active,stamp);
        if (!s.direct_mode) s.graphs.clear();
        if (!s.pin || !(s.direct_mode ? s.resident->synchronize(active) : s.resident->begin_sequence(s.order,active,s.span,s.committed,{queries,decode}))) {
            s.drain(); s.poisoned = true; return false;
        }
        s.publication_pairs.clear();
        s.publication_pairs.reserve(s.order.size());
        llama_kv_stream_publication_ticket publication;
        std::vector<std::shared_ptr<void>> owners{s.content};
        std::vector<ggml_backend_memory_lease_t> dependencies;
        dependencies.reserve(s.leases.size());
        for (const auto & lease : s.leases) dependencies.push_back(lease.get());
        const auto frontiers = s.publications->frontiers();
        if (frontiers.reserved != s.committed || frontiers.committed != s.committed ||
                !s.publications->reserve(s.committed,queries,s.config.policy.layers,owners,dependencies,publication)) {
            s.drain(); s.poisoned = true; return false;
        }
        s.publication = std::move(publication);
        s.running = true; s.produced = false; s.target = active; s.queries = queries; s.next = 0;
        return true;
    } catch (...) { s.drain(); s.poisoned = true; throw; }
}

bool llama_kv_stream_session::restore(size_t tokens) {
    auto & s = *impl;
    if (s.busy || s.running || s.poisoned || s.committed || tokens > s.content->host()->config().context_tokens) return false;
    session_operation guard(s.busy);
    if (s.content->generation() != s.expected_generation) return false;
    auto publications = llama_kv_stream_publications::create(
        {tokens,s.content->host()->config().context_tokens,s.expected_generation,1});
    if (!publications) return false;
    s.committed = tokens;
    s.target = tokens;
    s.graphs.clear();
    s.publications = std::move(publications);
    return true;
}

// Pair publication is a required dependency of each layer's attention call.
bool llama_kv_stream_session::produce(uint32_t layer, const ggml_tensor * k, const ggml_tensor * v) {
    auto & s = *impl;
    if (s.busy || !s.running || s.poisoned || s.produced || layer != s.next) return false;
    session_operation guard(s.busy);
    try {
        const bool valid=k && v && k->ne[1] == s.queries && v->ne[1] == s.queries;
        std::unique_ptr<llama_kv_stream_publication_pair> pair;
        if (!valid || !s.resident->prepare_write_pair(layer,s.committed,k,v,s.publication,s.next,pair) ||
                !pair || !pair->wait_device(s.backend)) {
            s.drain(); s.running = false; s.poisoned = true; return false;
        }
        s.produced=pair->device_ready() && s.publication.ready(s.next,llama_kv_stream_publication_domain::device);
        s.publication_pairs.push_back(std::move(pair));
        return s.produced;
    } catch (...) { s.drain(); s.running = false; s.poisoned = true; throw; }
}

// Finish all query tiles before advancing the serial layer cursor or committing the append frontier.
bool llama_kv_stream_session::attention(uint32_t layer, ggml_tensor * q, ggml_tensor * mask, ggml_tensor * output, float scale) {
    auto & s = *impl;
    if (s.busy || !s.running || s.poisoned || !s.produced || layer != s.next ||
            s.publication_pairs.empty() || !s.publication_pairs.back()->device_ready()) return false;
    session_operation guard(s.busy);
    try {
        if (!q || !mask || !output || q->ne[1] != s.queries || q->ne[2] != s.config.query_heads ||
                !(s.direct_mode ? s.replay(layer,q,mask,output,scale) :
                  s.resident->compute_streamed(layer,q,mask,output,s.target,scale,s.leases[2].get(),true,s.span))) {
            s.drain(); s.running = false; s.poisoned = true; return false;
        }
        if (!s.publication_pairs.back()->publish_host()) {
            s.drain(); s.running = false; s.poisoned = true; return false;
        }
        s.produced = false;
        if (++s.next == s.order.size()) {
            if (!s.publication.committed() || s.publications->frontiers().committed != s.target) {
                s.drain(); s.running = false; s.poisoned = true; return false;
            }
            ggml_backend_synchronize(s.backend);
            for (auto & pair : s.publication_pairs) if (!pair->release_device(s.backend)) {
                s.drain(); s.running = false; s.poisoned = true; return false;
            }
            s.publication_pairs.clear();
            if (s.report_layout) {
                const auto stats=s.resident->sequence_stats();
                LLAMA_LOG_WARN("%s: accepted KV layout copied %.2f MiB in %zu H2D calls, peak ring pages %zu\n",
                    __func__,stats.copy_bytes/1048576.0,stats.copy_calls,stats.peak_pages);
                s.report_layout = false;
            }
            if (!s.publication.retire()) {
                s.drain(); s.running = false; s.poisoned = true; return false;
            }
            s.committed = s.target; s.expected_generation = s.content->generation(); s.running = false;
            s.pin.reset();
        }
        return true;
    } catch (...) { s.drain(); s.running = false; s.poisoned = true; throw; }
}

// Cancellation is not inference-state rollback, even when no KV token frontier was committed yet.
void llama_kv_stream_session::abort() {
    auto & s = *impl;
    if (s.busy) return;
    session_operation guard(s.busy); s.drain(); s.running = false; s.poisoned = true; s.graphs.clear();
}
bool llama_kv_stream_session::active() const noexcept { return impl->running; }
bool llama_kv_stream_session::failed() const noexcept { return impl->poisoned; }
size_t llama_kv_stream_session::tokens() const noexcept { return impl->committed; }
size_t llama_kv_stream_session::granted_bytes() const noexcept { return impl->grant; }
uint64_t llama_kv_stream_session::layout_revision() const noexcept { return impl->revision; }
const llama_kv_stream_policy_state & llama_kv_stream_session::policy() const noexcept { return impl->state; }

llama_kv_stream_publication_frontiers llama_kv_stream_session::publication_frontiers() const noexcept {
    return impl->publications ? impl->publications->frontiers() : llama_kv_stream_publication_frontiers{};
}

bool llama_kv_stream_session::set_workspaces(const std::vector<ggml_backend_memory_lease_t> & workspaces) {
    auto & s = *impl;
    if (s.busy || s.running) return false;
    if (s.graph_bindings == workspaces) return true;
    session_operation guard(s.busy);
    std::vector<lease_ptr> retained;
    for (auto * lease : workspaces) {
        ggml_backend_memory_region region;
        if (!ggml_backend_memory_lease_get_region(lease,&region)) return false;
        retained.emplace_back(ggml_backend_memory_lease_retain(lease),ggml_backend_memory_lease_free);
    }
    auto bindings = workspaces;
    s.graphs.clear();
    s.resident->release_write_workspace();
    if (!bindings.empty()) {
        if (!s.resident->configure_writes(s.config.max_batch_rows,s.leases[1].get())) { s.poisoned = true; return false; }
    }
    s.graph_leases = std::move(retained); s.graph_bindings = std::move(bindings); return true;
}
void llama_kv_stream_session::release_graphs() { if (!impl->busy) impl->graphs.clear(); }
bool llama_kv_stream_session::set_attention_workspace(ggml_backend_memory_lease_t lease, bool decode) {
    auto & s = *impl;
    if (s.busy || s.running || s.poisoned) return false;
    size_t bytes = 0;
    if (lease) {
        auto * buffer = ggml_backend_memory_lease_buffer(lease);
        ggml_backend_memory_region region;
        if (!buffer || !ggml_backend_memory_lease_get_region(lease,&region) || !region.id ||
                ggml_backend_buffer_get_type(buffer) != ggml_backend_buffer_get_type(ggml_backend_memory_lease_buffer(s.leases[0].get()))) return false;
        const auto base = uintptr_t(ggml_backend_buffer_get_base(buffer));
        bytes = ggml_backend_buffer_get_size(buffer);
        if (!base || base%128 || bytes != region.size || base > UINTPTR_MAX-bytes) return false;
        size_t needed = s.content->host()->layout().bytes;
        if (!s.config.native_graph_attention) {
            ggml_kv_stream_block_layout partial;
            if (ggml_kv_stream_block_layout_make(size_t(s.config.query_heads)*s.config.max_batch_rows,256,partial).status != ggml_kv_stream_partial_status::success) return false;
            needed = partial.bytes;
        }
        if (decode && s.config.resume_decode) {
            auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(s.backend)),"ggml_backend_kv_stream_partial_ops"));
            ggml_kv_stream_resume_plan plan;
            if (!get || !get() || get()->version < 5 || !get()->resume_plan || !get()->resume_plan(s.backend,
                    s.config.policy.shape.type_k,s.config.policy.shape.type_v,s.config.query_heads,s.config.policy.shape.heads,
                    s.content->host()->layout().tokens,plan)) return false;
            needed = plan.bytes;
        }
        if (bytes < needed) return false;
        for (size_t i = 0; i < 2; ++i) {
            ggml_backend_memory_region other; ggml_backend_memory_lease_get_region(s.leases[i].get(),&other);
            auto * b = ggml_backend_memory_lease_buffer(s.leases[i].get());
            const auto p = uintptr_t(ggml_backend_buffer_get_base(b));
            if (region.id == other.id || (base < p+other.size && p < base+bytes)) return false;
        }
    }
    session_operation guard(s.busy);
    auto retained = lease_ptr(lease ? ggml_backend_memory_lease_retain(lease) : nullptr,ggml_backend_memory_lease_free);
    s.drain(); s.leases[2] = std::move(retained);
    s.grant = bytes;
    for (size_t i = 0; i < 2; ++i) s.grant += ggml_backend_buffer_get_size(ggml_backend_memory_lease_buffer(s.leases[i].get()));
    return true;
}
size_t llama_kv_stream_session::captured_layers() const {
    size_t count = 0; for (const auto & graph : impl->graphs) if (graph && graph->executor->is_captured()) ++count; return count;
}
