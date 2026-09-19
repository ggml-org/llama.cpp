#include "llama-kv-stream-session.h"
#include "llama-kv-stream-capture.h"
#include "llama-impl.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-kv-stream-device.h"
#include <algorithm>
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
    struct pool_rebind;
    struct pool_candidate {
        std::array<lease_ptr,3> leases{{
            {nullptr,ggml_backend_memory_lease_free},
            {nullptr,ggml_backend_memory_lease_free},
            {nullptr,ggml_backend_memory_lease_free},
        }};
        std::unique_ptr<llama_kv_stream_binding> binding;
        llama_kv_stream_resident * resident = nullptr;
        llama_kv_stream_policy_config policy;
        llama_kv_stream_policy_state state;
        size_t grant = 0;
    };
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
    bool direct_mode = false, report_layout = false, primed = false;
    size_t primed_active = 0;
    pool_rebind * transition = nullptr;
    bool transition_closed = false;

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
        primed = false; primed_active = 0;
        ggml_backend_synchronize(backend);
        const bool retired = retire_publication();
        pin.reset();
        return retired;
    }

    // Start only a layout-stable next-token decode; failure is an optimization miss, not session failure.
    bool prime_next() {
        if (!config.cross_token_prefetch || !config.resume_decode || primed || poisoned || !resident || !binding ||
                committed >= content->host()->config().context_tokens || !state.ring_slots) return false;
        const size_t active=committed+1;
        llama_kv_stream_policy_decision decision;
        if (!resident->recommend_policy(state,active,1,decision,true,false) || decision.layout_changed ||
                decision.next.decode_active_pages <= decision.next.resident_pages_per_layer) return false;
        const size_t next_span=std::max(size_t(1),resident->suggested_span_pages());
        if (!pin) pin=binding->acquire();
        if (!pin || !resident->prime_sequence(order,active,next_span,committed,{1,true})) {
            pin.reset(); return false;
        }
        state=decision.next; span=next_span; primed=true; primed_active=active;
        return true;
    }
    bool attention_bytes(bool decode, size_t & bytes) const {
        bytes = content->host()->layout().bytes;
        if (!decode || !config.resume_decode) return true;
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)),
                "ggml_backend_kv_stream_partial_ops"));
        ggml_kv_stream_resume_plan plan;
        if (!get || !get() || get()->version < 5 || !get()->resume_plan ||
                !get()->resume_plan(backend,config.policy.shape.type_k,
                    config.policy.shape.type_v,config.query_heads,
                    config.policy.shape.heads,content->host()->layout().tokens,plan)) return false;
        bytes = plan.bytes;
        return true;
    }

    // Construct a complete idle candidate without changing or retiring the active binding.
    bool build_pool(const llama_kv_stream_policy_state & candidate,
            const llama_kv_stream_policy_config & policy,
            ggml_backend_memory_lease_t pool,
            ggml_backend_memory_lease_t writer,
            ggml_backend_memory_lease_t attention,
            uint64_t previous_revision, pool_candidate & output) {
        if (previous_revision == UINT64_MAX || !pool || !writer || !attention) return false;
        const std::array<ggml_backend_memory_lease_t,3> grants{pool,writer,attention};
        pool_candidate next;
        for (size_t i = 0; i < grants.size(); ++i) {
            auto * buffer = ggml_backend_memory_lease_buffer(grants[i]);
            if (!buffer) return false;
            next.leases[i].reset(ggml_backend_memory_lease_retain(grants[i]));
            if (!next.leases[i]) return false;
            const size_t bytes = ggml_backend_buffer_get_size(buffer);
            if (bytes > SIZE_MAX-next.grant) return false;
            next.grant += bytes;
        }
        auto * buffer = ggml_backend_memory_lease_buffer(pool);
        next.binding = std::make_unique<llama_kv_stream_binding>(
            content->host()->cache_id(),ggml_backend_buffer_get_type(buffer),previous_revision);
        if (!next.binding->bind(pool,policy,[&](const auto & view) {
            auto result = llama_kv_stream_resident::create(view,content,backend,&candidate);
            next.resident = result.get(); return result;
        }) || !next.resident->configure_writes(config.max_batch_rows,next.leases[1].get()) ||
                !next.resident->configure_feedback(config.measure) ||
                !next.resident->configure_native_graph_attention(config.native_graph_attention) ||
                !next.resident->configure_resumed_decode(config.resume_decode)) return false;
        next.policy = policy;
        next.state = candidate;
        output = std::move(next);
        return true;
    }

    // Publish only a fully configured candidate; its binding revision is the device validity identity.
    void publish_pool(pool_candidate && candidate) {
        leases = std::move(candidate.leases);
        binding = std::move(candidate.binding);
        resident = candidate.resident;
        config.policy = candidate.policy;
        state = candidate.state;
        grant = candidate.grant;
        revision = binding->view()->revision;
    }

    // Prepare and configure replacement metadata before retiring the active device binding.
    bool install(const llama_kv_stream_policy_state & candidate,
            const llama_kv_stream_policy_config & policy, ggml_backend_memory_lease_t pool) {
        pool_candidate replacement;
        if (!build_pool(candidate,policy,pool,leases[1].get(),leases[2].get(),revision,replacement)) return false;
        graphs.clear();
        if (binding && binding->detach(*this).status != llama_memory_executor_status::retired) {
            poisoned = true; return false;
        }
        publish_pool(std::move(replacement));
        return true;
    }

    bool install(const llama_kv_stream_policy_state & candidate) {
        return install(candidate,config.policy,leases[0].get());
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


static bool same_pool_region(
        const ggml_backend_memory_region & a, const ggml_backend_memory_region & b) {
    return a.id == b.id && a.offset == b.offset && a.size == b.size &&
        a.alignment == b.alignment && a.flags == b.flags;
}

struct llama_kv_stream_session::implementation::pool_rebind : llama_memory_preparation {
    implementation & owner;
    std::array<llama_memory_resource_id,3> resources;
    size_t arena;
    std::array<ggml_backend_memory_region,3> before_regions;
    std::array<ggml_backend_memory_region,3> desired_regions;
    llama_kv_stream_policy_config before_policy;
    llama_kv_stream_policy_state before_state;
    llama_kv_stream_policy_config desired_policy;
    llama_kv_stream_policy_state desired_state;
    size_t before_committed;
    uint64_t before_generation;
    uint64_t before_revision;
    pool_candidate candidate;
    bool coordinated_scratch;
    bool affected = false;
    bool activated = false;
    bool recovered = false;

    pool_rebind(implementation & owner,
            std::array<llama_memory_resource_id,3> resources,
            size_t arena,
            std::array<ggml_backend_memory_region,3> before_regions,
            std::array<ggml_backend_memory_region,3> desired_regions,
            llama_kv_stream_policy_config desired_policy,
            llama_kv_stream_policy_state desired_state,
            bool coordinated_scratch) :
        owner(owner), resources(resources), arena(arena),
        before_regions(std::move(before_regions)),
        desired_regions(std::move(desired_regions)),
        before_policy(owner.config.policy), before_state(owner.state),
        desired_policy(std::move(desired_policy)),
        desired_state(std::move(desired_state)),
        before_committed(owner.committed),
        before_generation(owner.expected_generation),
        before_revision(owner.revision),
        coordinated_scratch(coordinated_scratch) {
        GGML_ASSERT(owner.transition == nullptr);
        owner.transition = this;
    }

    ~pool_rebind() override {
        if (owner.transition != this) return;
        if (affected && !recovered &&
                (!activated || !owner.binding || !owner.binding->ready())) owner.poisoned = true;
        owner.transition_closed = false;
        owner.transition = nullptr;
    }

    bool frontier_unchanged() const {
        if (!owner.publications || owner.publications->failed() ||
                owner.publication.pending() || owner.committed != before_committed ||
                owner.expected_generation != before_generation ||
                owner.content->generation() != before_generation) return false;
        const auto frontiers = owner.publications->frontiers();
        return frontiers.reserved == before_committed &&
            frontiers.device == before_committed &&
            frontiers.host == before_committed &&
            frontiers.committed == before_committed;
    }

    bool validate_lease(const llama_memory_region_binding & supplied,
            llama_memory_resource_id resource,
            const ggml_backend_memory_region & expected) const {
        if (supplied.arena != arena || supplied.region.id != resource ||
                !same_pool_region(supplied.region,expected) || !supplied.lease) return false;
        ggml_backend_memory_region actual;
        auto * buffer = ggml_backend_memory_lease_buffer(supplied.lease);
        auto * type = llama_kv_stream_device_buffer_type(ggml_backend_get_device(owner.backend));
        if (!buffer || !type || !ggml_backend_memory_lease_get_region(supplied.lease,&actual) ||
                !same_pool_region(actual,expected) ||
                ggml_backend_buffer_get_type(buffer) != type ||
                ggml_backend_buffer_get_size(buffer) != expected.size) return false;
        const auto base = uintptr_t(ggml_backend_buffer_get_base(buffer));
        return base && base%owner.config.policy.shape.alignment == 0 &&
            base <= UINTPTR_MAX-expected.size;
    }

    const llama_memory_region_binding * find_binding(
            const std::vector<llama_memory_region_binding> & bindings,
            llama_memory_resource_id resource,
            const ggml_backend_memory_region & expected) const {
        const llama_memory_region_binding * found = nullptr;
        for (const auto & binding : bindings) {
            if (binding.region.id != resource) continue;
            if (found || !validate_lease(binding,resource,expected)) return nullptr;
            found = &binding;
        }
        return found;
    }

    bool validate_grants(const std::array<ggml_backend_memory_lease_t,3> & grants) const {
        std::array<uintptr_t,3> addresses{};
        std::array<size_t,3> sizes{};
        for (size_t i = 0; i < grants.size(); ++i) {
            auto * buffer = ggml_backend_memory_lease_buffer(grants[i]);
            if (!buffer) return false;
            addresses[i] = uintptr_t(ggml_backend_buffer_get_base(buffer));
            sizes[i] = ggml_backend_buffer_get_size(buffer);
            if (!addresses[i] || addresses[i] > UINTPTR_MAX-sizes[i]) return false;
            for (size_t j = 0; j < i; ++j) {
                if (addresses[i] < addresses[j]+sizes[j] &&
                        addresses[j] < addresses[i]+sizes[i]) return false;
            }
        }
        return true;
    }

    void retain_external_grant_only() {
        owner.grant = 0;
        for (const auto & lease : owner.leases) {
            auto * buffer = ggml_backend_memory_lease_buffer(lease.get());
            const size_t bytes = buffer ? ggml_backend_buffer_get_size(buffer) : 0;
            if (bytes > SIZE_MAX-owner.grant) {
                owner.poisoned = true;
                owner.grant = 0;
                return;
            }
            owner.grant += bytes;
        }
    }

    bool quiesce(const std::vector<llama_memory_resource_id> & changed) override {
        if (owner.transition != this || owner.busy || owner.running || owner.poisoned ||
                !owner.binding || !owner.binding->ready() ||
                owner.revision != before_revision || !frontier_unchanged() ||
                std::find(changed.begin(),changed.end(),resources[0]) == changed.end()) return false;
        affected = true;
        owner.transition_closed = true;
        owner.binding->quiesce();
        return true;
    }

    bool drain() override {
        return !affected || owner.drain();
    }

    bool invalidate() override {
        if (!affected) return true;
        owner.graphs.clear();
        owner.graph_leases.clear();
        owner.graph_bindings.clear();
        if (!owner.binding) return true;
        if (owner.binding->detach(owner).status != llama_memory_executor_status::retired) return false;
        owner.resident = nullptr;
        return true;
    }

    bool release() override {
        if (!affected || owner.resident) return !affected;
        owner.binding.reset();
        owner.leases[0].reset();
        if (coordinated_scratch) {
            owner.leases[1].reset();
            owner.leases[2].reset();
        }
        retain_external_grant_only();
        return true;
    }

    bool bind(const std::vector<llama_memory_region_binding> & bindings) override {
        if (!affected || owner.binding || owner.leases[0] || candidate.binding ||
                !frontier_unchanged()) return false;
        const auto * pool = find_binding(bindings,resources[0],desired_regions[0]);
        const auto * writer = coordinated_scratch ?
            find_binding(bindings,resources[1],desired_regions[1]) : nullptr;
        const auto * attention = coordinated_scratch ?
            find_binding(bindings,resources[2],desired_regions[2]) : nullptr;
        const std::array<ggml_backend_memory_lease_t,3> grants{
            pool ? pool->lease : nullptr,
            coordinated_scratch ? (writer ? writer->lease : nullptr) : owner.leases[1].get(),
            coordinated_scratch ? (attention ? attention->lease : nullptr) : owner.leases[2].get(),
        };
        return pool && (!coordinated_scratch || (writer && attention)) &&
            validate_grants(grants) && owner.build_pool(
                desired_state,desired_policy,grants[0],grants[1],grants[2],
                owner.revision,candidate);
    }

    bool activate() override {
        if (!affected || !candidate.binding || !frontier_unchanged()) return false;
        owner.publish_pool(std::move(candidate));
        activated = true;
        return true;
    }

    struct restoration : llama_memory_preparation {
        pool_rebind & original;
        pool_candidate candidate;

        explicit restoration(pool_rebind & original) : original(original) {}

        bool fail() {
            original.owner.poisoned = true;
            return false;
        }

        bool quiesce(const std::vector<llama_memory_resource_id> &) override {
            auto & owner = original.owner;
            owner.transition_closed = true;
            if (owner.binding) owner.binding->quiesce();
            if (original.candidate.binding) original.candidate.binding->quiesce();
            return true;
        }

        bool drain() override {
            return original.owner.drain() || fail();
        }

        bool invalidate() override {
            auto & owner = original.owner;
            owner.graphs.clear();
            owner.graph_leases.clear();
            owner.graph_bindings.clear();
            if (owner.binding &&
                    owner.binding->detach(owner).status != llama_memory_executor_status::retired) return fail();
            owner.resident = nullptr;
            if (original.candidate.binding &&
                    original.candidate.binding->detach(owner).status != llama_memory_executor_status::retired) return fail();
            original.candidate.resident = nullptr;
            return true;
        }

        bool release() override {
            auto & owner = original.owner;
            owner.binding.reset();
            owner.leases[0].reset();
            if (original.coordinated_scratch) {
                owner.leases[1].reset();
                owner.leases[2].reset();
            }
            original.candidate = {};
            original.retain_external_grant_only();
            return !owner.poisoned;
        }

        bool bind(const std::vector<llama_memory_region_binding> & bindings) override {
            auto & owner = original.owner;
            if (owner.content->generation() != original.before_generation ||
                    owner.expected_generation != original.before_generation) return fail();
            const auto * pool = original.find_binding(
                bindings,original.resources[0],original.before_regions[0]);
            const auto * writer = original.coordinated_scratch ? original.find_binding(
                bindings,original.resources[1],original.before_regions[1]) : nullptr;
            const auto * attention = original.coordinated_scratch ? original.find_binding(
                bindings,original.resources[2],original.before_regions[2]) : nullptr;
            const std::array<ggml_backend_memory_lease_t,3> grants{
                pool ? pool->lease : nullptr,
                original.coordinated_scratch ?
                    (writer ? writer->lease : nullptr) : owner.leases[1].get(),
                original.coordinated_scratch ?
                    (attention ? attention->lease : nullptr) : owner.leases[2].get(),
            };
            if (!pool || (original.coordinated_scratch && (!writer || !attention)) ||
                    !original.validate_grants(grants) || !owner.build_pool(
                        original.before_state,original.before_policy,
                        grants[0],grants[1],grants[2],owner.revision,candidate)) return fail();
            return true;
        }

        bool activate() override {
            auto & owner = original.owner;
            if (!candidate.binding || owner.content->generation() != original.before_generation) return fail();
            owner.publish_pool(std::move(candidate));
            original.recovered = true;
            owner.transition_closed = false;
            return true;
        }
    };

    bool prepare_recovery(std::unique_ptr<llama_memory_preparation> & output) override {
        if (!affected) return true;
        if (owner.content->generation() != before_generation ||
                owner.expected_generation != before_generation) {
            owner.poisoned = true;
            return false;
        }
        output = std::make_unique<restoration>(*this);
        return true;
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
    const bool coordinated_scratch = config.writer_resource || config.attention_resource;
    const bool any_transition = config.pool_resource || coordinated_scratch ||
        config.prefill_stage || config.decode_stage;
    if (any_transition && (!config.pool_resource || !config.prefill_stage || !config.decode_stage ||
            config.prefill_stage == config.decode_stage ||
            (coordinated_scratch && (!config.writer_resource || !config.attention_resource ||
                config.writer_resource == config.pool_resource ||
                config.attention_resource == config.pool_resource ||
                config.writer_resource == config.attention_resource)))) return {};
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
    size_t partial_bytes = config.native_graph_attention ? content->host()->layout().bytes : work.bytes;
    if (config.initial_decode && config.resume_decode) {
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)),
                "ggml_backend_kv_stream_partial_ops"));
        ggml_kv_stream_resume_plan plan;
        if (!get || !get() || get()->version < 5 || !get()->resume_plan ||
                !get()->resume_plan(backend,config.policy.shape.type_k,
                    config.policy.shape.type_v,config.query_heads,
                    config.policy.shape.heads,content->host()->layout().tokens,plan)) return {};
        partial_bytes = plan.bytes;
    }
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
    if (s.busy || s.transition_closed || s.running || s.poisoned || !s.publications || s.publications->failed() || s.publication.pending() || !s.leases[2] ||
            !queries || queries > s.config.max_batch_rows || (decode && queries != 1) ||
            active < s.committed || active-s.committed != queries || active > s.content->host()->config().context_tokens) return false;
    session_operation guard(s.busy);
    if (s.content->generation() != s.expected_generation) { s.drain(); s.poisoned = true; return false; }
    try {
        bool adopted=false;
        if (s.primed) {
            adopted=decode && queries == 1 && active == s.primed_active && bool(s.pin) &&
                s.resident->adopt_sequence(s.order,active,s.span,s.committed,{queries,decode});
            if (!adopted) {
                s.resident->cancel_sequence(); s.pin.reset();
            }
            s.primed=false; s.primed_active=0;
        }
        if (!adopted) {
            llama_kv_stream_policy_decision decision;
            if (!s.resident->recommend_policy(s.state,active,queries,decision,decode,!decode)) return false;
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
            if (s.direct_mode) {
                if (!s.pin || !s.resident->synchronize(active)) {
                    s.drain(); s.poisoned = true; return false;
                }
            } else if (!s.pin || !s.resident->begin_sequence(s.order,active,s.span,s.committed,{queries,decode})) {
                s.drain(); s.poisoned = true; return false;
            }
        } else {
            s.direct_mode=false;
        }
        if (!s.direct_mode) s.graphs.clear();
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
    if (s.busy || s.transition_closed || s.running || s.poisoned || s.committed || tokens > s.content->host()->config().context_tokens) return false;
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


bool llama_kv_stream_session::reconstruct(size_t tokens) {
    auto & s = *impl;
    if (s.busy || s.transition || s.transition_closed || s.running ||
            tokens > s.content->host()->config().context_tokens) return false;
    session_operation guard(s.busy);
    if (!s.drain()) {
        s.poisoned = true;
        return false;
    }
    const uint64_t generation = s.content->generation();
    auto publications = llama_kv_stream_publications::create(
        {tokens,s.content->host()->config().context_tokens,generation,1});
    if (!publications) {
        s.poisoned = true;
        return false;
    }
    s.graphs.clear();
    s.publication_pairs.clear();
    s.publication = {};
    s.publications = std::move(publications);
    s.committed = s.target = tokens;
    s.next = s.queries = 0;
    s.produced = s.running = s.direct_mode = s.report_layout = false;
    s.expected_generation = generation;
    s.poisoned = false;
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
                const auto timing=s.resident->copy_feedback();
                LLAMA_LOG_WARN("%s: accepted KV layout copied %.2f MiB in %zu H2D calls, peak ring pages %zu, sampled copy %.3f ms over %.3f ms\n",
                    __func__,stats.copy_bytes/1048576.0,stats.copy_calls,stats.peak_pages,
                    timing.copy_ms,timing.elapsed_ms);
                s.report_layout = false;
            }
            if (!s.publication.retire()) {
                s.drain(); s.running = false; s.poisoned = true; return false;
            }
            s.committed = s.target; s.expected_generation = s.content->generation(); s.running = false;
            if (!s.prime_next()) s.pin.reset();
        }
        return true;
    } catch (...) { s.drain(); s.running = false; s.poisoned = true; throw; }
}

// Cancellation is not inference-state rollback, even when no KV token frontier was committed yet.
void llama_kv_stream_session::abort() {
    auto & s = *impl;
    if (s.busy || s.transition_closed) return;
    session_operation guard(s.busy); s.drain(); s.running = false; s.poisoned = true; s.graphs.clear();
}
bool llama_kv_stream_session::active() const noexcept { return impl->running; }
bool llama_kv_stream_session::failed() const noexcept { return impl->poisoned; }
size_t llama_kv_stream_session::tokens() const noexcept { return impl->committed; }
size_t llama_kv_stream_session::granted_bytes() const noexcept { return impl->grant; }
uint64_t llama_kv_stream_session::layout_revision() const noexcept { return impl->revision; }
bool llama_kv_stream_session::prefetch_primed() const noexcept { return impl->primed; }

bool llama_kv_stream_session::grow_pool(
        ggml_backend_memory_lease_t pool, size_t pool_bytes, bool decode) {
    return rebind_pool(pool,pool_bytes,decode,true);
}

bool llama_kv_stream_session::shrink_pool(
        ggml_backend_memory_lease_t pool, size_t pool_bytes, bool decode) {
    return rebind_pool(pool,pool_bytes,decode,false);
}

bool llama_kv_stream_session::rebind_pool(
        ggml_backend_memory_lease_t pool, size_t pool_bytes, bool decode, bool growing) {
    auto & s = *impl;
    if (s.busy || s.transition || s.transition_closed || s.running || s.poisoned || !s.publications || s.publications->failed() ||
            !pool || s.publication.pending() || s.content->generation() != s.expected_generation ||
            (growing ? pool_bytes <= s.config.policy.pool_bytes :
                       pool_bytes >= s.config.policy.pool_bytes)) return false;
    const auto frontiers = s.publications->frontiers();
    if (frontiers.reserved != s.committed || frontiers.device != s.committed ||
            frontiers.host != s.committed || frontiers.committed != s.committed) return false;
    auto * buffer = ggml_backend_memory_lease_buffer(pool);
    auto * current = ggml_backend_memory_lease_buffer(s.leases[0].get());
    ggml_backend_memory_region region;
    if (!buffer || !current || !ggml_backend_memory_lease_get_region(pool,&region) ||
            !region.id || region.size < pool_bytes ||
            ggml_backend_buffer_get_type(buffer) != ggml_backend_buffer_get_type(current) ||
            ggml_backend_buffer_get_size(buffer) != region.size) return false;
    const auto base = uintptr_t(ggml_backend_buffer_get_base(buffer));
    ggml_backend_memory_region old_region;
    if (!base || base%s.config.policy.shape.alignment || base > UINTPTR_MAX-region.size ||
            !ggml_backend_memory_lease_get_region(s.leases[0].get(),&old_region)) return false;
    const auto old_base = uintptr_t(ggml_backend_buffer_get_base(current));
    if (!old_base || old_base > UINTPTR_MAX-old_region.size || region.id == old_region.id ||
            (base < old_base+old_region.size && old_base < base+region.size)) return false;
    for (size_t i = 1; i < s.leases.size(); ++i) {
        ggml_backend_memory_region other;
        auto * other_buffer = ggml_backend_memory_lease_buffer(s.leases[i].get());
        if (!other_buffer || !ggml_backend_memory_lease_get_region(s.leases[i].get(),&other)) return false;
        const auto address = uintptr_t(ggml_backend_buffer_get_base(other_buffer));
        if (!address || address > UINTPTR_MAX-other.size || region.id == other.id ||
                (base < address+other.size && address < base+region.size)) return false;
    }
    llama_kv_stream_policy_rebind replacement;
    const auto planned = growing ?
        llama_kv_stream_policy_grow(
            s.config.policy,s.committed,decode,pool_bytes,replacement) :
        llama_kv_stream_policy_shrink(
            s.config.policy,s.committed,decode,pool_bytes,replacement);
    if (planned.status != llama_kv_stream_policy_status::success) return false;
    session_operation guard(s.busy);
    try {
        return s.install(replacement.state,replacement.config,pool);
    } catch (const std::bad_alloc &) {
        return false;
    }
}


bool llama_kv_stream_session::prepare(
        const llama_memory_transition_target & target, const llama_memory_layout & layout,
        std::unique_ptr<llama_memory_preparation> & output) {
    auto & s = *impl;
    const bool decode = target.stage == s.config.decode_stage;
    const bool coordinated_scratch =
        s.config.writer_resource && s.config.attention_resource;
    if (output || !s.config.pool_resource ||
            (!decode && target.stage != s.config.prefill_stage) ||
            s.transition || s.busy || s.running || s.poisoned || !s.binding ||
            !s.binding->ready() || !s.publications || s.publications->failed() ||
            s.publication.pending() || s.content->generation() != s.expected_generation ||
            llama_memory_plan_validate(target.plan).status !=
                llama_memory_plan_status::success) return false;

    const std::array<llama_memory_resource_id,3> resources{
        s.config.pool_resource,
        coordinated_scratch ? s.config.writer_resource : 0,
        coordinated_scratch ? s.config.attention_resource : 0,
    };
    std::array<const llama_memory_resource *,3> declarations{};
    for (size_t i = 0; i < resources.size(); ++i) {
        if (!resources[i]) continue;
        for (const auto & candidate : target.plan.resources) {
            if (candidate.id != resources[i]) continue;
            if (declarations[i]) return false;
            declarations[i] = &candidate;
        }
        const auto content = i == 0 ?
            llama_memory_content::reconstructible :
            llama_memory_content::discardable;
        if (!declarations[i] ||
                declarations[i]->allocation_class !=
                    LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL ||
                declarations[i]->content != content) return false;
    }

    const llama_memory_stage * stage = nullptr;
    for (const auto & candidate : target.plan.stages) {
        if (candidate.id != target.stage) continue;
        if (stage) return false;
        stage = &candidate;
    }
    if (!stage) return false;

    std::array<const llama_memory_requirement *,3> requirements{};
    std::array<ggml_backend_memory_region,3> desired{};
    size_t arena = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < resources.size(); ++i) {
        if (!resources[i]) continue;
        for (const auto & candidate : stage->requirements) {
            if (candidate.resource != resources[i]) continue;
            if (requirements[i]) return false;
            requirements[i] = &candidate;
        }
        const auto access = i == 0 ?
            LLAMA_MEMORY_ACCESS_READ_WRITE : LLAMA_MEMORY_ACCESS_WRITE;
        if (!requirements[i] || requirements[i]->access != access ||
                !(requirements[i]->capabilities &
                    LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS)) return false;

        bool found = false;
        for (size_t a = 0; a < layout.arenas.size(); ++a) {
            const auto & candidate_arena = layout.arenas[a];
            if (candidate_arena.budget.domain != declarations[i]->domain ||
                    candidate_arena.budget.allocation_class !=
                        declarations[i]->allocation_class) continue;
            for (const auto & region : candidate_arena.regions) {
                if (region.id != resources[i]) continue;
                if (found || region.size < requirements[i]->size_min ||
                        region.size > requirements[i]->size_preferred ||
                        region.alignment < requirements[i]->alignment ||
                        (arena != std::numeric_limits<size_t>::max() &&
                         arena != a)) return false;
                found = true;
                arena = a;
                desired[i] = region;
            }
        }
        if (!found) return false;
    }

    std::array<ggml_backend_memory_region,3> before{};
    for (size_t i = 0; i < before.size(); ++i) {
        if (!s.leases[i] ||
                !ggml_backend_memory_lease_get_region(
                    s.leases[i].get(),&before[i])) return false;
        if (resources[i] && before[i].id != resources[i]) return false;
    }
    if (before[0].size != s.config.policy.pool_bytes ||
            desired[0].size == s.config.policy.pool_bytes) return false;

    size_t attention_needed = 0;
    if (coordinated_scratch &&
            (!s.attention_bytes(decode,attention_needed) ||
             desired[1].size < 32768 ||
             desired[2].size < attention_needed)) return false;

    llama_kv_stream_policy_rebind replacement;
    const auto planned = desired[0].size > s.config.policy.pool_bytes ?
        llama_kv_stream_policy_grow(
            s.config.policy,s.committed,decode,desired[0].size,replacement) :
        llama_kv_stream_policy_shrink(
            s.config.policy,s.committed,decode,desired[0].size,replacement);
    if (planned.status != llama_kv_stream_policy_status::success) return false;

    const auto frontiers = s.publications->frontiers();
    if (frontiers.reserved != s.committed ||
            frontiers.device != s.committed ||
            frontiers.host != s.committed ||
            frontiers.committed != s.committed) return false;
    if (!coordinated_scratch) {
        desired[1] = before[1];
        desired[2] = before[2];
    }
    output = std::make_unique<implementation::pool_rebind>(
        s,resources,arena,before,desired,
        std::move(replacement.config),std::move(replacement.state),
        coordinated_scratch);
    return true;
}

llama_kv_stream_binding_view llama_kv_stream_session::binding_view() const noexcept {
    const auto * view = impl->binding ? impl->binding->view() : nullptr;
    return view ? *view : llama_kv_stream_binding_view{};
}
const llama_kv_stream_policy_state & llama_kv_stream_session::policy() const noexcept { return impl->state; }
llama_kv_stream_prefetch_stats llama_kv_stream_session::sequence_stats() const noexcept {
    return impl->resident ? impl->resident->sequence_stats() : llama_kv_stream_prefetch_stats{};
}
ggml_kv_stream_copy_feedback llama_kv_stream_session::copy_feedback() const noexcept {
    return impl->resident ? impl->resident->copy_feedback() : ggml_kv_stream_copy_feedback{};
}

llama_kv_stream_publication_frontiers llama_kv_stream_session::publication_frontiers() const noexcept {
    return impl->publications ? impl->publications->frontiers() : llama_kv_stream_publication_frontiers{};
}

bool llama_kv_stream_session::set_workspaces(const std::vector<ggml_backend_memory_lease_t> & workspaces) {
    auto & s = *impl;
    if (s.busy || s.transition || s.transition_closed || s.running) return false;
    if (s.graph_bindings == workspaces) return true;
    session_operation guard(s.busy);
    std::vector<lease_ptr> retained;
    for (auto * lease : workspaces) {
        ggml_backend_memory_region region;
        if (!ggml_backend_memory_lease_get_region(lease,&region)) return false;
        retained.emplace_back(ggml_backend_memory_lease_retain(lease),ggml_backend_memory_lease_free);
    }
    auto bindings = workspaces;
    if (!s.drain()) { s.poisoned=true; return false; }
    s.graphs.clear();
    s.resident->release_write_workspace();
    if (!bindings.empty()) {
        if (!s.resident->configure_writes(s.config.max_batch_rows,s.leases[1].get())) { s.poisoned = true; return false; }
    }
    s.graph_leases = std::move(retained); s.graph_bindings = std::move(bindings); return true;
}
void llama_kv_stream_session::release_graphs() { if (!impl->busy && !impl->transition_closed) impl->graphs.clear(); }
bool llama_kv_stream_session::set_attention_workspace(ggml_backend_memory_lease_t lease, bool decode) {
    auto & s = *impl;
    if (s.busy || s.transition || s.transition_closed || s.running || s.poisoned) return false;
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
size_t llama_kv_stream_session::writer_workspace_bytes() const noexcept {
    auto * buffer = ggml_backend_memory_lease_buffer(impl->leases[1].get());
    return buffer ? ggml_backend_buffer_get_size(buffer) : 0;
}
size_t llama_kv_stream_session::attention_workspace_bytes() const noexcept {
    auto * buffer = ggml_backend_memory_lease_buffer(impl->leases[2].get());
    return buffer ? ggml_backend_buffer_get_size(buffer) : 0;
}
