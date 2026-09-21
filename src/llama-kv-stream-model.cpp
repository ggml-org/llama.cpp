#include "llama-kv-stream-model.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "../ggml/src/ggml-kv-stream-device.h"
#include "ggml-cpp.h"
#include "llama-impl.h"
#include <array>
#include <atomic>
#include <cstring>
#include <algorithm>

using model_arena_ptr = std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
using model_lease_ptr = std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
static std::atomic<uint64_t> model_cache_id{1};

struct llama_kv_stream_model::implementation {
    llama_kv_stream_model_config config;
    std::shared_ptr<llama_kv_stream_host> host;
    std::shared_ptr<llama_kv_stream_content> content;
    model_arena_ptr arena{nullptr,ggml_backend_memory_arena_free};
    model_arena_ptr attention_arena{nullptr,ggml_backend_memory_arena_free};
    size_t decode_bytes = 0;
    std::array<model_lease_ptr,3> leases{{{nullptr,ggml_backend_memory_lease_free},{nullptr,ggml_backend_memory_lease_free},{nullptr,ggml_backend_memory_lease_free}}};
    std::unique_ptr<llama_kv_stream_session> session;
    const ggml_tensor * pending_k = nullptr;
    ggml_backend_buffer_ptr pending_owner;
    uint32_t pending_layer = 0, queries = 0;
    std::vector<const void *> checked_indices;
    std::vector<int64_t> indices;
    bool external_mutation = false;
    bool shared = false, suspended = false;
    size_t suspended_tokens = 0;
    ggml_backend_buffer_t shared_parent = nullptr;
    llama_memory_resource_id pool_resource = 0, writer_resource = 0, attention_resource = 0;
    llama_memory_stage_id prefill_stage = 0, decode_stage = 0;

    ~implementation() { abort(); }

    std::unique_ptr<llama_kv_stream_session> create_session(
            const std::array<model_lease_ptr,3> & grants,
            llama_memory_resource_id pool_id,
            llama_memory_resource_id writer_id,
            llama_memory_resource_id attention_id,
            llama_memory_stage_id prefill_id,
            llama_memory_stage_id decode_id) {
        llama_kv_stream_policy_config policy;
        policy.shape = config.host.shape; policy.capabilities = config.host.capabilities;
        policy.layers = config.host.layers;
        auto * pool_buffer = ggml_backend_memory_lease_buffer(grants[0].get());
        auto * attention_buffer = ggml_backend_memory_lease_buffer(grants[2].get());
        if (!pool_buffer || !attention_buffer) return {};
        policy.pool_bytes = ggml_backend_buffer_get_size(pool_buffer);
        llama_kv_stream_session_config session_config{
            policy,config.max_batch_rows,config.query_heads,config.measure,true,config.resume_decode};
        session_config.initial_decode = config.resume_decode &&
            (policy.pool_bytes > config.pool_bytes ||
             ggml_backend_buffer_get_size(attention_buffer) == decode_bytes);
        session_config.cross_token_prefetch = config.cross_token_prefetch && config.resume_decode;
        session_config.pool_resource = pool_id;
        session_config.writer_resource = writer_id;
        session_config.attention_resource = attention_id;
        session_config.prefill_stage = prefill_id;
        session_config.decode_stage = decode_id;
        auto next = llama_kv_stream_session::create(
            config.backend,content,session_config,grants[0].get(),grants[1].get(),grants[2].get());
        if (next && suspended_tokens && !next->restore(suspended_tokens)) return {};
        return next;
    }

    bool make_session() {
        auto next = create_session(leases,pool_resource,writer_resource,attention_resource,prefill_stage,decode_stage);
        if (!next) return false;
        session = std::move(next);
        suspended = false;
        return true;
    }
    // Scratch has no live conversation data. Release its backing before allocating the replacement.
    bool resize_attention(size_t bytes, bool decode) {
        if (shared) {
            GGML_UNUSED(decode);
            return session && session->attention_workspace_bytes() >= bytes;
        }
        if (attention_arena && ggml_backend_buffer_get_size(ggml_backend_memory_arena_parent(attention_arena.get())) == bytes) return true;
        const size_t previous = attention_arena ? ggml_backend_buffer_get_size(ggml_backend_memory_arena_parent(attention_arena.get())) : 0;
        if (session && !session->set_attention_workspace(nullptr,decode)) return false;
        leases[2].reset(); attention_arena.reset();
        auto * type = llama_kv_stream_device_buffer_type(ggml_backend_get_device(config.backend));
        attention_arena.reset(ggml_backend_memory_arena_new(type,bytes));
        if (!attention_arena || !ggml_backend_memory_arena_begin(attention_arena.get(),0) ||
                !ggml_backend_memory_arena_reserve_at(attention_arena.get(),config.host.cache_id*4+3,0,bytes,128,0,nullptr) ||
                !ggml_backend_memory_arena_commit(attention_arena.get())) { attention_arena.reset(); return false; }
        leases[2].reset(ggml_backend_memory_arena_acquire(attention_arena.get(),config.host.cache_id*4+3));
        if (!leases[2] || (session && !session->set_attention_workspace(leases[2].get(),decode))) {
            leases[2].reset(); attention_arena.reset(); return false;
        }
        if (previous) LLAMA_LOG_INFO("%s: KV attention workspace %.2f -> %.2f MiB (%s)\n",__func__,
            previous/1048576.0,bytes/1048576.0,decode ? "resumed decode" : "strict prefill");
        return true;
    }
    void release_device_memory() {
        session.reset();
        for (auto & lease : leases) lease.reset();
        attention_arena.reset();
        arena.reset();
        shared_parent = nullptr;
        shared = false;
    }

    bool allocate_private() {
        if (session || arena || attention_arena || leases[0] || leases[1] || leases[2]) return false;
        auto * type = llama_kv_stream_device_buffer_type(ggml_backend_get_device(config.backend));
        if (!type) return false;
        pool_resource = writer_resource = attention_resource = 0;
        prefill_stage = decode_stage = 0;
        shared_parent = nullptr;
        shared = false;
        const std::array<size_t,2> sizes{config.pool_bytes,32768};
        std::array<size_t,2> offsets{};
        size_t total = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (total > SIZE_MAX-127) return false;
            offsets[i] = (total+127)/128*128;
            if (sizes[i] > SIZE_MAX-offsets[i]) return false;
            total = offsets[i]+sizes[i];
        }
        arena.reset(ggml_backend_memory_arena_new(type,total));
        if (!arena || !ggml_backend_memory_arena_begin(arena.get(),0)) {
            release_device_memory();
            return false;
        }
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (!ggml_backend_memory_arena_reserve_at(
                    arena.get(),config.host.cache_id*4+i+1,offsets[i],sizes[i],128,0,nullptr)) {
                release_device_memory();
                return false;
            }
        }
        if (!ggml_backend_memory_arena_commit(arena.get())) {
            release_device_memory();
            return false;
        }
        for (size_t i = 0; i < sizes.size(); ++i) {
            leases[i].reset(ggml_backend_memory_arena_acquire(
                arena.get(),config.host.cache_id*4+i+1));
            if (!leases[i]) {
                release_device_memory();
                return false;
            }
        }
        if (!resize_attention(host->layout().bytes,false) || !make_session()) {
            release_device_memory();
            return false;
        }
        return true;
    }
    // Full-cache roots have stable addresses; views may change shape but cannot select another plane.
    bool plane(const ggml_tensor * t, uint32_t & layer, bool & value) const {
        if (!t) return false;
        const void * data = t->data;
        if (!data && t->view_src && t->view_src->data) data = static_cast<const char *>(t->view_src->data)+t->view_offs;
        for (uint32_t i = 0; i < config.host.layers; ++i) {
            llama_kv_stream_host_layer planes; host->layer(i,planes);
            if (data == planes.k || data == planes.v) { layer = i; value = data == planes.v; return true; }
        }
        return false;
    }
    bool supports(const ggml_tensor * op) const {
        switch (op->op) {
            case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_RESHAPE: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: return true;
            case GGML_OP_SET_ROWS: {
                uint32_t layer; bool value;
                return plane(op->src[2],layer,value) && op->src[0] && op->src[1] &&
                    op->src[0]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == config.host.shape.heads*256 &&
                    op->src[0]->ne[1] > 0 && op->src[0]->ne[1] <= config.max_batch_rows &&
                    op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1 &&
                    op->src[1]->type == GGML_TYPE_I64 && ggml_is_contiguous(op->src[1]) &&
                    ggml_nelements(op->src[1]) == op->src[0]->ne[1] && (!value || op->src[3]);
            }
            case GGML_OP_FLASH_ATTN_EXT: {
                uint32_t k_layer,v_layer; bool k_value,v_value;
                if (!plane(op->src[1],k_layer,k_value) || !plane(op->src[2],v_layer,v_value) ||
                        k_value || !v_value || k_layer != v_layer || !op->src[0] || !op->src[3] || op->src[4]) return false;
                float params[3]; std::memcpy(params,op->op_params,sizeof(params));
                return params[0] > 0 && params[1] == 0 && params[2] == 0 && op->src[0]->ne[0] == 256 &&
                    op->src[0]->ne[1] > 0 && op->src[0]->ne[1] <= config.max_batch_rows &&
                    op->src[0]->ne[2] == config.query_heads && op->src[0]->ne[3] == 1 && op->src[3]->type == GGML_TYPE_F16;
            }
            default: return false;
        }
    }
    // Validate actual SET_ROWS coordinates once per input buffer per append, not once per layer.
    bool validate_indices(const ggml_tensor * tensor) {
        if (std::find(checked_indices.begin(),checked_indices.end(),tensor->data) != checked_indices.end()) return true;
        if (ggml_nelements(tensor) != queries || !tensor->data) return false;
        indices.resize(queries); ggml_backend_tensor_get(tensor,indices.data(),0,queries*sizeof(int64_t));
        for (size_t i = 0; i < indices.size(); ++i) if (indices[i] != int64_t(session->tokens()+i)) return false;
        checked_indices.push_back(tensor->data); return true;
    }
    ggml_status compute(ggml_backend_t backend, ggml_tensor * op) {
        if (backend != config.backend) return GGML_STATUS_FAILED;
        if (op->op != GGML_OP_SET_ROWS && op->op != GGML_OP_FLASH_ATTN_EXT) return GGML_STATUS_SUCCESS;
        if (!session || !session->active()) return GGML_STATUS_FAILED;
        if (op->op == GGML_OP_SET_ROWS) {
            uint32_t layer; bool value;
            if (!plane(op->src[2],layer,value) || !validate_indices(op->src[1])) return GGML_STATUS_FAILED;
            if (!value) {
                if (pending_k) return GGML_STATUS_FAILED;
                pending_k = op->src[0]; pending_layer = layer;
                pending_owner.reset(ggml_backend_buffer_retain(pending_k->view_src ? pending_k->view_src->buffer : pending_k->buffer));
                return GGML_STATUS_SUCCESS;
            }
            // The extra dependency keeps K's allocator slot live until this V operation completes.
            if (!pending_k || layer != pending_layer || op->src[3] != pending_k) return GGML_STATUS_FAILED;
            const bool ok = session->produce(layer,pending_k,op->src[0]);
            pending_k = nullptr; pending_owner.reset();
            return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
        uint32_t layer; bool value;
        if (pending_k || !plane(op->src[1],layer,value)) return GGML_STATUS_FAILED;
        float scale; std::memcpy(&scale,op->op_params,sizeof(scale));
        return session->attention(layer,op->src[0],op->src[3],op,scale) ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }
    void abort() {
        if (session) session->abort();
        pending_k = nullptr; pending_owner.reset(); checked_indices.clear();
    }
    void modified() {
        if (external_mutation) return;
        abort();
        external_mutation = content->invalidate();
    }
    bool reset(bool clear) {
        abort();
        if (clear) ggml_backend_buffer_clear(host->buffer(),0);
        if (!content->invalidate()) return false;
        external_mutation = false;
        suspended_tokens = 0;
        if (shared) return session && session->reconstruct(0);
        session.reset();
        return resize_attention(host->layout().bytes,false) && make_session();
    }

    bool restore(size_t tokens) {
        if (!external_mutation || tokens > host->config().context_tokens) return false;
        suspended_tokens = tokens;
        bool restored = false;
        if (shared) {
            restored = session && session->reconstruct(tokens);
        } else {
            session.reset();
            restored = resize_attention(host->layout().bytes,false) && make_session();
        }
        if (!restored) return false;
        external_mutation = false;
        return true;
    }

    bool truncate(size_t tokens) {
        if (external_mutation || !session || session->active() || tokens > session->tokens()) return false;
        if (tokens == session->tokens()) return true;
        abort();
        if (!content->invalidate()) return false;
        external_mutation = true;
        return restore(tokens);
    }
};

std::unique_ptr<llama_kv_stream_model> llama_kv_stream_model::create(const llama_kv_stream_model_config & config) {
    if (!config.backend || !config.pool_bytes || !config.max_batch_rows || !config.query_heads ||
            !config.host.context_tokens || config.host.context_tokens > size_t(INT32_MAX)-255 ||
            config.query_heads > SIZE_MAX/config.max_batch_rows) return {};
    llama_kv_stream_policy_config policy;
    policy.shape = config.host.shape; policy.capabilities = config.host.capabilities;
    policy.layers = config.host.layers; policy.pool_bytes = config.pool_bytes;
    llama_kv_stream_policy_state initial;
    if (llama_kv_stream_policy_initialize(policy,initial).status != llama_kv_stream_policy_status::success) return {};
    auto * dev = ggml_backend_get_device(config.backend);
    auto * type = llama_kv_stream_device_buffer_type(dev);
    auto * host_type = llama_kv_stream_host_buffer_type(dev);
    if (!type || !host_type) return {};
    ggml_kv_stream_block_layout partial;
    if (ggml_kv_stream_block_layout_make(size_t(config.max_batch_rows)*config.query_heads,256,partial).status != ggml_kv_stream_partial_status::success) return {};
    try {
        auto s = std::make_shared<implementation>(); s->config = config;
        s->config.host.cache_id = model_cache_id.fetch_add(1,std::memory_order_relaxed);
        if (!s->config.host.cache_id || s->config.host.cache_id > (UINT64_MAX-3)/4) return {};
        s->host = llama_kv_stream_host::create(s->config.host,host_type);
        if (!s->host) return {};
        s->content = std::make_shared<llama_kv_stream_content>(s->host);
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        ggml_kv_stream_resume_plan plan;
        s->config.resume_decode = config.resume_decode && !std::getenv("LLAMA_KV_STREAM_DECODE_GATHER") && get && get() &&
            get()->version >= 5 && get()->resume_plan && get()->resume && get()->resume_plan(config.backend,
                config.host.shape.type_k,config.host.shape.type_v,config.query_heads,config.host.shape.heads,1,s->host->layout().tokens,plan);
        s->decode_bytes = s->config.resume_decode ? plan.bytes : s->host->layout().bytes;
        if (!s->allocate_private()) return {};
        const ggml_backend_execution_ops ops{
            [](void * p,const ggml_tensor * t) { return (*static_cast<std::shared_ptr<implementation> *>(p))->supports(t); },
            [](void * p,ggml_backend_t b,ggml_tensor * t) {
                auto & s = **static_cast<std::shared_ptr<implementation> *>(p);
                try { const auto result = s.compute(b,t); if (result != GGML_STATUS_SUCCESS) s.abort(); return result; }
                catch (...) { s.abort(); return GGML_STATUS_FAILED; }
            },
            [](void * p) { auto & s = **static_cast<std::shared_ptr<implementation> *>(p); return !s.session || !s.session->active(); },
            [](void * p) { (*static_cast<std::shared_ptr<implementation> *>(p))->modified(); },
            [](void * p) { delete static_cast<std::shared_ptr<implementation> *>(p); }
        };
        auto owner = std::make_unique<std::shared_ptr<implementation>>(s);
        std::unique_ptr<llama_kv_stream_model> result(new llama_kv_stream_model);
        result->impl = s;
        result->proxy = ggml_backend_execution_buffer_new(dev,s->host->buffer(),ops,owner.get());
        if (!result->proxy) return {};
        owner.release(); return result;
    } catch (const std::bad_alloc &) { return {}; }
}

llama_kv_stream_model::~llama_kv_stream_model() { ggml_backend_buffer_free(proxy); }
ggml_backend_buffer_t llama_kv_stream_model::buffer() const noexcept { return proxy; }
std::shared_ptr<llama_kv_stream_host> llama_kv_stream_model::host() const noexcept { return impl->host; }
bool llama_kv_stream_model::begin(size_t active,uint32_t queries,bool decode) {
    if (!impl->session || impl->external_mutation || impl->session->active() || impl->session->failed() || !queries || queries > impl->config.max_batch_rows ||
            (decode && queries != 1) || active < impl->session->tokens() || active-impl->session->tokens() != queries ||
            active > impl->host->config().context_tokens) return false;
    if (!impl->resize_attention(decode ? impl->decode_bytes : impl->host->layout().bytes,decode) ||
            !impl->session->begin(active,queries,decode)) return false;
    impl->queries = queries; impl->pending_k = nullptr; impl->pending_owner.reset(); impl->checked_indices.clear(); return true;
}
bool llama_kv_stream_model::complete() const noexcept { return impl->session && !impl->external_mutation && !impl->session->active() && !impl->session->failed() && !impl->pending_k; }
void llama_kv_stream_model::abort() { impl->abort(); }
bool llama_kv_stream_model::reset(bool clear) { return impl->reset(clear); }
bool llama_kv_stream_model::restore(size_t tokens) { return impl->restore(tokens); }
bool llama_kv_stream_model::truncate(size_t tokens) { return impl->truncate(tokens); }
size_t llama_kv_stream_model::tokens() const noexcept { return impl->session ? impl->session->tokens() : 0; }
size_t llama_kv_stream_model::granted_bytes() const noexcept {
    if (impl->shared) return device_grant_bytes();
    return (impl->arena ? ggml_backend_buffer_get_size(ggml_backend_memory_arena_parent(impl->arena.get())) : 0) +
        (impl->attention_arena ? ggml_backend_buffer_get_size(ggml_backend_memory_arena_parent(impl->attention_arena.get())) : 0);
}
bool llama_kv_stream_model::set_workspaces(const std::vector<ggml_backend_memory_lease_t> & leases) { return impl->session && impl->session->set_workspaces(leases); }
void llama_kv_stream_model::release_graphs() { if (impl->session) impl->session->release_graphs(); }
size_t llama_kv_stream_model::captured_layers() const { return impl->session ? impl->session->captured_layers() : 0; }

bool llama_kv_stream_model::memory_requirements(
        llama_kv_stream_memory_requirements & output) const noexcept {
    if (!impl || !impl->host) return false;
    auto * type = llama_kv_stream_device_buffer_type(ggml_backend_get_device(impl->config.backend));
    if (!type) return false;
    output = {
        type,
        impl->config.pool_bytes,
        32768,
        impl->host->layout().bytes,
        impl->decode_bytes,
        std::max(size_t(128),ggml_backend_buft_get_alignment(type)),
        impl->config.shared_device_memory_bytes,
    };
    return true;
}

bool llama_kv_stream_model::prepare_shared_memory() {
    auto & s = *impl;
    if (s.shared || s.pending_k || (s.session && s.session->active())) return false;
    if (s.suspended && !s.session) return true;
    s.suspended_tokens = s.session ? s.session->tokens() : s.suspended_tokens;
    s.release_device_memory();
    s.suspended = true;
    return true;
}

bool llama_kv_stream_model::resume_private_memory() {
    auto & s = *impl;
    return !s.shared && s.suspended && !s.session && s.allocate_private();
}

static bool model_region(
        ggml_backend_memory_lease_t lease, ggml_backend_buffer_type_t type,
        uint64_t id, size_t bytes, ggml_backend_memory_region & region,
        uintptr_t & base) {
    auto * buffer = ggml_backend_memory_lease_buffer(lease);
    if (!buffer || !ggml_backend_memory_lease_get_region(lease,&region) ||
            region.id != id || region.size != bytes ||
            ggml_backend_buffer_get_type(buffer) != type ||
            ggml_backend_buffer_get_size(buffer) != bytes) return false;
    base = uintptr_t(ggml_backend_buffer_get_base(buffer));
    return base && base <= UINTPTR_MAX-bytes;
}

bool llama_kv_stream_model::attach_shared_memory(
        const llama_kv_stream_memory_binding & binding) {
    auto & s = *impl;
    llama_kv_stream_memory_requirements requirements;
    if (!memory_requirements(requirements) || !s.suspended || s.session || s.shared ||
            !binding.parent || !binding.pool || !binding.writer || !binding.attention ||
            !binding.pool_resource || !binding.writer_resource || !binding.attention_resource ||
            binding.pool_resource == binding.writer_resource ||
            binding.pool_resource == binding.attention_resource ||
            binding.writer_resource == binding.attention_resource ||
            !binding.prefill_stage || !binding.decode_stage ||
            binding.prefill_stage == binding.decode_stage ||
            ggml_backend_buffer_get_type(binding.parent) != requirements.buffer_type) return false;

    const size_t attention_bytes = std::max(
        requirements.attention_prefill_bytes,requirements.attention_decode_bytes);
    auto * pool_buffer = ggml_backend_memory_lease_buffer(binding.pool);
    const size_t pool_bytes = pool_buffer ? ggml_backend_buffer_get_size(pool_buffer) : 0;
    if (pool_bytes < requirements.pool_bytes) return false;
    const std::array<ggml_backend_memory_lease_t,3> supplied{
        binding.pool,binding.writer,binding.attention};
    const std::array<uint64_t,3> ids{
        binding.pool_resource,binding.writer_resource,binding.attention_resource};
    const std::array<size_t,3> sizes{
        pool_bytes,requirements.writer_bytes,attention_bytes};
    std::array<ggml_backend_memory_region,3> regions{};
    std::array<uintptr_t,3> addresses{};
    const auto parent_base = uintptr_t(ggml_backend_buffer_get_base(binding.parent));
    const size_t parent_bytes = ggml_backend_buffer_get_size(binding.parent);
    if (!parent_base || parent_base > UINTPTR_MAX-parent_bytes) return false;
    for (size_t i = 0; i < supplied.size(); ++i) {
        if (!model_region(supplied[i],requirements.buffer_type,ids[i],sizes[i],
                regions[i],addresses[i]) ||
                addresses[i] < parent_base ||
                addresses[i]-parent_base > parent_bytes-sizes[i] ||
                addresses[i]%requirements.alignment) return false;
        for (size_t j = 0; j < i; ++j) {
            if (addresses[i] < addresses[j]+sizes[j] &&
                    addresses[j] < addresses[i]+sizes[i]) return false;
        }
    }

    std::array<model_lease_ptr,3> retained{{
        {nullptr,ggml_backend_memory_lease_free},
        {nullptr,ggml_backend_memory_lease_free},
        {nullptr,ggml_backend_memory_lease_free},
    }};
    for (size_t i = 0; i < retained.size(); ++i) {
        retained[i].reset(ggml_backend_memory_lease_retain(supplied[i]));
        if (!retained[i]) return false;
    }
    auto candidate = s.create_session(
        retained,binding.pool_resource,binding.writer_resource,binding.attention_resource,
        binding.prefill_stage,binding.decode_stage);
    if (!candidate) return false;

    s.session = std::move(candidate);
    s.pool_resource = binding.pool_resource;
    s.writer_resource = binding.writer_resource;
    s.attention_resource = binding.attention_resource;
    s.prefill_stage = binding.prefill_stage;
    s.decode_stage = binding.decode_stage;
    s.shared_parent = binding.parent;
    s.shared = true;
    s.suspended = false;
    s.arena.reset();
    s.attention_arena.reset();
    return true;
}

bool llama_kv_stream_model::detach_shared_memory() noexcept {
    auto & s = *impl;
    if (!s.shared) return s.suspended && !s.session;
    if (s.pending_k || (s.session && s.session->active())) return false;
    s.suspended_tokens = s.session ? s.session->tokens() : s.suspended_tokens;
    s.release_device_memory();
    s.pool_resource = s.writer_resource = s.attention_resource = 0;
    s.prefill_stage = s.decode_stage = 0;
    s.suspended = true;
    return true;
}

llama_memory_consumer * llama_kv_stream_model::memory_consumer() noexcept {
    return impl->shared && impl->session ? impl->session.get() : nullptr;
}

bool llama_kv_stream_model::uses_shared_memory() const noexcept {
    return impl->shared;
}

ggml_backend_buffer_t llama_kv_stream_model::shared_parent() const noexcept {
    return impl->shared_parent;
}

size_t llama_kv_stream_model::device_grant_bytes() const noexcept {
    if (impl->shared) return impl->session ? impl->session->granted_bytes() : 0;
    size_t total = 0;
    for (const auto & lease : impl->leases) {
        auto * buffer = ggml_backend_memory_lease_buffer(lease.get());
        if (!buffer) return 0;
        const size_t bytes = ggml_backend_buffer_get_size(buffer);
        if (bytes > SIZE_MAX-total) return 0;
        total += bytes;
    }
    return total;
}

size_t llama_kv_stream_model::pool_grant_bytes() const noexcept {
    return impl->session ? impl->session->binding_view().capacity : 0;
}

size_t llama_kv_stream_model::writer_grant_bytes() const noexcept {
    return impl->session ? impl->session->writer_workspace_bytes() : 0;
}

size_t llama_kv_stream_model::attention_grant_bytes() const noexcept {
    return impl->session ? impl->session->attention_workspace_bytes() : 0;
}

bool llama_kv_stream_model::runtime_diagnostics(
        llama_kv_stream_runtime_diagnostics & output) const noexcept {
    if (!impl->session) return false;
    const auto & policy = impl->session->policy();
    const auto copies = impl->session->sequence_stats();
    const auto timing = impl->session->copy_feedback();
    output = {
        impl->session->layout_revision(),
        pool_grant_bytes(),writer_grant_bytes(),attention_grant_bytes(),
        policy.resident_pages_per_layer,policy.ring_slots,policy.decode_active_pages,
        copies.copy_bytes,copies.copy_calls,timing.copy_ms,timing.elapsed_ms,policy.decode_active_pages != 0,
    };
    return true;
}
