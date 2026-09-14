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

    ~implementation() { abort(); }

    bool make_session() {
        llama_kv_stream_policy_config policy;
        policy.shape = config.host.shape; policy.capabilities = config.host.capabilities;
        policy.layers = config.host.layers; policy.pool_bytes = config.pool_bytes;
        session = llama_kv_stream_session::create(config.backend,content,{policy,config.max_batch_rows,config.query_heads,config.measure,true,config.resume_decode},
            leases[0].get(),leases[1].get(),leases[2].get());
        return bool(session);
    }
    // Scratch has no live conversation data. Release its backing before allocating the replacement.
    bool resize_attention(size_t bytes, bool decode) {
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
    bool reset(bool clear) {
        abort(); session.reset();
        if (clear) ggml_backend_buffer_clear(host->buffer(),0);
        if (!content->invalidate()) return false;
        return resize_attention(host->layout().bytes,false) && make_session();
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
                config.host.shape.type_k,config.host.shape.type_v,config.query_heads,config.host.shape.heads,s->host->layout().tokens,plan);
        s->decode_bytes = s->config.resume_decode ? plan.bytes : s->host->layout().bytes;
        const std::array<size_t,2> sizes{config.pool_bytes,32768};
        std::array<size_t,2> offsets{}; size_t total = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (total > SIZE_MAX-127) return {};
            offsets[i] = (total+127)/128*128;
            if (sizes[i] > SIZE_MAX-offsets[i]) return {};
            total = offsets[i]+sizes[i];
        }
        s->arena.reset(ggml_backend_memory_arena_new(type,total));
        if (!s->arena || !ggml_backend_memory_arena_begin(s->arena.get(),0)) return {};
        for (size_t i = 0; i < sizes.size(); ++i) if (!ggml_backend_memory_arena_reserve_at(s->arena.get(),s->config.host.cache_id*4+i+1,offsets[i],sizes[i],128,0,nullptr)) return {};
        if (!ggml_backend_memory_arena_commit(s->arena.get())) return {};
        for (size_t i = 0; i < sizes.size(); ++i) s->leases[i].reset(ggml_backend_memory_arena_acquire(s->arena.get(),s->config.host.cache_id*4+i+1));
        if (!s->resize_attention(s->host->layout().bytes,false) || !s->make_session()) return {};
        const ggml_backend_execution_ops ops{
            [](void * p,const ggml_tensor * t) { return (*static_cast<std::shared_ptr<implementation> *>(p))->supports(t); },
            [](void * p,ggml_backend_t b,ggml_tensor * t) {
                auto & s = **static_cast<std::shared_ptr<implementation> *>(p);
                try { const auto result = s.compute(b,t); if (result != GGML_STATUS_SUCCESS) s.abort(); return result; }
                catch (...) { s.abort(); return GGML_STATUS_FAILED; }
            },
            [](void * p) { auto & s = **static_cast<std::shared_ptr<implementation> *>(p); return !s.session || !s.session->active(); },
            [](void * p) { (void) (*static_cast<std::shared_ptr<implementation> *>(p))->reset(false); },
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
    if (!impl->session || impl->session->active() || impl->session->failed() || !queries || queries > impl->config.max_batch_rows ||
            (decode && queries != 1) || active < impl->session->tokens() || active-impl->session->tokens() != queries ||
            active > impl->host->config().context_tokens) return false;
    if (!impl->resize_attention(decode ? impl->decode_bytes : impl->host->layout().bytes,decode) ||
            !impl->session->begin(active,queries,decode)) return false;
    impl->queries = queries; impl->pending_k = nullptr; impl->pending_owner.reset(); impl->checked_indices.clear(); return true;
}
bool llama_kv_stream_model::complete() const noexcept { return impl->session && !impl->session->active() && !impl->session->failed() && !impl->pending_k; }
void llama_kv_stream_model::abort() { impl->abort(); }
bool llama_kv_stream_model::reset(bool clear) { return impl->reset(clear); }
size_t llama_kv_stream_model::tokens() const noexcept { return impl->session ? impl->session->tokens() : 0; }
size_t llama_kv_stream_model::granted_bytes() const noexcept {
    return ggml_backend_buffer_get_size(ggml_backend_memory_arena_parent(impl->arena.get())) +
        (impl->attention_arena ? ggml_backend_buffer_get_size(ggml_backend_memory_arena_parent(impl->attention_arena.get())) : 0);
}
bool llama_kv_stream_model::set_workspaces(const std::vector<ggml_backend_memory_lease_t> & leases) { return impl->session && impl->session->set_workspaces(leases); }
void llama_kv_stream_model::release_graphs() { if (impl->session) impl->session->release_graphs(); }
size_t llama_kv_stream_model::captured_layers() const { return impl->session ? impl->session->captured_layers() : 0; }
