#include "llama-kv-stream-model.h"
#include "llama-kv-stream-logical-cache.h"
#include "llama-kv-stream-writer.h"
#include "llama-context-memory.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "../ggml/src/ggml-kv-stream-device.h"
#include "ggml-cpp.h"
#include "llama-impl.h"
#include <array>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <cmath>

using model_arena_ptr = std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
using mtp_lease_ptr = std::unique_ptr<llama_kv_stream_complete_layer_lease,
    decltype(&llama_kv_stream_complete_layer_lease_free)>;
using model_lease_ptr = std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
static std::atomic<uint64_t> model_cache_id{1};

// Strict prefill uses native attention even when decode has a compiled streamed implementation.
static bool native_attention_available(const llama_kv_stream_model_config & config,uint32_t & unavailable) {
    ggml_kv_stream_layout layout;
    if (ggml_kv_stream_layout_make(config.host.shape,config.host.context_tokens,layout).status != ggml_kv_stream_status::success ||
            config.query_heads > INT32_MAX || config.query_heads%config.host.shape.heads) return false;
    const uint32_t maximum = uint32_t(std::min(size_t(config.max_batch_rows),config.host.context_tokens));
    for (uint32_t rows = 1; rows <= maximum; ++rows) {
        ggml_tensor q{}, k{}, v{}, mask{}, op{};
        q.type = op.type = GGML_TYPE_F32;
        k.type = ggml_type(config.host.shape.type_k); v.type = ggml_type(config.host.shape.type_v); mask.type = GGML_TYPE_F16;
        const int64_t tokens = int64_t((config.host.context_tokens+255)/256*256);
        const int64_t extents[5][4] = {
            {config.host.shape.head_dim_k,rows,config.query_heads,1},
            {config.host.shape.head_dim_k,tokens,config.host.shape.heads,1},
            {config.host.shape.head_dim_v,tokens,config.host.shape.heads,1},
            {tokens,rows,1,1},
            {config.host.shape.head_dim_v,config.query_heads,rows,1},
        };
        ggml_tensor * tensors[] = {&q,&k,&v,&mask,&op};
        for (size_t i = 0; i < 5; ++i) {
            auto * tensor = tensors[i];
            for (int dimension = 0; dimension < 4; ++dimension) tensor->ne[dimension] = extents[i][dimension];
            tensor->nb[0] = ggml_type_size(tensor->type);
            tensor->nb[1] = ggml_row_size(tensor->type,tensor->ne[0]);
            for (int dimension = 2; dimension < 4; ++dimension) {
                if (size_t(tensor->ne[dimension-1]) > SIZE_MAX/tensor->nb[dimension-1]) return false;
                tensor->nb[dimension] = tensor->nb[dimension-1]*size_t(tensor->ne[dimension-1]);
            }
        }
        op.op = GGML_OP_FLASH_ATTN_EXT;
        op.src[0] = &q; op.src[1] = &k; op.src[2] = &v; op.src[3] = &mask;
        ggml_flash_attn_ext_set_prec(&op,GGML_PREC_F32);
        if (!ggml_backend_supports_op(config.backend,&op)) { unavailable = rows; return false; }
    }
    return true;
}

struct llama_kv_stream_read_proxy::implementation {
    llama_kv_stream_model * source=nullptr;
    llama_context_memory * parent=nullptr;
    std::shared_ptr<llama_kv_stream_host> host;
    ggml_backend_dev_t device=nullptr;
    ggml_context_ptr metadata;
    ggml_tensor * k=nullptr,* v=nullptr;
    uint32_t layer=0,heads=0,queries=0;
    uint64_t alias_id=0,parent_revision=0;
    llama_memory_cache_read_binding binding;
    ggml_backend_buffer_ptr retained_parent;
    bool armed=false,busy=false;

    bool current() const noexcept {
        llama_memory_cache_requirement now;
        uint64_t revision=0;
        return armed && parent->valid() && retained_parent &&
            parent->shared_parent() == retained_parent.get() && source->shared_parent() == retained_parent.get() &&
            parent->graph_binding_revision() == parent_revision && source->read_source(now,&revision) &&
            llama_memory_cache_read_binding_valid(binding,now,0,uint32_t(source->tokens()),revision);
    }
    bool supports(const ggml_tensor * op) const {
        if (!op || op->op != GGML_OP_FLASH_ATTN_EXT || op->type != GGML_TYPE_F32 ||
                !op->src[0] || !op->src[1] || !op->src[2] || !op->src[3] || op->src[4] || op->src[5]) return false;
        const auto * q=op->src[0],* key=op->src[1],* value=op->src[2],* mask=op->src[3];
        const auto & shape=host->config().shape;
        llama_kv_stream_host_layer planes;
        if (!host->layer(layer,planes)) return false;
        float params[3]; std::memcpy(params,op->op_params,sizeof(params));
        return std::isfinite(params[0]) && params[0] > 0 && params[1] == 0 && params[2] == 0 &&
            q->type == GGML_TYPE_F32 && q->ne[0] == shape.head_dim_k && q->ne[1] > 0 &&
            q->ne[1] <= queries && q->ne[2] == heads && q->ne[3] == 1 &&
            key->type == shape.type_k && value->type == shape.type_v && key->data == planes.k && value->data == planes.v &&
            key->ne[0] == shape.head_dim_k && value->ne[0] == shape.head_dim_v && key->ne[1] > 0 &&
            key->ne[1] <= int64_t(host->layout().tokens) && value->ne[1] == key->ne[1] &&
            key->ne[2] == shape.heads && value->ne[2] == shape.heads && key->ne[3] == 1 && value->ne[3] == 1 &&
            key->nb[0] == ggml_type_size(key->type) && value->nb[0] == ggml_type_size(value->type) &&
            key->nb[1] == host->layout().k_token_bytes && value->nb[1] == host->layout().v_token_bytes &&
            key->nb[2] == ggml_row_size(key->type,shape.head_dim_k) &&
            value->nb[2] == ggml_row_size(value->type,shape.head_dim_v) &&
            mask->type == GGML_TYPE_F16 && mask->ne[0] == key->ne[1] && mask->ne[1] == q->ne[1] &&
            mask->ne[2] == 1 && mask->ne[3] == 1 && op->ne[0] == shape.head_dim_v &&
            op->ne[1] == heads && op->ne[2] == q->ne[1] && op->ne[3] == 1;
    }
    ggml_status compute(ggml_backend_t backend,ggml_tensor * op) {
        if (busy || ggml_backend_get_device(backend) != device || !supports(op) || !current() ||
                op->src[1]->ne[1] != int64_t((source->tokens()+255)/256*256)) return GGML_STATUS_FAILED;
        busy=true;
        struct guard { bool & value; ~guard() { value=false; } } done{busy};
        try {
            // Queries may be produced on another scheduler's queue on the same device.
            ggml_backend_synchronize(backend);
            if (!current()) return GGML_STATUS_FAILED;
            float scale; std::memcpy(&scale,op->op_params,sizeof(scale));
            return source->read_attention(binding,op->src[0],op->src[3],op,scale) ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        } catch (const std::exception & error) {
            LLAMA_LOG_ERROR("%s: read-only KV attention failed: %s\n",__func__,error.what());
            return GGML_STATUS_FAILED;
        } catch (...) { return GGML_STATUS_FAILED; }
    }
};

std::unique_ptr<llama_kv_stream_read_proxy> llama_kv_stream_read_proxy::create(llama_kv_stream_model & source,
        llama_context_memory & parent,uint32_t layer,const llama_memory_cache_format & format,
        uint32_t heads,uint32_t queries) {
    const auto host=source.host();
    if (!host || !parent.valid() || !parent.shares_kv_memory() || !source.uses_shared_memory() ||
            source.shared_parent() != parent.shared_parent() || layer >= host->config().layers ||
            !heads || heads > INT32_MAX || !queries || queries > source.max_decode_rows() ||
            queries > source.max_batch_rows()) return {};
    const auto & shape=host->config().shape;
    const llama_memory_cache_format actual{ggml_type(shape.type_k),ggml_type(shape.type_v),
        uint32_t(shape.head_dim_k),uint32_t(shape.head_dim_v),uint32_t(shape.heads),false};
    if (!llama_memory_cache_format_equal(format,actual) || heads%actual.kv_heads) return {};
    llama_kv_stream_memory_requirements requirements;
    if (!source.memory_requirements(requirements)) return {};
    auto * device=ggml_backend_buft_get_device(requirements.buffer_type);
    if (!device) return {};
    try {
        auto state=std::make_shared<implementation>();
        state->source=&source; state->parent=&parent; state->host=host; state->device=device;
        state->layer=layer; state->heads=heads; state->queries=queries;
        state->alias_id=model_cache_id.fetch_add(1,std::memory_order_relaxed);
        if (!state->alias_id || state->alias_id > (UINT64_MAX-3)/4) return {};
        state->metadata.reset(ggml_init({4096+2*ggml_tensor_overhead(),nullptr,true}));
        if (!state->metadata) return {};
        state->k=ggml_new_tensor_3d(state->metadata.get(),actual.key_type,
            int64_t(actual.key_head_dim)*actual.kv_heads,host->layout().tokens,1);
        state->v=ggml_new_tensor_3d(state->metadata.get(),actual.value_type,
            int64_t(actual.value_head_dim)*actual.kv_heads,host->layout().tokens,1);
        const ggml_backend_execution_ops ops{
            [](void * p,const ggml_tensor * op) { return (*static_cast<std::shared_ptr<implementation> *>(p))->supports(op); },
            [](void * p,ggml_backend_t backend,ggml_tensor * op) { return (*static_cast<std::shared_ptr<implementation> *>(p))->compute(backend,op); },
            [](void *) { return false; },
            [](void *) {},
            [](void * p) { delete static_cast<std::shared_ptr<implementation> *>(p); }
        };
        auto context=std::make_unique<std::shared_ptr<implementation>>(state);
        std::unique_ptr<llama_kv_stream_read_proxy> result(new llama_kv_stream_read_proxy);
        result->impl=state;
        result->proxy=ggml_backend_execution_buffer_new(device,host->buffer(),ops,context.get());
        if (!result->proxy) return {};
        context.release();
        llama_kv_stream_host_layer planes;
        if (!host->layer(layer,planes) || ggml_backend_tensor_alloc(result->proxy,state->k,planes.k) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(result->proxy,state->v,planes.v) != GGML_STATUS_SUCCESS) return {};
        return result;
    } catch (const std::bad_alloc &) { return {}; }
}

llama_kv_stream_read_proxy::~llama_kv_stream_read_proxy() { ggml_backend_buffer_free(proxy); }
ggml_tensor * llama_kv_stream_read_proxy::key() const noexcept { return impl->k; }
ggml_tensor * llama_kv_stream_read_proxy::value() const noexcept { return impl->v; }
ggml_backend_buffer_t llama_kv_stream_read_proxy::buffer() const noexcept { return proxy; }
bool llama_kv_stream_read_proxy::ready() const noexcept { return !impl->busy && impl->current(); }
bool llama_kv_stream_read_proxy::arm() {
    auto & s=*impl;
    if (s.busy) return false;
    if (s.armed) return s.current();
    llama_memory_cache_requirement source;
    uint64_t revision=0;
    if (!s.parent->valid() || !s.source->read_source(source,&revision) ||
            s.source->shared_parent() != s.parent->shared_parent()) return false;
    auto alias=source;
    alias.identity={s.alias_id,1}; alias.source=source.identity;
    alias.role=llama_memory_cache_role::read_only_source; alias.layers=1; alias.source_first_layer=s.layer;
    llama_memory_cache_read_binding next;
    if (!llama_memory_cache_read_binding_make(alias,source,0,uint32_t(s.source->tokens()),revision,next)) return false;
    ggml_backend_buffer_ptr retained(ggml_backend_buffer_retain(s.parent->shared_parent()));
    if (!retained) return false;
    s.binding=next; s.parent_revision=s.parent->graph_binding_revision(); s.retained_parent=std::move(retained); s.armed=true;
    return s.current();
}
bool llama_kv_stream_read_proxy::disarm() noexcept {
    if (impl->busy) return false;
    impl->armed=false; impl->binding={}; impl->retained_parent.reset();
    return true;
}

// Intent identifies this existing alias; publication supplies the later read's current generation.
bool llama_kv_stream_read_proxy::next_consumer(uint32_t queries,llama_kv_stream_next_consumer & output) const noexcept {
    if (!queries || queries > impl->queries || !impl->alias_id || impl->alias_id == impl->host->cache_id()) return false;
    llama_kv_stream_next_consumer next;
    next.access=llama_kv_stream_access_role::read;
    next.cache_id=impl->host->cache_id(); next.reader_id=impl->alias_id;
    next.layer=impl->layer; next.query_tokens=queries; next.query_heads=impl->heads;
    output=next;
    return true;
}

// Start transport before the reader graph computes its query, but only inside a current leased frame.
bool llama_kv_stream_read_proxy::prefetch(uint32_t queries) {
    return impl->current() && queries && queries <= impl->queries &&
        impl->source->prefetch_read(impl->binding,queries,impl->heads);
}

struct llama_kv_stream_model::implementation {
    llama_kv_stream_model_config config;
    std::shared_ptr<llama_kv_stream_host> host;
    std::shared_ptr<llama_kv_stream_content> content;
    llama_kv_stream_policy_config physical_policy;
    std::shared_ptr<llama_kv_stream_logical_cache> auxiliary_cache;
    std::shared_ptr<llama_kv_stream_transfer> transfer;
    llama_kv_stream_next_consumer after_append,after_read;
    llama_kv_stream_prefetch_stats append_stats,read_stats;
    bool decoding=false, auxiliary_handoff=false;
    bool acquire_mtp_layer(size_t future_tokens, bool prefetch);
    // The owning cache is shared; only the declared access role selects a frozen read or a future append.
    bool prefetch_next(const llama_kv_stream_next_consumer & next) {
        if (!session || !next.cache_id) return false;
        if (next.access == llama_kv_stream_access_role::read) {
            llama_memory_cache_requirement source;
            if (!session->read_source(source)) return false;
            auto alias=source;
            alias.identity={next.reader_id,1}; alias.source=source.identity;
            alias.role=llama_memory_cache_role::read_only_source;
            alias.layers=1; alias.source_first_layer=next.layer;
            llama_memory_cache_read_binding binding;
            return llama_memory_cache_read_binding_make(alias,source,0,uint32_t(session->tokens()),
                session->layout_revision(),binding) && session->prefetch_read(binding,next.query_tokens,next.query_heads);
        }
        if (next.cache_id == host->cache_id()) return session->prefetch_decode(next.future_tokens);
        if (auxiliary_cache && next.cache_id == auxiliary_cache->identity().id && auxiliary_cache->tokens() >= 4 && !mtp_owner) {
            const size_t future=std::min(size_t(next.future_tokens),config.host.context_tokens-session->tokens());
            return acquire_mtp_layer(future,true);
        }
        return false;
    }
    bool complete() const {
        return session && !session->device_suspended() && !external_mutation && !session->active() &&
            !session->failed() && !pending_k;
    }
    std::unique_ptr<llama_kv_stream_layer_lease_owner> mtp_owner;
    std::vector<mtp_lease_ptr> mtp_plans;
    std::shared_ptr<const llama_kv_stream_ring_guard> mtp_guard;
    model_arena_ptr arena{nullptr,ggml_backend_memory_arena_free};
    size_t mtp_reserved_tokens = 0;
    // A non-owning mirror stamp survives ring release but never pins an old arena.
    ggml_backend_buffer_t mtp_resident_buffer = nullptr;
    uint64_t mtp_resident_revision = 0, mtp_resident_arena_generation = 0;
    model_arena_ptr attention_arena{nullptr,ggml_backend_memory_arena_free};
    size_t decode_bytes = 0;
    size_t span_bytes = 0;
    size_t prefill_span_bytes = 0, prefill_bytes = 0;
    size_t writer_bytes = 0;
    bool compact_prefill_mask=false;
    std::array<model_lease_ptr,3> leases{{{nullptr,ggml_backend_memory_lease_free},{nullptr,ggml_backend_memory_lease_free},{nullptr,ggml_backend_memory_lease_free}}};
    std::shared_ptr<void> prepared_copies;
    size_t prepared_copy_capacity = 0;
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
    llama_memory_stage_id prefill_stage = 0, decode_stage = 0, suspend_stage = 0;

    ~implementation() {
        abort();
        // The target session drops its ring snapshot before the physical reservation.
        session.reset();
        mtp_guard.reset();
        mtp_plans.clear();
        mtp_owner.reset();
    }

    std::unique_ptr<llama_kv_stream_session> create_session(
            const std::array<model_lease_ptr,3> & grants,
            llama_memory_resource_id pool_id,
            llama_memory_resource_id writer_id,
            llama_memory_resource_id attention_id,
            llama_memory_stage_id prefill_id,
            llama_memory_stage_id decode_id, llama_memory_stage_id suspend_id = 0) {
        mtp_resident_buffer=nullptr;
        llama_kv_stream_policy_config policy = physical_policy;
        auto * pool_buffer = ggml_backend_memory_lease_buffer(grants[0].get());
        auto * attention_buffer = ggml_backend_memory_lease_buffer(grants[2].get());
        if (!pool_buffer || !attention_buffer) return {};
        policy.pool_bytes = ggml_backend_buffer_get_size(pool_buffer);
        llama_kv_stream_session_config session_config{
            policy,config.max_batch_rows,config.query_heads,config.measure,true,config.resume_decode};
        session_config.initial_decode = config.resume_decode &&
            (policy.pool_bytes > config.pool_bytes ||
             ggml_backend_buffer_get_size(attention_buffer) == decode_bytes);
        session_config.cross_token_prefetch = config.cross_token_prefetch && config.resume_decode && !after_append.cache_id;
        session_config.span_workspace_bytes = span_bytes;
        session_config.pool_resource = pool_id;
        session_config.writer_resource = writer_id;
        session_config.attention_resource = attention_id;
        session_config.prefill_stage = prefill_id;
        session_config.decode_stage = decode_id;
        session_config.suspend_stage = suspend_id;
        session_config.prepared_copies = prepared_copies;
        session_config.max_decode_rows = config.max_decode_rows;
        session_config.prefill_span_bytes=prefill_span_bytes;
        session_config.writer_workspace_bytes=writer_bytes;
        auto next = llama_kv_stream_session::create(
            config.backend,content,session_config,grants[0].get(),grants[1].get(),grants[2].get());
        if (next && suspended_tokens && !next->restore(suspended_tokens)) return {};
        return next;
    }

    // Final shared grants may exceed a bootstrap pool; grow only at an unbound admission boundary.
    bool prepare_copies(size_t budget) {
        const auto get = reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(config.backend)),"ggml_backend_kv_stream_copy_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!ops || ops->version < 11) return !prepared_copies;
        ggml_kv_stream_layout page;
        if (!ops->prepare || !ops->free_prepared || !ops->create_prepared ||
                ggml_kv_stream_layout_make(config.host.shape,size_t(config.host.shape.page_tokens),page).status !=
                    ggml_kv_stream_status::success || !page.bytes || budget < page.bytes) return false;
        const size_t capacity = budget/page.bytes;
        if (prepared_copies && capacity <= prepared_copy_capacity) return true;
        if (session) return false;
        prepared_copies.reset(); prepared_copy_capacity = 0;
        void * resources = ops->prepare(config.backend,capacity,config.measure);
        if (!resources) {
            LLAMA_LOG_ERROR("%s: KV copy-resource preflight failed: capacity=%zu slots, budget=%zu bytes\n",
                __func__,capacity,budget);
            return false;
        }
        prepared_copies = std::shared_ptr<void>(resources,ops->free_prepared);
        prepared_copy_capacity = capacity;
        LLAMA_LOG_INFO("%s: KV copy-resource preflight reserved capacity=%zu slots for budget=%zu bytes\n",
            __func__,capacity,budget);
        return true;
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
        if (!prepare_copies(config.shared_device_memory_bytes ? config.shared_device_memory_bytes : config.pool_bytes)) return false;
        if (session || arena || attention_arena || leases[0] || leases[1] || leases[2]) return false;
        auto * type = llama_kv_stream_device_buffer_type(ggml_backend_get_device(config.backend));
        if (!type) return false;
        pool_resource = writer_resource = attention_resource = 0;
        prefill_stage = decode_stage = suspend_stage = 0;
        shared_parent = nullptr;
        shared = false;
        const std::array<size_t,2> sizes{config.pool_bytes,writer_bytes};
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
        if (!resize_attention(prefill_bytes,false) || !make_session()) {
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
                    op->src[0]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == config.host.shape.heads*
                        (value ? config.host.shape.head_dim_v : config.host.shape.head_dim_k) &&
                    op->src[0]->ne[1] > 0 && op->src[0]->ne[1] <= config.max_batch_rows &&
                    op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1 &&
                    op->src[1]->type == GGML_TYPE_I64 && ggml_is_contiguous(op->src[1]) &&
                    ggml_nelements(op->src[1]) == op->src[0]->ne[1] && (!value || op->src[3]);
            }
            case GGML_OP_FLASH_ATTN_EXT: {
                uint32_t k_layer,v_layer; bool k_value,v_value;
                if (!plane(op->src[1],k_layer,k_value) || !plane(op->src[2],v_layer,v_value) ||
                        k_value || !v_value || k_layer != v_layer || !op->src[0] || (!op->src[3] && !op->src[5]) || op->src[4]) return false;
                ggml_tensor logical{}, mask{};
                if (op->src[5] && (!compact_prefill_mask ||
                        !ggml_backend_execution_mask_metadata(op,logical,mask))) return false;
                float params[3]; std::memcpy(params,op->op_params,sizeof(params));
                return params[0] > 0 && params[1] == 0 && params[2] == 0 && op->src[0]->ne[0] == config.host.shape.head_dim_k &&
                    op->src[0]->ne[1] > 0 && op->src[0]->ne[1] <= config.max_batch_rows &&
                    op->src[0]->ne[2] == config.query_heads && op->src[0]->ne[3] == 1 &&
                    (op->src[5] || op->src[3]->type == GGML_TYPE_F16);
            }
            default: return false;
        }
    }
    // Validate actual SET_ROWS coordinates once per input buffer per append, not once per layer.
    bool validate_indices(const ggml_tensor * tensor) {
        if (std::find(checked_indices.begin(),checked_indices.end(),tensor->data) != checked_indices.end()) return true;
        if (ggml_nelements(tensor) != queries || !tensor->data) return false;
        indices.resize(queries); ggml_backend_tensor_get(tensor,indices.data(),0,queries*sizeof(int64_t));
        for (size_t i = 0; i < indices.size(); ++i) if (indices[i] != int64_t(session->tokens()+i)) {
            LLAMA_LOG_WARN("%s: target KV index mismatch at row %zu: actual=%lld expected=%zu\n",
                __func__, i, (long long) indices[i], session->tokens()+i);
            return false;
        }
        checked_indices.push_back(tensor->data); return true;
    }
    ggml_status compute(ggml_backend_t backend, ggml_tensor * op) {
        if (backend != config.backend) {
            LLAMA_LOG_WARN("%s: target execution backend mismatch op=%s actual=%p expected=%p\n",
                __func__, ggml_op_name(op->op), (void *) backend, (void *) config.backend);
            return GGML_STATUS_FAILED;
        }
        if (op->op != GGML_OP_SET_ROWS && op->op != GGML_OP_FLASH_ATTN_EXT) return GGML_STATUS_SUCCESS;
        if (!session || !session->active()) {
            LLAMA_LOG_WARN("%s: target execution without active session: op=%s\n", __func__, ggml_op_name(op->op));
            return GGML_STATUS_FAILED;
        }
        if (op->op == GGML_OP_SET_ROWS) {
            uint32_t layer; bool value;
            if (!plane(op->src[2],layer,value) || !validate_indices(op->src[1])) {
                LLAMA_LOG_WARN("%s: target SET_ROWS rejected source plane or indices, rows=%u\n",
                    __func__, queries);
                return GGML_STATUS_FAILED;
            }
            if (!value) {
                if (pending_k) {
                    LLAMA_LOG_WARN("%s: target K producer already pending at layer %u\n", __func__, pending_layer);
                    return GGML_STATUS_FAILED;
                }
                pending_k = op->src[0]; pending_layer = layer;
                pending_owner.reset(ggml_backend_buffer_retain(pending_k->view_src ? pending_k->view_src->buffer : pending_k->buffer));
                return GGML_STATUS_SUCCESS;
            }
            // The extra dependency keeps K's allocator slot live until this V operation completes.
            if (!pending_k || layer != pending_layer || op->src[3] != pending_k) {
                LLAMA_LOG_WARN("%s: target V pair mismatch layer=%u pending=%u hasK=%d dep=%d\n",
                    __func__, layer, pending_layer, int(pending_k != nullptr), int(op->src[3] == pending_k));
                return GGML_STATUS_FAILED;
            }
            const bool ok = session->produce(layer,pending_k,op->src[0]);
            if (!ok) LLAMA_LOG_WARN("%s: target KV publication failed at layer %u, rows %u\n",
                __func__, layer, queries);
            pending_k = nullptr; pending_owner.reset();
            return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
        uint32_t layer; bool value;
        if (pending_k || !plane(op->src[1],layer,value)) {
            LLAMA_LOG_WARN("%s: target attention rejected plane or pending K: pending=%d rows=%u\n",
                __func__, int(pending_k != nullptr), queries);
            return GGML_STATUS_FAILED;
        }
        float scale; std::memcpy(&scale,op->op_params,sizeof(scale));
        const bool ok = session->attention(layer,op->src[0],op->src[5] ? op->src[5] : op->src[3],op,scale);
        if (ok && !session->active()) {
            append_stats=session->sequence_stats();
            if (decoding) prefetch_next(after_append);
        }
        if (!ok) LLAMA_LOG_WARN("%s: target attention failed at layer %u, rows %u\n",
            __func__, layer, queries);
        return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }
    void abort() {
        if (transfer) transfer->finish();
        if (session) session->abort();
        pending_k = nullptr; pending_owner.reset(); checked_indices.clear();
    }
    void modified() {
        if (external_mutation) return;
        abort();
        external_mutation = content->invalidate();
    }
    bool reset(bool clear) {
        if (session && session->device_suspended()) return false;
        abort();
        if (clear) ggml_backend_buffer_clear(host->buffer(),0);
        if (!content->invalidate()) return false;
        external_mutation = false;
        suspended_tokens = 0;
        if (shared) return session && session->reconstruct(0);
        session.reset();
        return resize_attention(prefill_bytes,false) && make_session();
    }

    bool restore(size_t tokens) {
        if (session && session->device_suspended()) return false;
        if (!external_mutation || tokens > host->config().context_tokens) return false;
        suspended_tokens = tokens;
        bool restored = false;
        if (shared) {
            restored = session && session->reconstruct(tokens);
        } else {
            session.reset();
            restored = resize_attention(prefill_bytes,false) && make_session();
        }
        if (!restored) return false;
        external_mutation = false;
        return true;
    }

    bool truncate(size_t tokens) {
        if (session && session->device_suspended()) return false;
        if (external_mutation || !session || session->active() || tokens > session->tokens()) return false;
        if (tokens == session->tokens()) return true;
        abort();
        if (!content->invalidate_suffix(tokens)) return false;
        external_mutation = true;
        if (!session->reconstruct(tokens)) return false;
        suspended_tokens = tokens;
        external_mutation = false;
        return true;
    }
};

std::unique_ptr<llama_kv_stream_model> llama_kv_stream_model::create(const llama_kv_stream_model_config & config,
        uint32_t * unavailable_queries) {
    if (unavailable_queries) *unavailable_queries = 0;
    if (!config.backend || !config.pool_bytes || !config.max_batch_rows || !config.query_heads ||
            !config.max_decode_rows || config.max_decode_rows > 8 ||
            !config.host.context_tokens || config.host.context_tokens > size_t(INT32_MAX)-255 ||
            config.query_heads > SIZE_MAX/config.max_batch_rows ||
            config.auxiliary_cache_layers > 1 ||
            config.host.layers > UINT32_MAX - config.auxiliary_cache_layers) return {};
    auto * dev = ggml_backend_get_device(config.backend);
    auto * type = llama_kv_stream_device_buffer_type(dev);
    auto * host_type = llama_kv_stream_host_buffer_type(dev);
    if (!type || !host_type) return {};
    uint32_t unavailable = 0;
    if (!native_attention_available(config,unavailable)) {
        if (unavailable_queries) *unavailable_queries = unavailable;
        return {};
    }
    ggml_kv_stream_block_layout partial;
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
    const auto * partial_ops = get ? get() : nullptr;
    const bool geometry=partial_ops && partial_ops->version >= 15 && partial_ops->geometry;
    if ((config.host.shape.head_dim_k != 256 || config.host.shape.head_dim_v != 256) && !geometry) {
        if (unavailable_queries) *unavailable_queries=1;
        return {};
    }
    if (config.host.shape.head_dim_k != 256 || config.host.shape.head_dim_v != 256) {
        ggml_kv_stream_resume_plan probe;
        size_t bytes=0;
        const size_t tokens=(config.host.context_tokens+255)/256*256;
        if (!ggml_kv_stream_resume_plan_query(partial_ops,config.backend,
                {config.host.shape,config.query_heads,1,tokens},probe) ||
                !ggml_kv_stream_workspace_query(partial_ops,config.backend,
                    {config.host.shape,config.query_heads,uint32_t(std::min(size_t(config.max_batch_rows),tokens)),tokens},
                    ggml_kv_stream_workspace_kind::prefill,bytes)) {
            if (unavailable_queries) *unavailable_queries=1;
            return {};
        }
    }
    if (ggml_kv_stream_block_layout_make(size_t(config.max_batch_rows)*config.query_heads,
            size_t(config.host.shape.head_dim_v),partial).status != ggml_kv_stream_partial_status::success) return {};
    try {
        auto s = std::make_shared<implementation>(); s->config = config;
        if (!llama_kv_stream_writer::requirements(type,config.host.shape,config.max_batch_rows,s->writer_bytes)) return {};
        s->config.host.cache_id = model_cache_id.fetch_add(config.auxiliary_cache_layers ? 2 : 1, std::memory_order_relaxed);
        if (!s->config.host.cache_id ||
                s->config.host.cache_id > (UINT64_MAX-3)/4 - config.auxiliary_cache_layers) return {};
        const uint64_t auxiliary_id = s->config.host.cache_id + config.auxiliary_cache_layers;
        s->physical_policy.shape = s->config.host.shape;
        s->physical_policy.capabilities = s->config.host.capabilities;
        s->physical_policy.layers = s->config.host.layers + config.auxiliary_cache_layers;
        s->physical_policy.pool_bytes = config.pool_bytes;
        if (config.auxiliary_cache_layers) s->physical_policy.caches = {
            {s->config.host.cache_id, s->config.host.layers},
            {auxiliary_id, config.auxiliary_cache_layers},
        };
        llama_kv_stream_policy_state initial;
        if (llama_kv_stream_policy_initialize(s->physical_policy, initial).status !=
                llama_kv_stream_policy_status::success) return {};
        s->host = llama_kv_stream_host::create(s->config.host,host_type);
        if (!s->host) { LLAMA_LOG_ERROR("%s: authoritative host KV allocation/registration failed\n",__func__); return {}; }
        s->content = std::make_shared<llama_kv_stream_content>(s->host);
        if (config.auxiliary_cache_layers) {
            auto auxiliary_host_config = s->config.host;
            auxiliary_host_config.cache_id = auxiliary_id;
            auxiliary_host_config.layers = config.auxiliary_cache_layers;
            auto auxiliary_host = llama_kv_stream_host::create(auxiliary_host_config, host_type);
            if (!auxiliary_host) { LLAMA_LOG_ERROR("%s: auxiliary host KV allocation/registration failed\n",__func__); return {}; }
            auto auxiliary = llama_kv_stream_logical_cache::create(std::move(auxiliary_host));
            if (!auxiliary) return {};
            s->auxiliary_cache = std::shared_ptr<llama_kv_stream_logical_cache>(std::move(auxiliary));
            s->transfer=llama_kv_stream_transfer::create(config.backend);
            if (!s->transfer) return {};
        }
        ggml_kv_stream_resume_plan plan;
        s->prefill_bytes=s->host->layout().bytes;
        if (!std::getenv("LLAMA_KV_STREAM_PREFILL_GATHER") && partial_ops && partial_ops->version >= 13 && (partial_ops->prefill_workspace || geometry)) {
            size_t bytes=0;
            bool available=true;
            bool compact=partial_ops->version >= 14 && (partial_ops->prefill_mask_bounds || geometry) &&
                !std::getenv("LLAMA_KV_STREAM_DENSE_MASK");
            const uint32_t maximum=uint32_t(std::min(size_t(config.max_batch_rows),s->host->layout().tokens));
            for (uint32_t rows=1;rows<=maximum;++rows) {
                size_t bound=0;
                const ggml_kv_stream_attention_request request{config.host.shape,config.query_heads,rows,s->host->layout().tokens};
                if (!ggml_kv_stream_workspace_query(partial_ops,config.backend,request,
                        ggml_kv_stream_workspace_kind::prefill,bound)) { available=false; break; }
                bytes=std::max(bytes,bound);
                if (compact && !ggml_kv_stream_workspace_query(partial_ops,config.backend,request,
                        ggml_kv_stream_workspace_kind::prefill_mask_bounds,bound)) compact=false;
            }
            if (available) {
                if (bytes > SIZE_MAX-s->prefill_bytes) return {};
                s->prefill_span_bytes=bytes; s->prefill_bytes=bytes;
                s->compact_prefill_mask=compact;
            }
        }
        // Embedded MTP retains its separately qualified gathered-prefill path in the same serial scratch lease.
        if (s->auxiliary_cache) s->prefill_bytes=std::max(s->prefill_bytes,s->auxiliary_cache->host()->layout().bytes);
        const bool resume_capable = config.resume_decode && !std::getenv("LLAMA_KV_STREAM_DECODE_GATHER") &&
            partial_ops && partial_ops->version >= 5 && (partial_ops->resume_plan || geometry) && partial_ops->resume;
        const auto plan_resume = [&](uint32_t queries) {
            return ggml_kv_stream_resume_plan_query(partial_ops,config.backend,
                {config.host.shape,config.query_heads,queries,s->host->layout().tokens},plan);
        };
        const uint32_t decode_rows = std::min(2u,config.max_batch_rows);
        s->config.resume_decode = resume_capable &&
            (plan_resume(decode_rows) || (decode_rows == 2 && plan_resume(1)));
        if (config.max_decode_rows > 4 && !s->config.resume_decode) {
            LLAMA_LOG_ERROR("%s: no supported resumable KV attention for K=%s, V=%s, decode widths 1-%u; check compiled streamed kernels and device resources\n",
                __func__,ggml_type_name(ggml_type(config.host.shape.type_k)),ggml_type_name(ggml_type(config.host.shape.type_v)),config.max_decode_rows);
            return {};
        }
        s->decode_bytes = s->config.resume_decode ? plan.bytes : s->host->layout().bytes;
        if (s->config.resume_decode && partial_ops->version >= 10 && (partial_ops->decode_workspace || geometry)) {
            size_t bytes=0;
            if (!ggml_kv_stream_workspace_query(partial_ops,config.backend,
                    {config.host.shape,config.query_heads,std::min(config.max_decode_rows,config.max_batch_rows),s->host->layout().tokens},
                    ggml_kv_stream_workspace_kind::decode,bytes)) {
                LLAMA_LOG_ERROR("%s: unsupported decode KV attention workspace for K=%s, V=%s, maximum query width=%u; check compiled streamed kernels and device resources\n",
                    __func__,ggml_type_name(ggml_type(config.host.shape.type_k)),ggml_type_name(ggml_type(config.host.shape.type_v)),
                    std::min(config.max_decode_rows,config.max_batch_rows));
                return {};
            }
            s->span_bytes=bytes; s->decode_bytes=std::max(s->decode_bytes,bytes);
        } else if (s->config.resume_decode && partial_ops->version >= 9 && (partial_ops->mma_workspace || geometry)) {
            size_t mma_bytes=0;
            if (ggml_kv_stream_workspace_query(partial_ops,config.backend,
                    {config.host.shape,config.query_heads,decode_rows,s->host->layout().tokens},
                    ggml_kv_stream_workspace_kind::mma,mma_bytes)) {
                s->span_bytes = mma_bytes;
                s->decode_bytes=std::max(s->decode_bytes,mma_bytes);
            }
        }
        // No legal ring can exceed the entire configured device budget in encoded pages.
        if (!s->prepare_copies(config.shared_device_memory_bytes ? config.shared_device_memory_bytes : config.pool_bytes)) return {};
        if (!s->allocate_private()) { LLAMA_LOG_ERROR("%s: device KV grant allocation/binding failed\n",__func__); return {}; }
        const ggml_backend_execution_ops ops{
            [](void * p,const ggml_tensor * t) { return (*static_cast<std::shared_ptr<implementation> *>(p))->supports(t); },
            [](void * p,ggml_backend_t b,ggml_tensor * t) {
                auto & s = **static_cast<std::shared_ptr<implementation> *>(p);
                try { const auto result = s.compute(b,t); if (result != GGML_STATUS_SUCCESS) s.abort(); return result; }
                catch (const std::exception & error) {
                    LLAMA_LOG_ERROR("%s: target managed op %s threw: %s\n", __func__,
                        ggml_op_name(t->op), error.what());
                    s.abort(); return GGML_STATUS_FAILED;
                } catch (...) {
                    LLAMA_LOG_ERROR("%s: target managed op %s threw unknown exception\n",
                        __func__, ggml_op_name(t->op));
                    s.abort(); return GGML_STATUS_FAILED;
                }
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
    } catch (const std::bad_alloc &) { LLAMA_LOG_ERROR("%s: host KV metadata allocation failed\n",__func__); return {}; }
}

llama_kv_stream_model::~llama_kv_stream_model() { ggml_backend_buffer_free(proxy); }
ggml_backend_buffer_t llama_kv_stream_model::buffer() const noexcept { return proxy; }
std::shared_ptr<llama_kv_stream_host> llama_kv_stream_model::host() const noexcept { return impl->host; }
std::shared_ptr<llama_kv_stream_logical_cache> llama_kv_stream_model::auxiliary_cache() const noexcept {
    return impl->auxiliary_cache;
}
llama_kv_stream_binding_view llama_kv_stream_model::binding_view() const noexcept {
    return impl->session ? impl->session->binding_view() : llama_kv_stream_binding_view{};
}
bool llama_kv_stream_model::acquire_mtp_layer(size_t future_tokens, bool prefetch) {
    return impl->acquire_mtp_layer(future_tokens,prefetch);
}

bool llama_kv_stream_model::implementation::acquire_mtp_layer(size_t future_tokens, bool prefetch) {
    auto & s = *this;
    if (!s.session || s.external_mutation || s.mtp_owner || s.mtp_guard ||
            !s.mtp_plans.empty() || !s.auxiliary_cache || !complete()) return false;
    const auto mtp = s.auxiliary_cache;
    const size_t target_tokens = s.session->tokens();
    const size_t mtp_tokens = mtp->tokens();
    if (mtp_tokens < 4 || mtp_tokens > target_tokens) return false;
    if (target_tokens > s.config.host.context_tokens || future_tokens > 4 ||
            future_tokens > s.config.host.context_tokens - target_tokens) return false;
    const size_t reserved_tokens = target_tokens + future_tokens;
    const uint32_t physical_layer = s.config.host.layers;
    if (!s.session->reserve_complete_layer(physical_layer, reserved_tokens)) return false;
    const auto view = s.session->binding_view();
    if (!view.lease || !view.buffer || view.config.layers <= s.config.host.layers ||
            view.config.caches.size() != 2 ||
            view.config.caches[0].id != s.host->cache_id() ||
            view.config.caches[0].layers != s.config.host.layers ||
            view.config.caches[1].id != mtp->identity().id ||
            view.config.caches[1].layers != 1) return false;

    const uint64_t revision = s.session->layout_revision();
    const uint64_t generation = mtp->identity().generation;
    const uint64_t arena_generation=ggml_backend_memory_lease_generation(view.lease);
    if (s.mtp_resident_buffer != view.buffer || s.mtp_resident_revision != revision ||
            s.mtp_resident_arena_generation != arena_generation) {
        if (!mtp->content()->reset_mirror()) return false;
        s.mtp_resident_buffer=nullptr;
    }
    auto owner = llama_kv_stream_layer_lease_owner::create(view.lease,
        {view.config, s.session->policy(), reserved_tokens, revision, generation, target_tokens});
    if (!owner) {
        LLAMA_LOG_WARN("%s: MTP physical layout rejected target=%zu host=%zu reserved=%zu resident=%u ring=%u revision=%llu\n",
            __func__, target_tokens, mtp_tokens, reserved_tokens,
            s.session->policy().resident_pages_per_layer, s.session->policy().ring_slots,
            (unsigned long long) revision);
        return false;
    }
    const llama_kv_stream_complete_layer_request initial{
        physical_layer,1,revision,generation,mtp->identity().id,mtp_tokens};
    mtp_lease_ptr reservation(owner->acquire(initial),llama_kv_stream_complete_layer_lease_free);
    if (!reservation) return false;
    auto guard=owner->hold_ring();
    // Drain target lookahead before any MTP population writes into the reserved ring slots.
    if (!guard || !s.session->set_ring_guard(guard)) return false;
    struct admission_guard {
        llama_kv_stream_session * session;
        llama_kv_stream_transfer * transfer;
        bool adopted=false;
        ~admission_guard() {
            if (!adopted) {
                if (transfer) transfer->finish();
                session->set_ring_guard({});
            }
        }
    } admission{s.session.get(),prefetch ? s.transfer.get() : nullptr};
    std::vector<mtp_lease_ptr> plans;
    plans.reserve(4);
    for (uint32_t width = 1; width <= 4; ++width) {
        const llama_kv_stream_complete_layer_request request{
            physical_layer, width, revision, generation, mtp->identity().id, mtp_tokens};
        auto * raw = width == 1 ?
            (prefetch && s.transfer ? owner->prefetch(*s.transfer,s.config.backend,request,*mtp,true) :
                owner->acquire_populated(s.config.backend, request, *mtp,true)) :
            owner->acquire(request);
        if (!raw) {
            const auto frontier = mtp->frontiers();
            LLAMA_LOG_WARN("%s: MTP TG%u population/plan rejected target=%zu host=%zu reserved=%zu generation=%llu frontiers=%zu/%zu/%zu/%zu\n",
                __func__, width, target_tokens, mtp_tokens, reserved_tokens,
                (unsigned long long) generation, frontier.reserved, frontier.host, frontier.device, frontier.committed);
            return false;
        }
        plans.emplace_back(raw, llama_kv_stream_complete_layer_lease_free);
    }
    s.mtp_owner = std::move(owner);
    s.mtp_plans = std::move(plans);
    s.mtp_guard = std::move(guard);
    s.mtp_reserved_tokens = reserved_tokens;
    s.mtp_resident_buffer=view.buffer;
    s.mtp_resident_revision=revision;
    s.mtp_resident_arena_generation=arena_generation;
    admission.adopted=true;
    return true;
}

bool llama_kv_stream_model::advance_mtp_layer_tail() {
    auto & s = *impl;
    if (!s.session || !s.mtp_owner || !s.mtp_guard || s.mtp_plans.size() != 4 ||
            !s.auxiliary_cache || !complete() ||
            s.session->layout_revision() != s.mtp_owner->layout_revision()) return false;
    const auto cache = s.auxiliary_cache;
    if (cache->tokens() < 4 || cache->tokens() > s.mtp_reserved_tokens) return false;
    const uint64_t generation = cache->identity().generation;
    if (s.mtp_owner->content_generation() != generation) {
        llama_kv_stream_population_stats delta;
        if (!s.mtp_owner->publish_tail(s.config.backend, s.mtp_plans.front().get(),
                *cache, delta)) return false;
    }
    // Allocation can fail after the tail copy. Keep the old (now invalid) handles
    // until all replacements exist; a retry here will not repeat the H2D transfer.
    std::vector<mtp_lease_ptr> refreshed;
    refreshed.reserve(4);
    for (uint32_t width = 1; width <= 4; ++width) {
        const llama_kv_stream_complete_layer_request request{
            s.config.host.layers, width, s.session->layout_revision(),
            generation, cache->identity().id, cache->tokens()};
        auto * raw = s.mtp_owner->acquire(request);
        if (!raw) return false;
        refreshed.emplace_back(raw, llama_kv_stream_complete_layer_lease_free);
    }
    s.mtp_plans.swap(refreshed);
    return true;
}

bool llama_kv_stream_model::stage_mtp_tail_async(
        ggml_backend_t backend, size_t first, bool value,
        const ggml_tensor * encoded, size_t row, size_t count,
        llama_kv_stream_population_stats & staged) {
    auto & s = *impl;
    return backend && s.session && s.mtp_owner && s.mtp_guard &&
        s.mtp_plans.size() == 4 && s.auxiliary_cache &&
        s.auxiliary_cache->tokens() == first &&
        s.session->layout_revision() == s.mtp_owner->layout_revision() &&
        s.mtp_owner->stage_tail_async(backend, s.mtp_plans.front().get(),
            first, value, encoded, row, count, staged);
}

llama_kv_stream_complete_layer_lease_t llama_kv_stream_model::provisional_mtp_layer(
        uint32_t query_tokens, size_t active_tokens) {
    auto & s = *impl;
    if (!s.session || !s.mtp_owner || !s.mtp_guard || !s.auxiliary_cache ||
            s.mtp_plans.size() != 4 || !query_tokens || query_tokens > 4 ||
            active_tokens <= s.auxiliary_cache->tokens() ||
            active_tokens > s.mtp_reserved_tokens ||
            s.session->layout_revision() != s.mtp_owner->layout_revision()) return nullptr;
    return s.mtp_owner->acquire({
        s.config.host.layers, query_tokens, s.session->layout_revision(),
        s.mtp_owner->content_generation(), s.auxiliary_cache->identity().id, active_tokens});
}

bool llama_kv_stream_model::advance_mtp_layer_tail_staged(
        const llama_kv_stream_population_stats & staged) {
    auto & s = *impl;
    if (!s.session || !s.mtp_owner || !s.mtp_guard || s.mtp_plans.size() != 4 ||
            !s.auxiliary_cache || !complete() ||
            s.session->layout_revision() != s.mtp_owner->layout_revision()) return false;
    const auto cache = s.auxiliary_cache;
    const uint64_t generation = cache->identity().generation;
    if (cache->tokens() < 4 || cache->tokens() > s.mtp_reserved_tokens ||
            generation <= s.mtp_owner->content_generation() ||
            !s.mtp_owner->adopt_staged_tail(s.mtp_plans.front().get(), *cache, staged))
        return false;
    std::vector<mtp_lease_ptr> refreshed;
    refreshed.reserve(4);
    for (uint32_t width = 1; width <= 4; ++width) {
        const llama_kv_stream_complete_layer_request request{
            s.config.host.layers, width, s.session->layout_revision(),
            generation, cache->identity().id, cache->tokens()};
        auto * raw = s.mtp_owner->acquire(request);
        if (!raw) return false;
        refreshed.emplace_back(raw, llama_kv_stream_complete_layer_lease_free);
    }
    s.mtp_plans.swap(refreshed);
    return true;
}


bool llama_kv_stream_model::truncate_mtp_layer(size_t tokens) {
    auto & s = *impl;
    const auto cache = s.auxiliary_cache;
    if (!cache || tokens > cache->tokens() || !complete()) return false;
    if (tokens == cache->tokens()) return true;
    if (!s.mtp_owner) return cache->truncate(tokens);
    if (!s.session || s.session->layout_revision() != s.mtp_owner->layout_revision() ||
            s.mtp_plans.size() != 4) return false;
    if (!cache->truncate(tokens)) return false;
    if (tokens < 4 || !s.mtp_owner->adopt_truncated_prefix(
            s.mtp_plans.front().get(), *cache)) return release_mtp_layer();
    try {
        std::vector<mtp_lease_ptr> refreshed;
        refreshed.reserve(4);
        for (uint32_t width = 1; width <= 4; ++width) {
            const llama_kv_stream_complete_layer_request request{
                s.config.host.layers, width, s.session->layout_revision(),
                cache->identity().generation, cache->identity().id, tokens};
            auto * raw = s.mtp_owner->acquire(request);
            if (!raw) return release_mtp_layer();
            refreshed.emplace_back(raw, llama_kv_stream_complete_layer_lease_free);
        }
        s.mtp_plans.swap(refreshed);
        return true;
    } catch (const std::bad_alloc &) {
        return release_mtp_layer();
    }
}
bool llama_kv_stream_model::release_mtp_layer() {
    auto & s = *impl;
    if (!s.mtp_owner) return true;
    if (s.transfer && !s.transfer->finish()) return false;
    if (!s.session || (!s.auxiliary_handoff && !s.session->set_ring_guard({}))) return false;
    s.mtp_guard.reset();
    s.mtp_plans.clear();
    s.mtp_owner.reset();
    s.mtp_reserved_tokens = 0;
    s.auxiliary_handoff=false;
    return true;
}

// Register an actual cache consumer rather than infer it from the model architecture.
bool llama_kv_stream_model::schedule_next_cache(uint64_t cache_id, uint32_t future_tokens) {
    if (future_tokens > impl->config.max_decode_rows) return false;
    llama_kv_stream_next_consumer next;
    next.cache_id=cache_id; next.future_tokens=cache_id ? future_tokens : 0;
    return schedule_next_consumer(llama_kv_stream_access_role::append,next);
}

// Repeating an unchanged intent must not drain the transport that already serves it.
static bool same_consumer(const llama_kv_stream_next_consumer & a,const llama_kv_stream_next_consumer & b) {
    return a.access == b.access && a.cache_id == b.cache_id && a.reader_id == b.reader_id && a.layer == b.layer &&
        a.query_tokens == b.query_tokens && a.query_heads == b.query_heads && a.future_tokens == b.future_tokens;
}

bool llama_kv_stream_model::schedule_next_consumer(llama_kv_stream_access_role producer,const llama_kv_stream_next_consumer & next) {
    auto & s=*impl;
    if ((producer != llama_kv_stream_access_role::append && producer != llama_kv_stream_access_role::read) ||
            (s.session && s.session->active())) return false;
    if (next.access == llama_kv_stream_access_role::append) {
        if (next.reader_id || next.layer || next.query_tokens || next.query_heads || next.future_tokens > s.config.max_decode_rows ||
                (!next.cache_id && next.future_tokens) || (next.cache_id && next.cache_id != s.host->cache_id() &&
                (!s.auxiliary_cache || next.cache_id != s.auxiliary_cache->identity().id))) return false;
    } else if (next.access == llama_kv_stream_access_role::read) {
        if (next.cache_id != s.host->cache_id() || !next.reader_id || next.reader_id == next.cache_id ||
                next.layer >= s.host->config().layers || !next.query_tokens || next.query_tokens > s.config.max_decode_rows ||
                next.query_tokens > s.config.max_batch_rows || !next.query_heads || next.query_heads > s.config.query_heads ||
                next.query_heads%s.config.host.shape.heads || next.future_tokens) return false;
    } else return false;
    auto & current=producer == llama_kv_stream_access_role::append ? s.after_append : s.after_read;
    if (same_consumer(current,next)) {
        if (producer == llama_kv_stream_access_role::append && !next.cache_id) s.after_read={};
        return true;
    }
    if (producer == llama_kv_stream_access_role::append) {
        const bool automatic=s.config.cross_token_prefetch && s.config.resume_decode && !next.cache_id;
        if (s.session && !s.session->set_automatic_prefetch(automatic)) return false;
    }
    current=next;
    if (producer == llama_kv_stream_access_role::append && !next.cache_id) s.after_read={};
    return true;
}

llama_kv_stream_next_consumer llama_kv_stream_model::next_consumer(llama_kv_stream_access_role producer) const noexcept {
    if (producer == llama_kv_stream_access_role::append) return impl->after_append;
    return producer == llama_kv_stream_access_role::read ? impl->after_read : llama_kv_stream_next_consumer{};
}

llama_kv_stream_prefetch_stats llama_kv_stream_model::sequence_stats(llama_kv_stream_access_role producer) const noexcept {
    if (producer == llama_kv_stream_access_role::append) return impl->append_stats;
    return producer == llama_kv_stream_access_role::read ? impl->read_stats : llama_kv_stream_prefetch_stats{};
}

// A confidence cutoff can finish before the configured last prediction; the next consumer is still the target.
bool llama_kv_stream_model::finish_read_access() {
    llama_kv_stream_next_consumer next; next.cache_id=impl->host->cache_id();
    if (!schedule_next_consumer(llama_kv_stream_access_role::read,next)) return false;
    impl->prefetch_next(next);
    return complete();
}

bool llama_kv_stream_model::wait_auxiliary_population(ggml_backend_t consumer) {
    auto & s=*impl;
    return s.mtp_owner && !s.mtp_plans.empty() && s.auxiliary_cache &&
        s.mtp_owner->wait_population(s.mtp_plans.front().get(),consumer,*s.auxiliary_cache);
}

// Queue target lookahead behind the final auxiliary reader, while retaining its publication metadata.
bool llama_kv_stream_model::prefetch_target_after(ggml_backend_t last_reader) {
    auto & s=*impl;
    if (s.auxiliary_handoff) return true;
    if (!s.after_append.cache_id || !s.transfer || !s.session || !s.mtp_owner ||
            !s.transfer->order(last_reader,s.config.backend) || !s.session->release_ring_guard_ordered()) return false;
    s.auxiliary_handoff=true;
    s.session->prefetch_decode(0);
    return true;
}

ggml_kv_stream_span_plan_t llama_kv_stream_model::mtp_layer_plan(uint32_t queries) const noexcept {
    const auto & plans = impl->mtp_plans;
    return queries >= 1 && queries <= plans.size() ?
        llama_kv_stream_complete_layer_lease_plan(plans[queries - 1].get()) : nullptr;
}

size_t llama_kv_stream_model::mtp_reserved_tokens() const noexcept {
    return impl->mtp_owner ? impl->mtp_reserved_tokens : 0;
}
llama_kv_stream_population_stats llama_kv_stream_model::mtp_layer_population() const noexcept {
    return impl->mtp_plans.empty() ? llama_kv_stream_population_stats{} :
        llama_kv_stream_complete_layer_lease_population(impl->mtp_plans.front().get());
}

bool llama_kv_stream_model::has_mtp_layer() const noexcept {
    return impl->mtp_owner != nullptr;
}

bool llama_kv_stream_model::set_ring_guard(std::shared_ptr<const llama_kv_stream_ring_guard> guard) {
    return impl->session && !impl->external_mutation && !impl->mtp_owner &&
        impl->session->set_ring_guard(std::move(guard));
}
bool llama_kv_stream_model::begin(size_t active,uint32_t queries,bool decode) {
    auto & s = *impl;
    if (!s.session || s.session->device_suspended() || s.external_mutation || s.session->active() || s.session->failed() || !queries || queries > s.config.max_batch_rows ||
            (decode && queries > s.config.max_decode_rows) || active < s.session->tokens() || active-s.session->tokens() != queries ||
            active > s.host->config().context_tokens) return false;
    // A new KV page can require physical repartition; retire the MTP guard
    // before admitting that page, never after an already submitted target copy.
    const size_t page = s.config.host.shape.page_tokens;
    if (s.mtp_owner && page && s.session->tokens() &&
            (active - 1)/page != (s.session->tokens() - 1)/page && !release_mtp_layer()) return false;
    if (!s.resize_attention(decode ? s.decode_bytes : s.prefill_bytes,decode)) return false;
    if (!s.session->begin(active,queries,decode)) {
        // Feedback can also request repartition within one page. A guarded
        // rejection that did not poison the session is safe to retry unguarded.
        if (!s.mtp_owner || s.session->failed() || !release_mtp_layer() ||
                !s.session->begin(active,queries,decode)) return false;
    }
    s.decoding=decode;
    s.queries = queries; s.pending_k = nullptr; s.pending_owner.reset(); s.checked_indices.clear(); return true;
}
bool llama_kv_stream_model::complete() const noexcept { return impl->session && !impl->session->device_suspended() && !impl->external_mutation && !impl->session->active() && !impl->session->failed() && !impl->pending_k; }
bool llama_kv_stream_model::read_source(llama_memory_cache_requirement & output, uint64_t * revision) const noexcept {
    if (!complete() || !impl->session->read_source(output)) return false;
    if (revision) *revision=impl->session->layout_revision();
    return true;
}
bool llama_kv_stream_model::read_attention(const llama_memory_cache_read_binding & binding,
        ggml_tensor * q,ggml_tensor * mask,ggml_tensor * output,float scale) {
    if (!complete() || !impl->session->read_attention(binding,q,mask,output,scale)) return false;
    impl->read_stats=impl->session->sequence_stats();
    impl->prefetch_next(impl->after_read);
    return true;
}
bool llama_kv_stream_model::prefetch_read(const llama_memory_cache_read_binding & binding,uint32_t queries,uint32_t heads) {
    return complete() && impl->session->prefetch_read(binding,queries,heads);
}
bool llama_kv_stream_model::device_suspended() const noexcept { return impl->session ? impl->session->device_suspended() : impl->suspended; }
bool llama_kv_stream_model::suspend_ready() const noexcept {
    return complete() && (!impl->auxiliary_cache || impl->auxiliary_cache->ready());
}
bool llama_kv_stream_model::resume_ready() const noexcept {
    return impl->session && impl->session->device_suspended() && !impl->session->failed() &&
        !impl->external_mutation && !impl->pending_k && (!impl->auxiliary_cache || impl->auxiliary_cache->ready());
}
bool llama_kv_stream_model::prefetch_primed() const noexcept { return impl->session && impl->session->prefetch_primed(); }
void llama_kv_stream_model::abort() { impl->abort(); }
bool llama_kv_stream_model::reset(bool clear) {
    return release_mtp_layer() && impl->reset(clear);
}
bool llama_kv_stream_model::restore(size_t tokens) { return !impl->mtp_owner && impl->restore(tokens); }
bool llama_kv_stream_model::truncate(size_t tokens) { return !impl->mtp_owner && impl->truncate(tokens); }
size_t llama_kv_stream_model::tokens() const noexcept { return impl->session ? impl->session->tokens() : 0; }
uint32_t llama_kv_stream_model::max_batch_rows() const noexcept { return impl->config.max_batch_rows; }
uint32_t llama_kv_stream_model::max_decode_rows() const noexcept { return impl->config.max_decode_rows; }
bool llama_kv_stream_model::bounded_prefill() const noexcept { return impl->prefill_span_bytes != 0; }
bool llama_kv_stream_model::compact_prefill_mask() const noexcept { return impl->compact_prefill_mask; }
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
        impl->writer_bytes,
        impl->prefill_bytes,
        impl->decode_bytes,
        std::max(size_t(128),ggml_backend_buft_get_alignment(type)),
        impl->config.shared_device_memory_bytes,
        impl->config.pool_bytes,
    };
    if (impl->auxiliary_cache) {
        ggml_kv_stream_execution page;
        const auto & shape = impl->physical_policy.shape;
        if (ggml_kv_stream_resolve(shape,impl->physical_policy.capabilities,
                size_t(shape.page_tokens),page).status != ggml_kv_stream_status::success) return false;
        const size_t tokens = impl->config.host.context_tokens;
        const size_t pages = tokens/size_t(shape.page_tokens) + (tokens%size_t(shape.page_tokens) != 0);
        if (!page.storage.bytes || pages > (SIZE_MAX-page.conversion.bytes)/page.storage.bytes) return false;
        output.pool_decode_min_bytes = std::max(output.pool_bytes,
            pages*page.storage.bytes+page.conversion.bytes);
    }
    return true;
}

bool llama_kv_stream_model::prepare_shared_memory() {
    auto & s = *impl;
    if (s.mtp_owner || s.shared || s.pending_k || (s.session && s.session->active())) return false;
    if (s.suspended && !s.session) return true;
    s.suspended_tokens = s.session ? s.session->tokens() : s.suspended_tokens;
    s.release_device_memory();
    s.suspended = true;
    return true;
}

bool llama_kv_stream_model::resume_private_memory() {
    auto & s = *impl;
    return !s.mtp_owner && !s.shared && s.suspended && !s.session && s.allocate_private();
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
    if (!memory_requirements(requirements) || s.mtp_owner || !s.suspended || s.session || s.shared ||
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
    if (!s.prepare_copies(parent_bytes)) return false;
    auto candidate = s.create_session(
        retained,binding.pool_resource,binding.writer_resource,binding.attention_resource,
        binding.prefill_stage,binding.decode_stage,binding.suspend_stage);
    if (!candidate) return false;

    s.session = std::move(candidate);
    s.pool_resource = binding.pool_resource;
    s.writer_resource = binding.writer_resource;
    s.attention_resource = binding.attention_resource;
    s.prefill_stage = binding.prefill_stage;
    s.decode_stage = binding.decode_stage;
    s.suspend_stage = binding.suspend_stage;
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
    if (s.mtp_owner || s.pending_k || (s.session && s.session->active())) return false;
    s.suspended_tokens = s.session ? s.session->tokens() : s.suspended_tokens;
    s.release_device_memory();
    s.pool_resource = s.writer_resource = s.attention_resource = 0;
    s.prefill_stage = s.decode_stage = s.suspend_stage = 0;
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
ggml_backend_buffer_t llama_kv_stream_model::mtp_attention_workspace() const noexcept {
    return impl->session ? impl->session->attention_workspace_buffer() : nullptr;
}
ggml_backend_buffer_t llama_kv_stream_model::mtp_writer_workspace() const noexcept {
    return impl->session ? impl->session->writer_workspace_buffer() : nullptr;
}

bool llama_kv_stream_model::runtime_diagnostics(
        llama_kv_stream_runtime_diagnostics & output) const noexcept {
    if (!impl->session) return false;
    if (impl->session->device_suspended()) {
        output = {};
        output.layout_revision = impl->session->layout_revision();
        return true;
    }
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
