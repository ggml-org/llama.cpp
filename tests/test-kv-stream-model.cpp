#include "kv-stream-block-test.h"
#include "../src/llama-kv-stream-model.h"
#include "../src/llama-context-memory.h"

// Quantization error is not adapter error: compare the exact bytes from an ordinary CUDA producer graph.
static std::vector<uint8_t> reference_bytes(ggml_backend_t backend,const std::vector<float> & data,ggml_type type,float scale) {
    const size_t rows = data.size()/512;
    ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
    auto * source = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,512,rows);
    auto * indices = ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I64,rows);
    auto * destination = ggml_new_tensor_2d(ctx.get(),type,512,rows);
    auto * output = ggml_set_rows(ctx.get(),destination,ggml_scale(ctx.get(),source,scale),indices);
    auto * graph = ggml_new_graph_custom(ctx.get(),64,false); ggml_build_forward_expand(graph,output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend)); GGML_ASSERT(buffer);
    std::vector<int64_t> ids(rows); for (size_t i = 0; i < rows; ++i) ids[i] = int64_t(i);
    ggml_backend_tensor_set(source,data.data(),0,data.size()*sizeof(float));
    ggml_backend_tensor_set(indices,ids.data(),0,ids.size()*sizeof(int64_t));
    llama_memory_cuda_executor executor(backend); GGML_ASSERT(executor.bind(graph,{},1));
    GGML_ASSERT(executor.compute_async({},1) == GGML_STATUS_SUCCESS && executor.drain());
    std::vector<uint8_t> result(ggml_nbytes(output)); ggml_backend_tensor_get(output,result.data(),0,result.size()); return result;
}

struct parent_view_fault {
    using factory = ggml_backend_buffer_t (*)(ggml_backend_buffer_t,size_t,size_t);
    inline static parent_view_fault * active = nullptr;
    ggml_backend_buffer_t parent;
    factory original;
    size_t failures;
    size_t calls = 0;

    parent_view_fault(ggml_backend_buffer_t parent,size_t failures) :
        parent(parent),original(parent ? parent->view_buffer : nullptr),failures(failures) {
        GGML_ASSERT(parent && original && !active);
        active = this;
        parent->view_buffer = create;
    }
    ~parent_view_fault() {
        parent->view_buffer = original;
        active = nullptr;
    }
    static ggml_backend_buffer_t create(
            ggml_backend_buffer_t parent,size_t offset,size_t size) {
        GGML_ASSERT(active && parent == active->parent);
        ++active->calls;
        if (active->failures) {
            --active->failures;
            return nullptr;
        }
        return active->original(parent,offset,size);
    }
};

int main(int argc,char ** argv) {
    testing t;
    if (argc < 2 || std::strcmp(argv[1],"--cuda")) {
        t.assert_true(!llama_kv_stream_model::create({})); return t.summary();
    }
    ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr)), cpu(ggml_backend_cpu_init());
    t.test("fixed_compute_and_kv_grants_share_one_device_parent", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page;
        ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        if (!t.assert_true(bool(model))) return;

        auto * device_type = llama_kv_stream_device_buffer_type(dev);
        auto * cpu_type = ggml_backend_cpu_buffer_type();
        std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
        std::vector<ggml_backend_buffer_type_t> types{device_type,cpu_type};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(
            backends.data(),types.data(),types.size(),256,false,true));
        if (!t.assert_true(bool(sched))) return;

        const size_t device_alignment = ggml_backend_buft_get_alignment(device_type);
        const size_t cpu_alignment = ggml_backend_buft_get_alignment(cpu_type);
        llama_compute_workspace_plan plan;
        plan.groups = {
            {device_type,2*1048576,device_alignment,0},
            {cpu_type,8192,cpu_alignment,1},
        };
        plan.phase_sizes = {
            {2*1048576,8192},
            {1048576,4096},
        };

        llama_kv_stream_memory_requirements requirements;
        if (!t.assert_true(model->memory_requirements(requirements))) return;
        auto owner = llama_context_memory::create(sched.get(),backends,plan,model.get());
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(owner->shares_kv_memory());
        t.assert_true(model->uses_shared_memory());
        t.assert_true(!model->prepare_shared_memory());
        t.assert_true(owner->shared_parent() == model->shared_parent());
        t.assert_true(owner->shared_parent() != nullptr);
        t.assert_equal(owner->shared_parent_capacity(),
            ggml_backend_buffer_get_size(owner->shared_parent()));
        const size_t expected_grants = requirements.pool_bytes + requirements.writer_bytes +
            std::max(requirements.attention_prefill_bytes,requirements.attention_decode_bytes);
        t.assert_equal(expected_grants,model->device_grant_bytes());
        t.assert_equal(plan.groups[0].size,
            ggml_backend_sched_get_buffer_size(sched.get(),backend.get()));
        size_t expected_parent = plan.groups[0].size;
        for (size_t bytes : {requirements.pool_bytes,requirements.writer_bytes,
                std::max(requirements.attention_prefill_bytes,requirements.attention_decode_bytes)}) {
            expected_parent = (expected_parent+device_alignment-1)&~(device_alignment-1);
            expected_parent += bytes;
        }
        t.assert_equal(expected_parent,owner->shared_parent_capacity());
        auto * base = ggml_backend_buffer_get_base(owner->shared_parent());
        const auto initial_generation = owner->shared_arena_generation();
        const auto initial_pool = model->pool_grant_bytes();
        const auto initial_writer = model->writer_grant_bytes();
        const auto initial_attention = model->attention_grant_bytes();
        t.assert_equal(requirements.pool_bytes,initial_pool);
        t.assert_equal(requirements.writer_bytes,initial_writer);
        t.assert_equal(requirements.attention_prefill_bytes,initial_attention);
        t.assert_equal(uint64_t(0),owner->phase_transition_count());
        ggml_backend_buffer_clear(model->buffer(),0);
        t.assert_true(model->restore(513));
        t.assert_equal(size_t(513),model->tokens());

        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::prefill,513,true,true,false}).status ==
            llama_memory_text_phase_status::changed);
        t.assert_equal(uint64_t(0),owner->phase_transition_count());
        t.assert_equal(initial_generation,owner->shared_arena_generation());

        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::decode,1,true,true,false}).status ==
            llama_memory_text_phase_status::changed);
        t.assert_equal(uint64_t(1),owner->phase_transition_count());
        t.assert_true(owner->shared_arena_generation() > initial_generation);
        t.assert_equal(plan.phase_sizes[1][0],
            ggml_backend_sched_get_buffer_size(sched.get(),backend.get()));
        t.assert_true(model->pool_grant_bytes() > initial_pool);
        t.assert_equal(initial_writer,model->writer_grant_bytes());
        t.assert_equal(requirements.attention_decode_bytes,model->attention_grant_bytes());
        t.assert_equal(model->pool_grant_bytes()+model->writer_grant_bytes()+
            model->attention_grant_bytes(),model->device_grant_bytes());
        const auto decode_generation = owner->shared_arena_generation();
        const auto decode_pool = model->pool_grant_bytes();
        t.out << "shared pool bytes: prefill=" << initial_pool << " decode=" << decode_pool
              << " reclaimed=" << decode_pool-initial_pool << '\n';

        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::decode,1,true,true,false}).status ==
            llama_memory_text_phase_status::unchanged);
        t.assert_equal(uint64_t(1),owner->phase_transition_count());
        t.assert_equal(decode_generation,owner->shared_arena_generation());
        t.assert_equal(decode_pool,model->pool_grant_bytes());

        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::prefill,128,true,true,false}).status ==
            llama_memory_text_phase_status::changed);
        t.assert_equal(uint64_t(2),owner->phase_transition_count());
        t.assert_equal(plan.phase_sizes[0][0],
            ggml_backend_sched_get_buffer_size(sched.get(),backend.get()));
        t.assert_equal(initial_pool,model->pool_grant_bytes());
        t.assert_equal(initial_attention,model->attention_grant_bytes());
        t.assert_true(base == ggml_backend_buffer_get_base(owner->shared_parent()));

        auto * consumer = model->memory_consumer();
        t.assert_true(consumer != nullptr);
        t.assert_true(model->reset(false));
        t.assert_true(consumer == model->memory_consumer());
        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::decode,1,true,true,false}).status ==
            llama_memory_text_phase_status::changed);
        t.assert_equal(uint64_t(3),owner->phase_transition_count());
        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::prefill,128,true,true,false}).status ==
            llama_memory_text_phase_status::changed);
        t.assert_equal(uint64_t(4),owner->phase_transition_count());
        t.assert_equal(size_t(0),model->tokens());
        t.assert_true(model->complete());

        owner.reset();
        t.assert_true(!model->uses_shared_memory());
        t.assert_true(model->shared_parent() == nullptr);
    });

    t.test("interrupted_phase_transition_recovers_and_retries", [&](testing & t) {
        {
            fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
            ggml_kv_stream_layout page;
            ggml_kv_stream_layout_make(f.policy.shape,256,page);
            auto model = llama_kv_stream_model::create(
                {backend.get(),f.host->config(),page.bytes*4,256,4});
            if (!t.assert_true(bool(model))) return;

            auto * device_type = llama_kv_stream_device_buffer_type(dev);
            auto * cpu_type = ggml_backend_cpu_buffer_type();
            std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
            std::vector<ggml_backend_buffer_type_t> types{device_type,cpu_type};
            ggml_backend_sched_ptr sched(ggml_backend_sched_new(
                backends.data(),types.data(),types.size(),256,false,true));
            llama_compute_workspace_plan plan;
            plan.groups = {
                {device_type,2*1048576,ggml_backend_buft_get_alignment(device_type),0},
                {cpu_type,8192,ggml_backend_buft_get_alignment(cpu_type),1},
            };
            plan.phase_sizes = {{2*1048576,8192},{1048576,4096}};
            auto owner = llama_context_memory::create(
                sched.get(),backends,plan,model.get());
            if (!t.assert_true(bool(owner))) return;
            const auto initial_pool = model->pool_grant_bytes();
            auto * base = ggml_backend_buffer_get_base(owner->shared_parent());
            ggml_backend_buffer_clear(model->buffer(),0);
            if (!t.assert_true(model->restore(513))) return;
            t.assert_true(owner->signal_text_phase({
                llama_memory_text_phase::prefill,513,true,true,false}).status ==
                llama_memory_text_phase_status::changed);

            size_t calls = 0;
            {
                parent_view_fault fault(
                    owner->shared_parent(),1);
                const auto failed = owner->signal_text_phase({
                    llama_memory_text_phase::decode,1,true,true,false});
                t.assert_true(failed.status ==
                    llama_memory_text_phase_status::transition_failed);
                calls = fault.calls;
            }
            t.assert_true(calls > 0);
            t.assert_true(base ==
                ggml_backend_buffer_get_base(owner->shared_parent()));
            t.assert_equal(initial_pool,model->pool_grant_bytes());
            t.assert_equal(size_t(513),model->tokens());
            t.assert_true(owner->text_phase().phase ==
                llama_memory_text_phase::prefill);

            const auto retry = owner->signal_text_phase({
                llama_memory_text_phase::decode,1,true,true,false});
            t.assert_true(model->complete());
            t.assert_true(retry.status ==
                llama_memory_text_phase_status::changed);
            t.assert_true(model->pool_grant_bytes() > initial_pool);
            t.assert_true(owner->signal_text_phase({
                llama_memory_text_phase::prefill,7,true,true,false}).status ==
                llama_memory_text_phase_status::changed);
            t.assert_equal(initial_pool,model->pool_grant_bytes());
        }
    });

    t.test("shared_parent_mismatch_rejects_and_restores_private_model", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page;
        ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        if (!t.assert_true(bool(model))) return;
        const size_t private_bytes = model->granted_bytes();

        auto * ordinary = ggml_backend_get_default_buffer_type(backend.get());
        auto * cpu_type = ggml_backend_cpu_buffer_type();
        std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
        std::vector<ggml_backend_buffer_type_t> types{ordinary,cpu_type};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(
            backends.data(),types.data(),types.size(),256,false,true));
        llama_compute_workspace_plan plan;
        plan.groups = {
            {ordinary,1048576,ggml_backend_buft_get_alignment(ordinary),0},
            {cpu_type,8192,ggml_backend_buft_get_alignment(cpu_type),1},
        };
        plan.phase_sizes = {
            {1048576,8192},
            {524288,4096},
        };
        t.assert_true(!llama_context_memory::create(sched.get(),backends,plan,model.get()));
        t.assert_true(!model->uses_shared_memory());
        t.assert_true(model->shared_parent() == nullptr);
        t.assert_true(model->complete());
        t.assert_equal(private_bytes,model->granted_bytes());
    });

    t.test("failed_scratch_replacement_keeps_host_state_and_can_retry", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model=llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        if (!t.assert_true(bool(model))) return;
        const auto initial=model->granted_bytes(); auto host=model->host();
        auto * type=llama_kv_stream_device_buffer_type(dev);
        const auto allocate=type->iface.alloc_buffer;
        type->iface.alloc_buffer=[](ggml_backend_buffer_type_t,size_t)->ggml_backend_buffer_t { return nullptr; };
        const bool began=model->begin(1,1,true);
        type->iface.alloc_buffer=allocate;
        t.assert_true(!began && model->complete() && model->tokens()==0 && model->host()==host);
        t.assert_true(model->granted_bytes()<initial);
        t.assert_true(model->begin(1,1,true)); model->abort();
        t.assert_true(model->reset(false)); t.assert_equal(initial,model->granted_bytes());
    });
    t.test("suffix_truncation_reopens_frontier_without_erasing_host_bytes", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model=llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        if (!t.assert_true(bool(model))) return;
        ggml_backend_buffer_clear(model->buffer(),0);
        t.assert_true(model->restore(513));
        auto * raw=static_cast<uint8_t *>(ggml_backend_buffer_get_base(model->host()->buffer()));
        raw[model->host()->bytes()-1]=91;
        t.assert_true(!model->truncate(514));
        t.assert_true(model->truncate(256));
        t.assert_equal(size_t(256),model->tokens());
        t.assert_equal(uint8_t(91),raw[model->host()->bytes()-1]);
        t.assert_true(model->begin(257,1,true)); model->abort();
    });
    t.test("ordinary_graph_dispatches_kv_producers_and_streamed_attention", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        if (!t.assert_true(bool(model))) return;
        const auto prefill_grants = model->granted_bytes();
        f.host = model->host();
        size_t active = 0;
        for (uint32_t rows : {256u,256u,1u}) {
            const size_t first = active; active += rows;
            ggml_context_ptr ctx(ggml_init({1024*1024,nullptr,true}));
            auto * source = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,512,rows);
            auto * indices = ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I64,rows);
            ggml_backend_buffer_ptr inputs(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
            std::vector<float> data(rows*512); std::vector<int64_t> ids(rows);
            for (size_t i = 0; i < data.size(); ++i) data[i] = .25f*std::sin(float((first*512+i)%677)*.07f);
            const auto expected_k = reference_bytes(backend.get(),data,GGML_TYPE_Q8_0,2);
            const auto expected_v = reference_bytes(backend.get(),data,GGML_TYPE_Q4_0,3);
            for (size_t i = 0; i < ids.size(); ++i) ids[i] = int64_t(first+i);
            ggml_backend_tensor_set(source,data.data(),0,data.size()*sizeof(float));
            ggml_backend_tensor_set(indices,ids.data(),0,ids.size()*sizeof(int64_t));
            auto * k = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q8_0,512,f.host->layout().tokens);
            auto * v = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q4_0,512,f.host->layout().tokens);
            llama_kv_stream_host_layer planes; f.host->layer(0,planes);
            t.assert_true(ggml_backend_tensor_alloc(model->buffer(),k,planes.k) == GGML_STATUS_SUCCESS);
            t.assert_true(ggml_backend_tensor_alloc(model->buffer(),v,planes.v) == GGML_STATUS_SUCCESS);
            auto * graph = ggml_new_graph_custom(ctx.get(),128,false);
            auto * k_source = ggml_scale(ctx.get(),source,2);
            auto * v_source = ggml_scale(ctx.get(),source,3);
            ggml_build_forward_expand(graph,k_source); ggml_build_forward_expand(graph,v_source);
            auto * k_write = ggml_set_rows(ctx.get(),k,k_source,indices);
            auto * v_write = ggml_set_rows(ctx.get(),v,v_source,indices); v_write->src[3] = k_source;
            ggml_build_forward_expand(graph,k_write); ggml_build_forward_expand(graph,v_write);
            block_inputs attn(f,active,rows);
            auto * key = ggml_view_3d(ctx.get(),k,256,attn.padded,2,f.host->layout().k_token_bytes,f.host->layout().k_row_bytes,0);
            auto * value = ggml_view_3d(ctx.get(),v,256,attn.padded,2,f.host->layout().v_token_bytes,f.host->layout().v_row_bytes,0);
            auto * output = ggml_flash_attn_ext(ctx.get(),attn.q,key,value,attn.mask,1.0f/16,0,0);
            ggml_flash_attn_ext_set_prec(output,GGML_PREC_F32);
            ggml_build_forward_expand(graph,output);
            ggml_backend_t backends[]{backend.get(),cpu.get()};
            ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends,nullptr,2,128,false,true));
            // Validate before allocation can abort on an unsupported preallocated destination.
            if (!t.assert_true(ggml_backend_supports_op(backend.get(),k_write) && ggml_backend_supports_op(backend.get(),v_write))) return;
            if (!t.assert_true(ggml_backend_sched_alloc_graph(sched.get(),graph))) return;
            if (!t.assert_true(model->begin(active,rows,rows == 1))) return;
            if (rows == 1) t.assert_true(model->granted_bytes() < prefill_grants);
            if (!t.assert_true(ggml_backend_sched_graph_compute(sched.get(),graph) == GGML_STATUS_SUCCESS)) return;
            t.assert_true(model->complete()); t.assert_equal(active,model->tokens());
            t.assert_true(!std::memcmp(static_cast<const char *>(planes.k)+first*f.host->layout().k_token_bytes,expected_k.data(),expected_k.size()));
            t.assert_true(!std::memcmp(static_cast<const char *>(planes.v)+first*f.host->layout().v_token_bytes,expected_v.data(),expected_v.size()));
            std::vector<float> actual(ggml_nelements(output)); ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
            close_values(t,oracle(f,0,active,rows,attn.qdata),actual,1e-3f);
        }
        auto * type=llama_kv_stream_device_buffer_type(dev);
        const auto allocate=type->iface.alloc_buffer;
        type->iface.alloc_buffer=[](ggml_backend_buffer_type_t,size_t)->ggml_backend_buffer_t { return nullptr; };
        const bool truncated=model->truncate(256);
        type->iface.alloc_buffer=allocate;
        t.assert_true(!truncated && !model->complete());
        t.assert_true(model->restore(256));
        t.assert_equal(size_t(256),model->tokens());
    });
    return t.summary();
}
