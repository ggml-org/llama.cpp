#include "kv-stream-block-test.h"

// Ordinary SET_ROWS supplies independent reference bytes for each producer plane.
struct producer_inputs {
    ggml_backend_t backend;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * k = nullptr, * v = nullptr;
    ggml_tensor * encoded[2] = {};
    ggml_cgraph * graph = nullptr;
    std::unique_ptr<llama_memory_cuda_executor> execution;
    producer_inputs(fixture & f, size_t rows) : backend(f.backend) {
        context.reset(ggml_init({65536,nullptr,true}));
        k = ggml_new_tensor_2d(context.get(),GGML_TYPE_F32,512,int64_t(rows));
        v = ggml_new_tensor_2d(context.get(),GGML_TYPE_F32,512,int64_t(rows));
        auto * indices = ggml_new_tensor_1d(context.get(),GGML_TYPE_I64,int64_t(rows));
        graph = ggml_new_graph_custom(context.get(),64,false);
        for (int side = 0; side < 2; ++side) {
            auto * dst = ggml_new_tensor_2d(context.get(),ggml_type(side ? f.policy.shape.type_v : f.policy.shape.type_k),512,int64_t(rows));
            encoded[side] = ggml_set_rows(context.get(),dst,side ? v : k,indices);
            ggml_build_forward_expand(graph,encoded[side]);
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(),backend)); GGML_ASSERT(buffer);
        std::vector<int64_t> idx(rows); for (size_t i = 0; i < rows; ++i) idx[i] = int64_t(i);
        ggml_backend_tensor_set(indices,idx.data(),0,rows*sizeof(int64_t));
        std::vector<float> data(rows*512);
        for (int side = 0; side < 2; ++side) {
            for (size_t i = 0; i < data.size(); ++i) data[i] = std::sin(float(i%677 + 17*side)*.05f);
            ggml_backend_tensor_set(side ? v : k,data.data(),0,data.size()*sizeof(float));
        }
        if (f.cuda) {
            execution = std::make_unique<llama_memory_cuda_executor>(backend);
            GGML_ASSERT(execution->bind(graph,{},1));
            GGML_ASSERT(execution->compute_async({},1) == GGML_STATUS_SUCCESS && execution->drain());
        } else GGML_ASSERT(ggml_backend_graph_compute(backend,graph) == GGML_STATUS_SUCCESS);
    }
    std::vector<uint8_t> bytes(bool value) const {
        std::vector<uint8_t> data(ggml_nbytes(encoded[value]));
        ggml_backend_tensor_get(encoded[value],data.data(),0,data.size()); return data;
    }
};

static void check_host(testing & t, const fixture & f, uint32_t layer, size_t first, const producer_inputs & input) {
    llama_kv_stream_host_layer planes; GGML_ASSERT(f.host->layer(layer,planes));
    for (bool value : {false,true}) {
        const auto expected = input.bytes(value);
        const size_t stride = value ? f.host->layout().v_token_bytes : f.host->layout().k_token_bytes;
        const auto * actual = static_cast<const char *>(value ? planes.v : planes.k)+first*stride;
        t.assert_true(!std::memcmp(expected.data(),actual,expected.size()));
    }
}

// Throw after real DMA submission; the generated ticket must survive until the backend drains.
struct download_fault {
    ggml_backend_t backend;
    decltype(ggml_backend_i::get_tensor_async) original;
    std::function<void()> on_copy;
    int calls = 0;
    inline static download_fault * active = nullptr;
    download_fault(ggml_backend_t backend, std::function<void()> on_copy) : backend(backend),
        original(backend->iface.get_tensor_async), on_copy(std::move(on_copy)) {
        GGML_ASSERT(original && !active); active = this;
        backend->iface.get_tensor_async = [](ggml_backend_t b, const ggml_tensor * t, void * p, size_t offset, size_t bytes) {
            active->original(b,t,p,offset,bytes);
            active->on_copy();
            if (++active->calls == 2) throw std::runtime_error("after V download submission");
        };
    }
    ~download_fault() { backend->iface.get_tensor_async = original; active = nullptr; }
};

int main(int argc, char ** argv) {
    const bool cuda = argc > 1 && !std::strcmp(argv[1],"--cuda");
    ggml_backend_ptr backend;
    if (cuda) { ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1; backend.reset(ggml_backend_dev_init(dev,nullptr)); }
    else backend.reset(ggml_backend_cpu_init());
    testing t;
    t.test("external_writer_lease_is_retained_until_explicit_release", [&](testing & t) {
        fixture f(backend.get(),cuda); if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire(); block_workspace work(f,32768);
        if (!t.assert_true(f.resident->configure_writes(33,work.lease.get()))) return;
        work.lease.reset();
        t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(work.arena.get()));
        producer_inputs input(f,33);
        t.assert_true(f.resident->write_rows(0,ggml_kv_stream_operand::k,0,input.k));
        t.assert_true(f.resident->write_rows(0,ggml_kv_stream_operand::v,0,input.v));
        check_host(t,f,0,0,input);
        f.resident->release_write_workspace();
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(work.arena.get()));
    });
    t.test("kv_region_cannot_be_reused_as_external_writer_workspace", [&](testing & t) {
        fixture f(backend.get(),cuda); if (!t.assert_true(f.attach())) return;
        t.assert_true(!f.resident->configure_writes(1,f.lease.get()));
    });
    t.test("writer_workspace_bounds_and_source_aliases_are_rejected", [&](testing & t) {
        fixture f(backend.get(),cuda); if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire(); block_workspace tiny(f,8), work(f,32768);
        t.assert_true(!f.resident->configure_writes(1,tiny.lease.get()));
        if (!t.assert_true(f.resident->configure_writes(1,work.lease.get()))) return;
        producer_inputs input(f,1);
        const auto generation = f.content->generation();
        auto alias = *input.k;
        alias.buffer = ggml_backend_memory_lease_buffer(work.lease.get());
        alias.data = ggml_backend_buffer_get_base(alias.buffer);
        t.assert_true(!f.resident->write_rows(0,ggml_kv_stream_operand::k,0,&alias));
        t.assert_equal(generation,f.content->generation());
        t.assert_true(f.resident->write_rows(0,ggml_kv_stream_operand::k,0,input.k));
        if (cuda) {
            fixture cpu(backend.get(),false); block_workspace host(cpu,32768);
            t.assert_true(!f.resident->configure_writes(1,host.lease.get()));
        }
    });
    if (cuda) t.test("producer_kv_publication_preserves_inflight_history_and_batched_attention", [&](testing & t) {
        for (auto pair : {std::pair{GGML_TYPE_F16,GGML_TYPE_F16},std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_Q4_0,GGML_TYPE_Q8_0}})
        for (size_t rows : {size_t(1),size_t(33),size_t(257)}) {
            const size_t active = 512+rows;
            fixture f(backend.get(),true,pair.first,pair.second,active,false,4);
            ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
            f.policy.pool_bytes = page.bytes*10; f.policy.initial_ring_slots = 6;
            if (!t.assert_true(f.attach())) return;
            auto pin = f.binding->acquire(); block_workspace work(f,32768);
            producer_inputs input(f,rows); block_inputs attention(f,active,rows);
            ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4*rows,256,layout);
            block_workspace partial(f,layout.bytes);
            if (!t.assert_true(f.resident->configure_writes(rows,work.lease.get()))) return;
            if (!t.assert_true(f.resident->begin_sequence({0,1,2,3},active,2,512,{uint32_t(rows),rows == 1}))) return;
            t.assert_true(f.resident->sequence_stats().copy_calls > 0);
            work.lease.reset();
            for (uint32_t layer = 0; layer < 4; ++layer) {
                const auto generation = f.content->generation();
                const auto copies = f.resident->sequence_stats().copy_calls;
                if (!t.assert_true(f.resident->write_sequence_rows(layer,512,input.k,input.v))) return;
                t.assert_equal(copies,f.resident->sequence_stats().copy_calls);
                t.assert_equal(generation+1,f.content->generation());
                check_host(t,f,layer,512,input);
                const auto stats = f.resident->last_write_stats();
                t.assert_equal(size_t(0),stats.d2d_bytes);
                t.assert_equal(input.bytes(false).size()+input.bytes(true).size(),stats.d2h_bytes);
                if (!t.assert_true(f.resident->compute_streamed(layer,attention.q,attention.mask,attention.output,active,1.0f/16,partial.lease.get(),true,2))) return;
                close_values(t,oracle(f,layer,active,rows,attention.qdata),attention.read(),1e-3f);
            }
            t.assert_true(!f.resident->sequence_active());
            t.assert_equal(4*(active-256)*(f.host->layout().k_token_bytes+f.host->layout().v_token_bytes),f.resident->sequence_stats().copy_bytes);
            f.resident->release_write_workspace();
            t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(work.arena.get()));
        }
    });
    if (cuda) t.test("invalid_producers_preserve_sequence_and_content", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,4);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire(); block_workspace work(f,32768); producer_inputs input(f,1);
        t.assert_true(f.resident->configure_writes(1));
        t.assert_true(f.resident->begin_sequence({0,1,2,3},513,2,512));
        t.assert_true(!f.resident->write_sequence_rows(0,512,input.k,input.v));
        f.resident->cancel_sequence();
        t.assert_true(f.resident->configure_writes(1,work.lease.get()));
        t.assert_true(!f.resident->write_sequence_rows(0,512,input.k,input.v));
        t.assert_true(f.resident->begin_sequence({0,1,2,3},513,2,512));
        const auto generation = f.content->generation();
        t.assert_true(!f.resident->write_sequence_rows(0,511,input.k,input.v));
        t.assert_true(!f.resident->write_sequence_rows(4,512,input.k,input.v));
        t.assert_true(!f.resident->write_sequence_rows(0,512,input.k,nullptr));
        auto invalid = *input.v; invalid.ne[1] = 0;
        t.assert_true(!f.resident->write_sequence_rows(0,512,input.k,&invalid));
        t.assert_true(!f.resident->configure_writes(1,work.lease.get()));
        f.resident->release_write_workspace();
        t.assert_equal(generation,f.content->generation());
        t.assert_true(f.resident->sequence_active());
        t.assert_true(f.resident->write_sequence_rows(0,512,input.k,input.v));
        f.resident->cancel_sequence();
    });
    if (cuda) t.test("failed_second_plane_download_is_atomic_and_recoverable", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,4);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire(); block_workspace work(f,32768); producer_inputs input(f,1);
        if (!t.assert_true(f.resident->configure_writes(1,work.lease.get()))) return;
        if (!t.assert_true(f.resident->begin_sequence({0,1,2,3},513,2,512))) return;
        llama_kv_stream_host_layer planes; f.host->layer(0,planes);
        std::vector<uint8_t> saved_k(f.host->layout().k_bytes), saved_v(f.host->layout().v_bytes);
        std::memcpy(saved_k.data(),planes.k,saved_k.size()); std::memcpy(saved_v.data(),planes.v,saved_v.size());
        const auto generation = f.content->generation();
        bool caught = false;
        {
            download_fault fault(backend.get(),[&] {
                t.assert_true(!f.resident->configure_writes(1,work.lease.get()));
                t.assert_true(!f.resident->write_sequence_rows(0,512,input.k,input.v));
                t.assert_true(!f.resident->publish_sequence_tail({}));
                f.resident->cancel_sequence();
                t.assert_true(f.resident->sequence_active());
            });
            try { (void) f.resident->write_sequence_rows(0,512,input.k,input.v); }
            catch (const std::runtime_error &) { caught = true; }
            t.assert_equal(2,fault.calls);
        }
        t.assert_true(caught && !f.resident->sequence_active());
        t.assert_equal(size_t(0),f.resident->sequence_stats().pending_pages);
        t.assert_equal(generation,f.content->generation());
        t.assert_true(!std::memcmp(saved_k.data(),planes.k,saved_k.size()));
        t.assert_true(!std::memcmp(saved_v.data(),planes.v,saved_v.size()));
        t.assert_true(f.resident->begin_sequence({0,1,2,3},513,2,512));
        t.assert_true(f.resident->write_sequence_rows(0,512,input.k,input.v));
        check_host(t,f,0,512,input); f.resident->cancel_sequence();
    });
    if (cuda) t.test("producer_tail_crosses_resident_boundary_without_losing_previous_rows", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q5_1,GGML_TYPE_Q4_1,513);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        f.policy.pool_bytes = page.bytes*8; f.policy.initial_ring_slots = 4;
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire(); block_workspace work(f,32768); producer_inputs input(f,2);
        block_inputs attention(f,513,2); ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(8,256,layout);
        block_workspace partial(f,layout.bytes);
        if (!t.assert_true(f.resident->configure_writes(2,work.lease.get()))) return;
        if (!t.assert_true(f.resident->begin_sequence({0,1},513,2,511))) return;
        for (uint32_t layer = 0; layer < 2; ++layer) {
            const auto history = unpack(f,layer,false);
            t.assert_true(f.resident->write_sequence_rows(layer,511,input.k,input.v));
            check_host(t,f,layer,511,input);
            const auto updated = unpack(f,layer,false);
            t.assert_true(std::equal(history.begin(),history.begin()+511*512,updated.begin()));
            if (!t.assert_true(f.resident->compute_streamed(layer,attention.q,attention.mask,attention.output,513,1.0f/16,partial.lease.get(),true,2))) return;
            close_values(t,oracle(f,layer,513,2,attention.qdata),attention.read(),1e-3f);
            t.assert_true(!f.resident->write_sequence_rows(layer,511,input.k,input.v));
        }
    });
    if (cuda) t.test("attention_cannot_overwrite_retained_writer_indices", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,513,false,4);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire(); block_workspace work(f,32768); producer_inputs input(f,1);
        block_inputs attention(f,513,1);
        if (!t.assert_true(f.resident->configure_writes(1,work.lease.get()))) return;
        if (!t.assert_true(f.resident->begin_sequence({0,1,2,3},513,2,512))) return;
        t.assert_true(f.resident->write_sequence_rows(0,512,input.k,input.v));
        t.assert_true(!f.resident->compute_streamed(0,attention.q,attention.mask,attention.output,513,1.0f/16,work.lease.get(),true,2));
        t.assert_true(!f.resident->sequence_active());
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace partial(f,layout.bytes);
        for (bool output : {false,true}) {
            t.assert_true(f.resident->begin_sequence({0,1,2,3},513,2,512));
            t.assert_true(f.resident->write_sequence_rows(0,512,input.k,input.v));
            auto alias = *(output ? attention.output : attention.q);
            alias.buffer = ggml_backend_memory_lease_buffer(work.lease.get());
            alias.data = ggml_backend_buffer_get_base(alias.buffer);
            t.assert_true(!f.resident->compute_streamed(0,output ? attention.q : &alias,attention.mask,output ? &alias : attention.output,513,1.0f/16,partial.lease.get(),true,2));
            t.assert_true(!f.resident->sequence_active());
        }
    });
    if (cuda) t.test("pending_sequence_teardown_releases_external_workspace_after_drain", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,2049,false,4);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire(); block_workspace work(f,32768);
        t.assert_true(f.resident->configure_writes(1,work.lease.get()));
        work.lease.reset();
        t.assert_true(f.resident->begin_sequence({0,1,2,3},2049,2,2048));
        t.assert_true(f.resident->sequence_stats().copy_calls > 0);
        f.binding.reset();
        t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(work.arena.get()));
        pin.reset();
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(work.arena.get()));
    });
    return t.summary();
}
