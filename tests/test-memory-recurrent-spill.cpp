#include "../src/llama-memory-recurrent-spill.h"
#include "../ggml/src/ggml-backend-memory.h"
#include "testing.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include <cstring>
#include <array>
#include <algorithm>
#include <stdexcept>

// Fail after a real transfer, so cancellation must treat its destination as potentially modified.
struct recurrent_transfer_fault {
    inline static recurrent_transfer_fault * active = nullptr;
    ggml_backend_t backend;
    decltype(ggml_backend_i::get_tensor_async) get;
    decltype(ggml_backend_i::set_tensor_async) set;
    recurrent_transfer_fault(ggml_backend_t backend, bool upload) : backend(backend),
        get(backend->iface.get_tensor_async), set(backend->iface.set_tensor_async) {
        GGML_ASSERT(!active); active=this;
        if (upload) backend->iface.set_tensor_async=[](ggml_backend_t b,ggml_tensor * tensor,const void * data,size_t offset,size_t bytes) {
            if (active->set) active->set(b,tensor,data,offset,bytes);
            else ggml_backend_tensor_set(tensor,data,offset,bytes);
            throw std::runtime_error("after recurrent H2D submission");
        };
        else backend->iface.get_tensor_async=[](ggml_backend_t b,const ggml_tensor * tensor,void * data,size_t offset,size_t bytes) {
            if (active->get) active->get(b,tensor,data,offset,bytes);
            else ggml_backend_tensor_get(tensor,data,offset,bytes);
            throw std::runtime_error("after recurrent D2H submission");
        };
    }
    ~recurrent_transfer_fault() {
        backend->iface.get_tensor_async=get; backend->iface.set_tensor_async=set; active=nullptr;
    }
};

// Observe the bank's separately owned copy backend without adding a production test hook.
struct recurrent_copy_backend_probe {
    inline static recurrent_copy_backend_probe * active = nullptr;
    ggml_backend_dev_t device;
    decltype(ggml_backend_device_i::init_backend) original;
    ggml_backend_t copy = nullptr;
    decltype(ggml_backend_i::get_tensor_async) get = nullptr;
    decltype(ggml_backend_i::synchronize) sync = nullptr;
    struct read { const ggml_tensor * tensor; void * data; size_t offset,bytes; };
    std::vector<read> pending;
    bool deferred = false;
    bool fail_prepare = false;
    size_t preparations = 0;
    size_t downloads = 0;
    size_t fail_download = 0;
    bool observed = false;
    explicit recurrent_copy_backend_probe(ggml_backend_dev_t device) : device(device),original(device->iface.init_backend) {
        GGML_ASSERT(!active); active=this;
        device->iface.init_backend=[](ggml_backend_dev_t dev,const char * params) {
            auto * backend=active->copy=active->original(dev,params);
            if (backend) {
                active->sync=backend->iface.synchronize;
                backend->iface.synchronize=[](ggml_backend_t b) {
                    ++active->preparations;
                    if (active->fail_prepare) { active->fail_prepare=false; throw std::runtime_error("publication resource preparation failed"); }
                    if (active->sync) active->sync(b);
                };
            }
            return backend;
        };
    }
    void flush() {
        for (const auto & read : pending) ggml_backend_tensor_get(read.tensor,read.data,read.offset,read.bytes);
        pending.clear();
    }
    // CPU-only deterministic queue: stage bytes are read only when its consumer is drained.
    void defer_downloads() {
        GGML_ASSERT(copy && !deferred);
        get=copy->iface.get_tensor_async; deferred=true;
        copy->iface.get_tensor_async=[](ggml_backend_t,const ggml_tensor * tensor,void * data,size_t offset,size_t bytes) {
            ++active->downloads;
            active->pending.push_back({tensor,data,offset,bytes});
            if (active->fail_download == active->downloads) throw std::runtime_error("after packed recurrent submission");
        };
        copy->iface.synchronize=[](ggml_backend_t) { active->flush(); };
        device->iface.init_backend=original;
    }
    void observe_downloads() {
        GGML_ASSERT(copy && !deferred && !observed);
        get=copy->iface.get_tensor_async; observed=true;
        copy->iface.get_tensor_async=[](ggml_backend_t backend,const ggml_tensor * tensor,void * data,size_t offset,size_t bytes) {
            ++active->downloads;
            if (active->get) active->get(backend,tensor,data,offset,bytes);
            else ggml_backend_tensor_get(tensor,data,offset,bytes);
        };
        device->iface.init_backend=original;
    }
    ~recurrent_copy_backend_probe() {
        if (deferred) { flush(); copy->iface.get_tensor_async=get; }
        if (observed) copy->iface.get_tensor_async=get;
        if (copy) copy->iface.synchronize=sync;
        device->iface.init_backend=original; active=nullptr;
    }
};

int main(int argc, char ** argv) {
    testing t;
    t.test("contiguous_snapshot_planes_batch_without_copying_padding_or_untouched_rows", [&](testing & t) {
        const bool gpu=argc > 1 && !std::strcmp(argv[1],"--cuda");
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        for (uint32_t r_width : {31u,32u}) for (uint32_t s_width : {63u,64u}) {
            ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
            auto * host_type=ggml_backend_dev_host_buffer_type(device);
            if (!host_type) host_type=ggml_backend_cpu_buffer_type();
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(
                    std::vector<llama_recurrent_spill_layer>(3,{GGML_TYPE_F32,GGML_TYPE_F32,r_width,s_width,true}),1,7,layout))) return;
            auto bank=llama_recurrent_spill_bank::create(layout,host_type,97,0);
            recurrent_copy_backend_probe probe(device);
            if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
            if (gpu) probe.observe_downloads(); else probe.defer_downloads();
            const auto original_stage=bank->staged_device_bytes(),original_host=bank->host_bytes();
            ggml_context_ptr ctx(ggml_init({1024*1024,nullptr,true}));
            std::vector<ggml_tensor *> input(42);
            for (size_t layer=0;layer<3;++layer) for (uint32_t snapshot=1;snapshot<=7;++snapshot)
                for (bool value : {false,true}) input[(layer*7+snapshot-1)*2+size_t(value)]=
                    ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,value ? s_width : r_width);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
            if (!t.assert_true(bool(buffer))) return;
            std::vector<uint64_t> generations(42);
            std::vector<std::vector<float>> expected(42);
            for (uint32_t width : {1u,2u,4u,8u}) {
                auto * graph=ggml_new_graph_custom(ctx.get(),256,false);
                for (size_t layer=0;layer<3;++layer) for (bool value : {false,true}) {
                    const uint32_t snapshots=value ? width-1 : 7;
                    for (uint32_t j=0;j<snapshots;++j) {
                        const uint32_t snapshot=snapshots-j;
                        const size_t index=(layer*7+snapshot-1)*2+size_t(value);
                        expected[index].assign(value ? s_width : r_width,float(1000*width+100*layer+10*snapshot+value));
                        generations[index]=width;
                        ggml_backend_tensor_set(input[index],expected[index].data(),0,ggml_nbytes(input[index]));
                        auto * staged=ggml_cpy(ctx.get(),input[index],bank->staging_tensor(layer,value,snapshot));
                        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),staged,bank->publication_tensor(layer,value,snapshot)));
                    }
                }
                const auto before=probe.downloads;
                if (!t.assert_true(bank->begin_staged_capture({97,0,width},width) &&
                        ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS &&
                        bank->publish_capture() && bank->complete_capture())) return;
                const size_t r_calls=r_width == 32 ? 1 : 7;
                const size_t s_calls=width == 1 ? 0 : s_width == 64 ? 1 : width-1;
                t.assert_equal(3*(r_calls+s_calls),probe.downloads-before);
                t.assert_equal(original_stage,bank->staged_device_bytes());
                t.assert_equal(original_host,bank->host_bytes());
                for (size_t layer=0;layer<3;++layer) for (uint32_t snapshot=1;snapshot<=7;++snapshot)
                    for (bool value : {false,true}) {
                        const size_t index=(layer*7+snapshot-1)*2+size_t(value);
                        t.assert_equal(generations[index],bank->snapshot_generation(layer,value,snapshot));
                        const auto * bytes=bank->snapshot_data(layer,value,snapshot,generations[index]);
                        if (generations[index]) t.assert_true(bytes && !std::memcmp(bytes,expected[index].data(),ggml_nbytes(input[index])));
                        else t.assert_true(!bytes && bank->snapshot_pristine(layer,value,snapshot));
                    }
            }
        }
    });
    t.test("packed_transfer_failure_invalidates_every_touched_snapshot_and_retries", [](testing & t) {
        for (size_t failure : {1u,2u}) {
            ggml_backend_ptr producer(ggml_backend_cpu_init());
            auto * device=ggml_backend_get_device(producer.get());
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(
                    {{GGML_TYPE_F32,GGML_TYPE_F32,32,64,true}},1,3,layout))) return;
            auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),99,0);
            recurrent_copy_backend_probe probe(device);
            if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_cpu_buffer_type(),2))) return;
            probe.defer_downloads();
            ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
            auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,32);
            auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,64);
            ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
            if (!t.assert_true(bool(source))) return;
            ggml_backend_tensor_memset(r,0x11,0,ggml_nbytes(r));
            ggml_backend_tensor_memset(s,0x22,0,ggml_nbytes(s));
            auto * graph=ggml_new_graph_custom(ctx.get(),64,false);
            for (bool value : {false,true}) for (uint32_t snapshot=1;snapshot<=3;++snapshot) {
                auto * copy=ggml_cpy(ctx.get(),value ? s : r,bank->staging_tensor(0,value,snapshot));
                ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),copy,bank->publication_tensor(0,value,snapshot)));
            }
            probe.fail_download=failure;
            if (!t.assert_true(bank->begin_staged_capture({99,0,1},4))) return;
            t.assert_true(ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_FAILED);
            t.assert_true(!bank->publish_capture());
            bank->cancel();
            t.assert_true(probe.pending.empty());
            for (uint32_t snapshot=1;snapshot<=3;++snapshot) {
                t.assert_true(!bank->snapshot_data(0,false,snapshot,1) && !bank->snapshot_pristine(0,false,snapshot));
                t.assert_equal(failure == 1,bank->snapshot_pristine(0,true,snapshot));
            }
            probe.fail_download=0;
            t.assert_true(bank->begin_staged_capture({99,0,2},4) &&
                ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->complete_capture());
            t.assert_true(bank->snapshot_data(0,true,3,2) != nullptr);
        }
    });
    t.test("publication_resources_are_prepared_transactionally_before_admission", [](testing & t) {
        ggml_backend_ptr producer(ggml_backend_cpu_init());
        auto * device=ggml_backend_get_device(producer.get());
        for (bool fail : {false,true}) {
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(
                    {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,7,layout))) return;
            auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),89,0);
            recurrent_copy_backend_probe probe(device);
            probe.fail_prepare=fail;
            const bool enabled=bank->enable_publication(device,ggml_backend_cpu_buffer_type(),2);
            t.assert_equal(!fail,enabled);
            if (fail && enabled) return;
            if (fail) {
                // Failed admission frees its candidate backend; the bank must permit a clean retry.
                probe.copy=nullptr;
                t.assert_true(!bank->staged_publication_enabled());
                t.assert_true(bank->enable_publication(device,ggml_backend_cpu_buffer_type(),2));
            }
            t.assert_true(probe.preparations > 0);
            t.assert_true(bank->begin_staged_capture({89,0,1},8));
            bank->cancel();
        }
    });
    t.test("streamed_rollback_plan_follows_verification_not_auxiliary_kv", [](testing & t) {
        for (uint32_t depth=0;depth<=7;++depth) {
            llama_recurrent_rollback_plan plan;
            if (!t.assert_true(llama_recurrent_rollback_plan_make(depth,false,256,plan))) return;
            t.assert_equal(depth,plan.depth);
            t.assert_equal(depth+1,plan.verification_rows);
            t.assert_equal(std::max(4u,depth+1),plan.workspace_rows);
            t.assert_equal(depth != 0,plan.host_spill);
            t.assert_equal(depth ? 2u : 0u,plan.stage_slots);
            for (bool mtp : {false,true}) {
                auto saved=plan;
                const bool valid=llama_recurrent_rollback_plan_make(depth,mtp,depth+1,plan);
                t.assert_equal(!mtp || depth <= 3,valid);
                if (!valid) {
                    t.assert_equal(saved.depth,plan.depth);
                    t.assert_equal(saved.verification_rows,plan.verification_rows);
                }
            }
        }
        llama_recurrent_rollback_plan sentinel;
        sentinel.depth=123; sentinel.verification_rows=456; sentinel.workspace_rows=789;
        for (auto bad : {std::pair{8u,256u},std::pair{UINT32_MAX,256u},std::pair{7u,7u},std::pair{0u,0u}}) {
            t.assert_true(!llama_recurrent_rollback_plan_make(bad.first,false,bad.second,sentinel));
            t.assert_equal(123u,sentinel.depth);
            t.assert_equal(456u,sentinel.verification_rows);
            t.assert_equal(789u,sentinel.workspace_rows);
        }
    });
    t.test("publication_stage_can_borrow_exact_persistent_storage_but_not_during_capture", [](testing & t) {
        auto * type=ggml_backend_cpu_buffer_type();
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        auto * device=ggml_backend_get_device(backend.get());
        llama_recurrent_spill_layout snapshots;
        if (!t.assert_true(llama_recurrent_spill_layout_make({{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,3,snapshots))) return;
        auto bank=llama_recurrent_spill_bank::create(snapshots,type,77,0);
        if (!t.assert_true(bank && bank->enable_publication(device,type,2))) return;
        const size_t bytes=bank->staged_device_bytes(), a=ggml_backend_buft_get_alignment(type);
        auto * arena=ggml_backend_memory_arena_new(type,bytes+2*a);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena,0) &&
                ggml_backend_memory_arena_reserve_at(arena,9,a,bytes,a,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_commit(arena))) return;
        auto * lease=ggml_backend_memory_arena_acquire(arena,9);
        if (!t.assert_true(lease && bank->bind_publication_stage(lease))) return;
        const auto base=uintptr_t(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(lease)));
        auto * tensor=bank->staging_tensor(0,true,3);
        t.assert_true(uintptr_t(tensor->data) >= base && uintptr_t(tensor->data)+ggml_nbytes(tensor) <= base+bytes);
        for (bool persistent : {false,true}) {
            const size_t size=persistent ? bytes-1 : bytes;
            std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> bad(
                ggml_backend_memory_arena_new(type,bytes+2*a),ggml_backend_memory_arena_free);
            if (!t.assert_true(bad && ggml_backend_memory_arena_begin(bad.get(),0) &&
                    ggml_backend_memory_arena_reserve_at(bad.get(),9,a,size,a,persistent ? GGML_BACKEND_MEMORY_REGION_PERSISTENT : 0,nullptr) &&
                    ggml_backend_memory_arena_commit(bad.get()))) return;
            auto * short_grant=ggml_backend_memory_arena_acquire(bad.get(),9);
            t.assert_true(!bank->bind_publication_stage(short_grant) && bank->staging_tensor(0,true,3) == tensor);
            ggml_backend_memory_lease_free(short_grant);
        }
        t.assert_true(bank->begin_staged_capture({77,0,1},4));
        t.assert_true(!bank->bind_publication_stage(lease));
        bank->cancel();
        t.assert_true(bank->staged_publication_enabled());
        ggml_backend_memory_lease_free(lease); ggml_backend_memory_arena_free(arena);
        t.assert_true(bank->staging_tensor(0,true,3) == tensor);
    });
    t.test("queued_recurrent_transfer_failure_invalidates_or_poison_restores", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,7,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),81,0);
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bank && buffer && bank->begin_capture({81,0,1}))) return;
        ggml_backend_tensor_memset(r,0x11,0,ggml_nbytes(r));
        ggml_backend_tensor_memset(s,0x22,0,ggml_nbytes(s));
        for (uint32_t slot=1;slot<=7;++slot)
            if (!t.assert_true(bank->capture(backend.get(),0,false,slot,r,0) &&
                    bank->capture(backend.get(),0,true,slot,s,0))) return;
        if (!t.assert_true(bank->publish_capture() && bank->complete_capture() && bank->begin_capture({81,0,2}))) return;
        ggml_backend_tensor_memset(r,0x33,0,ggml_nbytes(r));
        {
            recurrent_transfer_fault fault(backend.get(),false);
            t.assert_true(!bank->capture(backend.get(),0,false,1,r,0));
        }
        t.assert_true(!bank->publish_capture());
        bank->cancel();
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1) && !bank->begin_restore({81,0,1},1));
        t.assert_equal(uint64_t(1),bank->snapshot_generation(0,false,2));
        if (!t.assert_true(bank->begin_restore({81,0,1},2))) return;
        {
            recurrent_transfer_fault fault(backend.get(),true);
            t.assert_true(!bank->restore(backend.get(),0,false,r,0));
        }
        bank->cancel();
        t.assert_true(!bank->begin_capture({81,0,2}) && !bank->begin_restore({81,0,1},2));
        bank->reset(true);
        if (!t.assert_true(bank->begin_capture({81,0,2}) &&
                bank->capture(backend.get(),0,false,2,r,0) && bank->capture(backend.get(),0,true,2,s,0) &&
                bank->publish_capture() && bank->complete_capture() && bank->begin_restore({81,0,2},2) &&
                bank->restore(backend.get(),0,false,r,0))) return;
        {
            recurrent_transfer_fault fault(backend.get(),true);
            t.assert_true(!bank->restore(backend.get(),0,true,s,0));
        }
        t.assert_true(!bank->publish_restore());
        bank->cancel();
        t.assert_true(!bank->begin_capture({81,0,3}));
    });
    t.test("layout_matches_stock_recurrent_rows_and_counts_exact_bytes", [](testing & t) {
        const std::vector<llama_recurrent_spill_layer> layers{
            {GGML_TYPE_F32,GGML_TYPE_F32,24576,786432,true},
            {GGML_TYPE_F32,GGML_TYPE_F32,24576,786432,true},
            {GGML_TYPE_F32,GGML_TYPE_F32,24576,786432,false},
        };
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,3,layout))) return;
        t.assert_equal(size_t(98304),layout.layers[0].r_row_bytes);
        t.assert_equal(size_t(3145728),layout.layers[0].s_row_bytes);
        t.assert_equal(size_t(6488064),layout.device_current_bytes);
        t.assert_equal(size_t(19464192),layout.host_snapshot_bytes);
        t.assert_equal(size_t(25952256),layout.stock_device_bytes);
        t.assert_equal(layout.device_current_bytes+layout.host_snapshot_bytes,layout.stock_device_bytes);
        t.assert_equal(size_t(3244032),layout.device_stage_bytes);
        t.assert_true(!layout.layers[2].enabled);
        t.assert_equal(layout.layers[0].s_offset+3*layout.layers[0].s_row_bytes,layout.layers[1].r_offset);
        ggml_context * ctx=ggml_init({32768,nullptr,true});
        auto * r=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,24576,4);
        auto * s=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,786432,4);
        t.assert_equal(2*(ggml_nbytes(r)+ggml_nbytes(s)),layout.stock_device_bytes);
        ggml_free(ctx);
    });
    t.test("depth_seven_two_slot_footprint_is_bounded_by_one_layer", [](testing & t) {
        std::vector<llama_recurrent_spill_layer> layers(48,{GGML_TYPE_F32,GGML_TYPE_F32,30720,786432,true});
        llama_recurrent_spill_layout snapshots;
        llama_recurrent_stage_layout stage;
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,7,snapshots) &&
                llama_recurrent_stage_layout_make(snapshots,2,256,stage))) return;
        t.assert_equal(size_t(149.625*1048576),snapshots.device_current_bytes);
        t.assert_equal(size_t(1047.375*1048576),snapshots.host_snapshot_bytes);
        t.assert_equal(size_t(43.640625*1048576),stage.total_bytes);
        for (uint32_t slot=0;slot<2;++slot) for (uint32_t snapshot=1;snapshot<=7;++snapshot)
            for (bool value : {false,true}) {
                size_t offset=0,bytes=0;
                t.assert_true(llama_recurrent_stage_region(stage,47,value,snapshot,slot,offset,bytes));
                t.assert_true(offset <= stage.total_bytes && bytes <= stage.total_bytes-offset);
            }
    });
    t.test("wide_publication_reuses_two_slots_and_restores_every_candidate", [&](testing & t) {
        ggml_backend_load_all();
        const bool gpu=argc > 1 && !std::strcmp(argv[1],"--cuda");
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
        auto * host_type=ggml_backend_dev_host_buffer_type(device);
        if (!host_type) host_type=ggml_backend_cpu_buffer_type();
        std::vector<llama_recurrent_spill_layer> layers(4,{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true});
        layers[1].enabled=false;
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,2,7,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host_type,83,0);
        recurrent_copy_backend_probe probe(device);
        if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
        if (!gpu) probe.defer_downloads();
        const size_t original_bytes=bank->host_bytes(),original_stage=bank->staged_device_bytes();
        ggml_context_ptr ctx(ggml_init({1024*1024,nullptr,true}));
        std::vector<ggml_tensor *> inputs(4*7*2,nullptr);
        std::vector<std::vector<float>> expected(inputs.size());
        std::vector<uint64_t> generations(inputs.size(),0);
        const auto index=[](size_t layer,uint32_t snapshot,bool value) { return (layer*7+snapshot-1)*2+size_t(value); };
        for (size_t layer : {0u,2u,3u}) for (uint32_t snapshot=1;snapshot<=7;++snapshot)
            for (bool value : {false,true}) {
                const size_t i=index(layer,snapshot,value);
                inputs[i]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,value ? 8 : 4,2);
                expected[i].resize(value ? 16 : 8,0.0f);
            }
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(source))) return;
        for (uint32_t width=1;width<=8;++width) {
            auto * graph=ggml_new_graph_custom(ctx.get(),256,false);
            for (size_t layer : {0u,2u,3u}) {
                for (bool value : {false,true}) for (uint32_t j=1;j<= (value ? width-1 : 7);++j) {
                    const uint32_t snapshot=value ? j : 8-j;
                    const size_t i=index(layer,snapshot,value);
                    for (size_t k=0;k<expected[i].size();++k) expected[i][k]=float(10000*width+1000*layer+10*snapshot+value+k);
                    ggml_backend_tensor_set(inputs[i],expected[i].data(),0,ggml_nbytes(inputs[i]));
                    auto * staged=ggml_cpy(ctx.get(),inputs[i],bank->staging_tensor(layer,value,snapshot));
                    ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),staged,bank->publication_tensor(layer,value,snapshot)));
                }
            }
            if (!t.assert_true(bank->begin_staged_capture({83,0,width},width) &&
                    ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS && bank->publish_capture())) return;
            t.assert_equal(uint64_t(width-1),bank->snapshot_generation(0,false,1));
            t.assert_true(!bank->begin_restore({83,0,width},1) && !bank->snapshot_data(0,false,1,width));
            if (!gpu) t.assert_true(!probe.pending.empty());
            if (!t.assert_true(bank->stage_ready())) return;
            if (!gpu) t.assert_true(!probe.pending.empty());
            if (!t.assert_true(bank->complete_capture())) return;
            if (!gpu) t.assert_true(probe.pending.empty());
            t.assert_true(!bank->begin_staged_capture({83,0,width},width));
            for (size_t layer : {0u,2u,3u}) for (uint32_t snapshot=1;snapshot<=7;++snapshot)
                for (bool value : {false,true}) {
                    const size_t i=index(layer,snapshot,value);
                    if (!value || snapshot < width) generations[i]=width;
                    t.assert_equal(generations[i],bank->snapshot_generation(layer,value,snapshot));
                    const auto * actual=bank->snapshot_data(layer,value,snapshot,generations[i]);
                    if (generations[i]) t.assert_true(actual && !std::memcmp(actual,expected[i].data(),ggml_nbytes(inputs[i])));
                    else t.assert_true(!actual && bank->snapshot_pristine(layer,value,snapshot));
                }
            if (width > 1) t.assert_true(!bank->begin_restore({83,0,width-1},1));
            for (uint32_t selected=1;selected<=7;++selected) {
                if (!t.assert_true(bank->begin_restore({83,0,width},selected))) return;
                for (size_t layer : {0u,2u,3u}) for (bool value : {false,true})
                    if (!t.assert_true(bank->restore(producer.get(),layer,value,inputs[index(layer,1,value)],0))) return;
                if (!t.assert_true(bank->publish_restore() && bank->complete_restore())) return;
                for (size_t layer : {0u,2u,3u}) for (bool value : {false,true}) {
                    const size_t i=index(layer,selected,value);
                    std::vector<float> actual(expected[i].size());
                    ggml_backend_tensor_get(inputs[index(layer,1,value)],actual.data(),0,actual.size()*sizeof(float));
                    t.assert_true(actual == expected[i]);
                }
            }
            t.assert_equal(original_bytes,bank->host_bytes());
            t.assert_equal(original_stage,bank->staged_device_bytes());
        }
    });
    t.test("failed_staged_copy_cannot_republish_old_host_bytes", [](testing & t) {
        ggml_backend_ptr producer(ggml_backend_cpu_init());
        auto * device=ggml_backend_get_device(producer.get());
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,7,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),85,0);
        recurrent_copy_backend_probe probe(device);
        if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_cpu_buffer_type(),2) && probe.copy)) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * source=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(buffer))) return;
        ggml_backend_tensor_memset(source,0x11,0,ggml_nbytes(source));
        auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
        auto * stage=ggml_cpy(ctx.get(),source,bank->staging_tensor(0,false,7));
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),stage,bank->publication_tensor(0,false,7)));
        if (!t.assert_true(bank->begin_staged_capture({85,0,1},8))) return;
        {
            recurrent_transfer_fault fault(probe.copy,false);
            t.assert_equal(GGML_STATUS_FAILED,ggml_backend_graph_compute(producer.get(),graph));
        }
        t.assert_true(!bank->publish_capture());
        bank->cancel();
        t.assert_true(!bank->snapshot_pristine(0,false,7) && !bank->begin_restore({85,0,1},7));
        t.assert_true(bank->snapshot_pristine(0,false,6));
        bank->reset(true);
        t.assert_true(bank->snapshot_pristine(0,false,7));
    });
    t.test("two_slot_stage_is_bounded_by_one_layer_not_all_recurrent_layers", [](testing & t) {
        std::vector<llama_recurrent_spill_layer> layers(
            48,{GGML_TYPE_F32,GGML_TYPE_F32,30720,786432,true});
        llama_recurrent_spill_layout snapshots;
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,3,snapshots))) return;
        t.assert_equal(size_t(149.625*1024*1024),snapshots.device_current_bytes);
        t.assert_equal(size_t(448.875*1024*1024),snapshots.host_snapshot_bytes);
        llama_recurrent_stage_layout stage;
        if (!t.assert_true(llama_recurrent_stage_layout_make(snapshots,2,256,stage))) return;
        t.assert_equal(size_t(9805824),stage.bytes_per_slot);
        t.assert_equal(size_t(19611648),stage.total_bytes);
        size_t offset=0,bytes=0;
        t.assert_true(llama_recurrent_stage_region(stage,0,false,1,0,offset,bytes));
        t.assert_equal(size_t(0),offset);
        t.assert_equal(size_t(122880),bytes);
        t.assert_true(llama_recurrent_stage_region(stage,47,true,3,1,offset,bytes));
        t.assert_equal(size_t(9805824+368640+2*3145728),offset);
        t.assert_equal(size_t(3145728),bytes);
        auto inconsistent=stage;
        inconsistent.bytes_per_slot=1;
        offset=123; bytes=456;
        t.assert_true(!llama_recurrent_stage_region(inconsistent,0,false,1,0,offset,bytes));
        t.assert_equal(size_t(123),offset);
        t.assert_equal(size_t(456),bytes);
        t.assert_true(!llama_recurrent_stage_region(stage,48,true,1,0,offset,bytes));
        t.assert_true(!llama_recurrent_stage_region(stage,0,true,4,0,offset,bytes));
        t.assert_true(!llama_recurrent_stage_region(stage,0,true,1,2,offset,bytes));
        const auto original=stage.total_bytes;
        t.assert_true(!llama_recurrent_stage_layout_make(snapshots,2,3,stage));
        t.assert_equal(original,stage.total_bytes);
        t.assert_true(!llama_recurrent_stage_layout_make(
            snapshots,3,size_t(1) << (sizeof(size_t)*8-1),stage));
        t.assert_equal(original,stage.total_bytes);
        auto malformed=snapshots;
        malformed.layers[0].r_row_bytes=0;
        t.assert_true(!llama_recurrent_stage_layout_make(malformed,2,256,stage));
        t.assert_equal(original,stage.total_bytes);
    });

    t.test("typed_stage_tensors_alias_only_their_planned_ring_regions", [](testing & t) {
        llama_recurrent_spill_layout snapshots;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,16,32,true}},1,3,snapshots))) return;
        auto stage=llama_recurrent_stage_buffer::create(snapshots,2,ggml_backend_cpu_buffer_type());
        if (!t.assert_true(bool(stage))) return;
        t.assert_equal(stage->layout().total_bytes,stage->bytes());
        auto * base=static_cast<uint8_t *>(ggml_backend_buffer_get_base(stage->buffer()));
        size_t offset=0,bytes=0;
        if (!t.assert_true(llama_recurrent_stage_region(
                stage->layout(),0,false,2,1,offset,bytes))) return;
        auto * r=stage->tensor(0,false,2,1);
        t.assert_true(r && r->buffer == stage->buffer() && r->data == base+offset);
        t.assert_equal(bytes,ggml_nbytes(r));
        if (!t.assert_true(llama_recurrent_stage_region(
                stage->layout(),0,true,3,1,offset,bytes))) return;
        auto * s=stage->tensor(0,true,3,1);
        t.assert_true(s && s->buffer == stage->buffer() && s->data == base+offset);
        t.assert_equal(bytes,ggml_nbytes(s));
        t.assert_true(stage->tensor(1,false,1,0) == nullptr);
        t.assert_true(stage->tensor(0,true,4,0) == nullptr);
        t.assert_true(stage->tensor(0,true,1,2) == nullptr);
    });

    t.test("invalid_geometry_and_overflow_are_transactional", [](testing & t) {
        llama_recurrent_spill_layout layout;
        const std::vector<llama_recurrent_spill_layer> good{{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}};
        if (!t.assert_true(llama_recurrent_spill_layout_make(good,1,2,layout))) return;
        const auto bytes=layout.host_snapshot_bytes;
        t.assert_true(!llama_recurrent_spill_layout_make(good,UINT32_MAX,UINT32_MAX,layout));
        t.assert_equal(bytes,layout.host_snapshot_bytes);
        const std::vector<llama_recurrent_spill_layer> bad{{GGML_TYPE_Q8_0,GGML_TYPE_F32,1,8,true}};
        t.assert_true(!llama_recurrent_spill_layout_make(bad,1,2,layout));
        t.assert_equal(bytes,layout.host_snapshot_bytes);
    });
    t.test("host_bank_publishes_and_restores_exact_selected_state", [](testing & t) {
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,2,layout))) return;
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),7,0);
        if (!t.assert_true(bool(bank))) return;
        t.assert_equal(layout.host_snapshot_bytes,bank->host_bytes());
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        auto * dr=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * ds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(buffer))) return;
        const float expected_r[4]={1,2,3,4};
        t.assert_true(bank->snapshot_pristine(0,false,1));
        t.assert_true(bank->snapshot_pristine(0,true,2));
        const float expected_s[8]={9,8,7,6,5,4,3,2};
        ggml_backend_tensor_set(r,expected_r,0,sizeof(expected_r));
        ggml_backend_tensor_set(s,expected_s,0,sizeof(expected_s));
        if (!t.assert_true(bank->begin_capture({7,0,1}) &&
                bank->capture(backend.get(),0,false,1,r,0) &&
                bank->capture(backend.get(),0,true,1,s,0) && bank->publish_capture())) return;
        t.assert_true(!bank->begin_restore({7,0,1},1));
        if (!t.assert_true(bank->complete_capture() && bank->begin_restore({7,0,1},1))) return;
        t.assert_true(bank->restore(backend.get(),0,false,dr,0));
        t.assert_true(bank->restore(backend.get(),0,true,ds,0));
        if (!t.assert_true(bank->publish_restore() && bank->complete_restore())) return;
        float actual_r[4]={},actual_s[8]={};
        ggml_backend_tensor_get(dr,actual_r,0,sizeof(actual_r));
        ggml_backend_tensor_get(ds,actual_s,0,sizeof(actual_s));
        t.assert_true(std::memcmp(actual_r,expected_r,sizeof(expected_r)) == 0);
        t.assert_true(std::memcmp(actual_s,expected_s,sizeof(expected_s)) == 0);
        t.assert_equal(uint64_t(1),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        t.assert_true(!bank->begin_capture({7,1,2}));
        t.assert_true(bank->begin_capture({7,0,2}));
        t.assert_true(bank->capture(backend.get(),0,false,1,r,0));
        bank->cancel();
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        bank->reset(false);
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        bank->reset(true);
        t.assert_true(bank->snapshot_pristine(0,false,1));
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(bank->snapshot_pristine(0,false,2));
        if (!t.assert_true(bank->begin_restore({7,0,1},1) &&
                bank->restore(backend.get(),0,false,dr,0) &&
                bank->restore(backend.get(),0,true,ds,0) &&
                bank->publish_restore() && bank->complete_restore())) return;
        ggml_backend_tensor_get(dr,actual_r,0,sizeof(actual_r));
        ggml_backend_tensor_get(ds,actual_s,0,sizeof(actual_s));
        const float zero_r[4]={},zero_s[8]={};
        t.assert_true(std::memcmp(actual_r,zero_r,sizeof(actual_r)) == 0);
        t.assert_true(std::memcmp(actual_s,zero_s,sizeof(actual_s)) == 0);

    });

    t.test("mixed_generation_restore_keeps_unchanged_recurrent_plane", [](testing & t) {
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,4,true}},1,2,layout))) return;
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),19,0);
        if (!t.assert_true(bool(bank))) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * dr=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * ds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(buffer))) return;
        const float first_r[4]={1,2,3,4}, first_s[4]={5,6,7,8};
        const float second_r[4]={9,10,11,12};
        ggml_backend_tensor_set(r,first_r,0,sizeof(first_r));
        ggml_backend_tensor_set(s,first_s,0,sizeof(first_s));
        if (!t.assert_true(bank->begin_capture({19,0,1}) &&
                bank->capture(backend.get(),0,false,1,r,0) &&
                bank->capture(backend.get(),0,true,1,s,0) &&
                bank->publish_capture() && bank->complete_capture())) return;
        ggml_backend_tensor_set(r,second_r,0,sizeof(second_r));
        if (!t.assert_true(bank->begin_capture({19,0,2}) &&
                bank->capture(backend.get(),0,false,1,r,0) &&
                bank->publish_capture() && bank->complete_capture())) return;
        if (!t.assert_true(bank->begin_restore({19,0,2},1))) return;
        if (!t.assert_true(bank->restore(backend.get(),0,false,dr,0) &&
                bank->restore(backend.get(),0,true,ds,0) &&
                bank->publish_restore() && bank->complete_restore())) return;
        float actual_r[4]={},actual_s[4]={};
        ggml_backend_tensor_get(dr,actual_r,0,sizeof(actual_r));
        ggml_backend_tensor_get(ds,actual_s,0,sizeof(actual_s));
        t.assert_true(std::memcmp(actual_r,second_r,sizeof(actual_r)) == 0);
        t.assert_true(std::memcmp(actual_s,first_s,sizeof(actual_s)) == 0);
    });


    t.test("cpu_graph_publishes_recurrent_snapshots_through_two_slot_stage", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,1,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),31,0);
        if (!t.assert_true(bool(bank))) return;
        if (!t.assert_true(bank->enable_publication(ggml_backend_get_device(backend.get()),
                    ggml_backend_cpu_buffer_type(),2))) return;
        auto * dst_r=bank->publication_tensor(0,false,1);
        auto * dst_s=bank->publication_tensor(0,true,1);
        auto * stage_r=bank->staging_tensor(0,false,1);
        auto * stage_s=bank->staging_tensor(0,true,1);
        if (!t.assert_true(dst_r && dst_s && stage_r && stage_s &&
                bank->staged_publication_enabled())) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * src_r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * src_s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(source))) return;
        const float expected_r[4]={1,2,3,4};
        const float expected_s[8]={8,7,6,5,4,3,2,1};
        ggml_backend_tensor_set(src_r,expected_r,0,sizeof(expected_r));
        ggml_backend_tensor_set(src_s,expected_s,0,sizeof(expected_s));
        auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
        auto * copied_r=ggml_cpy(ctx.get(),src_r,stage_r);
        auto * copied_s=ggml_cpy(ctx.get(),src_s,stage_s);
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),copied_r,dst_r));
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),copied_s,dst_s));
        if (!t.assert_true(bank->begin_staged_capture({31,0,1},2))) return;
        if (!t.assert_true(ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready() && bank->complete_capture())) return;
        const auto * actual_r=bank->snapshot_data(0,false,1,1);
        const auto * actual_s=bank->snapshot_data(0,true,1,1);
        t.assert_true(actual_r && !std::memcmp(actual_r,expected_r,sizeof(expected_r)));
        t.assert_true(actual_s && !std::memcmp(actual_s,expected_s,sizeof(expected_s)));
        auto * partial=ggml_new_graph_custom(ctx.get(),16,false);
        ggml_build_forward_expand(partial,ggml_cpy(ctx.get(),copied_r,dst_r));
        if (!t.assert_true(bank->begin_staged_capture({31,0,2},2) &&
                ggml_backend_graph_compute(backend.get(),partial) == GGML_STATUS_SUCCESS)) return;
        t.assert_true(!bank->publish_capture());
        bank->cancel();
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        if (!t.assert_true(bank->begin_staged_capture({31,0,2},2) &&
                ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready() && bank->complete_capture())) return;
        t.assert_true(bank->snapshot_data(0,false,1,2) != nullptr);
        t.assert_true(bank->snapshot_data(0,true,1,2) != nullptr);
    });

    t.test("three_layers_reuse_two_slots_only_after_d2h_consumption", [&](testing & t) {
        ggml_backend_load_all();
        const bool gpu=argc > 1 && !std::strcmp(argv[1],"--cuda");
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
        auto * host_type=ggml_backend_dev_host_buffer_type(device);
        if (!host_type) host_type=ggml_backend_cpu_buffer_type();
        llama_recurrent_spill_layout layout;
        std::vector<llama_recurrent_spill_layer> layers(
            3,{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true});
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,1,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host_type,33,0);
        if (!t.assert_true(bool(bank) &&
                bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        std::array<ggml_tensor *,3> r{},s{};
        for (size_t layer=0;layer<3;++layer) {
            r[layer]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,4,1);
            s[layer]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,8,1);
        }
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(source))) return;
        std::array<std::array<float,4>,3> expected_r{};
        std::array<std::array<float,8>,3> expected_s{};
        auto * graph=ggml_new_graph_custom(ctx.get(),64,false);
        for (size_t layer=0;layer<3;++layer) {
            for (size_t i=0;i<4;++i) expected_r[layer][i]=float(100*layer+i);
            for (size_t i=0;i<8;++i) expected_s[layer][i]=float(-100*layer-int(i));
            ggml_backend_tensor_set(r[layer],expected_r[layer].data(),0,ggml_nbytes(r[layer]));
            ggml_backend_tensor_set(s[layer],expected_s[layer].data(),0,ggml_nbytes(s[layer]));
            auto * staged_r=ggml_cpy(ctx.get(),r[layer],bank->staging_tensor(layer,false,1));
            auto * staged_s=ggml_cpy(ctx.get(),s[layer],bank->staging_tensor(layer,true,1));
            ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),staged_r,
                bank->publication_tensor(layer,false,1)));
            ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),staged_s,
                bank->publication_tensor(layer,true,1)));
        }
        if (!t.assert_true(bank->begin_staged_capture({33,0,1},2) &&
                ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready() && bank->complete_capture())) return;
        for (size_t layer=0;layer<3;++layer) {
            const auto * actual_r=bank->snapshot_data(layer,false,1,1);
            const auto * actual_s=bank->snapshot_data(layer,true,1,1);
            t.assert_true(actual_r && !std::memcmp(actual_r,expected_r[layer].data(),ggml_nbytes(r[layer])));
            t.assert_true(actual_s && !std::memcmp(actual_s,expected_s[layer].data(),ggml_nbytes(s[layer])));
        }
    });

    if (argc > 1 && !std::strcmp(argv[1],"--cuda")) t.test(
        "cuda_graph_publishes_staged_snapshots_without_host_graph_sync", [](testing & t) {
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(device != nullptr)) return;
        ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,1024,2048,true}},1,1,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(
            layout,ggml_backend_dev_host_buffer_type(device),32,0);
        if (!t.assert_true(bool(bank) &&
                bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * src_r=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,1024,1);
        auto * src_s=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2048,1);
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(source))) return;
        std::vector<float> expected_r(1024),expected_s(2048);
        for (size_t i=0;i<expected_r.size();++i) expected_r[i]=float(i)*.125f;
        for (size_t i=0;i<expected_s.size();++i) expected_s[i]=float(i)*-.0625f;
        ggml_backend_tensor_set(src_r,expected_r.data(),0,ggml_nbytes(src_r));
        ggml_backend_tensor_set(src_s,expected_s.data(),0,ggml_nbytes(src_s));
        auto * write_r=ggml_cpy(ctx.get(),src_r,bank->staging_tensor(0,false,1));
        auto * write_s=ggml_cpy(ctx.get(),src_s,bank->staging_tensor(0,true,1));
        auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),write_r,bank->publication_tensor(0,false,1)));
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),write_s,bank->publication_tensor(0,true,1)));
        if (!t.assert_true(bank->begin_staged_capture({32,0,1},2))) return;
        if (!t.assert_true(ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready())) return;
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        if (!t.assert_true(bank->complete_capture())) return;
        const auto * actual_r=bank->snapshot_data(0,false,1,1);
        const auto * actual_s=bank->snapshot_data(0,true,1,1);
        t.assert_true(actual_r && !std::memcmp(actual_r,expected_r.data(),ggml_nbytes(src_r)));
        t.assert_true(actual_s && !std::memcmp(actual_s,expected_s.data(),ggml_nbytes(src_s)));
    });

    if (argc > 1 && !std::strcmp(argv[1],"--cuda")) t.test("cuda_pinned_host_capture_and_restore_keep_publication_order", [](testing & t) {
        ggml_backend_load_all();
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        auto * host_type=ggml_backend_dev_host_buffer_type(dev);
        if (!t.assert_true(host_type && ggml_backend_buft_is_host(host_type))) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,1024,2048,true}},1,2,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host_type,8,0);
        if (!t.assert_true(bool(bank))) return;
        auto stage=llama_recurrent_stage_buffer::create(layout,2,ggml_backend_dev_buffer_type(dev));
        if (!t.assert_true(bool(stage))) return;
        t.assert_true(!ggml_backend_buffer_is_host(stage->buffer()));
        t.assert_equal(stage->layout().total_bytes,stage->bytes());
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,1024);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,2048);
        auto * dr=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,1024);
        auto * ds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,2048);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(buffer))) return;
        std::vector<float> expected_r(1024),expected_s(2048),actual_r(1024),actual_s(2048);
        for (size_t i=0;i<expected_r.size();++i) expected_r[i]=float(i)*.125f;
        for (size_t i=0;i<expected_s.size();++i) expected_s[i]=float(i)*-.0625f;
        ggml_backend_tensor_set(r,expected_r.data(),0,ggml_nbytes(r));
        ggml_backend_tensor_set(s,expected_s.data(),0,ggml_nbytes(s));
        if (!t.assert_true(bank->begin_capture({8,0,1}) &&
                bank->capture(backend.get(),0,false,2,r,0) &&
                bank->capture(backend.get(),0,true,2,s,0) && bank->publish_capture())) return;
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,2));
        if (!t.assert_true(bank->complete_capture() && bank->begin_restore({8,0,1},2))) return;
        if (!t.assert_true(bank->restore(backend.get(),0,false,dr,0) &&
                bank->restore(backend.get(),0,true,ds,0) &&
                bank->publish_restore() && bank->complete_restore())) return;
        ggml_backend_tensor_get(dr,actual_r.data(),0,ggml_nbytes(dr));
        ggml_backend_tensor_get(ds,actual_s.data(),0,ggml_nbytes(ds));
        t.assert_true(std::memcmp(expected_r.data(),actual_r.data(),ggml_nbytes(r)) == 0);
        t.assert_true(std::memcmp(expected_s.data(),actual_s.data(),ggml_nbytes(s)) == 0);
    });

    return t.summary();
}
