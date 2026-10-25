#include "../src/llama-context-memory.h"
#include "../ggml/src/ggml-cuda-graph.h"
#include "testing.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-memory-bounded-kv.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

using phase_status = llama_memory_text_phase_status;
using text_phase = llama_memory_text_phase;

struct fixture {
    ggml_backend_ptr device;
    ggml_backend_ptr cpu{ggml_backend_cpu_init()};
    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> bufts;
    ggml_backend_sched_ptr sched;
    std::vector<ggml_backend_memory_workspace_group> groups;
    ggml_context_ptr ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;

    explicit fixture(ggml_backend_dev_t dev) : device(dev ? ggml_backend_dev_init(dev, nullptr) : ggml_backend_cpu_init()) {
        GGML_ASSERT(device && cpu);
        backends = {device.get(), cpu.get()};
        bufts = {ggml_backend_get_default_buffer_type(device.get()), ggml_backend_get_default_buffer_type(cpu.get())};
        sched.reset(ggml_backend_sched_new(backends.data(), bufts.data(), 2, 256, false, true));
        GGML_ASSERT(sched);
        size_t sizes[] = {8192, 8192, 4096, 4096}, count = 2;
        groups.resize(count);
        GGML_ASSERT(ggml_backend_memory_plan_workspace_groups(bufts.data(), sizes, 2, 2, groups.data(), &count));
        groups.resize(count);
    }

    // Exercise graph metadata reconstruction inside one unchanged scheduler-workspace lifetime.
    void rebuild(size_t elements = 16) {
        ggml_backend_sched_synchronize(sched.get());
        ggml_backend_sched_reset(sched.get());
        ctx.reset(ggml_init({1024*1024, nullptr, true}));
        input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, elements);
        output = ggml_scale(ctx.get(), input, 3.0f);
        ggml_set_input(input);
        ggml_set_output(input); // Preserve the input across repeated replay; otherwise GGML may reuse it in place.
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_set_tensor_backend(sched.get(), input, device.get());
        ggml_backend_sched_set_tensor_backend(sched.get(), output, device.get());
        GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), graph));
        // Reservation resets scheduler assignments; restore the explicit device placement before allocation.
        ggml_backend_sched_set_tensor_backend(sched.get(), input, device.get());
        ggml_backend_sched_set_tensor_backend(sched.get(), output, device.get());
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
        GGML_ASSERT(input->data != output->data);
    }

    // Real scheduler dispatch with output-copy completion checked after the owner drains.
    void run(testing & t, llama_context_memory & owner, float value) {
        float data[16], result[16];
        for (float & x : data) x = value;
        ggml_backend_tensor_set(input, data, 0, sizeof(data));
        for (int i = 0; i < 3; ++i) t.assert_true(owner.compute_async(graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(device.get(), output, result, 0, sizeof(result));
        owner.synchronize();
        for (float x : result) t.assert_equal(value*3, x);
    }
};

struct handoff_probe {
    inline static handoff_probe * active = nullptr;
    ggml_backend_t backend;
    decltype(ggml_backend_i::synchronize) original;
    llama_context_memory & parent;
    llama_context_memory & child;
    bool reentered = false, fail = true;
    handoff_probe(ggml_backend_t backend, llama_context_memory & parent, llama_context_memory & child) :
        backend(backend), original(backend->iface.synchronize), parent(parent), child(child) {
        GGML_ASSERT(!active); active = this;
        backend->iface.synchronize = [](ggml_backend_t backend) {
            active->reentered |= active->parent.prepare_serial_target();
            active->reentered |= active->child.prepare_serial_consumer(text_phase::prefill);
            active->reentered |= active->parent.signal_text_phase(
                {text_phase::decode, 1, true, true, false}).status != phase_status::transition_failed;
            if (active->fail) throw std::runtime_error("injected queue drain failure");
            if (active->original) active->original(backend);
        };
    }
    ~handoff_probe() { backend->iface.synchronize = original; active = nullptr; }
};

// Paired same-binary microbenchmark: dispatch cost only, not full-model tokens per second.
static void benchmark(ggml_backend_dev_t dev) {
    ggml_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    for (bool coordinated : {false, true, true, false}) {
        fixture f(dev);
        std::unique_ptr<llama_context_memory> owner;
        std::vector<llama_compute_arena_binding> legacy;
        if (coordinated) {
            owner = llama_context_memory::create(f.sched.get(), f.backends, f.groups);
            GGML_ASSERT(owner);
        } else {
            GGML_ASSERT(llama_prepare_compute_arena_bindings(f.sched.get(), f.backends, f.groups, legacy));
        }
        f.rebuild();
        float input[16];
        for (float & x : input) x = 1;
        ggml_backend_tensor_set(f.input, input, 0, sizeof(input));
        auto dispatch = [&] {
            const auto status = owner ? owner->compute_async(f.graph) : ggml_backend_sched_graph_compute_async(f.sched.get(), f.graph);
            GGML_ASSERT(status == GGML_STATUS_SUCCESS);
        };
        auto drain = [&] {
            if (owner) owner->synchronize();
            else ggml_backend_sched_synchronize(f.sched.get());
        };
        for (int i = 0; i < 64; ++i) dispatch();
        drain();
        double timings[2] = {};
        constexpr int iterations = 10000;
        for (int mode = 0; mode < 2; ++mode) {
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) {
                dispatch();
                if (mode) drain();
            }
            drain();
            timings[mode] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / iterations;
        }
        std::printf("workspace-bench,%s,queued_us=%.3f,synchronized_us=%.3f\n",
            coordinated ? "coordinated" : "legacy", timings[0], timings[1]);
        if (!owner) {
            if (dev) {
                auto release = reinterpret_cast<ggml_backend_cuda_graph_release_all_t>(
                    ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_release_all"));
                GGML_ASSERT(release);
                release(f.device.get());
            }
            for (const auto & binding : legacy) {
                GGML_ASSERT(ggml_backend_sched_detach_memory_lease(f.sched.get(), f.backends[binding.first_slot]));
            }
        }
    }
}

int main(int argc, char ** argv) {
    testing t;
    ggml_backend_dev_t dev = nullptr;
    if (argc > 1 && std::strcmp(argv[1], "--cuda") == 0) {
        ggml_backend_load_all();
        auto * reg = ggml_backend_reg_by_name("CUDA");
        if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
            t.assert_true("CUDA required", false);
            return t.summary();
        }
        dev = ggml_backend_reg_dev_get(reg, 0);
    }

    t.test("unsupported_and_invalid_inputs_reject_without_mutation", [&](testing & t) {
        fixture f(dev);
        t.assert_true(!llama_context_memory::supported({nullptr}));
        t.assert_true(!llama_context_memory::supported({}));
        if (dev) t.assert_true(!llama_context_memory::supported({f.device.get(), f.device.get(), f.cpu.get()}));
        t.assert_true(!llama_context_memory::create(nullptr, f.backends, f.groups));
        auto bad = f.groups;
        bad[0].first_slot = 99;
        t.assert_true(!llama_context_memory::create(f.sched.get(), f.backends, bad));
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.device.get()));
    });
    t.test("shared_reader_role_requires_a_cache_owning_serial_parent",[&](testing & t) {
        fixture target(dev),reader(dev);
        auto parent=llama_context_memory::create(target.sched.get(),target.backends,target.groups);
        if (!t.assert_true(bool(parent))) return;
        t.assert_true(!parent->register_shared_read_consumer());
        auto child=llama_context_memory::borrow_workspace(reader.sched.get(),reader.backends,reader.groups,*parent);
        if (!t.assert_true(bool(child))) return;
        t.assert_true(!child->register_shared_read_consumer());
        t.assert_true(parent->valid() && child->valid());
    });

    t.test("workspace_creation_distinguishes_quota_allocation_and_capture_failure", [&](testing & t) {
        using status = llama_context_memory_create_status;
        fixture target(dev), draft(dev);
        auto parent = llama_context_memory::create(target.sched.get(),target.backends,target.groups);
        if (!t.assert_true(bool(parent))) return;
        const auto original_revision=parent->graph_binding_revision();
        t.assert_true(!parent->grow_workspace({}));
        t.assert_true(parent->valid());
        t.assert_equal(original_revision,parent->graph_binding_revision());
        llama_compute_workspace_plan plan;
        plan.groups=draft.groups;
        plan.phase_sizes.assign(2,std::vector<size_t>(draft.groups.size()));
        for (auto & phase : plan.phase_sizes)
            for (size_t i=0;i<draft.groups.size();++i) phase[i]=draft.groups[i].size;
        llama_context_memory_create_result result;
        const size_t capacity=parent->shared_parent_capacity();
        plan.groups[0].size=capacity+4096;
        plan.phase_sizes[0][0]=plan.phase_sizes[1][0]=plan.groups[0].size;
        auto child=llama_context_memory::create(draft.sched.get(),draft.backends,plan,
            nullptr,parent.get(),false,{},0,&result);
        t.assert_true(!child && result.status == status::insufficient_budget);
        t.assert_equal(capacity,result.available_bytes);
        t.assert_equal(capacity+4096,result.required_bytes);
        t.assert_equal(size_t(4096),result.additional_bytes);
        t.assert_true(parent->valid() && parent->serial_ready());
        plan.groups=draft.groups;
        for (auto & phase : plan.phase_sizes) phase[0]=plan.groups[0].size;
        child=llama_context_memory::create(draft.sched.get(),draft.backends,plan,
            nullptr,parent.get(),false,[] { return false; },0,&result);
        t.assert_true(!child && result.status == status::capture_failed);
        t.assert_equal(size_t(0),result.additional_bytes);
        t.assert_true(parent->valid() && parent->serial_ready());
        child=llama_context_memory::create(draft.sched.get(),draft.backends,plan,
            nullptr,parent.get(),false,{},0,&result);
        t.assert_true(bool(child) && result.status == status::success);
        child.reset();
        t.assert_true(parent->prepare_serial_target());
        struct reject_allocation {
            ggml_backend_buffer_type_t buft;
            decltype(ggml_backend_buffer_type_i::alloc_buffer) original;
            explicit reject_allocation(ggml_backend_buffer_type_t buft) : buft(buft), original(buft->iface.alloc_buffer) {
                buft->iface.alloc_buffer=[](ggml_backend_buffer_type_t,size_t) -> ggml_backend_buffer_t { return nullptr; };
            }
            ~reject_allocation() { buft->iface.alloc_buffer=original; }
        };
        {
            reject_allocation failure(plan.groups[0].buft);
            auto separate=llama_context_memory::create(draft.sched.get(),draft.backends,plan,
                nullptr,nullptr,false,{},0,&result);
            t.assert_true(!separate && result.status == status::allocation_failed);
            t.assert_equal(size_t(0),result.additional_bytes);
        }
        auto separate=llama_context_memory::create(draft.sched.get(),draft.backends,plan,
            nullptr,nullptr,false,{},0,&result);
        t.assert_true(bool(separate) && result.status == status::success);
        separate.reset();
        auto short_parent=llama_context_memory::create(draft.sched.get(),draft.backends,plan,
            nullptr,nullptr,false,{},plan.groups[0].size-plan.groups[0].alignment,&result);
        t.assert_true(!short_parent && result.status == status::insufficient_budget);
        t.assert_equal(plan.groups[0].alignment,result.additional_bytes);
        for (size_t extra : {plan.groups[0].alignment,size_t(1048576)}) {
            auto too_large=plan;
            too_large.groups[0].size=capacity+extra;
            too_large.phase_sizes[0][0]=too_large.phase_sizes[1][0]=capacity+extra;
            reject_allocation failure(plan.groups[0].buft);
            auto rejected=llama_context_memory::create(draft.sched.get(),draft.backends,too_large,
                nullptr,parent.get(),false,{},0,&result);
            t.assert_true(!rejected && result.status == status::insufficient_budget);
            const auto quota=result;
            rejected=llama_context_memory::create(draft.sched.get(),draft.backends,too_large,
                nullptr,nullptr,false,{},0,&result);
            t.assert_true(!rejected && result.status == status::allocation_failed);
            t.assert_equal(extra,quota.additional_bytes);
            t.assert_equal(capacity+extra,quota.required_bytes);
        }
    });

    t.test("scheduler_owner_preserves_measured_maximum_and_computes", [&](testing & t) {
        fixture f(dev);
        t.assert_true(llama_context_memory::supported(f.backends));
        auto owner = llama_context_memory::create(f.sched.get(), f.backends, f.groups);
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(owner->uses_arenas());
        t.assert_equal(size_t(8192), ggml_backend_sched_get_buffer_size(f.sched.get(), f.device.get()));
        f.rebuild();
        f.run(t, *owner, 4);
        t.assert_true(owner->compute_async(nullptr) == GGML_STATUS_FAILED);
        owner.reset();
        for (auto * backend : f.backends) t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), backend));
    });
    t.test("root_workspace_can_recapture_after_graph_retirement_without_a_serial_child", [&](testing & t) {
        fixture f(dev);
        auto owner=llama_context_memory::create(f.sched.get(),f.backends,f.groups);
        if (!t.assert_true(bool(owner))) return;
        f.rebuild(); f.run(t,*owner,4);
        for (int retry=0;retry<2;++retry) {
            if (!t.assert_true(owner->retire_graph())) return;
            t.assert_true(owner->compute_async(f.graph) == GGML_STATUS_FAILED);
            if (!t.assert_true(owner->prepare_serial_target())) return;
            f.rebuild(); f.run(t,*owner,5+retry);
        }
    });
    t.test("bounded_cache_admission_reports_exact_restart_quota_without_mutation",[&](testing & t) {
        fixture shape(dev);
        llama_memory_cache_requirement cache;
        cache.identity={705,1}; cache.role=llama_memory_cache_role::bounded_resident;
        cache.layers=1; cache.history_rows=4096; cache.window_rows=16; cache.headroom_rows=16; cache.row_padding=32;
        cache.format={GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,32,32,1,false}; cache.row_bytes=52;
        llama_memory_bounded_kv_layout layout;
        if (!t.assert_true(llama_memory_bounded_kv_layout_make(cache,{9},shape.groups[0].buft,layout))) return;
        for (size_t staging_bytes : {size_t(0),size_t(1024)}) for (bool decode_larger : {false,true})
        for (size_t shortage : {size_t(1),layout.allocation.alignment,size_t(1024),size_t(0)}) {
            const size_t required=shape.groups[0].size+layout.allocation.bytes+staging_bytes;
            fixture f(dev);
            llama_compute_workspace_plan plan;
            plan.groups=f.groups;
            plan.phase_sizes.assign(2,std::vector<size_t>(f.groups.size()));
            for (size_t i=0;i<f.groups.size();++i) {
                plan.phase_sizes[0][i]=decode_larger ? f.groups[i].size/2 : f.groups[i].size;
                plan.phase_sizes[1][i]=decode_larger ? f.groups[i].size : f.groups[i].size/2;
            }
            auto parent=llama_context_memory::create(f.sched.get(),f.backends,plan,nullptr,nullptr,false,{},required-shortage);
            if (!t.assert_true(bool(parent))) return;
            auto staging=staging_bytes ? parent->reserve_staging(staging_bytes) : nullptr;
            if (!t.assert_true(!staging_bytes || bool(staging))) return;
            ggml_context_ptr metadata(ggml_init({4096+ggml_tensor_overhead(),nullptr,true}));
            ggml_tensor * marker=nullptr;
            if (staging) {
                auto * buffer=ggml_backend_memory_lease_buffer(staging->lease());
                ggml_backend_buffer_clear(buffer,0x6b);
                marker=ggml_new_tensor_1d(metadata.get(),GGML_TYPE_I8,staging_bytes);
                if (!t.assert_equal(GGML_STATUS_SUCCESS,ggml_backend_tensor_alloc(
                        buffer,marker,ggml_backend_buffer_get_base(buffer)))) return;
            }
            f.rebuild(); f.run(t,*parent,3);
            const auto generation=parent->shared_arena_generation(),revision=parent->graph_binding_revision();
            const auto transitions=parent->phase_transition_count();
            if (shortage==1) {
                for (int invalid=0;invalid<4;++invalid) {
                    auto bad=layout;
                    if (invalid==0) bad.allocation.bytes=0;
                    if (invalid==1) bad.allocation.alignment*=2;
                    if (invalid==2) bad.buft=nullptr;
                    if (invalid==3) bad.allocation.bytes=SIZE_MAX&~(bad.allocation.alignment-1);
                    llama_context_memory_create_result rejected;
                    t.assert_true(!parent->reserve_bounded_cache(bad,&rejected));
                    t.assert_true(rejected.status==llama_context_memory_create_status::invalid_configuration);
                    t.assert_equal(size_t(0),rejected.additional_bytes);
                    t.assert_true(parent->valid() && !parent->has_bounded_cache());
                }
                auto large=cache; large.window_rows=512;
                llama_memory_bounded_kv_layout large_layout;
                if (!t.assert_true(llama_memory_bounded_kv_layout_make(large,{9},f.groups[0].buft,large_layout))) return;
                t.assert_true(large_layout.allocation.bytes>parent->shared_parent_capacity());
                llama_context_memory_create_result rejected;
                t.assert_true(!parent->reserve_bounded_cache(large_layout,&rejected));
                t.assert_true(rejected.status==llama_context_memory_create_status::insufficient_budget);
                const size_t larger_required=f.groups[0].size+large_layout.allocation.bytes+staging_bytes;
                t.assert_equal(larger_required,rejected.required_bytes);
                t.assert_equal(larger_required-parent->shared_parent_capacity(),rejected.additional_bytes);
            }
            llama_context_memory_create_result admission;
            auto grant=parent->reserve_bounded_cache(layout,&admission);
            if (shortage) {
                t.assert_true(!grant && admission.status==llama_context_memory_create_status::insufficient_budget);
                t.assert_equal(required-shortage,admission.available_bytes);
                t.assert_equal(required,admission.required_bytes);
                t.assert_equal(shortage,admission.additional_bytes);
                t.assert_true(admission.stage && admission.resource);
                t.assert_true(parent->valid() && !parent->has_bounded_cache());
                t.assert_equal(generation,parent->shared_arena_generation());
                t.assert_equal(revision,parent->graph_binding_revision());
                t.assert_equal(transitions,parent->phase_transition_count());
            } else {
                t.assert_true(grant && admission.status==llama_context_memory_create_status::success);
                t.assert_equal(required,admission.available_bytes);
                t.assert_equal(required,admission.required_bytes);
                t.assert_equal(size_t(0),admission.additional_bytes);
                if (grant) t.assert_equal(layout.allocation.bytes,ggml_backend_buffer_get_size(
                    ggml_backend_memory_lease_buffer(grant->lease())));
                llama_context_memory_create_result duplicate;
                t.assert_true(!parent->reserve_bounded_cache(layout,&duplicate));
                t.assert_true(duplicate.status==llama_context_memory_create_status::invalid_configuration);
                t.assert_equal(size_t(0),duplicate.additional_bytes);
            }
            if (marker) {
                std::vector<uint8_t> contents(staging_bytes);
                ggml_backend_tensor_get(marker,contents.data(),0,contents.size());
                t.assert_true("quota rejection preserves the existing fixed suffix",std::all_of(
                    contents.begin(),contents.end(),[](uint8_t value) { return value==0x6b; }));
            }
            f.rebuild(); f.run(t,*parent,4);
        }
    });
    t.test("bounded_cache_stays_in_parent_across_workspace_handoff_and_cleanup", [&](testing & t) {
        fixture f(dev);
        llama_compute_workspace_plan plan;
        plan.groups=f.groups;
        plan.phase_sizes.assign(2,std::vector<size_t>(f.groups.size()));
        for (auto & phase : plan.phase_sizes) for (size_t i=0;i<f.groups.size();++i) phase[i]=f.groups[i].size;
        auto parent=llama_context_memory::create(f.sched.get(),f.backends,plan,nullptr,nullptr,false,{},16384);
        if (!t.assert_true(bool(parent))) return;
        llama_memory_cache_requirement cache;
        cache.identity={704,1}; cache.role=llama_memory_cache_role::bounded_resident;
        cache.layers=1; cache.history_rows=4096; cache.window_rows=16; cache.headroom_rows=16; cache.row_padding=32;
        cache.format={GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,32,32,1,false}; cache.row_bytes=52;
        llama_memory_bounded_kv_layout layout;
        if (!t.assert_true(llama_memory_bounded_kv_layout_make(cache,{9},f.groups[0].buft,layout))) return;
        auto stage=parent->reserve_staging(1024);
        if (!t.assert_true(bool(stage))) return;
        auto grant=parent->reserve_bounded_cache(layout);
        if (!t.assert_true(bool(grant))) return;
        t.assert_true(parent->has_bounded_cache() && parent->covers_workspace(plan));
        const auto revision_before_quote=parent->graph_binding_revision();
        const auto generation_before_quote=parent->shared_arena_generation();
        const size_t alignment=ggml_backend_buft_get_alignment(f.groups[0].buft);
        const size_t protected_bytes=layout.allocation.bytes+1024;
        const size_t available=16384-protected_bytes;
        for (const auto & loans : std::vector<std::vector<size_t>>{{available},{available-alignment,alignment}}) {
            llama_context_memory_create_result quote;
            t.assert_true(parent->quote_suspended_loans(loans,quote));
            t.assert_true(quote.status==llama_context_memory_create_status::success);
            t.assert_equal(size_t(16384),quote.available_bytes);
            t.assert_equal(size_t(16384),quote.required_bytes);
            t.assert_equal(size_t(0),quote.additional_bytes);
        }
        llama_context_memory_create_result short_quote;
        t.assert_true(!parent->quote_suspended_loans({available+1},short_quote));
        t.assert_true(short_quote.status==llama_context_memory_create_status::insufficient_budget);
        t.assert_equal(size_t(16384)+alignment,short_quote.required_bytes);
        t.assert_equal(alignment,short_quote.additional_bytes);
        for (const auto & loans : std::vector<std::vector<size_t>>{{},{0},{SIZE_MAX,1}}) {
            llama_context_memory_create_result invalid;
            t.assert_true(!parent->quote_suspended_loans(loans,invalid));
            t.assert_true(invalid.status!=llama_context_memory_create_status::success);
        }
        t.assert_equal(revision_before_quote,parent->graph_binding_revision());
        t.assert_equal(generation_before_quote,parent->shared_arena_generation());
        auto oversized=plan;
        for (auto & group : oversized.groups) group.size*=2;
        for (auto & phase : oversized.phase_sizes) for (auto & bytes : phase) bytes*=2;
        t.assert_true(!parent->covers_workspace(oversized));
        t.assert_true(!parent->reserve_bounded_cache(layout));
        auto * buffer=ggml_backend_memory_lease_buffer(grant->lease());
        ggml_backend_buffer_clear(buffer,0x5a);
        f.rebuild(); f.run(t,*parent,4);
        ggml_context_ptr storage(ggml_init({4096+ggml_tensor_overhead(),nullptr,true}));
        auto * bytes_tensor=ggml_new_tensor_1d(storage.get(),GGML_TYPE_F32,layout.allocation.bytes/4);
        t.assert_true(ggml_backend_tensor_alloc(buffer,bytes_tensor,ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
        std::vector<uint8_t> contents(layout.allocation.bytes);
        ggml_backend_tensor_get(bytes_tensor,contents.data(),0,contents.size());
        t.assert_true(std::all_of(contents.begin(),contents.end(),[](uint8_t x){return x == 0x5a;}));
        auto * retained=ggml_backend_memory_lease_retain(grant->lease());
        grant.reset();
        t.assert_true(!parent->reserve_bounded_cache(layout));
        ggml_backend_memory_lease_free(retained);
        t.assert_true(parent->release_bounded_cache());
        grant=parent->reserve_bounded_cache(layout);
        t.assert_true(bool(grant));
        parent.reset();
        t.assert_true(ggml_backend_memory_lease_buffer(grant->lease()) != nullptr);
        grant.reset();
        stage.reset();
    });

    t.test("repeated_graph_rebuild_and_workspace_recreation", [&](testing & t) {
        fixture f(dev);
        for (int cycle = 0; cycle < 4; ++cycle) {
            auto owner = llama_context_memory::create(f.sched.get(), f.backends, f.groups);
            if (!t.assert_true(bool(owner))) return;
            for (int graph = 0; graph < 4; ++graph) {
                f.rebuild();
                f.run(t, *owner, float(cycle + graph));
            }
            t.assert_true(owner->compute_async(f.graph) == GGML_STATUS_SUCCESS);
            owner.reset();
            t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.device.get()));
        }
    });


    t.test("failed_creation_preserves_foreign_attachment_and_can_retry", [&](testing & t) {
        ggml_backend_ptr first(ggml_backend_cpu_init()), second(ggml_backend_cpu_init());
        ggml_backend_buffer_type other = *ggml_backend_cpu_buffer_type();
        std::vector<ggml_backend_t> backends{first.get(), second.get()};
        ggml_backend_buffer_type_t bufts[] = {ggml_backend_cpu_buffer_type(), &other};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(), bufts, 2, 256, false, true));
        const size_t a = ggml_backend_buft_get_alignment(bufts[0]);
        std::vector<ggml_backend_memory_workspace_group> groups{{bufts[0], 8192, a, 0}, {bufts[1], 8192, a, 1}};
        ggml_backend_buffer_ptr foreign(ggml_backend_buft_alloc_buffer(bufts[1], 8192));
        GGML_ASSERT(ggml_backend_sched_set_buffer_range(sched.get(), second.get(), foreign.get(), 0, 8192));
        t.assert_true(!llama_context_memory::create(sched.get(), backends, groups));
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(sched.get(), first.get()));
        t.assert_equal(size_t(8192), ggml_backend_sched_get_buffer_size(sched.get(), second.get()));
        GGML_ASSERT(ggml_backend_sched_clear_buffer_range(sched.get(), second.get()));
        other.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t { return nullptr; };
        t.assert_true(!llama_context_memory::create(sched.get(), backends, groups));
        other.iface.alloc_buffer = ggml_backend_cpu_buffer_type()->iface.alloc_buffer;
        auto owner = llama_context_memory::create(sched.get(), backends, groups);
        t.assert_true(bool(owner));
    });

    t.test("phase_aware_owner_keeps_conservative_live_grant", [&](testing & t) {
        fixture f(dev);
        llama_compute_workspace_plan plan;
        plan.groups = f.groups;
        plan.phase_sizes.assign(2, std::vector<size_t>(f.groups.size()));
        for (size_t group = 0; group < f.groups.size(); ++group) {
            const size_t alignment = f.groups[group].alignment;
            const size_t smaller = std::max(alignment, f.groups[group].size / 2 / alignment * alignment);
            plan.phase_sizes[0][group] = smaller;
            plan.phase_sizes[1][group] = f.groups[group].size;
        }
        auto owner = llama_context_memory::create(f.sched.get(), f.backends, plan);
        if (!t.assert_true(bool(owner))) {
            return;
        }
        t.assert_equal(plan.groups[0].size,
            ggml_backend_sched_get_buffer_size(f.sched.get(), f.device.get()));
        f.rebuild();
        f.run(t, *owner, 3);

        auto invalid = plan;
        invalid.phase_sizes[1].pop_back();
        t.assert_true(!llama_context_memory::create(f.sched.get(), f.backends, invalid));
    });

    t.test("phase_signals_do_not_change_conservative_workspace", [&](testing & t) {
        fixture f(dev);
        auto owner = llama_context_memory::create(f.sched.get(), f.backends, f.groups);
        if (!t.assert_true(bool(owner))) {
            return;
        }
        t.assert_true(!owner->suspend_kv() && !owner->kv_device_suspended());
        t.assert_true(!owner->prepare_serial_decode());
        const auto bytes = ggml_backend_sched_get_buffer_size(f.sched.get(), f.device.get());
        auto result = owner->signal_text_phase({text_phase::prefill, 513, true, true, false});
        t.assert_true(result.status == phase_status::changed);
        t.assert_true(owner->signal_text_phase({text_phase::prefill, 1, true, true, false}).status ==
            phase_status::unchanged);
        t.assert_true(owner->signal_text_phase({text_phase::decode, 1, true, true, false}).status ==
            phase_status::changed);
        const auto before = owner->text_phase();
        t.assert_true(owner->signal_text_phase({text_phase::decode, 1, true, true, true}).status ==
            phase_status::unsupported_execution);
        const auto after = owner->text_phase();
        t.assert_true(after.phase == before.phase);
        t.assert_equal(before.revision, after.revision);
        t.assert_equal(bytes, ggml_backend_sched_get_buffer_size(f.sched.get(), f.device.get()));
        f.rebuild();
        f.run(t, *owner, 5);
    });

    t.test("borrowed_graph_writes_last_legal_byte_without_touching_parent_guard", [&](testing & t) {
        fixture target(dev), draft(dev);
        auto parent = llama_context_memory::create(target.sched.get(),target.backends,target.groups);
        if (!t.assert_true(bool(parent))) return;
        auto groups = draft.groups;
        groups[0].size = 4096;
        llama_compute_workspace_plan plan;
        plan.groups = groups;
        plan.phase_sizes.assign(2,std::vector<size_t>(groups.size()));
        for (auto & phase : plan.phase_sizes)
            for (size_t i = 0; i < groups.size(); ++i) phase[i] = groups[i].size;
        auto child = llama_context_memory::create(draft.sched.get(),draft.backends,plan,nullptr,parent.get());
        if (!t.assert_true(bool(child))) return;
        auto * buffer = parent->shared_parent();
        const size_t capacity = parent->shared_parent_capacity();
        ggml_backend_buffer_clear(buffer,0xa5);
        if (!t.assert_true(child->prepare_serial_draft(text_phase::prefill))) return;
        draft.rebuild(512);
        const auto end = std::max(reinterpret_cast<uintptr_t>(draft.input->data)+ggml_nbytes(draft.input),
            reinterpret_cast<uintptr_t>(draft.output->data)+ggml_nbytes(draft.output));
        t.assert_equal(reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer))+4096,end);
        std::vector<float> data(512,7.0f), result(512);
        ggml_backend_tensor_set(draft.input,data.data(),0,data.size()*sizeof(float));
        if (!t.assert_true(child->compute_async(draft.graph) == GGML_STATUS_SUCCESS)) return;
        ggml_backend_tensor_get_async(draft.device.get(),draft.output,result.data(),0,result.size()*sizeof(float));
        child->synchronize();
        t.assert_true("borrowed graph output",std::all_of(result.begin(),result.end(),[](float value) { return value == 21.0f; }));
        ggml_context_ptr metadata(ggml_init({16384,nullptr,true}));
        auto * entire = ggml_new_tensor_1d(metadata.get(),GGML_TYPE_F32,capacity/sizeof(float));
        if (!t.assert_true(ggml_backend_tensor_alloc(buffer,entire,ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS)) return;
        std::vector<uint8_t> guard(capacity-4096);
        ggml_backend_tensor_get(entire,guard.data(),4096,guard.size());
        t.assert_true("adjacent parent guard",!guard.empty() && std::all_of(guard.begin(),guard.end(),[](uint8_t value) { return value == 0xa5; }));
    });

    t.test("cancelled_serial_graph_retires_pins_and_can_recapture", [&](testing & t) {
        fixture target(dev),draft(dev);
        auto parent=llama_context_memory::create(target.sched.get(),target.backends,target.groups);
        if (!t.assert_true(bool(parent))) return;
        llama_compute_workspace_plan plan;
        plan.groups=draft.groups;
        plan.phase_sizes.assign(2,std::vector<size_t>(draft.groups.size()));
        for (auto & phase : plan.phase_sizes)
            for (size_t i=0;i<draft.groups.size();++i) phase[i]=draft.groups[i].size;
        auto child=llama_context_memory::create(draft.sched.get(),draft.backends,plan,nullptr,parent.get());
        if (!t.assert_true(bool(child) && child->prepare_serial_draft(text_phase::prefill))) return;
        draft.rebuild(); draft.run(t,*child,2);
        struct forced_compute {
            ggml_backend_t backend;
            decltype(ggml_backend_i::graph_compute) original;
            explicit forced_compute(ggml_backend_t backend,bool cancel=true) : backend(backend),original(backend->iface.graph_compute) {
                if (cancel) backend->iface.graph_compute=[](ggml_backend_t,ggml_cgraph *) { return GGML_STATUS_ABORTED; };
                else backend->iface.graph_compute=[](ggml_backend_t,ggml_cgraph *) { return GGML_STATUS_FAILED; };
            }
            ~forced_compute() { backend->iface.graph_compute=original; }
        };
        {
            forced_compute abort(draft.device.get());
            t.assert_true(child->compute_async(draft.graph) == GGML_STATUS_ABORTED);
        }
        t.assert_true(parent->valid() && child->valid());
        if (!t.assert_true(child->prepare_serial_draft(text_phase::prefill) && child->serial_ready())) return;
        draft.rebuild(); draft.run(t,*child,7);
        if (!t.assert_true(parent->prepare_serial_target())) return;
        target.rebuild(); target.run(t,*parent,3);
        if (!t.assert_true(child->prepare_serial_draft(text_phase::decode))) return;
        draft.rebuild(); draft.run(t,*child,5);
        {
            forced_compute failure(draft.device.get(),false);
            t.assert_true(child->compute_async(draft.graph) == GGML_STATUS_FAILED);
        }
        t.assert_true(!child->serial_ready() && !child->prepare_serial_draft(text_phase::decode));
        if (!t.assert_true(child->retire_graph() && child->prepare_serial_draft(text_phase::decode))) return;
        draft.rebuild(); draft.run(t,*child,11);
    });
    t.test("three_serial_schedulers_share_scratch_without_overlapping_admission", [&](testing & t) {
        fixture target(dev), draft(dev), vision(dev);
        auto parent = llama_context_memory::create(target.sched.get(), target.backends, target.groups);
        if (!t.assert_true(bool(parent))) return;
        llama_compute_workspace_plan draft_plan;
        draft_plan.groups = draft.groups;
        draft_plan.phase_sizes.assign(2, std::vector<size_t>(draft.groups.size()));
        for (auto & phase : draft_plan.phase_sizes)
            for (size_t i = 0; i < draft.groups.size(); ++i) phase[i] = draft.groups[i].size;
        auto first = llama_context_memory::create(draft.sched.get(), draft.backends, draft_plan, nullptr, parent.get());
        auto second = llama_context_memory::borrow_workspace(vision.sched.get(), vision.backends, vision.groups, *parent);
        if (!t.assert_true(bool(first) && bool(second))) return;
        t.assert_true(!llama_context_memory::borrow_workspace(draft.sched.get(), draft.backends, draft.groups, *parent));
        t.assert_true(!llama_context_memory::borrow_workspace(vision.sched.get(), vision.backends, vision.groups, *first));
        t.assert_true(parent->prepare_serial_target());
        target.rebuild();
        target.run(t, *parent, 2);
        float copied[16] = {};
        t.assert_true(parent->compute_async(target.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(target.device.get(), target.output, copied, 0, sizeof(copied));
        {
            handoff_probe probe(target.device.get(), *parent, *first);
            t.assert_true(!first->prepare_serial_consumer(text_phase::prefill));
            t.assert_true(parent->serial_ready());
            t.assert_true(!probe.reentered);
            probe.fail = false;
            t.assert_true(first->prepare_serial_consumer(text_phase::prefill));
            t.assert_true(!probe.reentered);
        }
        for (float value : copied) t.assert_equal(6.0f, value);
        t.assert_true(first->prepare_serial_consumer(text_phase::prefill));
        draft.rebuild();
        t.assert_true(parent->compute_async(target.graph) == GGML_STATUS_FAILED);
        t.assert_true(!parent->serial_ready());
        draft.run(t, *first, 3);
        t.assert_true(first->prepare_serial_consumer(text_phase::decode));
        draft.run(t, *first, 3);
        t.assert_true(second->prepare_serial_consumer(text_phase::prefill));
        vision.rebuild();
        t.assert_true(first->compute_async(draft.graph) == GGML_STATUS_FAILED);
        vision.run(t, *second, 4);
        if (dev) {
            auto query = reinterpret_cast<ggml_backend_cuda_graph_is_captured_t>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_is_captured"));
            if (!t.assert_true(query != nullptr)) return;
            const void * key = ggml_graph_node(vision.graph, 0);
            t.assert_true(query(vision.device.get(), key));
            t.assert_true(parent->prepare_serial_target());
            t.assert_true(!query(vision.device.get(), key));
        }
        const auto base = [](const llama_context_memory & owner) {
            return ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(owner.workspace_leases().front()));
        };
        t.assert_true(base(*parent) == base(*first) && base(*first) == base(*second));
        t.assert_true(!second->prepare_serial_consumer(text_phase::decode));
        for (int i = 0; i < 3; ++i) {
            t.assert_true(parent->prepare_serial_target());
            t.assert_true(!second->serial_ready());
            target.run(t, *parent, float(5 + i));
            t.assert_true(second->prepare_serial_consumer(text_phase::prefill));
            vision.run(t, *second, float(8 + i));
        }
        first.reset();
        t.assert_true(parent->prepare_serial_target());
        target.run(t, *parent, 11);
        parent.reset();
        t.assert_true(!second->prepare_serial_consumer(text_phase::prefill));
        t.assert_true(second->compute_async(vision.graph) == GGML_STATUS_FAILED);
        second.reset();
    });

    if (dev) {
        t.test("retirement_clears_scheduler_created_cuda_captures", [&](testing & t) {
            fixture f(dev);
            auto owner = llama_context_memory::create(f.sched.get(), f.backends, f.groups);
            if (!t.assert_true(bool(owner))) return;
            f.rebuild();
            f.run(t, *owner, 2);
            auto * reg = ggml_backend_dev_backend_reg(dev);
            auto query = reinterpret_cast<ggml_backend_cuda_graph_is_captured_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_is_captured"));
            if (!t.assert_true(query != nullptr)) return;
            const void * key = ggml_graph_node(f.graph, 0);
            t.assert_true(query(f.device.get(), key));
            owner.reset();
            t.assert_true(!query(f.device.get(), key));
        });
    }
    if (argc > 1 && (std::strcmp(argv[argc - 1], "--bench") == 0) && t.failures == 0) benchmark(dev);
    return t.summary();
}
