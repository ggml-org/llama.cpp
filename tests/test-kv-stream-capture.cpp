#include "kv-stream-block-test.h"
#include "../src/llama-kv-stream-capture.h"
#include <chrono>
#include <iomanip>
#ifdef KV_CAPTURE_CUDA_TEST
#include <cuda_runtime_api.h>
void kv_capture_test_begin(ggml_backend_t backend);
bool kv_capture_test_end(ggml_backend_t backend);
#endif

// All graph I/O uses one explicit lease; the KV planes retain their separate coarse lease.
struct capture_graph {
    fixture & f;
    block_workspace work;
    ggml_context_ptr context;
    ggml_tensor * q = nullptr, * mask = nullptr, * output = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<float> qdata;
    ggml_tallocr alloc{};
    size_t padded, queries;
    capture_graph(fixture & f, size_t active, uint32_t layers = 1, size_t queries = 1) : f(f), work(f,1024*1024), padded((active+255)/256*256), queries(queries) {
        context.reset(ggml_init({1024*1024,nullptr,true}));
        alloc = ggml_tallocr_new(ggml_backend_memory_lease_buffer(work.lease.get()));
        q = ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,256,int64_t(queries),4);
        mask = ggml_new_tensor_2d(context.get(),GGML_TYPE_F16,int64_t(padded),int64_t(queries));
        GGML_ASSERT(ggml_tallocr_alloc(&alloc,q) == GGML_STATUS_SUCCESS && ggml_tallocr_alloc(&alloc,mask) == GGML_STATUS_SUCCESS);
        auto * input = q;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            output = f.resident->attention(context.get(),layer,input,mask,active,1.0f/16);
            GGML_ASSERT(output && ggml_tallocr_alloc(&alloc,output) == GGML_STATUS_SUCCESS);
            if (layer+1 < layers) {
                input = ggml_permute(context.get(),output,0,2,1,3);
                GGML_ASSERT(ggml_backend_view_init(input) == GGML_STATUS_SUCCESS);
            }
        }
        graph = ggml_new_graph_custom(context.get(),1024,false);
        ggml_build_forward_expand(graph,output);
        set(active);
    }
    void set(size_t active, float multiplier = 1) {
        qdata.resize(1024*queries);
        for (size_t i = 0; i < qdata.size(); ++i) qdata[i] = multiplier*.3f*std::cos(float(i%211)*.07f);
        std::vector<ggml_fp16_t> values(padded*queries);
        for (size_t query = 0; query < queries; ++query) for (size_t i = 0; i < padded; ++i)
            values[query*padded+i] = ggml_fp32_to_fp16(i <= active-queries+query ? 0 : -INFINITY);
        ggml_backend_tensor_set(q,qdata.data(),0,qdata.size()*sizeof(float));
        ggml_backend_tensor_set(mask,values.data(),0,values.size()*sizeof(ggml_fp16_t));
    }
    std::vector<ggml_backend_memory_lease_t> leases() const { return {f.lease.get(),work.lease.get()}; }
    std::vector<float> read() const {
        std::vector<float> out(1024*queries); ggml_backend_tensor_get(output,out.data(),0,out.size()*sizeof(float)); return out;
    }
};

// Baseline is the already-qualified generic CUDA executor, without KV-specific replay admission checks.
static int benchmark(ggml_backend_t backend, bool guarded) {
    for (uint32_t layers : {1u,8u,16u}) {
        fixture f(backend,true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,256,false,layers);
        GGML_ASSERT(f.attach() && f.resident->synchronize(256));
        auto pin = f.binding->acquire(); capture_graph g(f,256,layers);
        llama_memory_cuda_executor executor(backend);
        llama_kv_stream_cuda_executor capture(backend);
        const auto leases = g.leases();
        GGML_ASSERT(guarded ? capture.bind(*f.resident,pin,g.graph,leases,256) : executor.bind(g.graph,leases,1));
        std::vector<double> times;
        for (int run = 0; run < 55; ++run) {
            const auto start = std::chrono::steady_clock::now();
            GGML_ASSERT(guarded ? capture.compute_async(256) == GGML_STATUS_SUCCESS && capture.drain() :
                executor.compute_async(leases,1) == GGML_STATUS_SUCCESS && executor.drain());
            if (run >= 5) times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        }
        std::sort(times.begin(),times.end()); double sum = 0; for (float x : g.read()) sum += x;
        std::cout << std::setprecision(10) << "BENCH," << layers << ',' << (times[24]+times[25])/2 << ',' << sum << ','
                  << (guarded ? capture.is_captured() : executor.is_captured()) << '\n';
    }
    return 0;
}

int main(int argc, char ** argv) {
    const bool guarded = argc > 1 && !std::strcmp(argv[1],"--bench-guarded");
    const bool bench = guarded || (argc > 1 && !std::strcmp(argv[1],"--bench"));
    const bool cuda = bench || (argc > 1 && !std::strcmp(argv[1],"--cuda"));
    const bool expect_capture = !(argc > 2 && !std::strcmp(argv[2],"--no-graphs"));
    testing t;
    t.test("unsupported_backend_preserves_the_callers_binding_pin", [&](testing & t) {
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        fixture f(cpu.get(),false); t.assert_true(f.attach());
        auto pin = f.binding->acquire(); llama_kv_stream_cuda_executor executor(cpu.get());
        t.assert_true(!executor.supported());
        t.assert_true(!executor.bind(*f.resident,pin,nullptr,{},256) && bool(pin));
        t.assert_true(executor.compute_async(256) == GGML_STATUS_FAILED);
    });
    if (!cuda) return t.summary();
    ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    if (argc > 2 && !std::strcmp(argv[2],"--unsupported")) {
        llama_kv_stream_cuda_executor executor(backend.get());
        t.test("unsupported_stream_optimizer_rejects_capture", [&](testing & t) { t.assert_true(!executor.supported()); });
        return t.summary();
    }
    if (bench) return benchmark(backend.get(),guarded);
    t.test("capture_status_hook_is_available", [&](testing & t) {
        auto * reg = ggml_backend_dev_backend_reg(dev);
        using query_t = bool (*)(ggml_backend_t);
        auto query = reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_is_capturing"));
        t.assert_true(query && !query(backend.get()));
    });
    auto warm = [&](llama_kv_stream_cuda_executor & executor, size_t active) {
        for (int i = 0; i < 3; ++i)
            if (!t.assert_true(executor.compute_async(active) == GGML_STATUS_SUCCESS && executor.drain())) return false;
        return t.assert_true(executor.is_captured() == expect_capture);
    };
    t.test("resident_replay_preserves_capture_across_data_updates_and_same_page_growth", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0);
        t.assert_true(f.attach() && f.resident->synchronize(257));
        capture_graph g(f,257); auto pin = f.binding->acquire();
        llama_kv_stream_cuda_executor executor(backend.get());
        if (!t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),257))) return;
        t.assert_true(!pin);
        if (!warm(executor,257)) return;
        close_values(t,oracle(f,0,257,1,g.qdata),g.read(),1e-3f);
        g.set(258,2);
        std::vector<float> row(512,.25f);
        const auto type = ggml_type(f.policy.shape.type_v);
        std::vector<uint8_t> encoded(ggml_row_size(type,256)*2);
        ggml_quantize_chunk(type,row.data(),encoded.data(),0,2,256,nullptr);
        llama_kv_stream_write write;
        t.assert_true(f.content->prepare({{0,ggml_kv_stream_operand::v,257*encoded.size(),encoded.data(),encoded.size()}},write) && f.content->commit(write));
        t.assert_true(executor.compute_async(257) == GGML_STATUS_FAILED); // Dirty mirror: no launch, but no pointer invalidation.
        t.assert_true(executor.is_captured() == expect_capture);
        t.assert_true(f.resident->synchronize(258));
        t.assert_true(executor.compute_async(258) == GGML_STATUS_SUCCESS && executor.drain());
        t.assert_true(executor.is_captured() == expect_capture);
        close_values(t,oracle(f,0,258,1,g.qdata),g.read(),1e-3f);
        t.assert_true(executor.retire_if_affected({99}).status == llama_memory_executor_status::unchanged);
        t.assert_true(executor.retire_if_affected({19}).status == llama_memory_executor_status::retired);
        t.assert_true(!executor.is_captured());
    });
    t.test("unleased_workspace_wrong_owner_and_mutated_descriptors_fail_closed", [&](testing & t) {
        fixture f(backend.get(),true), other(backend.get(),true);
        t.assert_true(f.attach() && other.attach() && f.resident->synchronize(256));
        capture_graph g(f,256); auto pin = f.binding->acquire(), wrong = other.binding->acquire();
        llama_kv_stream_cuda_executor executor(backend.get());
        t.assert_true(!executor.bind(*f.resident,wrong,g.graph,g.leases(),256) && bool(wrong));
        t.assert_true(!executor.bind(*f.resident,pin,g.graph,{f.lease.get()},256) && bool(pin));
        t.assert_true(!executor.bind(*f.resident,pin,g.graph,{g.work.lease.get()},256) && bool(pin));
        if (!t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256)) || !warm(executor,256)) return;
        const auto saved = g.read();
        auto * data = g.q->data; g.q->data = reinterpret_cast<void *>(uintptr_t(16));
        t.assert_true(executor.compute_async(256) == GGML_STATUS_FAILED);
        t.assert_true(!executor.is_captured()); g.q->data = data;
        t.assert_true(saved == g.read());
        pin = f.binding->acquire();
        t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256) && warm(executor,256));
        auto * key = g.output->src[1]; const auto width = key->ne[1]; key->ne[1] = 512;
        t.assert_true(executor.compute_async(256) == GGML_STATUS_FAILED && !executor.is_captured()); key->ne[1] = width;
    });
    t.test("an_alias_view_cannot_hide_a_write_into_the_kv_region", [&](testing & t) {
        fixture f(backend.get(),true); t.assert_true(f.attach() && f.resident->synchronize(256));
        capture_graph g(f,256); auto pin = f.binding->acquire();
        ggml_backend_buffer_ptr alias(ggml_backend_buffer_view(ggml_backend_memory_lease_buffer(f.lease.get()),0,8192));
        if (!t.assert_true(bool(alias))) return;
        llama_kv_stream_cuda_executor executor(backend.get());
        const auto original = *g.output;
        g.output->buffer = alias.get(); g.output->data = ggml_backend_buffer_get_base(alias.get());
        t.assert_true(!executor.bind(*f.resident,pin,g.graph,g.leases(),256));
        t.assert_true(bool(pin));
        *g.output = original;
    });
    t.test("identical_addresses_do_not_substitute_another_kv_metadata_owner", [&](testing & t) {
        fixture f(backend.get(),true); t.assert_true(f.attach() && f.resident->synchronize(256));
        capture_graph g(f,256);
        llama_kv_stream_binding other(37,ggml_backend_buffer_get_type(ggml_backend_memory_lease_buffer(f.lease.get())));
        llama_kv_stream_resident * resident = nullptr;
        t.assert_true(other.bind(f.lease.get(),f.policy,[&](const auto & view) {
            auto result = llama_kv_stream_resident::create(view,f.content,backend.get());
            resident = result.get(); return result;
        }));
        if (!resident) return;
        t.assert_true(resident->synchronize(256));
        auto pin = other.acquire(); llama_kv_stream_cuda_executor executor(backend.get());
        t.assert_true(!executor.bind(*resident,pin,g.graph,g.leases(),256));
        t.assert_true(bool(pin));
    });
    t.test("streamed_transition_cannot_resurrect_an_old_resident_capture", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513);
        ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
        f.policy.pool_bytes = page.storage.bytes*5; f.policy.initial_ring_slots = 3;
        t.assert_true(f.attach() && f.resident->synchronize(256));
        capture_graph g(f,256); auto pin = f.binding->acquire();
        llama_kv_stream_cuda_executor executor(backend.get());
        if (!t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256)) || !warm(executor,256)) return;
        block_inputs input(f,513,1); ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace scratch(f,layout.bytes);
        auto stream_pin = f.binding->acquire();
        t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,513,1.0f/16,scratch.lease.get(),true,2));
        close_values(t,oracle(f,0,513,1,input.qdata),input.read(),1e-3f);
        t.assert_true(f.resident->synchronize(256));
        t.assert_true(!executor.ready(256));
        t.assert_true(executor.compute_async(256) == GGML_STATUS_FAILED && !executor.is_captured());
        pin = f.binding->acquire();
        t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256) && warm(executor,256));
    });
    t.test("replacement_identity_and_page_extent_require_a_fresh_binding", [&](testing & t) {
        fixture f(backend.get(),true); t.assert_true(f.attach() && f.resident->synchronize(256));
        capture_graph g(f,256); auto pin = f.binding->acquire();
        llama_kv_stream_cuda_executor executor(backend.get());
        if (!t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256)) || !warm(executor,256)) return;
        t.assert_true(f.resident->synchronize(257));
        t.assert_true(executor.compute_async(257) == GGML_STATUS_FAILED && !executor.is_captured());
        t.assert_true(f.resident->synchronize(256)); pin = f.binding->acquire();
        t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256) && warm(executor,256));
        auto replacement = llama_kv_stream_host::create(f.host->config(),ggml_backend_buffer_get_type(f.host->buffer()));
        t.assert_true(bool(replacement)); if (!replacement) return;
        ggml_backend_buffer_clear(replacement->buffer(),0);
        t.assert_true(f.content->replace(replacement)); f.host = replacement;
        t.assert_true(f.resident->synchronize(256));
        t.assert_true(executor.compute_async(256) == GGML_STATUS_FAILED && !executor.is_captured());
        pin = f.binding->acquire();
        t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256) && warm(executor,256));
        const auto output = g.read(); t.assert_true(std::all_of(output.begin(),output.end(),[](float x) { return x == 0; }));
    });
    t.test("persistent_kv_lease_survives_an_unrelated_arena_commit", [&](testing & t) {
        fixture f(backend.get(),true);
        ggml_backend_memory_region region; t.assert_true(ggml_backend_memory_lease_get_region(f.lease.get(),&region));
        f.lease.reset();
        t.assert_true(ggml_backend_memory_arena_quiesce(f.arena.get()) && ggml_backend_memory_arena_begin(f.arena.get(),0));
        t.assert_true(ggml_backend_memory_arena_reserve_at(f.arena.get(),9,region.offset,region.size,region.alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr));
        t.assert_true(ggml_backend_memory_arena_commit(f.arena.get()) && ggml_backend_memory_arena_resume(f.arena.get()));
        f.lease.reset(ggml_backend_memory_arena_acquire(f.arena.get(),9));
        t.assert_true(f.attach() && f.resident->synchronize(256));
        capture_graph g(f,256); auto pin = f.binding->acquire();
        llama_kv_stream_cuda_executor executor(backend.get());
        if (!t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256)) || !warm(executor,256)) return;
        const auto generation = ggml_backend_memory_lease_generation(f.lease.get());
        t.assert_true(ggml_backend_memory_arena_quiesce(f.arena.get()) && ggml_backend_memory_arena_begin(f.arena.get(),0));
        t.assert_true(ggml_backend_memory_arena_reserve_at(f.arena.get(),9,region.offset,region.size,region.alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr));
        t.assert_true(ggml_backend_memory_arena_reserve_at(f.arena.get(),90,0,64,64,0,nullptr));
        t.assert_true(ggml_backend_memory_arena_commit(f.arena.get()) && ggml_backend_memory_arena_resume(f.arena.get()));
        lease_ptr fresh(ggml_backend_memory_arena_acquire(f.arena.get(),9),ggml_backend_memory_lease_free);
        t.assert_true(ggml_backend_memory_lease_generation(fresh.get()) > generation);
        t.assert_true(executor.retire_if_affected({90}).status == llama_memory_executor_status::unchanged);
        t.assert_true(executor.compute_async(256) == GGML_STATUS_SUCCESS && executor.drain());
        t.assert_true(executor.is_captured() == expect_capture);
    });
    t.test("nonleased_weight_backing_is_retained_but_unleased_workspace_is_not_admitted", [&](testing & t) {
        fixture f(backend.get(),true); t.assert_true(f.attach() && f.resident->synchronize(256));
        capture_graph g(f,256);
        auto * weight = ggml_new_tensor_3d(g.context.get(),GGML_TYPE_F32,256,4,1);
        ggml_backend_buffer_ptr weights(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend.get()),4096));
        t.assert_true(bool(weights)); if (!weights) return;
        ggml_backend_buffer_set_usage(weights.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        t.assert_true(ggml_backend_tensor_alloc(weights.get(),weight,ggml_backend_buffer_get_base(weights.get())) == GGML_STATUS_SUCCESS);
        std::vector<float> values(1024,1); ggml_backend_tensor_set(weight,values.data(),0,4096);
        g.output = ggml_add(g.context.get(),g.output,weight);
        t.assert_true(ggml_tallocr_alloc(&g.alloc,g.output) == GGML_STATUS_SUCCESS);
        ggml_build_forward_expand(g.graph,g.output);
        auto pin = f.binding->acquire(); llama_kv_stream_cuda_executor executor(backend.get());
        if (!t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256))) return;
        weights.reset(); // The capture wrapper now owns the final reference to this raw buffer.
        if (!warm(executor,256)) return;
        auto expected = oracle(f,0,256,1,g.qdata); for (auto & x : expected) x += 1;
        close_values(t,expected,g.read(),1e-3f);
    });
    t.test("prefill_and_conversion_fallback_remain_outside_resident_capture", [&](testing & t) {
        fixture f(backend.get(),true); t.assert_true(f.attach() && f.resident->synchronize(256));
        capture_graph g(f,256,1,33); auto pin = f.binding->acquire();
        llama_kv_stream_cuda_executor executor(backend.get());
        t.assert_true(!executor.bind(*f.resident,pin,g.graph,g.leases(),256) && bool(pin));
        fixture fallback(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,769,true);
        t.assert_true(fallback.attach() && fallback.resident->synchronize(256));
        llama_kv_stream_capture_stamp stamp; stamp.padded_tokens = 77;
        t.assert_true(!fallback.resident->capture_state(backend.get(),256,stamp) && stamp.padded_tokens == 77);
    });
    t.test("capture_retains_metadata_and_leases_until_native_retirement", [&](testing & t) {
        fixture f(backend.get(),true); t.assert_true(f.attach() && f.resident->synchronize(256));
        capture_graph g(f,256); auto pin = f.binding->acquire();
        llama_kv_stream_cuda_executor executor(backend.get());
        if (!t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256)) || !warm(executor,256)) return;
        for (int i = 0; i < 8; ++i) t.assert_true(executor.compute_async(256) == GGML_STATUS_SUCCESS);
        f.lease.reset(); g.work.lease.reset(); f.binding.reset(); // Wrapper's owner pin keeps resident metadata alive.
        t.assert_true(executor.drain());
        t.assert_true(executor.compute_async(256) == GGML_STATUS_SUCCESS && executor.drain());
        close_values(t,oracle(f,0,256,1,g.qdata),g.read(),1e-3f);
        t.assert_true(ggml_backend_memory_arena_begin(g.work.arena.get(),0));
        t.assert_true(!ggml_backend_memory_arena_commit(g.work.arena.get()));
        ggml_backend_memory_arena_rollback(g.work.arena.get());
        t.assert_true(executor.retire().status == llama_memory_executor_status::retired);
        t.assert_true(ggml_backend_memory_arena_begin(g.work.arena.get(),0));
        t.assert_true(ggml_backend_memory_arena_commit(g.work.arena.get()));
    });
#ifdef KV_CAPTURE_CUDA_TEST
    t.test("host_driven_streaming_rejects_active_capture_without_invalidating_it", [&](testing & t) {
        fixture f(backend.get(),true); t.assert_true(f.attach() && f.resident->synchronize(256));
        capture_graph g(f,256); auto pin = f.binding->acquire();
        llama_kv_stream_cuda_executor executor(backend.get());
        if (!t.assert_true(executor.bind(*f.resident,pin,g.graph,g.leases(),256)) || !warm(executor,256)) return;
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace scratch(f,layout.bytes);
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
        kv_capture_test_begin(backend.get());
        t.assert_true(!f.resident->synchronize(256));
        t.assert_true(!f.resident->begin_sequence({0,1},513,2));
        t.assert_true(!f.resident->compute_streamed(0,g.q,g.mask,g.output,256,1.0f/16,scratch.lease.get(),true,2));
        t.assert_true(!f.resident->configure_writes(1) && !f.resident->configure_feedback(true));
        f.resident->release_write_workspace();
        t.assert_true(!get()->partial(backend.get(),g.output,ggml_backend_memory_lease_buffer(scratch.lease.get()),false));
        t.assert_true(!get()->clear(backend.get(),g.output,ggml_backend_memory_lease_buffer(scratch.lease.get()),true));
        t.assert_true(executor.compute_async(256) == GGML_STATUS_FAILED);
        t.assert_true(executor.retire().status == llama_memory_executor_status::drain_failed);
        t.assert_true(kv_capture_test_end(backend.get()));
        t.assert_true(executor.compute_async(256) == GGML_STATUS_SUCCESS && executor.drain());
        t.assert_true(executor.is_captured() == expect_capture);
    });
#endif
    return t.summary();
}
