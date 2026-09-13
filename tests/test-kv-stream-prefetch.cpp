#include "kv-stream-block-test.h"
#include "../src/llama-kv-stream-prefetch.h"
#include <chrono>
#include <iomanip>
#include <thread>

// Eight attention consumers with fixed synthetic Q; this is not a model/server benchmark.
static int benchmark(ggml_backend_t backend, bool sequence) {
    for (size_t active : {size_t(257),size_t(2049),size_t(8193)}) for (size_t queries : {size_t(1),size_t(33)}) {
        fixture f(backend,true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,false,8);
        ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
        f.policy.pool_bytes = page.storage.bytes*16; f.policy.initial_ring_slots = 8;
        GGML_ASSERT(f.attach()); auto pin = f.binding->acquire();
        block_inputs input(f,active,queries);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(queries*4,256,layout);
        block_workspace workspace(f,layout.bytes);
        std::vector<double> times;
        size_t bytes = 0, calls = 0;
        for (size_t sample = 0; sample < 23; ++sample) {
            bytes = calls = 0;
            auto start = std::chrono::steady_clock::now();
            if (sequence) GGML_ASSERT(f.resident->begin_sequence({0,1,2,3,4,5,6,7},active,2));
            for (uint32_t layer = 0; layer < 8; ++layer) {
                GGML_ASSERT(f.resident->compute_streamed(layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),true,2));
                bytes += f.resident->last_upload_bytes(); calls += f.resident->last_upload_calls();
            }
            if (sequence) { const auto stats = f.resident->sequence_stats(); bytes = stats.copy_bytes; calls = stats.copy_calls; }
            if (sample >= 3) times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        }
        std::sort(times.begin(),times.end());
        double sum = 0; for (float value : input.read()) sum += value;
        std::cout << std::setprecision(10) << "BENCH," << active << ',' << queries << ',' << (times[9]+times[10])/2 << ',' << bytes << ',' << calls << ',' << sum << '\n';
    }
    return 0;
}

int main(int argc, char ** argv) {
    const bool sequence = argc > 1 && std::strcmp(argv[1],"--bench-sequence") == 0;
    const bool bench = sequence || (argc > 1 && std::strcmp(argv[1],"--bench") == 0);
    const bool cuda = bench || (argc > 1 && std::strcmp(argv[1],"--cuda") == 0);
    ggml_backend_ptr backend;
    if (cuda) { ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1; backend.reset(ggml_backend_dev_init(dev,nullptr)); }
    else backend.reset(ggml_backend_cpu_init());
    if (bench) return benchmark(backend.get(),sequence);
    testing t;
    t.test("bounded_reservations_preserve_order_despite_deferred_readiness", [](testing & t) {
        llama_kv_stream_prefetch_plan p;
        t.assert_true(p.start({256,256,256,256,256,256,256,256},257,512,256,8,2,256));
        llama_kv_stream_prefetch_request r;
        for (size_t layer = 0; layer < 8; ++layer) {
            if (!t.assert_true(p.reserve(r))) continue;
            t.assert_equal(layer,r.layer); t.assert_true(!r.stable);
        }
        t.assert_equal(size_t(8),p.used()); t.assert_true(!p.reserve(r));
        t.assert_true(p.make_ready(7));
        if (auto * future = p.request(7)) { t.assert_true(p.can_submit(*future)); t.assert_true(p.mark_submitted(7)); }
        t.assert_true(!p.consume(1));
        t.assert_true(p.make_ready(0));
        t.assert_true(p.mark_submitted(0));
        t.assert_true(p.consume(1)); t.assert_equal(size_t(7),p.used());
        if (auto * first = p.front()) t.assert_equal(size_t(1),first->layer);
    });
    t.test("many_waves_use_bounded_records_and_release_partial_spans", [](testing & t) {
        for (size_t slots : {size_t(1),size_t(3),size_t(8)}) for (size_t ceiling : {size_t(1),size_t(2),size_t(5)}) {
            llama_kv_stream_prefetch_plan p;
            const std::vector<size_t> prefixes{256,2304,0,512};
            t.assert_true(p.start(prefixes,2303,2304,256,slots,ceiling,2303));
            size_t layer = 0, token = prefixes[0], consumed = 0;
            for (;;) {
                llama_kv_stream_prefetch_request r;
                while (p.reserve(r)) {
                    t.assert_true(p.used() <= slots && p.pending() <= slots && r.pages <= ceiling && r.slot+r.pages <= slots);
                    t.assert_true(p.mark_submitted(p.pending()-1));
                }
                auto * front = p.front(); if (!front) break;
                while (token == 2304) token = prefixes[++layer];
                t.assert_equal(layer,front->layer); t.assert_equal(token,front->first);
                t.assert_true(p.consume(1)); token += 256; ++consumed;
            }
            t.assert_equal(size_t(24),consumed);
            t.assert_equal(size_t(0),p.used());
        }
    });
    t.test("invalid_padding_does_not_create_a_wholly_empty_source_page", [](testing & t) {
        llama_kv_stream_prefetch_plan p;
        t.assert_true(!p.start({0},300,550,256,2,1,300));
        t.assert_true(!p.start({1},300,512,256,2,1,300));
        t.assert_true(!p.start({0},300,512,0,2,1,300));
        t.assert_true(p.start({0},300,512,256,2,1,300));
    });
    if (cuda) t.test("cross_layer_queue_has_no_three_layer_limit", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,257,false,8);
        f.policy.initial_ring_slots = 8;
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(f.resident->begin_sequence({0,1,2,3,4,5,6,7},257,2))) return;
        t.assert_equal(size_t(8),f.resident->sequence_stats().pending_pages);
        t.assert_true(f.resident->sequence_stats().max_layer_distance > 3);
        block_inputs input(f,257,1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace workspace(f,layout.bytes);
        for (uint32_t layer = 0; layer < 8; ++layer) {
            if (t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get(),true,2)))
                close_values(t,oracle(f,layer,257,1,input.qdata),input.read(),2e-4f);
        }
        t.assert_true(!f.resident->sequence_active());
    });
    if (cuda) t.test("mutable_tail_publication_is_deferred_and_ordered", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,257,false,8);
        f.policy.initial_ring_slots = 8;
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(f.resident->begin_sequence({0,1,2,3,4,5,6,7},257,2,256))) return;
        t.assert_equal(size_t(8),f.resident->sequence_stats().pending_pages);
        t.assert_equal(size_t(0),f.resident->sequence_stats().copy_calls);
        t.assert_true(!f.resident->configure_writes(256) && !f.resident->synchronize(257));
        block_inputs input(f,257,1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace workspace(f,layout.bytes);
        std::vector<ggml_fp16_t> row(512,ggml_fp32_to_fp16(2));
        t.assert_true(!f.resident->publish_sequence_tail({{0,ggml_kv_stream_operand::v,0,row.data(),row.size()*2}}));
        for (uint32_t layer = 0; layer < 8; ++layer) {
            t.assert_true(f.resident->publish_sequence_tail({{layer,ggml_kv_stream_operand::v,256*row.size()*2,row.data(),row.size()*2}}));
            if (t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get(),true,2)))
                close_values(t,oracle(f,layer,257,1,input.qdata),input.read(),1e-3f);
        }
        t.assert_true(!f.resident->sequence_active());
        t.assert_equal(size_t(16),f.resident->sequence_stats().copy_calls);
    });
    if (cuda) t.test("cross_layer_waves_match_single_layer_and_cancel_invalid_order", [&](testing & t) {
        for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_IQ4_NL,GGML_TYPE_F32}})
        for (size_t slots : {size_t(1),size_t(3),size_t(8)}) {
            fixture f(backend.get(),true,pair.first,pair.second,2049,false,4);
            ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
            f.policy.pool_bytes = page.storage.bytes*(slots+4)+page.conversion.bytes; f.policy.initial_ring_slots = slots;
            if (!t.assert_true(f.attach())) continue;
            auto pin = f.binding->acquire();
            block_inputs input(f,2049,33);
            ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(132,256,layout);
            block_workspace workspace(f,layout.bytes);
            std::vector<std::vector<float>> expected;
            for (uint32_t layer = 0; layer < 4; ++layer) {
                t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,2));
                expected.push_back(input.read());
            }
            if (!t.assert_true(f.resident->begin_sequence({0,1,2,3},2049,2))) continue;
            for (uint32_t layer = 0; layer < 4; ++layer) {
                if (t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,2)))
                    close_values(t,expected[layer],input.read(),1e-6f);
                t.assert_true(f.resident->sequence_stats().pending_pages <= slots);
            }
            t.assert_true(!f.resident->sequence_active());
            t.assert_equal(size_t(4*1793)*(page.storage.k_token_bytes+page.storage.v_token_bytes),f.resident->sequence_stats().copy_bytes);
            t.assert_true(f.resident->begin_sequence({0,1,2,3},2049,2));
            auto before = input.read();
            t.assert_true(!f.resident->compute_streamed(1,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,2));
            t.assert_true(!f.resident->sequence_active() && before == input.read());
        }
    });
    if (cuda) t.test("concentrated_placement_skips_resident_layers_and_unknown_mutation_cancels", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,257,false,8);
        f.policy.initial_ring_slots = 8;
        llama_kv_stream_policy_state placement; llama_kv_stream_policy_initialize(f.policy,placement);
        placement.decode_active_pages = 2;
        if (!t.assert_true(f.attach(&placement))) return;
        auto pin = f.binding->acquire();
        t.assert_true(!f.resident->begin_sequence({0,0},257,2));
        t.assert_true(f.resident->begin_sequence({0,1,2,3,4,5,6,7},257,2));
        t.assert_true(f.resident->sequence_stats().max_layer_distance > 3);
        block_inputs input(f,257,1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace workspace(f,layout.bytes);
        for (uint32_t layer = 0; layer < 8; ++layer) {
            if (t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get(),true,2)))
                close_values(t,oracle(f,layer,257,1,input.qdata),input.read(),1e-3f);
        }
        t.assert_true(!f.resident->sequence_active());
        t.assert_true(f.resident->begin_sequence({0,1},257,2));
        auto previous = input.read();
        t.assert_true(f.content->invalidate());
        t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get(),true,2));
        t.assert_true(!f.resident->sequence_active() && previous == input.read());
        t.assert_true(f.resident->begin_sequence({0,1},257,2));
        f.resident->cancel_sequence();
        t.assert_true(!f.resident->sequence_active());
    });
    if (cuda) t.test("future_history_can_be_ready_before_the_reserved_demand_tail", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,257,false,3);
        ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
        f.policy.pool_bytes = page.storage.bytes*6; f.policy.initial_ring_slots = 3;
        llama_kv_stream_policy_state placement; llama_kv_stream_policy_initialize(f.policy,placement); placement.decode_active_pages = 2;
        if (!t.assert_true(f.attach(&placement))) return;
        auto pin = f.binding->acquire();
        t.assert_true(f.resident->begin_sequence({1,0,2},257,2,256));
        const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while (!f.resident->sequence_stats().ready_pages && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        const auto state = f.resident->sequence_stats();
        t.assert_true(state.ready_pages > 0 && !state.demand_ready);
        t.assert_equal(size_t(3),state.pending_pages);
        t.assert_equal(size_t(2),state.copy_calls);
        block_inputs input(f,257,1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace workspace(f,layout.bytes);
        for (uint32_t layer : {1u,0u,2u}) {
            if (t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get(),true,2)))
                close_values(t,oracle(f,layer,257,1,input.qdata),input.read(),1e-3f);
        }
        t.assert_true(!f.resident->sequence_active());
    });
    if (cuda) t.test("all_pairs_cross_layer_wrap_and_partial_reuse_match_controls", [&](testing & t) {
        const ggml_type types[] = {GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0,GGML_TYPE_F32,GGML_TYPE_IQ4_NL};
        for (auto key : types) for (auto value : types) {
            fixture f(backend.get(),true,key,value,513);
            ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
            f.policy.pool_bytes = page.storage.bytes*5+page.conversion.bytes; f.policy.initial_ring_slots = 3;
            if (!t.assert_true(f.attach())) continue;
            auto pin = f.binding->acquire();
            block_inputs input(f,513,8);
            ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(32,256,layout);
            block_workspace workspace(f,layout.bytes);
            std::vector<std::vector<float>> expected;
            for (uint32_t layer = 0; layer < 2; ++layer) {
                t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,513,1.0f/16,workspace.lease.get(),true,2));
                expected.push_back(input.read());
            }
            t.assert_true(f.resident->begin_sequence({0,1},513,2));
            for (uint32_t layer = 0; layer < 2; ++layer)
                if (t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,513,1.0f/16,workspace.lease.get(),true,2)))
                    close_values(t,expected[layer],input.read(),1e-6f);
            t.assert_equal(size_t(2*257)*(page.storage.k_token_bytes+page.storage.v_token_bytes),f.resident->sequence_stats().copy_bytes);
            t.assert_true(f.resident->sequence_stats().peak_pages <= 3 && !f.resident->sequence_active());
        }
    });
    if (cuda) t.test("resident_tail_refresh_rejects_reentrancy_without_cancelling_outer_call", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,257,false,4);
        f.policy.initial_ring_slots = 8;
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        t.assert_true(f.resident->begin_sequence({0,1},257,2,256));
        block_inputs input(f,257,1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace workspace(f,layout.bytes);
        std::vector<ggml_fp16_t> row(512,ggml_fp32_to_fp16(3));
        t.assert_true(f.resident->publish_sequence_tail({{0,ggml_kv_stream_operand::v,256*row.size()*2,row.data(),row.size()*2}}));
        auto * buffer = ggml_backend_memory_lease_buffer(f.lease.get());
        static decltype(buffer->iface.set_tensor) original;
        static std::function<void()> during_write;
        original = buffer->iface.set_tensor;
        struct restore { ggml_backend_buffer_t buffer; ~restore() { buffer->iface.set_tensor = original; during_write = {}; } } cleanup{buffer};
        bool called = false;
        during_write = [&] {
            called = true;
            t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get(),true,2));
            t.assert_true(f.resident->sequence_active());
            if (!f.resident->sequence_active()) throw std::runtime_error("reentrant call cancelled the outer sequence");
        };
        buffer->iface.set_tensor = [](ggml_backend_buffer_t b, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            during_write(); original(b,tensor,data,offset,size);
        };
        t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get(),true,2));
        t.assert_true(called);
        close_values(t,oracle(f,0,257,1,input.qdata),input.read(),1e-3f);
        f.resident->cancel_sequence();
    });
    if (cuda) t.test("failure_and_cancellation_retire_future_prefetch", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,513,false,4);
        f.policy.initial_ring_slots = 8;
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        block_inputs input(f,513,1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace workspace(f,layout.bytes);
        t.assert_true(f.resident->begin_sequence({0,1,2,3},513,2,512));
        std::vector<ggml_fp16_t> row(512,ggml_fp32_to_fp16(NAN));
        t.assert_true(f.resident->publish_sequence_tail({{0,ggml_kv_stream_operand::v,512*row.size()*2,row.data(),row.size()*2}}));
        auto before = input.read();
        t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,513,1.0f/16,workspace.lease.get(),true,2));
        t.assert_true(!f.resident->sequence_active() && before == input.read());
        std::fill(row.begin(),row.end(),ggml_fp32_to_fp16(1));
        llama_kv_stream_write write;
        t.assert_true(f.content->prepare({{0,ggml_kv_stream_operand::v,512*row.size()*2,row.data(),row.size()*2}},write) && f.content->commit(write));
        t.assert_true(f.resident->begin_sequence({0,1,2,3},513,2));
        t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,513,1.0f/16,workspace.lease.get(),true,2));
        close_values(t,oracle(f,0,513,1,input.qdata),input.read(),1e-3f);
        f.resident->cancel_sequence();
        t.assert_true(!f.resident->sequence_active() && f.resident->sequence_stats().pending_pages == 0);
        t.assert_true(f.resident->compute_streamed(1,input.q,input.mask,input.output,513,1.0f/16,workspace.lease.get(),true,2));
    });
    return t.summary();
}
