#include "kv-stream-block-test.h"
#include "../ggml/src/ggml-kv-stream-copy.h"
#include <chrono>
#include <iomanip>
#ifdef KV_COPY_CUDA_TEST
#include <cuda_runtime_api.h>
#include <atomic>
#include <thread>

// Hold an auxiliary CUDA stream without issuing CUDA calls from its callback.
struct delayed_event {
    ggml_backend_event_t event;
    cudaStream_t stream = nullptr;
    std::atomic<bool> opened{false};
    explicit delayed_event(ggml_backend_t backend) : event(ggml_backend_event_new(ggml_backend_get_device(backend))) {
        GGML_ASSERT(event && cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) == cudaSuccess);
    }
    void arm(ggml_backend_t backend) {
        GGML_ASSERT(cudaLaunchHostFunc(stream,[](void * p) {
            auto & gate = *static_cast<delayed_event *>(p);
            while (!gate.opened.load(std::memory_order_acquire)) std::this_thread::yield();
        },this) == cudaSuccess);
        GGML_ASSERT(cudaEventRecord(static_cast<cudaEvent_t>(event->context),stream) == cudaSuccess);
        ggml_backend_event_wait(backend,event);
    }
    void open() { opened.store(true,std::memory_order_release); }
    // Always release callbacks before joining GPU work, including failed test branches.
    ~delayed_event() { open(); cudaStreamSynchronize(stream); ggml_backend_event_free(event); cudaStreamDestroy(stream); }
};
#endif

// Fixed synthetic attention cases, not model/server token throughput.
static int benchmark(ggml_backend_t backend, bool overlap, size_t span_pages, size_t slots, bool fallback, bool wide) {
    const std::vector<size_t> query_counts = wide ? std::vector<size_t>{256,512,1025} : std::vector<size_t>{1,33};
    for (size_t active : {size_t(1025),size_t(8193),size_t(32769)}) for (size_t queries : query_counts) {
        fixture f(backend,true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,fallback);
        ggml_kv_stream_execution page;
        GGML_ASSERT(ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page).status == ggml_kv_stream_status::success);
        f.policy.pool_bytes = page.storage.bytes*(slots+2)+page.conversion.bytes; f.policy.initial_ring_slots = slots;
        GGML_ASSERT(f.attach()); auto pin = f.binding->acquire();
        block_inputs input(f,active,queries);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(queries*4,256,layout);
        block_workspace workspace(f,layout.bytes);
        std::vector<double> times;
        for (size_t i = 0; i < 23; ++i) {
            const auto begin = std::chrono::steady_clock::now();
            GGML_ASSERT(f.resident->compute_streamed(0,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),overlap,span_pages));
            const double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            if (i >= 3) times.push_back(ms);
        }
        std::sort(times.begin(),times.end());
        double sum = 0; for (float value : input.read()) sum += value;
        std::cout << std::setprecision(10) << "BENCH," << active << ',' << queries << ',' << (times[9]+times[10])/2
                  << ',' << f.resident->last_upload_bytes() << ',' << f.resident->last_upload_calls() << ',' << sum
                  << ',' << f.resident->last_attention_calls() << '\n';
    }
    return 0;
}

int main(int argc, char ** argv) {
    const bool wide = argc > 1 && std::strcmp(argv[1],"--bench-wide") == 0;
    const bool ordered = argc > 1 && std::strcmp(argv[1],"--bench-ordered") == 0;
    const bool spans = argc > 1 && std::strcmp(argv[1],"--bench-span") == 0;
    const bool fallback = argc > 1 && std::strcmp(argv[1],"--bench-fallback") == 0;
    const bool bench = wide || ordered || spans || fallback || (argc > 1 && std::strcmp(argv[1],"--bench") == 0);
    const bool cuda = bench || (argc > 1 && std::strcmp(argv[1],"--cuda") == 0);
    ggml_backend_ptr backend;
    if (cuda) {
        ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1;
        backend.reset(ggml_backend_dev_init(dev,nullptr));
    } else backend.reset(ggml_backend_cpu_init());
    if (bench) {
        const size_t span_pages = argc > 2 ? std::strtoull(argv[2],nullptr,10) : (spans ? 2 : 1);
        const size_t slots = argc > 3 ? std::strtoull(argv[3],nullptr,10) : 2;
        if (!span_pages || !slots || slots > 14 || argc > 4) return 2;
        return benchmark(backend.get(),!ordered,span_pages,slots,fallback,wide);
    }
    testing t;
    t.test("contiguous_spans_stop_at_wrap_and_admission_is_atomic", [](testing & t) {
        t.assert_equal(size_t(3),ggml_kv_stream_contiguous_pages(0,11,5,3));
        t.assert_equal(size_t(2),ggml_kv_stream_contiguous_pages(3,11,5,3));
        t.assert_equal(size_t(1),ggml_kv_stream_contiguous_pages(10,11,5,3));
        t.assert_equal(size_t(0),ggml_kv_stream_contiguous_pages(11,11,5,3));
        t.assert_equal(size_t(0),ggml_kv_stream_contiguous_pages(0,11,0,3));
        t.assert_equal(size_t(0),ggml_kv_stream_contiguous_pages(0,11,5,0));
        t.assert_equal(size_t(2),ggml_kv_stream_contiguous_pages(SIZE_MAX-2,SIZE_MAX,SIZE_MAX,SIZE_MAX));
        ggml_kv_stream_copy_state state(5);
        t.assert_true(state.begin() && state.queue(2));
        t.assert_true(!state.queue_span(0,3));
        t.assert_true(state.can_queue(0) && state.can_queue(1));
        t.assert_true(!state.queue_span(4,2) && !state.queue_span(0,0));
        t.assert_true(state.acquire(2) && state.release(2));
        t.assert_true(state.queue_span(0,3));
        t.assert_true(state.waiting(0) && state.waiting(1) && state.waiting(2));
    });
    t.test("slot_state_rejects_early_reuse_and_resets_after_drain", [](testing & t) {
        ggml_kv_stream_copy_state state(2), zero(0);
        t.assert_true(!zero.begin());
        t.assert_true(!state.queue(0));
        t.assert_true(state.begin());
        t.assert_true(!state.begin());
        for (size_t round = 0; round < 100; ++round) {
            const size_t slot = round%2;
            t.assert_true(state.can_queue(slot));
            t.assert_true(state.recycled(slot) == (round >= 2));
            t.assert_true(state.queue(slot));
            t.assert_true(!state.queue(slot));
            t.assert_true(!state.release(slot));
            t.assert_true(state.acquire(slot));
            t.assert_true(!state.acquire(slot) && !state.can_queue(slot));
            t.assert_true(state.release(slot));
            t.assert_true(!state.release(slot));
        }
        t.assert_true(!state.queue(SIZE_MAX) && !state.acquire(2) && !state.release(2));
        state.drained(); t.assert_true(state.begin());
        t.assert_true(!state.recycled(0) && state.queue(0));
        state.drained(); t.assert_true(state.begin() && state.queue(0));
    });
    if (cuda) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto get = reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_copy_ops"));
        t.test("copy_events_are_discoverable", [&](testing & t) { t.assert_true(get && get() && get()->version >= 2 && get()->enqueue_span && get()->stats); });
        if (!get || !get()) return t.summary();
        t.test("wide_microbatches_share_uploads_and_preserve_final_tile_readers", [&](testing & t) {
            for (bool fallback : {false,true}) for (bool token_major : {false,true}) {
                fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,2049,fallback);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*5+page.conversion.bytes; f.policy.initial_ring_slots = 3;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                for (size_t queries : {size_t(255),size_t(256),size_t(257),size_t(511),size_t(512),size_t(1025)}) {
                    block_inputs input(f,2049,queries);
                    if (token_major) {
                        std::vector<float> packed(input.qdata.size());
                        for (size_t q = 0; q < queries; ++q) for (size_t h = 0; h < 4; ++h)
                            std::copy_n(input.qdata.data()+(h*queries+q)*256,256,packed.data()+(q*4+h)*256);
                        ggml_backend_tensor_set(input.q,packed.data(),0,packed.size()*sizeof(float));
                        input.q->nb[1] = 4*256*sizeof(float); input.q->nb[2] = 256*sizeof(float);
                    }
                    ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(queries*4,256,layout);
                    block_workspace workspace(f,layout.bytes);
                    t.assert_true(f.resident->synchronize(256));
                    if (!t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,2))) continue;
                    const auto actual = input.read();
                    if (queries == 1025) for (size_t index : {size_t(255),size_t(256),size_t(1024)}) {
                        std::vector<float> query(4*256), values(4*256);
                        for (size_t head = 0; head < 4; ++head)
                            std::copy_n(input.qdata.data()+(head*queries+index)*256,256,query.data()+head*256);
                        std::copy_n(actual.data()+index*4*256,values.size(),values.data());
                        close_values(t,oracle(f,0,2049-queries+index+1,1,query),values,1e-3f);
                    }
                    const size_t bytes = f.resident->last_upload_bytes(), calls = f.resident->last_upload_calls();
                    t.assert_equal(size_t(1793)*(page.storage.k_token_bytes+page.storage.v_token_bytes),bytes);
                    t.assert_equal(size_t(10),calls);
                    t.assert_equal(fallback ? size_t(9) : size_t(6),f.resident->last_attention_calls());
                    // Independent <=128-query calls exercise different boundaries and one launch each.
                    for (size_t first = 0; first < queries; first += 128) {
                        const size_t count = std::min(size_t(128),queries-first);
                        auto q = *input.q, mask = *input.mask, output = *input.output;
                        q.data = static_cast<char *>(q.data)+first*q.nb[1]; q.ne[1] = int64_t(count);
                        mask.data = static_cast<char *>(mask.data)+first*mask.nb[1]; mask.ne[1] = int64_t(count);
                        mask.nb[2] = mask.nb[3] = count*mask.nb[1];
                        output.data = static_cast<char *>(output.data)+first*output.nb[2]; output.ne[2] = int64_t(count);
                        output.nb[3] = count*output.nb[2];
                        t.assert_true(f.resident->compute_streamed(0,&q,&mask,&output,2049,1.0f/16,workspace.lease.get(),false,2));
                    }
                    close_values(t,actual,input.read(),1e-6f);
                    // Reusing one ring across two layers must not overwrite the last query tile.
                    t.assert_true(f.resident->begin_sequence({0,1},2049,2));
                    t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,2));
                    close_values(t,actual,input.read(),1e-6f);
                    t.assert_true(f.resident->compute_streamed(1,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,2));
                    auto second = input.read();
                    t.assert_true(!f.resident->sequence_active());
                    t.assert_true(f.resident->compute_streamed(1,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),false,2));
                    close_values(t,second,input.read(),1e-6f);
                }
            }
        });
        t.test("late_query_failure_preserves_all_output_and_drains_sequence", [&](testing & t) {
            for (bool fallback : {false,true}) {
                fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,1025,fallback);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*5+page.conversion.bytes; f.policy.initial_ring_slots = 3;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                block_inputs input(f,1025,513);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(513*4,256,layout);
                block_workspace workspace(f,layout.bytes), short_workspace(f,layout.bytes-1);
                t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,short_workspace.lease.get(),true,2));
                const auto before = input.read();
                // Only the final query is invalid, after two valid complete tiles.
                std::vector<float> row(256,NAN);
                for (size_t head = 0; head < 4; ++head)
                    ggml_backend_tensor_set(input.q,row.data(),(head*513+512)*256*sizeof(float),row.size()*sizeof(float));
                t.assert_true(f.resident->begin_sequence({0,1},1025,2));
                t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true,2));
                t.assert_true(before == input.read() && !f.resident->sequence_active());
                ggml_backend_tensor_set(input.q,input.qdata.data(),0,input.qdata.size()*sizeof(float));
                t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true,2));
                std::vector<ggml_fp16_t> masked(input.padded*input.queries,ggml_fp32_to_fp16(-INFINITY));
                ggml_backend_tensor_set(input.mask,masked.data(),0,masked.size()*sizeof(ggml_fp16_t));
                t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true,2));
                const auto empty = input.read();
                t.assert_true(std::all_of(empty.begin(),empty.end(),[](float x) { return x == 0; }));
            }
        });
        t.test("batched_spans_preserve_values_and_reduce_copy_calls", [&](testing & t) {
            for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_IQ4_NL,GGML_TYPE_F32}})
            for (size_t slots : {size_t(3),size_t(5)}) for (bool overlap : {false,true})
            for (size_t limit : {size_t(2),size_t(3),SIZE_MAX}) {
                fixture f(backend.get(),true,pair.first,pair.second,2561);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*(2+slots)+page.conversion.bytes; f.policy.initial_ring_slots = slots;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                block_inputs input(f,2561,33);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(132,256,layout);
                block_workspace workspace(f,layout.bytes);
                t.assert_true(f.resident->synchronize(256));
                t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2561,1.0f/16,workspace.lease.get(),overlap,1));
                auto expected = input.read();
                const size_t before = f.resident->last_upload_calls();
                t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2561,1.0f/16,workspace.lease.get(),overlap,limit));
                close_values(t,expected,input.read(),1e-6f);
                size_t calls = 0;
                for (size_t block = 0; block < 10;) {
                    const size_t count = std::min({limit,slots-block%slots,size_t(10)-block});
                    block += count; calls += 2;
                }
                t.assert_equal(calls,f.resident->last_upload_calls());
                t.assert_equal(size_t(1)+(f.policy.capabilities.direct_pair ? calls/2 : 10),f.resident->last_attention_calls());
                t.assert_true(f.resident->last_upload_calls() < before);
                t.assert_equal(size_t(2305)*(page.storage.k_token_bytes+page.storage.v_token_bytes),f.resident->last_upload_bytes());
            }
        });
        t.test("overlap_matches_ordered_for_native_and_fallback_waves", [&](testing & t) {
            for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_IQ4_NL,GGML_TYPE_F32}})
            for (size_t slots : {size_t(1),size_t(2),size_t(3)}) for (size_t queries : {size_t(1),size_t(33),size_t(257)}) {
                fixture f(backend.get(),true,pair.first,pair.second,2049);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*(2+slots)+page.conversion.bytes; f.policy.initial_ring_slots = slots;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                block_inputs input(f,2049,queries);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(queries*4,256,layout);
                block_workspace workspace(f,layout.bytes);
                for (uint32_t layer : {0u,1u}) {
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),false))) continue;
                    auto expected = input.read();
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true))) continue;
                    t.assert_true(expected == input.read());
                    t.assert_equal(size_t(1793)*(page.storage.k_token_bytes+page.storage.v_token_bytes),f.resident->last_upload_bytes());
                    t.assert_equal(size_t(16),f.resident->last_upload_calls());
                }
            }
        });
        t.test("failed_attention_drains_prefetch_and_host_replacement_rebinds", [&](testing & t) {
            fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,2049);
            f.policy.pool_bytes = f.policy.pool_bytes/16*5; f.policy.initial_ring_slots = 3;
            if (!t.assert_true(f.attach())) return;
            auto pin = f.binding->acquire();
            block_inputs input(f,2049,8);
            ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(32,256,layout);
            block_workspace workspace(f,layout.bytes);
            std::vector<ggml_fp16_t> row(512,ggml_fp32_to_fp16(NAN));
            llama_kv_stream_write write;
            t.assert_true(f.content->prepare({{0,ggml_kv_stream_operand::v,300*row.size()*2,row.data(),row.size()*2}},write));
            t.assert_true(f.content->commit(write));
            t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true));
            auto untouched = input.read();
            t.assert_true(std::all_of(untouched.begin(),untouched.end(),[](float x){return x == -77;}));
            std::fill(row.begin(),row.end(),ggml_fp32_to_fp16(2));
            t.assert_true(f.content->prepare({{0,ggml_kv_stream_operand::v,300*row.size()*2,row.data(),row.size()*2}},write));
            t.assert_true(f.content->commit(write));
            t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),false));
            auto expected = input.read();
            t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true));
            t.assert_true(expected == input.read());
            auto replacement = llama_kv_stream_host::create(f.host->config(),ggml_backend_buffer_get_type(f.host->buffer()));
            populate(replacement);
            t.assert_true(f.content->replace(replacement)); f.host = replacement;
            t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),false));
            expected = input.read();
            t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true));
            t.assert_true(expected == input.read());
        });
        t.test("failed_subspan_drains_and_retries_without_publishing_partial_output", [&](testing & t) {
            for (auto value : {GGML_TYPE_F16,GGML_TYPE_F32}) {
                fixture f(backend.get(),true,GGML_TYPE_F16,value,2049);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*5+page.conversion.bytes; f.policy.initial_ring_slots = 3;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                block_inputs input(f,2049,8);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(32,256,layout);
                block_workspace workspace(f,layout.bytes);
                t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,0));
                for (size_t token : {size_t(300),size_t(2048)}) {
                    auto before = input.read();
                    std::vector<float> row(512,NAN); std::vector<uint8_t> encoded(page.storage.v_token_bytes);
                    ggml_quantize_chunk(value,row.data(),encoded.data(),0,2,256,nullptr);
                    llama_kv_stream_write write;
                    t.assert_true(f.content->prepare({{0,ggml_kv_stream_operand::v,token*encoded.size(),encoded.data(),encoded.size()}},write));
                    t.assert_true(f.content->commit(write));
                    t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,3));
                    t.assert_true(before == input.read());
                    std::fill(row.begin(),row.end(),2);
                    ggml_quantize_chunk(value,row.data(),encoded.data(),0,2,256,nullptr);
                    t.assert_true(f.content->prepare({{0,ggml_kv_stream_operand::v,token*encoded.size(),encoded.data(),encoded.size()}},write));
                    t.assert_true(f.content->commit(write));
                    t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),false,1));
                    auto expected = input.read();
                    t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,3));
                    close_values(t,expected,input.read(),1e-6f);
                }
            }
        });
        t.test("all_writable_pairs_match_with_two_slot_prefetch", [&](testing & t) {
            const ggml_type types[] = {GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0,GGML_TYPE_F32,GGML_TYPE_IQ4_NL};
            for (auto key : types) for (auto value : types) {
                fixture f(backend.get(),true,key,value,1025);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*4+page.conversion.bytes; f.policy.initial_ring_slots = 2;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                block_inputs input(f,1025,8);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(32,256,layout);
                block_workspace workspace(f,layout.bytes);
                if (!t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),false))) continue;
                auto expected = input.read();
                if (!t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true))) continue;
                t.assert_true(expected == input.read());
                t.assert_equal(size_t(769)*(page.storage.k_token_bytes+page.storage.v_token_bytes),f.resident->last_upload_bytes());
                t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true,2));
                close_values(t,expected,input.read(),1e-6f);
                t.assert_equal(size_t(4),f.resident->last_upload_calls());
                t.assert_equal(size_t(769)*(page.storage.k_token_bytes+page.storage.v_token_bytes),f.resident->last_upload_bytes());
            }
        });
#ifdef KV_COPY_CUDA_TEST
        const auto * ops = get();
        t.test("completed_consumer_release_is_explicit", [&](testing & t) {
            if (!t.assert_true(ops->version >= 3 && ops->release_completed)) return;
            fixture f(backend.get(),true);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,1),ops->free);
            if (!t.assert_true(bool(queue))) return;
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            delayed_event pending_consumer(backend.get());
            t.assert_true(ops->begin(queue.get()));
            t.assert_true(!ops->release_completed(queue.get(),0));
            t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            t.assert_true(ops->acquire(queue.get(),0));
            ggml_backend_synchronize(backend.get());
            t.assert_true(ops->release_completed(queue.get(),0));
            t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            t.assert_true(ops->acquire(queue.get(),0));
            pending_consumer.arm(backend.get());
            t.assert_true(ops->release(queue.get(),0)); // the ordinary queued release still works on reuse
            t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(20);
            while (!ops->ready(queue.get(),0) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            t.assert_true(!ops->ready(queue.get(),0));
            pending_consumer.open();
            ops->drain(queue.get());
        });
        t.test("batched_dma_waits_for_every_consumer_and_counts_actual_transfers", [&](testing & t) {
            fixture f(backend.get(),true);
            ggml_kv_stream_layout page,ring;
            ggml_kv_stream_layout_make(f.policy.shape,256,page); ggml_kv_stream_layout_make(f.policy.shape,512,ring);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,2),ops->free);
            if (!t.assert_true(bool(queue))) return;
            ggml_context_ptr context(ggml_init({65536,nullptr,true}));
            auto * raw = ggml_new_tensor_1d(context.get(),GGML_TYPE_I8,int64_t(ring.bytes));
            GGML_ASSERT(ggml_backend_tensor_alloc(device,raw,ggml_backend_buffer_get_base(device)) == GGML_STATUS_SUCCESS);
            ggml_backend_buffer_ptr sink(ggml_backend_buft_alloc_buffer(llama_kv_stream_host_buffer_type(ggml_backend_get_device(backend.get())),2*ring.k_bytes));
            auto * out = static_cast<uint8_t *>(ggml_backend_buffer_get_base(sink.get()));
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            delayed_event last_consumer(backend.get());
            t.assert_true(ops->begin(queue.get()));
            t.assert_true(!ops->enqueue_span(queue.get(),1,host.k,host.v,512,512));
            t.assert_equal(size_t(0),ops->stats(queue.get()).calls);
            t.assert_true(ops->enqueue_span(queue.get(),0,host.k,host.v,512,512));
            t.assert_equal(size_t(2),ops->stats(queue.get()).calls);
            t.assert_true(!ops->enqueue_span(queue.get(),0,host.k,host.v,512,512));
            t.assert_true(ops->acquire(queue.get(),0));
            ggml_backend_tensor_get_async(backend.get(),raw,out,0,page.k_bytes);
            t.assert_true(ops->release(queue.get(),0));
            t.assert_true(ops->acquire(queue.get(),1));
            last_consumer.arm(backend.get());
            ggml_backend_tensor_get_async(backend.get(),raw,out+page.k_bytes,page.k_bytes,page.k_bytes);
            t.assert_true(ops->release(queue.get(),1));
            auto * next_k = static_cast<char *>(host.k)+ring.k_bytes;
            auto * next_v = static_cast<char *>(host.v)+ring.v_bytes;
            t.assert_true(ops->enqueue_span(queue.get(),0,next_k,next_v,257,512));
            t.assert_true(!ops->ready(queue.get(),0) && !ops->ready(queue.get(),1));
            t.assert_equal(size_t(4),ops->stats(queue.get()).calls);
            t.assert_equal(size_t(769)*(page.k_token_bytes+page.v_token_bytes),ops->stats(queue.get()).bytes);
            last_consumer.open();
            t.assert_true(ops->acquire(queue.get(),0) && ops->acquire(queue.get(),1));
            ggml_backend_tensor_get_async(backend.get(),raw,out+ring.k_bytes,0,ring.k_bytes);
            t.assert_true(ops->release(queue.get(),0) && ops->release(queue.get(),1));
            ops->drain(queue.get());
            t.assert_true(std::memcmp(out,host.k,ring.k_bytes) == 0);
            t.assert_true(std::memcmp(out+ring.k_bytes,next_k,257*page.k_token_bytes) == 0);
            t.assert_true(std::all_of(out+ring.k_bytes+257*page.k_token_bytes,out+2*ring.k_bytes,[](uint8_t x){return x == 0;}));
            t.assert_true(ops->begin(queue.get()));
            t.assert_equal(size_t(0),ops->stats(queue.get()).bytes);
            ops->drain(queue.get());
        });
        t.test("invalid_storage_and_out_of_range_sources_do_not_queue", [&](testing & t) {
            fixture f(backend.get(),true);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
            ggml_backend_buffer_ptr pageable(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),page.bytes));
            t.assert_true(!ops->create(backend.get(),device,pageable.get(),f.policy.shape,1));
            t.assert_true(!ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,0));
            t.assert_true(!ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,SIZE_MAX));
            ggml_backend_buffer_ptr short_device(ggml_backend_buffer_view(device,0,page.bytes-1));
            t.assert_true(!ops->create(backend.get(),short_device.get(),f.host->buffer(),f.policy.shape,1));
            std::unique_ptr<void,void(*)(void*)> q(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,1),ops->free);
            if (!t.assert_true(bool(q))) return;
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            const auto end = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(f.host->buffer()))+ggml_backend_buffer_get_size(f.host->buffer());
            t.assert_true(!ops->acquire(q.get(),0) && !ops->release(q.get(),0));
            t.assert_true(ops->begin(q.get()) && !ops->begin(q.get()));
            t.assert_true(!ops->enqueue(q.get(),0,reinterpret_cast<void *>(end-1),host.v,1,256));
            t.assert_true(!ops->enqueue(q.get(),0,reinterpret_cast<void *>(UINTPTR_MAX),host.v,1,256));
            t.assert_true(ops->enqueue(q.get(),0,host.k,host.v,256,256));
            t.assert_true(ops->acquire(q.get(),0));
            ops->drain(q.get()); // cancel an acquired slot without issuing a final consumer
            t.assert_true(ops->begin(q.get()) && ops->enqueue(q.get(),0,host.k,host.v,256,256));
            ops->drain(q.get());
            t.assert_true(cudaGetLastError() == cudaSuccess);
        });
        t.test("producer_and_final_consumer_events_prevent_early_copy_and_reuse", [&](testing & t) {
            fixture f(backend.get(),true);
            ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,1),ops->free);
            if (!t.assert_true(bool(queue))) return;
            ggml_context_ptr context(ggml_init({65536,nullptr,true}));
            auto * raw = ggml_new_tensor_1d(context.get(),GGML_TYPE_I8,int64_t(page.bytes));
            GGML_ASSERT(ggml_backend_tensor_alloc(device,raw,ggml_backend_buffer_get_base(device)) == GGML_STATUS_SUCCESS);
            ggml_backend_buffer_ptr sink(ggml_backend_buft_alloc_buffer(llama_kv_stream_host_buffer_type(ggml_backend_get_device(backend.get())),2*page.k_bytes));
            auto * output = static_cast<uint8_t *>(ggml_backend_buffer_get_base(sink.get()));
            llama_kv_stream_host_layer host; GGML_ASSERT(f.host->layer(0,host));
            delayed_event producer(backend.get()), consumer(backend.get());
            producer.arm(backend.get());
            t.assert_true(ops->begin(queue.get()));
            t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            t.assert_true(!ops->ready(queue.get(),0));
            t.assert_true(cudaGetLastError() == cudaSuccess);
            t.assert_true(!ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            producer.open();
            t.assert_true(ops->acquire(queue.get(),0));
            consumer.arm(backend.get());
            ggml_backend_tensor_get_async(backend.get(),raw,output,0,page.k_bytes);
            t.assert_true(ops->release(queue.get(),0));
            auto * next_k = static_cast<char *>(host.k)+page.k_bytes;
            auto * next_v = static_cast<char *>(host.v)+page.v_bytes;
            t.assert_true(ops->enqueue(queue.get(),0,next_k,next_v,1,256));
            t.assert_true(!ops->ready(queue.get(),0));
            t.assert_true(cudaGetLastError() == cudaSuccess);
            consumer.open();
            t.assert_true(ops->acquire(queue.get(),0));
            ggml_backend_tensor_get_async(backend.get(),raw,output+page.k_bytes,0,page.k_bytes);
            t.assert_true(ops->release(queue.get(),0));
            ops->drain(queue.get());
            t.assert_true(std::memcmp(output,host.k,page.k_bytes) == 0);
            t.assert_true(std::memcmp(output+page.k_bytes,next_k,page.k_token_bytes) == 0);
            t.assert_true(std::all_of(output+page.k_bytes+page.k_token_bytes,output+2*page.k_bytes,[](uint8_t x){return x == 0;}));
            t.assert_true(!ops->ready(queue.get(),0) && ops->begin(queue.get()));
            t.assert_true(!ops->enqueue(queue.get(),1,host.k,host.v,256,256));
            t.assert_true(!ops->enqueue(queue.get(),0,host.k,host.v,257,256));
            t.assert_true(!ops->enqueue(queue.get(),0,output,host.v,256,256));
            t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            ops->drain(queue.get()); // cancel a queued, unacquired block
            t.assert_true(ops->begin(queue.get()));
            ops->drain(queue.get());
        });
        t.test("pending_teardown_drains_and_retains_backing", [&](testing & t) {
            fixture f(backend.get(),true);
            auto * device_type = llama_kv_stream_device_buffer_type(ggml_backend_get_device(backend.get()));
            auto * host_type = llama_kv_stream_host_buffer_type(ggml_backend_get_device(backend.get()));
            ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
            ggml_backend_buffer_ptr device(ggml_backend_buft_alloc_buffer(device_type,page.bytes));
            ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(host_type,page.bytes));
            std::memset(ggml_backend_buffer_get_base(host.get()),0x35,page.bytes);
            void * q = ops->create(backend.get(),device.get(),host.get(),f.policy.shape,1);
            if (!t.assert_true(q != nullptr)) return;
            auto * k = static_cast<char *>(ggml_backend_buffer_get_base(host.get()));
            delayed_event gate(backend.get());
            gate.arm(backend.get());
            t.assert_true(ops->begin(q));
            t.assert_true(ops->enqueue(q,0,k,k+page.v_offset,256,256));
            // Retention must make these releases non-blocking while the transfer is gated.
            host.reset(); device.reset();
            std::thread release([&] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); gate.open(); });
            ops->free(q);
            t.assert_true(gate.opened.load(std::memory_order_acquire));
            release.join();
        });
#endif
    }
    return t.summary();
}
