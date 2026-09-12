#include "kv-stream-resident-test.h"
#include "../ggml/src/ggml-kv-stream-device.h"

struct block_workspace {
    arena_ptr arena{nullptr, ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};
    // Exercise an exact-sized lease at a nonzero parent offset.
    block_workspace(fixture & f, size_t bytes) {
        auto * type = f.cuda ? llama_kv_stream_device_buffer_type(ggml_backend_get_device(f.backend)) : ggml_backend_cpu_buffer_type();
        arena.reset(ggml_backend_memory_arena_new(type, bytes + 256));
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena.get(), 0));
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(arena.get())));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 19, 128 + (128-base%128)%128, bytes, 64, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(), 19));
    }
};

struct block_inputs {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * q, * mask, * output;
    std::vector<float> qdata;
    size_t active, queries, padded;
    // Keep the full causal-mask row pitch while the two attention calls use separate key slices.
    block_inputs(fixture & f, size_t active, size_t queries, bool masked = false) : active(active), queries(queries), padded((active+255)/256*256) {
        context.reset(ggml_init({65536, nullptr, true}));
        q = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 256, int64_t(queries), 4);
        mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, int64_t(padded), int64_t(queries));
        output = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 256, 4, int64_t(queries));
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), f.backend)); GGML_ASSERT(buffer);
        qdata.resize(queries*4*256);
        for (size_t i = 0; i < qdata.size(); ++i) qdata[i] = .3f*std::cos(float(i%211)*.07f);
        std::vector<ggml_fp16_t> masks(padded*queries);
        for (size_t query = 0; query < queries; ++query) for (size_t token = 0; token < padded; ++token)
            masks[query*padded+token] = ggml_fp32_to_fp16(!masked && token <= active-queries+query ? 0 : -INFINITY);
        ggml_backend_tensor_set(q, qdata.data(), 0, qdata.size()*sizeof(float));
        ggml_backend_tensor_set(mask, masks.data(), 0, masks.size()*sizeof(ggml_fp16_t));
        std::vector<float> sentinel(queries*4*256, -77);
        ggml_backend_tensor_set(output, sentinel.data(), 0, sentinel.size()*sizeof(float));
    }
    // Read only the public output, not intermediate or padded scratch bytes.
    std::vector<float> read() const {
        std::vector<float> result(queries*4*256);
        ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(float));
        return result;
    }
};

// Check finiteness before measuring the worst absolute error.
static void close_values(testing & t, const std::vector<float> & expected, const std::vector<float> & actual, float tolerance) {
    if (!t.assert_equal(expected.size(), actual.size())) return;
    float error = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i])) { t.assert_true(false); return; }
        error = std::max(error, std::abs(actual[i]-expected[i]));
    }
    t.out << "max absolute error = " << error << '\n';
    t.assert_true(error < tolerance);
}

int main(int argc, char ** argv) {
    const bool cuda = argc > 1 && std::strcmp(argv[1], "--cuda") == 0;
    testing t;
    t.test("checked_one_block_workspace_layout", [](testing & t) {
        ggml_kv_stream_block_layout layout;
        t.assert_true(ggml_kv_stream_block_layout_make(4, 256, layout).status == ggml_kv_stream_partial_status::success);
        t.assert_equal(size_t(8256), layout.partial.bytes);
        t.assert_equal(size_t(8320), layout.second_offset);
        t.assert_equal(size_t(16640), layout.value_offset);
        t.assert_equal(size_t(4096), layout.value_bytes);
        t.assert_equal(size_t(20736), layout.status_offset);
        t.assert_equal(size_t(20740), layout.bytes);
        t.assert_true(ggml_kv_stream_block_layout_make(SIZE_MAX, 256, layout).status == ggml_kv_stream_partial_status::overflow);
        t.assert_true(ggml_kv_stream_block_layout_make(4, 0, layout).status == ggml_kv_stream_partial_status::invalid_shape);
        t.assert_equal(size_t(20740), layout.bytes);
    });
    ggml_backend_ptr backend;
    if (cuda) { ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1; backend.reset(ggml_backend_dev_init(dev, nullptr)); }
    else backend.reset(ggml_backend_cpu_init());
    t.test("ordinary_control_still_matches_reference", [&](testing & t) {
        fixture f(backend.get(), cuda); evaluate(t, f, false, 1, 257, 8);
    });
    if (!cuda) return t.summary();
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_partial_ops"));
    t.test("native_partial_hooks_are_discoverable", [&](testing & t) { t.assert_true(get && get() && get()->version == 1); });
    if (!get || !get()) return t.summary();
    t.test("resident_plus_one_block_matches_attention_and_partial_contract", [&](testing & t) {
        for (size_t pages : {size_t(3),size_t(5)}) {
            fixture f(backend.get(), true);
            f.policy.pool_bytes = f.policy.pool_bytes/16*pages; f.policy.initial_ring_slots = 1;
            if (!t.assert_true(f.attach())) return;
            const size_t prefix = (pages-1)/2*256;
            for (auto dimensions : {std::pair<size_t,size_t>{prefix+1,1}, {prefix+255,8}, {prefix+256,33}, {prefix+1,257}}) {
                block_inputs input(f, dimensions.first, dimensions.second);
                ggml_kv_stream_block_layout layout;
                GGML_ASSERT(ggml_kv_stream_block_layout_make(input.queries*4, 256, layout).status == ggml_kv_stream_partial_status::success);
                block_workspace workspace(f, layout.bytes);
                auto pin = f.binding->acquire();
                if (!t.assert_true(f.resident->compute_one_block(1, input.q, input.mask, input.output, input.active, 1.0f/16, workspace.lease.get()))) continue;
                auto actual = input.read();
                close_values(t, oracle(f, 1, input.active, input.queries, input.qdata), actual, 1e-3f);
                auto * wb = ggml_backend_memory_lease_buffer(workspace.lease.get());
                auto * raw = ggml_new_tensor_1d(input.context.get(), GGML_TYPE_I8, int64_t(layout.bytes));
                GGML_ASSERT(ggml_backend_tensor_alloc(wb, raw, ggml_backend_buffer_get_base(wb)) == GGML_STATUS_SUCCESS);
                ggml_kv_stream_partial_batch a, b, merged;
                for (auto item : {std::pair{&a, size_t(0)}, std::pair{&b, layout.second_offset}}) {
                    auto & batch = *item.first;
                    batch.rows = input.queries*4; batch.parts = 2; batch.width = 256;
                    batch.numerator.resize(layout.partial.elements); batch.meta.resize(layout.partial.entries);
                    ggml_backend_tensor_get(raw, batch.numerator.data(), item.second, layout.partial.numerator_bytes);
                    ggml_backend_tensor_get(raw, batch.meta.data(), item.second + layout.partial.meta_offset, layout.partial.meta_bytes);
                }
                if (!t.assert_true(ggml_kv_stream_partial_merge({a.view(), b.view()}, merged).status == ggml_kv_stream_partial_status::success)) continue;
                ggml_kv_stream_partial_value normalized;
                t.assert_true(ggml_kv_stream_partial_normalize(merged.view(), normalized).status == ggml_kv_stream_partial_status::success);
                close_values(t, normalized.value, actual, 1e-5f);
            }
        }
    });
    t.test("invalid_workspace_or_extra_block_preserves_output", [&](testing & t) {
        fixture f(backend.get(), true); f.policy.pool_bytes = f.policy.pool_bytes/16*3; f.policy.initial_ring_slots = 1;
        if (!t.assert_true(f.attach())) return;
        block_inputs input(f, 257, 1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4, 256, layout);
        block_workspace short_workspace(f, layout.bytes-1), workspace(f, layout.bytes);
        auto pin = f.binding->acquire();
        t.assert_true(!f.resident->compute_one_block(0, input.q, input.mask, input.output, 257, 1.0f/16, short_workspace.lease.get()));
        t.assert_true(!f.resident->compute_one_block(0, input.q, input.mask, input.output, 257, 1.0f/16, f.lease.get()));
        t.assert_true(!f.resident->compute_one_block(0, input.q, input.mask, input.output, 513, 1.0f/16, workspace.lease.get()));
        auto actual = input.read(); t.assert_true(std::all_of(actual.begin(), actual.end(), [](float x) { return x == -77; }));
    });
    t.test("invalid_layouts_and_storage_are_rejected_before_copy", [&](testing & t) {
        fixture f(backend.get(), true); f.policy.pool_bytes = f.policy.pool_bytes/16*3; f.policy.initial_ring_slots = 1;
        if (!t.assert_true(f.attach())) return;
        block_inputs input(f,257,2);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(8,256,layout);
        block_workspace workspace(f,layout.bytes);
        auto pin = f.binding->acquire();
        auto original_mask = *input.mask;
        input.mask->nb[1] = 256*2;
        t.assert_true(!f.resident->compute_one_block(0,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get()));
        *input.mask = original_mask;
        input.mask->ne[0] = INT64_MAX;
        t.assert_true(!f.resident->compute_one_block(0,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get()));
        *input.mask = original_mask;
        input.mask->nb[1] = SIZE_MAX;
        t.assert_true(!f.resident->compute_one_block(0,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get()));
        *input.mask = original_mask;
        auto original_output = *input.output;
        input.output->data = input.q->data;
        t.assert_true(!f.resident->compute_one_block(0,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get()));
        *input.output = original_output;
        t.assert_true(!f.resident->compute_one_block(0,input.q,nullptr,input.output,257,1.0f/16,workspace.lease.get()));
        t.assert_true(!f.resident->compute_one_block(0,input.q,input.mask,input.output,257,NAN,workspace.lease.get()));
        t.assert_true(!f.resident->compute_one_block(2,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get()));
        auto actual = input.read();
        t.assert_true(std::all_of(actual.begin(),actual.end(),[](float x){return x == -77;}));
        fixture quant(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0);
        quant.policy.pool_bytes = quant.policy.pool_bytes/16*3; quant.policy.initial_ring_slots = 1;
        if (t.assert_true(quant.attach()))
            t.assert_true(!quant.resident->compute_one_block(0,input.q,input.mask,input.output,257,1.0f/16,workspace.lease.get()));
    });
    t.test("device_merge_validates_payload_before_publication", [&](testing & t) {
        fixture f(backend.get(),true);
        block_inputs input(f,257,1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace workspace(f,layout.bytes);
        auto * wb = ggml_backend_memory_lease_buffer(workspace.lease.get());
        auto * raw = ggml_new_tensor_1d(input.context.get(),GGML_TYPE_I8,int64_t(layout.bytes));
        GGML_ASSERT(ggml_backend_tensor_alloc(wb,raw,ggml_backend_buffer_get_base(wb)) == GGML_STATUS_SUCCESS);
        std::vector<float> a(layout.partial.elements,0), b(a);
        std::vector<ggml_kv_stream_partial_meta> am(layout.partial.entries), bm(am);
        const auto upload = [&] {
            ggml_backend_tensor_set(raw,a.data(),0,layout.partial.numerator_bytes);
            ggml_backend_tensor_set(raw,am.data(),layout.partial.meta_offset,layout.partial.meta_bytes);
            ggml_backend_tensor_set(raw,b.data(),layout.second_offset,layout.partial.numerator_bytes);
            ggml_backend_tensor_set(raw,bm.data(),layout.second_offset+layout.partial.meta_offset,layout.partial.meta_bytes);
        };
        const auto seed = [&] {
            std::fill(a.begin(),a.end(),0); std::fill(b.begin(),b.end(),0);
            std::fill(am.begin(),am.end(),ggml_kv_stream_partial_meta{});
            std::fill(bm.begin(),bm.end(),ggml_kv_stream_partial_meta{});
            for (size_t row = 0; row < 4; ++row) {
                am[2*row] = bm[2*row] = {0,1};
                std::fill_n(a.begin()+row*512,256,1);
                std::fill_n(b.begin()+row*512,256,3);
            }
        };
        seed(); upload();
        t.assert_true(get()->merge(backend.get(),input.output,wb));
        close_values(t,std::vector<float>(1024,2),input.read(),1e-6f);
        // Invalid data appears in the final row: earlier rows must not be partially published.
        for (int problem = 0; problem < 9; ++problem) {
            seed();
            switch (problem) {
                case 0: bm[6].normalizer = NAN; break;
                case 1: bm[6].normalizer = -1; break;
                case 2: bm[6].max_logit = INFINITY; break;
                case 3: bm[6].max_logit = NAN; break;
                case 4: bm[6].normalizer = 0; break;
                case 5: b[3*512] = INFINITY; break;
                case 6: a[3*512] = b[3*512] = std::numeric_limits<float>::max(); break;
                case 7: am[6].normalizer = bm[6].normalizer = std::numeric_limits<float>::max(); break;
                case 8: am[6].normalizer = bm[6].normalizer = std::numeric_limits<float>::denorm_min(); break;
            }
            std::vector<float> sentinel(1024,-77);
            ggml_backend_tensor_set(input.output,sentinel.data(),0,sentinel.size()*sizeof(float));
            upload();
            t.assert_true(!get()->merge(backend.get(),input.output,wb));
            t.assert_true(input.read() == sentinel);
        }
        // Empty sentinels must not dominate a real, very negative maximum.
        seed();
        for (size_t row = 0; row < 4; ++row) {
            am[2*row].max_logit = -1e30f; bm[2*row].max_logit = -1e30f;
            am[2*row+1].max_logit = 100;
        }
        upload(); t.assert_true(get()->merge(backend.get(),input.output,wb));
        close_values(t,std::vector<float>(1024,2),input.read(),1e-6f);
    });
    t.test("masked_row_and_mutable_tail_are_safe", [&](testing & t) {
        fixture f(backend.get(), true); f.policy.pool_bytes = f.policy.pool_bytes/16*3; f.policy.initial_ring_slots = 1;
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(f.lease.get()),255);
        block_inputs masked(f, 257, 1, true), input(f, 257, 1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4, 256, layout);
        block_workspace workspace(f, layout.bytes);
        t.assert_true(f.resident->compute_one_block(0, masked.q, masked.mask, masked.output, 257, 1.0f/16, workspace.lease.get()));
        auto zeros = masked.read(); t.assert_true(std::all_of(zeros.begin(), zeros.end(), [](float x) { return x == 0; }));
        std::vector<float> row(512, 3.0f); std::vector<uint8_t> encoded(f.host->layout().v_token_bytes);
        ggml_quantize_chunk(GGML_TYPE_F16, row.data(), encoded.data(), 0, 2, 256, nullptr);
        llama_kv_stream_write write;
        t.assert_true(f.content->prepare({{0, ggml_kv_stream_operand::v, 256*encoded.size(), encoded.data(), encoded.size()}}, write));
        t.assert_true(f.content->commit(write));
        t.assert_true(f.resident->compute_one_block(0, input.q, input.mask, input.output, 257, 1.0f/16, workspace.lease.get()));
        close_values(t, oracle(f, 0, 257, 1, input.qdata), input.read(), 1e-3f);
        t.assert_equal(f.host->layout().k_token_bytes + f.host->layout().v_token_bytes, f.resident->last_upload_bytes());
    });
    return t.summary();
}
