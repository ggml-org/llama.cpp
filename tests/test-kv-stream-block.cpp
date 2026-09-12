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
    t.test("idle_placement_validation_and_zero_resident_metadata", [&](testing & t) {
        fixture f(backend.get(),cuda);
        f.policy.pool_bytes = f.policy.pool_bytes/16*3; f.policy.initial_ring_slots = 1;
        llama_kv_stream_policy_state placement;
        GGML_ASSERT(llama_kv_stream_policy_initialize(f.policy,placement).status == llama_kv_stream_policy_status::success);
        auto invalid = placement; invalid.ring_slots = UINT32_MAX;
        t.assert_true(!f.attach(&invalid));
        t.assert_true(!f.binding->ready());
        placement.resident_pages_per_layer = 0; placement.ring_slots = placement.budget.pages;
        t.assert_true(f.attach(&placement));
        t.assert_true(!f.resident->synchronize(257));
    });
    if (!cuda) return t.summary();
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_partial_ops"));
    t.test("native_partial_hooks_are_discoverable", [&](testing & t) { t.assert_true(get && get() && get()->version == 3 && get()->fold && get()->clear && get()->capabilities && get()->convert); });
    if (!get || !get()) return t.summary();
    t.test("capabilities_exclude_read_only_weight_quants_before_pool_planning", [&](testing & t) {
        size_t writable = 0, native = 0, fallback = 0;
        for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
            const auto caps = get()->capabilities(backend.get(),type,type);
            writable += caps.k.online_write;
        }
        t.assert_equal(size_t(9),writable);
        const ggml_type types[] = {GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0,GGML_TYPE_F32,GGML_TYPE_IQ4_NL};
        for (auto k : types) for (auto v : types) {
            const auto caps = get()->capabilities(backend.get(),k,v);
            ggml_kv_stream_execution execution;
            t.assert_true(ggml_kv_stream_resolve({k,v,256,256,2,256,128},caps,256,execution).status == ggml_kv_stream_status::success);
            if (execution.attention == ggml_kv_stream_attention::direct) { ++native; t.assert_equal(size_t(0),execution.conversion.bytes); }
            else { ++fallback; t.assert_equal(size_t(524288),execution.conversion.bytes); }
        }
        t.out << "native pairs = " << native << ", fallback pairs = " << fallback << '\n';
        t.assert_equal(size_t(81),native+fallback);
        for (int type : {-1,INT32_MAX,int(GGML_TYPE_Q8_1),int(GGML_TYPE_Q2_K)}) {
            const auto caps = get()->capabilities(backend.get(),type,GGML_TYPE_F16);
            ggml_kv_stream_execution execution;
            t.assert_true(ggml_kv_stream_resolve({type,GGML_TYPE_F16,256,256,2,256,128},caps,256,execution).status != ggml_kv_stream_status::success);
        }
    });
    t.test("fabricated_native_support_cannot_bypass_conversion_budget", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_IQ4_NL,GGML_TYPE_F32);
        const auto actual = f.policy.capabilities;
        ggml_kv_stream_execution execution;
        GGML_ASSERT(ggml_kv_stream_resolve(f.policy.shape,actual,256,execution).status == ggml_kv_stream_status::success);
        f.policy.capabilities.direct_pair = f.policy.capabilities.k.direct_attention = f.policy.capabilities.v.direct_attention = true;
        f.policy.pool_bytes = execution.storage.bytes*3; f.policy.initial_ring_slots = 1;
        t.assert_true(!f.attach());
        t.assert_true(!f.binding->ready());
        f.policy.capabilities = actual;
        f.policy.pool_bytes += execution.conversion.bytes;
        t.assert_true(f.attach());
    });
    t.test("conversion_plane_bounds_aliases_and_values", [&](testing & t) {
        for (auto type : {GGML_TYPE_F32,GGML_TYPE_BF16,GGML_TYPE_Q8_0,GGML_TYPE_Q4_1,GGML_TYPE_IQ4_NL}) {
            fixture f(backend.get(),true,type,type);
            ggml_context_ptr context(ggml_init({65536,nullptr,true}));
            auto * root = ggml_new_tensor_1d(context.get(),type,256*256*2);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(),backend.get()));
            ggml_kv_stream_layout encoded, half;
            ggml_kv_stream_layout_make(f.policy.shape,256,encoded);
            ggml_kv_stream_layout_make({GGML_TYPE_F16,GGML_TYPE_F16,256,256,2,256,128},256,half);
            llama_kv_stream_host_layer host; GGML_ASSERT(f.host->layer(0,host));
            ggml_backend_tensor_set(root,host.k,0,encoded.k_bytes);
            ggml_tensor source = *root;
            source.ne[0] = 256; source.ne[1] = 256; source.ne[2] = 2; source.ne[3] = 1;
            source.nb[1] = encoded.k_token_bytes; source.nb[2] = encoded.k_row_bytes; source.nb[3] = encoded.k_bytes;
            block_workspace exact(f,half.k_bytes), short_plane(f,half.k_bytes-1);
            ggml_tensor output = source; output.type = GGML_TYPE_F16; output.nb[0] = 2;
            output.nb[1] = half.k_token_bytes; output.nb[2] = half.k_row_bytes; output.nb[3] = half.k_bytes;
            output.buffer = ggml_backend_memory_lease_buffer(exact.lease.get()); output.data = ggml_backend_buffer_get_base(output.buffer);
            t.assert_true(get()->supports_conversion(backend.get(),&source,&output));
            if (t.assert_true(get()->convert(backend.get(),&source,&output))) {
                std::vector<ggml_fp16_t> values(256*256*2);
                ggml_backend_tensor_get(&output,values.data(),0,half.k_bytes);
                auto reference = unpack(f,0,false);
                for (size_t i = 0; i < values.size(); ++i)
                    if (!t.assert_equal(ggml_fp32_to_fp16(reference[i]),values[i])) break;
            }
            auto malformed = output;
            malformed.buffer = ggml_backend_memory_lease_buffer(short_plane.lease.get()); malformed.data = ggml_backend_buffer_get_base(malformed.buffer);
            t.assert_true(!get()->convert(backend.get(),&source,&malformed));
            malformed = output; malformed.buffer = source.buffer; malformed.data = source.data;
            t.assert_true(!get()->convert(backend.get(),&source,&malformed));
            malformed = output; malformed.data = static_cast<char *>(output.data)+2;
            t.assert_true(!get()->convert(backend.get(),&source,&malformed));
            malformed = output; malformed.nb[1] += 16;
            t.assert_true(!get()->convert(backend.get(),&source,&malformed));
        }
    });
    t.test("quant_pairs_stream_with_derived_plane_sizes", [&](testing & t) {
        const ggml_type types[] = {GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0,GGML_TYPE_F32,GGML_TYPE_IQ4_NL};
        for (auto k : types) for (auto v : types) for (bool forced : {false,true}) {
            t.out << ggml_type_name(k) << "/" << ggml_type_name(v) << (forced ? " forced fallback" : " selected path") << '\n';
            fixture f(backend.get(),true,k,v,1025,forced);
            ggml_kv_stream_execution execution;
            GGML_ASSERT(ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,execution).status == ggml_kv_stream_status::success);
            f.policy.pool_bytes = execution.storage.bytes*3+execution.conversion.bytes; f.policy.initial_ring_slots = 1;
            if (!t.assert_true(f.attach())) continue;
            auto pin = f.binding->acquire();
            block_inputs input(f,1025,8);
            ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(32,256,layout);
            block_workspace workspace(f,layout.bytes);
            if (t.assert_true(f.resident->compute_streamed(1,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get())))
                close_values(t,oracle(f,1,1025,8,input.qdata),input.read(),2e-4f);
        }
    });
    t.test("bounded_f16_fallback_handles_resident_and_streamed_pages", [&](testing & t) {
        for (auto pair : {std::pair{GGML_TYPE_F32,GGML_TYPE_IQ4_NL},std::pair{GGML_TYPE_IQ4_NL,GGML_TYPE_F32},
                          std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_F16,GGML_TYPE_BF16}}) {
            fixture f(backend.get(),true,pair.first,pair.second,1537,true);
            ggml_kv_stream_execution execution;
            GGML_ASSERT(ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,execution).status == ggml_kv_stream_status::success);
            f.policy.pool_bytes = execution.storage.bytes*5+execution.conversion.bytes; f.policy.initial_ring_slots = 1;
            if (!t.assert_true(f.attach())) continue;
            auto pin = f.binding->acquire();
            for (size_t active : {size_t(257),size_t(1537)}) {
                block_inputs input(f,active,33);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(132,256,layout);
                block_workspace workspace(f,layout.bytes);
                if (t.assert_true(f.resident->compute_streamed(1,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get())))
                    close_values(t,oracle(f,1,active,33,input.qdata),input.read(),2e-4f);
                if (get()->capabilities(backend.get(),pair.first,pair.second).direct_pair) {
                    fixture native(backend.get(),true,pair.first,pair.second,1537);
                    native.policy.pool_bytes = execution.storage.bytes*5; native.policy.initial_ring_slots = 1;
                    if (!t.assert_true(native.attach())) continue;
                    auto native_pin = native.binding->acquire();
                    block_inputs other(native,active,33);
                    block_workspace native_workspace(native,layout.bytes);
                    if (t.assert_true(native.resident->compute_streamed(1,other.q,other.mask,other.output,active,1.0f/16,native_workspace.lease.get())))
                        close_values(t,input.read(),other.read(),2e-4f);
                }
            }
        }
    });
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
    t.test("multi_wave_traversal_wraps_slots_and_keeps_scratch_bounded", [&](testing & t) {
        for (size_t slots : {size_t(1),size_t(2),size_t(3)}) {
            fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,2305);
            f.policy.pool_bytes = f.policy.pool_bytes/16*(2+slots); f.policy.initial_ring_slots = slots;
            f.policy.fixed_ring = true;
            if (!t.assert_true(f.attach())) continue;
            auto pin = f.binding->acquire();
            for (size_t active : {size_t(513),size_t(2048),size_t(2305),size_t(769)}) {
                block_inputs input(f,active,8);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(32,256,layout);
                block_workspace workspace(f,layout.bytes);
                auto * raw = ggml_new_tensor_1d(input.context.get(),GGML_TYPE_I8,int64_t(f.policy.pool_bytes));
                auto * buffer = ggml_backend_memory_lease_buffer(f.lease.get());
                GGML_ASSERT(ggml_backend_tensor_alloc(buffer,raw,ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
                ggml_kv_stream_layout page, ring;
                ggml_kv_stream_layout_make(f.policy.shape,256,page);
                ggml_kv_stream_layout_make(f.policy.shape,slots*256,ring);
                for (uint32_t layer : {0u,1u}) {
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get()))) continue;
                    close_values(t,oracle(f,layer,active,8,input.qdata),input.read(),1e-3f);
                    llama_kv_stream_host_layer host; GGML_ASSERT(f.host->layer(layer,host));
                    const size_t blocks = (input.padded-256)/256;
                    for (size_t slot = 0; slot < std::min(slots,blocks); ++slot) {
                        const size_t block = slot+(blocks-1-slot)/slots*slots;
                        const size_t first = 256+block*256, live = std::min(size_t(256),active-first);
                        for (bool value : {false,true}) {
                            const size_t bytes = value ? page.v_bytes : page.k_bytes;
                            const size_t stride = value ? page.v_token_bytes : page.k_token_bytes;
                            std::vector<uint8_t> actual(bytes), expected(bytes,0);
                            std::memcpy(expected.data(),static_cast<const char *>(value ? host.v : host.k)+first*stride,live*stride);
                            ggml_backend_tensor_get(raw,actual.data(),(value ? ring.v_offset : 0)+slot*bytes,bytes);
                            t.assert_true(actual == expected);
                        }
                    }
                    // The second layer call has no dirty resident bytes left.
                    if (layer == 1) t.assert_equal((active-256)*(f.host->layout().k_token_bytes+f.host->layout().v_token_bytes),f.resident->last_upload_bytes());
                }
            }
        }
    });
    t.test("wide_queries_consume_each_slot_before_reuse_without_extra_uploads", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,2305);
        f.policy.pool_bytes = f.policy.pool_bytes/16*3; f.policy.initial_ring_slots = 1;
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        t.assert_true(f.resident->synchronize(256));
        for (size_t queries : {size_t(1),size_t(33),size_t(257)}) {
            block_inputs input(f,1537,queries);
            ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(queries*4,256,layout);
            block_workspace workspace(f,layout.bytes);
            if (!t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1537,1.0f/16,workspace.lease.get()))) continue;
            close_values(t,oracle(f,0,1537,queries,input.qdata),input.read(),1e-3f);
            t.assert_equal(size_t(12),f.resident->last_upload_calls());
            t.assert_equal(size_t(1281)*(f.host->layout().k_token_bytes+f.host->layout().v_token_bytes),f.resident->last_upload_bytes());
        }
    });
    t.test("concentrated_and_zero_resident_placements_match_attention", [&](testing & t) {
        for (size_t slots : {size_t(1),size_t(2),size_t(4)}) {
            fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,2305);
            f.policy.pool_bytes = f.policy.pool_bytes/16*8; f.policy.initial_ring_slots = slots; f.policy.fixed_ring = true;
            llama_kv_stream_policy_state placement;
            GGML_ASSERT(llama_kv_stream_policy_initialize(f.policy,placement).status == llama_kv_stream_policy_status::success);
            placement.decode_active_pages = 4;
            llama_kv_stream_policy_layout policy_layout;
            t.assert_true(llama_kv_stream_policy_layout_make(f.policy,placement,1024,policy_layout).status == llama_kv_stream_policy_status::success);
            if (!t.assert_true(f.attach(&placement))) continue;
            auto pin = f.binding->acquire();
            for (size_t active : {size_t(1023),size_t(2305),size_t(257)}) {
                block_inputs input(f,active,33);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(132,256,layout);
                block_workspace workspace(f,layout.bytes);
                for (uint32_t layer : {0u,1u}) {
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get()))) continue;
                    close_values(t,oracle(f,layer,active,33,input.qdata),input.read(),1e-3f);
                }
            }
        }
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,2305);
        f.policy.pool_bytes = f.policy.pool_bytes/16*3; f.policy.initial_ring_slots = 1; f.policy.fixed_ring = true;
        llama_kv_stream_policy_state placement;
        GGML_ASSERT(llama_kv_stream_policy_initialize(f.policy,placement).status == llama_kv_stream_policy_status::success);
        placement.resident_pages_per_layer = 0; placement.ring_slots = placement.budget.pages;
        if (!t.assert_true(f.attach(&placement))) return;
        auto pin = f.binding->acquire();
        block_inputs input(f,2305,1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace workspace(f,layout.bytes);
        t.assert_true(!f.resident->synchronize(2305));
        if (t.assert_true(f.resident->compute_streamed(1,input.q,input.mask,input.output,2305,1.0f/16,workspace.lease.get()))) {
            close_values(t,oracle(f,1,2305,1,input.qdata),input.read(),1e-3f);
            t.assert_equal(size_t(2305)*(f.host->layout().k_token_bytes+f.host->layout().v_token_bytes),f.resident->last_upload_bytes());
        }
    });
    t.test("late_block_failure_preserves_output_and_retry_resets_accumulator", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,2305);
        f.policy.pool_bytes = f.policy.pool_bytes/16*3; f.policy.initial_ring_slots = 1;
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        block_inputs input(f,2305,1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace workspace(f,layout.bytes);
        std::vector<ggml_fp16_t> row(512,ggml_fp32_to_fp16(NAN));
        llama_kv_stream_write write;
        t.assert_true(f.content->prepare({{1,ggml_kv_stream_operand::v,2304*row.size()*2,row.data(),row.size()*2}},write));
        t.assert_true(f.content->commit(write));
        t.assert_true(!f.resident->compute_streamed(1,input.q,input.mask,input.output,2305,1.0f/16,workspace.lease.get()));
        auto actual = input.read(); t.assert_true(std::all_of(actual.begin(),actual.end(),[](float x){return x == -77;}));
        std::fill(row.begin(),row.end(),ggml_fp32_to_fp16(2));
        t.assert_true(f.content->prepare({{1,ggml_kv_stream_operand::v,2304*row.size()*2,row.data(),row.size()*2}},write));
        t.assert_true(f.content->commit(write));
        if (t.assert_true(f.resident->compute_streamed(1,input.q,input.mask,input.output,2305,1.0f/16,workspace.lease.get())))
            close_values(t,oracle(f,1,2305,1,input.qdata),input.read(),1e-3f);
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
        const auto caps = get()->capabilities(backend.get(),GGML_TYPE_Q2_K,GGML_TYPE_Q4_0);
        t.assert_true(!caps.k.online_write);
        ggml_kv_stream_execution execution;
        t.assert_true(ggml_kv_stream_resolve({GGML_TYPE_Q2_K,GGML_TYPE_Q4_0,256,256,2,256,128},caps,256,execution).status ==
            ggml_kv_stream_status::unsupported_write);
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
            upload();
            t.assert_true(get()->fold(backend.get(),input.output,wb) == (problem == 8));
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
    t.test("incremental_device_folding_matches_reference_without_early_normalization", [&](testing & t) {
        fixture f(backend.get(),true);
        block_inputs input(f,257,1);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(4,256,layout);
        block_workspace workspace(f,layout.bytes);
        auto * wb = ggml_backend_memory_lease_buffer(workspace.lease.get());
        auto * raw = ggml_new_tensor_1d(input.context.get(),GGML_TYPE_I8,int64_t(layout.bytes));
        GGML_ASSERT(ggml_backend_tensor_alloc(wb,raw,ggml_backend_buffer_get_base(wb)) == GGML_STATUS_SUCCESS);
        ggml_kv_stream_partial_batch part, accumulator;
        part.rows = 4; part.parts = 2; part.width = 256;
        part.numerator.resize(layout.partial.elements); part.meta.resize(layout.partial.entries);
        accumulator = part;
        t.assert_true(get()->clear(backend.get(),input.output,wb,false));
        for (size_t step = 0; step < 37; ++step) {
            for (size_t row = 0; row < 4; ++row) for (size_t p = 0; p < 2; ++p) {
                const size_t index = row*2+p;
                const float mass = step%5 == 0 ? 0 : float(step+p+1)/8;
                part.meta[index] = {float(int(step*17%201)-100),mass};
                for (size_t c = 0; c < 256; ++c) part.numerator[index*256+c] = mass*.25f*std::sin(float(c+step+row+p));
            }
            ggml_backend_tensor_set(raw,part.numerator.data(),layout.second_offset,layout.partial.numerator_bytes);
            ggml_backend_tensor_set(raw,part.meta.data(),layout.second_offset+layout.partial.meta_offset,layout.partial.meta_bytes);
            t.assert_true(ggml_kv_stream_partial_merge({accumulator.view(),part.view()},accumulator).status == ggml_kv_stream_partial_status::success);
            if (!t.assert_true(get()->fold(backend.get(),input.output,wb))) return;
            auto untouched = input.read();
            t.assert_true(std::all_of(untouched.begin(),untouched.end(),[](float x){return x == -77;}));
        }
        t.assert_true(get()->clear(backend.get(),input.output,wb,true));
        t.assert_true(get()->merge(backend.get(),input.output,wb));
        ggml_kv_stream_partial_value expected;
        t.assert_true(ggml_kv_stream_partial_normalize(accumulator.view(),expected).status == ggml_kv_stream_partial_status::success);
        close_values(t,expected.value,input.read(),2e-6f);
        // A representable unnormalized accumulator may have an overflowing intermediate quotient.
        // A later contribution can make final normalization representable; fold must not normalize early.
        std::fill(part.numerator.begin(),part.numerator.end(),1);
        std::fill(part.meta.begin(),part.meta.end(),ggml_kv_stream_partial_meta{0,std::numeric_limits<float>::denorm_min()});
        t.assert_true(get()->clear(backend.get(),input.output,wb,false));
        ggml_backend_tensor_set(raw,part.numerator.data(),layout.second_offset,layout.partial.numerator_bytes);
        ggml_backend_tensor_set(raw,part.meta.data(),layout.second_offset+layout.partial.meta_offset,layout.partial.meta_bytes);
        t.assert_true(get()->fold(backend.get(),input.output,wb));
        std::fill(part.numerator.begin(),part.numerator.end(),0);
        std::fill(part.meta.begin(),part.meta.end(),ggml_kv_stream_partial_meta{0,1});
        ggml_backend_tensor_set(raw,part.numerator.data(),layout.second_offset,layout.partial.numerator_bytes);
        ggml_backend_tensor_set(raw,part.meta.data(),layout.second_offset+layout.partial.meta_offset,layout.partial.meta_bytes);
        t.assert_true(get()->merge(backend.get(),input.output,wb));
        close_values(t,std::vector<float>(1024,1),input.read(),1e-6f);
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
