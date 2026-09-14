#pragma once
#include "kv-stream-resident-test.h"
#include "../ggml/src/ggml-kv-stream-device.h"

struct block_workspace {
    arena_ptr arena{nullptr, ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};
    // Exercise an exact-sized lease at a nonzero parent offset.
    block_workspace(fixture & f, size_t bytes, uint64_t id = 19) {
        auto * type = f.cuda ? llama_kv_stream_device_buffer_type(ggml_backend_get_device(f.backend)) : ggml_backend_cpu_buffer_type();
        arena.reset(ggml_backend_memory_arena_new(type, bytes + 256));
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena.get(), 0));
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(arena.get())));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), id, 128 + (128-base%128)%128, bytes, 64, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(), id));
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
