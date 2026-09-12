#pragma once

#include "../src/llama-kv-stream-resident.h"
#include "../src/llama-memory-executor-cuda.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <stdexcept>

#include <cmath>
#include <cstring>
#include <limits>

using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

// Encode deterministic token-major data; the independent oracle reads back the actual quantized values.
static void populate(const std::shared_ptr<llama_kv_stream_host> & host) {
    const auto & c = host->config();
    const size_t rows = host->layout().tokens*size_t(c.shape.heads);
    for (uint32_t layer = 0; layer < c.layers; ++layer) {
        llama_kv_stream_host_layer planes; GGML_ASSERT(host->layer(layer, planes));
        for (int side = 0; side < 2; ++side) {
            auto type = ggml_type(side ? c.shape.type_v : c.shape.type_k);
            const size_t dim = size_t(side ? c.shape.head_dim_v : c.shape.head_dim_k);
            std::vector<float> data(rows*dim);
            for (size_t i = 0; i < data.size(); ++i) data[i] = .2f*std::sin(float(i % 401 + 53*layer + 19*side)*.13f);
            ggml_quantize_chunk(type, data.data(), side ? planes.v : planes.k, 0, int64_t(rows), int64_t(dim), nullptr);
        }
    }
}

struct fixture {
    ggml_backend_t backend;
    bool cuda;
    std::shared_ptr<llama_kv_stream_host> host;
    std::shared_ptr<llama_kv_stream_content> content;
    llama_kv_stream_policy_config policy;
    arena_ptr arena{nullptr, ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};
    std::unique_ptr<llama_kv_stream_binding> binding;
    llama_kv_stream_resident * resident = nullptr;
    // The unused ring remains at the front; active resident planes must use policy offsets, not a flat host copy.
    fixture(ggml_backend_t backend, bool cuda, ggml_type k = GGML_TYPE_F16, ggml_type v = GGML_TYPE_F16, size_t context = 769) : backend(backend), cuda(cuda) {
        llama_kv_stream_host_config c{37, {k, v, 256, 256, 2, 256, 128},
            {{k, true, true, true, true}, {v, true, true, true, true}, true, true}, context, 2};
        auto * dev = ggml_backend_get_device(backend);
        auto * host_type = cuda ? llama_kv_stream_host_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        auto * device_type = cuda ? llama_kv_stream_device_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        GGML_ASSERT(host_type && device_type);
        host = llama_kv_stream_host::create(c, host_type);
        GGML_ASSERT(host);
        populate(host);
        content = std::make_shared<llama_kv_stream_content>(host);
        ggml_kv_stream_layout page; GGML_ASSERT(ggml_kv_stream_layout_make(c.shape, 256, page).status == ggml_kv_stream_status::success);
        policy.shape = c.shape; policy.capabilities = c.capabilities; policy.layers = c.layers; policy.pool_bytes = page.bytes*16;
        arena.reset(ggml_backend_memory_arena_new(device_type, policy.pool_bytes + 256));
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena.get(), 0));
        auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(arena.get())));
        const size_t offset = 128 + (128 - base % 128) % 128;
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 9, offset, policy.pool_bytes, 64, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(), 9));
        binding = std::make_unique<llama_kv_stream_binding>(c.cache_id, device_type);
    }
    bool attach(const llama_kv_stream_policy_state * placement = nullptr) {
        return binding->bind(lease.get(), policy, [&](const auto & view) {
            auto result = llama_kv_stream_resident::create(view, content, backend, placement);
            resident = result.get();
            return result;
        });
    }
};

// Decode exact GGML rows, without assuming that quantized values equal their original float inputs.
static std::vector<float> unpack(const fixture & f, uint32_t layer, bool value) {
    const auto & shape = f.host->config().shape;
    const auto type = ggml_type(value ? shape.type_v : shape.type_k);
    const size_t dim = size_t(value ? shape.head_dim_v : shape.head_dim_k);
    const size_t rows = f.host->layout().tokens*size_t(shape.heads);
    const size_t stride = ggml_row_size(type, int64_t(dim));
    llama_kv_stream_host_layer planes; f.host->layer(layer, planes);
    auto * source = static_cast<const uint8_t *>(value ? planes.v : planes.k);
    std::vector<float> result(rows*dim);
    auto convert = ggml_get_type_traits(type)->to_float;
    GGML_ASSERT(convert);
    for (size_t row = 0; row < rows; ++row) convert(source + row*stride, result.data() + row*dim, int64_t(dim));
    return result;
}

// Scalar causal softmax oracle, independent of GGML graph strides and attention dispatch.
static std::vector<float> oracle(const fixture & f, uint32_t layer, size_t active, size_t queries, const std::vector<float> & q) {
    constexpr size_t dim = 256, heads = 4, kv_heads = 2;
    auto k = unpack(f, layer, false), v = unpack(f, layer, true);
    std::vector<float> result(dim*heads*queries);
    for (size_t h = 0; h < heads; ++h) for (size_t query = 0; query < queries; ++query) {
        const size_t visible = active - queries + query + 1;
        std::vector<double> scores(visible);
        double peak = -INFINITY, sum = 0;
        for (size_t token = 0; token < visible; ++token) {
            double dot = 0;
            for (size_t x = 0; x < dim; ++x) dot += double(q[(h*queries + query)*dim + x])*k[(token*kv_heads + h/2)*dim + x];
            scores[token] = dot/16;
            peak = std::max(peak, scores[token]);
        }
        for (auto & score : scores) { score = std::exp(score - peak); sum += score; }
        for (size_t x = 0; x < dim; ++x) {
            double value = 0;
            for (size_t token = 0; token < visible; ++token) value += scores[token]*v[(token*kv_heads + h/2)*dim + x];
            result[(query*heads + h)*dim + x] = float(value/sum);
        }
    }
    return result;
}

// Compare ordinary head-major K/V allocation with leased token-major resident views on the same backend.
static bool evaluate(testing & t, fixture & f, bool resident, uint32_t layer, size_t active, size_t queries) {
    const size_t padded = (active + 255)/256*256;
    auto pin = resident ? f.binding->acquire() : llama_memory_execution{};
    if (resident && !pin) return t.assert_true(false);
    ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, true}));
    auto * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 256, int64_t(queries), 4, 1);
    auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, int64_t(padded), int64_t(queries));

    ggml_tensor * out = nullptr, * k = nullptr, * v = nullptr;
    if (resident) out = f.resident->attention(ctx.get(), layer, q, mask, active, 1.0f/16);
    else {
        k = ggml_new_tensor_3d(ctx.get(), ggml_type(f.policy.shape.type_k), 256, int64_t(padded), 2);
        v = ggml_new_tensor_3d(ctx.get(), ggml_type(f.policy.shape.type_v), 256, int64_t(padded), 2);
        out = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f/16, 0, 0);
    }
    if (!t.assert_true(out != nullptr)) return false;
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (!t.assert_true(ggml_backend_supports_op(f.backend, out))) return false;
    auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr work(ggml_backend_alloc_ctx_tensors(ctx.get(), f.backend));
    if (!t.assert_true(work != nullptr)) return false;
    std::vector<float> qdata(256*4*queries);
    for (size_t i = 0; i < qdata.size(); ++i) qdata[i] = .3f*std::cos(float(i % 211)*.07f);
    std::vector<ggml_fp16_t> mdata(padded*queries);
    for (size_t query = 0; query < queries; ++query) for (size_t token = 0; token < padded; ++token)
        mdata[query*padded + token] = ggml_fp32_to_fp16(token <= active - queries + query ? 0.0f : -INFINITY);
    ggml_backend_tensor_set(q, qdata.data(), 0, qdata.size()*sizeof(float));
    ggml_backend_tensor_set(mask, mdata.data(), 0, mdata.size()*sizeof(ggml_fp16_t));
    if (!resident) {
        llama_kv_stream_host_layer planes; f.host->layer(layer, planes);
        for (int side = 0; side < 2; ++side) {
            auto * tensor = side ? v : k;
            const size_t row = ggml_row_size(tensor->type, 256);
            std::vector<uint8_t> packed(row*padded*2);
            auto * src = static_cast<const uint8_t *>(side ? planes.v : planes.k);
            for (size_t h = 0; h < 2; ++h) for (size_t token = 0; token < padded; ++token)
                std::memcpy(packed.data() + (h*padded + token)*row, src + (token*2 + h)*row, row);
            ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size());
        }
    } else if (!t.assert_true(f.resident->synchronize(active) && f.resident->ready(active))) return false;
    if (f.cuda) {
        llama_memory_cuda_executor execution(f.backend);
        const std::vector<ggml_backend_memory_lease_t> deps = resident ? std::vector<ggml_backend_memory_lease_t>{f.lease.get()} : std::vector<ggml_backend_memory_lease_t>{};
        if (!t.assert_true(execution.bind(graph, deps, 1))) return false;
        for (int replay = 0; replay < 3; ++replay)
            if (!t.assert_true(execution.compute_async(deps, 1) == GGML_STATUS_SUCCESS && execution.drain())) return false;
    } else if (!t.assert_true(ggml_backend_graph_compute(f.backend, graph) == GGML_STATUS_SUCCESS)) return false;
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
    auto expected = oracle(f, layer, active, queries, qdata);
    float maximum = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i])) { t.assert_true(false); return false; }
        maximum = std::max(maximum, std::abs(result[i] - expected[i]));
    }
    t.out << "max absolute error = " << maximum << '\n';
    return t.assert_true(maximum < 1e-3f);
}
