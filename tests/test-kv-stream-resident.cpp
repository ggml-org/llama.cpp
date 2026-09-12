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
    fixture(ggml_backend_t backend, bool cuda, ggml_type k = GGML_TYPE_F16, ggml_type v = GGML_TYPE_F16) : backend(backend), cuda(cuda) {
        llama_kv_stream_host_config c{37, {k, v, 256, 256, 2, 256, 128},
            {{k, true, true, true, true}, {v, true, true, true, true}, true, true}, 769, 2};
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
    bool attach() {
        return binding->bind(lease.get(), policy, [&](const auto & view) {
            auto result = llama_kv_stream_resident::create(view, content, backend);
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

int main(int argc, char ** argv) {
    const bool cuda = argc > 1 && std::strcmp(argv[1], "--cuda") == 0;
    ggml_backend_ptr backend;
    if (cuda) { ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1; backend.reset(ggml_backend_dev_init(dev, nullptr)); }
    else backend.reset(ggml_backend_cpu_init());
    testing t;
    t.test("ordinary_attention_matches_scalar_reference", [&](testing & t) {
        fixture f(backend.get(), cuda);
        evaluate(t, f, false, 1, 257, 8);
    });
    t.test("resident_attention_matches_decode_and_prefill", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        for (size_t q : {size_t(1), size_t(8), size_t(33)}) evaluate(t, f, true, 1, 257, q);
        evaluate(t, f, true, 0, 769, 1); // padded keys exactly fill the four-page resident capacity
        t.assert_true(f.resident->synchronize(257));
        t.assert_equal(size_t(0), f.resident->last_upload_bytes());
    });
    t.test("resident_dirty_tail_changes_attention_without_rebinding", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        if (!evaluate(t, f, true, 1, 257, 1)) return;
        std::vector<float> row(512, 3.0f);
        std::vector<uint8_t> data(f.host->layout().v_token_bytes);
        ggml_quantize_chunk(GGML_TYPE_F16, row.data(), data.data(), 0, 2, 256, nullptr);
        llama_kv_stream_write write;
        t.assert_true(f.content->prepare({{1, ggml_kv_stream_operand::v, 256*data.size(), data.data(), data.size()}}, write));
        t.assert_true(f.content->commit(write));
        t.assert_true(!f.resident->ready(257));
        if (!evaluate(t, f, true, 1, 257, 1)) return;
        t.assert_equal(data.size(), f.resident->last_upload_bytes());
        t.assert_equal(size_t(1), f.resident->last_upload_calls());
    });
    t.test("invalid_metadata_and_streaming_requirement_are_rejected", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        t.assert_true(!f.resident->synchronize(0));
        t.assert_true(!f.resident->synchronize(770));
        ggml_context_ptr ctx(ggml_init({16384, nullptr, true}));
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, 1, 4);
        t.assert_true(!f.resident->attention(ctx.get(), 0, q, nullptr, 257, 1.0f/16));
        t.assert_true(!f.resident->attention(ctx.get(), 9, q, nullptr, 256, 1.0f/16));
        t.assert_true(!f.resident->attention(ctx.get(), 0, q, nullptr, 256, NAN));
    });
    t.test("policy_offsets_and_flat_roots_preserve_ring_bytes", [&](testing & t) {
        fixture f(backend.get(), cuda, cuda ? GGML_TYPE_Q8_0 : GGML_TYPE_F16, cuda ? GGML_TYPE_Q4_0 : GGML_TYPE_F16);
        auto * buffer = ggml_backend_memory_lease_buffer(f.lease.get());
        ggml_backend_buffer_clear(buffer, 0xa5);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        ggml_context_ptr ctx(ggml_init({16384, nullptr, true}));
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, 1, 4);
        auto * out = f.resident->attention(ctx.get(), 1, q, nullptr, 256, 1.0f/16);
        if (!t.assert_true(out != nullptr)) return;
        llama_kv_stream_policy_layout layout;
        const auto & binding = *f.binding->view();
        t.assert_true(llama_kv_stream_policy_layout_make(f.policy, binding.initial_policy, 0, layout).status == llama_kv_stream_policy_status::success);
        const auto & entry = layout.layers[1];
        t.assert_true(out->src[1]->data == static_cast<char *>(binding.base) + entry.offset);
        t.assert_true(out->src[2]->data == static_cast<char *>(binding.base) + entry.offset + entry.planes.v_offset);
        for (int side = 1; side <= 2; ++side) {
            auto * root = out->src[side]->view_src;
            t.assert_equal(ggml_nbytes(root), ggml_backend_buffer_get_alloc_size(buffer, root));
            t.assert_equal(side == 1 ? entry.planes.k_token_bytes : entry.planes.v_token_bytes, out->src[side]->nb[1]);
            t.assert_equal(side == 1 ? entry.planes.k_row_bytes : entry.planes.v_row_bytes, out->src[side]->nb[2]);
        }
        t.assert_true(f.resident->synchronize(256));
        auto * marker = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 128);
        t.assert_true(ggml_backend_tensor_alloc(buffer, marker, binding.base) == GGML_STATUS_SUCCESS);
        std::vector<uint8_t> bytes(128);
        ggml_backend_tensor_get(marker, bytes.data(), 0, bytes.size());
        t.assert_true(std::all_of(bytes.begin(), bytes.end(), [](uint8_t x) { return x == 0xa5; }));
    });
    t.test("insufficient_residency_and_conversion_fallback_are_not_dispatched", [&](testing & t) {
        fixture small(backend.get(), cuda);
        small.policy.pool_bytes = small.policy.pool_bytes/16*10;
        if (!t.assert_true(small.attach())) return;
        t.assert_true(small.resident->synchronize(256));
        t.assert_true(!small.resident->synchronize(257) && !small.resident->ready(256));
        fixture conversion(backend.get(), cuda);
        conversion.policy.capabilities.direct_pair = false;
        t.assert_true(!conversion.attach());
    });
    t.test("malformed_q_and_mask_reject_before_ggml_assertions", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
        for (int variant = 0; variant < 5; ++variant) {
            auto * q = ggml_new_tensor_4d(ctx.get(), variant == 0 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                variant == 1 ? 128 : 256, 8, variant == 2 ? 3 : 4, variant == 3 ? 2 : 1);
            auto * mask = ggml_new_tensor_2d(ctx.get(), variant == 4 ? GGML_TYPE_F32 : GGML_TYPE_F16, 256, 8);
            t.assert_true(f.resident->attention(ctx.get(), 0, q, mask, 256, 1.0f/16) == nullptr);
        }
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, 8, 4);
        auto * short_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 128, 8);
        t.assert_true(f.resident->attention(ctx.get(), 0, q, short_mask, 256, 1.0f/16) == nullptr);
    });
    t.test("copy_exception_stays_closed_and_can_retry", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        auto * buffer = ggml_backend_memory_lease_buffer(f.lease.get());
        const auto original = buffer->iface.set_tensor;
        buffer->iface.set_tensor = [](ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) { throw std::runtime_error("copy failure"); };
        bool caught = false;
        try { f.resident->synchronize(257); } catch (const std::runtime_error &) { caught = true; }
        buffer->iface.set_tensor = original;
        t.assert_true(caught && !f.resident->ready(257));
        t.assert_true(f.resident->synchronize(257) && f.resident->ready(257));
        t.assert_equal(size_t(512)*2*(f.host->layout().k_token_bytes + f.host->layout().v_token_bytes), f.resident->last_upload_bytes());
        evaluate(t, f, true, 0, 257, 1);
    });
    t.test("shorter_replacement_refreshes_same_binding", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        if (!evaluate(t, f, true, 0, 257, 1)) return;
        const auto arena_generation = ggml_backend_memory_arena_generation(f.arena.get());
        const auto binding_revision = f.binding->view()->revision;
        auto c = f.host->config();
        auto replacement = llama_kv_stream_host::create(c, ggml_backend_buffer_get_type(f.host->buffer()));
        populate(replacement);
        llama_kv_stream_host_layer plane; replacement->layer(0, plane);
        std::vector<float> row(512, 1.5f);
        ggml_quantize_chunk(GGML_TYPE_F16, row.data(), plane.v, 0, 2, 256, nullptr);
        f.host = replacement;
        t.assert_true(f.content->replace(replacement) && !f.resident->ready(257));
        if (!evaluate(t, f, true, 0, 129, 8)) return;
        t.assert_equal(arena_generation, ggml_backend_memory_arena_generation(f.arena.get()));
        t.assert_equal(binding_revision, f.binding->view()->revision);
        t.assert_true(f.content->reset_mirror() && !f.resident->ready(129));
        t.assert_true(f.resident->synchronize(129));
        t.assert_equal(size_t(256)*2*(f.host->layout().k_token_bytes + f.host->layout().v_token_bytes), f.resident->last_upload_bytes());
    });
    t.test("factory_and_replaced_geometry_are_validated", [&](testing & t) {
        fixture f(backend.get(), cuda);
        t.assert_true(!f.binding->bind(f.lease.get(), f.policy, [&](const auto & view) -> std::unique_ptr<llama_memory_executable> {
            auto wrong = view; ++wrong.cache_id;
            t.assert_true(llama_kv_stream_resident::create(wrong, f.content, backend.get()) == nullptr);
            wrong = view; --wrong.capacity;
            t.assert_true(llama_kv_stream_resident::create(wrong, f.content, backend.get()) == nullptr);
            t.assert_true(llama_kv_stream_resident::create(view, nullptr, backend.get()) == nullptr);
            return {};
        }));
        if (!t.assert_true(f.attach())) return;
        t.assert_true(f.resident->synchronize(256));
        auto c = f.host->config();
        c.shape.type_v = GGML_TYPE_Q4_0;
        c.capabilities.v.type = GGML_TYPE_Q4_0;
        auto replacement = llama_kv_stream_host::create(c, ggml_backend_buffer_get_type(f.host->buffer()));
        if (!t.assert_true(replacement != nullptr)) return;
        t.assert_true(f.content->replace(replacement));
        t.assert_true(!f.resident->synchronize(256) && !f.resident->ready(256));
    });
    t.test("external_graph_pin_blocks_resident_destruction", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        struct completion : llama_memory_executor_backend {
            ggml_backend_t backend;
            explicit completion(ggml_backend_t b) : backend(b) {}
            bool drain() override { ggml_backend_synchronize(backend); return true; }
        } done(backend.get());
        f.lease.reset();
        t.assert_true(f.binding->detach(done).status == llama_memory_executor_status::pending);
        t.assert_true(ggml_backend_memory_arena_lease_count(f.arena.get()) != 0);
        pin.reset();
        t.assert_true(f.binding->detach(done).status == llama_memory_executor_status::retired);
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.arena.get()));
    });
    t.test("wide_prefill_uses_ordinary_attention", [&](testing & t) {
        fixture f(backend.get(), cuda);
        if (!t.assert_true(f.attach())) return;
        evaluate(t, f, true, 0, 257, 257);
    });
    if (cuda) t.test("mixed_quant_resident_and_ordinary_paths_agree", [&](testing & t) {
        for (auto pair : {std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}, std::pair{GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}, std::pair{GGML_TYPE_Q5_1, GGML_TYPE_Q4_1}}) {
            fixture f(backend.get(), cuda, pair.first, pair.second);
            if (!t.assert_true(f.attach())) continue;
            evaluate(t, f, false, 0, 257, 1);
            evaluate(t, f, true, 0, 257, 1);
            evaluate(t, f, true, 1, 257, 8);
        }
    });
    return t.summary();
}
