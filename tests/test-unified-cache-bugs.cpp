// Unit tests for unified cache bug fixes (evict accounting, realloc failure handling, unaligned hash)
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-unified-cache-bugs

#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sycl/sycl.hpp>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

static bool test_evict_returns_bytes(sycl::queue & q) {
    printf("\n=== Test: evict() returns bytes freed ===\n");

    ggml_sycl::unified_cache cache(q, 4 * 1024);
    std::vector<uint8_t>     data_a(512, 0x11);
    std::vector<uint8_t>     data_b(512, 0x22);
    ggml_sycl_cache_id       key_a = ggml_sycl::test_make_cache_id(data_a.data());
    ggml_sycl_cache_id       key_b = ggml_sycl::test_make_cache_id(data_b.data());

    bool   needs_fill = false;
    void * ptr_a      = cache.ensure_cached_alloc(key_a, data_a.data(), data_a.size(), data_a.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);
    void * ptr_b      = cache.ensure_cached_alloc(key_b, data_b.data(), data_b.size(), data_b.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);

    if (!ptr_a || !ptr_b) {
        fprintf(stderr, "Failed to allocate cache entries for evict test\n");
        return false;
    }

    size_t freed = cache.evict(data_a.size());
    if (freed == 0) {
        fprintf(stderr, "evict() returned 0 bytes freed\n");
        return false;
    }

    printf("evict() freed %zu bytes\n", freed);
    return true;
}

static bool test_realloc_failure_keeps_entry(sycl::queue & q) {
    printf("\n=== Test: realloc failure preserves existing entry ===\n");

    const size_t             budget = 1ULL << 41;  // 2 TB budget to avoid budget gating
    ggml_sycl::unified_cache cache(q, budget);

    std::vector<uint8_t> data(256, 0x33);
    ggml_sycl_cache_id   key       = ggml_sycl::test_make_cache_id(data.data());
    const size_t         orig_size = data.size();

    bool   needs_fill = false;
    void * ptr =
        cache.ensure_cached_alloc(key, data.data(), orig_size, orig_size, ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                                  -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);
    if (!ptr) {
        fprintf(stderr, "Failed to allocate initial cache entry\n");
        return false;
    }

    const size_t used_before = cache.used();
    const size_t huge_alloc  = 1ULL << 40;  // 1 TB, should fail on all current devices

    void * realloc_ptr = cache.ensure_cached_alloc(key, data.data(), orig_size, huge_alloc,
                                                   ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS,
                                                   false, &needs_fill);
    if (realloc_ptr) {
        fprintf(stderr, "Unexpectedly succeeded in huge realloc\n");
        return false;
    }

    if (!cache.is_cached(key, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Existing entry was dropped after realloc failure\n");
        return false;
    }

    if (cache.used() != used_before) {
        fprintf(stderr, "Cache used() changed after realloc failure (before=%zu after=%zu)\n", used_before,
                cache.used());
        return false;
    }

    return true;
}

static bool test_realloc_eviction_failure_keeps_entry(sycl::queue & q) {
    printf("\n=== Test: realloc eviction failure preserves existing entry ===\n");

    ggml_sycl::unified_cache cache(q, 1024);
    std::vector<uint8_t>     data(512, 0x44);
    ggml_sycl_cache_id       key = ggml_sycl::test_make_cache_id(data.data());

    bool   needs_fill = false;
    void * ptr        = cache.ensure_cached_alloc(key, data.data(), data.size(), data.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);
    if (!ptr) {
        fprintf(stderr, "Failed to allocate initial cache entry for eviction test\n");
        return false;
    }

    cache.pin(key, GGML_LAYOUT_AOS);
    const size_t used_before = cache.used();

    void * realloc_ptr =
        cache.ensure_cached_alloc(key, data.data(), data.size(), 2048, ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                                  -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);

    if (realloc_ptr) {
        fprintf(stderr, "Unexpectedly succeeded in realloc with eviction failure\n");
        return false;
    }

    if (!cache.is_cached(key, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Entry dropped after eviction failure during realloc\n");
        return false;
    }

    if (cache.used() != used_before) {
        fprintf(stderr, "Cache used() changed after eviction failure (before=%zu after=%zu)\n", used_before,
                cache.used());
        return false;
    }

    return true;
}

static bool test_layout_size_mismatch_recaches(sycl::queue & q) {
    printf("\n=== Test: layout size mismatch recache ===\n");

    ggml_sycl::unified_cache cache(q, 4096);
    std::vector<uint8_t>     data(128, 0xad);

    ggml_sycl::cache_layout_request req{};
    req.key       = ggml_sycl::test_make_cache_id(data.data());
    req.src_ptr   = data.data();
    req.src_size  = data.size();
    req.dst_size  = data.size();
    req.type      = ggml_sycl::cache_entry_type::DENSE_WEIGHT;
    req.layer_id  = -1;
    req.expert_id = -1;
    req.layout    = GGML_LAYOUT_AOS;

    auto result = cache.ensure_cached_layout(req, {});
    if (result.status == ggml_sycl::cache_layout_status::IN_PROGRESS) {
        result.event.wait();
    }
    if (result.status != ggml_sycl::cache_layout_status::READY || !result.device_ptr) {
        fprintf(stderr, "Failed to cache initial layout for size mismatch test\n");
        return false;
    }

    req.dst_size = data.size() * 2;
    result       = cache.ensure_cached_layout(req, {});
    if (result.status == ggml_sycl::cache_layout_status::IN_PROGRESS) {
        result.event.wait();
    }
    if (result.status != ggml_sycl::cache_layout_status::READY || !result.device_ptr ||
        result.size != req.dst_size) {
        fprintf(stderr, "Size mismatch recache did not succeed\n");
        return false;
    }

    return true;
}

static bool test_layout_size_mismatch_pinned_fails(sycl::queue & q) {
    printf("\n=== Test: layout size mismatch pinned entry fails ===\n");

    ggml_sycl::unified_cache cache(q, 4096);
    std::vector<uint8_t>     data(128, 0xbe);

    ggml_sycl::cache_layout_request req{};
    req.key       = ggml_sycl::test_make_cache_id(data.data());
    req.src_ptr   = data.data();
    req.src_size  = data.size();
    req.dst_size  = data.size();
    req.type      = ggml_sycl::cache_entry_type::DENSE_WEIGHT;
    req.layer_id  = -1;
    req.expert_id = -1;
    req.layout    = GGML_LAYOUT_AOS;

    auto result = cache.ensure_cached_layout(req, {});
    if (result.status == ggml_sycl::cache_layout_status::IN_PROGRESS) {
        result.event.wait();
    }
    if (result.status != ggml_sycl::cache_layout_status::READY || !result.device_ptr) {
        fprintf(stderr, "Failed to cache initial layout for pinned mismatch test\n");
        return false;
    }

    cache.pin(req.key, req.layout);
    req.dst_size = data.size() * 2;
    result       = cache.ensure_cached_layout(req, {});
    if (result.status != ggml_sycl::cache_layout_status::FAILED) {
        fprintf(stderr, "Pinned size mismatch did not fail as expected (status=%d)\n", (int) result.status);
        return false;
    }

    return true;
}

static bool alloc_tensor_buffer(ggml_backend_buffer_type_t buft,
                                ggml_tensor *             tensor,
                                ggml_backend_buffer_usage usage,
                                std::vector<ggml_backend_buffer_t> & buffers) {
    if (tensor->view_src) {
        return ggml_backend_view_init(tensor) == GGML_STATUS_SUCCESS;
    }
    const size_t size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return false;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    buffers.push_back(buffer);
    return true;
}

static bool test_graph_pins_host_weights() {
    printf("\n=== Test: graph pins host MoE weights ===\n");

#if !defined(GGML_SYCL_GRAPH)
    printf("SKIP: GGML_SYCL_GRAPH not enabled\n");
    return true;
#else
    setenv("GGML_SYCL_DISABLE_GRAPH", "0", 1);

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        printf("SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    if (!ggml_sycl::test_backend_supports_graphs(sycl_backend)) {
        printf("SKIP: SYCL graphs not supported on this device\n");
        ggml_backend_free(sycl_backend);
        return true;
    }

    if (ggml_sycl::test_backend_graphs_disabled(sycl_backend)) {
        printf("SKIP: SYCL graphs disabled in backend\n");
        ggml_backend_free(sycl_backend);
        return true;
    }

    const ggml_init_params params = {
        64 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(sycl_backend);
        return false;
    }

    const int vocab     = 64;
    const int n_embd    = 256;
    const int n_tokens  = 4;
    const int n_used    = 4;
    const int n_experts = 8;
    const int out_dim   = 128;

    ggml_tensor * tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, vocab);
    ggml_set_name(tok_embd, "tok_embd.weight");
    ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token_ids, "token_ids");
    ggml_tensor * embd = ggml_get_rows(ctx, tok_embd, token_ids);
    ggml_set_name(embd, "embd");
    ggml_tensor * norm = ggml_rms_norm(ctx, embd, 1e-5f);
    ggml_set_name(norm, "rms_norm");
    ggml_tensor * norm3d = ggml_reshape_3d(ctx, norm, n_embd, 1, n_tokens);
    ggml_set_name(norm3d, "rms_norm_3d");
    ggml_tensor * moe_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_name(moe_ids, "moe_ids");
    ggml_tensor * moe_w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, n_embd, out_dim, n_experts);
    ggml_set_name(moe_w, "moe_weights");
    ggml_tensor * moe_out = ggml_mul_mat_id(ctx, moe_w, norm3d, moe_ids);
    ggml_set_name(moe_out, "moe_out");

    ggml_tensor * add1  = ggml_add(ctx, moe_out, moe_out);
    ggml_tensor * add2  = ggml_add(ctx, add1, moe_out);
    ggml_tensor * scale = ggml_scale(ctx, add2, 0.5f);
    ggml_tensor * silu  = ggml_silu(ctx, scale);
    ggml_tensor * add3  = ggml_add(ctx, silu, scale);
    ggml_tensor * out   = ggml_cont(ctx, add3);

    ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(sycl_backend);
    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();

    std::vector<ggml_backend_buffer_t> buffers;
    if (!alloc_tensor_buffer(host_buft, tok_embd, GGML_BACKEND_BUFFER_USAGE_WEIGHTS, buffers) ||
        !alloc_tensor_buffer(dev_buft, token_ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(dev_buft, embd, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(dev_buft, norm, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(dev_buft, norm3d, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(dev_buft, moe_ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(host_buft, moe_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS, buffers) ||
        !alloc_tensor_buffer(dev_buft, moe_out, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(dev_buft, add1, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(dev_buft, add2, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(dev_buft, scale, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(dev_buft, silu, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(dev_buft, add3, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers) ||
        !alloc_tensor_buffer(dev_buft, out, GGML_BACKEND_BUFFER_USAGE_COMPUTE, buffers)) {
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(sycl_backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, tok_embd);
        ggml_backend_sycl_register_host_weight_tensor(dev, moe_w);
    }

    std::vector<float> tok_embd_data(static_cast<size_t>(vocab) * n_embd, 0.25f);
    std::vector<int32_t> token_ids_data(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        token_ids_data[i] = i % vocab;
    }
    std::vector<int32_t> moe_ids_data(static_cast<size_t>(n_used) * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < n_used; ++i) {
            moe_ids_data[t * n_used + i] = (t + i) % n_experts;
        }
    }
    std::vector<uint8_t> moe_w_data(ggml_nbytes(moe_w), 0x11);

    ggml_backend_tensor_set(tok_embd, tok_embd_data.data(), 0, tok_embd_data.size() * sizeof(float));
    ggml_backend_tensor_set(token_ids, token_ids_data.data(), 0, token_ids_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(moe_ids, moe_ids_data.data(), 0, moe_ids_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(moe_w, moe_w_data.data(), 0, moe_w_data.size());

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_status status = ggml_backend_graph_compute(sycl_backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: graph compute failed\n");
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    const size_t pinned = ggml_sycl::test_graph_pinned_entry_count(sycl_backend);
    if (pinned == 0) {
        fprintf(stderr, "FAIL: graph pinned entries empty (expected > 0)\n");
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    for (ggml_backend_buffer_t buf : buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(sycl_backend);
    return true;
#endif
}

class stream_dma_noop_kernel;

static sycl::event stream_dma_noop(sycl::queue & q,
                                   void *,
                                   size_t,
                                   size_t,
                                   const void *,
                                   const std::vector<sycl::event> & deps) {
    return q.submit([&](sycl::handler & cgh) {
        cgh.depends_on(deps);
        cgh.single_task<stream_dma_noop_kernel>([]() {});
    });
}

static bool test_stream_dma_mmap_fail(sycl::queue & q) {
    printf("\n=== Test: stream_dma mmap failure flag ===\n");

    ggml_sycl::unified_cache cache(q, 1024 * 1024);
    std::vector<uint8_t>     data(256, 0x5a);

    ggml_sycl::cache_ptr_view view{};
    view.ptr      = data.data();
    view.size     = data.size();
    view.layout   = GGML_LAYOUT_AOS;
    view.type     = ggml_sycl::cache_entry_type::DENSE_WEIGHT;
    view.location = ggml_sycl::cache_location::HOST_MMAP;

    setenv("GGML_SYCL_TEST_DMA_FAIL", "1", 1);
    auto result = cache.stream_dma(view, data.size(), 64, 1, stream_dma_noop, nullptr, {});
    unsetenv("GGML_SYCL_TEST_DMA_FAIL");

    if (result.ok) {
        fprintf(stderr, "Expected DMA failure but result.ok was true\n");
        return false;
    }
    if (!result.used_mmap_direct || !result.mmap_direct_failed) {
        fprintf(stderr, "Expected mmap failure flags (used=%d failed=%d)\n",
                result.used_mmap_direct ? 1 : 0,
                result.mmap_direct_failed ? 1 : 0);
        return false;
    }

    return true;
}

static bool test_all_pinned_eviction_failure_new_entry(sycl::queue & q) {
    printf("\n=== Test: all-pinned eviction failure on new entry ===\n");

    ggml_sycl::unified_cache cache(q, 1024);
    std::vector<uint8_t>     data_a(512, 0x55);
    std::vector<uint8_t>     data_b(512, 0x66);
    std::vector<uint8_t>     data_c(512, 0x77);

    bool   needs_fill = false;
    void * ptr_a      = cache.ensure_cached_alloc(ggml_sycl::test_make_cache_id(data_a.data()),
                                                  data_a.data(),
                                                  data_a.size(),
                                                  data_a.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);
    void * ptr_b      = cache.ensure_cached_alloc(ggml_sycl::test_make_cache_id(data_b.data()),
                                                  data_b.data(),
                                                  data_b.size(),
                                                  data_b.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);

    if (!ptr_a || !ptr_b) {
        fprintf(stderr, "Failed to allocate pinned entries for eviction test\n");
        return false;
    }

    cache.pin(ggml_sycl::test_make_cache_id(data_a.data()), GGML_LAYOUT_AOS);
    cache.pin(ggml_sycl::test_make_cache_id(data_b.data()), GGML_LAYOUT_AOS);
    const size_t used_before = cache.used();

    void * ptr_c = cache.ensure_cached_alloc(ggml_sycl::test_make_cache_id(data_c.data()),
                                             data_c.data(),
                                             data_c.size(),
                                             data_c.size(),
                                             ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                             &needs_fill);

    if (ptr_c) {
        fprintf(stderr, "Unexpectedly succeeded allocating with all entries pinned\n");
        return false;
    }

    if (!cache.is_cached(ggml_sycl::test_make_cache_id(data_a.data()), GGML_LAYOUT_AOS) ||
        !cache.is_cached(ggml_sycl::test_make_cache_id(data_b.data()), GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Pinned entries were evicted unexpectedly\n");
        return false;
    }

    if (cache.used() != used_before) {
        fprintf(stderr, "Cache used() changed after all-pinned eviction failure (before=%zu after=%zu)\n", used_before,
                cache.used());
        return false;
    }

    return true;
}

static bool test_partial_eviction_insufficient(sycl::queue & q) {
    printf("\n=== Test: partial eviction insufficient for new entry ===\n");

    ggml_sycl::unified_cache cache(q, 1024);
    std::vector<uint8_t>     data_a(512, 0x88);
    std::vector<uint8_t>     data_b(512, 0x99);
    std::vector<uint8_t>     data_c(1024, 0xaa);

    bool   needs_fill = false;
    void * ptr_a      = cache.ensure_cached_alloc(ggml_sycl::test_make_cache_id(data_a.data()),
                                                  data_a.data(),
                                                  data_a.size(),
                                                  data_a.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);
    void * ptr_b      = cache.ensure_cached_alloc(ggml_sycl::test_make_cache_id(data_b.data()),
                                                  data_b.data(),
                                                  data_b.size(),
                                                  data_b.size(),
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);

    if (!ptr_a || !ptr_b) {
        fprintf(stderr, "Failed to allocate initial entries for partial eviction test\n");
        return false;
    }

    cache.pin(ggml_sycl::test_make_cache_id(data_a.data()), GGML_LAYOUT_AOS);
    const size_t used_before = cache.used();

    void * ptr_c = cache.ensure_cached_alloc(ggml_sycl::test_make_cache_id(data_c.data()),
                                             data_c.data(),
                                             data_c.size(),
                                             data_c.size(),
                                             ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                             &needs_fill);

    if (ptr_c) {
        fprintf(stderr, "Unexpectedly succeeded allocating with insufficient eviction\n");
        return false;
    }

    if (!cache.is_cached(ggml_sycl::test_make_cache_id(data_a.data()), GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Pinned entry was evicted during partial eviction test\n");
        return false;
    }

    // Eviction is deferred; drain the queue and process deferred frees before checking accounting.
    q.wait();
    cache.evict(0);

    if (cache.used() >= used_before) {
        fprintf(stderr, "Cache used() did not drop after partial eviction (before=%zu after=%zu)\n", used_before,
                cache.used());
        return false;
    }

    return true;
}

static bool test_allocation_failure_new_entry(sycl::queue & q) {
    printf("\n=== Test: allocation failure on new entry ===\n");

    const size_t             budget = std::numeric_limits<size_t>::max() / 2;
    ggml_sycl::unified_cache cache(q, budget);

    std::vector<uint8_t> data(256, 0xbb);
    const size_t         huge_alloc = 1ULL << 40;

    bool   needs_fill = false;
    void * ptr        = cache.ensure_cached_alloc(ggml_sycl::test_make_cache_id(data.data()),
                                                  data.data(),
                                                  data.size(),
                                                  huge_alloc,
                                                  ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                                  &needs_fill);

    if (ptr) {
        fprintf(stderr, "Unexpectedly succeeded allocating huge entry\n");
        return false;
    }

    if (cache.is_cached(ggml_sycl::test_make_cache_id(data.data()), GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Cache entry created despite allocation failure\n");
        return false;
    }

    if (cache.used() != 0) {
        fprintf(stderr, "Cache used() changed after allocation failure (used=%zu)\n", cache.used());
        return false;
    }

    return true;
}

static bool test_deferred_free_stress(sycl::queue & q) {
    printf("\n=== Test: deferred free stress ===\n");

    ggml_sycl::unified_cache          cache(q, 64 * 1024);
    std::vector<std::vector<uint8_t>> payloads(32, std::vector<uint8_t>(512, 0xcc));

    bool needs_fill = false;
    for (auto & payload : payloads) {
        void * ptr = cache.ensure_cached_alloc(ggml_sycl::test_make_cache_id(payload.data()),
                                               payload.data(),
                                               payload.size(),
                                               payload.size(),
                                               ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS,
                                               false, &needs_fill);
        if (!ptr) {
            fprintf(stderr, "Failed to allocate entry during deferred free stress\n");
            return false;
        }
    }

    for (auto & payload : payloads) {
        cache.remove(ggml_sycl::test_make_cache_id(payload.data()),
                     ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                     -1,
                     -1,
                     GGML_LAYOUT_AOS);
    }

    q.wait();
    cache.evict(0);

    if (cache.used() != 0 || cache.dense_count() != 0) {
        fprintf(stderr, "Cache not fully freed after deferred free stress (used=%zu count=%zu)\n", cache.used(),
                cache.dense_count());
        return false;
    }

    return true;
}

static bool test_unaligned_hash(sycl::queue & q) {
    printf("\n=== Test: unaligned hash input ===\n");

    ggml_sycl::unified_cache cache(q, 2 * 1024 * 1024);

    std::vector<uint8_t> raw(129, 0x5a);
    uint8_t *            misaligned = raw.data() + 1;
    const size_t         size       = 127;

    void * ptr = cache.ensure_cached(ggml_sycl::test_make_cache_id(misaligned),
                                     misaligned,
                                     size,
                                     ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                                     -1,
                                     -1,
                                     GGML_LAYOUT_AOS,
                                     true);

    if (!ptr) {
        fprintf(stderr, "ensure_cached failed for misaligned input\n");
        return false;
    }

    if (!cache.is_cached(ggml_sycl::test_make_cache_id(misaligned), GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Cache entry missing for misaligned input\n");
        return false;
    }

    return true;
}

static bool test_unpin_experts(sycl::queue & q) {
    printf("\n=== Test: unpin_experts only affects MoE entries ===\n");

    ggml_sycl::unified_cache cache(q, 2048);
    std::vector<uint8_t>     dense(128, 0x5b);
    std::vector<uint8_t>     expert(128, 0x6c);

    bool needs_fill = false;
    if (!cache.ensure_cached_alloc(ggml_sycl::test_make_cache_id(dense.data()),
                                   dense.data(),
                                   dense.size(),
                                   dense.size(),
                                   ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false,
                                   &needs_fill)) {
        fprintf(stderr, "Failed to allocate dense entry for unpin test\n");
        return false;
    }
    if (!cache.ensure_cached_alloc(ggml_sycl::test_make_cache_id(expert.data()),
                                   expert.data(),
                                   expert.size(),
                                   expert.size(),
                                   ggml_sycl::cache_entry_type::MOE_EXPERT, 0, 0, GGML_LAYOUT_AOS, false,
                                   &needs_fill)) {
        fprintf(stderr, "Failed to allocate expert entry for unpin test\n");
        return false;
    }

    const ggml_sycl_cache_id dense_key  = ggml_sycl::test_make_cache_id(dense.data());
    const ggml_sycl_cache_id expert_key = ggml_sycl::test_make_cache_id(expert.data());
    cache.pin(dense_key, GGML_LAYOUT_AOS);
    cache.pin(expert_key, GGML_LAYOUT_AOS);
    cache.unpin_experts();

    if (!cache.is_pinned(dense_key, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Dense entry was unpinned unexpectedly\n");
        return false;
    }
    if (cache.is_pinned(expert_key, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "Expert entry remained pinned after unpin_experts\n");
        return false;
    }

    return true;
}

static bool test_moe_overcommit_cap(sycl::queue & q) {
    printf("\n=== Test: MoE overcommit cap ===\n");

    const size_t             budget = 1024;
    ggml_sycl::unified_cache cache(q, budget);

    std::vector<uint8_t>            data_a(budget, 0x1a);
    ggml_sycl::cache_layout_request req{};
    req.key              = ggml_sycl::test_make_cache_id(data_a.data());
    req.src_ptr          = data_a.data();
    req.src_size         = data_a.size();
    req.dst_size         = data_a.size();
    req.type             = ggml_sycl::cache_entry_type::MOE_EXPERT;
    req.layer_id         = 0;
    req.expert_id        = 0;
    req.layout           = GGML_LAYOUT_AOS;

    auto result = cache.ensure_cached_layout(req, {});
    if (result.status == ggml_sycl::cache_layout_status::FAILED || !result.device_ptr) {
        fprintf(stderr, "Failed to cache initial MoE entry\n");
        return false;
    }
    if (result.status == ggml_sycl::cache_layout_status::IN_PROGRESS) {
        result.event.wait();
    }
    cache.pin(req.key, GGML_LAYOUT_AOS);

    std::vector<uint8_t> data_b(40, 0x2b);
    req.key       = ggml_sycl::test_make_cache_id(data_b.data());
    req.src_ptr   = data_b.data();
    req.src_size  = data_b.size();
    req.dst_size  = data_b.size();
    req.expert_id = 1;
    result        = cache.ensure_cached_layout(req, {});
    if (result.status == ggml_sycl::cache_layout_status::FAILED || !result.device_ptr) {
        fprintf(stderr, "Failed to cache second MoE entry\n");
        return false;
    }
    if (!result.host_resident) {
        fprintf(stderr, "Expected host fallback for second MoE entry\n");
        return false;
    }
    if (result.status == ggml_sycl::cache_layout_status::IN_PROGRESS) {
        result.event.wait();
    }
    cache.pin(req.key, GGML_LAYOUT_AOS);

    std::vector<uint8_t> data_c(100, 0x3c);
    req.key       = ggml_sycl::test_make_cache_id(data_c.data());
    req.src_ptr   = data_c.data();
    req.src_size  = data_c.size();
    req.dst_size  = data_c.size();
    req.expert_id = 2;
    result        = cache.ensure_cached_layout(req, {});

    if (result.status == ggml_sycl::cache_layout_status::FAILED || !result.device_ptr) {
        fprintf(stderr, "Failed to cache third MoE entry\n");
        return false;
    }
    if (!result.host_resident) {
        fprintf(stderr, "Expected host fallback for third MoE entry\n");
        return false;
    }

    return true;
}

static bool test_single_layout_enforced(sycl::queue & q) {
    printf("\n=== Test: single layout enforcement ===\n");

    ggml_sycl::unified_cache cache(q, 1 << 20);
    std::vector<uint8_t>     data(512, 0x7a);

    ggml_sycl::cache_layout_request req{};
    req.key      = ggml_sycl::test_make_cache_id(data.data());
    req.src_ptr  = data.data();
    req.src_size = data.size();
    req.dst_size = data.size();
    req.type     = ggml_sycl::cache_entry_type::DENSE_WEIGHT;
    req.layout   = GGML_LAYOUT_AOS;

    auto result = cache.ensure_cached_layout(req, {});
    if (result.status != ggml_sycl::cache_layout_status::READY || !result.device_ptr) {
        fprintf(stderr, "Failed to cache AoS layout\n");
        return false;
    }
    if (!cache.is_cached(req.key, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "AoS entry missing after caching\n");
        return false;
    }

    req.layout = GGML_LAYOUT_SOA;
    result     = cache.ensure_cached_layout(req, {});
    if (result.status != ggml_sycl::cache_layout_status::READY || !result.device_ptr) {
        fprintf(stderr, "Failed to cache SoA layout\n");
        return false;
    }
    if (cache.is_cached(req.key, GGML_LAYOUT_AOS)) {
        fprintf(stderr, "AoS entry still cached after SoA request\n");
        return false;
    }
    if (!cache.is_cached(req.key, GGML_LAYOUT_SOA)) {
        fprintf(stderr, "SoA entry missing after request\n");
        return false;
    }

    return true;
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    sycl::queue q;
    try {
        printf("Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    bool ok = true;
    ok &= test_evict_returns_bytes(q);
    ok &= test_realloc_failure_keeps_entry(q);
    ok &= test_realloc_eviction_failure_keeps_entry(q);
    ok &= test_layout_size_mismatch_recaches(q);
    ok &= test_layout_size_mismatch_pinned_fails(q);
    ok &= test_graph_pins_host_weights();
    ok &= test_stream_dma_mmap_fail(q);
    ok &= test_all_pinned_eviction_failure_new_entry(q);
    ok &= test_partial_eviction_insufficient(q);
    ok &= test_allocation_failure_new_entry(q);
    ok &= test_deferred_free_stress(q);
    ok &= test_unaligned_hash(q);
    ok &= test_unpin_experts(q);
    ok &= test_moe_overcommit_cap(q);
    ok &= test_single_layout_enforced(q);

    printf("\nUnified cache bug tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
