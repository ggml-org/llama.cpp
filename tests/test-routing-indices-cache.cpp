// Test: Routing indices cache extraction after argsort
// Purpose: Verify that MoE routing indices are captured after argsort
// for use by routing-aware expert pre-staging.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <thread>
#include <atomic>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

// Forward declaration of the API we're testing
// This will fail to link until we implement it
extern bool ggml_sycl_get_routing_indices(
    const char * tensor_name,
    std::vector<int32_t> & out_indices,
    int & n_expert_used,
    int & n_tokens
);

extern void ggml_sycl_clear_routing_indices_cache();

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft, ggml_tensor * tensor,
                                                 ggml_backend_buffer_usage usage) {
    const size_t size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

// Test 1: Basic argsort captures indices in cache
static bool test_argsort_captures_indices() {
    fprintf(stderr, "\n=== Test: argsort captures indices ===\n");

    // Setup: Create probabilities for top-k selection (MoE router output)
    const int n_experts = 8;      // Total experts
    const int n_tokens = 4;       // Batch of tokens
    const int n_expert_used = 2;  // Top-k experts per token

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        fprintf(stderr, "SKIP: Could not init SYCL backend\n");
        return true;  // Skip, not fail
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(sycl_backend);

    const ggml_init_params params = {
        16 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(sycl_backend);
        return false;
    }

    // Create router probability tensor [n_experts, n_tokens]
    // In MoE, this comes from softmax of router logits
    ggml_tensor * probs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, n_tokens);
    ggml_set_name(probs, "ffn_moe_router_probs");

    // Create argsort output [n_experts, n_tokens]
    ggml_tensor * sorted_indices = ggml_argsort(ctx, probs, GGML_SORT_ORDER_DESC);
    ggml_set_name(sorted_indices, "ffn_moe_topk");

    // Allocate buffers
    ggml_backend_buffer_t probs_buf = alloc_tensor_buffer(buft, probs, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t idx_buf = alloc_tensor_buffer(buft, sorted_indices, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    if (!probs_buf || !idx_buf) {
        fprintf(stderr, "FAIL: Could not allocate buffers\n");
        if (probs_buf) ggml_backend_buffer_free(probs_buf);
        if (idx_buf) ggml_backend_buffer_free(idx_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    // Fill probabilities with known pattern so we can verify sorting
    // Token 0: expert 3 highest, then expert 1
    // Token 1: expert 5 highest, then expert 2
    // Token 2: expert 0 highest, then expert 7
    // Token 3: expert 6 highest, then expert 4
    std::vector<float> prob_data(n_experts * n_tokens, 0.0f);
    // Token 0
    prob_data[0 * n_experts + 3] = 0.9f;
    prob_data[0 * n_experts + 1] = 0.8f;
    // Token 1
    prob_data[1 * n_experts + 5] = 0.85f;
    prob_data[1 * n_experts + 2] = 0.7f;
    // Token 2
    prob_data[2 * n_experts + 0] = 0.95f;
    prob_data[2 * n_experts + 7] = 0.6f;
    // Token 3
    prob_data[3 * n_experts + 6] = 0.88f;
    prob_data[3 * n_experts + 4] = 0.75f;

    ggml_backend_tensor_set(probs, prob_data.data(), 0, prob_data.size() * sizeof(float));

    // Clear any previous cache entries
    ggml_sycl_clear_routing_indices_cache();

    // Build and execute graph
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sorted_indices);

    ggml_status status = ggml_backend_graph_compute(sycl_backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: Graph compute failed\n");
        ggml_backend_buffer_free(probs_buf);
        ggml_backend_buffer_free(idx_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    // Verify: Check that indices were captured in cache
    std::vector<int32_t> cached_indices;
    int cached_n_expert_used = 0;
    int cached_n_tokens = 0;

    bool found = ggml_sycl_get_routing_indices("ffn_moe_topk", cached_indices, cached_n_expert_used, cached_n_tokens);

    if (!found) {
        fprintf(stderr, "FAIL: Routing indices not found in cache\n");
        ggml_backend_buffer_free(probs_buf);
        ggml_backend_buffer_free(idx_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    // For argsort, the cache should contain all sorted indices
    // In practice, n_expert_used would be extracted from the tensor shape
    if (cached_n_tokens != n_tokens) {
        fprintf(stderr, "FAIL: Expected %d tokens, got %d\n", n_tokens, cached_n_tokens);
        ggml_backend_buffer_free(probs_buf);
        ggml_backend_buffer_free(idx_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    // Verify the actual indices match expected top-k
    // Read back the output for verification
    std::vector<int32_t> gpu_indices(n_experts * n_tokens);
    ggml_backend_tensor_get(sorted_indices, gpu_indices.data(), 0, gpu_indices.size() * sizeof(int32_t));

    // Check that cached indices match GPU output
    if (cached_indices.size() != gpu_indices.size()) {
        fprintf(stderr, "FAIL: Cached size %zu != GPU size %zu\n", cached_indices.size(), gpu_indices.size());
        ggml_backend_buffer_free(probs_buf);
        ggml_backend_buffer_free(idx_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    for (size_t i = 0; i < cached_indices.size(); ++i) {
        if (cached_indices[i] != gpu_indices[i]) {
            fprintf(stderr, "FAIL: Mismatch at index %zu: cached=%d, gpu=%d\n",
                    i, cached_indices[i], gpu_indices[i]);
            ggml_backend_buffer_free(probs_buf);
            ggml_backend_buffer_free(idx_buf);
            ggml_free(ctx);
            ggml_backend_free(sycl_backend);
            return false;
        }
    }

    // Verify top-2 indices per token are correct (descending order)
    // Token 0: expect [3, 1, ...] (experts with highest probs)
    // Token 1: expect [5, 2, ...]
    // Token 2: expect [0, 7, ...]
    // Token 3: expect [6, 4, ...]
    int expected_top2[4][2] = {
        {3, 1},
        {5, 2},
        {0, 7},
        {6, 4}
    };

    for (int t = 0; t < n_tokens; ++t) {
        int idx0 = gpu_indices[t * n_experts + 0];
        int idx1 = gpu_indices[t * n_experts + 1];
        if (idx0 != expected_top2[t][0] || idx1 != expected_top2[t][1]) {
            fprintf(stderr, "FAIL: Token %d expected top-2 [%d, %d], got [%d, %d]\n",
                    t, expected_top2[t][0], expected_top2[t][1], idx0, idx1);
            ggml_backend_buffer_free(probs_buf);
            ggml_backend_buffer_free(idx_buf);
            ggml_free(ctx);
            ggml_backend_free(sycl_backend);
            return false;
        }
    }

    fprintf(stderr, "PASS: Argsort indices captured correctly\n");

    // Cleanup
    ggml_backend_buffer_free(probs_buf);
    ggml_backend_buffer_free(idx_buf);
    ggml_free(ctx);
    ggml_backend_free(sycl_backend);
    return true;
}

// Test 2: Non-MoE tensors are not captured
static bool test_non_moe_not_captured() {
    fprintf(stderr, "\n=== Test: non-MoE tensors not captured ===\n");

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        fprintf(stderr, "SKIP: Could not init SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(sycl_backend);

    const ggml_init_params params = {
        16 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(sycl_backend);
        return false;
    }

    // Create a non-MoE tensor with argsort
    ggml_tensor * data = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
    ggml_set_name(data, "regular_data");  // Not an MoE name

    ggml_tensor * sorted = ggml_argsort(ctx, data, GGML_SORT_ORDER_DESC);
    ggml_set_name(sorted, "sorted_indices");  // Not an MoE name

    ggml_backend_buffer_t data_buf = alloc_tensor_buffer(buft, data, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t sorted_buf = alloc_tensor_buffer(buft, sorted, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    if (!data_buf || !sorted_buf) {
        if (data_buf) ggml_backend_buffer_free(data_buf);
        if (sorted_buf) ggml_backend_buffer_free(sorted_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    // Fill with random data
    std::vector<float> rand_data(16 * 4);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float & v : rand_data) {
        v = dist(rng);
    }
    ggml_backend_tensor_set(data, rand_data.data(), 0, rand_data.size() * sizeof(float));

    ggml_sycl_clear_routing_indices_cache();

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sorted);

    ggml_status status = ggml_backend_graph_compute(sycl_backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: Graph compute failed\n");
        ggml_backend_buffer_free(data_buf);
        ggml_backend_buffer_free(sorted_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    // Should NOT find this in cache (not MoE related)
    std::vector<int32_t> cached_indices;
    int cached_n_expert_used = 0;
    int cached_n_tokens = 0;

    bool found = ggml_sycl_get_routing_indices("sorted_indices", cached_indices, cached_n_expert_used, cached_n_tokens);

    if (found) {
        fprintf(stderr, "FAIL: Non-MoE tensor was incorrectly cached\n");
        ggml_backend_buffer_free(data_buf);
        ggml_backend_buffer_free(sorted_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    fprintf(stderr, "PASS: Non-MoE tensors not captured\n");

    ggml_backend_buffer_free(data_buf);
    ggml_backend_buffer_free(sorted_buf);
    ggml_free(ctx);
    ggml_backend_free(sycl_backend);
    return true;
}

// Test 3: Thread safety - concurrent access to cache
static bool test_thread_safety() {
    fprintf(stderr, "\n=== Test: thread safety ===\n");

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        fprintf(stderr, "SKIP: Could not init SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(sycl_backend);

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

    // Create MoE tensor
    ggml_tensor * probs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_set_name(probs, "ffn_moe_probs");

    ggml_tensor * sorted = ggml_argsort(ctx, probs, GGML_SORT_ORDER_DESC);
    ggml_set_name(sorted, "ffn_moe_topk");

    ggml_backend_buffer_t probs_buf = alloc_tensor_buffer(buft, probs, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t sorted_buf = alloc_tensor_buffer(buft, sorted, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    if (!probs_buf || !sorted_buf) {
        if (probs_buf) ggml_backend_buffer_free(probs_buf);
        if (sorted_buf) ggml_backend_buffer_free(sorted_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    // Initialize data
    std::vector<float> prob_data(8 * 4, 0.1f);
    for (int i = 0; i < 4; ++i) {
        prob_data[i * 8 + i] = 0.9f;  // Each token prefers expert i
    }
    ggml_backend_tensor_set(probs, prob_data.data(), 0, prob_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sorted);

    // Execute to populate cache
    ggml_status status = ggml_backend_graph_compute(sycl_backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(probs_buf);
        ggml_backend_buffer_free(sorted_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    // Concurrent reads from multiple threads
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};
    const int num_threads = 4;
    const int reads_per_thread = 100;

    auto reader_thread = [&]() {
        for (int i = 0; i < reads_per_thread; ++i) {
            std::vector<int32_t> indices;
            int n_used = 0, n_tok = 0;

            bool ok = ggml_sycl_get_routing_indices("ffn_moe_topk", indices, n_used, n_tok);
            if (ok && n_tok == 4 && !indices.empty()) {
                success_count.fetch_add(1);
            } else {
                failure_count.fetch_add(1);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(reader_thread);
    }

    for (auto & t : threads) {
        t.join();
    }

    int total = success_count.load() + failure_count.load();
    int expected = num_threads * reads_per_thread;

    if (total != expected) {
        fprintf(stderr, "FAIL: Expected %d total reads, got %d\n", expected, total);
        ggml_backend_buffer_free(probs_buf);
        ggml_backend_buffer_free(sorted_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    if (failure_count.load() > 0) {
        fprintf(stderr, "FAIL: %d/%d reads failed\n", failure_count.load(), total);
        ggml_backend_buffer_free(probs_buf);
        ggml_backend_buffer_free(sorted_buf);
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return false;
    }

    fprintf(stderr, "PASS: Thread safety (%d concurrent reads successful)\n", total);

    ggml_backend_buffer_free(probs_buf);
    ggml_backend_buffer_free(sorted_buf);
    ggml_free(ctx);
    ggml_backend_free(sycl_backend);
    return true;
}

int main() {
    fprintf(stderr, "=== Routing Indices Cache Tests ===\n");

    int passed = 0;
    int failed = 0;

    if (test_argsort_captures_indices()) {
        passed++;
    } else {
        failed++;
    }

    if (test_non_moe_not_captured()) {
        passed++;
    } else {
        failed++;
    }

    if (test_thread_safety()) {
        passed++;
    } else {
        failed++;
    }

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}

#endif  // GGML_USE_SYCL
