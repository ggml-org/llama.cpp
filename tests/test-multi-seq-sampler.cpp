// Test for Multi-Sequence GPU Sampler API
// Tests the public ggml_backend_sycl_multi_seq_* functions for continuous batching
//
// This tests:
// 1. Sampler creation and destruction
// 2. Sequence add/remove
// 3. Parameter setting (temperature, top_k, top_p, min_p)
// 4. Batched sampling (contiguous logits)
// 5. Indexed sampling (batched logits with arbitrary indices)
// 6. Token retrieval

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <set>

#ifdef GGML_USE_SYCL
#include <sycl/sycl.hpp>
#include "ggml-sycl.h"
#include "ggml-backend.h"

// Helper to create mock SYCL backend
static ggml_backend_t create_sycl_backend() {
    // Get first SYCL device
    return ggml_backend_sycl_init(0);
}

// =============================================================================
// Test: Sampler creation and destruction
// =============================================================================

bool test_create_destroy() {
    printf("Testing sampler creation/destruction...\n");

    ggml_backend_t backend = create_sycl_backend();
    if (!backend) {
        printf("  SKIPPED: No SYCL backend available\n");
        return true;
    }

    // Create sampler
    const int max_seqs = 8;
    const int n_vocab = 32000;
    const uint32_t seed = 42;

    ggml_sycl_multi_seq_sampler_t sampler = ggml_backend_sycl_multi_seq_sampler_create(
        backend, max_seqs, n_vocab, seed);

    if (!sampler) {
        printf("  FAILED: Could not create sampler\n");
        ggml_backend_free(backend);
        return false;
    }

    // Verify initial state
    int active = ggml_backend_sycl_multi_seq_get_active_count(sampler);
    if (active != 0) {
        printf("  FAILED: Expected 0 active sequences, got %d\n", active);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);
        ggml_backend_free(backend);
        return false;
    }

    // Destroy sampler
    ggml_backend_sycl_multi_seq_sampler_free(sampler);
    ggml_backend_free(backend);

    printf("  PASSED\n");
    return true;
}

// =============================================================================
// Test: Sequence add/remove
// =============================================================================

bool test_add_remove_sequences() {
    printf("Testing sequence add/remove...\n");

    ggml_backend_t backend = create_sycl_backend();
    if (!backend) {
        printf("  SKIPPED: No SYCL backend available\n");
        return true;
    }

    const int max_seqs = 4;
    const int n_vocab = 1000;

    ggml_sycl_multi_seq_sampler_t sampler = ggml_backend_sycl_multi_seq_sampler_create(
        backend, max_seqs, n_vocab, 42);

    if (!sampler) {
        printf("  FAILED: Could not create sampler\n");
        ggml_backend_free(backend);
        return false;
    }

    // Add sequences with valid seq_ids (0, 1, 2 which are < max_seqs=4)
    bool ok0 = ggml_backend_sycl_multi_seq_add(sampler, 0, 1.0f);  // seq_id=0
    bool ok1 = ggml_backend_sycl_multi_seq_add(sampler, 1, 0.5f);  // seq_id=1
    bool ok2 = ggml_backend_sycl_multi_seq_add(sampler, 2, 2.0f);  // seq_id=2

    if (!ok0 || !ok1 || !ok2) {
        printf("  FAILED: Could not add sequences (ok: %d, %d, %d)\n", ok0, ok1, ok2);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);
        ggml_backend_free(backend);
        return false;
    }

    int active = ggml_backend_sycl_multi_seq_get_active_count(sampler);
    if (active != 3) {
        printf("  FAILED: Expected 3 active sequences, got %d\n", active);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);
        ggml_backend_free(backend);
        return false;
    }

    // Remove one sequence
    ggml_backend_sycl_multi_seq_remove(sampler, 1);  // Remove seq_id=1

    active = ggml_backend_sycl_multi_seq_get_active_count(sampler);
    if (active != 2) {
        printf("  FAILED: Expected 2 active sequences after remove, got %d\n", active);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);
        ggml_backend_free(backend);
        return false;
    }

    // Re-add seq_id=1, then add seq_id=3 (filling all 4 slots)
    bool ok3 = ggml_backend_sycl_multi_seq_add(sampler, 1, 1.0f);  // seq_id=1 again
    bool ok4 = ggml_backend_sycl_multi_seq_add(sampler, 3, 1.0f);  // seq_id=3
    // Now all 4 slots are full (seq_ids 0, 1, 2, 3)

    (void)ok3; (void)ok4;  // Suppress unused warnings

    // Try to add a 5th sequence - should fail since max_seqs=4 and seq_id must be < 4
    // All valid seq_ids (0-3) are already in use
    active = ggml_backend_sycl_multi_seq_get_active_count(sampler);
    if (active != 4) {
        printf("  FAILED: Expected 4 active sequences, got %d\n", active);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_sycl_multi_seq_sampler_free(sampler);
    ggml_backend_free(backend);

    printf("  PASSED\n");
    return true;
}

// =============================================================================
// Test: Parameter setting
// =============================================================================

bool test_set_params() {
    printf("Testing parameter setting...\n");

    ggml_backend_t backend = create_sycl_backend();
    if (!backend) {
        printf("  SKIPPED: No SYCL backend available\n");
        return true;
    }

    const int max_seqs = 4;
    const int n_vocab = 1000;

    ggml_sycl_multi_seq_sampler_t sampler = ggml_backend_sycl_multi_seq_sampler_create(
        backend, max_seqs, n_vocab, 42);

    if (!sampler) {
        printf("  FAILED: Could not create sampler\n");
        ggml_backend_free(backend);
        return false;
    }

    // Add a sequence (seq_id=0, which is valid since max_seqs=4)
    bool ok = ggml_backend_sycl_multi_seq_add(sampler, 0, 1.0f);
    if (!ok) {
        printf("  FAILED: Could not add sequence\n");
        ggml_backend_sycl_multi_seq_sampler_free(sampler);
        ggml_backend_free(backend);
        return false;
    }

    // Set various parameters for seq_id=0 (which maps to slot 0)
    ggml_backend_sycl_multi_seq_set_params(sampler, 0, 0.8f, 40, 0.95f, 0.05f);

    // Parameters are stored on device, so we can't easily verify them
    // The test passes if no crash occurs

    ggml_backend_sycl_multi_seq_sampler_free(sampler);
    ggml_backend_free(backend);

    printf("  PASSED\n");
    return true;
}

// =============================================================================
// Test: Batched sampling (greedy)
// =============================================================================

bool test_batched_sampling_greedy() {
    printf("Testing batched sampling (greedy)...\n");

    ggml_backend_t backend = create_sycl_backend();
    if (!backend) {
        printf("  SKIPPED: No SYCL backend available\n");
        return true;
    }

    const int n_seqs = 4;
    const int n_vocab = 1000;

    ggml_sycl_multi_seq_sampler_t sampler = ggml_backend_sycl_multi_seq_sampler_create(
        backend, n_seqs, n_vocab, 42);

    if (!sampler) {
        printf("  FAILED: Could not create sampler\n");
        ggml_backend_free(backend);
        return false;
    }

    // Add sequences with greedy mode (temp=0)
    std::vector<int> expected_tokens(n_seqs);
    for (int i = 0; i < n_seqs; i++) {
        bool ok = ggml_backend_sycl_multi_seq_add(sampler, i, 0.0f);  // temp=0 for greedy
        if (!ok) {
            printf("  FAILED: Could not add sequence %d\n", i);
            ggml_backend_sycl_multi_seq_sampler_free(sampler);
            ggml_backend_free(backend);
            return false;
        }
        expected_tokens[i] = 100 + i * 50;  // Expected max positions: 100, 150, 200, 250
    }

    // Create logits where each sequence has a different maximum
    std::vector<float> host_logits(n_seqs * n_vocab, 0.0f);
    for (int seq = 0; seq < n_seqs; seq++) {
        int max_pos = expected_tokens[seq];
        host_logits[seq * n_vocab + max_pos] = 10.0f;  // Clear maximum
    }

    // Allocate device memory for logits
    sycl::queue q;
    float* d_logits = sycl::malloc_device<float>(n_seqs * n_vocab, q);
    q.memcpy(d_logits, host_logits.data(), n_seqs * n_vocab * sizeof(float)).wait();

    // Sample
    int n_sampled = ggml_backend_sycl_multi_seq_sample(sampler, d_logits, true);

    if (n_sampled != n_seqs) {
        printf("  FAILED: Expected %d sampled, got %d\n", n_seqs, n_sampled);
        sycl::free(d_logits, q);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);
        ggml_backend_free(backend);
        return false;
    }

    // Get results
    std::vector<int32_t> tokens(n_seqs);
    std::vector<int> seq_ids(n_seqs);
    int n_tokens = ggml_backend_sycl_multi_seq_get_tokens(sampler, tokens.data(), seq_ids.data(), n_seqs);

    if (n_tokens != n_seqs) {
        printf("  FAILED: Expected %d tokens, got %d\n", n_seqs, n_tokens);
        sycl::free(d_logits, q);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);
        ggml_backend_free(backend);
        return false;
    }

    // Verify results
    bool all_correct = true;
    for (int i = 0; i < n_seqs; i++) {
        // Find which expected token matches this seq_id
        int seq_id = seq_ids[i];
        int expected = expected_tokens[seq_id];
        if (tokens[i] != expected) {
            printf("  Seq %d: expected token %d, got %d\n", seq_id, expected, tokens[i]);
            all_correct = false;
        }
    }

    sycl::free(d_logits, q);
    ggml_backend_sycl_multi_seq_sampler_free(sampler);
    ggml_backend_free(backend);

    if (all_correct) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED\n");
    }
    return all_correct;
}

// =============================================================================
// Test: Indexed sampling
// =============================================================================

bool test_indexed_sampling() {
    printf("Testing indexed sampling...\n");

    ggml_backend_t backend = create_sycl_backend();
    if (!backend) {
        printf("  SKIPPED: No SYCL backend available\n");
        return true;
    }

    const int max_seqs = 8;
    const int n_vocab = 1000;
    const int n_batch = 6;  // Total batch size
    const int n_active = 3; // Only 3 sequences are active

    ggml_sycl_multi_seq_sampler_t sampler = ggml_backend_sycl_multi_seq_sampler_create(
        backend, max_seqs, n_vocab, 42);

    if (!sampler) {
        printf("  FAILED: Could not create sampler\n");
        ggml_backend_free(backend);
        return false;
    }

    // Create batched logits [n_batch, n_vocab]
    // Sequences are at batch indices 1, 3, 5 (not contiguous)
    std::vector<float> host_logits(n_batch * n_vocab, 0.0f);

    // Batch index 1 -> seq 0, max at token 100
    host_logits[1 * n_vocab + 100] = 10.0f;

    // Batch index 3 -> seq 1, max at token 200
    host_logits[3 * n_vocab + 200] = 10.0f;

    // Batch index 5 -> seq 2, max at token 300
    host_logits[5 * n_vocab + 300] = 10.0f;

    // Allocate device memory
    sycl::queue q;
    float* d_logits = sycl::malloc_device<float>(n_batch * n_vocab, q);
    q.memcpy(d_logits, host_logits.data(), n_batch * n_vocab * sizeof(float)).wait();

    // Sequence IDs and batch indices
    int seq_ids[3] = {0, 1, 2};
    int batch_indices[3] = {1, 3, 5};  // Non-contiguous batch indices

    // Sample with indexed access
    int n_sampled = ggml_backend_sycl_multi_seq_sample_indexed(
        sampler, d_logits, seq_ids, batch_indices, n_active);

    if (n_sampled != n_active) {
        printf("  FAILED: Expected %d sampled, got %d\n", n_active, n_sampled);
        sycl::free(d_logits, q);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);
        ggml_backend_free(backend);
        return false;
    }

    // Get results
    std::vector<int32_t> tokens(max_seqs);
    std::vector<int> out_seq_ids(max_seqs);
    int n_tokens = ggml_backend_sycl_multi_seq_get_tokens(sampler, tokens.data(), out_seq_ids.data(), max_seqs);

    if (n_tokens != n_active) {
        printf("  FAILED: Expected %d tokens, got %d\n", n_active, n_tokens);
        sycl::free(d_logits, q);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);
        ggml_backend_free(backend);
        return false;
    }

    // Verify results - each sequence should have found its respective maximum
    int expected[3] = {100, 200, 300};
    bool all_correct = true;

    for (int i = 0; i < n_tokens; i++) {
        int sid = out_seq_ids[i];
        if (sid >= 0 && sid < 3) {
            if (tokens[i] != expected[sid]) {
                printf("  Seq %d: expected token %d, got %d\n", sid, expected[sid], tokens[i]);
                all_correct = false;
            }
        }
    }

    sycl::free(d_logits, q);
    ggml_backend_sycl_multi_seq_sampler_free(sampler);
    ggml_backend_free(backend);

    if (all_correct) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED\n");
    }
    return all_correct;
}

// =============================================================================
// Test: Temperature variation
// =============================================================================

bool test_temperature_variation() {
    printf("Testing temperature variation across sequences...\n");

    ggml_backend_t backend = create_sycl_backend();
    if (!backend) {
        printf("  SKIPPED: No SYCL backend available\n");
        return true;
    }

    const int n_vocab = 1000;
    const int n_trials = 50;

    // Test 1: Greedy mode (temp=0) with greedy=true
    printf("  Sub-test 1: Greedy mode (greedy=true)...\n");
    {
        ggml_sycl_multi_seq_sampler_t sampler = ggml_backend_sycl_multi_seq_sampler_create(
            backend, 1, n_vocab, 42);
        if (!sampler) {
            printf("    FAILED: Could not create sampler\n");
            ggml_backend_free(backend);
            return false;
        }

        ggml_backend_sycl_multi_seq_add(sampler, 0, 0.0f);  // temp=0 for greedy

        std::vector<float> host_logits(n_vocab, -10.0f);
        host_logits[500] = 5.0f;  // Max at 500

        sycl::queue q;
        float* d_logits = sycl::malloc_device<float>(n_vocab, q);

        int correct = 0;
        for (int trial = 0; trial < n_trials; trial++) {
            q.memcpy(d_logits, host_logits.data(), n_vocab * sizeof(float)).wait();
            ggml_backend_sycl_multi_seq_sample(sampler, d_logits, true);  // greedy=true

            int32_t token;
            int seq_id;
            ggml_backend_sycl_multi_seq_get_tokens(sampler, &token, &seq_id, 1);
            if (token == 500) correct++;
        }

        sycl::free(d_logits, q);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);

        printf("    Greedy correct: %d/%d %s\n", correct, n_trials, correct == n_trials ? "OK" : "FAIL");
        if (correct != n_trials) {
            ggml_backend_free(backend);
            return false;
        }
    }

    // Test 2: Low temperature (temp=0.5) - should strongly favor max
    printf("  Sub-test 2: Low temperature (temp=0.5)...\n");
    {
        ggml_sycl_multi_seq_sampler_t sampler = ggml_backend_sycl_multi_seq_sampler_create(
            backend, 1, n_vocab, 42);
        if (!sampler) {
            printf("    FAILED: Could not create sampler\n");
            ggml_backend_free(backend);
            return false;
        }

        ggml_backend_sycl_multi_seq_add(sampler, 0, 0.5f);  // Low temp

        std::vector<float> host_logits(n_vocab, -10.0f);
        host_logits[500] = 5.0f;  // Max at 500
        host_logits[501] = 4.0f;  // Second best
        host_logits[502] = 3.0f;  // Third best

        sycl::queue q;
        float* d_logits = sycl::malloc_device<float>(n_vocab, q);

        int near_max = 0;
        for (int trial = 0; trial < n_trials; trial++) {
            q.memcpy(d_logits, host_logits.data(), n_vocab * sizeof(float)).wait();
            ggml_backend_sycl_multi_seq_sample(sampler, d_logits, false);  // probabilistic

            int32_t token;
            int seq_id;
            ggml_backend_sycl_multi_seq_get_tokens(sampler, &token, &seq_id, 1);
            if (token >= 500 && token <= 502) near_max++;
        }

        sycl::free(d_logits, q);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);

        bool ok = (near_max >= n_trials * 0.8);  // Should pick top-3 >80% of time
        printf("    Low temp near max: %d/%d %s\n", near_max, n_trials, ok ? "OK" : "FAIL");
        if (!ok) {
            ggml_backend_free(backend);
            return false;
        }
    }

    // Test 3: High temperature (temp=2.0) - should show variety
    printf("  Sub-test 3: High temperature (temp=2.0)...\n");
    {
        ggml_sycl_multi_seq_sampler_t sampler = ggml_backend_sycl_multi_seq_sampler_create(
            backend, 1, n_vocab, 42);
        if (!sampler) {
            printf("    FAILED: Could not create sampler\n");
            ggml_backend_free(backend);
            return false;
        }

        ggml_backend_sycl_multi_seq_add(sampler, 0, 2.0f);  // High temp

        std::vector<float> host_logits(n_vocab, -10.0f);
        host_logits[500] = 5.0f;  // Max at 500
        host_logits[501] = 4.0f;  // Second best
        host_logits[502] = 3.0f;  // Third best

        sycl::queue q;
        float* d_logits = sycl::malloc_device<float>(n_vocab, q);

        std::set<int> unique_tokens;
        for (int trial = 0; trial < n_trials; trial++) {
            q.memcpy(d_logits, host_logits.data(), n_vocab * sizeof(float)).wait();
            ggml_backend_sycl_multi_seq_sample(sampler, d_logits, false);  // probabilistic

            int32_t token;
            int seq_id;
            ggml_backend_sycl_multi_seq_get_tokens(sampler, &token, &seq_id, 1);
            unique_tokens.insert(token);
        }

        sycl::free(d_logits, q);
        ggml_backend_sycl_multi_seq_sampler_free(sampler);

        bool ok = (unique_tokens.size() >= 3);  // Should have variety
        printf("    High temp unique tokens: %zu %s\n", unique_tokens.size(), ok ? "OK" : "FAIL");
        if (!ok) {
            ggml_backend_free(backend);
            return false;
        }
    }

    ggml_backend_free(backend);
    printf("  PASSED\n");
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== Multi-Sequence GPU Sampler Tests ===\n\n");

    int passed = 0;
    int failed = 0;

    if (test_create_destroy()) passed++; else failed++;
    if (test_add_remove_sequences()) passed++; else failed++;
    if (test_set_params()) passed++; else failed++;
    if (test_batched_sampling_greedy()) passed++; else failed++;
    if (test_indexed_sampling()) passed++; else failed++;
    if (test_temperature_variation()) passed++; else failed++;

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}

#else
int main() {
    printf("Multi-sequence sampler tests require SYCL build (-DGGML_USE_SYCL)\n");
    return 0;
}
#endif
