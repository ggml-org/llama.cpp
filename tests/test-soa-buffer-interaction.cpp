// Unit test for SoA buffer interaction bug
// Tests if F32 tensors (like inp_embd) are correctly read/written when
// Q4_0 weight tensors have SoA extras tracked in the same buffer context
//
// The hypothesis: SoA extras tracking for Q4_0 weights somehow corrupts
// or interferes with F32 tensor data during the decode phase
//
// Build: cmake --build build --target test-soa-buffer-interaction
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-soa-buffer-interaction

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Q4_0 block structure (matches ggml-common.h)
#define QK4_0 32
typedef struct {
    sycl::half d;        // delta (fp16)
    uint8_t qs[QK4_0/2]; // nibbles / quants (16 bytes)
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");

// Simulate the SoA reorder operation (matches ggml-sycl.cpp reorder_qw_q4_0)
void reorder_q4_0_to_soa(sycl::queue& q, uint8_t* data, int ncols, int nrows) {
    const int nblocks = nrows * (ncols / QK4_0);
    size_t size = nblocks * sizeof(block_q4_0);

    // Allocate temp buffer
    uint8_t* tmp_buf = sycl::malloc_device<uint8_t>(size, q);
    q.memcpy(tmp_buf, data, size).wait();

    // Compute SoA pointers
    uint8_t* qs_ptr = data;
    sycl::half* d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2);

    // Reorder AoS -> SoA
    q.parallel_for(nblocks, [=](auto i) {
        const block_q4_0* x = (const block_q4_0*)tmp_buf;
        const int ib = i;

        for (int j = 0; j < QK4_0/2; j++) {
            *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
        }
        *(d_ptr + ib) = x[ib].d;
    }).wait();

    sycl::free(tmp_buf, q);
}

// Simulate writing F32 embedding data (like inp_embd during decode)
void write_embedding_data(sycl::queue& q, float* dst, const float* src, int nelements) {
    q.memcpy(dst, src, nelements * sizeof(float)).wait();
}

// GPU kernel to read and verify F32 data (simulates RMS_NORM reading inp_embd)
void read_and_sum_f32(sycl::queue& q, const float* src, float* result, int nelements) {
    float* d_result = sycl::malloc_device<float>(1, q);
    q.memset(d_result, 0, sizeof(float)).wait();

    // Simple sum reduction to verify data is accessible
    q.parallel_for(1, [=](auto) {
        float sum = 0.0f;
        for (int i = 0; i < nelements; i++) {
            sum += src[i];
        }
        *d_result = sum;
    }).wait();

    q.memcpy(result, d_result, sizeof(float)).wait();
    sycl::free(d_result, q);
}

// GPU kernel to check for all zeros (like the bug symptom)
bool check_all_zeros_gpu(sycl::queue& q, const float* src, int nelements) {
    int* d_nonzero = sycl::malloc_device<int>(1, q);
    q.memset(d_nonzero, 0, sizeof(int)).wait();

    q.parallel_for(1, [=](auto) {
        for (int i = 0; i < nelements; i++) {
            if (src[i] != 0.0f) {
                *d_nonzero = 1;
            }
        }
    }).wait();

    int nonzero_count = 0;
    q.memcpy(&nonzero_count, d_nonzero, sizeof(int)).wait();
    sycl::free(d_nonzero, q);

    return (nonzero_count == 0);  // true if all zeros
}

// Test 1: Write/read F32 data without any Q4_0 tensors in buffer
// This is the baseline - should always work
bool test_f32_baseline() {
    printf("Test 1: F32 baseline (no Q4_0 tensors)\n");

    sycl::queue q{sycl::gpu_selector_v};
    printf("  Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const int nelements = 4096;  // Typical embedding dimension

    // Allocate device buffer for F32 data
    float* d_embd = sycl::malloc_device<float>(nelements, q);

    // Create host data with known values
    std::vector<float> h_embd(nelements);
    for (int i = 0; i < nelements; i++) {
        h_embd[i] = (float)(i + 1) * 0.001f;  // Non-zero values
    }

    // Write to device
    write_embedding_data(q, d_embd, h_embd.data(), nelements);

    // Read back via GPU kernel
    float sum = 0.0f;
    read_and_sum_f32(q, d_embd, &sum, nelements);

    // Expected sum: sum of 1..4096 * 0.001 = (4096*4097/2) * 0.001 = 8,390.656
    float expected = (float)(nelements * (nelements + 1) / 2) * 0.001f;

    // Check for zeros
    bool all_zeros = check_all_zeros_gpu(q, d_embd, nelements);

    sycl::free(d_embd, q);

    if (all_zeros) {
        printf("  FAIL: F32 data is ALL ZEROS!\n");
        return false;
    }

    if (fabsf(sum - expected) > expected * 0.01f) {  // 1% tolerance
        printf("  FAIL: Sum mismatch: got %.3f, expected %.3f\n", sum, expected);
        return false;
    }

    printf("  PASS: F32 baseline works (sum=%.3f)\n", sum);
    return true;
}

// Test 2: Write/read F32 data AFTER Q4_0 SoA reordering in same buffer allocation
// This simulates the decode phase where inp_embd follows weight reordering
bool test_f32_after_soa_reorder() {
    printf("Test 2: F32 after Q4_0 SoA reorder (simulates decode phase)\n");

    sycl::queue q{sycl::gpu_selector_v};

    // Allocate Q4_0 weight tensor (like layer weights)
    const int weight_rows = 128;
    const int weight_cols = 4096;
    const int nblocks = weight_rows * (weight_cols / QK4_0);
    const size_t q4_size = nblocks * sizeof(block_q4_0);

    uint8_t* d_weights = sycl::malloc_device<uint8_t>(q4_size, q);

    // Create AoS Q4_0 data on host
    std::vector<block_q4_0> h_weights(nblocks);
    for (int i = 0; i < nblocks; i++) {
        h_weights[i].d = sycl::half(1.0f + (float)(i % 100) / 1000.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            h_weights[i].qs[j] = (uint8_t)((i + j) & 0xFF);
        }
    }

    // Copy to device
    q.memcpy(d_weights, h_weights.data(), q4_size).wait();

    // Perform SoA reorder (this is what init_tensor does)
    printf("  Performing SoA reorder on Q4_0 weights...\n");
    reorder_q4_0_to_soa(q, d_weights, weight_cols, weight_rows);

    // Now allocate and write F32 embedding (simulates decode phase after prompt)
    const int embd_elements = 4096;
    float* d_embd = sycl::malloc_device<float>(embd_elements, q);

    // Create host embedding data
    std::vector<float> h_embd(embd_elements);
    for (int i = 0; i < embd_elements; i++) {
        h_embd[i] = (float)(i + 1) * 0.001f;
    }

    // Write embedding to device (simulates GET_ROWS result)
    write_embedding_data(q, d_embd, h_embd.data(), embd_elements);

    // Read back via GPU kernel (simulates RMS_NORM reading inp_embd)
    float sum = 0.0f;
    read_and_sum_f32(q, d_embd, &sum, embd_elements);

    float expected = (float)(embd_elements * (embd_elements + 1) / 2) * 0.001f;

    // Check for zeros
    bool all_zeros = check_all_zeros_gpu(q, d_embd, embd_elements);

    sycl::free(d_weights, q);
    sycl::free(d_embd, q);

    if (all_zeros) {
        printf("  FAIL: F32 data is ALL ZEROS after SoA reorder!\n");
        printf("  (This reproduces the inp_embd bug)\n");
        return false;
    }

    if (fabsf(sum - expected) > expected * 0.01f) {
        printf("  FAIL: Sum mismatch: got %.3f, expected %.3f\n", sum, expected);
        return false;
    }

    printf("  PASS: F32 works after SoA reorder (sum=%.3f)\n", sum);
    return true;
}

// Test 3: Multiple prompt/decode cycles with SoA
// Simulates multiple decode tokens after prompt
bool test_multiple_decode_cycles() {
    printf("Test 3: Multiple decode cycles with SoA reordering\n");

    sycl::queue q{sycl::gpu_selector_v};

    // Allocate Q4_0 weights
    const int weight_rows = 128;
    const int weight_cols = 4096;
    const int nblocks = weight_rows * (weight_cols / QK4_0);
    const size_t q4_size = nblocks * sizeof(block_q4_0);

    uint8_t* d_weights = sycl::malloc_device<uint8_t>(q4_size, q);

    std::vector<block_q4_0> h_weights(nblocks);
    for (int i = 0; i < nblocks; i++) {
        h_weights[i].d = sycl::half(1.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            h_weights[i].qs[j] = (uint8_t)((i + j) & 0xFF);
        }
    }
    q.memcpy(d_weights, h_weights.data(), q4_size).wait();

    // SoA reorder (prompt phase)
    printf("  Prompt phase: SoA reorder\n");
    reorder_q4_0_to_soa(q, d_weights, weight_cols, weight_rows);

    // F32 embedding buffer (reused across decode steps)
    const int embd_elements = 4096;
    float* d_embd = sycl::malloc_device<float>(embd_elements, q);

    int pass_count = 0;
    int fail_count = 0;

    // Simulate 10 decode tokens
    for (int token = 1; token <= 10; token++) {
        // Each token has different embedding values
        std::vector<float> h_embd(embd_elements);
        for (int i = 0; i < embd_elements; i++) {
            h_embd[i] = (float)(token * 1000 + i + 1) * 0.0001f;
        }

        // Write embedding (simulates GET_ROWS for this token)
        write_embedding_data(q, d_embd, h_embd.data(), embd_elements);

        // Read back via GPU (simulates kernel reading inp_embd)
        bool all_zeros = check_all_zeros_gpu(q, d_embd, embd_elements);

        if (all_zeros) {
            printf("  Token %d: FAIL - ALL ZEROS\n", token);
            fail_count++;
        } else {
            pass_count++;
        }
    }

    sycl::free(d_weights, q);
    sycl::free(d_embd, q);

    printf("  Results: %d/10 tokens OK, %d/10 had zeros\n", pass_count, fail_count);

    if (fail_count > 0) {
        printf("  FAIL: Some decode tokens had zero embeddings!\n");
        return false;
    }

    printf("  PASS: All decode tokens have valid embeddings\n");
    return true;
}

// Test 4: Buffer reuse after SoA (simulates graph allocator buffer_reset)
// This is the most likely scenario for the bug
bool test_buffer_reuse_after_soa() {
    printf("Test 4: Buffer reuse after SoA (simulates buffer_reset)\n");

    sycl::queue q{sycl::gpu_selector_v};

    // Phase 1: Allocate buffer, do SoA reorder
    const int weight_rows = 128;
    const int weight_cols = 4096;
    const int nblocks = weight_rows * (weight_cols / QK4_0);
    const size_t q4_size = nblocks * sizeof(block_q4_0);

    uint8_t* d_weights = sycl::malloc_device<uint8_t>(q4_size, q);

    std::vector<block_q4_0> h_weights(nblocks);
    for (int i = 0; i < nblocks; i++) {
        h_weights[i].d = sycl::half(1.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            h_weights[i].qs[j] = (uint8_t)((i + j) & 0xFF);
        }
    }
    q.memcpy(d_weights, h_weights.data(), q4_size).wait();
    reorder_q4_0_to_soa(q, d_weights, weight_cols, weight_rows);
    printf("  Phase 1: Q4_0 weights reordered to SoA\n");

    // Phase 2: Simulate buffer_reset - free the Q4_0 weights
    // In the real code, tensor_extras are cleared but tensor->extra might be stale
    sycl::free(d_weights, q);
    d_weights = nullptr;  // Simulate tensor_extras.clear()
    printf("  Phase 2: Simulated buffer_reset (freed Q4_0 weights)\n");

    // Phase 3: Allocate new F32 tensor in "fresh" buffer (decode phase)
    const int embd_elements = 4096;
    float* d_embd = sycl::malloc_device<float>(embd_elements, q);

    std::vector<float> h_embd(embd_elements);
    for (int i = 0; i < embd_elements; i++) {
        h_embd[i] = (float)(i + 1) * 0.001f;
    }

    write_embedding_data(q, d_embd, h_embd.data(), embd_elements);
    printf("  Phase 3: Allocated new F32 tensor, wrote data\n");

    // Phase 4: Read back and verify
    bool all_zeros = check_all_zeros_gpu(q, d_embd, embd_elements);

    float sum = 0.0f;
    read_and_sum_f32(q, d_embd, &sum, embd_elements);
    float expected = (float)(embd_elements * (embd_elements + 1) / 2) * 0.001f;

    sycl::free(d_embd, q);

    if (all_zeros) {
        printf("  FAIL: F32 data is ALL ZEROS after buffer reuse!\n");
        return false;
    }

    if (fabsf(sum - expected) > expected * 0.01f) {
        printf("  FAIL: Sum mismatch: got %.3f, expected %.3f\n", sum, expected);
        return false;
    }

    printf("  PASS: Buffer reuse works correctly (sum=%.3f)\n", sum);
    return true;
}

// Test 5: Interleaved Q4_0 and F32 operations (closest to actual decode)
// This tests the exact pattern that might cause the bug
bool test_interleaved_operations() {
    printf("Test 5: Interleaved Q4_0 and F32 operations\n");

    sycl::queue q{sycl::gpu_selector_v};

    // Allocate both Q4_0 weights and F32 embedding at same time
    const int weight_rows = 128;
    const int weight_cols = 4096;
    const int nblocks = weight_rows * (weight_cols / QK4_0);
    const size_t q4_size = nblocks * sizeof(block_q4_0);
    const int embd_elements = 4096;

    uint8_t* d_weights = sycl::malloc_device<uint8_t>(q4_size, q);
    float* d_embd = sycl::malloc_device<float>(embd_elements, q);

    // Initialize Q4_0 weights
    std::vector<block_q4_0> h_weights(nblocks);
    for (int i = 0; i < nblocks; i++) {
        h_weights[i].d = sycl::half(1.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            h_weights[i].qs[j] = (uint8_t)((i + j) & 0xFF);
        }
    }
    q.memcpy(d_weights, h_weights.data(), q4_size).wait();

    // Initialize F32 embedding with known values
    std::vector<float> h_embd(embd_elements);
    for (int i = 0; i < embd_elements; i++) {
        h_embd[i] = (float)(i + 1) * 0.001f;
    }
    write_embedding_data(q, d_embd, h_embd.data(), embd_elements);

    // Verify embedding BEFORE SoA reorder
    float sum_before = 0.0f;
    read_and_sum_f32(q, d_embd, &sum_before, embd_elements);
    bool zeros_before = check_all_zeros_gpu(q, d_embd, embd_elements);
    printf("  Before SoA: zeros=%d, sum=%.3f\n", zeros_before, sum_before);

    // Perform SoA reorder on Q4_0 weights
    printf("  Performing SoA reorder on Q4_0 weights...\n");
    reorder_q4_0_to_soa(q, d_weights, weight_cols, weight_rows);

    // Verify embedding AFTER SoA reorder (this is the critical test)
    float sum_after = 0.0f;
    read_and_sum_f32(q, d_embd, &sum_after, embd_elements);
    bool zeros_after = check_all_zeros_gpu(q, d_embd, embd_elements);
    printf("  After SoA: zeros=%d, sum=%.3f\n", zeros_after, sum_after);

    sycl::free(d_weights, q);
    sycl::free(d_embd, q);

    if (zeros_after && !zeros_before) {
        printf("  FAIL: SoA reorder CORRUPTED F32 embedding data!\n");
        return false;
    }

    if (zeros_after) {
        printf("  FAIL: F32 embedding is all zeros\n");
        return false;
    }

    float expected = (float)(embd_elements * (embd_elements + 1) / 2) * 0.001f;
    if (fabsf(sum_after - expected) > expected * 0.01f) {
        printf("  FAIL: Sum mismatch: got %.3f, expected %.3f\n", sum_after, expected);
        return false;
    }

    printf("  PASS: Interleaved operations work correctly\n");
    return true;
}

// Test 6: Test with ACTUAL ggml backend buffer operations
// This would require linking against ggml, but we simulate the key patterns
bool test_simulated_decode_sequence() {
    printf("Test 6: Simulated full decode sequence (5 prompt + 10 decode tokens)\n");

    sycl::queue q{sycl::gpu_selector_v};

    // Setup: Allocate weight and embedding buffers
    const int weight_rows = 4096;
    const int weight_cols = 4096;
    const int nblocks = weight_rows * (weight_cols / QK4_0);
    const size_t q4_size = nblocks * sizeof(block_q4_0);
    const int embd_elements = 4096;

    uint8_t* d_weights = sycl::malloc_device<uint8_t>(q4_size, q);
    float* d_embd = sycl::malloc_device<float>(embd_elements, q);

    // Initialize weights
    std::vector<block_q4_0> h_weights(nblocks);
    for (int i = 0; i < nblocks; i++) {
        h_weights[i].d = sycl::half(0.5f);
        for (int j = 0; j < QK4_0/2; j++) {
            h_weights[i].qs[j] = (uint8_t)(0x88);  // Neutral value
        }
    }
    q.memcpy(d_weights, h_weights.data(), q4_size).wait();

    printf("  === PROMPT PHASE (5 tokens batched) ===\n");

    // Prompt phase: batch of 5 embeddings
    std::vector<float> h_prompt_embd(embd_elements);
    for (int i = 0; i < embd_elements; i++) {
        h_prompt_embd[i] = 1.0f;  // All ones for easy verification
    }
    write_embedding_data(q, d_embd, h_prompt_embd.data(), embd_elements);

    // SoA reorder happens during prompt phase for MUL_MAT
    reorder_q4_0_to_soa(q, d_weights, weight_cols, weight_rows);

    // Verify prompt embedding is intact
    float prompt_sum = 0.0f;
    read_and_sum_f32(q, d_embd, &prompt_sum, embd_elements);
    printf("  Prompt embedding sum: %.1f (expected: %.1f)\n", prompt_sum, (float)embd_elements);

    printf("  === DECODE PHASE (10 tokens one at a time) ===\n");

    int decode_pass = 0;
    int decode_fail = 0;

    for (int token = 1; token <= 10; token++) {
        // Each decode token: new embedding written to buffer
        std::vector<float> h_decode_embd(embd_elements);
        for (int i = 0; i < embd_elements; i++) {
            h_decode_embd[i] = (float)token;  // Use token ID as value
        }

        // Write embedding (simulates GET_ROWS result)
        write_embedding_data(q, d_embd, h_decode_embd.data(), embd_elements);

        // Read back (simulates RMS_NORM reading)
        float decode_sum = 0.0f;
        read_and_sum_f32(q, d_embd, &decode_sum, embd_elements);
        float expected = (float)token * embd_elements;

        bool all_zeros = check_all_zeros_gpu(q, d_embd, embd_elements);

        if (all_zeros) {
            printf("  Token %d: ZEROS! (expected sum: %.1f)\n", token, expected);
            decode_fail++;
        } else if (fabsf(decode_sum - expected) > expected * 0.01f) {
            printf("  Token %d: WRONG! sum=%.1f expected=%.1f\n", token, decode_sum, expected);
            decode_fail++;
        } else {
            decode_pass++;
        }
    }

    sycl::free(d_weights, q);
    sycl::free(d_embd, q);

    printf("  Decode results: %d/10 passed, %d/10 failed\n", decode_pass, decode_fail);

    if (decode_fail > 0) {
        printf("  FAIL: Some decode tokens had incorrect embeddings!\n");
        return false;
    }

    printf("  PASS: Full sequence completed correctly\n");
    return true;
}

int main() {
    printf("=== SoA Buffer Interaction Tests ===\n");
    printf("Testing if SoA reordering affects F32 tensor data (inp_embd bug)\n\n");

    try {
        int passed = 0;
        int failed = 0;

        if (test_f32_baseline()) passed++; else failed++;
        printf("\n");

        if (test_f32_after_soa_reorder()) passed++; else failed++;
        printf("\n");

        if (test_multiple_decode_cycles()) passed++; else failed++;
        printf("\n");

        if (test_buffer_reuse_after_soa()) passed++; else failed++;
        printf("\n");

        if (test_interleaved_operations()) passed++; else failed++;
        printf("\n");

        if (test_simulated_decode_sequence()) passed++; else failed++;
        printf("\n");

        printf("=================================\n");
        printf("Results: %d passed, %d failed\n", passed, failed);

        if (failed > 0) {
            printf("\nNOTE: Failures here indicate the SoA buffer mechanism\n");
            printf("is corrupting F32 tensor data, matching the inp_embd bug.\n");
        } else {
            printf("\nNOTE: All tests passed. The bug is NOT in basic buffer\n");
            printf("operations. Must investigate higher-level integration.\n");
        }

        return failed > 0 ? 1 : 0;

    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 1;
    }
}
