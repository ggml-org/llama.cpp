#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cstdlib>
#include <algorithm>
#include <iostream>

static void fill_random_f32(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    float* data = (float*)dst->data;
    size_t n_elements = ggml_nelements(dst);

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = dis(gen);
    }
}

static void fill_random_f16(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    ggml_fp16_t* data = (ggml_fp16_t*)dst->data;
    size_t n_elements = ggml_nelements(dst);

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = ggml_fp32_to_fp16(dis(gen));
    }
}

static void print_tensor_info(const char* name, ggml_tensor* tensor) {
    printf("%s: [%ld, %ld, %ld, %ld] type=%s, elements=%ld\n",
           name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
           ggml_type_name(tensor->type), ggml_nelements(tensor));
}

static void print_f32_sample(const char* name, ggml_tensor* tensor, int max_elements = 10) {
    if (tensor->type != GGML_TYPE_F32) {
        printf("%s: Not F32 tensor\n", name);
        return;
    }
    
    float* data = (float*)tensor->data;
    size_t n_elements = ggml_nelements(tensor);
    int elements_to_print = std::min((size_t)max_elements, n_elements);
    
    printf("%s sample values: ", name);
    for (int i = 0; i < elements_to_print; i++) {
        printf("%.3f ", data[i]);
    }
    if (elements_to_print < n_elements) {
        printf("... (total %ld elements)", n_elements);
    }
    printf("\n");
}

int main() {
    printf("Testing Flash Attention with State Tensor\n");

    // Test parameters
    const int head_dim   = 16;
    const int n_heads    = 4;
    const int n_kv_heads = 2;
    const int seq_len    = 8;
    const int kv_len     = 32;
    const int n_threads  = 4;

    printf("Test Parameters:\n");
    printf("  head_dim=%d, n_heads=%d, n_kv_heads=%d, seq_len=%d, kv_len=%d\n",
           head_dim, n_heads, n_kv_heads, seq_len, kv_len);

    // Initialize ggml context
    const size_t ctx_size = 512*1024*1024; // 512MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize ggml context\n");
        return 1;
    }

    // Create tensors for flash attention
    // Format: [head_dim, seq_len, n_heads, 1] for Q
    // Format: [head_dim, kv_len, n_kv_heads, 1] for K, V
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);

    // Create mask tensor: [n_kv, n_seq, 1, 1] - padded to requirements
    const int padded_kv_len = GGML_PAD(kv_len, 64);
    const int padded_seq_len = GGML_PAD(seq_len, GGML_KQ_MASK_PAD);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded_kv_len, padded_seq_len);

    // Create state tensor: [2, n_heads * seq_len] for [M, S] pairs
    ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, n_heads * seq_len);

    print_tensor_info("Q", q);
    print_tensor_info("K", k);
    print_tensor_info("V", v);
    print_tensor_info("Mask", mask);
    print_tensor_info("State", state);

    // Fill tensors with test data
    fill_random_f32(q, -0.5f, 0.5f);
    fill_random_f16(k, -0.5f, 0.5f);
    fill_random_f16(v, -0.5f, 0.5f);

    // Initialize mask (simple causal mask)
    ggml_fp16_t* mask_data = (ggml_fp16_t*)mask->data;
    memset(mask_data, 0, ggml_nbytes(mask));
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < kv_len; j++) {
            if (j <= i + 10) {  // Allow seeing up to 10 positions ahead for this test
                mask_data[i * padded_kv_len + j] = ggml_fp32_to_fp16(0.0f);
            } else {
                mask_data[i * padded_kv_len + j] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
    }

    // Initialize state tensor with starting values
    // Format: [M, S] for each head/position
    float* state_data = (float*)state->data;
    for (int i = 0; i < n_heads * seq_len; i++) {
        state_data[i * 2 + 0] = -INFINITY;  // M (max KQ value)
        state_data[i * 2 + 1] = 0.0f;       // S (sum)
    }

    printf("\nInitial state values:\n");
    print_f32_sample("State", state, 20);

    // ============================================================================
    // Test 1: Standard Flash Attention (baseline)
    // ============================================================================
    printf("\n--- Testing Standard Flash Attention (baseline) ---\n");

    ggml_tensor * result_standard = ggml_flash_attn_ext(
        ctx, q, k, v, mask,
        1.0f / std::sqrt(head_dim),  // scale
        0.0f,  // max_bias
        0.0f   // logit_softcap
    );
    ggml_flash_attn_ext_set_prec(result_standard, GGML_PREC_F32);

    if (!result_standard) {
        printf("ERROR: Failed to create standard flash attention operation\n");
        ggml_free(ctx);
        return 1;
    }

    // Build and execute computation graph for standard implementation
    struct ggml_cgraph * graph_standard = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_standard, result_standard);

    printf("Computing standard flash attention...\n");
    enum ggml_status status_standard = ggml_graph_compute_with_ctx(ctx, graph_standard, n_threads);

    if (status_standard != GGML_STATUS_SUCCESS) {
        printf("ERROR: Standard flash attention computation failed with status: %d\n", status_standard);
        ggml_free(ctx);
        return 1;
    }

    printf("Standard flash attention computation successful\n");
    print_f32_sample("Standard result", result_standard, 20);

    // ============================================================================
    // Test 2: Flash Attention with State Tensor  
    // ============================================================================
    printf("\n--- Testing Flash Attention with State Tensor ---\n");

    ggml_tensor * result_with_state = ggml_flash_attn_ext_with_state(
        ctx, q, k, v, mask, state,
        1.0f / std::sqrt(head_dim),  // scale
        0.0f,  // max_bias
        0.0f   // logit_softcap
    );
    ggml_flash_attn_ext_set_prec(result_with_state, GGML_PREC_F32);

    if (!result_with_state) {
        printf("ERROR: Failed to create flash attention with state operation\n");
        ggml_free(ctx);
        return 1;
    }

    // Build and execute computation graph for state implementation
    struct ggml_cgraph * graph_with_state = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_with_state, result_with_state);

    printf("Computing flash attention with state...\n");
    enum ggml_status status_with_state = ggml_graph_compute_with_ctx(ctx, graph_with_state, n_threads);

    if (status_with_state != GGML_STATUS_SUCCESS) {
        printf("ERROR: Flash attention with state computation failed with status: %d\n", status_with_state);
        ggml_free(ctx);
        return 1;
    }

    printf("Flash attention with state computation successful\n");
    print_f32_sample("Result with state", result_with_state, 20);

    printf("\nFinal state values:\n");
    print_f32_sample("Final state", state, 20);

    // ============================================================================
    // Test 3: Compare Results
    // ============================================================================
    printf("\n--- Comparing Results ---\n");

    float* data_standard = (float*)result_standard->data;
    float* data_with_state = (float*)result_with_state->data;
    size_t n_elements = ggml_nelements(result_standard);

    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int different_elements = 0;

    for (size_t i = 0; i < n_elements; i++) {
        float diff = std::abs(data_standard[i] - data_with_state[i]);
        if (diff > 1e-6) {
            different_elements++;
        }
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= n_elements;

    printf("Comparison statistics:\n");
    printf("  Total elements: %ld\n", n_elements);
    printf("  Elements with significant differences (>1e-6): %d\n", different_elements);
    printf("  Maximum difference: %.2e\n", max_diff);
    printf("  Average difference: %.2e\n", avg_diff);

    // ============================================================================
    // Test 4: Multiple Calls (State Accumulation)
    // ============================================================================
    printf("\n--- Testing Multiple Calls (State Accumulation) ---\n");

    // Reset state for accumulation test
    for (int i = 0; i < n_heads * seq_len; i++) {
        state_data[i * 2 + 0] = -INFINITY;  // M (max KQ value)
        state_data[i * 2 + 1] = 0.0f;       // S (sum)
    }

    // Call flash attention with state multiple times to test accumulation
    for (int call = 0; call < 3; call++) {
        printf("Call %d:\n", call + 1);
        
        ggml_tensor * result_accumulate = ggml_flash_attn_ext_with_state(
            ctx, q, k, v, mask, state,
            1.0f / std::sqrt(head_dim),
            0.0f, 0.0f
        );
        ggml_flash_attn_ext_set_prec(result_accumulate, GGML_PREC_F32);

        struct ggml_cgraph * graph_accumulate = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph_accumulate, result_accumulate);

        enum ggml_status status_accumulate = ggml_graph_compute_with_ctx(ctx, graph_accumulate, n_threads);

        if (status_accumulate != GGML_STATUS_SUCCESS) {
            printf("ERROR: Accumulation call %d failed with status: %d\n", call + 1, status_accumulate);
            ggml_free(ctx);
            return 1;
        }

        printf("  State after call %d: ", call + 1);
        for (int i = 0; i < std::min(4, n_heads * seq_len); i++) {
            printf("[M=%.3f,S=%.3f] ", state_data[i * 2 + 0], state_data[i * 2 + 1]);
        }
        printf("...\n");
    }

    printf("\n=== All Tests Completed Successfully! ===\n");

    // Cleanup
    ggml_free(ctx);
    return 0;
}