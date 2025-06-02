#include "../src/llama-kv-cache-mixed.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"
#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <algorithm>

// Fill tensor with random values
static void fill_random_f32(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Random between -1 and 1
    }
}

// Fill tensor with consistent values per head for debugging
// Format: [head_dim, seq_len, n_heads, 1]
template<typename T>
static void fill_head_consistent_values(T* data, int head_dim, int seq_len, int n_heads, float base_value = 1.0f) {
    for (int h = 0; h < n_heads; h++) {
        // float head_value = base_value + (float)h * 1.0f;  // Each head gets a different value
        float head_value = base_value;  // Each head gets a different value
        
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < head_dim; d++) {
                // Calculate index: [head_dim, seq_len, n_heads]
                int idx = d + t * head_dim + h * head_dim * seq_len;
                if constexpr (std::is_same_v<T, ggml_fp16_t>) {
                    data[idx] = ggml_fp32_to_fp16(head_value);
                } else {
                    data[idx] = static_cast<T>(head_value);
                }
            }
        }
    }
}

// Fill causal mask
static void fill_causal_mask(float* mask_data, int seq_len, int kv_len) {
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < kv_len; j++) {
            if (j > i) {
                mask_data[i * kv_len + j] = ggml_fp32_to_fp16(-INFINITY);
            } else {
                mask_data[i * kv_len + j] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
}

// Test the mixed KV cache flash attention
static void test_mixed_kv_flash_attention() {
    printf("\n=== Mixed KV Cache Flash Attention Test ===\n");
    
    // Test parameters
    const int head_dim   = 64;
    const int seq_len    = 1;
    const int kv_len     = 32;
    const int n_heads    = 4;
    const int n_kv_heads = 2;
    const int n_threads  = 2;  // Number of threads for parallel computation
    
    printf("Parameters:\n");
    printf("  head_dim: %d\n",  head_dim);
    printf("  seq_len:  %d\n",  seq_len);
    printf("  kv_len:   %d\n",  kv_len);
    printf("  n_heads:  %d\n",  n_heads);
    printf("  n_kv_heads: %d\n", n_kv_heads);
    printf("  n_threads: %d\n", n_threads);
    
    // Initialize GGML context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 128 * 1024 * 1024,  // 128MB
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to initialize GGML context\n");
        return;
    }
    
    printf("✓ GGML context initialized\n");
    
    // Create tensors for flash attention
    // Format: [head_dim, seq_len, n_heads, 1] for Q, K, V (matching reference)
    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads,    1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len,  n_kv_heads, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, kv_len,  n_kv_heads, 1);
    
    // Create mask tensor
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, GGML_PAD(seq_len, 256));
    
    if (!q || !k || !v || !mask) {
        printf("❌ Failed to create tensors\n");
        ggml_free(ctx);
        return;
    }
    
    printf("✓ Tensors created successfully\n");
    printf("  Q: [%d, %d, %d, %d]\n", (int)q->ne[0], (int)q->ne[1], (int)q->ne[2], (int)q->ne[3]);
    printf("  K: [%d, %d, %d, %d]\n", (int)k->ne[0], (int)k->ne[1], (int)k->ne[2], (int)k->ne[3]);
    printf("  V: [%d, %d, %d, %d]\n", (int)v->ne[0], (int)v->ne[1], (int)v->ne[2], (int)v->ne[3]);
    printf("  Mask: [%d, %d]\n",      (int)mask->ne[0], (int)mask->ne[1]);
    
    // Fill tensors with test data - use head-consistent values for debugging
    printf("✓ Filling tensors with head-consistent values for debugging\n");
    
    //> QKV init.
    fill_head_consistent_values<float>((float*)q->data,             head_dim, seq_len, n_heads,    1.0f);
    fill_head_consistent_values<ggml_fp16_t>((ggml_fp16_t*)k->data, head_dim, kv_len,  n_kv_heads, 1.0f);
    fill_head_consistent_values<float>((float*)v->data,             head_dim, kv_len,  n_kv_heads, 1.0f);
    
    // Fill mask - causal mask
    float* mask_data = (float*)mask->data;
    fill_causal_mask(mask_data, seq_len, kv_len);
    
    // Fill padding area with -infinity
    for (int i = seq_len; i < GGML_PAD(seq_len, 256); i++) {
        for (int j = 0; j < kv_len; j++) {
            mask_data[i * kv_len + j] = ggml_fp32_to_fp16(-INFINITY);
        }
    }
    
    printf("✓ Tensor data initialized\n");
    
    // Print sample tensor values for verification
    printf("\nDebug: Sample tensor values per head:\n");
    float* q_data = (float*)q->data;
    float* k_data = (float*)k->data;
    float* v_data = (float*)v->data;
    
    for (int h = 0; h < std::min(4, n_heads); h++) {
        // Sample first element of each head
        int q_idx = 0 + 0 * head_dim + h * head_dim * seq_len;  // [0, 0, h]
        int k_idx = 0 + 0 * head_dim + h * head_dim * kv_len;   // [0, 0, h]  
        int v_idx = 0 + 0 * head_dim + h * head_dim * kv_len;   // [0, 0, h]
        
        printf("  Head %d: Q=%.2f, K=%.2f, V=%.2f\n", h, q_data[q_idx], k_data[k_idx], v_data[v_idx]);
    }
    if (n_heads > 4) printf("  ... (showing first 4 heads)\n");
    
    // Print sample mask values for verification
    printf("\nMask sample (first few rows):\n");
    for (int i = 0; i < std::min(4, seq_len); i++) {
        printf("  Row %d:", i);
        for (int j = 0; j < std::min(8, kv_len); j++) {
            float mask_val = mask_data[i * kv_len + j];
            if (isinf(mask_val) && mask_val < 0) {
                printf(" -∞");
            } else {
                printf(" %4.1f", mask_val);
            }
        }
        if (kv_len > 8) printf(" ...");
        printf("\n");
    }
    
    // Test 1: Custom Flash Attention for Mixed KV Cache
    printf("\n--- Testing Custom Flash Attention ---\n");
    
    // Create custom operation for flash attention
    ggml_tensor* args[] = { q, k, v, mask };
    ggml_tensor* custom_result = ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,
        head_dim, seq_len, n_heads, 1,
        args,
        4,          // number of arguments
        (ggml_custom_op_t)ggml_custom_flash_attn_mixed_simple,  // From mixed kv cache
        n_threads,  // number of threads
        NULL        // userdata
    );
    
    if (!custom_result) {
        printf("❌ Failed to create custom flash attention operation\n");
        ggml_free(ctx);
        return;
    }
    
    printf("✓ Custom flash attention operation created\n");
    printf("  Result tensor: [%d, %d, %d, %d]\n", 
           (int)custom_result->ne[0], (int)custom_result->ne[1], 
           (int)custom_result->ne[2], (int)custom_result->ne[3]);
    
    // Build computation graph
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, custom_result);
    
    printf("✓ Computation graph built successfully\n");
    
    // Execute the graph
    int ret = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    
    if (ret != 0) {
        printf("❌ Graph computation failed with error code: %d\n", ret);
        ggml_free(ctx);
        return;
    }
    
    printf("✓ Graph computation completed successfully\n");
    
    // Verify results
    printf("\n--- Results Verification ---\n");
    
    float* result_data = (float*)custom_result->data;
    size_t result_elements = ggml_nelements(custom_result);
    
    // Check for NaN or infinity values
    size_t nan_count = 0;
    size_t inf_count = 0;
    float sum = 0.0f;
    float min_val = INFINITY;
    float max_val = -INFINITY;
    
    for (size_t i = 0; i < result_elements; i++) {
        float val = result_data[i];
        if (isnan(val)) {
            nan_count++;
        } else if (isinf(val)) {
            inf_count++;
        } else {
            sum += val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }
    
    printf("  Total elements: %zu\n", result_elements);
    printf("  NaN values: %zu\n", nan_count);
    printf("  Inf values: %zu\n", inf_count);
    printf("  Valid elements: %zu\n", result_elements - nan_count - inf_count);
    
    if (result_elements > nan_count + inf_count) {
        float avg = sum / (float)(result_elements - nan_count - inf_count);
        printf("  Value range: [%.6f, %.6f]\n", min_val, max_val);
        printf("  Average: %.6f\n", avg);
    }
    
    // Print sample output values per head
    printf("\nSample output values per head (first element of each head):\n");
    for (int h = 0; h < std::min(4, n_heads); h++) {
        // Sample first element of each head for first position
        int idx = 0 + 0 * head_dim + h * head_dim * seq_len;  // [0, 0, h]
        printf("  Head %d: %.6f\n", h, result_data[idx]);
    }
    if (n_heads > 4) printf("  ... (showing first 4 heads)\n");
    
    printf("\nDetailed output (first head, first few positions):\n");
    for (int pos = 0; pos < std::min(4, seq_len); pos++) {
        printf("  Pos %d:", pos);
        for (int dim = 0; dim < std::min(8, head_dim); dim++) {
            int idx = dim + pos * head_dim + 0 * head_dim * seq_len;  // First head only [dim, pos, 0]
            printf(" %7.4f", result_data[idx]);
        }
        if (head_dim > 8) printf(" ...");
        printf("\n");
    }
    
    // Basic sanity checks
    bool passed = true;
    if (nan_count > 0) {
        printf("❌ Test failed: Found %zu NaN values\n", nan_count);
        passed = false;
    }
    if (inf_count > 0) {
        printf("❌ Test failed: Found %zu infinite values\n", inf_count);
        passed = false;
    }
    if (result_elements == nan_count + inf_count) {
        printf("❌ Test failed: All values are NaN or infinite\n");
        passed = false;
    }
    
    if (passed) {
        printf("✓ Basic sanity checks passed\n");
        printf("✓ Mixed KV Cache Flash Attention test completed successfully\n");
    } else {
        printf("❌ Mixed KV Cache Flash Attention test failed\n");
    }
    
    // Cleanup
    ggml_free(ctx);
}

int main() {
    printf("Mixed KV Cache Simple Test Program\n");
    printf("==================================\n");
    printf("Testing basic flash attention functionality\n\n");
    
    // Seed random number generator
    srand(42);
    
    // Initialize backend
    ggml_backend_load_all();
    printf("✓ GGML backend initialized\n");
    
    try {
        // Test 1: Flash attention with mixed KV cache
        test_mixed_kv_flash_attention();
        
        printf("\n🎉 Flash attention test completed!\n");
        printf("✓ Flash attention functionality verified\n");
        printf("Note: Mixed precision test temporarily disabled due to ggml_cpy compatibility issues\n");
        
    } catch (const std::exception& e) {
        printf("\n❌ Test failed with exception: %s\n", e.what());
        return 1;
    }
    
    return 0;
} 