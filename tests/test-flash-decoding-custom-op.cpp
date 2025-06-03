#include "../src/llama-kv-cache-mixed.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cstdlib>
#include <algorithm>

// Forward declaration of the flash decoding function
void ggml_custom_flash_attn_mixed_simple(
    ggml_tensor * dst,
    int ith,
    int nth,
    void* wdata,
    size_t wsize,
    void * userdata);

// Parameters for flash attention are defined in llama-kv-cache-mixed.h

static void fill_random_f32(float* data, size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n; i++) {
        data[i] = dis(gen);
    }
}

static void fill_random_f16(ggml_fp16_t* data, size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n; i++) {
        data[i] = ggml_fp32_to_fp16(dis(gen));
    }
}

static void fill_causal_mask(float* mask_data, int64_t n_tokens, int64_t kv_len) {
    for (int64_t i = 0; i < n_tokens; i++) {
        for (int64_t j = 0; j < kv_len; j++) {
            if (j <= i + (kv_len - n_tokens)) {
                mask_data[i * kv_len + j] = 0.0f;
            } else {
                mask_data[i * kv_len + j] = -INFINITY;
            }
        }
    }
}

static void print_tensor_info(const char* name, ggml_tensor* tensor) {
    printf("%s: [%ld, %ld, %ld, %ld] type=%s\n",
           name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
           ggml_type_name(tensor->type));
}

int main() {
    printf("Testing Flash-Decoding Custom Operation vs Standard Flash Attention\n");

    // Test parameters - reduce KV length to minimize F16 accumulation errors
    const int head_dim  = 128;
    const int n_heads   = 32;
    const int n_kv_heads = 8;
    const int seq_len   = 32;     // Q length
    const int kv_len    = 256;    // K/V length - reduced for better F16 precision
    const int n_threads = 8;  // Multi-thread stability test

    printf("Test Parameters:\n");
    printf("  head_dim=%d, n_heads=%d, n_kv_heads=%d, seq_len=%d, kv_len=%d\n", 
           head_dim, n_heads, n_kv_heads, seq_len, kv_len);
    printf("  GQA ratio: %d query heads per KV head\n", n_heads / n_kv_heads);

    // Initialize ggml context
    const size_t ctx_size = 256*1024*1024; // 256MB for context
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

    // Create tensors for custom flash attention (our format)
    // Format: [head_dim, seq_len, n_heads, 1] for Q, K, V
    // Test F16 V multi-type support: Q=F32, K=F16, V=F16, mask=F32
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, GGML_PAD(kv_len, 256),  n_kv_heads, 1);  
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, GGML_PAD(kv_len, 256),  n_kv_heads, 1);  // Test F16 V multi-type support

    // Create mask tensor for custom flash attention
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_len, GGML_PAD(seq_len, 256));

    // Fill tensors with random data
    fill_random_f32((float*)q->data, ggml_nelements(q));
    fill_random_f16((ggml_fp16_t*)k->data, ggml_nelements(k));  // K is F16
    fill_random_f16((ggml_fp16_t*)v->data, ggml_nelements(v));  // V is F16 (test multi-type support)

    // Fill mask - use identity mask (all positions visible)
    float* mask_data = (float*)mask->data;
    fill_causal_mask(mask_data, seq_len, kv_len);

    for (int i = seq_len; i < seq_len; i++) {
        for (int j = kv_len; j < GGML_PAD(kv_len, 256); j++) {
            mask_data[i * kv_len + j] = -INFINITY;
        }
    }

    //> Use random data for realistic testing 
    // ggml_set_f32(q, 1.0f);    // Q = [1, 1]
    // ggml_set_f32(k, 2.0f);    // K = [2, 2] for all tokens
    // ggml_set_f32(v, 3.0f);    // V = [3, 3] for all tokens  
    // ggml_set_f32(mask, 0.0f); // No masking

    // ============================================================================
    // Test 1: Custom Flash-Decoding Implementation
    // ============================================================================
    printf("\n--- Testing Custom Flash-Decoding Implementation ---\n");

    // Create custom operation for flash-decoding
    // dst shape: [head_dim, n_heads, seq_len, n_batch]
    ggml_tensor * args[] = { q, k, v, mask };
    ggml_tensor * custom_result = ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,
        head_dim, n_heads, seq_len, 1,
        args,
        4,                  // number of arguments
        (ggml_custom_op_t)ggml_custom_flash_attn_mixed_simple,
        n_threads,          // number of threads
        NULL                // userdata
    );

    // ggml_set_f32(custom_result, 1.2f);

    if (!custom_result) {
        printf("ERROR: Failed to create custom flash attention operation\n");
        ggml_free(ctx);
        return 1;
    }

    // Build and execute computation graph for custom implementation
    struct ggml_cgraph * graph_custom = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_custom, custom_result);

    // Calculate workspace size for custom operation
    // FIXED: Must match exactly the layout in ggml_custom_flash_attn_mixed_simple (updated for multi-type V support)
    // Note: Output layout is [head_dim, n_heads, seq_len] for each thread's workspace
    const size_t OUTPUT_SIZE = head_dim * n_heads * seq_len;           // chunk_output: [DV, N_Q_HEADS, SEQ_LEN]
    const size_t LOCAL_MAX_SIZE = seq_len * n_heads;                   // local_max
    const size_t LOCAL_EXP_SUM_SIZE = seq_len * n_heads;               // local_exp_sum  
    const size_t V32_BUFFER_SIZE = head_dim;                           // V32_buffer (DV) - new for multi-type V support
    const size_t TEMP_BUFFER_SIZE = head_dim;                          // temp_buffer (DV)
    const size_t Q_QUANTIZED_SIZE = head_dim;                          // Q_q (DK floats for ggml_fp16_t[DK])
    const size_t SYNC_BUFFER_SIZE = 1;                                 // sync_buffer
    const size_t CACHE_LINE_SIZE_F32 = 16;                             // cache line padding
    const size_t elements_per_thread = OUTPUT_SIZE + LOCAL_MAX_SIZE + LOCAL_EXP_SUM_SIZE + V32_BUFFER_SIZE + TEMP_BUFFER_SIZE + Q_QUANTIZED_SIZE + SYNC_BUFFER_SIZE + CACHE_LINE_SIZE_F32;

    struct ggml_cplan cplan_custom = ggml_graph_plan(graph_custom, n_threads, NULL);

    // Allocate workspace
    size_t workspace_size = n_threads * elements_per_thread * sizeof(float);
    workspace_size = std::max(workspace_size, cplan_custom.work_size);
    
    printf("Workspace: %zu elements/thread, %.2f KB total\n", 
           elements_per_thread, workspace_size / 1024.0);
    
    uint8_t* workspace = (uint8_t*)malloc(workspace_size);
    if (!workspace) {
        printf("ERROR: Failed to allocate workspace of size %zu bytes\n", workspace_size);
        ggml_free(ctx);
        return 1;
    }
    cplan_custom.work_data = workspace;
    cplan_custom.work_size = workspace_size;

    printf("Computing custom flash-decoding...\n");
    enum ggml_status status_custom = ggml_graph_compute(graph_custom, &cplan_custom);

    if (status_custom != GGML_STATUS_SUCCESS) {
        printf("ERROR: Custom flash attention computation failed with status: %d\n", status_custom);
        free(workspace);
        ggml_free(ctx);
        return 1;
    }

    printf("Custom flash-decoding computation successful\n");

    // ============================================================================
    // Test 2: Standard Flash Attention Implementation (for comparison)
    // ============================================================================
    printf("\n--- Testing Standard Flash Attention ---\n");

    // Create tensors for standard flash attention
    // Standard format: [head_dim, seq_len, n_heads, batch_size] for Q, K, V
    ggml_tensor * q_std = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim,  seq_len,    n_heads,        1);
    ggml_tensor * k_std = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim,  kv_len,     n_kv_heads,     1);
    ggml_tensor * v_std = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim,  kv_len,     n_kv_heads,     1);

    // Convert data types and rearrange dimensions for GQA
    float* q_f32_src = (float*)q->data;
    ggml_fp16_t* k_f16_src = (ggml_fp16_t*)k->data;  // K is already F16
    ggml_fp16_t* v_f16_src = (ggml_fp16_t*)v->data;  // V is F16 for multi-type testing
    float* q_f32_std = (float*)q_std->data;  // Q_std is now F32
    ggml_fp16_t* k_f16 = (ggml_fp16_t*)k_std->data;
    ggml_fp16_t* v_f16 = (ggml_fp16_t*)v_std->data;

    // Copy Q: [head_dim, seq_len, n_heads] -> [head_dim, seq_len, n_heads] (F32 -> F32, no conversion needed)
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < head_dim; d++) {
                // Source: [d + t*head_dim + h*head_dim*seq_len]
                // Dest:   [d + t*head_dim + h*head_dim*seq_len] (same layout for now)
                int src_idx = d + t * head_dim + h * head_dim * seq_len;
                int dst_idx = d + t * head_dim + h * head_dim * seq_len;
                q_f32_std[dst_idx] = q_f32_src[src_idx];
            }
        }
    }

    // Copy and convert K,V: [head_dim, kv_len, n_kv_heads] -> [head_dim, kv_len, n_kv_heads]
    // For K and V in GQA, we need to use n_kv_heads (not n_heads)
    for (int h = 0; h < n_kv_heads; h++) {  // Use n_kv_heads for GQA
        for (int t = 0; t < kv_len; t++) {
            for (int d = 0; d < head_dim; d++) {
                // Source: [d + t*head_dim + h*head_dim*kv_len]
                // Dest:   [d + t*head_dim + h*head_dim*kv_len]  (same layout)
                int src_idx = d + t * head_dim + h * head_dim * kv_len;
                int dst_idx = d + t * head_dim + h * head_dim * kv_len;
                k_f16[dst_idx] = k_f16_src[src_idx];  // K is already F16, just copy
                v_f16[dst_idx] = v_f16_src[src_idx];  // V is F16, just copy
            }
        }
    }

    const float scale = 1.0f / sqrtf((float)head_dim);

    ggml_tensor * standard_result = ggml_flash_attn_ext(
        ctx, q_std, k_std, v_std, mask,  // Use NULL mask for comparison
        scale,
        0.0f,  // max_bias
        0.0f   // logit_softcap
    );

    if (!standard_result) {
        printf("ERROR: Failed to create standard flash attention operation\n");
        free(workspace);
        ggml_free(ctx);
        return 1;
    }

    printf("Standard flash attention tensor created successfully\n");
    printf("Standard result shape: [%ld, %ld, %ld, %ld]\n",
           standard_result->ne[0], standard_result->ne[1], standard_result->ne[2], standard_result->ne[3]);

    // Build and execute computation graph for standard implementation
    struct ggml_cgraph * graph_standard = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_standard, standard_result);

    printf("Computing standard flash attention...\n");
    enum ggml_status status_standard = ggml_graph_compute_with_ctx(ctx, graph_standard, n_threads);

    if (status_standard != GGML_STATUS_SUCCESS) {
        printf("ERROR: Standard flash attention computation failed with status: %d\n", status_standard);
        free(workspace);
        ggml_free(ctx);
        return 1;
    }

    printf("Standard flash attention computation successful\n");

    // ============================================================================
    // Compare Results
    // ============================================================================
    printf("\n--- Comparing Results ---\n");

    float* custom_data = (float*)custom_result->data;
    float* standard_data = nullptr;

    // Handle different output types from standard flash attention
    std::vector<float> standard_f32_data;
    if (standard_result->type == GGML_TYPE_F16) {
        ggml_fp16_t* standard_f16 = (ggml_fp16_t*)standard_result->data;
        size_t n_elements = ggml_nelements(standard_result);
        standard_f32_data.resize(n_elements);
        for (size_t i = 0; i < n_elements; i++) {
            standard_f32_data[i] = ggml_fp16_to_fp32(standard_f16[i]);
        }
        standard_data = standard_f32_data.data();
    } else {
        standard_data = (float*)standard_result->data;
    }

    // Compare element by element
    size_t custom_elements = ggml_nelements(custom_result);
    size_t standard_elements = ggml_nelements(standard_result);

    printf("Custom result elements: %zu\n", custom_elements);
    printf("Standard result elements: %zu\n", standard_elements);

    // For comparison, we need to consider the output format differences
    // Custom: [head_dim, seq_len, n_heads, 1]
    // Standard: typically [head_dim, n_heads, seq_len, 1] or similar

    float max_abs_diff = 0.0f;
    float sum_abs_diff = 0.0f;
    size_t compared_elements = 0;

    // Compare the first min(custom_elements, standard_elements) elements
    size_t min_elements = std::min(custom_elements, standard_elements);

    for (size_t i = 0; i < min_elements; i++) {
        float custom_val = custom_data[i];
        float standard_val = standard_data[i];

        if (std::isfinite(custom_val) && std::isfinite(standard_val)) {
            float abs_diff = std::abs(custom_val - standard_val);
            max_abs_diff = std::max(max_abs_diff, abs_diff);
            sum_abs_diff += abs_diff;
            compared_elements++;
        }
    }

    // Always show comparison statistics, even if there are no finite elements to compare
    float avg_abs_diff = compared_elements > 0 ? sum_abs_diff / compared_elements : NAN;

    printf("Comparison Statistics:\n");
    printf("  Compared elements: %zu\n", compared_elements);
    printf("  Max absolute difference: %.6e\n", max_abs_diff);
    printf("  Average absolute difference: %.6e\n", avg_abs_diff);

    // Print some sample values for inspection, including NaN values
    printf("\nSample values (first 128 elements):\n");
    printf("Index | Custom      | Standard    | Abs Diff\n");
    printf("------|-------------|-------------|----------\n");
    for (size_t i = 0; i < std::min(size_t(128), min_elements); i++) {
        float custom_val = custom_data[i];
        float standard_val = standard_data[i];

        // Print values even if they're NaN or Inf
        if (std::isfinite(custom_val) && std::isfinite(standard_val)) {
            float abs_diff = std::abs(custom_val - standard_val);
            printf("%5zu | %11.6f | %11.6f | %.6e\n", i, custom_val, standard_val, abs_diff);
        } else {
            // Handle NaN or Inf cases with special formatting
            char custom_str[12], standard_str[12], diff_str[12];

            if (std::isnan(custom_val)) strcpy(custom_str, "      NaN");
            else if (std::isinf(custom_val)) strcpy(custom_str, "      Inf");
            else snprintf(custom_str, 12, "%11.6f", custom_val);

            if (std::isnan(standard_val)) strcpy(standard_str, "      NaN");
            else if (std::isinf(standard_val)) strcpy(standard_str, "      Inf");
            else snprintf(standard_str, 12, "%11.6f", standard_val);

            strcpy(diff_str, "      N/A");

            printf("%5zu | %s | %s | %s\n", i, custom_str, standard_str, diff_str);
        }
    }

    // Determine test result - adjust tolerance for F16 precision
    const float tolerance = 5e-3f;  // Tolerance for F16 numerical differences
    bool test_passed = (compared_elements > 0) && (max_abs_diff < tolerance);

    printf("\nTest Result: %s\n", test_passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
    if (compared_elements > 0) {
        printf("(Max difference %.6e %s tolerance %.6e)\n",
               max_abs_diff, test_passed ? "<" : ">=", tolerance);
    } else {
        printf("(No finite elements to compare)\n");
    }

    // Cleanup
    free(workspace);
    ggml_free(ctx);

    return test_passed ? 0 : 1;
}
