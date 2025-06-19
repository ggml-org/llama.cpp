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

// Use fixed seed for reproducible results
static std::mt19937 g_rng(42);

static void fill_tensor_f32(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    float* data = (float*)dst->data;
    size_t n_elements = ggml_nelements(dst);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = dis(g_rng);
    }
}

static void fill_tensor_f16(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    ggml_fp16_t* data = (ggml_fp16_t*)dst->data;
    size_t n_elements = ggml_nelements(dst);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = ggml_fp32_to_fp16(dis(g_rng));
    }
}

static void print_tensor_info(const char* name, ggml_tensor* tensor) {
    printf("%s: [%ld, %ld, %ld, %ld] type=%s, elements=%ld\n",
           name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
           ggml_type_name(tensor->type), ggml_nelements(tensor));
}

static void print_f32_sample(const char* name, ggml_tensor* tensor, int max_elements = 10) {
    if (tensor->type != GGML_TYPE_F32) {
        printf("%s: Not F32 tensor (type=%s)\n", name, ggml_type_name(tensor->type));
        return;
    }
    
    float* data = (float*)tensor->data;
    size_t n_elements = ggml_nelements(tensor);
    size_t elements_to_print = std::min((size_t)max_elements, n_elements);
    
    printf("%s sample values: ", name);
    for (size_t i = 0; i < elements_to_print; i++) {
        printf("%.6f ", data[i]);
    }
    if (elements_to_print < n_elements) {
        printf("... (total %ld elements)", n_elements);
    }
    printf("\n");
}

static float tensor_max_diff(ggml_tensor* a, ggml_tensor* b) {
    if (ggml_nelements(a) != ggml_nelements(b) || a->type != b->type) {
        printf("ERROR: Tensors have different sizes or types\n");
        return -1.0f;
    }
    
    if (a->type != GGML_TYPE_F32) {
        printf("ERROR: Only F32 tensors supported for comparison\n");
        return -1.0f;
    }
    
    float* data_a = (float*)a->data;
    float* data_b = (float*)b->data;
    size_t n_elements = ggml_nelements(a);
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < n_elements; i++) {
        float diff = std::abs(data_a[i] - data_b[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    return max_diff;
}

static void reset_state_tensor(ggml_tensor* state) {
    float* state_data = (float*)state->data;
    size_t n_pairs = ggml_nelements(state) / 2;
    
    for (size_t i = 0; i < n_pairs; i++) {
        state_data[i * 2 + 0] = -INFINITY;  // M (max KQ value)
        state_data[i * 2 + 1] = 0.0f;       // S (sum)
    }
}

int main() {
    printf("=== Flash Attention State Tensor - Comprehensive Test ===\n");

    // Test parameters
    const int head_dim   = 32;
    const int n_heads    = 8;
    const int n_kv_heads = 4;
    const int seq_len    = 2;
    const int kv_len     = 4;  // Will be split into segments
    const int n_threads  = 4;
    const int kv_segments = 2;  // Split KV into 2 segments
    const int kv_segment_len = kv_len / kv_segments;

    printf("Test Parameters:\n");
    printf("  head_dim=%d, n_heads=%d, n_kv_heads=%d\n", head_dim, n_heads, n_kv_heads);
    printf("  seq_len=%d, kv_len=%d\n", seq_len, kv_len);
    printf("  kv_segments=%d, kv_segment_len=%d\n", kv_segments, kv_segment_len);

    // Initialize ggml context
    const size_t ctx_size = 1024*1024*1024; // 1GB
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

    // ============================================================================
    // Create and initialize tensors with FIXED data
    // ============================================================================
    printf("\n--- Creating Fixed Test Data ---\n");

    // Create tensors for flash attention
    // Format: [head_dim, seq_len, n_heads, 1] for Q
    // Format: [head_dim, kv_len, n_kv_heads, 1] for K, V
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);

    // Create mask tensor with proper padding
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

    // Fill with FIXED reproducible data
    printf("\nGenerating fixed test data (seed=42)...\n");
    fill_tensor_f32(q, -0.8f, 0.8f);
    fill_tensor_f16(k, -0.6f, 0.6f);
    fill_tensor_f16(v, -0.7f, 0.7f);

    // Initialize mask (no causal mask - all positions can see all KV)
    ggml_fp16_t* mask_data = (ggml_fp16_t*)mask->data;
    memset(mask_data, 0, ggml_nbytes(mask));
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < kv_len; j++) {
            // No masking - all positions can see all KV tokens
            mask_data[i * padded_kv_len + j] = ggml_fp32_to_fp16(0.0f);
        }
    }

    printf("Fixed test data generated successfully\n");

    // ============================================================================
    // Test 1: Standard Flash Attention (Reference Result)
    // ============================================================================
    printf("\n--- Test 1: Standard Flash Attention (Reference) ---\n");

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

    struct ggml_cgraph * graph_standard = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_standard, result_standard);

    printf("Computing standard flash attention...\n");
    enum ggml_status status_standard = ggml_graph_compute_with_ctx(ctx, graph_standard, n_threads);

    if (status_standard != GGML_STATUS_SUCCESS) {
        printf("ERROR: Standard flash attention failed with status: %d\n", status_standard);
        ggml_free(ctx);
        return 1;
    }

    printf("Standard flash attention computation successful\n");
    print_f32_sample("Standard result", result_standard, 8);

    // ============================================================================
    // Test 2: Segmented Flash Attention with State Accumulation
    // ============================================================================
    printf("\n--- Test 2: Segmented Flash Attention with State ---\n");

    // Reset state tensor
    reset_state_tensor(state);
    
    // Create result tensor for accumulation (same shape as standard result)
    ggml_tensor * result_segmented = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
        head_dim, seq_len, n_heads, 1);

    // Initialize segmented result to zero
    memset(result_segmented->data, 0, ggml_nbytes(result_segmented));

    printf("Processing %d segments of KV cache (segment_len=%d)...\n", kv_segments, kv_segment_len);

    for (int seg = 0; seg < kv_segments; seg++) {
        printf("\n  Segment %d/%d (kv_pos %d-%d):\n", 
               seg + 1, kv_segments, seg * kv_segment_len, (seg + 1) * kv_segment_len - 1);

        // Print state before this segment
        printf("    State before segment %d: ", seg + 1);
        float* state_data = (float*)state->data;
        for (int i = 0; i < std::min(4, n_heads * seq_len); i++) {
            printf("[M=%.3f,S=%.3f] ", state_data[i * 2 + 0], state_data[i * 2 + 1]);
        }
        printf("...\n");

        // Create views of K and V for this segment using ggml_view_4d
        ggml_tensor * k_segment = ggml_view_4d(ctx, k, 
            head_dim, kv_segment_len, n_kv_heads, 1,  // ne
            k->nb[1], k->nb[2], k->nb[3],             // nb (strides)
            seg * kv_segment_len * k->nb[1]);         // offset

        ggml_tensor * v_segment = ggml_view_4d(ctx, v,
            head_dim, kv_segment_len, n_kv_heads, 1,  // ne
            v->nb[1], v->nb[2], v->nb[3],             // nb (strides)
            seg * kv_segment_len * v->nb[1]);         // offset

        // Create mask for this segment
        const int padded_segment_len = GGML_PAD(kv_segment_len, 64);
        ggml_tensor * mask_segment = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 
            padded_segment_len, padded_seq_len);

        // Fill segment mask
        ggml_fp16_t* mask_seg_data = (ggml_fp16_t*)mask_segment->data;
        memset(mask_seg_data, 0, ggml_nbytes(mask_segment));
        
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_segment_len; j++) {
                int global_j = seg * kv_segment_len + j;
                // No masking for segment - all positions can see all KV tokens in this segment
                mask_seg_data[i * padded_segment_len + j] = ggml_fp32_to_fp16(0.0f);
            }
        }

        // Debug: Print mask information for first segment
        if (seg == 0) {
            printf("    Debug - Global mask (first 4 seq positions, first 20 kv positions):\n");
            for (int i = 0; i < std::min(4, seq_len); i++) {
                printf("      seq[%d]: ", i);
                for (int j = 0; j < std::min(20, kv_len); j++) {
                    float mask_val = GGML_FP16_TO_FP32(mask_data[i * padded_kv_len + j]);
                    printf("%.0f ", mask_val == -INFINITY ? -1.0f : mask_val);
                }
                printf("...\n");
            }
            
            printf("    Debug - Segment mask (first 4 seq positions, all segment positions):\n");
            for (int i = 0; i < std::min(4, seq_len); i++) {
                printf("      seq[%d]: ", i);
                for (int j = 0; j < kv_segment_len; j++) {
                    float mask_val = GGML_FP16_TO_FP32(mask_seg_data[i * padded_segment_len + j]);
                    printf("%.0f ", mask_val == -INFINITY ? -1.0f : mask_val);
                }
                printf("\n");
            }
        }

        print_tensor_info("    K segment", k_segment);
        print_tensor_info("    V segment", v_segment);

        // Compute flash attention with state for this segment
        // CRITICAL: Create the operation but redirect its output to our accumulation tensor
        ggml_tensor * result_seg = ggml_flash_attn_ext_with_state(
            ctx, q, k_segment, v_segment, mask_segment, state,
            1.0f / std::sqrt(head_dim),  // scale
            0.0f,  // max_bias
            0.0f   // logit_softcap
        );
        ggml_flash_attn_ext_set_prec(result_seg, GGML_PREC_F32);

        if (!result_seg) {
            printf("ERROR: Failed to create segmented flash attention for segment %d\n", seg);
            ggml_free(ctx);
            return 1;
        }

        // CRITICAL FIX: Redirect the operation's output to our accumulation tensor
        // This ensures that each segment reads from and writes to the same tensor
        result_seg->data = result_segmented->data;
        result_seg->nb[0] = result_segmented->nb[0];
        result_seg->nb[1] = result_segmented->nb[1];
        result_seg->nb[2] = result_segmented->nb[2];
        result_seg->nb[3] = result_segmented->nb[3];

        struct ggml_cgraph * graph_seg = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph_seg, result_seg);

        enum ggml_status status_seg = ggml_graph_compute_with_ctx(ctx, graph_seg, n_threads);

        if (status_seg != GGML_STATUS_SUCCESS) {
            printf("ERROR: Segmented flash attention failed for segment %d with status: %d\n", seg, status_seg);
            ggml_free(ctx);
            return 1;
        }

        printf("    Segment %d computed successfully\n", seg + 1);
        print_f32_sample("    Segment result", result_segmented, 6);

        // Print state after this segment
        printf("    State after segment %d: ", seg + 1);
        for (int i = 0; i < std::min(4, n_heads * seq_len); i++) {
            printf("[M=%.3f,S=%.3f] ", state_data[i * 2 + 0], state_data[i * 2 + 1]);
        }
        printf("...\n");

        // No need to copy result since we're already writing to result_segmented
    }

    printf("\nSegmented computation completed\n");
    print_f32_sample("Final segmented result", result_segmented, 8);

    // ============================================================================
    // Test 3: Compare Results
    // ============================================================================
    printf("\n--- Test 3: Comparing Results ---\n");

    float max_diff = tensor_max_diff(result_standard, result_segmented);
    
    printf("Comparison between standard and segmented results:\n");
    printf("  Maximum absolute difference: %.2e\n", max_diff);
    
    const float tolerance = 1e-4;  // Reasonable tolerance for F16/F32 precision
    
    if (max_diff < tolerance) {
        printf("  ✅ PASS: Results match within tolerance (%.2e)\n", tolerance);
    } else {
        printf("  ❌ FAIL: Results differ beyond tolerance (%.2e)\n", tolerance);
        
        // Print detailed comparison for debugging
        printf("\nDetailed comparison:\n");
        print_f32_sample("Standard", result_standard, 20);
        print_f32_sample("Segmented", result_segmented, 20);
    }

    // ============================================================================
    // Test 4: State Tensor Analysis
    // ============================================================================
    printf("\n--- Test 4: State Tensor Analysis ---\n");

    printf("Final state tensor values:\n");
    print_f32_sample("Final state", state, 16);

    float* state_data = (float*)state->data;
    float min_m = INFINITY, max_m = -INFINITY;
    float min_s = INFINITY, max_s = -INFINITY;
    
    for (int i = 0; i < n_heads * seq_len; i++) {
        float m_val = state_data[i * 2 + 0];
        float s_val = state_data[i * 2 + 1];
        
        if (m_val != -INFINITY) {
            min_m = std::min(min_m, m_val);
            max_m = std::max(max_m, m_val);
        }
        
        min_s = std::min(min_s, s_val);
        max_s = std::max(max_s, s_val);
    }

    printf("State tensor statistics:\n");
    printf("  M values: min=%.6f, max=%.6f\n", min_m, max_m);
    printf("  S values: min=%.6f, max=%.6f\n", min_s, max_s);

    // ============================================================================
    // Final Results
    // ============================================================================
    printf("\n=== Final Test Results ===\n");
    
    if (max_diff < tolerance) {
        printf("🎉 ALL TESTS PASSED!\n");
        printf("✅ Segmented flash attention with state produces identical results\n");
        printf("✅ State tensor correctly accumulates across segments\n");
        printf("✅ Implementation is working correctly\n");
    } else {
        printf("❌ TESTS FAILED!\n");
        printf("❌ Results differ beyond acceptable tolerance\n");
        printf("❌ Implementation needs debugging\n");
    }

    printf("\nMax difference: %.2e (tolerance: %.2e)\n", max_diff, tolerance);

    // Cleanup
    ggml_free(ctx);
    return (max_diff < tolerance) ? 0 : 1;
}