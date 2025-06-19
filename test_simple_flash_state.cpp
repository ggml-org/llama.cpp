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
#include <algorithm>
#include <iostream>

int main() {
    printf("=== Simple Flash Attention State Test ===\n");

    // Simple test parameters
    const int head_dim = 32;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int seq_len = 1;
    const int kv_len = 4;
    const int n_threads = 1;

    // Initialize ggml context
    const size_t ctx_size = 256*1024*1024; // 256MB
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

    // Create tensors
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);

    const int padded_kv_len = GGML_PAD(kv_len, 64);
    const int padded_seq_len = GGML_PAD(seq_len, GGML_KQ_MASK_PAD);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded_kv_len, padded_seq_len);

    ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, n_heads * seq_len);

    // Initialize with simple data
    float* q_data = (float*)q->data;
    for (int i = 0; i < ggml_nelements(q); i++) {
        q_data[i] = 0.1f * (i % 10);
    }

    ggml_fp16_t* k_data = (ggml_fp16_t*)k->data;
    for (int i = 0; i < ggml_nelements(k); i++) {
        k_data[i] = ggml_fp32_to_fp16(0.1f * (i % 10));
    }

    ggml_fp16_t* v_data = (ggml_fp16_t*)v->data;
    for (int i = 0; i < ggml_nelements(v); i++) {
        v_data[i] = ggml_fp32_to_fp16(0.1f * (i % 10));
    }

    // Initialize mask (no masking)
    ggml_fp16_t* mask_data = (ggml_fp16_t*)mask->data;
    memset(mask_data, 0, ggml_nbytes(mask));

    // Initialize state
    float* state_data = (float*)state->data;
    for (int i = 0; i < n_heads * seq_len; i++) {
        state_data[i * 2 + 0] = -INFINITY;  // M
        state_data[i * 2 + 1] = 0.0f;       // S
    }

    printf("Input tensors initialized\n");

    // Test 1: Standard flash attention
    ggml_tensor * result_standard = ggml_flash_attn_ext(
        ctx, q, k, v, mask,
        1.0f / std::sqrt(head_dim), 0.0f, 0.0f
    );
    ggml_flash_attn_ext_set_prec(result_standard, GGML_PREC_F32);

    struct ggml_cgraph * graph_standard = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_standard, result_standard);
    ggml_graph_compute_with_ctx(ctx, graph_standard, n_threads);

    printf("Standard result: %.6f %.6f %.6f %.6f\n", 
           ((float*)result_standard->data)[0], ((float*)result_standard->data)[1],
           ((float*)result_standard->data)[2], ((float*)result_standard->data)[3]);

    // Test 2: Segmented flash attention with state
    // Create a persistent result tensor that will accumulate across segments
    ggml_tensor * result_segmented = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads, 1);
    memset(result_segmented->data, 0, ggml_nbytes(result_segmented));

    // Reset state
    for (int i = 0; i < n_heads * seq_len; i++) {
        state_data[i * 2 + 0] = -INFINITY;  // M
        state_data[i * 2 + 1] = 0.0f;       // S
    }

    printf("\nProcessing 2 segments...\n");

    for (int seg = 0; seg < 2; seg++) {
        printf("Segment %d:\n", seg + 1);
        
        // Create segment views
        int seg_len = 2;
        ggml_tensor * k_seg = ggml_view_4d(ctx, k, 
            head_dim, seg_len, n_kv_heads, 1,
            k->nb[1], k->nb[2], k->nb[3],
            seg * seg_len * k->nb[1]);

        ggml_tensor * v_seg = ggml_view_4d(ctx, v,
            head_dim, seg_len, n_kv_heads, 1,
            v->nb[1], v->nb[2], v->nb[3],
            seg * seg_len * v->nb[1]);

        const int padded_seg_len = GGML_PAD(seg_len, 64);
        ggml_tensor * mask_seg = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded_seg_len, padded_seq_len);
        memset(mask_seg->data, 0, ggml_nbytes(mask_seg));

        // CRITICAL: Create operation that writes directly to result_segmented
        ggml_tensor * op = ggml_flash_attn_ext_with_state(
            ctx, q, k_seg, v_seg, mask_seg, state,
            1.0f / std::sqrt(head_dim), 0.0f, 0.0f
        );
        ggml_flash_attn_ext_set_prec(op, GGML_PREC_F32);

        // CRITICAL: Replace the operation's data pointer to write directly to our accumulator
        op->data = result_segmented->data;
        op->nb[0] = result_segmented->nb[0];
        op->nb[1] = result_segmented->nb[1];
        op->nb[2] = result_segmented->nb[2];
        op->nb[3] = result_segmented->nb[3];

        struct ggml_cgraph * graph_seg = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph_seg, op);
        ggml_graph_compute_with_ctx(ctx, graph_seg, n_threads);

        printf("  After segment %d: %.6f %.6f %.6f %.6f\n", seg + 1,
               ((float*)result_segmented->data)[0], ((float*)result_segmented->data)[1],
               ((float*)result_segmented->data)[2], ((float*)result_segmented->data)[3]);
        printf("  State: M=%.6f, S=%.6f\n", state_data[0], state_data[1]);
    }

    // Compare results
    float* std_data = (float*)result_standard->data;
    float* seg_data = (float*)result_segmented->data;
    
    float max_diff = 0.0f;
    for (int i = 0; i < ggml_nelements(result_standard); i++) {
        float diff = std::abs(std_data[i] - seg_data[i]);
        max_diff = std::max(max_diff, diff);
    }

    printf("\nComparison:\n");
    printf("Standard:  %.6f %.6f %.6f %.6f\n", std_data[0], std_data[1], std_data[2], std_data[3]);
    printf("Segmented: %.6f %.6f %.6f %.6f\n", seg_data[0], seg_data[1], seg_data[2], seg_data[3]);
    printf("Max difference: %.6e\n", max_diff);

    const float tolerance = 1e-4;
    if (max_diff < tolerance) {
        printf("✅ TEST PASSED! (diff=%.6e < %.6e)\n", max_diff, tolerance);
    } else {
        printf("❌ TEST FAILED! (diff=%.6e >= %.6e)\n", max_diff, tolerance);
    }

    ggml_free(ctx);
    return (max_diff < tolerance) ? 0 : 1;
}