#include "ggml.h"
#include "ggml-cpu.h"
#include "log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Reference implementation of attention using matmul and softmax
// This function mimics the fallback path in llama-graph.cpp when flash attention is not available
ggml_tensor * reference_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    struct ggml_tensor * mask,
    float scale,
    float max_bias,
    bool v_trans,
    struct ggml_tensor * kq_bias = nullptr,
    struct ggml_tensor * v_mla = nullptr,
    float soft_cap = 0.0f) {

    // Calculate attention scores: Q*K^T
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

    // Set precision to F32 for better numerical stability
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    // Apply soft capping if needed
    if (soft_cap > 0.0f) {
        kq = ggml_scale(ctx, kq, 1.0f / soft_cap);
        kq = ggml_tanh(ctx, kq);
        kq = ggml_scale(ctx, kq, soft_cap);
    }

    // Add bias if provided
    if (kq_bias != nullptr) {
        kq = ggml_add(ctx, kq, kq_bias);
    }

    // Apply softmax with mask and scale
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, max_bias);

    // Prepare V for multiplication
    ggml_tensor * v_ready = v;
    if (!v_trans) {
        v_ready = ggml_cont(ctx, ggml_transpose(ctx, v));
    }

    // Calculate attention output: V * softmax(Q*K^T)
    ggml_tensor * kqv = ggml_mul_mat(ctx, v_ready, kq);

    // Apply MLA if provided (for MQA->MHA conversion)
    if (v_mla != nullptr) {
        kqv = ggml_mul_mat(ctx, v_mla, kqv);
    }

    // Rearrange dimensions
    ggml_tensor * result = ggml_permute(ctx, kqv, 0, 2, 1, 3);

    // Get final 2D shape
    const int n_head = q->ne[2];
    const int n_tokens = q->ne[1];
    result = ggml_cont_2d(ctx, result, result->ne[0] * n_head, n_tokens);

    return result;
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;

    printf("Testing Flash Attention\n");

    // Initialize ggml context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 128*1024*1024, // GGML_DEFAULT_GRAPH_SIZE
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize context\n");
        return 1;
    }

    // 使用小一点的参数，避免内存问题
    const int n_embd  = 4096;    // 嵌入维度
    const int n_head  = 32;     // 头数
    const int n_tokens = 32;    // 序列长度
    const int d_head  = n_embd / n_head; // 每个头的维度 = 8
    const int batch_size = 1;

    printf("Parameters: embd=%d, heads=%d, tokens=%d, d_head=%d\n", n_embd, n_head, n_tokens, d_head);

    // 创建QKV输入，使用F16数据类型
    // Note: As required by flash_attn_ext function, Q, K, V are 3D tensors with shape [d_head, n_tokens, n_head]
    // For this test, using 4D tensors with batch_size = 1
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_head, n_tokens,   batch_size);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_head, n_tokens,   batch_size);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_head, n_tokens,   batch_size);

    // Seed the random number generator for reproducibility
    srand((unsigned) time(NULL));

    // 填充数据 - 使用ggml_fp16_t填充
    const int n_elements_q = ggml_nelements(q);
    for (int i = 0; i < n_elements_q; i++) {
        float rand_q = (float)rand() / RAND_MAX;  // generate in [0,1]
        ((ggml_fp16_t*)q->data)[i] = ggml_fp32_to_fp16(rand_q);
    }

    // Fill K with random data
    const int n_elements_k = ggml_nelements(k);
    for (int i = 0; i < n_elements_k; i++) {
        float rand_k = (float)rand() / RAND_MAX;  // generate in [0,1]
        ((ggml_fp16_t*)k->data)[i] = ggml_fp32_to_fp16(rand_k);
    }

    // Fill V with random data
    const int n_elements_v = ggml_nelements(v);
    for (int i = 0; i < n_elements_v; i++) {
        float rand_v = (float)rand() / RAND_MAX;  // generate in [0,1]
        ((ggml_fp16_t*)v->data)[i] = ggml_fp32_to_fp16(rand_v);
    }

    printf("Created F16 tensors with random values: Q(%d els), K(%d els), V(%d els)\n", n_elements_q, n_elements_k, n_elements_v);

    const float scale = 1.0f / sqrtf(d_head);
    printf("Using scale = %f\n", scale);

    printf("Calling ggml_flash_attn_ext...\n");
    struct ggml_tensor * output = ggml_flash_attn_ext(
        ctx, q, k, v,     // q, k, v 张量
        NULL,             // mask 参数 (无掩码)
        scale,            // 缩放因子
        0.0f,             // 无软上限
        0.0f              // 无KQ 稀疏性参数
    );

    if (!output) {
        fprintf(stderr, "Flash attention returned NULL\n");
        ggml_free(ctx);
        return 1;
    }

    printf("Created output tensor with shape [%d, %d, %d]\n", (int)output->ne[0], (int)output->ne[1], (int)output->ne[2]);

    // 构建计算图并执行
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    printf("Executing computation graph...\n");
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    // ---------------------------------------------------------------------
    // Compute reference attention for verification
    // ---------------------------------------------------------------------
    struct ggml_tensor * ref_out = reference_attention(
            ctx,
            q,
            k,
            v,
            /*mask   =*/ NULL,
            /*scale  =*/ scale,
            /*max_bias=*/ 0.0f,
            /*v_trans=*/ false);

    struct ggml_cgraph * graph_ref = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_ref, ref_out);

    printf("Executing reference attention graph...\n");
    ggml_graph_compute_with_ctx(ctx, graph_ref, 1);

    // ---------------------------------------------------------------------
    // Compare results
    // ---------------------------------------------------------------------
    // The output sequence length is determined by q's sequence length (q->ne[1])
    const int output_seq_len = q->ne[1];
    const int total_elements_to_compare = d_head * n_head * output_seq_len;

    float max_abs_diff = 0.0f;

    for (int idx = 0; idx < total_elements_to_compare; ++idx) {
        float flash_val;
        float ref_val;

        if (output->type == GGML_TYPE_F16) {
            flash_val = ggml_fp16_to_fp32(((ggml_fp16_t *) output->data)[idx]);
        } else {
            flash_val = ((float *) output->data)[idx];
        }

        if (ref_out->type == GGML_TYPE_F16) {
            ref_val = ggml_fp16_to_fp32(((ggml_fp16_t *) ref_out->data)[idx]);
        } else {
            ref_val = ((float *) ref_out->data)[idx];
        }

        float diff = fabsf(flash_val - ref_val);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    printf("Max absolute difference between flash and reference: %.6f\n", max_abs_diff);
    printf("Comparison result: %s\n", (max_abs_diff < 1e-3f) ? "\033[32mMATCH\033[0m" : "\033[31mMISMATCH\033[0m");

    // ---------------------------------------------------------------------
    // (Optional) preview a few values from both tensors for manual inspection
    // ---------------------------------------------------------------------
    const int preview_batch_items = batch_size < 2 ? batch_size : 2; // Preview first few batch items
    const int preview_tokens_count = output_seq_len < 2 ? output_seq_len : 2; // Preview first few tokens (from q_len)
    const int preview_heads_count  = n_head < 2 ? n_head : 2;   // Preview first few heads
    const int preview_d_elements   = d_head < 128 ? d_head : 128;   // Preview first few elements within a head vector

    printf("\nSample values (flash | reference):\n");
    for (int b_idx = 0; b_idx < preview_batch_items; ++b_idx) {
        if (batch_size > 1) {
            printf("Batch index %d:\n", b_idx);
        }
        for (int t_idx = 0; t_idx < preview_tokens_count; ++t_idx) {
            printf("  Token index %d:\n", t_idx);
            for (int h_idx = 0; h_idx < preview_heads_count; ++h_idx) {
                printf("    Head index %d:\n", h_idx);
                for (int d_idx = 0; d_idx < preview_d_elements; ++d_idx) {
                    // output is [d_head, q_len, n_head, batch_size]
                    // ref_out is [d_head*n_head, q_len] (batch_size=1 assumed for ref_out construction)
                    // All indices are 0-based.

                    // For batch_size=1, output effectively [d_head, output_seq_len, n_head]
                    // Linear index for output[d_idx, t_idx, h_idx] (assuming batch_idx = 0)
                    size_t flash_offset = (size_t)b_idx * output->nb[3] + // batch stride
                                          (size_t)h_idx * output->nb[2] + // head stride
                                          (size_t)t_idx * output->nb[1] + // token stride
                                          (size_t)d_idx * output->nb[0];  // d_head element stride (usually type_size)

                    // ref_out is [d_head*n_head, output_seq_len]. (batch_idx = 0 assumed)
                    // Linear index for ref_out[ (h_idx * d_head + d_idx), t_idx ]
                    size_t ref_offset = (size_t)t_idx * ref_out->nb[1] + // token stride
                                       ((size_t)h_idx * d_head + d_idx) * ref_out->nb[0]; // element stride

                    float flash_val = NAN;
                    float ref_val   = NAN;

                    if (flash_offset < ggml_nbytes(output)) {
                        if (output->type == GGML_TYPE_F16) {
                            flash_val = ggml_fp16_to_fp32( ((ggml_fp16_t *) ((char *)output->data + flash_offset))[0] );
                        } else {
                            flash_val =                   ((float *)       ((char *)output->data + flash_offset))[0];
                        }
                    }

                    if (ref_offset < ggml_nbytes(ref_out)) {
                        if (ref_out->type == GGML_TYPE_F16) {
                           ref_val = ggml_fp16_to_fp32( ((ggml_fp16_t *) ((char *)ref_out->data + ref_offset))[0] );
                        } else {
                           ref_val =                   ((float *)       ((char *)ref_out->data + ref_offset))[0];
                        }
                    }
                    printf("      d_element %d: %.5f | %.5f\n", d_idx, flash_val, ref_val);
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Clean up
    // ---------------------------------------------------------------------
    ggml_free(ctx);
    printf("Test completed.\n");

    return (max_abs_diff < 1e-3f && total_elements_to_compare > 0) ? 0 : 1;
}
