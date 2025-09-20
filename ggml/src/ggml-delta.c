#include "ggml.h"
#include "ggml-delta.h"
#include "ggml-impl.h"

static void report_tensor_size(const char * tensor_name, const struct ggml_tensor * tensor) {
    GGML_LOG_INFO("[%s] tensor size is [%lu, %lu, %lu, %lu], strides [%lu, %lu, %lu, %lu]\n", 
        tensor_name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
}

// ggml_delta_net
struct ggml_tensor * ggml_delta_net(
        struct ggml_context * ctx,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * q,
        struct ggml_tensor  * g,
        struct ggml_tensor  * conv_weight,
        struct ggml_tensor  * conv_bias,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state,
        bool                  use_qk_l2norm,
        float                 scale) {
    
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));
    report_tensor_size("orig_k", k);
    report_tensor_size("orig_v", v);
    report_tensor_size("orig_q", q);
    report_tensor_size("orig_g", g);
    report_tensor_size("orig_beta", beta);
    report_tensor_size("orig_state", state);
    
    const int64_t S_k = k->ne[0];
    const int64_t H_k = k->ne[1];
    const int64_t batch_size = k->ne[2];  
    const int64_t n_tokens = k->ne[3];
    
    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    
    GGML_ASSERT(v->ne[3] == n_tokens);
    GGML_ASSERT(q->ne[3] == n_tokens);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[1] == batch_size && beta->ne[2] == n_tokens && beta->ne[3] == 1);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v && state->ne[2] == H_v && state->ne[3] == 1);
    
    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[3] == n_tokens);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[3] == n_tokens);
       
    GGML_ASSERT(g->ne[0] == S_v && g->ne[1] == H_v && g->ne[3] == n_tokens && g->ne[2] == batch_size);
       
    // Merge q, k, v into qkv
    struct ggml_tensor * mixed_qkv = ggml_concat(ctx, q, k, 1);
    report_tensor_size("mixed_qkv_qk", mixed_qkv);
    mixed_qkv = ggml_concat(ctx, mixed_qkv, v, 1);
    report_tensor_size("mixed_qkv_qkv", mixed_qkv);

    uint32_t dim = (S_v * H_v) + 2 * (H_k * S_k);

    mixed_qkv = ggml_reshape_3d(ctx, mixed_qkv, batch_size, dim, n_tokens);
    report_tensor_size("mixed_qkv_reshaped", mixed_qkv);
    struct ggml_tensor * mixed_qkv_padded = ggml_pad(ctx, mixed_qkv, conv_weight->ne[0] - 1, 0, 0, 0);
    report_tensor_size("mixed_qkv_padded", mixed_qkv_padded);

    // Apply convolution
    struct ggml_tensor * conv_out = ggml_ssm_conv(ctx, mixed_qkv_padded, conv_weight);
    report_tensor_size("conv_out", conv_out);

    if (conv_bias) {
        conv_out = ggml_add(ctx, conv_out, conv_bias);
        report_tensor_size("conv_out_bias", conv_out);
    }

    conv_out = ggml_silu(ctx, conv_out);
    report_tensor_size("conv_out_silu", conv_out);

    conv_out = ggml_reshape_4d(ctx, conv_out, dim, n_tokens, batch_size, 1);
    report_tensor_size("conv_out_reshaped", conv_out);

    conv_out = ggml_permute(ctx, conv_out, 0, 2, 1, 3);
    report_tensor_size("conv_out_transposed", conv_out);

    // Beta sigmoid
    struct ggml_tensor * beta_sigmoid = ggml_sigmoid(ctx, beta);
    report_tensor_size("beta_sigmoid", beta_sigmoid);

    // Gate calculations are done elsewhere in llama-model.cpp

    // Re-split the qkv tensors
    struct ggml_tensor * q_conv = ggml_view_4d(ctx, conv_out, S_k, H_k, conv_out->ne[1], conv_out->ne[2], 
                                               H_k * sizeof(float), conv_out->nb[1], conv_out->nb[2], 0);
    report_tensor_size("q_conv_view", q_conv);

    struct ggml_tensor * k_conv = ggml_view_4d(ctx, conv_out, S_k, H_k, conv_out->ne[1], conv_out->ne[2],
                                               H_k * sizeof(float), conv_out->nb[1], conv_out->nb[2], S_k * H_k * sizeof(q->type));
    report_tensor_size("k_conv_view", k_conv);

    struct ggml_tensor * v_conv = ggml_view_4d(ctx, conv_out, S_v, H_v, conv_out->ne[1], conv_out->ne[2], H_v * sizeof(float),
                                               conv_out->nb[1], conv_out->nb[2], (2 * S_k * H_k) * sizeof(q->type));
    report_tensor_size("v_conv_view", v_conv);

    struct ggml_tensor * q_broadcast = q_conv;
    struct ggml_tensor * k_broadcast = k_conv;
    
    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (H_k != H_v) {
        GGML_ASSERT(H_v % H_k == 0);
        int64_t repeat_factor = H_v / H_k;
        
        q_broadcast = ggml_cont_4d(ctx, q_conv, S_k, batch_size, H_k, n_tokens);
        report_tensor_size("q_broadcast_reshape1", q_broadcast);
        k_broadcast = ggml_cont_4d(ctx, k_conv, S_k, batch_size, H_k, n_tokens);
        report_tensor_size("k_broadcast_reshape1", k_broadcast);
        
        q_broadcast = ggml_repeat_4d(ctx, q_broadcast, S_k, batch_size * repeat_factor, H_k, n_tokens);
        report_tensor_size("q_broadcast_repeat", q_broadcast);
        k_broadcast = ggml_repeat_4d(ctx, k_broadcast, S_k, batch_size * repeat_factor, H_k, n_tokens);
        report_tensor_size("k_broadcast_repeat", k_broadcast);
        
        q_broadcast = ggml_reshape_4d(ctx, q_broadcast, S_k, H_v, n_tokens, batch_size);
        report_tensor_size("q_broadcast_reshape2", q_broadcast);
        k_broadcast = ggml_reshape_4d(ctx, k_broadcast, S_k, H_v, n_tokens, batch_size);
        report_tensor_size("k_broadcast_reshape2", k_broadcast);
    }

    struct ggml_tensor * v_reshape = ggml_cont_4d(ctx, v_conv, S_v, H_v, n_tokens, batch_size);
    report_tensor_size("v_reshape", v_reshape);
    struct ggml_tensor * beta_broadcast = ggml_cont_4d(ctx, beta, 1, H_v, n_tokens, batch_size);
    report_tensor_size("beta_broadcast", beta_broadcast);
    struct ggml_tensor * state_broadcast = ggml_cont(ctx, state);
    report_tensor_size("state_broadcast", state_broadcast);
    
    return ggml_delta_net_op(ctx, q_broadcast, k_broadcast, v_reshape, g, beta_broadcast, state_broadcast, use_qk_l2norm, scale);
}

struct ggml_tensor * ggml_delta_net_op(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * g,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state,
        bool                  use_qk_l2norm,
        float                 scale) {
    
    // Debug: Log input tensor dimensions
    report_tensor_size("q_input", q);
    report_tensor_size("k_input", k);
    report_tensor_size("v_input", v);
    report_tensor_size("g_input", g);
    report_tensor_size("beta_input", beta);
    report_tensor_size("state_input", state);
    
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));
    
    const int64_t S_k = q->ne[0];  
    const int64_t H_k = q->ne[1];  
    const int64_t n_tokens = q->ne[2];
    const int64_t batch_size = q->ne[3];  
    
    const int64_t S_v = v->ne[0];  
    const int64_t H_v = v->ne[1];

    GGML_LOG_INFO("S_k = %ld, S_v = %ld, H_k = %ld, H_v = %ld\n", S_k, S_v, H_k, H_v);
    GGML_ASSERT(H_k == H_v); // we broadcasted the tensors in the main function to guarantee this
    
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_v && k->ne[2] == n_tokens && k->ne[3] == batch_size);
    GGML_ASSERT(v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == batch_size);
    GGML_ASSERT(g->ne[0] == S_v && g->ne[1] == H_v && g->ne[3] == n_tokens && g->ne[2] == batch_size);
    GGML_ASSERT(beta->ne[0] == 1 && beta->ne[1] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == batch_size);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v && state->ne[2] == H_v && state->ne[3] == n_tokens);
    
    struct ggml_tensor * output = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v * S_v, H_v, batch_size, n_tokens);
    report_tensor_size("output", output);
    
    struct ggml_tensor * new_state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v * S_v, H_v, 1, n_tokens);
    
    new_state = ggml_cpy(ctx, state, new_state);
    report_tensor_size("new_state_copied", new_state);
    
    if (use_qk_l2norm) {
        q = ggml_l2_norm(ctx, q, 1e-6f);
        report_tensor_size("q_l2norm", q);
        k = ggml_l2_norm(ctx, k, 1e-6f);
        report_tensor_size("k_l2norm", k);
    }
    
    q = ggml_scale(ctx, q, scale);
    report_tensor_size("q_scaled", q);
    
    struct ggml_tensor * state_flat = ggml_reshape_2d(ctx, new_state, S_v * S_v, H_v);
    report_tensor_size("state_flat", state_flat);
    
    for (int64_t t = 0; t < n_tokens; ++t) {
        struct ggml_tensor * q_t = ggml_view_3d(ctx, q, S_k, H_k, batch_size,
                                               q->nb[1], q->nb[2], t * q->nb[2]);
        report_tensor_size("q_t_view", q_t);
        struct ggml_tensor * k_t = ggml_view_3d(ctx, k, S_k, H_k, batch_size,
                                               k->nb[1], k->nb[2], t * k->nb[2]);
        report_tensor_size("k_t_view", k_t);
        struct ggml_tensor * v_t = ggml_view_3d(ctx, v, S_v, H_v, batch_size,
                                               v->nb[1], v->nb[2], t * v->nb[2]);
        report_tensor_size("v_t_view", v_t);
        struct ggml_tensor * beta_t = ggml_view_3d(ctx, beta, 1, H_v, batch_size,
                                                  beta->nb[1], beta->nb[2], t * beta->nb[2]);
        report_tensor_size("beta_t_view", beta_t);
                
        struct ggml_tensor * q_t_reshaped = ggml_reshape_2d(ctx, q_t, S_k, H_k * batch_size);
        report_tensor_size("q_t_reshaped", q_t_reshaped);
        
        struct ggml_tensor * k_t_reshaped = ggml_reshape_2d(ctx, k_t, S_k, H_k * batch_size);
        report_tensor_size("k_t_reshaped", k_t_reshaped);
        
        struct ggml_tensor * v_t_reshaped = ggml_reshape_2d(ctx, v_t, S_v, H_v * batch_size);
        report_tensor_size("v_t_reshaped", v_t_reshaped);
        
        struct ggml_tensor * beta_t_reshaped = ggml_reshape_2d(ctx, beta_t, 1, H_v * batch_size);
        report_tensor_size("beta_t_reshaped", beta_t_reshaped);
        
        struct ggml_tensor * k_t_final = k_t_reshaped;
        if (H_k != H_v) {
            GGML_ASSERT(H_v % H_k == 0);
            
            struct ggml_tensor * k_t_4d = ggml_reshape_4d(ctx, k_t_reshaped, S_k, H_k, 1, batch_size);
            report_tensor_size("k_t_4d", k_t_4d);
            
            k_t_final = ggml_repeat_4d(ctx, k_t_4d, S_k, H_v, 1, batch_size);
            report_tensor_size("k_t_final_repeated", k_t_final);
            
            k_t_final = ggml_reshape_2d(ctx, k_t_final, S_k, H_v * batch_size);
            report_tensor_size("k_t_final_2d", k_t_final);
        }
        
        struct ggml_tensor * state_2d = ggml_reshape_2d(ctx, new_state, S_v * S_v, H_v);
        report_tensor_size("state_2d", state_2d);
        
        struct ggml_tensor * state_t = state_2d;
        report_tensor_size("state_t", state_t);
        
        struct ggml_tensor * state_t_transposed = ggml_cont(ctx, ggml_transpose(ctx, state_t));
        report_tensor_size("state_t_transposed", state_t_transposed);
       
        struct ggml_tensor * k_t_final_reshaped = ggml_reshape_4d(ctx, k_t_final, H_v, S_k, batch_size, 1);
        report_tensor_size("k_t_final_reshaped", k_t_final_reshaped);
        
        struct ggml_tensor * kv_mem = ggml_mul_mat(ctx, state_t_transposed, k_t_final_reshaped);
        report_tensor_size("kv_mem", kv_mem);
                
        struct ggml_tensor * v_t_final = v_t_reshaped;
        struct ggml_tensor * beta_t_final = beta_t_reshaped;
                
        struct ggml_tensor * kv_mem_reshaped = ggml_transpose(ctx, kv_mem);
        report_tensor_size("kv_mem_reshaped", kv_mem_reshaped);
                
        struct ggml_tensor * delta = ggml_mul(ctx, ggml_sub(ctx, v_t_final, kv_mem_reshaped), beta_t_final);
        report_tensor_size("delta", delta);
        
        struct ggml_tensor * delta_reshaped = ggml_reshape_2d(ctx, delta, S_v, H_v * batch_size);
        report_tensor_size("delta_reshaped", delta_reshaped);
                
        k_t_final = ggml_cont(ctx, k_t_reshaped);
        report_tensor_size("k_t_final_cont", k_t_final);
        
        struct ggml_tensor * k_t_for_outer;
        if (S_k == S_v) {
            k_t_for_outer = k_t_final;
        } else if (S_k < S_v) {
            struct ggml_tensor * padding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S_v - S_k, H_v * batch_size);
            report_tensor_size("k_t_padding", padding);
            k_t_for_outer = ggml_concat(ctx, k_t_final, padding, 0);
            report_tensor_size("k_t_for_outer_padded", k_t_for_outer);
        } else {
            k_t_for_outer = ggml_view_2d(ctx, k_t_final, S_v, H_v * batch_size, k_t_final->nb[1], 0);
            report_tensor_size("k_t_for_outer_truncated", k_t_for_outer);
        }
        
        k_t_for_outer = ggml_cont(ctx, k_t_for_outer);
        report_tensor_size("k_t_for_outer_cont", k_t_for_outer);
        
        struct ggml_tensor * k_t_reshaped_4d = ggml_reshape_4d(ctx, k_t_for_outer, S_v, H_v, 1, batch_size);
        report_tensor_size("k_t_reshaped_4d", k_t_reshaped_4d);
        
        struct ggml_tensor * delta_transposed = ggml_transpose(ctx, delta_reshaped);
        report_tensor_size("delta_transposed", delta_transposed);
        
        delta_transposed = ggml_cont(ctx, delta_transposed);
        report_tensor_size("delta_transposed_cont", delta_transposed);
        
        struct ggml_tensor * delta_reshaped_4d = ggml_reshape_4d(ctx, delta_transposed, H_v, S_v, 1, batch_size);
        report_tensor_size("delta_reshaped_4d", delta_reshaped_4d);
        
        struct ggml_tensor * k_t_transposed = ggml_transpose(ctx, k_t_reshaped_4d);
        report_tensor_size("k_t_transposed", k_t_transposed);
        
        struct ggml_tensor * temp_product = ggml_mul_mat(ctx, delta_reshaped_4d, k_t_transposed);
        report_tensor_size("temp_product", temp_product);
        
        struct ggml_tensor * outer_product_raw = ggml_transpose(ctx, temp_product);
        report_tensor_size("outer_product_raw", outer_product_raw);
        
        struct ggml_tensor * outer_product_cont = ggml_cont(ctx, outer_product_raw);
        report_tensor_size("outer_product_cont", outer_product_cont);
        
        struct ggml_tensor * outer_product = ggml_reshape_2d(ctx, outer_product_cont, S_v, S_v);
        report_tensor_size("outer_product", outer_product);
        
        struct ggml_tensor * outer_product_reshaped;
        if (outer_product->ne[0] == S_v && outer_product->ne[1] == S_v) {
            outer_product_reshaped = ggml_reshape_2d(ctx, outer_product, S_v * S_v, 1);
        } else {
            outer_product_reshaped = ggml_reshape_2d(ctx, outer_product,
                                                    outer_product->ne[0] * outer_product->ne[1], 1);
        }
        report_tensor_size("outer_product_reshaped", outer_product_reshaped);
        
        struct ggml_tensor * outer_product_repeated = ggml_repeat(ctx, outer_product_reshaped, state_flat);
        report_tensor_size("outer_product_repeated", outer_product_repeated);
        
        state_flat = ggml_add(ctx, state_flat, outer_product_repeated);
        report_tensor_size("state_flat_updated", state_flat);
        
        struct ggml_tensor * q_t_final = q_t;
        report_tensor_size("q_t_final", q_t_final);
        
        q_t_final = ggml_cont(ctx, q_t_final);
        report_tensor_size("q_t_final_cont", q_t_final);
        
        struct ggml_tensor * state_flat_cont = ggml_cont(ctx, state_flat);
        report_tensor_size("state_flat_cont", state_flat_cont);
        
        struct ggml_tensor * q_t_matrix = ggml_reshape_2d(ctx, q_t_final, S_k, H_v * batch_size);
        report_tensor_size("q_t_matrix", q_t_matrix);
        
        struct ggml_tensor * q_t_matrix_transposed = ggml_transpose(ctx, q_t_matrix);
        report_tensor_size("q_t_matrix_transposed", q_t_matrix_transposed);
        
        struct ggml_tensor * state_flat_transposed = ggml_transpose(ctx, state_flat_cont);
        report_tensor_size("state_flat_transposed", state_flat_transposed);
        
        struct ggml_tensor * q_t_matrix_final = ggml_transpose(ctx, q_t_matrix_transposed);
        report_tensor_size("q_t_matrix_final", q_t_matrix_final);
        
        struct ggml_tensor * state_flat_final = ggml_transpose(ctx, state_flat_transposed);
        report_tensor_size("state_flat_final", state_flat_final);
        
        struct ggml_tensor * q_t_broadcast = ggml_repeat(ctx, q_t_final, state_flat_cont);
        report_tensor_size("q_t_broadcast", q_t_broadcast);
        
        struct ggml_tensor * state_q_product = ggml_mul(ctx, state_flat_cont, q_t_broadcast);
        report_tensor_size("state_q_product", state_q_product);
               
        struct ggml_tensor * state_q_3d = ggml_reshape_3d(ctx, state_q_product, S_v * S_v, H_v, batch_size);
        report_tensor_size("state_q_3d", state_q_3d);
        state_q_3d = ggml_cont(ctx, state_q_3d);
        report_tensor_size("state_q_3d_cont", state_q_3d);
        
        struct ggml_tensor * ones_vector = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v);
        ones_vector = ggml_exp(ctx, ones_vector);      // exp(0) = 1
        report_tensor_size("ones_vector", ones_vector);
        
        struct ggml_tensor * ones_col = ggml_reshape_2d(ctx, ones_vector, H_v, 1);
        report_tensor_size("ones_col", ones_col);
        
        struct ggml_tensor * output_parts[batch_size];
        for (int64_t b = 0; b < batch_size; b++) {
            struct ggml_tensor * batch_slice = ggml_view_3d(ctx, state_q_3d, S_v * S_v, H_v, 1,
                                                           state_q_3d->nb[1], state_q_3d->nb[2], b * state_q_3d->nb[2]);
            batch_slice = ggml_cont(ctx, batch_slice);
            report_tensor_size("batch_slice", batch_slice);
            
            struct ggml_tensor * batch_slice_t = ggml_transpose(ctx, batch_slice);
            report_tensor_size("batch_slice_t", batch_slice_t);
            struct ggml_tensor * batch_sum = ggml_mul_mat(ctx, ones_col, batch_slice_t);
            report_tensor_size("batch_sum", batch_sum);
            
            struct ggml_tensor * batch_result = ggml_reshape_2d(ctx, batch_sum, S_v, S_v);
            report_tensor_size("batch_result", batch_result);
            output_parts[b] = batch_result;
        }
        
        struct ggml_tensor * output_concat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S_v * S_v, batch_size);
        for (int64_t b = 0; b < batch_size; b++) {
            struct ggml_tensor * batch_output = ggml_view_2d(ctx, output_concat, S_v * S_v, 1,
                                                            output_concat->nb[1], b * output_concat->nb[1]);
            batch_output = ggml_cpy(ctx, output_parts[b], batch_output);
        }
        
        struct ggml_tensor * output_t_reshaped = ggml_reshape_2d(ctx, output_concat, S_v, S_v);
        struct ggml_tensor * output_t = ggml_cont(ctx, output_t_reshaped);
        report_tensor_size("output_t", output_t);
              
        struct ggml_tensor * output_slice = ggml_view_3d(ctx, output, S_v, S_v, batch_size,
                                                        output->nb[1], output->nb[2], t * output->nb[2]);
        report_tensor_size("output_slice", output_slice);
        output_slice = ggml_cpy(ctx, output_t, output_slice);
        report_tensor_size("output_slice_copied", output_slice);
    }
    
    struct ggml_tensor * result = ggml_concat(ctx, output, new_state, 2);
    report_tensor_size("result_final", result);
    return result;
}
