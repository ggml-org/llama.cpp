#include "llm_build_qwen3next.h"
#include "../../ggml/src/ggml-impl.h"

#include <cmath>

// Implementation of depthwise 1D convolution using F32 to avoid F16 limitations
static ggml_tensor* ggml_conv_1d_dw_f32(
        ggml_context * ctx,
        ggml_tensor  * kernel,
        ggml_tensor  * input,
        int           stride,
        int           padding,
        int           dilation) {

    const int64_t n_seqs = input->ne[2];
    const int64_t channels = input->ne[1];

    if (n_seqs > 1) {
        ggml_tensor* reshaped_input = ggml_reshape_4d(ctx, input, input->ne[0], 1, input->ne[1] * input->ne[2], 1);
        
        // For the kernel in [kernel_size, 1, channel_size, 1] format:
        // 1. First reshape to 2D: [kernel_size, channel_size]
        ggml_tensor* kernel_2d = ggml_reshape_2d(ctx, kernel, kernel->ne[0], kernel->ne[2]);
        
        // 2. Repeat the kernel for each sequence: [kernel_size, channel_size * n_seqs]
        ggml_tensor* repeated_kernel = ggml_repeat(ctx, kernel_2d, 
            ggml_new_tensor_2d(ctx, kernel->type, kernel->ne[0], channels * n_seqs));
        
        // 3. Reshape back to 4D for im2col: [kernel_size, 1, channels * n_seqs, 1]
        ggml_tensor* reshaped_kernel = ggml_reshape_4d(ctx, repeated_kernel, kernel->ne[0], 1, channels * n_seqs, 1);
        
        // Apply im2col
        ggml_tensor* im2col_result = ggml_im2col(ctx, reshaped_kernel, reshaped_input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
        
        // Multiply with the kernel
        ggml_tensor* mul_result = ggml_mul_mat(ctx, im2col_result, reshaped_kernel);
        
        // Reshape result back to [output_len, channels, n_seqs]
        ggml_tensor* output_3d = ggml_reshape_3d(ctx, mul_result, mul_result->ne[0], channels, n_seqs);
        
        return output_3d;
    } else {
        // Single sequence case - kernel is already in [kernel_size, 1, channel_size, 1]
        ggml_tensor* reshaped_input = ggml_reshape_4d(ctx, input, input->ne[0], 1, input->ne[1], 1);
        ggml_tensor* im2col_result = ggml_im2col(ctx, kernel, reshaped_input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
        ggml_tensor* mul_result = ggml_mul_mat(ctx, im2col_result, kernel);
        ggml_tensor* output_3d = ggml_reshape_3d(ctx, mul_result, mul_result->ne[0], mul_result->ne[2], 1);
        return output_3d;
    }
}


llm_build_qwen3next::llm_build_qwen3next(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context_mamba(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    //GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;
        cur = build_q3n_norm(inpL, model.layers[il].attn_norm, il);
        cb(cur, "attn_norm", il);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_qwen3next_linear_attn_layer(inp->get_recr(), cur, model, ubatch, il);
        } else {
            // Full attention layer
            cur = build_qwen3next_attention_layer(cur, inp_pos, inp->get_attn(), model, n_embd_head, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        
        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;
        
        // Post-attention norm
        ggml_tensor * attn_post_norm = build_q3n_norm(cur, model.layers[il].attn_post_norm, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // FFN layer (MoE or dense) - without residual connection
        cur = build_layer_ffn(attn_post_norm, model, il);
        cb(cur, "ffn_out", il);
        
        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Final norm
    cur = build_q3n_norm(cur, model.output_norm, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

struct ggml_tensor * llm_build_qwen3next::build_q3n_norm(struct ggml_tensor * input, struct ggml_tensor * weights, int layer) {
    // ggml_tensor * input_norm = ggml_scale_bias(ctx0, weights, 1.0f, 1.0f);
    // EDIT: we moved the shifting part to the conversion, so we just call normal build_norm
    return build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
}

struct ggml_tensor * llm_build_qwen3next::build_q3n_gated_norm(struct ggml_tensor * input, struct ggml_tensor * weights, struct ggml_tensor * gate, int layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);    
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);
    return ggml_mul(ctx0, normalized, gated_silu);
}

struct ggml_tensor * llm_build_qwen3next::build_qwen3next_attention_layer(ggml_tensor *             cur,
                                                                   ggml_tensor *             inp_pos,
                                                                   llm_graph_input_attn_kv * inp_attn,
                                                                   const llama_model &       model,
                                                                   const int64_t             n_embd_head,
                                                                   const int                 il) {
    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    struct ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur);
    cb(Qcur_full, "Qcur_full", il);
    Qcur_full = ggml_reshape_4d(ctx0, Qcur_full, n_embd_head * 2, n_head, n_tokens, 1);
    // Split Q projection into query and gate
    // The split should be along dimension 0 (the feature dimension)
    struct ggml_tensor * Qcur = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1, Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], 0);
    struct ggml_tensor * gate = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1, Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], 
        n_embd_head * ggml_element_size(Qcur_full));
    cb(Qcur, "Qcur", il);
    cb(gate, "gate", il);
    
    // Now reshape Qcur to [n_embd_head, n_head, n_tokens] for multi-head attention
    Qcur = ggml_cont_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
    cb(Qcur, "Qcur_reshaped", il);
    
    // Apply Q normalization
    Qcur = build_q3n_norm(Qcur, model.layers[il].attn_q_norm, il);
    cb(Qcur, "Qcur_normed", il);

    struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    cb(Kcur, "Kcur", il);

    struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_q3n_norm(Kcur, model.layers[il].attn_k_norm, il);
    cb(Kcur, "Kcur_normed", il);
    
    // Reshape gate to [n_embd, n_tokens] for the sigmoid gating (flatten the heads)
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply RoPE
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                         attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                         attn_factor, beta_fast, beta_slow);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    cur = build_attn(inp_attn, nullptr, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    struct ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

// delta_net
// prepare all the tensor data for the operation so we only
// do the absolutely necessary steps in the op itself
struct ggml_tensor * llm_build_qwen3next::delta_net(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * g,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state,
        bool                  use_qk_l2norm,
        float                 eps_norm,
        const int             il
    ) {
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));

    const int64_t S_k = q->ne[0];
    const int64_t H_k = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[2] == n_tokens);
    GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v * H_v && state->ne[2] == 1 && state->ne[3] == n_seqs);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);

    GGML_ASSERT(H_k == H_v); // we did a repeat to make sure this is the case

    cb(q, "q_prenorm", il);
    cb(k, "k_prenorm", il);

    if (use_qk_l2norm) {
        q = ggml_l2_norm(ctx, q, eps_norm);
        k = ggml_l2_norm(ctx, k, eps_norm);
    }

    cb(k, "k_postnorm", il);
    cb(q, "q_prescale", il);

    int64_t pad_size = (GGML_DELTA_NET_CHUNK - n_tokens % GGML_DELTA_NET_CHUNK) % GGML_DELTA_NET_CHUNK;
    // yes, n_tokens, not H_k, the reference implementation has wrong naming
    int64_t num_chunks = (n_tokens + pad_size) / GGML_DELTA_NET_CHUNK;

    float scale = 1.0f / sqrtf(S_v);
    q = ggml_scale(ctx, q, scale);

    cb(beta, "beta_raw", il);
    beta = ggml_sigmoid(ctx, beta);

    cb(q, "q_postscale", il);
    cb(beta, "beta_sigmoid", il);   

    q = ggml_cont_4d(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
    k = ggml_cont_4d(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
    v = ggml_cont_4d(ctx, ggml_permute(ctx, v, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
    beta = ggml_cont(ctx, ggml_permute(ctx, beta, 0, 2, 1, 3));

    q = ggml_pad(ctx, q, 0, pad_size, 0, 0); 
    k = ggml_pad(ctx, k, 0, pad_size, 0, 0);
    v = ggml_pad(ctx, v, 0, pad_size, 0, 0);
    beta = ggml_pad(ctx, beta, 0, pad_size, 0, 0);

    beta = ggml_cont(ctx, ggml_permute(ctx, beta, 2, 1, 0, 3));
    cb(beta, "beta_reshape", il);

    g = ggml_cont(ctx, ggml_permute(ctx, g, 2, 0, 3, 1));
    cb(g, "g_permute", il);
    
    g = ggml_pad(ctx, g, pad_size, 0, 0, 0);

    cb(q, "q_pad", il);
    cb(k, "k_pad", il);
    cb(v, "v_pad", il);
    cb(beta, "beta_pad", il);
    cb(g, "g_pad", il);
    
    GGML_ASSERT(q->ne[1] % GGML_DELTA_NET_CHUNK == 0 && q->ne[0] == S_k && q->ne[2] == H_k && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[1] % GGML_DELTA_NET_CHUNK == 0 && k->ne[0] == S_k && k->ne[2] == H_k && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[1] % GGML_DELTA_NET_CHUNK == 0 && v->ne[0] == S_v && v->ne[2] == H_k && v->ne[3] == n_seqs);
    GGML_ASSERT(beta->ne[0] % GGML_DELTA_NET_CHUNK == 0 && beta->ne[1] == H_k && beta->ne[2] == 1 && beta->ne[3] == n_seqs);
    GGML_ASSERT(g->ne[0] % GGML_DELTA_NET_CHUNK == 0 && g->ne[2] == H_k && g->ne[1] == 1 && g->ne[3] == n_seqs);

    struct ggml_tensor * v_beta = ggml_mul(ctx, v, beta);
    v_beta = ggml_reshape_4d(ctx, v_beta, S_v, GGML_DELTA_NET_CHUNK, H_k * num_chunks, n_seqs);
    cb(v_beta, "v_beta", il);
    struct ggml_tensor * k_beta = ggml_mul(ctx, k, beta);
    k_beta = ggml_reshape_4d(ctx, k_beta, S_v, GGML_DELTA_NET_CHUNK, H_k * num_chunks, n_seqs);
    cb(k_beta, "k_beta", il);
    k = ggml_reshape_4d(ctx, k, S_v, GGML_DELTA_NET_CHUNK, H_k * num_chunks, n_seqs);
    cb(k_beta, "k_reshape", il);
    q = ggml_reshape_4d(ctx, q, S_v, GGML_DELTA_NET_CHUNK, H_k * num_chunks, n_seqs);
    cb(q, "q_reshape", il);
    v = ggml_reshape_4d(ctx, q, S_v, GGML_DELTA_NET_CHUNK, H_v * num_chunks, n_seqs);
    cb(v, "v_reshape", il);
    g = ggml_reshape_4d(ctx, g, GGML_DELTA_NET_CHUNK, 1, H_k * num_chunks, n_seqs);
    cb(g, "g_reshape", il);
    struct ggml_tensor * g_cumsum = ggml_cumsum(ctx, g);
    cb(g_cumsum, "g_cumsum", il);
        
    struct ggml_tensor * gcs_i = ggml_cont_4d(ctx, g_cumsum, GGML_DELTA_NET_CHUNK, 1, num_chunks * H_v, n_seqs);  // [chunk_size, 1, n_tokens, n_seqs]
    struct ggml_tensor * gcs_j = ggml_cont_4d(ctx, g_cumsum, 1, GGML_DELTA_NET_CHUNK, num_chunks * H_v, n_seqs);  // [1, chunk_size, n_tokens, n_seqs]
    
    // Broadcast both tensors to [chunk_size, chunk_size, H_v, n_seqs]
    struct ggml_tensor * gcs_i_broadcast = ggml_repeat_4d(ctx, gcs_i, GGML_DELTA_NET_CHUNK, GGML_DELTA_NET_CHUNK, num_chunks * H_v, n_seqs);  // [chunk_size, 1, H_v, n_seqs] -> [chunk_size, chunk_size, H_v, n_seqs]
    struct ggml_tensor * gcs_j_broadcast = ggml_repeat_4d(ctx, gcs_j, GGML_DELTA_NET_CHUNK, GGML_DELTA_NET_CHUNK, num_chunks * H_v, n_seqs);  // [1, chunk_size, H_v, n_seqs] -> [chunk_size, chunk_size, H_v, n_seqs]
    
    struct ggml_tensor * decay_mask = ggml_sub(ctx, gcs_j_broadcast, gcs_i_broadcast); 
    cb(decay_mask, "sub", il);
    
    // Apply lower triangular mask to ensure attention is causal (only past tokens influence current)
    decay_mask = ggml_tri_keep(ctx, decay_mask, GGML_TRI_TYPE_LOWER_DIAG); 
    cb(decay_mask, "sub_tri", il);
    // Apply exponential to get the decay mask values
    decay_mask = ggml_exp(ctx, decay_mask);
    cb(decay_mask, "sub_tri_exp", il);
    // Apply lower triangular mask again to ensure only lower triangular values remain
    decay_mask = ggml_tri_keep(ctx, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
    cb(decay_mask, "decay_mask", il);

    struct ggml_tensor * kmulkbeta = ggml_mul_mat(ctx, ggml_cont(ctx, k), ggml_cont(ctx, k_beta));
    cb(kmulkbeta, "k_beta @ k_t ", il);
    
    struct ggml_tensor * k_decay = ggml_mul(ctx, kmulkbeta, decay_mask);
    cb(k_decay, "(k_beta @ k_t) * decay_mask", il);

    struct ggml_tensor * attn = ggml_neg(ctx, ggml_tri_keep(ctx, k_decay, GGML_TRI_TYPE_LOWER));
    cb(attn, "attn_in", il);

    // We'll be returning the result as a 1D tensor due to the dimensions mismatch of the state and output tensors
    const int64_t total_dims = (S_v * H_v * n_tokens * n_seqs) + (S_v * S_v * H_v * n_seqs);
    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, total_dims);

    ggml_set_op_params_i32(result, 0, H_v);
    ggml_set_op_params_i32(result, 1, S_k);
    ggml_set_op_params_i32(result, 2, S_v);
    ggml_set_op_params_i32(result, 3, n_tokens); // Pass original n_tokens

    result->op     = GGML_OP_DELTA_NET;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = g_cumsum;
    result->src[4] = state;
    result->src[5] = decay_mask;
    result->src[6] = v_beta;
    result->src[7] = k_beta;
    result->src[8] = attn;

    return result;
}

// delta_net_recurrent
struct ggml_tensor * llm_build_qwen3next::delta_net_recurrent(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * g,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state,
        bool                  use_qk_l2norm,
        float                 eps_norm,
        const int             il
    ) {
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));

    const int64_t S_k = q->ne[0];
    const int64_t H_k = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[2] == n_tokens);
    GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v * H_v && state->ne[2] == 1 && state->ne[3] == n_seqs);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && q->ne[3] == n_seqs);

    GGML_ASSERT(H_k == H_v); // we did a repeat to make sure this is the case

    cb(q, "q_prenorm", il);
    cb(k, "k_prenorm", il);

    if (use_qk_l2norm) {
        q = ggml_l2_norm(ctx, q, eps_norm);
        k = ggml_l2_norm(ctx, k, eps_norm);
    }

    cb(k, "k_postnorm", il);
    cb(q, "q_prescale", il);

    float scale = 1.0f / sqrtf(S_v);
    q = ggml_scale(ctx, q, scale);

    cb(beta, "beta_raw", il);
    beta = ggml_sigmoid(ctx, beta);

    cb(q, "q_postscale", il);
    cb(beta, "beta_sigmoid", il);

    // Reshape tensors for recurrent computation
    // From [S_k, H_k, n_tokens, n_seqs] to [S_k, n_tokens, H_k, n_seqs]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    cb(q, "q_reshape", il);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    cb(k, "k_reshape", il);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    cb(v, "v_reshape", il);
    
    beta = ggml_cont(ctx, ggml_permute(ctx, beta, 1, 2, 0, 3));
    cb(beta, "beta_reshape", il);

    g = ggml_cont(ctx, ggml_permute(ctx, g, 2, 0, 3, 1));
    cb(g, "g_permute", il);

    ggml_tensor * q_tokens = ggml_cont_4d(ctx, q, n_tokens, S_k, H_k, n_seqs);
    ggml_tensor * k_tokens = ggml_cont_4d(ctx, k, n_tokens, S_k, H_k, n_seqs);
    ggml_tensor * v_tokens = ggml_cont_4d(ctx, v, n_tokens, S_v, H_k, n_seqs);
    ggml_tensor * g_tokens = ggml_cont_4d(ctx, g, n_tokens, 1, H_k, n_seqs);
    ggml_tensor * beta_tokens = ggml_cont_4d(ctx, beta, n_tokens, 1, H_k, n_seqs);

    state = ggml_cont_4d(ctx, state, S_v, S_v, H_k, n_seqs);
    ggml_tensor * g_tokens_exp = ggml_exp(ctx, g_tokens);

    // Create result tensor with the same dimensions as delta_net
    const int64_t total_dims = (S_v * H_v * n_tokens * n_seqs) + (S_v * S_v * H_v * n_seqs);
    ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, total_dims);

    cb(q_tokens, "q_tokens", il);
    cb(k_tokens, "k_tokens", il);
    cb(v_tokens, "v_tokens", il);
    cb(g_tokens, "g_tokens", il);
    cb(beta_tokens, "beta_tokens", il);
    cb(g_tokens_exp, "g_tokens_exp", il);
    cb(state, "state_pre", il);

    // Set operation parameters
    ggml_set_op_params_i32(result, 0, H_v);
    ggml_set_op_params_i32(result, 1, S_k);
    ggml_set_op_params_i32(result, 2, S_v);
    ggml_set_op_params_i32(result, 3, n_tokens); // Pass original n_tokens

    // Set operation and source tensors
    result->op     = GGML_OP_DELTA_NET_RECURRENT;
    result->src[0] = q_tokens;
    result->src[1] = k_tokens;
    result->src[2] = v_tokens;
    result->src[3] = g_tokens_exp;
    result->src[4] = beta_tokens;
    result->src[5] = state;
    
    return result;
}


ggml_tensor * llm_build_qwen3next::build_qwen3next_linear_attn_layer(llm_graph_input_rs * inp,
                                                                     ggml_tensor *        cur,
                                                                     const llama_model &  model,
                                                                     const llama_ubatch & ubatch,
                                                                     int                  il) {
    // Gated Delta Net implementation using the new delta_net function
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner  = hparams.ssm_d_inner;
    
    const int64_t n_seqs   = ubatch.n_seqs;

    const int64_t head_k_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim  = d_inner / num_v_heads;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    ggml_tensor * mixed_qkvz = build_lora_mm(model.layers[il].ssm_in, cur);
    cb(mixed_qkvz, "linear_attn_mixed_qkvz", il);

    ggml_tensor * mixed_ba = build_lora_mm(model.layers[il].ssm_beta_alpha, cur);
    cb(mixed_ba, "linear_attn_mixed_ba", il);

    int64_t       qkvz_new_dim        = 2 * head_k_dim + 2 * head_v_dim * (num_v_heads / num_k_heads);
    ggml_tensor * mixed_qkvz_reshaped = ggml_cont_4d(ctx0, mixed_qkvz, qkvz_new_dim, num_k_heads, n_seq_tokens, n_seqs);

    // Reshape mixed_ba: [batch, seq_len, hidden_size] -> [batch, seq_len, num_k_heads, 2*num_v_heads/num_k_heads]
    int64_t       ba_new_dim        = 2 * num_v_heads / num_k_heads;
    ggml_tensor * mixed_ba_reshaped = ggml_cont_4d(ctx0, mixed_ba, ba_new_dim, num_k_heads, n_seq_tokens, n_seqs);

    // Split mixed_ba into b and a (beta and alpha parameters)
    int64_t split_sizes_ba[2] = {
        num_v_heads / num_k_heads,  // beta size
        num_v_heads / num_k_heads   // alpha size
    };

    ggml_tensor * b = ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[0], num_k_heads, n_seq_tokens, n_seqs,
                        mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3], 0);
    cb(b, "b", il);

    ggml_tensor * a = ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[1], num_k_heads, n_seq_tokens, n_seqs,
                        mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3], 
                        split_sizes_ba[0] * ggml_element_size(mixed_ba_reshaped));
    cb(a, "a", il);

    // Reshape b and a to merge head dimensions: [batch, seq_len, num_k_heads, num_v_heads/num_k_heads] -> [batch, seq_len, num_v_heads]
    ggml_tensor * beta  = ggml_cont_3d(ctx0, b, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * alpha = ggml_cont_3d(ctx0, a, num_v_heads, n_seq_tokens, n_seqs);

    GGML_ASSERT(ggml_nelements(beta) + ggml_nelements(alpha) == ggml_nelements(mixed_ba));

    ggml_tensor * alpha_biased = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);
    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    // Split mixed_qkvz into query, key, value, z
    int64_t split_sizes_qkvz[4] = {
        head_k_dim,                              // query size
        head_k_dim,                              // key size
        head_v_dim * num_v_heads / num_k_heads,  // value size
        head_v_dim * num_v_heads / num_k_heads   // z size
    };

    ggml_tensor * query = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[0], num_k_heads, n_seq_tokens, n_seqs, 
                                            mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3], 0);
    cb(query, "q", il);

    ggml_tensor * key = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[1], num_k_heads, n_seq_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                        split_sizes_qkvz[0] * sizeof(float));
    cb(key, "k", il);

    ggml_tensor * value = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[2], num_k_heads, n_seq_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                        (split_sizes_qkvz[0] + split_sizes_qkvz[1]) * sizeof(float));
    cb(value, "v", il);

    ggml_tensor * z = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[3], num_k_heads, n_seq_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                        (split_sizes_qkvz[0] + split_sizes_qkvz[1] + split_sizes_qkvz[2]) * sizeof(float));
    cb(z, "z", il);

    GGML_ASSERT(ggml_nelements(query) + ggml_nelements(key) + ggml_nelements(value) +
                    ggml_nelements(z) ==
                ggml_nelements(mixed_qkvz));

    // After creating query, key, and value_reshaped, reshape each to flatten the head dimensions
    // query: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim * num_k_heads, n_tokens, n_seqs]
    ggml_tensor * query_flat = ggml_cont_3d(ctx0, query, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
    cb(query_flat, "query_flat", il);

    // key: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim * num_k_heads, n_tokens, n_seqs]
    ggml_tensor * key_flat = ggml_cont_3d(ctx0, key, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
    cb(key_flat, "key_flat", il);

    // value_reshaped: [head_v_dim, num_v_heads, n_tokens, n_seqs] -> [head_v_dim * num_v_heads, n_tokens, n_seqs]
    ggml_tensor * value_flat = ggml_cont_3d(ctx0, value, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(value_flat, "value_flat", il);

    // Get convolution states from cache
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    bool use_precomputed_states = n_seq_tokens == 1 && mctx_cur->has_previous_state();

    // Build the convolution states tensor
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    // Now concatenate along the feature dimension (dim 0) to get [conv_dim, n_tokens, n_seqs]
    ggml_tensor * qkv_mixed = ggml_concat(ctx0, query_flat, key_flat, 0);
    qkv_mixed = ggml_concat(ctx0, qkv_mixed, value_flat, 0);
    cb(qkv_mixed, "qkv_mixed", il);

    qkv_mixed = ggml_permute(ctx0, qkv_mixed, 1, 0, 2, 3);
    cb(qkv_mixed, "qkv_mixed_permuted", il);

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    // Calculate convolution kernel size
    ggml_tensor * conv_kernel = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;
    //conv_kernel = ggml_permute(ctx0, conv_kernel, 0, 2, 1, 3);
    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);    
    cb(conv_input, "conv_input", il);

    // Update convolution state cache
    // Extract the last (conv_kernel_size - 1) states from conv_input
    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1], conv_input->nb[2],
            (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target = ggml_view_1d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels * n_seqs,
        kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));
    cb(conv_states_all, "conv_states_updated", il);

    // Apply convolution
    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel); //ggml_conv_1d_dw_f32(ctx0, conv_kernel, conv_input, 1, conv_kernel_size - 1, 1);
    cb(conv_output_proper, "conv_output_raw", il);

    // Remove the padding
    // ggml_tensor * conv_output_no_padding = ggml_view_4d(ctx0, conv_output, conv_output->ne[0] - (conv_kernel_size - 1), conv_output->ne[1], conv_output->ne[2], conv_output->ne[3],
    //                                                     conv_output->nb[1], conv_output->nb[2], conv_output->nb[3], 
    //                                                     (conv_kernel_size - 1) * ggml_element_size(conv_output));
    // cb(conv_output_no_padding, "conv_output_no_padding", il);

    // // Take only the first n_seq_tokens values
    // ggml_tensor * conv_output_proper = ggml_view_4d(ctx0, conv_output_no_padding, n_seq_tokens, conv_output_no_padding->ne[1], conv_output_no_padding->ne[2], conv_output_no_padding->ne[3],
    //                                                     conv_output_no_padding->nb[1], conv_output_no_padding->nb[2], conv_output_no_padding->nb[3], 0);
    // cb(conv_output_proper, "conv_output_proper", il);

    conv_output_proper = ggml_cont(ctx0, ggml_transpose(ctx0, conv_output_proper));
    cb(conv_output_proper, "conv_output_pre_silu", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = ggml_cont_2d(ctx0, ggml_transpose(ctx0, conv_output_silu), qkv_dim, n_seq_tokens * n_seqs);
    cb(conv_qkv_mix, "conv_qkv_mix", il);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_2d(ctx0, conv_qkv_mix, head_k_dim * num_k_heads, n_seq_tokens * n_seqs,
        conv_qkv_mix->nb[1], 0);
    cb(q_conv, "q_conv", il);
    ggml_tensor * k_conv = ggml_view_2d(ctx0, conv_qkv_mix, head_k_dim * num_k_heads, n_seq_tokens * n_seqs, 
        conv_qkv_mix->nb[1], head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));
    cb(k_conv, "k_conv", il);
    ggml_tensor * v_conv = ggml_view_2d(ctx0, conv_qkv_mix, head_v_dim * num_v_heads, n_seq_tokens * n_seqs, 
        conv_qkv_mix->nb[1], 2 * head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));
    cb(v_conv, "v_conv", il);
    
    // Unsqueeze them
    q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    beta  = ggml_cont_4d(ctx0, b, num_v_heads, 1, n_seq_tokens, n_seqs);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim * num_v_heads, 1, n_seqs);
    cb(state, "state_predelta", il);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        int64_t repeat_factor = num_v_heads / num_k_heads;

        // repeat interleave: reshape to (repeat part, 1, remaining part), do repeat, then reshape back
        ggml_tensor * q_reshaped = ggml_reshape_3d(ctx0, q_conv, head_k_dim, 1, num_k_heads * n_seq_tokens * n_seqs);
        ggml_tensor * k_reshaped = ggml_reshape_3d(ctx0, k_conv, head_k_dim, 1, num_k_heads * n_seq_tokens * n_seqs);
        
        // Repeat along the third dimension (the new dimension with size 1)
        ggml_tensor * q_repeated = ggml_repeat_4d(ctx0, q_reshaped, head_k_dim, repeat_factor, num_k_heads * n_seq_tokens * n_seqs, 1);
        ggml_tensor * k_repeated = ggml_repeat_4d(ctx0, k_reshaped, head_k_dim, repeat_factor, num_k_heads * n_seq_tokens * n_seqs, 1);
        
        // Reshape back to merge the head and repeat dimensions
        // From [head_dim, num_k_heads, repeat_factor, n_seq_tokens * n_seqs]
        // Back to [head_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs]
        q_conv = ggml_reshape_4d(ctx0, q_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
        k_conv = ggml_reshape_4d(ctx0, k_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    // Choose between delta_net and delta_net_recurrent based on generation mode
    ggml_tensor * attn_out;
    if (use_precomputed_states) {
        // Use delta_net_recurrent for single token generation
        attn_out = delta_net_recurrent(ctx0, q_conv, k_conv, v_conv, gate, beta, state, true, hparams.f_norm_rms_eps, il);
    } else {
        // Use regular delta_net for prompt processing
        attn_out = delta_net(ctx0, q_conv, k_conv, v_conv, gate, beta, state, true, hparams.f_norm_rms_eps, il);
    }
    cb(attn_out, "attn_out", il);

    // The tensors were concatenated 1d, so we need to extract them 1d as well
    const int64_t output_flat_size = head_v_dim * num_v_heads * n_seq_tokens * n_seqs;
    ggml_tensor * attn_out_1d =
        ggml_view_1d(ctx0, attn_out, output_flat_size, 0);
    cb(attn_out_1d, "attn_out_1d", il);

    ggml_tensor * attn_out_reshaped = ggml_cont_4d(ctx0, attn_out_1d, head_v_dim, n_seq_tokens, num_v_heads, n_seqs);
    cb(attn_out_1d, "attn_out_reshaped", il);
    
    ggml_tensor * attn_out_final = ggml_cont(ctx0, ggml_permute(ctx0, attn_out_reshaped, 0, 2, 1, 3));
    cb(attn_out_final, "attn_out_final", il);
   
    // Extract the state part (second part of the concatenated tensor)
    // State starts after n_tokens elements along dimension 1
    const int64_t state_flat_size = head_v_dim * head_v_dim * num_v_heads * n_seqs;
    
    ggml_tensor * state_1d = ggml_view_1d(ctx0, attn_out, state_flat_size, output_flat_size * ggml_element_size(attn_out));
    cb(state_1d, "state_1d", il);
    
    // Update the recurrent states
    ggml_build_forward_expand(gf, 
        ggml_cpy(ctx0, state_1d, ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
            kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    GGML_ASSERT(ggml_nelements(attn_out_1d) + ggml_nelements(state_1d) == ggml_nelements(attn_out));

    // Reshape both attn_out_final and z to 2D tensors for normalization
    // attn_out_final: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * attn_out_2d_final = ggml_reshape_2d(ctx0, ggml_cont(ctx0, attn_out_final), head_v_dim, num_v_heads * n_seq_tokens * n_seqs);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_cont_2d(ctx0, z, head_v_dim, num_v_heads * n_seq_tokens * n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_q3n_gated_norm(attn_out_2d_final, model.layers[il].ssm_norm, z_2d, il);
    
    // Reshape back to original dimensions: [n_heads * n_tokens * n_seqs, head_dim] -> [head_dim, n_heads, n_tokens, n_seqs]
    ggml_tensor * gated_output_4d = ggml_reshape_4d(ctx0, attn_out_norm, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, gated_output_4d, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_cont(ctx0, ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs));
    return cur;
}

ggml_tensor * llm_build_qwen3next::build_layer_ffn(ggml_tensor * cur, const llama_model & model, const int il) {

    // Check if this is an MoE layer
    if (model.layers[il].ffn_gate_inp != nullptr) {
        // MoE branch
        ggml_tensor * moe_out =
            build_moe_ffn(cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
                          model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps, nullptr, n_expert,
                          n_expert_used, LLM_FFN_SILU, true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
        cb(moe_out, "ffn_moe_out", il);

        // Add shared experts if present - following Qwen3Next reference implementation
        if (model.layers[il].ffn_up_shexp != nullptr) {
            ggml_tensor * ffn_shexp =
                build_ffn(cur, model.layers[il].ffn_up_shexp, NULL, NULL, model.layers[il].ffn_gate_shexp, NULL, NULL,
                          model.layers[il].ffn_down_shexp, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            // Apply shared expert gating as in the reference implementation
            // The shared expert has its own gate that is sigmoided
            // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
            ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
            cb(shared_gate, "shared_expert_gate", il);
            
            // Apply sigmoid to the gate
            shared_gate = ggml_sigmoid(ctx0, shared_gate);
            cb(shared_gate, "shared_expert_gate_sigmoid", il);
            
            // The gate needs to be broadcast to match the dimensions of ffn_shexp
            // ffn_shexp is [n_embd, n_tokens, 1, 1] and shared_gate is [1, n_tokens, 1, 1]
            // We need to repeat the gate along the feature dimension
            shared_gate = ggml_repeat(ctx0, shared_gate, ffn_shexp);
            cb(shared_gate, "shared_expert_gate_broadcast", il);
            
            // Apply the gate to the shared expert output
            ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
            cb(ffn_shexp, "ffn_shexp_gated", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
    } else {
        // Dense FFN branch
        cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
    }
    return cur;
}

