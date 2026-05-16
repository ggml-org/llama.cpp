#include "models.h"

template <>
llm_build_mimo_v2_asr<true>::llm_build_mimo_v2_asr(const llama_model & model, const llm_graph_params & params) 
    : llm_graph_context(params), model(model) {
    
    const int64_t audio_channel = model.hparams.rvq_codebook_count;
    const int64_t text_channel = 1;
    const int64_t stride = audio_channel + text_channel;
    const int64_t T = ubatch.n_tokens / stride;
    const int64_t group_size = model.hparams.mimo_group_size;

    ggml_tensor* unused_text_embd_full = build_inp_embd(model.tok_embd);
    ggml_tensor* inp_tokens = res->t_inp_tokens;
    ggml_tensor* token_matrix = ggml_reshape_2d(ctx0, inp_tokens, stride, T);

    // For later alignment, the preceding text input [text 0 0 0] is sliced at intervals of group_size
    ggml_tensor* text_ids_groups_view = ggml_view_2d(ctx0, token_matrix, 1, T / group_size, token_matrix->nb[1] * group_size, 0);
    ggml_tensor* text_ids_groups = ggml_reshape_1d(ctx0, ggml_cont(ctx0, text_ids_groups_view), T / group_size);
    ggml_tensor* text_embd = ggml_get_rows(ctx0, model.tok_embd, text_ids_groups);
    
    ggml_tensor* audio_ids_view = ggml_view_2d(ctx0, token_matrix, audio_channel, T, token_matrix->nb[1], ggml_element_size(token_matrix));
    ggml_tensor* audio_ids_t    = ggml_transpose(ctx0, audio_ids_view);
    ggml_tensor* audio_ids_cont = ggml_cont(ctx0, audio_ids_t);

    ggml_tensor* audio_mask = build_audio_mask(text_ids_groups);
    ggml_tensor* audio_enc = build_input_audio_encoder(audio_ids_cont, audio_mask);

    res->t_embd = ggml_add(ctx0, text_embd, audio_enc);
    cb(res->t_embd, "result_embd", -1);

    ggml_build_forward_expand(gf, res->t_embd);
}

template <>
llm_build_mimo_v2_asr<false>::llm_build_mimo_v2_asr(const llama_model & model, const llm_graph_params & params) 
    : llm_graph_context(params), model(model) {
    // ==========================================================
    // Reuse the Qwen2 model
    // ==========================================================
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    
    // Feed multimodal features directly into the backbone network
    ggml_tensor * inpL = build_inp_embd(model.tok_embd); 
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    if (model.output_b != nullptr) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

template <bool is_enc>
ggml_tensor* llm_build_mimo_v2_asr<is_enc>::build_audio_mask(ggml_tensor* text_ids_groups){
    const int64_t T_groups = text_ids_groups->ne[0];

    ggml_tensor * ids_f32  = ggml_cast(ctx0, text_ids_groups, GGML_TYPE_F32);
    float empty_val_f32 = (float)model.hparams.mimo_text_empty_idx;

    ggml_tensor * diff = ggml_scale_bias(ctx0, ids_f32, 1.0f, -empty_val_f32);
    ggml_tensor * text_mask = ggml_step(ctx0, ggml_abs(ctx0, diff));
    ggml_tensor * audio_mask = ggml_scale_bias(ctx0, text_mask, -1.0f, 1.0f);

    return ggml_reshape_3d(ctx0, audio_mask, 1, 1, T_groups);
}

template <bool is_enc>
ggml_tensor* llm_build_mimo_v2_asr<is_enc>::build_input_audio_encoder(ggml_tensor* audio_ids_cont, ggml_tensor* audio_mask){
    const int64_t T = audio_ids_cont->ne[0];
    const int64_t group_size = model.hparams.mimo_group_size;
    const int64_t n_embd_audio = model.hparams.n_embd_audio;
    const int64_t T_groups = T / group_size;
    const int64_t n_head_audio = model.hparams.n_head_audio;
    const int64_t head_dim = n_embd_audio / n_head_audio;
    const float   rope_freq_base = model.hparams.rope_freq_base_train;
    
    ggml_tensor* audio_pos = ggml_view_1d(ctx0, build_inp_pos(), group_size, 0);
    ggml_tensor* audio_embds = nullptr;
    
    for(int i = 0; i < model.hparams.rvq_codebook_count; ++i){
        size_t offset = i * T * ggml_element_size(audio_ids_cont);
        ggml_tensor* cur_ids_1d = ggml_view_1d(ctx0, audio_ids_cont, T, offset);

        char tensor_name[128];
        snprintf(tensor_name, sizeof(tensor_name), "a.mimo_speech_embd.%d.weight", i);
        ggml_tensor * embd_weight = (ggml_tensor *)model.get_tensor(tensor_name);
        ggml_tensor * cur_embed = ggml_get_rows(ctx0, embd_weight, cur_ids_1d);
        
        audio_embds = (audio_embds == nullptr) ? cur_embed : ggml_add(ctx0, audio_embds, cur_embed);
    }
    ggml_tensor* cur = ggml_reshape_3d(ctx0, audio_embds, n_embd_audio, group_size, T_groups);
    cur = ggml_mul(ctx0, cur, audio_mask);

    for(int i = 0; i < model.hparams.n_audio_layer; ++i){
        ggml_tensor* inpL1 = cur; 

        // RMSNorm
        cur = ggml_rms_norm(ctx0, cur, model.hparams.f_norm_rms_eps);
        ggml_tensor* attn_norm_weight = model.layers[i].attn_norm_enc;
        cur = ggml_mul(ctx0, cur, attn_norm_weight);

        // Q/K/V MHA
        ggml_tensor* Q_cur = build_lora_mm(model.layers[i].wq_enc, cur);
        ggml_tensor* K_cur = build_lora_mm(model.layers[i].wk_enc, cur);
        ggml_tensor* V_cur = build_lora_mm(model.layers[i].wv_enc, cur);
        Q_cur = ggml_add(ctx0, Q_cur, model.layers[i].bq_enc);
        K_cur = ggml_add(ctx0, K_cur, model.layers[i].bk_enc);
        V_cur = ggml_add(ctx0, V_cur, model.layers[i].bv_enc);

        Q_cur = ggml_reshape_4d(ctx0, Q_cur, head_dim, n_head_audio, group_size, T_groups);
        K_cur = ggml_reshape_4d(ctx0, K_cur, head_dim, n_head_audio, group_size, T_groups);
        V_cur = ggml_reshape_4d(ctx0, V_cur, head_dim, n_head_audio, group_size, T_groups);

        Q_cur = ggml_rope_ext(ctx0, Q_cur, audio_pos, nullptr, head_dim, LLAMA_ROPE_TYPE_NEOX, 0, rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K_cur = ggml_rope_ext(ctx0, K_cur, audio_pos, nullptr, head_dim, LLAMA_ROPE_TYPE_NEOX, 0, rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        
        // permute for flashattn
        Q_cur = ggml_cont(ctx0, ggml_permute(ctx0, Q_cur, 0, 2, 1, 3));
        K_cur = ggml_cont(ctx0, ggml_permute(ctx0, K_cur, 0, 2, 1, 3));
        V_cur = ggml_cont(ctx0, ggml_permute(ctx0, V_cur, 0, 2, 1, 3));

        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx0, Q_cur, K_cur, V_cur, nullptr, 1.0f/sqrtf(head_dim), 0.0f, 0.0f);
        cur = ggml_reshape_3d(ctx0, attn_out, n_embd_audio, group_size, T_groups);
        cur = build_lora_mm(model.layers[i].wo_enc, cur);
        cur = ggml_add(ctx0, cur, inpL1);

        // FFN
        ggml_tensor* inpL2 = cur;
        
        cur = ggml_rms_norm(ctx0, cur, model.hparams.f_norm_rms_eps);
        cur = ggml_mul(ctx0, cur, model.layers[i].ffn_norm_enc);

        ggml_tensor* gate = build_lora_mm(model.layers[i].ffn_gate_enc, cur);
        ggml_tensor* up = build_lora_mm(model.layers[i].ffn_up_enc, cur);
        gate = ggml_silu(ctx0, gate);
        ggml_tensor* act = ggml_mul(ctx0, gate, up);
        cur = build_lora_mm(model.layers[i].ffn_down_enc, act);

        cur = ggml_add(ctx0, cur, inpL2);
    }
    cur = ggml_rms_norm(ctx0, cur, model.hparams.f_norm_rms_eps);
    cur = ggml_mul(ctx0, cur, model.output_norm_enc);

    cur = ggml_mul(ctx0, cur, audio_mask);

    // downcast
    cur = ggml_reshape_2d(ctx0, cur, n_embd_audio * group_size, T_groups);
    ggml_tensor* speech_downcast_weight = (ggml_tensor *)model.get_tensor("a.mimo_speech_downcast.weight");
    cur = build_lora_mm(speech_downcast_weight, cur);

    return cur;
}

template struct llm_build_mimo_v2_asr<true>;
template struct llm_build_mimo_v2_asr<false>;