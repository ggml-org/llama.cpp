#include "models.h"

void llama_model_k2_horizon::load_arch_hparams(llama_model_loader & ml) {
    // generic
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_GROUPNORM_GROUPS, hparams.n_norm_groups, false);

    hparams.f_norm_group_eps = hparams.f_norm_rms_eps;
    if (hparams.n_norm_groups == 0) hparams.n_norm_groups = 1;
    
    // moe
    if (hparams.n_expert > 0) {
        ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
        ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead, false);
        ml.get_key(LLM_KV_MOE_EVERY_N_LAYERS, hparams.moe_every_n_layers, false);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared, false);
        ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale, false);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm, false);
        ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);
        if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
        }
    }

    // mova
    ml.get_key(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT, hparams.n_value_expert, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, hparams.n_value_expert_used, false);
    if (hparams.n_value_expert > 0) {
        GGML_ASSERT(hparams.n_value_expert <= LLAMA_MAX_EXPERTS);
        GGML_ASSERT(hparams.n_value_expert_used > 0);
        GGML_ASSERT(hparams.n_value_expert_used <= hparams.n_value_expert);
    }
    else {
        GGML_ASSERT(hparams.n_value_expert_used == 0);
    }

    // model size info
    if (hparams.n_layer() == 28 && hparams.n_embd == 1536) {
        type = LLM_TYPE_1B;
    }
    else if (hparams.n_layer() == 48 && hparams.n_embd == 2560) {
        type = LLM_TYPE_36B;
    }
    else {
        type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_k2_horizon::load_arch_tensors(llama_model_loader & ml) {
    GGML_UNUSED(ml);
    LLAMA_LOAD_LOCALS; // initializing variables basically

    // embeddings
    tok_embd = create_tensor(
        tn(LLM_TENSOR_TOKEN_EMBD, "weight"),
        {n_embd, n_vocab},
        0
    );

    // final norm and output projection
    output_norm = create_tensor(
        tn(LLM_TENSOR_OUTPUT_NORM, "weight"),
        {n_embd},
        0
    );

    // output
    output = create_tensor(
        tn(LLM_TENSOR_OUTPUT, "weight"),
        {n_embd, n_vocab},
        TENSOR_NOT_REQUIRED // can be tied with embedding (indicated by tensor not found in .gguf). see next conditional
    );
    if (output == nullptr) {
        output = create_tensor(
            tn(LLM_TENSOR_TOKEN_EMBD, "weight"),
            {n_embd, n_vocab},
            TENSOR_DUPLICATED
        );
    }

    for (int i = 0; i < n_layer; i++){
        auto & layer = layers[i];
        const bool is_moe_layer = n_expert > 0 && static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead;
        const bool is_mova_layer = is_moe_layer && hparams.n_value_expert > 0; // in the architecture, if mova is moe as well
        
        // attn normalization
        layer.attn_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_NORM, "weight", i),
            {n_embd},
            0
        );
        
        // query and key tensors, always dense. and their optional normalization
        // query
        layer.wq = create_tensor(
            tn(LLM_TENSOR_ATTN_Q, "weight", i),
            {n_embd, n_embd_head_k * n_head},
            0
        );
        layer.attn_q_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i),
            {n_embd_head_k * n_head},
            TENSOR_NOT_REQUIRED
        );

        // key
        layer.wk = create_tensor(
            tn(LLM_TENSOR_ATTN_K, "weight", i),
            {n_embd, n_embd_k_gqa},
            0
        );
        layer.attn_k_norm = create_tensor(
            tn(LLM_TENSOR_ATTN_K_NORM, "weight", i),
            {n_embd_k_gqa},
            TENSOR_NOT_REQUIRED
        );

        // value tensors, possible MoVA
        if (is_mova_layer) {
            layer.attn_v_gate = create_tensor(
                tn(LLM_TENSOR_ATTN_V_GATE, "weight", i),
                {n_embd, hparams.n_value_expert},
                0
            );
            layer.attn_v_gate_b = create_tensor(
                tn(LLM_TENSOR_ATTN_V_GATE, "bias", i),
                {hparams.n_value_expert},
                TENSOR_NOT_REQUIRED
            );
            layer.attn_v_exps = create_tensor(
                tn(LLM_TENSOR_ATTN_V_EXPS, "weight", i),
                {n_embd, n_embd_v_gqa, hparams.n_value_expert},
                0
            );
        }
        else {
            layer.wv = create_tensor(
                tn(LLM_TENSOR_ATTN_V, "weight", i),
                {n_embd, n_embd_v_gqa},
                0
            );
        }

        // attn output projection
        layer.wo = create_tensor(
            tn(LLM_TENSOR_ATTN_OUT, "weight", i),
            {n_embd_head_v * n_head, n_embd},
            0
        );

        // optional softplus gate
        layer.wqkv_gate = create_tensor(
            tn(LLM_TENSOR_ATTN_GATE, "weight", i),
            {n_embd, n_embd_head_v * n_head},
            TENSOR_NOT_REQUIRED
        );

        // FFN normalization
        layer.ffn_norm = create_tensor(
            tn(LLM_TENSOR_FFN_NORM, "weight", i),
            {n_embd},
            0
        );

        // MoE stuff
        if (is_moe_layer) {
            const uint32_t n_ff_exp = hparams.n_ff_exp(i);
            if (n_ff_exp == 0){
                throw std::runtime_error("K2 MoE layer requires expert_feed_forward_length");
            }

            // moe router and it's optional bias
            layer.ffn_gate_inp = create_tensor(
                tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),
                {n_embd, n_expert},
                0
            );
            layer.ffn_exp_probs_b = create_tensor(
                tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i),
                {n_expert},
                TENSOR_NOT_REQUIRED
            );

            // routed experts (up, gate, and down)
            layer.ffn_up_exps = create_tensor(
                tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i),
                {n_embd, n_ff_exp, n_expert},
                0
            );
            layer.ffn_gate_exps = create_tensor(
                tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i),
                {n_embd, n_ff_exp, n_expert},
                0
            );
            layer.ffn_down_exps = create_tensor(
                tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                {n_ff_exp, n_embd, n_expert},
                0
            );

            // shared experts (always evaluated)
            if (hparams.n_expert_shared > 0) {
                int64_t n_ff_shexp;
                if (hparams.n_ff_shexp > 0) {
                    n_ff_shexp = hparams.n_ff_shexp;
                } else {
                    n_ff_shexp = n_ff_exp * hparams.n_expert_shared;
                }

                // up gate down
                layer.ffn_up_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i),
                    {n_embd, n_ff_shexp},
                    0
                );
                layer.ffn_gate_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i),
                    {n_embd, n_ff_shexp},
                    0
                );
                layer.ffn_down_shexp = create_tensor(
                    tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i),
                    {n_ff_shexp, n_embd},
                    0
                );
            }
        }
        else {
            // ordinary up gate down
            layer.ffn_up = create_tensor(
                tn(LLM_TENSOR_FFN_UP, "weight", i),
                {n_embd, n_ff},
                0
            );
            layer.ffn_gate = create_tensor(
                tn(LLM_TENSOR_FFN_GATE, "weight", i),
                {n_embd, n_ff},
                0
            );
            layer.ffn_down = create_tensor(
                tn(LLM_TENSOR_FFN_DOWN, "weight", i),
                {n_ff, n_embd},
                0
            );
        }
        
    }

}

// helper for grouped RMS norm
static ggml_tensor * k2_horizon_group_rms_norm(
    ggml_context * ctx,
    ggml_tensor * cur,
    ggml_tensor * weight,
    int64_t n_groups,
    float eps
) {
    GGML_ASSERT(n_groups > 0);
    GGML_ASSERT(cur->ne[0] % n_groups == 0);

    const int64_t n_embd = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];

    // separate embeddings into groups
    cur = ggml_reshape_3d(
        ctx,
        cur,
        n_embd / n_groups,
        n_groups,
        n_tokens
    );

    // norm it
    cur = ggml_rms_norm(ctx, cur, eps);

    // bring back shape
    cur = ggml_reshape_2d(ctx, cur, n_embd, n_tokens);

    // apply the learned normalization weights
    if (weight != nullptr) {
        cur = ggml_mul(ctx, cur, weight);
    }

    return cur;
}

ggml_tensor * llama_model_k2_horizon::graph::build_routed_value(
    const llama_layer & layer,
    ggml_tensor * cur,
    int il
) const {
    const int64_t n_embd     = cur->ne[0];
    const int64_t n_tokens   = cur->ne[1];
    const int64_t n_embd_gqa = hparams.n_embd_v_gqa(il);
    const int64_t n_values   = hparams.n_value_expert;
    const int64_t n_used     = hparams.n_value_expert_used;

    GGML_ASSERT(layer.attn_v_gate != nullptr);
    GGML_ASSERT(layer.attn_v_exps != nullptr);
    GGML_ASSERT(n_values > 0);
    GGML_ASSERT(n_used > 0);

    // router. logits and probs
    ggml_tensor * logits = build_lora_mm(layer.attn_v_gate, cur);
    ggml_tensor * probs = nullptr;
    
    // probs
    llama_expert_gating_func_type gating_func = static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func);
    switch(gating_func){
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            probs = ggml_soft_max(ctx0, logits);
            break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            probs = ggml_sigmoid(ctx0, logits);
            break;
        default:
            GGML_ABORT("Unsupported K2 Horizon value-router gating function");
    }

    // selection probs
    ggml_tensor * selection_probs = probs;
    if (layer.attn_v_gate_b != nullptr){
        selection_probs = ggml_add(ctx0, probs, layer.attn_v_gate_b);
        cb(selection_probs, "v_moe_probs_biased", il);
    }

    // select expert values
    ggml_tensor * selected_value_experts = ggml_argsort_top_k(ctx0, selection_probs, n_used);
    
    // reshaping and selecting the weights (probs) of the selected experts
    probs = ggml_reshape_3d(ctx0, probs, 1, n_values, n_tokens);
    ggml_tensor * selected_weights = ggml_get_rows(ctx0, probs, selected_value_experts);

    // if weights of value experts are to be normalized
    if (hparams.expert_weights_norm) {
        selected_weights = ggml_reshape_2d(ctx0, selected_weights, n_used, n_tokens);
        ggml_tensor * selected_weights_sum = ggml_sum_rows(ctx0, selected_weights);
        selected_weights_sum = ggml_clamp(ctx0, selected_weights_sum, 6.103515625e-5f, INFINITY);
        selected_weights = ggml_div(ctx0, selected_weights, selected_weights_sum);
        selected_weights = ggml_reshape_3d(ctx0, selected_weights, 1, n_used, n_tokens);
        cb(selected_weights, "v_moe_weights_norm", il);
    }

    // scaling
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        selected_weights = ggml_scale(ctx0, selected_weights, hparams.expert_weights_scale);
        cb(selected_weights, "v_moe_weights_scaled", il);
    }

    // labeling
    cb(logits, "v_moe_logits", il); 
    cb(probs, "v_moe_probs", il);
    cb(selected_value_experts->src[0], "v_moe_argsort", il);
    cb(selected_value_experts, "v_moe_topk", il);
    cb(selected_weights, "v_moe_weights", il);

    ggml_tensor * value_inp = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    // computing only on selected experts (the _id in the api)
    ggml_tensor * values = build_lora_mm_id(layer.attn_v_exps, value_inp, selected_value_experts);
    values = ggml_silu(ctx0, values);
    values = ggml_mul(ctx0, values, selected_weights);
    cb(values, "v_moe_weighted", il);

    // sum the multiple value outputs
    ggml_tensor * value_parts[LLAMA_MAX_EXPERTS] = {};
    for(int64_t i = 0; i < n_used; i++) {
        value_parts[i] = ggml_view_2d(ctx0, values, n_embd_gqa, n_tokens, values->nb[2], i * values->nb[1]);
    }
    ggml_tensor * value_out = value_parts[0];
    for (int64_t i = 1; i < n_used; ++i) {
        value_out = ggml_add(ctx0, value_out, value_parts[i]);
    }
    
    // making it contiguous in case it isn't (for one expert only)
    if (n_used == 1) value_out = ggml_cont(ctx0, value_out);

    cb(value_out, "Vcur_routed", il);
    return value_out;
}

llama_model_k2_horizon::graph::graph(
    const llama_model & model,
    const llm_graph_params & params
) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    
    // initialization or placeholders for computational artifacts
    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
        ggml_tensor * inpSA = inpL; // for residuals

        const bool is_moe_layer = n_expert > 0 && static_cast<uint32_t>(il) >= hparams.n_layer_dense_lead;
        const bool is_mova_layer = is_moe_layer && hparams.n_value_expert > 0;

        // ============ grouped rms norm
        cur = k2_horizon_group_rms_norm(
            ctx0,
            inpL,
            model.layers[il].attn_norm,
            hparams.n_norm_groups,
            hparams.f_norm_rms_eps
        );
        cb(cur, "attn_norm", il);

        // ============ setup attention tensors
        ggml_tensor * attn_inp = cur;

        // query
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
        if (model.layers[il].attn_q_norm != nullptr) {
            Qcur = k2_horizon_group_rms_norm(
                ctx0,
                Qcur,
                model.layers[il].attn_q_norm,
                n_head,
                hparams.f_norm_rms_eps
            );
        }

        // key
        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
        if (model.layers[il].attn_k_norm != nullptr) {
            Kcur = k2_horizon_group_rms_norm(
                ctx0,
                Kcur,
                model.layers[il].attn_k_norm,
                n_head_kv,
                hparams.f_norm_rms_eps
            );
        }

        // value
        ggml_tensor * Vcur;
        if (is_mova_layer) {
            Vcur = build_routed_value(model.layers[il], cur, il); // handle MoVA
        }
        else {
            Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
        }

        // reshaping
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        // applying RoPE
        Qcur = ggml_rope_ext(
            ctx0,
            Qcur,
            inp_pos,
            nullptr,
            n_rot,
            rope_type,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow
        );
        Kcur = ggml_rope_ext(
            ctx0,
            Kcur,
            inp_pos,
            nullptr,
            n_rot,
            rope_type,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow
        );

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        
        // ============ attention (with and without gating)
        const float kq_scale = 1.0f / sqrtf(static_cast<float>(n_embd_head));
        if(model.layers[il].wqkv_gate == nullptr){ // without gating
            cur = build_attn(
                inp_attn,
                model.layers[il].wo,
                model.layers[il].wo_b,
                model.layers[il].wo_s,
                Qcur,
                Kcur,
                Vcur,
                nullptr, // attention score bias
                nullptr, // attn sink
                nullptr, // MLA value transformation
                kq_scale,
                il
            );
        }
        else { // with gating
            // no output yet
            cur = build_attn(
                inp_attn,
                nullptr,
                nullptr,
                nullptr,
                Qcur,
                Kcur,
                Vcur,
                nullptr,
                nullptr,
                nullptr,
                kq_scale,
                il
            );

            // building the gate
            constexpr float LN2 = 0.6931471805599453f;
            constexpr float ONE_OVER_LN2 = 1.4426950408889634f;

            ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, attn_inp, model.layers[il].wqkv_gate_s);
            gate = ggml_scale(ctx0, gate, LN2);
            gate = ggml_softplus(ctx0, gate);
            gate = ggml_scale(ctx0, gate, ONE_OVER_LN2);

            // applying the gate
            cur = ggml_mul(ctx0, cur, gate);
            
            // projection
            cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);

            // bias
            if (model.layers[il].wo_b != nullptr) {
                cur = ggml_add(ctx0, cur, model.layers[il].wo_b);
            }
        }

        // ============ output layer, and take (usually) last token for generation
        if (il == n_layer - 1 && inp_out_ids != nullptr) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids); // pull the same positions for inpSA
        }

        // ============ add residuals
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // ============ group RMSNorm before FFN
        cur = k2_horizon_group_rms_norm(
            ctx0,
            ffn_inp,
            model.layers[il].ffn_norm,
            hparams.n_norm_groups,
            hparams.f_norm_rms_eps
        );
        cb(cur, "ffn_norm", il);

        // ============ Mixture of Experts
        if (is_moe_layer) {
            ggml_tensor * moe_out = build_moe_ffn(
                cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert,
                n_expert_used,
                LLM_FFN_SILU,
                hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
                il
            );

            // shared experts
            if (model.layers[il].ffn_gate_shexp != nullptr){
                ggml_tensor * shared_moe_out = build_ffn(
                    cur,
                    model.layers[il].ffn_up_shexp,
                    nullptr,
                    nullptr,
                    model.layers[il].ffn_gate_shexp,
                    nullptr,
                    nullptr,
                    model.layers[il].ffn_down_shexp,
                    nullptr,
                    nullptr,
                    nullptr,
                    LLM_FFN_SILU,
                    LLM_FFN_PAR,
                    il
                );
                cur = ggml_add(ctx0, moe_out, shared_moe_out);
            }
            else{
                cur = moe_out;
            }
        }
        else { // normal non moe FFN
            cur = build_ffn(
                cur,
                model.layers[il].ffn_up,
                nullptr,
                nullptr,
                model.layers[il].ffn_gate,
                nullptr,
                nullptr,
                model.layers[il].ffn_down,
                nullptr,
                nullptr,
                nullptr,
                LLM_FFN_SILU,
                LLM_FFN_PAR,
                il
            );
        }
        cb(cur, "ffn_out", il);
        
        // ============ FFN residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        
        // for next layer
        inpL = cur;
    }

    // final group rms norm. also becomes last layer embedding
    cur = k2_horizon_group_rms_norm(
        ctx0,
        inpL,
        model.output_norm,
        hparams.n_norm_groups,
        hparams.f_norm_rms_eps
    );
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // ============ vocab projection. also becomes logits
    cur = build_lora_mm(model.output, cur,model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // build everything
    ggml_build_forward_expand(gf, cur);
}


std::unique_ptr<llm_graph_context> llama_model_k2_horizon::build_arch_graph (
    const llm_graph_params & params
) const {
    return std::make_unique<graph>(*this, params);
}
