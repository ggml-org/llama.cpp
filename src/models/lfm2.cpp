#include "models.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-hybrid.h"

void llama_model_lfm2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        hparams.recurrent_layer_arr[il] = hparams.n_head_kv(il) == 0;
    }
    hparams.n_layer_dense_lead = hparams.n_layer;
    switch (hparams.n_ff()) {
        case  4608: type = LLM_TYPE_350M; break;
        case  6912: type = LLM_TYPE_700M; break;
        case  8192: type = LLM_TYPE_1_2B; break;
        case 10752: type = LLM_TYPE_2_6B; break;
        default:    type = LLM_TYPE_UNKNOWN;
    }
    if (const auto is_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false); is_swa && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        for (uint32_t il = 0; il < hparams.n_layer; ++il) {
            hparams.swa_layers[il] = !hparams.recurrent_layer_arr[il];
        }
    }
}

void llama_model_lfm2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM_LFM2, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const bool is_moe_layer = i >= static_cast<int>(hparams.n_layer_dense_lead);

        // ffn/moe is same for transformer and conv layers
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        if (is_moe_layer) {
            GGML_ASSERT(n_expert && n_expert_used);
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),  {n_embd, n_expert}, 0);
            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, hparams.n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {hparams.n_ff_exp,   n_embd, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i),   {n_embd, hparams.n_ff_exp, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        } else {  // dense
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        }

        // for operator_norm
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (!hparams.is_recurrent(i)) {
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
            GGML_ASSERT(n_embd_v_gqa == n_embd_k_gqa);

            create_tensor_qkv(layer, i, n_embd, n_embd, hparams.n_embd_k_gqa(i), hparams.n_embd_v_gqa(i), 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        } else {
            layer.shortconv.conv     = create_tensor(tn(LLM_TENSOR_SHORTCONV_CONV,    "weight", i), {hparams.n_shortconv_l_cache, n_embd}, 0);
            layer.shortconv.in_proj  = create_tensor(tn(LLM_TENSOR_SHORTCONV_INPROJ,  "weight", i), {n_embd, 3 * n_embd}, 0);
            layer.shortconv.out_proj = create_tensor(tn(LLM_TENSOR_SHORTCONV_OUTPROJ, "weight", i), {n_embd, n_embd}, 0);
        }
    }

    // for LFM2-ColBert-350M
    dense_2_out_layers   = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "weight"), {n_embd, hparams.n_embd_out()}, TENSOR_NOT_REQUIRED);
    dense_2_out_layers_b = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "bias"),   {hparams.n_embd_out()        }, TENSOR_NOT_REQUIRED);
}

std::unique_ptr<llm_graph_context> llama_model_lfm2::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
        return std::make_unique<graph<true>>(*this, params);
    } else {
        return std::make_unique<graph<false>>(*this, params);
    }
}

template <bool iswa>
llama_model_lfm2::graph<iswa>::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    using inp_hybrid_type = std::conditional_t<iswa, llm_graph_input_mem_hybrid_iswa,  llm_graph_input_mem_hybrid>;
    using inp_attn_type   = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa,     llm_graph_input_attn_kv>;
    using mem_hybrid_ctx  = std::conditional_t<iswa, llama_memory_hybrid_iswa_context, llama_memory_hybrid_context>;

    // lambda helpers for readability
    auto build_dense_feed_forward = [&model, this](ggml_tensor * cur, int il) -> ggml_tensor * {
        GGML_ASSERT(!model.layers[il].ffn_up_b);
        GGML_ASSERT(!model.layers[il].ffn_gate_b);
        GGML_ASSERT(!model.layers[il].ffn_down_b);
        return build_ffn(cur,
            model.layers[il].ffn_up, NULL, NULL,
            model.layers[il].ffn_gate, NULL, NULL,
            model.layers[il].ffn_down, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    };
    auto build_moe_feed_forward = [&model, this](ggml_tensor * cur, int il) -> ggml_tensor * {
        return build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
                il);
    };
    auto build_attn_block = [&model, this](ggml_tensor *   cur,
                                           ggml_tensor *   inp_pos,
                                           inp_attn_type * inp_attn,
                                           int             il) -> ggml_tensor * {
        GGML_ASSERT(hparams.n_embd_v_gqa(il) == hparams.n_embd_k_gqa(il));
        const auto n_embd_head = hparams.n_embd_head_v();
        const auto n_head_kv   = hparams.n_head_kv(il);

        auto [q, k, v] = build_qkv(model.layers[il], cur,
                n_embd_head, n_head, n_head_kv, il);

        // qk norm
        q = build_norm(q, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(q, "model.layers.{}.self_attn.q_layernorm", il);
        k = build_norm(k, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(k, "model.layers.{}.self_attn.k_layernorm", il);

        // RoPE
        q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                          attn_factor, beta_fast, beta_slow);
        k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                          attn_factor, beta_fast, beta_slow);

        cur = build_attn(inp_attn,
                model.layers[il].wo, NULL, model.layers[il].wo_s,
                q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);

        cb(cur, "model.layers.{}.self_attn.out_proj", il);

        return cur;
    };
    auto build_shortconv_block = [&model, this](ggml_tensor *        cur,
                                                llm_graph_input_rs * inp_recr,
                                                int                  il) -> ggml_tensor * {
        const auto * mctx_cur = static_cast<const mem_hybrid_ctx *>(mctx)->get_recr();
        const uint32_t kv_head      = mctx_cur->get_head();
        const int64_t  n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t  n_seqs       = ubatch.n_seqs;
        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs());
        GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

        GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
        const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;

        // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

        auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
        cb(bcx, "model.layers.{}.conv.in_proj", il);

        constexpr auto n_chunks = 3;
        GGML_ASSERT(bcx->ne[0] % n_chunks == 0);
        const auto chunk_size = bcx->ne[0] / n_chunks;
        auto *     b          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                             0 * chunk_size * ggml_element_size(bcx));
        auto *     c          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                             1 * chunk_size * ggml_element_size(bcx));
        auto *     x          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                             2 * chunk_size * ggml_element_size(bcx));

        auto * bx = ggml_transpose(ctx0, ggml_mul(ctx0, b, x));

        // read conv state
        auto * conv_state = mctx_cur->get_r_l(il);
        auto * conv_rs    = build_rs(inp_recr, conv_state, hparams.n_embd_r(), n_seqs);
        auto * conv       = ggml_reshape_3d(ctx0, conv_rs, d_conv, hparams.n_embd, n_seqs);

        bx = ggml_concat(ctx0, conv, bx, 0);
        GGML_ASSERT(bx->ne[0] > conv->ne[0]);

        // last d_conv columns is a new conv state
        auto * new_conv = ggml_view_3d(ctx0, bx, conv->ne[0], bx->ne[1], bx->ne[2], bx->nb[1], bx->nb[2],
                                       (bx->ne[0] - conv->ne[0]) * ggml_element_size(bx));
        GGML_ASSERT(ggml_are_same_shape(conv, new_conv));

        // write new conv conv state
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_conv,
                                               ggml_view_1d(ctx0, conv_state, ggml_nelements(new_conv),
                                                            kv_head * d_conv * n_embd * ggml_element_size(new_conv))));

        auto * conv_kernel = model.layers[il].shortconv.conv;
        auto * conv_out    = ggml_ssm_conv(ctx0, bx, conv_kernel);
        cb(conv_out, "model.layers.{}.conv.conv", il);

        auto * y = ggml_mul(ctx0, c, conv_out);
        y        = build_lora_mm(model.layers[il].shortconv.out_proj, y);
        cb(y, "model.layers.{}.conv.out_proj", il);
        // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
        y = ggml_reshape_2d(ctx0, y, y->ne[0], n_seq_tokens * n_seqs);

        return y;
    };

    // actual graph construction starts here
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.embed_tokens", -1);

    ggml_build_forward_expand(gf, cur);

    inp_hybrid_type * inp_hybrid = nullptr;
    if constexpr (iswa) {
        inp_hybrid = build_inp_mem_hybrid_iswa();
    } else {
        inp_hybrid = build_inp_mem_hybrid();
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const bool is_moe_layer = il >= static_cast<int>(hparams.n_layer_dense_lead);

        auto * prev_cur = cur;
        cur             = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "model.layers.{}.operator_norm", il);

        cur = hparams.is_recurrent(il) ? build_shortconv_block(cur, inp_hybrid->get_recr(), il) :
                                         build_attn_block(cur, inp_pos, inp_hybrid->get_attn(), il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            prev_cur = ggml_get_rows(ctx0, prev_cur, inp_out_ids);
        }

        cur = ggml_add(ctx0, prev_cur, cur);

        auto * ffn_norm_out = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(ffn_norm_out, "model.layers.{}.ffn_norm", il);

        ggml_tensor * ffn_out =
            is_moe_layer ? build_moe_feed_forward(ffn_norm_out, il) : build_dense_feed_forward(ffn_norm_out, il);
        cb(ffn_norm_out, "model.layers.{}.ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Explicit template instantiations
template struct llama_model_lfm2::graph<true>;
template struct llama_model_lfm2::graph<false>;
