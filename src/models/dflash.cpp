#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                  hparams.f_logit_scale, false);
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,      hparams.f_final_logit_softcapping, false);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,              hparams.f_embedding_scale, false);

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,        hparams.dflash_block_size,       false);
    ml.get_key(LLM_KV_DFLASH_CONV_KERNEL_SIZE, hparams.dflash_conv_kernel_size, false);
    ml.get_key(LLM_KV_DFLASH_CONV_GROUP_SIZE,  hparams.dflash_conv_group_size,  false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_RANK,    hparams.dflash_selector_rank,    false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,   hparams.dflash_selector_top_k,   false);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers = [%s]\n", __func__, layers.c_str());

    // DeepSeek-V4 DSpark backbone: stages are full DSV4 blocks, uniform sliding window (the draft KV ring)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,                hparams.n_lora_q);
        ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,             hparams.n_swa);
        ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,           hparams.n_ff_exp);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,                  hparams.n_expert_shared);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,                 hparams.expert_weights_scale);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,                  hparams.expert_weights_norm);
        ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                   hparams.expert_gating_func);
        ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,              hparams.swiglu_clamp_exp, hparams.n_layer_all);
        if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,       hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
            hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
        }
        ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
        ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
        ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
        ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
        ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS,            hparams.dsv4_compress_ratios, false);

        GGML_ASSERT(hparams.dsv4_o_group_count > 0); // avoid div by zero

        if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            throw std::runtime_error("DSpark DSV4 draft expects sqrtsoftplus MoE scoring");
        }
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (hparams.dsv4_compress_ratios[il] != 0) {
                throw std::runtime_error("DSpark DSV4 draft expects uncompressed attention on all stages");
            }
        }

        GGML_ASSERT(hparams.n_swa > 0);
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(0);
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            hparams.is_swa_impl[il] = true;
        }
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

        type = LLM_TYPE_UNKNOWN;
        return;
    }

    // optional interleaved sliding-window attention with per-layer pattern array.
    // DFlash has a single rope, so the SWA rope == main rope.
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // reduced draft vocab (optional): d2t maps draft rows to target token ids
    int64_t n_vocab_draft = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_vocab_draft = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    }

    // DSpark = DFlash + a semi-autoregressive Markov head and Confidence head
    //
    // TODO: only Qwen3-style backbones are supported for now; other backbones (e.g. Gemma4)
    //       need their own conversion path and graph tweaks
    const struct ggml_tensor * markov_meta = ml->get_tensor_meta("markov_w1.weight");
    if (markov_meta) {
        const int64_t dspark_markov_rank = markov_meta->ne[0];

        dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), { dspark_markov_rank, n_vocab }, 0);
        dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), { dspark_markov_rank, n_vocab_draft }, 0);

        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + dspark_markov_rank, 1 }, 0);
        dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   { 1 },             TENSOR_NOT_REQUIRED);

        LLAMA_LOG_INFO("%s: DFlash with DSpark markov head (rank = %lld)\n", __func__, (long long) dspark_markov_rank);
    }

    const struct ggml_tensor * selector_meta = ml->get_tensor_meta("selector_hidden.weight");
    if (selector_meta) {
        const int64_t rank = hparams.dflash_selector_rank;
        if (rank <= 0 || hparams.dflash_block_size <= 0 || hparams.dflash_selector_top_k <= 0 ||
                hparams.dflash_conv_kernel_size <= 0 || hparams.dflash_conv_group_size <= 0) {
            throw std::runtime_error("DFlash2 model is missing conv/selector metadata");
        }
        if (n_embd % hparams.dflash_conv_group_size != 0) {
            throw std::runtime_error("DFlash2 hidden size must be divisible by conv_group_size");
        }
        if (n_embd < hparams.dflash_selector_top_k * (hparams.dflash_selector_top_k + 1)) {
            throw std::runtime_error("DFlash2 hidden size is too small for the selector lattice");
        }

        dflash_selector_prev   = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_PREV,   "weight"), { rank, n_vocab }, 0);
        dflash_selector_next   = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_NEXT,   "weight"), { rank, n_vocab }, 0);
        dflash_selector_hidden = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_HIDDEN, "weight"), { n_embd, rank }, 0);

        LLAMA_LOG_INFO("%s: DFlash2 conv kernel = %u, group = %u, selector rank = %u, top-k = %u\n", __func__,
                hparams.dflash_conv_kernel_size, hparams.dflash_conv_group_size,
                hparams.dflash_selector_rank, hparams.dflash_selector_top_k);
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    if (hparams.dsv4_hc_mult > 0) {
        const int64_t q_lora_rank     = hparams.n_lora_q;
        const int64_t n_ff_exp        = hparams.n_ff_exp;
        const int64_t n_expert_shared = hparams.n_expert_shared;
        const int64_t n_embd_head     = hparams.n_embd_head_k();
        const int64_t o_groups        = hparams.dsv4_o_group_count;
        const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
        const int64_t hc_mult         = hparams.dsv4_hc_mult;
        const int64_t hc_dim          = hc_mult * n_embd;
        const int64_t hc_mix_dim      = (2 + hc_mult) * hc_mult;

        hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc_mult}, 0);
        hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc_mult}, 0);
        hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
            layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
            layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
            layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
            layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
            layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
            layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
            layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        }
        return;
    }

    // optional: reduced-vocab drafts ship their own, full-vocab drafts share the target's via ctx_other
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);

        if (selector_meta) {
            const int64_t kernel = hparams.dflash_conv_kernel_size;
            const int64_t groups = n_embd / hparams.dflash_conv_group_size;
            const int64_t projected = 2 * kernel * groups;
            layer.dflash_attn_conv_base = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_CONV_BASE, i), { n_embd, kernel, 2 }, 0);
            layer.dflash_attn_conv_proj = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_CONV_PROJ, "weight", i), { n_embd, projected }, 0);
            layer.dflash_ffn_conv_base  = create_tensor(tn(LLM_TENSOR_DFLASH_FFN_CONV_BASE, i), { n_embd, kernel, 2 }, 0);
            layer.dflash_ffn_conv_proj  = create_tensor(tn(LLM_TENSOR_DFLASH_FFN_CONV_PROJ,  "weight", i), { n_embd, projected }, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            if (hparams.dsv4_hc_mult > 0) {
                return std::make_unique<graph_dsv4>(*this, params);
            }
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params), model(model) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur, model.fc_s);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// DSpark (DFlash + Markov & Confidence head): Markov bias on the draft logits, chained per block position
static void build_dspark_markov_head(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && model.dspark_conf_proj && "DSpark markov/confidence weights not loaded");

    ggml_tensor * base = res->t_logits; // [n_vocab, n_tokens]
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    const auto it = model.gguf_kv.find("dflash.block_size");
    GGML_ASSERT(it != model.gguf_kv.end() && "DSpark draft requires 'dflash.block_size' in GGUF metadata");
    const int64_t block_size = std::stoi(it->second);
    GGML_ASSERT(block_size > 0);

    // bonus anchor (SpecForge exports): slot 0 is a bonus token, not a prediction slot
    const auto it_anchor          = model.gguf_kv.find("dflash.sample_from_anchor");
    const bool sample_from_anchor = it_anchor == model.gguf_kv.end() || it_anchor->second == "true";
    const int64_t i_draft_beg    = sample_from_anchor ? 0 : 1;

    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tok % n_blocks == 0 && "DSpark markov head requires equal-size blocks");
    // runtime tokens per block in this ubatch (anchor + drafted positions), bounded by training block_size
    const int64_t block_drafts = n_tok / n_blocks;
    if (block_drafts > block_size) {
        return;
    }

    // anchor (committed last) token of every block: token 0 of each block, i.e. a strided view
    const size_t token_stride = (size_t) block_drafts * tokens->nb[0];
    const size_t base_stride = (size_t) block_drafts * base->nb[1];

    ggml_tensor * prev = ggml_view_2d(ctx0, tokens, 1, n_blocks, token_stride, 0);
    prev = ggml_cont_1d(ctx0, prev, n_blocks);

    // confidence head input: predicts per-position acceptance
    ggml_tensor * conf_inp = res->t_embd; // [n_embd, n_tok]

    ggml_tensor * cat      = nullptr;
    ggml_tensor * cat_conf = nullptr;

    if (!sample_from_anchor) {
        // bonus anchor slot: pass the logits through unbiased, pad the (unread) confidence column
        cat      = ggml_cont(ctx0, ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, 0));
        cat_conf = ggml_sigmoid(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, base, 1, n_blocks, base_stride, 0)));
    }

    // TODO: the in-graph chain is greedy (argmax); sampling params affect only the final
    //       token pick, not the Markov conditioning path
    for (int64_t i = i_draft_beg; i < block_drafts; ++i) {
        ggml_tensor * w1_prev = ggml_get_rows(ctx0, w1, prev);   // [R, n_blocks]
        ggml_tensor * bias    = ggml_mul_mat(ctx0, w2, w1_prev); // [n_vocab_draft, n_blocks]
        if (model.d2t) {
            // reduced draft vocab: scatter the bias to the target rows (base is -inf on the others)
            const int64_t n_draft_vocab = bias->ne[0];
            ggml_tensor * full = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_blocks), 0.0f);
            bias = ggml_set_rows(ctx0, full,
                    ggml_reshape_3d(ctx0, bias,      1,             n_draft_vocab, n_blocks),
                    ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
            bias = ggml_reshape_2d(ctx0, bias, n_vocab, n_blocks);
        }

        // position i of every block: strided view [n_vocab, n_blocks]
        ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, i*base->nb[1]);
        ggml_tensor * col    = ggml_add(ctx0, base_i, bias);

        cat = cat ? ggml_concat(ctx0, cat, col, 1) : col;

        // conf(i) = sigmoid(conf_proj . [conf_inp(i); markov_w1[prev(i)]] + b)  -- [1, n_blocks]
        ggml_tensor * conf_inp_i = ggml_view_2d(ctx0, conf_inp, conf_inp->ne[0], n_blocks,
                                                (size_t) block_drafts * conf_inp->nb[1], i*conf_inp->nb[1]);
        ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, conf_inp_i), w1_prev, 0);
        ggml_tensor * conf = ggml_mul_mat(ctx0, model.dspark_conf_proj, feat);
        if (model.dspark_conf_proj_b) {
            conf = ggml_add(ctx0, conf, model.dspark_conf_proj_b);
        }
        conf = ggml_sigmoid(ctx0, conf);

        cat_conf = cat_conf ? ggml_concat(ctx0, cat_conf, conf, 1) : conf;

        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    // cat is position-major; restore ubatch block-major order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, block_drafts, n_blocks]
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    {
        ggml_tensor * conf = ggml_reshape_3d(ctx0, cat_conf, 1, n_blocks, block_drafts);
        conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));
        conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);

        // note: broadcast the [1, n_tok] confidences to n_embd-wide rows to be able to reuse `llama_get_embeddings_nextn`
        conf = ggml_repeat(ctx0, conf, res->t_embd);
        res->t_h_nextn = conf;
        ggml_build_forward_expand(g.gf, conf);
    }

    res->t_logits = out;
    ggml_build_forward_expand(g.gf, out);
}

static ggml_tensor * build_dflash2_conv(
        llm_graph_context & g,
        ggml_tensor * hidden,
        ggml_tensor * dynamic,
        ggml_tensor * base,
        int side) {
    const auto & hparams = g.hparams;
    const int64_t hidden_size = hidden->ne[0];
    const int64_t n_tokens    = hidden->ne[1];
    const int64_t n_blocks    = g.ubatch.n_seqs_unq;
    const int64_t kernel_size = hparams.dflash_conv_kernel_size;
    const int64_t group_size  = hparams.dflash_conv_group_size;
    const int64_t n_groups    = hidden_size / group_size;

    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    GGML_ASSERT(dynamic && base && side >= 0 && side < 2);

    const int64_t block_size = n_tokens / n_blocks;
    ggml_context * ctx0 = g.ctx0;
    hidden = ggml_cont_2d(ctx0, hidden, hidden_size, n_tokens);
    dynamic = ggml_cont_2d(ctx0, dynamic, dynamic->ne[0], n_tokens);
    ggml_tensor * blocks = ggml_reshape_3d(ctx0, hidden, hidden_size, block_size, n_blocks);
    ggml_tensor * grouped = ggml_reshape_3d(ctx0, hidden, group_size, n_groups, n_tokens);
    ggml_tensor * coeffs = ggml_reshape_4d(ctx0, dynamic, n_groups, kernel_size, 2, n_tokens);
    ggml_tensor * coeffs_side = ggml_view_3d(ctx0, coeffs, n_groups, kernel_size, n_tokens,
            coeffs->nb[1], coeffs->nb[3], side * coeffs->nb[2]);

    ggml_tensor * result = nullptr;
    for (int64_t tap = 0; tap < kernel_size; ++tap) {
        ggml_tensor * values = blocks;
        if (tap > 0) {
            ggml_tensor * zeros = ggml_fill(ctx0,
                    ggml_new_tensor_3d(ctx0, hidden->type, hidden_size, std::min(tap, block_size), n_blocks), 0.0f);
            if (tap < block_size) {
                ggml_tensor * previous = ggml_view_3d(ctx0, blocks, hidden_size, block_size - tap, n_blocks,
                        blocks->nb[1], blocks->nb[2], 0);
                values = ggml_concat(ctx0, zeros, previous, 1);
            } else {
                values = zeros;
            }
        }
        values = ggml_reshape_2d(ctx0, values, hidden_size, n_tokens);

        ggml_tensor * coeff = ggml_view_2d(ctx0, coeffs_side, n_groups, n_tokens,
                coeffs_side->nb[2], tap * coeffs_side->nb[1]);
        coeff = ggml_cont(ctx0, coeff);
        coeff = ggml_reshape_3d(ctx0, coeff, 1, n_groups, n_tokens);
        coeff = ggml_reshape_2d(ctx0, ggml_repeat(ctx0, coeff, grouped), hidden_size, n_tokens);

        ggml_tensor * base_tap = ggml_view_1d(ctx0, base, hidden_size,
                tap * base->nb[1] + side * base->nb[2]);
        ggml_tensor * weight = ggml_add(ctx0, coeff, ggml_repeat(ctx0, base_tap, hidden));
        ggml_tensor * term = ggml_mul(ctx0, weight, values);
        result = result ? ggml_add(ctx0, result, term) : term;
    }
    return result;
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
template <>
llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos  = build_inp_pos();

    // optional iSWA: pick the matching attention input
    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;

    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // KV cache injection
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);

        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, inp_g);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur_injected", il);
            cb(Vcur, "Vcur_injected", il);

            if (use_iswa) {
                // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
                const bool    is_swa = hparams.is_swa(il);
                const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
                // rotate K/V into the cache's rotated space
                ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa : inp_attn_iswa->self_k_rot;
                ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa : inp_attn_iswa->self_v_rot;
                if (k_rot) {
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
                }
                if (v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
                }
                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
            } else {
                // rotate K/V into the cache's rotated space
                if (inp_attn->self_k_rot) {
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
                }
                if (inp_attn->self_v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
                }
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
            }
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    // tok_embd from the target model (shared via ctx_other)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
    }
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * attn_dynamic = nullptr;
        if (layer.dflash_attn_conv_proj) {
            attn_dynamic = build_lora_mm(layer.dflash_attn_conv_proj, noise_norm);
            noise_norm = build_dflash2_conv(*this, noise_norm, attn_dynamic, layer.dflash_attn_conv_base, 0);
            cb(noise_norm, "attn_conv_in", il);
        }

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, noise_norm);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

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

        // cache-aware, non-causal attention
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, layer.wo, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il)
            : build_attn(inp_attn,      layer.wo, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

        if (attn_dynamic) {
            cur = build_dflash2_conv(*this, cur, attn_dynamic, layer.dflash_attn_conv_base, 1);
            cb(cur, "attn_conv_out", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * ffn_dynamic = nullptr;
        if (layer.dflash_ffn_conv_proj) {
            ffn_dynamic = build_lora_mm(layer.dflash_ffn_conv_proj, cur);
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 0);
            cb(cur, "ffn_conv_in", il);
        }

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (ffn_dynamic) {
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 1);
            cb(cur, "ffn_conv_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    // lm_head from the target model (shared via ctx_other)
    auto * output   = model.output;
    auto * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DFlash decoder requires the target model's output projection");
        output   = model_other->output;
        output_s = model_other->output_s;
    }

    cur = build_lora_mm(output, cur, output_s);

    if (hparams.f_logit_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
    }
    if (hparams.f_final_logit_softcapping > 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    // reduced-draft-vocab exports: scatter the draft logits to the target vocabulary via d2t
    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // DSpark: bias the draft logits with the Markov head
    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
}

template <bool is_enc>
void llama_model_dflash::graph<is_enc>::build_post_sampling() const {
    if constexpr (is_enc) {
        return;
    }

    if (!model.dflash_selector_hidden || !res->t_logits) {
        return;
    }

    const int64_t top_k    = hparams.dflash_selector_top_k;
    const int64_t rank     = hparams.dflash_selector_rank;
    const int64_t n_blocks = ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    GGML_ASSERT(res->t_logits->ne[1] == n_tokens);
    ggml_tensor * tokens = res->get_inp_tokens();
    if (!tokens) {
        return;
    }

    const int64_t tokens_per_block = n_tokens / n_blocks;
    const int64_t block_size = std::min<int64_t>(tokens_per_block, hparams.dflash_block_size);
    ggml_tensor * candidates = ggml_top_k(ctx0, res->t_logits, top_k);
    ggml_tensor * logits_rows = ggml_reshape_3d(ctx0, res->t_logits, 1, res->t_logits->ne[0], n_tokens);
    ggml_tensor * unary = ggml_reshape_2d(ctx0,
            ggml_get_rows(ctx0, logits_rows, candidates), top_k, n_tokens);

    std::vector<ggml_tensor *> candidate_ids(block_size);
    std::vector<ggml_tensor *> unary_logits(block_size);
    for (int64_t pos = 1; pos < block_size; ++pos) {
        candidate_ids[pos] = ggml_cont_2d(ctx0,
                ggml_view_2d(ctx0, candidates, top_k, n_blocks,
                    tokens_per_block * candidates->nb[1], pos * candidates->nb[1]),
                top_k, n_blocks);
        unary_logits[pos] = ggml_cont_2d(ctx0,
                ggml_view_2d(ctx0, unary, top_k, n_blocks,
                    tokens_per_block * unary->nb[1], pos * unary->nb[1]),
                top_k, n_blocks);
    }

    ggml_tensor * hidden = build_lora_mm(model.dflash_selector_hidden, res->t_embd);

    ggml_tensor * anchor_ids = ggml_view_2d(ctx0, tokens, 1, n_blocks,
            tokens_per_block * tokens->nb[0], 0);
    anchor_ids = ggml_cont_1d(ctx0, anchor_ids, n_blocks);

    ggml_tensor * packed = ggml_fill(ctx0,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, 1, n_blocks), 0.0f);

    for (int64_t pos = 1; pos < block_size; ++pos) {
        ggml_tensor * ids = candidate_ids[pos];
        ggml_tensor * unary = unary_logits[pos];
        ggml_tensor * successor = ggml_get_rows(ctx0, model.dflash_selector_next,
                ggml_reshape_1d(ctx0, ids, top_k * n_blocks));
        successor = ggml_reshape_3d(ctx0, successor, rank, top_k, n_blocks);

        ggml_tensor * hidden_pos = ggml_cont(ctx0, ggml_view_2d(ctx0, hidden, rank, n_blocks,
                tokens_per_block * hidden->nb[1], pos * hidden->nb[1]));
        hidden_pos = ggml_reshape_3d(ctx0, hidden_pos, rank, 1, n_blocks);

        ggml_tensor * predecessor;
        if (pos == 1) {
            predecessor = ggml_get_rows(ctx0, model.dflash_selector_prev, anchor_ids);
            predecessor = ggml_reshape_3d(ctx0, predecessor, rank, 1, n_blocks);
        } else {
            predecessor = ggml_get_rows(ctx0, model.dflash_selector_prev,
                    ggml_reshape_1d(ctx0, candidate_ids[pos - 1], top_k * n_blocks));
            predecessor = ggml_reshape_3d(ctx0, predecessor, rank, top_k, n_blocks);
        }

        ggml_tensor * conditioned = ggml_mul(ctx0, predecessor, ggml_repeat(ctx0, hidden_pos, predecessor));
        ggml_tensor * scores = ggml_mul_mat(ctx0, successor, conditioned);
        if (pos == 1) {
            scores = ggml_repeat_4d(ctx0, scores, top_k, top_k, n_blocks, 1);
        }
        ggml_tensor * unary_3d = ggml_reshape_3d(ctx0, unary, top_k, 1, n_blocks);
        scores = ggml_add(ctx0, scores, ggml_repeat(ctx0, unary_3d, scores));

        ggml_tensor * row = ggml_concat(ctx0,
                ggml_cast(ctx0, ids, GGML_TYPE_F32),
                ggml_reshape_2d(ctx0, scores, top_k * top_k, n_blocks), 0);
        row = ggml_pad(ctx0, row, n_embd - row->ne[0], 0, 0, 0);
        row = ggml_reshape_3d(ctx0, row, n_embd, 1, n_blocks);
        packed = ggml_concat(ctx0, packed, row, 1);
    }

    packed = ggml_reshape_2d(ctx0, packed, n_embd, block_size * n_blocks);
    cb(packed, "dflash2_lattice", -1);
    res->t_h_nextn = packed;
    ggml_build_forward_expand(gf, packed);
}

// DSV4 DSpark decoder, dual-mode by batch type (see the DFlash decoder above):
//   * embd batch  -> project main_x through each stage's wkv and inject K into the ring cache
//   * token batch -> noise block through 3 full DSV4 stages (hc + MLA + MoE), markov + confidence heads
llama_model_dflash::graph_dsv4::graph_dsv4(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    // KV cache injection: fused target features from the encoder
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);

        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            // main-track KV: kv_norm(wkv(main_x)) with rope on the trailing dims, same
            // rope parameters as the uncompressed layers in build_attention_impl
            ggml_tensor * kv = build_lora_mm(layer.wkv, inp_g);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);

            ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, n_tokens,
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head),
                    0);
            ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, n_tokens,
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head_nope));
            kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
            cb(kv, "kv_injected", il);

            if (inp_attn->self_k_rot_swa) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    // tok_embd from the target model (shared via ctx_other)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DSpark decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    const int64_t hc = hparams.dsv4_hc_mult;
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        ggml_tensor * cur = build_hc_pre(inpL,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_attn, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "l_out", il);
    }

    ggml_tensor * cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    // confidence head input: the reference scores the pre-norm collapsed hidden state
    res->t_embd = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // lm_head from the target model (shared via ctx_other)
    auto * output   = model.output;
    auto * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DSpark decoder requires the target model's output projection");
        output   = model_other->output;
        output_s = model_other->output_s;
    }

    cur = build_lora_mm(output, cur, output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
}
