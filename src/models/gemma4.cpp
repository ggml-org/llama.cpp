#include "models.h"

static ggml_tensor * gemma4_tile640_dequant(
        ggml_context * ctx,
        const llama_tile640_tensor * tensor) {
    GGML_ASSERT(tensor && tensor->valid());
    return ggml_tile640_dequant(
        ctx,
        tensor->packed,
        tensor->page_scales,
        tensor->lane_scales,
        tensor->outlier_row_offsets,
        tensor->outlier_cols,
        tensor->outlier_vals,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
}

static ggml_tensor * gemma4_tile640_get_rows(
        ggml_context * ctx,
        const llama_tile640_tensor * tensor,
        ggml_tensor * ids) {
    GGML_ASSERT(tensor && tensor->valid());
    return ggml_tile640_get_rows(
        ctx,
        tensor->packed,
        tensor->page_scales,
        tensor->lane_scales,
        tensor->outlier_row_offsets,
        tensor->outlier_cols,
        tensor->outlier_vals,
        ids,
        (int32_t) tensor->ne[0]);
}

ggml_tensor * llama_model_gemma4::graph::tile640_mul_mat(
        const llama_tile640_tensor * tensor,
        ggml_tensor * input) const {
    GGML_ASSERT(tensor && tensor->valid());
    return build_tile640_lora_mm(
        tensor->packed,
        tensor->page_scales,
        tensor->lane_scales,
        tensor->outlier_row_offsets,
        tensor->outlier_cols,
        tensor->outlier_vals,
        tensor->act_scale,
        input);
}

void llama_model_gemma4::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());

    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);

    hparams.n_layer_kv_from_start = hparams.n_layer_all - (int32_t)n_kv_shared_layers;
    hparams.f_attention_scale     = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    switch (hparams.n_layer()) {
        case 30: type = LLM_TYPE_26B_A4B; break;
        case 35: type = LLM_TYPE_E2B; break;
        case 42: type = LLM_TYPE_E4B; break;
        case 60: type = LLM_TYPE_31B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gemma4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const uint32_t n_embd_per_layer = hparams.n_embd_per_layer;
    const int64_t  n_ff_exp         = hparams.n_ff_exp;

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k_swa == n_embd_head_v_swa");
    }

    output = create_tensor_or_tile640(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL && get_tile640_tensor(tn(LLM_TENSOR_OUTPUT, "weight").str()) == nullptr) {
        output = create_tensor_or_tile640(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    tok_embd = create_tensor_or_tile640(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    if (n_embd_per_layer > 0) {
        per_layer_tok_embd   = create_tensor_or_tile640(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),    {n_embd_per_layer * n_layer, n_vocab}, 0);
        per_layer_model_proj = create_tensor_or_tile640(tn(LLM_TENSOR_PER_LAYER_MODEL_PROJ, "weight", 0), {n_embd, n_embd_per_layer * n_layer}, 0);
        per_layer_proj_norm  = create_tensor_or_tile640(tn(LLM_TENSOR_PER_LAYER_PROJ_NORM,  "weight", 0), {n_embd_per_layer}, 0);
    }

    output_norm = create_tensor_or_tile640(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_embd_k    = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v    = hparams.n_embd_v_gqa(i);
        const int     kv_flags    = hparams.has_kv(i) ? 0 : TENSOR_NOT_REQUIRED;

        layer.attn_norm = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)
        layer.wq = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head * n_head}, 0);
        layer.wk = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k}, kv_flags);
        layer.wv = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v}, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, 0);

        layer.attn_q_norm    = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_k_norm    = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head}, kv_flags);
        layer.attn_post_norm = create_tensor_or_tile640(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.out_scale = create_tensor_or_tile640(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, TENSOR_NOT_REQUIRED);

        if (!hparams.is_swa(i)) {
            // full_attention layers use rope_freqs for proportional rope
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head/2}, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        // handle use_double_wide_mlp
        int64_t n_ff_cur = hparams.n_ff(i);

        // for expert layers, we use normal FFN as shared expert (same as python code)
        layer.ffn_norm = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_up   = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_down = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_cur, n_embd}, 0);
        layer.ffn_post_norm = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        // MoE router
        layer.ffn_gate_inp = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        bool has_expert = layer.ffn_gate_inp != nullptr ||
            get_tile640_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i).str()) != nullptr;

        // norm
        if (has_expert) {
            layer.ffn_gate_inp_s = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_GATE_INP, "scale", i), {n_embd}, 0);

            layer.ffn_pre_norm_2  = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_PRE_NORM_2,  "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_1 = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_POST_NORM_1, "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_2 = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_POST_NORM_2, "weight", i), {n_embd}, 0);

            // MoE FFN
            layer.ffn_gate_up_exps  = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_GATE_UP_EXPS,  "weight", i), {n_embd, n_ff_exp * 2, n_expert}, TENSOR_NOT_REQUIRED);

            if (layer.ffn_gate_up_exps == nullptr &&
                get_tile640_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i).str()) == nullptr) {
                layer.ffn_gate_exps = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
                layer.ffn_up_exps   = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            }

            layer.ffn_down_exps = create_tensor_or_tile640(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

            // The generic post-load path only discovers this scale when the
            // conventional expert tensor exists. Tile640 replaces that tensor,
            // so load Gemma's per-expert contribution scale explicitly.
            if (get_tile640_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i).str()) != nullptr) {
                layer.ffn_down_exps_s = create_tensor_or_tile640(
                    tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i),
                    {n_expert},
                    TENSOR_NOT_REQUIRED);
            }
        }

        // per-layer embeddings
        if (n_embd_per_layer > 0) {
            layer.per_layer_inp_gate   = create_tensor_or_tile640(tn(LLM_TENSOR_PER_LAYER_INP_GATE,  "weight", i), {n_embd, n_embd_per_layer}, 0);
            layer.per_layer_proj       = create_tensor_or_tile640(tn(LLM_TENSOR_PER_LAYER_PROJ,      "weight", i), {n_embd_per_layer, n_embd}, 0);
            layer.per_layer_post_norm  = create_tensor_or_tile640(tn(LLM_TENSOR_PER_LAYER_POST_NORM, "weight", i), {n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// get 2D slice view from a 3D tensor, the idx corresponds to the 3rd dim
static ggml_tensor * ggml_view_2d_slice(ggml_context * ctx0, ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], ggml_row_size(x->type, x->ne[0]),
                        idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}

// TODO @ngxson : maybe improve this in the future
class llm_graph_input_logits_bias : public llm_graph_input_i {
public:
    llm_graph_input_logits_bias(const llama_vocab & vocab) {
        arr.resize(vocab.n_tokens(), 0.0f);
        for (llama_token id : vocab.get_suppress_tokens()) {
            if (0 <= id && id < (int32_t)vocab.n_tokens()) {
                arr[id] = -INFINITY;
            }
        }
    }
    virtual ~llm_graph_input_logits_bias() = default;

    void set_input(const llama_ubatch * /*ubatch*/) override {
        const int64_t n_vocab = arr.size();
        ggml_backend_tensor_set(logits_bias, arr.data(), 0, n_vocab*ggml_element_size(logits_bias));
    }

    bool can_reuse(const llm_graph_params & /*params*/) override {
        return true;
    }

    ggml_tensor * logits_bias = nullptr; // F32 [n_vocab]

    std::vector<float> arr;
};

llama_model_gemma4::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_per_layer(model.hparams.n_embd_per_layer) {
    const LLM_TN qtn(model.arch);
    ggml_tensor * cur;
    ggml_tensor * inpL;

    const uint32_t ane_prefill_layer_count = cparams.ane_prefill_layer_count;
    if (ane_prefill_layer_count > 0) {
        // The ANE slab owns embedding lookup, layer normalization, and all
        // operations through this exclusive layer boundary.  Its token-major
        // F32 result is supplied through the normal embedding-backed ubatch
        // input, then the Metal graph resumes at the next layer.  K/V for the
        // skipped layers are imported through the memory context before this
        // graph is constructed; this graph never fabricates a second cache.
        GGML_ASSERT(ane_prefill_layer_count < (uint32_t) n_layer);
        GGML_ASSERT(ubatch.embd && "ANE continuation requires post-slab hidden states");
        auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, ubatch.n_tokens);
        cb(inp->embd, "ane_prefill_hidden", (int) ane_prefill_layer_count - 1);
        ggml_set_input(inp->embd);
        inpL = inp->embd;
        res->t_inp_embd = inpL;
        res->add_input(std::move(inp));
    } else if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_TOKEN_EMBD, "weight").str())) {
        auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp());
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), ubatch.n_tokens);
        cb(inp->embd, "inp_embd", -1);
        ggml_set_input(inp->embd);
        std::array<ggml_tensor *, 2> inps = {
            gemma4_tile640_get_rows(ctx0, q, inp->tokens),
            inp->embd,
        };
        inpL = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);
        res->t_inp_embd = inpL;
        res->add_input(std::move(inp));
        ggml_build_forward_expand(gf, inpL);
    } else {
        inpL = build_inp_embd(model.tok_embd);
    }

    if (ane_prefill_layer_count == 0) {
        // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
        inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
        cb(inpL, "inp_scaled", -1);
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // TODO: is causal == true correct? might need some changes
    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inp_per_layer = nullptr;
    if (model.per_layer_tok_embd ||
        model.get_tile640_tensor(qtn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str()) != nullptr) {
        inp_per_layer = build_inp_per_layer();
        ggml_build_forward_expand(gf, inp_per_layer);

        // inp_per_layer shape: [n_embd_per_layer, n_tokens, n_layer]
        inp_per_layer = project_per_layer_inputs(inpL, inp_per_layer);
    }

    for (int il = (int) ane_prefill_layer_count; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        res->t_layer_inp[il] = inpL;

        // norm
        ggml_tensor * attn_norm = model.layers[il].attn_norm;
        if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_ATTN_NORM, "weight", il).str())) {
            attn_norm = gemma4_tile640_dequant(ctx0, q);
        }
        cur = build_norm(inpL, attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {
            // full_attention layers use rope_freqs for proportional rope
            freq_factors = model.layers[il].rope_freqs;
        }

        // Q projection (shared for both non-KV and KV layers)
        // this is to mirror Gemma4Attention in pytorch code
        ggml_tensor * Qcur;
        {
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_ATTN_Q, "weight", il).str())) {
                Qcur = tile640_mul_mat(q, cur);
            } else {
                Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
            }
            cb(Qcur, "Qcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            ggml_tensor * q_norm = model.layers[il].attn_q_norm;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_ATTN_Q_NORM, "weight", il).str())) {
                q_norm = gemma4_tile640_dequant(ctx0, q);
            }
            Qcur = build_norm(Qcur, q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_pos", il);
        }

        // self-attention
        if (hparams.has_kv(il)) {
            ggml_tensor * Kcur;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_ATTN_K, "weight", il).str())) {
                Kcur = tile640_mul_mat(q, cur);
            } else {
                Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
            }
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = Kcur;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_ATTN_V, "weight", il).str())) {
                Vcur = tile640_mul_mat(q, cur);
            } else if (model.layers[il].wv) {
                Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
            }
            cb(Vcur, "Vcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            ggml_tensor * k_norm = model.layers[il].attn_k_norm;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_ATTN_K_NORM, "weight", il).str())) {
                k_norm = gemma4_tile640_dequant(ctx0, q);
            }
            Kcur = build_norm(Kcur, k_norm, nullptr, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Kcur, "Kcur_pos", il);

            const auto * wo_q = model.get_tile640_tensor(qtn(LLM_TENSOR_ATTN_OUT, "weight", il).str());
            cur = build_attn(inp_attn, wo_q ? nullptr : model.layers[il].wo,
                    nullptr, wo_q ? nullptr : model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);
            if (wo_q) {
                cur = tile640_mul_mat(wo_q, cur);
            }
        } else {
            // reuse KV cache of earlier layers
            const auto * wo_q = model.get_tile640_tensor(qtn(LLM_TENSOR_ATTN_OUT, "weight", il).str());
            cur = build_attn(inp_attn,
                    wo_q ? nullptr : model.layers[il].wo, nullptr,
                    wo_q ? nullptr : model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
            if (wo_q) {
                cur = tile640_mul_mat(wo_q, cur);
            }
        }

        // TODO @ngxson : strip unused token right after the last KV layer to speed up prompt processing
        // keep all rows when extracting unmasked nextn embeddings (MTP target needs the hidden state for every token)
        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        ggml_tensor * attn_post_norm = model.layers[il].attn_post_norm;
        if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_ATTN_POST_NORM, "weight", il).str())) {
            attn_post_norm = gemma4_tile640_dequant(ctx0, q);
        }
        cur = build_norm(cur, attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward network
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr ||
            model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_GATE_INP, "weight", il).str()) != nullptr;
        if (is_moe_layer) {
            // MLP (shared exp)
            ggml_tensor * ffn_norm = model.layers[il].ffn_norm;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_NORM, "weight", il).str())) {
                ffn_norm = gemma4_tile640_dequant(ctx0, q);
            }
            ggml_tensor * cur_mlp = build_norm(attn_out,
                    ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_norm_1", il);

            const auto * shared_up_q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_UP, "weight", il).str());
            const auto * shared_gate_q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_GATE, "weight", il).str());
            const auto * shared_down_q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_DOWN, "weight", il).str());
            if (shared_up_q && shared_gate_q && shared_down_q) {
                ggml_tensor * up = tile640_mul_mat(shared_up_q, cur_mlp);
                ggml_tensor * gate = ggml_gelu(ctx0, tile640_mul_mat(shared_gate_q, cur_mlp));
                cur_mlp = tile640_mul_mat(shared_down_q, ggml_mul(ctx0, gate, up));
            } else {
                cur_mlp = build_ffn(cur_mlp,
                        model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                        model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                        model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                        nullptr,
                        LLM_FFN_GELU, LLM_FFN_PAR, il);
            }
            ggml_tensor * post_norm_1 = model.layers[il].ffn_post_norm_1;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_POST_NORM_1, "weight", il).str())) {
                post_norm_1 = gemma4_tile640_dequant(ctx0, q);
            }
            cur_mlp = build_norm(cur_mlp,
                    post_norm_1, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_mlp", il);

            // Expert FFN
            ggml_tensor * pre_norm_2 = model.layers[il].ffn_pre_norm_2;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_PRE_NORM_2, "weight", il).str())) {
                pre_norm_2 = gemma4_tile640_dequant(ctx0, q);
            }
            ggml_tensor * cur_moe = build_norm(attn_out,
                    pre_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_norm_2", il);

            // custom MoE logits calculation (router operates on attn_out, not cur)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            ggml_tensor * router_scale = model.layers[il].ffn_gate_inp_s;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_GATE_INP, "scale", il).str())) {
                router_scale = gemma4_tile640_dequant(ctx0, q);
            }
            tmp = ggml_mul(ctx0, tmp, router_scale);
            ggml_tensor * logits;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_GATE_INP, "weight", il).str())) {
                logits = tile640_mul_mat(q, tmp);
            } else {
                logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp);
            }
            cb(logits, "ffn_moe_logits", il);

            const auto * gate_up_tile = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", il).str());
            const auto * down_tile = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il).str());
            ggml_tensor * down_scale = model.layers[il].ffn_down_exps_s;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", il).str())) {
                down_scale = gemma4_tile640_dequant(ctx0, q);
            }
            cur_moe = build_moe_ffn(cur_moe,
                    nullptr, // gate_inp
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr, // exp_probs_b (not used for gemma4)
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    down_scale,
                    nullptr,
                    gate_up_tile,
                    down_tile);
            ggml_tensor * post_norm_2 = model.layers[il].ffn_post_norm_2;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_POST_NORM_2, "weight", il).str())) {
                post_norm_2 = gemma4_tile640_dequant(ctx0, q);
            }
            cur_moe = build_norm(cur_moe,
                    post_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
            cb(cur, "ffn_moe_combined", il);
        } else {
            ggml_tensor * ffn_norm = model.layers[il].ffn_norm;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_NORM, "weight", il).str())) {
                ffn_norm = gemma4_tile640_dequant(ctx0, q);
            }
            cur = build_norm(attn_out,
                    ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            const auto * up_q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_UP, "weight", il).str());
            const auto * gate_q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_GATE, "weight", il).str());
            const auto * down_q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_DOWN, "weight", il).str());
            if (up_q && gate_q && down_q) {
                ggml_tensor * up = tile640_mul_mat(up_q, cur);
                cb(up, "ffn_up", il);
                ggml_tensor * gate = tile640_mul_mat(gate_q, cur);
                cb(gate, "ffn_gate", il);
                gate = ggml_gelu(ctx0, gate);
                cb(gate, "ffn_gelu", il);
                cur = ggml_mul(ctx0, gate, up);
                cb(cur, "ffn_gate_par", il);
                cur = tile640_mul_mat(down_q, cur);
                cb(cur, "ffn_down", il);
            } else {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                        model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                        model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                        nullptr,
                        LLM_FFN_GELU, LLM_FFN_PAR, il);
            }
            cb(cur, "ffn_out", il);
        }
        ggml_tensor * ffn_post_norm = model.layers[il].ffn_post_norm;
        if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_FFN_POST_NORM, "weight", il).str())) {
            ffn_post_norm = gemma4_tile640_dequant(ctx0, q);
        }
        cur = build_norm(cur,
                ffn_post_norm, nullptr,
                LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        // residual connection
        cur = ggml_add(ctx0, cur, attn_out);

        // per-layer embedding
        if (inp_per_layer) {
            ggml_tensor * pe_in = cur;
            cb(cur, "pe_in", il);

            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_PER_LAYER_INP_GATE, "weight", il).str())) {
                cur = tile640_mul_mat(q, cur);
            } else {
                cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur);
            }
            cur = ggml_gelu(ctx0, cur);

            ggml_tensor * inp_this_layer = ggml_view_2d_slice(ctx0, inp_per_layer, il); // [n_embd_per_layer, n_tokens]

            // TODO @ngxson : improve this
            if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_PER_LAYER_PROJ, "weight", il).str())) {
                cur = tile640_mul_mat(q, cur);
            } else {
                cur = build_lora_mm(model.layers[il].per_layer_proj, cur);
            }
            ggml_tensor * per_layer_post_norm = model.layers[il].per_layer_post_norm;
            if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_PER_LAYER_POST_NORM, "weight", il).str())) {
                per_layer_post_norm = gemma4_tile640_dequant(ctx0, q);
            }
            cur = build_norm(cur, per_layer_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "per_layer_embd_out", il);

            // residual connection
            cur = ggml_add(ctx0, pe_in, cur);
        }

        // layer_scalar
        ggml_tensor * out_scale = model.layers[il].out_scale;
        if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", il).str())) {
            out_scale = gemma4_tile640_dequant(ctx0, q);
        }
        if (out_scale) {
            cur = ggml_mul(ctx0, cur, out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    ggml_tensor * output_norm = model.output_norm;
    if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_OUTPUT_NORM, "weight").str())) {
        output_norm = gemma4_tile640_dequant(ctx0, q);
    }
    cur = build_norm(cur,
            output_norm, nullptr,
            LLM_NORM_RMS, -1);

    // Expose the post-output-norm hidden state (the LM-head input feature) so that
    // MTP draft contexts can read it via llama_get_embeddings_nextn_ith() as the
    // recurrent h input. This matches the reference (transformers/vLLM/SGLang),
    // which feeds the drafter the target's post-final-norm hidden state.
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_OUTPUT, "weight").str())) {
        cur = tile640_mul_mat(q, cur);
    } else if (model.output) {
        cur = build_lora_mm(model.output, cur, model.output_s);
    } else {
        const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_TOKEN_EMBD, "weight").str());
        GGML_ASSERT(q != nullptr);
        cur = tile640_mul_mat(q, cur);
    }

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    // apply logits bias if needed (e.g. for gemma4_unified patch)
    // this is to mirror the suppress_tokens patch on transformers, to avoid model from outputing <image|> and <audio|> tokens (which is a known issue related to the checkpoint)
    // TODO: maybe handle this inside the sampling system in the future
    if (!model.vocab.get_suppress_tokens().empty()) {
        auto inp_bias = std::make_unique<llm_graph_input_logits_bias>(model.vocab);
        inp_bias->logits_bias = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, inp_bias->arr.size());
        cur = ggml_add(ctx0, cur, inp_bias->logits_bias);
        res->add_input(std::move(inp_bias));
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// equivalent to get_per_layer_inputs() in python code
// output shape: [n_embd_per_layer, n_layer, n_tokens]
ggml_tensor * llama_model_gemma4::graph::build_inp_per_layer() {
    const LLM_TN qtn(model.arch);
    const auto * per_layer_q = model.get_tile640_tensor(
        qtn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str());
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    ggml_tensor * inp_per_layer;
    float tok_embd_scale = sqrtf((float) n_embd_per_layer);
    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp_per_layer = per_layer_q
            ? gemma4_tile640_get_rows(ctx0, per_layer_q, inp->tokens)
            : ggml_get_rows(ctx0, model.per_layer_tok_embd, inp->tokens);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, n_tokens);
        inp_per_layer = ggml_scale     (ctx0, inp_per_layer, tok_embd_scale);
        cb(inp_per_layer, "inp_per_layer_selected", -1);

        res->add_input(std::move(inp));
    } else {
        // Multimodal embedding path: use padding token (ID=0) embedding
        // TODO: verify if this is the correct behavior in transformers implementation
        const int64_t embd_size = per_layer_q
            ? per_layer_q->ne[0]
            : model.per_layer_tok_embd->ne[0];

        // Extract and dequantize padding token embedding (row 0)
        ggml_tensor * per_layer_weights = per_layer_q
            ? gemma4_tile640_dequant(ctx0, per_layer_q)
            : model.per_layer_tok_embd;
        ggml_tensor * padding = ggml_view_1d(ctx0, per_layer_weights, embd_size, 0);
        inp_per_layer = ggml_cast (ctx0, padding, GGML_TYPE_F32);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, tok_embd_scale);

        // Reshape to [n_embd_per_layer, n_layer, 1]
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, 1);
        cb(inp_per_layer, "inp_per_layer_multimodal", -1);
    }
    return inp_per_layer;
}

// equivalent to project_per_layer_inputs() in python code
// this calculates the per-layer inputs, so the final tensor shape will have n_layer as the last dim
// inp_batch     shape: [n_embd, n_tokens]
// inp_per_layer shape: [n_embd_per_layer, n_layer, n_tokens] (from build_inp_per_layer)
// output shape: [n_embd_per_layer, n_tokens, n_layer]
ggml_tensor * llama_model_gemma4::graph::project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer) {
    const LLM_TN qtn(model.arch);
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

    // note: this matrix multiplication will be performed in the input layer (i.e. on the CPU)
    ggml_tensor * per_layer_proj;
    if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_PER_LAYER_MODEL_PROJ, "weight", 0).str())) {
        per_layer_proj = tile640_mul_mat(q, inp_batch);
    } else {
        per_layer_proj = ggml_mul_mat(ctx0, model.per_layer_model_proj, inp_batch);
    }
    per_layer_proj = ggml_scale     (ctx0, per_layer_proj, per_layer_projection_scale);
    per_layer_proj = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_per_layer, n_layer, n_tokens);

    ggml_tensor * per_layer_proj_norm = model.per_layer_proj_norm;
    if (const auto * q = model.get_tile640_tensor(qtn(LLM_TENSOR_PER_LAYER_PROJ_NORM, "weight", 0).str())) {
        per_layer_proj_norm = gemma4_tile640_dequant(ctx0, q);
    }
    per_layer_proj = build_norm(per_layer_proj, per_layer_proj_norm, nullptr, LLM_NORM_RMS, -1);
    cb(per_layer_proj, "per_layer_proj", -1);

    inp_per_layer = ggml_add  (ctx0, per_layer_proj, inp_per_layer);
    inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);
    cb(inp_per_layer, "inp_per_layer", -1);

    // permute to shape: [n_embd_per_layer, n_tokens, n_layer]
    inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
    return inp_per_layer;
}
