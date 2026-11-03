#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

// DeepSeek-V4 DSpark backbone: stages are full DSV4 blocks, uniform sliding window (the draft KV ring)
void llama_model_dflash::load_hparams_dsv4(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,                hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,             hparams.n_swa);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,    hparams.n_ff_exp_arr, hparams.n_layer_all);
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
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINGLE_PASS,         hparams.dsv4_hc_single_pass, false);

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
    if (hparams.dsv4_hc_single_pass) {
        // DeepSeek-V4.1 DSpark: block queries see the main-KV ring plus the block via the mask row of the
        // first query, so that row's SWA window must reach window_size positions before the anchor
        hparams.n_swa += 1;
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
}

// Mistral Large 4 DSpark: dense MLA stages, one sliding window, plain YaRN
void llama_model_dflash::load_hparams_mistral_large4(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL,  hparams.rope_yarn_log_mul, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,   hparams.n_swa);

    if (hparams.n_swa == 0) {
        throw std::runtime_error("DSpark Mistral Large 4 draft expects a sliding window");
    }
    if (hparams.n_embd_head_k_full != hparams.n_lora_kv + hparams.n_rot()) {
        throw std::runtime_error("DSpark Mistral Large 4 draft expects key_length == kv_lora_rank + rope dim");
    }

    // same as DSV4.1 DSpark: block uses the window of the anchor
    hparams.n_swa += 1;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
}

// DSpark = DFlash + a semi-autoregressive Markov head and Confidence head
void llama_model_dflash::load_tensors_dspark(int64_t n_vocab_draft) {
    LLAMA_LOAD_LOCALS;

    const int64_t dspark_markov_rank = ml->get_tensor_meta("markov_w1.weight")->ne[0];

    dspark_markov_w1   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), { dspark_markov_rank, n_vocab }, 0);
    dspark_markov_w2   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), { dspark_markov_rank, n_vocab_draft }, 0);
    dspark_markov_w2_s = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);

    dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + dspark_markov_rank, 1 }, TENSOR_NOT_REQUIRED);
    dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   { 1 },             TENSOR_NOT_REQUIRED);

    dspark_joint_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "weight"), { n_embd + 2*dspark_markov_rank, 3*dspark_markov_rank }, TENSOR_NOT_REQUIRED);
    dspark_joint_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "bias"),   { 3*dspark_markov_rank }, dspark_joint_proj ? 0 : TENSOR_NOT_REQUIRED);

    LLAMA_LOG_INFO("%s: DFlash with DSpark markov head (rank = %lld)\n", __func__, (long long) dspark_markov_rank);
}

void llama_model_dflash::load_tensors_layers_dsv4() {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_embd_head     = hparams.n_embd_head_k();
    const int64_t o_groups        = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult         = hparams.dsv4_hc_mult;
    const int64_t hc_dim          = hc_mult * n_embd;
    const int64_t hc_mix_dim      = (2 + hc_mult) * hc_mult;

    // single-pass mHC: collapse with the last FFN pre-mix, no hc_head
    if (!hparams.dsv4_hc_single_pass) {
        hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc_mult}, 0);
        hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc_mult}, 0);
        hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);
    }

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
}

void llama_model_dflash::load_tensors_layers_mistral_large4() {
    LLAMA_LOAD_LOCALS;

    const int64_t kv_lora_rank      = hparams.n_lora_kv;
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_nope  = n_embd_head_k_mla - n_rot;

    dspark_mask_embd = create_tensor(tn(LLM_TENSOR_DSPARK_MASK_EMBD, "weight"), { n_embd, hparams.dflash_block_size - 1 }, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd }, 0);
        layer.wq             = create_tensor(tn(LLM_TENSOR_ATTN_Q,         "weight", i), { n_embd, n_head * n_embd_head_k_mla }, 0);
        layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), { n_embd, kv_lora_rank + n_rot }, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), { kv_lora_rank }, 0);
        layer.wk_b           = create_tensor(tn(LLM_TENSOR_ATTN_K_B,       "weight", i), { n_embd_head_nope, kv_lora_rank, n_head }, 0);
        layer.wv_b           = create_tensor(tn(LLM_TENSOR_ATTN_V_B,       "weight", i), { kv_lora_rank, n_embd_head_v_mla, n_head }, 0);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), { n_head * n_embd_head_v_mla, n_embd }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

// DSpark (DFlash + Markov & Confidence head): Markov bias on the draft logits, chained per block position
static void build_dspark_markov_head(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens, ggml_tensor * hidden = nullptr) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && "DSpark markov weights not loaded");

    // confidence head is optional
    const bool has_conf = model.dspark_conf_proj != nullptr;

    // rnn head: gated state carried across block positions
    const bool is_rnn = model.dspark_joint_proj != nullptr;
    GGML_ASSERT(!is_rnn || hidden != nullptr);

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

    ggml_tensor * cat      = nullptr;
    ggml_tensor * cat_conf = nullptr;

    if (!sample_from_anchor) {
        // bonus anchor slot: pass the logits through unbiased, pad the (unread) confidence column
        cat = ggml_cont(ctx0, ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, 0));
        if (has_conf) {
            cat_conf = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, base, 1, n_blocks, base_stride, 0));
        }
    }

    ggml_tensor * state = nullptr; // [R, n_blocks]

    // TODO: the in-graph chain is greedy (argmax); sampling params affect only the final
    //       token pick, not the Markov conditioning path
    for (int64_t i = i_draft_beg; i < block_drafts; ++i) {
        ggml_tensor * w1_prev = ggml_get_rows(ctx0, w1, prev); // [R, n_blocks]
        ggml_tensor * markov  = w1_prev;

        if (is_rnn) {
            const int64_t rank = w1_prev->ne[0];
            if (state == nullptr) {
                state = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rank, n_blocks), 0.0f);
            }

            ggml_tensor * hidden_i = ggml_view_2d(ctx0, hidden, hidden->ne[0], n_blocks,
                                                  (size_t) block_drafts * hidden->nb[1], i*hidden->nb[1]);
            ggml_tensor * joint = ggml_concat(ctx0, ggml_concat(ctx0, state, w1_prev, 0), ggml_cont(ctx0, hidden_i), 0);
            joint = ggml_add(ctx0, ggml_mul_mat(ctx0, model.dspark_joint_proj, joint), model.dspark_joint_proj_b); // [3R, n_blocks]

            const auto chunk = [&](int64_t k) {
                return ggml_cont(ctx0, ggml_view_2d(ctx0, joint, rank, n_blocks, joint->nb[1], k*rank*ggml_element_size(joint)));
            };
            ggml_tensor * gate = ggml_sigmoid(ctx0, chunk(0));
            ggml_tensor * cand = ggml_tanh(ctx0, chunk(1));

            // state = gate * state + (1 - gate) * cand
            state  = ggml_add(ctx0, cand, ggml_mul(ctx0, gate, ggml_sub(ctx0, state, cand)));
            markov = ggml_tanh(ctx0, chunk(2));
        }

        ggml_tensor * bias = g.build_lora_mm(w2, markov, model.dspark_markov_w2_s); // [n_vocab_draft, n_blocks]
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

        if (has_conf) {
            // confidence head input: predicts per-position acceptance
            ggml_tensor * conf_inp   = res->t_embd; // [n_embd, n_tok]
            // conf(i) = sigmoid(conf_proj . [conf_inp(i); markov_w1[prev(i)]] + b)  -- [1, n_blocks]
            ggml_tensor * conf_inp_i = ggml_view_2d(ctx0, conf_inp, conf_inp->ne[0], n_blocks,
                                                    (size_t) block_drafts * conf_inp->nb[1], i*conf_inp->nb[1]);
            ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, conf_inp_i), markov, 0);
            ggml_tensor * conf = ggml_mul_mat(ctx0, model.dspark_conf_proj, feat);
            if (model.dspark_conf_proj_b) {
                conf = ggml_add(ctx0, conf, model.dspark_conf_proj_b);
            }
            conf = ggml_sigmoid(ctx0, conf);

            cat_conf = cat_conf ? ggml_concat(ctx0, cat_conf, conf, 1) : conf;
        }

        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    // cat is position-major; restore ubatch block-major order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, block_drafts, n_blocks]
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    if (has_conf) {
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

llama_model_dflash::graph_dspark::graph_dspark(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    ggml_tensor * inp_tokens = build_inp_tokens();
    ggml_tensor * cur        = build_layers(inp_tokens);

    build_logits(cur);

    // bias the draft logits with the Markov head
    build_dspark_markov_head(*this, model, inp_tokens);
}

// DSV4 DSpark decoder, dual-mode by batch type (see graph_base):
//   * embd batch  -> project main_x through each stage's wkv and inject K into the ring cache
//   * token batch -> noise block through 3 full DSV4 stages (hc + MLA + MoE), markov + confidence heads
llama_model_dflash::graph_dsv4::graph_dsv4(const llm_graph_params & params) :
    llama_model_deepseek4::graph(params, no_build{}) {
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    // KV cache injection: fused target features from the encoder
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        ggml_tensor * inp_g = build_inject_inp(*this, model);
        cb(inp_g, "inp_g_embeddings", -1);

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            // main-track KV: kv_norm(wkv(main_x)) with rope on the trailing dims, same
            // rope parameters as the uncompressed layers in build_attention_impl
            ggml_tensor * kv = build_lora_mm(layer.wkv, inp_g);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);

            kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            kv = ggml_rope_set_offset(kv, n_embd_head_nope);
            cb(kv, "kv_injected", il);

            if (inp_attn->self_k_rot_swa[il]) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    ggml_tensor * tok_embd = get_tok_embd(*this, model);

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
        hc_pre attn_hc = build_hc_pre(inpL,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                il);
        ggml_tensor * cur = attn_hc.cur;
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(inp_attn, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, attn_hc.post, attn_hc.comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        hc_pre ffn_hc = build_hc_pre(inpL,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                il);
        cur = ffn_hc.cur;
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used(),
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

        inpL = build_hc_post(cur, residual, ffn_hc.post, ffn_hc.comb, il);
        cb(inpL, "l_out", il);
    }

    ggml_tensor * cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    // confidence head input: the reference scores the pre-norm collapsed hidden state
    res->t_embd = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    const auto [output, output_s] = get_lm_head(*this, model);
    cur = build_lora_mm(output, cur, output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
}

// DeepSeek-V4.1 DSpark decoder (mtp.0..2), dual-mode by batch type like graph_dsv4:
//   * embd batch  -> writes every stage's kv_norm(wkv(main_proj(target features))) into its window ring
//   * token batch -> [anchor, noise...] block through the stages, single-pass mHC, no per-head q norm
// window of a block = window of its first query, see n_swa

llama_model_dflash::graph_dsv41::graph_dsv41(const llm_graph_params & params) :
    llama_model_deepseek41::graph(params, no_build{}) {
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    GGML_ASSERT(hparams.dsv4_hc_single_pass);

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    // ratio-0 layers: plain rope at rope_theta, no YaRN (original_seq_len 0)
    const auto rope = [&](ggml_tensor * t) {
        t = ggml_rope_ext(ctx0, t, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        return ggml_rope_set_offset(t, n_embd_head_nope);
    };

    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        ggml_tensor * inp_g = build_inject_inp(*this, model);
        cb(inp_g, "main_x", -1);

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * kv = build_lora_mm(layer.wkv, inp_g);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
            kv = rope(kv);
            cb(kv, "kv_injected", il);

            if (inp_attn->self_k_rot_swa[il]) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    ggml_tensor * tok_embd = get_tok_embd(*this, model);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    const int64_t hc            = hparams.dsv4_hc_mult;
    const int64_t n_groups      = hparams.dsv4_o_group_count;
    const int64_t n_heads_group = n_head / n_groups;
    const int64_t o_lora_rank   = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim   = n_heads_group*n_embd_head;
    const float   kq_scale      = 1.0f/sqrtf(float(n_embd_head));

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // the whole block shares the mask row of its first query (the anchor)
    inp_attn->self_attn_mask_swa.block_uniform = true;

    ggml_tensor * pre_prev = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);

        ggml_tensor * cur;
        if (pre_prev == nullptr) {
            // identity pre-mix of the first stage: copy 0
            cur = ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[2], 0));
        } else {
            cur = build_hc_pre(inpL, pre_prev, il);
        }
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // non-causal draft context: block queries see the main-KV ring plus every block key
        {
            ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
            qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(qr, "qr_norm", il);

            ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
            q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
            q = rope(q);
            cb(q, "q", il);

            ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
            kv = rope(kv);
            cb(kv, "kv", il);

            ggml_tensor * out = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    q, kv, kv,
                    nullptr, layer.attn_sinks, nullptr,
                    kq_scale, il);
            cb(out, "attn_raw", il);

            out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, n_tokens);
            out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            out = ggml_rope_set_offset(out, n_embd_head_nope);
            cb(out, "attn_derope", il);

            out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, n_tokens);
            out = ggml_permute(ctx0, out, 0, 2, 1, 3);
            ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
            cb(oa, "attn_wo_a", il);
            oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
            oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, n_tokens);

            cur = build_lora_mm(layer.wo_b, oa);
            cb(cur, "attn_out", il);
        }

        inpL = build_hc_post(cur, residual, attn_mix.post, attn_mix.comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        const hc_mixes ffn_mix = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        pre_prev = ffn_mix.pre;

        cur = build_hc_pre(inpL, attn_mix.pre, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used(),
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

        inpL = build_hc_post(cur, residual, ffn_mix.post, ffn_mix.comb, il);

        ggml_tensor * l_out = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        cb(l_out, "l_out", il);
        ggml_build_forward_expand(gf, l_out);
    }

    // no hc_head: collapse with the last FFN's pre-mix
    ggml_tensor * cur = build_hc_pre(inpL, pre_prev, -1);
    cb(cur, "hc_out", -1);

    // confidence head takes the un-normed collapsed state
    res->t_embd = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    const auto [output, output_s] = get_lm_head(*this, model);
    cur = build_lora_mm(output, cur, output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
}

// Mistral Large 4 DSpark, dual-mode like graph_dsv41:
// - embd batch: write latent K of target_norm(fc(features)) to the cache, no attn norm
// - token batch: decode [anchor, mask_1, ...] block, non-causal
llama_model_dflash::graph_mistral_large4::graph_mistral_large4(const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t kv_lora_rank      = hparams.n_lora_kv;
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_nope  = n_embd_head_k_mla - n_rot;

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head_k_mla));

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    // no LeRoPE in the draft
    const auto rope = [&](ggml_tensor * t) {
        return ggml_rope_ext(ctx0, t, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    };

    const auto build_kv = [&](ggml_tensor * x, int il) -> std::pair<ggml_tensor *, ggml_tensor *> {
        const auto & layer = model.layers[il];

        ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, x);
        cb(kv_cmpr_pe, "kv_cmpr_pe", il);

        ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot), 0);
        kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        cb(kv_cmpr, "kv_cmpr", il);

        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_rot, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_rot),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
        k_pe = rope(k_pe);
        cb(k_pe, "k_pe", il);

        return { ggml_concat(ctx0, kv_cmpr, k_pe, 0), kv_cmpr };
    };

    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        ggml_tensor * inp_g = build_inject_inp(*this, model);
        cb(inp_g, "main_x", -1);

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * kv = build_kv(inp_g, il).first;
            cb(kv, "kv_injected", il);

            if (inp_attn->self_k_rot_swa[il]) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    ggml_tensor * tok_embd = get_tok_embd(*this, model);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    res->add_input(std::move(inp));

    const int64_t n_blocks     = ubatch.n_seqs_unq;
    const int64_t block_drafts = n_blocks > 0 ? n_tokens / n_blocks : 0;

    ggml_tensor * inpL;
    if (block_drafts > 0 && n_tokens % n_blocks == 0 && block_drafts <= (int64_t) hparams.dflash_block_size) {
        // slot k > 0 uses mask row k - 1
        ggml_tensor * anchors = ggml_view_2d(ctx0, inp_tokens, 1, n_blocks, (size_t) block_drafts * inp_tokens->nb[0], 0);
        anchors = ggml_get_rows(ctx0, tok_embd, ggml_cont_1d(ctx0, anchors, n_blocks));
        inpL = ggml_reshape_3d(ctx0, anchors, n_embd, 1, n_blocks);
        if (block_drafts > 1) {
            ggml_tensor * masks = ggml_view_3d(ctx0, model.dspark_mask_embd, n_embd, block_drafts - 1, 1,
                    model.dspark_mask_embd->nb[1], model.dspark_mask_embd->nb[2], 0);
            masks = ggml_repeat_4d(ctx0, ggml_cast(ctx0, masks, GGML_TYPE_F32), n_embd, block_drafts - 1, n_blocks, 1);
            inpL = ggml_concat(ctx0, inpL, masks, 1);
        }
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd, n_tokens);
    } else {
        // not a draft block (e.g. graph reserve)
        inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);
    }
    cb(inpL, "inp_noise_embd", -1);

    // the whole block shares the mask row of its first query (the anchor)
    inp_attn->self_attn_mask_swa.block_uniform = true;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq, cur);
            cb(q, "q", il);

            // {n_embd_head_nope, n_tokens, n_head}
            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k_mla),
                    ggml_row_size(q->type, n_embd_head_k_mla) * n_head, 0);
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

            // {kv_lora_rank, n_head, n_tokens}
            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "q_nope_absorbed", il);

            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_rot, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k_mla),
                    ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
                    ggml_row_size(q->type, n_embd_head_nope));
            q_pe = rope(q_pe);
            cb(q_pe, "q_pe", il);

            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
            cb(Qcur, "Qcur", il);

            auto [Kcur, Vcur] = build_kv(cur, il);
            cb(Kcur, "Kcur", il);

            // apply wv_b after attn, so the K rotation is already undone
            ggml_tensor * out = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur,
                    nullptr, nullptr, nullptr,
                    kq_scale, il);
            cb(out, "attn_latent", il);

            // {kv_lora_rank, n_tokens, n_head} -> {n_embd_head_v_mla, n_tokens, n_head}
            out = ggml_reshape_3d(ctx0, out, kv_lora_rank, n_head, n_tokens);
            out = ggml_mul_mat(ctx0, layer.wv_b, ggml_permute(ctx0, out, 0, 2, 1, 3));
            out = ggml_cont_2d(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3), n_embd_head_v_mla*n_head, n_tokens);

            cur = build_lora_mm(layer.wo, out, layer.wo_s);
            cb(cur, "attn_out", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        inpL = ggml_add(ctx0, cur, ffn_inp);
        cb(inpL, "l_out", il);
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    const auto [output, output_s] = get_lm_head(*this, model);
    ggml_tensor * logits = build_lora_mm(output, cur, output_s);

    // softcap does not apply to the markov bias
    if (hparams.f_final_logit_softcapping) {
        logits = ggml_scale(ctx0, logits, 1.0f / hparams.f_final_logit_softcapping);
        logits = ggml_tanh(ctx0, logits);
        logits = ggml_scale(ctx0, logits, hparams.f_final_logit_softcapping);
    }
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens, cur);
    }
}
