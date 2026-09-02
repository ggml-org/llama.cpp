// XingChen4: DeepSeek-V3 style MLA attention + DeepSeekMoE, with
// Manifold-Constrained Hyper-Connections (MHC) multi-residual-stream blocks and
// a DeepSeek-V3 MTP head.
//
// The layer body (MLA + MoE) mirrors deepseek2.cpp, the MHC helpers mirror
// deepseek4.cpp, with two model-specific differences:
//   * the comb logits are clamped to [-30, 30] before the Sinkhorn iterations
//   * the residual streams are averaged (xc4_hc_mean) at the head, no hc_head

#include "models.h"

#include <cmath>
#include <stdexcept>
#include <string>

static size_t xc4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * xc4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, xc4_elem_offset(t, i0));
}

static ggml_tensor * xc4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], xc4_elem_offset(t, i0));
}

// mean over the hyper-connection streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
static ggml_tensor * xc4_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

static ggml_tensor * xc4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

void llama_model_xingchen4::load_arch_hparams(llama_model_loader & ml) {
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        // fall back for GGUFs that have no expert_gating_func model parameter set
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
    }

    // Manifold-Constrained Hyper-Connections (MHC)
    // Note: reuse the DSV4 hparams members, the saver writes them unconditionally
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult == 4 && "xingchen4 requires 4 residual streams");

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        // cancel the factor from the convert script
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    // NextN/MTP
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn == 0 ||
        hparams.n_layer() + hparams.n_layer_nextn == hparams.n_layer_all);
}

void llama_model_xingchen4::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t xc4_hc_dim      = hparams.dsv4_hc_mult * n_embd;
    const int64_t xc4_hc_mix_dim  = (2 + hparams.dsv4_hc_mult) * hparams.dsv4_hc_mult;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (hparams.n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    const int trunk_flags = mtp_only  ? TENSOR_NOT_REQUIRED : 0;
    int       mtp_flags   = trunk_only ? TENSOR_NOT_REQUIRED : 0;

    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    GGML_ASSERT(hparams.is_mla() && "xingchen4 requires MLA");
    GGML_ASSERT(hparams.n_lora_q > 0 && "xingchen4 requires q_lora_rank");

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // try to load output.weight, if not found, use token_embd (tied embeddings)
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const int flags = i < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, flags);

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);

        // note: only old legacy GGUF files will have the unsplit wkv_b tensor in
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, flags);

        // MHC tensors (trunk layers only)
        if (i < n_layer) {
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {xc4_hc_dim, xc4_hc_mix_dim}, flags);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {xc4_hc_mix_dim}, flags);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {xc4_hc_dim, xc4_hc_mix_dim}, flags);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {xc4_hc_mix_dim}, flags);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED | flags);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, flags);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        // NextN/MTP tensors
        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, mtp_flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, mtp_flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, mtp_flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED | flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_xingchen4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_xingchen4::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "xingchen4 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "xingchen4 MTP currently only supports a single MTP block");
    GGML_ASSERT(hparams.is_mla() && "xingchen4 MTP requires MLA");

    // The appended MTP block is stored immediately after the main decoder layers.
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    GGML_ASSERT((uint32_t) il >= hparams.n_layer_dense_lead && "xingchen4 MTP block expected to use MoE FFN");

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const int64_t kv_lora_rank        = hparams.n_lora_kv;

    GGML_ASSERT(n_embd_head_qk_nope >= 1);
    GGML_ASSERT(hparams.n_lora_q > 0);
    GGML_ASSERT(layer.wq_a);
    GGML_ASSERT(layer.attn_q_a_norm);
    GGML_ASSERT(layer.wq_b);
    GGML_ASSERT(layer.wkv_a_mqa);
    GGML_ASSERT(layer.attn_kv_a_norm);
    GGML_ASSERT(layer.wk_b);

    const bool has_split_exps =
            layer.ffn_up_exps   != nullptr &&
            layer.ffn_gate_exps != nullptr;

    const bool has_fused_exps = layer.ffn_gate_up_exps != nullptr;

    GGML_ASSERT(has_split_exps || has_fused_exps);
    GGML_ASSERT(layer.ffn_norm);
    GGML_ASSERT(layer.ffn_gate_inp);
    GGML_ASSERT(layer.ffn_down_exps);
    GGML_ASSERT(layer.ffn_gate_shexp);
    GGML_ASSERT(layer.ffn_down_shexp);
    GGML_ASSERT(layer.ffn_up_shexp);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens
                ? layer.nextn.embed_tokens
                : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn_k = build_attn_inp_k();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
    cb(q, "mtp_q_a", il);

    q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(q, "mtp_q_a_norm", il);

    q = ggml_mul_mat(ctx0, layer.wq_b, q);
    cb(q, "mtp_q_b", il);

    ggml_tensor * q_nope =
        ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head, 0);
    cb(q_nope, "mtp_q_nope", il);

    ggml_tensor * q_pe =
        ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
                ggml_row_size(q->type, n_embd_head_qk_nope));
    cb(q_pe, "mtp_q_pe", il);

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_cmpr_pe, "mtp_kv_cmpr_pe", il);

    ggml_tensor * kv_cmpr =
        ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    cb(kv_cmpr, "mtp_kv_cmpr", il);

    ggml_tensor * k_pe =
        ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
    cb(k_pe, "mtp_k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "mtp_kv_cmpr_norm", il);

    GGML_ASSERT(ext_factor >= 0.0f);

    const float attn_factor_org =
            attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    const float mscale =
            attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));

    const float kq_scale =
            1.0f * mscale * mscale / sqrtf(float(n_embd_head_k_mla));

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "mtp_q_pe_rope", il);

    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_pe, "mtp_k_pe_rope", il);

    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    cb(q_nope, "mtp_q_nope_perm", il);

    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    cb(q_nope_absorbed, "mtp_q_nope_absorbed", il);

    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
    cb(q_nope_absorbed, "mtp_q_nope_absorbed_perm", il);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
    cb(Qcur, "mtp_Qcur", il);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, hparams.n_lora_kv, 1, n_tokens);
    cb(kv_cmpr, "mtp_kv_cmpr_reshape", il);

    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    cb(Kcur, "mtp_Kcur", il);

    ggml_tensor * Vcur = kv_cmpr;
    cb(Vcur, "mtp_Vcur", il);

    cur = build_attn(inp_attn_k,
            layer.wo, nullptr, layer.wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale, il);
    cb(cur, "mtp_attn_out", il);

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            nullptr,
            layer.ffn_gate_up_exps);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "xingchen4 MTP: missing both nextn.shared_head_norm and output_norm");

    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head
            ? layer.nextn.shared_head_head
            : model.output;

    ggml_tensor * head_s = layer.nextn.shared_head_head
            ? layer.nextn.shared_head_head_s
            : model.output_s;

    GGML_ASSERT(head_w && "xingchen4 MTP: missing LM head (nextn.shared_head_head or model.output)");

    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_xingchen4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * weights,
        int           il) const {
    GGML_UNUSED(il);

    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_xc4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_xc4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_XC4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llama_model_xingchen4::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [src_hc, dst_hc, n_tokens] (ne0 = src). Mirrors the reference
    // mhc_pre_big_fuse_with_clamp_tilelang kernel: softmax over src, then
    // sinkhorn_iterations x (col-norm over src, row-norm over dst); eps is only
    // added to the normalization denominators, never to the values.
    // ggml_soft_max also performs the reference's first column normalization.
    //
    // NOTE: this decomposed fallback (only used when fused_xc4_hc_comb is off)
    // cannot add eps to the first src-normalization because ggml_soft_max
    // divides by the raw sum. The fused ggml_xc4_hc_comb path replicates
    // with_clamp bit-exactly (exp(-max), then all eps-normalizations in the
    // loop); this path differs from it by ~1e-6 (eps missing in the very first
    // src normalization).
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    // normalize over ne1 (dst): each src row sums to 1
    auto norm_dst = [&]() {
        ggml_tensor * sum = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        sum = ggml_sum_rows(ctx0, sum);
        sum = ggml_add(ctx0, sum, eps);
        sum = ggml_permute(ctx0, sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, sum);
    };

    // normalize over ne0 (src): each dst column sums to 1
    auto norm_src = [&]() {
        ggml_tensor * sum = ggml_sum_rows(ctx0, comb);
        sum = ggml_add(ctx0, sum, eps);
        comb = ggml_div(ctx0, comb, sum);
    };

    norm_dst();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_src();
        norm_dst();
    }

    return comb;
}

ggml_tensor * llama_model_xingchen4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = xc4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = xc4_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = xc4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = xc4_view_1d(ctx0, hc_base, hc, hc);

    ggml_tensor * pre = xc4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = xc4_hc_affine(ctx0, pre, scale_pre, base_pre);
    // the reference with_clamp kernel has no hc_pre_eps on the pre gate
    pre = ggml_sigmoid(ctx0, pre);
    cb(pre, "hc_pre", il);

    *post = xc4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = xc4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    if (cparams.fused_xc4_hc_comb) {
        *comb = ggml_xc4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_XC4_HC_COMB, *comb, il});
    } else {
        ggml_tensor * scale_comb = xc4_view_1d(ctx0, hc_scale, 1, 2);
        ggml_tensor * base_comb  = xc4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

        *comb = xc4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = xc4_hc_affine(ctx0, *comb, scale_comb, base_comb);
        // the reference MHC kernel clamps the raw comb logits before Sinkhorn
        *comb = ggml_clamp(ctx0, *comb, -30.0f, 30.0f);
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);

    ggml_tensor * result = build_hc_pre(x, pre, il);
    return result;
}

ggml_tensor * llama_model_xingchen4::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_UNUSED(il);

    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);

    if (cparams.fused_xc4_hc_post) {
        ggml_tensor * result = ggml_xc4_hc_post(ctx0, x, residual, post, comb);
        res->add_fused_node({LLM_FUSED_OP_XC4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            // comb is [src_hc, dst_hc, n_tokens]: the element [src, dst] holds the
            // mixing weight from residual stream src to output stream dst.
            // NOTE: xingchen4's vllm reference (with_clamp kernel) transposes the
            // comb flat index relative to deepseek4's non-clamp kernel, so here we
            // must read comb[src, dst] (deepseek4 reads comb[dst, src]).
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    src*comb->nb[0] + dst*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

llama_model_xingchen4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t hc = hparams.dsv4_hc_mult;

    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    // And also: https://github.com/ggml-org/llama.cpp/pull/17945 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]

    // first cancel the adjustment from llama_hparams::yarn_attn_factor_adjust to get the original attn_factor
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    // use the original attn_factor to pre-scale the kq_scale
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_k = build_attn_inp_k();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // MLA self-attention (identical to deepseek2)
        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(q, "q_a", il);

        q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(q, "q_anorm", il);

        q = ggml_mul_mat(ctx0, layer.wq_b, q);
        cb(q, "q_b", il);

        // split into {n_embd_head_qk_nope, n_head, n_tokens}
        ggml_tensor * q_nope =
            ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                         ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
        cb(q_nope, "q_nope", il);

        // and {n_embd_head_qk_rope, n_head, n_tokens}
        ggml_tensor * q_pe = ggml_view_3d(
            ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
        cb(q_pe, "q_pe_in", il);

        ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
        cb(kv_cmpr_pe, "kv_cmpr_pe", il);

        // split into {kv_lora_rank, n_tokens}
        ggml_tensor * kv_cmpr =
            ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                         ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
        cb(kv_cmpr, "kv_cmpr_in", il);

        // and {n_embd_head_qk_rope, 1, n_tokens}
        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
        cb(k_pe, "k_pe_in", il);

        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q_pe, "q_pe", il);

        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(k_pe, "k_pe", il);

        kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(kv_cmpr, "kv_cmpr", il);

        // {n_embd_head_qk_nope, n_tokens, n_head}
        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        cb(q_nope, "q_nope_perm", il);

        // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        cb(q_nope_absorbed, "q_nope_absorbed", il);

        // {kv_lora_rank, n_head, n_tokens}
        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
        cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

        // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
        // note: rope must go first for in-place context shifting in build_rope_shift()
        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        cb(Qcur, "Qcur", il);

        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        cb(kv_cmpr, "kv_cmpr_reshape", il);

        // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
        ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
        cb(Kcur, "Kcur", il);

        // {kv_lora_rank, 1, n_tokens}
        ggml_tensor * Vcur = kv_cmpr;
        cb(Vcur, "Vcur", il);

        // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
        cur = build_attn(inp_attn_k,
                layer.wo, nullptr, layer.wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale, il);
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_reshape_3d(ctx0,
                    ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, residual, n_embd*hc, n_tokens), inp_out_ids),
                    n_embd, hc, n_outputs);
            post     = ggml_get_rows(ctx0, post, inp_out_ids);
            comb     = ggml_reshape_3d(ctx0,
                    ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, comb, hc*hc, n_tokens), inp_out_ids),
                    hc, hc, n_outputs);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                layer.ffn_up, nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                layer.ffn_gate_up_exps);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        layer.ffn_up_shexp, nullptr, nullptr,
                        layer.ffn_gate_shexp, nullptr, nullptr,
                        layer.ffn_down_shexp, nullptr, nullptr,
                        nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }
    cur = inpL;

    // average the residual streams -> single stream for the head
    cur = xc4_hc_mean(ctx0, cur);
    cb(cur, "hc_mean", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
