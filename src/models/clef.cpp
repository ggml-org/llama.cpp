#include "models.h"

#include "llama-ext.h"

#include <cmath>

void llama_model_clef::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen35::load_arch_hparams(ml);

    ml.get_key(LLM_KV_DECISION_ROUTING_BLOCK_COUNT, n_layer_routing);
    ml.get_key(LLM_KV_DECISION_BLOCK_COUNT,         n_layer_joint);
    ml.get_key(LLM_KV_DECISION_HEAD_COUNT,          n_head_decision);

    if (n_head_decision == 0 || n_layer_routing > LLAMA_MAX_LAYERS || n_layer_joint > LLAMA_MAX_LAYERS) {
        throw std::runtime_error("invalid size of the decision head");
    }

    // used by the head
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    // the output is one score per token, see llama_batch_ext_set_decision_order()
    hparams.n_embd_out_impl = 1;
}

void llama_model_clef::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen35::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const auto * w_memory = ml.get_weight(tn(LLM_TENSOR_DECISION_PROJ_MEMORY, "weight").str().c_str());
    const auto * w_ffn    = ml.get_weight(tn(LLM_TENSOR_DEC_FFN_UP, "weight", 0).str().c_str());
    if (w_memory == nullptr || w_ffn == nullptr) {
        throw std::runtime_error("the decision head is missing");
    }
    const int64_t n_embd_h = w_memory->tensor->ne[1];
    const int64_t n_ff_h   = w_ffn->tensor->ne[1];

    if (n_embd_h % n_head_decision != 0) {
        throw std::runtime_error("invalid width of the decision head");
    }

    auto load_norm = [&](norm & n, llm_tensor type, int64_t size, int il = -1) {
        n.w = il < 0 ? create_tensor(tn(type, "weight"), {size}, 0) : create_tensor(tn(type, "weight", il), {size}, 0);
        n.b = il < 0 ? create_tensor(tn(type, "bias"),   {size}, 0) : create_tensor(tn(type, "bias",   il), {size}, 0);
    };

    auto load_attn = [&](attn & a, llm_tensor q, llm_tensor k, llm_tensor v, llm_tensor o, int il) {
        a.wq = create_tensor(tn(q, "weight", il), {n_embd_h, n_embd_h}, 0);
        a.bq = create_tensor(tn(q, "bias",   il), {n_embd_h}, 0);
        a.wk = create_tensor(tn(k, "weight", il), {n_embd_h, n_embd_h}, 0);
        a.bk = create_tensor(tn(k, "bias",   il), {n_embd_h}, 0);
        a.wv = create_tensor(tn(v, "weight", il), {n_embd_h, n_embd_h}, 0);
        a.bv = create_tensor(tn(v, "bias",   il), {n_embd_h}, 0);
        a.wo = create_tensor(tn(o, "weight", il), {n_embd_h, n_embd_h}, 0);
        a.bo = create_tensor(tn(o, "bias",   il), {n_embd_h}, 0);
    };

    head_layers.resize(n_layer_routing + n_layer_joint);
    for (int il = 0; il < (int) head_layers.size(); ++il) {
        auto & layer = head_layers[il];

        if (il < (int) n_layer_routing) {
            load_norm(layer.cross_norm_kv, LLM_TENSOR_DEC_CROSS_ATTN_NORM_KV, n_embd_h, il);
        } else {
            load_norm(layer.self_norm, LLM_TENSOR_DEC_ATTN_NORM, n_embd_h, il);
            load_attn(layer.self_attn, LLM_TENSOR_DEC_ATTN_Q, LLM_TENSOR_DEC_ATTN_K, LLM_TENSOR_DEC_ATTN_V, LLM_TENSOR_DEC_ATTN_OUT, il);
        }

        load_norm(layer.cross_norm, LLM_TENSOR_DEC_CROSS_ATTN_NORM, n_embd_h, il);
        load_attn(layer.cross_attn, LLM_TENSOR_DEC_CROSS_ATTN_Q, LLM_TENSOR_DEC_CROSS_ATTN_K, LLM_TENSOR_DEC_CROSS_ATTN_V, LLM_TENSOR_DEC_CROSS_ATTN_OUT, il);

        load_norm(layer.ffn_norm, LLM_TENSOR_DEC_FFN_NORM, n_embd_h, il);
        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "weight", il), {n_embd_h, n_ff_h}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "bias",   il), {n_ff_h}, 0);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "weight", il), {n_ff_h, n_embd_h}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "bias",   il), {n_embd_h}, 0);
    }

    load_norm(hidden_norm,         LLM_TENSOR_DECISION_HIDDEN_NORM,         n_embd);
    load_norm(option_summary_norm, LLM_TENSOR_DECISION_OPTION_SUMMARY_NORM, n_embd_h);
    load_norm(field_norm,          LLM_TENSOR_DECISION_FIELD_NORM,          n_embd_h);
    load_norm(option_norm,         LLM_TENSOR_DECISION_OPTION_NORM,         n_embd_h);

    proj_memory          = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_MEMORY,          "weight"), {n_embd, n_embd_h}, 0);
    proj_question        = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_QUESTION,        "weight"), {n_embd, n_embd_h}, 0);
    proj_option_question = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_OPTION_QUESTION, "weight"), {n_embd, n_embd_h}, 0);
    proj_global          = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_GLOBAL,          "weight"), {n_embd, n_embd_h}, 0);
    proj_option_context  = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_OPTION_CONTEXT,  "weight"), {n_embd, n_embd_h}, 0);
    proj_option_lexical  = create_tensor(tn(LLM_TENSOR_DECISION_PROJ_OPTION_LEXICAL,  "weight"), {n_embd, n_embd_h}, 0);

    scales    = create_tensor(tn(LLM_TENSOR_DECISION_SCALES),          {3}, 0);
    type_embd = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"),    {n_embd_h, 3}, 0);

    scorer       = create_tensor(tn(LLM_TENSOR_DECISION_SCORER,     "weight"), {4 * n_embd_h, n_embd_h}, 0);
    scorer_b     = create_tensor(tn(LLM_TENSOR_DECISION_SCORER,     "bias"),   {n_embd_h}, 0);
    scorer_out   = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT, "weight"), {n_embd_h, 1}, 0);
    scorer_out_b = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT, "bias"),   {1}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_clef::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// spans read by the head, [start, end) in ubatch token indices
struct clef_spans {
    struct question {
        int32_t type; // noul, choice, score
        int32_t start;
        int32_t end;
    };
    struct option {
        int32_t question;
        int32_t start;
        int32_t end;
    };
    std::vector<question> questions;
    std::vector<option>   options;
    bool valid = false;
};

// see llama_batch_ext_set_decision_order()
// if the batch has no usable order, returns one empty question with one empty option
static clef_spans clef_get_spans(const llama_ubatch & ubatch) {
    clef_spans res;
    std::vector<bool> has_option;

    // TODO: support multiple sequences
    bool ok = ubatch.decision_order != nullptr && ubatch.n_seqs_unq == 1;

    const int32_t n_tokens = ubatch.n_tokens;
    for (int32_t i = 0; ok && i < n_tokens;) {
        const int32_t order = ubatch.decision_order[i];
        int32_t end = i + 1;
        while (end < n_tokens && ubatch.decision_order[end] == order) {
            end++;
        }
        switch (order) {
            case LLAMA_DECISION_ORDER_NONE:
                break;
            case LLAMA_DECISION_ORDER_QUESTION_NOUL:
            case LLAMA_DECISION_ORDER_QUESTION_CHOICE:
            case LLAMA_DECISION_ORDER_QUESTION_SCORE:
                res.questions.push_back({ order - LLAMA_DECISION_ORDER_QUESTION_NOUL, i, end });
                has_option.push_back(false);
                break;
            case LLAMA_DECISION_ORDER_OPTION:
                ok = !res.questions.empty();
                if (ok) {
                    res.options.push_back({ (int32_t) res.questions.size() - 1, i, end });
                    has_option.back() = true;
                }
                break;
            default:
                ok = false;
        }
        i = end;
    }

    // each question needs an option
    res.valid = ok && !res.questions.empty() && std::find(has_option.begin(), has_option.end(), false) == has_option.end();
    if (!res.valid) {
        res.questions.clear();
        res.options.clear();
        res.questions.push_back({ 0, 0, 0 });
        res.options.push_back({ 0, 0, 0 });
    }
    return res;
}

// pooling matrices and indices computed from the spans
class llama_model_clef::input_decision : public llm_graph_input_i {
public:
    input_decision(const llama_ubatch & ubatch) : n_tokens(ubatch.n_tokens) {
        const auto spans = clef_get_spans(ubatch);
        n_questions = spans.questions.size();
        n_options   = spans.options.size();
    }

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->token);
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens * sizeof(llama_token));

        const auto spans = clef_get_spans(*ubatch);
        GGML_ASSERT(spans.questions.size() == n_questions && spans.options.size() == n_options);

        // the scores are NaN if the batch has a decision order that cannot be used
        const float status_data = spans.valid || ubatch->decision_order == nullptr ? 0.0f : NAN;
        ggml_backend_tensor_set(status, &status_data, 0, ggml_nbytes(status));

        std::vector<float>   pool_q_data(n_tokens * n_questions, 0.0f);
        std::vector<int32_t> types(n_questions);
        for (size_t i = 0; i < n_questions; i++) {
            const auto & q = spans.questions[i];
            for (int32_t j = q.start; j < q.end; j++) {
                pool_q_data[i * n_tokens + j] = 1.0f / (q.end - q.start);
            }
            types[i] = q.type;
        }

        std::vector<float>   pool_o_data(n_tokens * n_options, 0.0f);
        std::vector<int32_t> owner(n_options);
        std::vector<float>   mask(n_options * n_questions, -INFINITY);
        for (size_t i = 0; i < n_options; i++) {
            const auto & o = spans.options[i];
            for (int32_t j = o.start; j < o.end; j++) {
                pool_o_data[i * n_tokens + j] = 1.0f / (o.end - o.start);
            }
            owner[i] = o.question;
            mask[o.question * n_options + i] = 0.0f;
        }

        ggml_backend_tensor_set(pool_q,          pool_q_data.data(), 0, ggml_nbytes(pool_q));
        ggml_backend_tensor_set(pool_o,          pool_o_data.data(), 0, ggml_nbytes(pool_o));
        ggml_backend_tensor_set(question_type,   types.data(),       0, ggml_nbytes(question_type));
        ggml_backend_tensor_set(option_question, owner.data(),       0, ggml_nbytes(option_question));
        ggml_backend_tensor_set(option_mask,     mask.data(),        0, ggml_nbytes(option_mask));
    }

    bool can_reuse(const llm_graph_params & params) override {
        // the values are computed again in set_input(), only the shapes must match
        const auto spans = clef_get_spans(params.ubatch);
        return spans.questions.size() == n_questions && spans.options.size() == n_options;
    }

    ggml_tensor * tokens          = nullptr; // I32 [n_tokens]
    ggml_tensor * pool_q          = nullptr; // F32 [n_tokens, n_questions], mean over the span of the question
    ggml_tensor * pool_o          = nullptr; // F32 [n_tokens, n_options], mean over the span of the option
    ggml_tensor * question_type   = nullptr; // I32 [n_questions]
    ggml_tensor * option_question = nullptr; // I32 [n_options]
    ggml_tensor * option_mask     = nullptr; // F32 [n_options, n_questions], 0 if the option belongs to the question, else -inf
    ggml_tensor * status          = nullptr; // F32 [1], added to the scores: 0, or NaN on invalid input

    const int64_t n_tokens;
    size_t n_questions;
    size_t n_options;
};

// the backbone is copied from llama_model_qwen35::graph, without the memory module
llama_model_clef::graph::graph(const llama_model & model_base, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(static_cast<const llama_model_clef &>(model_base)) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_causal();

    auto inp_decision_ptr = std::make_unique<input_decision>(ubatch);
    auto * inp_decision   = inp_decision_ptr.get();
    res->add_input(std::move(inp_decision_ptr));

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(cur, il);
        } else {
            cur = build_layer_attn(inp_attn, cur, inp_pos, sections, il);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = build_ffn(cur,
            model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
            model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
            model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // the head is always evaluated, so that the graph has the same nodes for every batch
    cur = build_head(cur, inp_decision);

    // row i of the output is the score of option i
    cur = ggml_pad(ctx0, cur, 0, n_tokens - cur->ne[1], 0, 0);
    cur = ggml_add(ctx0, cur, inp_decision->status);
    cb(cur, "result_decision", -1);

    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

// same as build_attn_inp_no_cache(), but the mask is causal even if the batch is processed by the encoder path
llm_graph_input_attn_no_cache * llama_model_clef::graph::build_attn_inp_causal() {
    llama_cparams cparams_causal = cparams;
    cparams_causal.causal_attn = true;

    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams_causal);

    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);
    cb(inp->self_kq_mask, "self_kq_mask", -1);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llama_model_clef::graph::build_layer_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor *                   cur,
        ggml_tensor *                   inp_pos,
        int *                           sections,
        int                             il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    // the Q projection outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

// gated delta net over the whole batch, the conv and recurrent states start from zero and are not kept
ggml_tensor * llama_model_clef::graph::build_layer_attn_linear(
        ggml_tensor * cur,
        int           il) {
    const auto & layer = model.layers[il];

    const int64_t d_inner     = hparams.ssm_d_inner;
    const int64_t head_k_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim  = d_inner / num_v_heads;
    const int64_t n_seqs      = 1;

    ggml_tensor * qkv_mixed = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_tokens, n_seqs);

    ggml_tensor * z = build_lora_mm(layer.wqkv_gate, cur, layer.wqkv_gate_s);

    ggml_tensor * beta = build_lora_mm(layer.ssm_beta, cur, layer.ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);

    ggml_tensor * alpha = build_lora_mm(layer.ssm_alpha, cur, layer.ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_tokens, n_seqs);

    ggml_tensor * gate = ggml_softplus(ctx0, ggml_add(ctx0, alpha, layer.ssm_dt));
    gate = ggml_mul(ctx0, gate, layer.ssm_a);
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_tokens, n_seqs);

    const int64_t conv_kernel_size = layer.ssm_conv1d->ne[0];
    const int64_t conv_channels    = d_inner + 2 * num_k_heads * head_k_dim;

    ggml_tensor * conv_states = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, conv_kernel_size - 1, conv_channels, n_seqs), 0.0f);
    ggml_tensor * conv_input  = ggml_concat(ctx0, conv_states, ggml_transpose(ctx0, qkv_mixed), 0);

    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, layer.ssm_conv1d));
    cb(conv_out, "conv_output_silu", il);

    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_tokens,
            0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_out));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, n_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv * n_tokens,
            ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

    q_conv = build_gdn_l2_norm(ctx0, q_conv, hparams.f_norm_rms_eps);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, hparams.f_norm_rms_eps);

    // note: need explicit repeat only if we are not using the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_tokens, n_seqs);
    }

    ggml_tensor * state = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_v_dim, head_v_dim, num_v_heads, n_seqs), 0.0f);

    ggml_tensor * output = build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il).first;
    cb(output, "attn_output", il);

    // gated normalization
    ggml_tensor * z_4d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_tokens, n_seqs);
    output = build_norm(output, layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
    output = ggml_mul(ctx0, output, ggml_silu(ctx0, z_4d));

    output = ggml_reshape_3d(ctx0, output, head_v_dim * num_v_heads, n_tokens, n_seqs);

    cur = build_lora_mm(layer.ssm_out, output, layer.ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
}

ggml_tensor * llama_model_clef::graph::build_head_norm(ggml_tensor * cur, const norm & n) {
    return build_norm(cur, n.w, n.b, LLM_NORM, -1);
}

// q: [n_embd_h, n_q], kv: [n_embd_h, n_kv], no mask
ggml_tensor * llama_model_clef::graph::build_head_attn(ggml_tensor * q, ggml_tensor * kv, const attn & a) {
    const int64_t n_embd_h = q->ne[0];
    const int64_t n_head_h = model.n_head_decision;
    const int64_t d_head   = n_embd_h / n_head_h;
    const int64_t n_q      = q->ne[1];
    const int64_t n_kv     = kv->ne[1];

    ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, a.wq, q),  a.bq);
    ggml_tensor * Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, a.wk, kv), a.bk);
    ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, a.wv, kv), a.bv);

    Qcur = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Qcur, d_head, n_head_h, n_q),  0, 2, 1, 3); // [d_head, n_q,  n_head]
    Kcur = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kcur, d_head, n_head_h, n_kv), 0, 2, 1, 3); // [d_head, n_kv, n_head]
    Vcur = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Vcur, d_head, n_head_h, n_kv), 1, 2, 0, 3); // [n_kv, d_head, n_head]
    Vcur = ggml_cont(ctx0, Vcur);

    ggml_tensor * kq = ggml_mul_mat(ctx0, Kcur, Qcur); // [n_kv, n_q, n_head]
    kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f / sqrtf(float(d_head)), 0.0f);

    ggml_tensor * cur = ggml_mul_mat(ctx0, Vcur, kq); // [d_head, n_q, n_head]
    cur = ggml_cont_2d(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3), n_embd_h, n_q);

    return ggml_add(ctx0, ggml_mul_mat(ctx0, a.wo, cur), a.bo);
}

ggml_tensor * llama_model_clef::graph::build_head_ffn(ggml_tensor * cur, const head_layer & layer) {
    cur = build_head_norm(cur, layer.ffn_norm);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_up, cur), layer.ffn_up_b);
    cur = ggml_gelu_erf(ctx0, cur);
    return ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_down, cur), layer.ffn_down_b);
}

// ref: JointSchemaHead in joint_schema_model.py of the model repo
ggml_tensor * llama_model_clef::graph::build_head(ggml_tensor * hidden, input_decision * inp) {
    const int64_t n_questions = inp->n_questions;
    const int64_t n_options   = inp->n_options;

    inp->tokens          = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp->pool_q          = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_questions);
    inp->pool_o          = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_options);
    inp->question_type   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_questions);
    inp->option_question = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_options);
    inp->option_mask     = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_options, n_questions);
    inp->status          = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { inp->tokens, inp->pool_q, inp->pool_o, inp->question_type, inp->option_question, inp->option_mask, inp->status }) {
        ggml_set_input(t);
    }

    hidden = build_head_norm(hidden, model.hidden_norm);

    ggml_tensor * memory = ggml_mul_mat(ctx0, model.proj_memory, hidden); // [n_embd_h, n_tokens]

    const int64_t n_embd_h = memory->ne[0];

    // hidden state of the last token
    ggml_tensor * global = ggml_view_2d(ctx0, hidden, n_embd, 1, hidden->nb[1], (n_tokens - 1) * hidden->nb[1]);

    // mean of the hidden states over each span
    ggml_tensor * hidden_t = ggml_cont(ctx0, ggml_transpose(ctx0, hidden));
    ggml_tensor * question = ggml_mul_mat(ctx0, hidden_t, inp->pool_q); // [n_embd, n_questions]
    ggml_tensor * opt_ctx  = ggml_mul_mat(ctx0, hidden_t, inp->pool_o); // [n_embd, n_options]

    // mean of the output embeddings of the tokens of each option
    ggml_tensor * lexical = ggml_get_rows(ctx0, model.output, inp->tokens);
    lexical = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, lexical)), inp->pool_o); // [n_embd, n_options]

    // one query per option
    ggml_tensor * options = ggml_mul_mat(ctx0, model.proj_option_context, opt_ctx);
    options = ggml_add(ctx0, options, ggml_mul_mat(ctx0, model.proj_option_lexical, lexical));
    options = ggml_add(ctx0, options, ggml_get_rows(ctx0, ggml_mul_mat(ctx0, model.proj_option_question, question), inp->option_question));

    // the options read the prompt
    for (uint32_t il = 0; il < model.n_layer_routing; ++il) {
        const auto & layer = model.head_layers[il];

        ggml_tensor * cur = build_head_attn(
                build_head_norm(options, layer.cross_norm),
                build_head_norm(memory,  layer.cross_norm_kv),
                layer.cross_attn);
        options = ggml_add(ctx0, options, cur);
        options = ggml_add(ctx0, options, build_head_ffn(options, layer));
    }
    cb(options, "decision_options", -1);

    // one vector per question: its text, a summary of its options, the end of the prompt and its type
    ggml_tensor * fields = ggml_mul_mat(ctx0, model.proj_question, question); // [n_embd_h, n_questions]
    {
        // each question weights its own options
        ggml_tensor * weights = ggml_mul_mat(ctx0, options, fields); // [n_options, n_questions]
        weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(n_embd_h)));
        weights = ggml_soft_max(ctx0, ggml_add(ctx0, weights, inp->option_mask));

        ggml_tensor * summary = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, options)), weights);

        fields = ggml_add(ctx0, fields, build_head_norm(summary, model.option_summary_norm));
        fields = ggml_add(ctx0, fields, ggml_mul_mat(ctx0, model.proj_global, global));
        fields = ggml_add(ctx0, fields, ggml_get_rows(ctx0, model.type_embd, inp->question_type));
    }

    // the questions read each other and the prompt
    for (uint32_t il = model.n_layer_routing; il < model.head_layers.size(); ++il) {
        const auto & layer = model.head_layers[il];

        ggml_tensor * cur = build_head_norm(fields, layer.self_norm);
        fields = ggml_add(ctx0, fields, build_head_attn(cur, cur, layer.self_attn));

        cur = build_head_norm(fields, layer.cross_norm);
        fields = ggml_add(ctx0, fields, build_head_attn(cur, memory, layer.cross_attn));

        fields = ggml_add(ctx0, fields, build_head_ffn(fields, layer));
    }
    fields = build_head_norm(fields, model.field_norm);
    cb(fields, "decision_fields", -1);

    const float eps = 1e-12f;

    // prior: the output embeddings of the option against the question and the end of the prompt
    ggml_tensor * anchor = ggml_l2_norm(ctx0, ggml_add(ctx0, question, global), eps);
    anchor = ggml_get_rows(ctx0, anchor, inp->option_question);
    ggml_tensor * prior = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_l2_norm(ctx0, lexical, eps), anchor)); // [1, n_options]

    // joint: each option against the vector of its question
    options = build_head_norm(options, model.option_norm);
    ggml_tensor * field = ggml_get_rows(ctx0, fields, inp->option_question); // [n_embd_h, n_options]

    ggml_tensor * cosine = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_l2_norm(ctx0, field, eps), ggml_l2_norm(ctx0, options, eps)));

    ggml_tensor * features = ggml_concat(ctx0, field, options, 0);
    features = ggml_concat(ctx0, features, ggml_mul(ctx0, field, options), 0);
    features = ggml_concat(ctx0, features, ggml_abs(ctx0, ggml_sub(ctx0, field, options)), 0);

    ggml_tensor * residual = ggml_add(ctx0, ggml_mul_mat(ctx0, model.scorer, features), model.scorer_b);
    residual = ggml_gelu_erf(ctx0, residual);
    residual = ggml_add(ctx0, ggml_mul_mat(ctx0, model.scorer_out, residual), model.scorer_out_b); // [1, n_options]

    auto scale = [&](int i) {
        return ggml_view_1d(ctx0, model.scales, 1, i * ggml_element_size(model.scales));
    };

    ggml_tensor * joint = ggml_add(ctx0, ggml_mul(ctx0, cosine, scale(1)), residual);

    return ggml_add(ctx0, ggml_mul(ctx0, prior, scale(0)), ggml_mul(ctx0, joint, scale(2)));
}
