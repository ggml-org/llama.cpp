#include "models.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-hybrid.h"

#include <algorithm>

// question types of a decision model: choice, score, noul
static const uint32_t N_DECISION_TYPES = 3;

void llama_model_lfm2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = hparams.n_head_kv(il) == 0;
    }

    hparams.n_layer_dense_lead = hparams.n_layer();

    switch (hparams.n_ff()) {
        case  2560: type = LLM_TYPE_230M; break;
        case  4608: type = LLM_TYPE_350M; break;
        case  6912: type = LLM_TYPE_700M; break;
        case  8192: type = LLM_TYPE_1_2B; break;
        case 10752: type = LLM_TYPE_2_6B; break;
        default:    type = LLM_TYPE_UNKNOWN;
    }

    ml.get_key(LLM_KV_DECISION_BLOCK_COUNT, hparams.n_layer_decision, false);
    if (hparams.n_layer_decision > 0) {
        if (hparams.n_layer_decision >= hparams.n_layer() || hparams.causal_attn) {
            throw std::runtime_error("invalid decision head");
        }
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        hparams.n_embd_out_impl = N_DECISION_TYPES;
    }

    if (const auto is_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false); is_swa && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            hparams.is_swa_impl[il] = !hparams.is_recr_impl[il];
        }
    }
}

void llama_model_lfm2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM_LFM2, "weight"), {n_embd}, 0);

    if (hparams.n_layer_decision > 0) {
        // decision head: plain pre-norm blocks with biases
        for (int i = n_layer - (int) hparams.n_layer_decision; i < n_layer; ++i) {
            auto & layer = layers[i];
            const int64_t n_ff_head = hparams.n_ff(i);

            layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
            layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias",   i), {n_embd}, 0);

            layer.wqkv   = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, 3 * n_embd}, 0);
            layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   i), {3 * n_embd}, 0);
            layer.wo     = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
            layer.wo_b   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0);

            layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
            layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias",   i), {n_embd}, 0);
            layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff_head}, 0);
            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias",   i), {n_ff_head}, 0);
            layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_head, n_embd}, 0);
            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);
        }

        if (n_token_types != N_DECISION_TYPES) {
            throw std::runtime_error("decision model must have one token type per question type");
        }
        type_embd = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_token_types}, 0);

        cls_norm   = create_tensor(tn(LLM_TENSOR_CLS_NORM, "weight"), {n_embd},    0);
        cls_norm_b = create_tensor(tn(LLM_TENSOR_CLS_NORM, "bias"),   {n_embd},    0);
        cls        = create_tensor(tn(LLM_TENSOR_CLS,      "weight"), {n_embd, n_embd}, 0);
        cls_b      = create_tensor(tn(LLM_TENSOR_CLS,      "bias"),   {n_embd},    0);
        cls_out    = create_tensor(tn(LLM_TENSOR_CLS_OUT,  "weight"), {n_embd, 1}, 0);
        cls_out_b  = create_tensor(tn(LLM_TENSOR_CLS_OUT,  "bias"),   {1},         0);
    } else {
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer - (int) hparams.n_layer_decision; ++i) {
        auto & layer = layers[i];

        const bool is_moe_layer = i >= static_cast<int>(hparams.n_layer_dense_lead);

        // ffn/moe is same for transformer and conv layers
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        if (is_moe_layer) {
            GGML_ASSERT(n_expert && n_expert_used);
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),  {n_embd, n_expert}, 0);
            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, hparams.n_ff_exp(), n_expert}, 0);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {hparams.n_ff_exp(),   n_embd, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i),   {n_embd, hparams.n_ff_exp(), n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        } else {  // dense
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        }

        // for operator_norm
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (!hparams.is_recr(i)) {
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
    if (hparams.n_layer_decision > 0) {
        return std::make_unique<graph_decision>(*this, params);
    }
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

        // causal prepends the state, non-causal pads symmetrically for a centered window
        if (hparams.causal_attn) {
            bx = ggml_concat(ctx0, conv, bx, 0);
        } else {
            const int64_t pad = (hparams.n_shortconv_l_cache - 1) / 2;
            auto * left = ggml_cont(ctx0,
                ggml_view_3d(ctx0, conv, pad, hparams.n_embd, n_seqs, conv->nb[1], conv->nb[2], (d_conv - pad) * conv->nb[0]));
            bx = ggml_pad_ext(ctx0, ggml_concat(ctx0, left, bx, 0), 0, pad, 0, 0, 0, 0, 0, 0);
        }
        GGML_ASSERT(bx->ne[0] > conv->ne[0]);

        // write conv states: slot 0 = the final state, slot s = the state s tokens back (partial rollback)
        const int64_t K         = hparams.causal_attn && cparams.n_rs_seq > 0 ? (int64_t) cparams.n_rs_seq + 1 : 1;
        const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);
        const auto    mem_size  = mctx_cur->get_size();
        const size_t  row_size  = ggml_row_size(conv_state->type, (int64_t) d_conv * n_embd);

        for (int64_t slot = 0; slot < n_written; ++slot) {
            auto * conv_snap = ggml_view_3d(ctx0, bx, d_conv, bx->ne[1], bx->ne[2], bx->nb[1], bx->nb[2],
                                            (bx->ne[0] - d_conv - slot) * ggml_element_size(bx));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_snap,
                                                   ggml_view_2d(ctx0, conv_state, (int64_t) d_conv * n_embd, n_seqs,
                                                                conv_state->nb[1],
                                                                ((size_t) slot * mem_size + kv_head) * row_size)));
        }

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
        res->t_layer_inp[il] = cur;

        const bool is_moe_layer = il >= static_cast<int>(hparams.n_layer_dense_lead);

        auto * prev_cur = cur;
        cur             = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "model.layers.{}.operator_norm", il);

        cur = hparams.is_recr(il) ? build_shortconv_block(cur, inp_hybrid->get_recr(), il) :
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

    if (!cparams.embeddings) {
        cur = build_lora_mm(model.output, cur, model.output_s);
        cb(cur, "result_output", -1);

        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}

// media entries (an image or audio prefix) are embeddings, text entries are tokens
static bool lfm2_is_media(const llama_ubatch & ubatch, int64_t i) {
    return ubatch.is_mixed() ? ubatch.type[i] != 0 : ubatch.token == nullptr;
}

// non-causal within a sequence, the media never reads the text, so it is a function of the media alone
// in the head, the text and the media only read their own kind
class llm_graph_input_attn_media : public llm_graph_input_attn_no_cache {
public:
    llm_graph_input_attn_media(const llama_hparams & hparams, const llama_cparams & cparams, bool is_head) :
        llm_graph_input_attn_no_cache(hparams, cparams), is_head(is_head) {}

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<bool> is_media(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            is_media[i] = lfm2_is_media(*ubatch, i);
        }

        const auto fill_mask = [&](auto * data, auto zero, auto ninf) {
            for (int64_t i1 = 0; i1 < n_tokens; ++i1) {
                for (int64_t i0 = 0; i0 < n_tokens; ++i0) {
                    bool visible = ubatch->seq_id[i0][0] == ubatch->seq_id[i1][0];
                    if (is_head) {
                        visible = visible && is_media[i0] == is_media[i1];
                    } else {
                        visible = visible && !(is_media[i1] && !is_media[i0]);
                    }
                    data[i1 * n_tokens + i0] = visible ? zero : ninf;
                }
            }
        };

        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
        if (self_kq_mask->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_fp32_to_fp16(0.0f), ggml_fp32_to_fp16(-INFINITY));
        } else {
            fill_mask((float *) self_kq_mask->data, 0.0f, -INFINITY);
        }
    }

    const bool is_head;
};

// 1 where the previous (next) token is the left (right) neighbor in the same sequence
// the last media entry does not read the text on its right
class llm_graph_input_conv_mask : public llm_graph_input_i {
public:
    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<float> data_left(n_tokens, 0.0f);
        std::vector<float> data_right(n_tokens, 0.0f);
        for (int64_t i = 0; i + 1 < n_tokens; ++i) {
            const bool is_next = ubatch->seq_id[i][0] == ubatch->seq_id[i + 1][0] && ubatch->pos[i] + 1 == ubatch->pos[i + 1];
            data_right[i]    = is_next && !(lfm2_is_media(*ubatch, i) && !lfm2_is_media(*ubatch, i + 1));
            data_left[i + 1] = is_next;
        }
        ggml_backend_tensor_set(left,  data_left.data(),  0, ggml_nbytes(left));
        ggml_backend_tensor_set(right, data_right.data(), 0, ggml_nbytes(right));
    }

    ggml_tensor * left  = nullptr; // F32 [1, n_tokens]
    ggml_tensor * right = nullptr; // F32 [1, n_tokens]
};

llama_model_lfm2::graph_decision::graph_decision(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int     n_layer_enc = n_layer - hparams.n_layer_decision;

    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.embed_tokens", -1);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    llm_graph_input_attn_no_cache * inp_attn[2];
    for (bool is_head : {false, true}) {
        auto inp = std::make_unique<llm_graph_input_attn_media>(hparams, cparams, is_head);
        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
        inp_attn[is_head] = (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
    }

    auto inp_conv = std::make_unique<llm_graph_input_conv_mask>();
    inp_conv->left  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    inp_conv->right = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    ggml_set_input(inp_conv->left);
    ggml_set_input(inp_conv->right);
    ggml_tensor * conv_left  = inp_conv->left;
    ggml_tensor * conv_right = inp_conv->right;
    res->add_input(std::move(inp_conv));

    for (int il = 0; il < n_layer_enc; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * inpL = cur;
        cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "model.layers.{}.operator_norm", il);

        if (hparams.is_recr(il)) {
            ggml_tensor * bcx = build_lora_mm(layer.shortconv.in_proj, cur);
            cb(bcx, "model.layers.{}.conv.in_proj", il);

            ggml_tensor * b = ggml_view_2d(ctx0, bcx, n_embd, n_tokens, bcx->nb[1], 0 * n_embd * ggml_element_size(bcx));
            ggml_tensor * c = ggml_view_2d(ctx0, bcx, n_embd, n_tokens, bcx->nb[1], 1 * n_embd * ggml_element_size(bcx));
            ggml_tensor * x = ggml_view_2d(ctx0, bcx, n_embd, n_tokens, bcx->nb[1], 2 * n_embd * ggml_element_size(bcx));

            // centred 3-tap conv, a tap outside the sequence reads 0
            ggml_tensor * bx  = ggml_mul(ctx0, b, x);
            ggml_tensor * bxp = ggml_pad_ext(ctx0, bx, 0, 0, 1, 1, 0, 0, 0, 0);
            ggml_tensor * prv = ggml_view_2d(ctx0, bxp, n_embd, n_tokens, bxp->nb[1], 0);
            ggml_tensor * nxt = ggml_view_2d(ctx0, bxp, n_embd, n_tokens, bxp->nb[1], 2 * bxp->nb[1]);

            GGML_ASSERT(hparams.n_shortconv_l_cache == 3);
            ggml_tensor * taps = ggml_cont(ctx0, ggml_transpose(ctx0, layer.shortconv.conv));
            ggml_tensor * tap0 = ggml_view_1d(ctx0, taps, n_embd, 0 * taps->nb[1]);
            ggml_tensor * tap1 = ggml_view_1d(ctx0, taps, n_embd, 1 * taps->nb[1]);
            ggml_tensor * tap2 = ggml_view_1d(ctx0, taps, n_embd, 2 * taps->nb[1]);

            ggml_tensor * y = ggml_mul(ctx0, bx, tap1);
            y = ggml_add(ctx0, y, ggml_mul(ctx0, ggml_mul(ctx0, prv, tap0), conv_left));
            y = ggml_add(ctx0, y, ggml_mul(ctx0, ggml_mul(ctx0, nxt, tap2), conv_right));
            cb(y, "model.layers.{}.conv.conv", il);

            cur = build_lora_mm(layer.shortconv.out_proj, ggml_mul(ctx0, c, y));
            cb(cur, "model.layers.{}.conv.out_proj", il);
        } else {
            auto [q, k, v] = build_qkv(layer, cur, n_embd_head, n_head, hparams.n_head_kv(il), il);

            q = build_norm(q, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            k = build_norm(k, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

            q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                              attn_factor, beta_fast, beta_slow);
            k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                              attn_factor, beta_fast, beta_slow);

            cur = build_attn(inp_attn[0],
                    layer.wo, NULL, layer.wo_s,
                    q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "model.layers.{}.self_attn.out_proj", il);
        }

        cur = ggml_add(ctx0, cur, inpL);

        ggml_tensor * ffn_out = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        ffn_out = build_ffn(ffn_out,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);

        cur = ggml_add(ctx0, cur, ffn_out);
        cb(cur, "l_out", il);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    cur = build_decision_head(model, cur, inp_attn[1], inp_out_ids);

    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}

// same as llama_model_modern_bert::graph::build_decision_head(), with the head counts of the head layers
ggml_tensor * llama_model_lfm2::graph_decision::build_decision_head(
        const llama_model & model,
        ggml_tensor * inp,
        llm_graph_input_attn_no_cache * inp_attn,
        ggml_tensor * inp_out_ids) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int     n_layer_enc = n_layer - hparams.n_layer_decision;

    ggml_tensor * scores = nullptr;

    // the question type is not a graph input, so the head is evaluated for each of them
    for (uint32_t it = 0; it < N_DECISION_TYPES; ++it) {
        ggml_tensor * type_row = ggml_view_1d(ctx0, model.type_embd, n_embd, it * model.type_embd->nb[1]);
        ggml_tensor * inpL = ggml_add(ctx0, inp, type_row);

        for (int il = n_layer_enc; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * cur = build_norm(inpL, layer.attn_norm, layer.attn_norm_b, LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // no positional encoding in the head
            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, hparams.n_head(il), hparams.n_head_kv(il), il);

            cur = build_attn(inp_attn,
                        layer.wo, layer.wo_b, layer.wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp, layer.ffn_norm, layer.ffn_norm_b, LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    layer.ffn_up,   layer.ffn_up_b,   NULL,
                    NULL,           NULL,             NULL,
                    layer.ffn_down, layer.ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_RELU,
                    LLM_FFN_SEQ, il);

            inpL = ggml_add(ctx0, cur, ffn_inp);
        }

        // scorer
        ggml_tensor * cur = build_norm(inpL, model.cls_norm, model.cls_norm_b, LLM_NORM, -1);
        cur = ggml_add(ctx0, build_lora_mm(model.cls, cur), model.cls_b);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_add(ctx0, build_lora_mm(model.cls_out, cur), model.cls_out_b);

        scores = scores ? ggml_concat(ctx0, scores, cur, 0) : cur;
    }
    cb(scores, "decision_scores", -1);

    return scores;
}

// Explicit template instantiations
template struct llama_model_lfm2::graph<true>;
template struct llama_model_lfm2::graph<false>;
