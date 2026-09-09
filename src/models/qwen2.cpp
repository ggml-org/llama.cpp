#include "models.h"

void llama_model_qwen2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_5B : LLM_TYPE_1B; break;
        case 28: type = hparams.n_embd == 1536 ? LLM_TYPE_1_5B : LLM_TYPE_7B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 36: type = LLM_TYPE_3B; break;
        case 40: type = hparams.n_head() == 20 ? LLM_TYPE_4B : LLM_TYPE_13B; break;
        case 48: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        case 80: type = LLM_TYPE_70B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_b    = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // --- OpenMythos-style weight-tied recurrence ---
    // One decoder layer applied T times with frozen anchor injection + LTI state.
    // h_{t+1} = A * h_t + B * e + decoder(h_t + e)
    const int RECURRENT_T   = [] { const char * v = std::getenv("RECURRENT_T");   return v ? std::atoi(v) : 1; }();
    const float RECURRENT_A = [] { const char * v = std::getenv("RECURRENT_A");   return v ? std::atof(v) : 0.90f; }();
    const float RECURRENT_B = [] { const char * v = std::getenv("RECURRENT_B");   return v ? std::atof(v) : 0.10f; }();
    const int n_rec_layer   = [] { const char * v = std::getenv("RECURRENT_LAYER"); return v ? std::atoi(v) : -1; }();
    const float kq_scale    = 1.0f / sqrtf(float(n_embd_head));

    auto qwen2_decoder = [&](int il, ggml_tensor * input) -> ggml_tensor* {
        ggml_tensor * cur_a = build_norm(input, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur_a, "attn_norm", il);

        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur_a,
                n_embd_head, n_head, n_head_kv, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        ggml_tensor * cur   = build_attn(inp_attn,
                model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il, true);
        cb(cur, "attn_out", il);

        ggml_tensor * inpSA = input;
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        return cur;
    };

    if (RECURRENT_T > 1 && n_rec_layer >= 0 && n_rec_layer < n_layer) {
        for (int il = 0; il < n_rec_layer; ++il) inpL = qwen2_decoder(il, inpL);
        ggml_tensor * anchor_e = inpL;
        ggml_tensor * h = inpL;
        for (int t = 0; t < RECURRENT_T; ++t) {
            ggml_tensor * combined  = ggml_add(ctx0, h, anchor_e);
            ggml_tensor * block_out = qwen2_decoder(n_rec_layer, combined);
            h = ggml_add(ctx0, ggml_add(ctx0, ggml_scale(ctx0, h, RECURRENT_A),
                                              ggml_scale(ctx0, anchor_e, RECURRENT_B)),
                         block_out);
        }
        inpL = h;
        for (int il = n_rec_layer + 1; il < n_layer; ++il) inpL = qwen2_decoder(il, inpL);
    } else {
        for (int il = 0; il < n_layer; ++il) inpL = qwen2_decoder(il, inpL);
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    if (model.output_b != nullptr) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
