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

    int N = n_layer;
    int S = 50; // Устойчивость
    int D = 4;  // Глубина рекуррентности

    if (const char * env_s = std::getenv("RECURRENT_S")) {
        S = std::atoi(env_s);
    }
    if (const char * env_d = std::getenv("RECURRENT_D")) {
        D = std::atoi(env_d);
    }

    int k = N / 4;
    int r = N % 4;

    int size1 = k, size2 = k, size3 = k, size4 = k + r;
    int start1 = 0, start2 = k, start3 = 2 * k, start4 = 3 * k;

    int offset1 = (S * (size1 - 1)) / 100;
    int offset2 = (S * (size2 - 1)) / 100;
    int offset3 = (S * (size3 - 1)) / 100;
    int offset4 = (S * (size4 - 1)) / 100;

    // L1 is not looped, used only for understanding
    // int L1 = start1 + offset1; 
    int L2 = start2 + offset2;
    int L3 = start3 + offset3;
    int L4 = start4 + offset4;

    int c2 = (D + 3) / 6;
    int c3 = (D + 1) / 2;
    int c4 = D - c2 - c3;

    if (const char * env_c2 = std::getenv("RECURRENT_C2")) c2 = std::atoi(env_c2);
    if (const char * env_c3 = std::getenv("RECURRENT_C3")) c3 = std::atoi(env_c3);
    if (const char * env_c4 = std::getenv("RECURRENT_C4")) c4 = std::atoi(env_c4);

    for (int il = 0; il < n_layer; ++il) {
        int iters = 1;
        if (il == L2) iters = c2;
        else if (il == L3) iters = c3;
        else if (il == L4) iters = c4;

        ggml_tensor * e = inpL; // save for LTI injection

        for (int iter = 0; iter < iters; ++iter) {
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
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il, iter == 0);
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

            if (iters > 1) {
                // Euler-scaling: h_{t+1} = alpha * cur + (1 - alpha) * inpSA
                // where alpha = 1.0 / iters (or customizable via env)
                float alpha = 1.0f / iters;
                if (const char * env_a = std::getenv("RECURRENT_ALPHA")) {
                    alpha = std::atof(env_a);
                }
                float beta = 1.0f - alpha;
                if (const char * env_b = std::getenv("RECURRENT_BETA")) {
                    beta = std::atof(env_b);
                }
                ggml_tensor * scaled_h = ggml_scale(ctx0, cur, alpha);
                ggml_tensor * scaled_inp = ggml_scale(ctx0, inpSA, beta);
                cur = ggml_add(ctx0, scaled_h, scaled_inp);
            }

            // input for next layer or next iteration
            inpL = cur;
        }
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
