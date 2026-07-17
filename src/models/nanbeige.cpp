#include "models.h"

void llama_model_nanbeige::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    uint32_t n_loops_u = 1;
    ml.get_key(LLM_KV_NUM_LOOPS, n_loops_u, false);
    n_loops = n_loops_u < 1 ? 1 : (int) n_loops_u;

    skip_loop_final_norm = false;
    ml.get_key(LLM_KV_SKIP_LOOP_FINAL_NORM, skip_loop_final_norm, false);

    n_layer_phys = (int) hparams.n_layer();

    // Expand logical layer count before load_tensors() allocates layers / KV.
    if (n_loops > 1) {
        GGML_ASSERT(n_layer_phys * n_loops <= (int) LLAMA_MAX_LAYERS);
        for (int j = 1; j < n_loops; ++j) {
            for (int i = 0; i < n_layer_phys; ++i) {
                const int dst = i + j * n_layer_phys;
                hparams.n_head_arr[dst]    = hparams.n_head_arr[i];
                hparams.n_head_kv_arr[dst] = hparams.n_head_kv_arr[i];
                hparams.n_ff_arr[dst]      = hparams.n_ff_arr[i];
                hparams.is_swa_impl[dst]   = hparams.is_swa_impl[i];
                hparams.is_recr_impl[dst]  = hparams.is_recr_impl[i];
            }
        }
        hparams.n_layer_all = (uint32_t) (n_layer_phys * n_loops);
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_nanbeige::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    const int n_phys = n_layer_phys > 0 ? n_layer_phys : n_layer;
    for (int i = 0; i < n_phys; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2},
                TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }

    // Share physical weights across loops; each slot still has its own KV index.
    if (n_loops > 1) {
        for (int j = 1; j < n_loops; ++j) {
            for (int i = 0; i < n_phys; ++i) {
                layers[i + j * n_phys] = layers[i];
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_nanbeige::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_nanbeige::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params) {
    const auto & nb = static_cast<const llama_model_nanbeige &>(model);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int n_phys  = nb.n_layer_phys > 0 ? nb.n_layer_phys : (int) n_layer;
    const int n_loops = nb.n_loops > 0 ? nb.n_loops : 1;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f / sqrtf(float(n_embd_head))
        : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;

        if (n_loops > 1 &&
            ((il + 1) % n_phys) == 0 &&
            (il + 1) < n_layer &&
            !nb.skip_loop_final_norm) {
            cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "loop_norm", il);
            inpL = cur;
        }
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

