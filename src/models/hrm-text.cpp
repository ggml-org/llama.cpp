#include "models.h"

// HRM-Text: alternating low/high transformer stacks over the same token stream.
// Reference: HrmTextModel in transformers, DFM Mimir 1B.

void llama_model_hrm_text::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);

    ml.get_key(LLM_KV_LAYERS_PER_STACK, hparams.n_hrm_layers_per_stack);
    ml.get_key(LLM_KV_H_CYCLES, hparams.n_hrm_h_cycles);
    ml.get_key(LLM_KV_L_CYCLES, hparams.n_hrm_l_cycles);

    // prefix-LM prefill is not implemented (causal attention only); kept for round-trip
    ml.get_key(LLM_KV_PREFIX_LM, hparams.hrm_prefix_lm, false);

    GGML_ASSERT(hparams.n_hrm_layers_per_stack > 0);
    GGML_ASSERT(hparams.n_hrm_h_cycles > 0);
    GGML_ASSERT(hparams.n_hrm_l_cycles > 0);

    // the GGUF block count is the expanded cache-slot count
    const uint32_t n_slot = hparams.n_hrm_layers_per_stack * hparams.n_hrm_h_cycles * (hparams.n_hrm_l_cycles + 1);
    GGML_ASSERT(hparams.n_layer() == n_slot);

    switch (hparams.n_embd) {
        case 1536:
            type = LLM_TYPE_1B;
            break;
        default:
            type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_hrm_text::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    hrm_z_l_init = create_tensor(tn(LLM_TENSOR_HRM_Z_L_INIT), { n_embd }, 0);

    const int lps = hparams.n_hrm_layers_per_stack;

    // blocks [0, lps) hold the low stack, blocks [lps, 2*lps) hold the high stack.
    // the first low and high passes create the layers; later passes alias them.
    const int l_first = 0;
    const int h_first = hparams.n_hrm_l_cycles * lps;

    for (int h = 0; h < (int) hparams.n_hrm_h_cycles; ++h) {
        for (int l = 0; l < (int) hparams.n_hrm_l_cycles + 1; ++l) {
            const int slot_base = (h * (hparams.n_hrm_l_cycles + 1) + l) * lps;
            const int blk_base  = l == (int) hparams.n_hrm_l_cycles ? lps : 0;

            if (h > 0 || (l > 0 && l < (int) hparams.n_hrm_l_cycles)) {
                // alias pass: these cache slots hold the same layers as the first passes
                const int src_base = l == (int) hparams.n_hrm_l_cycles ? h_first : l_first;
                for (int il = 0; il < lps; ++il) {
                    layers[slot_base + il] = layers[src_base + il];
                }
                continue;
            }

            for (int il = 0; il < lps; ++il) {
                auto &    layer = layers[slot_base + il];
                const int bid   = blk_base + il;

                create_tensor_qkv(layer, bid, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);

                // sigmoid attention gate, applied to the attention output before o_proj
                layer.wqkv_gate =
                    create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", bid), { n_embd, n_embd_head_k * n_head }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", bid), { n_embd_head_k * n_head, n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", bid), { n_embd, n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", bid), { n_ff, n_embd }, 0);
                layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", bid), { n_embd, n_ff }, 0);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_hrm_text::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// one stack invocation: lps pre-norm decoder layers, then the parameterless final norm
ggml_tensor * llama_model_hrm_text::graph::build_stack(llm_graph_input_attn_kv * inp_attn,
                                                       ggml_tensor *             inp_pos,
                                                       ggml_tensor *             cur,
                                                       int                       slot_base) const {
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    const int lps = model.hparams.n_hrm_layers_per_stack;

    for (int il = 0; il < lps; ++il) {
        const int    s     = slot_base + il;
        const auto & layer = model.layers[s];

        ggml_tensor * inpSA = cur;

        cur = build_norm(cur, nullptr, nullptr, LLM_NORM_RMS, s);
        cb(cur, "attn_norm", s);

        // sigmoid-gated self-attention (same shape as qwen3next attention layers)
        {
            ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, cur);
            cb(gate, "attn_gate_proj", s);

            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head_k, n_head, n_head_kv, s);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", s);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur", s);

            cur = build_attn(inp_attn, nullptr, nullptr, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale,
                             s);
            cb(cur, "attn_pregate", s);

            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "attn_gate_sigmoid", s);

            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_gated", s);

            cur = build_lora_mm(layer.wo, cur, layer.wo_s);
            cb(cur, "attn_out", s);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_add", s);

        inpSA = cur;
        cur   = build_norm(cur, nullptr, nullptr, LLM_NORM_RMS, s);
        cb(cur, "ffn_norm", s);

        cur = build_ffn(cur, layer.ffn_up, nullptr, nullptr, layer.ffn_gate, nullptr, nullptr, layer.ffn_down, nullptr,
                        nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, s);
        cb(cur, "ffn_out", s);

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "ffn_add", s);

        cur = build_cvec(cur, s);
        cb(cur, "l_out", s);
    }

    cur = build_norm(cur, nullptr, nullptr, LLM_NORM_RMS, slot_base);
    cb(cur, "stack_norm", slot_base);

    return cur;
}

llama_model_hrm_text::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params),
    model(model) {
    ggml_tensor * cur;

    // {n_embd, n_tokens}, scaled by hparams.f_embedding_scale inside build_inp_embd
    ggml_tensor * zH = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // the learned low-cycle state is [n_embd]; binary ops broadcast it over [n_embd, n_tokens]
    ggml_tensor * zL = model.hrm_z_l_init;

    for (uint32_t h = 0; h < model.hparams.n_hrm_h_cycles; ++h) {
        for (uint32_t l = 0; l < model.hparams.n_hrm_l_cycles; ++l) {
            const int slot_base = (h * (model.hparams.n_hrm_l_cycles + 1) + l) * model.hparams.n_hrm_layers_per_stack;

            zL = build_stack(inp_attn, inp_pos, ggml_add(ctx0, zH, zL), slot_base);
        }

        const int slot_base = (h * (model.hparams.n_hrm_l_cycles + 1) + model.hparams.n_hrm_l_cycles) *
                              model.hparams.n_hrm_layers_per_stack;

        zH = build_stack(inp_attn, inp_pos, ggml_add(ctx0, zH, zL), slot_base);
    }

    cur = zH;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
