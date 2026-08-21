#include "models.h"

// AMD Instella-MoE: DeepSeek-V3 style MLA + sigmoid-routed MoE.
//
// This file is derived from deepseek2.cpp (the MLA absorption path). The MLA attention math, the
// YaRN kq_scale pre-scaling, sigmoid routing with e_score_correction_bias, the shared experts and
// the leading dense layers are all unchanged from there. Only two things differ, and both are
// tagged inline so a reviewer can find them with a grep:
//
//  - [TAG_INSTELLA_GATED_ATTN] the attention output is scaled by sigmoid(attn_gate @ x) before wo.
//    x is the post-attn_norm input, not the raw residual.
//  - [TAG_INSTELLA_FARSKIP] FarSkip-Collective: the block carries two residual streams, so
//    attention and the FFN read different tensors and run in parallel.
//
// Deliberately not ported from deepseek2.cpp: the q_lora_rank path, the non-absorbed
// (is_mla == false) path, and the llama-4 attention temperature scaling. The convert script
// rejects checkpoints that would need any of them - see InstellaMoEModel in conversion/deepseek.py.

void llama_model_instella_moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm,  false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        // cancel the factor from the convert script
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    switch (hparams.n_layer()) {
        case 27: type = LLM_TYPE_16B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_instella_moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    // note: these are the actual head sizes you get after "decompression" using wv_b
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,        "weight", i), {n_embd, n_head * n_embd_head_k_mla}, 0);
        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);
        layer.wk_b      = create_tensor(tn(LLM_TENSOR_ATTN_K_B,      "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
        layer.wv_b      = create_tensor(tn(LLM_TENSOR_ATTN_V_B,      "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);

        // [TAG_INSTELLA_GATED_ATTN] the only tensor this arch adds over deepseek2. Both the enum
        // and its self_attn.gate_proj mapping already exist for afmoe, so nothing new was needed
        // in llama-arch.cpp or tensor_mapping.py.
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head * n_embd_head_v_mla}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_instella_moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_instella_moe::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
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

    // {n_embd, n_tokens}
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_k();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // [TAG_INSTELLA_FARSKIP]
    // FarSkip-Collective keeps two residual streams: the full one, and one that excludes the
    // routed-expert contribution of the layer that produced it. Attention consumes the latter,
    // which decouples it from the expert combine of the preceding layer.
    //
    // Both start out as the same node. That is what lets layer 0 - dense, so it has no routed
    // experts to exclude - fall out of the uniform loop below instead of needing the separate
    // non-tuple branch that FarSkipDecoderLayer has in modeling_instella_moe.py.
    ggml_tensor * res_full = inpL;
    ggml_tensor * res_nort = inpL;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_moe = (uint32_t) il >= hparams.n_layer_dense_lead;

        // norm
        // [TAG_INSTELLA_FARSKIP] attention normalises the routed-free stream; deepseek2 normalises
        // the single residual here
        cur = build_norm(res_nort, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // [TAG_INSTELLA_GATED_ATTN] computed from the post-attn_norm activations, matching
            // MLAGatedAttention in modeling_instella_moe.py - not from the raw residual
            ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur);
            cb(gate, "attn_gate_proj", il);

            ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            cb(q, "q", il);

            // split into {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            // and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            // {n_embd_head_qk_nope, n_tokens, n_head}
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            cb(q_nope, "q_nope_perm", il);

            // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
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
            cur = build_attn(inp_attn,
                    NULL, NULL, NULL, // [TAG_INSTELLA_GATED_ATTN] wo is applied after the gating
                    Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
            cb(cur, "attn_out", il);

            // [TAG_INSTELLA_FARSKIP] three tensors are sliced here where deepseek2 slices two: the
            // gate and res_full are both still live below. res_nort deliberately is not - it was
            // already consumed by attn_norm above and there is no next layer to feed.
            if (il == n_layer - 1 && inp_out_ids) {
                cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
                gate     = ggml_get_rows(ctx0, gate,     inp_out_ids);
                res_full = ggml_get_rows(ctx0, res_full, inp_out_ids);
            }

            // [TAG_INSTELLA_GATED_ATTN] the only extra arithmetic this arch adds over deepseek2
            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
            cb(cur, "attn_gated", il);

            cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
            cb(cur, "attn_o_proj", il);
        }

        ggml_tensor * attn_out = ggml_add(ctx0, cur, res_full);
        cb(attn_out, "attn_res", il);

        // [TAG_INSTELLA_FARSKIP] the FFN reads the residual from before the attention update, so
        // the two run in parallel. deepseek2 normalises the post-attention residual (attn_out)
        // here, which is what makes its blocks sequential.
        cur = build_norm(res_full, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (!is_moe) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            res_full = ggml_add(ctx0, attn_out, cur);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                model.layers[il].ffn_gate_up_exps);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            // [TAG_INSTELLA_FARSKIP] the two-stream update. res_nort omits moe_out, so the next
            // layer's attention does not depend on this layer's expert combine. res_full sums the
            // same four terms deepseek2 accumulates (attn + residual + shared + routed), only
            // associated differently so res_nort falls out for free.
            res_nort = ggml_add(ctx0, attn_out, ffn_shexp);
            cb(res_nort, "l_out_no_routed", il);

            res_full = ggml_add(ctx0, res_nort, moe_out);
        }

        res_full = build_cvec(res_full, il);
        cb(res_full, "l_out", il);

        if (!is_moe) {
            // [TAG_INSTELLA_FARSKIP] a dense layer has no routed experts, so both streams stay
            // identical - this is the collapse that removes the layer-0 special case
            res_nort = res_full;
        }
    }

    cur = build_norm(res_full, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
