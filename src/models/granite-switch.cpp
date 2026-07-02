#include "models.h"

#include <cmath>

void llama_model_granite_switch::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);
    ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale, false);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE,             hparams.f_attention_scale, false);

    bool rope_finetuned = true;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, /* required */ false);

    ml.get_key(LLM_KV_NUM_ADAPTERS,  n_adapters);
    ml.get_key(LLM_KV_MAX_LORA_RANK, max_lora_rank);
    ml.get_key(LLM_KV_ROUTER_GAIN, router_gain, /* required */ false);

    std::vector<llama_token> token_ids;
    std::vector<llama_token> substitute_ids;
    ml.get_arr(LLM_KV_ADAPTER_TOKEN_IDS,            token_ids);
    ml.get_arr(LLM_KV_ADAPTER_SUBSTITUTE_TOKEN_IDS, substitute_ids);

    GGML_ASSERT(token_ids.size() == n_adapters);
    GGML_ASSERT(substitute_ids.size() == n_adapters);

    adapter_token_to_slot.clear();
    adapter_token_to_substitute.clear();
    for (uint32_t i = 0; i < n_adapters; ++i) {
        // adapter i -> stacked slot i+1 (slot 0 is the base/zero delta)
        adapter_token_to_slot[token_ids[i]]       = (int32_t) (i + 1);
        adapter_token_to_substitute[token_ids[i]] = substitute_ids[i];
    }

    // extra single-head attention layer at the END (index n_real) holds the router
    // K/V. reusing n_layer_nextn keeps n_layer() == n_real, so the regular layers
    // keep their indices and the KV cache shift/defrag skips the router layer
    const uint32_t n_real = hparams.n_layer();
    hparams.router_layer  = (int32_t) n_real;
    hparams.n_layer_all   = n_real + 1;
    hparams.n_layer_nextn = 1;

    hparams.n_head_arr[n_real]    = 1;
    hparams.n_head_kv_arr[n_real] = 1;
    hparams.n_ff_arr[n_real]      = 0;
}

void llama_model_granite_switch::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_slots_i64 = (int64_t) n_slots();
    const int64_t n_rank      = (int64_t) max_lora_rank;
    const int64_t n_embd_q    = n_embd_head_k * n_head;
    const int64_t n_embd_kv   = n_embd_k_gqa;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd_q + 2*n_embd_kv}, 0);
        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        auto & sl = layer.switch_lora;

        sl.a_q = create_tensor(tn(LLM_TENSOR_ATTN_Q, "lora_a", i), {n_embd,  n_rank, n_slots_i64}, 0);
        sl.b_q = create_tensor(tn(LLM_TENSOR_ATTN_Q, "lora_b", i), {n_rank, n_embd_q, n_slots_i64}, 0);
        sl.a_k = create_tensor(tn(LLM_TENSOR_ATTN_K, "lora_a", i), {n_embd,  n_rank, n_slots_i64}, 0);
        sl.b_k = create_tensor(tn(LLM_TENSOR_ATTN_K, "lora_b", i), {n_rank, n_embd_kv, n_slots_i64}, 0);
        sl.a_v = create_tensor(tn(LLM_TENSOR_ATTN_V, "lora_a", i), {n_embd,  n_rank, n_slots_i64}, 0);
        sl.b_v = create_tensor(tn(LLM_TENSOR_ATTN_V, "lora_b", i), {n_rank, n_embd_kv, n_slots_i64}, 0);

        sl.a_o = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "lora_a", i), {n_embd_q, n_rank, n_slots_i64}, 0);
        sl.b_o = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "lora_b", i), {n_rank,   n_embd, n_slots_i64}, 0);

        sl.a_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "lora_a", i), {n_embd, n_rank, n_slots_i64}, 0);
        sl.b_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "lora_b", i), {n_rank,  n_ff,  n_slots_i64}, 0);
        sl.a_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "lora_a", i), {n_embd, n_rank, n_slots_i64}, 0);
        sl.b_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "lora_b", i), {n_rank,  n_ff,  n_slots_i64}, 0);
        sl.a_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "lora_a", i), {  n_ff, n_rank, n_slots_i64}, 0);
        sl.b_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "lora_b", i), {n_rank, n_embd, n_slots_i64}, 0);
    }
}

class llm_graph_input_switch : public llm_graph_input_i {
public:
    llm_graph_input_switch(const llama_model_granite_switch & smodel) : smodel(smodel) {}
    virtual ~llm_graph_input_switch() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * sub_tokens  = nullptr; // I32 [n_tokens] adapter-substituted token ids
    ggml_tensor * router_ksig = nullptr; // F32 [n_tokens] router K signal (±gain)
    ggml_tensor * router_vval = nullptr; // F32 [n_tokens] router V value (adapter slot / 0)
    ggml_tensor * router_q    = nullptr; // F32 [n_tokens] router Q value (constant 1.0)

    const llama_model_granite_switch & smodel;
};

// K dim-0 is +gain for an adapter token, -gain otherwise; the causal softmax then
// lets a single visible adapter token dominate so the readback recovers its slot.
void llm_graph_input_switch::set_input(const llama_ubatch * ubatch) {
    if (!ubatch->token) {
        return;
    }

    const int64_t n_tokens = ubatch->n_tokens;

    std::vector<int32_t> sub (n_tokens);
    std::vector<float>   ksig(n_tokens);
    std::vector<float>   vval(n_tokens);
    std::vector<float>   q   (n_tokens, 1.0f);

    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_token tok = ubatch->token[i];

        const auto it = smodel.adapter_token_to_slot.find(tok);
        if (it != smodel.adapter_token_to_slot.end()) {
            ksig[i] = +smodel.router_gain;
            vval[i] = (float) it->second;
        } else {
            ksig[i] = -smodel.router_gain;
            vval[i] = 0.0f;
        }

        const auto sit = smodel.adapter_token_to_substitute.find(tok);
        sub[i] = (sit != smodel.adapter_token_to_substitute.end())
            ? (int32_t) sit->second
            : (int32_t) tok;
    }

    ggml_backend_tensor_set(sub_tokens,  sub.data(),  0, n_tokens*ggml_element_size(sub_tokens));
    ggml_backend_tensor_set(router_ksig, ksig.data(), 0, n_tokens*ggml_element_size(router_ksig));
    ggml_backend_tensor_set(router_vval, vval.data(), 0, n_tokens*ggml_element_size(router_vval));
    ggml_backend_tensor_set(router_q,    q.data(),    0, n_tokens*ggml_element_size(router_q));
}

std::unique_ptr<llm_graph_context> llama_model_granite_switch::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// per-token switched LoRA delta: B_a*(A_a*x), adapter selected per token via ids.
// cur: {n_in, n_tokens}, ids: {n_tokens} -> {n_out, n_tokens}
ggml_tensor * llama_model_granite_switch::graph::build_switched_lora_delta(
          ggml_tensor * lora_a,
          ggml_tensor * lora_b,
          ggml_tensor * cur,
          ggml_tensor * ids) {
    const int64_t n_in     = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];

    ggml_tensor * x    = ggml_reshape_3d(ctx0, cur, n_in, 1, n_tokens);
    ggml_tensor * ids2 = ggml_reshape_2d(ctx0, ids, 1, n_tokens);

    ggml_tensor * a = ggml_mul_mat_id(ctx0, lora_a, x, ids2); // {max_rank, 1, n_tokens}
    ggml_tensor * d = ggml_mul_mat_id(ctx0, lora_b, a, ids2); // {n_out,    1, n_tokens}

    return ggml_reshape_2d(ctx0, d, d->ne[0], n_tokens);
}

ggml_tensor * llama_model_granite_switch::graph::build_switched_lora_mm(
          ggml_tensor * w,
          ggml_tensor * lora_a,
          ggml_tensor * lora_b,
          ggml_tensor * cur,
          ggml_tensor * ids) {
    ggml_tensor * base  = ggml_mul_mat(ctx0, w, cur);
    ggml_tensor * delta = build_switched_lora_delta(lora_a, lora_b, cur, ids);
    return ggml_add(ctx0, base, delta);
}

llama_model_granite_switch::graph::graph(
    const llama_model & model,
    const llm_graph_params & params)
    : llm_graph_context(params) {

    const auto & smodel = static_cast<const llama_model_granite_switch &>(model);

    // TODO: support raw embedding input (multimodal / pre-embedded tokens) when needed
    GGML_ASSERT(ubatch.token && "granite-switch requires token input");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    auto inp_switch = std::make_unique<llm_graph_input_switch>(smodel);
    inp_switch->sub_tokens  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp_switch->router_ksig = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_tokens);
    inp_switch->router_vval = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_tokens);
    inp_switch->router_q    = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_tokens);
    ggml_set_input(inp_switch->sub_tokens);
    ggml_set_input(inp_switch->router_ksig);
    ggml_set_input(inp_switch->router_vval);
    ggml_set_input(inp_switch->router_q);
    ggml_tensor * sub_tokens  = inp_switch->sub_tokens;
    ggml_tensor * router_ksig = inp_switch->router_ksig;
    ggml_tensor * router_vval = inp_switch->router_vval;
    ggml_tensor * router_q    = inp_switch->router_q;
    res->add_input(std::move(inp_switch));

    // embed the substituted ids directly; build_inp_embd would embed the raw tokens
    ggml_tensor * inpL = ggml_get_rows(ctx0, model.tok_embd, sub_tokens);
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
    }
    cb(inpL, "inp_embd", -1);

    ggml_tensor * inp_pos = nullptr;
    if (hparams.rope_finetuned) {
        inp_pos = build_inp_pos();
    }
    auto * inp_attn = build_attn_inp_kv();

    // single causal head at layer R recovers the adapter index in-graph: only dim 0
    // carries signal (Q[0]=1, K[0]=+/-gain, V[0]=slot/0), the rest is zero-padded.
    const int R = hparams.router_layer;
    GGML_ASSERT(R >= 0);
    auto router_lane = [&](ggml_tensor * sig1d) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, sig1d, 1, 1, n_tokens);
        return ggml_pad(ctx0, t, (int) n_embd_head - 1, 0, 0, 0);
    };
    ggml_tensor * Qr = router_lane(router_q);
    ggml_tensor * Kr = router_lane(router_ksig);
    ggml_tensor * Vr = router_lane(router_vval);

    ggml_tensor * router_out = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qr, Kr, Vr, nullptr, nullptr, nullptr, /*kq_scale=*/1.0f, /*il=*/R);
    cb(router_out, "router_out", R);

    // row 0 of router_out is the attended slot; clamp+round to an I32 index
    ggml_tensor * slot_f = ggml_cont(ctx0,
        ggml_view_2d(ctx0, router_out, 1, n_tokens, router_out->nb[1], 0));
    slot_f = ggml_reshape_1d(ctx0, slot_f, n_tokens);
    slot_f = ggml_clamp(ctx0, slot_f, 0.0f, (float) smodel.n_adapters);
    slot_f = ggml_round(ctx0, slot_f);
    ggml_tensor * adapter_ids = ggml_cast(ctx0, slot_f, GGML_TYPE_I32);
    cb(adapter_ids, "adapter_ids", -1);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * cur;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention_layer(cur, inp_pos, adapter_ids, inp_attn, model, n_embd_head, il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            // keep adapter_ids aligned to the kept rows (2D round-trip for get_rows)
            const int64_t n_out = inp_out_ids->ne[0];
            adapter_ids = ggml_get_rows(ctx0,
                ggml_reshape_2d(ctx0, adapter_ids, 1, adapter_ids->ne[0]), inp_out_ids);
            adapter_ids = ggml_reshape_1d(ctx0, adapter_ids, n_out);
        }

        cur = build_layer_ffn(cur, inpSA, adapter_ids, model, il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_granite_switch::graph::build_attention_layer(
          ggml_tensor             * cur,
          ggml_tensor             * inp_pos,
          ggml_tensor             * adapter_ids,
          llm_graph_input_attn_kv * inp_attn,
    const llama_model             & model,
    const int64_t                 n_embd_head,
    const int                     il) {

    const auto & layer = model.layers[il];
    const auto & sl    = layer.switch_lora;

    const int64_t n_head    = hparams.n_head(il);
    const int64_t n_head_kv = hparams.n_head_kv(il);

    ggml_tensor * qkv = ggml_mul_mat(ctx0, layer.wqkv, cur);
    cb(qkv, "wqkv", il);

    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;

    // slice fused qkv into Q/K/V, made contiguous so LoRA deltas can be added
    ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_embd_q,  qkv->ne[1], qkv->nb[1], 0));
    ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_embd_kv, qkv->ne[1], qkv->nb[1], n_embd_q*ggml_element_size(qkv)));
    ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_embd_kv, qkv->ne[1], qkv->nb[1], (n_embd_q + n_embd_kv)*ggml_element_size(qkv)));

    Qcur = ggml_add(ctx0, Qcur, build_switched_lora_delta(sl.a_q, sl.b_q, cur, adapter_ids));
    Kcur = ggml_add(ctx0, Kcur, build_switched_lora_delta(sl.a_k, sl.b_k, cur, adapter_ids));
    Vcur = ggml_add(ctx0, Vcur, build_switched_lora_delta(sl.a_v, sl.b_v, cur, adapter_ids));

    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    if (hparams.rope_finetuned) {
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    }
    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // wo = nullptr so build_attn returns concatenated heads; o-proj is switched below
    ggml_tensor * attn = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(attn, "attn_pre_o", il);

    cur = build_switched_lora_mm(layer.wo, sl.a_o, sl.b_o, attn, adapter_ids);
    cb(cur, "attn_out", il);
    return cur;
}

ggml_tensor * llama_model_granite_switch::graph::build_layer_ffn(
          ggml_tensor       * cur,
          ggml_tensor       * inpSA,
          ggml_tensor       * adapter_ids,
    const llama_model       & model,
    const int                 il) {

    const auto & layer = model.layers[il];
    const auto & sl    = layer.switch_lora;

    if (hparams.f_residual_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
    }
    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "ffn_norm", il);

    ggml_tensor * g = build_switched_lora_mm(layer.ffn_gate, sl.a_gate, sl.b_gate, cur, adapter_ids);
    ggml_tensor * u = build_switched_lora_mm(layer.ffn_up,   sl.a_up,   sl.b_up,   cur, adapter_ids);
    g = ggml_silu(ctx0, g);
    ggml_tensor * gu = ggml_mul(ctx0, g, u);
    cur = build_switched_lora_mm(layer.ffn_down, sl.a_down, sl.b_down, gu, adapter_ids);
    cb(cur, "ffn_out", il);

    if (hparams.f_residual_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
    }
    cur = ggml_add(ctx0, cur, ffn_inp);

    cur = build_cvec(cur, il);
    cb(cur, "l_out", il);

    return cur;
}
