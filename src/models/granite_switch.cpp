#include "models.h"

#include <cmath>

// ============================================================================
// Granite Switch: a dense, all-attention Granite-4.1 model with N embedded LoRA
// adapters selected per-token by control tokens.
//
// The per-token adapter index is recovered IN-GRAPH by a single-head causal
// "router" attention (ported faithfully from the vLLM/HF backends), whose K/V
// live in the model KV cache at an extra layer R == hparams.router_layer:
//   - set_input fills pure per-token signals (no history): router Q[0]=1,
//     K[0]=±gain (+ for a control token, - otherwise), V[0]=adapter slot / 0,
//     plus the token-exchange (control id -> substitute id before embedding).
//   - the causal softmax over the single visible control token (single-switch
//     contract) puts ~all weight on it, so attended V[0] ≈ that adapter's slot;
//     readback = clamp(round(V[0]), 0, n_adapters) -> I32 adapter index.
// Because the selection state lives in the per-sequence KV cache (like vLLM's
// PagedAttention router), CONCURRENT requests are isolated for free — there is
// no global sticky variable to leak between sequences (the prior POC's bug).
//
// SINGLE-SWITCH CONTRACT / KNOWN LIMITATION (identical to the vLLM & HF
// backends): the gain is flat (no recency), so within one sequence there is "no
// mechanism to transition back to base mid-sequence" (vLLM single.py docstring).
// A control token's K stays causally visible to all later tokens in the SAME
// sequence, so once an adapter fires it stays on until that sequence ends.
// vLLM/HF never observe this because each served request is a fresh sequence; a
// client that continues ONE KV cache across chat turns (e.g. `ollama run`) must
// start a fresh sequence per turn to return to base. A recency-biased router
// would lift this limit but is a deliberate divergence from vLLM and is not done
// here. See scratch/multiturn_leak_test.cpp and scratch/concurrent_switch_test.cpp.
//
// Then a standard Granite decoder, with LoRA added per-token on qkv / o / gate /
// up / down via ggml_mul_mat_id over stacked tensors. The stacked dim is
// N = num_adapters + 1; slot 0 is zero-filled so base tokens add an exact-zero
// delta. B is pre-scaled by alpha/rank at compose time, so the runtime LoRA
// scale is 1.0.
// ============================================================================

void llama_model_granite_switch::load_arch_hparams(llama_model_loader & ml) {
    // Granite multipliers / eps / rope (mirrors llama_model_granite).
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);
    ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale, false);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE,             hparams.f_attention_scale, false);

    // Granite uses rope_finetuned as a switch for rope, so default to true.
    bool rope_finetuned = true;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    // Shared FFN length (dense path uses n_ff; this is informational/parity).
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, /* required */ false);

    // The per-token LoRA tensors use GGML_OP_MUL_MAT_ID with n_expert_used = 1.
    // The model is dense (n_expert == 0, so the generic loader checks require
    // n_expert_used == 0), but the buffer-type probe for MUL_MAT_ID tensors in
    // create_tensor builds a synthetic op sized by hparams.n_expert_used and
    // asserts it is > 0. Set it to 1 here (after the generic checks have run) so
    // the probe matches how we actually invoke mul_mat_id.
    hparams.n_expert_used = 1;

    // --- switch metadata ---
    ml.get_key(LLM_KV_NUM_ADAPTERS,  n_adapters);
    ml.get_key(LLM_KV_MAX_LORA_RANK, max_lora_rank);
    n_slots = n_adapters + 1;

    std::vector<int32_t> token_ids;
    std::vector<int32_t> substitute_ids;
    ml.get_arr(LLM_KV_ADAPTER_TOKEN_IDS,            token_ids);
    ml.get_arr(LLM_KV_ADAPTER_SUBSTITUTE_TOKEN_IDS, substitute_ids);

    GGML_ASSERT(token_ids.size() == n_adapters &&
                "granite-switch: adapter_token_ids length must equal num_adapters");
    GGML_ASSERT(substitute_ids.size() == n_adapters &&
                "granite-switch: adapter_substitute_token_ids length must equal num_adapters");

    control_token_to_index.clear();
    control_token_to_substitute.clear();
    for (uint32_t i = 0; i < n_adapters; ++i) {
        // adapter i -> stacked slot i+1 (slot 0 is the base/zero delta).
        control_token_to_index[(llama_token) token_ids[i]]      = (int32_t) (i + 1);
        control_token_to_substitute[(llama_token) token_ids[i]] = (llama_token) substitute_ids[i];
    }

    // --- router KV layer (per-token adapter selection lives in the KV cache) ---
    // The GGUF block_count covers only the real decoder layers, so at this point
    // hparams.n_layer_all == n_real and hparams.n_layer() == n_real. Append ONE
    // extra single-head attention layer at index R = n_real to hold the router
    // K/V. We bump n_layer_all to n_real+1 so the KV-cache allocator (which loops
    // [0, n_layer_all)) gives the router its own per-sequence slot, and set
    // n_layer_nextn = 1 so n_layer() == n_layer_all - n_layer_nextn stays n_real
    // — the decoder loop and tensor loading (both bounded by n_layer()) are left
    // completely unchanged and never touch layer R.
    const uint32_t n_real = hparams.n_layer(); // == n_layer_all here (nextn==0)
    hparams.router_layer  = (int32_t) n_real;
    hparams.n_layer_all   = n_real + 1;
    hparams.n_layer_nextn = 1;

    // The router is a single causal head. n_embd_head_k/v are global (the router
    // inherits the model head dim and uses only dim 0; the rest is zero-padded).
    hparams.n_head_arr[n_real]    = 1;
    hparams.n_head_kv_arr[n_real] = 1;
    hparams.n_ff_arr[n_real]      = 0; // no FFN for the router layer
}

void llama_model_granite_switch::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_slots_i64 = (int64_t) n_slots;
    const int64_t n_rank       = (int64_t) max_lora_rank;
    const int64_t n_embd_q     = n_embd_head_k * n_head; // fused Q out dim
    const int64_t n_embd_kv    = n_embd_k_gqa;           // K (== V) out dim per slice

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        // tied to the input embeddings
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // fused qkv base + separate o
        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd_q + 2*n_embd_kv}, 0);
        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        // dense (non-MoE) FFN
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        // ---- stacked per-token LoRA tensors (no .weight suffix) ----
        // A: {n_in, max_rank, N}   B: {max_rank, n_out, N}
        auto & sl = layer.switch_lora;

        sl.a_q = create_tensor(tn(LLM_TENSOR_ATTN_QKV_LORA_A_Q, i), {n_embd,  n_rank, n_slots_i64}, 0);
        sl.b_q = create_tensor(tn(LLM_TENSOR_ATTN_QKV_LORA_B_Q, i), {n_rank, n_embd_q, n_slots_i64}, 0);
        sl.a_k = create_tensor(tn(LLM_TENSOR_ATTN_QKV_LORA_A_K, i), {n_embd,  n_rank, n_slots_i64}, 0);
        sl.b_k = create_tensor(tn(LLM_TENSOR_ATTN_QKV_LORA_B_K, i), {n_rank, n_embd_kv, n_slots_i64}, 0);
        sl.a_v = create_tensor(tn(LLM_TENSOR_ATTN_QKV_LORA_A_V, i), {n_embd,  n_rank, n_slots_i64}, 0);
        sl.b_v = create_tensor(tn(LLM_TENSOR_ATTN_QKV_LORA_B_V, i), {n_rank, n_embd_kv, n_slots_i64}, 0);

        sl.a_o = create_tensor(tn(LLM_TENSOR_ATTN_OUT_LORA_A, i), {n_embd_q, n_rank, n_slots_i64}, 0);
        sl.b_o = create_tensor(tn(LLM_TENSOR_ATTN_OUT_LORA_B, i), {n_rank,   n_embd, n_slots_i64}, 0);

        sl.a_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE_LORA_A, i), {n_embd, n_rank, n_slots_i64}, 0);
        sl.b_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE_LORA_B, i), {n_rank,  n_ff,  n_slots_i64}, 0);
        sl.a_up   = create_tensor(tn(LLM_TENSOR_FFN_UP_LORA_A,   i), {n_embd, n_rank, n_slots_i64}, 0);
        sl.b_up   = create_tensor(tn(LLM_TENSOR_FFN_UP_LORA_B,   i), {n_rank,  n_ff,  n_slots_i64}, 0);
        sl.a_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN_LORA_A, i), {  n_ff, n_rank, n_slots_i64}, 0);
        sl.b_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN_LORA_B, i), {n_rank, n_embd, n_slots_i64}, 0);
    }
}

// Router differential gain. Each token's router K dim-0 is +GAIN for a control
// token and -GAIN otherwise; Q dim-0 is 1.0. In the causal softmax a single
// visible control token then dominates (its logit beats a normal token's by
// 2*GAIN), so the readback recovers that adapter's integer slot. 15.0 matches the
// vLLM/HF backend default (control_token_gain in config.py) and is F16-safe
// (no F32 KV cache needed).
static constexpr float GRANITE_SWITCH_ROUTER_GAIN = 15.0f;

// ----------------------------------------------------------------------------
// switch input: pure per-token maps. The adapter index is NOT computed here —
// it is recovered in-graph by the router attention (see build_arch_graph). This
// is intentionally stateless: no cross-ubatch carry. Per-sequence isolation
// (no leak between concurrent sequences) comes from the KV cache, not from here.
// ----------------------------------------------------------------------------
void llm_graph_input_switch::set_input(const llama_ubatch * ubatch) {
    if (!ubatch->token) {
        return; // embedding input path is not used by this arch
    }

    const int64_t n_tokens = ubatch->n_tokens;

    std::vector<int32_t> sub (n_tokens);
    std::vector<float>   ksig(n_tokens);
    std::vector<float>   vval(n_tokens);
    std::vector<float>   q   (n_tokens, 1.0f); // Q dim-0 is constant 1.0

    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_token tok = ubatch->token[i];

        // router K/V signals: control token -> (+gain, slot); else -> (-gain, 0).
        auto it = smodel.control_token_to_index.find(tok);
        if (it != smodel.control_token_to_index.end()) {
            ksig[i] = +GRANITE_SWITCH_ROUTER_GAIN;
            vval[i] = (float) it->second; // adapter slot (1 + adapter)
        } else {
            ksig[i] = -GRANITE_SWITCH_ROUTER_GAIN;
            vval[i] = 0.0f;               // base
        }

        // token-exchange: rewrite a control token to its substitute id.
        auto sit = smodel.control_token_to_substitute.find(tok);
        sub[i] = (sit != smodel.control_token_to_substitute.end())
            ? (int32_t) sit->second
            : (int32_t) tok;
    }

    ggml_backend_tensor_set(sub_tokens,  sub.data(),  0, n_tokens*ggml_element_size(sub_tokens));
    ggml_backend_tensor_set(router_ksig, ksig.data(), 0, n_tokens*ggml_element_size(router_ksig));
    ggml_backend_tensor_set(router_vval, vval.data(), 0, n_tokens*ggml_element_size(router_vval));
    ggml_backend_tensor_set(router_q,    q.data(),    0, n_tokens*ggml_element_size(router_q));
}

// ----------------------------------------------------------------------------
// graph
// ----------------------------------------------------------------------------
std::unique_ptr<llm_graph_context> llama_model_granite_switch::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// per-token switched LoRA delta only: B_a·(A_a·x), selected per token.
// cur: {n_in, n_tokens}, ids: {n_tokens} -> returns {n_out, n_tokens}.
ggml_tensor * llama_model_granite_switch::graph::build_switched_lora_delta(
          ggml_tensor * lora_a,
          ggml_tensor * lora_b,
          ggml_tensor * cur,
          ggml_tensor * ids) {
    const int64_t n_in     = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];

    // mul_mat_id wants the activation as {n_in, n_expert_used=1, n_tokens}
    // and ids as {n_expert_used=1, n_tokens}.
    ggml_tensor * x   = ggml_reshape_3d(ctx0, cur, n_in, 1, n_tokens);
    ggml_tensor * ids2 = ggml_reshape_2d(ctx0, ids, 1, n_tokens);

    ggml_tensor * a = ggml_mul_mat_id(ctx0, lora_a, x, ids2); // {max_rank, 1, n_tokens}
    ggml_tensor * d = ggml_mul_mat_id(ctx0, lora_b, a, ids2); // {n_out,    1, n_tokens}

    return ggml_reshape_2d(ctx0, d, d->ne[0], n_tokens);      // {n_out, n_tokens}
}

// per-token switched matmul: base (plain 2D weight) + selected LoRA delta.
ggml_tensor * llama_model_granite_switch::graph::build_switched_lora_mm(
          ggml_tensor * w,
          ggml_tensor * lora_a,
          ggml_tensor * lora_b,
          ggml_tensor * cur,   // {n_in, n_tokens}
          ggml_tensor * ids) { // {n_tokens}
    ggml_tensor * base  = ggml_mul_mat(ctx0, w, cur); // {n_out, n_tokens}
    ggml_tensor * delta = build_switched_lora_delta(lora_a, lora_b, cur, ids);
    return ggml_add(ctx0, base, delta);
}

llama_model_granite_switch::graph::graph(
    const llama_model & model,
    const llm_graph_params & params)
    : llm_graph_context(params) {

    const auto & smodel = static_cast<const llama_model_granite_switch &>(model);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    // --- switch input: substituted ids + per-token router K/V/Q signals ---
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

    // embed the token-exchanged ids ourselves (we cannot use build_inp_embd
    // because it embeds the raw ubatch tokens). Apply Granite embedding scale.
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

    // --- router attention: recover the per-token adapter index in-graph ---
    // A single causal head whose K/V live in the KV cache at layer R == router_layer.
    // Only dim 0 carries signal (rest zero-padded): Q[0]=1, K[0]=±gain, V[0]=slot/0.
    // The causal softmax over a single visible control token (the single-switch
    // contract) puts ~all weight on it, so attended V[0] ≈ that adapter's slot.
    // State persists across decode steps and is isolated per-sequence because the
    // control token's K/V stay in the per-sequence KV cache (vLLM/HF parity). No
    // RoPE on the router Q/K — positions are irrelevant to the selection.
    const int   R = hparams.router_layer;
    GGML_ASSERT(R >= 0 && "granite-switch: router_layer not configured");
    // build [n_embd_head, 1, n_tokens] with row 0 = signal, rows 1.. = 0.
    auto router_lane = [&](ggml_tensor * sig1d) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, sig1d, 1, 1, n_tokens);
        return ggml_pad(ctx0, t, (int) n_embd_head - 1, 0, 0, 0); // pad dim0 right with zeros
    };
    ggml_tensor * Qr = router_lane(router_q);
    ggml_tensor * Kr = router_lane(router_ksig);
    ggml_tensor * Vr = router_lane(router_vval);

    ggml_tensor * router_out = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qr, Kr, Vr, nullptr, nullptr, nullptr, /*kq_scale=*/1.0f, /*il=*/R);
    cb(router_out, "router_out", R);

    // readback: router_out is {n_embd_head, n_tokens}; row 0 is the attended slot.
    // adapter_index = clamp(round(row0), 0, n_adapters), as I32 for mul_mat_id.
    ggml_tensor * slot_f = ggml_cont(ctx0,
        ggml_view_2d(ctx0, router_out, 1, n_tokens, router_out->nb[1], 0));
    slot_f = ggml_reshape_1d(ctx0, slot_f, n_tokens);
    slot_f = ggml_clamp(ctx0, slot_f, 0.0f, (float) smodel.n_adapters);
    slot_f = ggml_round(ctx0, slot_f);                   // round BEFORE cast (cast truncates)
    ggml_tensor * adapter_ids = ggml_cast(ctx0, slot_f, GGML_TYPE_I32);
    cb(adapter_ids, "adapter_ids", -1);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * cur;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // attn norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention (with per-token switched LoRA)
        cur = build_attention_layer(cur, inp_pos, adapter_ids, inp_attn, model, n_embd_head, il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            // adapter_ids is 1D {n_tokens}; ggml_get_rows on a 1D tensor would
            // gather whole rows of width n_tokens. Reshape to {1, n_tokens} so we
            // select per-token columns, then flatten back to 1D {n_out}.
            const int64_t n_out = inp_out_ids->ne[0];
            adapter_ids = ggml_get_rows(ctx0,
                ggml_reshape_2d(ctx0, adapter_ids, 1, adapter_ids->ne[0]), inp_out_ids);
            adapter_ids = ggml_reshape_1d(ctx0, adapter_ids, n_out);
        }

        // ffn (with per-token switched LoRA)
        cur = build_layer_ffn(cur, inpSA, adapter_ids, model, il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    // Granite logit scaling
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

    // fused qkv base + per-slice switched LoRA deltas.
    ggml_tensor * qkv = ggml_mul_mat(ctx0, layer.wqkv, cur);
    cb(qkv, "wqkv", il);

    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;

    // slice base qkv into Q/K/V (contiguous copies so we can add LoRA deltas).
    ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_embd_q,  qkv->ne[1], qkv->nb[1], 0));
    ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_embd_kv, qkv->ne[1], qkv->nb[1], n_embd_q*ggml_element_size(qkv)));
    ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_embd_kv, qkv->ne[1], qkv->nb[1], (n_embd_q + n_embd_kv)*ggml_element_size(qkv)));

    // add per-token switched LoRA deltas to each base slice (delta only — the
    // fused base qkv is already computed above).
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

    // wo = nullptr: build_attn returns the concatenated heads (pre-o-proj),
    // then we apply the switched o-proj manually.
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

    // Granite residual scale
    if (hparams.f_residual_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
    }
    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "ffn_norm", il);

    // gated SILU FFN with per-token switched LoRA on gate / up / down.
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
