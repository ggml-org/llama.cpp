#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"

#include <cmath>

// iHC (independent Hyper-Connections) helpers. Same layout as the DeepSeek-V4 HC, but without
// the comb/sinkhorn term: hc_fn makes only 2*hc coefficients (pre + post). The streams mix
// through the pre-reduce / post-distribute round trip instead.

static size_t hy_v4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * hy_v4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, hy_v4_elem_offset(t, i0));
}

static ggml_tensor * hy_v4_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], hy_v4_elem_offset(t, i0));
}

void llama_model_hy_v4::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm,  false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func,   false);

    // routed-expert SwiGLU logits clamp (shared/dense experts are NOT clamped, so
    // swiglu_clamp_shexp is intentionally left at its 0 default)
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, hparams.n_layer_all, false);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,     hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,   hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HYPER_CONNECTION_MAGNITUDE, hparams.hc_magnitude);

    // DSA is absent on the all-full_attention checkpoints, so indexer_top_k stays 0 there
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head,    false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k,     false);

    if (hparams.indexer_top_k > 0) {
        // the reference plumbs rms_norm_eps into the indexer k_norm LayerNorm, and build_norm
        // reads f_norm_eps for LLM_NORM
        hparams.f_norm_eps = hparams.f_norm_rms_eps;

        if (hparams.indexer_n_head == 0 || hparams.indexer_head_size <= hparams.n_rot()) {
            throw std::runtime_error("hy_v4: bad indexer head count / key length");
        }

        ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer(), false);
        if (!hparams.is_indexer_full(0)) {
            throw std::runtime_error("hy_v4: layer 0 must own an indexer, nothing precedes it to share");
        }
    }

    GGML_ASSERT(hparams.is_mla());

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_hy_v4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla   = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t kv_lora_rank    = hparams.n_lora_kv;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t hc              = hparams.dsv4_hc_mult;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // global iHC head (collapses hc streams before the final norm)
    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc * n_embd, hc}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);

        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);
        layer.wkv_a_mqa     = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);
        layer.attn_kv_a_norm= create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM,"weight", i), {kv_lora_rank}, 0);
        layer.wk_b          = create_tensor(tn(LLM_TENSOR_ATTN_K_B,      "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
        layer.wv_b          = create_tensor(tn(LLM_TENSOR_ATTN_V_B,      "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
        layer.wo            = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);
        layer.wqkv_gate     = create_tensor(tn(LLM_TENSOR_ATTN_GATE,     "weight", i), {n_embd, n_head * n_embd_head_v_mla}, 0);

        // only "full" indexer layers ship weights; "shared" layers reuse their top-k
        if (hparams.indexer_top_k > 0 && hparams.is_indexer_full(i)) {
            const int64_t n_indexer_head = hparams.indexer_n_head;
            const int64_t n_embd_indexer = hparams.indexer_head_size;

            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, n_indexer_head * n_embd_indexer}, 0);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, n_embd_indexer}, 0);
            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_indexer}, 0);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {n_embd_indexer}, 0);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_indexer_head}, 0);
        }

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc * n_embd, 2 * hc}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {2 * hc}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {2}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc * n_embd, 2 * hc}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {2 * hc}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {2}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_hy_v4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// reduce hc streams x[:,i,:] weighted by w[i,:] -> [n_embd, n_tokens]
// reference runs this in fp32 (inside the float() / autocast(fp32) context)
static ggml_tensor * hy_v4_hc_reduce(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, int64_t hc, int64_t n_embd, int64_t nt, ggml_type out_type) {
    ggml_tensor * x_f32 = ggml_cast(ctx0, x, GGML_TYPE_F32);
    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x_f32, n_embd, nt, x_f32->nb[2], ih * x_f32->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, w, 1, nt, w->nb[1], ih * w->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }
    return ggml_cast(ctx0, result, out_type);
}

ggml_tensor * llama_model_hy_v4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        int il) const {
    const int64_t hc  = hparams.dsv4_hc_mult;
    const int64_t nt  = x->ne[2];
    GGML_ASSERT(x->ne[0] == n_embd && x->ne[1] == hc);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc * n_embd, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm); // [2*hc, nt]
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = hy_v4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = hy_v4_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * base_pre   = hy_v4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post  = hy_v4_view_1d(ctx0, hc_base, hc, hc);

    // pre = sigmoid(mixes[:hc]*scale_pre + base_pre) + eps
    ggml_tensor * pre = hy_v4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = ggml_mul(ctx0, pre, scale_pre);
    pre = ggml_add(ctx0, pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    // post = magnitude*sigmoid(mixes[hc:2hc]*scale_post + base_post) + eps
    ggml_tensor * po = hy_v4_view_2d(ctx0, mixes, hc, nt, hc);
    po = ggml_mul(ctx0, po, scale_post);
    po = ggml_add(ctx0, po, base_post);
    po = ggml_sigmoid(ctx0, po);
    po = ggml_scale(ctx0, po, hparams.hc_magnitude);
    po = ggml_scale_bias(ctx0, po, 1.0f, hparams.dsv4_hc_eps);
    *post = po;
    cb(po, "hc_post_gate", il);

    return hy_v4_hc_reduce(ctx0, x, pre, hc, n_embd, nt, x->type);
}

ggml_tensor * llama_model_hy_v4::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        int il) const {
    GGML_UNUSED(il);
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hc);

    // reference HC post runs entirely in fp32 to avoid bf16 rounding accumulation
    // across 78 layers: post.float() * x.float() + residual.float() -> .to(dtype)
    ggml_tensor * x_f32    = ggml_cast(ctx0, x, GGML_TYPE_F32);
    ggml_tensor * post_f32 = ggml_cast(ctx0, post, GGML_TYPE_F32);
    ggml_tensor * res_f32  = ggml_cast(ctx0, residual, GGML_TYPE_F32);

    ggml_tensor * out = nullptr;
    for (int64_t i = 0; i < hc; ++i) {
        ggml_tensor * res_i  = ggml_view_2d(ctx0, res_f32, n_embd, nt, res_f32->nb[2], i * res_f32->nb[1]);
        ggml_tensor * post_i = ggml_view_2d(ctx0, post_f32, 1, nt, post_f32->nb[1], i * post_f32->nb[0]);
        ggml_tensor * cur = ggml_add(ctx0, res_i, ggml_mul(ctx0, x_f32, post_i));
        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    // cast back to the original type (bf16)
    out = ggml_cast(ctx0, out, residual->type);
    return out; // [n_embd, hc, nt]
}

ggml_tensor * llama_model_hy_v4::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc * n_embd, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm); // [hc, nt]
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = ggml_mul(ctx0, mixes, hc_scale);
    pre = ggml_add(ctx0, pre, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return hy_v4_hc_reduce(ctx0, x, pre, hc, n_embd, nt, x->type);
}

ggml_tensor * llama_model_hy_v4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_attn_k * inp_attn,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        float kq_scale,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head_k       = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const uint32_t kv_lora_rank       = hparams.n_lora_kv;

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
    q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_mul_mat(ctx0, layer.wq_b, q);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "q_pe", il);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_pe, "k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "kv_cmpr", il);

    // MLA absorption: q_nope @ wk_b -> compressed space
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

    // note: rope must go first for in-place context shifting in build_rope_shift()
    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    ggml_tensor * Vcur = kv_cmpr;

    // MLA-as-MQA; wo applied manually below so the gated-MLA gate can sit before o_proj
    ggml_tensor * attn = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, layer.wv_b, kq_scale, il);
    cb(attn, "attn_kqv", il); // [n_head * n_embd_head_v, n_tokens]

    // gated MLA: elementwise sigmoid gate on the decompressed attention output
    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.wqkv_gate, cur);
    gate = ggml_sigmoid(ctx0, gate);
    attn = ggml_mul(ctx0, attn, gate);
    cb(attn, "attn_gated", il);

    ggml_tensor * out = build_lora_mm(layer.wo, attn);
    cb(out, "attn_out", il);

    return out;
}

ggml_tensor * llama_model_hy_v4::graph::build_indexer_top_k(
        const llama_model & model,
        llm_graph_input_attn_k_dsa * inp_attn_dsa,
        ggml_tensor * cur,
        ggml_tensor * qr,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_indexer_head       = hparams.indexer_n_head;
    const int64_t n_embd_indexer       = hparams.indexer_head_size;
    const int64_t n_embd_indexer_rope  = hparams.n_rot();
    const int64_t n_embd_indexer_nope  = n_embd_indexer - n_embd_indexer_rope;

    // nope rows come first, so rope only the last n_embd_indexer_rope rows, same as the MLA path
    ggml_tensor * iq = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);

    iq = ggml_reshape_3d(ctx0, iq, n_embd_indexer, n_indexer_head, n_tokens);

    iq = ggml_rope_ext(ctx0, iq, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base,
         freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    iq = ggml_rope_set_offset(iq, n_embd_indexer_nope);
    cb(iq, "indexer_q", il);

    ggml_tensor * ik = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);

    ik = build_norm(ik, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);

    ik = ggml_reshape_3d(ctx0, ik, n_embd_indexer, 1, n_tokens);

    ik = ggml_rope_ext(ctx0, ik, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base,
         freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    ik = ggml_rope_set_offset(ik, n_embd_indexer_nope);
    cb(ik, "indexer_k", il);

    // the reference applies a Hadamard rotation here, but it only helps its FP8 kernels.
    // it is orthogonal, so it does not change q.k and we can skip it.

    const auto * mctx_lid   = inp_attn_dsa->mctx->get_lid();
    const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
    ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, ik, k_idxs_lid, il));

    ggml_tensor * iw = ggml_mul_mat(ctx0, layer.indexer_proj, cur);

    ik = mctx_lid->get_k(ctx0, il);

    const auto n_stream = ik->ne[3];
    iq = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], iq->ne[2]/n_stream, n_stream,
            iq->nb[1], iq->nb[2], iq->nb[3]/n_stream, 0);
    iw = ggml_view_4d(ctx0, iw, iw->ne[0], iw->ne[1]/n_stream, iw->ne[2], n_stream,
            iw->nb[1], iw->nb[2]/n_stream, iw->nb[3]/n_stream, 0);

    // fold both reference scale factors into the weights before the big score tensor
    iw = ggml_scale(ctx0, iw, 1.0f / sqrtf(float(n_embd_indexer * n_indexer_head)));

    ggml_tensor * score = nullptr;
    if (cparams.fused_lid) {
        score = ggml_lightning_indexer(ctx0, iq, ik, iw, inp_attn_dsa->get_kq_mask_lid());
        cb(score, "indexer_score", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    } else {
        iq = ggml_permute(ctx0, iq, 0, 2, 1, 3);
        ik = ggml_permute(ctx0, ik, 0, 2, 1, 3);

        score = ggml_mul_mat(ctx0, ik, iq);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));
        score = ggml_relu(ctx0, score);
        score = ggml_mul(ctx0, score, iw);
        score = ggml_sum_rows(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));
        score = ggml_add(ctx0, score, inp_attn_dsa->get_kq_mask_lid());
        cb(score, "indexer_score", il);
    }

    const uint32_t n_top_k = score->ne[0] < (int64_t) hparams.indexer_top_k ? score->ne[0] : hparams.indexer_top_k;

    return ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_k));
}

ggml_tensor * llama_model_hy_v4::graph::build_attention_dsa(
        const llama_model & model,
        llm_graph_input_attn_k_dsa * inp_attn_dsa,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor ** last_top_k,
        float kq_scale,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head_k       = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const uint32_t kv_lora_rank       = hparams.n_lora_kv;

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);

    if (hparams.is_indexer_full(il)) {
        *last_top_k = build_indexer_top_k(model, inp_attn_dsa, cur, qr, inp_pos, il);
        cb(*last_top_k, "top_k", il);
    }
    GGML_ASSERT(*last_top_k != nullptr);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "q_pe", il);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_pe, "k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "kv_cmpr", il);

    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    ggml_tensor * Vcur = kv_cmpr;

    ggml_tensor * attn = build_attn(inp_attn_dsa,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, layer.wv_b, *last_top_k, kq_scale, il);
    cb(attn, "attn_kqv", il);

    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.wqkv_gate, cur);
    gate = ggml_sigmoid(ctx0, gate);
    attn = ggml_mul(ctx0, attn, gate);
    cb(attn, "attn_gated", il);

    ggml_tensor * out = build_lora_mm(layer.wo, attn);
    cb(out, "attn_out", il);

    return out;
}

llama_model_hy_v4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;

    const bool is_dsa = hparams.indexer_top_k > 0;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    llm_graph_input_attn_k     * inp_attn     = is_dsa ? nullptr : build_attn_inp_k();
    llm_graph_input_attn_k_dsa * inp_attn_dsa = is_dsa ? build_attn_inp_k_dsa() : nullptr;
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // top-k of the last "full" indexer layer, reused by the following "shared" layers
    ggml_tensor * last_top_k = nullptr;

    // expand the single embedding into hc parallel residual streams
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;

        cur = build_hc_pre(inpL, model.layers[il].hc_attn_fn, model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base, &post, il);
        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = is_dsa
            ? build_attention_dsa(model, inp_attn_dsa, cur, inp_pos, &last_top_k, kq_scale, il)
            : build_attention(model, inp_attn, cur, inp_pos, kq_scale, il);

        inpL = build_hc_post(cur, residual, post, il);
        cb(inpL, "hc_attn_out", il);

        residual = inpL;
        cur = build_hc_pre(inpL, model.layers[il].hc_ffn_fn, model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base, &post, il);
        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
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
                    nullptr);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   NULL, NULL,
                    layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        inpL = build_hc_post(cur, residual, post, il);
        cb(inpL, "l_out", il);
    }

    // prune to the requested output rows once, after all HC streams are done
    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
