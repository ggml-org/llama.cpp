#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// ERNIE 5.5 (Ernie5_5_MoeForCausalLM): a DeepSeek-V4 hybrid.
//
//  - HCA layers (compress ratio 128 + 128-token window) are the DSV4 layers, with a sigmoid gate on the grouped
//    low-rank output and a VHA head mix (I + V U^T, folded at conversion); they are the SWA layers of the raw cache.
//  - MLA layers (ratio 0 here, -2 in the checkpoint) attend to the window plus the top-k of a lightning indexer.
//    They keep the absorbed latent [kv | rope] and the indexer key in one row of the base (full context) half of
//    the raw cache. Up to top_k + window cells the indexer would select them all, so attention is dense there.
//  - mHC with 4 streams; the per-layer pre-norm factor is 1 / (sqrt(mean(x^2)) + eps), the head keeps the V4 one.
//  - one dense situ FFN layer, then latent MoE: the experts run in a smaller space between a down projection and a
//    norm + up projection, routed by two sigmoid routers summed, renormalized and scaled per expert.
//  - logits pass through a learned MultiMax.
//  - the MTP block is one more MLA layer on [enorm(e) | hnorm(h)] with its own mHC head; it attends densely.

void llama_model_ernie5_5::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);

    uint32_t n_embd_head_k_mla = 0;
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   n_embd_head_k_mla);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.ernie_mla_v);
    GGML_ASSERT(n_embd_head_k_mla > hparams.n_rot());
    hparams.ernie_mla_qk_nope = n_embd_head_k_mla - hparams.n_rot();

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_LATENT_LENGTH,        hparams.n_expert_latent);
    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult == 4 && "mHC with hc_mult != 4 is not supported");

    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer_all) {
        throw std::runtime_error("ERNIE 5.5 compress_ratios is shorter than block_count");
    }
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);

    ml.get_arr(LLM_KV_LOGIT_MULTIMAX, hparams.ernie_multimax);

    // the MTP hidden state is the 4-stream residual before the mHC head
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    // MLA cache rows are [kv latent | rope | indexer key]: attention reads the first two, the indexer the last
    hparams.n_embd_head_k_full = hparams.n_lora_kv + hparams.n_rot() + hparams.indexer_head_size;

    // the indexer's k_norm is a LayerNorm
    hparams.f_norm_eps = 1e-6f;

    // HCA layers are the SWA layers of the DSV4 raw cache, MLA layers live in its full half
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        const uint32_t ratio = hparams.dsv4_compress_ratios[il];
        if (ratio != 0 && ratio != 128) {
            throw std::runtime_error("ERNIE 5.5 loader only supports compression ratios 0 (MLA) and 128 (HCA)");
        }
        hparams.is_swa_impl[il] = ratio != 0;
    }

    switch (hparams.n_layer()) {
        case 43: type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_ernie5_5::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t n_ff_exp     = hparams.n_ff_exp();
    const int64_t n_embd_lat   = hparams.n_expert_latent;
    const int64_t n_shexp      = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k(0); // HCA head, K = V
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    const int64_t qk_nope = hparams.ernie_mla_qk_nope;
    const int64_t v_mla   = hparams.ernie_mla_v;

    const int64_t n_idx_head = hparams.indexer_n_head;
    const int64_t n_embd_idx = hparams.indexer_head_size;

    const auto nf = nextn_flags(ml);

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc_mult}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc_mult}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const int flags = i < n_layer ? nf.trunk : nf.mtp;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);

        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, flags);
        layer.attn_vha      = create_tensor(tn(LLM_TENSOR_ATTN_VHA,      "weight", i), {n_head, n_head}, flags);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);

        if (hparams.dsv4_compress_ratios[i] != 0) {
            // HCA
            const int64_t ratio = hparams.dsv4_compress_ratios[i];

            layer.wq_b         = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,     "weight", i), {q_lora_rank, n_head * n_embd_head}, flags);
            layer.wkv          = create_tensor(tn(LLM_TENSOR_ATTN_KV,      "weight", i), {n_embd, n_embd_head}, flags);
            layer.attn_kv_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM, "weight", i), {n_embd_head}, flags);
            layer.wo_a         = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,   "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, flags | TENSOR_ALLOW_RESHAPE);
            layer.wo_b         = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,   "weight", i), {o_groups * o_lora_rank, n_embd}, flags);
            layer.wqkv_gate    = create_tensor(tn(LLM_TENSOR_ATTN_GATE,    "weight", i), {q_lora_rank, o_groups * o_lora_rank}, flags);

            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, n_embd_head}, flags);
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, flags);
            layer.attn_comp_ape   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_APE,   "weight", i), {n_embd_head, ratio}, flags);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, flags);
        } else {
            // MLA with the DSA indexer
            layer.wq_b           = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,       "weight", i), {q_lora_rank, n_head * (qk_nope + n_rot)}, flags);
            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), {n_embd, kv_lora_rank + n_rot}, flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);
            layer.wk_b           = create_tensor(tn(LLM_TENSOR_ATTN_K_B,       "weight", i), {qk_nope, kv_lora_rank, n_head}, flags);
            layer.wv_b           = create_tensor(tn(LLM_TENSOR_ATTN_V_B,       "weight", i), {kv_lora_rank, v_mla, n_head}, flags);
            layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_head * v_mla, n_embd}, flags);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", i), {q_lora_rank, n_head * v_mla}, flags);

            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, n_embd_idx}, flags);
            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_idx}, flags);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {n_embd_idx}, flags);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, n_idx_head * n_embd_idx}, flags);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_idx_head}, flags);
        }

        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, flags);
            layer.nextn.hc_head_fn       = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_FN,       "weight", i), {hc_dim, hc_mult}, flags);
            layer.nextn.hc_head_base     = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_BASE,     "weight", i), {hc_mult}, flags);
            layer.nextn.hc_head_scale    = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_SCALE,    "weight", i), {1}, flags);
        }

        if ((uint32_t) i < hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, flags);
            continue;
        }

        // two routers stacked into [n_embd, 2*n_expert]
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, 2 * n_expert}, flags);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, flags);
        layer.ffn_exp_scale   = create_tensor(tn(LLM_TENSOR_FFN_EXP_SCALE,   "weight", i), {n_expert}, flags);

        layer.ffn_routed_down = create_tensor(tn(LLM_TENSOR_FFN_ROUTED_DOWN, "weight", i), {n_embd, n_embd_lat}, flags);
        layer.ffn_routed_up   = create_tensor(tn(LLM_TENSOR_FFN_ROUTED_UP,   "weight", i), {n_embd_lat, n_embd}, flags);
        layer.ffn_routed_norm = create_tensor(tn(LLM_TENSOR_FFN_ROUTED_NORM, "weight", i), {n_embd_lat}, flags);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd_lat, n_ff_exp, n_expert}, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd_lat, n_expert}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd_lat, n_ff_exp, n_expert}, flags);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_shexp}, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_shexp, n_embd}, flags);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_shexp}, flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_ernie5_5::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

llama_model_deepseek4::graph::hc_pre llama_model_ernie5_5::graph::build_hc_pre_ernie(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        int il) const {
    const int64_t hc_dim = hparams.dsv4_hc_mult*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);

    // 1 / (sqrt(mean(x^2)) + eps), not rsqrt(mean(x^2) + eps)
    ggml_tensor * rms = ggml_sum_rows(ctx0, ggml_sqr(ctx0, flat));
    rms = ggml_sqrt(ctx0, ggml_scale(ctx0, rms, 1.0f/hc_dim));
    rms = ggml_scale_bias(ctx0, rms, 1.0f, norm_rms_eps);

    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat);
    mixes = ggml_div(ctx0, mixes, rms);
    cb(mixes, "hc_mixes", il);

    const hc_mixes m = build_hc_split_mixes(mixes, hc_scale, hc_base, il);

    return { build_hc_pre(x, m.pre, il), m.post, m.comb };
}

llama_model_ernie5_5::graph::mla_qk llama_model_ernie5_5::graph::build_mla_qk(
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_rope  = hparams.n_rot();
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t qk_nope      = hparams.ernie_mla_qk_nope;
    const int64_t qk_head      = qk_nope + n_embd_rope;
    const int64_t n_embd_idx   = hparams.indexer_head_size;
    const int64_t nt           = cur->ne[1];

    ggml_tensor * q_c = build_lora_mm(layer.wq_a, cur);
    q_c = build_norm(q_c, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(q_c, "q_c", il);

    ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, q_c));
    cb(gate, "attn_gate", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, q_c);
    q = ggml_reshape_3d(ctx0, q, qk_head, n_head, nt);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, qk_nope, n_head, nt, q->nb[1], q->nb[2], 0);
    ggml_tensor * q_pe   = ggml_view_3d(ctx0, q, n_embd_rope, n_head, nt, q->nb[1], q->nb[2],
            ggml_row_size(q->type, qk_nope));

    ggml_tensor * kv_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_pe, "kv_cmpr_pe", il);

    ggml_tensor * kv_c = ggml_view_2d(ctx0, kv_pe, kv_lora_rank, nt, kv_pe->nb[1], 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_pe, n_embd_rope, 1, nt, kv_pe->nb[1], kv_pe->nb[1],
            ggml_row_size(kv_pe->type, kv_lora_rank));

    kv_c = build_norm(kv_c, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_c, "kv_cmpr", il);

    // plain interleaved RoPE at rope_theta, no YaRN (that is HCA only)
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_rope, rope_type, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_embd_rope, rope_type, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    cb(q_pe, "q_pe", il);
    cb(k_pe, "k_pe", il);

    // absorb k_b into the query: [qk_nope, n_head, nt] -> [kv_lora_rank, n_head, nt]
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    ggml_tensor * q_abs = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_abs = ggml_permute(ctx0, q_abs, 0, 2, 1, 3);
    cb(q_abs, "q_nope_absorbed", il);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_abs, q_pe, 0);
    cb(Qcur, "Qcur", il);

    // the indexer key: LayerNorm, then NEOX RoPE on its first n_rot dims. The reference also rotates q and k by a
    // Hadamard matrix before its fp8 cast; that leaves q.k unchanged, so it is left out here.
    ggml_tensor * idx_k = build_lora_mm(layer.indexer_attn_k, cur);
    idx_k = build_norm(idx_k, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
    idx_k = ggml_reshape_3d(ctx0, idx_k, n_embd_idx, 1, nt);
    idx_k = ggml_rope_ext(ctx0, idx_k, inp_pos, nullptr, n_embd_rope, LLAMA_ROPE_TYPE_NEOX, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    cb(idx_k, "indexer_k", il);

    kv_c = ggml_reshape_3d(ctx0, kv_c, kv_lora_rank, 1, nt);
    ggml_tensor * Kcur = ggml_concat(ctx0, ggml_concat(ctx0, kv_c, k_pe, 0), idx_k, 0);
    cb(Kcur, "Kcur", il);

    return { Qcur, Kcur, q_c, gate };
}

ggml_tensor * llama_model_ernie5_5::graph::build_mla_top_k(
        ggml_tensor * cur,
        ggml_tensor * q_c,
        ggml_tensor * k,
        ggml_tensor * idx_mask,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_rope = hparams.n_rot();
    const int64_t n_embd_idx  = hparams.indexer_head_size;
    const int64_t n_idx_head  = hparams.indexer_n_head;
    const int64_t nt          = cur->ne[1];
    const int64_t n_kv        = k->ne[2];
    const int64_t n_stream    = k->ne[3];

    GGML_ASSERT(nt % n_stream == 0);
    const int64_t n_tps = nt/n_stream;

    // the indexer keys, the tail of each cache row
    ggml_tensor * idx_k = ggml_view_4d(ctx0, k, n_embd_idx, 1, n_kv, n_stream, k->nb[1], k->nb[2], k->nb[3],
            ggml_row_size(k->type, hparams.n_lora_kv + n_embd_rope));
    cb(idx_k, "indexer_k_cache", il);

    ggml_tensor * idx_q = build_lora_mm(layer.indexer_attn_q_b, q_c);
    idx_q = ggml_reshape_3d(ctx0, idx_q, n_embd_idx, n_idx_head, nt);
    idx_q = ggml_rope_ext(ctx0, idx_q, inp_pos, nullptr, n_embd_rope, LLAMA_ROPE_TYPE_NEOX, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    cb(idx_q, "indexer_q", il);

    ggml_tensor * idx_w = build_lora_mm(layer.indexer_proj, cur);
    idx_w = ggml_scale(ctx0, idx_w, 1.0f/sqrtf(float(n_embd_idx*n_idx_head)));
    cb(idx_w, "indexer_weights", il);

    // split the batch into streams: q [n_embd_idx, n_head, n_tps, n_stream], w [n_head, n_tps, 1, n_stream]
    idx_q = ggml_view_4d(ctx0, idx_q, n_embd_idx, n_idx_head, n_tps, n_stream,
            idx_q->nb[1], idx_q->nb[2], idx_q->nb[2]*n_tps, 0);
    idx_w = ggml_view_4d(ctx0, idx_w, n_idx_head, n_tps, 1, n_stream,
            idx_w->nb[1], idx_w->nb[1]*n_tps, idx_w->nb[1]*n_tps, 0);

    ggml_tensor * score = nullptr;
    if (cparams.fused_lid) {
        score = ggml_lightning_indexer(ctx0, idx_q, idx_k, idx_w, idx_mask);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    } else {
        // sum over heads of relu(q_h . k) * w_h
        ggml_tensor * kq = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, idx_k, 0, 2, 1, 3)),
                ggml_permute(ctx0, idx_q, 0, 2, 1, 3));                     // [n_kv, n_tps, n_head, n_stream]
        kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));          // [n_head, n_tps, n_kv, n_stream]
        score = ggml_mul(ctx0, ggml_relu(ctx0, kq), idx_w);
        score = ggml_sum_rows(ctx0, score);                                // [1, n_tps, n_kv, n_stream]
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));    // [n_kv, n_tps, 1, n_stream]
        score = ggml_add(ctx0, score, idx_mask);
    }
    cb(score, "indexer_score", il);

    // the window cells score +inf, so top_k + window always holds them
    const int64_t n_top = std::min<int64_t>(n_kv, hparams.indexer_top_k + hparams.n_swa);
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top));
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

ggml_tensor * llama_model_ernie5_5::graph::build_mla_out(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * kq_mask,
        ggml_tensor * gate,
        int64_t n_kv_max,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t n_embd_rope  = hparams.n_rot();
    const int64_t v_mla        = hparams.ernie_mla_v;
    const int64_t qk_head      = hparams.ernie_mla_qk_nope + n_embd_rope;
    const int64_t nt           = q->ne[2];

    GGML_ASSERT(!ggml_is_quantized(k->type) && "ERNIE 5.5 MLA layers need an unquantized K cache");

    // K is [kv latent | rope] of each row, V its latent
    ggml_tensor * k_att = ggml_view_4d(ctx0, k, kv_lora_rank + n_embd_rope, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);
    ggml_tensor * v     = ggml_view_4d(ctx0, k, kv_lora_rank,               k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * out = build_attn_mha(q, k_att, v, nullptr, kq_mask, nullptr,
            layer.attn_sinks, layer.wv_b, n_kv_max, 1.0f/sqrtf(float(qk_head)), il);
    cb(out, "attn_mla", il);

    out = ggml_reshape_3d(ctx0, out, v_mla, n_head, nt);
    out = build_head_mix(out, layer.attn_vha, il);
    out = ggml_reshape_2d(ctx0, out, v_mla*n_head, nt);

    out = ggml_mul(ctx0, out, gate);
    out = build_lora_mm(layer.wo, out);
    cb(out, "attn_out", il);

    return out;
}

ggml_tensor * llama_model_ernie5_5::graph::build_mla_attention(
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    GGML_ASSERT(inp_attn->self_k_idxs_full && inp_attn->self_attn_mask_full.raw);

    const mla_qk qk = build_mla_qk(cur, inp_pos, il);

    ggml_build_forward_expand(gf, qk.q);
    ggml_build_forward_expand(gf, mctx_raw->cpy_k_full(ctx0, qk.k, inp_attn->self_k_idxs_full, il));

    ggml_tensor * k = mctx_raw->get_k_full(ctx0, il);

    ggml_tensor * kq_mask  = inp_attn->self_attn_mask_full.raw;
    int64_t       n_kv_max = 0;
    if (inp_attn->self_attn_mask_idx.raw) {
        ggml_tensor * top_k = build_mla_top_k(cur, qk.q_c, k, inp_attn->self_attn_mask_idx.raw, inp_pos, il);
        kq_mask  = llm_graph_context::build_top_k_mask(kq_mask, top_k, il);
        n_kv_max = top_k->ne[0];
    }

    return build_mla_out(qk.q, k, kq_mask, qk.gate, n_kv_max, il);
}

ggml_tensor * llama_model_ernie5_5::graph::build_latent_moe(ggml_tensor * cur, int il) const {
    const auto & layer = model.layers[il];

    const int64_t nt = cur->ne[1];

    // split-feature routing: sigmoid(router 0) + sigmoid(router 1)
    ggml_tensor * gate_inp = layer.ffn_gate_inp;
    ggml_tensor * router0 = ggml_view_2d(ctx0, gate_inp, n_embd, n_expert, gate_inp->nb[1], 0);
    ggml_tensor * router1 = ggml_view_2d(ctx0, gate_inp, n_embd, n_expert, gate_inp->nb[1], n_expert*gate_inp->nb[1]);

    ggml_tensor * probs = ggml_add(ctx0,
            ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, router0, cur)),
            ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, router1, cur)));
    cb(probs, "ffn_moe_probs_split", il);

    // the bias-corrected scores pick the experts, the plain scores weight them
    ggml_tensor * selected = ggml_argsort_top_k(ctx0, ggml_add(ctx0, probs, layer.ffn_exp_probs_b), n_expert_used);
    cb(selected, "ffn_moe_topk_ernie", il);

    // renormalize over the selection and apply the per-expert scale on the dense scores, so build_moe_ffn's
    // gather yields the final weights
    ggml_tensor * w_sel = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, n_expert, nt), selected);
    ggml_tensor * w_sum = ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, w_sel, n_expert_used, nt));
    w_sum = ggml_clamp(ctx0, w_sum, 1e-20f, INFINITY);

    probs = ggml_div(ctx0, ggml_mul(ctx0, probs, layer.ffn_exp_scale), w_sum);
    cb(probs, "ffn_moe_probs_final", il);

    ggml_tensor * routed = build_lora_mm(layer.ffn_routed_down, cur);
    cb(routed, "ffn_routed_down", il);

    ggml_tensor * moe_out = build_moe_ffn(routed,
            nullptr,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SITU, false, 1.0f,
            LLAMA_EXPERT_GATING_FUNC_TYPE_NONE,
            il,
            probs,
            nullptr, nullptr, nullptr, nullptr,
            selected);
    cb(moe_out, "ffn_moe_out", il);

    moe_out = build_norm(moe_out, layer.ffn_routed_norm, nullptr, LLM_NORM_RMS, il);
    moe_out = build_lora_mm(layer.ffn_routed_up, moe_out);
    cb(moe_out, "ffn_routed_up", il);

    ggml_tensor * shexp = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SITU, LLM_FFN_PAR, il);
    cb(shexp, "ffn_shexp", il);

    return ggml_add(ctx0, moe_out, shexp);
}

ggml_tensor * llama_model_ernie5_5::graph::build_multimax(ggml_tensor * logits) const {
    // l + t0*relu(r0 - l) + t1*relu(l - r1) + t2*relu(r2 - l)^2 + t3*relu(l - r3)^2
    const auto & r = hparams.ernie_multimax;
    const float * t = r.data() + 4;

    ggml_tensor * below0 = ggml_relu(ctx0, ggml_scale_bias(ctx0, logits, -1.0f,  r[0]));
    ggml_tensor * above1 = ggml_relu(ctx0, ggml_scale_bias(ctx0, logits,  1.0f, -r[1]));
    ggml_tensor * below2 = ggml_relu(ctx0, ggml_scale_bias(ctx0, logits, -1.0f,  r[2]));
    ggml_tensor * above3 = ggml_relu(ctx0, ggml_scale_bias(ctx0, logits,  1.0f, -r[3]));

    ggml_tensor * out = logits;
    out = ggml_add(ctx0, out, ggml_scale(ctx0, below0, t[0]));
    out = ggml_add(ctx0, out, ggml_scale(ctx0, above1, t[1]));
    out = ggml_add(ctx0, out, ggml_scale(ctx0, ggml_sqr(ctx0, below2), t[2]));
    out = ggml_add(ctx0, out, ggml_scale(ctx0, ggml_sqr(ctx0, above3), t[3]));

    return out;
}

llama_model_ernie5_5::graph::graph(const llm_graph_params & params) :
    llama_model_deepseek4::graph(params, no_build{}) {
    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_attn_mask.raw);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = build_hc_mean(inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        const hc_pre attn_hc = build_hc_pre_ernie(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
        cb(attn_hc.cur, "hc_attn_pre", il);

        cur = build_norm(attn_hc.cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_swa(il)) {
            cur = build_attention(inp_dsv4, cur, inp_pos, il);
        } else {
            cur = build_mla_attention(inp_dsv4, cur, inp_pos, il);
        }

        inpL = build_hc_post(cur, residual, attn_hc.post, attn_hc.comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        const hc_pre ffn_hc = build_hc_pre_ernie(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        cb(ffn_hc.cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, ffn_hc.post);
        ggml_build_forward_expand(gf, ffn_hc.comb);

        cur = build_norm(ffn_hc.cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SITU, LLM_FFN_PAR, il);
        } else {
            cur = build_latent_moe(cur, il);
        }
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, ffn_hc.post, ffn_hc.comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = build_hc_mean(inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    ggml_tensor * flat     = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;

    // the MTP block reads the residual before the mHC head, of the output rows or of every row
    res->t_h_nextn = cparams.embeddings_nextn_masked ? flat_out : flat;
    cb(res->t_h_nextn, "h_nextn", -1);

    if (inp_out_ids) {
        inpL = ggml_reshape_3d(ctx0, flat_out, n_embd, hc, n_outputs);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output_raw", -1);

    cur = build_multimax(cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_ernie5_5::graph_mtp::graph_mtp(const llm_graph_params & params) :
    graph(params, no_build{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "ERNIE 5.5 has a single MTP block");
    GGML_ASSERT(cparams.nextn_layer_offset == 0 && "nextn_layer_offset out of range [0, n_layer_nextn)");
    ASSERT_EMBD_OR_TOKEN(ubatch);

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "ERNIE 5.5 MTP hidden width mismatch");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block not loaded");
    GGML_ASSERT(layer.nextn.hc_head_fn && layer.nextn.shared_head_norm && "MTP block without its head");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd = ubatch.token ? ggml_get_rows(ctx0, model.tok_embd, inp->tokens) : inp->embd;
    cb(tok_embd, "mtp_tok_embd", il);

    // the trunk's residual before its mHC head
    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    // h_proj(hnorm(h)) on every stream plus e_proj(enorm(e)): one eh_proj on [enorm(e) | hnorm(h)]
    ggml_tensor * h_norm = build_norm(h_state, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens);
    e_norm = ggml_repeat_4d(ctx0, e_norm, n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // as one 2-D product over hc*n_tokens rows: llama-imatrix skips a mul_mat whose ne[1] (here hc) is below 16
    ggml_tensor * eh = ggml_concat(ctx0, e_norm, h_norm, 0);
    ggml_tensor * inpL = build_lora_mm(layer.nextn.eh_proj, ggml_reshape_2d(ctx0, eh, 2*n_embd, hc*n_tokens));
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, hc, n_tokens);
    cb(inpL, "mtp_eh_proj", il);

    ggml_tensor * residual = inpL;
    const hc_pre attn_hc = build_hc_pre_ernie(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
    cb(attn_hc.cur, "mtp_hc_attn_pre", il);

    ggml_tensor * cur = build_norm(attn_hc.cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    // MLA over the MTP context's own cache, dense: drafts are verified, the indexer only matters past top_k + window
    {
        GGML_ASSERT(!hparams.is_swa(il) && "the ERNIE 5.5 MTP block is an MLA layer");

        const mla_qk qk = build_mla_qk(cur, inp_pos, il);
        const auto * mctx_base = inp_attn->mctx->get_base();

        ggml_build_forward_expand(gf, qk.q);
        ggml_build_forward_expand(gf, mctx_base->cpy_k(ctx0, qk.k, inp_attn->get_k_idxs(), il));

        cur = build_mla_out(qk.q, mctx_base->get_k(ctx0, il), inp_attn->self_attn_mask.raw, qk.gate, 0, il);
    }

    inpL = build_hc_post(cur, residual, attn_hc.post, attn_hc.comb, il);
    cb(inpL, "mtp_hc_attn_post", il);

    residual = inpL;
    const hc_pre ffn_hc = build_hc_pre_ernie(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
    cb(ffn_hc.cur, "mtp_hc_ffn_pre", il);

    cur = build_norm(ffn_hc.cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    cur = build_latent_moe(cur, il);
    cb(cur, "mtp_ffn_out", il);

    inpL = build_hc_post(cur, residual, ffn_hc.post, ffn_hc.comb, il);
    cb(inpL, "mtp_l_out", il);

    // the next draft step reads this block's residual in place of the trunk's
    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * h_nextn = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;
    cb(h_nextn, "h_nextn", -1);
    res->t_h_nextn = h_nextn;

    inpL = ggml_reshape_3d(ctx0, h_nextn, n_embd, hc, h_nextn->ne[1]);

    cur = build_hc_head(inpL, layer.nextn.hc_head_fn, layer.nextn.hc_head_scale, layer.nextn.hc_head_base);
    cb(cur, "mtp_hc_head", -1);

    cur = build_norm(cur, layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);
    res->t_embd = cur;

    // the trunk's head and MultiMax
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output_raw", -1);

    cur = build_multimax(cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
