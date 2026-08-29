#include "models.h"

#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <stdexcept>

//
// GLM-5.3-Flash: hybrid trunk, KDA linear attention on 3 of every 4 layers and MLA
// on the rest, with each attention and FFN block wrapped in hyper-connections (mHC).
//   - KDA is the kimi-linear tensor layout with the sigmoid decay gate of kimi-k3
//   - MLA is nope-only (qk_rope_head_dim = 0), so this graph has no positions
//   - mHC is the deepseek4 formulation, except the final collapse is a plain mean
//
// The full layers pick the cells they attend to with the DSA k-pool indexer, held in
// llama_memory_hybrid_idx next to the recurrent states and the MLA cache. A GGUF without
// indexer weights still builds the dense graph.
//
// The NextN/MTP block appended after the trunk is a plain pre-norm decoder block, with
// neither hyper-connections nor KDA. It has indexer weights, but the reference shares the
// trunk index for the MTP step (index_share_for_mtp_iteration), which a separate MTP
// context cannot see, so the draft head attends densely. See graph_mtp below.
//

void llama_model_glm5next::load_arch_hparams(llama_model_loader & ml) {
    // read first: everything below uses n_layer() == n_layer_all - n_layer_nextn
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // the indexer k_norm is a plain LayerNorm - fall back to the torch default eps
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps, false);
    if (hparams.f_norm_eps == 0.0f) {
        hparams.f_norm_eps = 1e-5f;
    }

    // MLA
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl);

    if (!hparams.is_mla()) {
        throw std::runtime_error("GLM5NEXT requires MLA (key_length_mla / value_length_mla)");
    }

    // mla_use_nope: the graph relies on the model being position-free
    GGML_ASSERT(hparams.n_rot() == 0 && "GLM5NEXT is nope-only: rope.dimension_count must be 0");

    // MoE
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,  hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm,  false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func,   false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    // swiglu_limit clamps the gate before the SiLU (deepseek4 semantics, see build_ffn)
    // and applies to the routed experts, the shared expert and the dense MLPs alike
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, hparams.n_layer_all, false)) {
        LLAMA_LOG_WARN("%s: glm5next.swiglu_clamp_exp is missing, "
                "assuming the GLM-5.3-Flash swiglu_limit of 10.0\n", __func__);
        hparams.swiglu_clamp_exp.fill(10.0f);
    }
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all, false)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    // DSA indexer - absent in a GGUF without indexer weights, which then runs dense MLA
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,  hparams.indexer_n_head,     false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,  hparams.indexer_head_size,  false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,       hparams.indexer_top_k,      false);
    // index_kpool: tokens per compressed key, always_select_tail is implied (see build_dsa_top_k)
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,  hparams.indexer_block_size, false);
    if (hparams.indexer_block_size == 0) {
        // some converters write the k-pool size under `indexer.kpool` rather than
        // `indexer.block_size`; accept it so those GGUFs load without an override
        ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL, hparams.indexer_block_size, false);
    }

    if (hparams.indexer_head_size > 0) {
        GGML_ASSERT(hparams.indexer_n_head > 0);
        GGML_ASSERT(hparams.indexer_block_size > 0 && "GLM5NEXT requires index_kpool");
        GGML_ASSERT(hparams.indexer_top_k % hparams.indexer_block_size == 0);
    }

    // GLM-5.3-Flash has indexer_types = "full" everywhere
    std::fill(hparams.is_indexer_full_impl.begin(), hparams.is_indexer_full_impl.end(), 1);
    ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer(), false);

    // hyper-connections (mHC)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult == 4          && "GLM5NEXT expects hyper_connection.count == 4");
    GGML_ASSERT(hparams.dsv4_hc_sinkhorn_iters > 0 && "GLM5NEXT expects hyper_connection.sinkhorn_iterations > 0");

    // the hc streams collapse to a mean, so the output is plain n_embd. deepseek4 sets
    // this to hc_mult*n_embd only to size its MTP buffer; keeping such a value here
    // would make llama_context read past the end of t_embd.
    hparams.n_embd_out_impl = 0;

    // KDA
    ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv);
    if (!ml.get_key(LLM_KV_KDA_HEAD_DIM, hparams.n_embd_head_kda, false)) {
        // older GGUFs store the KDA head dim as ssm.state_size
        ml.get_key(LLM_KV_SSM_STATE_SIZE, hparams.n_embd_head_kda);
    }
    if (!ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND, hparams.kda_gate_lower_bound, false)) {
        // linear_attn_config.gate_lower_bound
        hparams.kda_gate_lower_bound = -5.0f;
    }
    // only the bounded sigmoid gate is implemented, the softplus branch is dead here
    GGML_ASSERT(hparams.kda_gate_lower_bound < 0.0f);

    // note: n_embd_r()/n_embd_s() size the recurrent state with n_head()*n_embd_head_kda,
    // which works only because linear_attn_config.num_heads == num_attention_heads

    // MLA forces num_key_value_heads = 1 on every layer at conversion time, so the
    // kimi-linear "n_head_kv == 0" recurrent marker is not available here.
    // the per-layer arrays cover the trunk only, like the other glm5next ones - the
    // NextN block is never recurrent, so leave its entry at 0
    std::fill(hparams.is_recr_impl.begin(), hparams.is_recr_impl.end(), 0);
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer(), false)) {
        uint32_t full_attn_interval = 4; // layer_types: full attention on 3, 7, 11, ...
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        GGML_ASSERT(full_attn_interval > 0);
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            hparams.is_recr_impl[il] = (il < hparams.n_layer()) && ((il + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 45: type = LLM_TYPE_312B_A17B; break; // GLM-5.3-Flash
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_glm5next::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t kv_lora_rank    = hparams.n_lora_kv;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = std::max<int64_t>(1, hparams.n_expert_shared);

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla   = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot(); // 0, nope-only
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t n_idx_head = hparams.indexer_n_head;
    const int64_t n_idx_dim  = hparams.indexer_head_size;
    const int64_t kpool      = hparams.indexer_block_size;

    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc * n_embd;
    const int64_t hc_mix_dim = (2 + hc) * hc;

    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t d_inner  = head_dim * n_head;

    // a GGUF that declares nextn layers but ships the trunk alone still loads
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);

    // the MTP block is materialized only for an MTP context (--spec-type draft-mtp)
    int mtp_flags = trunk_only ? TENSOR_NOT_REQUIRED : 0;
    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }
    // note: no hc_head_*, the hyper-connection head is a plain mean

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        const bool is_mtp  = i >= n_layer;
        const bool is_recr = !is_mtp && hparams.is_recr(i);
        const int  flags   = is_mtp ? mtp_flags : 0;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, flags);

        // hyper-connections wrap the trunk blocks only, the MTP block uses plain residuals
        if (!is_mtp) {
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim},         0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3},                  0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim},         0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3},                  0);
        }

        if (is_recr) {
            // conv1d may be stored 4D [d_conv, 1, d_inner, 1] or 3D (quantization drops the trailing 1)
            auto conv = [&](llm_tensor tid) {
                ggml_tensor * t = create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
                return t ? t : create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner}, 0);
            };
            layer.ssm_q_conv = conv(LLM_TENSOR_SSM_CONV1D_Q);
            layer.ssm_k_conv = conv(LLM_TENSOR_SSM_CONV1D_K);
            layer.ssm_v_conv = conv(LLM_TENSOR_SSM_CONV1D_V);

            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0);

            layer.ssm_f_a  = create_tensor(tn(LLM_TENSOR_SSM_F_A,  "weight", i), {n_embd,   head_dim},  0);
            layer.ssm_f_b  = create_tensor(tn(LLM_TENSOR_SSM_F_B,  "weight", i), {head_dim, d_inner},   0);
            layer.ssm_g_a  = create_tensor(tn(LLM_TENSOR_SSM_G_A,  "weight", i), {n_embd,   head_dim},  0);
            layer.ssm_g_b  = create_tensor(tn(LLM_TENSOR_SSM_G_B,  "weight", i), {head_dim, d_inner},   0);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd,   n_head},    0);

            // ssm_a holds -exp(A_log), folded at conversion time (kimi-linear/kimi-k3 convention)
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {n_head}, 0);

            // some converters emit dt_bias under the default ".weight" suffix
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_dt_b) {
                layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "weight", i), {d_inner}, 0);
            }

            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim}, 0);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else {
            layer.wq_a           = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,       "weight", i), {n_embd, q_lora_rank}, flags);
            layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, flags);
            layer.wq_b           = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,       "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, flags);
            // nope-only: kv_lora_rank + 0
            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);
            layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, flags);

            // DSA indexer + k-pool compressor. The MTP block ships one too, but the draft
            // head runs dense (see graph_mtp), so it stays loaded and unused there
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, n_idx_head * n_idx_dim}, flags | TENSOR_NOT_REQUIRED);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, n_idx_dim}, flags | TENSOR_NOT_REQUIRED);
            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_idx_dim}, flags | TENSOR_NOT_REQUIRED);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {n_idx_dim}, flags | TENSOR_NOT_REQUIRED);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_idx_head}, flags | TENSOR_NOT_REQUIRED);

            layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {n_idx_dim, kpool}, flags | TENSOR_NOT_REQUIRED);
            layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", i), {n_embd, n_idx_dim}, flags | TENSOR_NOT_REQUIRED);
        }

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, flags);
        } else {
            if (n_expert == 0 || n_expert_used == 0) {
                throw std::runtime_error("GLM5NEXT requires n_expert > 0 and n_expert_used > 0");
            }

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, flags | TENSOR_NOT_REQUIRED);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        if (is_mtp) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2*n_embd, n_embd}, mtp_flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, mtp_flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, mtp_flags);

            // GLM-5.3-Flash ties these to the trunk embeddings / LM head, so they are absent
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab}, mtp_flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab}, mtp_flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, mtp_flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_glm5next::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }

    return std::make_unique<graph>(*this, params);
}

//
// hyper-connections (mHC), per token, with hc_dim = hc*n_embd:
//
//   mixes             = hc_fn @ rms_norm(streams.flatten())         [(2 + hc)*hc]
//   pre, post, comb_w = mixes.split([hc, hc, hc*hc])
//
//   pre  = sigmoid(pre *scale[0] + base[  0:  hc]) + eps            [hc]
//   post = sigmoid(post*scale[1] + base[ hc:2*hc])*2                [hc]
//   comb = sinkhorn(softmax(comb_w*scale[2] + base[2*hc:]) + eps)   [hc, hc]
//
//   sublayer_in = sum_h pre[h]*streams[h]
//   out[dst]    = post[dst]*sublayer_out + sum_src comb[dst, src]*streams[src]
//
// comb is stored as [dst_hc, src_hc, n_tokens]: the reference matrix is row-major
// with the softmax over its last axis, so ne0 is that axis after the reshape, which
// is the transpose the reference matmul asks for.
//

static ggml_tensor * glm5next_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
}

static ggml_tensor * glm5next_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
}

static ggml_tensor * glm5next_hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
    return ggml_add(ctx, ggml_mul(ctx, x, scale), base);
}

// Glm5NextTextHyperHead: unweighted mean over the streams
// [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
static ggml_tensor * glm5next_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }

    return ggml_scale(ctx, acc, 1.0f/hc);
}

ggml_tensor * llama_model_glm5next::graph::build_hc_collapse(ggml_tensor * x, ggml_tensor * weights, int il) {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == (int64_t) hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llama_model_glm5next::graph::build_hc_sinkhorn(ggml_tensor * comb, int il) {
    GGML_UNUSED(il);

    const float eps = hparams.dsv4_hc_eps;

    // comb is [dst_hc, src_hc, n_tokens]: ne0 is the reference's softmax axis
    comb = ggml_soft_max(ctx0, comb);
    comb = ggml_scale_bias(ctx0, comb, 1.0f, eps);

    // normalize over the reference's dim=-2, which is our ne1
    auto norm_cols = [&]() {
        ggml_tensor * t   = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * sum = ggml_sum_rows(ctx0, t);
        sum  = ggml_scale_bias(ctx0, sum, 1.0f, eps);
        sum  = ggml_permute(ctx0, sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, sum);
    };

    // normalize over the reference's dim=-1, which is our ne0
    auto norm_rows = [&]() {
        ggml_tensor * sum = ggml_sum_rows(ctx0, comb);
        sum  = ggml_scale_bias(ctx0, sum, 1.0f, eps);
        comb = ggml_div(ctx0, comb, sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

ggml_tensor * llama_model_glm5next::graph::build_hc_pre(
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int            il) {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hc);
    GGML_ASSERT(hc_fn->ne[0] == hc_dim);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    // DeepseekV4UnweightedRMSNorm: no learned gain
    ggml_tensor * flat  = ggml_rms_norm(ctx0, ggml_reshape_2d(ctx0, x, hc_dim, nt), norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = glm5next_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = glm5next_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = glm5next_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = glm5next_view_1d(ctx0, hc_base, hc, hc);

    ggml_tensor * pre = glm5next_view_2d(ctx0, mixes, hc, nt, 0);
    pre = glm5next_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    *post = glm5next_view_2d(ctx0, mixes, hc, nt, hc);
    *post = glm5next_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    if (cparams.fused_dsv4_hc_comb &&
            hc_scale->type == GGML_TYPE_F32 && hc_base->type == GGML_TYPE_F32) {
        *comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, *comb, il});
    } else {
        ggml_tensor * scale_comb = glm5next_view_1d(ctx0, hc_scale, 1,     2);
        ggml_tensor * base_comb  = glm5next_view_1d(ctx0, hc_base,  hc*hc, 2*hc);

        *comb = glm5next_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = glm5next_hc_affine(ctx0, *comb, scale_comb, base_comb);
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);

    return build_hc_collapse(x, pre, il);
}

ggml_tensor * llama_model_glm5next::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int           il) {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == (int64_t) hparams.dsv4_hc_mult);

    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur      = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_sd = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2], dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_sd));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

//
// KDA layer
//

// causal conv1d over one of Q/K/V. `qkv` selects which third of the conv state to use
static ggml_tensor * glm5next_causal_conv1d(
        ggml_cgraph * gf, ggml_context * ctx0,
        ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
        int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
        int64_t d_conv, int64_t head_dim, int64_t n_head,
        int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens,
        int64_t cache_head, uint32_t mem_size, uint32_t n_rs_seq) {
    const int64_t d_inner          = head_dim * n_head;
    const int64_t conv_state_size  = (d_conv - 1) * d_inner;
    const int64_t total_state_size = 3 * conv_state_size;

    ggml_tensor * conv_state = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
            (d_conv - 1)     * ggml_element_size(conv_state_all),
            total_state_size * ggml_element_size(conv_state_all),
            qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    x_proj = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state, ggml_transpose(ctx0, x_proj), 0);

    // one snapshot per rollback slot, newest first
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, (int64_t) n_rs_seq + 1);

    for (int64_t slot = 0; slot < n_written; ++slot) {
        ggml_tensor * snap = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
                conv_x->nb[1], conv_x->nb[2], (conv_x->ne[0] - (d_conv - 1) - slot) * conv_x->nb[0]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, snap,
                ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                    (d_conv - 1)     * ggml_element_size(conv_states_all),
                    total_state_size * ggml_element_size(conv_states_all),
                    ((slot * mem_size + cache_head) * total_state_size + qkv * conv_state_size)
                        * ggml_element_size(conv_states_all))));
    }

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * out = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    out = ggml_silu(ctx0, ggml_reshape_2d(ctx0, out, d_inner, n_tokens));

    return ggml_reshape_4d(ctx0, out, head_dim, n_head, n_seq_tokens, n_seqs);
}

ggml_tensor * llama_model_glm5next::graph::build_kda_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs,
        int64_t n_seq_tokens, int64_t n_seqs, int il) {
    const int64_t n_head_kda = hparams.n_head();
    const int64_t head_dim   = hparams.n_embd_head_kda;
    const int64_t d_inner    = n_head_kda * head_dim;
    const int64_t d_conv     = hparams.ssm_d_conv;

    const auto * mctx_cur  = inp_rs->mctx;
    const auto   cache_head = mctx_cur->get_head();
    const auto   mem_size   = mctx_cur->get_size();

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

    ggml_tensor * q = glm5next_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0,
            cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda,
            n_seq_tokens, n_seqs, n_tokens, cache_head, mem_size, cparams.n_rs_seq);
    ggml_tensor * k = glm5next_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1,
            cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda,
            n_seq_tokens, n_seqs, n_tokens, cache_head, mem_size, cparams.n_rs_seq);
    ggml_tensor * v = glm5next_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2,
            cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda,
            n_seq_tokens, n_seqs, n_tokens, cache_head, mem_size, cparams.n_rs_seq);
    cb(q, "kda_q_conv", il);
    cb(k, "kda_k_conv", il);
    cb(v, "kda_v_conv", il);

    // forget gate: g = gate_lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)).
    // ssm_a holds -exp(A_log), so exp(A_log)*(...) == -(ssm_a*(...)).
    ggml_tensor * g = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
    g = ggml_mul_mat(ctx0, layer.ssm_f_b, g);
    g = ggml_add(ctx0, g, layer.ssm_dt_b);
    g = ggml_reshape_3d(ctx0, g, head_dim, n_head_kda, n_tokens);
    g = ggml_mul(ctx0, g, ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kda, 1));
    g = ggml_sigmoid(ctx0, ggml_scale(ctx0, g, -1.0f));
    g = ggml_scale(ctx0, g, hparams.kda_gate_lower_bound);
    g = ggml_reshape_4d(ctx0, g, head_dim, n_head_kda, n_seq_tokens, n_seqs);
    cb(g, "kda_gate", il);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
    beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, n_head_kda, n_seq_tokens, n_seqs));
    cb(beta, "kda_beta", il);

    // the reference uses a hard-coded 1e-6 here; build_delta_net applies the
    // 1/sqrt(head_dim) scaling of q after this
    q = ggml_l2_norm(ctx0, q, 1e-6f);
    k = ggml_l2_norm(ctx0, k, 1e-6f);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_kda, n_seqs);

    ggml_tensor * out = ggml_cont(ctx0, build_recurrent_attn(
            inp_rs, ssm_states_all, q, k, v, g, beta, state, il));

    // Glm5NextTextRMSNormGated: RMSNorm first, then a SIGMOID gate (not SiLU)
    ggml_tensor * o_gate = ggml_mul_mat(ctx0, layer.ssm_g_a, cur);
    o_gate = ggml_mul_mat(ctx0, layer.ssm_g_b, o_gate);
    o_gate = ggml_reshape_3d(ctx0, o_gate, head_dim, n_head_kda, n_tokens);

    out = ggml_reshape_3d(ctx0, out, head_dim, n_head_kda, n_tokens);
    out = build_norm(out, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, o_gate));
    cb(out, "kda_normed_gated", il);

    cur = ggml_mul_mat(ctx0, layer.wo, ggml_cont_2d(ctx0, out, d_inner, n_tokens));
    cb(cur, "kda_out", il);

    return cur;
}

//
// DSA k-pool indexer
//
// The full layers attend to index_topk cells picked by a lightning indexer that scores
// k-pools of index_kpool tokens instead of single tokens. A pool key is a learned per-channel
// convex mix softmax(gate + ape) . keys of its members, so the cache must hold both the
// indexer key and the gate logits of every token.
//
// Everything that depends on the cache layout is computed host-side in set_input; the graph
// only gathers, pools and scores. One input serves every layer: the pool metadata is the same
// for all of them.

class llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t kpool) :
        mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_kpool(pool_cells, pool_bias, tail_cells, ubatch, kpool);

        GGML_ASSERT(ggml_backend_buffer_is_host(ape_slots->buffer));
        int32_t * data = (int32_t *) ape_slots->data;
        for (int64_t i = 0; i < ape_slots->ne[0]; ++i) {
            data[i] = (int32_t) i;
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx_cur = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        mctx = mctx_cur;

        bool res = true;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= pool_bias->ne[0] == (int64_t) (mctx_cur->get_idx()->get_n_kv()/kpool);
        res &= pool_bias->ne[1]*pool_bias->ne[2] == params.ubatch.n_tokens;

        return res;
    }

    ggml_tensor * k_idxs     = nullptr; // I64 [n_tokens]
    ggml_tensor * ape_slots  = nullptr; // I32 [kpool], the identity - reads the ape rows in order
    ggml_tensor * pool_cells = nullptr; // I32 [kpool*n_pools, n_stream]
    ggml_tensor * pool_bias  = nullptr; // F32 [n_pools, n_tokens/n_stream, n_stream]
    ggml_tensor * tail_cells = nullptr; // I32 [kpool-1, n_tokens/n_stream, 1, n_stream], null when kpool == 1

    const llama_memory_hybrid_idx_context * mctx;

    const uint32_t kpool;
};

llm_graph_input_kpool * llama_model_glm5next::graph::build_inp_kpool(llm_graph_input_mem_hybrid_idx * inp_hyb) {
    const auto * mctx_idx = inp_hyb->mctx->get_idx();

    if (!mctx_idx) {
        return nullptr;
    }

    const int64_t r      = hparams.indexer_block_size;
    const int64_t n_kv   = mctx_idx->get_n_kv();
    const int64_t n_pool = n_kv/r;
    // the KQ mask carries the stream count that build_attn and the cache views agree on
    const int64_t ns     = inp_hyb->get_attn()->get_kq_mask()->ne[3];

    // the top-k indices address the attention cache, so the two must agree cell for cell
    GGML_ASSERT(n_kv == (int64_t) inp_hyb->mctx->get_attn()->get_n_kv());
    GGML_ASSERT(n_tokens % ns == 0);

    if (n_pool == 0) {
        return nullptr;
    }

    auto kp = std::make_unique<llm_graph_input_kpool>(inp_hyb->mctx, (uint32_t) r);

    kp->k_idxs     = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    kp->ape_slots  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r);
    kp->pool_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_pool, ns);
    kp->pool_bias  = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_pool, n_tokens/ns, ns);

    ggml_set_input(kp->ape_slots);
    ggml_set_input(kp->pool_cells);
    ggml_set_input(kp->pool_bias);

    if (r > 1) {
        // 4d, so it concatenates straight onto the expanded pools in build_dsa_top_k
        kp->tail_cells = ggml_new_tensor_4d(ctx0, GGML_TYPE_I32, r - 1, n_tokens/ns, 1, ns);
        ggml_set_input(kp->tail_cells);
    }

    return (llm_graph_input_kpool *) res->add_input(std::move(kp));
}

ggml_tensor * llama_model_glm5next::graph::build_dsa_top_k(
        llm_graph_input_kpool * inp, ggml_tensor * cur,
        ggml_tensor * qr, const llama_layer & layer, int il) {
    const auto * mctx_idx = inp->mctx->get_idx();

    const int64_t d      = hparams.indexer_head_size;
    const int64_t nh     = hparams.indexer_n_head;
    const int64_t r      = hparams.indexer_block_size;
    const int64_t n_kv   = mctx_idx->get_n_kv();
    const int64_t ns     = inp->pool_cells->ne[1];
    const int64_t n_pool = inp->pool_cells->ne[0]/r;
    const int64_t n_tps  = n_tokens/ns;

    // key and gate logits packed into one cache row: the gate depends on the token hidden
    // state, so it cannot be recomputed from the cache later
    ggml_tensor * k = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);
    k = build_norm(k, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);

    ggml_tensor * g = ggml_mul_mat(ctx0, layer.indexer_comp_wgate, cur);

    ggml_tensor * packed = ggml_reshape_3d(ctx0, ggml_concat(ctx0, k, g, 0), 2*d, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, packed, inp->k_idxs, il));

    // one key head, so cache rows are contiguous
    ggml_tensor * all = mctx_idx->get_k(ctx0, il);
    all = ggml_view_3d(ctx0, all, 2*d, n_kv, ns, all->nb[2], all->nb[3], 0);

    // gathers per stream: pool_cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, all, inp->pool_cells);
    members = ggml_reshape_4d(ctx0, members, 2*d, r, n_pool, ns);

    ggml_tensor * m_k = ggml_cont(ctx0, ggml_view_4d(ctx0, members, d, r, n_pool, ns,
            members->nb[1], members->nb[2], members->nb[3], 0));
    ggml_tensor * m_g = ggml_cont(ctx0, ggml_view_4d(ctx0, members, d, r, n_pool, ns,
            members->nb[1], members->nb[2], members->nb[3], ggml_row_size(members->type, d)));

    // pool key = softmax(gate + ape) . keys over the r members, channel by channel. The ape is
    // an intra-pool position bias and, with no rope anywhere, the only ordering signal here.
    m_g = ggml_add(ctx0, m_g, ggml_get_rows(ctx0, layer.indexer_comp_ape, inp->ape_slots));

    // softmax normalizes ne[0], so bring the member axis there
    ggml_tensor * w = ggml_soft_max(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, m_g, 1, 0, 2, 3)));
    ggml_tensor * v = ggml_cont(ctx0, ggml_permute(ctx0, m_k, 1, 0, 2, 3));

    ggml_tensor * pooled = ggml_sum_rows(ctx0, ggml_mul(ctx0, v, w));
    pooled = ggml_cont(ctx0, ggml_permute(ctx0, pooled, 1, 0, 2, 3));
    pooled = ggml_reshape_3d(ctx0, pooled, d, n_pool, ns);
    cb(pooled, "indexer_k_pooled", il);

    // nope-only: no rope on the indexer query either. mul_mat matches ne[2], so stream s's
    // queries only meet stream s's pools.
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);
    q = ggml_reshape_3d(ctx0, q, d, nh*n_tps, ns);
    cb(q, "indexer_q", il);

    // relu(x*s) == s*relu(x) for s > 0, so both positive scalars (the softmax scale and
    // n_heads^-1/2) fold into the small weights tensor instead of the big score one
    ggml_tensor * wts = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
    wts = ggml_scale(ctx0, wts, 1.0f/sqrtf(float(d*nh)));
    wts = ggml_reshape_4d(ctx0, wts, nh, 1, n_tps, ns);

    // index_topk cells worth of whole pools, i.e. the reference's index_topk/index_kpool
    const int64_t n_sel = std::min<int64_t>(n_pool, (int64_t) hparams.indexer_top_k/r);

    // expand each selected pool into its members. get_rows indexes src0 ne[2] with the index
    // ne[1], so the stream axis must stay there and n_sel*n_tps folds into one row axis
    ggml_tensor * pools = ggml_reshape_3d(ctx0, inp->pool_cells, r, n_pool, ns);

    // The scores are [n_pool, nh, n_tokens] and the head reduction needs the head axis in
    // ne[0], so the tensor is materialised twice - once by the mul_mat and once by the
    // permute+cont below. That is 2*n_pool*nh*n_tokens*4 B PER DEVICE (the DSA layers are
    // spread across the trunk, so every device in a layer split pays it): 16 MiB per token
    // at n_ctx 262144 with kpool 4 and 32 heads. It therefore caps ubatch - -ub 4096 there
    // asks for ~70 GiB on a single device.
    //
    // Scoring a token depends only on its own query and on `pooled`, which is shared across
    // the batch, and no reduction in this path runs across tokens. So the token loop can be
    // split into chunks with identical results, and ggml-alloc reuses one buffer across the
    // chunks - bounding the scratch by the chunk size instead of by the ubatch.
    //
    // The chunk is sized to keep that scratch near a fixed target: small enough to leave room
    // on ~8 GB devices, large enough that the extra kernel launches stay amortised. Short
    // contexts, where the tensor is small anyway, come out unchunked and pay nothing.
    // One step covers nc*ns tokens, so ns belongs in the divisor.
    constexpr int64_t idx_scratch_target = 2ll*1024*1024*1024;

    const int64_t idx_bytes_per_step = 2*n_pool*nh*ns*(int64_t) sizeof(float);
    const int64_t idx_chunk = std::clamp<int64_t>(idx_scratch_target/idx_bytes_per_step, 1, n_tps);

    ggml_tensor * top_k = nullptr;

    for (int64_t t0 = 0; t0 < n_tps; t0 += idx_chunk) {
        const int64_t nc = std::min<int64_t>(idx_chunk, n_tps - t0);

        ggml_tensor * q_c    = q;
        ggml_tensor * wts_c  = wts;
        ggml_tensor * bias_c = inp->pool_bias;
        ggml_tensor * tail_c = inp->tail_cells;

        if (nc != n_tps) {
            // q packs (head, token) in ne[1] with the head fastest, so a token range is the
            // row range [nh*t0, nh*(t0 + nc)). Strides are kept, so this is also correct for
            // n_stream > 1, where a token slice is not contiguous across streams.
            q_c = ggml_view_3d(ctx0, q, d, nh*nc, ns,
                    q->nb[1], q->nb[2], (size_t) (nh*t0)*q->nb[1]);
            wts_c = ggml_view_4d(ctx0, wts, nh, 1, nc, ns,
                    wts->nb[1], wts->nb[2], wts->nb[3], (size_t) t0*wts->nb[2]);
            bias_c = ggml_view_3d(ctx0, inp->pool_bias, n_pool, nc, ns,
                    inp->pool_bias->nb[1], inp->pool_bias->nb[2],
                    (size_t) t0*inp->pool_bias->nb[1]);
            if (tail_c) {
                tail_c = ggml_view_4d(ctx0, inp->tail_cells, r - 1, nc, 1, ns,
                        inp->tail_cells->nb[1], inp->tail_cells->nb[2], inp->tail_cells->nb[3],
                        (size_t) t0*inp->tail_cells->nb[1]);
            }
        }

        ggml_tensor * score = ggml_mul_mat(ctx0, pooled, q_c);
        score = ggml_relu(ctx0, ggml_reshape_4d(ctx0, score, n_pool, nh, nc, ns));

        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));
        score = ggml_sum_rows(ctx0, ggml_mul(ctx0, score, wts_c));
        score = ggml_reshape_3d(ctx0, score, n_pool, nc, ns);

        // the cut is on whole pools, never on single cells. Scoring cells with their pool's
        // score and cutting there is not the same thing: relu sends many distinct pools to
        // exactly 0.0 and ggml_top_k is unordered among equal keys, so the cut splits pools
        // apart. Diagnosis and the reference-free check for it (count partly selected pools)
        // are from PR #27754.
        score = ggml_add(ctx0, score, bias_c);

        ggml_tensor * sel = ggml_top_k(ctx0, score, n_sel);

        // only meaningful when the loop runs once; otherwise an eval-callback dump would get
        // one identically-named tensor per chunk per layer
        if (nc == n_tps) {
            cb(score, "indexer_score_pools", il);
            cb(sel,   "indexer_top_k_pools", il);
        }

        ggml_tensor * tk = ggml_get_rows(ctx0, pools,
                ggml_reshape_3d(ctx0, sel, n_sel*nc, ns, 1));

        // member j of the i-th selected pool is at i*r + j in both layouts, so this is a reshape
        tk = ggml_reshape_4d(ctx0, tk, r*n_sel, nc, 1, ns);

        // index_kpool_always_select_tail: the trailing incomplete pool has no pool key and can
        // never be picked above, so its cells are appended instead of taking pool budget
        if (tail_c) {
            tk = ggml_concat(ctx0, tk, tail_c, 0);
        }

        // appends this chunk's tokens along the token axis; recopies earlier chunks each
        // iteration, which is negligible I32 traffic next to the scoring GEMMs
        top_k = top_k ? ggml_concat(ctx0, top_k, tk, 1) : tk;
    }

    // build_attn_mask_top_k reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

//
// MLA layer (nope-only, absorbed - i.e. MQA with a single group)
//

ggml_tensor * llama_model_glm5next::graph::build_mla_layer(
        ggml_tensor * cur, const llama_layer & layer,
        llm_graph_input_attn_k * inp_attn, llm_graph_input_kpool * inp_kpool,
        float kq_scale, int il) {
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank  = hparams.n_lora_kv;

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr", il);

    // nope-only: the whole of q is the "nope" part, no split and no rope
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
    cb(q, "q", il);

    // {n_embd_head_k, n_tokens, n_head} x wk_b -> {kv_lora_rank, n_tokens, n_head}
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wk_b, q);
    Qcur = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
    cb(Qcur, "Qcur", il);

    // nope-only: wkv_a_mqa outputs exactly kv_lora_rank, no k_pe to split off
    ggml_tensor * kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    cb(kv_cmpr, "kv_cmpr", il);

    // null top_k = attend to the whole cache, for a GGUF without indexer weights
    ggml_tensor * top_k = inp_kpool && layer.indexer_attn_q_b
        ? build_dsa_top_k(inp_kpool, cur, qr, layer, il) : nullptr;

    cur = build_attn(inp_attn, layer.wo, nullptr, layer.wo_s,
            Qcur, kv_cmpr, kv_cmpr, nullptr, nullptr, layer.wv_b, top_k, kq_scale, il);
    cb(cur, "mla_out", il);

    return cur;
}

//
// FFN: leading dense layers, then MoE with a shared expert
//

ggml_tensor * llama_model_glm5next::graph::build_ffn_layer(ggml_tensor * cur, const llama_layer & layer, int il) {
    if ((uint32_t) il < hparams.n_layer_dense_lead) {
        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
        return cur;
    }

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
            layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    ggml_tensor * shexp = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
            layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
            layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(shexp, "ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, shexp);
    cb(cur, "ffn_out", il);

    return cur;
}

//
// trunk graph
//

llama_model_glm5next::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {

    const int64_t hc           = hparams.dsv4_hc_mult;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // nope-only, so no YaRN mscale correction on kq_scale
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_mla()));

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    cb(inp, "inp_embd", -1);

    // MLA with absorption uses a K-only cache (V is a view of K)
    auto * inp_hyb  = build_inp_mem_hybrid_idx();
    auto * inp_rs   = inp_hyb->get_recr();
    auto * inp_attn = inp_hyb->get_attn();

    // the k-pool metadata is the same for every full layer, so build it once
    auto * inp_kpool = build_inp_kpool(inp_hyb);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // inputs_embeds.unsqueeze(2).expand(-1, -1, hc, -1)
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        // attention site
        ggml_tensor * cur = build_hc_pre(inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = hparams.is_recr(il)
            ? build_kda_layer(cur, layer, inp_rs, n_seq_tokens, n_seqs, il)
            : build_mla_layer(cur, layer, inp_attn, inp_kpool, kq_scale, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        // FFN site
        residual = inpL;
        cur = build_hc_pre(inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn_layer(cur, layer, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // unmasked nextn embeddings need every row, so narrow after the final norm instead
    const bool narrow_late = cparams.embeddings_nextn && !cparams.embeddings_nextn_masked;

    // narrow to the output rows before collapsing the streams
    if (inp_out_ids && !narrow_late) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    ggml_tensor * cur = glm5next_hc_mean(ctx0, inpL);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // post-norm hidden state feeds the NextN/MTP draft head
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids && narrow_late) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

//
// NextN/MTP draft head
//
// enorm(embed) + hnorm(prev_hidden) -> concat(e, h) -> eh_proj -> one plain pre-norm
// decoder block (nope-only MLA + sigmoid-gated MoE with a shared expert, built the same
// way as build_mla_layer/build_ffn_layer build the trunk) -> shared_head_norm -> LM head.
//
// Differences from a trunk layer:
//   - no hyper-connections: the block has no hc_* tensors, it uses plain residuals
//   - dense attention: the reference shares the trunk index for the MTP step, which this
//     separate context cannot see. This costs acceptance rate, never correctness.
//

llama_model_glm5next::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "GLM5NEXT MTP supports a single NextN block");
    GGML_ASSERT(hparams.is_mla() && "GLM5NEXT MTP requires MLA");
    GGML_ASSERT(hparams.n_rot() == 0 && "GLM5NEXT MTP is nope-only");

    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;

    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj (load with --spec-type draft-mtp)");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp  && "MTP block missing ffn_gate_inp");

    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank  = hparams.n_lora_kv;

    // nope-only, so no YaRN mscale correction - must match the trunk graph
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    // no build_inp_pos(): glm5next is position-free
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MLA with the absorption optimization uses a K-only cache (V is a view of K)
    auto * inp_attn = build_attn_inp_k();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    {
        ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
        qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(qr, "mtp_qr", il);

        // nope-only: the whole of q is the "nope" part, no split and no rope
        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
        q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
        cb(q, "mtp_q", il);

        // {n_embd_head_k, n_tokens, n_head} x wk_b -> {kv_lora_rank, n_tokens, n_head}
        q = ggml_permute(ctx0, q, 0, 2, 1, 3);
        ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wk_b, q);
        Qcur = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        cb(Qcur, "mtp_Qcur", il);

        // nope-only: wkv_a_mqa outputs exactly kv_lora_rank, no k_pe to split off
        ggml_tensor * kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
        kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        cb(kv_cmpr, "mtp_kv_cmpr", il);

        cur = build_attn(inp_attn, layer.wo, nullptr, layer.wo_s,
                Qcur, kv_cmpr, kv_cmpr, nullptr, nullptr, layer.wv_b, kq_scale, il);
        cb(cur, "mtp_attn_out", il);
    }

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    // the NextN block is always past n_layer_dense_lead, so there is no dense-MLP branch
    {
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
                layer.ffn_gate_up_exps,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s);
        cb(moe_out, "mtp_ffn_moe_out", il);

        ggml_tensor * shexp = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(shexp, "mtp_ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, shexp);
        cb(cur, "mtp_ffn_out", il);
    }

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "GLM5NEXT MTP: missing both nextn.shared_head_norm and output_norm");

    // the post-norm hidden state would seed a chained head, unused at n_layer_nextn == 1
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "GLM5NEXT MTP: missing LM head (nextn.shared_head_head or model.output)");

    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
