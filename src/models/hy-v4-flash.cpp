#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"

#include <cmath>

// hy_v4 iHC, gated MLA and sinks, plus: sparse and sliding-window layers over one MLA geometry,
// sandwich norms, and routed experts at n_expert_latent instead of n_embd

// a token that cannot be hashed: outside the map, or an embedding (image) position
static constexpr int64_t HY_V4_FLASH_ENGRAM_DEAD = -1;

// the reference Engram norms leave eps unset, so they use the fp32 machine epsilon, not rms_norm_eps
static constexpr float HY_V4_FLASH_ENGRAM_EPS = 1.1920929e-07f;

// the weights keep the reference's interleaved rope pairs, the M-RoPE kernels need NeoX halves
// q and k get the same reorder, so their dot product does not change
static ggml_tensor * hy_v4_flash_rope_to_neox(ggml_context * ctx, ggml_tensor * x) {
    const int64_t n_rot = x->ne[0];
    x = ggml_reshape_4d(ctx, ggml_cont(ctx, x), 2, n_rot/2, x->ne[1], x->ne[2]);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    return ggml_reshape_3d(ctx, x, n_rot, x->ne[2], x->ne[3]);
}

// IMROPE rotates the last pairs with the 4th position, the reference uses t there
static ggml_tensor * hy_v4_flash_pos(ggml_context * ctx, ggml_tensor * pos, int64_t n_tokens) {
    return ggml_concat(ctx, ggml_view_1d(ctx, pos, 3*n_tokens, 0), ggml_view_1d(ctx, pos, n_tokens, 0), 0);
}

static size_t hy_v4_flash_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * hy_v4_flash_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, hy_v4_flash_elem_offset(t, i0));
}

static ggml_tensor * hy_v4_flash_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], hy_v4_flash_elem_offset(t, i0));
}

void llama_model_hy_v4_flash::load_arch_hparams(llama_model_loader & ml) {
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
    ml.get_key(LLM_KV_EXPERT_LATENT_LENGTH,        hparams.n_expert_latent,      false);

    // routed experts only; the shared and dense FFNs are not clamped
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, hparams.n_layer_all, false);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,     hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,   hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HYPER_CONNECTION_MAGNITUDE, hparams.hc_magnitude);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,       hparams.rope_freq_base_train_swa);
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    // the sections are only used by the VL path (M-RoPE), a text-only file runs the normal rope
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);
    if (hparams.use_mrope()) {
        const auto & s = hparams.rope_sections;
        if (s[0] < 0 || s[1] < 0 || s[2] < 0 || s[3] < 0 || 2 * ((int64_t) s[0] + s[1] + s[2] + s[3]) != (int64_t) hparams.n_rot()) {
            throw std::runtime_error("hy_v4_flash: rope sections must sum to n_rot / 2");
        }
    }

    // the sparse layers select their context with a pooled indexer
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K, hparams.indexer_top_k);
    {
        ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,        hparams.indexer_n_head);
        ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,        hparams.indexer_head_size);
        ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL,             hparams.indexer_kpool);
        ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, hparams.indexer_kpool_select_tail, false);
        ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer(), false);

        // indexer.types says which layers own an indexer, none means no indexer
        bool any = false;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            any = any || hparams.is_indexer_full(il);
        }

        if (!any) {
            throw std::runtime_error("hy_v4_flash: no layer owns an indexer");
        } else {
            if (hparams.indexer_kpool < 2 || hparams.indexer_top_k % hparams.indexer_kpool != 0) {
                throw std::runtime_error(format(
                            "hy_v4_flash: indexer top_k %u must be a multiple of the pool size %u",
                            hparams.indexer_top_k, hparams.indexer_kpool));
            }
            if (hparams.indexer_kpool_select_tail && hparams.indexer_top_k / hparams.indexer_kpool < 2) {
                throw std::runtime_error("hy_v4_flash: indexer top_k must reserve at least one history group and one tail group");
            }
            // the indexer cache mirrors the full-attention child, so a window layer cannot own one
            for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                if (!hparams.is_indexer_full(il)) {
                    continue;
                }
                if (hparams.is_swa(il)) {
                    throw std::runtime_error(format(
                                "hy_v4_flash: layer %u is a sliding-window layer and cannot own an indexer", il));
                }
                // the indexer splits each head into a nope front and a rope tail, so the tail must fit
                if (hparams.n_rot(il) >= hparams.indexer_head_size) {
                    throw std::runtime_error(format(
                                "hy_v4_flash: layer %u indexer key length %u must be larger than the rope dimension %u",
                                il, hparams.indexer_head_size, hparams.n_rot(il)));
                }
            }
        }
    }

    GGML_ASSERT(hparams.is_mla());

    if (!hparams.is_swa_any()) {
        throw std::runtime_error("hy_v4_flash: no sliding-window layer, expected a hybrid stack");
    }

    // both layer types share one MLA geometry, only the head count differs
    uint32_t n_embd_head_k_mla_swa = 0;
    uint32_t n_embd_head_v_mla_swa = 0;
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   n_embd_head_k_mla_swa, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, n_embd_head_v_mla_swa, false);

    if ((n_embd_head_k_mla_swa && n_embd_head_k_mla_swa != hparams.n_embd_head_k_mla()) ||
        (n_embd_head_v_mla_swa && n_embd_head_v_mla_swa != hparams.n_embd_head_v_mla())) {
        throw std::runtime_error(format(
                    "hy_v4_flash: needs one MLA geometry for both layer types, but the "
                    "sliding-window layers declare key/value %u/%u against %u/%u",
                    n_embd_head_k_mla_swa, n_embd_head_v_mla_swa,
                    hparams.n_embd_head_k_mla(), hparams.n_embd_head_v_mla()));
    }

    // MLA runs as MQA: one latent row per token
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (hparams.n_head_kv(il) != 1) {
            throw std::runtime_error(format(
                        "hy_v4_flash: layer %u has n_head_kv %u, expected 1 for MLA",
                        il, hparams.n_head_kv(il)));
        }
    }

    // Engram is optional: deepseek41 n-gram hash tables, one module per Engram layer
    // multipliers, primes and offsets are flat [module][...] arrays, so their lengths give the module count
    std::vector<uint32_t> e_layers;
    ml.get_arr(LLM_KV_PLE_LAYERS, e_layers, false);
    if (!e_layers.empty()) {
        std::vector<uint64_t> mult;
        std::vector<uint64_t> offsets;
        std::vector<uint64_t> primes;
        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,             hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM,        hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.ple_head_dim);
        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS,      mult);
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,           offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES,       primes);
        ml.get_key(LLM_KV_ENGRAM_PAD_TOKEN_ID,        hparams.engram_pad_id);
        ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,           engram_token_map);

        const size_t n_e    = e_layers.size();
        const size_t n_gram = hparams.ple_ngram_size;
        const size_t n_cols = n_gram >= 2 ? (n_gram - 1)*hparams.ple_heads_per_ngram : 0;
        if (n_cols == 0 || hparams.ple_head_dim == 0 ||
                mult.size() != n_e*n_gram || primes.size() != n_e*n_cols || offsets.size() != n_e*n_cols ||
                primes.size() > LLAMA_MAX_PLE_HEADS ||
                hparams.engram_pad_id >= engram_token_map.size()) {
            throw std::runtime_error("hy_v4_flash engram metadata is inconsistent");
        }
        for (size_t e = 0; e < n_e; ++e) {
            if (e_layers[e] >= hparams.n_layer() || (e > 0 && e_layers[e] <= e_layers[e - 1])) {
                throw std::runtime_error(format("hy_v4_flash engram layer %u is out of range or out of order", e_layers[e]));
            }
            hparams.is_ple_impl.set(e_layers[e]);
        }
        for (size_t c = 0; c < n_e*n_cols; ++c) {
            // a bucket count of 0 traps in the hash; row ids are int32
            if (primes[c] < 2 || offsets[c] > (uint64_t) INT32_MAX || primes[c] > (uint64_t) INT32_MAX - offsets[c]) {
                throw std::runtime_error(format("hy_v4_flash engram head %zu: bucket range out of range", c));
            }
            hparams.ple_head_offsets[c]     = (uint32_t) offsets[c];
            hparams.ple_head_vocab_sizes[c] = (uint32_t) primes[c];
        }
        engram_multipliers = std::move(mult);
        hparams.ple_n_heads   = (uint32_t) n_cols;
        hparams.ple_n_modules = (uint32_t) n_e;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_hy_v4_flash::load_arch_tensors(llama_model_loader &) {
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
    const int64_t n_embd_latent   = hparams.n_expert_latent ? hparams.n_expert_latent : n_embd;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc * n_embd, hc}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    const int64_t ple_n_heads = hparams.ple_n_heads;
    const int64_t ple_head_dim = hparams.ple_head_dim;
    if (ple_n_heads > 0) {
        // all modules share one table, so the row count is the end of the last bucket
        int64_t n_rows = 0;
        for (size_t c = 0; c < (size_t) hparams.ple_n_modules*ple_n_heads; ++c) {
            n_rows = std::max(n_rows, (int64_t) hparams.ple_head_offsets[c] + hparams.ple_head_vocab_sizes[c]);
        }
        // 118 GiB at the full size: read lazily, so only the rows a batch hashes to are faulted in
        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"), {ple_head_dim, n_rows}, TENSOR_READ_LAZY);
    }

    // the MTP block is a sparse layer after the trunk, without hyper-connections
    for (int i = 0; i < (int) hparams.n_layer_all; ++i) {
        auto & layer = layers[i];
        const bool is_nextn = i >= n_layer;

        // the MTP block is not supported yet, its tensors are skipped
        const int flags = is_nextn ? TENSOR_SKIP | TENSOR_NOT_REQUIRED : 0;

        // the sliding-window layers carry more heads over the same latent geometry
        const int64_t n_head_l = hparams.n_head(i);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, flags);
        layer.attn_sinks     = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,     "weight", i), {n_head_l}, flags);

        layer.wq_a           = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,       "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, flags);
        layer.wq_b           = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,       "weight", i), {q_lora_rank, n_head_l * n_embd_head_k_mla}, flags);
        layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);
        layer.wk_b           = create_tensor(tn(LLM_TENSOR_ATTN_K_B,       "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head_l}, flags);
        layer.wv_b           = create_tensor(tn(LLM_TENSOR_ATTN_V_B,       "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head_l}, flags);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_head_l * n_embd_head_v_mla, n_embd}, flags);
        layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", i), {n_embd, n_head_l * n_embd_head_v_mla}, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, flags);

        if (!is_nextn) {
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc * n_embd, 2 * hc}, flags);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {2 * hc}, flags);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {2}, flags);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc * n_embd, 2 * hc}, flags);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {2 * hc}, flags);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {2}, flags);
        } else {
            // the draft block folds the next token's embedding into the trunk's final hidden state
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, flags);
        }

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, flags);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, flags);

        // is_indexer_full() only covers the trunk, and the draft block always owns an indexer
        if (is_nextn || hparams.is_indexer_full(i)) {
            const int64_t n_idx      = hparams.indexer_head_size;
            const int64_t n_idx_head = hparams.indexer_n_head;
            const int64_t kpool      = hparams.indexer_kpool;

            layer.indexer_attn_k     = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,      "weight", i), {n_embd, n_idx}, flags);
            layer.indexer_k_norm     = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,      "weight", i), {n_idx}, flags);
            // the pool averages a projection of the normed key, weighted by another projection of it
            layer.indexer_kpool_value = create_tensor(tn(LLM_TENSOR_INDEXER_KPOOL_VALUE, "weight", i), {n_idx, n_idx}, flags);
            layer.indexer_kpool_gate  = create_tensor(tn(LLM_TENSOR_INDEXER_KPOOL_GATE,  "weight", i), {n_idx, n_idx}, flags);
            layer.indexer_kpool_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_KPOOL_APE,   "weight", i), {n_idx, kpool}, flags);
            layer.indexer_comp_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM, "weight", i), {n_idx}, flags);
            layer.indexer_attn_q_b   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B,    "weight", i), {q_lora_rank, n_idx*n_idx_head}, flags);
            layer.indexer_proj       = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,        "weight", i), {n_embd, n_idx_head}, flags);
        }

        if (!is_nextn && engram_index(i) >= 0) {
            // hc key projections and the value projection fused into one matmul, as deepseek41, plus a bias
            layer.engram_kv          = create_tensor(tn(LLM_TENSOR_ENGRAM_KV,          "weight", i), {ple_n_heads * ple_head_dim, n_embd * (hc + 1)}, flags);
            layer.engram_kv_b        = create_tensor(tn(LLM_TENSOR_ENGRAM_KV_B,        "weight", i), {n_embd, hc + 1}, flags);
            layer.engram_gate        = create_tensor(tn(LLM_TENSOR_ENGRAM_GATE,        "weight", i), {n_embd, hc}, flags);
            layer.engram_lookup_norm = create_tensor(tn(LLM_TENSOR_ENGRAM_LOOKUP_NORM, "weight", i), {ple_head_dim, ple_n_heads}, flags);
            layer.engram_pre_gamma   = create_tensor(tn(LLM_TENSOR_ENGRAM_PRE_GAMMA,   "weight", i), {n_embd}, flags);
            layer.engram_hc_scale    = create_tensor(tn(LLM_TENSOR_ENGRAM_HC_SCALE,    "weight", i), {n_embd, hc}, flags);
        }

        if (!is_nextn && i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        } else {
            if (n_expert == 0 || n_expert_used == 0) {
                throw std::runtime_error("hy_v4_flash: n_expert and n_expert_used must be > 0");
            }

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED | flags);

            if (hparams.n_expert_latent) {
                layer.ffn_routed_down = create_tensor(tn(LLM_TENSOR_FFN_ROUTED_DOWN, "weight", i), {n_embd, n_embd_latent}, flags);
                layer.ffn_routed_up   = create_tensor(tn(LLM_TENSOR_FFN_ROUTED_UP,   "weight", i), {n_embd_latent, n_embd}, flags);
            }

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd_latent, n_ff_exp,      n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd_latent, n_ff_exp,      n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,      n_embd_latent, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }
    }
}

int llama_model_hy_v4_flash::engram_index(int il) const {
    if (hparams.ple_n_heads == 0 || !hparams.is_ple(il)) {
        return -1;
    }
    int e = 0;
    for (int j = 0; j < il; ++j) {
        e += hparams.is_ple(j);
    }
    return e;
}

void llama_model_hy_v4_flash::engram_hash(int e, const llama_ubatch & ubatch,
        const std::vector<llama_token> & prev,
        std::vector<int32_t> & ids, std::vector<float> & valid) const {
    const int64_t n_gram = hparams.ple_ngram_size;
    const int64_t n_prev = n_gram - 1;
    const int64_t n_head = hparams.ple_heads_per_ngram;
    const int64_t n_cols = hparams.ple_n_heads;

    const uint64_t * mult   = engram_multipliers.data()           + (size_t) e*n_gram;
    const uint32_t * primes = hparams.ple_head_vocab_sizes.data()  + (size_t) e*n_cols;
    const uint32_t * offs   = hparams.ple_head_offsets.data()      + (size_t) e*n_cols;
    const int64_t    pad    = engram_token_map[hparams.engram_pad_id];

    const auto cid = [&](llama_token t) -> int64_t {
        return t < 0 || t >= (llama_token) engram_token_map.size() ? HY_V4_FLASH_ENGRAM_DEAD : engram_token_map[t];
    };

    ids.resize(ubatch.n_tokens*n_cols);
    valid.resize(ubatch.n_tokens*n_cols);

    std::vector<uint64_t> toks(n_gram);

    for (int64_t i = 0; i < ubatch.n_tokens; ++i) {
        // a short history is padded with the pad id, only an embedding position is invalid
        const bool is_embd = !ubatch.token || (ubatch.is_mixed() && ubatch.type[i]);
        int64_t c = is_embd ? HY_V4_FLASH_ENGRAM_DEAD : cid(ubatch.token[i]);
        const bool dead = c == HY_V4_FLASH_ENGRAM_DEAD;
        toks[0] = dead ? pad : c;
        for (int64_t s = 1; s < n_gram; ++s) {
            c = ubatch.pos[i] - s < 0 ? HY_V4_FLASH_ENGRAM_DEAD : cid(prev[i*n_prev + (n_prev - s)]);
            toks[s] = c == HY_V4_FLASH_ENGRAM_DEAD ? pad : c;
        }

        uint64_t rolling = toks[0]*mult[0];
        for (int64_t o = 1; o < n_gram; ++o) {
            rolling ^= toks[o]*mult[o];
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t col = (o - 1)*n_head + h;
                ids  [i*n_cols + col] = (int32_t) (rolling % primes[col] + offs[col]);
                valid[i*n_cols + col] = dead ? 0.0f : 1.0f;
            }
        }
    }
}

// the row ids and their validity are host-side, so they are graph inputs rather than computed
class llm_graph_input_hy_v4_flash : public llm_graph_input_i {
public:
    llm_graph_input_hy_v4_flash(const llama_model_hy_v4_flash & model,
            const llama_kv_cache_context * mctx) : model(model), mctx(mctx) {}

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::vector<ggml_tensor *> engram_ids;
    std::vector<ggml_tensor *> engram_valid;

    const llama_model_hy_v4_flash & model;
    const llama_kv_cache_context * mctx;
};

void llm_graph_input_hy_v4_flash::set_input(const llama_ubatch * ubatch) {
    if (engram_ids.empty()) {
        return;
    }

    // a token shared by several sequences has the same history in each, so its first one resolves it
    std::vector<llama_token> prev;
    mctx->get_prev_tokens(*ubatch, model.hparams.ple_ngram_size - 1, prev);

    // hash every module before the first gather, so a lazily read table gets one batched readahead
    std::vector<std::vector<int32_t>> ids(engram_ids.size());
    std::vector<std::vector<float>>   valid(engram_ids.size());
    std::vector<int32_t> all;
    for (size_t e = 0; e < engram_ids.size(); ++e) {
        model.engram_hash((int) e, *ubatch, prev, ids[e], valid[e]);
        all.insert(all.end(), ids[e].begin(), ids[e].end());
    }
    if (model.can_prefetch.count(model.per_layer_tok_embd)) {
        llama_prefetch_rows(model.per_layer_tok_embd, all.data(), all.size());
    }
    for (size_t e = 0; e < engram_ids.size(); ++e) {
        if (engram_ids[e]->buffer) {
            ggml_backend_tensor_set(engram_ids[e], ids[e].data(), 0, ids[e].size()*sizeof(int32_t));
        }
        if (engram_valid[e]->buffer) {
            ggml_backend_tensor_set(engram_valid[e], valid[e].data(), 0, valid[e].size()*sizeof(float));
        }
    }
}

bool llm_graph_input_hy_v4_flash::can_reuse(const llm_graph_params & params) {
    // the memory context is rebuilt for every batch, re-fetch it as deepseek41 does
    mctx = static_cast<const llama_kv_cache_dsa_iswa_context *>(params.mctx)->get_dsa()->get_mla();

    bool res = true;
    for (auto * t : engram_ids) {
        res &= t->ne[0] == (int64_t) (model.hparams.ple_n_heads*params.ubatch.n_tokens);
    }
    return res;
}

std::unique_ptr<llm_graph_context> llama_model_hy_v4_flash::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("hy_v4_flash: the MTP block is not supported yet");
    }
    return std::make_unique<graph>(*this, params);
}

ggml_tensor * llama_model_hy_v4_flash::graph::build_engram(
        const llama_model & model,
        ggml_tensor * x,
        ggml_tensor * rows,
        ggml_tensor * valid,
        int il) const {
    const auto & layer = model.layers[il];
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];
    const int64_t n_cols   = hparams.ple_n_heads;
    const int64_t head_dim = hparams.ple_head_dim;

    GGML_ASSERT(rows->ne[0] == head_dim && rows->ne[1] == n_cols && rows->ne[2] == nt);

    // unlike deepseek41, each looked-up row is RMS-normed and scaled by its head weight
    ggml_tensor * r = ggml_rms_norm(ctx0, rows, HY_V4_FLASH_ENGRAM_EPS);
    r = ggml_mul(ctx0, r, layer.engram_lookup_norm);
    r = ggml_mul(ctx0, r, ggml_reshape_3d(ctx0, valid, 1, n_cols, nt));
    r = ggml_reshape_2d(ctx0, r, head_dim*n_cols, nt);
    cb(r, "engram_rows_normed", il);

    // the fused key/value projection has a bias
    ggml_tensor * kv = build_lora_mm(layer.engram_kv, r);
    kv = ggml_add(ctx0, ggml_reshape_3d(ctx0, kv, n_embd, hc + 1, nt), layer.engram_kv_b);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, hc, nt, kv->nb[1], kv->nb[2], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, 1,  nt, kv->nb[1], kv->nb[2], hc*kv->nb[1]));

    // the gate is deepseek41's: engram_gate folds norm_key.w * norm_query.w
    ggml_tensor * egate = ggml_is_quantized(layer.engram_gate->type) ? ggml_cast(ctx0, layer.engram_gate, GGML_TYPE_F32) : layer.engram_gate;
    ggml_tensor * dot = ggml_mul(ctx0, ggml_mul(ctx0, ggml_rms_norm(ctx0, x, HY_V4_FLASH_ENGRAM_EPS), ggml_rms_norm(ctx0, key, HY_V4_FLASH_ENGRAM_EPS)), egate);
    dot = ggml_scale(ctx0, ggml_sum_rows(ctx0, dot), 1.0f/sqrtf((float) n_embd));
    cb(dot, "engram_dot", il);

    // sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)); sign(0) = +1 as in copysign
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_neg(ctx0, dot)), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, mag, sign));
    cb(gate, "engram_gate", il);

    // the delta is the normed value scaled by a learned gamma
    ggml_tensor * signal = ggml_mul(ctx0, ggml_rms_norm(ctx0, value, HY_V4_FLASH_ENGRAM_EPS), layer.engram_pre_gamma);
    // the per-hc offset is additive and per channel
    ggml_tensor * scale = ggml_add(ctx0, ggml_repeat_4d(ctx0, gate, n_embd, hc, nt, 1), layer.engram_hc_scale);
    cb(scale, "engram_scale", il);

    ggml_tensor * delta = ggml_mul(ctx0, ggml_repeat_4d(ctx0, signal, n_embd, hc, nt, 1), scale);

    // mask the whole delta of a token without a valid n-gram, as the bias still reaches the value path
    ggml_tensor * any = ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, valid, n_cols, nt));
    any = ggml_step(ctx0, any);
    delta = ggml_mul(ctx0, delta, ggml_reshape_3d(ctx0, any, 1, 1, nt));
    cb(delta, "engram_delta", il);

    return ggml_add(ctx0, x, delta);
}

// reduce hc streams x[:,i,:] weighted by w[i,:] -> [n_embd, n_tokens]
ggml_tensor * llama_model_hy_v4_flash::graph::build_hc_reduce(
        ggml_tensor * x,
        ggml_tensor * weights,
        int il) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hc);

    // the fused node is registered per layer, so the head reduce (il < 0) keeps the loop
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih * x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih * weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }
    return result;
}

llama_model_hy_v4_flash::graph::hc_pre
llama_model_hy_v4_flash::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        int il) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];
    GGML_ASSERT(x->ne[0] == n_embd && x->ne[1] == hc);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc * n_embd, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = hy_v4_flash_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = hy_v4_flash_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * base_pre   = hy_v4_flash_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post  = hy_v4_flash_view_1d(ctx0, hc_base, hc, hc);

    ggml_tensor * pre = hy_v4_flash_view_2d(ctx0, mixes, hc, nt, 0);
    pre = ggml_mul(ctx0, pre, scale_pre);
    pre = ggml_add(ctx0, pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    ggml_tensor * post = hy_v4_flash_view_2d(ctx0, mixes, hc, nt, hc);
    post = ggml_mul(ctx0, post, scale_post);
    post = ggml_add(ctx0, post, base_post);
    post = ggml_sigmoid(ctx0, post);
    post = ggml_scale(ctx0, post, hparams.hc_magnitude);
    post = ggml_scale_bias(ctx0, post, 1.0f, hparams.dsv4_hc_eps);
    cb(post, "hc_post_gate", il);

    return { build_hc_reduce(x, pre, il), post };
}

ggml_tensor * llama_model_hy_v4_flash::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        int il) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hc);

    // comb is the identity here: each stream keeps its own residual, hy_v4 has no cross-stream mix
    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, x, residual, post, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    ggml_tensor * out = nullptr;
    for (int64_t i = 0; i < hc; ++i) {
        ggml_tensor * res_i  = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], i * residual->nb[1]);
        ggml_tensor * post_i = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], i * post->nb[0]);
        ggml_tensor * cur = ggml_add(ctx0, res_i, ggml_mul(ctx0, x, post_i));
        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llama_model_hy_v4_flash::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc * n_embd, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = ggml_mul(ctx0, mixes, hc_scale);
    pre = ggml_add(ctx0, pre, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_reduce(x, pre, -1);
}

// the pools of the indexer, rebuilt from the cells of the indexer cache for every ubatch
class llama_model_hy_v4_flash::llm_graph_input_pool : public llm_graph_input_i {
public:
    llm_graph_input_pool(const llama_kv_cache_context * mctx, uint32_t kpool) : mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_pool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->set_input_kpool(pool_idxs, pool_pos, pool_mask, tail_idxs, ubatch, kpool);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsa_iswa_context *>(params.mctx)->get_dsa()->get_lid();

        bool res = true;
        res &= pool_idxs->ne[1] == n_pool(mctx, params.cparams.n_seq_max, kpool);
        res &= tail_idxs->ne[1] == params.ubatch.n_tokens;
        return res;
    }

    // every cell can start a pool, plus one misaligned pool per sequence
    static int64_t n_pool(const llama_kv_cache_context * mctx, uint32_t n_seq_max, uint32_t kpool) {
        return mctx->get_n_kv()/kpool + n_seq_max;
    }

    ggml_tensor * pool_idxs = nullptr; // I32 [kpool, n_pool]
    ggml_tensor * pool_pos  = nullptr; // I32 [n_pool*n_pos_per_embd]
    ggml_tensor * pool_mask = nullptr; // F16 [n_pool, n_tokens]
    ggml_tensor * tail_idxs = nullptr; // I32 [kpool, n_tokens]

    const llama_kv_cache_context * mctx;
    const uint32_t kpool;
};

llama_model_hy_v4_flash::llm_graph_input_pool * llama_model_hy_v4_flash::graph::build_inp_pool(
        llm_graph_input_attn_k_dsa_iswa * inp_attn) {
    const auto * mctx_lid = inp_attn->get_dsa()->mctx->get_lid();

    const uint32_t kpool  = hparams.indexer_kpool;
    const int64_t  n_pool = llm_graph_input_pool::n_pool(mctx_lid, cparams.n_seq_max, kpool);

    auto inp = std::make_unique<llm_graph_input_pool>(mctx_lid, kpool);

    inp->pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_pool);
    inp->pool_pos  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool*hparams.n_pos_per_embd());
    inp->pool_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_pool, n_tokens);
    inp->tail_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_tokens);

    ggml_set_input(inp->pool_idxs);
    ggml_set_input(inp->pool_pos);
    ggml_set_input(inp->pool_mask);
    ggml_set_input(inp->tail_idxs);

    return (llm_graph_input_pool *) res->add_input(std::move(inp));
}

// rotate only the trailing n_rot of each indexer head, leaving the head's front half untouched
ggml_tensor * llama_model_hy_v4_flash::graph::build_indexer_rope(
        ggml_tensor * x, ggml_tensor * pos, int64_t n_idx, float freq_base, int il) const {
    const int64_t n_nope = n_idx - n_rot;
    GGML_ASSERT(n_nope > 0);

    ggml_tensor * front = ggml_view_3d(ctx0, x, n_nope, x->ne[1], x->ne[2],
            x->nb[1], x->nb[2], 0);
    ggml_tensor * back  = ggml_view_3d(ctx0, x, n_rot, x->ne[1], x->ne[2],
            x->nb[1], x->nb[2], ggml_row_size(x->type, n_nope));

    // the same base the layer's own attention uses, so a cparams override reaches the indexer too
    if (hparams.use_mrope()) {
        int sections[GGML_MROPE_SECTIONS];
        std::copy(hparams.rope_sections.begin(), hparams.rope_sections.end(), sections);

        back = ggml_rope_multi(ctx0, hy_v4_flash_rope_to_neox(ctx0, back), pos, nullptr, n_rot, sections, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    } else {
        back = ggml_rope_ext(ctx0, ggml_cont(ctx0, back), pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }
    cb(back, "indexer_rope", il);

    return ggml_concat(ctx0, ggml_cont(ctx0, front), back, 0);
}

// the cells a token attends to: the best scored pools of its sequence plus its own pool
ggml_tensor * llama_model_hy_v4_flash::graph::build_pool_select(
        ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * inp_pos, float freq_base,
        const llama_layer & layer,
        llm_graph_input_attn_k_dsa_iswa * inp_attn, llm_graph_input_pool * inp_pool,
        ggml_tensor * kq_mask, int il) {
    const auto * inp_dsa  = inp_attn->get_dsa();
    const auto * mctx_lid = inp_dsa->mctx->get_lid();

    const int64_t n_idx_head = hparams.indexer_n_head;
    const int64_t n_idx      = hparams.indexer_head_size;
    const int64_t kpool      = hparams.indexer_kpool;
    const int64_t n_pool     = inp_pool->pool_idxs->ne[1];
    const int64_t n_kv       = mctx_lid->get_n_kv();

    // the query comes from the q lora residual of the main attention
    ggml_tensor * iq = build_lora_mm(layer.indexer_attn_q_b, qr);
    iq = ggml_reshape_3d(ctx0, iq, n_idx, n_idx_head, n_tokens);
    iq = build_indexer_rope(iq, inp_pos, n_idx, freq_base, il);
    cb(iq, "indexer_q", il);

    // key side: weight-only RMSNorm, unlike glm5-next's LayerNorm with a bias this model lacks
    ggml_tensor * ik = build_lora_mm(layer.indexer_attn_k, cur);
    ik = build_norm(ik, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(ik, "indexer_k", il);

    // the value and the gate are linear in the token's key, so the cache holds them and the pools are built from them
    ggml_tensor * iv = build_lora_mm(layer.indexer_kpool_value, ik);
    ggml_tensor * ig = build_lora_mm(layer.indexer_kpool_gate, ik);
    cb(ig, "indexer_gate", il);

    ggml_tensor * packed = ggml_reshape_3d(ctx0, ggml_concat(ctx0, iv, ig, 0), 2*n_idx, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, packed, inp_dsa->get_k_idxs_lid(), il));

    // the members of every pool, from the cache this ubatch just wrote to
    ggml_tensor * k_lid = mctx_lid->get_k(ctx0, il);
    GGML_ASSERT(k_lid->ne[0] == 2*n_idx && k_lid->ne[1] == 1 && k_lid->ne[2] == n_kv && k_lid->ne[3] == 1);
    k_lid = ggml_view_2d(ctx0, k_lid, 2*n_idx, n_kv, k_lid->nb[2], 0);

    ggml_tensor * rows = ggml_get_rows(ctx0, k_lid, ggml_reshape_1d(ctx0, inp_pool->pool_idxs, kpool*n_pool));
    rows = ggml_reshape_3d(ctx0, rows, 2*n_idx, kpool, n_pool);

    ggml_tensor * pv = ggml_view_3d(ctx0, rows, n_idx, kpool, n_pool, rows->nb[1], rows->nb[2], 0);
    ggml_tensor * pg = ggml_view_3d(ctx0, rows, n_idx, kpool, n_pool, rows->nb[1], rows->nb[2],
            ggml_row_size(rows->type, n_idx));

    // per channel softmax over the members of the gate plus the member's position bias, then the weighted sum of the values
    ggml_tensor * logits = ggml_add(ctx0, pg, layer.indexer_kpool_ape);
    logits = ggml_cont(ctx0, ggml_permute(ctx0, logits, 1, 0, 2, 3)); // [kpool, n_idx, n_pool]
    // fold the pool axis into rows: soft_max maps ne[2] to a CUDA gridDim.y that caps at 65535
    ggml_tensor * probs = ggml_soft_max(ctx0, ggml_reshape_2d(ctx0, logits, kpool, n_idx*n_pool));
    probs = ggml_reshape_3d(ctx0, probs, kpool, n_idx, n_pool);

    pv = ggml_cont(ctx0, ggml_permute(ctx0, pv, 1, 0, 2, 3));
    ggml_tensor * pooled = ggml_sum_rows(ctx0, ggml_mul(ctx0, probs, pv)); // [1, n_idx, n_pool]
    pooled = ggml_reshape_2d(ctx0, pooled, n_idx, n_pool);

    // the pooled key is normed, then rotated at the position of its first member
    pooled = build_norm(pooled, layer.indexer_comp_norm, nullptr, LLM_NORM_RMS, il);
    pooled = build_indexer_rope(ggml_reshape_3d(ctx0, pooled, n_idx, 1, n_pool), inp_pool->pool_pos, n_idx, freq_base, il);
    cb(pooled, "indexer_pool_k", il);

    ggml_tensor * weights = build_lora_mm(layer.indexer_proj, cur);
    weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(n_idx * n_idx_head)));
    cb(weights, "indexer_weights", il);

    ggml_tensor * score = ggml_lightning_indexer(ctx0, iq, pooled, weights, inp_pool->pool_mask);
    cb(score, "indexer_score", il);

    // the token's own pool is always kept, so it does not take a pick
    const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool - hparams.indexer_kpool_select_tail);
    ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool); // [n_top_pool, n_tokens], unordered

    ggml_tensor * tail = hparams.indexer_kpool_select_tail ? inp_pool->tail_idxs : nullptr;

    if (cparams.fused_top_k_mask) {
        // a pick scored -inf (another sequence, a later or unused pool) is dropped
        ggml_tensor * sel = ggml_top_k_mask(ctx0, kq_mask, top_k, score, inp_pool->pool_idxs, tail);
        res->add_fused_node({LLM_FUSED_OP_TOP_K_MASK, sel, il});
        cb(sel, "indexer_sel", il);

        return sel;
    }

    // without the fused op, the picks are expanded to their cells
    // a pick scored -inf is only taken once every visible pool is, so its cells are either hidden by the kq_mask or already kept
    ggml_tensor * cells = ggml_get_rows(ctx0, inp_pool->pool_idxs,
            ggml_reshape_1d(ctx0, ggml_cont(ctx0, top_k), n_top_pool*n_tokens)); // [kpool, n_top_pool*n_tokens]
    cells = ggml_reshape_2d(ctx0, cells, kpool*n_top_pool, n_tokens);
    if (tail) {
        cells = ggml_concat(ctx0, cells, tail, 0);
    }

    ggml_tensor * sel = build_top_k_mask(kq_mask, cells, il);
    cb(sel, "indexer_sel", il);

    return sel;
}

ggml_tensor * llama_model_hy_v4_flash::graph::build_attention(
        const llama_model & model,
        llm_graph_input_attn_k_dsa_iswa * inp_attn,
        llm_graph_input_pool * inp_pool,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) {
    const auto & layer = model.layers[il];

    const int64_t n_head_l            = hparams.n_head(il);
    const int64_t n_embd_head_k       = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const uint32_t kv_lora_rank       = hparams.n_lora_kv;

    const float kq_scale    = 1.0f / sqrtf(float(n_embd_head_k));
    const float freq_base_l = model.get_rope_freq_base(cparams, il);

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "q_resid", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head_l, n_tokens,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head_l, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head_l, n_tokens,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head_l,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv_cmpr_pe = build_lora_mm(layer.wkv_a_mqa, cur);
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    if (hparams.use_mrope()) {
        q_pe = hy_v4_flash_rope_to_neox(ctx0, q_pe);
        k_pe = hy_v4_flash_rope_to_neox(ctx0, k_pe);
    }
    q_pe = build_rope(q_pe, inp_pos, il);
    cb(q_pe, "q_pe", il);
    k_pe = build_rope(k_pe, inp_pos, il);
    cb(k_pe, "k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "kv_cmpr", il);

    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

    // rope goes last so build_rope_shift() can shift the cache in place
    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    ggml_tensor * Vcur = kv_cmpr;

    ggml_tensor * attn;

    if (hparams.is_swa(il)) {
        attn = build_attn(inp_attn->get_swa(),
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, layer.wv_b, kq_scale, il);
    } else {
        // as build_attn of the DSA input, with the cells of the picked pools instead of a top-k of cells
        auto * inp_dsa = inp_attn->get_dsa();
        const auto * mctx_mla = inp_dsa->mctx->get_mla();

        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Vcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, mctx_mla->cpy_k(ctx0, Kcur, inp_dsa->get_k_idxs_mla(), il));

        ggml_tensor * sel = build_pool_select(cur, qr, inp_pos, freq_base_l, layer,
                inp_attn, inp_pool, inp_dsa->self_attn_mask_mla.raw, il);

        const int64_t n_sel = hparams.indexer_top_k;

        ggml_tensor * k = mctx_mla->get_k(ctx0, il);
        ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

        attn = build_attn_mha(Qcur, k, v, nullptr, sel, nullptr, layer.attn_sinks, layer.wv_b, n_sel, kq_scale, il);
    }
    cb(attn, "attn_kqv", il);

    ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, cur);
    gate = ggml_sigmoid(ctx0, gate);
    attn = ggml_mul(ctx0, attn, gate);
    cb(attn, "attn_gated", il);

    ggml_tensor * out = build_lora_mm(layer.wo, attn);
    out = build_norm(out, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(out, "attn_out", il);

    return out;
}

ggml_tensor * llama_model_hy_v4_flash::graph::build_latent_moe(
        const llama_model & model,
        ggml_tensor * cur,
        int il) const {
    const auto & layer = model.layers[il];

    // the router scores the full-width input, the experts take the latent one
    ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
    cb(logits, "ffn_moe_logits", il);

    ggml_tensor * routed_in = layer.ffn_routed_down
        ? build_lora_mm(layer.ffn_routed_down, cur)
        : cur;

    ggml_tensor * moe_out = build_moe_ffn(routed_in,
            nullptr,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            logits);
    cb(moe_out, "ffn_moe_out", il);

    if (layer.ffn_routed_up) {
        moe_out = build_lora_mm(layer.ffn_routed_up, moe_out);
        cb(moe_out, "ffn_moe_up", il);
    }

    ggml_tensor * shexp = build_ffn(cur,
            layer.ffn_up_shexp,   NULL, NULL,
            layer.ffn_gate_shexp, NULL, NULL,
            layer.ffn_down_shexp, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(shexp, "ffn_shexp", il);

    return ggml_add(ctx0, moe_out, shexp);
}

llama_model_hy_v4_flash::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t hc = hparams.dsv4_hc_mult;

    ggml_tensor * cur;

    ggml_tensor * inp_pos = build_inp_pos();
    if (hparams.use_mrope()) {
        inp_pos = hy_v4_flash_pos(ctx0, inp_pos, n_tokens);
    }

    // the sparse layers are in the DSA cache with their indexer, the sliding-window layers in the SWA cache
    llm_graph_input_attn_k_dsa_iswa * inp_attn = build_attn_inp_k_dsa_iswa();

    llm_graph_input_pool * inp_pool = build_inp_pool(inp_attn);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Engram gathers go before the token embeddings, so the mixed embd copies after them land on the GPU
    // and the host split keeps only small single threaded lookups
    auto inp_eng = std::make_unique<llm_graph_input_hy_v4_flash>(
            static_cast<const llama_model_hy_v4_flash &>(model), inp_attn->get_dsa()->mctx->get_mla());
    std::vector<ggml_tensor *> engram_rows;
    std::vector<ggml_tensor *> engram_valid;
    for (int il = 0; ubatch.token && hparams.ple_n_heads > 0 && il < n_layer; ++il) {
        if (!hparams.is_ple(il)) {
            continue;
        }
        const int64_t n_cols = hparams.ple_n_heads;

        ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cols*n_tokens);
        ggml_set_input(ids);
        ggml_set_name(ids, ("engram_ids-" + std::to_string(il)).c_str());
        inp_eng->engram_ids.push_back(ids);

        ggml_tensor * valid = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_cols*n_tokens);
        ggml_set_input(valid);
        ggml_set_name(valid, ("engram_valid-" + std::to_string(il)).c_str());
        inp_eng->engram_valid.push_back(valid);
        engram_valid.push_back(valid);

        ggml_tensor * rows = ggml_get_rows(ctx0, model.per_layer_tok_embd, ids);
        rows = ggml_reshape_3d(ctx0, rows, hparams.ple_head_dim, n_cols, n_tokens);
        cb(rows, "engram_rows", il);
        ggml_build_forward_expand(gf, rows);
        engram_rows.push_back(rows);
    }
    res->add_input(std::move(inp_eng));

    ggml_tensor * inp = build_inp_embd(model.tok_embd);

    // one embedding feeds hc parallel residual streams
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // Engram writes into the streams before the attention pre-mix reads them
        const int e = static_cast<const llama_model_hy_v4_flash &>(model).engram_index(il);
        if (e >= 0 && !engram_rows.empty()) {
            inpL = build_engram(model, inpL, engram_rows[e], engram_valid[e], il);
            cb(inpL, "engram_out", il);
        }

        ggml_tensor * residual = inpL;
        hc_pre attn_hc = build_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
        cur = build_norm(attn_hc.cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_attn, inp_pool, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, attn_hc.post, il);
        cb(inpL, "hc_attn_out", il);

        residual = inpL;
        hc_pre ffn_hc = build_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        cur = build_norm(ffn_hc.cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
            cur = build_latent_moe(model, cur, il);
        }

        cur = build_norm(cur, layer.ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, ffn_hc.post, il);
        cb(inpL, "l_out", il);
    }

    // prune the output rows after all streams, or after the nextn state when it needs every row
    if (crop_before_nextn(inp_out_ids)) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // the draft block reads this state through its hnorm
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

