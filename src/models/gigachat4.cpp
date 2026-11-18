#include "models.h"
#include "llama-impl.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid-idx.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

void llama_model_gigachat4::load_arch_hparams(llama_model_loader & ml) {
    // the MTP block takes the trunk's output rows as its hidden state
    if (hparams.n_embd_out() != hparams.n_embd) {
        throw std::runtime_error("GigaChat4: the output embedding length must equal the embedding length");
    }
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    if (!(std::isfinite(hparams.f_norm_rms_eps) && hparams.f_norm_rms_eps > 0.0f)) {
        throw std::runtime_error("GigaChat4: RMS norm epsilon must be finite and positive");
    }
    ml.get_key(LLM_KV_NORM_GATE_RANK, hparams.n_norm_gate_rank);
    if (hparams.n_norm_gate_rank == 0) {
        throw std::runtime_error("GigaChat4: norm gate rank must be positive");
    }
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);

    if (hparams.n_layer_nextn >= hparams.n_layer_all) {
        throw std::runtime_error("GigaChat4: nextn_predict_layers must be smaller than block_count");
    }
    if (hparams.n_layer_dense_lead > hparams.n_layer()) {
        throw std::runtime_error("GigaChat4: leading_dense_block_count exceeds the layer count");
    }

    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl);
    if (hparams.n_embd_head_k_mla_impl == 0 || hparams.n_embd_head_v_mla_impl == 0) {
        throw std::runtime_error("GigaChat4: MLA key and value lengths must be positive");
    }
    if (hparams.n_embd_head_k() != uint64_t(hparams.n_lora_kv) + hparams.n_rot()) {
        throw std::runtime_error("GigaChat4: cache key length must equal KV LoRA rank plus rotary dimensions");
    }
    if (hparams.n_rot() % 2 != 0) {
        throw std::runtime_error("GigaChat4: rotary dimensions must be even");
    }

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
    if (!(std::isfinite(hparams.expert_weights_scale) && hparams.expert_weights_scale > 0.0f)) {
        throw std::runtime_error("GigaChat4: expert weights scale must be finite and positive");
    }
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID) {
        throw std::runtime_error("GigaChat4: the expert gating function must be sigmoid");
    }
    for (uint32_t il = hparams.n_layer_dense_lead; il < hparams.n_layer(); ++il) {
        if (hparams.n_expert_used(il) == 0 || hparams.n_expert_used(il) > hparams.n_expert) {
            throw std::runtime_error("GigaChat4: active expert count must be positive and not exceed expert count");
        }
    }

    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,   hparams.swiglu_clamp_exp,   hparams.n_layer_all);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    if (hparams.ssm_n_group == 0 || hparams.ssm_dt_rank == 0 || hparams.ssm_dt_rank % hparams.ssm_n_group != 0) {
        throw std::runtime_error("GigaChat4: SSM head counts must be positive and value heads divisible by key heads");
    }
    if (hparams.ssm_d_conv < 2) {
        throw std::runtime_error("GigaChat4: SSM convolution kernel must be at least 2");
    }
    if (hparams.ssm_d_state == 0 || hparams.ssm_d_inner == 0) {
        throw std::runtime_error("GigaChat4: SSM dimensions must be positive");
    }
    if (hparams.ssm_d_inner != uint64_t(hparams.ssm_d_state) * hparams.ssm_dt_rank) {
        throw std::runtime_error("GigaChat4: SSM inner size must match the value head dimensions");
    }
    const uint64_t conv_dim = hparams.ssm_d_inner + 2 * uint64_t(hparams.ssm_n_group) * hparams.ssm_d_state;
    if (conv_dim > INT32_MAX || (hparams.ssm_d_conv - 1) * conv_dim > INT32_MAX ||
        uint64_t(hparams.ssm_d_state) * hparams.ssm_d_inner > INT32_MAX) {
        throw std::runtime_error("GigaChat4: SSM cache dimensions exceed int32 range");
    }

    ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all);
    bool has_recr = false;
    bool has_attn = false;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        has_recr = has_recr || hparams.is_recr(il);
        has_attn = has_attn || !hparams.is_recr(il);
    }
    if (!has_recr || !has_attn) {
        throw std::runtime_error("GigaChat4: the main layers must include both recurrent and attention layers");
    }

    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping);
    if (!(std::isfinite(hparams.f_final_logit_softcapping) && hparams.f_final_logit_softcapping >= 0.0f)) {
        throw std::runtime_error("GigaChat4: final logit softcapping must be finite and not negative");
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer());
    // all zero on a dense model: the tensors follow top_k, the memory type follows the key length
    const bool has_indexer = hparams.indexer_top_k > 0;
    if (has_indexer != (hparams.indexer_n_head > 0) || has_indexer != (hparams.indexer_head_size > 0) ||
            (has_indexer && (hparams.indexer_head_size < hparams.n_rot() || hparams.indexer_head_size % 2 != 0)) ||
            uint64_t(hparams.indexer_n_head) * hparams.indexer_head_size > INT32_MAX) {
        throw std::runtime_error("GigaChat4: invalid indexer dimensions");
    }

    // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
    // cancel the factor from the convert script
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul);
    hparams.rope_yarn_log_mul /= 0.1f;

    switch (hparams.n_layer()) {
        case 32: type = LLM_TYPE_37B_A4B; break;
        case 64: type = LLM_TYPE_708B_A41B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gigachat4::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    const int64_t norm_gate_rank = hparams.n_norm_gate_rank;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm           = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,           "weight"), { n_embd }, 0);
    output_norm_gate_up   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM_GATE_UP,   "weight"), { n_embd, norm_gate_rank }, 0);
    output_norm_gate_down = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM_GATE_DOWN, "weight"), { norm_gate_rank, n_embd }, 0);
    // try to load output.weight, if not found, use token_embd (tied embeddings)
    output                = create_tensor(tn(LLM_TENSOR_OUTPUT,                "weight"), { n_embd, n_vocab }, trunk_flags | TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const int64_t n_ff_exp   = hparams.n_ff_exp();
    const int64_t n_ff_shexp = hparams.n_ff_shexp;
    if (n_ff_exp <= 0) {
        throw std::runtime_error("GigaChat4: expert feed-forward length must be positive");
    }
    if (n_ff_shexp <= 0) {
        throw std::runtime_error("GigaChat4: shared expert feed-forward length must be positive");
    }

    const int64_t n_embd_head_k_mla      = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla      = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope    = hparams.n_rot();
    const int64_t n_embd_head_qk_nope    = n_embd_head_k_mla - n_embd_head_qk_rope;
    const int64_t q_lora_rank            = hparams.n_lora_q;
    const int64_t kv_lora_rank           = hparams.n_lora_kv;

    if (n_embd_head_qk_nope < 1) {
        throw std::runtime_error("GigaChat4: MLA key length must exceed rotary dimensions");
    }
    if (q_lora_rank <= 0 || kv_lora_rank <= 0) {
        throw std::runtime_error("GigaChat4: Q and KV LoRA ranks must be positive");
    }

    const int64_t head_k_dim  = hparams.ssm_d_state;
    // assumes linear_key_head_dim == linear_value_head_dim, ssm_inner_size would disagree otherwise
    const int64_t head_v_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t key_dim     = head_k_dim * num_k_heads;
    const int64_t value_dim   = head_v_dim * num_v_heads;
    const int64_t conv_dim    = 2 * key_dim + value_dim;

    for (int il = 0; il < n_layer_all; ++il) {
        auto & layer = layers[il];
        const bool is_mtp = il >= n_layer;
        const int flags = is_mtp ? mtp_flags : trunk_flags;
        const int64_t n_head_l = hparams.n_head(il);

        layer.attn_norm           = create_tensor(tn(LLM_TENSOR_ATTN_NORM,           "weight", il), { n_embd }, flags);
        layer.attn_norm_gate_up   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_GATE_UP,   "weight", il), { n_embd, norm_gate_rank }, flags);
        layer.attn_norm_gate_down = create_tensor(tn(LLM_TENSOR_ATTN_NORM_GATE_DOWN, "weight", il), { norm_gate_rank, n_embd }, flags);

        layer.attn_post_norm           = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM,           "weight", il), { n_embd }, flags);
        layer.attn_post_norm_gate_up   = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM_GATE_UP,   "weight", il), { n_embd, norm_gate_rank }, flags);
        layer.attn_post_norm_gate_down = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM_GATE_DOWN, "weight", il), { norm_gate_rank, n_embd }, flags);

        layer.ffn_norm           = create_tensor(tn(LLM_TENSOR_FFN_NORM,           "weight", il), { n_embd }, flags);
        layer.ffn_norm_gate_up   = create_tensor(tn(LLM_TENSOR_FFN_NORM_GATE_UP,   "weight", il), { n_embd, norm_gate_rank }, flags);
        layer.ffn_norm_gate_down = create_tensor(tn(LLM_TENSOR_FFN_NORM_GATE_DOWN, "weight", il), { norm_gate_rank, n_embd }, flags);

        layer.ffn_post_norm           = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,           "weight", il), { n_embd }, flags);
        layer.ffn_post_norm_gate_up   = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_GATE_UP,   "weight", il), { n_embd, norm_gate_rank }, flags);
        layer.ffn_post_norm_gate_down = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_GATE_DOWN, "weight", il), { norm_gate_rank, n_embd }, flags);

        if (is_mtp || !hparams.is_recr(il)) {
            if (n_head_l == 0) {
                throw std::runtime_error("GigaChat4: MLA layers need at least one head");
            }
            if (hparams.n_head_kv(il) != 1) {
                throw std::runtime_error("GigaChat4: MLA requires one KV head");
            }
            if (uint64_t(n_head_l) * std::max(n_embd_head_k_mla, n_embd_head_v_mla) > UINT32_MAX) {
                throw std::runtime_error("GigaChat4: MLA projection size exceeds uint32 range");
            }
            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", il), { n_embd, q_lora_rank }, flags);
            layer.attn_q_a_norm           = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,           "weight", il), { q_lora_rank }, flags);
            layer.attn_q_a_norm_gate_up   = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM_GATE_UP,   "weight", il), { q_lora_rank, norm_gate_rank }, flags);
            layer.attn_q_a_norm_gate_down = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM_GATE_DOWN, "weight", il), { norm_gate_rank, q_lora_rank }, flags);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", il), { q_lora_rank, n_head_l * n_embd_head_k_mla }, flags);

            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", il), { n_embd, kv_lora_rank + n_embd_head_qk_rope }, flags);
            layer.attn_kv_a_norm           = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM,           "weight", il), { kv_lora_rank }, flags);
            layer.attn_kv_a_norm_gate_up   = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM_GATE_UP,   "weight", il), { kv_lora_rank, norm_gate_rank }, flags);
            layer.attn_kv_a_norm_gate_down = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM_GATE_DOWN, "weight", il), { norm_gate_rank, kv_lora_rank }, flags);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", il), { n_embd_head_qk_nope, kv_lora_rank, n_head_l }, flags);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", il), { kv_lora_rank, n_embd_head_v_mla, n_head_l }, flags);
            layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_head_l * n_embd_head_v_mla, n_embd }, flags);

            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, n_head_l * n_embd_head_v_mla }, flags);

            // DSA lightning indexer, the NextN block of a sparse model always has one
            if (hparams.indexer_top_k > 0 && (is_mtp || hparams.is_indexer_full(il))) {
                const int64_t n_idx_head = hparams.indexer_n_head;
                const int64_t n_idx_dim  = hparams.indexer_head_size;

                layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", il), { q_lora_rank, n_idx_head * n_idx_dim }, flags);
                layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", il), { n_embd, n_idx_dim }, flags);
                layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", il), { n_idx_dim }, flags);
                layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", il), { n_embd, n_idx_head }, flags);
            }
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, conv_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            if (layer.ssm_conv1d && layer.ssm_conv1d->type != GGML_TYPE_F32) {
                throw std::runtime_error("GigaChat4: the SSM convolution weights must be F32");
            }
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, num_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, num_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (il < (int) hparams.n_layer_dense_lead || is_mtp) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { n_embd, n_ff }, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), { n_embd, n_ff }, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff,   n_embd }, flags);
        } else {
            // MoE branch
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", il), { n_embd, n_expert }, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   il), { n_expert }, flags);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
            create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd,    n_ff_shexp }, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", il), { n_embd,    n_ff_shexp }, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { n_ff_shexp, n_embd }, flags);
        }

        // NextN/MTP tensors
        if (is_mtp) {
            if (flags & TENSOR_SKIP) {
                // the generic scale pass only visits loaded weights
                for (const auto tensor : { LLM_TENSOR_ATTN_OUT, LLM_TENSOR_ATTN_GATE,
                        LLM_TENSOR_FFN_GATE, LLM_TENSOR_FFN_UP, LLM_TENSOR_FFN_DOWN,
                        LLM_TENSOR_NEXTN_EH_PROJ, LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD }) {
                    create_tensor(tn(tensor, "scale", il),       { 1 }, flags | TENSOR_NOT_REQUIRED);
                    create_tensor(tn(tensor, "input_scale", il), { 1 }, flags | TENSOR_NOT_REQUIRED);
                }
            }
            layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm           = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,           "weight", il), { n_embd }, flags);
            layer.nextn.enorm_gate_up   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM_GATE_UP,   "weight", il), { n_embd, norm_gate_rank }, flags);
            layer.nextn.enorm_gate_down = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM_GATE_DOWN, "weight", il), { norm_gate_rank, n_embd }, flags);
            layer.nextn.hnorm           = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,           "weight", il), { n_embd }, flags);
            layer.nextn.hnorm_gate_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM_GATE_UP,   "weight", il), { n_embd, norm_gate_rank }, flags);
            layer.nextn.hnorm_gate_down = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM_GATE_DOWN, "weight", il), { norm_gate_rank, n_embd }, flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm           = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,           "weight", il), { n_embd }, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm_gate_up   = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM_GATE_UP,   "weight", il), { n_embd, norm_gate_rank }, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm_gate_down = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM_GATE_DOWN, "weight", il), { norm_gate_rank, n_embd }, TENSOR_NOT_REQUIRED | flags);
            if (bool(layer.nextn.shared_head_norm) != bool(layer.nextn.shared_head_norm_gate_up) ||
                bool(layer.nextn.shared_head_norm) != bool(layer.nextn.shared_head_norm_gate_down)) {
                throw std::runtime_error("GigaChat4: incomplete MTP shared head norm in layer " + std::to_string(il));
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_gigachat4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }

    return std::make_unique<graph>(params);
}

llama_model_gigachat4::graph::graph(const llm_graph_params & params) : llm_build_delta_net_base(params) {
    GGML_ASSERT(hparams.is_mla());

    const float kq_scale = yarn_kq_scale();

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid_k();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;

        // norm
        ggml_tensor * cur = build_gated_norm(
                inpL, layer.attn_norm, layer.attn_norm_gate_up, layer.attn_norm_gate_down, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        // self-attention
        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn_mla(inp->get_attn(), cur, inp_pos, kq_scale, il);
        }
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        cur = build_gated_norm(
                cur, layer.attn_post_norm, layer.attn_post_norm_gate_up, layer.attn_post_norm_gate_down, il);
        cb(cur, "attn_post_norm", il);

        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "attn_residual", il);

        residual = cur;
        cur = build_gated_norm(
                cur, layer.ffn_norm, layer.ffn_norm_gate_up, layer.ffn_norm_gate_down, il);
        cb(cur, "ffn_norm", il);

        // feed-forward network
        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        cur = build_gated_norm(
                cur, layer.ffn_post_norm, layer.ffn_post_norm_gate_up, layer.ffn_post_norm_gate_down, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "ffn_residual", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    ggml_tensor * cur = build_gated_norm(
            inpL, model.output_norm, model.output_norm_gate_up, model.output_norm_gate_down, -1);

    // unlike DeepSeek, the MTP block was trained on the normed state, also when it chains its own output
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_head(cur, model.output, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_gigachat4::graph_mtp::graph_mtp(const llm_graph_params & params) : graph(params, no_build{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "GigaChat4 MTP requires n_layer_nextn > 0");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "GigaChat4 MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "GigaChat4 MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "GigaChat4 MTP block missing nextn.hnorm");

    const float kq_scale = yarn_kq_scale();

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

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

    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd);

    ggml_tensor * h_embd = inp->h;
    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // a sparse MTP block has its own indexer, its keys are in the hybrid memory like those of the trunk
    llm_graph_input_attn_k * inp_attn = nullptr;
    if (layer.indexer_attn_q_b) {
        auto * inp_hyb = build_inp_mem_hybrid_k();
        inp_attn = inp_hyb->get_attn();

        // the draft memory has no recurrent layer, but its input still has to be allocated
        ggml_build_forward_expand(gf, inp_hyb->get_recr()->s_copy);
    } else {
        inp_attn = build_attn_inp_k();
    }

    ggml_tensor * h_norm = build_gated_norm(
            h_embd, layer.nextn.hnorm, layer.nextn.hnorm_gate_up, layer.nextn.hnorm_gate_down, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_gated_norm(
            tok_embd, layer.nextn.enorm, layer.nextn.enorm_gate_up, layer.nextn.enorm_gate_down, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * residual = cur;
    cur = build_gated_norm(
            cur, layer.attn_norm, layer.attn_norm_gate_up, layer.attn_norm_gate_down, il);
    cb(cur, "mtp_attn_norm", il);

    cur = build_layer_attn_mla(inp_attn, cur, inp_pos, kq_scale, il);
    cb(cur, "mtp_attn_out", il);

    if (crop_before_nextn(inp_out_ids)) {
        cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
        residual = ggml_get_rows(ctx0, residual, inp_out_ids);
    }

    cur = build_gated_norm(
            cur, layer.attn_post_norm, layer.attn_post_norm_gate_up, layer.attn_post_norm_gate_down, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = ggml_add(ctx0, residual, cur);
    cb(cur, "mtp_attn_residual", il);

    residual = cur;
    cur = build_gated_norm(
            cur, layer.ffn_norm, layer.ffn_norm_gate_up, layer.ffn_norm_gate_down, il);
    cb(cur, "mtp_ffn_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = build_gated_norm(
            cur, layer.ffn_post_norm, layer.ffn_post_norm_gate_up, layer.ffn_post_norm_gate_down, il);
    cb(cur, "mtp_ffn_post_norm", il);

    cur = ggml_add(ctx0, residual, cur);
    cb(cur, "mtp_ffn_residual", il);

    const bool own_norm = layer.nextn.shared_head_norm != nullptr;
    cur = build_gated_norm(cur,
            own_norm ? layer.nextn.shared_head_norm           : model.output_norm,
            own_norm ? layer.nextn.shared_head_norm_gate_up   : model.output_norm_gate_up,
            own_norm ? layer.nextn.shared_head_norm_gate_down : model.output_norm_gate_down,
            -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "GigaChat4 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_head(cur, head_w, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}

ggml_tensor * llama_model_gigachat4::graph::build_head(ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * scale) {
    cur = build_lora_mm(weight, cur, scale);
    if (hparams.f_final_logit_softcapping > 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }
    return cur;
}

float llama_model_gigachat4::graph::yarn_kq_scale() const {
    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    // And also: https://github.com/ggml-org/llama.cpp/pull/17945 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]

    // first cancel the adjustment from llama_hparams::yarn_attn_factor_adjust to get the original attn_factor
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    // use the original attn_factor to pre-scale the kq_scale
    const float mscale = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    return mscale * mscale / sqrtf(float(hparams.n_embd_head_k_mla()));
}

ggml_tensor * llama_model_gigachat4::graph::build_gated_norm(
        ggml_tensor * input,
        ggml_tensor * weight,
        ggml_tensor * gate_up,
        ggml_tensor * gate_down,
        int           il) {
    GGML_ASSERT(gate_up   != nullptr);
    GGML_ASSERT(gate_down != nullptr);

    ggml_tensor * cur = build_norm(input, weight, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * gate = build_lora_mm(gate_up, cur);
    gate = ggml_silu(ctx0, gate);
    gate = build_lora_mm(gate_down, gate);
    gate = ggml_sigmoid(ctx0, gate);

    cur = ggml_mul(ctx0, cur, gate);

    return ggml_scale(ctx0, cur, 2.0f); // layernorm_gating_weight
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_gigachat4::graph::build_qkvz(
        ggml_tensor * input,
        int           il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv = ggml_reshape_3d(ctx0, qkv, qkv->ne[0], n_seq_tokens, n_seqs);
    cb(qkv, "linear_attn_qkv", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "linear_attn_z", il);

    return { qkv, z };
}

ggml_tensor * llama_model_gigachat4::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(d_inner == head_v_dim * num_v_heads);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv = qkvz.first;
    ggml_tensor * z   = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "linear_attn_beta", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    alpha = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    alpha = ggml_softplus(ctx0, alpha);
    cb(alpha, "linear_attn_alpha", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(gate, "linear_attn_gate", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    const int64_t conv_kernel_size = model.layers[il].ssm_conv1d->ne[0];
    const int64_t conv_channels    = d_inner + 2 * num_k_heads * head_k_dim;
    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "linear_attn_state", il);

    ggml_tensor * conv = ggml_ssm_conv(ctx0, conv_input, model.layers[il].ssm_conv1d);
    conv = ggml_silu(ctx0, conv);
    cb(conv, "linear_attn_conv", il);

    const int64_t key_dim = head_k_dim * num_k_heads;
    const int64_t qkv_dim = 2 * key_dim + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv->type, qkv_dim);

    ggml_tensor * q = ggml_view_4d(ctx0, conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k = ggml_view_4d(ctx0, conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv->type, key_dim));
    ggml_tensor * v = ggml_view_4d(ctx0, conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv->type, 2 * key_dim));

    const float eps_norm = 1e-6f;
    q = build_gdn_l2_norm(ctx0, q, eps_norm);
    k = build_gdn_l2_norm(ctx0, k, eps_norm);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // TODO: avoid repeats for fused GDN, needs broadcast configuration for GDN op [TAG_GGML_GDN_BCAST]
    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        const int64_t repeat_factor = num_v_heads / num_k_heads;

        // repeat interleave: reshape to (repeat part, 1, remaining part...), do repeat, then reshape back
        ggml_tensor * q_reshaped = ggml_reshape_4d(ctx0, q,
                head_k_dim, 1, num_k_heads, n_seq_tokens * n_seqs);
        ggml_tensor * k_reshaped = ggml_reshape_4d(ctx0, k,
                head_k_dim, 1, num_k_heads, n_seq_tokens * n_seqs);

        ggml_tensor * q_repeated = ggml_repeat_4d(ctx0, q_reshaped,
                head_k_dim, repeat_factor, num_k_heads, n_seq_tokens * n_seqs);
        ggml_tensor * k_repeated = ggml_repeat_4d(ctx0, k_reshaped,
                head_k_dim, repeat_factor, num_k_heads, n_seq_tokens * n_seqs);

        q = ggml_reshape_4d(ctx0, q_repeated, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k = ggml_reshape_4d(ctx0, k_repeated, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q, "linear_attn_q", il);
    cb(k, "linear_attn_k", il);
    cb(v, "linear_attn_v", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q, k, v, gate, beta, state, il);

    z = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    output = build_norm(output, model.layers[il].ssm_norm, nullptr, LLM_NORM_RMS, il);
    output = ggml_mul(ctx0, output, ggml_sigmoid(ctx0, z));
    output = ggml_scale(ctx0, output, 2.0f); // linear_sigmoid_gate_scale
    cb(output, "linear_attn_norm", il);

    output = ggml_reshape_3d(ctx0, output, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cur = build_lora_mm(model.layers[il].ssm_out, output, model.layers[il].ssm_out_s);
    cur = ggml_reshape_2d(ctx0, cur, hparams.n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_gigachat4::graph::build_layer_attn_mla(
        llm_graph_input_attn_k * inp_attn,
        ggml_tensor *            cur,
        ggml_tensor *            inp_pos,
        float                    kq_scale,
        int                      il) {
    const auto & layer = model.layers[il];

    const int64_t n_head_l              = hparams.n_head(il);
    const int64_t n_embd_head_k_mla     = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope   = hparams.n_rot();
    const int64_t n_embd_head_qk_nope   = n_embd_head_k_mla - n_embd_head_qk_rope;
    const int64_t q_lora_rank           = hparams.n_lora_q;
    const int64_t kv_lora_rank          = hparams.n_lora_kv;

    GGML_ASSERT(hparams.n_head_kv(il) == 1);
    GGML_ASSERT(n_embd_head_qk_nope > 0);
    GGML_ASSERT(kv_lora_rank > 0);

    ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, cur, layer.wqkv_gate_s);
    cb(gate, "attn_gate", il);

    // the indexer reads the same q latent
    const float alpha_q = sqrtf(float(hparams.n_embd) / float(q_lora_rank));
    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_gated_norm(qr, layer.attn_q_a_norm, layer.attn_q_a_norm_gate_up, layer.attn_q_a_norm_gate_down, il);
    qr = ggml_scale(ctx0, qr, alpha_q);
    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    cb(q, "q", il);

    // split into {n_embd_head_qk_nope, n_head, n_tokens}
    ggml_tensor * q_nope = ggml_view_3d(ctx0, q,
            n_embd_head_qk_nope, n_head_l, n_tokens,
            ggml_row_size(q->type, n_embd_head_k_mla),
            ggml_row_size(q->type, n_embd_head_k_mla) * n_head_l,
            0);
    // and {n_embd_head_qk_rope, n_head, n_tokens}
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q,
            n_embd_head_qk_rope, n_head_l, n_tokens,
            ggml_row_size(q->type, n_embd_head_k_mla),
            ggml_row_size(q->type, n_embd_head_k_mla) * n_head_l,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv = build_lora_mm(layer.wkv_a_mqa, cur);
    cb(kv, "kv_cmpr_pe", il);

    // split into {kv_lora_rank, n_tokens}
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv,
            kv_lora_rank, n_tokens,
            ggml_row_size(kv->type, kv_lora_rank + n_embd_head_qk_rope),
            0);
    // and {n_embd_head_qk_rope, 1, n_tokens}
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv,
            n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(kv->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv->type, kv_lora_rank));

    q_pe = build_rope(q_pe, inp_pos, il);
    k_pe = build_rope(k_pe, inp_pos, il);
    cb(q_pe, "q_pe", il);
    cb(k_pe, "k_pe", il);

    const float alpha_kv = sqrtf(float(hparams.n_embd) / float(kv_lora_rank));
    kv_cmpr = build_gated_norm(kv_cmpr,
            layer.attn_kv_a_norm, layer.attn_kv_a_norm_gate_up, layer.attn_kv_a_norm_gate_down, il);
    kv_cmpr = ggml_scale(ctx0, kv_cmpr, alpha_kv);
    cb(kv_cmpr, "kv_cmpr", il);

    // {n_embd_head_qk_nope, n_tokens, n_head}
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    // {kv_lora_rank, n_head, n_tokens}
    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
    cb(q_nope_absorbed, "q_nope_absorbed", il);

    // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
    cb(Qcur, "Qcur", il);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    // {kv_lora_rank, 1, n_tokens}
    ggml_tensor * Vcur = kv_cmpr;
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    if (layer.indexer_attn_q_b) {
        ggml_tensor * sel = build_indexer_mask(inp_attn, cur, qr, inp_pos, il);

        const auto * mctx_mla = inp_attn->mctx;

        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, mctx_mla->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));

        ggml_tensor * k = mctx_mla->get_k(ctx0, il);
        ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3],
            k->nb[1], k->nb[2], k->nb[3], 0);

        // the selection is an additive mask with at most n_sel finite entries per row, flash attention backends that support it skip the masked cells
        const int64_t n_sel = std::min<int64_t>(mctx_mla->get_n_kv(), hparams.indexer_top_k);
        cur = build_attn_mha(Qcur, k, v, nullptr, sel, nullptr, nullptr, layer.wv_b, n_sel, kq_scale, il);
    } else {
        // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
        cur = build_attn(inp_attn,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    gate = ggml_sigmoid(ctx0, gate);
    cur = ggml_mul(ctx0, cur, gate);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    return cur;
}

// copied from kimi-k31, with RMSNorm for the indexer k and RoPE for the indexer q and k
ggml_tensor * llama_model_gigachat4::graph::build_indexer_mask(
        llm_graph_input_attn_k * inp_attn,
        ggml_tensor *            cur,
        ggml_tensor *            qr,
        ggml_tensor *            inp_pos,
        int                      il) {
    const auto & layer = model.layers[il];

    const int64_t n_idx_head = hparams.indexer_n_head;
    const int64_t n_idx_dim  = hparams.indexer_head_size;

    // the trunk and the MTP block of a sparse model both run on llama_memory_hybrid_idx
    const llama_kv_cache_context * mctx_idx = static_cast<const llama_memory_hybrid_idx_context *>(mctx)->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    ggml_tensor * iq = build_lora_mm(layer.indexer_attn_q_b, qr);
    iq = ggml_reshape_3d(ctx0, iq, n_idx_dim, n_idx_head, n_tokens);
    iq = build_rope_indexer(iq, inp_pos, il);

    ggml_tensor * ik = build_lora_mm(layer.indexer_attn_k, cur);
    // the indexer key norm has its own eps
    ik = ggml_rms_norm(ctx0, ik, 1e-6f);
    ik = ggml_mul(ctx0, ik, layer.indexer_k_norm);
    ik = ggml_reshape_3d(ctx0, ik, n_idx_dim, 1, n_tokens);
    ik = build_rope_indexer(ik, inp_pos, il);

    // when the cache has a rotation matrix its K-shift expects rotated keys, rotating q as well keeps the scores
    if (k_rot_idx.empty()) {
        k_rot_idx = mctx_idx->get_k_rot();
    }
    ggml_tensor * k_rot = k_rot_idx[il];
    if (k_rot) {
        iq = llama_mul_mat_hadamard(ctx0, iq, k_rot);
        ik = llama_mul_mat_hadamard(ctx0, ik, k_rot);
    }
    cb(ik, "indexer_k", il);

    // the indexer keys go to the cells of the matching attention layer
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, ik, inp_attn->get_k_idxs(), il));

    ggml_tensor * iw = build_lora_mm(layer.indexer_proj, cur);
    // pre-scale so the huge score tensor is not scaled afterwards
    iw = ggml_scale(ctx0, iw, 1.0f / sqrtf(float(n_idx_dim * n_idx_head)));

    ik = mctx_idx->get_k(ctx0, il);

    // split the ubatch into its streams, k carries one row per stream
    const int64_t n_stream = ik->ne[3];
    iq = ggml_view_4d(ctx0, iq, n_idx_dim, n_idx_head, n_tokens/n_stream, n_stream,
            iq->nb[1], iq->nb[2], iq->nb[3]/n_stream, 0);
    iw = ggml_view_4d(ctx0, iw, n_idx_head, n_tokens/n_stream, 1, n_stream,
            iw->nb[1], iw->nb[2]/n_stream, iw->nb[2]/n_stream, 0);

    // the fused indexer takes an F16 mask, the plain ops add it to F32 scores; one cast serves every sparse layer
    if (kq_mask_lid == nullptr) {
        kq_mask_lid = inp_attn->self_attn_mask.raw;
        const ggml_type type_mask = cparams.fused_lid ? GGML_TYPE_F16 : GGML_TYPE_F32;
        if (kq_mask_lid->type != type_mask) {
            kq_mask_lid = ggml_cast(ctx0, kq_mask_lid, type_mask);
        }
    }

    ggml_tensor * score;
    if (cparams.fused_lid) {
        score = ggml_lightning_indexer(ctx0, iq, ik, iw, kq_mask_lid);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    } else {
        ggml_tensor * q_p = ggml_permute(ctx0, iq, 0, 2, 1, 3); // [n_idx_dim, n_tokens, n_idx_head]
        ggml_tensor * k_p = ggml_permute(ctx0, ik, 0, 2, 1, 3); // [n_idx_dim, n_kv, 1]

        ggml_tensor * kq = ggml_mul_mat(ctx0, k_p, q_p);        // [n_kv, n_tokens, n_idx_head]
        kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3)); // [n_idx_head, n_tokens, n_kv]

        score = ggml_relu(ctx0, kq);
        score = ggml_mul(ctx0, score, iw);
        score = ggml_sum_rows(ctx0, score);                      // [1, n_tokens, n_kv]
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)); // [n_kv, n_tokens, 1]

        score = ggml_add(ctx0, score, kq_mask_lid);
    }
    cb(score, "indexer_score", il);

    const int64_t n_top_k = std::min<int64_t>(score->ne[0], hparams.indexer_top_k);
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_k));
    cb(top_k, "indexer_top_k", il);

    ggml_tensor * sel = build_top_k_mask(inp_attn->self_attn_mask.raw, top_k, il);
    cb(sel, "kq_mask_dsa", il);

    return sel;
}

ggml_tensor * llama_model_gigachat4::graph::build_layer_ffn(ggml_tensor * cur, int il) {
    const auto & layer = model.layers[il];

    if ((uint32_t) il < hparams.n_layer_dense_lead) {
        return build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
    }

    // MoE branch
    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, hparams.n_expert_used(il),
            LLM_FFN_SILU,
            hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
            il,
            nullptr,
            layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // FFN shared expert
    ggml_tensor * shared = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
            layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
            layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(shared, "ffn_shared", il);

    return ggml_add(ctx0, moe_out, shared);
}
