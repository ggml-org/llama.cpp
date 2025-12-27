/**
 * mtmd-tts.cpp - Text-to-Speech pipeline for Qwen3-Omni
 *
 * Generates speech from Thinker embeddings via:
 *   1. Text projection MLP (2048 -> 1024)
 *   2. Talker autoregressive generation (codec tokens)
 *   3. Code Predictor (1 -> 16 codebooks)
 *   4. Code2Wav vocoder (see mtmd-tts-code2wav.cpp)
 *
 * Reference: HuggingFace transformers modeling_qwen3_omni_moe.py
 *
 * Extracted from tools/qwen3omni-tts/main.cpp
 */

#include "mtmd-tts.h"
#include "mtmd-tts-gpu.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

// Internal headers for model tensor access
#include "llama-model.h"
#include "llama-context.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <set>

// =============================================================================
// Constants
// =============================================================================

// Special tokens for Talker (from talker_config in config.json)
static const int TALKER_CODEC_PAD_ID = 2148;       // codec_pad_id
static const int TALKER_CODEC_BOS_ID = 2149;       // codec_bos_id
static const int TALKER_CODEC_EOS_ID = 2150;       // codec_eos_token_id
static const int TALKER_CODEC_NOTHINK_ID = 2155;   // codec_nothink_id
static const int TALKER_CODEC_THINK_BOS_ID = 2156; // codec_think_bos_id
static const int TALKER_CODEC_THINK_EOS_ID = 2157; // codec_think_eos_id

// TTS special tokens from Thinker vocab
static const int TTS_PAD_TOKEN_ID = 151671;  // tts_pad_token_id
static const int TTS_BOS_TOKEN_ID = 151672;  // tts_bos_token_id
static const int TTS_EOS_TOKEN_ID = 151673;  // tts_eos_token_id

// Code Predictor dimensions - defined in mtmd-tts-gpu.h as macros:
// CP_N_EMBD, CP_N_HEAD, CP_N_HEAD_KV, CP_HEAD_DIM, CP_N_FF,
// CP_N_LAYER, CP_VOCAB, CP_N_CODEBOOKS, CP_ROPE_THETA

// =============================================================================
// Forward declarations (from mtmd-tts-code2wav.cpp)
// =============================================================================

extern std::vector<float> mtmd_code2wav_run(
    const llama_model * model,
    const std::vector<std::vector<int>> & all_codebook_tokens,
    bool verbose,
    bool cpu_only);

// =============================================================================
// TTS Context
// =============================================================================

struct mtmd_tts_context {
    const llama_model * thinker_model;
    const llama_model * talker_model;
    llama_context * talker_ctx;
    struct mtmd_tts_params params;

    // RNG for sampling
    std::mt19937 rng;

    // Talker token embeddings (cached on CPU)
    std::vector<float> tok_embd_data;
    int64_t tok_embd_dim;
    int64_t tok_embd_vocab;

    // Thinker embedding dimension
    int n_embd_thinker;
    int n_embd_talker;

    // GPU-accelerated Code Predictor context (optional, can be nullptr)
    mtmd_cp_gpu_context * cp_gpu_ctx;

    mtmd_tts_context()
        : thinker_model(nullptr)
        , talker_model(nullptr)
        , talker_ctx(nullptr)
        , tok_embd_dim(0)
        , tok_embd_vocab(0)
        , n_embd_thinker(0)
        , n_embd_talker(0)
        , cp_gpu_ctx(nullptr) {}
};

// =============================================================================
// Helper Functions
// =============================================================================

// SiLU activation: x * sigmoid(x)
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

// RMSNorm: x / sqrt(mean(x^2) + eps) * weight
static void rms_norm(const float * x, const float * weight, float * out, int n, float eps = 1e-6f) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; ++i) {
        out[i] = (x[i] / rms) * weight[i];
    }
}

// Matrix multiplication: out[m,n] = a[m,k] @ b[k,n]^T (b stored as [n,k])
static void matmul(const float * a, const float * b, float * out, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[j * k + l];
            }
            out[i * n + j] = sum;
        }
    }
}

// SwiGLU FFN: out = (silu(gate(x)) * up(x)) @ down
static void swiglu_ffn(const float * x, int n_embd, int n_ff,
                       const float * gate_w, const float * up_w, const float * down_w,
                       float * out, float * scratch) {
    std::vector<float> gate(n_ff), up(n_ff);
    matmul(x, gate_w, gate.data(), 1, n_embd, n_ff);
    matmul(x, up_w, up.data(), 1, n_embd, n_ff);

    for (int i = 0; i < n_ff; ++i) {
        scratch[i] = silu(gate[i]) * up[i];
    }

    matmul(scratch, down_w, out, 1, n_ff, n_embd);
}

// Apply RoPE to a single head
static void apply_rope_to_head(float * qk, int head_dim, int pos, float rope_theta) {
    const int half_dim = head_dim / 2;
    for (int i = 0; i < half_dim; ++i) {
        float freq = 1.0f / powf(rope_theta, 2.0f * i / head_dim);
        float angle = pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        float x0 = qk[i];
        float x1 = qk[i + half_dim];
        qk[i] = x0 * cos_val - x1 * sin_val;
        qk[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

static void apply_rope(float * qk, int n_head, int head_dim, int pos, float rope_theta) {
    for (int h = 0; h < n_head; ++h) {
        apply_rope_to_head(qk + h * head_dim, head_dim, pos, rope_theta);
    }
}

// Copy tensor data to CPU vector
static bool copy_tensor_to_cpu(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) return false;
    int64_t n_elem = ggml_nelements(t);
    out.resize(n_elem);

    if (t->buffer) {
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, out.data(), 0, n_elem * sizeof(float));
            return true;
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(n_elem);
            ggml_backend_tensor_get(t, tmp.data(), 0, n_elem * sizeof(ggml_fp16_t));
            for (int64_t i = 0; i < n_elem; ++i) {
                out[i] = ggml_fp16_to_fp32(tmp[i]);
            }
            return true;
        }
        return false;
    }

    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, n_elem * sizeof(float));
        return true;
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n_elem; ++i) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
        return true;
    }
    return false;
}

// Sample token with temperature, top-k, top-p
static int sample_token(const float * logits, int n_vocab, float temperature, int top_k,
                        std::mt19937 & rng, const std::vector<int> & recent_tokens = {},
                        float rep_penalty = 1.1f, float top_p = 0.9f) {
    // Greedy for temp <= 0
    if (temperature <= 0.0f) {
        float max_logit = -1e30f;
        int best_token = 0;
        for (int i = 0; i < n_vocab; ++i) {
            if (i >= 2048 && i != TALKER_CODEC_EOS_ID && i != TALKER_CODEC_THINK_EOS_ID) {
                continue;
            }
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                best_token = i;
            }
        }
        return best_token;
    }

    std::vector<std::pair<float, int>> logits_sorted;
    logits_sorted.reserve(n_vocab);
    std::set<int> recent_set(recent_tokens.begin(), recent_tokens.end());

    for (int i = 0; i < n_vocab; ++i) {
        // Skip special tokens except EOS
        if (i >= 2048 && i != TALKER_CODEC_EOS_ID && i != TALKER_CODEC_THINK_EOS_ID) {
            continue;
        }
        float logit = logits[i];
        if (recent_set.count(i) > 0) {
            logit = (logit > 0) ? logit / rep_penalty : logit * rep_penalty;
        }
        logits_sorted.push_back({logit, i});
    }

    std::sort(logits_sorted.begin(), logits_sorted.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });

    // Temperature + softmax
    float max_logit = logits_sorted[0].first;
    std::vector<float> probs(logits_sorted.size());
    float sum = 0.0f;

    for (size_t i = 0; i < logits_sorted.size(); ++i) {
        float p = expf((logits_sorted[i].first - max_logit) / temperature);
        probs[i] = p;
        sum += p;
    }

    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }

    // Top-p sampling
    float cumsum = 0.0f;
    size_t top_p_cutoff = probs.size();
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (cumsum >= top_p) {
            top_p_cutoff = i + 1;
            break;
        }
    }

    size_t actual_k = std::min({(size_t)top_k, top_p_cutoff, probs.size()});

    // Renormalize
    sum = 0.0f;
    for (size_t i = 0; i < actual_k; ++i) {
        sum += probs[i];
    }
    for (size_t i = 0; i < actual_k; ++i) {
        probs[i] /= sum;
    }

    // Sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    cumsum = 0.0f;

    for (size_t i = 0; i < actual_k; ++i) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return logits_sorted[i].second;
        }
    }

    return logits_sorted[0].second;
}

// Check if token is EOS
static bool is_talker_eos(int token) {
    return token == TALKER_CODEC_EOS_ID || token == TALKER_CODEC_THINK_EOS_ID;
}

// =============================================================================
// Causal attention with KV cache (for Code Predictor)
// =============================================================================

static void causal_attention_with_kv_cache(
    const float * x, int n_embd, int n_head, int n_head_kv, int head_dim, int pos,
    const float * wq, const float * wk, const float * wv, const float * wo,
    const float * q_norm_w, const float * k_norm_w,
    float rope_theta,
    float * kv_cache_k, float * kv_cache_v,
    float * out) {

    int gqa_ratio = n_head / n_head_kv;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Project Q, K, V
    std::vector<float> q(n_head * head_dim), k(n_head_kv * head_dim), v(n_head_kv * head_dim);
    matmul(x, wq, q.data(), 1, n_embd, n_head * head_dim);
    matmul(x, wk, k.data(), 1, n_embd, n_head_kv * head_dim);
    matmul(x, wv, v.data(), 1, n_embd, n_head_kv * head_dim);

    // Apply QK normalization
    std::vector<float> q_normed(n_head * head_dim), k_normed(n_head_kv * head_dim);
    for (int h = 0; h < n_head; ++h) {
        rms_norm(q.data() + h * head_dim, q_norm_w, q_normed.data() + h * head_dim, head_dim);
    }
    for (int h = 0; h < n_head_kv; ++h) {
        rms_norm(k.data() + h * head_dim, k_norm_w, k_normed.data() + h * head_dim, head_dim);
    }

    // Apply RoPE
    apply_rope(q_normed.data(), n_head, head_dim, pos, rope_theta);
    apply_rope(k_normed.data(), n_head_kv, head_dim, pos, rope_theta);

    // Store in KV cache
    for (int h = 0; h < n_head_kv; ++h) {
        for (int d = 0; d < head_dim; ++d) {
            kv_cache_k[pos * n_head_kv * head_dim + h * head_dim + d] = k_normed[h * head_dim + d];
            kv_cache_v[pos * n_head_kv * head_dim + h * head_dim + d] = v[h * head_dim + d];
        }
    }

    // Compute attention for each head
    std::vector<float> attn_out(n_head * head_dim, 0.0f);
    for (int qh = 0; qh < n_head; ++qh) {
        int kv_h = qh / gqa_ratio;

        std::vector<float> scores(pos + 1);
        float max_score = -1e30f;
        for (int p = 0; p <= pos; ++p) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_normed[qh * head_dim + d] *
                         kv_cache_k[p * n_head_kv * head_dim + kv_h * head_dim + d];
            }
            score *= scale;
            scores[p] = score;
            if (score > max_score) max_score = score;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int p = 0; p <= pos; ++p) {
            scores[p] = expf(scores[p] - max_score);
            sum_exp += scores[p];
        }
        for (int p = 0; p <= pos; ++p) {
            scores[p] /= sum_exp;
        }

        // Weighted sum
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int p = 0; p <= pos; ++p) {
                acc += scores[p] * kv_cache_v[p * n_head_kv * head_dim + kv_h * head_dim + d];
            }
            attn_out[qh * head_dim + d] = acc;
        }
    }

    // Output projection
    matmul(attn_out.data(), wo, out, 1, n_head * head_dim, n_embd);
}

// =============================================================================
// Text Projection MLP (2048 -> 1024)
// =============================================================================

static bool apply_text_projection(
    const llama_model * model,
    const float * input,
    float * output,
    int n_tokens,
    bool verbose) {

    if (!model->talker_text_proj_fc1 || !model->talker_text_proj_fc2) {
        fprintf(stderr, "Warning: Text projection tensors not found\n");
        return false;
    }

    const int n_embd_in = 2048;
    const int n_hidden = 2048;
    const int n_embd_out = 1024;

    // Copy weights to CPU
    std::vector<float> fc1_w_data, fc1_b_data, fc2_w_data, fc2_b_data;

    if (!copy_tensor_to_cpu(model->talker_text_proj_fc1, fc1_w_data) ||
        !copy_tensor_to_cpu(model->talker_text_proj_fc2, fc2_w_data)) {
        fprintf(stderr, "Warning: Failed to copy text projection weights\n");
        return false;
    }

    bool has_fc1_b = copy_tensor_to_cpu(model->talker_text_proj_fc1_b, fc1_b_data);
    bool has_fc2_b = copy_tensor_to_cpu(model->talker_text_proj_fc2_b, fc2_b_data);

    std::vector<float> hidden(n_tokens * n_hidden);

    // fc1 + SiLU
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_hidden; ++h) {
            float sum = 0.0f;
            for (int i = 0; i < n_embd_in; ++i) {
                sum += input[t * n_embd_in + i] * fc1_w_data[h * n_embd_in + i];
            }
            if (has_fc1_b) {
                sum += fc1_b_data[h];
            }
            hidden[t * n_hidden + h] = silu(sum);
        }
    }

    // fc2
    for (int t = 0; t < n_tokens; ++t) {
        for (int o = 0; o < n_embd_out; ++o) {
            float sum = 0.0f;
            for (int h = 0; h < n_hidden; ++h) {
                sum += hidden[t * n_hidden + h] * fc2_w_data[o * n_hidden + h];
            }
            if (has_fc2_b) {
                sum += fc2_b_data[o];
            }
            output[t * n_embd_out + o] = sum;
        }
    }

    return true;
}

// =============================================================================
// Code Predictor (1 -> 16 codebooks)
// =============================================================================

static bool run_code_predictor_inline(
    const llama_model * model,
    const std::vector<float> & past_hidden,
    const std::vector<float> & last_id_hidden,
    std::vector<std::vector<float>> & codec_embeddings,
    std::vector<int> & codebook_tokens,
    std::mt19937 & rng,
    float temperature,
    bool verbose) {

    const int max_seq_len = 20;

    codec_embeddings.clear();
    codec_embeddings.reserve(CP_N_CODEBOOKS);
    codebook_tokens.clear();
    codebook_tokens.reserve(CP_N_CODEBOOKS);

    // KV cache
    std::vector<std::vector<float>> kv_cache_k(CP_N_LAYER,
        std::vector<float>(max_seq_len * CP_N_HEAD_KV * CP_HEAD_DIM, 0.0f));
    std::vector<std::vector<float>> kv_cache_v(CP_N_LAYER,
        std::vector<float>(max_seq_len * CP_N_HEAD_KV * CP_HEAD_DIM, 0.0f));

    // Buffers
    std::vector<float> cur(CP_N_EMBD);
    std::vector<float> residual(CP_N_EMBD);
    std::vector<float> normed(CP_N_EMBD);
    std::vector<float> attn_out(CP_N_EMBD);
    std::vector<float> ffn_out(CP_N_EMBD);
    std::vector<float> scratch(CP_N_FF);

    // Pre-copy layer weights
    struct LayerWeights {
        std::vector<float> attn_norm_w, wq, wk, wv, wo, q_norm_w, k_norm_w;
        std::vector<float> ffn_norm_w, ffn_gate, ffn_up, ffn_down;
    };
    std::vector<LayerWeights> layer_weights(CP_N_LAYER);

    std::vector<float> output_norm_w;
    if (!copy_tensor_to_cpu(model->talker_cp_output_norm, output_norm_w)) {
        fprintf(stderr, "Error: Failed to copy Code Predictor output norm\n");
        return false;
    }

    for (int il = 0; il < CP_N_LAYER; ++il) {
        const auto & layer = model->talker_cp_layers[il];
        auto & lw = layer_weights[il];

        if (!copy_tensor_to_cpu(layer.attn_norm, lw.attn_norm_w) ||
            !copy_tensor_to_cpu(layer.wq, lw.wq) ||
            !copy_tensor_to_cpu(layer.wk, lw.wk) ||
            !copy_tensor_to_cpu(layer.wv, lw.wv) ||
            !copy_tensor_to_cpu(layer.wo, lw.wo) ||
            !copy_tensor_to_cpu(layer.attn_q_norm, lw.q_norm_w) ||
            !copy_tensor_to_cpu(layer.attn_k_norm, lw.k_norm_w) ||
            !copy_tensor_to_cpu(layer.ffn_norm, lw.ffn_norm_w) ||
            !copy_tensor_to_cpu(layer.ffn_gate, lw.ffn_gate) ||
            !copy_tensor_to_cpu(layer.ffn_up, lw.ffn_up) ||
            !copy_tensor_to_cpu(layer.ffn_down, lw.ffn_down)) {
            fprintf(stderr, "Error: Failed to copy Code Predictor layer %d weights\n", il);
            return false;
        }
    }

    // Pre-cache LM heads and codec embeddings to eliminate GPU→CPU transfers in hot loop
    std::vector<std::vector<float>> lm_head_cache(CP_N_CODEBOOKS);
    std::vector<std::vector<float>> codec_embd_cache(CP_N_CODEBOOKS);

    for (int cb = 0; cb < CP_N_CODEBOOKS; ++cb) {
        if (cb < (int)model->talker_cp_lm_head.size()) {
            if (!copy_tensor_to_cpu(model->talker_cp_lm_head[cb], lm_head_cache[cb])) {
                fprintf(stderr, "Error: Failed to cache LM head %d\n", cb);
                return false;
            }
        }
        if (cb < (int)model->talker_cp_codec_embd.size()) {
            if (!copy_tensor_to_cpu(model->talker_cp_codec_embd[cb], codec_embd_cache[cb])) {
                fprintf(stderr, "Error: Failed to cache codec embedding %d\n", cb);
                return false;
            }
        }
    }

    // Pre-allocate logits buffer (moved out of loop)
    std::vector<float> logits(CP_VOCAB);

    // Transformer step lambda
    auto run_transformer_step = [&](const std::vector<float>& input, int pos) {
        cur = input;
        for (int il = 0; il < CP_N_LAYER; ++il) {
            const auto & lw = layer_weights[il];
            residual = cur;

            // Attention
            rms_norm(cur.data(), lw.attn_norm_w.data(), normed.data(), CP_N_EMBD);
            causal_attention_with_kv_cache(
                normed.data(), CP_N_EMBD, CP_N_HEAD, CP_N_HEAD_KV, CP_HEAD_DIM, pos,
                lw.wq.data(), lw.wk.data(), lw.wv.data(), lw.wo.data(),
                lw.q_norm_w.data(), lw.k_norm_w.data(),
                CP_ROPE_THETA,
                kv_cache_k[il].data(), kv_cache_v[il].data(),
                attn_out.data());

            for (int i = 0; i < CP_N_EMBD; ++i) {
                cur[i] = residual[i] + attn_out[i];
            }

            // FFN
            residual = cur;
            rms_norm(cur.data(), lw.ffn_norm_w.data(), normed.data(), CP_N_EMBD);
            swiglu_ffn(normed.data(), CP_N_EMBD, CP_N_FF,
                       lw.ffn_gate.data(), lw.ffn_up.data(), lw.ffn_down.data(),
                       ffn_out.data(), scratch.data());

            for (int i = 0; i < CP_N_EMBD; ++i) {
                cur[i] = residual[i] + ffn_out[i];
            }
        }
    };

    // Process 2-token input sequence
    run_transformer_step(past_hidden, 0);
    run_transformer_step(last_id_hidden, 1);

    // Generate 15 codebook tokens autoregressively
    for (int cb = 0; cb < CP_N_CODEBOOKS; ++cb) {
        int pos = cb + 2;

        // Apply output norm and LM head
        rms_norm(cur.data(), output_norm_w.data(), normed.data(), CP_N_EMBD);

        if (cb >= (int)lm_head_cache.size() || lm_head_cache[cb].empty()) {
            fprintf(stderr, "Error: LM head %d not cached\n", cb);
            return false;
        }

        // Use pre-cached LM head (no GPU→CPU copy here)
        matmul(normed.data(), lm_head_cache[cb].data(), logits.data(), 1, CP_N_EMBD, CP_VOCAB);

        // Sample
        float cp_temp = (temperature <= 0.0f) ? 0.0f : 0.9f;
        int best_token = sample_token(logits.data(), CP_VOCAB, cp_temp, 50, rng, {}, 1.0f, 0.8f);
        codebook_tokens.push_back(best_token);

        // Get embedding for this token (use pre-cached codec embeddings)
        if (cb < (int)codec_embd_cache.size() && !codec_embd_cache[cb].empty()) {
            std::vector<float> token_embd(CP_N_EMBD);
            if (best_token >= 0 && best_token < CP_VOCAB) {
                const float * embd_row = codec_embd_cache[cb].data() + best_token * CP_N_EMBD;
                for (int i = 0; i < CP_N_EMBD; ++i) {
                    token_embd[i] = embd_row[i];
                }
            }
            codec_embeddings.push_back(token_embd);
            run_transformer_step(token_embd, pos);
        }
    }

    // Add last_residual_hidden from final embedding table (use cached)
    if (!codebook_tokens.empty() && !codec_embd_cache.empty()) {
        int last_table_idx = (int)codec_embd_cache.size() - 1;
        if (!codec_embd_cache[last_table_idx].empty()) {
            int last_token = codebook_tokens.back();
            std::vector<float> last_residual_hidden(CP_N_EMBD);
            if (last_token >= 0 && last_token < CP_VOCAB) {
                const float * embd_row = codec_embd_cache[last_table_idx].data() + last_token * CP_N_EMBD;
                for (int i = 0; i < CP_N_EMBD; ++i) {
                    last_residual_hidden[i] = embd_row[i];
                }
            }
            codec_embeddings.push_back(last_residual_hidden);
        }
    }

    if (verbose) {
        fprintf(stderr, "  Code Predictor: generated %zu codebook tokens, %zu embeddings\n",
                codebook_tokens.size(), codec_embeddings.size());
    }

    return true;
}

// =============================================================================
// Public API
// =============================================================================

struct mtmd_tts_params mtmd_tts_params_default(void) {
    struct mtmd_tts_params params = {
        /* .temperature      = */ 0.9f,
        /* .top_k            = */ 50,
        /* .top_p            = */ 0.9f,
        /* .max_codec_tokens = */ 500,
        /* .speaker_id       = */ 2302,  // Ethan
        /* .sample_rate      = */ 24000,
        /* .verbose          = */ false,
        /* .cpu_only         = */ false,
    };
    return params;
}

mtmd_tts_context * mtmd_tts_init(
    const struct llama_model * thinker_model,
    const struct llama_model * talker_model,
    struct mtmd_tts_params params) {

    if (!thinker_model || !talker_model) {
        fprintf(stderr, "Error: Both Thinker and Talker models required\n");
        return nullptr;
    }

    mtmd_tts_context * ctx = new mtmd_tts_context();
    ctx->thinker_model = thinker_model;
    ctx->talker_model = talker_model;
    ctx->params = params;

    // Initialize RNG
    std::random_device rd;
    ctx->rng.seed(rd());

    // Get embedding dimensions
    ctx->n_embd_thinker = llama_model_n_embd(thinker_model);
    ctx->n_embd_talker = llama_model_n_embd(talker_model);

    // Create Talker context
    llama_context_params talker_cparams = llama_context_default_params();
    talker_cparams.n_ctx = 2048;
    talker_cparams.n_batch = 512;
    talker_cparams.embeddings = true;

    ctx->talker_ctx = llama_init_from_model(const_cast<llama_model *>(talker_model), talker_cparams);
    if (!ctx->talker_ctx) {
        fprintf(stderr, "Error: Failed to create Talker context\n");
        delete ctx;
        return nullptr;
    }

    // Enable debug layer outputs to match standalone TTS behavior
    // This preserves layer output tensors which may be needed for hidden state extraction
    llama_set_debug_layer_outputs(ctx->talker_ctx, true);

    // Cache Talker token embeddings on CPU
    const llama_model * tm = talker_model;
    if (tm->tok_embd) {
        ctx->tok_embd_dim = tm->tok_embd->ne[0];
        ctx->tok_embd_vocab = tm->tok_embd->ne[1];
        ctx->tok_embd_data.resize(ctx->tok_embd_dim * ctx->tok_embd_vocab);

        if (tm->tok_embd->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(tm->tok_embd, ctx->tok_embd_data.data(), 0,
                                    ggml_nbytes(tm->tok_embd));
        } else if (tm->tok_embd->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> f16_buf(ctx->tok_embd_dim * ctx->tok_embd_vocab);
            ggml_backend_tensor_get(tm->tok_embd, f16_buf.data(), 0, ggml_nbytes(tm->tok_embd));
            for (size_t i = 0; i < f16_buf.size(); ++i) {
                ctx->tok_embd_data[i] = ggml_fp16_to_fp32(f16_buf[i]);
            }
        }
    }

    if (params.verbose) {
        fprintf(stderr, "TTS: Thinker embd=%d, Talker embd=%d, vocab=%lld\n",
                ctx->n_embd_thinker, ctx->n_embd_talker, (long long)ctx->tok_embd_vocab);
    }

    // Initialize GPU Code Predictor context (with pre-cached weights)
    // This is optional - if it fails, we fall back to the inline CPU implementation
    ctx->cp_gpu_ctx = mtmd_cp_gpu_init(talker_model, 4);
    if (ctx->cp_gpu_ctx && params.verbose) {
        fprintf(stderr, "TTS: GPU Code Predictor context initialized\n");
    }

    return ctx;
}

void mtmd_tts_free(mtmd_tts_context * ctx) {
    if (ctx) {
        if (ctx->cp_gpu_ctx) {
            mtmd_cp_gpu_free(ctx->cp_gpu_ctx);
        }
        if (ctx->talker_ctx) {
            llama_free(ctx->talker_ctx);
        }
        delete ctx;
    }
}

int mtmd_tts_generate(
    mtmd_tts_context * ctx,
    const float * embeddings,
    int n_tokens,
    float * output_samples,
    int max_samples) {

    if (!ctx || !embeddings || n_tokens <= 0) {
        return -1;
    }

    const int n_embd_in = ctx->n_embd_thinker;
    const int n_embd = ctx->n_embd_talker;
    const bool verbose = ctx->params.verbose;

    // Step 1: Apply text projection (2048 -> 1024)
    if (verbose) {
        fprintf(stderr, "TTS: Applying text projection...\n");
    }

    std::vector<float> projected(n_tokens * n_embd);
    if (!apply_text_projection(ctx->talker_model, embeddings, projected.data(), n_tokens, verbose)) {
        fprintf(stderr, "Error: Text projection failed\n");
        return -1;
    }

    // Helper to get codec embedding
    auto get_codec_embd = [&](int token_id, float * out) {
        if (token_id >= 0 && token_id < ctx->tok_embd_vocab) {
            for (int i = 0; i < n_embd; ++i) {
                out[i] = ctx->tok_embd_data[token_id * ctx->tok_embd_dim + i];
            }
        }
    };

    // Extract TTS special embeddings from Thinker
    const llama_model * tm = ctx->thinker_model;
    auto extract_thinker_embd = [&](int token_id, std::vector<float> & raw, std::vector<float> & proj) {
        raw.resize(n_embd_in, 0.0f);
        proj.resize(n_embd, 0.0f);

        if (tm->tok_embd && token_id < (int64_t)tm->tok_embd->ne[1]) {
            size_t row_offset = (size_t)token_id * n_embd_in;
            size_t elem_size = ggml_type_size(tm->tok_embd->type);
            size_t byte_offset = row_offset * elem_size;

            if (tm->tok_embd->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(tm->tok_embd, raw.data(), byte_offset, n_embd_in * sizeof(float));
            } else if (tm->tok_embd->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16_buf(n_embd_in);
                ggml_backend_tensor_get(tm->tok_embd, f16_buf.data(), byte_offset, n_embd_in * sizeof(ggml_fp16_t));
                for (int i = 0; i < n_embd_in; ++i) {
                    raw[i] = ggml_fp16_to_fp32(f16_buf[i]);
                }
            }

            apply_text_projection(ctx->talker_model, raw.data(), proj.data(), 1, false);
        }
    };

    std::vector<float> tts_pad_raw, tts_pad_embed;
    std::vector<float> tts_bos_raw, tts_bos_embed;
    std::vector<float> tts_eos_raw, tts_eos_embed;

    extract_thinker_embd(TTS_PAD_TOKEN_ID, tts_pad_raw, tts_pad_embed);
    extract_thinker_embd(TTS_BOS_TOKEN_ID, tts_bos_raw, tts_bos_embed);
    extract_thinker_embd(TTS_EOS_TOKEN_ID, tts_eos_raw, tts_eos_embed);

    // Step 2: Build prefill sequence (9 positions)
    if (verbose) {
        fprintf(stderr, "TTS: Building prefill sequence...\n");
    }

    const int n_prefill = 9;
    std::vector<float> prefill_embeds(n_prefill * n_embd, 0.0f);

    // Position 0-2: text[0:3]
    for (int p = 0; p < 3 && p < n_tokens; ++p) {
        memcpy(&prefill_embeds[p * n_embd], &projected[p * n_embd], n_embd * sizeof(float));
    }

    // Position 3: tts_pad + nothink_embed
    {
        float * dst = &prefill_embeds[3 * n_embd];
        std::vector<float> codec_embd(n_embd);
        get_codec_embd(TALKER_CODEC_NOTHINK_ID, codec_embd.data());
        for (int i = 0; i < n_embd; ++i) {
            dst[i] = tts_pad_embed[i] + codec_embd[i];
        }
    }

    // Position 4: tts_pad + think_bos_embed
    {
        float * dst = &prefill_embeds[4 * n_embd];
        std::vector<float> codec_embd(n_embd);
        get_codec_embd(TALKER_CODEC_THINK_BOS_ID, codec_embd.data());
        for (int i = 0; i < n_embd; ++i) {
            dst[i] = tts_pad_embed[i] + codec_embd[i];
        }
    }

    // Position 5: tts_pad + think_eos_embed
    {
        float * dst = &prefill_embeds[5 * n_embd];
        std::vector<float> codec_embd(n_embd);
        get_codec_embd(TALKER_CODEC_THINK_EOS_ID, codec_embd.data());
        for (int i = 0; i < n_embd; ++i) {
            dst[i] = tts_pad_embed[i] + codec_embd[i];
        }
    }

    // Position 6: tts_pad + speaker_id_embed
    {
        float * dst = &prefill_embeds[6 * n_embd];
        std::vector<float> codec_embd(n_embd);
        get_codec_embd(ctx->params.speaker_id, codec_embd.data());
        for (int i = 0; i < n_embd; ++i) {
            dst[i] = tts_pad_embed[i] + codec_embd[i];
        }
    }

    // Position 7: tts_bos + codec_pad_embed
    {
        float * dst = &prefill_embeds[7 * n_embd];
        std::vector<float> codec_embd(n_embd);
        get_codec_embd(TALKER_CODEC_PAD_ID, codec_embd.data());
        for (int i = 0; i < n_embd; ++i) {
            dst[i] = tts_bos_embed[i] + codec_embd[i];
        }
    }

    // Position 8: text[3] + codec_bos_embed
    {
        float * dst = &prefill_embeds[8 * n_embd];
        std::vector<float> codec_embd(n_embd);
        get_codec_embd(TALKER_CODEC_BOS_ID, codec_embd.data());
        if (n_tokens > 3) {
            for (int i = 0; i < n_embd; ++i) {
                dst[i] = projected[3 * n_embd + i] + codec_embd[i];
            }
        } else {
            for (int i = 0; i < n_embd; ++i) {
                dst[i] = tts_pad_embed[i] + codec_embd[i];
            }
        }
    }

    // Build trailing_text_hidden = [text[4:], tts_eos_embed]
    int n_trailing_text = (n_tokens > 4) ? (n_tokens - 4) : 0;
    int n_trailing = n_trailing_text + 1;
    std::vector<float> trailing_text_hidden(n_trailing * n_embd);

    for (int t = 0; t < n_trailing_text; ++t) {
        memcpy(&trailing_text_hidden[t * n_embd], &projected[(t + 4) * n_embd], n_embd * sizeof(float));
    }
    memcpy(&trailing_text_hidden[n_trailing_text * n_embd], tts_eos_embed.data(), n_embd * sizeof(float));

    // Step 3: Run Talker prefill
    if (verbose) {
        fprintf(stderr, "TTS: Running Talker prefill...\n");
    }

    llama_memory_clear(llama_get_memory(ctx->talker_ctx), true);

    llama_batch batch = llama_batch_init(n_prefill, n_embd, 1);
    batch.n_tokens = n_prefill;
    memcpy(batch.embd, prefill_embeds.data(), n_prefill * n_embd * sizeof(float));
    for (int p = 0; p < n_prefill; ++p) {
        batch.pos[p] = p;
        batch.n_seq_id[p] = 1;
        batch.seq_id[p][0] = 0;
        batch.logits[p] = (p == n_prefill - 1) ? 1 : 0;
    }

    if (llama_decode(ctx->talker_ctx, batch) != 0) {
        fprintf(stderr, "Error: Talker prefill failed\n");
        llama_batch_free(batch);
        return -1;
    }
    llama_batch_free(batch);

    // Get initial hidden state
    std::vector<float> past_hidden(n_embd, 0.0f);
    float * embs = llama_get_embeddings(ctx->talker_ctx);
    if (embs) {
        memcpy(past_hidden.data(), embs, n_embd * sizeof(float));
    }

    // Sample first token
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ctx->talker_model));
    float * logits = llama_get_logits_ith(ctx->talker_ctx, -1);
    if (!logits) {
        fprintf(stderr, "Error: Failed to get logits\n");
        return -1;
    }

    // Track codec tokens for repetition penalty (matching HuggingFace)
    std::vector<int> codec_tokens;
    codec_tokens.reserve(ctx->params.max_codec_tokens);

    // HuggingFace uses top_p=1.0 (nucleus sampling disabled)
    int last_token = sample_token(logits, n_vocab, ctx->params.temperature, ctx->params.top_k,
                                  ctx->rng, {}, 1.05f, 1.0f);
    codec_tokens.push_back(last_token);

    if (verbose) {
        fprintf(stderr, "TTS: First token from prefill: %d\n", last_token);
    }

    // Step 4: Autoregressive generation
    if (verbose) {
        fprintf(stderr, "TTS: Generating codec tokens...\n");
    }

    std::vector<std::vector<int>> all_codebook_tokens;
    all_codebook_tokens.reserve(ctx->params.max_codec_tokens);

    std::vector<float> cur_embd(n_embd);
    std::vector<float> last_id_hidden(n_embd);
    int pos = n_prefill;

    for (int step = 0; step < ctx->params.max_codec_tokens; ++step) {
        // Get codec token embedding
        get_codec_embd(last_token, last_id_hidden.data());

        // Run Code Predictor
        std::vector<std::vector<float>> cp_hidden_states;
        std::vector<int> cp_codebook_tokens;

        if (run_code_predictor_inline(ctx->talker_model, past_hidden, last_id_hidden,
                                      cp_hidden_states, cp_codebook_tokens, ctx->rng,
                                      ctx->params.temperature, verbose && step < 3)) {
            // Sum embeddings: last_id_hidden + all Code Predictor outputs
            std::fill(cur_embd.begin(), cur_embd.end(), 0.0f);
            for (int i = 0; i < n_embd; ++i) {
                cur_embd[i] = last_id_hidden[i];
            }
            for (const auto & hs : cp_hidden_states) {
                for (int i = 0; i < n_embd; ++i) {
                    cur_embd[i] += hs[i];
                }
            }

            // Store all 16 codebook tokens
            std::vector<int> frame_tokens(16);
            frame_tokens[0] = last_token;
            for (size_t cb = 0; cb < cp_codebook_tokens.size() && cb + 1 < 16; ++cb) {
                frame_tokens[cb + 1] = cp_codebook_tokens[cb];
            }
            all_codebook_tokens.push_back(frame_tokens);
        } else {
            // Fallback
            for (int i = 0; i < n_embd; ++i) {
                cur_embd[i] = last_id_hidden[i];
            }
        }

        // Add text embedding
        if (step < n_trailing) {
            for (int i = 0; i < n_embd; ++i) {
                cur_embd[i] += trailing_text_hidden[step * n_embd + i];
            }
        } else {
            for (int i = 0; i < n_embd; ++i) {
                cur_embd[i] += tts_pad_embed[i];
            }
        }

        // Run Talker step
        llama_batch step_batch = llama_batch_init(1, n_embd, 1);
        step_batch.n_tokens = 1;
        memcpy(step_batch.embd, cur_embd.data(), n_embd * sizeof(float));
        step_batch.pos[0] = pos++;
        step_batch.n_seq_id[0] = 1;
        step_batch.seq_id[0][0] = 0;
        step_batch.logits[0] = 1;

        if (llama_decode(ctx->talker_ctx, step_batch) != 0) {
            fprintf(stderr, "Error: Talker decode failed at step %d\n", step);
            llama_batch_free(step_batch);
            break;
        }
        llama_batch_free(step_batch);

        // Update past_hidden
        embs = llama_get_embeddings(ctx->talker_ctx);
        if (embs) {
            memcpy(past_hidden.data(), embs, n_embd * sizeof(float));
        }

        // Sample next token
        logits = llama_get_logits_ith(ctx->talker_ctx, -1);
        if (!logits) break;

        // Get recent tokens for repetition penalty (last 32 tokens)
        std::vector<int> recent_tokens;
        size_t lookback = std::min(codec_tokens.size(), (size_t)32);
        for (size_t i = codec_tokens.size() - lookback; i < codec_tokens.size(); ++i) {
            recent_tokens.push_back(codec_tokens[i]);
        }

        // HuggingFace uses top_p=1.0 (nucleus sampling disabled)
        last_token = sample_token(logits, n_vocab, ctx->params.temperature, ctx->params.top_k,
                                  ctx->rng, recent_tokens, 1.05f, 1.0f);
        codec_tokens.push_back(last_token);

        if (is_talker_eos(last_token)) {
            if (verbose) {
                fprintf(stderr, "TTS: EOS token at step %d\n", step);
            }
            break;
        }
    }

    if (verbose) {
        fprintf(stderr, "TTS: Generated %zu codec frames\n", all_codebook_tokens.size());
    }

    if (all_codebook_tokens.empty()) {
        return 0;
    }

    // Step 5: Run Code2Wav vocoder
    if (verbose) {
        fprintf(stderr, "TTS: Running Code2Wav vocoder...\n");
    }

    std::vector<float> audio = mtmd_code2wav_run(
        ctx->talker_model, all_codebook_tokens, verbose, ctx->params.cpu_only);

    // Apply fade in/out
    int n_samples = (int)audio.size();
    int fade_samples = std::min(500, n_samples / 8);
    for (int i = 0; i < fade_samples; ++i) {
        float fade = (float)i / fade_samples;
        audio[i] *= fade;
        audio[n_samples - 1 - i] *= fade;
    }

    // Copy to output
    int output_size = std::min(n_samples, max_samples);
    memcpy(output_samples, audio.data(), output_size * sizeof(float));

    if (verbose) {
        fprintf(stderr, "TTS: Generated %d audio samples (%.2f sec)\n",
                output_size, output_size / 24000.0f);
    }

    return output_size;
}

int mtmd_tts_generate_from_text(
    mtmd_tts_context * ctx,
    struct llama_context * llama_ctx,
    const char * text,
    float * output_samples,
    int max_samples) {

    if (!ctx || !llama_ctx || !text) {
        fprintf(stderr, "TTS: Invalid parameters\n");
        return -1;
    }

    bool verbose = ctx->params.verbose;
    const llama_model * model = llama_get_model(llama_ctx);
    if (!model) {
        fprintf(stderr, "TTS: Cannot get model from context\n");
        return -1;
    }

    // Step 1: Tokenize the text with assistant role prefix
    // The Talker expects text embeddings to include the assistant role tokens:
    // <|im_start|>assistant\n{text}
    // This matches how the standalone TTS (qwen3omni-tts) extracts the assistant segment
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        fprintf(stderr, "TTS: Cannot get vocab from model\n");
        return -1;
    }

    // Build full text with assistant role prefix
    std::string full_text = "<|im_start|>assistant\n";
    full_text += text;

    int text_len = (int)full_text.size();
    int max_tokens = text_len + 128;  // Allow for token expansion
    std::vector<llama_token> tokens(max_tokens);

    // IMPORTANT: parse_special=true is required to recognize <|im_start|> as a special token
    // Without this, the special token markers are tokenized as regular text which breaks TTS
    int n_tokens = llama_tokenize(vocab, full_text.c_str(), text_len, tokens.data(), max_tokens, false, true);
    if (n_tokens < 0) {
        fprintf(stderr, "TTS: Tokenization failed\n");
        return -1;
    }
    tokens.resize(n_tokens);

    if (verbose) {
        fprintf(stderr, "TTS: Tokenized with assistant prefix: %d tokens\n", n_tokens);
    }

    if (n_tokens == 0) {
        fprintf(stderr, "TTS: No tokens to speak\n");
        return -1;
    }

    // Step 2: Extract token embeddings from Thinker's tok_embd
    int n_embd = ctx->n_embd_thinker;
    if (n_embd <= 0) {
        n_embd = llama_model_n_embd(model);
    }

    // Access the token embedding tensor from the model
    // Cast away const to access internal model structure
    const llama_model * thinker = ctx->thinker_model ? ctx->thinker_model : model;
    struct ggml_tensor * tok_embd = ((llama_model *)thinker)->tok_embd;

    if (!tok_embd) {
        fprintf(stderr, "TTS: Model has no tok_embd tensor\n");
        return -1;
    }

    std::vector<float> embeddings(n_tokens * n_embd);
    size_t elem_size = ggml_type_size(tok_embd->type);

    for (int t = 0; t < n_tokens; ++t) {
        int token_id = tokens[t];
        size_t row_offset = (size_t)token_id * n_embd;
        size_t byte_offset = row_offset * elem_size;
        float * dst = &embeddings[t * n_embd];

        if (tok_embd->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(tok_embd, dst, byte_offset, n_embd * sizeof(float));
        } else if (tok_embd->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> f16_buf(n_embd);
            ggml_backend_tensor_get(tok_embd, f16_buf.data(), byte_offset, n_embd * sizeof(ggml_fp16_t));
            for (int i = 0; i < n_embd; ++i) {
                dst[i] = ggml_fp16_to_fp32(f16_buf[i]);
            }
        } else {
            fprintf(stderr, "TTS: Unsupported tok_embd type: %d\n", tok_embd->type);
            return -1;
        }
    }

    if (verbose) {
        fprintf(stderr, "TTS: Extracted embeddings for %d tokens (%d dims each)\n", n_tokens, n_embd);
    }

    // Step 3: Generate speech using the extracted embeddings
    return mtmd_tts_generate(ctx, embeddings.data(), n_tokens, output_samples, max_samples);
}
