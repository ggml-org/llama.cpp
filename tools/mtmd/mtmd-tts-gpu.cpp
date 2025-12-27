/**
 * mtmd-tts-gpu.cpp - GPU-accelerated Code Predictor for Qwen3-Omni TTS
 *
 * This provides a GPU-accelerated implementation of the Code Predictor.
 * Current status: Stub implementation that prepares infrastructure for GPU
 * acceleration while providing CPU fallback.
 *
 * TODO: Full GPU implementation requires:
 *   - Building ggml computation graph with proper KV cache handling
 *   - Using ggml_backend_sched for backend scheduling
 *   - Handling graph rebuild for variable sequence lengths
 */

#include "mtmd-tts-gpu.h"
#include "llama-model.h"

#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>

// Internal context structure
struct mtmd_cp_gpu_context {
    const llama_model * model;

    // Backends (prepared for GPU, currently uses CPU)
    ggml_backend_t backend_cpu;
    bool gpu_available;

    // Pre-cached weights (avoids GPU->CPU copies in hot loop)
    std::vector<std::vector<float>> lm_head_cache;
    std::vector<std::vector<float>> codec_embd_cache;
    std::vector<float> output_norm_w;

    // Pre-cached layer weights
    struct LayerWeights {
        std::vector<float> attn_norm_w, wq, wk, wv, wo, q_norm_w, k_norm_w;
        std::vector<float> ffn_norm_w, ffn_gate, ffn_up, ffn_down;
    };
    std::vector<LayerWeights> layer_weights;

    // KV cache (CPU for now)
    std::vector<std::vector<float>> kv_cache_k;
    std::vector<std::vector<float>> kv_cache_v;

    // RNG for sampling
    std::mt19937 rng;

    // Thread count
    int n_threads;
};

// Forward declarations
static bool copy_tensor_to_cpu(const ggml_tensor * t, std::vector<float> & out);
static void rms_norm(const float * x, const float * weight, float * out, int n, float eps = 1e-6f);
static void matmul(const float * a, const float * b, float * out, int m, int k, int n);
static void apply_rope(float * q, float * k, int head_dim, int n_head, int n_head_kv, int pos, float theta);
static void causal_attention(
    const float * q, const float * k, const float * v,
    float * out,
    int n_head, int n_head_kv, int head_dim,
    int n_kv, int pos);
static void swiglu_ffn(
    const float * x, int n_embd, int n_ff,
    const float * gate, const float * up, const float * down,
    float * out, float * scratch);
static int sample_token(const float * logits, int n_vocab, float temp, std::mt19937 & rng);

// =============================================================================
// Public API
// =============================================================================

mtmd_cp_gpu_context * mtmd_cp_gpu_init(
    const struct llama_model * model,
    int n_threads) {

    if (!model) {
        return nullptr;
    }

    // Check if model has Code Predictor weights
    if (model->talker_cp_layers.empty() ||
        model->talker_cp_lm_head.empty() ||
        model->talker_cp_codec_embd.empty()) {
        fprintf(stderr, "mtmd_cp_gpu_init: Model missing Code Predictor weights\n");
        return nullptr;
    }

    auto * ctx = new mtmd_cp_gpu_context();
    ctx->model = model;
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    ctx->gpu_available = false;  // TODO: Enable when GPU graph is implemented

    // Seed RNG
    std::random_device rd;
    ctx->rng.seed(rd());

    // Initialize CPU backend
    ctx->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!ctx->backend_cpu) {
        fprintf(stderr, "mtmd_cp_gpu_init: Failed to initialize CPU backend\n");
        delete ctx;
        return nullptr;
    }

    // Pre-cache all weights to avoid GPU->CPU transfers during inference

    // Output norm
    if (!copy_tensor_to_cpu(model->talker_cp_output_norm, ctx->output_norm_w)) {
        fprintf(stderr, "mtmd_cp_gpu_init: Failed to cache output norm\n");
        ggml_backend_free(ctx->backend_cpu);
        delete ctx;
        return nullptr;
    }

    // Layer weights
    ctx->layer_weights.resize(CP_N_LAYER);
    for (int il = 0; il < CP_N_LAYER; ++il) {
        const auto & layer = model->talker_cp_layers[il];
        auto & lw = ctx->layer_weights[il];

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
            fprintf(stderr, "mtmd_cp_gpu_init: Failed to cache layer %d weights\n", il);
            ggml_backend_free(ctx->backend_cpu);
            delete ctx;
            return nullptr;
        }
    }

    // LM heads
    ctx->lm_head_cache.resize(CP_N_CODEBOOKS);
    for (int cb = 0; cb < CP_N_CODEBOOKS; ++cb) {
        if (cb < (int)model->talker_cp_lm_head.size()) {
            if (!copy_tensor_to_cpu(model->talker_cp_lm_head[cb], ctx->lm_head_cache[cb])) {
                fprintf(stderr, "mtmd_cp_gpu_init: Failed to cache LM head %d\n", cb);
                ggml_backend_free(ctx->backend_cpu);
                delete ctx;
                return nullptr;
            }
        }
    }

    // Codec embeddings
    ctx->codec_embd_cache.resize(CP_N_CODEBOOKS);
    for (int cb = 0; cb < CP_N_CODEBOOKS; ++cb) {
        if (cb < (int)model->talker_cp_codec_embd.size()) {
            if (!copy_tensor_to_cpu(model->talker_cp_codec_embd[cb], ctx->codec_embd_cache[cb])) {
                fprintf(stderr, "mtmd_cp_gpu_init: Failed to cache codec embedding %d\n", cb);
                ggml_backend_free(ctx->backend_cpu);
                delete ctx;
                return nullptr;
            }
        }
    }

    // Allocate KV cache
    ctx->kv_cache_k.resize(CP_N_LAYER);
    ctx->kv_cache_v.resize(CP_N_LAYER);
    for (int il = 0; il < CP_N_LAYER; ++il) {
        ctx->kv_cache_k[il].resize(CP_MAX_SEQ * CP_N_HEAD_KV * CP_HEAD_DIM, 0.0f);
        ctx->kv_cache_v[il].resize(CP_MAX_SEQ * CP_N_HEAD_KV * CP_HEAD_DIM, 0.0f);
    }

    fprintf(stderr, "mtmd_cp_gpu_init: Initialized with pre-cached weights (CPU mode)\n");

    return ctx;
}

void mtmd_cp_gpu_free(mtmd_cp_gpu_context * ctx) {
    if (!ctx) return;

    if (ctx->backend_cpu) {
        ggml_backend_free(ctx->backend_cpu);
    }

    delete ctx;
}

bool mtmd_cp_gpu_available(const mtmd_cp_gpu_context * ctx) {
    return ctx && ctx->gpu_available;
}

void mtmd_cp_gpu_reset(mtmd_cp_gpu_context * ctx) {
    if (!ctx) return;

    // Zero KV cache
    for (int il = 0; il < CP_N_LAYER; ++il) {
        std::fill(ctx->kv_cache_k[il].begin(), ctx->kv_cache_k[il].end(), 0.0f);
        std::fill(ctx->kv_cache_v[il].begin(), ctx->kv_cache_v[il].end(), 0.0f);
    }
}

bool mtmd_cp_gpu_generate(
    mtmd_cp_gpu_context * ctx,
    const float * past_hidden,
    const float * last_id_hidden,
    int * codebook_tokens,
    float * codec_embeddings,
    uint64_t rng_seed,
    float temperature,
    bool verbose) {

    if (!ctx || !past_hidden || !last_id_hidden || !codebook_tokens || !codec_embeddings) {
        return false;
    }

    // Seed RNG if provided
    if (rng_seed != 0) {
        ctx->rng.seed(rng_seed);
    }

    // Reset KV cache
    mtmd_cp_gpu_reset(ctx);

    // Working buffers
    std::vector<float> cur(CP_N_EMBD);
    std::vector<float> residual(CP_N_EMBD);
    std::vector<float> normed(CP_N_EMBD);
    std::vector<float> attn_out(CP_N_EMBD);
    std::vector<float> ffn_out(CP_N_EMBD);
    std::vector<float> scratch(CP_N_FF);
    std::vector<float> logits(CP_VOCAB);

    // Transformer step lambda (uses pre-cached weights)
    auto run_transformer_step = [&](const float * input, int pos) {
        std::copy(input, input + CP_N_EMBD, cur.begin());

        for (int il = 0; il < CP_N_LAYER; ++il) {
            const auto & lw = ctx->layer_weights[il];
            std::copy(cur.begin(), cur.end(), residual.begin());

            // Attention norm
            rms_norm(cur.data(), lw.attn_norm_w.data(), normed.data(), CP_N_EMBD);

            // Q/K/V projections
            std::vector<float> q(CP_N_HEAD * CP_HEAD_DIM);
            std::vector<float> k(CP_N_HEAD_KV * CP_HEAD_DIM);
            std::vector<float> v(CP_N_HEAD_KV * CP_HEAD_DIM);

            matmul(normed.data(), lw.wq.data(), q.data(), 1, CP_N_EMBD, CP_N_HEAD * CP_HEAD_DIM);
            matmul(normed.data(), lw.wk.data(), k.data(), 1, CP_N_EMBD, CP_N_HEAD_KV * CP_HEAD_DIM);
            matmul(normed.data(), lw.wv.data(), v.data(), 1, CP_N_EMBD, CP_N_HEAD_KV * CP_HEAD_DIM);

            // Q/K norms (per head)
            for (int h = 0; h < CP_N_HEAD; ++h) {
                std::vector<float> q_head(CP_HEAD_DIM);
                std::copy(q.data() + h * CP_HEAD_DIM, q.data() + (h + 1) * CP_HEAD_DIM, q_head.begin());
                rms_norm(q_head.data(), lw.q_norm_w.data(), &q[h * CP_HEAD_DIM], CP_HEAD_DIM);
            }
            for (int h = 0; h < CP_N_HEAD_KV; ++h) {
                std::vector<float> k_head(CP_HEAD_DIM);
                std::copy(k.data() + h * CP_HEAD_DIM, k.data() + (h + 1) * CP_HEAD_DIM, k_head.begin());
                rms_norm(k_head.data(), lw.k_norm_w.data(), &k[h * CP_HEAD_DIM], CP_HEAD_DIM);
            }

            // RoPE
            apply_rope(q.data(), k.data(), CP_HEAD_DIM, CP_N_HEAD, CP_N_HEAD_KV, pos, CP_ROPE_THETA);

            // Update KV cache
            std::copy(k.begin(), k.end(), ctx->kv_cache_k[il].begin() + pos * CP_N_HEAD_KV * CP_HEAD_DIM);
            std::copy(v.begin(), v.end(), ctx->kv_cache_v[il].begin() + pos * CP_N_HEAD_KV * CP_HEAD_DIM);

            // Attention
            causal_attention(
                q.data(), ctx->kv_cache_k[il].data(), ctx->kv_cache_v[il].data(),
                attn_out.data(),
                CP_N_HEAD, CP_N_HEAD_KV, CP_HEAD_DIM,
                pos + 1, pos);

            // Output projection
            std::vector<float> attn_proj(CP_N_EMBD);
            matmul(attn_out.data(), lw.wo.data(), attn_proj.data(), 1, CP_N_HEAD * CP_HEAD_DIM, CP_N_EMBD);

            // Residual
            for (int i = 0; i < CP_N_EMBD; ++i) {
                cur[i] = residual[i] + attn_proj[i];
            }

            // FFN
            std::copy(cur.begin(), cur.end(), residual.begin());
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

    // Generate 15 codebook tokens
    for (int cb = 0; cb < CP_N_CODEBOOKS; ++cb) {
        int pos = cb + 2;

        // Output norm + LM head
        rms_norm(cur.data(), ctx->output_norm_w.data(), normed.data(), CP_N_EMBD);
        matmul(normed.data(), ctx->lm_head_cache[cb].data(), logits.data(), 1, CP_N_EMBD, CP_VOCAB);

        // Sample
        float cp_temp = (temperature <= 0.0f) ? 0.0f : 0.9f;
        int token = sample_token(logits.data(), CP_VOCAB, cp_temp, ctx->rng);
        codebook_tokens[cb] = token;

        // Get embedding
        const float * embd_row = ctx->codec_embd_cache[cb].data() + token * CP_N_EMBD;
        std::copy(embd_row, embd_row + CP_N_EMBD, &codec_embeddings[cb * CP_N_EMBD]);

        // Next transformer step (except last)
        if (cb < CP_N_CODEBOOKS - 1) {
            run_transformer_step(&codec_embeddings[cb * CP_N_EMBD], pos);
        }
    }

    // Final embedding for Code2Wav
    int last_token = codebook_tokens[CP_N_CODEBOOKS - 1];
    const float * last_embd = ctx->codec_embd_cache[CP_N_CODEBOOKS - 1].data() + last_token * CP_N_EMBD;
    std::copy(last_embd, last_embd + CP_N_EMBD, &codec_embeddings[CP_N_CODEBOOKS * CP_N_EMBD]);

    if (verbose) {
        fprintf(stderr, "  Code Predictor (GPU ctx): generated %d codebook tokens\n", CP_N_CODEBOOKS);
    }

    return true;
}

// =============================================================================
// Internal Implementation
// =============================================================================

static bool copy_tensor_to_cpu(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) return false;

    int64_t n_elem = ggml_nelements(t);
    out.resize(n_elem);

    if (t->buffer) {
        // Tensor is on a backend, need to copy
        if (ggml_is_quantized(t->type) || t->type == GGML_TYPE_F16) {
            // Need to dequantize
            std::vector<uint8_t> raw(ggml_nbytes(t));
            ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));

            // Use ggml to dequantize
            const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
            if (traits && traits->to_float) {
                traits->to_float(raw.data(), out.data(), n_elem);
            } else {
                return false;
            }
        } else {
            ggml_backend_tensor_get(t, out.data(), 0, n_elem * sizeof(float));
        }
    } else if (t->data) {
        // Tensor is CPU-resident
        if (t->type == GGML_TYPE_F32) {
            memcpy(out.data(), t->data, n_elem * sizeof(float));
        } else {
            return false;
        }
    } else {
        return false;
    }

    return true;
}

static void rms_norm(const float * x, const float * weight, float * out, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    float inv_rms = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; ++i) {
        out[i] = (x[i] * inv_rms) * weight[i];
    }
}

static void matmul(const float * a, const float * b, float * out, int m, int k, int n) {
    // out[m,n] = a[m,k] @ b[k,n]^T  (b stored as [n,k])
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

static void apply_rope(float * q, float * k, int head_dim, int n_head, int n_head_kv, int pos, float theta) {
    // Apply rotary position embedding
    for (int h = 0; h < n_head; ++h) {
        float * qh = q + h * head_dim;
        for (int i = 0; i < head_dim / 2; ++i) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float q0 = qh[i];
            float q1 = qh[i + head_dim / 2];
            qh[i] = q0 * cos_a - q1 * sin_a;
            qh[i + head_dim / 2] = q0 * sin_a + q1 * cos_a;
        }
    }

    for (int h = 0; h < n_head_kv; ++h) {
        float * kh = k + h * head_dim;
        for (int i = 0; i < head_dim / 2; ++i) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float k0 = kh[i];
            float k1 = kh[i + head_dim / 2];
            kh[i] = k0 * cos_a - k1 * sin_a;
            kh[i + head_dim / 2] = k0 * sin_a + k1 * cos_a;
        }
    }
}

static void causal_attention(
    const float * q, const float * k_cache, const float * v_cache,
    float * out,
    int n_head, int n_head_kv, int head_dim,
    int n_kv, int pos) {

    (void)pos;  // pos is implicit in n_kv

    // GQA: n_head queries share n_head_kv key/values
    int n_rep = n_head / n_head_kv;

    std::fill(out, out + n_head * head_dim, 0.0f);

    for (int h = 0; h < n_head; ++h) {
        int kv_head = h / n_rep;
        const float * qh = q + h * head_dim;

        // Compute attention scores
        std::vector<float> scores(n_kv);
        for (int t = 0; t < n_kv; ++t) {
            const float * kt = k_cache + t * n_head_kv * head_dim + kv_head * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += qh[d] * kt[d];
            }
            scores[t] = dot / sqrtf((float)head_dim);
        }

        // Softmax
        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        for (int t = 0; t < n_kv; ++t) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        for (int t = 0; t < n_kv; ++t) {
            scores[t] /= sum;
        }

        // Weighted sum of values
        float * oh = out + h * head_dim;
        for (int t = 0; t < n_kv; ++t) {
            const float * vt = v_cache + t * n_head_kv * head_dim + kv_head * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                oh[d] += scores[t] * vt[d];
            }
        }
    }
}

static void swiglu_ffn(
    const float * x, int n_embd, int n_ff,
    const float * gate, const float * up, const float * down,
    float * out, float * scratch) {

    // gate_out = silu(x @ gate^T)
    // up_out = x @ up^T
    // hidden = gate_out * up_out
    // out = hidden @ down^T

    std::vector<float> gate_out(n_ff);
    std::vector<float> up_out(n_ff);

    matmul(x, gate, gate_out.data(), 1, n_embd, n_ff);
    matmul(x, up, up_out.data(), 1, n_embd, n_ff);

    // SiLU activation on gate
    for (int i = 0; i < n_ff; ++i) {
        float g = gate_out[i];
        gate_out[i] = g / (1.0f + expf(-g));  // silu(x) = x * sigmoid(x)
    }

    // Element-wise multiply
    for (int i = 0; i < n_ff; ++i) {
        scratch[i] = gate_out[i] * up_out[i];
    }

    // Down projection
    matmul(scratch, down, out, 1, n_ff, n_embd);
}

static int sample_token(const float * logits, int n_vocab, float temp, std::mt19937 & rng) {
    if (temp <= 0.0f) {
        // Greedy
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    // Temperature sampling
    std::vector<float> probs(n_vocab);
    float max_logit = *std::max_element(logits, logits + n_vocab);

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = expf((logits[i] - max_logit) / temp);
        sum += probs[i];
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng) * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        cumsum += probs[i];
        if (r <= cumsum) return i;
    }

    return n_vocab - 1;
}
