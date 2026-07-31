//
// tessera-ppl.cpp
//
// E2E perplexity / KL-divergence probe for quantized model evaluation.
// Numerically stable softmax, KL(p||q), cross-entropy PPL.
//

#include "tessera-ppl.h"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

static uint32_t ts_ppl_xorshift32(uint32_t * state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Deterministic random token IDs in [0, vocab_size), shared by the probe
// and the L4 compare so both models see identical input.
static void ts_ppl_gen_tokens(int32_t * tokens, int64_t n_tokens,
                              int64_t vocab_size, uint32_t seed) {
    uint32_t rng = seed;
    for (int64_t i = 0; i < n_tokens; i++) {
        tokens[i] = (int32_t)(ts_ppl_xorshift32(&rng) % (uint32_t)vocab_size);
    }
}

// ---------------------------------------------------------------------------
// Softmax helpers (in-place, numerically stable)
// ---------------------------------------------------------------------------

static void ts_softmax_inplace(float * row, int64_t n) {
    float mx = row[0];
    for (int64_t i = 1; i < n; i++) {
        if (row[i] > mx) mx = row[i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        row[i] = expf(row[i] - mx);
        sum += row[i];
    }
    float inv = 1.0f / sum;
    for (int64_t i = 0; i < n; i++) {
        row[i] *= inv;
    }
}

// ---------------------------------------------------------------------------
// KL divergence
// ---------------------------------------------------------------------------

float ts_ppl_kl_divergence(const float * logits_ref,
                           const float * logits_quant,
                           int64_t n_tokens, int64_t vocab_size) {
    std::vector<float> p(vocab_size);
    std::vector<float> q(vocab_size);

    double kl_sum = 0.0;
    for (int64_t t = 0; t < n_tokens; t++) {
        const float * lr = logits_ref   + t * vocab_size;
        const float * lq = logits_quant + t * vocab_size;

        for (int64_t i = 0; i < vocab_size; i++) {
            p[i] = lr[i];
            q[i] = lq[i];
        }
        ts_softmax_inplace(p.data(), vocab_size);
        ts_softmax_inplace(q.data(), vocab_size);

        for (int64_t i = 0; i < vocab_size; i++) {
            if (p[i] > 0.0f) {
                kl_sum += (double)p[i] * log((double)p[i] / (double)q[i]);
            }
        }
    }
    return (float)(kl_sum / n_tokens);
}

// ---------------------------------------------------------------------------
// Perplexity
// ---------------------------------------------------------------------------

float ts_ppl_perplexity(const float * logits, const int32_t * targets,
                        int64_t n_tokens, int64_t vocab_size) {
    std::vector<float> row(vocab_size);

    double nll_sum = 0.0;
    for (int64_t t = 0; t < n_tokens; t++) {
        const float * l = logits + t * vocab_size;
        for (int64_t i = 0; i < vocab_size; i++) {
            row[i] = l[i];
        }
        ts_softmax_inplace(row.data(), vocab_size);

        int32_t tgt = targets[t];
        float prob = (tgt >= 0 && tgt < vocab_size) ? row[tgt] : 0.0f;
        if (prob < 1e-30f) prob = 1e-30f;
        nll_sum -= log((double)prob);
    }
    return (float)exp(nll_sum / n_tokens);
}

// ---------------------------------------------------------------------------
// E2E probe
// ---------------------------------------------------------------------------

int ts_ppl_probe(ts_ppl_forward_fn forward_ref, void * ref_ctx,
                 ts_ppl_forward_fn forward_quant, void * quant_ctx,
                 const ts_ppl_params * params,
                 ts_ppl_result * result) {
    if (!forward_ref || !forward_quant || !params || !result) {
        return -1;
    }

    int64_t n_tokens   = params->n_tokens   > 0 ? params->n_tokens   : 256;
    int64_t vocab_size = params->vocab_size > 0 ? params->vocab_size : 32000;
    uint32_t seed      = params->seed ? params->seed : 42;

    // generate random token IDs
    std::vector<int32_t> tokens(n_tokens);
    ts_ppl_gen_tokens(tokens.data(), n_tokens, vocab_size, seed);

    size_t buf_size = (size_t)n_tokens * (size_t)vocab_size;
    std::vector<float> logits_ref(buf_size);
    std::vector<float> logits_quant(buf_size);

    forward_ref(tokens.data(), logits_ref.data(), n_tokens, vocab_size, ref_ctx);
    forward_quant(tokens.data(), logits_quant.data(), n_tokens, vocab_size, quant_ctx);

    result->kl_divergence = ts_ppl_kl_divergence(
        logits_ref.data(), logits_quant.data(), n_tokens, vocab_size);

    float ppl_ref  = ts_ppl_perplexity(logits_ref.data(),  tokens.data(), n_tokens, vocab_size);
    float ppl_quant = ts_ppl_perplexity(logits_quant.data(), tokens.data(), n_tokens, vocab_size);

    result->ppl_ratio     = ppl_quant / ppl_ref;
    result->delta_ppl     = ppl_quant - ppl_ref;
    result->n_tokens_used = n_tokens;

    return 0;
}

// ---------------------------------------------------------------------------
// L4 end-to-end comparison
// ---------------------------------------------------------------------------

int ts_ppl_compare(ts_ppl_forward_fn forward_ref, void * ref_ctx,
                   ts_ppl_forward_fn forward_quant, void * quant_ctx,
                   const ts_ppl_params * params,
                   float pass_threshold,
                   ts_ppl_compare_result * result) {
    if (!forward_ref || !forward_quant || !params || !result) {
        return -1;
    }

    int64_t n_tokens   = params->n_tokens   > 0 ? params->n_tokens   : 256;
    int64_t vocab_size = params->vocab_size > 0 ? params->vocab_size : 32000;
    uint32_t seed      = params->seed ? params->seed : 42;
    float threshold    = pass_threshold > 0.0f ? pass_threshold : 0.5f;

    std::vector<int32_t> tokens(n_tokens);
    ts_ppl_gen_tokens(tokens.data(), n_tokens, vocab_size, seed);

    size_t buf_size = (size_t)n_tokens * (size_t)vocab_size;
    std::vector<float> logits_ref(buf_size);
    std::vector<float> logits_quant(buf_size);

    forward_ref(tokens.data(), logits_ref.data(), n_tokens, vocab_size, ref_ctx);
    forward_quant(tokens.data(), logits_quant.data(), n_tokens, vocab_size, quant_ctx);

    result->ppl_ref   = ts_ppl_perplexity(logits_ref.data(),   tokens.data(), n_tokens, vocab_size);
    result->ppl_quant = ts_ppl_perplexity(logits_quant.data(), tokens.data(), n_tokens, vocab_size);
    result->kl_divergence = ts_ppl_kl_divergence(
        logits_ref.data(), logits_quant.data(), n_tokens, vocab_size);

    result->delta_ppl     = result->ppl_quant - result->ppl_ref;
    result->ppl_ratio     = result->ppl_ref > 0.0f ? result->ppl_quant / result->ppl_ref : 0.0f;
    result->threshold     = threshold;
    result->pass          = result->delta_ppl < threshold;
    result->n_tokens_used = n_tokens;

    return 0;
}
