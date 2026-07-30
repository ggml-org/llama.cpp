#pragma once

//
// tessera-ppl.h
//
// E2E perplexity / KL-divergence probe for quantized model evaluation.
// Data-free KL variant on random tokens (HIGGS probe metric).
//

#include <cstdint>

struct ts_ppl_params {
    int64_t  n_tokens;      // number of probe tokens (default 256)
    int64_t  vocab_size;    // vocabulary size for softmax (default 32000)
    bool     use_kl;        // true = KL divergence, false = cross-entropy PPL
    uint32_t seed;          // for random token generation
};

struct ts_ppl_result {
    float   kl_divergence;  // KL(ref || quant), nats
    float   ppl_ratio;      // PPL(quant) / PPL(ref)
    float   delta_ppl;      // PPL(quant) - PPL(ref)
    int64_t n_tokens_used;
};

// Mean KL(ref || quant) per token.
// logits_ref, logits_quant: (n_tokens x vocab_size) row-major.
float ts_ppl_kl_divergence(const float * logits_ref,
                           const float * logits_quant,
                           int64_t n_tokens, int64_t vocab_size);

// Perplexity from logits and target token IDs.
// logits: (n_tokens x vocab_size). targets: (n_tokens,) token IDs.
float ts_ppl_perplexity(const float * logits, const int32_t * targets,
                        int64_t n_tokens, int64_t vocab_size);

// Forward callback: fills logits_out (n_tokens x vocab_size) for given tokens.
typedef void (*ts_ppl_forward_fn)(const int32_t * tokens, float * logits_out,
                                  int64_t n_tokens, int64_t vocab_size,
                                  void * model_ctx);

// Full E2E probe: random tokens -> both forwards -> KL + PPL metrics.
// Returns 0 on success, -1 on invalid params.
int ts_ppl_probe(ts_ppl_forward_fn forward_ref, void * ref_ctx,
                 ts_ppl_forward_fn forward_quant, void * quant_ctx,
                 const ts_ppl_params * params,
                 ts_ppl_result * result);
