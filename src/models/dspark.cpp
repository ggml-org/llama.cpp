#include "models.h"

// DSpark loads the full DFlash backbone, then the extra Markov / confidence head tensors.
// The graph (encoder + KV-injection decoder) is reused unchanged from llama_model_dflash; the
// Markov bias and confidence pruning are applied CPU/GPU-side in the speculative loop.
void llama_model_dspark::load_arch_tensors(llama_model_loader & ml) {
    // DFlash backbone: fc, encoder/decoder norms, per-layer attention + FFN
    llama_model_dflash::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    uint32_t markov_rank = 0;
    ml.get_key(LLM_KV_MARKOV_RANK, markov_rank, /*required*/ true);
    const int64_t R = (int64_t) markov_rank;

    // Markov head: torch [n_vocab, R] -> ggml {R, n_vocab}. w1 = prev-token embed, w2 = bias proj.
    dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), { R, n_vocab }, 0);
    dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), { R, n_vocab }, 0);

    // Confidence head (optional): torch proj [1, n_embd + R] -> ggml {n_embd + R, 1}, + [1] bias.
    dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + R, 1 }, TENSOR_NOT_REQUIRED);
    dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   { 1 },             TENSOR_NOT_REQUIRED);
}
