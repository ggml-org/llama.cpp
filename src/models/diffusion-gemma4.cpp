#include "models.h"

// diffusion_gemma4 reuses the gemma4 decoder block wholesale (hparams loading and the
// graph). The only extra weights are the top-level self-conditioning MLP. The block
// diffusion sampling loop and the self-conditioning / bidirectional / encoder-KV graph
// wiring are added on top of this base in a later step.

void llama_model_diffusion_gemma4::load_arch_tensors(llama_model_loader & ml) {
    // load the shared gemma4 tensors (token embd, attention, dual dense+MoE FFN, norms,
    // per-layer layer_scalar, output)
    llama_model_gemma4::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    // self_conditioning is a gated MLP at hidden_size -> intermediate_size -> hidden_size
    const int64_t n_ff_sc = n_ff;

    self_cond_norm = create_tensor(tn(LLM_TENSOR_SELF_COND_NORM, "weight"), {n_embd},           0);
    self_cond_gate = create_tensor(tn(LLM_TENSOR_SELF_COND_GATE, "weight"), {n_embd,  n_ff_sc}, 0);
    self_cond_up   = create_tensor(tn(LLM_TENSOR_SELF_COND_UP,   "weight"), {n_embd,  n_ff_sc}, 0);
    self_cond_down = create_tensor(tn(LLM_TENSOR_SELF_COND_DOWN, "weight"), {n_ff_sc, n_embd},  0);
}
