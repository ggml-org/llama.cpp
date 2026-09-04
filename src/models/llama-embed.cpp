#include "models.h"

void llama_model_llama_embed::load_arch_tensors(llama_model_loader & ml) {
    // Load all standard LLaMA tensors (tok_embd, layers, output_norm, etc.)
    llama_model_llama::load_arch_tensors(ml);

    // Load optional classification head for reranking models.
    // For non-reranking LLaMA embed models, these are TENSOR_NOT_REQUIRED
    // and will be NULL (the build_pooling RANK branch is never entered).
    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {hparams.n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   {hparams.n_cls_out}, TENSOR_NOT_REQUIRED);
}

std::unique_ptr<llm_graph_context> llama_model_llama_embed::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<true>>(*this, params);
}
