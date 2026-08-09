#include "models.h"

// Stub to allow llama-quantize to open mmproj GGUFs

void llama_model_clip::load_arch_hparams(llama_model_loader &) {
    GGML_ABORT("CLIP is a quant-only stub; load_arch_hparams should not be called");
}

void llama_model_clip::load_arch_tensors(llama_model_loader &) {
    GGML_ABORT("CLIP is a quant-only stub; load_arch_tensors should not be called");
}

std::unique_ptr<llm_graph_context> llama_model_clip::build_arch_graph(const llm_graph_params &) const {
    GGML_ABORT("CLIP has no inference graph via llama_model dispatch; runtime lives in tools/mtmd/clip.cpp");
}
