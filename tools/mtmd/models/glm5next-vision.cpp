#include "models.h"

// the GLM-OCR ViT plus a clamp on the SwiGLU gate and up projections, which the per-block MLP
// and the merger both pick up from hparams.ffn_op
// ref: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/modeling_glm5_next.py
ggml_cgraph * clip_graph_glm5next::build() {
    GGML_ASSERT(hparams.ffn_op == FFN_SILU_CLAMP);
    GGML_ASSERT(model.norm_embd_w == nullptr);
    GGML_ASSERT(model.position_embeddings == nullptr);

    return clip_graph_glm4v::build();
}
