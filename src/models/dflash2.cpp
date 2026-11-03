#include "models.h"

#include "llama-impl.h"

void llama_model_dflash::load_tensors_dflash2() {
    LLAMA_LOAD_LOCALS;

    const int64_t rank = hparams.dflash_selector_rank;
    if (rank <= 0 || hparams.dflash_block_size <= 0 || hparams.dflash_selector_top_k <= 0 ||
            hparams.dflash_conv_kernel_size <= 0 || hparams.dflash_conv_group_size <= 0) {
        throw std::runtime_error("DFlash2 model is missing conv/selector metadata");
    }
    if (n_embd < hparams.dflash_selector_top_k * (hparams.dflash_selector_top_k + 1)) {
        throw std::runtime_error("DFlash2 hidden size is too small for the selector lattice");
    }

    dflash_selector_prev   = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_PREV,   "weight"), { rank, n_vocab }, 0);
    dflash_selector_next   = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_NEXT,   "weight"), { rank, n_vocab }, 0);
    dflash_selector_hidden = create_tensor(tn(LLM_TENSOR_DFLASH_SELECTOR_HIDDEN, "weight"), { n_embd, rank }, 0);

    load_tensors_dflash2_conv();

    LLAMA_LOG_INFO("%s: DFlash2 conv kernel = %u, group = %u, selector rank = %u, top-k = %u\n", __func__,
            hparams.dflash_conv_kernel_size, hparams.dflash_conv_group_size,
            hparams.dflash_selector_rank, hparams.dflash_selector_top_k);
}

void llama_model_dflash::load_tensors_dflash2_conv() {
    LLAMA_LOAD_LOCALS;

    const bool conv_meta = hparams.dflash_conv_kernel_size > 0 && hparams.dflash_conv_group_size > 0;
    if (!conv_meta) {
        return;
    }
    if (n_embd % hparams.dflash_conv_group_size != 0) {
        throw std::runtime_error("DFlash hidden size must be divisible by conv_group_size");
    }

    const int64_t kernel = hparams.dflash_conv_kernel_size;
    const int64_t groups = n_embd / hparams.dflash_conv_group_size;
    const int64_t projected = 2 * kernel * groups;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.dflash_attn_conv_base = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_CONV_BASE, i), { n_embd, kernel, 2 }, 0);
        layer.dflash_attn_conv_proj = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_CONV_PROJ, "weight", i), { n_embd, projected }, 0);
        layer.dflash_ffn_conv_base  = create_tensor(tn(LLM_TENSOR_DFLASH_FFN_CONV_BASE, i), { n_embd, kernel, 2 }, 0);
        layer.dflash_ffn_conv_proj  = create_tensor(tn(LLM_TENSOR_DFLASH_FFN_CONV_PROJ,  "weight", i), { n_embd, projected }, 0);
    }
}
