#include "llama-hparams.h"

#include "ggml.h"

// Only define functions that are not already inline in the header
uint32_t llama_hparams::n_embd_k_s() const {
    if (wkv_head_size != 0) {
        // for RWKV models
        return token_shift_count * n_embd;
    }

    // TODO: maybe support other convolution strides than 1
    // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
    return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * ssm_d_inner;
}

uint32_t llama_hparams::n_embd_v_s() const {
    if (wkv_head_size != 0) {
        // corresponds to RWKV's wkv_states size
        return n_embd * wkv_head_size;
    }

    // corresponds to Mamba's ssm_states size
    return ssm_d_state * ssm_d_inner;
}