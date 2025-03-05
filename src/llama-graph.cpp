#include "llama-graph.h"

#include "llama-impl.h"

ggml_tensor * llama_graph_input_attn_i::get_kq_mask() {
    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);
    return nullptr;
}

ggml_tensor * llama_graph_input_attn_i::get_kq_mask_swa() {
    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);
    return nullptr;
}

ggml_tensor * llama_graph_input_attn_i::get_kq_mask_cross() {
    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);
    return nullptr;
}

llama_graph_i::llama_graph_i(llama_graph_type type) : type(type) {}

llama_graph_input_ptr llama_graph_i::build_inp_cross_embd(
            ggml_context * ctx0) const {
    GGML_UNUSED(ctx0);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);
    return nullptr;
}

ggml_tensor * llama_graph_i::build_attn(
        llama_graph_input_attn_i * inp,
        ggml_context * ctx0,
         ggml_cgraph * gf,
         ggml_tensor * q_cur,
         ggml_tensor * k_cur,
         ggml_tensor * v_cur,
         ggml_tensor * kq_b,
               float   kq_scale,
                 int   il) const {
    GGML_UNUSED(inp);
    GGML_UNUSED(ctx0);
    GGML_UNUSED(gf);
    GGML_UNUSED(q_cur);
    GGML_UNUSED(k_cur);
    GGML_UNUSED(v_cur);
    GGML_UNUSED(kq_b);
    GGML_UNUSED(kq_scale);
    GGML_UNUSED(il);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);
    return nullptr;
}

ggml_tensor * llama_graph_i::build_attn_cross(
        llama_graph_input_attn_i * inp,
        ggml_context * ctx0,
         ggml_cgraph * gf,
         ggml_tensor * q_cur,
         ggml_tensor * k_cur,
         ggml_tensor * v_cur,
         ggml_tensor * kq_b,
             float     kq_scale,
             int       il) const {
    GGML_UNUSED(inp);
    GGML_UNUSED(ctx0);
    GGML_UNUSED(gf);
    GGML_UNUSED(q_cur);
    GGML_UNUSED(k_cur);
    GGML_UNUSED(v_cur);
    GGML_UNUSED(kq_b);
    GGML_UNUSED(kq_scale);
    GGML_UNUSED(il);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);
    return nullptr;
}

llama_graph_input_ptr llama_graph_i::build_inp_s_copy (
            ggml_context * ctx0) const {
    GGML_UNUSED(ctx0);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

llama_graph_input_ptr llama_graph_i::build_inp_s_mask(
            ggml_context * ctx0) const {
    GGML_UNUSED(ctx0);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_copy_mask_state(
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * s,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
                 int32_t   n_state,
                 int32_t   n_seqs) const {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(gf);
    GGML_UNUSED(s);
    GGML_UNUSED(state_copy);
    GGML_UNUSED(state_mask);
    GGML_UNUSED(n_state);
    GGML_UNUSED(n_seqs);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_mamba_layer(
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * cur,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il) const {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(gf);
    GGML_UNUSED(cur);
    GGML_UNUSED(state_copy);
    GGML_UNUSED(state_mask);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(il);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_rwkv_token_shift_load(
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il) const {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(gf);
    GGML_UNUSED(state_copy);
    GGML_UNUSED(state_mask);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(il);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_rwkv_token_shift_store(
            ggml_context * ctx0,
             ggml_tensor * token_shift,
      const llama_ubatch & ubatch,
                     int   il) const {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(token_shift);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(il);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_rwkv6_time_mix(
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * cur,
             ggml_tensor * x_prev,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il) const {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(gf);
    GGML_UNUSED(cur);
    GGML_UNUSED(x_prev);
    GGML_UNUSED(state_copy);
    GGML_UNUSED(state_mask);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(il);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}
