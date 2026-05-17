#include "mtp.h"

LLAMA_API int llama_model_load_mtp_from_file(
        llama_model * model,
        const char * path_mtp,
        llama_model_params params);

LLAMA_API bool llama_model_has_mtp_assistant(const llama_model * model);

LLAMA_API uint32_t llama_model_mtp_n_embd_backbone(const llama_model * model);

LLAMA_API void llama_set_embeddings_pre_norm(llama_context * ctx, bool value);

LLAMA_API float * llama_get_embeddings_pre_norm(llama_context * ctx);

LLAMA_API float * llama_get_embeddings_pre_norm_ith(llama_context * ctx, int32_t i);

LLAMA_API int32_t llama_decode_mtp(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos attn_pos,
        llama_token last_token,
        float * h_prev,
        int32_t n_steps,
        llama_token * out_drafts,
        float * out_logits,
        float * out_h_prev_last);

int common_mtp_assistant_load_from_file(
        llama_model * model,
        const char * path_mtp,
        llama_model_params params) {
    return llama_model_load_mtp_from_file(model, path_mtp, params);
}

bool common_mtp_assistant_is_attached(const llama_model * model) {
    return llama_model_has_mtp_assistant(model);
}

uint32_t common_mtp_assistant_n_embd_backbone(const llama_model * model) {
    return llama_model_mtp_n_embd_backbone(model);
}

void common_mtp_set_embeddings_pre_norm(llama_context * ctx, bool value) {
    llama_set_embeddings_pre_norm(ctx, value);
}

float * common_mtp_get_embeddings_pre_norm(llama_context * ctx) {
    return llama_get_embeddings_pre_norm(ctx);
}

float * common_mtp_get_embeddings_pre_norm_ith(llama_context * ctx, int32_t i) {
    return llama_get_embeddings_pre_norm_ith(ctx, i);
}

int32_t common_mtp_decode(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos attn_pos,
        llama_token last_token,
        float * h_prev,
        int32_t n_steps,
        llama_token * out_drafts,
        float * out_logits,
        float * out_h_prev_last) {
    return llama_decode_mtp(ctx, seq_id, attn_pos, last_token, h_prev, n_steps, out_drafts, out_logits, out_h_prev_last);
}
