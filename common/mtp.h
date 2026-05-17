#pragma once

#include "llama.h"

#include <cstdint>

int common_mtp_assistant_load_from_file(
        llama_model * model,
        const char * path_mtp,
        llama_model_params params);

bool common_mtp_assistant_is_attached(const llama_model * model);

uint32_t common_mtp_assistant_n_embd_backbone(const llama_model * model);

void common_mtp_set_embeddings_pre_norm(llama_context * ctx, bool value);

float * common_mtp_get_embeddings_pre_norm(llama_context * ctx);

float * common_mtp_get_embeddings_pre_norm_ith(llama_context * ctx, int32_t i);

int32_t common_mtp_decode(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos attn_pos,
        llama_token last_token,
        float * h_prev,
        int32_t n_steps,
        llama_token * out_drafts,
        float * out_logits,
        float * out_h_prev_last);
