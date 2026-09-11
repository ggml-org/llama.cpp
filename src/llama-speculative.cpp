#include "llama.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

/*
 * seq_state: Tracks the continuous lifecycle of a single sequence during a speculative burst.
 * This holds the mathematical checkpoints required to safely roll back hybrid/RNN architectures.
 */
struct seq_state {
    llama_seq_id             id;
    llama_pos                current_pos;
    llama_token              current_token;
    int32_t                  batch_idx;
    int32_t                  verify_batch_base;
    std::vector<float>       h_row; // Stores the target model's hidden states for MTP projection routing
    std::vector<llama_token> draft_tokens;
    std::vector<llama_token> accepted_tokens;
    std::vector<uint8_t>     ckpt_tgt; // Unconditional byte-level backup of target state prior to verification
    std::vector<uint8_t>     ckpt_dft; // Unconditional byte-level backup of draft state prior to verification
};

struct llama_speculative_context {
    llama_context *          ctx_tgt; // Primary target model context
    llama_context *          ctx_dft; // Draft model context (or the MTP self-drafting context)
    llama_sampler *          sampler; // Shared sampler chain (expected to be greedy for mathematical correctness)
    llama_speculative_params params;

    llama_batch batch_dft;
    llama_batch batch_vfy;
};

// Helper: Standard batch construction for draft and verification phases
static void spec_batch_add(struct llama_batch *              batch,
                           llama_token                       id,
                           llama_pos                         pos,
                           const std::vector<llama_seq_id> & seq_ids,
                           bool                              logits) {
    batch->token[batch->n_tokens]    = id;
    batch->pos[batch->n_tokens]      = pos;
    batch->n_seq_id[batch->n_tokens] = (int32_t) seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); i++) {
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    batch->logits[batch->n_tokens] = logits;
    batch->n_tokens++;
}

// Helper: MTP-specific batch construction that injects target hidden states directly into the draft pipeline
static void spec_batch_add_mtp(struct llama_batch *              batch,
                               llama_token                       id,
                               const float *                     h_row,
                               int32_t                           n_embd,
                               llama_pos                         pos,
                               const std::vector<llama_seq_id> & seq_ids,
                               bool                              logits) {
    batch->token[batch->n_tokens] = id;
    // Explicitly route the base model's h_row into the MTP context's embedding buffer
    memcpy(batch->embd + ((size_t) batch->n_tokens * n_embd), h_row, (size_t) n_embd * sizeof(float));
    batch->pos[batch->n_tokens]      = pos;
    batch->n_seq_id[batch->n_tokens] = (int32_t) seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); i++) {
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    batch->logits[batch->n_tokens] = logits;
    batch->n_tokens++;
}

struct llama_speculative_params llama_speculative_default_params(void) {
    llama_speculative_params params;
    params.n_draft   = 16;
    params.n_predict = -1;
    params.is_mtp    = false;
    return params;
}

struct llama_speculative_context * llama_speculative_init(struct llama_context *                  ctx_tgt,
                                                          struct llama_context *                  ctx_dft,
                                                          struct llama_sampler *                  sampler,
                                                          const struct llama_speculative_params * params) {
    llama_speculative_context * spec_ctx = new llama_speculative_context();
    spec_ctx->ctx_tgt                    = ctx_tgt;
    spec_ctx->ctx_dft                    = ctx_dft;
    spec_ctx->sampler                    = sampler;
    spec_ctx->params                     = *params;

    int32_t max_seqs = llama_n_seq_max(ctx_tgt);
    // Use n_embd_out to safely size buffer allocations for modern architectures where n_embd != n_embd_out
    int32_t n_embd   = params->is_mtp ? llama_model_n_embd_out(llama_get_model(ctx_tgt)) : 0;

    spec_ctx->batch_dft = llama_batch_init(params->n_draft * max_seqs, n_embd, 1);
    // MTP bypasses token evaluations, but we allocate a dummy token buffer to satisfy batch constraints
    if (params->is_mtp && spec_ctx->batch_dft.token == nullptr) {
        spec_ctx->batch_dft.token = (llama_token *) malloc(sizeof(llama_token) * params->n_draft * max_seqs);
    }
    spec_ctx->batch_vfy = llama_batch_init((params->n_draft + 1) * max_seqs, 0, 1);

    // Initialize Multi-Token Prediction routing mechanics
    if (params->is_mtp) {
        // Target context evaluates normally but extracts embeddings
        llama_set_embeddings_nextn(ctx_tgt, true, false);
        // Draft context bypasses token processing and directly evaluates projection layers
        llama_set_embeddings_nextn(ctx_dft, true, true);
    }
    return spec_ctx;
}

void llama_speculative_free(struct llama_speculative_context * spec_ctx) {
    if (spec_ctx != nullptr) {
        if (spec_ctx->params.is_mtp && spec_ctx->batch_dft.token != nullptr) {
            free(spec_ctx->batch_dft.token);
            spec_ctx->batch_dft.token = nullptr;
        }
        llama_batch_free(spec_ctx->batch_dft);
        llama_batch_free(spec_ctx->batch_vfy);
        delete spec_ctx;
    }
}

int32_t llama_speculative_decode(struct llama_speculative_context * spec_ctx,
                                 struct llama_batch *               batch,
                                 struct llama_speculative_result *  results,
                                 int32_t                            max_results) {
    // 1. SAFETY TOGGLE: Ensure MTP masked attention is OFF for linear batches
    if (spec_ctx->params.is_mtp) {
        llama_set_embeddings_nextn(spec_ctx->ctx_tgt, true, false);
    }

    // 1.A Vision & prefill detection
    const bool is_embd_batch = (batch->embd != nullptr) || (batch->token == nullptr);

    bool has_logits = false;
    if (batch->logits != nullptr) {
        for (int32_t i = 0; i < batch->n_tokens; ++i) {
            if (batch->logits[i]) {
                has_logits = true;
                break;
            }
        }
    }

    // 2. Evaluate target context on incoming batch (Prompt or accepted tokens)
    if (llama_decode(spec_ctx->ctx_tgt, *batch) != 0) {
        return 0;
    }

    // 2.A Synchronize the draft context KV cache with the incoming batch
    if (!is_embd_batch && spec_ctx->ctx_dft && spec_ctx->ctx_dft != spec_ctx->ctx_tgt && !spec_ctx->params.is_mtp) {
        if (llama_decode(spec_ctx->ctx_dft, *batch) != 0) {
            return 0;
        }
    }

    // 2.B. Early exit for vision bypass
    if (is_embd_batch || !has_logits) {
        // The batch was successfully processed into the KV cache, but no generation is needed yet.
        return 0;
    }

    // 3. Identify active sequences from batch and extract their sampled base token
    std::map<llama_seq_id, seq_state> active_seqs;
    for (int32_t i = 0; i < batch->n_tokens; i++) {
        for (int32_t j = 0; j < batch->n_seq_id[i]; j++) {
            llama_seq_id seq = batch->seq_id[i][j];
            if (active_seqs.find(seq) == active_seqs.end() || batch->pos[i] >= active_seqs[seq].current_pos) {
                active_seqs[seq].id          = seq;
                active_seqs[seq].current_pos = batch->pos[i];
                active_seqs[seq].batch_idx   = i;
            }
        }
    }

    const struct llama_model * tgt_model = llama_get_model(spec_ctx->ctx_tgt);
    const int32_t n_embd    = spec_ctx->params.is_mtp ? llama_model_n_embd_out(tgt_model) : 0;

    // Evaluate if the model uses recurrent states that prohibit zero-copy rollback
    bool is_rnn = llama_model_is_recurrent(tgt_model) || llama_model_is_hybrid(tgt_model);

    for (auto & pair : active_seqs) {
        // Evaluate native sampler chain (must be greedy) to get the base diverging token
        llama_token current_token = llama_sampler_sample(spec_ctx->sampler, spec_ctx->ctx_tgt, pair.second.batch_idx);
        llama_sampler_accept(spec_ctx->sampler, current_token);
        pair.second.current_token = current_token;
        pair.second.accepted_tokens.push_back(current_token);
        pair.second.current_pos += 1;

        // If MTP is enabled, extract the target model's hidden states to feed the first draft projection head
        if (spec_ctx->params.is_mtp) {
            pair.second.h_row.resize(n_embd);
            float * src_h = llama_get_embeddings_nextn_ith(spec_ctx->ctx_tgt, pair.second.batch_idx);
            if (src_h) {
                memcpy(pair.second.h_row.data(), src_h, (size_t) n_embd * sizeof(float));
            }
        }

        // Byte-level backups are strictly isolated to recurrent architectures now
        if (is_rnn) {
            // Unconditional byte-level backup of the mathematical state. Crucial for M-RoPE/RNN hybrid architectures.
            size_t tgt_size = llama_state_seq_get_size(spec_ctx->ctx_tgt, pair.first);
            pair.second.ckpt_tgt.resize(tgt_size);
            llama_state_seq_get_data(spec_ctx->ctx_tgt, pair.second.ckpt_tgt.data(), tgt_size, pair.first);

            if (spec_ctx->ctx_dft && !spec_ctx->params.is_mtp) {
                size_t dft_size = llama_state_seq_get_size(spec_ctx->ctx_dft, pair.first);
                pair.second.ckpt_dft.resize(dft_size);
                llama_state_seq_get_data(spec_ctx->ctx_dft, pair.second.ckpt_dft.data(), dft_size, pair.first);
            }
        }
    }

    // 4. Draft Phase

    // Wipe the stateless MTP context clean before building a new tree
    if (spec_ctx->params.is_mtp && spec_ctx->ctx_dft) {
        llama_memory_t mem_dft = llama_get_memory(spec_ctx->ctx_dft);
        for (auto & pair : active_seqs) {
            llama_memory_seq_rm(mem_dft, pair.first, 0, -1);
        }
    }

    for (int32_t i = 0; i < spec_ctx->params.n_draft; i++) {
        spec_ctx->batch_dft.n_tokens = 0;

        for (auto & pair : active_seqs) {
            llama_token draft_base = (i == 0) ? pair.second.current_token : pair.second.draft_tokens.back();
            pair.second.batch_idx  = spec_ctx->batch_dft.n_tokens;

            if (spec_ctx->params.is_mtp) {
                // Do not increment the position (+ i). Since MTP projection heads
                // are stateless and don't write to the KV cache, the cache position remains
                // frozen. Passing 'current_pos' for every draft token satisfies the Y = X + 1 check.
                spec_batch_add_mtp(&spec_ctx->batch_dft, draft_base, pair.second.h_row.data(), n_embd,
                                   pair.second.current_pos, { pair.first }, true);
            } else {
                spec_batch_add(&spec_ctx->batch_dft, draft_base, pair.second.current_pos + i, { pair.first }, true);
            }
        }

        if (llama_decode(spec_ctx->ctx_dft, spec_ctx->batch_dft) != 0) {
            break;
        }

        for (auto & pair : active_seqs) {
            llama_token draft_token = llama_sampler_sample(spec_ctx->sampler, spec_ctx->ctx_dft, pair.second.batch_idx);
            llama_sampler_accept(spec_ctx->sampler, draft_token);
            pair.second.draft_tokens.push_back(draft_token);

            // In MTP, extract the hidden states outputted by the current projection head to feed the next one
            if (spec_ctx->params.is_mtp) {
                float * next_h = llama_get_embeddings_nextn_ith(spec_ctx->ctx_dft, pair.second.batch_idx);
                if (next_h) {
                    memcpy(pair.second.h_row.data(), next_h, (size_t) n_embd * sizeof(float));
                }
            }
        }
    }

    // 5. Verify Phase
    // The base token is explicitly prepended to the verification batch to align target context
    spec_ctx->batch_vfy.n_tokens = 0;
    for (auto & pair : active_seqs) {
        pair.second.verify_batch_base = spec_ctx->batch_vfy.n_tokens;
        spec_batch_add(&spec_ctx->batch_vfy, pair.second.current_token, pair.second.current_pos, { pair.first }, true);

        for (size_t i = 0; i < pair.second.draft_tokens.size(); i++) {
            spec_batch_add(&spec_ctx->batch_vfy, pair.second.draft_tokens[i],
                           pair.second.current_pos + (llama_pos) i + 1, { pair.first }, true);
        }
    }

    if (llama_decode(spec_ctx->ctx_tgt, spec_ctx->batch_vfy) == 0) {
        for (auto & pair : active_seqs) {
            int32_t      accepted  = 0;
            const size_t n_drafted = pair.second.draft_tokens.size();

            for (size_t i = 0; i < n_drafted; i++) {
                int32_t     batch_idx    = pair.second.verify_batch_base + (int32_t) i;
                llama_token target_token = llama_sampler_sample(spec_ctx->sampler, spec_ctx->ctx_tgt, batch_idx);

                if (target_token == pair.second.draft_tokens[i]) {
                    accepted++;
                    pair.second.accepted_tokens.push_back(target_token);
                    llama_sampler_accept(spec_ctx->sampler, target_token);
                } else {
                    pair.second.accepted_tokens.push_back(target_token);
                    llama_sampler_accept(spec_ctx->sampler, target_token);
                    break;
                }
            }

            // If all tokens were accepted, sample the bonus token from the last position
            if (accepted == (int32_t) n_drafted && n_drafted > 0) {
                int32_t     last_batch_idx = pair.second.verify_batch_base + (int32_t) n_drafted;
                llama_token bonus_token    = llama_sampler_sample(spec_ctx->sampler, spec_ctx->ctx_tgt, last_batch_idx);
                pair.second.accepted_tokens.push_back(bonus_token);
                llama_sampler_accept(spec_ctx->sampler, bonus_token);

                // Synchronize ctx_dft: decode the final draft token so ctx_dft reaches current_pos + n_drafted
                if (spec_ctx->ctx_dft && spec_ctx->ctx_dft != spec_ctx->ctx_tgt && !spec_ctx->params.is_mtp) {
                    llama_batch sync_batch  = llama_batch_init(1, 0, 1);
                    sync_batch.token[0]     = pair.second.draft_tokens.back();
                    sync_batch.pos[0]       = pair.second.current_pos + (llama_pos) n_drafted;
                    sync_batch.n_seq_id[0]  = 1;
                    sync_batch.seq_id[0][0] = pair.first;
                    sync_batch.logits[0]    = false;
                    sync_batch.n_tokens     = 1;

                    llama_decode(spec_ctx->ctx_dft, sync_batch);
                    llama_batch_free(sync_batch);
                }
            }

            // KV Cache Rollback & State Recovery for rejected tokens
            if (accepted < (int32_t) n_drafted) {
                if (!is_rnn) {
                    // ZERO-COPY ROLLBACK (Transformers)
                    // The KV cache holds the base token + all evaluated draft tokens. Truncate rejected tokens instantly.
                    llama_memory_t mem_tgt = llama_get_memory(spec_ctx->ctx_tgt);
                    llama_memory_seq_rm(mem_tgt, pair.first, pair.second.current_pos + accepted + 1, -1);

                    if (spec_ctx->ctx_dft && !spec_ctx->params.is_mtp) {
                        llama_memory_t mem_dft = llama_get_memory(spec_ctx->ctx_dft);
                        llama_memory_seq_rm(mem_dft, pair.first, pair.second.current_pos + accepted + 1, -1);
                    }
                } else {
                    // FALLBACK FOR RNNs & HYBRIDs
                    llama_memory_t mem_tgt = llama_get_memory(spec_ctx->ctx_tgt);

                    // 1. Wipe invalid KV cache and restore exact state prior to verification
                    llama_memory_seq_rm(mem_tgt, pair.first, pair.second.current_pos, -1);
                    llama_state_seq_set_data(spec_ctx->ctx_tgt, pair.second.ckpt_tgt.data(),
                                             pair.second.ckpt_tgt.size(), pair.first);

                    // 2. Re-decode the base token + accepted draft tokens to safely catch the RNN Math up to speed
                    int32_t     redecode_count = 1 + accepted;
                    llama_batch fix_batch      = llama_batch_init(redecode_count, 0, 1);
                    spec_batch_add(&fix_batch, pair.second.current_token, pair.second.current_pos, { pair.first },
                                   true);

                    for (int i = 0; i < accepted; i++) {
                        spec_batch_add(&fix_batch, pair.second.draft_tokens[i], pair.second.current_pos + i + 1,
                                       { pair.first }, true);
                    }
                    llama_decode(spec_ctx->ctx_tgt, fix_batch);
                    llama_batch_free(fix_batch);

                    // 3. Do the same for the draft context ONLY if using standard speculation
                    if (spec_ctx->ctx_dft && !spec_ctx->params.is_mtp) {
                        llama_memory_t mem_dft = llama_get_memory(spec_ctx->ctx_dft);
                        llama_memory_seq_rm(mem_dft, pair.first, pair.second.current_pos, -1);
                        llama_state_seq_set_data(spec_ctx->ctx_dft, pair.second.ckpt_dft.data(),
                                                 pair.second.ckpt_dft.size(), pair.first);

                        llama_batch fix_batch_dft = llama_batch_init(redecode_count, 0, 1);
                        spec_batch_add(&fix_batch_dft, pair.second.current_token, pair.second.current_pos,
                                       { pair.first }, true);

                        for (int i = 0; i < accepted; i++) {
                            spec_batch_add(&fix_batch_dft, pair.second.draft_tokens[i], pair.second.current_pos + i + 1,
                                           { pair.first }, true);
                        }
                        llama_decode(spec_ctx->ctx_dft, fix_batch_dft);
                        llama_batch_free(fix_batch_dft);
                    }
                }
            }
        }
    }

    // 6. Populate C# Results Array (Truncated to fixed max size 32 for ABI compatibility)
    int32_t out_count = 0;
    for (auto & pair : active_seqs) {
        if (out_count >= max_results) {
            break;
        }
        results[out_count].seq_id = pair.first;
        results[out_count].count  = std::min((int32_t) pair.second.accepted_tokens.size(), 32);
        for (int32_t i = 0; i < results[out_count].count; i++) {
            results[out_count].tokens[i] = pair.second.accepted_tokens[i];
        }
        out_count++;
    }

    return out_count;
}

