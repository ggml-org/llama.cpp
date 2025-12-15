#include "llama-pp-scheduler.h"
#include "llama-impl.h"

#include <algorithm>
#include <cassert>
#include <map>

// Helper to add a token to a batch (matches common_batch_add from common/)
static void pp_batch_add(
    llama_batch & batch,
    llama_token id,
    llama_pos pos,
    llama_seq_id seq_id,
    bool logits
) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id  [batch.n_tokens][0] = seq_id;
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

//
// llama_pp_scheduler implementation
//

llama_pp_scheduler::llama_pp_scheduler(int32_t chunk_size, bool chunked_prefill_enabled)
    : chunk_size(chunk_size)
    , chunked_prefill_enabled(chunked_prefill_enabled)
{
    if (chunk_size > 0 && chunked_prefill_enabled) {
        LLAMA_LOG_INFO("%s: chunked prefill enabled with chunk_size=%d\n", __func__, chunk_size);
    }
}

void llama_pp_scheduler::set_chunk_size(int32_t new_chunk_size) {
    chunk_size = new_chunk_size;
}

int32_t llama_pp_scheduler::submit_prefill(
    llama_seq_id seq_id,
    const llama_token * tokens,
    int32_t n_tokens,
    llama_pos pos_start
) {
    if (n_tokens <= 0) {
        return 0;
    }

    auto chunks = chunk_prompt(seq_id, tokens, n_tokens, pos_start);

    for (auto & chunk : chunks) {
        prefill_queue.push_back(std::move(chunk));
    }

    scheduler_stats.total_prefill_tokens += n_tokens;
    scheduler_stats.total_chunks += chunks.size();

    return static_cast<int32_t>(chunks.size());
}

void llama_pp_scheduler::submit_decode(
    llama_seq_id seq_id,
    llama_token token,
    llama_pos pos
) {
    llama_pp_request req;
    req.type = LLAMA_PP_REQUEST_DECODE;
    req.seq_id = seq_id;
    req.last_token = token;
    req.pos = pos;
    req.is_last_chunk = true;  // Decode requests are always "complete"

    decode_queue.push_back(std::move(req));
    scheduler_stats.total_decode_tokens++;
}

bool llama_pp_scheduler::get_next_batch(
    llama_batch & batch,
    int32_t max_tokens,
    std::vector<llama_pp_request> & requests_out
) {
    requests_out.clear();

    if (!has_pending_work()) {
        return false;
    }

    int32_t tokens_in_batch = 0;

    // Decode-first policy: Process all pending decode requests first
    // This minimizes latency for ongoing generations
    while (!decode_queue.empty() && tokens_in_batch < max_tokens) {
        auto & req = decode_queue.front();

        // Each decode request is 1 token
        if (tokens_in_batch + 1 > max_tokens) {
            break;
        }

        // Add to batch
        pp_batch_add(batch, req.last_token, req.pos, req.seq_id, true);
        requests_out.push_back(std::move(req));
        decode_queue.pop_front();
        tokens_in_batch++;
    }

    // If there's room and decode queue is empty, add prefill chunks
    // This implements the "interleaving" aspect of chunked prefill
    while (!prefill_queue.empty() && tokens_in_batch < max_tokens) {
        auto & req = prefill_queue.front();

        int32_t req_tokens = static_cast<int32_t>(req.tokens.size());
        if (tokens_in_batch + req_tokens > max_tokens) {
            // Can't fit this chunk, wait for next batch
            // Note: In a more sophisticated implementation, we could split
            // the chunk further, but for simplicity we wait
            break;
        }

        // Add all tokens from this chunk to the batch
        for (size_t i = 0; i < req.tokens.size(); i++) {
            llama_pos pos = req.pos_start + static_cast<llama_pos>(i);
            bool is_last = (i == req.tokens.size() - 1) && req.is_last_chunk;
            pp_batch_add(batch, req.tokens[i], pos, req.seq_id, is_last);
        }

        requests_out.push_back(std::move(req));
        prefill_queue.pop_front();
        tokens_in_batch += req_tokens;
    }

    if (tokens_in_batch > 0) {
        scheduler_stats.batches_processed++;
    }

    return tokens_in_batch > 0;
}

bool llama_pp_scheduler::has_pending_work() const {
    return !decode_queue.empty() || !prefill_queue.empty();
}

void llama_pp_scheduler::clear() {
    decode_queue.clear();
    prefill_queue.clear();
}

void llama_pp_scheduler::reset_stats() {
    scheduler_stats = stats{};
}

std::vector<llama_pp_request> llama_pp_scheduler::chunk_prompt(
    llama_seq_id seq_id,
    const llama_token * tokens,
    int32_t n_tokens,
    llama_pos pos_start
) {
    std::vector<llama_pp_request> chunks;

    // If chunking is disabled or prompt fits in one chunk, create single request
    if (!chunked_prefill_enabled || chunk_size <= 0 || n_tokens <= chunk_size) {
        llama_pp_request req;
        req.type = LLAMA_PP_REQUEST_PREFILL;
        req.seq_id = seq_id;
        req.tokens.assign(tokens, tokens + n_tokens);
        req.pos_start = pos_start;
        req.is_last_chunk = true;
        chunks.push_back(std::move(req));
        return chunks;
    }

    // Split into multiple chunks
    int32_t offset = 0;
    while (offset < n_tokens) {
        int32_t chunk_tokens = std::min(chunk_size, n_tokens - offset);

        llama_pp_request req;
        req.type = LLAMA_PP_REQUEST_PREFILL;
        req.seq_id = seq_id;
        req.tokens.assign(tokens + offset, tokens + offset + chunk_tokens);
        req.pos_start = pos_start + offset;
        req.is_last_chunk = (offset + chunk_tokens >= n_tokens);

        chunks.push_back(std::move(req));
        offset += chunk_tokens;
    }

    LLAMA_LOG_DEBUG("%s: chunked prompt of %d tokens into %zu chunks\n",
        __func__, n_tokens, chunks.size());

    return chunks;
}

//
// Utility functions
//

namespace llama_pp {

std::vector<llama_batch> chunk_batch(
    const llama_batch & batch,
    int32_t chunk_size
) {
    std::vector<llama_batch> chunks;

    if (chunk_size <= 0 || batch.n_tokens <= chunk_size) {
        // Return a copy of the original batch
        // Note: In production, we'd want to avoid this copy
        chunks.push_back(batch);
        return chunks;
    }

    // For now, simple chunking - in practice, this should consider
    // sequence boundaries to avoid splitting mid-sequence
    int32_t offset = 0;
    while (offset < batch.n_tokens) {
        int32_t n = std::min(chunk_size, batch.n_tokens - offset);

        llama_batch chunk = llama_batch_init(n, 0, 1);
        for (int32_t i = 0; i < n; i++) {
            chunk.token[i] = batch.token[offset + i];
            chunk.pos[i] = batch.pos[offset + i];
            chunk.n_seq_id[i] = batch.n_seq_id[offset + i];
            for (int32_t j = 0; j < batch.n_seq_id[offset + i]; j++) {
                chunk.seq_id[i][j] = batch.seq_id[offset + i][j];
            }
            chunk.logits[i] = batch.logits[offset + i];
        }
        chunk.n_tokens = n;

        chunks.push_back(chunk);
        offset += n;
    }

    return chunks;
}

bool is_prefill_batch(const llama_batch & batch) {
    if (batch.n_tokens <= 1) {
        return false;
    }

    // Check if multiple tokens belong to the same sequence
    // A prefill batch typically has multiple tokens for one sequence
    if (batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return batch.n_tokens > 1;
    }

    // Count tokens per sequence
    std::map<llama_seq_id, int32_t> tokens_per_seq;
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            tokens_per_seq[batch.seq_id[i][j]]++;
        }
    }

    // If any sequence has more than 1 token, it's a prefill
    for (auto it = tokens_per_seq.begin(); it != tokens_per_seq.end(); ++it) {
        if (it->second > 1) {
            return true;
        }
    }

    return false;
}

bool is_decode_batch(const llama_batch & batch) {
    // A decode batch has exactly one token per sequence
    if (batch.n_tokens == 0) {
        return false;
    }

    if (batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return batch.n_tokens == 1;
    }

    // Count tokens per sequence
    std::map<llama_seq_id, int32_t> tokens_per_seq;
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            tokens_per_seq[batch.seq_id[i][j]]++;
        }
    }

    // All sequences should have exactly 1 token
    for (auto it = tokens_per_seq.begin(); it != tokens_per_seq.end(); ++it) {
        if (it->second != 1) {
            return false;
        }
    }

    return true;
}

} // namespace llama_pp
