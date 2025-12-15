#pragma once

#include "llama.h"
#include "llama-batch.h"

#include <deque>
#include <vector>
#include <cstdint>

// Forward declarations
struct llama_context;

//
// Pipeline Parallelism Scheduler (vLLM-style)
//
// Implements:
// 1. Chunked prefill: Split large prompts into smaller chunks
// 2. Decode-first scheduling: Prioritize decode requests over prefill chunks
// 3. Batch interleaving: Mix decode and prefill in same batch when possible
//

// Request types for scheduling
enum llama_pp_request_type {
    LLAMA_PP_REQUEST_PREFILL,  // Initial prompt processing
    LLAMA_PP_REQUEST_DECODE,   // Token generation
};

// A scheduled request (either prefill chunk or decode)
struct llama_pp_request {
    llama_pp_request_type type;
    llama_seq_id seq_id;

    // For prefill chunks
    std::vector<llama_token> tokens;
    llama_pos pos_start;     // Starting position in sequence
    bool is_last_chunk;      // True if this is the final chunk of a prompt

    // For decode requests
    llama_token last_token;  // Token to decode (for decode requests)
    llama_pos pos;           // Position (for decode requests)
};

// Scheduler state for managing prefill/decode interleaving
class llama_pp_scheduler {
public:
    llama_pp_scheduler(int32_t chunk_size, bool chunked_prefill_enabled);
    ~llama_pp_scheduler() = default;

    // Configuration
    void set_chunk_size(int32_t chunk_size);
    int32_t get_chunk_size() const { return chunk_size; }
    bool is_chunked_prefill_enabled() const { return chunked_prefill_enabled; }

    // Submit a new prefill request (will be chunked if necessary)
    // Returns the number of chunks created
    int32_t submit_prefill(
        llama_seq_id seq_id,
        const llama_token * tokens,
        int32_t n_tokens,
        llama_pos pos_start = 0
    );

    // Submit a decode request (single token generation)
    void submit_decode(
        llama_seq_id seq_id,
        llama_token token,
        llama_pos pos
    );

    // Get the next batch to process (decode-first policy)
    // Returns true if there's work to do, false if queues are empty
    // max_tokens: maximum tokens to include in batch
    bool get_next_batch(
        llama_batch & batch,
        int32_t max_tokens,
        std::vector<llama_pp_request> & requests_out
    );

    // Check if there are pending requests
    bool has_pending_work() const;

    // Get queue sizes for monitoring
    size_t prefill_queue_size() const { return prefill_queue.size(); }
    size_t decode_queue_size() const { return decode_queue.size(); }

    // Clear all queues
    void clear();

    // Statistics
    struct stats {
        int64_t total_prefill_tokens = 0;
        int64_t total_decode_tokens = 0;
        int64_t total_chunks = 0;
        int64_t batches_processed = 0;
    };
    const stats & get_stats() const { return scheduler_stats; }
    void reset_stats();

private:
    int32_t chunk_size;           // Max tokens per prefill chunk (0 = no chunking)
    bool chunked_prefill_enabled; // Whether chunked prefill is active

    // Request queues (decode-first policy means decode queue is drained first)
    std::deque<llama_pp_request> decode_queue;
    std::deque<llama_pp_request> prefill_queue;

    stats scheduler_stats;

    // Helper to chunk a prompt into multiple requests
    std::vector<llama_pp_request> chunk_prompt(
        llama_seq_id seq_id,
        const llama_token * tokens,
        int32_t n_tokens,
        llama_pos pos_start
    );
};

// Utility functions for PP scheduling
namespace llama_pp {

// Split a batch into chunks suitable for chunked prefill
// Returns a vector of batches, each with at most chunk_size tokens
std::vector<llama_batch> chunk_batch(
    const llama_batch & batch,
    int32_t chunk_size
);

// Check if a batch is a prefill batch (multiple tokens for same sequence)
bool is_prefill_batch(const llama_batch & batch);

// Check if a batch is a decode batch (single token per sequence)
bool is_decode_batch(const llama_batch & batch);

} // namespace llama_pp
