#pragma once

// CPUOPTI: Context block deduplication for KV cache
// Eliminates redundant KV projections for repeated context blocks.
// Tier 1 — Exact: bitwise identical outputs.

#include "llama.h"
#include "llama-context-hash.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

//
// A deduplicated context block entry
//

struct llama_opt_kv_block {
    uint64_t                 hash;         // FNV-1a hash of the token block
    uint32_t                 n_tokens;     // Number of tokens in this block
    std::vector<llama_token> tokens;       // Actual tokens (for collision verification)
    uint32_t                 ref_count;    // Active sequence references
    uint64_t                 last_access;  // Turn counter for LRU eviction
};

//
// KV slot dedup status
//

struct llama_opt_kv_slot_status {
    enum status_type {
        OWNED,  // Slot has its own computed K/V tensors
        DEDUP,  // Slot references a canonical block
    };

    status_type              status       = OWNED;
    const llama_opt_kv_block * dedup_source = nullptr;
};

//
// Block deduplication pool
//

class llama_opt_block_pool {
public:
    explicit llama_opt_block_pool(uint32_t max_blocks);

    // Look up a block by hash. Returns nullptr if not found.
    // On hit, verifies tokens match exactly (collision resistance).
    const llama_opt_kv_block * lookup(
        uint64_t             hash,
        const llama_token  * tokens,
        uint32_t             n_tokens);

    // Insert a new block into the pool. Returns pointer to the inserted block.
    // If pool is full, evicts the least-recently-used unreferenced block.
    llama_opt_kv_block * insert(
        uint64_t             hash,
        const llama_token  * tokens,
        uint32_t             n_tokens);

    // Reference counting
    void ref(const llama_opt_kv_block * block);
    void unref(const llama_opt_kv_block * block);

    // Evict unreferenced blocks older than min_turn
    // Returns number of blocks evicted
    uint32_t evict_stale(uint64_t min_turn);

    // Advance the turn counter (call once per turn)
    void advance_turn();

    // Pool state
    uint32_t size() const;
    uint32_t capacity() const;
    uint64_t current_turn() const;

private:
    uint32_t max_blocks_;
    uint64_t turn_counter_ = 0;

    // Hash → list of blocks (handles hash collisions via chaining)
    std::unordered_map<uint64_t, std::vector<llama_opt_kv_block>> blocks_;
    uint32_t total_blocks_ = 0;

    // Find a specific block within the chain for a given hash
    llama_opt_kv_block * find_exact(
        uint64_t             hash,
        const llama_token  * tokens,
        uint32_t             n_tokens);

    // Evict one LRU unreferenced block to make room
    bool evict_one();
};

//
// Dedup analysis result for a context
//

struct llama_opt_dedup_plan {
    struct block_action {
        uint32_t block_index;  // Index in the segmented block list
        bool     is_hit;       // True if this block is in the pool
        const llama_opt_kv_block * source; // Non-null if hit
    };

    std::vector<block_action> actions;
    uint32_t                  blocks_total = 0;
    uint32_t                  blocks_hit   = 0;
    uint32_t                  tokens_saved = 0;
};

// Analyze a context for dedup opportunities
// Does NOT modify the pool — call pool.insert() separately for misses
llama_opt_dedup_plan llama_opt_dedup_analyze(
    llama_opt_block_pool & pool,
    const llama_token    * tokens,
    uint32_t               n_tokens,
    uint32_t               block_size);
