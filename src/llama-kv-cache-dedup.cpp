// CPUOPTI: Context block deduplication for KV cache

#include "llama-kv-cache-dedup.h"
#include "llama-impl.h"

#include <algorithm>
#include <limits>

//
// llama_opt_block_pool
//

llama_opt_block_pool::llama_opt_block_pool(uint32_t max_blocks)
    : max_blocks_(max_blocks) {}

const llama_opt_kv_block * llama_opt_block_pool::lookup(
        uint64_t             hash,
        const llama_token  * tokens,
        uint32_t             n_tokens) {

    auto it = blocks_.find(hash);
    if (it == blocks_.end()) {
        return nullptr;
    }

    // Search chain for exact token match
    for (auto & block : it->second) {
        if (block.n_tokens != n_tokens) {
            continue;
        }
        // Verify tokens match exactly
        bool match = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (block.tokens[i] != tokens[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            // Update access time (const_cast needed because lookup is logically const
            // but LRU tracking requires mutation)
            const_cast<llama_opt_kv_block &>(block).last_access = turn_counter_;
            return &block;
        }
    }

    return nullptr;
}

llama_opt_kv_block * llama_opt_block_pool::insert(
        uint64_t             hash,
        const llama_token  * tokens,
        uint32_t             n_tokens) {

    // Make room if necessary
    while (total_blocks_ >= max_blocks_) {
        if (!evict_one()) {
            // All blocks are referenced, cannot evict
            LLAMA_LOG_WARN("%s: CPUOPTI dedup pool full, all blocks referenced\n", __func__);
            return nullptr;
        }
    }

    auto & chain = blocks_[hash];

    llama_opt_kv_block block;
    block.hash        = hash;
    block.n_tokens    = n_tokens;
    block.tokens.assign(tokens, tokens + n_tokens);
    block.ref_count   = 0;
    block.last_access = turn_counter_;

    chain.push_back(std::move(block));
    total_blocks_++;

    return &chain.back();
}

void llama_opt_block_pool::ref(const llama_opt_kv_block * block) {
    if (block) {
        const_cast<llama_opt_kv_block *>(block)->ref_count++;
    }
}

void llama_opt_block_pool::unref(const llama_opt_kv_block * block) {
    if (block && block->ref_count > 0) {
        const_cast<llama_opt_kv_block *>(block)->ref_count--;
    }
}

uint32_t llama_opt_block_pool::evict_stale(uint64_t min_turn) {
    uint32_t evicted = 0;

    for (auto it = blocks_.begin(); it != blocks_.end(); ) {
        auto & chain = it->second;
        for (auto bit = chain.begin(); bit != chain.end(); ) {
            if (bit->ref_count == 0 && bit->last_access < min_turn) {
                bit = chain.erase(bit);
                total_blocks_--;
                evicted++;
            } else {
                ++bit;
            }
        }
        if (chain.empty()) {
            it = blocks_.erase(it);
        } else {
            ++it;
        }
    }

    return evicted;
}

void llama_opt_block_pool::advance_turn() {
    turn_counter_++;
}

uint32_t llama_opt_block_pool::size() const {
    return total_blocks_;
}

uint32_t llama_opt_block_pool::capacity() const {
    return max_blocks_;
}

uint64_t llama_opt_block_pool::current_turn() const {
    return turn_counter_;
}

llama_opt_kv_block * llama_opt_block_pool::find_exact(
        uint64_t             hash,
        const llama_token  * tokens,
        uint32_t             n_tokens) {

    auto it = blocks_.find(hash);
    if (it == blocks_.end()) {
        return nullptr;
    }

    for (auto & block : it->second) {
        if (block.n_tokens != n_tokens) {
            continue;
        }
        bool match = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (block.tokens[i] != tokens[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            return &block;
        }
    }

    return nullptr;
}

bool llama_opt_block_pool::evict_one() {
    // Find the LRU unreferenced block across all chains
    llama_opt_kv_block * lru_block = nullptr;
    uint64_t             lru_hash  = 0;
    uint64_t             lru_time  = std::numeric_limits<uint64_t>::max();

    for (auto & [hash, chain] : blocks_) {
        for (auto & block : chain) {
            if (block.ref_count == 0 && block.last_access < lru_time) {
                lru_time  = block.last_access;
                lru_block = &block;
                lru_hash  = hash;
            }
        }
    }

    if (lru_block == nullptr) {
        return false;
    }

    // Remove the LRU block
    auto & chain = blocks_[lru_hash];
    for (auto it = chain.begin(); it != chain.end(); ++it) {
        if (&(*it) == lru_block) {
            chain.erase(it);
            total_blocks_--;
            break;
        }
    }
    if (chain.empty()) {
        blocks_.erase(lru_hash);
    }

    return true;
}

//
// Dedup analysis
//

llama_opt_dedup_plan llama_opt_dedup_analyze(
        llama_opt_block_pool & pool,
        const llama_token    * tokens,
        uint32_t               n_tokens,
        uint32_t               block_size) {

    llama_opt_dedup_plan plan;

    auto blocks = llama_opt_segment_and_hash(tokens, n_tokens, block_size);

    plan.blocks_total = (uint32_t) blocks.size();

    for (uint32_t i = 0; i < blocks.size(); i++) {
        const auto & bh = blocks[i];

        const llama_opt_kv_block * found = pool.lookup(
            bh.hash, tokens + bh.offset, bh.n_tokens);

        llama_opt_dedup_plan::block_action action;
        action.block_index = i;
        action.is_hit      = (found != nullptr);
        action.source      = found;

        if (found) {
            plan.blocks_hit++;
            plan.tokens_saved += bh.n_tokens;
        }

        plan.actions.push_back(action);
    }

    return plan;
}
