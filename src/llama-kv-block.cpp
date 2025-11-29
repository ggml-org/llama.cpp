#include "llama-kv-block.h"

#include <algorithm>
#include <cstring>

// =============================================================================
// llama_kv_block_manager implementation
// =============================================================================

llama_kv_block_manager::llama_kv_block_manager(uint32_t n_blocks, uint32_t block_size)
    : n_blocks(n_blocks), block_size(block_size) {

    blocks.resize(n_blocks);

    // Initialize all blocks and add to free list
    for (uint32_t i = 0; i < n_blocks; ++i) {
        blocks[i].block_id = i;
        blocks[i].ref_count = 0;
        blocks[i].is_free = true;
        blocks[i].content_hash = 0;
        free_list.push(i);
    }
}

int32_t llama_kv_block_manager::allocate() {
    if (free_list.empty()) {
        return -1;  // Out of memory
    }

    uint32_t block_id = free_list.front();
    free_list.pop();

    blocks[block_id].is_free = false;
    blocks[block_id].ref_count = 1;
    blocks[block_id].content_hash = 0;

    return static_cast<int32_t>(block_id);
}

void llama_kv_block_manager::free(uint32_t block_id) {
    assert(block_id < n_blocks);
    assert(!blocks[block_id].is_free);

    if (blocks[block_id].ref_count > 0) {
        blocks[block_id].ref_count--;
    }

    if (blocks[block_id].ref_count == 0) {
        blocks[block_id].is_free = true;
        blocks[block_id].content_hash = 0;
        free_list.push(block_id);
    }
}

void llama_kv_block_manager::ref_inc(uint32_t block_id) {
    assert(block_id < n_blocks);
    assert(!blocks[block_id].is_free);

    blocks[block_id].ref_count++;
}

bool llama_kv_block_manager::ref_dec(uint32_t block_id) {
    assert(block_id < n_blocks);
    assert(!blocks[block_id].is_free);
    assert(blocks[block_id].ref_count > 0);

    blocks[block_id].ref_count--;

    if (blocks[block_id].ref_count == 0) {
        blocks[block_id].is_free = true;
        blocks[block_id].content_hash = 0;
        free_list.push(block_id);
        return true;  // Block was freed
    }

    return false;
}

uint32_t llama_kv_block_manager::get_ref_count(uint32_t block_id) const {
    assert(block_id < n_blocks);
    return blocks[block_id].ref_count;
}

std::vector<int32_t> llama_kv_block_manager::allocate_n(uint32_t n) {
    if (get_free_blocks() < n) {
        return {};  // Not enough blocks
    }

    std::vector<int32_t> result;
    result.reserve(n);

    for (uint32_t i = 0; i < n; ++i) {
        int32_t block_id = allocate();
        assert(block_id >= 0);  // Should never fail since we checked above
        result.push_back(block_id);
    }

    return result;
}

void llama_kv_block_manager::free_n(const std::vector<int32_t> & block_ids) {
    for (int32_t block_id : block_ids) {
        if (block_id >= 0) {
            free(static_cast<uint32_t>(block_id));
        }
    }
}

void llama_kv_block_manager::reset() {
    // Clear free list
    while (!free_list.empty()) {
        free_list.pop();
    }

    // Reset all blocks and rebuild free list
    for (uint32_t i = 0; i < n_blocks; ++i) {
        blocks[i].ref_count = 0;
        blocks[i].is_free = true;
        blocks[i].content_hash = 0;
        free_list.push(i);
    }
}

float llama_kv_block_manager::get_fragmentation_ratio() const {
    if (n_blocks == 0) {
        return 0.0f;
    }

    const uint32_t used = get_used_blocks();
    if (used == 0) {
        return 0.0f;  // No blocks used, no fragmentation
    }

    // Find the highest used block index
    uint32_t max_used_idx = 0;
    for (uint32_t i = 0; i < n_blocks; ++i) {
        if (!blocks[i].is_free) {
            max_used_idx = i;
        }
    }

    // Fragmentation = 1 - (used_blocks / (max_used_idx + 1))
    // If all used blocks are contiguous at the start, fragmentation = 0
    // If used blocks are scattered, fragmentation approaches 1
    const uint32_t occupied_range = max_used_idx + 1;
    if (occupied_range <= used) {
        return 0.0f;  // No gaps, perfect compaction
    }

    return 1.0f - (static_cast<float>(used) / static_cast<float>(occupied_range));
}

bool llama_kv_block_manager::needs_defrag(float threshold) const {
    return get_fragmentation_ratio() > threshold;
}

std::vector<llama_kv_block_manager::defrag_move> llama_kv_block_manager::compute_defrag_plan() const {
    std::vector<defrag_move> moves;

    const uint32_t used = get_used_blocks();
    if (used == 0) {
        return moves;  // Nothing to defrag
    }

    // Goal: Move all used blocks to indices [0, used-1]
    // Strategy: Find gaps (free blocks with index < used) and fill from high indices

    // Find free slots in the target range [0, used-1]
    std::vector<uint32_t> free_slots;
    for (uint32_t i = 0; i < used && i < n_blocks; ++i) {
        if (blocks[i].is_free) {
            free_slots.push_back(i);
        }
    }

    // Find used blocks with index >= used (these need to move)
    std::vector<uint32_t> high_blocks;
    for (uint32_t i = used; i < n_blocks; ++i) {
        if (!blocks[i].is_free) {
            high_blocks.push_back(i);
        }
    }

    // Match free slots with high blocks
    const size_t n_moves = std::min(free_slots.size(), high_blocks.size());
    for (size_t i = 0; i < n_moves; ++i) {
        moves.push_back({high_blocks[i], free_slots[i]});
    }

    return moves;
}

void llama_kv_block_manager::apply_defrag(const std::vector<defrag_move> & moves) {
    // Clear and rebuild free list based on updated state
    while (!free_list.empty()) {
        free_list.pop();
    }

    // Apply the moves to block metadata
    for (const auto & move : moves) {
        // Update hash table if prefix caching is enabled
        if (prefix_caching_enabled && blocks[move.src_block].content_hash != 0) {
            // Remove old hash->block mapping
            hash_to_block.erase(blocks[move.src_block].content_hash);
            // Add new hash->block mapping
            hash_to_block[blocks[move.src_block].content_hash] = move.dst_block;
        }

        // Copy block metadata from src to dst
        blocks[move.dst_block].ref_count = blocks[move.src_block].ref_count;
        blocks[move.dst_block].is_free = false;
        blocks[move.dst_block].content_hash = blocks[move.src_block].content_hash;

        // Mark src as free
        blocks[move.src_block].ref_count = 0;
        blocks[move.src_block].is_free = true;
        blocks[move.src_block].content_hash = 0;
    }

    // Rebuild free list with all free blocks
    for (uint32_t i = 0; i < n_blocks; ++i) {
        if (blocks[i].is_free) {
            free_list.push(i);
        }
    }
}

// =============================================================================
// Prefix Caching implementation
// =============================================================================

void llama_kv_block_manager::set_content_hash(uint32_t block_id, uint64_t hash) {
    assert(block_id < n_blocks);
    assert(!blocks[block_id].is_free);

    // Remove old hash mapping if it exists
    uint64_t old_hash = blocks[block_id].content_hash;
    if (old_hash != 0 && prefix_caching_enabled) {
        auto it = hash_to_block.find(old_hash);
        if (it != hash_to_block.end() && it->second == block_id) {
            hash_to_block.erase(it);
        }
    }

    // Set the new hash
    blocks[block_id].content_hash = hash;

    // Add to hash table if prefix caching is enabled and hash is non-zero
    if (prefix_caching_enabled && hash != 0) {
        hash_to_block[hash] = block_id;
    }
}

uint64_t llama_kv_block_manager::get_content_hash(uint32_t block_id) const {
    assert(block_id < n_blocks);
    return blocks[block_id].content_hash;
}

int32_t llama_kv_block_manager::find_block_by_hash(uint64_t hash) const {
    if (!prefix_caching_enabled || hash == 0) {
        return -1;
    }

    auto it = hash_to_block.find(hash);
    if (it != hash_to_block.end()) {
        uint32_t block_id = it->second;
        // Verify the block is still valid and has the expected hash
        if (!blocks[block_id].is_free && blocks[block_id].content_hash == hash) {
            return static_cast<int32_t>(block_id);
        }
    }

    return -1;
}

std::pair<int32_t, bool> llama_kv_block_manager::allocate_or_share(uint64_t hash) {
    // First, try to find an existing block with the same hash
    if (prefix_caching_enabled && hash != 0) {
        int32_t existing = find_block_by_hash(hash);
        if (existing >= 0) {
            // Found a matching block, increment its ref count
            ref_inc(static_cast<uint32_t>(existing));
            prefix_cache_hits++;
            return {existing, true};
        }
    }

    // No matching block found, allocate a new one
    prefix_cache_misses++;
    int32_t new_block = allocate();
    if (new_block >= 0 && hash != 0) {
        // Set the hash for the new block (will be registered when prefix caching is enabled)
        set_content_hash(static_cast<uint32_t>(new_block), hash);
    }
    return {new_block, false};
}

// =============================================================================
// llama_kv_cache_paged implementation
// =============================================================================

llama_kv_cache_paged::llama_kv_cache_paged(
    uint32_t n_ctx,
    uint32_t n_seq_max,
    uint32_t block_size
) : n_ctx(n_ctx),
    n_seq_max(n_seq_max),
    block_size(block_size),
    max_blocks_per_seq(llama_kv_block_table::tokens_to_blocks(n_ctx, block_size)),
    block_manager(
        // Total blocks = enough for all sequences at max length
        // But in practice, we share blocks, so this is an upper bound
        // For now, allocate blocks for n_ctx total (shared across all sequences)
        llama_kv_block_table::tokens_to_blocks(n_ctx, block_size),
        block_size
    ) {
    seq_tables.reserve(n_seq_max);
}

bool llama_kv_cache_paged::allocate_for_sequence(llama_seq_id seq_id, uint32_t n_tokens) {
    const uint32_t n_blocks_needed = llama_kv_block_table::tokens_to_blocks(n_tokens, block_size);

    // Check if we have enough blocks
    if (!block_manager.can_allocate(n_blocks_needed)) {
        return false;
    }

    // Create or get the block table for this sequence
    auto & table = seq_tables[seq_id];
    table.set_seq_id(seq_id);
    table.clear();

    // Allocate blocks
    std::vector<int32_t> new_blocks = block_manager.allocate_n(n_blocks_needed);
    if (new_blocks.empty() && n_blocks_needed > 0) {
        return false;
    }

    for (int32_t block_id : new_blocks) {
        table.append_block(block_id);
    }

    table.set_num_tokens(n_tokens);
    return true;
}

bool llama_kv_cache_paged::extend_sequence(llama_seq_id seq_id, uint32_t n_tokens) {
    auto it = seq_tables.find(seq_id);
    if (it == seq_tables.end()) {
        // Sequence doesn't exist, create it
        return allocate_for_sequence(seq_id, n_tokens);
    }

    auto & table = it->second;
    const uint32_t current_tokens = table.get_num_tokens();
    const uint32_t new_total = current_tokens + n_tokens;

    const uint32_t current_blocks = table.num_blocks();
    const uint32_t needed_blocks = llama_kv_block_table::tokens_to_blocks(new_total, block_size);

    // Allocate additional blocks if needed
    if (needed_blocks > current_blocks) {
        const uint32_t additional = needed_blocks - current_blocks;
        if (!block_manager.can_allocate(additional)) {
            return false;
        }

        std::vector<int32_t> new_blocks = block_manager.allocate_n(additional);
        for (int32_t block_id : new_blocks) {
            table.append_block(block_id);
        }
    }

    table.set_num_tokens(new_total);
    return true;
}

void llama_kv_cache_paged::free_sequence(llama_seq_id seq_id) {
    auto it = seq_tables.find(seq_id);
    if (it == seq_tables.end()) {
        return;
    }

    // Free all blocks
    const auto & block_ids = it->second.get_block_ids();
    for (int32_t block_id : block_ids) {
        if (block_id >= 0) {
            block_manager.free(static_cast<uint32_t>(block_id));
        }
    }

    seq_tables.erase(it);
}

bool llama_kv_cache_paged::copy_sequence(llama_seq_id src_seq_id, llama_seq_id dst_seq_id) {
    auto it = seq_tables.find(src_seq_id);
    if (it == seq_tables.end()) {
        return false;
    }

    const auto & src_table = it->second;

    // Create destination table
    auto & dst_table = seq_tables[dst_seq_id];
    dst_table.set_seq_id(dst_seq_id);
    dst_table.clear();

    // Copy-on-write: increment ref counts instead of copying data
    const auto & src_block_ids = src_table.get_block_ids();
    for (int32_t block_id : src_block_ids) {
        if (block_id >= 0) {
            block_manager.ref_inc(static_cast<uint32_t>(block_id));
            dst_table.append_block(block_id);
        }
    }

    dst_table.set_num_tokens(src_table.get_num_tokens());
    return true;
}

const llama_kv_block_table * llama_kv_cache_paged::get_block_table(llama_seq_id seq_id) const {
    auto it = seq_tables.find(seq_id);
    if (it == seq_tables.end()) {
        return nullptr;
    }
    return &it->second;
}

llama_kv_block_table * llama_kv_cache_paged::get_block_table(llama_seq_id seq_id) {
    auto it = seq_tables.find(seq_id);
    if (it == seq_tables.end()) {
        return nullptr;
    }
    return &it->second;
}

bool llama_kv_cache_paged::has_sequence(llama_seq_id seq_id) const {
    return seq_tables.find(seq_id) != seq_tables.end();
}

uint32_t llama_kv_cache_paged::get_sequence_length(llama_seq_id seq_id) const {
    auto it = seq_tables.find(seq_id);
    if (it == seq_tables.end()) {
        return 0;
    }
    return it->second.get_num_tokens();
}

void llama_kv_cache_paged::truncate_sequence(llama_seq_id seq_id, uint32_t n_tokens) {
    auto it = seq_tables.find(seq_id);
    if (it == seq_tables.end()) {
        return;
    }

    auto & table = it->second;
    const uint32_t current_blocks = table.num_blocks();
    const uint32_t needed_blocks = llama_kv_block_table::tokens_to_blocks(n_tokens, block_size);

    // Free excess blocks
    while (table.num_blocks() > needed_blocks) {
        int32_t block_id = table.pop_block();
        if (block_id >= 0) {
            block_manager.free(static_cast<uint32_t>(block_id));
        }
    }

    table.set_num_tokens(n_tokens);
}

std::vector<int32_t> llama_kv_cache_paged::build_block_table_tensor(
    const std::vector<llama_seq_id> & seq_ids,
    uint32_t max_blocks
) const {
    const size_t n_seqs = seq_ids.size();
    std::vector<int32_t> result(n_seqs * max_blocks, -1);  // Initialize with -1 (invalid)

    for (size_t s = 0; s < n_seqs; ++s) {
        const auto * table = get_block_table(seq_ids[s]);
        if (table != nullptr) {
            const auto & block_ids = table->get_block_ids();
            const size_t n_blocks = std::min(static_cast<size_t>(max_blocks), block_ids.size());
            for (size_t b = 0; b < n_blocks; ++b) {
                result[s * max_blocks + b] = block_ids[b];
            }
        }
    }

    return result;
}

float llama_kv_cache_paged::get_memory_efficiency() const {
    uint32_t total_tokens = 0;
    uint32_t total_capacity = get_used_blocks() * block_size;

    for (const auto & [seq_id, table] : seq_tables) {
        total_tokens += table.get_num_tokens();
    }

    if (total_capacity == 0) {
        return 1.0f;  // No memory used, so efficiency is perfect
    }

    return static_cast<float>(total_tokens) / static_cast<float>(total_capacity);
}

void llama_kv_cache_paged::reset() {
    seq_tables.clear();
    block_manager.reset();
}

void llama_kv_cache_paged::apply_defrag(const std::vector<llama_kv_block_manager::defrag_move> & moves) {
    if (moves.empty()) {
        return;
    }

    // Build a mapping from old block IDs to new block IDs
    std::unordered_map<int32_t, int32_t> block_remap;
    for (const auto & move : moves) {
        block_remap[static_cast<int32_t>(move.src_block)] = static_cast<int32_t>(move.dst_block);
    }

    // Update all sequence block tables
    for (auto & [seq_id, table] : seq_tables) {
        const auto & old_blocks = table.get_block_ids();
        std::vector<int32_t> new_blocks;
        new_blocks.reserve(old_blocks.size());

        for (int32_t block_id : old_blocks) {
            auto it = block_remap.find(block_id);
            if (it != block_remap.end()) {
                // This block was moved, use new location
                new_blocks.push_back(it->second);
            } else {
                // This block wasn't moved, keep old location
                new_blocks.push_back(block_id);
            }
        }

        // Rebuild the table with new block IDs
        const uint32_t saved_tokens = table.get_num_tokens();
        table.clear();
        for (int32_t block_id : new_blocks) {
            table.append_block(block_id);
        }
        table.set_num_tokens(saved_tokens);
    }

    // Apply the moves to the block manager's internal state
    block_manager.apply_defrag(moves);
}

// =============================================================================
// Prefix Caching implementation for llama_kv_cache_paged
// =============================================================================

bool llama_kv_cache_paged::allocate_with_prefix(llama_seq_id seq_id, uint32_t n_tokens, const std::vector<uint64_t> & hashes) {
    const uint32_t n_blocks_needed = llama_kv_block_table::tokens_to_blocks(n_tokens, block_size);

    // Verify we have the right number of hashes
    if (hashes.size() < n_blocks_needed) {
        return false;
    }

    // Count how many new blocks we need (vs shared blocks)
    uint32_t new_blocks_needed = 0;
    for (uint32_t i = 0; i < n_blocks_needed; ++i) {
        if (block_manager.find_block_by_hash(hashes[i]) < 0) {
            new_blocks_needed++;
        }
    }

    // Check if we have enough free blocks for new allocations
    if (!block_manager.can_allocate(new_blocks_needed)) {
        return false;
    }

    // Create or get the block table for this sequence
    auto & table = seq_tables[seq_id];
    table.set_seq_id(seq_id);
    table.clear();

    // Allocate or share blocks
    for (uint32_t i = 0; i < n_blocks_needed; ++i) {
        auto [block_id, is_shared] = block_manager.allocate_or_share(hashes[i]);
        if (block_id < 0) {
            // This shouldn't happen since we checked capacity, but handle gracefully
            // Free any blocks we've already allocated
            const auto & allocated = table.get_block_ids();
            for (int32_t bid : allocated) {
                if (bid >= 0) {
                    block_manager.free(static_cast<uint32_t>(bid));
                }
            }
            seq_tables.erase(seq_id);
            return false;
        }
        table.append_block(block_id);
    }

    table.set_num_tokens(n_tokens);
    return true;
}

void llama_kv_cache_paged::set_block_hash(llama_seq_id seq_id, uint32_t block_idx, uint64_t hash) {
    auto it = seq_tables.find(seq_id);
    if (it == seq_tables.end()) {
        return;
    }

    const auto & block_ids = it->second.get_block_ids();
    if (block_idx >= block_ids.size()) {
        return;
    }

    int32_t block_id = block_ids[block_idx];
    if (block_id >= 0) {
        block_manager.set_content_hash(static_cast<uint32_t>(block_id), hash);
    }
}

std::vector<uint64_t> llama_kv_cache_paged::compute_block_hashes(const int32_t * tokens, uint32_t n_tokens) const {
    const uint32_t n_blocks = llama_kv_block_table::tokens_to_blocks(n_tokens, block_size);
    std::vector<uint64_t> hashes;
    hashes.reserve(n_blocks);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint32_t end_pos = std::min((b + 1) * block_size, n_tokens);

        // Hash the tokens in this block
        // For prefix caching to work correctly, we need to include ALL previous blocks
        // in the hash, not just the current block. This ensures different prefixes
        // with the same suffix don't collide.
        uint64_t hash = llama_hash_tokens(tokens, end_pos, 0);  // Hash from start to end_pos
        hashes.push_back(hash);
    }

    return hashes;
}

float llama_kv_cache_paged::get_prefix_sharing_ratio() const {
    uint32_t total_refs = 0;
    uint32_t shared_refs = 0;

    // Count references across all sequences
    for (const auto & [seq_id, table] : seq_tables) {
        const auto & block_ids = table.get_block_ids();
        for (int32_t block_id : block_ids) {
            if (block_id >= 0) {
                uint32_t ref_count = block_manager.get_ref_count(static_cast<uint32_t>(block_id));
                total_refs++;
                if (ref_count > 1) {
                    // This block is shared
                    shared_refs++;
                }
            }
        }
    }

    if (total_refs == 0) {
        return 0.0f;
    }

    return static_cast<float>(shared_refs) / static_cast<float>(total_refs);
}
