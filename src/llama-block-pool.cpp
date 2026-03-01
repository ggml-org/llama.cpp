#include "llama-block-pool.h"

#include "llama-hparams.h"
#include "llama-impl.h"

#include <algorithm>
#include <cassert>

// ========================================
// Prefix Caching: Token Hash Implementation
// ========================================

// FNV-1a 64-bit hash for token sequences
// Fast, good distribution, widely used for hashing
token_hash_t compute_token_hash(const llama_token * tokens, size_t n_tokens) {
    if (tokens == nullptr || n_tokens == 0) {
        return 0;
    }

    // FNV-1a 64-bit constants
    const token_hash_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    const token_hash_t FNV_PRIME = 1099511628211ULL;

    token_hash_t hash = FNV_OFFSET_BASIS;

    for (size_t i = 0; i < n_tokens; ++i) {
        hash ^= static_cast<token_hash_t>(tokens[i]);
        hash *= FNV_PRIME;
    }

    return hash;
}

bool llama_block_pool::init(
        ggml_context * ctx,
        const llama_hparams & hparams,
        uint32_t n_blocks,
        uint32_t block_sz,
        ggml_type type_k,
        ggml_type type_v,
        bool v_trans) {

    this->ctx = ctx;
    this->num_blocks = n_blocks;
    this->block_size = block_sz;
    this->n_layer = hparams.n_layer;
    this->n_head = hparams.n_head();
    this->head_dim = hparams.n_embd_head_k;

    // Initialize block metadata
    blocks.resize(num_blocks);
    free_blocks.clear();
    for (uint32_t i = 0; i < num_blocks; ++i) {
        blocks[i].ref_count = 0;
        blocks[i].is_free = true;
        blocks[i].seq_id = -1;
        blocks[i].logical_block = -1;
        free_blocks.insert(i);
    }

    // Create per-layer K pool tensors (2D each, avoids Metal buffer issues with views)
    // Shape per layer: [n_embd_k_gqa, block_size * num_blocks]
    k_pool.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        const int64_t ne[2] = {
            (int64_t)hparams.n_embd_k_gqa(),
            (int64_t)block_size * (int64_t)num_blocks
        };
        k_pool[il] = ggml_new_tensor_2d(ctx, type_k, ne[0], ne[1]);
        ggml_format_name(k_pool[il], "block_pool_k_l%u", il);
    }

    // Create per-layer V pool tensors (2D each)
    // Shape per layer: [n_embd_v_gqa, block_size * num_blocks]
    v_pool.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        const int64_t ne[2] = {
            (int64_t)hparams.n_embd_v_gqa(),
            (int64_t)block_size * (int64_t)num_blocks
        };
        v_pool[il] = ggml_new_tensor_2d(ctx, type_v, ne[0], ne[1]);
        ggml_format_name(v_pool[il], "block_pool_v_l%u", il);
    }

    // Verify all tensors created successfully
    for (uint32_t il = 0; il < n_layer; ++il) {
        GGML_ASSERT(k_pool[il] != nullptr);
        GGML_ASSERT(v_pool[il] != nullptr);
    }

    return true;
}

int32_t llama_block_pool::allocate_block(llama_seq_id seq_id, int32_t logical_block) {
    if (free_blocks.empty()) {
        // Pool is full
        return -1;
    }

    // Get the first free block
    int32_t block_id = *free_blocks.begin();
    free_blocks.erase(free_blocks.begin());

    // Update metadata
    blocks[block_id].ref_count = 1;
    blocks[block_id].is_free = false;
    blocks[block_id].seq_id = seq_id;
    blocks[block_id].logical_block = logical_block;

    return block_id;
}

void llama_block_pool::free_block(int32_t block_id) {
    if (block_id < 0 || block_id >= (int32_t)num_blocks) {
        return;
    }

    blocks[block_id].ref_count--;

    if (blocks[block_id].ref_count <= 0) {
        // Remove from hash index if present
        if (prefix_caching_enabled && blocks[block_id].token_hash != 0) {
            hash_to_block.erase(blocks[block_id].token_hash);
        }

        blocks[block_id].is_free = true;
        blocks[block_id].seq_id = -1;
        blocks[block_id].logical_block = -1;
        blocks[block_id].token_hash = 0;
        blocks[block_id].is_prefix_block = false;
        free_blocks.insert(block_id);
    }
}

int32_t llama_block_pool::get_physical_block(llama_seq_id seq_id, int32_t logical_block) const {
    auto it = block_tables.find(seq_id);
    if (it == block_tables.end()) {
        return -1;
    }

    const auto & table = it->second;
    if (logical_block < 0 || logical_block >= (int32_t)table.logical_to_physical.size()) {
        return -1;
    }

    return table.logical_to_physical[logical_block];
}

int32_t llama_block_pool::add_block_to_sequence(llama_seq_id seq_id, int32_t logical_block) {
    // Get or create block table entry
    auto & table = block_tables[seq_id];
    table.seq_id = seq_id;

    // Ensure the vector is large enough
    if (logical_block >= (int32_t)table.logical_to_physical.size()) {
        table.logical_to_physical.resize(logical_block + 1, -1);
    }

    // Allocate a physical block
    int32_t physical_block = allocate_block(seq_id, logical_block);
    if (physical_block < 0) {
        return -1;
    }

    table.logical_to_physical[logical_block] = physical_block;
    return physical_block;
}

void llama_block_pool::remove_sequence(llama_seq_id seq_id) {
    auto it = block_tables.find(seq_id);
    if (it == block_tables.end()) {
        return;
    }

    // Free all blocks in this sequence
    const auto & table = it->second;
    for (int32_t block_id : table.logical_to_physical) {
        free_block(block_id);
    }

    // Remove the block table entry
    block_tables.erase(it);
}

size_t llama_block_pool::n_blocks_for_sequence(llama_seq_id seq_id) const {
    auto it = block_tables.find(seq_id);
    if (it == block_tables.end()) {
        return 0;
    }
    return it->second.logical_to_physical.size();
}

size_t llama_block_pool::memory_size() const {
    size_t size = 0;
    for (const auto * t : k_pool) {
        if (t) {
            size += ggml_nbytes(t);
        }
    }
    for (const auto * t : v_pool) {
        if (t) {
            size += ggml_nbytes(t);
        }
    }
    return size;
}

void llama_block_pool::defragment() {
    // Simple defragmentation: compact blocks to the beginning
    // This is a placeholder - a full implementation would:
    // 1. Identify fragmented blocks
    // 2. Move data to consolidate free space
    // 3. Update block tables

    // For now, just clear empty sequences
    std::vector<llama_seq_id> empty_seqs;
    for (const auto & [seq_id, table] : block_tables) {
        if (table.logical_to_physical.empty()) {
            empty_seqs.push_back(seq_id);
        }
    }

    for (llama_seq_id seq_id : empty_seqs) {
        block_tables.erase(seq_id);
    }
}

// ========================================
// Prefix Caching Implementation (Phase 2A)
// ========================================

int32_t llama_block_pool::find_block_by_hash(token_hash_t hash) const {
    if (!prefix_caching_enabled || hash == 0) {
        return -1;
    }

    auto it = hash_to_block.find(hash);
    if (it != hash_to_block.end()) {
        int32_t block_id = it->second;
        // Verify block is still valid and not free
        if (block_id >= 0 && block_id < (int32_t)num_blocks && !blocks[block_id].is_free) {
            return block_id;
        }
    }

    return -1;
}

bool llama_block_pool::share_block(int32_t block_id, llama_seq_id seq_id, int32_t logical_block) {
    if (!prefix_caching_enabled) {
        return false;
    }

    if (block_id < 0 || block_id >= (int32_t)num_blocks) {
        return false;
    }

    block_info & info = blocks[block_id];
    if (info.is_free) {
        return false;
    }

    // Increment reference count
    info.ref_count++;

    // Add to block table of the new sequence
    auto & table = block_tables[seq_id];
    table.seq_id = seq_id;

    if (logical_block >= (int32_t)table.logical_to_physical.size()) {
        table.logical_to_physical.resize(logical_block + 1, -1);
    }

    table.logical_to_physical[logical_block] = block_id;

    return true;
}

int32_t llama_block_pool::cow_block(int32_t block_id, llama_seq_id new_seq_id, int32_t new_logical_block) {
    if (block_id < 0 || block_id >= (int32_t)num_blocks) {
        return -1;
    }

    block_info & info = blocks[block_id];

    // If ref_count is 1, no copy needed - this sequence owns it exclusively
    if (info.ref_count <= 1) {
        return block_id;
    }

    // Need to copy: allocate new block
    if (free_blocks.empty()) {
        // Pool is full, cannot copy
        return -1;
    }

    // Allocate new physical block
    int32_t new_block_id = *free_blocks.begin();
    free_blocks.erase(free_blocks.begin());

    block_info & new_info = blocks[new_block_id];
    new_info.ref_count = 1;
    new_info.is_free = false;
    new_info.seq_id = new_seq_id;
    new_info.logical_block = new_logical_block;
    new_info.token_hash = info.token_hash;  // Copy hash
    new_info.is_prefix_block = false;  // New block is not a shared prefix

    // Decrement old block's ref count
    info.ref_count--;

    // Note: Actual KV data copy must be done by the caller (llama_kv_cache)
    // because this block pool only manages metadata, not the actual tensor data

    return new_block_id;
}

void llama_block_pool::register_block_hash(int32_t block_id, token_hash_t hash) {
    if (!prefix_caching_enabled || block_id < 0 || block_id >= (int32_t)num_blocks || hash == 0) {
        return;
    }

    blocks[block_id].token_hash = hash;
    blocks[block_id].is_prefix_block = true;

    // Add to hash index
    hash_to_block[hash] = block_id;
}

token_hash_t llama_block_pool::get_block_hash(int32_t block_id) const {
    if (block_id < 0 || block_id >= (int32_t)num_blocks) {
        return 0;
    }
    return blocks[block_id].token_hash;
}

void llama_block_pool::clear_prefix_caching() {
    hash_to_block.clear();

    // Reset prefix flags on all blocks
    for (auto & info : blocks) {
        info.is_prefix_block = false;
        info.token_hash = 0;
    }
}
