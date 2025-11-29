#pragma once

#include <cstdint>
#include <queue>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <utility>

// =============================================================================
// Prefix Caching: Content-addressable block lookup
// =============================================================================

// FNV-1a hash function for fast, good-quality hashing
// Used to compute content hashes for prefix caching
inline uint64_t llama_hash_fnv1a(const void * data, size_t len, uint64_t seed = 0xcbf29ce484222325ULL) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = seed;
    constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

// Hash a sequence of token IDs (for prefix caching)
// layer_id is included to ensure different layers have different hashes for same tokens
inline uint64_t llama_hash_tokens(const int32_t * tokens, uint32_t n_tokens, uint32_t layer_id = 0) {
    // Include layer_id in the seed to differentiate layers
    uint64_t seed = 0xcbf29ce484222325ULL ^ (static_cast<uint64_t>(layer_id) << 32);
    return llama_hash_fnv1a(tokens, n_tokens * sizeof(int32_t), seed);
}

// Sequence ID type (matches llama.h definition)
using llama_seq_id = int32_t;

// Default block size in tokens (matches vLLM default, aligns with XMX tile size of 16)
constexpr uint32_t LLAMA_KV_BLOCK_SIZE = 16;

// Maximum number of blocks per sequence (for 128K context with block_size=16 = 8192 blocks)
constexpr uint32_t LLAMA_KV_MAX_BLOCKS_PER_SEQ = 8192;

// Individual KV block metadata
struct llama_kv_block {
    uint32_t block_id;      // Physical block ID (index into block storage)
    uint32_t ref_count;     // Number of sequences referencing this block (for copy-on-write)
    bool     is_free;       // True if block is in free list

    // For prefix caching (future optimization)
    uint64_t content_hash;  // Hash of block contents for prefix sharing

    llama_kv_block() : block_id(0), ref_count(0), is_free(true), content_hash(0) {}
};

// Per-sequence block table: maps logical blocks to physical blocks
class llama_kv_block_table {
public:
    llama_kv_block_table() : seq_id(-1), num_tokens(0) {}

    explicit llama_kv_block_table(llama_seq_id id) : seq_id(id), num_tokens(0) {}

    // Get the physical block ID for a given token position
    // Returns -1 if position is beyond allocated blocks
    int32_t get_physical_block(uint32_t pos, uint32_t block_size = LLAMA_KV_BLOCK_SIZE) const {
        const uint32_t logical_block = pos / block_size;
        if (logical_block >= block_ids.size()) {
            return -1;
        }
        return block_ids[logical_block];
    }

    // Get the offset within a block for a given token position
    static uint32_t get_block_offset(uint32_t pos, uint32_t block_size = LLAMA_KV_BLOCK_SIZE) {
        return pos % block_size;
    }

    // Get number of allocated blocks
    uint32_t num_blocks() const {
        return static_cast<uint32_t>(block_ids.size());
    }

    // Add a new block to the sequence
    void append_block(int32_t physical_block_id) {
        block_ids.push_back(physical_block_id);
    }

    // Remove the last block from the sequence
    int32_t pop_block() {
        if (block_ids.empty()) {
            return -1;
        }
        int32_t last = block_ids.back();
        block_ids.pop_back();
        return last;
    }

    // Clear all blocks
    void clear() {
        block_ids.clear();
        num_tokens = 0;
    }

    // Get the raw block table (for passing to kernels)
    const std::vector<int32_t> & get_block_ids() const {
        return block_ids;
    }

    // Get/set sequence ID
    llama_seq_id get_seq_id() const { return seq_id; }
    void set_seq_id(llama_seq_id id) { seq_id = id; }

    // Get/set number of tokens
    uint32_t get_num_tokens() const { return num_tokens; }
    void set_num_tokens(uint32_t n) { num_tokens = n; }

    // Calculate required blocks for a given number of tokens
    static uint32_t tokens_to_blocks(uint32_t n_tokens, uint32_t block_size = LLAMA_KV_BLOCK_SIZE) {
        return (n_tokens + block_size - 1) / block_size;
    }

private:
    llama_seq_id seq_id;
    uint32_t num_tokens;                // Current number of tokens in sequence
    std::vector<int32_t> block_ids;     // block_ids[i] = physical block for logical block i
};

// Block manager: handles allocation, deallocation, and reference counting
class llama_kv_block_manager {
public:
    llama_kv_block_manager(uint32_t n_blocks, uint32_t block_size = LLAMA_KV_BLOCK_SIZE);
    ~llama_kv_block_manager() = default;

    // Allocate a new block
    // Returns physical block ID, or -1 if no blocks available
    int32_t allocate();

    // Free a block (decrements ref count, adds to free list if ref_count == 0)
    void free(uint32_t block_id);

    // Increment reference count (for copy-on-write / prefix sharing)
    void ref_inc(uint32_t block_id);

    // Decrement reference count (returns true if block was freed)
    bool ref_dec(uint32_t block_id);

    // Get reference count for a block
    uint32_t get_ref_count(uint32_t block_id) const;

    // Allocate multiple blocks at once
    // Returns vector of block IDs, or empty vector if not enough blocks
    std::vector<int32_t> allocate_n(uint32_t n);

    // Free multiple blocks
    void free_n(const std::vector<int32_t> & block_ids);

    // Statistics
    uint32_t get_total_blocks() const { return n_blocks; }
    uint32_t get_free_blocks() const { return static_cast<uint32_t>(free_list.size()); }
    uint32_t get_used_blocks() const { return n_blocks - get_free_blocks(); }
    uint32_t get_block_size() const { return block_size; }

    // Check if there are enough free blocks
    bool can_allocate(uint32_t n) const { return get_free_blocks() >= n; }

    // Reset all blocks to free state
    void reset();

    // Get block metadata (for debugging)
    const llama_kv_block & get_block(uint32_t block_id) const {
        assert(block_id < n_blocks);
        return blocks[block_id];
    }

    // Defragmentation support
    struct defrag_move {
        uint32_t src_block;  // Source physical block
        uint32_t dst_block;  // Destination physical block
    };

    // Compute fragmentation ratio: 0.0 = no fragmentation, 1.0 = maximum fragmentation
    // Fragmentation = 1 - (used_blocks / (max_used_block_id + 1))
    float get_fragmentation_ratio() const;

    // Compute defragmentation plan: moves blocks to compact used space at low indices
    // Returns list of (src, dst) moves to execute
    std::vector<defrag_move> compute_defrag_plan() const;

    // Apply defragmentation: update internal state after moves are executed
    // The caller must have already copied the KV data
    void apply_defrag(const std::vector<defrag_move> & moves);

    // Check if defragmentation would be beneficial
    // threshold: trigger defrag if fragmentation_ratio > threshold (default 0.2 = 20% waste)
    bool needs_defrag(float threshold = 0.2f) const;

    // =========================================================================
    // Prefix Caching Support
    // =========================================================================

    // Set the content hash for a block (call after block is filled with data)
    // This registers the block in the hash lookup table for future sharing
    void set_content_hash(uint32_t block_id, uint64_t hash);

    // Get the content hash for a block
    uint64_t get_content_hash(uint32_t block_id) const;

    // Find a block with matching content hash
    // Returns block_id if found, -1 if not found
    int32_t find_block_by_hash(uint64_t hash) const;

    // Allocate a new block or share an existing one with matching content
    // If a block with the same hash exists and is shareable, increments its ref_count and returns it
    // Otherwise allocates a new block
    // Returns: {block_id, is_shared} where is_shared=true if block was reused
    std::pair<int32_t, bool> allocate_or_share(uint64_t hash);

    // Check if prefix caching is enabled
    bool is_prefix_caching_enabled() const { return prefix_caching_enabled; }

    // Enable/disable prefix caching
    void set_prefix_caching(bool enabled) { prefix_caching_enabled = enabled; }

    // Get prefix cache statistics
    uint32_t get_prefix_cache_hits() const { return prefix_cache_hits; }
    uint32_t get_prefix_cache_misses() const { return prefix_cache_misses; }
    void reset_prefix_cache_stats() { prefix_cache_hits = 0; prefix_cache_misses = 0; }

private:
    uint32_t n_blocks;
    uint32_t block_size;
    std::vector<llama_kv_block> blocks;
    std::queue<uint32_t> free_list;

    // Prefix caching: hash -> block_id lookup table
    bool prefix_caching_enabled = false;
    std::unordered_map<uint64_t, uint32_t> hash_to_block;
    uint32_t prefix_cache_hits = 0;
    uint32_t prefix_cache_misses = 0;
};

// Paged KV cache manager: coordinates block allocation for multiple sequences
class llama_kv_cache_paged {
public:
    llama_kv_cache_paged(
        uint32_t n_ctx,             // Maximum context length
        uint32_t n_seq_max,         // Maximum number of sequences
        uint32_t block_size = LLAMA_KV_BLOCK_SIZE
    );

    ~llama_kv_cache_paged() = default;

    // Allocate blocks for a sequence to hold n_tokens
    // Returns true on success, false if not enough memory
    bool allocate_for_sequence(llama_seq_id seq_id, uint32_t n_tokens);

    // Extend a sequence by n_tokens (allocates new blocks as needed)
    bool extend_sequence(llama_seq_id seq_id, uint32_t n_tokens);

    // Free all blocks for a sequence
    void free_sequence(llama_seq_id seq_id);

    // Copy blocks from one sequence to another (copy-on-write)
    bool copy_sequence(llama_seq_id src_seq_id, llama_seq_id dst_seq_id);

    // Get the block table for a sequence
    const llama_kv_block_table * get_block_table(llama_seq_id seq_id) const;
    llama_kv_block_table * get_block_table(llama_seq_id seq_id);

    // Check if sequence exists
    bool has_sequence(llama_seq_id seq_id) const;

    // Get number of tokens in a sequence
    uint32_t get_sequence_length(llama_seq_id seq_id) const;

    // Truncate a sequence to n_tokens
    void truncate_sequence(llama_seq_id seq_id, uint32_t n_tokens);

    // Build a flattened block table tensor for kernel dispatch
    // Format: [n_seqs, max_blocks_per_seq] with -1 padding for unused slots
    std::vector<int32_t> build_block_table_tensor(
        const std::vector<llama_seq_id> & seq_ids,
        uint32_t max_blocks_per_seq
    ) const;

    // Statistics
    uint32_t get_free_blocks() const { return block_manager.get_free_blocks(); }
    uint32_t get_used_blocks() const { return block_manager.get_used_blocks(); }
    uint32_t get_total_blocks() const { return block_manager.get_total_blocks(); }
    uint32_t get_block_size() const { return block_size; }
    uint32_t get_num_sequences() const { return static_cast<uint32_t>(seq_tables.size()); }

    // Memory efficiency metric: used_tokens / (used_blocks * block_size)
    float get_memory_efficiency() const;

    // Reset all state
    void reset();

    // Defragmentation support

    // Get fragmentation ratio
    float get_fragmentation_ratio() const { return block_manager.get_fragmentation_ratio(); }

    // Check if defragmentation is beneficial
    bool needs_defrag(float threshold = 0.2f) const { return block_manager.needs_defrag(threshold); }

    // Compute defragmentation plan
    std::vector<llama_kv_block_manager::defrag_move> compute_defrag_plan() const {
        return block_manager.compute_defrag_plan();
    }

    // Apply defragmentation: updates block tables to use new physical locations
    // The caller must execute KV data copies based on the moves before calling this
    void apply_defrag(const std::vector<llama_kv_block_manager::defrag_move> & moves);

    // =========================================================================
    // Prefix Caching Support
    // =========================================================================

    // Enable/disable prefix caching
    void set_prefix_caching(bool enabled) { block_manager.set_prefix_caching(enabled); }
    bool is_prefix_caching_enabled() const { return block_manager.is_prefix_caching_enabled(); }

    // Allocate blocks with prefix sharing for a sequence
    // hashes: content hashes for each block (size = number of blocks needed)
    // For blocks with matching hashes, existing blocks will be shared (ref_count incremented)
    // Returns true on success, false if not enough memory for new blocks
    bool allocate_with_prefix(llama_seq_id seq_id, uint32_t n_tokens, const std::vector<uint64_t> & hashes);

    // Mark a block as having a specific content hash (called after KV data is computed)
    // This allows the block to be shared with future sequences that have the same prefix
    void set_block_hash(llama_seq_id seq_id, uint32_t block_idx, uint64_t hash);

    // Compute hashes for all blocks in a sequence (typically from token IDs)
    // tokens: token IDs for the sequence
    // n_tokens: number of tokens
    // Returns: vector of hashes, one per block
    std::vector<uint64_t> compute_block_hashes(const int32_t * tokens, uint32_t n_tokens) const;

    // Get prefix cache statistics
    uint32_t get_prefix_cache_hits() const { return block_manager.get_prefix_cache_hits(); }
    uint32_t get_prefix_cache_misses() const { return block_manager.get_prefix_cache_misses(); }
    void reset_prefix_cache_stats() { block_manager.reset_prefix_cache_stats(); }

    // Get prefix sharing ratio: shared_refs / total_refs
    // Higher = more memory saved through prefix sharing
    float get_prefix_sharing_ratio() const;

private:
    uint32_t n_ctx;
    uint32_t n_seq_max;
    uint32_t block_size;
    uint32_t max_blocks_per_seq;

    llama_kv_block_manager block_manager;
    std::unordered_map<llama_seq_id, llama_kv_block_table> seq_tables;
};
