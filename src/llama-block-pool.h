#pragma once

#include "ggml.h"
#include "ggml-alloc.h"  // for ggml_backend_buffer_t
#include "llama.h"       // for llama_seq_id

#include <cstdint>
#include <vector>
#include <set>
#include <map>

struct llama_context;
struct llama_hparams;

// Block pool for paged attention
// Manages a pool of fixed-size blocks that can be dynamically allocated to sequences
struct llama_block_pool {
    // Block metadata
    struct block_info {
        int32_t ref_count = 0;     // Reference count for Copy-on-Write
        bool is_free = true;       // Whether block is available
        int32_t seq_id = -1;       // Sequence ID that owns this block (-1 if free)
        int32_t logical_block = -1; // Logical block index within the sequence
    };

    // Block table entry per sequence
    struct block_table_entry {
        std::vector<int32_t> logical_to_physical;  // Maps logical block index to physical block ID
        int32_t seq_id = -1;
        uint32_t n_tokens = 0;  // Current number of tokens in this sequence
    };

    // Configuration
    uint32_t block_size;       // Tokens per block (e.g., 16)
    uint32_t num_blocks;       // Total number of blocks in pool
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t head_dim;

    // Backend buffers (pool storage)
    // Per-layer 2D tensors: [n_embd_gqa, block_size * num_blocks]
    // This avoids creating view tensors that don't have Metal buffers
    std::vector<ggml_tensor *> k_pool;  // One 2D tensor per layer
    std::vector<ggml_tensor *> v_pool;  // One 2D tensor per layer

    // Metadata
    std::vector<block_info> blocks;
    std::set<int32_t> free_blocks;

    // Block tables per sequence
    std::map<llama_seq_id, block_table_entry> block_tables;

    // GGML context for tensor management
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    llama_block_pool() = default;

    // Initialize the block pool
    bool init(
        ggml_context * ctx,
        const llama_hparams & hparams,
        uint32_t n_blocks,
        uint32_t block_size,
        ggml_type type_k,
        ggml_type type_v,
        bool v_trans);

    // Allocate a new block
    // Returns block ID on success, -1 on failure (pool full)
    int32_t allocate_block(llama_seq_id seq_id, int32_t logical_block);

    // Free a block (decrement ref count)
    void free_block(int32_t block_id);

    // Get the physical block for a logical block in a sequence
    int32_t get_physical_block(llama_seq_id seq_id, int32_t logical_block) const;

    // Add a new block to a sequence's block table
    int32_t add_block_to_sequence(llama_seq_id seq_id, int32_t logical_block);

    // Remove a sequence and free all its blocks
    void remove_sequence(llama_seq_id seq_id);

    // Get number of free blocks
    size_t n_free_blocks() const { return free_blocks.size(); }

    // Get number of blocks used by a sequence
    size_t n_blocks_for_sequence(llama_seq_id seq_id) const;

    // Get total memory usage
    size_t memory_size() const;

    // Defragmentation (compact blocks to reduce fragmentation)
    void defragment();
};
