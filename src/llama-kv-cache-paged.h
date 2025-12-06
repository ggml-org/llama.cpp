#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

// Forward declare the paged KV cache class
class llama_kv_cache_paged;

//
// llama_kv_cache_paged_context - Context for PagedAttention operations
//
// NOTE: The graph building code uses static_cast to llama_kv_cache_context* and calls
// methods specific to that class. To avoid crashes, we inherit from llama_kv_cache_context
// and provide stub implementations that will be replaced with proper PagedAttention logic.
//
class llama_kv_cache_context;  // forward declare

class llama_kv_cache_paged_context : public llama_memory_context_i {
public:
    llama_kv_cache_paged_context(llama_memory_status status);
    llama_kv_cache_paged_context(llama_kv_cache_paged * kv_paged);

    virtual ~llama_kv_cache_paged_context() {
        fprintf(stderr, "llama_kv_cache_paged_context::~llama_kv_cache_paged_context() called\n");
    }

    // llama_memory_context_i interface
    bool next() override {
        fprintf(stderr, "llama_kv_cache_paged_context::next() called\n");
        return false;
    }
    bool apply() override {
        fprintf(stderr, "llama_kv_cache_paged_context::apply() called\n");
        return true;
    }
    llama_memory_status get_status() const override {
        fprintf(stderr, "llama_kv_cache_paged_context::get_status() called\n");
        return status;
    }
    const llama_ubatch & get_ubatch() const override {
        fprintf(stderr, "llama_kv_cache_paged_context::get_ubatch() called\n");
        return ubatch;
    }

    // Get the underlying paged cache
    llama_kv_cache_paged * get_kv_paged() const { return kv_paged; }

    // Stub methods to match llama_kv_cache_context interface
    // These are called by graph building code via static_cast
    // TODO: Implement proper PagedAttention versions
    uint32_t get_n_kv() const { return 0; }
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_k_shift(ggml_tensor * dst) const;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    llama_memory_status status;
    llama_kv_cache_paged * kv_paged;
    llama_ubatch ubatch;  // dummy ubatch
};

//
// llama_kv_cache_paged - PagedAttention KV cache implementation
//
// This cache divides memory into fixed-size blocks (similar to virtual memory paging)
// to reduce fragmentation and enable efficient memory sharing between sequences.
//
// Key concepts:
// - Block: Fixed-size unit of KV cache storage (e.g., 16 tokens)
// - Block Table: Maps logical token positions to physical blocks per sequence
// - Block Pool: Manages allocation/deallocation of physical blocks
//

class llama_kv_cache_paged : public llama_memory_i {
public:
    // Physical block in memory containing KV data for multiple tokens
    struct block {
        uint32_t id;              // unique block ID
        ggml_tensor * k_data;     // K cache data for this block
        ggml_tensor * v_data;     // V cache data for this block
        uint32_t ref_count;       // reference count for block sharing
        bool is_free;             // whether block is in free pool

        block() : id(0), k_data(nullptr), v_data(nullptr), ref_count(0), is_free(true) {}
    };

    llama_kv_cache_paged(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   block_size,        // tokens per block
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache_paged() = default;

    //
    // llama_memory_i interface
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // PagedAttention specific API
    //

    // Get block size (tokens per block)
    uint32_t get_block_size() const { return block_size; }

    // Get total number of blocks
    uint32_t get_num_blocks() const { return num_blocks; }

    // Get number of free blocks
    uint32_t get_num_free_blocks() const { return static_cast<uint32_t>(free_blocks.size()); }

    // Get block table for a sequence (maps token positions to block IDs)
    const std::vector<uint32_t> & get_block_table(llama_seq_id seq_id) const;

    // Get sequence lengths for all sequences
    std::vector<int32_t> get_seq_lens() const;

    // Access to block data tensors (for CUDA kernels)
    ggml_tensor * get_k_blocks(int32_t il) const;
    ggml_tensor * get_v_blocks(int32_t il) const;

    // Get block tables tensor (for CUDA kernels)
    ggml_tensor * build_block_tables_tensor(ggml_context * ctx) const;

    // Get sequence lengths tensor (for CUDA kernels)
    ggml_tensor * build_seq_lens_tensor(ggml_context * ctx) const;

    // Populate tensor data for CUDA kernels
    void populate_block_tables_tensor(ggml_tensor * tensor) const;
    void populate_seq_lens_tensor(ggml_tensor * tensor) const;

private:
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif
    const llama_model & model;
    const llama_hparams & hparams;

    // Block storage per layer
    struct kv_layer {
        uint32_t il;  // layer index in model

        // All blocks for this layer (both used and free)
        std::vector<block> blocks;

        // Contiguous tensors holding all blocks for this layer
        // Shape: [num_blocks, block_size, num_kv_heads, head_size]
        ggml_tensor * k_all_blocks = nullptr;
        ggml_tensor * v_all_blocks = nullptr;
    };

    const ggml_type type_k;            // data type for K cache
    const ggml_type type_v;            // data type for V cache
    const uint32_t n_seq_max = 1;      // max number of sequences
#ifdef __clang__
#pragma clang diagnostic pop
#endif
    const uint32_t block_size = 16;    // tokens per block (must be power of 2)
    const uint32_t num_blocks = 0;     // total number of blocks

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // ggml contexts for the KV cache along with allocated backend buffers
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // Block management
    std::vector<uint32_t> free_blocks;  // IDs of free blocks

    // Per-sequence block tables (seq_id -> list of block IDs)
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> block_tables;

    // Per-sequence metadata
    struct seq_metadata {
        llama_pos pos_min = -1;  // minimum position in sequence
        llama_pos pos_max = -1;  // maximum position in sequence
        uint32_t length = 0;     // sequence length in tokens
    };
    std::unordered_map<llama_seq_id, seq_metadata> seq_meta;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    // Block management functions
    uint32_t allocate_block();
    void free_block(uint32_t block_id);
    void allocate_blocks_for_sequence(llama_seq_id seq_id, uint32_t num_tokens);

    // Helper functions
    size_t total_size() const;
    size_t size_k_bytes() const;
    size_t size_v_bytes() const;
};
