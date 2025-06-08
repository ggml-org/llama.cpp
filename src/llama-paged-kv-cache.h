#ifndef LLAMA_PAGED_KV_CACHE_H
#define LLAMA_PAGED_KV_CACHE_H

#include "llama-memory.h"
#include "llama-paged-kv-cells.h" // Manages the actual page storage
#include "llama-context.h"       // For llama_context, llama_batch types

// Forward declarations
class llama_paged_kv_cache_state; // Implements llama_memory_state_i

class llama_paged_kv_cache : public llama_memory_i {
public:
    // Constructor
    // Takes page_size in bytes, and number of initial pages to allocate.
    // Also takes model parameters like n_embd, n_layer, n_ctx for configuring KV store.
    // kv_mem_ctx is the ggml_context from which the page pool will be allocated.
    // This context's allocator for the paged_kv_buffer_type must be a paged allocator.
    // The paged_kv_buffer_type is the type of buffer that will store the pages.
    llama_paged_kv_cache(const struct llama_model_params & mparams,
                           const struct llama_context_params & cparams,
                           ggml_backend_buffer_type_t paged_kv_buffer_type,
                           struct ggml_context * kv_mem_ctx);


    // Destructor
    ~llama_paged_kv_cache() override;

    // llama_memory_i interface methods
    llama_memory_state_i * init_batch(const llama_batch & batch, uint32_t n_ubatch, bool embd_pooled, bool logits_all) override;
    llama_memory_state_i * init_full() override;
    llama_memory_state_i * init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    void seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override; // d is divisor

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    size_t state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    size_t state_read(llama_io_read_i & io, llama_seq_id seq_id = -1) override;

    // Helper to get underlying paged_cells - useful for state object
    llama_paged_kv_cells& get_paged_cells() { return paged_cells_; }
    const llama_paged_kv_cells& get_paged_cells() const { return paged_cells_; }

    // Get model/context parameters needed for KV layout
    uint32_t get_n_embd() const { return n_embd_; }
    uint32_t get_n_layer() const { return n_layer_; }
    uint32_t get_n_ctx() const { return n_ctx_; }
    uint32_t get_n_head_kv() const { return n_head_kv_; }
    uint32_t get_n_embd_head() const { return n_embd_head_; }


private:
    llama_paged_kv_cells paged_cells_;

    // Store necessary parameters from llama_model_params and llama_context_params
    // These are needed to determine the size and layout of K and V tensors for each token.
    uint32_t n_embd_;
    uint32_t n_layer_;
    uint32_t n_ctx_; // Max context size (not directly used for total pool size, but for initial page count)
    uint32_t n_head_kv_;
    uint32_t n_embd_head_; // n_embd / n_head
    ggml_type type_k_; // KV cache type for K
    ggml_type type_v_; // KV cache type for V

    // GGML specific members for managing the main page pool
    struct ggml_context * kv_mem_ctx_; // GGML context used for the page pool allocation
    ggml_backend_buffer_type_t paged_kv_buffer_type_; // Buffer type for paged KV cache pool
    struct ggml_tensor * main_page_pool_tensor_; // A GGML tensor representing the entire page pool
    uint8_t * main_page_pool_data_;      // Pointer to the data of the main page pool tensor
    size_t    main_page_pool_size_bytes_; // Total size of the allocated page pool

    size_t    default_page_size_bytes_;   // Calculated physical page size in bytes for llama_paged_kv_cells
    size_t    initial_page_count_;        // Calculated initial number of pages to "fill" from the pool

    // Helper to calculate the size of K AND V data for a single token across all layers.
    size_t get_kv_item_size_bytes() const;
};

// --- llama_paged_kv_cache_state ---

class llama_paged_kv_cache_state : public llama_memory_state_i {
public:
    llama_paged_kv_cache_state(llama_paged_kv_cache & cache, const llama_batch & batch, uint32_t n_ubatch_in, bool embd_pooled_in, bool logits_all_in);
    llama_paged_kv_cache_state(llama_paged_kv_cache & cache); // For init_full
    llama_paged_kv_cache_state(llama_paged_kv_cache & cache, llama_context * lctx_in, bool optimize_in); // For init_update

    ~llama_paged_kv_cache_state() override;

    // llama_memory_state_i interface methods
    bool next() override;
    void apply() override;
    const std::vector<llama_seq_id> & out_ids() const override;
    llama_kv_cache_view get_ubatch() override;
    llama_memory_status get_status() const override;

private:
    llama_paged_kv_cache & cache_; // Reference to the parent cache

    // State for init_batch
    llama_batch batch_ref_; // Reference or copy of the batch data
    uint32_t n_ubatch_total_;
    uint32_t current_ubatch_idx_;
    bool embd_pooled_;
    bool logits_all_;
    std::vector<llama_seq_id> current_out_ids_;
    llama_kv_cache_view current_kv_view_;

    // State for init_update
    llama_context * lctx_ref_; // Pointer to llama_context
    bool optimize_;

    // Common status
    llama_memory_status status_;

    // Helper to prepare current_kv_view_ for the current ubatch
    void prepare_kv_view_for_ubatch();
};


#endif // LLAMA_PAGED_KV_CACHE_H
