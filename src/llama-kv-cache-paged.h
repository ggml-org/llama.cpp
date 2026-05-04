#pragma once

#include "llama-batch.h"
#include "llama-block-manager.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-sequence-group.h"

#include <cmath>

//
// llama_kv_cache_paged
//

class llama_kv_cache_paged : public llama_memory_i {
  public:
    llama_kv_cache_paged(uint32_t head_dim,
                         uint32_t n_head_kv,
                         uint32_t block_size,
                         uint32_t n_layers,
                         uint32_t n_ubatch,
                         uint32_t n_seq_max);

    void init(ggml_backend_t backend_gpu,
              ggml_backend_t backend_cpu,
              enum ggml_type type,
              uint32_t       n_gpu_blocks,
              uint32_t       n_cpu_blocks,
              float          watermark);  // percentage

    bool allocate(int32_t num_tokens, llama_sequence_group & group);
    void free_blocks(llama_sequence_group & group);
    bool swap_in(llama_sequence_group & group);
    bool swap_out(llama_sequence_group & group);

    void     set_paged_batch_info(const llama_paged_batch_info * info);
    uint32_t get_num_gpu_blocks() const;

    //
    // llama_memory_i
    //
    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    struct ggml_tensor * get_kv_tensor(int layer_idx) const;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;

    void seq_cp(llama_seq_id /*seq_id_src*/,
                llama_seq_id /*seq_id_dst*/,
                llama_pos /*p0*/,
                llama_pos /*p1*/) override { /* implement later CoW mechanism */
    }

    void seq_keep(llama_seq_id /*seq_id*/) override {}

    void seq_add(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/, llama_pos /*shift*/) override {}

    void seq_div(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/, int /*d*/) override {}

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load
    void state_write(llama_io_write_i & /*io*/,
                     llama_seq_id /*seq_id*/         = -1,
                     llama_state_seq_flags /*flags*/ = 0) const override {}

    void state_read(llama_io_read_i & /*io*/,
                    llama_seq_id /*seq_id*/         = -1,
                    llama_state_seq_flags /*flags*/ = 0) override {}

    //
    // Helpers to llama_memory_i
    //
    void set_seq_min_pos(llama_seq_id seq_id, llama_pos new_min);
    void set_seq_max_pos(llama_seq_id seq_id, llama_pos new_max);

  private:
    void concat_block_ids(llama_block_ids & to_block_table, const llama_block_ids & from_block_table);
    void do_block_copy(const llama_block_ids & src_ids, const llama_block_ids & new_ids, bool to_gpu);

    // Master physical buffer
    // For CUDA: memory is interleaved
    // For other backends: we treat the exact same memory buffer as two virtual views
    std::vector<struct ggml_tensor *> kv_gpu_layers;
    std::vector<struct ggml_tensor *> kv_cpu_layers;

    enum ggml_type kv_type;

    llama_block_manager block_manager;

    // Non-owning pointer to the batch currently being processed.
    // Lifetime: set by the scheduler at the end of step(), cleared at the
    // start of the next step() (before the batch's paged_* arrays are freed).
    // The ordering in llama_paged_scheduler_impl::clear_batch is load-bearing;
    // do not reorder without updating init_batch's contract.
    const llama_paged_batch_info * last_paged_info = nullptr;

    const uint32_t head_dim;
    const uint32_t n_heads_kv;
    const uint32_t block_size;
    const uint32_t n_layers;
    const uint32_t n_ubatch;
    const uint32_t n_seq_max;
    uint32_t       num_gpu_blocks;
    uint32_t       num_cpu_blocks;
    uint32_t       block_bytes;

    ggml_backend_t gpu_backend;
    ggml_backend_t cpu_backend;

    struct seq_range {
        llama_pos min = -1;
        llama_pos max = -1;
    };

    std::unordered_map<llama_seq_id, seq_range> sequence_positions;
};

class llama_kv_cache_paged_context : public llama_memory_context_i {
  public:
    llama_kv_cache_paged_context(llama_kv_cache_paged * parent, const std::vector<llama_ubatch> & in_ubatch) :
        manager(parent),
        ubatches(in_ubatch) {
        i_cur = 0;
    }

    llama_kv_cache_paged_context(llama_memory_status status) : status(status) {}

    void    set_batch_data(const llama_paged_batch_info & info);
    int32_t get_n_tokens() const;
    int32_t get_batch_size() const;
    int32_t get_max_blocks() const;

    int32_t * get_write_slots() const;
    int32_t * get_block_table() const;
    int32_t * get_context_lens() const;
    int32_t * get_batch_offsets() const;
    int32_t * get_batch_lens() const;

    void set_n_tokens(int32_t new_n_tokens);
    void set_batch_size(int32_t new_batch_size);
    void set_max_blocks(int32_t new_max_blocks);

    struct ggml_tensor * get_k(int layer_idx) const;
    struct ggml_tensor * get_v(int layer_idx) const;

    //
    // llama_memory_context_i
    //
    bool                 next() override;
    bool                 apply() override;
    const llama_ubatch & get_ubatch() const override;

    llama_memory_status get_status() const override { return status; }

  private:
    const llama_kv_cache_paged * manager;

    //
    // batch processing context
    //
    std::vector<llama_ubatch> ubatches;
    size_t                    i_cur = 0;      // index of ubatch to process

    int32_t * paged_write_slots   = nullptr;  // [n_tokens]
    int32_t * paged_block_table   = nullptr;  // [batch_size, max_blocks]
    int32_t * paged_context_lens  = nullptr;  // [batch_size]
    int32_t * paged_batch_offsets = nullptr;  // [batch_size]
    int32_t * paged_batch_lens    = nullptr;  // [batch_size]

    int32_t n_tokens   = 0;
    int32_t batch_size = 0;
    int32_t max_blocks = 0;

    llama_memory_status status = LLAMA_MEMORY_STATUS_SUCCESS;
};
