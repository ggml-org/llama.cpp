#pragma once

#include "llama-kv-cache.h"

struct llama_model;

// External paged KV cache backed by Kapsl's GpuBlockPool.
//
// This class is the integration point for Kapsl-owned KV memory. The initial
// skeleton validates the ABI and fails fast at decode time until the graph-facing
// K/V views and CUDA kernels are mapped to Kapsl physical blocks.
class llama_kv_cache_kapsl : public llama_memory_i {
public:
    llama_kv_cache_kapsl(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
    llama_kapsl_kv_pool_desc *   pool,
                     uint64_t   session_id);

    ~llama_kv_cache_kapsl() override;

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
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // Promote newly computed KV blocks to the prefix cache.
    //
    // Must be called after a successful llama_decode() to make the session's
    // freshly written KV blocks available for future requests.  No-op when the
    // pool descriptor does not have a `promote_prefix` callback.
    //
    // n_new_logical: number of logical blocks beyond the cached prefix to donate.
    void promote_to_cache(uint32_t n_new_logical);

private:
    bool reserve_for_tokens(uint32_t tokens_needed);
    bool reserve_for_tokens_prefix(const std::vector<int32_t> & tokens);
    void release_session();

    llama_memory_context_ptr make_status_context(llama_memory_status status) const;

    llama_kapsl_kv_pool_desc * pool;
    uint64_t session_id;
    uint32_t * block_table_device = nullptr;
    uint32_t n_reserved_blocks    = 0;
    uint32_t n_reserved_tokens    = 0;
    bool     has_reservation      = false;

    // Prefix-cache state — valid only when pool->reserve_prefix != nullptr.
    uint32_t              n_prefix_hits = 0;  // leading logical blocks from cache
    std::vector<int32_t>  last_tokens;         // token sequence from last init_batch
};
