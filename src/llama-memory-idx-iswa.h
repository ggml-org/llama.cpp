#pragma once

#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-kpool.h"

#include <memory>
#include <vector>

//
// llama_memory_idx_iswa
//

// llama_kv_cache_iswa plus an indexer cache over the base cells, for block-sparse attention (hy_v4_flash)
// only the non-SWA layers own an indexer

class llama_memory_idx_iswa : public llama_memory_i {
public:
    llama_memory_idx_iswa(
            const llama_model & model,
                         bool   no_alloc,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                                /* layer filters */
        const layer_filter_cb & filter,
                                /* must select non-SWA layers only */
        const layer_filter_cb & filter_idx);

    //
    // llama_memory_i
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
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_idx_iswa specific API
    //

    llama_kv_cache_iswa  * get_mem_attn() const;
    llama_kv_cache_kpool * get_mem_idx () const;

private:
    // forget seq_id (all of it if seq_id < 0) in both caches at once, so a failed restore cannot leave them out of step
    void state_drop(llama_seq_id seq_id);

    const std::unique_ptr<llama_kv_cache_iswa>  mem_attn;
    const std::unique_ptr<llama_kv_cache_kpool> mem_idx;
};

// it is the iSWA context itself, so the stock iSWA graph input works on it
class llama_memory_idx_iswa_context : public llama_kv_cache_iswa_context {
public:
    using kpool_access    = llama_kv_cache_kpool_context::kpool_access;
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_idx_iswa_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_idx_iswa_context(llama_memory_idx_iswa * mem);

    // used to create an update context
    llama_memory_idx_iswa_context(
            llama_memory_idx_iswa * mem,
                    llama_context * lctx,
                             bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_idx_iswa_context(
            llama_memory_idx_iswa * mem,
                  slot_info_vec_t   sinfos_base,
                  slot_info_vec_t   sinfos_swa,
                  slot_info_vec_t   sinfos_idx,
        std::vector<llama_ubatch>   ubatches);

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status get_status() const override;

    //
    // llama_memory_idx_iswa_context specific API, see llama_kv_cache_kpool_context
    //

    const llama_kv_cache_context * get_idx() const { return kpool.get_idx(); }

    uint32_t get_n_kpool    () const { return kpool.get_n_kpool();     }
    uint32_t get_n_kpool_new() const { return kpool.get_n_kpool_new(); }

    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const {
        return kpool.get_kpool_access(ctx, il, n_embd);
    }

    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
        kpool.set_input_kpool(pool_cells, pool_idxs, pool_mask, tail_idxs, sel_mask, new_pool_idxs, new_pool_rep, ubatch, new_pool_pos);
    }

private:
    llama_kv_cache_kpool_context kpool;

    const llama_memory_status status;
};
