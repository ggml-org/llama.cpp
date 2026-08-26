#pragma once

#include "llama-memory-hybrid.h"

#include <map>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus qwen4exp-specific per-token indexer and PLE host state.
// Keeping these in a distinct memory type leaves the shared hybrid/KV paths used by
// every other architecture unchanged. The indexer cache takes the attention cache's
// exact slot layout, so cell j identifies the same token in both caches.

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

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

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    llama_kv_cache * get_mem_idx() const;
    llama_memory_recurrent * get_mem_ple() const;

    // PLE needs logical token predecessors, not just primary positions: M-RoPE
    // image patches can share a position. Prefixes are captured during apply().
    using ple_prefix_map = std::map<llama_seq_id, std::vector<llama_token>>;
    void ple_apply_ubatch(const llama_ubatch & ubatch, ple_prefix_map & prefix);

private:
    struct ple_history_entry {
        llama_pos   pos;
        llama_token token;
    };
    using ple_seq_history = std::vector<ple_history_entry>;

    void ple_append  (llama_seq_id seq_id, llama_pos pos, llama_token token);
    void ple_seq_rm  (llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    void ple_seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
    void ple_seq_keep(llama_seq_id seq_id);
    void ple_seq_add (llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift);
    void ple_seq_div (llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d);
    void ple_state_write(llama_io_write_i & io, llama_seq_id seq_id) const;
    void ple_state_read (llama_io_read_i  & io, llama_seq_id seq_id);

    const llama_hparams & hparams;
    const uint32_t n_seq_max;

    // The indexer stores one key head per full-attention layer and needs a
    // stable hparams object because llama_kv_cache retains it by reference.
    llama_hparams hparams_idx;
    const std::unique_ptr<llama_kv_cache> mem_idx;

    // PLE convolution history is mirrored and independent of the TP-sharded
    // GDN state. Combining them in one row makes neither placement correct.
    const std::unique_ptr<llama_memory_recurrent> mem_ple;

    std::map<llama_seq_id, ple_seq_history> ple_history;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    explicit llama_memory_hybrid_idx_context(llama_memory_status status);
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    bool next()  override;
    bool apply() override;
    llama_memory_status get_status() const override;

    const llama_kv_cache_context * get_idx() const;
    const llama_memory_recurrent_context * get_ple() const;
    uint32_t get_n_stream() const;

    // QSA is safe only when each stream has one sequence with unique,
    // contiguous primary positions. Other layouts use dense attention.
    bool qsa_compatible(const llama_ubatch * ubatch, uint32_t ratio) const;
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio) const;

    llama_token ple_prefix_token(llama_seq_id seq_id, size_t lookback, llama_token fallback) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // Expected stream ids for each actual ubatch, captured before slot infos are
    // moved into ctx_idx. Empty for reserve/update contexts.
    const std::vector<std::vector<llama_seq_id>> stream_ids_ubatch;

    const llama_memory_context_ptr ctx_idx;
    const llama_memory_context_ptr ctx_ple;
    size_t i_cur = 0;

    llama_memory_hybrid_idx::ple_prefix_map ple_prefix;
};
