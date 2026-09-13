#pragma once

#include "llama-memory-hybrid.h"

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

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

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias,
                       ggml_tensor * dirty_cells = nullptr,
                       ggml_tensor * dirty_pos   = nullptr,
                       ggml_tensor * dirty_rows  = nullptr) const;

    // [TAG_QSA_POOLED_CACHE] cache of the indexer's block summary keys (mean-pooled,
    // normalized, roped), one f32 row per position block, written by the graph via set_rows.
    // Only complete blocks are scored and a complete block's members never change, so rows are
    // write-once per content epoch. Validity is a per-sequence block watermark; seq_rm clamps
    // it and replay recomputes the range. Rows at or beyond the watermark may hold stale but
    // finite data and are masked by the -inf bias. Single-stream memories only.
    ggml_tensor * get_pooled_k(int32_t il) const;              // nullptr: no indexer / multi-stream
    uint32_t get_pooled_rows() const { return pooled_rows; }   // rows per stream, incl. trailing dustbin row

    // blocks of seq_id whose pooled rows are known valid; mutable like a cache's bookkeeping
    int64_t & pooled_valid(llama_seq_id seq_id) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // [TAG_QSA_POOLED_CACHE] storage + watermarks; empty unless the model has an indexer
    // one buffer per device: each layer's rows must live with that layer's indexer cache,
    // or every decode step would copy the rows across the (slow) inter-GPU links
    std::vector<ggml_context_ptr>        pooled_ctxs;
    std::vector<ggml_backend_buffer_ptr> pooled_bufs;
    std::map<int32_t, ggml_tensor *> pooled_k;

    uint32_t pooled_rows  = 0;
    uint32_t pooled_ratio = 0;

    mutable std::unordered_map<llama_seq_id, int64_t> pooled_w;

    // clamp helpers, one per llama_memory_i operation that can invalidate rows
    void pooled_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    void pooled_reset(llama_seq_id seq_id);   // -1 resets every sequence
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // [TAG_QSA_POOLED_CACHE] the dirty_* tensors are optional: when given, the fill also
    // resolves which blocks must be (re)pooled this ubatch — the range from the sequence's
    // watermark to its last complete block — and advances the watermark.
    //   dirty_cells I32 [ratio*n_dirty_max, ns] cells of each block to (re)pool, 0-padded
    //   dirty_pos   I32 [4*n_dirty_max*ns]      mrope position rows of those blocks
    //   dirty_rows  I64 [n_dirty_max*ns]        pooled-cache rows to write, dustbin-padded
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias,
                       ggml_tensor * dirty_cells = nullptr,
                       ggml_tensor * dirty_pos   = nullptr,
                       ggml_tensor * dirty_rows  = nullptr) const;

    // [TAG_QSA_POOLED_CACHE] pooled tensor for il, or nullptr when the cache is unavailable
    // (no indexer, multi-stream memory, or a non-batch context)
    ggml_tensor * get_pooled_k(int32_t il) const;

    uint32_t get_pooled_rows() const;

    // capacity the dirty tables need for this ubatch: completed blocks plus pending refill
    // below the watermark; stable at 1 during steady decode so graph reuse holds
    uint32_t qsa_pooled_n_dirty_max(const llama_ubatch & ubatch, uint32_t ratio) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
