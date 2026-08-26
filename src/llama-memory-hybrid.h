#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid
//

// utilizes instances of llama_memory_recurrent and llama_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class llama_memory_hybrid : public llama_memory_i {
public:
    llama_memory_hybrid(
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
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr,
                            /* optional per-token indexer keys (QSA), attention layers only */
    const layer_filter_cb & filter_lid  = nullptr,
                            /* optional second recurrent state (qwen4exp PLE), its own layers */
    const layer_filter_cb & filter_ple  = nullptr);

    ~llama_memory_hybrid() = default;

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
    // llama_memory_hybrid specific API
    //

    llama_kv_cache * get_mem_attn() const;
    llama_memory_recurrent * get_mem_recr() const;

    // null unless the model asked for an indexer cache
    llama_kv_cache * get_mem_lid() const;
    llama_memory_recurrent * get_mem_ple() const;

private:
    const llama_hparams & hparams;

    // the indexer cache stores one key per token, so it needs its own head layout
    llama_hparams hparams_lid;
    llama_hparams hparams_ple;

    const std::unique_ptr<llama_kv_cache> mem_attn;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;
    std::unique_ptr<llama_kv_cache> mem_lid;
    std::unique_ptr<llama_memory_recurrent> mem_ple;
};

class llama_memory_hybrid_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // init failure
    explicit llama_memory_hybrid_context(llama_memory_status status);

    // init full
    explicit llama_memory_hybrid_context(llama_memory_hybrid * mem);

    // init update
    explicit llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize);

    // init success
    llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
                  slot_info_vec_t   sinfos_lid,
        std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_context() = default;

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_hybrid_context
    //

    const llama_kv_cache_context * get_attn() const;
    const llama_memory_recurrent_context * get_recr() const;

    // null unless the model asked for an indexer cache
    const llama_kv_cache_context * get_lid() const;

    // null unless the model asked for a second recurrent state
    const llama_memory_recurrent_context * get_ple() const;

    // QSA selects blocks of keys over token positions, but the caches store them by cell, and the
    // two only agree for an append-only single-sequence cache. These maps let the graph translate.
    // ref: src/llama-kv-cache-msa.cpp

    // length of the position axis: past the largest position in the cache, padded so that the
    // graph shape stays constant across batches and can be reused
    uint32_t get_n_pos(uint32_t n_pad, uint32_t n_seq_max) const;

    // I32/F32 [n_pos, n_stream] pos -> cell, of the indexer cache when lid, else of the attention
    // cache. Positions with no cell, and cells past the first n_rows (all the graph views), get 0,
    // which aliases cell 0. That is only harmless while a sequence occupies a contiguous range of
    // positions, which is what an append-only cache gives. A hole in the middle of the range (a
    // mid-range seq_rm, a ranged seq_cp) makes a consumer read cell 0 for the missing positions.
    void set_input_pos_cell(ggml_tensor * dst, const llama_ubatch * ubatch, bool lid, int64_t n_rows) const;

    // I32 [n_kv, n_stream] attention cell -> pos/div. Empty and other-sequence cells get 0.
    void set_input_cell_blk(ggml_tensor * dst, const llama_ubatch * ubatch, int32_t div) const;

private:
    llama_memory_hybrid * mem = nullptr;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_attn;
    const llama_memory_context_ptr ctx_recr;
    const llama_memory_context_ptr ctx_lid;
    const llama_memory_context_ptr ctx_ple;

    const llama_memory_status status;
};
