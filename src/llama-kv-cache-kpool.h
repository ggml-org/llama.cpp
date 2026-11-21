#pragma once

#include "llama-kv-cache.h"

#include <array>
#include <limits>
#include <memory>
#include <vector>

//
// llama_kv_cache_kpool
//

// indexer cache with one key per token, and the pools of kpool tokens over it, for block-sparse attention
// a side buffer over an attention cache: same size, padding, streams and slots, so cell j is one token in both

class llama_kv_cache_kpool {
public:
    llama_kv_cache_kpool(
            const llama_model & model,
                         bool   no_alloc,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const llama_kv_cache::layer_filter_cb & filter);

    // Defined out of line because layout is incomplete here.
    ~llama_kv_cache_kpool();

    llama_kv_cache * get_cache() const { return cache.get(); }

    // The model's indexer pool size, 0 for a per-token indexer.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

    // Whether pools are kpool consecutive cells in sequence order (qwen4exp) instead of kpool consecutive positions.
    bool get_kpool_by_order() const { return hparams_idx.indexer_kpool_by_order; }

    // the memory_i edits of the cache, which also stale the pooled keys they regroup
    void clear   (bool data);
    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1);
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
    void seq_keep(llama_seq_id seq_id);
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift);
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d);

    // drop seq_id without asking anything, to undo a failed restore
    void state_clear(llama_seq_id seq_id);

    // the restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    void state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, const llama_kv_cache::slot_info_vec_t & sinfos_attn);

    // Which cells of a sequence make up which pool of kpool consecutive positions (or cells, in order mode).
    // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
    // sequence's first one, so a ubatch only ever appends to it. A sequence edit drops it from the stale
    // position on, or all of it for an edit at or before the first position, see stale.
    struct layout;

    const layout & layout_update();
    const layout & layout_get() const;

    // The pooled keys persist in the cache across batches. A sequence edit can regroup the pools
    // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;

    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & stale_get() const { return stale; }
    void stale_clear() { stale.fill(POS_CLEAN); }

private:
    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> cache;

    // unique_ptr because layout is incomplete here
    std::unique_ptr<layout> lay;

    // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
    void stale_set(llama_seq_id seq_id, llama_pos p0);

    // the position an edit at p0 stales the sequence from
    llama_pos stale_pos(llama_seq_id seq_id, llama_pos p0) const;

    stale_pos_t stale = stale_pos_clean();
};

// the indexer part of a memory context: the cache context plus which pools each ubatch must re-pool
class llama_kv_cache_kpool_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llama_kv_cache_kpool_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors, and by memories without an indexer
    llama_kv_cache_kpool_context();

    // used to create a full-cache context
    explicit llama_kv_cache_kpool_context(llama_kv_cache_kpool * mem);

    // used to create an update context
    llama_kv_cache_kpool_context(
            llama_kv_cache_kpool * mem,
                   llama_context * lctx,
                            bool   optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_kpool_context(
            llama_kv_cache_kpool * mem,
                 slot_info_vec_t   sinfos_idx,
       std::vector<llama_ubatch>   ubatches);

    ~llama_kv_cache_kpool_context(); // Defined out of line because kpool_state is incomplete here.

    // follow the owning context's next() / apply(), res is the owner's own apply() result
    void next();
    bool apply(bool res);

    llama_memory_status get_status() const;

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // complete pools of kpool cells per sequence, scored as whole pools
    uint32_t get_n_kpool    () const; // Padded pool count, where the last pool is always unused.
    uint32_t get_n_kpool_new() const; // Pools to re-pool this ubatch, padded to a stable bound, never below 1.
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    // tail_idxs (I32 [n_tail, n_tokens]): n_tail = kpool - 1 keeps the incomplete tail, n_tail = kpool also reserves the current pool
    // sel_mask (F32 [n_sel, 1, 1, n_tokens], can be null): 0 for the live selection slots, -inf for the dead ones
    // new_pool_pos (I32 [n_new] or [4*n_new] for M-RoPE): position of each new pool's first member, for pooled keys rotated at pooling time
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr) const;

private:
    llama_kv_cache_kpool * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
    // sits in cell idxs[s][i] of stream strm[s] of sinfos_kpool[u], and several cells can share a position
    const slot_info_vec_t sinfos_kpool;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the owner's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llama_kv_cache_kpool::stale_pos_t stale_batch = llama_kv_cache_kpool::stale_pos_clean();
};
