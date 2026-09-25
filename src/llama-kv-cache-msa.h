#pragma once

#include "llama-kv-cache.h"

#include <vector>

// llama_kv_cache_msa

// uses two instances of llama_kv_cache, one for K/V tensors, and one for the MSA indexer tensors
// both receive identical sequence operations and identical ubatches, so their cell layouts stay in synced.
// the context also exposes per-ubatch pos - cell translation maps populated from llama_kv_cells via
// llama_kv_cache::get_cells(), which the model graph uses to run MSA block selection in position space

class llama_kv_cache_msa : public llama_memory_i {
public:
    llama_kv_cache_msa(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const layer_filter_cb & filter_idx,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache_msa() = default;

    // llama_memory_i

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

    // llama_kv_cache_msa specific API

    llama_kv_cache * get_base() const;
    llama_kv_cache * get_idx () const;

    uint32_t       get_n_pad()    const { return n_pad; }
    uint32_t       get_n_seq_max() const { return n_seq_max; }
    uint32_t       get_n_swa()    const { return n_swa; }
    llama_swa_type get_swa_type() const { return swa_type; }

private:
    // keep the indexer KV cache hparams instance here as llama_kv_cache stores only a reference
    llama_hparams hparams_idx;

    const uint32_t n_stream  = 1;
    const uint32_t n_seq_max = 1;
    const uint32_t n_pad     = 1;

    const uint32_t       n_swa    = 0;
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    std::unique_ptr<llama_kv_cache> kv_base;
    std::unique_ptr<llama_kv_cache> kv_idx;
};

class llama_kv_cache_msa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    llama_kv_cache_msa_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_msa_context(
            llama_kv_cache_msa * kv);

    // used to create an update context
    llama_kv_cache_msa_context(
            llama_kv_cache_msa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_msa_context(
            llama_kv_cache_msa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_idx,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_msa_context();

    // llama_memory_context_i

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    // llama_kv_cache_msa_context specific API

    const llama_kv_cache_context * get_base() const;
    const llama_kv_cache_context * get_idx () const;

    // max position currently present in the cache plus one, padded MSA blocks are defined over token positions
    // so the block-selection tensors are sized by this value rather than by the number of cells
    uint32_t get_n_pos() const;

    // position <-> cell translation maps, populated from the base cache cells
    // the model graph relates cache contents to token positions only through these per ubatch inputs
    // value for empty or other-sequence cells is 0 so consumers must mask them
    void set_input_cell_pos(ggml_tensor * dst, const llama_ubatch * ubatch, int32_t div) const;
    // positions without a cell map to cell 0, consumers must mask them assumes one sequence per stream
    void set_input_pos_slot(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_pos_mask(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    llama_kv_cache_msa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_base;
    const llama_memory_context_ptr ctx_idx;

    const llama_memory_status status;
};
