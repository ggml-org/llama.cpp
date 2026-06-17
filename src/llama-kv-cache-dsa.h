#pragma once

#include "llama-kv-cache.h"

#include <vector>

//
// llama_kv_cache_dsa
//

// utilizes two instances of llama_kv_cache:
// - the first instance is for caching key tensors of the model,
// - the second instance is for caching lightning indexer key tensors

class llama_kv_cache_dsa : public llama_memory_i {
public:
    llama_kv_cache_dsa(
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
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache_dsa() = default;

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
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache_dsa specific API
    //

    llama_kv_cache * get_mla() const;
    llama_kv_cache * get_lid() const;

private:
    // we keep indexer KV cache hparams instance here as llama_kv_cache stores only reference to it
    llama_hparams hparams_lid;
    const uint32_t n_stream  = 1;

    std::unique_ptr<llama_kv_cache> kv_mla;
    std::unique_ptr<llama_kv_cache> kv_lid;
};

class llama_kv_cache_dsa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    llama_kv_cache_dsa_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_dsa_context(
            llama_kv_cache_dsa * kv);

    // used to create an update context
    llama_kv_cache_dsa_context(
            llama_kv_cache_dsa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_dsa_context(
            llama_kv_cache_dsa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_ik,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_dsa_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_dsa_context specific API
    //

    const llama_kv_cache_context * get_mla() const;
    const llama_kv_cache_context * get_lid()  const;

private:
    //llama_kv_cache_dsa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_mla;
    const llama_memory_context_ptr ctx_lid;

    const llama_memory_status status;
};
