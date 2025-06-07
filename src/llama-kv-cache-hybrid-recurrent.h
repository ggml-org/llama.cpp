#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache-recurrent.h"
#include "llama-kv-cache-unified.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <memory>
#include <vector>

//
// llama_kv_cache_hybrid_recurrent
//

// utilizes instances of llama_kv_cache_recurrent and llama_kv_cache_unified to
//   support models where each layer may be either attention-based or recurrent

class llama_kv_cache_hybrid_recurrent : public llama_memory_i {
public:
    llama_kv_cache_hybrid_recurrent(
            const llama_model & model,
                                /* attn */
                    ggml_type   attn_type_k,
                    ggml_type   attn_type_v,
                         bool   attn_v_trans,
                     uint32_t   attn_kv_size,
                     uint32_t   attn_n_pad,
                     uint32_t   attn_n_swa,
               llama_swa_type   attn_swa_type,
                                /* recurrent */
                    ggml_type   recurrent_type_k,
                    ggml_type   recurrent_type_v,
                     uint32_t   recurrent_kv_size,
                                /* common */
                     uint32_t   n_seq_max,
                         bool   offload);

    ~llama_kv_cache_hybrid_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_state_ptr init_batch(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled,
            bool logits_all) override;

    llama_memory_state_ptr init_full() override;

    llama_memory_state_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear() override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // llama_kv_cache_hybrid_recurrent specific API
    //

    llama_kv_cache_unified   * get_kv_attn     () const;
    llama_kv_cache_recurrent * get_kv_recurrent() const;

private:
    const llama_hparams & hparams;

    const std::unique_ptr<llama_kv_cache_unified>   kv_attn;
    const std::unique_ptr<llama_kv_cache_recurrent> kv_recurrent;
};

class llama_kv_cache_hybrid_recurrent_state : public llama_memory_state_i {
public:
    using llama_kv_cache_unified_state_ptr   = std::unique_ptr<llama_kv_cache_unified_state>;
    using llama_kv_cache_recurrent_state_ptr = std::unique_ptr<llama_kv_cache_recurrent_state>;

    // init failure
    explicit llama_kv_cache_hybrid_recurrent_state(llama_memory_status status);

    // init full
    explicit llama_kv_cache_hybrid_recurrent_state(llama_kv_cache_hybrid_recurrent * kv);

    // init update
    explicit llama_kv_cache_hybrid_recurrent_state(
        llama_kv_cache_hybrid_recurrent * kv,
           llama_kv_cache_unified_state * state_unified,
         llama_kv_cache_recurrent_state * state_recurrent);

    // init success
    llama_kv_cache_hybrid_recurrent_state(
        llama_kv_cache_hybrid_recurrent * kv,
                           llama_sbatch   sbatch,
                  std::vector<uint32_t>   heads_attn,
              std::vector<llama_ubatch>   ubatches);

    ~llama_kv_cache_hybrid_recurrent_state() = default;

    bool next()  override;
    bool apply() override;

    std::vector<int64_t> & out_ids() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_hybrid_recurrent_state
    //

    const llama_kv_cache_unified_state   * get_state_attn     () const;
    const llama_kv_cache_recurrent_state * get_state_recurrent() const;

private:
    const llama_memory_status status;

    llama_kv_cache_hybrid_recurrent * kv;

    llama_sbatch sbatch;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<uint32_t>     heads_attn;
    std::vector<llama_ubatch> ubatches;

    const llama_kv_cache_unified_state_ptr   state_attn;
    const llama_kv_cache_recurrent_state_ptr state_recurrent;
};
