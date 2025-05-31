#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache_unified
//

class llama_kv_cache_unified : public llama_kv_cache {
public:
    static uint32_t get_padding(const llama_cparams & cparams);

    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    llama_kv_cache_unified(
            const llama_model &  model,
              layer_filter_cb && filter,
                    ggml_type    type_k,
                    ggml_type    type_v,
                         bool    v_trans,
                         bool    offload,
                     uint32_t    kv_size,
                     uint32_t    n_seq_max,
                     uint32_t    n_pad,
                     uint32_t    n_swa,
               llama_swa_type    swa_type);

    ~llama_kv_cache_unified() = default;

    //
    // llama_memory_i
    //

    void clear() override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    //
    // llama_kv_cache
    //

    llama_memory_state_ptr init_batch(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled,
            bool logits_all) override;

    llama_memory_state_ptr init_full() override;

    bool update(llama_context & lctx) override;

    void defrag_sched(float thold) override;

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // llama_kv_cache_unified specific API
    //

    uint32_t get_size() const;

    //
    // graph_build API
    //

    uint32_t get_n_kv() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il, uint32_t head_cur) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il, uint32_t head_cur) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the head locations
    // return empty vector on failure
    std::vector<uint32_t> prepare(const std::vector<llama_ubatch> & ubatches);

    // return the cell position where we can insert the ubatch
    // return -1 on failure to find a contiguous slot of kv cells
    int32_t find_slot(const llama_ubatch & ubatch) const;

    // emplace the ubatch context into slot: [head_cur, head_cur + ubatch.n_tokens)
    void apply_ubatch(uint32_t head_cur, const llama_ubatch & ubatch);

    //
    // set_input API
    //

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;
    };

    bool do_defrag = false;
    bool v_trans   = true;  // the value tensor is transposed

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    uint32_t head = 0;

    const uint32_t n_seq_max = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    llama_kv_cells_unified cells;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    // defrag
    struct {
        std::vector<uint32_t> ids;
    } defrag_info;

    // return true if cells have been moved
    bool defrag_prepare(int32_t n_max_nodes);

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    bool is_masked_swa(llama_pos p0, llama_pos p1) const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale) const;

    llm_graph_result_ptr build_graph_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_cgraph * gf) const;

    llm_graph_result_ptr build_graph_defrag(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_cgraph * gf) const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_kv_cache_unified_state : public llama_memory_state_i {
public:
    // used for errors
    llama_kv_cache_unified_state(llama_memory_status status);

    // used to create a full-cache state
    llama_kv_cache_unified_state(
            llama_memory_status status,
            llama_kv_cache_unified * kv);

    // used to create a state from a batch
    llama_kv_cache_unified_state(
            llama_memory_status status,
            llama_kv_cache_unified * kv,
            llama_sbatch sbatch,
            std::vector<uint32_t> heads,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_unified_state();

    //
    // llama_memory_state_i
    //

    bool next()  override;
    bool apply() override;

    std::vector<int64_t> & out_ids() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_unified_state specific API
    //

    uint32_t get_n_kv() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    const llama_memory_status status;

    llama_kv_cache_unified * kv;

    llama_sbatch sbatch;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<uint32_t> heads;
    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;

    // the beginning of the current slot in which the ubatch will be inserted
    int32_t head;
};
