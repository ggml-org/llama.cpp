#pragma once

#include "llama.h"
#include "llama-io.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-kv-cells.h"

#include "ggml-cpp.h"

#include <set>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

struct llama_kv_cache : public llama_memory_i {
    virtual ~llama_kv_cache() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a state object containing the ubatches and KV cache state required to process them
    // check the llama_memory_state_i::get_status() for the result
    virtual llama_memory_state_ptr init_batch(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled,
            bool logits_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_state_ptr init_full() = 0;

    // process any pending defrag/shift/etc. operations
    // optionally call once before processing a new batch
    // return true if any operations were performed
    virtual bool update(llama_context & lctx) = 0;

    // schedule a defrag if the fragmentation threshold is exceeded. otherwise, do nothing
    // TODO: change to
    //   llama_memory_state_ptr init_defrag(float thold) = 0;
    //
    virtual void defrag_sched(float thold) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    bool get_can_edit() const override { return get_can_shift(); }

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1) = 0;
};

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

//
// llama_kv_cache_unified_iswa
//

// utilizes two instances of llama_kv_cache_unified
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class llama_kv_cache_unified_iswa : public llama_kv_cache {
public:
    llama_kv_cache_unified_iswa(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad);

    ~llama_kv_cache_unified_iswa() = default;

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
    // llama_kv_cache_unified_iswa specific API
    //

    llama_kv_cache_unified * get_base() const;
    llama_kv_cache_unified * get_swa () const;

private:
    const llama_hparams & hparams;

    std::unique_ptr<llama_kv_cache_unified> kv_base;
    std::unique_ptr<llama_kv_cache_unified> kv_swa;
};

class llama_kv_cache_unified_iswa_state : public llama_memory_state_i {
public:
    // used for errors
    llama_kv_cache_unified_iswa_state(llama_memory_status status);

    // used to create a full-cache state
    llama_kv_cache_unified_iswa_state(
            llama_memory_status status,
            llama_kv_cache_unified_iswa * kv);

    // used to create a state from a batch
    llama_kv_cache_unified_iswa_state(
            llama_memory_status status,
            llama_kv_cache_unified_iswa * kv,
            llama_sbatch sbatch,
            std::vector<uint32_t> heads_base,
            std::vector<uint32_t> heads_swa,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_unified_iswa_state();

    //
    // llama_memory_state_i
    //

    bool next()  override;
    bool apply() override;

    std::vector<int64_t> & out_ids() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_unified_iswa_state specific API
    //

    const llama_kv_cache_unified_state * get_base() const;
    const llama_kv_cache_unified_state * get_swa()  const;

private:
    const llama_memory_status status;

    //llama_kv_cache_unified_iswa * kv;

    llama_sbatch sbatch;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    std::unique_ptr<llama_kv_cache_unified_state> state_base;
    std::unique_ptr<llama_kv_cache_unified_state> state_swa;
};

//
// llama_kv_cache_recurrent
//

// TODO: extract the KV cache state used for graph computation into llama_kv_cache_recurrent_state_i
//       see the implementation of llama_kv_cache_unified_state_i for an example how to do it
class llama_kv_cache_recurrent : public llama_kv_cache {
public:
    llama_kv_cache_recurrent(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   offload,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max);

    ~llama_kv_cache_recurrent() = default;

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

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of kv cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // TODO: temporary methods - they are not really const as they do const_cast<>, fix this
    int32_t s_copy(int i) const;
    float   s_mask(int i) const;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    // TODO: optimize for recurrent state needs
    struct kv_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to copy states
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const kv_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<kv_cell> cells;

    std::vector<ggml_tensor *> k_l; // per layer
    std::vector<ggml_tensor *> v_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_kv_cache_recurrent_state : public llama_memory_state_i {
public:
    // used for errors
    llama_kv_cache_recurrent_state(llama_memory_status status);

    // used to create a full-cache state
    llama_kv_cache_recurrent_state(
            llama_memory_status status,
            llama_kv_cache_recurrent * kv);

    // used to create a state from a batch
    llama_kv_cache_recurrent_state(
            llama_memory_status status,
            llama_kv_cache_recurrent * kv,
            llama_sbatch sbatch,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_recurrent_state();

    //
    // llama_memory_state_i
    //

    bool next()  override;
    bool apply() override;

    std::vector<int64_t> & out_ids() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_recurrent_state specific API
    //

    uint32_t get_n_kv() const;
    uint32_t get_head() const;
    uint32_t get_size() const;

    ggml_tensor * get_k_l(int32_t il) const;
    ggml_tensor * get_v_l(int32_t il) const;

    int32_t s_copy(int i) const;
    float   s_mask(int i) const;

private:
    const llama_memory_status status;

    llama_kv_cache_recurrent * kv;

    llama_sbatch sbatch;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
