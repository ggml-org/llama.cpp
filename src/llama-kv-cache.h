#pragma once

#include "llama.h"
#include "llama-io.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-kv-cells.h"

#include "ggml-cpp.h"

#include <set>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_ubatch;
struct llama_sbatch;
struct llama_model;
struct llama_context;

struct llama_kv_cache : public llama_memory_i {

    // some child types need to perform different caching for each layer, so
    // this callback can be used to determine which layers a given cache should
    // be used for
    using layer_filter_cb = std::function<bool(int32_t il)>;

    virtual ~llama_kv_cache() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // check the llama_memory_decode_state_i::get_status() for the result
    virtual llama_memory_decode_state_ptr init(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled,
            bool logits_all) = 0;

    // process any pending defrag/shift/etc. operations
    // optionally call once before processing a new batch
    // return true if any operations were performed
    virtual bool update(llama_context & lctx) = 0;

    // schedule a defrag if the fragmentation threshold is exceeded. otherwise, do nothing
    virtual void defrag_sched(float thold) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    // TODO: remove
    virtual void set_full() = 0;

    // sometimes it is useful to check whether a cache can remove a sequence
    // before attempting to mutate the cache (eg a hybrid cache with multiple
    // children to keep in sync)
    virtual bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const = 0;

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

    llama_memory_decode_state_ptr init(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled,
            bool logits_all) override;

    bool update(llama_context & lctx) override;

    void defrag_sched(float thold) override;

    void set_full() override;

    bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const override;

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // llama_kv_cache_unified specific API
    //

    uint32_t get_n()    const;
    uint32_t get_size() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the current head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il) const;

    // find places for the provided ubatches in the cache, returns the head locations
    // return empty vector on failure
    std::vector<uint32_t> prepare(const std::vector<llama_ubatch> & ubatches);

    // return the cell position where we can insert the ubatch
    // return -1 on failure to find a contiguous slot of kv cells
    int32_t find_slot(const llama_ubatch & ubatch) const;

    // emplace the ubatch context into cells [head_cur, head_cur + ubatch.n_tokens)
    // updates head = head_cur
    void fill_slot(uint32_t head_cur, const llama_ubatch & ubatch);

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

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())

    // computed before each graph build
    // TODO: cells should start to maintain this value dynamically based on the edits
    uint32_t n = 0;

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
                     uint32_t   n_batch,
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

    llama_memory_decode_state_ptr init(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled,
            bool logits_all) override;

    bool update(llama_context & lctx) override;

    void defrag_sched(float thold) override;

    void set_full() override;

    bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const override;

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // llama_kv_cache_unified_iswa specific API
    //

    llama_kv_cache_unified * get_kv_base() const;
    llama_kv_cache_unified * get_kv_swa () const;

private:
    const llama_hparams & hparams;

    std::unique_ptr<llama_kv_cache_unified> kv_base;
    std::unique_ptr<llama_kv_cache_unified> kv_swa;
};

//
// llama_kv_cache_recurrent
//

class llama_kv_cache_recurrent : public llama_kv_cache {
public:
    llama_kv_cache_recurrent(
            const llama_model &  model,
              layer_filter_cb && filter,
                    ggml_type    type_k,
                    ggml_type    type_v,
                         bool    offload,
                     uint32_t    kv_size,
                     uint32_t    n_seq_max);

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

    llama_memory_decode_state_ptr init(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled,
            bool logits_all) override;

    bool update(llama_context & lctx) override;

    void defrag_sched(float thold) override;

    void set_full() override;

    bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const override;

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

//
// llama_kv_cache_hybrid
//

// utilizes multiple different cache types with each layer assigned to exactly
//   one cache. This is typically used for hybrid attention / recurrent caching

class llama_kv_cache_hybrid : public llama_kv_cache {
public:

    struct child_cache {
        std::unique_ptr<llama_kv_cache> child;
        std::vector<size_t>             layer_ids;

        child_cache(std::unique_ptr<llama_kv_cache> child_, std::vector<size_t> layer_ids_)
            : child(std::move(child_)), layer_ids(std::move(layer_ids_)) {}
    };

    llama_kv_cache_hybrid(
        const llama_hparams            & hparams,
              std::vector<child_cache>   children);

    virtual ~llama_kv_cache_hybrid() = default;

    // getters for specific child cache type
    // NOTE: This will fail if there are multiple of the given type
    template<typename child_t>
    const child_t * get_child_cache() const {
        const child_t * child = nullptr;
        for (const auto & child_cache : m_children) {
            const child_t * child_cast = dynamic_cast<const child_t *>(child_cache.get());
            if (child_cast) {
                GGML_ASSERT(!child);
                child = child_cast;
            }
        }
        return child;
    }

    //
    // llama_memory_i
    //

    void clear() override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos delta) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    //
    // llama_kv_cache
    //

    void restore() override;
    void commit()  override;

    bool update(llama_context & ctx) override;

    void defrag_sched(float thold) override;

    void set_full() override;

    bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const override;

    llama_sbatch sbatch_init(const llama_batch & batch, bool logits_all) override;

    llama_ubatch ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const override;

    // updates the cache head
    // Note: On success, it's important that cache.head points
    // to the first cell of the slot.
    bool find_slot(const llama_ubatch & batch) override;

    int32_t get_n_tokens()   const override;
    int32_t get_used_cells() const override;

    // TODO: better data structures to reduce the cost of this operation
    llama_pos get_pos_max() const override;

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1) override;

private:

    const llama_hparams                             & m_hparams;
    const std::set<std::unique_ptr<llama_kv_cache>>   m_children; // Ordered for state IO
    const bool                                        m_has_recurrent;
};
