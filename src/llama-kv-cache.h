#pragma once

#include "llama.h"
#include "llama-io.h"
#include "llama-graph.h"
#include "llama-memory.h"

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
    virtual ~llama_kv_cache() = default;

    // call if batch processing fails - restores the cache state
    virtual void restore() = 0;

    // call after successful batch processing - clears any pending state
    virtual void commit()  = 0;

    // process any pending defrag/shift/etc. operations
    // optionally call once before processing a new batch
    virtual bool update(llama_context & lctx) = 0;

    // schedule a defrag if the fragmentation threshold is exceeded. otherwise, do nothing
    virtual void defrag_sched(float thold) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual void set_full() = 0;

    //
    // batch processing
    //

    // =============================================================================================================
    // TODO: refactor  and simplify this

    virtual llama_sbatch sbatch_init(const llama_batch & batch, bool logits_all) = 0;

    // different KV caches require different batch splitting strategies
    virtual llama_ubatch ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const = 0;

    // find an empty slot of size "n_tokens" in the cache
    virtual bool find_slot(const llama_ubatch & batch) = 0;

    // =============================================================================================================

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
// llama_kv_cache_guard
//

struct llama_kv_cache_guard {
    llama_kv_cache_guard(llama_kv_cache * kv) : kv(kv) {}

    ~llama_kv_cache_guard() {
        kv->restore();
    }

    void commit() {
        kv->commit();
    }

private:
    llama_kv_cache * kv;
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

    llama_sbatch sbatch_init(const llama_batch & batch, bool logits_all) override;
    llama_ubatch ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const override;

    // updates the cache head
    // Note: On success, it's important that cache.head points
    // to the first cell of the slot.
    bool find_slot(const llama_ubatch & batch) override;

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // llama_kv_cache_unified specific API
    //

    uint32_t get_n() const;
    uint32_t get_size() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the current head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il) const;

    void prune_swa(llama_seq_id seq_id, llama_pos pmin, llama_pos pmax);

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_cell {
        llama_pos pos   = -1;
        llama_pos delta =  0;

        // TODO: replace with bitset uint64_t
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

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;
    };

    bool has_shift = false;
    bool do_defrag = false;
    bool v_trans   = true;  // the value tensor is transposed

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id) (TODO: add `struct kv_cells` and keep track automaticallt)

    // computed before each graph build
    uint32_t n = 0;

    const uint32_t n_seq_max = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<kv_cell>  cells;  // TODO: replace with `struct kv_cells`
    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    // recovery information used to restore the KV cells to their original state in case of a failure
    struct {
        void clear() {
            cells.clear();
        }

        std::unordered_map<uint32_t, kv_cell> cells;
    } recovery;

    // defrag
    struct {
        std::vector<uint32_t> ids;
    } defrag_info;

    // return true if cells have been moved
    bool defrag_prepare(int32_t n_max_nodes);

    // find how many cells are currently in use
    uint32_t cell_max() const;

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
//   upon successful commit, the SWA cache removes old tokens outside the n_swa window

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

    llama_sbatch sbatch_init(const llama_batch & batch, bool logits_all) override;
    llama_ubatch ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const override;

    bool find_slot(const llama_ubatch & batch) override;

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

    bool do_prune = true;

    struct {
        struct entry {
            llama_pos pmin;
            llama_pos pmax;
        };

        void clear() {
            pos.clear();
        }

        // used to perform SWA pruning of old tokens
        std::unordered_map<llama_seq_id, entry> pos;
    } pending;

    std::unique_ptr<llama_kv_cache_unified> kv_base;
    std::unique_ptr<llama_kv_cache_unified> kv_swa;
};

//
// llama_kv_cache_recurrent
//

class llama_kv_cache_recurrent : public llama_kv_cache {
public:
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

    llama_sbatch sbatch_init(const llama_batch & batch, bool logits_all) override;
    llama_ubatch ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const override;

    bool find_slot(const llama_ubatch & batch) override;

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

    std::vector<kv_cell> cells;

    std::vector<ggml_tensor *> k_l; // per layer
    std::vector<ggml_tensor *> v_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    // commit/restore cache
    // TODO: rework for recurrent cache
    struct slot_range {
        uint32_t c0 = 0; // note: these are cell indices, not sequence positions
        uint32_t c1 = 0;
    };

    // pending cell updates that are not yet committed
    struct {
        std::vector<slot_range> ranges;
    } pending;

    const uint32_t n_seq_max = 1;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // find how many cells are currently in use
    uint32_t cell_max() const;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};
