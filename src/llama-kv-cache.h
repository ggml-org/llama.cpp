#pragma once

#include "llama.h"
#include "llama-io.h"
#include "llama-memory.h"

#include "ggml-cpp.h"

#include <functional>
#include <set>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_ubatch;

struct llama_kv_cache : public llama_memory_i {
    // can be used to query data from the model if needed
    struct callbacks {
        std::function<ggml_tensor * (uint32_t n_ctx_per_seq, int il)> get_rope_factors;

        // get the buffer type of layer il, can be used to offload KV cache layers to a different device
        std::function<ggml_backend_buffer_type_t (int il)> get_buft;
    };

    virtual ~llama_kv_cache() = default;

    using llama_memory_i::llama_memory_i;

    virtual void restore() = 0; // call if batch processing fails - restores the cache state
    virtual void commit() = 0;  // call after successful batch processing - clears any pending state

    virtual int32_t get_n_tokens()   const = 0;
    virtual int32_t get_used_cells() const = 0; // TODO: remove, this is too-specific to the unified cache

    virtual bool get_has_shift() const = 0;
    virtual bool get_do_defrag() const = 0;

    virtual llama_pos get_pos_max() const = 0;

    virtual bool get_can_shift() const = 0;

    bool get_can_edit() const override { return get_can_shift(); }

    virtual bool find_slot(const llama_ubatch & batch) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual void set_full() = 0;

    virtual size_t size_k_bytes() const = 0;
    virtual size_t size_v_bytes() const = 0;

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1) = 0;
};

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

struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta =  0;
    int32_t   src   = -1; // used by recurrent state models to copy states
    int32_t   tail  = -1;

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};

// ring-buffer of cached KV data
// TODO: pimpl
// TODO: add notion of max sequences
class llama_kv_cache_unified : public llama_kv_cache {
public:
    llama_kv_cache_unified(
            const llama_hparams & hparams,
                      callbacks   cbs,
                      ggml_type   type_k,
                      ggml_type   type_v,
                           bool   v_trans,
                       uint32_t   kv_size);

    ~llama_kv_cache_unified() = default;

    int32_t get_n_tokens()   const override;
    int32_t get_used_cells() const override;

    bool get_has_shift() const override;
    bool get_do_defrag() const override;

    size_t total_size() const;

    // TODO: better data structures to reduce the cost of this operation
    llama_pos get_pos_max() const override;

    void clear() override;
    void defrag() override;

    void restore() override;
    void commit() override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos delta) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    bool get_can_shift() const override;

    // find an empty slot of size "n_tokens" in the cache
    // updates the cache head
    // Note: On success, it's important that cache.head points
    // to the first cell of the slot.
    bool find_slot(const llama_ubatch & batch) override;

    static uint32_t get_padding(const llama_cparams & cparams);

    // find how many cells are currently in use
    uint32_t cell_max() const;

    void set_full() override;

    size_t size_k_bytes() const override;
    size_t size_v_bytes() const override;

    // defrag

    struct {
        std::vector<uint32_t> ids;
    } defrag_info;

    // return true if cells have been moved
    bool defrag_prepare(int32_t n_max_nodes);

    // commit/restore cache

    struct slot_range {
        uint32_t c0 = 0; // note: these are cell indices, not sequence positions
        uint32_t c1 = 0;
    };

    // pending cell updates that are not yet committed
    struct {
        std::vector<slot_range> ranges;
    } pending;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1) override;

    // members

    const llama_hparams & hparams;

    callbacks cbs;

    bool has_shift = false;
    bool do_defrag = false;

    bool v_trans   = true;  // the value tensor is transposed
    bool can_shift = false;

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_impl also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    std::vector<llama_kv_cell> cells;

    std::vector<ggml_tensor *> k_l; // per layer
    std::vector<ggml_tensor *> v_l;

private:
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_kv_cache_recurrent : public llama_kv_cache {
public:
    llama_kv_cache_recurrent(
            const llama_hparams & hparams,
                      callbacks   cbs,
                      ggml_type   type_k,
                      ggml_type   type_v,
                       uint32_t   kv_size);

    ~llama_kv_cache_recurrent() = default;

    int32_t get_n_tokens()   const override;
    int32_t get_used_cells() const override;

    bool get_has_shift() const override;
    bool get_do_defrag() const override;

    size_t total_size() const;

    // TODO: better data structures to reduce the cost of this operation
    llama_pos get_pos_max() const override;

    void clear() override;
    void defrag() override;

    void restore() override;
    void commit() override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos delta) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    bool get_can_shift() const override;

    // find an empty slot of size "n_tokens" in the cache
    // updates the cache head
    // Note: On success, it's important that cache.head points
    // to the first cell of the slot.
    bool find_slot(const llama_ubatch & batch) override;

    // find how many cells are currently in use
    uint32_t cell_max() const;

    void set_full() override;

    size_t size_k_bytes() const override;
    size_t size_v_bytes() const override;

    // commit/restore cache

    struct slot_range {
        uint32_t c0 = 0; // note: these are cell indices, not sequence positions
        uint32_t c1 = 0;
    };

    // pending cell updates that are not yet committed
    struct {
        std::vector<slot_range> ranges;
    } pending;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1) override;

    // members

    const llama_hparams & hparams;

    callbacks cbs;

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_impl also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    std::vector<llama_kv_cell> cells;

    std::vector<ggml_tensor *> k_l; // per layer
    std::vector<ggml_tensor *> v_l;

private:
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};


//
// kv cache view
//

llama_kv_cache_view llama_kv_cache_view_init(const llama_kv_cache & kv, int32_t n_seq_max);

void llama_kv_cache_view_update(llama_kv_cache_view * view, const llama_kv_cache * kv);
