#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter,
                     uint32_t   n_rs_rec = 0);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // ReplaySSM (ggml_gated_delta_net_rec): with n_rs_rec > 0 the states have no rollback snapshots. A ubatch of at
    // most n_rs_rec tokens records its updates instead of advancing the SSM state, and the next ubatch commits the
    // ones that were kept: all of them, or as many as a seq_rm left. The conv state is a window of d_conv - 1 columns
    // into time-major records of the conv input (rec_conv_cols columns of every channel per cell): a ubatch starts
    // from the window at the commit offset and writes its own conv input (a recording ubatch) or its last columns (a
    // larger one) at the start, so that the window of the kept tokens starts as many columns in. One sequence only.
    uint32_t n_rs_rec = 0;

    // the SSM records per cell (see ggml_gated_delta_net_rec): n_rs_rec rows of delta, then two halves of n_rs_rec
    // rows of k and of exp(g), in floats
    uint32_t rec_d() const { return hparams.ssm_d_inner; }
    uint32_t rec_k() const { return hparams.ssm_d_state*hparams.ssm_n_group; }
    uint32_t rec_g() const { return hparams.ssm_dt_rank; }
    size_t   rec_off_k(uint32_t half) const { return (size_t) n_rs_rec*(rec_d() + half*rec_k()); }
    size_t   rec_off_g(uint32_t half) const { return (size_t) n_rs_rec*(rec_d() + 2*rec_k() + half*rec_g()); }
    size_t   rec_size() const { return (size_t) n_rs_rec*(rec_d() + 2*rec_k() + 2*rec_g()); }

    // the conv records: channels, the window's columns and all columns per cell
    uint32_t rec_conv_channels() const { return hparams.n_embd_r()/(hparams.ssm_d_conv - 1); }
    uint32_t rec_conv_w()        const { return hparams.ssm_d_conv - 1; }
    uint32_t rec_conv_cols()     const { return rec_conv_w() + n_rs_rec; }

    struct rec_state {
        uint32_t  n    = 0;  // updates the last ubatch recorded
        llama_pos p0   = -1; // the position of the first
        uint32_t  m    = 0;  // updates the next ubatch commits
        uint32_t  half = 0;  // the half of the records they are in (the next recording ubatch writes the other)
    };
    rec_state rec;

    // set by apply for the ubatch being built: whether it records, and the updates it commits first and their half
    bool     rec_recording   = false;
    uint32_t rec_commit      = 0;
    uint32_t rec_commit_half = 0;

    bool rec_records(const llama_ubatch & ubatch) const {
        return n_rs_rec > 0 && ubatch.n_seq_tokens <= n_rs_rec;
    }

    void rec_apply(const llama_ubatch & ubatch);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;
    // ReplaySSM records of the SSM updates, rec_size() floats per cell (see ggml_gated_delta_net_rec)
    std::vector<ggml_tensor *> u_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // true if no layers - can happen if the layer filter removes all layers
    bool is_empty() const;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);

    void state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;
    ggml_tensor * get_u_l(int32_t il) const;

    int32_t s_copy(int i) const;

    // ReplaySSM: the state source without the rollback snapshot (the SSM state has none), whether the ubatch records,
    // and the updates it commits first (see llama_memory_recurrent::n_rs_rec)
    int32_t  s_copy_s(int i) const;
    uint32_t get_n_rs_rec() const;
    uint32_t get_rec_conv_w() const;
    bool     rec_records(const llama_ubatch & ubatch) const;
    uint32_t rec_commit() const;
    uint32_t rec_commit_half() const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
