#pragma once

#include "llama-memory.h"

#include "ggml-cpp.h"

#include <map>
#include <string>
#include <vector>

struct llama_model;

// per-sequence state tensors of any shape, declared by the model (conv left-contexts, transposed-conv tails,
// k/v windows of a streaming decoder, position counters); graphs read them and write them back in place.
// The slots of a sequence are allocated, zeroed, by its first use and freed when the sequence is removed
class llama_memory_slots : public llama_memory_i {
public:
    struct slot_desc {
        std::string name;
        int64_t     ne0;
        int64_t     ne1;
    };

    llama_memory_slots(const llama_model & model, const std::vector<slot_desc> & slots, uint32_t n_seq_max, bool offload);

    // llama_memory_i
    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    [[noreturn]] void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    [[noreturn]] void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // the slot of a sequence, F32 [ne0, ne1]; allocates the slots of the sequence on first use
    ggml_tensor * view(ggml_context * ctx, const std::string & name, llama_seq_id seq_id) const;

    // the number of positions a sequence has gone through, for seq_pos_max
    void advance(llama_seq_id seq_id, llama_pos n);

private:
    // the slots of one sequence
    struct seq_slots {
        ggml_context_ptr        ctx;
        ggml_backend_buffer_ptr buf;
        std::map<std::string, ggml_tensor *> tensors; // F32 [ne0, ne1]
    };

    seq_slots & alloc_seq(llama_seq_id seq_id) const;
    void        free_seq (llama_seq_id seq_id);

    const uint32_t n_seq_max;

    std::vector<slot_desc> descs;
    std::vector<llama_pos> n_pos; // per sequence, -1 = empty

    ggml_backend_buffer_type_t buft = nullptr;

    mutable std::vector<seq_slots> seqs; // allocated by the first use of the sequence
};

class llama_memory_slots_context : public llama_memory_context_i {
public:
    llama_memory_slots_context(llama_memory_slots * mem, llama_memory_status status) : mem(mem), status(status) {}
    llama_memory_slots_context(llama_memory_slots * mem, std::vector<llama_ubatch> ubatches);

    bool next() override;
    bool apply() override;

    const llama_ubatch & get_ubatch() const override;
    llama_memory_status  get_status() const override { return status; }

    llama_memory_slots * get_mem() const { return mem; }

private:
    llama_memory_slots * mem;
    llama_memory_status  status;

    size_t i_next = 0;
    std::vector<llama_ubatch> ubatches;
};
