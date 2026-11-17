#pragma once

// libllama internal: the per-sequence state of the gen pipelines of the clip engine, as the memory of a GEN context

#include "llama-memory.h"

#include <map>

struct clip_gen_ctx;
struct clip_gen_state;

// RNG, decoder state slots and host DSP state of each sequence; created on first use
// only whole sequences can be removed (a cold start), the state cannot be saved
class llama_memory_clip_gen : public llama_memory_i {
public:
    llama_memory_clip_gen() = default;
    ~llama_memory_clip_gen() override;

    // the state of seq_id, created with its seed on first use, nullptr on failure
    clip_gen_state * get(clip_gen_ctx * host, llama_seq_id seq_id);

    // the seed of seq_id (UINT32_MAX = random), takes effect from a cold start
    void set_seed(llama_seq_id seq_id, uint32_t seed);

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

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }

    [[noreturn]] void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    [[noreturn]] void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

private:
    std::map<llama_seq_id, clip_gen_state *> states;
    std::map<llama_seq_id, uint32_t>         seeds;
};
