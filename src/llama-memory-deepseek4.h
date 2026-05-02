#pragma once

#include "llama-batch.h"
#include "llama-memory.h"
#include "ggml-cpp.h"

#include <vector>

struct ggml_context;
struct ggml_tensor;

struct llama_model;
struct llama_context;

class llama_memory_deepseek4 : public llama_memory_i {
public:
    struct layer_state {
        ggml_tensor * attn_kv = nullptr;

        ggml_tensor * attn_comp_kv_state    = nullptr;
        ggml_tensor * attn_comp_score_state = nullptr;

        ggml_tensor * indexer_kv = nullptr;

        ggml_tensor * indexer_comp_kv_state    = nullptr;
        ggml_tensor * indexer_comp_score_state = nullptr;
    };

    llama_memory_deepseek4(
            const llama_model & model,
                    ggml_type   type_k,
                         bool   offload,
                     uint32_t   n_ctx_seq,
                     uint32_t   n_seq_max);

    ~llama_memory_deepseek4() override = default;

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
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    const layer_state & get_layer(int32_t il) const;
    uint32_t get_n_ctx_seq() const;

private:
    friend class llama_memory_deepseek4_context;

    const llama_model & model;

    const uint32_t n_ctx_seq;
    const uint32_t n_seq_max;

    std::vector<layer_state> layers;
    std::vector<llama_pos> seq_pos_min_v;
    std::vector<llama_pos> seq_pos_max_v;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
};

class llama_memory_deepseek4_context : public llama_memory_context_i {
public:
    llama_memory_deepseek4_context(llama_memory_status status);

    llama_memory_deepseek4_context(
            llama_memory_deepseek4 * mem,
            std::vector<llama_ubatch> ubatches);

    ~llama_memory_deepseek4_context() override = default;

    bool next() override;
    bool apply() override;

    const llama_ubatch & get_ubatch() const override;
    llama_memory_status get_status() const override;

    const llama_memory_deepseek4::layer_state & get_layer(int32_t il) const;
    uint32_t get_n_ctx_seq() const;

private:
    const llama_memory_status status;

    llama_memory_deepseek4 * mem = nullptr;

    size_t i_next = 0;
    std::vector<llama_ubatch> ubatches;
};
