#pragma once

#include "llama.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-adapter.h"

#include "ggml-cpp.h"

#include <map>
#include <unordered_map>
#include <vector>
#include <set>

class llama_io_read_i;
class llama_io_write_i;

using llama_loras = std::unordered_map<struct llama_adapter_lora *, float>;

struct llama_context : public llama_graph_i {
    llama_context(
            const llama_model & model,
            const llama_context_params & params);

    virtual ~llama_context();

    // init scheduler and compute buffers
    // call once after the context is constructed
    virtual void init();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    virtual uint32_t n_ctx()         const;
    virtual uint32_t n_ctx_per_seq() const;
    virtual uint32_t n_batch()       const;
    virtual uint32_t n_ubatch()      const;
    virtual uint32_t n_seq_max()     const = 0;

    virtual uint32_t n_threads()       const;
    virtual uint32_t n_threads_batch() const;

    virtual int32_t max_nodes() const;

    virtual       llama_kv_cache * get_kv_self()       = 0;
    virtual const llama_kv_cache * get_kv_self() const = 0;

    virtual void kv_self_update() = 0;

    virtual enum llama_pooling_type pooling_type() const;

    virtual float * get_logits();
    virtual float * get_logits_ith(int32_t i);

    virtual float * get_embeddings();
    virtual float * get_embeddings_ith(int32_t i);
    virtual float * get_embeddings_seq(llama_seq_id seq_id);

    virtual int64_t n_pos_per_token() const; // vision

    virtual void attach_threadpool(
            ggml_threadpool_t   threadpool,
            ggml_threadpool_t   threadpool_batch);

    virtual void detach_threadpool();

    virtual void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    virtual void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    virtual void set_embeddings (bool value);
    virtual void set_causal_attn(bool value);

    virtual void set_adapter_lora(
            struct llama_adapter_lora * adapter,
            float scale);

    virtual bool rm_adapter_lora(
            struct llama_adapter_lora * adapter);

    virtual void clear_adapter_lora();

    virtual bool apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    ////

    virtual void synchronize();

    // zero-out inputs and create ggml_context
    virtual ggml_cgraph * graph_init();

    // TODO: add encode/decode graphs
    virtual llama_graph_result graph_build(
            ggml_context * ctx,
             ggml_cgraph * gf,
      const llama_ubatch & ubatch,
                    bool   worst_case);

    // returns the result of ggml_backend_sched_graph_compute_async execution
    virtual enum ggml_status graph_compute(
            ggml_cgraph * gf,
                   bool   batched);

    virtual void input_set(const llama_ubatch & ubatch);

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    virtual int32_t output_reserve(int32_t n_outputs);

    // make the outputs have the same order they had in the user-provided batch
    // TODO: maybe remove this
    virtual void output_reorder();

    // encode a batch of tokens by evaluating the encoder part of the transformer
    //
    //   - lctx:      llama context
    //   - batch:     batch to evaluate
    //
    // return 0 on success
    // return positive int on warning
    // return negative int on error
    //
    virtual int encode(llama_batch & inp_batch) = 0;

    // decode a batch of tokens by evaluating the transformer
    // in case of unsuccessful decoding (error or warning),
    // the kv_cache state will be returned to its original state
    // (for non-recurrent models) or cleaned (for recurrent models)
    //
    //   - lctx:      llama context
    //   - inp_batch: batch to evaluate
    //
    // return 0 on success
    // return positive int on warning
    // return negative int on error
    //
    virtual int decode(llama_batch & inp_batch) = 0;

    //
    // graph build API (generic)
    //

    virtual void build_cb(
             ggml_tensor * cur,
              const char * name,
      const llama_ubatch & ubatch,
                     int   il);

    // apply control vector for layer il
    virtual ggml_tensor * build_cvec(
            ggml_context * ctx0,
             ggml_tensor * cur,
                     int   il);

    // do mat_mul, while optionally apply lora
    virtual ggml_tensor * build_lora_mm(
            ggml_context * ctx0,
             ggml_tensor * w,
             ggml_tensor * cur);

    // do mat_mul_id, while optionally apply lora
    virtual ggml_tensor * build_lora_mm_id(
            ggml_context * ctx0,
             ggml_tensor * w,   // struct ggml_tensor * as
             ggml_tensor * cur, // struct ggml_tensor * b
             ggml_tensor * ids);

    virtual ggml_tensor * build_rope_factors(int il);

    virtual ggml_tensor * build_rope_shift(
            ggml_context * ctx0,
             ggml_tensor * cur,
             ggml_tensor * shift,
             ggml_tensor * factors,
             ggml_backend_buffer * bbuf);

    virtual ggml_tensor * build_inp_embd(
            ggml_context * ctx0,
             ggml_tensor * tok_embd,
      const llama_ubatch & ubatch);

    virtual ggml_tensor * build_inp_pos(
            ggml_context * ctx0,
                 int32_t   n_tokens);

    virtual ggml_tensor * build_inp_out_ids(
            ggml_context * ctx0,
                 int32_t   n_tokens,
                    bool   worst_case);

    virtual ggml_tensor * build_inp_mean(
            ggml_context * ctx0,
                 int32_t   n_tokens);

    virtual ggml_tensor * build_inp_cls(
            ggml_context * ctx0,
                 int32_t   n_tokens);

    // state save/load

    virtual size_t state_get_size();
    virtual size_t state_get_data(      uint8_t * dst, size_t size);
    virtual size_t state_set_data(const uint8_t * src, size_t size);

    virtual size_t state_seq_get_size(llama_seq_id seq_id);
    virtual size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size);
    virtual size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size);

    virtual bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    virtual bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    virtual size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    virtual size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    // perf

    virtual llama_perf_context_data perf_get_data() const;
    virtual void perf_reset();

protected:

    // state save/load

    virtual size_t state_get_data(llama_io_write_i & io);
    virtual size_t state_set_data(llama_io_read_i  & io);

    virtual size_t state_seq_get_data(llama_io_write_i & io, llama_seq_id seq_id);
    virtual size_t state_seq_set_data(llama_io_read_i  & io, llama_seq_id seq_id);

    // input tensors

    struct ggml_tensor * inp_tokens;  // I32 [n_batch]
    struct ggml_tensor * inp_embd;    // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos;     // I32 [n_batch]
    struct ggml_tensor * inp_out_ids; // I32 [n_outputs]
    struct ggml_tensor * inp_mean;    // F32 [n_batch, n_batch]
    struct ggml_tensor * inp_cls;     // I32 [n_batch]

    // members

    const llama_model & model;

    llama_cparams      cparams;
    llama_adapter_cvec cvec;
    llama_loras        loras;
    llama_sbatch       sbatch;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    ggml_backend_sched_ptr sched;

    ggml_context_ptr ctx_compute;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // TODO: remove
    bool logits_all = false;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    int32_t output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    bool need_reserve       = false;
    bool has_evaluated_once = false;

    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls
};

// transformer with a self-attention KV cache
class llama_context_kv_self : public llama_context {
public:
    llama_context_kv_self(
            const llama_model & model,
            const llama_context_params & params);

    virtual ~llama_context_kv_self();

    virtual uint32_t n_seq_max() const override;

    virtual       llama_kv_cache * get_kv_self()       override;
    virtual const llama_kv_cache * get_kv_self() const override;

    virtual void kv_self_update() override;

    virtual ggml_cgraph * graph_init() override;

    virtual void input_set(const llama_ubatch & ubatch) override;

    virtual int encode(llama_batch & inp_batch) override;
    virtual int decode(llama_batch & inp_batch) override;

    // max token position across all sequences in the current context
    llama_pos pos_max() const;

    // certain implementations could require a padding for the context size
    uint32_t get_ctx_padding(const llama_cparams & cparams) const;

    // === KV cache ===

    llama_kv_cache kv_self;

    ggml_tensor * inp_KQ_mask;         // F32 [kv_size, n_batch]
    ggml_tensor * inp_KQ_mask_cnv;     //     [kv_size, n_batch]
    ggml_tensor * inp_KQ_mask_swa;     // F32 [kv_size, n_batch]
    ggml_tensor * inp_KQ_mask_swa_cnv; //     [kv_size, n_batch]
    ggml_tensor * inp_k_shift;         // I32 [kv_size]

    virtual ggml_tensor * build_inp_k_shift(ggml_context * ctx0) override;

    virtual void build_attn_inp(
            ggml_context * ctx0,
                 int32_t   n_tokens,
                    bool   causal,
                    bool   swa,
                    bool   worst_case) override;

    virtual void build_attn_kv_store(
            ggml_context * ctx0,
             ggml_cgraph * graph,
             ggml_tensor * k_cur,
             ggml_tensor * v_cur,
                 int32_t   n_tokens,
                 int64_t   il,
                 bool      worst_case) override;

    virtual ggml_tensor * build_attn_qkv(
            ggml_context * ctx0,
             ggml_cgraph * graph,
             ggml_tensor * wo,
             ggml_tensor * wo_b,
             ggml_tensor * q_cur,
                 int32_t   n_tokens,
                 float     kq_scale,
                 int       il,
                 bool      worst_case) override;

    virtual ggml_tensor * build_attn_soft_max(
            ggml_context * ctx0,
             ggml_tensor * kq,
                 float     kq_scale) override;

    // === encoder-decoder ===

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    struct ggml_tensor * inp_embd_enc;      // F32 [n_embd, n_outputs_enc]
    struct ggml_tensor * inp_pos_bucket;    // I32 [n_batch|n_kv, n_batch]
    struct ggml_tensor * inp_KQ_mask_cross; // F32 [n_outputs_enc, n_batch]

    virtual ggml_tensor * build_inp_embd_enc(
            ggml_context * ctx0,
                 int32_t   n_tokens,
                    bool   worst_case) override;

    virtual ggml_tensor * build_inp_KQ_mask_cross(
            ggml_context * ctx0,
                 int32_t   n_tokens,
                    bool   worst_case) override;

    // === recurrent ===

    struct ggml_tensor * inp_s_copy;        // I32 [kv_size]
    struct ggml_tensor * inp_s_mask;        // F32 [1, n_kv]

    // TODO: add recurrent cache
    // TODO: add mamba-specific llama_context

    // TODO: change these to build_mamba_inp and hide `state_copy` and `state_mask` inside the llama_context impl
    virtual ggml_tensor * build_inp_s_copy(
            ggml_context * ctx0,
                    bool   worst_case) override;

    virtual ggml_tensor * build_inp_s_mask(
            ggml_context * ctx0,
                    bool   worst_case) override;

    virtual ggml_tensor * build_copy_mask_state(
            ggml_context * ctx0,
             ggml_cgraph * graph,
             ggml_tensor * s,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
                 int32_t   n_tokens,
                 int32_t   n_state,
                 int32_t   n_seqs,
                    bool   worst_case) override;

    virtual ggml_tensor * build_mamba_layer(
            ggml_context * ctx0,
             ggml_cgraph * graph,
             ggml_tensor * cur,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il,
                    bool   worst_case) override;

    virtual ggml_tensor * build_rwkv_token_shift_load(
            ggml_context * ctx0,
             ggml_cgraph * graph,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il,
                    bool   worst_case) override;

    virtual ggml_tensor * build_rwkv_token_shift_store(
            ggml_context * ctx0,
             ggml_tensor * token_shift,
      const llama_ubatch & ubatch,
                     int   il,
                    bool   worst_case) override;

    virtual ggml_tensor * build_rwkv6_time_mix(
            ggml_context * ctx0,
             ggml_cgraph * graph,
             ggml_tensor * cur,
             ggml_tensor * x_prev,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il,
                    bool   worst_case) override;

protected:
    virtual size_t state_get_data(llama_io_write_i & io) override;
    virtual size_t state_set_data(llama_io_read_i  & io) override;

    virtual size_t state_seq_get_data(llama_io_write_i & io, llama_seq_id seq_id) override;
    virtual size_t state_seq_set_data(llama_io_read_i  & io, llama_seq_id seq_id) override;
};

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(struct llama_context * ctx);
