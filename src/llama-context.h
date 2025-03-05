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

// abstract interface corresponding to the public C API
class llama_context_i {
public:
    llama_context_i() = default;
    virtual ~llama_context_i() = default;

    virtual void init() = 0;

    virtual void synchronize() = 0;

    virtual const llama_model & get_model() const = 0;

    virtual uint32_t n_ctx()         const = 0;
    virtual uint32_t n_ctx_per_seq() const = 0;
    virtual uint32_t n_batch()       const = 0;
    virtual uint32_t n_ubatch()      const = 0;
    virtual uint32_t n_seq_max()     const = 0;

    virtual uint32_t n_threads()       const = 0;
    virtual uint32_t n_threads_batch() const = 0;

    // self-attention:

    // if the context does not have a KV cache, return nullptr
    virtual       llama_kv_cache * get_kv_self()       = 0;
    virtual const llama_kv_cache * get_kv_self() const = 0;

    // if the context does not have a KV cache, noop
    virtual void kv_self_update() = 0;

    virtual enum llama_pooling_type pooling_type() const = 0;

    virtual float * get_logits()              = 0;
    virtual float * get_logits_ith(int32_t i) = 0;

    virtual float * get_embeddings()                        = 0;
    virtual float * get_embeddings_ith(int32_t i)           = 0;
    virtual float * get_embeddings_seq(llama_seq_id seq_id) = 0;

    virtual void attach_threadpool(
            ggml_threadpool_t   threadpool,
            ggml_threadpool_t   threadpool_batch) = 0;

    virtual void detach_threadpool() = 0;

    virtual void set_n_threads(int32_t n_threads, int32_t n_threads_batch) = 0;

    virtual void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) = 0;

    virtual void set_embeddings (bool value) = 0;
    virtual void set_causal_attn(bool value) = 0;

    virtual void set_adapter_lora(
            llama_adapter_lora * adapter,
            float scale) = 0;

    virtual bool rm_adapter_lora(
            llama_adapter_lora * adapter) = 0;

    virtual void clear_adapter_lora() = 0;

    virtual bool apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) = 0;

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
    // perf
    //

    virtual llama_perf_context_data perf_get_data() const = 0;
    virtual void perf_reset() = 0;

    //
    // state save/load
    //

    virtual size_t state_get_size()                                 = 0;
    virtual size_t state_get_data(      uint8_t * dst, size_t size) = 0;
    virtual size_t state_set_data(const uint8_t * src, size_t size) = 0;

    virtual size_t state_seq_get_size(llama_seq_id seq_id)                                   = 0;
    virtual size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size) = 0;
    virtual size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size) = 0;

    virtual bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out) = 0;

    virtual bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count) = 0;

    virtual size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out) = 0;

    virtual size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count) = 0;
};

// C alias
struct llama_context : public llama_context_i {
    using llama_context_i::llama_context_i;
};

// basic transformer without KV cache
class llama_context_base : public llama_context, public llama_graph_i {
public:
    llama_context_base(
            const llama_model & model,
                  llama_context_params params,
                  llama_graph_type gtype);

    virtual ~llama_context_base();

    // init scheduler and compute buffers, reserve worst-case graphs
    // call once after the context is constructed
    void init() override;

    void synchronize() override;

protected:
    // called by init() to reserve the worst-case graphs
    // override in child classes
    virtual void reserve();

public:
    const llama_model & get_model() const override;

    uint32_t n_ctx()         const override;
    uint32_t n_ctx_per_seq() const override;
    uint32_t n_batch()       const override;
    uint32_t n_ubatch()      const override;
    uint32_t n_seq_max()     const override;

    uint32_t n_threads()       const override;
    uint32_t n_threads_batch() const override;

          llama_kv_cache * get_kv_self()       override;
    const llama_kv_cache * get_kv_self() const override;

    void kv_self_update() override;

    enum llama_pooling_type pooling_type() const override;

    float * get_logits()              override;
    float * get_logits_ith(int32_t i) override;

    float * get_embeddings()                        override;
    float * get_embeddings_ith(int32_t i)           override;
    float * get_embeddings_seq(llama_seq_id seq_id) override;

    void attach_threadpool(
            ggml_threadpool_t   threadpool,
            ggml_threadpool_t   threadpool_batch) override;

    void detach_threadpool() override;

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch) override;

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) override;

    void set_embeddings (bool value) override;
    void set_causal_attn(bool value) override;

    void set_adapter_lora(
            llama_adapter_lora * adapter,
            float scale) override;

    bool rm_adapter_lora(
            llama_adapter_lora * adapter) override;

    void clear_adapter_lora() override;

    bool apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) override;

    int encode(llama_batch & inp_batch) override;
    int decode(llama_batch & inp_batch) override;

protected:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    virtual int32_t output_reserve(int32_t n_outputs);

    // make the outputs have the same order they had in the user-provided batch
    // TODO: maybe remove this
    virtual void output_reorder();

    //
    // graph
    //

    virtual int32_t graph_max_nodes() const;

    // zero-out inputs and create the ctx_compute for the compute graph
    virtual ggml_cgraph * graph_init();

    // TODO: add encode/decode graphs
    virtual llama_graph_result_ptr graph_build(
            ggml_context * ctx,
             ggml_cgraph * gf,
      const llama_ubatch & ubatch);

    // returns the result of ggml_backend_sched_graph_compute_async execution
    virtual enum ggml_status graph_compute(
            ggml_cgraph * gf,
                   bool   batched);

    ggml_context_ptr ctx_compute;

public:
    //
    // graph build
    //

    int32_t get_n_outputs() const override;

    void build_cb(
             ggml_tensor * cur,
              const char * name,
      const llama_ubatch & ubatch,
                     int   il) const override;

    // apply control vector for layer il
    ggml_tensor * build_cvec(
            ggml_context * ctx0,
             ggml_tensor * cur,
                     int   il) const override;

    // do mat_mul, while optionally apply lora
    ggml_tensor * build_lora_mm(
            ggml_context * ctx0,
             ggml_tensor * w,
             ggml_tensor * cur) const override;

    // do mat_mul_id, while optionally apply lora
    ggml_tensor * build_lora_mm_id(
            ggml_context * ctx0,
             ggml_tensor * w,   // struct ggml_tensor * as
             ggml_tensor * cur, // struct ggml_tensor * b
             ggml_tensor * ids) const override;

    ggml_tensor * build_rope_factors(int il) const override;

    llama_graph_input_ptr build_inp_embd(
                  ggml_context * ctx0,
                   ggml_tensor * tok_embd,
            const llama_ubatch & ubatch) const override;

    llama_graph_input_ptr build_inp_pos_bucket(
            ggml_context * ctx0,
                 int32_t   n_tokens) const override;

    llama_graph_input_attn_ptr build_attn_inp(
            ggml_context * ctx0,
                 int32_t   n_tokens,
                    bool   causal,
                    bool   swa) const override;

    ggml_tensor * build_attn(
            llama_graph_input_attn_i * inp,
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * q_cur,
             ggml_tensor * k_cur,
             ggml_tensor * v_cur,
             ggml_tensor * kq_b,
                   float   kq_scale,
                     int   il) const override;

protected:
    // note: optionally set the backend to be the same as the bbuf's backend
    ggml_tensor * build_rope_shift(
            ggml_context * ctx0,
             ggml_tensor * cur,
             ggml_tensor * shift,
             ggml_tensor * factors,
             ggml_backend_buffer * bbuf) const;

    ggml_tensor * build_attn_mha(
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * q,
             ggml_tensor * k,
             ggml_tensor * v,
             ggml_tensor * kq_b,
             ggml_tensor * kq_mask,
                 bool      v_trans,
                 float     kq_scale) const;

public:
    //
    // perf
    //

    llama_perf_context_data perf_get_data() const override;
    void perf_reset()                             override;

protected:
    // TODO: become private
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

public:
    //
    // state save/load
    //

    size_t state_get_size()                                 override;
    size_t state_get_data(      uint8_t * dst, size_t size) override;
    size_t state_set_data(const uint8_t * src, size_t size) override;

    size_t state_seq_get_size(llama_seq_id seq_id)                                   override;
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size) override;
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size) override;

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out) override;

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count) override;

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out) override;

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count) override;

protected:
    virtual size_t state_write_data(llama_io_write_i & io);
    virtual size_t state_read_data (llama_io_read_i  & io);

    virtual size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id);
    virtual size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id);

    //
    // members
    //
    // TODO: become private / move to llama_graph_i

    const llama_model & model;

    llama_cparams      cparams;
    llama_adapter_cvec cvec;
    llama_loras        loras;
    llama_sbatch       sbatch;

    ggml_backend_sched_ptr sched;

    // TODO: these below likely need some rework in the future, together with the batch-refactoring

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

    int32_t n_outputs     = 0; // number of actually-used outputs in the current ubatch or last logical batch
    int32_t n_outputs_max = 0; // capacity (of tokens positions) for the output buffers

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

private:
    // base functionality - should not leak into derived classes

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;
};

// transformer with a self-attention KV cache
class llama_context_kv_self : public llama_context_base {
public:
    llama_context_kv_self(
            const llama_model & model,
                  llama_context_params params,
                  llama_graph_type gtype);

    virtual ~llama_context_kv_self();

protected:
    void reserve() override;

public:
          llama_kv_cache * get_kv_self()       override;
    const llama_kv_cache * get_kv_self() const override;

    void kv_self_update() override;

    int encode(llama_batch & inp_batch) override;
    int decode(llama_batch & inp_batch) override;

protected:
    //
    // graph
    //

    ggml_cgraph * graph_init() override;

public:
    //
    // graph build
    //

    llama_graph_input_ptr build_inp_pos_bucket(
            ggml_context * ctx0,
                 int32_t   n_tokens) const override;

    llama_graph_input_attn_ptr build_attn_inp(
            ggml_context * ctx0,
                 int32_t   n_tokens,
                    bool   causal,
                    bool   swa) const override;

    ggml_tensor * build_attn(
            llama_graph_input_attn_i * inp,
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * q_cur,
             ggml_tensor * k_cur,
             ggml_tensor * v_cur,
             ggml_tensor * kq_b,
                   float   kq_scale,
                     int   il) const override;

protected:
    llama_graph_result_ptr graph_build_kv_self_shift(
            ggml_context * ctx0,
            ggml_cgraph * gf) const;

    // find holes from the beginning of the KV cache and fill them by moving data from the end of the cache
    llama_graph_result_ptr graph_build_kv_self_defrag(
            ggml_context * ctx0,
            ggml_cgraph * gf) const;

    //
    // state save/load
    //

    size_t state_write_data(llama_io_write_i & io) override;
    size_t state_read_data (llama_io_read_i  & io) override;

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id) override;
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id) override;

private:
    //
    // members
    //

    std::unique_ptr<llama_kv_cache_unified> kv_self;
};

// a recurrent transformer (ie.e RWKV, Mamba)
class llama_context_recurrent : public llama_context_base {
public:
    llama_context_recurrent(
            const llama_model & model,
                  llama_context_params params,
                  llama_graph_type gtype);

    virtual ~llama_context_recurrent();

protected:
    void reserve() override;

public:
          llama_kv_cache * get_kv_self()       override;
    const llama_kv_cache * get_kv_self() const override;

    void kv_self_update() override;

    int encode(llama_batch & inp_batch) override;
    int decode(llama_batch & inp_batch) override;

protected:
    //
    // graph
    //

    ggml_cgraph * graph_init() override;

public:
    //
    // graph build
    //

    llama_graph_input_ptr build_inp_s_copy(
            ggml_context * ctx0) const override;

    llama_graph_input_ptr build_inp_s_mask(
            ggml_context * ctx0) const override;

    ggml_tensor * build_copy_mask_state(
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * s,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
                 int32_t   n_state,
                 int32_t   n_seqs) const override;

    ggml_tensor * build_mamba_layer(
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * cur,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il) const override;

    ggml_tensor * build_rwkv_token_shift_load(
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il) const override;

    ggml_tensor * build_rwkv_token_shift_store(
            ggml_context * ctx0,
             ggml_tensor * token_shift,
      const llama_ubatch & ubatch,
                     int   il) const override;

    ggml_tensor * build_rwkv6_time_mix(
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * cur,
             ggml_tensor * x_prev,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il) const override;

protected:
    //
    // state save/load
    //

    size_t state_write_data(llama_io_write_i & io) override;
    size_t state_read_data (llama_io_read_i  & io) override;

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id) override;
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id) override;

private:
    //
    // members
    //

    // TODO: change name to something more meaningful -- does "KV cache" make sense for recurrent models?
    std::unique_ptr<llama_kv_cache_recurrent> kv_self;
};

//
// enc-dec
//

// TODO: tmp - need something better to pass the data from the encoder to the decoder
struct llama_cross {
    // the output embeddings from the encoder as a ggml tensor
    // TODO: this needs more work to be correct, for now copy the embeddings data to host memory
    //       ref: https://github.com/ggml-org/llama.cpp/pull/11213#discussion_r1969892524
    ggml_tensor * t_embd = nullptr;

    // embeddings data copied to host memory (tmp)
    float * v_embd = nullptr;

    // needed to construct the cross-attention mask in the decoder
    std::vector<std::set<llama_seq_id>> seq_ids_enc;
};

class llama_context_enc : public llama_context_base {
public:
    using llama_context_base::llama_context_base;

    int encode(llama_batch & inp_batch) override;

    llama_cross * cross = nullptr; // TODO: hacky, rework
};

class llama_context_dec : public llama_context_kv_self {
public:
    using llama_context_kv_self::llama_context_kv_self;

protected:
    void reserve() override;

    //
    // graph
    //

    ggml_cgraph * graph_init() override;

    llama_graph_input_ptr build_inp_cross_embd(
            ggml_context * ctx0) const override;

    llama_graph_input_attn_ptr build_attn_inp(
            ggml_context * ctx0,
                 int32_t   n_tokens,
                    bool   causal,
                    bool   swa) const override;

    ggml_tensor * build_attn_cross(
            llama_graph_input_attn_i * inp,
            ggml_context * ctx0,
             ggml_cgraph * gf,
             ggml_tensor * q_cur,
             ggml_tensor * k_cur,
             ggml_tensor * v_cur,
             ggml_tensor * kq_b,
                 float     kq_scale,
                 int       il) const override;

public:
    llama_cross * cross = nullptr; // TODO: hacky, rework
};

class llama_context_enc_dec : public llama_context {
public:
    llama_context_enc_dec(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context_enc_dec();

    void init() override;

    void synchronize() override;

    const llama_model & get_model() const override;

    // TODO: the default implementation of these getters calls the corresponding getter of the enc or dec context
    //       in the future, the public API in llama.h should allow to get references to the context that the user wants
    //       this will allow to specify the desired context explicitly
    //       for example:
    //
    //          // this can be an enc-dec context
    //          llama_context_t ctx = llama_init_from_model(...);
    //
    //          ...
    //
    //          llama_context_t ctx_enc = llama_get_ctx_enc(ctx);
    //          llama_set_embeddings(ctx_enc, true);
    //
    //          llama_context_t ctx_dec = llama_get_ctx_dec(ctx);
    //          llama_set_causal_attn(ctx_dec, true);
    //
    uint32_t n_ctx()         const override;
    uint32_t n_ctx_per_seq() const override;
    uint32_t n_batch()       const override;
    uint32_t n_ubatch()      const override;
    uint32_t n_seq_max()     const override;

    uint32_t n_threads()       const override;
    uint32_t n_threads_batch() const override;

          llama_kv_cache * get_kv_self()       override;
    const llama_kv_cache * get_kv_self() const override;

    void kv_self_update() override;

    enum llama_pooling_type pooling_type() const override;

    float * get_logits()              override;
    float * get_logits_ith(int32_t i) override;

    float * get_embeddings()                        override;
    float * get_embeddings_ith(int32_t i)           override;
    float * get_embeddings_seq(llama_seq_id seq_id) override;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch) override;

    void detach_threadpool() override;

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch) override;

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) override;

    void set_embeddings (bool value) override;
    void set_causal_attn(bool value) override;

    void set_adapter_lora(
            llama_adapter_lora * adapter,
            float scale) override;

    bool rm_adapter_lora(
            llama_adapter_lora * adapter) override;

    void clear_adapter_lora() override;

    bool apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) override;

    int encode(llama_batch & inp_batch) override;
    int decode(llama_batch & inp_batch) override;

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const override;
    void perf_reset() override;

    //
    // state save/load
    //

    size_t state_get_size()                                 override;
    size_t state_get_data(      uint8_t * dst, size_t size) override;
    size_t state_set_data(const uint8_t * src, size_t size) override;

    size_t state_seq_get_size(llama_seq_id seq_id)                                   override;
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size) override;
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size) override;

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out) override;

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count) override;

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out) override;

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count) override;

private:
    std::unique_ptr<llama_context_enc> ctx_enc;
    std::unique_ptr<llama_context_dec> ctx_dec;

    llama_cross cross;
};

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(struct llama_context * ctx);
