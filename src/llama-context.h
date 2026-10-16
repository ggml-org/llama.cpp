#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "moe_pager.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);
    // ---- sparse MoE direct-IO slot pool ----
    struct moe_pool {
        ggml_context        * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        size_t               buf_bytes = 0;
        std::vector<ggml_tensor *> pool_gate, pool_up, pool_gate_up, pool_down, slot_table;
        std::vector<int32_t> slot_of, slot_key;
        std::vector<uint32_t> slot_freq;
        std::vector<uint32_t> dec_freq;   // n_layer*n_expert: decode-only routing counts
        moe_lru lru;
        std::vector<uint16_t> pred16;     // n_layer*16: predictor candidates for the next step
        long pred_hit = 0, pred_fresh = 0;
        int n_layer = 0, n_expert = 0, n_used = 0, s_per_layer = 0;
        int n_slots = 0;   // s_per_layer plus the sentinel slot when armed
        long n_hit = 0, n_miss = 0, n_page_fail = 0;
        bool count_freq = true;
        std::atomic<long> ra_done{0};
        std::vector<uint16_t> exp_rank;  // n_layer*n_expert: LFU rank within layer
        long prefill_resident = 0, prefill_miss = 0;
        // zero-copy pinned host pool: flat array of H uniform slots holding
        // whole experts (gate|up|down compact). served with plain H2D on the
        // compute stream, no staging copy and no extra stream.
        char * host_pool = nullptr;
        size_t host_slot_bytes = 0;
        int    host_nslots = 0;
        int    host_rr = 0;
        int    host_ring = 0;
        std::vector<void *> host_ev;
        // per-layer compact role sizes within a host slot
        std::vector<size_t> host_gate, host_up, host_down;
        // previous decode step's top-k per layer, for the recency predictor
        // (moe_step_sel[il*n_used+k]); empty until the first decode step
        std::vector<int32_t> step_sel;
        // ffn_moe_topk-<il> node of the graph in flight, kept so the
        // between-launches refresh can read the routing it just produced
        std::vector<ggml_tensor *> step_topk;
        // last ubatch width seen; used to detect the prefill->decode edge
        uint32_t last_ntok = 0;
        // Direction 1: one extra all-zero slot per layer absorbing cold misses
        bool     sentinel_slot = false;
        // adaptive swap tier (MOE_ADAPT>=1): decayed routing demand per
        // (layer, expert) plus the pass that swaps cold VRAM victims out for
        // the hottest non-resident candidates.
        std::vector<float>   adapt_decay;   // n_layer * n_expert
        std::vector<int32_t> adapt_last;    // last pass the expert was demanded
        std::vector<int32_t> adapt_pend_e;  // n_layer * n_expert: pending swap candidate
        long adapt_pass  = 0;               // decode steps seen
        long adapt_swaps = 0, adapt_bytes = 0, adapt_cold = 0;
    };
    moe_pool moe;
    moe_pager pager;
    // page-cache read-ahead pool: background threads only issue readahead(),
    // never touch CUDA or slot tables
    std::vector<std::thread> or_thr;
    std::atomic<bool> or_run{false};
    std::mutex or_mtx;
    std::vector<uint32_t> or_q;
    std::vector<uint16_t> coact;      // (n_layer-1)*n_expert*4: learned L+1 candidates
    std::vector<uint16_t> coact_cnt;
    std::vector<uint32_t> co_hint;    // n_layer*n_expert: token stamp of last hint
    uint32_t co_tok = 0;
    long co_push = 0, co_need = 0, co_cover = 0;
    int  hot_k = 0;
    int  hot_pos = 0;
    bool hot_cycles = false;
    // persistent pool for the staging-ring preads
    struct moe_io_job { int32_t e; int vslot; int rslot; };
    std::mutex io_mtx;
    std::condition_variable io_cv, io_done_cv;
    std::vector<moe_io_job> io_jobs, io_jobs_scratch;
    int  io_pfd = -1, io_il = 0;
    size_t io_stride = 0, io_per = 0, io_nth = 0;
    std::atomic<size_t> io_next{0};
    std::atomic<long> io_fail{0};
    std::atomic<long> io_finished{0};
    long io_thr_active = 0;
    unsigned long long io_gen = 0;
    std::atomic<unsigned long long> io_gen_done{0};
    std::atomic<bool> io_run{false};
    std::vector<std::thread> io_thr;
    bool      moe_ready = false;
    bool moe_enabled() const { return moe_ready; }
    int  moe_page(int il, int e);
    int  moe_page_pinned(int il, int e, const bool * pinned);
    void moe_page_batch(int il, const int32_t * sel, int n);
    void moe_page_batch_sorted(int il, const int32_t * sel, int n);
    // pipelined host-pool paging for one resolve batch: returns false only
    // when the host pool is unavailable (caller falls back to pager fetch)
    bool moe_page_host_batch(int il, const int32_t * uniq, size_t nunq,
        std::vector<char> & pinned, int base, int s_pl, void * stream);
    // queue async reads for experts the next layer(s) will likely need.
    // never blocks; admission-controlled inside.
    void moe_prefetch_ahead(int il, const int32_t * sel, int n);
    void moe_publish(int il, int e, int slot);
    // adaptive swap tier (MOE_ADAPT>=1)
    bool moe_adapt_on() const;
    void moe_adapt_note(int il, int e);
    void moe_adapt_run();
    // page every layer's experts the step just selected and republish the tables
    // between graph launches (MOE_POST_REFRESH=1): the decode graph then runs
    // with no split at all, which is where the per-token latency was going.
    void moe_post_refresh();
    // neural lookahead prefetch via 31M native CUDA router predictor
    void moe_predict_and_prefetch();
    const ggml_tensor * moe_pool_tensor(const char * which, int il) const;
    // pre-populate every layer's slots with the offline hot profile so the
    // first decode step starts warm; pages synchronously, failures are
    // tolerated (slots stay -1). capped at 2s to bound server startup.
    void moe_seed_hot();
    void moe_hot_save();
    void moe_hot_cycle(int fd);
    // re-seed once when decoding starts after a long prefill: micro-chunk
    // paging churns the hot set, so restore the profile before token 0.
    // bounded by the same budget; no-op for pure decode streams.
    void moe_reseed_decode();

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;


    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
