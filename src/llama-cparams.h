#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    // Number of initial decoder layers supplied by a qualified external
    // prefill slab.  The remaining graph consumes ubatch.embd as the
    // post-slab hidden state and starts at this layer boundary.  This is
    // intentionally internal context state: only the ANE prefill handoff
    // establishes it after validating the artifact and cache contract.
    uint32_t ane_prefill_layer_count = 0;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    bool imatrix_observers;

    // Per-scope observer state. The verifier and drafter each keep their own
    // filter, user_data, and epoch so that two llama_context instances in the
    // same process (one verifier, one drafter) can collect independent
    // importance statistics. imatrix_observer_scope selects which slot
    // llama_set_imatrix_observer_filter / llama_bump_imatrix_observer_epoch
    // operate on (default VERIFIER); DFlash drafter graphs read the DRAFTER
    // slot regardless of the user setting.
    enum llama_observer_scope imatrix_observer_scope = LLAMA_OBSERVER_SCOPE_VERIFIER;
    llama_imatrix_observer_filter imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_DRAFTER + 1] = { nullptr, nullptr };
    void * imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER + 1] = { nullptr, nullptr };
    uint64_t imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER + 1] = { 0, 0 };

    llama_context * ctx_other;
};
