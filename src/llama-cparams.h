#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

enum llm_fused_op : int; // llama-graph.h

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
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

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    // [LLM_FUSED_OP_COUNT][n_layer()] per-layer enablement, resolved by resolve_fused_ops().
    // An empty inner vector means every layer may use the fused form. Lets a device that
    // cannot run a fused op disable it only for its own layers instead of for the whole graph.
    // Flash attention and the lightning indexer are deliberately never per-layer: both imply
    // graph-wide properties (KV padding, mask dtype) that cannot vary between layers.
    std::vector<std::vector<bool>> fused_op_layers;

    bool use_fused(enum llm_fused_op op, int il) const {
        const size_t i = (size_t) op;
        if (i >= fused_op_layers.size()) {
            return true;
        }
        const std::vector<bool> & layers = fused_op_layers[i];
        if (layers.empty() || il < 0 || (size_t) il >= layers.size()) {
            return true;
        }
        return layers[il];
    }

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
