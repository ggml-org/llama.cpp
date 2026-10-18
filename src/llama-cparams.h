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
    size_t kv_stream_pool_bytes = 0;
    size_t shared_device_memory_bytes = 0;
    uint32_t kv_stream_auxiliary_layers = 0;
    uint32_t kv_stream_verify_max = 0;
    bool kv_streaming() const noexcept {
        return kv_stream_pool_bytes || shared_device_memory_bytes;
    }
    // Embedded MTP retains its four-row graph catalog; independent target rollback admits depth+1 rows.
    uint32_t kv_stream_decode_rows() const noexcept {
        if (kv_stream_verify_max) return kv_stream_verify_max;
        return kv_stream_auxiliary_layers ? 4 : n_rs_seq ? n_rs_seq+1 : n_seq_max;
    }
    bool kv_stream_decode = false;
    bool kv_stream_prefill = false;
    bool kv_stream_mask = false;
    bool mtp_publish_host = false;
    bool mtp_span_attention = false;
    bool kv_shared_read = false; // read-only assistant over target-owned full and bounded caches
    uint32_t serial_block_rows = 0; // separately owned block drafter, not an auxiliary MTP cache
    // Rebuild the same declared draft workload that startup measured for the serial lease.
    uint32_t serial_graph_reserve_rows(bool decode) const noexcept {
        const uint32_t maximum = n_ctx < n_ubatch ? n_ctx : n_ubatch;
        return serial_block_rows ? serial_block_rows : decode && maximum > 4 ? 4 : maximum;
    }
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
    bool training;           // set by llama_opt_init()

    size_t moe_cache_size;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
