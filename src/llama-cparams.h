#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

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
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    bool    pshard = false;
    int32_t cpu_backend_id = -1;
};

inline constexpr int32_t PSHARD_BACKENDS_PER_DEV = 3;

struct pshard_dev_layout {
    int32_t compute;
    int32_t shard_a;
    int32_t shard_b;
    int32_t cpu;

    int32_t shard(uint32_t il) const { return shard_a + (il % 2); }

    static pshard_dev_layout for_device(size_t dev_idx, int32_t cpu_backend_id) {
        const int32_t base = (int32_t)(dev_idx * PSHARD_BACKENDS_PER_DEV);
        return { base, base + 1, base + 2, cpu_backend_id };
    }

    static int32_t compute_cpu_backend_id(size_t n_devices) {
        int32_t n_accel = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_ACCEL) n_accel++;
        }
        return (int32_t)(n_devices * PSHARD_BACKENDS_PER_DEV) + n_accel;
    }
};
