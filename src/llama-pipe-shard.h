#pragma once

#include "ggml-cpp.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
struct pshard_dev_layout;
typedef struct ggml_backend_sched * ggml_backend_sched_t;

// pipe-shard memory interface for KV cache or recurrent state.
struct llama_memory_pipe_shard_i {
    // t1/t2 are k/v for KV cache, r/s for recurrent state
    struct layer {
        uint32_t      il;
        ggml_tensor * t1_gpu;
        ggml_tensor * t2_gpu;
        ggml_tensor * t1_cpu;
        ggml_tensor * t2_cpu;
        void        * t1_gpu_addr   = nullptr;
        void        * t2_gpu_addr   = nullptr;
        size_t        alloc_size    = 0;
        bool          is_pinned     = false;
        bool          prefetched_t1 = false;
        bool          prefetched_t2 = false;
    };

    virtual ~llama_memory_pipe_shard_i() = default;

    virtual const std::vector<layer> & get_layers() const = 0;

    // per-batch write_cells (KV only). indexed by stream.
    const std::vector<std::vector<uint32_t>> * write_cells = nullptr;
    void set_write_cells(const std::vector<std::vector<uint32_t>> * wc) { write_cells = wc; }

    virtual void clear_prefetch() = 0;

    // sched split callbacks
    virtual bool prefetch_if_owned(ggml_tensor * t, ggml_backend_t be) = 0;
    virtual bool upload_if_owned(ggml_tensor * t, ggml_backend_t be) = 0;
    virtual bool download_if_owned(ggml_tensor * t, ggml_backend_t be) = 0;

    // plan switch
    virtual void upload_for_switch(int32_t il, ggml_backend_t be) = 0;
    virtual void download_for_switch(int32_t il, ggml_backend_t be) = 0;

    virtual void activate_gpu(int32_t il) = 0;
    virtual void activate_cpu(int32_t il) = 0;

    // ensure all layers are host-accessible (for state save/load, KV shift, defrag)
    virtual void prepare_for_host_access() = 0;

    virtual void pin_layer(int32_t il) = 0;
    virtual void unpin_layer(int32_t il) = 0;

    virtual void set_external_addrs(int32_t il, void * a1, void * a2, size_t sz) = 0;

    virtual void refresh_stream_views(int32_t il) = 0;

    virtual void assign_tensors(
            ggml_backend_sched_t sched,
            const std::unordered_map<int, int32_t> & layer_bids,
            const std::vector<ggml_backend_ptr> & backends,
            const pshard_dev_layout & layout) = 0;

    virtual size_t current_pinned_size() const = 0;
};
