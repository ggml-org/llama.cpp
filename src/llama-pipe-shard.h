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
    };

    virtual ~llama_memory_pipe_shard_i() = default;

    virtual const std::vector<layer> & get_layers() const = 0;

    virtual void activate_gpu(int32_t il) = 0;
    virtual void activate_cpu(int32_t il) = 0;

    virtual void assign_tensors(
            ggml_backend_sched_t sched,
            const std::unordered_map<int, int32_t> & layer_bids,
            const std::vector<ggml_backend_ptr> & backends,
            const pshard_dev_layout & layout) = 0;
};
