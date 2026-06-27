#pragma once

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#endif

#include "ggml-backend.h"
#include "llama.h"
#include <atomic>
#include <mutex>
#include <vector>

enum class llama_pool_state : uint8_t {
    LLM_RESIDENT    = 0,
    SWAPPING_OUT    = 1,
    MMPROJ_RESIDENT = 2,
    SWAPPING_IN     = 3,
    CORRUPTED       = 4,
    DISABLED        = 5,
};

struct llama_mmproj_pool {
    ggml_backend_buffer_t host_buf      = nullptr;
    void *                host_ptr      = nullptr;
    size_t                host_buf_size = 0;

    std::vector<ggml_tensor *> evicted_tensors;
    std::vector<size_t>        evicted_offsets;

    struct tensor_mapping {
        ggml_tensor          * vision_t;
        void                 * gpu_data;
        ggml_backend_buffer_t  gpu_buffer;
        void                 * host_data;
        size_t                 size;
    };
    std::vector<tensor_mapping> mappings;

    std::atomic<llama_pool_state> state { llama_pool_state::DISABLED };
    std::mutex                    mutex;

    int64_t n_swaps       = 0;
    double  total_swap_ms = 0.0;
};


struct llama_mmproj_pool * llama_mmproj_pool_init(
        struct llama_model         * model,
        int                          n_swap_layers,
        std::vector<ggml_tensor *> & mmproj_tensors,
        size_t                       dynamic_overhead_bytes);

bool llama_mmproj_pool_swap_in(struct llama_mmproj_pool * pool, struct llama_context * ctx);
void llama_mmproj_pool_swap_back(struct llama_mmproj_pool * pool, struct llama_context * ctx);
void llama_mmproj_pool_free(struct llama_mmproj_pool * pool);
void llama_mmproj_pool_log_stats(const struct llama_mmproj_pool * pool);
