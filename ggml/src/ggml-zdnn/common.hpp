#ifndef GGML_ZDNN_COMMON_HPP
#define GGML_ZDNN_COMMON_HPP

#include "ggml.h"
#include "ggml-impl.h"

#include "zdnn.h"

#include <algorithm>
#include <cstdlib>
#include <future>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#define GGML_ZDNN_NAME          "zDNN"
#define GGML_ZDNN_VERSION       ZDNN_VERNUM
#define GGML_ZDNN_SCRATCH_SLOTS 3  // int8 inputs, DFloat16 partial results, partial results for grouped matmul

#define ZDNN_CHECK(stmt)                \
    do {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } while (0);

struct ggml_backend_zdnn_device_context {
    int zdnn_device;
    int zdnn_device_ref_count;

    bool has_parmblkformat_0;
    bool has_parmblkformat_1;  // checks for z17

    size_t max_size;

    char name[128];
};

struct ggml_backend_zdnn_buffer {
    void * data;
    ggml_backend_zdnn_buffer * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<float> scales;          // per-row scales for quantized weights
    std::vector<float> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    char name[GGML_MAX_NAME];
};

struct ggml_backend_zdnn_buffer_context {
    void * all_data;
    size_t all_size;
    bool owned;

    int n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_buffer>> buffers;
};

struct ggml_backend_zdnn_context {
    int device;
    int32_t n_threads;  // for the CPU work around zDNN calls
    ggml_cgraph * gf;

    // zero bias for stacked quantized matmuls, keyed by (n_stacks, ne0)
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<ggml_backend_zdnn_buffer>> zero_bias;

    std::vector<int8_t> scratch_q;
    std::vector<float>  scratch_f32;

    void * scratch_ztensor[GGML_ZDNN_SCRATCH_SLOTS]      = {};  // 4K aligned
    size_t scratch_ztensor_size[GGML_ZDNN_SCRATCH_SLOTS] = {};
};

template <typename T>
static inline T * ggml_zdnn_resize_scratch_max(std::vector<T> & scratch, int64_t n) {
    if ((int64_t)scratch.size() < n) {
        scratch.resize(n);
    }

    return scratch.data();
}

//
// Scratch buffers
//

static inline void ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context * ctx,
                                                  zdnn_tensor_desc          * pre_tfm_desc,
                                                  zdnn_tensor_desc          * tfm_desc,
                                                  zdnn_ztensor              * ztensor,
                                                  int32_t                     slot) {

    const uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    if (ctx->scratch_ztensor_size[slot] < size) {
        free(ctx->scratch_ztensor[slot]);
        ctx->scratch_ztensor[slot]      = std::aligned_alloc(4096, GGML_PAD(size, 4096));
        ctx->scratch_ztensor_size[slot] = GGML_PAD(size, 4096);
        GGML_ASSERT(ctx->scratch_ztensor[slot] != nullptr);
    }

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->buffer      = ctx->scratch_ztensor[slot];
    ztensor->buffer_size = size;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
template <typename F>
static inline void ggml_zdnn_parallel_for(int32_t n_threads, int64_t n, const F & f) {
    n_threads = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_threads, n));

    std::vector<std::future<void>> tasks;
    for (int32_t ith = 1; ith < n_threads; ith++) {
        const int64_t start = (ith + 0) * n / n_threads;
        const int64_t end   = (ith + 1) * n / n_threads;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_threads);

    for (auto & task : tasks) {
        task.get();
    }
}

#endif  // GGML_ZDNN_COMMON_HPP
