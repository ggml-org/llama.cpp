#include "argsort.cuh"
#include "diffusion-sampling.cuh"
#include "../ggml-backend-impl.h"

#include <algorithm>
#include <cfloat>
#include <map>
#include <mutex>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_DIFFUSION_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif
#endif

static int next_power_of_2_host(int x) {
    int n = 1;
    while (n < x) {
        n <<= 1;
    }
    return n;
}

struct diffusion_sample_scratch {
    int   * top_ids  = nullptr;
    int   * sampled  = nullptr;
    int   * argmax   = nullptr;
    int   * prev_argmax = nullptr;
    int   * stop     = nullptr;
    float * entropy  = nullptr;
    int   * sc_ids   = nullptr;
    float * sc_probs = nullptr;

    size_t top_ids_cap  = 0;
    size_t sampled_cap  = 0;
    size_t argmax_cap   = 0;
    size_t prev_argmax_cap = 0;
    size_t stop_cap     = 0;
    size_t entropy_cap  = 0;
    size_t sc_ids_cap   = 0;
    size_t sc_probs_cap = 0;
};

static std::mutex g_diffusion_scratch_mutex;
static std::map<cudaStream_t, diffusion_sample_scratch> g_diffusion_scratch;

template<typename T>
static void diffusion_scratch_reserve(cudaStream_t stream, T ** ptr, size_t * cap, const size_t need, bool * synced) {
    if (*cap >= need) {
        return;
    }
    if (*ptr) {
        if (!*synced) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            *synced = true;
        }
        CUDA_CHECK(cudaFree(*ptr));
        *ptr = nullptr;
        *cap = 0;
    }
    CUDA_CHECK(cudaMalloc((void **) ptr, need * sizeof(T)));
    *cap = need;
}

static diffusion_sample_scratch * diffusion_get_scratch(
        cudaStream_t stream,
        const int n_tokens,
        const int heap_k,
        const int sc_k) {
    std::lock_guard<std::mutex> lock(g_diffusion_scratch_mutex);
    diffusion_sample_scratch & scratch = g_diffusion_scratch[stream];
    bool synced = false;

    diffusion_scratch_reserve(stream, &scratch.top_ids,  &scratch.top_ids_cap,  (size_t) n_tokens * heap_k, &synced);
    diffusion_scratch_reserve(stream, &scratch.sampled,  &scratch.sampled_cap,  (size_t) n_tokens,          &synced);
    diffusion_scratch_reserve(stream, &scratch.argmax,   &scratch.argmax_cap,   (size_t) n_tokens,          &synced);
    diffusion_scratch_reserve(stream, &scratch.prev_argmax, &scratch.prev_argmax_cap, (size_t) n_tokens,     &synced);
    diffusion_scratch_reserve(stream, &scratch.stop,     &scratch.stop_cap,     (size_t) 1,                 &synced);
    diffusion_scratch_reserve(stream, &scratch.entropy,  &scratch.entropy_cap,  (size_t) n_tokens,          &synced);
    diffusion_scratch_reserve(stream, &scratch.sc_ids,   &scratch.sc_ids_cap,   (size_t) n_tokens * sc_k,   &synced);
    diffusion_scratch_reserve(stream, &scratch.sc_probs, &scratch.sc_probs_cap, (size_t) n_tokens * sc_k,   &synced);

    return &scratch;
}

static __device__ __forceinline__ bool diffusion_should_swap_desc(
        const float a_val, const int a_id,
        const float b_val, const int b_id) {
    return a_val < b_val || (a_val == b_val && a_id > b_id);
}

template<int LOCAL_K>
static __global__ void diffusion_select_topk_local_kernel(
        const float * __restrict__ logits,
        int * __restrict__ top_ids,
        const int n_vocab,
        const int heap_k) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    float vals[LOCAL_K];
    int ids[LOCAL_K];
#pragma unroll
    for (int i = 0; i < LOCAL_K; ++i) {
        vals[i] = -FLT_MAX;
        ids[i] = 0;
    }

    const float * row_logits = logits + (size_t) row * n_vocab;
    for (int v = tid; v < n_vocab; v += blockDim.x) {
        const float x = row_logits[v];
        if (diffusion_should_swap_desc(vals[LOCAL_K - 1], ids[LOCAL_K - 1], x, v)) {
            int pos = LOCAL_K - 1;
#pragma unroll
            for (int i = LOCAL_K - 1; i > 0; --i) {
                if (pos == i && diffusion_should_swap_desc(vals[i - 1], ids[i - 1], x, v)) {
                    vals[i] = vals[i - 1];
                    ids[i] = ids[i - 1];
                    --pos;
                }
            }
            vals[pos] = x;
            ids[pos] = v;
        }
    }

    constexpr int candidate_count = LOCAL_K * 256;

    extern __shared__ unsigned char smem[];
    float * s_vals = (float *) smem;
    int * s_ids = (int *) (s_vals + candidate_count);

#pragma unroll
    for (int i = 0; i < LOCAL_K; ++i) {
        const int dst = tid * LOCAL_K + i;
        s_vals[dst] = vals[i];
        s_ids[dst] = ids[i];
    }
    __syncthreads();

    for (int k = 2; k <= candidate_count; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < candidate_count; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool descending = (i & k) == 0;
                    const bool swap = descending
                        ? diffusion_should_swap_desc(s_vals[i], s_ids[i], s_vals[ixj], s_ids[ixj])
                        : diffusion_should_swap_desc(s_vals[ixj], s_ids[ixj], s_vals[i], s_ids[i]);
                    if (swap) {
                        const float tv = s_vals[i];
                        s_vals[i] = s_vals[ixj];
                        s_vals[ixj] = tv;
                        const int ti = s_ids[i];
                        s_ids[i] = s_ids[ixj];
                        s_ids[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < heap_k; i += blockDim.x) {
        top_ids[(size_t) row * heap_k + i] = s_ids[i];
    }
}

static void diffusion_select_topk_local(
        const float * logits,
        int * top_ids,
        const int n_vocab,
        const int n_tokens,
        const int heap_k,
        cudaStream_t stream) {
    constexpr int block_size = 256;

    if (heap_k <= 256) {
        constexpr int local_k = 8;
        const size_t smem = (size_t) local_k * block_size * (sizeof(float) + sizeof(int));
        diffusion_select_topk_local_kernel<local_k><<<n_tokens, block_size, smem, stream>>>(
                logits, top_ids, n_vocab, heap_k);
    } else {
        constexpr int local_k = 16;
        const size_t smem = (size_t) local_k * block_size * (sizeof(float) + sizeof(int));
        diffusion_select_topk_local_kernel<local_k><<<n_tokens, block_size, smem, stream>>>(
                logits, top_ids, n_vocab, heap_k);
    }
}

#ifdef CUB_DIFFUSION_TOP_K_AVAILABLE
static void diffusion_top_k_cub(
        ggml_cuda_pool & pool,
        const float * src,
        int * dst,
        const int ncols,
        const int k,
        cudaStream_t stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };
    auto indexes_in   = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k, env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void * d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k, env));
}
#endif

static __global__ void diffusion_sort_top_ids_kernel(
        const float * __restrict__ logits,
        int * __restrict__ top_ids,
        const int n_vocab,
        const int heap_k,
        const int heap_k_pad,
        const float inv_temp) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ unsigned char smem[];
    int * ids = (int *) smem;
    float * vals = (float *) (ids + heap_k_pad);

    const int base = row * heap_k;
    for (int i = tid; i < heap_k_pad; i += blockDim.x) {
        if (i < heap_k) {
            const int id = top_ids[base + i];
            ids[i] = id;
            vals[i] = logits[(size_t) row * n_vocab + id] * inv_temp;
        } else {
            ids[i] = 0;
            vals[i] = -FLT_MAX;
        }
    }
    __syncthreads();

    for (int k = 2; k <= heap_k_pad; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (ixj > tid && ixj < heap_k_pad) {
                const bool up = (tid & k) == 0;
                const bool swap = up ? (vals[tid] < vals[ixj]) : (vals[tid] > vals[ixj]);
                if (swap) {
                    const float tv = vals[tid];
                    vals[tid] = vals[ixj];
                    vals[ixj] = tv;
                    const int ti = ids[tid];
                    ids[tid] = ids[ixj];
                    ids[ixj] = ti;
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < heap_k; i += blockDim.x) {
        top_ids[base + i] = ids[i];
    }
}

static __device__ __forceinline__ uint32_t diffusion_rng_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static __device__ __forceinline__ float diffusion_rng_uniform(uint32_t seed, uint32_t step, uint32_t row) {
    const uint32_t x = diffusion_rng_u32(seed ^ (step * 0x9e3779b9u) ^ (row * 0x85ebca6bu));
    return ((x >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

static __device__ __forceinline__ bool diffusion_pair_gt(float a_val, int a_id, float b_val, int b_id) {
    return a_val > b_val || (a_val == b_val && a_id > b_id);
}

static __global__ void diffusion_update_canvas_kernel(
        const float * __restrict__ entropy,
        const int * __restrict__ sampled,
        int * __restrict__ canvas_tokens,
        const int n_tokens,
        const int n_vocab,
        const float entropy_bound,
        const uint32_t seed,
        const uint32_t step) {
    const int tid = threadIdx.x;

    __shared__ float s_entropy[1024];
    __shared__ int   s_index[1024];
    __shared__ unsigned char s_accept[1024];

    if (tid < n_tokens) {
        s_entropy[tid] = entropy[tid];
        s_index[tid]   = tid;
        s_accept[tid]  = 0;
    } else {
        s_entropy[tid] = FLT_MAX;
        s_index[tid]   = tid;
    }
    __syncthreads();

    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (ixj > tid) {
                const bool ascending = (tid & k) == 0;
                const float a_val = s_entropy[tid];
                const int   a_idx = s_index[tid];
                const float b_val = s_entropy[ixj];
                const int   b_idx = s_index[ixj];
                const bool swap = ascending ?
                    diffusion_pair_gt(a_val, a_idx, b_val, b_idx) :
                    diffusion_pair_gt(b_val, b_idx, a_val, a_idx);
                if (swap) {
                    s_entropy[tid] = b_val;
                    s_index[tid]   = b_idx;
                    s_entropy[ixj] = a_val;
                    s_index[ixj]   = a_idx;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        float prefix = 0.0f;
        for (int i = 0; i < n_tokens; ++i) {
            const int pos = s_index[i];
            if (prefix <= entropy_bound) {
                s_accept[pos] = 1;
                prefix += s_entropy[i];
            } else {
                break;
            }
        }
    }
    __syncthreads();

    if (tid < n_tokens) {
        if (s_accept[tid]) {
            canvas_tokens[tid] = sampled[tid];
        } else {
            const uint32_t r = diffusion_rng_u32(seed ^ ((step + 1u) * 0x9e3779b9u) ^ ((uint32_t) tid * 0x7f4a7c15u) ^ 0xa5a5a5a5u);
            canvas_tokens[tid] = (int) (r % (uint32_t) n_vocab);
        }
    }
}

static __global__ void diffusion_stop_state_kernel(
        const float * __restrict__ entropy,
        const int * __restrict__ argmax,
        int * __restrict__ prev_argmax,
        int * __restrict__ stop,
        const int n_tokens,
        const float confidence_threshold,
        const int stability_threshold,
        const int check_stop,
        const int reset_state) {
    const int tid = threadIdx.x;

    __shared__ float s_entropy[1024];
    __shared__ int   s_diff[1024];

    if (tid < n_tokens) {
        s_entropy[tid] = entropy[tid];
        s_diff[tid] = reset_state ? 1 : (prev_argmax[tid] != argmax[tid]);
    } else {
        s_entropy[tid] = 0.0f;
        s_diff[tid] = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_entropy[tid] += s_entropy[tid + stride];
            s_diff[tid] += s_diff[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && check_stop) {
        const bool stable = !reset_state && (stability_threshold == 0 || s_diff[0] == 0);
        const bool confident = confidence_threshold > 0.0f &&
                (s_entropy[0] / (float) n_tokens) < confidence_threshold;
        stop[0] = (stable && confident) ? 1 : 0;
    }
    __syncthreads();

    if (tid < n_tokens) {
        prev_argmax[tid] = argmax[tid];
    }
}

static __global__ void diffusion_sample_kernel(
        const float * __restrict__ logits,
        const int * __restrict__ top_ids,
        const int n_vocab,
        const int n_tokens,
        const int top_k,
        const int heap_k,
        const int sc_k,
        const float inv_temp,
        const uint32_t seed,
        const uint32_t step,
        const bool tail_correction,
        int * __restrict__ sampled,
        int * __restrict__ argmax,
        float * __restrict__ entropy,
        int * __restrict__ sc_ids,
        float * __restrict__ sc_probs) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= n_tokens) {
        return;
    }

    __shared__ float s_val[1024];
    __shared__ float s_sum[1024];
    __shared__ int   s_idx[1024];

    const float * row_logits = logits + (size_t) row * n_vocab;
    const int * row_top = top_ids + (size_t) row * heap_k;

    float local_max = -FLT_MAX;
    int local_idx = 0;

    if (top_k == 0 || tail_correction) {
        for (int v = tid; v < n_vocab; v += blockDim.x) {
            const float x = row_logits[v] * inv_temp;
            if (x > local_max) {
                local_max = x;
                local_idx = v;
            }
        }
    } else {
        for (int i = tid; i < top_k; i += blockDim.x) {
            const int v = row_top[i];
            const float x = row_logits[v] * inv_temp;
            if (x > local_max) {
                local_max = x;
                local_idx = v;
            }
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }

    const float max_l = s_val[0];
    const int amax = s_idx[0];

    float local_sum = 0.0f;
    float local_t = 0.0f;

    if (top_k == 0 || tail_correction) {
        for (int v = tid; v < n_vocab; v += blockDim.x) {
            const float d = row_logits[v] * inv_temp - max_l;
            const float e = expf(d);
            local_sum += e;
            local_t   += d * e;
        }
    } else {
        for (int i = tid; i < top_k; i += blockDim.x) {
            const int v = row_top[i];
            const float d = row_logits[v] * inv_temp - max_l;
            const float e = expf(d);
            local_sum += e;
            local_t   += d * e;
        }
    }

    s_sum[tid] = local_sum;
    s_val[tid] = local_t;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_val[tid] += s_val[tid + stride];
        }
        __syncthreads();
    }

    const float z = s_sum[0];
    const float t = s_val[0];

    if (tid == 0) {
        argmax[row] = amax;
        entropy[row] = logf(z) - t / z;

        float sample_z = z;
        if (top_k > 0 && tail_correction) {
            sample_z = 0.0f;
            for (int i = 0; i < top_k; ++i) {
                const int v = row_top[i];
                sample_z += expf(row_logits[v] * inv_temp - max_l);
            }
        }

        const float r = diffusion_rng_uniform(seed, step, row) * sample_z;
        float cum = 0.0f;
        int tok = amax;

        if (top_k == 0) {
            for (int v = 0; v < n_vocab; ++v) {
                cum += expf(row_logits[v] * inv_temp - max_l);
                if (cum >= r) {
                    tok = v;
                    break;
                }
            }
        } else {
            for (int i = 0; i < top_k; ++i) {
                const int v = row_top[i];
                cum += expf(row_logits[v] * inv_temp - max_l);
                if (cum >= r) {
                    tok = v;
                    break;
                }
            }
        }
        sampled[row] = tok;

        const int n_sc = top_k == 0 ? sc_k : min(sc_k, top_k);
        for (int i = 0; i < sc_k; ++i) {
            const int out = row * sc_k + i;
            if (i < n_sc) {
                const int v = row_top[i];
                sc_ids[out] = v;
                sc_probs[out] = expf(row_logits[v] * inv_temp - max_l) / sample_z;
            } else {
                sc_ids[out] = 0;
                sc_probs[out] = 0.0f;
            }
        }
    }
}

bool ggml_cuda_diffusion_sample_topk(
        ggml_backend_t backend,
        const ggml_tensor * logits,
        const ggml_cuda_diffusion_sample_params * params,
        ggml_cuda_diffusion_sample_result * result) {
    if (!backend || !logits || !params || !result) {
        return false;
    }
    if (!ggml_backend_is_cuda(backend) || logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits)) {
        return false;
    }
    const int n_vocab = params->n_vocab > 0 ? params->n_vocab : (int) logits->ne[0];
    const int n_tokens = params->n_tokens > 0 ? params->n_tokens : (int) ggml_nrows(logits);
    if (n_vocab <= 0 || n_tokens <= 0 || logits->ne[0] != n_vocab || ggml_nrows(logits) < n_tokens) {
        return false;
    }

    int top_k = params->top_k;
    if (top_k <= 0 || top_k >= n_vocab) {
        top_k = 0;
    }

    const int sc_k = params->self_cond_top_k;
    if (sc_k <= 0 || sc_k > 1024) {
        return false;
    }

    if (result->update_canvas_on_device) {
        if (n_tokens > 1024 ||
            result->canvas_tokens_tensor == nullptr ||
            result->canvas_tokens_tensor->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(result->canvas_tokens_tensor) ||
            result->canvas_tokens_tensor->data == nullptr ||
            result->canvas_tokens_tensor->buffer == nullptr ||
            ggml_backend_buffer_is_host(result->canvas_tokens_tensor->buffer) ||
            ggml_nbytes(result->canvas_tokens_tensor) != (size_t) n_tokens * sizeof(int)) {
            return false;
        }
    }
    if (result->update_stop_state_on_device || result->check_stop_on_device || result->reset_stop_state) {
        if (n_tokens > 1024 || (result->check_stop_on_device && result->stop == nullptr)) {
            return false;
        }
    }

    const bool have_self_cond_host = result->self_cond_ids != nullptr && result->self_cond_probs != nullptr;
    const bool have_self_cond_tensor = result->self_cond_ids_tensor != nullptr && result->self_cond_probs_tensor != nullptr;
    if (!have_self_cond_host && !have_self_cond_tensor) {
        return false;
    }
    if ((result->self_cond_ids == nullptr) != (result->self_cond_probs == nullptr)) {
        return false;
    }
    if ((result->self_cond_ids_tensor == nullptr) != (result->self_cond_probs_tensor == nullptr)) {
        return false;
    }
    if (have_self_cond_tensor) {
        if (result->self_cond_ids_tensor->type != GGML_TYPE_I32 ||
            result->self_cond_probs_tensor->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(result->self_cond_ids_tensor) ||
            !ggml_is_contiguous(result->self_cond_probs_tensor) ||
            result->self_cond_ids_tensor->data == nullptr ||
            result->self_cond_probs_tensor->data == nullptr ||
            result->self_cond_ids_tensor->buffer == nullptr ||
            result->self_cond_probs_tensor->buffer == nullptr ||
            ggml_backend_buffer_is_host(result->self_cond_ids_tensor->buffer) ||
            ggml_backend_buffer_is_host(result->self_cond_probs_tensor->buffer)) {
            return false;
        }
        if (ggml_nbytes(result->self_cond_ids_tensor) != (size_t) n_tokens * sc_k * sizeof(int) ||
            ggml_nbytes(result->self_cond_probs_tensor) != (size_t) n_tokens * sc_k * sizeof(float)) {
            return false;
        }
    }

    const int heap_k = top_k == 0 ? sc_k : std::max(top_k, sc_k);
    if (heap_k <= 0 || heap_k > 1024 || heap_k > n_vocab) {
        return false;
    }

    const float temp = params->temperature > 0.0f ? params->temperature : 1.0f;
    const float inv_temp = 1.0f / temp;

    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);
    ggml_cuda_pool & pool = ctx->pool();
    cudaStream_t stream = ctx->stream();

    const float * logits_d = (const float *) logits->data;

    diffusion_sample_scratch * scratch = diffusion_get_scratch(stream, n_tokens, heap_k, sc_k);
    int * top_ids = scratch->top_ids;
    bool top_ids_sorted = false;

    const bool use_fast_topk = params->fast_top_k && heap_k <= 1024 && n_vocab >= heap_k;
    bool sync_required = !use_fast_topk;
    if (use_fast_topk) {
        diffusion_select_topk_local(logits_d, top_ids, n_vocab, n_tokens, heap_k, stream);
        top_ids_sorted = true;
    } else {
#ifdef CUB_DIFFUSION_TOP_K_AVAILABLE
        for (int row = 0; row < n_tokens; ++row) {
            diffusion_top_k_cub(pool, logits_d + (size_t) row * n_vocab, top_ids + (size_t) row * heap_k, n_vocab, heap_k, stream);
        }
#elif defined(GGML_CUDA_USE_CUB)
        ggml_cuda_pool_alloc<int> sorted_ids_alloc(pool, (size_t) n_tokens * n_vocab);
        int * sorted_ids = sorted_ids_alloc.get();
        argsort_f32_i32_cuda_cub(pool, logits_d, sorted_ids, n_vocab, n_tokens, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(top_ids, heap_k * sizeof(int), sorted_ids, n_vocab * sizeof(int),
                                     heap_k * sizeof(int), n_tokens, cudaMemcpyDeviceToDevice, stream));
        top_ids_sorted = true;
#else
        if (n_vocab > 1024) {
            return false;
        }
        ggml_cuda_pool_alloc<int> sorted_ids_alloc(pool, (size_t) n_tokens * n_vocab);
        int * sorted_ids = sorted_ids_alloc.get();
        argsort_f32_i32_cuda_bitonic(logits_d, sorted_ids, n_vocab, n_tokens, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(top_ids, heap_k * sizeof(int), sorted_ids, n_vocab * sizeof(int),
                                     heap_k * sizeof(int), n_tokens, cudaMemcpyDeviceToDevice, stream));
        top_ids_sorted = true;
#endif
    }

    if (!top_ids_sorted) {
        const int heap_k_pad = next_power_of_2_host(heap_k);
        const int sort_threads = std::max(32, heap_k_pad);
        const size_t sort_smem = (size_t) heap_k_pad * (sizeof(int) + sizeof(float));
        diffusion_sort_top_ids_kernel<<<n_tokens, sort_threads, sort_smem, stream>>>(
                logits_d, top_ids, n_vocab, heap_k, heap_k_pad, inv_temp);
    }

    constexpr int block_size = 256;
    diffusion_sample_kernel<<<n_tokens, block_size, 0, stream>>>(
            logits_d, top_ids, n_vocab, n_tokens, top_k, heap_k, sc_k, inv_temp,
            params->seed, params->step, params->top_k_tail_correction,
            scratch->sampled, scratch->argmax, scratch->entropy,
            scratch->sc_ids, scratch->sc_probs);

    if (result->update_canvas_on_device) {
        const int update_threads = next_power_of_2_host(n_tokens);
        diffusion_update_canvas_kernel<<<1, update_threads, 0, stream>>>(
                scratch->entropy, scratch->sampled, (int *) result->canvas_tokens_tensor->data,
                n_tokens, n_vocab, result->entropy_bound, params->seed, params->step);
    }
    if (result->update_stop_state_on_device || result->check_stop_on_device || result->reset_stop_state) {
        const int stop_threads = next_power_of_2_host(n_tokens);
        diffusion_stop_state_kernel<<<1, stop_threads, 0, stream>>>(
                scratch->entropy, scratch->argmax, scratch->prev_argmax, scratch->stop,
                n_tokens, result->confidence_threshold, result->stability_threshold,
                result->check_stop_on_device ? 1 : 0, result->reset_stop_state ? 1 : 0);
    }

    if (result->sampled) {
        CUDA_CHECK(cudaMemcpyAsync(result->sampled, scratch->sampled, (size_t) n_tokens * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (result->argmax) {
        CUDA_CHECK(cudaMemcpyAsync(result->argmax, scratch->argmax, (size_t) n_tokens * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (result->entropy) {
        CUDA_CHECK(cudaMemcpyAsync(result->entropy, scratch->entropy, (size_t) n_tokens * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (result->final_tokens) {
        CUDA_CHECK(cudaMemcpyAsync(result->final_tokens, scratch->argmax, (size_t) n_tokens * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (result->stop) {
        CUDA_CHECK(cudaMemcpyAsync(result->stop, scratch->stop, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (have_self_cond_host) {
        CUDA_CHECK(cudaMemcpyAsync(result->self_cond_ids, scratch->sc_ids, (size_t) n_tokens * sc_k * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(result->self_cond_probs, scratch->sc_probs, (size_t) n_tokens * sc_k * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (have_self_cond_tensor) {
        CUDA_CHECK(cudaMemcpyAsync(result->self_cond_ids_tensor->data, scratch->sc_ids, (size_t) n_tokens * sc_k * sizeof(int),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(result->self_cond_probs_tensor->data, scratch->sc_probs, (size_t) n_tokens * sc_k * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    const bool host_outputs_requested = result->sampled || result->argmax || result->entropy ||
        result->final_tokens || result->stop || have_self_cond_host;
    CUDA_CHECK(cudaGetLastError());
    if (sync_required || host_outputs_requested) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return true;
}
