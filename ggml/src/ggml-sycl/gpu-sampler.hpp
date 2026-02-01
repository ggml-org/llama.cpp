// GPU-side sampler kernels for multi-step decoding
// Implements: temp_scale -> softmax -> sample
//
// This enables multi-step decode without CPU sync by keeping
// the entire sampling pipeline on GPU.
//
// Design for large vocabularies (32K-128K tokens):
// - Multi-block reductions using device memory for intermediate results
// - Work-efficient parallel algorithms
// - Minimal synchronization points

#pragma once

#include "common.hpp"
#include <random>

template <typename T>
static inline T * ggml_sycl_gpu_sampler_alloc(size_t count, sycl::queue & q) {
    return ggml_sycl_malloc_device_t<T>(count, q, "gpu_sampler");
}

// Constants
constexpr int GPU_SAMPLER_BLOCK_SIZE = 256;
constexpr int GPU_SAMPLER_MAX_BLOCKS = 512;  // Max blocks for reduction
constexpr int GPU_SAMPLER_TOKEN_BUFFER_SIZE = 8;  // Multi-step token ring buffer size

// GPU sampler configuration
struct ggml_sycl_sampler_config {
    float temp;           // Temperature (1.0 = no scaling)
    int32_t top_k;        // Top-k filtering (0 = disabled, not yet implemented)
    float top_p;          // Top-p/nucleus filtering (1.0 = disabled, not yet implemented)
    float min_p;          // Min-p filtering (0.0 = disabled, not yet implemented)
    uint32_t seed;        // RNG seed
    bool greedy;          // If true, use argmax instead of sampling
};

// GPU sampler state (persistent across calls)
struct ggml_sycl_sampler_state {
    uint32_t rng_state;   // Current RNG state

    // Work buffers (allocated lazily)
    float* block_max;     // Per-block max values [MAX_BLOCKS]
    float* block_sum;     // Per-block sum values [MAX_BLOCKS]
    int32_t* block_idx;   // Per-block argmax indices [MAX_BLOCKS]
    float* probs;         // Softmax probabilities [n_vocab] (optional, for top-p)
    int32_t* selected;    // Selected token ID [1]
    float* random_val;    // Random value on device [1]

    // Multi-step token ring buffer (device memory)
    int32_t* token_buffer;      // Ring buffer for sampled tokens [TOKEN_BUFFER_SIZE]
    int token_write_idx;        // Current write position in ring buffer
    int token_count;            // Number of tokens written since last reset

    size_t n_vocab;       // Vocabulary size
    bool initialized;
};

// Xorshift32 GPU RNG (deterministic)
// Returns uniform random float in [0, 1)
inline float gpu_rng_uniform(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state) / static_cast<float>(UINT32_MAX);
}

// Device-side RNG for use in kernels
inline float device_rng_uniform(uint32_t* state) {
    uint32_t s = *state;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *state = s;
    return static_cast<float>(s) / static_cast<float>(UINT32_MAX);
}

// =============================================================================
// Phase 1: Temperature scaling (simple parallel, O(n) work)
// =============================================================================

template<typename T>
void gpu_temp_scale_kernel(
    T* logits,
    const int n_vocab,
    const float inv_temp,  // Pre-computed 1.0/temp
    const sycl::nd_item<1>& item
) {
    const int idx = item.get_global_id(0);
    if (idx < n_vocab) {
        logits[idx] *= inv_temp;
    }
}

// =============================================================================
// Phase 2a: Multi-block argmax reduction (Pass 1)
// Each block finds local max and writes to block_max/block_idx
// =============================================================================

void gpu_argmax_block_kernel(
    const float* logits,
    float* block_max,       // [n_blocks] output
    int32_t* block_idx,     // [n_blocks] output
    const int n_vocab,
    const sycl::nd_item<1>& item,
    float* slm_max,         // SLM [block_size]
    int32_t* slm_idx        // SLM [block_size]
) {
    const int tid = item.get_local_id(0);
    const int block_id = item.get_group(0);
    const int block_size = item.get_local_range(0);
    const int gid = item.get_global_id(0);

    // Each thread loads one element
    float local_max = (gid < n_vocab) ? logits[gid] : -INFINITY;
    int32_t local_idx = gid;

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Tree reduction in SLM
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    // Thread 0 writes block result
    if (tid == 0) {
        block_max[block_id] = slm_max[0];
        block_idx[block_id] = slm_idx[0];
    }
}

// =============================================================================
// Phase 2b: Final argmax reduction (Pass 2 - single block)
// Reduces block_max/block_idx to find global maximum
// =============================================================================

void gpu_argmax_final_kernel(
    const float* block_max,   // [n_blocks] input
    const int32_t* block_idx, // [n_blocks] input
    int32_t* result,          // [1] output
    const int n_blocks,
    const sycl::nd_item<1>& item,
    float* slm_max,           // SLM [block_size]
    int32_t* slm_idx          // SLM [block_size]
) {
    const int tid = item.get_local_id(0);
    const int block_size = item.get_local_range(0);

    // Load block results (may need multiple loads per thread)
    float local_max = -INFINITY;
    int32_t local_idx = 0;

    for (int i = tid; i < n_blocks; i += block_size) {
        if (block_max[i] > local_max) {
            local_max = block_max[i];
            local_idx = block_idx[i];
        }
    }

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Tree reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    if (tid == 0) {
        *result = slm_idx[0];
    }
}

// =============================================================================
// Phase 3a: Multi-block softmax reduction (Pass 1)
// Each block computes local max and sum(exp(x - local_max))
// =============================================================================

void gpu_softmax_block_kernel(
    const float* logits,
    float* block_max,       // [n_blocks] output: local max
    float* block_sum,       // [n_blocks] output: sum of exp(logits - local_max)
    const int n_vocab,
    const sycl::nd_item<1>& item,
    float* slm              // SLM [block_size]
) {
    const int tid = item.get_local_id(0);
    const int block_id = item.get_group(0);
    const int block_size = item.get_local_range(0);
    const int gid = item.get_global_id(0);

    // Load and find local max
    float local_val = (gid < n_vocab) ? logits[gid] : -INFINITY;
    slm[tid] = local_val;
    sycl::group_barrier(item.get_group());

    // Tree reduction for max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] = sycl::fmax(slm[tid], slm[tid + stride]);
        }
        sycl::group_barrier(item.get_group());
    }

    float local_max = slm[0];
    sycl::group_barrier(item.get_group());

    // Compute exp(x - local_max) and sum
    // Handle -INFINITY values correctly: exp(-inf - x) should always be 0, not NaN or 1
    // This is important after top-k filtering sets many logits to -INFINITY
    float local_exp = 0.0f;
    if (gid < n_vocab && !sycl::isinf(local_val)) {
        local_exp = sycl::exp(local_val - local_max);
    }
    slm[tid] = local_exp;
    sycl::group_barrier(item.get_group());

    // Tree reduction for sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }

    // Thread 0 writes block results
    if (tid == 0) {
        block_max[block_id] = local_max;
        block_sum[block_id] = slm[0];
    }
}

// =============================================================================
// Phase 3b: Softmax global reduction + sampling (Pass 2 - single block)
// Computes global max, global sum, then samples from distribution
// Uses online softmax algorithm for numerical stability
// =============================================================================

void gpu_softmax_sample_kernel(
    const float* logits,
    const float* block_max,     // [n_blocks] input
    const float* block_sum,     // [n_blocks] input
    int32_t* result,            // [1] output: sampled token ID
    const int n_vocab,
    const int n_blocks,
    const float random_val,     // Uniform random in [0, 1)
    [[maybe_unused]] const bool greedy,          // If true, return argmax instead
    const sycl::nd_item<1>& item,
    float* slm                  // SLM [block_size]
) {
    const int tid = item.get_local_id(0);
    const int block_size = item.get_local_range(0);

    // Step 1: Find global max from block maxes
    float global_max = -INFINITY;
    for (int i = tid; i < n_blocks; i += block_size) {
        global_max = sycl::fmax(global_max, block_max[i]);
    }
    slm[tid] = global_max;
    sycl::group_barrier(item.get_group());

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] = sycl::fmax(slm[tid], slm[tid + stride]);
        }
        sycl::group_barrier(item.get_group());
    }
    global_max = slm[0];
    sycl::group_barrier(item.get_group());

    // Step 2: Compute global sum by rescaling block sums
    // sum = sum_i(block_sum[i] * exp(block_max[i] - global_max))
    float global_sum = 0.0f;
    for (int i = tid; i < n_blocks; i += block_size) {
        global_sum += block_sum[i] * sycl::exp(block_max[i] - global_max);
    }
    slm[tid] = global_sum;
    sycl::group_barrier(item.get_group());

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }
    global_sum = slm[0];

    // Step 3: Single thread samples from distribution using cumulative approach
    // This is O(n) but only done by one thread, so GPU-parallel is not needed
    if (tid == 0) {
        float threshold = random_val * global_sum;
        float cumsum = 0.0f;
        int32_t selected = n_vocab - 1;  // Default to last token

        // Linear scan through vocabulary
        // This is sequential but runs on GPU avoiding CPU sync
        // Handle -INFINITY values correctly: exp(-inf - x) should be 0, not NaN or 1
        // This is important after top-k filtering sets many logits to -INFINITY
        for (int i = 0; i < n_vocab; i++) {
            float logit_val = logits[i];
            float prob = sycl::isinf(logit_val) ? 0.0f : sycl::exp(logit_val - global_max);
            cumsum += prob;
            if (cumsum >= threshold) {
                selected = i;
                break;
            }
        }

        *result = selected;
    }
}

// =============================================================================
// Greedy argmax sample kernel (single block, optimized for small n_blocks)
// =============================================================================

void gpu_argmax_sample_kernel(
    [[maybe_unused]] const float* logits,
    const float* block_max,     // [n_blocks] input
    const int32_t* block_idx,   // [n_blocks] input
    int32_t* result,            // [1] output
    const int n_blocks,
    const sycl::nd_item<1>& item,
    float* slm_max,             // SLM [block_size]
    int32_t* slm_idx            // SLM [block_size]
) {
    const int tid = item.get_local_id(0);
    const int block_size = item.get_local_range(0);

    // Load block results
    float local_max = -INFINITY;
    int32_t local_idx = 0;

    for (int i = tid; i < n_blocks; i += block_size) {
        if (block_max[i] > local_max) {
            local_max = block_max[i];
            local_idx = block_idx[i];
        }
    }

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Tree reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    if (tid == 0) {
        *result = slm_idx[0];
    }
}

// =============================================================================
// Phase 4: Top-K filtering
// Keep only the top-k highest logit values, set others to -INFINITY
//
// Algorithm (for k << n_vocab):
// - Pass 1: Each block finds local min/max and histogram
// - Pass 2: Binary search to find k-th largest value threshold
// - Pass 3: Filter logits below threshold
//
// Simplified approach for typical k values (32-100):
// - Use iterative threshold refinement
// =============================================================================

// Pass 1: Count values above threshold per block
void gpu_topk_count_kernel(
    const float* logits,
    int32_t* block_counts,     // [n_blocks] output: count of values >= threshold per block
    const int n_vocab,
    const float threshold,
    const sycl::nd_item<1>& item,
    int32_t* slm_count         // SLM [1]
) {
    const int tid = item.get_local_id(0);
    const int block_id = item.get_group(0);
    const int gid = item.get_global_id(0);

    // Each thread counts elements >= threshold
    int local_count = 0;
    if (gid < n_vocab && logits[gid] >= threshold) {
        local_count = 1;
    }

    // Use atomic add for block-level sum (simpler than tree reduction for counting)
    if (tid == 0) {
        slm_count[0] = 0;
    }
    sycl::group_barrier(item.get_group());

    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space> atomic_count(slm_count[0]);
    atomic_count.fetch_add(local_count);

    sycl::group_barrier(item.get_group());

    if (tid == 0) {
        block_counts[block_id] = slm_count[0];
    }
}

// Pass 2: Sum block counts
void gpu_topk_sum_counts_kernel(
    const int32_t* block_counts,
    int32_t* total_count,      // [1] output
    const int n_blocks,
    const sycl::nd_item<1>& item,
    int32_t* slm              // SLM [block_size]
) {
    const int tid = item.get_local_id(0);
    const int block_size = item.get_local_range(0);

    // Sum all block counts
    int32_t local_sum = 0;
    for (int i = tid; i < n_blocks; i += block_size) {
        local_sum += block_counts[i];
    }

    slm[tid] = local_sum;
    sycl::group_barrier(item.get_group());

    // Tree reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }

    if (tid == 0) {
        *total_count = slm[0];
    }
}

// Pass 3: Apply top-k filter (set values below threshold to -INFINITY)
void gpu_topk_filter_kernel(
    float* logits,
    const int n_vocab,
    const float threshold,
    const sycl::nd_item<1>& item
) {
    const int idx = item.get_global_id(0);
    if (idx < n_vocab) {
        if (logits[idx] < threshold) {
            logits[idx] = -INFINITY;
        }
    }
}

// Find k-th largest value using binary search on threshold
// Returns the threshold value that keeps approximately k elements
inline float gpu_topk_find_threshold(
    ggml_backend_sycl_context& ctx,
    const float* logits,
    const int n_vocab,
    const int k,
    float* block_counts_buf,    // [n_blocks] temp buffer (reuse block_max)
    int32_t* total_count_buf,   // [1] temp buffer (reuse selected)
    float global_max,           // Pre-computed max logit
    float global_min            // Pre-computed min logit (or estimate)
) {
    sycl::queue& q = *ctx.stream();
    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((int)((n_vocab + block_size - 1) / block_size),
                                  GPU_SAMPLER_MAX_BLOCKS);

    // Use block_counts as int32_t*
    int32_t* block_counts = reinterpret_cast<int32_t*>(block_counts_buf);

    // Binary search for threshold
    float lo = global_min;
    float hi = global_max;
    float threshold = hi;

    // ~20 iterations gives precision of (hi-lo)/2^20
    for (int iter = 0; iter < 20; iter++) {
        float mid = (lo + hi) * 0.5f;

        // Count elements >= mid
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<int32_t, 1> slm_count(1, h);

            h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_topk_count_kernel(logits, block_counts, n_vocab, mid, item,
                                         SYCL_LOCAL_ACC_PTR(slm_count));
                });
        });

        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<int32_t, 1> slm(block_size, h);

            h.parallel_for(sycl::nd_range<1>(block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_topk_sum_counts_kernel(block_counts, total_count_buf, n_blocks, item,
                                              SYCL_LOCAL_ACC_PTR(slm));
                });
        });

        int32_t count;
        q.memcpy(&count, total_count_buf, sizeof(int32_t)).wait();

        if (count >= k) {
            lo = mid;
            threshold = mid;
        } else {
            hi = mid;
        }

        // Early exit if we found exact k
        if (count == k) break;
    }

    return threshold;
}

// =============================================================================
// Phase 5: Top-P (nucleus) filtering
// Keep tokens with highest probability until cumulative probability > p
//
// Algorithm:
// 1. Compute softmax probabilities
// 2. Sort probabilities descending (with index tracking)
// 3. Find cutoff where cumsum > p
// 4. Set logits for tokens below cutoff to -INFINITY
//
// For simplicity, we use a sequential scan after parallel softmax.
// This is acceptable because top-p typically keeps many tokens.
// =============================================================================

// Compute softmax probabilities in-place (for top-p)
void gpu_softmax_probs_kernel(
    const float* logits,
    float* probs,              // [n_vocab] output
    const int n_vocab,
    const float global_max,
    const float global_sum,
    const sycl::nd_item<1>& item
) {
    const int idx = item.get_global_id(0);
    if (idx < n_vocab) {
        probs[idx] = sycl::exp(logits[idx] - global_max) / global_sum;
    }
}

// Apply top-p filter by scanning probabilities
// This is sequential but runs on GPU
// Sets logits to -INFINITY for tokens with low probability
void gpu_topp_filter_kernel(
    float* logits,
    const float* probs,        // [n_vocab] input (already computed softmax)
    [[maybe_unused]] const float* sorted_probs, // [n_vocab] sorted descending (not used in simple version)
    [[maybe_unused]] const int32_t* sorted_idx, // [n_vocab] indices of sorted probs (not used in simple version)
    const int n_vocab,
    const float top_p,
    const sycl::nd_item<1>& item
) {
    // Only thread 0 does the work (sequential scan)
    if (item.get_global_id(0) != 0) return;

    // Simple O(n^2) approach: for each token, check if it's in top-p
    // We scan through all tokens, accumulating prob for tokens with higher prob

    // First pass: find the probability threshold
    float prob_threshold = 0.0f;

    // Count tokens above each probability level
    // This is O(n^2) but simple and works for any n_vocab
    for (int i = 0; i < n_vocab; i++) {
        float p = probs[i];

        // Count cumulative probability of tokens with prob >= p
        float cum = 0.0f;
        for (int j = 0; j < n_vocab; j++) {
            if (probs[j] >= p) {
                cum += probs[j];
            }
        }

        if (cum >= top_p) {
            prob_threshold = p;
        }
    }

    // Second pass: filter tokens below threshold
    for (int i = 0; i < n_vocab; i++) {
        if (probs[i] < prob_threshold) {
            logits[i] = -INFINITY;
        }
    }
}

// Simpler top-p filter: iterative threshold refinement
// Much faster than O(n^2) - approximately O(n * log(1/p))
void gpu_topp_filter_simple_kernel(
    float* logits,
    const float* probs,        // [n_vocab] softmax probabilities
    [[maybe_unused]] int32_t* mask,             // [n_vocab] work buffer (1 = keep, 0 = filter)
    const int n_vocab,
    const float top_p,
    const sycl::nd_item<1>& item
) {
    // Single thread does sequential scan
    // Can be parallelized but for simplicity we keep it simple
    if (item.get_global_id(0) != 0) return;

    // Find max probability
    float max_prob = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
        }
    }

    // Binary search for probability threshold
    float lo = 0.0f;
    float hi = max_prob;

    for (int iter = 0; iter < 30; iter++) {
        float mid = (lo + hi) * 0.5f;

        // Sum probabilities >= mid
        float cumsum = 0.0f;
        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] >= mid) {
                cumsum += probs[i];
            }
        }

        if (cumsum >= top_p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Apply filter
    float threshold = lo;
    for (int i = 0; i < n_vocab; i++) {
        if (probs[i] < threshold) {
            logits[i] = -INFINITY;
        }
    }
}

// =============================================================================
// Main GPU sampler function
// Runs the full sampling pipeline on GPU and returns the selected token ID
//
// Pipeline: logits -> temp_scale -> top_k -> top_p -> softmax_reduce -> sample
//
// For multi-step decode, this function can be called in a loop without
// CPU sync (except for final result read), with result feeding back to
// embedding lookup.
//
// Complexity:
//   - Pass 1 (parallel): O(n_vocab/block_size) blocks, each O(block_size) work
//   - Pass 2 (single block): O(n_blocks) + O(n_vocab) sequential sample
//   - Total: O(n_vocab) work, O(log n) depth for reductions
// =============================================================================

inline int32_t ggml_sycl_sample_token(
    ggml_backend_sycl_context& ctx,
    float* logits,                          // [n_vocab] on device
    const ggml_sycl_sampler_config& config,
    ggml_sycl_sampler_state& state
) {
    const int n_vocab = state.n_vocab;
    sycl::queue& q = *ctx.stream();

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((int)((n_vocab + block_size - 1) / block_size),
                                  GPU_SAMPLER_MAX_BLOCKS);

    // Lazy allocation of work buffers
    if (!state.initialized) {
        state.block_max = ggml_sycl_gpu_sampler_alloc<float>(GPU_SAMPLER_MAX_BLOCKS, q);
        state.block_sum = ggml_sycl_gpu_sampler_alloc<float>(GPU_SAMPLER_MAX_BLOCKS, q);
        state.block_idx = ggml_sycl_gpu_sampler_alloc<int32_t>(GPU_SAMPLER_MAX_BLOCKS, q);
        state.selected = ggml_sycl_gpu_sampler_alloc<int32_t>(1, q);
        state.random_val = ggml_sycl_gpu_sampler_alloc<float>(1, q);
        state.token_buffer = ggml_sycl_gpu_sampler_alloc<int32_t>(GPU_SAMPLER_TOKEN_BUFFER_SIZE, q);
        state.probs = nullptr;  // Only allocated if top-p is used
        state.rng_state = config.seed;
        state.initialized = true;
    }

    // Step 1: Temperature scaling (if temp != 1.0)
    if (config.temp > 0.0f && config.temp != 1.0f) {
        const float inv_temp = 1.0f / config.temp;
        q.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_temp_scale_kernel(logits, n_vocab, inv_temp, item);
            });
    }

    // Step 2: Top-K filtering (if enabled and not greedy)
    // Skip filtering for greedy since argmax doesn't need filtered logits
    if (!config.greedy && config.top_k > 0 && config.top_k < n_vocab) {
        // First find global max for threshold search
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm_max(block_size, h);
            sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

            float* bmax = state.block_max;
            int32_t* bidx = state.block_idx;

            h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_argmax_block_kernel(logits, bmax, bidx, n_vocab, item,
                                           SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
                });
        });
        q.wait();

        // Get global max from block results
        float global_max = -INFINITY;
        std::vector<float> h_block_max(n_blocks);
        q.memcpy(h_block_max.data(), state.block_max, n_blocks * sizeof(float)).wait();
        for (int i = 0; i < n_blocks; i++) {
            if (h_block_max[i] > global_max) global_max = h_block_max[i];
        }

        // Use a reasonable min estimate
        float global_min = global_max - 100.0f;  // Logits typically span ~20-50

        // Find threshold for top-k
        float threshold = gpu_topk_find_threshold(ctx, logits, n_vocab, config.top_k,
                                                  state.block_max, state.selected,
                                                  global_max, global_min);

        // Apply filter
        q.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_topk_filter_kernel(logits, n_vocab, threshold, item);
            });
        q.wait();
    }

    // Step 3: Top-P filtering (if enabled, < 1.0, and not greedy)
    // Skip filtering for greedy since argmax doesn't need filtered logits
    if (!config.greedy && config.top_p > 0.0f && config.top_p < 1.0f) {
        // Allocate probs buffer if not done yet
        if (state.probs == nullptr) {
            state.probs = ggml_sycl_gpu_sampler_alloc<float>(n_vocab, q);
        }

        // First compute softmax to get probabilities
        // Pass 1: Each block computes local max and sum
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            float* bmax = state.block_max;
            float* bsum = state.block_sum;

            h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_softmax_block_kernel(logits, bmax, bsum, n_vocab, item,
                                            SYCL_LOCAL_ACC_PTR(slm));
                });
        });

        // Compute global max and sum on host (simpler for one-time use)
        std::vector<float> h_block_max(n_blocks), h_block_sum(n_blocks);
        q.memcpy(h_block_max.data(), state.block_max, n_blocks * sizeof(float)).wait();
        q.memcpy(h_block_sum.data(), state.block_sum, n_blocks * sizeof(float)).wait();

        float global_max = -INFINITY;
        for (int i = 0; i < n_blocks; i++) {
            if (h_block_max[i] > global_max) global_max = h_block_max[i];
        }

        float global_sum = 0.0f;
        for (int i = 0; i < n_blocks; i++) {
            global_sum += h_block_sum[i] * std::exp(h_block_max[i] - global_max);
        }

        // Compute softmax probabilities
        float* probs = state.probs;
        q.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_softmax_probs_kernel(logits, probs, n_vocab, global_max, global_sum, item);
            });

        // Apply top-p filter (single thread kernel for simplicity)
        float top_p_val = config.top_p;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_topp_filter_simple_kernel(logits, probs, nullptr, n_vocab, top_p_val, item);
                });
        });
        q.wait();
    }

    // Step 4: Greedy (argmax) path - two-pass reduction
    if (config.greedy || config.temp == 0.0f) {
        // Pass 1: Each block finds local argmax
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm_max(block_size, h);
            sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

            float* bmax = state.block_max;
            int32_t* bidx = state.block_idx;

            h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_argmax_block_kernel(logits, bmax, bidx, n_vocab, item,
                                           SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
                });
        });

        // Pass 2: Single block reduces block results
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm_max(block_size, h);
            sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

            const float* bmax = state.block_max;
            const int32_t* bidx = state.block_idx;
            int32_t* result = state.selected;

            h.parallel_for(sycl::nd_range<1>(block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_argmax_sample_kernel(logits, bmax, bidx, result, n_blocks, item,
                                            SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
                });
        });

        // Read result (single sync point)
        int32_t result;
        q.memcpy(&result, state.selected, sizeof(int32_t)).wait();
        return result;
    }

    // Step 5: Probabilistic sampling path
    // Generate random value on host, copy to device
    float random_val = gpu_rng_uniform(state.rng_state);

    // Pass 1: Each block computes local max and sum(exp(x - local_max))
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(block_size, h);

        float* bmax = state.block_max;
        float* bsum = state.block_sum;

        h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_softmax_block_kernel(logits, bmax, bsum, n_vocab, item,
                                        SYCL_LOCAL_ACC_PTR(slm));
            });
    });

    // Pass 2: Single block computes global max/sum and samples
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(block_size, h);

        const float* bmax = state.block_max;
        const float* bsum = state.block_sum;
        int32_t* result = state.selected;

        h.parallel_for(sycl::nd_range<1>(block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_softmax_sample_kernel(logits, bmax, bsum, result, n_vocab, n_blocks,
                                         random_val, false, item, SYCL_LOCAL_ACC_PTR(slm));
            });
    });

    // Read result (single sync point)
    int32_t result;
    q.memcpy(&result, state.selected, sizeof(int32_t)).wait();
    return result;
}

// Initialize sampler state (call before first use)
inline void ggml_sycl_sampler_init(
    ggml_sycl_sampler_state& state,
    size_t n_vocab,
    uint32_t seed
) {
    state.n_vocab = n_vocab;
    state.rng_state = seed;
    state.initialized = false;
    state.block_max = nullptr;
    state.block_sum = nullptr;
    state.block_idx = nullptr;
    state.probs = nullptr;
    state.selected = nullptr;
    state.random_val = nullptr;
    // Multi-step token buffer
    state.token_buffer = nullptr;
    state.token_write_idx = 0;
    state.token_count = 0;
}

// Free sampler state (call when done)
inline void ggml_sycl_sampler_free(
    ggml_sycl_sampler_state& state,
    sycl::queue& q
) {
    if (state.initialized) {
        if (state.block_max) sycl::free(state.block_max, q);
        if (state.block_sum) sycl::free(state.block_sum, q);
        if (state.block_idx) sycl::free(state.block_idx, q);
        if (state.probs) sycl::free(state.probs, q);
        if (state.selected) sycl::free(state.selected, q);
        if (state.random_val) sycl::free(state.random_val, q);
        if (state.token_buffer) sycl::free(state.token_buffer, q);
        state.initialized = false;
    }
}

// =============================================================================
// Async version: Returns immediately, result written to state.selected
// Use this for multi-step decode to avoid CPU sync between steps
// Call ggml_sycl_sample_token_wait() to get the result
// =============================================================================

inline void ggml_sycl_sample_token_async(
    ggml_backend_sycl_context& ctx,
    float* logits,                          // [n_vocab] on device
    const ggml_sycl_sampler_config& config,
    ggml_sycl_sampler_state& state
) {
    const int n_vocab = state.n_vocab;
    sycl::queue& q = *ctx.stream();

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((int)((n_vocab + block_size - 1) / block_size),
                                  GPU_SAMPLER_MAX_BLOCKS);

    // Lazy allocation
    if (!state.initialized) {
        state.block_max = ggml_sycl_gpu_sampler_alloc<float>(GPU_SAMPLER_MAX_BLOCKS, q);
        state.block_sum = ggml_sycl_gpu_sampler_alloc<float>(GPU_SAMPLER_MAX_BLOCKS, q);
        state.block_idx = ggml_sycl_gpu_sampler_alloc<int32_t>(GPU_SAMPLER_MAX_BLOCKS, q);
        state.selected = ggml_sycl_gpu_sampler_alloc<int32_t>(1, q);
        state.random_val = ggml_sycl_gpu_sampler_alloc<float>(1, q);
        state.token_buffer = ggml_sycl_gpu_sampler_alloc<int32_t>(GPU_SAMPLER_TOKEN_BUFFER_SIZE, q);
        state.probs = nullptr;
        state.rng_state = config.seed;
        state.initialized = true;
    }

    // Temperature scaling
    if (config.temp > 0.0f && config.temp != 1.0f) {
        const float inv_temp = 1.0f / config.temp;
        q.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_temp_scale_kernel(logits, n_vocab, inv_temp, item);
            });
    }

    if (config.greedy || config.temp == 0.0f) {
        // Greedy path
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm_max(block_size, h);
            sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

            float* bmax = state.block_max;
            int32_t* bidx = state.block_idx;

            h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_argmax_block_kernel(logits, bmax, bidx, n_vocab, item,
                                           SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
                });
        });

        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm_max(block_size, h);
            sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

            const float* bmax = state.block_max;
            const int32_t* bidx = state.block_idx;
            int32_t* result = state.selected;

            h.parallel_for(sycl::nd_range<1>(block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_argmax_sample_kernel(logits, bmax, bidx, result, n_blocks, item,
                                            SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
                });
        });
    } else {
        // Probabilistic sampling path
        float random_val = gpu_rng_uniform(state.rng_state);

        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            float* bmax = state.block_max;
            float* bsum = state.block_sum;

            h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_softmax_block_kernel(logits, bmax, bsum, n_vocab, item,
                                            SYCL_LOCAL_ACC_PTR(slm));
                });
        });

        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            const float* bmax = state.block_max;
            const float* bsum = state.block_sum;
            int32_t* result = state.selected;

            h.parallel_for(sycl::nd_range<1>(block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_softmax_sample_kernel(logits, bmax, bsum, result, n_vocab, n_blocks,
                                             random_val, false, item, SYCL_LOCAL_ACC_PTR(slm));
                });
        });
    }
    // No wait - returns immediately
}

// Wait for async sampling to complete and get result
inline int32_t ggml_sycl_sample_token_wait(
    ggml_backend_sycl_context& ctx,
    ggml_sycl_sampler_state& state
) {
    sycl::queue& q = *ctx.stream();
    int32_t result;
    q.memcpy(&result, state.selected, sizeof(int32_t)).wait();
    return result;
}

// =============================================================================
// Multi-step token buffer API
// These functions support generating multiple tokens on GPU without CPU sync
// =============================================================================

// Reset the token buffer (call before starting multi-step generation)
inline void ggml_sycl_sampler_reset_buffer(
    ggml_sycl_sampler_state& state
) {
    state.token_write_idx = 0;
    state.token_count = 0;
}

// Sample token and store in ring buffer (no CPU sync)
// Returns the buffer index where token was written
inline int ggml_sycl_sample_token_to_buffer(
    ggml_backend_sycl_context& ctx,
    float* logits,
    const ggml_sycl_sampler_config& config,
    ggml_sycl_sampler_state& state
) {
    sycl::queue& q = *ctx.stream();
    const int n_vocab = state.n_vocab;

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((int)((n_vocab + block_size - 1) / block_size),
                                  GPU_SAMPLER_MAX_BLOCKS);

    // Lazy allocation
    if (!state.initialized) {
        state.block_max = ggml_sycl_gpu_sampler_alloc<float>(GPU_SAMPLER_MAX_BLOCKS, q);
        state.block_sum = ggml_sycl_gpu_sampler_alloc<float>(GPU_SAMPLER_MAX_BLOCKS, q);
        state.block_idx = ggml_sycl_gpu_sampler_alloc<int32_t>(GPU_SAMPLER_MAX_BLOCKS, q);
        state.selected = ggml_sycl_gpu_sampler_alloc<int32_t>(1, q);
        state.random_val = ggml_sycl_gpu_sampler_alloc<float>(1, q);
        state.token_buffer = ggml_sycl_gpu_sampler_alloc<int32_t>(GPU_SAMPLER_TOKEN_BUFFER_SIZE, q);
        state.probs = nullptr;
        state.rng_state = config.seed;
        state.initialized = true;
    }

    // Temperature scaling
    if (config.temp > 0.0f && config.temp != 1.0f) {
        const float inv_temp = 1.0f / config.temp;
        q.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_temp_scale_kernel(logits, n_vocab, inv_temp, item);
            });
    }

    // Get current buffer position
    const int buf_idx = state.token_write_idx % GPU_SAMPLER_TOKEN_BUFFER_SIZE;
    int32_t* token_dst = state.token_buffer + buf_idx;

    if (config.greedy || config.temp == 0.0f) {
        // Greedy sampling - argmax
        float* block_max = state.block_max;
        int32_t* block_idx = state.block_idx;

        // Pass 1: Block-level argmax
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm_max(block_size, h);
            sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);
            h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_argmax_block_kernel(logits, block_max, block_idx, n_vocab, item,
                                           slm_max.get_multi_ptr<sycl::access::decorated::no>().get(),
                                           slm_idx.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });

        // Pass 2: Final reduction - write directly to token buffer
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm_max(GPU_SAMPLER_MAX_BLOCKS, h);
            sycl::local_accessor<int32_t, 1> slm_idx(GPU_SAMPLER_MAX_BLOCKS, h);
            h.parallel_for(sycl::nd_range<1>(GPU_SAMPLER_MAX_BLOCKS, GPU_SAMPLER_MAX_BLOCKS),
                [=](sycl::nd_item<1> item) {
                    gpu_argmax_final_kernel(block_max, block_idx, token_dst, n_blocks, item,
                                           slm_max.get_multi_ptr<sycl::access::decorated::no>().get(),
                                           slm_idx.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    } else {
        // Probabilistic sampling - softmax + categorical sample
        float* bmax = state.block_max;
        float* bsum = state.block_sum;

        // Generate random value on host
        float rand = gpu_rng_uniform(state.rng_state);

        // Pass 1: Block-level max + sum(exp) - uses single SLM for tree reduction
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);
            h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_softmax_block_kernel(logits, bmax, bsum, n_vocab, item,
                                            slm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });

        // Pass 2: Merge block results + sample - write to token buffer
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);
            h.parallel_for(sycl::nd_range<1>(block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_softmax_sample_kernel(logits, bmax, bsum, token_dst, n_vocab, n_blocks,
                                             rand, false, item,
                                             slm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }

    // IMPORTANT: Wait for the sampling kernel to complete before returning
    // The token pointer will be used by D2H copy which may use a different queue instance
    q.wait();

    // Update buffer position
    state.token_write_idx++;
    state.token_count++;

    return buf_idx;
}

// Get device pointer to token at given buffer index
inline int32_t* ggml_sycl_sampler_get_token_ptr(
    ggml_sycl_sampler_state& state,
    int buf_idx
) {
    if (!state.initialized || !state.token_buffer) {
        return nullptr;
    }
    return state.token_buffer + (buf_idx % GPU_SAMPLER_TOKEN_BUFFER_SIZE);
}

// Get device pointer to current (most recently written) token
inline int32_t* ggml_sycl_sampler_get_current_token_ptr(
    ggml_sycl_sampler_state& state
) {
    if (!state.initialized || !state.token_buffer || state.token_count == 0) {
        return nullptr;
    }
    int last_idx = (state.token_write_idx - 1) % GPU_SAMPLER_TOKEN_BUFFER_SIZE;
    return state.token_buffer + last_idx;
}

// Copy tokens from device buffer to host (sync point)
// Returns number of tokens copied
inline int ggml_sycl_sampler_get_tokens(
    ggml_backend_sycl_context& ctx,
    ggml_sycl_sampler_state& state,
    int32_t* host_tokens,      // Host buffer to copy to
    int max_tokens             // Max tokens to copy
) {
    if (!state.initialized || !state.token_buffer || state.token_count == 0) {
        return 0;
    }

    sycl::queue& q = *ctx.stream();

    int n_to_copy = std::min(state.token_count, max_tokens);
    n_to_copy = std::min(n_to_copy, GPU_SAMPLER_TOKEN_BUFFER_SIZE);

    // Calculate start position in ring buffer
    int start_idx = 0;
    if (state.token_count > GPU_SAMPLER_TOKEN_BUFFER_SIZE) {
        // Buffer wrapped - start from oldest token
        start_idx = state.token_write_idx % GPU_SAMPLER_TOKEN_BUFFER_SIZE;
    }

    // Handle wrap-around in ring buffer
    if (start_idx + n_to_copy <= GPU_SAMPLER_TOKEN_BUFFER_SIZE) {
        // No wrap - single copy
        q.memcpy(host_tokens, state.token_buffer + start_idx,
                 n_to_copy * sizeof(int32_t)).wait();
    } else {
        // Wrap - two copies
        int first_part = GPU_SAMPLER_TOKEN_BUFFER_SIZE - start_idx;
        int second_part = n_to_copy - first_part;
        q.memcpy(host_tokens, state.token_buffer + start_idx,
                 first_part * sizeof(int32_t));
        q.memcpy(host_tokens + first_part, state.token_buffer,
                 second_part * sizeof(int32_t)).wait();
    }

    return n_to_copy;
}

// =============================================================================
// GPU-side Speculative Verification
// Verifies draft tokens against model logits entirely on GPU
// Returns the number of accepted tokens (longest matching prefix)
// =============================================================================

// Kernel: Compute argmax for each position and compare with draft
// Each work-group handles one position's argmax + comparison
void gpu_verify_speculative_kernel(
    const float* all_logits,      // [n_draft, n_vocab] batched logits
    const int32_t* draft_tokens,  // [n_draft] draft token IDs
    int32_t* matches,             // [n_draft] output: 1 if argmax == draft, 0 otherwise
    const int n_vocab,
    const int n_draft,
    const sycl::nd_item<2>& item,
    float* slm_max,               // SLM [block_size]
    int32_t* slm_idx              // SLM [block_size]
) {
    const int pos = item.get_group(0);  // Which position (0 to n_draft-1)
    const int tid = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (pos >= n_draft) return;

    // Point to this position's logits
    const float* logits = all_logits + pos * n_vocab;

    // Phase 1: Each thread finds max in its chunk
    float local_max = -INFINITY;
    int32_t local_idx = 0;

    for (int i = tid; i < n_vocab; i += block_size) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Phase 2: Parallel reduction to find global argmax
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    // Thread 0 writes result
    if (tid == 0) {
        int32_t argmax_token = slm_idx[0];
        int32_t draft_token = draft_tokens[pos];
        matches[pos] = (argmax_token == draft_token) ? 1 : 0;
    }
}

// Kernel: Compute argmax for each position, compare with draft, AND output sampled tokens
// Extended version that also returns the argmax tokens
void gpu_verify_speculative_with_tokens_kernel(
    const float* all_logits,      // [n_draft, n_vocab] batched logits
    const int32_t* draft_tokens,  // [n_draft] draft token IDs
    int32_t* matches,             // [n_draft] output: 1 if argmax == draft, 0 otherwise
    int32_t* sampled_tokens,      // [n_draft] output: argmax token at each position
    const int n_vocab,
    const int n_draft,
    const sycl::nd_item<2>& item,
    float* slm_max,               // SLM [block_size]
    int32_t* slm_idx              // SLM [block_size]
) {
    const int pos = item.get_group(0);  // Which position (0 to n_draft-1)
    const int tid = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (pos >= n_draft) return;

    // Point to this position's logits
    const float* logits = all_logits + pos * n_vocab;

    // Phase 1: Each thread finds max in its chunk
    float local_max = -INFINITY;
    int32_t local_idx = 0;

    for (int i = tid; i < n_vocab; i += block_size) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Phase 2: Parallel reduction to find global argmax
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    // Thread 0 writes result
    if (tid == 0) {
        int32_t argmax_token = slm_idx[0];
        int32_t draft_token = draft_tokens[pos];
        matches[pos] = (argmax_token == draft_token) ? 1 : 0;
        sampled_tokens[pos] = argmax_token;  // Also output the sampled token
    }
}

// Kernel: Find first mismatch (prefix scan to find first 0)
// Single work-group kernel
void gpu_find_first_mismatch_kernel(
    const int32_t* matches,       // [n_draft] match flags
    int32_t* n_accepted,          // [1] output: number of accepted tokens
    const int n_draft,
    const sycl::nd_item<1>& item
) {
    // Single-threaded for simplicity (n_draft is typically small, <16)
    if (item.get_global_id(0) != 0) return;

    int accepted = 0;
    for (int i = 0; i < n_draft; i++) {
        if (matches[i] == 0) break;
        accepted++;
    }
    *n_accepted = accepted;
}

// Main verification function
// Returns number of accepted tokens (prefix that matches argmax)
inline int ggml_sycl_verify_speculative(
    ggml_backend_sycl_context& ctx,
    ggml_sycl_sampler_state& state,
    const float* all_logits,      // [n_draft, n_vocab] on device
    const int32_t* draft_tokens,  // [n_draft] on device (or host, will be copied)
    int n_draft,
    int n_vocab
) {
    sycl::queue& q = *ctx.stream();

    // Allocate temporary buffers if needed
    // Reuse state.block_idx for matches array (size >= n_draft typically)
    int32_t* matches = nullptr;
    int32_t* n_accepted_dev = nullptr;
    bool allocated_matches = false;
    bool allocated_n_accepted = false;

    // Use block_idx if available and large enough, otherwise allocate
    if (state.initialized && state.block_idx &&
        GPU_SAMPLER_MAX_BLOCKS >= n_draft) {
        matches = state.block_idx;
    } else {
        matches = ggml_sycl_gpu_sampler_alloc<int32_t>(n_draft, q);
        allocated_matches = true;
    }

    // Use selected for n_accepted output
    if (state.initialized && state.selected) {
        n_accepted_dev = state.selected;
    } else {
        n_accepted_dev = ggml_sycl_gpu_sampler_alloc<int32_t>(1, q);
        allocated_n_accepted = true;
    }

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;

    // Kernel 1: Compute argmax for each position and compare with draft
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm_max(block_size, h);
        sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_draft, block_size),
                              sycl::range<2>(1, block_size)),
            [=](sycl::nd_item<2> item) {
                gpu_verify_speculative_kernel(
                    all_logits, draft_tokens, matches, n_vocab, n_draft, item,
                    SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
            });
    });

    // Kernel 2: Find first mismatch
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(1, 1),
            [=](sycl::nd_item<1> item) {
                gpu_find_first_mismatch_kernel(matches, n_accepted_dev, n_draft, item);
            });
    });

    // Copy result back (single sync point)
    int32_t n_accepted;
    q.memcpy(&n_accepted, n_accepted_dev, sizeof(int32_t)).wait();

    // Cleanup temporary allocations
    if (allocated_matches) sycl::free(matches, q);
    if (allocated_n_accepted) sycl::free(n_accepted_dev, q);

    return n_accepted;
}

// Convenience wrapper that handles draft tokens on host
// Copies draft_tokens to device, then calls the main function
inline int ggml_sycl_verify_speculative_host(
    ggml_backend_sycl_context& ctx,
    ggml_sycl_sampler_state& state,
    const float* all_logits,           // [n_draft, n_vocab] on device
    const int32_t* draft_tokens_host,  // [n_draft] on host
    int n_draft,
    int n_vocab
) {
    sycl::queue& q = *ctx.stream();

    // Copy draft tokens to device
    int32_t* draft_tokens_dev = ggml_sycl_gpu_sampler_alloc<int32_t>(n_draft, q);
    q.memcpy(draft_tokens_dev, draft_tokens_host, n_draft * sizeof(int32_t)).wait();

    // Call main function
    int n_accepted = ggml_sycl_verify_speculative(
        ctx, state, all_logits, draft_tokens_dev, n_draft, n_vocab);

    // Cleanup
    sycl::free(draft_tokens_dev, q);

    return n_accepted;
}

// =============================================================================
// PHASE 1: Multi-Sequence GPU Sampler for Continuous Batching
// Samples tokens for multiple sequences in parallel on GPU
// =============================================================================

// Maximum concurrent sequences for continuous batching
constexpr int GPU_SAMPLER_MAX_SEQS = 128;

// Multi-sequence sampler state for continuous batching
struct ggml_sycl_multi_seq_sampler_state {
    int n_seqs;                    // Number of active sequences (0 to max_seqs)
    int max_seqs;                  // Maximum sequences this state can handle
    int n_vocab;                   // Vocabulary size

    // Per-sequence parameters [max_seqs]
    float* temperatures;           // Temperature per sequence (device memory)
    uint32_t* rng_states;          // RNG state per sequence (device memory)
    int32_t* sampled_tokens;       // Output tokens per sequence (device memory)

    // Per-sequence top-k/top-p parameters [max_seqs]
    int32_t* top_k_values;         // Top-k per sequence (device memory)
    float* top_p_values;           // Top-p per sequence (device memory)
    float* min_p_values;           // Min-p per sequence (device memory)
    uint8_t* greedy_flags;         // Greedy mode per sequence (device memory, 0=false, 1=true)

    // Work buffers for multi-block reductions [n_seqs, MAX_BLOCKS]
    float* block_max;              // Per-block max values
    float* block_sum;              // Per-block sum values
    int32_t* block_idx;            // Per-block argmax indices

    // Sequence management
    int* seq_ids;                  // Sequence ID mapping (device memory)

    bool initialized;
};

// =============================================================================
// Multi-sequence temperature scaling kernel
// Each work-item handles one logit value, organized as [seq, vocab]
// =============================================================================

void gpu_multi_seq_temp_scale_kernel(
    float* all_logits,            // [n_seqs, n_vocab] batched logits
    const float* temperatures,     // [n_seqs] per-sequence temperatures
    const int n_vocab,
    const int n_seqs,
    const sycl::nd_item<2>& item
) {
    const int seq = item.get_global_id(0);
    const int vocab = item.get_global_id(1);

    if (seq >= n_seqs || vocab >= n_vocab) return;

    float temp = temperatures[seq];
    if (temp > 0.0f && temp != 1.0f) {
        float inv_temp = 1.0f / temp;
        all_logits[seq * n_vocab + vocab] *= inv_temp;
    }
}

// =============================================================================
// Multi-sequence block-level max + sum(exp) kernel
// Each work-group handles one (sequence, block) pair
// Block dimension 0 = sequence, Block dimension 1 = vocab block
// =============================================================================

void gpu_multi_seq_softmax_block_kernel(
    const float* all_logits,      // [n_seqs, n_vocab]
    float* block_max,             // [n_seqs, n_blocks] output
    float* block_sum,             // [n_seqs, n_blocks] output
    const int n_vocab,
    const int n_seqs,
    const int n_blocks,
    const sycl::nd_item<2>& item,
    float* slm                    // SLM [block_size]
) {
    const int seq = item.get_group(0);
    const int block_id = item.get_group(1);
    const int tid = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs) return;

    // Global position in vocabulary
    const int vocab_start = block_id * block_size;
    const int gid = vocab_start + tid;

    // Load and find local max
    const float* seq_logits = all_logits + seq * n_vocab;
    float local_val = (gid < n_vocab) ? seq_logits[gid] : -INFINITY;
    slm[tid] = local_val;
    sycl::group_barrier(item.get_group());

    // Tree reduction for max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] = sycl::fmax(slm[tid], slm[tid + stride]);
        }
        sycl::group_barrier(item.get_group());
    }

    float local_max = slm[0];
    sycl::group_barrier(item.get_group());

    // Compute exp(x - local_max) and sum
    float local_exp = 0.0f;
    if (gid < n_vocab && !sycl::isinf(local_val)) {
        local_exp = sycl::exp(local_val - local_max);
    }
    slm[tid] = local_exp;
    sycl::group_barrier(item.get_group());

    // Tree reduction for sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }

    // Thread 0 writes block results
    if (tid == 0) {
        int out_idx = seq * n_blocks + block_id;
        block_max[out_idx] = local_max;
        block_sum[out_idx] = slm[0];
    }
}

// =============================================================================
// Multi-sequence final softmax + sample kernel
// Each work-group handles one sequence's final reduction and sampling
// =============================================================================

void gpu_multi_seq_softmax_sample_kernel(
    const float* all_logits,      // [n_seqs, n_vocab]
    const float* block_max,       // [n_seqs, n_blocks]
    const float* block_sum,       // [n_seqs, n_blocks]
    int32_t* results,             // [n_seqs] output: sampled token IDs
    const uint32_t* rng_states,   // [n_seqs] RNG states
    const uint8_t* greedy_flags,  // [n_seqs] greedy mode flags (0=false, 1=true)
    const int n_vocab,
    const int n_seqs,
    const int n_blocks,
    const sycl::nd_item<2>& item,
    float* slm                    // SLM [block_size]
) {
    const int seq = item.get_group(0);
    const int tid = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs) return;

    const float* seq_logits = all_logits + seq * n_vocab;
    const float* seq_block_max = block_max + seq * n_blocks;
    const float* seq_block_sum = block_sum + seq * n_blocks;
    bool is_greedy = (greedy_flags[seq] != 0);

    // Step 1: Find global max from block maxes
    float global_max = -INFINITY;
    for (int i = tid; i < n_blocks; i += block_size) {
        global_max = sycl::fmax(global_max, seq_block_max[i]);
    }
    slm[tid] = global_max;
    sycl::group_barrier(item.get_group());

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] = sycl::fmax(slm[tid], slm[tid + stride]);
        }
        sycl::group_barrier(item.get_group());
    }
    global_max = slm[0];
    sycl::group_barrier(item.get_group());

    // Step 2: Compute global sum by rescaling block sums
    float global_sum = 0.0f;
    for (int i = tid; i < n_blocks; i += block_size) {
        global_sum += seq_block_sum[i] * sycl::exp(seq_block_max[i] - global_max);
    }
    slm[tid] = global_sum;
    sycl::group_barrier(item.get_group());

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }
    global_sum = slm[0];

    // Step 3: Single thread samples from distribution
    if (tid == 0) {
        if (is_greedy) {
            // Argmax for greedy decoding
            float max_val = -INFINITY;
            int32_t max_idx = 0;
            for (int i = 0; i < n_vocab; i++) {
                if (seq_logits[i] > max_val) {
                    max_val = seq_logits[i];
                    max_idx = i;
                }
            }
            results[seq] = max_idx;
        } else {
            // Sample from distribution
            // Use RNG state to generate random value
            uint32_t rng = rng_states[seq];
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            float random_val = static_cast<float>(rng) / static_cast<float>(UINT32_MAX);

            float threshold = random_val * global_sum;
            float cumsum = 0.0f;
            int32_t selected = n_vocab - 1;

            for (int i = 0; i < n_vocab; i++) {
                float logit_val = seq_logits[i];
                float prob = sycl::isinf(logit_val) ? 0.0f : sycl::exp(logit_val - global_max);
                cumsum += prob;
                if (cumsum >= threshold) {
                    selected = i;
                    break;
                }
            }
            results[seq] = selected;
        }
    }
}

// =============================================================================
// Multi-sequence argmax kernel (greedy decoding path)
// Each work-group handles one (sequence, block) pair
// =============================================================================

void gpu_multi_seq_argmax_block_kernel(
    const float* all_logits,      // [n_seqs, n_vocab]
    float* block_max,             // [n_seqs, n_blocks] output
    int32_t* block_idx,           // [n_seqs, n_blocks] output
    const int n_vocab,
    const int n_seqs,
    const int n_blocks,
    const sycl::nd_item<2>& item,
    float* slm_max,               // SLM [block_size]
    int32_t* slm_idx              // SLM [block_size]
) {
    const int seq = item.get_group(0);
    const int block_id = item.get_group(1);
    const int tid = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs) return;

    const int vocab_start = block_id * block_size;
    const int gid = vocab_start + tid;

    const float* seq_logits = all_logits + seq * n_vocab;

    // Load and find local max
    float local_max = (gid < n_vocab) ? seq_logits[gid] : -INFINITY;
    int32_t local_idx = gid;

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Tree reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    // Thread 0 writes block result
    if (tid == 0) {
        int out_idx = seq * n_blocks + block_id;
        block_max[out_idx] = slm_max[0];
        block_idx[out_idx] = slm_idx[0];
    }
}

// =============================================================================
// Multi-sequence final argmax kernel
// Each work-group handles one sequence's final argmax reduction
// =============================================================================

void gpu_multi_seq_argmax_final_kernel(
    const float* block_max,       // [n_seqs, n_blocks]
    const int32_t* block_idx,     // [n_seqs, n_blocks]
    int32_t* results,             // [n_seqs] output
    const int n_seqs,
    const int n_blocks,
    const sycl::nd_item<2>& item,
    float* slm_max,               // SLM [block_size]
    int32_t* slm_idx              // SLM [block_size]
) {
    const int seq = item.get_group(0);
    const int tid = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs) return;

    const float* seq_block_max = block_max + seq * n_blocks;
    const int32_t* seq_block_idx = block_idx + seq * n_blocks;

    // Load block results
    float local_max = -INFINITY;
    int32_t local_idx = 0;

    for (int i = tid; i < n_blocks; i += block_size) {
        if (seq_block_max[i] > local_max) {
            local_max = seq_block_max[i];
            local_idx = seq_block_idx[i];
        }
    }

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Tree reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    if (tid == 0) {
        results[seq] = slm_idx[0];
    }
}

// =============================================================================
// Multi-sequence sampler initialization
// =============================================================================

inline void ggml_sycl_multi_seq_sampler_init(
    ggml_sycl_multi_seq_sampler_state& state,
    int max_seqs,
    int n_vocab
) {
    state.n_seqs = 0;
    state.max_seqs = max_seqs;
    state.n_vocab = n_vocab;
    state.initialized = false;

    // All pointers null until allocated
    state.temperatures = nullptr;
    state.rng_states = nullptr;
    state.sampled_tokens = nullptr;
    state.top_k_values = nullptr;
    state.top_p_values = nullptr;
    state.min_p_values = nullptr;
    state.greedy_flags = nullptr;
    state.block_max = nullptr;
    state.block_sum = nullptr;
    state.block_idx = nullptr;
    state.seq_ids = nullptr;
}

// =============================================================================
// Multi-sequence sampler allocation (lazy, called on first use)
// =============================================================================

inline void ggml_sycl_multi_seq_sampler_alloc(
    ggml_sycl_multi_seq_sampler_state& state,
    sycl::queue& q
) {
    if (state.initialized) return;

    const int max_seqs = state.max_seqs;
    const int n_vocab = state.n_vocab;
    const int n_blocks = (n_vocab + GPU_SAMPLER_BLOCK_SIZE - 1) / GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks_capped = std::min(n_blocks, GPU_SAMPLER_MAX_BLOCKS);

    // Per-sequence arrays [max_seqs]
    state.temperatures = ggml_sycl_gpu_sampler_alloc<float>(max_seqs, q);
    state.rng_states = ggml_sycl_gpu_sampler_alloc<uint32_t>(max_seqs, q);
    state.sampled_tokens = ggml_sycl_gpu_sampler_alloc<int32_t>(max_seqs, q);
    state.top_k_values = ggml_sycl_gpu_sampler_alloc<int32_t>(max_seqs, q);
    state.top_p_values = ggml_sycl_gpu_sampler_alloc<float>(max_seqs, q);
    state.min_p_values = ggml_sycl_gpu_sampler_alloc<float>(max_seqs, q);
    state.greedy_flags = ggml_sycl_gpu_sampler_alloc<uint8_t>(max_seqs, q);
    state.seq_ids = ggml_sycl_gpu_sampler_alloc<int>(max_seqs, q);

    // Work buffers [max_seqs, n_blocks]
    state.block_max = ggml_sycl_gpu_sampler_alloc<float>(max_seqs * n_blocks_capped, q);
    state.block_sum = ggml_sycl_gpu_sampler_alloc<float>(max_seqs * n_blocks_capped, q);
    state.block_idx = ggml_sycl_gpu_sampler_alloc<int32_t>(max_seqs * n_blocks_capped, q);

    // Initialize default values
    std::vector<float> h_temps(max_seqs, 1.0f);
    std::vector<int32_t> h_top_k(max_seqs, 0);
    std::vector<float> h_top_p(max_seqs, 1.0f);
    std::vector<float> h_min_p(max_seqs, 0.0f);
    std::vector<uint8_t> h_greedy(max_seqs, 0);  // 0 = false (probabilistic sampling)

    q.memcpy(state.temperatures, h_temps.data(), max_seqs * sizeof(float));
    q.memcpy(state.top_k_values, h_top_k.data(), max_seqs * sizeof(int32_t));
    q.memcpy(state.top_p_values, h_top_p.data(), max_seqs * sizeof(float));
    q.memcpy(state.min_p_values, h_min_p.data(), max_seqs * sizeof(float));
    q.memcpy(state.greedy_flags, h_greedy.data(), max_seqs * sizeof(uint8_t));
    q.wait();

    state.initialized = true;
}

// =============================================================================
// Multi-sequence sampler free
// =============================================================================

inline void ggml_sycl_multi_seq_sampler_free(
    ggml_sycl_multi_seq_sampler_state& state,
    sycl::queue& q
) {
    if (!state.initialized) return;

    if (state.temperatures) sycl::free(state.temperatures, q);
    if (state.rng_states) sycl::free(state.rng_states, q);
    if (state.sampled_tokens) sycl::free(state.sampled_tokens, q);
    if (state.top_k_values) sycl::free(state.top_k_values, q);
    if (state.top_p_values) sycl::free(state.top_p_values, q);
    if (state.min_p_values) sycl::free(state.min_p_values, q);
    if (state.greedy_flags) sycl::free(state.greedy_flags, q);
    if (state.block_max) sycl::free(state.block_max, q);
    if (state.block_sum) sycl::free(state.block_sum, q);
    if (state.block_idx) sycl::free(state.block_idx, q);
    if (state.seq_ids) sycl::free(state.seq_ids, q);

    state.initialized = false;
}

// =============================================================================
// Set per-sequence sampling parameters
// =============================================================================

inline void ggml_sycl_multi_seq_sampler_set_params(
    ggml_sycl_multi_seq_sampler_state& state,
    sycl::queue& q,
    int seq_idx,
    float temp,
    int32_t top_k,
    float top_p,
    float min_p,
    uint32_t seed
) {
    if (!state.initialized || seq_idx >= state.max_seqs) return;

    uint8_t is_greedy = (temp <= 0.0f) ? 1 : 0;

    q.memcpy(state.temperatures + seq_idx, &temp, sizeof(float));
    q.memcpy(state.top_k_values + seq_idx, &top_k, sizeof(int32_t));
    q.memcpy(state.top_p_values + seq_idx, &top_p, sizeof(float));
    q.memcpy(state.min_p_values + seq_idx, &min_p, sizeof(float));
    q.memcpy(state.greedy_flags + seq_idx, &is_greedy, sizeof(uint8_t));
    q.memcpy(state.rng_states + seq_idx, &seed, sizeof(uint32_t));
    q.wait();
}

// =============================================================================
// Main multi-sequence sampling function
// Samples tokens for all active sequences in parallel
// =============================================================================

inline void ggml_sycl_sample_multi_sequence(
    ggml_backend_sycl_context& ctx,
    float* all_logits,                        // [n_seqs, n_vocab] on device
    ggml_sycl_multi_seq_sampler_state& state
) {
    if (state.n_seqs == 0) return;

    sycl::queue& q = *ctx.stream();

    // Lazy allocation
    if (!state.initialized) {
        ggml_sycl_multi_seq_sampler_alloc(state, q);
    }

    const int n_seqs = state.n_seqs;
    const int n_vocab = state.n_vocab;

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((int)((n_vocab + block_size - 1) / block_size),
                                  GPU_SAMPLER_MAX_BLOCKS);

    // Step 1: Temperature scaling (2D grid: seqs x vocab_blocks)
    {
        sycl::range<2> global(n_seqs, ((n_vocab + block_size - 1) / block_size) * block_size);
        sycl::range<2> local(1, block_size);

        float* temps = state.temperatures;

        q.parallel_for(sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) {
                gpu_multi_seq_temp_scale_kernel(all_logits, temps, n_vocab, n_seqs, item);
            });
    }

    // Step 2: Block-level softmax (max + sum) (2D grid: seqs x vocab_blocks)
    {
        sycl::range<2> global(n_seqs, n_blocks);
        sycl::range<2> local(1, block_size);

        float* bmax = state.block_max;
        float* bsum = state.block_sum;

        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_seqs, n_blocks * block_size),
                                              sycl::range<2>(1, block_size)),
                [=](sycl::nd_item<2> item) {
                    gpu_multi_seq_softmax_block_kernel(
                        all_logits, bmax, bsum, n_vocab, n_seqs, n_blocks, item,
                        SYCL_LOCAL_ACC_PTR(slm));
                });
        });
    }

    // Step 3: Final softmax merge + sampling (1 work-group per sequence)
    {
        float* bmax = state.block_max;
        float* bsum = state.block_sum;
        int32_t* results = state.sampled_tokens;
        uint32_t* rng = state.rng_states;
        uint8_t* greedy = state.greedy_flags;

        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_seqs, block_size),
                                              sycl::range<2>(1, block_size)),
                [=](sycl::nd_item<2> item) {
                    gpu_multi_seq_softmax_sample_kernel(
                        all_logits, bmax, bsum, results, rng, greedy,
                        n_vocab, n_seqs, n_blocks, item, SYCL_LOCAL_ACC_PTR(slm));
                });
        });
    }

    // Update RNG states (on device, advance each state)
    // This is done in the sampling kernel itself
}

// =============================================================================
// Get sampled tokens (copy from device to host)
// =============================================================================

inline void ggml_sycl_multi_seq_sampler_get_tokens(
    ggml_sycl_multi_seq_sampler_state& state,
    sycl::queue& q,
    int32_t* host_tokens,         // [n_seqs] output on host
    int n_tokens                  // Number of tokens to copy (usually = n_seqs)
) {
    if (!state.initialized || state.n_seqs == 0) return;

    int n_copy = std::min(n_tokens, state.n_seqs);
    q.memcpy(host_tokens, state.sampled_tokens, n_copy * sizeof(int32_t)).wait();
}

// =============================================================================
// Extended Speculative Verification (returns sampled tokens too)
// For llama-lookup and other tools that need to know the actual sampled tokens
// =============================================================================

// Extended verification function
// Returns number of accepted tokens AND the sampled tokens for each position
// sampled_tokens_out: [n_draft] host array to receive argmax tokens
inline int ggml_sycl_verify_speculative_with_tokens(
    ggml_backend_sycl_context& ctx,
    [[maybe_unused]] ggml_sycl_sampler_state& state,
    const float* all_logits,           // [n_draft, n_vocab] on device
    const int32_t* draft_tokens_host,  // [n_draft] on host
    int32_t* sampled_tokens_out,       // [n_draft] on host - output sampled tokens
    int n_draft,
    int n_vocab
) {
    sycl::queue& q = *ctx.stream();

    // Allocate temporary buffers
    int32_t* draft_tokens_dev = ggml_sycl_gpu_sampler_alloc<int32_t>(n_draft, q);
    int32_t* matches = ggml_sycl_gpu_sampler_alloc<int32_t>(n_draft, q);
    int32_t* sampled_tokens_dev = ggml_sycl_gpu_sampler_alloc<int32_t>(n_draft, q);
    int32_t* n_accepted_dev = ggml_sycl_gpu_sampler_alloc<int32_t>(1, q);

    // Copy draft tokens to device
    q.memcpy(draft_tokens_dev, draft_tokens_host, n_draft * sizeof(int32_t)).wait();

    // Debug: Print draft tokens
    GGML_LOG_DEBUG("[GPU VERIFY] n_draft=%d, draft_tokens:", n_draft);
    for (int i = 0; i < n_draft && i < 10; i++) {
        GGML_LOG_DEBUG(" [%d]=%d", i, draft_tokens_host[i]);
    }
    GGML_LOG_DEBUG("\n");

    // Debug: Read first few logits values to verify data is valid
    float debug_logits[5];
    q.memcpy(debug_logits, all_logits, 5 * sizeof(float)).wait();
    GGML_LOG_DEBUG("GPU verify logits sample: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
                   debug_logits[0], debug_logits[1], debug_logits[2], debug_logits[3], debug_logits[4]);

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;

    // Kernel 1: Compute argmax for each position and compare with draft
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm_max(block_size, h);
        sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_draft, block_size),
                              sycl::range<2>(1, block_size)),
            [=](sycl::nd_item<2> item) {
                gpu_verify_speculative_with_tokens_kernel(
                    all_logits, draft_tokens_dev, matches, sampled_tokens_dev,
                    n_vocab, n_draft, item,
                    SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
            });
    });

    // Kernel 2: Find first mismatch
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(1, 1),
            [=](sycl::nd_item<1> item) {
                gpu_find_first_mismatch_kernel(matches, n_accepted_dev, n_draft, item);
            });
    });

    // Copy results back (single sync point)
    int32_t n_accepted;
    q.memcpy(&n_accepted, n_accepted_dev, sizeof(int32_t));
    q.memcpy(sampled_tokens_out, sampled_tokens_dev, n_draft * sizeof(int32_t)).wait();

    // Debug: Print sampled tokens for analysis
    GGML_LOG_DEBUG("[GPU VERIFY] n_accepted=%d, sampled_tokens:", n_accepted);
    for (int i = 0; i < n_draft && i < 10; i++) {
        GGML_LOG_DEBUG(" [%d]=%d", i, sampled_tokens_out[i]);
    }
    GGML_LOG_DEBUG("\n");

    // Cleanup
    sycl::free(draft_tokens_dev, q);
    sycl::free(matches, q);
    sycl::free(sampled_tokens_dev, q);
    sycl::free(n_accepted_dev, q);

    return n_accepted;
}
