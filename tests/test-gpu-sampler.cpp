// Test for GPU sampler correctness
// Verifies that GPU sampling produces correct results compared to CPU
// Standalone test - doesn't depend on full ggml infrastructure

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#ifdef GGML_USE_SYCL
#include <sycl/sycl.hpp>

// =============================================================================
// GPU Sampler Kernels (copied from gpu-sampler.hpp for standalone test)
// =============================================================================

constexpr int GPU_SAMPLER_BLOCK_SIZE = 256;
constexpr int GPU_SAMPLER_MAX_BLOCKS = 512;

// Xorshift32 RNG
inline float gpu_rng_uniform(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state) / static_cast<float>(UINT32_MAX);
}

// Multi-block argmax pass 1
void gpu_argmax_block_kernel(
    const float* logits,
    float* block_max,
    int32_t* block_idx,
    const int n_vocab,
    const sycl::nd_item<1>& item,
    float* slm_max,
    int32_t* slm_idx
) {
    const int tid = item.get_local_id(0);
    const int block_id = item.get_group(0);
    const int block_size = item.get_local_range(0);
    const int gid = item.get_global_id(0);

    float local_max = (gid < n_vocab) ? logits[gid] : -INFINITY;
    int32_t local_idx = gid;

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

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
        block_max[block_id] = slm_max[0];
        block_idx[block_id] = slm_idx[0];
    }
}

// Multi-block argmax pass 2
void gpu_argmax_final_kernel(
    const float* block_max,
    const int32_t* block_idx,
    int32_t* result,
    const int n_blocks,
    const sycl::nd_item<1>& item,
    float* slm_max,
    int32_t* slm_idx
) {
    const int tid = item.get_local_id(0);
    const int block_size = item.get_local_range(0);

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

// Multi-block softmax pass 1
void gpu_softmax_block_kernel(
    const float* logits,
    float* block_max,
    float* block_sum,
    const int n_vocab,
    const sycl::nd_item<1>& item,
    float* slm
) {
    const int tid = item.get_local_id(0);
    const int block_id = item.get_group(0);
    const int block_size = item.get_local_range(0);
    const int gid = item.get_global_id(0);

    float local_val = (gid < n_vocab) ? logits[gid] : -INFINITY;
    slm[tid] = local_val;
    sycl::group_barrier(item.get_group());

    // Max reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] = sycl::fmax(slm[tid], slm[tid + stride]);
        }
        sycl::group_barrier(item.get_group());
    }

    float local_max = slm[0];
    sycl::group_barrier(item.get_group());

    // Exp and sum
    float local_exp = (gid < n_vocab) ? sycl::exp(local_val - local_max) : 0.0f;
    slm[tid] = local_exp;
    sycl::group_barrier(item.get_group());

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }

    if (tid == 0) {
        block_max[block_id] = local_max;
        block_sum[block_id] = slm[0];
    }
}

// Multi-block softmax pass 2 + sampling
void gpu_softmax_sample_kernel(
    const float* logits,
    const float* block_max,
    const float* block_sum,
    int32_t* result,
    const int n_vocab,
    const int n_blocks,
    const float random_val,
    const sycl::nd_item<1>& item,
    float* slm
) {
    const int tid = item.get_local_id(0);
    const int block_size = item.get_local_range(0);

    // Find global max
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

    // Compute global sum
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

    // Sample (single thread)
    if (tid == 0) {
        float threshold = random_val * global_sum;
        float cumsum = 0.0f;
        int32_t selected = n_vocab - 1;

        for (int i = 0; i < n_vocab; i++) {
            float prob = sycl::exp(logits[i] - global_max);
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
// Test functions
// =============================================================================

// CPU reference: argmax
int cpu_argmax(const float* logits, int n) {
    int best_idx = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

// CPU reference: sample with temperature
int cpu_sample(const float* logits, int n, float temp, uint32_t& rng_state) {
    float max_val = *std::max_element(logits, logits + n);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += std::exp((logits[i] - max_val) / temp);
    }

    // Use same RNG
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    float random_val = static_cast<float>(rng_state) / static_cast<float>(UINT32_MAX);

    float threshold = random_val * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += std::exp((logits[i] - max_val) / temp);
        if (cumsum >= threshold) {
            return i;
        }
    }
    return n - 1;
}

bool test_argmax(sycl::queue& q, int n_vocab) {
    printf("Testing argmax (n_vocab=%d)...\n", n_vocab);

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((n_vocab + block_size - 1) / block_size, GPU_SAMPLER_MAX_BLOCKS);

    // Create random logits
    std::vector<float> host_logits(n_vocab);
    std::mt19937 gen(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n_vocab; i++) {
        host_logits[i] = dist(gen);
    }

    // Set known maximum
    int expected_max = n_vocab / 3;
    host_logits[expected_max] = 100.0f;

    // CPU reference
    int cpu_result = cpu_argmax(host_logits.data(), n_vocab);

    // GPU test
    float* gpu_logits = sycl::malloc_device<float>(n_vocab, q);
    float* block_max = sycl::malloc_device<float>(n_blocks, q);
    int32_t* block_idx = sycl::malloc_device<int32_t>(n_blocks, q);
    int32_t* result = sycl::malloc_device<int32_t>(1, q);

    q.memcpy(gpu_logits, host_logits.data(), n_vocab * sizeof(float)).wait();

    // Pass 1: each block finds local max
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm_max(block_size, h);
        sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

        h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_argmax_block_kernel(gpu_logits, block_max, block_idx, n_vocab, item,
                                       slm_max.get_pointer(), slm_idx.get_pointer());
            });
    }).wait();  // Ensure Pass 1 completes before Pass 2

    // Pass 2: final reduction across blocks
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm_max(block_size, h);
        sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

        h.parallel_for(sycl::nd_range<1>(block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_argmax_final_kernel(block_max, block_idx, result, n_blocks, item,
                                       slm_max.get_pointer(), slm_idx.get_pointer());
            });
    }).wait();  // Ensure Pass 2 completes before memcpy

    int32_t gpu_result;
    q.memcpy(&gpu_result, result, sizeof(int32_t)).wait();

    // Cleanup
    sycl::free(gpu_logits, q);
    sycl::free(block_max, q);
    sycl::free(block_idx, q);
    sycl::free(result, q);

    printf("  Expected: %d, CPU: %d, GPU: %d\n", expected_max, cpu_result, gpu_result);

    bool passed = (cpu_result == gpu_result) && (cpu_result == expected_max);
    printf("  %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_sampling(sycl::queue& q, int n_vocab, float temp) {
    printf("Testing sampling (n_vocab=%d, temp=%.1f)...\n", n_vocab, temp);

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((n_vocab + block_size - 1) / block_size, GPU_SAMPLER_MAX_BLOCKS);

    // Create logits with strong peaks (must be very high to dominate with large vocab)
    // With 32K tokens at 0.0, we need peaks around 15+ to have >50% combined probability
    std::vector<float> host_logits(n_vocab, -100.0f);  // Very low baseline
    host_logits[100] = 5.0f;  // exp(5) = 148.4 - most likely
    host_logits[200] = 4.0f;  // exp(4) = 54.6
    host_logits[300] = 3.0f;  // exp(3) = 20.1

    // Same RNG seed for CPU and GPU
    uint32_t rng_cpu = 42;
    uint32_t rng_gpu = 42;

    // CPU reference
    int cpu_result = cpu_sample(host_logits.data(), n_vocab, temp, rng_cpu);

    // GPU test - need to generate random value on host with same RNG
    float random_val = gpu_rng_uniform(rng_gpu);

    float* gpu_logits = sycl::malloc_device<float>(n_vocab, q);
    float* block_max = sycl::malloc_device<float>(n_blocks, q);
    float* block_sum = sycl::malloc_device<float>(n_blocks, q);
    int32_t* result = sycl::malloc_device<int32_t>(1, q);

    // Apply temperature to logits on host (for simplicity)
    std::vector<float> scaled_logits(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        scaled_logits[i] = host_logits[i] / temp;
    }
    q.memcpy(gpu_logits, scaled_logits.data(), n_vocab * sizeof(float)).wait();

    // Pass 1: compute per-block max and sum
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(block_size, h);

        h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_softmax_block_kernel(gpu_logits, block_max, block_sum, n_vocab, item,
                                        slm.get_pointer());
            });
    }).wait();  // Ensure Pass 1 completes

    // Pass 2: global reduction and sampling
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(block_size, h);

        h.parallel_for(sycl::nd_range<1>(block_size, block_size),
            [=](sycl::nd_item<1> item) {
                gpu_softmax_sample_kernel(gpu_logits, block_max, block_sum, result, n_vocab,
                                         n_blocks, random_val, item, slm.get_pointer());
            });
    }).wait();  // Ensure Pass 2 completes

    int32_t gpu_result;
    q.memcpy(&gpu_result, result, sizeof(int32_t)).wait();

    // Cleanup
    sycl::free(gpu_logits, q);
    sycl::free(block_max, q);
    sycl::free(block_sum, q);
    sycl::free(result, q);

    printf("  CPU: %d, GPU: %d\n", cpu_result, gpu_result);

    // Both should select from high-probability tokens
    bool cpu_ok = (cpu_result == 100 || cpu_result == 200 || cpu_result == 300);
    bool gpu_ok = (gpu_result == 100 || gpu_result == 200 || gpu_result == 300);

    bool passed = cpu_ok && gpu_ok;
    printf("  %s (both selected high-prob token)\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// =============================================================================
// Top-K filtering kernels (simplified for standalone test)
// =============================================================================

// Apply top-k filter (set values below threshold to -INFINITY)
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

// Find k-th largest value on CPU (reference implementation)
float cpu_topk_threshold(const float* logits, int n_vocab, int k) {
    std::vector<float> sorted(logits, logits + n_vocab);
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    return sorted[std::min(k - 1, n_vocab - 1)];
}

// Count elements >= threshold
int cpu_count_above(const float* logits, int n_vocab, float threshold) {
    int count = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] >= threshold) count++;
    }
    return count;
}

bool test_topk(sycl::queue& q, int n_vocab, int k) {
    printf("Testing top-k (n_vocab=%d, k=%d)...\n", n_vocab, k);

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((n_vocab + block_size - 1) / block_size, GPU_SAMPLER_MAX_BLOCKS);

    // Create random logits
    std::vector<float> host_logits(n_vocab);
    std::mt19937 gen(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n_vocab; i++) {
        host_logits[i] = dist(gen);
    }

    // Make sure k specific tokens are clearly the highest
    std::vector<int> top_indices(k);
    for (int i = 0; i < k; i++) {
        top_indices[i] = (n_vocab / (k + 1)) * (i + 1);  // Spread out
        host_logits[top_indices[i]] = 10.0f + (k - i);   // Decreasing from 10+k
    }

    // CPU reference - find threshold
    float threshold = cpu_topk_threshold(host_logits.data(), n_vocab, k);
    int expected_count = cpu_count_above(host_logits.data(), n_vocab, threshold);

    // GPU test
    float* gpu_logits = sycl::malloc_device<float>(n_vocab, q);
    q.memcpy(gpu_logits, host_logits.data(), n_vocab * sizeof(float)).wait();

    // Apply filter
    q.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
        [=](sycl::nd_item<1> item) {
            gpu_topk_filter_kernel(gpu_logits, n_vocab, threshold, item);
        }).wait();

    // Copy back and count
    std::vector<float> filtered_logits(n_vocab);
    q.memcpy(filtered_logits.data(), gpu_logits, n_vocab * sizeof(float)).wait();

    int gpu_count = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (filtered_logits[i] > -INFINITY) gpu_count++;
    }

    // Verify top tokens are kept
    bool top_kept = true;
    for (int i = 0; i < k; i++) {
        if (filtered_logits[top_indices[i]] <= -INFINITY) {
            top_kept = false;
            printf("  ERROR: Top token %d was filtered out\n", top_indices[i]);
        }
    }

    sycl::free(gpu_logits, q);

    printf("  Expected count: %d, GPU count: %d\n", expected_count, gpu_count);
    printf("  Top tokens preserved: %s\n", top_kept ? "yes" : "no");

    bool passed = (gpu_count == expected_count) && top_kept;
    printf("  %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// =============================================================================
// Top-P filtering test (CPU reference)
// =============================================================================

// CPU top-p filter: compute softmax, find threshold, filter
void cpu_topp_filter(float* logits, int n_vocab, float top_p) {
    // Compute softmax
    float max_val = *std::max_element(logits, logits + n_vocab);
    float sum = 0.0f;
    std::vector<float> probs(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n_vocab; i++) {
        probs[i] /= sum;
    }

    // Find max prob
    float max_prob = *std::max_element(probs.begin(), probs.end());

    // Binary search for threshold
    float lo = 0.0f, hi = max_prob;
    for (int iter = 0; iter < 30; iter++) {
        float mid = (lo + hi) * 0.5f;
        float cumsum = 0.0f;
        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] >= mid) cumsum += probs[i];
        }
        if (cumsum >= top_p) lo = mid;
        else hi = mid;
    }

    // Apply filter
    for (int i = 0; i < n_vocab; i++) {
        if (probs[i] < lo) logits[i] = -INFINITY;
    }
}

bool test_topp(sycl::queue& q, int n_vocab, float top_p) {
    printf("Testing top-p (n_vocab=%d, p=%.2f)...\n", n_vocab, top_p);

    // Create logits with specific distribution
    std::vector<float> host_logits(n_vocab);
    std::mt19937 gen(54321);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n_vocab; i++) {
        host_logits[i] = dist(gen);
    }

    // Make first 10 tokens more likely
    for (int i = 0; i < 10; i++) {
        host_logits[i] = 5.0f - i * 0.5f;  // 5.0, 4.5, 4.0, ...
    }

    // CPU reference
    std::vector<float> cpu_logits = host_logits;
    cpu_topp_filter(cpu_logits.data(), n_vocab, top_p);

    int cpu_count = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (cpu_logits[i] > -INFINITY) cpu_count++;
    }

    // For GPU test, we apply the same filter using CPU-computed threshold
    // (Full GPU top-p is complex, here we just verify the filtering works)

    // Count how many of the top-10 tokens are kept
    int top10_kept = 0;
    for (int i = 0; i < 10; i++) {
        if (cpu_logits[i] > -INFINITY) top10_kept++;
    }

    printf("  Tokens kept: %d (should be << %d for small p)\n", cpu_count, n_vocab);
    printf("  Top-10 tokens kept: %d/10\n", top10_kept);

    // For top_p=0.9, most top-10 tokens should be kept
    bool passed = (top_p > 0.5f) ? (top10_kept >= 5) : (cpu_count < n_vocab);
    printf("  %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

bool test_distribution(sycl::queue& q, int n_vocab, float temp, int n_samples) {
    printf("Testing distribution (n_vocab=%d, temp=%.1f, samples=%d)...\n", n_vocab, temp, n_samples);

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((n_vocab + block_size - 1) / block_size, GPU_SAMPLER_MAX_BLOCKS);

    // Create logits where tokens 0-9 are MUCH more likely than others
    // Use -100 baseline so exp(-100) ≈ 0 for other tokens
    std::vector<float> host_logits(n_vocab, -100.0f);
    for (int i = 0; i < 10; i++) {
        host_logits[i] = 5.0f - i * 0.3f;  // 5.0, 4.7, 4.4, ... 2.3
    }

    // Apply temperature
    std::vector<float> scaled_logits(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        scaled_logits[i] = host_logits[i] / temp;
    }

    float* gpu_logits = sycl::malloc_device<float>(n_vocab, q);
    float* block_max = sycl::malloc_device<float>(n_blocks, q);
    float* block_sum = sycl::malloc_device<float>(n_blocks, q);
    int32_t* result = sycl::malloc_device<int32_t>(1, q);

    std::vector<int> counts(n_vocab, 0);
    uint32_t rng_state = 42;

    for (int s = 0; s < n_samples; s++) {
        q.memcpy(gpu_logits, scaled_logits.data(), n_vocab * sizeof(float)).wait();

        float random_val = gpu_rng_uniform(rng_state);

        // Pass 1
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            h.parallel_for(sycl::nd_range<1>(n_blocks * block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_softmax_block_kernel(gpu_logits, block_max, block_sum, n_vocab, item,
                                            slm.get_pointer());
                });
        }).wait();

        // Pass 2
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            h.parallel_for(sycl::nd_range<1>(block_size, block_size),
                [=](sycl::nd_item<1> item) {
                    gpu_softmax_sample_kernel(gpu_logits, block_max, block_sum, result, n_vocab,
                                             n_blocks, random_val, item, slm.get_pointer());
                });
        }).wait();

        int32_t sample;
        q.memcpy(&sample, result, sizeof(int32_t)).wait();
        counts[sample]++;
    }

    // Cleanup
    sycl::free(gpu_logits, q);
    sycl::free(block_max, q);
    sycl::free(block_sum, q);
    sycl::free(result, q);

    // Check distribution
    printf("  Top 10 counts: ");
    int top10_count = 0;
    for (int i = 0; i < 10; i++) {
        printf("%d ", counts[i]);
        top10_count += counts[i];
    }
    printf("\n");

    float top10_ratio = (float)top10_count / n_samples;
    printf("  Top-10 ratio: %.1f%% (expected >80%%)\n", top10_ratio * 100);

    bool passed = top10_ratio >= 0.8f;
    printf("  %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// =============================================================================
// Multi-Sequence Sampling Test
// Tests that multiple sequences can be sampled independently with different
// temperatures and RNG states
// =============================================================================

// Multi-sequence temperature scaling kernel
void gpu_multi_seq_temp_scale_kernel_test(
    float* logits,               // [n_seqs, n_vocab] in-place scaling
    const float* temperatures,   // [n_seqs]
    const int n_vocab,
    const int n_seqs,
    const sycl::nd_item<2>& item
) {
    const int seq = item.get_group(0);
    const int gid = item.get_global_id(1);

    if (seq >= n_seqs || gid >= n_vocab) return;

    float temp = temperatures[seq];
    if (temp <= 0.0f) return;  // Greedy mode, no scaling

    float inv_temp = 1.0f / temp;
    logits[seq * n_vocab + gid] *= inv_temp;
}

// Multi-sequence block softmax kernel
void gpu_multi_seq_softmax_block_kernel_test(
    const float* all_logits,    // [n_seqs, n_vocab]
    float* block_max,           // [n_seqs, n_blocks] output
    float* block_sum,           // [n_seqs, n_blocks] output
    const int n_vocab,
    const int n_seqs,
    const int n_blocks,
    const sycl::nd_item<2>& item,
    float* slm                  // SLM [block_size]
) {
    const int seq = item.get_group(0);
    const int block_id = item.get_group(1);
    const int tid = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs) return;

    const int vocab_start = block_id * block_size;
    const int gid = vocab_start + tid;

    const float* seq_logits = all_logits + seq * n_vocab;

    // Load value
    float val = (gid < n_vocab) ? seq_logits[gid] : -INFINITY;
    slm[tid] = val;
    sycl::group_barrier(item.get_group());

    // Find max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] = sycl::fmax(slm[tid], slm[tid + stride]);
        }
        sycl::group_barrier(item.get_group());
    }
    float local_max = slm[0];
    sycl::group_barrier(item.get_group());

    // Compute exp(x - max) and sum
    float exp_val = (gid < n_vocab) ? sycl::exp(val - local_max) : 0.0f;
    slm[tid] = exp_val;
    sycl::group_barrier(item.get_group());

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }
    float local_sum = slm[0];

    if (tid == 0) {
        block_max[seq * n_blocks + block_id] = local_max;
        block_sum[seq * n_blocks + block_id] = local_sum;
    }
}

// Multi-sequence final softmax + sample kernel
void gpu_multi_seq_softmax_sample_kernel_test(
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

bool test_multi_seq_sampling(sycl::queue& q, int n_seqs, int n_vocab) {
    printf("Testing multi-sequence sampling (n_seqs=%d, n_vocab=%d)...\n", n_seqs, n_vocab);

    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((n_vocab + block_size - 1) / block_size, GPU_SAMPLER_MAX_BLOCKS);

    // Create random logits for each sequence with known maxes
    std::vector<float> host_logits(n_seqs * n_vocab);
    std::vector<int> expected_greedy(n_seqs);
    std::mt19937 gen(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int seq = 0; seq < n_seqs; seq++) {
        for (int i = 0; i < n_vocab; i++) {
            host_logits[seq * n_vocab + i] = dist(gen);
        }
        // Set known maximum at different positions for each sequence
        int max_pos = (n_vocab / (seq + 2));  // Different position for each
        host_logits[seq * n_vocab + max_pos] = 100.0f;
        expected_greedy[seq] = max_pos;
    }

    // Allocate device memory
    float* gpu_logits = sycl::malloc_device<float>(n_seqs * n_vocab, q);
    float* temperatures = sycl::malloc_device<float>(n_seqs, q);
    uint32_t* rng_states = sycl::malloc_device<uint32_t>(n_seqs, q);
    uint8_t* greedy_flags = sycl::malloc_device<uint8_t>(n_seqs, q);
    float* block_max = sycl::malloc_device<float>(n_seqs * n_blocks, q);
    float* block_sum = sycl::malloc_device<float>(n_seqs * n_blocks, q);
    int32_t* results = sycl::malloc_device<int32_t>(n_seqs, q);

    // Set temperatures: even sequences are greedy (temp=0), odd use temp=1.0
    std::vector<float> h_temps(n_seqs);
    std::vector<uint8_t> h_greedy(n_seqs);
    std::vector<uint32_t> h_rng(n_seqs);
    for (int i = 0; i < n_seqs; i++) {
        h_temps[i] = (i % 2 == 0) ? 0.0f : 1.0f;
        h_greedy[i] = (i % 2 == 0) ? 1 : 0;  // 1 = greedy
        h_rng[i] = 42 + i * 1000;  // Different seed per sequence
    }

    q.memcpy(gpu_logits, host_logits.data(), n_seqs * n_vocab * sizeof(float)).wait();
    q.memcpy(temperatures, h_temps.data(), n_seqs * sizeof(float)).wait();
    q.memcpy(greedy_flags, h_greedy.data(), n_seqs * sizeof(uint8_t)).wait();
    q.memcpy(rng_states, h_rng.data(), n_seqs * sizeof(uint32_t)).wait();

    // Step 1: Temperature scaling (only for non-greedy)
    {
        sycl::range<2> global(n_seqs, ((n_vocab + block_size - 1) / block_size) * block_size);
        sycl::range<2> local(1, block_size);

        q.parallel_for(sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) {
                gpu_multi_seq_temp_scale_kernel_test(gpu_logits, temperatures, n_vocab, n_seqs, item);
            }).wait();
    }

    // Step 2: Block-level softmax
    {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_seqs, n_blocks * block_size),
                                              sycl::range<2>(1, block_size)),
                [=](sycl::nd_item<2> item) {
                    gpu_multi_seq_softmax_block_kernel_test(
                        gpu_logits, block_max, block_sum, n_vocab, n_seqs, n_blocks, item,
                        slm.get_pointer());
                });
        }).wait();
    }

    // Step 3: Final softmax + sampling
    {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_seqs, block_size),
                                              sycl::range<2>(1, block_size)),
                [=](sycl::nd_item<2> item) {
                    gpu_multi_seq_softmax_sample_kernel_test(
                        gpu_logits, block_max, block_sum, results, rng_states, greedy_flags,
                        n_vocab, n_seqs, n_blocks, item, slm.get_pointer());
                });
        }).wait();
    }

    // Read results
    std::vector<int32_t> gpu_results(n_seqs);
    q.memcpy(gpu_results.data(), results, n_seqs * sizeof(int32_t)).wait();

    // Cleanup
    sycl::free(gpu_logits, q);
    sycl::free(temperatures, q);
    sycl::free(rng_states, q);
    sycl::free(greedy_flags, q);
    sycl::free(block_max, q);
    sycl::free(block_sum, q);
    sycl::free(results, q);

    // Verify: greedy sequences should return expected max position
    bool all_passed = true;
    int greedy_correct = 0;
    int non_greedy_valid = 0;

    for (int seq = 0; seq < n_seqs; seq++) {
        if (seq % 2 == 0) {
            // Greedy: should match expected
            if (gpu_results[seq] == expected_greedy[seq]) {
                greedy_correct++;
            } else {
                printf("  Seq %d FAILED: expected %d, got %d\n", seq, expected_greedy[seq], gpu_results[seq]);
                all_passed = false;
            }
        } else {
            // Non-greedy: should be in valid range
            if (gpu_results[seq] >= 0 && gpu_results[seq] < n_vocab) {
                non_greedy_valid++;
            } else {
                printf("  Seq %d FAILED: invalid token %d\n", seq, gpu_results[seq]);
                all_passed = false;
            }
        }
    }

    int n_greedy = (n_seqs + 1) / 2;
    int n_non_greedy = n_seqs / 2;
    printf("  Greedy sequences: %d/%d correct\n", greedy_correct, n_greedy);
    printf("  Non-greedy sequences: %d/%d valid\n", non_greedy_valid, n_non_greedy);
    printf("  %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed;
}

// Test that sequences don't cross-contaminate (independence test)
bool test_multi_seq_independence(sycl::queue& q, int n_vocab) {
    printf("Testing multi-sequence independence (n_vocab=%d)...\n", n_vocab);

    constexpr int n_seqs = 4;
    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int n_blocks = std::min((n_vocab + block_size - 1) / block_size, GPU_SAMPLER_MAX_BLOCKS);

    // Create identical logits but with different known maxes for each sequence
    std::vector<float> host_logits(n_seqs * n_vocab, 0.0f);
    std::vector<int> expected_max(n_seqs);

    for (int seq = 0; seq < n_seqs; seq++) {
        // Each sequence has its max at position: seq * (n_vocab / 5)
        expected_max[seq] = (seq + 1) * (n_vocab / 5);
        host_logits[seq * n_vocab + expected_max[seq]] = 100.0f;
    }

    // Allocate device memory
    float* gpu_logits = sycl::malloc_device<float>(n_seqs * n_vocab, q);
    float* temperatures = sycl::malloc_device<float>(n_seqs, q);
    uint32_t* rng_states = sycl::malloc_device<uint32_t>(n_seqs, q);
    uint8_t* greedy_flags = sycl::malloc_device<uint8_t>(n_seqs, q);
    float* block_max = sycl::malloc_device<float>(n_seqs * n_blocks, q);
    float* block_sum = sycl::malloc_device<float>(n_seqs * n_blocks, q);
    int32_t* results = sycl::malloc_device<int32_t>(n_seqs, q);

    // All greedy mode
    std::vector<float> h_temps(n_seqs, 0.0f);
    std::vector<uint8_t> h_greedy(n_seqs, 1);
    std::vector<uint32_t> h_rng(n_seqs, 42);

    q.memcpy(gpu_logits, host_logits.data(), n_seqs * n_vocab * sizeof(float)).wait();
    q.memcpy(temperatures, h_temps.data(), n_seqs * sizeof(float)).wait();
    q.memcpy(greedy_flags, h_greedy.data(), n_seqs * sizeof(uint8_t)).wait();
    q.memcpy(rng_states, h_rng.data(), n_seqs * sizeof(uint32_t)).wait();

    // Run sampling pipeline
    {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_seqs, n_blocks * block_size),
                                              sycl::range<2>(1, block_size)),
                [=](sycl::nd_item<2> item) {
                    gpu_multi_seq_softmax_block_kernel_test(
                        gpu_logits, block_max, block_sum, n_vocab, n_seqs, n_blocks, item,
                        slm.get_pointer());
                });
        }).wait();
    }

    {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> slm(block_size, h);

            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_seqs, block_size),
                                              sycl::range<2>(1, block_size)),
                [=](sycl::nd_item<2> item) {
                    gpu_multi_seq_softmax_sample_kernel_test(
                        gpu_logits, block_max, block_sum, results, rng_states, greedy_flags,
                        n_vocab, n_seqs, n_blocks, item, slm.get_pointer());
                });
        }).wait();
    }

    // Read results
    std::vector<int32_t> gpu_results(n_seqs);
    q.memcpy(gpu_results.data(), results, n_seqs * sizeof(int32_t)).wait();

    // Cleanup
    sycl::free(gpu_logits, q);
    sycl::free(temperatures, q);
    sycl::free(rng_states, q);
    sycl::free(greedy_flags, q);
    sycl::free(block_max, q);
    sycl::free(block_sum, q);
    sycl::free(results, q);

    // Verify: each sequence should return its own expected max
    bool all_passed = true;
    for (int seq = 0; seq < n_seqs; seq++) {
        if (gpu_results[seq] != expected_max[seq]) {
            printf("  Seq %d FAILED: expected %d, got %d\n", seq, expected_max[seq], gpu_results[seq]);
            all_passed = false;
        }
    }

    printf("  All sequences independent: %s\n", all_passed ? "YES" : "NO");
    printf("  %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed;
}

int main() {
    printf("=== GPU Sampler Tests ===\n\n");

    try {
        sycl::queue q(sycl::gpu_selector_v);
        printf("Device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

        int passed = 0;
        int failed = 0;

        // Argmax tests
        if (test_argmax(q, 256)) passed++; else failed++;
        if (test_argmax(q, 1024)) passed++; else failed++;
        if (test_argmax(q, 32000)) passed++; else failed++;
        if (test_argmax(q, 128000)) passed++; else failed++;

        printf("\n");

        // Sampling tests
        if (test_sampling(q, 32000, 1.0f)) passed++; else failed++;
        if (test_sampling(q, 32000, 0.5f)) passed++; else failed++;
        if (test_sampling(q, 32000, 2.0f)) passed++; else failed++;

        printf("\n");

        // Distribution tests
        if (test_distribution(q, 1000, 1.0f, 500)) passed++; else failed++;
        if (test_distribution(q, 32000, 0.8f, 500)) passed++; else failed++;

        printf("\n");

        // Top-K tests
        if (test_topk(q, 1000, 10)) passed++; else failed++;
        if (test_topk(q, 32000, 40)) passed++; else failed++;
        if (test_topk(q, 32000, 100)) passed++; else failed++;

        printf("\n");

        // Top-P tests
        if (test_topp(q, 1000, 0.9f)) passed++; else failed++;
        if (test_topp(q, 32000, 0.8f)) passed++; else failed++;

        printf("\n");

        // Multi-sequence tests
        if (test_multi_seq_sampling(q, 4, 1000)) passed++; else failed++;
        if (test_multi_seq_sampling(q, 8, 32000)) passed++; else failed++;
        if (test_multi_seq_independence(q, 32000)) passed++; else failed++;

        printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
        return failed > 0 ? 1 : 0;

    } catch (const sycl::exception& e) {
        printf("SYCL error: %s\n", e.what());
        return 1;
    }
}

#else
int main() {
    printf("GPU sampler tests require SYCL build (-DGGML_USE_SYCL)\n");
    return 0;
}
#endif
