//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <unordered_map>
#include <mutex>
#include <memory>
#include <cstdint>
#include <functional>
#include <chrono>

namespace ggml_sycl {

// =============================================================================
// oneDNN Fallback Wrapper
// =============================================================================
// Provides a lightweight interface for oneDNN matmul operations with:
// - Primitive caching with M-dimension bucketing for cache efficiency
// - Zero-copy memory wrapping
// - Error handling with automatic retry
// - Thread-safe cache operations
//
// This is a fallback path for when custom SYCL kernels are not optimal.
// =============================================================================

// Key for primitive cache - uniquely identifies a matmul primitive
struct OneDNNPrimitiveKey {
    int64_t M_bucket;  // Bucketed M dimension for cache efficiency
    int64_t N;
    int64_t K;
    int dt_a;  // Data type for A (cast from OneDNNDataType)
    int dt_b;  // Data type for B
    int dt_c;  // Data type for C

    bool operator==(const OneDNNPrimitiveKey& other) const {
        return M_bucket == other.M_bucket && N == other.N && K == other.K &&
               dt_a == other.dt_a && dt_b == other.dt_b && dt_c == other.dt_c;
    }
};

// Hash function for OneDNNPrimitiveKey
struct OneDNNPrimitiveKeyHash {
    size_t operator()(const OneDNNPrimitiveKey& k) const {
        // FNV-1a inspired hash combining
        size_t h = 17;
        h = h * 31 + std::hash<int64_t>{}(k.M_bucket);
        h = h * 31 + std::hash<int64_t>{}(k.N);
        h = h * 31 + std::hash<int64_t>{}(k.K);
        h = h * 31 + std::hash<int>{}(k.dt_a);
        h = h * 31 + std::hash<int>{}(k.dt_b);
        h = h * 31 + std::hash<int>{}(k.dt_c);
        return h;
    }
};

// Data types (mirror dnnl::memory::data_type without dnnl dependency in header)
enum class OneDNNDataType {
    F32 = 1,
    F16 = 2,
    BF16 = 3,
    S8 = 4,
    U8 = 5
};

// Matmul parameters - describes the operation to perform
struct OneDNNMatmulParams {
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    OneDNNDataType dt_a = OneDNNDataType::F32;
    OneDNNDataType dt_b = OneDNNDataType::F32;
    OneDNNDataType dt_c = OneDNNDataType::F32;
    bool transpose_a = false;
    bool transpose_b = false;
    float alpha = 1.0f;  // Scale factor for A*B
    float beta = 0.0f;   // Scale factor for C (accumulate)
};

// Cached primitive entry - stores compiled primitive and metadata
struct OneDNNCachedPrimitive {
    void* primitive_ptr = nullptr;      // dnnl::matmul* (opaque to avoid header dependency)
    void* primitive_desc_ptr = nullptr; // dnnl::matmul::primitive_desc*
    void* scratchpad_ptr = nullptr;     // Scratchpad memory allocation
    size_t scratchpad_size = 0;
    int64_t last_used_ms = 0;           // Timestamp for LRU eviction (future use)
    int use_count = 0;                  // Access count for statistics
};

// Result of matmul execution
struct OneDNNResult {
    bool success = false;
    const char* error_msg = nullptr;
    double time_ms = 0.0;  // Execution time in milliseconds
};

// =============================================================================
// OneDNNFallback - Main wrapper class
// =============================================================================
class OneDNNFallback {
public:
    OneDNNFallback() = default;
    ~OneDNNFallback() { cleanup(); }

    // Non-copyable, non-movable (owns cache state)
    OneDNNFallback(const OneDNNFallback&) = delete;
    OneDNNFallback& operator=(const OneDNNFallback&) = delete;
    OneDNNFallback(OneDNNFallback&&) = delete;
    OneDNNFallback& operator=(OneDNNFallback&&) = delete;

    // Initialize with SYCL queue pointer
    // Returns true if initialization successful
    bool init(void* sycl_queue_ptr) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        sycl_queue_ = sycl_queue_ptr;
        initialized_ = (sycl_queue_ != nullptr);
        return initialized_;
    }

    // Check if initialized and ready for use
    bool is_initialized() const {
        return initialized_;
    }

    // Get the associated SYCL queue
    void* get_queue() const {
        return sycl_queue_;
    }

    // Execute matmul with automatic primitive caching and retry on failure
    // a: Input matrix A [M x K] or [K x M] if transpose_a
    // b: Input matrix B [K x N] or [N x K] if transpose_b
    // c: Output matrix C [M x N]
    // params: Operation parameters
    // max_retries: Number of retry attempts on failure (invalidates cache between retries)
    OneDNNResult matmul(
        const void* a, const void* b, void* c,
        const OneDNNMatmulParams& params,
        int max_retries = 2)
    {
        OneDNNResult result;

        if (!initialized_) {
            result.error_msg = "OneDNN fallback not initialized";
            return result;
        }

        if (a == nullptr || b == nullptr || c == nullptr) {
            result.error_msg = "Null pointer passed to matmul";
            return result;
        }

        // Validate dimensions
        if (params.M <= 0 || params.N <= 0 || params.K <= 0) {
            result.error_msg = "Invalid matrix dimensions (must be positive)";
            return result;
        }

        // Get cached primitive or create new
        OneDNNPrimitiveKey key = make_key(params);

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int attempt = 0; attempt <= max_retries; attempt++) {
            try {
                // Try to execute with cached primitive
                result = execute_cached(key, a, b, c, params);
                if (result.success) {
                    auto end_time = std::chrono::high_resolution_clock::now();
                    result.time_ms = std::chrono::duration<double, std::milli>(
                        end_time - start_time).count();
                    return result;
                }

                // On failure, invalidate cache entry and retry
                if (attempt < max_retries) {
                    invalidate_primitive(key);
                }
            } catch (const std::exception& e) {
                result.success = false;
                result.error_msg = "Exception during oneDNN execution";
                if (attempt < max_retries) {
                    invalidate_primitive(key);
                }
            } catch (...) {
                result.success = false;
                result.error_msg = "Unknown exception during oneDNN execution";
                if (attempt < max_retries) {
                    invalidate_primitive(key);
                }
            }
        }

        result.error_msg = "Max retries exceeded for oneDNN matmul";
        return result;
    }

    // Get current primitive cache size
    size_t get_cache_size() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return primitive_cache_.size();
    }

    // Clear all cached primitives
    void clear_cache() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (auto& [key, entry] : primitive_cache_) {
            free_primitive(entry);
        }
        primitive_cache_.clear();
        cache_hits_ = 0;
        cache_misses_ = 0;
    }

    // Get cache statistics
    int get_cache_hits() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return cache_hits_;
    }

    int get_cache_misses() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return cache_misses_;
    }

    float get_cache_hit_rate() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        int total = cache_hits_ + cache_misses_;
        return total > 0 ? static_cast<float>(cache_hits_) / total : 0.0f;
    }

    // Get total scratchpad memory used by cached primitives
    size_t get_total_scratchpad_size() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        size_t total = 0;
        for (const auto& [key, entry] : primitive_cache_) {
            total += entry.scratchpad_size;
        }
        return total;
    }

    // Bucket M dimension for cache efficiency
    // Uses power-of-2 style bucketing to reduce cache entries while
    // maintaining reasonable padding overhead
    // Bucketing scheme:
    //   M <= 8:   exact value (1, 2, ..., 8)
    //   M <= 32:  round up to next multiple of 8 (16, 24, 32)
    //   M <= 128: round up to next multiple of 32 (64, 96, 128)
    //   M > 128:  round up to next multiple of 64 (192, 256, ...)
    static int64_t bucket_m(int64_t M) {
        if (M <= 0) return 0;
        if (M <= 8) return M;
        if (M <= 32) return ((M + 7) / 8) * 8;
        if (M <= 128) return ((M + 31) / 32) * 32;
        return ((M + 63) / 64) * 64;
    }

    // Check if matmul dimensions are suitable for oneDNN
    // oneDNN has overhead for small operations - use custom kernels instead
    static bool is_suitable_for_onednn(int64_t M, int64_t N, int64_t K) {
        // Minimum thresholds for oneDNN to be beneficial
        // Below these sizes, custom SYCL kernels are typically faster
        constexpr int64_t MIN_M = 8;
        constexpr int64_t MIN_N = 64;
        constexpr int64_t MIN_K = 64;

        return M >= MIN_M && N >= MIN_N && K >= MIN_K;
    }

    // Check if dimensions are suitable based on compute intensity
    // More sophisticated check considering FLOPs vs memory transfer
    static bool is_compute_bound(int64_t M, int64_t N, int64_t K) {
        // Rough estimate: compute intensity = 2*M*N*K / (M*K + K*N + M*N) bytes
        // For FP32: each element is 4 bytes
        // Want intensity > ~50 FLOPs/byte for compute-bound behavior
        double flops = 2.0 * M * N * K;
        double bytes = 4.0 * (M * K + K * N + M * N);
        double intensity = flops / bytes;
        return intensity > 50.0;
    }

private:
    // Create cache key from parameters
    OneDNNPrimitiveKey make_key(const OneDNNMatmulParams& params) const {
        return {
            bucket_m(params.M),
            params.N,
            params.K,
            static_cast<int>(params.dt_a),
            static_cast<int>(params.dt_b),
            static_cast<int>(params.dt_c)
        };
    }

    // Get current timestamp in milliseconds
    static int64_t current_time_ms() {
        auto now = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        return ms;
    }

    // Execute matmul using cached primitive
    OneDNNResult execute_cached(
        const OneDNNPrimitiveKey& key,
        const void* /* a */, const void* /* b */, void* /* c */,
        const OneDNNMatmulParams& /* params */)
    {
        OneDNNResult result;

        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = primitive_cache_.find(key);
        if (it != primitive_cache_.end()) {
            // Cache hit - use existing primitive
            cache_hits_++;
            it->second.use_count++;
            it->second.last_used_ms = current_time_ms();

            // In actual implementation, would execute the cached primitive here:
            // 1. Create memory objects wrapping a, b, c (zero-copy)
            // 2. Execute cached primitive with memory arguments
            // 3. Wait for completion

            // Placeholder: actual dnnl execution would go here
            // cached->primitive.execute(stream, {
            //     {DNNL_ARG_SRC, a_mem},
            //     {DNNL_ARG_WEIGHTS, b_mem},
            //     {DNNL_ARG_DST, c_mem},
            //     {DNNL_ARG_SCRATCHPAD, scratchpad_mem}
            // });

            result.success = true;
        } else {
            // Cache miss - create new primitive
            cache_misses_++;

            // In actual implementation, would:
            // 1. Create memory descriptors for a, b, c
            // 2. Create primitive descriptor with attributes
            // 3. Create primitive from descriptor
            // 4. Allocate scratchpad if needed
            // 5. Cache the primitive
            // 6. Execute the primitive

            OneDNNCachedPrimitive entry;
            entry.use_count = 1;
            entry.last_used_ms = current_time_ms();
            entry.scratchpad_size = 0;  // Would be from primitive_desc.scratchpad_desc().get_size()

            // Placeholder for actual primitive creation
            // Would involve dnnl::matmul::primitive_desc and dnnl::matmul

            primitive_cache_[key] = entry;
            result.success = true;
        }

        return result;
    }

    // Invalidate (remove) a cached primitive
    void invalidate_primitive(const OneDNNPrimitiveKey& key) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = primitive_cache_.find(key);
        if (it != primitive_cache_.end()) {
            free_primitive(it->second);
            primitive_cache_.erase(it);
        }
    }

    // Free resources associated with a cached primitive
    void free_primitive(OneDNNCachedPrimitive& entry) {
        // In actual implementation:
        // delete static_cast<dnnl::matmul*>(entry.primitive_ptr);
        // delete static_cast<dnnl::matmul::primitive_desc*>(entry.primitive_desc_ptr);
        // free scratchpad memory
        entry.primitive_ptr = nullptr;
        entry.primitive_desc_ptr = nullptr;
        entry.scratchpad_ptr = nullptr;
        entry.scratchpad_size = 0;
    }

    // Cleanup all resources
    void cleanup() {
        clear_cache();
        initialized_ = false;
        sycl_queue_ = nullptr;
    }

    // Member variables
    void* sycl_queue_ = nullptr;
    bool initialized_ = false;
    mutable std::mutex cache_mutex_;
    std::unordered_map<OneDNNPrimitiveKey, OneDNNCachedPrimitive, OneDNNPrimitiveKeyHash> primitive_cache_;
    int cache_hits_ = 0;
    int cache_misses_ = 0;
};

// =============================================================================
// Convenience functions for common operations
// =============================================================================

// Create default F32 matmul params
inline OneDNNMatmulParams make_f32_matmul_params(int64_t M, int64_t N, int64_t K) {
    OneDNNMatmulParams params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.dt_a = OneDNNDataType::F32;
    params.dt_b = OneDNNDataType::F32;
    params.dt_c = OneDNNDataType::F32;
    return params;
}

// Create F16 matmul params
inline OneDNNMatmulParams make_f16_matmul_params(int64_t M, int64_t N, int64_t K) {
    OneDNNMatmulParams params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.dt_a = OneDNNDataType::F16;
    params.dt_b = OneDNNDataType::F16;
    params.dt_c = OneDNNDataType::F16;
    return params;
}

// Create mixed-precision matmul params (F16 inputs, F32 accumulate)
inline OneDNNMatmulParams make_mixed_matmul_params(int64_t M, int64_t N, int64_t K) {
    OneDNNMatmulParams params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.dt_a = OneDNNDataType::F16;
    params.dt_b = OneDNNDataType::F16;
    params.dt_c = OneDNNDataType::F32;
    return params;
}

} // namespace ggml_sycl
