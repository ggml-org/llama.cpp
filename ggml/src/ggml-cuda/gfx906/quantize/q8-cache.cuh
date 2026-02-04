#pragma once

// Q8_1 activation cache for GFX906 - reuses quantized data across MUL_MAT ops

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <cstdint>

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#define Q8_CACHE_CHECK(err) do { if ((err) != hipSuccess) { fprintf(stderr, "HIP error: %s\n", hipGetErrorString(err)); abort(); } } while(0)
#define Q8_CACHE_MALLOC(ptr, size) Q8_CACHE_CHECK(hipMalloc(ptr, size))
#define Q8_CACHE_FREE(ptr) Q8_CACHE_CHECK(hipFree(ptr))
#else
#include <cuda_runtime.h>
#define Q8_CACHE_CHECK(err) do { if ((err) != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); abort(); } } while(0)
#define Q8_CACHE_MALLOC(ptr, size) Q8_CACHE_CHECK(cudaMalloc(ptr, size))
#define Q8_CACHE_FREE(ptr) Q8_CACHE_CHECK(cudaFree(ptr))
#endif

struct ggml_tensor;

struct q8_cache_key {
    const ggml_tensor* src_tensor;
    int layout;
    bool operator==(const q8_cache_key& other) const {
        return src_tensor == other.src_tensor && layout == other.layout;
    }
};

struct q8_cache_key_hash {
    size_t operator()(const q8_cache_key& k) const {
        return std::hash<const void*>{}(k.src_tensor) ^ (std::hash<int>{}(k.layout) << 1);
    }
};

struct q8_cache_entry {
    void* q8_data = nullptr;
    size_t buffer_size = 0;
    int64_t ne10_padded = 0;
    int64_t ne11 = 0;
    int64_t ne12 = 0;
    int64_t ne13 = 0;
};

// Epoch-based buffer pool: buffers reusable after SAFE_EPOCH_DELAY graphs
struct q8_hashmap_cache {
    std::unordered_map<q8_cache_key, q8_cache_entry, q8_cache_key_hash> entries;

    struct buffer_slot {
        void* ptr = nullptr;
        size_t size = 0;
        uint64_t written_epoch = 0;
    };
    std::vector<buffer_slot> buffer_pool;

    uint64_t current_epoch = 0;
    static constexpr uint64_t SAFE_EPOCH_DELAY = 2;

    void clear() {
        entries.clear();
        current_epoch++;
    }

    void* get_buffer(size_t size) {
        for (auto& slot : buffer_pool) {
            if (slot.size >= size && current_epoch - slot.written_epoch >= SAFE_EPOCH_DELAY) {
                slot.written_epoch = current_epoch;
                return slot.ptr;
            }
        }
        void* ptr = nullptr;
        Q8_CACHE_MALLOC(&ptr, size);
        buffer_pool.push_back({ptr, size, current_epoch});
        return ptr;
    }

    const q8_cache_entry* lookup(const ggml_tensor* tensor, int layout,
                                  int64_t ne10p, int64_t ne11, int64_t ne12, int64_t ne13) {
        auto it = entries.find({tensor, layout});
        if (it != entries.end()) {
            const auto& e = it->second;
            if (e.ne10_padded == ne10p && e.ne11 == ne11 && e.ne12 == ne12 && e.ne13 == ne13) {
                return &e;
            }
        }
        return nullptr;
    }

    void store(const ggml_tensor* tensor, int layout, void* data, size_t size,
               int64_t ne10p, int64_t ne11, int64_t ne12, int64_t ne13) {
        entries[{tensor, layout}] = {data, size, ne10p, ne11, ne12, ne13};
    }

    void free_all() {
        for (auto& slot : buffer_pool) {
            if (slot.ptr) Q8_CACHE_FREE(slot.ptr);
        }
        buffer_pool.clear();
        entries.clear();
        current_epoch = 0;
    }

    size_t size() const { return entries.size(); }
};

// Multi-consumer fusion info
struct prequantized_q8_info {
    char* buffer_ptr;
    int64_t ne10, ne11, ne12, ne13;
};
