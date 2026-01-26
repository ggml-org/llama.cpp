#pragma once
#include <vector>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace ggml_sycl {

// Key for crossover lookup
struct CrossoverKey {
    int64_t N;           // Matrix N dimension
    int64_t K;           // Matrix K dimension
    int quant_type;      // Quantization type

    bool operator==(const CrossoverKey& other) const {
        return N == other.N && K == other.K && quant_type == other.quant_type;
    }
};

struct CrossoverKeyHash {
    size_t operator()(const CrossoverKey& k) const {
        size_t h = 17;
        h = h * 31 + std::hash<int64_t>{}(k.N);
        h = h * 31 + std::hash<int64_t>{}(k.K);
        h = h * 31 + std::hash<int>{}(k.quant_type);
        return h;
    }
};

// Crossover discovery result
struct CrossoverResult {
    int64_t crossover_batch;  // Batch size where oneDNN wins
    float unified_ms;         // Unified kernel time at crossover
    float onednn_ms;          // oneDNN time at crossover
    float confidence;         // How confident is this crossover
    int samples_taken;
};

// Benchmark function types
using BenchmarkFunc = std::function<float(int64_t M, int64_t N, int64_t K)>;

class CrossoverDiscovery {
public:
    // Test batch sizes for crossover search
    static constexpr int64_t TEST_BATCHES[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    static constexpr int NUM_TEST_BATCHES = 11;

    CrossoverDiscovery() = default;

    // Find crossover point using binary search
    CrossoverResult find_crossover(
        int64_t N, int64_t K, int /*quant_type*/,
        BenchmarkFunc unified_bench,
        BenchmarkFunc onednn_bench,
        int warmup_iters = 3,
        int bench_iters = 5)
    {
        CrossoverResult result;
        result.crossover_batch = -1;
        result.samples_taken = 0;
        result.unified_ms = 0.0f;
        result.onednn_ms = 0.0f;
        result.confidence = 0.0f;

        // Binary search for crossover point
        int lo = 0, hi = NUM_TEST_BATCHES - 1;

        while (lo < hi) {
            int mid = (lo + hi) / 2;
            int64_t M = TEST_BATCHES[mid];

            // Warmup
            for (int i = 0; i < warmup_iters; i++) {
                unified_bench(M, N, K);
                onednn_bench(M, N, K);
            }

            // Benchmark
            float unified_sum = 0, onednn_sum = 0;
            for (int i = 0; i < bench_iters; i++) {
                unified_sum += unified_bench(M, N, K);
                onednn_sum += onednn_bench(M, N, K);
            }

            float unified_ms = unified_sum / bench_iters;
            float onednn_ms = onednn_sum / bench_iters;
            result.samples_taken += 2 * (warmup_iters + bench_iters);

            if (onednn_ms < unified_ms) {
                hi = mid;  // oneDNN faster, crossover is lower
                result.crossover_batch = M;
                result.unified_ms = unified_ms;
                result.onednn_ms = onednn_ms;
            } else {
                lo = mid + 1;  // Unified faster, crossover is higher
            }
        }

        // No crossover found - unified always better or oneDNN always better
        if (result.crossover_batch < 0) {
            result.crossover_batch = TEST_BATCHES[lo];
        }

        // Confidence based on difference at crossover
        if (result.unified_ms > 0 && result.onednn_ms > 0) {
            float diff = std::abs(result.unified_ms - result.onednn_ms);
            float avg = (result.unified_ms + result.onednn_ms) / 2;
            result.confidence = 1.0f - std::min(1.0f, diff / avg);
        }

        return result;
    }

    // Store discovered crossover
    void store_crossover(const CrossoverKey& key, const CrossoverResult& result) {
        crossovers_[key] = result;
    }

    // Lookup crossover (returns nullptr if not found)
    const CrossoverResult* get_crossover(int64_t N, int64_t K, int quant_type) const {
        CrossoverKey key{N, K, quant_type};
        auto it = crossovers_.find(key);
        return it != crossovers_.end() ? &it->second : nullptr;
    }

    // Check if should use oneDNN for given batch size
    bool should_use_onednn(int64_t M, int64_t N, int64_t K, int quant_type) const {
        auto* result = get_crossover(N, K, quant_type);
        if (!result) return false;  // No data, use unified
        return M >= result->crossover_batch;
    }

    // Get number of stored crossovers
    size_t get_crossover_count() const { return crossovers_.size(); }

    // Clear all stored crossovers
    void clear() { crossovers_.clear(); }

    // Standard dimension configurations for discovery
    static std::vector<std::pair<int64_t, int64_t>> get_standard_dimensions() {
        return {
            {4096, 4096},    // Mistral 7B, LLaMA 7B
            {5120, 5120},    // LLaMA 13B
            {6144, 6144},    // LLaMA 30B
            {8192, 8192},    // LLaMA 65B
            {4096, 14336},   // Mistral 7B FFN
            {14336, 4096},   // Mistral 7B FFN transpose
        };
    }

private:
    std::unordered_map<CrossoverKey, CrossoverResult, CrossoverKeyHash> crossovers_;
};

} // namespace ggml_sycl
