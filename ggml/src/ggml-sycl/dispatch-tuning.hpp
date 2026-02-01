#pragma once

#include "ggml.h"
#include "kernel-selection.hpp"
#include "tuning-engine.hpp"

#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace ggml_sycl {
namespace dispatch_tuning {

struct DispatchTuningKey {
    int32_t                     quant_type = 0;
    ggml_sycl_tuning::BatchBucket batch_bucket = ggml_sycl_tuning::BatchBucket::SINGLE;
    int32_t                     k_bucket = 0;
    int32_t                     n_bucket = 0;

    bool operator==(const DispatchTuningKey & other) const {
        return quant_type == other.quant_type &&
               batch_bucket == other.batch_bucket &&
               k_bucket == other.k_bucket &&
               n_bucket == other.n_bucket;
    }
};

struct DispatchTuningEntry {
    DispatchTuningKey          key{};
    ggml_sycl_mul_mat_kernel   kernel = ggml_sycl_mul_mat_kernel::MMVQ_AOS;
    int64_t                    instances = 0;
};

struct DispatchTuningKeyHasher {
    size_t operator()(const DispatchTuningKey & key) const noexcept {
        size_t h = 0;
        auto combine = [&h](size_t v) {
            h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        combine(std::hash<int32_t>{}(key.quant_type));
        combine(std::hash<uint8_t>{}(static_cast<uint8_t>(key.batch_bucket)));
        combine(std::hash<int32_t>{}(key.k_bucket));
        combine(std::hash<int32_t>{}(key.n_bucket));
        return h;
    }
};

class DispatchTuningCache {
  public:
    std::optional<DispatchTuningEntry> lookup(const DispatchTuningKey & key) const;
    void insert(const DispatchTuningEntry & entry);
    bool empty() const;
    size_t size() const;
    void clear();

  private:
    std::unordered_map<DispatchTuningKey, DispatchTuningEntry, DispatchTuningKeyHasher> cache_;
    mutable std::shared_mutex                                   mutex_;
};

// Load tuning results from a benchmark summary JSON file.
// Returns true if at least one entry was inserted.
bool load_dispatch_tuning_from_file(const std::string & path, DispatchTuningCache & cache, std::string * error = nullptr);

// Build a tuning key from raw dimensions.
DispatchTuningKey make_dispatch_tuning_key(ggml_type type, int64_t M, int64_t N, int64_t K);

// Initialize tuning cache for a model id (runs once per model).
void ensure_model_loaded(uint64_t model_id);

// Lookup tuned kernel for a specific shape (model_id scoped).
std::optional<ggml_sycl_mul_mat_kernel> lookup_kernel(uint64_t model_id,
                                                      ggml_type type,
                                                      int64_t M,
                                                      int64_t N,
                                                      int64_t K);

}  // namespace dispatch_tuning
}  // namespace ggml_sycl
