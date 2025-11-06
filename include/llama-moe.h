#pragma once

#ifdef LLAMA_MOE_ENABLE

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-graph.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef GGML_USE_CUDA
struct CUstream_st;
using cudaStream_t = CUstream_st *;
#endif

struct llm_graph_result;

enum class llama_moe_weight_kind : uint8_t {
    GATE_WEIGHT = 0,
    UP_WEIGHT   = 1,
    DOWN_WEIGHT = 2,
    GATE_BIAS   = 3,
    UP_BIAS     = 4,
    DOWN_BIAS   = 5,
    UNKNOWN     = 255,
};

struct llama_moe_expert_handle {
    int32_t id = -1;
    int32_t layer = -1;
    int32_t expert_index = -1;
    llama_moe_weight_kind kind = llama_moe_weight_kind::UNKNOWN;
    ggml_tensor * tensor = nullptr;
    size_t bytes = 0;
    size_t offset = 0;
    void * host_ptr = nullptr;
    ggml_type type = GGML_TYPE_F32;
    int32_t n_dims = 0;
    std::array<int64_t, GGML_MAX_DIMS> ne = {0, 0, 0, 0};
    std::array<size_t,  GGML_MAX_DIMS> nb = {0, 0, 0, 0};
    int64_t rows = 0;
    int64_t cols = 0;
    int32_t slice_axis = -1;
    bool is_quantized = false;
    bool is_contiguous = false;
    bool is_view = false;
};

static inline int32_t llama_moe_compose_id(int32_t layer, int32_t expert_index, llama_moe_weight_kind kind) {
    constexpr int32_t KIND_FACTOR   = 10;
    constexpr int32_t EXPERT_FACTOR = 1000;

    return layer * EXPERT_FACTOR * KIND_FACTOR +
           expert_index * KIND_FACTOR +
           static_cast<int32_t>(kind);
}

struct llama_moe_router_handle {
    int32_t layer = -1;
    ggml_tensor * tensor = nullptr;
    size_t bytes = 0;
};

struct llama_context;
class ExpertCache;

struct llama_moe_dispatch_desc {
    llama_context * ctx = nullptr;
    ExpertCache * cache = nullptr;
    ggml_backend_t backend = nullptr;
    int32_t layer = -1;
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_embd = 0;
    int32_t n_tokens = 0;
    int32_t n_ff = 0;
    llm_ffn_op_type activation = LLM_FFN_SILU;
    bool has_gate = false;
    bool has_gate_in = false;
    bool has_gate_bias = false;
    bool has_up_bias = false;
    bool has_down_bias = false;
    bool weight_before_ffn = false;
    bool allow_quantized = false;
    bool use_cuda = false;
};

struct llama_moe_cache_stats {
    size_t resident = 0;
    size_t capacity_bytes = 0;
    uint64_t loads = 0;
    uint64_t hits = 0;
    uint64_t evictions = 0;
    uint64_t prefetch_requests = 0;
    struct device_stats {
        int device = -1;
        size_t resident = 0;
        size_t capacity_bytes = 0;
        uint64_t loads = 0;
        uint64_t hits = 0;
        uint64_t evictions = 0;
    };
    std::vector<device_stats> per_device;
};

struct llama_moe_prefetch_stats {
    uint64_t updates = 0;
    uint64_t prefetch_calls = 0;
    uint64_t tokens_observed = 0;
};

ggml_tensor * llama_moe_build_dispatch(
        ggml_context * ctx,
        ggml_tensor * input,
        ggml_tensor * selected_experts,
        ggml_tensor * weights,
        const llama_moe_dispatch_desc & desc,
        llm_graph_result * owner = nullptr);

class ExpertCache {
public:
    struct Config {
        size_t vram_pool_bytes = 0;
        uint32_t max_resident_experts = 0;
        bool enable_prefetch = false;
        uint32_t prefetch_lookahead = 1;
        struct DevicePolicy {
            int device = -1;
            size_t capacity_bytes = 0;
            uint32_t max_resident_experts = 0;
            float weight = 0.0f;
        };
        std::vector<DevicePolicy> device_policies;
        bool auto_assign_devices = true;
    };

    ExpertCache();
    explicit ExpertCache(const Config & cfg);
    ~ExpertCache();

    ExpertCache(const ExpertCache &) = delete;
    ExpertCache & operator=(const ExpertCache &) = delete;

    void configure(const Config & cfg);

    void register_expert(const llama_moe_expert_handle & handle);
    void register_experts(const std::vector<llama_moe_expert_handle> & handles);

    void clear();

    bool has_resident(int32_t expert_id) const;
    const llama_moe_expert_handle * find(int32_t expert_id) const;
    llama_moe_cache_stats stats() const;
    void reset_stats();

#ifdef GGML_USE_CUDA
    void attach_stream(cudaStream_t stream, int device);
    cudaStream_t stream() const;
#endif

    // Ensures the expert is present on the active device. Returns device pointer or nullptr on failure.
    void * ensure_loaded(int32_t expert_id, int device = -1, ggml_backend_buffer_t device_buffer = nullptr);

    // Prefetch a list of experts asynchronously
    void prefetch(const std::vector<int32_t> & expert_ids);

    size_t resident_count() const;
    size_t capacity_bytes() const;

private:
    struct DeviceSlot {
        int32_t expert_id = -1;
        void * device_ptr = nullptr;
        size_t bytes = 0;
        uint64_t last_used = 0;
        uint64_t hits = 0;
        void * host_staging = nullptr;
        size_t staging_capacity = 0;
    };

    struct DevicePool {
        std::vector<DeviceSlot> slots;
        size_t pool_bytes = 0;
        struct Stats {
            uint64_t loads = 0;
            uint64_t hits = 0;
            uint64_t evictions = 0;
        } stats;
    };

    struct ExpertRecord {
        llama_moe_expert_handle handle;
        std::unordered_map<int, size_t> slot_by_device;
    };

    using ExpertMap = std::unordered_map<int32_t, ExpertRecord>;

    void allocate_pool();
    void release_pool();
    DeviceSlot * find_lru(int device);
    DevicePool & get_or_create_pool(int device);
    size_t capacity_for_device(int device) const;
    size_t max_slots_for_device(int device) const;
    int select_device_for_expert(int32_t expert_id, int device_hint) const;

    Config config_;
    ExpertMap experts_;
    std::unordered_map<int, DevicePool> device_pools_;
    size_t pool_bytes_ = 0;
    uint64_t timestamp_ = 0;
    mutable llama_moe_cache_stats stats_;

#ifdef GGML_USE_CUDA
    cudaStream_t stream_ = nullptr;
    std::unordered_map<int, cudaStream_t> device_streams_;
    int current_device_ = -1;
#endif
    std::vector<Config::DevicePolicy> device_policies_;
    std::unordered_map<int, Config::DevicePolicy> device_policy_by_id_;
    double device_policy_total_weight_ = 0.0;
    bool auto_assign_devices_ = true;

    mutable std::mutex mutex_;
};

#endif // LLAMA_MOE_ENABLE
