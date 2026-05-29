#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_tensor;
typedef struct ggml_backend_sched * ggml_backend_sched_t;

// Per-op FLOPS/bytes metrics extracted from a ggml tensor node.
// Used by roofline prediction to classify ops as compute-bound or memory-bound.
struct llama_op_metrics {
    double      ops         = 0.0;
    double      bytes       = 0.0;
    const char * quant_type = nullptr;

    int64_t N = 0, K = 0, M = 0;
    int64_t ctx_len         = 0;
    int64_t n_elements      = 0;
    int64_t n_kv_heads      = 0;
    int64_t n_experts_used  = 0;
    int64_t head_dim        = 0;
    int64_t n_q_heads       = 0;
    int64_t n_tokens        = 0;
};

// Extract ops/bytes metrics from a compute graph node.
// Zero-cost view ops (RESHAPE, VIEW, PERMUTE, TRANSPOSE, NONE) return {0,0}.
llama_op_metrics llama_op_metrics_compute(const ggml_tensor * node);

// One row from profiler output (CPU or GPU).
// For FLASH_ATTN ops, n_heads stores n_kv_heads (written by profiler).
struct llama_benchmark_entry {
    std::string op_name;
    std::string quant;
    int         threads = 0;

    double ai          = 0.0;   // arithmetic intensity (FLOP/byte)
    double bw_gb_s     = 0.0;   // effective memory bandwidth (GB/s)
    double peak_gflops = 0.0;   // standalone compute throughput (GFLOP/s)
    double ridge       = 0.0;   // roofline ridge point (FLOP/byte)
    double eff_gflops  = 0.0;   // concurrent compute (CPU: under PCIe load; GPU: == peak)
    double eff_pcie_bw = 0.0;   // effective PCIe BW during concurrent compute (CPU only)

    int64_t N = 0, K = 0, B = 0;
    int64_t n_tokens   = 0;
    int64_t ctx_len    = 0;
    int64_t n_heads    = 0;     // n_kv_heads for FLASH_ATTN, unused for other ops
    int64_t head_dim   = 0;
    int64_t n_elements = 0;
};

// Global bandwidth/compute stats parsed from profiler file headers.
struct llama_benchmark_stats {
    // CPU (from cpu profiler header)
    double peak_system_bw   = 0.0;   // peak DRAM BW (GB/s)
    double peak_pcie_bw     = 0.0;   // peak PCIe BW standalone (GB/s)
    double eff_system_bw    = 0.0;   // DRAM BW under concurrent PCIe load (GB/s)
    double eff_pcie_bw      = 0.0;   // PCIe BW under concurrent CPU load (GB/s)

    // GPU (from gpu profiler header)
    double peak_gpu_mem_bw  = 0.0;   // peak GPU memory BW (GB/s)
    double peak_gpu_compute = 0.0;   // peak GPU compute (GFLOP/s)
};

// Aggregate timing prediction for a set of graph nodes (typically one sched split).
struct llama_split_timing {
    double time_ms     = 0.0;
    double eff_pcie_bw = 0.0;   // time-weighted avg PCIe BW (CPU splits only)
    int    op_count    = 0;
    int    n_cache_hit = 0;     // timing cache reuses
    int    n_exact     = 0;     // exact benchmark hash hits
    int    n_nearest   = 0;     // nearest-neighbor matches
    int    n_fallback  = 0;     // memory BW fallback (no benchmark match)
};

// Benchmark data + O(1) hash maps + nearest-neighbor lookup + split timing prediction.
// Owns entry vectors; map pointers are stable after load.
struct llama_benchmark_predictor {
    std::vector<llama_benchmark_entry> cpu_entries;
    std::vector<llama_benchmark_entry> gpu_entries;
    llama_benchmark_stats stats = {};

    std::unordered_map<std::string, const llama_benchmark_entry *> cpu_map;
    std::unordered_map<std::string, const llama_benchmark_entry *> gpu_map;

    // Timing cache: same shape + same mode = same timing across layers.
    // Key: make_timing_key(), Value: {time_ms, pcie_contrib}.
    using timing_cache_t = std::unordered_map<std::string, std::pair<double, double>>;

    llama_benchmark_predictor() = default;
    llama_benchmark_predictor(const llama_benchmark_predictor &) = delete;
    llama_benchmark_predictor & operator=(const llama_benchmark_predictor &) = delete;

    // Parse profiler output, populate entries + hash maps.
    // CPU loader filters by n_threads; returns false if file missing or empty.
    bool load_cpu(const char * filepath, int n_threads);
    bool load_gpu(const char * filepath);

    bool has_cpu() const { return !cpu_entries.empty(); }
    bool has_gpu() const { return !gpu_entries.empty(); }

    // Predict aggregate timing for a set of graph nodes.
    // nodes/n_nodes: typically from one sched split (via ggml_graph_nodes/n_nodes).
    // async_copy: true when PCIe transfers run concurrently (uses eff_* bandwidths for CPU).
    llama_split_timing predict_split(
            struct ggml_tensor ** nodes, int n_nodes,
            bool is_gpu, int32_t batch_size, bool async_copy = true,
            timing_cache_t * timing_cache = nullptr) const;

    // Nearest-neighbor search with weighted scoring (batch > dims > quant).
    // For FLASH_ATTN queries, pass op_name="FLASH_ATTN" -- matches all FLASH_ATTN_* entries.
    static const llama_benchmark_entry * find_nearest(
            const std::vector<llama_benchmark_entry> & entries,
            const char * op_name, const char * quant,
            int64_t N, int64_t K, int64_t ctx_len, int64_t n_elements,
            int64_t n_kv_heads = -1, int64_t target_batch = 1);

    // Key generators for hash map lookup (used by predict to build exact-match keys).
    static std::string make_key(
            const std::string & op, const std::string & quant,
            int64_t N, int64_t K, int64_t batch);

    static std::string make_attn_key(
            int64_t ctx_len, int64_t n_kv_heads, int64_t n_tokens);

    static std::string make_elem_key(
            const std::string & op, int64_t n_elements);

    // Timing cache key: encodes backend type, async mode, op, shape, and batch.
    static std::string make_timing_key(
            bool is_gpu, bool async_copy, const char * op_name,
            const llama_op_metrics & m, int32_t batch_size);

    // End-to-end TPS prediction across all sched splits.
    // kv_size: total KV cells per layer (for partial upload ratio).
    // has_rs: true if model has recurrent state layers (RS uses full-tensor transfer, not cell-granular).
    // Returns 0 if no splits or no benchmark data.
    double predict_tps(
            ggml_backend_sched_t sched,
            int cpu_backend_id,
            uint32_t kv_size,
            int32_t batch_size,
            uint32_t n_outputs = 0,
            bool has_rs = false) const;

private:
    void build_maps();
};
