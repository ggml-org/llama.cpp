#pragma once

#include "llama.h"
#include "llama-hparams.h"

#include <string>
#include <vector>
#include <cstdint>

// MoE expert usage statistics per layer
struct llama_moe_layer_stats {
    uint32_t il;                              // layer index
    std::vector<uint64_t> expert_counts;      // [n_expert] selection counts
    uint64_t total_selections = 0;            // total expert selections (tokens * n_expert_used)

    void reset(uint32_t n_expert);
    void update(const int32_t * expert_ids, int n_tokens, int n_expert_used);
    float usage_ratio(uint32_t expert_id) const;
};

// MoE expert usage profile for a model
struct llama_moe_profile {
    // model configuration (for validation)
    uint32_t n_layer       = 0;
    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;

    // per-layer statistics
    std::vector<llama_moe_layer_stats> layer_stats;

    // profiling metadata
    uint64_t    total_tokens_profiled = 0;
    std::string model_path;

    // placement decisions (populated after analyze())
    // expert_on_gpu[il][e] = true if expert e in layer il should be on GPU
    std::vector<std::vector<bool>> expert_on_gpu;

    // the gpu_fraction used in last analyze() call (for determining if offload is needed)
    float analyzed_gpu_fraction = 1.0f;

    // initialize profile for a model
    void init(uint32_t n_layer, uint32_t n_expert, uint32_t n_expert_used);

    // reset all statistics
    void reset();

    // update statistics with expert selection from a layer
    void update(uint32_t il, const int32_t * expert_ids, int n_tokens);

    // analyze usage and determine placement
    // gpu_fraction: 0.0-1.0, fraction of experts to keep on GPU per layer
    void analyze(float gpu_fraction);

    // check if any layer has mixed placement (some experts GPU, some CPU)
    bool has_mixed_placement() const;

    // check if specific layer has mixed placement
    bool has_mixed_placement(uint32_t il) const;

    // get number of GPU experts for a layer
    uint32_t n_experts_on_gpu(uint32_t il) const;

    // serialization
    bool save(const std::string & path) const;
    bool load(const std::string & path);

    // print summary to log
    void print_summary() const;

    //
    // Phase 2: Placement decisions and override generation
    //

    // calculate cold expert ratio for a layer (0.0 = all hot, 1.0 = all cold)
    // cold = experts with usage below threshold (default: 1/n_expert expected usage)
    float cold_expert_ratio(uint32_t il, float threshold_ratio = 1.0f) const;

    // analyze using absolute usage threshold instead of top-K fraction
    // threshold_ratio: experts with usage < (threshold_ratio / n_expert) are cold
    void analyze_by_threshold(float threshold_ratio = 0.5f);

    // get layers that should have their experts on CPU (cold_ratio > threshold)
    // expert_threshold_ratio: what fraction of expected usage counts as "cold" (default 0.1 = <10% of expected)
    std::vector<uint32_t> get_cpu_layers(float cold_threshold = 0.5f, float expert_threshold_ratio = 0.1f) const;

    // generate tensor buffer override strings for fit algorithm
    // returns vector of {pattern, buffer_type_name} pairs
    // e.g., {"blk\\.5\\.ffn_(up|down|gate)_exps", "CPU"}
    // expert_threshold_ratio: what fraction of expected usage counts as "cold"
    std::vector<std::pair<std::string, std::string>> generate_layer_overrides(
        float cold_threshold = 0.5f,
        const char * cpu_buffer_name = "CPU",
        float expert_threshold_ratio = 0.1f) const;

    // estimate memory savings from mixed placement (in bytes)
    // requires expert tensor sizes from model
    struct memory_estimate {
        size_t total_expert_memory;     // total expert tensor memory
        size_t gpu_expert_memory;       // memory for GPU experts
        size_t cpu_expert_memory;       // memory for CPU experts
        size_t savings;                 // bytes saved on GPU
        float  savings_percent;         // percentage saved
    };
    memory_estimate estimate_memory(size_t bytes_per_expert) const;

    // print override CLI string for copy-paste
    void print_override_cli() const;
};

// Context extension for MoE profiling
struct llama_moe_profiler {
    bool enabled = false;
    bool warmup_mode = false;          // dedicated warmup pass
    uint32_t warmup_tokens_target = 0; // tokens to process in warmup

    llama_moe_profile profile;

    // pending reads for async profiling
    struct pending_read {
        uint32_t il;
        struct ggml_tensor * tensor;
        int n_tokens;
        int n_expert_used;
    };
    std::vector<pending_read> pending_reads;
    std::vector<int32_t> read_buffer;

    void init(const struct llama_hparams & hparams);
    void schedule_capture(uint32_t il, struct ggml_tensor * expert_ids, int n_tokens, int n_expert_used);
    void flush(struct ggml_backend * backend);
};
