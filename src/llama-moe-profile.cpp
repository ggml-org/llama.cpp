#include "llama-moe-profile.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>

// Magic number for profile file format
static const uint32_t LLAMA_MOE_PROFILE_MAGIC   = 0x4D4F4550; // "MOEP"
static const uint32_t LLAMA_MOE_PROFILE_VERSION = 1;

//
// llama_moe_layer_stats
//

void llama_moe_layer_stats::reset(uint32_t n_expert) {
    expert_counts.assign(n_expert, 0);
    total_selections = 0;
}

void llama_moe_layer_stats::update(const int32_t * expert_ids, int n_tokens, int n_expert_used) {
    for (int t = 0; t < n_tokens; t++) {
        for (int k = 0; k < n_expert_used; k++) {
            int32_t exp_id = expert_ids[t * n_expert_used + k];
            if (exp_id >= 0 && exp_id < (int32_t)expert_counts.size()) {
                expert_counts[exp_id]++;
                total_selections++;
            }
        }
    }
}

float llama_moe_layer_stats::usage_ratio(uint32_t expert_id) const {
    if (total_selections == 0 || expert_id >= expert_counts.size()) {
        return 0.0f;
    }
    return (float)expert_counts[expert_id] / (float)total_selections;
}

//
// llama_moe_profile
//

void llama_moe_profile::init(uint32_t n_layer_, uint32_t n_expert_, uint32_t n_expert_used_) {
    n_layer       = n_layer_;
    n_expert      = n_expert_;
    n_expert_used = n_expert_used_;

    layer_stats.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; il++) {
        layer_stats[il].il = il;
        layer_stats[il].reset(n_expert);
    }

    expert_on_gpu.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; il++) {
        expert_on_gpu[il].assign(n_expert, true); // default: all on GPU
    }

    total_tokens_profiled = 0;
}

void llama_moe_profile::reset() {
    for (auto & stats : layer_stats) {
        stats.reset(n_expert);
    }
    total_tokens_profiled = 0;
}

void llama_moe_profile::update(uint32_t il, const int32_t * expert_ids, int n_tokens) {
    if (il < layer_stats.size()) {
        layer_stats[il].update(expert_ids, n_tokens, n_expert_used);
        // only count tokens once (use layer 0 as reference)
        if (il == 0) {
            total_tokens_profiled += n_tokens;
        }
    }
}

void llama_moe_profile::analyze(float gpu_fraction) {
    gpu_fraction = std::max(0.0f, std::min(1.0f, gpu_fraction));
    analyzed_gpu_fraction = gpu_fraction;  // store for later use

    for (uint32_t il = 0; il < n_layer; il++) {
        const auto & stats = layer_stats[il];

        // rank experts by usage
        std::vector<std::pair<float, uint32_t>> ranked;
        ranked.reserve(n_expert);
        for (uint32_t e = 0; e < n_expert; e++) {
            ranked.push_back({stats.usage_ratio(e), e});
        }

        // sort descending by usage
        std::sort(ranked.begin(), ranked.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });

        // top gpu_fraction experts go on GPU
        uint32_t n_gpu = (uint32_t)(n_expert * gpu_fraction + 0.5f);
        n_gpu = std::max(1u, std::min(n_expert, n_gpu)); // at least 1, at most all

        expert_on_gpu[il].assign(n_expert, false);
        for (uint32_t i = 0; i < n_gpu; i++) {
            expert_on_gpu[il][ranked[i].second] = true;
        }
    }
}

bool llama_moe_profile::has_mixed_placement() const {
    for (uint32_t il = 0; il < n_layer; il++) {
        if (has_mixed_placement(il)) {
            return true;
        }
    }
    return false;
}

bool llama_moe_profile::has_mixed_placement(uint32_t il) const {
    if (il >= expert_on_gpu.size() || expert_on_gpu[il].empty()) {
        return false;
    }

    bool first = expert_on_gpu[il][0];
    for (uint32_t e = 1; e < expert_on_gpu[il].size(); e++) {
        if (expert_on_gpu[il][e] != first) {
            return true;
        }
    }
    return false;
}

uint32_t llama_moe_profile::n_experts_on_gpu(uint32_t il) const {
    if (il >= expert_on_gpu.size()) {
        return 0;
    }
    return std::count(expert_on_gpu[il].begin(), expert_on_gpu[il].end(), true);
}

bool llama_moe_profile::save(const std::string & path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        LLAMA_LOG_ERROR("%s: failed to open file for writing: %s\n", __func__, path.c_str());
        return false;
    }

    // write header
    file.write(reinterpret_cast<const char *>(&LLAMA_MOE_PROFILE_MAGIC), sizeof(uint32_t));
    file.write(reinterpret_cast<const char *>(&LLAMA_MOE_PROFILE_VERSION), sizeof(uint32_t));

    // write model config
    file.write(reinterpret_cast<const char *>(&n_layer), sizeof(uint32_t));
    file.write(reinterpret_cast<const char *>(&n_expert), sizeof(uint32_t));
    file.write(reinterpret_cast<const char *>(&n_expert_used), sizeof(uint32_t));
    file.write(reinterpret_cast<const char *>(&total_tokens_profiled), sizeof(uint64_t));

    // write model path
    uint32_t path_len = model_path.size();
    file.write(reinterpret_cast<const char *>(&path_len), sizeof(uint32_t));
    file.write(model_path.data(), path_len);

    // write per-layer statistics
    for (uint32_t il = 0; il < n_layer; il++) {
        const auto & stats = layer_stats[il];
        file.write(reinterpret_cast<const char *>(&stats.total_selections), sizeof(uint64_t));
        file.write(reinterpret_cast<const char *>(stats.expert_counts.data()),
                   n_expert * sizeof(uint64_t));
    }

    // write placement decisions
    for (uint32_t il = 0; il < n_layer; il++) {
        for (uint32_t e = 0; e < n_expert; e++) {
            uint8_t on_gpu = expert_on_gpu[il][e] ? 1 : 0;
            file.write(reinterpret_cast<const char *>(&on_gpu), sizeof(uint8_t));
        }
    }

    LLAMA_LOG_INFO("%s: saved MoE profile to %s (%" PRIu64 " tokens profiled)\n",
                   __func__, path.c_str(), total_tokens_profiled);
    return true;
}

bool llama_moe_profile::load(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        LLAMA_LOG_ERROR("%s: failed to open file for reading: %s\n", __func__, path.c_str());
        return false;
    }

    // read and verify header
    uint32_t magic, version;
    file.read(reinterpret_cast<char *>(&magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));

    if (magic != LLAMA_MOE_PROFILE_MAGIC) {
        LLAMA_LOG_ERROR("%s: invalid magic number in %s\n", __func__, path.c_str());
        return false;
    }
    if (version != LLAMA_MOE_PROFILE_VERSION) {
        LLAMA_LOG_ERROR("%s: unsupported version %u in %s (expected %u)\n",
                        __func__, version, path.c_str(), LLAMA_MOE_PROFILE_VERSION);
        return false;
    }

    // read model config
    file.read(reinterpret_cast<char *>(&n_layer), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&n_expert), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&n_expert_used), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&total_tokens_profiled), sizeof(uint64_t));

    // read model path
    uint32_t path_len;
    file.read(reinterpret_cast<char *>(&path_len), sizeof(uint32_t));
    model_path.resize(path_len);
    file.read(&model_path[0], path_len);

    // initialize structures
    layer_stats.resize(n_layer);
    expert_on_gpu.resize(n_layer);

    // read per-layer statistics
    for (uint32_t il = 0; il < n_layer; il++) {
        layer_stats[il].il = il;
        layer_stats[il].expert_counts.resize(n_expert);

        file.read(reinterpret_cast<char *>(&layer_stats[il].total_selections), sizeof(uint64_t));
        file.read(reinterpret_cast<char *>(layer_stats[il].expert_counts.data()),
                  n_expert * sizeof(uint64_t));
    }

    // read placement decisions
    for (uint32_t il = 0; il < n_layer; il++) {
        expert_on_gpu[il].resize(n_expert);
        for (uint32_t e = 0; e < n_expert; e++) {
            uint8_t on_gpu;
            file.read(reinterpret_cast<char *>(&on_gpu), sizeof(uint8_t));
            expert_on_gpu[il][e] = (on_gpu != 0);
        }
    }

    LLAMA_LOG_INFO("%s: loaded MoE profile from %s (%" PRIu64 " tokens profiled)\n",
                   __func__, path.c_str(), total_tokens_profiled);
    return true;
}

void llama_moe_profile::print_summary() const {
    LLAMA_LOG_INFO("\n");
    LLAMA_LOG_INFO("=== MoE Expert Usage Profile ===\n");
    LLAMA_LOG_INFO("Model: %s\n", model_path.c_str());
    LLAMA_LOG_INFO("Layers: %u, Experts: %u, Top-K: %u\n", n_layer, n_expert, n_expert_used);
    LLAMA_LOG_INFO("Tokens profiled: %" PRIu64 "\n", total_tokens_profiled);
    LLAMA_LOG_INFO("\n");

    // show top 5 and bottom 5 experts per layer (if enough experts)
    uint32_t n_show = std::min(5u, n_expert);

    for (uint32_t il = 0; il < n_layer; il++) {
        const auto & stats = layer_stats[il];

        // rank experts
        std::vector<std::pair<float, uint32_t>> ranked;
        for (uint32_t e = 0; e < n_expert; e++) {
            ranked.push_back({stats.usage_ratio(e), e});
        }
        std::sort(ranked.begin(), ranked.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });

        LLAMA_LOG_INFO("Layer %2u: GPU=%2u/%u | Hot: ",
                       il, n_experts_on_gpu(il), n_expert);

        for (uint32_t i = 0; i < n_show; i++) {
            LLAMA_LOG_INFO("e%u(%.1f%%) ", ranked[i].second, ranked[i].first * 100.0f);
        }

        if (n_expert > n_show * 2) {
            LLAMA_LOG_INFO("... Cold: ");
            for (uint32_t i = n_expert - n_show; i < n_expert; i++) {
                LLAMA_LOG_INFO("e%u(%.1f%%) ", ranked[i].second, ranked[i].first * 100.0f);
            }
        }
        LLAMA_LOG_INFO("\n");
    }
    LLAMA_LOG_INFO("\n");
}

//
// llama_moe_profiler
//

void llama_moe_profiler::init(const struct llama_hparams & hparams) {
    if (hparams.n_expert == 0) {
        enabled = false;
        return;
    }

    profile.init(hparams.n_layer, hparams.n_expert, hparams.n_expert_used);
    read_buffer.reserve(4096 * hparams.n_expert_used); // pre-allocate for typical batch
}

void llama_moe_profiler::schedule_capture(uint32_t il, struct ggml_tensor * expert_ids,
                                           int n_tokens, int n_expert_used) {
    if (!enabled || expert_ids == nullptr) {
        return;
    }

    pending_reads.push_back({il, expert_ids, n_tokens, n_expert_used});
}

void llama_moe_profiler::flush(struct ggml_backend * backend) {
    GGML_UNUSED(backend);

    if (pending_reads.empty()) {
        return;
    }

    for (const auto & read : pending_reads) {
        // The tensor has shape [n_expert_used, n_tokens] from ggml_argsort_top_k
        // Total elements = tensor->ne[0] * tensor->ne[1]
        const size_t n_elements = ggml_nelements(read.tensor);
        const size_t tensor_size = ggml_nbytes(read.tensor);

        if (read_buffer.size() < n_elements) {
            read_buffer.resize(n_elements);
        }

        // synchronous read from device
        ggml_backend_tensor_get(read.tensor, read_buffer.data(), 0, tensor_size);

        // update profile - tensor layout is [n_expert_used, n_tokens]
        // so we need to read n_tokens * n_expert_used elements
        const int n_tokens = read.tensor->ne[1];
        const int n_expert_used = read.tensor->ne[0];
        profile.update(read.il, read_buffer.data(), n_tokens);

        GGML_UNUSED(n_expert_used); // used for documentation
    }

    pending_reads.clear();
}

//
// Phase 2: Placement decisions and override generation
//

float llama_moe_profile::cold_expert_ratio(uint32_t il, float threshold_ratio) const {
    if (il >= layer_stats.size() || n_expert == 0) {
        return 0.0f;
    }

    const auto & stats = layer_stats[il];
    if (stats.total_selections == 0) {
        return 1.0f; // no data = assume all cold
    }

    // expected uniform usage = n_expert_used/n_expert
    // (each token activates n_expert_used experts out of n_expert total)
    // threshold = threshold_ratio * expected_uniform_usage
    float expected_usage = (float)n_expert_used / (float)n_expert;
    float threshold = threshold_ratio * expected_usage;

    uint32_t n_cold = 0;
    for (uint32_t e = 0; e < n_expert; e++) {
        if (stats.usage_ratio(e) < threshold) {
            n_cold++;
        }
    }

    return (float)n_cold / (float)n_expert;
}

void llama_moe_profile::analyze_by_threshold(float threshold_ratio) {
    threshold_ratio = std::max(0.0f, threshold_ratio);

    for (uint32_t il = 0; il < n_layer; il++) {
        const auto & stats = layer_stats[il];

        // expected uniform usage = n_expert_used/n_expert
        // threshold = threshold_ratio * expected_uniform_usage
        float expected_usage = (float)n_expert_used / (float)n_expert;
        float threshold = threshold_ratio * expected_usage;

        expert_on_gpu[il].assign(n_expert, false);
        for (uint32_t e = 0; e < n_expert; e++) {
            // experts with usage >= threshold go on GPU
            expert_on_gpu[il][e] = (stats.usage_ratio(e) >= threshold);
        }

        // ensure at least n_expert_used experts on GPU (needed for inference)
        uint32_t n_on_gpu = n_experts_on_gpu(il);
        if (n_on_gpu < n_expert_used) {
            // add more experts until we have enough
            std::vector<std::pair<float, uint32_t>> ranked;
            for (uint32_t e = 0; e < n_expert; e++) {
                if (!expert_on_gpu[il][e]) {
                    ranked.push_back({stats.usage_ratio(e), e});
                }
            }
            std::sort(ranked.begin(), ranked.end(),
                [](const auto & a, const auto & b) { return a.first > b.first; });

            for (size_t i = 0; i < ranked.size() && n_on_gpu < n_expert_used; i++) {
                expert_on_gpu[il][ranked[i].second] = true;
                n_on_gpu++;
            }
        }
    }
}

std::vector<uint32_t> llama_moe_profile::get_cpu_layers(float cold_threshold, float expert_threshold_ratio) const {
    std::vector<uint32_t> cpu_layers;

    for (uint32_t il = 0; il < n_layer; il++) {
        // if most experts are cold, put entire layer on CPU
        // expert_threshold_ratio controls what counts as "cold" at the expert level
        // (e.g., 0.1 means experts with < 10% of expected usage are cold)
        if (cold_expert_ratio(il, expert_threshold_ratio) >= cold_threshold) {
            cpu_layers.push_back(il);
        }
    }

    return cpu_layers;
}

std::vector<std::pair<std::string, std::string>> llama_moe_profile::generate_layer_overrides(
    float cold_threshold, const char * cpu_buffer_name, float expert_threshold_ratio) const {

    std::vector<std::pair<std::string, std::string>> overrides;

    // Pattern for MoE expert tensors in a layer: blk\.N\.ffn_(up|down|gate)_(ch|)exps
    const char * exp_pattern = "\\.ffn_(up|down|gate)_(ch|)exps";

    for (uint32_t il = 0; il < n_layer; il++) {
        // Use expert_threshold_ratio to determine what counts as "cold"
        // e.g., 0.1 means experts with < 10% of expected usage are cold
        float cold_ratio = cold_expert_ratio(il, expert_threshold_ratio);
        if (cold_ratio >= cold_threshold) {
            // generate regex for this layer's expert tensors
            char pattern[128];
            snprintf(pattern, sizeof(pattern), "blk\\.%u%s", il, exp_pattern);
            overrides.push_back({pattern, cpu_buffer_name});
        }
    }

    return overrides;
}

llama_moe_profile::memory_estimate llama_moe_profile::estimate_memory(size_t bytes_per_expert) const {
    memory_estimate est = {};

    if (n_layer == 0 || n_expert == 0) {
        return est;
    }

    // 3 tensors per expert: gate, up, down
    size_t bytes_per_expert_total = bytes_per_expert * 3;

    for (uint32_t il = 0; il < n_layer; il++) {
        uint32_t n_gpu = n_experts_on_gpu(il);
        uint32_t n_cpu = n_expert - n_gpu;

        est.gpu_expert_memory += n_gpu * bytes_per_expert_total;
        est.cpu_expert_memory += n_cpu * bytes_per_expert_total;
    }

    est.total_expert_memory = est.gpu_expert_memory + est.cpu_expert_memory;
    est.savings = est.cpu_expert_memory; // bytes moved off GPU
    est.savings_percent = (est.total_expert_memory > 0)
        ? (100.0f * est.savings / est.total_expert_memory)
        : 0.0f;

    return est;
}

void llama_moe_profile::print_override_cli() const {
    // Only recommend CPU offload if user is actually memory-constrained
    // (gpu_fraction < 1.0 means they can't fit everything on GPU)
    if (analyzed_gpu_fraction >= 0.99f) {
        // Model fits in VRAM - no benefit to CPU offload, would only hurt performance
        return;
    }

    // Use a high threshold (0.9) to only recommend CPU offload for layers
    // where 90%+ of experts are truly cold (well below expected usage).
    // This is conservative because sparse MoE naturally has non-uniform distribution.
    auto overrides = generate_layer_overrides(0.9f, "CPU");

    if (overrides.empty()) {
        LLAMA_LOG_INFO("No CPU overrides needed - all layers have active experts\n");
        return;
    }

    LLAMA_LOG_INFO("\n=== Recommended CLI Override ===\n");
    LLAMA_LOG_INFO("Add to your command:\n");
    LLAMA_LOG_INFO("  -ot \"");

    for (size_t i = 0; i < overrides.size(); i++) {
        if (i > 0) {
            LLAMA_LOG_INFO(",");
        }
        LLAMA_LOG_INFO("%s=%s", overrides[i].first.c_str(), overrides[i].second.c_str());
    }
    LLAMA_LOG_INFO("\"\n\n");

    // also show the simpler -n-cpu-moe if layers are contiguous from 0
    bool contiguous_from_zero = true;
    for (size_t i = 0; i < overrides.size(); i++) {
        // extract layer number from pattern "blk\.N\..."
        uint32_t layer_num;
        if (sscanf(overrides[i].first.c_str(), "blk\\.%u", &layer_num) != 1 || layer_num != i) {
            contiguous_from_zero = false;
            break;
        }
    }

    if (contiguous_from_zero && !overrides.empty()) {
        LLAMA_LOG_INFO("Or use simplified form:\n");
        LLAMA_LOG_INFO("  -n-cpu-moe %zu\n\n", overrides.size());
    }
}
