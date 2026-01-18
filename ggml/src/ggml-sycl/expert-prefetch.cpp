// Expert Prefetcher implementation
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-eqa)
//
// TDD: Implementation follows RED-GREEN-REFACTOR cycle.
// Tests in test-sycl-expert-prefetch.cpp define expected behavior.

#include "expert-prefetch.hpp"

#include <algorithm>
#include <cstdio>
#include <unordered_set>
#include <vector>

namespace ggml_sycl {

// ============================================================================
// Implementation structure (pimpl pattern)
// ============================================================================
struct ExpertPrefetcher::Impl {
    uint32_t num_layers{ 0 };
    uint32_t num_experts{ 0 };
    size_t   expert_size{ 0 };

    std::vector<LayerStats> layer_stats;

    // Stub data storage for mock expert data
    // In production, this would be integrated with ChunkManager/unified_cache
    std::vector<uint8_t> mock_data;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================
ExpertPrefetcher::ExpertPrefetcher() : impl_(new Impl()) {}

ExpertPrefetcher::~ExpertPrefetcher() {
    delete impl_;
}

// ============================================================================
// Configuration
// ============================================================================
void ExpertPrefetcher::configure(uint32_t num_layers, uint32_t num_experts, size_t expert_size) {
    impl_->num_layers  = num_layers;
    impl_->num_experts = num_experts;
    impl_->expert_size = expert_size;

    impl_->layer_stats.clear();
    impl_->layer_stats.resize(num_layers);

    // Allocate mock data buffer (for testing without real SYCL device)
    impl_->mock_data.resize(expert_size > 0 ? expert_size : 1024);
}

// ============================================================================
// Start prefetch - sorts experts by score and initiates prefetching
// ============================================================================
PrefetchBatch ExpertPrefetcher::start_prefetch(uint32_t      layer_id,
                                               const float * router_scores,
                                               size_t        num_experts,
                                               size_t        top_k) {
    PrefetchBatch batch;
    batch.layer_id       = layer_id;
    batch.num_prefetched = 0;
    batch.num_selected   = 0;
    batch.num_hits       = 0;
    batch.num_misses     = 0;

    // Handle edge cases
    if (router_scores == nullptr || num_experts == 0 || top_k == 0) {
        return batch;
    }

    // 1. Score and sort all experts by descending score
    // Use pair<score, expert_id> for sorting, with expert_id as tiebreaker
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(num_experts);
    for (uint32_t i = 0; i < num_experts; i++) {
        scored.emplace_back(router_scores[i], i);
    }

    // Sort descending by score, then ascending by expert_id (for stable tiebreaking)
    std::sort(scored.begin(), scored.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) {
            return a.first > b.first;  // Higher score first
        }
        return a.second < b.second;    // Lower ID first for ties
    });

    // 2. Determine prefetch count based on layer accuracy (adaptive)
    size_t prefetch_count = top_k;
    if (layer_id < impl_->layer_stats.size()) {
        prefetch_count = impl_->layer_stats[layer_id].recommended_prefetch_count(top_k);
    }
    prefetch_count = std::min(prefetch_count, num_experts);

    // 3. Build predictions list for top experts
    batch.predictions.reserve(prefetch_count);
    for (size_t i = 0; i < prefetch_count; i++) {
        ExpertPrediction pred;
        pred.expert_id = scored[i].second;
        pred.score     = scored[i].first;
        pred.selected  = false;

        // In production: initiate async DMA here
        // pred.prefetch_future = async_stream_expert(layer_id, pred.expert_id);

        batch.predictions.push_back(pred);
    }

    batch.num_prefetched = prefetch_count;
    return batch;
}

// ============================================================================
// Get expert data - uses prefetch if available, else streams on-demand
// ============================================================================
void * ExpertPrefetcher::get_expert_data(uint32_t layer_id, uint32_t expert_id, PrefetchBatch & batch) {
    (void) layer_id;  // Used in production for actual streaming

    // Check if expert was prefetched
    for (const auto & pred : batch.predictions) {
        if (pred.expert_id == expert_id) {
            // Expert was prefetched - this is a hit
            batch.num_hits++;

            // In production: return pred.prefetch_future.get();
            // For testing: return mock data pointer
            return impl_->mock_data.data();
        }
    }

    // Expert was NOT prefetched - this is a miss (must stream on-demand)
    batch.num_misses++;

    // In production: stream_expert_sync(layer_id, expert_id)
    // For testing: return mock data pointer
    return impl_->mock_data.data();
}

// ============================================================================
// Record selections - updates accuracy tracking
// ============================================================================
void ExpertPrefetcher::record_selections(uint32_t                      layer_id,
                                         const std::vector<uint32_t> & selected_experts,
                                         PrefetchBatch &               batch) {
    // Record number of selections
    batch.num_selected = selected_experts.size();

    // Build set for O(1) lookup
    std::unordered_set<uint32_t> selected_set(selected_experts.begin(), selected_experts.end());

    // Count how many predictions were correct
    size_t correct = 0;
    for (auto & pred : batch.predictions) {
        if (selected_set.count(pred.expert_id)) {
            pred.selected = true;
            correct++;
        }
    }

    // Update layer statistics
    if (layer_id < impl_->layer_stats.size()) {
        impl_->layer_stats[layer_id].total_predictions.fetch_add(batch.num_prefetched, std::memory_order_relaxed);
        impl_->layer_stats[layer_id].correct_predictions.fetch_add(correct, std::memory_order_relaxed);
    }
}

// ============================================================================
// Get prefetch order - returns position in prefetch queue, or -1 if not prefetched
// ============================================================================
int ExpertPrefetcher::get_prefetch_order(const PrefetchBatch & batch, uint32_t expert_id) const {
    for (size_t i = 0; i < batch.predictions.size(); i++) {
        if (batch.predictions[i].expert_id == expert_id) {
            return static_cast<int>(i);
        }
    }
    return -1;  // Not prefetched
}

// ============================================================================
// Get layer accuracy
// ============================================================================
float ExpertPrefetcher::get_layer_accuracy(uint32_t layer_id) const {
    if (layer_id >= impl_->layer_stats.size()) {
        return 0.0f;
    }
    return impl_->layer_stats[layer_id].accuracy();
}

// ============================================================================
// Get stats - aggregate statistics across all layers
// ============================================================================
PrefetchStats ExpertPrefetcher::get_stats() const {
    PrefetchStats stats;
    stats.total_predictions = 0;
    stats.total_correct     = 0;

    for (const auto & layer : impl_->layer_stats) {
        stats.total_predictions += layer.total_predictions.load(std::memory_order_relaxed);
        stats.total_correct += layer.correct_predictions.load(std::memory_order_relaxed);
    }

    if (stats.total_predictions > 0) {
        stats.overall_accuracy = static_cast<float>(stats.total_correct) / static_cast<float>(stats.total_predictions);
    } else {
        stats.overall_accuracy = 0.0f;
    }

    return stats;
}

// ============================================================================
// Reset stats
// ============================================================================
void ExpertPrefetcher::reset_stats() {
    for (auto & layer : impl_->layer_stats) {
        layer.total_predictions.store(0, std::memory_order_relaxed);
        layer.correct_predictions.store(0, std::memory_order_relaxed);
    }
}

// ============================================================================
// Print stats
// ============================================================================
void ExpertPrefetcher::print_stats() const {
    printf("=== Expert Prefetcher Stats ===\n");
    printf("Configured: %u layers, %u experts, %zu bytes/expert\n", impl_->num_layers, impl_->num_experts,
           impl_->expert_size);

    for (uint32_t i = 0; i < impl_->num_layers; i++) {
        const auto & stats = impl_->layer_stats[i];
        printf("Layer %u: accuracy=%.1f%% (%lu/%lu)\n", i, stats.accuracy() * 100.0f, stats.correct_predictions.load(),
               stats.total_predictions.load());
    }

    auto aggregate = get_stats();
    printf("Overall: accuracy=%.1f%% (%lu/%lu)\n", aggregate.overall_accuracy * 100.0f, aggregate.total_correct,
           aggregate.total_predictions);
}

}  // namespace ggml_sycl
