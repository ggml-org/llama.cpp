//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

// Transfer Learning Manager: Bootstrap tuning for new models from similar existing caches
//
// This header provides:
// - ModelFingerprint: Model characteristics for similarity comparison
// - model_similarity(): Compute similarity score between two models
// - TransferableEntry: Tuning entry with confidence decay
// - TransferLearningManager: Register models and find transferable caches

#include "tuning-engine.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

// Use TuningKey and TunedParams from tuning-engine.hpp
using ggml_sycl_tuning::BatchBucket;
using ggml_sycl_tuning::TunedParams;
using ggml_sycl_tuning::TuningKey;

// =============================================================================
// ModelFingerprint: Model characteristics for similarity comparison
// =============================================================================
// Captures the key architectural properties that affect kernel performance.
// Models with similar fingerprints tend to have similar optimal configurations.

struct ModelFingerprint {
    std::string name;
    int         n_embd     = 0;  // Hidden dimension (embedding size)
    int         n_head     = 0;  // Number of attention heads
    int         n_layer    = 0;  // Number of transformer layers
    int         vocab_size = 0;  // Vocabulary size
    int         ffn_dim    = 0;  // FFN intermediate dimension
    int         quant_type = 0;  // GGML_TYPE_* enum value

    // Generate a unique string key for this fingerprint
    std::string to_string() const {
        return name + "_" + std::to_string(n_embd) + "_" + std::to_string(n_layer) + "_" + std::to_string(quant_type);
    }
};

// =============================================================================
// model_similarity(): Compute similarity score between two models
// =============================================================================
// Returns a score from 0.0 (completely different) to 1.0 (identical).
// Uses weighted distance based on key architectural properties.
//
// Weight factors:
// - n_embd (40%): Most important for kernel tile sizing
// - quant_type (30%): Affects memory access patterns
// - n_layer (15%): Affects overall model structure
// - n_head (15%): Affects attention computation
//
// The distance is converted to similarity using exponential decay:
// similarity = exp(-2 * distance)

inline float model_similarity(const ModelFingerprint & a, const ModelFingerprint & b) {
    // Weight factors for each dimension
    constexpr float W_EMBD  = 0.4f;
    constexpr float W_QUANT = 0.3f;
    constexpr float W_LAYER = 0.15f;
    constexpr float W_HEAD  = 0.15f;

    // Normalized differences (0.0 = same, 1.0 = maximally different)
    float d_embd = 0.0f;
    if (a.n_embd > 0 && b.n_embd > 0) {
        d_embd = std::abs(a.n_embd - b.n_embd) / static_cast<float>(std::max(a.n_embd, b.n_embd));
    }

    float d_layer = 0.0f;
    if (a.n_layer > 0 && b.n_layer > 0) {
        d_layer = std::abs(a.n_layer - b.n_layer) / static_cast<float>(std::max(a.n_layer, b.n_layer));
    }

    float d_head = 0.0f;
    if (a.n_head > 0 && b.n_head > 0) {
        d_head = std::abs(a.n_head - b.n_head) / static_cast<float>(std::max(a.n_head, b.n_head));
    }

    // Quant type: exact match = 0, different = 1
    float d_quant = (a.quant_type == b.quant_type) ? 0.0f : 1.0f;

    // Weighted distance
    float distance = W_EMBD * d_embd + W_QUANT * d_quant + W_LAYER * d_layer + W_HEAD * d_head;

    // Convert to similarity using exponential decay
    // exp(-2 * 0) = 1.0 (identical)
    // exp(-2 * 0.5) = 0.37 (half distance)
    // exp(-2 * 1.0) = 0.14 (max distance)
    return std::exp(-2.0f * distance);
}

// =============================================================================
// TransferableEntry: Tuning entry with confidence decay
// =============================================================================
// When transferring tuning parameters from a similar model, confidence is
// reduced to reflect uncertainty about whether the same params are optimal.

struct TransferableEntry {
    TuningKey   key;
    TunedParams params;
    float       original_confidence;     // Confidence from source model
    float       transferred_confidence;  // Decayed confidence after transfer
    std::string source_model;            // Name of model params came from
};

// =============================================================================
// TransferLearningManager: Register models and find transferable caches
// =============================================================================
// Maintains a registry of model fingerprints and their tuning caches.
// When a new model is loaded, finds the most similar registered model
// and transfers its tuning parameters with decayed confidence.

class TransferLearningManager {
  public:
    // Configuration thresholds
    static constexpr float SIMILARITY_THRESHOLD = 0.8f;  // Minimum similarity to transfer
    static constexpr float CONFIDENCE_DECAY     = 0.5f;  // 50% confidence for transferred params
    static constexpr int   RETUNE_THRESHOLD     = 100;   // Re-tune after 100 tokens if underperforming

    TransferLearningManager() = default;

    // Register a model's tuning cache for future transfer
    void register_model(const ModelFingerprint &                               fingerprint,
                        const std::vector<std::pair<TuningKey, TunedParams>> & entries) {
        model_caches_[fingerprint.to_string()] = { fingerprint, entries };
    }

    // Find transferable cache entries for a new model
    // Returns empty vector if no similar model found (below threshold)
    std::vector<TransferableEntry> find_transferable_cache(const ModelFingerprint & target) {
        std::vector<TransferableEntry> result;

        float       best_similarity = 0.0f;
        std::string best_model;

        // Find most similar registered model
        for (const auto & [name, cache] : model_caches_) {
            float sim = model_similarity(target, cache.fingerprint);
            if (sim > best_similarity) {
                best_similarity = sim;
                best_model      = name;
            }
        }

        // Only transfer if similarity meets threshold
        if (best_similarity >= SIMILARITY_THRESHOLD && !best_model.empty()) {
            const auto & source = model_caches_[best_model];

            for (const auto & [key, params] : source.entries) {
                TransferableEntry entry;
                entry.key                    = key;
                entry.params                 = params;
                entry.original_confidence    = params.confidence;
                entry.transferred_confidence = params.confidence * CONFIDENCE_DECAY;
                entry.source_model           = source.fingerprint.name;
                result.push_back(entry);
            }
        }

        return result;
    }

    // Get similarity score to best matching registered model
    float get_best_similarity(const ModelFingerprint & target) const {
        float best = 0.0f;
        for (const auto & [name, cache] : model_caches_) {
            best = std::max(best, model_similarity(target, cache.fingerprint));
        }
        return best;
    }

    // Check if a model with given name has a registered cache
    bool has_cache(const std::string & model_name) const {
        for (const auto & [name, cache] : model_caches_) {
            if (cache.fingerprint.name == model_name) {
                return true;
            }
        }
        return false;
    }

    // Get number of registered models
    size_t get_model_count() const { return model_caches_.size(); }

    // Clear all registered caches
    void clear() { model_caches_.clear(); }

  private:
    // Internal cache storage per model
    struct ModelCache {
        ModelFingerprint                               fingerprint;
        std::vector<std::pair<TuningKey, TunedParams>> entries;
    };

    std::unordered_map<std::string, ModelCache> model_caches_;
};

}  // namespace ggml_sycl
