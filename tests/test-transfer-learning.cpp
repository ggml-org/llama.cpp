//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Unit tests for transfer-learning.hpp
// Tests model similarity computation, fingerprint generation, and cache transfer

#include "../ggml/src/ggml-sycl/transfer-learning.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace ggml_sycl;
using namespace ggml_sycl_tuning;

// Constants for GGML quantization types (from ggml.h)
// Redefined here to avoid full ggml.h dependency
[[maybe_unused]] constexpr int GGML_TYPE_F32  = 0;
constexpr int                  GGML_TYPE_F16  = 1;
constexpr int                  GGML_TYPE_Q4_0 = 2;
constexpr int                  GGML_TYPE_Q8_0 = 8;

// Helper to check float equality with tolerance
inline bool float_eq(float a, float b, float eps = 1e-6f) {
    return std::abs(a - b) < eps;
}

// =============================================================================
// Test 1: Identical models have similarity 1.0
// =============================================================================
bool test_identical_similarity() {
    ModelFingerprint a;
    a.name       = "mistral-7b";
    a.n_embd     = 4096;
    a.n_head     = 32;
    a.n_layer    = 32;
    a.vocab_size = 32000;
    a.ffn_dim    = 14336;
    a.quant_type = GGML_TYPE_Q4_0;

    float sim = model_similarity(a, a);
    assert(float_eq(sim, 1.0f));

    std::cout << "[PASS] test_identical_similarity: sim=" << sim << "\n";
    return true;
}

// =============================================================================
// Test 2: Different quant type reduces similarity
// =============================================================================
bool test_quant_difference() {
    ModelFingerprint a;
    a.name       = "model";
    a.n_embd     = 4096;
    a.n_head     = 32;
    a.n_layer    = 32;
    a.vocab_size = 32000;
    a.ffn_dim    = 14336;
    a.quant_type = GGML_TYPE_Q4_0;

    ModelFingerprint b = a;
    b.quant_type       = GGML_TYPE_Q8_0;

    float sim = model_similarity(a, b);
    // Different quant = 0.3 penalty, so distance = 0.3, sim = exp(-0.6) = 0.549
    assert(sim < 0.9f);
    assert(sim > 0.5f);

    std::cout << "[PASS] test_quant_difference: sim=" << sim << "\n";
    return true;
}

// =============================================================================
// Test 3: Different n_embd reduces similarity
// =============================================================================
bool test_embd_difference() {
    ModelFingerprint a;
    a.name       = "model";
    a.n_embd     = 4096;
    a.n_head     = 32;
    a.n_layer    = 32;
    a.vocab_size = 32000;
    a.ffn_dim    = 14336;
    a.quant_type = GGML_TYPE_Q4_0;

    ModelFingerprint b = a;
    b.n_embd           = 8192;  // 50% larger

    float sim = model_similarity(a, b);
    // 50% difference in n_embd = 0.5 distance, weighted 0.4 = 0.2
    // sim = exp(-0.4) = 0.67
    assert(sim < 0.8f);
    assert(sim > 0.5f);

    std::cout << "[PASS] test_embd_difference: sim=" << sim << "\n";
    return true;
}

// =============================================================================
// Test 4: Similar models (Mistral 7B vs LLaMA 7B)
// =============================================================================
bool test_similar_models() {
    ModelFingerprint mistral;
    mistral.name       = "mistral-7b";
    mistral.n_embd     = 4096;
    mistral.n_head     = 32;
    mistral.n_layer    = 32;
    mistral.vocab_size = 32000;
    mistral.ffn_dim    = 14336;
    mistral.quant_type = GGML_TYPE_Q4_0;

    ModelFingerprint llama7b;
    llama7b.name       = "llama-7b";
    llama7b.n_embd     = 4096;
    llama7b.n_head     = 32;
    llama7b.n_layer    = 32;
    llama7b.vocab_size = 32000;
    llama7b.ffn_dim    = 11008;  // Different FFN dim (not used in similarity)
    llama7b.quant_type = GGML_TYPE_Q4_0;

    float sim = model_similarity(mistral, llama7b);
    // Identical core architecture, same quant type
    assert(sim > 0.9f);

    std::cout << "[PASS] test_similar_models: sim=" << sim << "\n";
    return true;
}

// =============================================================================
// Test 5: Transfer learning manager registration
// =============================================================================
bool test_registration() {
    TransferLearningManager mgr;

    ModelFingerprint fp;
    fp.name       = "test-model";
    fp.n_embd     = 4096;
    fp.n_head     = 32;
    fp.n_layer    = 32;
    fp.vocab_size = 32000;
    fp.ffn_dim    = 14336;
    fp.quant_type = GGML_TYPE_Q4_0;

    std::vector<std::pair<TuningKey, TunedParams>> entries;
    mgr.register_model(fp, entries);

    assert(mgr.get_model_count() == 1);
    assert(mgr.has_cache("test-model") == true);
    assert(mgr.has_cache("nonexistent") == false);

    std::cout << "[PASS] test_registration\n";
    return true;
}

// =============================================================================
// Test 6: Find transferable cache for similar model
// =============================================================================
bool test_find_transferable() {
    TransferLearningManager mgr;

    // Register source model with tuning entry
    ModelFingerprint source;
    source.name       = "mistral-7b";
    source.n_embd     = 4096;
    source.n_head     = 32;
    source.n_layer    = 32;
    source.vocab_size = 32000;
    source.ffn_dim    = 14336;
    source.quant_type = GGML_TYPE_Q4_0;

    TuningKey key;
    key.quant_type   = GGML_TYPE_Q4_0;
    key.batch_bucket = BatchBucket::SINGLE;
    key.K            = 4096;
    key.N            = 4096;

    TunedParams params;
    params.tile_m     = 16;
    params.tile_n     = 64;
    params.tile_k     = 32;
    params.confidence = 0.9f;

    mgr.register_model(source, {
                                   { key, params }
    });

    // Find for similar target (same architecture, same quant)
    ModelFingerprint target;
    target.name       = "llama-7b";
    target.n_embd     = 4096;
    target.n_head     = 32;
    target.n_layer    = 32;
    target.vocab_size = 32000;
    target.ffn_dim    = 11008;
    target.quant_type = GGML_TYPE_Q4_0;

    auto transferred = mgr.find_transferable_cache(target);

    assert(transferred.size() == 1);
    assert(transferred[0].params.tile_m == 16);
    assert(transferred[0].params.tile_n == 64);
    assert(transferred[0].source_model == "mistral-7b");
    // Confidence decayed by 50%: 0.9 * 0.5 = 0.45
    assert(float_eq(transferred[0].transferred_confidence, 0.45f, 0.01f));

    std::cout << "[PASS] test_find_transferable\n";
    return true;
}

// =============================================================================
// Test 7: No transfer for dissimilar models
// =============================================================================
bool test_no_transfer_dissimilar() {
    TransferLearningManager mgr;

    ModelFingerprint source;
    source.name       = "mistral-7b";
    source.n_embd     = 4096;
    source.n_head     = 32;
    source.n_layer    = 32;
    source.vocab_size = 32000;
    source.ffn_dim    = 14336;
    source.quant_type = GGML_TYPE_Q4_0;

    TuningKey   key{ GGML_TYPE_Q4_0, BatchBucket::SINGLE, 4096, 4096 };
    TunedParams params;
    params.tile_m = 16;
    mgr.register_model(source, {
                                   { key, params }
    });

    // Very different target (GPT-3 scale, F16)
    ModelFingerprint target;
    target.name       = "gpt-3";
    target.n_embd     = 12288;  // 3x larger
    target.n_head     = 96;     // 3x more heads
    target.n_layer    = 96;     // 3x more layers
    target.vocab_size = 50000;
    target.ffn_dim    = 49152;
    target.quant_type = GGML_TYPE_F16;  // Different quant

    auto transferred = mgr.find_transferable_cache(target);

    // Should be empty (below similarity threshold)
    assert(transferred.empty());

    std::cout << "[PASS] test_no_transfer_dissimilar\n";
    return true;
}

// =============================================================================
// Test 8: Best similarity query
// =============================================================================
bool test_best_similarity() {
    TransferLearningManager mgr;

    ModelFingerprint m1;
    m1.name       = "model-a";
    m1.n_embd     = 4096;
    m1.n_head     = 32;
    m1.n_layer    = 32;
    m1.vocab_size = 32000;
    m1.ffn_dim    = 14336;
    m1.quant_type = GGML_TYPE_Q4_0;

    ModelFingerprint m2;
    m2.name       = "model-b";
    m2.n_embd     = 8192;  // Larger
    m2.n_head     = 64;
    m2.n_layer    = 64;
    m2.vocab_size = 32000;
    m2.ffn_dim    = 28672;
    m2.quant_type = GGML_TYPE_Q4_0;

    mgr.register_model(m1, {});
    mgr.register_model(m2, {});

    // Query with fingerprint similar to m1
    ModelFingerprint target;
    target.name       = "target";
    target.n_embd     = 4096;
    target.n_head     = 32;
    target.n_layer    = 32;
    target.vocab_size = 32000;
    target.ffn_dim    = 14336;
    target.quant_type = GGML_TYPE_Q4_0;

    float best = mgr.get_best_similarity(target);

    // Should be very similar to m1 (near 1.0)
    assert(best > 0.95f);

    std::cout << "[PASS] test_best_similarity: best=" << best << "\n";
    return true;
}

// =============================================================================
// Test 9: ModelFingerprint to_string
// =============================================================================
bool test_fingerprint_string() {
    ModelFingerprint fp;
    fp.name       = "mistral";
    fp.n_embd     = 4096;
    fp.n_head     = 32;
    fp.n_layer    = 32;
    fp.vocab_size = 32000;
    fp.ffn_dim    = 14336;
    fp.quant_type = GGML_TYPE_Q4_0;

    std::string s = fp.to_string();

    assert(s.find("mistral") != std::string::npos);
    assert(s.find("4096") != std::string::npos);
    assert(s.find("32") != std::string::npos);

    std::cout << "[PASS] test_fingerprint_string: \"" << s << "\"\n";
    return true;
}

// =============================================================================
// Test 10: Clear manager
// =============================================================================
bool test_clear() {
    TransferLearningManager mgr;

    ModelFingerprint fp;
    fp.name       = "test";
    fp.n_embd     = 4096;
    fp.n_head     = 32;
    fp.n_layer    = 32;
    fp.vocab_size = 32000;
    fp.ffn_dim    = 14336;
    fp.quant_type = GGML_TYPE_Q4_0;

    mgr.register_model(fp, {});

    assert(mgr.get_model_count() == 1);

    mgr.clear();
    assert(mgr.get_model_count() == 0);
    assert(mgr.has_cache("test") == false);

    std::cout << "[PASS] test_clear\n";
    return true;
}

// =============================================================================
// Test 11: Similarity symmetry (a,b) == (b,a)
// =============================================================================
bool test_similarity_symmetry() {
    ModelFingerprint a;
    a.name       = "model-a";
    a.n_embd     = 4096;
    a.n_head     = 32;
    a.n_layer    = 32;
    a.vocab_size = 32000;
    a.ffn_dim    = 14336;
    a.quant_type = GGML_TYPE_Q4_0;

    ModelFingerprint b;
    b.name       = "model-b";
    b.n_embd     = 5120;
    b.n_head     = 40;
    b.n_layer    = 40;
    b.vocab_size = 32000;
    b.ffn_dim    = 13824;
    b.quant_type = GGML_TYPE_Q8_0;

    float sim_ab = model_similarity(a, b);

    assert(float_eq(sim_ab, model_similarity(b, a)));

    std::cout << "[PASS] test_similarity_symmetry: sim=" << sim_ab << "\n";
    return true;
}

// =============================================================================
// Test 12: Multiple tuning entries transfer
// =============================================================================
bool test_multiple_entries_transfer() {
    TransferLearningManager mgr;

    ModelFingerprint source;
    source.name       = "source-model";
    source.n_embd     = 4096;
    source.n_head     = 32;
    source.n_layer    = 32;
    source.vocab_size = 32000;
    source.ffn_dim    = 14336;
    source.quant_type = GGML_TYPE_Q4_0;

    // Multiple tuning entries for different batch buckets
    std::vector<std::pair<TuningKey, TunedParams>> entries;

    TuningKey   key1{ GGML_TYPE_Q4_0, BatchBucket::SINGLE, 4096, 4096 };
    TunedParams params1;
    params1.tile_m     = 8;
    params1.confidence = 0.8f;
    entries.push_back({ key1, params1 });

    TuningKey   key2{ GGML_TYPE_Q4_0, BatchBucket::SMALL, 4096, 4096 };
    TunedParams params2;
    params2.tile_m     = 16;
    params2.confidence = 0.9f;
    entries.push_back({ key2, params2 });

    TuningKey   key3{ GGML_TYPE_Q4_0, BatchBucket::MEDIUM, 4096, 4096 };
    TunedParams params3;
    params3.tile_m     = 32;
    params3.confidence = 0.7f;
    entries.push_back({ key3, params3 });

    mgr.register_model(source, entries);

    // Similar target
    ModelFingerprint target;
    target.name       = "target-model";
    target.n_embd     = 4096;
    target.n_head     = 32;
    target.n_layer    = 32;
    target.vocab_size = 32000;
    target.ffn_dim    = 11008;
    target.quant_type = GGML_TYPE_Q4_0;

    auto transferred = mgr.find_transferable_cache(target);

    assert(transferred.size() == 3);

    // Verify all entries transferred with correct confidence decay
    for (const auto & entry : transferred) {
        // Use if-return instead of assert to avoid unused variable warning in Release builds
        if (entry.source_model != "source-model") {
            return false;
        }
        // Confidence should be halved
        if (!float_eq(entry.transferred_confidence, entry.original_confidence * 0.5f, 0.01f)) {
            return false;
        }
    }

    std::cout << "[PASS] test_multiple_entries_transfer: transferred " << transferred.size() << " entries\n";
    return true;
}

// =============================================================================
// Test 13: Zero dimensions handled gracefully
// =============================================================================
bool test_zero_dimensions() {
    ModelFingerprint a;
    a.name   = "partial";
    a.n_embd = 4096;
    // Leave other dimensions at 0

    ModelFingerprint b;
    b.name   = "also-partial";
    b.n_embd = 4096;

    float sim = model_similarity(a, b);
    // Should not crash, and identical non-zero dims should still match
    assert(sim > 0.5f);

    std::cout << "[PASS] test_zero_dimensions: sim=" << sim << "\n";
    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main() {
    int passed = 0;
    int failed = 0;

    std::cout << "=== Transfer Learning Unit Tests ===\n\n";

    auto run_test = [&](bool (*test_fn)(), const char * name) {
        try {
            if (test_fn()) {
                passed++;
            } else {
                std::cerr << "[FAIL] " << name << "\n";
                failed++;
            }
        } catch (const std::exception & e) {
            std::cerr << "[FAIL] " << name << " (exception: " << e.what() << ")\n";
            failed++;
        }
    };

    run_test(test_identical_similarity, "test_identical_similarity");
    run_test(test_quant_difference, "test_quant_difference");
    run_test(test_embd_difference, "test_embd_difference");
    run_test(test_similar_models, "test_similar_models");
    run_test(test_registration, "test_registration");
    run_test(test_find_transferable, "test_find_transferable");
    run_test(test_no_transfer_dissimilar, "test_no_transfer_dissimilar");
    run_test(test_best_similarity, "test_best_similarity");
    run_test(test_fingerprint_string, "test_fingerprint_string");
    run_test(test_clear, "test_clear");
    run_test(test_similarity_symmetry, "test_similarity_symmetry");
    run_test(test_multiple_entries_transfer, "test_multiple_entries_transfer");
    run_test(test_zero_dimensions, "test_zero_dimensions");

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}
