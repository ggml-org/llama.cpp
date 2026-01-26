/**
 * @file test-heat-classifier.cpp
 * @brief Unit tests for tensor heat classification system
 *
 * Tests the TensorHeat classification based on tensor name patterns:
 * - HOT: attention tensors (attn, q_proj, k_proj, v_proj, o_proj)
 * - WARM: FFN tensors (ffn, mlp, gate, up_proj, down_proj)
 * - COLD: MoE expert tensors (expert, moe)
 * - FROZEN: embedding tensors (embed, token_embd, output)
 */

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Include the heat classifier header
#include "ggml-sycl/heat-classifier.hpp"

// Test assertion that works in release mode (unlike assert which is disabled with NDEBUG)
#define TEST_ASSERT(cond, msg)                                                              \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            fprintf(stderr, "ASSERTION FAILED: %s\n  at %s:%d\n", msg, __FILE__, __LINE__); \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

//
// Test utilities
//

static ggml_tensor create_mock_tensor_with_name(const char * name) {
    ggml_tensor t = {};
    t.type        = GGML_TYPE_F32;
    if (name && strlen(name) < GGML_MAX_NAME) {
        strncpy(t.name, name, GGML_MAX_NAME - 1);
        t.name[GGML_MAX_NAME - 1] = '\0';
    }
    return t;
}

//
// Classification tests
//

static void test_classify_attention() {
    printf("Testing attention tensor classification (HOT)...\n");

    // Test various attention-related tensor names
    const char * attn_names[] = {
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "blk.15.attn_norm.weight",
    };

    for (const char * name : attn_names) {
        ggml_tensor t    = create_mock_tensor_with_name(name);
        TensorHeat  heat = classify_tensor(&t);
        if (heat != TensorHeat::HOT) {
            printf("  FAIL: '%s' classified as %d, expected HOT (%d)\n", name, static_cast<int>(heat),
                   static_cast<int>(TensorHeat::HOT));
            TEST_ASSERT(false, "Attention tensor should be HOT");
        }
        printf("  PASS: '%s' -> HOT\n", name);
    }

    printf("  All attention tensor tests passed!\n\n");
}

static void test_classify_ffn() {
    printf("Testing FFN tensor classification (WARM)...\n");

    // Test various FFN-related tensor names
    const char * ffn_names[] = {
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_gate.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "blk.5.ffn_norm.weight",
    };

    for (const char * name : ffn_names) {
        ggml_tensor t    = create_mock_tensor_with_name(name);
        TensorHeat  heat = classify_tensor(&t);
        if (heat != TensorHeat::WARM) {
            printf("  FAIL: '%s' classified as %d, expected WARM (%d)\n", name, static_cast<int>(heat),
                   static_cast<int>(TensorHeat::WARM));
            TEST_ASSERT(false, "FFN tensor should be WARM");
        }
        printf("  PASS: '%s' -> WARM\n", name);
    }

    printf("  All FFN tensor tests passed!\n\n");
}

static void test_classify_moe_expert() {
    printf("Testing MoE expert tensor classification (COLD)...\n");

    // Test various MoE expert tensor names
    const char * moe_names[] = {
        "blk.0.ffn_gate_exps.3.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.1.w2.weight",
        "blk.5.expert.3.ffn_up.weight",
        "layers.0.moe.gate.weight",
    };

    for (const char * name : moe_names) {
        ggml_tensor t    = create_mock_tensor_with_name(name);
        TensorHeat  heat = classify_tensor(&t);
        if (heat != TensorHeat::COLD) {
            printf("  FAIL: '%s' classified as %d, expected COLD (%d)\n", name, static_cast<int>(heat),
                   static_cast<int>(TensorHeat::COLD));
            TEST_ASSERT(false, "MoE expert tensor should be COLD");
        }
        printf("  PASS: '%s' -> COLD\n", name);
    }

    printf("  All MoE expert tensor tests passed!\n\n");
}

static void test_classify_embedding() {
    printf("Testing embedding tensor classification (FROZEN)...\n");

    // Test various embedding-related tensor names
    const char * embed_names[] = {
        "token_embd.weight",
        "output.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",  // Often tied to output
        "embeddings.word_embeddings.weight",
    };

    for (const char * name : embed_names) {
        ggml_tensor t    = create_mock_tensor_with_name(name);
        TensorHeat  heat = classify_tensor(&t);
        if (heat != TensorHeat::FROZEN) {
            printf("  FAIL: '%s' classified as %d, expected FROZEN (%d)\n", name, static_cast<int>(heat),
                   static_cast<int>(TensorHeat::FROZEN));
            TEST_ASSERT(false, "Embedding tensor should be FROZEN");
        }
        printf("  PASS: '%s' -> FROZEN\n", name);
    }

    printf("  All embedding tensor tests passed!\n\n");
}

static void test_classify_unknown() {
    printf("Testing unknown tensor classification (default WARM)...\n");

    // Test various unknown tensor names - should default to WARM
    const char * unknown_names[] = {
        "random_tensor_name",
        "blk.0.something_else.weight",
        "layer_norm.weight",
        "bias.0",
        "",  // Empty name
    };

    for (const char * name : unknown_names) {
        ggml_tensor t    = create_mock_tensor_with_name(name);
        TensorHeat  heat = classify_tensor(&t);
        if (heat != TensorHeat::WARM) {
            printf("  FAIL: '%s' classified as %d, expected WARM (%d)\n", name, static_cast<int>(heat),
                   static_cast<int>(TensorHeat::WARM));
            TEST_ASSERT(false, "Unknown tensor should default to WARM");
        }
        printf("  PASS: '%s' -> WARM (default)\n", name[0] ? name : "(empty)");
    }

    // Test null tensor
    TensorHeat null_heat = classify_tensor(nullptr);
    if (null_heat != TensorHeat::WARM) {
        printf("  FAIL: nullptr classified as %d, expected WARM (%d)\n", static_cast<int>(null_heat),
               static_cast<int>(TensorHeat::WARM));
        TEST_ASSERT(false, "Null tensor should default to WARM");
    }
    printf("  PASS: nullptr -> WARM (default)\n");

    printf("  All unknown tensor tests passed!\n\n");
}

static void test_heat_policy_priority() {
    printf("Testing heat policy priorities...\n");

    TensorHeatPolicy hot_policy    = get_heat_policy(TensorHeat::HOT);
    TensorHeatPolicy warm_policy   = get_heat_policy(TensorHeat::WARM);
    TensorHeatPolicy cold_policy   = get_heat_policy(TensorHeat::COLD);
    TensorHeatPolicy frozen_policy = get_heat_policy(TensorHeat::FROZEN);

    // Print policies
    printf("  HOT:    priority=%.1f, can_stream=%d\n", hot_policy.eviction_priority, hot_policy.can_stream);
    printf("  WARM:   priority=%.1f, can_stream=%d\n", warm_policy.eviction_priority, warm_policy.can_stream);
    printf("  COLD:   priority=%.1f, can_stream=%d\n", cold_policy.eviction_priority, cold_policy.can_stream);
    printf("  FROZEN: priority=%.1f, can_stream=%d\n", frozen_policy.eviction_priority, frozen_policy.can_stream);

    // Verify priority ordering: HOT > WARM > FROZEN > COLD
    TEST_ASSERT(hot_policy.eviction_priority > warm_policy.eviction_priority, "HOT priority should be > WARM priority");
    TEST_ASSERT(warm_policy.eviction_priority > frozen_policy.eviction_priority,
                "WARM priority should be > FROZEN priority");
    TEST_ASSERT(frozen_policy.eviction_priority > cold_policy.eviction_priority,
                "FROZEN priority should be > COLD priority");

    printf("  PASS: Priority ordering HOT > WARM > FROZEN > COLD\n");

    // Verify streaming policies
    TEST_ASSERT(!hot_policy.can_stream, "HOT tensors should not stream");
    TEST_ASSERT(!warm_policy.can_stream, "WARM tensors should not stream");
    TEST_ASSERT(cold_policy.can_stream, "COLD tensors can stream on-demand");
    TEST_ASSERT(frozen_policy.can_stream, "FROZEN tensors can stream (one-time load)");

    printf("  PASS: Streaming policies (HOT/WARM=no, COLD/FROZEN=yes)\n");

    // Verify heat values match policies
    TEST_ASSERT(hot_policy.heat == TensorHeat::HOT, "HOT policy should have HOT heat");
    TEST_ASSERT(warm_policy.heat == TensorHeat::WARM, "WARM policy should have WARM heat");
    TEST_ASSERT(cold_policy.heat == TensorHeat::COLD, "COLD policy should have COLD heat");
    TEST_ASSERT(frozen_policy.heat == TensorHeat::FROZEN, "FROZEN policy should have FROZEN heat");

    printf("  PASS: Heat values in policies match\n");

    printf("  All heat policy tests passed!\n\n");
}

static void test_classification_priority_order() {
    printf("Testing classification priority (more specific patterns first)...\n");

    // MoE gate should be COLD (expert pattern) not WARM (gate pattern)
    // This tests that expert/moe checks happen before ffn/gate checks
    {
        ggml_tensor t    = create_mock_tensor_with_name("blk.0.ffn_gate_exps.weight");
        TensorHeat  heat = classify_tensor(&t);
        TEST_ASSERT(heat == TensorHeat::COLD, "MoE gate tensor should be COLD (expert pattern takes priority)");
        printf("  PASS: 'blk.0.ffn_gate_exps.weight' -> COLD (expert > gate)\n");
    }

    // MoE up should be COLD not WARM
    {
        ggml_tensor t    = create_mock_tensor_with_name("blk.0.ffn_up_exps.weight");
        TensorHeat  heat = classify_tensor(&t);
        TEST_ASSERT(heat == TensorHeat::COLD, "MoE up tensor should be COLD (expert pattern takes priority)");
        printf("  PASS: 'blk.0.ffn_up_exps.weight' -> COLD (expert > up_proj)\n");
    }

    printf("  All classification priority tests passed!\n\n");
}

//
// Main
//

int main() {
    printf("=== Tensor Heat Classifier Tests ===\n\n");

    // Run all tests
    test_classify_attention();
    test_classify_ffn();
    test_classify_moe_expert();
    test_classify_embedding();
    test_classify_unknown();
    test_heat_policy_priority();
    test_classification_priority_order();

    printf("=== All tests passed! ===\n");
    return 0;
}
