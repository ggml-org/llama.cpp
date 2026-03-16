#include "gguf-model-data.h"

#include <cstdio>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (line %d): %s\n", #cond, __LINE__, msg); \
            return 1; \
        } \
    } while (0)

int main() {
    fprintf(stderr, "=== test-gguf-model-data ===\n");

    // Fetch Qwen3-0.6B Q8_0 metadata
    auto result = gguf_fetch_model_meta("ggml-org/Qwen3-0.6B-GGUF", "Q8_0");

    if (!result.has_value()) {
        fprintf(stderr, "SKIP: could not fetch model metadata (no network or HTTP disabled)\n");
        return 0;
    }

    const auto & model = result.value();

    fprintf(stderr, "Architecture:  %s\n", model.architecture.c_str());
    fprintf(stderr, "n_embd:        %u\n", model.n_embd);
    fprintf(stderr, "n_ff:          %u\n", model.n_ff);
    fprintf(stderr, "n_vocab:       %u\n", model.n_vocab);
    fprintf(stderr, "n_layer:       %u\n", model.n_layer);
    fprintf(stderr, "n_head:        %u\n", model.n_head);
    fprintf(stderr, "n_head_kv:     %u\n", model.n_head_kv);
    fprintf(stderr, "n_expert:      %u\n", model.n_expert);
    fprintf(stderr, "n_embd_head_k: %u\n", model.n_embd_head_k);
    fprintf(stderr, "n_embd_head_v: %u\n", model.n_embd_head_v);
    fprintf(stderr, "tensors:       %zu\n", model.tensors.size());

    // Verify architecture
    TEST_ASSERT(model.architecture == "qwen3", "expected architecture 'qwen3'");

    // Verify key dimensions (Qwen3-0.6B)
    TEST_ASSERT(model.n_layer == 28, "expected n_layer == 28");
    TEST_ASSERT(model.n_embd == 1024, "expected n_embd == 1024");
    TEST_ASSERT(model.n_head == 16, "expected n_head == 16");
    TEST_ASSERT(model.n_head_kv == 8, "expected n_head_kv == 8");
    TEST_ASSERT(model.n_expert == 0, "expected n_expert == 0 (not MoE)");
    TEST_ASSERT(model.n_vocab == 151936, "expected n_vocab == 151936");

    // Verify tensor count
    TEST_ASSERT(model.tensors.size() == 311, "expected tensor count == 311");

    // Verify known tensor names exist
    bool found_attn_q = false;
    bool found_token_embd = false;
    bool found_output_norm = false;
    for (const auto & t : model.tensors) {
        if (t.name == "blk.0.attn_q.weight") {
            found_attn_q = true;
        }
        if (t.name == "token_embd.weight") {
            found_token_embd = true;
        }
        if (t.name == "output_norm.weight") {
            found_output_norm = true;
        }
    }
    TEST_ASSERT(found_attn_q, "expected tensor 'blk.0.attn_q.weight'");
    TEST_ASSERT(found_token_embd, "expected tensor 'token_embd.weight'");
    TEST_ASSERT(found_output_norm, "expected tensor 'output_norm.weight'");

    // Verify token_embd.weight shape
    for (const auto & t : model.tensors) {
        if (t.name == "token_embd.weight") {
            TEST_ASSERT(t.ne[0] == 1024, "expected token_embd.weight ne[0] == 1024");
            TEST_ASSERT(t.n_dims == 2, "expected token_embd.weight to be 2D");
            break;
        }
    }

    // Test that second call uses cache (just call again, it should work)
    auto result2 = gguf_fetch_model_meta("ggml-org/Qwen3-0.6B-GGUF", "Q8_0");
    TEST_ASSERT(result2.has_value(), "cached fetch should succeed");
    TEST_ASSERT(result2->tensors.size() == model.tensors.size(), "cached result should match");

    // Test a split MoE model without specifying quant (should default to Q8_0)
    auto result3 = gguf_fetch_model_meta("ggml-org/GLM-4.6V-GGUF");
    if (!result3.has_value()) {
        fprintf(stderr, "SKIP: could not fetch GLM-4.6V metadata (no network?)\n");
        return 0;
    }
    const auto & model3 = result3.value();

    fprintf(stderr, "Architecture:  %s\n", model3.architecture.c_str());
    fprintf(stderr, "n_embd:        %u\n", model3.n_embd);
    fprintf(stderr, "n_ff:          %u\n", model3.n_ff);
    fprintf(stderr, "n_vocab:       %u\n", model3.n_vocab);
    fprintf(stderr, "n_layer:       %u\n", model3.n_layer);
    fprintf(stderr, "n_head:        %u\n", model3.n_head);
    fprintf(stderr, "n_head_kv:     %u\n", model3.n_head_kv);
    fprintf(stderr, "n_expert:      %u\n", model3.n_expert);
    fprintf(stderr, "n_embd_head_k: %u\n", model3.n_embd_head_k);
    fprintf(stderr, "n_embd_head_v: %u\n", model3.n_embd_head_v);
    fprintf(stderr, "tensors:       %zu\n", model3.tensors.size());

    // Verify architecture
    TEST_ASSERT(model3.architecture == "glm4moe", "expected architecture 'glm4moe'");

    // Verify key dimensions (GLM-4.6V)
    TEST_ASSERT(model3.n_layer == 46, "expected n_layer == 46");
    TEST_ASSERT(model3.n_embd == 4096, "expected n_embd == 4096");
    TEST_ASSERT(model3.n_head == 96, "expected n_head == 96");
    TEST_ASSERT(model3.n_head_kv == 8, "expected n_head_kv == 8");
    TEST_ASSERT(model3.n_expert == 128, "expected n_expert == 128 (MoE)");
    TEST_ASSERT(model3.n_vocab == 151552, "expected n_vocab == 151552");

    // Verify tensor count
    TEST_ASSERT(model3.tensors.size() == 780, "expected tensor count == 780");

    fprintf(stderr, "=== ALL TESTS PASSED ===\n");
    return 0;
}
