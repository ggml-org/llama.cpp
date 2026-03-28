// Test for custom attention mask API (llama_set_attn_mask)
// Usage: test-attn-mask <model.gguf>
//
// Tests:
// 1. No custom mask (baseline)
// 2. Custom mask = nullptr (should be identical to baseline)
// 3. Diagonal mask (window=2) — should produce different output
// 4. Clear mask — should restore baseline behavior
// 5. Full zeros mask (all visible) — should be identical to baseline (causal still applied)
// 6. Full -inf mask (nothing visible) — should differ from baseline
// 7. Dynamic mask change between two decodes — second mask should take effect

#include "llama.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

struct test_context {
    llama_model   * model;
    llama_context * ctx;
    const llama_vocab * vocab;

    std::vector<llama_token> tokens;
    int n_tokens;
};

static bool tokenize_prompt(test_context & tctx, const char * prompt) {
    std::vector<llama_token> buf(128);
    int n = llama_tokenize(tctx.vocab, prompt, strlen(prompt), buf.data(), 128, true, false);
    if (n < 0) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        return false;
    }
    buf.resize(n);
    tctx.tokens = buf;
    tctx.n_tokens = n;
    return true;
}

static std::vector<float> eval_prompt(test_context & tctx, const char * prompt) {
    if (!tokenize_prompt(tctx, prompt)) return {};

    // clear KV cache
    llama_memory_clear(llama_get_memory(tctx.ctx), true);

    // create batch and evaluate
    llama_batch batch = llama_batch_get_one(tctx.tokens.data(), tctx.n_tokens);
    if (llama_decode(tctx.ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode\n");
        return {};
    }

    // get logits for last token
    float * logits = llama_get_logits_ith(tctx.ctx, tctx.n_tokens - 1);
    int n_vocab = llama_vocab_n_tokens(tctx.vocab);

    return std::vector<float>(logits, logits + n_vocab);
}

static float logits_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) return -1.0f;

    double sum_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = (double)a[i] - (double)b[i];
        sum_diff += d * d;
    }
    return (float)std::sqrt(sum_diff / a.size());
}

// Helper: build position array [0, 1, 2, ..., n-1]
static std::vector<llama_pos> make_positions(int n) {
    std::vector<llama_pos> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = i;
    return pos;
}

// Helper: build mask where mask[i][j] = val for all i,j
static std::vector<float> make_uniform_mask(int n, float val) {
    return std::vector<float>(n * n, val);
}

// Helper: build causal sliding window mask
static std::vector<float> make_window_mask(int n, int window) {
    std::vector<float> mask(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j <= i && (i - j) <= window) {
                mask[i * n + j] = 0.0f;    // visible
            } else {
                mask[i * n + j] = -INFINITY; // masked
            }
        }
    }
    return mask;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * prompt = "The capital of France is";

    // load model
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    // create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 128;
    cparams.n_batch = 128;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    test_context tctx = { model, ctx, llama_model_get_vocab(model), {}, 0 };

    int n_passed = 0;
    int n_failed = 0;

    auto pass = [&](const char * msg) { printf("PASS: %s\n", msg); n_passed++; };
    auto fail = [&](const char * msg) { printf("FAIL: %s\n", msg); n_failed++; };

    // Pre-tokenize to know n_tokens for mask construction
    if (!tokenize_prompt(tctx, prompt)) {
        fprintf(stderr, "Failed to tokenize prompt, aborting\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    const int n_tok = tctx.n_tokens;
    printf("Prompt: \"%s\" → %d tokens\n", prompt, n_tok);

    // === Test 1: Baseline (no custom mask) ===
    printf("\n=== Test 1: Baseline (no custom mask) ===\n");
    auto logits_baseline = eval_prompt(tctx, prompt);
    if (logits_baseline.empty()) {
        fail("could not get baseline logits");
    } else {
        char buf[128];
        snprintf(buf, sizeof(buf), "baseline logits obtained (%zu values)", logits_baseline.size());
        pass(buf);
    }

    // === Test 2: Custom mask = nullptr (should be identical) ===
    printf("\n=== Test 2: Custom mask = nullptr ===\n");
    llama_set_attn_mask(ctx, nullptr, nullptr, 0);
    {
        auto logits = eval_prompt(tctx, prompt);
        float diff = logits_diff(logits_baseline, logits);
        printf("  diff vs baseline: %.6e\n", diff);
        if (diff < 1e-5f) {
            pass("nullptr mask produces identical output");
        } else {
            char buf[128]; snprintf(buf, sizeof(buf), "nullptr mask should produce identical output (diff=%.6e)", diff);
            fail(buf);
        }
    }

    // === Test 3: Diagonal mask (window=2) — should differ ===
    printf("\n=== Test 3: Diagonal mask (window=2) ===\n");
    {
        auto positions = make_positions(n_tok);
        auto mask = make_window_mask(n_tok, 2);

        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_tok);
        auto logits = eval_prompt(tctx, prompt);
        float diff = logits_diff(logits_baseline, logits);
        printf("  diff vs baseline: %.6e\n", diff);

        if (diff > 0.1f) {
            char buf[128]; snprintf(buf, sizeof(buf), "diagonal mask produces different output (diff=%.6e)", diff);
            pass(buf);
        } else {
            char buf[128]; snprintf(buf, sizeof(buf), "diagonal mask should produce different output (diff=%.6e)", diff);
            fail(buf);
        }
    }

    // === Test 4: Clear mask — should restore baseline ===
    printf("\n=== Test 4: Clear mask (restore baseline) ===\n");
    llama_set_attn_mask(ctx, nullptr, nullptr, 0);
    {
        auto logits = eval_prompt(tctx, prompt);
        float diff = logits_diff(logits_baseline, logits);
        printf("  diff vs baseline: %.6e\n", diff);
        if (diff < 1e-5f) {
            pass("cleared mask restores baseline");
        } else {
            char buf[128]; snprintf(buf, sizeof(buf), "cleared mask should restore baseline (diff=%.6e)", diff);
            fail(buf);
        }
    }

    // === Test 5: Full zeros mask (all visible) — causal mask still applied underneath ===
    // With AND logic, custom 0.0 + causal = causal. So output should match baseline.
    printf("\n=== Test 5: Full zeros mask (all visible) ===\n");
    {
        auto positions = make_positions(n_tok);
        auto mask = make_uniform_mask(n_tok, 0.0f);

        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_tok);
        auto logits = eval_prompt(tctx, prompt);
        float diff = logits_diff(logits_baseline, logits);
        printf("  diff vs baseline: %.6e\n", diff);

        if (diff < 1e-5f) {
            pass("full-zeros mask identical to baseline (causal still applied)");
        } else {
            char buf[128]; snprintf(buf, sizeof(buf), "full-zeros mask should be identical to baseline (diff=%.6e)", diff);
            fail(buf);
        }
    }

    // === Test 6: Full -inf mask (block everything except self-attention via causal) ===
    // Every position is masked by the custom mask. Only the diagonal survives from the causal mask
    // (where token sees itself). This should produce very different output.
    printf("\n=== Test 6: Full -inf mask (block all cross-attention) ===\n");
    {
        auto positions = make_positions(n_tok);
        auto mask = make_uniform_mask(n_tok, -INFINITY);

        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_tok);
        auto logits = eval_prompt(tctx, prompt);
        float diff = logits_diff(logits_baseline, logits);
        printf("  diff vs baseline: %.6e\n", diff);

        if (diff > 0.1f) {
            char buf[128]; snprintf(buf, sizeof(buf), "full-inf mask produces different output (diff=%.6e)", diff);
            pass(buf);
        } else {
            char buf[128]; snprintf(buf, sizeof(buf), "full-inf mask should produce different output (diff=%.6e)", diff);
            fail(buf);
        }
    }

    // === Test 7: Dynamic mask change between two decodes ===
    // First decode with window=1 (very restrictive), then window=4 (less restrictive).
    // Both should differ from baseline, and they should differ from each other.
    printf("\n=== Test 7: Dynamic mask change between decodes ===\n");
    {
        auto positions = make_positions(n_tok);

        // Decode with window=1
        auto mask1 = make_window_mask(n_tok, 1);
        llama_set_attn_mask(ctx, mask1.data(), positions.data(), n_tok);
        auto logits_w1 = eval_prompt(tctx, prompt);

        // Decode with window=4
        auto mask2 = make_window_mask(n_tok, 4);
        llama_set_attn_mask(ctx, mask2.data(), positions.data(), n_tok);
        auto logits_w4 = eval_prompt(tctx, prompt);

        float diff_w1_base = logits_diff(logits_baseline, logits_w1);
        float diff_w4_base = logits_diff(logits_baseline, logits_w4);
        float diff_w1_w4   = logits_diff(logits_w1, logits_w4);

        printf("  window=1 vs baseline: %.6e\n", diff_w1_base);
        printf("  window=4 vs baseline: %.6e\n", diff_w4_base);
        printf("  window=1 vs window=4: %.6e\n", diff_w1_w4);

        bool ok = true;
        if (diff_w1_base <= 0.1f) {
            fail("window=1 should differ from baseline"); ok = false;
        }
        if (diff_w4_base <= 0.01f) {
            // window=4 on a 6-token prompt might be close to causal, but should still differ
            // (token at pos 5 can normally see pos 0, but window=4 blocks it)
            fail("window=4 should differ from baseline"); ok = false;
        }
        if (diff_w1_w4 <= 0.01f) {
            fail("window=1 and window=4 should produce different output"); ok = false;
        }
        if (ok) {
            pass("dynamic mask change correctly applies different masks across decodes");
        }
    }

    // Clear mask for clean exit
    llama_set_attn_mask(ctx, nullptr, nullptr, 0);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", n_passed, n_failed);
    printf("========================================\n");

    // cleanup
    llama_free(ctx);
    llama_model_free(model);

    return n_failed > 0 ? 1 : 0;
}
