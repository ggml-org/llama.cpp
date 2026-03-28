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
// 8. Per-head broadcast (n_head_groups=0) — identical to baseline
// 9. Per-head mask (n_head_groups=n_head) — different from broadcast
// 10. Grouped heads (n_head_groups=n_head/n_kv_head) — different from both
// 11. Clear after per-head — restore baseline
// 12. Multi-slot: per-head on slot 0, broadcast on global
// 13. Per-head uniform zeros — identical to baseline (causal still applied per-head)

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

// Helper: build per-head window mask where each head has a different window size
// mask layout: [n_heads * n * n], head h has window = min_window + h * (max_window - min_window) / (n_heads - 1)
static std::vector<float> make_perhead_window_mask(int n, int n_heads, int min_window, int max_window) {
    std::vector<float> mask(n_heads * n * n);
    for (int h = 0; h < n_heads; ++h) {
        int window = (n_heads > 1)
            ? min_window + h * (max_window - min_window) / (n_heads - 1)
            : min_window;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float val;
                if (j <= i && (i - j) <= window) {
                    val = 0.0f;    // visible
                } else {
                    val = -INFINITY; // masked
                }
                mask[h * n * n + i * n + j] = val;
            }
        }
    }
    return mask;
}

// Helper: build per-head uniform mask (same value for all positions, per head)
static std::vector<float> make_perhead_uniform_mask(int n, int n_heads, float val) {
    return std::vector<float>(n_heads * n * n, val);
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
    const int n_head    = llama_model_n_head(model);
    const int n_head_kv = llama_model_n_head_kv(model);
    printf("Prompt: \"%s\" → %d tokens\n", prompt, n_tok);
    printf("Model: n_head=%d, n_head_kv=%d\n", n_head, n_head_kv);

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
    llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);
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

        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_tok, 0, -1);
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
    llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);
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

        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_tok, 0, -1);
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

        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_tok, 0, -1);
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
        llama_set_attn_mask(ctx, mask1.data(), positions.data(), n_tok, 0, -1);
        auto logits_w1 = eval_prompt(tctx, prompt);

        // Decode with window=4
        auto mask2 = make_window_mask(n_tok, 4);
        llama_set_attn_mask(ctx, mask2.data(), positions.data(), n_tok, 0, -1);
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

    // =============================================
    // Per-head mask tests (require n_head from model)
    // =============================================

    // === Test 8: Per-head broadcast (n_head_groups=0) — should be identical to baseline ===
    printf("\n=== Test 8: Per-head broadcast (n_head_groups=0, explicit) ===\n");
    {
        auto positions = make_positions(n_tok);
        auto mask = make_uniform_mask(n_tok, 0.0f);

        // n_head_groups=0 means broadcast — same mask for all heads
        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_tok, 0, -1);
        auto logits = eval_prompt(tctx, prompt);
        float diff = logits_diff(logits_baseline, logits);
        printf("  diff vs baseline: %.6e\n", diff);
        if (diff < 1e-5f) {
            pass("per-head broadcast (n_head_groups=0) identical to baseline");
        } else {
            char buf[128]; snprintf(buf, sizeof(buf), "per-head broadcast should match baseline (diff=%.6e)", diff);
            fail(buf);
        }
    }
    llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);

    // === Test 9: Per-head mask (n_head_groups=n_head) — each head gets different window ===
    printf("\n=== Test 9: Per-head mask (n_head_groups=%d) ===\n", n_head);
    {
        auto positions = make_positions(n_tok);
        // Head 0 gets window=1 (very restrictive), head n_head-1 gets window=n_tok (full causal)
        auto mask = make_perhead_window_mask(n_tok, n_head, 1, n_tok);

        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_tok, n_head, -1);
        auto logits_perhead = eval_prompt(tctx, prompt);

        // Compare vs broadcast with window=1
        auto mask_bcast = make_window_mask(n_tok, 1);
        llama_set_attn_mask(ctx, mask_bcast.data(), positions.data(), n_tok, 0, -1);
        auto logits_bcast_w1 = eval_prompt(tctx, prompt);

        float diff_perhead_bcast = logits_diff(logits_perhead, logits_bcast_w1);
        float diff_perhead_base  = logits_diff(logits_perhead, logits_baseline);

        printf("  per-head vs broadcast(w=1): %.6e\n", diff_perhead_bcast);
        printf("  per-head vs baseline:       %.6e\n", diff_perhead_base);

        bool ok = true;
        if (diff_perhead_bcast <= 0.01f) {
            fail("per-head should differ from uniform broadcast window=1"); ok = false;
        }
        if (diff_perhead_base <= 0.01f) {
            fail("per-head should differ from baseline (no mask)"); ok = false;
        }
        if (ok) {
            char buf[256]; snprintf(buf, sizeof(buf),
                "per-head mask produces distinct output (vs bcast=%.3e, vs base=%.3e)",
                diff_perhead_bcast, diff_perhead_base);
            pass(buf);
        }
    }
    llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);

    // === Test 10: Grouped heads (n_head_groups = n_head/n_head_kv = GQA ratio) ===
    {
        const int n_groups = (n_head_kv > 0 && n_head % n_head_kv == 0)
            ? n_head / n_head_kv : 0;

        if (n_groups > 1) {
            printf("\n=== Test 10: Grouped heads (n_head_groups=%d, GQA ratio) ===\n", n_groups);

            auto positions = make_positions(n_tok);
            auto mask_grouped = make_perhead_window_mask(n_tok, n_groups, 1, n_tok);

            llama_set_attn_mask(ctx, mask_grouped.data(), positions.data(), n_tok, n_groups, -1);
            auto logits_grouped = eval_prompt(tctx, prompt);

            // Also get per-head result for comparison
            auto mask_full = make_perhead_window_mask(n_tok, n_head, 1, n_tok);
            llama_set_attn_mask(ctx, mask_full.data(), positions.data(), n_tok, n_head, -1);
            auto logits_full = eval_prompt(tctx, prompt);

            float diff_grouped_base = logits_diff(logits_grouped, logits_baseline);
            float diff_grouped_full = logits_diff(logits_grouped, logits_full);

            printf("  grouped vs baseline:  %.6e\n", diff_grouped_base);
            printf("  grouped vs full per-head: %.6e\n", diff_grouped_full);

            bool ok = true;
            if (diff_grouped_base <= 0.01f) {
                fail("grouped heads should differ from baseline"); ok = false;
            }
            if (diff_grouped_full <= 0.001f) {
                // grouped and full per-head use different granularity, should differ
                fail("grouped heads should differ from full per-head"); ok = false;
            }
            if (ok) {
                char buf[256]; snprintf(buf, sizeof(buf),
                    "grouped heads (%d groups) produces distinct output (vs base=%.3e, vs full=%.3e)",
                    n_groups, diff_grouped_base, diff_grouped_full);
                pass(buf);
            }
        } else {
            printf("\n=== Test 10: Grouped heads — SKIPPED (n_head=%d, n_head_kv=%d, no valid grouping) ===\n",
                   n_head, n_head_kv);
            pass("grouped heads test skipped (model has no GQA or invalid grouping)");
        }
    }
    llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);

    // === Test 11: Clear after per-head — should restore baseline ===
    printf("\n=== Test 11: Clear after per-head (restore baseline) ===\n");
    {
        auto positions = make_positions(n_tok);
        auto mask = make_perhead_window_mask(n_tok, n_head, 1, 2);

        // Set per-head mask
        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_tok, n_head, -1);
        eval_prompt(tctx, prompt); // decode with per-head

        // Clear
        llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);
        auto logits = eval_prompt(tctx, prompt);
        float diff = logits_diff(logits_baseline, logits);
        printf("  diff vs baseline after clear: %.6e\n", diff);
        if (diff < 1e-5f) {
            pass("clear after per-head restores baseline");
        } else {
            char buf[128]; snprintf(buf, sizeof(buf), "clear after per-head should restore baseline (diff=%.6e)", diff);
            fail(buf);
        }
    }

    // === Test 12: Multi-slot — per-head on slot 0, different on global ===
    printf("\n=== Test 12: Multi-slot (per-head slot 0 + broadcast global) ===\n");
    {
        auto positions = make_positions(n_tok);

        // Set global broadcast mask (window=3)
        auto mask_global = make_window_mask(n_tok, 3);
        llama_set_attn_mask(ctx, mask_global.data(), positions.data(), n_tok, 0, -1);

        // Set per-head mask on slot 0 (window varies per head)
        auto mask_slot0 = make_perhead_window_mask(n_tok, n_head, 1, n_tok);
        llama_set_attn_mask(ctx, mask_slot0.data(), positions.data(), n_tok, n_head, 0);

        // Note: we can't easily test which slot is used during decode without server mode,
        // but we verify it doesn't crash and produces output
        auto logits = eval_prompt(tctx, prompt);
        if (!logits.empty()) {
            pass("multi-slot set (per-head slot 0 + broadcast global) runs without crash");
        } else {
            fail("multi-slot set crashed or produced empty logits");
        }

        // Clear all slots
        llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);
    }

    // === Test 13: Per-head uniform zeros — identical to baseline (causal still applies) ===
    printf("\n=== Test 13: Per-head uniform zeros (all visible, per-head) ===\n");
    {
        auto positions = make_positions(n_tok);
        auto mask = make_perhead_uniform_mask(n_tok, n_head, 0.0f);

        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_tok, n_head, -1);
        auto logits = eval_prompt(tctx, prompt);
        float diff = logits_diff(logits_baseline, logits);
        printf("  diff vs baseline: %.6e\n", diff);
        if (diff < 1e-5f) {
            pass("per-head uniform zeros identical to baseline (causal still applied)");
        } else {
            char buf[128]; snprintf(buf, sizeof(buf), "per-head zeros should match baseline (diff=%.6e)", diff);
            fail(buf);
        }
    }

    // Clear mask for clean exit
    llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", n_passed, n_failed);
    printf("========================================\n");

    // cleanup
    llama_free(ctx);
    llama_model_free(model);

    return n_failed > 0 ? 1 : 0;
}
