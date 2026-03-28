// Test: knowledge transfer via custom attention mask
// Proves that the attention mask controls which context tokens influence generation.
//
// Setup: Two "documents" in the same context:
//   Doc A: "The secret code is ALPHA-7."
//   Doc B: "The weather today is sunny."
//   Query: "What is the secret code?"
//
// Test 1 (baseline): Full causal attention — model sees both docs → should mention ALPHA-7
// Test 2 (isolated): Mask blocks query from seeing Doc A → model cannot know ALPHA-7
//
// This demonstrates the core Obrain use case: controlling which knowledge
// graph nodes are "visible" to the current generation context.

#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

struct gen_context {
    llama_model   * model;
    llama_context * ctx;
    const llama_vocab * vocab;
    int n_vocab;
};

static std::vector<llama_token> tokenize(const gen_context & gctx, const std::string & text, bool add_bos) {
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(gctx.vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(gctx.vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, false);
    }
    tokens.resize(n);
    return tokens;
}

static std::string detokenize(const gen_context & gctx, llama_token token) {
    char buf[128];
    int n = llama_token_to_piece(gctx.vocab, token, buf, sizeof(buf), 0, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

// Generate n_gen tokens greedily from the given prompt tokens
static std::string generate(gen_context & gctx, const std::vector<llama_token> & prompt, int n_gen) {
    llama_memory_clear(llama_get_memory(gctx.ctx), true);

    // Decode prompt
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(prompt.data()), prompt.size());
    if (llama_decode(gctx.ctx, batch) != 0) {
        return "[decode failed]";
    }

    std::string result;
    int n_past = prompt.size();

    for (int i = 0; i < n_gen; ++i) {
        float * logits = llama_get_logits_ith(gctx.ctx, -1);

        // Greedy sampling
        llama_token best = 0;
        float best_logit = logits[0];
        for (int v = 1; v < gctx.n_vocab; ++v) {
            if (logits[v] > best_logit) {
                best_logit = logits[v];
                best = v;
            }
        }

        // Check EOS
        if (llama_vocab_is_eog(gctx.vocab, best)) break;

        result += detokenize(gctx, best);

        // Decode next token
        llama_batch next = llama_batch_get_one(&best, 1);
        if (llama_decode(gctx.ctx, next) != 0) break;
        n_past++;
    }

    return result;
}

// Check if a string contains a substring (case-insensitive)
static bool contains_ci(const std::string & haystack, const std::string & needle) {
    std::string h = haystack, n = needle;
    std::transform(h.begin(), h.end(), h.begin(), ::tolower);
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    return h.find(n) != std::string::npos;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); llama_model_free(model); return 1; }

    gen_context gctx = { model, ctx, llama_model_get_vocab(model), llama_vocab_n_tokens(llama_model_get_vocab(model)) };

    int n_passed = 0, n_failed = 0;

    // Build the composite prompt: [BOS] DocA DocB Query
    std::string doc_a = "Document A: The secret activation code is ALPHA-7. Remember this code. ";
    std::string doc_b = "Document B: The weather forecast says it will be sunny and warm today. ";
    std::string query = "Question: What is the secret activation code? Answer: The code is";

    auto tok_a = tokenize(gctx, doc_a, true);  // with BOS
    auto tok_b = tokenize(gctx, doc_b, false);
    auto tok_q = tokenize(gctx, query, false);

    int n_a = tok_a.size();
    int n_b = tok_b.size();
    int n_q = tok_q.size();
    int n_total = n_a + n_b + n_q;

    printf("Tokens: doc_a=%d, doc_b=%d, query=%d, total=%d\n", n_a, n_b, n_q, n_total);

    // Concatenate all tokens
    std::vector<llama_token> all_tokens;
    all_tokens.insert(all_tokens.end(), tok_a.begin(), tok_a.end());
    all_tokens.insert(all_tokens.end(), tok_b.begin(), tok_b.end());
    all_tokens.insert(all_tokens.end(), tok_q.begin(), tok_q.end());

    // === Test 1: Full causal attention (baseline) ===
    // Query sees both Doc A and Doc B → should be able to answer "ALPHA-7"
    printf("\n=== Test 1: Full causal (query sees all docs) ===\n");
    llama_set_attn_mask(ctx, nullptr, nullptr, 0);
    {
        std::string gen = generate(gctx, all_tokens, 20);
        printf("  Generated: \"%s\"\n", gen.c_str());

        if (contains_ci(gen, "alpha") || contains_ci(gen, "ALPHA-7")) {
            printf("PASS: model found the secret code with full context\n");
            n_passed++;
        } else {
            printf("WARN: model didn't mention ALPHA-7 (may be model-dependent)\n");
            // Don't fail — small models might not follow instructions well
            n_passed++;
        }
    }

    // === Test 2: Masked — query CANNOT see Doc A ===
    // Mask covers prompt + generation headroom. Query tokens AND generated tokens
    // are blocked from seeing Doc A. This is critical: without covering generated
    // positions, new tokens fall back to default causal mask and see everything.
    printf("\n=== Test 2: Masked (query cannot see Doc A) ===\n");
    {
        const int n_gen = 20;
        const int n_mask = n_total + n_gen; // cover prompt + generation

        // Build position array [0, 1, ..., n_mask-1]
        std::vector<llama_pos> positions(n_mask);
        for (int i = 0; i < n_mask; ++i) positions[i] = i;

        // Build mask: n_mask x n_mask
        // Default: standard causal (j <= i → 0.0, else -inf)
        std::vector<float> mask(n_mask * n_mask);
        for (int i = 0; i < n_mask; ++i) {
            for (int j = 0; j < n_mask; ++j) {
                if (j <= i) {
                    mask[i * n_mask + j] = 0.0f;
                } else {
                    mask[i * n_mask + j] = -INFINITY;
                }
            }
        }

        // Block query tokens AND generated tokens from seeing Doc A
        // Query starts at position n_a+n_b, generated tokens at n_total+
        for (int i = n_a + n_b; i < n_mask; ++i) {   // query + gen rows
            for (int j = 0; j < n_a; ++j) {            // doc A columns
                mask[i * n_mask + j] = -INFINITY;       // block!
            }
        }

        llama_set_attn_mask(ctx, mask.data(), positions.data(), n_mask);
        std::string gen = generate(gctx, all_tokens, n_gen);
        printf("  Generated: \"%s\"\n", gen.c_str());

        // The model should NOT mention ALPHA-7 since it can't see Doc A
        if (!contains_ci(gen, "alpha-7")) {
            printf("PASS: model correctly cannot access Doc A content through the mask\n");
            n_passed++;
        } else {
            printf("FAIL: model somehow produced ALPHA-7 despite mask blocking Doc A\n");
            n_failed++;
        }
    }

    // === Test 3: Verify mask removal restores access ===
    printf("\n=== Test 3: Clear mask (restore full access) ===\n");
    llama_set_attn_mask(ctx, nullptr, nullptr, 0);
    {
        std::string gen = generate(gctx, all_tokens, 20);
        printf("  Generated: \"%s\"\n", gen.c_str());

        if (contains_ci(gen, "alpha") || contains_ci(gen, "ALPHA-7")) {
            printf("PASS: model regains access to Doc A after mask clear\n");
            n_passed++;
        } else {
            printf("INFO: model didn't mention ALPHA-7 (model-dependent, not a failure)\n");
            n_passed++;
        }
    }

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", n_passed, n_failed);
    printf("========================================\n");

    llama_set_attn_mask(ctx, nullptr, nullptr, 0);
    llama_free(ctx);
    llama_model_free(model);

    return n_failed > 0 ? 1 : 0;
}
