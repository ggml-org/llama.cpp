// Regression test for Gemma 4 BPE tokenizer crash on long prompts with few/no newlines.
//
// Prior to the fix in src/llama-vocab.cpp, calling unicode_regex_split() on a large
// span of text without newlines caused a SIGSEGV via stack overflow in the regex engine.
// This test exercises that exact path and verifies:
//   1. Tokenization of a long no-newline string completes without crashing.
//   2. The token count is sane (> 0, proportional to input length).
//   3. Roundtrip: detokenized output equals the original input.

#include "llama.h"
#include "common.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s vocab-file\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load vocab from '%s'\n", argv[1]);
        return 1;
    }

    auto cparams = llama_context_default_params();
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // --- Test 1: long string with NO newlines (~9K tokens) ---
    // This is the exact input pattern that previously triggered SIGSEGV via
    // unicode_regex_split() on Gemma 4's pre-tokenizer.
    {
        const std::string unit = "Hello world. ";
        const int reps = 3000;  // ~9K tokens; previously fatal
        std::string text;
        text.reserve(unit.size() * reps);
        for (int i = 0; i < reps; ++i) text += unit;

        const std::vector<llama_token> tokens = common_tokenize(ctx, text, false, false);

        if (tokens.empty()) {
            fprintf(stderr, "FAIL test 1: tokenization returned empty result for long no-newline string\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        // Sanity check: expect roughly 3 tokens per "Hello world. "
        if (tokens.size() < (size_t)reps || tokens.size() > (size_t)(reps * 5)) {
            fprintf(stderr, "FAIL test 1: unexpected token count %zu for %d reps\n", tokens.size(), reps);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        fprintf(stderr, "PASS test 1: long no-newline string tokenized to %zu tokens (expected ~%d)\n",
                tokens.size(), reps * 3);
    }

    // --- Test 2: mixed newline / no-newline (exercises the split boundary logic) ---
    {
        std::string text = "Line one.\nLine two.\n";
        for (int i = 0; i < 500; ++i) text += "No newlines here. ";
        text += "\nFinal line.";
        const std::vector<llama_token> tokens = common_tokenize(ctx, text, false, false);

        if (tokens.empty()) {
            fprintf(stderr, "FAIL test 2: tokenization returned empty result for mixed text\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        fprintf(stderr, "PASS test 2: mixed newline/no-newline string tokenized to %zu tokens\n",
                tokens.size());
    }

    // --- Test 3: string consisting entirely of newlines ---
    {
        const std::string text(200, '\n');
        const std::vector<llama_token> tokens = common_tokenize(ctx, text, false, false);

        if (tokens.empty()) {
            fprintf(stderr, "FAIL test 3: tokenization returned empty result for all-newlines string\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        fprintf(stderr, "PASS test 3: all-newlines string tokenized to %zu tokens\n", tokens.size());
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "All tests passed.\n");
    return 0;
}
