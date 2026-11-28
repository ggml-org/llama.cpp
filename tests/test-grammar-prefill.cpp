// common_sampler_init must advance the grammar past the generation prompt for every tokenizer
// e.g. SentencePiece adds a space after a special token: "<s>assistant" -> "<s>", " assistant"

#include "llama.h"
#include "common.h"
#include "sampling.h"

#include <cstdio>
#include <stdexcept>
#include <string>

static int n_failed = 0;

static void check(bool ok, const std::string & name) {
    printf("  %s: %s\n", ok ? "OK  " : "FAIL", name.c_str());
    if (!ok) {
        n_failed++;
    }
}

static common_sampler * init(const llama_model * model, const std::string & grammar, const std::string & gen_prompt) {
    common_params_sampling params;
    params.grammar           = { COMMON_GRAMMAR_TYPE_TOOL_CALLS, grammar };
    params.grammar_lazy      = false;
    params.generation_prompt = gen_prompt;
    return common_sampler_init(model, params);
}

static bool init_ok(const llama_model * model, const std::string & grammar, const std::string & gen_prompt) {
    try {
        common_sampler * smpl = init(model, grammar, gen_prompt);
        common_sampler_free(smpl);
        return smpl != nullptr;
    } catch (const std::exception & e) {
        fprintf(stderr, "    exception: %s\n", e.what());
        return false;
    }
}

static bool init_throws(const llama_model * model, const std::string & grammar, const std::string & gen_prompt) {
    try {
        common_sampler_free(init(model, grammar, gen_prompt));
        return false;
    } catch (const std::exception &) {
        return true;
    }
}

// a token whose piece is exactly `text`, or LLAMA_TOKEN_NULL
static llama_token find_token(const llama_vocab * vocab, const std::string & text) {
    for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); id++) {
        if (common_token_to_piece(vocab, id, true) == text) {
            return id;
        }
    }
    return LLAMA_TOKEN_NULL;
}

// does the grammar accept `token` right after the prefill?
static bool accepts_next(const llama_model * model, const std::string & grammar, const std::string & gen_prompt, llama_token token) {
    common_sampler * smpl = nullptr;
    bool ok = true;
    try {
        smpl = init(model, grammar, gen_prompt);
        common_sampler_accept(smpl, token, /* is_generated = */ true);
    } catch (const std::exception &) {
        ok = false;
    }
    common_sampler_free(smpl);
    return ok;
}

static std::string gbnf_literal(const std::string & s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            default:   out += c;
        }
    }
    return out + "\"";
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <vocab-file>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load vocab '%s'\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    printf("vocab: %s\n", argv[1]);

    // a special token that opens the turn header
    llama_token special = LLAMA_TOKEN_NULL;
    for (const char * text : {"<|im_start|>", "<|assistant|>"}) {
        if (special == LLAMA_TOKEN_NULL) {
            special = find_token(vocab, text);
        }
    }
    if (special == LLAMA_TOKEN_NULL) {
        special = llama_vocab_bos(vocab);
    }
    for (llama_token id = 0; special == LLAMA_TOKEN_NULL && id < llama_vocab_n_tokens(vocab); id++) {
        if (llama_vocab_is_control(vocab, id) && !llama_vocab_is_eog(vocab, id)) {
            special = id;
        }
    }
    const std::string special_text = common_token_to_piece(vocab, special, true);

    const llama_token tok_x = find_token(vocab, "x");
    const llama_token tok_y = find_token(vocab, "y");

    // "\n" alone: a special token with rstrip (phi-3) takes it into the token
    for (const std::string rest : {"assistant\n", "\n"}) {
        const std::string gen_prompt = special_text + rest;

        std::string spelled;
        for (llama_token t : common_tokenize(vocab, gen_prompt, false, true)) {
            spelled += common_token_to_piece(vocab, t, true);
        }
        printf("generation prompt: '%s'\ntokens spell:      '%s'\n", gen_prompt.c_str(), spelled.c_str());

        // text literal as the chat auto-parser builds it, token id as token-aware parsers (gemma4) build it
        const std::string grammar_text  = "root ::= " + gbnf_literal(gen_prompt) + " \"x\"";
        const std::string grammar_token = "root ::= <[" + std::to_string(special) + "]> " + gbnf_literal(rest) + " \"x\"";

        check(init_ok(model, grammar_text, gen_prompt),  "prefill into a text-literal grammar");
        check(init_ok(model, grammar_token, gen_prompt), "prefill into a token-id grammar");
        check(init_throws(model, "root ::= " + gbnf_literal(special_text + "user\n") + " \"x\"", gen_prompt), "prefill that does not match the grammar fails");

        if (tok_x != LLAMA_TOKEN_NULL && tok_y != LLAMA_TOKEN_NULL) {
            check(accepts_next(model, grammar_text, gen_prompt, tok_x),  "text-literal grammar accepts 'x' after the prefill");
            check(!accepts_next(model, grammar_text, gen_prompt, tok_y), "text-literal grammar rejects 'y' after the prefill");
            check(accepts_next(model, grammar_token, gen_prompt, tok_x), "token-id grammar accepts 'x' after the prefill");
        }
    }

    llama_model_free(model);
    llama_backend_free();

    printf("%s\n", n_failed == 0 ? "all tests passed" : "some tests failed");
    return n_failed == 0 ? 0 : 1;
}
