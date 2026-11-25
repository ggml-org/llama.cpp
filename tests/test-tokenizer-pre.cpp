// pre-tokenizer tests that need no vocab file
//
// a tiny byte-level BPE vocab is built in memory: the 256 byte tokens (id = byte value) and a few merges
// a merge can only apply inside one pre-tokenizer piece, so the token ids show where the pieces were split
// expected ids come from HF tokenizers, with the same vocab and the pre_tokenizer from the model's tokenizer.json

#include "llama.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// GPT-2 byte-level encoding of a byte, as UTF-8
static std::string byte_to_utf8(int b) {
    auto printable = [](int c) {
        return (c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || c >= 0xAE;
    };

    int cp = b;
    if (!printable(b)) {
        // the other bytes map to 256, 257, ... in byte order
        cp = 256;
        for (int c = 0; c < b; c++) {
            cp += printable(c) ? 0 : 1;
        }
    }

    std::string res;
    if (cp < 0x80) {
        res += (char) cp;
    } else {
        res += (char) (0xC0 | (cp >> 6));
        res += (char) (0x80 | (cp & 0x3F));
    }
    return res;
}

// byte tokens 0..255, then 256 = "\r\n" and 257 = "\n\r"
static llama_model * make_vocab(const char * pre) {
    const std::string cr = byte_to_utf8('\r');
    const std::string lf = byte_to_utf8('\n');

    std::vector<std::string> tokens;
    for (int b = 0; b < 256; b++) {
        tokens.push_back(byte_to_utf8(b));
    }
    tokens.push_back(cr + lf);
    tokens.push_back(lf + cr);

    const std::vector<std::string> merges = { cr + " " + lf, lf + " " + cr };

    std::vector<const char *> tokens_c;
    for (const auto & t : tokens) {
        tokens_c.push_back(t.c_str());
    }
    std::vector<const char *> merges_c;
    for (const auto & m : merges) {
        merges_c.push_back(m.c_str());
    }
    const std::vector<int32_t> types(tokens.size(), LLAMA_TOKEN_TYPE_NORMAL);

    gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture", "llama");
    gguf_set_val_str(ctx, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(ctx, "tokenizer.ggml.pre",   pre);
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", tokens_c.data(), tokens_c.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_arr_str(ctx, "tokenizer.ggml.merges", merges_c.data(), merges_c.size());

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;

    llama_model * model = llama_model_init_from_user(ctx, nullptr, nullptr, params);
    gguf_free(ctx);
    return model;
}

static std::string escape(const std::string & s) {
    std::string res;
    for (const char c : s) {
        switch (c) {
            case '\r': res += "\\r"; break;
            case '\n': res += "\\n"; break;
            default:   res += c;     break;
        }
    }
    return res;
}

static bool run_tests(const char * pre, const std::vector<std::pair<std::string, std::vector<llama_token>>> & tests) {
    llama_model * model = make_vocab(pre);
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to create vocab with pre-tokenizer '%s'\n", __func__, pre);
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    bool success = true;
    for (const auto & test : tests) {
        std::vector<llama_token> res(test.first.size() + 1);
        const int32_t n = llama_tokenize(vocab, test.first.c_str(), test.first.size(), res.data(), res.size(), false, false);
        res.resize(n < 0 ? 0 : n);

        if (res != test.second) {
            fprintf(stderr, "%s : %s: '%s' expected:", __func__, pre, escape(test.first).c_str());
            for (const llama_token t : test.second) {
                fprintf(stderr, " %d", t);
            }
            fprintf(stderr, ", got:");
            for (const llama_token t : res) {
                fprintf(stderr, " %d", t);
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    llama_model_free(model);
    return success;
}

int main() {
    llama_backend_init();

    bool success = true;

    // Laguna splits with "(?:\r?\n)+(?!\r?\n)" and behavior MergedWithNext, so a \r\n is never split
    success &= run_tests("laguna", {
        { "x\r\ny",                    { 120, 256, 121 } },
        { "a\r\n\r\nb",                { 97, 256, 256, 98 } },
        { "\n\rX",                     { 257, 88 } },
        { "a\rb",                      { 97, 13, 98 } },
        { "a\n\nb",                    { 97, 10, 10, 98 } },
        { "if x:\r\n    return 1\r\n", { 105, 102, 32, 120, 58, 256, 32, 32, 32, 32, 114, 101, 116, 117, 114, 110, 32, 49, 256 } },
    });

    llama_backend_free();

    return success ? 0 : 1;
}
