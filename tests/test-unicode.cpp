#include "../src/unicode.h"

#include "ggml-cpp.h"
#include "gguf.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

static bool test_tokenizer_nfc() {
    // Byte-level tokens for "e", U+00E9, U+0301, and newline.
    std::vector<const char *> tokens = {
        "<unk>", "<s>", "</s>", "e", "\xC3\x83", "\xC2\xA9", "\xC3\x8C", "\xC4\xA3", "\xC4\x8A",
    };
    gguf_context_ptr metadata(gguf_init_empty());
    gguf_set_val_str(metadata.get(), "general.architecture", "llama");
    gguf_set_val_str(metadata.get(), "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(metadata.get(), "tokenizer.ggml.pre", "qwen2");
    gguf_set_arr_str(metadata.get(), "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_str(metadata.get(), "tokenizer.ggml.merges", nullptr, 0);
    gguf_set_val_u32(metadata.get(), "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(metadata.get(), "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_u32(metadata.get(), "tokenizer.ggml.unknown_token_id", 0);

    for (bool nfc : {false, true}) {
        if (nfc) {
            gguf_set_val_bool(metadata.get(), "tokenizer.ggml.normalizer.nfc", true);
        }
        auto params = llama_model_default_params();
        params.vocab_only = true;
        llama_model_ptr model(llama_model_init_from_user(metadata.get(), nullptr, nullptr, params));
        if (!model) {
            fprintf(stderr, "failed to create BPE tokenizer for NFC test\n");
            return false;
        }

        llama_token result[3];
        const int32_t count = llama_tokenize(llama_model_get_vocab(model.get()), "e\xCC\x81", 3, result, 3, false, false);
        const std::vector<llama_token> expected = nfc ? std::vector<llama_token>{4, 5} : std::vector<llama_token>{3, 6, 7};
        if (count != (int32_t) expected.size() || !std::equal(expected.begin(), expected.end(), result)) {
            fprintf(stderr, "unexpected BPE tokenization (nfc=%d)\n", nfc);
            return false;
        }
    }
    return true;
}

int main() {
    bool success = test_tokenizer_nfc();
    const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> normalization = {
        {{}, {}},
        {{0x65, 0x301}, {0xE9}},
        {{0xE9}, {0xE9}},
        {{0x212B}, {0xC5}},
        {{0x61, 0x315, 0x300}, {0xE0, 0x315}},
        {{0x61, 0x305, 0x301}, {0x61, 0x305, 0x301}},
        {{0x301, 0x327}, {0x327, 0x301}},
        {{0x1100, 0x1161, 0x11A8}, {0xAC01}},
        {{0xAC01}, {0xAC01}},
        {{0xFB01, 0xFF21}, {0xFB01, 0xFF21}},
        {{0x958}, {0x915, 0x93C}},
        {{0x1D15E}, {0x1D157, 0x1D165}},
    };
    for (const auto & test : normalization) {
        if (unicode_cpts_normalize_nfc(test.first) != test.second || unicode_cpts_normalize_nfc(test.second) != test.second) {
            fprintf(stderr, "unexpected NFC normalization\n");
            success = false;
            break;
        }
    }

    const struct {
        std::string text;
        std::vector<uint32_t> expected;
    } utf8_tests[] = {
        { "\x7f",                 { 0x7f } },
        { "\xc2\x80",             { 0x80 } },
        { "\xdf\xbf",             { 0x7ff } },
        { "\xe0\xa0\x80",         { 0x800 } },
        { "\xef\xbf\xbf",         { 0xffff } },
        { "\xf0\x90\x80\x80",     { 0x10000 } },
        { "\xf4\x8f\xbf\xbf",     { 0x10ffff } },
        // Invalid-input cases preserve the existing recovery behavior: one U+FFFD per byte.
        { "\xc0\xa0",             { 0xfffd, 0xfffd } },
        { "\xc1\xbf",             { 0xfffd, 0xfffd } },
        { "\xe0\x80\xa0",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xe0\x9f\xbf",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x80\x80\xa0",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x8f\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf4\x90\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf5\x80\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf7\xbf\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xe2\x80",             { 0xfffd, 0xfffd } },
    };

    for (const auto & test : utf8_tests) {
        auto expected = test.expected;
        expected.insert(expected.begin(), 'a');
        expected.push_back('b');
        const std::string text = "a" + test.text + "b";
        if (unicode_cpts_from_utf8(text) != expected ||
            unicode_cpts_from_utf8(text.data() + 1, test.text.size()) != test.expected) {
            fprintf(stderr, "unexpected UTF-8 decoding for bytes:");
            for (const unsigned char byte : test.text) {
                fprintf(stderr, " %02x", byte);
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    if (unicode_cpts_from_utf8("\xC3\xA9", 1) != std::vector<uint32_t>{0xfffd}) {
        fprintf(stderr, "UTF-8 decoding crossed the fragment boundary\n");
        success = false;
    }

    const std::vector<std::string> regex_exprs = {
        "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
    };
    const std::vector<std::string> expected = { " ~", "foo" };
    const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "unexpected split:");
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        success = false;
    }

    return success ? 0 : 1;
}
