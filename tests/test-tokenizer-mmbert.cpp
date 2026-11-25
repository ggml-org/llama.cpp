#include "llama.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct gguf_deleter {
    void operator()(gguf_context * ctx) const {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

struct model_deleter {
    void operator()(llama_model * model) const {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }
};

using gguf_ptr  = std::unique_ptr<gguf_context, gguf_deleter>;
using model_ptr = std::unique_ptr<llama_model, model_deleter>;

static model_ptr load_synthetic_vocab(
        const char * tokenizer_model,
        const char * tokenizer_pre,
        const char * prepend_scheme) {
    gguf_ptr writer(gguf_init_empty());
    if (!writer) {
        throw std::runtime_error("gguf_init_empty failed");
    }

    gguf_set_val_str(writer.get(), "general.architecture", "modern-bert");
    gguf_set_val_str(writer.get(), "tokenizer.ggml.model", tokenizer_model);
    if (tokenizer_pre != nullptr) {
        gguf_set_val_str(writer.get(), "tokenizer.ggml.pre", tokenizer_pre);
    }
    if (prepend_scheme != nullptr) {
        gguf_set_val_str(writer.get(), "tokenizer.ggml.metaspace.prepend_scheme", prepend_scheme);
    }

    // U+2581 = \xE2\x96\x81, U+00E9 (e-acute) = \xC3\xA9
    const char * tokens[] = {
        "<unk>",                   // 0
        "<s>",                     // 1
        "</s>",                    // 2
        "<mask>",                  // 3
        "\xE2\x96\x81",            // 4: metaspace
        "a",                       // 5
        "b",                       // 6
        "\xE2\x96\x81" "a",        // 7
        "\xE2\x96\x81" "b",        // 8
        "ab",                      // 9
        "\xE2\x96\x81" "ab",       // 10
        "\n",                      // 11
        "\xE2\x96\x81" "a\xE2\x96\x81" "b", // 12: must not be merged across pre-tokenized splits
        "\xC3\xA9",                // 13: e-acute
        "\xE2\x96\x81\xC3\xA9",    // 14
        "!",                       // 15
        "\xE2\x96\x81!",           // 16
        "\n\n",                    // 17
    };
    static const int32_t token_types[] = {
        LLAMA_TOKEN_TYPE_UNKNOWN,
        LLAMA_TOKEN_TYPE_CONTROL,
        LLAMA_TOKEN_TYPE_CONTROL,
        LLAMA_TOKEN_TYPE_USER_DEFINED,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_NORMAL,
    };
    const char * merges[] = {
        "\xE2\x96\x81 a",
        "\xE2\x96\x81 b",
        "a b",
        "\xE2\x96\x81" "a b",
        "\xE2\x96\x81" "a \xE2\x96\x81" "b",
        "\xE2\x96\x81 \xC3\xA9",
        "\xE2\x96\x81 !",
    };

    const size_t n_tokens = sizeof(tokens) / sizeof(tokens[0]);
    const size_t n_merges = sizeof(merges) / sizeof(merges[0]);

    gguf_set_arr_str(writer.get(), "tokenizer.ggml.tokens", tokens, n_tokens);
    gguf_set_arr_data(writer.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types, n_tokens);
    gguf_set_arr_str(writer.get(), "tokenizer.ggml.merges", merges, n_merges);

    gguf_set_val_u32(writer.get(), "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(writer.get(), "tokenizer.ggml.bos_token_id",     1);
    gguf_set_val_u32(writer.get(), "tokenizer.ggml.eos_token_id",     2);
    gguf_set_val_u32(writer.get(), "tokenizer.ggml.mask_token_id",    3);
    gguf_set_val_bool(writer.get(), "tokenizer.ggml.add_bos_token",   false);
    gguf_set_val_bool(writer.get(), "tokenizer.ggml.add_eos_token",   false);

    const size_t meta_size = gguf_get_meta_size(writer.get());
    std::vector<uint8_t> buf(meta_size);
    gguf_get_meta_data(writer.get(), buf.data());

    gguf_init_params gparams = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_ptr reader(gguf_init_from_buffer(buf.data(), buf.size(), gparams));
    if (!reader) {
        throw std::runtime_error("gguf_init_from_buffer failed");
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only   = true;
    mparams.n_gpu_layers = 0;

    return model_ptr(llama_model_init_from_user(reader.get(), nullptr, nullptr, mparams));
}

static std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text, bool parse_special) {
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, false, parse_special);
    if (n == 0) {
        return {};
    }
    const int32_t cap = n < 0 ? -n : n;
    std::vector<llama_token> out(cap);
    int32_t actual = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), cap, false, parse_special);
    if (actual < 0) {
        throw std::runtime_error("llama_tokenize failed");
    }
    out.resize(actual);
    return out;
}

static std::string detokenize_tokens(const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    if (ids.empty()) {
        return {};
    }
    int32_t n = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(), nullptr, 0, false, true);
    const int32_t cap = n < 0 ? -n : n;
    std::string out(cap, '\0');
    int32_t actual = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(), out.data(), cap, false, true);
    if (actual < 0) {
        throw std::runtime_error("llama_detokenize failed");
    }
    out.resize(actual);
    return out;
}

static void expect_eq(
        const char * label,
        const std::vector<llama_token> & actual,
        const std::vector<llama_token> & expected) {
    if (actual != expected) {
        std::fprintf(stderr, "FAIL [%s]: token mismatch (got %zu, expected %zu)\n",
                label, actual.size(), expected.size());
        std::fprintf(stderr, "  actual:  ");
        for (llama_token id : actual) {
            std::fprintf(stderr, " %d", id);
        }
        std::fprintf(stderr, "\n  expected:");
        for (llama_token id : expected) {
            std::fprintf(stderr, " %d", id);
        }
        std::fprintf(stderr, "\n");
        throw std::runtime_error("tokenization mismatch");
    }
}

} // namespace

int main() {
    llama_backend_init();

    // 1. Baseline: mmBERT without prepend_scheme key must load.
    model_ptr m_legacy = load_synthetic_vocab("gpt2", "mmbert", nullptr);
    if (!m_legacy) {
        std::fprintf(stderr, "FAIL: mmbert without prepend_scheme failed to load\n");
        llama_backend_free();
        return 1;
    }

    // 2. Regression (#32882): mmBERT with prepend_scheme='always' must load and match legacy behavior.
    model_ptr m_always = load_synthetic_vocab("gpt2", "mmbert", "always");
    if (!m_always) {
        std::fprintf(stderr, "FAIL: mmbert with prepend_scheme='always' failed to load\n");
        llama_backend_free();
        return 1;
    }

    const llama_vocab * v_legacy = llama_model_get_vocab(m_legacy.get());
    const llama_vocab * v_always = llama_model_get_vocab(m_always.get());

    struct test_case {
        const char * name;
        std::string  input;
        bool         parse_special;
        std::vector<llama_token> expected;
    };

    const std::vector<test_case> cases = {
        { "plain_a",         "a",                     false, { 7 } },
        { "leading_space_a", " a",                    false, { 7 } },
        { "metaspace_a",     "\xE2\x96\x81" "a",      false, { 7 } },
        { "merge_ab",        "ab",                    false, { 10 } },
        { "split_a_b",       "a b",                   false, { 7, 8 } },
        { "double_space",    "a  b",                  false, { 7, 4, 8 } },
        { "raw_utf8",        "\xC3\xA9!",             false, { 14, 15 } },
        { "special_mask",    "a<mask>b",              true,  { 7, 3, 8 } },
        { "empty",           "",                      false, {} },
        { "double_newline",  "a\n\nb",                false, { 7, 17, 8 } },
    };

    try {
        for (const auto & tc : cases) {
            const auto ids_legacy = tokenize_text(v_legacy, tc.input, tc.parse_special);
            const auto ids_always = tokenize_text(v_always, tc.input, tc.parse_special);
            expect_eq(tc.name, ids_legacy, tc.expected);
            expect_eq(tc.name, ids_always, tc.expected);

            const auto dec_legacy = detokenize_tokens(v_legacy, ids_legacy);
            const auto dec_always = detokenize_tokens(v_always, ids_always);
            if (dec_legacy != dec_always) {
                std::fprintf(stderr, "FAIL [%s]: detokenize mismatch ('%s' vs '%s')\n",
                        tc.name, dec_legacy.c_str(), dec_always.c_str());
                throw std::runtime_error("detokenize mismatch");
            }
        }

        // 3. Explicit override on mmBERT and invalid scheme validation.
        model_ptr m_mmbert_never = load_synthetic_vocab("gpt2", "mmbert", "never");
        if (!m_mmbert_never) {
            throw std::runtime_error("mmbert with prepend_scheme='never' failed to load");
        }
        const llama_vocab * v_mmbert_never = llama_model_get_vocab(m_mmbert_never.get());
        expect_eq("mmbert_never",         tokenize_text(v_mmbert_never, "a",        false), { 5 });
        expect_eq("mmbert_never_special", tokenize_text(v_mmbert_never, "a<mask>b", true),  { 5, 3, 6 });

        if (load_synthetic_vocab("gpt2", "mmbert", "bogus")) {
            throw std::runtime_error("unknown prepend_scheme should fail");
        }

        // 4. General BPE prepend_scheme coverage ('always', 'first', 'never') via domyn-small.
        model_ptr m_domyn_always = load_synthetic_vocab("gpt2", "domyn-small", "always");
        model_ptr m_domyn_first  = load_synthetic_vocab("gpt2", "domyn-small", "first");
        model_ptr m_domyn_never  = load_synthetic_vocab("gpt2", "domyn-small", "never");
        if (!m_domyn_always || !m_domyn_first || !m_domyn_never) {
            throw std::runtime_error("domyn-small always/first/never failed to load");
        }
        const llama_vocab * v_domyn_always = llama_model_get_vocab(m_domyn_always.get());
        const llama_vocab * v_domyn_first  = llama_model_get_vocab(m_domyn_first.get());
        const llama_vocab * v_domyn_never  = llama_model_get_vocab(m_domyn_never.get());

        expect_eq("domyn_always_a",       tokenize_text(v_domyn_always, "a",        false), { 7 });
        expect_eq("domyn_always_a_b",     tokenize_text(v_domyn_always, "a b",      false), { 7, 8 });
        expect_eq("domyn_always_special", tokenize_text(v_domyn_always, "a<mask>b", true),  { 7, 3, 8 });
        expect_eq("domyn_always_newline", tokenize_text(v_domyn_always, "a\n\nb",   false), { 7, 11, 11, 8 });

        expect_eq("domyn_first_a",        tokenize_text(v_domyn_first,  "a",        false), { 7 });
        expect_eq("domyn_first_special",  tokenize_text(v_domyn_first,  "a<mask>b", true),  { 7, 3, 6 });
        expect_eq("domyn_first_a_b",      tokenize_text(v_domyn_first,  "a b",      false), { 12 });

        expect_eq("domyn_never_a",        tokenize_text(v_domyn_never,  "a",        false), { 5 });
        expect_eq("domyn_never_special",  tokenize_text(v_domyn_never,  "a<mask>b", true),  { 5, 3, 6 });
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    std::printf("OK: all mmbert metaspace prepend_scheme tests passed\n");
    llama_backend_free();
    return 0;
}
