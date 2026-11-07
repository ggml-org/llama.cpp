#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"
#include "llama-cpp.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include "../src/llama-grammar.h"
#include "../src/llama-model-loader.h"
#include "../src/llama-vocab.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <thread>

using grammar_ptr = std::unique_ptr<llama_grammar, decltype(&llama_grammar_free_impl)>;

static grammar_ptr make_grammar(const llama_vocab & vocab, const char * source) {
    grammar_ptr grammar(llama_grammar_init_impl(&vocab, source, "root", false, nullptr, 0, nullptr, 0), llama_grammar_free_impl);
    assert(grammar);
    return grammar;
}

struct mask_vocab {
    static constexpr llama_token byte_base = 6;
    llama_vocab vocab;
    std::vector<llama_token> edge_tokens;

    mask_vocab(bool alternate = false) {
        std::vector<std::string> tokens = { "<unk>", "<s>", "</s>", "<|eot_id|>", "<|eom_id|>", "unused" };
        std::vector<int32_t> types = { LLAMA_TOKEN_TYPE_UNKNOWN, LLAMA_TOKEN_TYPE_CONTROL, LLAMA_TOKEN_TYPE_CONTROL,
                                      LLAMA_TOKEN_TYPE_CONTROL, LLAMA_TOKEN_TYPE_CONTROL, LLAMA_TOKEN_TYPE_UNUSED };
        for (int i = 0; i < 256; ++i) {
            char text[7];
            snprintf(text, sizeof(text), "<0x%02X>", i);
            tokens.emplace_back(text);
            types.push_back(LLAMA_TOKEN_TYPE_BYTE);
        }
        for (const auto & piece : { "a", "b", "ab", "ac", "ax", "a!", "xy", "Oa", "\xe2\x82\xac", "\xe2\x28\xa1", "\xc0\xaf" }) {
            tokens.emplace_back(piece);
            types.push_back(LLAMA_TOKEN_TYPE_NORMAL);
        }
        std::vector<std::string> edges = {
            "a", "abc", "abcd", "abd", "abx", "<special>",
            "\xe2", "\xe2\x82", "\xe2\x82\xac" "a", "\xf0\x9f", "\xf0\x9f\x98\x80",
            "\xc0", "\xc0\x80", "\xc0\x80" "a", "a\xc0\x80" "b", "a\x80", "ab\xff", "\xff", "\xf8\xbf\xbf\xbf",
            std::string("a\0b", 3), std::string("a\0\xe2", 3), std::string("\0a", 2), std::string("\xe2\0x", 3),
            "\xe2\x28", "\xe2\x28\xa1", "\xc0\xaf", "\xf0\x00", "", std::string(256, 'a'), "\xc1", "\xc1\xaf",
        };
        uint32_t random = 0x12345678;
        for (size_t i = 0; i < 512; ++i) {
            std::string piece;
            for (size_t j = 0; j < i % 8; ++j) {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                piece.push_back(static_cast<char>(random & 255));
            }
            edges.push_back(piece);
        }
        for (size_t i = 0; i < edges.size(); ++i) {
            edge_tokens.push_back(tokens.size());
            tokens.push_back("edge-" + std::to_string(i));
            types.push_back(i == 5 ? LLAMA_TOKEN_TYPE_CONTROL : LLAMA_TOKEN_TYPE_NORMAL);
        }
        if (alternate) {
            tokens[byte_base + 256] = "z";
        }
        std::vector<const char *> texts;
        for (const auto & token : tokens) {
            texts.push_back(token.c_str());
        }
        gguf_context_ptr metadata(gguf_init_empty());
        gguf_set_val_str(metadata.get(), "general.architecture", "llama");
        gguf_set_val_str(metadata.get(), "tokenizer.ggml.model", "llama");
        gguf_set_arr_str(metadata.get(), "tokenizer.ggml.tokens", texts.data(), texts.size());
        gguf_set_arr_data(metadata.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        gguf_set_val_u32(metadata.get(), "tokenizer.ggml.eos_token_id", 2);
        gguf_set_val_u32(metadata.get(), "tokenizer.ggml.eot_token_id", 3);
        gguf_set_val_u32(metadata.get(), "tokenizer.ggml.eom_token_id", 4);
        std::vector<std::string> splits;
        llama_model_loader loader(metadata.get(), nullptr, nullptr, "", splits, nullptr, LLAMA_LOAD_MODE_MMAP, false, true, false, nullptr, nullptr);
        vocab.load(loader, LLM_KV(LLM_ARCH_LLAMA));
        for (size_t i = 0; i < edges.size(); ++i) {
            const_cast<std::string &>(vocab.token_to_piece(edge_tokens[i])) = edges[i];
        }
        assert(vocab.token_to_piece(5).empty());
        assert(vocab.token_to_piece(byte_base) == std::string(1, '\0'));
        assert(vocab.n_tokens() % 32 != 0);
    }
};

static bool mask_allows(const std::vector<uint32_t> & mask, llama_token token) {
    return (mask[token / 32] & (uint32_t(1) << (token % 32))) != 0;
}

static size_t trie_masks = 0;
static size_t fallback_masks = 0;

static std::vector<uint32_t> check_mask(const llama_grammar & grammar) {
    const auto stacks = grammar.stacks;
    const auto partial_utf8 = grammar.partial_utf8;
    const auto trigger_buffer = grammar.trigger_buffer;
    const bool awaiting_trigger = grammar.awaiting_trigger;
    const auto mask = llama_grammar_get_allowed_tokens(grammar);
    bool character_only = !grammar.awaiting_trigger;
    for (const auto & rule : grammar.rules) {
        for (const auto & element : rule) {
            if (element.type == LLAMA_GRETYPE_TOKEN || element.type == LLAMA_GRETYPE_TOKEN_NOT) {
                character_only = false;
            }
        }
    }
    std::vector<uint32_t> trie_mask;
    const bool trie_supported = character_only && grammar.vocab->get_grammar_trie() != nullptr;
    const bool trie_used = llama_grammar_get_allowed_tokens_trie(grammar, trie_mask);
    assert(!trie_used || trie_supported);
    if (trie_used) {
        assert(trie_mask == mask);
        ++trie_masks;
    } else {
        ++fallback_masks;
    }

    assert(mask == llama_grammar_get_allowed_tokens(grammar));
    const auto n_vocab = grammar.vocab->n_tokens();
    assert(mask.size() == (n_vocab + 31) / 32);
    if (n_vocab % 32 != 0) {
        assert((mask.back() >> (n_vocab % 32)) == 0);
    }
    std::vector<llama_token_data> tokens;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tokens.push_back({ static_cast<llama_token>(i), 0.25f, 0.0f });
    }
    llama_token_data_array candidates = { tokens.data(), tokens.size(), -1, false };
    llama_grammar_apply_impl(grammar, &candidates);
    for (const auto & token : tokens) {
        if (mask_allows(mask, token.id) != (token.logit != -INFINITY)) {
            fprintf(stderr, "mask mismatch for token %d, partial %u/%d\n", token.id, partial_utf8.value, partial_utf8.n_remain);
            assert(false);
        }
    }
    assert(grammar.stacks == stacks);
    assert(grammar.partial_utf8.value == partial_utf8.value && grammar.partial_utf8.n_remain == partial_utf8.n_remain);
    assert(grammar.trigger_buffer == trigger_buffer && grammar.awaiting_trigger == awaiting_trigger);
    return mask;
}

static void test_allowed_tokens(const mask_vocab & fixture) {
    const auto & vocab = fixture.vocab;
    const auto a = vocab.text_to_token("a");
    const auto b = vocab.text_to_token("b");
    for (const char * source : {
            "root ::= (\"a\" | \"ab\") [b-c]?",
            "root ::= item \"b\"\nitem ::= \"a\" item?",
            "root ::= [^x-z]*",
            "root ::= .+",
            "root ::= \"a\" tail\ntail ::= \"b\" | \"c\"",
            "root ::= \"ab\" | \"ac\"" }) {
        auto grammar = make_grammar(vocab, source);
        const auto mask = check_mask(*grammar);
        assert(mask_allows(mask, a) == mask_allows(mask, mask_vocab::byte_base + 'a'));
        assert(!mask_allows(mask, 5));
        assert(!mask_allows(mask, mask_vocab::byte_base));
        llama_grammar_accept_impl(*grammar, a);
        check_mask(*grammar);
        llama_grammar_accept_impl(*grammar, b);
        check_mask(*grammar);
    }

    auto grammar = make_grammar(vocab, "root ::= \"a\"?");
    for (const auto token : { 2, 3, 4 }) {
        assert(vocab.is_eog(token));
        assert(mask_allows(check_mask(*grammar), token));
    }
    llama_grammar_accept_impl(*grammar, a);
    const auto finished = check_mask(*grammar);
    for (uint32_t i = 0; i < vocab.n_tokens(); ++i) {
        if (vocab.is_eog(i)) {
            assert(mask_allows(finished, i));
        }
        if (vocab.token_to_piece(i) == "a") {
            assert(!mask_allows(finished, i));
        }
    }

    grammar = make_grammar(vocab, "root ::= \"a\"");
    for (const auto token : { 2, 3, 4 }) {
        assert(!mask_allows(check_mask(*grammar), token));
    }

    for (const auto & source : {
            "root ::= <[" + std::to_string(a) + "]>",
            "root ::= !<[" + std::to_string(a) + "]>",
            "root ::= \"a\" <[" + std::to_string(a) + "]> | <[" + std::to_string(b) + "]> \"b\"" }) {
        grammar = make_grammar(vocab, source.c_str());
        check_mask(*grammar);
    }
    grammar = make_grammar(vocab, ("root ::= <[" + std::to_string(a) + "]> <[" + std::to_string(b) + "]>").c_str());
    for (const auto token : { a, b }) {
        assert(mask_allows(check_mask(*grammar), token));
        llama_grammar_accept_impl(*grammar, token);
    }
    check_mask(*grammar);
    grammar = make_grammar(vocab, "root ::= <[4294967295]>");
    const auto unmatched_token = check_mask(*grammar);
    assert(!mask_allows(unmatched_token, a));
    assert(!mask_allows(unmatched_token, vocab.token_eos()));
    assert(mask_allows(unmatched_token, fixture.edge_tokens[12]));
}

static void test_mask_utf8(const mask_vocab & fixture) {
    const auto & vocab = fixture.vocab;
    auto grammar = make_grammar(vocab, "root ::= \"\\u20ac\\u00e9\"");
    check_mask(*grammar);
    for (const auto byte : { 0xe2, 0x82, 0xac, 0xc3, 0xa9 }) {
        const auto token = mask_vocab::byte_base + byte;
        assert(mask_allows(check_mask(*grammar), token));
        llama_grammar_accept_impl(*grammar, token);
        check_mask(*grammar);
    }

    grammar = make_grammar(vocab, "root ::= \"\\u20ac\"?");
    llama_grammar_accept_impl(*grammar, mask_vocab::byte_base + 0xe2);
    assert(grammar->partial_utf8.n_remain == 2);
    assert(!mask_allows(check_mask(*grammar), vocab.token_eos()));

    grammar = make_grammar(vocab, "root ::= \"\\u00a0\"");
    grammar->partial_utf8 = { 2, 1 };
    const auto first = check_mask(*grammar);
    const auto misses = grammar->mask_cache.misses;
    grammar->partial_utf8.value = 3;
    assert(check_mask(*grammar) != first);
    assert(grammar->mask_cache.misses == misses + 1);
    grammar->partial_utf8.n_remain = -1;
    check_mask(*grammar);
    assert(grammar->mask_cache.misses == misses + 2);
}

static void test_mask_lazy(const mask_vocab & fixture) {
    const auto & vocab = fixture.vocab;
    const llama_token trigger = vocab.text_to_token("a");
    grammar_ptr grammar(llama_grammar_init_impl(&vocab, "root ::= \"ab\"", "root", true, nullptr, 0, &trigger, 1), llama_grammar_free_impl);
    assert(grammar);
    const auto all = check_mask(*grammar);
    for (uint32_t i = 0; i < vocab.n_tokens(); ++i) {
        assert(mask_allows(all, i));
    }
    llama_grammar_accept_impl(*grammar, trigger);
    assert(!grammar->awaiting_trigger);
    assert(check_mask(*grammar) != all);

    const char * pattern = "GO(a)";
    grammar.reset(llama_grammar_init_impl(&vocab, "root ::= \"ab\"", "root", true, &pattern, 1, nullptr, 0));
    assert(grammar);
    check_mask(*grammar);
    llama_grammar_accept_impl(*grammar, mask_vocab::byte_base + 'G');
    assert(grammar->awaiting_trigger);
    assert(check_mask(*grammar) == all);
    llama_grammar_accept_impl(*grammar, vocab.text_to_token("Oa"));
    assert(!grammar->awaiting_trigger);
    assert(mask_allows(check_mask(*grammar), vocab.text_to_token("b")));
}

static void test_mask_cache(const mask_vocab & fixture) {
    const auto & vocab = fixture.vocab;
    auto grammar = make_grammar(vocab, "root ::= item \"b\" | item \"c\"\nitem ::= \"a\"");
    const auto initial_stacks = grammar->stacks;
    auto & cache = grammar->mask_cache;
    const auto initial = check_mask(*grammar);
    assert(cache.misses == 1 && cache.hits == 1);
    grammar->partial_utf8.value = 'x';
    assert(check_mask(*grammar) == initial);
    assert(cache.misses == 1 && cache.hits == 3);
    grammar->partial_utf8.value = 0;
    std::reverse(grammar->stacks.begin(), grammar->stacks.end());
    grammar->stacks.push_back(grammar->stacks.back());
    assert(check_mask(*grammar) == initial);
    assert(cache.misses == 1 && cache.hits == 5);
    auto detached = llama_grammar_get_allowed_tokens(*grammar);
    detached[0] ^= 1;
    assert(llama_grammar_get_allowed_tokens(*grammar) == initial);

    grammar->awaiting_trigger = true;
    assert(check_mask(*grammar) != initial);
    assert(cache.misses == 2);
    grammar->awaiting_trigger = false;
    assert(check_mask(*grammar) == initial);
    assert(cache.misses == 2);

    const mask_vocab other(true);
    grammar->vocab = &other.vocab;
    assert(check_mask(*grammar) != initial);
    assert(cache.misses == 3);
    grammar->vocab = &vocab;
    assert(check_mask(*grammar) == initial);
    assert(cache.misses == 3);

    grammar->stacks = { initial_stacks[0] };
    const auto branch = check_mask(*grammar);
    grammar->stacks = { initial_stacks[1] };
    assert(grammar->stacks[0].back() == initial_stacks[0].back());
    assert(check_mask(*grammar) != branch);

    grammar_ptr clone(llama_grammar_clone_impl(*grammar), llama_grammar_free_impl);
    assert(clone->mask_cache.entries.empty());
    assert(check_mask(*clone) == check_mask(*grammar));
    grammar.reset();
    check_mask(*clone);

    grammar = make_grammar(vocab, "root ::= \"abc\"");
    auto & lru = grammar->mask_cache;
    lru.max_entries = 2;
    const auto start = grammar->stacks;
    const auto start_utf8 = grammar->partial_utf8;
    check_mask(*grammar);
    llama_grammar_accept_impl(*grammar, vocab.text_to_token("a"));
    const auto middle = grammar->stacks;
    const auto middle_utf8 = grammar->partial_utf8;
    check_mask(*grammar);
    grammar->stacks = start;
    grammar->partial_utf8 = start_utf8;
    check_mask(*grammar);
    grammar->stacks = middle;
    grammar->partial_utf8 = middle_utf8;
    llama_grammar_accept_impl(*grammar, vocab.text_to_token("b"));
    check_mask(*grammar);
    assert(lru.entries.size() == 2 && lru.bytes <= lru.max_bytes);
    auto misses = lru.misses;
    grammar->stacks = start;
    grammar->partial_utf8 = start_utf8;
    check_mask(*grammar);
    assert(lru.misses == misses);
    grammar->stacks = middle;
    grammar->partial_utf8 = middle_utf8;
    check_mask(*grammar);
    assert(lru.misses == misses + 1);

    lru.max_bytes = lru.entries.front().bytes;
    check_mask(*grammar);
    assert(lru.entries.size() == 1 && lru.bytes <= lru.max_bytes);
    lru.max_bytes = 1;
    misses = lru.misses;
    check_mask(*grammar);
    assert(lru.entries.empty() && lru.bytes == 0 && lru.misses == misses + 2);
    lru.max_bytes = sizeof(llama_grammar_mask_cache::entry) + 2 * sizeof(void *) + ((vocab.n_tokens() + 31) / 32) * sizeof(uint32_t);
    check_mask(*grammar);
    assert(lru.entries.empty() && lru.bytes == 0);
    lru.max_bytes = 8 * 1024 * 1024;
    lru.max_entries = 0;
    check_mask(*grammar);
    assert(lru.entries.empty());
}

static void test_trie_boundaries(const mask_vocab & fixture) {
    const auto & vocab = fixture.vocab;
    auto grammar = make_grammar(vocab, "root ::= left right\nleft ::= \"ab\"\nright ::= \"cd\" | \"d\"");
    auto mask = check_mask(*grammar);
    for (const auto piece : { "a", "ab", "abc", "abcd", "abd" }) {
        bool found = false;
        for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
            if (vocab.token_to_piece(id) == piece) {
                assert(mask_allows(mask, id));
                found = true;
            }
        }
        assert(found);
    }
    assert(!mask_allows(mask, vocab.text_to_token("ax")));
    const auto * trie = vocab.get_grammar_trie();
    assert(trie && trie == vocab.get_grammar_trie());
    std::vector<uint32_t> direct;
    assert(llama_grammar_get_allowed_tokens_trie(*grammar, direct));
    assert(direct == mask);
    grammar_ptr clone(llama_grammar_clone_impl(*grammar), llama_grammar_free_impl);
    assert(check_mask(*clone) == mask);
    assert(clone->vocab->get_grammar_trie() == trie);
    std::vector<uint32_t> limited;
    assert(!llama_grammar_get_allowed_tokens_trie(*grammar, limited, 0));
    assert(!llama_grammar_get_allowed_tokens_trie(*grammar, limited, 32));

    for (const auto source : {
            "root ::= \"ab\" | \"ac\" | \"abcd\"",
            "root ::= \"a\" (\"b\" | \"c\" | \"bcd\")",
            "root ::= (\"a\"? \"b\")*",
            "root ::= \"a\" root? \"b\" | \"c\"",
            "root ::= [^x-z]*",
            "root ::= [^x]",
            "root ::= [a-c]+",
            "root ::= .+",
            "root ::= \"\\u0000\" \"a\"",
            "root ::= \"\\u20ac\" | \"\\U0001f600\"",
            "root ::= \"\\u2a21\" | \"/\" | \"a\"",
            "root ::= (\"a\" | \"\\u20ac\")*" }) {
        grammar = make_grammar(vocab, source);
        grammar->mask_cache.max_entries = 0;
        check_mask(*grammar);
        mask = llama_grammar_get_allowed_tokens(*grammar);
        for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
            if (!mask_allows(mask, id) || vocab.is_eog(id)) {
                continue;
            }
            clone.reset(llama_grammar_clone_impl(*grammar));
            try {
                llama_grammar_accept_impl(*clone, id);
            } catch (const std::runtime_error &) {
                assert(clone->stacks.empty());
                continue;
            }
            check_mask(*clone);
        }
        for (const auto partial : { llama_partial_utf8{ 0, -1 }, { 0, 0 }, { 0, 1 }, { 1, 1 },
                                   { 2, 1 }, { 0x20, 1 }, { 2, 2 }, { 0, 3 }, { 0x1f, 3 } }) {
            grammar->partial_utf8 = partial;
            check_mask(*grammar);
        }
    }

    auto factored = make_grammar(vocab, "root ::= \"a\" (\"b\" | \"c\" | \"bcd\")");
    auto unfactored = make_grammar(vocab, "root ::= \"ab\" | \"ac\" | \"abcd\"");
    assert(check_mask(*factored) == check_mask(*unfactored));
    llama_grammar_accept_impl(*factored, vocab.text_to_token("a"));
    llama_grammar_accept_impl(*unfactored, vocab.text_to_token("a"));
    assert(check_mask(*factored) == check_mask(*unfactored));

    grammar = make_grammar(vocab, "root ::= \"a\"");
    mask = check_mask(*grammar);
    assert(mask_allows(mask, fixture.edge_tokens[19]));
    assert(!mask_allows(mask, fixture.edge_tokens[21]));
    assert(!mask_allows(mask, fixture.edge_tokens[15]));
    assert(mask_allows(mask, fixture.edge_tokens[14]));

    grammar = make_grammar(vocab, "root ::= \"\\u2a21\"");
    mask = check_mask(*grammar);
    assert(mask_allows(mask, fixture.edge_tokens[24]));
    assert(mask_allows(mask, fixture.edge_tokens[23]));
    llama_grammar_accept_impl(*grammar, mask_vocab::byte_base + 0xe2);
    mask = check_mask(*grammar);
    assert(!mask_allows(mask, mask_vocab::byte_base + 0x28));

    grammar = make_grammar(vocab, "root ::= \"/\"");
    mask = check_mask(*grammar);
    assert(!mask_allows(mask, fixture.edge_tokens[11]));
    assert(mask_allows(mask, fixture.edge_tokens[25]));

    grammar = make_grammar(vocab, "root ::= [^x]");
    mask = check_mask(*grammar);
    assert(!mask_allows(mask, fixture.edge_tokens[29]));
    assert(mask_allows(mask, fixture.edge_tokens[30]));

    const auto ordinary_a = vocab.text_to_token("a");
    const auto duplicate_a = fixture.edge_tokens[0];
    for (const auto op : { "", "!" }) {
        grammar = make_grammar(vocab, ("root ::= " + std::string(op) + "<[" + std::to_string(ordinary_a) + "]>").c_str());
        mask = check_mask(*grammar);
        assert(mask_allows(mask, ordinary_a) != mask_allows(mask, duplicate_a));
    }
}

static void test_trie_shared_vocab() {
    mask_vocab fixture;
    std::vector<const llama_vocab_trie *> tries(4);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < tries.size(); ++i) {
        threads.emplace_back([&, i]() { tries[i] = fixture.vocab.get_grammar_trie(); });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    for (const auto * trie : tries) {
        assert(trie && trie == tries[0]);
    }
}

static void test_trie_preprocessing_limit() {
    mask_vocab fixture;
    const_cast<std::string &>(fixture.vocab.token_to_piece(fixture.edge_tokens.back())) =
        std::string(64 * 1024 * 1024 / sizeof(llama_vocab_trie::node), 'a');
    assert(fixture.vocab.get_grammar_trie() == nullptr);
    assert(fixture.vocab.get_grammar_trie() == nullptr);
    auto grammar = make_grammar(fixture.vocab, "root ::= \"b\"");
    check_mask(*grammar);
}

static void test_candidate_storage() {
    mask_vocab fixture;
    const auto & vocab = fixture.vocab;
    auto grammar = make_grammar(vocab, "root ::= .+");
    const std::vector<std::string> pieces = {"a", "ab", std::string(4096, 'a'), "a\x80", "\xe2\x82\xac"};
    std::vector<llama_token_data> values;
    for (auto id : fixture.edge_tokens) {
        values.push_back({id, 0.0f, 0.0f});
    }
    for (size_t i = 0; i < pieces.size(); ++i) {
        const_cast<std::string &>(vocab.token_to_piece(values[i].id)) = pieces[i];
    }
    const std::vector<std::vector<uint32_t>> decoded = {{'a', 0}, {'a', 'b', 0}, {}, {0}, {0x20ac, 0}};
    llama_token_data_array cur = {values.data(), values.size(), -1, false};
    llama_grammar_apply_impl(*grammar, &cur);
    const auto & scratch = grammar->scratch;
    for (size_t i = 0; i < pieces.size(); ++i) {
        const auto * points = scratch.candidates[i].code_points;
        if (i == 2) {
            for (size_t j = 0; j < 4096; ++j) {
                assert(points[j] == 'a');
            }
            assert(points[4096] == 0);
        } else {
            assert(std::equal(decoded[i].begin(), decoded[i].end(), points));
        }
    }
    assert(scratch.candidates[3].partial_utf8.n_remain == -1);
    const auto * arena = scratch.code_points.data();
    const auto * candidates = scratch.candidates.data();
    llama_grammar_apply_impl(*grammar, &cur);
    assert(scratch.code_points.data() == arena && scratch.candidates.data() == candidates);
    grammar_ptr clone(llama_grammar_clone_impl(*grammar), llama_grammar_free_impl);
    grammar.reset();
    llama_grammar_apply_impl(*clone, &cur);
    for (const auto & stack : clone->stacks) {
        for (const auto * point : stack) {
            assert(clone->rule_positions.count(point) == 1);
        }
    }
}

static void test_backend_mask_representations() {
    mask_vocab fixture;
    const auto & vocab = fixture.vocab;
    const int32_t n_vocab = vocab.n_tokens();
    for (int32_t id = 0; id < n_vocab; ++id) {
        const_cast<std::string &>(vocab.token_to_piece(id)) = id % 2 ? "a" : "b";
    }
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    assert(backend);
    const std::vector<std::pair<std::string, int32_t>> cases = {
        {"root ::= <[" + std::to_string(n_vocab - 1) + "]>", GGML_TOKEN_MASK_ALLOW_IDS},
        {"root ::= .+", GGML_TOKEN_MASK_DENY_IDS},
        {"root ::= \"a\"+", GGML_TOKEN_MASK_BITSET},
        {"root ::= <[4294967295]>", GGML_TOKEN_MASK_NONE},
        {"root ::= .+", GGML_TOKEN_MASK_ALL},
    };
    for (const auto & item : cases) {
        llama_sampler_ptr sampler(item.second == GGML_TOKEN_MASK_ALL ?
                llama_sampler_init_grammar_lazy_patterns(&vocab, item.first.c_str(), "root", nullptr, 0, nullptr, 0) :
                llama_sampler_init_grammar(&vocab, item.first.c_str(), "root"));
        assert(sampler && sampler->iface->backend_init(sampler.get(), ggml_backend_get_default_buffer_type(backend.get()), 1));
        ggml_context_ptr ctx(ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
        auto * graph = ggml_new_graph(ctx.get());
        auto * logits = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_vocab);
        llama_sampler_data data = {logits, nullptr, nullptr, nullptr};
        sampler->iface->backend_apply(sampler.get(), ctx.get(), graph, &data);
        ggml_build_forward_expand(graph, data.logits);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        assert(buffer);
        std::vector<float> actual(n_vocab, 0.0f);
        ggml_backend_tensor_set(logits, actual.data(), 0, actual.size()*sizeof(float));
        sampler->iface->backend_set_input(sampler.get(), nullptr);
        int32_t desc[4];
        ggml_backend_tensor_get(data.logits->src[3], desc, 0, sizeof(desc));
        assert(desc[0] == item.second);
        if (desc[0] == GGML_TOKEN_MASK_ALLOW_IDS || desc[0] == GGML_TOKEN_MASK_DENY_IDS) {
            std::vector<int32_t> ids(desc[1]);
            ggml_backend_tensor_get(data.logits->src[2], ids.data(), desc[2]*sizeof(int32_t), ids.size()*sizeof(int32_t));
            assert(std::is_sorted(ids.begin(), ids.end()));
            for (auto id : ids) {
                assert(id >= 0 && id < n_vocab);
            }
        }
        assert(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(data.logits, actual.data(), 0, actual.size()*sizeof(float));
        std::vector<llama_token_data> expected(n_vocab);
        for (int32_t id = 0; id < n_vocab; ++id) {
            expected[id] = {id, 0.0f, 0.0f};
        }
        llama_token_data_array cur = {expected.data(), expected.size(), -1, false};
        llama_sampler_apply(sampler.get(), &cur);
        for (int32_t id = 0; id < n_vocab; ++id) {
            assert(actual[id] == expected[id].logit);
        }
        sampler->iface->backend_reset(sampler.get());
    }
}

int main()
{
    test_candidate_storage();
    test_backend_mask_representations();
    const mask_vocab fixture;
    test_allowed_tokens(fixture);
    test_mask_utf8(fixture);
    test_mask_lazy(fixture);
    test_mask_cache(fixture);
    test_trie_boundaries(fixture);
    test_trie_shared_vocab();
    test_trie_preprocessing_limit();
    printf("Differential masks: %zu accelerated, %zu fallback\n", trie_masks, fallback_masks);

    llama_grammar_parser parsed_grammar;

    std::vector<std::pair<std::string, uint32_t>> expected = {
        {"expr", 2},
        {"expr_6", 6},
        {"expr_7", 7},
        {"ident", 8},
        {"ident_10", 10},
        {"num", 9},
        {"num_11", 11},
        {"root", 0},
        {"root_1", 1},
        {"root_5", 5},
        {"term", 4},
        {"ws", 3},
        {"ws_12", 12},
    };

    std::vector<std::vector<llama_grammar_element>> expected_rules = {
        {{LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_CHAR, 10},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 4}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_END, 0}},
        {{LLAMA_GRETYPE_RULE_REF, 12}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 8},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_RULE_REF, 9},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 40},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 41},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 1}, {LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_ALT, 0}, {LLAMA_GRETYPE_RULE_REF, 1}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 45},
            {LLAMA_GRETYPE_CHAR_ALT, 43},
            {LLAMA_GRETYPE_CHAR_ALT, 42},
            {LLAMA_GRETYPE_CHAR_ALT, 47},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 6}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_ALT, 0}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 11}, {LLAMA_GRETYPE_RULE_REF, 3}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_CHAR_ALT, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_CHAR_ALT, 95},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_RULE_REF, 11},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 32},
            {LLAMA_GRETYPE_CHAR_ALT, 9},
            {LLAMA_GRETYPE_CHAR_ALT, 10},
            {LLAMA_GRETYPE_RULE_REF, 12},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
    };

    for (auto pair : expected)
    {
        parsed_grammar.symbol_ids[pair.first] = pair.second;
    }

    for (auto rule : expected_rules)
    {
        parsed_grammar.rules.emplace_back();
        for (auto element : rule)
        {
            parsed_grammar.rules.back().push_back(element);
        }
    }

    std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());

    llama_grammar * grammar = llama_grammar_init_impl(nullptr, grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    if (grammar == nullptr) {
        throw std::runtime_error("Failed to initialize llama_grammar");
    }

    std::vector<std::vector<llama_grammar_element>> expected_stacks = {
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 40},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 97},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 40},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 97},
        }};

    auto index = 0;
    for (const llama_grammar_stack & stack : llama_grammar_get_stacks(grammar))
    {
        // compare stack to expected_stack
        for (uint32_t i = 0; i < stack.size(); i++)
        {
            const llama_grammar_element * element = stack[i];
            const llama_grammar_element & expected_element = expected_stacks[index][i];

            // pretty print error message before asserting
            if (expected_element.type != element->type || expected_element.value != element->value)
            {
                fprintf(stderr, "index: %d\n", index);
                fprintf(stderr, "expected_element: %d, %u\n", expected_element.type, expected_element.value);
                fprintf(stderr, "actual_element: %d, %u\n", element->type, element->value);
                fprintf(stderr, "expected_element != actual_element\n");
            }

            assert(expected_element.type == element->type && expected_element.value == element->value);
        }
        index++;
    }

    std::vector<llama_grammar_candidate> next_candidates;
    next_candidates.resize(23);

    for (size_t i = 0; i < 23; ++i)
    {
        uint32_t *cp = new uint32_t[2]; // dynamically allocate memory for code_point
        cp[0] = 37 + i;
        cp[1] = 0;
        next_candidates[i] = {i, cp, {}, 0};
    }

    std::vector<std::vector<std::pair<uint32_t, uint16_t>>> expected_reject = {
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
        },
    };

    std::vector<llama_grammar_candidate> rejects = llama_grammar_reject_candidates_for_stack(llama_grammar_get_rules(grammar), llama_grammar_get_stacks(grammar)[0], next_candidates);

    std::vector<std::vector<llama_grammar_candidate>> all_rejects;

    for (std::size_t count = 0; count < llama_grammar_get_stacks(grammar).size(); ++count)
    {
        rejects = llama_grammar_reject_candidates_for_stack(llama_grammar_get_rules(grammar), llama_grammar_get_stacks(grammar)[count], next_candidates);
        all_rejects.push_back(rejects);
    }

    index = 0;
    for (auto rej : all_rejects)
    {
        for (uint32_t i = 0; i < rej.size(); i++)
        {
            auto element = rej[i];
            auto expected_element = expected_reject[index][i];
            assert(element.index == expected_element.first && *element.code_points == expected_element.second);
        }
        index++;
    }

    for (auto &candidate : next_candidates)
    {
        delete[] candidate.code_points;
        candidate.code_points = nullptr;
    }

    llama_grammar_free_impl(grammar);

    return 0;
}
