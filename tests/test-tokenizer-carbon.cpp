// Unit tests for the Carbon-3B (HybridDNATokenizer) DNA-aware helpers.
// Doesn't require a GGUF: drives the pure helper functions with a synthetic
// vocab lookup so the test is fast and hermetic.

#include "../src/llama-vocab-carbon.h"

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Synthetic vocab IDs for the test. Real Carbon uses ~151k for BPE + 4k for
// k-mers, but the test only needs distinguishable IDs.
constexpr llama_token ID_DNA_BEGIN = 1000;
constexpr llama_token ID_DNA_END   = 1001;
constexpr llama_token ID_DNA_OOV   = 1002;
constexpr llama_token ID_BPE_BASE  = 2000; // BPE fallback emits ID_BPE_BASE + length

// Maps every valid 6-mer to a unique synthetic ID starting at 10000. Built
// lazily on first lookup.
llama_token kmer_id(const std::string & kmer) {
    static std::map<std::string, llama_token> table;
    if (table.empty()) {
        const char bases[] = "ACGT";
        llama_token next = 10000;
        std::string buf(6, 'A');
        for (char a : bases) for (char b : bases) for (char c : bases)
        for (char d : bases) for (char e : bases) for (char f : bases) {
            buf[0]=a; buf[1]=b; buf[2]=c; buf[3]=d; buf[4]=e; buf[5]=f;
            table[buf] = next++;
        }
    }
    auto it = table.find(kmer);
    return it == table.end() ? LLAMA_TOKEN_NULL : it->second;
}

// A fake BPE tokenizer: emits one synthetic token per UTF-8 byte for ease of
// inspection. The exact mapping doesn't matter — the tests only check that
// non-DNA segments produce the expected *count* of tokens and the right span
// boundaries.
void fake_bpe(const std::string & text, std::vector<llama_token> & output) {
    for (char c : text) {
        output.push_back(ID_BPE_BASE + static_cast<unsigned char>(c));
    }
}

std::string fmt(const std::vector<llama_token> & v) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

#define EXPECT_EQ(actual, expected, label) do {                              \
    if (!((actual) == (expected))) {                                         \
        std::fprintf(stderr,                                                 \
            "FAIL %s:%d %s: expected %s, got %s\n",                          \
            __FILE__, __LINE__, label,                                       \
            fmt(expected).c_str(), fmt(actual).c_str());                     \
        return 1;                                                            \
    }                                                                        \
} while (0)

#define EXPECT_TRUE(cond, label) do {                                        \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, label);  \
        return 1;                                                            \
    }                                                                        \
} while (0)

int test_plain_text_no_dna() {
    std::vector<llama_token> out;
    llama_carbon::tokenize_carbon(
        "hello", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    const std::vector<llama_token> expected = {
        ID_BPE_BASE + 'h', ID_BPE_BASE + 'e', ID_BPE_BASE + 'l',
        ID_BPE_BASE + 'l', ID_BPE_BASE + 'o',
    };
    EXPECT_EQ(out, expected, "plain text falls through to BPE");
    return 0;
}

int test_single_kmer() {
    std::vector<llama_token> out;
    llama_carbon::tokenize_carbon(
        "<dna>ATCGAT</dna>", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    const std::vector<llama_token> expected = {
        ID_DNA_BEGIN, kmer_id("ATCGAT"), ID_DNA_END,
    };
    EXPECT_EQ(out, expected, "single 6-mer between dna tags");
    return 0;
}

int test_two_kmers() {
    std::vector<llama_token> out;
    llama_carbon::tokenize_carbon(
        "<dna>ATCGATCGTACG</dna>", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    const std::vector<llama_token> expected = {
        ID_DNA_BEGIN, kmer_id("ATCGAT"), kmer_id("CGTACG"), ID_DNA_END,
    };
    EXPECT_EQ(out, expected, "two consecutive 6-mers");
    return 0;
}

int test_lowercase_uppercased() {
    std::vector<llama_token> out;
    llama_carbon::tokenize_carbon(
        "<dna>atcgat</dna>", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    const std::vector<llama_token> expected = {
        ID_DNA_BEGIN, kmer_id("ATCGAT"), ID_DNA_END,
    };
    EXPECT_EQ(out, expected, "lowercase is upper-cased before lookup");
    return 0;
}

int test_invalid_base_becomes_oov() {
    std::vector<llama_token> out;
    llama_carbon::tokenize_carbon(
        "<dna>ATCGAN</dna>", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    // 'N' isn't ACGT, so the whole 6-mer is <oov>.
    const std::vector<llama_token> expected = {
        ID_DNA_BEGIN, ID_DNA_OOV, ID_DNA_END,
    };
    EXPECT_EQ(out, expected, "invalid base -> <oov>");
    return 0;
}

int test_trailing_partial_kmer_right_padded() {
    std::vector<llama_token> out;
    // "ATCGA" is 5 bases — right-pad with 'A' to "ATCGAA".
    llama_carbon::tokenize_carbon(
        "<dna>ATCGA</dna>", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    const std::vector<llama_token> expected = {
        ID_DNA_BEGIN, kmer_id("ATCGAA"), ID_DNA_END,
    };
    EXPECT_EQ(out, expected, "trailing partial 6-mer right-padded with A");
    return 0;
}

int test_full_kmer_plus_partial() {
    std::vector<llama_token> out;
    // 8 bases: "ATCGAT" + "CG" -> "ATCGAT", "CGAAAA"
    llama_carbon::tokenize_carbon(
        "<dna>ATCGATCG</dna>", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    const std::vector<llama_token> expected = {
        ID_DNA_BEGIN, kmer_id("ATCGAT"), kmer_id("CGAAAA"), ID_DNA_END,
    };
    EXPECT_EQ(out, expected, "full 6-mer + padded partial");
    return 0;
}

int test_text_then_dna_then_text() {
    std::vector<llama_token> out;
    llama_carbon::tokenize_carbon(
        "hi <dna>ATCGAT</dna>!", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    const std::vector<llama_token> expected = {
        ID_BPE_BASE + 'h', ID_BPE_BASE + 'i', ID_BPE_BASE + ' ',
        ID_DNA_BEGIN, kmer_id("ATCGAT"), ID_DNA_END,
        ID_BPE_BASE + '!',
    };
    EXPECT_EQ(out, expected, "mixed text and DNA");
    return 0;
}

int test_empty_dna_region() {
    std::vector<llama_token> out;
    llama_carbon::tokenize_carbon(
        "<dna></dna>", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    const std::vector<llama_token> expected = { ID_DNA_BEGIN, ID_DNA_END };
    EXPECT_EQ(out, expected, "empty <dna></dna>");
    return 0;
}

int test_unterminated_dna_swallows_rest() {
    std::vector<llama_token> out;
    llama_carbon::tokenize_carbon(
        "x<dna>ATCGAT", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    // 'x' goes through BPE, then <dna>, then ATCGAT, then break (no </dna>).
    const std::vector<llama_token> expected = {
        ID_BPE_BASE + 'x', ID_DNA_BEGIN, kmer_id("ATCGAT"),
    };
    EXPECT_EQ(out, expected, "unterminated <dna> emits begin + DNA + no end");
    return 0;
}

int test_two_dna_regions() {
    std::vector<llama_token> out;
    llama_carbon::tokenize_carbon(
        "<dna>ATCGAT</dna> <dna>CGTACG</dna>", ID_DNA_BEGIN, ID_DNA_END, ID_DNA_OOV, 6,
        kmer_id, fake_bpe, out);
    const std::vector<llama_token> expected = {
        ID_DNA_BEGIN, kmer_id("ATCGAT"), ID_DNA_END,
        ID_BPE_BASE + ' ',
        ID_DNA_BEGIN, kmer_id("CGTACG"), ID_DNA_END,
    };
    EXPECT_EQ(out, expected, "two DNA regions separated by text");
    return 0;
}

int test_kmer_unknown_to_vocab() {
    std::vector<llama_token> out;
    // Force the lookup to miss by using a known-bad k-mer (none possible if
    // ACGT-only) — simulate by passing through emit_dna_kmers with a lookup
    // that always returns LLAMA_TOKEN_NULL.
    auto miss_lookup = [](const std::string &) { return LLAMA_TOKEN_NULL; };
    llama_carbon::emit_dna_kmers("ATCGAT", 6, ID_DNA_OOV, miss_lookup, out);
    const std::vector<llama_token> expected = { ID_DNA_OOV };
    EXPECT_EQ(out, expected, "k-mer missing from vocab -> <oov>");
    return 0;
}

} // namespace

int main() {
    int failed = 0;
    failed += test_plain_text_no_dna();
    failed += test_single_kmer();
    failed += test_two_kmers();
    failed += test_lowercase_uppercased();
    failed += test_invalid_base_becomes_oov();
    failed += test_trailing_partial_kmer_right_padded();
    failed += test_full_kmer_plus_partial();
    failed += test_text_then_dna_then_text();
    failed += test_empty_dna_region();
    failed += test_unterminated_dna_swallows_rest();
    failed += test_two_dna_regions();
    failed += test_kmer_unknown_to_vocab();

    if (failed) {
        std::fprintf(stderr, "test-tokenizer-carbon: %d test(s) FAILED\n", failed);
        return 1;
    }
    std::printf("test-tokenizer-carbon: all tests passed\n");
    return 0;
}
