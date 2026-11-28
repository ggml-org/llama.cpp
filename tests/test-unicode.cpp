#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

static int check(const char * name,
                 const std::string & text,
                 const std::vector<std::string> & regex_exprs,
                 const std::vector<std::string> & expected) {
    const auto actual = unicode_regex_split(text, regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "%s: unexpected split:", name);
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        return 1;
    }

    return 0;
}

int main() {
    int nfail = 0;

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
            nfail++;
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
        if (unicode_cpts_from_utf8("a" + test.text + "b") != expected) {
            fprintf(stderr, "unexpected UTF-8 decoding for bytes:");
            for (const unsigned char byte : test.text) {
                fprintf(stderr, " %02x", byte);
            }
            fprintf(stderr, "\n");
            nfail++;
        }
    }

    nfail += check("symbols", " ~foo", { "[~][A-Za-z]+| ?[\\p{S}]+|\\s+" }, { " ~", "foo" });

    // GPT4O uses a hand written splitter, std::regex overflows the stack on long runs
    // keep in sync with LLAMA_VOCAB_PRE_TYPE_GPT4O in src/llama-vocab.cpp
    const std::vector<std::string> gpt4o = {
        "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    nfail += check("gpt4o words", "Hello world!", gpt4o, { "Hello", " world", "!" });
    nfail += check("gpt4o case",  "getUserByID",  gpt4o, { "get", "User", "By", "ID" });

    // a run long enough to exhaust the stack of the std::regex based fallback
    const std::string long_run(1 << 17, 'x');
    nfail += check("gpt4o long run", long_run, gpt4o, { long_run });

    return nfail;
}
