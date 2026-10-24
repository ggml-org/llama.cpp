#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

int main() {
    bool success = true;
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
        if (unicode_cpts_from_utf8("a" + test.text + "b") != expected) {
            fprintf(stderr, "unexpected UTF-8 decoding for bytes:");
            for (const unsigned char byte : test.text) {
                fprintf(stderr, " %02x", byte);
            }
            fprintf(stderr, "\n");
            success = false;
        }
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

    // Laguna pre-tokenizer: \r\n must stay together, as in tokenizer.json (expected pieces from HF tokenizers)
    const std::vector<std::string> laguna_regex_exprs = {
        "(?:\\r?\\n)+(?:[^\\r\\n]|\\r(?!\\n))*|(?:[^\\r\\n]|\\r(?!\\n))+",
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    const struct {
        std::string text;
        std::vector<std::string> expected;
    } laguna_tests[] = {
        { "x\r\ny",                     { "x", "\r\n", "y" } },
        { "a\r\n\r\nb",                 { "a", "\r\n\r\n", "b" } },
        { "foo.\r\nbar",                { "foo", ".", "\r\n", "bar" } },
        { "\n\rX",                      { "\n\r", "X" } },
        { "a\rb",                       { "a", "\r", "b" } },
        { "x \r\n  y",                  { "x", " ", "\r\n", " ", " y" } },
        { "if x:\r\n    return 1\r\n",  { "if", " x", ":", "\r\n", "   ", " return", " ", "1", "\r\n" } },
    };
    for (const auto & test : laguna_tests) {
        const auto pieces = unicode_regex_split(test.text, laguna_regex_exprs, false);
        if (pieces != test.expected) {
            fprintf(stderr, "unexpected Laguna split:");
            for (const auto & piece : pieces) {
                fprintf(stderr, " [%s]", piece.c_str());
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    return success ? 0 : 1;
}
