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

    const std::vector<std::string> gigachat4_regex = {
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    const std::vector<std::pair<std::string, std::vector<std::string>>> gigachat4_tests = {
        { "", {} },
        { "Hello.\nWhat", { "Hello", ".\n", "What" } },
        { "I'm WE'RE CAN'T he'll we'd I'd dog's A'\u017f", { "I'm", " WE'RE", " CAN'T", " he'll", " we'd", " I'd", " dog's", " A'\u017f" } },
        { "\u041f\u0440\u0438\u0432\u0435\u0442\u041c\u0418\u0420\u043f\u0440\u0438\u0432\u0435\u0442 \u043a\u0412\u0442 \u0413\u0413\u0446 \u043c\u0410\u0447 iPhone HTMLParser", { "\u041f\u0440\u0438\u0432\u0435\u0442", "\u041c\u0418\u0420\u043f\u0440\u0438\u0432\u0435\u0442", " \u043a", "\u0412\u0442", " \u0413\u0413\u0446", " \u043c", "\u0410\u0447", " i", "Phone", " HTMLParser" } },
        { "1234567 \uff11\uff12\uff13\uff14 \u0661\u0662\u0663\u0664 \u2167\u00b2\u00b345", { "123", "456", "7", " ", "\uff11\uff12\uff13", "\uff14", " ", "\u0661\u0662\u0663", "\u0664", " ", "\u2167\u00b2\u00b3", "45" } },
        { "foo!\r\n//bar /\n/next", { "foo", "!\r\n//", "bar", " /\n/", "next" } },
        { "a\u0301A A\u0301B \u0301ABC \u4e2d\u6587ABC A\u4e2dB", { "a\u0301", "A", " A\u0301", "B", " \u0301", "ABC", " \u4e2d\u6587", "ABC", " A\u4e2d", "B" } },
        { "\u01c5\u01c6\u01c4\u01c5a \u00dfA \u00b5A \u0130x \u0131A \u02b0A", { "\u01c5\u01c6", "\u01c4\u01c5a", " \u00df", "A", " \u00b5", "A", " \u0130x", " \u0131", "A", " \u02b0", "A" } },
        { "x  y\t\r\n  z\u00a0Q\u2028\n", { "x", " ", " y", "\t\r\n", " ", " z", "\u00a0Q", "\u2028\n" } },
        { "\u0301ABC", { "\u0301", "ABC" } },
        { "a\U0001d400\U0001d41aB", { "a", "\U0001d400\U0001d41a", "B" } },
    };
    for (size_t i = 0; i < gigachat4_tests.size(); ++i) {
        if (unicode_regex_split(gigachat4_tests[i].first, gigachat4_regex, false) != gigachat4_tests[i].second) {
            fprintf(stderr, "unexpected GigaChat4 split for case %zu\n", i);
            success = false;
        }
    }

    const std::string long_word(100000, 'A');
    if (unicode_regex_split(long_word + "a", gigachat4_regex, false) != std::vector<std::string>{long_word + "a"}) {
        fprintf(stderr, "unexpected GigaChat4 split for a long word\n");
        success = false;
    }

    return success ? 0 : 1;
}
