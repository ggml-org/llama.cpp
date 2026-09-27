#include "../src/unicode.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

int main() {
    {
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
            return 1;
        }
    }

    {
        const std::vector<std::string> regex_exprs = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        const std::string input = "ab\u200ccd ef\u200dgh cafe\u0301";
        const std::vector<std::string> expected = {
            "ab\u200ccd",
            " ef\u200dgh",
            " cafe\u0301",
        };

        try {
            const auto actual = unicode_regex_split(input, regex_exprs, false);

            if (actual != expected) {
                fprintf(stderr, "unexpected K2-Horizon split:");
                for (const auto & piece : actual) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }
                fprintf(stderr, "\n");
                return 1;
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "K2-Horizon regex split threw exception: %s\n", e.what());
            return 1;
        }
    }


    {
        const std::vector<std::string> llama3_regex = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        const std::vector<std::string> k2_horizon_regex = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        const std::vector<std::string> inputs = {
            "Hello, world!",
            "can't won't we're they'll",
            "123 1234 1234567",
            "Hello!!!\\nNext line",
            "alpha beta gamma",
            " café résumé",
        };

        for (const auto & input : inputs) {
            const auto llama3 = unicode_regex_split(input, llama3_regex, false);
            const auto k2 = unicode_regex_split(input, k2_horizon_regex, false);

            if (llama3 != k2) {
                fprintf(stderr, "K2-Horizon diverged from Llama 3 for ordinary input: %s\n", input.c_str());

                fprintf(stderr, "Llama 3:");
                for (const auto & piece : llama3) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\nK2-Horizon:");
                for (const auto & piece : k2) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\n");
                return 1;
            }
        }
    }


    {
        const std::vector<std::string> k2_horizon_regex = {
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
            "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        auto check_k2 = [&] (
                const char * name,
                const std::string & input,
                const std::vector<std::string> & expected) {
            const auto actual = unicode_regex_split(input, k2_horizon_regex, false);

            if (actual != expected) {
                fprintf(stderr, "K2-Horizon reference mismatch for %s\n", name);

                fprintf(stderr, "expected:");
                for (const auto & piece : expected) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\nactual:");
                for (const auto & piece : actual) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\n");
                return false;
            }

            return true;
        };

        if (!check_k2(
                "ZWNJ",
                "ab\u200ccd",
                { "ab\u200ccd" })) {
            return 1;
        }

        if (!check_k2(
                "ZWJ",
                "ef\u200dgh",
                { "ef\u200dgh" })) {
            return 1;
        }

        if (!check_k2(
                "combined ZWNJ/ZWJ",
                "ab\u200ccd ef\u200dgh",
                { "ab\u200ccd", " ef\u200dgh" })) {
            return 1;
        }

        // The reference tokenizer NFC-normalizes cafe + combining acute
        // to the composed form before pre-tokenization.
        if (!check_k2(
                "NFC accent",
                "caf\u00e9",
                { "caf\u00e9" })) {
            return 1;
        }

        if (!check_k2(
                "Persian ZWNJ",
                "\u0645\u06cc\u200c\u0631\u0648\u0645",
                { "\u0645\u06cc\u200c\u0631\u0648\u0645" })) {
            return 1;
        }

        if (!check_k2(
                "Devanagari ZWJ",
                "\u0915\u094d\u200d\u0937",
                { "\u0915\u094d\u200d\u0937" })) {
            return 1;
        }

        if (!check_k2(
                "contractions",
                "can't won't we're they'll",
                { "can", "'t", " won", "'t", " we", "'re", " they", "'ll" })) {
            return 1;
        }

        if (!check_k2(
                "Unicode contraction",
                "'\u017fa",
                { "'\u017f", "a" })) {
            return 1;
        }

        if (!check_k2(
                "empty input",
                "",
                { })) {
            return 1;
        }

        if (!check_k2(
                "numbers",
                "123 1234 1234567",
                { "123", " ", "123", "4", " ", "123", "456", "7" })) {
            return 1;
        }

        if (!check_k2(
                "punctuation/newline",
                "Hello!!!\nNext line",
                { "Hello", "!!!\n", "Next", " line" })) {
            return 1;
        }
    }

    return 0;
}
