#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <vector>

static const std::string shared_regex =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

static std::vector<std::string> split(const std::string & text, const std::string & regex) {
    return unicode_regex_split(text, {
        "\\p{N}{1,3}",
        u8"[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
        regex,
    }, false);
}

int main() {
    const std::vector<std::string> cases = {
        "Hello, world! DeepSeek tokenizer regression.\n",
        " leading words\twith whitespace  and trailing spaces   \n",
        u8"cafe\u0301 caf\u00e9 na\u00efve \u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac\n",
        u8"\u4e2d\u6587\u6bb5\u843d\u3002\u3072\u3089\u304c\u306a\u3001\u30ab\u30bf\u30ab\u30ca\uff01\n",
        "~A ~~ !Z $value _name 1234 999999\r\n",
        "json {\"tool\":\"read_file\",\"path\":\"/tmp/example\"}\n",
        "blank-lines-follow\r\n\r\n\nend\n",
    };

    // Wrap the regex to bypass the custom handler.
    const std::string fallback_regex = "(?:" + shared_regex + ")";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto custom = split(cases[i], shared_regex);
        const auto fallback = split(cases[i], fallback_regex);
        if (custom != fallback) {
            fprintf(stderr, "Shared regex mismatch for case %zu: custom=%zu fallback=%zu\n", i, custom.size(), fallback.size());
            return 1;
        }
    }

    const std::string long_letters(131072, 'Z');
    for (const auto & text : { long_letters, "!" + long_letters }) {
        const auto pieces = split(text, shared_regex);
        if (pieces.size() != 1 || pieces[0] != text) {
            fprintf(stderr, "Shared regex long-run mismatch: input=%zu pieces=%zu\n", text.size(), pieces.size());
            return 1;
        }
    }

    return 0;
}
