#include "server-decision.h"
#include "testing.h"

#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>
#include <utility>
#include <vector>

// "<|name|>" with U+00A6 in place of "|"
static std::string escaped(const std::string & name) {
    return "<\xC2\xA6" + name + "\xC2\xA6>";
}

static void test_escape_special_tokens(testing & t) {
    t.test("cases", [](testing & t) {
        const std::vector<std::pair<std::string, std::string>> cases = {
            { "",                 ""                               },
            { "no tokens",        "no tokens"                      },
            { "<|im_start|>user", escaped("im_start") + "user"     },
            { "<|a_1|><|B|>",     escaped("a_1") + escaped("B")    },
            { "<|<|a|>",          "<|" + escaped("a")              },
            { "<|a|>|>",          escaped("a") + "|>"              },
            { "<||>",             "<||>"                           },
            { "<|a b|>",          "<|a b|>"                        },
            { "<|a|b|>",          "<|a|b|>"                        },
            { "<|caf\xC3\xA9|>",  "<|caf\xC3\xA9|>"                },
            { "<|abc",            "<|abc"                          },
        };
        for (const auto & [text, expected] : cases) {
            t.assert_equal(text, expected, server_decision_escape_special_tokens(text));
        }
    });

    // every string of up to 6 chars from "<|>a-" gives the text of the regex
    t.test("same as std::regex", [](testing & t) {
        static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
        const std::string chars = "<|>a-";
        size_t n_strings = 0;
        size_t n_diff    = 0;
        for (size_t len = 0; len <= 6; len++) {
            size_t n = 1;
            for (size_t k = 0; k < len; k++) {
                n *= chars.size();
            }
            for (size_t idx = 0; idx < n; idx++) {
                std::string text;
                for (size_t k = 0, v = idx; k < len; k++, v /= chars.size()) {
                    text += chars[v % chars.size()];
                }
                if (server_decision_escape_special_tokens(text) != std::regex_replace(text, re_special, "<\xC2\xA6$1\xC2\xA6>")) {
                    n_diff++;
                }
                n_strings++;
            }
        }
        t.assert_equal("strings that differ, of " + std::to_string(n_strings), (size_t) 0, n_diff);
    });

    // std::regex overflowed the stack on these
    t.test("long input", [](testing & t) {
        const std::string name(1000000, 'a');
        t.assert_true("not closed", server_decision_escape_special_tokens("<|" + name) == "<|" + name);
        t.assert_true("closed", server_decision_escape_special_tokens("<|" + name + "|>") == escaped(name));
    });
}

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    t.test("escape_special_tokens", test_escape_special_tokens);

    return t.summary();
}
