#include "tests.h"
#include "test_harness.h"

#include "chat-peg-parser.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <cctype>

// Assertions specific to chat-peg-parser
static void assert_result_equal(testing & t, common_chat_parse_result_type expected, common_chat_parse_result_type actual) {
    t.assert_equal(common_chat_parse_result_type_name(expected), common_chat_parse_result_type_name(actual));
}

// Helper function to produce hex dump for non-printable characters
static std::string hex_dump(const std::string& str) {
    std::ostringstream oss;
    for (unsigned char c : str) {
        if (std::isprint(c)) {
            oss << c;
        } else {
            oss << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c);
        }
    }
    return oss.str();
}

void test_unicode(testing &t) {
    struct test_case {
        std::string input;
        std::string expected_text;
        common_chat_parse_result_type expected_result;
    };

    t.test("any", [](testing &t) {
        std::vector<test_case> test_cases {
            // Valid UTF-8 sequences
            {"Hello", "Hello", COMMON_CHAT_PARSE_RESULT_SUCCESS},
            {std::string("Caf\xC3\xA9"), std::string("Caf\xC3\xA9"), COMMON_CHAT_PARSE_RESULT_SUCCESS},
            {std::string("\xE4\xBD\xA0\xE5\xA5\xBD"), std::string("\xE4\xBD\xA0\xE5\xA5\xBD"), COMMON_CHAT_PARSE_RESULT_SUCCESS},
            {std::string("\xF0\x9F\x9A\x80"), std::string("\xF0\x9F\x9A\x80"), COMMON_CHAT_PARSE_RESULT_SUCCESS},

            // Incomplete UTF-8 sequences (partial bytes at end)
            {std::string("Caf\xC3"), "Caf", COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT},
            {std::string("\xE4\xBD"), "", COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT},
            {std::string("\xF0\x9F\x9A"), "", COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT},

            // Invalid/malformed UTF-8 sequences
            {std::string("\xFF\xFE"), "", COMMON_CHAT_PARSE_RESULT_FAIL},
            {std::string("Hello\x80World"), "Hello", COMMON_CHAT_PARSE_RESULT_FAIL},
            {std::string("\xC3\x28"), "", COMMON_CHAT_PARSE_RESULT_FAIL},
        };

        auto parser = build_peg_parser([](common_chat_peg_parser_builder& p) {
            return p.one_or_more(p.any()) + p.end();
        });

        for (size_t i = 0; i < test_cases.size(); i++) {
            const auto & tc = test_cases[i];
            std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

            t.test(test_name, [&](testing &t) {
                common_chat_parse_context ctx(tc.input, false);
                auto result = parser.parse(ctx);

                // Assert result type matches
                assert_result_equal(t, tc.expected_result, result.type);

                // Assert matched text if success or need_more_input
                if (result.success() || result.need_more_input()) {
                    std::string matched = tc.input.substr(result.start, result.end - result.start);
                    t.assert_equal(tc.expected_text, matched);
                }
            });
        }
    });

    t.test("char classes", [](testing &t) {
        t.test("unicode range U+4E00-U+9FFF (CJK)", [](testing &t) {
            std::vector<test_case> test_cases {
                // Within range - CJK Unified Ideographs
                {std::string("\xE4\xB8\x80"), std::string("\xE4\xB8\x80"), COMMON_CHAT_PARSE_RESULT_SUCCESS}, // U+4E00
                {std::string("\xE4\xBD\xA0"), std::string("\xE4\xBD\xA0"), COMMON_CHAT_PARSE_RESULT_SUCCESS}, // U+4F60
                {std::string("\xE5\xA5\xBD"), std::string("\xE5\xA5\xBD"), COMMON_CHAT_PARSE_RESULT_SUCCESS}, // U+597D
                {std::string("\xE9\xBF\xBF"), std::string("\xE9\xBF\xBF"), COMMON_CHAT_PARSE_RESULT_SUCCESS}, // U+9FFF

                // Outside range - should fail
                {"a", "", COMMON_CHAT_PARSE_RESULT_FAIL},                                                     // ASCII
                {std::string("\xE4\xB7\xBF"), "", COMMON_CHAT_PARSE_RESULT_FAIL},                            // U+4DFF (before range)
                {std::string("\xEA\x80\x80"), "", COMMON_CHAT_PARSE_RESULT_FAIL},                            // U+A000 (after range)

                // Incomplete sequences in range
                {std::string("\xE4\xB8"), "", COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT},                     // Incomplete U+4E00
                {std::string("\xE5\xA5"), "", COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT},                     // Incomplete U+597D
            };

            auto parser = build_peg_parser([](common_chat_peg_parser_builder& p) {
                return p.chars(R"([\u4E00-\u9FFF])") + p.end();
            });

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    common_chat_parse_context ctx(tc.input, false);
                    auto result = parser.parse(ctx);

                    // Assert result type matches
                    assert_result_equal(t, tc.expected_result, result.type);

                    // Assert matched text if success or need_more_input
                    if (result.success() || result.need_more_input()) {
                        std::string matched = tc.input.substr(result.start, result.end - result.start);
                        t.assert_equal(tc.expected_text, matched);
                    }
                });
            }
        });

        t.test("unicode range U+1F600-U+1F64F (emoticons)", [](testing &t) {
            std::vector<test_case> test_cases {
                // Within range - Emoticons (all 4-byte UTF-8)
                {std::string("\xF0\x9F\x98\x80"), std::string("\xF0\x9F\x98\x80"), COMMON_CHAT_PARSE_RESULT_SUCCESS}, // U+1F600
                {std::string("\xF0\x9F\x98\x81"), std::string("\xF0\x9F\x98\x81"), COMMON_CHAT_PARSE_RESULT_SUCCESS}, // U+1F601
                {std::string("\xF0\x9F\x99\x8F"), std::string("\xF0\x9F\x99\x8F"), COMMON_CHAT_PARSE_RESULT_SUCCESS}, // U+1F64F

                // Outside range
                {std::string("\xF0\x9F\x97\xBF"), "", COMMON_CHAT_PARSE_RESULT_FAIL}, // U+1F5FF (before range)
                {std::string("\xF0\x9F\x99\x90"), "", COMMON_CHAT_PARSE_RESULT_FAIL}, // U+1F650 (after range)
                {std::string("\xF0\x9F\x9A\x80"), "", COMMON_CHAT_PARSE_RESULT_FAIL}, // U+1F680 (outside range)

                // Incomplete sequences
                {std::string("\xF0\x9F\x98"), "", COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT}, // Incomplete emoji
                {std::string("\xF0\x9F"), "", COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT},     // Very incomplete
            };

            auto parser = build_peg_parser([](common_chat_peg_parser_builder& p) {
                return p.chars(R"([\U0001F600-\U0001F64F])") + p.end();
            });

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    common_chat_parse_context ctx(tc.input, false);
                    auto result = parser.parse(ctx);

                    // Assert result type matches
                    assert_result_equal(t, tc.expected_result, result.type);

                    // Assert matched text if success or need_more_input
                    if (result.success() || result.need_more_input()) {
                        std::string matched = tc.input.substr(result.start, result.end - result.start);
                        t.assert_equal(tc.expected_text, matched);
                    }
                });
            }
        });

        t.test("mixed unicode ranges", [](testing &t) {
            std::vector<test_case> test_cases {
                // Match CJK
                {std::string("\xE4\xB8\x80"), std::string("\xE4\xB8\x80"), COMMON_CHAT_PARSE_RESULT_SUCCESS}, // U+4E00
                {std::string("\xE4\xBD\xA0"), std::string("\xE4\xBD\xA0"), COMMON_CHAT_PARSE_RESULT_SUCCESS}, // U+4F60

                // Match emoticons
                {std::string("\xF0\x9F\x98\x80"), std::string("\xF0\x9F\x98\x80"), COMMON_CHAT_PARSE_RESULT_SUCCESS}, // U+1F600

                // Match ASCII digits
                {"5", "5", COMMON_CHAT_PARSE_RESULT_SUCCESS},

                // Don't match outside any range
                {"a", "", COMMON_CHAT_PARSE_RESULT_FAIL},
                {std::string("\xF0\x9F\x9A\x80"), "", COMMON_CHAT_PARSE_RESULT_FAIL}, // U+1F680

                // Incomplete
                {std::string("\xE4\xB8"), "", COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT},
                {std::string("\xF0\x9F\x98"), "", COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT},
            };

            auto parser = build_peg_parser([](common_chat_peg_parser_builder& p) {
                return p.chars(R"([\u4E00-\u9FFF\U0001F600-\U0001F64F0-9])") + p.end();
            });

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    common_chat_parse_context ctx(tc.input, false);
                    auto result = parser.parse(ctx);

                    // Assert result type matches
                    assert_result_equal(t, tc.expected_result, result.type);

                    // Assert matched text if success or need_more_input
                    if (result.success() || result.need_more_input()) {
                        std::string matched = tc.input.substr(result.start, result.end - result.start);
                        t.assert_equal(tc.expected_text, matched);
                    }
                });
            }
        });
    });
}
