#include "tests.h"

#include "peg-parser.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <cctype>

static void assert_result_equal(testing & t, common_peg_parse_result_type expected, common_peg_parse_result_type actual) {
    t.assert_equal(common_peg_parse_result_type_name(expected), common_peg_parse_result_type_name(actual));
}

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
        common_peg_parse_result_type expected_result;
    };

    t.test("any", [](testing &t) {
        std::vector<test_case> test_cases {
            // Valid UTF-8 sequences
            {"Hello", "Hello", COMMON_PEG_PARSE_RESULT_SUCCESS},
            {std::string("Caf\xC3\xA9"), std::string("Caf\xC3\xA9"), COMMON_PEG_PARSE_RESULT_SUCCESS},
            {std::string("\xE4\xBD\xA0\xE5\xA5\xBD"), std::string("\xE4\xBD\xA0\xE5\xA5\xBD"), COMMON_PEG_PARSE_RESULT_SUCCESS},
            {std::string("\xF0\x9F\x9A\x80"), std::string("\xF0\x9F\x9A\x80"), COMMON_PEG_PARSE_RESULT_SUCCESS},

            // Incomplete UTF-8 sequences (partial bytes at end)
            {std::string("Caf\xC3"), "Caf", COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},
            {std::string("\xE4\xBD"), "", COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},
            {std::string("\xF0\x9F\x9A"), "", COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},

            // Invalid/malformed UTF-8 sequences
            {std::string("\xFF\xFE"), "", COMMON_PEG_PARSE_RESULT_FAIL},
            {std::string("Hello\x80World"), "Hello", COMMON_PEG_PARSE_RESULT_FAIL},
            {std::string("\xC3\x28"), "", COMMON_PEG_PARSE_RESULT_FAIL},
        };

        auto parser = build_peg_parser([](common_peg_parser_builder& p) {
            return p.sequence({p.one_or_more(p.any()), p.end()});
        });

        for (size_t i = 0; i < test_cases.size(); i++) {
            const auto & tc = test_cases[i];
            std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

            t.test(test_name, [&](testing &t) {
                common_peg_parse_context ctx(tc.input, true);
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
                {std::string("\xE4\xB8\x80"), std::string("\xE4\xB8\x80"), COMMON_PEG_PARSE_RESULT_SUCCESS}, // U+4E00
                {std::string("\xE4\xBD\xA0"), std::string("\xE4\xBD\xA0"), COMMON_PEG_PARSE_RESULT_SUCCESS}, // U+4F60
                {std::string("\xE5\xA5\xBD"), std::string("\xE5\xA5\xBD"), COMMON_PEG_PARSE_RESULT_SUCCESS}, // U+597D
                {std::string("\xE9\xBF\xBF"), std::string("\xE9\xBF\xBF"), COMMON_PEG_PARSE_RESULT_SUCCESS}, // U+9FFF

                // Outside range - should fail
                {"a", "", COMMON_PEG_PARSE_RESULT_FAIL},                                                     // ASCII
                {std::string("\xE4\xB7\xBF"), "", COMMON_PEG_PARSE_RESULT_FAIL},                            // U+4DFF (before range)
                {std::string("\xEA\x80\x80"), "", COMMON_PEG_PARSE_RESULT_FAIL},                            // U+A000 (after range)

                // Incomplete sequences in range
                {std::string("\xE4\xB8"), "", COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},                     // Incomplete U+4E00
                {std::string("\xE5\xA5"), "", COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},                     // Incomplete U+597D
            };

            auto parser = build_peg_parser([](common_peg_parser_builder& p) {
                return p.sequence({p.chars(R"([\u4E00-\u9FFF])"), p.end()});
            });

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    common_peg_parse_context ctx(tc.input, true);
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
                {std::string("\xF0\x9F\x98\x80"), std::string("\xF0\x9F\x98\x80"), COMMON_PEG_PARSE_RESULT_SUCCESS}, // U+1F600
                {std::string("\xF0\x9F\x98\x81"), std::string("\xF0\x9F\x98\x81"), COMMON_PEG_PARSE_RESULT_SUCCESS}, // U+1F601
                {std::string("\xF0\x9F\x99\x8F"), std::string("\xF0\x9F\x99\x8F"), COMMON_PEG_PARSE_RESULT_SUCCESS}, // U+1F64F

                // Outside range
                {std::string("\xF0\x9F\x97\xBF"), "", COMMON_PEG_PARSE_RESULT_FAIL}, // U+1F5FF (before range)
                {std::string("\xF0\x9F\x99\x90"), "", COMMON_PEG_PARSE_RESULT_FAIL}, // U+1F650 (after range)
                {std::string("\xF0\x9F\x9A\x80"), "", COMMON_PEG_PARSE_RESULT_FAIL}, // U+1F680 (outside range)

                // Incomplete sequences
                {std::string("\xF0\x9F\x98"), "", COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT}, // Incomplete emoji
                {std::string("\xF0\x9F"), "", COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},     // Very incomplete
            };

            auto parser = build_peg_parser([](common_peg_parser_builder& p) {
                return p.sequence({p.chars(R"([\U0001F600-\U0001F64F])"), p.end()});
            });

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    common_peg_parse_context ctx(tc.input, true);
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
                {std::string("\xE4\xB8\x80"), std::string("\xE4\xB8\x80"), COMMON_PEG_PARSE_RESULT_SUCCESS}, // U+4E00
                {std::string("\xE4\xBD\xA0"), std::string("\xE4\xBD\xA0"), COMMON_PEG_PARSE_RESULT_SUCCESS}, // U+4F60

                // Match emoticons
                {std::string("\xF0\x9F\x98\x80"), std::string("\xF0\x9F\x98\x80"), COMMON_PEG_PARSE_RESULT_SUCCESS}, // U+1F600

                // Match ASCII digits
                {"5", "5", COMMON_PEG_PARSE_RESULT_SUCCESS},

                // Don't match outside any range
                {"a", "", COMMON_PEG_PARSE_RESULT_FAIL},
                {std::string("\xF0\x9F\x9A\x80"), "", COMMON_PEG_PARSE_RESULT_FAIL}, // U+1F680

                // Incomplete
                {std::string("\xE4\xB8"), "", COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},
                {std::string("\xF0\x9F\x98"), "", COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},
            };

            auto parser = build_peg_parser([](common_peg_parser_builder& p) {
                return p.sequence({p.chars(R"([\u4E00-\u9FFF\U0001F600-\U0001F64F0-9])"), p.end()});
            });

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    common_peg_parse_context ctx(tc.input, true);
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

    t.test("until parser", [](testing &t) {
        t.test("ASCII delimiter with Unicode content", [](testing &t) {
            std::vector<test_case> test_cases {
                // CJK characters before delimiter
                {std::string("\xE4\xBD\xA0\xE5\xA5\xBD</tag>"), std::string("\xE4\xBD\xA0\xE5\xA5\xBD"), COMMON_PEG_PARSE_RESULT_SUCCESS},

                // Emoji before delimiter
                {std::string("\xF0\x9F\x98\x80</tag>"), std::string("\xF0\x9F\x98\x80"), COMMON_PEG_PARSE_RESULT_SUCCESS},

                // Mixed content
                {std::string("Hello \xE4\xB8\x96\xE7\x95\x8C!</tag>"), std::string("Hello \xE4\xB8\x96\xE7\x95\x8C!"), COMMON_PEG_PARSE_RESULT_SUCCESS},
            };

            auto parser = build_peg_parser([](common_peg_parser_builder& p) {
                return p.until("</tag>");
            });

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    common_peg_parse_context ctx(tc.input, false);
                    auto result = parser.parse(ctx);

                    assert_result_equal(t, tc.expected_result, result.type);

                    if (result.success()) {
                        std::string matched = tc.input.substr(result.start, result.end - result.start);
                        t.assert_equal(tc.expected_text, matched);
                    }
                });
            }
        });

        t.test("incomplete UTF-8 at end", [](testing &t) {
            std::vector<test_case> test_cases {
                // Incomplete emoji at end, no delimiter
                {std::string("content\xF0\x9F\x98"), std::string("content"), COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},

                // Incomplete CJK at end, no delimiter
                {std::string("hello\xE4\xB8"), std::string("hello"), COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},

                // Complete content, no delimiter (should consume all valid UTF-8)
                {std::string("\xE4\xBD\xA0\xE5\xA5\xBD"), std::string("\xE4\xBD\xA0\xE5\xA5\xBD"), COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},
            };

            auto parser = build_peg_parser([](common_peg_parser_builder& p) {
                return p.until("</tag>");
            });

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    common_peg_parse_context ctx(tc.input, true);
                    auto result = parser.parse(ctx);

                    assert_result_equal(t, tc.expected_result, result.type);

                    if (result.success() || result.need_more_input()) {
                        std::string matched = tc.input.substr(result.start, result.end - result.start);
                        t.assert_equal(tc.expected_text, matched);
                    }
                });
            }
        });

        t.test("malformed UTF-8", [](testing &t) {
            std::vector<test_case> test_cases {
                // Invalid UTF-8 bytes
                {std::string("Hello\xFF\xFE"), "", COMMON_PEG_PARSE_RESULT_FAIL},

                // Continuation byte without lead byte
                {std::string("Hello\x80World"), "", COMMON_PEG_PARSE_RESULT_FAIL},

                // Invalid continuation byte
                {std::string("\xC3\x28"), "", COMMON_PEG_PARSE_RESULT_FAIL},
            };

            auto parser = build_peg_parser([](common_peg_parser_builder& p) {
                return p.until("</tag>");
            });

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    common_peg_parse_context ctx(tc.input, false);
                    auto result = parser.parse(ctx);

                    assert_result_equal(t, tc.expected_result, result.type);
                });
            }
        });
    });

    t.test("json_string parser", [](testing &t) {
        t.test("valid UTF-8 characters", [](testing &t) {
            std::vector<test_case> test_cases {
                // ASCII only
                {"Hello World\"", "Hello World", COMMON_PEG_PARSE_RESULT_SUCCESS},

                // 2-byte UTF-8 (accented characters)
                {std::string("Caf\xC3\xA9\""), std::string("Caf\xC3\xA9"), COMMON_PEG_PARSE_RESULT_SUCCESS},

                // 3-byte UTF-8 (CJK)
                {std::string("\xE4\xBD\xA0\xE5\xA5\xBD\""), std::string("\xE4\xBD\xA0\xE5\xA5\xBD"), COMMON_PEG_PARSE_RESULT_SUCCESS},

                // 4-byte UTF-8 (emoji)
                {std::string("\xF0\x9F\x98\x80\""), std::string("\xF0\x9F\x98\x80"), COMMON_PEG_PARSE_RESULT_SUCCESS},

                // Mixed content
                {std::string("Hello \xE4\xB8\x96\xE7\x95\x8C!\""), std::string("Hello \xE4\xB8\x96\xE7\x95\x8C!"), COMMON_PEG_PARSE_RESULT_SUCCESS},
            };

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    auto parser = build_peg_parser([](common_peg_parser_builder& p) {
                        return p.sequence({p.json_string_content(), p.literal("\"")});
                    });

                    common_peg_parse_context ctx(tc.input, false);
                    auto result = parser.parse(ctx);

                    assert_result_equal(t, tc.expected_result, result.type);

                    if (result.success()) {
                        std::string matched = tc.input.substr(result.start, result.end - result.start - 1);  // -1 to exclude closing quote
                        t.assert_equal(tc.expected_text, matched);
                    }
                });
            }
        });

        t.test("incomplete UTF-8", [](testing &t) {
            std::vector<test_case> test_cases {
                // Incomplete 2-byte sequence
                {std::string("Caf\xC3"), std::string("Caf"), COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},

                // Incomplete 3-byte sequence
                {std::string("Hello\xE4\xB8"), std::string("Hello"), COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},

                // Incomplete 4-byte sequence
                {std::string("Text\xF0\x9F\x98"), std::string("Text"), COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},

                // Incomplete at very start
                {std::string("\xE4\xBD"), std::string(""), COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT},
            };

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    auto parser = build_peg_parser([](common_peg_parser_builder& p) {
                        return p.json_string_content();
                    });

                    common_peg_parse_context ctx(tc.input, true);
                    auto result = parser.parse(ctx);

                    assert_result_equal(t, tc.expected_result, result.type);

                    if (result.need_more_input()) {
                        std::string matched = tc.input.substr(result.start, result.end - result.start);
                        t.assert_equal(tc.expected_text, matched);
                    }
                });
            }
        });

        t.test("malformed UTF-8", [](testing &t) {
            std::vector<test_case> test_cases {
                // Invalid UTF-8 bytes
                {std::string("Hello\xFF\xFE"), "", COMMON_PEG_PARSE_RESULT_FAIL},

                // Continuation byte without lead byte
                {std::string("Hello\x80World"), "", COMMON_PEG_PARSE_RESULT_FAIL},

                // Invalid continuation byte
                {std::string("\xC3\x28"), "", COMMON_PEG_PARSE_RESULT_FAIL},

                // Overlong encoding (security issue)
                {std::string("\xC0\x80"), "", COMMON_PEG_PARSE_RESULT_FAIL},
            };

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    auto parser = build_peg_parser([](common_peg_parser_builder& p) {
                        return p.json_string_content();
                    });

                    common_peg_parse_context ctx(tc.input, false);
                    auto result = parser.parse(ctx);

                    assert_result_equal(t, tc.expected_result, result.type);
                });
            }
        });

        t.test("escape sequences with UTF-8", [](testing &t) {
            std::vector<test_case> test_cases {
                // Unicode escape sequence
                {"Hello\\u0041\"", "Hello\\u0041", COMMON_PEG_PARSE_RESULT_SUCCESS},

                // Mix of UTF-8 and escape sequences
                {std::string("\xE4\xBD\xA0\\n\xE5\xA5\xBD\""), std::string("\xE4\xBD\xA0\\n\xE5\xA5\xBD"), COMMON_PEG_PARSE_RESULT_SUCCESS},

                // Escaped quote in UTF-8 string
                {std::string("\xE4\xBD\xA0\\\"\xE5\xA5\xBD\""), std::string("\xE4\xBD\xA0\\\"\xE5\xA5\xBD"), COMMON_PEG_PARSE_RESULT_SUCCESS},
            };

            for (size_t i = 0; i < test_cases.size(); i++) {
                const auto & tc = test_cases[i];
                std::string test_name = "case " + std::to_string(i) + ": " + hex_dump(tc.input);

                t.test(test_name, [&](testing &t) {
                    auto parser = build_peg_parser([](common_peg_parser_builder& p) {
                        return p.sequence({p.json_string_content(), p.literal("\"")});
                    });

                    common_peg_parse_context ctx(tc.input, false);
                    auto result = parser.parse(ctx);

                    assert_result_equal(t, tc.expected_result, result.type);

                    if (result.success()) {
                        std::string matched = tc.input.substr(result.start, result.end - result.start - 1);  // -1 to exclude closing quote
                        t.assert_equal(tc.expected_text, matched);
                    }
                });
            }
        });
    });
}
