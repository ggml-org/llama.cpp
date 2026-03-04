#include "tests.h"

void test_basic(testing & t) {
    t.test("chars", [](testing & t) {
        // Test common escape sequences - newline
        t.test("escape_sequence_newline", [](testing &t) {
            auto common_chat_combinator_parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("[\\n\\t\\\\]"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("\n");
            result = common_chat_combinator_parser.parse(ctx);
            t.assert_equal("escape_sequence_newline", true, result.success());
        });

        // Test common escape sequences - tab
        t.test("escape_sequence_tab", [](testing &t) {
            auto common_chat_combinator_parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("[\\n\\t\\\\]"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("\t");
            result = common_chat_combinator_parser.parse(ctx);
            t.assert_equal("escape_sequence_tab", true, result.success());
        });

        // Test common escape sequences - backslash
        t.test("escape_sequence_backslash", [](testing &t) {
            auto common_chat_combinator_parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("[\\n\\t\\\\]"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("\\");
            result = common_chat_combinator_parser.parse(ctx);
            t.assert_equal("escape_sequence_backslash", true, result.success());
        });

        // Test common escape sequences - space (should ())
        t.test("escape_sequence_space_fail", [](testing &t) {
            auto common_chat_combinator_parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("[\\n\\t\\\\]"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context(" ");
            result = common_chat_combinator_parser.parse(ctx);
            t.assert_equal("escape_sequence_space_fail", true, result.fail());
        });

        // Test escaped dash - 'a' should succeed
        t.test("escaped_dash_a", [](testing &t) {
            auto common_chat_combinator_parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("[a\\-z]"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("a");
            result = common_chat_combinator_parser.parse(ctx);
            t.assert_equal("escaped_dash_a", true, result.success());
        });

        // Test escaped dash - '-' should succeed (literal dash)
        t.test("escaped_dash_literal", [](testing &t) {
            auto common_chat_combinator_parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("[a\\-z]"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("-");
            result = common_chat_combinator_parser.parse(ctx);
            t.assert_equal("escaped_dash_literal", true, result.success());
        });

        // Test escaped dash - 'z' should succeed
        t.test("escaped_dash_z", [](testing &t) {
            auto common_chat_combinator_parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("[a\\-z]"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("z");
            result = common_chat_combinator_parser.parse(ctx);
            t.assert_equal("escaped_dash_z", true, result.success());
        });

        // Test escaped dash - 'b' should NOT match (since \- is literal dash, not range)
        t.test("escaped_dash_b_fail", [](testing &t) {
            auto common_chat_combinator_parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("[a\\-z]"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("b");
            result = common_chat_combinator_parser.parse(ctx);
            t.assert_equal("escaped_dash_b_fail", true, result.fail());
        });
    });


    t.test("optional", [](testing & t) {
        // Full match with optional part present
        t.test("optional_present", [](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                return p.literal("hello") + p.optional(p.literal(" world"));
            });

            auto ctx    = common_peg_parse_context("hello world");
            auto result = parser.parse(ctx);
            t.assert_equal("optional_present", true, result.success());
            t.assert_equal("optional_present_end", 11u, result.end);
        });

        // Full match with optional part absent
        t.test("optional_absent", [](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                return p.literal("hello") + p.optional(p.literal(" world"));
            });

            auto ctx    = common_peg_parse_context("hello", false);
            auto result = parser.parse(ctx);
            t.assert_equal("optional_absent", true, result.success());
            t.assert_equal("optional_absent_end", 5u, result.end);
        });

        // Partial match - waiting for more input to determine if optional matches
        t.test("partial_match_need_more", [](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                return p.literal("hello") + p.optional(p.literal(" world"));
            });

            auto ctx    = common_peg_parse_context("hello ", true);
            auto result = parser.parse(ctx);
            t.assert_equal("partial_match_need_more", true, result.need_more_input());
        });
    });

    t.test("partial parsing", [](testing & t) {
        // Literals - Basic Success
        t.test("literal_success", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("hello"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("hello");
            result = parser.parse(ctx);
            t.assert_equal("literal_success", true, result.success());
        });

        // Char Classes - Basic Lowercase Success
        t.test("char_class_lowercase_success", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("a-z"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("a");
            result = parser.parse(ctx);
            t.assert_equal("char_class_lowercase_success", true, result.success());
        });

        // Char Classes - Uppercase Fail
        t.test("char_class_uppercase_fail", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("a-z"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("A");
            result = parser.parse(ctx);
            t.assert_equal("char_class_uppercase_fail", true, result.fail());
        });

        // Char Classes with Dash - Lowercase Success
        t.test("char_class_with_dash_lowercase", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("a-z-"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("f");
            result = parser.parse(ctx);
            t.assert_equal("char_class_with_dash_lowercase", true, result.success());
        });

        // Char Classes with Dash - Literal Dash Success
        t.test("char_class_with_dash_literal_dash", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("a-z-"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("-");
            result = parser.parse(ctx);
            t.assert_equal("char_class_with_dash_literal_dash", true, result.success());
        });

        // Char Classes with Dash - Uppercase Fail
        t.test("char_class_with_dash_uppercase_fail", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.chars("a-z-"); });

            common_peg_parse_context ctx;
            common_peg_parse_result  result;

            ctx    = common_peg_parse_context("A");
            result = parser.parse(ctx);
            t.assert_equal("char_class_with_dash_uppercase_fail", true, result.fail());
        });

        // Sequences - Partial Match 1
        t.test("sequence_partial_match_1", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("<think>") + p.literal("</think>"); });

            auto ctx    = common_peg_parse_context("<thi", true);
            auto result = parser.parse(ctx);
            t.assert_equal("sequence_partial_match_1", true, result.need_more_input());
        });

        // Sequences - Partial Match 2
        t.test("sequence_partial_match_2", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("begin") + p.literal("end"); });

            auto ctx    = common_peg_parse_context("begin", true);
            auto result = parser.parse(ctx);
            t.assert_equal("sequence_partial_match_2", true, result.need_more_input());
        });

        // Sequences - Partial Match 3
        t.test("sequence_partial_match_3", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("<think>") + p.literal("</think>"); });

            auto ctx    = common_peg_parse_context("<think></", true);
            auto result = parser.parse(ctx);
            t.assert_equal("sequence_partial_match_3", true, result.need_more_input());
        });

        // Sequences - Full Match
        t.test("sequence_full_match", [&](testing & t) {
            auto common_chat_combinator_parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("hello") + p.literal("world"); });

            auto ctx    = common_peg_parse_context("helloworld", false);
            auto result = common_chat_combinator_parser.parse(ctx);
            t.assert_equal("sequence_full_match", true, result.success());
        });

        // Sequences - No Match
        t.test("sequence_no_match", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("<think>") + p.literal("</think>"); });

            auto ctx    = common_peg_parse_context("<think>I am common_chat_combinator_parser", true);
            auto result = parser.parse(ctx);
            t.assert_equal("sequence_no_match", true, result.fail());
        });

        // Choices - Partial Match 1
        t.test("choices_partial_match_1", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("option1") | p.literal("option2"); });

            auto ctx    = common_peg_parse_context("opt", true);
            auto result = parser.parse(ctx);
            t.assert_equal("choices_partial_match_1", true, result.need_more_input());
        });

        // Choices - Partial Match 2
        t.test("choices_partial_match_2", [&](testing & t) {
            auto parser =
                build_peg_parser([](common_peg_parser_builder & p) { return p.literal("choice_a") | p.literal("choice_b"); });

            auto ctx    = common_peg_parse_context("choice", true);
            auto result = parser.parse(ctx);
            t.assert_equal("choices_partial_match_2", true, result.need_more_input());
        });

        // Choices - Full Match 1
        t.test("choices_full_match_1", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("first") | p.literal("second"); });

            auto ctx    = common_peg_parse_context("first", false);
            auto result = parser.parse(ctx);
            t.assert_equal("choices_full_match_1", true, result.success());
        });

        // Choices - Full Match 2
        t.test("choices_full_match_2", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("alpha") | p.literal("beta"); });

            auto ctx    = common_peg_parse_context("beta", false);
            auto result = parser.parse(ctx);
            t.assert_equal("choices_full_match_2", true, result.success());
        });

        // Choices - No Match
        t.test("choices_no_match", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("good") | p.literal("better"); });

            auto ctx    = common_peg_parse_context("best", false);
            auto result = parser.parse(ctx);
            t.assert_equal("choices_no_match", true, result.fail());
        });

        // Zero or More - Partial Match 1
        t.test("zero_or_more_partial_match_1", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.zero_or_more(p.literal("ab")); });

            auto ctx    = common_peg_parse_context("a", true);
            auto result = parser.parse(ctx);
            t.assert_equal("zero_or_more_partial_match_1", true, result.need_more_input());
        });

        // Zero or More - Partial Match 2
        t.test("zero_or_more_partial_match_2", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.zero_or_more(p.literal("xy")); });

            auto ctx    = common_peg_parse_context("xyx", true);
            auto result = parser.parse(ctx);
            t.assert_equal("zero_or_more_partial_match_2", true, result.need_more_input());
        });

        // Zero or More - Full Match
        t.test("zero_or_more_full_match", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.zero_or_more(p.literal("test")); });

            auto ctx    = common_peg_parse_context("test", false);
            auto result = parser.parse(ctx);
            t.assert_equal("zero_or_more_full_match", true, result.success());
        });

        // One or More - Partial Match 1
        t.test("one_or_more_partial_match_1", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one_or_more(p.literal("repeat")); });

            auto ctx    = common_peg_parse_context("rep", true);
            auto result = parser.parse(ctx);
            t.assert_equal("one_or_more_partial_match_1", true, result.need_more_input());
        });

        // One or More - Partial Match 2
        t.test("one_or_more_partial_match_2", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one_or_more(p.literal("ab")); });

            auto ctx    = common_peg_parse_context("aba", true);
            auto result = parser.parse(ctx);
            t.assert_equal("one_or_more_partial_match_2", true, result.need_more_input());
        });

        // One or More - Full Match
        t.test("one_or_more_full_match", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one_or_more(p.literal("single")); });

            auto ctx    = common_peg_parse_context("single", false);
            auto result = parser.parse(ctx);
            t.assert_equal("one_or_more_full_match", true, result.success());
        });

        // One or More - No Match
        t.test("one_or_more_no_match", [&](testing & t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one_or_more(p.literal("()")); });

            auto ctx    = common_peg_parse_context("success", false);
            auto result = parser.parse(ctx);
            t.assert_equal("one_or_more_no_match", true, result.fail());
        });
    });


    t.test("recursive rules", [](testing &t) {
        // Test simple number
        t.test("simple_number", [](testing &t) {
            auto value_parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.rule("number", p.chars("0-9"));
                p.rule("list", p.literal("[") + p.ref("value") + p.literal("]"));
                return p.rule("value", p.ref("number") | p.ref("list"));
            });

            common_peg_parse_context ctx("1", false);
            auto           result = value_parser.parse(ctx);

            t.assert_equal("result_is_success", true, result.success());
        });

        // Test simple list
        t.test("simple_list", [](testing &t) {
            auto value_parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.rule("number", p.chars("0-9"));
                p.rule("list", p.literal("[") + p.ref("value") + p.literal("]"));
                return p.rule("value", p.ref("number") | p.ref("list"));
            });

            common_peg_parse_context ctx("[1]", false);
            auto           result = value_parser.parse(ctx);

            t.assert_equal("result_is_success", true, result.success());
        });

        // Test nested list
        t.test("nested_list", [](testing &t) {
            auto value_parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.rule("number", p.chars("0-9"));
                p.rule("list", p.literal("[") + p.ref("value") + p.literal("]"));
                return p.rule("value", p.ref("number") | p.ref("list"));
            });

            common_peg_parse_context ctx("[[2]]", false);
            auto           result = value_parser.parse(ctx);

            t.assert_equal("result_is_success", true, result.success());
        });

        // Test deeply nested list
        t.test("deeply_nested_list", [](testing &t) {
            auto value_parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.rule("number", p.chars("0-9"));
                p.rule("list", p.literal("[") + p.ref("value") + p.literal("]"));
                return p.rule("value", p.ref("number") | p.ref("list"));
            });

            common_peg_parse_context ctx("[[[3]]]", false);
            auto           result = value_parser.parse(ctx);

            t.assert_equal("result_is_success", true, result.success());
        });

        // Test need_more_input match
        t.test("need_more_input_match", [](testing &t) {
            auto value_parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.rule("number", p.chars("0-9"));
                p.rule("list", p.literal("[") + p.ref("value") + p.literal("]"));
                return p.rule("value", p.ref("number") | p.ref("list"));
            });

            common_peg_parse_context ctx("[[", true);
            auto           result = value_parser.parse(ctx);

            t.assert_equal("result_is_need_more_input", true, result.need_more_input());
        });

        // Test no match
        t.test("no_match", [](testing &t) {
            auto value_parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.rule("number", p.chars("0-9"));
                p.rule("list", p.literal("[") + p.ref("value") + p.literal("]"));
                return p.rule("value", p.ref("number") | p.ref("list"));
            });

            common_peg_parse_context ctx("[a]", false);
            auto           result = value_parser.parse(ctx);

            t.assert_equal("result_is_fail", true, result.fail());
        });
    });
}
