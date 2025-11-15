#include "tests.h"

test_one::test_one() : compound_test("test_one") {
    // Test common escape sequences - newline
    add_test(
        [](test_harness h) {
            auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

            common_chat_parse_context ctx;
            common_chat_parse_result  result;

            ctx    = common_chat_parse_context("\n");
            result = common_chat_combinator_parser.parse(ctx);
            h.assert_equals("escape_sequence_newline", true, result.success());
        },
        "escape_sequence_newline");

    // Test common escape sequences - tab
    add_test(
        [](test_harness h) {
            auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

            common_chat_parse_context ctx;
            common_chat_parse_result  result;

            ctx    = common_chat_parse_context("\t");
            result = common_chat_combinator_parser.parse(ctx);
            h.assert_equals("escape_sequence_tab", true, result.success());
        },
        "escape_sequence_tab");

    // Test common escape sequences - backslash
    add_test(
        [](test_harness h) {
            auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

            common_chat_parse_context ctx;
            common_chat_parse_result  result;

            ctx    = common_chat_parse_context("\\");
            result = common_chat_combinator_parser.parse(ctx);
            h.assert_equals("escape_sequence_backslash", true, result.success());
        },
        "escape_sequence_backslash");

    // Test common escape sequences - space (should ())
    add_test(
        [](test_harness h) {
            auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

            common_chat_parse_context ctx;
            common_chat_parse_result  result;

            ctx    = common_chat_parse_context(" ");
            result = common_chat_combinator_parser.parse(ctx);
            h.assert_equals("escape_sequence_space_fail", true, result.fail());
        },
        "escape_sequence_space_fail");

    // Test escaped dash - 'a' should succeed
    add_test(
        [](test_harness h) {
            auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[a\\-z]"); });

            common_chat_parse_context ctx;
            common_chat_parse_result  result;

            ctx    = common_chat_parse_context("a");
            result = common_chat_combinator_parser.parse(ctx);
            h.assert_equals("escaped_dash_a", true, result.success());
        },
        "escaped_dash_a");

    // Test escaped dash - '-' should succeed (literal dash)
    add_test(
        [](test_harness h) {
            auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[a\\-z]"); });

            common_chat_parse_context ctx;
            common_chat_parse_result  result;

            ctx    = common_chat_parse_context("-");
            result = common_chat_combinator_parser.parse(ctx);
            h.assert_equals("escaped_dash_literal", true, result.success());
        },
        "escaped_dash_literal");

    // Test escaped dash - 'z' should succeed
    add_test(
        [](test_harness h) {
            auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[a\\-z]"); });

            common_chat_parse_context ctx;
            common_chat_parse_result  result;

            ctx    = common_chat_parse_context("z");
            result = common_chat_combinator_parser.parse(ctx);
            h.assert_equals("escaped_dash_z", true, result.success());
        },
        "escaped_dash_z");

    // Test escaped dash - 'b' should NOT match (since \- is literal dash, not range)
    add_test(
        [](test_harness h) {
            auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[a\\-z]"); });

            common_chat_parse_context ctx;
            common_chat_parse_result  result;

            ctx    = common_chat_parse_context("b");
            result = common_chat_combinator_parser.parse(ctx);
            h.assert_equals("escaped_dash_b_fail", true, result.fail());
        },
        "escaped_dash_b_fail");
}
