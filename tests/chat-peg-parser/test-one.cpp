#include "tests.h"

void test_one(testing &t) {
    // Test common escape sequences - newline
    t.test("escape_sequence_newline", [](testing &t) {
        auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

        common_chat_parse_context ctx;
        common_chat_parse_result  result;

        ctx    = common_chat_parse_context("\n");
        result = common_chat_combinator_parser.parse(ctx);
        t.assert_equal("escape_sequence_newline", true, result.success());
    });

    // Test common escape sequences - tab
    t.test("escape_sequence_tab", [](testing &t) {
        auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

        common_chat_parse_context ctx;
        common_chat_parse_result  result;

        ctx    = common_chat_parse_context("\t");
        result = common_chat_combinator_parser.parse(ctx);
        t.assert_equal("escape_sequence_tab", true, result.success());
    });

    // Test common escape sequences - backslash
    t.test("escape_sequence_backslash", [](testing &t) {
        auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

        common_chat_parse_context ctx;
        common_chat_parse_result  result;

        ctx    = common_chat_parse_context("\\");
        result = common_chat_combinator_parser.parse(ctx);
        t.assert_equal("escape_sequence_backslash", true, result.success());
    });

    // Test common escape sequences - space (should ())
    t.test("escape_sequence_space_fail", [](testing &t) {
        auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

        common_chat_parse_context ctx;
        common_chat_parse_result  result;

        ctx    = common_chat_parse_context(" ");
        result = common_chat_combinator_parser.parse(ctx);
        t.assert_equal("escape_sequence_space_fail", true, result.fail());
    });

    // Test escaped dash - 'a' should succeed
    t.test("escaped_dash_a", [](testing &t) {
        auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[a\\-z]"); });

        common_chat_parse_context ctx;
        common_chat_parse_result  result;

        ctx    = common_chat_parse_context("a");
        result = common_chat_combinator_parser.parse(ctx);
        t.assert_equal("escaped_dash_a", true, result.success());
    });

    // Test escaped dash - '-' should succeed (literal dash)
    t.test("escaped_dash_literal", [](testing &t) {
        auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[a\\-z]"); });

        common_chat_parse_context ctx;
        common_chat_parse_result  result;

        ctx    = common_chat_parse_context("-");
        result = common_chat_combinator_parser.parse(ctx);
        t.assert_equal("escaped_dash_literal", true, result.success());
    });

    // Test escaped dash - 'z' should succeed
    t.test("escaped_dash_z", [](testing &t) {
        auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[a\\-z]"); });

        common_chat_parse_context ctx;
        common_chat_parse_result  result;

        ctx    = common_chat_parse_context("z");
        result = common_chat_combinator_parser.parse(ctx);
        t.assert_equal("escaped_dash_z", true, result.success());
    });

    // Test escaped dash - 'b' should NOT match (since \- is literal dash, not range)
    t.test("escaped_dash_b_fail", [](testing &t) {
        auto common_chat_combinator_parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[a\\-z]"); });

        common_chat_parse_context ctx;
        common_chat_parse_result  result;

        ctx    = common_chat_parse_context("b");
        result = common_chat_combinator_parser.parse(ctx);
        t.assert_equal("escaped_dash_b_fail", true, result.fail());
    });
}
