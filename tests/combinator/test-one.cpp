#include "tests.h"

test_one::test_one() : compound_test("test_one") {
    // Test common escape sequences - newline
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

            parser_context ctx;
            parser_result  result;

            ctx    = parser_context("\n");
            result = parser.parse(ctx);
            h.assert_equals("escape_sequence_newline", true, result.is_success());
        },
        "escape_sequence_newline");

    // Test common escape sequences - tab
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

            parser_context ctx;
            parser_result  result;

            ctx    = parser_context("\t");
            result = parser.parse(ctx);
            h.assert_equals("escape_sequence_tab", true, result.is_success());
        },
        "escape_sequence_tab");

    // Test common escape sequences - backslash
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

            parser_context ctx;
            parser_result  result;

            ctx    = parser_context("\\");
            result = parser.parse(ctx);
            h.assert_equals("escape_sequence_backslash", true, result.is_success());
        },
        "escape_sequence_backslash");

    // Test common escape sequences - space (should fail)
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) { return p.one("[\\n\\t\\\\]"); });

            parser_context ctx;
            parser_result  result;

            ctx    = parser_context(" ");
            result = parser.parse(ctx);
            h.assert_equals("escape_sequence_space_fail", true, result.is_fail());
        },
        "escape_sequence_space_fail");

    // Test escaped dash - 'a' should succeed
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) { return p.one("[a\\-z]"); });

            parser_context ctx;
            parser_result  result;

            ctx    = parser_context("a");
            result = parser.parse(ctx);
            h.assert_equals("escaped_dash_a", true, result.is_success());
        },
        "escaped_dash_a");

    // Test escaped dash - '-' should succeed (literal dash)
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) { return p.one("[a\\-z]"); });

            parser_context ctx;
            parser_result  result;

            ctx    = parser_context("-");
            result = parser.parse(ctx);
            h.assert_equals("escaped_dash_literal", true, result.is_success());
        },
        "escaped_dash_literal");

    // Test escaped dash - 'z' should succeed
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) { return p.one("[a\\-z]"); });

            parser_context ctx;
            parser_result  result;

            ctx    = parser_context("z");
            result = parser.parse(ctx);
            h.assert_equals("escaped_dash_z", true, result.is_success());
        },
        "escaped_dash_z");

    // Test escaped dash - 'b' should NOT match (since \- is literal dash, not range)
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) { return p.one("[a\\-z]"); });

            parser_context ctx;
            parser_result  result;

            ctx    = parser_context("b");
            result = parser.parse(ctx);
            h.assert_equals("escaped_dash_b_fail", true, result.is_fail());
        },
        "escaped_dash_b_fail");
}
