#include "tests.h"
#include "test_harness.h"

void test_partial_parsing(testing &t) {
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
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one("a-z"); });

        common_peg_parse_context ctx;
        common_peg_parse_result  result;

        ctx    = common_peg_parse_context("a");
        result = parser.parse(ctx);
        t.assert_equal("char_class_lowercase_success", true, result.success());
    });

    // Char Classes - Uppercase Fail
    t.test("char_class_uppercase_fail", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one("a-z"); });

        common_peg_parse_context ctx;
        common_peg_parse_result  result;

        ctx    = common_peg_parse_context("A");
        result = parser.parse(ctx);
        t.assert_equal("char_class_uppercase_fail", true, result.fail());
    });

    // Char Classes with Dash - Lowercase Success
    t.test("char_class_with_dash_lowercase", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one("a-z-"); });

        common_peg_parse_context ctx;
        common_peg_parse_result  result;

        ctx    = common_peg_parse_context("f");
        result = parser.parse(ctx);
        t.assert_equal("char_class_with_dash_lowercase", true, result.success());
    });

    // Char Classes with Dash - Literal Dash Success
    t.test("char_class_with_dash_literal_dash", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one("a-z-"); });

        common_peg_parse_context ctx;
        common_peg_parse_result  result;

        ctx    = common_peg_parse_context("-");
        result = parser.parse(ctx);
        t.assert_equal("char_class_with_dash_literal_dash", true, result.success());
    });

    // Char Classes with Dash - Uppercase Fail
    t.test("char_class_with_dash_uppercase_fail", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one("a-z-"); });

        common_peg_parse_context ctx;
        common_peg_parse_result  result;

        ctx    = common_peg_parse_context("A");
        result = parser.parse(ctx);
        t.assert_equal("char_class_with_dash_uppercase_fail", true, result.fail());
    });

    // Sequences - Partial Match 1
    t.test("sequence_partial_match_1", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("<think>") + p.literal("</think>"); });

        auto ctx    = common_peg_parse_context("<thi", false);
        auto result = parser.parse(ctx);
        t.assert_equal("sequence_partial_match_1", true, result.need_more_input());
    });

    // Sequences - Partial Match 2
    t.test("sequence_partial_match_2", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("begin") + p.literal("end"); });

        auto ctx    = common_peg_parse_context("begin", false);
        auto result = parser.parse(ctx);
        t.assert_equal("sequence_partial_match_2", true, result.need_more_input());
    });

    // Sequences - Partial Match 3
    t.test("sequence_partial_match_3", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("<think>") + p.literal("</think>"); });

        auto ctx    = common_peg_parse_context("<think></", false);
        auto result = parser.parse(ctx);
        t.assert_equal("sequence_partial_match_3", true, result.need_more_input());
    });

    // Sequences - Full Match
    t.test("sequence_full_match", [&](testing & t) {
        auto common_chat_combinator_parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("hello") + p.literal("world"); });

        auto ctx    = common_peg_parse_context("helloworld", true);
        auto result = common_chat_combinator_parser.parse(ctx);
        t.assert_equal("sequence_full_match", true, result.success());
    });

    // Sequences - No Match
    t.test("sequence_no_match", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("<think>") + p.literal("</think>"); });

        auto ctx    = common_peg_parse_context("<think>I am common_chat_combinator_parser", false);
        auto result = parser.parse(ctx);
        t.assert_equal("sequence_no_match", true, result.fail());
    });

    // Choices - Partial Match 1
    t.test("choices_partial_match_1", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("option1") | p.literal("option2"); });

        auto ctx    = common_peg_parse_context("opt", false);
        auto result = parser.parse(ctx);
        t.assert_equal("choices_partial_match_1", true, result.need_more_input());
    });

    // Choices - Partial Match 2
    t.test("choices_partial_match_2", [&](testing & t) {
        auto parser =
            build_peg_parser([](common_peg_parser_builder & p) { return p.literal("choice_a") | p.literal("choice_b"); });

        auto ctx    = common_peg_parse_context("choice", false);
        auto result = parser.parse(ctx);
        t.assert_equal("choices_partial_match_2", true, result.need_more_input());
    });

    // Choices - Full Match 1
    t.test("choices_full_match_1", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("first") | p.literal("second"); });

        auto ctx    = common_peg_parse_context("first", true);
        auto result = parser.parse(ctx);
        t.assert_equal("choices_full_match_1", true, result.success());
    });

    // Choices - Full Match 2
    t.test("choices_full_match_2", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("alpha") | p.literal("beta"); });

        auto ctx    = common_peg_parse_context("beta", true);
        auto result = parser.parse(ctx);
        t.assert_equal("choices_full_match_2", true, result.success());
    });

    // Choices - No Match
    t.test("choices_no_match", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.literal("good") | p.literal("better"); });

        auto ctx    = common_peg_parse_context("best", true);
        auto result = parser.parse(ctx);
        t.assert_equal("choices_no_match", true, result.fail());
    });

    // Zero or More - Partial Match 1
    t.test("zero_or_more_partial_match_1", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.zero_or_more(p.literal("ab")); });

        auto ctx    = common_peg_parse_context("a", false);
        auto result = parser.parse(ctx);
        t.assert_equal("zero_or_more_partial_match_1", true, result.need_more_input());
    });

    // Zero or More - Partial Match 2
    t.test("zero_or_more_partial_match_2", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.zero_or_more(p.literal("xy")); });

        auto ctx    = common_peg_parse_context("xyx", false);
        auto result = parser.parse(ctx);
        t.assert_equal("zero_or_more_partial_match_2", true, result.need_more_input());
    });

    // Zero or More - Full Match
    t.test("zero_or_more_full_match", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.zero_or_more(p.literal("test")); });

        auto ctx    = common_peg_parse_context("test", true);
        auto result = parser.parse(ctx);
        t.assert_equal("zero_or_more_full_match", true, result.success());
    });

    // One or More - Partial Match 1
    t.test("one_or_more_partial_match_1", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one_or_more(p.literal("repeat")); });

        auto ctx    = common_peg_parse_context("rep", false);
        auto result = parser.parse(ctx);
        t.assert_equal("one_or_more_partial_match_1", true, result.need_more_input());
    });

    // One or More - Partial Match 2
    t.test("one_or_more_partial_match_2", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one_or_more(p.literal("ab")); });

        auto ctx    = common_peg_parse_context("aba", false);
        auto result = parser.parse(ctx);
        t.assert_equal("one_or_more_partial_match_2", true, result.need_more_input());
    });

    // One or More - Full Match
    t.test("one_or_more_full_match", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one_or_more(p.literal("single")); });

        auto ctx    = common_peg_parse_context("single", true);
        auto result = parser.parse(ctx);
        t.assert_equal("one_or_more_full_match", true, result.success());
    });

    // One or More - No Match
    t.test("one_or_more_no_match", [&](testing & t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.one_or_more(p.literal("()")); });

        auto ctx    = common_peg_parse_context("success", true);
        auto result = parser.parse(ctx);
        t.assert_equal("one_or_more_no_match", true, result.fail());
    });
}
