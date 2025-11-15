#include "tests.h"

test_optional::test_optional() : compound_test("test_optional") {
    // Full match with optional part present
    add_test(
        [](test_harness h) {
            auto parser =
                build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.literal("hello") + p.optional(p.literal(" world")); });

            auto ctx    = common_chat_parse_context("hello world");
            auto result = parser.parse(ctx);
            h.assert_equals("optional_present", true, result.success());
            int end_pos = result.end;
            h.assert_equals("optional_present_end", 11, end_pos);
        },
        "optional_present");

    // Full match with optional part absent
    add_test(
        [](test_harness h) {
            auto parser =
                build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.literal("hello") + p.optional(p.literal(" world")); });

            auto ctx    = common_chat_parse_context("hello", true);
            auto result = parser.parse(ctx);
            h.assert_equals("optional_absent", true, result.success());
            int end_pos = result.end;
            h.assert_equals("optional_absent_end", 5, end_pos);
        },
        "optional_absent");

    // Partial match - waiting for more input to determine if optional matches
    add_test(
        [](test_harness h) {
            auto parser =
                build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.literal("hello") + p.optional(p.literal(" world")); });

            auto ctx    = common_chat_parse_context("hello ", false);
            auto result = parser.parse(ctx);
            h.assert_equals("partial_match_need_more", true, result.need_more_input());
        },
        "partial_match_need_more");
}
