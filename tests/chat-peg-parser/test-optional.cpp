#include "tests.h"

void test_optional(testing &t) {
    // Full match with optional part present
    t.test("optional_present", [](testing &t) {
        auto parser =
            build_peg_parser([](common_chat_peg_parser_builder & p) { return p.literal("hello") + p.optional(p.literal(" world")); });

        auto ctx    = common_chat_parse_context("hello world");
        auto result = parser.parse(ctx);
        t.assert_equal("optional_present", true, result.success());
        int end_pos = result.end;
        t.assert_equal("optional_present_end", 11, end_pos);
    });

    // Full match with optional part absent
    t.test("optional_absent", [](testing &t) {
        auto parser =
            build_peg_parser([](common_chat_peg_parser_builder & p) { return p.literal("hello") + p.optional(p.literal(" world")); });

        auto ctx    = common_chat_parse_context("hello", true);
        auto result = parser.parse(ctx);
        t.assert_equal("optional_absent", true, result.success());
        int end_pos = result.end;
        t.assert_equal("optional_absent_end", 5, end_pos);
    });

    // Partial match - waiting for more input to determine if optional matches
    t.test("partial_match_need_more", [](testing &t) {
        auto parser =
            build_peg_parser([](common_chat_peg_parser_builder & p) { return p.literal("hello") + p.optional(p.literal(" world")); });

        auto ctx    = common_chat_parse_context("hello ", false);
        auto result = parser.parse(ctx);
        t.assert_equal("partial_match_need_more", true, result.need_more_input());
    });
}
