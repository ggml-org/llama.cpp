#include "tests.h"

void test_recursive_references(testing &t) {
    // Test simple number
    t.test("simple_number", [](testing &t) {
        auto value_parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
            p.rule("number", p.one_or_more(p.one("0-9")));
            p.rule("list", p.sequence({ p.literal("["), p.ref("value"), p.literal("]") }));
            return p.rule("value", p.ref("number") | p.ref("list"));
        });

        common_chat_parse_context ctx("1", true);
        auto           result = value_parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
    });

    // Test simple list
    t.test("simple_list", [](testing &t) {
        auto value_parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
            p.rule("number", p.one_or_more(p.one("0-9")));
            p.rule("list", p.sequence({ p.literal("["), p.ref("value"), p.literal("]") }));
            return p.rule("value", p.ref("number") | p.ref("list"));
        });

        common_chat_parse_context ctx("[1]", true);
        auto           result = value_parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
    });

    // Test nested list
    t.test("nested_list", [](testing &t) {
        auto value_parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
            p.rule("number", p.one_or_more(p.one("0-9")));
            p.rule("list", p.sequence({ p.literal("["), p.ref("value"), p.literal("]") }));
            return p.rule("value", p.ref("number") | p.ref("list"));
        });

        common_chat_parse_context ctx("[[2]]", true);
        auto           result = value_parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
    });

    // Test deeply nested list
    t.test("deeply_nested_list", [](testing &t) {
        auto value_parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
            p.rule("number", p.one_or_more(p.one("0-9")));
            p.rule("list", p.sequence({ p.literal("["), p.ref("value"), p.literal("]") }));
            return p.rule("value", p.ref("number") | p.ref("list"));
        });

        common_chat_parse_context ctx("[[[3]]]", true);
        auto           result = value_parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
    });

    // Test need_more_input match
    t.test("need_more_input_match", [](testing &t) {
        auto value_parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
            p.rule("number", p.one_or_more(p.one("0-9")));
            p.rule("list", p.sequence({ p.literal("["), p.ref("value"), p.literal("]") }));
            return p.rule("value", p.ref("number") | p.ref("list"));
        });

        common_chat_parse_context ctx("[[", false);
        auto           result = value_parser.parse(ctx);

        t.assert_equal("result_is_need_more_input", true, result.need_more_input());
    });

    // Test no match
    t.test("no_match", [](testing &t) {
        auto value_parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
            p.rule("number", p.one_or_more(p.one("0-9")));
            p.rule("list", p.sequence({ p.literal("["), p.ref("value"), p.literal("]") }));
            return p.rule("value", p.ref("number") | p.ref("list"));
        });

        common_chat_parse_context ctx("[a]", true);
        auto           result = value_parser.parse(ctx);

        t.assert_equal("result_is_fail", true, result.fail());
    });
}
