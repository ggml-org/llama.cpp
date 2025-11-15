#include "tests.h"

test_recursive_references::test_recursive_references() : compound_test("test_recursive_references") {
    // Test simple number
    add_test(
        [](test_harness h) {
            auto value_parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) {
                p.add_rule("number", p.one_or_more(p.one("0-9")));
                p.add_rule("list", p.sequence({ p.literal("["), p.rule("value"), p.literal("]") }));
                return p.add_rule("value", p.rule("number") | p.rule("list"));
            });

            common_chat_parse_context ctx("1", true);
            auto           result = value_parser.parse(ctx);

            h.assert_equals("result_is_success", true, result.success());
        },
        "simple_number");

    // Test simple list
    add_test(
        [](test_harness h) {
            auto value_parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) {
                p.add_rule("number", p.one_or_more(p.one("0-9")));
                p.add_rule("list", p.sequence({ p.literal("["), p.rule("value"), p.literal("]") }));
                return p.add_rule("value", p.rule("number") | p.rule("list"));
            });

            common_chat_parse_context ctx("[1]", true);
            auto           result = value_parser.parse(ctx);

            h.assert_equals("result_is_success", true, result.success());
        },
        "simple_list");

    // Test nested list
    add_test(
        [](test_harness h) {
            auto value_parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) {
                p.add_rule("number", p.one_or_more(p.one("0-9")));
                p.add_rule("list", p.sequence({ p.literal("["), p.rule("value"), p.literal("]") }));
                return p.add_rule("value", p.rule("number") | p.rule("list"));
            });

            common_chat_parse_context ctx("[[2]]", true);
            auto           result = value_parser.parse(ctx);

            h.assert_equals("result_is_success", true, result.success());
        },
        "nested_list");

    // Test deeply nested list
    add_test(
        [](test_harness h) {
            auto value_parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) {
                p.add_rule("number", p.one_or_more(p.one("0-9")));
                p.add_rule("list", p.sequence({ p.literal("["), p.rule("value"), p.literal("]") }));
                return p.add_rule("value", p.rule("number") | p.rule("list"));
            });

            common_chat_parse_context ctx("[[[3]]]", true);
            auto           result = value_parser.parse(ctx);

            h.assert_equals("result_is_success", true, result.success());
        },
        "deeply_nested_list");

    // Test need_more_input match
    add_test(
        [](test_harness h) {
            auto value_parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) {
                p.add_rule("number", p.one_or_more(p.one("0-9")));
                p.add_rule("list", p.sequence({ p.literal("["), p.rule("value"), p.literal("]") }));
                return p.add_rule("value", p.rule("number") | p.rule("list"));
            });

            common_chat_parse_context ctx("[[", false);
            auto           result = value_parser.parse(ctx);

            h.assert_equals("result_is_need_more_input", true, result.need_more_input());
        },
        "need_more_input_match");

    // Test no match
    add_test(
        [](test_harness h) {
            auto value_parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) {
                p.add_rule("number", p.one_or_more(p.one("0-9")));
                p.add_rule("list", p.sequence({ p.literal("["), p.rule("value"), p.literal("]") }));
                return p.add_rule("value", p.rule("number") | p.rule("list"));
            });

            common_chat_parse_context ctx("[a]", true);
            auto           result = value_parser.parse(ctx);

            h.assert_equals("result_is_fail", true, result.fail());
        },
        "no_match");
}
