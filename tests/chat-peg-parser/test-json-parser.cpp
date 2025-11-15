#include "tests.h"

test_json_parser::test_json_parser() : compound_test("test_json_parser") {
    // Test parsing a simple JSON object
    add_test(
        [](test_harness h) {
            auto json = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.json(); });

            std::string    input = R"({"name": "test", "value": 42, "flag": true})";
            common_chat_parse_context ctx(input);

            auto result = json.parse(ctx);

            h.assert_equals("result_is_success", true, result.success());
            h.assert_equals("result_end", input.size(), result.end);
        },
        "simple JSON object parsing");

    // Test parsing a JSON array with mixed types
    add_test(
        [](test_harness h) {
            auto json = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.json(); });

            std::string    input = R"([1, "hello", true, null, 3.14])";
            common_chat_parse_context ctx(input);

            auto result = json.parse(ctx);

            h.assert_equals("result_is_success", true, result.success());
            h.assert_equals("result_end", input.size(), result.end);
        },
        "JSON array with mixed types");

    // Test parsing nested JSON with objects and arrays
    add_test(
        [](test_harness h) {
            auto json = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.json(); });

            std::string input =
                R"({"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2, "metadata": {"version": "1.0", "tags": ["admin", "user"]}})";
            common_chat_parse_context ctx(input);

            auto result = json.parse(ctx);

            h.assert_equals("result_is_success", true, result.success());
            h.assert_equals("result_end", input.size(), result.end);
        },
        "nested JSON with objects and arrays");

    // Test need_more_input() parsing - incomplete object
    add_test(
        [](test_harness h) {
            auto json = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.json(); });

            std::string    input = R"({"name": "test", "value": )";
            common_chat_parse_context ctx(input, false);

            auto result = json.parse(ctx);

            h.assert_equals("result_is_need_more_input", true, result.need_more_input());
        },
        "need_more_input() parsing - incomplete object");

    // Test need_more_input() parsing - incomplete array
    add_test(
        [](test_harness h) {
            auto json = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.json(); });

            std::string    input = R"([1, 2, 3, )";
            common_chat_parse_context ctx(input, false);

            auto result = json.parse(ctx);

            h.assert_equals("result_is_need_more_input", true, result.need_more_input());
        },
        "need_more_input() parsing - incomplete array");

    // Test need_more_input() parsing - incomplete nested structure
    add_test(
        [](test_harness h) {
            auto json = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.json(); });

            std::string    input = R"({"data": {"nested": )";
            common_chat_parse_context ctx(input, false);

            auto result = json.parse(ctx);

            h.assert_equals("result_is_need_more_input", true, result.need_more_input());
        },
        "need_more_input() parsing - incomplete nested structure");
}
