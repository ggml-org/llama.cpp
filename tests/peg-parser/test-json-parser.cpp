#include "tests.h"

void test_json_parser(testing &t) {
    // Test parsing a simple JSON object
    t.test("simple JSON object parsing", [](testing &t) {
        auto json = build_peg_parser([](common_peg_parser_builder & p) { return p.json(); });

        std::string    input = R"({"name": "test", "value": 42, "flag": true})";
        common_peg_parse_context ctx(input);

        auto result = json.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test parsing a JSON array with mixed types
    t.test("JSON array with mixed types", [](testing &t) {
        auto json = build_peg_parser([](common_peg_parser_builder & p) { return p.json(); });

        std::string    input = R"([1, "hello", true, null, 3.14])";
        common_peg_parse_context ctx(input);

        auto result = json.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test parsing nested JSON with objects and arrays
    t.test("nested JSON with objects and arrays", [](testing &t) {
        auto json = build_peg_parser([](common_peg_parser_builder & p) { return p.json(); });

        std::string input =
            R"({"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2, "metadata": {"version": "1.0", "tags": ["admin", "user"]}})";
        common_peg_parse_context ctx(input);

        auto result = json.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test need_more_input() parsing - incomplete object
    t.test("need_more_input() parsing - incomplete object", [](testing &t) {
        auto json = build_peg_parser([](common_peg_parser_builder & p) { return p.json(); });

        std::string    input = R"({"name": "test", "value": )";
        common_peg_parse_context ctx(input, true);

        auto result = json.parse(ctx);

        t.assert_equal("result_is_need_more_input", true, result.need_more_input());
    });

    // Test need_more_input() parsing - incomplete array
    t.test("need_more_input() parsing - incomplete array", [](testing &t) {
        auto json = build_peg_parser([](common_peg_parser_builder & p) { return p.json(); });

        std::string    input = R"([1, 2, 3, )";
        common_peg_parse_context ctx(input, true);

        auto result = json.parse(ctx);

        t.assert_equal("result_is_need_more_input", true, result.need_more_input());
    });

    // Test need_more_input() parsing - incomplete nested structure
    t.test("need_more_input() parsing - incomplete nested structure", [](testing &t) {
        auto json = build_peg_parser([](common_peg_parser_builder & p) { return p.json(); });

        std::string    input = R"({"data": {"nested": )";
        common_peg_parse_context ctx(input, true);

        auto result = json.parse(ctx);

        t.assert_equal("result_is_need_more_input", true, result.need_more_input());
    });

    t.test("object member", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.json_member("name", "\"" + p.chars("[a-z]") + "\"");
        });

        t.test("success", [&](testing &t) {
            std::string input = R"("name": "bob")";
            common_peg_parse_context ctx(input, false);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
        });

        t.test("partial", [&](testing &t) {
            std::string input = R"("name": "bo)";
            common_peg_parse_context ctx(input, true);

            auto result = parser.parse(ctx);
            t.assert_true("need more input", result.need_more_input());
        });

        t.test("failed", [&](testing &t) {
            std::string input = R"([])";
            common_peg_parse_context ctx(input, false);

            auto result = parser.parse(ctx);
            t.assert_true("fail", result.fail());
        });
    });
}
