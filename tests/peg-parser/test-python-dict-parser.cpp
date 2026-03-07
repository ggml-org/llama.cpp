#include "tests.h"

void test_python_dict_parser(testing &t) {
    // Test parsing a simple Python dict object with single quotes
    t.test("simple Python dict object parsing", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "{'name': 'test', 'value': 42, 'flag': True}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test parsing a Python array with mixed types
    t.test("Python array with mixed types", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "[1, 'hello', True, None, 3.14]";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test parsing nested Python dict with objects and arrays
    t.test("nested Python dict with objects and arrays", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string input =
            "{'users': [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}], 'count': 2, 'metadata': {'version': '1.0', 'tags': ['admin', 'user']}}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test parsing Python dict with escaped single quotes
    t.test("Python dict with escaped single quotes", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "{'message': 'It\\'s working!'}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test parsing Python dict with double quotes inside single quotes
    t.test("Python dict with double quotes inside single quotes", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "{'quote': 'He said \"Hello\"'}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test the example from the requirements
    t.test("complex Python dict example from requirements", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "{ 'obj' : { 'something': 1, 'other \"something\"' : 'foo\\'s bar' } }";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test need_more_input() parsing - incomplete object
    t.test("need_more_input() parsing - incomplete object", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "{'name': 'test', 'value': ";
        common_peg_parse_context ctx(input, true);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_need_more_input", true, result.need_more_input());
    });

    // Test need_more_input() parsing - incomplete single-quoted string
    t.test("need_more_input() parsing - incomplete single-quoted string", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "{'name': 'test";
        common_peg_parse_context ctx(input, true);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_need_more_input", true, result.need_more_input());
    });

    // Test unicode in Python dict strings
    t.test("unicode in Python dict strings", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "{'message': 'Hello, 世界!'}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test Python dict with unicode escapes
    t.test("Python dict with unicode escapes", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "{'unicode': 'Hello\\u0041'}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test that Python parser accepts double-quoted strings too
    t.test("Python parser accepts double-quoted strings", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "{\"name\": \"test\"}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test Python parser with mixed quote styles
    t.test("Python parser with mixed quote styles", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        std::string    input = "{\"name\": 'test', 'value': \"hello\"}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test Python True/False/None
    t.test("Python True/False/None", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_value(); });

        t.test("True", [&](testing &t) {
            std::string input = "True";
            common_peg_parse_context ctx(input);
            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("False", [&](testing &t) {
            std::string input = "False";
            common_peg_parse_context ctx(input);
            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("None", [&](testing &t) {
            std::string input = "None";
            common_peg_parse_context ctx(input);
            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("rejects JSON-style true/false/null", [&](testing &t) {
            for (const auto & kw : {"true", "false", "null"}) {
                std::string input = kw;
                common_peg_parse_context ctx(input);
                auto result = parser.parse(ctx);
                t.assert_true(std::string("rejects ") + kw, result.fail());
            }
        });
    });

    // Test single-quoted string content parser directly
    t.test("single-quoted string content parser", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.sequence({ p.literal("'"), p.single_quoted_string_content(), p.literal("'"), p.space() });
        });

        t.test("simple string", [&](testing &t) {
            std::string input = "'hello'";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("string with escaped single quote", [&](testing &t) {
            std::string input = "'it\\'s'";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("string with double quotes", [&](testing &t) {
            std::string input = "'say \"hello\"'";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("incomplete string", [&](testing &t) {
            std::string input = "'hello";
            common_peg_parse_context ctx(input, true);

            auto result = parser.parse(ctx);
            t.assert_true("need_more_input", result.need_more_input());
        });
    });

    // Test json() with pre-registered flexible json-string rule (python dict support)
    t.test("json() parser with flexible json-string rule", [](testing &t) {
        t.test("json() rejects single quotes by default", [&](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                return p.json();
            });

            std::string input = "{'name': 'test'}";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("fail", result.fail());
        });

        t.test("json() accepts single quotes with pre-registered flexible json-string rule", [&](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                // Pre-register json-string rule with both quote styles
                p.rule("json-string", [&]() {
                    return p.choice({ p.double_quoted_string(), p.single_quoted_string() });
                });
                return p.json();
            });

            std::string input = "{'name': 'test'}";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("json() still accepts double quotes with flexible json-string rule", [&](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.rule("json-string", [&]() {
                    return p.choice({ p.double_quoted_string(), p.single_quoted_string() });
                });
                return p.json();
            });

            std::string input = "{\"name\": \"test\"}";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("json() accepts mixed quote styles with flexible json-string rule", [&](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.rule("json-string", [&]() {
                    return p.choice({ p.double_quoted_string(), p.single_quoted_string() });
                });
                return p.json();
            });

            std::string input = "{\"name\": 'test', 'value': \"hello\"}";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("complex nested structure with flexible json-string rule", [&](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.rule("json-string", [&]() {
                    return p.choice({ p.double_quoted_string(), p.single_quoted_string() });
                });
                return p.json();
            });

            std::string input = "{ 'obj' : { 'something': 1, 'other \"something\"' : 'foo\\'s bar' } }";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });
    });
}
