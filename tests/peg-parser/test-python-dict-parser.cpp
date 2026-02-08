#include "tests.h"

void test_python_dict_parser(testing &t) {
    // Test parsing a simple Python dict object with single quotes
    t.test("simple Python dict object parsing", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string    input = "{'name': 'test', 'value': 42, 'flag': true}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test parsing a Python dict array with mixed types
    t.test("Python dict array with mixed types", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string    input = "[1, 'hello', true, null, 3.14]";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test parsing nested Python dict with objects and arrays
    t.test("nested Python dict with objects and arrays", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string input =
            "{'users': [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}], 'count': 2, 'metadata': {'version': '1.0', 'tags': ['admin', 'user']}}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test parsing Python dict with escaped single quotes
    t.test("Python dict with escaped single quotes", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string    input = "{'message': 'It\\'s working!'}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test parsing Python dict with double quotes inside single quotes
    t.test("Python dict with double quotes inside single quotes", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string    input = "{'quote': 'He said \"Hello\"'}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test the example from the requirements
    t.test("complex Python dict example from requirements", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string    input = "{ 'obj' : { 'something': 1, 'other \"something\"' : 'foo\\'s bar' } }";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test need_more_input() parsing - incomplete object
    t.test("need_more_input() parsing - incomplete object", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string    input = "{'name': 'test', 'value': ";
        common_peg_parse_context ctx(input, true);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_need_more_input", true, result.need_more_input());
    });

    // Test need_more_input() parsing - incomplete single-quoted string
    t.test("need_more_input() parsing - incomplete single-quoted string", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string    input = "{'name': 'test";
        common_peg_parse_context ctx(input, true);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_need_more_input", true, result.need_more_input());
    });

    // Test unicode in Python dict strings
    t.test("unicode in Python dict strings", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string    input = "{'message': 'Hello, 世界!'}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test Python dict with unicode escapes
    t.test("Python dict with unicode escapes", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string    input = "{'unicode': 'Hello\\u0041'}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_end", input.size(), result.end);
    });

    // Test that JSON double-quoted strings fail with Python dict parser
    t.test("JSON double-quoted strings fail with Python dict parser", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) { return p.python_dict(); });

        std::string    input = "{\"name\": \"test\"}";
        common_peg_parse_context ctx(input);

        auto result = parser.parse(ctx);

        t.assert_equal("result_is_fail", true, result.fail());
    });

    // Test Python dict string content parser directly
    t.test("python dict string content parser", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.sequence({ p.literal("'"), p.python_dict_string_content(), p.literal("'"), p.space() });
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

    // Test allow_python_dict_format flag usage
    t.test("allow_python_dict_format flag", [](testing &t) {
        t.test("flag is false by default", [&](testing &t) {
            common_peg_parser_builder builder;
            t.assert_equal("default_value", false, builder.get_allow_python_dict_format());
        });

        t.test("flag can be set to true", [&](testing &t) {
            common_peg_parser_builder builder;
            builder.set_allow_python_dict_format(true);
            t.assert_equal("after_set", true, builder.get_allow_python_dict_format());
        });

        t.test("flag can be set back to false", [&](testing &t) {
            common_peg_parser_builder builder;
            builder.set_allow_python_dict_format(true);
            builder.set_allow_python_dict_format(false);
            t.assert_equal("after_reset", false, builder.get_allow_python_dict_format());
        });
    });

    // Test that the flag actually affects json() parser behavior
    t.test("json() parser with allow_python_dict_format flag", [](testing &t) {
        t.test("json() rejects single quotes when flag is false", [&](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.set_allow_python_dict_format(false);
                return p.json();
            });

            std::string input = "{'name': 'test'}";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("fail", result.fail());
        });

        t.test("json() accepts single quotes when flag is true", [&](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.set_allow_python_dict_format(true);
                return p.json();
            });

            std::string input = "{'name': 'test'}";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("json() still accepts double quotes when flag is true", [&](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.set_allow_python_dict_format(true);
                return p.json();
            });

            std::string input = "{\"name\": \"test\"}";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("json() accepts mixed quote styles when flag is true", [&](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.set_allow_python_dict_format(true);
                return p.json();
            });

            std::string input = "{\"name\": 'test', 'value': \"hello\"}";
            common_peg_parse_context ctx(input);

            auto result = parser.parse(ctx);
            t.assert_true("success", result.success());
            t.assert_equal("end", input.size(), result.end);
        });

        t.test("complex nested structure with flag true", [&](testing &t) {
            auto parser = build_peg_parser([](common_peg_parser_builder & p) {
                p.set_allow_python_dict_format(true);
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
