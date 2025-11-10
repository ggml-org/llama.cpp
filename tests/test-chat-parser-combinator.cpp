#include <iostream>
#include <string>

#include "chat-parser-combinator.h"

template <class T>
static void assert_equals(const std::string_view label, const T & expected, const T & actual) {
    if (expected != actual) {
        std::cerr << label << "\n";
        std::cerr << "Expected: " << expected << "\n";
        std::cerr << "Actual: " << actual << "\n";
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

template <class T>
static void assert_equals(const T & expected, const T & actual) {
    assert_equals("", expected, actual);
}

static void assert_equals(const char * expected, const std::string & actual) {
    assert_equals<std::string>(expected, actual);
}

static void test_partial_parsing() {
    {
        // Test literal
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("hello");
        });

        parser_context ctx;
        parser_result result;

        ctx = parser_context{"hello", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());
    }
    {
        // Test char class
        auto parser = build_parser([](parser_builder& p) {
            return p.char_class("a-z");
        });

        parser_context ctx;
        parser_result result;

        ctx = parser_context{"a", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context{"A", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());

        parser = build_parser([](parser_builder& p) {
            return p.char_class("a-z-");
        });

        ctx = parser_context{"f", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context{"-", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context{"A", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());
    }
    {
        // Test sequences and literals
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("<think>") + p.literal("</think>");
        });

        // Partial matches
        auto ctx = parser_context{"<thi", parse_cache(), false};
        auto result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        ctx = parser_context{"<think>", parse_cache(), false};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context{"<think></", parse_cache(), false};
        result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        // Full match
        ctx = parser_context{"<think></think>", parse_cache(), true};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        // No match, since it does not adhere to the grammar
        ctx = parser_context{"<think>I am parser", parse_cache(), false};
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());
    }
    {
        // Test choices
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("<think>") | p.literal("<reasoning>");
        });

        // Partial matches
        auto ctx = parser_context{"<thi", parse_cache(), false};
        auto result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        ctx = parser_context{"<reas", parse_cache(), false};
        result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        // Full match
        ctx = parser_context{"<think>", parse_cache(), true};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context{"<reasoning>", parse_cache(), true};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        // No match
        ctx = parser_context{"<thought>", parse_cache(), true};
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());
    }
    {
        // Test zero_or_more
        auto parser = build_parser([](parser_builder& p) {
            return p.zero_or_more(p.literal("ab"));
        });

        // Partial matches
        auto ctx = parser_context{"a", parse_cache(), false};
        auto result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        ctx = parser_context{"aba", parse_cache(), false};
        result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        // Full match
        ctx = parser_context{"ab", parse_cache(), true};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());
    }
    {
        // Test one_or_more
        auto parser = build_parser([](parser_builder& p) {
            return p.one_or_more(p.literal("ab"));
        });

        // Partial matches
        auto ctx = parser_context{"a", parse_cache(), false};
        auto result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        ctx = parser_context{"aba", parse_cache(), false};
        result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        // Full match
        ctx = parser_context{"ab", parse_cache(), true};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        // No match
        ctx = parser_context{"cd", parse_cache(), true};
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());
    }
}

static void test_capture_groups() {
    {
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("<think>") +
                   p.group("reasoning_content",
                       p.zero_or_more(~p.literal("</think>") + p.any())
                   ) +
                   p.literal("</think>");
        });

        std::string input = "<think>I have a thought</think>";
        auto ctx = parser_context{input, parse_cache()};
        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());

        auto it = result.groups.find("reasoning_content");
        assert_equals(true, it != result.groups.end());
        assert_equals("I have a thought", std::string(it->second.view(input)));
    }
    {
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("<think>") +
                   p.group("reasoning_content",
                       p.zero_or_more(~p.literal("</think>") + p.any())
                   ) +
                   p.literal("</think>");
        });

        std::string input = "<think>I have a ";
        auto ctx = parser_context{input, parse_cache(), false};
        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());

        auto it = result.groups.find("reasoning_content");
        assert_equals(true, it != result.groups.end());
        assert_equals("I have a ", std::string(it->second.view(input)));
    }
    {
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("<think>") +
                   p.group("reasoning_content",
                       p.zero_or_more(~p.literal("</think>") + p.any())
                   ) +
                   p.literal("</think>") +
                   p.group("content", p.zero_or_more(p.any()));
        });

        std::string input = "<think>The user said hello.</think>Hello!";
        auto ctx = parser_context{input, parse_cache(), true};
        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());

        auto it = result.groups.find("reasoning_content");
        assert_equals(true, it != result.groups.end());
        assert_equals("The user said hello.", std::string(it->second.view(input)));

        it = result.groups.find("content");
        assert_equals(true, it != result.groups.end());
        assert_equals("Hello!", std::string(it->second.view(input)));
    }
}

static void test_char_class() {
    {
        // Test common escape sequences
        auto parser = build_parser([](parser_builder& p) {
            return p.char_class("[\\n\\t\\\\]");
        });

        parser_context ctx;
        parser_result result;

        ctx = parser_context{"\n", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context{"\t", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context{"\\", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context{" ", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());
    }
    {
        // Test escaped dash (literal dash, not a range)
        auto parser = build_parser([](parser_builder& p) {
            return p.char_class("[a\\-z]");
        });

        parser_context ctx;
        parser_result result;

        ctx = parser_context{"a", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context{"-", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context{"z", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        // Should NOT match 'b' since \- is a literal dash, not a range
        ctx = parser_context{"b", parse_cache()};
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());
    }
}

static void test_recursive_references() {
    auto value_parser = build_parser([](parser_builder& p) {
        p.add_rule("number", p.one_or_more(p.char_class("0-9")));
        p.add_rule("list", p.sequence({
            p.literal("["),
            p.rule("value"),
            p.literal("]")
        }));
        return p.add_rule("value", p.rule("number") | p.rule("list"));
    });

    parser_context ctx;
    parser_result result;

    // Test simple number
    ctx = parser_context{"1", parse_cache(), true};
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_success());

    // Test simple list
    ctx = parser_context{"[1]", parse_cache(), true};
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_success());

    // Test nested list
    ctx = parser_context{"[[2]]", parse_cache(), true};
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_success());

    // Test deeply nested list
    ctx = parser_context{"[[[3]]]", parse_cache(), true};
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_success());

    // Test partial match
    ctx = parser_context{"[[", parse_cache(), false};
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_success());

    // Test no match
    ctx = parser_context{"[a]", parse_cache(), true};
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_fail());
}

static void test_optional() {
    // Test optional with a match
    auto parser = build_parser([](parser_builder& p) {
        return p.literal("hello") + p.optional(p.literal(" world"));
    });

    // Full match with optional part present
    auto ctx = parser_context{"hello world", parse_cache()};
    auto result = parser.parse(ctx);
    assert_equals(true, result.is_success());
    assert_equals((size_t)11, result.end);

    // Full match with optional part absent
    ctx = parser_context{"hello", parse_cache(), true};
    result = parser.parse(ctx);
    assert_equals(true, result.is_success());
    assert_equals((size_t)5, result.end);

    // Partial match - waiting for more input to determine if optional matches
    ctx = parser_context{"hello ", parse_cache(), false};
    result = parser.parse(ctx);
    assert_equals(true, result.is_need_more_input());
}

static void test_json_parser() {
    auto json = build_parser([](parser_builder & p) {
        return p.add_json_rule("json");
    });

    // Test parsing a simple JSON object
    std::string input = R"({"name": "test", "value": 42, "flag": true})";
    parser_context ctx{input, parse_cache()};

    auto result = json.parse(ctx);

    assert_equals(true, result.is_success());
    assert_equals(input.size(), result.end);
}

static void test_complete_example() {
    auto parser = build_parser([](parser_builder & p) {
        auto space = p.add_rule("space", p.space());

        auto reasoning = p.add_rule("reasoning",
            p.literal("<think>") + space +
            p.group("reasoning-content",
                p.zero_or_more(~(space + p.literal("</think>")) + p.any())) +
            space + p.literal("</think>"));

        auto content = p.add_rule("content",
            p.group("content",
                p.zero_or_more(~(space + p.literal("<tool_call>")) + p.any())));

        auto ident_chars = p.add_rule("ident-chars", p.char_class("[a-zA-Z\\-_]"));
        auto json = p.add_json_rule("json");

        auto tool_call_name = p.add_rule("tool-call-name",
            p.literal("<name>") + space +
            p.group("tool-name", p.one_or_more(~p.literal("</name>") + ident_chars)) +
            space + p.literal("</name>"));

        auto tool_call_args = p.add_rule("tool-call-args",
            p.literal("<args>") + space +
            p.group("tool-args", json) +
            space + p.literal("</args>"));

        auto tool_call = p.add_rule("tool-call",
            p.literal("<tool_call>") + space +
            tool_call_name + space +
            tool_call_args + space +
            p.literal("</tool_call>"));

        return p.add_rule("root", reasoning + p.optional(content) + p.optional(tool_call));
    });

    // Test complete input
    std::string input = R"(<think>I need to call get_weather with city = New York</think><tool_call><name>get_weather</name><args>{"city": "New York"}</args></tool_call>)";
    parser_context ctx{input, parse_cache()};

    auto result = parser.parse(ctx);

    assert_equals(true, result.is_success());
    assert_equals(input.size(), result.end);
    assert_equals(std::string("I need to call get_weather with city = New York"), *result.group("reasoning-content", ctx.input));
    assert_equals(std::string("get_weather"), *result.group("tool-name", ctx.input));
    assert_equals(std::string(R"({"city": "New York"})"), *result.group("tool-args", ctx.input));

    // Test partial input
    input = R"(<think>I need to call get_weather )";
    ctx = parser_context{input, parse_cache(), /* .is_input_complete = */ false};
    result = parser.parse(ctx);

    assert_equals(true, result.is_success());
    assert_equals(std::string("I need to call get_weather"), *result.group("reasoning-content", ctx.input));

    input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather)";
    ctx = parser_context{input, parse_cache(), /* .is_input_complete = */ false};
    result = parser.parse(ctx);

    assert_equals(true, result.is_success());
    assert_equals(std::string("I need to call get_weather"), *result.group("reasoning-content", ctx.input));

    input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather</na)";
    ctx = parser_context{input, parse_cache(), /* .is_input_complete = */ false};
    result = parser.parse(ctx);

    assert_equals(true, result.is_need_more_input());
    assert_equals(std::string("I need to call get_weather"), *result.group("reasoning-content", ctx.input));
    assert_equals(std::string("get_weather"), *result.group("tool-name", ctx.input));

    input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather</name><args>{"cit)";
    ctx = parser_context{input, parse_cache(), /* .is_input_complete = */ false};
    result = parser.parse(ctx);

    assert_equals(true, result.is_success());
    assert_equals(std::string("I need to call get_weather"), *result.group("reasoning-content", ctx.input));
    assert_equals(std::string("get_weather"), *result.group("tool-name", ctx.input));
    assert_equals(std::string(R"({"cit)"), *result.group("tool-args", ctx.input));
}

int main() {
    test_partial_parsing();
    test_char_class();
    test_capture_groups();
    test_recursive_references();
    test_optional();
    test_json_parser();
    test_complete_example();
    std::cout << "All tests passed!\n";
    return 0;
}
