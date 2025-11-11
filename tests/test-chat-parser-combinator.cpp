#include <iostream>
#include <string>

#include "chat-parser-combinator.h"
#include "json-schema-to-grammar.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

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

        ctx = parser_context("hello");
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

        ctx = parser_context("a");
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context("A");
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());

        parser = build_parser([](parser_builder& p) {
            return p.char_class("a-z-");
        });

        ctx = parser_context("f");
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context("-");
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context("A");
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());
    }
    {
        // Test sequences and literals
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("<think>") + p.literal("</think>");
        });

        // Partial matches
        auto ctx = parser_context("<thi", false);
        auto result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        ctx = parser_context("<think>", false);
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context("<think></", false);
        result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        // Full match
        ctx = parser_context("<think></think>", true);
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        // No match, since it does not adhere to the grammar
        ctx = parser_context("<think>I am parser", false);
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());
    }
    {
        // Test choices
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("<think>") | p.literal("<reasoning>");
        });

        // Partial matches
        auto ctx = parser_context("<thi", false);
        auto result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        ctx = parser_context("<reas", false);
        result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        // Full match
        ctx = parser_context("<think>", true);
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context("<reasoning>", true);
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        // No match
        ctx = parser_context("<thought>", true);
        result = parser.parse(ctx);
        assert_equals(true, result.is_fail());
    }
    {
        // Test zero_or_more
        auto parser = build_parser([](parser_builder& p) {
            return p.zero_or_more(p.literal("ab"));
        });

        // Partial matches
        auto ctx = parser_context("a", false);
        auto result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        ctx = parser_context("aba", false);
        result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        // Full match
        ctx = parser_context("ab", true);
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());
    }
    {
        // Test one_or_more
        auto parser = build_parser([](parser_builder& p) {
            return p.one_or_more(p.literal("ab"));
        });

        // Partial matches
        auto ctx = parser_context("a", false);
        auto result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        ctx = parser_context("aba", false);
        result = parser.parse(ctx);
        assert_equals(true, result.is_need_more_input());

        // Full match
        ctx = parser_context("ab", true);
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        // No match
        ctx = parser_context("cd", true);
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
        auto ctx = parser_context(input);
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
        auto ctx = parser_context(input, false);
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
        auto ctx = parser_context(input, true);
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

        ctx = parser_context("\n");
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context("\t");
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context("\\");
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context(" ");
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

        ctx = parser_context("a");
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context("-");
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        ctx = parser_context("z");
        result = parser.parse(ctx);
        assert_equals(true, result.is_success());

        // Should NOT match 'b' since \- is a literal dash, not a range
        ctx = parser_context("b");
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
    ctx = parser_context("1", true);
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_success());

    // Test simple list
    ctx = parser_context("[1]", true);
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_success());

    // Test nested list
    ctx = parser_context("[[2]]", true);
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_success());

    // Test deeply nested list
    ctx = parser_context("[[[3]]]", true);
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_success());

    // Test partial match
    ctx = parser_context("[[", false);
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_success());

    // Test no match
    ctx = parser_context("[a]", true);
    result = value_parser.parse(ctx);
    assert_equals(true, result.is_fail());
}

static void test_optional() {
    // Test optional with a match
    auto parser = build_parser([](parser_builder& p) {
        return p.literal("hello") + p.optional(p.literal(" world"));
    });

    // Full match with optional part present
    auto ctx = parser_context("hello world");
    auto result = parser.parse(ctx);
    assert_equals(true, result.is_success());
    assert_equals((size_t)11, result.end);

    // Full match with optional part absent
    ctx = parser_context("hello", true);
    result = parser.parse(ctx);
    assert_equals(true, result.is_success());
    assert_equals((size_t)5, result.end);

    // Partial match - waiting for more input to determine if optional matches
    ctx = parser_context("hello ", false);
    result = parser.parse(ctx);
    assert_equals(true, result.is_need_more_input());
}

static void test_json_parser() {
    auto json = build_parser([](parser_builder & p) {
        return p.json();
    });

    {
        // Test parsing a simple JSON object
        std::string input = R"({"name": "test", "value": 42, "flag": true})";
        parser_context ctx(input);

        auto result = json.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals(input.size(), result.end);
    }
    {
        // Test parsing a JSON array with mixed types
        std::string input = R"([1, "hello", true, null, 3.14])";
        parser_context ctx(input);

        auto result = json.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals(input.size(), result.end);
    }
    {
        // Test parsing nested JSON with objects and arrays
        std::string input = R"({"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2, "metadata": {"version": "1.0", "tags": ["admin", "user"]}})";
        parser_context ctx(input);

        auto result = json.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals(input.size(), result.end);
    }
    {
        // Test partial parsing - incomplete object
        std::string input = R"({"name": "test", "value": )";
        parser_context ctx(input, false);

        auto result = json.parse(ctx);

        assert_equals(true, result.is_success());
    }
    {
        // Test partial parsing - incomplete array
        std::string input = R"([1, 2, 3, )";
        parser_context ctx(input, false);

        auto result = json.parse(ctx);

        assert_equals(true, result.is_success());
    }
    {
        // Test partial parsing - incomplete nested structure
        std::string input = R"({"data": {"nested": )";
        parser_context ctx(input, false);

        auto result = json.parse(ctx);

        assert_equals(true, result.is_success());
    }
}

static void test_complete_example() {
    // Parser for a fictitious model that outputs:
    //
    //   <think>
    //   ... reasoning content ...
    //   </think>
    //   ... content ...
    //   <tool_call>
    //   <name>tool_name</name>
    //   <args>{ ... json args ... }</args>
    //   </tool_call>
    //
    auto parser = build_parser([](parser_builder & p) {
        auto reasoning = p.add_rule("reasoning",
            p.literal("<think>")
            << p.group("reasoning-content", p.until("</think>"))
            << p.literal("</think>"));

        auto content = p.add_rule("content",
            p.group("content", p.until("<tool_call>")));

        auto json = p.json();

        auto tool_call_name = p.add_rule("tool-call-name",
            p.literal("<name>")
            << p.group("tool-name", p.one_or_more(p.char_class("[a-zA-Z\\-_]")))
            << p.literal("</name>"));

        auto schema = nlohmann::ordered_json::parse(R"({"type": "object"})");

        auto tool_call_args = p.add_rule("tool-call-args",
            p.literal("<args>")
            << p.group("tool-args", p.schema(json, "get_weather", schema))
            << p.literal("</args>"));

        auto tool_call = p.add_rule("tool-call",
            p.literal("<tool_call>")
            << tool_call_name
            << tool_call_args
            << p.literal("</tool_call>"));

        return reasoning << p.optional(content) << p.optional(tool_call);
    });

    // Test complete input
    std::string input = R"(<think>I need to call get_weather with city = New York</think><tool_call><name>get_weather</name><args>{"city": "New York"}</args></tool_call>)";
    parser_context ctx(input);

    auto result = parser.parse(ctx);

    assert_equals(true, result.is_success());
    assert_equals(input.size(), result.end);
    assert_equals(std::string("I need to call get_weather with city = New York"), *result.group("reasoning-content", ctx.input));
    assert_equals(std::string("get_weather"), *result.group("tool-name", ctx.input));
    assert_equals(std::string(R"({"city": "New York"})"), *result.group("tool-args", ctx.input));

    // Test partial input
    input = R"(<think>I need to call get_weather )";
    ctx = parser_context(input, /* .is_input_complete = */ false);
    result = parser.parse(ctx);

    assert_equals(true, result.is_success());
    assert_equals(std::string("I need to call get_weather"), *result.group("reasoning-content", ctx.input));

    input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather)";
    ctx = parser_context(input, /* .is_input_complete = */ false);
    result = parser.parse(ctx);

    assert_equals(true, result.is_success());
    assert_equals(std::string("I need to call get_weather"), *result.group("reasoning-content", ctx.input));

    input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather</na)";
    ctx = parser_context(input, /* .is_input_complete = */ false);
    result = parser.parse(ctx);

    assert_equals(true, result.is_need_more_input());
    assert_equals(std::string("I need to call get_weather"), *result.group("reasoning-content", ctx.input));
    assert_equals(std::string("get_weather"), *result.group("tool-name", ctx.input));

    input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather</name><args>{"cit)";
    ctx = parser_context(input, /* .is_input_complete = */ false);
    result = parser.parse(ctx);

    assert_equals(true, result.is_success());
    assert_equals(std::string("I need to call get_weather"), *result.group("reasoning-content", ctx.input));
    assert_equals(std::string("get_weather"), *result.group("tool-name", ctx.input));
    assert_equals(std::string(R"({"cit)"), *result.group("tool-args", ctx.input));

    auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
        parser.build_grammar(builder);
    });

    std::cout << "Grammar:\n" << gbnf << "\n";
}

static void test_gbnf_generation() {
    {
        // Test literal
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("hello");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= \"hello\"") != std::string::npos);
        assert_equals(true, gbnf.find("space ::=") != std::string::npos);
    }
    {
        // Test char class
        auto parser = build_parser([](parser_builder& p) {
            return p.char_class("[a-z]");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= [a-z]") != std::string::npos);
    }
    {
        // Test sequence
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("hello") + p.literal(" ") + p.literal("world");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= \"hello\" \" \" \"world\"") != std::string::npos);
    }
    {
        // Test choice
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("cat") | p.literal("dog");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= \"cat\" | \"dog\"") != std::string::npos);
    }
    {
        // Test one_or_more
        auto parser = build_parser([](parser_builder& p) {
            return p.one_or_more(p.char_class("[0-9]"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= [0-9]+") != std::string::npos);
    }
    {
        // Test zero_or_more
        auto parser = build_parser([](parser_builder& p) {
            return p.zero_or_more(p.char_class("[a-z]"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= [a-z]*") != std::string::npos);
    }
    {
        // Test optional
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("hello") + p.optional(p.literal(" world"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= \"hello\" \" world\"?") != std::string::npos);
    }
    {
        // Test until
        auto parser = build_parser([](parser_builder& p) {
            return p.until("</tag>");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        // Should generate pattern that prevents matching the full delimiter
        assert_equals(true, gbnf.find("root ::= ([^<] | \"<\" [^/] | \"</\" [^t] | \"</t\" [^a] | \"</ta\" [^g] | \"</tag\" [^>])*") != std::string::npos);
    }
    {
        // Test groups are transparent
        auto parser = build_parser([](parser_builder& p) {
            return p.group("test", p.literal("hello"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= \"hello\"") != std::string::npos);
    }
    {
        // Test complex expression with parentheses
        auto parser = build_parser([](parser_builder& p) {
            return p.one_or_more(p.literal("a") | p.literal("b"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= (\"a\" | \"b\")+") != std::string::npos);
    }
    {
        // Test rule references
        auto parser = build_parser([](parser_builder& p) {
            auto digit = p.add_rule("digit", p.char_class("[0-9]"));
            return p.one_or_more(digit);
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        // Should have digit rule defined and referenced
        assert_equals(true, gbnf.find("digit ::= [0-9]") != std::string::npos);
        assert_equals(true, gbnf.find("root ::= digit+") != std::string::npos);
    }
    {
        // Test escaping in literals
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("hello\nworld\t!");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= \"hello\\nworld\\t!\"") != std::string::npos);
    }
    {
        // Test operator<< (whitespace insertion)
        auto parser = build_parser([](parser_builder& p) {
            return p.literal("hello") << p.literal("world");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        // Should inline the whitespace pattern
        assert_equals(true, gbnf.find("\"hello\"") != std::string::npos);
        assert_equals(true, gbnf.find("\"world\"") != std::string::npos);
    }
}

int main() {
    test_partial_parsing();
    test_char_class();
    test_capture_groups();
    test_recursive_references();
    test_optional();
    test_json_parser();
    test_complete_example();
    test_gbnf_generation();
    std::cout << "All tests passed!\n";
    return 0;
}
