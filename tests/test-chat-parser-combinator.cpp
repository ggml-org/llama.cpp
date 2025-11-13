#include <iostream>
#include <string>
#include <chrono>

#include "nlohmann/json.hpp"

#include "chat.h"
#include "chat-parser.h"
#include "chat-parser-combinator.h"
#include "common.h"
#include "json-schema-to-grammar.h"

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
            return p.one("a-z");
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
            return p.one("a-z-");
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
        assert_equals(true, result.is_need_more_input());

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

static void test_one() {
    {
        // Test common escape sequences
        auto parser = build_parser([](parser_builder& p) {
            return p.one("[\\n\\t\\\\]");
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
            return p.one("[a\\-z]");
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
        p.add_rule("number", p.one_or_more(p.one("0-9")));
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
    assert_equals(true, result.is_need_more_input());

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

        assert_equals(true, result.is_need_more_input());
    }
    {
        // Test partial parsing - incomplete array
        std::string input = R"([1, 2, 3, )";
        parser_context ctx(input, false);

        auto result = json.parse(ctx);

        assert_equals(true, result.is_need_more_input());
    }
    {
        // Test partial parsing - incomplete nested structure
        std::string input = R"({"data": {"nested": )";
        parser_context ctx(input, false);

        auto result = json.parse(ctx);

        assert_equals(true, result.is_need_more_input());
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
            "<think>" << p.append_reasoning(p.until("</think>")) << "</think>");

        auto content = p.add_rule("content",
            p.append_content(p.until("<tool_call>")));

        auto json = p.json();

        auto tool_call_name = p.add_rule("tool-call-name",
            "<name>" << p.capture_tool_call_name(p.until("</name>")) << "</name>");

        auto schema = nlohmann::ordered_json::parse(R"({"type": "object"})");

        auto tool_call_args = p.add_rule("tool-call-args",
            "<args>" << p.capture_tool_call_args(p.schema(p.succeed(json), "get_weather", schema)) << "</args>");

        auto tool_call = p.add_rule("tool-call",
            "<tool_call>" << p.add_tool_call(tool_call_name << p.succeed(tool_call_args)) << "</tool_call>");

        return reasoning << p.optional(content) << p.optional(tool_call);
    });

    // Test complete input
    {
        std::string input = R"(<think>I need to call get_weather with city = New York</think><tool_call><name>get_weather</name><args>{"city": "New York"}</args></tool_call>)";
        parser_environment env;
        parser_context ctx(input, &env);

        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals(input.size(), result.end);
        assert_equals("I need to call get_weather with city = New York", env.result.reasoning_content);
        assert_equals((size_t)1, env.result.tool_calls.size());
        assert_equals("", env.result.tool_calls[0].id);
        assert_equals("get_weather", env.result.tool_calls[0].name);
        assert_equals(R"({"city": "New York"})", env.result.tool_calls[0].arguments);
    }

    // Test partial input
    {
        std::string input = R"(<think>I need to call get_weather)";
        parser_environment env = parser_environment();
        parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

        auto result = parser.parse(ctx);

        assert_equals(true, result.is_need_more_input());
        assert_equals("I need to call get_weather", env.result.reasoning_content);
    }
    {
        std::string input = R"(<think>I need to call </thi get_weather</th)";
        parser_environment env = parser_environment();
        parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

        auto result = parser.parse(ctx);

        assert_equals(true, result.is_need_more_input());
    }
    {
        std::string input = R"(<think>I need to call get_weather</th)";
        parser_environment env = parser_environment();
        parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

        auto result = parser.parse(ctx);

        assert_equals(true, result.is_need_more_input());
    }
    {
        std::string input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather)";
        parser_environment env = parser_environment();
        parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

        auto result = parser.parse(ctx);

        assert_equals(true, result.is_need_more_input());
        assert_equals("I need to call get_weather", env.result.reasoning_content);
    }
    {
        std::string input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather</na)";
        parser_environment env = parser_environment();
        parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

        auto result = parser.parse(ctx);

        assert_equals(true, result.is_need_more_input());
        assert_equals("I need to call get_weather", env.result.reasoning_content);
    }
    {
        std::string input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather</name><args>{"cit)";
        parser_environment env = parser_environment();
        parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

        auto result = parser.parse(ctx);

        assert_equals(true, result.is_need_more_input());
        assert_equals("I need to call get_weather", env.result.reasoning_content);
        assert_equals("get_weather", env.result.tool_calls[0].name);
        assert_equals(R"({"cit)", env.result.tool_calls[0].arguments);
    }

    auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
        parser.build_grammar(builder);
    });

    std::cout << "Grammar:\n" << gbnf << "\n";
}

static void test_actions() {
    {
        // Test simple action - append matched text to content
        auto parser = build_parser([](parser_builder& p) {
            auto word = p.chars("[a-z]+");
            return p.action(word, [](const parser_result &, std::string_view matched, parser_environment & env) {
                env.result.content += std::string(matched);
            });
        });

        parser_environment env;
        parser_context ctx("hello", &env);
        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals("hello", env.result.content);
    }
    {
        // Test multiple sequential actions - build a sentence
        auto parser = build_parser([](parser_builder& p) {
            auto greeting = p.action(p.literal("hello"), [](const parser_result &, std::string_view matched, parser_environment & env) {
                env.result.content += std::string(matched) + " ";
            });

            auto name = p.action(p.chars("[A-Z][a-z]+"), [](const parser_result &, std::string_view matched, parser_environment & env) {
                env.result.content += std::string(matched);
                env.scratchpad["name"] = std::string(matched);
            });

            return greeting + p.literal(" ") + name;
        });

        parser_environment env;
        parser_context ctx("hello Alice", &env);
        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals("hello Alice", env.result.content);
        assert_equals("Alice", std::get<std::string>(env.scratchpad["name"]));
    }
    {
        // Test using scratchpad for intermediate calculations
        auto parser = build_parser([](parser_builder& p) {
            auto digit = p.action(p.one("[0-9]"), [](const parser_result &, std::string_view matched, parser_environment & env) {
                auto it = env.scratchpad.find("sum");
                int current_sum = it != env.scratchpad.end() ? std::get<int>(it->second) : 0;
                current_sum += (matched[0] - '0');
                env.scratchpad["sum"] = current_sum;
            });

            return p.one_or_more(digit + p.optional(p.literal("+")));
        });

        parser_environment env;
        parser_context ctx("1+2+3+4", &env);
        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals(10, std::get<int>(env.scratchpad["sum"]));  // 1+2+3+4 = 10
    }
    {
        // Test actions don't run when parse fails
        auto parser = build_parser([](parser_builder& p) {
            return p.action(p.literal("success"), [](const parser_result &, std::string_view, parser_environment & env) {
                env.result.content = "action_ran";
            });
        });

        parser_environment env;
        parser_context ctx("failure", &env);
        auto result = parser.parse(ctx);

        assert_equals(true, result.is_fail());
        assert_equals("", env.result.content);  // Action should not have run
    }
    {
        // Test Actions work with partial parsing
        auto parser = build_parser([](parser_builder& p) {
            auto content = p.action(p.until("<end>"), [](const parser_result &, std::string_view matched, parser_environment & env) {
                env.result.content += std::string(matched);
            });
            return "<start>" << content << "<end>";
        });

        {
            parser_environment env;
            parser_context ctx("<start>hello ", &env, false);
            auto result = parser.parse(ctx);

            assert_equals(true, result.is_need_more_input());
            assert_equals("hello ", env.result.content);
        }
        {
            parser_environment env;
            parser_context ctx("<start>hello world", &env, false);
            auto result = parser.parse(ctx);

            assert_equals(true, result.is_need_more_input());
            assert_equals("hello world", env.result.content);
        }
        {
            parser_environment env;
            parser_context ctx("<start>hello world<end>", &env, true);
            auto result = parser.parse(ctx);

            assert_equals(true, result.is_success());
            assert_equals("hello world", env.result.content);
        }
    }
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
            return p.one("[a-z]");
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
            return p.one_or_more(p.one("[0-9]"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= [0-9]+") != std::string::npos);
    }
    {
        // Test zero_or_more
        auto parser = build_parser([](parser_builder& p) {
            return p.zero_or_more(p.one("[a-z]"));
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
            auto digit = p.add_rule("digit", p.one("[0-9]"));
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

// Simple tokenize function that splits by space and special chars
static std::vector<std::string> simple_tokenize(const std::string & input) {
    std::vector<std::string> result;
    std::string current;

    for (size_t i = 0; i < input.size(); i++) {
        switch (input[i]) {
        case ' ':
        case '\n':
        case '\t':
        case '{':
        case '}':
        case ',':
        case '[':
        case '"':
        case ']':
        case '.':
        case '<':
        case '>':
        case '=':
        case '/':
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
        }
        current += input[i];
    }

    if (!current.empty()) {
        result.push_back(current);
    }

    return result;
}

static void example_qwen3_coder() {
    auto parser = build_parser([](parser_builder & p) {
        auto thinking = p.add_rule("thinking",
            "<think>" << p.append_reasoning(p.until("</think>")) << "</think>");

        auto content = p.add_rule("content", p.append_content(p.until("<tool_call>")));

        auto arg_start = p.add_rule("arg-start",
            p.action("<parameter=", [](const parser_result &, std::string_view, parser_environment & env) {
                if (env.tool_call_args != "{") {
                    env.tool_call_args += ",";
                }
                env.tool_call_args += "\"";
            })
            + p.action(p.chars("[a-zA-Z0-9_]"), [](const parser_result &, std::string_view match, parser_environment & env) {
                env.tool_call_args += std::string(match);
            })
            + p.action(">", [](const parser_result &, std::string_view, parser_environment & env) {
                env.tool_call_args += "\":";
            }));

        auto arg_end = p.add_rule("arg-end", "</parameter>");

        auto string_arg = p.add_rule("arg-string",
            p.action(arg_start, [&](const parser_result &, std::string_view, parser_environment & env) {
                env.tool_call_args += "\"";
            })
            << p.action(p.until("</parameter>"), [&](const parser_result &, std::string_view match, parser_environment & env) {
                // TODO: add a JSON escape helper
                env.tool_call_args += std::string(match);
            })
            << p.action(arg_end, [&](const parser_result &, std::string_view, parser_environment & env) {
                env.tool_call_args += "\"";
            }));

        auto json = p.json();

        auto json_arg = p.add_rule("arg-json",
            arg_start
            << p.action(json, [&](const parser_result &, std::string_view match, parser_environment & env) {
                // JSON should already be properly formatted
                env.tool_call_args += std::string(match);

                // This can be streamed by passing p.success(json), but we have
                // to be mindful of the potential backtracking--it only works
                // if we only keep the last value...
            })
            << arg_end);

        auto function = p.add_rule("function", p.add_tool_call(
                "<function="
                + p.capture_tool_call_name(p.chars("[a-zA-Z0-9_]"))
                + p.action(">", [&](const parser_result &, std::string_view, parser_environment & env) {
                    env.tool_call_args += "{";
                })
                + p.one_or_more(p.space() + (json_arg | string_arg))
                << p.action("</function>", [&](const parser_result &, std::string_view, parser_environment & env) {
                    env.tool_call_args += "}";
                })));

        auto tool_call = p.add_rule("tool-call",
            "<tool_call>" << p.one_or_more(function) << "</tool_call>");


        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call);
    });

    std::string input =
        "<think>The user wants to find large log files that haven't been accessed recently. "
        "I should search for files with .log extension, filter by size (over 100MB), "
        "and check access time within the last 30 days. I'll need to use the search_files function.</think>"
        "Based on your requirements, I'll search for log files over 100MB that haven't been "
        "accessed in the last month. This will help identify candidates for cleanup or archival.\n\n"
        "<tool_call>\n"
        "<function=search_files>\n"
        "<parameter=path>/var/log</parameter>\n"
        "<parameter=pattern>*.log</parameter>\n"
        "<parameter=min_size_mb>100</parameter>\n"
        "<parameter=max_depth>5</parameter>\n"
        "<parameter=include_hidden>false</parameter>\n"
        "<parameter=modified_days_ago>30</parameter>\n"
        "<parameter=case_sensitive>true</parameter>\n"
        "<parameter=sort_by>size</parameter>\n"
        "<parameter=filters>{\"exclude_patterns\": [\"*temp*\", \"*cache*\"], \"file_types\": [\"regular\"]}</parameter>\n"
        "</function>\n"
        "</tool_call>";

    std::vector<std::string> tokens = simple_tokenize(input);

    common_chat_msg prev;

    for (auto it = tokens.begin(); it != tokens.end(); it++) {
        std::string in = std::accumulate(tokens.begin(), it, std::string());

        parser_environment env;
        parser_context ctx(in, &env, it == tokens.end() - 1);

        auto result = parser.parse(ctx);
        if (result.is_fail()) {
            break;
        }

        /*
        std::cout << "Input:\n" << in << "\n\n";
        std::cout << "Reasoning: " << prev.reasoning_content << "\n";
        std::cout << "Content  : " << prev.content << "\n";
        if (!prev.tool_calls.empty()) {
            std::cout << "\n=== Tool Calls ===\n";
            for (const auto & tc : prev.tool_calls) {
                std::cout << "ID  : " << tc.id << "\n";
                std::cout << "Name: " << tc.name << "\n";
                std::cout << "Args: " << tc.arguments << "\n";
            }
        }
        */

        // This shouldn't emit any runtime errors
        auto diffs = common_chat_msg_diff::compute_diffs(prev, env.result);
        prev = env.result;

        /*
        std::cout << "----\n";
        std::cout << "Reasoning: " << prev.reasoning_content << "\n";
        std::cout << "Content  : " << prev.content << "\n";
        if (!prev.tool_calls.empty()) {
            std::cout << "\n=== Tool Calls ===\n";
            for (const auto & tc : prev.tool_calls) {
                std::cout << "ID  : " << tc.id << "\n";
                std::cout << "Name: " << tc.name << "\n";
                std::cout << "Args: " << tc.arguments << "\n";
            }
        }
        std::cout << "======================\n";
        */

        /*
        std::cout << "=== Diffs ===\n\n";
        if (!diffs.empty()) {
            for (size_t i = 0; i < diffs.size(); ++i) {
                const auto& diff = diffs[i];

                std::cout << "Diff #" << (i + 1) << "\n";

                if (!diff.reasoning_content_delta.empty()) {
                    std::cout << "  [Reasoning Content]: " << diff.reasoning_content_delta << "\n";
                }

                if (!diff.content_delta.empty()) {
                    std::cout << "  [Content]: " << diff.content_delta << "\n";
                }

                if (diff.tool_call_index != std::string::npos) {
                    std::cout << "  [Tool Call #" << diff.tool_call_index << "]" << "\n";

                    if (!diff.tool_call_delta.id.empty()) {
                        std::cout << "    ID: " << diff.tool_call_delta.id << "\n";
                    }

                    if (!diff.tool_call_delta.name.empty()) {
                        std::cout << "    Name: " << diff.tool_call_delta.name << "\n";
                    }

                    if (!diff.tool_call_delta.arguments.empty()) {
                        std::cout << "    Arguments: " << diff.tool_call_delta.arguments << "\n";
                    }
                }

                std::cout << "\n";
            }
        } else {
            std::cout << "No changes detected.\n";
        }
        */
    }
}

static parser create_command_r7b_parser() {
    auto parser = build_parser([](parser_builder & p) {
        auto thinking = p.add_rule("thinking",
            "<|START_THINKING|>" << p.append_reasoning(p.until("<|END_THINKING|>")) << "<|END_THINKING|>");

        auto response = p.add_rule("response",
            "<|START_RESPONSE|>" << p.append_content(p.until("<|END_RESPONSE|>")) << "<|END_RESPONSE|>");

        auto json = p.add_rule("json", p.json());

        auto tool_call_id = p.add_rule("tool-call-id",
            p.json_key("tool_call_id", "\"" + p.capture_tool_call_id(p.json_string(), /* unescape_json = */ true) + "\""));

        auto tool_call_name = p.add_rule("tool-name",
            p.json_key("tool_name", "\"" + p.capture_tool_call_name(p.json_string(), /* unescape_json = */ true) + "\""));

        auto tool_call_args = p.add_rule("tool-args", p.json_key("parameters", p.capture_tool_call_args(json)));

        auto tool_call_fields = p.add_rule("tool-call-fields", tool_call_id | tool_call_name | tool_call_args);

        auto tool_call = p.add_rule("tool-call",
            "{" << p.add_tool_call(tool_call_fields << p.zero_or_more(p.literal(",") << tool_call_fields)) << "}");

        auto tool_calls = p.add_rule("tool-calls",
            "<|START_ACTION|>"
            << ("[" << tool_call << p.zero_or_more(p.literal(",") << tool_call) << "]")
            << "<|END_ACTION|>");

        return p.optional(thinking) << p.add_rule("content", tool_calls | response);
    });

    auto grammar = build_grammar([&](const common_grammar_builder & builder) {
        parser.build_grammar(builder);
    });

    std::cout << "=== Grammar ===\n\n" << grammar << "\n\n";

    return parser;
}

static void test_command_r7b_parser(const parser & p, const std::string & input, bool partial, bool print_results = false) {
    parser_environment env;
    parser_context ctx(input, &env, !partial);
    p.parse(ctx);

    if (print_results) {
        std::cout << "== Parsed (new) ==\n";
        std::cout << "=== Reasoning ===\n";
        std::cout << env.result.reasoning_content << "\n";
        std::cout << "\n\n=== Content ===\n";
        std::cout << env.result.content << "\n";
        std::cout << "\n\n=== Tool Calls ===\n";
        for (const auto & tc : env.result.tool_calls) {
            std::cout << "id: " << tc.id << "\n";
            std::cout << "name: " << tc.name << "\n";
            std::cout << "args: " << tc.arguments << "\n";
        }
    }
}

static void test_command_r7b_legacy_parser(const std::string & input, bool partial, bool print_results = false) {
    // Original parser taken from chat.cpp
    common_chat_msg_parser builder(input,
        /* is_partial= */ partial, {
        /* .format = */ COMMON_CHAT_FORMAT_GENERIC,
        /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
        /* .reasoning_in_content = */ false,
        /* .thinking_forced_open = */ false,
    });

    builder.try_parse_reasoning("<|START_THINKING|>", "<|END_THINKING|>");

    static const common_regex start_action_regex("<\\|START_ACTION\\|>");
    static const common_regex end_action_regex("<\\|END_ACTION\\|>");
    static const common_regex start_response_regex("<\\|START_RESPONSE\\|>");
    static const common_regex end_response_regex("<\\|END_RESPONSE\\|>");

    if (auto res = builder.try_find_regex(start_action_regex)) {
        // If we didn't extract thoughts, prelude includes them.
        auto tool_calls = builder.consume_json_with_dumped_args({{"parameters"}});
        for (const auto & tool_call : tool_calls.value) {
            std::string name = tool_call.contains("tool_name") ? tool_call.at("tool_name") : "";
            std::string id = tool_call.contains("tool_call_id") ? tool_call.at("tool_call_id") : "";
            std::string arguments = tool_call.contains("parameters") ? tool_call.at("parameters") : "";
            if (!builder.add_tool_call(name, id, arguments) || tool_calls.is_partial) {
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
        }
        if (tool_calls.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool call");
        }
        builder.consume_regex(end_action_regex);
    } else if (auto res = builder.try_find_regex(start_response_regex)) {
        if (!builder.try_find_regex(end_response_regex)) {
            builder.add_content(builder.consume_rest());
            throw common_chat_msg_partial_exception(end_response_regex.str());
        }
    } else {
        builder.add_content(builder.consume_rest());
    }

    if (print_results) {
        std::cout << "== Parsed (legacy) ==\n";
        std::cout << "=== Reasoning ===\n";
        std::cout << builder.result().reasoning_content << "\n";
        std::cout << "\n\n=== Content ===\n";
        std::cout << builder.result().content << "\n";
        std::cout << "\n\n=== Tool Calls ===\n";
        for (const auto & tc : builder.result().tool_calls) {
            std::cout << "id: " << tc.id << "\n";
            std::cout << "name: " << tc.name << "\n";
            std::cout << "args: " << tc.arguments << "\n";
        }
    }
}

struct bench_tool_call {
    std::string id;
    std::string name;
    nlohmann::ordered_json args;
};

static void benchmark_compare(
    const std::string & reasoning,
    const std::string & content,
    const std::vector<bench_tool_call> & tool_calls,
    int iterations) {

    // Build response
    std::vector<std::string> tokens; // Since we don't have a command r7b tokenizer, we're going to "simulate" them.

    if (!reasoning.empty()) {
        auto tokenized = simple_tokenize(reasoning);
        tokens.emplace_back("<|START_THINKING|>");
        tokens.insert(tokens.end(), tokenized.begin(), tokenized.end());
        tokens.emplace_back("<|END_THINKING|>");
    }

    if (!content.empty()) {
        auto tokenized = simple_tokenize(content);
        tokens.emplace_back("<|START_RESPONSE|>");
        tokens.insert(tokens.end(), tokenized.begin(), tokenized.end());
        tokens.emplace_back("<|END_RESPONSE|>");
    }

    if (!tool_calls.empty()) {
        tokens.emplace_back("<|START_ACTION|>");

        auto json = nlohmann::json::array();
        for (const auto & tc : tool_calls) {
            auto tc_json = nlohmann::json::object();
            tc_json["tool_call_id"] = tc.id;
            tc_json["tool_name"] = tc.name;
            tc_json["parameters"] = tc.args;
            json.push_back(tc_json);
        }

        auto tokenized = simple_tokenize(json.dump(-1, ' ', true));
        tokens.insert(tokens.end(), tokenized.begin(), tokenized.end());

        tokens.emplace_back("<|END_ACTION|>");
    }

    auto run = [&](const std::function<void(const std::string &, bool, bool)> & fn) {
        std::string input = std::accumulate(tokens.begin(), tokens.end(), std::string());

        std::chrono::microseconds duration(0);
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            fn(input, false, i == 0);
            auto end = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        }
        return duration.count() / iterations;
    };

    auto parser = create_command_r7b_parser();

    auto duration_new = run([&](const std::string & input, bool partial, bool print_content) {
        test_command_r7b_parser(parser, input, partial, print_content);
    });

    auto duration_legacy = run([&](const std::string & input, bool partial, bool print_content) {
        try {
            test_command_r7b_legacy_parser(input, partial, print_content);
        } catch (const common_chat_msg_partial_exception &) { }
    });

    std::cout << "   New parser avg: " << duration_new << " us\n";
    std::cout << "Legacy parser avg: " << duration_legacy << " us\n";
}

int main() {
    test_partial_parsing();
    test_one();
    test_recursive_references();
    test_optional();
    test_json_parser();
    test_complete_example();
    test_actions();
    test_gbnf_generation();
    std::cout << "All tests passed!\n";

    example_qwen3_coder();

    std::cout << "\n== Benchmarks ==\n";
    std::string example_reasoning =
        "To plan an effective trip to Japan that includes both historical sites and modern attractions within a budget of $4000 for a two-week stay, we need to:\n\n"
        "1. Identify key historical sites and modern attractions in Japan.\n"
        "2. Find affordable accommodation options that provide a balance between comfort and cost.\n"
        "3. Determine the best modes of transportation for getting around Japan.\n"
        "4. Create a day-by-day itinerary that ensures the user gets to see a variety of attractions without overspending.\n"
        "5. Provide a detailed cost breakdown that includes accommodation, transportation, meals, and entry fees to attractions.";

    std::string example_content =
        "For a two-week trip to Japan with a $4,000 budget, I recommend planning an itinerary that balances historical sites with modern attractions. The destination will be Japan, with a duration of 14 days.\n\n"
        "Given your interests in both historical sites and modern attractions, you'll want to focus on cities like Kyoto for its temples and traditional culture, Tokyo for its cutting-edge technology and entertainment districts, and possibly Hiroshima or Nara for additional historical significance.\n\n"
        "For accommodation, I suggest looking for affordable options such as budget hotels, hostels, or guesthouses that offer good value without sacrificing too much comfort. Japan has excellent mid-range accommodation options that can keep your lodging costs manageable.\n\n"
        "Transportation should prioritize efficiency—consider getting a JR Rail Pass for intercity travel, which allows unlimited rides on most JR trains including the Shinkansen (bullet train). Within cities, use local trains and subways, which are both affordable and highly reliable.\n\n"
        "For meals, embrace local cuisine by eating at neighborhood restaurants, ramen shops, and izakayas rather than touristy establishments. This will give you an authentic experience while keeping costs reasonable—you can enjoy excellent meals for $10-20 per person at local spots.\n\n";

    std::vector<bench_tool_call> example_tool_calls = {{
        "call_0",
        "plan_trip",
        nlohmann::json::parse(R"({
            "destination": "Japan",
            "duration": 14,
            "budget": 4000,
            "interests": ["historical sites", "modern attractions"],
            "accommodation_preferences": "affordable",
            "transportation_preferences": "efficient",
            "meal_preferences": "local cuisine"
        })")
    }};

    std::cout << "\nReasoning + Content:\n";
    benchmark_compare(example_reasoning, example_content, std::vector<bench_tool_call>(), 100);

    std::cout << "\nReasoning + Tool Call:\n";
    benchmark_compare(example_reasoning, "", example_tool_calls, 100);
    return 0;
}
