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
        auto handle_reasoning = [](const parser_result &, std::string_view match, parser_environment & env) {
            env.reasoning_content += match;
        };

        auto handle_content = [](const parser_result &, std::string_view match, parser_environment & env) {
            env.content += match;
        };

        auto handle_tool_call_name = [](const parser_result &, std::string_view match, parser_environment & env) {
            env.scratchpad["tool_name"] = std::string(match);
        };

        auto handle_tool_call_args = [](const parser_result &, std::string_view match, parser_environment & env) {
            env.scratchpad["tool_args"] = std::string(match);
        };

        auto handle_tool_call = [](const parser_result &, std::string_view, parser_environment & env) {
            auto name = env.scratchpad.find("tool_name");
            auto args = env.scratchpad.find("tool_args");
            if (name != env.scratchpad.end() && args != env.scratchpad.end()) {
                auto tool_call = common_chat_tool_call{
                    std::get<std::string>(name->second),
                    std::get<std::string>(args->second),
                    std::string()
                };

                env.tool_calls.push_back(tool_call);
            }
        };

        auto reasoning = p.add_rule("reasoning",
            "<think>" << p.action(p.until("</think>"), handle_reasoning) << "</think>");

        auto content = p.add_rule("content",
            p.action(p.until("<tool_call>"), handle_content));

        auto json = p.json();

        auto tool_call_name = p.add_rule("tool-call-name",
            "<name>" << p.action(p.until("</name>"), handle_tool_call_name) << "</name>");

        auto schema = nlohmann::ordered_json::parse(R"({"type": "object"})");

        auto tool_call_args = p.add_rule("tool-call-args",
            "<args>" << p.action(p.schema(json, "get_weather", schema), handle_tool_call_args) << "</args>");

        auto tool_call = p.add_rule("tool-call",
            "<tool_call>" << p.action(tool_call_name << tool_call_args, handle_tool_call) << "</tool_call>");

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
        assert_equals("I need to call get_weather with city = New York", env.reasoning_content);
        assert_equals((size_t)1, env.tool_calls.size());
        assert_equals("", env.tool_calls[0].id);
        assert_equals("get_weather", env.tool_calls[0].name);
        assert_equals(R"({"city": "New York"})", env.tool_calls[0].arguments);
    }

    // Test partial input
    {
        std::string input = R"(<think>I need to call get_weather )";
        parser_environment env = parser_environment();
        parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals("I need to call get_weather", env.reasoning_content);
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

        assert_equals(true, result.is_success());
        assert_equals("I need to call get_weather", env.reasoning_content);
    }
    {
        std::string input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather</na)";
        parser_environment env = parser_environment();
        parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

        auto result = parser.parse(ctx);

        assert_equals(true, result.is_need_more_input());
        assert_equals("I need to call get_weather", env.reasoning_content);
    }
    {
        std::string input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather</name><args>{"cit)";
        parser_environment env = parser_environment();
        parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals("I need to call get_weather", env.reasoning_content);
        assert_equals("get_weather", std::get<std::string>(env.scratchpad["tool_name"]));
        assert_equals(R"({"cit)", std::get<std::string>(env.scratchpad["tool_args"]));
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
                env.content += std::string(matched);
            });
        });

        parser_environment env;
        parser_context ctx("hello", &env);
        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals("hello", env.content);
    }
    {
        // Test multiple sequential actions - build a sentence
        auto parser = build_parser([](parser_builder& p) {
            auto greeting = p.action(p.literal("hello"), [](const parser_result &, std::string_view matched, parser_environment & env) {
                env.content += std::string(matched) + " ";
            });

            auto name = p.action(p.chars("[A-Z][a-z]+"), [](const parser_result &, std::string_view matched, parser_environment & env) {
                env.content += std::string(matched);
                env.scratchpad["name"] = std::string(matched);
            });

            return greeting + p.literal(" ") + name;
        });

        parser_environment env;
        parser_context ctx("hello Alice", &env);
        auto result = parser.parse(ctx);

        assert_equals(true, result.is_success());
        assert_equals("hello Alice", env.content);
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
                env.content = "action_ran";
            });
        });

        parser_environment env;
        parser_context ctx("failure", &env);
        auto result = parser.parse(ctx);

        assert_equals(true, result.is_fail());
        assert_equals("", env.content);  // Action should not have run
    }
    {
        // Test Actions work with partial parsing
        auto parser = build_parser([](parser_builder& p) {
            auto content = p.action(p.until("<end>"), [](const parser_result &, std::string_view matched, parser_environment & env) {
                env.content += std::string(matched);
            });
            return "<start>" << content << "<end>";
        });

        {
            parser_environment env;
            parser_context ctx("<start>hello ", &env, false);
            auto result = parser.parse(ctx);

            assert_equals(true, result.is_success());
            assert_equals("hello", env.content);
        }
        {
            parser_environment env;
            parser_context ctx("<start>hello world", &env, false);
            auto result = parser.parse(ctx);

            assert_equals(true, result.is_success());
            assert_equals("hello world", env.content);
        }
        {
            parser_environment env;
            parser_context ctx("<start>hello world<end>", &env, true);
            auto result = parser.parse(ctx);

            assert_equals(true, result.is_success());
            assert_equals("hello world", env.content);
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

static parser create_command_r7b_parser() {
    auto parser = build_parser([](parser_builder & p) {
        auto thinking = p.add_rule("thinking",
            "<|START_THINKING|>" << p.until("<|END_THINKING|>") << "<|END_THINKING|>");

        auto response = p.add_rule("response",
            "<|START_RESPONSE|>" << p.until("<|END_RESPONSE|>") << "<|END_RESPONSE|>");

        auto json = p.add_rule("json", p.json());
        auto tool_call_id = p.add_rule("tool-call-id", p.json_key("tool_call_id", p.json_string(p.until("\""))));
        auto tool_call_name = p.add_rule("tool-name", p.json_key("tool_name", p.json_string(p.until("\""))));
        auto tool_call_args = p.add_rule("tool-args", p.json_key("parameters", json));
        auto tool_call_fields = p.add_rule("tool-call-fields", tool_call_id | tool_call_name | tool_call_args);

        auto tool_call = p.add_rule("tool-call",
            "{" << tool_call_fields << p.zero_or_more(p.literal(",") << tool_call_fields) << "}");

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

static void test_command_r7b_parser(const parser & p, const std::string & input, bool partial) {
    parser_context ctx(input, !partial);
    p.parse(ctx);
}

static void test_command_r7b_legacy_parser(const std::string & input, bool partial) {
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
}

struct bench_tool_call {
    std::string id;
    std::string name;
    nlohmann::ordered_json args;
};

// Simple tokenize function that splits by space
static std::vector<std::string> simple_tokenize(const std::string & input) {
    std::vector<std::string> result;
    std::string current;

    for (size_t i = 0; i < input.size(); i++) {
        if (input[i] == ' ') {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
            current += ' ';
        } else {
            current += input[i];
        }
    }

    if (!current.empty()) {
        result.push_back(current);
    }

    return result;
}

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

    auto run = [&](const std::function<void(const std::string &, bool)> & fn) {
        std::string input = std::accumulate(tokens.begin(), tokens.end(), std::string());

        std::chrono::microseconds duration(0);
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            fn(input, false);
            auto end = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        }
        return duration.count() / iterations;
    };

    auto parser = create_command_r7b_parser();

    auto duration_new = run([&](const std::string & input, bool partial) {
        test_command_r7b_parser(parser, input, partial);
    });

    auto duration_legacy = run([&](const std::string & input, bool partial) {
        try {
            test_command_r7b_legacy_parser(input, partial);
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
