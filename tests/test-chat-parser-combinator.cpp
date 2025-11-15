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
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.literal("hello");
        });

        common_chat_parse_context ctx;
        common_chat_parse_result result;

        ctx = common_chat_parse_context("hello");
        result = parser.parse(ctx);
        assert_equals(true, result.success());
    }
    {
        // Test char class
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.one("a-z");
        });

        common_chat_parse_context ctx;
        common_chat_parse_result result;

        ctx = common_chat_parse_context("a");
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        ctx = common_chat_parse_context("A");
        result = parser.parse(ctx);
        assert_equals(true, result.fail());

        parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.one("a-z-");
        });

        ctx = common_chat_parse_context("f");
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        ctx = common_chat_parse_context("-");
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        ctx = common_chat_parse_context("A");
        result = parser.parse(ctx);
        assert_equals(true, result.fail());
    }
    {
        // Test sequences and literals
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.literal("<think>") + p.literal("</think>");
        });

        // Partial matches
        auto ctx = common_chat_parse_context("<thi", false);
        auto result = parser.parse(ctx);
        assert_equals(true, result.need_more_input());

        ctx = common_chat_parse_context("<think>", false);
        result = parser.parse(ctx);
        assert_equals(true, result.need_more_input());

        ctx = common_chat_parse_context("<think></", false);
        result = parser.parse(ctx);
        assert_equals(true, result.need_more_input());

        // Full match
        ctx = common_chat_parse_context("<think></think>", true);
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        // No match, since it does not adhere to the grammar
        ctx = common_chat_parse_context("<think>I am parser", false);
        result = parser.parse(ctx);
        assert_equals(true, result.fail());
    }
    {
        // Test choices
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.literal("<think>") | p.literal("<reasoning>");
        });

        // Partial matches
        auto ctx = common_chat_parse_context("<thi", false);
        auto result = parser.parse(ctx);
        assert_equals(true, result.need_more_input());

        ctx = common_chat_parse_context("<reas", false);
        result = parser.parse(ctx);
        assert_equals(true, result.need_more_input());

        // Full match
        ctx = common_chat_parse_context("<think>", true);
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        ctx = common_chat_parse_context("<reasoning>", true);
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        // No match
        ctx = common_chat_parse_context("<thought>", true);
        result = parser.parse(ctx);
        assert_equals(true, result.fail());
    }
    {
        // Test zero_or_more
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.zero_or_more(p.literal("ab"));
        });

        // Partial matches
        auto ctx = common_chat_parse_context("a", false);
        auto result = parser.parse(ctx);
        assert_equals(true, result.need_more_input());

        ctx = common_chat_parse_context("aba", false);
        result = parser.parse(ctx);
        assert_equals(true, result.need_more_input());

        // Full match
        ctx = common_chat_parse_context("ab", true);
        result = parser.parse(ctx);
        assert_equals(true, result.success());
    }
    {
        // Test one_or_more
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.one_or_more(p.literal("ab"));
        });

        // Partial matches
        auto ctx = common_chat_parse_context("a", false);
        auto result = parser.parse(ctx);
        assert_equals(true, result.need_more_input());

        ctx = common_chat_parse_context("aba", false);
        result = parser.parse(ctx);
        assert_equals(true, result.need_more_input());

        // Full match
        ctx = common_chat_parse_context("ab", true);
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        // No match
        ctx = common_chat_parse_context("cd", true);
        result = parser.parse(ctx);
        assert_equals(true, result.fail());
    }
}

static void test_one() {
    {
        // Test common escape sequences
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.one("[\\n\\t\\\\]");
        });

        common_chat_parse_context ctx;
        common_chat_parse_result result;

        ctx = common_chat_parse_context("\n");
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        ctx = common_chat_parse_context("\t");
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        ctx = common_chat_parse_context("\\");
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        ctx = common_chat_parse_context(" ");
        result = parser.parse(ctx);
        assert_equals(true, result.fail());
    }
    {
        // Test escaped dash (literal dash, not a range)
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.one("[a\\-z]");
        });

        common_chat_parse_context ctx;
        common_chat_parse_result result;

        ctx = common_chat_parse_context("a");
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        ctx = common_chat_parse_context("-");
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        ctx = common_chat_parse_context("z");
        result = parser.parse(ctx);
        assert_equals(true, result.success());

        // Should NOT match 'b' since \- is a literal dash, not a range
        ctx = common_chat_parse_context("b");
        result = parser.parse(ctx);
        assert_equals(true, result.fail());
    }
}

static void test_recursive_references() {
    auto value_parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
        p.add_rule("number", p.one_or_more(p.one("0-9")));
        p.add_rule("list", p.sequence({
            p.literal("["),
            p.rule("value"),
            p.literal("]")
        }));
        return p.add_rule("value", p.rule("number") | p.rule("list"));
    });

    common_chat_parse_context ctx;
    common_chat_parse_result result;

    // Test simple number
    ctx = common_chat_parse_context("1", true);
    result = value_parser.parse(ctx);
    assert_equals(true, result.success());

    // Test simple list
    ctx = common_chat_parse_context("[1]", true);
    result = value_parser.parse(ctx);
    assert_equals(true, result.success());

    // Test nested list
    ctx = common_chat_parse_context("[[2]]", true);
    result = value_parser.parse(ctx);
    assert_equals(true, result.success());

    // Test deeply nested list
    ctx = common_chat_parse_context("[[[3]]]", true);
    result = value_parser.parse(ctx);
    assert_equals(true, result.success());

    // Test partial match
    ctx = common_chat_parse_context("[[", false);
    result = value_parser.parse(ctx);
    assert_equals(true, result.need_more_input());

    // Test no match
    ctx = common_chat_parse_context("[a]", true);
    result = value_parser.parse(ctx);
    assert_equals(true, result.fail());
}

static void test_optional() {
    // Test optional with a match
    auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
        return p.literal("hello") + p.optional(p.literal(" world"));
    });

    // Full match with optional part present
    auto ctx = common_chat_parse_context("hello world");
    auto result = parser.parse(ctx);
    assert_equals(true, result.success());
    assert_equals((size_t)11, result.end);

    // Full match with optional part absent
    ctx = common_chat_parse_context("hello", true);
    result = parser.parse(ctx);
    assert_equals(true, result.success());
    assert_equals((size_t)5, result.end);

    // Partial match - waiting for more input to determine if optional matches
    ctx = common_chat_parse_context("hello ", false);
    result = parser.parse(ctx);
    assert_equals(true, result.need_more_input());
}

static void test_json_parser() {
    auto json = build_combinator_parser([](common_chat_combinator_parser_builder & p) {
        return p.json();
    });

    {
        // Test parsing a simple JSON object
        std::string input = R"({"name": "test", "value": 42, "flag": true})";
        common_chat_parse_context ctx(input);

        auto result = json.parse(ctx);

        assert_equals(true, result.success());
        assert_equals(input.size(), result.end);
    }
    {
        // Test parsing a JSON array with mixed types
        std::string input = R"([1, "hello", true, null, 3.14])";
        common_chat_parse_context ctx(input);

        auto result = json.parse(ctx);

        assert_equals(true, result.success());
        assert_equals(input.size(), result.end);
    }
    {
        // Test parsing nested JSON with objects and arrays
        std::string input = R"({"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2, "metadata": {"version": "1.0", "tags": ["admin", "user"]}})";
        common_chat_parse_context ctx(input);

        auto result = json.parse(ctx);

        assert_equals(true, result.success());
        assert_equals(input.size(), result.end);
    }
    {
        // Test partial parsing - incomplete object
        std::string input = R"({"name": "test", "value": )";
        common_chat_parse_context ctx(input, false);

        auto result = json.parse(ctx);

        assert_equals(true, result.need_more_input());
    }
    {
        // Test partial parsing - incomplete array
        std::string input = R"([1, 2, 3, )";
        common_chat_parse_context ctx(input, false);

        auto result = json.parse(ctx);

        assert_equals(true, result.need_more_input());
    }
    {
        // Test partial parsing - incomplete nested structure
        std::string input = R"({"data": {"nested": )";
        common_chat_parse_context ctx(input, false);

        auto result = json.parse(ctx);

        assert_equals(true, result.need_more_input());
    }
}

static void test_actions() {
    {
        // Test simple action - append matched text to content
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            auto word = p.chars("[a-z]+");
            return p.action(word, [](const common_chat_parse_action & act) {
                act.env.result.content += std::string(act.match);
            });
        });

        common_chat_parse_semantics env;
        common_chat_parse_context ctx("hello", &env);
        auto result = parser.parse(ctx);

        assert_equals(true, result.success());
        assert_equals("hello", env.result.content);
    }
    {
        // Test multiple sequential actions - build a sentence
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            auto greeting = p.action(p.literal("hello"), [](const common_chat_parse_action & act) {
                act.env.result.content += std::string(act.match) + " ";
            });

            auto name = p.action(p.chars("[A-Z][a-z]+"), [](const common_chat_parse_action & act) {
                act.env.result.content += std::string(act.match);
                act.env.captures["name"] = std::string(act.match);
            });

            return greeting + p.literal(" ") + name;
        });

        common_chat_parse_semantics env;
        common_chat_parse_context ctx("hello Alice", &env);
        auto result = parser.parse(ctx);

        assert_equals(true, result.success());
        assert_equals("hello Alice", env.result.content);
        assert_equals("Alice", env.captures["name"]);
    }
    {
        // Test actions don't run when parse fails
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.action(p.literal("success"), [](const common_chat_parse_action & act) {
                act.env.result.content = "action_ran";
            });
        });

        common_chat_parse_semantics env;
        common_chat_parse_context ctx("failure", &env);
        auto result = parser.parse(ctx);

        assert_equals(true, result.fail());
        assert_equals("", env.result.content);  // Action should not have run
    }
    {
        // Test Actions work with partial parsing
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            auto content = p.action(p.until("<end>"), [](const common_chat_parse_action & act) {
                act.env.result.content += std::string(act.match);
            });
            return "<start>" << content << "<end>";
        });

        {
            common_chat_parse_semantics env;
            common_chat_parse_context ctx("<start>hello ", &env, false);
            auto result = parser.parse(ctx);

            assert_equals(true, result.need_more_input());
            assert_equals("hello ", env.result.content);
        }
        {
            common_chat_parse_semantics env;
            common_chat_parse_context ctx("<start>hello world", &env, false);
            auto result = parser.parse(ctx);

            assert_equals(true, result.need_more_input());
            assert_equals("hello world", env.result.content);
        }
        {
            common_chat_parse_semantics env;
            common_chat_parse_context ctx("<start>hello world<end>", &env, true);
            auto result = parser.parse(ctx);

            assert_equals(true, result.success());
            assert_equals("hello world", env.result.content);
        }
    }
}

static void test_sax_events() {
    {
        // Test basic event firing
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.add_rule("greeting", p.literal("hello"));
        });

        common_chat_parse_semantics env;
        std::vector<common_chat_parse_event> events;

        common_chat_parse_context ctx("hello", &env, [&](const common_chat_parse_event& evt, common_chat_parse_semantics&) {
            events.push_back(evt);
        });

        auto result = parser.parse(ctx);

        assert_equals(true, result.success());
        assert_equals((size_t)2, events.size());
        assert_equals(COMMON_CHAT_PARSE_EVENT_NODE_START, events[0].type);
        assert_equals("greeting", events[0].rule);
        assert_equals((size_t)0, events[0].start);
        assert_equals(0, events[0].depth);

        assert_equals(COMMON_CHAT_PARSE_EVENT_NODE_END, events[1].type);
        assert_equals("greeting", events[1].rule);
        assert_equals((size_t)0, events[1].start);
        assert_equals((size_t)5, events[1].end);
        assert_equals("hello", std::string(events[1].text));
        assert_equals(COMMON_CHAT_PARSE_RESULT_SUCCESS, events[1].status);
        assert_equals(0, events[1].depth);
    }
}

static void test_triggers() {
    {
        // Test basic trigger functionality with lazy mode
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            auto greeting = p.trigger(p.literal("hello"));
            auto farewell = p.trigger(p.literal("goodbye"));
            return greeting | farewell;
        });

        // Non-lazy mode: triggers are transparent
        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, false);
        });

        assert_equals(true, gbnf.find("root ::= \"hello\" | \"goodbye\"") != std::string::npos);

        // Lazy mode: triggers create synthetic rules
        auto gbnf_lazy = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, true);
        });

        // Should have trigger-1 and trigger-2 synthetic rules
        assert_equals(true, gbnf_lazy.find("trigger-1 ::= \"hello\"") != std::string::npos);
        assert_equals(true, gbnf_lazy.find("trigger-2 ::= \"goodbye\"") != std::string::npos);
        // Root should be alternation of triggers
        assert_equals(true, gbnf_lazy.find("root ::= trigger-1 | trigger-2") != std::string::npos);
    }
    {
        // Test that only reachable rules from triggers are generated
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            // Add multiple rules
            auto digit = p.add_rule("digit", p.one("[0-9]"));
            auto letter = p.add_rule("letter", p.one("[a-z]"));
            auto word = p.add_rule("word", p.one_or_more(letter));
            auto number = p.add_rule("number", p.one_or_more(digit));

            // Only trigger the word path, not the number path
            auto triggered_word = p.trigger(word);
            return triggered_word;
        });

        // Non-lazy mode: all rules generated
        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, false);
        });

        assert_equals(true, gbnf.find("digit ::=") != std::string::npos);
        assert_equals(true, gbnf.find("letter ::=") != std::string::npos);
        assert_equals(true, gbnf.find("word ::=") != std::string::npos);
        assert_equals(true, gbnf.find("number ::=") != std::string::npos);

        // Lazy mode: only rules reachable from trigger
        auto gbnf_lazy = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, true);
        });

        // Should have letter and word (reachable from trigger)
        assert_equals(true, gbnf_lazy.find("letter ::=") != std::string::npos);
        assert_equals(true, gbnf_lazy.find("word ::=") != std::string::npos);
        // Should NOT have digit and number (not reachable from trigger)
        assert_equals(true, gbnf_lazy.find("digit ::=") == std::string::npos);
        assert_equals(true, gbnf_lazy.find("number ::=") == std::string::npos);
        // Should have trigger-1 synthetic rule
        assert_equals(true, gbnf_lazy.find("trigger-1 ::=") != std::string::npos);
    }
}

static void test_gbnf_generation() {
    {
        // Test literal
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
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
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.one("[a-z]");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= [a-z]") != std::string::npos);
    }
    {
        // Test sequence
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.literal("hello") + p.literal(" ") + p.literal("world");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= \"hello\" \" \" \"world\"") != std::string::npos);
    }
    {
        // Test choice
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.literal("cat") | p.literal("dog");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= \"cat\" | \"dog\"") != std::string::npos);
    }
    {
        // Test one_or_more
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.one_or_more(p.one("[0-9]"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= [0-9]+") != std::string::npos);
    }
    {
        // Test zero_or_more
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.zero_or_more(p.one("[a-z]"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= [a-z]*") != std::string::npos);
    }
    {
        // Test optional
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.literal("hello") + p.optional(p.literal(" world"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= \"hello\" \" world\"?") != std::string::npos);
    }
    {
        // Test until
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
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
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.one_or_more(p.literal("a") | p.literal("b"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= (\"a\" | \"b\")+") != std::string::npos);
    }
    {
        // Test rule references
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
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
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
            return p.literal("hello\nworld\t!");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_equals(true, gbnf.find("root ::= \"hello\\nworld\\t!\"") != std::string::npos);
    }
    {
        // Test operator<< (whitespace insertion)
        auto parser = build_combinator_parser([](common_chat_combinator_parser_builder& p) {
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
    auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) {
        auto thinking = p.add_rule("raw-reasoning",
            "<think>" << p.add_rule("reasoning-content", p.until("</think>")) << "</think>");

        auto content = p.add_rule("content", p.until("<tool_call>"));

        auto arg_name = p.add_rule("arg-start", "<parameter=" + p.capture("arg-name", p.chars("[a-zA-Z0-9_]")) + ">");
        auto arg_end = p.add_rule("arg-end", "</parameter>" + p.peek(p.literal("<parameter=") | "</function>"));

        auto string_arg_content = p.add_rule("arg-string-content",
            p.until_one_of({"</parameter><parameter=", "</parameter></function>"}));

        auto string_arg = p.add_rule("arg-string", arg_name + string_arg_content + arg_end);

        auto json = p.json();

        auto json_arg = p.add_rule("arg-json", arg_name + p.add_rule("arg-json-content", json) + arg_end);

        auto function = p.add_rule("function",
                p.add_rule("function-start", "<function=" + p.capture("tool-name", p.chars("[a-zA-Z0-9_]")) + ">")
                + p.one_or_more(json_arg | string_arg)
                + "</function>");

        auto tool_call = p.trigger(p.add_rule("tool-call",
            "<tool_call>" + p.one_or_more(function) + "</tool_call>"));

        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call);
    });

    std::cout << "Grammar (lazy=false):\n";
    auto grammar = build_grammar([&](const common_grammar_builder & builder) {
        parser.build_grammar(builder);
    });
    std::cout << grammar << "\n";

    std::cout << "Grammar (lazy=true):\n";
    auto lazy_grammar = build_grammar([&](const common_grammar_builder & builder) {
        parser.build_grammar(builder, true);
    });
    std::cout << lazy_grammar << "\n";

    auto handler = [&](const common_chat_parse_event & ev, common_chat_parse_semantics & env) {
        if (ev.rule == "reasoning-content" && ev.ending()) {
            env.result.reasoning_content = ev.text;
        }

        if (ev.rule == "content" && ev.ending()) {
            env.result.content = ev.text;
        }

        if (ev.rule == "function-start" && ev.ending() && ev.success()) {
            env.result.tool_calls.emplace_back();
            auto & tc = env.result.tool_calls.back();
            tc.name = env.captures["tool-name"];
        }

        if (ev.rule == "arg-start" && ev.ending() && ev.success()) {
            auto & tc = env.result.tool_calls.back();
            auto name = env.captures["arg-name"];
            if (tc.arguments.empty()) {
                tc.arguments += "{";
            } else {
                tc.arguments += ", ";
            }
            tc.arguments += "\"" + name + "\": ";
        }

        if (ev.rule == "arg-string-content" && ev.ending() && ev.success()) {
            auto & tc = env.result.tool_calls.back();
            tc.arguments += "\"" + std::string(ev.text);
        }

        if (ev.rule == "arg-string" && ev.ending() && ev.success()) {
            auto & tc = env.result.tool_calls.back();
            tc.arguments += "\"";
        }

        if (ev.rule == "arg-json-content" && ev.ending() && (ev.success() || ev.need_more_input())) {
            auto & tc = env.result.tool_calls.back();
            tc.arguments += std::string(ev.text);
        }
    };

    std::string input =
        "<think>The user wants to find large log files that haven't been accessed recently. "
        "I should search for files with .log extension, filter by size (over 100MB), "
        "and check access time within the last 30 days. I'll need to use the search_files function.</think>"
        "Based on your requirements, I'll search for log files over 100MB that haven't been "
        "accessed in the last month. This will help identify candidates for cleanup or archival.\n\n"
        "<tool_call>"
        "<function=search_files>"
        "<parameter=path>/var/log</parameter>"
        "<parameter=pattern>*.log</parameter>"
        "<parameter=pattern2>searching for </parameter> blah</parameter>"
        "<parameter=min_size_mb>100</parameter>"
        "<parameter=max_depth>5</parameter>"
        "<parameter=include_hidden>false</parameter>"
        "<parameter=modified_days_ago>30</parameter>"
        "<parameter=case_sensitive>true</parameter>"
        "<parameter=sort_by>size</parameter>"
        "<parameter=filters>{\"exclude_patterns\": [\"*temp*\", \"*cache*\"], \"file_types\": [\"regular\"]}</parameter>"
        "</function>"
        "</tool_call>";

    std::vector<std::string> tokens = simple_tokenize(input);

    common_chat_msg prev;

    for (auto it = tokens.begin(); it != tokens.end(); it++) {
        std::string in = std::accumulate(tokens.begin(), it, std::string());

        common_chat_parse_semantics env;
        common_chat_parse_context ctx(in, &env, it == tokens.end() - 1);
        ctx.event_handler = handler;

        auto parse_result = parser.parse(ctx);
        assert_equals(false, parse_result.fail());

        // This shouldn't emit any runtime errors
        auto diffs = common_chat_msg_diff::compute_diffs(prev, env.result);
        prev = env.result;

#if 0
        std::cout << "=================================\n";
        std::cout << in << "\n\n";
        std::cout << "----\n";
        std::cout << "Reasoning: " << prev.reasoning_content << "\n";
        std::cout << "Content  : " << prev.content << "\n";
        if (!prev.tool_calls.empty()) {
            std::cout << "\n-- tool calls --\n";
            for (const auto & tc : prev.tool_calls) {
                std::cout << "  ID  : " << tc.id << "\n";
                std::cout << "  Name: " << tc.name << "\n";
                std::cout << "  Args: " << tc.arguments << "\n\n";
            }
        }
#endif
    }
}

static common_chat_combinator_parser create_command_r7b_parser() {
    auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) {
        auto thinking = p.add_rule("thinking",
            "<|START_THINKING|>" << p.add_rule("reasoning-content", p.until("<|END_THINKING|>")) << "<|END_THINKING|>");

        auto response = p.add_rule("response",
            "<|START_RESPONSE|>" << p.add_rule("content", p.until("<|END_RESPONSE|>")) << "<|END_RESPONSE|>");

        auto json = p.add_rule("json", p.json());

        auto tool_call_id = p.add_rule("tool-call-id",
            "\"tool_call_id\"" << (":" << p.add_rule("tool-call-id-value", "\"" + p.json_string() + "\"")));

        auto tool_call_name = p.add_rule("tool-name",
            "\"tool_name\"" << (":" << p.add_rule("tool-name-value", "\"" + p.json_string() + "\"")));

        auto tool_call_args = p.add_rule("tool-args",
            "\"parameters\"" << (":" << p.add_rule("tool-args-value", json)));

        auto tool_call_fields = p.add_rule("tool-call-fields", tool_call_id | tool_call_name | tool_call_args);

        auto tool_call = p.add_rule("tool-call",
            "{" << tool_call_fields << p.zero_or_more(p.literal(",") << tool_call_fields) << "}");

        auto tool_calls = p.add_rule("tool-calls",
            "<|START_ACTION|>"
            << ("[" << tool_call << p.zero_or_more(p.literal(",") << tool_call) << "]")
            << "<|END_ACTION|>");

        return p.optional(thinking) << (tool_calls | response);
    });

    auto grammar = build_grammar([&](const common_grammar_builder & builder) {
        parser.build_grammar(builder);
    });

    std::cout << "=== Grammar ===\n\n" << grammar << "\n\n";

    return parser;
}

static void test_command_r7b_parser(const common_chat_combinator_parser & p, const std::string & input, bool partial, bool print_results = false) {
    common_chat_parse_semantics env;
    common_chat_parse_context ctx(input, &env, !partial);

    ctx.event_handler = [&](const common_chat_parse_event & ev, common_chat_parse_semantics & env) {
        if (ev.rule == "reasoning-content" && ev.ending()) {
            env.result.reasoning_content = ev.text;
        }

        if (ev.rule == "content" && ev.ending()) {
            env.result.content = ev.text;
        }

        if (ev.rule == "tool-call" && ev.starting()) {
            env.result.tool_calls.emplace_back();
        }

        if (ev.rule == "tool-call-id-value" && ev.ending() && ev.success()) {
            auto & tc = env.result.tool_calls.back();
            tc.id = nlohmann::json::parse(ev.text).get<std::string>();
        }

        if (ev.rule == "tool-name-value" && ev.ending() && ev.success()) {
            auto & tc = env.result.tool_calls.back();
            tc.name = nlohmann::json::parse(ev.text).get<std::string>();
        }

        if (ev.rule == "tool-args-value" && ev.ending() && (ev.success() || ev.need_more_input())) {
            auto & tc = env.result.tool_calls.back();
            tc.arguments = ev.text;
        }
    };

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
    test_actions();
    test_sax_events();
    test_triggers();
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
