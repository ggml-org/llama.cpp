#include <string>
#include <iostream>
#include <numeric>

#include "chat-parser.h"
#include "chat-peg-parser.h"
#include "json-schema-to-grammar.h"
#include "peg-parser/test_harness.h"
#include "peg-parser/simple_tokenizer.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

static json create_tools();
static void test_example_qwen3_coder(testing & t);
static void test_command7_parser_compare(testing & t);

int main(int argc, char *argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("qwen3 coder", test_example_qwen3_coder);
    t.test("comparison", test_command7_parser_compare);

    return t.summary();
}

static json create_tools() {
    json tools = json::array();

    json tool_weather = {
        {"type", "function"},
        {"function", {
            {"name", "get_current_weather"},
            {"description", "Get the current weather in a given location"},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"location", {
                        {"type", "string"},
                        {"description", "The city and state, e.g. San Francisco, CA"}
                    }},
                    {"unit", {
                        {"type", "string"},
                        {"enum", {"celsius", "fahrenheit"}},
                        {"description", "The temperature unit to use. Infer this from the users location."}
                    }}
                }},
                {"required", {"location", "unit"}},
            }},
        }}
    };
    tools.push_back(tool_weather);

    json tool_mortgage = {
        {"type", "function"},
        {"function", {
            {"name", "calculate_mortgage"},
            {"description", "Calculate the monthly mortgage payment based on principal, rate, and term."},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"principal", {
                        {"type", "number"},
                        {"description", "The loan amount in dollars."}
                    }},
                    {"interest_rate", {
                        {"type", "number"},
                        {"description", "Annual interest rate in percentage (e.g., 5.5 for 5.5%)."}
                    }},
                    {"years", {
                        {"type", "integer"},
                        {"description", "The loan term in years."}
                    }}
                }},
                {"required", {"principal", "interest_rate", "years"}},
                {"additionalProperties", false}
            }},
            {"strict", true}
        }}
    };
    tools.push_back(tool_mortgage);

    json tool_search = {
        {"type", "function"},
        {"function", {
            {"name", "search_knowledge_base"},
            {"description", "Search the internal technical documentation knowledge base."},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"query", {
                        {"type", "string"},
                        {"description", "The search query string."}
                    }},
                    {"max_results", {
                        {"type", "integer"},
                        {"description", "The maximum number of results to return."},
                        {"default", 5}
                    }},
                    {"category", {
                        {"type", "string"},
                        {"enum", {"api", "troubleshooting", "billing", "general"}},
                        {"description", "Filter search by specific category."}
                    }}
                }},
                {"required", {"query", "category"}},
                {"additionalProperties", false}
            }},
            {"strict", true}
        }}
    };
    tools.push_back(tool_search);

    return tools;
}

struct tool_argument {
    std::string name;
    std::string type;
    bool is_required;
    json schema;
};

struct tool_definition {
    std::string name;
    std::vector<tool_argument> arguments;
    json schema;
};

static void foreach_tool(const json & json_tools, const std::function<void(tool_definition &)> & fn) {
    if (!json_tools.is_array()) {
        return;
    }

    for (const auto & item : json_tools) {
        if (!item.contains("function") || !item["function"].is_object()) {
            continue;
        }

        const auto & func_node = item["function"];

        tool_definition tool;
        tool.name = func_node.value("name", "unknown_tool");
        tool.schema = func_node;

        if (func_node.contains("parameters") && func_node["parameters"].is_object()) {
            const auto& params_node = func_node["parameters"];

            std::vector<std::string> required_list;
            if (params_node.contains("required") && params_node["required"].is_array()) {
                required_list = params_node["required"].get<std::vector<std::string>>();
            }

            if (params_node.contains("properties") && params_node["properties"].is_object()) {
                for (const auto & [key, value] : params_node["properties"].items()) {
                    tool_argument arg;

                    arg.name = key;
                    arg.type = value.value("type", "string");
                    arg.schema = value;

                    auto it = std::find(required_list.begin(), required_list.end(), arg.name);
                    arg.is_required = (it != required_list.end());

                    tool.arguments.push_back(arg);
                }
            }
        }

        fn(tool);
    }
}

static void test_example_qwen3_coder(testing & t) {
    auto tools = create_tools();
    auto parser = build_chat_peg_constructed_parser([&](common_chat_peg_constructed_builder & p) {
        auto content = p.rule("content", p.content(p.until("<tool_call>")));

        std::vector<common_peg_parser> tool_parsers;
        foreach_tool(tools, [&](const tool_definition & def) {
            t.log(def.name);

            std::vector<common_peg_parser> arg_parsers;
            for (const auto & arg_def : def.arguments) {
                auto arg = p.tool_arg(
                    p.tool_arg_open("<parameter=" + p.tool_arg_name(p.literal(arg_def.name)) + ">") +
                    p.tool_arg_json_value(p.schema(p.json(), "tool-" + def.name + "-arg-" + def.name + "-schema", arg_def.schema)) +
                    p.tool_arg_close(p.literal("</parameter>"))
                );

                arg_parsers.push_back(arg_def.is_required ?
                    p.rule("tool-" + def.name + "-arg-" + arg_def.name, arg) :
                    p.optional(p.rule("tool-" + def.name + "-arg-" + arg_def.name, arg)));
            }

            tool_parsers.push_back(p.rule("tool-" + def.name,
                p.tool_open("<function=" + p.tool_name(p.literal(def.name)) + ">") +
                p.sequence(arg_parsers) +
                p.tool_close(p.literal("</function>"))
            ));
        });

        auto tool_call = p.trigger_rule("tool-call", "<tool_call>" + p.choice(tool_parsers) + "</tool_call>");
        return content + p.zero_or_more(p.space() + tool_call) + p.end();
    });

    auto grammar = build_grammar([&](const common_grammar_builder & builder) {
        foreach_tool(tools, [&](tool_definition & def) {
            builder.resolve_refs(def.schema);
        });
        parser.build_grammar(builder);
    });

    t.log("Grammar:\n" + grammar);

    t.test("incremental parsing", [&](testing &t) {
        std::string input =
            "Let me search the knowledge base for cat pictures."
            "<tool_call>"
            "<function=search_knowledge_base>"
            "<parameter=query>\"cat pictures\"</parameter>"
            "<parameter=category>\"general\"</parameter>"
            "</function>"
            "</tool_call>";

        std::vector<std::string> tokens = simple_tokenize(input);

        common_chat_msg prev;
        for (auto it = tokens.begin(); it != tokens.end(); it++) {
            std::string in = std::accumulate(tokens.begin(), it + 1, std::string());

            common_peg_parse_context ctx(in, it == tokens.end() - 1);

            auto result = parser.parse(ctx);
            if (!t.assert_equal("not fail", false, result.fail())) {
                t.log(in.substr(0, result.end) + "[failed->]" + in.substr(result.end));
            }

            common_chat_msg msg;
            auto extractor = common_chat_peg_constructed_extractor(msg);
            ctx.ast_arena.visit(result, extractor.visitor());

            //t.log("Input: " + input);
            t.log("===========================================");
            t.log("Iteration " + std::to_string(in.size()));
            t.log("Reasoning: " + msg.reasoning_content);
            t.log("Content  : " + msg.content);
            for (const auto & tc : msg.tool_calls) {
                t.log("Tool name: " + tc.name);
                t.log("Tool args: " + tc.arguments);
            }

            try {
                // This shouldn't emit any runtime errors
                auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
            } catch(const std::exception & e) {
                t.log(in.substr(0, result.end) + "[failed->]" + in.substr(result.end));
                t.assert_true(std::string("failed with ") + e.what(), false);
            }

            prev = msg;
        }
    });
}

void test_command7_parser_compare(testing & t) {
    auto parser = build_chat_peg_native_parser([](common_chat_peg_native_builder & p) {
        auto thinking = p.reasoning_block(
            "<|START_THINKING|>" << p.reasoning(p.until("<|END_THINKING|>")) << "<|END_THINKING|>");

        auto response = "<|START_RESPONSE|>" << p.content(p.until("<|END_RESPONSE|>")) << "<|END_RESPONSE|>";

        auto tool_call_id = p.atomic("\"tool_call_id\"" << (":" << "\"" + p.tool_id(p.json_string_content()) + "\""));
        auto tool_call_name = p.atomic("\"tool_name\"" << (":" << "\"" + p.tool_name(p.json_string_content()) + "\""));
        auto tool_call_args = "\"parameters\"" << (":" << p.tool_args(p.json()));

        auto tool_call_fields = p.rule("tool-call-fields", tool_call_id | tool_call_name | tool_call_args);
        auto tool_call = p.rule("tool-call", p.tool(
            p.tool_open(p.literal("{"))
            << tool_call_fields
            << p.zero_or_more( p.literal(",") << tool_call_fields)
            << p.tool_close(p.literal("}"))
        ));

        auto tool_calls = p.rule("tool-calls",
            "<|START_ACTION|>"
            << ("[" << tool_call << p.zero_or_more(p.literal(",") << tool_call) << "]")
            << "<|END_ACTION|>");

        return p.optional(thinking) << (tool_calls | response) + p.end();
    });

    auto test_current = [&](const common_peg_arena & p, const std::string & input, bool need_more_input, bool print_results) {
        common_peg_parse_context ctx(input, !need_more_input);
        auto result = p.parse(ctx);

        common_chat_msg msg;
        auto extractor = common_chat_peg_native_extractor(msg);
        ctx.ast_arena.visit(result, extractor.visitor());

        if (print_results) {
            std::cout << "== Parsed (new) ==\n";
            std::cout << "=== Reasoning ===\n";
            std::cout << msg.reasoning_content << "\n";
            std::cout << "\n\n=== Content ===\n";
            std::cout << msg.content << "\n";
            std::cout << "\n\n=== Tool Calls ===\n";
            for (const auto & tc : msg.tool_calls) {
                std::cout << "id: " << tc.id << "\n";
                std::cout << "name: " << tc.name << "\n";
                std::cout << "args: " << tc.arguments << "\n";
            }
        }
    };

    auto test_legacy = [&](const std::string & input, bool need_more_input, bool print_results) {
        // Original common_chat_combinator_parser taken from chat.cpp
        common_chat_msg_parser builder(
            input,
            /* .is_partial = */ need_more_input,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GENERIC,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ false,
            }
        );

        builder.try_parse_reasoning("<|START_THINKING|>", "<|END_THINKING|>");

        static const common_regex start_action_regex("<\\|START_ACTION\\|>");
        static const common_regex end_action_regex("<\\|END_ACTION\\|>");
        static const common_regex start_response_regex("<\\|START_RESPONSE\\|>");
        static const common_regex end_response_regex("<\\|END_RESPONSE\\|>");

        if (auto res = builder.try_find_regex(start_action_regex)) {
            // If we didn't extract thoughts, prelude includes them.
            auto tool_calls = builder.consume_json_with_dumped_args({ { "parameters" } });
            for (const auto & tool_call : tool_calls.value) {
                std::string name      = tool_call.contains("tool_name") ? tool_call.at("tool_name") : "";
                std::string id        = tool_call.contains("tool_call_id") ? tool_call.at("tool_call_id") : "";
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
    };

    std::string reasoning = "To plan an effective trip to Japan that includes both historical sites and modern attractions within a "
            "budget of $4000 for a two-week stay, we need to:\n\n"
            "1. Identify key historical sites and modern attractions in Japan.\n"
            "2. Find affordable accommodation options that provide a balance between comfort and cost.\n"
            "3. Determine the best modes of transportation for getting around Japan.\n"
            "4. Create a day-by-day itinerary that ensures the user gets to see a variety of attractions without "
            "overspending.\n"
            "5. Provide a detailed cost breakdown that includes accommodation, transportation, meals, and entry fees "
            "to attractions.";

    std::vector<std::tuple<std::string, std::string, nlohmann::json>> tool_calls = {{
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

    std::vector<std::string> tokens;

    // Build tokens
    if (!reasoning.empty()) {
        auto tokenized = simple_tokenize(reasoning);
        tokens.emplace_back("<|START_THINKING|>");
        tokens.insert(tokens.end(), tokenized.begin(), tokenized.end());
        tokens.emplace_back("<|END_THINKING|>");
    }

    if (!tool_calls.empty()) {
        tokens.emplace_back("<|START_ACTION|>");

        auto json = nlohmann::json::array();
        for (const auto & tc : tool_calls) {
            auto tc_json = nlohmann::json::object();
            tc_json["tool_call_id"] = std::get<0>(tc);
            tc_json["tool_name"] = std::get<1>(tc);
            tc_json["parameters"] = std::get<2>(tc);
            json.push_back(tc_json);
        }

        auto tokenized = simple_tokenize(json.dump(-1, ' ', true));
        tokens.insert(tokens.end(), tokenized.begin(), tokenized.end());

        tokens.emplace_back("<|END_ACTION|>");
    }

    std::string input = std::accumulate(tokens.begin(), tokens.end(), std::string());

    // Run tests
    t.test("legacy_parse", [&](testing & /* t */) {
        test_legacy(input, false, false);
    });

    t.test("current_parse", [&](testing & /* t */) {
        test_current(parser, input, false, false);
    });

    // Run benchmarks
    t.bench("legacy_parse_benchmark", [&]() {
        std::string in;
        for (auto i = 0u; i < tokens.size(); i++) {
            in += tokens[i];

            try {
                test_legacy(in, i + 1 < tokens.size(), false);
            } catch (common_chat_msg_partial_exception & e) {
                // Do nothing, this is expected
            }
        }
    }, 20);

    t.bench("current_parse_benchmark", [&]() {
        std::string in;
        for (auto i = 0u; i < tokens.size(); i++) {
            in += tokens[i];
            test_current(parser, input, i + 1 < tokens.size(), false);
        }
    }, 20);
}
