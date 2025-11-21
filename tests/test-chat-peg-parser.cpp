#include <string>
#include <iostream>
#include <numeric>

#include "chat-peg-parser.h"
#include "json-schema-to-grammar.h"
#include "peg-parser/test_harness.h"
#include "peg-parser/simple_tokenizer.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

static json create_tools();
static void test_example_qwen3_coder(testing & t);

int main(int argc, char *argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("qwen3 coder", test_example_qwen3_coder);

    //t.test("seed_oss", test_example_seed_oss);
    //t.test("minimax_m2", test_example_minimax_m2);
    //t.test("command7_parser_compare", test_command7_parser_compare);

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
};

void foreach_tool(const json & json_tools, const std::function<void(const tool_definition &)> & fn) {
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
                    (arg_def.type == "string" ?
                        p.tool_arg_string_value(p.until_one_of({"</parameter><parameter=", "</parameter></function>"})) :
                        p.tool_arg_json_value(p.schema(p.json(), "tool-" + def.name + "-arg-" + def.name + "-schema", arg_def.schema))) +
                    p.tool_arg_close("</parameter>" + p.peek(p.literal("<parameter=") | p.literal("</function>")))
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

        auto tool_call = p.rule("tool-call", "<tool_call>" + p.choice(tool_parsers) + "</tool_call>", true);
        return content + p.zero_or_more(p.space() + tool_call) + p.end();
    });

    auto grammar = build_grammar([&](const common_grammar_builder & builder) {
        parser.build_grammar(builder);
    });

    t.log("Grammar:\n" + grammar);

    t.test("incremental parsing", [&](testing &t) {
        std::string input =
            "Let me search the knowledge base for cat pictures."
            "<tool_call>"
            "<function=search_knowledge_base>"
            "<parameter=query>cat pictures</parameter>"
            "<parameter=category>general</parameter>"
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
            auto extractor = common_chat_peg_constructed_builder::extractor(msg);
            ctx.ast_arena.visit(result, extractor.visitor());

            //t.log("Input: " + input);
            t.log("===========================================");
            t.log("Iteration " + std::to_string(input.size()));
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
