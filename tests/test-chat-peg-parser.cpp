#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "peg-parser.h"
#include "testing.h"
#include "peg-parser/simple-tokenize.h"

#include <iostream>
#include <numeric>
#include <string>

#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

static json create_tools();
static void test_example_native(testing & t);
static void test_example_qwen3_coder(testing & t);
static void test_example_qwen3_non_coder(testing & t);
static void test_command7_parser_compare(testing & t);
static void test_prefix_tool_names(testing & t);
static void test_tagged_peg_parser(testing & t);

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    t.test("native", test_example_native);
    t.test("qwen3 coder", test_example_qwen3_coder);
    t.test("qwen3 non-coder", test_example_qwen3_non_coder);
    t.test("comparison", test_command7_parser_compare);
    t.test("prefix tool names", test_prefix_tool_names);
    t.test("tagged peg parser", test_tagged_peg_parser);

    return t.summary();
}

static json create_tools() {
    json tools = json::array();

    json tool_weather = {
        { "type",     "function" },
        { "function",
         {
              { "name", "get_current_weather" },
              { "description", "Get the current weather in a given location" },
              { "parameters",
                {
                    { "type", "object" },
                    { "properties",
                      { { "location",
                          { { "type", "string" }, { "description", "The city and state, e.g. San Francisco, CA" } } },
                        { "unit",
                          { { "type", "string" },
                            { "enum", { "celsius", "fahrenheit" } },
                            { "description",
                              "The temperature unit to use. Infer this from the users location." } } } } },
                    { "required", { "location", "unit" } },
                } },
          }                      }
    };
    tools.push_back(tool_weather);

    json tool_forecast = {
        { "type",     "function" },
        { "function",
         {
              { "name", "get_forecast" },
              { "description", "Get the weather forecast for a given location" },
              { "parameters",
                {
                    { "type", "object" },
                    { "properties",
                      { { "location",
                          { { "type", "string" }, { "description", "The city and state, e.g. San Francisco, CA" } } },
                        { "unit",
                          { { "type", "string" },
                            { "enum", { "celsius", "fahrenheit" } },
                            { "description", "The temperature unit to use. Infer this from the users location." } } },
                        { "days",
                          { { "type", "integer" },
                            { "description", "Number of days to forecast (1-10)" },
                            { "minimum", 1 },
                            { "maximum", 10 } } } } },
                    { "required", { "location", "unit" } },
                } },
          }                      }
    };
    tools.push_back(tool_forecast);

    json tool_search = {
        { "type",     "function" },
        { "function",
         { { "name", "search_knowledge_base" },
            { "description", "Search the internal technical documentation knowledge base." },
            { "parameters",
              { { "type", "object" },
                { "properties",
                  { { "query", { { "type", "string" }, { "description", "The search query string." } } },
                    { "max_results",
                      { { "type", "integer" },
                        { "description", "The maximum number of results to return." },
                        { "default", 5 } } },
                    { "category",
                      { { "type", "string" },
                        { "enum", { "api", "troubleshooting", "billing", "general" } },
                        { "description", "Filter search by specific category." } } } } },
                { "required", { "query", "category" } },
                { "additionalProperties", false } } },
            { "strict", true } } }
    };
    tools.push_back(tool_search);

    return tools;
}

struct tool_argument {
    std::string name;
    std::string type;
    bool        is_required;
    json        schema;
};

struct tool_definition {
    std::string                name;
    std::vector<tool_argument> arguments;
    json                       schema;
};

// Test fictitious model output that emits arguments as JSON.
static void test_example_native(testing & t) {
    struct test_case {
        // Parameters
        std::string             name;
        json                    tools;
        common_chat_tool_choice tool_choice;
        common_reasoning_format reasoning_format;
        json                    json_schema;
        bool                    parallel_tool_calls;
        bool                    thinking_forced_open;
        std::string             input;

        // Expect
        std::string                        expect_reasoning;
        std::string                        expect_content;
        std::vector<common_chat_tool_call> expect_tool_calls;
    };

    auto build_parser = [](const test_case & tc) {
        return build_chat_peg_parser([&](common_chat_peg_builder & p) {
            auto reasoning_in_content = (tc.reasoning_format == COMMON_REASONING_FORMAT_NONE);
            auto reasoning            = p.eps();
            if (tc.thinking_forced_open) {
                // If thinking is forced open, expect a closing tag
                reasoning = p.reasoning(p.until("</think>")) + "</think>" + p.space();
            } else {
                // Otherwise, optionally accept thinking wrapped in tags
                reasoning = p.optional("<think>" + p.reasoning(p.until("</think>")) + "</think>" + p.space());
            }

            // tool calling parser
            if (tc.tools.is_array() && !tc.tools.empty()) {
                auto tool_call =
                    p.standard_json_tools("<tool_call>[", "]</tool_call>", tc.tools, tc.parallel_tool_calls,
                                          tc.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED);

                return p.sequence({ (reasoning_in_content ? p.eps() : reasoning), p.content(p.until("<tool_call>")),
                                    p.optional(p.space() + tool_call), p.space(), p.end() });
            }

            // response_format parser
            if (tc.json_schema.is_object() && !tc.json_schema.empty()) {
                return p.sequence({ (reasoning_in_content ? p.eps() : reasoning),
                                    p.content(p.schema(p.json(), "response-output", tc.json_schema)), p.space(),
                                    p.end() });
            }

            // Content-only parser
            return p.sequence({ (reasoning_in_content ? p.eps() : reasoning), p.content(p.rest()), p.end() });
        });
    };

    std::vector<test_case> test_cases = std::vector<test_case>{
        {
         /* .name =                 */ "content with thinking_forced_open = false",
         /* .tools =                */ {},
         /* .tool_choice =          */ COMMON_CHAT_TOOL_CHOICE_NONE,
         /* .reasoning_format =     */ COMMON_REASONING_FORMAT_AUTO,
         /* .json_schema =          */ {},
         /* .parallel_tool_calls =  */ false,
         /* .thinking_forced_open = */ false,
         /* .input =                */ ("<think>The user said hello, I must say hello back</think>\nHello"),
         /* .expect_reasoning =     */ "The user said hello, I must say hello back",
         /* .expect_content =       */ "Hello",
         /* .expect_tool_calls =    */ {},
         },
        {
         /* .name =                 */ "content with thinking_forced_open = false and no reasoning",
         /* .tools =                */ {},
         /* .tool_choice =          */ COMMON_CHAT_TOOL_CHOICE_NONE,
         /* .reasoning_format =     */ COMMON_REASONING_FORMAT_AUTO,
         /* .json_schema =          */ {},
         /* .parallel_tool_calls =  */ false,
         /* .thinking_forced_open = */ false,
         /* .input =                */ ("Hello"),
         /* .expect_reasoning =     */ "",
         /* .expect_content =       */ "Hello",
         /* .expect_tool_calls =    */ {},
         },
        {
         /* .name =                 */ "content with thinking_forced_open = false and reasoning_format = none",
         /* .tools =                */ {},
         /* .tool_choice =          */ COMMON_CHAT_TOOL_CHOICE_NONE,
         /* .reasoning_format =     */ COMMON_REASONING_FORMAT_NONE,
         /* .json_schema =          */ {},
         /* .parallel_tool_calls =  */ false,
         /* .thinking_forced_open = */ true,
         /* .input =                */ ("<think>The user said hello, I must say hello back</think>\nHello"),
         /* .expect_reasoning =     */ "",
         /* .expect_content =       */ "<think>The user said hello, I must say hello back</think>\nHello",
         /* .expect_tool_calls =    */ {},
         },
        {
         /* .name =                 */ "content with thinking_forced_open = true",
         /* .tools =                */ {},
         /* .tool_choice =          */ COMMON_CHAT_TOOL_CHOICE_NONE,
         /* .reasoning_format =     */ COMMON_REASONING_FORMAT_AUTO,
         /* .json_schema =          */ {},
         /* .parallel_tool_calls =  */ false,
         /* .thinking_forced_open = */ true,
         /* .input =                */ ("The user said hello, I must say hello back</think>\nHello"),
         /* .expect_reasoning =     */ "The user said hello, I must say hello back",
         /* .expect_content =       */ "Hello",
         /* .expect_tool_calls =    */ {},
         },
        {
         /* .name =                 */ "content with thinking_forced_open = true and reasoning_format = none",
         /* .tools =                */ {},
         /* .tool_choice =          */ COMMON_CHAT_TOOL_CHOICE_NONE,
         /* .reasoning_format =     */ COMMON_REASONING_FORMAT_NONE,
         /* .json_schema =          */ {},
         /* .parallel_tool_calls =  */ false,
         /* .thinking_forced_open = */ true,
         /* .input =                */ ("The user said hello, I must say hello back</think>\nHello"),
         /* .expect_reasoning =     */ "",
         /* .expect_content =       */ "The user said hello, I must say hello back</think>\nHello",
         /* .expect_tool_calls =    */ {},
         },
        {
         /* .name =                 */ "tools with tool_choice = auto and no parallel_tool_calls",
         /* .tools =                */ create_tools(),
         /* .tool_choice =          */ COMMON_CHAT_TOOL_CHOICE_AUTO,
         /* .reasoning_format =     */ COMMON_REASONING_FORMAT_AUTO,
         /* .json_schema =          */ {},
         /* .parallel_tool_calls =  */ false,
         /* .thinking_forced_open = */ true,
         /* .input =                */
            ("I must get the weather in New York</think>\n"
             "<tool_call>["
             R"({"name": "get_current_weather", "arguments": {"location": "New York City, NY", "unit": "fahrenheit"}})"
             "]</tool_call>"),
         /* .expect_reasoning =     */ "I must get the weather in New York",
         /* .expect_content =       */ "",
         /* .expect_tool_calls =    */
            { {
                /* .name =      */ "get_current_weather",
                /* .arguments = */ R"({"location": "New York City, NY", "unit": "fahrenheit"})",
                /* .id =        */ "",
            } },
         },
        {
         /* .name =                 */ "tools with tool_choice = auto and parallel_tool_calls",
         /* .tools =                */ create_tools(),
         /* .tool_choice =          */ COMMON_CHAT_TOOL_CHOICE_AUTO,
         /* .reasoning_format =     */ COMMON_REASONING_FORMAT_AUTO,
         /* .json_schema =          */ {},
         /* .parallel_tool_calls =  */ true,
         /* .thinking_forced_open = */ true,
         /* .input =                */
            ("I must get the weather in New York and San Francisco and a 3 day forecast of each.</think>\nLet me "
             "search that for you."
             "<tool_call>["
             R"({"name": "get_current_weather", "arguments": {"location": "New York City, NY", "unit": "fahrenheit"}})"
             ", "
             R"({"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}})"
             ", "
             R"({"name": "get_forecast", "arguments": {"location": "New York City, NY", "unit": "fahrenheit", "days": 3}})"
             ", "
             R"({"name": "get_forecast", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit", "days": 3}})"
             "]</tool_call>"),
         /* .expect_reasoning =     */
            "I must get the weather in New York and San Francisco and a 3 day forecast of each.",                                                                     /* .expect_content =       */ "Let me search that for you.",
         /* .expect_tool_calls =    */
            { {
                  /* .name =      */ "get_current_weather",
                  /* .arguments = */ R"({"location": "New York City, NY", "unit": "fahrenheit"})",
                  /* .id =        */ "",
              },
              {
                  /* .name =      */ "get_current_weather",
                  /* .arguments = */ R"({"location": "San Francisco, CA", "unit": "fahrenheit"})",
                  /* .id =        */ "",
              },
              {
                  /* .name =      */ "get_forecast",
                  /* .arguments = */ R"({"location": "New York City, NY", "unit": "fahrenheit", "days": 3})",
                  /* .id =        */ "",
              },
              {
                  /* .name =      */ "get_forecast",
                  /* .arguments = */ R"({"location": "San Francisco, CA", "unit": "fahrenheit", "days": 3})",
                  /* .id =        */ "",
              } },
         },
        {
         /* .name =                 */ "response_format with thinking_forced_open = true",
         /* .tools =                */ {},
         /* .tool_choice =          */ COMMON_CHAT_TOOL_CHOICE_NONE,
         /* .reasoning_format =     */ COMMON_REASONING_FORMAT_AUTO,
         /* .json_schema =          */
            { { "type", "object" },
              { "properties",
                { { "invoice_number", { { "type", "string" } } },
                  { "amount", { { "type", "number" } } },
                  { "due_date", { { "type", "string" } } } } },
              { "required", { "invoice_number", "amount", "due_date" } } },
         /* .parallel_tool_calls =  */ false,
         /* .thinking_forced_open = */ true,
         /* .input =                */
            ("I must produce the invoice in the requested format</think>\n"
             R"({"invoice_number": "INV-2025-001", "amount": 1250.50, "due_date": "2025-12-31"})"),
         /* .expect_reasoning =     */ "I must produce the invoice in the requested format",
         /* .expect_content =       */
            R"({"invoice_number": "INV-2025-001", "amount": 1250.50, "due_date": "2025-12-31"})", /* .expect_tool_calls =    */ {},
         },
    };

    for (const auto & tc : test_cases) {
        t.test(tc.name, [&](testing & t) {
            auto parser  = build_parser(tc);
            auto lazy    = !tc.tools.empty() && tc.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            auto grammar = build_grammar([&](const common_grammar_builder & builder) {
                for (const auto & def : tc.tools) {
                    auto function   = def.at("function");
                    auto parameters = function.at("parameters");
                    builder.resolve_refs(parameters);
                };
                parser.build_grammar(builder, lazy);
            });

            t.log("Grammar:");
            for (const auto & line : string_split(grammar, "\n")) {
                t.log(line);
            }

            common_peg_parse_context ctx(tc.input, false);
            auto                     result = parser.parse(ctx);

            t.assert_true("success", result.success());

            common_chat_msg msg;
            auto            mapper = common_chat_peg_mapper(msg);
            mapper.from_ast(ctx.ast, result);

            t.assert_equal("content equal", tc.expect_content, msg.content);
            t.assert_equal("reasoning equal", tc.expect_reasoning, msg.reasoning_content);
            t.assert_equal("number of tool calls", tc.expect_tool_calls.size(), msg.tool_calls.size());
            for (auto i = 0u; i < std::min(tc.expect_tool_calls.size(), msg.tool_calls.size()); i++) {
                t.assert_equal("tool name", tc.expect_tool_calls[i].name, msg.tool_calls[i].name);
                t.assert_equal("tool args", tc.expect_tool_calls[i].arguments, msg.tool_calls[i].arguments);
            }
        });
    }
}

static void test_example_qwen3_coder(testing & t) {
    auto tools  = create_tools();
    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto content = p.rule("content", p.content(p.until("<tool_call>")));

        std::vector<common_peg_parser> tool_parsers;
        for (const auto & def : tools) {
            auto        function   = def.at("function");
            std::string name       = function.at("name");
            auto        parameters = function.at("parameters");
            auto        properties = parameters.at("properties");

            std::set<std::string> required_properties;
            if (function.contains("required")) {
                function.at("required").get_to(required_properties);
            }

            std::vector<common_peg_parser> arg_parsers;
            for (const auto & [param_name, param_schema] : properties.items()) {
                bool is_required = required_properties.find(param_name) != required_properties.end();
                auto type        = param_schema.value("type", "object");

                auto arg = p.tool_arg(
                    p.sequence({ p.tool_arg_open("<parameter=" + p.tool_arg_name(p.literal(param_name)) + ">"),
                                 (type == "string" ?
                                      p.tool_arg_string_value(p.schema(
                                          p.until_one_of({ "</parameter>\n<parameter=", "</parameter>\n</function>" }),
                                          "tool-" + name + "-arg-" + param_name + "-schema", param_schema, true)) :
                                      p.tool_arg_json_value(p.schema(
                                          p.json(), "tool-" + name + "-arg-" + param_name + "-schema", param_schema))),
                                 p.tool_arg_close("</parameter>\n" +
                                                  p.peek(p.literal("<parameter=") | p.literal("</function>"))) }));

                arg_parsers.push_back(is_required ? p.rule("tool-" + name + "-arg-" + param_name, arg) :
                                                    p.optional(p.rule("tool-" + name + "-arg-" + param_name, arg)));
            }

            tool_parsers.push_back(p.rule("tool-" + name, p.tool_open("<function=" + p.tool_name(p.literal(name)) + ">")
                                                              << p.sequence(arg_parsers)
                                                              << p.tool_close(p.literal("</function>"))));
        };

        auto tool_call = p.trigger_rule("tool-call", "<tool_call>" << p.choice(tool_parsers) << "</tool_call>");

        return content + p.zero_or_more(p.space() + tool_call) + p.end();
    });

    auto grammar = build_grammar([&](const common_grammar_builder & builder) {
        for (const auto & def : tools) {
            auto function   = def.at("function");
            auto parameters = function.at("parameters");
            builder.resolve_refs(parameters);
        };
        parser.build_grammar(builder);
    });

    t.log("Grammar:");
    for (const auto & line : string_split(grammar, "\n")) {
        t.log(line);
    }

    t.test("incremental parsing", [&](testing & t) {
        std::string input =
            "Let me search the knowledge base for cat pictures."
            "<tool_call>\n"
            "<function=search_knowledge_base>\n"
            "<parameter=query>cat pictures</parameter>\n"
            "<parameter=category>general</parameter>\n"
            "</function>\n"
            "</tool_call>";

        std::vector<std::string> tokens = simple_tokenize(input);

        common_chat_msg prev;
        for (auto it = tokens.begin(); it != tokens.end(); it++) {
            std::string in = std::accumulate(tokens.begin(), it + 1, std::string());

            common_peg_parse_context ctx(in, it + 1 < tokens.end());

            auto result = parser.parse(ctx);
            if (!t.assert_equal("not fail", false, result.fail())) {
                t.log(in.substr(0, result.end) + "[failed->]" + in.substr(result.end));
            }

            common_chat_msg msg;
            auto            mapper = common_chat_peg_mapper(msg);
            mapper.from_ast(ctx.ast, result);

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
            } catch (const std::exception & e) {
                t.log(in.substr(0, result.end) + "[failed->]" + in.substr(result.end));
                t.assert_true(std::string("failed with ") + e.what(), false);
            }

            prev = msg;
        }
    });
}

static void test_example_qwen3_non_coder(testing & t) {
    auto tools  = create_tools();
    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // tool calling parser using standard JSON format
        auto tool_call = p.standard_json_tools("<tool_call>", "</tool_call>", tools, true, false);

        return p.sequence({ p.content(p.until("<tool_call>")), p.optional(p.space() + tool_call), p.end() });
    });

    auto grammar = build_grammar([&](const common_grammar_builder & builder) {
        for (const auto & def : tools) {
            auto function   = def.at("function");
            auto parameters = function.at("parameters");
            builder.resolve_refs(parameters);
        };
        parser.build_grammar(builder);
    });

    t.log("Grammar:");
    for (const auto & line : string_split(grammar, "\n")) {
        t.log(line);
    }

    t.test("tool call parsing", [&](testing & t) {
        std::string input =
            "I need to get the weather.\n"
            "<tool_call>"
            "{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"New York City, NY\", \"unit\": "
            "\"fahrenheit\"}}"
            "</tool_call>";

        common_peg_parse_context ctx(input, false);
        auto                     result = parser.parse(ctx);

        t.assert_true("success", result.success());

        common_chat_msg msg;
        auto            mapper = common_chat_peg_mapper(msg);
        mapper.from_ast(ctx.ast, result);

        t.assert_equal("content", "I need to get the weather.\n", msg.content);
        t.assert_equal("reasoning", "", msg.reasoning_content);
        t.assert_equal("tool calls count", 1u, msg.tool_calls.size());
        if (!msg.tool_calls.empty()) {
            t.assert_equal("tool name", "get_current_weather", msg.tool_calls[0].name);
            t.assert_equal("tool args", "{\"location\": \"New York City, NY\", \"unit\": \"fahrenheit\"}",
                           msg.tool_calls[0].arguments);
        }
    });

    t.test("incremental parsing", [&](testing & t) {
        std::string input =
            "I need to get the weather.\n"
            "<tool_call>"
            "{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"New York City, NY\", \"unit\": "
            "\"fahrenheit\"}}"
            "</tool_call>";

        std::vector<std::string> tokens = simple_tokenize(input);

        common_chat_msg prev;
        for (auto it = tokens.begin(); it != tokens.end(); it++) {
            std::string in = std::accumulate(tokens.begin(), it + 1, std::string());

            common_peg_parse_context ctx(in, it + 1 < tokens.end());

            auto result = parser.parse(ctx);
            if (!t.assert_equal("not fail", false, result.fail())) {
                t.log(in.substr(0, result.end) + "[failed->]" + in.substr(result.end));
            }

            common_chat_msg msg;
            auto            mapper = common_chat_peg_mapper(msg);
            mapper.from_ast(ctx.ast, result);

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
            } catch (const std::exception & e) {
                t.log(in.substr(0, result.end) + "[failed->]" + in.substr(result.end));
                t.assert_true(std::string("failed with ") + e.what(), false);
            }

            prev = msg;
        }
    });
}

void test_command7_parser_compare(testing & t) {
    auto parser = build_chat_peg_parser([](common_chat_peg_builder & p) {
        auto thinking =
            p.reasoning_block("<|START_THINKING|>" << p.reasoning(p.until("<|END_THINKING|>")) << "<|END_THINKING|>");

        auto response = "<|START_RESPONSE|>" << p.content(p.until("<|END_RESPONSE|>")) << "<|END_RESPONSE|>";

        auto tool_call_id = p.atomic("\"tool_call_id\"" << (":" << ("\"" + p.tool_id(p.json_string_content()) + "\"")));
        auto tool_call_name =
            p.atomic("\"tool_name\"" << (":" << ("\"" + p.tool_name(p.json_string_content()) + "\"")));
        auto tool_call_args = "\"parameters\"" << (":" << p.tool_args(p.json()));

        auto tool_call_fields = p.rule("tool-call-fields", tool_call_id | tool_call_name | tool_call_args);
        auto tool_call =
            p.rule("tool-call", p.tool(p.tool_open(p.literal("{"))
                                       << tool_call_fields << p.zero_or_more(p.literal(",") << tool_call_fields)
                                       << p.tool_close(p.literal("}"))));

        auto tool_calls = p.rule(
            "tool-calls", "<|START_ACTION|>" << ("[" << tool_call << p.zero_or_more(p.literal(",") << tool_call) << "]")
                                             << "<|END_ACTION|>");

        return p.optional(thinking) << (tool_calls | response) + p.end();
    });

    auto test_current = [&](const common_peg_arena & p, const std::string & input, bool is_partial,
                            bool print_results) {
        common_peg_parse_context ctx(input, is_partial);
        auto                     result = p.parse(ctx);

        common_chat_msg msg;
        auto            mapper = common_chat_peg_mapper(msg);
        mapper.from_ast(ctx.ast, result);

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

    std::string reasoning =
        "To plan an effective trip to Japan that includes both historical sites and modern attractions within a "
        "budget of $4000 for a two-week stay, we need to:\n\n"
        "1. Identify key historical sites and modern attractions in Japan.\n"
        "2. Find affordable accommodation options that provide a balance between comfort and cost.\n"
        "3. Determine the best modes of transportation for getting around Japan.\n"
        "4. Create a day-by-day itinerary that ensures the user gets to see a variety of attractions without "
        "overspending.\n"
        "5. Provide a detailed cost breakdown that includes accommodation, transportation, meals, and entry fees "
        "to attractions.";

    std::vector<std::tuple<std::string, std::string, nlohmann::json>> tool_calls = {
        { "call_0", "plan_trip", nlohmann::json::parse(R"({
            "destination": "Japan",
            "duration": 14,
            "budget": 4000,
            "interests": ["historical sites", "modern attractions"],
            "accommodation_preferences": "affordable",
            "transportation_preferences": "efficient",
            "meal_preferences": "local cuisine"
        })") }
    };

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
            auto tc_json            = nlohmann::json::object();
            tc_json["tool_call_id"] = std::get<0>(tc);
            tc_json["tool_name"]    = std::get<1>(tc);
            tc_json["parameters"]   = std::get<2>(tc);
            json.push_back(tc_json);
        }

        auto tokenized = simple_tokenize(json.dump(-1, ' ', true));
        tokens.insert(tokens.end(), tokenized.begin(), tokenized.end());

        tokens.emplace_back("<|END_ACTION|>");
    }

    std::string input = std::accumulate(tokens.begin(), tokens.end(), std::string());

    t.test("current_parse", [&](testing & /* t */) { test_current(parser, input, false, false); });
    t.bench("current_parse_benchmark complete", [&]() { test_current(parser, input, false, false); }, 100);
    t.bench(
        "current_parse_benchmark incremental",
        [&]() {
            std::string in;
            for (auto i = 0u; i < tokens.size(); i++) {
                in += tokens[i];
                test_current(parser, in, i + 1 < tokens.size(), false);
            }
        },
        20);
}

// Test that tool names that are proper prefixes of other tool names don't cause
// premature matching during incremental parsing.
// For example, "special_function" should not match when parsing "special_function_with_opt".
static void test_prefix_tool_names(testing & t) {
    // Create tools where one name is a proper prefix of another
    json tools = json::array();

    json tool_short = {
        { "type", "function" },
        { "function",
          {
              { "name", "special_function" },
              { "description", "A special function" },
              { "parameters",
                {
                    { "type", "object" },
                    { "properties",
                      {
                          { "arg1", { { "type", "integer" } } },
                      } },
                    { "required", { "arg1" } },
                } },
          } }
    };
    tools.push_back(tool_short);

    json tool_long = {
        { "type", "function" },
        { "function",
          {
              { "name", "special_function_with_opt" },
              { "description", "A special function with optional params" },
              { "parameters",
                {
                    { "type", "object" },
                    { "properties",
                      {
                          { "arg1", { { "type", "integer" } } },
                          { "arg2", { { "type", "integer" } } },
                      } },
                    { "required", { "arg1" } },
                } },
          } }
    };
    tools.push_back(tool_long);

    // Use standard_constructed_tools which had the prefix matching bug
    std::map<std::string, std::string> markers = {
        { "tool_call_start_marker", "<tool_call>" },
        { "tool_call_end_marker", "</tool_call>" },
        { "function_opener", "<function=" },
        { "function_closer", "</function>" },
        { "function_name_suffix", ">" },
        { "parameter_key_prefix", "<param=" },
        { "parameter_key_suffix", ">" },
        { "parameter_closer", "</param>" },
    };

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto content   = p.rule("content", p.content(p.until("<tool_call>")));
        auto tool_call = p.standard_constructed_tools(markers, tools, false, false);
        return content + p.zero_or_more(p.space() + tool_call) + p.end();
    });

    // Test parsing the long tool name - this should NOT trigger the short tool name
    t.test("parse long tool name", [&](testing & t) {
        std::string input =
            "Let me call the function."
            "<tool_call>"
            "<function=special_function_with_opt>"
            "<param=arg1>42</param>"
            "</function>"
            "</tool_call>";

        common_peg_parse_context ctx(input, false);
        auto                     result = parser.parse(ctx);

        t.assert_true("success", result.success());

        common_chat_msg msg;
        auto            mapper = common_chat_peg_mapper(msg);
        mapper.from_ast(ctx.ast, result);

        t.assert_equal("content", "Let me call the function.", msg.content);
        t.assert_equal("tool calls count", 1u, msg.tool_calls.size());
        if (!msg.tool_calls.empty()) {
            t.assert_equal("tool name", "special_function_with_opt", msg.tool_calls[0].name);
        }
    });

    // Test incremental parsing - the key test case
    // This ensures that when incrementally parsing "special_function_with_opt",
    // we don't prematurely emit "special_function" as a tool call
    t.test("incremental parse long tool name", [&](testing & t) {
        std::string input =
            "Let me call the function."
            "<tool_call>"
            "<function=special_function_with_opt>"
            "<param=arg1>42</param>"
            "</function>"
            "</tool_call>";

        std::vector<std::string> tokens = simple_tokenize(input);

        common_chat_msg prev;
        for (auto it = tokens.begin(); it != tokens.end(); it++) {
            std::string in = std::accumulate(tokens.begin(), it + 1, std::string());

            common_peg_parse_context ctx(in, it + 1 < tokens.end());
            auto                     result = parser.parse(ctx);

            if (!t.assert_equal("not fail", false, result.fail())) {
                t.log(in.substr(0, result.end) + "[failed->]" + in.substr(result.end));
                return;
            }

            common_chat_msg msg;
            auto            mapper = common_chat_peg_mapper(msg);
            mapper.from_ast(ctx.ast, result);

            // The critical check: during incremental parsing, we should never
            // see "special_function" as the tool name when parsing "special_function_with_opt"
            for (const auto & tc : msg.tool_calls) {
                if (!t.assert_equal("tool name should not be short prefix", false,
                                    tc.name == "special_function")) {
                    t.log("Premature tool name match at input: " + in);
                    return;
                }
            }

            try {
                auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
            } catch (const std::exception & e) {
                t.log(in.substr(0, result.end) + "[failed->]" + in.substr(result.end));
                t.assert_true(std::string("diff failed with ") + e.what(), false);
                return;
            }

            prev = msg;
        }

        // Final check: the complete parse should have the correct tool name
        t.assert_equal("final tool calls count", 1u, prev.tool_calls.size());
        if (!prev.tool_calls.empty()) {
            t.assert_equal("final tool name", "special_function_with_opt", prev.tool_calls[0].name);
        }
    });

    // Test parsing the short tool name still works
    t.test("parse short tool name", [&](testing & t) {
        std::string input =
            "Let me call the function."
            "<tool_call>"
            "<function=special_function>"
            "<param=arg1>42</param>"
            "</function>"
            "</tool_call>";

        common_peg_parse_context ctx(input, false);
        auto                     result = parser.parse(ctx);

        t.assert_true("success", result.success());

        common_chat_msg msg;
        auto            mapper = common_chat_peg_mapper(msg);
        mapper.from_ast(ctx.ast, result);

        t.assert_equal("content", "Let me call the function.", msg.content);
        t.assert_equal("tool calls count", 1u, msg.tool_calls.size());
        if (!msg.tool_calls.empty()) {
            t.assert_equal("tool name", "special_function", msg.tool_calls[0].name);
        }
    });
}

static void test_tagged_peg_parser(testing & t) {
    t.test("basic tag extraction", [&](testing & t) {
        auto parser = build_tagged_peg_parser([](common_peg_parser_builder & p) {
            return p.tag("greeting", p.until(" ")) + " " + p.tag("name", p.rest()) + p.end();
        });

        auto result = parser.parse_and_extract("Hello World");
        t.assert_true("success", result.result.success());
        t.assert_equal("greeting tag", "Hello", result.tags.at("greeting"));
        t.assert_equal("name tag", "World", result.tags.at("name"));
    });

    t.test("duplicate tags overwrite", [&](testing & t) {
        auto parser = build_tagged_peg_parser([](common_peg_parser_builder & p) {
            return p.tag("item", p.until(",")) + "," + p.tag("item", p.rest()) + p.end();
        });

        auto result = parser.parse_and_extract("first,second");
        t.assert_true("success", result.result.success());
        t.assert_equal("item tag", "second", result.tags.at("item"));
    });

    t.test("no tags extracted", [&](testing & t) {
        auto parser = build_tagged_peg_parser([](common_peg_parser_builder & p) {
            return p.rest() + p.end();
        });

        auto result = parser.parse_and_extract("Hello");
        t.assert_true("success", result.result.success());
        t.assert_equal("empty tags", 0u, result.tags.size());
    });

    t.test("structured extraction", [&](testing & t) {
        auto parser = build_tagged_peg_parser([](common_peg_parser_builder & p) {
            auto header = p.tag("header", p.until("\n"));
            auto body = p.tag("body", p.rest());
            return header + "\n" + body + p.end();
        });

        auto result = parser.parse_and_extract("Title\nBody content here");
        t.assert_true("success", result.result.success());
        t.assert_equal("header", "Title", result.tags.at("header"));
        t.assert_equal("body", "Body content here", result.tags.at("body"));
    });

    t.test("partial parse", [&](testing & t) {
        auto parser = build_tagged_peg_parser([](common_peg_parser_builder & p) {
            return p.tag("prefix", p.until(":")) + ":" + p.tag("value", p.rest()) + p.end();
        });

        auto result = parser.parse_and_extract("key:val", true);
        t.assert_true("not fail", !result.result.fail());
        t.assert_equal("prefix tag", "key", result.tags.at("prefix"));
        t.assert_equal("value tag", "val", result.tags.at("value"));
    });

    t.test("find in the middle", [&](testing & t) {
        auto parser = build_tagged_peg_parser([](common_peg_parser_builder & p) {
            return p.choice({ p.literal("{"), p.literal(":") }) + p.space() + p.literal("\"") + p.atomic(p.literal("fun_name"));
        });

        std::string tpl = "This is a very long jinja template string. We have tools. We will try to call them now: <tool_call>{ \"fun_name\" : { \"arg\" : 1 }</tool_call>";
        auto result = parser.parse_anywhere_and_extract(tpl);
        t.assert_true("success", result.result.success());
    });

    t.test("fail find in the middle", [&](testing & t) {
        auto parser = build_tagged_peg_parser([](common_peg_parser_builder & p) {
            return p.choice({ p.literal("{"), p.literal(":") }) + p.space() + p.literal("\"") + p.atomic(p.literal("fun_name"));
        });

        std::string tpl = "This is a very long jinja template string. We have tools. We will try to call them now: <tool_call><fun=fun_name><arg name=arg>1</arg></tool_call>";
        auto result = parser.parse_anywhere_and_extract(tpl);
        t.assert_true("failure", result.result.fail());
    });

    t.test("find function tag with name", [&](testing &t) {
        std::string haystack = "\n<tool_call>\n<function=foofoo>\n<parameter=first>\nXXXX\n</parameter>\n<parameter=second>\nYYYY\n</parameter>\n</function>\n</tool_call>\n";
        auto parser = build_tagged_peg_parser([](common_peg_parser_builder & p) {
            std::string needle = "foofoo";
            return p.tag("fun_marker", p.choice({
            p.tag("fun_pre", p.literal("<") + p.until_one_of({ ">", needle })) + p.literal(needle) +
                p.tag("fun_post", p.negate(p.space() + p.literal("<")) + p.until(">") + p.literal(">")) + p.space(),
            p.tag("fun_pre", p.literal("[") + p.until_one_of({ "]", needle })) + p.literal(needle) +
                p.tag("fun_post", p.negate(p.space() + p.literal("[") + p.until("]") + p.literal("]")) + p.space()) }));
        });
        auto result = parser.parse_anywhere_and_extract(haystack);
        t.assert_true("success", result.result.success());
        t.assert_equal("fun_pre should be '<function='", "<function=", result.tags["fun_pre"]);
        t.assert_equal("fun_post should be '>'", ">", result.tags["fun_post"]);
    });
}
