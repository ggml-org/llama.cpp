#include "json-schema-to-grammar.h"
#include "tests.h"
#include <nlohmann/json.hpp>

class test_complete_example : public compound_test {
  public:
    test_complete_example() : compound_test("test_complete_example") {
        /* Parser for a fictitious model that outputs:
         *
         *    <reasoning>
         *   ... reasoning content ...
         *    </reasoning>
         *   ... content ...
         *    <tool_call>
         *   <name>tool_name</name>
         *   <args>{ ... json args ... }</args>
         *    </tool_call>
         */
        auto parser = build_parser([](parser_builder & p) {
            auto reasoning =
                p.add_rule("reasoning", "<reasoning>" << p.append_reasoning(p.until("</reasoning>")) << "</reasoning>");

            auto content = p.add_rule("content", p.append_content(p.until("<tool_call>")));

            auto json = p.json();

            auto tool_call_name =
                p.add_rule("tool-call-name", "<name>" << p.capture_tool_call_name(p.until("</name>")) << "</name>");

            auto schema = nlohmann::json::parse(R"({"type": "object"})");

            auto tool_call_args = p.add_rule(
                "tool-call-args",
                "<args>" << p.capture_tool_call_args(p.schema(p.succeed(json), "get_weather", schema)) << "</args>");

            auto tool_call =
                p.add_rule("tool-call", "<tool_call>" << p.add_tool_call(tool_call_name << p.succeed(tool_call_args))
                                                      << "</tool_call>");

            return reasoning << p.optional(content) << p.optional(tool_call);
        });

        // Test complete input
        std::string input =
            std::string(R"(<reasoning>I need to call get_weather with city = New York</reasoning><tool_call><name>get_weather</name><args>{"city": "New York"}</args></tool_call>)");
        parser_environment env;
        parser_context     ctx(input, &env);

        auto result = parser.parse(ctx);

        // Test complete input with reasoning and tool call
        add_test(
            [env, input, result](test_harness h) {
                h.assert_equals("parse_success", true, result.is_success());
                h.assert_equals("parse_end", (size_t) input.size(), result.end);
                h.assert_equals("reasoning_content", std::string("I need to call get_weather with city = New York"),
                                env.result.reasoning_content);
                h.assert_equals("tool_calls_size", (size_t) 1, env.result.tool_calls.size());
                h.assert_equals("tool_call_id", std::string(""), env.result.tool_calls[0].id);
                h.assert_equals("tool_call_name", std::string("get_weather"), env.result.tool_calls[0].name);
                h.assert_equals("tool_call_args", std::string(R"({"city": "New York"})"),
                                env.result.tool_calls[0].arguments);
            },
            "complete_tool_call_parsing");

        // Test partial input
        add_test([parser](test_harness h) {
            std::string input = R"(<think>I need to call get_weather)";
            parser_environment env = parser_environment();
            parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

            auto result = parser.parse(ctx);

            h.assert_equals("needs_more_input", true, result.is_need_more_input());
            h.assert_equals("reasoning_content", std::string("I need to call get_weather"), env.result.reasoning_content);
        }, "partial_input");


        add_test([parser](test_harness h) {
            std::string input = R"(<think>I need to call </thi get_weather</th)";
            parser_environment env = parser_environment();
            parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

            auto result = parser.parse(ctx);

            h.assert_equals("needs_more_input", true, result.is_need_more_input());
        }, "input_incomplete");

        add_test([parser](test_harness h) {
            std::string input = R"(<think>I need to call get_weather</th)";
            parser_environment env = parser_environment();
            parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

            auto result = parser.parse(ctx);

            h.assert_equals("needs_more_input", true, result.is_need_more_input());
        }, "input_incomplete_2");
        add_test([parser](test_harness h) {
            std::string input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather)";
            parser_environment env = parser_environment();
            parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

            auto result = parser.parse(ctx);

            h.assert_equals("needs_more_input", true, result.is_need_more_input());
            h.assert_equals("reasoning_content", std::string("I need to call get_weather"), env.result.reasoning_content);
        }, "tool_call_incomplete");
        add_test([parser](test_harness h) {
            std::string input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather</na)";
            parser_environment env = parser_environment();
            parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

            auto result = parser.parse(ctx);

            h.assert_equals("needs_more_input", true, result.is_need_more_input());
            h.assert_equals("reasoning_content", std::string("I need to call get_weather"), env.result.reasoning_content);
        }, "tool_call_incomplete_2");
        add_test([parser](test_harness h) {
            std::string input = R"(<think>I need to call get_weather</think><tool_call><name>get_weather</name><args>{"cit)";
            parser_environment env = parser_environment();
            parser_context ctx = parser_context(input, &env, /* .is_input_complete = */ false);

            auto result = parser.parse(ctx);

            h.assert_equals("needs_more_input", true, result.is_need_more_input());
            h.assert_equals("reasoning_content", std::string("I need to call get_weather"), env.result.reasoning_content);
            h.assert_equals("tool_name", std::string("get_weather"), env.result.tool_calls[0].name);
            h.assert_equals("tool_incomplete_arg", std::string(R"({"cit)"), env.result.tool_calls[0].arguments);
        }, "tool_call_arg_incomplete");

        auto gbnf = build_grammar([parser](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        add_test([gbnf](test_harness h) {
            h.assert_equals("not_empty", false, gbnf.empty());
        }, "grammar_is_there");
    }

    // Provide a convenient way to run all tests
    void run_all_tests() {
        run_all();
        summary();
    }
};
