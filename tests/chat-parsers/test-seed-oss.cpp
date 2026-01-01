#include "../test-chat.h"

void test_seed_oss_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    template_capabilities template_caps;
    template_caps.name = "Seed OSS";
    template_caps.jinja_path = "models/templates/ByteDance-Seed-OSS.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_SEED_OSS;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;
    template_caps.supports_thinking = ThinkingSupport::Yes;
    template_caps.think_open_tag = "<seed:think>";
    template_caps.think_close_tag = "</seed:think>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::Yes;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::Yes;
    template_caps.end_tokens = { "<seed:eos>" };

    // Seed-OSS format tests
    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    assert_equals(COMMON_CHAT_FORMAT_SEED_OSS, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_SEED_OSS, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);

    // Create inputs with reasoning enabled (includes process_data for multi-param tests)
    common_chat_templates_inputs inputs_tools_reasoning;
    inputs_tools_reasoning.messages = {message_user};
    inputs_tools_reasoning.tools = {special_function_tool, process_data_tool};
    inputs_tools_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs_tools_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);

    // Get syntax with parser for tool call tests (with reasoning)
    auto params = common_chat_templates_apply(tmpls.get(), inputs_tools_reasoning);
    common_chat_syntax syntax = get_syntax(params, COMMON_REASONING_FORMAT_DEEPSEEK);

    // Syntax with reasoning for content-only tests
    common_chat_syntax syntax_reasoning;
    syntax_reasoning.format = params.format;
    syntax_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    if (!params.parser.empty()) {
        syntax_reasoning.parser.load(params.parser);
    }

    // PEG parser-specific tests (only run with experimental parser)
    // Legacy format-based parser has different whitespace handling for these cases
    if (impl == chat_parser_impl::EXPERIMENTAL) {
        // Test simple reasoning content
        assert_msg_equals(
            simple_assist_msg("Hello, world!", "I'm thinking about the answer"),
            common_chat_parse(
                "<seed:think>I'm thinking about the answer</seed:think>Hello, world!",
                /* is_partial= */ false,
                syntax_reasoning));

        // Test budget reflection tags
        common_chat_msg msg_budget_reflect;
        msg_budget_reflect.role = "assistant";
        msg_budget_reflect.content = "<seed:cot_budget_reflect>Token usage: 45/1000\nI should continue thinking to find the best solution.</seed:cot_budget_reflect>I need to calculate this step by step.";
        msg_budget_reflect.reasoning_content = "Token usage: 45/1000\nI should continue thinking to find the best solution.";
        assert_msg_equals(
            msg_budget_reflect,
            common_chat_parse(
                "<seed:think>Token usage: 45/1000\nI should continue thinking to find the best solution.</seed:think>"
                "<seed:cot_budget_reflect>Token usage: 45/1000\nI should continue thinking to find the best solution.</seed:cot_budget_reflect>"
                "I need to calculate this step by step.",
                /* is_partial= */ false,
                syntax_reasoning));

        // Test tool calls with Seed-OSS format (using special_function from inputs_tools)
        common_chat_msg msg_tool_call;
        msg_tool_call.role = "assistant";
        msg_tool_call.tool_calls.push_back({"special_function", "{\"arg1\":42}", ""});
        assert_msg_equals(
            msg_tool_call,
            common_chat_parse(
                "<seed:tool_call>\n"
                "<function=special_function>\n"
                "<parameter=arg1>\n42\n</parameter>\n"
                "</function>\n"
                "</seed:tool_call>",
                /* is_partial= */ false,
                syntax));

        // Test multiple parameters in tool call
        common_chat_msg msg_multi_param;
        msg_multi_param.role = "assistant";
        msg_multi_param.tool_calls.push_back({"process_data", "{\"input\":\"test\",\"format\":\"json\"}", ""});
        assert_msg_equals(
            msg_multi_param,
            common_chat_parse(
                "<seed:tool_call>\n"
                "<function=process_data>\n"
                "<parameter=input>\ntest\n</parameter>\n"
                "<parameter=format>\njson\n</parameter>\n"
                "</function>\n"
                "</seed:tool_call>",
                /* is_partial= */ false,
                syntax));

        // Test reasoning + tool call combination
        common_chat_msg msg_reasoning_tool;
        msg_reasoning_tool.role = "assistant";
        msg_reasoning_tool.content = "";
        msg_reasoning_tool.reasoning_content = "I need to call the special function";
        msg_reasoning_tool.tool_calls.push_back({"special_function", "{\"arg1\":42}", ""});
        assert_msg_equals(
            msg_reasoning_tool,
            common_chat_parse(
                "<seed:think>I need to call the special function</seed:think>"
                "<seed:tool_call>\n"
                "<function=special_function>\n"
                "<parameter=arg1>\n42\n</parameter>\n"
                "</function>\n"
                "</seed:tool_call>",
                /* is_partial= */ false,
                syntax_reasoning));

        // Test deltas: the number of tool calls in partial parses should never decrease
        std::string tool_msg = "<seed:tool_call>\n"
            "<function=special_function>\n"
            "<parameter=arg1>\n42\n</parameter>\n"
            "</function>";
        std::size_t previousToolCalls = 0;
        for (std::size_t i = std::string("<seed:tool_call>").length(); i < tool_msg.length() - 1; i++) {
            auto partial = tool_msg.substr(0, i);
            auto partial_res = common_chat_parse(partial, true, syntax);
            if (partial_res.tool_calls.size() < previousToolCalls) {
                throw std::runtime_error("Tool call size decreased on partial: " + partial + " from " + std::to_string(previousToolCalls) + " to " + std::to_string(partial_res.tool_calls.size()));
            }
            previousToolCalls = partial_res.tool_calls.size();
        }

        // Test partial parsing for incomplete string parameter - captures partial value
        assert_msg_equals(
            simple_assist_msg("", "", "process_data", "{\"input\":\"test"),
            common_chat_parse(
                "<seed:tool_call>\n"
                "<function=process_data>\n"
                "<parameter=input>\ntest",
                /* is_partial= */ true,
                syntax));

        auto make_invalid_delta = [&](const std::function<void(std::string &)> & mutate) {
            test_templates(
                impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
                /* expected_delta = */ "", /* expect_grammar_triggered = */ true,
                /* test_grammar_if_triggered = */ true,
                COMMON_REASONING_FORMAT_NONE,
                /* ignore_whitespace_differences = */ false,
                /* expect_parse_failure = */ true,
                mutate);
        };

        // Wrong function name should fail parsing once tool-call trigger fires
        make_invalid_delta([](std::string & delta) {
            const std::string needle = "function=special_function";
            auto pos = delta.find(needle);
            GGML_ASSERT(pos != std::string::npos);
            delta.replace(pos, needle.size(), "function=unknown_function");
        });

        // Wrong argument type should also fail (string instead of integer)
        make_invalid_delta([](std::string & delta) {
            const std::string param_open = "<parameter=arg1>";
            const std::string param_close = "</parameter>";
            auto start = delta.find(param_open);
            GGML_ASSERT(start != std::string::npos);
            auto end = delta.find(param_close, start);
            GGML_ASSERT(end != std::string::npos);
            end += param_close.size();
            const std::string replacement = "<parameter=arg1>\n\"not-a-number\"\n</parameter>";
            delta.replace(start, end - start, replacement);
        });

        // Test incomplete reasoning tag
        assert_msg_equals(
            simple_assist_msg("", "I was thinking"),
            common_chat_parse(
                "<seed:think>I was thinking",
                /* is_partial= */ true,
                syntax_reasoning));

        // Test content without reasoning
        assert_msg_equals(
            simple_assist_msg("This is a simple response without reasoning."),
            common_chat_parse(
                "This is a simple response without reasoning.",
                /* is_partial= */ false,
                syntax));
    } // end PEG parser-specific tests
}
