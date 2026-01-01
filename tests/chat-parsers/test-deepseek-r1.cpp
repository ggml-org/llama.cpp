#include "../test-chat.h"
#include "chat.h"

void test_deepseek_r1_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    common_chat_templates_inputs inputs_tools_builtin;
    inputs_tools_builtin.messages           = {message_user};
    inputs_tools_builtin.tools              = {python_tool};

    {
        // Templates with thinking support
        template_capabilities template_caps;
        template_caps.name = "DeepSeek R1";
        template_caps.jinja_path = "models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja";
        template_caps.legacy_format = COMMON_CHAT_FORMAT_DEEPSEEK_R1;
        template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
        template_caps.supports_thinking = ThinkingSupport::Yes;
        template_caps.think_open_tag = "<think>";
        template_caps.think_close_tag = "</think>";
        template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
        template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;
        template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::Yes;
        
        auto tmpls = read_templates(template_caps.jinja_path);
        // TODO(ochafik): re-enable once PEG parser handles this template correctly
        // run_template_test_suite(impl, template_caps, tmpls);

        // Test the exact scenario that fails in server test
        // (tool_choice=required, tool named "test", specific model output)
        if (impl == chat_parser_impl::EXPERIMENTAL) {
            common_chat_tool test_tool = {
                /* .name = */ "test",
                /* .description = */ "",
                /* .parameters = */ R"({
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "const": true}
                    },
                    "required": ["success"]
                })",
            };

            common_chat_templates_inputs inputs;
            inputs.messages = {message_user};
            inputs.tools = {test_tool};
            inputs.parallel_tool_calls = false;
            inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            inputs.experimental_new_parsers = true;

            auto params = common_chat_templates_apply(tmpls.get(), inputs);
            auto syntax = get_syntax(params);
            assert_equals(COMMON_CHAT_FORMAT_PEG_NATIVE, params.format);

            // Expected result
            common_chat_msg expected;
            expected.role = "assistant";
            expected.tool_calls = {{
                /* .name = */ "test",
                /* .arguments = */ R"({ "success" : true })",
                /* .id = */ "",
            }};

            // Try to parse the exact model output from server test (with leading space+newline)
            std::string model_output =
                " \n                    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test\n"
                "```json\n"
                "{ \"success\" : true } \n"
                "```<｜tool▁call▁end｜> ";

            auto msg = common_chat_parse(model_output, /* is_partial= */ false, syntax);
            assert_msg_equals(expected, msg);

            // Also test streaming
            test_parser_with_streaming(
                expected,
                model_output,
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, syntax); });
        }
    }
    {
        // Replacement DeepSeek R1 template. Makes the Distill Qwen 7B/32B models happy to call tools and all.
        template_capabilities template_caps;
        template_caps.name = "DeepSeek R1 (fixed)";
        template_caps.jinja_path = "models/templates/llama-cpp-deepseek-r1.jinja";
        template_caps.legacy_format = COMMON_CHAT_FORMAT_DEEPSEEK_R1;
        template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
        template_caps.supports_thinking = ThinkingSupport::Yes;
        template_caps.think_open_tag = "<think>";
        template_caps.think_close_tag = "</think>";
        template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
        template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;
        template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::Yes;
        template_caps.supports_disable_thinking = SupportsDisableThinking::No;
        template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
        template_caps.end_tokens = { "<｜end▁of▁sentence｜>" };

        auto tmpls = read_templates(template_caps.jinja_path);

        // run_template_test_suite(impl, template_caps, tmpls);

        {
            common_chat_templates_inputs inputs;
            inputs.messages = {message_user};
            inputs.tools = {special_function_tool};
            inputs.parallel_tool_calls = true;
            inputs.experimental_new_parsers = impl == chat_parser_impl::EXPERIMENTAL;

            auto params = common_chat_templates_apply(tmpls.get(), inputs);
            auto syntax = get_syntax(params);
            assert_equals(inputs.experimental_new_parsers ? COMMON_CHAT_FORMAT_PEG_NATIVE : COMMON_CHAT_FORMAT_DEEPSEEK_R1, params.format);

            test_parser_with_streaming(
                message_assist_call,
                "                    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n",
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, syntax); });
        }

        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1,                   common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1,                   common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_thoughts, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                    /* .reasoning_in_content = */ false,
                    /* .thinking_forced_open = */ true,
                }));

        assert_msg_equals(message_assist_call_thoughts_unparsed,
            common_chat_parse(
                "<think>I'm\nthinking</think>\n\n"
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));
        assert_msg_equals(message_assist_call,
            common_chat_parse(
                "<｜tool▁calls｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_DEEPSEEK_R1}));

        assert_msg_equals(message_assist_call_thoughts,
            common_chat_parse(
                "<think>I'm\nthinking</think>\n\n"
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_DEEPSEEK_R1,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));
        // TODO(ochafik): DeepSeek R1 has unicode chars in its tokens, PEG parsing infra escapes them incorrectly:
        // test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
        //         "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
        //         "```json\n"
        //         "{\"arg1\": 1}\n"
        //         "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>");
    }
}
