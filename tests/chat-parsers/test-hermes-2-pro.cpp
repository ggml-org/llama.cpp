#include "../test-chat.h"

void test_hermes_2_pro_parser(chat_parser_impl impl)
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
        auto tmpls = read_templates("models/templates/Qwen-QwQ-32B.jinja");

        assert_equals(COMMON_CHAT_FORMAT_HERMES_2_PRO, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_HERMES_2_PRO, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
    }
    
    auto tmpls = read_templates("models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja");
    template_capabilities template_caps;
    template_caps.name = "Hermes 2 Pro";
    template_caps.jinja_path = "models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_HERMES_2_PRO;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::No;
    template_caps.think_open_tag = "<think>";
    template_caps.think_close_tag = "</think>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::No;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
    template_caps.end_tokens = { "<|im_end|>" };

    assert_equals(COMMON_CHAT_FORMAT_HERMES_2_PRO, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_HERMES_2_PRO, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
    assert_equals(
        COMMON_CHAT_FORMAT_HERMES_2_PRO,
        common_chat_templates_apply(
            read_templates("models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja").get(),
            inputs_tools)
            .format);
    assert_equals(
        COMMON_CHAT_FORMAT_HERMES_2_PRO,
        common_chat_templates_apply(
            read_templates("models/templates/Qwen-Qwen2.5-7B-Instruct.jinja").get(),
            inputs_tools)
            .format);

    // Test parsing
    assert_msg_equals(
        simple_assist_msg("", "", "python", ""),
        common_chat_parse(
            "```json\n"
            "<function_call> { \"name\" : \"python\"",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        simple_assist_msg("Let's call something\n"),
        common_chat_parse(
            "Let's call something\n"
            "<tool_call>{\"name\"",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(
        simple_assist_msg("Let's call something\n"),
        common_chat_parse(
            "Let's call something\n"
            "<tool_call>{\"name",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(message_assist_call_thoughts,
        common_chat_parse(
            // QwQ-32B's template adds a trailing <think> if add_generation_prompt
            "I'm\nthinking</think>\n"
            "<tool_call>{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}</tool_call>",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ true,
            }));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "<tool_call>\n"
            "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "</tool_call>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(message_assist_call_content,
        common_chat_parse(
            "Hello, world!\nWhat's up?<tool_call>\n"
            "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "</tool_call>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "<function=special_function>{\"arg1\": 1}</function>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "<function name=\"special_function\">\n"
            "{\"arg1\": 1}\n"
            "</function>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "<tool>\n"
            "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "</tool>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "<tools>\n"
            "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "</tools>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "<response>\n"
            "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "</response>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "```xml\n"
            "<response>\n"
            "    {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "</response>\n"
            "```",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "```xml\n"
            "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "```",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "```\n"
            "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "```",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "```\n"
            "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "```",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "```json\n"
            "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "```",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "```json\n"
            "\n"
            "                    <function_call> {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}} \n"
            "                    </function_call> \n"
            "``` ",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "<json>\n"
            "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "</json>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "<xml>\n"
            "  {\n"
            "    \"name\": \"special_function\", \"arguments\": {\"arg1\": 1}\n"
            "  }\n"
            "</xml>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "<JSON>\n"
            "  {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "</JSON>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(
        message_assist_call,
        common_chat_parse(
            "{\n  \"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));

    // Test multiple tool calls
    common_chat_msg message_assist_multiple_calls;
    message_assist_multiple_calls.role = "assistant";
    message_assist_multiple_calls.content = "";
    message_assist_multiple_calls.tool_calls.push_back({"special_function", "{\"arg1\": 1}", ""});
    message_assist_multiple_calls.tool_calls.push_back({"python", "{\"code\":\"print('hello')\"}", ""});

    assert_msg_equals(
        message_assist_multiple_calls,
        common_chat_parse(
            "<tool_call>\n"
            "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "{\"name\": \"python\", \"arguments\": {\"code\":\"print('hello')\"}}\n"
            "</tool_call>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));

    assert_msg_equals(
        message_assist_multiple_calls,
        common_chat_parse(
            "<function=special_function>{\"arg1\": 1}</function>\n"
            "<function=python>{\"code\":\"print('hello')\"}</function>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));

    assert_msg_equals(
        simple_assist_msg(
            "This is not a tool call:",
            "",
            "special_function",
            "{\"arg1\": 1}"),
        common_chat_parse(
            "This is not a tool call:\n"
            "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(message_assist,
        common_chat_parse(
            "Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
        common_chat_parse(
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_HERMES_2_PRO}));
    // assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
    //     common_chat_parse(
    //         "I'm\nthinking</think>Hello, world!\nWhat's up?",
    //         COMMON_CHAT_FORMAT_HERMES_2_PRO));
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(message_assist_thoughts_unparsed_md,
        common_chat_parse(
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}```",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ true,
                /* .thinking_forced_open = */ false,
                /* .parse_tool_calls = */ false,
            }));
    assert_msg_equals(message_assist_thoughts_unparsed_md_partial,
        common_chat_parse(
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n```json\n{}```",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ true,
                /* .thinking_forced_open = */ false,
            }));
    assert_msg_equals(message_assist_thoughts_unopened_unparsed,
        common_chat_parse(
            "I'm\nthinking</think>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "I'm\nthinking</think>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ true,
            }));

    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
                    "<tool_call>\n"
                    "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                    "</tool_call>");

    // Test multiple tool calls with template
    common_chat_msg message_assist_multiple_calls_template;
    message_assist_multiple_calls_template.role = "assistant";
    message_assist_multiple_calls_template.content = "";
    message_assist_multiple_calls_template.tool_calls.push_back({"special_function", "{\"arg1\": 1}", ""});
    message_assist_multiple_calls_template.tool_calls.push_back({"python", "{\"code\":\"print('test')\"}", ""});

    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_multiple_calls_template, tools,
                    "<tool_call>\n"
                    "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                    "</tool_call>\n"
                    "<tool_call>\n"
                    "{\"name\": \"python\", \"arguments\": {\"code\":\"print('test')\"}}\n"
                    "</tool_call>");

    // TODO(ochafik): Fix this test - the template produces a format that doesn't match expected
    // test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call_python_lines, tools,
    //               "<tool_call>\n"
    //               "{\"name\": \"python\", \"arguments\": {\"code\":\"# This is a program:\\nprint('hey')\"}}\n"
    //               "</tool_call>");
    assert_msg_equals(
        simple_assist_msg("", /* reasoning_content= */ "<tool_call>nah uhg</tool_call>"),
        common_chat_parse(
            "<think><tool_call>nah uhg</tool_call>",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_HERMES_2_PRO,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));

    run_template_test_suite(impl, template_caps, tmpls);
}
