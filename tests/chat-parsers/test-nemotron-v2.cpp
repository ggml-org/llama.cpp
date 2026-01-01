#include "../test-chat.h"
#include "chat.h"

void test_nemotron_v2_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    template_capabilities template_caps;
    template_caps.name = "Nemotron V2";
    template_caps.jinja_path = "models/templates/NVIDIA-Nemotron-Nano-v2.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_NEMOTRON_V2;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::Yes;
    template_caps.think_open_tag = "<think>";
    template_caps.think_close_tag = "</think>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::No;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
    template_caps.end_tokens = { "<SPECIAL_12>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    assert_equals(COMMON_CHAT_FORMAT_NEMOTRON_V2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_NEMOTRON_V2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

    // Test parsing regular content
    assert_msg_equals(message_assist,
        common_chat_parse(
            "Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_NEMOTRON_V2}));

    // Test parsing content with thinking
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_NEMOTRON_V2,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));

    // Test parsing tool calls
    assert_msg_equals(message_assist_call,
        common_chat_parse(
            "<TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_NEMOTRON_V2}));

    // Test parsing tool calls with thinking
    assert_msg_equals(message_assist_call_thoughts,
        common_chat_parse(
            "<think>I'm\nthinking</think><TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>",
            /* is_partial= */ false,
            {
                /*  .format = */ COMMON_CHAT_FORMAT_NEMOTRON_V2,
                /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }));

    // Test tool calls with extra content
    assert_msg_equals(message_assist_call_content,
        common_chat_parse(
            "<TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_NEMOTRON_V2}
        ));

    // Test tool calls with extra content AND thinking
    assert_msg_equals(message_assist_call_thoughts_content,
        common_chat_parse(
            "<think>I'm\nthinking</think><TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {
                /*  .format = */ COMMON_CHAT_FORMAT_NEMOTRON_V2,
                /*  .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
            }));

    // Test template generation for regular content
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools,
                  "Hello, world!\nWhat's up?\n",
                  /* expect_grammar_triggered= */ false);

    // Test template generation for tool calls
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
                  "<TOOLCALL>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]</TOOLCALL>",
                  /* expect_grammar_triggered= */ true
    );
}