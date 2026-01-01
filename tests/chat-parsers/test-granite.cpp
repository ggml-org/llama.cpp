#include "../test-chat.h"

void test_granite_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    template_capabilities template_caps;
    template_caps.name = "Granite";
    template_caps.jinja_path = "models/templates/llama-cpp-ibm-granite-granite-3.3-2B-Instruct.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_GRANITE;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::Yes;
    template_caps.think_open_tag = "<think>";
    template_caps.think_close_tag = "</think>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::Yes;
    template_caps.supports_disable_thinking = SupportsDisableThinking::Yes;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
    template_caps.end_tokens = { "<|end_of_text|>" };

    auto tmpls = read_templates(template_caps.jinja_path);
    run_template_test_suite(impl, template_caps, tmpls);


    assert_equals(COMMON_CHAT_FORMAT_GRANITE, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);

    assert_equals(COMMON_CHAT_FORMAT_GRANITE, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

    // Test parsing regular content
    assert_msg_equals(message_assist,
        common_chat_parse(
            "Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_GRANITE}));
    assert_msg_equals(
        message_assist,
        common_chat_parse(
            "Hello, world!\nWhat's up?",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_GRANITE}));

    // Test parsing content with thinking
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
        common_chat_parse(
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_GRANITE}));
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?</response>",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(simple_assist_msg("<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?</response>"),
        common_chat_parse(
            "<think>I'm\nthinking</think><response>Hello, world!\nWhat's up?</response>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_GRANITE}));
    assert_msg_equals(message_assist_empty,
        common_chat_parse(
            "<think",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(message_assist_empty,
        common_chat_parse(
            "<think",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_GRANITE}));
    assert_msg_equals(message_assist_thoughts_no_content,
        common_chat_parse(
            "<think>I'm\nthinking",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(
        message_assist_empty,
        common_chat_parse(
            "<think>I'm\nthinking</think><response",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_GRANITE}));

    // Test parsing tool calls
    assert_msg_equals(message_assist_call,
        common_chat_parse(
            "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_GRANITE}));
    assert_msg_equals(
        message_assist_call_empty_args,
        common_chat_parse(
            "<|tool_call|>[{\"name\": \"special_function\"",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_GRANITE}));
    assert_msg_equals(
        message_assist_call_cutoff_args,
        common_chat_parse(
            "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_GRANITE}));
    assert_msg_equals(
        message_assist_call_cutoff_args,
        common_chat_parse(
            "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));

    // Test parsing tool calls with thinking
    assert_msg_equals(
        message_assist_call_thoughts,
        common_chat_parse(
            "<think>I'm\nthinking</think><|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}, {",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GRANITE,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));

    // Test template generation for regular content
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools,
                  "Hello, world!\nWhat's up?",
                  /* expect_grammar_triggered= */ false);

    // Test template generation for tool calls
    // Skip the full template test for now - parser loops over AUTO/REQUIRED and only REQUIRED works without content
    // test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
    //               "<|tool_call|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]",
    //               /* expect_grammar_triggered= */ true
    // );
}