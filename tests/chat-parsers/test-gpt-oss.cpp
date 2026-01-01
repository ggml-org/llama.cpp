#include "../test-chat.h"

void test_gpt_oss_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    template_capabilities template_caps;
    template_caps.name = "GPT OSS";
    template_caps.jinja_path = "models/templates/openai-gpt-oss-120b.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_GPT_OSS;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::Yes;
    template_caps.think_open_tag = "<|inner_thoughts_begin|>";
    template_caps.think_close_tag = "<|inner_thoughts_end|>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::Yes;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;  // Template always outputs final content
    // See eos_token_id in https://huggingface.co/openai/gpt-oss-20b/blob/main/generation_config.json
    template_caps.end_tokens = { "<|return|>", "<|call|>", "<|endoftext|>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);


    assert_equals(COMMON_CHAT_FORMAT_GPT_OSS, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_GPT_OSS, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

    assert_msg_equals(simple_assist_msg("", "I'm\nthink"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthink",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
    assert_msg_equals(simple_assist_msg("", "I'm\nthinking"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
    assert_msg_equals(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
    assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
    assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.special_function<|message|>{\"arg1",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
    assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
    assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>analysis to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
    assert_msg_equals(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>commentary<|message|>Hello, world!\nWhat's up?",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
    assert_msg_equals(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>commentary<|message|>Hello, world!\nWhat's up?<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));

    // Test parse_tool_calls == false
    assert_msg_equals(
        simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ false,
                /* .parse_tool_calls = */ false,
            }));
    assert_msg_equals(
        simple_assist_msg("", "I'm\nthinking"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.special_function<|message|>{\"arg1",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ false,
                /* .parse_tool_calls = */ false,
            }));
    assert_msg_equals(
        simple_assist_msg("", "I'm\nthinking"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ false,
                /* .parse_tool_calls = */ false,
            }));

    // Test reasoning formats
    assert_msg_equals(
        simple_assist_msg(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>Hello, world!\nWhat's up?"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_NONE,
            }));

    assert_msg_equals(
        simple_assist_msg(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>Hello, world!\nWhat's up?"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                /* .reasoning_in_content = */ true,
            }));

    // Test tool calling in role header
    assert_msg_equals(simple_assist_msg("", "", "special_function", "{\"arg1\": 1}"),
        common_chat_parse(
            " to=functions.special_function<|channel|>commentary <|constrain|>json<|message|>{\"arg1\": 1}",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
    assert_msg_equals(simple_assist_msg("", "", "special_function", "{\"arg1\": 1}"),
        common_chat_parse(
            " to=functions.special_function<|channel|>analysis <|constrain|>json<|message|>{\"arg1\": 1}",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
    assert_msg_equals(simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}"),
        common_chat_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant to=functions.special_function<|channel|>analysis <|constrain|>json<|message|>{\"arg1\": 1}",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GPT_OSS,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
}