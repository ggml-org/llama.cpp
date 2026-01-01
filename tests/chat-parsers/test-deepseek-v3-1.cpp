#include "../test-chat.h"

void test_deepseek_v3_1_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    template_capabilities template_caps;
    template_caps.name = "DeepSeek V3.1";
    template_caps.jinja_path = "models/templates/deepseek-ai-DeepSeek-V3.1.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_DEEPSEEK_V3_1;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::Yes;
    template_caps.think_open_tag = "<think>";
    template_caps.think_close_tag = "</think>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::Yes;
    template_caps.supports_disable_thinking = SupportsDisableThinking::No;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
    template_caps.end_tokens = { "<｜end▁of▁sentence｜>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    for (const auto & inputs : { inputs_no_tools, inputs_tools }) {
        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_V3_1, params.format);
        assert_equals(true, params.thinking_forced_open);
    }

    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_thoughts, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
    assert_msg_equals(
        simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking"),
        common_chat_parse(
            "I'm\nthinking</think>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {
                COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ true,
            }));
    // variant: thinking forced open, reasoning_format none
    assert_msg_equals(
        simple_assist_msg("REASONING</think>ok", ""),
        common_chat_parse(
            "REASONING</think>ok",
            /* is_partial= */ false,
            {
                COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_NONE,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ true,
                /* .parse_tool_calls = */ true,
            }));
    // variant: happy path for when it works as the model card says it should
    assert_msg_equals(
        simple_assist_msg("", "", "get_time", "{\"city\":\"Tokyo\"}"),
        common_chat_parse(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            /* is_partial= */ false,
            {
                COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ false,
                /* .parse_tool_calls = */ true,
            }));
    // variant: simple + thinking open
    assert_msg_equals(
        simple_assist_msg("", "REASONING", "get_time", "{\"city\":\"Tokyo\"}"),
        common_chat_parse(
            "REASONING</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            /* is_partial= */ false,
            {
                COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ true,
                /* .parse_tool_calls = */ true,
            }));
    // variant: simple + multiple tool calls
    common_chat_msg message_assist_multiple_calls;
    message_assist_multiple_calls.role = "assistant";
    message_assist_multiple_calls.content = "CONTENT";
    message_assist_multiple_calls.tool_calls.push_back({"get_time", "{\"city\":\"Paris\"}", ""});
    message_assist_multiple_calls.tool_calls.push_back({"get_weather", "{\"city\":\"Paris\"}", ""});
    assert_msg_equals(
        message_assist_multiple_calls,
        common_chat_parse(
            "CONTENT<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            /* is_partial= */ false,
            {
                COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ false,
                /* .parse_tool_calls = */ true,
            }));
    // variant: thinking forced open + tool call in reasoning content
    assert_msg_equals(
        simple_assist_msg("", "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time2<｜tool▁sep｜>{\"city\": \"Tokyo2\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>REASONING", "get_time", "{\"city\":\"Tokyo\"}"),
        common_chat_parse(
            "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time2<｜tool▁sep｜>{\"city\": \"Tokyo2\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>REASONING</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            /* is_partial= */ false,
            {
                COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ true,
                /* .parse_tool_calls = */ true,
            }));
    // variant: thinking forced open + tool call in reasoning content + no closing think + not partial
    //          This is a bit of a fine tuning issue on the model's part IMO. It really should not be attempting
    //          to make tool calls in reasoning content according to the model card, but it does sometimes, so
    //          add the reasoning content as regular content and parse the tool calls.
    assert_msg_equals(
        simple_assist_msg("REASONING", "", "get_time", "{\"city\":\"Tokyo\"}"),
        common_chat_parse(
            "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            /* is_partial= */ false,
            {
                COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ true,
                /* .parse_tool_calls = */ true,
            }));
    // variant: thinking forced open + tool call in reasoning content + no closing think + partial
    assert_msg_equals(
        simple_assist_msg("", "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>", "", ""),
        common_chat_parse(
            "REASONING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"city\": \"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            /* is_partial= */ true,
            {
                COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ true,
                /* .parse_tool_calls = */ true,
            }));
    // variant: thinking not forced open + missing reasoning + no tool calls
    assert_msg_equals(
        simple_assist_msg("CONTENT", ""),
        common_chat_parse(
            "CONTENT",
            /* is_partial= */ false,
            {
                COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ false,
                /* .parse_tool_calls = */ true,
            }));
}