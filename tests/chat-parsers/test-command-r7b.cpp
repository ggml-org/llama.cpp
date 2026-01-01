#include "../test-chat.h"
#include "common.h"

void test_command_r7b_parser(chat_parser_impl impl)
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
        // Command R template is not supported yet and not coverered by this parser.
        auto tmpls = read_templates("models/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja");
        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_GENERIC, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
    }

    template_capabilities template_caps;
    template_caps.name = "Command R7B";
    template_caps.jinja_path = "models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_COMMAND_R7B;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::Yes;
    template_caps.think_open_tag = "<|START_THINKING|>";
    template_caps.think_close_tag = "<|END_THINKING|>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::Yes;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::Yes;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::Yes;
    template_caps.tool_calls_have_ids = ToolCallsHaveIds::Yes;
    template_caps.end_tokens = { "<|END_OF_TURN_TOKEN|>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    for (const auto & inputs : { inputs_no_tools, inputs_tools }) {
        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        assert_equals(COMMON_CHAT_FORMAT_COMMAND_R7B, params.format);
        assert_equals(false, params.thinking_forced_open);
    }

    assert_msg_equals(message_assist,
        common_chat_parse(
            "Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_COMMAND_R7B}));
    assert_msg_equals(message_assist,
        common_chat_parse(
            "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_COMMAND_R7B}));
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
            "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(message_assist_thoughts_unparsed_deepseek,
        common_chat_parse(
            "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
            "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ true,
                /* .thinking_forced_open = */ false,
            }));
    assert_msg_equals(message_assist_thoughts_unparsed_r7b,
        common_chat_parse(
            "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
            "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_COMMAND_R7B}));
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
            "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(message_assist_thoughts_call_idx,
        common_chat_parse(
            "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
            "<|START_ACTION|>[\n"
            "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
            "]<|END_ACTION|>",
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));
    assert_msg_equals(message_assist_thoughts_no_content,
        common_chat_parse(
            "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
            "<|START_ACTION|>[\n"
            "    {\"tool_call_id\": \"0\", \"tool_name\": \"special",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_COMMAND_R7B,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
            }));

    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call_idx, tools,
                    "<|START_THINKING|><|END_THINKING|>"
                    "<|START_ACTION|>[\n"
                    "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
                    "]<|END_ACTION|>",
                    /* expect_grammar_triggered= */ true,
                    /* test_grammar_if_triggered= */ true,
                    COMMON_REASONING_FORMAT_DEEPSEEK);
    // TODO(ochafik): Template defeats the delta logic, as emits <|START_OF_TURN_TOKEN|> (in prefix) vs. <|START_RESPONSE|> (full)
    // test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools,
    //                 "<|START_RESPONSE|>Hello, world!\n"
    //                 "What's up?<|END_RESPONSE|>",
    //                 /* expect_grammar_triggered= */ false,
    //                 /* test_grammar_if_triggered= */ true,
    //                 /* reasoning_format= */ COMMON_REASONING_FORMAT_NONE,
    //                 // TODO(ochafik): check why a trailing newline creeped in here
    //                 /* ignore_whitespace_differences= */ true);
}