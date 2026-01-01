#include "../test-chat.h"

void test_functionary_v3_1_llama_3_1_parser(chat_parser_impl impl)
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

    template_capabilities template_caps;
    template_caps.name = "Functionary V3.1";
    template_caps.jinja_path = "models/templates/meetkai-functionary-medium-v3.1.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::No;
    template_caps.think_open_tag = nullptr;
    template_caps.think_close_tag = nullptr;
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::Yes;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::Yes;
    template_caps.tool_calls_have_ids = ToolCallsHaveIds::No;
    template_caps.end_tokens = { "<|eom_id|>", "<|eot_id|>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY,
                    common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1,
        common_chat_templates_apply(tmpls.get(), inputs_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY,
                    common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);

    for (auto is_partial : { false, true }) {
        assert_equals(
            message_assist_call,
            common_chat_parse(
                "<function=special_function>{\"arg1\": 1}</function>",
                is_partial,
                {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1}));
    }

    assert_equals(
        message_assist_call,
        common_chat_parse(
            "<function=special_function>{\"arg1\": 1}<",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1}));

    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
                    "<function=special_function>{\"arg1\": 1}</function>");

}