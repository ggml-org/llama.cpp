#include "../test-chat.h"

void test_mistral_nemo_parser(chat_parser_impl impl)
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
    template_caps.name = "Mistral Nemo";
    template_caps.jinja_path = "models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_MISTRAL_NEMO;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::No;
    template_caps.think_open_tag = nullptr;
    template_caps.think_close_tag = nullptr;
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::No;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
    template_caps.tool_calls_have_ids = ToolCallsHaveIds::Yes;
    template_caps.end_tokens = { "</s>" };

    auto tmpls = read_templates(template_caps.jinja_path);
    run_template_test_suite(impl, template_caps, tmpls);


    assert_equals(COMMON_CHAT_FORMAT_MISTRAL_NEMO, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
    test_templates(
        impl, tmpls.get(), template_caps.end_tokens, message_assist_call_id, tools,
        "[TOOL_CALLS][{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}, \"id\": \"123456789\"}]");
}
            