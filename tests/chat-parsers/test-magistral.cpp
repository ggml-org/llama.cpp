#include "../test-chat.h"
#include "chat.h"

void test_magistral_parser(chat_parser_impl impl)
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
    template_caps.name = "Magistral (unsloth)";
    template_caps.jinja_path = "models/templates/unsloth-Magistral-Small-2509.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_MAGISTRAL;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::No;
    template_caps.think_open_tag = nullptr;
    template_caps.think_close_tag = nullptr;
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::Yes;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::Yes;
    // Template format [TOOL_CALLS]name[ARGS]{...} doesn't include ids
    template_caps.tool_calls_have_ids = ToolCallsHaveIds::No;
    
    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);
    
    assert_msg_equals(
        simple_assist_msg("Réponse", "raisonnement"),
        common_chat_parse(
            message_assist_thoughts_unparsed_magistral.content,
            /* is_partial= */ false,
            {
                /* .format = */ COMMON_CHAT_FORMAT_MAGISTRAL,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
            }));
}
    