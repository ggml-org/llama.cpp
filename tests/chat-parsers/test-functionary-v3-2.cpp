#include "../test-chat.h"

void test_functionary_v3_2_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    template_capabilities template_caps;
    template_caps.name = "Functionary V3.2";
    template_caps.jinja_path = "models/templates/meetkai-functionary-medium-v3.2.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::No;
    template_caps.think_open_tag = nullptr;
    template_caps.think_close_tag = nullptr;
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::Yes;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::Yes;
    template_caps.end_tokens = { "<|eom_id|>", "<|eot_id|>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);
    
    assert_equals(COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

    assert_msg_equals(
        simple_assist_msg(
            "Hello, world!\nnono\nWhat's up?",
            "",
            "special_function",
            "{\"arg1\": 1}"),
        common_chat_parse(
            "all\n"
            "Hello, world!\n"
            "nono\n"
            "What's up?>>>special_function\n"
            "{\"arg1\": 1}\n",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
    assert_msg_equals(message_assist_call_python_lines,
        common_chat_parse(
            "python\n"
            "# This is a program:\n"
            "print('hey')",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
    assert_msg_equals(message_assist_call_python_lines_unclosed,
        common_chat_parse(
            "python\n"
            "# This is a program:\n"
            "print('hey')",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
    assert_msg_equals(message_assist_call,
        common_chat_parse(
            "special_function\n"
            "{\"arg1\": 1} \n                    ",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));
    assert_msg_equals(message_assist,
        common_chat_parse(
            "all\n"
            "Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2}));

    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, {},
                  "all\n"
                  "Hello, world!\n"
                  "What's up?",
                  /* expect_grammar_triggered= */ false);
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
                  "special_function\n"
                  "{\"arg1\": 1}");
}
