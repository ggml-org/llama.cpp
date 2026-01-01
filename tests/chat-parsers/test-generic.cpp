#include "../test-chat.h"

void test_generic_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));
    
    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    template_capabilities template_caps;
    template_caps.name = "Generic";
    template_caps.jinja_path = "models/templates/google-gemma-2-2b-it.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_GENERIC;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::No;
    template_caps.think_open_tag = nullptr;
    template_caps.think_close_tag = nullptr;
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;  // Generic format: EITHER tool_calls OR response, not both
    template_caps.end_tokens = { "<end_of_turn>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_GENERIC, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_GENERIC,
                  common_chat_templates_apply(
                      read_templates("models/templates/microsoft-Phi-3.5-mini-instruct.jinja").get(),
                      inputs_tools)
                      .format);

    // Generic tool calls doesn't generate / parse content-only messages symmetrically.

    assert_equals(
        simple_assist_msg("{ \"tool_call\" : { \"name\" : \"t"),
        common_chat_parse(
            "{ \"tool_call\" : { \"name\" : \"t",
            /* is_partial= */ true,
            {
                /* .format = */ COMMON_CHAT_FORMAT_GENERIC,
                /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                /* .reasoning_in_content = */ false,
                /* .thinking_forced_open = */ true,
                /* .parse_tool_calls = */ false,
            }));
    assert_equals(
        message_assist_empty,
        common_chat_parse(
            "{ \"tool_call\" : { \"name\" : \"t",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_GENERIC}));

    assert_equals(
        simple_assist_msg("", "", "puppeteer_screenshot", "{\"name\":\"servethehome_homepage\","),
        common_chat_parse(
            R"({"tool_call": {"name": "puppeteer_screenshot", "arguments": {"name": "servethehome_homepage",)",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_GENERIC}));

    assert_equals(
        message_assist_call_empty_args,
        common_chat_parse(
            "{ \"tool_call\" : { \"name\" : \"special_function\"",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_GENERIC}));
    assert_equals(
        message_assist_call_cutoff_args,
        common_chat_parse(
            "{ \"tool_call\" : { \"name\" : \"special_function\", \"arguments\" : { \"arg",
            /* is_partial= */ true,
            {COMMON_CHAT_FORMAT_GENERIC}));

    assert_msg_equals(message_assist,
        common_chat_parse(
            "{\n"
            "  \"response\": \"Hello, world!\\nWhat's up?\"\n"
            "}",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_GENERIC}));
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call_id, tools,
                  "{\n"
                  "  \"tool_calls\": [\n"
                  "    {\n"
                  "      \"name\": \"special_function\",\n"
                  "      \"arguments\": {\n"
                  "        \"arg1\": 1\n"
                  "      },\n"
                  "      \"id\": \"123456789\"\n"
                  "    }\n"
                  "  ],\n"
                  "  \"content\": \"\"\n"
                  "}");
}
            