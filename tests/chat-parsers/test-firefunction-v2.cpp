#include "../test-chat.h"

void test_firefunction_v2_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    // Note: template uses `functions` not `tools`, so minja's supports_tools detection returns false
    template_capabilities template_caps;
    template_caps.name = "Firefunction V2";
    template_caps.jinja_path = "models/templates/fireworks-ai-llama-3-firefunction-v2.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_FIREFUNCTION_V2;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::No;
    template_caps.end_tokens = { "<|eot_id|>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_FIREFUNCTION_V2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
                  " functools[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]");
}