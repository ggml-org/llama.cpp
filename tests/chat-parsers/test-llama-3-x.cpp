#include "../test-chat.h"

void test_llama_3_x_parser(chat_parser_impl impl)
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
        template_capabilities template_caps;
        template_caps.name = "Llama 3.1";
        template_caps.jinja_path = "models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja";
        template_caps.legacy_format = COMMON_CHAT_FORMAT_LLAMA_3_X;
        template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
        template_caps.supports_thinking = ThinkingSupport::No;
        template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
        template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;
        template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
        template_caps.supports_disable_thinking = SupportsDisableThinking::No;
        template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
        template_caps.tool_calls_have_ids = ToolCallsHaveIds::No;
        template_caps.end_tokens = { "<|eom_id|>", "<|eot_id|>" };

        auto tmpls = read_templates(template_caps.jinja_path);

        // Skip run_template_test_suite - it uses python_tool which triggers builtin tools format
        // The second block below tests builtin tools

        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);

        test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
                        "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}");
    }

    {
        template_capabilities template_caps;
        template_caps.name = "Llama 3.1";
        template_caps.jinja_path = "models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja";
        template_caps.legacy_format = COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS;
        template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
        template_caps.supports_thinking = ThinkingSupport::No;
        template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
        template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;
        template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
        template_caps.supports_disable_thinking = SupportsDisableThinking::No;
        template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
        template_caps.tool_calls_have_ids = ToolCallsHaveIds::No;
        template_caps.end_tokens = { "<|eom_id|>", "<|eot_id|>" };

        auto tmpls = read_templates(template_caps.jinja_path);

        run_template_test_suite(impl, template_caps, tmpls);


        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X, common_chat_templates_apply(tmpls.get(), inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
                        common_chat_templates_apply(tmpls.get(), inputs_tools_builtin).format);
        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
                        common_chat_templates_apply(
                            read_templates("models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja").get(),
                            inputs_tools_builtin)
                            .format);

        assert_equals(
            message_assist_call,
            common_chat_parse(
                "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_LLAMA_3_X}));

        // test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools, R"(?)", /* expect_grammar_triggered= */ false);
        test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call_code_interpreter, llama_3_1_tools,
                        "<|python_tag|>code_interpreter.call(code=\"print('hey')\")");
        test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call_python, tools,
                        "<|python_tag|>python.call(code=\"print('hey')\")");
        test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
                        "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}");
    }
}