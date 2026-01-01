#include "../test-chat.h"

static const char * invoice_schema = R"({
    "type": "object",
    "properties": {
        "amount": {"type": "number"},
        "date": {"type": "string"}
    }
})";

void test_ministral_3_parser(chat_parser_impl impl)
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
    template_caps.name = "Ministral V3";
    template_caps.jinja_path = "models/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::No;
    template_caps.think_open_tag = nullptr;
    template_caps.think_close_tag = nullptr;
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::No;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
    template_caps.tool_calls_have_ids = ToolCallsHaveIds::No;

    auto tmpls = read_templates(template_caps.jinja_path);
    run_template_test_suite(impl, template_caps, tmpls);

    // Test basic message
    test_peg_parser(impl, tmpls.get(), [&](auto & t) {
        t.input = "Hello, world!\nWhat's up?";
        t.expect = message_assist;
    });

    // Test basic message and reasoning with reasoning_format = none
    test_peg_parser(impl, tmpls.get(), [&](auto & t) {
        t.input = "[THINK]I'm\nthinking[/THINK]Hello, world!\nWhat's up?";
        t.expect.content = "[THINK]I'm\nthinking[/THINK]Hello, world!\nWhat's up?";
    });

    // Test basic message and reasoning with reasoning_format = auto
    test_peg_parser(impl, tmpls.get(), [&](auto & t) {
        t.input = "[THINK]I'm\nthinking[/THINK]Hello, world!\nWhat's up?";
        t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

        t.expect = message_assist_thoughts;
    });

    // Test tool call
    test_peg_parser(impl, tmpls.get(), [&](auto & t) {
        t.input = R"([TOOL_CALLS]special_function[ARGS]{"arg1":1})";
        t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        t.params.tools = {special_function_tool};

        t.expect = message_assist_call;
    });

    // Test tool call with reasoning
    test_peg_parser(impl, tmpls.get(), [&](auto & t) {
        t.input = "[THINK]I'm\nthinking[/THINK]"
                    R"([TOOL_CALLS]special_function[ARGS]{"arg1":1})";
        t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        t.params.tools = {special_function_tool};

        t.expect = message_assist_call_thoughts;
    });

    // Test parallel tool calls
    test_peg_parser(impl, tmpls.get(), [&](auto & t) {
        t.input = R"([TOOL_CALLS]special_function[ARGS]{"arg1": 1})"
                    R"([TOOL_CALLS]special_function_with_opt[ARGS]{"arg1": 1, "arg2": 2})";
        t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        t.params.parallel_tool_calls = true;
        t.params.tools = {special_function_tool, special_function_tool_with_optional_param};

        t.expect.tool_calls = {{
            /* .name = */      "special_function",
            /* .arguments = */ R"({"arg1": 1})",
            /* .id = */        {},
        }, {
            /* .name = */      "special_function_with_opt",
            /* .arguments = */ R"({"arg1": 1, "arg2": 2})",
            /* .id = */        {},
        }};
    });

    // Test response format
    test_peg_parser(impl, tmpls.get(), [&](auto & t) {
        t.input = "[THINK]I need to output the invoice details in JSON[/THINK]"
                    "```json\n"
                    R"({"amount": 123.45, "date": "2025-12-03"})"
                    "\n```";
        t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        t.params.json_schema = invoice_schema;

        t.expect.reasoning_content = "I need to output the invoice details in JSON";
        t.expect.content =R"({"amount": 123.45, "date": "2025-12-03"})";
    });
}
