#include "../test-chat.h"

static const char * invoice_schema = R"({
    "type": "object",
    "properties": {
        "amount": {"type": "number"},
        "date": {"type": "string"}
    }
})";

void test_nemotron_v3_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    template_capabilities template_caps;
    template_caps.name = "Nemotron V3";
    template_caps.jinja_path = "models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;
    template_caps.supports_thinking = ThinkingSupport::Yes;
    template_caps.think_open_tag = "<think>";
    template_caps.think_close_tag = "</think>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::No;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
    template_caps.end_tokens = { "<|im_end|>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    if (impl == chat_parser_impl::LEGACY) {
        // Test basic message
        test_peg_parser(impl, tmpls.get(), [&](auto & t) {
            t.input = "Hello, world!\nWhat's up?";
            t.expect = message_assist;
        });

        // Test basic message and reasoning with reasoning_format = none
        test_peg_parser(impl, tmpls.get(), [&](auto & t) {
            t.input = "I'm\nthinking\n</think>\nHello, world!\nWhat's up?";
            t.expect.content = "I'm\nthinking\n</think>\nHello, world!\nWhat's up?";
        });

        // Test basic message and reasoning with reasoning_format = auto
        test_peg_parser(impl, tmpls.get(), [&](auto & t) {
            t.input = "I'm\nthinking\n</think>\nHello, world!\nWhat's up?";
            t.params.enable_thinking = true;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

            t.expect = message_assist_thoughts;
        });

        // Test tool call
        test_peg_parser(impl, tmpls.get(), [&](auto & t) {
            t.input =
                "<tool_call>\n"
                "<function=special_function>\n"
                "<parameter=arg1>\n"
                "1\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>";
            t.params.enable_thinking = false;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.tools = {special_function_tool};

            t.expect = message_assist_call;
        });

        // Test tool call with reasoning
        test_peg_parser(impl, tmpls.get(), [&](auto & t) {
            t.input =
                "I'm\nthinking\n</think>\n"
                "<tool_call>\n"
                "<function=special_function>\n"
                "<parameter=arg1>\n"
                "1\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>";
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.tools = {special_function_tool};

            t.expect = message_assist_call_thoughts;
        });

        // Test parallel tool calls
        test_peg_parser(impl, tmpls.get(), [&](auto & t) {
            t.input =
                "<tool_call>\n"
                "<function=special_function>\n"
                "<parameter=arg1>\n"
                "1\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n"
                "<tool_call>\n"
                "<function=special_function_with_opt>\n"
                "<parameter=arg1>\n"
                "1\n"
                "</parameter>\n"
                "<parameter=arg2>\n"
                "2\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>";
            t.params.enable_thinking = false;
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

        // Test tool call with string parameter
        test_peg_parser(impl, tmpls.get(), [&](auto & t) {
            t.input =
                "<tool_call>\n"
                "<function=python>\n"
                "<parameter=code>\n"
                "def hello():\n"
                "    print(\"Hello, world!\")\n"
                "\n"
                "hello()\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>";
            t.params.enable_thinking = false;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.tools = {python_tool};

            t.expect.tool_calls = {{
                /* .name = */      "python",
                /* .arguments = */ "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}",
                /* .id = */        {},
            }};
        });

        // Test tool call with string parameter and no closing </parameter> tag
        test_peg_parser(impl, tmpls.get(), [&](auto & t) {
            t.input =
                "<tool_call>\n"
                "<function=python>\n"
                "<parameter=code>\n"
                "def hello():\n"
                "    print(\"Hello, world!\")\n"
                "\n"
                "hello()\n"
                "</function>\n"
                "</tool_call>";
            t.params.enable_thinking = false;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.tools = {python_tool};

            t.expect.tool_calls = {{
                /* .name = */      "python",
                /* .arguments = */ "{\"code\": \"def hello():\\n    print(\\\"Hello, world!\\\")\\n\\nhello()\"}",
                /* .id = */        {},
            }};
        });

        // Test response format
        test_peg_parser(impl, tmpls.get(), [&](auto & t) {
            t.input =
                "I need to output the invoice details in JSON\n"
                "</think>\n"
                R"({"amount": 123.45, "date": "2025-12-03"})";
            t.params.enable_thinking = true;
            t.params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            t.params.json_schema = invoice_schema;

            t.expect.reasoning_content = "I need to output the invoice details in JSON";
            t.expect.content = R"({"amount": 123.45, "date": "2025-12-03"})";
        });
    }
}