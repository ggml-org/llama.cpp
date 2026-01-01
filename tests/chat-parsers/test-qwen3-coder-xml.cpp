#include "../test-chat.h"
#include "chat.h"

void test_qwen3_coder_xml_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool};

    template_capabilities template_caps;
    template_caps.name = "Qwen3 Coder";
    template_caps.jinja_path = "models/templates/Qwen3-Coder.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_QWEN3_CODER_XML;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;
    template_caps.supports_thinking = ThinkingSupport::No;
    template_caps.think_open_tag = nullptr;
    template_caps.think_close_tag = nullptr;
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::No;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::No;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
    template_caps.end_tokens = { "<|im_end|>", "<|endoftext|>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    {
        common_chat_templates_inputs inputs;
        inputs.messages = {message_user};
        inputs.tools = {special_function_tool};
        inputs.parallel_tool_calls = true;
        inputs.experimental_new_parsers = impl == chat_parser_impl::EXPERIMENTAL;

        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        auto syntax = get_syntax(params);
        assert_equals(inputs.experimental_new_parsers ? COMMON_CHAT_FORMAT_PEG_CONSTRUCTED : COMMON_CHAT_FORMAT_QWEN3_CODER_XML, params.format);

        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                " <tool_call>\n"
                "<function=special_function> <parameter=arg1>1\n"
                "</parameter>\n"
                "</function> </tool_call>\n"
                "\n"
                "\n",
                /* is_partial= */ false,
                syntax));

        // Test streaming diff computation (used by the server for SSE streaming).
        // This catches bugs that run_template_test_suite misses because it exercises
        // common_chat_msg_diff::compute_diffs() which the server uses for streaming.
        test_parser_with_streaming(
            message_assist_call,
            " <tool_call>\n"
            "<function=special_function> <parameter=arg1>1\n"
            "</parameter>\n"
            "</function> </tool_call>\n",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, syntax); });
    }

    // Test Qwen3-Coder XML format
    {
        // Load template and build parser with tools

        // Define all tools used in these tests with proper types matching test expectations
        std::vector<common_chat_tool> qwen3_coder_tools = {
            { "special_function", "A special function", R"({"type":"object","properties":{"arg1":{"type":"integer"}},"required":["arg1"]})" },
            { "special_function_with_opt", "A function with optional param", R"({"type":"object","properties":{"arg1":{"type":"integer"},"arg2":{"type":"integer"}},"required":["arg1"]})" },
            { "complex_function", "A complex function", R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"active":{"type":"boolean"},"score":{"type":"number"}},"required":["name","age","active","score"]})" },
            { "unicode_function", "A unicode function", R"({"type":"object","properties":{"message":{"type":"string"}},"required":["message"]})" },
            { "code_function", "A code function", R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})" },
            { "json_function", "A JSON function", R"({"type":"object","properties":{"config":{"type":"object"}},"required":["config"]})" },
            { "array_function", "An array function", R"({"type":"object","properties":{"items":{"type":"array"}},"required":["items"]})" },
            { "empty_function", "An empty param function", R"({"type":"object","properties":{"empty_param":{"type":"string"}},"required":["empty_param"]})" },
            { "boolean_function", "A boolean function", R"({"type":"object","properties":{"enabled":{"type":"boolean"},"debug":{"type":"boolean"}},"required":["enabled","debug"]})" },
            { "null_function", "A null function", R"({"type":"object","properties":{"optional_param":{"type":"null"}},"required":["optional_param"]})" },
            { "math_function", "A math function", R"({"type":"object","properties":{"negative":{"type":"integer"},"decimal":{"type":"number"},"scientific":{"type":"number"},"formula":{"type":"string"}}})" },
            { "xml_function", "An XML function", R"({"type":"object","properties":{"xml_content":{"type":"string"}},"required":["xml_content"]})" },
            { "quote_function", "A quote function", R"({"type":"object","properties":{"message":{"type":"string"}},"required":["message"]})" },
            { "long_function", "A long text function", R"({"type":"object","properties":{"long_text":{"type":"string"}},"required":["long_text"]})" },
            { "search_function", "A search function", R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})" },
            { "compact_function", "A compact function", R"({"type":"object","properties":{"param":{"type":"string"}},"required":["param"]})" },
            { "get_user_data_v2", "A user data function", R"({"type":"object","properties":{"user_id":{"type":"integer"}},"required":["user_id"]})" },
            { "test_function", "A test function", R"({"type":"object","properties":{"param_1":{"type":"string"},"param_2_name":{"type":"string"},"param3":{"type":"integer"}},"required":["param_1","param_2_name","param3"]})" },
            { "xml_parser", "An XML parser function", R"({"type":"object","properties":{"xml":{"type":"string"}},"required":["xml"]})" },
            { "whitespace_function", "A whitespace function", R"({"type":"object","properties":{"spaces":{"type":"string"}},"required":["spaces"]})" },
            { "tab_function", "A tab function", R"({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]})" },
            { "control_function", "A control function", R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})" },
            { "emoji_function", "An emoji function", R"({"type":"object","properties":{"message":{"type":"string"}},"required":["message"]})" },
            { "number_function", "A number function", R"({"type":"object","properties":{"big_int":{"type":"integer"}},"required":["big_int"]})" },
            { "binary_function", "A binary function", R"({"type":"object","properties":{"data":{"type":"string"}},"required":["data"]})" },
            { "sql_function", "A SQL function", R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})" },
            { "html_function", "An HTML function", R"({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]})" },
            { "python", "A python function", R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})" },
        };

        // Build parser with tools
        common_chat_templates_inputs qwen3_inputs;
        qwen3_inputs.messages = {message_user};
        qwen3_inputs.tools = qwen3_coder_tools;
        qwen3_inputs.parallel_tool_calls = true;
        auto qwen3_params = common_chat_templates_apply(tmpls.get(), qwen3_inputs);
        auto qwen3_syntax = get_syntax(qwen3_params);

        // Basic XML tool call parsing
        assert_msg_equals(
            message_assist_call,
            common_chat_parse(
                "<tool_call>\n"
                "  <function=special_function>\n"
                "    <parameter=arg1>\n"
                "      1\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
                /* is_partial= */ false,
                qwen3_syntax));

        // Multiple parameters with different types
        common_chat_msg expected_multi_param;
        expected_multi_param.role = "assistant";
        expected_multi_param.tool_calls = {
            { "complex_function", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}", "" }
        };

        test_parser_with_streaming(expected_multi_param,
                "<tool_call>\n"
                "  <function=complex_function>\n"
                "    <parameter=name>\n"
                "      John Doe\n"
                "    </parameter>\n"
                "    <parameter=age>\n"
                "      30\n"
                "    </parameter>\n"
                "    <parameter=active>\n"
                "      true\n"
                "    </parameter>\n"
                "    <parameter=score>\n"
                "      95.5\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Special characters and Unicode
        common_chat_msg expected_special_chars;
        expected_special_chars.role = "assistant";
        expected_special_chars.tool_calls = {
            { "unicode_function", "{\"message\":\"Hello 世界! 🌍 Special chars: @#$%^&*()\"}", "" }
        };

        test_parser_with_streaming(expected_special_chars,
                "<tool_call>\n"
                "  <function=unicode_function>\n"
                "    <parameter=message>\n"
                "      Hello 世界! 🌍 Special chars: @#$%^&*()\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Multiline content with newlines and indentation
        common_chat_msg expected_multiline;
        expected_multiline.role = "assistant";
        expected_multiline.tool_calls = {
            { "code_function", "{\"code\":\"def hello():\\n    print(\\\"Hello, World!\\\")\\n    return True\"}", "" }
        };

        test_parser_with_streaming(expected_multiline,
                "<tool_call>\n"
                "  <function=code_function>\n"
                "    <parameter=code>\n"
                "def hello():\n"
                "    print(\"Hello, World!\")\n"
                "    return True\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // JSON object as parameter value
        common_chat_msg expected_json_param;
        expected_json_param.role = "assistant";
        expected_json_param.tool_calls = {
            { "json_function", "{\"config\":{\"host\":\"localhost\",\"port\":8080,\"ssl\":false}}", "" }
        };

        test_parser_with_streaming(
            expected_json_param,
                "<tool_call>\n"
                "  <function=json_function>\n"
                "    <parameter=config>\n"
                "      {\"host\": \"localhost\", \"port\": 8080, \"ssl\": false}\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Array as parameter value
        common_chat_msg expected_array_param;
        expected_array_param.role = "assistant";
        expected_array_param.tool_calls = {
            { "array_function", "{\"items\":[\"apple\",\"banana\",\"cherry\"]}", "" }
        };

        test_parser_with_streaming(
            expected_array_param,
                "<tool_call>\n"
                "  <function=array_function>\n"
                "    <parameter=items>\n"
                "      [\"apple\", \"banana\", \"cherry\"]\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Empty parameter
        common_chat_msg expected_empty_param;
        expected_empty_param.role = "assistant";
        expected_empty_param.tool_calls = {
            { "empty_function", "{\"empty_param\":\"\"}", "" }
        };

        test_parser_with_streaming(
            expected_empty_param,
                "<tool_call>\n"
                "  <function=empty_function>\n"
                "    <parameter=empty_param>\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Boolean values (true/false)
        common_chat_msg expected_boolean;
        expected_boolean.role = "assistant";
        expected_boolean.tool_calls = {
            { "boolean_function", "{\"enabled\":true,\"debug\":false}", "" }
        };

        test_parser_with_streaming(
            expected_boolean,
                "<tool_call>\n"
                "  <function=boolean_function>\n"
                "    <parameter=enabled>\n"
                "      true\n"
                "    </parameter>\n"
                "    <parameter=debug>\n"
                "      false\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Null value
        common_chat_msg expected_null;
        expected_null.role = "assistant";
        expected_null.tool_calls = {
            { "null_function", "{\"optional_param\":null}", "" }
        };

        test_parser_with_streaming(
            expected_null,
                "<tool_call>\n"
                "  <function=null_function>\n"
                "    <parameter=optional_param>\n"
                "      null\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Negative numbers and scientific notation
        common_chat_msg expected_numbers;
        expected_numbers.role = "assistant";
        expected_numbers.tool_calls = {
            { "math_function", "{\"negative\":-42,\"decimal\":-3.14,\"scientific\":1.23e-4}", "" }
        };

        test_parser_with_streaming(
            expected_numbers,
                "<tool_call>\n"
                "  <function=math_function>\n"
                "    <parameter=negative>\n"
                "      -42\n"
                "    </parameter>\n"
                "    <parameter=decimal>\n"
                "      -3.14\n"
                "    </parameter>\n"
                "    <parameter=scientific>\n"
                "      1.23e-4\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // XML-like content in parameters (should be escaped)
        common_chat_msg expected_xml_content;
        expected_xml_content.role = "assistant";
        expected_xml_content.tool_calls = {
            { "xml_function", "{\"xml_content\":\"<root><item>value</item></root>\"}", "" }
        };

        test_parser_with_streaming(
            expected_xml_content,
                "<tool_call>\n"
                "  <function=xml_function>\n"
                "    <parameter=xml_content>\n"
                "      <root><item>value</item></root>\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Quotes and escape characters
        common_chat_msg expected_quotes;
        expected_quotes.role = "assistant";
        expected_quotes.tool_calls = {
            { "quote_function", "{\"message\":\"She said \\\"Hello!\\\" and left.\"}", "" }
        };

        test_parser_with_streaming(
            expected_quotes,
                "<tool_call>\n"
                "  <function=quote_function>\n"
                "    <parameter=message>\n"
                "      She said \"Hello!\" and left.\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Long parameter value (simplified)
        std::string long_text = "This is a long text parameter that should test the parser's ability to handle larger amounts of text data.";

        common_chat_msg expected_long_text;
        expected_long_text.role = "assistant";
        expected_long_text.tool_calls = {
            { "long_function", "{\"long_text\":\"" + long_text + "\"}", "" }
        };

        test_parser_with_streaming(
            expected_long_text,
                "<tool_call>\n"
                "  <function=long_function>\n"
                "    <parameter=long_text>\n"
                "      " + long_text + "\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Mixed content with text before and after tool call
        common_chat_msg expected_mixed_content;
        expected_mixed_content.role = "assistant";
        expected_mixed_content.content = "I'll help you search for products. ";
        expected_mixed_content.tool_calls = {
            { "search_function", "{\"query\":\"laptops\"}", "" }
        };

        test_parser_with_streaming(
            expected_mixed_content,
                "I'll help you search for products. <tool_call>\n"
                "  <function=search_function>\n"
                "    <parameter=query>\n"
                "      laptops\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Compact format (no extra whitespace)
        common_chat_msg expected_compact;
        expected_compact.role = "assistant";
        expected_compact.tool_calls = {
            { "compact_function", "{\"param\":\"value\"}", "" }
        };

        test_parser_with_streaming(
            expected_compact,
                "<tool_call><function=compact_function><parameter=param>value</parameter></function></tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Function name with underscores and numbers
        common_chat_msg expected_complex_name;
        expected_complex_name.role = "assistant";
        expected_complex_name.tool_calls = {
            { "get_user_data_v2", "{\"user_id\":12345}", "" }
        };

        test_parser_with_streaming(
            expected_complex_name,
                "<tool_call>\n"
                "  <function=get_user_data_v2>\n"
                "    <parameter=user_id>\n"
                "      12345\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Parameter names with underscores and numbers
        common_chat_msg expected_complex_params;
        expected_complex_params.role = "assistant";
        expected_complex_params.tool_calls = {
            { "test_function", "{\"param_1\":\"value1\",\"param_2_name\":\"value2\",\"param3\":123}", "" }
        };

        test_parser_with_streaming(
            expected_complex_params,
                "<tool_call>\n"
                "  <function=test_function>\n"
                "    <parameter=param_1>\n"
                "      value1\n"
                "    </parameter>\n"
                "    <parameter=param_2_name>\n"
                "      value2\n"
                "    </parameter>\n"
                "    <parameter=param3>\n"
                "      123\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Very deeply nested XML content in parameter
        common_chat_msg expected_deep_xml;
        expected_deep_xml.role = "assistant";
        expected_deep_xml.tool_calls = {
            { "xml_parser", "{\"xml\":\"<root><level1><level2><level3>deep content</level3></level2></level1></root>\"}", "" }
        };

        test_parser_with_streaming(
            expected_deep_xml,
                "<tool_call>\n"
                "  <function=xml_parser>\n"
                "    <parameter=xml>\n"
                "      <root><level1><level2><level3>deep content</level3></level2></level1></root>\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Parameter with only whitespace
        common_chat_msg expected_whitespace_param;
        expected_whitespace_param.role = "assistant";
        expected_whitespace_param.tool_calls = {
            { "whitespace_function", "{\"spaces\":\"\"}", "" }
        };

        test_parser_with_streaming(
            expected_whitespace_param,
                "<tool_call>\n"
                "  <function=whitespace_function>\n"
                "    <parameter=spaces>\n"
                "      \n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Parameter with tabs and mixed whitespace
        common_chat_msg expected_mixed_whitespace;
        expected_mixed_whitespace.role = "assistant";
        expected_mixed_whitespace.tool_calls = {
            { "tab_function", "{\"content\":\"line1\\n\\tindented line\\n    spaces\"}", "" }
        };

        test_parser_with_streaming(
            expected_mixed_whitespace,
                "<tool_call>\n"
                "  <function=tab_function>\n"
                "    <parameter=content>\n"
                "line1\n"
                "\tindented line\n"
                "    spaces\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Control characters and special Unicode
        common_chat_msg expected_control_chars;
        expected_control_chars.role = "assistant";
        expected_control_chars.tool_calls = {
            { "control_function", "{\"text\":\"Line1\\nLine2\\tTabbed\\rCarriage return\"}", "" }
        };

        test_parser_with_streaming(
            expected_control_chars,
                "<tool_call>\n"
                "  <function=control_function>\n"
                "    <parameter=text>\n"
                "Line1\nLine2\tTabbed\rCarriage return\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Emoji and extended Unicode characters
        common_chat_msg expected_emoji;
        expected_emoji.role = "assistant";
        expected_emoji.tool_calls = {
            { "emoji_function", "{\"message\":\"Hello! 👋 🌟 🚀 Testing emojis: 😀😃😄😁 and symbols: ∑∏∆∇\"}", "" }
        };

        test_parser_with_streaming(
            expected_emoji,
                "<tool_call>\n"
                "  <function=emoji_function>\n"
                "    <parameter=message>\n"
                "      Hello! 👋 🌟 🚀 Testing emojis: 😀😃😄😁 and symbols: ∑∏∆∇\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Mathematical expressions and formulas
        common_chat_msg expected_math;
        expected_math.role = "assistant";
        expected_math.tool_calls = {
            { "math_function", "{\"formula\":\"E = mc² and ∫f(x)dx = F(x) + C\"}", "" }
        };

        test_parser_with_streaming(
            expected_math,
                "<tool_call>\n"
                "  <function=math_function>\n"
                "    <parameter=formula>\n"
                "      E = mc² and ∫f(x)dx = F(x) + C\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // SQL injection-like content (should be safely escaped)
        common_chat_msg expected_sql;
        expected_sql.role = "assistant";
        expected_sql.tool_calls = {
            { "sql_function", "{\"query\":\"SELECT * FROM users WHERE id = 1; DROP TABLE users; --\"}", "" }
        };

        test_parser_with_streaming(
            expected_sql,
                "<tool_call>\n"
                "  <function=sql_function>\n"
                "    <parameter=query>\n"
                "      SELECT * FROM users WHERE id = 1; DROP TABLE users; --\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // HTML/XML injection content
        common_chat_msg expected_html;
        expected_html.role = "assistant";
        expected_html.tool_calls = {
            { "html_function", "{\"content\":\"<script>alert('xss')</script><img src=x onerror=alert(1)>\"}", "" }
        };

        test_parser_with_streaming(
            expected_html,
                "<tool_call>\n"
                "  <function=html_function>\n"
                "    <parameter=content>\n"
                "      <script>alert('xss')</script><img src=x onerror=alert(1)>\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Binary-like content (base64)
        common_chat_msg expected_binary;
        expected_binary.role = "assistant";
        expected_binary.tool_calls = {
            { "binary_function", "{\"data\":\"SGVsbG8gV29ybGQhIFRoaXMgaXMgYmFzZTY0IGVuY29kZWQgdGV4dC4=\"}", "" }
        };

        test_parser_with_streaming(
            expected_binary,
                "<tool_call>\n"
                "  <function=binary_function>\n"
                "    <parameter=data>\n"
                "      SGVsbG8gV29ybGQhIFRoaXMgaXMgYmFzZTY0IGVuY29kZWQgdGV4dC4=\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });

        // Very large numbers (should be parsed as scientific notation)
        common_chat_msg expected_large_numbers;
        expected_large_numbers.role = "assistant";
        expected_large_numbers.tool_calls = {
            { "number_function", "{\"big_int\":1e+60}", "" }  // Large number becomes scientific notation
        };

        test_parser_with_streaming(
            expected_large_numbers,
                "<tool_call>\n"
                "  <function=number_function>\n"
                "    <parameter=big_int>\n"
                "      999999999999999999999999999999999999999999999999999999999999\n"
                "    </parameter>\n"
                "  </function>\n"
                "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, qwen3_syntax); });
    }

    {
        // Qwen3-Coder template
        common_chat_templates_inputs inputs;
        inputs.messages = { message_user };

        common_chat_tool qwen_union_tool {
            /* .name = */ "qwen_union",
            /* .description = */ "Test tool for union/anyOf handling",
            /* .parameters = */ R"({
                "type": "object",
                "properties": {
                    "priority": { "type": ["number", "null"] },
                    "maybe_text": { "anyOf": [ { "type": "string" } ] },
                    "config": { "anyOf": [ { "type": "object" }, { "type": "null" } ] }
                },
                "required": []
            })",
        };
        inputs.tools = { qwen_union_tool };

        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        assert_equals(COMMON_CHAT_FORMAT_QWEN3_CODER_XML, params.format);
        assert_equals(false, params.grammar.empty());

        // Grammar should compile successfully
        auto grammar = build_grammar(params.grammar);
        GGML_ASSERT(grammar && "Failed to build Qwen3-Coder grammar with union types");
    }
}
