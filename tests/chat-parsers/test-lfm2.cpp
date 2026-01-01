#include "../test-chat.h"

void test_lfm2_parser(chat_parser_impl impl)
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
    template_caps.name = "LFM2";
    template_caps.jinja_path = "models/templates/llama-cpp-lfm2.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::No;
    template_caps.think_open_tag = nullptr;
    template_caps.think_close_tag = nullptr;
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::Yes;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::Yes;
    template_caps.tool_calls_have_ids = ToolCallsHaveIds::Yes;
    template_caps.end_tokens = { "<|im_end|>" };
    
    auto tmpls = read_templates(template_caps.jinja_path);

    // Skip needle test suite for legacy - legacy parser requires "force json schema." marker in system message
    if (impl != chat_parser_impl::LEGACY) {
        run_template_test_suite(impl, template_caps, tmpls);
    }
    

    auto inputs_tools_forced_json_schema = std::invoke([&]() -> common_chat_templates_inputs {
        common_chat_templates_inputs inputs;
        inputs.messages = {
            std::invoke([&]() -> common_chat_msg {
                common_chat_msg msg;
                msg.role = "system";
                msg.content = "force json schema.\n";
                return msg;
            }),
            message_user,
        };
        inputs.tools = {special_function_tool};
        return inputs;
    });

    {
        auto params = common_chat_templates_apply(tmpls.get(), inputs_no_tools);
        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, params.format);
        assert_equals(false, params.grammar_lazy);
        assert_equals(std::string(R"(<|im_start|>user
Hey there!<|im_end|>
<|im_start|>assistant
)"), params.prompt);
    }

    {
        auto params = common_chat_templates_apply(tmpls.get(), inputs_tools);
        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, params.format);
        assert_equals(false, params.grammar_lazy);
        assert_equals(std::string(R"(<|im_start|>system
List of tools: <|tool_list_start|>[{"type": "function", "function": {"name": "special_function", "description": "I'm special", "parameters": {"type": "object", "properties": {"arg1": {"type": "integer", "description": "The arg."}}, "required": ["arg1"]}}}]<|tool_list_end|><|im_end|>
<|im_start|>user
Hey there!<|im_end|>
<|im_start|>assistant
)"), params.prompt);
        assert_equals(true, params.grammar.empty());
    }

    {
        auto params = common_chat_templates_apply(tmpls.get(), inputs_tools_forced_json_schema);
        assert_equals(COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS, params.format);
        assert_equals(true, params.grammar_lazy);
        assert_equals(std::string(R"(<|im_start|>system
List of tools: <|tool_list_start|>[{"type": "function", "function": {"name": "special_function", "description": "I'm special", "parameters": {"type": "object", "properties": {"arg1": {"type": "integer", "description": "The arg."}}, "required": ["arg1"]}}}]<|tool_list_end|><|im_end|>
<|im_start|>user
Hey there!<|im_end|>
<|im_start|>assistant
)"), params.prompt);
        assert_equals(false, params.grammar.empty());
    }

    // Test parsing regular content
    assert_msg_equals(message_assist,
        common_chat_parse(
            "Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

    // Test single tool call with JSON format
    common_chat_msg msg_single_tool_call;
    msg_single_tool_call.role = "assistant";
    msg_single_tool_call.tool_calls.push_back({"special_function", "{\"arg1\":1}", ""});
    assert_msg_equals(
        msg_single_tool_call,
        common_chat_parse(
            "<|tool_call_start|>[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]<|tool_call_end|>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

    // Test tool call with string argument
    common_chat_msg msg_tool_call_string;
    msg_tool_call_string.role = "assistant";
    msg_tool_call_string.tool_calls.push_back({"get_weather", "{\"location\":\"Paris\"}", ""});
    assert_msg_equals(
        msg_tool_call_string,
        common_chat_parse(
            "<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]<|tool_call_end|>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

    // Test tool call with multiple arguments
    common_chat_msg msg_multi_args;
    msg_multi_args.role = "assistant";
    msg_multi_args.tool_calls.push_back({"calculate", "{\"x\":10,\"y\":20,\"operation\":\"add\"}", ""});
    assert_msg_equals(
        msg_multi_args,
        common_chat_parse(
            "<|tool_call_start|>[{\"name\": \"calculate\", \"arguments\": {\"x\": 10, \"y\": 20, \"operation\": \"add\"}}]<|tool_call_end|>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

    // Test multiple tool calls in single array
    common_chat_msg msg_multiple_tools;
    msg_multiple_tools.role = "assistant";
    msg_multiple_tools.tool_calls.push_back({"get_weather", "{\"location\":\"Paris\"}", ""});
    msg_multiple_tools.tool_calls.push_back({"get_time", "{\"timezone\":\"UTC\"}", ""});
    assert_msg_equals(
        msg_multiple_tools,
        common_chat_parse(
            "<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}, {\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC\"}}]<|tool_call_end|>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

    // Test tool call with content before
    common_chat_msg msg_content_before_tool;
    msg_content_before_tool.role = "assistant";
    msg_content_before_tool.content = "Let me check the weather for you.";
    msg_content_before_tool.tool_calls.push_back({"get_weather", "{\"location\":\"Paris\"}", ""});
    assert_msg_equals(
        msg_content_before_tool,
        common_chat_parse(
            "Let me check the weather for you.<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]<|tool_call_end|>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

    // Test tool call with content after
    common_chat_msg msg_content_after_tool;
    msg_content_after_tool.role = "assistant";
    msg_content_after_tool.content = "Here's the result.";
    msg_content_after_tool.tool_calls.push_back({"get_weather", "{\"location\":\"Paris\"}", ""});
    assert_msg_equals(
        msg_content_after_tool,
        common_chat_parse(
            "<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}]<|tool_call_end|>Here's the result.",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

    // Test tool call with newlines (common in LLM output)
    common_chat_msg msg_tool_call_newlines;
    msg_tool_call_newlines.role = "assistant";
    msg_tool_call_newlines.tool_calls.push_back({"get_current_time", "{\"location\":\"Paris\"}", ""});
    assert_msg_equals(
        msg_tool_call_newlines,
        common_chat_parse(
            "<|tool_call_start|>[{\n    \"name\": \"get_current_time\",\n    \"arguments\": {\n        \"location\": \"Paris\"\n    }\n}]<|tool_call_end|>",
            /* is_partial= */ false,
            {COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS}));

    // Note: LFM2 uses JSON format for tool calls: [{"name": "...", "arguments": {...}}]
    // Unlike other formats, LFM2 template does not render tool calls in conversation history,
    // so we don't use test() for tool call generation. Instead, the parsing tests
    // above verify edge cases and format variations for the tool call output format.
}
