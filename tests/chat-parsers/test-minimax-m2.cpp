#include "../test-chat.h"

void test_minimax_m2_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = {special_function_tool, special_function_tool_with_optional_param};

    template_capabilities template_caps;
    template_caps.name = "MiniMax M2";
    template_caps.jinja_path = "models/templates/MiniMax-M2.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_MINIMAX_M2;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;
    template_caps.supports_thinking = ThinkingSupport::Yes;
    template_caps.think_open_tag = "<think>";
    template_caps.think_close_tag = "</think>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::No;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
    template_caps.end_tokens = { "[e~[" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    assert_equals(COMMON_CHAT_FORMAT_MINIMAX_M2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_MINIMAX_M2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

    // Create inputs for parser tests - without reasoning (for content-only tests)
    common_chat_templates_inputs inputs_tools_no_reasoning;
    inputs_tools_no_reasoning.messages = {message_user};
    inputs_tools_no_reasoning.tools = {special_function_tool, special_function_tool_with_optional_param};
    inputs_tools_no_reasoning.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    inputs_tools_no_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);

    // Create inputs with reasoning enabled for reasoning tests
    common_chat_templates_inputs inputs_tools_reasoning;
    inputs_tools_reasoning.messages = {message_user};
    inputs_tools_reasoning.tools = {special_function_tool, special_function_tool_with_optional_param};
    inputs_tools_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs_tools_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);

    // Get syntax for content-only tests
    auto params_no_reasoning = common_chat_templates_apply(tmpls.get(), inputs_tools_no_reasoning);
    common_chat_syntax syntax;
    syntax.format = params_no_reasoning.format;
    if (!params_no_reasoning.parser.empty()) {
        syntax.parser.load(params_no_reasoning.parser);
    }

    // Get syntax with reasoning for reasoning tests
    auto params_reasoning = common_chat_templates_apply(tmpls.get(), inputs_tools_reasoning);
    common_chat_syntax syntax_reasoning;
    syntax_reasoning.format = params_reasoning.format;
    syntax_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    if (!params_reasoning.parser.empty()) {
        syntax_reasoning.parser.load(params_reasoning.parser);
    }

    // PEG parser-specific tests (only run with experimental parser)
    // Legacy format-based parser has different whitespace handling for these cases
    if (impl == chat_parser_impl::EXPERIMENTAL) {
        // Test parsing regular content
        assert_msg_equals(message_assist,
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                syntax));

        // Test parsing content with thinking (thinking_forced_open: model output starts with reasoning directly)
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "I'm\nthinking</think>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                syntax_reasoning));

    // Test parsing tool calls (with proper newlines expected by parser)
    assert_msg_equals(message_assist_call,
        common_chat_parse(
            "<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
            /* is_partial= */ false,
            syntax));

    // Test parsing tool calls with thinking (thinking_forced_open)
    assert_msg_equals(message_assist_call_thoughts,
        common_chat_parse(
            "I'm\nthinking</think><minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
            /* is_partial= */ false,
            syntax_reasoning));

    // Test tool calls with extra content
    assert_msg_equals(message_assist_call_content,
        common_chat_parse(
            "<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            syntax));

    // Test tool calls with extra content AND thinking (thinking_forced_open)
    assert_msg_equals(message_assist_call_thoughts_content,
        common_chat_parse(
            "I'm\nthinking</think><minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            syntax_reasoning));

    // Test streaming (thinking_forced_open: no <think> prefix in input)
    test_parser_with_streaming(message_assist_call_thoughts_content,
        "I'm\nthinking\n</think>Hello, world!\nWhat's up?\n<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, syntax_reasoning); });
    test_parser_with_streaming(message_assist_call_thoughts_content,
        "I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?\n\n<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>\n",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, syntax_reasoning); });
    test_parser_with_streaming(message_assist_call_withopt,
        "<minimax:tool_call>\n<invoke name=\"special_function_with_opt\">\n<parameter name=\"arg1\">1</parameter>\n<parameter name=\"arg2\">2</parameter>\n</invoke>\n</minimax:tool_call>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, syntax); });

    // Test compact format (no extra whitespace) - verifies whitespace flexibility
    assert_msg_equals(message_assist_call,
        common_chat_parse(
            "<minimax:tool_call><invoke name=\"special_function\"><parameter name=\"arg1\">1</parameter></invoke></minimax:tool_call>",
            /* is_partial= */ false,
            syntax));
    } // end PEG parser-specific tests

    // Test template generation for regular content
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools,
                  "Hello, world!\nWhat's up?",
                  /* expect_grammar_triggered= */ false);

    // Test template generation for tool calls
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
                  "<minimax:tool_call>\n<invoke name=\"special_function\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
                  /* expect_grammar_triggered= */ true,
                  /* test_grammar_if_triggered= */ true,
                  /* reasoning_format= */ COMMON_REASONING_FORMAT_NONE,
                  /* ignore_whitespace_differences= */ true
    );

    // Test template generation for tools with optional parameters
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call_noopt, tools,
                  "<minimax:tool_call>\n<invoke name=\"special_function_with_opt\">\n<parameter name=\"arg1\">1</parameter>\n</invoke>\n</minimax:tool_call>",
                  /* expect_grammar_triggered= */ true,
                  /* test_grammar_if_triggered= */ true,
                  /* reasoning_format= */ COMMON_REASONING_FORMAT_NONE,
                  /* ignore_whitespace_differences= */ true
    );
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call_withopt, tools,
                  "<minimax:tool_call>\n<invoke name=\"special_function_with_opt\">\n<parameter name=\"arg1\">1</parameter>\n<parameter name=\"arg2\">2</parameter>\n</invoke>\n</minimax:tool_call>",
                  /* expect_grammar_triggered= */ true,
                  /* test_grammar_if_triggered= */ true,
                  /* reasoning_format= */ COMMON_REASONING_FORMAT_NONE,
                  /* ignore_whitespace_differences= */ true
    );
}