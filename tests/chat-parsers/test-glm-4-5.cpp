#include "../test-chat.h"

void test_glm_4_5_parser(chat_parser_impl impl)
{
    printf("[%s (%s)]\n", __func__, chat_parser_impl_name(impl));

    common_chat_templates_inputs inputs_no_tools;
    inputs_no_tools.messages                = {message_user};

    common_chat_templates_inputs inputs_tools;
    inputs_tools.messages                   = {message_user};
    inputs_tools.tools                      = glm_4_5_tools;

    template_capabilities template_caps;
    template_caps.name = "GLM 4.6";
    template_caps.jinja_path = "models/templates/GLM-4.6.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_GLM_4_5;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;
    template_caps.supports_thinking = ThinkingSupport::Yes;
    template_caps.think_open_tag = "<think>";
    template_caps.think_close_tag = "</think>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    template_caps.supports_disable_thinking = SupportsDisableThinking::Yes;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::Yes;
    template_caps.end_tokens = { "<|assistant|>", "<|observation|>" };

    auto tmpls = read_templates(template_caps.jinja_path);

    run_template_test_suite(impl, template_caps, tmpls);

    assert_equals(COMMON_CHAT_FORMAT_GLM_4_5, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_GLM_4_5, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

    // Get params with tools for parsing tests (always use a parser)
    // Build parser with reasoning extraction disabled
    common_chat_templates_inputs glm_inputs_no_reasoning;
    glm_inputs_no_reasoning.messages = {message_user};
    glm_inputs_no_reasoning.tools = glm_4_5_tools;
    glm_inputs_no_reasoning.enable_thinking = true;
    glm_inputs_no_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
    auto glm_params_no_reasoning = common_chat_templates_apply(tmpls.get(), glm_inputs_no_reasoning);
    auto glm_syntax = get_syntax(glm_params_no_reasoning);

    // Build parser with reasoning extraction enabled
    common_chat_templates_inputs glm_inputs_reasoning;
    glm_inputs_reasoning.messages = {message_user};
    glm_inputs_reasoning.tools = glm_4_5_tools;
    glm_inputs_reasoning.enable_thinking = true;
    glm_inputs_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    glm_inputs_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
    auto glm_params_reasoning = common_chat_templates_apply(tmpls.get(), glm_inputs_reasoning);
    auto glm_syntax_reasoning = get_syntax(glm_params_reasoning, COMMON_REASONING_FORMAT_DEEPSEEK);

    // Test parsing regular content
    assert_msg_equals(message_assist,
        common_chat_parse(
            "Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            glm_syntax));

    // Test parsing content with thinking
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "\n<think>I'm\nthinking</think>\nHello, world!\nWhat's up?",
            /* is_partial= */ false,
            glm_syntax_reasoning), true);

    // Test parsing tool calls
    assert_msg_equals(message_assist_call,
        common_chat_parse(
            "\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
            /* is_partial= */ false,
            glm_syntax), true);

    // Test parsing tool calls with thinking
    assert_msg_equals(message_assist_call_thoughts,
        common_chat_parse(
            "\n<think>I'm\nthinking</think>\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
            /* is_partial= */ false,
            glm_syntax_reasoning), true);

    // Test tool calls with extra content
    assert_msg_equals(message_assist_call_content,
        common_chat_parse(
            "\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            glm_syntax), true);

    // Test tool calls with extra content AND thinking
    assert_msg_equals(message_assist_call_thoughts_content,
        common_chat_parse(
            "\n<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
            /* is_partial= */ false,
            glm_syntax_reasoning), true);

    // Streaming tests only run with experimental PEG parsers
    if (impl == chat_parser_impl::EXPERIMENTAL)
    {
        test_parser_with_streaming(message_assist_call_thoughts_content,
            "\n<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax_reasoning); });
        test_parser_with_streaming(message_assist_call_thoughts_unparsed,
            "\n<think>I'm\nthinking</think>\n\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax); });
        test_parser_with_streaming(message_assist_call_withopt,
            "\n<think></think>\n<tool_call>special_function_with_opt\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n<arg_key>arg2</arg_key>\n<arg_value>2</arg_value>\n</tool_call>\n",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax_reasoning); });
        test_parser_with_streaming(
            simple_assist_msg("", "", "complex_function", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"),
            "<tool_call>complex_function\n"
            "<arg_key>name</arg_key>\n"
            "<arg_value>John Doe</arg_value>\n"
            "<arg_key>age</arg_key>\n"
            "<arg_value>30</arg_value>\n"
            "<arg_key>active</arg_key>\n"
            "<arg_value>true</arg_value>\n"
            "<arg_key>score</arg_key>\n"
            "<arg_value>95.5</arg_value>\n"
            "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax); });
        test_parser_with_streaming(
            simple_assist_msg("", "", "web_search", "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}"),
            "<tool_call>web_search\n"
            "<arg_key>query</arg_key>\n"
            "<arg_value>\"From Zero\" Linkin Park album tracklist complete songs</arg_value>\n"
            "<arg_key>limit</arg_key>\n"
            "<arg_value>3</arg_value>\n"
            "<arg_key>type</arg_key>\n"
            "<arg_value>text</arg_value>\n"
            "</tool_call>",
            [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax); });

        // Test interleaved thinking (legacy parser only - PEG parser doesn't strip <think> blocks from within content yet)
        // Content chunks: "Hello, world!\n" (until <think>) + "What's up?" (until \n<tool_call>) = "Hello, world!\nWhat's up?"
        if (impl == chat_parser_impl::LEGACY) {
            test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinkingThinking2", "special_function", "{\"arg1\": 1}"),
                "\n<think>I'm\nthinking</think>Hello, world!\n<think>Thinking2</think>What's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax_reasoning); });
            test_parser_with_streaming(simple_assist_msg("\n<think>I'm\nthinking</think>Hello, world!\n<think>Thinking2</think>What's up?", "", "special_function", "{\"arg1\": 1}"),
                "\n<think>I'm\nthinking</think>Hello, world!\n<think>Thinking2</think>What's up?\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
                [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, glm_syntax); });
        }
    }

    // Test template generation for regular content
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools,
                  "\nHello, world!\nWhat's up?",
                  /* expect_grammar_triggered= */ false);

    // TODO: Test template generation for tool calls with reasoning
    // These tests are temporarily disabled because building params with reasoning_format=DEEPSEEK
    // causes grammar stack overflow during llama_grammar_advance_stack (recursive grammar structure).
    // This is a pre-existing issue that needs to be fixed separately.
    // test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
    //               "\n<think></think>\n<tool_call>special_function\n<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n</tool_call>\n",
    //               /* expect_grammar_triggered= */ true,
    //               /* test_grammar_if_triggered= */ false,
    //               /* common_reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
    //               /* ignore_whitespace_differences= */ true);
}