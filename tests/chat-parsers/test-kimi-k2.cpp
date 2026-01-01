#include "../test-chat.h"

void test_kimi_k2_parser(chat_parser_impl impl)
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
    template_caps.name = "Kimi K2";
    template_caps.jinja_path = "models/templates/Kimi-K2-Thinking.jinja";
    template_caps.legacy_format = COMMON_CHAT_FORMAT_KIMI_K2;
    template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    template_caps.supports_thinking = ThinkingSupport::Yes;
    template_caps.think_open_tag = "<think>";
    template_caps.think_close_tag = "</think>";
    template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
    template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
    template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
    // Note: Kimi template always outputs <think></think> tags, and discards reasoning_content
    // for the last non-tool-call assistant message (puts it in hist_msgs). This means the
    // needle tests expecting reasoning extraction won't work with this template's structure.
    template_caps.supports_disable_thinking = SupportsDisableThinking::No;
    template_caps.supports_reasoning_only = SupportsReasoningOnly::No;
    template_caps.tool_calls_have_ids = ToolCallsHaveIds::Yes;
    template_caps.end_tokens = { "<|im_end|>" };
    
    auto tmpls = read_templates(template_caps.jinja_path);

    // Note: Kimi template splits messages into hist_msgs (reasoning discarded) and suffix_msgs
    // (reasoning preserved). The needle tests use a single assistant message which becomes
    // the "last non-tool-call assistant" and goes to hist_msgs, so reasoning is discarded.
    // This makes the template incompatible with reasoning needle tests. Manual tests below
    // properly test the parser's reasoning extraction capabilities.
    if (impl == chat_parser_impl::LEGACY) {
        run_template_test_suite(impl, template_caps, tmpls);
    }

    assert_equals(COMMON_CHAT_FORMAT_KIMI_K2, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
    assert_equals(COMMON_CHAT_FORMAT_KIMI_K2, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

    // Build parser with tools (always use a parser)
    common_chat_templates_inputs kimi_inputs;
    kimi_inputs.messages = {message_user};
    kimi_inputs.tools = kimi_k2_tools;
    kimi_inputs.enable_thinking = true;
    kimi_inputs.parallel_tool_calls = true;
    kimi_inputs.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
    auto kimi_params = common_chat_templates_apply(tmpls.get(), kimi_inputs);
    auto kimi_syntax = get_syntax(kimi_params);

    // Build parser with reasoning extraction enabled
    common_chat_templates_inputs kimi_inputs_reasoning;
    kimi_inputs_reasoning.messages = {message_user};
    kimi_inputs_reasoning.tools = kimi_k2_tools;
    kimi_inputs_reasoning.enable_thinking = true;
    kimi_inputs_reasoning.parallel_tool_calls = true;
    kimi_inputs_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    kimi_inputs_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
    auto kimi_params_reasoning = common_chat_templates_apply(tmpls.get(), kimi_inputs_reasoning);
    auto kimi_syntax_reasoning = get_syntax(kimi_params_reasoning, COMMON_REASONING_FORMAT_DEEPSEEK);

    // Build content-only parser (no tools) for content-only tests
    common_chat_templates_inputs kimi_inputs_content_only;
    kimi_inputs_content_only.messages = {message_user};
    kimi_inputs_content_only.enable_thinking = true;
    kimi_inputs_content_only.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
    auto kimi_params_content = common_chat_templates_apply(tmpls.get(), kimi_inputs_content_only);
    auto kimi_syntax_content = get_syntax(kimi_params_content);

    // Build content-only parser with reasoning
    common_chat_templates_inputs kimi_inputs_content_reasoning;
    kimi_inputs_content_reasoning.messages = {message_user};
    kimi_inputs_content_reasoning.enable_thinking = true;
    kimi_inputs_content_reasoning.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    kimi_inputs_content_reasoning.experimental_new_parsers = (impl == chat_parser_impl::EXPERIMENTAL);
    auto kimi_params_content_reasoning = common_chat_templates_apply(tmpls.get(), kimi_inputs_content_reasoning);
    auto kimi_syntax_content_reasoning = get_syntax(kimi_params_content_reasoning, COMMON_REASONING_FORMAT_DEEPSEEK);

    // Test parsing regular content (content-only parser)
    assert_msg_equals(message_assist,
        common_chat_parse(
            "Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            kimi_syntax_content));

    // Test parsing content with thinking (content-only parser with reasoning)
    assert_msg_equals(message_assist_thoughts,
        common_chat_parse(
            "<think>I'm\nthinking</think>Hello, world!\nWhat's up?",
            /* is_partial= */ false,
            kimi_syntax_content_reasoning));

    // Tool call and streaming tests only run with experimental PEG parsers
    // (legacy parser doesn't extract tool IDs correctly for Kimi format)
    if (impl == chat_parser_impl::EXPERIMENTAL) {
        // Test parsing tool calls (Kimi format includes tool ID after the colon)
        assert_msg_equals(message_assist_call_idx,
            common_chat_parse(
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
                /* is_partial= */ false,
                kimi_syntax));

        // Test parsing tool calls with thinking
        assert_msg_equals(message_assist_thoughts_call_idx,
            common_chat_parse(
                "<think>I'm\nthinking</think><|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
                /* is_partial= */ false,
                kimi_syntax_reasoning));

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content_idx,
            common_chat_parse(
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                kimi_syntax));

        // Test tool calls with extra content AND thinking
        assert_msg_equals(message_assist_call_thoughts_content_idx,
            common_chat_parse(
                "<think>I'm\nthinking</think><|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                kimi_syntax_reasoning));

        // Test streaming
        test_parser_with_streaming(message_assist_call_thoughts_content_idx,
        "<think>I'm\nthinking\n</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
    test_parser_with_streaming(simple_assist_msg("<think>I'm\nthinking</think>\n\n", "", "special_function", "{\"arg1\": 1}", "0"),
        "<think>I'm\nthinking</think>\n\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
    test_parser_with_streaming(message_assist_call_thoughts_content_idx,
        "<think>I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?\n\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_calls_section_end|>\n",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
    test_parser_with_streaming(simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1, \"arg2\": 2}", "0"),
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function_with_opt:0<|tool_call_argument_begin|>{\"arg1\": 1, \"arg2\": 2}<|tool_call_end|><|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
    test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": \"123456\"}", "0"),
        "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": \"123456\"}<|tool_call_end|><|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
    test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": [1, 2, \"345\", 6]}", "0"),
        "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": [1, 2, \"345\", 6]}<|tool_call_end|><|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
    test_parser_with_streaming(simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": {\"12\": 34, \"5\": [67, 8], \"9\": \"10\"}}", "0"),
        "<think>I'm\nthinking</think>Hello, world!\nWhat's up?\n<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{\"arg1\": {\"12\": 34, \"5\": [67, 8], \"9\": \"10\"}}<|tool_call_end|><|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
    test_parser_with_streaming(
            simple_assist_msg("", "", "complex_function", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}", "0"),
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function:0<|tool_call_argument_begin|>"
            "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
            "<|tool_call_end|><|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
    test_parser_with_streaming(
            simple_assist_msg("", "", "web_search", "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}", "0"),
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:0<|tool_call_argument_begin|>"
            "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}"
            "<|tool_call_end|><|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
    test_parser_with_streaming(
            simple_assist_msg("", "", "read_file", "{\"args\": [{\"path\": \"src/providers/ThemeProvider.tsx\"}, {\"path\": \"src/components/Header.tsx\"}, {\"path\": \"src/components/ThemeToggle.tsx\"}, {\"path\": \"src/app/globals.css\"}, {\"path\": \"src/app/layout.tsx\"}]}", "0"),
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
            "{\"args\": [{\"path\": \"src/providers/ThemeProvider.tsx\"}, {\"path\": \"src/components/Header.tsx\"}, {\"path\": \"src/components/ThemeToggle.tsx\"}, {\"path\": \"src/app/globals.css\"}, {\"path\": \"src/app/layout.tsx\"}]}"
            "<|tool_call_end|><|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
    test_parser_with_streaming(
            simple_assist_msg(
                    "Let me start by examining the relevant files to understand the current implementation.", "",
                    "read_file",
                    "{\"files\": [{\"path\": \"src/app/Partners.tsx\", \"line_ranges\": [\"1-100\"]}]}", "0"),
            "Let me start by examining the relevant files to understand the current implementation."
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
            "{\"files\":[{\"path\":\"src/app/Partners.tsx\",\"line_ranges\":[\"1-100\"]}]}"
            "<|tool_call_end|><|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax); });
    auto multi_tool_msg = simple_assist_msg("Let me call multiple tools.", "I'm thinking.");
    multi_tool_msg.tool_calls.push_back({ "read_file", "{\"files\": [{\"path\": \"src/app/Partners.tsx\", \"line_ranges\": [\"1-100\"]}]}", "0" });
    multi_tool_msg.tool_calls.push_back({ "web_search", "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}", "1" });
    multi_tool_msg.tool_calls.push_back({ "complex_function", "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}", "2" });
    multi_tool_msg.tool_calls.push_back({ "emoji_function", "{\"message\":\"Hello! 👋 🌟 🚀 Testing emojis: 😀😃😄😁 and symbols: ∑∏∆∇\"}", "3" });
    test_parser_with_streaming(multi_tool_msg,
            "<think>I'm thinking.</think>Let me call multiple tools."
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
            "{\"files\":[{\"path\":\"src/app/Partners.tsx\",\"line_ranges\":[\"1-100\"]}]}"
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.web_search:1<|tool_call_argument_begin|>"
            "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}"
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.complex_function:2<|tool_call_argument_begin|>"
            "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.emoji_function:3<|tool_call_argument_begin|>"
            "{\"message\":\"Hello! 👋 🌟 🚀 Testing emojis: 😀😃😄😁 and symbols: ∑∏∆∇\"}"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>",
        [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
    } // end experimental parser tests

    // TODO: These tests are for tool calls embedded in <think> blocks, which is an edge case
    // that requires special parser handling not yet implemented. The parser currently
    // treats all content inside <think>...</think> as reasoning_content.
    // test_parser_with_streaming(
    //         simple_assist_msg("", "I'm thinking", "complex_function_in_think", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"),
    //         "<think>I'm thinking<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function_in_think:0<|tool_call_argument_begin|>"
    //         "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
    //         "<|tool_call_end|><|tool_calls_section_end|>",
    //     [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });
    // test_parser_with_streaming(
    //         simple_assist_msg("Hello", "I'm thinkingI'm still thinking", "complex_function_in_think", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"),
    //         "<think>I'm thinking<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function_in_think:0<|tool_call_argument_begin|>"
    //         "{\"name\": \"John Doe\", \"age\": 30, \"active\": true, \"score\": 95.5}"
    //         "<|tool_call_end|><|tool_calls_section_end|>I'm still thinking</think>Hello",
    //     [&](const std::string &msg) { return common_chat_parse(msg, /* is_partial= */ true, kimi_syntax_reasoning); });

    // Test template rendering
    common_chat_templates_inputs conversation_with_tools = inputs_tools;
    conversation_with_tools.messages.push_back(simple_assist_msg("Let's do it", "Think first", "complex_function", "{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}"));
    conversation_with_tools.messages.push_back({
        "tool",
        "Tool response 1",
        /* .content_parts = */ {},
        /* .tool_calls = */ {},
        /* .reasoning_content = */ "",
        /* .tool_name = */ "complex_function",
        /* .tool_call_id = */ "",
    });
    conversation_with_tools.messages.push_back(simple_assist_msg("Continue", "Think next", "web_search", "{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}"));
    conversation_with_tools.messages.push_back({
        "tool",
        "Tool response 2",
        /* .content_parts = */ {},
        /* .tool_calls = */ {},
        /* .reasoning_content = */ "",
        /* .tool_name = */ "web_search",
        /* .tool_call_id = */ "",
    });
    conversation_with_tools.messages.push_back(simple_assist_msg("CC", "Think last", "read_file", "{\"args\": [{\"path\": \"src/providers/ThemeProvider.tsx\"}, {\"path\": \"src/components/Header.tsx\"}, {\"path\": \"src/components/ThemeToggle.tsx\"}, {\"path\": \"src/app/globals.css\"}, {\"path\": \"src/app/layout.tsx\"}]}"));
    conversation_with_tools.messages.push_back({
        "tool",
        "Tool response 3",
        /* .content_parts = */ {},
        /* .tool_calls = */ {},
        /* .reasoning_content = */ "",
        /* .tool_name = */ "read_file",
        /* .tool_call_id = */ "",
    });
    // TODO(ochafik): Fix (regression?)
    // assert_equals(common_chat_templates_apply(tmpls.get(), conversation_with_tools).prompt, std::string("<|im_system|>tool_declare<|im_middle|>[{\"type\": \"function\", \"function\": {\"name\": \"special_function\", \"description\": \"I'm special\", \"parameters\": {\"type\": \"object\", \"properties\": {\"arg1\": {\"type\": \"integer\", \"description\": \"The arg.\"}}, \"required\": [\"arg1\"]}}}]<|im_end|><|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|><|im_user|>user<|im_middle|>Hey there!<|im_end|><|im_assistant|>assistant<|im_middle|><think>Think first</think>Let's do it<|tool_calls_section_begin|><|tool_call_begin|>functions.complex_function:0<|tool_call_argument_begin|>{\"name\":\"John Doe\",\"age\":30,\"active\":true,\"score\":95.5}<|tool_call_end|><|tool_calls_section_end|><|im_end|><|im_system|>complex_function<|im_middle|>## Return of functions.complex_function:0\nTool response 1<|im_end|><|im_assistant|>assistant<|im_middle|><think>Think next</think>Continue<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:1<|tool_call_argument_begin|>{\"query\":\"\\\"From Zero\\\" Linkin Park album tracklist complete songs\",\"limit\":3,\"type\":\"text\"}<|tool_call_end|><|tool_calls_section_end|><|im_end|><|im_system|>web_search<|im_middle|>## Return of functions.web_search:1\nTool response 2<|im_end|><|im_assistant|>assistant<|im_middle|><think>Think last</think>CC<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:2<|tool_call_argument_begin|>{\"args\": [{\"path\": \"src/providers/ThemeProvider.tsx\"}, {\"path\": \"src/components/Header.tsx\"}, {\"path\": \"src/components/ThemeToggle.tsx\"}, {\"path\": \"src/app/globals.css\"}, {\"path\": \"src/app/layout.tsx\"}]}<|tool_call_end|><|tool_calls_section_end|><|im_end|><|im_system|>read_file<|im_middle|>## Return of functions.read_file:2\nTool response 3<|im_end|><|im_assistant|>assistant<|im_middle|>"));

    // Test template generation for regular content
    test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools,
                    "<think></think>Hello, world!\nWhat's up?",
                    /* expect_grammar_triggered= */ false);

    // Tool call tests require PEG parser for correct ID extraction
    if (impl == chat_parser_impl::EXPERIMENTAL) {
        // Test template generation for tool calls (Kimi format includes ID after colon)
        // Note: JSON formatting may vary, so we skip delta comparison and just test parsing
        test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call_idx, tools,
                        /* expected_delta= */ "",
                        /* expect_grammar_triggered= */ true,
                        /* test_grammar_if_triggered= */ true,
                        /* reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                        /* ignore_whitespace_differences= */ true
        );

        // Test template generation for tools with optional parameters
        test_templates(impl, tmpls.get(), template_caps.end_tokens, simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1}", "0"), tools,
                        /* expected_delta= */ "",
                        /* expect_grammar_triggered= */ true,
                        /* test_grammar_if_triggered= */ true,
                        /* reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                        /* ignore_whitespace_differences= */ true
        );
        test_templates(impl, tmpls.get(), template_caps.end_tokens, simple_assist_msg("", "", "special_function_with_opt", "{\"arg1\": 1, \"arg2\": 2}", "0"), tools,
                        /* expected_delta= */ "",
                        /* expect_grammar_triggered= */ true,
                        /* test_grammar_if_triggered= */ true,
                        /* reasoning_format= */ COMMON_REASONING_FORMAT_DEEPSEEK,
                        /* ignore_whitespace_differences= */ true
        );
    }
}
