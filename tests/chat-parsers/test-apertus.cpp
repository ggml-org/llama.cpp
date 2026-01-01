#include "../test-chat.h"

void test_apertus_parser(chat_parser_impl impl)
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
        template_caps.name = "Apertus";
        template_caps.jinja_path = "models/templates/Apertus-8B-Instruct.jinja";
        template_caps.legacy_format = COMMON_CHAT_FORMAT_APERTUS;
        template_caps.experimental_format = COMMON_CHAT_FORMAT_PEG_NATIVE;
        template_caps.supports_thinking = ThinkingSupport::Yes;
        template_caps.think_open_tag = "<|inner_prefix|>";
        template_caps.think_close_tag = "<|inner_suffix|>";
        template_caps.reasoning_requires_tools = ReasoningRequiresTools::No;
        template_caps.tools_emit_content_with_calls = ToolsEmitContentWithCalls::Yes;
        template_caps.inject_reasoning_after_format = InjectReasoningAfterFormat::No;
        template_caps.supports_disable_thinking = SupportsDisableThinking::Yes;
        template_caps.supports_reasoning_only = SupportsReasoningOnly::Yes;
        template_caps.end_tokens = {"<|assistant_end|>" };

        auto tmpls = read_templates(template_caps.jinja_path);
        run_template_test_suite(impl, template_caps, tmpls);


        assert_equals(COMMON_CHAT_FORMAT_APERTUS, common_chat_templates_apply(tmpls.get(), inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_APERTUS, common_chat_templates_apply(tmpls.get(), inputs_tools).format);

        // Test parsing regular content
        assert_msg_equals(message_assist,
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_APERTUS}));

        // Test parsing content with thinking
        assert_msg_equals(message_assist_thoughts,
            common_chat_parse(
                "<|inner_prefix|>I'm\nthinking<|inner_suffix|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
                }));

        // Test parsing tool calls
        assert_msg_equals(message_assist_call,
            common_chat_parse(
                "<|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_APERTUS}));

        // Test parsing tool calls with thinking
        assert_msg_equals(message_assist_call_thoughts,
            common_chat_parse(
                "<|inner_prefix|>I'm\nthinking<|inner_suffix|><|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));

        // Test tool calls with extra content
        assert_msg_equals(message_assist_call_content,
            common_chat_parse(
                "<|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {COMMON_CHAT_FORMAT_APERTUS}
            ));

        // Test tool calls with extra content AND thinking
        assert_msg_equals(message_assist_call_thoughts_content,
            common_chat_parse(
                "<|inner_prefix|>I'm\nthinking<|inner_suffix|><|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>Hello, world!\nWhat's up?",
                /* is_partial= */ false,
                {
                    /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
                    /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK
                }));


//         assert_msg_equals(
//             simple_assist_msg("", "I'm\nthinking", "", ""),
//             common_chat_parse(
//                 "<|tools_prefix|>[ { \"test\" : { \"success\" : true } } ] <|tools_suffix|>",
//                 /* is_partial= */ false,
//                 {
//                     /* .format = */ COMMON_CHAT_FORMAT_APERTUS,
//                     /* .reasoning_format = */ COMMON_REASONING_FORMAT_DEEPSEEK,
//                 }));

// res  remove_waiti: remove task 0 from waiting list. current waiting = 1 (before remove)
// srv          stop: cancel task, id_task = 0
// res  remove_waiti: remove task 0 from waiting list. current waiting = 0 (before remove)
// que          post: new task, id = 70/1, front = 1
// que    start_loop: processing new tasks
// que    start_loop: processing task, id = 70
// que    start_loop: update slots
// srv  update_slots: all slots are idle
// que    start_loop: waiting for new tasks
// srv    operator(): got exception: {"error":{"code":500,"message":"Failed to parse input at pos 0","type":"server_error"}}
// srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 500
// srv  log_server_r: request:  {"max_tokens": 512, "messages": [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "Write an example"}], "tool_choice": "required", "tools": [{"type": "function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {"success": {"type": "boolean", "const": true}}, "required": ["success"]}}}], "parallel_tool_calls": false, "stream": false}

        // Test template generation for regular content
        test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist, tools,
                      "Hello, world!\nWhat's up?",
                      /* expect_grammar_triggered= */ false);

        // Test template generation for tool calls
        test_templates(impl, tmpls.get(), template_caps.end_tokens, message_assist_call, tools,
                      "<|tools_prefix|>[{\"special_function\": {\"arg1\": 1}}]<|tools_suffix|>",
                      /* expect_grammar_triggered= */ true
        );

        assert_equals(true, common_chat_templates_support_enable_thinking(tmpls.get()));
    }
}
