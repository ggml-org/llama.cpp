#include "parsers.h"

// Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
// The ID contains both the function name and an incrementing counter
common_chat_params common_chat_params_init_kimi_k2(const common_chat_template &    tmpl,
                                                          const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.preserved_tokens  = {
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        "<think>",
        "</think>",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    const std::string SECTION_BEGIN = "<|tool_calls_section_begin|>";
    const std::string SECTION_END   = "<|tool_calls_section_end|>";
    const std::string CALL_BEGIN    = "<|tool_call_begin|>";
    const std::string ARGS_BEGIN    = "<|tool_call_argument_begin|>";
    const std::string CALL_END      = "<|tool_call_end|>";

    const std::string THINK_START = "<think>";
    const std::string THINK_END   = "</think>";
    const std::string GEN_PROMPT  = "<|im_assistant|>assistant<|im_middle|>";

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = {THINK_END};

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PROMPT + THINK_START + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // Kimi K2 Thinking format:
        // - Reasoning: <think>{reasoning}</think>
        // - Content: text after reasoning
        // - Tool calls section:
        //   <|tool_calls_section_begin|>
        //   <|tool_call_begin|>functions.<name>:<index><|tool_call_argument_begin|>{json_args}<|tool_call_end|>
        //   ...
        //   <|tool_calls_section_end|>
        // The ID format is: functions.<function_name>:<counter> where counter is 0, 1, 2, ...

        // Tool call markers
        auto end = p.end();

        // Note: this model is CRAZY. It can diverge from its supposed tool calling pattern in so many ways it's not funny.
        // For example, it can call tools at the end of reasoning without closing reasoning...
        auto reasoning = extract_reasoning ? p.optional(THINK_START + p.reasoning(
            p.until_one_of({ THINK_END, "<|tool_calls_section_begin|>", "<|tool_call_begin|>" })) +
            p.optional(p.literal(THINK_END))) : p.eps();
        auto generation_prompt = p.literal(GEN_PROMPT);


        // Content only parser (no tools)
        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        // Build tool call parsers for each available function
        // The ID format is: functions.<name>:<index>
        // We need to match: functions.<name>:<digits>
        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto & schema   = function.at("parameters");

            // Match: functions.<name>:<digits>
            // Capture the full call id (functions.<name>:<digits>) using tool_id tag
            auto tool_id = p.tool_id(p.literal("functions.") + p.tool_name(p.literal(name)) + p.literal(":") + p.chars("[0-9]", 1, -1));
            auto tool_parser = p.tool(
                p.tool_open(tool_id + p.literal(ARGS_BEGIN)) +
                p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)) +
                p.tool_close(p.optional((p.literal(CALL_END))))
            );

            tool_choice |= p.rule("tool-" + name, tool_parser);
        });

        // Tool calls section: <|tool_calls_section_begin|> tool_calls <|tool_calls_section_end|>
        auto min_calls  = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
        auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
        // Use trigger_rule so grammar generator knows where to start generating rules
        auto tool_calls = p.rule("tool-calls",
            p.optional(p.literal(SECTION_BEGIN)) +
            p.trigger_rule("tool-call", p.repeat(CALL_BEGIN + tool_choice, min_calls, max_calls) +
                p.optional(p.literal(SECTION_END)))
        );

        auto content_before_tools = p.content(p.until_one_of({ SECTION_BEGIN, CALL_BEGIN }));

        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|tool_call_begin|>" }
        };
    }

    return data;
}
