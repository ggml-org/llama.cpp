// Kimi K2 tool call format
// Format: <|tool_calls_section_begin|><|tool_call_begin|>function_name<|tool_call_argument_begin|>{"key": value}<|tool_call_end|><|tool_calls_section_end|>
// With optional <think>...</think> reasoning blocks

#include "chat-parsers-internal.h"
#include "chat.h"

common_chat_params common_chat_params_init_kimi_k2_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
        "<|im_end|>",
        "<|im_system|>",
        "<|im_middle|>",
    };

    data.additional_stops.insert(data.additional_stops.end(), {
        "<|im_end|>",
        "<|im_middle|>"
    });

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;
        auto optional_newline = [&]() {
            return p.optional(p.literal("\n"));
        };

        auto reasoning = p.eps();
        if (inputs.enable_thinking && extract_reasoning) {
            auto reasoning_content = p.tag(Tag::REASONING, p.until("</think>")) + ("</think>" | p.end());
            reasoning = p.optional(optional_newline() + "<think>" + reasoning_content);
        }

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return reasoning << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        // Format: <|tool_call_begin|>functions.{name}:{counter}<|tool_call_argument_begin|>{...}<|tool_call_end|>
        bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|tool_calls_section_begin|>"});
            }

            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](const auto &, const auto & name, const auto & parameters, const auto &) {
                // Match: functions.{name}:{id}
                // Counter must be one or more digits (matching original [0-9]+ pattern)
                // Use atomic_tag to ensure tool calls are only created when fully matched
                auto tool_open = p.literal("<|tool_call_begin|>")
                    + "functions." + p.literal_tag(Tag::TOOL_NAME, name) + ":"
                    + p.tag(Tag::TOOL_ID, p.chars("[0-9]", 1, 10))
                    + "<|tool_call_argument_begin|>";
                auto tool_close = p.literal("<|tool_call_end|>");
                auto tool_args = p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters));

                tool_choice |= p.rule("tool-" + name,
                    p.atomic_tag(Tag::TOOL_OPEN, tool_open)
                    + tool_args
                    + p.atomic_tag(Tag::TOOL_CLOSE, tool_close));
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = "<|tool_calls_section_begin|>"
                + p.repeat(tool_choice, min_calls, max_calls)
                + "<|tool_calls_section_end|>";

            auto content_before = optional_newline() + p.tag(Tag::CONTENT, p.until("<|tool_calls_section_begin|>"));
            auto content_after = optional_newline() + p.tag(Tag::CONTENT, p.rest());
            if (require_tools) {
                return p.space() + reasoning + tool_calls;
            }
            return reasoning << content_before << tool_calls << content_after;
        }

        // Content only parser
        include_grammar = false;
        return reasoning << optional_newline() << p.tag(Tag::CONTENT, p.rest());
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    return data;
}
