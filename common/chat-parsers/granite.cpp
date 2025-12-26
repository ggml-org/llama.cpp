// Granite tool call format
// Format: {"tool_calls": [{"name": "func", "arguments": {...}}], "content": "..."}
// With optional <think>...</think> and <response>...</response> tags

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_granite_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Pass thinking context for Granite template
    json additional_context = {
        {"thinking", inputs.enable_thinking},
    };

    data.prompt = apply(tmpl, inputs, /* messages_override= */ std::nullopt, /* tools_override= */ std::nullopt, additional_context);
    data.format = COMMON_CHAT_FORMAT_GRANITE;

    if (string_ends_with(data.prompt, "<think>\n") || string_ends_with(data.prompt, "<think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<response>",
        "</response>",
        "<|end_of_text|>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        auto consume_eot = [&]() {
            return p.optional(p.literal("<|end_of_text|>")) + p.optional(p.space());
        };

        auto reasoning = p.eps();
        if (inputs.enable_thinking && extract_reasoning) {
            auto reasoning_content = p.tag(Tag::REASONING, p.until("</think>")) + ("</think>" | p.end());
            if (data.thinking_forced_open) {
                reasoning = reasoning_content;
            } else {
                reasoning = p.optional("<think>" + reasoning_content);
            }
        }

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return reasoning << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser: Granite emits <|tool_call|>[{"name": "func", "arguments": {...}}]
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {

            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                if (data.grammar.find("<|tool_call|>") != std::string::npos) {
                    data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|tool_call|>"});
                }
            }
                
            auto tool_call = p.tag(Tag::TOOL,
                p.atomic_tag(Tag::TOOL_OPEN, p.literal("<|tool_call|>"))
                + p.tag(Tag::TOOL_ARGS, p.json())
            );

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, min_calls, max_calls));

            bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            if (require_tools) {
                return reasoning << tool_calls << consume_eot();
            }
            return reasoning << p.tag(Tag::CONTENT, p.until("<|tool_call|>")) << tool_calls << consume_eot();
        }

        // Content-only parser: trim trailing <|end_of_text|> and optionally handle <response> blocks
        auto response_block = p.literal("<response>") + p.tag(Tag::CONTENT, p.until("</response>")) + (p.literal("</response>") | p.end());
        auto content_until_eot = p.tag(Tag::CONTENT, p.until("<|end_of_text|>")) << consume_eot();

        include_grammar = false;
        return reasoning << p.choice({response_block, content_until_eot, p.tag(Tag::CONTENT, p.rest())});
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    
    return data;
}
