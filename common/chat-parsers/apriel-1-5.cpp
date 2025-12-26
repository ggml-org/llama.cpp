// Apriel 1.5 tool call format
// Format: <tool_calls>[{"name": "func", "arguments": {...}}]</tool_calls>
// With optional <thinking>...</thinking> reasoning blocks

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_apriel_1_5_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_APRIEL_1_5;

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<thinking>\n") || string_ends_with(data.prompt, "<thinking>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</thinking>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    data.preserved_tokens = {
        "<thinking>",
        "</thinking>",
        "<tool_calls>",
        "</tool_calls>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    
    const bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;
        const bool has_reasoning = inputs.enable_thinking && extract_reasoning;

        auto reasoning_block = p.eps();
        if (has_reasoning) {
            auto reasoning_content = p.tag(Tag::REASONING, p.until("</thinking>")) + ("</thinking>" | p.end());
            reasoning_block = data.thinking_forced_open
                ? reasoning_content
                : p.literal("<thinking>") + reasoning_content;
        }

        auto build_content_expr = [&](const std::string & delimiter) {
            auto base_content = p.tag(Tag::CONTENT, p.until(delimiter));
            if (!has_reasoning) {
                return base_content;
            }

            auto content_before_reasoning = p.tag(Tag::CONTENT, p.until("<thinking>"));
            auto content_after_reasoning = p.tag(Tag::CONTENT, p.until(delimiter));
            auto reasoning_after_content = p.atomic(content_before_reasoning + reasoning_block + content_after_reasoning);
            auto reasoning_only = p.atomic(reasoning_block + content_after_reasoning);
            return p.choice({reasoning_after_content, reasoning_only, base_content});
        };

        auto parse_content_until = [&](const std::string & marker) {
            return p.choice({build_content_expr("\n" + marker), build_content_expr(marker)});
        };

        auto consume_end = [&]() {
            return p.optional(p.literal("\n"))
                + p.optional(p.literal("<|end|>"))
                + p.optional(p.literal("\n"));
        };

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return (has_reasoning ? p.optional(reasoning_block) : p.eps())
                << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema))
                << consume_end();
        }

        // Tool call parser
        // Format: <tool_calls>[{"name": "func", "arguments": {...}}]</tool_calls>
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_calls>"});
            }

            auto tool_call = p.tag(Tag::TOOL,
                p.atomic_tag(Tag::TOOL_OPEN, p.literal("<tool_calls>"))
                + p.tag(Tag::TOOL_ARGS, p.until("</tool_calls>"))
                + p.atomic_tag(Tag::TOOL_CLOSE, p.literal("</tool_calls>"))
            );

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, min_calls, max_calls));
            auto newline_before_tools = p.optional(p.literal("\n"));

            if (require_tools) {
                return (has_reasoning ? p.optional(reasoning_block) : p.eps())
                    << newline_before_tools
                    << tool_calls
                    << consume_end();
            }

            auto content_before_tools = parse_content_until("<tool_calls>");
            return content_before_tools << newline_before_tools << tool_calls << consume_end();
        }

        return parse_content_until("<|end|>") << consume_end();
    });

    common_chat_build_peg_grammar(inputs, parser, data);

    return data;
}
