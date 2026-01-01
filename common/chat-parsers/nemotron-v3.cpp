// Nemotron 3 Nano 30B A3B tool call format
// Format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
// With optional <think>...</think> reasoning blocks

#include "chat-parsers-internal.h"
#include <optional>

common_chat_params common_chat_params_init_nemotron_v3_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<toolcall>",
        "</toolcall>",
        "<SPECIAL_11>Assistant",
        "<SPECIAL_11>User",
        "<SPECIAL_12>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;

    bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;
        auto newline = p.choice({p.literal("\r\n"), p.literal("\n")});
        auto whitespace = p.repeat(p.choice({newline, p.literal(" "), p.literal("\t")}), 0, -1);
        auto assistant_header = p.literal("<|im_start|>assistant") + p.choice({p.literal("\r\n"), p.literal("\n")});
        auto assistant_prefix = whitespace + p.optional(assistant_header);
        auto assistant_suffix = whitespace + p.optional(p.literal("<|im_end|>")) + whitespace;
        const auto & after_reasoning_gap = whitespace;
        auto think_open = p.literal("<think>") + p.optional(newline);
        auto think_close = p.literal("</think>");
        auto reasoning = p.eps();
        if (inputs.enable_thinking && extract_reasoning) {
            auto reasoning_content = p.tag(Tag::REASONING, p.until("</think>")) + think_close;
            if (data.thinking_forced_open) {
                reasoning = reasoning_content;
            } else {
                reasoning = p.optional(think_open + reasoning_content);
            }
        } else {
            if (data.thinking_forced_open) {
                reasoning = p.until("</think>") + think_close;
            } else {
                reasoning = p.optional(think_open + p.until("</think>") + think_close);
            }
        }

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return assistant_prefix + reasoning + after_reasoning_gap + p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema)) + assistant_suffix;
        }

        // Tool call parser
        if (has_tools) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers = {
                    {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>"}
                };
            }

            generic_tool_call_format format;
            format.tool_call_start = p.space() + "<tool_call>" + p.space() + "<function=";
            format.tool_call_name_params_sep = ">" + p.space();
            format.tool_call_end = "</function>" + p.space() + "</tool_call>" + p.space();
            format.param_start = p.literal("<parameter=");
            format.param_name_value_sep = ">" + p.space();
            // Note: The leading \n is consumed by the space() in the value parser (space_around_json=true),
            // so param_ends should NOT include it. The trailing \n should be included to consume it.
            format.param_ends = { "</parameter>\n", "</parameter>" };
            auto tool_calls = build_generic_tool_calls_peg_parser(p, inputs, format);

            auto stop_before = std::vector<std::string>{
                "\n<tool_call>", "\r\n<tool_call>", "<tool_call>",
                "\n<toolcall>", "\r\n<toolcall>", "<toolcall>"
            };
            auto stop_after = std::vector<std::string>{
                "\n<|im_end|>", "\r\n<|im_end|>", "<|im_end|>"
            };
            auto content_before = p.optional(p.tag(Tag::CONTENT, p.until_one_of(stop_before)));
            auto content_after = p.optional(p.tag(Tag::CONTENT, p.until_one_of(stop_after)));
            auto pre_tool_gap = p.repeat(newline, 0, -1);
            if (require_tools) {
                // Simplified: just space + tool_calls, no extra patterns
                return p.space() + tool_calls;
            }
            return assistant_prefix + reasoning + after_reasoning_gap + content_before + pre_tool_gap + tool_calls + content_after + assistant_suffix;
        }

        // Content only parser
        include_grammar = false;
        // Handle reasoning only when enabled, otherwise just capture all content
        if (inputs.enable_thinking && extract_reasoning) {
            return reasoning + after_reasoning_gap + p.tag(Tag::CONTENT, p.rest());
        }
        return p.tag(Tag::CONTENT, p.rest());
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;

    return data;
}
