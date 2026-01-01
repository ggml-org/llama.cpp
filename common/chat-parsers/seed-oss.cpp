// Seed OSS tool call format
// Format: <seed:tool_call><function=name><parameter=key>value</parameter></function></seed:tool_call>
// With optional <seed:think>...</seed:think> reasoning blocks

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_seed_oss_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<seed:think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</seed:think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    data.preserved_tokens = {
        "<seed:think>",
        "</seed:think>",
        "<seed:tool_call>",
        "</seed:tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
        "<seed:eos>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        auto eos = p.space() + p.optional(p.literal("<seed:eos>")) + p.space();

        auto reasoning_block = p.literal("<seed:think>")
            + p.tag(Tag::REASONING, p.until("</seed:think>"))
            + (p.literal("</seed:think>") | p.end());
        auto reasoning = extract_reasoning && inputs.enable_thinking && data.thinking_forced_open
            ? reasoning_block
            : p.optional(reasoning_block);

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return reasoning << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema)) << p.space();
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers = {
                    {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<seed:tool_call>"}
                };
            }

            generic_tool_call_format format;
            format.tool_call_start = p.space() + "<seed:tool_call>\n<function=";
            format.tool_call_name_params_sep = p.literal(">\n");
            format.tool_call_end = "</function>" + p.space() + "</seed:tool_call>";
            format.param_start = p.literal("<parameter=");
            format.param_name_value_sep = p.literal(">");
            format.param_ends = { "</parameter>\n", "</parameter>" };
            auto tool_calls = build_generic_tool_calls_peg_parser(p, inputs, format);

            // Use original until_one_of patterns to avoid capturing trailing newlines in content
            auto content_before = p.optional(p.tag(Tag::CONTENT, p.until_one_of({
                "\r\n\r\n<seed:tool_call>", "\n\n<seed:tool_call>",
                "\r\n<seed:tool_call>", "\n<seed:tool_call>", "<seed:tool_call>"
            })));
            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return reasoning << p.space() << tool_calls << eos;
            }
            auto with_tools = content_before << p.space() << tool_calls << eos;
            // Content-only fallback: optional content until eos, then optional rest
            auto content_until_eos = p.optional(p.tag(Tag::CONTENT, p.until_one_of({
                "\r\n\r\n<seed:eos>", "\n\n<seed:eos>",
                "\r\n<seed:eos>", "\n<seed:eos>", "<seed:eos>"
            })));
            auto content_rest = p.optional(p.tag(Tag::CONTENT, p.rest()));
            auto content_only = content_until_eos << content_rest << eos;
            return reasoning << p.choice({with_tools, content_only});
        }

        // Content only parser: optional content until eos, then optional rest
        auto content_until_eos = p.optional(p.tag(Tag::CONTENT, p.until_one_of({
            "\r\n\r\n<seed:eos>", "\n\n<seed:eos>",
            "\r\n<seed:eos>", "\n<seed:eos>", "<seed:eos>"
        })));
        auto content_rest = p.optional(p.tag(Tag::CONTENT, p.rest()));
        return reasoning << content_until_eos << content_rest << eos;
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;

    return data;
}
