// Firefunction V2 tool call format
// Format: functools[{"name":"func","arguments":{}}]

#include "chat-parsers-internal.h"
common_chat_params common_chat_params_init_firefunction_v2_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    const std::optional<json> tools_override = json();
    const std::optional<json> additional_context = json {
        {"datetime", format_time(inputs.now, "%b %d %Y %H:%M:%S GMT")},
        {"functions", json(inputs.tools.empty() ? "" : inputs.tools.dump(2))},
    };
    data.preserved_tokens = {
        " functools[",
    };
    data.prompt = apply(tmpl, inputs, /* messages_override =*/ std::nullopt, tools_override, additional_context);

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    // Build the PEG parser
    bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // Stop tokens for Firefunction V2
        std::vector<std::string> stop_tokens = {"<|eot_id|>", "<|start_header_id|>"};

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {            
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, " functools["});
            }

            // Tool call parser: content followed by functools[ and JSON array
            auto tool_call = p.tag(Tag::TOOL,
                p.atomic_tag(Tag::TOOL_OPEN, p.literal(" functools"))
                + p.tag(Tag::TOOL_ARGS, p.json())
            );

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, min_calls, max_calls));

            if (require_tools) {
                return tool_calls;
            }
            return p.tag(Tag::CONTENT, p.until(" functools")) + tool_calls;
        }

        // Content only parser
        return p.tag(Tag::CONTENT, p.until_one_of(stop_tokens));
    });

    data.format = COMMON_CHAT_FORMAT_FIREFUNCTION_V2;

    // Add stop tokens
    data.additional_stops = {
        "<|eot_id|>",
        "<|start_header_id|>"
    };

    common_chat_build_peg_grammar(inputs, parser, data);

    return data;
}
