// Xiaomi MiMo tool call format
// Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_xiaomi_mimo_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_XIAOMI_MIMO;

    data.preserved_tokens = {
        "<tool_call>",
        "</tool_call>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto include_grammar = true;

    bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        // Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>"});
            }

            auto tool_call = p.tag(Tag::TOOL,
                p.atomic_tag(Tag::TOOL_OPEN, p.literal("<tool_call>\n"))
                + p.tag(Tag::TOOL_ARGS, p.json())
                + p.atomic_tag(Tag::TOOL_CLOSE, p.literal("\n</tool_call>"))
            );

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, min_calls, max_calls));

            if (require_tools) {
                return tool_calls;
            }
            return p.tag(Tag::CONTENT, p.until("<tool_call>")) << tool_calls;
        }

        // Content only parser
        include_grammar = false;
        return p.tag(Tag::CONTENT, p.rest());
    });

    common_chat_build_peg_grammar(inputs, parser, data);

    return data;
}
