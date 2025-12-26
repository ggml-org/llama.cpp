// Mistral Nemo tool call format
// Format: [TOOL_CALLS][{"name":"func","arguments":{},"id":"abc123def"}]

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_mistral_nemo_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_MISTRAL_NEMO;

    data.preserved_tokens = {
        "[TOOL_CALLS]",
    };

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    // Build the PEG parser
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
            }
            // Tool call parser: [TOOL_CALLS] followed by a JSON array of tool calls
            // The template generates: [TOOL_CALLS][{"name": "fn1", ...}, {"name": "fn2", ...}]
            // So we capture [TOOL_CALLS] once, then the entire JSON array
            auto tool_call = p.tag(Tag::TOOL,
                p.atomic_tag(Tag::TOOL_OPEN, p.literal("[TOOL_CALLS]"))
                + p.tag(Tag::TOOL_ARGS, p.json())
            );

            // No repeat needed - [TOOL_CALLS] appears once with the entire array
            auto tool_calls = p.trigger_rule("tool-call-root", tool_call);

            bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            if (require_tools) {
                return tool_calls;
            }
            return p.tag(Tag::CONTENT, p.until("[TOOL_CALLS]")) << tool_calls;
        }

        // Content only parser
        return p.tag(Tag::CONTENT, p.rest());
    });

    common_chat_build_peg_grammar(inputs, parser, data);

    return data;
}
