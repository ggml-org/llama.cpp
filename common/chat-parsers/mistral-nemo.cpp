// Mistral Nemo tool call format
// Format: [TOOL_CALLS][{"name":"func","arguments":{},"id":"abc123def"}]

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_mistral_nemo_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);

    data.preserved_tokens = {
        "[TOOL_CALLS]",
    };

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    // Build the PEG parser
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        if (has_tools) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
            }

            static const json id_schema {
                {"type", "string"},
                {"pattern", "^[a-zA-Z0-9]{9}$"},  // Enforce ID format (exactly 9 alphanumeric)
            };
            // Tool call parser: [TOOL_CALLS][{"name":"func","arguments":{},"id":"abc123def"}]
            // Mistral Nemo format with ID at end: {"name": "...", "arguments": {...}, "id": "..."}
            auto any_tool_call = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto &) {
                any_tool_call |= p.tag(Tag::TOOL, p.sequence()
                    + p.literal_tag(Tag::TOOL_OPEN, "{")
                    << "\"name\"" << ":" << ("\"" + p.literal_tag(Tag::TOOL_NAME, name) + "\"") << ","
                    << "\"arguments\"" << ":" << p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters)) << ","
                    << "\"id\"" << ":" << p.tag(Tag::TOOL_ID, p.schema(p.json(), "tool-id", id_schema))
                    << p.literal_tag(Tag::TOOL_CLOSE, "}"));
            });

            auto tool_calls = p.trigger_rule("tool-call-root",
                p.space()
                    + p.literal("[TOOL_CALLS][")
                    + any_tool_call + p.repeat(p.literal(",") << any_tool_call, 0, inputs.parallel_tool_calls ? -1 : 0)
                    + p.literal("]"));

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return tool_calls;
            }
            return p.tag(Tag::CONTENT, p.until("[TOOL_CALLS]")) << tool_calls;
        }

        // Content only parser
        return p.tag(Tag::CONTENT, p.rest());
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    return data;
}
