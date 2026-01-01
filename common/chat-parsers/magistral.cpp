// Magistral tool call format
// Format: [THINK]...[/THINK][TOOL_CALLS][{"name":"func","arguments":{},"id":"abc123def"}]

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_magistral_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);

    data.preserved_tokens = {
        "[THINK]",
        "[/THINK]",
    };

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    // Build the PEG parser
    bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // Optional reasoning block
        auto reasoning = extract_reasoning
            ? p.optional("[THINK]" + p.tag(Tag::REASONING, p.until("[/THINK]")) + "[/THINK]")
            : p.eps();

        // Response format parser (json_schema support)
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return reasoning << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        if (has_tools) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
                data.preserved_tokens.push_back("[TOOL_CALLS]");
            }

            // Template format: [TOOL_CALLS]name[ARGS]{...}
            auto any_tool_call = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto &) {
                any_tool_call |= p.tag(Tag::TOOL, p.sequence()
                    + p.literal_tag(Tag::TOOL_OPEN, "[TOOL_CALLS]")
                    // Wrap name + delimiter in atomic so TOOL_NAME isn't emitted prematurely
                    // when one tool name is a prefix of another (e.g., special_function vs special_function_with_opt).
                    + p.atomic(p.literal_tag(Tag::TOOL_NAME, name) + p.literal("[ARGS]"))
                    + p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters))
                    + p.literal_tag(Tag::TOOL_CLOSE, ""));
            });

            auto tool_calls = p.trigger_rule("tool-call-root",
                p.space()
                    + any_tool_call + p.repeat(any_tool_call, 0, inputs.parallel_tool_calls ? -1 : 0));

            if (require_tools) {
                return reasoning << tool_calls;
            }
            // Allow either: content before tool calls, or content only
            auto content_before = p.tag(Tag::CONTENT, p.until("[TOOL_CALLS]"));
            auto with_tools = content_before << tool_calls;
            auto content_only = p.tag(Tag::CONTENT, p.rest());
            return reasoning << p.choice({with_tools, content_only});
        }

        // Content only parser
        return reasoning << p.tag(Tag::CONTENT, p.rest());
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    return data;
}
