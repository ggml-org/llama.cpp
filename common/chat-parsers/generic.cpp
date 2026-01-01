// Generic tool call format (fallback)
// Format: {"tool_calls": [...]} OR {"response": "..."} (not both together)
// Or plain text response without tools

#include "chat-parsers-internal.h"
#include "chat.h"

common_chat_params common_chat_params_init_generic_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Build PEG parser for generic JSON format
    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        if (has_tools) {
            static const json id_schema {
                {"type", "string"},
                {"minLength", 4},
            };
            // Tool call: [{"name": "...", "arguments": {...}, "id": "..."}]
            // Generic format with optional ID at end: {"name": "...", "arguments": {...}, "id": "..."}
            auto any_tool_call = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto &) {
                // Make ID field optional since some models don't generate it
                auto id_field = p.optional(
                    p.literal(",") << "\"id\"" << ":" << p.tag(Tag::TOOL_ID, p.schema(p.json(), "tool-id", id_schema))
                );
                any_tool_call |= p.tag(Tag::TOOL, p.sequence()
                    + p.literal_tag(Tag::TOOL_OPEN, "{")
                    << "\"name\"" << ":" << ("\"" + p.literal_tag(Tag::TOOL_NAME, name) + "\"") << ","
                    << "\"arguments\"" << ":" << p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters))
                    << id_field
                    << p.literal_tag(Tag::TOOL_CLOSE, "}"));
            });

            auto tool_calls_parser =
                p.literal("[") + p.space()
                + any_tool_call + p.repeat(p.space() + p.literal(",") + p.space() << any_tool_call, 0, inputs.parallel_tool_calls ? -1 : 0)
                + p.space() + p.literal("]");

            // Allow optional "content": "" field after tool_calls (upstream now adds this by default)
            auto optional_content_field = p.optional(
                p.literal(",") << "\"content\"" << ":" << "\"\""
            );

            auto tool_calls = p.trigger_rule("tool-call-root",
                p.space()  // Allow optional leading whitespace
                + p.literal("{")
                << "\"tool_calls\""
                << ":"
                << tool_calls_parser
                << optional_content_field
                << "}");

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                // Only tool calls allowed when required
                return tool_calls;
            }

            // Allow EITHER tool_calls OR response, but NOT both together
            auto response = p.literal("{") << "\"response\"" << ":" << p.tag(Tag::CONTENT, p.schema(p.json(), "response", json {{"type", "string"}})) << "}";
            return tool_calls | response;
        }

        // json_schema without tools - parse directly without {response: ...} wrapper
        if (!inputs.json_schema.is_null()) {
            return p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // No tools and no json_schema - just capture all content
        return p.tag(Tag::CONTENT, p.rest());
    });

    // Only add JSON format system message when tools are involved
    if (has_tools) {
        auto tweaked_messages = common_chat_template::add_system(
            inputs.messages,
            "Respond in JSON format, either {\"tool_calls\": [...]} or {\"response\": \"...\"}");
        data.prompt = apply(tmpl, inputs, /* messages_override= */ tweaked_messages);
    } else {
        data.prompt = apply(tmpl, inputs);
    }

    // ChatML-style end token (used by many templates when Generic fallback is triggered)
    data.additional_stops.push_back("<|im_end|>");

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    return data;
}
