// Command R7B tool call format
// Format: <|START_THINKING|>...<|END_THINKING|><|START_ACTION|>[{"tool_call_id":"1","tool_name":"func","parameters":{}}]<|END_ACTION|>

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_command_r7b_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto has_reasoning_content = msg.contains("reasoning_content") && msg.at("reasoning_content").is_string();
        auto has_tool_calls = msg.contains("tool_calls") && msg.at("tool_calls").is_array();
        auto adjusted_message = msg;
        if (has_reasoning_content && has_tool_calls) {
            adjusted_message["tool_plan"] = msg.at("reasoning_content");
            adjusted_message.erase("reasoning_content");
        }
        adjusted_messages.push_back(adjusted_message);
    }
    data.prompt = apply(tmpl, inputs, /* messages_override= */ adjusted_messages);

    if (string_ends_with(data.prompt, "<|START_THINKING|>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "<|END_THINKING|>";
        } else {
            data.thinking_forced_open = true;
        }
    } else if (!inputs.enable_thinking && string_ends_with(data.prompt, "<|CHATBOT_TOKEN|>")) {
        data.prompt += "<|START_THINKING|><|END_THINKING|>";
    }

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    data.preserved_tokens = {
        "<|START_ACTION|>",
        "<|END_ACTION|>",
        "<|START_RESPONSE|>",
        "<|END_RESPONSE|>",
        "<|START_THINKING|>",
        "<|END_THINKING|>",
    };

    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    // Build PEG parser
    const bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        auto response_block = p.optional(
            p.optional(p.literal("<|START_OF_TURN_TOKEN|>"))
            + p.optional(p.literal("<|CHATBOT_TOKEN|>"))
            + (p.literal("<|START_RESPONSE|>") | p.literal("RESPONSE|>"))
            + p.tag(Tag::CONTENT, p.until_one_of({"<|END_RESPONSE|>", "END_RESPONSE|>"}))
            + (p.literal("<|END_RESPONSE|>") | p.literal("END_RESPONSE|>"))
        );

        // Always handle thinking block (consume tags even if not extracting reasoning)
        auto reasoning = p.eps();
        if (data.thinking_forced_open) {
            // Thinking was already started by template
            if (extract_reasoning) {
                reasoning = p.tag(Tag::REASONING, p.until("<|END_THINKING|>")) + "<|END_THINKING|>";
            } else {
                reasoning = p.until("<|END_THINKING|>") + "<|END_THINKING|>";
            }
        } else {
            if (extract_reasoning) {
                reasoning = p.optional("<|START_THINKING|>" + p.tag(Tag::REASONING, p.until("<|END_THINKING|>")) + "<|END_THINKING|>");
            } else {
                reasoning = p.optional("<|START_THINKING|>" + p.until("<|END_THINKING|>") + "<|END_THINKING|>");
            }
        }

        // Response format parser (json_schema support)
        // Note: template wraps response in RESPONSE tags even for json_schema
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            auto json_response = p.optional(
                p.optional(p.literal("<|START_OF_TURN_TOKEN|>"))
                + p.optional(p.literal("<|CHATBOT_TOKEN|>"))
                + (p.literal("<|START_RESPONSE|>") | p.literal("RESPONSE|>"))
                + p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema))
                + (p.literal("<|END_RESPONSE|>") | p.literal("END_RESPONSE|>"))
            );
            return reasoning << json_response << p.optional(p.rest());
        }

        const auto eot = p.optional(p.literal("<|END_OF_TURN_TOKEN|>"));

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                    std::string(data.thinking_forced_open ? "[\\s\\S]*?(<\\|END_THINKING\\|>\\s*)" : "(?:<\\|START_THINKING\\|>[\\s\\S]*?<\\|END_THINKING\\|>\\s*)?") +
                        "(<\\|START_ACTION\\|>)[\\s\\S]*"
                });
            }

            // Format: <|START_ACTION|>[{"tool_call_id": "1", "tool_name": "func", "parameters": {...}}]<|END_ACTION|>
            static const json id_schema {
                {"type", "string"},
                // Command-R's template expects an integer string.
                {"pattern", "^[0-9]{1,10}$"},
            };
            // Command R7B: {"tool_call_id": "...", "tool_name": "...", "parameters": {...}}
            auto any_tool_call = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto &) {
                using Tag = common_chat_peg_tag;
                any_tool_call |= p.tag(Tag::TOOL, p.sequence()
                    + p.literal_tag(Tag::TOOL_OPEN, "{")
                    << "\"tool_call_id\"" << ":" << p.tag(Tag::TOOL_ID, p.schema(p.json(), "tool-call-id", id_schema)) << ","
                    << "\"tool_name\"" << ":" << ("\"" + p.literal_tag(Tag::TOOL_NAME, name) + "\"") << ","
                    << "\"parameters\"" << ":" << p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters))
                    << p.literal_tag(Tag::TOOL_CLOSE, "}"));
            });

            auto tool_calls =
                p.space()  // Allow optional leading whitespace
                + p.literal("<|START_ACTION|>[") + p.space()
                + any_tool_call + p.repeat(p.literal(",") + p.space() << any_tool_call, 0, inputs.parallel_tool_calls ? -1 : 0)
                + p.space() + "]<|END_ACTION|>";

            if (require_tools) {
                return reasoning << tool_calls << eot;
            }

            return reasoning << response_block << tool_calls << eot;
        }

        // Content only parser
        return reasoning << response_block << eot;
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    return data;
}
