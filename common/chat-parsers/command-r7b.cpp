// Command R7B tool call format
// Format: <|START_THINKING|>...<|END_THINKING|><|START_ACTION|>[{"tool_call_id":"1","tool_name":"func","parameters":{}}]<|END_ACTION|>

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_command_r7b_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto has_reasoning_content = msg.contains("reasoning_content") && msg.at("reasoning_content").is_string();
        auto has_tool_calls = msg.contains("tool_calls") && msg.at("tool_calls").is_array();
        if (has_reasoning_content && has_tool_calls) {
            auto adjusted_message = msg;
            adjusted_message["tool_plan"] = msg.at("reasoning_content");
            adjusted_message.erase("reasoning_content");
            adjusted_messages.push_back(adjusted_message);
        } else {
            adjusted_messages.push_back(msg);
        }
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

    data.format = COMMON_CHAT_FORMAT_COMMAND_R7B;
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

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                    std::string(data.thinking_forced_open ? "[\\s\\S]*?(<\\|END_THINKING\\|>\\s*)" : "(?:<\\|START_THINKING\\|>[\\s\\S]*?<\\|END_THINKING\\|>\\s*)?") +
                        "(<\\|START_ACTION\\|>)[\\s\\S]*"
                });
            }

            // Tool call: <|START_ACTION|>[...json array...]<|END_ACTION|>
            auto tool_call = p.tag(Tag::TOOL,
                p.atomic_tag(Tag::TOOL_OPEN, p.literal("<|START_ACTION|>"))
                + p.tag(Tag::TOOL_ARGS, p.json())  // JSON array with tool calls
                + p.atomic_tag(Tag::TOOL_CLOSE, p.literal("<|END_ACTION|>"))
            );

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, min_calls, max_calls));

            if (require_tools) {
                return reasoning << response_block << tool_calls << p.optional(p.rest());
            }

            return reasoning << response_block << tool_calls << p.optional(p.rest());
        }

        // Content only parser
        return reasoning << response_block << p.optional(p.rest());
    });

    common_chat_build_peg_grammar(inputs, parser, data);

    return data;
}
