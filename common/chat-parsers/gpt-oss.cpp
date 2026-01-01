// GPT-OSS tool call format
// Uses channel-based messaging with special tokens:
// - <|channel|>analysis, <|channel|>commentary, <|channel|>final
// - <|message|>...content...<|end|>
// - <|start|>assistant
// Tool calls format:
// - In role: to=functions.name<|channel|>analysis|commentary<|message|>{...}
// - In channel: <|channel|>analysis|commentary to=functions.name<|message|>{...}

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_gpt_oss_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Copy reasoning to the "thinking" field as expected by the gpt-oss template
    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto adjusted_message = msg;
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            adjusted_message["thinking"] = msg.at("reasoning_content");
            adjusted_message.erase("reasoning_content");
        }
        adjusted_messages.push_back(adjusted_message);
    }
    auto prompt = apply(tmpl, inputs, /* messages_override= */ adjusted_messages);

    // Check if we need to replace the return token with end token during
    // inference and without generation prompt. For more details see:
    // https://github.com/ggml-org/llama.cpp/issues/15417
    if (inputs.is_inference && !inputs.add_generation_prompt) {
        static constexpr std::string_view return_token = "<|return|>";
        static constexpr std::string_view end_token    = "<|end|>";
        if (size_t pos = prompt.rfind(return_token); pos != std::string::npos) {
            prompt.replace(pos, return_token.length(), end_token);
        }
    }

    data.prompt = prompt;

    // These special tokens are required to parse properly, so we include them
    // even if parse_tool_calls is false.
    data.preserved_tokens = {
        "<|channel|>",
        "<|constrain|>",
        "<|message|>",
        "<|start|>",
        "<|end|>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    // Build PEG parser for GPT-OSS format
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        auto assistant_prefix = [&]() {
            return p.optional(p.literal("<|start|>") + "assistant");
        };

        auto commentary_content = p.rule("gpt-oss-commentary",
            assistant_prefix()
            + p.literal("<|channel|>") + "commentary"
            + p.literal("<|message|>")
            + p.tag(Tag::CONTENT, p.until("<|end|>"))
            + p.literal("<|end|>")
        );

        auto final_content = p.rule("gpt-oss-final",
            assistant_prefix()
            + p.literal("<|channel|>") + "final"
            + p.optional(p.literal(" ") + p.literal("<|constrain|>") + p.until("<|message|>"))
            + p.literal("<|message|>")
            + p.tag(Tag::CONTENT, p.until("<|end|>"))
            + p.literal("<|end|>")
        );

        auto reasoning_block = p.eps();
        if (extract_reasoning) {
            // Only tag the content between <|message|> and <|end|>, not the surrounding tokens
            reasoning_block = p.optional(
                p.literal("<|channel|>") + "analysis" + p.literal("<|message|>")
                + p.tag(Tag::REASONING, p.until("<|end|>")) + p.literal("<|end|>")
                + assistant_prefix()
            );
        }

        // Response format parser (with JSON schema constraint)
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            // Final channel with JSON content
            return reasoning_block << p.optional(p.literal("<|channel|>") + "final") << p.optional(p.space())
                << p.optional(p.literal("<|constrain|>") + p.until("<|message|>"))
                << p.literal("<|message|>")
                << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                // Trigger on tool calls that appear in the commentary channel
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                    "<\\|channel\\|>(commentary|analysis) to"
                });

                // Trigger tool calls that appear in the role section, either at the
                // start or in the middle.
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                    "^ to"
                });

                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                    "<\\|start\\|>assistant to"
                });
            }

            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](const auto &, const auto & name, const auto & parameters, const auto &) {
                // Tool call in channel: <|channel|>analysis|commentary to=functions.name<|message|>{...}<|end|>
                tool_choice |= p.rule("tool-channel-" + name, p.tag(Tag::TOOL,
                    p.literal("<|channel|>")
                    + (p.literal("analysis") | "commentary")
                    + p.atomic_tag(Tag::TOOL_OPEN, p.literal(" to=functions."))
                    + p.literal_tag(Tag::TOOL_NAME, name)
                    + p.optional(" " + p.literal("<|constrain|>") + "json")
                    + p.literal("<|message|>")
                    + p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-params", parameters))
                    + p.tag(Tag::TOOL_CLOSE, p.literal("<|end|>"))
                ));

                // Tool call in role: <|start|>assistant to=functions.name<|channel|>analysis|commentary json<|message|>{...}<|call|>
                // Note: <|call|> is an end token (in additional_stops) so the model stops before producing it.
                // We make it optional so parsing works with or without it.
                tool_choice |= p.rule("tool-role-" + name, p.tag(Tag::TOOL,
                    assistant_prefix()
                    + p.optional(p.literal(" "))
                    + p.atomic_tag(Tag::TOOL_OPEN, p.literal("to=functions."))
                    + p.literal_tag(Tag::TOOL_NAME, name)
                    + p.literal("<|channel|>")
                    + (p.literal("analysis") | "commentary")
                    + p.optional(p.literal(" ") + p.until("<|message|>"))  // content type (e.g., "json") without <|constrain|>
                    + p.literal("<|message|>")
                    + p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-params", parameters))
                    + p.tag(Tag::TOOL_CLOSE, p.optional(p.literal("<|call|>")))
                ));
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call-root",
                p.space()
                + p.repeat(tool_choice, min_calls, max_calls));

            bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            if (require_tools) {
                return reasoning_block << tool_calls;
            }

            auto pre_tool_content = p.repeat(commentary_content, 0, -1);

            // Allow direct tool calls (role format) or commentary followed by tool calls (channel format)
            return reasoning_block << p.choice({
                tool_calls,                      // Direct tool call (e.g., <|start|>assistant to=functions.name...)
                pre_tool_content << tool_calls   // Commentary then tool (e.g., <|channel|>commentary...<|end|>...)
            });
        }

        // Content only parser with optional reasoning
        auto content_sequence = p.sequence();
        content_sequence += p.repeat(commentary_content, 0, -1);
        content_sequence += p.choice({final_content, commentary_content});

        return reasoning_block << content_sequence;
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    return data;
}
