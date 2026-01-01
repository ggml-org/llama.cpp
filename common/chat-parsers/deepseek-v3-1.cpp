// DeepSeek V3.1 tool call format
// Format: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>name<｜tool▁sep｜>{"arg":"value"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
// With optional <think>...</think> reasoning blocks

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_deepseek_v3_1_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Pass thinking context for DeepSeek V3.1 template
    json additional_context = {
        {"thinking", inputs.enable_thinking},
    };

    auto prompt = apply(tmpl, inputs,
                       /* messages_override= */ inputs.messages,
                       /* tools_override= */ std::nullopt,
                       additional_context);
    if (string_ends_with(data.prompt, "<think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }
    data.prompt = prompt;

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    data.grammar_lazy = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED && inputs.json_schema.is_null();

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
        "<｜tool▁sep｜>",
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    };

    // Build PEG parser
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        auto consume_eos = [&]() {
            return p.optional(p.literal("<｜end▁of▁sentence｜>")) + p.optional(p.space());
        };

        // Optional thinking block
        auto reasoning = p.eps();
        if (extract_reasoning) {
            if (data.thinking_forced_open) {
                reasoning = p.tag(Tag::REASONING, p.until("</think>")) + "</think>";
            } else {
                reasoning = p.optional("<think>" + p.tag(Tag::REASONING, p.until("</think>")) + "</think>");
            }
        }

        if (has_tools) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                    std::string(data.thinking_forced_open ? "[\\s\\S]*?(</think>\\s*)" : "(?:<think>[\\s\\S]*?</think>\\s*)?") +
                        "(<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)[\\s\\S]*"
                });
            }

            auto any_tool_call = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto &) {
                any_tool_call |= p.tag(Tag::TOOL, p.sequence()
                    + p.tag(Tag::TOOL_OPEN, p.literal("<｜tool▁call▁begin｜>"))
                    + p.tag(Tag::TOOL_NAME, p.literal(name))
                    + "<｜tool▁sep｜>"
                    << p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters))
                    << p.tag(Tag::TOOL_CLOSE, p.literal("<｜tool▁call▁end｜>")));
            });

            auto tool_calls =
                p.space()  // Allow optional leading whitespace
                + p.literal("<｜tool▁calls▁begin｜>")
                + any_tool_call + p.repeat(p.space() << any_tool_call, 0, inputs.parallel_tool_calls ? -1 : 0)
                + p.literal("<｜tool▁calls▁end｜>")
                << consume_eos();

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return reasoning << tool_calls;
            }

            // Content until tool calls marker
            auto content = p.tag(Tag::CONTENT, 
                inputs.json_schema.is_null()
                    ? p.until_one_of({
                        "<｜tool▁calls▁begin｜>",
                        "<｜tool_calls_begin｜>",
                        "<｜tool calls begin｜>",
                        "<｜tool\\_calls\\_begin｜>",
                        "<｜tool▁calls｜>"})
                    : p.schema(p.json(), "response-format", inputs.json_schema)
            );

            return reasoning << content << tool_calls;
        }

        // Content only parser
        auto content_only = p.sequence({
            p.tag(Tag::CONTENT, p.until("<｜end▁of▁sentence｜>")),
            consume_eos()
        });
        return reasoning << p.choice({content_only, p.tag(Tag::CONTENT, p.rest())});
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    return data;
}
