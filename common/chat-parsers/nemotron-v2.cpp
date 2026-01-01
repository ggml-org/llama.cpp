// Nemotron v2 tool call format
// Format: <TOOLCALL>[{"name": "...", "arguments": {...}}]</TOOLCALL>
// With optional <think>...</think> reasoning blocks

#include "chat-parsers-internal.h"
#include "chat.h"

common_chat_params common_chat_params_init_nemotron_v2_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Note: thoughts are not re-rendered by the template.
    auto adjusted_messages = json::array({
        json {
            {"role", "user"},
            {"content", inputs.enable_thinking ? "/think" : "/nothink"},
        }
    });
    for (const auto & msg : inputs.messages) {
        adjusted_messages.push_back(msg);
    }
    data.prompt = apply(tmpl, inputs, /* messages_override= */ adjusted_messages);

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<TOOLCALL>",
        "</TOOLCALL>",
        "<SPECIAL_12>",
        "<SPECIAL_11>Assistant",
        "<SPECIAL_11>User",
        "<SPECIAL_10>System",
    };


    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;

    bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;
        auto skip_special_markers = [&]() {
            auto marker = p.rule("nemotron-special-marker",
                p.optional(p.literal("\n"))
                + p.choice({
                    p.literal("<SPECIAL_12>"),
                    p.literal("<SPECIAL_11>Assistant"),
                    p.literal("<SPECIAL_11>User"),
                    p.literal("<SPECIAL_10>System")
                })
                + p.optional(p.literal("\n"))
            );
            return p.repeat(marker, 0, -1);
        };

        auto reasoning = p.eps();
        if (inputs.enable_thinking && extract_reasoning) {
            auto reasoning_content = p.tag(Tag::REASONING, p.until("</think>")) + ("</think>" | p.end());
            if (data.thinking_forced_open) {
                reasoning = reasoning_content;
            }
        }

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return reasoning << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser - JSON array format
        // Format: <TOOLCALL>[{"name": "...", "arguments": {...}}]</TOOLCALL>
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers = {
                    {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<TOOLCALL>"}
                };
            }

            auto any_tool_call = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto &) {
                any_tool_call |= p.tag(Tag::TOOL, p.sequence()
                    + p.literal_tag(Tag::TOOL_OPEN, "{")
                    << "\"name\"" << ":" << ("\"" + p.literal_tag(Tag::TOOL_NAME, name) + "\"") << ","
                    << "\"arguments\"" << ":" << p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters))
                    << p.literal_tag(Tag::TOOL_CLOSE, "}"));
            });

            auto tool_calls = p.trigger_rule("tool-call-root",
                p.space()
                    + p.literal("<TOOLCALL>[")
                    + any_tool_call + p.repeat(p.literal(",") << any_tool_call, 0, inputs.parallel_tool_calls ? -1 : 0)
                    + p.literal("]</TOOLCALL>"));

            if (require_tools) {
                return reasoning << tool_calls;
            }

            auto specials = skip_special_markers();
            auto stop_before = std::vector<std::string> {
                "\n<TOOLCALL>", "<TOOLCALL>",
                "\n<SPECIAL_12>", "<SPECIAL_12>",
                "\n<SPECIAL_11>Assistant", "<SPECIAL_11>Assistant",
                "\n<SPECIAL_11>User", "<SPECIAL_11>User",
                "\n<SPECIAL_10>System", "<SPECIAL_10>System",
            };
            auto stop_after = std::vector<std::string> {
                "\n<SPECIAL_12>", "<SPECIAL_12>",
                "\n<SPECIAL_11>Assistant", "<SPECIAL_11>Assistant",
                "\n<SPECIAL_11>User", "<SPECIAL_11>User",
                "\n<SPECIAL_10>System", "<SPECIAL_10>System",
            };
            auto content_before = p.optional(p.tag(Tag::CONTENT, p.until_one_of(stop_before)));
            auto content_after = (p.optional(p.tag(Tag::CONTENT, p.until_one_of(stop_after))) << specials);
            return reasoning << specials << content_before << specials << tool_calls << specials << content_after;
        }

        // Content only parser
        include_grammar = false;
        auto stop_only = std::vector<std::string> {
            "\n<SPECIAL_12>", "<SPECIAL_12>",
            "\n<SPECIAL_11>Assistant", "<SPECIAL_11>Assistant",
            "\n<SPECIAL_11>User", "<SPECIAL_11>User",
            "\n<SPECIAL_10>System", "<SPECIAL_10>System",
        };
        return reasoning << skip_special_markers() << p.tag(Tag::CONTENT, p.until_one_of(stop_only)) << skip_special_markers();
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    
    return data;
}
