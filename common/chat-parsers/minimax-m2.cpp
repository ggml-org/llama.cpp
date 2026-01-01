// MiniMax-M2 tool call format
// Format: <minimax:tool_call><invoke name="function"><parameter name="key">value</parameter></invoke></minimax:tool_call>
// With optional <think>...</think> reasoning blocks

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_minimax_m2_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);

    // Handle thinking tags based on prompt ending
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>\n\n";
        } else {
            data.thinking_forced_open = true;
        }
    }

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<minimax:tool_call>",
        "</minimax:tool_call>",
        "<invoke name=",
        "</invoke>",
        "<parameter name=",
        "</parameter>",
    };

    data.additional_stops.push_back("[e~[");

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;
        auto consume_footer = [&]() {
            return p.optional(p.literal("[e~[")) + p.optional(p.space());
        };
        auto reasoning = p.eps();
        if (inputs.enable_thinking && extract_reasoning) {
            auto reasoning_content = p.tag(Tag::REASONING, p.until("</think>")) + ("</think>" | p.end());
            if (data.thinking_forced_open) {
                reasoning = reasoning_content;
            } else {
                auto reasoning_block = p.choice({
                    p.literal("<think>") + reasoning_content,
                    reasoning_content,
                });
                reasoning = p.optional(reasoning_block);
            }
        }

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return reasoning
                << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema))
                << consume_footer();
        }

        // Tool call parser
        if (has_tools) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<minimax:tool_call>"});
            }

            generic_tool_call_format format;
            format.tool_calls_start = p.space() + "<minimax:tool_call>";
            format.tool_calls_sep = p.eps();
            format.tool_calls_end = p.literal("</minimax:tool_call>");
            format.tool_call_start = p.space() + "<invoke name=\"";
            format.tool_call_name_params_sep = p.literal("\">");
            format.tool_call_end = p.space() + "</invoke>" + p.space();
            format.param_start = p.space() + "<parameter name=\"";
            format.param_name_value_sep = p.literal("\">");
            format.param_ends = { "</parameter>" };
            format.allow_raw_string_param_value = true;
            auto tool_calls = build_generic_tool_calls_peg_parser(p, inputs, format);
            
            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return reasoning << tool_calls;
            }

            auto stop_before = std::vector<std::string> {
                "\n<minimax:tool_call>", "<minimax:tool_call>",
                "\n<TOOLCALL>", "<TOOLCALL>",
                "\n<SPECIAL_12>", "<SPECIAL_12>",
                "\n<SPECIAL_11>Assistant", "<SPECIAL_11>Assistant",
                "\n<SPECIAL_11>User", "<SPECIAL_11>User",
                "\n<SPECIAL_10>System", "<SPECIAL_10>System",
            };
            auto stop_after = std::vector<std::string> {
                "\n<SPECIAL_12>", "<SPECIAL_12>",
                "\n<TOOLCALL>", "<TOOLCALL>",
                "\n<SPECIAL_11>Assistant", "<SPECIAL_11>Assistant",
                "\n<SPECIAL_11>User", "<SPECIAL_11>User",
                "\n<SPECIAL_10>System", "<SPECIAL_10>System",
                "\n<minimax:tool_call>", "<minimax:tool_call>",
            };
            auto content_before = p.optional(p.tag(Tag::CONTENT, p.until_one_of(stop_before)));
            auto content_after = p.optional(p.choice({
                p.sequence({p.tag(Tag::CONTENT, p.until_one_of(stop_after)), consume_footer()}),
                p.tag(Tag::CONTENT, p.rest())
            }));
            auto with_tools = content_before << tool_calls << content_after;
            auto content_only = p.choice({
                p.sequence({p.tag(Tag::CONTENT, p.until_one_of(stop_before)), consume_footer()}),
                p.tag(Tag::CONTENT, p.rest())
            });
            return reasoning << p.choice({with_tools, content_only});
        }

        // Content only parser
        auto stop_only = std::vector<std::string> {
            "\n<SPECIAL_12>", "<SPECIAL_12>",
            "\n<minimax:tool_call>", "<minimax:tool_call>",
            "\n<TOOLCALL>", "<TOOLCALL>",
            "\n<SPECIAL_11>Assistant", "<SPECIAL_11>Assistant",
            "\n<SPECIAL_11>User", "<SPECIAL_11>User",
            "\n<SPECIAL_10>System", "<SPECIAL_10>System",
        };
        auto content_tail = p.choice({
            p.sequence({p.tag(Tag::CONTENT, p.until_one_of(stop_only)), consume_footer()}),
            p.tag(Tag::CONTENT, p.rest())
        });
        return reasoning << content_tail;
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;

    return data;
}
