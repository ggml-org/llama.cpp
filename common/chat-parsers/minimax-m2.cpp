// MiniMax-M2 tool call format
// Format: <minimax:tool_call><invoke name="function"><parameter name="key">value</parameter></invoke></minimax:tool_call>
// With optional <think>...</think> reasoning blocks

#include "chat-parsers-internal.h"
#include "chat.h"

common_chat_params common_chat_params_init_minimax_m2_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_MINIMAX_M2;

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

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;

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
            return reasoning << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<minimax:tool_call>"});
            }

            auto invoke_choice = p.choice();
            foreach_function(inputs.tools, [&](const auto & function, const auto & name, const auto & parameters, const auto & schema_info) {
                // Format: <invoke name="function_name"><parameter name="key">value</parameter></invoke>
                auto tool_open = "<invoke name=\"" + p.literal_tag(Tag::TOOL_NAME, name) + "\">" + p.space();
                auto tool_close = p.space() + p.literal("</invoke>") + p.space();

                auto parameter_choice = p.choice();
                bool has_parameter_rules = false;

                auto arg_close = p.literal("</parameter>") + p.space();

                foreach_parameter(function, [&](const auto & param_name, const json & param_schema, bool /* is_required */) {
                    auto rule_name = "tool-" + name + "-arg-" + param_name;

                    auto arg_open = "<parameter name=\"" + p.literal_tag(Tag::TOOL_ARG_NAME, param_name) + "\">";
                    auto arg_value = p.schema_or_raw_string_until(rule_name + "-schema", param_schema, "</parameter>",
                        schema_info, Tag::TOOL_ARG_STRING_VALUE, Tag::TOOL_ARG_JSON_VALUE, false);

                    auto arg_rule = p.rule(rule_name,
                        p.atomic_tag(Tag::TOOL_ARG_OPEN, arg_open)
                        + arg_value
                        + p.atomic_tag(Tag::TOOL_ARG_CLOSE, arg_close));
                    parameter_choice |= arg_rule;
                    has_parameter_rules = true;
                });

                // By JSON Schema spec, missing additionalProperties defaults to true
                bool allow_additional = false;
                bool additional_has_schema = false;
                json additional_schema;
                if (parameters.contains("additionalProperties")) {
                    const json & additional = parameters.at("additionalProperties");
                    if (additional.is_boolean()) {
                        allow_additional = additional.get<bool>();
                    } else if (additional.is_object()) {
                        allow_additional = true;
                        additional_has_schema = true;
                        additional_schema = additional;
                    }
                }

                if (allow_additional || !has_parameter_rules) {
                    auto dynamic_key = "<parameter name=\"" + p.tag(Tag::TOOL_ARG_NAME, p.until("\"")) + "\">";
                    auto additional_value = additional_has_schema
                        ? p.schema_or_raw_string_until("tool-" + name + "-arg-generic", additional_schema, "</parameter>",
                            schema_info, Tag::TOOL_ARG_STRING_VALUE, Tag::TOOL_ARG_JSON_VALUE, false)
                        : p.tag(Tag::TOOL_ARG_STRING_VALUE, p.until("</parameter>"));

                    auto additional_rule = p.rule("tool-" + name + "-arg-generic",
                        p.atomic_tag(Tag::TOOL_ARG_OPEN, dynamic_key)
                        + additional_value
                        + p.atomic_tag(Tag::TOOL_ARG_CLOSE, arg_close));
                    parameter_choice |= additional_rule;
                    has_parameter_rules = true;
                }

                common_peg_parser args = has_parameter_rules ? p.repeat(parameter_choice, 0, -1) : p.eps();

                // Add p.space() after TOOL tag to consume whitespace between parallel tool calls
                invoke_choice |= p.rule("tool-" + name, p.tag(Tag::TOOL,
                    p.atomic_tag(Tag::TOOL_OPEN, tool_open)
                    + args
                    + p.atomic_tag(Tag::TOOL_CLOSE, tool_close)) + p.space());
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_block = p.rule("tool-call-block",
                p.literal("<minimax:tool_call>")
                + p.space()
                + p.repeat(invoke_choice, /* min = */ 1, /* max = */ -1)
                + p.literal("</minimax:tool_call>")
                + p.space());
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_block, /* min = */ min_calls, /* max = */ max_calls));

            bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            if (require_tools) {
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
            return reasoning << content_before << tool_calls << content_after;
        }

        // Content only parser
        include_grammar = false;
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

    return data;
}
