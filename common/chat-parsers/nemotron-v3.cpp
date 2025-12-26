// Nemotron 3 Nano 30B A3B tool call format
// Format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
// With optional <think>...</think> reasoning blocks

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_nemotron_v3_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_NEMOTRON_V3;

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
        "<tool_call>",
        "</tool_call>",
        "<toolcall>",
        "</toolcall>",
        "<SPECIAL_11>Assistant",
        "<SPECIAL_11>User",
        "<SPECIAL_12>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;

    bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;
        auto newline = p.choice({p.literal("\r\n"), p.literal("\n")});
        auto whitespace = p.repeat(p.choice({newline, p.literal(" "), p.literal("\t")}), 0, -1);
        auto skip_blank_lines = whitespace;
        auto assistant_header = p.literal("<|im_start|>assistant") + p.choice({p.literal("\r\n"), p.literal("\n")});
        auto assistant_prefix = whitespace + p.optional(assistant_header);
        auto assistant_suffix = whitespace + p.optional(p.literal("<|im_end|>")) + whitespace;
        auto after_reasoning_gap = whitespace;
        auto think_open = p.literal("<think>") + p.optional(newline);
        auto think_close = p.literal("</think>");
        auto reasoning = p.eps();
        if (inputs.enable_thinking && extract_reasoning) {
            auto reasoning_content = p.tag(Tag::REASONING, p.until("</think>")) + think_close;
            if (data.thinking_forced_open) {
                reasoning = reasoning_content;
            } else {
                reasoning = p.optional(think_open + reasoning_content);
            }
        } else {
            reasoning = p.optional(think_open + p.until("</think>") + think_close);
        }

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return assistant_prefix + reasoning + after_reasoning_gap + p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema)) + assistant_suffix;
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers = {
                    {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>"}
                };
            }
            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const auto & parameters, const auto & schema_info) {
                // Default to false for stricter parsing - only allow explicitly defined parameters
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

                auto tool_open = "<function=" + p.literal_tag(Tag::TOOL_NAME, name) + ">\n";
                auto tool_close = p.literal("</function>\n");

                // Build schema-aware parameter rules
                auto args = p.sequence();
                foreach_parameter(parameters, [&](const std::string & param_name, const json & param_schema, bool /* is_required */) {
                    auto rule_name = "nemotron-v3-" + name + "-arg-" + param_name;
                    auto arg_body = p.rule(rule_name + "-body", p.until_one_of({
                        "\n</parameter>",
                        "\n<parameter=",
                        "\n</function>"
                    }));

                    auto arg_value = p.eps();
                    if (schema_info.resolves_to_string(param_schema)) {
                        arg_value = p.tag(Tag::TOOL_ARG_STRING_VALUE, arg_body);
                    } else {
                        // For non-string types, parse as JSON value
                        arg_value = p.tag(Tag::TOOL_ARG_JSON_VALUE, arg_body);
                    }

                    auto arg_rule = p.rule(rule_name,
                        p.atomic_tag(Tag::TOOL_ARG_OPEN,
                            p.literal("<parameter=")
                            + p.literal_tag(Tag::TOOL_ARG_NAME, param_name)
                            + p.literal(">\n"))
                        + arg_value
                        + p.optional(newline)
                        + p.optional(p.atomic_tag(Tag::TOOL_ARG_CLOSE, p.literal("</parameter>\n"))));
                    args += p.repeat(arg_rule, /* min = */ 0, /* max = */ 1);
                });

                // Add generic rule for additional properties
                if (allow_additional) {
                    auto generic_arg_body = p.rule("nemotron-v3-" + name + "-arg-generic-body", p.until_one_of({
                        "\n</parameter>",
                        "\n<parameter=",
                        "\n</function>"
                    }));

                    auto additional_value = p.eps();
                    if (additional_has_schema && !schema_info.resolves_to_string(additional_schema)) {
                        additional_value = p.tag(Tag::TOOL_ARG_JSON_VALUE, generic_arg_body);
                    } else {
                        additional_value = p.tag(Tag::TOOL_ARG_STRING_VALUE, generic_arg_body);
                    }

                    auto generic_arg = p.rule("nemotron-v3-" + name + "-arg-generic",
                        p.atomic_tag(Tag::TOOL_ARG_OPEN,
                            p.literal("<parameter=")
                            + p.tag(Tag::TOOL_ARG_NAME, p.until(">"))
                            + p.literal(">\n"))
                        + additional_value
                        + p.optional(newline)
                        + p.optional(p.atomic_tag(Tag::TOOL_ARG_CLOSE, p.literal("</parameter>\n"))));
                    args += p.repeat(generic_arg, 0, -1);
                }

                tool_choice |= p.rule("tool-" + name, p.atomic_tag(Tag::TOOL_OPEN, tool_open) + args + p.atomic_tag(Tag::TOOL_CLOSE, tool_close));
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_call_open = p.choice({p.literal("<tool_call>"), p.literal("<toolcall>")}) + skip_blank_lines;
            auto tool_call_close = p.choice({p.literal("</tool_call>"), p.literal("</toolcall>")});
            auto tool_call = p.rule("tool-call",
                tool_call_open
                + tool_choice
                + tool_call_close
                + skip_blank_lines);
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, /* min = */ min_calls, /* max = */ max_calls));

            auto stop_before = std::vector<std::string>{
                "\n<tool_call>", "\r\n<tool_call>", "<tool_call>",
                "\n<toolcall>", "\r\n<toolcall>", "<toolcall>"
            };
            auto stop_after = std::vector<std::string>{
                "\n<|im_end|>", "\r\n<|im_end|>", "<|im_end|>"
            };
            auto content_before = p.optional(p.tag(Tag::CONTENT, p.until_one_of(stop_before)));
            auto content_after = p.optional(p.tag(Tag::CONTENT, p.until_one_of(stop_after)));
            auto pre_tool_gap = p.repeat(newline, 0, -1);
            if (require_tools) {
                return assistant_prefix + reasoning + after_reasoning_gap + pre_tool_gap + tool_calls + assistant_suffix;
            }
            return assistant_prefix + reasoning + after_reasoning_gap + content_before + pre_tool_gap + tool_calls + content_after + assistant_suffix;
        }

        // Content only parser
        include_grammar = false;
        auto content_body = p.optional(p.tag(Tag::CONTENT, p.until_one_of({
            "\n<|im_end|>", "\r\n<|im_end|>", "<|im_end|>"
        })));
        return assistant_prefix + reasoning + after_reasoning_gap + content_body + assistant_suffix;
    });

    common_chat_build_peg_grammar(inputs, parser, data);

    return data;
}
