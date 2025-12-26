// Seed OSS tool call format
// Format: <seed:tool_call><function=name><parameter=key>value</parameter></function></seed:tool_call>
// With optional <seed:think>...</seed:think> reasoning blocks

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_seed_oss_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_SEED_OSS;

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<seed:think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</seed:think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    data.preserved_tokens = {
        "<seed:think>",
        "</seed:think>",
        "<seed:tool_call>",
        "</seed:tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
        "<seed:eos>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;
    bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;
        auto newline = p.choice({p.literal("\r\n"), p.literal("\n")});
        // Limit newlines around <seed:eos> to prevent grammar from accepting unlimited newlines
        auto eos = p.optional(p.repeat(newline, 0, 2) + p.literal("<seed:eos>") + p.repeat(newline, 0, 2));
        auto reasoning = p.eps();
        auto reasoning_block = p.literal("<seed:think>")
            + p.tag(Tag::REASONING, p.until("</seed:think>"))
            + (p.literal("</seed:think>") | p.end());
        if (extract_reasoning) {
            if (inputs.enable_thinking && data.thinking_forced_open) {
                reasoning = reasoning_block;
            } else if (inputs.enable_thinking) {
                reasoning = p.optional(reasoning_block);
            } else {
                reasoning = p.optional(reasoning_block);
            }
        } else {
            reasoning = p.optional(reasoning_block);
        }

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return reasoning << p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers = {
                    {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<seed:tool_call>"}
                };
            }
            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto & schema_info) {
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

                auto tool_open = "<function=" + p.literal_tag(Tag::TOOL_NAME, name) + ">";
                auto tool_close = p.literal("</function>");
                auto args = p.sequence();

                foreach_parameter(parameters, [&](const auto & param_name, const json & param_schema, bool is_required) {
                    auto rule_name = "tool-" + name + "-arg-" + param_name;

                    auto arg_open = "<parameter=" + p.literal_tag(Tag::TOOL_ARG_NAME, param_name) + ">";
                    auto arg_close = p.literal("</parameter>");
                    auto arg_value = p.schema_or_raw_string_until(rule_name + "-schema", param_schema, "</parameter>",
                        schema_info, Tag::TOOL_ARG_STRING_VALUE, Tag::TOOL_ARG_JSON_VALUE, true);

                    auto arg_rule = p.rule(rule_name,
                        p.atomic_tag(Tag::TOOL_ARG_OPEN, arg_open)
                        + arg_value
                        + p.atomic_tag(Tag::TOOL_ARG_CLOSE, arg_close)
                        + p.space());
                    // Enforce required parameters:
                    // - Non-string types: always enforced via schema
                    // - String types with maxLength: enforced via length-limited grammar
                    // - String types without maxLength: not enforced (unlimited p.until doesn't constrain model)
                    int max_length = param_schema.contains("maxLength") && param_schema["maxLength"].is_number_integer()
                        ? param_schema["maxLength"].get<int>() : -1;
                    bool can_enforce = !schema_info.resolves_to_string(param_schema) || max_length > 0;
                    bool enforce_required = is_required && can_enforce;
                    args += p.repeat(arg_rule, /* min = */ enforce_required ? 1 : 0, /* max = */ 1);
                });

                if (allow_additional) {
                    auto dynamic_name = p.tag(Tag::TOOL_ARG_NAME, p.until(">"));
                    auto additional_value = additional_has_schema
                        ? p.schema_or_raw_string_until("seed-oss-additional-" + name, additional_schema, "</parameter>",
                            schema_info, Tag::TOOL_ARG_STRING_VALUE, Tag::TOOL_ARG_JSON_VALUE, true)
                        : p.tag(Tag::TOOL_ARG_STRING_VALUE, p.until("</parameter>"));

                    auto additional_rule = p.rule("seed-parameter-generic-" + name,
                        p.atomic_tag(Tag::TOOL_ARG_OPEN, "<parameter=" + dynamic_name + ">")
                        + additional_value
                        + p.atomic_tag(Tag::TOOL_ARG_CLOSE, p.literal("</parameter>"))
                        + p.space());
                    args += p.repeat(additional_rule, 0, -1);
                }

                tool_choice |= p.rule("tool-" + name,
                    p.atomic_tag(Tag::TOOL_OPEN, tool_open)
                    << args
                    << p.atomic_tag(Tag::TOOL_CLOSE, tool_close));
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            // Add p.space() after </seed:tool_call> to consume whitespace between parallel tool calls
            auto tool_call = p.rule("tool-call",
                p.literal("<seed:tool_call>")
                + p.space()
                + tool_choice
                + p.space()
                + p.literal("</seed:tool_call>")
                + p.space());
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, /* min = */ min_calls, /* max = */ max_calls));

            auto stop_before = std::vector<std::string> {
                "\r\n\r\n<seed:tool_call>", "\n\n<seed:tool_call>",
                "\r\n<seed:tool_call>", "\n<seed:tool_call>", "<seed:tool_call>",
                "\r\n\r\n<seed:toolcall>", "\n\n<seed:toolcall>",
                "\r\n<seed:toolcall>", "\n<seed:toolcall>", "<seed:toolcall>",
            };
            auto content_before = p.optional(p.tag(Tag::CONTENT, p.until_one_of(stop_before)));
            // After tool calls, only allow limited trailing whitespace (not arbitrary content)
            // to prevent the grammar from allowing unlimited newlines
            auto post_tool_gap = p.repeat(newline, 0, 2);
            auto pre_calls_gap = p.repeat(newline, 0, -1);
            if (require_tools) {
                return reasoning << pre_calls_gap << tool_calls << post_tool_gap << eos;
            }
            return reasoning << content_before << pre_calls_gap << tool_calls << post_tool_gap << eos;
        }

        // Content only parser
        include_grammar = false;
        auto content_tail = p.optional(p.tag(Tag::CONTENT, p.until_one_of({
            "\r\n\r\n<seed:eos>", "\n\n<seed:eos>",
            "\r\n<seed:eos>", "\n<seed:eos>", "<seed:eos>"
        })));
        // Limit trailing newlines before eos to prevent grammar from accepting unlimited newlines
        auto pre_eos_gap = p.repeat(newline, 0, 2);
        return reasoning << content_tail << pre_eos_gap << eos;
    });

    common_chat_build_peg_grammar(inputs, parser, data);

    return data;
}
