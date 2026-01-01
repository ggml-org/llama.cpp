// Llama 3.x tool call format
// Format: {"type":"function","name":"func","parameters":{...}}
// Also supports builtin tools: <|python_tag|>python.call(code="...")

#include "chat-parsers-internal.h"
#include "chat.h"
#include "common.h"

static void expect_tool_parameters(const std::string & name, const json & parameters, const std::vector<std::string> & expected_properties) {
    if (!parameters.contains("properties") || !parameters.at("properties").is_object()) {
        throw std::runtime_error("Tool " + name + " is missing properties");
    }
    const auto & properties = parameters.at("properties");
    for (const auto & prop_name : expected_properties) {
        if (!properties.contains(prop_name)) {
            std::vector<std::string> prop_names;
            for (auto it = properties.begin(); it != properties.end(); ++it) {
                prop_names.push_back(it.key());
            }
            throw std::runtime_error("Tool " + name + " is missing property: " + prop_name + " (found: " + string_join(prop_names, ", ") + ")");
        }
    }
}

common_chat_params common_chat_params_init_llama_3_x_peg(const common_chat_template & tmpl, const struct templates_params & inputs, bool allow_python_tag_builtin_tools) {
    auto builtin_tools = json::array();
    common_chat_params data;

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    data.preserved_tokens = {};

    // Build PEG parser
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        const auto consume_message_end = [&]() {
            auto seq = p.sequence();
            seq += p.optional(p.choice({
                p.literal("<|eot_id|>"),
                p.literal("<|eom_id|>"),
                p.literal("<|end|>")
            }));
            seq += p.optional(p.space());
            return seq;
        };

        // Build tool call alternatives
        auto tool_choice = p.choice();

        // Check for builtin tools
        std::vector<std::string> builtin_tool_names;

        foreach_function(inputs.tools, [&](const auto &, const auto & name, const auto & parameters, const auto &) {
            // Check if this is a builtin tool
            if (allow_python_tag_builtin_tools) {
                if (name == "wolfram_alpha" || name == "web_search" || name == "brave_search") {
                    // Validate that builtin tools have expected properties
                    expect_tool_parameters(name, parameters, {"query"});
                    builtin_tool_names.push_back(name);
                    builtin_tools.push_back(name);

                    // Builtin tool format: <|python_tag|>name.call(key="value")
                    common_peg_parser args = p.eps();
                    if (parameters.contains("properties")) {
                        bool first = true;
                        for (auto it = parameters.at("properties").begin(); it != parameters.at("properties").end(); ++it) {
                            if (!first) {
                                args = args + ", ";
                            }
                            // Use schema validation for each argument value
                            args = args + p.literal_tag(Tag::TOOL_ARG_NAME, it.key()) + "=" +
                                   p.tag(Tag::TOOL_ARG_JSON_VALUE, p.schema(p.json(), "builtin-" + name + "-arg-" + it.key(), it.value()));
                            first = false;
                        }
                    }

                    tool_choice |= p.rule("builtin-" + name, p.tag(Tag::TOOL,
                        p.atomic_tag(Tag::TOOL_OPEN, p.literal("<|python_tag|>") + p.literal_tag(Tag::TOOL_NAME, name) + ".call(")
                        + args
                        + p.literal_tag(Tag::TOOL_CLOSE, ")")
                    ));
                } else if (name == "python" || name == "code_interpreter") {
                    // Validate that builtin tools have expected properties
                    expect_tool_parameters(name, parameters, {"code"});
                    builtin_tool_names.push_back(name);
                    builtin_tools.push_back(name);

                    // Builtin tool format: <|python_tag|>name.call(key="value")
                    common_peg_parser args = p.eps();
                    if (parameters.contains("properties")) {
                        bool first = true;
                        for (auto it = parameters.at("properties").begin(); it != parameters.at("properties").end(); ++it) {
                            if (!first) {
                                args = args + ", ";
                            }
                            // Use schema validation for each argument value
                            args = args + p.literal_tag(Tag::TOOL_ARG_NAME, it.key()) + "=" +
                                   p.tag(Tag::TOOL_ARG_JSON_VALUE, p.schema(p.json(), "builtin-" + name + "-arg-" + it.key(), it.value()));
                            first = false;
                        }
                    }

                    tool_choice |= p.rule("builtin-" + name, p.tag(Tag::TOOL,
                        p.atomic_tag(Tag::TOOL_OPEN, p.literal("<|python_tag|>") + p.literal_tag(Tag::TOOL_NAME, name) + ".call(")
                        + args
                        + p.literal_tag(Tag::TOOL_CLOSE, ")")
                    ));
                }
            }

            // Standard JSON format: {"type":"function","name":"name","parameters":{...}}
            tool_choice |= p.rule("tool-" + name, p.tag(Tag::TOOL,
                p.literal_tag(Tag::TOOL_OPEN, "{")
                << p.optional("\"type\"" << p.literal(":") << "\"function\"" << ",")
                << "\"name\"" << ":" << "\"" + p.literal_tag(Tag::TOOL_NAME, name) + "\"" << ","
                << "\"parameters\"" << ":"
                << p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-params", parameters))
                << p.atomic_tag(Tag::TOOL_CLOSE, p.space() + "}")
            ));
        });

        bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                // Grammar triggers
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                    "(\\{\\s*(?:\"type\"\\s*:\\s*\"function\"\\s*,\\s*)?\"name\"\\s*:\\s*\")[\\s\\S]*",
                });
                if (!builtin_tools.empty()) {
                    data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|python_tag|>"});
                    data.preserved_tokens.push_back("<|python_tag|>");
                }
            }

            data.additional_stops.push_back("<|eom_id|>");
            
            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;

            // Content until we see start of JSON object or python_tag
            std::vector<std::string> delimiters = {"{"};
            if (!builtin_tool_names.empty()) {
                delimiters.push_back("<|python_tag|>");
            }
            auto content = p.tag(Tag::CONTENT, p.until_one_of(delimiters)) << consume_message_end();
            auto tool_calls = p.trigger_rule("tool-call-root",
                p.space()
                + p.repeat(tool_choice, min_calls, max_calls));

            if (require_tools) {
                return tool_calls;
            }
            return content << tool_calls;
        }

        // Content only parser
        auto content_only = p.sequence({
            p.tag(Tag::CONTENT, p.until_one_of({"<|eot_id|>", "<|eom_id|>", "<|end|>"})),
            consume_message_end()
        });
        return p.choice({content_only, p.tag(Tag::CONTENT, p.rest())});
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    data.prompt = apply(tmpl, inputs, /* messages_override =*/ std::nullopt, /* tools_override= */ std::nullopt, json {
        {"date_string", format_time(inputs.now, "%d %b %Y")},
        {"tools_in_user_message", false},
        {"builtin_tools", builtin_tools.empty() ? json() : builtin_tools},
    });

    return data;
}
