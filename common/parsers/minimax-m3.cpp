#include "parsers.h"

common_chat_params common_chat_params_init_minimax_m3(const common_chat_template &          tmpl,
                                                             const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_MINIMAX_M3;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<mm:think>";
    data.thinking_end_tags  = {"</mm:think>"};

    // M3 prefixes every tool tag with the namespace token "]<]minimax[>[";
    // params use the parameter name as the tag (<file_path>...</file_path>).
    const std::string NS          = "]<]minimax[>[";
    const std::string THINK_START = "<mm:think>";
    const std::string THINK_END   = "</mm:think>";
    const std::string FC_START    = NS + "<tool_call>";
    const std::string FC_END      = NS + "</tool_call>";
    const std::string INVOKE_END  = NS + "</invoke>";

    data.preserved_tokens = {
        NS,
        "<tool_call>",
        "</tool_call>",
        THINK_START,
        THINK_END,
    };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "]~b]ai"        },
        { COMMON_CHAT_ROLE_USER,      "]~b]user"      },
        { COMMON_CHAT_ROLE_TOOL,      "]~b]tool"      },
        { COMMON_CHAT_ROLE_SYSTEM,    "]~b]developer" },
        { COMMON_CHAT_ROLE_SYSTEM,    "]~b]system"    },
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    const std::string GEN_PROMPT = data.generation_prompt;

    using mm3 = common_chat_peg_minimax_m3_mapper;

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PROMPT + THINK_START + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.prefix(GEN_PROMPT, THINK_START);
        auto end = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning) {
            auto block = inputs.enable_thinking
                             ? p.literal(THINK_START) + p.space() +
                                   p.ac(p.reasoning(p.until(THINK_END)) + p.literal(THINK_END), THINK_END)
                             : p.literal(THINK_START) + p.ac(p.until(THINK_END) + p.literal(THINK_END), THINK_END);

            // A turn without reasoning is prefixed with a bare </mm:think>, written either by the
            // generation prompt (thinking_mode = "disabled") or by the model itself.
            reasoning = p.optional(p.choice({ block, p.literal(THINK_END) }));
        }

        if (has_response_format) {
            auto response_format = p.rule("response-format",
                p.literal("```json") + p.space() +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                p.space() + p.literal("```"));
            return generation_prompt + reasoning + response_format + end;
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto alternatives_of = [](const json & schema) -> std::optional<json> {
            for (const auto * keyword : { "oneOf", "anyOf" }) {
                if (schema.contains(keyword) && schema.at(keyword).is_array() && !schema.at(keyword).empty()) {
                    return schema.at(keyword);
                }
            }
            return std::nullopt;
        };

        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            auto         params   = function.contains("parameters") ? function.at("parameters") : json::object();

            auto schema_info = common_schema_info();
            schema_info.resolve_refs(params);

            // The template expands argument values recursively in XML (see the to_xml() macro)
            std::function<common_peg_parser(const json &, const std::string &, const std::string &)> value_of;
            std::function<common_peg_parser(const json &, const std::string &)>                      members_of;

            auto element_of = [&](const std::string & tag, const json & schema, const std::string & rule_name) {
                const std::string close = NS + "</" + tag + ">";
                return p.rule(rule_name,
                    p.tool_arg(
                        p.tool_arg_open(
                            p.literal(NS + "<") +
                            p.tool_arg_name(p.literal(tag)) +
                            p.literal(">")) +
                        value_of(schema, rule_name, close)));
            };

            value_of = [&](const json & schema,
                           const std::string & rule_name,
                           const std::string & close) -> common_peg_parser {
                auto close_tag = p.tool_arg_close(p.literal(close));

                // A string accepts anything, so a union with a string alternative is a string
                if (schema_info.resolves_to_string(schema)) {
                    return p.ac(p.tool_arg_string_value(p.until(close)) + close_tag, close);
                }

                if (auto alternatives = alternatives_of(schema)) {
                    std::vector<common_peg_parser> choices;

                    size_t index = 0;
                    for (const auto & alternative : *alternatives) {
                        const std::string alt_name = rule_name + "-" + std::to_string(index++);

                        // There is a risk that this breaks streaming deltas, but that's a risk we
                        // assume to provide tool arg streaming.
                        choices.push_back(value_of(alternative, alt_name, close));
                    }

                    return p.choice(choices);
                }

                const std::string type = schema.contains("type") && schema.at("type").is_string()
                                             ? schema.at("type").get<std::string>()
                                             : "";

                if (type == "object" && schema.contains("properties")) {
                    return p.tag(mm3::TOOL_ARG_OBJECT, members_of(schema, rule_name)) + p.space() + close_tag;
                }

                if (type == "array" && schema.contains("items")) {
                    const std::string item_close = NS + "</item>";
                    auto item = p.rule(rule_name + "-item",
                        p.tag(mm3::TOOL_ARG_ITEM,
                              p.literal(NS + "<item>") +
                                  value_of(schema.at("items"), rule_name + "-item", item_close)));
                    return p.tag(mm3::TOOL_ARG_ARRAY, p.repeat(p.space() + item, 0, -1)) + p.space() + close_tag;
                }

                return p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", schema, false)) + close_tag;
            };

            // Required properties in schema order, then any number of optional ones in any order.
            members_of = [&](const json & schema, const std::string & rule_prefix) -> common_peg_parser {
                const auto & props = schema.at("properties");

                std::set<std::string> required;
                if (schema.contains("required")) {
                    required = schema.at("required").get<std::set<std::string>>();
                }

                std::vector<common_peg_parser> required_elements;
                std::vector<common_peg_parser> optional_elements;
                for (const auto & [key, key_schema] : props.items()) {
                    auto element = element_of(key, key_schema, rule_prefix + "-" + key);
                    if (required.find(key) != required.end()) {
                        required_elements.push_back(element);
                    } else {
                        optional_elements.push_back(element);
                    }
                }

                common_peg_parser members = p.eps();
                for (size_t i = 0; i < required_elements.size(); i++) {
                    if (i > 0) {
                        members = members + p.space();
                    }
                    members = members + required_elements[i];
                }

                if (!optional_elements.empty()) {
                    common_peg_parser any_optional = p.choice();
                    for (const auto & element : optional_elements) {
                        any_optional |= element;
                    }
                    members = members + p.repeat(p.space() + any_optional, 0, -1);
                }

                return members;
            };

            common_peg_parser invoke_body =
                params.contains("properties") ? members_of(params, "tool-" + name + "-arg") : p.eps();

            auto func_parser = p.tool(
                p.tool_open(p.literal(NS + "<invoke name=\"") +
                            p.tool_name(p.literal(name)) + p.literal("\">")) +
                p.space() + invoke_body + p.space() +
                p.tool_close(p.literal(INVOKE_END)));

            tool_choice |= p.rule("tool-" + name, func_parser);
        });

        auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice +
                p.zero_or_more(p.space() + tool_choice) + p.space() + p.literal(FC_END));
        } else {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice + p.space() + p.literal(FC_END));
        }

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        auto content_before_tools = p.content(p.until(FC_START));
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, FC_START },
        };
    }

    return data;
}
