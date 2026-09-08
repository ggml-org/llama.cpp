#include "parsers.h"

// The DeepSeek V4 reference implementation renders consecutive tool results into a single
// user block, ordered by the tool call order of the preceding assistant message (matched
// by tool call id) rather than by the order they appear in the conversation.
static json deepseek_v4_sort_tool_results(const json & messages) {
    json adjusted = messages;
    std::map<std::string, size_t> call_order;

    for (size_t i = 0; i < adjusted.size();) {
        const auto & msg  = adjusted[i];
        const auto   role = msg.value("role", "");

        if (role == "assistant" && msg.contains("tool_calls") &&
                msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty()) {
            call_order.clear();
            const auto & tool_calls = msg.at("tool_calls");
            for (size_t idx = 0; idx < tool_calls.size(); idx++) {
                auto id = tool_calls[idx].value("id", "");
                if (!id.empty()) {
                    call_order[id] = idx;
                }
            }
            i++;
            continue;
        }

        if (role != "user" && role != "tool") {
            i++;
            continue;
        }

        // collect a maximal run of user/tool messages - they render into one user block
        std::vector<size_t> tool_positions;
        size_t run_end = i;
        for (; run_end < adjusted.size(); run_end++) {
            const auto r = adjusted[run_end].value("role", "");
            if (r == "tool") {
                tool_positions.push_back(run_end);
            } else if (r != "user") {
                break;
            }
        }

        if (tool_positions.size() > 1 && !call_order.empty()) {
            std::vector<json> results;
            results.reserve(tool_positions.size());
            for (auto pos : tool_positions) {
                results.push_back(adjusted[pos]);
            }
            std::stable_sort(results.begin(), results.end(), [&](const json & a, const json & b) {
                const auto order = [&](const json & m) {
                    auto it = call_order.find(m.value("tool_call_id", ""));
                    return it == call_order.end() ? (size_t) 0 : it->second;
                };
                return order(a) < order(b);
            });
            for (size_t k = 0; k < tool_positions.size(); k++) {
                adjusted[tool_positions[k]] = std::move(results[k]);
            }
        }

        i = run_end;
    }

    return adjusted;
}

common_chat_params common_chat_params_init_deepseek_v3_2(const common_chat_template &    tmpl,
                                                                 const autoparser::generation_params & inputs) {
    common_chat_params data;

    // V4 uses the same DSML markup as V3.2, but names the tool call block "tool_calls"
    // instead of "function_calls", renders tool results in tool call order and its
    // non-thinking generation prompt ends with a bare </think> instead of an empty
    // <think></think> pair.
    const bool is_v4 = tmpl.source().find("function_calls") == std::string::npos;

    std::optional<json> adjusted_messages;
    if (is_v4) {
        adjusted_messages = deepseek_v4_sort_tool_results(inputs.messages);
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    std::optional<json> additional_context;
    if (is_v4 && has_response_format) {
        additional_context = json{ { "response_format", inputs.json_schema } };
    }

    const std::string DSML         = "｜DSML｜";
    const std::string THINK_START  = "<think>";
    const std::string THINK_END    = "</think>";
    const std::string TC_BLOCK     = is_v4 ? "tool_calls" : "function_calls";
    const std::string FC_START     = "<" + DSML + TC_BLOCK + ">";
    const std::string FC_END       = "</" + DSML + TC_BLOCK + ">";
    const std::string INVOKE_START = "<" + DSML + "invoke";
    const std::string INVOKE_END   = "</" + DSML + "invoke>";
    const std::string PARAM_START  = "<" + DSML + "parameter";
    const std::string PARAM_END    = "</" + DSML + "parameter>";
    const std::string GEN_PROMPT   = "<｜Assistant｜>";
    const std::string TC_SEPARATOR = "\n\n";

    data.prompt = common_chat_template_direct_apply_impl(
        tmpl, inputs, adjusted_messages, std::nullopt, additional_context);
    data.generation_prompt = common_chat_template_generation_prompt_impl(
        tmpl, inputs, adjusted_messages, std::nullopt, additional_context);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = {THINK_END, FC_START};
    data.preserved_tokens   = {
        DSML,
        THINK_START,
        THINK_END,
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        if (is_v4 && msg.reasoning_content.empty()) {
            data.generation_prompt = GEN_PROMPT + THINK_END;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += msg.render_content();
            }
        } else {
            data.generation_prompt = GEN_PROMPT + THINK_START + msg.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += THINK_END + msg.render_content();
            }
        }

        data.prompt += data.generation_prompt;
    }

    bool require_tools   = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    bool has_tool_calls = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PROMPT);
        auto end               = p.end();

        // build tool call section first since we might need it in reasoning
        auto tool_choice = p.choice();
        if (has_tool_calls) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                auto         params   = function.contains("parameters") ? function.at("parameters") : json::object();
                const auto & props    = params.contains("properties") ? params.at("properties") : json::object();

                std::set<std::string> required;
                if (params.contains("required")) {
                    required = params.at("required").get<std::set<std::string>>();
                }

                auto schema_info = common_schema_info();
                schema_info.resolve_refs(params);

                std::vector<common_peg_parser> required_parsers;
                std::vector<common_peg_parser> optional_parsers;
                for (const auto & [param_name, param_schema] : props.items()) {
                    bool is_required = required.find(param_name) != required.end();
                    bool is_string   = schema_info.resolves_to_string(param_schema);

                    auto arg = p.tool_arg(
                        p.tool_arg_open(p.literal(PARAM_START + " name=\"") + p.tool_arg_name(p.literal(param_name)) +
                                        p.literal("\" string=\"" + std::string(is_string ? "true" : "false") + "\">")) +
                        (is_string ?
                             p.tool_arg_string_value(p.until(PARAM_END)) :
                             p.tool_arg_json_value(p.schema(p.json(), "tool-" + name + "-arg-" + param_name + "-schema",
                                                            param_schema, false))) +
                        p.tool_arg_close(p.literal(PARAM_END)));

                    auto named_arg = p.rule("tool-" + name + "-arg-" + param_name, arg);
                    if (is_required) {
                        required_parsers.push_back(named_arg);
                    } else {
                        optional_parsers.push_back(named_arg);
                    }
                }

                common_peg_parser args_seq = p.eps();
                for (size_t i = 0; i < required_parsers.size(); i++) {
                    if (i > 0) {
                        args_seq = args_seq + p.space();
                    }
                    args_seq = args_seq + required_parsers[i];
                }

                if (!optional_parsers.empty()) {
                    common_peg_parser any_opt = p.choice();
                    for (const auto & opt : optional_parsers) {
                        any_opt |= opt;
                    }
                    args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
                }

                common_peg_parser invoke_body = args_seq;
                auto              func_parser = p.tool(p.tool_open(p.literal(INVOKE_START + " name=\"") +
                                                                   p.tool_name(p.literal(name)) + p.literal("\">\n")) +
                                                       invoke_body + p.space() + p.tool_close(p.literal(INVOKE_END)));

                tool_choice |= p.rule("tool-" + name, func_parser);
            });
        }

        common_peg_parser tool_calls = p.eps();
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice +
                p.zero_or_more(p.space() + tool_choice) + p.space() + p.literal(FC_END));
        } else {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice + p.space() + p.literal(FC_END));
        }

        auto reasoning = p.eps();
        auto reasoning_with_tc = p.eps();
        auto obligatory_tool_calls = tool_calls;
        bool allow_reasoning_with_tc = false;

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        if (extract_reasoning && inputs.enable_thinking) {
            reasoning = p.optional(THINK_START + p.reasoning(p.until(THINK_END)) + THINK_END);
            reasoning_with_tc = THINK_START +
                p.reasoning(p.until_one_of({ TC_SEPARATOR + FC_START, FC_START, THINK_END })) +
                p.space() + obligatory_tool_calls;
            allow_reasoning_with_tc = true;
        } else if (extract_reasoning) {
            // Thinking disabled but reasoning extraction requested: the generation prompt
            // contains an empty <think></think> pair (V3.2) or a bare </think> (V4) that
            // must still be consumed.
            reasoning = is_v4
                ? p.optional(p.literal(THINK_END))
                : p.optional(p.literal(THINK_START) + p.until(THINK_END) + p.literal(THINK_END));
        }

        if (has_response_format) {
            auto response_format = p.rule("response-format",
                p.literal("```json") + p.space() +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                p.space() + p.literal("```"));
            return generation_prompt + reasoning + response_format + end;
        }

        if (!has_tool_calls) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto content_before_tools = p.negate(p.literal(THINK_START)) +
            p.content(p.until_one_of({ TC_SEPARATOR + FC_START, FC_START })) +
            p.space();
        return allow_reasoning_with_tc ? generation_prompt + (reasoning_with_tc | (reasoning + content_before_tools + tool_calls)) + end :
            generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && !require_tools;
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
