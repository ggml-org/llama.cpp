#include "parsers.h"

static void common_chat_gigachat35_sort_tool_results(json & messages) {
    const auto is_result = [](const json & msg) {
        const auto role = msg.value("role", "");
        return role == "tool" || role == "function" || role == "function result" || role == "function results";
    };
    std::map<std::string, size_t> call_order;
    size_t n_calls = 0;
    for (size_t i = 0; i < messages.size();) {
        const auto & msg = messages[i];
        const auto role = msg.value("role", "");
        if (role == "assistant") {
            call_order.clear();
            n_calls = 0;
            if (msg.contains("tool_calls") && msg.at("tool_calls").is_array()) {
                const auto & calls = msg.at("tool_calls");
                n_calls = calls.size();
                for (size_t j = 0; j < n_calls; ++j) {
                    const auto id = calls[j].value("id", "");
                    if (!id.empty() && !call_order.emplace(id, j).second) {
                        throw std::invalid_argument("Duplicate tool call ID: " + id);
                    }
                }
            }
        }
        if (role == "function call" || role == "reasoning function call") {
            // The template combines native calls with the preceding assistant.
            n_calls = 0;
        }
        if (!is_result(msg) || n_calls == 0) {
            ++i;
            continue;
        }

        size_t end = i;
        bool packed = false;
        while (end < messages.size() && is_result(messages[end])) {
            packed |= messages[end].contains("content") && messages[end].at("content").is_array();
            ++end;
        }
        if (packed) {
            // Native result arrays are expanded by the template.
            i = end;
            continue;
        }
        if (end - i != n_calls) {
            throw std::invalid_argument("Tool response count does not match preceding tool calls");
        }
        std::vector<json> replies(n_calls);
        for (size_t j = i; j < end; ++j) {
            size_t index = j - i;
            const auto id = messages[j].value("tool_call_id", "");
            if (!id.empty()) {
                const auto it = call_order.find(id);
                if (it == call_order.end()) {
                    throw std::invalid_argument("Unknown tool call ID: " + id);
                }
                index = it->second;
            }
            if (!replies[index].is_null()) {
                throw std::invalid_argument("Duplicate tool response for call " + std::to_string(index));
            }
            replies[index] = messages[j];
        }
        for (size_t j = 0; j < n_calls; ++j) {
            messages[i + j] = std::move(replies[j]);
        }
        i = end;
    }
}

static autoparser::generation_params common_chat_gigachat35_render_inputs(
        const autoparser::generation_params & inputs) {
    auto render_inputs = inputs;
    if (!render_inputs.extra_context.is_object()) {
        render_inputs.extra_context = json::object();
    }
    if (!render_inputs.extra_context.contains("reasoning")) {
        render_inputs.extra_context["reasoning"] = inputs.enable_thinking;
    }
    // Older embedded templates match tool replies by position.
    common_chat_gigachat35_sort_tool_results(render_inputs.messages);
    // Older embedded templates expect reasoning in a separate field.
    for (auto & msg : render_inputs.messages) {
        if (msg.value("role", "") != "assistant" || !msg.contains("content") || !msg.at("content").is_string()) {
            continue;
        }
        if (msg.contains("reasoning_content") && !msg.at("reasoning_content").is_null()) {
            continue;
        }
        const auto content = msg.at("content").get<std::string>();
        if (content.compare(0, 7, "<think>") != 0) {
            continue;
        }
        const size_t end = content.find("</think>", 7);
        if (end == std::string::npos) {
            continue;
        }
        msg["reasoning_content"] = content.substr(7, end - 7);
        const size_t start = end + 8;
        msg["content"] = content.substr(start + (content.compare(start, 2, "\n\n") == 0 ? 2 : 0));
    }
    return render_inputs;
}

static bool allows_unconstrained_string(const common_chat_schema & schema) {
    auto plain_string = [](const common_chat_schema & node) {
        if (node.kind() != common_chat_schema::KIND_STRING) {
            return node.kind() == common_chat_schema::KIND_ANY;
        }
        const auto & str = node.as<common_chat_schema_string>();
        return str.pattern.empty() && str.format == common_chat_schema::FORMAT_NONE && str.min_length == 0 && str.max_length < 0;
    };
    if (schema.kind() == common_chat_schema::KIND_ANY_OF) {
        const auto & children = schema.as<common_chat_schema_any_of>().children;
        return std::any_of(children.begin(), children.end(), [&](const auto & child) { return plain_string(*child); });
    }
    return plain_string(schema);
}

common_chat_params common_chat_params_init_gigachat35(
        const common_chat_template & tmpl,
        const autoparser::generation_params & inputs) {

    auto render_inputs = common_chat_gigachat35_render_inputs(inputs);

    common_chat_params data;

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, render_inputs);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, render_inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<think>";
    data.thinking_end_tags  = {"</think>"};
    data.preserved_tokens   = {
        "<|message_sep|>\n\n",
        "<|role_sep|>\n",
        "｜GCML｜",
        "<think>",
        "</think>",
    };

    const std::string role_sep     = "<|role_sep|>\n";
    const std::string gen_prompt   = "assistant" + role_sep;
    const std::string gcml         = "｜GCML｜";
    const std::string think_start  = "<think>";
    const std::string think_end    = "</think>";
    const std::string fc_start     = "<" + gcml + "tool_calls>";
    const std::string fc_end       = "</" + gcml + "tool_calls>";
    const std::string invoke_start = "<" + gcml + "invoke";
    const std::string invoke_end   = "</" + gcml + "invoke>";
    const std::string param_start  = "<" + gcml + "parameter";
    const std::string param_end    = "</" + gcml + "parameter>";

    // the reasoning template unconditionally opens a think block in the assistant turn
    const bool template_opens_think =
        data.generation_prompt.size() >= think_start.size() &&
        data.generation_prompt.compare(data.generation_prompt.size() - think_start.size(),
                                       think_start.size(), think_start) == 0;

    if (render_inputs.has_continuation()) {
        const auto & msg = render_inputs.continue_msg;

        data.generation_prompt = gen_prompt;
        if (render_inputs.continue_final_message == COMMON_CHAT_CONTINUATION_REASONING) {
            data.generation_prompt += think_start + msg.reasoning_content;
        } else {
            // preserve the template's think block before continuation content
            const auto content = msg.render_content();
            if (msg.has_reasoning_content()) {
                data.generation_prompt += think_start + msg.reasoning_content + think_end + "\n\n";
            } else if (template_opens_think && content.compare(0, think_start.size(), think_start) != 0) {
                data.generation_prompt += think_start + think_end + "\n\n";
            }
            data.generation_prompt += content;
        }

        data.prompt.append(data.generation_prompt);
    }

    auto has_tools           = render_inputs.tools.is_array() && !render_inputs.tools.empty();
    auto has_response_format = render_inputs.json_schema.is_object();
    auto extract_reasoning   = render_inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && render_inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // start before <think> so the template's reasoning opener and continuation prefill are parsed
        auto generation_prompt = p.literal(gen_prompt);
        auto end               = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning || has_response_format || has_tools) {
            auto text = p.until(think_end);
            if (extract_reasoning) {
                text = p.reasoning(text);
            }
            auto block = think_start + text + think_end + p.optional(p.literal("\n\n"));
            reasoning = p.optional(extract_reasoning ? block : p.content(block));
        }

        if (has_response_format) {
            return generation_prompt + reasoning + p.content(p.schema(p.json(), "response-format", render_inputs.json_schema)) + end;
        }

        if (!has_tools || render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_choice = p.choice();
        foreach_function(render_inputs.tools, [&](size_t tool_index, const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const std::string tool_rule = "tool-" + std::to_string(tool_index);

            std::vector<common_peg_parser> required_parsers;
            std::vector<common_peg_parser> optional_parsers;
            foreach_parameter(function, [&](size_t param_index, const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                // string parameters use raw values; other types use JSON
                const auto types = param.schema->value_types();
                const std::string arg_rule = tool_rule + "-arg-" + std::to_string(param_index);

                auto arg_open = p.literal(param_start + " name=\"") +
                    p.tool_arg_name(p.literal(param.name)) + p.literal("\" string=\"");
                const auto & schema = *param.schema;
                auto arg_value = p.choice();
                auto raw = p.tool_arg_open(arg_open + p.literal("true\">")) +
                           p.tool_arg_string_value(p.until(param_end));
                bool use_json = !types.is_only(common_chat_schema::TYPE_STRING);
                if (types.has(common_chat_schema::TYPE_STRING)) {
                    std::vector<std::string> literals;
                    auto add_literal = [&](const json & value) {
                        if (value.is_string()) {
                            const auto str = value.get<std::string>();
                            if (str.find(param_end) == std::string::npos && str.find('\0') == std::string::npos) {
                                literals.push_back(gbnf_format_literal(str));
                            } else {
                                use_json = true;
                            }
                        }
                    };
                    if (schema.kind() == common_chat_schema::KIND_ENUM) {
                        for (const auto & value : schema.as<common_chat_schema_enum>().values) {
                            add_literal(value);
                        }
                    } else if (schema.kind() == common_chat_schema::KIND_CONST) {
                        add_literal(schema.as<common_chat_schema_const>().value);
                    }
                    if (!literals.empty()) {
                        arg_value |= p.tool_arg_open(arg_open + p.literal("true\">")) +
                                     p.tool_arg_string_value(p.gbnf(p.until(param_end), "(" + string_join(literals, " | ") + ")"));
                    } else if (allows_unconstrained_string(schema)) {
                        arg_value |= raw;
                    } else {
                        use_json = true;
                    }
                }
                if (use_json) {
                    auto encoded = p.rule(arg_rule + "-json", p.tool_arg_open(arg_open + p.literal("false\">")) +
                        p.tool_arg_json_value(p.schema(p.json(), arg_rule + "-schema", doc, schema)));
                    // Complex string constraints use GCML's JSON form during sampling.
                    arg_value |= types.has(common_chat_schema::TYPE_STRING)
                        ? p.gbnf(encoded | raw, arg_rule + "-json") : encoded;
                }
                auto arg = p.tool_arg(arg_value + p.tool_arg_close(p.literal(param_end)));

                auto named_arg = p.rule(arg_rule, arg);
                if (param.required) {
                    required_parsers.push_back(named_arg);
                } else {
                    optional_parsers.push_back(named_arg);
                }
            });

            auto optional_args = p.eps();
            if (!optional_parsers.empty()) {
                optional_args = p.zero_or_more(p.choice(optional_parsers) + p.space());
            }
            for (auto & arg : required_parsers) {
                arg = arg + p.space() + optional_args;
            }
            auto args_seq = p.space() + optional_args + p.permute(tool_rule + "-args", required_parsers);

            auto func_parser = p.tool(
                p.tool_open(p.literal(invoke_start + " name=\"") +
                            p.tool_name(p.literal(name)) + p.literal("\">\n")) +
                args_seq + p.space() +
                p.tool_close(p.literal(invoke_end)));

            tool_choice |= p.rule(tool_rule, func_parser);
        });

        auto require_tools = render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();
        if (render_inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(fc_start) + p.space() + tool_choice +
                p.zero_or_more(p.space() + tool_choice) + p.space() + p.literal(fc_end));
        } else {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(fc_start) + p.space() + tool_choice + p.space() + p.literal(fc_end));
        }

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        auto content_before_tools = (extract_reasoning ? p.negate(p.literal(think_start)) : p.eps()) + p.content(p.until(fc_start));
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        if (!has_response_format) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, fc_start },
            };
        }
    }

    return data;
}
