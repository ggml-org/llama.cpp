#include "parsers.h"

static void gigachat_v3_5_sort_tool_results(json & messages) {
    const auto is_result = [](const json & msg) {
        const auto role = msg.value("role", "");
        return role == "tool" || role == "function" || role == "function result" || role == "function results";
    };

    for (size_t i = 0; i < messages.size();) {
        if (!is_result(messages[i])) {
            ++i;
            continue;
        }

        const size_t first = i++;
        while (i < messages.size() && is_result(messages[i])) {
            ++i;
        }

        const json * prev = first > 0 ? &messages[first - 1] : nullptr;
        const auto role = prev ? prev->value("role", "") : "";
        const json * calls = role == "assistant" && prev->contains("tool_calls") ? &prev->at("tool_calls") : nullptr;
        if (!calls || !calls->is_array() || calls->empty()) {
            // legacy templates also accept a separate call role or assistant.function_call
            if (role == "function call" || (role == "assistant" && !prev->value("function_call", json()).empty())) {
                continue;
            }
            throw std::invalid_argument("GigaChat3.5: tool result without preceding tool calls");
        }
        if (calls->size() < 2 || calls->size() != i - first) {
            continue;
        }

        // the template matches by position when the ids are incomplete or ambiguous
        std::map<std::string, size_t> call_order;
        for (size_t j = 0; j < calls->size(); ++j) {
            const auto id = (*calls)[j].value("id", "");
            if (!id.empty()) {
                call_order.emplace(id, j);
            }
        }
        if (call_order.size() != calls->size()) {
            continue;
        }
        std::vector<json> replies(calls->size());
        size_t j = first;
        for (; j < i; ++j) {
            const auto it = call_order.find(messages[j].value("tool_call_id", ""));
            if (it == call_order.end() || !replies[it->second].is_null()) {
                break;
            }
            replies[it->second] = messages[j];
        }
        if (j == i) {
            for (size_t k = 0; k < replies.size(); ++k) {
                messages[first + k] = std::move(replies[k]);
            }
        }
    }
}

static bool gigachat_v3_5_allows_unconstrained_string(const common_chat_schema & schema) {
    auto plain_string = [](const common_chat_schema & node) {
        if (node.kind() != common_chat_schema::KIND_STRING) {
            return node.kind() == common_chat_schema::KIND_ANY;
        }
        const auto & str = node.as<common_chat_schema_string>();
        return str.pattern.empty() && str.format == common_chat_schema::FORMAT_NONE && str.min_length == 0 && str.max_length < 0;
    };
    if (schema.kind() == common_chat_schema::KIND_ANY_OF) {
        for (const auto & child : schema.as<common_chat_schema_any_of>().children) {
            if (plain_string(*child)) {
                return true;
            }
        }
        return false;
    }
    return plain_string(schema);
}

common_chat_params common_chat_params_init_gigachat_v3_5(
        const common_chat_template & tmpl,
        const autoparser::generation_params & inputs) {

    const std::string ROLE_SEP     = "<|role_sep|>\n";
    const std::string GEN_PROMPT   = "assistant" + ROLE_SEP;
    const std::string GCML         = "｜GCML｜";
    const std::string THINK_START  = "<think>";
    const std::string THINK_END    = "</think>";
    const std::string FC_START     = "<" + GCML + "tool_calls>";
    const std::string FC_END       = "</" + GCML + "tool_calls>";
    const std::string INVOKE_START = "<" + GCML + "invoke";
    const std::string INVOKE_END   = "</" + GCML + "invoke>";
    const std::string PARAM_START  = "<" + GCML + "parameter";
    const std::string PARAM_END    = "</" + GCML + "parameter>";

    auto render_inputs = inputs;
    if (!render_inputs.extra_context.is_object()) {
        render_inputs.extra_context = json::object();
    }
    if (!render_inputs.extra_context.contains("reasoning")) {
        render_inputs.extra_context["reasoning"] = inputs.enable_thinking;
    }
    // the templates match tool replies to calls by position
    gigachat_v3_5_sort_tool_results(render_inputs.messages);
    // the published templates expect reasoning in a separate field
    for (auto & msg : render_inputs.messages) {
        if (msg.value("role", "") != "assistant" || msg.contains("reasoning_content") ||
                !msg.contains("content") || !msg.at("content").is_string()) {
            continue;
        }
        const auto content = msg.at("content").get<std::string>();
        if (!string_starts_with(content, THINK_START)) {
            continue;
        }
        const size_t end = content.find(THINK_END, THINK_START.size());
        if (end == std::string::npos) {
            continue;
        }
        msg["reasoning_content"] = content.substr(THINK_START.size(), end - THINK_START.size());
        const size_t start = end + THINK_END.size();
        msg["content"] = content.substr(start + (content.compare(start, 2, "\n\n") == 0 ? 2 : 0));
    }

    common_chat_params data;

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, render_inputs);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, render_inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = { THINK_END };
    data.preserved_tokens   = {
        "<|message_sep|>\n\n",
        ROLE_SEP,
        GCML,
        THINK_START,
        THINK_END,
    };

    // lets the server find user turns in the prompt and place context checkpoints there
    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, GEN_PROMPT },
        { COMMON_CHAT_ROLE_USER,      "user" + ROLE_SEP },
    };

    // the reasoning template unconditionally opens a think block in the assistant turn
    const bool template_opens_think = string_ends_with(data.generation_prompt, THINK_START);

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PROMPT;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_REASONING) {
            data.generation_prompt += THINK_START + msg.reasoning_content;
        } else {
            // preserve the template's think block before continuation content
            const auto content = msg.render_content();
            if (!msg.reasoning_content.empty()) {
                data.generation_prompt += THINK_START + msg.reasoning_content + THINK_END + "\n\n";
            } else if (template_opens_think && !string_starts_with(content, THINK_START)) {
                data.generation_prompt += THINK_START + THINK_END + "\n\n";
            }
            data.generation_prompt += content;
        }

        data.prompt.append(data.generation_prompt);
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // start before <think> so the template's reasoning opener and continuation prefill are parsed
        auto generation_prompt = p.literal(GEN_PROMPT);
        auto end               = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning || has_response_format || has_tools) {
            auto text = p.until(THINK_END);
            if (extract_reasoning) {
                text = p.reasoning(text);
            }
            auto block = THINK_START + text + THINK_END + p.optional(p.literal("\n\n"));
            reasoning = p.optional(extract_reasoning ? block : p.content(block));
        }

        if (has_response_format) {
            return generation_prompt + reasoning + p.content(p.schema(p.json(), "response-format", inputs.json_schema)) + end;
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_choice = p.choice();
        common_chat_foreach_tool(inputs.tools, inputs.tool_choice, [&](size_t tool_index, const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const std::string tool_rule = "tool-" + std::to_string(tool_index);

            std::vector<common_peg_parser> required_parsers;
            std::vector<common_peg_parser> optional_parsers;
            foreach_parameter(function, [&](size_t param_index, const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                const auto types = param.schema->value_types();
                const std::string arg_rule = tool_rule + "-arg-" + std::to_string(param_index);

                auto arg_open = p.literal(PARAM_START + " name=\"") +
                    p.tool_arg_name(p.literal(param.name)) + p.literal("\" string=\"");
                const auto & schema = *param.schema;
                auto arg_value = p.choice();
                auto raw = p.tool_arg_open(arg_open + p.literal("true\">")) +
                           p.tool_arg_string_value(p.until(PARAM_END));
                bool use_json = !types.is_only(common_chat_schema::TYPE_STRING);
                if (types.has(common_chat_schema::TYPE_STRING)) {
                    std::vector<std::string> literals;
                    auto add_literal = [&](const json & value) {
                        if (value.is_string()) {
                            const auto str = value.get<std::string>();
                            if (str.find(PARAM_END) == std::string::npos && str.find('\0') == std::string::npos) {
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
                                     p.tool_arg_string_value(p.gbnf(p.until(PARAM_END), "(" + string_join(literals, " | ") + ")"));
                    } else if (gigachat_v3_5_allows_unconstrained_string(schema)) {
                        arg_value |= raw;
                    } else {
                        use_json = true;
                    }
                }
                if (use_json) {
                    auto encoded = p.rule(arg_rule + "-json", p.tool_arg_open(arg_open + p.literal("false\">")) +
                        p.tool_arg_json_value(p.schema(p.json(), arg_rule + "-schema", doc, schema)));
                    // complex string constraints use the GCML JSON form during sampling
                    arg_value |= types.has(common_chat_schema::TYPE_STRING)
                        ? p.gbnf(encoded | raw, arg_rule + "-json") : encoded;
                }
                auto arg = p.tool_arg(arg_value + p.tool_arg_close(p.literal(PARAM_END)));

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
                p.tool_open(p.literal(INVOKE_START + " name=\"") +
                            p.tool_name(p.literal(name)) + p.literal("\">\n")) +
                args_seq + p.space() +
                p.tool_close(p.literal(INVOKE_END)));

            tool_choice |= p.rule(tool_rule, func_parser);
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

        if (require_tools) {
            // no free text before a required call
            return generation_prompt + reasoning + p.space() + tool_calls + end;
        }

        auto content_before_tools = (extract_reasoning ? p.negate(p.literal(THINK_START)) : p.eps()) + p.content(p.until(FC_START));
        return generation_prompt + reasoning + content_before_tools + p.optional(tool_calls) + end;
    });

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        if (!has_response_format) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, FC_START },
            };
        }
    }

    return data;
}
