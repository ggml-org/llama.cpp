#include "parsers.h"

static autoparser::generation_params common_chat_gigachat35_render_inputs(
        const autoparser::generation_params & inputs) {
    auto render_inputs = inputs;
    if (!render_inputs.extra_context.is_object()) {
        render_inputs.extra_context = json::object();
    }
    if (!render_inputs.extra_context.contains("reasoning")) {
        render_inputs.extra_context["reasoning"] = inputs.enable_thinking;
    }
    return render_inputs;
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

    // the reasoning template unconditionally opens a think block in the assistant turn
    const bool template_opens_think =
        data.generation_prompt.size() >= THINK_START.size() &&
        data.generation_prompt.compare(data.generation_prompt.size() - THINK_START.size(),
                                       THINK_START.size(), THINK_START) == 0;

    if (render_inputs.has_continuation()) {
        const auto & msg = render_inputs.continue_msg;

        data.generation_prompt = GEN_PROMPT;
        if (render_inputs.continue_final_message == COMMON_CHAT_CONTINUATION_REASONING) {
            data.generation_prompt += THINK_START + msg.reasoning_content;
        } else {
            // preserve the template's think block before continuation content
            if (!msg.reasoning_content.empty()) {
                data.generation_prompt += THINK_START + msg.reasoning_content + THINK_END + "\n\n";
            } else if (template_opens_think) {
                data.generation_prompt += THINK_START + THINK_END + "\n\n";
            }
            data.generation_prompt += msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    auto has_tools           = render_inputs.tools.is_array() && !render_inputs.tools.empty();
    auto has_response_format = render_inputs.json_schema.is_object();
    auto extract_reasoning   = render_inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && render_inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // start before <think> so the template's reasoning opener and continuation prefill are parsed
        auto generation_prompt = p.literal(GEN_PROMPT);
        auto end               = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning || has_response_format) {
            auto text = p.until(THINK_END);
            if (extract_reasoning) {
                text = p.reasoning(text);
            }
            reasoning = p.optional(
                THINK_START +
                text +
                THINK_END +
                p.optional(p.literal("\n\n")));
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

                auto arg_open = p.literal(PARAM_START + " name=\"") +
                    p.tool_arg_name(p.literal(param.name)) + p.literal("\" string=\"");
                auto arg_value = p.choice();
                if (types.has(common_chat_schema::TYPE_STRING)) {
                    arg_value |= p.tool_arg_open(arg_open + p.literal("true\">")) +
                        p.tool_arg_string_value(p.until(PARAM_END));
                }
                if (!types.is_only(common_chat_schema::TYPE_STRING)) {
                    arg_value |= p.tool_arg_open(arg_open + p.literal("false\">")) +
                        p.tool_arg_json_value(p.schema(p.json(), arg_rule + "-schema", doc, *param.schema));
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

        auto require_tools = render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();
        if (render_inputs.parallel_tool_calls) {
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

        auto content_before_tools = (extract_reasoning ? p.negate(p.literal(THINK_START)) : p.eps()) + p.content(p.until(FC_START));
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
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
