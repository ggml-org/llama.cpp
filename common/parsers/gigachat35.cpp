#include "parsers.h"

static autoparser::generation_params common_chat_gigachat35_render_inputs(
        const autoparser::generation_params & inputs) {
    auto render_inputs = inputs;
    if (!render_inputs.extra_context.is_object()) {
        render_inputs.extra_context = json::object();
    }
    if (!render_inputs.extra_context.contains("reasoning")) {
        render_inputs.extra_context["reasoning"] = false;
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
            // assistant turns are laid out as "<think>...</think>\n\n{content}",
            // with an empty think block when the template forces reasoning
            if (!msg.reasoning_content.empty()) {
                data.generation_prompt += THINK_START + msg.reasoning_content + THINK_END + "\n\n";
            } else if (template_opens_think) {
                data.generation_prompt += THINK_START + THINK_END + "\n\n";
            }
            data.generation_prompt += msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto has_tools         = render_inputs.tools.is_array() && !render_inputs.tools.empty();
    auto extract_reasoning = render_inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && render_inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // anchor on the bare generation prompt: the "<think>" opener rendered by the template
        // and any continuation prefill are re-parsed, so prefilled reasoning/content ends up
        // in the parsed message
        auto generation_prompt = p.literal(GEN_PROMPT);
        auto end               = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning) {
            reasoning = p.optional(
                THINK_START +
                p.reasoning(p.until(THINK_END)) +
                THINK_END +
                p.optional(p.literal("\n\n")));
        }

        if (!has_tools || render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_choice = p.choice();
        foreach_function(render_inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");

            std::vector<common_peg_parser> required_parsers;
            std::vector<common_peg_parser> optional_parsers;
            foreach_parameter(function, [&](const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                // the template renders string="true" with a raw value whenever the parameter
                // can be a string, and string="false" with a JSON-encoded value otherwise
                bool is_string = param.schema->value_types().has(common_chat_schema::TYPE_STRING);

                auto arg = p.tool_arg(
                    p.tool_arg_open(
                        p.literal(PARAM_START + " name=\"") +
                        p.tool_arg_name(p.literal(param.name)) +
                        p.literal("\" string=\"" + std::string(is_string ? "true" : "false") + "\">")) +
                    (is_string
                         ? p.tool_arg_string_value(p.until(PARAM_END))
                         : p.tool_arg_json_value(p.schema(p.json(),
                                                          "tool-" + name + "-arg-" + param.name + "-schema",
                                                          doc, *param.schema))) +
                    p.tool_arg_close(p.literal(PARAM_END)));

                auto named_arg = p.rule("tool-" + name + "-arg-" + param.name, arg);
                if (param.required) {
                    required_parsers.push_back(named_arg);
                } else {
                    optional_parsers.push_back(named_arg);
                }
            });

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

            auto func_parser = p.tool(
                p.tool_open(p.literal(INVOKE_START + " name=\"") +
                            p.tool_name(p.literal(name)) + p.literal("\">\n")) +
                args_seq + p.space() +
                p.tool_close(p.literal(INVOKE_END)));

            tool_choice |= p.rule("tool-" + name, func_parser);
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

        auto content_before_tools = p.negate(p.literal(THINK_START)) + p.content(p.until(FC_START));
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, FC_START },
        };
    }

    return data;
}
