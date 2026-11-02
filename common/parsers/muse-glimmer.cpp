#include "parsers.h"

// An assistant turn is rendered as one or more messages, each
// "<|start|>assistant to=<recipient><|message|>{content}{END}" where END is
// <|eom|> (more messages follow) or <|eot|> (end of turn):
//   - chain-of-thought: to=self, terminated by <|eom|>
//   - final answer:     to=user, terminated by <|eot|>
// The generation prompt is just "<|start|>assistant"; the model emits its own
// " to=...<|message|>".
common_chat_params common_chat_params_init_muse_glimmer(const common_chat_template &          tmpl,
                                                               const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = "<|start|>assistant";
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    data.preserved_tokens = {
        "<|start|>", "<|message|>", "<|eom|>", "<|eot|>",
        // ATEM tool-call markup emitted on " to=<tool>" turns.
        "<atem:function_calls>", "<atem:invoke", "<atem:parameter", "</atem:parameter>",
        "</atem:invoke>", "</atem:function_calls>",
    };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|start|>assistant" },
        { COMMON_CHAT_ROLE_USER,      "<|start|>user"      },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|start|>system"    },
        { COMMON_CHAT_ROLE_TOOL,      "<|start|>tool"      },
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = "<|start|>assistant to=self<|message|>" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += "<|eom|><|start|>assistant to=user<|message|>" + msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    // Constrained grammar whenever tools are offered or a response format is requested.
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    data.parser = build_chat_peg_parser(inputs.specials, [&](common_chat_peg_builder & p) {
        auto start_token = p.token("<|start|>");
        auto message     = p.token("<|message|>");
        auto eom         = p.token("<|eom|>");
        auto eot         = p.token("<|eot|>");
        auto start       = p.rule("start", start_token + p.literal("assistant"));

        if (!extract_reasoning && !include_grammar) {
            return start + p.content(p.rest());
        }

        if (extract_reasoning) {
            p.rule("analysis", p.literal(" to=self") + message + p.reasoning(p.until(eom)) + eom);
        } else {
            p.rule("analysis", p.literal(" to=self") + message + p.content(p.until(eom)) + eom);
        }
        auto analysis = p.ref("analysis");

        auto recipient  = p.optional(p.literal(" to=user"));
        auto final_msg  = p.rule("final", recipient + message + p.content(p.until(eot | eom)));

        if (has_response_format) {
            auto response_json   = p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema));
            auto response_format = p.rule("response-format",
                recipient + message +
                ((p.literal("```json") + p.space() + response_json + p.space() + p.literal("```")) | response_json));

            return p.zero_or_more(start + analysis) + start + response_format;
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto string_value = p.through(p.tool_arg_string_value(p.until("</atem:parameter>")),
                                        p.tool_arg_close(p.literal("</atem:parameter>")));

            auto tool_choice = p.choice();
            std::vector<std::pair<common_peg_parser, common_peg_parser>> calls;
            foreach_function(inputs.tools, [&](size_t tool_index, const json & tool) {
                const auto &      function = tool.at("function");
                const std::string name     = function.at("name");

                std::vector<common_peg_parser> arg_rules;
                foreach_parameter(function, [&](size_t param_index, const common_chat_schema_property & prop, const common_chat_schema_document_ptr & doc) {
                    auto value_parser = p.eps();
                    if (prop.schema->may_be_string()) {
                        value_parser = string_value;
                    } else {
                        value_parser = p.tool_arg_json_value(
                                p.schema(p.json(), "tool-" + std::to_string(tool_index) + "-arg-" + std::to_string(param_index) + "-schema", doc, *prop.schema))
                            + p.tool_arg_close(p.literal("</atem:parameter>"));
                    }

                    arg_rules.push_back(p.tool_arg(
                        p.tool_arg_open(p.literal("<atem:parameter name=\"") + p.tool_arg_name(p.literal(prop.name)) + p.literal("\">")) +
                        value_parser));
                });

                auto args = p.eps();
                if (!arg_rules.empty()) {
                    args = p.zero_or_more(p.choice(arg_rules) + p.space());
                }

                // The recipient names the tool, so the header up to <|message|> is the trigger
                auto open = p.tool_open(start_token + p.literal("assistant to=") + p.tool_name(p.literal(name)) + message);
                auto body = p.rule("tool-" + std::to_string(tool_index),
                    p.literal("<atem:function_calls>") + p.space() + p.literal("<atem:invoke name=\"" + name + "\">") + p.space()
                    << p.tool_args(args)
                    << p.tool_close(p.literal("</atem:invoke>") + p.space() + p.literal("</atem:function_calls>")));

                tool_choice |= open + body;
                calls.emplace_back(open, body);
            });

            auto more       = inputs.parallel_tool_calls ? p.zero_or_more(eom + tool_choice) : p.eps();
            auto tool_calls = p.choice();
            for (size_t i = 0; i < calls.size(); i++) {
                tool_calls |= p.trigger_rule("tool-call-" + std::to_string(i), calls[i].first, calls[i].second + more);
            }

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return p.zero_or_more(start + analysis) + tool_calls;
            }
            auto trailing_calls = p.optional(eom + tool_calls);
            return p.zero_or_more(start + analysis) + (tool_calls | (start + final_msg + trailing_calls));
        }

        return p.zero_or_more(start + analysis) + start + final_msg;
    });

    if (include_grammar) {
        // The grammar scans for the triggers itself, so the sampler runs it from the first token
        data.grammar_lazy = false;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO);
        });
    }

    return data;
}
