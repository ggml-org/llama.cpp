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

        data.prompt += data.generation_prompt;
    }

    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    // Constrained grammar whenever tools are offered.
    auto include_grammar = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.rule("start", p.literal("<|start|>assistant"));

        if (!extract_reasoning && !include_grammar) {
            return start + p.content(p.rest());
        }

        if (extract_reasoning) {
            p.rule("analysis", p.literal(" to=self<|message|>") + p.reasoning(p.until("<|eom|>")) + p.literal("<|eom|>"));
        } else {
            p.rule("analysis", p.literal(" to=self<|message|>") + p.content(p.until("<|eom|>")) + p.literal("<|eom|>"));
        }
        auto analysis = p.ref("analysis");

        auto recipient  = p.optional(p.literal(" to=user"));
        auto final_msg  = p.rule("final", recipient + p.literal("<|message|>") +
                                              p.content(p.until_one_of({ "<|eot|>", "<|eom|>" })));

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto string_value = p.ac(
                p.tool_arg_string_value(p.until("</atem:parameter>")) + p.tool_arg_close(p.literal("</atem:parameter>")),
                "</atem:parameter>");

            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto &      function = tool.at("function");
                const std::string name     = function.at("name");
                auto              params   = function.contains("parameters") ? function.at("parameters") : json::object();

                auto args = p.eps();
                if (params.contains("properties") && params.at("properties").is_object() && !params.at("properties").empty()) {
                    auto schema_info = common_schema_info();
                    schema_info.resolve_refs(params);

                    auto arg_choice = p.choice();
                    for (const auto & [prop_name, prop_schema] : params.at("properties").items()) {
                        auto value_parser = p.eps();
                        if (schema_info.resolves_to_string(prop_schema)) {
                            value_parser = string_value;
                        } else {
                            value_parser = p.tool_arg_json_value(
                                    p.schema(p.json(), "tool-" + name + "-arg-" + prop_name + "-schema", prop_schema, false))
                                + p.tool_arg_close(p.literal("</atem:parameter>"));
                        }

                        auto arg_rule = p.tool_arg(
                            p.tool_arg_open(p.literal("<atem:parameter name=\"") + p.tool_arg_name(p.literal(prop_name)) + p.literal("\">")) +
                            value_parser);

                        arg_choice |= arg_rule;
                    }
                    args = p.zero_or_more(arg_choice + p.space());
                }

                auto tool_parser = p.tool(
                    p.tool_open(p.literal(" to=") + p.until("<|message|>") +
                                p.literal("<|message|><atem:function_calls>") + p.space() +
                                p.literal("<atem:invoke name=\"") + p.tool_name(p.literal(name)) + p.literal("\">") + p.space())
                    << p.tool_args(args)
                    << p.tool_close(p.literal("</atem:invoke>") + p.space() + p.literal("</atem:function_calls>")));

                tool_choice |= p.rule("tool-" + name, tool_parser);
            });

            auto tool_calls = inputs.parallel_tool_calls
                ? p.trigger_rule("tool-call", tool_choice + p.zero_or_more(p.literal("<|eom|>") + start + tool_choice))
                : p.trigger_rule("tool-call", tool_choice);


            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return p.zero_or_more(start + analysis) + start + tool_calls;
            }
            auto trailing_calls = p.optional(p.literal("<|eom|>") + start + tool_calls);
            return p.zero_or_more(start + analysis) + start + (tool_calls | (final_msg + trailing_calls));
        }

        return p.zero_or_more(start + analysis) + start + final_msg;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });
        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
              "<\\|start\\|>assistant( to=(?!self<\\|message\\|>)(?!user<\\|message\\|>)[^<]*?<\\|message\\|>)" },
        };
    }

    return data;
}
