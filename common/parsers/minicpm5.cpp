#include "parsers.h"

// MiniCPM5 format:
// - Reasoning: <think>{reasoning}</think> (optional)
// - Tool calls: <function name="foo"><param name="bar">value</param></function>
common_chat_params common_chat_params_init_minicpm5(const common_chat_template &          tmpl,
                                                           const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.preserved_tokens  = {
        "<function",
        "<param",
        "</function>",
        "</param>",
        "<think>",
        "</think>",
    };

    data.thinking_start_tag = "<think>";
    data.thinking_end_tags  = {"</think>"};

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|im_start|>assistant"             },
        { COMMON_CHAT_ROLE_TOOL,      "<|im_start|>user\n<tool_response>" },
        { COMMON_CHAT_ROLE_USER,      "<|im_start|>user"                  },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|im_start|>system"                },
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = "<|im_start|>assistant\n<think>\n" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += "\n</think>\n\n" + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal("<|im_start|>assistant\n");

        auto reasoning = p.eps();
        if (extract_reasoning) {
            reasoning = ("<think>" << p.reasoning(p.until("</think>")) << "</think>") + p.space();
        }

        // Response format parser
        if (has_response_format) {
            return generation_prompt + reasoning + p.content(p.schema(p.json(), "response-format", inputs.json_schema));
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // CDATA lets a value carry characters that would otherwise close the tag (e.g.
            // </param>); capture the inner text only, excluding the CDATA markers.
            auto string_value = p.choice({
                p.literal("<![CDATA[") + p.ac(p.tool_arg_string_value(p.until("]]>")) + p.literal("]]>"), "]]>") + p.tool_arg_close(p.literal("</param>")),
                p.negate(p.literal("<![CDATA[")) + p.ac(p.tool_arg_string_value(p.until("</param>")) + p.tool_arg_close(p.literal("</param>")), "</param>")
            });

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
                                    p.schema(p.json(), "tool-" + name + "-arg-" + prop_name + "-schema", prop_schema, false)
                                ) + p.tool_arg_close(p.literal("</param>"));
                        }

                        auto arg_rule = p.tool_arg(
                            p.tool_arg_open(p.literal("<param name=\"") + p.tool_arg_name(p.literal(prop_name)) + p.literal("\">")) +
                            value_parser
                        );

                        arg_choice |= arg_rule;
                    }
                    args = p.zero_or_more(arg_choice + p.space());
                }

                auto tool_parser = p.tool(
                    p.tool_open(p.literal("<function name=\"") + p.tool_name(p.literal(name)) + p.literal("\">"))
                    << p.tool_args(args)
                    << p.tool_close(p.literal("</function>")));

                tool_choice |= p.rule("tool-" + name, tool_parser);
            });

            auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call", p.repeat(tool_choice + p.space(), 1, max_calls));

            auto content = p.content(p.until("<function"));

            return generation_prompt + reasoning + content + tool_calls + p.end();
        }

        return generation_prompt + reasoning + p.content(p.rest()) + p.end();
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
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<function" },
        };
    }

    return data;
}
