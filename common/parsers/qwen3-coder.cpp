#include "parsers.h"

common_chat_params common_chat_params_init_qwen3_coder(const common_chat_template &          tmpl,
                                                              const autoparser::generation_params & inputs) {
    common_chat_params data;

    const std::string GEN_PREFIX = "<|im_start|>assistant\n";

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;

    auto supports_reasoning = tmpl.source().find("<think>") != std::string::npos;

    data.supports_thinking = supports_reasoning;
    data.preserved_tokens  = {
        "<tool_call>",
        "</tool_call>",
    };

    auto is_qwen3_coder  = !supports_reasoning;

    if (supports_reasoning) {
        data.thinking_start_tag = "<think>";
        // Support both </think> and <tool_call> as reasoning end sequences.
        // <function= is omitted, as it is a workaround for Qwen3-Coder which is not a thinking model
        data.thinking_end_tags = { "</think>", "<tool_call>" };
        data.preserved_tokens.insert(data.preserved_tokens.end(), { "<think>", "</think>" });
    }

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|im_start|>assistant"             },
        { COMMON_CHAT_ROLE_TOOL,      "<|im_start|>user\n<tool_response>" }, // Qwen3-Coder, Qwen3.5, Nemotron Nano 3
        { COMMON_CHAT_ROLE_TOOL,      "<|im_start|>tool_response"         }, // StepFun-3.5-Flash
        { COMMON_CHAT_ROLE_USER,      "<|im_start|>user"                  },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|im_start|>system"                },
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PREFIX;
        if (supports_reasoning) {
            data.generation_prompt += "<think>\n" + msg.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += "\n</think>\n\n";
            }
        }
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    std::vector<std::string> tool_call_starts = { "<tool_call>" };

    if (is_qwen3_coder) {
        // Match complete <function=name> opener for Qwen3-Coder models that occasionally omit the
        // starting <tool_call>. The model may hallucinate a tool name, but it is preferable over
        // constraining on <function which may occur in valid content generation, e.g. #include <functional>
        foreach_function(inputs.tools, [&](const json & tool) {
            const std::string name = tool.at("function").at("name");
            tool_call_starts.push_back("<function=" + name + ">");
        });
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PREFIX);

        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            reasoning = p.optional("<think>" + p.space() +
                                   p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                   (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
        }

        // Response format parser
        if (has_response_format) {
            return generation_prompt + (reasoning << p.content(p.schema(p.json(), "response-format", inputs.json_schema)));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto arg_close  = p.tool_arg_close(p.literal("\n</parameter>\n"));
            auto arg_string = p.rule("xml-arg-string",
                p.ac(p.tool_arg_string_value(p.until("\n</parameter>\n")) + arg_close, "\n</parameter>\n"));

            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function   = tool.at("function");
                std::string  name       = function.at("name");
                auto         parameters = function.contains("parameters") ? function.at("parameters") : json::object();

                auto schema_info = common_schema_info();
                schema_info.resolve_refs(parameters);

                std::vector<common_peg_parser> required_args;
                std::vector<common_peg_parser> optional_args;

                foreach_parameter(function, [&](const std::string & param_name, const json & param_schema, bool is_required) {
                    auto rule_name = "tool-" + name + "-arg-" + param_name;

                    auto arg_open = p.tool_arg_open("<parameter=" + p.tool_arg_name(p.literal(param_name)) + ">\n");

                    auto arg_value = schema_info.resolves_to_string(param_schema) ?
                        arg_string :
                        p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", param_schema)) + arg_close;

                    auto arg_rule = p.rule(rule_name, p.tool_arg(arg_open + arg_value));

                    (is_required ? required_args : optional_args).push_back(arg_rule);
                });

                // Accept required arguments in any order, as Qwen does not always adhere to the
                // order provided.
                auto args = p.permute("tool-" + name + "-args", required_args);
                if (!optional_args.empty()) {
                    args = args + p.zero_or_more(p.choice(optional_args));
                }

                auto func = p.tool(p.tool_open("<function=" + p.tool_name(p.literal(name)) + ">\n") +
                                   p.tool_args(args) +
                                   p.tool_close(p.literal("</function>\n")));

                tool_choice |= p.rule("tool-" + name, func);
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;

            auto tool_call_body = tool_choice + "</tool_call>" + p.space();
            auto tool_call      = p.rule("tool-call", "<tool_call>\n" + tool_call_body);

            // Qwen3-Coder models may occasionally omit the <tool_call> token.
            auto tool_call_first = is_qwen3_coder ?
                p.rule("tool-call-first", p.optional(p.literal("<tool_call>\n")) + tool_call_body) :
                tool_call;

            auto calls      = inputs.parallel_tool_calls ? tool_call_first + p.zero_or_more(tool_call) : tool_call_first;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(calls, min_calls, 1));

            return generation_prompt +
                   (reasoning << p.content(p.until_one_of(tool_call_starts)) << tool_calls);
        }

        // Content only parser
        return generation_prompt + (reasoning << p.content(p.rest()));
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
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

        if (data.grammar_lazy) {
            for (const auto & start : tool_call_starts) {
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, start });
            }
        }
    }

    return data;
}
