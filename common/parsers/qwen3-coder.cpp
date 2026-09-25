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
        // The newline variant comes first so it is included in the forced message
        // <function= is omitted, as it is a workaround for Qwen3-Coder which is not a thinking model
        data.thinking_end_tags = { "\n</think>", "</think>", "<tool_call>" };
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

    auto parser = build_chat_peg_parser(inputs.token_table, [&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PREFIX);

        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            reasoning = p.optional(p.token("<think>") + p.space() +
                                   p.reasoning(p.until({ p.token("</think>"), p.token("<tool_call>") })) +
                                   (p.token("</think>") | p.peek(p.token("<tool_call>"))));
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

            struct function_parsers {
                std::string       name;
                common_peg_parser opener;
                common_peg_parser body;
            };
            std::vector<function_parsers> functions;

            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");

                std::vector<common_peg_parser> required_args;
                std::vector<common_peg_parser> optional_args;

                foreach_parameter(function, [&](const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                    auto rule_name = "tool-" + name + "-arg-" + param.name;

                    auto arg_open = p.tool_arg_open("<parameter=" + p.tool_arg_name(p.literal(param.name)) + ">\n");

                    auto types = param.schema->value_types();

                    auto arg_value = p.eps();
                    if (!types.has(common_chat_schema::TYPE_STRING)) {
                        arg_value = p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", doc, *param.schema)) + arg_close;
                    } else if (types.is_only(common_chat_schema::TYPE_STRING)) {
                        arg_value = arg_string;
                    } else {
                        // The string alternative accepts any text, so the grammar only keeps the raw string
                        // rule. The parser still tries the JSON alternatives first to type the value.
                        auto json_value = p.choice();
                        if (types.has(common_chat_schema::TYPE_OBJECT)) {
                            json_value |= p.json_object();
                        }
                        if (types.has(common_chat_schema::TYPE_ARRAY)) {
                            json_value |= p.json_array();
                        }
                        if (types.has(common_chat_schema::TYPE_NUMBER) || types.has(common_chat_schema::TYPE_INTEGER)) {
                            json_value |= p.json_number();
                        }
                        if (types.has(common_chat_schema::TYPE_BOOLEAN)) {
                            json_value |= p.json_bool();
                        }
                        if (types.has(common_chat_schema::TYPE_NULL)) {
                            json_value |= p.json_null();
                        }
                        arg_value = p.gbnf(p.atomic(p.tool_arg_json_value(json_value) + arg_close) | arg_string, "xml-arg-string");
                    }

                    auto arg_rule = p.rule(rule_name, p.tool_arg(arg_open + arg_value));

                    (param.required ? required_args : optional_args).push_back(arg_rule);
                });

                // Accept required arguments in any order, as Qwen does not always adhere to the
                // order provided.
                auto args = p.permute("tool-" + name + "-args", required_args);
                if (!optional_args.empty()) {
                    args = args + p.zero_or_more(p.choice(optional_args));
                }

                auto opener = p.tool_open("<function=" + p.tool_name(p.literal(name)) + ">");
                auto body   = p.literal("\n") + p.tool_args(args) + p.tool_close(p.literal("</function>\n"));

                tool_choice |= p.rule("tool-" + name, p.tool(opener + body));
                functions.push_back({ name, opener, body });
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;

            auto tool_call_body = tool_choice + p.token("</tool_call>") + p.space();
            auto tool_call      = p.rule("tool-call", p.token("<tool_call>") + p.literal("\n") + tool_call_body);
            auto more           = inputs.parallel_tool_calls ? p.zero_or_more(tool_call) : p.eps();

            // Each trigger as the plain delimiter the grammar waits for, the tagged parser that consumes it, and
            // the rest that follows it
            struct trigger {
                common_peg_parser delimiter;
                common_peg_parser start;
                common_peg_parser rest;
            };
            auto tool_call_token = p.token("<tool_call>");
            std::vector<trigger> triggers = { { tool_call_token, tool_call_token, p.literal("\n") + tool_call_body + more } };

            if (is_qwen3_coder) {
                // Qwen3-Coder models may occasionally omit the <tool_call> token, so the complete <function=name>
                // opener is a trigger as well. The model may hallucinate a tool name, but it is preferable over
                // constraining on <function which may occur in valid content generation, e.g. #include <functional>
                for (const auto & f : functions) {
                    triggers.push_back({ p.literal("<function=" + f.name + ">"), f.opener,
                                         f.body + p.token("</tool_call>") + p.space() + more });
                }
            }

            std::vector<common_peg_parser>    tool_call_start;
            std::vector<common_peg_ac_branch> branches;
            auto                              tool_calls = p.choice();
            for (const auto & t : triggers) {
                tool_call_start.push_back(t.delimiter);
                branches.push_back({ t.delimiter, t.rest });
                tool_calls |= t.start + t.rest;
            }

            // The grammar lets content through until a trigger completes and then constrains the rest
            auto tool_section = p.trigger_rule("tool-section",
                p.ac(p.content(p.until(tool_call_start)) << p.repeat(tool_calls, min_calls, 1), branches, min_calls == 0));

            return generation_prompt + (reasoning << tool_section);
        }

        // Content only parser
        return generation_prompt + (reasoning << p.content(p.rest()));
    });

    data.parser = parser.save();

    if (include_grammar) {
        // The tool section grammar waits for the triggers itself, so the sampler runs it from the first token
        data.grammar_lazy = false;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, !has_response_format);
        });
    }

    return data;
}
