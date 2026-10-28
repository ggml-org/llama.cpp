#include "parsers.h"

// Harmony with XML tool calls: <|start|>assistant to=NAME<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="KEY">\nVALUE\n</rfl:parameter>
common_chat_params common_chat_params_init_harmony_xml(const common_chat_template &          tmpl,
                                                       const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|start|>assistant" },
        { COMMON_CHAT_ROLE_USER,      "<|start|>user"      },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|start|>developer" },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|start|>system"    },
    };

    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    data.thinking_start_tag = "<|channel|>analysis<|message|>";
    data.thinking_end_tags  = {"<|end|>"};

    // EOG text must reach the parser: an XML call has no closing tag of its own
    static const std::vector<std::string> EOG = {
        "<|call|>", "<|return|>", "<|endoftext|>", "<|im_end|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
    };

    data.preserved_tokens = {
        "<|channel|>", "<|constrain|>", "<|message|>", "<|start|>", "<|end|>",
    };
    data.preserved_tokens.insert(data.preserved_tokens.end(), EOG.begin(), EOG.end());

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = "<|start|>assistant<|channel|>analysis<|message|>" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += "<|end|><|start|>assistant<|channel|>final<|message|>" + msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    static const std::string PARAM_OPEN  = "<rfl:parameter name=\"";
    static const std::string PARAM_CLOSE = "\n</rfl:parameter>";

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.rule("start", p.literal("<|start|>assistant"));

        auto eog = p.choice();
        for (const auto & tok : EOG) {
            eog |= p.literal(tok);
        }
        auto with_eog = [&](std::vector<std::string> delims) {
            delims.insert(delims.end(), EOG.begin(), EOG.end());
            return delims;
        };

        auto analysis_open  = p.literal("<|channel|>analysis<|message|>");
        auto analysis_body  = p.until_one_of(with_eog({ "<|end|>", "<|constrain|><|start|>assistant", "<|start|>assistant" }));
        auto analysis_close = p.optional(p.literal("<|end|>") | p.literal("<|constrain|>") | eog);
        if (extract_reasoning) {
            p.rule("analysis", analysis_open + p.reasoning(analysis_body) + analysis_close);
        } else {
            p.rule("analysis", p.atomic(p.content(analysis_open)) + p.content(analysis_body + analysis_close));
        }
        auto analysis = p.ref("analysis");

        auto final_msg = p.rule("final",
            p.literal("<|channel|>final<|message|>") +
            p.content(p.until_one_of(with_eog({ "<|end|>" }))) +
            p.optional(p.literal("<|end|>") | eog));

        if (has_response_format) {
            auto constrain_type = p.chars("[A-Za-z0-9_-]", 1, -1);
            auto constraint     = p.optional(p.space() + p.optional(p.literal("<|constrain|>")) + constrain_type);
            auto response_format = p.rule("response-format",
                p.literal("<|channel|>final") + constraint + p.literal("<|message|>") +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                p.optional(p.literal("<|end|>") | eog));

            return start + p.zero_or_more(analysis + start) + response_format;
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // tolerate a missing closing tag
            std::vector<std::string> value_ends = with_eog({ PARAM_OPEN, "<|end|>" });
            std::vector<std::string> string_ends = { PARAM_CLOSE };
            auto next_open = p.choice();
            for (const auto & e : value_ends) {
                string_ends.push_back("\n" + e);
                next_open |= p.literal(e);
            }
            auto arg_close = p.tool_arg_close(p.gbnf(
                p.literal(PARAM_CLOSE) | (p.literal("\n") + p.peek(p.optional(p.literal("\n")) + next_open)),
                gbnf_format_literal("\n") + " " + gbnf_format_literal(PARAM_CLOSE.substr(1)) + "?"));
            auto arg_string = p.rule("rfl-arg-string",
                p.ac(p.tool_arg_string_value(p.until_one_of(string_ends)) + arg_close, PARAM_CLOSE));
            auto arg_sep = p.optional(p.literal("\n"));

            std::string call_close_gbnf = gbnf_format_literal("<|end|>");
            for (const auto & tok : EOG) {
                call_close_gbnf += " | " + gbnf_format_literal(tok);
            }
            auto call_close = p.gbnf(p.literal("<|end|>") | eog | p.peek(p.literal("<|start|>assistant")),
                "(" + call_close_gbnf + ")?");

            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](size_t tool_index, const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");

                std::vector<common_peg_parser> arg_rules;
                foreach_parameter(function, [&](size_t param_index, const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                    auto rule_name = "tool-" + std::to_string(tool_index) + "-arg-" + std::to_string(param_index);
                    auto arg_open  = p.tool_arg_open(p.literal(PARAM_OPEN) + p.tool_arg_name(p.literal(param.name)) + p.literal("\">\n"));

                    auto types     = param.schema->value_types();
                    auto arg_value = p.eps();
                    if (!types.has(common_chat_schema::TYPE_STRING) &&
                        (types.has(common_chat_schema::TYPE_OBJECT) || types.has(common_chat_schema::TYPE_ARRAY))) {
                        // the schema grammar fixes the key order, and the model writes keys in its own order
                        arg_value = p.tool_arg_json_value(p.json()) + arg_close;
                    } else if (!types.has(common_chat_schema::TYPE_STRING)) {
                        arg_value = p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", doc, *param.schema)) + arg_close;
                    } else if (types.is_only(common_chat_schema::TYPE_STRING)) {
                        arg_value = arg_string;
                    } else {
                        // grammar keeps only the string rule; the parser tries JSON first to type the value
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
                        arg_value = p.gbnf(p.atomic(p.tool_arg_json_value(json_value) + arg_close) | arg_string, "rfl-arg-string");
                    }

                    arg_rules.push_back(p.rule(rule_name, p.tool_arg(arg_open + arg_value)));
                });

                if (arg_rules.empty()) {
                    auto any_name = p.chars("[^\"\\n<]", 1, -1);
                    arg_rules.push_back(p.rule("tool-" + std::to_string(tool_index) + "-arg-any",
                        p.tool_arg(p.tool_arg_open(p.literal(PARAM_OPEN) + p.tool_arg_name(any_name) + p.literal("\">\n")) + arg_string)));
                }

                auto args = p.zero_or_more(p.choice(arg_rules) + arg_sep);

                auto call = p.tool(
                    p.tool_open(p.literal(" to=") + p.tool_name(p.literal(name)) +
                                p.literal("<|channel|>analysis <|constrain|>xml<|message|>")) +
                    p.tool_args(args) +
                    p.tool_close(call_close));

                tool_choice |= p.rule("tool-" + std::to_string(tool_index), call);
            });

            auto tool_calls = inputs.parallel_tool_calls
                ? tool_choice + p.zero_or_more(start + tool_choice)
                : tool_choice;
            auto tool_call = p.trigger_rule("tool-call", tool_calls);

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return start + p.zero_or_more(analysis + start) + tool_call;
            }

            return start + p.zero_or_more(analysis + start) + (tool_call | final_msg | p.content(p.rest()));
        }

        auto unsolicited = p.rule("unsolicited", p.atomic(p.literal(" to=") + p.rest()));

        // output without a channel header is returned as content instead of failing the request
        return start + p.zero_or_more(analysis + start) + (final_msg | unsolicited | p.content(p.rest()));
    });

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        if (data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "^\\s+to$" },
                { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|start\\|>assistant(\\s+to)" },
            };
        }
    }

    return data;
}
