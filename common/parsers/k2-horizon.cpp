#include "parsers.h"

// K2 Horizon format:
// - Reasoning: <ifm|think>...</ifm|think>, or <ifm|think_fast>/<ifm|think_faster> for medium/low reasoning_effort
// - Tool calls: <ifm|tool_calls><ifm|tool_call>...</ifm|tool_call>...</ifm|tool_calls>, one call per <ifm|tool_call>:
//   xml (default): name <ifm|arg_key>k</ifm|arg_key> [<ifm|arg_type>t</ifm|arg_type>] <ifm|arg_value>v</ifm|arg_value> ...
//   json:          {"name": "...", "arguments": {...}}
common_chat_params common_chat_params_init_k2_horizon(const common_chat_template &          tmpl,
                                                      const autoparser::generation_params & inputs) {
    common_chat_params data;

    // The template requires a thinking field on every assistant message
    auto messages = inputs.messages;
    for (auto & msg : messages) {
        if (msg.value("role", "") == "assistant" && !msg.contains("reasoning_content")) {
            msg["reasoning_content"] = "";
        }
    }

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, messages);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs, messages);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    const std::string effort      = inputs.extra_context.value("reasoning_effort", "high");
    const std::string call_format = inputs.extra_context.value("tool_call_format", "xml");

    // Templates that handle enable_thinking disable it with an empty <ifm|think></ifm|think> block for every effort
    const bool thinking_off = !inputs.enable_thinking && tmpl.source().find("enable_thinking") != std::string::npos;
    const std::string think = thinking_off       ? "ifm|think"        :
                              effort == "medium" ? "ifm|think_fast"   :
                              effort == "low"    ? "ifm|think_faster" : "ifm|think";

    const std::string GEN_PREFIX    = "<|ifm|im_start|>assistant\n";
    const std::string THINK_START   = "<" + think + ">";
    const std::string THINK_END     = "</" + think + ">";
    const std::string SECTION_START = "<ifm|tool_calls>";
    const std::string SECTION_END   = "</ifm|tool_calls>";
    const std::string CALL_START    = "<ifm|tool_call>";
    const std::string CALL_END      = "</ifm|tool_call>";
    const std::string ARG_KEY       = "<ifm|arg_key>";
    const std::string ARG_KEY_END   = "</ifm|arg_key>";
    const std::string ARG_TYPE      = "<ifm|arg_type>";
    const std::string ARG_TYPE_END  = "</ifm|arg_type>";
    const std::string ARG_VAL       = "<ifm|arg_value>";
    const std::string ARG_VAL_END   = "</ifm|arg_value>";

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = { THINK_END };

    data.preserved_tokens = data.thinking_end_tags;
    data.preserved_tokens.insert(data.preserved_tokens.end(), {
        THINK_START, SECTION_START, SECTION_END, CALL_START, CALL_END,
        ARG_KEY, ARG_KEY_END, ARG_TYPE, ARG_TYPE_END, ARG_VAL, ARG_VAL_END,
    });

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|ifm|im_start|>assistant" },
        { COMMON_CHAT_ROLE_USER,      "<|ifm|im_start|>user"      },
        { COMMON_CHAT_ROLE_TOOL,      "<|ifm|im_start|>tool"      },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|ifm|im_start|>system"    },
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PREFIX + THINK_START + "\n" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PREFIX);

        auto think_end = p.choice();
        for (const auto & tag : data.thinking_end_tags) {
            think_end |= p.literal(tag);
        }
        auto think_body  = p.until_one_of(data.thinking_end_tags);
        auto think_block = [&](const common_peg_parser & body) {
            return p.optional(THINK_START + p.space() + p.ac(body + think_end, data.thinking_end_tags));
        };
        auto reasoning = extract_reasoning ? think_block(p.reasoning(think_body)) : p.eps();

        if (has_response_format) {
            // The answer must be bare JSON, so the think block is consumed even when it is not extracted
            auto thoughts = extract_reasoning ? reasoning : think_block(think_body);
            return generation_prompt + (thoughts << p.content(p.schema(p.json(), "response-format", inputs.json_schema)));
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + (reasoning << p.content(p.rest()));
        }

        auto tool_choice = p.choice();
        if (call_format == "json") {
            tool_choice = p.standard_json_tools(CALL_START, CALL_END, inputs.tools, false, true);
        } else {
            auto arg_close  = p.tool_arg_close(p.literal(ARG_VAL_END));
            auto arg_string = p.rule("xml-arg-string", p.ac(p.tool_arg_string_value(p.until(ARG_VAL_END)) + arg_close, ARG_VAL_END));

            // The models leave out <ifm|arg_type> even when asked for xml_typed
            auto arg_type = call_format == "xml_typed" ? p.optional(ARG_TYPE + p.until(ARG_TYPE_END) + ARG_TYPE_END + p.space()) : p.eps();

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");

                std::vector<common_peg_parser> required_args;
                std::vector<common_peg_parser> optional_args;
                foreach_parameter(function, [&](const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                    auto rule_name = "tool-" + name + "-arg-" + param.name;
                    auto types     = param.schema->value_types();
                    auto arg_value = arg_string;
                    if (!types.has(common_chat_schema::TYPE_STRING)) {
                        arg_value = p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", doc, *param.schema)) + arg_close;
                    }
                    if (types.has(common_chat_schema::TYPE_STRING) && !types.is_only(common_chat_schema::TYPE_STRING)) {
                        // The string alternative accepts any text, so only the parser needs the JSON alternatives.
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

                    auto arg = p.space() + p.tool_arg(p.tool_arg_open(ARG_KEY + p.tool_arg_name(p.literal(param.name)) + ARG_KEY_END) <<
                                                      arg_type + ARG_VAL + arg_value);
                    (param.required ? required_args : optional_args).push_back(p.rule(rule_name, arg));
                });

                auto args = p.permute("tool-" + name + "-args", required_args);
                if (!optional_args.empty()) {
                    args = args + p.zero_or_more(p.choice(optional_args));
                }

                tool_choice |= p.rule("tool-" + name, p.tool(
                    p.tool_open(CALL_START + p.tool_name(p.literal(name)) + "\n") + p.tool_args(args) << p.tool_close(p.literal(CALL_END))));
            });
        }

        auto required   = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        auto calls      = inputs.parallel_tool_calls ? tool_choice + p.zero_or_more(p.space() + tool_choice) : tool_choice;
        auto tool_calls = p.trigger_rule("tool-calls", p.repeat(SECTION_START << calls << SECTION_END, required ? 1 : 0, 1));

        // Keep thinking inline when required calls bypass the content parser.
        if (required && !extract_reasoning) {
            reasoning = p.content(think_block(think_body));
        }

        // A required call follows the reasoning directly, the models otherwise keep writing content
        auto content = required ? p.eps() : p.content(p.until(SECTION_START));

        return generation_prompt + (reasoning << content << tool_calls);
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED);
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        if (data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, SECTION_START },
            };
        }
    }

    return data;
}
