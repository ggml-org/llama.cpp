#include "parsers.h"

// K2 Horizon - reasoning effort picks one of three think tag pairs, tool calls are tagged:
//   assistant := <ifm|think[_fast|_faster]> ... </ifm|think[_fast|_faster]> [content]
//                [<ifm|tool_calls> {<ifm|tool_call>CALL</ifm|tool_call>} </ifm|tool_calls>]
//   CALL (tool_call_format=xml, default) := name {<ifm|arg_key>k</ifm|arg_key> [<ifm|arg_type>t</ifm|arg_type>]
//                                                 <ifm|arg_value>v</ifm|arg_value>}
//   CALL (tool_call_format=json)          := {"name": name, "arguments": {...}}
// The generation prompt pre-opens the think block, so the model never emits the
// opening tag. Reasoning ends at the close tag or at a tool call section start.
common_chat_params common_chat_params_init_k2_horizon(const common_chat_template &          tmpl,
                                                      const autoparser::generation_params & inputs) {
    common_chat_params data;

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

    const std::string ROLE          = "<|ifm|im_start|>assistant";
    const std::string TURN_END      = "<|ifm|im_end|>";
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

    // reasoning_effort high/medium/low opens <ifm|think>/<ifm|think_fast>/<ifm|think_faster>;
    // the pair in use is the last one the generation prompt opened
    std::string think = "ifm|think";
    size_t      think_pos = std::string::npos;
    for (const std::string tag : { "ifm|think", "ifm|think_fast", "ifm|think_faster" }) {
        auto pos = data.generation_prompt.rfind("<" + tag + ">");
        if (pos != std::string::npos && (think_pos == std::string::npos || pos > think_pos)) {
            think     = tag;
            think_pos = pos;
        }
    }
    const std::string THINK_START = "<" + think + ">";
    const std::string THINK_END   = "</" + think + ">";

    data.preserved_tokens = {
        THINK_START, THINK_END, SECTION_START, SECTION_END, CALL_START, CALL_END,
        ARG_KEY, ARG_KEY_END, ARG_TYPE, ARG_TYPE_END, ARG_VAL, ARG_VAL_END, TURN_END,
    };

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = { THINK_END, SECTION_START };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|ifm|im_start|>assistant" },
        { COMMON_CHAT_ROLE_USER,      "<|ifm|im_start|>user"      },
        { COMMON_CHAT_ROLE_TOOL,      "<|ifm|im_start|>tool"      },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|ifm|im_start|>system"    },
    };

    // the turn ends with <|ifm|im_end|>, but only <|endoftext|> is EOG in the vocab
    data.additional_stops = { TURN_END };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = ROLE + "\n" + THINK_START + "\n" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    bool think_open = false;
    if (inputs.has_continuation()) {
        think_open = inputs.continue_final_message != COMMON_CHAT_CONTINUATION_CONTENT;
    } else {
        think_open = think_pos != std::string::npos && data.generation_prompt.find(THINK_END, think_pos) == std::string::npos;
    }

    std::string call_format = "xml";
    if (inputs.extra_context.contains("tool_call_format") && inputs.extra_context.at("tool_call_format").is_string()) {
        call_format = inputs.extra_context.at("tool_call_format");
    }

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto end = p.end();

        // the effective parse input is generation_prompt + model output
        auto opener = p.optional(p.literal(ROLE) + p.optional(p.space()));

        auto body_end   = think_open ? p.until_one_of({ THINK_END, SECTION_START }) : p.until(THINK_END);
        auto think_body = extract_reasoning ? p.reasoning(body_end) : p.content(body_end);
        // the template writes "<tag>\n" and "</tag>\n"; those newlines are markup, not text
        auto nl         = p.optional(p.literal("\n"));
        auto reasoning  = p.optional(p.optional(p.literal(THINK_START) + nl) + think_body + p.optional(p.literal(THINK_END) + nl));

        auto content = p.optional(p.content(p.until_one_of({ SECTION_START, TURN_END })));
        auto tail    = p.optional(p.content(p.until(TURN_END))) + p.optional(p.literal(TURN_END));

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return opener + reasoning + tail + end;
        }

        auto tool_choices = p.choice();
        auto arg_close    = p.tool_arg_close(p.literal(ARG_VAL_END));
        auto arg_string   = p.rule("k2h-arg-string", p.tool_arg_string_value(p.until(ARG_VAL_END)) + arg_close);
        auto arg_type     = p.optional(p.optional(p.space()) + p.literal(ARG_TYPE) + p.until(ARG_TYPE_END) + p.literal(ARG_TYPE_END));

        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");

            if (call_format == "json") {
                auto schema = common_chat_tool_parameters(function);
                auto call   = p.tool(p.tool_open(p.literal(CALL_START) + p.literal("{\"name\": \"") + p.tool_name(p.literal(name)) +
                                                 p.literal("\", \"arguments\": ")) +
                                     p.tool_args(p.schema(p.json(), "k2h-tool-" + name + "-schema", schema)) +
                                     p.tool_close(p.literal("}") + p.literal(CALL_END)));
                tool_choices |= p.rule("k2h-tool-" + name, call);
                return;
            }

            // xml / xml_typed: strings are raw text up to the closing tag, other types are JSON
            std::vector<common_peg_parser> required_args;
            std::vector<common_peg_parser> optional_args;
            foreach_parameter(function, [&](const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                auto rule_name = "k2h-arg-" + name + "-" + param.name;
                auto types     = param.schema->value_types();
                auto json_val  = p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", doc, *param.schema)) + arg_close;

                auto arg_value = types.is_only(common_chat_schema::TYPE_STRING) ? arg_string :
                                 !types.has(common_chat_schema::TYPE_STRING)    ? json_val :
                                 p.gbnf(p.atomic(json_val) | arg_string, "k2h-arg-string");

                auto arg = p.rule(rule_name,
                    p.optional(p.space()) +
                    p.tool_arg(p.tool_arg_open(p.literal(ARG_KEY) + p.tool_arg_name(p.literal(param.name)) + p.literal(ARG_KEY_END)) +
                               arg_type + p.optional(p.space()) + p.literal(ARG_VAL) + arg_value));

                (param.required ? required_args : optional_args).push_back(arg);
            });

            auto args = p.permute("k2h-" + name + "-args", required_args);
            if (!optional_args.empty()) {
                args = args + p.zero_or_more(p.choice(optional_args));
            }

            auto call = p.tool(p.tool_open(p.literal(CALL_START) + p.tool_name(p.literal(name)) + p.optional(p.space())) +
                               p.tool_args(args) +
                               p.tool_close(p.optional(p.space()) + p.literal(CALL_END)));
            tool_choices |= p.rule("k2h-tool-" + name, call);
        });

        auto calls = inputs.parallel_tool_calls ? tool_choices + p.zero_or_more(p.space() + tool_choices) : tool_choices;

        auto tools_section = p.trigger_rule("k2h-tool-call",
            p.literal(SECTION_START) + p.space() + calls + p.space() + p.literal(SECTION_END));

        // a required call follows the reasoning directly, as for gemma4 and gpt-oss
        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
            return opener + reasoning + p.optional(p.space()) + tools_section + tail + end;
        }

        return opener + reasoning + content + p.optional(tools_section) + tail + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, SECTION_START },
        };
    }

    return data;
}
