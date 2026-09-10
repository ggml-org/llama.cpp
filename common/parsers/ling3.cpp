#include "parsers.h"

// Ling 3.0 / Bailing V3 - <role>X</role> sections with tagged tool calls:
//   assistant := [<think> ... </think>] [content] {<tool_call>name
//                  <arg_key>k</arg_key>\n<arg_value>v</arg_value> ...</tool_call>}
// The generation prompt ends with "<role>ASSISTANT</role>\n<think>", so the model
// never emits the opening think tag, and a tool call can arrive before any
// </think>. Reasoning therefore terminates at the think close tag or at a tool
// call start, like the Qwen3-Coder and Kimi K3 parsers. With thinking off the
// template pre-closes the think block instead, and the model emits bare content.
common_chat_params common_chat_params_init_ling3(const common_chat_template &          tmpl,
                                                        const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    const std::string ROLE        = "<role>ASSISTANT</role>";
    const std::string THINK_START = "<think>";
    const std::string THINK_END   = "</think>";
    const std::string CALL_START  = "<tool_call>";
    const std::string CALL_END    = "</tool_call>";
    const std::string ARG_KEY     = "<arg_key>";
    const std::string ARG_KEY_END = "</arg_key>";
    const std::string ARG_VAL     = "<arg_value>";
    const std::string ROLE_END    = "<|role_end|>";
    const std::string ARG_VAL_END = "</arg_value>";

    data.preserved_tokens = {
        THINK_START, THINK_END, CALL_START, CALL_END,
        ARG_KEY, ARG_KEY_END, ARG_VAL, ARG_VAL_END, "<|role_end|>",
    };

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = { THINK_END };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<role>ASSISTANT</role>" },
        { COMMON_CHAT_ROLE_USER,      "<role>HUMAN</role>"     },
        { COMMON_CHAT_ROLE_TOOL,      "<role>OBSERVATION</role>" },
        { COMMON_CHAT_ROLE_SYSTEM,    "<role>SYSTEM</role>"    },
    };

    // the model may spell the end-of-turn control token out as text tokens,
    // which does not stop generation; a literal stop string catches it either
    // way (as the Laguna patch does for its </assistant> token)
    data.additional_stops = { ROLE_END };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = ROLE + "\n" + THINK_START + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    // The think block is open at the end of the generation prompt when thinking
    // is on (or a reasoning continuation is pending); with thinking off the
    // template pre-closes it and everything the model emits is content.
    auto last_open  = data.generation_prompt.rfind(THINK_START);
    auto last_close = data.generation_prompt.rfind(THINK_END);
    bool think_open = last_open != std::string::npos &&
                      (last_close == std::string::npos || last_open > last_close);

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto end = p.end();

        // the effective parse input is generation_prompt + model output, so the
        // assistant opener is optionally consumed here
        auto opener = p.optional(p.literal(ROLE) + p.optional(p.space()));

        // the generation prompt pre-opens the think block, so the opening tag is
        // optional; reasoning stops at the close tag or at a tool call start, and
        // a missing close tag does not swallow the tool call. With thinking off
        // the pre-closed pair is consumed and nothing is classified as reasoning.
        // with the block open, a tool call ends the reasoning even without a
        // close tag; once it is closed (thinking off, or a content continuation),
        // only the close tag ends it and everything after is content
        auto body_end   = think_open ? p.until_one_of({ THINK_END, CALL_START }) : p.until_one_of({ THINK_END });
        auto think_body = extract_reasoning ? p.reasoning(body_end) : p.content(body_end);

        auto reasoning = p.optional(p.optional(p.literal(THINK_START)) + think_body +
                                    p.optional(p.literal(THINK_END)));

        // content between the think block and the first tool call, plus any
        // trailing text after the last tool call, are plain content
        auto content = p.optional(p.content(p.until_one_of({ CALL_START })));

        // a trailing end-of-turn token is consumed instead of leaking into content
        auto tail = p.optional(p.content(p.until(ROLE_END))) + p.optional(p.literal(ROLE_END));

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return opener + reasoning + tail + end;
        }

        auto tool_choices = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            auto         parameters = function.contains("parameters") ? function.at("parameters") : json::object();

            auto schema_info = common_schema_info();
            schema_info.resolve_refs(parameters);

            std::vector<common_peg_parser> required_args;
            std::vector<common_peg_parser> optional_args;

            // each argument may be preceded by whitespace: the model emits
            // newlines between arguments, the template history does not
            foreach_parameter(function, [&](const std::string & key, const json & param_schema, bool is_required) {
                auto rule_name = "ling3-arg-" + name + "-" + key;

                auto arg = p.rule(rule_name,
                    p.optional(p.space()) +
                    p.tool_arg(p.tool_arg_open(p.literal(ARG_KEY) + p.tool_arg_name(p.literal(key)) +
                                               p.literal(ARG_KEY_END)) +
                               p.optional(p.space()) + p.literal(ARG_VAL) +
                               (schema_info.resolves_to_string(param_schema) ?
                                   p.tool_arg_string_value(p.until(ARG_VAL_END)) :
                                   p.tool_arg_json_value(p.schema(
                                       p.json(), rule_name + "-schema", param_schema, false))) +
                               p.tool_arg_close(p.literal(ARG_VAL_END))));

                (is_required ? required_args : optional_args).push_back(arg);
            });

            // required arguments in any order (as Qwen3-Coder does), then
            // optional ones in any order and number
            auto args = p.permute("ling3-" + name + "-args", required_args);
            if (!optional_args.empty()) {
                args = args + p.zero_or_more(p.choice(optional_args));
            }

            auto call = p.tool(p.tool_open(p.literal(CALL_START) + p.tool_name(p.literal(name)) +
                                           p.optional(p.space())) +
                               p.tool_args(args) +
                               p.tool_close(p.optional(p.space()) + p.literal(CALL_END)));

            tool_choices |= p.rule("ling3-tool-" + name, call);
        });

        auto calls = inputs.parallel_tool_calls ?
                     tool_choices + p.zero_or_more(p.space() + tool_choices) :
                     tool_choices;

        auto tools_section = p.trigger_rule("ling3-tool-call", calls + p.space() +
                                        p.optional(p.content(p.until(ROLE_END))) + p.optional(p.literal(ROLE_END)));

        auto tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? tools_section :
                                                                             p.optional(tools_section);

        return opener + reasoning + content + tools + tail + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                if (function.contains("parameters")) {
                    auto schema = function.at("parameters");
                    builder.resolve_refs(schema);
                }
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, CALL_START },
        };
    }

    return data;
}
