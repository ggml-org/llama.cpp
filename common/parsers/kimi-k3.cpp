#include "parsers.h"

// Kimi K3 - XTML tagged format, built by open_tag/close_tag macros:
//   open_tag(t, attrs) = <|open|>t k="v"...<|sep|>   close_tag(t) = <|close|>t<|sep|>
//   assistant := [think] [response] [tools] close_tag(message) <|end_of_msg|>
// the generation prompt already opens the think (or response) section, so the
// section opener is optional here - same as Kimi K2 Thinking
common_chat_params common_chat_params_init_kimi_k3(const common_chat_template &          tmpl,
                                                          const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    const std::string SEP         = "<|sep|>";
    const std::string MSG_START   = "<|open|>message role=\"assistant\"<|sep|>";
    const std::string THINK_START = "<|open|>think<|sep|>";
    const std::string THINK_END   = "<|close|>think<|sep|>";
    const std::string RESP_START  = "<|open|>response<|sep|>";
    const std::string RESP_END    = "<|close|>response<|sep|>";
    const std::string TOOLS_START = "<|open|>tools<|sep|>";
    const std::string TOOLS_END   = "<|close|>tools<|sep|>";
    const std::string CALL_START  = "<|open|>call tool=\"";
    const std::string CALL_END    = "<|close|>call<|sep|>";
    const std::string ARG_START   = "<|open|>argument key=\"";
    const std::string ARG_END     = "<|close|>argument<|sep|>";
    const std::string MSG_END     = "<|close|>message<|sep|>";
    const std::string EOM_TOKEN   = "<|end_of_msg|>";

    // only the markers are special tokens. tag names ("think", "response", ...) are
    // normal tokens and must not be preserved, or prose with those words is broken
    data.preserved_tokens = {
        "<|open|>",
        "<|close|>",
        "<|sep|>",
        "<|end_of_msg|>",
    };

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = { THINK_END };

    // per-role message-start delimiters. user/assistant messages only have the role
    // attribute, so the full opener is used. system and tool messages have more
    // attributes, so those delimiters stop after the closing quote of the role
    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|open|>message role=\"assistant\"<|sep|>" },
        { COMMON_CHAT_ROLE_USER,      "<|open|>message role=\"user\"<|sep|>"      },
        { COMMON_CHAT_ROLE_TOOL,      "<|open|>message role=\"tool\""             },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|open|>message role=\"system\""           },
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = MSG_START + THINK_START + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + RESP_START + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto end = p.end();

        auto start = p.optional(p.literal(MSG_START));

        // the think section is always consumed, even with reasoning extraction off:
        // the generation prompt ends with open_tag('think'), so it is always present.
        // reasoning stops at its own closer, or at the response opener if the model
        // skips the closer
        auto think_body = extract_reasoning ? p.reasoning(p.until_one_of({ THINK_END, RESP_START })) :
                                              p.content(p.until_one_of({ THINK_END, RESP_START }));

        auto reasoning = p.optional(p.optional(p.literal(THINK_START)) + think_body +
                                    p.optional(p.literal(THINK_END)));

        // content runs to the response closer, or to the next section if truncated
        auto response = p.optional(p.literal(RESP_START)) +
                        p.content(p.until_one_of({ RESP_END, TOOLS_START, MSG_END })) +
                        p.optional(p.literal(RESP_END));

        // the EOG token after the message closer reaches the parser as text,
        // so it must be consumed or the parse stays incomplete
        auto trailer = p.optional(p.literal(MSG_END)) + p.optional(p.literal(EOM_TOKEN));

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return start + reasoning + response + trailer + end;
        }

        auto tool_choices = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const json   schema   = function.contains("parameters") ? function.at("parameters") : json::object();

            // arguments come one tag per key, with the JSON type in a type="..."
            // attribute. the type is taken from the tool schema instead, as it tells
            // us if the value is JSON or a literal string
            auto args = p.eps();
            if (schema.contains("properties") && !schema.at("properties").empty()) {
                auto arg_choices = p.choice();
                for (const auto & prop : schema.at("properties").items()) {
                    const std::string & key = prop.key();

                    std::string type = "string";
                    if (prop.value().is_object() && prop.value().contains("type") &&
                        prop.value().at("type").is_string()) {
                        type = prop.value().at("type").get<std::string>();
                    }

                    auto value = type == "string" ? p.tool_arg_string_value(p.until(ARG_END)) :
                                                    p.tool_arg_value(p.until(ARG_END));

                    // skip the trailing type="..." attribute: anything up to <|sep|>
                    arg_choices |= p.rule("kimi-k3-arg-" + name + "-" + key,
                                          p.tool_arg(p.tool_arg_open(p.literal(ARG_START)) +
                                                     p.tool_arg_name(p.literal(key)) + p.literal("\"") +
                                                     p.until(SEP) + p.literal(SEP) + value +
                                                     p.tool_arg_close(p.literal(ARG_END))));
                }
                args = p.zero_or_more(arg_choices);
            }

            // skip the trailing index="N" attribute the same way
            auto call = p.tool(p.tool_open(p.literal(CALL_START) + p.tool_name(p.literal(name)) + p.literal("\"") +
                                           p.until(SEP) + p.literal(SEP)) +
                               p.tool_args(args) + p.tool_close(p.literal(CALL_END)));

            tool_choices |= p.rule("kimi-k3-tool-" + name, call);
        });

        // all calls go inside one tools section, then the message is closed. the
        // message closer is part of the trigger rule, or else the lazy grammar
        // rejects it once tool calls have started
        auto tools_section =
            p.trigger_rule("kimi-k3-tool-call", p.literal(TOOLS_START) + p.one_or_more(tool_choices) +
                                                    p.literal(TOOLS_END) + p.optional(p.literal(MSG_END)) +
                                                    p.optional(p.literal(EOM_TOKEN)));

        auto tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? tools_section :
                                                                              p.optional(tools_section);

        return start + reasoning + response + tools + trailer + end;
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
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, TOOLS_START },
        };
    }

    return data;
}
