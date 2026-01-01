// Xiaomi MiMo tool call format
// Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>

#include "chat-parsers-internal.h"
#include <optional>

common_chat_params common_chat_params_init_xiaomi_mimo_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);

    data.preserved_tokens = {
        "<tool_call>",
        "</tool_call>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto include_grammar = true;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        // Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        if (has_tools) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>"});
            }

            // Template format: <tool_call>\n{"name": ...}\n</tool_call>
            auto any_tool_call = p.choice();
            foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto &) {
                any_tool_call |= p.tag(Tag::TOOL, p.sequence()
                    + p.literal_tag(Tag::TOOL_OPEN, "{")
                    << "\"name\"" << ":" << ("\"" + p.literal_tag(Tag::TOOL_NAME, name) + "\"") << ","
                    << "\"arguments\"" << ":" << p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters))
                    << p.literal_tag(Tag::TOOL_CLOSE, "}"));
            });

            auto tool_calls = p.trigger_rule("tool-call-root",
                p.space()
                    + p.literal("<tool_call>\n")
                    + any_tool_call + p.repeat(p.literal("\n</tool_call>\n<tool_call>\n") << any_tool_call, 0, inputs.parallel_tool_calls ? -1 : 0)
                    + p.literal("\n</tool_call>"));

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return tool_calls;
            }

            // Content until <tool_call>, then consume optional newline before tools
            return p.tag(Tag::CONTENT, p.until_one_of({"<tool_call>", "\n<tool_call>"}))
                << p.optional(p.literal("\n")) << tool_calls;
        }

        // Content only parser - stop before end-of-message token
        include_grammar = false;
        return p.tag(Tag::CONTENT, p.until("<|im_end|>"));
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    return data;
}
