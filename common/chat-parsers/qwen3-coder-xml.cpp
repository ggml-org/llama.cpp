// Qwen3 Coder XML tool call format
// Format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>

#include "chat-parsers-internal.h"
#include <optional>

common_chat_params common_chat_params_init_qwen3_coder_xml_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);

    data.additional_stops = {
        "<|im_end|>",
        "<|endoftext|>",
    };

    data.preserved_tokens = {
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto include_grammar = true;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // Match optional content before <tool_call>, but don't tag whitespace-only content
        const auto content_before_tool = p.optional(
            p.space()  // Consume leading whitespace without tagging
            + p.optional(p.rule("qwen-tool-prefix",
                p.tag(Tag::CONTENT, p.until("<tool_call>"))
                + p.peek(p.literal("<tool_call>"))
            ))
        );

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>"});
            }

            generic_tool_call_format format;
            format.tool_call_start = p.space() + "<tool_call>\n<function=";
            format.tool_call_name_params_sep = ">" + p.space();
            format.tool_call_end = "</function>" + p.space() + "</tool_call>";
            format.param_start = p.literal("<parameter=");
            format.param_name_value_sep = ">" + p.space();
            format.param_ends = { "\n</parameter>\n", "</parameter>\n", "</parameter>" };
            format.allow_raw_string_param_value = true;
            auto tool_calls = build_generic_tool_calls_peg_parser(p, inputs, format);

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return tool_calls;
            }
            return p.optional(content_before_tool) + tool_calls;
        }

        // Content only parser
        include_grammar = false;
        return p.tag(Tag::CONTENT, p.rest());
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;

    return data;
}
