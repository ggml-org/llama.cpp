// Functionary v3.1 (Llama 3.1 style) tool call format
// Format: <function=name>{...}</function>
// Also supports: <|python_tag|>code...

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_functionary_v3_1_llama_3_1_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    auto has_raw_python = false;
    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1;

    // Detect python tool (for <|python_tag|> support)
    if (has_tools) {
        foreach_function(inputs.tools, [&](const auto &, const auto & name, const json &, const auto &) {
            if (name == "python" || name == "ipython") {
                has_raw_python = true;
            }
        });
    }

    // Set up preserved tokens
    data.preserved_tokens = {};
    if (has_raw_python) {
        data.preserved_tokens.push_back("<|python_tag|>");
    }

    // Build PEG parser for <function=name>{...}</function> format
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<function="});
            }

            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](const auto &, const auto & name, const auto & parameters, const auto &) {
                // Format: <function=name>{...}</function>
                tool_choice |= p.rule("tool-" + name, p.tag(Tag::TOOL,
                    p.atomic_tag(Tag::TOOL_OPEN, p.literal("<function="))
                    + p.literal_tag(Tag::TOOL_NAME, name)
                    + ">"
                    + p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-params", parameters))
                    + p.atomic_tag(Tag::TOOL_CLOSE, p.literal("</function>"))
                ));
            });

            // Add python tag support if present
            if (has_raw_python) {
                // <|python_tag|>code... (raw python code wrapped in arguments)
                tool_choice |= p.rule("python-raw", p.tag(Tag::TOOL,
                    p.atomic_tag(Tag::TOOL_OPEN, p.literal("<|python_tag|>") + p.literal_tag(Tag::TOOL_NAME, "python"))
                    + p.tag(Tag::TOOL_ARGS, p.rest())
                ));
            }

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;

            std::vector<std::string> delimiters = {"<function="};
            if (has_raw_python) {
                delimiters.push_back("<|python_tag|>");
            }

            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_choice, min_calls, max_calls));
            bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            if (require_tools) {
                return tool_calls;
            }
            return p.tag(Tag::CONTENT, p.until_one_of(delimiters)) << tool_calls;
        }

        // Content only parser
        // Stop tokens for Functionary v3.1
        return p.tag(Tag::CONTENT, p.until_one_of({"<|eot_id|>", "<|eom_id|>", "<|end|>", "<|start_header_id|>"}));
    });

    common_chat_build_peg_grammar(inputs, parser, data);

    return data;
}
