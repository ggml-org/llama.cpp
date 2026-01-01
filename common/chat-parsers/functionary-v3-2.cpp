// Functionary v3.2 tool call format
// Format: >>>all\ntext>>>fn1\n{...}>>>fn2\n{...}...
// ALL tool calls use >>> prefix (template generates >>> for every call)
// Python tool can have raw code (without opening {)

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_functionary_v3_2_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.prompt = apply(tmpl, inputs);
    data.preserved_tokens = {
        "<|end_header_id|>",
    };
    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    // Build PEG parser for >>>function_name\n{...} format
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // Response format parser
        // Note: template outputs "all\n" prefix even for json_schema responses
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            auto json_content = p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
            auto with_all = "all\n" + json_content;
            return with_all | json_content;
        }

        // Tool call parser: first tool call has no >>> prefix (it's in the generation prompt),
        // subsequent calls have >>> prefix
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // First tool call: no >>> prefix (since >>> is in generation prompt)
            auto first_tool_call = p.choice();
            // Subsequent tool calls: with >>> prefix
            auto subsequent_tool_call = p.choice();

            foreach_function(inputs.tools, [&](const auto &, const auto & name, const auto & parameters, const auto &) {
                std::string args_pattern;
                if (name == "python") {
                    // Python can have raw code or JSON
                    auto python_args = p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-params", parameters))
                                     | p.tag(Tag::TOOL_ARGS, p.until(">>>"));
                    // First tool needs empty TOOL_OPEN to create tool call object (>>> is in generation prompt)
                    first_tool_call |= p.rule("tool-first-" + name, p.tag(Tag::TOOL,
                        p.literal_tag(Tag::TOOL_OPEN, "") + p.literal_tag(Tag::TOOL_NAME, name) + "\n" + python_args
                    ));
                    subsequent_tool_call |= p.rule("tool-" + name, p.tag(Tag::TOOL,
                        p.literal_tag(Tag::TOOL_OPEN, ">>>") + p.literal_tag(Tag::TOOL_NAME, name) + "\n" + python_args
                    ));
                    args_pattern = "[\\s\\S]*";
                } else {
                    // Regular JSON tool
                    auto tool_args = p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-params", parameters));
                    // First tool needs empty TOOL_OPEN to create tool call object (>>> is in generation prompt)
                    first_tool_call |= p.rule("tool-first-" + name, p.tag(Tag::TOOL,
                        p.literal_tag(Tag::TOOL_OPEN, "") + p.literal_tag(Tag::TOOL_NAME, name) + "\n" + tool_args
                    ));
                    subsequent_tool_call |= p.rule("tool-" + name, p.tag(Tag::TOOL,
                        p.literal_tag(Tag::TOOL_OPEN, ">>>") + p.literal_tag(Tag::TOOL_NAME, name) + "\n" + tool_args
                    ));
                    args_pattern = "\\{" + args_pattern;
                }
                if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                    data.grammar_triggers.push_back({
                        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                        "^(>>>)?" + regex_escape(name) + "\n" + args_pattern,
                    });
                }
            });
            data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<function="});

            // Build pattern: optional content with "all\n" marker, then tool calls
            // Format with content: all\n<content>>>>name\n{...}>>>name2\n{...}
            // Format without content: name\n{...}>>>name2\n{...}
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;

            // Content marker: "all\n" followed by text until >>>
            auto content_marker = "all\n" + p.tag(Tag::CONTENT, p.until(">>>"));

            // Subsequent tool calls (with >>> prefix)
            auto more_tool_calls = p.repeat(subsequent_tool_call, 0, max_calls > 0 ? max_calls - 1 : -1);

            // Optional trailing content, stop at end tokens
            auto trailing_content = p.optional(p.tag(Tag::CONTENT, p.until_one_of({"<|eot_id|>", "<|start_header_id|>"})));

            // Pattern 1: content marker + tool calls (all with >>> since content ends at >>>)
            auto with_content = p.trigger_rule("tool-with-content", content_marker)
                << p.repeat(subsequent_tool_call, 1, max_calls) << trailing_content;
            // Pattern 2: first tool (no >>>) + subsequent tools (with >>>)
            auto without_content = p.trigger_rule("tool-without-content", first_tool_call)
                << more_tool_calls << trailing_content;

            bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            if (require_tools) {
                // In REQUIRED mode, only return tool calls without content
                return p.trigger_rule("tool-required", first_tool_call) << more_tool_calls;
            }
            return with_content | without_content;
        }

        // Content only parser
        // Handle optional "all\n" content marker used by Functionary v3.2
        auto content_with_all = "all\n" + p.tag(Tag::CONTENT, p.until_one_of({"<|eot_id|>", "<|start_header_id|>"}));
        auto content_without_all = p.tag(Tag::CONTENT, p.until_one_of({"<|eot_id|>", "<|start_header_id|>"}));
        return content_with_all | content_without_all;
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

    return data;
}
