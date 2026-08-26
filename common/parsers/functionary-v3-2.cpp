#include "parsers.h"

// Functionary v3.2 - uses recipient-based format: >>>recipient\n{content}
common_chat_params common_chat_params_init_functionary_v3_2(const common_chat_template &    tmpl,
                                                                   const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens  = {
        ">>>all",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;
        data.generation_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n>>>all\n" + msg.render_content();
        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // Functionary v3.2 format:
        // - Normal content: >>>all\n{content}
        // - Tool calls: >>>function_name\n{json_args}
        // Generation prompt ends with ">>>" so model outputs recipient immediately

        // Build content parser for >>>all\n{content}
        // When tools are present, content stops before the next ">>>" (tool call)
        // When no tools, content goes until end
        auto content_until_tool = p.literal("all\n") + p.content(p.until(">>>"));
        auto content_until_end  = p.literal("all\n") + p.content(p.rest());
        auto generation_prompt  = p.literal("<|start_header_id|>assistant<|end_header_id|>\n\n>>>");

        // If no tools or tool_choice is NONE, just parse content
        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            // When no tools, just match the prefix and capture everything after
            return generation_prompt + content_until_end + p.end();
        }

        // Build tool call parsers for each available function
        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto & schema   = function.at("parameters");

            // Tool format: >>>function_name\n{json_args}
            auto tool_parser = p.tool(
                p.tool_open(p.tool_name(p.literal(name)) + p.literal("\n")) +
                p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema))
            );

            tool_choice |= p.rule("tool-" + name, tool_parser);
        });

        auto content_only = content_until_end;
        auto tools_only = p.trigger_rule("tools", p.one_or_more(tool_choice));
        auto content_and_tools = content_until_tool + tools_only;

        auto ret = p.eps();
        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
            if (inputs.parallel_tool_calls) {
                ret = p.choice({ content_and_tools, tools_only }) + p.end();
            } else {
                ret = p.choice({ content_until_tool + tool_choice, tools_only }) + p.end();
            }
        } else if (inputs.parallel_tool_calls) {
            ret = p.choice({ content_and_tools, content_only, tools_only }) + p.end();
        } else {
            auto content_and_tool = content_until_tool + tool_choice;
            ret = p.choice({ content_and_tool, content_only, tool_choice }) + p.end();
        }
        return generation_prompt + ret;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Grammar trigger for when the model starts outputting a tool call
        // (after the initial ">>>" in the generation prompt but recipient other than "all")
        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, ">>>(?!all)" }
        };
    }

    return data;
}
