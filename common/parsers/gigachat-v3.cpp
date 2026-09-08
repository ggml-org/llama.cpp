#include "parsers.h"

common_chat_params common_chat_params_init_gigachat_v3(
        const common_chat_template & tmpl,
        const autoparser::generation_params & inputs) {

    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = false;
    data.preserved_tokens  = {
        "<|message_sep|>\n\n",
        "<|role_sep|>\n",
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;
        data.generation_prompt = "assistant<|role_sep|>\n" + msg.render_content();
        data.prompt += data.generation_prompt;
    }

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    const auto *tool_call_start_prefix = "<|message_sep|>\n\nfunction call<|role_sep|>\n";

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto ret = p.eps();
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // Build a choice of all available tools
            auto tool_choice = p.choice();
            for (const auto & tool : inputs.tools) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                const auto & schema = function.at("parameters");

                auto tool_name = p.json_member("name", "\"" + p.tool_name(p.literal(name)) + "\"");
                auto tool_args = p.json_member("arguments", p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)));

                auto tool_open = p.tool_open(p.literal("{") << tool_name);

                tool_choice |= p.rule("tool-" + name, tool_open << "," << tool_args << "}");
            }

            // Define the tool call structure
            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = 1; // parallel toolcalls are not supported
            auto tool_call = p.rule("tool-call", p.literal(tool_call_start_prefix) + tool_choice);
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, /* min = */ min_calls, /* max = */ max_calls));

            ret = p.content(p.until("<|message_sep|>\n\n")) << tool_calls;
        } else {
            // Content only parser
            include_grammar = false;
            ret = p.content(p.rest());
        }

        return p.literal("assistant<|role_sep|>\n") + ret;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto schema = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, tool_call_start_prefix}
        };
    }
    return data;
}
