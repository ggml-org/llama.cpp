#include "parsers.h"

// LFM2 format detection: template uses <|tool_list_start|>[...]<|tool_list_end|> around the tool list
// and <|tool_call_start|>[...]<|tool_call_end|> around each tool call
bool is_lfm2_template(const std::string & src) {
    return src.find("<|tool_list_start|>") != std::string::npos &&
           src.find("<|tool_list_end|>")   != std::string::npos;
}

// LFM2/LFM2.5 parser. Tool calls are almost Python-style and parallel-capable
// (except dotted names and JSON literals true/false/null).
// Always wrapped in <|tool_call_start|>[name(args)]<|tool_call_end|> with optional <think> reasoning.
// tool_list_tokens preserves LFM2 system tool-list markers.
common_chat_params common_chat_params_init_lfm2(const common_chat_template &          tmpl,
                                                       const autoparser::generation_params & inputs,
                                                       bool tool_list_tokens) {
    common_chat_params data;

    const std::string TOOL_CALL_START = "<|tool_call_start|>";
    const std::string TOOL_CALL_END   = "<|tool_call_end|>";
    const std::string TOOL_LIST_START = "<|tool_list_start|>";
    const std::string TOOL_LIST_END   = "<|tool_list_end|>";
    const std::string THINK_START     = "<think>";
    const std::string THINK_END       = "</think>";
    const std::string GEN_PROMPT      = "<|im_start|>assistant\n";

    // Copy reasoning to the "thinking" field the template expects
    auto adjusted_messages = json::array();
    for (auto msg : inputs.messages) {
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            msg["thinking"] = msg.at("reasoning_content");
        }
        adjusted_messages.push_back(msg);
    }

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, adjusted_messages);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs, adjusted_messages);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.preserved_tokens  = { TOOL_CALL_START, TOOL_CALL_END, THINK_START, THINK_END };
    if (tool_list_tokens) {
        data.preserved_tokens.push_back(TOOL_LIST_START);
        data.preserved_tokens.push_back(TOOL_LIST_END);
    }

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = {THINK_END};

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    // Gate by reasoning format and whether the template supports <think>
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE &&
                             tmpl.source().find(THINK_START) != std::string::npos;
    auto include_grammar   = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PROMPT + THINK_START + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PROMPT);
        auto end = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning) {
            reasoning = p.optional(THINK_START + p.reasoning(p.until(THINK_END)) + THINK_END);
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (has_response_format) {
                auto response_format = p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema));
                return generation_prompt + reasoning + response_format + end;
            }
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }
        auto tool_calls = p.rule("tool-calls",
            p.trigger_rule("tool-call",
                p.literal(TOOL_CALL_START) +
                p.python_style_tool_calls(inputs.tools, inputs.parallel_tool_calls, /* allow_json_literals = */ true) +
                p.literal(TOOL_CALL_END)
            )
        );

        auto content = p.content(p.until(TOOL_CALL_START));

        return generation_prompt + reasoning + content + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, TOOL_CALL_START }
        };
    }

    return data;
}
