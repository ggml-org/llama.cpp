#include "parsers.h"

// Cohere2 MoE (a.k.a. "North Code") parser.
//
// The assistant turn is fully marker-wrapped:
//   <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
//     <|START_THINKING|>{reasoning}<|END_THINKING|>
//     then EITHER content:    <|START_TEXT|>{content}<|END_TEXT|>
//          OR     tool calls: <|START_ACTION|>[
//                                 {"tool_call_id": "0", "tool_name": "f", "parameters": {...}}, ...
//                             ]<|END_ACTION|>
//   <|END_OF_TURN_TOKEN|>
//
// The generation prompt forces a leading <|START_THINKING|> (when reasoning is enabled, which is
// the template default), so the model's output continues from *inside* the thinking block. The
// parser literal therefore only covers the stable <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|> prefix
// and the reasoning rule consumes the <|START_THINKING|> ... <|END_THINKING|> markers itself,
// regardless of whether they came from the generation prompt or the generated text.
common_chat_params common_chat_params_init_cohere2moe(const common_chat_template &          tmpl,
                                                              const autoparser::generation_params & inputs) {
    common_chat_params data;

    const std::string TURN_START    = "<|START_OF_TURN_TOKEN|>";
    const std::string TURN_END      = "<|END_OF_TURN_TOKEN|>";
    const std::string CHATBOT       = "<|CHATBOT_TOKEN|>";
    const std::string USER          = "<|USER_TOKEN|>";
    const std::string SYSTEM        = "<|SYSTEM_TOKEN|>";
    const std::string THINK_START   = "<|START_THINKING|>";
    const std::string THINK_END     = "<|END_THINKING|>";
    const std::string TEXT_START    = "<|START_TEXT|>";
    const std::string TEXT_END      = "<|END_TEXT|>";
    const std::string ACTION_START  = "<|START_ACTION|>";
    const std::string ACTION_END    = "<|END_ACTION|>";
    const std::string RESULT_START  = "<|START_TOOL_RESULT|>";
    const std::string RESULT_END    = "<|END_TOOL_RESULT|>";

    // Stable prefix of the generation prompt that precedes the (forced) <|START_THINKING|> marker.
    const std::string GEN_PREFIX = TURN_START + CHATBOT;

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = {THINK_END};
    data.preserved_tokens   = {
        TURN_START, TURN_END, CHATBOT, USER, SYSTEM,
        THINK_START, THINK_END,
        TEXT_START, TEXT_END,
        ACTION_START, ACTION_END,
        RESULT_START, RESULT_END,
    };

    // Declare per-role message delimiters. Tool results are rendered with the
    // system token followed by <|START_TOOL_RESULT|>, so the "tool" delimiter must be listed before
    // the plain "system" one (it is a strict superset, and the role split tries delimiters in order).
    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, GEN_PREFIX },
        { COMMON_CHAT_ROLE_USER,      TURN_START + USER },
        { COMMON_CHAT_ROLE_TOOL,      TURN_START + SYSTEM + RESULT_START },
        { COMMON_CHAT_ROLE_SYSTEM,    TURN_START + SYSTEM },
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PREFIX + THINK_START + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + TEXT_START + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PREFIX);
        auto end               = p.end();

        // The thinking block is always present (the generation prompt forces <|START_THINKING|>).
        // When extracting reasoning, capture its body; otherwise keep the whole block (markers
        // included) inline as content, matching reasoning_format=NONE conventions.
        common_peg_parser reasoning = p.eps();
        if (extract_reasoning) {
            reasoning = p.optional(p.literal(THINK_START) +
                                   p.reasoning(p.until_one_of({ THINK_END, TEXT_START, ACTION_START })) +
                                   p.optional(p.literal(THINK_END)));
        } else {
            reasoning = p.optional(p.content(p.literal(THINK_START) +
                                             p.until_one_of({ THINK_END, TEXT_START, ACTION_START }) +
                                             p.optional(p.literal(THINK_END))));
        }

        auto text_content = has_response_format
            ? p.literal(TEXT_START) +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                p.optional(p.literal(TEXT_END))
            : p.literal(TEXT_START) + p.content(p.until(TEXT_END)) + p.optional(p.literal(TEXT_END));

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + text_content + p.optional(p.literal(TURN_END)) + end;
        }

        auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        // <|START_ACTION|>[ {"tool_call_id": "0", "tool_name": "f", "parameters": {...}}, ... ]<|END_ACTION|>
        auto tool_calls = p.standard_json_tools(ACTION_START, ACTION_END, inputs.tools, inputs.parallel_tool_calls,
                                                /* force_tool_calls = */ true,
                                                /* name_key         = */ "tool_name",
                                                /* args_key         = */ "parameters",
                                                /* array_wrapped    = */ true,
                                                /* function_is_key  = */ false,
                                                /* call_id_key      = */ "",
                                                /* gen_call_id_key  = */ "tool_call_id",
                                                /* parameters_order = */ { "tool_call_id", "tool_name", "parameters" });

        // Content and tool calls are mutually exclusive in this format.
        common_peg_parser body = require_tools ? tool_calls : p.choice({ tool_calls, text_content });

        return generation_prompt + reasoning + body + p.optional(p.literal(TURN_END)) + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
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
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, ACTION_START }
        };
    }

    return data;
}
