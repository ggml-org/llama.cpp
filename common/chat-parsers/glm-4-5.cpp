// GLM 4.5 tool call format
// Format: <tool_call>function_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>
// With optional <think>...</think> reasoning blocks

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_glm_4_5_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    std::string prompt = apply(tmpl, inputs);

    // match the existing trimming behavior
    if (inputs.add_bos && string_starts_with(prompt, tmpl.bos_token())) {
        prompt.erase(0, tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(prompt, tmpl.eos_token())) {
        prompt.erase(prompt.size() - tmpl.eos_token().size());
    }
    if (string_ends_with(prompt, "<think>")) {
        if (!inputs.enable_thinking) {
            prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    data.prompt = prompt;

    // add GLM preserved tokens
    data.preserved_tokens = {
        "<|endoftext|>",
        "[MASK]",
        "[gMASK]",
        "[sMASK]",
        "<sop>",
        "<eop>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
        "<|begin_of_audio|>",
        "<|end_of_audio|>",
        "<|begin_of_transcription|>",
        "<|end_of_transcription|>",
        "<|code_prefix|>",
        "<|code_middle|>",
        "<|code_suffix|>",
        "/nothink",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>"
    };

    // extra GLM 4.5 stop word
    data.additional_stops.insert(data.additional_stops.end(), {
        "<|user|>",
        "<|observation|>"
    });

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // =============================================================
        // root ::= thinking? (tools | content)
        // content ::= json_schema | rest
        // =============================================================

        // THINKING - optional reasoning block at the start
        auto thinking = [&]() {
            if (!extract_reasoning) {
                return p.eps();
            }
            if (data.thinking_forced_open) {
                // Prompt ends with <think>, expect content until </think>
                return p.optional(p.literal("\n"))
                     + p.tag(Tag::REASONING, p.until("</think>"))
                     + ("</think>" | p.end());
            }
            // Optional <think>...</think> block
            return p.optional(
                p.optional(p.literal("\n"))
                + "<think>" + p.tag(Tag::REASONING, p.until("</think>")) + "</think>"
            );
        }();

        // CONTENT - either json_schema or rest (both allow optional leading newline)
        auto content = [&]() {
            if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
                return p.optional(p.literal("\n")) + p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
            }
            return p.optional(p.literal("\n")) + p.tag(Tag::CONTENT, p.rest());
        }();

        // TOOLS
        if (has_tools) {
            if (inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>"});
            }

            generic_tool_call_format format;
            format.tool_call_start = p.space() + "<tool_call>";
            format.tool_call_name_params_sep = p.space();
            format.tool_call_end = p.space() + "</tool_call>";
            format.param_start = p.space() + "<arg_key>";
            format.param_name_value_sep = "</arg_key>" + p.space() + "<arg_value>";
            format.param_ends = { "</arg_value>\n", "</arg_value>" };
            format.allow_raw_string_param_value = true;
            auto tool_calls = build_generic_tool_calls_peg_parser(p, inputs, format);
            
            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                // thinking? space? tools
                return thinking + p.space() + tool_calls;
            }

            // Either: thinking? content_before? space? tools content_after?
            // Or:     thinking? content (when no tool calls present)
            auto content_before = p.optional(
                p.optional(p.literal("\n"))
                + p.tag(Tag::CONTENT, p.until_one_of({"\n<tool_call>", "<tool_call>"}))
            );
            auto content_after = p.optional(p.tag(Tag::CONTENT, p.rest()));
            auto with_tools = content_before + p.space() + tool_calls + content_after;
            auto content_only = p.optional(p.literal("\n")) + p.tag(Tag::CONTENT, p.rest());
            return thinking + p.choice({with_tools, content_only});
        }

        // No tools: thinking? content
        include_grammar = false;
        return thinking + content;
    });

    common_chat_build_peg_grammar(inputs, parser, data);
    data.format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;

    return data;
}
