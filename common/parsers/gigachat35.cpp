#include "parsers.h"

static autoparser::generation_params common_chat_gigachat35_render_inputs(
        const autoparser::generation_params & inputs) {
    auto render_inputs = inputs;
    if (!render_inputs.extra_context.is_object()) {
        render_inputs.extra_context = json::object();
    }
    if (!render_inputs.extra_context.contains("reasoning")) {
        render_inputs.extra_context["reasoning"] = inputs.enable_thinking;
    }
    return render_inputs;
}

common_chat_params common_chat_params_init_gigachat35(
        const common_chat_template & tmpl,
        const autoparser::generation_params & inputs) {

    auto render_inputs = common_chat_gigachat35_render_inputs(inputs);

    common_chat_params data;

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, render_inputs);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, render_inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<think>";
    data.thinking_end_tags  = {"</think>"};
    data.preserved_tokens   = {
        "<|message_sep|>\n\n",
        "<|role_sep|>\n",
        "｜GCML｜",
        "<think>",
        "</think>",
    };

    const std::string ROLE_SEP     = "<|role_sep|>\n";
    const std::string GEN_PROMPT   = "assistant" + ROLE_SEP;
    const std::string GCML         = "｜GCML｜";
    const std::string THINK_START  = "<think>";
    const std::string THINK_END    = "</think>";
    const std::string FC_START     = "<" + GCML + "tool_calls>";

    // the reasoning template unconditionally opens a think block in the assistant turn
    const bool template_opens_think =
        data.generation_prompt.size() >= THINK_START.size() &&
        data.generation_prompt.compare(data.generation_prompt.size() - THINK_START.size(),
                                       THINK_START.size(), THINK_START) == 0;

    if (render_inputs.has_continuation()) {
        const auto & msg = render_inputs.continue_msg;

        data.generation_prompt = GEN_PROMPT;
        if (render_inputs.continue_final_message == COMMON_CHAT_CONTINUATION_REASONING) {
            data.generation_prompt += THINK_START + msg.reasoning_content;
        } else {
            // preserve the template's think block before continuation content
            if (!msg.reasoning_content.empty()) {
                data.generation_prompt += THINK_START + msg.reasoning_content + THINK_END + "\n\n";
            } else if (template_opens_think) {
                data.generation_prompt += THINK_START + THINK_END + "\n\n";
            }
            data.generation_prompt += msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    auto has_tools           = render_inputs.tools.is_array() && !render_inputs.tools.empty();
    auto has_response_format = render_inputs.json_schema.is_object();
    auto extract_reasoning   = render_inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && render_inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // start before <think> so the template's reasoning opener and continuation prefill are parsed
        auto generation_prompt = p.literal(GEN_PROMPT);
        auto end               = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning || has_response_format) {
            auto text = p.until(THINK_END);
            if (extract_reasoning) {
                text = p.reasoning(text);
            }
            reasoning = p.optional(
                THINK_START +
                text +
                THINK_END +
                p.optional(p.literal("\n\n")));
        }

        if (has_response_format) {
            return generation_prompt + reasoning + p.content(p.schema(p.json(), "response-format", render_inputs.json_schema)) + end;
        }

        if (!has_tools || render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto require_tools = render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        auto tool_calls = build_chat_peg_tool_calls(
            p, render_inputs.tools, GCML, "tool_calls", render_inputs.parallel_tool_calls, true);

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        auto content_before_tools = (extract_reasoning ? p.negate(p.literal(THINK_START)) : p.eps()) + p.content(p.until(FC_START));
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        if (!has_response_format) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, FC_START },
            };
        }
    }

    return data;
}
