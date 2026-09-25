#include "parsers.h"

// Inkling / TML typed-content-block parser: <|end_message|> separates blocks within a turn,
// <|content_model_end_sampling|> is the sole end-of-generation token (mirrors sglang TmlDetector).
common_chat_params common_chat_params_init_inkling(const common_chat_template &          tmpl,
                                                          const autoparser::generation_params & inputs) {
    common_chat_params data;

    const std::string MSG_MODEL    = "<|message_model|>";
    const std::string MSG_USER     = "<|message_user|>";
    const std::string MSG_SYSTEM   = "<|message_system|>";
    const std::string MSG_TOOL     = "<|message_tool|>";
    const std::string THINK        = "<|content_thinking|>";
    const std::string TEXT         = "<|content_text|>";
    const std::string END_MESSAGE  = "<|end_message|>";
    const std::string END_SAMPLING = "<|content_model_end_sampling|>";
    const std::string INVOKE_TOOL  = "<|content_invoke_tool_json|>";

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = THINK;
    data.thinking_end_tags  = {END_MESSAGE};
    data.preserved_tokens   = {
        MSG_MODEL, MSG_USER, MSG_SYSTEM, MSG_TOOL,
        THINK, TEXT, END_MESSAGE, END_SAMPLING, INVOKE_TOOL,
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, MSG_MODEL },
        { COMMON_CHAT_ROLE_USER,      MSG_USER },
        { COMMON_CHAT_ROLE_SYSTEM,    MSG_SYSTEM },
        { COMMON_CHAT_ROLE_TOOL,      MSG_TOOL },
    };

    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = MSG_MODEL + THINK + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += END_MESSAGE + TEXT + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(MSG_MODEL);
        auto end               = p.end();

        // thinking block; may also reappear mid-turn (after content), so it is both an optional
        // prefix and a choice inside the block loops. With reasoning_format=NONE keep it
        // (markers included) inline as content
        common_peg_parser reasoning_block = p.eps();
        if (extract_reasoning) {
            reasoning_block = p.literal(THINK) +
                              p.reasoning(p.until_one_of({ END_MESSAGE, TEXT, END_SAMPLING })) +
                              p.optional(p.literal(END_MESSAGE));
        } else {
            reasoning_block = p.content(p.literal(THINK) +
                                        p.until_one_of({ END_MESSAGE, TEXT, END_SAMPLING }) +
                                        p.optional(p.literal(END_MESSAGE)));
        }
        auto reasoning = p.optional(reasoning_block);

        // TML re-emits <|message_model|> before each content block; a turn may contain several
        // text blocks (one per content part), so the block repeats and bodies concatenate.
        // THINK stops the content scan so a mid-turn thinking block is never leaked as text
        auto text_block = p.optional(p.literal(MSG_MODEL)) +
                          p.optional(p.literal(TEXT)) +
                          p.content(p.until_one_of({ THINK, END_MESSAGE, END_SAMPLING })) +
                          p.optional(p.literal(END_MESSAGE));
        auto text_content = p.one_or_more(p.choice({ reasoning_block, text_block }));

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + text_content +
                   p.optional(p.literal(END_SAMPLING)) + end;
        }

        // each call is its own block (role opener + bare name echo + JSON section);
        // force_tool_calls=true makes the JSON section required so a pure-text answer fails the
        // block cleanly; parallel calls are separate blocks, hence repeat + parallel=false
        auto tool_section = p.standard_json_tools(
            INVOKE_TOOL, END_MESSAGE, inputs.tools, /* parallel_tool_calls = */ false,
            /* force_tool_calls = */ true,
            /* name_key          = */ "name",
            /* args_key          = */ "args",
            /* array_wrapped     = */ false,
            /* function_is_key   = */ false,
            /* call_id_key       = */ "",
            /* gen_call_id_key   = */ "",
            /* parameters_order  = */ {},
            /* accept_openai_wrapper = */ false,
            /* require_object_args   = */ true);
        // the name-echo scan must stop at any block marker: a greedy until(INVOKE_TOOL) returns
        // NEED_MORE_INPUT mid-stream, which choice() treats as a match and shadows the text branch
        auto tool_block = p.optional(p.literal(MSG_MODEL)) +
                          p.until_one_of({ INVOKE_TOOL, TEXT, THINK, END_MESSAGE, END_SAMPLING }) +
                          tool_section;
        auto tool_calls = inputs.parallel_tool_calls ? p.one_or_more(tool_block) : tool_block;
        // turns may interleave narration, thinking and calls; parse block-by-block (tool block
        // first) since a whole-body choice would let the text branch swallow tool blocks into
        // visible content
        auto mixed_body = p.one_or_more(p.choice({ tool_block, reasoning_block, text_block }));
        auto body       = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED
                              ? tool_calls
                              : mixed_body;

        return generation_prompt + reasoning + body +
               p.optional(p.literal(END_SAMPLING)) + end;
    });

    data.parser = parser.save();

    return data;
}
