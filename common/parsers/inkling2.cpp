#include "parsers.h"

// each image goes in its own user block
static void inkling2_split_media_blocks(common_chat_prompt & prompt) {
    const std::string TEXT        = "<|content_text|>";
    const std::string END_MESSAGE = "<|end_message|>";
    const std::string MEDIA_OPEN  = "<__media_";
    const std::string MEDIA_CLOSE = "__>";

    std::vector<common_chat_prompt_part> out;
    for (size_t i = 0; i < prompt.parts.size(); i++) {
        const auto & part = prompt.parts[i];
        const bool in_text_block = part.is_input && !out.empty() && !out.back().is_input &&
                                   string_ends_with(out.back().text, TEXT) && i + 1 < prompt.parts.size() &&
                                   !prompt.parts[i + 1].is_input && string_starts_with(prompt.parts[i + 1].text, END_MESSAGE);
        const size_t header_pos = in_text_block ? out.back().text.rfind("<|message_") : std::string::npos;
        if (header_pos == std::string::npos || part.text.find(MEDIA_OPEN) == std::string::npos) {
            out.push_back(part);
            continue;
        }
        auto & prev = out.back().text;
        const std::string header = prev.substr(header_pos, prev.size() - TEXT.size() - header_pos);
        prev.resize(prev.size() - TEXT.size());

        bool first = true;
        auto add_block = [&](const std::string & body, bool is_media) {
            const std::string open = (first ? "" : END_MESSAGE + header) + (is_media ? "" : TEXT);
            if (!open.empty()) {
                out.push_back({ open, false });
            }
            out.push_back({ body, true });
            first = false;
        };
        std::string rest = part.text;
        while (!rest.empty()) {
            const size_t open  = rest.find(MEDIA_OPEN);
            const size_t close = open == std::string::npos ? std::string::npos : rest.find(MEDIA_CLOSE, open);
            if (close == std::string::npos) {
                add_block(rest, false);
                break;
            }
            if (open > 0) {
                add_block(rest.substr(0, open), false);
            }
            add_block(rest.substr(open, close + MEDIA_CLOSE.size() - open), true);
            rest = rest.substr(close + MEDIA_CLOSE.size());
        }
    }
    prompt.parts = std::move(out);
}

// <|message_model|><|content_thinking|>...<|end_message|>
// <|message_model|><|content_text|>...<|end_message|>
// <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
common_chat_params common_chat_params_init_inkling2(const common_chat_template &          tmpl,
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
    inkling2_split_media_blocks(data.prompt);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = THINK;
    data.thinking_end_tags  = { END_MESSAGE };
    data.preserved_tokens   = {
        MSG_MODEL, MSG_USER, MSG_SYSTEM, MSG_TOOL, THINK, TEXT, END_MESSAGE, END_SAMPLING, INVOKE_TOOL,
    };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, MSG_MODEL  },
        { COMMON_CHAT_ROLE_USER,      MSG_USER   },
        { COMMON_CHAT_ROLE_SYSTEM,    MSG_SYSTEM },
        { COMMON_CHAT_ROLE_TOOL,      MSG_TOOL   },
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        const bool continue_content = inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT;
        if (continue_content && msg.reasoning_content.empty()) {
            data.generation_prompt = MSG_MODEL + TEXT + msg.render_content();
        } else {
            data.generation_prompt = MSG_MODEL + THINK + msg.reasoning_content;
            if (continue_content) {
                data.generation_prompt += END_MESSAGE + MSG_MODEL + TEXT + msg.render_content();
            }
        }

        data.prompt.append(data.generation_prompt);
    }

    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty() &&
                             inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.optional(p.literal(MSG_MODEL));
        auto end   = p.end();

        common_peg_parser reasoning_block = p.eps();
        if (extract_reasoning) {
            reasoning_block = p.literal(THINK) +
                              p.reasoning(p.until_one_of({ END_MESSAGE, TEXT, END_SAMPLING })) +
                              p.optional(p.literal(END_MESSAGE));
        } else {
            // a partial "<|content_t" may become <|content_text|>
            reasoning_block = p.peek(p.literal(THINK)) +
                              p.content(p.literal(THINK) +
                                        p.until_one_of({ END_MESSAGE, TEXT, END_SAMPLING }) +
                                        p.optional(p.literal(END_MESSAGE)));
        }

        auto text_block = p.optional(p.literal(TEXT)) +
                          p.content(p.until_one_of({ MSG_MODEL, THINK, END_MESSAGE, END_SAMPLING })) +
                          p.optional(p.literal(END_MESSAGE));
        auto block_start = p.optional(p.literal(MSG_MODEL));

        if (!has_tools) {
            auto blocks = p.one_or_more(block_start + p.choice({ reasoning_block, text_block }));
            return start + blocks + p.optional(p.literal(END_SAMPLING)) + end;
        }

        // longest first: "get" must not shadow "get_weather"
        std::vector<std::string> names;
        foreach_function(inputs.tools, [&](size_t, const json & tool) {
            names.push_back(tool.at("function").at("name").get<std::string>());
        });
        std::stable_sort(names.begin(), names.end(), [](const std::string & a, const std::string & b) { return a.size() > b.size(); });
        auto tool_name = p.choice();
        for (const auto & name : names) {
            tool_name |= p.literal(name);
        }
        auto call = p.standard_json_tools(INVOKE_TOOL, END_MESSAGE, inputs.tools, /* parallel_tool_calls = */ false,
                                          /* force_tool_calls = */ true,
                                          /* name_key         = */ "name",
                                          /* args_key         = */ "args");
        auto calls = inputs.parallel_tool_calls
            ? p.trigger_rule("tool-calls", call + p.zero_or_more(p.literal(MSG_MODEL) + tool_name + call))
            : call;
        auto tool_block = tool_name + calls;

        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
            return start + p.optional(reasoning_block) + block_start + tool_block +
                   p.optional(p.literal(END_SAMPLING)) + end;
        }
        auto blocks = p.one_or_more(block_start + p.choice({ tool_block, reasoning_block, text_block }));
        return start + blocks + p.optional(p.literal(END_SAMPLING)) + end;
    });

    if (has_tools) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });
        if (data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, INVOKE_TOOL },
            };
        }
    }

    return data;
}
