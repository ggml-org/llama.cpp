#include "parsers.h"

bool is_inkling_template(const std::string & src) {
    return src.find("<|message_user|>") != std::string::npos &&
           src.find("<|content_image|>") != std::string::npos;
}

// the template makes one message block per content part, but a media marker comes inside the text of a message
// <header><|content_text|>{media}text<|end_message|> --> <header>{media}<|end_message|><header><|content_text|>text<|end_message|>
static void inkling_split_media(common_chat_prompt & prompt) {
    const std::string TEXT      = "<|content_text|>";
    const std::string END       = "<|end_message|>";
    const std::string MEDIA_BEG = "<__media_";
    const std::string MEDIA_END = "__>";

    std::vector<common_chat_prompt_part> out;
    for (size_t i = 0; i < prompt.parts.size(); i++) {
        const auto & part = prompt.parts[i];
        const bool in_text_block = part.is_input && part.text.find(MEDIA_BEG) != std::string::npos &&
            !out.empty() && !out.back().is_input && string_ends_with(out.back().text, TEXT) &&
            i + 1 < prompt.parts.size() && !prompt.parts[i + 1].is_input && string_starts_with(prompt.parts[i + 1].text, END);
        const size_t header_pos = in_text_block ? out.back().text.rfind("<|message_") : std::string::npos;
        if (header_pos == std::string::npos) {
            out.push_back(part);
            continue;
        }
        const std::string header = out.back().text.substr(header_pos, out.back().text.size() - TEXT.size() - header_pos);

        std::string rest = part.text;
        for (bool first = true; !rest.empty(); first = false) {
            const size_t beg      = rest.find(MEDIA_BEG);
            const size_t end      = beg == std::string::npos ? beg : rest.find(MEDIA_END, beg);
            const bool   is_media = beg == 0 && end != std::string::npos;
            const size_t len      = is_media ? end + MEDIA_END.size() : (end == std::string::npos ? rest.size() : beg);
            if (!first) {
                out.push_back({ END + header + (is_media ? "" : TEXT), false });
            } else if (is_media) {
                // mtmd opens the media block
                out.back().text.resize(out.back().text.size() - TEXT.size());
            }
            out.push_back({ rest.substr(0, len), true });
            rest.erase(0, len);
        }
    }
    prompt.parts = std::move(out);
}

// Inkling 1 and 2 - the assistant turn is a list of blocks, all closed by <|end_message|>:
//   <|message_model|><|content_thinking|>...<|end_message|>
//   <|message_model|><|content_text|>...<|end_message|>
//   <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
common_chat_params common_chat_params_init_inkling(const common_chat_template &          tmpl,
                                                   const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    inkling_split_media(data.prompt);

    const std::string HEADER    = "<|message_model|>";
    const std::string THINK     = "<|content_thinking|>";
    const std::string TEXT      = "<|content_text|>";
    const std::string TOOL_JSON = "<|content_invoke_tool_json|>";
    const std::string END       = "<|end_message|>";

    data.preserved_tokens = { HEADER, THINK, TEXT, TOOL_JSON, END };

    data.thinking_start_tag = THINK;
    data.thinking_end_tags  = { END };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, HEADER               },
        { COMMON_CHAT_ROLE_USER,      "<|message_user|>"   },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|message_system|>" },
        { COMMON_CHAT_ROLE_TOOL,      "<|message_tool|>"   },
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt.clear();
        if (!msg.reasoning_content.empty()) {
            data.generation_prompt = HEADER + THINK + msg.reasoning_content;
        }
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            if (!data.generation_prompt.empty()) {
                data.generation_prompt += END;
            }
            data.generation_prompt += HEADER + TEXT + msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto use_tools           = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    data.parser = build_chat_peg_parser(inputs.specials, [&](common_chat_peg_builder & p) {
        auto header = p.token(HEADER);
        auto end    = p.token(END);

        auto think    = p.token(THINK);
        auto thinking = p.rule("thinking", extract_reasoning ? think + p.reasoning(p.until(end)) + end :
                                                               p.atomic(p.content(think)) + p.content(p.until(end) + end));
        auto text     = p.rule("text", p.token(TEXT) + p.content(p.until(end)) + end);

        if (has_response_format) {
            auto response_format = p.rule("response-format", p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));
            return p.zero_or_more(header + thinking) + header + p.token(TEXT) + response_format + end;
        }

        auto blocks = p.zero_or_more(header + (thinking | text));

        if (!use_tools) {
            return blocks;
        }

        // the call takes the name in the JSON
        auto tool_json = p.token(TOOL_JSON);
        auto tool_body = [&](const common_peg_parser & json_name) {
            return p.tool_open(p.literal("{")) << p.json_member("name", p.literal("\"") + p.tool_name(json_name) + p.literal("\"")) << p.literal(",")
                                               << p.json_member("args", p.tool_args(p.json())) << p.tool_close(p.literal("}")) + end;
        };

        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](size_t tool_index, const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                std::string  quoted   = json(name).dump();

                // The research model does not generate a tool name before the tool call token
                auto json_name = p.literal(quoted.substr(1, quoted.size() - 2));
                tool_choice |= p.rule("tool-" + std::to_string(tool_index), p.tool(p.optional(p.literal(name)) + tool_json + tool_body(json_name)));
            });

            auto calls = inputs.parallel_tool_calls ? tool_choice + p.zero_or_more(header + tool_choice) : tool_choice;
            return blocks + header + calls;
        }

        auto name  = p.until(tool_json);
        auto body  = tool_body(p.string_content('"'));
        auto next  = p.rule("tool-call", header + p.tool(name + tool_json + body));
        auto more  = inputs.parallel_tool_calls ? p.zero_or_more(next) : p.eps();
        auto calls = header + name + p.trigger_rule("tool-calls", tool_json, body + more);

        return blocks + p.optional(calls);
    });

    if (has_response_format || use_tools) {
        data.grammar_lazy = false;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO);
        });
    }

    return data;
}
