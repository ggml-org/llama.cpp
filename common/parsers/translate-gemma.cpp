#include "parsers.h"
#include "log.h"

// TranslateGemma does not support tools or reasoning, it only needs user messages in its own content schema
common_chat_params common_chat_params_init_translate_gemma(
        const common_chat_template & tmpl,
        const autoparser::generation_params & inputs) {

    common_chat_params data;

    // default to chat_template_kwargs, or en-GB if not specified
    std::string src_lang = inputs.extra_context.value("source_lang_code", "en-GB");
    std::string tgt_lang = inputs.extra_context.value("target_lang_code", "en-GB");
    for (const char * key : { "source_lang_code", "target_lang_code" }) {
        if (!inputs.extra_context.contains(key)) {
            LOG_WRN("TranslateGemma: %s not set in chat_template_kwargs, defaulting to en-GB\n", key);
        }
    }

    json messages = inputs.messages;
    for (auto & message : messages) {
        if (message.value("role", "") != "user") {
            continue;
        }
        std::string text;
        const auto & content = message.contains("content") ? message.at("content") : json();
        if (content.is_string()) {
            text = content.get<std::string>();
        } else if (content.is_array()) {
            for (const auto & part : content) {
                if (!text.empty()) {
                    text += "\n";
                }
                text += part.value("text", "");
            }
        }
        message["content"] = json::array({
            json{
                {"type", "text"},
                {"text", text},
                {"source_lang_code", src_lang},
                {"target_lang_code", tgt_lang},
            }
        });
    }

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, messages);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs, messages);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = false;

    if (inputs.has_continuation()) {
        data.generation_prompt = "<start_of_turn>model\n" + inputs.continue_msg.render_content();
        data.prompt += data.generation_prompt;
    }

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        return p.literal(data.generation_prompt) << p.content(p.rest());
    });

    return data;
}
