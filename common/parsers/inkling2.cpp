#include "parsers.h"

bool is_inkling2_template(const std::string & src) {
    return src.find("<|message_user|>") != std::string::npos &&
           src.find("<|content_image|>") != std::string::npos;
}

// the template makes one message block per content part, but a media marker comes inside the text of a message
// <header><|content_text|>{media}text<|end_message|> --> <header>{media}<|end_message|><header><|content_text|>text<|end_message|>
void common_chat_inkling2_split_media(common_chat_prompt & prompt) {
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
