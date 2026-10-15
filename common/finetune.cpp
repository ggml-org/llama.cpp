#include "finetune.h"

#include "chat.h"
#include "common.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// the assistant turns are located in the rendered text through the turn markers of the chat format
struct sft_turn_markers {
    const char * start;
    const char * end;
};

static const sft_turn_markers sft_turn_markers_all[] = {
    { "<|im_start|>assistant\n", "<|im_end|>"    }, // ChatML
    { "<start_of_turn>model\n",  "<end_of_turn>" }, // Gemma 3
    { "<|turn>model\n",          "<turn|>"       }, // Gemma 4
    { "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>" }, // Llama 3
};

ggml_opt_dataset_t common_opt_sft_dataset_init(
        struct llama_context * ctx,
        const std::string    & json_content,
        const std::string    & chat_template) {
    using json = nlohmann::json;

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    common_chat_templates_ptr chat_templates;
    try {
        chat_templates = common_chat_templates_init(llama_get_model(ctx), chat_template);
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to initialize the chat template: %s\n", __func__, e.what());
        return nullptr;
    }
    LOG_INF("%s: using the %s chat template\n", __func__, chat_template.empty() ? "model's built-in" : "custom");

    std::vector<json> conversations;
    std::istringstream content_stream(json_content);
    std::string line;
    while (std::getline(content_stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        try {
            json conv = json::parse(line);
            if (conv.contains("messages") && conv["messages"].is_array()) {
                conversations.push_back(std::move(conv));
            }
        } catch (const json::exception & e) {
            LOG_DBG("%s: skipping a line that is not valid JSON: %s\n", __func__, e.what());
        }
    }
    if (conversations.empty()) {
        LOG_ERR("%s: no conversations with a \"messages\" array found\n", __func__);
        return nullptr;
    }
    LOG_INF("%s: loaded %zu conversations\n", __func__, conversations.size());

    const int64_t ne_datapoint = llama_n_ctx(ctx);

    std::vector<std::vector<llama_token>> all_tokens;
    std::vector<std::vector<int32_t>>     all_masks;

    for (size_t i = 0; i < conversations.size(); ++i) {
        const auto & messages = conversations[i]["messages"];
        if (messages.empty()) {
            continue;
        }

        common_chat_templates_inputs inputs;
        for (const auto & msg : messages) {
            if (!msg.contains("role") || !msg.contains("content") || !msg["role"].is_string() || !msg["content"].is_string()) {
                LOG_ERR("%s: conversation %zu has a message without a string \"role\" and \"content\"\n", __func__, i);
                return nullptr;
            }
            common_chat_msg chat_msg;
            chat_msg.role    = msg["role"].get<std::string>();
            chat_msg.content = msg["content"].get<std::string>();
            inputs.messages.push_back(std::move(chat_msg));
        }
        inputs.add_generation_prompt = false;
        inputs.use_jinja             = true;
        inputs.enable_thinking       = std::any_of(inputs.messages.begin(), inputs.messages.end(),
            [](const common_chat_msg & m) { return !m.reasoning_content.empty(); });

        std::string render;
        try {
            render = common_chat_templates_apply(chat_templates.get(), inputs).prompt.str();
        } catch (const std::exception & e) {
            LOG_ERR("%s: the chat template failed on conversation %zu: %s\n", __func__, i, e.what());
            return nullptr;
        }

        const sft_turn_markers * markers = nullptr;
        for (const auto & m : sft_turn_markers_all) {
            if (render.find(m.start) != std::string::npos) {
                markers = &m;
                break;
            }
        }
        if (!markers) {
            LOG_WRN("%s: conversation %zu has no assistant turn, skipped\n", __func__, i);
            continue;
        }

        // anything but whitespace after the last turn end is not part of the conversation
        {
            const size_t last_end = render.rfind(markers->end);
            size_t pos = last_end + strlen(markers->end);
            while (pos < render.size() && (render[pos] == '\n' || render[pos] == '\r' || render[pos] == ' ')) {
                pos++;
            }
            if (pos < render.size()) {
                render.resize(last_end + strlen(markers->end));
            }
        }

        // character spans of the assistant turns: the content and the end marker, not the turn header
        std::vector<std::pair<size_t, size_t>> spans;
        for (size_t from = 0;;) {
            const size_t open = render.find(markers->start, from);
            if (open == std::string::npos) {
                break;
            }
            const size_t lo    = open + strlen(markers->start);
            const size_t close = render.find(markers->end, lo);
            const size_t hi    = close == std::string::npos ? render.size() : close + strlen(markers->end);
            spans.push_back({lo, hi});
            if (close == std::string::npos) {
                break;
            }
            from = hi;
        }

        const std::vector<llama_token> tokens = common_tokenize(ctx, render, /*add_special=*/ false, /*parse_special=*/ true);
        if (tokens.empty()) {
            continue;
        }

        // a token is supervised when its text overlaps an assistant span
        std::vector<int32_t> mask(tokens.size(), 0);
        size_t offset = 0;
        for (size_t t = 0; t < tokens.size(); ++t) {
            const size_t tok_lo = offset;
            const size_t tok_hi = offset + common_token_to_piece(ctx, tokens[t], /*special=*/ true).size();
            for (const auto & sp : spans) {
                if (tok_lo < sp.second && tok_hi > sp.first) {
                    mask[t] = 1;
                    break;
                }
            }
            offset = tok_hi;
        }
        if (offset != render.size()) {
            // the pieces do not add up to the text (tokenizer normalization), count the tokens of each prefix instead
            LOG_WRN("%s: conversation %zu: token pieces cover %zu of %zu bytes, mask boundaries are approximate\n",
                __func__, i, offset, render.size());
            std::fill(mask.begin(), mask.end(), 0);
            for (const auto & sp : spans) {
                const size_t t_lo = std::min(common_tokenize(ctx, render.substr(0, sp.first),  false, true).size(), tokens.size());
                const size_t t_hi = std::min(common_tokenize(ctx, render.substr(0, sp.second), false, true).size(), tokens.size());
                std::fill(mask.begin() + t_lo, mask.begin() + t_hi, 1);
            }
        }

        if (std::count(mask.begin(), mask.end(), 1) == 0) {
            LOG_WRN("%s: conversation %zu has no assistant tokens, skipped\n", __func__, i);
            continue;
        }

        all_tokens.push_back(tokens);
        all_masks.push_back(mask);
    }

    if (all_tokens.empty()) {
        LOG_ERR("%s: no usable conversation in %zu (supported chat formats: ChatML, Gemma 3, Gemma 4)\n", __func__, conversations.size());
        return nullptr;
    }

    llama_token pad_token = llama_vocab_pad(vocab);
    if (pad_token == LLAMA_TOKEN_NULL) {
        pad_token = llama_vocab_eos(vocab);
    }

    // count the samples that fit before allocating the dataset
    int64_t ndata = 0;
    for (size_t i = 0; i < all_tokens.size(); ++i) {
        if ((int64_t) all_tokens[i].size() > ne_datapoint) {
            LOG_WRN("%s: skipping conversation %zu: %zu tokens do not fit the context of %" PRId64 "\n", __func__, i, all_tokens[i].size(), ne_datapoint);
            continue;
        }
        ndata++;
    }
    if (ndata == 0) {
        LOG_ERR("%s: no conversation fits the context size\n", __func__);
        return nullptr;
    }

    ggml_opt_dataset_t result = ggml_opt_dataset_init_with_masks(
        GGML_TYPE_I32, GGML_TYPE_I32, GGML_TYPE_I32,
        /*ne_datapoint=*/ne_datapoint, /*ne_label=*/ne_datapoint, /*ne_mask=*/ne_datapoint,
        /*ndata=*/ndata, /*ndata_shard=*/1);
    if (result == nullptr) {
        return nullptr;
    }

    int32_t * data   = (int32_t *) ggml_opt_dataset_data(result)->data;
    int32_t * labels = (int32_t *) ggml_opt_dataset_labels(result)->data;
    int32_t * masks  = (int32_t *) ggml_opt_dataset_masks(result)->data;

    int64_t idata = 0;
    for (size_t i = 0; i < all_tokens.size(); ++i) {
        const auto & conv_tokens = all_tokens[i];
        const auto & conv_mask   = all_masks[i];

        if ((int64_t) conv_tokens.size() > ne_datapoint) {
            continue;
        }

        int32_t * data_row   = data   + idata * ne_datapoint;
        int32_t * labels_row = labels + idata * ne_datapoint;
        int32_t * masks_row  = masks  + idata * ne_datapoint;

        const int64_t n_tokens = conv_tokens.size();

        // inputs, padded to the full context
        for (int64_t t = 0; t < ne_datapoint; ++t) {
            data_row[t] = t < n_tokens ? conv_tokens[t] : pad_token;
        }

        // labels: the next token for every position (the masked cross entropy needs real tokens)
        for (int64_t t = 0; t < ne_datapoint - 1; ++t) {
            labels_row[t] = data_row[t + 1];
        }
        labels_row[ne_datapoint - 1] = data_row[ne_datapoint - 1]; // last token predicts itself (will be masked)

        // masks: position t predicts token t+1, so it is trained on when token t+1 is an assistant token;
        // padding is never trained on
        for (int64_t t = 0; t < ne_datapoint - 1; ++t) {
            masks_row[t] = (t + 1 < n_tokens && conv_mask[t + 1] == 1) ? 1 : 0;
        }
        masks_row[ne_datapoint - 1] = 0;

        idata++;
    }

    return result;
}
