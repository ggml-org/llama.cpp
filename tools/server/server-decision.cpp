#include "server-decision.h"

#include "../../src/llama-ext.h" // staging API: llama_decision_order

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

static const char * decision_question_type_name(server_decision_question_type type) {
    switch (type) {
        case SERVER_DECISION_QUESTION_CHOICE: return "choice";
        case SERVER_DECISION_QUESTION_SCORE:  return "score";
        case SERVER_DECISION_QUESTION_NOUL:   return "noul";
    }
    return "";
}

// lev reads noul from a rating scale: 0 = certainly no, 8 = certainly yes
static const size_t DECISION_LEV_N_RATINGS = 9;

// h2o-lightning follows serve_config.json and h2o_lightning_shim.py of the model repo
// a longer state keeps its head and its tail
static const size_t DECISION_H2O_MAX_STATE_TOKENS = 32000;

// the recursive functions of this file walk request JSON, a deeper value would overflow the stack
static const int RECURSIVE_LIMIT = 128;

static std::string decision_meta_str(const llama_model * model, const std::string & key) {
    char buf[256];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? "" : std::string(buf);
}

//
// model-specific setup
//

void server_decision_context::init(const llama_model * model) {
    *this = server_decision_context(); // the model can be reloaded

    const common_decision_type model_type = common_get_decision_type(model);
    if (model_type == COMMON_DECISION_TYPE_NONE) {
        return;
    }

    const std::string prefix    = decision_meta_str(model, "general.architecture") + ".decision.";
    const std::string type_name = decision_meta_str(model, prefix + "type");

    vocab = llama_model_get_vocab(model);

    const char * tmpl_src = llama_model_chat_template(model, "systemone");
    if (tmpl_src == nullptr) {
        throw std::runtime_error("decision model has no \"systemone\" template");
    }
    tmpl = std::make_shared<const common_chat_template>(tmpl_src, "", "");

    const std::string prefix_temp = prefix + "temperature.";
    for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
        char key[256];
        char val[64];
        if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix_temp)) {
            continue;
        }
        if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) {
            continue;
        }
        const float temp = std::strtof(val, nullptr);
        if (temp <= 0.0f) {
            throw std::runtime_error(string_format("invalid decision temperature: %s = %s", key, val));
        }
        temperatures[key + prefix_temp.size()] = temp;
    }

    // "<n_levels>.<level>" -> offset
    const std::string prefix_bias = prefix + "score_bias.";
    for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
        char key[256];
        char val[64];
        if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix_bias)) {
            continue;
        }
        if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) {
            continue;
        }
        const auto parts = string_split<std::string>(key + prefix_bias.size(), '.');
        const size_t n     = parts.size() == 2 ? std::strtoul(parts[0].c_str(), nullptr, 10) : 0;
        const size_t level = parts.size() == 2 ? std::strtoul(parts[1].c_str(), nullptr, 10) : 0;
        if (n < 2 || n > 255 || level >= n) {
            throw std::runtime_error(string_format("invalid decision score bias: %s", key));
        }
        auto & bias = score_bias[n];
        bias.resize(n, 0.0f);
        bias[level] = std::strtof(val, nullptr);
    }

    if (model_type == COMMON_DECISION_TYPE_OPENJEV) {
        // one letter per option, each must be a single token
        const std::string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        for (const char c : letters) {
            const auto toks = common_tokenize(vocab, std::string(1, c), false, false);
            if (toks.size() != 1) {
                throw std::runtime_error(string_format("decision label '%c' is not a single token", c));
            }
            labels.push_back(toks[0]);
        }
        n_options_max   = labels.size();
        noul_true_first = true;
    } else if (model_type == COMMON_DECISION_TYPE_LEV || model_type == COMMON_DECISION_TYPE_GRANITE_SWITCH || model_type == COMMON_DECISION_TYPE_NIMBLE || model_type == COMMON_DECISION_TYPE_PPLX_DECIDER || model_type == COMMON_DECISION_TYPE_VLLM_D3) {
        // label codes are A..Z then AA..ZZ, only the ones that are a single token are used
        std::vector<std::string> codes;
        for (char a = 'A'; a <= 'Z'; a++) {
            codes.push_back(std::string(1, a));
        }
        for (char a = 'A'; a <= 'Z'; a++) {
            for (char b = 'A'; b <= 'Z'; b++) {
                codes.push_back(std::string{a, b});
            }
        }
        for (const auto & code : codes) {
            const auto toks = common_tokenize(vocab, code, false, false);
            if (toks.size() == 1 && labels.size() < 255) {
                labels.push_back(toks[0]);
                label_texts.push_back(code);
            }
        }
        n_options_max = labels.size();
        if (model_type == COMMON_DECISION_TYPE_VLLM_D3) {
            canonical_legend = true;
        }
        if (model_type == COMMON_DECISION_TYPE_GRANITE_SWITCH) {
            noul_true_first = true;
        }
    } else if (model_type == COMMON_DECISION_TYPE_KEV) {
        // the hidden state of an option is read at the token that ends it
        const auto toks = common_tokenize(vocab, "<|box_end|>", false, true);
        if (toks.size() != 1) {
            throw std::runtime_error("decision model has no <|box_end|> token");
        }
        token_marker  = toks[0];
        n_options_max = 255;
    } else if (model_type == COMMON_DECISION_TYPE_LAYA) {
        token_marker = llama_vocab_mask(vocab);
        token_sep    = llama_vocab_sep(vocab);
        if (token_marker == LLAMA_TOKEN_NULL || token_sep == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("decision model has no mask or sep token");
        }
        text_marker = common_token_to_piece(vocab, token_marker, true);

        const std::string val = decision_meta_str(model, prefix + "max_head_tokens");
        max_head_tokens = std::strtoul(val.c_str(), nullptr, 10);
        if (max_head_tokens == 0) {
            throw std::runtime_error("decision model has no valid max_head_tokens");
        }
        n_options_max = 255;
    } else if (model_type == COMMON_DECISION_TYPE_CLEF) {
        n_options_max   = 255;
        noul_true_first = true;
        choice_sorted   = true;
    } else if (model_type == COMMON_DECISION_TYPE_VLLM_DECISION2) {
        n_options_max       = 255;
        noul_criteria_order = true;
        strict_payload      = true;
        canonical_legend    = true;
    } else if (model_type == COMMON_DECISION_TYPE_LFM2_D1 || model_type == COMMON_DECISION_TYPE_LFM3_D1 || model_type == COMMON_DECISION_TYPE_LFM3_D1_OMNI) {
        n_options_max   = 255;
        noul_true_first = true;
        if (model_type != COMMON_DECISION_TYPE_LFM2_D1) {
            // the texts of the special tokens are broken in the input
            init_quoted_spellings(LLAMA_TOKEN_ATTR_CONTROL, 2);
        }
    } else if (model_type == COMMON_DECISION_TYPE_LFM2_D1_OMNI) {
        token_marker = llama_vocab_mask(vocab);
        if (token_marker == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("decision model has no mask token");
        }
        n_options_max = 255;
    } else if (model_type == COMMON_DECISION_TYPE_H2O_LIGHTNING) {
        // the answer is read at the token of a label, its text is given to the template
        for (uint32_t i = 0; llama_model_decision_label(model, i) != nullptr; i++) {
            const std::string text = llama_model_decision_label(model, i);
            const auto toks = common_tokenize(vocab, text, false, false);
            if (toks.size() != 1 || std::find(labels.begin(), labels.end(), toks[0]) != labels.end()) {
                throw std::runtime_error(string_format("decision label \"%s\" is not a single distinct token", text.c_str()));
            }
            labels.push_back(toks[0]);
            label_texts.push_back(text);
        }
        if (labels.empty()) {
            throw std::runtime_error("decision model has no labels");
        }
        // the texts of the control and user-defined tokens are quoted in the input, so that they stay text
        init_quoted_spellings(LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED, 1);
        n_options_max   = labels.size();
        noul_true_first = true;
    } else if (model_type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV) {
        // the classes come from the GGUF classifier labels, the request API is binary: [inconsistent, consistent]
        const uint32_t n_classes = llama_model_n_cls_out(model);
        if (n_classes != 2 || llama_model_n_embd_out(model) != (int32_t) n_classes) {
            throw std::runtime_error(string_format("decision model must have 2 classifier labels and output one score per class, got %u labels and %d scores",
                n_classes, llama_model_n_embd_out(model)));
        }
        n_options_max = n_classes;
        choice_sorted = true;
    } else {
        throw std::runtime_error("unsupported decision model type: " + type_name);
    }
    type = model_type;

    SRV_INF("decision model type: %s\n", type_name.c_str());
}

void server_decision_context::init_quoted_spellings(uint32_t attrs, size_t min_size) {
    for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); id++) {
        if (llama_vocab_get_attr(vocab, id) & attrs) {
            const std::string text = llama_vocab_get_text(vocab, id);
            if (text.size() >= min_size) {
                quoted_spellings.push_back(text);
            }
        }
    }
    std::stable_sort(quoted_spellings.begin(), quoted_spellings.end(), [](const std::string & a, const std::string & b) {
        return a.size() > b.size();
    });
    for (const auto & spelling : quoted_spellings) {
        if (quoted_first.find(spelling[0]) == std::string::npos) {
            quoted_first += spelling[0];
        }
    }
}

//
// request parsing
//

// lfm3-d1-omni: a spoken question, its instructions are an input_audio part
static bool decision_is_spoken(const json & instructions) {
    return instructions.is_object() && json_value(instructions, "type", std::string()) == "input_audio";
}

static bool decision_is_payload(const json & val) {
    return val.is_string() || val.is_object() || val.is_array();
}

std::vector<server_decision_question> server_decision_context::parse_questions(const json & body) const {
    // lfm2-d1, lfm2-d1-omni, lfm3-d1 and vllm-d3 accept a null state, for example to ask about images only
    const bool null_state = type == COMMON_DECISION_TYPE_LFM2_D1 || type == COMMON_DECISION_TYPE_LFM2_D1_OMNI || type == COMMON_DECISION_TYPE_VLLM_D3 || type == COMMON_DECISION_TYPE_LFM3_D1 || type == COMMON_DECISION_TYPE_LFM3_D1_OMNI;
    if (!body.contains("state") || (body.at("state").is_null() && !null_state)) {
        throw std::invalid_argument("\"state\" must be provided");
    }
    if (strict_payload && !decision_is_payload(body.at("state"))) {
        throw std::invalid_argument("\"state\" must be text, an object or an array");
    }
    if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
        throw std::invalid_argument("\"questions\" must be a non-empty object");
    }
    if (type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV && body.at("questions").size() != 1) {
        throw std::invalid_argument("this model answers exactly one question");
    }

    std::vector<server_decision_question> questions;
    for (const auto & [id, q] : body.at("questions").items()) {
        auto err = [&id = id](const std::string & msg) {
            return std::invalid_argument("questions." + id + ": " + msg);
        };
        if (!q.is_object()) {
            throw err("must be an object");
        }
        // the vllm-d3 prompt has a default question
        // h2o-lightning: the instructions can be left out, the template has a default, and the type defaults to choice
        const bool is_h2o = type == COMMON_DECISION_TYPE_H2O_LIGHTNING;
        const json instructions = q.contains("instructions") ? q.at("instructions") : (is_h2o ? json(std::string()) : json());
        // minicpm-ocr-jev answers one fixed question, the prompt is entirely in the template
        const bool is_ocr_jev = type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV;
        if (instructions.is_null() && type != COMMON_DECISION_TYPE_VLLM_D3 && !is_h2o && !is_ocr_jev) {
            throw err("\"instructions\" must be provided");
        }
        if (strict_payload && (!decision_is_payload(instructions) || instructions == "")) {
            throw err("\"instructions\" must be non-empty text, an object or an array");
        }
        if (type == COMMON_DECISION_TYPE_LFM3_D1 && decision_is_spoken(instructions)) {
            throw err("spoken instructions need a model with audio input");
        }

        server_decision_question question;
        question.id           = id;
        question.instructions = instructions;

        const std::string type_name = json_value(q, "type", std::string(is_h2o ? "choice" : ""));
        const json        criteria  = q.contains("criteria") ? q.at("criteria") : json();

        if (type_name == "choice") {
            question.type = SERVER_DECISION_QUESTION_CHOICE;
            if (!criteria.is_object() || criteria.empty()) {
                throw err("\"criteria\" must be a non-empty object");
            }
            for (const auto & [key, description] : criteria.items()) {
                if (strict_payload && (key.empty() || !(description.is_null() || decision_is_payload(description)))) {
                    throw err("option keys must be non-empty, descriptions must be null, text, an object or an array");
                }
                question.options.push_back({key, description});
            }
            if (strict_payload && question.options.size() < 2) {
                throw err("\"criteria\" must have at least 2 options");
            }
            if (choice_sorted) {
                std::sort(question.options.begin(), question.options.end(), [](const auto & a, const auto & b) {
                    return a.key < b.key;
                });
            }
        } else if (type_name == "score") {
            question.type = SERVER_DECISION_QUESTION_SCORE;
            // h2o-lightning: from 1 level, the maximum is n_options_max
            if (is_h2o && (!criteria.is_array() || criteria.empty())) {
                throw err("\"criteria\" must be a non-empty array");
            }
            if (!is_h2o && (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10)) {
                throw err("\"criteria\" must be an array of 2 to 10 levels");
            }
            for (size_t i = 0; i < criteria.size(); i++) {
                if (strict_payload && !decision_is_payload(criteria.at(i))) {
                    throw err("level descriptions must be text, an object or an array");
                }
                question.options.push_back({std::to_string(i), criteria.at(i)});
            }
        } else if (type_name == "noul") {
            question.type = SERVER_DECISION_QUESTION_NOUL;
            if (!criteria.is_null() && !criteria.is_object()) {
                throw err("\"criteria\" must be an object");
            }
            if (strict_payload && criteria.is_object()) {
                for (const auto & [key, description] : criteria.items()) {
                    if ((key != "false" && key != "true") || !decision_is_payload(description)) {
                        throw err("\"criteria\" may only describe \"false\" and \"true\" with text, an object or an array");
                    }
                }
            }
            for (const char * key : {"false", "true"}) {
                json description;
                if (criteria.is_object() && criteria.contains(key)) {
                    description = criteria.at(key);
                } else if (criteria.is_object() && (type == COMMON_DECISION_TYPE_LFM2_D1_OMNI || is_h2o)) {
                    // lfm2-d1-omni and h2o-lightning also read the descriptions under "no" and "yes"
                    const char * alias = std::string(key) == "true" ? "yes" : "no";
                    description = criteria.contains(alias) ? criteria.at(alias) : json();
                }
                question.options.push_back({key, description});
            }
            if (noul_true_first || (noul_criteria_order && criteria.is_object() && criteria.size() == 2 && criteria.begin().key() == "true")) {
                std::swap(question.options[0], question.options[1]);
            }
        } else {
            throw err("\"type\" must be one of: choice, score, noul");
        }

        // the classes are fixed: "0" is inconsistent, "1" is consistent
        if (type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV) {
            // the task text is in the template, an instructions from the request would be dropped
            if (!instructions.is_null()) {
                throw err("\"instructions\" is not used by this model, the task is part of its prompt");
            }
            if (question.type == SERVER_DECISION_QUESTION_CHOICE) {
                const auto & opts = question.options;
                if (opts.size() != 2 || opts[0].key != "0" || opts[1].key != "1" || !opts[0].description.is_null() || !opts[1].description.is_null()) {
                    throw err("\"criteria\" must be {\"0\": null, \"1\": null}");
                }
            } else if (question.type == SERVER_DECISION_QUESTION_NOUL) {
                if (!criteria.is_null()) {
                    throw err("\"criteria\" is not supported for a noul question");
                }
            } else {
                throw err("only choice and noul questions are supported");
            }
        }

        if (question.options.size() > n_options_max) {
            throw err(string_format("too many options (%zu), this model supports at most %zu", question.options.size(), n_options_max));
        }

        questions.push_back(std::move(question));
    }
    return questions;
}

//
// images
//

static const size_t DECISION_MAX_IMAGES = 8;

// any media, mtmd tells an audio clip or a video from an image by its content
static void decision_load_image(const json & url, std::vector<raw_buffer> & files) {
    if (!url.is_string() || !string_starts_with(url.get<std::string>(), "data:")) {
        throw std::invalid_argument("media must be data URLs (data:image/..., data:audio/... or data:video/...;base64,...)");
    }
    if (files.size() >= DECISION_MAX_IMAGES) {
        throw std::invalid_argument(string_format("too many images, the maximum is %zu", DECISION_MAX_IMAGES));
    }
    handle_media(files, url.get<std::string>(), "");
}

// audio is base64 data, as a data URL or not (OpenAI input_audio)
static void decision_load_audio(const json & data, std::vector<raw_buffer> & files) {
    if (!data.is_string() || string_starts_with(data.get<std::string>(), "http") || string_starts_with(data.get<std::string>(), "file://")) {
        throw std::invalid_argument("audio must be base64 data");
    }
    if (files.size() >= DECISION_MAX_IMAGES) {
        throw std::invalid_argument(string_format("too many media files, the maximum is %zu", DECISION_MAX_IMAGES));
    }
    handle_media(files, data.get<std::string>(), "");
}

json server_decision_context::parse_state(const json & body, std::vector<raw_buffer> & files) const {
    if (type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV && !body.at("state").is_string()) {
        throw std::invalid_argument("\"state\" must be the OCR text");
    }
    if (body.contains("videos") && !body.at("videos").is_null() && !body.at("videos").empty()) {
        throw std::invalid_argument("\"videos\" is not supported");
    }
    // "images" is an alias of "files"
    for (const char * key : {"files", "images"}) {
        if (!body.contains(key) || body.at(key).is_null()) {
            continue;
        }
        if (!body.at(key).is_array()) {
            throw std::invalid_argument(string_format("\"%s\" must be an array", key));
        }
        for (const auto & url : body.at(key)) {
            decision_load_image(url, files);
        }
    }

    // the prompt has the original image, then the image rendered from the OCR text
    if (type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV && files.size() != 2) {
        throw std::invalid_argument(string_format("this model needs exactly 2 images (the original, then the rendered one), got %zu", files.size()));
    }

    const json & state = body.at("state");
    const bool is_wrapped = state.is_object() && state.contains("messages");
    const json & messages = is_wrapped ? state.at("messages") : state;
    if (!messages.is_array()) {
        return state;
    }

    // chat messages: take the image and audio parts out of the content
    json messages_out = json::array();
    for (const auto & msg : messages) {
        if (!msg.is_object() || !msg.contains("content") || !msg.at("content").is_array()) {
            messages_out.push_back(msg);
            continue;
        }
        json content = json::array();
        for (const auto & part : msg.at("content")) {
            if (part.is_object() && json_value(part, "type", std::string()) == "image_url" && part.contains("image_url")) {
                const json & image_url = part.at("image_url");
                decision_load_image(image_url.is_object() && image_url.contains("url") ? image_url.at("url") : image_url, files);
            } else if (part.is_object() && json_value(part, "type", std::string()) == "input_audio" && part.contains("input_audio")) {
                const json input_audio = json_value(part, "input_audio", json::object());
                decision_load_audio(input_audio.contains("data") ? input_audio.at("data") : json_value(input_audio, "url", json()), files);
            } else if (is_lfm3_d1() && part.is_object() && json_value(part, "type", std::string()) == "video_url" && part.contains("video_url")) {
                const json & video_url = part.at("video_url");
                decision_load_image(video_url.is_object() && video_url.contains("url") ? video_url.at("url") : video_url, files);
            } else {
                content.push_back(part);
            }
        }
        json msg_out = msg;
        msg_out["content"] = content;
        messages_out.push_back(msg_out);
    }

    if (!is_wrapped) {
        return messages_out;
    }
    json state_out = state;
    state_out["messages"] = messages_out;
    return state_out;
}

//
// prompt
//

// replace text in all strings of a JSON value
static json decision_replace_text(const json & val, const std::string & search, const std::string & replace, int depth = 0) {
    if (depth >= RECURSIVE_LIMIT) {
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    }
    if (val.is_string()) {
        std::string str = val.get<std::string>();
        string_replace_all(str, search, replace);
        return str;
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_replace_text(item, search, replace, depth + 1));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[key] = decision_replace_text(item, search, replace, depth + 1);
        }
        return out;
    }
    return val;
}

// sort the keys of all objects of a JSON value
static json decision_sort_keys(const json & val, int depth = 0) {
    if (depth >= RECURSIVE_LIMIT) {
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_sort_keys(item, depth + 1));
        }
        return out;
    }
    if (val.is_object()) {
        std::map<std::string, json> sorted;
        for (const auto & [key, item] : val.items()) {
            sorted[key] = decision_sort_keys(item, depth + 1);
        }
        json out = json::object();
        for (const auto & [key, item] : sorted) {
            out[key] = item;
        }
        return out;
    }
    return val;
}

// kev flattens a JSON value into text, the keys of an object are kept as labels (kev/api.py: render)
static std::string decision_kev_render(const json & val, int indent = 0, int depth = 0) {
    if (depth >= RECURSIVE_LIMIT) {
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    }
    const std::string pad(2 * indent, ' ');
    if (val.is_null()) {
        return "";
    }
    if (val.is_string()) {
        return val.get<std::string>();
    }
    if (val.is_boolean()) {
        return val.get<bool>() ? "True" : "False";
    }
    if (val.is_array()) {
        std::string out;
        for (const auto & item : val) {
            const std::string text = decision_kev_render(item, indent + 1, depth + 1);
            out += (out.empty() ? "" : "\n") + pad + "- " + text.substr(std::min(text.size(), text.find_first_not_of(" \t\n\r")));
        }
        return out;
    }
    if (val.is_object()) {
        std::string out;
        for (const auto & [key, item] : val.items()) {
            const bool is_nested = item.is_object() || item.is_array();
            out += (out.empty() ? "" : "\n") + pad + key + (is_nested ? ":\n" : ": ") + decision_kev_render(item, is_nested ? indent + 1 : 0, depth + 1);
        }
        return out;
    }
    return val.dump();
}

// same result as std::regex_replace with <\|([A-Za-z0-9_]+)\|>
// std::regex is not used: libstdc++ recurses for each char, a long input overflows the stack
std::string server_decision_escape_special_tokens(const std::string & text) {
    const auto is_name = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    };
    std::string out;
    size_t n_done = 0; // the chars of text that are in out
    for (size_t i = text.find("<|"); i != std::string::npos; i = text.find("<|", std::max(i + 1, n_done))) {
        size_t j = i + 2;
        while (j < text.size() && is_name(text[j])) {
            j++;
        }
        if (j > i + 2 && text.compare(j, 2, "|>") == 0) {
            out.append(text, n_done, i - n_done);
            out += "<\xC2\xA6";
            out.append(text, i + 2, j - i - 2);
            out += "\xC2\xA6>";
            n_done = j + 2;
        }
    }
    if (n_done == 0) {
        return text;
    }
    out.append(text, n_done, std::string::npos);
    return out;
}

// kev text input: special tokens written in the text must not be parsed as such
static std::string decision_kev_text(const json & val) {
    return server_decision_escape_special_tokens(decision_kev_render(val));
}

// lfm2-d1-omni, lfm3-d1: special tokens written in the input must not be parsed as such, in keys too
template <typename F>
static json decision_escape(const json & val, const F & escape, int depth = 0) {
    if (depth >= RECURSIVE_LIMIT) {
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    }
    if (val.is_string()) {
        return escape(val.get<std::string>());
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_escape(item, escape, depth + 1));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[escape(key)] = decision_escape(item, escape, depth + 1);
        }
        return out;
    }
    return val;
}

std::string server_decision_context::d1_break(const std::string & text) const {
    static const std::string zwsp = "\xE2\x80\x8B";
    std::string cur = text;
    // again for a match that overlapped the one broken before it
    for (bool changed = true; changed;) {
        changed = false;
        std::string out;
        for (size_t i = 0; i < cur.size();) {
            const std::string * hit = nullptr;
            if (quoted_first.find(cur[i]) != std::string::npos) {
                for (const auto & spelling : quoted_spellings) {
                    if (cur.compare(i, spelling.size(), spelling) == 0) {
                        hit = &spelling;
                        break;
                    }
                }
            }
            if (hit == nullptr) {
                out += cur[i++];
                continue;
            }
            // after the first character, its UTF-8 bytes
            const unsigned char c = (*hit)[0];
            const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            out += hit->substr(0, n) + zwsp + hit->substr(n);
            i += hit->size();
            changed = true;
        }
        cur = std::move(out);
    }
    return cur;
}

//
// h2o-lightning: the texts are rendered the way the reference shim renders them with Python (json.dumps, str.lstrip)
//

// one code point of UTF-8 text at s[i], i moves past it; an invalid byte is read as is
static uint32_t decision_utf8_next(const std::string & s, size_t & i) {
    const unsigned char c = s[i];
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : (c >> 3) == 0x1e ? 4 : 1;
    if (i + n > s.size()) {
        n = 1;
    }
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1f) : n == 3 ? (c & 0x0f) : (c & 0x07);
    for (size_t k = 1; k < n; k++) {
        cp = (cp << 6) | (s[i + k] & 0x3f);
    }
    i += n;
    return cp;
}

// str.isspace() of Python
static bool decision_py_isspace(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

static std::string decision_py_rstrip(const std::string & s) {
    size_t end = 0;
    for (size_t i = 0; i < s.size();) {
        if (!decision_py_isspace(decision_utf8_next(s, i))) {
            end = i;
        }
    }
    return s.substr(0, end);
}

static std::string decision_py_lstrip(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        size_t next = i;
        if (!decision_py_isspace(decision_utf8_next(s, next))) {
            break;
        }
        i = next;
    }
    return s.substr(i);
}

// a code point as JSON escapes, a surrogate pair above the BMP
static std::string decision_py_escape_cp(uint32_t cp) {
    if (cp < 0x10000) {
        return string_format("\\u%04x", cp);
    }
    cp -= 0x10000;
    return string_format("\\u%04x\\u%04x", 0xd800 + (cp >> 10), 0xdc00 + (cp & 0x3ff));
}

// repr() of a Python float: the shortest digits that read back the same, in exponent form below 1e-4 and from 1e16
static std::string decision_py_float(double v) {
    char buf[64];
    // the fewest digits that read back the same
    for (int prec = 0; prec < 17; prec++) {
        snprintf(buf, sizeof(buf), "%.*e", prec, v);
        if (std::strtod(buf, nullptr) == v) {
            break;
        }
    }
    const std::string sci(buf); // for example "-1.25e+03"
    const size_t pos_e = sci.find('e');
    const int    exp   = std::atoi(sci.c_str() + pos_e + 1);
    std::string sign;
    std::string digits;
    for (size_t i = 0; i < pos_e; i++) {
        if (sci[i] == '-') {
            sign = "-";
        } else if (sci[i] != '.') {
            digits += sci[i];
        }
    }
    const int n = (int) digits.size();
    if (exp < -4 || exp >= 16) {
        return sign + digits.substr(0, 1) + (n > 1 ? "." + digits.substr(1) : "") + string_format("e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
    }
    if (exp >= n - 1) {
        return sign + digits + std::string(exp - (n - 1), '0') + ".0";
    }
    if (exp >= 0) {
        return sign + digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
    }
    return sign + "0." + std::string(-exp - 1, '0') + digits;
}

static void decision_py_str(std::string & out, const std::string & s, bool ensure_ascii) {
    out += '"';
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        switch (c) {
            case '"':  out += "\\\""; i++; continue;
            case '\\': out += "\\\\"; i++; continue;
            case '\b': out += "\\b";  i++; continue;
            case '\f': out += "\\f";  i++; continue;
            case '\n': out += "\\n";  i++; continue;
            case '\r': out += "\\r";  i++; continue;
            case '\t': out += "\\t";  i++; continue;
            default:   break;
        }
        if (c < 0x20 || (ensure_ascii && c == 0x7f)) {
            out += string_format("\\u%04x", c);
            i++;
        } else if (c < 0x80 || !ensure_ascii) {
            out += (char) c;
            i++;
        } else {
            out += decision_py_escape_cp(decision_utf8_next(s, i));
        }
    }
    out += '"';
}

// json.dumps() of Python: indent < 0 is no indent, keys stay in their order
static void decision_py_dump(std::string & out, const json & val, int indent, int level, const std::string & item_sep, const std::string & key_sep, bool ensure_ascii) {
    if (level >= RECURSIVE_LIMIT) {
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    }
    const auto newline = [&](int lvl) {
        return indent < 0 ? std::string() : "\n" + std::string(indent * lvl, ' ');
    };
    if (val.is_null()) {
        out += "null";
    } else if (val.is_boolean()) {
        out += val.get<bool>() ? "true" : "false";
    } else if (val.is_number_float()) {
        out += decision_py_float(val.get<double>());
    } else if (val.is_number()) {
        out += val.dump();
    } else if (val.is_string()) {
        decision_py_str(out, val.get<std::string>(), ensure_ascii);
    } else if (val.is_array()) {
        if (val.empty()) {
            out += "[]";
            return;
        }
        out += "[" + newline(level + 1);
        for (size_t i = 0; i < val.size(); i++) {
            if (i > 0) {
                out += item_sep + newline(level + 1);
            }
            decision_py_dump(out, val.at(i), indent, level + 1, item_sep, key_sep, ensure_ascii);
        }
        out += newline(level) + "]";
    } else if (val.is_object()) {
        if (val.empty()) {
            out += "{}";
            return;
        }
        out += "{" + newline(level + 1);
        bool first = true;
        for (const auto & [key, item] : val.items()) {
            if (!first) {
                out += item_sep + newline(level + 1);
            }
            first = false;
            decision_py_str(out, key, ensure_ascii);
            out += key_sep;
            decision_py_dump(out, item, indent, level + 1, item_sep, key_sep, ensure_ascii);
        }
        out += newline(level) + "}";
    }
}

static std::string decision_py_dumps(const json & val, int indent, const std::string & item_sep, const std::string & key_sep, bool ensure_ascii) {
    std::string out;
    decision_py_dump(out, val, indent, 0, item_sep, key_sep, ensure_ascii);
    return out;
}

// a state longer than max_state_tokens keeps its head and its tail around a "..." line
std::string server_decision_context::h2o_cut_state(const std::string & text) const {
    // a byte-level BPE never has more tokens than bytes
    const size_t max_state_tokens = DECISION_H2O_MAX_STATE_TOKENS;
    if (text.size() <= max_state_tokens) {
        return text;
    }
    const llama_tokens tokens = common_tokenize(vocab, text, false, true);
    if (tokens.size() <= max_state_tokens) {
        return text;
    }
    // each half is the text of its token pieces, a cut inside a character gives U+FFFD as in a JSON response of the server
    const size_t half = max_state_tokens / 2;
    auto text_of = [&](llama_tokens::const_iterator begin, llama_tokens::const_iterator end) {
        std::string out;
        for (auto it = begin; it != end; ++it) {
            out += common_token_to_piece(vocab, *it, true);
        }
        return json::parse(json::make(out).dump_safe()).get<std::string>();
    };
    return text_of(tokens.begin(), tokens.begin() + half) + "\n...\n" + text_of(tokens.end() - half, tokens.end());
}

// the special token texts written in the input are quoted with JSON escapes, so they stay text
std::string server_decision_context::h2o_quote(const std::string & text) const {
    std::string out;
    for (size_t i = 0; i < text.size();) {
        bool is_quoted = false;
        if (quoted_first.find(text[i]) == std::string::npos) {
            out += text[i++];
            continue;
        }
        for (const auto & spelling : quoted_spellings) {
            if (text.compare(i, spelling.size(), spelling) == 0) {
                std::string quoted = spelling;
                string_replace_all(quoted, "<", "\\u003c");
                string_replace_all(quoted, ">", "\\u003e");
                out += quoted;
                i += spelling.size();
                is_quoted = true;
                break;
            }
        }
        if (!is_quoted) {
            out += text[i++];
        }
    }
    return out;
}

json server_decision_context::prepare_state(const json & state) const {
    if (type != COMMON_DECISION_TYPE_H2O_LIGHTNING) {
        return state;
    }
    // done once per request: compact JSON, the cut, the quoting
    return h2o_quote(h2o_cut_state(state.is_string() ? state.get<std::string>() : decision_py_dumps(state, -1, ",", ":", false)));
}

json server_decision_context::render_h2o(const json & state, const server_decision_question & question) const {
    // the instructions are JSON with an indent of 1, the descriptions are default JSON, the state is from prepare_state()
    const std::string instructions = decision_py_lstrip(question.instructions.is_string() ?
            question.instructions.get<std::string>() : decision_py_dumps(question.instructions, 1, ",", ": ", true));

    json options = json::array();
    for (size_t i = 0; i < question.options.size(); i++) {
        const auto & opt = question.options[i];
        json description; // null if not provided, the template has a default, except a score level that is shown as JSON null
        if (question.type == SERVER_DECISION_QUESTION_SCORE && opt.description.is_null()) {
            description = "null";
        } else if (!opt.description.is_null()) {
            description = h2o_quote(opt.description.is_string() ? opt.description.get<std::string>() : decision_py_dumps(opt.description, -1, ", ", ": ", false));
        }
        const std::string key = h2o_quote(opt.key);

        // the user turn is stripped like Python's str.strip(), the template only trims ASCII whitespace
        // so the text that ends the turn is stripped here, a space keeps a description that is only whitespace
        if (i + 1 == question.options.size()) {
            const bool is_key = question.type == SERVER_DECISION_QUESTION_CHOICE && description.is_null();
            if (is_key || description.is_string()) {
                const std::string text = is_key ? key : description.get<std::string>();
                const std::string stripped = decision_py_rstrip(text);
                if (stripped != text) {
                    description = stripped.empty() ? " " : stripped;
                }
            }
        }

        json option = json::object();
        option["label"]       = label_texts[i];
        option["key"]         = key;
        option["description"] = description;
        options.push_back(option);
    }

    json inp = json::object();
    inp["type"]         = decision_question_type_name(question.type);
    inp["state"]        = state;
    inp["instructions"] = h2o_quote(instructions);
    inp["options"]      = options;
    return inp;
}

// given to the lfm2-d1-omni template: text between the pieces of the prompt, and at the start of the pieces that are cut to a token budget
static const std::string D1OMNI_MARKER        = "<<d1omni:";
static const std::string D1OMNI_SEP           = "<<d1omni:sep>>";
static const std::string D1OMNI_MARK_STATE    = "<<d1omni:state>>";
static const std::string D1OMNI_MARK_QUESTION = "<<d1omni:question>>";
static const std::string D1OMNI_MARK_OPTION   = "<<d1omni:option>>";

// max_length, image_text_length and audio_text_length of the model config, the converter checks them
static const size_t D1OMNI_MAX_TOKENS       = 16384;
static const size_t D1OMNI_MAX_TOKENS_IMAGE = 896;
static const size_t D1OMNI_MAX_TOKENS_AUDIO = 15360;

size_t server_decision_context::n_variants(const server_decision_question & question) const {
    // lev shows the options of a choice in 2 orders, to cancel the preference for the first label
    if (type == COMMON_DECISION_TYPE_LEV && question.type == SERVER_DECISION_QUESTION_CHOICE && question.options.size() > 1) {
        return 2;
    }
    return 1;
}

size_t server_decision_context::n_outputs(const server_decision_question & question) const {
    if (type == COMMON_DECISION_TYPE_LEV && question.type == SERVER_DECISION_QUESTION_NOUL) {
        return DECISION_LEV_N_RATINGS;
    }
    return question.options.size();
}

// label codes follow prompt.py of the model repo
void server_decision_context::d1_labels(const server_decision_question & question, std::vector<std::string> & texts, std::vector<llama_tokens> & groups) const {
    const size_t n_options = question.options.size();

    auto get_single_tokens = [&](const std::vector<std::string> & forms) {
        llama_tokens out;
        for (const auto & form : forms) {
            const auto toks = common_tokenize(vocab, form, false, false);
            if (toks.size() == 1 && std::find(out.begin(), out.end(), toks[0]) == out.end()) {
                out.push_back(toks[0]);
            }
        }
        return out;
    };

    if (question.type != SERVER_DECISION_QUESTION_CHOICE) {
        for (const auto & opt : question.options) {
            llama_tokens group;
            if (question.type == SERVER_DECISION_QUESTION_SCORE) {
                group = get_single_tokens({opt.key});
            } else if (opt.key == "true") {
                group = get_single_tokens({"yes", "Yes", "YES"});
            } else {
                group = get_single_tokens({"no", "No", "NO"});
            }
            if (group.empty()) {
                throw std::runtime_error("decision label is not a single token: " + opt.key);
            }
            texts.push_back(opt.key);
            groups.push_back(group);
        }
        return;
    }

    bool is_letters = true;
    for (const auto & opt : question.options) {
        is_letters = is_letters && opt.key.size() == 1 && std::isalpha((unsigned char) opt.key[0]);
    }

    std::vector<std::string> codes;
    for (size_t i = 0; i < n_options; i++) {
        if (is_letters) {
            codes.push_back(question.options[i].key);
        } else if (n_options <= 26) {
            codes.push_back(std::string(1, 'A' + i));
        } else if (type == COMMON_DECISION_TYPE_LFM3_D1 || type == COMMON_DECISION_TYPE_LFM3_D1_OMNI) {
            // A..Z, then AA, AB..
            codes.push_back(i < 26 ? std::string(1, 'A' + i) : std::string{(char) ('A' + (i - 26) / 26), (char) ('A' + (i - 26) % 26)});
        } else {
            codes.push_back(string_format("%02zu", i));
        }
    }

    std::vector<std::string> pool;
    for (char c = 'A'; c <= 'Z'; c++) {
        pool.push_back(std::string(1, c));
    }
    for (int i = 0; i < 100; i++) {
        pool.push_back(string_format("%02d", i));
    }
    for (char c = 'a'; c <= 'z'; c++) {
        pool.push_back(std::string(1, c));
    }
    for (int i = 0; i < 200; i++) {
        pool.push_back(string_format("#%d", i));
    }
    for (char a = 'A'; a <= 'Z'; a++) {
        for (char b = 'A'; b <= 'Z'; b++) {
            pool.push_back(std::string{a, b});
        }
    }

    llama_tokens used;
    auto take = [&](const std::string & code) {
        const auto toks = common_tokenize(vocab, code, false, false);
        if (toks.size() != 1 || std::find(used.begin(), used.end(), toks[0]) != used.end()) {
            return false;
        }
        used.push_back(toks[0]);
        llama_tokens group = {toks[0]};
        for (const llama_token tok : get_single_tokens({" " + code})) {
            if (tok != toks[0]) {
                group.push_back(tok);
            }
        }
        texts.push_back(code);
        groups.push_back(group);
        return true;
    };
    for (const auto & code : codes) {
        bool is_taken = take(code);
        for (size_t i = 0; !is_taken && i < pool.size(); i++) {
            is_taken = take(pool[i]);
        }
        if (!is_taken) {
            throw std::invalid_argument(string_format("no single-token label left for %zu options", n_options));
        }
    }
}

json server_decision_context::render_options(const server_decision_question & question, size_t variant) const {
    const size_t n_options = question.options.size();

    std::vector<std::string>  d1_texts;
    std::vector<llama_tokens> d1_groups;
    if (type == COMMON_DECISION_TYPE_LFM2_D1 || type == COMMON_DECISION_TYPE_LFM3_D1 || type == COMMON_DECISION_TYPE_LFM3_D1_OMNI) {
        d1_labels(question, d1_texts, d1_groups);
    }

    // the second variant shows the options in the reverse order
    json options = json::array();
    for (size_t i = 0; i < n_options; i++) {
        const auto & opt = question.options[variant == 0 ? i : n_options - 1 - i];
        json option = json{
            {"key",         opt.key},
            {"description", opt.description},
        };
        if (type == COMMON_DECISION_TYPE_KEV) {
            option["key"] = decision_kev_text(opt.key);
            if (!opt.description.is_null()) {
                option["description"] = decision_kev_text(opt.description);
            }
        }
        if (!label_texts.empty()) {
            option["label"] = label_texts[i];
        }
        if (!d1_texts.empty()) {
            option["label"] = d1_texts[i];
        }
        options.push_back(option);
    }
    return options;
}

std::string server_decision_context::render(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const server_decision_question & question,
        size_t variant,
        size_t n_images,
        bool is_audio) const {
    // the template is given raw JSON values, it serializes the ones that are not strings
    // h2o-lightning: the template is given texts
    json inp = type == COMMON_DECISION_TYPE_H2O_LIGHTNING ? render_h2o(state, question) : json{
        {"id",           question.id},
        {"type",         decision_question_type_name(question.type)},
        {"instructions", question.instructions},
        {"state",        state},
        {"options",      render_options(question, variant)},
    };

    // the nimble prompt lists all the questions of the request
    if (type == COMMON_DECISION_TYPE_NIMBLE) {
        inp["questions"] = json::array();
        for (const auto & q : questions) {
            inp["questions"].push_back(json{
                {"id",           q.id},
                {"type",         decision_question_type_name(q.type)},
                {"instructions", q.instructions},
                {"options",      render_options(q, 0)},
            });
        }
    }

    // lev was trained with sorted keys
    if (type == COMMON_DECISION_TYPE_LEV) {
        inp = decision_sort_keys(inp);
    }

    // the kev template only takes text
    if (type == COMMON_DECISION_TYPE_KEV) {
        inp["state"]        = decision_kev_text(state);
        inp["instructions"] = decision_kev_text(question.instructions);
    }

    // the OCR text must not be parsed as special tokens
    if (type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV) {
        inp["state"] = server_decision_escape_special_tokens(state.get<std::string>());
    }

    // the input must not contain the marker of the options
    if (!text_marker.empty()) {
        inp = decision_replace_text(inp, text_marker, " ");
    }

    // lfm3-d1-omni: a spoken question is heard where its text would be, after the media of the state
    const bool is_spoken = decision_is_spoken(question.instructions);
    if (is_spoken) {
        inp["instructions"] = "";
    }
    if (is_lfm3_d1()) {
        inp = decision_escape(inp, [this](const std::string & text) { return d1_break(text); });
    }

    if (type == COMMON_DECISION_TYPE_LFM2_D1_OMNI) {
        inp = decision_replace_text(decision_escape(inp, server_decision_escape_special_tokens), D1OMNI_MARKER, "<<d1omni ");
        inp["audio"]         = is_audio;
        inp["sep"]           = D1OMNI_SEP;
        inp["mark_state"]    = D1OMNI_MARK_STATE;
        inp["mark_question"] = D1OMNI_MARK_QUESTION;
        inp["mark_option"]   = D1OMNI_MARK_OPTION;
    }

    // the template puts one media marker per image
    json images = json::array();
    if (n_images > 0 || is_spoken) {
        inp = decision_replace_text(inp, get_media_marker(), " ");
        for (size_t i = 0; i < n_images; i++) {
            images.push_back(get_media_marker());
        }
    }
    inp["images"] = images;
    if (is_spoken) {
        inp["instructions"] = get_media_marker();
    }

    jinja::context ctx(tmpl->source());
    jinja::global_from_json(ctx, inp, false);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl->prog);
    return jinja::runtime::gather_string_parts(results)->as_string().str();
}

void server_decision_context::fill_task(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const server_decision_question & question,
        size_t variant,
        const std::vector<raw_buffer> & files,
        mtmd_context * mctx,
        const mtmd_helper_init_opt & init_opt,
        server_task & task) const {
    // each question has its own prompt, with the spans that the head reads
    if (type == COMMON_DECISION_TYPE_VLLM_DECISION2) {
        fill_task_joint(state, { question }, files, mctx, init_opt, task);
        return;
    }
    if (type == COMMON_DECISION_TYPE_LFM2_D1_OMNI) {
        fill_task_d1omni(state, questions, question, variant, files, mctx, init_opt, task);
        return;
    }

    const std::string prompt = render(state, questions, question, variant, files.size());

    if (type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV) {
        task.tokens          = process_mtmd_prompt(mctx, prompt, files, init_opt);
        task.decision.pooled = true;
        return;
    }

    std::vector<raw_buffer> media = files;
    if (decision_is_spoken(question.instructions)) {
        const json audio = json_value(question.instructions, "input_audio", json::object());
        decision_load_audio(audio.contains("data") ? audio.at("data") : json(), media);
    }

    if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_GRANITE_SWITCH || type == COMMON_DECISION_TYPE_LEV || type == COMMON_DECISION_TYPE_NIMBLE || type == COMMON_DECISION_TYPE_PPLX_DECIDER || type == COMMON_DECISION_TYPE_VLLM_D3 ||
        type == COMMON_DECISION_TYPE_H2O_LIGHTNING) {
        // lev reads the ratings of a noul question at its first labels, not at the digits
        task.decision.labels.assign(labels.begin(), labels.begin() + n_outputs(question));
    }
    if (type == COMMON_DECISION_TYPE_LFM2_D1 || type == COMMON_DECISION_TYPE_LFM3_D1 || type == COMMON_DECISION_TYPE_LFM3_D1_OMNI) {
        std::vector<std::string>  texts;
        std::vector<llama_tokens> groups;
        d1_labels(question, texts, groups);
        for (const auto & group : groups) {
            task.decision.labels.insert(task.decision.labels.end(), group.begin(), group.end());
            task.decision.label_groups.push_back(group.size());
        }
        // the short convs of lfm3 d1 have no memory to separate the sequences of a batch
        task.decision.no_batching = is_lfm3_d1();
    }
    if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_LEV || type == COMMON_DECISION_TYPE_NIMBLE || type == COMMON_DECISION_TYPE_PPLX_DECIDER || type == COMMON_DECISION_TYPE_VLLM_D3 || type == COMMON_DECISION_TYPE_LFM2_D1 ||
        type == COMMON_DECISION_TYPE_H2O_LIGHTNING || type == COMMON_DECISION_TYPE_LFM3_D1 || type == COMMON_DECISION_TYPE_LFM3_D1_OMNI) {
        if (!media.empty()) {
            // the d1 reference sees the frames of a video, not its soundtrack
            mtmd_helper_init_opt opt = init_opt;
            opt.video_params.no_soundtrack = is_lfm3_d1();
            task.tokens = process_mtmd_prompt(mctx, prompt, media, opt);
            return;
        }
    }

    llama_tokens tokens = common_tokenize(vocab, prompt, false, true);
    if (type == COMMON_DECISION_TYPE_LAYA) {
        fill_task_laya(tokens, question, task);
    }
    if (type == COMMON_DECISION_TYPE_KEV) {
        // an option is read at its end token, the question at the last token
        for (size_t i = 0; i < tokens.size(); i++) {
            if (tokens[i] == token_marker) {
                task.decision.markers.push_back(i);
            }
        }
        if (task.decision.markers.size() != question.options.size()) {
            throw std::runtime_error("unexpected layout of the decision prompt");
        }
        task.decision.pointer = tokens.size() - 1;
    }
    task.tokens = server_tokens(tokens, false);
}

// the prompt is: [cls] question [sep] ([marker] option)* [sep] state [sep]
// options and question are cut to fit max_head_tokens, the same way the model was trained
void server_decision_context::fill_task_laya(llama_tokens & tokens, const server_decision_question & question, server_task & task) const {
    const size_t n_options = question.options.size();

    std::vector<size_t> markers;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == token_marker) {
            markers.push_back(i);
        }
    }
    const auto invalid = std::runtime_error("unexpected layout of the decision prompt");
    if (markers.size() != n_options || markers[0] < 2 || tokens[markers[0] - 1] != token_sep || tokens.back() != token_sep) {
        throw invalid;
    }
    const size_t head_end = markers[0] - 1;
    const size_t opts_end = std::find(tokens.begin() + markers.back(), tokens.end(), token_sep) - tokens.begin();
    if (opts_end + 1 >= tokens.size()) {
        throw invalid;
    }

    // marker + text of each option
    std::vector<llama_tokens> options;
    size_t n_options_tokens = 0;
    auto set_max = [&](size_t n_max) {
        n_options_tokens = 0;
        for (auto & opt : options) {
            opt.resize(std::min(opt.size(), n_max));
            n_options_tokens += opt.size();
        }
    };
    for (size_t i = 0; i < n_options; i++) {
        const size_t end = i + 1 < n_options ? markers[i + 1] : opts_end;
        options.emplace_back(tokens.begin() + markers[i], tokens.begin() + end);
    }
    set_max(max_option_tokens + 1);
    if (n_options_tokens + 16 > max_head_tokens) {
        // too many or too long options, shrink them evenly
        set_max(std::max((size_t) 4, (max_head_tokens - std::min(max_head_tokens, (size_t) 16)) / n_options));
    }
    const size_t n_question_max = std::max((size_t) 8, max_head_tokens - std::min(max_head_tokens, n_options_tokens));

    llama_tokens out;
    out.push_back(tokens[0]);
    out.insert(out.end(), tokens.begin() + 1, tokens.begin() + std::min(head_end, 1 + n_question_max));
    out.push_back(token_sep);
    for (const auto & opt : options) {
        task.decision.markers.push_back(out.size());
        out.insert(out.end(), opt.begin(), opt.end());
    }
    out.insert(out.end(), tokens.begin() + opts_end, tokens.end());
    tokens = std::move(out);

    // the output has one score per question type
    task.decision.column = question.type;
}

// each piece is cut to its budget as the model was trained (d1-omni prompt.py: encode), the state gets the room that is left
// the media come first, the text after them has a budget of its own
void server_decision_context::fill_task_d1omni(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const server_decision_question & question,
        size_t variant,
        const std::vector<raw_buffer> & files,
        mtmd_context * mctx,
        const mtmd_helper_init_opt & init_opt,
        server_task & task) const {
    const auto invalid = std::runtime_error("unexpected layout of the decision prompt");

    // the prompt depends on the kind of media, mtmd tells an audio clip from an image by its content
    std::string markers;
    server_tokens media(llama_tokens(), false);
    bool is_audio = false;
    if (!files.empty()) {
        for (size_t i = 0; i < files.size(); i++) {
            markers += get_media_marker();
        }
        media = process_mtmd_prompt(mctx, markers, files, init_opt);
        for (size_t i = 0; i < media.size(); i++) {
            if (media[i] == LLAMA_TOKEN_NULL) {
                const auto & chunk = media.find_chunk(i);
                is_audio = is_audio || mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_AUDIO;
                i += mtmd_input_chunk_get_n_tokens(chunk.get()) - 1;
            }
        }
    }
    if (is_audio && files.size() > 1) {
        throw std::invalid_argument("a request has images or one audio clip, not both");
    }
    const size_t n_media = media.size();

    const std::string prompt = render(state, questions, question, variant, files.size(), is_audio);
    std::vector<std::string> pieces = string_split(prompt, D1OMNI_SEP);
    if (pieces.empty() || pieces[0] != markers) {
        throw invalid;
    }

    size_t n_max = D1OMNI_MAX_TOKENS;
    if (!files.empty()) {
        n_max = std::min(is_audio ? D1OMNI_MAX_TOKENS_AUDIO : D1OMNI_MAX_TOKENS_IMAGE, D1OMNI_MAX_TOKENS - std::min(D1OMNI_MAX_TOKENS, n_media));
        if (n_max < 64) {
            throw std::invalid_argument(string_format("the media take %zu of the %zu positions, send fewer images", n_media, D1OMNI_MAX_TOKENS));
        }
    }

    // the options get max(96, min(24 n + 32, max / 2)) tokens, shared evenly
    const int64_t n_options      = question.options.size();
    const int64_t n_budget       = std::max<int64_t>(96, std::min<int64_t>(24 * n_options + 32, n_max / 2));
    const int64_t n_option_max   = std::max<int64_t>(2, (n_budget - 3 * n_options) / n_options);
    const int64_t n_question_max = std::max<int64_t>(16, n_budget);

    llama_tokens head;  // before the state
    llama_tokens body;  // the state
    llama_tokens tail;  // after the state
    bool has_state = false;
    for (size_t i_piece = 1; i_piece < pieces.size(); i_piece++) {
        std::string piece = pieces[i_piece];
        int64_t n_piece_max = -1;
        bool is_state = false;
        if (string_starts_with(piece, D1OMNI_MARK_STATE)) {
            piece    = piece.substr(D1OMNI_MARK_STATE.size());
            is_state = true;
        } else if (string_starts_with(piece, D1OMNI_MARK_QUESTION)) {
            piece = piece.substr(D1OMNI_MARK_QUESTION.size());
            n_piece_max = n_question_max;
        } else if (string_starts_with(piece, D1OMNI_MARK_OPTION)) {
            piece = piece.substr(D1OMNI_MARK_OPTION.size());
            n_piece_max = n_option_max;
        }

        llama_tokens tokens = common_tokenize(vocab, piece, false, true);
        if (n_piece_max >= 0 && (int64_t) tokens.size() > n_piece_max) {
            tokens.resize(n_piece_max);
        }

        if (is_state) {
            if (has_state) {
                throw invalid;
            }
            body      = std::move(tokens);
            has_state = true;
        } else {
            llama_tokens & dst = has_state ? tail : head;
            dst.insert(dst.end(), tokens.begin(), tokens.end());
        }
    }
    if (!has_state) {
        throw invalid;
    }

    const size_t n_room = n_max - std::min(n_max, head.size() + tail.size());
    body.resize(std::min(body.size(), n_room));

    llama_tokens tokens = std::move(head);
    tokens.insert(tokens.end(), body.begin(), body.end());
    tokens.insert(tokens.end(), tail.begin(), tail.end());
    tokens.resize(std::min(tokens.size(), n_max));

    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == token_marker) {
            task.decision.markers.push_back(n_media + i);
        }
    }
    if ((int64_t) task.decision.markers.size() != n_options) {
        throw std::invalid_argument("the options do not fit in the context");
    }

    // the output has one score per question type
    task.decision.column = question.type;
    if (files.empty()) {
        task.tokens = server_tokens(tokens, false);
    } else {
        task.tokens = std::move(media);
        for (const llama_token token : tokens) {
            task.tokens.push_back(token);
        }
    }
}

//
// joint prompt (clef)
//

// given to the template: text between the pieces of the prompt, and at the start of the span of a question or of an option
static const std::string CLEF_MARKER        = "<<clef:";
static const std::string CLEF_SEP           = "<<clef:sep>>";
static const std::string CLEF_MARK_QUESTION = "<<clef:question>>";
static const std::string CLEF_MARK_OPTION   = "<<clef:option>>";

void server_decision_context::fill_task_joint(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const std::vector<raw_buffer> & files,
        mtmd_context * mctx,
        const mtmd_helper_init_opt & init_opt,
        server_task & task) const {
    json inp_questions = json::array();
    for (const auto & question : questions) {
        json options = json::array();
        for (const auto & opt : question.options) {
            options.push_back(json{
                {"key",         opt.key},
                {"description", opt.description},
            });
        }
        inp_questions.push_back(json{
            {"id",           question.id},
            {"type",         decision_question_type_name(question.type)},
            {"instructions", question.instructions},
            {"options",      options},
        });
    }

    // the template is given raw JSON values with sorted keys, and no marker in the input
    json inp = json{
        {"state",     state},
        {"questions", inp_questions},
    };
    inp = decision_replace_text(decision_sort_keys(inp), CLEF_MARKER, "<<clef ");

    // the template puts one media marker per image
    json images = json::array();
    if (!files.empty()) {
        inp = decision_replace_text(inp, get_media_marker(), " ");
        for (size_t i = 0; i < files.size(); i++) {
            images.push_back(get_media_marker());
        }
    }
    inp["images"]        = images;
    inp["sep"]           = CLEF_SEP;
    inp["mark_question"] = CLEF_MARK_QUESTION;
    inp["mark_option"]   = CLEF_MARK_OPTION;

    jinja::context ctx(tmpl->source());
    jinja::global_from_json(ctx, inp, false);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl->prog);
    const std::string prompt   = jinja::runtime::gather_string_parts(results)->as_string().str();

    const auto invalid = std::runtime_error("unexpected layout of the decision prompt");

    // the model was trained with the pieces tokenized one by one
    const std::vector<std::string> pieces = string_split(prompt, CLEF_SEP);
    size_t i_piece = 0;

    task.tokens = server_tokens(llama_tokens(), false);
    if (!files.empty()) {
        // the text before the images is tokenized with them, a vision token separates the pieces anyway
        std::string head;
        while (i_piece < pieces.size() && head.find(get_media_marker()) == std::string::npos) {
            head += pieces[i_piece++];
        }
        if (head.find(get_media_marker()) == std::string::npos || head.find(CLEF_MARKER) != std::string::npos) {
            throw invalid;
        }
        task.tokens = process_mtmd_prompt(mctx, head, files, init_opt);
        task.decision.order.resize(task.tokens.size(), LLAMA_DECISION_ORDER_NONE);
    }

    size_t i_question = 0;
    for (; i_piece < pieces.size(); i_piece++) {
        std::string piece = pieces[i_piece];
        int32_t order = LLAMA_DECISION_ORDER_NONE;
        if (string_starts_with(piece, CLEF_MARK_QUESTION)) {
            piece = piece.substr(CLEF_MARK_QUESTION.size());
            if (i_question >= questions.size()) {
                throw invalid;
            }
            switch (questions[i_question++].type) {
                case SERVER_DECISION_QUESTION_NOUL:   order = LLAMA_DECISION_ORDER_QUESTION_NOUL;   break;
                case SERVER_DECISION_QUESTION_CHOICE: order = LLAMA_DECISION_ORDER_QUESTION_CHOICE; break;
                case SERVER_DECISION_QUESTION_SCORE:  order = LLAMA_DECISION_ORDER_QUESTION_SCORE;  break;
            }
        } else if (string_starts_with(piece, CLEF_MARK_OPTION)) {
            piece = piece.substr(CLEF_MARK_OPTION.size());
            order = LLAMA_DECISION_ORDER_OPTION;
            task.decision.n_scores++;
        }

        const llama_tokens piece_tokens = common_tokenize(vocab, piece, false, true);
        if (order != LLAMA_DECISION_ORDER_NONE && piece_tokens.empty()) {
            throw std::invalid_argument("the instructions and the options of a question must not be empty");
        }
        for (const llama_token token : piece_tokens) {
            task.tokens.push_back(token);
        }
        task.decision.order.resize(task.tokens.size(), order);
    }

    size_t n_options = 0;
    for (const auto & question : questions) {
        n_options += question.options.size();
    }
    if (i_question != questions.size() || (size_t) task.decision.n_scores != n_options) {
        throw invalid;
    }
}

//
// answer
//

float server_decision_context::get_temperature(const server_decision_question & question) const {
    const size_t n = question.options.size();
    const std::string type_name = decision_question_type_name(question.type);

    // the temperature can depend on the number of options, the buckets are the ones used to fit it
    std::string bucket;
    if (type == COMMON_DECISION_TYPE_LEV) {
        bucket = n <= 8 ? "small" : n <= 26 ? "mid" : "large";
    } else {
        bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
    }

    for (const auto & name : {type_name + "." + bucket, type_name}) {
        const auto it = temperatures.find(name);
        if (it != temperatures.end()) {
            return it->second;
        }
    }
    return 1.0f;
}

// confidence formulas are the ones published by TypeSafe

static double decision_confidence_choice(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const double uniform = 1.0 / probs.size();
    const double p_max   = *std::max_element(probs.begin(), probs.end());
    return std::max(0.0, (p_max - uniform) / (1.0 - uniform));
}

static double decision_confidence_score(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const size_t n    = probs.size();
    const size_t mode = std::max_element(probs.begin(), probs.end()) - probs.begin();

    // mean distance to the mode, relative to the one of a uniform distribution around its center
    double dist         = 0.0;
    double dist_uniform = 0.0;
    for (size_t i = 0; i < n; i++) {
        dist         += probs[i] * std::fabs((double) i - (double) mode);
        dist_uniform += std::fabs((double) i - (n - 1) / 2.0) / n;
    }
    return std::max(0.0, 1.0 - dist / dist_uniform);
}

// VLLM_DECISION2, VLLM_D3: normalized entropy, as in the reference implementation
static double decision_confidence_entropy(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    double entropy = 0.0;
    for (const double p : probs) {
        if (p > 0.0) {
            entropy -= p * std::log(p);
        }
    }
    return std::max(0.0, std::min(1.0, 1.0 - entropy / std::log((double) probs.size())));
}

double server_decision_context::get_decision_confidence(server_decision_question_type question_type, const std::vector<double> & probs) const {
    if (type == COMMON_DECISION_TYPE_VLLM_DECISION2 || type == COMMON_DECISION_TYPE_VLLM_D3) {
        return decision_confidence_entropy(probs);
    }
    // lfm3-d1: the probability of the answer (d1 api.py)
    if (is_lfm3_d1()) {
        return *std::max_element(probs.begin(), probs.end());
    }
    switch (question_type) {
        case SERVER_DECISION_QUESTION_CHOICE: return decision_confidence_choice(probs);
        case SERVER_DECISION_QUESTION_SCORE:  return decision_confidence_score(probs);
        case SERVER_DECISION_QUESTION_NOUL:   break;
    }
    throw std::runtime_error("no confidence for this question type");
}

json server_decision_context::format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores, bool has_media) const {
    const size_t n = n_outputs(question);
    if (scores.size() != n_variants(question)) {
        throw std::runtime_error("decision result does not match the number of variants");
    }

    // softmax over the outputs of each variant, then the average of the variants
    // lfm2-d1-omni: image and audio answers are not calibrated
    float temperature = has_media && type == COMMON_DECISION_TYPE_LFM2_D1_OMNI ? 1.0f : get_temperature(question);
    // a model can have its own temperature for requests with media
    const auto it_image = temperatures.find("image");
    if (has_media && it_image != temperatures.end()) {
        temperature = it_image->second;
    }
    std::vector<double> probs(n, 0.0);
    for (size_t v = 0; v < scores.size(); v++) {
        const auto & s = scores[v];
        if (s.size() != n) {
            throw std::runtime_error("decision result does not match the number of options");
        }
        // a joint head returns NaN if it could not use the decision order, +inf would make the softmax NaN
        if (std::any_of(s.begin(), s.end(), [](float v) { return std::isnan(v) || v == INFINITY; })) {
            throw std::runtime_error("the model could not evaluate the decision");
        }
        std::vector<double> logits(s.begin(), s.end());
        if (question.type == SERVER_DECISION_QUESTION_SCORE && score_bias.count(n) > 0) {
            for (size_t i = 0; i < n; i++) {
                logits[i] += score_bias.at(n)[i];
            }
        }
        const double logit_max = *std::max_element(logits.begin(), logits.end());
        std::vector<double> p(n);
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            p[i] = std::exp((logits[i] - logit_max) / temperature);
            sum += p[i];
        }
        for (size_t i = 0; i < n; i++) {
            // the second variant is in the reverse order
            probs[v == 0 ? i : n - 1 - i] += p[i] / sum / scores.size();
        }
    }

    json answer = json{{"type", decision_question_type_name(question.type)}};

    if (question.type == SERVER_DECISION_QUESTION_NOUL) {
        if (type == COMMON_DECISION_TYPE_LEV) {
            double expected = 0.0;
            for (size_t i = 0; i < n; i++) {
                expected += probs[i] * i / (n - 1);
            }
            answer["noul"] = expected;
            return answer;
        }
        for (size_t i = 0; i < n; i++) {
            if (question.options[i].key == "true") {
                answer["noul"] = probs[i];
            }
        }
        return answer;
    }

    json probabilities = json::object();
    for (size_t i = 0; i < n; i++) {
        probabilities[question.options[i].key] = probs[i];
    }

    if (question.type == SERVER_DECISION_QUESTION_CHOICE) {
        const size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
        answer["choice"]        = question.options[best].key;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = get_decision_confidence(question.type, probs);
    } else {
        double expected = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < n; i++) {
            expected += i * probs[i];
            const json & description = question.options[i].description;
            legend[question.options[i].key] = canonical_legend && !description.is_string() ? json(decision_sort_keys(description).dump()) : description;
        }
        // lfm3-d1: the median level, the first whose cumulative probability reaches one half (d1 api.py)
        if (is_lfm3_d1()) {
            double mass = 0.0;
            size_t level = n - 1;
            for (size_t i = 0; i < n && mass < 0.5; i++) {
                mass += probs[i];
                level = i;
            }
            expected = (double) level;
        }
        answer["score"]         = expected;
        answer["legend"]        = legend;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = get_decision_confidence(question.type, probs);
    }
    return answer;
}

//
// shared prompt prefix
//

std::vector<server_task> server_decision_group_tasks(std::vector<server_task> && tasks, size_t n_slots) {
    n_slots = std::max(n_slots, (size_t) 1);

    std::vector<server_task> groups;
    for (size_t i = 0; i < tasks.size(); i += n_slots) {
        const size_t end = std::min(tasks.size(), i + n_slots);
        server_task & parent = tasks[i];

        // every task must have at least one token of its own to evaluate
        size_t n_shared = parent.tokens.size() - 1;
        for (size_t j = i + 1; j < end; j++) {
            n_shared = std::min(n_shared, parent.tokens.get_common_prefix(tasks[j].tokens));
            n_shared = std::min({n_shared, tasks[j].tokens.size() - 1, (size_t) parent.decision.pos_first(), (size_t) tasks[j].decision.pos_first()});
        }

        if (end - i < 2 || n_shared == 0) {
            for (size_t j = i; j < end; j++) {
                groups.push_back(std::move(tasks[j]));
            }
            continue;
        }

        parent.n_tokens_shared = n_shared;
        for (size_t j = i + 1; j < end; j++) {
            tasks[j].id_parent = parent.id;
            parent.child_tasks.push_back(std::move(tasks[j]));
        }
        groups.push_back(std::move(parent));
    }
    return groups;
}
