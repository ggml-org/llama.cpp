#include "server-decision.h"

#include "../../src/llama-ext.h" // staging API: llama_decision_order

#include <algorithm>
#include <cmath>
#include <regex>
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
    } else if (model_type == COMMON_DECISION_TYPE_LEV || model_type == COMMON_DECISION_TYPE_NIMBLE) {
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
    } else {
        throw std::runtime_error("unsupported decision model type: " + type_name);
    }
    type = model_type;

    SRV_INF("decision model type: %s\n", type_name.c_str());
}

//
// request parsing
//

std::vector<server_decision_question> server_decision_context::parse_questions(const json & body) const {
    if (!body.contains("state") || body.at("state").is_null()) {
        throw std::invalid_argument("\"state\" must be provided");
    }
    if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
        throw std::invalid_argument("\"questions\" must be a non-empty object");
    }

    std::vector<server_decision_question> questions;
    for (const auto & [id, q] : body.at("questions").items()) {
        auto err = [&id = id](const std::string & msg) {
            return std::invalid_argument("questions." + id + ": " + msg);
        };
        if (!q.is_object()) {
            throw err("must be an object");
        }
        if (!q.contains("instructions") || q.at("instructions").is_null()) {
            throw err("\"instructions\" must be provided");
        }

        server_decision_question question;
        question.id           = id;
        question.instructions = q.at("instructions");

        const std::string type_name = json_value(q, "type", std::string());
        const json        criteria  = q.contains("criteria") ? q.at("criteria") : json();

        if (type_name == "choice") {
            question.type = SERVER_DECISION_QUESTION_CHOICE;
            if (!criteria.is_object() || criteria.empty()) {
                throw err("\"criteria\" must be a non-empty object");
            }
            for (const auto & [key, description] : criteria.items()) {
                question.options.push_back({key, description});
            }
            if (choice_sorted) {
                std::sort(question.options.begin(), question.options.end(), [](const auto & a, const auto & b) {
                    return a.key < b.key;
                });
            }
        } else if (type_name == "score") {
            question.type = SERVER_DECISION_QUESTION_SCORE;
            if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10) {
                throw err("\"criteria\" must be an array of 2 to 10 levels");
            }
            for (size_t i = 0; i < criteria.size(); i++) {
                question.options.push_back({std::to_string(i), criteria.at(i)});
            }
        } else if (type_name == "noul") {
            question.type = SERVER_DECISION_QUESTION_NOUL;
            if (!criteria.is_null() && !criteria.is_object()) {
                throw err("\"criteria\" must be an object");
            }
            for (const char * key : {"false", "true"}) {
                question.options.push_back({key, criteria.is_object() && criteria.contains(key) ? criteria.at(key) : json()});
            }
            if (noul_true_first) {
                std::swap(question.options[0], question.options[1]);
            }
        } else {
            throw err("\"type\" must be one of: choice, score, noul");
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

static void decision_load_image(const json & url, std::vector<raw_buffer> & files) {
    if (!url.is_string() || !string_starts_with(url.get<std::string>(), "data:image/")) {
        throw std::invalid_argument("images must be data URLs (data:image/...;base64,...)");
    }
    if (files.size() >= DECISION_MAX_IMAGES) {
        throw std::invalid_argument(string_format("too many images, the maximum is %zu", DECISION_MAX_IMAGES));
    }
    handle_media(files, url.get<std::string>(), "");
}

json server_decision_context::parse_state(const json & body, std::vector<raw_buffer> & files) const {
    if (body.contains("videos") && !body.at("videos").is_null() && !body.at("videos").empty()) {
        throw std::invalid_argument("\"videos\" is not supported");
    }
    if (body.contains("images") && !body.at("images").is_null()) {
        if (!body.at("images").is_array()) {
            throw std::invalid_argument("\"images\" must be an array");
        }
        for (const auto & url : body.at("images")) {
            decision_load_image(url, files);
        }
    }

    const json & state = body.at("state");
    const bool is_wrapped = state.is_object() && state.contains("messages");
    const json & messages = is_wrapped ? state.at("messages") : state;
    if (!messages.is_array()) {
        return state;
    }

    // chat messages: take the image parts out of the content
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
static json decision_replace_text(const json & val, const std::string & search, const std::string & replace) {
    if (val.is_string()) {
        std::string str = val.get<std::string>();
        string_replace_all(str, search, replace);
        return str;
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_replace_text(item, search, replace));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[key] = decision_replace_text(item, search, replace);
        }
        return out;
    }
    return val;
}

// sort the keys of all objects of a JSON value
static json decision_sort_keys(const json & val) {
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_sort_keys(item));
        }
        return out;
    }
    if (val.is_object()) {
        std::map<std::string, json> sorted;
        for (const auto & [key, item] : val.items()) {
            sorted[key] = decision_sort_keys(item);
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
static std::string decision_kev_render(const json & val, int indent = 0) {
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
            const std::string text = decision_kev_render(item, indent + 1);
            out += (out.empty() ? "" : "\n") + pad + "- " + text.substr(std::min(text.size(), text.find_first_not_of(" \t\n\r")));
        }
        return out;
    }
    if (val.is_object()) {
        std::string out;
        for (const auto & [key, item] : val.items()) {
            const bool is_nested = item.is_object() || item.is_array();
            out += (out.empty() ? "" : "\n") + pad + key + (is_nested ? ":\n" : ": ") + decision_kev_render(item, is_nested ? indent + 1 : 0);
        }
        return out;
    }
    return val.dump();
}

// kev text input: special tokens written in the text must not be parsed as such
static std::string decision_kev_text(const json & val) {
    static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
    return std::regex_replace(decision_kev_render(val), re_special, "<\xC2\xA6$1\xC2\xA6>");
}

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

json server_decision_context::render_options(const server_decision_question & question, size_t variant) const {
    const size_t n_options = question.options.size();

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
        options.push_back(option);
    }
    return options;
}

std::string server_decision_context::render(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const server_decision_question & question,
        size_t variant,
        size_t n_images) const {
    // the template is given raw JSON values, it serializes the ones that are not strings
    json inp = json{
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

    // the input must not contain the marker of the options
    if (!text_marker.empty()) {
        inp = decision_replace_text(inp, text_marker, " ");
    }

    // the template puts one media marker per image
    json images = json::array();
    if (n_images > 0) {
        inp = decision_replace_text(inp, get_media_marker(), " ");
        for (size_t i = 0; i < n_images; i++) {
            images.push_back(get_media_marker());
        }
    }
    inp["images"] = images;

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
    const std::string prompt = render(state, questions, question, variant, files.size());

    if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_LEV || type == COMMON_DECISION_TYPE_NIMBLE) {
        // lev reads the ratings of a noul question at its first labels, not at the digits
        task.decision.labels.assign(labels.begin(), labels.begin() + n_outputs(question));
        if (!files.empty()) {
            task.tokens = process_mtmd_prompt(mctx, prompt, files, init_opt);
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

json server_decision_context::format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores) const {
    const size_t n = n_outputs(question);
    if (scores.size() != n_variants(question)) {
        throw std::runtime_error("decision result does not match the number of variants");
    }

    // softmax over the outputs of each variant, then the average of the variants
    const float temperature = get_temperature(question);
    std::vector<double> probs(n, 0.0);
    for (size_t v = 0; v < scores.size(); v++) {
        const auto & s = scores[v];
        if (s.size() != n) {
            throw std::runtime_error("decision result does not match the number of options");
        }
        // a joint head returns NaN if it could not use the decision order
        if (std::any_of(s.begin(), s.end(), [](float v) { return std::isnan(v); })) {
            throw std::runtime_error("the model could not evaluate the decision");
        }
        const float score_max = *std::max_element(s.begin(), s.end());
        std::vector<double> p(n);
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            p[i] = std::exp((double) (s[i] - score_max) / temperature);
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
        answer["confidence"]    = decision_confidence_choice(probs);
    } else {
        double expected = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < n; i++) {
            expected += i * probs[i];
            legend[question.options[i].key] = question.options[i].description;
        }
        answer["score"]         = expected;
        answer["legend"]        = legend;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = decision_confidence_score(probs);
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
            n_shared = std::min(n_shared, tasks[j].tokens.size() - 1);
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
