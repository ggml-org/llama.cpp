#include "server-decision.h"

#include "../../src/llama-ext.h" // staging API: llama_decision_order

#include <algorithm>
#include <cerrno>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

static const char * decision_question_type_name(server_decision_question_type type) {
    switch (type) {
        case SERVER_DECISION_QUESTION_CHOICE: return "choice";
        case SERVER_DECISION_QUESTION_SCORE:  return "score";
        case SERVER_DECISION_QUESTION_NOUL:   return "noul";
    }
    return "";
}

// lev reads noul from a rating scale: 0 = certainly no, 8 = certainly yes
static constexpr size_t DECISION_LEV_N_RATINGS = 9;

// Jev allows at most 255 options; a model can be narrower and init() derives that bound
static constexpr size_t DECISION_OPTIONS_MAX_API = 255;

// saturating add and multiply, defined below with the budgets; declared here for the head clip
size_t decision_saturating_add(size_t a, size_t b);
size_t decision_saturating_mul(size_t a, size_t b);

// attacker JSON is walked recursively, cap the depth so a nested body cannot overflow the stack
static constexpr size_t DECISION_JSON_MAX_DEPTH = 32;

// a request with more questions cannot be admitted anyway, refuse it during parse before it allocates
static constexpr size_t DECISION_QUESTIONS_MAX = 4096;

// total measured JSON one request may carry, enforced even when the token budget is unlimited
static constexpr size_t DECISION_JSON_MAX_BYTES = 8 * 1024 * 1024;

// content bytes below which the byte pre-flight always passes, so a small request reaches the
// token gate that names exact tokens. Rendering that much is cheap; the byte gate exists to skip
// expensive renders, and without the floor a tiny budget would answer small-budget refusals with
// a byte message where the token message belongs.
static constexpr size_t DECISION_BYTE_GATE_FLOOR_BYTES = 4096;

// iterative depth check with an explicit stack, so attacker nesting cannot overflow the call stack.
// runs on the parsed body before any dump() or recursive walk, which are only safe once depth is bounded.
// the tree is never mutated during the walk, so pointers into it stay valid.
void decision_check_json_depth(const json & val, const char * field) {
    std::vector<std::pair<const json *, size_t>> stack;
    stack.emplace_back(&val, 0);
    while (!stack.empty()) {
        const auto top = stack.back();
        stack.pop_back();
        if (top.second > DECISION_JSON_MAX_DEPTH) {
            throw server_invalid_request(string_format("\"%s\" is nested too deeply", field));
        }
        if (top.first->is_array()) {
            for (const auto & item : *top.first) {
                stack.emplace_back(&item, top.second + 1);
            }
        } else if (top.first->is_object()) {
            for (const auto & kv : top.first->items()) {
                stack.emplace_back(&kv.value(), top.second + 1);
            }
        }
    }
}

// depth check on the raw body text before json::parse, whose own recursion is unbounded.
// brackets inside strings do not count; an unbalanced close is left for the parser to refuse.
void decision_check_raw_json_depth(const std::string & body) {
    size_t depth = 0;
    bool in_string = false;
    for (size_t i = 0; i < body.size(); i++) {
        const char c = body[i];
        if (in_string) {
            if (c == '\\') {
                i++; // the escaped byte cannot close the string
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '[' || c == '{') {
            if (++depth > DECISION_JSON_MAX_DEPTH) {
                throw server_invalid_request("\"body\" is nested too deeply");
            }
        } else if (c == ']' || c == '}') {
            if (depth > 0) {
                depth--;
            }
        }
    }
}

// sum of string bytes without dumping the whole value, so one huge option cannot force a big
// allocation to learn it is huge. Under-estimates the dump size (no JSON overhead), which only
// misses toward the token gate. The tree is never mutated during the walk, so pointers into it stay valid.
size_t decision_json_content_bytes(const json & val) {
    size_t n = 0;
    std::vector<const json *> stack;
    stack.push_back(&val);
    while (!stack.empty()) {
        const json * node = stack.back();
        stack.pop_back();
        if (node->is_string()) {
            n = decision_saturating_add(n, node->get<std::string>().size());
        } else if (node->is_array()) {
            for (const auto & item : *node) {
                stack.push_back(&item);
            }
        } else if (node->is_object()) {
            for (const auto & kv : node->items()) {
                n = decision_saturating_add(n, kv.key().size());
                stack.push_back(&kv.value());
            }
        }
    }
    return n;
}

// LAYA reads the question and options in one max_head_tokens window and shortens the options to fit

static constexpr size_t DECISION_HEAD_OVERHEAD_TOKENS = 16;
static constexpr size_t DECISION_OPTION_MIN_TOKENS   = 4;

// the tokens of a head's window left for the options
static size_t decision_head_room(size_t max_head_tokens) {
    return max_head_tokens - std::min(max_head_tokens, DECISION_HEAD_OVERHEAD_TOKENS);
}

// clamp a wide count for the int32 fields of a task, so a huge prefix never wraps
static int32_t decision_clamp_int32(size_t v) {
    return v > (size_t) std::numeric_limits<int32_t>::max()
        ? std::numeric_limits<int32_t>::max()
        : (int32_t) v;
}

// one place for the cap>0 / cap==0 / derived rule shared by the queue and token budgets
static size_t decision_resolve_cap(int cap, size_t derived) {
    if (cap > 0) {
        return (size_t) cap;
    }
    if (cap == 0) {
        return 0;
    }
    return derived;
}

decision_head_clip decision_clip_head(const std::vector<size_t> & option_sizes, size_t head_end, size_t max_head_tokens, size_t max_option_tokens) {
    decision_head_clip clip;
    clip.n_option_max = decision_saturating_add(max_option_tokens, 1);
    for (const size_t size : option_sizes) {
        clip.n_options_full = decision_saturating_add(clip.n_options_full, size);
        clip.n_options_kept = decision_saturating_add(clip.n_options_kept, std::min(size, clip.n_option_max));
    }
    if (!option_sizes.empty() && clip.n_options_kept > decision_head_room(max_head_tokens)) {
        // too many or too long options, shrink them evenly
        clip.n_option_max   = std::max(DECISION_OPTION_MIN_TOKENS, decision_head_room(max_head_tokens) / option_sizes.size());
        clip.n_options_kept = 0;
        for (const size_t size : option_sizes) {
            clip.n_options_kept = decision_saturating_add(clip.n_options_kept, std::min(size, clip.n_option_max));
        }
    }
    clip.n_question_max  = std::max((size_t) 8, max_head_tokens - std::min(max_head_tokens, clip.n_options_kept));
    clip.n_question_full = head_end > 0 ? head_end - 1 : 0;
    clip.n_question_kept = head_end > 0 ? std::min(head_end, decision_saturating_add(1, clip.n_question_max)) - 1 : 0;
    clip.clipped_options  = clip.n_options_kept < clip.n_options_full;
    clip.clipped_question = head_end > decision_saturating_add(1, clip.n_question_max);
    return clip;
}

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
        if (!(temp > 0.0f) || !std::isfinite(temp)) {
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
        traits.n_options_max   = labels.size();
        traits.noul_true_first = true;
    } else if (model_type == COMMON_DECISION_TYPE_LEV || model_type == COMMON_DECISION_TYPE_NIMBLE || model_type == COMMON_DECISION_TYPE_PPLX_DECIDER) {
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
            if (toks.size() == 1 && labels.size() < DECISION_OPTIONS_MAX_API) {
                labels.push_back(toks[0]);
                traits.label_texts.push_back(code);
            }
        }
        traits.n_options_max = labels.size();
        if (traits.n_options_max == 0) {
            throw std::runtime_error("decision model has no single-token labels");
        }
    } else if (model_type == COMMON_DECISION_TYPE_KEV) {
        // the hidden state of an option is read at the token that ends it
        const auto toks = common_tokenize(vocab, "<|box_end|>", false, true);
        if (toks.size() != 1) {
            throw std::runtime_error("decision model has no <|box_end|> token");
        }
        token_marker  = toks[0];
        // the options are separated by the marker token instead of being named, so the tokenizer sets no limit on how many there are: each one is a position in the prompt
        traits.n_options_max = DECISION_OPTIONS_MAX_API;
    } else if (model_type == COMMON_DECISION_TYPE_LAYA) {
        token_marker = llama_vocab_mask(vocab);
        token_sep    = llama_vocab_sep(vocab);
        if (token_marker == LLAMA_TOKEN_NULL || token_sep == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("decision model has no mask or sep token");
        }
        traits.text_marker = common_token_to_piece(vocab, token_marker, true);

        const std::string val = decision_meta_str(model, prefix + "max_head_tokens");
        errno = 0;
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(val.c_str(), &end, 10);
        if (val.empty() || end == val.c_str() || *end != '\0' || errno != 0 || parsed == 0) {
            throw std::runtime_error("decision model has no valid max_head_tokens");
        }
        max_head_tokens = (size_t) parsed;
        if (max_head_tokens == 0) {
            throw std::runtime_error("decision model has no valid max_head_tokens");
        }
        // as kev, the options are marker-separated positions rather than named labels; tokenizer sets no limit on how many there are.
        traits.n_options_max = std::min(DECISION_OPTIONS_MAX_API,
                                        decision_head_room(max_head_tokens) / DECISION_OPTION_MIN_TOKENS);
        if (traits.n_options_max == 0) {
            throw std::runtime_error("the decision head cannot hold a single option");
        }
    } else if (model_type == COMMON_DECISION_TYPE_CLEF) {
        // the joint head reads the options of every question from the one prompt, so it has no window of its own to derive a bound from and the API ceiling is all it sets
        traits.n_options_max   = DECISION_OPTIONS_MAX_API;
        traits.noul_true_first = true;
        traits.choice_sorted   = true;
    } else {
        throw std::runtime_error("unsupported decision model type: " + type_name);
    }
    type = model_type;

    SRV_INF("decision model type: %s, at most %zu options per question\n", type_name.c_str(), traits.n_options_max);
}

//
// request parsing
//

// a criterion must carry content: Jev's EntryType minus the falsy values Jinja would branch on. note: the backing json reports size() 1 and empty() false for a string, so read the string out
static bool decision_instructions_valid(const json & val) {
    if (val.is_string()) {
        return !val.get<std::string>().empty();
    }
    return (val.is_object() || val.is_array()) && !val.empty();
}

std::vector<server_decision_question> decision_parse_questions(const json & body, const decision_model_traits & traits) {
    if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
        throw server_invalid_request("\"questions\" must be a non-empty object");
    }
    if (body.at("questions").size() > DECISION_QUESTIONS_MAX) {
        throw server_status_error(ERROR_TYPE_REQUEST_TOO_LARGE,
            string_format("The request has %zu questions, the maximum is %zu",
                body.at("questions").size(), DECISION_QUESTIONS_MAX));
    }

    std::vector<server_decision_question> questions;
    for (const auto & [id, q] : body.at("questions").items()) {
        if (id.empty()) {
            throw server_invalid_request("\"questions\" has an empty question id");
        }
        if (questions.size() >= DECISION_QUESTIONS_MAX) {
            throw server_status_error(ERROR_TYPE_REQUEST_TOO_LARGE,
                string_format("The request has more than %zu questions", DECISION_QUESTIONS_MAX));
        }
        auto err = [&id = id](const std::string & msg) {
            return server_invalid_request("questions." + id + ": " + msg);
        };
        if (!q.is_object()) {
            throw err("must be an object");
        }
        if (!q.contains("instructions") || !decision_instructions_valid(q.at("instructions"))) {
            throw err("\"instructions\" must be a non-empty string, object, or array");
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
            // refuse before copying, so a huge criteria cannot force an allocation to learn it is huge
            if (criteria.size() > traits.n_options_max) {
                throw err(string_format("\"criteria\" has %zu options, at most %zu are supported",
                                        criteria.size(),
                                        traits.n_options_max));
            }
            for (const auto & [key, description] : criteria.items()) {
                question.options.push_back({key, description});
            }
            if (traits.choice_sorted) {
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
            if (traits.noul_true_first) {
                std::swap(question.options[0], question.options[1]);
            }
        } else {
            throw err("\"type\" must be one of: choice, score, noul");
        }

        if (question.options.size() > traits.n_options_max) {
            // the bound is the tokenizer's or the API's depending on the type, so do not call it the model's
            throw err(string_format("\"criteria\" has %zu options, at most %zu are supported",
                                    question.options.size(),
                                    traits.n_options_max));
        }

        questions.push_back(std::move(question));
    }
    return questions;
}

server_decision_request server_decision_context::parse_request(const json & body) const {
    // before any dump() or recursive walk below, which are only safe once depth is bounded
    decision_check_json_depth(body);
    server_decision_request request;
    request.questions = decision_parse_questions(body, traits);
    request.state     = decision_parse_state(body, request.files);
    return request;
}

//
// images
//

static void decision_load_image(const json & url, std::vector<raw_buffer> & files) {
    if (!url.is_string() || !string_starts_with(url.get<std::string>(), "data:image/")) {
        throw server_invalid_request("\"images\" must hold data URLs (data:image/...;base64,...)");
    }
    if (files.size() >= DECISION_MAX_IMAGES) {
        throw server_invalid_request(string_format("\"images\" holds %zu, the maximum is %zu", files.size() + 1, DECISION_MAX_IMAGES));
    }
    try {
        handle_media(files, url.get<std::string>(), "");
    } catch (const std::bad_alloc &) {
        throw; // out of memory is a server fault, never a 422
    } catch (const std::invalid_argument & e) {
        // handle_media answers 400 on every route, but here the body is valid JSON with a bad value
        throw server_invalid_request(string_format("\"images\" entry %zu: %s", files.size() + 1, e.what()));
    } catch (const std::runtime_error & e) {
        // decode failures arrive as runtime_error (bad base64, unsupported input), still the caller's value
        throw server_invalid_request(string_format("\"images\" entry %zu: %s", files.size() + 1, e.what()));
    }
}

json decision_parse_state(const json & body, std::vector<raw_buffer> & files) {
    // checked here, not in decision_parse_questions: the field belongs to this half of the body, and body.at() below would raise the 400 that ex_wrapper maps a json error to
    if (!body.contains("state") || body.at("state").is_null()) {
        throw server_invalid_request("\"state\" must be provided");
    }
    // a field this route does not take is a 422, like every other field this half refuses
    // an empty array means no videos, anything else present is refused so "" is not silently ignored
    if (body.contains("videos") && !body.at("videos").is_null()) {
        const json & videos = body.at("videos");
        const bool empty_array = videos.is_array() && videos.empty();
        if (!empty_array) {
            throw server_invalid_request("\"videos\" is not supported");
        }
    }
    if (body.contains("images") && !body.at("images").is_null()) {
        if (!body.at("images").is_array()) {
            throw server_invalid_request("\"images\" must be an array");
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

static json decision_replace_text(const json & val, const std::string & search, const std::string & replace, size_t depth = 0) {
    if (depth > DECISION_JSON_MAX_DEPTH) {
        throw server_invalid_request("\"state\" is nested too deeply");
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

static json decision_sort_keys(const json & val, size_t depth = 0) {
    if (depth > DECISION_JSON_MAX_DEPTH) {
        throw server_invalid_request("\"state\" is nested too deeply");
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
static std::string decision_kev_render(const json & val, int indent = 0) {
    if (indent < 0 || (size_t) indent > DECISION_JSON_MAX_DEPTH) {
        throw server_invalid_request("\"state\" is nested too deeply");
    }
    const std::string pad(2 * (size_t) indent, ' ');
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
static bool decision_kev_name_char(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

static std::string decision_kev_text(const json & val) {
    const std::string rendered = decision_kev_render(val);
    std::string out;
    out.reserve(rendered.size());
    for (size_t i = 0; i < rendered.size();) {
        if (i + 3 < rendered.size() && rendered[i] == '<' && rendered[i + 1] == '|') {
            size_t j = i + 2;
            while (j < rendered.size() && decision_kev_name_char(rendered[j])) {
                j++;
            }
            if (j + 1 < rendered.size() && rendered[j] == '|' && rendered[j + 1] == '>' && j > i + 2) {
                out += "<\xC2\xA6";
                out.append(rendered, i + 2, j - (i + 2));
                out += "\xC2\xA6>";
                i = j + 2;
                continue;
            }
        }
        out += rendered[i++];
    }
    return out;
}

size_t server_decision_context::n_variants(const server_decision_question & question) const {
    // lev shows the options of a choice in 2 orders, to cancel the preference for the first label
    if (type == COMMON_DECISION_TYPE_LEV && question.type == SERVER_DECISION_QUESTION_CHOICE &&
        question.options.size() > 1) {
        return 2;
    }
    return 1;
}

size_t server_decision_context::n_outputs(const server_decision_question & question) const {
    // lev reads a noul on its ratings, every other question reads one output per option
    if (type == COMMON_DECISION_TYPE_LEV && question.type == SERVER_DECISION_QUESTION_NOUL) {
        return DECISION_LEV_N_RATINGS;
    }
    return question.options.size();
}

json decision_template_options(common_decision_type             type,
                               const decision_model_traits &    traits,
                               const server_decision_question & question,
                               size_t                           variant) {
    if (variant > 1) {
        throw std::runtime_error("invalid decision variant");
    }
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
        if (!traits.label_texts.empty()) {
            option["label"] = traits.label_texts[i];
        }
        options.push_back(option);
    }
    return options;
}

std::string server_decision_context::render(const json & state,
                                            const std::vector<server_decision_question> & questions,
                                            const server_decision_question & question,
                                            size_t variant,
                                            size_t n_images) const {
    return decision_render_template(
        *tmpl, decision_template_input({ type, traits, state, questions, question, variant, n_images }));
}

std::string decision_render_template(const common_chat_template & tmpl, const json & inp) {
    jinja::context ctx(tmpl.src);
    jinja::global_from_json(ctx, inp, false);
    jinja::runtime     runtime(ctx);
    const jinja::value results = runtime.execute(tmpl.prog);
    return jinja::runtime::gather_string_parts(results)->as_string().str();
}

json decision_template_input(const decision_prompt & prompt) {
    // the template is given raw JSON values, it serializes the ones that are not strings
    json inp = json{
        {"id",           prompt.question.id},
        {"type",         decision_question_type_name(prompt.question.type)},
        {"instructions", prompt.question.instructions},
        {"state",        prompt.state},
        {"options",      decision_template_options(prompt.type, prompt.traits, prompt.question, prompt.variant)},
    };

    // the nimble prompt lists all the questions of the request
    if (prompt.type == COMMON_DECISION_TYPE_NIMBLE) {
        inp["questions"] = json::array();
        for (const auto & q : prompt.questions) {
            inp["questions"].push_back(json{
                {"id",           q.id},
                {"type",         decision_question_type_name(q.type)},
                {"instructions", q.instructions},
                {"options",      decision_template_options(prompt.type, prompt.traits, q, 0)},
            });
        }
    }

    // lev was trained with sorted keys
    if (prompt.type == COMMON_DECISION_TYPE_LEV) {
        inp = decision_sort_keys(inp);
    }

    // the kev template only takes text
    if (prompt.type == COMMON_DECISION_TYPE_KEV) {
        inp["state"]        = decision_kev_text(prompt.state);
        inp["instructions"] = decision_kev_text(prompt.question.instructions);
    }

    // the input must not contain the marker of the options
    if (!prompt.traits.text_marker.empty()) {
        inp = decision_replace_text(inp, prompt.traits.text_marker, " ");
    }

    // the template puts one media marker per image
    json images = json::array();
    if (prompt.n_images > 0) {
        inp = decision_replace_text(inp, get_media_marker(), " ");
        for (size_t i = 0; i < prompt.n_images; i++) {
            images.push_back(get_media_marker());
        }
    }
    inp["images"] = images;

    return inp;
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

    if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_LEV || type == COMMON_DECISION_TYPE_NIMBLE || type == COMMON_DECISION_TYPE_PPLX_DECIDER) {
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

// the prompt is: [cls] question [sep] ([marker] option)* [sep] state [sep]. options and question are cut to fit max_head_tokens, the same way the model was trained
void server_decision_context::fill_task_laya(llama_tokens & tokens, const server_decision_question & question, server_task & task) const {
    const size_t n_options = question.options.size();

    std::vector<size_t> markers;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == token_marker) {
            markers.push_back(i);
        }
    }
    const auto invalid = std::runtime_error("unexpected layout of the decision prompt");
    if (markers.size() != n_options) {
        throw invalid;
    }
    if (tokens.empty() || markers[0] < 2 || tokens[markers[0] - 1] != token_sep || tokens.back() != token_sep) {
        throw invalid;
    }
    const size_t head_end = markers[0] - 1;
    const size_t opts_end = std::find(tokens.begin() + markers.back(), tokens.end(), token_sep) - tokens.begin();
    if (opts_end + 1 >= tokens.size()) {
        throw invalid;
    }

    // marker + text of each option, sized for the clip below
    std::vector<llama_tokens> options;
    for (size_t i = 0; i < n_options; i++) {
        const size_t end = i + 1 < n_options ? markers[i + 1] : opts_end;
        options.emplace_back(tokens.begin() + markers[i], tokens.begin() + end);
    }
    std::vector<size_t> option_sizes;
    for (const auto & opt : options) {
        option_sizes.push_back(opt.size());
    }
    // one cap both clippings agree on, so the warning below reports exactly what is kept
    const decision_head_clip clip = decision_clip_head(option_sizes, head_end, max_head_tokens, max_option_tokens);
    for (auto & opt : options) {
        opt.resize(std::min(opt.size(), clip.n_option_max));
    }
    const size_t n_question_max = clip.n_question_max;

    // the head window no longer holds what the caller sent: warn with the budget and what was dropped
    if (clip.clipped_options || clip.clipped_question) {
        SRV_WRN("the model's head reads %zu tokens, the %zu options were shortened from %zu to %zu tokens, "
                "the question to %zu of %zu\n",
                max_head_tokens, n_options, clip.n_options_full, clip.n_options_kept,
                clip.n_question_kept, clip.n_question_full);
    }

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
        inp_questions.push_back(json{
            {"id",           question.id},
            {"type",         decision_question_type_name(question.type)},
            {"instructions", question.instructions},
            {"options",      decision_template_options(type, traits, question, 0)},
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

    const std::string prompt = decision_render_template(*tmpl, inp);

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
            throw server_invalid_request("questions." + questions[i_question - 1].id +
                                         ": the instructions and the options of a question must not be empty");
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

server_decision_tasks server_decision_context::build_tasks(
        const server_decision_request & request,
        const std::function<int()> &    next_id,
        mtmd_context *                  mctx,
        const mtmd_helper_init_opt &    init_opt,
        size_t                          n_slots,
        size_t                          max_tasks,
        size_t                          max_prompt_tokens) const {
    std::vector<server_task> tasks;

    // cheap pre-flight: the count needs no render, so an over-count request is refused before Jinja or the tokenizer run. Same kind as the token budget, but its own message: the token budget names rendered tokens, this one names tasks.
    const size_t n_tasks = count_tasks(request);
    if (max_tasks != 0 && n_tasks > max_tasks) {
        throw server_status_error(
            ERROR_TYPE_REQUEST_TOO_LARGE,
            string_format("The request builds %zu decision tasks, the maximum is %zu. "
                          "Give fewer \"questions\", or raise the limit with "
                          "--decision-max-prompt-tokens",
                          n_tasks, max_tasks));
    }
    // byte pre-flight before Jinja: one huge option must not force a render to learn it is huge.
    // measured without building a dump string, so the check itself cannot allocate.
    {
        size_t n_bytes = decision_json_content_bytes(request.state);
        for (const auto & question : request.questions) {
            n_bytes = decision_saturating_add(n_bytes, question.id.size());
            n_bytes = decision_saturating_add(n_bytes, decision_json_content_bytes(question.instructions));
            for (const auto & opt : question.options) {
                n_bytes = decision_saturating_add(n_bytes, opt.key.size());
                if (opt.description.is_string()) {
                    n_bytes = decision_saturating_add(n_bytes, opt.description.get<std::string>().size());
                } else {
                    n_bytes = decision_saturating_add(n_bytes, decision_json_content_bytes(opt.description));
                }
            }
        }
        // always on, so an unlimited token budget cannot be forced to render unbounded input
        if (n_bytes > DECISION_JSON_MAX_BYTES) {
            throw server_status_error(
                ERROR_TYPE_REQUEST_TOO_LARGE,
                string_format("The request is %zu bytes, over the %zu byte maximum. "
                              "Give a shorter \"state\" or fewer questions",
                              n_bytes, DECISION_JSON_MAX_BYTES));
        }
        // 32 bytes per token bounds JSON overhead, so a request over it cannot fit the token budget.
        // the floor keeps small requests on the token gate that names exact tokens; the message names
        // the threshold that actually applied, not just the budget it derives from.
        const size_t byte_threshold = std::max(DECISION_BYTE_GATE_FLOOR_BYTES, decision_saturating_mul(max_prompt_tokens, 32));
        if (max_prompt_tokens != 0 && n_bytes > byte_threshold) {
            throw server_status_error(
                ERROR_TYPE_REQUEST_TOO_LARGE,
                string_format("The request is %zu bytes, over the %zu byte threshold for the %zu token budget. "
                              "Give a shorter \"state\", fewer questions, or raise the limit with "
                              "--decision-max-prompt-tokens",
                              n_bytes, byte_threshold, max_prompt_tokens));
        }
    }
    // an empty prompt would be answered on the completion result path, which the route cannot cast
    const auto refuse_empty = [](const server_task & task, const std::string & field) {
        if (task.tokens.empty()) {
            throw server_invalid_request(field +
                                         ": the rendered prompt is empty; "
                                         "provide a non-empty \"state\" or non-empty \"instructions\"");
        }
    };

    // count every task before grouping, so the throw lands at most one task past the budget. This is the held work, not the billed work: the envelope bills evaluated tokens after discounting the shared prefix.
    size_t n_prompt_tokens = 0;
    const auto count_prompt = [&](const server_task & task) {
        n_prompt_tokens = decision_saturating_add(n_prompt_tokens, task.n_tokens() < 0 ? 0 : (size_t) task.n_tokens());
        if (max_prompt_tokens != 0 && n_prompt_tokens > max_prompt_tokens) {
            throw server_status_error(
                ERROR_TYPE_REQUEST_TOO_LARGE,
                string_format("The request renders %zu prompt tokens, the maximum is %zu. "
                              "Give a shorter \"state\", fewer questions, or raise the limit with "
                              "--decision-max-prompt-tokens",
                              n_prompt_tokens, max_prompt_tokens));
        }
    };

    if (is_joint()) {
        server_task task = server_task(SERVER_TASK_TYPE_DECISION);
        task.id          = next_id();
        fill_task_joint(request.state, request.questions, request.files, mctx, init_opt, task);
        refuse_empty(task, "questions");
        count_prompt(task);
        tasks.push_back(std::move(task));
        return { std::move(tasks), 0 };
    }

    for (const auto & question : request.questions) {
        for (size_t variant = 0; variant < n_variants(question); variant++) {
            server_task task = server_task(SERVER_TASK_TYPE_DECISION);
            task.id          = next_id();
            fill_task(request.state, request.questions, question, variant, request.files, mctx, init_opt, task);
            refuse_empty(task, "questions." + question.id);
            count_prompt(task);
            tasks.push_back(std::move(task));
        }
    }

    // a model that cannot share keeps n_shared at 0, so the route can subtract it unconditionally
    return can_share_prompt() ? server_decision_group_tasks(std::move(tasks), n_slots)
                              : server_decision_tasks{ std::move(tasks), 0 };
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

// confidence is only returned, never gated on, and only for a choice or score where there is a distribution to summarize, not noul.

// the choice formula TypeSafe publishes: how far p_max sits above uniform, in [0, 1]
static double decision_confidence_choice(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const double uniform = 1.0 / probs.size();
    const double p_max   = *std::max_element(probs.begin(), probs.end());
    // p_max <= 1, so only the lower clamp can fire, and only for a uniform distribution
    return std::max(0.0, (p_max - uniform) / (1.0 - uniform));
}

// the score rule: mean distance to the most likely level, relative to a uniform distribution. TypeSafe publishes it for 2 and 3 levels only
static double decision_confidence_score(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const size_t n    = probs.size();
    const size_t mode = std::max_element(probs.begin(), probs.end()) - probs.begin();

    double dist         = 0.0;
    double dist_uniform = 0.0;
    for (size_t i = 0; i < n; i++) {
        dist         += probs[i] * std::fabs((double) i - (double) mode);
        dist_uniform += std::fabs((double) i - (n - 1) / 2.0) / n;
    }
    // mass at the far end puts dist above dist_uniform, so both clamps can fire
    return std::min(1.0, std::max(0.0, 1.0 - dist / dist_uniform));
}

std::vector<double> decision_average_variants(const std::vector<std::vector<float>> & scores,
                                              size_t                                  n,
                                              float                                   temperature) {
    if (n == 0) {
        throw std::runtime_error("decision result does not match the number of options");
    }
    if (!(temperature > 0.0f) || !std::isfinite(temperature)) {
        throw std::runtime_error("invalid decision temperature");
    }
    // variant 0 is forward, the rest are reversed; production uses at most 2, tests pin the sum for more
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
            probs[v == 0 ? i : n - 1 - i] += p[i] / sum / scores.size();
        }
    }
    return probs;
}

std::vector<float> decision_joint_scores(const std::vector<float> &         scores,
                                         size_t                           offset,
                                         const server_decision_question & question) {
    const size_t n = question.options.size();
    if (offset + n > scores.size()) {
        throw std::runtime_error("the joint head returned fewer scores than the request has options");
    }
    return { scores.begin() + offset, scores.begin() + offset + n };
}

json server_decision_context::format_answer(const server_decision_question &        question,
                                            const std::vector<std::vector<float>> & scores) const {
    const size_t n = n_outputs(question);
    if (scores.size() != n_variants(question)) {
        throw std::runtime_error("decision result does not match the number of variants");
    }

    const std::vector<double> probs = decision_average_variants(scores, n, get_temperature(question));

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

// a group takes one slot per task, so it is bounded by the slots the server has, and a server with none still runs one task
static size_t decision_group_size(size_t n_tasks, size_t n_slots) {
    return std::min(n_tasks, std::max(n_slots, (size_t) 1));
}

size_t decision_queue_cap(int cap, int n_parallel) {
    const size_t derived = decision_saturating_mul(
        (size_t) common_params::COMMON_DECISION_QUEUE_CAP_PER_SLOT, (size_t) std::max(n_parallel, 1));
    return decision_resolve_cap(cap, derived); // 0 stays unlimited for try_post()
}

// saturating arithmetic: a wrapped budget is a small number where a large one was meant, which refuses every request instead of admitting the ones the operator sized the server for
size_t decision_saturating_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return std::numeric_limits<size_t>::max();
    }
    return a + b;
}

size_t decision_saturating_mul(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return std::numeric_limits<size_t>::max();
    }
    return a * b;
}

size_t decision_usage_input_tokens(size_t n_tokens_sum, size_t n_shared) {
    return n_shared > n_tokens_sum ? 0 : n_tokens_sum - n_shared;
}

size_t decision_prompt_budget(int cap, int n_parallel, int n_ctx_slot) {
    // a server with no slot, or a slot with no context, still gets one of each, so the derivation never lands on 0 and turns the gate off by accident
    const size_t slots  = (size_t) std::max(n_parallel, 1);
    const size_t ctx    = (size_t) std::max(n_ctx_slot, 1);
    size_t       budget = decision_saturating_mul((size_t) common_params::COMMON_DECISION_QUEUE_CAP_PER_SLOT, slots);
    budget = decision_saturating_mul(budget, ctx);
    return decision_resolve_cap(cap, budget); // 0 stays unlimited for build_tasks()
}

size_t decision_task_budget(int cap, int n_parallel, int n_ctx_slot) {
    if (cap == 0) {
        return 0; // unlimited, like the token budget
    }
    // one task per token of one slot context: the derived prompt budget in tasks. An explicit token budget only ever raises this, never lowers it into the range where tiny budgets would trip the count before the token gate.
    const size_t slots = (size_t) std::max(n_parallel, 1);
    const size_t ctx   = (size_t) std::max(n_ctx_slot, 1);
    size_t       bound = decision_saturating_mul(slots, ctx);
    if (cap > 0) {
        bound = std::max(bound, (size_t) cap / common_params::COMMON_DECISION_QUEUE_CAP_PER_SLOT);
    }
    return bound;
}

size_t server_decision_context::count_tasks(const server_decision_request & request) const {
    if (is_joint()) {
        return 1;
    }
    size_t n = 0;
    for (const auto & question : request.questions) {
        n = decision_saturating_add(n, n_variants(question));
    }
    return n;
}

server_decision_tasks server_decision_group_tasks(std::vector<server_task> && tasks, size_t n_slots) {
    const size_t n_per_group = decision_group_size(tasks.size(), n_slots);

    server_decision_tasks grouped;
    grouped.tasks.reserve(tasks.size());
    size_t n_shared_wide = 0;
    for (size_t i = 0; i < tasks.size(); i += n_per_group) {
        const size_t  end    = std::min(tasks.size(), i + n_per_group);
        server_task & parent = tasks[i];

        // an empty prompt has no prefix to share
        size_t n_shared = parent.tokens.size() == 0 ? 0 : parent.tokens.size() - 1;
        for (size_t j = i + 1; j < end; j++) {
            n_shared = std::min(n_shared, parent.tokens.get_common_prefix(tasks[j].tokens));
            if (tasks[j].tokens.size() > 0) {
                n_shared = std::min(n_shared, tasks[j].tokens.size() - 1);
            } else {
                n_shared = 0;
            }
        }

        if (end - i < 2 || n_shared == 0) {
            for (size_t j = i; j < end; j++) {
                grouped.tasks.push_back(std::move(tasks[j]));
            }
            continue;
        }

        parent.n_tokens_shared = decision_clamp_int32(n_shared);
        // this is the only place the prefix is discounted: the parent evaluates it once, so the caller subtracts it once per child from the sum of the prompt lengths
        n_shared_wide = decision_saturating_add(n_shared_wide, decision_saturating_mul(n_shared, end - i - 1));
        for (size_t j = i + 1; j < end; j++) {
            tasks[j].id_parent = parent.id;
            parent.child_tasks.push_back(std::move(tasks[j]));
        }
        grouped.tasks.push_back(std::move(parent));
    }
    grouped.n_shared = decision_clamp_int32(n_shared_wide);
    return grouped;
}
