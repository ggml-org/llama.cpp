#include "llama-quant-recipe.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

static std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string to_lower(const std::string & s) {
    std::string r = s;
    for (auto & c : r) c = (char)std::tolower(c);
    return r;
}

static ggml_type parse_ggml_type(const std::string & s) {
    std::string upper = s;
    for (auto & c : upper) c = (char)std::toupper(c);

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const char * name = ggml_type_name((ggml_type)i);
        if (!name) continue;
        std::string n(name);
        for (auto & c : n) c = (char)std::toupper(c);
        if (upper == n) return (ggml_type)i;
    }
    return GGML_TYPE_COUNT;
}

static tensor_category parse_category(const std::string & s) {
    std::string u = s;
    for (auto & c : u) c = (char)std::toupper(c);
    if (u == "OUTPUT")           return tensor_category::OUTPUT;
    if (u == "TOKEN_EMBD")       return tensor_category::TOKEN_EMBD;
    if (u == "ATTENTION_V")      return tensor_category::ATTENTION_V;
    if (u == "ATTENTION_K")      return tensor_category::ATTENTION_K;
    if (u == "ATTENTION_Q")      return tensor_category::ATTENTION_Q;
    if (u == "ATTENTION_QKV")    return tensor_category::ATTENTION_QKV;
    if (u == "ATTENTION_KV_B")   return tensor_category::ATTENTION_KV_B;
    if (u == "ATTENTION_WV")     return tensor_category::ATTENTION_WV;
    if (u == "ATTENTION_OUTPUT") return tensor_category::ATTENTION_OUTPUT;
    if (u == "FFN_UP")           return tensor_category::FFN_UP;
    if (u == "FFN_GATE")         return tensor_category::FFN_GATE;
    if (u == "FFN_DOWN")         return tensor_category::FFN_DOWN;
    return tensor_category::OTHER;
}

// find the first comparison operator in s, return its position and length
// checks >=, <=, != (2-char) before >, <, = (1-char)
static bool find_operator(const std::string & s, size_t & pos, size_t & len, recipe_comparison & cmp) {
    // two-char operators first
    for (size_t i = 0; i + 1 < s.size(); i++) {
        if (s[i] == '>' && s[i+1] == '=') { pos = i; len = 2; cmp = recipe_comparison::GTE; return true; }
        if (s[i] == '<' && s[i+1] == '=') { pos = i; len = 2; cmp = recipe_comparison::LTE; return true; }
        if (s[i] == '!' && s[i+1] == '=') { pos = i; len = 2; cmp = recipe_comparison::NEG; return true; }
    }
    // single-char operators
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '=') { pos = i; len = 1; cmp = recipe_comparison::EQ;  return true; }
        if (s[i] == '<') { pos = i; len = 1; cmp = recipe_comparison::LT;  return true; }
        if (s[i] == '>') { pos = i; len = 1; cmp = recipe_comparison::GT;  return true; }
    }
    return false;
}

// parse a single condition token like "arch=falcon", "layer<1/16", "more_bits", "!has_imatrix"
static recipe_condition parse_condition(const std::string & raw) {
    std::string tok = trim(raw);
    recipe_condition cond;

    // bare keywords
    if (tok == "more_bits") {
        cond.type = recipe_condition_type::CATEGORY;
        cond.more_bits = true;
        return cond;
    }
    if (tok == "has_imatrix") {
        cond.type = recipe_condition_type::HAS_IMATRIX;
        cond.cmp  = recipe_comparison::EQ;
        return cond;
    }
    if (tok == "!has_imatrix") {
        cond.type = recipe_condition_type::HAS_IMATRIX;
        cond.cmp  = recipe_comparison::NEG;
        return cond;
    }

    // key<op>value
    size_t pos, len;
    if (!find_operator(tok, pos, len, cond.cmp)) {
        throw std::runtime_error("cannot parse condition: " + tok);
    }

    std::string key = to_lower(trim(tok.substr(0, pos)));
    std::string val = trim(tok.substr(pos + len));

    if (key == "layer" || key == "category") {
        cond.type = (key == "layer") ? recipe_condition_type::LAYER : recipe_condition_type::CATEGORY;
        if (to_lower(val) == "more_bits") {
            cond.more_bits = true;
        } else {
            size_t slash = val.find('/');
            if (slash == std::string::npos) {
                throw std::runtime_error(key + " requires fraction (e.g., " + key + "<1/8), got: " + val);
            }
            cond.frac_num = std::stoi(val.substr(0, slash));
            cond.frac_den = std::stoi(val.substr(slash + 1));
            if (cond.frac_den == 0) {
                throw std::runtime_error("zero denominator in: " + tok);
            }
        }
    } else if (key == "arch") {
        cond.type = recipe_condition_type::ARCH;
        cond.str_val = to_lower(val);
    } else if (key == "model_type") {
        cond.type = recipe_condition_type::MODEL_TYPE;
        cond.str_val = val;
    } else if (key == "index") {
        cond.type = recipe_condition_type::INDEX;
        cond.int_val = std::stoi(val);
    } else if (key == "n_expert") {
        cond.type = recipe_condition_type::N_EXPERT;
        cond.int_val = std::stoi(val);
    } else if (key == "n_gqa") {
        cond.type = recipe_condition_type::N_GQA;
        cond.int_val = std::stoi(val);
    } else {
        throw std::runtime_error("unknown condition key: " + key);
    }

    return cond;
}

static quant_recipe parse_stream(std::istream & in) {
    quant_recipe recipe;
    tensor_category current_cat = tensor_category::OTHER;
    bool in_block = false;
    std::string line;
    int line_num = 0;

    while (std::getline(in, line)) {
        line_num++;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        // "name ..."
        if (line.rfind("name", 0) == 0 && line.size() > 4 && (line[4] == ' ' || line[4] == '\t')) {
            recipe.name = trim(line.substr(4));
            continue;
        }

        // "default ..."
        if (line.rfind("default", 0) == 0 && line.size() > 7 && (line[7] == ' ' || line[7] == '\t')) {
            std::string type_str = trim(line.substr(7));
            recipe.default_type = parse_ggml_type(type_str);
            if (recipe.default_type == GGML_TYPE_COUNT) {
                throw std::runtime_error("line " + std::to_string(line_num) + ": unknown type: " + type_str);
            }
            continue;
        }

        // "[category]"
        if (line[0] == '[') {
            size_t close = line.find(']');
            if (close == std::string::npos) {
                throw std::runtime_error("line " + std::to_string(line_num) + ": missing ']'");
            }
            current_cat = parse_category(trim(line.substr(1, close - 1)));
            if (current_cat == tensor_category::OTHER) {
                throw std::runtime_error("line " + std::to_string(line_num) + ": unknown category");
            }
            in_block = true;
            continue;
        }

        if (!in_block) {
            throw std::runtime_error("line " + std::to_string(line_num) + ": rule outside of category block");
        }

        // rule: "conditions : type"
        size_t colon = line.find(':');
        if (colon == std::string::npos) {
            throw std::runtime_error("line " + std::to_string(line_num) + ": missing ':'");
        }

        std::string lhs = trim(line.substr(0, colon));
        std::string rhs = trim(line.substr(colon + 1));

        recipe_rule rule;
        rule.type = parse_ggml_type(rhs);
        if (rule.type == GGML_TYPE_COUNT) {
            throw std::runtime_error("line " + std::to_string(line_num) + ": unknown type: " + rhs);
        }

        // parse comma-separated conditions ("*" or empty = unconditional)
        if (!lhs.empty() && lhs != "*") {
            std::istringstream cs(lhs);
            std::string tok;
            while (std::getline(cs, tok, ',')) {
                tok = trim(tok);
                if (!tok.empty()) {
                    rule.conditions.push_back(parse_condition(tok));
                }
            }
        }

        recipe.categories[current_cat].push_back(rule);
    }

    return recipe;
}

quant_recipe recipe_parse_string(const std::string & content) {
    std::istringstream stream(content);
    return parse_stream(stream);
}

quant_recipe recipe_parse_file(const std::string & filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open recipe file: " + filepath);
    }
    return parse_stream(file);
}

