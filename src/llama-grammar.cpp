#include "llama-grammar.h"

#include "llama-impl.h"
#include "llama-vocab.h"
#include "llama-sampler.h"

#include <cmath>
#include <algorithm>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <set>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <mutex>

#define MAX_REPETITION_THRESHOLD 2000
#define GRAMMAR_RECURSIVE_LIMIT 256
//
// helpers
//

// NOTE: assumes valid utf8 (but checks for overrun)
static std::pair<uint32_t, const char *> decode_utf8(const char * src) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t  first_byte = static_cast<uint8_t>(*src);
    uint8_t  highbits   = first_byte >> 4;
    int      len        = lookup[highbits];
    uint8_t  mask       = (1 << (8 - len)) - 1;
    uint32_t value      = first_byte & mask;
    const char * end    = src + len; // may overrun!
    const char * pos    = src + 1;
    for ( ; pos < end && *pos; pos++) {
        value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
    }
    return std::make_pair(value, pos);
}

static llama_partial_utf8 decode_utf8(
        const std::string & src,
        llama_partial_utf8 partial_start,
        std::vector<uint32_t> & code_points) {
    static const int      lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    const char          * pos      = src.c_str();
    const size_t start = code_points.size();
    uint32_t value    = partial_start.value;
    int      n_remain = partial_start.n_remain;

    // continue previous decode, if applicable
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            // invalid sequence, abort
            code_points.push_back(0);
            return { 0, -1 };
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }

    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // decode any subsequent utf-8 sequences, which may end in an incomplete one
    while (*pos != 0) {
        uint8_t first_byte = static_cast<uint8_t>(*pos);
        uint8_t highbits   = first_byte >> 4;
        n_remain   = lookup[highbits] - 1;

        if (n_remain < 0) {
            // invalid sequence, abort
            code_points.resize(start);
            code_points.push_back(0);
            return { 0, n_remain };
        }

        uint8_t mask  = (1 << (7 - n_remain)) - 1;
        value = first_byte & mask;

        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }
    code_points.push_back(0);

    return { value, n_remain };
}

llama_partial_utf8 llama_grammar_decode_utf8(
        const std::string & src,
        llama_partial_utf8 partial_start,
        std::vector<uint32_t> & code_points) {
    return decode_utf8(src, partial_start, code_points);
}

void llama_grammar::candidate_storage::clear() {
    candidates.clear();
    offsets.clear();
    code_points.clear();
}

void llama_grammar::candidate_storage::add(size_t index, llama_token id, const std::string & piece, llama_partial_utf8 partial_utf8) {
    offsets.push_back(code_points.size());
    const auto partial = decode_utf8(piece, partial_utf8, code_points);
    candidates.push_back({ index, nullptr, partial, id });
}

void llama_grammar::candidate_storage::set_pointers() {
    for (size_t i = 0; i < candidates.size(); ++i) {
        candidates[i].code_points = code_points.data() + offsets[i];
    }
}

static std::shared_ptr<const llama_grammar::rule_position_map> llama_grammar_index_rules(const llama_grammar_rules & rules) {
    auto positions = std::make_shared<llama_grammar::rule_position_map>();
    for (size_t ir = 0; ir < rules.size(); ++ir) {
        for (size_t ie = 0; ie < rules[ir].size(); ++ie) {
            positions->emplace(&rules[ir][ie], std::make_pair(ir, ie));
        }
    }
    return positions;
}

static bool is_digit_char(char c) {
    return '0' <= c && c <= '9';
}

static bool is_word_char(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || is_digit_char(c);
}

static std::pair<uint32_t, const char *> parse_hex(const char * src, int size) {
    const char * pos   = src;
    const char * end   = src + size;
    uint32_t     value = 0;
    for ( ; pos < end && *pos; pos++) {
        value <<= 4;
        char c = *pos;
        if ('a' <= c && c <= 'f') {
            value += c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            value += c - 'A' + 10;
        } else if ('0' <= c && c <= '9') {
            value += c - '0';
        } else {
            break;
        }
    }
    if (pos != end) {
        throw std::runtime_error("expecting " + std::to_string(size) + " hex chars at " + src);
    }
    return std::make_pair(value, pos);
}

static const char * parse_space(const char * src, bool newline_ok) {
    const char * pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
            (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') {
            while (*pos && *pos != '\r' && *pos != '\n') {
                pos++;
            }
        } else {
            pos++;
        }
    }
    return pos;
}

static const char * parse_name(const char * src) {
    const char * pos = src;
    while (is_word_char(*pos)) {
        pos++;
    }
    if (pos == src) {
        throw std::runtime_error(std::string("expecting name at ") + src);
    }
    return pos;
}

static const char * parse_int(const char * src) {
    const char * pos = src;
    while (is_digit_char(*pos)) {
        pos++;
    }
    if (pos == src) {
        throw std::runtime_error(std::string("expecting integer at ") + src);
    }
    return pos;
}

static std::pair<uint32_t, const char *> parse_char(const char * src) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2);
            case 'u': return parse_hex(src + 2, 4);
            case 'U': return parse_hex(src + 2, 8);
            case 't': return std::make_pair('\t', src + 2);
            case 'r': return std::make_pair('\r', src + 2);
            case 'n': return std::make_pair('\n', src + 2);
            case '\\':
            case '"':
            case '[':
            case ']':
            case '-':
                      return std::make_pair(src[1], src + 2);
            default:
                      throw std::runtime_error(std::string("unknown escape at ") + src);
        }
    } else if (*src) {
        return decode_utf8(src);
    }
    throw std::runtime_error("unexpected end of input");
}

static std::pair<uint32_t, const char *> parse_token(const llama_vocab * vocab, const char * src) {
    const char * pos = src;
    if (*pos != '<') {
        throw std::runtime_error(std::string("expecting '<' at ") + pos);
    }
    pos++;

    // Parse <[id]>
    if (*pos == '[') {
        pos++;
        const char * int_end = parse_int(pos);
        unsigned long id = std::stoul(std::string(pos, int_end - pos));
        if (id > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(std::string("parsed token id is too big at ") + pos);
        }
        uint32_t token_id = static_cast<uint32_t>(id);
        pos = int_end;
        if (*pos != ']') {
            throw std::runtime_error(std::string("expecting ']' at ") + pos);
        }
        pos++;
        if (*pos != '>') {
            throw std::runtime_error(std::string("expecting '>' at ") + pos);
        }
        pos++;
        return std::make_pair(token_id, pos);
    }

    if (vocab == nullptr) {
        throw std::runtime_error(std::string("no vocab to parse token at ") + src);
    }

    // Parse <token> and tokenize to obtain the token id
    while (*pos != 0 && *pos != '>') {
        pos++;
    }
    if (*pos != '>') {
        throw std::runtime_error(std::string("expecting '>' at ") + pos);
    }
    pos++;

    llama_token tokens[2];
    int32_t n_tokens = vocab->tokenize(src, static_cast<int32_t>(pos - src), tokens, 2, false, true);
    if (n_tokens != 1) {
        // must tokenize to exactly 1 token
        throw std::runtime_error("invalid token '" + std::string(src, pos - src) + "'");
    }
    return std::make_pair(tokens[0], pos);
}

static void print_grammar_char(FILE * file, uint32_t c) {
    if (0x20 <= c && c <= 0x7f) {
        fprintf(file, "%c", static_cast<char>(c));
    } else {
        // cop out of encoding UTF-8
        fprintf(file, "<U+%04X>", c);
    }
}

static bool is_char_element(llama_grammar_element elem) {
    switch (elem.type) {
        case LLAMA_GRETYPE_CHAR:           return true;
        case LLAMA_GRETYPE_CHAR_NOT:       return true;
        case LLAMA_GRETYPE_CHAR_ALT:       return true;
        case LLAMA_GRETYPE_CHAR_RNG_UPPER: return true;
        case LLAMA_GRETYPE_CHAR_ANY:       return true;
        default:                           return false;
    }
}

static void print_rule_binary(FILE * file, const llama_grammar_rule & rule) {
    for (auto elem : rule) {
        switch (elem.type) {
            case LLAMA_GRETYPE_END:            fprintf(file, "END");            break;
            case LLAMA_GRETYPE_ALT:            fprintf(file, "ALT");            break;
            case LLAMA_GRETYPE_RULE_REF:       fprintf(file, "RULE_REF");       break;
            case LLAMA_GRETYPE_CHAR:           fprintf(file, "CHAR");           break;
            case LLAMA_GRETYPE_CHAR_NOT:       fprintf(file, "CHAR_NOT");       break;
            case LLAMA_GRETYPE_CHAR_RNG_UPPER: fprintf(file, "CHAR_RNG_UPPER"); break;
            case LLAMA_GRETYPE_CHAR_ALT:       fprintf(file, "CHAR_ALT");       break;
            case LLAMA_GRETYPE_CHAR_ANY:       fprintf(file, "CHAR_ANY");       break;
            case LLAMA_GRETYPE_TOKEN:          fprintf(file, "TOKEN");          break;
            case LLAMA_GRETYPE_TOKEN_NOT:      fprintf(file, "TOKEN_NOT");      break;
        }
        switch (elem.type) {
            case LLAMA_GRETYPE_END:
            case LLAMA_GRETYPE_ALT:
            case LLAMA_GRETYPE_RULE_REF:
                fprintf(file, "(%u) ", elem.value);
                break;
            case LLAMA_GRETYPE_CHAR:
            case LLAMA_GRETYPE_CHAR_NOT:
            case LLAMA_GRETYPE_CHAR_RNG_UPPER:
            case LLAMA_GRETYPE_CHAR_ALT:
            case LLAMA_GRETYPE_CHAR_ANY:
                fprintf(file, "(\"");
                print_grammar_char(file, elem.value);
                fprintf(file, "\") ");
                break;
            case LLAMA_GRETYPE_TOKEN:
                fprintf(file, "<[");
                fprintf(file, "%u", elem.value);
                fprintf(file, "]> ");
                break;
            case LLAMA_GRETYPE_TOKEN_NOT:
                fprintf(file, "!");
                fprintf(file, "<[");
                fprintf(file, "%u", elem.value);
                fprintf(file, "]> ");
                break;
        }
    }
    fprintf(file, "\n");
}

static void print_rule(
        FILE     * file,
        uint32_t   rule_id,
        const llama_grammar_rule & rule,
        const std::map<uint32_t, std::string> & symbol_id_names) {
    if (rule.empty() || rule.back().type != LLAMA_GRETYPE_END) {
        throw std::runtime_error(
            "malformed rule, does not end with LLAMA_GRETYPE_END: " + std::to_string(rule_id));
    }
    fprintf(file, "%s ::= ", symbol_id_names.at(rule_id).c_str());
    for (size_t i = 0, end = rule.size() - 1; i < end; i++) {
        llama_grammar_element elem = rule[i];
        switch (elem.type) {
            case LLAMA_GRETYPE_END:
                throw std::runtime_error(
                    "unexpected end of rule: " + std::to_string(rule_id) + "," +
                    std::to_string(i));
            case LLAMA_GRETYPE_ALT:
                fprintf(file, "| ");
                break;
            case LLAMA_GRETYPE_RULE_REF:
                fprintf(file, "%s ", symbol_id_names.at(elem.value).c_str());
                break;
            case LLAMA_GRETYPE_CHAR:
                fprintf(file, "[");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_NOT:
                fprintf(file, "[^");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                fprintf(file, "-");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_ALT:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "LLAMA_GRETYPE_CHAR_ALT without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_ANY:
                fprintf(file, ".");
                break;
            case LLAMA_GRETYPE_TOKEN:
                fprintf(file, "<[");
                fprintf(file, "%u", elem.value);
                fprintf(file, "]> ");
                break;
            case LLAMA_GRETYPE_TOKEN_NOT:
                fprintf(file, "!");
                fprintf(file, "<[");
                fprintf(file, "%u", elem.value);
                fprintf(file, "]> ");
                break;
        }
        if (is_char_element(elem)) {
            switch (rule[i + 1].type) {
                case LLAMA_GRETYPE_CHAR_ALT:
                case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                case LLAMA_GRETYPE_CHAR_ANY:
                    break;
                default:
                    fprintf(file, "] ");
            }
        }
    }
    fprintf(file, "\n");
}

//
// Regex utilities
//

size_t llama_grammar_trigger_pattern::find(const std::string & input) const {
    auto find_start_pos = [](const std::smatch & match) {
        // get from the first matched capturing group to the end of the string
        size_t start = std::string::npos;
        for (auto i = 1u; i < match.size(); i++) {
            if (match.length(i) > 0) {
                start = match.position(i);
                break;
            }
        }
        if (start == std::string::npos) {
            start = match.position(0);
        }
        return start;
    };

    if (!pattern.empty() && pattern.front() == '^' && pattern.back() == '$') {
        // match against the entire input
        std::smatch match;
        if (std::regex_match(input, match, regex)) {
            return find_start_pos(match);
        }
    }

    // search anywhere
    std::smatch match;
    if (std::regex_search(input, match, regex)) {
        return find_start_pos(match);
    }

    return std::string::npos;
}


//
// implementation
//

uint32_t llama_grammar_parser::get_symbol_id(const char * src, size_t len) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    auto result = symbol_ids.emplace(std::string(src, len), next_id);
    return result.first->second;
}

uint32_t llama_grammar_parser::generate_symbol_id(const std::string & base_name) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
    return next_id;
}

void llama_grammar_parser::add_rule(uint32_t rule_id, const llama_grammar_rule & rule) {
    if (rules.size() <= rule_id) {
        rules.resize(rule_id + 1);
    }
    rules[rule_id] = rule;
}

const char * llama_grammar_parser::parse_alternates(
        const char        * src,
        const std::string & rule_name,
        uint32_t            rule_id,
        bool                is_nested,
        uint32_t            depth) {
    llama_grammar_rule rule;
    const char * pos = parse_sequence(src, rule_name, rule, is_nested, depth);
    while (*pos == '|') {
        rule.push_back({LLAMA_GRETYPE_ALT, 0});
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(pos, rule_name, rule, is_nested, depth);
    }
    rule.push_back({LLAMA_GRETYPE_END, 0});
    add_rule(rule_id, rule);
    return pos;
}

const char * llama_grammar_parser::parse_sequence(
        const char         * src,
        const std::string  & rule_name,
        llama_grammar_rule & rule,
        bool               is_nested,
        uint32_t           depth) {
    size_t last_sym_start = rule.size();
    const char * pos = src;
    uint64_t n_prev_rules = 1;

    // use UINT64_MAX as the empty value because we aligned to the proper uint64_t type so -1 can't be used
    // (though it's technically the same as -1 now)
    auto handle_repetitions = [&](uint64_t min_times, uint64_t max_times) {
        bool no_max = max_times == UINT64_MAX;
        if (last_sym_start == rule.size()) {
            throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + pos);
        }

        // apply transformation to previous symbol (last_sym_start to end) according to
        // the following rewrite rules:
        // S{m,n} --> S S S (m times) S'(n-m)
        //            S'(x)   ::= S S'(x-1) |
        //            (... n-m definitions of these S' rules ...)
        //            S'(1)   ::= S |
        // S{m,} -->  S S S (m times) S'
        //            S'     ::= S S' |
        // S*     --> S{0,}
        //        --> S'     ::= S S' |
        // S+     --> S{1,}
        //        --> S S'
        //            S'     ::= S S' |
        // S?     --> S{0,1}
        //        --> S'
        //            S'     ::= S |

        llama_grammar_rule prev_rule(rule.begin() + last_sym_start, rule.end());
        // Calculate the total number of rules that will be generated by this repetition
        uint64_t total_rules = 1; // Start with 1 for the original rule
        if (!no_max && max_times > 0) {
            total_rules = max_times;
        } else if (min_times > 0) {
            total_rules = min_times;
        }

        if (n_prev_rules * total_rules > MAX_REPETITION_THRESHOLD) {
            throw std::runtime_error("number of rules that are going to be repeated multiplied by the new repetition exceeds sane defaults, please reduce the number of repetitions or rule complexity");
        }

        if (min_times == 0) {
            rule.resize(last_sym_start);
        } else {
            // Repeat the previous elements (min_times - 1) times
            for (uint64_t i = 1; i < min_times; i++) {
                rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
            }
        }

        uint32_t last_rec_rule_id = 0;
        auto n_opt = no_max ? 1 : max_times - min_times;

        llama_grammar_rule rec_rule(prev_rule);
        for (uint64_t i = 0; i < n_opt; i++) {
            rec_rule.resize(prev_rule.size());
            uint32_t rec_rule_id = generate_symbol_id( rule_name);
            if (i > 0 || no_max) {
                rec_rule.push_back({LLAMA_GRETYPE_RULE_REF, no_max ? rec_rule_id : last_rec_rule_id});
            }
            rec_rule.push_back({LLAMA_GRETYPE_ALT, 0});
            rec_rule.push_back({LLAMA_GRETYPE_END, 0});
            add_rule( rec_rule_id, rec_rule);
            last_rec_rule_id = rec_rule_id;
        }
        if (n_opt > 0) {
            rule.push_back({LLAMA_GRETYPE_RULE_REF, last_rec_rule_id});
        }
        n_prev_rules *= total_rules;
        GGML_ASSERT(n_prev_rules >= 1);
    };

    while (*pos) {
        if (*pos == '"') { // literal string
            pos++;
            last_sym_start = rule.size();
            n_prev_rules = 1;
            while (*pos != '"') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                     pos       = char_pair.second;
                rule.push_back({LLAMA_GRETYPE_CHAR, char_pair.first});
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') { // char range(s)
            pos++;
            enum llama_gretype start_type = LLAMA_GRETYPE_CHAR;
            if (*pos == '^') {
                pos++;
                start_type = LLAMA_GRETYPE_CHAR_NOT;
            }
            last_sym_start = rule.size();
            n_prev_rules = 1;
            while (*pos != ']') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                     pos       = char_pair.second;
                enum llama_gretype type = last_sym_start < rule.size()
                    ? LLAMA_GRETYPE_CHAR_ALT
                    : start_type;

                rule.push_back({type, char_pair.first});
                if (pos[0] == '-' && pos[1] != ']') {
                    if (!pos[1]) {
                        throw std::runtime_error("unexpected end of input");
                    }
                    auto endchar_pair = parse_char(pos + 1);
                         pos          = endchar_pair.second;
                    rule.push_back({LLAMA_GRETYPE_CHAR_RNG_UPPER, endchar_pair.first});
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '<' || *pos == '!') { // token
            auto type = LLAMA_GRETYPE_TOKEN;
            if (*pos == '!') { // token inverse
                type = LLAMA_GRETYPE_TOKEN_NOT;
                pos++;
            }
            auto token_pair = parse_token(vocab, pos);
            const char * token_end  = token_pair.second;
            last_sym_start = rule.size();
            n_prev_rules = 1;
            rule.push_back({type, token_pair.first});
            pos = parse_space(token_end, is_nested);
        } else if (is_word_char(*pos)) { // rule reference
            const char * name_end    = parse_name(pos);
            uint32_t ref_rule_id = get_symbol_id(pos, name_end - pos);
            pos = parse_space(name_end, is_nested);
            last_sym_start = rule.size();
            n_prev_rules = 1;
            rule.push_back({LLAMA_GRETYPE_RULE_REF, ref_rule_id});
        } else if (*pos == '(') { // grouping
            if (depth >= GRAMMAR_RECURSIVE_LIMIT) {
                throw std::runtime_error("grammar recursion depth limit exceeded (max " + std::to_string(GRAMMAR_RECURSIVE_LIMIT) + "), please reduce nesting depth");
            }
            // parse nested alternates into synthesized rule
            pos = parse_space(pos + 1, true);
            uint32_t n_rules_before = symbol_ids.size();
            uint32_t sub_rule_id = generate_symbol_id(rule_name);
            pos = parse_alternates(pos, rule_name, sub_rule_id, true, depth + 1);
            n_prev_rules = std::max(1u, (uint32_t)symbol_ids.size() - n_rules_before);
            last_sym_start = rule.size();
            // output reference to synthesized rule
            rule.push_back({LLAMA_GRETYPE_RULE_REF, sub_rule_id});
            if (*pos != ')') {
                throw std::runtime_error(std::string("expecting ')' at ") + pos);
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '.') { // any char
            last_sym_start = rule.size();
            n_prev_rules = 1;
            rule.push_back({LLAMA_GRETYPE_CHAR_ANY, 0});
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '*') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, -1);
        } else if (*pos == '+') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(1, -1);
        } else if (*pos == '?') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, 1);
        } else if (*pos == '{') {
            pos = parse_space(pos + 1, is_nested);

            if (!is_digit_char(*pos)) {
                throw std::runtime_error(std::string("expecting an int at ") + pos);
            }
            const char * int_end = parse_int(pos);
            uint64_t min_times = std::stoull(std::string(pos, int_end - pos));
            pos = parse_space(int_end, is_nested);

            uint64_t max_times = UINT64_MAX; // default: no max limit

            if (*pos == '}') {
                max_times = min_times;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == ',') {
                pos = parse_space(pos + 1, is_nested);

                if (is_digit_char(*pos)) {
                    const char * int_end = parse_int(pos);
                    max_times = std::stoull(std::string(pos, int_end - pos));
                    pos = parse_space(int_end, is_nested);
                }

                if (*pos != '}') {
                    throw std::runtime_error(std::string("expecting '}' at ") + pos);
                }
                pos = parse_space(pos + 1, is_nested);
            } else {
                throw std::runtime_error(std::string("expecting ',' at ") + pos);
            }
            if (min_times > MAX_REPETITION_THRESHOLD) {
                throw std::runtime_error(std::string("number of repetitions exceeds sane defaults, please reduce the number of repetitions"));
            }
            if (max_times != UINT64_MAX && max_times > MAX_REPETITION_THRESHOLD) {
                max_times = UINT64_MAX;
            }
            handle_repetitions(min_times, max_times);
        } else {
            break;
        }
    }
    return pos;
}

const char * llama_grammar_parser::parse_rule(const char * src) {
    const char * name_end = parse_name(src);
    const char * pos      = parse_space(name_end, false);
    size_t       name_len = name_end - src;
    uint32_t     rule_id  = get_symbol_id(src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error(std::string("expecting ::= at ") + pos);
    }
    pos = parse_space(pos + 3, true);

    pos = parse_alternates(pos, name, rule_id, false);

    if (*pos == '\r') {
        pos += pos[1] == '\n' ? 2 : 1;
    } else if (*pos == '\n') {
        pos++;
    } else if (*pos) {
        throw std::runtime_error(std::string("expecting newline or end at ") + pos);
    }
    return parse_space(pos, true);
}

bool llama_grammar_parser::parse(const char * src) {
    try {
        const char * pos = parse_space(src, true);
        while (*pos) {
            pos = parse_rule(pos);
        }
        // Validate the state to ensure that all rules are defined
        for (const auto & rule : rules) {
            if (rule.empty()) {
                throw std::runtime_error("Undefined rule");
            }
            for (const auto & elem : rule) {
                if (elem.type == LLAMA_GRETYPE_RULE_REF) {
                    // Ensure that the rule at that location exists
                    if (elem.value >= rules.size() || rules[elem.value].empty()) {
                        // Get the name of the rule that is missing
                        for (const auto & kv : symbol_ids) {
                            if (kv.second == elem.value) {
                                throw std::runtime_error("Undefined rule identifier '" + kv.first + "'");
                            }
                        }
                    }
                }
            }
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: error parsing grammar: %s\n\n%s\n", __func__, err.what(), src);
        rules.clear();
        return false;
    }

    return true;
}

void llama_grammar_parser::print(FILE * file) {
    try {
        std::map<uint32_t, std::string> symbol_id_names;
        for (const auto & kv : symbol_ids) {
            symbol_id_names[kv.second] = kv.first;
        }
        for (size_t i = 0, end = rules.size(); i < end; i++) {
            // fprintf(file, "%zu: ", i);
            // print_rule_binary(file, rules[i]);
            print_rule(file, uint32_t(i), rules[i], symbol_id_names);
            // fprintf(file, "\n");
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "\n%s: error printing grammar: %s\n", __func__, err.what());
    }
}

llama_grammar_stack llama_grammar_parser::c_rules() const {
    llama_grammar_stack ret;
    ret.reserve(rules.size());
    for (const auto & rule : rules) {
        ret.push_back(rule.data());
    }
    return ret;
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool llama_grammar_is_end_of_sequence(const llama_grammar_element * pos) {
    switch (pos->type) {
        case LLAMA_GRETYPE_END: return true;  // NOLINT
        case LLAMA_GRETYPE_ALT: return true;  // NOLINT
        default:                return false;
    }
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static std::pair<bool, const llama_grammar_element *> llama_grammar_match_char(
        const llama_grammar_element * pos,
        const uint32_t                chr) {
    bool found            = false;
    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;

    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT); // NOLINT

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
            // Any character matches "."
            found = true;
            pos += 1;
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return std::make_pair(found == is_positive_char, pos);
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static bool llama_grammar_match_partial_char(
        const llama_grammar_element * pos,
        const llama_partial_utf8      partial_utf8,
        bool                          allow_overlong = false) {
    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;
    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT);

    uint32_t partial_value = partial_utf8.value;
    int      n_remain      = partial_utf8.n_remain;

    // invalid sequence or 7-bit char split across 2 bytes (overlong)
    if (n_remain < 0 || (!allow_overlong && n_remain == 1 && partial_value < 2)) {
        return false;
    }

    // range of possible code points this partial UTF-8 sequence could complete to
    uint32_t low  = partial_value << (n_remain * 6);
    uint32_t high = low | ((1 << (n_remain * 6)) - 1);

    if (low == 0 && !allow_overlong) {
        if (n_remain == 2) {
            low = 1 << 11;
        } else if (n_remain == 3) {
            low = 1 << 16;
        }
    }

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            if (pos->value <= high && low <= pos[1].value) {
                return is_positive_char;
            }
            pos += 2;
        } else if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
            // Any character matches "."
            return true;
        } else {
            // exact char match, e.g. [a] or "a"
            if (low <= pos->value && pos->value <= high) {
                return is_positive_char;
            }
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return !is_positive_char;
}

// returns true iff token matches the rule at pos (regular or inverse)
// asserts that pos is pointing to a token element
static bool llama_grammar_match_token(
    const llama_grammar_element * pos,
    const llama_token             token) {
    GGML_ASSERT(pos->type == LLAMA_GRETYPE_TOKEN || pos->type == LLAMA_GRETYPE_TOKEN_NOT);
    if (pos->type == LLAMA_GRETYPE_TOKEN) {
        return pos->value == static_cast<uint32_t>(token);
    }
    if (pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
        return pos->value != static_cast<uint32_t>(token);
    }
    return false;
}

// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static bool llama_grammar_advance_stack(
        const llama_grammar_rules  & rules,
        const llama_grammar_stack  & stack,
        llama_grammar_stacks & new_stacks,
        size_t * budget = nullptr) {
    std::vector<llama_grammar_stack> todo;
    todo.push_back(stack);

    auto stack_cmp = [](const llama_grammar_stack & a, const llama_grammar_stack & b) {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
            [](const llama_grammar_element * pa, const llama_grammar_element * pb) {
                return pa < pb;  // Compare pointer addresses
            }
        );
    };

    std::set<llama_grammar_stack, decltype(stack_cmp)> seen(stack_cmp);

    while (!todo.empty()) {
        if (budget) {
            if (todo.back().size() >= *budget) {
                return false;
            }
            *budget -= todo.back().size() + 1;
        }
        llama_grammar_stack curr_stack_candidate = std::move(todo.back());
        todo.pop_back();

        auto [curr_stack_it, inserted] = seen.insert(std::move(curr_stack_candidate));
        if (!inserted) {
            continue;
        }
        const llama_grammar_stack & curr_stack = *curr_stack_it;

        if (curr_stack.empty()) {
            if (std::find(new_stacks.begin(), new_stacks.end(), curr_stack) == new_stacks.end()) {
                new_stacks.emplace_back(curr_stack);
            }
            continue;
        }

        const llama_grammar_element * pos = curr_stack.back();

        switch (pos->type) {
        case LLAMA_GRETYPE_RULE_REF: {
            const size_t                  rule_id = static_cast<size_t>(pos->value);
            const llama_grammar_element * subpos  = rules[rule_id].data();
            do {
                // init new stack without the top (pos)
                llama_grammar_stack next_stack(curr_stack.begin(), curr_stack.end() - 1);
                if (!llama_grammar_is_end_of_sequence(pos + 1)) {
                    // if this rule ref is followed by another element, add that to stack
                    next_stack.push_back(pos + 1);
                }
                if (!llama_grammar_is_end_of_sequence(subpos)) {
                    // if alternate is nonempty, add to stack
                    next_stack.push_back(subpos);
                }
                if (budget) {
                    if (next_stack.size() >= *budget) {
                        return false;
                    }
                    *budget -= next_stack.size() + 1;
                }
                todo.push_back(std::move(next_stack));
                while (!llama_grammar_is_end_of_sequence(subpos)) {
                    // scan to end of alternate def
                    subpos++;
                }
                if (subpos->type == LLAMA_GRETYPE_ALT) {
                    // there's another alternate def of this rule to process
                    subpos++;
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case LLAMA_GRETYPE_CHAR:
        case LLAMA_GRETYPE_CHAR_NOT:
        case LLAMA_GRETYPE_CHAR_ANY:
        case LLAMA_GRETYPE_TOKEN:
        case LLAMA_GRETYPE_TOKEN_NOT:
            if (std::find(new_stacks.begin(), new_stacks.end(), curr_stack) == new_stacks.end()) {
                // only add the stack if it's not a duplicate of one we already have
                new_stacks.emplace_back(curr_stack);
            }
            break;
        default:
            // end of alternate (LLAMA_GRETYPE_END, LLAMA_GRETYPE_ALT) or middle of char range
            // (LLAMA_GRETYPE_CHAR_ALT, LLAMA_GRETYPE_CHAR_RNG_UPPER); stack should never be left on
            // those
            GGML_ABORT("fatal error");
        }
    }
    return true;
}

// Left-factor grammar rules to reduce redundant parallel stacks.
//
// When a rule has many alternates (e.g. tool-choice ::= tc0 | tc1 | ... | tc119)
// and the referenced rules share a common element-content prefix, extract that
// prefix so the parser processes it once instead of N times.
//
// Handles two cases:
// 1. Inline common prefix: rule = "ab1" | "ab2" -> rule = "ab" rule', rule' = "1" | "2"
// 2. Cross-rule prefix: rule = A | B | C where A,B,C start with same elements
//    -> rule = common_prefix rule', rule' = A_suffix | B_suffix | C_suffix
static void llama_grammar_left_factor(llama_grammar_rules & rules) {
    auto elem_eq = [](const llama_grammar_element & a, const llama_grammar_element & b) {
        return a.type == b.type && a.value == b.value;
    };

    // Keep the prefix boundary outside character classes.
    auto trim_prefix = [](const llama_grammar_rule & rule, size_t start, size_t length) {
        while (length > 0) {
            const auto type = rule[start + length].type;
            if (type != LLAMA_GRETYPE_CHAR_ALT && type != LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                break;
            }
            length--;
        }
        return length;
    };

    struct alt_info {
        size_t rule_idx;
        size_t start;
        size_t end;
    };

    const size_t n_rules_original = rules.size();
    // Original rules factor at most once; generated rules do not resolve references.
    // Each new suffix removes a non-empty prefix.
    bool changed = true;
    while (changed) {
        changed = false;
        const size_t n_rules = rules.size();

        for (size_t ri = 0; ri < n_rules; ri++) {
            const auto & rule = rules[ri];

            // Split rule into alternates
            std::vector<std::pair<size_t, size_t>> alt_ranges;
            size_t alt_start = 0;
            for (size_t j = 0; j < rule.size(); j++) {
                if (rule[j].type == LLAMA_GRETYPE_ALT || rule[j].type == LLAMA_GRETYPE_END) {
                    alt_ranges.push_back({alt_start, j});
                    alt_start = j + 1;
                }
            }

            if (alt_ranges.size() < 4) {
                continue;
            }

            // Resolve each alternate: if it's a single RULE_REF, follow it to get the
            // actual element sequence from the referenced rule.
            std::vector<alt_info> resolved;
            resolved.reserve(alt_ranges.size());
            bool all_single_refs = true;

            for (const auto & [astart, aend] : alt_ranges) {
                size_t len = aend - astart;
                // Resolving references in generated suffixes can expand recursive rules forever.
                if (ri < n_rules_original && len == 1 && rule[astart].type == LLAMA_GRETYPE_RULE_REF) {
                    size_t ref_id = static_cast<size_t>(rule[astart].value);
                    if (ref_id < rules.size()) {
                        size_t ref_end = 0;
                        for (size_t k = 0; k < rules[ref_id].size(); k++) {
                            if (rules[ref_id][k].type == LLAMA_GRETYPE_ALT ||
                                rules[ref_id][k].type == LLAMA_GRETYPE_END) {
                                ref_end = k;
                                break;
                            }
                        }
                        bool single_alt = (rules[ref_id][ref_end].type == LLAMA_GRETYPE_END);
                        if (single_alt) {
                            resolved.push_back({ref_id, 0, ref_end});
                            continue;
                        }
                    }
                }
                all_single_refs = false;
                resolved.push_back({ri, astart, aend});
            }

            if (!all_single_refs) {
                resolved.clear();
                for (const auto & [astart, aend] : alt_ranges) {
                    resolved.push_back({ri, astart, aend});
                }
            }

            size_t prefix_len = 0;
            bool prefix_match = true;
            while (prefix_match) {
                const auto & first = resolved[0];
                if (first.start + prefix_len >= first.end) {
                    break;
                }
                const auto & ref_elem = rules[first.rule_idx][first.start + prefix_len];

                for (size_t ai = 1; ai < resolved.size(); ai++) {
                    const auto & alt = resolved[ai];
                    if (alt.start + prefix_len >= alt.end ||
                        !elem_eq(rules[alt.rule_idx][alt.start + prefix_len], ref_elem)) {
                        prefix_match = false;
                        break;
                    }
                }
                if (prefix_match) {
                    prefix_len++;
                }
            }

            for (const auto & alt : resolved) {
                prefix_len = trim_prefix(rules[alt.rule_idx], alt.start, prefix_len);
            }
            if (prefix_len == 0) {
                continue;
            }

            uint32_t suffix_choice_id = static_cast<uint32_t>(rules.size());
            llama_grammar_rule suffix_choice;

            for (size_t ai = 0; ai < resolved.size(); ai++) {
                if (ai > 0) {
                    suffix_choice.push_back({LLAMA_GRETYPE_ALT, 0});
                }
                const auto & alt = resolved[ai];
                for (size_t j = alt.start + prefix_len; j < alt.end; j++) {
                    suffix_choice.push_back(rules[alt.rule_idx][j]);
                }
            }
            suffix_choice.push_back({LLAMA_GRETYPE_END, 0});

            llama_grammar_rule new_rule;
            const auto & first = resolved[0];
            for (size_t j = 0; j < prefix_len; j++) {
                new_rule.push_back(rules[first.rule_idx][first.start + j]);
            }
            new_rule.push_back({LLAMA_GRETYPE_RULE_REF, suffix_choice_id});
            new_rule.push_back({LLAMA_GRETYPE_END, 0});

            rules.push_back(std::move(suffix_choice));
            rules[ri] = std::move(new_rule);
            changed = true;
        }
    }
}

static bool llama_grammar_detect_left_recursion(
        const llama_grammar_rules & rules,
        size_t rule_index,
        std::vector<bool> * rules_visited,
        std::vector<bool> * rules_in_progress,
        std::vector<bool> * rules_may_be_empty,
        uint32_t depth = 0) {
    if ((*rules_in_progress)[rule_index]) {
        return true;
    }
    if (depth > GRAMMAR_RECURSIVE_LIMIT) {
        throw std::runtime_error("grammar rules are chained more than " + std::to_string(GRAMMAR_RECURSIVE_LIMIT) + " levels deep");
    }

    (*rules_in_progress)[rule_index] = true;

    const llama_grammar_rule & rule = rules[rule_index];

    // First check if the rule might produce the empty string. This could be done combined with the second
    // step but it's more readable as two steps.
    bool at_rule_start = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (llama_grammar_is_end_of_sequence(&rule[i])) {
            if (at_rule_start) {
                (*rules_may_be_empty)[rule_index] = true;
                break;
            }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
    }

    // Second, recurse into leftmost nonterminals (or next-leftmost as long as the previous nonterminal may
    // be empty)
    bool recurse_into_nonterminal = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (rule[i].type == LLAMA_GRETYPE_RULE_REF && recurse_into_nonterminal) {
            if (llama_grammar_detect_left_recursion(rules, (size_t)rule[i].value, rules_visited, rules_in_progress, rules_may_be_empty, depth + 1)) {
                return true;
            }
            if (!((*rules_may_be_empty)[(size_t)rule[i].value])) {
                recurse_into_nonterminal = false;
            }
        } else if (llama_grammar_is_end_of_sequence(&rule[i])) {
            recurse_into_nonterminal = true;
        } else {
            recurse_into_nonterminal = false;
        }
    }

    (*rules_in_progress)[rule_index] = false;
    (*rules_visited)[rule_index] = true;

    return false;
}

const llama_grammar_rules & llama_grammar_get_rules(const struct llama_grammar * grammar) {
    return *grammar->rules;
}

llama_grammar_stacks & llama_grammar_get_stacks(struct llama_grammar * grammar) {
    return grammar->stacks;
}

static void llama_grammar_accept_chr(
        struct llama_grammar       & grammar,
        const llama_grammar_stack  & stack,
              uint32_t               chr,
              llama_grammar_stacks & new_stacks) {
    if (stack.empty()) {
        return;
    }

    const llama_grammar_element * pos = stack.back();

    // ignore if this turns into a token
    if (pos->type == LLAMA_GRETYPE_TOKEN || pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
        return;
    }

    auto match = llama_grammar_match_char(pos, chr);
    if (match.first) {
        llama_grammar_stack new_stack(stack.begin(), stack.end() - 1);
        if (!llama_grammar_is_end_of_sequence(match.second)) {
            new_stack.push_back(match.second);
        }
        llama_grammar_advance_stack(*grammar.rules, new_stack, new_stacks);
    }
}

void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr) {
    auto & stacks_new = grammar->accept_scratch.stacks_new;
    stacks_new.clear();
    stacks_new.reserve(grammar->stacks.size());

    for (const auto & stack : grammar->stacks) {
        llama_grammar_accept_chr(*grammar, stack, chr, stacks_new);
    }

    grammar->stacks.swap(stacks_new);
}

// checks the code point at depth of each candidate (an index into the input) against the stack
// returns true when some match: they go to next, and the stacks that follow to next_stacks
static bool llama_grammar_reject_step(
        const llama_grammar_rules      & rules,
        const llama_grammar_stack      & stack,
        const llama_grammar_candidates & candidates,
        size_t                           depth,
        const std::vector<uint32_t>    & cur,
              std::vector<uint32_t>    & rejects,
              std::vector<uint32_t>    & next,
              llama_grammar_stacks     & next_stacks) {
    rejects.reserve(cur.size());

    if (stack.empty()) {
        for (const uint32_t i : cur) {
            const auto & tok = candidates[i];
            if (tok.code_points[depth] != 0 || tok.partial_utf8.n_remain != 0) {
                rejects.push_back(i);
            }
        }
        return false;
    }

    const llama_grammar_element * stack_pos = stack.back();

    // if the top of the stack is a token rule, then we only need to check the token id
    if (stack_pos->type == LLAMA_GRETYPE_TOKEN || stack_pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
        for (const uint32_t i : cur) {
            const auto & tok = candidates[i];
            if (tok.code_points[depth] == 0) {
                // reached the end of a token consumed by char rules, reject iff it ended
                // in a partial response
                if (tok.partial_utf8.n_remain != 0) {
                    rejects.push_back(i);
                }
            } else if (!llama_grammar_match_token(stack_pos, tok.id)) {
                rejects.push_back(i);
            }
        }
        return false;
    }

    next.reserve(cur.size());

    for (const uint32_t i : cur) {
        const auto & tok = candidates[i];
        if (tok.code_points[depth] == 0) {
            // reached end of full codepoints in token, reject iff it ended in a partial sequence
            // that cannot satisfy this position in grammar
            if (tok.partial_utf8.n_remain != 0 &&
                    !llama_grammar_match_partial_char(stack_pos, tok.partial_utf8)) {
                rejects.push_back(i);
            }
        } else if (llama_grammar_match_char(stack_pos, tok.code_points[depth]).first) {
            next.push_back(i);
        } else {
            rejects.push_back(i);
        }
    }

    const auto * stack_pos_after = llama_grammar_match_char(stack_pos, 0).second;

    // update top of stack to next element, if any
    llama_grammar_stack stack_after(stack.begin(), stack.end() - 1);
    if (!llama_grammar_is_end_of_sequence(stack_pos_after)) {
        stack_after.push_back(stack_pos_after);
    }
    llama_grammar_advance_stack(rules, stack_after, next_stacks);

    return true;
}

static std::vector<uint32_t> llama_grammar_reject_indices(
        const llama_grammar_rules      & rules,
        const llama_grammar_stacks     & stacks,
        const llama_grammar_candidates & candidates,
        std::vector<uint32_t>            all);

static llama_grammar_candidates llama_grammar_reject_candidates(
        const llama_grammar_rules      & rules,
        const llama_grammar_stacks     & stacks,
        const llama_grammar_candidates & candidates) {
    GGML_ASSERT(!stacks.empty()); // REVIEW

    std::vector<uint32_t> all(candidates.size());
    std::iota(all.begin(), all.end(), 0u);

    llama_grammar_candidates rejects;
    for (const uint32_t i : llama_grammar_reject_indices(rules, stacks, candidates, std::move(all))) {
        rejects.push_back(candidates[i]);
    }
    return rejects;
}

static int llama_grammar_scan_threads() {
    static const int n_threads = [] {
        if (const char * env = getenv("LLAMA_GRAMMAR_THREADS")) {
            return std::max(1, atoi(env));
        }
        return (int) std::clamp(std::thread::hardware_concurrency() / 2, 1u, 8u);
    }();
    return n_threads;
}

// llama_grammar_reject_candidates over chunks of the candidates on several threads, for full-vocabulary scans
static llama_grammar_candidates llama_grammar_reject_candidates_mt(
        const llama_grammar_rules      & rules,
        const llama_grammar_stacks     & stacks,
        const llama_grammar_candidates & candidates) {
    const int n_threads = llama_grammar_scan_threads();
    if (n_threads <= 1 || candidates.size() < 16384) {
        return llama_grammar_reject_candidates(rules, stacks, candidates);
    }
    GGML_ASSERT(!stacks.empty());

    // more chunks than threads balance the cost, which varies along the vocabulary
    const size_t n_chunks = 4 * (size_t) n_threads;
    const size_t chunk = (candidates.size() + n_chunks - 1) / n_chunks;
    std::vector<std::vector<uint32_t>> results(n_chunks);
    std::atomic<size_t> next{0};
    const auto worker = [&] {
        for (size_t c = next++; c < n_chunks; c = next++) {
            const size_t begin = std::min(candidates.size(), c * chunk);
            const size_t end = std::min(candidates.size(), begin + chunk);
            std::vector<uint32_t> subset(end - begin);
            std::iota(subset.begin(), subset.end(), uint32_t(begin));
            results[c] = llama_grammar_reject_indices(rules, stacks, candidates, std::move(subset));
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(n_threads - 1);
    for (int t = 1; t < n_threads; ++t) {
        threads.emplace_back(worker);
    }
    worker();
    for (auto & thread : threads) {
        thread.join();
    }

    llama_grammar_candidates rejects;
    for (const auto & result : results) {
        for (const uint32_t i : result) {
            rejects.push_back(candidates[i]);
        }
    }
    return rejects;
}

// the candidates among `all` that no stack accepts, each stack narrows the rejects of the previous one
// no recursion: a level per code point, a token from the GGUF can be long
static std::vector<uint32_t> llama_grammar_reject_indices(
        const llama_grammar_rules      & rules,
        const llama_grammar_stacks     & stacks,
        const llama_grammar_candidates & candidates,
        std::vector<uint32_t>            all) {

    // a level checks the candidates that reach its code point against its stacks in turn
    struct level {
        llama_grammar_stacks  stacks;
        size_t                i;       // the stack in progress
        std::vector<uint32_t> cur;     // the candidates rejected by stacks[0, i)
        std::vector<uint32_t> rejects; // the rejects of stacks[i] at this code point
    };

    std::vector<level> levels;
    levels.push_back({ stacks, 0, std::move(all), {} });

    while (true) {
        level & lv = levels.back();

        if (lv.i == lv.stacks.size() || lv.cur.empty()) {
            if (levels.size() == 1) {
                break;
            }
            // the rejects at the next code point belong to stacks[i] of the parent level
            level & parent = levels[levels.size() - 2];
            parent.rejects.insert(parent.rejects.end(), lv.cur.begin(), lv.cur.end());
            levels.pop_back();
            parent.cur = std::move(parent.rejects);
            parent.rejects = {};
            parent.i++;
            continue;
        }

        std::vector<uint32_t> rejects;
        std::vector<uint32_t> next;
        llama_grammar_stacks  next_stacks;

        if (!llama_grammar_reject_step(rules, lv.stacks[lv.i], candidates, levels.size() - 1, lv.cur, rejects, next, next_stacks)) {
            lv.cur = std::move(rejects);
            lv.i++;
            continue;
        }

        GGML_ASSERT(!next_stacks.empty());
        lv.rejects = std::move(rejects);
        levels.push_back({ std::move(next_stacks), 0, std::move(next), {} });
    }

    return std::move(levels[0].cur);
}

llama_grammar_candidates llama_grammar_reject_candidates_for_stack(
        const llama_grammar_rules      & rules,
        const llama_grammar_stack      & stack,
        const llama_grammar_candidates & candidates) {
    return llama_grammar_reject_candidates(rules, llama_grammar_stacks{ stack }, candidates);
}

////////////////////

struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
        const llama_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index) {
    const llama_grammar_element * pos;

    // copy rule definitions into vectors
    llama_grammar_rules vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
    }

    // Validate that all rule references point to valid rules
    for (size_t i = 0; i < n_rules; i++) {
        for (const auto & elem : vec_rules[i]) {
            if (elem.type == LLAMA_GRETYPE_RULE_REF) {
                if (elem.value >= n_rules || vec_rules[elem.value].empty()) {
                    LLAMA_LOG_ERROR("invalid grammar: rule %zu references undefined rule %u\n", i, elem.value);
                    return nullptr;
                }
            }
        }
    }

    // Left-factor rules to reduce redundant parallel stacks
    llama_grammar_left_factor(vec_rules);

    // Check for left recursion (use post-factoring rule count)
    const size_t n_rules_total = vec_rules.size();
    std::vector<bool> rules_visited(n_rules_total);
    std::vector<bool> rules_in_progress(n_rules_total);
    std::vector<bool> rules_may_be_empty(n_rules_total);
    try {
        for (size_t i = 0; i < n_rules_total; i++) {
            if (rules_visited[i]) {
                continue;
            }
            if (llama_grammar_detect_left_recursion(vec_rules, i, &rules_visited, &rules_in_progress, &rules_may_be_empty)) {
                LLAMA_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu", i);
                return nullptr;
            }
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("unsupported grammar: %s\n", err.what());
        return nullptr;
    }

    // loop over alternates of start rule to build initial stacks
    llama_grammar_stacks stacks;
    pos = vec_rules[start_rule_index].data();
    do {
        llama_grammar_stack stack;
        if (!llama_grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        llama_grammar_advance_stack(vec_rules, stack, stacks);
        while (!llama_grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == LLAMA_GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    // Important: vec_rules has to be moved here, not copied, because stacks contains
    // pointers to elements of vec_rules. If vec_rules were copied into llama_grammar
    // then the pointers would be invalidated when the local vec_rules goes out of scope.
    auto * result = new llama_grammar;
    result->vocab = vocab;
    result->rules = std::make_shared<const llama_grammar_rules>(std::move(vec_rules));
    result->rule_positions = llama_grammar_index_rules(*result->rules);
    result->stacks = std::move(stacks);
    result->stacks_initial = result->stacks;
    result->mask_cache = std::make_shared<llama_grammar_mask_cache>();
    return result;
}

// Parsed programs and their mask caches, interned by vocab, grammar text and root, so that requests with the same grammar
// skip parsing and reuse masks: the cache key identifies the program by address.
struct llama_grammar_program {
    const llama_vocab * vocab;
    std::string text;
    std::string root;
    std::shared_ptr<const llama_grammar_rules> rules;
    std::shared_ptr<const llama_grammar::rule_position_map> rule_positions;
    llama_grammar_stacks stacks_initial;
    std::shared_ptr<llama_grammar_mask_cache> mask_cache;
    uint64_t used = 0;
};

static constexpr size_t LLAMA_GRAMMAR_MAX_PROGRAMS = 8;

static std::mutex & llama_grammar_programs_mutex() {
    static std::mutex mutex;
    return mutex;
}

static std::vector<llama_grammar_program> & llama_grammar_programs() {
    static std::vector<llama_grammar_program> programs;
    return programs;
}

static uint64_t llama_grammar_programs_clock = 0;

static bool llama_grammar_programs_disabled() {
    static const bool disabled = getenv("LLAMA_GRAMMAR_NO_INTERN") != nullptr;
    return disabled;
}

static bool llama_grammar_program_find(const llama_vocab * vocab, const char * text, const char * root, llama_grammar_program & out) {
    if (llama_grammar_programs_disabled()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(llama_grammar_programs_mutex());
    for (auto & program : llama_grammar_programs()) {
        if (program.vocab == vocab && program.root == root && program.text == text) {
            program.used = ++llama_grammar_programs_clock;
            out = program;
            return true;
        }
    }
    return false;
}

static void llama_grammar_program_add(llama_grammar_program program) {
    if (llama_grammar_programs_disabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(llama_grammar_programs_mutex());
    auto & programs = llama_grammar_programs();
    if (programs.size() >= LLAMA_GRAMMAR_MAX_PROGRAMS) {
        auto victim = std::min_element(programs.begin(), programs.end(), [](const auto & a, const auto & b) { return a.used < b.used; });
        programs.erase(victim);
    }
    program.used = ++llama_grammar_programs_clock;
    programs.push_back(std::move(program));
}

void llama_grammar_programs_drop(const llama_vocab * vocab) {
    std::lock_guard<std::mutex> lock(llama_grammar_programs_mutex());
    auto & programs = llama_grammar_programs();
    programs.erase(std::remove_if(programs.begin(), programs.end(), [vocab](const auto & p) { return p.vocab == vocab; }), programs.end());
}

struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens) {
    std::vector<llama_token>    vec_trigger_tokens;
    std::vector<llama_grammar_trigger_pattern> vec_trigger_patterns;
    for (size_t i = 0; i < num_trigger_tokens; i++) {
        GGML_ASSERT(trigger_tokens != nullptr);
        vec_trigger_tokens.push_back(trigger_tokens[i]);
    }
    for (size_t i = 0; i < num_trigger_patterns; i++) {
        GGML_ASSERT(trigger_patterns != nullptr);
        auto & trigger = vec_trigger_patterns.emplace_back();
        trigger.pattern = trigger_patterns[i];
        trigger.regex = std::regex(trigger.pattern);
    }

    const auto make_grammar = [&](const llama_grammar_program & program) {
        auto * result = new llama_grammar;
        result->vocab = vocab;
        result->rules = program.rules;
        result->rule_positions = program.rule_positions;
        result->stacks = program.stacks_initial;
        result->stacks_initial = program.stacks_initial;
        result->lazy = lazy;
        result->awaiting_trigger = lazy;
        result->trigger_tokens = std::move(vec_trigger_tokens);
        result->trigger_patterns = std::move(vec_trigger_patterns);
        result->mask_cache = program.mask_cache;
        return result;
    };

    {
        llama_grammar_program program;
        if (vocab && llama_grammar_program_find(vocab, grammar_str, grammar_root, program)) {
            return make_grammar(program);
        }
    }

    llama_grammar_parser parser(vocab);

    // if there is a grammar, parse it
    // rules will be empty (default) if there are parse errors
    if (!parser.parse(grammar_str) || parser.rules.empty()) {
        LLAMA_LOG_ERROR("failed to parse grammar\n");
        return nullptr;
    }

    // Ensure that the grammar contains the start symbol
    if (parser.symbol_ids.find(grammar_root) == parser.symbol_ids.end()) {
        LLAMA_LOG_ERROR("grammar does not contain a '%s' symbol\n", grammar_root);
        return nullptr;
    }

    std::vector<const llama_grammar_element *> grammar_rules(parser.c_rules());

    const size_t n_rules = grammar_rules.size();
    const size_t start_rule_index = parser.symbol_ids.at(grammar_root);

    const llama_grammar_element * pos;

    // copy rule definitions into vectors
    llama_grammar_rules vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = grammar_rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
    }

    // Left-factor rules to reduce redundant parallel stacks
    llama_grammar_left_factor(vec_rules);

    // Check for left recursion (use post-factoring rule count)
    const size_t n_rules_total = vec_rules.size();
    std::vector<bool> rules_visited(n_rules_total);
    std::vector<bool> rules_in_progress(n_rules_total);
    std::vector<bool> rules_may_be_empty(n_rules_total);
    try {
        for (size_t i = 0; i < n_rules_total; i++) {
            if (rules_visited[i]) {
                continue;
            }
            if (llama_grammar_detect_left_recursion(vec_rules, i, &rules_visited, &rules_in_progress, &rules_may_be_empty)) {
                LLAMA_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu\n", i);
                return nullptr;
            }
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("unsupported grammar: %s\n", err.what());
        return nullptr;
    }

    // loop over alternates of start rule to build initial stacks
    llama_grammar_stacks stacks;
    pos = vec_rules[start_rule_index].data();
    do {
        llama_grammar_stack stack;
        if (!llama_grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        llama_grammar_advance_stack(vec_rules, stack, stacks);
        while (!llama_grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == LLAMA_GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    // Important: vec_rules has to be moved here, not copied, because stacks contains
    // pointers to elements of vec_rules. If vec_rules were copied into llama_grammar
    // then the pointers would be invalidated when the local vec_rules goes out of scope.
    llama_grammar_program program;
    program.vocab = vocab;
    program.text = grammar_str;
    program.root = grammar_root;
    program.rules = std::make_shared<const llama_grammar_rules>(std::move(vec_rules));
    program.rule_positions = llama_grammar_index_rules(*program.rules);
    program.stacks_initial = std::move(stacks);
    program.mask_cache = std::make_shared<llama_grammar_mask_cache>();
    if (vocab) {
        llama_grammar_program_add(program);
    }
    return make_grammar(program);
}

void llama_grammar_free_impl(struct llama_grammar * grammar) {
    if (grammar == nullptr) {
        return;
    }

    delete grammar;
}

struct llama_grammar * llama_grammar_clone_impl(const struct llama_grammar & grammar) {
    auto * result = new llama_grammar(grammar);
    result->scratch = {};
    result->accept_scratch = {};
    return result;
}

void llama_grammar_reset_impl(struct llama_grammar & grammar) {
    grammar.stacks = grammar.stacks_initial;
    grammar.partial_utf8 = {0, 0};
    grammar.awaiting_trigger = grammar.lazy;
    grammar.trigger_buffer.clear();
    grammar.trigger_buffer_positions.clear();
}

bool llama_grammar_mask_cache::key::operator==(const key & other) const {
    return program == other.program && vocab == other.vocab && n_vocab == other.n_vocab &&
           partial_utf8.value == other.partial_utf8.value && partial_utf8.n_remain == other.partial_utf8.n_remain &&
           enabled == other.enabled && stacks == other.stacks;
}

static bool llama_grammar_mask_cache_key(const llama_grammar & grammar, size_t max_words, llama_grammar_mask_cache::key & key) {
    key.program = grammar.rules.get();
    key.vocab = grammar.vocab;
    key.n_vocab = grammar.vocab->n_tokens();
    key.partial_utf8 = grammar.partial_utf8;
    if (key.partial_utf8.n_remain == 0) {
        key.partial_utf8.value = 0;
    }
    key.enabled = !grammar.awaiting_trigger;

    size_t n_words = 0;
    for (const auto & stack : grammar.stacks) {
        if (n_words == max_words || stack.size() > (max_words - n_words - 1) / 2) {
            return false;
        }
        n_words += 1 + 2 * stack.size();
    }

    std::vector<std::vector<size_t>> stacks;
    stacks.reserve(grammar.stacks.size());
    for (const auto & stack : grammar.stacks) {
        std::vector<size_t> positions;
        positions.reserve(2 * stack.size());
        for (const auto * pos : stack) {
            const auto it = grammar.rule_positions->find(pos);
            GGML_ASSERT(it != grammar.rule_positions->end());
            positions.push_back(it->second.first);
            positions.push_back(it->second.second);
        }
        stacks.push_back(std::move(positions));
    }
    std::sort(stacks.begin(), stacks.end());
    stacks.erase(std::unique(stacks.begin(), stacks.end()), stacks.end());

    key.stacks.clear();
    key.stacks.reserve(n_words);
    for (const auto & stack : stacks) {
        key.stacks.push_back(stack.size() / 2);
        key.stacks.insert(key.stacks.end(), stack.begin(), stack.end());
    }
    return true;
}

static uint64_t llama_grammar_mask_cache_hash(const llama_grammar_mask_cache::key & key) {
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](uint64_t v) {
        h = (h ^ v) * 1099511628211ull;
    };
    mix((uintptr_t) key.program);
    mix((uintptr_t) key.vocab);
    mix(key.n_vocab);
    mix((uint64_t(key.partial_utf8.value) << 8) | uint8_t(key.partial_utf8.n_remain));
    mix(key.enabled);
    for (const auto word : key.stacks) {
        mix(word);
    }
    return h;
}

// evict the least recently used entry
static void llama_grammar_mask_cache_evict(llama_grammar_mask_cache & cache) {
    GGML_ASSERT(!cache.entries.empty());
    size_t victim = 0;
    for (size_t i = 1; i < cache.entries.size(); ++i) {
        if (cache.entries[i].used < cache.entries[victim].used) {
            victim = i;
        }
    }
    auto & entry = cache.entries[victim];
    cache.bytes -= entry.bytes;
    auto it = cache.index.find(entry.hash);
    if (it != cache.index.end() && it->second == victim) {
        cache.index.erase(it);
    }
    if (victim + 1 != cache.entries.size()) {
        entry = std::move(cache.entries.back());
        auto moved = cache.index.find(entry.hash);
        if (moved != cache.index.end() && moved->second == cache.entries.size() - 1) {
            moved->second = victim;
        }
    }
    cache.entries.pop_back();
}

static llama_grammar_mask_packed llama_grammar_mask_pack(std::vector<uint32_t> words, uint32_t n_vocab) {
    llama_grammar_mask_packed packed;
    packed.n_vocab = n_vocab;
    int64_t n_allowed = 0;
    for (const auto word : words) {
        n_allowed += std::bitset<32>(word).count();
    }
    if (n_allowed == 0 || n_allowed == n_vocab) {
        packed.kind = n_allowed == 0 ? GGML_TOKEN_MASK_NONE : GGML_TOKEN_MASK_ALL;
        return packed;
    }
    if (std::min<int64_t>(n_allowed, n_vocab - n_allowed) < (int64_t) words.size()) {
        const bool allow_ids = n_allowed <= n_vocab - n_allowed;
        auto ids = std::make_shared<std::vector<uint32_t>>();
        ids->reserve(allow_ids ? n_allowed : n_vocab - n_allowed);
        for (size_t i = 0; i < words.size(); ++i) {
            uint32_t word = allow_ids ? words[i] : ~words[i];
            if (i + 1 == words.size() && n_vocab % 32) {
                word &= (uint32_t(1) << (n_vocab % 32)) - 1;
            }
            while (word) {
#if defined(_MSC_VER)
                unsigned long bit;
                _BitScanForward(&bit, word);
#else
                const int bit = __builtin_ctz(word);
#endif
                ids->push_back(uint32_t(32*i + bit));
                word &= word - 1;
            }
        }
        packed.kind = allow_ids ? GGML_TOKEN_MASK_ALLOW_IDS : GGML_TOKEN_MASK_DENY_IDS;
        packed.payload = std::move(ids);
        return packed;
    }
    packed.kind = GGML_TOKEN_MASK_BITSET;
    packed.payload = std::make_shared<const std::vector<uint32_t>>(std::move(words));
    return packed;
}

static std::vector<uint32_t> llama_grammar_mask_expand(const llama_grammar_mask_packed & packed) {
    const uint32_t n_vocab = packed.n_vocab;
    std::vector<uint32_t> mask(n_vocab / 32 + (n_vocab % 32 != 0), 0);
    switch (packed.kind) {
        case GGML_TOKEN_MASK_NONE:
            break;
        case GGML_TOKEN_MASK_ALL:
        case GGML_TOKEN_MASK_DENY_IDS:
            std::fill(mask.begin(), mask.end(), UINT32_MAX);
            if (n_vocab % 32 != 0) {
                mask.back() = (uint32_t(1) << (n_vocab % 32)) - 1;
            }
            if (packed.kind == GGML_TOKEN_MASK_DENY_IDS) {
                for (const auto id : *packed.payload) {
                    mask[id / 32] &= ~(uint32_t(1) << (id % 32));
                }
            }
            break;
        case GGML_TOKEN_MASK_ALLOW_IDS:
            for (const auto id : *packed.payload) {
                mask[id / 32] |= uint32_t(1) << (id % 32);
            }
            break;
        case GGML_TOKEN_MASK_BITSET:
            mask = *packed.payload;
            break;
        default:
            GGML_ABORT("invalid packed mask kind");
    }
    return mask;
}

bool llama_grammar_get_allowed_tokens_trie(const llama_grammar & grammar, std::vector<uint32_t> & mask, size_t max_work) {
    if (grammar.awaiting_trigger || grammar.stacks.empty()) {
        return false;
    }
    for (const auto & rule : *grammar.rules) {
        for (const auto & element : rule) {
            if (max_work == 0) {
                return false;
            }
            --max_work;
            if (element.type == LLAMA_GRETYPE_TOKEN || element.type == LLAMA_GRETYPE_TOKEN_NOT) {
                return false;
            }
        }
    }
    const auto * trie = grammar.vocab->get_grammar_trie();
    if (!trie) {
        return false;
    }

    const uint32_t n_vocab = grammar.vocab->n_tokens();
    mask.assign(n_vocab / 32 + (n_vocab % 32 != 0), 0);
    auto allow = [&](llama_token id) { mask[id / 32] |= uint32_t(1) << (id % 32); };
    for (const auto & stack : grammar.stacks) {
        if (stack.empty()) {
            for (const auto id : trie->eog) {
                allow(id);
            }
            break;
        }
    }

    struct state {
        llama_grammar_stacks stacks;
        llama_partial_utf8 partial;
        bool strict;  // Only the continuation of a previous token validates continuation bytes.
        bool nul = false;
        bool stopped = false; // A decoded zero terminates the match, but decoding continues.
    };
    struct level {
        uint32_t child;
        state current;
    };
    for (const auto & stack : grammar.stacks) {
        if (stack.size() >= max_work) {
            return false;
        }
        max_work -= stack.size() + 1;
    }
    std::vector<level> levels;
    levels.push_back({ trie->nodes[0].child, { grammar.stacks, grammar.partial_utf8, grammar.partial_utf8.n_remain > 0 } });
    while (!levels.empty()) {
        auto & parent = levels.back();
        if (parent.child == llama_vocab_trie::none) {
            levels.pop_back();
            continue;
        }
        const auto & node = trie->nodes[parent.child];
        parent.child = node.sibling;
        size_t words = 1;
        for (const auto & stack : parent.current.stacks) {
            if (stack.size() >= max_work || words > max_work - stack.size() - 1) {
                return false;
            }
            words += stack.size() + 1;
        }
        if (words > max_work) {
            return false;
        }
        max_work -= words;
        state next = parent.current;
        if (!next.nul) {
            const uint8_t byte = node.byte;
            if (byte == 0) {
                next.nul = true;
            } else {
                if (next.partial.n_remain > 0) {
                    if (next.strict && (byte >> 6) != 2) {
                        continue;
                    }
                    next.partial.value = (next.partial.value << 6) + (byte & 0x3f);
                    --next.partial.n_remain;
                } else {
                    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
                    next.partial.n_remain = lookup[byte >> 4] - 1;
                    if (next.partial.n_remain < 0) {
                        continue;
                    }
                    next.partial.value = byte & ((1 << (7 - next.partial.n_remain)) - 1);
                }
                if (next.partial.n_remain == 0) {
                    next.strict = false;
                    if (next.partial.value == 0) {
                        next.stopped = true;
                    }
                    if (!next.stopped) {
                        llama_grammar_stacks after;
                        for (const auto & stack : next.stacks) {
                            if (stack.empty()) {
                                continue;
                            }
                            const auto match = llama_grammar_match_char(stack.back(), next.partial.value);
                            if (!match.first) {
                                continue;
                            }
                            llama_grammar_stack remaining(stack.begin(), stack.end() - 1);
                            if (!llama_grammar_is_end_of_sequence(match.second)) {
                                remaining.push_back(match.second);
                            }
                            if (!llama_grammar_advance_stack(*grammar.rules, remaining, after, &max_work)) {
                                return false;
                            }
                        }
                        next.stacks = std::move(after);
                        if (next.stacks.empty()) {
                            continue;
                        }
                    }
                }
            }
        }

        if (!next.nul && !next.stopped && next.partial.n_remain > 0 && next.partial.value != 0 &&
                next.partial.value <= (UINT32_MAX >> (6 * next.partial.n_remain))) {
            bool viable = false;
            for (const auto & stack : next.stacks) {
                if (!stack.empty() && (stack.back()->type == LLAMA_GRETYPE_CHAR_NOT ||
                        llama_grammar_match_partial_char(stack.back(), next.partial, true))) {
                    viable = true;
                    break;
                }
            }
            if (!viable) {
                continue;
            }
        }

        if (node.terminal != llama_vocab_trie::none) {
            bool valid = false;
            for (const auto & stack : next.stacks) {
                if (next.partial.n_remain == 0 || (!stack.empty() &&
                        llama_grammar_match_partial_char(stack.back(), next.partial))) {
                    valid = true;
                    break;
                }
            }
            if (valid) {
                for (uint32_t terminal = node.terminal; terminal != llama_vocab_trie::none; terminal = trie->terminals[terminal].next) {
                    if (max_work == 0) {
                        return false;
                    }
                    --max_work;
                    allow(trie->terminals[terminal].id);
                }
            }
        }
        levels.push_back({ node.child, std::move(next) });
    }
    return true;
}

static std::vector<uint32_t> llama_grammar_compute_allowed_tokens(const llama_grammar & grammar) {
    const uint32_t n_vocab = grammar.vocab->n_tokens();
    std::vector<uint32_t> mask(n_vocab / 32 + (n_vocab % 32 != 0), UINT32_MAX);
    if (n_vocab % 32 != 0) {
        mask.back() = (uint32_t(1) << (n_vocab % 32)) - 1;
    }
    if (grammar.awaiting_trigger) {
        return mask;
    }

    bool allow_eog = false;
    for (const auto & stack : grammar.stacks) {
        if (stack.empty()) {
            allow_eog = true;
            break;
        }
    }

    const auto * decoded = grammar.partial_utf8.n_remain == 0 ? grammar.vocab->get_grammar_decoded() : nullptr;
    if (decoded) {
        for (size_t i = 0; i < mask.size(); ++i) {
            mask[i] &= ~(decoded->unusable[i] | (allow_eog ? 0 : decoded->eog[i]));
        }
        const auto rejects = llama_grammar_reject_candidates_mt(*grammar.rules, grammar.stacks, decoded->candidates);
        for (const auto & reject : rejects) {
            mask[reject.index / 32] &= ~(uint32_t(1) << (reject.index % 32));
        }
        return mask;
    }

    auto & scratch = grammar.scratch;
    scratch.clear();
    scratch.candidates.reserve(n_vocab);
    scratch.offsets.reserve(n_vocab);

    for (uint32_t i = 0; i < n_vocab; ++i) {
        const llama_token id = static_cast<llama_token>(i);
        const std::string & piece = grammar.vocab->token_to_piece(id);
        if (grammar.vocab->is_eog(id)) {
            if (!allow_eog) {
                mask[i / 32] &= ~(uint32_t(1) << (i % 32));
            }
        } else if (piece.empty() || piece[0] == 0) {
            mask[i / 32] &= ~(uint32_t(1) << (i % 32));
        } else {
            scratch.add(i, id, piece, grammar.partial_utf8);
        }
    }

    scratch.set_pointers();
    const auto rejects = llama_grammar_reject_candidates_mt(*grammar.rules, grammar.stacks, scratch.candidates);
    for (const auto & reject : rejects) {
        mask[reject.index / 32] &= ~(uint32_t(1) << (reject.index % 32));
    }
    return mask;
}

static bool llama_grammar_get_allowed_packed_impl(const llama_grammar & grammar, bool cached_only, llama_grammar_mask_packed & result);

// a negated character class on top of a stack, as in the body of a JSON string, accepts most of the vocabulary:
// the trie walk would exhaust its budget before the exact scan runs anyway
static bool llama_grammar_is_wide(const llama_grammar & grammar) {
    static const bool disabled = getenv("LLAMA_GRAMMAR_TRIE_WIDE") != nullptr;
    if (disabled) {
        return false;
    }
    for (const auto & stack : grammar.stacks) {
        if (!stack.empty() && stack.back()->type == LLAMA_GRETYPE_CHAR_NOT) {
            return true;
        }
    }
    return false;
}

llama_grammar_mask_packed llama_grammar_get_allowed_packed(const llama_grammar & grammar) {
    llama_grammar_mask_packed result;
    llama_grammar_get_allowed_packed_impl(grammar, false, result);
    return result;
}

bool llama_grammar_get_allowed_packed_cached(const llama_grammar & grammar, llama_grammar_mask_packed & result) {
    return llama_grammar_get_allowed_packed_impl(grammar, true, result);
}

static bool llama_grammar_get_allowed_packed_impl(const llama_grammar & grammar, bool cached_only, llama_grammar_mask_packed & result) {
    GGML_ASSERT(grammar.vocab != nullptr);
    GGML_ASSERT(grammar.mask_cache != nullptr);
    auto & cache = *grammar.mask_cache;
    const uint32_t n_vocab = grammar.vocab->n_tokens();

    // the packed payload never exceeds the bitset, so the bitset size bounds an entry's payload
    const size_t mask_bytes = (n_vocab / 32 + (n_vocab % 32 != 0)) * sizeof(uint32_t);
    const size_t overhead = sizeof(llama_grammar_mask_cache::entry) + sizeof(std::vector<uint32_t>) + 2 * sizeof(void *);

    llama_grammar_mask_cache::key key;
    uint64_t hash = 0;
    bool cacheable = false;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        while (!cache.entries.empty() && (cache.entries.size() > cache.max_entries || cache.bytes > cache.max_bytes)) {
            llama_grammar_mask_cache_evict(cache);
        }
        cacheable = cache.max_entries > 0 && cache.max_bytes >= overhead &&
            mask_bytes <= cache.max_bytes - overhead &&
            llama_grammar_mask_cache_key(grammar, (cache.max_bytes - overhead - mask_bytes) / sizeof(size_t), key);
        if (cacheable) {
            hash = llama_grammar_mask_cache_hash(key);
            const auto it = cache.index.find(hash);
            if (it != cache.index.end() && cache.entries[it->second].state == key) {
                auto & entry = cache.entries[it->second];
                entry.used = ++cache.clock;
                ++cache.hits;
                result = { entry.kind, n_vocab, entry.payload };
                return true;
            }
        }
        if (cached_only) {
            return false;
        }
        ++cache.misses;
    }

    const int64_t t_start_us = ggml_time_us();
    std::vector<uint32_t> words;
    const bool trie = !llama_grammar_is_wide(grammar) && llama_grammar_get_allowed_tokens_trie(grammar, words);
    if (!trie) {
        words = llama_grammar_compute_allowed_tokens(grammar);
    }
    auto packed = llama_grammar_mask_pack(std::move(words), n_vocab);
    const int64_t t_miss_us = ggml_time_us() - t_start_us;

    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        cache.t_miss_us += t_miss_us;
        cache.scans += !trie;
    }
    if (cacheable) {
        std::lock_guard<std::mutex> lock(cache.mutex);
        const size_t bytes = overhead + key.stacks.capacity() * sizeof(size_t) +
            (packed.payload ? packed.payload->capacity() * sizeof(uint32_t) : 0);
        if (bytes <= cache.max_bytes) {
            while (!cache.entries.empty() && (cache.entries.size() >= cache.max_entries || bytes > cache.max_bytes - cache.bytes)) {
                llama_grammar_mask_cache_evict(cache);
            }
            cache.index[hash] = cache.entries.size();
            cache.entries.push_back({ std::move(key), hash, ++cache.clock, packed.kind, packed.payload, bytes });
            cache.bytes += bytes;
        }
    }
    result = std::move(packed);
    return true;
}

std::vector<uint32_t> llama_grammar_get_allowed_tokens(const llama_grammar & grammar) {
    return llama_grammar_mask_expand(llama_grammar_get_allowed_packed(grammar));
}

void llama_grammar_apply_impl(const struct llama_grammar & grammar, llama_token_data_array * cur_p) {
    GGML_ASSERT(grammar.vocab != nullptr);

    if (grammar.awaiting_trigger) {
        return;
    }

    bool allow_eog = false;
    for (const auto & stack : grammar.stacks) {
        if (stack.empty()) {
            allow_eog = true;
            break;
        }
    }

    auto & scratch = grammar.scratch;
    scratch.clear();

    scratch.candidates.reserve(cur_p->size);
    scratch.offsets.reserve(cur_p->size);

    const auto * decoded = grammar.partial_utf8.n_remain == 0 && cur_p->size >= 1024 ?
            grammar.vocab->get_grammar_decoded() : nullptr;
    if (decoded) {
        bool identity = cur_p->size == decoded->candidate_index.size();
        for (size_t i = 0; identity && i < cur_p->size; ++i) {
            identity = cur_p->data[i].id == (llama_token) i;
        }
        if (identity) {
            for (size_t i = 0; i < cur_p->size; ++i) {
                const uint32_t word = decoded->unusable[i / 32] | (allow_eog ? 0 : decoded->eog[i / 32]);
                if ((word >> (i % 32)) & 1) {
                    cur_p->data[i].logit = -INFINITY;
                }
            }
            const auto rejects = llama_grammar_reject_candidates_mt(*grammar.rules, grammar.stacks, decoded->candidates);
            for (const auto & reject : rejects) {
                cur_p->data[reject.index].logit = -INFINITY;
            }
            return;
        }
        for (size_t i = 0; i < cur_p->size; ++i) {
            const llama_token id = cur_p->data[i].id;
            const auto position = decoded->candidate_index[id];
            if (position == llama_vocab_decoded::none) {
                if (!allow_eog || !((decoded->eog[id / 32] >> (id % 32)) & 1)) {
                    cur_p->data[i].logit = -INFINITY;
                }
            } else {
                const auto & candidate = decoded->candidates[position];
                scratch.candidates.push_back({ i, candidate.code_points, candidate.partial_utf8, id });
            }
        }
    } else {
        for (size_t i = 0; i < cur_p->size; ++i) {
            const llama_token id      = cur_p->data[i].id;
            const std::string & piece = grammar.vocab->token_to_piece(id);

            if (grammar.vocab->is_eog(id)) {
                if (!allow_eog) {
                    cur_p->data[i].logit = -INFINITY;
                }
            } else if (piece.empty() || piece[0] == 0) {
                cur_p->data[i].logit = -INFINITY;
            } else {
                scratch.add(i, id, piece, grammar.partial_utf8);
            }
        }
        scratch.set_pointers();
    }

    const auto rejects = llama_grammar_reject_candidates_mt(*grammar.rules, grammar.stacks, scratch.candidates);
    for (const auto & reject : rejects) {
        cur_p->data[reject.index].logit = -INFINITY;
    }
}

void llama_grammar_accept_impl(struct llama_grammar & grammar, llama_token token) {
    GGML_ASSERT(grammar.vocab != nullptr);

    const auto & piece = grammar.vocab->token_to_piece(token);

    if (grammar.awaiting_trigger) {
        if (std::find(grammar.trigger_tokens.begin(), grammar.trigger_tokens.end(), token) != grammar.trigger_tokens.end()) {
            grammar.awaiting_trigger = false;
            grammar.trigger_buffer.clear();
            llama_grammar_accept_token(grammar, token, piece);
            LLAMA_LOG_DEBUG("Grammar triggered on token %u (`%s`)", token, piece.c_str());
            return;
        } else {
            auto position = std::make_pair(grammar.trigger_buffer.size(), grammar.trigger_buffer.size() + piece.size());
            grammar.trigger_buffer_positions.push_back(std::make_pair(token, position));
            grammar.trigger_buffer += piece;

            for (const auto & trigger_pattern : grammar.trigger_patterns) {
                auto start = trigger_pattern.find(grammar.trigger_buffer);
                if (start != std::string::npos) {
                    grammar.awaiting_trigger = false;

                    // replay tokens that overlap with [start, end)
                    for (const auto & [tok, tok_pos] : grammar.trigger_buffer_positions) {
                        auto [tok_start, tok_end] = tok_pos;
                        if (tok_end <= start) {
                            continue;
                        }

                        size_t piece_start = (tok_start < start) ? start : tok_start; // allow for partial token pieces
                        size_t piece_len = tok_end - piece_start;
                        auto tok_piece = grammar.trigger_buffer.substr(piece_start, piece_len);
                        llama_grammar_accept_token(grammar, tok, tok_piece);
                    }

                    auto constrained_str = grammar.trigger_buffer.substr(start);
                    grammar.trigger_buffer.clear();
                    grammar.trigger_buffer_positions.clear();
                    LLAMA_LOG_DEBUG("Grammar triggered on regex: '%s'\n", constrained_str.c_str());
                    return;
                }
            }
            LLAMA_LOG_DEBUG("Grammar still awaiting trigger after token %d (`%s`)\n", token, piece.c_str());
            return;
        }
    }

    if (grammar.vocab->is_eog(token)) {
        for (const auto & stack : grammar.stacks) {
            if (stack.empty()) {
                return;
            }
        }
        GGML_ABORT("fatal error");
    }

    llama_grammar_accept_token(grammar, token, piece);
}

void llama_grammar_accept_str(struct llama_grammar & grammar, const std::string & piece) {
    // Note terminating 0 in decoded string
    auto & code_points = grammar.accept_scratch.code_points;
    code_points.clear();
    const auto partial_utf8 = decode_utf8(piece, grammar.partial_utf8, code_points);

    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        llama_grammar_accept(&grammar, *it);
    }

    grammar.partial_utf8 = partial_utf8;
    if (grammar.stacks.empty()) {
        throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + piece);
    }
}

void llama_grammar_accept_token(struct llama_grammar & grammar, llama_token token, const std::string & piece) {
    // Note terminating 0 in decoded string
    auto & scratch = grammar.accept_scratch;
    auto & code_points = scratch.code_points;
    code_points.clear();
    const auto partial_utf8 = decode_utf8(piece, grammar.partial_utf8, code_points);

    auto & stacks_new = scratch.stacks_new;
    stacks_new.clear();
    stacks_new.reserve(grammar.stacks.size());

    for (const auto & stack : grammar.stacks) {
        if (stack.empty()) {
            continue;
        }

        const llama_grammar_element * pos = stack.back();

        if (pos->type == LLAMA_GRETYPE_TOKEN || pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
            if (llama_grammar_match_token(pos, token)) {
                llama_grammar_stack new_stack(stack.begin(), stack.end() - 1);
                if (!llama_grammar_is_end_of_sequence(pos + 1)) {
                    new_stack.push_back(pos + 1);
                }
                llama_grammar_advance_stack(*grammar.rules, new_stack, stacks_new);
            }
        } else {
            auto & current_stacks = scratch.current_stacks;
            current_stacks.resize(1);
            current_stacks[0] = stack;

            for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
                auto & next_stacks = scratch.next_stacks;
                next_stacks.clear();

                for (const auto & cur_stack : current_stacks) {
                    llama_grammar_accept_chr(grammar, cur_stack, *it, next_stacks);
                }

                current_stacks.swap(next_stacks);
                if (current_stacks.empty()) {
                    break;
                }
            }

            for (auto & surviving_stack : current_stacks) {
                if (std::find(stacks_new.begin(), stacks_new.end(), surviving_stack) == stacks_new.end()) {
                    stacks_new.emplace_back(surviving_stack);
                }
            }
        }
    }

    grammar.stacks.swap(stacks_new);
    grammar.partial_utf8 = partial_utf8;

    if (grammar.stacks.empty()) {
        throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + piece + " (" + std::to_string(token) + ")");
    }
}

std::shared_ptr<llama_grammar_gpu_context> llama_grammar_gpu_init(ggml_backend_buffer_type_t buft, const llama_grammar * grammar) {
    GGML_UNUSED(buft);
    GGML_UNUSED(grammar);
    const char * enabled = getenv("LLAMA_GRAMMAR_GPU_MASK");
    if (!enabled || strcmp(enabled, "1") != 0) { return nullptr; }
    return nullptr;
}
