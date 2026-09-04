#pragma once

#include "llama.h"

#include <functional>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_vocab;

// grammar element type
enum llama_gretype {
    // end of rule definition
    LLAMA_GRETYPE_END            = 0,

    // start of alternate definition for rule
    LLAMA_GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    LLAMA_GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    LLAMA_GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    LLAMA_GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding LLAMA_GRETYPE_CHAR or
    // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    LLAMA_GRETYPE_CHAR_ALT       = 6,

    // any character (.)
    LLAMA_GRETYPE_CHAR_ANY       = 7,

    // terminal element: token (<[token-id]>)
    LLAMA_GRETYPE_TOKEN          = 8,

    // inverse token (!<[token-id]>)
    LLAMA_GRETYPE_TOKEN_NOT      = 9,
};

typedef struct llama_grammar_element {
    enum llama_gretype type;
    uint32_t           value; // Unicode code point, rule ID, or token ID
} llama_grammar_element;

struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_partial_utf8   partial_utf8;
    llama_token          id;
};

using llama_grammar_rule  = std::vector<      llama_grammar_element>;
using llama_grammar_stack = std::vector<const llama_grammar_element *>;

using llama_grammar_rules      = std::vector<llama_grammar_rule>;
using llama_grammar_stacks     = std::vector<llama_grammar_stack>;
using llama_grammar_candidates = std::vector<llama_grammar_candidate>;

// TODO: remove, needed for tests atm
const llama_grammar_rules  & llama_grammar_get_rules (const struct llama_grammar * grammar);
      llama_grammar_stacks & llama_grammar_get_stacks(      struct llama_grammar * grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llama_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr);

std::vector<llama_grammar_candidate> llama_grammar_reject_candidates_for_stack(
        const llama_grammar_rules      & rules,
        const llama_grammar_stack      & stack,
        const llama_grammar_candidates & candidates);

struct llama_grammar_parser {
    const llama_vocab * vocab;
    std::map<std::string, uint32_t> symbol_ids;

    llama_grammar_rules rules;

    llama_grammar_parser(const struct llama_vocab * vocab = nullptr) : vocab(vocab) {}

    llama_grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const llama_grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested);

    const char * parse_sequence(
            const char         * src,
            const std::string  & rule_name,
            llama_grammar_rule & rule,
            bool               is_nested);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct llama_grammar_trigger_pattern {
    std::string pattern;
    std::regex  regex;

    size_t find(const std::string & input) const;
};

// Hash/equality helpers for llama_grammar_reject_candidates_for_stack.
// Content-based comparison avoids false cache hits when the heap reuses code_points addresses.
struct llama_grammar_stack_hash {
    size_t operator()(const llama_grammar_stack & stack) const noexcept {
        size_t h = 0;
        for (const auto * p : stack) {
            h ^= std::hash<const void *>{}(p) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct llama_grammar_stack_eq {
    bool operator()(const llama_grammar_stack & a, const llama_grammar_stack & b) const noexcept {
        return a == b;
    }
};

struct llama_grammar_candidates_hash {
    size_t operator()(const llama_grammar_candidates & cands) const noexcept {
        size_t h = cands.size();
        for (const auto & c : cands) {
            h ^= std::hash<size_t>{}(c.index)                 + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<llama_token>{}(c.id)               + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(c.partial_utf8.value)  + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(c.partial_utf8.n_remain)    + 0x9e3779b9 + (h << 6) + (h >> 2);
            // hash code-point content, not the pointer
            for (const uint32_t * cp = c.code_points; *cp != 0; ++cp) {
                h ^= std::hash<uint32_t>{}(*cp) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
        }
        return h;
    }
};

struct llama_grammar_candidates_eq {
    bool operator()(const llama_grammar_candidates & a, const llama_grammar_candidates & b) const noexcept {
        if (a.size() != b.size()) { return false; }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].index != b[i].index || a[i].id != b[i].id ||
                a[i].partial_utf8.value   != b[i].partial_utf8.value ||
                a[i].partial_utf8.n_remain != b[i].partial_utf8.n_remain) {
                return false;
            }
            // compare code-point content, not the pointer
            const uint32_t * pa = a[i].code_points;
            const uint32_t * pb = b[i].code_points;
            while (*pa != 0 && *pb != 0) {
                if (*pa != *pb) { return false; }
                ++pa; ++pb;
            }
            if (*pa != *pb) { return false; }
        }
        return true;
    }
};

struct llama_grammar {
    // maintain a list of llama_tokens and their positions in the trigger_buffer
    using token_pos = std::pair<llama_token, std::pair<size_t, size_t>>;

    // note: allow null vocab for testing (not great)
    const llama_vocab * vocab;

    const llama_grammar_rules  rules;  // TODO: shared ptr
          llama_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
    // (useful e.g. for tool_choice=required)
    bool                     lazy             = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Will be cleared once trigger is found.
    std::vector<token_pos>   trigger_buffer_positions; // Tokens buffered by lazy grammar. Used to replay when a trigger is found.
    std::vector<llama_token> trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
    std::vector<llama_grammar_trigger_pattern>
                             trigger_patterns;         // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                                                       // string, and the grammar will be given the string from the first match group onwards.

    // mutable because apply_impl takes const llama_grammar &.
    mutable std::unordered_map<
        llama_grammar_stack,
        std::unordered_map<llama_grammar_candidates, llama_grammar_candidates,
                           llama_grammar_candidates_hash, llama_grammar_candidates_eq>,
        llama_grammar_stack_hash, llama_grammar_stack_eq> memo_cache;

};

//
// internal API
//

// note: needed for tests (not great)
struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
        const llama_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);

void llama_grammar_free_impl(struct llama_grammar * grammar);

struct llama_grammar * llama_grammar_clone_impl(const struct llama_grammar & grammar);

// TODO: move the API below as member functions of llama_grammar
void llama_grammar_apply_impl(
        const struct llama_grammar & grammar,
            llama_token_data_array * cur_p);

void llama_grammar_accept_impl(
              struct llama_grammar & grammar,
                       llama_token   token);

void llama_grammar_accept_str(
              struct llama_grammar & grammar,
                 const std::string & piece);

void llama_grammar_accept_token(
              struct llama_grammar & grammar,
                       llama_token   token,
                 const std::string & piece);
