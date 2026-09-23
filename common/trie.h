#pragma once

#include "unicode.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

// Trie for matching multiple literals.
// This is used in common_peg_until_parser and to build a GBNF exclusion grammar
struct common_trie {
    // A trie symbol, either a Unicode codepoint or a token id. Codepoints sort before tokens.
    struct symbol {
        enum kind_type : uint8_t { CODEPOINT, TOKEN };

        kind_type kind  = CODEPOINT;
        uint32_t  value = 0;

        static symbol codepoint(uint32_t cpt) { return { CODEPOINT, cpt }; }
        static symbol token(int32_t id) { return { TOKEN, (uint32_t) id }; }

        bool is_token() const { return kind == TOKEN; }

        bool operator<(const symbol & other) const {
            return kind != other.kind ? kind < other.kind : value < other.value;
        }
        bool operator==(const symbol & other) const {
            return kind == other.kind && value == other.value;
        }
    };

    // The symbol read from the input at some position, and how many bytes it covers
    struct step {
        enum utf8_parse_result::status status;
        common_trie::symbol            symbol;
        size_t                         bytes_consumed;
    };

    struct node {
        std::map<symbol, size_t> children;
        int32_t pattern = -1;                 // index of the pattern ending at this node, -1 if none
    };

    std::vector<node> nodes;

    // every token id used by a pattern, the input is split so these match as single symbols
    std::unordered_set<int32_t> tokens;

    common_trie() {
        create_node(); // root node
    }

    common_trie(const std::vector<std::string> & words) : common_trie() {
        for (const auto & w : words) {
            insert(w);
        }
    }

    enum match_result { NO_MATCH, PARTIAL_MATCH, COMPLETE_MATCH };

    // Read the symbol at pos. token_map is empty or holds, for each byte of sv, the id of the token that starts
    // there or a negative value. A token in the trie is read as one symbol that runs to the next token or the end
    // of the input, and anything else is read as a UTF-8 codepoint.
    step next_symbol(std::string_view sv, const std::vector<int32_t> & token_map, size_t pos) const;

    // Check if a delimiter starts at the given position
    match_result check_at(std::string_view sv, size_t start_pos) const;
    match_result check_at(std::string_view sv, const std::vector<int32_t> & token_map, size_t start_pos) const;

    // Insert a word as a sequence of Unicode codepoints, returns its pattern index
    int32_t insert(const std::string & word);

    // Insert a symbol sequence, returns its pattern index (insertion order,
    // duplicates keep the first index)
    int32_t insert(const std::vector<symbol> & symbols);

  private:
    int32_t n_patterns = 0;

    size_t create_node() {
        size_t index = nodes.size();
        nodes.emplace_back();
        return index;
    }
};

// Aho-Corasick automaton
struct common_aho_corasick {
    common_trie          t;
    std::vector<size_t>  fail;     // failure links
    std::vector<size_t>  order;    // states in BFS order
    std::vector<int32_t> match;    // longest pattern ending at each state (directly or via a suffix link), -1 if none
    std::set<common_trie::symbol> alphabet; // every symbol with a transition

    common_aho_corasick(common_trie trie);

    common_aho_corasick(const std::vector<std::string> & strings)
        : common_aho_corasick(common_trie(strings)) {}

    size_t num_states()          const { return t.nodes.size(); }
    bool   is_terminal(size_t s) const { return match[s] >= 0; }

    // index of the longest pattern ending at this state, -1 if none
    int32_t match_pattern(size_t s) const { return match[s]; }

    // follow failure links until a transition on `ch` exists.
    size_t next(size_t state, common_trie::symbol ch) const;
};
