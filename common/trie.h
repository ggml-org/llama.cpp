#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

// Trie for matching multiple literals.
// This is used in common_peg_until_parser and to build a GBNF exclusion grammar
struct common_trie {
    struct node {
        std::map<uint32_t, size_t> children;  // Use uint32_t to store Unicode codepoints
        bool is_word;
    };

    std::vector<node> nodes;

    common_trie() {
        create_node(); // root node
    }

    common_trie(const std::vector<std::string> & words) : common_trie() {
        for (const auto & w : words) {
            insert(w);
        }
    }

    enum match_result { NO_MATCH, PARTIAL_MATCH, COMPLETE_MATCH };

    // Check if a delimiter starts at the given position
    match_result check_at(std::string_view sv, size_t start_pos) const;

    // Insert a word as a sequence of Unicode codepoints
    void insert(const std::string & word);

    // Insert a raw symbol sequence
    void insert(const std::vector<uint32_t> & symbols);

  private:
    size_t create_node() {
        size_t index = nodes.size();
        nodes.emplace_back();
        return index;
    }
};

// Aho-Corasick automaton
struct common_aho_corasick {
    common_trie         t;
    std::vector<size_t> fail;      // failure links
    std::vector<size_t> order;     // states in BFS order
    std::vector<bool>   terminal;  // match states (directly or via a suffix link)
    std::set<uint32_t>  alphabet;  // every character with a transition

    common_aho_corasick(common_trie trie);

    common_aho_corasick(const std::vector<std::string> & strings)
        : common_aho_corasick(common_trie(strings)) {}

    size_t num_states()          const { return t.nodes.size(); }
    bool   is_terminal(size_t s) const { return terminal[s]; }

    // follow failure links until a transition on `ch` exists.
    size_t next(size_t state, uint32_t ch) const;
};
