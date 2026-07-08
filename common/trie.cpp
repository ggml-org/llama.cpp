#include "trie.h"

#include "unicode.h"

#include <deque>

common_trie::match_result common_trie::check_at(std::string_view sv, size_t start_pos) const {
    size_t current = 0; // Start at root
    size_t pos = start_pos;

    // LOG_DBG("%s: checking at pos %zu, sv='%s'\n", __func__, start_pos, std::string(sv).c_str());

    while (pos < sv.size()) {
        auto result = common_parse_utf8_codepoint(sv, pos);
        if (result.status != utf8_parse_result::SUCCESS) {
            break;
        }

        auto it = nodes[current].children.find(result.codepoint);
        if (it == nodes[current].children.end()) {
            // Can't continue matching
            return match_result{match_result::NO_MATCH};
        }

        current = it->second;
        pos += result.bytes_consumed;

        // Check if we've matched a complete word
        if (nodes[current].is_word) {
            return match_result{match_result::COMPLETE_MATCH};
        }
    }

    // Reached end of input while still in the trie (not at root)
    if (current != 0) {
        // We're in the middle of a potential match
        return match_result{match_result::PARTIAL_MATCH};
    }

    // Reached end at root (no match)
    return match_result{match_result::NO_MATCH};
}

void common_trie::insert(const std::string & word) {
    std::vector<uint32_t> symbols;
    size_t pos = 0;
    while (pos < word.length()) {
        auto result = common_parse_utf8_codepoint(word, pos);
        if (result.status != utf8_parse_result::SUCCESS) {
            break;
        }

        symbols.push_back(result.codepoint);
        pos += result.bytes_consumed;
    }
    insert(symbols);
}

void common_trie::insert(const std::vector<uint32_t> & symbols) {
    size_t current = 0;
    for (uint32_t ch : symbols) {
        auto it = nodes[current].children.find(ch);
        if (it == nodes[current].children.end()) {
            size_t child = create_node();
            nodes[current].children[ch] = child;
            current = child;
        } else {
            current = it->second;
        }
    }
    nodes[current].is_word = true;
}

common_aho_corasick::common_aho_corasick(common_trie trie) : t(std::move(trie)) {
    const auto & nodes = t.nodes;
    const size_t n = nodes.size();

    fail.assign(n, 0);
    order.reserve(n);

    std::deque<size_t> queue{ 0 };
    while (!queue.empty()) {
        size_t u = queue.front();
        queue.pop_front();
        order.push_back(u);
        for (const auto & [ch, v] : nodes[u].children) {
            if (u != 0) {
                size_t f = fail[u];
                while (f && nodes[f].children.find(ch) == nodes[f].children.end()) {
                    f = fail[f];
                }
                auto it = nodes[f].children.find(ch);
                fail[v] = (it != nodes[f].children.end() && it->second != v) ? it->second : 0;
            }
            queue.push_back(v);
        }
    }

    terminal.assign(n, false);
    for (size_t u : order) {
        terminal[u] = nodes[u].is_word || (u != 0 && terminal[fail[u]]);
    }

    for (const auto & node : nodes) {
        for (const auto & [ch, v] : node.children) {
            alphabet.insert(ch);
        }
    }
}

size_t common_aho_corasick::next(size_t state, uint32_t ch) const {
    const auto & nodes = t.nodes;
    while (state && nodes[state].children.find(ch) == nodes[state].children.end()) {
        state = fail[state];
    }
    auto it = nodes[state].children.find(ch);
    return it != nodes[state].children.end() ? it->second : 0;
}
