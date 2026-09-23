#include "trie.h"

#include "unicode.h"

#include <deque>

common_trie::step common_trie::next_symbol(std::string_view sv, const std::vector<int32_t> & token_map, size_t pos) const {
    if (!tokens.empty() && pos < token_map.size()) {
        const int32_t id = token_map[pos];
        if (id >= 0 && tokens.find(id) != tokens.end()) {
            size_t end = pos + 1;
            while (end < token_map.size() && token_map[end] < 0) {
                ++end;
            }
            return { utf8_parse_result::SUCCESS, symbol::token(id), end - pos };
        }
    }

    auto result = common_parse_utf8_codepoint(sv, pos);
    return { result.status, symbol::codepoint(result.codepoint), result.bytes_consumed };
}

common_trie::match_result common_trie::check_at(std::string_view sv, size_t start_pos) const {
    return check_at(sv, {}, start_pos);
}

common_trie::match_result common_trie::check_at(std::string_view sv, const std::vector<int32_t> & token_map, size_t start_pos) const {
    size_t current = 0; // Start at root
    size_t pos = start_pos;

    // LOG_DBG("%s: checking at pos %zu, sv='%s'\n", __func__, start_pos, std::string(sv).c_str());

    while (pos < sv.size()) {
        auto next = next_symbol(sv, token_map, pos);
        if (next.status != utf8_parse_result::SUCCESS) {
            break;
        }

        auto it = nodes[current].children.find(next.symbol);
        if (it == nodes[current].children.end()) {
            // Can't continue matching
            return match_result{match_result::NO_MATCH};
        }

        current = it->second;
        pos += next.bytes_consumed;

        // Check if we've matched a complete word
        if (nodes[current].pattern >= 0) {
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

int32_t common_trie::insert(const std::string & word) {
    std::vector<symbol> symbols;
    size_t pos = 0;
    while (pos < word.length()) {
        auto result = common_parse_utf8_codepoint(word, pos);
        if (result.status != utf8_parse_result::SUCCESS) {
            break;
        }

        symbols.push_back(symbol::codepoint(result.codepoint));
        pos += result.bytes_consumed;
    }
    return insert(symbols);
}

int32_t common_trie::insert(const std::vector<symbol> & symbols) {
    size_t current = 0;
    for (symbol ch : symbols) {
        if (ch.is_token()) {
            tokens.insert((int32_t) ch.value);
        }
        auto it = nodes[current].children.find(ch);
        if (it == nodes[current].children.end()) {
            size_t child = create_node();
            nodes[current].children[ch] = child;
            current = child;
        } else {
            current = it->second;
        }
    }
    if (nodes[current].pattern < 0) {
        nodes[current].pattern = n_patterns++;
    }
    return nodes[current].pattern;
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

    // fail[u] points to a strictly shorter suffix, so the first pattern found on
    // the fail chain (including u itself) is the longest pattern ending at u
    match.assign(n, -1);
    for (size_t u : order) {
        match[u] = nodes[u].pattern >= 0 ? nodes[u].pattern : (u != 0 ? match[fail[u]] : -1);
    }

    for (const auto & node : nodes) {
        for (const auto & [ch, v] : node.children) {
            alphabet.insert(ch);
        }
    }
}

size_t common_aho_corasick::next(size_t state, common_trie::symbol ch) const {
    const auto & nodes = t.nodes;
    while (state && nodes[state].children.find(ch) == nodes[state].children.end()) {
        state = fail[state];
    }
    auto it = nodes[state].children.find(ch);
    return it != nodes[state].children.end() ? it->second : 0;
}
