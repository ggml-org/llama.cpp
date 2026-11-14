#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// TTS normalization patterns: a small, fully specified matching language for TTS text normalization templates
//
// it is NOT a regex engine: the semantics are defined by llama.cpp (see docs/tts-norm-patterns.md), so that this
// implementation and the python reference (scripts/chat_tts_norm.py) give the same result for every input
//
// two functions are exposed to task templates (e.g. "tts") via common_chat_tts_norm_add_builtins():
//   tts_norm_scan(text, rules)        -> list of segments {rule, text, groups}
//   tts_norm_sub(text, pattern, repl) -> string

namespace jinja {
struct context;
}

struct tts_norm_pattern_node;

class common_tts_norm_pattern {
public:
    // throws std::runtime_error on invalid pattern
    explicit common_tts_norm_pattern(const std::string & src);
    ~common_tts_norm_pattern();
    common_tts_norm_pattern(common_tts_norm_pattern &&) noexcept;
    common_tts_norm_pattern & operator=(common_tts_norm_pattern &&) noexcept;

    // match anchored at codepoint index pos; on success, end is the codepoint index after the match
    // groups[i] holds [start, end) codepoint indices of capture i+1, or {-1, -1} if the group did not participate
    bool match_at(const std::vector<uint32_t> & cpts, size_t pos, size_t & end, std::vector<std::pair<int64_t, int64_t>> & groups) const;

    size_t n_groups() const;

private:
    std::unique_ptr<tts_norm_pattern_node> root;
    size_t n_caps = 0;
    bool icase = false;
};

// segment produced by common_tts_norm_scan(); byte offsets into the input text
struct common_tts_norm_segment {
    int    rule  = -1; // index of the matched rule, -1 for unmatched text
    size_t start = 0;
    size_t end   = 0;
    std::vector<std::pair<size_t, size_t>> groups; // byte offsets; {npos, npos} for groups that did not participate
};

// split text into matched and unmatched segments
// at each position, rules are tried in order and the first non-empty match wins; otherwise one codepoint is skipped
std::vector<common_tts_norm_segment> common_tts_norm_scan(const std::string & text, const std::vector<const common_tts_norm_pattern *> & rules);

// replace every non-overlapping match of pattern with repl
// repl references: %0 (whole match), %1..%9 (groups), %% (literal percent)
std::string common_tts_norm_sub(const std::string & text, const common_tts_norm_pattern & pattern, const std::string & repl);

// add tts_norm_scan() and tts_norm_sub() to the context of a task template
// they are not jinja global builtins on purpose: chat templates are not affected
void common_chat_tts_norm_add_builtins(jinja::context & ctx);
