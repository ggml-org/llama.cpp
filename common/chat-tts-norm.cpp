#include "chat-tts-norm.h"

#include "jinja/runtime.h"
#include "jinja/value.h"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

// TTS normalization patterns, see docs/tts-norm-patterns.md for the specification
// keep this file in sync with the python reference: scripts/chat_tts_norm.py

using namespace jinja;

//
// character classes
// they are defined with explicit codepoint ranges, independent of any unicode database version,
// so that the C++ and python implementations agree on every codepoint
//

struct tts_norm_range {
    uint32_t lo;
    uint32_t hi;
};

static const tts_norm_range TTS_NORM_SPACE[] = {
    {0x0009, 0x000D}, {0x0020, 0x0020}, {0x0085, 0x0085}, {0x00A0, 0x00A0}, {0x1680, 0x1680},
    {0x2000, 0x200A}, {0x2028, 0x2029}, {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000},
};

static const tts_norm_range TTS_NORM_PUNCT[] = {
    {0x0021, 0x002F}, {0x003A, 0x0040}, {0x005B, 0x0060}, {0x007B, 0x007E},
    {0x00A1, 0x00A9}, {0x00AB, 0x00B4}, {0x00B6, 0x00B9}, {0x00BB, 0x00BF}, {0x00D7, 0x00D7}, {0x00F7, 0x00F7},
    {0x2010, 0x2027}, {0x2030, 0x205E}, {0x20A0, 0x20CF}, {0x2100, 0x218F}, {0x2190, 0x2BFF},
    {0x3001, 0x3003}, {0x3008, 0x3020}, {0x3030, 0x3030}, {0x303D, 0x303D}, {0x30FB, 0x30FB},
    {0xFE10, 0xFE19}, {0xFE30, 0xFE4F}, {0xFE50, 0xFE6B},
    {0xFF01, 0xFF0F}, {0xFF1A, 0xFF20}, {0xFF3B, 0xFF40}, {0xFF5B, 0xFF65},
    {0x1F000, 0x1FAFF},
};

static const tts_norm_range TTS_NORM_CONTROL[] = {
    {0x0000, 0x001F}, {0x007F, 0x009F}, {0x200B, 0x200F}, {0x2060, 0x206F}, {0xFEFF, 0xFEFF},
};

static const tts_norm_range TTS_NORM_HAN[] = {
    {0x2E80, 0x2FDF}, {0x3005, 0x3005}, {0x3007, 0x3007}, {0x3021, 0x3029}, {0x3038, 0x303B},
    {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xF900, 0xFAFF}, {0x20000, 0x323AF},
};

template<size_t N>
static bool tts_norm_in(const tts_norm_range (&ranges)[N], uint32_t c) {
    for (const auto & r : ranges) {
        if (c >= r.lo && c <= r.hi) {
            return true;
        }
    }
    return false;
}

static bool tts_norm_is_alpha(uint32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
        return true;
    }
    if (c == 0xAA || c == 0xB5 || c == 0xBA) {
        return true;
    }
    return c >= 0xC0 && !tts_norm_in(TTS_NORM_SPACE, c) && !tts_norm_in(TTS_NORM_PUNCT, c) && !tts_norm_in(TTS_NORM_CONTROL, c);
}

// kind is one of "dspawulhc"
static bool tts_norm_class_match(char kind, uint32_t c) {
    switch (kind) {
        case 'd': return c >= '0' && c <= '9';
        case 's': return tts_norm_in(TTS_NORM_SPACE, c);
        case 'p': return tts_norm_in(TTS_NORM_PUNCT, c);
        case 'a': return tts_norm_is_alpha(c);
        case 'w': return tts_norm_is_alpha(c) || (c >= '0' && c <= '9');
        case 'u': return c >= 'A' && c <= 'Z';
        case 'l': return c >= 'a' && c <= 'z';
        case 'h': return tts_norm_in(TTS_NORM_HAN, c);
        case 'c': return tts_norm_in(TTS_NORM_CONTROL, c);
    }
    return false;
}

static bool tts_norm_is_class_letter(uint32_t c) {
    if (c >= 'A' && c <= 'Z') {
        c += 'a' - 'A';
    }
    return c < 0x80 && std::string("dspawulhc").find(static_cast<char>(c)) != std::string::npos;
}

static uint32_t tts_norm_ascii_lower(uint32_t c) {
    return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

static uint32_t tts_norm_ascii_swapcase(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
    if (c >= 'a' && c <= 'z') return c - ('a' - 'A');
    return c;
}

static bool tts_norm_is_ascii_alnum(uint32_t c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// invalid bytes are decoded as U+FFFD, one per byte; offs[i] is the byte offset of codepoint i, offs[n] = text size
static void tts_norm_decode_utf8(const std::string & s, std::vector<uint32_t> & cpts, std::vector<size_t> & offs) {
    cpts.clear();
    offs.clear();
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        offs.push_back(i);
        const unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        size_t len = 0;
        if (c < 0x80) {
            cp = c; len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F; len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F; len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07; len = 4;
        }
        bool ok = len > 0 && i + len <= n;
        for (size_t k = 1; ok && k < len; k++) {
            const unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) {
                ok = false;
            } else {
                cp = (cp << 6) | (cc & 0x3F);
            }
        }
        if (!ok) {
            cpts.push_back(0xFFFD);
            i += 1;
        } else {
            cpts.push_back(cp);
            i += len;
        }
    }
    offs.push_back(n);
}

//
// AST
//

enum class tts_norm_op {
    LITERAL,    // one codepoint
    SET,        // one codepoint from a set; class escapes like %d are sets with a single class
    ANY,        // any codepoint
    START,      // ^
    END,        // $
    BOUNDARY,   // %b
    GROUP,      // (...), (?:...), (?<name>...)
    LOOKAHEAD,  // (?=...), (?!...)
    LOOKBEHIND, // (?<=c), (?<!c) - exactly one codepoint
};

struct tts_norm_class {
    char kind;
    bool neg;
};

struct tts_norm_set {
    bool neg = false;
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    std::vector<tts_norm_class> classes;
};

struct tts_norm_item;
using tts_norm_seq = std::vector<tts_norm_item>;

struct tts_norm_pattern_node {
    tts_norm_op op = tts_norm_op::GROUP;
    uint32_t cp = 0;           // LITERAL
    tts_norm_set set;               // SET
    std::vector<tts_norm_seq> alts; // GROUP, LOOKAHEAD; LOOKBEHIND holds a single item
    int capture = -1;          // GROUP: capture index, -1 for non-capturing
    bool neg = false;          // LOOKAHEAD, LOOKBEHIND
};

static constexpr int TTS_NORM_INF = INT_MAX;
static constexpr int TTS_NORM_MAX_REPEAT = 1000;
static constexpr size_t TTS_NORM_MAX_PATTERN_LEN = 4096;

struct tts_norm_item {
    tts_norm_pattern_node atom;
    int min = 1;
    int max = 1;
};

static bool tts_norm_is_zero_width(tts_norm_op op) {
    return op == tts_norm_op::START || op == tts_norm_op::END || op == tts_norm_op::BOUNDARY
        || op == tts_norm_op::LOOKAHEAD || op == tts_norm_op::LOOKBEHIND;
}

//
// parser
//

struct tts_norm_parser {
    std::vector<uint32_t> s;
    size_t p = 0;
    size_t & n_caps;

    tts_norm_parser(std::vector<uint32_t> src, size_t & n_caps) : s(std::move(src)), n_caps(n_caps) {}

    [[noreturn]] void fail(const std::string & msg) const {
        throw std::runtime_error("invalid TTS norm pattern: " + msg + " at position " + std::to_string(p));
    }

    bool eof() const { return p >= s.size(); }
    uint32_t peek() const { return s[p]; }
    bool peek_is(uint32_t c) const { return !eof() && s[p] == c; }

    void expect(uint32_t c, const char * what) {
        if (!peek_is(c)) {
            fail(std::string("expected ") + what);
        }
        p++;
    }

    std::vector<tts_norm_seq> parse_alts() {
        std::vector<tts_norm_seq> alts;
        alts.push_back(parse_seq());
        while (peek_is('|')) {
            p++;
            alts.push_back(parse_seq());
        }
        return alts;
    }

    tts_norm_seq parse_seq() {
        tts_norm_seq seq;
        while (!eof() && peek() != '|' && peek() != ')') {
            tts_norm_item item;
            item.atom = parse_atom();
            parse_quantifier(item);
            seq.push_back(std::move(item));
        }
        return seq;
    }

    int parse_int() {
        if (eof() || peek() < '0' || peek() > '9') {
            fail("expected number in quantifier");
        }
        int v = 0;
        while (!eof() && peek() >= '0' && peek() <= '9') {
            v = v * 10 + static_cast<int>(peek() - '0');
            if (v > TTS_NORM_MAX_REPEAT) {
                fail("repetition count exceeds " + std::to_string(TTS_NORM_MAX_REPEAT));
            }
            p++;
        }
        return v;
    }

    void parse_quantifier(tts_norm_item & item) {
        if (eof()) {
            return;
        }
        const uint32_t c = peek();
        if (c != '?' && c != '*' && c != '+' && c != '{') {
            return;
        }
        if (tts_norm_is_zero_width(item.atom.op)) {
            fail("zero-width assertion cannot be repeated");
        }
        p++;
        if (c == '?') {
            item.min = 0; item.max = 1;
        } else if (c == '*') {
            item.min = 0; item.max = TTS_NORM_INF;
        } else if (c == '+') {
            item.min = 1; item.max = TTS_NORM_INF;
        } else {
            item.min = parse_int();
            item.max = item.min;
            if (peek_is(',')) {
                p++;
                item.max = peek_is('}') ? TTS_NORM_INF : parse_int();
            }
            expect('}', "'}' to close quantifier");
            if (item.max < item.min) {
                fail("quantifier max is smaller than min");
            }
        }
        if (!eof() && (peek() == '?' || peek() == '*' || peek() == '+' || peek() == '{')) {
            fail("multiple quantifiers");
        }
    }

    tts_norm_pattern_node parse_group() {
        // '(' already consumed
        tts_norm_pattern_node node;
        node.op = tts_norm_op::GROUP;
        if (peek_is('?')) {
            p++;
            if (eof()) {
                fail("unterminated group");
            }
            const uint32_t c = peek();
            if (c == ':') {
                p++;
            } else if (c == '=' || c == '!') {
                p++;
                node.op  = tts_norm_op::LOOKAHEAD;
                node.neg = c == '!';
            } else if (c == '<' && p + 1 < s.size() && (s[p + 1] == '=' || s[p + 1] == '!')) {
                node.op  = tts_norm_op::LOOKBEHIND;
                node.neg = s[p + 1] == '!';
                p += 2;
                tts_norm_item item;
                if (eof() || peek() == ')') {
                    fail("lookbehind must contain exactly one codepoint matcher");
                }
                item.atom = parse_atom();
                if (item.atom.op != tts_norm_op::LITERAL && item.atom.op != tts_norm_op::SET && item.atom.op != tts_norm_op::ANY) {
                    fail("lookbehind must contain exactly one codepoint matcher");
                }
                if (!peek_is(')')) {
                    fail("lookbehind must contain exactly one codepoint matcher");
                }
                p++;
                node.alts.push_back(tts_norm_seq{});
                node.alts[0].push_back(std::move(item));
                return node;
            } else if (c == 'i') {
                fail("(?i) is only allowed at the start of the pattern");
            } else {
                fail("unknown group type");
            }
        } else {
            node.capture = static_cast<int>(n_caps++);
        }
        node.alts = parse_alts();
        expect(')', "')' to close group");
        return node;
    }

    // after '%' inside or outside of a set
    // returns true if it is a class (stored in cls), false if it is a literal (stored in lit)
    bool parse_escape(bool in_set, tts_norm_class & cls, uint32_t & lit, bool & boundary) {
        boundary = false;
        if (eof()) {
            fail("dangling '%'");
        }
        const uint32_t e = s[p++];
        if (tts_norm_is_class_letter(e)) {
            cls.kind = static_cast<char>(tts_norm_ascii_lower(e));
            cls.neg  = e >= 'A' && e <= 'Z';
            return true;
        }
        if (e == 'b' && !in_set) {
            boundary = true;
            return false;
        }
        if (tts_norm_is_ascii_alnum(e)) {
            p--;
            fail(std::string("unknown escape '%") + static_cast<char>(e) + "'");
        }
        lit = e;
        return false;
    }

    tts_norm_pattern_node parse_set() {
        // '[' already consumed
        tts_norm_pattern_node node;
        node.op = tts_norm_op::SET;
        if (peek_is('^')) {
            node.set.neg = true;
            p++;
        }
        bool empty = true;
        while (true) {
            if (eof()) {
                fail("unterminated set");
            }
            if (peek() == ']') {
                p++;
                break;
            }
            empty = false;
            uint32_t lo = 0;
            if (peek() == '%') {
                p++;
                tts_norm_class cls{};
                bool boundary = false;
                if (parse_escape(true, cls, lo, boundary)) {
                    node.set.classes.push_back(cls);
                    continue;
                }
            } else {
                lo = s[p++];
            }
            uint32_t hi = lo;
            if (peek_is('-') && p + 1 < s.size() && s[p + 1] != ']') {
                p++;
                if (peek() == '%') {
                    p++;
                    tts_norm_class cls{};
                    bool boundary = false;
                    if (parse_escape(true, cls, hi, boundary)) {
                        fail("character class cannot be used in a range");
                    }
                } else {
                    hi = s[p++];
                }
                if (hi < lo) {
                    fail("invalid range in set");
                }
            }
            node.set.ranges.emplace_back(lo, hi);
        }
        if (empty) {
            fail("empty set");
        }
        return node;
    }

    tts_norm_pattern_node parse_atom() {
        const uint32_t c = s[p++];
        tts_norm_pattern_node node;
        switch (c) {
            case '(':
                return parse_group();
            case '[':
                return parse_set();
            case '.':
                node.op = tts_norm_op::ANY;
                return node;
            case '^':
                node.op = tts_norm_op::START;
                return node;
            case '$':
                node.op = tts_norm_op::END;
                return node;
            case '%': {
                tts_norm_class cls{};
                uint32_t lit = 0;
                bool boundary = false;
                if (parse_escape(false, cls, lit, boundary)) {
                    node.op = tts_norm_op::SET;
                    node.set.classes.push_back(cls);
                } else if (boundary) {
                    node.op = tts_norm_op::BOUNDARY;
                } else {
                    node.op = tts_norm_op::LITERAL;
                    node.cp = lit;
                }
                return node;
            }
            case '?':
            case '*':
            case '+':
            case '{':
                p--;
                fail("nothing to repeat");
            case ']':
            case '}':
                p--;
                fail(std::string("unescaped '") + static_cast<char>(c) + "', use '%" + static_cast<char>(c) + "'");
            default:
                node.op = tts_norm_op::LITERAL;
                node.cp = c;
                return node;
        }
    }
};

//
// matcher
// PEG-style: alternatives are ordered (the first one that matches wins) and quantifiers are greedy and possessive
// (they never give back what they consumed), so there is no backtracking
//

struct tts_norm_matcher {
    const std::vector<uint32_t> & t;
    bool icase;
    std::vector<std::pair<int64_t, int64_t>> caps;

    bool is_w(size_t i) const {
        return i < t.size() && tts_norm_class_match('w', t[i]);
    }

    bool set_match(const tts_norm_set & st, uint32_t c) const {
        auto test = [&](uint32_t x) {
            for (const auto & r : st.ranges) {
                if (x >= r.first && x <= r.second) {
                    return true;
                }
            }
            for (const auto & k : st.classes) {
                if (tts_norm_class_match(k.kind, x) != k.neg) {
                    return true;
                }
            }
            return false;
        };
        bool m = test(c);
        if (!m && icase) {
            const uint32_t o = tts_norm_ascii_swapcase(c);
            if (o != c) {
                m = test(o);
            }
        }
        return m != st.neg;
    }

    bool match_alts(const std::vector<tts_norm_seq> & alts, size_t i, size_t & end) {
        for (const auto & seq : alts) {
            auto saved = caps;
            if (match_seq(seq, i, end)) {
                return true;
            }
            caps = std::move(saved);
        }
        return false;
    }

    bool match_seq(const tts_norm_seq & seq, size_t i, size_t & end) {
        size_t pos = i;
        for (const auto & it : seq) {
            int count = 0;
            while (count < it.max) {
                const bool is_group = it.atom.op == tts_norm_op::GROUP;
                std::vector<std::pair<int64_t, int64_t>> saved;
                if (is_group) {
                    saved = caps;
                }
                size_t e = pos;
                if (!match_once(it.atom, pos, e)) {
                    if (is_group) {
                        caps = std::move(saved);
                    }
                    break;
                }
                if (e == pos) {
                    // an empty iteration would repeat forever, it satisfies any remaining minimum
                    count = std::max(count + 1, it.min);
                    break;
                }
                pos = e;
                count++;
            }
            if (count < it.min) {
                return false;
            }
        }
        end = pos;
        return true;
    }

    bool match_once(const tts_norm_pattern_node & node, size_t i, size_t & end) {
        const size_t n = t.size();
        switch (node.op) {
            case tts_norm_op::LITERAL:
                if (i < n && (t[i] == node.cp || (icase && tts_norm_ascii_lower(t[i]) == tts_norm_ascii_lower(node.cp)))) {
                    end = i + 1;
                    return true;
                }
                return false;
            case tts_norm_op::SET:
                if (i < n && set_match(node.set, t[i])) {
                    end = i + 1;
                    return true;
                }
                return false;
            case tts_norm_op::ANY:
                if (i < n) {
                    end = i + 1;
                    return true;
                }
                return false;
            case tts_norm_op::START:
                end = i;
                return i == 0;
            case tts_norm_op::END:
                end = i;
                return i == n;
            case tts_norm_op::BOUNDARY:
                end = i;
                return (i > 0 && is_w(i - 1)) != is_w(i);
            case tts_norm_op::GROUP: {
                size_t e = i;
                if (!match_alts(node.alts, i, e)) {
                    return false;
                }
                if (node.capture >= 0) {
                    caps[node.capture] = {static_cast<int64_t>(i), static_cast<int64_t>(e)};
                }
                end = e;
                return true;
            }
            case tts_norm_op::LOOKAHEAD: {
                // captures made inside a lookahead are discarded
                auto saved = caps;
                size_t e = i;
                const bool ok = match_alts(node.alts, i, e);
                caps = std::move(saved);
                end = i;
                return ok != node.neg;
            }
            case tts_norm_op::LOOKBEHIND: {
                size_t e = 0;
                const bool ok = i > 0 && match_once(node.alts[0][0].atom, i - 1, e) && e == i;
                end = i;
                return ok != node.neg;
            }
        }
        return false;
    }
};

//
// common_tts_norm_pattern
//

common_tts_norm_pattern::common_tts_norm_pattern(const std::string & src) {
    if (src.size() > TTS_NORM_MAX_PATTERN_LEN) {
        throw std::runtime_error("invalid TTS norm pattern: pattern is longer than " + std::to_string(TTS_NORM_MAX_PATTERN_LEN) + " bytes");
    }
    std::vector<uint32_t> cpts;
    std::vector<size_t> offs;
    tts_norm_decode_utf8(src, cpts, offs);

    tts_norm_parser parser(std::move(cpts), n_caps);
    static const std::vector<uint32_t> icase_prefix = {'(', '?', 'i', ')'};
    if (parser.s.size() >= 4 && std::equal(icase_prefix.begin(), icase_prefix.end(), parser.s.begin())) {
        icase = true;
        parser.p = 4;
    }

    root = std::make_unique<tts_norm_pattern_node>();
    root->op   = tts_norm_op::GROUP;
    root->alts = parser.parse_alts();
    if (!parser.eof()) {
        parser.fail("unbalanced ')'");
    }
}

common_tts_norm_pattern::~common_tts_norm_pattern() = default;
common_tts_norm_pattern::common_tts_norm_pattern(common_tts_norm_pattern &&) noexcept = default;
common_tts_norm_pattern & common_tts_norm_pattern::operator=(common_tts_norm_pattern &&) noexcept = default;

bool common_tts_norm_pattern::match_at(const std::vector<uint32_t> & cpts, size_t pos, size_t & end, std::vector<std::pair<int64_t, int64_t>> & groups) const {
    tts_norm_matcher m{cpts, icase, std::vector<std::pair<int64_t, int64_t>>(n_caps, {-1, -1})};
    size_t e = pos;
    if (!m.match_alts(root->alts, pos, e)) {
        return false;
    }
    end = e;
    groups = std::move(m.caps);
    return true;
}

size_t common_tts_norm_pattern::n_groups() const {
    return n_caps;
}

//
// scan / sub
//

std::vector<common_tts_norm_segment> common_tts_norm_scan(const std::string & text, const std::vector<const common_tts_norm_pattern *> & rules) {
    std::vector<uint32_t> cpts;
    std::vector<size_t> offs;
    tts_norm_decode_utf8(text, cpts, offs);

    std::vector<common_tts_norm_segment> out;
    const size_t n = cpts.size();
    size_t i = 0;
    size_t pending = 0; // start of the current unmatched run
    std::vector<std::pair<int64_t, int64_t>> groups;

    while (i < n) {
        bool matched = false;
        for (size_t r = 0; r < rules.size(); r++) {
            size_t end = i;
            if (rules[r]->match_at(cpts, i, end, groups) && end > i) {
                if (pending < i) {
                    out.push_back({-1, offs[pending], offs[i], {}});
                }
                common_tts_norm_segment seg;
                seg.rule  = static_cast<int>(r);
                seg.start = offs[i];
                seg.end   = offs[end];
                for (const auto & g : groups) {
                    if (g.first < 0) {
                        seg.groups.emplace_back(std::string::npos, std::string::npos);
                    } else {
                        seg.groups.emplace_back(offs[g.first], offs[g.second]);
                    }
                }
                out.push_back(std::move(seg));
                i = end;
                pending = i;
                matched = true;
                break;
            }
        }
        if (!matched) {
            i++;
        }
    }
    if (pending < n) {
        out.push_back({-1, offs[pending], offs[n], {}});
    }
    return out;
}

struct tts_norm_repl_piece {
    std::string literal;
    int group = -1; // 0 = whole match, 1.. = groups
};

static std::vector<tts_norm_repl_piece> tts_norm_parse_repl(const std::string & repl, const common_tts_norm_pattern & pattern) {
    std::vector<tts_norm_repl_piece> pieces;
    std::string lit;
    auto flush = [&]() {
        if (!lit.empty()) {
            pieces.push_back({lit, -1});
            lit.clear();
        }
    };
    for (size_t i = 0; i < repl.size(); i++) {
        if (repl[i] != '%') {
            lit += repl[i];
            continue;
        }
        if (i + 1 >= repl.size()) {
            throw std::runtime_error("invalid TTS norm replacement: dangling '%'");
        }
        const char c = repl[++i];
        if (c == '%') {
            lit += '%';
        } else if (c >= '0' && c <= '9') {
            const int g = c - '0';
            if (static_cast<size_t>(g) > pattern.n_groups()) {
                throw std::runtime_error("invalid TTS norm replacement: group %" + std::string(1, c) + " does not exist");
            }
            flush();
            pieces.push_back({"", g});
        } else {
            throw std::runtime_error("invalid TTS norm replacement: unknown escape '%" + std::string(1, c) + "'");
        }
    }
    flush();
    return pieces;
}

std::string common_tts_norm_sub(const std::string & text, const common_tts_norm_pattern & pattern, const std::string & repl) {
    const auto pieces = tts_norm_parse_repl(repl, pattern);
    std::string out;
    for (const auto & seg : common_tts_norm_scan(text, {&pattern})) {
        if (seg.rule < 0) {
            out.append(text, seg.start, seg.end - seg.start);
            continue;
        }
        for (const auto & piece : pieces) {
            if (piece.group < 0) {
                out += piece.literal;
            } else if (piece.group == 0) {
                out.append(text, seg.start, seg.end - seg.start);
            } else {
                const auto & g = seg.groups[piece.group - 1];
                if (g.first != std::string::npos) {
                    out.append(text, g.first, g.second - g.first);
                }
            }
        }
    }
    return out;
}

//
// template builtins
//

// rules can be: a pattern string (rule name "match"), a dict {name: pattern}, or a list of [name, pattern] pairs
static std::vector<std::pair<std::string, std::string>> tts_norm_rules_from_value(const value & v) {
    std::vector<std::pair<std::string, std::string>> rules;
    if (is_val<value_string>(v)) {
        rules.emplace_back("match", v->as_string().str());
    } else if (is_val<value_object>(v)) {
        for (const auto & kv : v->as_ordered_object()) {
            if (!is_val<value_string>(kv.second)) {
                throw std::runtime_error("tts_norm_scan: rule pattern must be a string");
            }
            rules.emplace_back(kv.first->as_string().str(), kv.second->as_string().str());
        }
    } else if (is_val<value_array>(v)) {
        for (const auto & item : v->as_array()) {
            if (!is_val<value_array>(item) || item->as_array().size() != 2
                    || !is_val<value_string>(item->as_array()[0]) || !is_val<value_string>(item->as_array()[1])) {
                throw std::runtime_error("tts_norm_scan: each rule must be a [name, pattern] pair of strings");
            }
            rules.emplace_back(item->as_array()[0]->as_string().str(), item->as_array()[1]->as_string().str());
        }
    } else {
        throw std::runtime_error("tts_norm_scan: rules must be a string, a dict or a list of [name, pattern] pairs, got " + v->type());
    }
    return rules;
}

static value tts_norm_builtin_scan(const func_args & args) {
    args.ensure_count(2, 2);
    args.ensure_val<value_string>(args.get_pos(0));
    const string text_val = args.get_pos(0)->as_string();
    const std::string text = text_val.str();
    const bool is_input = text_val.all_parts_are_input();

    const auto rule_srcs = tts_norm_rules_from_value(args.get_pos(1));
    std::vector<common_tts_norm_pattern> patterns;
    patterns.reserve(rule_srcs.size());
    std::vector<const common_tts_norm_pattern *> ptrs;
    for (const auto & r : rule_srcs) {
        patterns.emplace_back(r.second);
    }
    for (const auto & p : patterns) {
        ptrs.push_back(&p);
    }

    auto mk_str = [&](size_t start, size_t end) -> value {
        return mk_val<value_string>(string(text.substr(start, end - start), is_input));
    };

    auto out = mk_val<value_array>();
    for (const auto & seg : common_tts_norm_scan(text, ptrs)) {
        auto groups = mk_val<value_array>();
        for (const auto & g : seg.groups) {
            // groups that did not participate are empty strings, so they render as nothing and are falsy
            groups->push_back(g.first == std::string::npos ? mk_str(0, 0) : mk_str(g.first, g.second));
        }
        auto obj = mk_val<value_object>();
        obj->insert("rule", seg.rule < 0 ? value(mk_val<value_none>()) : value(mk_val<value_string>(rule_srcs[seg.rule].first)));
        obj->insert("text", mk_str(seg.start, seg.end));
        obj->insert("groups", groups);
        out->push_back(obj);
    }
    return out;
}

static value tts_norm_builtin_sub(const func_args & args) {
    args.ensure_count(3, 3);
    args.ensure_vals<value_string, value_string, value_string>();
    const string text_val = args.get_pos(0)->as_string();
    const common_tts_norm_pattern pattern(args.get_pos(1)->as_string().str());
    const std::string res = common_tts_norm_sub(text_val.str(), pattern, args.get_pos(2)->as_string().str());
    return mk_val<value_string>(string(res, text_val.all_parts_are_input()));
}

void common_chat_tts_norm_add_builtins(context & ctx) {
    ctx.set_val("tts_norm_scan", mk_val<value_func>("tts_norm_scan", tts_norm_builtin_scan));
    ctx.set_val("tts_norm_sub",  mk_val<value_func>("tts_norm_sub",  tts_norm_builtin_sub));
}

