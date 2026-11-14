#include "m5_tokenizer.h"

#include "unicode_tables.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace m5 {
namespace {

constexpr char kMagic[5] = {'M', '5', 'T', 'K', '1'};

struct Reader {
    const unsigned char * p;
    size_t left;
    bool ok = true;

    uint32_t u32() {
        if (left < 4) { ok = false; return 0; }
        const uint32_t v = ((uint32_t) p[0] << 24) | ((uint32_t) p[1] << 16) |
                           ((uint32_t) p[2] << 8) | (uint32_t) p[3];
        p += 4; left -= 4;
        return v;
    }
    // Java writes strings as int length + raw UTF-8 bytes.
    std::string str() {
        const int32_t n = (int32_t) u32();
        if (n < 0 || (size_t) n > left) { ok = false; return {}; }
        std::string out(reinterpret_cast<const char *>(p), (size_t) n);
        p += n; left -= (size_t) n;
        return out;
    }
};

bool in_ranges(uint32_t cp, const unicode::CharRange * table, size_t count) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < table[mid].lo) hi = mid;
        else if (cp > table[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

struct Codepoint {
    uint32_t value;
    size_t bytes;
};

// Lenient UTF-8 decoder: invalid bytes come back as their own value so a broken
// input can never make the scanner stall.
Codepoint decode(const char * text, size_t length) {
    const unsigned char * s = reinterpret_cast<const unsigned char *>(text);
    const unsigned char c = s[0];
    if (c < 0x80) return {c, 1};
    size_t need = 0;
    uint32_t value = 0;
    if ((c & 0xE0) == 0xC0) { need = 1; value = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { need = 2; value = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { need = 3; value = c & 0x07; }
    else return {c, 1};
    if (length <= need) return {c, 1};
    for (size_t i = 1; i <= need; i++) {
        const unsigned char cc = s[i];
        if ((cc & 0xC0) != 0x80) return {c, 1};
        value = (value << 6) | (cc & 0x3F);
    }
    return {value, need + 1};
}

inline bool is_letter(uint32_t cp) { return (unicode_class(cp) & kClassLetter) != 0; }
inline bool is_number(uint32_t cp) { return (unicode_class(cp) & kClassNumber) != 0; }

// Java spells the pattern's \s as [\t\n\x0B\f\r\x85\p{Z}] so that the desktop
// JVM and Android ICU agree; keep exactly that set.
inline bool is_space(uint32_t cp) {
    if (cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D || cp == 0x85) return true;
    return (unicode_class(cp) & kClassSeparator) != 0;
}

bool equals_ascii_ci(const char * text, size_t length, const char * word) {
    const size_t n = strlen(word);
    if (length < n) return false;
    for (size_t i = 0; i < n; i++) {
        char c = text[i];
        if (c >= 'A' && c <= 'Z') c = (char) (c - 'A' + 'a');
        if (c != word[i]) return false;
    }
    return true;
}

}  // namespace

int unicode_class(uint32_t codepoint) {
    int bits = 0;
    if (in_ranges(codepoint, unicode::kLetters, unicode::kLetters_count)) bits |= kClassLetter;
    if (in_ranges(codepoint, unicode::kNumbers, unicode::kNumbers_count)) bits |= kClassNumber;
    if (in_ranges(codepoint, unicode::kSeparators, unicode::kSeparators_count)) bits |= kClassSeparator;
    return bits;
}

bool Tokenizer::load(const std::string & path, std::string & error) {
    FILE * file = fopen(path.c_str(), "rb");
    if (!file) { error = "cannot open " + path; return false; }
    std::string data;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), file)) > 0) data.append(buf, n);
    fclose(file);
    if (data.size() < 5 || memcmp(data.data(), kMagic, 5) != 0) { error = "not an M5TK1 file"; return false; }

    Reader r{reinterpret_cast<const unsigned char *>(data.data()) + 5, data.size() - 5};
    bos_ = (int32_t) r.u32();
    eos_ = (int32_t) r.u32();
    r.str();  // the split pattern is reimplemented below; the copy is unused

    const int32_t count = (int32_t) r.u32();
    if (!r.ok || count <= 0 || count > 400000) { error = "bad vocab size"; return false; }
    vocab_.reserve((size_t) count * 2);
    for (int32_t i = 0; i < count && r.ok; i++) {
        std::string key = r.str();
        const int32_t id = (int32_t) r.u32();
        vocab_.emplace(std::move(key), id);
    }
    for (int i = 0; i < 256 && r.ok; i++) {
        const uint32_t character = r.u32();
        byte_id_[i] = (int32_t) r.u32();
        // UTF-16 code units only; every byte maps below U+0800 in practice.
        std::string utf8;
        if (character < 0x80) utf8.push_back((char) character);
        else if (character < 0x800) {
            utf8.push_back((char) (0xC0 | (character >> 6)));
            utf8.push_back((char) (0x80 | (character & 0x3F)));
        } else {
            utf8.push_back((char) (0xE0 | (character >> 12)));
            utf8.push_back((char) (0x80 | ((character >> 6) & 0x3F)));
            utf8.push_back((char) (0x80 | (character & 0x3F)));
        }
        byte_text_[i] = std::move(utf8);
    }
    const int32_t merge_count = (int32_t) r.u32();
    if (!r.ok || merge_count <= 0 || merge_count > 1000000) { error = "bad merge size"; return false; }
    merges_.reserve((size_t) merge_count * 2);
    for (int32_t rank = 0; rank < merge_count && r.ok; rank++) {
        const uint32_t a = r.u32(), b = r.u32();
        const uint32_t id = r.u32();
        const uint64_t key = ((uint64_t) a << 32) | b;
        // Hugging Face keeps the first rank when a pair repeats.
        merges_.emplace(key, ((uint64_t) rank << 32) | id);
    }
    const int32_t special_count = (int32_t) r.u32();
    if (!r.ok || special_count < 0 || special_count > 10000) { error = "bad special size"; return false; }
    specials_.reserve((size_t) special_count);
    for (int32_t i = 0; i < special_count && r.ok; i++) {
        std::string key = r.str();
        const int32_t id = (int32_t) r.u32();
        specials_.emplace_back(std::move(key), id);
    }
    if (!r.ok) { error = "truncated tokenizer file"; return false; }
    if (r.left != 0) { error = "trailing bytes in tokenizer file"; return false; }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
    return true;
}

// The Java pattern, written out as straight-line code:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?\p{L}+
//   |\p{N}{1,3}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
// Alternatives are ordered, so first match wins; the two whitespace rules need
// the backtracking behaviour of the lookahead, which is spelled out inline.
std::vector<std::pair<size_t, size_t>> Tokenizer::split(const char * text, size_t length) const {
    std::vector<std::pair<size_t, size_t>> pieces;
    size_t at = 0, gap = 0;
    auto match = [&](size_t i, size_t & end) -> bool {
        const Codepoint c = decode(text + i, length - i);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c.value == '\'') {
            static const char * kSuffix[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            for (const char * suffix : kSuffix) {
                const size_t n = strlen(suffix);
                if (i + 1 + n <= length && equals_ascii_ci(text + i + 1, length - i - 1, suffix)) {
                    end = i + 1 + n;
                    return true;
                }
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (is_letter(c.value)) {
            size_t j = i;
            while (j < length) {
                const Codepoint d = decode(text + j, length - j);
                if (!is_letter(d.value)) break;
                j += d.bytes;
            }
            end = j;
            return true;
        }
        if (c.value != '\r' && c.value != '\n' && !is_number(c.value) && i + c.bytes < length) {
            const Codepoint d = decode(text + i + c.bytes, length - i - c.bytes);
            if (is_letter(d.value)) {
                size_t j = i + c.bytes;
                while (j < length) {
                    const Codepoint e = decode(text + j, length - j);
                    if (!is_letter(e.value)) break;
                    j += e.bytes;
                }
                end = j;
                return true;
            }
        }
        // \p{N}{1,3}
        if (is_number(c.value)) {
            size_t j = i;
            int taken = 0;
            while (j < length && taken < 3) {
                const Codepoint d = decode(text + j, length - j);
                if (!is_number(d.value)) break;
                j += d.bytes;
                taken++;
            }
            end = j;
            return true;
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        {
            size_t j = c.value == ' ' ? i + c.bytes : i;
            size_t k = j;
            int junk = 0;
            while (k < length) {
                const Codepoint d = decode(text + k, length - k);
                if (is_space(d.value) || is_letter(d.value) || is_number(d.value)) break;
                k += d.bytes;
                junk++;
            }
            if (junk > 0) {
                while (k < length) {
                    const Codepoint d = decode(text + k, length - k);
                    if (d.value != '\r' && d.value != '\n') break;
                    k += d.bytes;
                }
                end = k;
                return true;
            }
        }
        // \s*[\r\n]+ | \s+(?!\S) | \s+
        if (is_space(c.value)) {
            size_t j = i, last_newline_end = 0, last_bytes = 0;
            while (j < length) {
                const Codepoint d = decode(text + j, length - j);
                if (!is_space(d.value)) break;
                j += d.bytes;
                last_bytes = d.bytes;
                if (d.value == '\r' || d.value == '\n') last_newline_end = j;
            }
            if (last_newline_end) {
                end = last_newline_end;                    // \s*[\r\n]+
            } else if (j == length) {
                end = j;                                   // \s+(?!\S) at EOF
            } else {
                end = j - last_bytes;                      // \s+(?!\S) holds back the last space
            }
            return true;
        }
        return false;
    };

    while (at < length) {
        size_t end = at;
        if (match(at, end) && end > at) {
            if (at > gap) pieces.emplace_back(gap, at - gap);  // text the pattern skipped
            pieces.emplace_back(at, end - at);
            at = end;
            gap = end;
        } else {
            at += decode(text + at, length - at).bytes;
        }
    }
    if (gap < length) pieces.emplace_back(gap, length - gap);
    return pieces;
}

void Tokenizer::encode_piece(const char * text, size_t length, std::vector<long> & out) const {
    if (length == 0) return;
    std::string mapped;
    mapped.reserve(length * 2);
    std::vector<int> ids;
    ids.reserve(length);
    for (size_t i = 0; i < length; i++) {
        const unsigned b = (unsigned char) text[i];
        mapped += byte_text_[b];
        ids.push_back(byte_id_[b]);
    }
    // ignore_merges=true: a piece already in the vocabulary is emitted whole.
    auto direct = vocab_.find(mapped);
    if (direct != vocab_.end()) { out.push_back(direct->second); return; }

    while (ids.size() > 1) {
        size_t best = 0;
        uint64_t best_rank = UINT64_MAX;
        int merged = -1;
        for (size_t i = 0; i + 1 < ids.size(); i++) {
            const uint64_t key = ((uint64_t) (uint32_t) ids[i] << 32) | (uint32_t) ids[i + 1];
            auto it = merges_.find(key);
            if (it == merges_.end()) continue;
            const uint64_t rank = it->second >> 32;
            if (rank < best_rank) { best_rank = rank; best = i; merged = (int) (uint32_t) it->second; }
        }
        if (merged < 0) break;
        ids[best] = merged;
        ids.erase(ids.begin() + (long) best + 1);
    }
    for (int id : ids) out.push_back(id);
}

void Tokenizer::encode_plain(const char * text, size_t length, std::vector<long> & out) const {
    for (const auto & piece : split(text, length)) encode_piece(text + piece.first, piece.second, out);
}

std::vector<long> Tokenizer::encode(const std::string & text) const {
    std::vector<long> out;
    out.push_back(bos_);
    size_t plain = 0, at = 0;
    while (at < text.size()) {
        size_t matched_end = 0;
        int matched_id = -1;
        for (const auto & special : specials_) {  // longest first
            const size_t n = special.first.size();
            if (at + n <= text.size() && text.compare(at, n, special.first) == 0) {
                matched_end = at + n;
                matched_id = special.second;
                break;
            }
        }
        if (matched_id >= 0) {
            encode_plain(text.data() + plain, at - plain, out);
            out.push_back(matched_id);
            at = matched_end;
            plain = matched_end;
        } else {
            at += decode(text.data() + at, text.size() - at).bytes;
        }
    }
    encode_plain(text.data() + plain, text.size() - plain, out);
    out.push_back(eos_);
    return out;
}

}  // namespace m5
