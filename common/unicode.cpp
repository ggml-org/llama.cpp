#include "unicode.h"

size_t utf8_sequence_length(unsigned char first_byte) {
    // Lookup table based on high 4 bits:
    // 0xxx xxxx = 1 byte  (ASCII)
    // 110x xxxx = 2 bytes
    // 1110 xxxx = 3 bytes
    // 1111 0xxx = 4 bytes (only 0xF0–0xF4 are valid starts)
    static const size_t lookup[] = {
        1, 1, 1, 1, 1, 1, 1, 1,  // 0000–0111 (0x00–0x7F) ASCII
        0, 0, 0, 0,              // 1000–1011 (0x80–0xBF) continuation bytes
        2, 2,                    // 1100–1101 (0xC0–0xDF) 2-byte sequences
        3,                       // 1110      (0xE0–0xEF) 3-byte sequences
        4                        // 1111      (0xF0–0xFF) potential 4-byte sequences
    };

    size_t len = lookup[first_byte >> 4];

    // Filter out invalid first bytes:
    // - 0xC0–0xC1: overlong 2-byte sequences (would encode U+0000–U+007F)
    // - 0xF5–0xFF: would encode beyond U+10FFFF or invalid 5+ byte sequences
    if (first_byte == 0xC0 || first_byte == 0xC1 || first_byte >= 0xF5) {
        return 0;
    }

    return len;
}

utf8_parse_result parse_utf8_codepoint(std::string_view input, size_t offset) {
    if (offset >= input.size()) {
        return utf8_parse_result(utf8_parse_result::INCOMPLETE);
    }

    const unsigned char first = static_cast<unsigned char>(input[offset]);

    // ASCII fast path (1-byte sequence)
    if (first < 0x80) {
        return utf8_parse_result(utf8_parse_result::SUCCESS, first, 1);
    }

    // Determine expected sequence length (0 means invalid first byte)
    size_t seq_len = utf8_sequence_length(first);
    if (seq_len == 0) {
        return utf8_parse_result(utf8_parse_result::INVALID);
    }

    size_t available = input.size() - offset;

    // Handle incomplete sequences: not enough bytes for the promised length.
    if (available < seq_len) {
        // We want INCOMPLETE only if this prefix *could* still become valid.
        // So we validate as much as we can, and reject prefixes that are
        // already impossible regardless of future bytes.

        if (available >= 2) {
            unsigned char second = static_cast<unsigned char>(input[offset + 1]);

            // Second byte must be a continuation byte.
            if ((second & 0xC0) != 0x80) {
                return utf8_parse_result(utf8_parse_result::INVALID);
            }

            // Apply lead+second byte constraints that are necessary for any
            // valid UTF-8 sequence (these mirror the usual per-byte rules).

            if (seq_len == 3) {
                // 3-byte sequences (first in 0xE0–0xEF):
                // - E0 A0–BF ..   => valid (U+0800–U+0FFF)
                // - E0 80–9F ..   => overlong (U+0000–U+07FF) => impossible
                // - ED 80–9F ..   => valid (U+D000–U+D7FF)
                // - ED A0–BF ..   => surrogates (U+D800–U+DFFF) => impossible
                if (first == 0xE0 && second < 0xA0) {
                    return utf8_parse_result(utf8_parse_result::INVALID);
                }
                if (first == 0xED && second > 0x9F) {
                    return utf8_parse_result(utf8_parse_result::INVALID);
                }
            } else if (seq_len == 4) {
                // 4-byte sequences (first in 0xF0–0xF4):
                // - F0 90–BF .. .. => valid (U+10000–U+3FFFF)
                // - F0 80–8F .. .. => overlong (U+0000–U+FFFF) => impossible
                // - F4 80–8F .. .. => valid (U+100000–U+10FFFF)
                // - F4 90–BF .. .. => > U+10FFFF => impossible
                if (first == 0xF0 && second < 0x90) {
                    return utf8_parse_result(utf8_parse_result::INVALID);
                }
                if (first == 0xF4 && second > 0x8F) {
                    return utf8_parse_result(utf8_parse_result::INVALID);
                }
            }

            // For any further available bytes, just enforce the continuation pattern.
            for (size_t i = 2; i < available; ++i) {
                unsigned char byte = static_cast<unsigned char>(input[offset + i]);
                if ((byte & 0xC0) != 0x80) {
                    return utf8_parse_result(utf8_parse_result::INVALID);
                }
            }
        }

        // If we reach here, the prefix is syntactically and range-wise
        // compatible with *some* valid UTF-8 code point; we just ran out of bytes.
        return utf8_parse_result(utf8_parse_result::INCOMPLETE);
    }

    // We have at least seq_len bytes: validate all continuation bytes.
    for (size_t i = 1; i < seq_len; ++i) {
        unsigned char byte = static_cast<unsigned char>(input[offset + i]);
        if ((byte & 0xC0) != 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
    }

    // Decode based on sequence length.
    uint32_t codepoint = 0;

    if (seq_len == 2) {
        // 110xxxxx 10xxxxxx
        codepoint =
            ((first & 0x1F) << 6) |
            (static_cast<unsigned char>(input[offset + 1]) & 0x3F);

        // 0xC0 and 0xC1 were filtered out, so this always yields U+0080–U+07FF.
    } else if (seq_len == 3) {
        // 1110xxxx 10xxxxxx 10xxxxxx
        codepoint =
            ((first & 0x0F) << 12) |
            ((static_cast<unsigned char>(input[offset + 1]) & 0x3F) << 6) |
            (static_cast<unsigned char>(input[offset + 2]) & 0x3F);

        // Reject overlong encodings: 3-byte must encode U+0800–U+FFFF.
        if (codepoint < 0x800) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }

        // Reject surrogate code points U+D800–U+DFFF (invalid in UTF-8).
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
    } else if (seq_len == 4) {
        // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        codepoint =
            ((first & 0x07) << 18) |
            ((static_cast<unsigned char>(input[offset + 1]) & 0x3F) << 12) |
            ((static_cast<unsigned char>(input[offset + 2]) & 0x3F) << 6) |
            (static_cast<unsigned char>(input[offset + 3]) & 0x3F);

        // Reject overlong encodings: 4-byte must encode U+10000–U+10FFFF.
        if (codepoint < 0x10000) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }

        // Reject code points beyond Unicode max (U+10FFFF).
        if (codepoint > 0x10FFFF) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
    }

    return utf8_parse_result(utf8_parse_result::SUCCESS, codepoint, seq_len);
}
