#include "unicode.h"

size_t utf8_sequence_length(unsigned char first_byte) {
    // Lookup table based on high 4 bits
    // 0xxx xxxx = 1 byte  (ASCII)
    // 110x xxxx = 2 bytes
    // 1110 xxxx = 3 bytes
    // 1111 0xxx = 4 bytes (only 0xF0-0xF7, not 0xF8-0xFF)
    static const size_t lookup[] = {
        1, 1, 1, 1, 1, 1, 1, 1,  // 0000-0111 (0x00-0x7F)
        0, 0, 0, 0,              // 1000-1011 (continuation bytes 0x80-0xBF, invalid as first byte)
        2, 2,                    // 1100-1101 (0xC0-0xDF)
        3,                       // 1110      (0xE0-0xEF)
        4                        // 1111      (0xF0-0xFF, but need to check 0xF8-0xFF separately)
    };
    size_t len = lookup[first_byte >> 4];

    // Additional validation for invalid first bytes:
    // - 0xC0-0xC1: would create overlong 2-byte sequences
    // - 0xF8-0xFF: invalid 5+ byte sequences
    if (first_byte >= 0xF8 || (first_byte >= 0xC0 && first_byte <= 0xC1)) {
        return 0;  // Invalid
    }

    return len;
}

utf8_parse_result parse_utf8_codepoint(std::string_view input, size_t offset) {
    if (offset >= input.size()) {
        return utf8_parse_result(utf8_parse_result::INCOMPLETE);
    }

    const unsigned char first = static_cast<unsigned char>(input[offset]);

    // ASCII fast path (most common case)
    if (first < 0x80) {
        return utf8_parse_result(utf8_parse_result::SUCCESS, first, 1);
    }

    // Invalid first byte (continuation byte 10xxxxxx as first byte, or 0xF8-0xFF)
    if ((first & 0xC0) == 0x80) {
        return utf8_parse_result(utf8_parse_result::INVALID);
    }

    size_t seq_len = utf8_sequence_length(first);
    if (seq_len == 0) {
        // Invalid first byte (e.g., 0xF8-0xFF)
        return utf8_parse_result(utf8_parse_result::INVALID);
    }

    size_t available = input.size() - offset;

    // Check if we have enough bytes for the complete sequence
    if (available < seq_len) {
        return utf8_parse_result(utf8_parse_result::INCOMPLETE);
    }

    uint32_t codepoint = 0;

    // Decode based on sequence length
    if (seq_len == 2) {
        // 110xxxxx 10xxxxxx
        if ((first & 0xE0) != 0xC0) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        const unsigned char second = static_cast<unsigned char>(input[offset + 1]);
        if ((second & 0xC0) != 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        codepoint = ((first & 0x1F) << 6) | (second & 0x3F);
        // Check for overlong encoding
        if (codepoint < 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
    } else if (seq_len == 3) {
        // 1110xxxx 10xxxxxx 10xxxxxx
        if ((first & 0xF0) != 0xE0) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        const unsigned char second = static_cast<unsigned char>(input[offset + 1]);
        const unsigned char third = static_cast<unsigned char>(input[offset + 2]);
        if ((second & 0xC0) != 0x80 || (third & 0xC0) != 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        codepoint = ((first & 0x0F) << 12) | ((second & 0x3F) << 6) | (third & 0x3F);
        // Check for overlong encoding
        if (codepoint < 0x800) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        // Check for surrogate pairs (0xD800-0xDFFF are invalid in UTF-8)
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
    } else if (seq_len == 4) {
        // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        if ((first & 0xF8) != 0xF0) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        const unsigned char second = static_cast<unsigned char>(input[offset + 1]);
        const unsigned char third = static_cast<unsigned char>(input[offset + 2]);
        const unsigned char fourth = static_cast<unsigned char>(input[offset + 3]);
        if ((second & 0xC0) != 0x80 || (third & 0xC0) != 0x80 || (fourth & 0xC0) != 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        codepoint = ((first & 0x07) << 18) | ((second & 0x3F) << 12) |
                    ((third & 0x3F) << 6) | (fourth & 0x3F);
        // Check for overlong encoding
        if (codepoint < 0x10000) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        // Check for valid Unicode range (max is 0x10FFFF)
        if (codepoint > 0x10FFFF) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
    } else {
        // Invalid sequence length
        return utf8_parse_result(utf8_parse_result::INVALID);
    }

    return utf8_parse_result(utf8_parse_result::SUCCESS, codepoint, seq_len);
}
