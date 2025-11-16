#include "unicode.h"

size_t utf8_sequence_length(unsigned char first_byte) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(first_byte) >> 4;
    return lookup[highbits];
}

utf8_parse_result parse_utf8_codepoint(std::string_view input, size_t offset) {
    if (offset >= input.size()) {
        return utf8_parse_result(utf8_parse_result::INCOMPLETE);
    }

    const unsigned char first = static_cast<unsigned char>(input[offset]);

    // ASCII fast path
    if (!(first & 0x80)) {
        return utf8_parse_result(utf8_parse_result::SUCCESS, first, 1);
    }

    // Invalid: continuation byte as first byte
    if (!(first & 0x40)) {
        return utf8_parse_result(utf8_parse_result::INVALID);
    }

    // 2-byte sequence
    if (!(first & 0x20)) {
        if (offset + 1 >= input.size()) {
            return utf8_parse_result(utf8_parse_result::INCOMPLETE);
        }
        const unsigned char second = static_cast<unsigned char>(input[offset + 1]);
        if ((second & 0xc0) != 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        auto result = ((first & 0x1f) << 6) | (second & 0x3f);
        return utf8_parse_result(utf8_parse_result::SUCCESS, result, 2);
    }

    // 3-byte sequence
    if (!(first & 0x10)) {
        if (offset + 2 >= input.size()) {
            return utf8_parse_result(utf8_parse_result::INCOMPLETE);
        }
        const unsigned char second = static_cast<unsigned char>(input[offset + 1]);
        const unsigned char third = static_cast<unsigned char>(input[offset + 2]);
        if ((second & 0xc0) != 0x80 || (third & 0xc0) != 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        auto result = ((first & 0x0f) << 12) | ((second & 0x3f) << 6) | (third & 0x3f);
        return utf8_parse_result(utf8_parse_result::SUCCESS, result, 3);
    }

    // 4-byte sequence
    if (!(first & 0x08)) {
        if (offset + 3 >= input.size()) {
            return utf8_parse_result(utf8_parse_result::INCOMPLETE);
        }
        const unsigned char second = static_cast<unsigned char>(input[offset + 1]);
        const unsigned char third = static_cast<unsigned char>(input[offset + 2]);
        const unsigned char fourth = static_cast<unsigned char>(input[offset + 3]);
        if ((second & 0xc0) != 0x80 || (third & 0xc0) != 0x80 || (fourth & 0xc0) != 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        auto result = ((first & 0x07) << 18) | ((second & 0x3f) << 12) | ((third & 0x3f) << 6) | (fourth & 0x3f);
        return utf8_parse_result(utf8_parse_result::SUCCESS, result, 4);
    }

    // Invalid first byte
    return utf8_parse_result(utf8_parse_result::INVALID);
}
