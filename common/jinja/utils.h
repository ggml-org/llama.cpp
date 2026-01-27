#pragma once

#include <string>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace jinja {

static void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

// for displaying source code around error position
static std::string peak_source(const std::string & source, size_t pos, size_t max_peak_chars = 40) {
    if (source.empty()) {
        return "(no source available)";
    }
    std::string output;
    size_t start = (pos >= max_peak_chars) ? (pos - max_peak_chars) : 0;
    size_t end = std::min(pos + max_peak_chars, source.length());
    std::string substr = source.substr(start, end - start);
    string_replace_all(substr, "\n", "↵");
    output += "..." + substr + "...\n";
    std::string spaces(pos - start + 3, ' ');
    output += spaces + "^";
    return output;
}

static std::string fmt_error_with_source(const std::string & tag, const std::string & msg, const std::string & source, size_t pos) {
    std::ostringstream oss;
    oss << tag << ": " << msg << "\n";
    oss << peak_source(source, pos);
    return oss.str();
}

// Note: this is a simple hasher, not cryptographically secure, just for hash table usage
struct hasher {
    static constexpr auto size_t_digits = sizeof(size_t) * 8;
    static constexpr size_t prime = size_t_digits == 64 ? 0x100000001b3 : 0x01000193;
    static constexpr size_t seed = size_t_digits == 64 ? 0xcbf29ce484222325 : 0x811c9dc5;
    static constexpr auto block_size = sizeof(size_t); // in bytes; allowing the compiler to vectorize the computation

    static_assert(size_t_digits == 64 || size_t_digits == 32);
    static_assert(block_size == 8 || block_size == 4);

    uint8_t buffer[block_size];
    size_t idx = 0; // current index in buffer
    size_t state = seed;

    hasher() = default;
    hasher(const std::type_info & type_inf) noexcept {
        const auto type_hash = type_inf.hash_code();
        update(&type_hash, sizeof(type_hash));
    }

    // Properties:
    //   - update is not associative: update(a).update(b) != update(b).update(a)
    //   - update(a ~ b) == update(a).update(b) with ~ as concatenation operator --> useful for streaming
    //   - update("", 0) --> state unchanged with empty input
    hasher& update(void const * bytes, size_t len) noexcept {
        const uint8_t * c = static_cast<uint8_t const *>(bytes);
        if (len == 0) {
            return *this;
        }
        size_t processed = 0;

        // first, fill the existing buffer if it's partial
        if (idx > 0) {
            size_t to_fill = block_size - idx;
            if (to_fill > len) {
                to_fill = len;
            }
            std::memcpy(buffer + idx, c, to_fill);
            idx += to_fill;
            processed += to_fill;
            if (idx == block_size) {
                update_block(buffer);
                idx = 0;
            }
        }

        // process full blocks from the remaining input
        for (; processed + block_size <= len; processed += block_size) {
            update_block(c + processed);
        }

        // buffer any remaining bytes
        size_t remaining = len - processed;
        if (remaining > 0) {
            std::memcpy(buffer, c + processed, remaining);
            idx = remaining;
        }
        return *this;
    }

    // convenience function for testing only
    hasher& update(const std::string & s) noexcept {
        return update(s.data(), s.size());
    }

    // finalize and get the hash value
    // note: after calling digest, the hasher state is modified, do not call update() again
    size_t digest() noexcept {
        // if there are remaining bytes in buffer, fill the rest with zeros and process
        if (idx > 0) {
            for (size_t i = idx; i < block_size; ++i) {
                buffer[i] = 0;
            }
            update_block(buffer);
            idx = 0;
        }

        return state;
    }

private:
    // IMPORTANT: block must have at least block_size bytes
    void update_block(const uint8_t * block) noexcept {
        size_t blk = static_cast<uint32_t>(block[0])
                    | (static_cast<uint32_t>(block[1]) << 8)
                    | (static_cast<uint32_t>(block[2]) << 16)
                    | (static_cast<uint32_t>(block[3]) << 24);
        if constexpr (block_size == 8) {
            blk = blk | (static_cast<uint64_t>(block[4]) << 32)
                      | (static_cast<uint64_t>(block[5]) << 40)
                      | (static_cast<uint64_t>(block[6]) << 48)
                      | (static_cast<uint64_t>(block[7]) << 56);
        }
        state ^= blk;
        state *= prime;
    }
};

} // namespace jinja
