#pragma once

#include <string>
#include <sstream>
#include <algorithm>
#include <limits>

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

// FNV-1a hash function that takes initial seed hash
// No need to worry about the whole hash_combine drama
static constexpr auto size_t_digits = std::numeric_limits<size_t>::digits;
static_assert(size_t_digits == 64 || size_t_digits == 32);

template <typename... Args>
static size_t hash_bytes(size_t seed, void const * bytes, size_t len, Args&&... args) noexcept
{
    static_assert(sizeof...(args) % 2 == 0);
    static constexpr size_t prime = size_t_digits == 64 ? 0x100000001b3 : 0x01000193;

    unsigned char const * c = static_cast<unsigned char const *>(bytes);
    unsigned char const * const e = c + len;

    for (; c < e; ++c) {
        seed = (seed ^ *c) * prime;
    }

    if constexpr (sizeof...(args) > 0) {
        seed = hash_bytes(seed, std::forward<Args>(args)...);
    }

    return seed;
}

template <typename... Args>
static size_t hash_bytes(void const * bytes, size_t len, Args&&... args) noexcept
{
    static constexpr size_t seed = size_t_digits == 64 ? 0xcbf29ce484222325 : 0x811c9dc5;

    return hash_bytes(seed, bytes, len, std::forward<Args>(args)...);
}

} // namespace jinja
