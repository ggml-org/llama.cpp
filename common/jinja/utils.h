#pragma once

#include <string>
#include <sstream>
#include <algorithm>

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

} // namespace jinja
