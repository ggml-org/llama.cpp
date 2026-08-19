#pragma once

#include <string>
#include <vector>
#include <cstddef>

// Internal tokenizer helpers. Not part of the public mtmd API.

struct mtmd_marker_part {
    std::string text;
    int bitmap_i; // -1 = literal text
};

// Split on delimiter and keep the delimiter as its own part.
// "a <m> b" + "<m>" -> {"a ", "<m>", " b"}
inline std::vector<std::string> mtmd_split_text(const std::string & input, const std::string & delimiter) {
    std::vector<std::string> result;
    if (input.empty()) {
        return result;
    }
    size_t start = 0;
    size_t pos = 0;
    while ((pos = input.find(delimiter, start)) != std::string::npos) {
        if (pos > start) {
            result.push_back(input.substr(start, pos - start));
        }
        result.push_back(delimiter);
        start = pos + delimiter.length();
    }
    if (start < input.length()) {
        result.push_back(input.substr(start));
    }
    return result;
}

// Bind the first n_bitmaps marker hits as media slots. Extra hits stay as text.
// Returns false if the prompt has fewer markers than bitmaps.
inline bool mtmd_bind_media_markers(
        const std::string & input,
        const std::string & marker,
        size_t n_bitmaps,
        std::vector<mtmd_marker_part> & out) {
    out.clear();
    size_t i_bm = 0;
    for (auto & part : mtmd_split_text(input, marker)) {
        if (part == marker && i_bm < n_bitmaps) {
            out.push_back({"", (int) i_bm++});
        } else {
            out.push_back({std::move(part), -1});
        }
    }
    return i_bm == n_bitmaps;
}
