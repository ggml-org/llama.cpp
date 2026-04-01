#include "../tool-registry.h"
#include "../permission.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <vector>

namespace fs = std::filesystem;

// ANSI color codes
static const char * ANSI_RED    = "\033[31m";
static const char * ANSI_GREEN  = "\033[32m";
static const char * ANSI_RESET  = "\033[0m";
static const char * ANSI_DIM    = "\033[2m";

// Split string into lines
static std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }
    return lines;
}

// Generate a simple colored diff
static std::string generate_diff(const std::string & old_text, const std::string & new_text, const std::string & file_path) {
    auto old_lines = split_lines(old_text);
    auto new_lines = split_lines(new_text);

    std::ostringstream diff;
    diff << ANSI_DIM << "--- " << file_path << ANSI_RESET << "\n";
    diff << ANSI_DIM << "+++ " << file_path << ANSI_RESET << "\n";

    for (const auto & line : old_lines) {
        diff << ANSI_RED << "- " << line << ANSI_RESET << "\n";
    }
    for (const auto & line : new_lines) {
        diff << ANSI_GREEN << "+ " << line << ANSI_RESET << "\n";
    }

    return diff.str();
}

static std::string read_file(const fs::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static bool write_file(const fs::path & path, const std::string & content) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file << content;
    return !file.fail();
}

// Normalize text for fuzzy matching: trim trailing whitespace per line,
// replace smart quotes, unicode dashes, and special spaces with ASCII equivalents.
static std::string normalize_for_fuzzy_match(const std::string & text) {
    std::string result;
    result.reserve(text.size());

    const unsigned char * s = reinterpret_cast<const unsigned char *>(text.data());
    const unsigned char * end = s + text.size();

    while (s < end) {
        unsigned char c = *s;

        // U+00A0 (NBSP): C2 A0 -> ' '
        if (c == 0xC2 && s + 1 < end && s[1] == 0xA0) {
            result += ' ';
            s += 2;
            continue;
        }

        // E2 80 XX: dashes, smart quotes, spaces
        if (c == 0xE2 && s + 2 < end && s[1] == 0x80) {
            unsigned char b3 = s[2];
            char replacement = 0;

            if (b3 >= 0x90 && b3 <= 0x95) replacement = '-';        // U+2010-U+2015 dashes
            else if (b3 >= 0x98 && b3 <= 0x9B) replacement = '\'';  // U+2018-U+201B single quotes
            else if (b3 >= 0x9C && b3 <= 0x9F) replacement = '"';   // U+201C-U+201F double quotes
            else if ((b3 >= 0x82 && b3 <= 0x8A) || b3 == 0xAF) replacement = ' ';  // U+2002-U+200A, U+202F

            if (replacement) {
                result += replacement;
                s += 3;
                continue;
            }
        }

        // U+2212 (MINUS SIGN): E2 88 92 -> '-'
        if (c == 0xE2 && s + 2 < end && s[1] == 0x88 && s[2] == 0x92) {
            result += '-';
            s += 3;
            continue;
        }

        // U+205F (MEDIUM MATHEMATICAL SPACE): E2 81 9F -> ' '
        if (c == 0xE2 && s + 2 < end && s[1] == 0x81 && s[2] == 0x9F) {
            result += ' ';
            s += 3;
            continue;
        }

        // U+3000 (IDEOGRAPHIC SPACE): E3 80 80 -> ' '
        if (c == 0xE3 && s + 2 < end && s[1] == 0x80 && s[2] == 0x80) {
            result += ' ';
            s += 3;
            continue;
        }

        result += (char)c;
        s++;
    }

    // Trim trailing whitespace per line
    std::string trimmed;
    trimmed.reserve(result.size());
    std::istringstream stream(result);
    std::string line;
    bool first = true;

    while (std::getline(stream, line)) {
        if (!first) trimmed += '\n';
        first = false;
        size_t last = line.find_last_not_of(" \t");
        if (last != std::string::npos) {
            trimmed.append(line, 0, last + 1);
        }
    }
    // Preserve trailing newline if original had one
    if (!result.empty() && result.back() == '\n') {
        trimmed += '\n';
    }

    return trimmed;
}

static tool_result edit_execute(const json & args, const tool_context & ctx) {
    std::string file_path = args.value("file_path", "");
    std::string old_string = args.value("old_string", "");
    std::string new_string = args.value("new_string", "");
    bool replace_all = args.value("replace_all", false);

    if (file_path.empty()) {
        return {false, "", "file_path parameter is required"};
    }

    if (old_string.empty()) {
        return {false, "", "old_string parameter is required"};
    }

    if (old_string == new_string) {
        return {false, "", "old_string and new_string must be different"};
    }

    // Make absolute if relative
    fs::path path(file_path);
    if (path.is_relative()) {
        path = fs::path(ctx.working_dir) / path;
    }

    // Check if file exists
    if (!fs::exists(path)) {
        return {false, "", "File not found: " + path.string()};
    }

    // Block sensitive files
    if (permission_manager::is_sensitive_file(path.string())) {
        return {false, "", "Cannot edit sensitive file (contains credentials/secrets): " + path.string()};
    }

    // Read file content
    std::string content = read_file(path);
    if (content.empty() && fs::file_size(path) > 0) {
        return {false, "", "Cannot read file: " + path.string()};
    }

    // Find occurrences (exact match first)
    size_t first_pos = content.find(old_string);
    bool fuzzy_matched = false;
    std::string working_content = content;
    std::string working_old = old_string;

    if (first_pos == std::string::npos) {
        // Try fuzzy match: normalize both content and search string
        std::string fuzzy_content = normalize_for_fuzzy_match(content);
        std::string fuzzy_old = normalize_for_fuzzy_match(old_string);

        if (fuzzy_old.empty()) {
            return {false, "", "old_string not found in file. Make sure you're using the exact text including whitespace and indentation."};
        }

        size_t fuzzy_pos = fuzzy_content.find(fuzzy_old);
        if (fuzzy_pos == std::string::npos) {
            return {false, "", "old_string not found in file (tried exact and fuzzy match). Make sure you're using the exact text including whitespace and indentation."};
        }

        // Map the fuzzy match position back to the original content.
        // Build byte offset mapping: for each byte in fuzzy_content, track the
        // corresponding byte in content. Then replace only the matched range in
        // the original, preserving all other content unchanged.
        std::vector<size_t> offset_map;
        offset_map.reserve(fuzzy_content.size() + 1);
        {
            // Rebuild the normalization, tracking original byte positions
            const unsigned char * s = reinterpret_cast<const unsigned char *>(content.data());
            const unsigned char * s_end = s + content.size();
            // First pass: unicode normalization with offset tracking
            std::string unicode_normed;
            std::vector<size_t> unicode_offsets; // offset in content for each byte in unicode_normed
            unicode_normed.reserve(content.size());
            unicode_offsets.reserve(content.size());

            while (s < s_end) {
                size_t orig_pos = s - reinterpret_cast<const unsigned char *>(content.data());
                unsigned char c = *s;
                int skip = 0;
                char replacement = 0;

                if (c == 0xC2 && s + 1 < s_end && s[1] == 0xA0) { skip = 2; replacement = ' '; }
                else if (c == 0xE2 && s + 2 < s_end && s[1] == 0x80) {
                    unsigned char b3 = s[2];
                    if (b3 >= 0x90 && b3 <= 0x95) { skip = 3; replacement = '-'; }
                    else if (b3 >= 0x98 && b3 <= 0x9B) { skip = 3; replacement = '\''; }
                    else if (b3 >= 0x9C && b3 <= 0x9F) { skip = 3; replacement = '"'; }
                    else if ((b3 >= 0x82 && b3 <= 0x8A) || b3 == 0xAF) { skip = 3; replacement = ' '; }
                }
                else if (c == 0xE2 && s + 2 < s_end && s[1] == 0x88 && s[2] == 0x92) { skip = 3; replacement = '-'; }
                else if (c == 0xE2 && s + 2 < s_end && s[1] == 0x81 && s[2] == 0x9F) { skip = 3; replacement = ' '; }
                else if (c == 0xE3 && s + 2 < s_end && s[1] == 0x80 && s[2] == 0x80) { skip = 3; replacement = ' '; }

                if (skip > 0) {
                    unicode_normed += replacement;
                    unicode_offsets.push_back(orig_pos);
                    s += skip;
                } else {
                    unicode_normed += (char)c;
                    unicode_offsets.push_back(orig_pos);
                    s++;
                }
            }
            // Sentinel: end of content
            unicode_offsets.push_back(content.size());

            // Second pass: trailing whitespace trimming with offset tracking
            // Split unicode_normed into lines, trim trailing ws, build final offset map
            size_t line_start = 0;
            for (size_t i = 0; i <= unicode_normed.size(); i++) {
                if (i == unicode_normed.size() || unicode_normed[i] == '\n') {
                    // Find last non-ws in this line
                    size_t line_end = i;
                    size_t trimmed_end = line_end;
                    while (trimmed_end > line_start &&
                           (unicode_normed[trimmed_end - 1] == ' ' || unicode_normed[trimmed_end - 1] == '\t')) {
                        trimmed_end--;
                    }
                    for (size_t j = line_start; j < trimmed_end; j++) {
                        offset_map.push_back(unicode_offsets[j]);
                    }
                    if (i < unicode_normed.size()) {
                        offset_map.push_back(unicode_offsets[i]); // the \n itself
                    }
                    line_start = i + 1;
                }
            }
            // Preserve trailing newline mapping
            if (!unicode_normed.empty() && unicode_normed.back() == '\n') {
                // Already added above
            }
            offset_map.push_back(content.size()); // sentinel
        }

        // Now map fuzzy_pos back to original
        size_t orig_start = (fuzzy_pos < offset_map.size()) ? offset_map[fuzzy_pos] : content.size();
        size_t fuzzy_end_pos = fuzzy_pos + fuzzy_old.size();
        size_t orig_end = (fuzzy_end_pos < offset_map.size()) ? offset_map[fuzzy_end_pos] : content.size();

        first_pos = orig_start;
        fuzzy_matched = true;
        // Set working_old to the original bytes at the matched range
        working_old = content.substr(orig_start, orig_end - orig_start);
        // working_content stays as original content
    }

    // Check for multiple occurrences
    size_t last_pos = working_content.rfind(working_old);
    if (first_pos != last_pos && !replace_all) {
        int count = 0;
        size_t pos = 0;
        while ((pos = working_content.find(working_old, pos)) != std::string::npos) {
            count++;
            pos += working_old.length();
        }
        std::string suffix = fuzzy_matched ? " (fuzzy match)" : "";
        return {false, "", "Found " + std::to_string(count) + " occurrences of old_string" + suffix + ". Provide more context to make it unique, or set replace_all=true to replace all occurrences."};
    }

    // Perform replacement
    std::string new_content;
    int replacements = 0;

    if (replace_all) {
        size_t pos = 0;
        size_t last_end = 0;
        while ((pos = working_content.find(working_old, last_end)) != std::string::npos) {
            new_content += working_content.substr(last_end, pos - last_end);
            new_content += new_string;
            last_end = pos + working_old.length();
            replacements++;
        }
        new_content += working_content.substr(last_end);
    } else {
        new_content = working_content.substr(0, first_pos) + new_string + working_content.substr(first_pos + working_old.length());
        replacements = 1;
    }

    // Write file
    if (!write_file(path, new_content)) {
        return {false, "", "Failed to write changes to file"};
    }

    std::string diff = generate_diff(old_string, new_string, path.string());
    std::string fuzzy_note = fuzzy_matched ? " (fuzzy match applied)" : "";
    std::string msg = "Successfully replaced " + std::to_string(replacements) + " occurrence(s) in " + path.string() + fuzzy_note + "\n\n" + diff;
    return {true, msg, ""};
}

static tool_def edit_tool = {
    "edit",
    "Make targeted edits to a file by finding and replacing specific text. The old_string must match exactly (including whitespace and indentation). For multiple matches, either provide more context or use replace_all.",
    R"json({
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit (absolute or relative to working directory)"
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to find and replace. Include enough context (surrounding lines) to uniquely identify the location."
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace old_string with"
            },
            "replace_all": {
                "type": "boolean",
                "description": "If true, replace all occurrences. Default is false (single replacement)."
            }
        },
        "required": ["file_path", "old_string", "new_string"]
    })json",
    edit_execute
};

REGISTER_TOOL(edit, edit_tool);
