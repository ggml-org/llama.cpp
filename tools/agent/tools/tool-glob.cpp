#include "../tool-registry.h"

#include <filesystem>
#include <regex>
#include <vector>
#include <algorithm>
#include <sstream>

namespace fs = std::filesystem;

// Convert glob pattern to regex
static std::string glob_to_regex(const std::string & pattern) {
    std::string regex;
    bool in_bracket = false;

    for (size_t i = 0; i < pattern.length(); i++) {
        char c = pattern[i];

        if (in_bracket) {
            if (c == ']') {
                in_bracket = false;
                regex += c;
            } else if (c == '\\') {
                regex += "\\\\";
            } else {
                regex += c;
            }
            continue;
        }

        switch (c) {
            case '*':
                if (i + 1 < pattern.length() && pattern[i + 1] == '*') {
                    // ** matches any path
                    regex += ".*";
                    i++;  // Skip next *
                } else {
                    // * matches anything except /
                    regex += "[^/]*";
                }
                break;
            case '?':
                regex += "[^/]";
                break;
            case '[':
                in_bracket = true;
                regex += c;
                break;
            case '.':
            case '(':
            case ')':
            case '+':
            case '|':
            case '^':
            case '$':
            case '{':
            case '}':
            case '\\':
                regex += '\\';
                regex += c;
                break;
            default:
                regex += c;
        }
    }

    return regex;
}

static tool_result glob_execute(const json & args, const tool_context & ctx) {
    std::string pattern = args.value("pattern", "");
    std::string search_path = args.value("path", ctx.working_dir);

    if (pattern.empty()) {
        return {false, "", "pattern parameter is required"};
    }

    // Make search path absolute if relative
    fs::path base_path(search_path);
    if (base_path.is_relative()) {
        base_path = fs::path(ctx.working_dir) / base_path;
    }

    if (!fs::exists(base_path)) {
        return {false, "", "Directory not found: " + base_path.string()};
    }

    if (!fs::is_directory(base_path)) {
        return {false, "", "Not a directory: " + base_path.string()};
    }

    // Convert glob pattern to regex
    std::string regex_pattern = glob_to_regex(pattern);
    std::regex pattern_regex;
    try {
        pattern_regex = std::regex(regex_pattern, std::regex::ECMAScript | std::regex::icase);
    } catch (const std::regex_error & e) {
        return {false, "", "Invalid pattern: " + std::string(e.what())};
    }

    // Collect matching files with modification times
    std::vector<std::pair<fs::path, fs::file_time_type>> matches;
    const int limit = 100;

    try {
        for (const auto & entry : fs::recursive_directory_iterator(
                base_path, fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;

            // Match against full relative path or just filename depending on pattern
            std::string to_match;
            if (pattern.find('/') != std::string::npos || pattern.find("**") != std::string::npos) {
                // Pattern contains path separator, match against relative path
                to_match = fs::relative(entry.path(), base_path).string();
            } else {
                // Pattern is just a filename pattern
                to_match = entry.path().filename().string();
            }

            if (std::regex_match(to_match, pattern_regex)) {
                matches.emplace_back(entry.path(), entry.last_write_time());
                if ((int)matches.size() >= limit) {
                    break;
                }
            }
        }
    } catch (const fs::filesystem_error & e) {
        // Continue with what we have
    }

    // Sort by modification time (most recent first)
    std::sort(matches.begin(), matches.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });

    // Build output
    std::ostringstream output;
    if (matches.empty()) {
        output << "No files found matching pattern: " << pattern;
    } else {
        for (const auto & match : matches) {
            // Output relative paths
            output << fs::relative(match.first, base_path).string() << "\n";
        }

        if ((int)matches.size() >= limit) {
            output << "\n[Results limited to " << limit << " files. Use a more specific pattern.]";
        } else {
            output << "\n[" << matches.size() << " file(s) found]";
        }
    }

    return {true, output.str(), ""};
}

static tool_def glob_tool = {
    "glob",
    "Find files matching a glob pattern. Supports * (any characters except /), ** (any path), ? (single character), [abc] (character class). Results are sorted by modification time (most recent first).",
    R"json({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g., '*.cpp', 'src/**/*.ts', 'test_*.py')"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: working directory)"
            }
        },
        "required": ["pattern"]
    })json",
    glob_execute
};

REGISTER_TOOL(glob, glob_tool);
