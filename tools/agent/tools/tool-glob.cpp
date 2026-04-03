#include "../tool-registry.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <vector>
#include <algorithm>
#include <sstream>

namespace fs = std::filesystem;

// Convert glob pattern to regex
static std::string glob_to_regex(const std::string & pattern) {
    std::string regex;
    bool in_bracket = false;
    bool in_brace   = false;

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

        if (in_brace) {
            if (c == '}') {
                in_brace = false;
                regex += ')';
            } else if (c == ',') {
                regex += '|';
            } else if (c == '*') {
                if (i + 1 < pattern.length() && pattern[i + 1] == '*') {
                    regex += ".*";
                    i++;
                } else {
                    regex += "[^/]*";
                }
            } else if (c == '?') {
                regex += "[^/]";
            } else {
                switch (c) {
                    case '.': case '(': case ')': case '+':
                    case '|': case '^': case '$': case '\\':
                        regex += '\\';
                        regex += c;
                        break;
                    default:
                        regex += c;
                }
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
            case '{':
                in_brace = true;
                regex += "(?:";
                break;
            case '.':
            case '(':
            case ')':
            case '+':
            case '|':
            case '^':
            case '$':
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

// Directories to always skip (before gitignore parsing)
static bool is_always_skipped_dir(const std::string & name) {
    static const std::vector<std::string> skip = {
        ".git", "node_modules", "__pycache__", ".venv", "venv"
    };
    for (const auto & s : skip) {
        if (name == s) return true;
    }
    return false;
}

// Find the git root by walking up from start looking for .git/
static std::string find_git_root(const fs::path & start) {
    try {
        fs::path current = fs::absolute(start);
        while (true) {
            if (fs::exists(current / ".git")) {
                return current.string();
            }
            fs::path parent = current.parent_path();
            if (parent == current) break;
            current = parent;
        }
    } catch (const fs::filesystem_error &) {}
    return "";
}

struct gitignore_pattern {
    std::regex re;
    bool negation;
    bool dir_only;
};

static std::vector<gitignore_pattern> parse_gitignore(const fs::path & gitignore_path) {
    std::vector<gitignore_pattern> patterns;

    std::ifstream file(gitignore_path);
    if (!file.is_open()) return patterns;

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        std::string pat = line;
        bool negation = false;
        bool dir_only = false;
        bool anchored = false;

        if (pat[0] == '!') {
            negation = true;
            pat = pat.substr(1);
        }

        if (!pat.empty() && pat.back() == '/') {
            dir_only = true;
            pat.pop_back();
        }

        if (pat.empty()) continue;

        if (pat[0] == '/') {
            anchored = true;
            pat = pat.substr(1);
        } else if (pat.find('/') != std::string::npos) {
            anchored = true;
        }

        std::string regex_str;
        if (anchored) {
            regex_str = "^" + glob_to_regex(pat) + "(/.*)?$";
        } else {
            regex_str = "(^|/)" + glob_to_regex(pat) + "(/.*)?$";
        }

        try {
            patterns.push_back({
                std::regex(regex_str, std::regex::ECMAScript),
                negation,
                dir_only
            });
        } catch (const std::regex_error &) {}
    }

    return patterns;
}

static bool is_gitignored(const std::string & rel_path, bool is_dir,
                           const std::vector<gitignore_pattern> & patterns) {
    bool ignored = false;
    for (const auto & p : patterns) {
        if (p.dir_only && !is_dir) continue;
        if (std::regex_search(rel_path, p.re)) {
            ignored = !p.negation;
        }
    }
    return ignored;
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

    // Load gitignore patterns
    std::string git_root = find_git_root(base_path);
    std::vector<gitignore_pattern> gi_patterns;
    fs::path git_root_path;
    bool have_gi = false;

    if (!git_root.empty()) {
        git_root_path = fs::path(git_root);
        fs::path gi_file = git_root_path / ".gitignore";
        if (fs::exists(gi_file)) {
            gi_patterns = parse_gitignore(gi_file);
            have_gi = true;
        }
    }

    // Collect matching files with modification times
    std::vector<std::pair<fs::path, fs::file_time_type>> matches;
    const int limit = 500;
    bool match_path = (pattern.find('/') != std::string::npos || pattern.find("**") != std::string::npos);

    try {
        auto it = fs::recursive_directory_iterator(
            base_path, fs::directory_options::skip_permission_denied);

        for (; it != fs::recursive_directory_iterator(); ++it) {
            const auto & entry = *it;

            // Skip always-ignored directories and gitignored directories
            if (entry.is_directory()) {
                std::string dirname = entry.path().filename().string();
                bool skip = is_always_skipped_dir(dirname);
                if (!skip && have_gi) {
                    std::string rel = fs::relative(entry.path(), git_root_path).string();
                    std::replace(rel.begin(), rel.end(), '\\', '/');
                    skip = is_gitignored(rel, true, gi_patterns);
                }
                if (skip) {
                    it.disable_recursion_pending();
                }
                continue;
            }

            if (!entry.is_regular_file()) continue;

            // Check gitignore for files
            if (have_gi) {
                std::string rel = fs::relative(entry.path(), git_root_path).string();
                std::replace(rel.begin(), rel.end(), '\\', '/');
                if (is_gitignored(rel, false, gi_patterns)) {
                    continue;
                }
            }

            // Match against full relative path or just filename depending on pattern
            std::string to_match;
            if (match_path) {
                to_match = fs::relative(entry.path(), base_path).string();
            } else {
                to_match = entry.path().filename().string();
            }

            if (std::regex_match(to_match, pattern_regex)) {
                matches.emplace_back(entry.path(), entry.last_write_time());
                if ((int)matches.size() >= limit) {
                    break;
                }
            }
        }
    } catch (const fs::filesystem_error &) {
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
    "Find files matching a glob pattern. Supports * (any chars), ** (any path), ? (single char), [abc] (char class), {a,b} (alternatives). Results sorted by modification time.",
    R"json({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g. '*.cpp', 'src/**/*.ts', '*.{jpg,png,gif}')"
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
