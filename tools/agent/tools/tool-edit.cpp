#include "../tool-registry.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

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

    // Read file content
    std::string content = read_file(path);
    if (content.empty() && fs::file_size(path) > 0) {
        return {false, "", "Cannot read file: " + path.string()};
    }

    // Find occurrences
    size_t first_pos = content.find(old_string);
    if (first_pos == std::string::npos) {
        return {false, "", "old_string not found in file. Make sure you're using the exact text including whitespace and indentation."};
    }

    // Check for multiple occurrences
    size_t last_pos = content.rfind(old_string);
    if (first_pos != last_pos && !replace_all) {
        // Count occurrences
        int count = 0;
        size_t pos = 0;
        while ((pos = content.find(old_string, pos)) != std::string::npos) {
            count++;
            pos += old_string.length();
        }
        return {false, "", "Found " + std::to_string(count) + " occurrences of old_string. Provide more context to make it unique, or set replace_all=true to replace all occurrences."};
    }

    // Perform replacement
    std::string new_content;
    int replacements = 0;

    if (replace_all) {
        size_t pos = 0;
        size_t last_end = 0;
        while ((pos = content.find(old_string, last_end)) != std::string::npos) {
            new_content += content.substr(last_end, pos - last_end);
            new_content += new_string;
            last_end = pos + old_string.length();
            replacements++;
        }
        new_content += content.substr(last_end);
    } else {
        new_content = content.substr(0, first_pos) + new_string + content.substr(first_pos + old_string.length());
        replacements = 1;
    }

    // Write file
    if (!write_file(path, new_content)) {
        return {false, "", "Failed to write changes to file"};
    }

    std::string msg = "Successfully replaced " + std::to_string(replacements) + " occurrence(s) in " + path.string();
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
