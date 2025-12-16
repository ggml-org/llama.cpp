#include "../tool-registry.h"
#include "../permission.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

static const int DEFAULT_LIMIT = 2000;
static const int MAX_LINE_LENGTH = 2000;

static tool_result read_execute(const json & args, const tool_context & ctx) {
    std::string file_path = args.value("file_path", "");
    int offset = args.value("offset", 0);
    int limit = args.value("limit", DEFAULT_LIMIT);

    if (file_path.empty()) {
        return {false, "", "file_path parameter is required"};
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

    // Check if it's a file (not directory)
    if (!fs::is_regular_file(path)) {
        return {false, "", "Not a regular file: " + path.string()};
    }

    // Block sensitive files
    if (permission_manager::is_sensitive_file(path.string())) {
        return {false, "", "Cannot read sensitive file (contains credentials/secrets): " + path.string()};
    }

    // Open file
    std::ifstream file(path);
    if (!file.is_open()) {
        return {false, "", "Cannot open file: " + path.string()};
    }

    // Read lines
    std::vector<std::string> lines;
    std::string line;
    int line_num = 0;
    int total_lines = 0;

    while (std::getline(file, line)) {
        total_lines++;
        if (line_num >= offset && (int)lines.size() < limit) {
            // Truncate very long lines
            if (line.length() > MAX_LINE_LENGTH) {
                line = line.substr(0, MAX_LINE_LENGTH) + "...";
            }
            lines.push_back(line);
        }
        line_num++;
    }

    // Build output with line numbers
    std::ostringstream output;
    for (size_t i = 0; i < lines.size(); i++) {
        int num = offset + i + 1;
        output << std::setw(6) << num << "| " << lines[i] << "\n";
    }

    // Add file info
    if (offset > 0 || offset + (int)lines.size() < total_lines) {
        output << "\n";
        if (offset > 0) {
            output << "[Lines " << (offset + 1) << "-" << (offset + lines.size());
        } else {
            output << "[Lines 1-" << lines.size();
        }
        output << " of " << total_lines << " total]";

        if (offset + (int)lines.size() < total_lines) {
            output << " Use offset=" << (offset + lines.size()) << " to read more.";
        }
    }

    return {true, output.str(), ""};
}

static tool_def read_tool = {
    "read",
    "Read the contents of a file. Returns numbered lines for easy reference. Use offset and limit for large files.",
    R"json({
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read (absolute or relative to working directory)"
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (0-based, default 0)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (default 2000)"
            }
        },
        "required": ["file_path"]
    })json",
    read_execute
};

REGISTER_TOOL(read, read_tool);
