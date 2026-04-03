#include "../tool-registry.h"
#include "../permission.h"

#include "base64.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <iterator>
#include <set>

namespace fs = std::filesystem;

static const int DEFAULT_LIMIT = 2000;
static const int MAX_LINE_LENGTH = 2000;

static const std::set<std::string> IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"};

static std::string get_mime_type(const std::string & ext) {
    if (ext == ".png")                  return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".gif")                  return "image/gif";
    if (ext == ".bmp")                  return "image/bmp";
    if (ext == ".webp")                 return "image/webp";
    return "application/octet-stream";
}

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

    // Check for image files
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (IMAGE_EXTENSIONS.count(ext)) {
        if (!ctx.has_vision) {
            return {true, "Binary image file: " + path.string() + " (model does not support vision)", ""};
        }

        // Cap image size at 10MB (matches server's HTTP download limit)
        static const size_t MAX_IMAGE_SIZE = 10 * 1024 * 1024;
        auto file_size = fs::file_size(path);
        if (file_size > MAX_IMAGE_SIZE) {
            return {true, "Image file too large: " + path.string() + " (" +
                    std::to_string(file_size / (1024 * 1024)) + "MB, max 10MB)", ""};
        }

        // Read binary
        std::ifstream img(path, std::ios::binary);
        if (!img.is_open()) {
            return {false, "", "Cannot open image file: " + path.string()};
        }
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(img)),
                                    std::istreambuf_iterator<char>());

        // Base64 encode
        std::string b64 = base64::encode(
            reinterpret_cast<const char *>(bytes.data()), bytes.size());

        std::string mime     = get_mime_type(ext);
        std::string label    = "Image: " + path.filename().string() + " [" + mime + "]";
        std::string data_uri = "data:" + mime + ";base64," + b64;

        tool_result result;
        result.success     = true;
        result.output      = label;
        result.image_bytes = std::move(bytes);
        result.image_mime  = mime;
        result.content = json::array({
            {{"type", "text"}, {"text", label}},
            {{"type", "image_url"}, {"image_url", {{"url", data_uri}}}}
        });
        return result;
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
