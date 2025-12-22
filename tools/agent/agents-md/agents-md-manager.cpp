#include "agents-md-manager.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

std::string agents_md_manager::escape_xml_attr(const std::string & str) {
    std::string result;
    result.reserve(str.size());
    for (char c : str) {
        switch (c) {
            case '&':  result += "&amp;";  break;
            case '<':  result += "&lt;";   break;
            case '>':  result += "&gt;";   break;
            case '"':  result += "&quot;"; break;
            case '\'': result += "&apos;"; break;
            default:   result += c;        break;
        }
    }
    return result;
}

std::string agents_md_manager::find_git_root(const std::string & start_dir) {
    try {
        fs::path current = fs::absolute(start_dir);

        // Walk up until we find .git or hit root
        while (!current.empty() && current.has_parent_path()) {
            fs::path git_dir = current / ".git";
            if (fs::exists(git_dir)) {
                return current.string();
            }

            fs::path parent = current.parent_path();
            if (parent == current) {
                break;  // Reached filesystem root
            }
            current = parent;
        }
    } catch (const fs::filesystem_error &) {
        // Ignore errors, return empty
    }

    return "";
}

std::optional<std::string> agents_md_manager::read_file(const std::string & path) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return std::nullopt;
        }

        // Read content
        std::ostringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();

        // Check for binary content (null bytes in first 8KB)
        size_t check_size = std::min(content.size(), size_t(8192));
        if (content.substr(0, check_size).find('\0') != std::string::npos) {
            return std::nullopt;  // Binary file
        }

        return content;
    } catch (...) {
        return std::nullopt;
    }
}

int agents_md_manager::discover(const std::string & working_dir) {
    return discover(working_dir, "");
}

int agents_md_manager::discover(const std::string & working_dir, const std::string & config_dir) {
    files_.clear();

    try {
        fs::path current = fs::absolute(working_dir);
        std::string git_root = find_git_root(working_dir);
        fs::path stop_at = git_root.empty() ? current : fs::path(git_root);

        int depth = 0;
        const int max_depth = 100;  // Sanity limit

        // Walk from working dir up to git root (or just working dir if not in git)
        while (depth < max_depth) {
            // Check for AGENTS.md in current directory
            fs::path agents_path = current / "AGENTS.md";

            if (fs::exists(agents_path) && fs::is_regular_file(agents_path)) {
                auto content = read_file(agents_path.string());
                if (content && !content->empty()) {
                    agents_md_file file;
                    file.path = fs::absolute(agents_path).string();
                    file.content = *content;
                    file.depth = depth;

                    // Compute relative path from git root or working dir
                    if (!git_root.empty()) {
                        file.relative_path = fs::relative(agents_path, git_root).string();
                    } else {
                        file.relative_path = "AGENTS.md";
                    }

                    files_.push_back(file);
                }
            }

            // Stop if we've reached git root or filesystem root
            if (current == stop_at) {
                break;
            }

            fs::path parent = current.parent_path();
            if (parent == current) {
                break;  // Reached filesystem root
            }

            current = parent;
            depth++;
        }

        // Check for global AGENTS.md in config directory (lowest precedence)
        if (!config_dir.empty()) {
            fs::path global_agents = fs::path(config_dir) / "AGENTS.md";
            if (fs::exists(global_agents) && fs::is_regular_file(global_agents)) {
                auto content = read_file(global_agents.string());
                if (content && !content->empty()) {
                    agents_md_file file;
                    file.path = fs::absolute(global_agents).string();
                    file.content = *content;
                    file.depth = depth + 1;  // Lower precedence than any project file
                    file.relative_path = "(global)";
                    files_.push_back(file);
                }
            }
        }
    } catch (const fs::filesystem_error &) {
        // Ignore filesystem errors
    }

    // Files are already in order (closest first, global last)
    return static_cast<int>(files_.size());
}

size_t agents_md_manager::total_content_size() const {
    size_t total = 0;
    for (const auto & file : files_) {
        total += file.content.size();
    }
    return total;
}

std::string agents_md_manager::generate_prompt_section() const {
    if (files_.empty()) {
        return "";
    }

    std::ostringstream xml;
    xml << "<project_context>\n";
    xml << "Project guidance from AGENTS.md files (closest to working directory takes precedence):\n\n";

    for (const auto & file : files_) {
        xml << "<agents_md path=\"" << escape_xml_attr(file.relative_path) << "\"";
        if (file.depth == 0) {
            xml << " precedence=\"highest\"";
        }
        xml << ">\n";
        xml << file.content;
        // Ensure content ends with newline
        if (!file.content.empty() && file.content.back() != '\n') {
            xml << "\n";
        }
        xml << "</agents_md>\n\n";
    }

    xml << "</project_context>";
    return xml.str();
}
