#include "skills-manager.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

// Trim whitespace from both ends of a string
static std::string trim(const std::string & str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

// Parse a simple YAML key-value line like "key: value"
static std::pair<std::string, std::string> parse_yaml_line(const std::string & line) {
    size_t colon = line.find(':');
    if (colon == std::string::npos) {
        return {"", ""};
    }
    std::string key = trim(line.substr(0, colon));
    std::string value = trim(line.substr(colon + 1));

    // Remove quotes if present
    if (value.size() >= 2) {
        if ((value.front() == '"' && value.back() == '"') ||
            (value.front() == '\'' && value.back() == '\'')) {
            value = value.substr(1, value.size() - 2);
        }
    }

    return {key, value};
}

bool skills_manager::validate_name(const std::string & name) {
    if (name.empty() || name.size() > 64) {
        return false;
    }

    // Cannot start or end with hyphen
    if (name.front() == '-' || name.back() == '-') {
        return false;
    }

    // No XML tags allowed (spec requirement)
    if (name.find('<') != std::string::npos || name.find('>') != std::string::npos) {
        return false;
    }

    bool prev_hyphen = false;
    for (char c : name) {
        if (c == '-') {
            // No consecutive hyphens
            if (prev_hyphen) return false;
            prev_hyphen = true;
        } else if (std::islower(static_cast<unsigned char>(c))) {
            prev_hyphen = false;
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            prev_hyphen = false;
        } else {
            // Invalid character (uppercase, special chars, etc.)
            return false;
        }
    }

    return true;
}

std::string skills_manager::escape_xml(const std::string & str) {
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

std::optional<skill_metadata> skills_manager::parse_frontmatter(const std::string & content,
                                                                  const std::string & path) {
    // Find frontmatter delimiters (---)
    if (content.substr(0, 3) != "---") {
        return std::nullopt;
    }

    size_t end_delim = content.find("\n---", 3);
    if (end_delim == std::string::npos) {
        return std::nullopt;
    }

    std::string frontmatter = content.substr(3, end_delim - 3);

    skill_metadata skill;
    skill.path = path;

    // Parse YAML-like frontmatter line by line
    std::istringstream stream(frontmatter);
    std::string line;
    bool in_metadata = false;

    while (std::getline(stream, line)) {
        // Check for indentation before trimming (for metadata detection)
        bool is_indented = !line.empty() && (line.front() == ' ' || line.front() == '\t');
        line = trim(line);
        if (line.empty()) continue;

        // Check for metadata section
        if (line == "metadata:") {
            in_metadata = true;
            continue;
        }

        // Handle indented metadata key-value pairs
        if (in_metadata && is_indented) {
            auto [key, value] = parse_yaml_line(line);
            if (!key.empty() && !value.empty()) {
                skill.metadata[key] = value;
            }
            continue;
        } else if (in_metadata) {
            in_metadata = false;
        }

        auto [key, value] = parse_yaml_line(line);
        if (key.empty()) continue;

        if (key == "name") {
            skill.name = value;
        } else if (key == "description") {
            skill.description = value;
        } else if (key == "license") {
            skill.license = value;
        } else if (key == "compatibility") {
            skill.compatibility = value;
        } else if (key == "allowed-tools") {
            // Parse space-delimited tool list (experimental per spec)
            std::istringstream tools_stream(value);
            std::string tool;
            while (tools_stream >> tool) {
                skill.allowed_tools.push_back(tool);
            }
        }
    }

    // Validate required fields
    if (skill.name.empty() || skill.description.empty()) {
        return std::nullopt;
    }

    // Validate name format
    if (!validate_name(skill.name)) {
        return std::nullopt;
    }

    // Validate description length (max 1024 chars, truncate if needed)
    if (skill.description.size() > 1024) {
        skill.description = skill.description.substr(0, 1024);
    }

    // Validate compatibility length (max 500 chars per spec)
    if (skill.compatibility.size() > 500) {
        skill.compatibility = skill.compatibility.substr(0, 500);
    }

    return skill;
}

std::optional<skill_metadata> skills_manager::parse_skill(const std::string & skill_dir) {
    fs::path skill_path = fs::path(skill_dir) / "SKILL.md";

    if (!fs::exists(skill_path) || !fs::is_regular_file(skill_path)) {
        return std::nullopt;
    }

    // Read file content
    std::ifstream file(skill_path);
    if (!file) {
        return std::nullopt;
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    auto skill = parse_frontmatter(content, skill_path.string());
    if (!skill) {
        return std::nullopt;
    }

    // Verify skill name matches directory name (spec requirement)
    std::string dir_name = fs::path(skill_dir).filename().string();
    if (skill->name != dir_name) {
        return std::nullopt;
    }

    // Store skill directory for script execution
    skill->skill_dir = fs::absolute(skill_dir).string();

    // Discover scripts in scripts/ subdirectory
    fs::path scripts_dir = fs::path(skill_dir) / "scripts";
    if (fs::exists(scripts_dir) && fs::is_directory(scripts_dir)) {
        try {
            for (const auto & entry : fs::directory_iterator(scripts_dir)) {
                if (entry.is_regular_file()) {
                    std::string script_name = entry.path().filename().string();
                    // Skip hidden files
                    if (!script_name.empty() && script_name[0] != '.') {
                        skill->scripts.push_back("scripts/" + script_name);
                    }
                }
            }
            std::sort(skill->scripts.begin(), skill->scripts.end());
        } catch (const fs::filesystem_error &) {
            // Skip inaccessible scripts directory
        }
    }

    return skill;
}

int skills_manager::discover(const std::vector<std::string> & search_paths) {
    skills_.clear();
    int count = 0;

    for (const auto & search_path : search_paths) {
        if (!fs::exists(search_path) || !fs::is_directory(search_path)) {
            continue;
        }

        try {
            for (const auto & entry : fs::directory_iterator(search_path)) {
                if (!entry.is_directory()) {
                    continue;
                }

                // Skip hidden directories
                std::string name = entry.path().filename().string();
                if (!name.empty() && name[0] == '.') {
                    continue;
                }

                auto skill = parse_skill(entry.path().string());
                if (skill) {
                    // Check for duplicates (first discovered wins)
                    bool duplicate = false;
                    for (const auto & existing : skills_) {
                        if (existing.name == skill->name) {
                            duplicate = true;
                            break;
                        }
                    }

                    if (!duplicate) {
                        skills_.push_back(*skill);
                        count++;
                    }
                }
            }
        } catch (const fs::filesystem_error &) {
            // Skip inaccessible directories
            continue;
        }
    }

    // Sort skills by name for consistent ordering
    std::sort(skills_.begin(), skills_.end(),
              [](const skill_metadata & a, const skill_metadata & b) {
                  return a.name < b.name;
              });

    return count;
}

std::string skills_manager::generate_prompt_section() const {
    if (skills_.empty()) {
        return "";
    }

    std::ostringstream xml;
    xml << "<available_skills>\n";

    for (const auto & skill : skills_) {
        xml << "<skill>\n";
        xml << "  <name>" << escape_xml(skill.name) << "</name>\n";
        xml << "  <description>" << escape_xml(skill.description) << "</description>\n";
        xml << "  <location>" << escape_xml(skill.path) << "</location>\n";
        xml << "  <skill_dir>" << escape_xml(skill.skill_dir) << "</skill_dir>\n";

        // Include scripts if present
        if (!skill.scripts.empty()) {
            xml << "  <scripts>\n";
            for (const auto & script : skill.scripts) {
                xml << "    <script>" << escape_xml(script) << "</script>\n";
            }
            xml << "  </scripts>\n";
        }

        // Include allowed-tools if specified (experimental)
        if (!skill.allowed_tools.empty()) {
            xml << "  <allowed_tools>";
            for (size_t i = 0; i < skill.allowed_tools.size(); i++) {
                if (i > 0) xml << " ";
                xml << escape_xml(skill.allowed_tools[i]);
            }
            xml << "</allowed_tools>\n";
        }

        xml << "</skill>\n";
    }

    xml << "</available_skills>";
    return xml.str();
}
