#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Metadata parsed from SKILL.md frontmatter
struct skill_metadata {
    std::string name;           // Required: 1-64 chars, lowercase+numbers+hyphens
    std::string description;    // Required: 1-1024 chars
    std::string path;           // Absolute path to SKILL.md
    std::string license;        // Optional
    std::string compatibility;  // Optional: environment requirements
    json metadata;              // Optional: additional key-value pairs
};

// Manages skill discovery and prompt generation
// Implements agentskills.io specification
class skills_manager {
public:
    // Discover skills from a list of search paths
    // Each path should be a directory containing skill subdirectories
    // Returns number of skills discovered
    int discover(const std::vector<std::string> & search_paths);

    // Get all discovered skills
    const std::vector<skill_metadata> & get_skills() const { return skills_; }

    // Generate XML section for system prompt injection
    // Returns empty string if no skills discovered
    std::string generate_prompt_section() const;

    // Check if a skill name is valid according to spec
    // - 1-64 characters
    // - lowercase letters, numbers, hyphens only
    // - cannot start or end with hyphen
    // - no consecutive hyphens
    static bool validate_name(const std::string & name);

private:
    std::vector<skill_metadata> skills_;

    // Parse a single skill directory
    // Returns nullopt if SKILL.md not found or invalid
    std::optional<skill_metadata> parse_skill(const std::string & skill_dir);

    // Parse YAML frontmatter from SKILL.md content
    // Only extracts frontmatter, not the markdown body
    std::optional<skill_metadata> parse_frontmatter(const std::string & content,
                                                     const std::string & path);

    // Escape XML special characters
    static std::string escape_xml(const std::string & str);
};
