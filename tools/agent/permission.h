#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>

enum class permission_state {
    ALLOW,         // Auto-execute
    ASK,           // Prompt user
    DENY,          // Block
    ALLOW_SESSION, // User chose "always" for this session
    DENY_SESSION   // User chose "never" for this session
};

enum class permission_type {
    BASH,
    FILE_READ,
    FILE_WRITE,
    FILE_EDIT,
    GLOB,
    EXTERNAL_DIR,  // Operation outside working directory
};

struct permission_request {
    permission_type type;
    std::string tool_name;
    std::string description;
    std::string details;  // command, file path, etc.
    bool is_dangerous = false;
};

enum class permission_response {
    ALLOW_ONCE,
    DENY_ONCE,
    ALLOW_ALWAYS,
    DENY_ALWAYS,
};

class permission_manager {
public:
    permission_manager();

    // Set project root for external directory checks
    void set_project_root(const std::string & path);

    // Enable yolo mode (skip all permission prompts)
    void set_yolo_mode(bool enabled) { yolo_mode_ = enabled; }

    // Check if a tool execution is allowed
    permission_state check_permission(const permission_request & request);

    // Interactive prompt for permission (returns user's choice)
    permission_response prompt_user(const permission_request & request);

    // Record a tool call for doom-loop detection
    void record_tool_call(const std::string & tool, const std::string & args_hash);

    // Check for doom-loop (3+ identical calls)
    bool is_doom_loop(const std::string & tool, const std::string & args_hash) const;

    // Clear session state
    void clear_session();

private:
    std::string project_root_;
    bool yolo_mode_ = false;
    std::map<std::string, permission_state> session_overrides_;

    // Recent tool calls for doom-loop detection
    struct tool_call_record {
        std::string tool;
        std::string args_hash;
        int count;
    };
    std::vector<tool_call_record> recent_calls_;

    // Default permission states
    std::map<permission_type, permission_state> defaults_;

    // Dangerous bash patterns
    std::vector<std::string> dangerous_patterns_;

    // Safe bash patterns (auto-allow)
    std::vector<std::string> safe_patterns_;

    bool matches_pattern(const std::string & cmd, const std::vector<std::string> & patterns) const;
    bool is_path_in_project(const std::string & path) const;

public:
    // Check if a file path is sensitive (should be blocked)
    static bool is_sensitive_file(const std::string & path);

    // Check if path is outside working directory
    bool is_external_path(const std::string & path) const;
};
