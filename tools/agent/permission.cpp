#include "permission.h"
#include "console.h"

#include <algorithm>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

permission_manager::permission_manager() {
    // Set default permissions
    defaults_[permission_type::BASH]       = permission_state::ASK;
    defaults_[permission_type::FILE_READ]  = permission_state::ALLOW;
    defaults_[permission_type::FILE_WRITE] = permission_state::ASK;
    defaults_[permission_type::FILE_EDIT]  = permission_state::ASK;
    defaults_[permission_type::GLOB]       = permission_state::ALLOW;

    // Dangerous bash patterns (always ask with warning)
    dangerous_patterns_ = {
        "rm -rf", "rm -r /", "sudo ", "chmod 777",
        "curl | sh", "wget | sh", "> /dev/",
        "dd if=", "mkfs.", ":(){:|:&};:"
    };

    // Safe bash patterns (auto-allow)
    safe_patterns_ = {
        "ls", "pwd", "cat ", "head ", "tail ",
        "grep ", "find ", "wc ", "diff ",
        "git status", "git log", "git diff", "git branch",
        "echo ", "which ", "type ", "file "
    };
}

void permission_manager::set_project_root(const std::string & path) {
    project_root_ = fs::absolute(path).string();
}

bool permission_manager::matches_pattern(const std::string & cmd, const std::vector<std::string> & patterns) const {
    for (const auto & pattern : patterns) {
        if (cmd.find(pattern) == 0 || cmd.find(" " + pattern) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool permission_manager::is_path_in_project(const std::string & path) const {
    if (project_root_.empty()) return true;

    try {
        std::string abs_path = fs::absolute(path).string();
        return abs_path.find(project_root_) == 0;
    } catch (...) {
        return false;
    }
}

permission_state permission_manager::check_permission(const permission_request & request) {
    // Check session overrides first
    std::string key = request.tool_name + ":" + request.details;
    auto it = session_overrides_.find(key);
    if (it != session_overrides_.end()) {
        return it->second;
    }

    // For bash commands, check patterns
    if (request.type == permission_type::BASH) {
        if (matches_pattern(request.details, dangerous_patterns_)) {
            return permission_state::ASK;  // Always ask for dangerous commands
        }
        if (matches_pattern(request.details, safe_patterns_)) {
            return permission_state::ALLOW;  // Auto-allow safe commands
        }
    }

    // Return default for this type
    auto def_it = defaults_.find(request.type);
    if (def_it != defaults_.end()) {
        return def_it->second;
    }

    return permission_state::ASK;  // Default to asking
}

permission_response permission_manager::prompt_user(const permission_request & request) {
    console::set_display(DISPLAY_TYPE_RESET);

    // Display permission request
    console::log("\n");
    console::log("+-- PERMISSION: %s ", request.tool_name.c_str());
    for (size_t i = request.tool_name.length() + 16; i < 60; i++) console::log("-");
    console::log("+\n");

    if (!request.details.empty()) {
        console::log("| %s\n", request.details.c_str());
    }

    if (request.is_dangerous) {
        console::set_display(DISPLAY_TYPE_ERROR);
        console::log("| WARNING: Potentially dangerous operation\n");
        console::set_display(DISPLAY_TYPE_RESET);
    }

    console::log("+");
    for (int i = 0; i < 59; i++) console::log("-");
    console::log("+\n");

    console::log("| [y]es  [n]o  [a]lways  [N]ever\n");
    console::log("> ");
    console::flush();

    std::string input;
    std::getline(std::cin, input);

    if (input.empty() || input == "n" || input == "N" || input == "no") {
        return permission_response::DENY_ONCE;
    }
    if (input == "y" || input == "Y" || input == "yes") {
        return permission_response::ALLOW_ONCE;
    }
    if (input == "a" || input == "A" || input == "always") {
        // Store session override
        std::string key = request.tool_name + ":" + request.details;
        session_overrides_[key] = permission_state::ALLOW_SESSION;
        return permission_response::ALLOW_ALWAYS;
    }
    if (input == "never") {
        std::string key = request.tool_name + ":" + request.details;
        session_overrides_[key] = permission_state::DENY_SESSION;
        return permission_response::DENY_ALWAYS;
    }

    // Default to deny
    return permission_response::DENY_ONCE;
}

void permission_manager::record_tool_call(const std::string & tool, const std::string & args_hash) {
    // Check if this is same as last call
    if (!recent_calls_.empty()) {
        auto & last = recent_calls_.back();
        if (last.tool == tool && last.args_hash == args_hash) {
            last.count++;
            return;
        }
    }

    // Add new record
    recent_calls_.push_back({tool, args_hash, 1});

    // Keep only last 10 records
    if (recent_calls_.size() > 10) {
        recent_calls_.erase(recent_calls_.begin());
    }
}

bool permission_manager::is_doom_loop(const std::string & tool, const std::string & args_hash) const {
    if (recent_calls_.empty()) return false;

    const auto & last = recent_calls_.back();
    return last.tool == tool && last.args_hash == args_hash && last.count >= 3;
}

void permission_manager::clear_session() {
    session_overrides_.clear();
    recent_calls_.clear();
}
