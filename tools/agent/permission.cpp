#include "permission.h"
#include "console.h"

#include <algorithm>
#include <iostream>
#include <filesystem>

#if defined(_WIN32)
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

// Read a single character without waiting for Enter
static char read_single_char() {
#if defined(_WIN32)
    return static_cast<char>(_getch());
#else
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    char ch = static_cast<char>(getchar());
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
#endif
}

namespace fs = std::filesystem;

permission_manager::permission_manager() {
    // Set default permissions
    defaults_[permission_type::BASH]       = permission_state::ASK;
    defaults_[permission_type::FILE_READ]  = permission_state::ALLOW;
    defaults_[permission_type::FILE_WRITE] = permission_state::ASK;
    defaults_[permission_type::FILE_EDIT]  = permission_state::ASK;
    defaults_[permission_type::GLOB]       = permission_state::ALLOW;
    defaults_[permission_type::EXTERNAL_DIR] = permission_state::ASK;

    // Dangerous bash patterns (always ask with warning)
    dangerous_patterns_ = {
        // Destructive commands
        "rm -rf", "rm -r /", "rm -f", "rmdir",
        // Privilege escalation
        "sudo ", "su -", "doas ",
        // Dangerous permissions
        "chmod 777", "chmod -R", "chown -R",
        // Remote code execution
        "curl | sh", "curl | bash", "wget | sh", "wget | bash",
        "curl -s | sh", "wget -O - |",
        // System damage
        "> /dev/", "dd if=", "mkfs.", ":(){:|:&};:",
        // Package managers (can modify system)
        "pip install", "pip3 install", "npm i -g", "npm install -g",
        "brew install", "apt install", "apt-get install", "yum install",
        // Git destructive
        "git push -f", "git push --force", "git reset --hard",
        // Process control
        "kill -9", "killall", "pkill"
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
        // Exact match is always in project
        if (abs_path == project_root_) return true;
        // Check that path is within project_root directory (not just a prefix match)
        // e.g., /repo/file.txt is in /repo, but /repo_evil/file.txt is NOT
        std::string prefix = project_root_;
        if (prefix.back() != '/' && prefix.back() != '\\') {
            prefix += '/';
        }
        return abs_path.find(prefix) == 0;
    } catch (...) {
        return false;
    }
}

permission_state permission_manager::check_permission(const permission_request & request) {
    // YOLO mode - allow everything
    if (yolo_mode_) {
        return permission_state::ALLOW;
    }

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

    console::log("| [y]es  [n]o  [a]lways  [d]eny always: ");
    console::flush();

    char ch = read_single_char();
    console::log("%c\n", ch);  // Echo the character

    if (ch == 'n' || ch == 'N') {
        return permission_response::DENY_ONCE;
    }
    if (ch == 'y' || ch == 'Y') {
        return permission_response::ALLOW_ONCE;
    }
    if (ch == 'a' || ch == 'A') {
        // Store session override
        std::string key = request.tool_name + ":" + request.details;
        session_overrides_[key] = permission_state::ALLOW_SESSION;
        return permission_response::ALLOW_ALWAYS;
    }
    if (ch == 'd' || ch == 'D') {  // 'd' for deny always (since 'N' is deny once)
        std::string key = request.tool_name + ":" + request.details;
        session_overrides_[key] = permission_state::DENY_SESSION;
        return permission_response::DENY_ALWAYS;
    }

    // Default to deny for any other key
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

bool permission_manager::is_sensitive_file(const std::string & path) {
    // Get filename and extension
    fs::path p(path);
    std::string filename = p.filename().string();
    std::string ext = p.extension().string();

    // Convert to lowercase for comparison
    std::string filename_lower = filename;
    std::string ext_lower = ext;
    for (auto & c : filename_lower) c = std::tolower(c);
    for (auto & c : ext_lower) c = std::tolower(c);

    // Sensitive file patterns
    static const std::vector<std::string> sensitive_names = {
        ".env", ".env.local", ".env.production", ".env.development",
        ".netrc", ".npmrc", ".pypirc",
        "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
        "credentials", "credentials.json", "credentials.yaml",
        "secrets", "secrets.json", "secrets.yaml", "secrets.yml",
        ".htpasswd", ".htaccess",
        "shadow", "passwd",
        "private_key", "privatekey",
        "service_account", "service-account",
        "token", "token.json",
        "keystore", "keystore.jks",
        ".pgpass", ".my.cnf",
    };

    static const std::vector<std::string> sensitive_extensions = {
        ".pem", ".key", ".p12", ".pfx", ".jks",
        ".keystore", ".secret", ".secrets",
        ".cert", ".crt", ".cer",
    };

    // Check exact filename matches
    for (const auto & name : sensitive_names) {
        if (filename_lower == name) {
            return true;
        }
        // Also check if filename contains the sensitive name (e.g., "prod.env")
        if (filename_lower.find(name) != std::string::npos && name.front() != '.') {
            return true;
        }
    }

    // Check extensions
    for (const auto & sensitive_ext : sensitive_extensions) {
        if (ext_lower == sensitive_ext) {
            return true;
        }
    }

    // Check for AWS/cloud credentials
    if (filename_lower.find("aws") != std::string::npos &&
        (filename_lower.find("credential") != std::string::npos ||
         filename_lower.find("config") != std::string::npos)) {
        return true;
    }

    return false;
}

bool permission_manager::is_external_path(const std::string & path) const {
    return !is_path_in_project(path);
}
