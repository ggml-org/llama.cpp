#include "permission-async.h"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

permission_manager_async::permission_manager_async() {
    // Set default permissions (same as sync version)
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

void permission_manager_async::set_project_root(const std::string & path) {
    std::lock_guard<std::mutex> lock(mutex_);
    project_root_ = fs::absolute(path).string();
}

std::string permission_manager_async::generate_request_id() {
    uint64_t counter = request_counter_.fetch_add(1);
    std::stringstream ss;
    ss << "perm_" << std::hex << std::setfill('0') << std::setw(8) << counter;
    return ss.str();
}

bool permission_manager_async::is_compound_command(const std::string & cmd) {
    for (const auto & sep : {"|", "&&", "||", ";"}) {
        if (cmd.find(sep) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool permission_manager_async::matches_pattern(const std::string & cmd, const std::vector<std::string> & patterns) const {
    for (const auto & pattern : patterns) {
        if (cmd.find(pattern) == 0) {
            return true;
        }
    }
    return false;
}

bool permission_manager_async::is_path_in_project(const std::string & path) const {
    if (project_root_.empty()) return true;

    try {
        std::string abs_path = fs::absolute(path).string();
        if (abs_path == project_root_) return true;
        std::string prefix = project_root_;
        if (prefix.back() != '/' && prefix.back() != '\\') {
            prefix += '/';
        }
        return abs_path.find(prefix) == 0;
    } catch (...) {
        return false;
    }
}

permission_state permission_manager_async::check_permission(const permission_request & request) {
    // YOLO mode - allow everything
    if (yolo_mode_) {
        return permission_state::ALLOW;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Check session overrides first
    std::string key = permission_override_key(request.tool_name, request.details);
    auto it = session_overrides_.find(key);
    if (it != session_overrides_.end()) {
        return it->second;
    }

    // For bash commands, check patterns
    if (request.type == permission_type::BASH) {
        if (matches_pattern(request.details, dangerous_patterns_)) {
            return permission_state::ASK;
        }
        if (!is_compound_command(request.details) && matches_pattern(request.details, safe_patterns_)) {
            return permission_state::ALLOW;
        }
    }

    // Return default for this type
    auto def_it = defaults_.find(request.type);
    if (def_it != defaults_.end()) {
        return def_it->second;
    }

    return permission_state::ASK;
}

std::string permission_manager_async::request_permission(const permission_request & request) {
    std::string id = generate_request_id();

    permission_request_async async_req;
    async_req.id = id;
    async_req.request = request;
    async_req.created_at = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_requests_[id] = async_req;
    }

    // Notify callback if set (outside lock to avoid deadlock)
    if (callback_) {
        callback_(async_req);
    }

    return id;
}

bool permission_manager_async::respond(const std::string & request_id, bool allowed, permission_scope scope) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if request exists
    auto it = pending_requests_.find(request_id);
    if (it == pending_requests_.end()) {
        return false;  // Request not found or already responded
    }

    // Store response
    permission_response_async response;
    response.request_id = request_id;
    response.allowed = allowed;
    response.scope = scope;
    responses_[request_id] = response;

    // Handle session scope
    if (scope == permission_scope::SESSION) {
        const auto & req = it->second.request;
        std::string key = permission_override_key(req.tool_name, req.details);
        session_overrides_[key] = allowed ? permission_state::ALLOW_SESSION : permission_state::DENY_SESSION;
    }

    // Remove from pending
    pending_requests_.erase(it);

    // Wake up any waiting threads
    cv_.notify_all();

    return true;
}

std::optional<permission_response_async> permission_manager_async::wait_for_response(
    const std::string & request_id,
    int timeout_ms) {
    return wait_for_response_or_stop(request_id, timeout_ms, nullptr);
}

std::optional<permission_response_async> permission_manager_async::wait_for_response_or_stop(
    const std::string & request_id,
    int timeout_ms,
    std::function<bool()> stop_pred) {

    std::unique_lock<std::mutex> lock(mutex_);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (true) {
        // Check if response is available
        auto it = responses_.find(request_id);
        if (it != responses_.end()) {
            permission_response_async result = it->second;
            responses_.erase(it);
            return result;
        }

        // Check if request was cancelled
        if (pending_requests_.find(request_id) == pending_requests_.end()) {
            return std::nullopt;
        }

        // Check stop predicate
        if (stop_pred) {
            // Temporarily release lock to call stop_pred (it may access atomics)
            lock.unlock();
            bool should_stop = stop_pred();
            lock.lock();
            if (should_stop) {
                // Cancel this request and return
                pending_requests_.erase(request_id);
                cv_.notify_all();
                return std::nullopt;
            }
        }

        // Wait in short intervals (100ms) so we can check stop_pred frequently
        auto wait_until = std::min(deadline,
            std::chrono::steady_clock::now() + std::chrono::milliseconds(100));

        if (cv_.wait_until(lock, wait_until) == std::cv_status::timeout) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return std::nullopt;
            }
            // Otherwise just a 100ms interval — loop back to check stop_pred
        }
    }
}

void permission_manager_async::cancel_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_requests_.clear();
    cv_.notify_all();
}

std::vector<permission_request_async> permission_manager_async::pending() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<permission_request_async> result;
    result.reserve(pending_requests_.size());
    for (const auto & [id, req] : pending_requests_) {
        result.push_back(req);
    }
    return result;
}

bool permission_manager_async::is_pending(const std::string & request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_requests_.find(request_id) != pending_requests_.end();
}

bool permission_manager_async::cancel(const std::string & request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pending_requests_.find(request_id);
    if (it == pending_requests_.end()) {
        return false;
    }
    pending_requests_.erase(it);
    cv_.notify_all();
    return true;
}

void permission_manager_async::record_tool_call(const std::string & tool, const std::string & args_hash) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!recent_calls_.empty()) {
        auto & last = recent_calls_.back();
        if (last.tool == tool && last.args_hash == args_hash) {
            last.count++;
            return;
        }
    }

    recent_calls_.push_back({tool, args_hash, 1});

    if (recent_calls_.size() > 10) {
        recent_calls_.erase(recent_calls_.begin());
    }
}

bool permission_manager_async::is_doom_loop(const std::string & tool, const std::string & args_hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (recent_calls_.empty()) return false;

    const auto & last = recent_calls_.back();
    return last.tool == tool && last.args_hash == args_hash && last.count >= 3;
}

void permission_manager_async::clear_session() {
    std::lock_guard<std::mutex> lock(mutex_);
    session_overrides_.clear();
    recent_calls_.clear();
    pending_requests_.clear();
    responses_.clear();
    cv_.notify_all();
}

bool permission_manager_async::is_sensitive_file(const std::string & path) {
    // Delegate to the static sync version
    return permission_manager::is_sensitive_file(path);
}

bool permission_manager_async::is_external_path(const std::string & path) const {
    return !is_path_in_project(path);
}
