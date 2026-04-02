#pragma once

#include "permission.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>

// Async permission request with unique ID
struct permission_request_async {
    std::string id;               // Unique request ID
    permission_request request;   // Original permission request
    std::chrono::steady_clock::time_point created_at;
};

// Scope for permission responses
enum class permission_scope {
    ONCE,     // Allow/deny this specific request only
    SESSION,  // Remember for this session
    // ALWAYS could be added for persistent storage
};

// Async permission response
struct permission_response_async {
    std::string request_id;
    bool allowed;
    permission_scope scope;
};

// Callback type for permission request notifications
// Called when a new permission request is queued
using permission_callback = std::function<void(const permission_request_async &)>;

// Async permission manager for API-based permission handling
// Instead of blocking on stdin, queues requests and waits for external responses
class permission_manager_async {
public:
    permission_manager_async();

    // Set project root for external directory checks
    void set_project_root(const std::string & path);

    // Enable yolo mode (skip all permission prompts)
    void set_yolo_mode(bool enabled) { yolo_mode_ = enabled; }

    // Check if a tool execution is allowed (non-blocking)
    // Returns ALLOW, DENY, or ASK based on rules
    permission_state check_permission(const permission_request & request);

    // Request permission asynchronously (non-blocking)
    // Returns unique request ID for tracking
    // If callback is set, it will be called with the new request
    std::string request_permission(const permission_request & request);

    // Respond to a pending permission request
    // Returns false if request_id not found or already responded
    bool respond(const std::string & request_id, bool allowed, permission_scope scope = permission_scope::ONCE);

    // Wait for a response to a specific request (blocking with timeout)
    // Returns nullopt if timeout expires or request not found
    std::optional<permission_response_async> wait_for_response(
        const std::string & request_id,
        int timeout_ms = 30000);

    // Wait for a response with a stop predicate (checks every 100ms)
    // Returns nullopt if timeout expires, request not found, or stop_pred returns true
    std::optional<permission_response_async> wait_for_response_or_stop(
        const std::string & request_id,
        int timeout_ms,
        std::function<bool()> stop_pred);

    // Cancel all pending requests and wake all waiters
    void cancel_all();

    // Get all pending permission requests
    std::vector<permission_request_async> pending();

    // Check if a specific request is still pending
    bool is_pending(const std::string & request_id) const;

    // Cancel a pending request (removes it without response)
    bool cancel(const std::string & request_id);

    // Set callback for new permission requests (for SSE notifications)
    void set_callback(permission_callback cb) { callback_ = std::move(cb); }

    // Record a tool call for doom-loop detection
    void record_tool_call(const std::string & tool, const std::string & args_hash);

    // Check for doom-loop (3+ identical calls)
    bool is_doom_loop(const std::string & tool, const std::string & args_hash) const;

    // Clear session state
    void clear_session();

    // Static utility: Check if a file path is sensitive
    static bool is_sensitive_file(const std::string & path);

    // Check if path is outside working directory
    bool is_external_path(const std::string & path) const;

private:
    std::string project_root_;
    bool yolo_mode_ = false;
    std::atomic<uint64_t> request_counter_{0};

    // Thread safety
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    // Pending requests (request_id -> request)
    std::map<std::string, permission_request_async> pending_requests_;

    // Responses (request_id -> response)
    std::map<std::string, permission_response_async> responses_;

    // Session overrides (tool:details -> state)
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

    // Dangerous and safe bash patterns
    std::vector<std::string> dangerous_patterns_;
    std::vector<std::string> safe_patterns_;

    // Optional callback for new requests
    permission_callback callback_;

    // Helper functions
    bool is_compound_command(const std::string & cmd);
    bool matches_pattern(const std::string & cmd, const std::vector<std::string> & patterns) const;
    bool is_path_in_project(const std::string & path) const;
    std::string generate_request_id();
};
