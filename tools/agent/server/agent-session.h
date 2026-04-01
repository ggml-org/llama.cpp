#pragma once

#include "../agent-loop.h"
#include "../permission-async.h"

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// Forward declarations
struct server_context;
struct common_params;

// Configuration for creating a new session
struct agent_session_config {
    bool yolo_mode = false;               // Skip permission prompts
    int max_iterations = 50;
    int tool_timeout_ms = 120000;
    std::string working_dir;

    // Skills configuration (agentskills.io spec)
    bool enable_skills = true;
    std::vector<std::string> extra_skills_paths;

    // AGENTS.md configuration (agents.md spec)
    bool enable_agents_md = true;
};

// State of an agent session
enum class agent_session_state {
    IDLE,              // Ready for input
    RUNNING,           // Processing a prompt
    COMPLETED,         // Session ended normally
    ERROR              // Session ended with error
};

// Information about a session (for listing)
struct agent_session_info {
    std::string id;
    agent_session_state state;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_activity;
    int message_count;
    session_stats stats;
};

// An individual agent session
class agent_session {
public:
    agent_session(const std::string & id,
                  server_context & server_ctx,
                  const common_params & params,
                  const agent_session_config & config);

    ~agent_session();

    // Get session ID
    const std::string & id() const { return id_; }

    // Get current state
    agent_session_state state() const { return state_.load(); }

    // Get session info
    agent_session_info info() const;

    // Send a message and get streaming events
    // Returns immediately - events are delivered via callback
    // Call is_complete() and get_result() to check status
    void send_message(const std::string & content,
                      agent_event_callback on_event);

    // Check if the current operation is complete
    bool is_complete() const { return !is_running_.load(); }

    // Get the result of the last operation (only valid after is_complete())
    std::optional<agent_loop_result> get_result();

    // Cancel the current operation
    void cancel();

    // Get pending permissions
    std::vector<permission_request_async> pending_permissions();

    // Respond to a permission request
    bool respond_permission(const std::string & request_id, bool allowed, permission_scope scope);

    // Get conversation history
    json get_messages() const;

    // Get session statistics
    session_stats get_stats() const;

    // Clear conversation history
    void clear();

private:
    std::string id_;
    server_context & server_ctx_;
    const common_params & params_;
    agent_session_config config_;

    std::unique_ptr<agent_loop> loop_;
    permission_manager_async permissions_;
    std::atomic<agent_session_state> state_{agent_session_state::IDLE};
    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_interrupted_{false};

    // Result from last operation
    std::optional<agent_loop_result> last_result_;
    mutable std::mutex result_mutex_;

    // Background processing thread
    std::thread worker_thread_;

    // Timestamps
    std::chrono::steady_clock::time_point created_at_;
    std::chrono::steady_clock::time_point last_activity_;

    // Discovered Skills and AGENTS.md content (cached at session creation)
    std::string skills_prompt_section_;
    std::string agents_md_prompt_section_;
};

// Manages multiple agent sessions
class agent_session_manager {
public:
    agent_session_manager(server_context & server_ctx, const common_params & params);
    ~agent_session_manager();

    // Create a new session with the given configuration
    // Returns session ID
    std::string create_session(const agent_session_config & config = {});

    // Get a session by ID (nullptr if not found)
    // Returns shared_ptr so the session stays alive for the full request
    std::shared_ptr<agent_session> get_session(const std::string & id);

    // Delete a session
    bool delete_session(const std::string & id);

    // List all sessions
    std::vector<agent_session_info> list_sessions() const;

    // Get session count
    size_t session_count() const;

    // Clean up expired/idle sessions (optional TTL in seconds)
    void cleanup(int idle_timeout_seconds = 3600);

private:
    server_context & server_ctx_;
    const common_params & params_;

    mutable std::mutex mutex_;
    std::map<std::string, std::shared_ptr<agent_session>> sessions_;
    std::atomic<uint64_t> session_counter_{0};

    std::string generate_session_id();
};
