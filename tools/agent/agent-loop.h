#pragma once

#include "tool-registry.h"
#include "permission.h"
#include "permission-async.h"
#include "chat.h"

#include "server-context.h"
#include "server-task.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <functional>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Callback for reporting tool calls (used by subagents to report to parent display)
// Parameters: tool_name, args_summary, elapsed_ms
using tool_call_callback = std::function<void(const std::string &, const std::string &, int)>;

// Stop reasons for the agent loop
enum class agent_stop_reason {
    COMPLETED,        // Model finished without tool calls
    MAX_ITERATIONS,   // Hit iteration limit
    USER_CANCELLED,   // User interrupted
    AGENT_ERROR,      // Error occurred (not ERROR due to Windows macro conflict)
};

// Configuration for the agent
struct agent_config {
    int max_iterations = 50;
    int tool_timeout_ms = 120000;
    std::string working_dir;
    bool verbose = false;
    bool yolo_mode = false;  // Skip all permission prompts

    // Skills configuration (agentskills.io spec)
    bool enable_skills = true;
    std::vector<std::string> skills_search_paths;  // Additional search paths
    std::string skills_prompt_section;             // Pre-generated XML for prompt injection

    // AGENTS.md configuration (agents.md spec)
    bool enable_agents_md = true;
    std::string agents_md_prompt_section;          // Pre-generated XML for prompt injection
};

// Result from running the agent loop
struct agent_loop_result {
    agent_stop_reason stop_reason;
    std::string final_response;
    int iterations = 0;
};

// Session-level statistics for token tracking
struct session_stats {
    int32_t total_input = 0;       // Total prompt tokens processed
    int32_t total_output = 0;      // Total tokens generated
    int32_t total_cached = 0;      // Total tokens served from KV cache
    double total_prompt_ms = 0;    // Total prompt evaluation time
    double total_predicted_ms = 0; // Total generation time

    // Subagent-specific stats (subset of totals above)
    int32_t subagent_input = 0;    // Prompt tokens from subagents
    int32_t subagent_output = 0;   // Output tokens from subagents
    int32_t subagent_cached = 0;   // Cached tokens from subagents
    int32_t subagent_count = 0;    // Number of subagent runs
};

// Event types for streaming API
enum class agent_event_type {
    TEXT_DELTA,           // Streaming LLM token output
    REASONING_DELTA,      // Streaming reasoning/thinking content
    TOOL_START,           // Tool execution starting
    TOOL_RESULT,          // Tool execution completed
    PERMISSION_REQUIRED,  // Permission required - waiting for response
    PERMISSION_RESOLVED,  // Permission was granted or denied
    ITERATION_START,      // New iteration starting
    COMPLETED,            // Agent finished successfully
    ERROR,                // Error occurred
};

// Event data for streaming API
struct agent_event {
    agent_event_type type;
    json data;  // Event-specific payload

    // Convenience constructors
    static agent_event text_delta(const std::string & content) {
        return {agent_event_type::TEXT_DELTA, {{"content", content}}};
    }

    static agent_event reasoning_delta(const std::string & content) {
        return {agent_event_type::REASONING_DELTA, {{"content", content}}};
    }

    static agent_event tool_start(const std::string & name, const std::string & args) {
        return {agent_event_type::TOOL_START, {{"name", name}, {"args", args}}};
    }

    static agent_event tool_result(const std::string & name, bool success,
                                    const std::string & output, int64_t duration_ms) {
        return {agent_event_type::TOOL_RESULT, {
            {"name", name},
            {"success", success},
            {"output", output},
            {"duration_ms", duration_ms}
        }};
    }

    static agent_event permission_required(const std::string & request_id,
                                            const std::string & tool_name,
                                            const std::string & details,
                                            bool is_dangerous) {
        return {agent_event_type::PERMISSION_REQUIRED, {
            {"request_id", request_id},
            {"tool", tool_name},
            {"details", details},
            {"dangerous", is_dangerous}
        }};
    }

    static agent_event permission_resolved(const std::string & request_id, bool allowed) {
        return {agent_event_type::PERMISSION_RESOLVED, {
            {"request_id", request_id},
            {"allowed", allowed}
        }};
    }

    static agent_event iteration_start(int iteration, int max_iterations) {
        return {agent_event_type::ITERATION_START, {
            {"iteration", iteration},
            {"max_iterations", max_iterations}
        }};
    }

    static agent_event completed(agent_stop_reason reason, const session_stats & stats) {
        std::string reason_str;
        switch (reason) {
            case agent_stop_reason::COMPLETED:      reason_str = "completed"; break;
            case agent_stop_reason::MAX_ITERATIONS: reason_str = "max_iterations"; break;
            case agent_stop_reason::USER_CANCELLED: reason_str = "user_cancelled"; break;
            case agent_stop_reason::AGENT_ERROR:    reason_str = "error"; break;
        }
        return {agent_event_type::COMPLETED, {
            {"reason", reason_str},
            {"stats", {
                {"input_tokens", stats.total_input},
                {"output_tokens", stats.total_output},
                {"cached_tokens", stats.total_cached}
            }}
        }};
    }

    static agent_event error(const std::string & message) {
        return {agent_event_type::ERROR, {{"message", message}}};
    }
};

// Callback type for streaming events
using agent_event_callback = std::function<void(const agent_event &)>;

// The main agent loop class
class agent_loop {
public:
    // Standard constructor for main agent
    agent_loop(server_context & server_ctx,
               const common_params & params,
               const agent_config & config,
               std::atomic<bool> & is_interrupted);

    // Constructor for subagents with filtered tools and custom system prompt
    agent_loop(server_context & server_ctx,
               const common_params & params,
               const agent_config & config,
               std::atomic<bool> & is_interrupted,
               const std::set<std::string> & allowed_tools,
               const std::vector<std::string> & bash_patterns,  // For read-only bash filtering
               const std::string & custom_system_prompt,
               int subagent_depth,
               tool_call_callback on_tool_call = nullptr);

    // Run the agent loop with an initial user prompt
    agent_loop_result run(const std::string & user_prompt);

    // Run the agent loop with streaming events via callback
    // This is the API-friendly version that emits events instead of console output
    // The callback is called for each event (text deltas, tool calls, permissions, etc.)
    // should_stop is polled to check if the client wants to abort
    // async_perms: optional async permission manager for non-blocking permission handling
    agent_loop_result run_streaming(
        const std::string & user_prompt,
        agent_event_callback on_event,
        std::function<bool()> should_stop = nullptr,
        permission_manager_async * async_perms = nullptr);

    // Clear conversation history
    void clear();

    // Get current messages (for debugging)
    const json & get_messages() const { return messages_; }

    // Get session statistics
    const session_stats & get_stats() const { return stats_; }

private:
    // Build prompt + parser metadata using server chat template config.
    common_chat_params format_chat_with_tools(const std::vector<common_chat_tool> & chat_tools);

    // Generate a completion and get the parsed response with tool calls
    common_chat_msg generate_completion(result_timings & out_timings);

    // Generate completion with streaming events via callback
    common_chat_msg generate_completion_streaming(
        result_timings & out_timings,
        agent_event_callback on_event,
        std::function<bool()> should_stop);

    // Execute a single tool call
    tool_result execute_tool_call(const common_chat_tool_call & call);

    // Execute tool call with async permission handling (for streaming API)
    // Emits PERMISSION_REQUIRED events and waits for async responses
    tool_result execute_tool_call_async(
        const common_chat_tool_call & call,
        agent_event_callback on_event,
        permission_manager_async & async_perms,
        std::function<bool()> should_stop);

    // Format tool result as message
    void add_tool_result_message(const std::string & tool_name,
                                  const std::string & call_id,
                                  const tool_result & result);

    server_context & server_ctx_;
    const common_params * params_;  // Stored for subagent construction
    agent_config config_;
    std::atomic<bool> & is_interrupted_;

    json messages_;
    task_params task_defaults_;
    permission_manager permission_mgr_;
    tool_context tool_ctx_;
    session_stats stats_;

    // Subagent support
    std::set<std::string> allowed_tools_;     // Empty = all tools allowed
    std::vector<std::string> bash_patterns_;  // Allowed bash command prefixes (for read-only subagents)
    tool_call_callback on_tool_call_;         // Optional callback for tool reporting
    bool is_subagent_ = false;                // True if this is a subagent

    // Common initialization logic
    void init_common(const common_params & params);
    void init_system_prompt(const std::string & custom_system_prompt = "");
};
