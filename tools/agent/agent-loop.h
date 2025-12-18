#pragma once

#include "tool-registry.h"
#include "permission.h"
#include "chat.h"

#include "server-context.h"
#include "server-task.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Stop reasons for the agent loop
enum class agent_stop_reason {
    COMPLETED,        // Model finished without tool calls
    MAX_ITERATIONS,   // Hit iteration limit
    USER_CANCELLED,   // User interrupted
    ERROR,            // Error occurred
};

// Configuration for the agent
struct agent_config {
    int max_iterations = 50;
    int tool_timeout_ms = 120000;
    std::string working_dir;
    bool verbose = false;
    bool yolo_mode = false;  // Skip all permission prompts
};

// Result from running the agent loop
struct agent_loop_result {
    agent_stop_reason stop_reason;
    std::string final_response;
    int iterations = 0;
};

// The main agent loop class
class agent_loop {
public:
    agent_loop(server_context & server_ctx,
               const common_params & params,
               const agent_config & config,
               std::atomic<bool> & is_interrupted);

    // Run the agent loop with an initial user prompt
    agent_loop_result run(const std::string & user_prompt);

    // Clear conversation history
    void clear();

    // Get current messages (for debugging)
    const json & get_messages() const { return messages_; }

private:
    // Generate a completion and get the parsed response with tool calls
    common_chat_msg generate_completion(result_timings & out_timings);

    // Execute a single tool call
    tool_result execute_tool_call(const common_chat_tool_call & call);

    // Format tool result as message
    void add_tool_result_message(const std::string & tool_name,
                                  const std::string & call_id,
                                  const tool_result & result);

    server_context & server_ctx_;
    agent_config config_;
    std::atomic<bool> & is_interrupted_;

    json messages_;
    task_params task_defaults_;
    permission_manager permission_mgr_;
    tool_context tool_ctx_;
};
