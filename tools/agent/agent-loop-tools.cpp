#include "agent-loop.h"
#include "console.h"
#include "terminal-image.h"

#include <chrono>
#include <filesystem>
#include <functional>

tool_result agent_loop::execute_tool_call(const common_chat_tool_call & call) {
    auto & registry = tool_registry::instance();

    // Check if tool exists
    const tool_def * tool = registry.get_tool(call.name);
    if (!tool) {
        return {false, "", "Unknown tool: " + call.name};
    }

    // Parse arguments
    json args;
    try {
        args = json::parse(call.arguments);
    } catch (const json::parse_error & e) {
        return {false, "", std::string("Invalid JSON arguments: ") + e.what()};
    }

    // Determine permission type
    permission_type ptype = permission_type::BASH;
    if (call.name == "read") ptype = permission_type::FILE_READ;
    else if (call.name == "write") ptype = permission_type::FILE_WRITE;
    else if (call.name == "edit") ptype = permission_type::FILE_EDIT;
    else if (call.name == "update_plan") ptype = permission_type::GLOB;

    // Build permission request
    permission_request req;
    req.type = ptype;
    req.tool_name = call.name;
    req.details = call.arguments;

    // Check for external directory access on file operations
    if (call.name == "read" || call.name == "write" || call.name == "edit") {
        std::string file_path = args.value("file_path", "");
        if (!file_path.empty()) {
            // Make path absolute for comparison
            std::filesystem::path path(file_path);
            if (path.is_relative()) {
                path = std::filesystem::path(tool_ctx_.working_dir) / path;
            }
            if (permission_mgr_.is_external_path(path.string())) {
                permission_request ext_req;
                ext_req.type = permission_type::EXTERNAL_DIR;
                ext_req.tool_name = call.name;
                ext_req.details = "External file: " + path.string();
                ext_req.is_dangerous = true;
                ext_req.description = "Operation outside working directory";

                auto response = permission_mgr_.prompt_user(ext_req);
                if (response == permission_response::DENY_ONCE ||
                    response == permission_response::DENY_ALWAYS) {
                    return {false, "", "Blocked: File is outside working directory"};
                }
            }
        }
    }

    // Check for dangerous commands
    if (call.name == "bash") {
        std::string cmd = args.value("command", "");
        req.details = cmd;
        // Check for dangerous patterns
        for (const auto & pattern : {"rm -rf", "sudo ", "chmod 777"}) {
            if (cmd.find(pattern) != std::string::npos) {
                req.is_dangerous = true;
                break;
            }
        }
    }

    // Check doom loop
    std::hash<std::string> hasher;
    std::string args_hash = std::to_string(hasher(call.arguments));
    if (permission_mgr_.is_doom_loop(call.name, args_hash)) {
        req.description = "Detected repeated identical tool calls (doom loop)";
        auto response = permission_mgr_.prompt_user(req);
        if (response == permission_response::DENY_ONCE ||
            response == permission_response::DENY_ALWAYS) {
            return {false, "", "Blocked: Detected repeated identical tool calls"};
        }
    }

    // Check permission
    permission_state state = permission_mgr_.check_permission(req);
    if (state == permission_state::DENY || state == permission_state::DENY_SESSION) {
        return {false, "", "Permission denied for " + call.name};
    }

    if (state == permission_state::ASK) {
        auto response = permission_mgr_.prompt_user(req);
        if (response == permission_response::DENY_ONCE ||
            response == permission_response::DENY_ALWAYS) {
            return {false, "", "User denied permission for " + call.name};
        }
    }

    // Record this call
    permission_mgr_.record_tool_call(call.name, args_hash);

    // Display tool execution
    console::set_display(DISPLAY_TYPE_INFO);
    if (call.name == "bash") {
        std::string cmd = args.value("command", "");
        if (cmd.length() > 100) {
            cmd = cmd.substr(0, 100) + "...";
        }
        console::log("\n› %s %s", call.name.c_str(), cmd.c_str());
    } else if (call.name == "read" || call.name == "write" || call.name == "edit") {
        std::string path = args.value("path", args.value("file_path", ""));
        console::log("\n› %s %s", call.name.c_str(), path.c_str());
    } else {
        console::log("\n› %s ", call.name.c_str());
    }
    console::spinner::start();
    console::set_display(DISPLAY_TYPE_RESET);

    // Execute the tool with timing
    auto start_time = std::chrono::steady_clock::now();
    tool_result result = registry.execute(call.name, args, tool_ctx_);
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    console::spinner::stop();
    console::log("\n");

    // Display result summary
    if (result.success) {
        // Truncate long output for display
        std::string display_output = result.output;
        if (display_output.length() > 500) {
            display_output = display_output.substr(0, 500) + "\n... (truncated)";
        }
        console::log("%s\n", display_output.c_str());
        if (!result.image_bytes.empty()) {
            render_image_to_terminal(result.image_bytes.data(), result.image_bytes.size(),
                                     result.image_mime);
        }
    } else {
        // Show output if available (e.g., bash stderr), plus error if set
        if (!result.output.empty()) {
            std::string display_output = result.output;
            if (display_output.length() > 500) {
                display_output = display_output.substr(0, 500) + "\n... (truncated)";
            }
            console::error("%s\n", display_output.c_str());
        }
        if (!result.error.empty()) {
            console::error("Error: %s\n", result.error.c_str());
        }
        if (result.output.empty() && result.error.empty()) {
            console::error("Error: Tool failed with no output\n");
        }
    }

    // Display elapsed time
    console::set_display(DISPLAY_TYPE_INFO);
    if (elapsed_ms < 1000) {
        console::log("└─ %lldms\n", (long long)elapsed_ms);
    } else {
        console::log("└─ %.1fs\n", elapsed_ms / 1000.0);
    }
    console::set_display(DISPLAY_TYPE_RESET);

    return result;
}

// Async version of execute_tool_call for API use
// Uses async permission manager and emits events instead of blocking on console
tool_result agent_loop::execute_tool_call_async(
    const common_chat_tool_call & call,
    agent_event_callback on_event,
    permission_manager_async & async_perms,
    std::function<bool()> should_stop) {

    auto & registry = tool_registry::instance();

    // Check if tool exists
    const tool_def * tool = registry.get_tool(call.name);
    if (!tool) {
        return {false, "", "Unknown tool: " + call.name};
    }

    // Parse arguments
    json args;
    try {
        args = json::parse(call.arguments);
    } catch (const json::parse_error & e) {
        return {false, "", std::string("Invalid JSON arguments: ") + e.what()};
    }

    // Determine permission type
    permission_type ptype = permission_type::BASH;
    if (call.name == "read") {
        ptype = permission_type::FILE_READ;
    } else if (call.name == "write") {
        ptype = permission_type::FILE_WRITE;
    } else if (call.name == "edit") {
        ptype = permission_type::FILE_EDIT;
    } else if (call.name == "update_plan") {
        ptype = permission_type::GLOB;
    }

    // Build permission request
    permission_request req;
    req.type = ptype;
    req.tool_name = call.name;
    req.details = call.arguments;

    // Check for external directory access on file operations
    if (call.name == "read" || call.name == "write" || call.name == "edit") {
        std::string file_path = args.value("file_path", "");
        if (!file_path.empty()) {
            std::filesystem::path path(file_path);
            if (path.is_relative()) {
                path = std::filesystem::path(tool_ctx_.working_dir) / path;
            }
            if (async_perms.is_external_path(path.string())) {
                permission_request ext_req;
                ext_req.type = permission_type::EXTERNAL_DIR;
                ext_req.tool_name = call.name;
                ext_req.details = "External file: " + path.string();
                ext_req.is_dangerous = true;
                ext_req.description = "Operation outside working directory";

                // Request permission asynchronously
                std::string req_id = async_perms.request_permission(ext_req);
                on_event(agent_event::permission_required(req_id, call.name, ext_req.details, true));

                // Wait for response (cancellable via should_stop)
                auto response = async_perms.wait_for_response_or_stop(req_id, 300000, should_stop);
                if (should_stop()) {
                    on_event(agent_event::permission_resolved(req_id, false));
                    return {false, "", "Operation cancelled"};
                }
                if (!response || !response->allowed) {
                    on_event(agent_event::permission_resolved(req_id, false));
                    return {false, "", "Blocked: File is outside working directory"};
                }
                on_event(agent_event::permission_resolved(req_id, true));
            }
        }
    }

    // Check for dangerous commands
    if (call.name == "bash") {
        std::string cmd = args.value("command", "");
        req.details = cmd;
        for (const auto & pattern : {"rm -rf", "sudo ", "chmod 777"}) {
            if (cmd.find(pattern) != std::string::npos) {
                req.is_dangerous = true;
                break;
            }
        }
    }

    // Check doom loop
    std::hash<std::string> hasher;
    std::string args_hash = std::to_string(hasher(call.arguments));
    if (async_perms.is_doom_loop(call.name, args_hash)) {
        req.description = "Detected repeated identical tool calls (doom loop)";

        std::string req_id = async_perms.request_permission(req);
        on_event(agent_event::permission_required(req_id, call.name, req.details, true));

        auto response = async_perms.wait_for_response_or_stop(req_id, 300000, should_stop);
        if (should_stop()) {
            on_event(agent_event::permission_resolved(req_id, false));
            return {false, "", "Operation cancelled"};
        }
        if (!response || !response->allowed) {
            on_event(agent_event::permission_resolved(req_id, false));
            return {false, "", "Blocked: Detected repeated identical tool calls"};
        }
        on_event(agent_event::permission_resolved(req_id, true));
    }

    // Check permission
    permission_state state = async_perms.check_permission(req);
    if (state == permission_state::DENY || state == permission_state::DENY_SESSION) {
        return {false, "", "Permission denied for " + call.name};
    }

    if (state == permission_state::ASK) {
        // Request permission asynchronously
        std::string req_id = async_perms.request_permission(req);
        on_event(agent_event::permission_required(req_id, call.name, req.details, req.is_dangerous));

        // Wait for response (cancellable via should_stop)
        auto response = async_perms.wait_for_response_or_stop(req_id, 300000, should_stop);

        if (should_stop()) {
            on_event(agent_event::permission_resolved(req_id, false));
            return {false, "", "Operation cancelled"};
        }

        if (!response) {
            on_event(agent_event::permission_resolved(req_id, false));
            return {false, "", "Permission request timed out"};
        }

        on_event(agent_event::permission_resolved(req_id, response->allowed));

        if (!response->allowed) {
            return {false, "", "User denied permission for " + call.name};
        }
    }

    // Record this call
    async_perms.record_tool_call(call.name, args_hash);

    // Execute the tool
    return registry.execute(call.name, args, tool_ctx_);
}
