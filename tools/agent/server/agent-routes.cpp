#include "agent-routes.h"
#include "../tool-registry.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

// Helper to create error response
server_http_res_ptr agent_routes::make_error(int status, const std::string & message) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->data = json{{"error", message}}.dump();
    return res;
}

// Helper to create JSON response
server_http_res_ptr agent_routes::make_json(const json & data, int status) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->data = data.dump();
    return res;
}

// SSE streaming response implementation
struct sse_stream_res : server_http_res {
    std::queue<std::string> chunks;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> done{false};

    sse_stream_res() {
        content_type = "text/event-stream";
        headers["Cache-Control"] = "no-cache";
        headers["Connection"] = "keep-alive";

        next = [this](std::string & output) -> bool {
            std::unique_lock<std::mutex> lock(mutex);

            // Wait for data or done signal
            cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !chunks.empty() || done.load();
            });

            if (!chunks.empty()) {
                output = chunks.front();
                chunks.pop();
                return true;
            }

            if (done.load()) {
                return false;
            }

            // Keep alive with empty chunk
            output = "";
            return true;
        };
    }

    void send(const std::string & event_type, const json & data) {
        std::string chunk = "event: " + event_type + "\n";
        chunk += "data: " + data.dump() + "\n\n";

        {
            std::lock_guard<std::mutex> lock(mutex);
            chunks.push(chunk);
        }
        cv.notify_one();
    }

    void finish() {
        done.store(true);
        cv.notify_one();
    }
};

// Wrapper that holds shared_ptr to sse_stream_res to extend its lifetime
// This ensures the SSE response object lives until both:
// 1. The HTTP framework is done with it
// 2. The worker thread callback is done with it
struct sse_shared_wrapper : server_http_res {
    std::shared_ptr<sse_stream_res> sse;

    explicit sse_shared_wrapper(std::shared_ptr<sse_stream_res> s) : sse(std::move(s)) {
        content_type = sse->content_type;
        headers = sse->headers;
        next = [this](std::string & output) -> bool {
            return sse->next(output);
        };
    }
};

agent_routes::agent_routes(agent_session_manager & session_mgr)
    : session_mgr_(session_mgr) {

    // GET /health
    get_health = [](const server_http_req &) -> server_http_res_ptr {
        return make_json({{"status", "ok"}});
    };

    // POST /v1/agent/session - Create a new session
    post_session = [this](const server_http_req & req) -> server_http_res_ptr {
        agent_session_config config;

        // Parse optional config from body
        if (!req.body.empty()) {
            try {
                json body = json::parse(req.body);

                if (body.contains("yolo")) {
                    config.yolo_mode = body["yolo"].get<bool>();
                }
                if (body.contains("max_iterations")) {
                    config.max_iterations = body["max_iterations"].get<int>();
                }
                if (body.contains("working_dir")) {
                    config.working_dir = body["working_dir"].get<std::string>();
                }
                // Skills configuration
                if (body.contains("enable_skills")) {
                    config.enable_skills = body["enable_skills"].get<bool>();
                }
                if (body.contains("skills_paths") && body["skills_paths"].is_array()) {
                    for (const auto & path : body["skills_paths"]) {
                        config.extra_skills_paths.push_back(path.get<std::string>());
                    }
                }
                // AGENTS.md configuration
                if (body.contains("enable_agents_md")) {
                    config.enable_agents_md = body["enable_agents_md"].get<bool>();
                }
            } catch (const json::exception & e) {
                return make_error(400, std::string("Invalid JSON: ") + e.what());
            }
        }

        std::string session_id = session_mgr_.create_session(config);
        return make_json({{"session_id", session_id}}, 201);
    };

    // GET /v1/agent/session/:id - Get session info
    get_session = [this](const server_http_req & req) -> server_http_res_ptr {
        std::string session_id = req.get_param("id");
        if (session_id.empty()) {
            return make_error(400, "Missing session ID");
        }

        auto session = session_mgr_.get_session(session_id);
        if (!session) {
            return make_error(404, "Session not found");
        }

        auto info = session->info();
        json response = {
            {"session_id", info.id},
            {"state", static_cast<int>(info.state)},
            {"message_count", info.message_count},
            {"stats", {
                {"input_tokens", info.stats.total_input},
                {"output_tokens", info.stats.total_output},
                {"cached_tokens", info.stats.total_cached}
            }}
        };
        return make_json(response);
    };

    // DELETE /v1/agent/session/:id - Delete session
    delete_session = [this](const server_http_req & req) -> server_http_res_ptr {
        std::string session_id = req.get_param("id");
        if (session_id.empty()) {
            return make_error(400, "Missing session ID");
        }

        if (session_mgr_.delete_session(session_id)) {
            return make_json({{"deleted", true}});
        }
        return make_error(404, "Session not found");
    };

    // GET /v1/agent/sessions - List all sessions
    get_sessions = [this](const server_http_req &) -> server_http_res_ptr {
        auto sessions = session_mgr_.list_sessions();
        json response = json::array();
        for (const auto & info : sessions) {
            response.push_back({
                {"session_id", info.id},
                {"state", static_cast<int>(info.state)},
                {"message_count", info.message_count}
            });
        }
        return make_json({{"sessions", response}});
    };

    // POST /v1/agent/session/:id/chat - Send message with streaming response
    post_chat = [this](const server_http_req & req) -> server_http_res_ptr {
        std::string session_id = req.get_param("id");
        if (session_id.empty()) {
            return make_error(400, "Missing session ID");
        }

        auto session = session_mgr_.get_session(session_id);
        if (!session) {
            return make_error(404, "Session not found");
        }

        // Parse message content from body (string or array of content blocks)
        json content;
        try {
            json body = json::parse(req.body);
            if (!body.contains("content")) {
                return make_error(400, "Missing 'content' field");
            }
            auto & c = body["content"];
            if (c.is_string()) {
                content = c;
            } else if (c.is_array()) {
                // Validate content block structure
                for (const auto & block : c) {
                    if (!block.is_object() || !block.contains("type")) {
                        return make_error(400, "Each content block must have a 'type' field");
                    }
                    const auto & type = block["type"];
                    if (!type.is_string() || (type != "text" && type != "image_url")) {
                        return make_error(400, "Content block type must be 'text' or 'image_url'");
                    }
                }
                content = c;
            } else {
                return make_error(400, "'content' must be a string or array");
            }
        } catch (const json::exception & e) {
            return make_error(400, std::string("Invalid JSON: ") + e.what());
        }

        // Create SSE streaming response with shared ownership
        // The shared_ptr ensures the response lives until both:
        // 1. The HTTP framework is done streaming
        // 2. The worker thread callback is done
        auto sse_shared = std::make_shared<sse_stream_res>();

        // Start processing in background - capture shared_ptr to extend lifetime
        session->send_message(content, [sse_shared](const agent_event & event) {
            std::string event_type;
            switch (event.type) {
                case agent_event_type::TEXT_DELTA:
                    event_type = "text_delta";
                    break;
                case agent_event_type::REASONING_DELTA:
                    event_type = "reasoning_delta";
                    break;
                case agent_event_type::TOOL_START:
                    event_type = "tool_start";
                    break;
                case agent_event_type::TOOL_RESULT:
                    event_type = "tool_result";
                    break;
                case agent_event_type::PERMISSION_REQUIRED:
                    event_type = "permission_required";
                    break;
                case agent_event_type::PERMISSION_RESOLVED:
                    event_type = "permission_resolved";
                    break;
                case agent_event_type::ITERATION_START:
                    event_type = "iteration_start";
                    break;
                case agent_event_type::COMPACTION_COMPLETED:
                    event_type = "compaction_completed";
                    break;
                case agent_event_type::COMPLETED:
                    event_type = "completed";
                    sse_shared->send(event_type, event.data);
                    sse_shared->finish();
                    return;
                case agent_event_type::ERROR:
                    event_type = "error";
                    sse_shared->send(event_type, event.data);
                    sse_shared->finish();
                    return;
            }
            sse_shared->send(event_type, event.data);
        });

        // Return wrapper that holds shared_ptr reference
        return std::make_unique<sse_shared_wrapper>(sse_shared);
    };

    // GET /v1/agent/session/:id/messages - Get conversation history
    get_messages = [this](const server_http_req & req) -> server_http_res_ptr {
        std::string session_id = req.get_param("id");
        if (session_id.empty()) {
            return make_error(400, "Missing session ID");
        }

        auto session = session_mgr_.get_session(session_id);
        if (!session) {
            return make_error(404, "Session not found");
        }

        return make_json({{"messages", session->get_messages()}});
    };

    // GET /v1/agent/session/:id/permissions - Get pending permissions
    get_permissions = [this](const server_http_req & req) -> server_http_res_ptr {
        std::string session_id = req.get_param("id");
        if (session_id.empty()) {
            return make_error(400, "Missing session ID");
        }

        auto session = session_mgr_.get_session(session_id);
        if (!session) {
            return make_error(404, "Session not found");
        }

        auto pending = session->pending_permissions();
        json response = json::array();
        for (const auto & perm : pending) {
            response.push_back({
                {"request_id", perm.id},
                {"tool", perm.request.tool_name},
                {"details", perm.request.details},
                {"dangerous", perm.request.is_dangerous}
            });
        }
        return make_json({{"permissions", response}});
    };

    // POST /v1/agent/permission/:id - Respond to permission request
    post_permission = [this](const server_http_req & req) -> server_http_res_ptr {
        std::string request_id = req.get_param("id");
        if (request_id.empty()) {
            return make_error(400, "Missing request ID");
        }

        // Parse response from body
        bool allowed = false;
        permission_scope scope = permission_scope::ONCE;
        try {
            json body = json::parse(req.body);
            if (!body.contains("allow")) {
                return make_error(400, "Missing 'allow' field");
            }
            allowed = body["allow"].get<bool>();
            if (body.contains("scope")) {
                std::string scope_str = body["scope"].get<std::string>();
                if (scope_str == "session") {
                    scope = permission_scope::SESSION;
                }
            }
        } catch (const json::exception & e) {
            return make_error(400, std::string("Invalid JSON: ") + e.what());
        }

        // Find the session with this permission request
        for (const auto & info : session_mgr_.list_sessions()) {
            auto session = session_mgr_.get_session(info.id);
            if (session && session->respond_permission(request_id, allowed, scope)) {
                return make_json({{"success", true}});
            }
        }

        return make_error(404, "Permission request not found");
    };

    // GET /v1/agent/tools - List available tools
    get_tools = [](const server_http_req &) -> server_http_res_ptr {
        auto tools = tool_registry::instance().to_chat_tools();
        json response = json::array();
        for (const auto & tool : tools) {
            response.push_back({
                {"name", tool.name},
                {"description", tool.description},
                {"parameters", json::parse(tool.parameters)}
            });
        }
        return make_json({{"tools", response}});
    };

    // GET /v1/agent/session/:id/stats - Get session statistics
    get_stats = [this](const server_http_req & req) -> server_http_res_ptr {
        std::string session_id = req.get_param("id");
        if (session_id.empty()) {
            return make_error(400, "Missing session ID");
        }

        auto session = session_mgr_.get_session(session_id);
        if (!session) {
            return make_error(404, "Session not found");
        }

        auto stats = session->get_stats();
        return make_json({
            {"input_tokens", stats.total_input},
            {"output_tokens", stats.total_output},
            {"cached_tokens", stats.total_cached},
            {"prompt_ms", stats.total_prompt_ms},
            {"predicted_ms", stats.total_predicted_ms}
        });
    };
}

