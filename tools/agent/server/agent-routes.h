#pragma once

#include "agent-session.h"
#include "server-http.h"

#include <functional>

// Agent API routes for HTTP server
// These handlers implement the /v1/agent/* endpoints
struct agent_routes {
    using handler_t = server_http_context::handler_t;

    // Health check
    handler_t get_health;

    // Session management
    handler_t post_session;         // POST /v1/agent/session - Create session
    handler_t get_session;          // GET /v1/agent/session/:id - Get session info
    handler_t delete_session;       // DELETE /v1/agent/session/:id - Delete session
    handler_t get_sessions;         // GET /v1/agent/sessions - List all sessions

    // Chat/messaging
    handler_t post_chat;            // POST /v1/agent/session/:id/chat - Send message (streaming)
    handler_t get_messages;         // GET /v1/agent/session/:id/messages - Get conversation history

    // Permissions
    handler_t get_permissions;      // GET /v1/agent/session/:id/permissions - Get pending permissions
    handler_t post_permission;      // POST /v1/agent/permission/:id - Respond to permission

    // Tools
    handler_t get_tools;            // GET /v1/agent/tools - List available tools

    // Statistics
    handler_t get_stats;            // GET /v1/agent/session/:id/stats - Get session stats

    // Constructor: set up all handlers
    agent_routes(agent_session_manager & session_mgr);

private:
    agent_session_manager & session_mgr_;

    // Helper to create error response
    static server_http_res_ptr make_error(int status, const std::string & message);

    // Helper to create JSON response
    static server_http_res_ptr make_json(const json & data, int status = 200);
};
