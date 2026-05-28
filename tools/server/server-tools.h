#pragma once

#include "server-common.h"
#include "server-http.h"

#include <mutex>
#include <unordered_map>

struct server_tool_artifact_item {
    std::string name;
    std::string mime_type;
    std::string base64_data;
    std::string text_content;
};

struct server_tool {
    std::string name;
    std::string display_name;
    bool permission_write = false;

    virtual ~server_tool() = default;
    virtual json get_definition() = 0;
    virtual json invoke(const json & params, const json & context, class server_tools * state) = 0;

    json to_json();
};

struct server_tools {
    std::vector<std::unique_ptr<server_tool>> tools;
    std::mutex mutex_state;
    json pending_questions = json::object();
    std::unordered_map<std::string, std::unordered_map<std::string, server_tool_artifact_item>> artifacts;

    void setup(const std::vector<std::string> & enabled_tools);
    json invoke(const std::string & name, const json & params, const json & context = json::object());
    json list_pending(const std::string & conversation_id);
    json reply(const json & body);

    server_http_context::handler_t handle_get;
    server_http_context::handler_t handle_post;
    server_http_context::handler_t handle_get_pending;
    server_http_context::handler_t handle_post_reply;
};
