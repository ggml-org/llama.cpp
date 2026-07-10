#pragma once

#include "server-common.h"
#include "server-http.h"

struct server_tool {
    std::string name;
    std::string display_name;
    bool permission_write = false;

    virtual ~server_tool() = default;
    virtual json get_definition() const = 0;
    virtual json invoke(json params) const = 0;

    json to_json() const;
};

struct server_tools {
    std::vector<std::unique_ptr<server_tool>> tools;

    void setup(const std::vector<std::string> & enabled_tools);
    json invoke(const std::string & name, const json & params);

    server_http_context::handler_t handle_get;
    server_http_context::handler_t handle_post;
};
