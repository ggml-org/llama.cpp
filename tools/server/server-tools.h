#pragma once

#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"

#include <atomic>
#include <functional>

struct server_tool {
    std::string name;
    std::string display_name;
    bool permission_write = false;
    bool support_stream = false; // if true, output can be streamed

    virtual ~server_tool() = default;
    virtual json get_definition() const = 0;

    struct stream {
        server_response & qr;
        int id;
        std::function<bool()> alive;
        void push(const std::string & chunk);
    };
    virtual json invoke(json params, stream * st = nullptr) const = 0;

    json to_json() const;
};

struct server_tools {
    std::vector<std::unique_ptr<server_tool>> tools;

    // for streaming
    server_response queue_res;
    std::atomic<int> res_id{0};

    void setup(const std::vector<std::string> & enabled_tools);

    server_http_context::handler_t handle_get;
    server_http_context::handler_t handle_post;
};
