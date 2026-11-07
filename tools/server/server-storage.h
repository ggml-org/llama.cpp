#pragma once

#include "server-common.h"
#include "server-http.h"

#include <filesystem>
#include <mutex>

// UI storage (--ui-storage): the Web UI's conversations and settings kept as JSON files on disk
// the server does not interpret the documents, it only tracks a revision per conversation so
// concurrent writers (tabs, browsers) get a 409 instead of silently overwriting each other
//
// layout:
//   <dir>/conversations/<id>.json  { "rev": n, "conv": {...}, "messages": [...] }
//   <dir>/index.json               { "<id>": { "rev": n, "conv": {...} } }, rebuilt when missing
//   <dir>/settings.json            { "<key>": "<value>" }
struct server_storage {
    void setup(const std::filesystem::path & dir);

    server_http_context::handler_t handle_list;          // GET    /storage/conversations
    server_http_context::handler_t handle_get;           // GET    /storage/conversations/:id
    server_http_context::handler_t handle_put;           // POST   /storage/conversations/:id
    server_http_context::handler_t handle_delete;        // DELETE /storage/conversations/:id
    server_http_context::handler_t handle_get_settings;  // GET    /storage/settings
    server_http_context::handler_t handle_put_settings;  // POST   /storage/settings

private:
    std::filesystem::path dir;
    std::mutex mtx;
    json index = json::object();

    std::filesystem::path conversation_path(const std::string & id) const;
    void load_index();
    void save_index() const;
};
