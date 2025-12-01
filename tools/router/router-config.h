#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

struct ProgressNotification {
    std::string message;
};

using NotificationSink = std::function<void(const ProgressNotification &)>;

struct SpawnConfig {
    std::vector<std::string> command;
    std::vector<std::string> proxy_endpoints;
    std::string              health_endpoint;
};

inline bool is_spawn_empty(const SpawnConfig & spawn) {
    return spawn.command.empty() && spawn.proxy_endpoints.empty() && spawn.health_endpoint.empty();
}

struct ModelConfig {
    std::string              name;
    std::string              path;
    std::string              state;
    std::string              group;
    SpawnConfig              spawn;
};

struct RouterOptions {
    std::string host;
    int         port = 0;
    int         base_port = 0;
    int         connection_timeout_s = 5;
    int         read_timeout_s       = 600;
    std::string admin_token;
    bool        notify_model_swap = false;
};

struct RouterConfig {
    std::string              version;
    SpawnConfig              default_spawn;
    RouterOptions            router;
    std::vector<ModelConfig> models;
};

struct RescanResult {
    RouterConfig config;
    size_t       added   = 0;
    size_t       removed = 0;
};

std::string get_default_config_path();
std::string expand_user_path(const std::string & path);
const SpawnConfig & get_default_spawn();
const RouterOptions &             get_default_router_options();

RouterConfig load_config(const std::string & path);
RouterConfig generate_default_config(const std::string & path);
void         write_config_file(const RouterConfig & cfg, const std::string & path);
RescanResult rescan_auto_models(const RouterConfig & existing);

std::string get_model_group(const ModelConfig & cfg);
