#pragma once

#include <string>
#include <vector>

struct ModelConfig {
    std::string              name;
    std::string              path;
    std::string              state;
    std::vector<std::string> spawn;
};

struct RouterOptions {
    std::string host;
    int         port = 0;
    int         base_port = 0;
};

struct RouterConfig {
    std::string              version;
    std::vector<std::string> default_spawn;
    RouterOptions            router;
    std::vector<ModelConfig> models;
};

std::string get_default_config_path();
std::string expand_user_path(const std::string & path);
const std::vector<std::string> & get_default_spawn();
const RouterOptions &             get_default_router_options();

RouterConfig load_config(const std::string & path);
RouterConfig generate_default_config(const std::string & path);
void         write_config_file(const RouterConfig & cfg, const std::string & path);
