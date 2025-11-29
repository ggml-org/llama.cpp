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
    std::string host      = "0.0.0.0";
    int         port      = 8082;
    int         base_port = 28081;
    std::string log_level = "info";
};

struct RouterConfig {
    std::string              version;
    std::vector<std::string> default_spawn;
    RouterOptions            router;
    std::vector<ModelConfig> models;
};

std::string get_default_config_path();
std::string expand_user_path(const std::string & path);

RouterConfig load_config(const std::string & path);
RouterConfig generate_default_config(const std::string & path);
void         write_config_file(const RouterConfig & cfg, const std::string & path);
