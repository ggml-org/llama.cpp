#pragma once

#include "server-common.h"

#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct server_local_model {
    std::string name;
    std::string path;
    std::string path_mmproj;
};

class server_config_manager {
public:
    explicit server_config_manager(const std::string & models_dir);

    bool enabled() const;

    void sync(const std::vector<server_local_model> & models, const std::vector<std::string> & base_args);

    std::map<std::string, std::string> env_for(const std::string & name);

private:
    void ensure_loaded();
    void write_locked();

private:
    std::string path;
    std::string models_dir;
    std::map<std::string, std::map<std::string, std::string>> data;
    std::mutex mutex;
};

bool is_router_control_arg(const std::string & arg);

