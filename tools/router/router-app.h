#pragma once

#include "router-config.h"
#include "router-process.h"

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

class RouterApp {
public:
    explicit RouterApp(RouterConfig cfg);
    ~RouterApp();

    void start_auto_models();
    bool ensure_running(const std::string & model_name, std::string & error);
    std::string upstream_for(const std::string & model_name);
    std::string get_last_spawned_model();
    SpawnConfig get_spawn_config(const std::string & model_name);
    void stop_all();

    const RouterConfig & get_config() const { return config; }

private:
    RouterConfig config;
    std::atomic<int> next_port;
    std::mutex mutex;
    std::unordered_map<std::string, ModelConfig> model_lookup;
    std::unordered_map<std::string, ProcessHandle> processes;
    std::unordered_map<std::string, int> model_ports;
    std::string last_spawned_model;

    SpawnConfig resolve_spawn_config(const ModelConfig & cfg) const;
};
