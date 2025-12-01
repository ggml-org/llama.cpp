#pragma once

#include "router-config.h"
#include "router-process.h"

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class RouterApp {
public:
    explicit RouterApp(RouterConfig cfg);
    ~RouterApp();

    bool ensure_running(const std::string & model_name, std::string & error);
    std::string upstream_for(const std::string & model_name);
    std::string get_last_spawned_model();
    SpawnConfig get_spawn_config(const std::string & model_name);
    void stop_all();
    void update_config(RouterConfig cfg);

    void set_notification_sink(NotificationSink sink);
    void clear_notification_sink();

    const RouterConfig & get_config() const { return config; }

private:
    RouterConfig config;
    std::atomic<int> next_port;
    std::mutex mutex;
    std::optional<NotificationSink> notification_sink;
    std::mutex notification_mutex;
    std::unordered_map<std::string, ModelConfig> model_lookup;
    std::unordered_map<std::string, ProcessHandle> processes;
    std::unordered_map<std::string, int> model_ports;
    std::string last_spawned_model;

    SpawnConfig resolve_spawn_config(const ModelConfig & cfg) const;
    void notify_progress(const std::string & message);
};
