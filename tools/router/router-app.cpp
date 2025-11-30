#include "router-app.h"

#include "log.h"
#include "router-constants.h"
#include "router-config.h"
#include "router-process.h"

#include <sstream>
#include <vector>

RouterApp::RouterApp(RouterConfig cfg)
    : config(std::move(cfg)), next_port(config.router.base_port) {
    for (const auto & model : config.models) {
        model_lookup.emplace(model.name, model);
    }
}

RouterApp::~RouterApp() { stop_all(); }

SpawnConfig RouterApp::resolve_spawn_config(const ModelConfig & cfg) const {
    return is_spawn_empty(cfg.spawn) ? config.default_spawn : cfg.spawn;
}

SpawnConfig RouterApp::get_spawn_config(const std::string & model_name) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = model_lookup.find(model_name);
    if (it == model_lookup.end()) {
        return config.default_spawn;
    }
    return resolve_spawn_config(it->second);
}

void RouterApp::start_auto_models() {
    for (const auto & model : config.models) {
        if (model.state == "auto") {
            std::string err;
            if (!ensure_running(model.name, err)) {
                LOG_WRN("auto-start for %s failed: %s\n", model.name.c_str(), err.c_str());
            } else {
                LOG_INF("auto-started %s\n", model.name.c_str());
            }
        }
    }
}

bool RouterApp::ensure_running(const std::string & model_name, std::string & error) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it_cfg = model_lookup.find(model_name);
    if (it_cfg == model_lookup.end()) {
        error = "model not found";
        return false;
    }

    const ModelConfig & cfg          = it_cfg->second;
    const std::string   target_group = get_model_group(cfg);

    for (auto it_proc = processes.begin(); it_proc != processes.end();) {
        const auto it_model = model_lookup.find(it_proc->first);
        const bool unknown  = it_model == model_lookup.end();
        const std::string running_group = unknown ? std::string() : get_model_group(it_model->second);

        if (!unknown && running_group == target_group) {
            ++it_proc;
            continue;
        }

        LOG_INF("Stopping %s (group '%s') to spawn %s (group '%s')\n",
                it_proc->first.c_str(),
                running_group.c_str(),
                model_name.c_str(),
                target_group.c_str());

        terminate_process(it_proc->second);
        wait_for_process_exit(it_proc->second, ROUTER_PROCESS_SHUTDOWN_TIMEOUT_MS);
        model_ports.erase(it_proc->first);
        it_proc = processes.erase(it_proc);
    }

    auto it = processes.find(model_name);
    if (it != processes.end() && process_running(it->second)) {
        LOG_DBG("Model %s already running on port %d\n", model_name.c_str(), model_ports[model_name]);
        return true;
    }

    if (it != processes.end()) {
        close_process(it->second);
        processes.erase(it);
        model_ports.erase(model_name);
    }

    int port = next_port.fetch_add(1);
    model_ports[model_name] = port;

    const SpawnConfig spawn_cfg = resolve_spawn_config(cfg);

    std::vector<std::string> command = spawn_cfg.command;
    command.push_back("--model");
    command.push_back(expand_user_path(cfg.path));
    command.push_back("--port");
    command.push_back(std::to_string(port));
    command.push_back("--host");
    command.push_back("127.0.0.1");

    LOG_INF("Starting %s on port %d\n", model_name.c_str(), port);

    ProcessHandle handle = spawn_process(command);
    if (!process_running(handle)) {
        error = "failed to start process";
        LOG_ERR("Failed to start %s on port %d: %s\n", model_name.c_str(), port, error.c_str());
        terminate_process(handle);
        return false;
    }

    auto [proc_it, _] = processes.emplace(model_name, std::move(handle));
    last_spawned_model = model_name;
    LOG_INF("Spawned %s (group '%s') with %zu args\n", model_name.c_str(), target_group.c_str(), command.size());

    const std::string health_endpoint = spawn_cfg.health_endpoint.empty() ? "/health" : spawn_cfg.health_endpoint;
    if (!wait_for_backend_ready(port, health_endpoint, ROUTER_BACKEND_READY_TIMEOUT_MS, &proc_it->second)) {
        error = "backend not ready";
        LOG_ERR("Backend for %s did not become ready on port %d within %d ms\n",
                model_name.c_str(),
                port,
                ROUTER_BACKEND_READY_TIMEOUT_MS);
        terminate_process(proc_it->second);
        processes.erase(proc_it);
        model_ports.erase(model_name);
        return false;
    }

    LOG_INF("Backend ready on port %d\n", port);
    return true;
}

std::string RouterApp::upstream_for(const std::string & model_name) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = model_ports.find(model_name);
    if (it == model_ports.end()) {
        return {};
    }
    std::ostringstream os;
    os << "http://127.0.0.1:" << it->second;
    return os.str();
}

std::string RouterApp::get_last_spawned_model() {
    std::lock_guard<std::mutex> lock(mutex);
    return last_spawned_model;
}

void RouterApp::stop_all() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto & kv : processes) {
        LOG_INF("Stopping managed model %s\n", kv.first.c_str());
        terminate_process(kv.second);
    }
    processes.clear();
}
