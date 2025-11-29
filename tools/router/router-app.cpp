#include "router-app.h"

#include "log.h"
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

void RouterApp::start_auto_models() {
    for (const auto & model : config.models) {
        if (model.state == "auto") {
            std::string err;
            if (!ensure_running(model.name, err)) {
                LOG_WRN("auto-start for %s failed: %s\n", model.name.c_str(), err.c_str());
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
        model_ports.erase(it_proc->first);
        it_proc = processes.erase(it_proc);
    }

    auto it = processes.find(model_name);
    if (it != processes.end() && process_running(it->second)) {
        return true;
    }

    if (it != processes.end()) {
        close_process(it->second);
        processes.erase(it);
        model_ports.erase(model_name);
    }

    int port = next_port.fetch_add(1);
    model_ports[model_name] = port;

    std::vector<std::string> command = cfg.spawn.empty() ? config.default_spawn : cfg.spawn;
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
        terminate_process(handle);
        return false;
    }

    processes.emplace(model_name, handle);
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

void RouterApp::stop_all() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto & kv : processes) {
        terminate_process(kv.second);
    }
    processes.clear();
}
