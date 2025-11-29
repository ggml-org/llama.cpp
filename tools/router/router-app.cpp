#include "router-app.h"

#include "log.h"
#include "router-config.h"
#include "router-process.h"

#include <filesystem>
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
        wait_for_process_exit(it_proc->second, 2000);
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

    std::vector<std::string> command = cfg.spawn.empty() ? config.default_spawn : cfg.spawn;
    command.push_back("--model");
    command.push_back(expand_user_path(cfg.path));
    command.push_back("--port");
    command.push_back(std::to_string(port));
    command.push_back("--host");
    command.push_back("127.0.0.1");

    LOG_INF("Starting %s on port %d\n", model_name.c_str(), port);

    std::string log_path;
    if (!config.router.log_dir.empty()) {
        std::filesystem::path log_dir(expand_user_path(config.router.log_dir));
        std::error_code      ec;
        std::filesystem::create_directories(log_dir, ec);
        if (ec) {
            LOG_WRN("Could not create log directory %s: ec=%d\n", log_dir.string().c_str(), ec.value());
        } else {
            log_path = (log_dir / (model_name + ".log")).string();
            LOG_INF("Capturing stdout/stderr for %s in %s\n", model_name.c_str(), log_path.c_str());
        }
    }

    ProcessHandle handle = spawn_process(command, log_path);
    if (!process_running(handle)) {
        error = "failed to start process";
        LOG_ERR("Failed to start %s on port %d: %s\n", model_name.c_str(), port, error.c_str());
        terminate_process(handle);
        return false;
    }

    processes.emplace(model_name, handle);
    last_spawned_model = model_name;
    LOG_INF("Spawned %s (group '%s') with %zu args\n", model_name.c_str(), target_group.c_str(), command.size());
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
