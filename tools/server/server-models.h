#pragma once

#include "common.h"
#include "preset.h"
#include "server-common.h"
#include "server-http.h"

#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <set>

/**
 * state diagram:
 *
 * UNLOADED ──► LOADING ──► LOADED
 *  ▲            │            │
 *  └───failed───┘            │
 *  └────────unloaded─────────┘
 */
enum server_model_status {
    // TODO: also add downloading state when the logic is added
    SERVER_MODEL_STATUS_UNLOADED,
    SERVER_MODEL_STATUS_LOADING,
    SERVER_MODEL_STATUS_LOADED
};

static server_model_status server_model_status_from_string(const std::string & status_str) {
    if (status_str == "unloaded") {
        return SERVER_MODEL_STATUS_UNLOADED;
    }
    if (status_str == "loading") {
        return SERVER_MODEL_STATUS_LOADING;
    }
    if (status_str == "loaded") {
        return SERVER_MODEL_STATUS_LOADED;
    }
    throw std::runtime_error("invalid server model status");
}

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_UNLOADED: return "unloaded";
        case SERVER_MODEL_STATUS_LOADING:  return "loading";
        case SERVER_MODEL_STATUS_LOADED:   return "loaded";
        default:                           return "unknown";
    }
}

struct server_model_meta {
    common_preset preset;
    std::string name;
    std::set<std::string> aliases; // additional names that resolve to this model
    std::set<std::string> tags;    // informational tags, not used for routing
    int port = 0;
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED;
    int64_t last_used = 0; // for LRU unloading
    int64_t idle_start = 0; // for idle timeout: when the model became idle (timestamp from ggml_time_ms())
    std::vector<std::string> args; // args passed to the model instance, will be populated by render_args()
    int exit_code = 0; // exit code of the model instance process (only valid if status == FAILED)
    int stop_timeout = 0; // seconds to wait before force-killing the model instance during shutdown
    int32_t priority = 0; // priority for preemptive scheduling (0 = normal)

    bool is_ready() const {
        return status == SERVER_MODEL_STATUS_LOADED;
    }

    bool is_running() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING;
    }

    bool is_failed() const {
        return status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0;
    }

    void update_args(common_preset_context & ctx_presets, std::string bin_path);
};

struct subprocess_s;

struct server_models {
private:
    struct instance_t {
        std::shared_ptr<subprocess_s> subproc; // shared between main thread and monitoring thread
        std::thread th;
        server_model_meta meta;
        FILE * stdin_file = nullptr;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, instance_t> mapping;

    // for stopping models
    std::condition_variable cv_stop;
    std::set<std::string> stopping_models;

    // for priority scheduling: track pending request counts per model
    std::map<std::string, int> pending_counts;

    common_preset_context ctx_preset;

    common_params base_params;
    std::string bin_path;
    std::vector<std::string> base_env;
    common_preset base_preset; // base preset from llama-server CLI args

    // idle timeout in seconds (0 = disabled)
    int idle_timeout = 0;
    std::thread idle_thread;
    std::atomic<bool> idle_stopped{false};
    void idle_loop();
    void idle_thread_start();
    void idle_thread_stop();

    void update_meta(const std::string & name, const server_model_meta & meta);

    // unload least recently used models if the limit is reached
    void unload_lru();

    // not thread-safe, caller must hold mutex
    void add_model(server_model_meta && meta);

public:
    server_models(const common_params & params, int argc, char ** argv);
    ~server_models();

    void load_models();

    // check if a model instance exists (thread-safe)
    bool has_model(const std::string & name);

    // return a copy of model metadata (thread-safe)
    std::optional<server_model_meta> get_meta(const std::string & name);

    // return a copy of all model metadata (thread-safe)
    std::vector<server_model_meta> get_all_meta();

    // load and unload model instances
    // these functions are thread-safe
    void load(const std::string & name);
    void unload(const std::string & name);
    void unload_all();

    // update the status of a model instance (thread-safe)
    void update_status(const std::string & name, server_model_status status, int exit_code);

    // wait until the model instance is fully loaded (thread-safe)
    // return when the model no longer in "loading" state
    void wait_until_loading_finished(const std::string & name);

    // ensure the model is in ready state (thread-safe)
    // return false if model is ready
    // otherwise, load the model and blocking wait until it's ready, then return true (meta may need to be refreshed)
    bool ensure_model_ready(const std::string & name);

    // proxy an HTTP request to the model instance
    server_http_res_ptr proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used);

    // priority-aware proxy request with preemption and drain window
    server_http_res_ptr proxy_request_priority(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used);

    // check if a model has pending requests beyond the one being served
    bool has_pending_requests(const std::string & name);

    // get the running model's name (thread-safe, returns empty if none)
    std::string get_running_model_name();

    // return true if the current process is a child server instance
    static bool is_child_server();

    // notify the router server that a model instance is ready
    // return the monitoring thread (to be joined by the caller)
    static std::thread setup_child_server(const std::function<void(int)> & shutdown_handler);

    };

struct server_models_routes {
    common_params params;
    json webui_settings = json::object();
    server_models models;
    server_models_routes(const common_params & params, int argc, char ** argv)
            : params(params), models(params, argc, argv) {
        if (!this->params.webui_config_json.empty()) {
            try {
                webui_settings = json::parse(this->params.webui_config_json);
            } catch (const std::exception & e) {
                LOG_ERR("%s: failed to parse webui config: %s\n", __func__, e.what());
                throw;
            }
        }
        init_routes();
    }

    void init_routes();
    // handlers using lambda function, so that they can capture `this` without `std::bind`
    server_http_context::handler_t get_router_props;
    server_http_context::handler_t proxy_get;
    server_http_context::handler_t proxy_post;
    server_http_context::handler_t proxy_post_priority;
    server_http_context::handler_t get_router_models;
    server_http_context::handler_t post_router_models_load;
    server_http_context::handler_t post_router_models_unload;
};

/**
 * A simple HTTP proxy that forwards requests to another server
 * and relays the responses back.
 */
struct server_http_proxy : server_http_res {
    std::function<void()> cleanup = nullptr;
public:
    server_http_proxy(const std::string & method,
                      const std::string & scheme,
                      const std::string & host,
                      int port,
                      const std::string & path,
                      const std::map<std::string, std::string> & headers,
                      const std::string & body,
                      const std::map<std::string, uploaded_file> & files,
                      const std::function<bool()> should_stop,
                      int32_t timeout_read,
                      int32_t timeout_write,
                      std::function<void(const std::string &)> on_chunk = nullptr
                      );
    ~server_http_proxy() {
        if (cleanup) {
            cleanup();
        }
    }
private:
    std::thread thread;
    struct msg_t {
        std::map<std::string, std::string> headers;
        int status = 0;
        std::string data;
        std::string content_type;
    };
};
