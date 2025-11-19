#pragma once

#include "common.h"
#include "server-http.h"

#include <queue>
#include <mutex>
#include <condition_variable>

enum server_model_status {
    SERVER_MODEL_STATUS_UNLOADED,
    SERVER_MODEL_STATUS_LOADING,
    SERVER_MODEL_STATUS_LOADED,
    SERVER_MODEL_STATUS_FAILED
};

static server_model_status server_model_status_from_string(const std::string & status_str) {
    if (status_str == "unloaded") {
        return SERVER_MODEL_STATUS_UNLOADED;
    } else if (status_str == "loading") {
        return SERVER_MODEL_STATUS_LOADING;
    } else if (status_str == "loaded") {
        return SERVER_MODEL_STATUS_LOADED;
    } else if (status_str == "failed") {
        return SERVER_MODEL_STATUS_FAILED;
    } else {
        throw std::runtime_error("invalid server model status");
    }
}

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_UNLOADED: return "unloaded";
        case SERVER_MODEL_STATUS_LOADING:  return "loading";
        case SERVER_MODEL_STATUS_LOADED:   return "loaded";
        case SERVER_MODEL_STATUS_FAILED:   return "failed";
    }
}

struct server_model_meta {
    std::string name;
    std::string path;
    std::string path_mmproj; // only available if in_cache=false
    bool in_cache = false; // if true, use -hf; use -m otherwise
    int port = 0;
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED;
};

struct server_models {
private:
    struct instance_t {
        pid_t pid = 0;
        std::thread th;
        server_model_meta meta;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, instance_t> mapping;

    common_params base_params;
    std::vector<std::string> base_args;
    std::vector<std::string> base_env;

    void update_meta(const std::string & name, const server_model_meta & meta);

public:
    server_models(const common_params & params, int argc, char ** argv, char ** envp);

    // check if a model instance exists
    bool has_model(const std::string & name);

    // return a copy of model metadata
    std::optional<server_model_meta> get_meta(const std::string & name);

    // return a copy of all model metadata
    std::vector<server_model_meta> get_all_meta();

    void load(const std::string & name);
    void unload(const std::string & name);
    void unload_all();

    // update the status of a model instance
    void update_status(const std::string & name, server_model_status status);

    // wait until the model instance is fully loaded
    // return when the model is loaded or failed to load
    void wait_until_loaded(const std::string & name);

    // load the model if not loaded, otherwise do nothing
    void ensure_model_loaded(const std::string & name);

    // proxy an HTTP request to the model instance
    server_http_res_ptr proxy_request(const server_http_req & req, const std::string & method, const std::string & name);

    // notify the router server that a model instance is ready
    static void notify_router_server_ready(const std::string & name);
};

/**
 * A simple HTTP proxy that forwards requests to another server
 * and relays the responses back.
 */
struct server_http_proxy : server_http_res {
    std::function<void()> cleanup = nullptr;
public:
    server_http_proxy(const std::string & method,
                      const std::string & host,
                      int port,
                      const std::string & path,
                      const std::map<std::string, std::string> & headers,
                      const std::string & body,
                      const std::function<bool()> should_stop);
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
    };
};
