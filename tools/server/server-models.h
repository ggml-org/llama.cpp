#pragma once

#include "common.h"
#include "download.h"
#include "preset.h"
#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"

#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

/**
 * state diagram:
 *
 * DOWNLOADING ──► DOWNLOADED ──► (replaced by new instance)
 *
 * UNLOADED ──► LOADING ──► LOADED ◄──── SLEEPING
 *  ▲            │            │               ▲
 *  └───failed───┘            │               │
 *  ▲                         └──sleeping─────┘
 *  └────────unloaded─────────┘
 */
enum server_model_status {
    // TODO: also add downloading state when the logic is added
    SERVER_MODEL_STATUS_DOWNLOADING,
    SERVER_MODEL_STATUS_DOWNLOADED,
    SERVER_MODEL_STATUS_UNLOADED,
    SERVER_MODEL_STATUS_LOADING,
    SERVER_MODEL_STATUS_LOADED,
    SERVER_MODEL_STATUS_SLEEPING
};

enum server_model_source {
    SERVER_MODEL_SOURCE_PRESET,
    SERVER_MODEL_SOURCE_MODELS_DIR,
    SERVER_MODEL_SOURCE_CACHE,
};

enum server_child_mode {
    SERVER_CHILD_MODE_NORMAL,   // load the model and run normally
    SERVER_CHILD_MODE_DOWNLOAD, // download the model and exit
};

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_DOWNLOADING: return "downloading";
        case SERVER_MODEL_STATUS_DOWNLOADED:  return "downloaded";
        case SERVER_MODEL_STATUS_UNLOADED:    return "unloaded";
        case SERVER_MODEL_STATUS_LOADING:     return "loading";
        case SERVER_MODEL_STATUS_LOADED:      return "loaded";
        case SERVER_MODEL_STATUS_SLEEPING:    return "sleeping";
        default:                              return "unknown";
    }
}

static std::string server_model_source_to_string(server_model_source source) {
    switch (source) {
        case SERVER_MODEL_SOURCE_PRESET:     return "preset";
        case SERVER_MODEL_SOURCE_MODELS_DIR: return "models_dir";
        case SERVER_MODEL_SOURCE_CACHE:      return "cache";
        default:                             return "unknown";
    }
}

struct server_model_meta {
    server_model_source source = SERVER_MODEL_SOURCE_CACHE;
    common_preset preset;
    std::string name;
    std::set<std::string> aliases; // additional names that resolve to this model
    std::set<std::string> tags;    // informational tags, not used for routing
    int port = 0;
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED;
    int64_t last_used = 0; // for LRU unloading
    std::vector<std::string> args; // args passed to the model instance, will be populated by render_args()
    json loaded_info; // info to be reflected via /v1/models endpoint ; if in DOWNLOADING state, it should contain download progress info
    json progress; // reflect load or download progress info, if any
    int exit_code = 0; // exit code of the model instance process (only valid if status == FAILED)
    int stop_timeout = 0; // seconds to wait before force-killing the model instance during shutdown
    mtmd_caps multimodal; // multimodal capabilities
    // bool need_download = false; // whether the model needs to be downloaded before loading // TODO @ngxson: implement this

    bool is_ready() const {
        return status == SERVER_MODEL_STATUS_LOADED;
    }

    bool is_running() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING || status == SERVER_MODEL_STATUS_SLEEPING;
    }

    bool is_failed() const {
        return status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0;
    }

    void update_args(common_preset_context & ctx_presets, std::string bin_path);
    void update_caps();
};

struct server_models_routes;
struct server_subproc; // defined in server-models.cpp

struct server_models {
    friend struct server_models_routes;

private:
    struct instance_t {
        std::shared_ptr<server_subproc> subproc; // shared between main thread and monitoring thread
        std::thread th;
        server_model_meta meta;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, instance_t> mapping;

    // for stopping models
    std::condition_variable cv_stop;
    std::set<std::string> stopping_models;

    // set to true while load_models() is executing a reload; load() will wait until clear
    bool is_reloading = false;

    // if true, the next get_meta() will trigger a reload of model list
    bool need_reload = false;

    // conv_id -> model name that currently serves its stream session, lets the resumable stream
    // routes go straight to the owning child instead of polling every one. populated when
    // proxy_request forwards a POST carrying an X-Conversation-Id. best effort: a stale entry just
    // makes the child answer not found and the client recovers. owns its lock, one mutex per struct
    struct conv_model_tracker {
        void remember(const std::string & conv_id, const std::string & model) {
            if (conv_id.empty() || model.empty()) {
                return;
            }
            std::lock_guard<std::mutex> lock(mu);
            map[conv_id] = model;
        }

        std::optional<std::string> lookup(const std::string & conv_id) {
            if (conv_id.empty()) {
                return std::nullopt;
            }
            std::lock_guard<std::mutex> lock(mu);
            auto it = map.find(conv_id);
            if (it == map.end()) {
                return std::nullopt;
            }
            return it->second;
        }

        void forget(const std::string & conv_id) {
            if (conv_id.empty()) {
                return;
            }
            std::lock_guard<std::mutex> lock(mu);
            map.erase(conv_id);
        }

      private:
        std::mutex                                   mu;
        std::unordered_map<std::string, std::string> map;
    };

    common_preset_context ctx_preset;

    common_params base_params;
    std::string bin_path;
    std::vector<std::string> base_env;
    common_preset base_preset; // base preset from llama-server CLI args

    void update_meta(const std::string & name, const server_model_meta & meta);

    // unload least recently used models if the limit is reached
    void unload_lru();

    // not thread-safe, caller must hold mutex
    void add_model(server_model_meta && meta);

    // notify SSE clients
    void notify_sse(const std::string & event, const std::string & model_id, const json & data = nullptr);

public:
    // conv_id -> model tracker for the resumable stream routes, owns its lock
    conv_model_tracker conv_models;

    server_models(const common_params & params, int argc, char ** argv);

    server_response sse; // for real-time updates via SSE endpoint

    // (re-)load the list of models from various sources and prepare the metadata mapping
    // - if this is called the first time, simply populate the metadata
    // - if this is called subsequently (e.g. when refreshing from disk):
    //   - if a model is running but updated or removed from the source, it will be unloaded
    //   - if a model is not running, it will be added or updated according to the source
    void load_models();

    // check if a model instance exists (thread-safe)
    bool has_model(const std::string & name);

    // return a copy of model metadata (thread-safe)
    std::optional<server_model_meta> get_meta(const std::string & name);

    // return a copy of all model metadata (thread-safe)
    std::vector<server_model_meta> get_all_meta();

    struct load_options {
        server_child_mode mode = SERVER_CHILD_MODE_NORMAL;
        // used for spawning a downloading child process
        std::optional<server_model_meta> custom_meta = std::nullopt;
    };

    // load and unload model instances
    // these functions are thread-safe
    void load(const std::string & name);
    void load(const std::string & name, const load_options & opts);
    void unload(const std::string & name);
    void unload_all();

    struct update_status_args {
        server_model_status status;
        int exit_code = 0; // only valid if status == UNLOADED
        json loaded_info = nullptr;
        json progress = nullptr;
    };
    // update the status of a model instance (thread-safe)
    // also send SSE notification to /models/sse endpoint
    void update_status(const std::string & name, const update_status_args & args);
    void update_download_progress(const std::string & name, const common_download_progress & progress, bool done, bool ok = true);

    // remove a cache model from disk and update the list (thread-safe)
    // note: only cache models can be removed; returns false if the model doesn't exist or is not a cache model
    bool remove(const std::string & name);

    // wait until the model instance is fully loaded (thread-safe)
    // note: predicate is called while holding the lock
    // return when the model no longer in "loading" state
    void wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate);
    void wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate);

    // ensure the model is in ready state (thread-safe)
    // return false if model is ready
    // otherwise, load the model and blocking wait until it's ready, then return true (meta may need to be refreshed)
    bool ensure_model_ready(const std::string & name);

    // proxy an HTTP request to the model instance
    server_http_res_ptr proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used);

    // handle message sent from server_child::notify_to_router()
    // raw input must starts with CMD_CHILD_TO_ROUTER_STATE, followed by a JSON string
    // this function is not thread-safe, must be called from instance's monitoring thread
    // payload per state:
    //     state = loading     -> payload = {} (TODO: add progress info)
    //     state = ready       -> payload = model_info (json), or {} if wakeup from sleeping
    //     state = sleeping    -> payload = {}
    void handle_child_state(const std::string & name, const std::string & raw_input);
};

struct server_child {
    // serializes the notify_to_router writes
    std::mutex mtx_stdout;
    std::atomic<bool> is_finished_downloading = false; // set by run_download

    // return true if the current process is a child server instance
    bool is_child();
    server_child_mode get_mode();
    int run_download(common_params & params);

    // register the shutdown_handler to be called by the router
    // return the monitoring thread (to be joined by the caller)
    std::thread setup(const std::function<void(int)> & shutdown_handler);

    // notify router server for status changes (e.g. loading, downloading, sleeping, etc.)
    // message will be handled by server_models::handle_child_state() on the router side
    void notify_to_router(const std::string & state_name, const json & payload);
};

struct server_models_routes {
    common_params params;
    json ui_settings = json::object();     // Primary: new name
    std::atomic<bool> stopping = false;    // for graceful disconnecting SSE clients during shutdown
    server_models models;
    server_models_routes(const common_params & params, int argc, char ** argv)
            : params(params), models(params, argc, argv) {
        const std::string & cfg = this->params.ui_config_json;
        if (!cfg.empty()) {
            try {
                json json_settings = json::parse(cfg);
                ui_settings = json_settings;
            } catch (const std::exception & e) {
                LOG_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
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
    server_http_context::handler_t get_router_models;
    server_http_context::handler_t post_router_models_load;
    server_http_context::handler_t post_router_models_unload;
    // management API
    server_http_context::handler_t get_router_models_sse;
    server_http_context::handler_t post_router_models;
    server_http_context::handler_t del_router_models;

    // router side handlers for the resumable streaming routes. each resolves the child that owns
    // a conversation through the conv_id -> model map, no probing or fan out
    server_http_context::handler_t router_stream_get;
    server_http_context::handler_t router_streams_lookup;
    server_http_context::handler_t router_stream_delete;
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
                      int32_t timeout_write
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
