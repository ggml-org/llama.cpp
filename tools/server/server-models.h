#pragma once

#include "common.h"
#include "download.h"
#include "preset.h"
#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

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
    SERVER_MODEL_STATUS_INCOMPLETE, // a download that never finished, listed without an instance
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

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_DOWNLOADING: return "downloading";
        case SERVER_MODEL_STATUS_INCOMPLETE:  return "incomplete";
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

// memory of a model instance per device, measured by a child process (see server_run_measure)
// or reported by the running instance; sizes are in bytes
struct server_model_memory {
    struct device_t {
        std::string kind;    // "device", "host" or "buffer_type", as in the --log-jsonl memory breakdown
        std::string name;    // ggml device name, "Host" for host memory
        size_t total   = 0;  // device memory, 0 unless kind == "device"
        size_t free    = 0;  // free device memory when this was measured, 0 unless kind == "device"
        size_t model   = 0;  // held by the instance, by kind of buffer
        size_t context = 0;
        size_t compute = 0;

        bool   is_device() const { return kind == "device"; }
        size_t self()      const { return model + context + compute; }
    };

    uint32_t n_ctx = 0; // context size this was measured at
    std::vector<device_t> devices;

    server_model_memory() = default;
    server_model_memory(uint32_t n_ctx, const std::vector<struct common_memory_breakdown_row> & rows);

    json to_json() const;
    static server_model_memory from_json(const json & j); // throws on a malformed input
};

// where a model stands in the memory measurement done on startup, only used with --models-memory-margin
enum server_model_memory_status {
    SERVER_MODEL_MEMORY_UNMEASURED, // queued for a measure child
    SERVER_MODEL_MEMORY_MEASURING,  // its measure child is running
    SERVER_MODEL_MEMORY_MEASURED,
    SERVER_MODEL_MEMORY_FAILED,     // the measure child failed or crashed, memory limits do not apply to this model
};

static std::string server_model_memory_status_to_string(server_model_memory_status status) {
    switch (status) {
        case SERVER_MODEL_MEMORY_UNMEASURED: return "unmeasured";
        case SERVER_MODEL_MEMORY_MEASURING:  return "measuring";
        case SERVER_MODEL_MEMORY_MEASURED:   return "measured";
        case SERVER_MODEL_MEMORY_FAILED:     return "failed";
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
    bool hidden = false; // hidden from GET /models, but still accept if requested
    json architecture = server_model_architecture_json(false, false, false, {"text"});
    uint32_t n_ctx_train = 0; // trained context, read from the GGUF metadata; 0 when unknown

    // memory accounting, only populated with --models-memory-margin, see "Memory-based unloading" in the README
    server_model_memory_status memory_status = SERVER_MODEL_MEMORY_UNMEASURED;
    int memory_measure_attempts = 0; // measure children spawned for this model, bounds the retries
    std::optional<server_model_memory> memory_min       = std::nullopt; // at the smallest context --fit may pick
    std::optional<server_model_memory> memory_max       = std::nullopt; // at the largest context that fit the free memory when measured
    std::optional<server_model_memory> memory_effective = std::nullopt; // what the running instance allocated, reset when it exits

    json memory_to_json() const;

    bool is_ready() const {
        return status == SERVER_MODEL_STATUS_LOADED;
    }

    bool is_running() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING || status == SERVER_MODEL_STATUS_SLEEPING;
    }

    bool is_ready_or_sleep() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_SLEEPING;
    }

    bool is_failed() const {
        return status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0;
    }

    void update_args(common_preset_context & ctx_presets, std::string bin_path);
    void update_caps(const common_params & base);
};

// a download that never finished: the tag it was asked under, the sidecar kinds it
// pulls and the partial files its plan writes
struct server_unfinished_download {
    std::string                          model;
    std::vector<std::string>             sidecars;
    std::vector<common_download_partial> partials;
};

// the downloads that never finished, recorded next to the model cache. The cache names
// partial files after their content hash, so this record is the only trace of what is
// still incomplete: it lets a restarted router list, resume or remove them
// a partial another recorded download also writes is never deleted
// these functions are thread-safe
struct server_unfinished_downloads {
    // read the record an earlier run left
    void load();

    // record a download that starts, replacing an earlier record of the same model,
    // and delete the partials of that record the new plan no longer writes
    void add(const server_unfinished_download & download);

    // forget a download whose files are complete
    void forget(const std::string & model);

    // forget a download and delete its partials; returns false when it is not recorded
    bool remove(const std::string & model);

    // drop the sidecar `kind` from a download and delete its partials; returns false
    // when the download is not recorded with that kind
    bool remove_sidecar(const std::string & model, const std::string & kind);

    // every recorded download
    std::vector<server_unfinished_download> get_all();

private:
    std::mutex mutex;
    std::vector<server_unfinished_download> downloads;

    // write the record back; callers hold the mutex
    void save();

    // delete the partials of `parts` no recorded download writes; callers hold the mutex
    void delete_partials(const std::vector<common_download_partial> & parts);
};

struct server_models_routes;
struct server_lru_sched; // defined in server-models.cpp
struct server_monitor;   // defined in server-models.cpp

struct server_models {
    friend struct server_models_routes;
    friend struct server_lru_sched;
    friend struct server_monitor;

private:
    struct instance_t {
        std::shared_ptr<server_subproc> subproc; // shared with the monitor thread
        server_model_meta meta;
        int req_count = 0; // number of active proxy requests

        // bytes per device currently counted against device_memory for this instance, see reserve_memory()
        std::map<std::string, size_t> memory_held = {};

        // ask the child to exit (it handles the command on its stdin, see server_role::setup)
        void request_exit() const;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, instance_t> mapping;

    // models asked to stop, still counted as running until the monitor records their exit
    std::set<std::string> stopping_models;

    // downloads that never finished, with the sidecar kinds they pull
    server_unfinished_downloads unfinished_downloads;

    // set to true while load_models() is executing a reload; load() will wait until clear
    bool is_reloading = false;

    // if true, the next get_meta() will trigger a reload of model list
    bool need_reload = false;

    // models marked with load-on-startup, unset once load_startup_models() drains it
    // no value means the startup phase is over, so a reload must not queue anything
    std::optional<std::vector<std::string>> startup_models{std::in_place};

    // conv_id -> model name that currently serves its stream session, lets the resumable stream
    // routes go straight to the owning child instead of polling every one. populated when
    // proxy_request forwards a POST carrying an X-Conversation-Id. best effort: a stale entry just
    // makes the child answer not found and the client recovers. owns its lock, one mutex per struct
    struct conv_model_tracker {
        // returns the ticket of this registration, 0 when nothing was registered. erasing or
        // replacing the entry invalidates the ticket, which is how a stop cancels a request
        // parked in the model load wait
        uint64_t remember(const std::string & conv_id, const std::string & model) {
            if (conv_id.empty() || model.empty()) {
                return 0;
            }
            std::lock_guard<std::mutex> lock(mu);
            uint64_t ticket = next_ticket++;
            map[conv_id] = { model, ticket };
            return ticket;
        }

        // false means a stop erased the entry or a newer request replaced it
        bool alive(const std::string & conv_id, uint64_t ticket) {
            std::lock_guard<std::mutex> lock(mu);
            auto it = map.find(conv_id);
            return it != map.end() && it->second.ticket == ticket;
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
            return it->second.model;
        }

        void forget(const std::string & conv_id) {
            if (conv_id.empty()) {
                return;
            }
            std::lock_guard<std::mutex> lock(mu);
            map.erase(conv_id);
        }

      private:
        struct entry_t {
            std::string model;
            uint64_t    ticket;
        };
        std::mutex                               mu;
        uint64_t                                 next_ticket = 1;
        std::unordered_map<std::string, entry_t> map;
    };

    common_preset_context ctx_preset;

    common_params base_params;
    std::string bin_path;
    std::vector<std::string> base_env;
    common_preset base_preset; // base preset from llama-server CLI args

    // queue of requests waiting for a models_max slot
    std::unique_ptr<server_lru_sched> sched;

    // if true, add some delay to simulate works (useful for testing)
    bool debug_fake_timing = false;

    //
    // memory accounting (--models-memory-margin), the router itself never touches a GPU:
    // the instances are tracked here by what they hold (memory_held), a report from a child only calibrates what
    // else is on the device. this works whether a backend reports free memory system-wide (CUDA) or only net of
    // the reporting process (Metal)
    //

    // models waiting for a measure child, one runs at a time (see measure_next)
    std::deque<std::string> measure_queue;
    std::string measuring; // model whose measure child is running, empty when idle
    std::shared_ptr<server_subproc> measure_proc;

    struct device_memory_t {
        int64_t total    = 0; // bytes
        int64_t external = 0; // bytes used by anything the router does not track: other processes, backend contexts
    };
    std::map<std::string, device_memory_t> device_memory; // by device name, empty until the first child reports

    bool memory_limits_enabled() const { return base_params.models_memory_margin > 0; }

    // all not thread-safe, caller must hold mutex
    void enqueue_measure(const std::string & name, bool front = false);
    void measure_next();
    // a child reported what it sees: free memory and what it holds. reporter is the instance that sent it, empty for a measure child
    void calibrate_device_memory(const server_model_memory & mem, const std::string & reporter);
    void reserve_memory(const std::string & name); // count the instance's memory (effective, else min) until it sleeps or exits
    void release_memory(const std::string & name);
    int64_t device_memory_available(const std::string & dev) const; // total - external - what the instances hold
    // bytes missing per device for name to start at its minimum context, empty when it fits (or is not measured)
    std::map<std::string, size_t> memory_deficit(const std::string & name) const;
    // true when loading name now would exceed models_max or the memory margin; verbose logs the shortfall
    bool limits_exceeded(std::unique_lock<std::mutex> & lk, const std::string & name, bool verbose = true);

    // called from the monitor thread
    void on_measure_exit(const std::string & name, const std::shared_ptr<server_subproc> & proc, int exit_code);
    void handle_measured(const std::string & name, const json & payload);

    void update_meta(const std::string & name, const server_model_meta & meta);

    // unload least recently used models until name fits within models_max and the memory margin
    void unload_lru(const std::string & name);

    // not thread-safe, caller must hold mutex
    void add_model(server_model_meta && meta);

    // ask the monitor to stop a running instance; send_exit is false for a child that was already force-killed
    // not thread-safe, caller must hold mutex
    void request_stop(const std::string & name, bool send_exit = true);

    // called by the monitor once a child exited and was reaped
    void on_child_exit(const std::string & name, const std::shared_ptr<server_subproc> & proc, server_role_type role, int exit_code);

    // notify SSE clients
    void notify_sse(const std::string & event, const std::string & model_id, const json & data = nullptr);

public:
    // conv_id -> model tracker for the resumable stream routes, owns its lock
    conv_model_tracker conv_models;

    server_models(const common_params & params, int argc, char ** argv);
    ~server_models();

    server_response sse; // for real-time updates via SSE endpoint

    // (re-)load the list of models from various sources and prepare the metadata mapping
    // - if this is called the first time, simply populate the metadata
    // - if this is called subsequently (e.g. when refreshing from disk):
    //   - if a model is running but updated or removed from the source, it will be unloaded
    //   - if a model is not running, it will be added or updated according to the source
    void load_models();

    // lazy-load startup_models, to be called after main() setup phase
    void load_startup_models();

    // check if a model instance exists (thread-safe)
    bool has_model(const std::string & name);

    // downloads that never finished, with their sidecar kinds (thread-safe)
    std::vector<server_unfinished_download> unfinished_downloads_get();

    // drop a download that will never be resumed, its record and its partial files;
    // throws for a name that is not recorded (thread-safe)
    void unfinished_downloads_remove(const std::string & name);

    // remove the sidecar `kind` of a model that is not in use, its files and partials (thread-safe)
    bool remove_sidecar(const std::string & name, const std::string & kind);

    // return a copy of model metadata (thread-safe)
    std::optional<server_model_meta> get_meta(const std::string & name);

    // return a copy of all model metadata (thread-safe)
    std::vector<server_model_meta> get_all_meta();

    struct load_options {
        server_role_type role = SERVER_ROLE_INSTANCE;
        // used for spawning a downloading child process
        std::optional<server_model_meta> custom_meta = std::nullopt;
        // download mode: the sidecar kinds pulled with the model, as -hf would, and
        // the partial files the download plan writes
        std::vector<std::string>             sidecars;
        std::vector<common_download_partial> partials;
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
        std::optional<server_model_memory> memory = std::nullopt; // what the instance allocated, sent with READY
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
    // if models_max is reached, the request waits in a queue until a slot frees up
    // throws if the load fails, or if should_stop fires while waiting
    bool ensure_model_ready(const std::string & name, const std::function<bool()> & should_stop = nullptr);

    // proxy an HTTP request to the model instance
    server_http_res_ptr proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used, bool detached = false);
    // forward a websocket connection to the child, returns when either side closes
    void proxy_ws(const server_http_req & req, server_http_context::ws_conn & conn, const std::string & name);

    // handle message sent from server_role::notify_to_router()
    // raw input must starts with CMD_CHILD_TO_ROUTER_STATE, followed by a JSON string
    // called from the monitor thread
    // payload per state:
    //     state = loading     -> payload = {} (TODO: add progress info)
    //     state = ready       -> payload = model_info (json) with a "memory" field, or {} if wakeup from sleeping
    //     state = sleeping    -> payload = {}
    //     state = measuring   -> payload = {result: "measured", min, max} or {result: "failed", error}
    void handle_child_state(const std::string & name, const std::string & raw_input);

private:
    // one thread watching every child; keep last, the destructor joins the thread
    std::unique_ptr<server_monitor> monitor;
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
    server_http_context::ws_handler_t proxy_ws;

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
        if (cleanup_pipes) {
            cleanup_pipes();
        }
        if (cleanup) {
            cleanup();
        }
    }
private:
    std::function<void()> cleanup_pipes = nullptr;
    std::thread thread;
    struct msg_t {
        std::map<std::string, std::string> headers;
        int status = 0;
        std::string data;
        std::string content_type;
    };
};
